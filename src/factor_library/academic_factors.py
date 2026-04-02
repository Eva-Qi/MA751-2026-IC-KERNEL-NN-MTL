"""
academic_factors.py — 6 个学术校准因子实现

Stage 0 使用这 6 个已知有效的因子来验证整个 pipeline 基础设施是否正确。

因子列表
--------
1. EarningsYield       — E/P，价值因子            (Basu 1977)
2. GrossProfitability  — GP/A，质量因子           (Novy-Marx 2013)
3. AssetGrowth         — 总资产同比增长，投资因子  (Cooper et al. 2008)
4. Accruals            — Sloan 应计比率，质量因子  (Sloan 1996)
5. Momentum12_1        — 12-1 月价格动量           (Jegadeesh & Titman 1993)
6. NetDebtEBITDA       — 净负债/EBITDA，杠杆因子   (Penman et al. 2007)

函数签名统一规范
--------------
  compute_xxx(xbrl_df, prices_df, as_of_date) → pd.Series[ticker → raw_value]

  xbrl_df 列: [cik, ticker, concept, xbrl_tag, period_end, filed_date,
               available_date, value, unit, form_type, fiscal_year, fiscal_period]

  prices_df: index=date (DatetimeIndex), columns=ticker, values=adj_close

  as_of_date: str, 格式 "YYYY-MM-DD"

反 look-ahead bias 规则（不可破坏）
-------------------------------------
  所有 XBRL 查询必须用: xbrl_df["available_date"] <= as_of_date
  永远不能用:           xbrl_df["period_end"]     <= as_of_date

  available_date = filed_date + 1 个交易日（在 xbrl_loader 中计算）
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from alpha_system.core.data_pipeline.taxonomy_map import CONCEPT_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内部常量
# ---------------------------------------------------------------------------

# 从 CONCEPT_MAP 提取所有有效 XBRL tag 集合，用于快速查找
_ALL_VALID_TAGS: frozenset[str] = frozenset(
    tag for tags in CONCEPT_MAP.values() for tag in tags
)

# concept_key → fallback tag 列表的有序映射
_CONCEPT_TAGS: dict[str, list[str]] = CONCEPT_MAP

# shares 类概念需要用 unit="shares" 过滤（而非 "USD"）
_SHARES_CONCEPTS: frozenset[str] = frozenset({
    "SHARES_OUTSTANDING",
})


# ---------------------------------------------------------------------------
# 性能优化：预分组缓存
# ---------------------------------------------------------------------------

# 将 xbrl_df 按 ticker 预分组，避免每次 get_latest_xbrl_value 全表扫描
# 缓存按 DataFrame id 存储，xbrl_df 变化时自动重建
_GROUPED_CACHE: dict[int, dict[str, pd.DataFrame]] = {}


def _get_ticker_sub(xbrl_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    获取某 ticker 的 xbrl_df 子集。首次调用时建立 groupby 缓存。

    将每次扫描从 ~346K 行降至 ~690 行（~500x 加速）。
    """
    df_id = id(xbrl_df)
    if df_id not in _GROUPED_CACHE:
        _GROUPED_CACHE.clear()  # 只缓存一个 DataFrame
        _GROUPED_CACHE[df_id] = {
            t: sub for t, sub in xbrl_df.groupby("ticker")
        }
    return _GROUPED_CACHE[df_id].get(ticker, xbrl_df.iloc[:0])


# ---------------------------------------------------------------------------
# Private helper: _tag_priority_col
# ---------------------------------------------------------------------------

def _tag_priority_col(concept_series: pd.Series, tags: list[str]) -> pd.Series:
    """
    对 concept 列按 tags 列表中的位置生成优先级整数 Series。

    低索引 = 高优先级；不在 tags 中的 concept 取 len(tags)（最低优先级）。
    用于 get_latest_xbrl_value 和 compute_ttm 中排序时避免重复 lambda。
    """
    n = len(tags)
    tag_rank = {t: i for i, t in enumerate(tags)}
    return concept_series.map(lambda c: tag_rank.get(c, n))


# ---------------------------------------------------------------------------
# Helper: get_latest_xbrl_value
# ---------------------------------------------------------------------------

def get_latest_xbrl_value(
    xbrl_df: pd.DataFrame,
    ticker: str,
    concept_key: str,
    as_of_date: str,
    unit_filter: str | None = "USD",
) -> float | None:
    """
    获取某个存量概念（balance sheet item）在 as_of_date 时点最新的公布值。

    使用 available_date（不是 period_end）过滤，严格防止 look-ahead bias。
    对同一 period_end 有多条记录（如 10-K/A 修正）时取最新 available_date。

    参数
    ----
    xbrl_df : pd.DataFrame
        xbrl_facts SQLite 表加载的 DataFrame。
    ticker : str
        股票代码。
    concept_key : str
        CONCEPT_MAP 中的标准概念键，如 "TOTAL_ASSETS"。
    as_of_date : str
        截止日期，格式 "YYYY-MM-DD"。
    unit_filter : str | None
        限制 unit 列的值，默认 "USD"。对 shares 类概念传 "shares"。
        传 None 不做 unit 过滤。

    返回
    ----
    float | None
        最新公布值；若无可用数据则返回 None。

    示例
    ----
    >>> assets = get_latest_xbrl_value(xbrl_df, "AAPL", "TOTAL_ASSETS", "2023-01-31")
    """
    tags = _CONCEPT_TAGS.get(concept_key, [])
    if not tags:
        return None

    # 使用预分组缓存：~690 行 vs ~346K 行全表扫描
    ticker_df = _get_ticker_sub(xbrl_df, ticker)
    if ticker_df.empty:
        return None

    # 按 available_date 过滤（反 look-ahead bias 核心）
    mask = (
        (ticker_df["available_date"] <= as_of_date)
        & (ticker_df["concept"].isin(tags))
    )
    # unit 过滤：防止 shares 类记录混入 USD 查询（反之亦然）
    if unit_filter is not None and "unit" in ticker_df.columns:
        mask = mask & (ticker_df["unit"].str.lower() == unit_filter.lower())
    sub = ticker_df.loc[mask]

    if sub.empty:
        return None

    # 按 concept 的 fallback 优先级排序，取 period_end 最新的那条
    # 若同一 period_end 有多条（amended filing），取 available_date 最大的
    sub = sub.copy()
    sub["_tag_priority"] = _tag_priority_col(sub["concept"], tags)
    sub = sub.sort_values(
        ["period_end", "available_date", "_tag_priority"],
        ascending=[False, False, True],
    )

    best = sub.iloc[0]
    val = best["value"]
    if pd.isna(val):
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Helper: compute_ttm
# ---------------------------------------------------------------------------

def compute_ttm(
    xbrl_df: pd.DataFrame,
    ticker: str,
    concept_key: str,
    as_of_date: str,
) -> float | None:
    """
    计算 Trailing Twelve Months（TTM，过去 12 个月累计值）。

    优先策略（按顺序）：
    1. 若存在 fiscal_period="FY" 且 available_date <= as_of_date，
       且该年报不早于 18 个月（避免使用过期年报），直接使用年度数据。
    2. Fallback：取最近 4 个不重叠季度（fiscal_period in Q1..Q4）之和。
    3. 若季度数 < 4，返回 None（数据不足）。

    应计逻辑（去重复）
    ------------------
    由于 10-K 和 10-Q 均可报告全年/季度数据，须避免重复计算：
    - 只使用各 (fiscal_year, fiscal_period) 组合中 available_date 最新的记录
    - 取最近 4 个唯一季度（按 period_end 降序）

    参数
    ----
    xbrl_df : pd.DataFrame
        xbrl_facts 表。
    ticker : str
        股票代码。
    concept_key : str
        CONCEPT_MAP 中的概念键，如 "NET_INCOME"。
    as_of_date : str
        截止日期。

    返回
    ----
    float | None
        TTM 值；数据不足时返回 None。
    """
    tags = _CONCEPT_TAGS.get(concept_key, [])
    if not tags:
        return None

    as_of_ts = pd.Timestamp(as_of_date)
    stale_cutoff = (as_of_ts - pd.DateOffset(months=18)).strftime("%Y-%m-%d")

    # 使用预分组缓存
    ticker_df = _get_ticker_sub(xbrl_df, ticker)
    if ticker_df.empty:
        return None

    mask = (
        (ticker_df["available_date"] <= as_of_date)
        & (ticker_df["concept"].isin(tags))
    )
    # Unit filter: financial flow/stock concepts should use USD records only.
    # This prevents mixing USD + EUR (e.g. CRH) or other currency records.
    if "unit" in ticker_df.columns:
        mask = mask & (ticker_df["unit"].str.lower() == "usd")
    sub = ticker_df.loc[mask].copy()

    if sub.empty:
        return None

    # tag 优先级（低索引 = 高优先级）
    sub["_tag_priority"] = _tag_priority_col(sub["concept"], tags)

    # --- 策略 1: 年度数据 ---
    annual = sub[
        (sub["fiscal_period"] == "FY")
        & (sub["period_end"] >= stale_cutoff)
    ].sort_values(
        ["period_end", "available_date", "_tag_priority"],
        ascending=[False, False, True],
    )

    if not annual.empty:
        val = annual.iloc[0]["value"]
        if not pd.isna(val):
            return float(val)

    # --- 策略 2: 最近 4 个季度之和 ---
    quarterly = sub[sub["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if quarterly.empty:
        return None

    # 对每个 (fiscal_year, fiscal_period) 去重：取 available_date 最新的记录
    quarterly = quarterly.sort_values(
        ["fiscal_year", "fiscal_period", "available_date", "_tag_priority"],
        ascending=[True, True, False, True],
    )
    quarterly = quarterly.drop_duplicates(
        subset=["fiscal_year", "fiscal_period"], keep="first"
    )

    # 按 period_end 降序取最近 4 个季度
    quarterly = quarterly.sort_values("period_end", ascending=False)
    recent_4 = quarterly.head(4)

    if len(recent_4) < 4:
        return None

    values = recent_4["value"].dropna()
    if len(values) < 4:
        return None

    return float(values.sum())


# ---------------------------------------------------------------------------
# Helper: _get_shares_outstanding
# ---------------------------------------------------------------------------

def _get_shares_outstanding(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    获取所有 ticker 在 as_of_date 时点的流通股数（股）。

    从 xbrl_df 提取 SHARES_OUTSTANDING 概念的最新公布值，
    并以 prices_df 的列（tickers）为基准对齐。

    参数
    ----
    xbrl_df : pd.DataFrame
        xbrl_facts 表。
    prices_df : pd.DataFrame
        价格 DataFrame，columns=ticker。
    as_of_date : str
        截止日期。

    返回
    ----
    pd.Series
        index=ticker, values=shares_outstanding (float)。缺失者为 NaN。
    """
    tickers = prices_df.columns.tolist()
    shares_tags = _CONCEPT_TAGS.get("SHARES_OUTSTANDING", [])

    if not shares_tags:
        return pd.Series(np.nan, index=tickers)

    # 预过滤：只取 shares 类概念 + available_date <= as_of_date
    mask = (
        (xbrl_df["ticker"].isin(tickers))
        & (xbrl_df["available_date"] <= as_of_date)
        & (xbrl_df["concept"].isin(shares_tags))
    )
    # unit 过滤：shares 类概念只取 unit="shares"，避免 USD 记录混入
    if "unit" in xbrl_df.columns:
        mask = mask & (xbrl_df["unit"].str.lower() == "shares")
    sub = xbrl_df.loc[mask].copy()

    if sub.empty:
        return pd.Series(np.nan, index=tickers)

    sub["_tag_priority"] = sub["concept"].apply(
        lambda c: shares_tags.index(c) if c in shares_tags else len(shares_tags)
    )
    sub = sub.sort_values(
        ["ticker", "period_end", "available_date", "_tag_priority"],
        ascending=[True, False, False, True],
    )
    # 每个 ticker 取最新一条
    latest = sub.drop_duplicates(subset=["ticker"], keep="first")
    latest = latest.set_index("ticker")["value"]
    latest = latest.reindex(tickers)
    latest = pd.to_numeric(latest, errors="coerce")

    # Shares plausibility guard: S&P 500 stocks have ~50M–20B shares outstanding.
    # Values outside [1M, 100B] are almost certainly XBRL scaling errors
    # (e.g. reported in millions instead of raw, or data entry errors like 7.96e14).
    _MIN_PLAUSIBLE_SHARES = 1e6
    _MAX_PLAUSIBLE_SHARES = 1e11
    implausible = (latest < _MIN_PLAUSIBLE_SHARES) | (latest > _MAX_PLAUSIBLE_SHARES)
    n_implausible = implausible.sum()
    if n_implausible > 0:
        bad_tickers = latest.index[implausible].tolist()
        logger.warning(
            "_get_shares_outstanding @ %s: %d tickers with implausible shares "
            "(outside [%.0e, %.0e]), setting to NaN: %s",
            as_of_date, n_implausible,
            _MIN_PLAUSIBLE_SHARES, _MAX_PLAUSIBLE_SHARES,
            bad_tickers[:10],
        )
    latest = latest.where(~implausible, other=np.nan)

    return latest


# ---------------------------------------------------------------------------
# Helper: _get_price_at_date
# ---------------------------------------------------------------------------

def _get_price_at_date(prices_df: pd.DataFrame, as_of_date: str) -> pd.Series:
    """
    获取最近交易日（<= as_of_date）的收盘价。

    参数
    ----
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    as_of_date : str
        截止日期。

    返回
    ----
    pd.Series
        index=ticker, values=price。
    """
    date_ts = pd.Timestamp(as_of_date)
    valid_idx = prices_df.index[prices_df.index <= date_ts]
    if valid_idx.empty:
        return pd.Series(dtype=float)
    return prices_df.loc[valid_idx[-1]]


# ---------------------------------------------------------------------------
# Helper: cross_sectional_zscore
# ---------------------------------------------------------------------------

def cross_sectional_zscore(
    raw_values: pd.Series,
    winsorize_sigma: float = 3.0,
    min_samples: int = 5,
) -> pd.Series:
    """
    截面 z-score 标准化（带 Winsorization）。

    步骤：
    1. 剔除 NaN（计算 mean/std 时不含 NaN）
    2. 将极端值裁剪到 ±winsorize_sigma × std（Winsorize）
    3. 重新计算 mean/std 后进行 z-score 归一化

    边界情况处理：
    - 全部 NaN → 返回全 NaN Series（保留原始 index）
    - std = 0（所有值相同）→ 返回全 0 Series
    - 有效样本 < min_samples → 返回全 NaN Series

    参数
    ----
    raw_values : pd.Series
        原始截面因子值，index=ticker。
    winsorize_sigma : float
        Winsorization 阈值（sigma 倍数），默认 3.0。
    min_samples : int
        最小有效样本数，默认 5。低于此值返回全 NaN。

    返回
    ----
    pd.Series
        index 与 raw_values 相同，values 为 z-score（NaN 保留为 NaN）。

    示例
    ----
    >>> raw = pd.Series({"AAPL": 0.05, "MSFT": 0.03, "GOOG": 0.04})
    >>> z = cross_sectional_zscore(raw)
    >>> abs(z.mean()) < 1e-10  # mean ≈ 0
    True
    """
    if raw_values.empty:
        return raw_values.copy()

    valid = raw_values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=raw_values.index)

    if len(valid) < min_samples:
        logger.warning(
            "cross_sectional_zscore: 有效样本仅 %d 个 (min=%d)，返回 NaN",
            len(valid), min_samples,
        )
        return pd.Series(np.nan, index=raw_values.index)

    mean_raw = valid.mean()
    std_raw = valid.std()

    if pd.isna(std_raw) or std_raw == 0.0:
        return pd.Series(0.0, index=raw_values.index).where(raw_values.notna(), np.nan)

    # Winsorize
    lo = mean_raw - winsorize_sigma * std_raw
    hi = mean_raw + winsorize_sigma * std_raw
    clipped = raw_values.clip(lower=lo, upper=hi)

    # 重新计算 mean/std（仅对 non-NaN 位置）
    valid_clipped = clipped.dropna()
    mean_clip = valid_clipped.mean()
    std_clip = valid_clipped.std()

    if pd.isna(std_clip) or std_clip == 0.0:
        return pd.Series(0.0, index=raw_values.index).where(raw_values.notna(), np.nan)

    return (clipped - mean_clip) / std_clip


# ---------------------------------------------------------------------------
# Factor 1: Earnings Yield (E/P)
# ---------------------------------------------------------------------------

def compute_earnings_yield(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 Earnings Yield（E/P 比）。

    公式
    ----
        EY = net_income_ttm / market_cap
        market_cap = price * shares_outstanding

    实现细节
    --------
    - net_income_ttm: 使用 compute_ttm() 获取，优先年度数据
    - market_cap: as_of_date 当日价格 × 最新公布股数
    - 市值为 0 或负数时返回 NaN（避免无意义比率）
    - 所有 XBRL 查询严格过滤 available_date <= as_of_date

    参数
    ----
    xbrl_df : pd.DataFrame
        xbrl_facts 表，必须包含 available_date 列。
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    as_of_date : str
        计算日期，格式 "YYYY-MM-DD"。

    返回
    ----
    pd.Series
        index=ticker, values=E/P（原始未标准化）。数据缺失者为 NaN。
    """
    tickers = prices_df.columns.tolist()
    prices_now = _get_price_at_date(prices_df, as_of_date)
    shares = _get_shares_outstanding(xbrl_df, prices_df, as_of_date)

    results: dict[str, float] = {}
    for ticker in tickers:
        price = prices_now.get(ticker)
        share_count = shares.get(ticker)

        if pd.isna(price) or pd.isna(share_count) or share_count <= 0:
            results[ticker] = np.nan
            continue

        mkt_cap = float(price) * float(share_count)
        if mkt_cap <= 0:
            results[ticker] = np.nan
            continue

        # Market cap plausibility: S&P 500 members should have mkt_cap > $1B
        _MIN_PLAUSIBLE_MKTCAP = 1e9
        if mkt_cap < _MIN_PLAUSIBLE_MKTCAP:
            logger.warning(
                "compute_earnings_yield: %s @ %s — market_cap=%.2e < $1B "
                "(price=%.2f, shares=%.2e) — likely scaling error, setting EY to NaN",
                ticker, as_of_date, mkt_cap, float(price), float(share_count),
            )
            results[ticker] = np.nan
            continue

        net_income = compute_ttm(xbrl_df, ticker, "NET_INCOME", as_of_date)
        if net_income is None or pd.isna(net_income):
            results[ticker] = np.nan
            continue

        ey_value = float(net_income) / mkt_cap

        # EY magnitude guard: |EY| > 1.0 is implausible for any real stock
        _MAX_REASONABLE_EY = 1.0
        if abs(ey_value) > _MAX_REASONABLE_EY:
            logger.warning(
                "compute_earnings_yield: %s @ %s — |EY|=%.2f > 1.0 "
                "(net_income=%.2e, mkt_cap=%.2e) — setting to NaN",
                ticker, as_of_date, ey_value, net_income, mkt_cap,
            )
            results[ticker] = np.nan
            continue

        results[ticker] = ey_value

    ey_series = pd.Series(results, name="EarningsYield")
    valid = ey_series.dropna()
    if not valid.empty:
        logger.info(
            "compute_earnings_yield @ %s: n=%d, median=%.4f, mean=%.4f, "
            "std=%.4f, [min=%.4f, max=%.4f], NaN=%d",
            as_of_date, len(valid), valid.median(), valid.mean(),
            valid.std(), valid.min(), valid.max(),
            ey_series.isna().sum(),
        )
    return ey_series


# ---------------------------------------------------------------------------
# Factor 2: Gross Profitability (Novy-Marx 2013)
# ---------------------------------------------------------------------------

def compute_gross_profitability(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 Gross Profitability（GP/A）。

    公式
    ----
        GP/A = gross_profit_ttm / total_assets

    备用公式
    --------
    若 GROSS_PROFIT 不可用，尝试：
        gross_profit = REVENUE - COST_OF_REVENUE

    实现细节
    --------
    - total_assets: 存量科目，用 get_latest_xbrl_value()
    - 分母 total_assets <= 0 时返回 NaN

    参数
    ----
    同 compute_earnings_yield。

    返回
    ----
    pd.Series
        index=ticker, values=GP/A（原始）。
    """
    tickers = prices_df.columns.tolist()
    results: dict[str, float] = {}

    for ticker in tickers:
        # 分母：总资产（存量，用最新值）
        total_assets = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_ASSETS", as_of_date
        )
        if total_assets is None or total_assets <= 0:
            results[ticker] = np.nan
            continue

        # Assets plausibility guard
        _MIN_PLAUSIBLE_ASSETS = 1e8
        if total_assets < _MIN_PLAUSIBLE_ASSETS:
            logger.warning(
                "compute_gross_profitability: %s @ %s — implausible assets "
                "(%.2e), setting to NaN", ticker, as_of_date, total_assets,
            )
            results[ticker] = np.nan
            continue

        # 分子：毛利润 TTM
        gross_profit = compute_ttm(xbrl_df, ticker, "GROSS_PROFIT", as_of_date)

        # 备用：Revenue - COGS
        if gross_profit is None or pd.isna(gross_profit):
            rev = compute_ttm(xbrl_df, ticker, "REVENUE", as_of_date)
            cogs = compute_ttm(xbrl_df, ticker, "COST_OF_REVENUE", as_of_date)
            if rev is not None and cogs is not None:
                gross_profit = float(rev) - float(cogs)
            else:
                results[ticker] = np.nan
                continue

        results[ticker] = float(gross_profit) / float(total_assets)

    return pd.Series(results, name="GrossProfitability")


# ---------------------------------------------------------------------------
# Factor 3: Asset Growth (Cooper et al. 2008)
# ---------------------------------------------------------------------------

def compute_asset_growth(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 Asset Growth（总资产同比增长率）。

    公式
    ----
        AG = total_assets(t) / total_assets(t - 4Q) - 1

    实现细节
    --------
    - 取最近一期（存量）的 total_assets 作为分子
    - 取该记录之前约 4 个季度（365 天）的 total_assets 作为分母
    - 使用 available_date 过滤保证 point-in-time

    参数
    ----
    同 compute_earnings_yield。

    返回
    ----
    pd.Series
        index=ticker, values=资产增长率（原始）。
    """
    tickers = prices_df.columns.tolist()

    # 以 as_of_date 为当前；一年前截止
    as_of_ts = pd.Timestamp(as_of_date)
    prior_date = (as_of_ts - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

    tags = _CONCEPT_TAGS.get("TOTAL_ASSETS", [])
    results: dict[str, float] = {}

    for ticker in tickers:
        # 当期总资产（最新可用）
        assets_now = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_ASSETS", as_of_date
        )
        if assets_now is None or assets_now <= 0:
            results[ticker] = np.nan
            continue

        # 一年前总资产
        assets_prior = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_ASSETS", prior_date
        )
        if assets_prior is None or assets_prior <= 0:
            results[ticker] = np.nan
            continue

        # Assets plausibility: S&P 500 companies should have assets > $100M.
        # Values like $130 or $1,000 are XBRL data entry errors (AMCR, ARES).
        _MIN_PLAUSIBLE_ASSETS = 1e8
        if assets_now < _MIN_PLAUSIBLE_ASSETS or assets_prior < _MIN_PLAUSIBLE_ASSETS:
            logger.warning(
                "compute_asset_growth: %s @ %s — implausible assets "
                "(now=%.2e, prior=%.2e), setting AG to NaN",
                ticker, as_of_date, assets_now, assets_prior,
            )
            results[ticker] = np.nan
            continue

        ag_value = float(assets_now) / float(assets_prior) - 1.0

        # AG magnitude guard: |AG| > 10 (1000% growth) is implausible for S&P 500
        _MAX_REASONABLE_AG = 10.0
        if abs(ag_value) > _MAX_REASONABLE_AG:
            logger.warning(
                "compute_asset_growth: %s @ %s — |AG|=%.2f > 10.0 "
                "(assets_now=%.2e, assets_prior=%.2e) — setting to NaN",
                ticker, as_of_date, ag_value, assets_now, assets_prior,
            )
            results[ticker] = np.nan
            continue

        results[ticker] = ag_value

    ag_series = pd.Series(results, name="AssetGrowth")
    valid = ag_series.dropna()
    if not valid.empty:
        logger.info(
            "compute_asset_growth @ %s: n=%d, median=%.4f, mean=%.4f, "
            "std=%.4f, [min=%.4f, max=%.4f], NaN=%d",
            as_of_date, len(valid), valid.median(), valid.mean(),
            valid.std(), valid.min(), valid.max(),
            ag_series.isna().sum(),
        )
    return ag_series


# ---------------------------------------------------------------------------
# Factor 4: Accruals (Sloan 1996)
# ---------------------------------------------------------------------------

def compute_accruals(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 Sloan Accruals（应计比率）。

    公式
    ----
        Accruals = (net_income_ttm - CFO_ttm) / avg_total_assets
        avg_total_assets = (total_assets_now + total_assets_prior) / 2

    实现细节
    --------
    - net_income 和 CFO 均使用 TTM（流量科目）
    - total_assets 使用最新值和一年前值的均值
    - avg_total_assets = 0 时返回 NaN

    参数
    ----
    同 compute_earnings_yield。

    返回
    ----
    pd.Series
        index=ticker, values=Accruals 比率（原始）。
        高应计 = 低盈利质量（负向预测因子）。
    """
    tickers = prices_df.columns.tolist()
    as_of_ts = pd.Timestamp(as_of_date)
    prior_date = (as_of_ts - pd.DateOffset(months=12)).strftime("%Y-%m-%d")

    results: dict[str, float] = {}

    for ticker in tickers:
        # 净利润 TTM
        net_income = compute_ttm(xbrl_df, ticker, "NET_INCOME", as_of_date)
        if net_income is None:
            results[ticker] = np.nan
            continue

        # 经营现金流 TTM
        cfo = compute_ttm(xbrl_df, ticker, "CFO", as_of_date)
        if cfo is None:
            results[ticker] = np.nan
            continue

        # 平均总资产
        assets_now = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_ASSETS", as_of_date
        )
        assets_prior = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_ASSETS", prior_date
        )

        if assets_now is None or assets_prior is None:
            results[ticker] = np.nan
            continue

        # Assets plausibility guard (same as compute_asset_growth)
        _MIN_PLAUSIBLE_ASSETS = 1e8
        if assets_now < _MIN_PLAUSIBLE_ASSETS or assets_prior < _MIN_PLAUSIBLE_ASSETS:
            logger.warning(
                "compute_accruals: %s @ %s — implausible assets "
                "(now=%.2e, prior=%.2e), setting to NaN",
                ticker, as_of_date, assets_now, assets_prior,
            )
            results[ticker] = np.nan
            continue

        avg_assets = (float(assets_now) + float(assets_prior)) / 2.0
        if avg_assets == 0.0:
            results[ticker] = np.nan
            continue

        results[ticker] = (float(net_income) - float(cfo)) / avg_assets

    return pd.Series(results, name="Accruals")


# ---------------------------------------------------------------------------
# Factor 5: 12-1 Momentum (Jegadeesh & Titman 1993)
# ---------------------------------------------------------------------------

def compute_momentum_12_1(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 12-1 月价格动量。

    公式
    ----
        Mom = price(t - 21) / price(t - 252) - 1

    其中 t 为 as_of_date 最近交易日，交易日数以 prices_df 实际 index 计算。
    跳过最近 21 个交易日（约 1 个月）以规避短期反转效应。

    实现细节
    --------
    - 纯价格因子：完全不使用 xbrl_df（参数保留以统一接口签名）
    - 要求价格历史 >= 252 个交易日，否则返回 NaN
    - 上市不足 252 天的新股自动为 NaN

    参数
    ----
    xbrl_df : pd.DataFrame
        未使用，保留以统一接口。
    prices_df : pd.DataFrame
        index=date (DatetimeIndex), columns=ticker, values=adj_close。
    as_of_date : str
        计算日期。

    返回
    ----
    pd.Series
        index=ticker, values=12-1 动量（原始收益率）。
    """
    date_ts = pd.Timestamp(as_of_date)
    valid_dates = prices_df.index[prices_df.index <= date_ts]

    if len(valid_dates) < 252:
        # 没有足够历史
        return pd.Series(np.nan, index=prices_df.columns, name="Momentum12_1")

    # 找两个时间点的索引位置（0-indexed）
    # t-21:  跳过最近 1 个月（skip 21 trading days）
    # t-252: 12 个月前的起点
    n = len(valid_dates)
    idx_end = n - 22     # 从 0 开始，n-22 即倒数第 22 天（skip 最近 21 天）
    idx_start = n - 252  # 12 个月前

    if idx_end <= max(idx_start, 0):
        return pd.Series(np.nan, index=prices_df.columns, name="Momentum12_1")

    date_end   = valid_dates[idx_end]
    date_start = valid_dates[max(idx_start, 0)]

    p_end   = prices_df.loc[date_end]
    p_start = prices_df.loc[date_start]

    mom = (p_end / p_start) - 1.0
    mom = mom.where(p_start > 0, np.nan)   # 起点价格 <= 0 时为 NaN
    mom.name = "Momentum12_1"
    return mom


# ---------------------------------------------------------------------------
# Factor 6: Net Debt / EBITDA
# ---------------------------------------------------------------------------

def compute_net_debt_ebitda(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    计算截面 Net Debt / EBITDA（净债务杠杆比率）。

    公式
    ----
        Net Debt  = (TOTAL_DEBT + CURRENT_DEBT) - CASH
        EBITDA    = OPERATING_INCOME_ttm + DEPRECIATION_ttm
        ND/EBITDA = Net Debt / EBITDA

    实现细节
    --------
    - TOTAL_DEBT 和 CASH 为存量科目，用 get_latest_xbrl_value()
    - OPERATING_INCOME 和 DEPRECIATION 为流量科目，用 compute_ttm()
    - CURRENT_DEBT 存量，用 get_latest_xbrl_value()
    - EBITDA <= 0（亏损/无折旧）时返回 NaN（避免符号翻转导致误解）
    - 净债务 < 0（净现金公司）时保留负值（合法经济含义）

    参数
    ----
    同 compute_earnings_yield。

    返回
    ----
    pd.Series
        index=ticker, values=净债务/EBITDA（原始）。
        高值 = 高杠杆（负向预测因子）。
    """
    tickers = prices_df.columns.tolist()
    results: dict[str, float] = {}

    for ticker in tickers:
        # ---- 分子: Net Debt ----
        total_debt = get_latest_xbrl_value(
            xbrl_df, ticker, "TOTAL_DEBT", as_of_date
        )
        current_debt = get_latest_xbrl_value(
            xbrl_df, ticker, "CURRENT_DEBT", as_of_date
        )
        cash = get_latest_xbrl_value(
            xbrl_df, ticker, "CASH", as_of_date
        )

        if total_debt is None and current_debt is None:
            results[ticker] = np.nan
            continue
        if cash is None:
            results[ticker] = np.nan
            continue

        td = float(total_debt) if total_debt is not None else 0.0
        cd = float(current_debt) if current_debt is not None else 0.0
        net_debt = td + cd - float(cash)

        # ---- 分母: EBITDA ----
        op_income = compute_ttm(xbrl_df, ticker, "OPERATING_INCOME", as_of_date)
        depreciation = compute_ttm(xbrl_df, ticker, "DEPRECIATION", as_of_date)

        if op_income is None or depreciation is None:
            results[ticker] = np.nan
            continue

        ebitda = float(op_income) + float(depreciation)
        if ebitda <= 0 or abs(ebitda) < 1e6:
            # EBITDA <= 0 或绝对值 < $1M: 除法结果极端，返回 NaN
            results[ticker] = np.nan
            continue

        results[ticker] = net_debt / ebitda

    return pd.Series(results, name="NetDebtEBITDA")


# ---------------------------------------------------------------------------
# Private helper: _run_factor_batch
# ---------------------------------------------------------------------------

def _run_factor_batch(
    factor_functions: dict[str, callable],
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
    *,
    log_prefix: str,
) -> pd.DataFrame:
    """
    遍历 factor_functions，对每个因子调用计算函数，生成 raw / zscore / quintile 三列。

    参数
    ----
    factor_functions : dict[str, callable]
        {factor_name: fn(xbrl_df, prices_df, as_of_date) -> pd.Series} 的有序映射。
    xbrl_df : pd.DataFrame
        xbrl_facts 表。
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    as_of_date : str
        计算日期，格式 "YYYY-MM-DD"。
    log_prefix : str
        用于日志消息前缀（如 "compute_all_factors" 或 "compute_control_factors"）。

    返回
    ----
    pd.DataFrame
        index=ticker（index.name="ticker"），列为 {name} / {name}_zscore / {name}_quintile。
    """
    result_frames: dict[str, pd.Series] = {}

    for factor_name, fn in factor_functions.items():
        try:
            raw: pd.Series = fn(xbrl_df, prices_df, as_of_date)
        except Exception as exc:
            logger.error(
                "%s: %s 计算失败 (date=%s) — %s",
                log_prefix, factor_name, as_of_date, exc,
            )
            raw = pd.Series(np.nan, index=prices_df.columns, name=factor_name)

        raw.name = factor_name
        z = cross_sectional_zscore(raw)
        z.name = f"{factor_name}_zscore"

        quintile = _compute_quintile(raw)
        quintile.name = f"{factor_name}_quintile"

        result_frames[factor_name]                   = raw
        result_frames[f"{factor_name}_zscore"]       = z
        result_frames[f"{factor_name}_quintile"]     = quintile

        non_null = raw.notna().sum()
        logger.info(
            "%s: %s @ %s → %d non-null / %d tickers",
            log_prefix, factor_name, as_of_date, non_null, len(raw),
        )

    df = pd.DataFrame(result_frames)
    df.index.name = "ticker"
    return df


# ---------------------------------------------------------------------------
# compute_all_factors — 批量计算入口
# ---------------------------------------------------------------------------

def compute_all_factors(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    """
    对给定日期计算全部 6 个学术因子，返回原始值、z-score 和五分位排名。

    输出 DataFrame 列说明
    --------------------
    对每个因子 {name}，生成以下三列：
    - {name}           : 原始截面因子值
    - {name}_zscore    : 经 Winsorize + z-score 标准化后的值
    - {name}_quintile  : 截面五分位排名（1=最低，5=最高；NaN=数据缺失）

    参数
    ----
    xbrl_df : pd.DataFrame
        xbrl_facts 表（含 available_date 列）。
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    as_of_date : str
        计算日期，格式 "YYYY-MM-DD"。

    返回
    ----
    pd.DataFrame
        index=ticker，列如上所述。
        行数等于 prices_df.columns 中 ticker 的数量。

    示例
    ----
    >>> factor_df = compute_all_factors(xbrl_df, prices_df, "2023-01-31")
    >>> factor_df["EarningsYield"].dropna().shape[0] > 100
    True
    """
    factor_functions: dict[str, callable] = {
        "EarningsYield":      compute_earnings_yield,
        "GrossProfitability": compute_gross_profitability,
        "AssetGrowth":        compute_asset_growth,
        "Accruals":           compute_accruals,
        "Momentum12_1":       compute_momentum_12_1,
        "NetDebtEBITDA":      compute_net_debt_ebitda,
    }
    return _run_factor_batch(
        factor_functions, xbrl_df, prices_df, as_of_date,
        log_prefix="compute_all_factors",
    )


# ---------------------------------------------------------------------------
# Control Factor: Future Return (Positive Control — Intentional Look-Ahead)
# ---------------------------------------------------------------------------

def compute_future_return_control(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    POSITIVE CONTROL: Uses 1-month-ahead returns as a "factor."

    This is intentional look-ahead bias. The pipeline MUST accept this factor
    with IC t-stat > 5.0 if it's working correctly. If it doesn't, the IC
    computation or data alignment has a bug.

    WARNING: This factor is for pipeline validation ONLY. It must never be
    included in production factor libraries or count toward multiple testing.

    Parameters
    ----------
    xbrl_df : pd.DataFrame
        Unused, kept for interface consistency.
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=adj_close.
    as_of_date : str
        Computation date, format "YYYY-MM-DD".

    Returns
    -------
    pd.Series
        index=ticker, values=21-trading-day forward return (intentional leak).
    """
    date_ts = pd.Timestamp(as_of_date)
    valid_dates = prices_df.index[prices_df.index >= date_ts]

    if len(valid_dates) < 22:
        return pd.Series(np.nan, index=prices_df.columns, name="FutureReturnControl")

    price_now = prices_df.loc[valid_dates[0]]
    price_fwd = prices_df.loc[valid_dates[min(21, len(valid_dates) - 1)]]

    fwd_ret = (price_fwd / price_now) - 1.0
    fwd_ret = fwd_ret.where(price_now > 0, np.nan)
    fwd_ret.name = "FutureReturnControl"
    return fwd_ret


# ---------------------------------------------------------------------------
# Control Factor: Random Noise (Negative Control)
# ---------------------------------------------------------------------------

def compute_random_noise_control(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.Series:
    """
    NEGATIVE CONTROL: Pure random noise as a "factor."

    The pipeline MUST reject this factor with IC t-stat < 1.0. If it passes,
    the pipeline has systematic bias or the thresholds are too loose.

    Uses a deterministic seed derived from as_of_date for reproducibility
    across runs, while ensuring different values each month.

    Parameters
    ----------
    xbrl_df : pd.DataFrame
        Unused, kept for interface consistency.
    prices_df : pd.DataFrame
        index=date, columns=ticker — used only for the ticker list.
    as_of_date : str
        Computation date. Used to derive per-month seed for reproducibility.

    Returns
    -------
    pd.Series
        index=ticker, values=random noise drawn from N(0,1).
    """
    # Deterministic seed: base seed 42 + date hash for per-month variation
    date_hash = int(pd.Timestamp(as_of_date).strftime("%Y%m%d"))
    rng = np.random.RandomState(42 + date_hash)
    noise = rng.randn(len(prices_df.columns))
    result = pd.Series(noise, index=prices_df.columns, name="RandomNoiseControl")
    return result


# ---------------------------------------------------------------------------
# compute_control_factors — batch entry for pipeline validation
# ---------------------------------------------------------------------------

def compute_control_factors(
    xbrl_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Compute positive and negative control factors for pipeline validation.

    These factors are NOT academic factors — they exist solely to validate
    that the IC computation pipeline is mechanically correct:
    - FutureReturnControl (positive): intentional look-ahead, must get IC t-stat > 5.0
    - RandomNoiseControl (negative): pure noise, must get IC t-stat < 1.0

    Returns DataFrame with same structure as compute_all_factors
    (raw, zscore, quintile columns for each control factor).
    """
    control_functions: dict[str, callable] = {
        "FutureReturnControl": compute_future_return_control,
        "RandomNoiseControl":  compute_random_noise_control,
    }
    return _run_factor_batch(
        control_functions, xbrl_df, prices_df, as_of_date,
        log_prefix="compute_control_factors",
    )


def _compute_quintile(raw: pd.Series) -> pd.Series:
    """
    按截面值分为 5 组（quintile），1 = 最低，5 = 最高。

    NaN 值排名为 NaN（不参与五分位划分）。
    采用分位数边界切割（pd.qcut），避免 tied-ranks 导致空组。

    参数
    ----
    raw : pd.Series
        原始截面因子值。

    返回
    ----
    pd.Series
        五分位整数（1-5），NaN 处仍为 NaN。
    """
    valid = raw.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=raw.index)

    try:
        quintile_codes, _ = pd.qcut(
            valid,
            q=5,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        quintile_codes = quintile_codes + 1  # 1-indexed
    except Exception as exc:
        logger.warning("_compute_quintile: pd.qcut 失败 — %s", exc)
        return pd.Series(np.nan, index=raw.index)

    return quintile_codes.reindex(raw.index)
