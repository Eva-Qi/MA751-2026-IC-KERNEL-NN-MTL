"""
data_quality.py — 数据质量检查模块

核心功能:
1. check_coverage(all_facts, concepts, year) → DataFrame
   对 CONCEPT_MAP 中所有概念计算非空覆盖率，wraps taxonomy_map.run_coverage_report()
   并将结果写入 SQLite data_quality_log 表。

2. check_outliers(factor_df, sigma_threshold=5.0) → DataFrame
   对因子值矩阵（date × ticker）计算 z-score，flag |z-score| > threshold 的记录，
   不自动删除，仅标记并写入 SQLite。

3. check_staleness(filing_dates_df, max_lag_days=120) → DataFrame
   检测最新 10-K/10-Q 申报日期距今是否超过 max_lag_days，flag 超期公司。

4. check_price_completeness(prices_df, min_coverage=0.95) → DataFrame
   检测每个 ticker 的价格非空率是否 >= min_coverage。

5. run_all_quality_checks(all_facts, prices_df, factor_dfs, db_path) → dict
   串联运行所有检查，写入 SQLite，打印格式化摘要，返回 summary dict。

SQLite 表结构（由调用方或 schema 预先建立）:
    CREATE TABLE IF NOT EXISTS data_quality_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        check_type  TEXT NOT NULL,
        ticker      TEXT,
        concept     TEXT,
        details     TEXT,
        flagged_at  TEXT DEFAULT (datetime('now'))
    );

设计原则:
- 所有检查均为非破坏性操作：flag，不修改输入数据
- 对空 DataFrame、缺失数据等边界情况均做防御处理
- 日志同时写入 console（print）和 SQLite
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from alpha_system.core.data_pipeline.taxonomy_map import CONCEPT_MAP, run_coverage_report

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_COVERAGE_THRESHOLD: float = 0.85
_DEFAULT_SIGMA: float = 5.0
_DEFAULT_MAX_LAG_DAYS: int = 120
_DEFAULT_MIN_PRICE_COVERAGE: float = 0.95

_CHECK_TYPES = {
    "COVERAGE":   "COVERAGE",
    "OUTLIER":    "OUTLIER",
    "STALE":      "STALE",
    "MISSING":    "MISSING",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------


def _utcnow_str() -> str:
    """返回 'YYYY-MM-DD HH:MM:SS' 格式的 UTC 时间字符串，用于 flagged_at 字段。"""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextlib.contextmanager
def _get_connection(db_path: str):
    """
    打开 SQLite 连接，确保 data_quality_log 表存在，并在退出时关闭连接。

    作为 context manager 使用：
        with _get_connection(db_path) as conn:
            ...

    Parameters
    ----------
    db_path : str
        SQLite 数据库文件路径。

    Yields
    ------
    sqlite3.Connection
        已建立连接；退出 with 块时自动关闭（无论是否发生异常）。
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                check_type  TEXT NOT NULL,
                ticker      TEXT,
                concept     TEXT,
                details     TEXT,
                flagged_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        yield conn
    finally:
        conn.close()


def _insert_log_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """
    批量插入 data_quality_log 记录。

    Parameters
    ----------
    conn : sqlite3.Connection
        已打开的 SQLite 连接。
    rows : list[dict]
        每个 dict 需包含键：check_type, ticker, concept, details。
        flagged_at 由数据库默认值或由调用方写入。
    """
    if not rows:
        return
    now_str = _utcnow_str()
    conn.executemany(
        """
        INSERT INTO data_quality_log (check_type, ticker, concept, details, flagged_at)
        VALUES (:check_type, :ticker, :concept, :details, :flagged_at)
        """,
        [
            {
                "check_type": r.get("check_type", "UNKNOWN"),
                "ticker":     r.get("ticker"),
                "concept":    r.get("concept"),
                "details":    r.get("details"),
                "flagged_at": r.get("flagged_at", now_str),
            }
            for r in rows
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. check_coverage
# ---------------------------------------------------------------------------


def check_coverage(
    all_facts: dict[str, dict[str, Any]],
    concepts: list[str],
    year: int,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    对 CONCEPT_MAP 中每个概念计算 S&P 500 范围内的非空覆盖率。

    内部调用 taxonomy_map.run_coverage_report() 获取原始覆盖率 DataFrame，
    再对未通过阈值的概念写入 SQLite data_quality_log 表（如果提供了 db_path）。

    Parameters
    ----------
    all_facts : dict[str, dict]
        {cik_str: companyfacts_dict}，键为 10 位 CIK，
        来自 xbrl_loader.fetch_all_sp500_facts()。
    concepts : list[str]
        要检查的 CONCEPT_MAP 概念键列表，例如 ["REVENUE", "NET_INCOME"]。
        若传入空列表则检查 CONCEPT_MAP 中全部概念。
    year : int
        检查的目标财年，例如 2023。
    db_path : str | None
        SQLite 数据库路径；为 None 时跳过持久化写入。

    Returns
    -------
    pd.DataFrame
        列: concept, total, non_null, coverage_rate, pass
        行数等于 CONCEPT_MAP 中的概念数（含派生概念行，coverage_rate=None）。

    Notes
    -----
    - 覆盖率阈值固定为 0.85（与 taxonomy_map._COVERAGE_THRESHOLD 一致）。
    - 对 FAIL 的概念在 console 额外输出警告。
    """
    print(f"\n[data_quality] check_coverage — fiscal_year={year}, threshold={_COVERAGE_THRESHOLD:.0%}")

    # 防御：空输入
    if not all_facts:
        logger.warning("check_coverage: all_facts 为空，返回空 DataFrame")
        print("  [WARN] all_facts 为空，跳过 coverage 检查")
        return pd.DataFrame(columns=["concept", "total", "non_null", "coverage_rate", "pass"])

    # 调用 taxonomy_map 的覆盖率报告（会打印自身的格式化输出）
    report_df: pd.DataFrame = run_coverage_report(all_facts, fiscal_year=year)

    # 如果调用方指定了概念子集，过滤报告行
    if concepts:
        concept_set = set(concepts)
        report_df = report_df[report_df["concept"].isin(concept_set)].reset_index(drop=True)

    # 找出 FAIL 的行，写日志
    fail_rows: list[dict[str, Any]] = []
    now_str = _utcnow_str()

    for _, row in report_df.iterrows():
        passed = row["pass"]
        # None 表示跳过检查的派生概念（如 MARKET_CAP_PROXY）
        if passed is None or pd.isna(passed):
            continue
        if not passed:
            rate_str = f"{row['coverage_rate']:.4f}" if pd.notna(row["coverage_rate"]) else "N/A"
            msg = (
                f"coverage={rate_str} < {_COVERAGE_THRESHOLD} "
                f"(non_null={row['non_null']}/{row['total']})"
            )
            print(f"  [FAIL] {row['concept']}: {msg}")
            logger.warning("check_coverage FAIL — concept=%s %s", row["concept"], msg)
            fail_rows.append({
                "check_type": _CHECK_TYPES["COVERAGE"],
                "ticker":     None,
                "concept":    row["concept"],
                "details":    msg,
                "flagged_at": now_str,
            })

    # 持久化写入 SQLite
    if db_path and fail_rows:
        try:
            with _get_connection(db_path) as conn:
                _insert_log_rows(conn, fail_rows)
            print(f"  [data_quality] 已写入 {len(fail_rows)} 条 COVERAGE FAIL 记录到 {db_path}")
        except sqlite3.Error as exc:
            logger.error("check_coverage: SQLite 写入失败 — %s", exc)
            print(f"  [ERROR] SQLite 写入失败: {exc}")

    n_pass = report_df["pass"].eq(True).sum()
    n_fail = report_df["pass"].eq(False).sum()
    n_skip = report_df["pass"].isna().sum()
    print(
        f"  [data_quality] coverage 汇总: PASS={n_pass}, FAIL={n_fail}, SKIP(派生)={n_skip}"
    )

    return report_df


# ---------------------------------------------------------------------------
# 2. check_outliers
# ---------------------------------------------------------------------------


def check_outliers(
    factor_df: pd.DataFrame,
    sigma_threshold: float = _DEFAULT_SIGMA,
    db_path: str | None = None,
    factor_name: str = "unknown",
) -> pd.DataFrame:
    """
    对因子值矩阵逐截面计算 z-score，flag |z-score| > sigma_threshold 的记录。

    不自动移除异常值——仅标记（flagged=True），保持数据原样。

    Parameters
    ----------
    factor_df : pd.DataFrame
        index=date (pd.Timestamp 或字符串), columns=ticker (str),
        values=float 因子原始值。
    sigma_threshold : float
        z-score 绝对值超过该阈值则标记为异常，默认 5.0。
    db_path : str | None
        SQLite 数据库路径；为 None 时跳过持久化写入。
    factor_name : str
        因子名称，用于日志记录（写入 concept 字段）。

    Returns
    -------
    pd.DataFrame
        列: date, ticker, factor_value, zscore, flagged
        仅包含 flagged=True 的行（若无异常则返回空 DataFrame，列结构不变）。

    Notes
    -----
    - z-score 按每个截面（每个 date）独立计算，使用 mean/std（ddof=1）。
    - 若某截面标准差为 0 或 NaN，该截面所有值的 z-score 设为 NaN，不标记异常。
    - NaN 因子值不参与 z-score 计算，不被标记为异常。
    """
    _empty_cols = ["date", "ticker", "factor_value", "zscore", "flagged"]
    _empty = pd.DataFrame(columns=_empty_cols)

    print(
        f"\n[data_quality] check_outliers — factor={factor_name}, "
        f"sigma_threshold={sigma_threshold}, shape={factor_df.shape}"
    )

    if factor_df.empty:
        logger.warning("check_outliers: factor_df 为空，跳过")
        print("  [WARN] factor_df 为空，跳过 outlier 检查")
        return _empty

    flagged_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    now_str = _utcnow_str()

    for date_idx in factor_df.index:
        cross_section: pd.Series = factor_df.loc[date_idx].dropna()
        if cross_section.empty or len(cross_section) < 2:
            continue

        mean_val: float = float(cross_section.mean())
        std_val: float = float(cross_section.std(ddof=1))

        if std_val == 0.0 or np.isnan(std_val):
            continue

        zscores: pd.Series = (cross_section - mean_val) / std_val
        flagged_mask: pd.Series = zscores.abs() > sigma_threshold

        for ticker in flagged_mask[flagged_mask].index:
            raw_val = float(cross_section[ticker])
            z_val = float(zscores[ticker])
            date_str = str(date_idx)

            flagged_rows.append({
                "date":         date_str,
                "ticker":       ticker,
                "factor_value": raw_val,
                "zscore":       round(z_val, 6),
                "flagged":      True,
            })
            log_rows.append({
                "check_type": _CHECK_TYPES["OUTLIER"],
                "ticker":     ticker,
                "concept":    factor_name,
                "details":    json.dumps({
                    "date":    date_str,
                    "value":   raw_val,
                    "zscore":  round(z_val, 6),
                    "threshold": sigma_threshold,
                }),
                "flagged_at": now_str,
            })

    if not flagged_rows:
        print(f"  [data_quality] 未发现 |z-score| > {sigma_threshold} 的异常值")
        return _empty

    result_df = pd.DataFrame(flagged_rows, columns=_empty_cols)
    result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce")

    print(
        f"  [data_quality] 发现 {len(result_df)} 个异常值记录 "
        f"(|z-score| > {sigma_threshold})"
    )
    logger.info(
        "check_outliers: factor=%s flagged=%d records (sigma>%.1f)",
        factor_name, len(result_df), sigma_threshold,
    )

    # 持久化写入 SQLite
    if db_path and log_rows:
        try:
            with _get_connection(db_path) as conn:
                _insert_log_rows(conn, log_rows)
            print(f"  [data_quality] 已写入 {len(log_rows)} 条 OUTLIER 记录到 {db_path}")
        except sqlite3.Error as exc:
            logger.error("check_outliers: SQLite 写入失败 — %s", exc)
            print(f"  [ERROR] SQLite 写入失败: {exc}")

    return result_df


# ---------------------------------------------------------------------------
# 3. check_staleness
# ---------------------------------------------------------------------------


def check_staleness(
    filing_dates_df: pd.DataFrame,
    max_lag_days: int = _DEFAULT_MAX_LAG_DAYS,
    reference_date: pd.Timestamp | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    检测各公司最新 10-K/10-Q 申报日期距参考日期是否超过 max_lag_days。

    Parameters
    ----------
    filing_dates_df : pd.DataFrame
        来自 xbrl_loader.extract_filing_dates() 的 DataFrame，
        必须包含列: ticker, cik, filed_date, form_type。
        若无 ticker 列（单公司模式），cik 列用作标识。
    max_lag_days : int
        超过该天数则标记为 stale，默认 120 天。
    reference_date : pd.Timestamp | None
        用于计算 days_since 的基准日期；None 时使用当前 UTC 日期。
    db_path : str | None
        SQLite 数据库路径；为 None 时跳过持久化写入。

    Returns
    -------
    pd.DataFrame
        列: ticker, cik, latest_filing_date, days_since, stale
        每行对应一个公司（去重后）。

    Notes
    -----
    - 若 filing_dates_df 为空，返回空 DataFrame（列结构不变）。
    - 仅考虑 10-K、10-Q、10-K/A、10-Q/A 四种表单类型。
    - days_since 为整数（向下取整）。
    """
    _empty_cols = ["ticker", "cik", "latest_filing_date", "days_since", "stale"]
    _empty = pd.DataFrame(columns=_empty_cols)

    print(
        f"\n[data_quality] check_staleness — max_lag_days={max_lag_days}, "
        f"input_rows={len(filing_dates_df)}"
    )

    if filing_dates_df.empty:
        logger.warning("check_staleness: filing_dates_df 为空，跳过")
        print("  [WARN] filing_dates_df 为空，跳过 staleness 检查")
        return _empty

    df = filing_dates_df.copy()

    # 规范化列名
    if "ticker" not in df.columns:
        # 单公司模式：用 cik 列作为 ticker
        if "cik" in df.columns:
            df["ticker"] = df["cik"].astype(str)
        else:
            logger.warning("check_staleness: 既无 ticker 列也无 cik 列，跳过")
            return _empty

    if "cik" not in df.columns:
        df["cik"] = None

    # 过滤有效申报类型
    _valid_forms = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    if "form_type" in df.columns:
        df = df[df["form_type"].isin(_valid_forms)].copy()

    if df.empty:
        print("  [WARN] 过滤后无有效 10-K/10-Q 申报记录")
        return _empty

    # 解析 filed_date
    df["filed_date"] = pd.to_datetime(df.get("filed_date", pd.NaT), errors="coerce")
    df = df.dropna(subset=["filed_date"])

    if df.empty:
        print("  [WARN] 所有 filed_date 均无法解析")
        return _empty

    # 按 ticker 取最新申报日期
    latest_by_ticker = (
        df.groupby("ticker", sort=False)
        .agg(
            latest_filing_date=("filed_date", "max"),
            cik=("cik", "first"),
        )
        .reset_index()
    )

    # 参考日期
    ref_date: pd.Timestamp = (
        reference_date
        if reference_date is not None
        else pd.Timestamp.now(tz="UTC").tz_convert(None)
    )

    latest_by_ticker["days_since"] = (
        (ref_date - latest_by_ticker["latest_filing_date"])
        .dt.days
        .astype("Int64")
    )
    latest_by_ticker["stale"] = latest_by_ticker["days_since"] > max_lag_days

    # 输出统计
    n_stale = latest_by_ticker["stale"].sum()
    n_total = len(latest_by_ticker)
    print(f"  [data_quality] staleness 结果: {n_stale}/{n_total} 家公司申报过期 (>{max_lag_days}天)")

    stale_df = latest_by_ticker[latest_by_ticker["stale"]]
    if not stale_df.empty:
        log_rows: list[dict[str, Any]] = []
        now_str = _utcnow_str()
        for _, row in stale_df.iterrows():
            msg = (
                f"latest_filing={row['latest_filing_date'].strftime('%Y-%m-%d')}, "
                f"days_since={row['days_since']}, max_lag={max_lag_days}"
            )
            logger.warning(
                "check_staleness STALE — ticker=%s %s", row["ticker"], msg
            )
            log_rows.append({
                "check_type": _CHECK_TYPES["STALE"],
                "ticker":     str(row["ticker"]),
                "concept":    None,
                "details":    msg,
                "flagged_at": now_str,
            })

        if db_path and log_rows:
            try:
                with _get_connection(db_path) as conn:
                    _insert_log_rows(conn, log_rows)
                print(f"  [data_quality] 已写入 {len(log_rows)} 条 STALE 记录到 {db_path}")
            except sqlite3.Error as exc:
                logger.error("check_staleness: SQLite 写入失败 — %s", exc)
                print(f"  [ERROR] SQLite 写入失败: {exc}")

    result = latest_by_ticker[_empty_cols].reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 4. check_price_completeness
# ---------------------------------------------------------------------------


def check_price_completeness(
    prices_df: pd.DataFrame,
    min_coverage: float = _DEFAULT_MIN_PRICE_COVERAGE,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    检查每个 ticker 的价格序列中非 NaN 交易日占比是否满足最低要求。

    Parameters
    ----------
    prices_df : pd.DataFrame
        index=date, columns=ticker, values=float 价格（通常为收盘价）。
        每列对应一个 ticker。
    min_coverage : float
        非 NaN 比例的最低门槛，默认 0.95（95%）。
    db_path : str | None
        SQLite 数据库路径；为 None 时跳过持久化写入。

    Returns
    -------
    pd.DataFrame
        列: ticker, total_days, non_null_days, coverage, pass
        每行对应一个 ticker；coverage = non_null_days / total_days。

    Notes
    -----
    - 若 prices_df 为空，返回空 DataFrame（列结构不变）。
    - total_days 指价格矩阵的总行数（含 NaN 行）。
    - 非破坏性：仅标记，不删除任何列或行。
    """
    _empty_cols = ["ticker", "total_days", "non_null_days", "coverage", "pass"]
    _empty = pd.DataFrame(columns=_empty_cols)

    print(
        f"\n[data_quality] check_price_completeness — min_coverage={min_coverage:.0%}, "
        f"shape={prices_df.shape}"
    )

    if prices_df.empty:
        logger.warning("check_price_completeness: prices_df 为空，跳过")
        print("  [WARN] prices_df 为空，跳过 price completeness 检查")
        return _empty

    total_days: int = len(prices_df)
    rows: list[dict[str, Any]] = []

    for ticker in prices_df.columns:
        series = prices_df[ticker]
        non_null_days = int(series.notna().sum())
        coverage = non_null_days / total_days if total_days > 0 else 0.0
        passed = coverage >= min_coverage

        rows.append({
            "ticker":       ticker,
            "total_days":   total_days,
            "non_null_days": non_null_days,
            "coverage":     round(coverage, 6),
            "pass":         passed,
        })

    result_df = pd.DataFrame(rows, columns=_empty_cols)

    # 找出不合格的 ticker
    fail_df = result_df[~result_df["pass"]]
    n_fail = len(fail_df)
    n_total = len(result_df)
    print(
        f"  [data_quality] price completeness 结果: "
        f"{n_total - n_fail}/{n_total} 通过 (coverage >= {min_coverage:.0%})"
    )

    if not fail_df.empty:
        log_rows: list[dict[str, Any]] = []
        now_str = _utcnow_str()
        for _, row in fail_df.iterrows():
            msg = (
                f"coverage={row['coverage']:.4f} < {min_coverage} "
                f"(non_null={row['non_null_days']}/{row['total_days']})"
            )
            logger.warning(
                "check_price_completeness FAIL — ticker=%s %s", row["ticker"], msg
            )
            print(f"  [FAIL] {row['ticker']}: {msg}")
            log_rows.append({
                "check_type": _CHECK_TYPES["MISSING"],
                "ticker":     str(row["ticker"]),
                "concept":    None,
                "details":    msg,
                "flagged_at": now_str,
            })

        if db_path and log_rows:
            try:
                with _get_connection(db_path) as conn:
                    _insert_log_rows(conn, log_rows)
                print(f"  [data_quality] 已写入 {len(log_rows)} 条 MISSING 记录到 {db_path}")
            except sqlite3.Error as exc:
                logger.error("check_price_completeness: SQLite 写入失败 — %s", exc)
                print(f"  [ERROR] SQLite 写入失败: {exc}")

    return result_df


# ---------------------------------------------------------------------------
# 5. Unit Consistency Check (NEW)
# ---------------------------------------------------------------------------

# Financial concepts expected to have unit=USD
_USD_CONCEPTS: set[str] = {
    "NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet", "SalesRevenueGoodsNet", "GrossProfit", "Assets",
    "LongTermDebt", "CashAndCashEquivalentsAtCarryingValue",
    "OperatingIncomeLoss", "StockholdersEquity", "AssetsCurrent",
    "LiabilitiesCurrent", "NetCashProvidedByUsedInOperatingActivities",
    "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
}

# Shares concepts expected to have unit=shares
_SHARES_CONCEPTS: set[str] = {
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
}


def check_unit_consistency(xbrl_df: pd.DataFrame) -> dict[str, Any]:
    """
    Check that financial concepts use USD units and shares concepts use shares units.

    Returns dict with n_usd_issues, n_shares_issues, affected_tickers.
    """
    result: dict[str, Any] = {"n_usd_issues": 0, "n_shares_issues": 0, "affected_tickers": []}

    # USD concepts
    fin_mask = xbrl_df["concept"].isin(_USD_CONCEPTS)
    fin_rows = xbrl_df.loc[fin_mask]
    non_usd = fin_rows[fin_rows["unit"].str.lower() != "usd"]
    result["n_usd_issues"] = len(non_usd)

    # Shares concepts
    sh_mask = xbrl_df["concept"].isin(_SHARES_CONCEPTS)
    sh_rows = xbrl_df.loc[sh_mask]
    non_shares = sh_rows[sh_rows["unit"].str.lower() != "shares"]
    result["n_shares_issues"] = len(non_shares)

    affected = set()
    if len(non_usd) > 0:
        affected.update(non_usd["ticker"].unique())
    if len(non_shares) > 0:
        affected.update(non_shares["ticker"].unique())
    result["affected_tickers"] = sorted(affected)

    total = result["n_usd_issues"] + result["n_shares_issues"]
    if total > 0:
        logger.warning(
            "check_unit_consistency: %d non-USD financial records, "
            "%d non-shares share records, affecting %d tickers",
            result["n_usd_issues"], result["n_shares_issues"], len(affected),
        )
    else:
        logger.info("check_unit_consistency: all units consistent")

    return result


# ---------------------------------------------------------------------------
# 6. Value Magnitude Check (NEW)
# ---------------------------------------------------------------------------


def check_value_magnitude(
    xbrl_df: pd.DataFrame,
    min_shares: float = 1e6,
    max_shares: float = 1e11,
) -> dict[str, Any]:
    """
    Check XBRL values are in plausible ranges for S&P 500 companies.

    Returns dict with n_bad_shares, n_zero_shares, bad_shares_tickers.
    """
    result: dict[str, Any] = {
        "n_bad_shares": 0,
        "n_zero_shares": 0,
        "bad_shares_tickers": [],
    }

    shares_data = xbrl_df[
        xbrl_df["concept"].isin(_SHARES_CONCEPTS)
        & (xbrl_df["unit"].str.lower() == "shares")
    ]

    if shares_data.empty:
        return result

    zero = shares_data[shares_data["value"] == 0]
    too_small = shares_data[(shares_data["value"] > 0) & (shares_data["value"] < min_shares)]
    too_large = shares_data[shares_data["value"] > max_shares]

    bad = pd.concat([zero, too_small, too_large])
    result["n_bad_shares"] = len(too_small) + len(too_large)
    result["n_zero_shares"] = len(zero)
    result["bad_shares_tickers"] = sorted(bad["ticker"].unique()) if len(bad) > 0 else []

    total = result["n_bad_shares"] + result["n_zero_shares"]
    if total > 0:
        logger.warning(
            "check_value_magnitude: %d implausible shares records "
            "(%d zero, %d out-of-range [%.0e, %.0e]), affecting %d tickers",
            total, result["n_zero_shares"], result["n_bad_shares"],
            min_shares, max_shares, len(result["bad_shares_tickers"]),
        )
    else:
        logger.info("check_value_magnitude: all shares values plausible")

    return result


# ---------------------------------------------------------------------------
# 7. run_all_quality_checks
# ---------------------------------------------------------------------------


def run_all_quality_checks(
    all_facts: dict[str, dict[str, Any]],
    prices_df: pd.DataFrame,
    factor_dfs: dict[str, pd.DataFrame],
    db_path: str,
    year: int = 2023,
    filing_dates_df: pd.DataFrame | None = None,
    xbrl_df: pd.DataFrame | None = None,
    sigma_threshold: float = _DEFAULT_SIGMA,
    max_lag_days: int = _DEFAULT_MAX_LAG_DAYS,
    min_price_coverage: float = _DEFAULT_MIN_PRICE_COVERAGE,
) -> dict[str, Any]:
    """
    串联运行所有数据质量检查，将结果持久化到 SQLite，打印格式化摘要报告。

    Parameters
    ----------
    all_facts : dict[str, dict]
        {cik_str: companyfacts_dict}，传给 check_coverage()。
    prices_df : pd.DataFrame
        index=date, columns=ticker 的价格矩阵，传给 check_price_completeness()。
    factor_dfs : dict[str, pd.DataFrame]
        {factor_name: factor_df}，每个 factor_df 的 index=date, columns=ticker，
        传给 check_outliers()。
    db_path : str
        SQLite 数据库文件路径，各检查函数均写入此处。
    year : int
        XBRL coverage 检查的目标财年，默认 2023。
    filing_dates_df : pd.DataFrame | None
        来自 xbrl_loader.extract_filing_dates() 的 DataFrame，传给 check_staleness()。
        若为 None 则跳过 staleness 检查。
    sigma_threshold : float
        outlier 检查的 z-score 阈值，默认 5.0。
    max_lag_days : int
        staleness 检查的最大申报滞后天数，默认 120。
    min_price_coverage : float
        价格完整性检查的最低非空比例，默认 0.95。

    Returns
    -------
    dict[str, Any]
        summary 字典，包含以下键:
        - coverage_pass   : bool — 所有可检查概念均通过覆盖率阈值
        - n_outliers      : int  — 跨所有因子的总 outlier 标记数
        - n_stale         : int  — stale 公司数（若跳过 staleness 则为 None）
        - n_incomplete    : int  — 价格不完整的 ticker 数
        - overall_pass    : bool — 以上所有检查均通过

    Notes
    -----
    - 各子检查互相独立：一个子检查失败不影响其他子检查的执行。
    - 对每个子检查的异常（Exception）做 try/except，记录日志后继续。
    - 最终打印格式化的 ASCII 摘要报告。
    """
    _sep = "=" * 65
    _sub = "-" * 65
    print(f"\n{_sep}")
    print("  DATA QUALITY REPORT — run_all_quality_checks()")
    print(f"  fiscal_year={year}  db={db_path}")
    print(_sep)

    results: dict[str, Any] = {
        "coverage_pass":  None,
        "n_outliers":     0,
        "n_stale":        None,
        "n_incomplete":   0,
        "n_unit_issues":  0,
        "n_magnitude_issues": 0,
        "overall_pass":   False,
    }

    # ---- 1. Coverage check ----
    coverage_df = pd.DataFrame()
    try:
        concepts_all = list(CONCEPT_MAP.keys())
        coverage_df = check_coverage(
            all_facts=all_facts,
            concepts=concepts_all,
            year=year,
            db_path=db_path,
        )
        # coverage_pass = True 当所有可检查概念均通过（None/派生 概念忽略）
        checkable = coverage_df[coverage_df["pass"].notna()]
        if checkable.empty:
            results["coverage_pass"] = True  # 无可检查概念视为通过
        else:
            results["coverage_pass"] = bool(checkable["pass"].all())
    except Exception as exc:
        logger.error("run_all_quality_checks: coverage check 异常 — %s", exc)
        print(f"  [ERROR] coverage 检查失败: {exc}")
        results["coverage_pass"] = False

    # ---- 2. Outlier check (每个因子独立运行) ----
    total_outliers: int = 0
    try:
        if factor_dfs:
            for factor_name, fdf in factor_dfs.items():
                outlier_df = check_outliers(
                    factor_df=fdf,
                    sigma_threshold=sigma_threshold,
                    db_path=db_path,
                    factor_name=factor_name,
                )
                total_outliers += len(outlier_df)
        else:
            print("\n[data_quality] check_outliers — 未提供 factor_dfs，跳过")
        results["n_outliers"] = total_outliers
    except Exception as exc:
        logger.error("run_all_quality_checks: outlier check 异常 — %s", exc)
        print(f"  [ERROR] outlier 检查失败: {exc}")

    # ---- 3. Staleness check ----
    try:
        if filing_dates_df is not None:
            stale_df = check_staleness(
                filing_dates_df=filing_dates_df,
                max_lag_days=max_lag_days,
                db_path=db_path,
            )
            results["n_stale"] = int(stale_df["stale"].sum()) if not stale_df.empty else 0
        else:
            print("\n[data_quality] check_staleness — 未提供 filing_dates_df，跳过")
            results["n_stale"] = None
    except Exception as exc:
        logger.error("run_all_quality_checks: staleness check 异常 — %s", exc)
        print(f"  [ERROR] staleness 检查失败: {exc}")
        results["n_stale"] = None

    # ---- 4. Price completeness check ----
    try:
        price_df = check_price_completeness(
            prices_df=prices_df,
            min_coverage=min_price_coverage,
            db_path=db_path,
        )
        results["n_incomplete"] = int((~price_df["pass"]).sum()) if not price_df.empty else 0
    except Exception as exc:
        logger.error("run_all_quality_checks: price completeness check 异常 — %s", exc)
        print(f"  [ERROR] price completeness 检查失败: {exc}")

    # ---- 5. Unit consistency check ----
    try:
        if xbrl_df is not None:
            unit_result = check_unit_consistency(xbrl_df)
            results["n_unit_issues"] = (
                unit_result["n_usd_issues"] + unit_result["n_shares_issues"]
            )
        else:
            print("\n[data_quality] check_unit_consistency — 未提供 xbrl_df，跳过")
    except Exception as exc:
        logger.error("run_all_quality_checks: unit consistency check 异常 — %s", exc)
        print(f"  [ERROR] unit consistency 检查失败: {exc}")

    # ---- 6. Value magnitude check ----
    try:
        if xbrl_df is not None:
            mag_result = check_value_magnitude(xbrl_df)
            results["n_magnitude_issues"] = (
                mag_result["n_bad_shares"] + mag_result["n_zero_shares"]
            )
        else:
            print("\n[data_quality] check_value_magnitude — 未提供 xbrl_df，跳过")
    except Exception as exc:
        logger.error("run_all_quality_checks: value magnitude check 异常 — %s", exc)
        print(f"  [ERROR] value magnitude 检查失败: {exc}")

    # ---- 综合 overall_pass ----
    coverage_ok = results["coverage_pass"] is True
    outlier_ok = results["n_outliers"] == 0
    # staleness: None 表示跳过，视为通过
    stale_ok = results["n_stale"] is None or results["n_stale"] == 0
    price_ok = results["n_incomplete"] == 0
    results["overall_pass"] = all([coverage_ok, outlier_ok, stale_ok, price_ok])

    # ---- 格式化摘要输出 ----
    def _status(ok: bool | None) -> str:
        if ok is None:
            return "SKIP"
        return "PASS" if ok else "FAIL"

    print(f"\n{_sep}")
    print("  SUMMARY")
    print(_sub)
    print(f"  {'Coverage check':<30} {_status(results['coverage_pass'])}")
    print(
        f"  {'Outlier flags':<30} "
        f"{results['n_outliers']} record(s)  "
        f"[{_status(outlier_ok)}]"
    )
    stale_display = (
        f"{results['n_stale']} company(ies)  [{_status(stale_ok)}]"
        if results["n_stale"] is not None
        else "SKIP (no filing_dates_df)"
    )
    print(f"  {'Stale filings':<30} {stale_display}")
    print(
        f"  {'Incomplete price series':<30} "
        f"{results['n_incomplete']} ticker(s)  "
        f"[{_status(price_ok)}]"
    )
    unit_ok = results["n_unit_issues"] == 0
    print(
        f"  {'Unit consistency':<30} "
        f"{results['n_unit_issues']} issue(s)  "
        f"[{_status(unit_ok)}]"
    )
    mag_ok = results["n_magnitude_issues"] == 0
    print(
        f"  {'Value magnitude':<30} "
        f"{results['n_magnitude_issues']} issue(s)  "
        f"[{_status(mag_ok)}]"
    )
    print(_sub)
    overall_label = "PASS" if results["overall_pass"] else "FAIL"
    print(f"  {'OVERALL':<30} {overall_label}")
    print(f"{_sep}\n")

    logger.info(
        "run_all_quality_checks: overall=%s coverage=%s n_outliers=%d "
        "n_stale=%s n_incomplete=%d",
        overall_label,
        results["coverage_pass"],
        results["n_outliers"],
        results["n_stale"],
        results["n_incomplete"],
    )

    return results
