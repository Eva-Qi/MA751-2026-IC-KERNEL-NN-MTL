"""
price_loader.py
===============
价格数据获取与衍生计算模块。

核心功能:
1. fetch_prices(tickers, start, end)           → DataFrame (adj_close, date × ticker)
2. fetch_volume(tickers, start, end)           → DataFrame (volume,    date × ticker)
3. compute_forward_returns(prices, periods)    → DataFrame (N 日前瞻收益率)
4. compute_trailing_returns(prices, periods)   → DataFrame (N 日滚动动量)
5. compute_market_cap(prices, shares_df)       → DataFrame (市值)
6. liquidity_filter(prices, volume, ...)       → DataFrame[bool] (流动性掩码)
7. get_monthly_rebalance_dates(prices, freq)   → DatetimeIndex (月末交易日)
8. store_prices_to_db(prices, volume, db_path) → None (写入 SQLite)
9. load_prices_from_db(db_path, ...)           → DataFrame

注意 (look-ahead bias):
    compute_forward_returns 在时间 t 使用 t+N 的价格，因此严禁在
    实盘信号生成中调用。调用方有责任只在 validation 阶段使用。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]          # alpha_system/ 的上层
_DATA_DIR = _REPO_ROOT / "alpha_system" / "data"
_PRICES_CACHE = _DATA_DIR / "prices_cache.parquet"
_PRICES_UNADJ_CACHE = _DATA_DIR / "prices_unadj_cache.parquet"
_VOLUME_CACHE = _DATA_DIR / "volume_cache.parquet"

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _ensure_data_dir() -> None:
    """确保数据目录存在。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_with_retry(
    tickers: list[str],
    start: str,
    end: str,
    max_attempts: int = 3,
    sleep_sec: float = 5.0,
) -> pd.DataFrame:
    """
    带重试的 yfinance 批量下载。

    Parameters
    ----------
    tickers : list[str]
        股票代码列表。
    start : str
        起始日期，格式 "YYYY-MM-DD"。
    end : str
        结束日期，格式 "YYYY-MM-DD"（含）。
    max_attempts : int
        最多重试次数，默认 3。
    sleep_sec : float
        每次重试前等待秒数，默认 5。

    Returns
    -------
    pd.DataFrame
        yfinance 原始返回的 DataFrame（可能含 MultiIndex columns）。

    Raises
    ------
    RuntimeError
        所有重试均失败时抛出。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                "yfinance download attempt %d/%d: %d tickers, %s → %s",
                attempt, max_attempts, len(tickers), start, end,
            )
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,    # Close 即已复权价格
                progress=False,
                threads=True,
            )
            return raw
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < max_attempts:
                time.sleep(sleep_sec)

    raise RuntimeError(
        f"yfinance download failed after {max_attempts} attempts"
    ) from last_exc


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    从 yfinance 返回的 DataFrame 中提取单一字段。

    yfinance 在多 ticker 时返回 MultiIndex columns: (field, ticker)。
    单 ticker 时返回扁平 columns。

    Parameters
    ----------
    raw : pd.DataFrame
        yfinance 原始输出。
    field : str
        目标字段，例如 "Close" 或 "Volume"。
    tickers : list[str]
        请求的 ticker 列表，用于识别缺失 ticker。

    Returns
    -------
    pd.DataFrame
        index=date, columns=ticker。
    """
    if raw.empty:
        logger.error("yfinance returned empty DataFrame.")
        return pd.DataFrame()

    # MultiIndex columns → (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            logger.error("Field '%s' not found in MultiIndex columns.", field)
            return pd.DataFrame()
        result = raw[field].copy()
    else:
        # 单 ticker 场景：columns 为字段名
        if field not in raw.columns:
            logger.error("Field '%s' not found in single-ticker columns.", field)
            return pd.DataFrame()
        ticker = tickers[0] if len(tickers) == 1 else "UNKNOWN"
        result = raw[[field]].rename(columns={field: ticker})

    # 确保 index 为 DatetimeIndex
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"

    # 删除全为 NaN 的列（下载失败的 ticker）
    failed = [t for t in tickers if t not in result.columns or result[t].isna().all()]
    if failed:
        logger.warning("Tickers failed to download or all-NaN: %s", failed)
    result = result.dropna(axis=1, how="all")

    return result


# ---------------------------------------------------------------------------
# 内部辅助：公共下载逻辑
# ---------------------------------------------------------------------------

def _fetch_market_data(
    tickers: list[str],
    start: str,
    end: str,
    field: str,
    cache_path: Path,
) -> pd.DataFrame:
    """
    下载单一 yfinance 字段并缓存到 parquet（供 fetch_prices / fetch_volume 共用）。

    Parameters
    ----------
    tickers : list[str]
        股票代码列表。
    start : str
        起始日期，格式 "YYYY-MM-DD"。
    end : str
        结束日期（含），格式 "YYYY-MM-DD"。
    field : str
        yfinance 字段名，例如 "Close" 或 "Volume"。
    cache_path : Path
        缓存写入路径（.parquet）。

    Returns
    -------
    pd.DataFrame
        index=date (DatetimeIndex), columns=ticker。
        下载失败的 ticker 不出现在列中。
    """
    _ensure_data_dir()
    raw = _download_with_retry(tickers, start, end)
    result = _extract_field(raw, field, tickers)

    if not result.empty:
        result.to_parquet(cache_path)
        logger.info("%s cached to %s (%s rows × %s tickers).", field, cache_path, *result.shape)
    else:
        logger.warning("No %s data downloaded; cache not updated.", field)

    return result


def _fetch_unadjusted_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Download unadjusted close prices (auto_adjust=False).

    These are the actual traded prices at the time, NOT retroactively
    adjusted for splits. Use these for market_cap = price × shares
    when shares come from SEC XBRL (which are also not split-adjusted).
    """
    _ensure_data_dir()
    last_exc = None
    for attempt in range(1, 4):
        try:
            logger.info(
                "yfinance unadj download attempt %d/3: %d tickers, %s → %s",
                attempt, len(tickers), start, end,
            )
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            import time as _time
            _time.sleep(5)
    else:
        raise RuntimeError("yfinance unadj download failed") from last_exc

    result = _extract_field(raw, "Close", tickers)

    if not result.empty:
        result.to_parquet(cache_path)
        logger.info("Unadj Close cached to %s (%s rows × %s tickers).",
                     cache_path, *result.shape)
    else:
        logger.warning("No unadj Close data downloaded.")

    return result


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    批量下载复权收盘价，并缓存到 parquet。

    Parameters
    ----------
    tickers : list[str]
        股票代码列表，例如 ["AAPL", "MSFT"]。
    start : str
        起始日期，默认 "2014-01-01"。
    end : str
        结束日期（含），默认 "2024-12-31"。

    Returns
    -------
    pd.DataFrame
        index=date (DatetimeIndex), columns=ticker, values=adj_close (float)。
        下载失败的 ticker 不出现在列中（已记录 warning 日志）。
    """
    return _fetch_market_data(tickers, start, end, "Close", _PRICES_CACHE)


def fetch_unadjusted_prices(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Download unadjusted close prices for market_cap computation.

    Use these when multiplying by SEC XBRL shares_outstanding, which
    are NOT split-adjusted. adjusted_price × raw_shares = WRONG.
    unadjusted_price × raw_shares = CORRECT market_cap.
    """
    return _fetch_unadjusted_prices(tickers, start, end, _PRICES_UNADJ_CACHE)


def fetch_volume(
    tickers: list[str],
    start: str = "2014-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    批量下载成交量数据，并缓存到 parquet。

    Parameters
    ----------
    tickers : list[str]
        股票代码列表。
    start : str
        起始日期，默认 "2014-01-01"。
    end : str
        结束日期（含），默认 "2024-12-31"。

    Returns
    -------
    pd.DataFrame
        index=date (DatetimeIndex), columns=ticker, values=volume (float)。
        下载失败的 ticker 不出现在列中。
    """
    return _fetch_market_data(tickers, start, end, "Volume", _VOLUME_CACHE)


def compute_forward_returns(
    prices: pd.DataFrame,
    periods: int = 21,
) -> pd.DataFrame:
    """
    计算 N 交易日**前瞻**收益率（用于 IC validation）。

    公式::

        return_t = price_{t+N} / price_t - 1

    最后 N 行结果为 NaN（预期行为）。

    .. warning::
        本函数存在未来数据泄露 (look-ahead bias)。
        调用方有责任仅在离线 validation 阶段使用，
        严禁在实盘或回测信号生成路径中调用。

    Parameters
    ----------
    prices : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    periods : int
        前瞻天数，默认 21（约 1 个月）。

    Returns
    -------
    pd.DataFrame
        与 prices 同形，values 为 N 日前瞻收益率。
        最后 N 行为 NaN。
    """
    if periods <= 0:
        raise ValueError(f"periods must be positive, got {periods}")
    return prices.shift(-periods) / prices - 1


def compute_trailing_returns(
    prices: pd.DataFrame,
    periods: int = 252,
) -> pd.DataFrame:
    """
    计算 N 交易日**滚动**（动量）收益率。

    公式::

        return_t = price_t / price_{t-N} - 1

    前 N 行结果为 NaN（预期行为）。
    本函数无未来数据泄露，可安全用于信号生成。

    Parameters
    ----------
    prices : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    periods : int
        回看天数，默认 252（约 1 年）。

    Returns
    -------
    pd.DataFrame
        与 prices 同形，values 为 N 日滚动收益率。
        前 N 行为 NaN。
    """
    if periods <= 0:
        raise ValueError(f"periods must be positive, got {periods}")
    return prices / prices.shift(periods) - 1


def compute_market_cap(
    prices: pd.DataFrame,
    shares_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算市值。

    公式::

        market_cap_{t,i} = price_{t,i} × shares_{t,i}

    Parameters
    ----------
    prices : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    shares_df : pd.DataFrame
        index=date, columns=ticker, values=shares_outstanding。
        通常由季报数据 forward-fill 得到，与 prices 对齐。

    Returns
    -------
    pd.DataFrame
        与 prices 同形，values 为市值（单位：元/美元，取决于输入单位）。
    """
    # 对齐 index 和 columns，避免意外 NaN
    shares_aligned = shares_df.reindex(index=prices.index, columns=prices.columns)
    return prices * shares_aligned


def liquidity_filter(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    min_notional: float = 5e6,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    基于日均成交额的流动性过滤器。

    过滤条件::

        rolling(lookback).mean(price × volume) > min_notional

    Parameters
    ----------
    prices : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    volume : pd.DataFrame
        index=date, columns=ticker, values=成交量（股数）。
    min_notional : float
        日均成交额门槛（美元），默认 5,000,000。
    lookback : int
        滚动窗口（交易日），默认 20。

    Returns
    -------
    pd.DataFrame
        index=date, columns=ticker, dtype=bool。
        True 表示通过流动性过滤，False 表示不足。
        前 ``lookback-1`` 行为 NaN（rolling 预热期）。
    """
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    # 对齐 volume 与 prices
    vol_aligned = volume.reindex(index=prices.index, columns=prices.columns)
    dollar_volume = prices * vol_aligned
    avg_dollar_vol = dollar_volume.rolling(window=lookback, min_periods=lookback).mean()
    return avg_dollar_vol > min_notional


def get_monthly_rebalance_dates(
    prices: pd.DataFrame,
    freq: str = "ME",
) -> pd.DatetimeIndex:
    """
    从价格 index 中提取月末交易日，作为因子计算/再平衡时间点。

    Parameters
    ----------
    prices : pd.DataFrame
        index=date (DatetimeIndex)，为实际交易日历。
    freq : str
        pandas offset alias，默认 "ME"（月末）。
        也接受旧别名 "M"（pandas < 2.2 兼容）。

    Returns
    -------
    pd.DatetimeIndex
        价格 index 中实际存在的月末交易日，升序排列。
    """
    if prices.empty:
        return pd.DatetimeIndex([])

    trade_index = pd.DatetimeIndex(prices.index)

    # 生成日历月末序列，再 snap 到最近的实际交易日
    try:
        month_ends = pd.date_range(trade_index[0], trade_index[-1], freq=freq)
    except ValueError:
        # 兼容旧 pandas alias
        month_ends = pd.date_range(trade_index[0], trade_index[-1], freq="M")

    rebal_dates = []
    for me in month_ends:
        # 取 <= me 的最后一个实际交易日
        candidates = trade_index[trade_index <= me]
        if len(candidates) > 0:
            rebal_dates.append(candidates[-1])

    return pd.DatetimeIndex(sorted(set(rebal_dates)))


# ---------------------------------------------------------------------------
# SQLite 持久化
# ---------------------------------------------------------------------------

def store_prices_to_db(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    db_path: str | Path,
) -> None:
    """
    将价格和成交量数据写入 SQLite prices 表（幂等 upsert）。

    表结构（对应 STAGE_0.md schema）::

        CREATE TABLE IF NOT EXISTS prices (
            ticker  TEXT NOT NULL,
            date    TEXT NOT NULL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (ticker, date)
        );

    Parameters
    ----------
    prices : pd.DataFrame
        index=date, columns=ticker, values=adj_close。
    volume : pd.DataFrame
        index=date, columns=ticker, values=成交量。
    db_path : str | Path
        SQLite 数据库文件路径。

    Returns
    -------
    None
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 转为 long format
    prices_long = prices.stack().rename("close").reset_index()
    prices_long.columns = ["date", "ticker", "close"]
    prices_long["date"] = prices_long["date"].astype(str)

    vol_aligned = volume.reindex(index=prices.index, columns=prices.columns)
    vol_long = vol_aligned.stack().rename("volume").reset_index()
    vol_long.columns = ["date", "ticker", "volume"]
    vol_long["date"] = vol_long["date"].astype(str)

    merged = prices_long.merge(vol_long, on=["date", "ticker"], how="outer")

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker  TEXT NOT NULL,
                date    TEXT NOT NULL,
                close   REAL,
                volume  REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")

        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (ticker, date, close, volume)
            VALUES (:ticker, :date, :close, :volume)
            """,
            merged.to_dict(orient="records"),
        )
        conn.commit()

    logger.info(
        "Stored %d rows to prices table in %s.", len(merged), db_path
    )


def load_prices_from_db(
    db_path: str | Path,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    从 SQLite prices 表加载复权收盘价。

    Parameters
    ----------
    db_path : str | Path
        SQLite 数据库文件路径。
    tickers : list[str] | None
        若指定，只加载这些 ticker；None 表示全部。
    start : str | None
        起始日期过滤（含），格式 "YYYY-MM-DD"。
    end : str | None
        结束日期过滤（含），格式 "YYYY-MM-DD"。

    Returns
    -------
    pd.DataFrame
        index=date (DatetimeIndex), columns=ticker, values=adj_close。
        格式与 fetch_prices() 一致。
        若数据库不存在或表为空，返回空 DataFrame。
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.warning("Database not found: %s", db_path)
        return pd.DataFrame()

    conditions: list[str] = []
    params: list[str] = []

    if tickers:
        placeholders = ",".join("?" * len(tickers))
        conditions.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"SELECT ticker, date, close FROM prices {where_clause} ORDER BY date"

    with sqlite3.connect(db_path) as conn:
        try:
            df_long = pd.read_sql_query(query, conn, params=params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to query prices from %s: %s", db_path, exc)
            return pd.DataFrame()

    if df_long.empty:
        logger.info("No rows returned for given filters.")
        return pd.DataFrame()

    # Pivot to wide format: index=date, columns=ticker
    df_wide = (
        df_long
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .pivot(index="date", columns="ticker", values="close")
    )
    df_wide.index.name = "date"
    df_wide.columns.name = None

    return df_wide
