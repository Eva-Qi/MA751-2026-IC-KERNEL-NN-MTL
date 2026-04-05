"""
taxonomy_map.py — XBRL 概念标准化模块

核心功能:
1. CONCEPT_MAP — concept_key → fallback XBRL tag 列表（按常见度排序）
2. get_concept_value()           → float | None
3. get_concept_with_metadata()   → dict | None
4. build_standardized_financials() → pd.DataFrame
5. check_concept_coverage()      → float
6. run_coverage_report()         → pd.DataFrame
7. compute_derived_concepts()    → dict

问题背景:
不同公司使用不同的 XBRL tag 表示同一经济概念。
例如 Apple 用 NetIncomeLoss，Tesla 用 ProfitLoss。
本模块通过 fallback 列表统一解析，返回第一个命中的非空值。

数据来源格式 (SEC EDGAR companyfacts API):
    facts["us-gaap"][tag]["units"]["USD"] → list of records, each:
    {
        "accn":  "0000320193-23-000077",
        "cik":   320193,
        "entityName": "Apple Inc.",
        "loc":   "US-CA",
        "end":   "2023-09-30",
        "start": "2022-10-01",   # 流量科目有 start，存量科目无
        "val":   96995000000,
        "accession": ...,
        "fy":    2023,
        "fp":    "FY",
        "form":  "10-K",
        "filed": "2023-11-03",
        "frame": "CY2023"
    }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONCEPT_MAP
# ---------------------------------------------------------------------------
# 每个 concept_key 对应一个 fallback 列表，按 XBRL 使用频率由高到低排列。
# 扩展方式：直接在对应列表末尾追加新 tag，无需修改任何函数。
# ---------------------------------------------------------------------------

CONCEPT_MAP: dict[str, list[str]] = {
    "NET_INCOME": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "IncomeLossFromContinuingOperations",  # 49% coverage, fills IPO gaps
    ],
    "REVENUE": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "GROSS_PROFIT": [
        "GrossProfit",
    ],
    "COST_OF_REVENUE": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "TOTAL_ASSETS": [
        "Assets",
    ],
    "TOTAL_DEBT": [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    ],
    "CURRENT_DEBT": [
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
    ],
    "CASH": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "CFO": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "DEPRECIATION": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",  # 70% coverage, shorter form many companies use
    ],
    "OPERATING_INCOME": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "STOCKHOLDERS_EQUITY": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "RETAINED_EARNINGS": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "PP_AND_E": [
        "PropertyPlantAndEquipmentNet",
    ],
    "GOODWILL": [
        "Goodwill",
    ],
    "CURRENT_ASSETS": [
        "AssetsCurrent",
    ],
    "CURRENT_LIABILITIES": [
        "LiabilitiesCurrent",
    ],
    "SHARES_OUTSTANDING": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    # MARKET_CAP_PROXY 不在 XBRL 中，由 price × shares 计算得出
    "MARKET_CAP_PROXY": [],
}

# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

_COVERAGE_THRESHOLD: float = 0.85


def _get_usd_records(us_gaap_facts: dict[str, Any], tag: str) -> list[dict]:
    """
    从 us-gaap facts 中提取指定 tag 的 USD 单位记录列表。

    参数:
        us_gaap_facts: companyfacts["facts"]["us-gaap"] 字典。
        tag: XBRL tag 名称，例如 "NetIncomeLoss"。

    返回:
        USD 记录列表；若 tag 不存在或无 USD 单位则返回空列表。
    """
    if tag not in us_gaap_facts:
        return []
    units: dict[str, Any] = us_gaap_facts[tag].get("units", {})
    if not isinstance(units, dict):
        return []
    records = units.get("USD", [])
    if not isinstance(records, list):
        return []
    return records


def _filter_records(
    records: list[dict],
    fiscal_year: int,
    fiscal_period: str,
) -> list[dict]:
    """
    按 fiscal year 和 fiscal period 过滤记录。

    参数:
        records: USD 记录列表。
        fiscal_year: 目标财年，例如 2023。
        fiscal_period: 财报期间，例如 "FY"、"Q1"。

    返回:
        过滤后的记录列表。
    """
    return [
        r for r in records
        if r.get("fy") == fiscal_year and r.get("fp") == fiscal_period
    ]


def _pick_latest(records: list[dict]) -> dict | None:
    """
    从多条记录中选取 filed 日期最新的一条（处理 amended filings）。

    参数:
        records: 已过滤的记录列表。

    返回:
        filed 日期最新的记录，若列表为空则返回 None。
    """
    if not records:
        return None
    return max(records, key=lambda r: r.get("filed", ""))


# ---------------------------------------------------------------------------
# 公开函数
# ---------------------------------------------------------------------------

def get_concept_value(
    us_gaap_facts: dict[str, Any],
    concept_key: str,
    fiscal_year: int,
    fiscal_period: str = "FY",
) -> float | None:
    """
    按 CONCEPT_MAP 中的 fallback 顺序查找概念值，返回第一个命中的非空值。

    参数:
        us_gaap_facts: companyfacts["facts"]["us-gaap"] 字典。
        concept_key: CONCEPT_MAP 中定义的标准概念键，例如 "REVENUE"。
        fiscal_year: 目标财年，例如 2023。
        fiscal_period: 财报期间，默认 "FY"（年度）。

    返回:
        匹配记录的数值（float），若所有 tag 均无匹配则返回 None。

    示例:
        >>> val = get_concept_value(aapl_facts["facts"]["us-gaap"], "REVENUE", 2023)
        >>> assert val > 0
    """
    tags = CONCEPT_MAP.get(concept_key, [])
    for tag in tags:
        records = _get_usd_records(us_gaap_facts, tag)
        matches = _filter_records(records, fiscal_year, fiscal_period)
        best = _pick_latest(matches)
        if best is not None:
            val = best.get("val")
            if val is not None:
                return float(val)
    return None


def get_concept_with_metadata(
    us_gaap_facts: dict[str, Any],
    concept_key: str,
    fiscal_year: int,
    fiscal_period: str = "FY",
) -> dict[str, Any] | None:
    """
    与 get_concept_value 逻辑相同，但返回包含完整元数据的字典。
    用于 filing date 对齐和数据溯源。

    参数:
        us_gaap_facts: companyfacts["facts"]["us-gaap"] 字典。
        concept_key: CONCEPT_MAP 中定义的标准概念键。
        fiscal_year: 目标财年。
        fiscal_period: 财报期间，默认 "FY"。

    返回:
        命中时返回如下结构的字典：
        {
            "value":        float,     # 数值
            "tag_used":     str,       # 实际命中的 XBRL tag
            "filed_date":   str,       # 提交日期，格式 "YYYY-MM-DD"
            "period_start": str | None,# 期间起始日（流量科目有，存量科目无）
            "period_end":   str,       # 期间终止日
            "form":         str,       # 表单类型，例如 "10-K"
        }
        若所有 tag 均无匹配则返回 None。
    """
    tags = CONCEPT_MAP.get(concept_key, [])
    for tag in tags:
        records = _get_usd_records(us_gaap_facts, tag)
        matches = _filter_records(records, fiscal_year, fiscal_period)
        best = _pick_latest(matches)
        if best is not None and best.get("val") is not None:
            return {
                "value":        float(best["val"]),
                "tag_used":     tag,
                "filed_date":   best.get("filed", ""),
                "period_start": best.get("start"),
                "period_end":   best.get("end", ""),
                "form":         best.get("form", ""),
            }
    return None


def build_standardized_financials(
    cik_str: str,
    facts: dict[str, Any],
    years: range = range(2014, 2025),
) -> pd.DataFrame:
    """
    从单家公司的 companyfacts 构建标准化财务数据 DataFrame。

    每行对应一个财年，每个 CONCEPT_MAP 概念产生三列：
      - {concept}       : 标准化数值（float 或 None）
      - {concept}_tag   : 实际命中的 XBRL tag（str 或 None）
      - {concept}_filed : 提交日期（str 或 None）

    参数:
        cik_str: 10 位 CIK 字符串，例如 "0000320193"。
        facts: SEC EDGAR companyfacts 完整 JSON dict。
        years: 需要提取的财年范围，默认 2014–2024。

    返回:
        pd.DataFrame，列为 ["fiscal_year", "cik", concept..., concept_tag..., concept_filed...]。
        行数等于 len(years)，包含无数据的年份（值为 None/NaN）。
    """
    us_gaap: dict[str, Any] = facts.get("facts", {}).get("us-gaap", {})
    if not isinstance(us_gaap, dict):
        logger.warning("CIK %s: facts['us-gaap'] 缺失或格式异常", cik_str)
        us_gaap = {}

    rows: list[dict[str, Any]] = []
    for year in years:
        row: dict[str, Any] = {"fiscal_year": year, "cik": cik_str}
        for concept in CONCEPT_MAP:
            meta = get_concept_with_metadata(us_gaap, concept, year, "FY")
            if meta is not None:
                row[concept] = meta["value"]
                row[f"{concept}_tag"] = meta["tag_used"]
                row[f"{concept}_filed"] = meta["filed_date"]
            else:
                row[concept] = None
                row[f"{concept}_tag"] = None
                row[f"{concept}_filed"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def check_concept_coverage(
    all_company_facts: dict[str, dict[str, Any]],
    concept_key: str,
    fiscal_year: int = 2023,
) -> float:
    """
    计算某概念在所有公司中的非空覆盖率。

    覆盖率 = 能取到非空值的公司数 / 有有效 us-gaap 数据的公司总数。
    合格门槛：>= 0.85（见 _COVERAGE_THRESHOLD）。

    参数:
        all_company_facts: {cik_str: companyfacts_dict} 字典，
                           键为 10 位 CIK，值为完整 companyfacts JSON。
        concept_key: CONCEPT_MAP 中定义的标准概念键。
        fiscal_year: 检查的目标财年，默认 2023。

    返回:
        non-null rate，取值范围 [0.0, 1.0]。
        若无任何公司有有效数据则返回 0.0。
    """
    total: int = 0
    non_null: int = 0

    for cik, facts in all_company_facts.items():
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        if not isinstance(us_gaap, dict) or not us_gaap:
            continue
        total += 1
        val = get_concept_value(us_gaap, concept_key, fiscal_year)
        if val is not None:
            non_null += 1

    if total == 0:
        logger.warning("check_concept_coverage: 没有找到任何有效的 us-gaap 数据")
        return 0.0

    return non_null / total


def run_coverage_report(
    all_company_facts: dict[str, dict[str, Any]],
    fiscal_year: int = 2023,
) -> pd.DataFrame:
    """
    对 CONCEPT_MAP 中所有概念运行覆盖率检查，输出汇总报告并打印 pass/fail 状态。

    参数:
        all_company_facts: {cik_str: companyfacts_dict} 字典。
        fiscal_year: 检查的目标财年，默认 2023。

    返回:
        pd.DataFrame，列为：
          - concept       : 概念键
          - total         : 有效 us-gaap 公司总数
          - non_null      : 命中非空值的公司数
          - coverage_rate : non_null / total，保留 4 位小数
          - pass          : coverage_rate >= _COVERAGE_THRESHOLD
    """
    # 预先计算 total（所有概念共享同一分母）
    total_companies: int = sum(
        1 for facts in all_company_facts.values()
        if isinstance(facts.get("facts", {}).get("us-gaap"), dict)
        and facts["facts"]["us-gaap"]
    )

    report_rows: list[dict[str, Any]] = []
    for concept, tags in CONCEPT_MAP.items():
        if not tags:
            # 纯派生概念（如 MARKET_CAP_PROXY），跳过覆盖率检查
            report_rows.append({
                "concept":       concept,
                "total":         total_companies,
                "non_null":      0,
                "coverage_rate": None,
                "pass":          None,
            })
            continue

        rate = check_concept_coverage(all_company_facts, concept, fiscal_year)
        non_null_count = round(rate * total_companies)
        passed = rate >= _COVERAGE_THRESHOLD

        report_rows.append({
            "concept":       concept,
            "total":         total_companies,
            "non_null":      non_null_count,
            "coverage_rate": round(rate, 4),
            "pass":          passed,
        })

    df = pd.DataFrame(report_rows)

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"Coverage Report — fiscal_year={fiscal_year}  threshold={_COVERAGE_THRESHOLD:.0%}")
    print(f"{'=' * 60}")
    for _, row in df.iterrows():
        rate_is_null = pd.isna(row["coverage_rate"])
        if rate_is_null:
            status = "SKIP (computed)"
        elif row["pass"]:
            status = "PASS"
        else:
            status = "FAIL"
        rate_str = f"{row['coverage_rate']:.2%}" if not rate_is_null else "  N/A  "
        print(
            f"  [{status:<14}]  {row['concept']:<25}  "
            f"{row['non_null']:>5} / {row['total']:<5}  ({rate_str})"
        )
    print(f"{'=' * 60}\n")

    return df


def compute_derived_concepts(financials_row: dict[str, Any]) -> dict[str, float | None]:
    """
    对单行标准化财务数据计算派生概念。

    当前派生规则：
      - EBITDA          = OPERATING_INCOME + DEPRECIATION
      - TOTAL_DEBT_NET  = TOTAL_DEBT + CURRENT_DEBT - CASH
      - MARKET_CAP_PROXY: 需外部 price 数据，此处返回 None

    参数:
        financials_row: build_standardized_financials 返回 DataFrame 的一行（dict 形式）。
                        至少包含 CONCEPT_MAP 中各概念键对应的值（可为 None）。

    返回:
        {派生概念名: 数值 or None}。
        任一分量为 None 时，对应派生值也为 None（避免静默产生错误数据）。

    示例:
        >>> row = df.iloc[0].to_dict()
        >>> derived = compute_derived_concepts(row)
        >>> ebitda = derived["EBITDA"]
    """
    def _safe_add(*vals: float | None) -> float | None:
        """所有分量均不为 None 时求和，否则返回 None。"""
        if any(v is None for v in vals):
            return None
        return sum(float(v) for v in vals)  # type: ignore[arg-type]

    operating_income: float | None = financials_row.get("OPERATING_INCOME")
    depreciation: float | None     = financials_row.get("DEPRECIATION")
    total_debt: float | None       = financials_row.get("TOTAL_DEBT")
    current_debt: float | None     = financials_row.get("CURRENT_DEBT")
    cash: float | None             = financials_row.get("CASH")

    # EBITDA = OPERATING_INCOME + DEPRECIATION
    ebitda: float | None = _safe_add(operating_income, depreciation)

    # TOTAL_DEBT_NET = TOTAL_DEBT + CURRENT_DEBT - CASH
    if total_debt is not None and current_debt is not None and cash is not None:
        total_debt_net: float | None = float(total_debt) + float(current_debt) - float(cash)
    else:
        total_debt_net = None

    # MARKET_CAP_PROXY 需要外部价格数据，此处不可计算
    market_cap_proxy: float | None = None

    return {
        "EBITDA":           ebitda,
        "TOTAL_DEBT_NET":   total_debt_net,
        "MARKET_CAP_PROXY": market_cap_proxy,
    }
