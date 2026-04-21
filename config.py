"""
Shared configuration for MA751 cross-sectional return prediction.

Single source of truth for column names, feature lists, and constants.
Imported by: main.py, rung3_gam.py, rung5_planned.py, rung5_combined.py,
             statistical_tests.py, load_data.py
"""

# ── Original 6 z-scored factors ──────────────────────────────────────────

FACTOR_ZSCORE_COLS = [
    "EarningsYield_zscore",
    "GrossProfitability_zscore",
    "AssetGrowth_zscore",
    "Accruals_zscore",
    "Momentum12_1_zscore",
    "NetDebtEBITDA_zscore",
]

FACTOR_RAW_COLS = [
    "EarningsYield",
    "GrossProfitability",
    "AssetGrowth",
    "Accruals",
    "Momentum12_1",
    "NetDebtEBITDA",
]

# ── New WRDS-derived features (Phase 1B) ─────────────────────────────────

ANALYST_COLS = [
    "SUE_zscore",
    "AnalystRevision_zscore",
    "AnalystDispersion_zscore",
    "RevisionBreadth_zscore",
]

RISK_COLS = [
    "Beta_zscore",
    "IVOL_zscore",
]

ALT_DATA_COLS = [
    "ShortInterestRatio_zscore",
    "InstOwnershipChg_zscore",
]

# ── New computed features (Phase 2) ─────────────────────────────────────

LIQUIDITY_COLS = [
    "Turnover_zscore",
    "AmihudIlliquidity_zscore",
]

PRICE_PATTERN_COLS = [
    "High52W_Proximity_zscore",
    "MaxDailyReturn_zscore",
    "ReturnSkewness_zscore",
]

GROWTH_COLS = [
    "ImpliedEPSGrowth_zscore",
    "QRevGrowthYoY_zscore",
]

COVERAGE_COLS = [
    "AnalystCoverageChg_zscore",
]

# ── Macro features ───────────────────────────────────────────────────────

MACRO_COLS = [
    "T10Y2Y", "VIXCLS", "UMCSENT", "CFNAI", "UNRATE",
    "BAMLH0A0HYM2", "CPI_YOY", "VIX_TERM_STRUCTURE", "LEADING_COMPOSITE",
]

# ── Regime variables ─────────────────────────────────────────────────────

REGIME_COLS = [
    "mkt_vol_regime",
    "mkt_trend_regime",
]

# HMM-derived regime posteriors (used by Regime-Gated MoE)
N_REGIMES = 3
REGIME_HMM_COLS = [f"regime_p{k}" for k in range(N_REGIMES)]

# HMM input features (market-level, monthly)
HMM_FEATURE_COLS = [
    "mkt_ret_1m",
    "mkt_rv_1m",
    "VIXCLS",
    "T10Y2Y",
    "BAMLH0A0HYM2",
]

# ── Composite feature sets ───────────────────────────────────────────────

# V1: original 15 features (backward compatible)
ALL_FEATURE_COLS_V1 = FACTOR_ZSCORE_COLS + MACRO_COLS

# V2: stock-level features only (macro/regime excluded — zero CS variance)
# Macro/regime are kept as metadata columns for conditioning, not model input
ALL_FEATURE_COLS_V2 = (
    FACTOR_ZSCORE_COLS
    + ANALYST_COLS
    + RISK_COLS
    + ALT_DATA_COLS
)
# 14 stock-level features total

# V3: V2 + Phase 2 computed features (22 stock-level features total)
ALL_FEATURE_COLS_V3 = (
    ALL_FEATURE_COLS_V2  # original 14
    + LIQUIDITY_COLS     # +2
    + PRICE_PATTERN_COLS # +3
    + GROWTH_COLS        # +2
    + COVERAGE_COLS      # +1
)
# 22 stock-level features total

# V2 with macro interactions (optional, for models that can use them)
ALL_FEATURE_COLS_V2_WITH_MACRO = ALL_FEATURE_COLS_V2 + MACRO_COLS + REGIME_COLS

# Default — V2 stock-level features
ALL_FEATURE_COLS = ALL_FEATURE_COLS_V2
FACTOR_COLS = ALL_FEATURE_COLS  # alias used by main.py / rung5

# ── Enhanced Regime MoE feature set (V3-updated, data-driven) ────────────

# V3 LASSO top features (>20% selection rate across 58 folds on 22-feature panel)
ENHANCED_MOE_FEATURE_COLS = [
    "AmihudIlliquidity_zscore",  # 95% LASSO V3, Phase 2 liquidity, top rank
    "Turnover_zscore",           # 95% LASSO V3, Phase 2 liquidity
    "GrossProfitability_zscore", # 79% LASSO V3, Tier 1 quality
    "Beta_zscore",               # 69% LASSO V3, regime-dependent
    "IVOL_zscore",               # 29% LASSO V3, within-sector alpha
    "ShortInterestRatio_zscore", # 26% LASSO V3
    "AnalystDispersion_zscore",  # 17% LASSO V3, regime-conditional signal
]

# Gate conditioning: regime posteriors + key macro (fed to gate, NOT encoder)
GATE_MACRO_COLS = ["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2"]

# ── Column name constants ────────────────────────────────────────────────

TARGET_COL = "fwd_ret_1m"
DATE_COL = "date"
STOCK_COL = "ticker"
SECTOR_COL = "sector"
VOL_COL = "realized_vol"
FWD_VOL_COL = "fwd_realized_vol"   # next-month vol — MTL auxiliary target
RET3M_COL = "fwd_ret_3m"

# ── Type A imputation: sectors where factors are economically undefined ──

TYPE_A_SECTORS = {
    "GrossProfitability_zscore": ["Financials", "Utilities", "Real Estate"],
    "NetDebtEBITDA_zscore": ["Financials", "Real Estate"],
}

# ── Missingness indicators ───────────────────────────────────────────────
# Binary flags (0.0/1.0) for features where NaN means "no data coverage",
# not "zero signal". Used as auxiliary inputs to models that can exploit them.
#   has_sue           : 1 = IBES covers this stock this month
#   has_short_interest: 1 = Compustat has short interest data
#   has_inst_ownership: 1 = 13F filing was reported this month (quarterly)
MISSINGNESS_INDICATORS = [
    "has_sue",
    "has_short_interest",
    "has_inst_ownership",
]

# ── Walk-forward defaults ────────────────────────────────────────────────

DEFAULT_MIN_TRAIN_MONTHS = 60
DEFAULT_PURGE_MONTHS = 1

# ── Sector-adjusted return target ────────────────────────────────────────
SECADJ_TARGET_COL = "fwd_ret_1m_secadj"
