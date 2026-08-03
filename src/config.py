"""Central paths and constants for the Nifty 50 regime-aware forecasting pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Raw inputs (as delivered) -------------------------------------------------
RAW_TRAIN_CSV = PROJECT_ROOT / "Nifty50_train.csv"
RAW_TEST_CSV = PROJECT_ROOT / "Nifty50_test.csv"

# --- Outputs -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"      # cleaned OHLC, no features yet
PROCESSED_DIR = DATA_DIR / "processed"  # feature matrices (step 2 onwards)
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Market conventions (NSE) --------------------------------------------------
TIMEZONE = "Asia/Kolkata"
SESSION_START = "09:15"
SESSION_END = "15:15"        # open time of the final 15-min bar
BAR_MINUTES = 15
BARS_PER_SESSION = 25        # 09:15 .. 15:15 inclusive

# Columns present in the raw files.
OHLC_COLS = ["open", "high", "low", "close"]
# Indicator columns shipped with the raw file. Provenance/warm-up is unknown and
# they disagree across the vendor's export chunks, so they are carried through
# with a `src_` prefix for reference only. Step 2 recomputes everything.
SOURCE_INDICATOR_COLS = [
    "pivot",
    "200_EMA",
    "Supertrend(12,3)",
    "Supertrend(11,2)",
    "Supertrend(10,1)",
]
