"""Step 1 — Nifty 50 market data ingestion.

Turns the two delivered CSVs into a clean, gap-audited, tz-aware 15-minute OHLC
series that every downstream stage (features -> HMM regimes -> models) can rely on.

The cleaning decisions here are deliberate; see `docs/11_Data_Step1.md` for the
evidence behind each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Quality report container
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    """Everything the cleaner observed, so nothing is dropped silently."""

    split: str
    rows_in: int = 0
    rows_out: int = 0
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    sessions: int = 0
    duplicate_ts_rows_dropped: int = 0
    duplicate_dates: list = field(default_factory=list)
    special_session_rows_dropped: int = 0
    special_session_dates: list = field(default_factory=list)
    ohlc_violations: int = 0
    non_positive_prices: int = 0
    nulls: dict = field(default_factory=dict)
    short_sessions: dict = field(default_factory=dict)   # date -> bar count (< 25)
    intraday_gaps: int = 0
    max_abs_bar_return_bps: float = 0.0

    def to_text(self) -> str:
        lines = [
            f"### {self.split}",
            "",
            f"- Rows in / out: **{self.rows_in:,} -> {self.rows_out:,}**",
            f"- Range: **{self.first_ts} -> {self.last_ts}**",
            f"- Trading sessions: **{self.sessions:,}**",
            f"- Duplicate-timestamp rows dropped: **{self.duplicate_ts_rows_dropped:,}** "
            f"across {len(self.duplicate_dates)} dates",
            f"- Special-session rows dropped (Muhurat / live-test): "
            f"**{self.special_session_rows_dropped}** on {len(self.special_session_dates)} dates",
            f"- OHLC consistency violations: **{self.ohlc_violations}**",
            f"- Non-positive prices: **{self.non_positive_prices}**",
            f"- Nulls after cleaning: **{self.nulls or 'none'}**",
            f"- Sessions with < {config.BARS_PER_SESSION} bars: **{len(self.short_sessions)}**",
            f"- Missing intraday bars (holes inside a session): **{self.intraday_gaps}**",
            f"- Largest single-bar move: **{self.max_abs_bar_return_bps:.0f} bps**",
        ]
        if self.short_sessions:
            worst = sorted(self.short_sessions.items(), key=lambda kv: kv[1])[:10]
            lines.append(
                "- Shortest sessions: "
                + ", ".join(f"{d} ({n} bars)" for d, n in worst)
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw CSV and build a single tz-aware timestamp index column.

    The raw files carry `date` (DD-MM-YYYY) and `time` (HH:MM:SS+05:30) separately.
    The offset is constant (+05:30) so it is parsed off and replaced with a proper
    Asia/Kolkata localisation.
    """
    df = pd.read_csv(path)

    missing = {"date", "time", *config.OHLC_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

    ts = pd.to_datetime(
        df["date"].str.strip() + " " + df["time"].str.slice(0, 8),
        format="%d-%m-%Y %H:%M:%S",
    ).dt.tz_localize(config.TIMEZONE)

    df = df.drop(columns=["date", "time"])
    df.insert(0, "timestamp", ts)

    df = df.rename(columns={c: f"src_{c}" for c in config.SOURCE_INDICATOR_COLS
                            if c in df.columns})
    return df


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #
def _drop_duplicate_timestamps(df: pd.DataFrame, rep: QualityReport) -> pd.DataFrame:
    """Drop the second copy of every duplicated bar.

    The vendor export was stitched from overlapping chunks: 74 boundary dates in
    train (8 in test) appear twice, back to back. `close` is identical between the
    copies; `open`/`high`/`low` differ on a minority of bars and the *second*
    copy's first open is discontinuous with the previous session's close by ~95
    index points on average, versus ~1.3 for the first copy. The first copy is
    therefore the one that chains correctly, so `keep="first"` is used.
    """
    dup = df["timestamp"].duplicated(keep="first")
    if dup.any():
        rep.duplicate_ts_rows_dropped = int(dup.sum())
        rep.duplicate_dates = sorted(
            {str(d) for d in df.loc[dup, "timestamp"].dt.date}
        )
    return df.loc[~dup].copy()


def _drop_special_sessions(df: pd.DataFrame, rep: QualityReport) -> pd.DataFrame:
    """Remove bars outside the 09:15-15:15 regular session.

    These are Muhurat (Diwali) trading sessions and the 24-Feb-2021 special live
    session: 4-5 bars each, held in the evening, with no regular-session bars on
    the same date. They carry no usable intraday structure, break the fixed
    25-bar session grid, and would pollute both the technical indicators and the
    HMM's volatility estimates.
    """
    tod = df["timestamp"].dt.strftime("%H:%M")
    in_session = (tod >= config.SESSION_START) & (tod <= config.SESSION_END)
    if (~in_session).any():
        rep.special_session_rows_dropped = int((~in_session).sum())
        rep.special_session_dates = sorted(
            {str(d) for d in df.loc[~in_session, "timestamp"].dt.date}
        )
    return df.loc[in_session].copy()


def _audit_integrity(df: pd.DataFrame, rep: QualityReport) -> None:
    """Record OHLC sanity, nulls, session completeness and intraday holes."""
    o, h, l, c = (df[k] for k in config.OHLC_COLS)

    rep.ohlc_violations = int(
        ((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).sum()
    )
    rep.non_positive_prices = int((df[config.OHLC_COLS] <= 0).to_numpy().sum())

    nulls = df.isna().sum()
    rep.nulls = {k: int(v) for k, v in nulls.items() if v}

    per_session = df.groupby(df["timestamp"].dt.date).size()
    rep.sessions = int(per_session.size)
    rep.short_sessions = {
        str(d): int(n) for d, n in per_session.items() if n < config.BARS_PER_SESSION
    }
    rep.intraday_gaps = int(
        (config.BARS_PER_SESSION * per_session.size) - per_session.sum()
    )

    ret = c.pct_change()
    rep.max_abs_bar_return_bps = float(ret.abs().max() * 1e4)


def clean(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, QualityReport]:
    """Full step-1 cleaning pass. Returns the clean frame and its audit."""
    rep = QualityReport(split=split, rows_in=len(df))

    df = df.sort_values("timestamp", kind="stable")   # stable => copy order preserved
    df = _drop_duplicate_timestamps(df, rep)
    df = _drop_special_sessions(df, rep)
    df = df.reset_index(drop=True)

    # Session bookkeeping used by every later stage (no feature may cross a
    # session boundary without being told about it).
    df["session_date"] = df["timestamp"].dt.date.astype("string")
    df["bar_of_day"] = df.groupby("session_date").cumcount().astype("int16")
    df["is_session_open"] = df["bar_of_day"].eq(0)
    # The final row has no successor to compare against, but it *is* a session
    # close — fill before inverting so the column stays a plain bool.
    next_same_day = df["session_date"].eq(df["session_date"].shift(-1)).fillna(False)
    df["is_session_close"] = (~next_same_day.astype(bool)).astype(bool)

    _audit_integrity(df, rep)

    rep.rows_out = len(df)
    rep.first_ts = df["timestamp"].iloc[0]
    rep.last_ts = df["timestamp"].iloc[-1]
    return df, rep


# --------------------------------------------------------------------------- #
# Split-level guarantees
# --------------------------------------------------------------------------- #
def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """The test set must start strictly after the last training bar."""
    t_end = train["timestamp"].max()
    s_start = test["timestamp"].min()
    if s_start <= t_end:
        raise AssertionError(
            f"Chronological split violated: test starts {s_start} <= train ends {t_end}"
        )
    overlap = set(train["timestamp"]) & set(test["timestamp"])
    if overlap:
        raise AssertionError(f"{len(overlap)} timestamps appear in both splits")


def load_clean(split: str) -> pd.DataFrame:
    """Convenience reader for downstream steps."""
    path = config.INTERIM_DIR / f"nifty50_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/01_prepare_data.py` first."
        )
    return pd.read_parquet(path)
