"""Step 4 (part 3) — scoring.

Accuracy on its own is not a result on this problem. The classes are close to
balanced (50.8% up on test) so accuracy looks respectable at 51%, and a
one-line persistence rule already reaches 61.4%. Every number here is therefore
reported next to those two baselines, and the headline figure is the **lift over
persistence**, not raw accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "classification_metrics",
    "directional_strategy",
    "pick_threshold",
    "regime_breakdown",
]


def pick_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximising accuracy, chosen on VALIDATION only.

    Applying 0.5 blindly is a mild but real handicap when the positive rate is
    not 50%. The chosen value is then frozen and applied to test — tuning it on
    test would be scoring on the answer key.
    """
    grid = np.linspace(0.30, 0.70, 81)
    scores = [(t, accuracy_score(y_true, (prob >= t).astype(int))) for t in grid]
    return float(max(scores, key=lambda kv: kv[1])[0])


def classification_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
    baselines: dict | None = None,
) -> dict:
    """Full classification report for one model on one split."""
    y_true = np.asarray(y_true).astype(int).ravel()
    prob = np.asarray(prob).astype("float64").ravel()
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        # MCC is the honest single number on a near-balanced binary problem: it
        # collapses to ~0 for any constant predictor, which accuracy does not.
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_up_rate": float(pred.mean()),
    }
    if baselines:
        out["baseline_majority"] = float(baselines["majority"])
        out["baseline_persistence"] = float(baselines["persistence"])
        out["lift_over_majority"] = out["accuracy"] - out["baseline_majority"]
        out["lift_over_persistence"] = out["accuracy"] - out["baseline_persistence"]
        out["beats_persistence"] = bool(out["accuracy"] > out["baseline_persistence"])
    return out


def regime_breakdown(
    y_true: np.ndarray,
    prob: np.ndarray,
    regimes: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Accuracy per HMM regime.

    A model can post a decent overall number while being useless in exactly the
    conditions that matter. The test window is calmer than train (5.9% high-vol
    vs 10.6%), so an overall figure flatters any model that struggles when
    volatility spikes.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    pred = (np.asarray(prob).ravel() >= threshold).astype(int)
    df = pd.DataFrame({"regime": regimes, "y": y_true, "pred": pred})

    rows = []
    for regime, g in df.groupby("regime", observed=True):
        rows.append({
            "regime": regime,
            "n": len(g),
            "share_pct": 100.0 * len(g) / len(df),
            "positive_rate": float(g["y"].mean()),
            "accuracy": float((g["y"] == g["pred"]).mean()),
            "mcc": float(matthews_corrcoef(g["y"], g["pred"]))
            if g["y"].nunique() > 1 and g["pred"].nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def cost_sensitivity(
    prob: np.ndarray,
    forward_return: np.ndarray,
    threshold: float = 0.5,
    cost_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0),
) -> pd.DataFrame:
    """Net return across a sweep of per-flip costs, plus the break-even cost.

    This is the number that decides whether a classification edge means
    anything. At 15-minute frequency the model changes position ~12 times a
    session, so cost enters roughly 3,000 times a year and the break-even is
    reached at a level well inside real Indian index-futures costs (spread +
    STT + exchange fee + GST + stamp duty).
    """
    prob = np.asarray(prob).ravel()
    fwd = np.asarray(forward_return).ravel()

    position = np.where(prob >= threshold, 1.0, -1.0)
    turnover = np.abs(np.diff(position, prepend=0.0)) / 2.0
    gross = float((position * fwd).sum())
    n_flips = float(turnover.sum())

    rows = [{"cost_bps": c,
             "net_return_pct": (gross - n_flips * c / 1e4) * 100,
             "profitable": gross - n_flips * c / 1e4 > 0}
            for c in cost_grid]
    out = pd.DataFrame(rows)
    out.attrs["break_even_bps"] = float(gross / n_flips * 1e4) if n_flips else np.nan
    out.attrs["gross_return_pct"] = gross * 100
    return out


def persistence_agreement(prob: np.ndarray, prev_return: np.ndarray,
                          threshold: float = 0.5) -> float:
    """Fraction of bars where the model's call equals the persistence rule's.

    High agreement means the network has largely rediscovered "keep going the
    same way" — worth knowing before claiming it learned something richer.
    """
    pred = (np.asarray(prob).ravel() >= threshold).astype(int)
    pers = (np.asarray(prev_return).ravel() > 0).astype(int)
    return float((pred == pers).mean())


def directional_strategy(
    prob: np.ndarray,
    forward_return: np.ndarray,
    threshold: float = 0.5,
    cost_bps: float = 1.0,
) -> dict:
    """A deliberately naive long/short strategy, as a reality check on accuracy.

    Long when p >= threshold, short otherwise, one unit, every bar. This is not
    a trading system — there is no sizing, no risk limit and no abstention — it
    exists to answer one question: does the classification edge survive contact
    with transaction costs?

    `cost_bps` is charged on every position *change*, not every bar. At 15-minute
    frequency a model that flips constantly pays this many times a day, which is
    usually what kills an apparently profitable signal.
    """
    prob = np.asarray(prob).ravel()
    fwd = np.asarray(forward_return).ravel()

    position = np.where(prob >= threshold, 1.0, -1.0)
    turnover = np.abs(np.diff(position, prepend=0.0)) / 2.0
    costs = turnover * cost_bps / 1e4

    gross = position * fwd
    net = gross - costs

    n_bars = len(net)
    bars_per_year = 25 * 250
    ann = bars_per_year / n_bars if n_bars else np.nan

    return {
        "gross_return_pct": float(gross.sum() * 100),
        "net_return_pct": float(net.sum() * 100),
        "cost_drag_pct": float(costs.sum() * 100),
        "hit_rate": float((gross > 0).mean()),
        "position_flips": int((turnover > 0).sum()),
        "flips_per_session": float((turnover > 0).sum() / (n_bars / 25)),
        "sharpe_annualised": float(net.mean() / net.std() * np.sqrt(bars_per_year))
        if net.std() > 0 else np.nan,
        "annualised_net_pct": float(net.sum() * ann * 100),
    }
