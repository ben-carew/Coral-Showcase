"""Multi-ring metrics: how many rings, which ones, and how well fitted.

Comparing a multi-ring fit to truth means solving an assignment problem first:
reconstructed ring 2 is not necessarily true ring 2. Rings are matched by
minimum total opening angle (a 1-1 Hungarian assignment), which is what makes
"the momentum residual of the second ring" a well-defined quantity.

Counting conventions, all of which change the numbers if got wrong:

* only **visible** true rings count (above Cherenkov threshold) -- a
  sub-threshold proton is not something any fitter can find, and counting it as
  missed would penalise every algorithm equally and uninformatively;
* only the **primary sub-event** is used by default; later sub-events are
  muon-decay electrons, which have no true ring in the primary record;
* a reconstructed ring with no true partner is **fake**, a true ring with no
  reconstructed partner is **missed**, and the counts are reported per event
  rather than pooled so an event with 6 fakes cannot hide among 100 good ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.schema import STAGE_FINAL
from .residuals import opening_angle

DEFAULT_MAX_RINGS = 6

#: Matched pairs further apart than this are treated as not matched at all: a
#: "match" at 70 degrees is the assignment algorithm pairing two unrelated
#: rings because it had to pair something.
MAX_MATCH_ANGLE_DEG = 60.0


def _assignment(cost: np.ndarray):
    """1-1 assignment minimising total cost; falls back to greedy without scipy."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:                                # pragma: no cover
        rows, cols = [], []
        used_r, used_c = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            if r in used_r or c in used_c:
                continue
            rows.append(r); cols.append(c)
            used_r.add(r); used_c.add(c)
        return np.array(rows), np.array(cols)


def match_rings(reco: pd.DataFrame, true: pd.DataFrame,
                max_angle_deg: float = MAX_MATCH_ANGLE_DEG) -> pd.DataFrame:
    """Match one event's reconstructed rings to its true rings."""
    if reco.empty or true.empty:
        return pd.DataFrame(columns=["reco_index", "true_index", "angle_deg"])
    r = reco[["dx", "dy", "dz"]].to_numpy(float)
    t = true[["dx", "dy", "dz"]].to_numpy(float)
    cost = np.degrees(np.arccos(np.clip(r @ t.T, -1.0, 1.0)))
    cost = np.nan_to_num(cost, nan=180.0)
    ri, ti = _assignment(cost)
    angles = cost[ri, ti]
    keep = angles <= max_angle_deg
    return pd.DataFrame({"reco_index": ri[keep], "true_index": ti[keep],
                         "angle_deg": angles[keep]})


def build_tables(reco_rings: pd.DataFrame, truth_rings: pd.DataFrame, *,
                 sub_event: int | None = 0, visible_only: bool = True,
                 max_rings: int = DEFAULT_MAX_RINGS,
                 max_angle_deg: float = MAX_MATCH_ANGLE_DEG) -> tuple:
    """(per_ring, per_event) tables for one arm's multi-ring output."""
    rings = reco_rings[reco_rings["stage"] == STAGE_FINAL]
    if sub_event is not None and not rings.empty:
        rings = rings[rings["sub_event"] == sub_event]
    truth = truth_rings[truth_rings["visible"]] if visible_only else truth_rings
    if rings.empty or truth.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rank reconstructed rings by momentum so slot 1 is always the leading ring.
    rings = rings.sort_values(["event", "p"], ascending=[True, False])
    truth = truth.sort_values(["event", "p"], ascending=[True, False])

    ring_rows, event_rows = [], []
    truth_by_event = dict(tuple(truth.groupby("event")))

    for event, reco_event in rings.groupby("event"):
        true_event = truth_by_event.get(event)
        if true_event is None:
            continue
        reco_event = reco_event.reset_index(drop=True)
        true_event = true_event.reset_index(drop=True)
        matches = match_rings(reco_event, true_event, max_angle_deg)
        matched_reco = dict(zip(matches["reco_index"], matches["true_index"]))
        matched_angle = dict(zip(matches["reco_index"], matches["angle_deg"]))

        for slot, (index, row) in enumerate(reco_event.iterrows(), start=1):
            true_index = matched_reco.get(index)
            matched = true_index is not None
            true_row = true_event.iloc[true_index] if matched else None
            ring_rows.append({
                "event": event, "slot": slot, "ring": int(row["ring"]),
                "n_reco": len(reco_event), "n_true": len(true_event),
                "hypothesis": row["hypothesis"], "p": row["p"],
                "dconv": row["dconv"], "nll": row["nll"],
                "matched": matched,
                "angle_deg": matched_angle.get(index, np.nan),
                "p_true": true_row["p"] if matched else np.nan,
                "particle_true": true_row["particle"] if matched else None,
                "dp_pct": (100.0 * (row["p"] - true_row["p"]) / true_row["p"]
                           if matched and true_row["p"] > 0 else np.nan),
                "type_correct": (bool(row["hypothesis"] == true_row["particle"])
                                 if matched else False),
            })

        n_reco, n_true = len(reco_event), len(true_event)
        lead = reco_event.iloc[0]
        event_rows.append({
            "event": event, "n_reco": n_reco, "n_true": n_true,
            "n_matched": len(matches),
            "n_fake": n_reco - len(matches), "n_missed": n_true - len(matches),
            "exact_count": bool(n_reco == n_true),
            "count_error": n_reco - n_true,
            "nll": lead["nll"], "t": lead["t"],
            "x": lead["x"], "y": lead["y"], "z": lead["z"],
        })

    per_ring = pd.DataFrame(ring_rows)
    per_event = pd.DataFrame(event_rows)
    if not per_event.empty:
        per_event["n_reco_clipped"] = per_event["n_reco"].clip(1, max_rings)
        per_event["n_true_clipped"] = per_event["n_true"].clip(1, max_rings)
    return per_ring, per_event


def counting_summary(per_event: pd.DataFrame, label: str = "") -> pd.DataFrame:
    if per_event.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "arm": label, "events": len(per_event),
        "mean_n_reco": per_event["n_reco"].mean(),
        "mean_n_true": per_event["n_true"].mean(),
        "exact_count_pct": 100.0 * per_event["exact_count"].mean(),
        # Balanced accuracy: the mean of the per-true-count accuracies. Pooled
        # accuracy is dominated by whichever ring count the sample happens to be
        # full of, so a fitter that always says "1 ring" scores well on a
        # mostly-1-ring sample without being right about anything.
        "balanced_count_pct": 100.0 * (
            per_event.groupby("n_true")["exact_count"].mean().mean()),
        "fake_per_event": per_event["n_fake"].mean(),
        "missed_per_event": per_event["n_missed"].mean(),
    }])


def counting_confusion(per_event: pd.DataFrame,
                       max_rings: int = DEFAULT_MAX_RINGS) -> pd.DataFrame:
    if per_event.empty:
        return pd.DataFrame()
    return pd.crosstab(per_event["n_true_clipped"], per_event["n_reco_clipped"],
                       rownames=["n_true"], colnames=["n_reco"])


def ring_summary(per_ring: pd.DataFrame, label: str = "",
                 max_slots: int = DEFAULT_MAX_RINGS) -> pd.DataFrame:
    """Per-slot quality: match rate, angular error, momentum bias, PID accuracy."""
    if per_ring.empty:
        return pd.DataFrame()
    from . import stats

    rows = []
    for slot in range(1, max_slots + 1):
        sel = per_ring[per_ring["slot"] == slot]
        if sel.empty:
            continue
        matched = sel[sel["matched"]]
        rows.append({
            "arm": label, "slot": slot, "n": len(sel),
            "matched_pct": 100.0 * sel["matched"].mean(),
            "median_angle_deg": stats.median(matched["angle_deg"]),
            "median_dp_pct": stats.median(matched["dp_pct"]),
            "iqr68_dp_pct": stats.iqr68(matched["dp_pct"]),
            "type_correct_pct": (100.0 * matched["type_correct"].mean()
                                 if len(matched) else np.nan),
        })
    return pd.DataFrame(rows)


def invariant_mass(p1, d1, p2, d2) -> float:
    """Invariant mass of two massless rings -- the pi0 mass estimator."""
    cos = float(np.dot(np.asarray(d1, float), np.asarray(d2, float)))
    return float(np.sqrt(max(2.0 * p1 * p2 * (1.0 - cos), 0.0)))
