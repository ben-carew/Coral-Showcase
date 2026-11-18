"""Sanity checks that catch the silent failures, before they become plots.

The expensive mistakes in this project have not been wrong formulas, they have
been *correctly computed comparisons between unrelated things*: an arm scored
against a truth file from a different sample, an event-id stride that aliases
two events onto one id, a fiTQun file whose t0 convention was never offset.
Each produced a full set of plausible-looking plots.

Every check returns a `Finding` rather than raising, because a report is more
useful with a loud warning in it than not produced at all. `severity` is
'error' for something that makes the numbers meaningless and 'warning' for
something that merely deserves a look.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: A vertex residual larger than this means the arm and the truth are almost
#: certainly not describing the same events: the detector is only ~33 m across,
#: so a median error of 10 m is not a bad fit, it is a different event.
IMPLAUSIBLE_VERTEX_MEDIAN_CM = 1000.0

#: Below this overlap the two tables are probably keyed differently.
MIN_EVENT_OVERLAP_FRACTION = 0.5


@dataclass
class Finding:
    check: str
    severity: str          # 'error' | 'warning' | 'ok'
    message: str
    detail: dict = None

    def __str__(self):
        mark = {"error": "ERROR", "warning": "WARN ", "ok": "ok   "}[self.severity]
        return f"[{mark}] {self.check}: {self.message}"


def check_event_overlap(reco_events, truth_events, *, label="") -> Finding:
    reco = set(np.asarray(reco_events).tolist())
    truth = set(np.asarray(truth_events).tolist())
    if not reco or not truth:
        return Finding("event-overlap", "error",
                       f"{label}: one side has no events "
                       f"(reco {len(reco)}, truth {len(truth)})")
    overlap = len(reco & truth)
    fraction = overlap / len(reco)
    detail = {"n_reco": len(reco), "n_truth": len(truth), "n_overlap": overlap,
              "fraction": fraction}
    if overlap == 0:
        return Finding("event-overlap", "error",
                       f"{label}: no event id appears in both the reconstruction "
                       f"and the truth. Wrong truth file, or a mismatched event-id "
                       f"stride (see productions.py).", detail)
    if fraction < MIN_EVENT_OVERLAP_FRACTION:
        return Finding("event-overlap", "warning",
                       f"{label}: only {fraction:.0%} of reconstructed events "
                       f"have truth ({overlap}/{len(reco)}). One job reads one "
                       f"input file, so a run spans as many files as it has "
                       f"jobs -- build truth over that many files, not over "
                       f"events/events_per_file.", detail)
    return Finding("event-overlap", "ok",
                   f"{label}: {overlap} events matched ({fraction:.0%}).", detail)


def check_truth_pairing(residuals: pd.DataFrame, *, label="") -> Finding:
    """A vertex residual far larger than the detector means wrong truth."""
    if residuals.empty or "vertex_dist" not in residuals:
        return Finding("truth-pairing", "warning", f"{label}: no residuals to check")
    median = float(np.nanmedian(residuals["vertex_dist"].to_numpy(float)))
    detail = {"median_vertex_dist_cm": median}
    if not np.isfinite(median):
        return Finding("truth-pairing", "error",
                       f"{label}: vertex residual is not finite.", detail)
    if median > IMPLAUSIBLE_VERTEX_MEDIAN_CM:
        return Finding("truth-pairing", "error",
                       f"{label}: median vertex residual {median:.0f} cm is larger "
                       f"than any real reconstruction error -- the arm is almost "
                       f"certainly matched against the wrong truth sample.", detail)
    return Finding("truth-pairing", "ok",
                   f"{label}: median vertex residual {median:.1f} cm.", detail)


def check_time_pedestal(residuals_by_arm: dict, tolerance_ns: float = 5.0) -> Finding:
    """Arms should share a time pedestal; a lone outlier has a wrong t0 offset."""
    medians = {label: float(np.nanmedian(r["t"].to_numpy(float)))
               for label, r in residuals_by_arm.items()
               if not r.empty and "t" in r}
    if len(medians) < 2:
        return Finding("time-pedestal", "ok", "fewer than two arms to compare",
                       medians)
    spread = max(medians.values()) - min(medians.values())
    if spread > tolerance_ns:
        return Finding("time-pedestal", "warning",
                       f"arms disagree on the time pedestal by {spread:.1f} ns "
                       f"({medians}). Check each arm's t0 convention -- both "
                       f"codes should report trigger-relative time.",
                       medians)
    return Finding("time-pedestal", "ok",
                   f"arms agree on the time pedestal to {spread:.1f} ns.", medians)


def check_duplicate_keys(table: pd.DataFrame, keys, *, name="") -> Finding:
    if table.empty:
        return Finding("duplicate-keys", "ok", f"{name}: empty")
    present = [k for k in keys if k in table.columns]
    duplicates = len(table) - len(table.drop_duplicates(present))
    if duplicates:
        return Finding("duplicate-keys", "error",
                       f"{name}: {duplicates} rows share a {present} key, so any "
                       f"join against this table is ambiguous.",
                       {"n_duplicate": duplicates})
    return Finding("duplicate-keys", "ok", f"{name}: keys unique on {present}.")


def check_success_rate(fits: pd.DataFrame, *, label="",
                       minimum: float = 0.5) -> Finding:
    if fits.empty:
        return Finding("success-rate", "error", f"{label}: no fits at all")
    rate = float(fits["success"].mean())
    detail = {"success_rate": rate, "n": len(fits)}
    if rate < minimum:
        return Finding("success-rate", "warning",
                       f"{label}: only {rate:.0%} of fits converged.", detail)
    return Finding("success-rate", "ok", f"{label}: {rate:.0%} converged.", detail)


def check_hypothesis_kinematics(arm) -> Finding:
    """A fiTQun merge made before 2026-11-14 holds ONE fit's kinematics per event.

    Its e- and mu- rows then share the vertex, direction and momentum of the
    sample's own hypothesis (the muon fit on a muon sample), so anything that
    mixes hypotheses -- PID against the electron momentum, an e- vs mu- fit
    comparison -- silently uses the wrong fit. The NLLs are per hypothesis in
    both formats, so the flat-cut PID numbers are unaffected.
    """
    from ..io import fitqun as fitqun_io

    meta = arm.meta or {}
    if meta.get("format") != "fitqun-merged":
        return Finding("hypothesis-kinematics", "ok",
                       f"{arm.label}: each hypothesis has its own fit.")
    if fitqun_io.merged_has_both_hypotheses(meta.get("path", "")):
        return Finding("hypothesis-kinematics", "ok",
                       f"{arm.label}: the merge carries both hypotheses' kinematics.")
    return Finding("hypothesis-kinematics", "warning",
                   f"{arm.label}: old fiTQun merge -- the e- and mu- rows share ONE "
                   f"fit's kinematics (the sample's own hypothesis). Resolution of "
                   f"that hypothesis and flat-cut PID are right; anything using the "
                   f"other hypothesis's vertex or momentum is not. Re-merge the run "
                   f"(exp.py merge) to get both.")


#: A fit this far from its truth vertex is not a bad fit of that event.
MISJOIN_FAR_CM = 500.0
#: ...and this close to a NEIGHBOURING event's truth vertex, it is that event's fit.
MISJOIN_NEAR_CM = 150.0
#: Above this share of shifted events the numbering of the two tables differs.
MISJOIN_FRACTION = 0.001


def check_event_numbering(fits: pd.DataFrame, truth_primary: pd.DataFrame, *,
                          label="") -> Finding:
    """Do the fits and the truth number their events the same way?

    Coral numbers an event by the WCSim header EvtNum, the truth builder and
    fiTQun's merge by the tree entry. In a WCSim file that skipped an event
    number the two differ by one from the gap onwards, and every later fit is
    scored against the NEXT event's truth: on the 100k tech-note electrons that
    was 0.55% of events, read for weeks as fit failures (1.32% against fiTQun's
    0.80%; really 0.79%). The signature is unambiguous and needs no production
    registry: a fit far from its own truth vertex and right on a neighbour's.
    """
    need = {"event", "x", "y", "z"}
    if fits.empty or truth_primary.empty or not need <= set(fits.columns):
        return Finding("event-numbering", "ok", f"{label}: nothing to check")
    f = fits.drop_duplicates("event").set_index("event")[["x", "y", "z"]]
    t = truth_primary.drop_duplicates("event").set_index("event")[["x", "y", "z"]]
    events = f.index.intersection(t.index)
    if len(events) < 100:
        return Finding("event-numbering", "ok", f"{label}: too few events to check")
    here = f.loc[events].to_numpy(float)
    own = np.linalg.norm(here - t.loc[events].to_numpy(float), axis=1)
    far = own > MISJOIN_FAR_CM
    shifted = {}
    for shift in (-2, -1, 1, 2):
        other = t.reindex(events + shift).to_numpy(float)
        near = np.linalg.norm(here - other, axis=1) < MISJOIN_NEAR_CM
        shifted[shift] = int(np.sum(far & near))
    worst = max(shifted, key=shifted.get)
    n = shifted[worst]
    detail = {"n_events": int(len(events)), "n_far": int(far.sum()), "shifted": shifted}
    if n > max(5, MISJOIN_FRACTION * len(events)):
        ids = events[far & (np.linalg.norm(here - t.reindex(events + worst).to_numpy(float), axis=1)
                            < MISJOIN_NEAR_CM)]
        return Finding("event-numbering", "error",
                       f"{label}: {n} fits ({100.0 * n / len(events):.2f}%) are > "
                       f"{MISJOIN_FAR_CM / 100:.0f} m from their own truth and on the truth of "
                       f"event {worst:+d} (first ids: {list(ids[:5])}). The two tables number "
                       f"events differently in some input files (WCSim EvtNum against tree "
                       f"entry): failure rates and tails against truth are wrong until the "
                       f"arm is renumbered (Analysis/Scripts/evtnum_gap_map.py).", detail)
    return Finding("event-numbering", "ok",
                   f"{label}: fits and truth number events the same way "
                   f"({int(far.sum())} fits > {MISJOIN_FAR_CM / 100:.0f} m from truth, "
                   f"none on a neighbour's).", detail)


def run_all(arms, truth, residuals_by_arm=None) -> list:
    """Every check, over a set of arms and their truth."""
    findings = []
    truth_events = truth.events["event"].to_numpy() if truth is not None else []
    for arm in arms:
        label = arm.label
        findings.append(check_success_rate(arm.fits, label=label))
        if getattr(arm, "algorithm", "") == "fitqun":
            findings.append(check_hypothesis_kinematics(arm))
        if len(truth_events):
            findings.append(check_event_overlap(
                arm.fits["event"].unique(), truth_events, label=label))
        findings.append(check_duplicate_keys(
            arm.fits, ["event", "sub_event", "hypothesis", "ring", "stage"],
            name=f"{label} fits"))
        if truth is not None and hasattr(truth, "primary"):
            first = arm.fits[arm.fits["sub_event"] == 0] if "sub_event" in arm.fits else arm.fits
            findings.append(check_event_numbering(first, truth.primary, label=label))
        if not arm.stages.empty:
            findings.append(check_duplicate_keys(
                arm.stages, ["event", "sub_event", "hypothesis", "stage"],
                name=f"{label} stages"))
    if residuals_by_arm:
        for label, residuals in residuals_by_arm.items():
            findings.append(check_truth_pairing(residuals, label=label))
        findings.append(check_time_pedestal(residuals_by_arm))
    return findings


def to_frame(findings) -> pd.DataFrame:
    return pd.DataFrame([{"check": f.check, "severity": f.severity,
                          "message": f.message} for f in findings])


def worst_severity(findings) -> str:
    order = {"ok": 0, "warning": 1, "error": 2}
    return max((f.severity for f in findings), key=lambda s: order[s],
               default="ok")
