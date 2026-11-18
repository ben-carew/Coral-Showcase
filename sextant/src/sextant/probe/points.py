"""Choosing where to evaluate the algorithms' internals.

A parameter point is one 7-vector `(t, x, y, z, theta, phi, p)` for one
(event, sub_event, hypothesis), plus a name saying where it came from.

The names matter more than they look. Comparing two codes' predicted charges at
each code's *own* fitted vertex answers "do these two different events look
different", which is not a question anyone asked. The comparison that isolates
the charge models is at a **common** point:

    truth   both codes evaluated at the MC truth parameters. The cleanest
            comparison of the models themselves, and the only point at which
            "which code predicts the observed light better" is well posed.
    ref     both codes evaluated at a chosen reference arm's fitted parameters.
            Answers "given fiTQun's answer, what would Coral have predicted
            there" -- which separates a bad model from a bad minimisation.
    self    each code at its own fit. Useful for the likelihood decomposition
            (where the value at the converged point is the thing of interest),
            never for a cross-code per-PMT charge comparison.

`build` refuses to produce a `self` point set shared between two arms, because
that combination is the mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.schema import POINT_REF, POINT_SELF, POINT_TRUTH, STAGE_FINAL

POINT_COLUMNS = ["event", "sub_event", "hypothesis", "point",
                 "t", "x", "y", "z", "theta", "phi", "p"]


def _angles(dx, dy, dz):
    theta = np.arccos(np.clip(np.asarray(dz, float), -1.0, 1.0))
    phi = np.arctan2(np.asarray(dy, float), np.asarray(dx, float))
    return theta, phi


def from_truth(truth_rings: pd.DataFrame, *, hypotheses=("e-", "mu-"),
               sub_event: int = 0) -> pd.DataFrame:
    """Parameter points at the MC truth of each event's primary ring."""
    if truth_rings.empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    theta, phi = _angles(truth_rings["dx"], truth_rings["dy"], truth_rings["dz"])
    base = pd.DataFrame({
        "event": truth_rings["event"].to_numpy(), "sub_event": sub_event,
        "point": POINT_TRUTH,
        "t": truth_rings["t"].to_numpy(float),
        "x": truth_rings["x"].to_numpy(float),
        "y": truth_rings["y"].to_numpy(float),
        "z": truth_rings["z"].to_numpy(float),
        "theta": theta, "phi": phi,
        "p": truth_rings["p"].to_numpy(float),
    })
    # One copy per hypothesis: the truth parameters are the same, but each
    # hypothesis predicts different light from them, which is the comparison.
    return pd.concat([base.assign(hypothesis=h) for h in hypotheses],
                     ignore_index=True)[POINT_COLUMNS]


def from_fits(fits: pd.DataFrame, *, point_name: str = POINT_SELF,
              hypotheses=None, sub_event: int | None = 0,
              only_success: bool = True) -> pd.DataFrame:
    """Parameter points at an arm's fitted parameters."""
    if fits.empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    df = fits[fits["stage"] == STAGE_FINAL]
    if sub_event is not None:
        df = df[df["sub_event"] == sub_event]
    if only_success:
        df = df[df["success"]]
    if hypotheses is not None:
        df = df[df["hypothesis"].isin(list(hypotheses))]
    if df.empty:
        return pd.DataFrame(columns=POINT_COLUMNS)
    theta, phi = _angles(df["dx"], df["dy"], df["dz"])
    return pd.DataFrame({
        "event": df["event"].to_numpy(), "sub_event": df["sub_event"].to_numpy(),
        "hypothesis": df["hypothesis"].to_numpy(), "point": point_name,
        "t": df["t"].to_numpy(float), "x": df["x"].to_numpy(float),
        "y": df["y"].to_numpy(float), "z": df["z"].to_numpy(float),
        "theta": theta, "phi": phi, "p": df["p"].to_numpy(float),
    })[POINT_COLUMNS]


def time_pedestal(fits, truth_rings) -> float:
    """The constant offset between an algorithm's time origin and truth's.

    Generators start the primary at t=0; a reconstruction measures the vertex
    time against the detector's trigger, so the two live on origins separated
    by a constant. Evaluating a likelihood at the truth point without adding it
    back puts the time PDF at the wrong place by tens of nanoseconds, which
    swamps every other term and makes the truth point look absurdly bad.
    """
    a = fits[["event", "t"]].drop_duplicates("event")
    b = truth_rings[["event", "t"]].drop_duplicates("event")
    m = a.merge(b, on="event", suffixes=("_fit", "_true"))
    if m.empty:
        raise ValueError("no events in common, cannot measure a time pedestal")
    return float(np.median(m["t_fit"] - m["t_true"]))


def build(truth=None, reference_arm=None, *, hypotheses=("e-", "mu-"),
          events=None, points=(POINT_TRUTH, POINT_REF),
          time_offset: float = 0.0) -> pd.DataFrame:
    """Assemble the point set a probe run will evaluate.

    `events` restricts to an explicit event list, which it almost always should:
    a per-PMT table is ~50k rows per event per hypothesis, so this is a
    diagnostic run over tens of events, not a production pass.

    `time_offset` is added to the truth vertex time, putting the truth point on
    the algorithm's time origin (see `time_pedestal`).
    """
    frames = []
    if POINT_TRUTH in points:
        if truth is None:
            raise ValueError("point set includes 'truth' but no truth was given")
        truth_point = from_truth(truth.primary, hypotheses=hypotheses)
        if time_offset:
            truth_point = truth_point.assign(t=truth_point["t"] + time_offset)
        frames.append(truth_point)
    if POINT_REF in points:
        if reference_arm is None:
            raise ValueError("point set includes 'ref' but no reference arm "
                             "was given")
        frames.append(from_fits(reference_arm.fits, point_name=POINT_REF,
                                hypotheses=hypotheses))
    if POINT_SELF in points:
        raise ValueError(
            "'self' cannot be part of a shared point set: each arm's own fit is "
            "a different point, so a per-PMT comparison across arms at 'self' "
            "measures the difference between two vertices, not between two "
            "charge models. Use 'truth' or 'ref'.")
    if not frames:
        return pd.DataFrame(columns=POINT_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    if events is not None:
        table = table[table["event"].isin(list(events))]
    return table.reset_index(drop=True)


def select_events(truth, n: int = 20, *, seed: int = 0, require_visible=True,
                  within=None):
    """A reproducible sample of events to probe.

    Seeded, because a diagnostic that looks at different events every time it
    runs cannot be used to check whether a change helped.

    `within` restricts the draw to a set of event ids -- normally the events a
    run actually reconstructed. Comparing a probe against a run's own fitted
    results is only meaningful on events both of them saw, and a blind draw
    from a large production will miss a run covering the first few hundred
    files almost entirely.
    """
    events = truth.events
    if require_visible and "n_visible_rings" in events.columns:
        events = events[events["n_visible_rings"] > 0]
    ids = np.sort(events["event"].to_numpy())
    if within is not None:
        keep = np.asarray(sorted(set(within)), dtype=ids.dtype)
        ids = np.intersect1d(ids, keep)
        if len(ids) == 0:
            raise ValueError(
                "no truth events overlap the requested event set; the probe "
                "and the run are looking at different events")
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=n, replace=False))
