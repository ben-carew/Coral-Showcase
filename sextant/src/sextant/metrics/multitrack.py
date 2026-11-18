"""Multiple-track events: what the HK validation spec asks of them.

For events with more than one visible true track the spec asks for

    (b) the interaction vertex;
    (c) the most energetic particle's direction and momentum resolution;
    (d) the most energetic particle's PID efficiency;

each also against momentum, dwall and towall. Sub-eventing (a) is scored by
`metrics.subevents` on the same truth.

### What is compared with what

*Truth.* The **most energetic visible particle** -- by total energy, so a 500
MeV/c muon outranks a 450 MeV/c electron -- placed at the true **interaction
vertex** (`truth_events.vtx_*`), which is where every primary starts. Its
direction defines the along-track / transverse split of the vertex residual,
and towall is measured along it.

*Reconstruction.* The **leading ring** of the arm's final multi-ring fit
(highest reconstructed momentum, sub-event 0), with the fit's common vertex.
An arm without a multi-ring fit falls back to its single-ring fit, taking the
hypothesis with the lower NLL -- which is what a single-ring analysis of a
multi-track event would report. The leading ring is compared with the most
energetic true particle directly, not after a ring-to-ring assignment: an
analysis has only the leading ring, so choosing the wrong ring to lead IS part
of the resolution it inherits. (`metrics.rings` holds the per-ring view after a
1-1 assignment.)

*PID.* Correct when the leading ring's type equals the most energetic
particle's ring type ('e-' for e+-, gamma and pi0 photons; 'mu-' for mu+-),
reported per true type.

Residuals are built by `residuals.compute`, so every statistic, profile and
table is the single-track machinery, unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.schema import (HYP_ELECTRON, HYP_MUON, PDG_MASS_MEV, STAGE_FINAL)
from . import residuals as residual_metrics

#: Events need at least this many visible true tracks to count as multi-track.
MIN_TRACKS = 2

#: Too few multi-track events and the section is not produced.
MIN_EVENTS = 20

SOURCE_MULTIRING = "multi-ring fit"
SOURCE_SINGLE_RING = "single-ring fit (lower-NLL hypothesis)"


def most_energetic(truth, *, min_tracks: int = MIN_TRACKS) -> pd.DataFrame:
    """One true ring per multi-track event, at the interaction vertex."""
    rings = truth.rings[truth.rings["visible"]]
    if rings.empty:
        return rings
    counts = rings.groupby("event").size()
    rings = rings[rings["event"].isin(counts.index[counts >= min_tracks])]
    if rings.empty:
        return rings
    mass = np.array([PDG_MASS_MEV.get(abs(int(p)), 0.0) for p in rings["pdg"]])
    energy = np.sqrt(rings["p"].to_numpy(float) ** 2 + mass ** 2)
    lead = (rings.assign(_energy=energy)
                 .sort_values(["event", "_energy"], ascending=[True, False])
                 .drop_duplicates("event").drop(columns="_energy"))
    vertex = truth.events.set_index("event")[["vtx_t", "vtx_x", "vtx_y", "vtx_z"]]
    lead = lead.join(vertex, on="event")
    for axis in ("t", "x", "y", "z"):
        lead[axis] = lead[f"vtx_{axis}"].where(lead[f"vtx_{axis}"].notna(), lead[axis])
    return (lead.drop(columns=["vtx_t", "vtx_x", "vtx_y", "vtx_z"])
                .reset_index(drop=True))


def leading_reco(arm, *, sub_event: int = 0) -> tuple:
    """(one row per event: the leading reconstructed ring, which fit it came from)."""
    rings = arm.multiring
    if not rings.empty:
        rings = rings[(rings["stage"] == STAGE_FINAL) & (rings["sub_event"] == sub_event)]
    if not rings.empty:
        lead = (rings.sort_values(["event", "p"], ascending=[True, False])
                     .drop_duplicates("event"))
        return lead.reset_index(drop=True), SOURCE_MULTIRING
    fits = arm.fits
    fits = fits[(fits["stage"] == STAGE_FINAL) & (fits["sub_event"] == sub_event)
                & fits["hypothesis"].isin([HYP_ELECTRON, HYP_MUON])
                & np.isfinite(fits["nll"])]
    lead = (fits.sort_values(["event", "nll"]).drop_duplicates("event"))
    return lead.reset_index(drop=True), SOURCE_SINGLE_RING


def build(arms, truth, *, min_tracks: int = MIN_TRACKS,
          min_events: int = MIN_EVENTS) -> dict:
    """{arm label: residual table of the leading ring}, or {} when the sample
    has too few multi-track events for the section to mean anything.

    The residual table is `residuals.compute`'s, plus `particle_true`,
    `reco_type`, `pid_correct` and the reconstruction `source`.
    """
    true = most_energetic(truth, min_tracks=min_tracks)
    if len(true) < min_events:
        return {}
    out = {}
    for arm in arms:
        reco, source = leading_reco(arm)
        reco = reco[reco["event"].isin(true["event"])]
        if reco.empty:
            continue
        table = residual_metrics.compute(reco, true, truth_events=truth.events)
        types = (reco.set_index("event")["hypothesis"]
                 .reindex(table["event"]).to_numpy(object))
        particle = (true.set_index("event")["particle"]
                    .reindex(table["event"]).to_numpy(object))
        table["reco_type"] = types
        table["particle_true"] = particle
        table["pid_correct"] = types == particle
        table.attrs["source"] = source
        table.attrs["n_multitrack"] = len(true)
        out[arm.label] = table
    return out


def pid_summary(tables: dict) -> pd.DataFrame:
    """Leading-ring type efficiency per true type of the most energetic particle."""
    from . import stats

    rows = []
    for label, table in tables.items():
        for particle, group in table.groupby("particle_true"):
            if len(group) < stats.MIN_RATE_EVENTS:
                continue
            rows.append({"arm": label, "true type": particle,
                         "source": table.attrs.get("source", ""),
                         **stats.rate(group["pid_correct"].to_numpy())})
    return pd.DataFrame(rows)


def pid_classified(tables: dict) -> dict:
    """The `pid.efficiency_profiles` input: {arm: {true type: per-event table}}."""
    out = {}
    for label, table in tables.items():
        out[label] = {
            particle: pd.DataFrame({
                "event": group["event"], "correct": group["pid_correct"],
                "p_true": group["p_true"], "dwall": group["dwall"],
                "towall": group["towall"]})
            for particle, group in table.groupby("particle_true")
            if particle in (HYP_ELECTRON, HYP_MUON)}
    return out
