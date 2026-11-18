"""Residuals of a reconstruction against MC truth.

Residual convention throughout Sextant: **reconstructed minus true**. A positive
momentum residual means the fit over-estimated the momentum.

The per-parameter residuals (t, x, y, z, p) are the raw differences. The three
*summary* metrics are the ones that actually get quoted, because they are
rotationally sensible where the raw components are not:

    vertex_dist   |reco vertex - true vertex|, in cm
    vertex_long   the same displacement projected ON the true direction, signed
    vertex_trans  its magnitude perpendicular to the true direction
    angle         opening angle between reco and true direction, in radians
    p_res_pct     (p_reco - p_true) / p_true, in per cent

### Why the vertex is decomposed

`vertex_dist` is positive definite, so it cannot show a bias -- and both codes
carry a large one. Measured on 2979 v1.12.29 electrons, Coral's vertex sits
**10.3 cm upstream** along the track and fiTQun's 4.2 cm, growing to 16.8 cm and
9.6 cm respectively within 256 cm of the wall. None of that is visible in a
distance, and correcting it is worth 14.06 -> 12.14 cm on Coral.

The split is also the physically meaningful one. For a Cherenkov cone the vertex
time and the ALONG-TRACK vertex are exactly degenerate -- dT/ds =
(1/beta - n cos alpha)/c vanishes identically on the cone -- so the longitudinal
component is the weakly constrained direction and the transverse one is not.
Quoting a single distance averages a soft direction against two stiff ones.

The projection is onto the TRUE direction, so an angular error cannot leak into
the longitudinal component. (A fitter applying a correction at run time has only
its reconstructed direction; on this sample the two agree to ~1.4 deg and the
difference is negligible.)

The direction is reported **both** ways. The opening angle is the quantity to
quote: it is rotationally well behaved. The separate theta and phi differences
are reported alongside it for diagnosis -- they show *which way* a fit is wrong,
which the opening angle cannot -- but they carry a coordinate artefact that the
opening angle does not: phi wraps (handled here) and is degenerate near the
poles, so a large phi residual on a near-polar track can mean almost nothing.
Read them as diagnostics, not as performance numbers.

### The time pedestal

Both algorithms report time on the TRIGGER clock: WCSim writes each digit at
(photon time - trigger-0 date + 950 ns) and both codes remove the 950. The
generator starts the gun at Time = 0 on its own clock, so against raw truth
every fit sits at -(trigger date): 5 ns-quantised, vertex-dependent, ~-39 ns
median and tens of ns wide. That spread is trigger placement, not resolution
(measured 2026-09-25: digit time - true photon time = 950 - date0 + 0.8 ns).

When the truth file carries each event's trigger date (`truth_events.
trigger_time`), truth t is moved onto the trigger clock and the t residual is a
real measurement -- bias and width both. `compute` records which clock it used
in `attrs['t_clock']` ('trigger' or 'generator'). Only on the generator clock is
the median still removed (`needs_pedestal`), as a last resort that keeps the
shape and discards the bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.geometry import dwall as _dwall, towall as _towall

#: Momentum below which a *fractional* momentum residual stops being meaningful:
#: dp/p diverges as p_true -> 0, and a handful of near-threshold events can
#: dominate a bin mean by orders of magnitude. Absolute residuals are unaffected
#: and are always computed.
DEFAULT_P_MIN_MEV = 100.0

#: Per-parameter residuals. theta/phi are derived from the direction
#: vectors rather than stored, and phi is wrapped to (-pi, pi].
PARAMETERS = ("t", "x", "y", "z", "p", "theta", "phi")
PARAMETER_UNITS = {"t": "ns", "x": "cm", "y": "cm", "z": "cm", "p": "MeV/c",
                   "theta": "deg", "phi": "deg",
                   "vertex_dist": "cm", "vertex_long": "cm", "vertex_trans": "cm",
                   "angle": "rad", "angle_deg": "deg",
                   "p_res_pct": "%", "p_res": "MeV/c"}

#: The seven per-parameter residuals, and the three aggregate performance
#: measures. Reported on separate pages: the first set diagnoses *how* a fit is
#: wrong, the second is what gets quoted.
COMPONENT_QUANTITIES = ("t", "x", "y", "z", "theta", "phi", "p")
AGGREGATE_QUANTITIES = ("vertex_dist", "vertex_long", "vertex_trans",
                        "angle_deg", "p_res_pct")


def wrap_to_pi(angle):
    return (np.asarray(angle, float) + np.pi) % (2 * np.pi) - np.pi


def opening_angle(dx1, dy1, dz1, dx2, dy2, dz2):
    """Angle between two unit directions, in radians."""
    dot = (np.asarray(dx1, float) * np.asarray(dx2, float)
           + np.asarray(dy1, float) * np.asarray(dy2, float)
           + np.asarray(dz1, float) * np.asarray(dz2, float))
    return np.arccos(np.clip(dot, -1.0, 1.0))


def compute(reco: pd.DataFrame, truth_rings: pd.DataFrame, *,
            truth_events: pd.DataFrame | None = None,
            p_min_mev: float = DEFAULT_P_MIN_MEV) -> pd.DataFrame:
    """Join a per-event fit table to truth and return one residual row per event.

    `reco` is any table in the canonical `fits` schema reduced to one row per
    event (`Arm.hypothesis(...)`); `truth_rings` is one true ring per event
    (`Truth.primary`). Events present in only one of them are dropped, and the
    count of what survived is available as `attrs['n_matched']`.
    """
    if reco.empty or truth_rings.empty:
        return _empty_residuals()

    merged = reco.merge(truth_rings, on="event", suffixes=("_reco", "_true"),
                        validate="one_to_one")
    if merged.empty:
        return _empty_residuals()

    out = pd.DataFrame({"event": merged["event"].to_numpy()})
    for key in ("t", "x", "y", "z", "p"):
        out[key] = (merged[f"{key}_reco"].to_numpy(float)
                    - merged[f"{key}_true"].to_numpy(float))

    # theta/phi come from the direction vectors, in degrees to match the
    # opening angle they sit beside. phi is wrapped so a fit at +179 deg
    # against a truth at -179 reads as 2 deg, not 358.
    theta_reco, phi_reco = _spherical(merged, "reco")
    theta_true, phi_true = _spherical(merged, "true")
    out["theta"] = np.degrees(theta_reco - theta_true)
    out["phi"] = np.degrees(wrap_to_pi(phi_reco - phi_true))

    dx = merged["x_reco"].to_numpy(float) - merged["x_true"].to_numpy(float)
    dy = merged["y_reco"].to_numpy(float) - merged["y_true"].to_numpy(float)
    dz = merged["z_reco"].to_numpy(float) - merged["z_true"].to_numpy(float)
    out["vertex_dist"] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Longitudinal (signed, + = downstream of truth) and transverse (magnitude)
    # components of the same displacement, projected on the TRUE direction.
    ux = merged["dx_true"].to_numpy(float)
    uy = merged["dy_true"].to_numpy(float)
    uz = merged["dz_true"].to_numpy(float)
    norm = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy, uz = ux / norm, uy / norm, uz / norm
    out["vertex_long"] = dx * ux + dy * uy + dz * uz
    out["vertex_trans"] = np.sqrt(np.maximum(
        out["vertex_dist"].to_numpy() ** 2 - out["vertex_long"].to_numpy() ** 2, 0.0))

    out["angle"] = opening_angle(
        merged["dx_reco"], merged["dy_reco"], merged["dz_reco"],
        merged["dx_true"], merged["dy_true"], merged["dz_true"])
    out["angle_deg"] = np.degrees(out["angle"].to_numpy())

    p_true = merged["p_true"].to_numpy(float)
    out["p_res"] = out["p"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(p_true > 1e-6, out["p_res"].to_numpy() / p_true * 100.0, np.nan)
    if p_min_mev > 0:
        pct = np.where(p_true >= p_min_mev, pct, np.nan)
    out["p_res_pct"] = pct

    # Binning variables travel with the residuals so every "resolution against
    # X" plot is a groupby on this one table rather than another join.
    out["p_true"] = p_true
    out["x_true"] = merged["x_true"].to_numpy(float)
    out["y_true"] = merged["y_true"].to_numpy(float)
    out["z_true"] = merged["z_true"].to_numpy(float)
    out["dwall"] = _dwall(out["x_true"], out["y_true"], out["z_true"])
    out["towall"] = _towall(out["x_true"], out["y_true"], out["z_true"],
                            merged["dx_true"], merged["dy_true"], merged["dz_true"])
    out["nll"] = merged["nll"].to_numpy(float) if "nll" in merged else np.nan

    if truth_events is not None and not truth_events.empty:
        extra = [c for c in ("nhits", "totq", "mode", "is_cc", "nu_energy",
                             "n_visible_rings", "n_sub_events")
                 if c in truth_events.columns]
        out = out.merge(truth_events[["event"] + extra], on="event", how="left")

    trig = (truth_events.set_index("event")["trigger_time"]
            if truth_events is not None and "trigger_time" in truth_events
            else None)
    on_trigger = (trig is not None and len(out) > 0
                  and bool(np.isfinite(trig.reindex(out["event"]).to_numpy(float)).all()))
    out.attrs["t_clock"] = "trigger" if on_trigger else "generator"
    out.attrs["n_reco"] = len(reco)
    out.attrs["n_truth"] = len(truth_rings)
    out.attrs["n_matched"] = len(out)
    return out


def _spherical(merged: pd.DataFrame, suffix: str):
    """(theta, phi) in radians from a direction vector column set."""
    dz = np.clip(merged[f"dz_{suffix}"].to_numpy(float), -1.0, 1.0)
    return (np.arccos(dz),
            np.arctan2(merged[f"dy_{suffix}"].to_numpy(float),
                       merged[f"dx_{suffix}"].to_numpy(float)))


def _empty_residuals() -> pd.DataFrame:
    cols = (["event"] + list(PARAMETERS)
            + ["vertex_dist", "vertex_long", "vertex_trans",
               "angle", "angle_deg", "p_res", "p_res_pct",
               "p_true", "x_true", "y_true", "z_true", "dwall", "towall", "nll"])
    return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})


def match_events(frames: dict) -> dict:
    """Restrict several residual tables to the events all of them have.

    Comparing arms on different event sets is the commonest way to get a
    difference that is really a sample difference. Every paired comparison in
    Sextant runs through here first.
    """
    non_empty = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not non_empty:
        return {k: v for k, v in frames.items()}
    common = None
    for table in non_empty.values():
        events = set(table["event"].to_numpy())
        common = events if common is None else (common & events)
    return {k: (v[v["event"].isin(common)].sort_values("event")
                 .reset_index(drop=True) if k in non_empty else v)
            for k, v in frames.items()}


#: The metrics a summary table reports, in the order it reports them.
#: Which statistical summary each quantity gets (convention set 2026-09-11).
#:
#: "one_sided" -- a magnitude that cannot go negative and has only an upper
#: tail. Summarised by a SINGLE number, the 68th percentile: the error 68% of
#: events come in under. A median and a width for a positive-definite quantity
#: says less than one honest quantile.
#:
#: "gaussian" -- a signed residual expected to be Gaussian about truth.
#: Summarised by a fitted mu (bias) and sigma (width), not by a mean and a
#: standard deviation, which a handful of mis-converged events move enormously
#: while leaving the core untouched.
#:
#: `t` is Gaussian *after its pedestal is removed* -- both codes sit ~-39 ns
#: against the truth Time by convention, which is not a bias in either (see
#: "The time pedestal" above). `vertex_long` is Gaussian because its MEAN is a
#: real, correctable quantity: it is what the along-track vertex bias
#: correction acts on.
QUANTITY_KIND = {
    "vertex_dist": "one_sided",
    "vertex_trans": "one_sided",
    "angle": "one_sided",
    "angle_deg": "one_sided",
    "vertex_long": "gaussian",
    "t": "gaussian",
    "x": "gaussian", "y": "gaussian", "z": "gaussian",
    "theta": "gaussian", "phi": "gaussian",
    "p": "gaussian", "p_res": "gaussian", "p_res_pct": "gaussian",
}

#: Quantities whose Gaussian fit must run on the pedestal-subtracted values.
PEDESTAL_QUANTITIES = ("t",)


def kind_of(quantity: str) -> str:
    """"one_sided" or "gaussian" for a residual column; defaults to gaussian."""
    return QUANTITY_KIND.get(quantity, "gaussian")


#: The statistics each family reports, in the order a table shows them.
KIND_STATISTICS = {
    "one_sided": ("n", "p68"),
    "gaussian": ("n", "gauss_mu", "gauss_sigma"),
}
KIND_CI = {"one_sided": ("p68",), "gaussian": ("gauss_mu", "gauss_sigma")}


#: The quantities the headline table reports, in order. A one-sided magnitude
#: contributes ONE number (its 68th percentile), a signed residual TWO (its
#: fitted bias and width) -- nine in total. The resolution pages profile the
#: same nine against a binning variable, reading this one definition, so a
#: binned page and the headline table cannot drift apart.
HEADLINE_METRICS = (
    ("vertex_dist", "vertex", "cm"),
    ("vertex_long", "vertex along track", "cm"),
    ("vertex_trans", "vertex transverse", "cm"),
    ("angle_deg", "angle", "deg"),
    ("t", "time", "ns"),
    ("p_res_pct", "momentum", "%"),
)

#: How each statistic is named in a column heading and on a panel.
STATISTIC_WORD = {"p68": "p68", "gauss_mu": "bias", "gauss_sigma": "sigma"}


def headline_panels() -> tuple:
    """(column, statistic, name, unit) for each of the nine headline numbers."""
    out = []
    for key, name, unit in HEADLINE_METRICS:
        for statistic in KIND_STATISTICS[kind_of(key)][1:]:
            out.append((key, statistic, name, unit))
    return tuple(out)


SUMMARY_METRICS = (
    ("vertex_dist", "vertex distance", "cm"),
    ("vertex_long", "vertex bias along track", "cm"),
    ("vertex_trans", "vertex distance, transverse", "cm"),
    ("angle_deg", "angular error", "deg"),
    ("p_res_pct", "momentum residual", "%"),
    ("p_res", "momentum residual", "MeV/c"),
    ("t", "time residual", "ns"),
)


def summarise(residuals: pd.DataFrame, label: str = "", *, seed: int = 0) -> pd.DataFrame:
    """The standard per-arm summary table, with bootstrap CIs.

    Each quantity is summarised by the convention its shape deserves -- a single
    68th percentile for the one-sided magnitudes, a fitted mu and sigma for the
    signed ones. See QUANTITY_KIND. The legacy median/iqr68 columns are kept
    alongside so earlier campaign numbers stay comparable.
    """
    from . import stats

    rows = []
    for key, name, unit in SUMMARY_METRICS:
        if key not in residuals.columns:
            continue
        values = residuals[key]
        kind = kind_of(key)
        if key in PEDESTAL_QUANTITIES and needs_pedestal(residuals):
            # Generator-clock truth: the offset is trigger placement, not a
            # bias, so the Gaussian fit describes the recentred residual.
            values = values - stats.median(values)
        row = stats.summarise(values, name=name, seed=seed,
                              statistics=("n", "median", "iqr68")
                                         + KIND_STATISTICS[kind][1:],
                              ci_for=KIND_CI[kind])
        row.update({"run": label, "quantity": key, "unit": unit, "kind": kind})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    lead = ["run", "quantity", "metric", "unit", "kind", "n",
            "p68", "gauss_mu", "gauss_sigma", "median", "iqr68"]
    return table[[c for c in lead if c in table.columns]
                 + [c for c in table.columns if c not in lead]]


def needs_pedestal(residuals: pd.DataFrame) -> bool:
    """True unless the t residual was measured against trigger-clock truth."""
    return residuals.attrs.get("t_clock") != "trigger"


def subtract_pedestal(residuals: pd.DataFrame, column: str = "t",
                      pedestal: float | None = None) -> pd.DataFrame:
    """Recentre a residual column on zero, returning the offset applied.

    With `pedestal=None` the arm's own median is removed, which is the right
    thing for the time residual: it discards an offset that is a convention and
    keeps the spread, which is the physics. Pass an explicit value to remove the
    *same* offset from several arms and preserve the differences between them.
    """
    out = residuals.copy()
    offset = float(np.nanmedian(out[column].to_numpy(float))) if pedestal is None \
        else float(pedestal)
    out[column] = out[column].to_numpy(float) - offset
    out.attrs = dict(residuals.attrs)
    out.attrs[f"{column}_pedestal"] = offset
    return out
