"""The bound as a comparison arm: what an efficient estimator would report.

For every truth-matched event the Cramer-Rao covariance C_i is computed at the TRUE
parameters. Residuals are then DRAWN from N(0, C_i) -- `n_draw` per event -- and put in
a table with the same columns as `metrics.residuals.compute`. Every statistic Sextant
reports (a 68th percentile of a vertex distance, a fitted Gaussian width of the
momentum residual, per bin of momentum or dwall) is then computed for the bound by the
SAME code as for Coral and fiTQun.

That matters because the quantities are not the covariance's diagonal. The 68th
percentile of |vertex| over a bin is a property of a MIXTURE of 3-D Gaussians of
different sizes and shapes; the opening angle is the length of a 2-D Gaussian vector.
Converting sigma to those by hand is where a bound quietly stops being comparable.
Drawing from the covariance and measuring the draws cannot get that wrong.

An efficient estimator is unbiased, so the bound's bias panels sit at zero by
construction and only the width panels are compared.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .fisher import bound

#: Above this many events the bound is computed on a seeded random subset: a few
#: hundred events per bin pin a 68th percentile to a few per cent, and the per-event
#: cost (~50 ms) would otherwise make a 100k-event report take over an hour.
MAX_EVENTS = 4000


def true_particle(truth_rings: pd.DataFrame):
    """The single particle type of a particle-gun sample, or None if mixed."""
    if truth_rings.empty or "particle" not in truth_rings:
        return None
    counts = truth_rings["particle"].value_counts()
    top = counts.index[0]
    return top if counts.iloc[0] >= 0.99 * counts.sum() else None


def event_bounds(model, residuals: pd.DataFrame, truth_rings: pd.DataFrame,
                 particle: str, *, timing="measured", shower=True, floor=True,
                 cov_key="cov", max_events=MAX_EVENTS, seed=0, log=print):
    """(table, covariances) for the events of `residuals`.

    table has one row per event: sigma_t, sigma_long, sigma_perp, sigma_angle
    [rad], sigma_p_rel and the true kinematics; covariances is (n, 7, 7).
    """
    light = model.light[particle]
    pmin = float(np.exp(light.logp_ctr[0]))
    events = residuals[["event"]].merge(
        truth_rings[["event", "x", "y", "z", "dx", "dy", "dz", "p"]],
        on="event", how="inner")
    events = events[events["p"] >= pmin]
    if len(events) > max_events:
        events = events.sample(max_events, random_state=seed)
    events = events.sort_values("event").reset_index(drop=True)
    rows, covs = [], []
    t0 = time.time()
    for i, e in enumerate(events.itertuples(index=False)):
        v = np.array([e.x, e.y, e.z], float)
        d = np.array([e.dx, e.dy, e.dz], float)
        try:
            b = bound(model, particle, v, d, float(e.p), timing=timing, shower=shower, floor=floor)
            ok = np.all(np.isfinite(b["cov"])) and np.all(np.diag(b["cov"]) > 0)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            ok = False
        if not ok:
            continue
        rows.append({"event": int(e.event), "p_true": float(e.p),
                     "sigma_t": b["sigma_t"], "sigma_long": b["sigma_long"],
                     "sigma_perp": b["sigma_perp"], "sigma_angle": b["sigma_angle"],
                     "sigma_p_rel": b["sigma_p_rel"], "npe": b["npe"],
                     "nhit": b["nhit"], "shower_sigma": b["shower_sigma"],
                     "dx_true": e.dx, "dy_true": e.dy, "dz_true": e.dz})
        covs.append(b[cov_key])
        if log and (i + 1) % 500 == 0:
            rate = (time.time() - t0) / (i + 1)
            log(f"  Cramer-Rao bound: {i + 1}/{len(events)} events "
                f"({rate * 1e3:.0f} ms/event, ~{rate * (len(events) - i - 1):.0f} s left)")
    table = pd.DataFrame(rows)
    table.attrs.update(particle=particle, timing=timing, shower=shower, floor=floor,
                       n_requested=len(events), p_min=pmin)
    return table, np.array(covs)


def efficient_residuals(table: pd.DataFrame, covs, residuals: pd.DataFrame, *,
                        n_draw=40, seed=0) -> pd.DataFrame:
    """Residuals an efficient estimator would produce, n_draw per event, carrying
    the same binning columns (p_true, dwall, towall, ...) as `residuals`."""
    if table.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    n = len(table)
    L = np.linalg.cholesky(covs + 1e-30 * np.eye(7))
    z = rng.standard_normal((n, n_draw, 7))
    draws = np.einsum("nij,nkj->nki", L, z)              # (n, n_draw, 7)
    d = table[["dx_true", "dy_true", "dz_true"]].to_numpy(float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dv = draws[..., 1:4]
    along = np.einsum("nkj,nj->nk", dv, d)
    dist = np.linalg.norm(dv, axis=2)
    p_true = table["p_true"].to_numpy(float)[:, None]
    out = pd.DataFrame({
        "event": np.repeat(table["event"].to_numpy(), n_draw),
        "t": draws[..., 0].ravel(),
        "vertex_dist": dist.ravel(),
        "vertex_long": along.ravel(),
        "vertex_trans": np.sqrt(np.maximum(dist ** 2 - along ** 2, 0)).ravel(),
        "angle": np.hypot(draws[..., 4], draws[..., 5]).ravel(),
        "p_res": draws[..., 6].ravel(),
        "p_res_pct": (100.0 * draws[..., 6] / p_true).ravel(),
    })
    out["angle_deg"] = np.degrees(out["angle"])
    keep = [c for c in ("event", "p_true", "dwall", "towall", "x_true", "y_true",
                        "z_true", "nhits") if c in residuals.columns]
    out = out.merge(residuals[keep].drop_duplicates("event"), on="event", how="left")
    out.attrs.update(table.attrs)
    out.attrs["n_draw"] = n_draw
    out.attrs["n_events"] = n
    return out


def bound_arm(model, residuals_by_arm: dict, truth, hypothesis, *, timing="measured",
              shower=True, n_draw=40, seed=0, log=print):
    """(bound residual table, per-event table), or (None, reason) when the sample
    cannot support a single-ring bound."""
    particle = true_particle(truth.primary)
    if particle is None:
        return None, "the sample is not a single-particle gun sample"
    if particle not in model.light:
        return None, f"the model has no light profile for {particle}"
    if hypothesis != particle:
        return None, (f"the fits are {hypothesis} hypotheses on a {particle} sample; "
                      "the bound is for the true particle")
    frames = [r for r in residuals_by_arm.values() if r is not None and not r.empty]
    if not frames:
        return None, "no residuals"
    union = (pd.concat(frames)[["event", "p_true", "dwall", "towall", "x_true",
                                "y_true", "z_true"]]
             .drop_duplicates("event"))
    table, covs = event_bounds(model, union, truth.primary, particle, timing=timing,
                               shower=shower, seed=seed, log=log)
    if table.empty:
        return None, "the bound could not be computed for any event"
    return efficient_residuals(table, covs, union, n_draw=n_draw, seed=seed), table
