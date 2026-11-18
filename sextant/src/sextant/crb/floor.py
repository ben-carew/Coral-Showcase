"""Physics floors: event-level fluctuations that no single-ring fit can remove.

The Fisher bound (`fisher.py`) assumes every event of a given (vertex, direction, p) emits light
with the AVERAGE angular and longitudinal profile; only photon counting and the shower's total-light
fluctuation are random. Real particles are not average. Per event:

  * the Cherenkov light axis is not the initial direction (shower development, multiple scattering);
  * the light is emitted from a centroid that moves along the track (shower start and depth);
  * the emitting line is offset sideways from the true track (shower spread, lever arm of the tilt);
  * a muon's track length at fixed momentum straggles, and with it its total light.

A fit with a straight track and an average emission profile -- Coral's and fiTQun's model family --
measures the light, so these fluctuations pass straight into its parameters. They are measured here
from WCSim's per-photoelectron truth (emission point, landing point; direct photons only). Each
detected photon is weighted by 1 / (its detection probability: attenuation over its path, PMT
acceptance at its angle of incidence, both from the CRB model), so the weighted photons describe the
EMITTED light: which part of the cone happens to be detected is geometry, which a fitter models, not
physics. Each event's own photon-counting noise is removed by bootstrapping its photons, because the
Fisher part of the bound already counts photons. The light axis is the weighted mean photon direction
(a cone fit is pulled around by the broad-angle light of low-energy shower particles). Shift-type floors add to the bound's covariance exactly (an error
on top of an independent error); the muon yield enters as the light-scale nuisance the electron
shower yield already uses.

    events = measure("Data/prod_sims_h5", "e-")        # one row per event
    floors = summarise(events, "e-")                    # sigmas per momentum bin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MM_TO_CM = 0.1
C_VAC = 29.9792458          # cm/ns
N_DIRECT_MAX = 1.42         # group-index ceiling for unscattered light (photon_truth.py)
SUBDIR = {"e-": "electrons", "mu-": "muons"}
#: Momentum bins for the floors (MeV/c), log-interpolated between centres.
P_EDGES = np.array([100, 200, 300, 450, 650, 900, 1300, 2000, 3500], float)
#: Only events whose true track is well inside the tank measure the floor: near a wall the
#: detected photons cover the cone unevenly, which biases the axis by geometry (a fitter models
#: that), not by physics.
MIN_DWALL, MIN_TOWALL = 200.0, 500.0
#: How much of each truth fluctuation a likelihood fit actually inherits: the regression slope of a
#: fit's per-event error on the truth quantity, bulk events of the v1.12.22 guns whose photon truth
#: is used here. Each entry is the SMALLER of Coral's (production of 2026-08-26) and fiTQun's, so the
#: floor is what neither existing algorithm removes; slopes [corr] Coral / fiTQun: light axis
#: 0.26 [0.54] / 0.22 [0.54]; emission centroid along the track 0.83 [0.76] / 0.79 [0.85]; emitting
#: line transverse 0.31 [0.32] / 0.10 [0.14]. The all-photon axis also carries broad-angle shower
#: light a ring fit ignores (axis spread 4-5 deg); a ring-only axis was tried and is unstable. The
#: muon angle is Coral's (fiTQun's muon sample is too small). An algorithm that modelled each
#: shower's own development could go below these floors; none does today.
RESPONSE = {"e-": dict(angle=0.22, along=0.79, perp=0.10),
            "mu-": dict(angle=0.12, along=0.0, perp=0.0)}         # along / perp unused (APPLIES)
#: The along-track floor shrinks toward the wall the track points at (fiTQun, slope x centroid
#: spread, relative to towall >= 10 m): (towall [cm], factor), linear between, clamped outside.
ALONG_TOWALL = ((450.0, 0.77), (800.0, 0.88), (1250.0, 1.0))
C_CM_PER_NS = 29.98
#: Which floors apply to which particle. Muon tracks start sharply, so an along / transverse floor
#: from the emission CENTROID would charge them for range straggling twice (it is in the yield).
APPLIES = {"e-": ("angle", "along", "perp"), "mu-": ("angle", "yield")}


def _frame(d):
    from .fisher import frame
    return frame(d)


def _primary(tracks, event):
    t = tracks[(tracks["EvtNum"] == event) & (tracks["Parenttype"] == 0) & (tracks["Id"] == 1)]
    if len(t) == 0:
        return None
    t = t[0]
    d = np.array([t["Dir_x0"], t["Dir_x1"], t["Dir_x2"]], float)
    return dict(v=np.array([t["Start_x0"], t["Start_x1"], t["Start_x2"]], float), d=d / np.linalg.norm(d),
                stop=np.array([t["Stop_x0"], t["Stop_x1"], t["Stop_x2"]], float), p=float(t["P"]))


#: Grazing photons (cos incidence below this) are dropped: their 1/acceptance weight is noise.
MIN_COS_INCIDENCE = 0.25


def _detection_weight(start, end, u, consts, *, radius, half_height):
    """1 / P(detected | emitted towards this PMT): attenuation and angular acceptance."""
    atten, eps_pol4 = consts
    r = np.hypot(end[:, 0], end[:, 1])
    cap = np.abs(end[:, 2]) > half_height - 60.0
    normal = np.where(cap[:, None], np.c_[np.zeros_like(r), np.zeros_like(r), -np.sign(end[:, 2])],
                      np.c_[-end[:, 0] / r, -end[:, 1] / r, np.zeros_like(r)])
    eta = -(u * normal).sum(1)
    eps = np.polyval(eps_pol4[::-1], np.clip(eta, 0.0, 1.0))
    w = np.exp(np.linalg.norm(end - start, axis=1) / atten) / np.maximum(eps, 1e-6)
    return np.where(eta >= MIN_COS_INCIDENCE, w, 0.0)


def _event_quantities(u, start, w, v, d, e1, e2):
    a = (w[:, None] * u).sum(0)
    a /= np.linalg.norm(a)
    tau1, tau2 = a @ e1, a @ e2
    rel = start - v
    W = w.sum()
    s_c = (w * (rel @ d)).sum() / W
    c1, c2 = (w * (rel @ e1)).sum() / W, (w * (rel @ e2)).sum() / W
    return np.array([tau1, tau2, s_c, c1 - s_c * tau1, c2 - s_c * tau2])


def _model_constants(model_path):
    with np.load(model_path, allow_pickle=False) as a:
        n_phase, n_group, atten = a["const"]
        return (float(atten), np.asarray(a["eps_pol4"], float))


def measure(sample_dir, particle, model_path, *, max_files=None, n_boot=40, min_photons=50, seed=0, log=print):
    """One row per event: true p, geometry, light-axis tilts [rad], emission centroid along the
    track [cm], transverse offset of the emitting line at the vertex [cm], each with its
    photon-counting variance; and (muons) the true track length."""
    import h5py
    from ..io.geometry import dwall, towall
    from ..io.geometry import DEFAULT_RADIUS_CM, DEFAULT_HALF_HEIGHT_CM
    consts = _model_constants(model_path)
    rng = np.random.default_rng(seed)
    files = sorted((Path(sample_dir) / SUBDIR[particle]).glob("*.h5"))[:max_files]
    rows = []
    for f in files:
        with h5py.File(f, "r") as h:
            w = h["wcsimT"]
            tracks, hits, times = w["Tracks"][:], w["CherenkovHits"][:], w["CherenkovHitTimes"][:]
        for ev in np.unique(hits["EvtNum"]):
            tr = _primary(tracks, ev)
            if tr is None:
                continue
            t = times[times["EvtNum"] == ev]
            keep = t["PrimaryParentID"] >= 0
            start = np.stack([t[f"PhotonStartPos_x{i}"] for i in range(3)], 1)[keep] * MM_TO_CM
            end = np.stack([t[f"PhotonEndPos_x{i}"] for i in range(3)], 1)[keep] * MM_TO_CM
            step = end - start
            dist = np.linalg.norm(step, axis=1)
            travel = (t["Truetime"] - t["PhotonStartTime"])[keep]
            with np.errstate(divide="ignore", invalid="ignore"):
                n_eff = travel * C_VAC / dist
            direct = (dist > 0) & (n_eff <= N_DIRECT_MAX)
            if direct.sum() < min_photons:
                continue
            u = step[direct] / dist[direct][:, None]
            start, end = start[direct], end[direct]
            w = _detection_weight(start, end, u, consts, radius=DEFAULT_RADIUS_CM, half_height=DEFAULT_HALF_HEIGHT_CM)
            keep = w > 0
            u, start, w = u[keep], start[keep], w[keep]
            if len(u) < min_photons:
                continue
            v, d = tr["v"], tr["d"]
            _, e1, e2 = _frame(d)
            q = _event_quantities(u, start, w, v, d, e1, e2)
            n = len(u)
            boot = np.array([_event_quantities(u[i], start[i], w[i], v, d, e1, e2)
                             for i in (rng.integers(0, n, n) for _ in range(n_boot))])
            var = boot.var(axis=0, ddof=1)
            rows.append(dict(file=f.name, event=int(ev), p=tr["p"], n_direct=n,
                             dwall=float(dwall(*v)), towall=float(towall(*v, *d)),
                             tau1=q[0], tau2=q[1], s_c=q[2], t1=q[3], t2=q[4],
                             v_tau=0.5 * (var[0] + var[1]), v_s=var[2], v_t=0.5 * (var[3] + var[4]),
                             length=float(np.linalg.norm(tr["stop"] - v)),
                             stop_inside=bool(dwall(*tr["stop"]) > 0)))
        if log:
            log(f"  {f.name}: {len(rows)} events so far")
    return pd.DataFrame(rows)


def _half68(x):
    lo, hi = np.percentile(np.asarray(x, float), [16, 84])
    return 0.5 * (hi - lo)


def summarise(events, particle, *, edges=P_EDGES, min_events=30):
    """Floors per momentum bin: dict(logp, angle [rad, per tilt], along [cm], perp [cm, per axis],
    yield [fraction], n). Spreads are robust (half the central 68%); each event's counting variance
    is subtracted in quadrature; trends with momentum are removed before the spread is taken; each
    spread is scaled by the fit's measured RESPONSE to it. The muon yield is the straggling of the
    track length at fixed momentum (contained muons), entered as the light-scale nuisance."""
    e = events[(events.dwall > MIN_DWALL) & (events.towall > MIN_TOWALL)].copy()
    lp = np.log(e.p)
    # the emission centroid moves down the track as log p grows: an average profile knows that
    e["s_res"] = e.s_c - np.polyval(np.polyfit(lp, e.s_c, 2), lp)
    yield_res = None
    if particle == "mu-":
        m = events[events.stop_inside & (events.length > 0)]
        lpm = np.log(m.p)
        yield_res = (np.log(m.length) - np.polyval(np.polyfit(lpm, np.log(m.length), 3), lpm), lpm)
    out = dict(logp=[], angle=[], along=[], perp=[], yld=[], n=[])
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = e[(e.p >= lo) & (e.p < hi)]
        if len(b) < min_events:
            continue
        ang = np.r_[b.tau1, b.tau2] - np.r_[b.tau1, b.tau2].mean()
        perp = np.r_[b.t1, b.t2] - np.r_[b.t1, b.t2].mean()
        out["logp"].append(np.log(np.sqrt(lo * hi)))
        k = RESPONSE[particle]
        out["angle"].append(k["angle"] * np.sqrt(max(_half68(ang) ** 2 - b.v_tau.mean(), 0.0)))
        out["along"].append(k["along"] * np.sqrt(max(_half68(b.s_res) ** 2 - b.v_s.mean(), 0.0)))
        out["perp"].append(k["perp"] * np.sqrt(max(_half68(perp) ** 2 - b.v_t.mean(), 0.0)))
        if yield_res is not None:
            r, lpm = yield_res
            sel = (lpm >= np.log(lo)) & (lpm < np.log(hi))
            out["yld"].append(_half68(r[sel]) if sel.sum() >= min_events else np.nan)
        else:
            out["yld"].append(np.nan)
        out["n"].append(len(b))
    return {k: np.array(v, float) for k, v in out.items()}


def along_towall_factor(towall):
    t, f = zip(*ALONG_TOWALL)
    return float(np.interp(towall, t, f))


class Floors:
    """Per-particle floors, interpolated in log p (clamped at the measured ends)."""

    def __init__(self, logp, angle, along, perp, yld, applies):
        self.logp, self.angle, self.along, self.perp, self.yld = logp, angle, along, perp, yld
        self.applies = tuple(applies)

    def _at(self, arr, p):
        ok = np.isfinite(arr)
        if not ok.any():
            return 0.0
        return float(np.interp(np.log(p), self.logp[ok], arr[ok]))

    def sigmas(self, p):
        """dict(angle [rad per tilt], along [cm], perp [cm per axis], yield [fraction]); 0 when not applied."""
        return dict(angle=self._at(self.angle, p) if "angle" in self.applies else 0.0,
                    along=self._at(self.along, p) if "along" in self.applies else 0.0,
                    perp=self._at(self.perp, p) if "perp" in self.applies else 0.0,
                    yld=self._at(self.yld, p) if "yield" in self.applies else 0.0)

    def covariance(self, d, e1, e2, p, towall=None):
        """7x7 covariance in (t, x, y, z, tilt1, tilt2, p) to ADD to the Fisher bound's."""
        s = self.sigmas(p)
        along = s["along"] * (along_towall_factor(towall) if towall is not None else 1.0)
        cov = np.zeros((7, 7))
        # an emission-centroid shift moves the fitted vertex along the track and its time with it
        # (both codes: dt / d_along = 0.9-1.07 x 1/c, corr ~0.9)
        shift = np.r_[1.0 / C_CM_PER_NS, d]
        cov[0:4, 0:4] = along ** 2 * np.outer(shift, shift)
        cov[1:4, 1:4] += s["perp"] ** 2 * (np.outer(e1, e1) + np.outer(e2, e2))
        cov[4, 4] = cov[5, 5] = s["angle"] ** 2
        return cov

    def to_arrays(self, tag):
        return {f"{tag}_floor_logp": self.logp, f"{tag}_floor_angle": self.angle,
                f"{tag}_floor_along": self.along, f"{tag}_floor_perp": self.perp,
                f"{tag}_floor_yield": self.yld, f"{tag}_floor_applies": np.array(",".join(self.applies))}

    @classmethod
    def from_arrays(cls, a, tag):
        if f"{tag}_floor_logp" not in a:
            return None
        return cls(a[f"{tag}_floor_logp"], a[f"{tag}_floor_angle"], a[f"{tag}_floor_along"],
                   a[f"{tag}_floor_perp"], a[f"{tag}_floor_yield"],
                   str(a[f"{tag}_floor_applies"]).split(","))


def add_to_model(model_path, sample_dir=None, *, events=None, particles=("e-", "mu-"), max_files=None, log=print):
    """Measure the floors (or reuse `events`, {particle: table} from an earlier measure) and write them
    into an existing model .npz (other keys untouched). Returns {particle: (events table, floors dict)}."""
    with np.load(model_path, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    out = {}
    for particle in particles:
        ev = events[particle] if events else measure(sample_dir, particle, model_path, max_files=max_files, log=log)
        fl = summarise(ev, particle)
        arrays.update(Floors(fl["logp"], fl["angle"], fl["along"], fl["perp"], fl["yld"],
                             APPLIES[particle]).to_arrays(particle.replace("-", "")))
        out[particle] = (ev, fl)
    np.savez_compressed(model_path, **arrays)
    return out
