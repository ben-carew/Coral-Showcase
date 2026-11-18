"""Build the Cramer-Rao light and timing model from a WCSim particle-gun sample.

Everything the bound needs about the detector is MEASURED here from truth-matched
simulation, in three tables per particle type:

1. **Light profile** h(delta_alpha | p, R): mean true photoelectrons per PMT, per
   emitted Cherenkov photon, per unit geometric acceptance, as a function of the
   PMT's angle from the ring edge (delta_alpha = alpha - theta_edge(p)), the
   vertex-PMT distance R and the momentum. It is estimated as a ratio of sums over
   every PMT of every event -- hit or not --

       h(cell) = SUM n_pe / SUM N_gamma(p) J(R, eta)

   which is the Poisson maximum-likelihood rate in the cell and is correct however
   unevenly the detector samples it. (A sum of per-hit ratios over hit PMTs, the
   estimator the first version of this bound used, weights each cell by how many
   PMTs happen to sit in it.) The profile includes scattered and reflected light,
   because a PMT cannot tell them apart from direct light by charge.

   Why not fiTQun's own CProfile tables: they are binned at 0.04 in cos(theta0),
   about 3.4 deg at the ring, and cannot represent the ring's OUTER edge -- a
   kinematic edge (light from further down the track always lands inside the cone
   seen from the vertex) that the data show sharp to well under a degree. The
   direction bound lives on that edge.

2. **Timing information** I_t(delta_alpha, mu): the Fisher information for a shift of
   the digitised hit-time distribution, per hit, measured from the residual of every
   digitised hit against Coral's direct-light arrival time at the true parameters.
   Scattered light, the extended track, dark noise and the digitiser are all in it.
   A hit dominated by scattered light carries almost no timing information, and this
   table is what knows that.

3. The charge response (single-pe width and threshold, see `response.py`) is a
   property of the PMT and is shared.

    sextant crb build --sample Data/prod_sims_h5 \\
        --tunings Software/Coral/Tunings --out Output/Sextant/crb_model/crb_model.npz
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os

import numpy as np

from .model import C_LIGHT, Detector
from .profile import CherenkovProfile
from .response import location_information

PARTICLES = {"e-": ("electrons", "CProf_11_fit_WCSim.root", 0.511),
             "mu-": ("muons", "CProf_13_fit_WCSim.root", 105.658)}

#: Cherenkov-angle refractive index (phase index, photon-weighted). Fitted so that
#: the steepest fall of the electron profile sits at delta_alpha = 0; see `build`.
N_PHASE = 1.349
#: Coral's group index for arrival times ([physics] refraction_index).
N_GROUP = 1.38
ATTEN_LENGTH = 9500.0
EPS_POL4 = np.array([-3.53688089e-03, 2.68575804e+00, -5.52989632e+00,
                     7.02090757e+00, -3.17244887e+00])

P_EDGES = {"e-": np.array([50., 150., 300., 500., 750., 1050., 1600.]),
           "mu-": np.array([180., 300., 450., 650., 900., 1200., 1700.])}
R_EDGES = np.array([0., 500., 900., 1400., 2000., 2800., 4000., 10000.])
DA_EDGES = np.unique(np.r_[np.arange(-50., -30., 5.), np.arange(-30., -16., 1.),
                          np.arange(-16., 8., 0.5), np.arange(8., 20., 1.),
                          np.arange(20., 145., 5.)])
T_DA_EDGES = np.array([-50., -20., -10., -5., -2., 0., 2., 5., 10., 20., 40.,
                       70., 100., 145.])
T_MU_EDGES = np.array([0., 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e9])


def theta_edge(p, mass, n_phase=N_PHASE):
    """Cherenkov angle at the start of the track, degrees (0 below threshold)."""
    beta = p / np.sqrt(p * p + mass * mass)
    c = np.clip(1.0 / (n_phase * beta), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def acceptance(det, vertex, atten=ATTEN_LENGTH, eps=EPS_POL4):
    """Geometric acceptance of every PMT for light from `vertex`: exact disc solid
    angle x angular response x attenuation. Returns (J, R, eta)."""
    r = det.pos - vertex
    R = np.sqrt(np.einsum("ij,ij->i", r, r))
    a = det.pmt_radius
    omega = 2.0 * np.pi * (1.0 - R / np.sqrt(R * R + a * a))
    eta = np.clip(-np.einsum("ij,ij->i", r, det.ori) / R, 0.0, 1.0)
    return omega * np.polyval(eps[::-1], eta) * np.exp(-R / atten), R, eta


def iter_events(sample_dir, subdir, max_files=None):
    """Yield (event_key, p, t0, vertex, direction, npe_by_row, digi) per event.

    digi = (rows, Q, T) of the digitised hits, earliest digit per tube.
    """
    files = sorted(glob.glob(os.path.join(sample_dir, subdir, "*.h5")))
    if max_files:
        files = files[:max_files]
    import h5py
    row_of = None
    for ifile, path in enumerate(files):
        with h5py.File(path) as f:
            if row_of is None:
                tub = f["wcsimGeoT/PMT"]["TubeNo"][:]
                row_of = np.full(int(tub.max()) + 1, -1, np.int64)
                row_of[tub] = np.arange(len(tub))
                n_pmt = len(tub)
            tr = f["wcsimT/Tracks"][:]
            ch = f["wcsimT/CherenkovHits"][:]
            dg = f["wcsimT/CherenkovDigiHits/DigiHits"][:]
        prim = tr[(tr["Parenttype"] == 0) & (tr["Id"] == 1)]
        ch_ev = _split(ch, "EvtNum")
        dg_ev = _split(dg, "EvtNum")
        for r in prim:
            ev = int(r["EvtNum"])
            d = np.array([r["Dir_x0"], r["Dir_x1"], r["Dir_x2"]], float)
            nd = np.linalg.norm(d)
            if nd == 0 or ev not in ch_ev:
                continue
            c = ch_ev[ev]
            rows = row_of[c["TubeID"]]
            ok = rows >= 0
            npe = np.bincount(rows[ok], c["TotalPe_1"][ok], minlength=n_pmt)
            g = dg_ev.get(ev)
            if g is None or len(g) == 0:
                digi = (np.zeros(0, int), np.zeros(0), np.zeros(0))
            else:
                grow = row_of[g["TubeID"]]
                order = np.lexsort((g["T"], grow))
                grow, gq, gt = grow[order], g["Q"][order], g["T"][order]
                first = np.r_[True, grow[1:] != grow[:-1]] & (grow >= 0)
                digi = (grow[first], gq[first], gt[first])
            yield ((ifile, ev), float(r["P"]), float(r["Time"]),
                   np.array([r["Start_x0"], r["Start_x1"], r["Start_x2"]], float),
                   d / nd, npe, digi)


def _split(arr, key):
    if len(arr) == 0:
        return {}
    order = np.argsort(arr[key], kind="stable")
    a = arr[order]
    keys, starts = np.unique(a[key], return_index=True)
    ends = np.r_[starts[1:], len(a)]
    return {int(k): a[s:e] for k, s, e in zip(keys, starts, ends)}


def _bin(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def build_particle(det, sample_dir, particle, prof, *, max_files=None,
                   subset=None, n_phase=N_PHASE, log=print):
    subdir, _, mass = PARTICLES[particle]
    pe = P_EDGES[particle]
    npb, nrb, nab = len(pe) - 1, len(R_EDGES) - 1, len(DA_EDGES) - 1
    num = np.zeros((npb, nrb, nab))
    den = np.zeros((npb, nrb, nab))
    rsum = np.zeros((npb, nrb))
    rw = np.zeros((npb, nrb))
    lp_sum = np.zeros(npb)
    lp_n = np.zeros(npb)
    kept = []
    for i, (key, p, t0, v, d, npe, digi) in enumerate(
            iter_events(sample_dir, subdir, max_files)):
        if subset == "even" and i % 2:
            continue
        if subset == "odd" and not i % 2:
            continue
        if not (pe[0] <= p < pe[-1]):
            continue
        J, R, _ = acceptance(det, v)
        alpha = np.degrees(np.arccos(np.clip(((det.pos - v) @ d) / R, -1, 1)))
        da = alpha - theta_edge(p, mass, n_phase)
        ib = _bin(p, pe)
        rb = _bin(R, R_EDGES)
        ab = _bin(da, DA_EDGES)
        flat = (rb * nab + ab)
        ng = prof.n_photons(p)
        num[ib] += np.bincount(flat, npe, nrb * nab).reshape(nrb, nab)
        den[ib] += np.bincount(flat, ng * J, nrb * nab).reshape(nrb, nab)
        rsum[ib] += np.bincount(rb, R * J, nrb)
        rw[ib] += np.bincount(rb, J, nrb)
        lp_sum[ib] += np.log(p)
        lp_n[ib] += 1
        kept.append((p, t0, v, d, digi, npe.sum()))
    h = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    table = dict(h=h, num=num, den=den,
                 logp_ctr=lp_sum / np.maximum(lp_n, 1),
                 r_ctr=rsum / np.maximum(rw, 1e-300),
                 n_events=lp_n)
    log(f"  {particle}: light profile from {len(kept)} events "
        f"({', '.join(str(int(n)) for n in lp_n)} per momentum bin)")
    return table, kept


def build_timing(det, lm_eval, kept, particle, *, n_phase=N_PHASE, cap=20000,
                 seed=0, log=print):
    """Per-hit timing information in (delta_alpha, mu) cells."""
    _, _, mass = PARTICLES[particle]
    nda, nmu = len(T_DA_EDGES) - 1, len(T_MU_EDGES) - 1
    samples = [[[] for _ in range(nmu)] for _ in range(nda)]
    rng = np.random.default_rng(seed)
    for p, t0, v, d, (rows, q, t), _ in kept:
        if rows.size < 20:
            continue
        mu = lm_eval(v, d, p)
        half = 0.5 * lm_eval.track_length(p)
        mid = v + half * d
        tpred = t0 + half / C_LIGHT + np.linalg.norm(det.pos[rows] - mid, axis=1) * N_GROUP / C_LIGHT
        R = np.linalg.norm(det.pos[rows] - v, axis=1)
        alpha = np.degrees(np.arccos(np.clip(((det.pos[rows] - v) @ d) / R, -1, 1)))
        da = alpha - theta_edge(p, mass, n_phase)
        res = t - tpred
        core = (np.abs(da) < 4.0) & (mu[rows] > 0.5)
        if core.sum() < 10:
            continue
        res = res - np.median(res[core])           # the trigger offset
        ia = _bin(da, T_DA_EDGES)
        im = _bin(mu[rows], T_MU_EDGES)
        for a, m, r in zip(ia, im, res):
            samples[a][m].append(r)
    info = np.full((nda, nmu), np.nan)
    count = np.zeros((nda, nmu), int)
    for a in range(nda):
        for m in range(nmu):
            x = np.asarray(samples[a][m])
            if x.size > cap:
                x = rng.choice(x, cap, replace=False)
            info[a, m], count[a, m] = location_information(x)
    # Cells too sparse to fit borrow from the nearest filled cell in mu, then in
    # angle: an empty corner of the table is a place hits almost never happen.
    info = _fill(info)
    log(f"  {particle}: timing information from "
        f"{int(sum(len(s) for row in samples for s in row))} digitised hits")
    return dict(info=info, count=count)


def _fill(a):
    a = a.copy()
    for _ in range(a.shape[0] + a.shape[1]):
        if not np.isnan(a).any():
            break
        b = a.copy()
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isnan(a[i, j]):
                    nb = [a[x, y] for x, y in ((i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j))
                          if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]
                          and not np.isnan(a[x, y])]
                    if nb:
                        b[i, j] = np.mean(nb)
        a = b
    return a


def build(sample_dir, tunings_dir, out, *, max_files=None, subset=None, log=print):
    from .model import CRBModel
    first = sorted(glob.glob(os.path.join(sample_dir, "electrons", "*.h5")))[0]
    det = Detector.from_wcsim_h5(first)
    arrays = det.to_arrays()
    arrays["const"] = np.array([N_PHASE, N_GROUP, ATTEN_LENGTH])
    arrays["eps_pol4"] = EPS_POL4
    arrays["t_da_edges"] = T_DA_EDGES
    arrays["t_mu_edges"] = T_MU_EDGES
    arrays["da_edges"] = DA_EDGES
    arrays["r_edges"] = R_EDGES
    provenance = {"sample": os.path.abspath(sample_dir),
                  "tunings": os.path.abspath(tunings_dir),
                  "subset": subset, "max_files": max_files,
                  "built": _dt.datetime.now().isoformat(timespec="seconds"),
                  "particles": {}}
    for particle, (_, cprof, mass) in PARTICLES.items():
        prof = CherenkovProfile.from_root(os.path.join(tunings_dir, cprof))
        table, kept = build_particle(det, sample_dir, particle, prof,
                                     max_files=max_files, subset=subset, log=log)
        tag = particle.replace("-", "")
        arrays[f"{tag}_p_edges"] = P_EDGES[particle]
        arrays[f"{tag}_mass"] = np.array(mass)
        arrays[f"{tag}_nphot_pars"] = prof.nphot_pars
        arrays[f"{tag}_maxd_pars"] = prof.maxd_pars
        for k, val in table.items():
            arrays[f"{tag}_light_{k}"] = val
        # Timing needs the light model (mu per PMT picks the cell), so assemble one
        # from what exists so far.
        partial = CRBModel.from_arrays(arrays, particles=[particle], allow_missing_timing=True)
        timing = build_timing(det, partial.light[particle], kept, particle, log=log)
        arrays[f"{tag}_timing_info"] = timing["info"]
        arrays[f"{tag}_timing_count"] = timing["count"]
        provenance["particles"][particle] = {
            "events": int(sum(table["n_events"])),
            "events_per_p_bin": [int(x) for x in table["n_events"]]}
    arrays["provenance"] = np.array(json.dumps(provenance))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **arrays)
    log(f"wrote {out}")
    return out
