#!/usr/bin/env python
"""R54: learned post-fit estimator, trained on files 0-299 and FROZEN, applied to files 300-699, paired
against production (same events, same code version: r54_train_{e,mu} / r54_test_{e,mu}).

Features and targets are those of post_fit_estimator_cv.py (both hypotheses' fits, fitted geometry,
observed totals as ratios to p; no momentum magnitude). Targets: ln(p_fit/p_true), the vertex error along
the fitted track and across it, the direction error (deg) in the fitted frame; each is predicted and
subtracted. Pre-registered PASS (roadmap R54): e- p core <= 0.90x production (CI), no (true p, true
towall) bin of any width worse beyond CI, per-bin |median bias| not worse beyond CI, PID untouched.

    python r54_heldout.py --hyp e-     # or mu-
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Software/Sextant/src"))
from sextant import Arm, Truth, analysis                          # noqa: E402
from sextant.io.geometry import dwall as gdwall, towall as gtowall  # noqa: E402
from sextant.metrics.stats import gauss_sigma                     # noqa: E402
RNG = np.random.default_rng(54)
TARGETS = ["logp", "along", "u", "v", "a1", "a2"]


def table(spec, hyp, sample, truth):
    """Features X, targets Y, fitted frame and truth for one run (converged and not)."""
    arm = Arm.from_file("C", ROOT / f"Output/Coral/Reco/{spec}/coral_{sample}_coral.root")
    res = analysis.compute_residuals([arm], truth, hypothesis=hyp)["C"].drop_duplicates("event").set_index("event")
    f = arm.fits[(arm.fits.stage == -1) & (arm.fits.sub_event == 0)]
    hyps = {h: f[f.hypothesis == h].drop_duplicates("event").set_index("event") for h in ("e-", "mu-")}
    own, oth = hyps[hyp], hyps["mu-" if hyp == "e-" else "e-"]
    tp = truth.primary.drop_duplicates("event").set_index("event")
    ev = own.index.intersection(oth.index).intersection(res.index).intersection(tp.index)
    o, t, T, r = own.loc[ev], oth.loc[ev], tp.loc[ev], res.loc[ev]
    d = np.c_[o.dx, o.dy, o.dz]; d /= np.linalg.norm(d, axis=1)[:, None]
    ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0, 0, 1.]], [[1., 0, 0]])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(d, e1)
    X = pd.DataFrame(index=ev)
    X["dlogp"] = np.log(t.p/o.p)
    X["dwall"] = gdwall(o.x, o.y, o.z); X["towall"] = gtowall(o.x, o.y, o.z, o.dx, o.dy, o.dz)
    X["r"] = np.hypot(o.x, o.y); X["z"] = o.z; X["cz"] = d[:, 2]; X["crad"] = (o.x*d[:, 0] + o.y*d[:, 1])/np.maximum(X.r, 1)
    X["dnll"] = t.nll - o.nll; X["nll"] = o.nll
    dvh = np.c_[t.x - o.x, t.y - o.y, t.z - o.z]
    X["dv_along"] = (dvh*d).sum(1); X["dv_u"] = (dvh*e1).sum(1); X["dv_v"] = (dvh*e2).sum(1); X["dt"] = t.t - o.t
    dd = np.c_[t.dx, t.dy, t.dz]; X["dd_u"] = np.degrees((dd*e1).sum(1)); X["dd_v"] = np.degrees((dd*e2).sum(1))
    X["e1z"] = e1[:, 2]; X["e2z"] = e2[:, 2]
    X["e1rad"] = (e1[:, 0]*o.x + e1[:, 1]*o.y)/np.maximum(X.r, 1); X["e2rad"] = (e2[:, 0]*o.x + e2[:, 1]*o.y)/np.maximum(X.r, 1)
    X["q_per_p"] = np.log(r.totq.to_numpy()/o.p.to_numpy()); X["hits_per_p"] = np.log(r.nhits.to_numpy()/o.p.to_numpy())
    td = np.c_[T.dx, T.dy, T.dz]; td /= np.linalg.norm(td, axis=1)[:, None]
    dv = np.c_[o.x - T.x, o.y - T.y, o.z - T.z]
    Y = pd.DataFrame({"logp": np.log1p(r.p_res_pct.to_numpy()/100), "along": (dv*d).sum(1), "u": (dv*e1).sum(1),
                      "v": (dv*e2).sum(1), "a1": np.degrees((td*e1).sum(1)), "a2": np.degrees((td*e2).sum(1))}, index=ev)
    ok = (r.angle_deg < 20).to_numpy() & np.isfinite(Y.logp.to_numpy())   # NaN features are handled by the model (q_per_p: totq unknown)
    twt = np.asarray(gtowall(T.x, T.y, T.z, T.dx, T.dy, T.dz))
    return X[ok], Y[ok], T.p.to_numpy()[ok], twt[ok]


def core(x): return 100*gauss_sigma(np.asarray(x))
def p68(x): return np.percentile(np.abs(np.asarray(x)), 68)


def metrics(Y, i):
    return {"p_core": core(Y.logp.to_numpy()[i]),
            "vtx68": p68(np.sqrt(Y.along**2 + Y.u**2 + Y.v**2).to_numpy()[i]),
            "ang68": p68(np.hypot(Y.a1, Y.a2).to_numpy()[i]),
            "abs_bias": 100*abs(np.median(Y.logp.to_numpy()[i]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default="e-")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    smp, sample = ("e", "electron") if a.hyp == "e-" else ("mu", "muon")
    truth = Truth.from_file(ROOT / f"Output/Truth/truth_{sample}_v1.12.29.root")
    Xa, Ya, _, _ = table(f"r54_train_{smp}", a.hyp, sample, truth)
    assert (Xa.index // 100).max() < 300, "training events must come from files 0-299"
    models = {c: HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.04, min_samples_leaf=50,
                                               loss="absolute_error").fit(Xa, Ya[c]) for c in TARGETS}
    out = ROOT / f"Output/Sextant/r54"; out.mkdir(parents=True, exist_ok=True)
    pickle.dump({"features": list(Xa.columns), "models": models}, open(out / f"post_fit_estimator_{smp}.pkl", "wb"))
    Xb, Yb, ptrue, twt = table(f"r54_test_{smp}", a.hyp, sample, truth)
    assert (Xb.index // 100).min() >= 300
    Cb = Yb - pd.DataFrame({c: models[c].predict(Xb) for c in TARGETS}, index=Yb.index)
    print(f"{a.hyp}: trained on {len(Xa)} events (files 0-299), applied to {len(Xb)} held-out events (files 300-699)")
    idx = np.arange(len(Xb))
    def compare(i, label):
        m0, m1 = metrics(Yb, i), metrics(Cb, i)
        bs = {k: [] for k in m0}
        for _ in range(a.boot):
            j = RNG.choice(i, len(i))
            b0, b1 = metrics(Yb, j), metrics(Cb, j)
            for k in m0:
                bs[k].append(b1[k]/b0[k] if k != "abs_bias" else b1[k] - b0[k])
        row = {"bin": label, "n": len(i)}
        for k in m0:
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            row[k] = (m0[k], m1[k], lo, hi)
        return row
    allr = compare(idx, "all")
    print("  all events: " + "  ".join(f"{k} {v[0]:.2f} -> {v[1]:.2f} [{v[2]:.3f},{v[3]:.3f}]" for k, v in allr.items() if k not in ("bin", "n")))
    pb = [0, 200, 400, 600, 800, 1200, 3000] if a.hyp == "e-" else [0, 400, 600, 800, 1000, 1500, 3000]
    tb = [0, 300, 600, 1500, 1e9]
    rows, worse = [], []
    for plo, phi in zip(pb[:-1], pb[1:]):
        for tlo, thi in zip(tb[:-1], tb[1:]):
            i = np.flatnonzero((ptrue >= plo) & (ptrue < phi) & (twt >= tlo) & (twt < thi))
            if len(i) < 40:
                continue
            r = compare(i, f"p {plo}-{phi}, towall {tlo/100:.0f}-{min(thi/100, 99):.0f} m")
            rows.append(r)
            for k in ("p_core", "vtx68", "ang68"):
                if r[k][2] > 1.0:
                    worse.append(f"{r['bin']} {k} x{r[k][1]/r[k][0]:.2f} [{r[k][2]:.2f},{r[k][3]:.2f}]")
            if r["abs_bias"][2] > 0:
                worse.append(f"{r['bin']} |bias| {r['abs_bias'][0]:.2f} -> {r['abs_bias'][1]:.2f}%")
    print(f"  {len(rows)} (p, towall) bins with >= 40 events; ratios after/before:")
    for r in rows:
        print(f"    {r['bin']:30s} n={r['n']:4d}  p_core x{r['p_core'][1]/r['p_core'][0]:.2f}  vtx68 x{r['vtx68'][1]/r['vtx68'][0]:.2f}"
              f"  ang68 x{r['ang68'][1]/r['ang68'][0]:.2f}  |bias| {r['abs_bias'][0]:.2f} -> {r['abs_bias'][1]:.2f}%")
    print("  worse beyond CI:", worse if worse else "none")
    if a.hyp == "e-":
        v, lo, hi = allr["p_core"][1]/allr["p_core"][0], allr["p_core"][2], allr["p_core"][3]
        print(f"  e- p core ratio {v:.3f} [{lo:.3f},{hi:.3f}] -> {'PASS' if hi <= 0.90 and not worse else 'FAIL'} "
              f"(pre-reg: <= 0.90x with CI, no bin worse)")


if __name__ == "__main__":
    main()
