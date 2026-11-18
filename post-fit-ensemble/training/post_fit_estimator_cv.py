#!/usr/bin/env python
"""Feasibility of a learned post-fit estimator for Coral's single-ring fits (no new fit, no truth inputs).

For every event both hypotheses are already fitted (for PID). Their disagreement (momentum ratio, vertex
shift, nLL difference) and the fitted geometry are used to predict the signed error of the hypothesis'
momentum, vertex (along / two transverse) and direction; the prediction is subtracted. Gradient boosting,
5-fold cross-validation with whole input files held out. No momentum magnitude is an input (it would let
the model learn the training spectrum); observed totals enter only as ratios to the fitted momentum.

    python post_fit_estimator_cv.py --hyp e- --spec sum10k_e      # or --hyp mu- --spec sum10k_mu
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Software/Sextant/src"))
from sextant import Arm, Truth, analysis                          # noqa: E402
from sextant.io.geometry import dwall as gdwall, towall as gtowall  # noqa: E402
from sextant.metrics.stats import gauss_sigma                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hyp", default="e-")
ap.add_argument("--spec", default="sum10k_e")
ap.add_argument("--production", default="v1.12.29")
a = ap.parse_args()
sample = "electron" if a.hyp == "e-" else "muon"
reco = lambda algo: ROOT / f"Output/{'Coral' if algo == 'coral' else 'fiTQun'}/Reco/{a.spec}/{algo}_{sample}_{algo}.root"
truth = Truth.from_file(ROOT / f"Output/Truth/truth_{sample}_{a.production}.root")
arm = Arm.from_file("C", reco("coral"))
res = analysis.compute_residuals([arm], truth, hypothesis=a.hyp)["C"].drop_duplicates("event").set_index("event")
fq = analysis.compute_residuals([Arm.from_file("F", reco("fitqun"), production=a.production, sample=sample)], truth,
                                hypothesis=a.hyp)["F"].drop_duplicates("event").set_index("event")
f = arm.fits[(arm.fits.stage == -1) & (arm.fits.sub_event == 0)]
hyps = {h: f[f.hypothesis == h].drop_duplicates("event").set_index("event") for h in ("e-", "mu-")}
own, oth = hyps[a.hyp], hyps["mu-" if a.hyp == "e-" else "e-"]
tp = truth.primary.drop_duplicates("event").set_index("event")
ev = own.index.intersection(oth.index).intersection(res.index).intersection(tp.index)
o, t, T, r = own.loc[ev], oth.loc[ev], tp.loc[ev], res.loc[ev]

# fitted-track frame
d = np.c_[o.dx, o.dy, o.dz]; d /= np.linalg.norm(d, axis=1)[:, None]
ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0, 0, 1.]], [[1., 0, 0]])
e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(d, e1)
X = pd.DataFrame(index=ev)
X["dlogp"] = np.log(t.p / o.p)
X["dwall"] = gdwall(o.x, o.y, o.z); X["towall"] = gtowall(o.x, o.y, o.z, o.dx, o.dy, o.dz)
X["r"] = np.hypot(o.x, o.y); X["z"] = o.z; X["cz"] = d[:, 2]; X["crad"] = (o.x*d[:, 0] + o.y*d[:, 1])/np.maximum(X.r, 1)
X["dnll"] = t.nll - o.nll; X["nll"] = o.nll
dvh = np.c_[t.x - o.x, t.y - o.y, t.z - o.z]
X["dv_along"] = (dvh*d).sum(1); X["dv_u"] = (dvh*e1).sum(1); X["dv_v"] = (dvh*e2).sum(1); X["dt"] = t.t - o.t
dd = np.c_[t.dx, t.dy, t.dz]; X["dd_u"] = np.degrees((dd*e1).sum(1)); X["dd_v"] = np.degrees((dd*e2).sum(1))
X["e1z"] = e1[:, 2]; X["e2z"] = e2[:, 2]
X["e1rad"] = (e1[:, 0]*o.x + e1[:, 1]*o.y)/np.maximum(X.r, 1); X["e2rad"] = (e2[:, 0]*o.x + e2[:, 1]*o.y)/np.maximum(X.r, 1)
X["q_per_p"] = np.log(r.totq.to_numpy()/o.p.to_numpy()); X["hits_per_p"] = np.log(r.nhits.to_numpy()/o.p.to_numpy())

# signed targets in the fitted frame
td = np.c_[T.dx, T.dy, T.dz]; td /= np.linalg.norm(td, axis=1)[:, None]
dv = np.c_[o.x - T.x, o.y - T.y, o.z - T.z]
Y = pd.DataFrame({"logp": np.log1p(r.p_res_pct.to_numpy()/100), "along": (dv*d).sum(1), "u": (dv*e1).sum(1),
                  "v": (dv*e2).sum(1), "a1": np.degrees((td*e1).sum(1)), "a2": np.degrees((td*e2).sum(1))}, index=ev)
ok = (r.angle_deg < 20).to_numpy() & np.isfinite(Y.logp.to_numpy())
X, Y = X[ok], Y[ok]
groups = X.index.to_numpy() // 100
P = pd.DataFrame(0.0, index=Y.index, columns=Y.columns)
for c in Y.columns:
    for trn, te in GroupKFold(5).split(X, Y[c], groups):
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.04, min_samples_leaf=50,
                                          loss="absolute_error").fit(X.iloc[trn], Y[c].iloc[trn])
        P.iloc[te, P.columns.get_loc(c)] = m.predict(X.iloc[te])
C = Y - P

core = lambda x: 100*gauss_sigma(np.asarray(x))
def half68(x):
    lo, hi = np.percentile(np.asarray(x) - np.median(x), [16, 84]); return 100*(hi - lo)/2
p68 = lambda x: np.percentile(np.abs(x), 68)
ptrue = T.p.reindex(X.index).to_numpy()
twt = gtowall(T.x, T.y, T.z, T.dx, T.dy, T.dz)[ok]
print(f"{a.hyp} {a.spec}: {len(X)} events converged (< 20 deg), {len(set(groups))} files")
for name, sel in (("all", np.ones(len(X), bool)), ("true towall > 15 m", twt > 1500)):
    vert0 = np.sqrt(Y.along**2 + Y.u**2 + Y.v**2)[sel]; vert1 = np.sqrt(C.along**2 + C.u**2 + C.v**2)[sel]
    print(f"  {name}: momentum core {core(Y.logp[sel]):.2f} -> {core(C.logp[sel]):.2f}%, iqr68/2 {half68(Y.logp[sel]):.2f} -> {half68(C.logp[sel]):.2f}% | "
          f"vertex p68 {p68(vert0):.1f} -> {p68(vert1):.1f} cm, transverse {p68(np.hypot(Y.u, Y.v)[sel]):.1f} -> {p68(np.hypot(C.u, C.v)[sel]):.1f} | "
          f"direction {p68(np.hypot(Y.a1, Y.a2)[sel]):.2f} -> {p68(np.hypot(C.a1, C.a2)[sel]):.2f} deg")
bins = [0, 200, 400, 600, 800, 1200, 3000] if a.hyp == "e-" else [0, 400, 600, 800, 1000, 1500, 3000]
print("  momentum core within true-p bins:", "  ".join(
    f"{lo}-{hi}: {core(Y.logp[(ptrue >= lo) & (ptrue < hi)]):.2f}->{core(C.logp[(ptrue >= lo) & (ptrue < hi)]):.2f}"
    for lo, hi in zip(bins[:-1], bins[1:])))
fqc = fq[(fq.angle_deg < 20) & fq.p_res_pct.notna()]
same = X.index.isin(fqc.index)
yf = np.log1p(fqc.loc[X.index[same], "p_res_pct"].to_numpy()/100)
for name, sel in (("all", np.ones(same.sum(), bool)), ("true towall > 15 m", twt[same] > 1500)):
    print(f"  same events as fiTQun, {name}: momentum core Coral {core(Y.logp[same][sel]):.2f}, Coral + estimator "
          f"{core(C.logp[same][sel]):.2f}, fiTQun {core(yf[sel]):.2f}%")
