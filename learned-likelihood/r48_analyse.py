#!/usr/bin/env python
"""R48: learned-likelihood fits (r48_fit.py output) vs Coral's own e- fit, paired on the same held-out events.

Residuals vs truth: momentum (scale-removed Gaussian core and half-68), vertex p68, transverse p68, angle p68,
median bias; paired bootstrap of NN/Coral ratios. Step-1 PASS (roadmap R48): NN-fit e- p core AND vertex p68
narrower than Coral beyond CI. Diagnostics: the truth-seeded NN fit, and how often the network prefers the
true track to Coral's point.

    python r48_analyse.py --fits 'fits_*.csv' --truth truth_electron_v1.12.29.root
"""
import argparse, glob
import numpy as np, pandas as pd, uproot

ap = argparse.ArgumentParser()
ap.add_argument("--fits", required=True); ap.add_argument("--truth", required=True)
a = ap.parse_args()
rng = np.random.default_rng(480)
d = pd.concat([pd.read_csv(f) for f in glob.glob(a.fits)]).drop_duplicates("event").set_index("event")
tr = uproot.open(a.truth)["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum").loc[d.index]
td = tr[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); td /= np.linalg.norm(td, axis=1)[:, None]
tv = tr[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float)


def resid(pfx):
    st = np.sin(d[f"{pfx}_theta"])
    u = np.c_[st*np.cos(d[f"{pfx}_phi"]), st*np.sin(d[f"{pfx}_phi"]), np.cos(d[f"{pfx}_theta"])]
    dv = d[[f"{pfx}_x", f"{pfx}_y", f"{pfx}_z"]].to_numpy() - tv
    along = (dv*td).sum(1)
    return dict(p=100*(d[f"{pfx}_p"].to_numpy()/tr.P.to_numpy() - 1),
                vtx=np.linalg.norm(dv, axis=1), trans=np.sqrt(np.maximum((dv**2).sum(1) - along**2, 0)),
                along=along, ang=np.degrees(np.arccos(np.clip((u*td).sum(1), -1, 1))))


def gcore(x):
    x = np.asarray(x); m, s = np.median(x), 1.4826*np.median(np.abs(x - np.median(x)))
    for _ in range(5):
        sel = np.abs(x - m) < 2*s
        m, s = x[sel].mean(), x[sel].std()
    return s/(1 + m/100)


def half68(x):
    lo, m, hi = np.percentile(x, [16, 50, 84]); return (hi - lo)/2/(1 + m/100)


R = {"coral": resid("coralfit"), "nn": resid("coral")}
if "truth_p" in d:
    R["nn_truthseed"] = resid("truth")
ok = np.all([r["ang"] < 20 for r in R.values()], axis=0)
idx = np.flatnonzero(ok)
S = {"p_core": lambda r, i: gcore(r["p"][i]), "p_half68": lambda r, i: half68(r["p"][i]),
     "vtx68": lambda r, i: np.percentile(r["vtx"][i], 68), "trans68": lambda r, i: np.percentile(r["trans"][i], 68),
     "ang68": lambda r, i: np.percentile(r["ang"][i], 68)}
print(f"{len(d)} events fitted, {len(idx)} converged (< 20 deg) in every fit")
print("coral:        " + "  ".join(f"{k} {f(R['coral'], idx):.3f}" for k, f in S.items())
      + f"  p bias {np.median(R['coral']['p'][idx]):+.2f}%  along {np.median(R['coral']['along'][idx]):+.1f} cm")
res = {}
for name in [k for k in R if k != "coral"]:
    line = []
    for k, f in S.items():
        v = f(R[name], idx)/f(R["coral"], idx) - 1
        bs = [f(R[name], i)/f(R["coral"], i) - 1 for i in (rng.choice(idx, len(idx)) for _ in range(500))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[(name, k)] = (v, lo, hi)
        line.append(f"{k} {100*v:+.1f}% [{100*lo:+.1f},{100*hi:+.1f}]")
    print(f"{name:13s} " + "  ".join(line) + f"  p bias {np.median(R[name]['p'][idx]):+.2f}%  along {np.median(R[name]['along'][idx]):+.1f} cm")
print(f"network prefers the true track to Coral's point in {100*np.mean(d.nll_true_pt < d.nll_coral_pt):.1f}% of events"
      f" (median nLL(true) - nLL(coral) {np.median(d.nll_true_pt - d.nll_coral_pt):+.1f})")
if "coral_nll" in d:
    print(f"NN fit from Coral's point: median nLL gain {np.median(d.nll_coral_pt - d.coral_nll):.1f}, evaluations {d.coral_nfev.median():.0f}, {d.coral_s.median():.0f} s/event")
p, v = res[("nn", "p_core")], res[("nn", "vtx68")]
print("STEP-1 PASS" if p[2] < 0 and v[2] < 0 else "STEP-1 FAIL", "(pre-reg: NN-fit e- p core AND vertex p68 narrower than Coral beyond CI)")
