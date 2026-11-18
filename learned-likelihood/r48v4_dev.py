#!/usr/bin/env python
"""R48 round 2, DEV (files 700-739, events 0-24): network-alone fits vs truth, paired.

    python r48v4_dev.py --pairs lbfgs=v2b:coral,nm=v2b:nm          # L-BFGS vs Nelder-Mead (same fit files)
    python r48v4_dev.py --pairs v2b=v2b:coral,v4=v4:coral,v2x4=v2x4:coral
Each pair = label=<fit tag>:<column prefix>; the first is the reference.
"""
import argparse, glob
import numpy as np, pandas as pd, uproot
from r54n import METRICS, compare, unit, gcore

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--dir", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn")
ap.add_argument("--events", default="0-24")
a = ap.parse_args()
lo, hi = (int(x) for x in a.events.split("-"))
for smp, sample in (("e", "electron"), ("mu", "muon")):
    tr = uproot.open(f"{a.dir}/truth_{sample}_v1.12.29.root")["Tracks"].arrays(library="pd")
    tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
    R, tags = {}, {}
    for spec in a.pairs.split(","):
        lab, rest = spec.split("="); tag, pre = rest.split(":")
        f = pd.concat([pd.read_csv(x) for x in glob.glob(f"{a.dir}/fits/{tag}_r54n2_test_{smp}_{smp}/*.csv")]).set_index("event")
        f = f[(f.index % 100 >= lo) & (f.index % 100 <= hi)]
        R[lab] = (f, pre); tags[lab] = tag
    ev = None
    for f, _ in R.values():
        ev = f.index if ev is None else ev.intersection(f.index)
    ev = ev.intersection(tr.index)
    T = tr.loc[ev]
    td = T[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); td /= np.linalg.norm(td, axis=1)[:, None]
    tv, tp = T[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float), T.P.to_numpy(float)
    res = {}
    for lab, (f, pre) in R.items():
        g = f.loc[ev]
        d = unit(g[f"{pre}_theta"].to_numpy(), g[f"{pre}_phi"].to_numpy())
        res[lab] = dict(p=100*(g[f"{pre}_p"].to_numpy()/tp - 1),
                        vtx=np.linalg.norm(g[[f"{pre}_x", f"{pre}_y", f"{pre}_z"]].to_numpy() - tv, axis=1),
                        ang=np.degrees(np.arccos(np.clip((d*td).sum(1), -1, 1))))
    labs = list(res)
    idx = np.flatnonzero(np.all([res[l]["ang"] < 20 for l in labs], axis=0))
    print(f"{smp}: {len(ev)} DEV events, {len(idx)} converged in all")
    for l in labs:
        f, pre = R[l]
        s = f.loc[ev, f"{pre}_s"] if f"{pre}_s" in f else pd.Series(np.nan)
        print(f"  {l:8s} " + "  ".join(f"{k} {fn(res[l], idx):.3f}" for k, fn in METRICS.items()) + f"  s/fit {s.median():.1f}")
    for l in labs[1:]:
        print(f"  {l} vs {labs[0]}: " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in compare(res[l], res[labs[0]], idx, 1000).items()))
        f0, p0 = R[labs[0]]; f1, p1 = R[l]
        if tags[l] == tags[labs[0]]:
            dn = (f1.loc[ev, f"{p1}_nll"] - f0.loc[ev, f"{p0}_nll"]).to_numpy()
            print(f"    dnLL ({l} - {labs[0]}): median {np.median(dn):+.3f}, |dnLL| median {np.median(np.abs(dn)):.3f}, "
                  f"> +1: {100*np.mean(dn > 1):.1f}%, < -1: {100*np.mean(dn < -1):.1f}%")
