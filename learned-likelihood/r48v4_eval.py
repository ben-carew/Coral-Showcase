#!/usr/bin/env python
"""R48 round 2: held-out per-event -log L at the TRUE track (every unmasked PMT) for several networks / ensembles,
on DEV (files 700-739, events 0-24): a fit-free, like-for-like comparison of density models.

    python r48v4_eval.py --smp e --models v2b=net2b_e.pt,s0=net4_e_s0.pt,ens=net4_e_s0.pt+net4_e_s1.pt
"""
import argparse, glob
import numpy as np, pandas as pd, torch, uproot
from r48_lib import Ensemble

ap = argparse.ArgumentParser()
ap.add_argument("--smp", default="e"); ap.add_argument("--models", required=True)
ap.add_argument("--dir", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn"); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sample = "electron" if a.smp == "e" else "muon"
models = {lab: Ensemble([f"{a.dir}/{p}" for p in paths.split("+")]) for lab, paths in (m.split("=") for m in a.models.split(","))}
pm = pd.read_csv(f"{a.dir}/pmt_table.csv").sort_values("tubeid").reset_index(drop=True)
tube_to_idx = pd.Series(np.arange(len(pm)), index=pm.tubeid.to_numpy())
P_all = torch.tensor(pm[["x", "y", "z"]].to_numpy(), dtype=torch.float64)
D_all = torch.tensor(pm[["ox", "oy", "oz"]].to_numpy(), dtype=torch.float64)
tr = uproot.open(f"{a.dir}/truth_{sample}_v1.12.29.root")["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
h = pd.concat([pd.DataFrame(dict(np.load(f))) for f in glob.glob(f"{a.dir}/parts/r54n2_test_{a.smp}/hits_*.npz")])
h = h[(h.event % 100) < 25]
rows = []
for ev, g in h.groupby("event"):
    if ev not in tr.index:
        continue
    hitm = g.q.notna().to_numpy()
    hp = tube_to_idx.loc[g.tubeid[hitm]].to_numpy()
    masked = tube_to_idx.reindex(g.tubeid[~hitm]).dropna().to_numpy().astype(int)
    use = np.ones(len(pm), bool); use[masked] = False
    hit = np.zeros(len(pm)); lq = np.zeros(len(pm)); tt = np.zeros(len(pm))
    hit[hp] = 1; lq[hp] = np.log(np.maximum(g.q[hitm].to_numpy(float), 1e-3)); tt[hp] = g.t[hitm].to_numpy(float)
    idx = np.flatnonzero(use)
    tp = tr.loc[ev]
    d = np.array([tp.Dir_x0, tp.Dir_x1, tp.Dir_x2], float); d /= np.linalg.norm(d)
    th = torch.tensor([[tp.Time - tp.TriggerTime, tp.Start_x0, tp.Start_x1, tp.Start_x2, *d, tp.P]], dtype=torch.float64).expand(len(idx), 8)
    with torch.no_grad():
        rows.append({"event": ev, **{lab: -float(m.logp(th, P_all[idx], D_all[idx], torch.tensor(hit[idx]),
                                                         torch.tensor(lq[idx]), torch.tensor(tt[idx])).sum()) for lab, m in models.items()}})
df = pd.DataFrame(rows).set_index("event")
ref = df.columns[0]
rng = np.random.default_rng(1)
print(f"{a.smp}: {len(df)} DEV events, -log L at the true track per event (lower = better), vs {ref}:")
for c in df.columns:
    dn = (df[c] - df[ref]).to_numpy()
    bs = [rng.choice(dn, len(dn)).mean() for _ in range(1000)]
    print(f"  {c:10s} mean {df[c].mean():9.1f}   diff {dn.mean():+7.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]  better in {100*np.mean(dn < 0):.0f}% of events")
