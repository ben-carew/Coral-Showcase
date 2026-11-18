#!/usr/bin/env python
"""R48: fit held-out e- events with the learned per-PMT likelihood (Nelder-Mead over (t, x, y, z, theta, phi, p)
on -sum of the network logits over every non-masked PMT), from Coral's fitted point and/or the true track.

    python r48_fit.py --hits hit_sets.csv --pmts pmt_table.csv --truth truth.root --fits coral.root \
        --model net_e.pt --part 0 --nparts 50 --out fits_0.csv
"""
import argparse, time
import numpy as np, pandas as pd, torch, uproot
from scipy.optimize import minimize
from r48_lib import Net, features, theta_from_x

ap = argparse.ArgumentParser()
for k in ("hits", "pmts", "truth", "fits", "model", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--seeds", default="coral,truth"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)

ck = torch.load(a.model, weights_only=False)
net = Net(); net.load_state_dict(ck["state"]); net.eval()
wall = ck["wall"]
pm = pd.read_csv(a.pmts).sort_values("tubeid").reset_index(drop=True)
tube_to_idx = pd.Series(np.arange(len(pm)), index=pm.tubeid.to_numpy())
P_all = torch.tensor(pm[["x", "y", "z"]].to_numpy(), dtype=torch.float64)
D_all = torch.tensor(pm[["ox", "oy", "oz"]].to_numpy(), dtype=torch.float64)

tr = uproot.open(a.truth)["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
f = pd.DataFrame(uproot.open(a.fits)["fits"].arrays(["event", "sub_event", "particle_code", "t", "x", "y", "z", "px", "py", "pz", "p"], library="np"))
f = f[(f.sub_event == 0) & (f.particle_code == 0)].drop_duplicates("event").set_index("event")
h = pd.read_csv(a.hits, on_bad_lines="skip")
h = h[h.sub_event == 0]
events = np.intersect1d(np.intersect1d(h.event.unique(), f.index), tr.index)
events = events[a.part::a.nparts]

STEPS = np.array([5.0, 50.0, 50.0, 50.0, 0.05, 0.05, 0.0])


def to_x(r):
    n = np.sqrt(r.px**2 + r.py**2 + r.pz**2)
    return np.array([r.t, r.x, r.y, r.z, np.arccos(r.pz/n), np.arctan2(r.py, r.px) % (2*np.pi), r.p])


rows = []
for ev in events:
    g = h[h.event == ev]
    hitm = g.q.notna().to_numpy()
    hp = tube_to_idx.loc[g.tubeid[hitm]].to_numpy()
    masked = tube_to_idx.reindex(g.tubeid[~hitm]).dropna().to_numpy().astype(int)
    use = np.ones(len(pm), bool); use[masked] = False
    hit = np.zeros(len(pm)); lq = np.zeros(len(pm)); tt = np.zeros(len(pm))
    hit[hp] = 1; lq[hp] = np.log(np.maximum(g.q[hitm].to_numpy(float), 1e-3)); tt[hp] = g.t[hitm].to_numpy(float)
    idx = np.flatnonzero(use)
    Pp, Dp = P_all[idx], D_all[idx]
    Ht, Lq, Tt = torch.tensor(hit[idx]), torch.tensor(lq[idx]), torch.tensor(tt[idx])
    nev = [0]

    def nll(x):
        if not (0.0 <= x[4] <= np.pi) or x[6] <= 1.0:
            return 1e12
        nev[0] += 1
        th = torch.tensor(theta_from_x(x), dtype=torch.float64).expand(len(idx), 8)
        with torch.no_grad():
            return -float(net(features(th, Pp, Dp, Ht, Lq, Tt, wall)).sum())

    tp = tr.loc[ev]
    dt = np.array([tp.Dir_x0, tp.Dir_x1, tp.Dir_x2], float); dt /= np.linalg.norm(dt)
    x_true = np.array([tp.Time - tp.TriggerTime, tp.Start_x0, tp.Start_x1, tp.Start_x2,
                       np.arccos(dt[2]), np.arctan2(dt[1], dt[0]) % (2*np.pi), tp.P])
    x_coral = to_x(f.loc[ev])
    out = {"event": ev, "nll_coral_pt": nll(x_coral), "nll_true_pt": nll(x_true)}
    for name, x0 in (("coral", x_coral), ("truth", x_true)):
        if name not in a.seeds:
            continue
        t1 = time.time(); nev[0] = 0
        steps = STEPS.copy(); steps[6] = 0.1*x0[6]
        sim = np.vstack([x0] + [x0 + np.eye(7)[k]*steps[k] for k in range(7)])
        r = minimize(nll, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=sim, adaptive=True, maxiter=3000, xatol=0.5, fatol=0.1))
        out.update({f"{name}_{k}": v for k, v in zip(("t", "x", "y", "z", "theta", "phi", "p"), r.x)})
        out[f"{name}_nll"], out[f"{name}_nfev"], out[f"{name}_s"] = r.fun, nev[0], time.time() - t1
    out.update({f"coralfit_{k}": v for k, v in zip(("t", "x", "y", "z", "theta", "phi", "p"), x_coral)})
    rows.append(out)
    print(f"event {ev}: nll coral pt {out['nll_coral_pt']:.1f} true pt {out['nll_true_pt']:.1f}"
          + (f" -> nn fit {out['coral_nll']:.1f} ({out['coral_nfev']} evals, {out['coral_s']:.0f} s)" if "coral_nll" in out else ""),
          flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
