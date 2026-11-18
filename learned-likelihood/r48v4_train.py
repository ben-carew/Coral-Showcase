#!/usr/bin/env python
"""R48 v4: train one member of the per-PMT density ensemble on Coral's sub-event-0 hit sets.

Rows = every hit PMT + `--unhit` random unhit PMTs per event (weight n_unhit / kept; masked PMTs excluded),
weighted NLL at the true track. `--arch v2` reproduces the v2b network (data-size ablation); `--seed` sets the
initialisation, unhit draw and shuffle of one ensemble member. 5% of events held out for the validation loss.

    python r48v4_train.py --hits hit_sets.csv --pmts pmt_table.csv --truth truth.root --fits coral.root --out net4_e_s0.pt
"""
import argparse, time
import numpy as np, pandas as pd, torch, uproot
from r48_lib import DensityNet, DensityNet4, geometry, geometry4, pmt_logp

ap = argparse.ArgumentParser()
for k in ("hits", "pmts", "truth", "fits", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--arch", default="v4"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--unhit", type=int, default=1000); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=32768)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--max-events", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
rng = np.random.default_rng(48 + a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

pm = pd.read_csv(a.pmts).sort_values("tubeid").reset_index(drop=True)
tube_to_idx = pd.Series(np.arange(len(pm)), index=pm.tubeid.to_numpy())
pmt_pos = torch.tensor(pm[["x", "y", "z"]].to_numpy(), dtype=torch.float64, device=dev)
pmt_dir = torch.tensor(pm[["ox", "oy", "oz"]].to_numpy(), dtype=torch.float64, device=dev)
wall = (float(np.hypot(pm.x, pm.y).max()), float(pm.z.abs().max()))

tr = uproot.open(a.truth)["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
d = tr[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); d /= np.linalg.norm(d, axis=1)[:, None]
theta_all = pd.DataFrame(np.c_[tr.Time - tr.TriggerTime, tr[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float), d, tr.P],
                         index=tr.index)

t0 = time.time()
h = pd.read_csv(a.hits, on_bad_lines="skip", dtype={"event": "int64", "sub_event": "int16", "tubeid": "int32",
                                                     "q": "float32", "t": "float32"})
h = h[h.sub_event == 0]
fitted = np.unique(uproot.open(a.fits)["fits"].arrays(["event"], library="np")["event"])
events = np.intersect1d(np.intersect1d(h.event.unique(), theta_all.index), fitted)
if a.max_events:
    events = events[:a.max_events]
h = h[h.event.isin(events)]
ev_pos = pd.Series(np.arange(len(events)), index=events)
theta = torch.tensor(theta_all.loc[events].to_numpy(), dtype=torch.float64, device=dev)
cols = {k: [] for k in ("ev", "pmt", "hit", "lq", "t", "w")}
for ev, g in h.groupby("event", sort=False):
    hitm = g.q.notna().to_numpy()
    hp = tube_to_idx.loc[g.tubeid[hitm]].to_numpy()
    masked = tube_to_idx.reindex(g.tubeid[~hitm]).dropna().to_numpy().astype(int)
    free = np.ones(len(pm), bool); free[hp] = False; free[masked] = False
    un = np.flatnonzero(free)
    keep = rng.choice(un, min(a.unhit, len(un)), replace=False)
    e = ev_pos[ev]
    n_h, n_u = len(hp), len(keep)
    cols["ev"] += [np.full(n_h + n_u, e, np.int32)]
    cols["pmt"] += [hp.astype(np.int32), keep.astype(np.int32)]
    cols["hit"] += [np.ones(n_h, np.int8), np.zeros(n_u, np.int8)]
    cols["lq"] += [np.log(np.maximum(g.q[hitm].to_numpy(np.float32), 1e-3)), np.zeros(n_u, np.float32)]
    cols["t"] += [g.t[hitm].to_numpy(np.float32), np.zeros(n_u, np.float32)]
    cols["w"] += [np.ones(n_h, np.float32), np.full(n_u, len(un)/max(n_u, 1), np.float32)]
del h
X = {k: np.concatenate(v) for k, v in cols.items()}
del cols
n = len(X["ev"])
print(f"{len(events)} events, {n} rows ({int(X['hit'].sum())} hit), built in {time.time() - t0:.0f} s, device {dev}", flush=True)

val_ev = rng.random(len(events)) < 0.05
is_val = val_ev[X["ev"]]
T = {k: torch.from_numpy(v).to(dev) for k, v in X.items()}
del X

v4 = a.arch == "v4"
net = (DensityNet4() if v4 else DensityNet()).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
steps_per_epoch = -(-int((~is_val).sum()) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs*steps_per_epoch, pct_start=0.05)
tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)


def batch_logp(i):
    th, pp, pd_ = theta[T["ev"][i]], pmt_pos[T["pmt"][i]], pmt_dir[T["pmt"][i]]
    tt = T["t"][i].double()
    if v4:
        g, tp = geometry4(th, pp, pd_, wall)
        return net.logp(g, tp, T["hit"][i], T["lq"][i], tt)
    g, tp = geometry(th, pp, pd_, wall)
    return pmt_logp(net, g, tp, T["hit"][i], T["lq"][i], tt)


def run(indices, train):
    tot, wsum = 0.0, 0.0
    for b in range(0, len(indices), a.batch):
        i = torch.from_numpy(indices[b:b + a.batch]).to(dev)
        w = T["w"][i]
        with torch.set_grad_enabled(train):
            loss = -(batch_logp(i)*w).sum()/w.sum()
            if train:
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()*w.sum().item(); wsum += w.sum().item()
    return tot/wsum if wsum else float("nan")


for ep in range(a.epochs):
    t1 = time.time()
    rng.shuffle(tr_idx)
    ltr = run(tr_idx, True)
    with torch.no_grad():
        lva = run(va_idx, False)
    print(f"epoch {ep}: train {ltr:.5f}  val {lva:.5f}  ({time.time() - t1:.0f} s)", flush=True)
    torch.save({"state": net.state_dict(), "wall": wall, "arch": a.arch, "val": lva}, a.out)
print("saved", a.out)
