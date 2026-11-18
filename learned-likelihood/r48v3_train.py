#!/usr/bin/env python
"""R48: train the per-PMT density network (v2) on Coral's sub-event-0 hit sets (e- guns, files 0-599).

Rows = every hit PMT + `--unhit` random unhit PMTs per event (weight n_unhit / kept; masked PMTs excluded).
Each row is a positive (its event's true track) and a negative (another event's true track); weighted
binary cross-entropy. 5% of events held out for the validation loss.

    python r48_train.py --hits hit_sets.csv --pmts pmt_table.csv --truth truth_electron.root --out net_e.pt
"""
import argparse, time
import numpy as np, pandas as pd, torch, uproot
from r48_lib import DensityNet, geometry, pmt_logp
import coral_mu

ap = argparse.ArgumentParser()
ap.add_argument("--hits", required=True); ap.add_argument("--pmts", required=True)
ap.add_argument("--truth", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fits", required=True, help="merged Coral output: only events with a written e- fit (complete dump)")
ap.add_argument("--unhit", type=int, default=1000); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--batch", type=int, default=32768); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--max-events", type=int, default=0)
ap.add_argument("--coral-src", required=True); ap.add_argument("--coral-config", required=True); ap.add_argument("--infile", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
rng = np.random.default_rng(48)

pm = pd.read_csv(a.pmts).sort_values("tubeid").reset_index(drop=True)
tube_to_idx = pd.Series(np.arange(len(pm)), index=pm.tubeid.to_numpy())
pmt_pos = torch.tensor(pm[["x", "y", "z"]].to_numpy(), dtype=torch.float64)
pmt_dir = torch.tensor(pm[["ox", "oy", "oz"]].to_numpy(), dtype=torch.float64)
wall = (float(np.hypot(pm.x, pm.y).max()), float(pm.z.abs().max()))

tr = uproot.open(a.truth)["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
d = tr[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); d /= np.linalg.norm(d, axis=1)[:, None]
theta_all = pd.DataFrame(np.c_[tr.Time - tr.TriggerTime, tr[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float), d, tr.P],
                         index=tr.index)

t0 = time.time()
h = pd.read_csv(a.hits, on_bad_lines="skip")
h = h[h.sub_event == 0]
fitted = np.unique(uproot.open(a.fits)["fits"].arrays(["event"], library="np")["event"])
events = np.intersect1d(np.intersect1d(h.event.unique(), theta_all.index), fitted)
if a.max_events:
    events = events[:a.max_events]
h = h[h.event.isin(events)]
ev_pos = pd.Series(np.arange(len(events)), index=events)
theta = torch.tensor(theta_all.loc[events].to_numpy(), dtype=torch.float64)
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
    cols["ev"] += [np.full(n_h + n_u, e)]
    cols["pmt"] += [hp, keep]
    cols["hit"] += [np.ones(n_h), np.zeros(n_u)]
    cols["lq"] += [np.log(np.maximum(g.q[hitm].to_numpy(float), 1e-3)), np.zeros(n_u)]
    cols["t"] += [g.t[hitm].to_numpy(float), np.zeros(n_u)]
    cols["w"] += [np.ones(n_h), np.full(n_u, len(un)/max(n_u, 1))]
X = {k: np.concatenate(v) for k, v in cols.items()}
# Coral's predicted direct light at the true track, for every row's PMT
mu_fn = coral_mu.init(a.coral_src, a.coral_config, a.infile)
pos_np, ori_np, loc_np = pm[["x", "y", "z"]].to_numpy(float), pm[["ox", "oy", "oz"]].to_numpy(float), pm["Loc"].to_numpy(np.int64)
th_np = theta.numpy()
X["lmu"] = np.zeros(len(X["ev"]))
starts = np.r_[0, np.flatnonzero(np.diff(X["ev"])) + 1]
for s0, s1 in zip(starts, np.r_[starts[1:], len(X["ev"])]):
    th = th_np[X["ev"][s0]]
    x7 = [th[0], th[1], th[2], th[3], np.arccos(np.clip(th[6], -1, 1)), np.arctan2(th[5], th[4]) % (2*np.pi), th[7]]
    X["lmu"][s0:s1] = np.log(mu_fn(x7, pos_np, ori_np, loc_np)[X["pmt"][s0:s1]] + 1e-3)
print(f"Coral light computed, median log mu of hit rows {np.median(X['lmu'][X['hit'] == 1]):.2f}", flush=True)
n = len(X["ev"])
print(f"{len(events)} events, {n} rows ({int(X['hit'].sum())} hit), built in {time.time() - t0:.0f} s", flush=True)

val_ev = rng.random(len(events)) < 0.05
is_val = val_ev[X["ev"]]
T = {k: torch.tensor(v) for k, v in X.items()}


def batch_geo(idx):
    g, tp = geometry(theta[T["ev"][idx]], pmt_pos[T["pmt"][idx]], pmt_dir[T["pmt"][idx]], wall)
    return torch.cat([g, T["lmu"][idx].float()[:, None]], 1), tp


net = DensityNet(n_in=11)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)


def run(indices, train):
    tot, wsum = 0.0, 0.0
    for b in range(0, len(indices), a.batch):
        i = torch.tensor(indices[b:b + a.batch])
        g, tp = batch_geo(i)
        w = T["w"][i].float()
        with torch.set_grad_enabled(train):
            loss = -(pmt_logp(net, g, tp, T["hit"][i], T["lq"][i], T["t"][i].double())*w).sum()/w.sum()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*w.sum().item(); wsum += w.sum().item()
    return tot/wsum if wsum else float("nan")


for ep in range(a.epochs):
    t1 = time.time()
    rng.shuffle(tr_idx)
    ltr = run(tr_idx, True)
    lva = run(va_idx, False)
    sched.step()
    print(f"epoch {ep}: train {ltr:.5f}  val {lva:.5f}  ({time.time() - t1:.0f} s)", flush=True)
    torch.save({"state": net.state_dict(), "wall": wall}, a.out)
print("saved", a.out)
