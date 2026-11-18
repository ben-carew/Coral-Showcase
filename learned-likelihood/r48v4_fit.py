#!/usr/bin/env python
"""R48 v4: fit events with a learned per-PMT density ensemble, seeded at Coral's fit of one hypothesis.

--method lbfgs: autograd L-BFGS in the seed's frame (t, along, u, v, two tangent angles, log p), then the 7x7
Hessian there -> h_* = sqrt(diag(H^-1)) in physical units (ns, cm, cm, cm, deg, deg, fraction).
--method nm: the v2 Nelder-Mead.  --compare-nm also runs NM and writes it as nm_* (L-BFGS validation).
Output columns follow r48v2_fit.py (coral_* = the network fit seeded at Coral, coralfit_* = Coral's fit).

    python r48v4_fit.py --hits parts/ --pmts pmt_table.csv --truth truth.root --fits coral.root \
        --models net4_e_s0.pt,net4_e_s1.pt --code 0 --part 0 --nparts 200 --out fits_0.csv
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, uproot
from scipy.optimize import minimize
from r48_lib import Ensemble, theta_from_x

ap = argparse.ArgumentParser()
for k in ("hits", "pmts", "truth", "fits", "models", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--code", type=int, default=0, help="Coral fit used as the seed: 0 = e-, 1 = mu-")
ap.add_argument("--method", default="lbfgs"); ap.add_argument("--compare-nm", action="store_true")
ap.add_argument("--max-events", type=int, default=0); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)

ens = Ensemble(a.models.split(","))
pm = pd.read_csv(a.pmts).sort_values("tubeid").reset_index(drop=True)
tube_to_idx = pd.Series(np.arange(len(pm)), index=pm.tubeid.to_numpy())
P_all = torch.tensor(pm[["x", "y", "z"]].to_numpy(), dtype=torch.float64)
D_all = torch.tensor(pm[["ox", "oy", "oz"]].to_numpy(), dtype=torch.float64)

tr = uproot.open(a.truth)["Tracks"].arrays(library="pd")
tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
f = pd.DataFrame(uproot.open(a.fits)["fits"].arrays(["event", "sub_event", "particle_code", "t", "x", "y", "z", "px", "py", "pz", "p"], library="np"))
f = f[(f.sub_event == 0) & (f.particle_code == a.code)].drop_duplicates("event").set_index("event")
if Path(a.hits).is_dir():
    h = pd.DataFrame(dict(np.load(Path(a.hits)/f"hits_{a.part}.npz")))
    events = np.intersect1d(np.intersect1d(h.event.unique(), f.index), tr.index)
else:
    h = pd.read_csv(a.hits, on_bad_lines="skip")
    h = h[h.sub_event == 0]
    events = np.intersect1d(np.intersect1d(h.event.unique(), f.index), tr.index)[a.part::a.nparts]
if a.max_events:
    events = events[:a.max_events]
hg = dict(tuple(h[h.event.isin(events)].groupby("event")))

STEPS = np.array([5.0, 50.0, 50.0, 50.0, 0.05, 0.05, 0.0])
SCALE = torch.tensor([1.0, 10.0, 10.0, 10.0, 0.01, 0.01, 0.01], dtype=torch.float64)
NAMES = ("t", "x", "y", "z", "theta", "phi", "p")


def to_x(r):
    n = np.sqrt(r.px**2 + r.py**2 + r.pz**2)
    return np.array([r.t, r.x, r.y, r.z, np.arccos(r.pz/n), np.arctan2(r.py, r.px) % (2*np.pi), r.p])


def frame(d):
    e1 = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


rows = pd.read_csv(a.out).to_dict("records") if Path(a.out).exists() else []
done = {r["event"] for r in rows}
for ev in [e for e in events if e not in done]:
    g = hg[ev]
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

    def nll_theta(th):
        return -ens.logp(th.expand(len(idx), 8), Pp, Dp, Ht, Lq, Tt).sum()

    def nll(x):
        if not (0.0 <= x[4] <= np.pi) or x[6] <= 1.0:
            return 1e12
        nev[0] += 1
        with torch.no_grad():
            return float(nll_theta(torch.tensor(theta_from_x(x), dtype=torch.float64)[None]))

    tp = tr.loc[ev]
    dt = np.array([tp.Dir_x0, tp.Dir_x1, tp.Dir_x2], float); dt /= np.linalg.norm(dt)
    x_true = np.array([tp.Time - tp.TriggerTime, tp.Start_x0, tp.Start_x1, tp.Start_x2,
                       np.arccos(dt[2]), np.arctan2(dt[1], dt[0]) % (2*np.pi), tp.P])
    x0 = to_x(f.loc[ev])
    out = {"event": ev, "nll_coral_pt": nll(x0), "nll_true_pt": nll(x_true)}
    out.update({f"coralfit_{k}": v for k, v in zip(NAMES, x0)})
    res = {}
    if a.method == "nm" or a.compare_nm:
        t1 = time.time(); nev[0] = 0
        steps = STEPS.copy(); steps[6] = 0.1*x0[6]
        sim = np.vstack([x0] + [x0 + np.eye(7)[k]*steps[k] for k in range(7)])
        r = minimize(nll, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=sim, adaptive=True, maxiter=3000, xatol=0.5, fatol=0.1))
        res["nm"] = (r.x, r.fun, nev[0], time.time() - t1, None)
    if a.method == "lbfgs":
        t1 = time.time()
        d0 = theta_from_x(x0)[4:7]; e1, e2 = frame(d0)
        B = torch.tensor(np.stack([d0, e1, e2]), dtype=torch.float64)
        X0 = torch.tensor(x0, dtype=torch.float64)

        def theta_of(z):
            u = z*SCALE
            v = X0[1:4] + u[1:4] @ B
            d = B[0] + u[4]*B[1] + u[5]*B[2]
            d = d/torch.linalg.norm(d)
            return torch.cat([(X0[0] + u[0])[None], v, d, (X0[6]*torch.exp(u[6]))[None]])[None]

        z = torch.zeros(7, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([z], lr=1.0, max_iter=200, tolerance_grad=1e-3, tolerance_change=1e-3,
                                history_size=20, line_search_fn="strong_wolfe")
        ncall = [0]

        def closure():
            opt.zero_grad(); ncall[0] += 1
            loss = nll_theta(theta_of(z)); loss.backward()
            return loss

        opt.step(closure)
        zf = z.detach().clone()
        Hs = torch.autograd.functional.hessian(lambda q: nll_theta(theta_of(q)), zf)
        th = theta_of(zf)[0].numpy()
        xf = np.array([th[0], th[1], th[2], th[3], np.arccos(np.clip(th[6], -1, 1)), np.arctan2(th[5], th[4]) % (2*np.pi), th[7]])
        Hn, sig, mineig = Hs.numpy(), np.full(7, np.nan), np.nan
        if np.isfinite(Hn).all():
            try:
                sig = np.sqrt(np.clip(np.diag(np.linalg.inv(Hn)), 0, None))*SCALE.numpy()
                mineig = float(np.linalg.eigvalsh(Hn).min())
            except np.linalg.LinAlgError:
                pass
        sig[4:6] = np.degrees(sig[4:6])
        hess = {f"h_{k}": s for k, s in zip(("t", "along", "u", "v", "a", "b", "lp"), sig)}
        hess["h_mineig"] = mineig
        res["lbfgs"] = (xf, float(nll(xf)), ncall[0], time.time() - t1, hess)
    for key, (x, fun, ne, s, hess) in res.items():
        pre = "coral" if key == a.method else key
        out.update({f"{pre}_{k}": v for k, v in zip(NAMES, x)})
        out[f"{pre}_nll"], out[f"{pre}_nfev"], out[f"{pre}_s"] = fun, ne, s
        if hess:
            out.update(hess)
    rows.append(out)
    print(f"event {ev}: nll coral pt {out['nll_coral_pt']:.1f} true pt {out['nll_true_pt']:.1f} -> "
          + "  ".join(f"{k} {v[1]:.1f} ({v[2]} calls, {v[3]:.1f} s)" for k, v in res.items()), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
