#!/usr/bin/env python
"""R54n round 2 (idea 4): better stacking models on the same inputs, chosen by 5-fold CV by file.

  prep   build the design table per sample from network fits (fits/<model>_<set>_<smp>_<hyp>/*.csv), Coral's
         written fits and truth -> <dir>/design_<model>_<smp>.pkl (training sets) / design_<model>_<smp>_test.pkl
  cv     out-of-fold predictions of one candidate -> <dir>/oof/<smp>_<name>.npz
  score  every oof file vs candidate 'base' (R54n as is) and vs production
  final  train the chosen candidate on all training events, apply ONCE to the test table

Candidates: --learner gbm (--params JSON, --bag N seeds with feature subsampling) or mlp; --hess adds the
network fits' Hessian widths.

    python r54n2.py prep --model v2b --smp e
    python r54n2.py cv --smp e --name base --params '{"max_iter":1500,"max_depth":6,"learning_rate":0.02,"min_samples_leaf":30}'
    python r54n2.py score --smp e
"""
import argparse, glob, json, os, pickle
from pathlib import Path
import numpy as np, pandas as pd, uproot
from r54n import TARGETS, METRICS, fit_block, r54_features, cross_features, targets, residuals, compare

GATE_CM = 100.0
HESS = ["h_t", "h_along", "h_u", "h_v", "h_a", "h_b", "h_lp"]
TRAIN_SETS = ["r54n2_a", "r54n2_b"]


def load_set(fdir, model, sset, smp, sample, reco, tr, nhits):
    nn = {}
    for h in ("e", "mu"):
        files = [x for x in glob.glob(f"{fdir}/{model}_{sset}_{smp}_{h}/*.csv") if os.path.getsize(x) > 0]
        nn[h] = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("event").set_index("event")
    f = pd.DataFrame(uproot.open(f"{reco}/{sset}_{smp}/coral_{sample}_coral.root")["fits"].arrays(
        ["event", "sub_event", "particle_code", "nL", "t", "x", "y", "z", "px", "py", "pz", "p"], library="np"))
    f = f[f.sub_event == 0].drop_duplicates(["event", "particle_code"])
    cor = {}
    for h, code in (("e", 0), ("mu", 1)):
        g = f[f.particle_code == code].set_index("event")
        n = np.sqrt(g.px**2 + g.py**2 + g.pz**2)
        cor[h] = pd.DataFrame({"c_t": g.t, "c_x": g.x, "c_y": g.y, "c_z": g.z, "c_theta": np.arccos(g.pz/n),
                               "c_phi": np.arctan2(g.py, g.px), "c_p": g.p, "c_nll": g.nL})
    ev = nn["e"].index.intersection(nn["mu"].index).intersection(cor["e"].index).intersection(cor["mu"].index)
    ev = ev.intersection(tr.index).intersection(nhits.index).sort_values()
    oth = "mu" if smp == "e" else "e"
    nn_own, nn_oth = fit_block(nn[smp].loc[ev], "coral_"), fit_block(nn[oth].loc[ev], "coral_")
    c_own, c_oth = fit_block(cor[smp].loc[ev], "c_"), fit_block(cor[oth].loc[ev], "c_")
    T = tr.loc[ev]
    td = T[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); td /= np.linalg.norm(td, axis=1)[:, None]
    truth = (T[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float), td, T.P.to_numpy(float))
    nh = nhits.loc[ev].to_numpy(float)
    X = pd.concat([r54_features(nn_own, nn_oth, nh),
                   r54_features(c_own, c_oth, nh).add_prefix("cor_").drop(columns=["cor_hits_per_p"]),
                   cross_features(nn_own, c_own, c_oth[4] - c_own[4], "coral")], axis=1)
    H = pd.DataFrame({**{f"{k}_own": np.log(nn[smp].loc[ev, k].to_numpy()) for k in HESS},
                      **{f"{k}_oth": np.log(nn[oth].loc[ev, k].to_numpy()) for k in HESS},
                      "h_mineig_own": nn[smp].loc[ev, "h_mineig"].to_numpy()}) if HESS[0] in nn[smp] else None
    return dict(ev=ev.to_numpy(), X=X, H=H, Y=targets(nn_own, truth), base=nn_own, prod=c_own, truth=truth)


def concat(parts):
    cat = lambda bl: tuple(np.concatenate([b[k] for b in bl]) for k in range(len(bl[0])))
    return dict(ev=np.concatenate([p["ev"] for p in parts]), X=pd.concat([p["X"] for p in parts], ignore_index=True),
                H=None if parts[0]["H"] is None else pd.concat([p["H"] for p in parts], ignore_index=True),
                Y=pd.concat([p["Y"] for p in parts], ignore_index=True), base=cat([p["base"] for p in parts]),
                prod=cat([p["prod"] for p in parts]), truth=cat([p["truth"] for p in parts]))

R54N_MU_PARAMS = '{"max_iter": 800, "max_depth": 5, "learning_rate": 0.03, "min_samples_leaf": 40}'


def zero(n):
    return pd.DataFrame(0.0, index=range(n), columns=TARGETS)


def converged(D):
    n = len(D["ev"])
    return (residuals(D["prod"], zero(n), D["truth"])["ang"] < 20) & (residuals(D["base"], zero(n), D["truth"])["ang"] < 20)


def fit_predict(a, Xtr, Ytr, Xte, seed=0):
    """(n_te, 6) predicted errors."""
    if a.learner == "mlp":
        return mlp_fit_predict(Xtr, Ytr, Xte, seed)
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = json.loads(a.params)
    out = np.zeros((len(Xte), len(TARGETS)))
    for b in range(a.bag):
        kw = dict(params, random_state=seed + b)
        if a.bag > 1:
            kw.setdefault("max_features", 0.7)
        for j, c in enumerate(TARGETS):
            out[:, j] += HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(Xtr, Ytr[c]).predict(Xte)/a.bag
    return out


def mlp_fit_predict(Xtr, Ytr, Xte, seed, return_model=False):
    import torch
    torch.manual_seed(seed)
    med, iqr = Xtr.median(), (Xtr.quantile(0.75) - Xtr.quantile(0.25)).replace(0, 1).fillna(1)
    prep = lambda X: np.c_[((X - med)/iqr).clip(-10, 10).fillna(0).to_numpy(), X.isna().to_numpy()].astype(np.float32)
    ys = (Ytr.quantile(0.75) - Ytr.quantile(0.25)).to_numpy()
    xt, yt = torch.tensor(prep(Xtr)), torch.tensor((Ytr.to_numpy()/ys).astype(np.float32))
    rng = np.random.default_rng(seed)
    va = rng.random(len(xt)) < 0.1
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, len(TARGETS)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tri, vai = np.flatnonzero(~va), np.flatnonzero(va)
    best, state = np.inf, None
    for ep in range(epochs):
        net.train(); rng.shuffle(tri)
        for b in range(0, len(tri), 512):
            i = tri[b:b + 512]
            loss = (net(xt[i]) - yt[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            lv = (net(xt[vai]) - yt[vai]).abs().mean().item()
        if lv < best:
            best, state = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = net(torch.tensor(prep(Xte))).numpy()*ys
    return (P, net, med, iqr, ys) if return_model else P


def design(D, hess):
    return pd.concat([D["X"], D["H"]], axis=1) if hess else D["X"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--smp", default="e"); ap.add_argument("--model", default="v2b")
    ap.add_argument("--name", default="base"); ap.add_argument("--learner", default="gbm")
    ap.add_argument("--params", default='{"max_iter":1500,"max_depth":6,"learning_rate":0.02,"min_samples_leaf":30}')
    ap.add_argument("--bag", type=int, default=1); ap.add_argument("--hess", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blends", default="", help="score: comma list of a+b(+c) OOF averages")
    ap.add_argument("--blend", default="", help="final: JSON list of per-member overrides (learner, params, bag, seed); prediction = mean")
    ap.add_argument("--dir", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn")
    ap.add_argument("--reco", default="/sps/hyperk/bcarew/Output/Coral/Reco")
    ap.add_argument("--sextant", default="/sps/hyperk/bcarew/Software/Sextant/src")
    ap.add_argument("--test-events", default="25-99", help="event numbers within each file used by 'final'")
    a = ap.parse_args()
    sample = "electron" if a.smp == "e" else "muon"
    out = Path(a.dir)
    if a.mode == "prep":
        import sys; sys.path.insert(0, a.sextant)
        from sextant import Truth
        tfile = f"{a.dir}/truth_{sample}_v1.12.29.root"
        nhits = Truth.from_file(tfile).events.set_index("event").nhits
        tr = uproot.open(tfile)["Tracks"].arrays(library="pd")
        tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
        D = concat([load_set(f"{a.dir}/fits", a.model, s, a.smp, sample, a.reco, tr, nhits) for s in TRAIN_SETS])
        pickle.dump(D, open(out/f"design_{a.model}_{a.smp}.pkl", "wb"))
        Dt = load_set(f"{a.dir}/fits", a.model, "r54n2_test", a.smp, sample, a.reco, tr, nhits)
        pickle.dump(Dt, open(out/f"design_{a.model}_{a.smp}_test.pkl", "wb"))
        print(f"{a.smp}: train {len(D['ev'])} events, test {len(Dt['ev'])}; hessian {'yes' if D['H'] is not None else 'no'}")
        return
    D = pickle.load(open(out/f"design_{a.model}_{a.smp}.pkl", "rb"))
    conv = converged(D)
    groups = D["ev"]//100
    if a.mode == "cv":
        from sklearn.model_selection import GroupKFold
        X = design(D, a.hess)
        P = np.zeros((len(X), len(TARGETS)))
        for k, (trn, te) in enumerate(GroupKFold(5).split(X, groups=groups)):
            trn = trn[conv[trn]]
            P[te] = fit_predict(a, X.iloc[trn], D["Y"].iloc[trn], X.iloc[te], seed=k + 100*a.seed)
        (out/"oof").mkdir(exist_ok=True)
        np.savez(out/"oof"/f"{a.model}_{a.smp}_{a.name}.npz", P=P, args=json.dumps(vars(a)))
        print("saved", a.name)
        return
    gate = (D["X"].towall.to_numpy() >= GATE_CM)[:, None]
    idx = np.flatnonzero(conv)
    R_prod = residuals(D["prod"], zero(len(conv)), D["truth"])
    if a.mode == "score":
        def oof(name):
            return np.mean([np.load(out/"oof"/f"{a.model}_{a.smp}_{n}.npz")["P"] for n in name.split("+")], axis=0)

        def res(f):
            P = oof(f) if "+" in str(f) else np.load(f)["P"]
            return residuals(D["base"], pd.DataFrame(P*gate, columns=TARGETS), D["truth"])
        ref = res(out/"oof"/f"{a.model}_{a.smp}_base.npz")
        print(f"{a.smp} ({a.model}): {len(conv)} events, {len(idx)} converged; base vs production: "
              + "  ".join(f"{k} {v[0]:+.1f}" for k, v in compare(ref, R_prod, idx, boot=50).items()))
        rows = []
        cands = [(f, Path(f).stem.split(f"{a.smp}_", 1)[1]) for f in sorted(glob.glob(str(out/"oof"/f"{a.model}_{a.smp}_*.npz")))]
        cands += [(b, b) for b in a.blends.split(",") if b]
        for f, name in cands:
            c = compare(res(f), ref, idx, boot=200)
            rows.append((np.mean([v[0] for v in c.values()]), name, c))
        for m, name, c in sorted(rows, key=lambda x: x[0]):
            print(f"  {name:24s} mean {m:+.2f}%  " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in c.items()))
        return
    if a.mode == "final":
        Dt = pickle.load(open(out/f"design_{a.model}_{a.smp}_test.pkl", "rb"))
        lo, hi = (int(x) for x in a.test_events.split("-"))
        keep = np.flatnonzero(((Dt["ev"] % 100) >= lo) & ((Dt["ev"] % 100) <= hi))
        X, Xt = design(D, a.hess), design(Dt, a.hess)
        a_base = argparse.Namespace(**{**vars(a), "learner": "gbm", "bag": 1,
                                       "params": ap.get_default("params") if a.smp == "e" else R54N_MU_PARAMS})
        gt = (Dt["X"].towall.to_numpy() >= GATE_CM)[:, None]
        members = [argparse.Namespace(**{**vars(a), **m}) for m in json.loads(a.blend)] if a.blend else [a]
        Pc = np.mean([fit_predict(m, X.iloc[idx], D["Y"].iloc[idx], Xt, seed=100*m.seed) for m in members], axis=0)*gt
        Pb = fit_predict(a_base, D["X"].iloc[idx], D["Y"].iloc[idx], Dt["X"])*gt
        n = len(Dt["ev"])
        Rp = residuals(Dt["prod"], zero(n), Dt["truth"])
        Rn = residuals(Dt["base"], zero(n), Dt["truth"])
        Rb = residuals(Dt["base"], pd.DataFrame(Pb, columns=TARGETS), Dt["truth"])
        Rc = residuals(Dt["base"], pd.DataFrame(Pc, columns=TARGETS), Dt["truth"])
        ti = keep[(Rp["ang"][keep] < 20) & (Rn["ang"][keep] < 20)]
        print(f"{a.smp} FINAL ({a.model}, candidate {a.name}): {len(keep)} test events, {len(ti)} converged")
        for lab, R in (("production", Rp), ("R54n as is", Rb), ("candidate", Rc)):
            print(f"  {lab:12s} " + "  ".join(f"{k} {f(R, ti):.3f}" for k, f in METRICS.items()))
        for lab, R1, R0 in (("candidate vs R54n as is", Rc, Rb), ("candidate vs production", Rc, Rp), ("R54n as is vs production", Rb, Rp)):
            print(f"  {lab:26s} " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in compare(R1, R0, ti, 1000).items()))


if __name__ == "__main__":
    main()
