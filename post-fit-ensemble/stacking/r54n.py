#!/usr/bin/env python
"""R54n: the post-fit estimator retrained on the learned likelihood's fits (R48 v2b), alone or stacked with Coral's.

Configurations (base estimate -> its signed errors are predicted and subtracted):
  a  base = network fit;  R54 features from the network's e-/mu- fits
  b  base = network fit;  a + Coral's (production) fit of the same hypothesis relative to it + Coral's dnLL
  c  base = Coral fit;    R54 features from Coral's fits + the network's fit relative to it + network dnLL
Selection: 5-fold CV by file on the training set (files 600-699); --final trains on all of it and applies ONCE
to the held-out set (files 700-739), paired against production (Coral's written fits = Coral + R54).

    python r54n.py --hyp e- --cv                 # CV table over configurations / gates / GBM settings
    python r54n.py --hyp e- --final --config b --gate 1 --gbm 0
"""
import argparse, glob, itertools
import numpy as np, pandas as pd, uproot
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

R0, H0 = 3250.0, 3300.0
TARGETS = ["logp", "along", "u", "v", "a1", "a2"]
GBMS = [dict(max_iter=400, max_depth=4, learning_rate=0.04, min_samples_leaf=50),
        dict(max_iter=800, max_depth=5, learning_rate=0.03, min_samples_leaf=40),
        dict(max_iter=300, max_depth=3, learning_rate=0.05, min_samples_leaf=80),
        dict(max_iter=1500, max_depth=6, learning_rate=0.02, min_samples_leaf=30)]
rng = np.random.default_rng(5454)


def unit(th, ph):
    return np.c_[np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]


def towall(v, d):
    a = d[:, 0]**2 + d[:, 1]**2; b = 2*(v[:, 0]*d[:, 0] + v[:, 1]*d[:, 1]); c = v[:, 0]**2 + v[:, 1]**2 - R0**2
    disc = np.maximum(b*b - 4*a*c, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(a > 1e-9, (-b + np.sqrt(disc))/(2*a), np.inf)
        tz = np.where(d[:, 2] > 0, (H0 - v[:, 2])/d[:, 2], np.where(d[:, 2] < 0, (-H0 - v[:, 2])/d[:, 2], np.inf))
    return np.minimum(tr, tz)


def frame(d):
    ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0, 0, 1.]], [[1., 0, 0]])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1, axis=1)[:, None]
    return e1, np.cross(d, e1)


def fit_block(df, pre):
    """(t, v (n,3), d (n,3), p, nll) from columns pre_t, pre_x, ..."""
    return (df[f"{pre}t"].to_numpy(), df[[f"{pre}x", f"{pre}y", f"{pre}z"]].to_numpy(),
            unit(df[f"{pre}theta"].to_numpy(), df[f"{pre}phi"].to_numpy()), df[f"{pre}p"].to_numpy(),
            df[f"{pre}nll"].to_numpy())


def r54_features(own, oth, nhits):
    t, v, d, p, nll = own
    t2, v2, d2, p2, nll2 = oth
    e1, e2 = frame(d)
    r = np.hypot(v[:, 0], v[:, 1])
    dv = v2 - v; dd = d2
    return pd.DataFrame({"dlogp": np.log(p2/p), "dwall": np.minimum(R0 - r, H0 - np.abs(v[:, 2])), "towall": towall(v, d),
                         "r": r, "z": v[:, 2], "cz": d[:, 2], "crad": (v[:, 0]*d[:, 0] + v[:, 1]*d[:, 1])/np.maximum(r, 1),
                         "dnll": nll2 - nll, "nll": nll, "dv_along": (dv*d).sum(1), "dv_u": (dv*e1).sum(1), "dv_v": (dv*e2).sum(1),
                         "dt": t2 - t, "dd_u": np.degrees((dd*e1).sum(1)), "dd_v": np.degrees((dd*e2).sum(1)),
                         "e1z": e1[:, 2], "e2z": e2[:, 2], "e1rad": (e1[:, 0]*v[:, 0] + e1[:, 1]*v[:, 1])/np.maximum(r, 1),
                         "e2rad": (e2[:, 0]*v[:, 0] + e2[:, 1]*v[:, 1])/np.maximum(r, 1), "hits_per_p": np.log(nhits/p)})


def cross_features(base, other, other_dnll, tag):
    """The other estimator's fit of the same hypothesis, in the base fit's frame."""
    t, v, d, p, _ = base
    t2, v2, d2, p2, _ = other
    e1, e2 = frame(d)
    dv = v2 - v
    return pd.DataFrame({f"{tag}_dlogp": np.log(p2/p), f"{tag}_dalong": (dv*d).sum(1), f"{tag}_du": (dv*e1).sum(1),
                         f"{tag}_dv": (dv*e2).sum(1), f"{tag}_dt": t2 - t, f"{tag}_dd_u": np.degrees((d2*e1).sum(1)),
                         f"{tag}_dd_v": np.degrees((d2*e2).sum(1)), f"{tag}_pid": other_dnll})


def load(prefix, smp, hyp, root, truth_file, nhits):
    """Per event: network fits (both hypotheses), Coral's written fits (both), truth."""
    other = "mu" if hyp == "e" else "e"
    nn = {h: pd.concat([pd.read_csv(f) for f in glob.glob(f"{prefix}_{smp}_{h}_*.csv")]).drop_duplicates("event").set_index("event")
          for h in ("e", "mu")}
    f = pd.DataFrame(uproot.open(root)["fits"].arrays(["event", "sub_event", "particle_code", "nL", "t", "x", "y", "z", "px", "py", "pz", "p"], library="np"))
    f = f[f.sub_event == 0].drop_duplicates(["event", "particle_code"])
    cor = {}
    for h, code in (("e", 0), ("mu", 1)):
        g = f[f.particle_code == code].set_index("event")
        n = np.sqrt(g.px**2 + g.py**2 + g.pz**2)
        cor[h] = pd.DataFrame({"c_t": g.t, "c_x": g.x, "c_y": g.y, "c_z": g.z, "c_theta": np.arccos(g.pz/n),
                               "c_phi": np.arctan2(g.py, g.px), "c_p": g.p, "c_nll": g.nL})
    tr = uproot.open(truth_file)["Tracks"].arrays(library="pd")
    tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
    ev = nn["e"].index.intersection(nn["mu"].index).intersection(cor["e"].index).intersection(cor["mu"].index).intersection(tr.index).intersection(nhits.index)
    nnb = {h: fit_block(nn[h].loc[ev].rename(columns={"coral_nll": "coral_nll"}), "coral_") for h in ("e", "mu")}
    cb = {h: fit_block(cor[h].loc[ev], "c_") for h in ("e", "mu")}
    T = tr.loc[ev]
    td = T[["Dir_x0", "Dir_x1", "Dir_x2"]].to_numpy(float); td /= np.linalg.norm(td, axis=1)[:, None]
    truth = (T[["Start_x0", "Start_x1", "Start_x2"]].to_numpy(float), td, T.P.to_numpy(float))
    return ev, nnb[hyp], nnb[other], cb[hyp], cb[other], truth, nhits.loc[ev].to_numpy(float)


def targets(base, truth):
    _, v, d, p, _ = base
    tv, td, tp = truth
    e1, e2 = frame(d)
    dv = v - tv
    return pd.DataFrame({"logp": np.log(p/tp), "along": (dv*d).sum(1), "u": (dv*e1).sum(1), "v": (dv*e2).sum(1),
                         "a1": np.degrees((td*e1).sum(1)), "a2": np.degrees((td*e2).sum(1))})


def design(cfg, nn_own, nn_oth, c_own, c_oth, nh):
    if cfg == "a":
        return nn_own, r54_features(nn_own, nn_oth, nh)
    if cfg == "b":
        X = r54_features(nn_own, nn_oth, nh)
        return nn_own, pd.concat([X, cross_features(nn_own, c_own, c_oth[4] - c_own[4], "coral")], axis=1)
    if cfg == "d":
        X = r54_features(nn_own, nn_oth, nh)
        Xc = r54_features(c_own, c_oth, nh).add_prefix("cor_").drop(columns=["cor_hits_per_p"])
        return nn_own, pd.concat([X, Xc, cross_features(nn_own, c_own, c_oth[4] - c_own[4], "coral")], axis=1)
    X = r54_features(c_own, c_oth, nh)
    return c_own, pd.concat([X, cross_features(c_own, nn_own, nn_oth[4] - nn_own[4], "nn")], axis=1)


def residuals(base, corr, truth):
    """momentum %, vertex distance, angle (deg) of base corrected by the predicted errors corr (DataFrame)."""
    _, v, d, p, _ = base
    tv, td, tp = truth
    e1, e2 = frame(d)
    pc = p*np.exp(-corr.logp.to_numpy())
    vc = v - (corr.along.to_numpy()[:, None]*d + corr.u.to_numpy()[:, None]*e1 + corr.v.to_numpy()[:, None]*e2)
    dc = d + np.radians(corr.a1.to_numpy())[:, None]*e1 + np.radians(corr.a2.to_numpy())[:, None]*e2
    dc /= np.linalg.norm(dc, axis=1)[:, None]
    return dict(p=100*(pc/tp - 1), vtx=np.linalg.norm(vc - tv, axis=1),
                ang=np.degrees(np.arccos(np.clip((dc*td).sum(1), -1, 1))))


def gcore(x):
    x = np.asarray(x); m, s = np.median(x), 1.4826*np.median(np.abs(x - np.median(x)))
    for _ in range(5):
        sel = np.abs(x - m) < 2*s
        m, s = x[sel].mean(), x[sel].std()
    return s/(1 + m/100)


METRICS = {"p_core": lambda r, i: gcore(r["p"][i]),
           "p_half68": lambda r, i: np.subtract(*np.percentile(r["p"][i], [84, 16]))/2,
           "vtx68": lambda r, i: np.percentile(r["vtx"][i], 68), "ang68": lambda r, i: np.percentile(r["ang"][i], 68)}


def compare(R1, R0_, idx, boot=300):
    out = {}
    for k, f in METRICS.items():
        v = f(R1, idx)/f(R0_, idx) - 1
        bs = [f(R1, j)/f(R0_, j) - 1 for j in (rng.choice(idx, len(idx)) for _ in range(boot))]
        out[k] = (100*v, *np.percentile(100*np.array(bs), [2.5, 97.5]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default="e-"); ap.add_argument("--cv", action="store_true"); ap.add_argument("--final", action="store_true")
    ap.add_argument("--config", default="b"); ap.add_argument("--gate", type=float, default=0.0); ap.add_argument("--gbm", type=int, default=0)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--configs", default="abc"); ap.add_argument("--gbms", default="0,1,2"); ap.add_argument("--gates", default="0,1,3")
    ap.add_argument("--extra", action="store_true", help="add the files 800-999 training set (r54n2_*)")
    ap.add_argument("--frac", type=float, default=1.0, help="CV learning curve: fraction of training FILES used to train")
    ap.add_argument("--reco", default="/sps/hyperk/bcarew/Output/Coral/Reco")
    ap.add_argument("--sextant", default="/sps/hyperk/bcarew/Software/Sextant/src")
    a = ap.parse_args()
    smp, sample = ("e", "electron") if a.hyp == "e-" else ("mu", "muon")
    import sys; sys.path.insert(0, a.sextant)
    from sextant import Truth
    tfile = f"{a.dir}/truth_{sample}_v1.12.29.root"
    nhits = Truth.from_file(tfile).events.set_index("event").nhits
    Rd = a.reco
    sets = [(f"{a.dir}/r54n", f"{Rd}/r54n_train_{smp}/coral_{sample}_coral.root")]
    if a.extra and glob.glob(f"{a.dir}/r54n2_{smp}_e_*.csv"):
        sets.append((f"{a.dir}/r54n2", f"{Rd}/r54n_train2_{smp}/coral_{sample}_coral.root"))
    parts = [load(p, smp, smp, r, tfile, nhits) for p, r in sets]
    cat = lambda blocks: tuple(np.concatenate([b[k] for b in blocks]) for k in range(len(blocks[0])))
    ev = pd.Index(np.concatenate([p[0].to_numpy() for p in parts]))
    nn_own, nn_oth, c_own, c_oth, truth = (cat([p[k] for p in parts]) for k in (1, 2, 3, 4, 5))
    nh = np.concatenate([p[6] for p in parts])
    zero = lambda n: pd.DataFrame(0.0, index=range(n), columns=TARGETS)
    R_prod = residuals(c_own, zero(len(ev)), truth)
    conv = (R_prod["ang"] < 20) & (residuals(nn_own, zero(len(ev)), truth)["ang"] < 20)
    groups = ev.to_numpy()//100
    if a.cv:
        print(f"{a.hyp} training set ({len(sets)} part(s)): {len(ev)} events, {conv.sum()} converged in both")
        idx = np.flatnonzero(conv)
        rn = residuals(nn_own, zero(len(ev)), truth)
        print("  network fit alone vs production: " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in compare(rn, R_prod, idx).items()))
        for cfg, g in itertools.product(a.configs, [int(x) for x in a.gbms.split(",")]):
            base, X = design(cfg, nn_own, nn_oth, c_own, c_oth, nh)
            Y = targets(base, truth)
            P = pd.DataFrame(0.0, index=range(len(ev)), columns=TARGETS)
            for c in TARGETS:
                for trn, te in GroupKFold(5).split(X, Y[c], groups):
                    trn = trn[conv[trn]]
                    if a.frac < 1.0:
                        keep_files = np.unique(groups[trn]); keep_files = keep_files[:max(1, int(a.frac*len(keep_files)))]
                        trn = trn[np.isin(groups[trn], keep_files)]
                    m = HistGradientBoostingRegressor(loss="absolute_error", **GBMS[g]).fit(X.iloc[trn], Y[c].iloc[trn])
                    P.iloc[te, P.columns.get_loc(c)] = m.predict(X.iloc[te])
            for gate in [float(x) for x in a.gates.split(",")]:
                gm = (X.towall.to_numpy() >= 100*gate)[:, None]
                r = residuals(base, P*gm, truth)
                cmp = compare(r, R_prod, idx, boot=200)
                print(f"  config {cfg} gbm {g} gate {gate} m: " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in cmp.items()), flush=True)
    if a.final:
        base, X = design(a.config, nn_own, nn_oth, c_own, c_oth, nh)
        Y = targets(base, truth)
        idx_tr = np.flatnonzero(conv)
        models = {c: HistGradientBoostingRegressor(loss="absolute_error", **GBMS[a.gbm]).fit(X.iloc[idx_tr], Y[c].iloc[idx_tr]) for c in TARGETS}
        te = load(f"{a.dir}/pid", smp, smp, f"{Rd}/r48_pid_{smp}/coral_{sample}_coral.root", tfile, nhits)
        ev2, nn_own2, nn_oth2, c_own2, c_oth2, truth2, nh2 = te
        base2, X2 = design(a.config, nn_own2, nn_oth2, c_own2, c_oth2, nh2)
        gm = (X2.towall.to_numpy() >= 100*a.gate)[:, None]
        P2 = pd.DataFrame({c: models[c].predict(X2) for c in TARGETS})*gm
        Rp = residuals(c_own2, zero(len(ev2)), truth2)
        Rn = residuals(nn_own2, zero(len(ev2)), truth2)
        Rc = residuals(base2, P2, truth2)
        idx = np.flatnonzero((Rp["ang"] < 20) & (Rn["ang"] < 20))
        print(f"{a.hyp} FINAL held-out (files 700-739): {len(ev2)} events, {len(idx)} converged; config {a.config} gbm {a.gbm} gate {a.gate} m")
        print("  production (Coral + R54): " + "  ".join(f"{k} {f(Rp, idx):.3f}" for k, f in METRICS.items()))
        print("  network fit alone vs production:  " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in compare(Rn, Rp, idx, 1000).items()))
        print("  R54n vs production:               " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]" for k, v in compare(Rc, Rp, idx, 1000).items()))
        print("  R54n absolute: " + "  ".join(f"{k} {f(Rc, idx):.3f}" for k, f in METRICS.items()) + f"  p bias {np.median(Rc['p'][idx]):+.2f}%")


if __name__ == "__main__":
    main()
