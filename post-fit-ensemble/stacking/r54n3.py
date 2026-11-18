#!/usr/bin/env python
"""R54n3: which fit types to stack. Sources, each with an e- and a mu- hypothesis fit per event:
  C   Coral as written (8d2387e: vertex-bias curves + R54; R54 is a function of Coral's own two fits)
  N   network-likelihood fit v2b (+ Hessian)      N4  network-likelihood fit v4
  R   HK ResNet-152 (vertex + 3-momentum models; no time, no likelihood)
Study 'tn' (technote runs, Coral 1527e03 with both output corrections off): C = Coral raw, V = + vertex-bias
curves, P = + R54 (both applied offline by tn_postprocess.py); train on tn_cal (true e- / mu- run 0, files
0-299), test on tn_val (run 1, 100k).
A combination (e.g. C+N+R) is stacked with the 10-28 recipe: base = first source; features = the base's R54
features, every other source's R54 features and its fit relative to the base (+ N's Hessian widths); learner =
mean of 3 MLPs and a 5-seed bagged GBM. Events: both hypotheses present in every source; widths on events where
C and N (tn: C) are within 20 deg of truth.

  prep   [--study tn] -> design3_<study>_<smp>.pkl (training sets r54n2_a + r54n2_b, test set r54n2_test)
  run    --combo C+N --member mlp0|mlp1|mlp2|gbm: 5-fold-by-file OOF on the training sets + full-train test prediction
  score  raw fits, production and every combination (members averaged), CV and test (events 25-99)
  apply  --study tn: write tn_val with the stack's corrected fits (tn/coral_<val>_coral_stack.root)
"""
import argparse, glob, os, pickle
from pathlib import Path
import numpy as np, pandas as pd, uproot
from r54n import METRICS, fit_block, frame, r54_features, cross_features, targets, residuals, compare, unit
from r54n2 import fit_predict, zero, HESS

SETS = {"r54n2_a": "train", "r54n2_b": "train", "r54n2_test": "test"}
MEMBERS = {"mlp0": dict(learner="mlp", seed=0), "mlp1": dict(learner="mlp", seed=1), "mlp2": dict(learner="mlp", seed=2),
           "gbm": dict(learner="gbm", seed=0, bag=5, params='{"max_iter": 1000, "learning_rate": 0.03, "max_depth": null, '
                       '"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 3, "max_features": 0.7}')}


def coral(path):
    f = pd.DataFrame(uproot.open(path)["fits"].arrays(
        ["event", "sub_event", "particle_code", "nL", "t", "x", "y", "z", "px", "py", "pz", "p"], library="np"))
    f = f[f.sub_event == 0].drop_duplicates(["event", "particle_code"])
    out = {}
    for h, code in (("e", 0), ("mu", 1)):
        g = f[f.particle_code == code].set_index("event")
        n = np.sqrt(g.px**2 + g.py**2 + g.pz**2)
        out[h] = pd.DataFrame({"t": g.t, "x": g.x, "y": g.y, "z": g.z, "theta": np.arccos(g.pz/n),
                               "phi": np.arctan2(g.py, g.px), "p": g.p, "nll": g.nL})
    return out


def network(fdir, model, sset, smp):
    out = {}
    for h in ("e", "mu"):
        files = [x for x in glob.glob(f"{fdir}/{model}_{sset}_{smp}_{h}/*.csv") if os.path.getsize(x) > 0]
        d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("event").set_index("event")
        out[h] = d.rename(columns={f"coral_{k}": k for k in ("t", "x", "y", "z", "theta", "phi", "p", "nll")})
    return out


def resnet(path):
    d = pd.read_csv(path).dropna().set_index("event")
    out = {}
    for h in ("e", "mu"):
        pv = d[[f"{h}_px", f"{h}_py", f"{h}_pz"]].to_numpy()
        p = np.linalg.norm(pv, axis=1)
        out[h] = pd.DataFrame({"t": np.nan, "x": d[f"{h}_x"], "y": d[f"{h}_y"], "z": d[f"{h}_z"],
                               "theta": np.arccos(pv[:, 2]/p), "phi": np.arctan2(pv[:, 1], pv[:, 0]), "p": p, "nll": np.nan},
                              index=d.index)
    return out


def stack_sources(a, sset, sample):
    return {"C": coral(f"{a.reco}/{sset}_{a.smp}/coral_{sample}_coral.root"),
            "N": network(f"{a.dir}/fits", "v2b", sset, a.smp), "N4": network(f"{a.dir}/fits", "v4", sset, a.smp),
            "R": resnet(f"{a.dir}/resnet/pred_{sset}_{a.smp}.csv")}


def tracks_truth(tfile):
    tr = uproot.open(tfile)["Tracks"].arrays(library="pd")
    tr = tr[(tr.Parenttype == 0) & (tr.Id == 1)].drop_duplicates("EvtNum").set_index("EvtNum")
    return pd.DataFrame({"x": tr.Start_x0, "y": tr.Start_x1, "z": tr.Start_x2, "dx": tr.Dir_x0, "dy": tr.Dir_x1,
                         "dz": tr.Dir_x2, "p": tr.P})


def rings_truth(tfile):
    f = uproot.open(tfile)
    r = f["rings"].arrays(library="pd")
    r = r[r.is_primary_lepton.astype(bool)].drop_duplicates("event").set_index("event")
    nh = f["events"].arrays(["event", "nhits_trigger0"], library="pd").set_index("event").nhits_trigger0
    return r[["x", "y", "z", "dx", "dy", "dz", "p"]], nh


def load_set(a, src, tr, nhits, conv):
    ev = tr.index.intersection(nhits.index)
    for s in src.values():
        for d in s.values():
            ev = ev.intersection(d.index)
    ev = ev.sort_values()
    oth = "mu" if a.smp == "e" else "e"
    blocks = {k: (fit_block(s[a.smp].loc[ev], ""), fit_block(s[oth].loc[ev], "")) for k, s in src.items()}
    H = pd.DataFrame({**{f"{k}_own": np.log(src["N"][a.smp].loc[ev, k].to_numpy()) for k in HESS},
                      **{f"{k}_oth": np.log(src["N"][oth].loc[ev, k].to_numpy()) for k in HESS},
                      "h_mineig_own": src["N"][a.smp].loc[ev, "h_mineig"].to_numpy()}) if "N" in src else None
    T = tr.loc[ev]
    td = T[["dx", "dy", "dz"]].to_numpy(float); td = td/np.linalg.norm(td, axis=1)[:, None]
    truth = (T[["x", "y", "z"]].to_numpy(float), td, T.p.to_numpy(float))
    return dict(ev=ev.to_numpy(), blocks=blocks, H=H, truth=truth, nh=nhits.loc[ev].to_numpy(float), conv=conv)


def concat(parts):
    def cat(bl):
        return tuple(np.concatenate([b[k] for b in bl]) for k in range(len(bl[0])))
    H = None if parts[0]["H"] is None else pd.concat([p["H"] for p in parts], ignore_index=True)
    return dict(ev=np.concatenate([p["ev"] for p in parts]), H=H, conv=parts[0]["conv"],
                blocks={k: tuple(cat([p["blocks"][k][j] for p in parts]) for j in range(2)) for k in parts[0]["blocks"]},
                truth=cat([p["truth"] for p in parts]), nh=np.concatenate([p["nh"] for p in parts]))


def design(D, combo):
    """(base block, X) of one combination; columns that are NaN everywhere (no time / likelihood) dropped."""
    srcs = combo.split("+")
    own, oth = D["blocks"][srcs[0]]
    X = [r54_features(own, oth, D["nh"])]
    for s in srcs[1:]:
        so, sx = D["blocks"][s]
        X.append(r54_features(so, sx, D["nh"]).add_prefix(f"{s}_").drop(columns=[f"{s}_hits_per_p"]))
        X.append(cross_features(own, so, sx[4] - so[4], f"x{s}"))
    if "N" in srcs:
        X.append(D["H"])
    X = pd.concat(X, axis=1).replace([np.inf, -np.inf], np.nan)
    return own, X.loc[:, X.notna().any()]


def converged(D):
    n = len(D["ev"])
    return np.all([residuals(D["blocks"][s][0], zero(n), D["truth"])["ang"] < 20 for s in D["conv"]], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--smp", default="e"); ap.add_argument("--study", default="stack")
    ap.add_argument("--combo", default="C"); ap.add_argument("--member", default="mlp0")
    ap.add_argument("--combos", default="C,C+N,C+R,C+N+R,N+C,C+N+N4,C+N+N4+R")
    ap.add_argument("--dir", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn")
    ap.add_argument("--reco", default="/sps/hyperk/bcarew/Output/Coral/Reco")
    ap.add_argument("--sextant", default="/sps/hyperk/bcarew/Software/Sextant/src")
    ap.add_argument("--test-events", default="25-99")
    ap.add_argument("--tn", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn/tn")
    a = ap.parse_args()
    sample = "electron" if a.smp == "e" else "muon"
    out = Path(a.dir)
    pkl = out/f"design3_{a.study}_{a.smp}.pkl"
    if a.mode == "prep" and a.study == "tn":
        cal, val = ("elneg", "elneg-r1") if a.smp == "e" else ("muon", "muneg-r1")
        tr, nh = rings_truth(f"{a.tn}/truth_{cal}_v1.12.29_f0-299.root")
        D = load_set(a, {"C": coral(f"{a.tn}/coral_{cal}_coral_renum.root")}, tr, nh, ["C"])
        tr, nh = rings_truth(f"{a.tn}/truth_{val}_v1.12.29.root")
        Dt = load_set(a, {"C": coral(f"{a.tn}/coral_{val}_coral_renum.root"), "V": coral(f"{a.tn}/coral_{val}_coral_vb.root"),
                          "P": coral(f"{a.tn}/coral_{val}_coral_r54.root")}, tr, nh, ["C"])
        pickle.dump((D, Dt), open(pkl, "wb"))
        print(f"{a.smp} tn: train {len(D['ev'])} events ({converged(D).sum()} converged), test {len(Dt['ev'])}")
        return
    if a.mode == "prep":
        import sys; sys.path.insert(0, a.sextant)
        from sextant import Truth
        tfile = f"{a.dir}/truth_{sample}_v1.12.29.root"
        nhits = Truth.from_file(tfile).events.set_index("event").nhits
        tr = tracks_truth(tfile)
        D = concat([load_set(a, stack_sources(a, s, sample), tr, nhits, ["C", "N"]) for s, k in SETS.items() if k == "train"])
        Dt = load_set(a, stack_sources(a, "r54n2_test", sample), tr, nhits, ["C", "N"])
        pickle.dump((D, Dt), open(pkl, "wb"))
        print(f"{a.smp}: train {len(D['ev'])} events ({converged(D).sum()} converged), test {len(Dt['ev'])}")
        return
    D, Dt = pickle.load(open(pkl, "rb"))
    conv = converged(D)
    if a.mode == "run":
        from sklearn.model_selection import GroupKFold
        m = argparse.Namespace(**{"bag": 1, "params": "{}", **MEMBERS[a.member]})
        base, X = design(D, a.combo)
        _, Xt = design(Dt, a.combo)
        Xt = Xt.reindex(columns=X.columns)
        Y = targets(base, D["truth"])
        P = np.zeros((len(X), Y.shape[1]))
        for k, (trn, te) in enumerate(GroupKFold(5).split(X, groups=D["ev"]//100)):
            trn = trn[conv[trn]]
            P[te] = fit_predict(m, X.iloc[trn], Y.iloc[trn], X.iloc[te], seed=k + 100*m.seed)
        idx = np.flatnonzero(conv)
        Pt = fit_predict(m, X.iloc[idx], Y.iloc[idx], Xt, seed=100*m.seed)
        (out/"oof3").mkdir(exist_ok=True)
        np.savez(out/"oof3"/f"{a.study}_{a.smp}_{a.combo}_{a.member}.npz", P=P, Pt=Pt, cols=np.array(X.columns))
        print("saved", a.combo, a.member, X.shape[1], "features")
        return
    if a.mode == "apply":
        # tn: the stack (members averaged, trained on tn_cal) applied to tn_val, written like tn_postprocess.py's
        # output: the vertex-bias file with this sample's hypothesis, sub-event 0, replaced by the corrected raw fit
        assert a.study == "tn"
        val = "elneg-r1" if a.smp == "e" else "muneg-r1"
        Pt = np.mean([np.load(out/"oof3"/f"tn_{a.smp}_C_{mm}.npz")["Pt"] for mm in MEMBERS], axis=0)
        t, v, d, p, _ = Dt["blocks"]["C"][0]
        e1, e2 = frame(d)
        al, u, w, a1, a2 = (Pt[:, k] for k in range(1, 6))
        vc = v - (al[:, None]*d + u[:, None]*e1 + w[:, None]*e2)
        dc = d + np.radians(a1)[:, None]*e1 + np.radians(a2)[:, None]*e2
        dc /= np.linalg.norm(dc, axis=1)[:, None]
        pc = p*np.exp(-Pt[:, 0])
        fin = uproot.open(f"{a.tn}/coral_{val}_coral_vb.root")
        fits = fin["fits"].arrays(library="pd")
        code = 0 if a.smp == "e" else 1
        rows = fits.index[(fits.sub_event == 0) & (fits.particle_code == code)]
        sel = fits.loc[rows].drop_duplicates("event")
        pos = pd.Series(np.arange(len(Dt["ev"])), index=Dt["ev"])
        sel = sel[sel.event.isin(pos.index)]
        k = pos.loc[sel.event].to_numpy()
        fits.loc[sel.index, ["x", "y", "z"]] = vc[k]
        fits.loc[sel.index, ["px", "py", "pz"]] = dc[k]
        fits.loc[sel.index, "p"] = pc[k]
        dst = f"{a.tn}/coral_{val}_coral_stack.root"
        with uproot.recreate(dst) as fo:
            def tree(name, arr):
                fo.mktree(name, {c: x.dtype for c, x in arr.items()}).extend(arr)
            tree("fits", {c: fits[c].to_numpy() for c in fits.columns})
            for key in fin.keys(cycle=False):
                if key != "fits" and fin[key].classname == "TTree":
                    tree(key, fin[key].arrays(library="np"))
        print(f"wrote {dst}: {len(sel)} of {len(rows)} sub-event-0 {'e-' if code == 0 else 'mu-'} fits corrected")
        chk = uproot.open(dst)["fits"].arrays(library="pd")
        chk = chk[(chk.sub_event == 0) & (chk.particle_code == code)].drop_duplicates("event").set_index("event").loc[Dt["ev"]]
        n = np.sqrt(chk.px**2 + chk.py**2 + chk.pz**2)
        blk = fit_block(pd.DataFrame({"t": chk.t, "x": chk.x, "y": chk.y, "z": chk.z, "theta": np.arccos(chk.pz/n),
                                      "phi": np.arctan2(chk.py, chk.px), "p": chk.p, "nll": chk.nL}), "")
        R = residuals(blk, zero(len(Dt["ev"])), Dt["truth"])
        it = np.flatnonzero(converged(Dt))
        print("  read back, converged events: " + "  ".join(f"{m} {f(R, it):.3f}" for m, f in METRICS.items())
              + f"  p median {np.median(R['p'][it]):+.3f}%")
        for s, lab in (("C", "raw"), ("P", "vb + R54")):
            Rs = residuals(Dt["blocks"][s][0], zero(len(Dt["ev"])), Dt["truth"])
            print(f"  {lab}: p median {np.median(Rs['p'][it]):+.3f}%")
        return
    if a.mode == "score":
        lo, hi = (int(x) for x in a.test_events.split("-"))
        convt = converged(Dt)
        it = np.flatnonzero(convt & ((Dt["ev"] % 100) >= lo) & ((Dt["ev"] % 100) <= hi)) if a.study == "stack" else np.flatnonzero(convt)
        ic = np.flatnonzero(conv)
        rows = {}
        labels = ((("C", "Coral raw"), ("V", "Coral + vertex-bias curves"), ("P", "Coral + vb + R54 (production)"))
                  if a.study == "tn" else (("C", "Coral + R54 (production)"), ("N", "network fit v2b"),
                                           ("N4", "network fit v4"), ("R", "ResNet")))
        for s, lab in labels:
            rows[lab] = tuple(residuals(DD["blocks"][s][0], zero(len(DD["ev"])), DD["truth"]) if s in DD["blocks"] else None
                              for DD in (D, Dt))
        for c in (["C"] if a.study == "tn" else a.combos.split(",")):
            fs = [out/"oof3"/f"{a.study}_{a.smp}_{c}_{mm}.npz" for mm in MEMBERS]
            if not all(f.exists() for f in fs):
                print(f"  (missing members for {c})"); continue
            zs = [np.load(f) for f in fs]
            base = (D["blocks"][c.split("+")[0]][0], Dt["blocks"][c.split("+")[0]][0])
            P = np.mean([z["P"] for z in zs], axis=0); Pt = np.mean([z["Pt"] for z in zs], axis=0)
            rows[f"stack {c}"] = (residuals(base[0], pd.DataFrame(P, columns=["logp", "along", "u", "v", "a1", "a2"]), D["truth"]),
                                  residuals(base[1], pd.DataFrame(Pt, columns=["logp", "along", "u", "v", "a1", "a2"]), Dt["truth"]))
        print(f"{a.smp} {a.study}: CV {len(ic)} events, test {len(it)} events")
        print(f"  {'':28s} " + "  ".join(f"{k:>9s}" for k in METRICS) + "   |  test: " + "  ".join(f"{k:>9s}" for k in METRICS))
        for lab, (R, Rt) in rows.items():
            cv = "  ".join(f"{f(R, ic):9.3f}" for f in METRICS.values()) if R is not None else " "*43
            print(f"  {lab:30s} {cv}   |        " + "  ".join(f"{f(Rt, it):9.3f}" for f in METRICS.values()))
        refs = ("Coral raw", "Coral + vb + R54 (production)") if a.study == "tn" else ("Coral + R54 (production)", "stack C")
        for ref in refs:
            for part, j, ii in (("CV", 0, ic), ("test", 1, it)):
                if ref not in rows or rows[ref][j] is None or (a.study == "tn" and part == "CV"):
                    continue
                print(f"  vs {ref} ({part}, % [95% CI]):")
                for lab, R in rows.items():
                    if lab != ref and R[j] is not None and (lab.startswith("stack") or a.study == "tn"):
                        print(f"    {lab:30s} " + "  ".join(f"{k} {v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]"
                                                          for k, v in compare(R[j], rows[ref][j], ii, 200).items()))


if __name__ == "__main__":
    main()
