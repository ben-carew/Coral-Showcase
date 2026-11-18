#!/usr/bin/env python
"""Train the Coral-only stack (R54n3 'tn' study: raw Coral 1527e03 fits of both hypotheses -> corrections of one
hypothesis' fit) on all converged tn_cal events and export it to plain arrays: 3 MLPs (weights + input / output
scaling) and a 5-seed bagged GBM (trees in R54's array format). Prediction = mean of the 3 MLPs and the GBM bag.
Checks the numpy evaluator against torch / sklearn and against the corrections already applied to tn_val.

    python r54n3_export.py --smp e --out <dir>      # -> <dir>/stack_coral_e.npz
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from r54n import targets
from r54n2 import mlp_fit_predict, TARGETS
from r54n3 import MEMBERS, design, converged

ROOT = Path(__file__).resolve().parents[4]


def export_trees(m, pre):
    """sklearn HistGradientBoostingRegressor -> flat node arrays (as Analysis/Archive/2026-11_tidy/Scripts/r54_export.py)."""
    nodes = [p[0].nodes for p in m._predictors]
    roots = np.cumsum([0] + [len(n) for n in nodes[:-1]]).astype(np.int64)
    alln = np.concatenate(nodes)
    off = np.repeat(roots, [len(n) for n in nodes])
    return {f"{pre}_feature": alln["feature_idx"].astype(np.int64), f"{pre}_threshold": alln["num_threshold"].astype(np.float64),
            f"{pre}_nan_left": alln["missing_go_to_left"].astype(np.bool_), f"{pre}_left": alln["left"].astype(np.int64) + off,
            f"{pre}_right": alln["right"].astype(np.int64) + off, f"{pre}_leaf": alln["is_leaf"].astype(np.bool_),
            f"{pre}_value": alln["value"].astype(np.float64), f"{pre}_roots": roots,
            f"{pre}_baseline": np.float64(np.ravel(m._baseline_prediction)[0])}


def tree_predict(arr, pre, X):
    """Same walk as R54's evaluator, vectorised over events."""
    f, th, nl, le, ri, lf, va = (arr[f"{pre}_{k}"] for k in ("feature", "threshold", "nan_left", "left", "right", "leaf", "value"))
    out = np.full(len(X), float(arr[f"{pre}_baseline"]))
    rows = np.arange(len(X))
    for r in arr[f"{pre}_roots"]:
        n = np.full(len(X), r)
        act = ~lf[n]
        while act.any():
            i, m = rows[act], n[act]
            v = X[i, f[m]]
            go_left = np.where(np.isnan(v), nl[m], v <= th[m])
            n[act] = np.where(go_left, le[m], ri[m])
            act[act] = ~lf[n[act]]
        out += va[n]
    return out


def predict(arr, X):
    """(n, 6) corrections (logp, along, u, v [cm], a1, a2 [deg]) from the feature matrix X (columns = arr['features'])."""
    X = np.asarray(X, float)
    mlps = []
    for k in range(int(arr["n_mlp"])):
        z = np.clip((X - arr[f"mlp{k}_med"])/arr[f"mlp{k}_iqr"], -10, 10)
        h = np.c_[np.nan_to_num(z, nan=0.0), np.isnan(X)]
        for j in range(4):
            h = h @ arr[f"mlp{k}_w{j}"].T + arr[f"mlp{k}_b{j}"]
            if j < 3:
                h = h/(1 + np.exp(-h))
        mlps.append(h*arr[f"mlp{k}_ys"])
    nb = int(arr["n_gbm"])
    gbm = np.column_stack([np.mean([tree_predict(arr, f"gbm{b}_{t}", X) for b in range(nb)], axis=0) for t in arr["targets"]])
    return np.mean(mlps + [gbm], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smp", default="e"); ap.add_argument("--dir", default="/sps/hyperk/bcarew/Output/Coral/Jobs/r48_nn")
    ap.add_argument("--out", required=True); ap.add_argument("--check", type=int, default=2000)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    D, Dt = pickle.load(open(Path(a.dir)/f"design3_tn_{a.smp}.pkl", "rb"))
    base, X = design(D, "C")
    _, Xt = design(Dt, "C")
    Xt = Xt.reindex(columns=X.columns)
    Y = targets(base, D["truth"])
    idx = np.flatnonzero(converged(D))
    arr = {"features": np.array(X.columns, dtype=str), "targets": np.array(TARGETS), "hypothesis": np.array("e-" if a.smp == "e" else "mu-"),
           "n_mlp": np.int64(3), "n_gbm": np.int64(5),
           "trained_on": np.array(f"tn_cal ({'elneg' if a.smp == 'e' else 'muon'} run 0, files 0-299), Coral 1527e03 raw, {len(idx)} events")}
    ref = []
    for k, mm in enumerate(("mlp0", "mlp1", "mlp2")):
        P, net, med, iqr, ys = mlp_fit_predict(X.iloc[idx], Y.iloc[idx], Xt, seed=100*MEMBERS[mm]["seed"], return_model=True)
        lin = [m for m in net if hasattr(m, "weight")]
        arr.update({f"mlp{k}_med": med.to_numpy(float), f"mlp{k}_iqr": iqr.to_numpy(float), f"mlp{k}_ys": np.asarray(ys, float)})
        for j, l in enumerate(lin):
            arr[f"mlp{k}_w{j}"] = l.weight.detach().numpy().astype(np.float64); arr[f"mlp{k}_b{j}"] = l.bias.detach().numpy().astype(np.float64)
        ref.append(P)
    g = MEMBERS["gbm"]
    gp = np.zeros((len(Xt), len(TARGETS)))
    for b in range(5):
        kw = dict(json.loads(g["params"]), random_state=g["seed"] + b)
        for j, t in enumerate(TARGETS):
            m = HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(X.iloc[idx], Y.iloc[idx][t])
            arr.update(export_trees(m, f"gbm{b}_{t}"))
            gp[:, j] += m.predict(Xt)/5
    ref.append(gp)
    lib = np.mean(ref, axis=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f"stack_coral_{a.smp}.npz", **arr)
    arr = dict(np.load(out/f"stack_coral_{a.smp}.npz"))
    n = min(a.check, len(Xt))
    mine = predict(arr, Xt.to_numpy(float)[:n])
    applied = np.mean([np.load(Path(a.dir)/"oof3"/f"tn_{a.smp}_C_{mm}.npz")["Pt"] for mm in MEMBERS], axis=0)
    print(f"{a.smp}: {out/f'stack_coral_{a.smp}.npz'} ({(out/f'stack_coral_{a.smp}.npz').stat().st_size/1e6:.1f} MB), "
          f"{len(arr['features'])} features, trained on {len(idx)} events")
    print(f"  numpy evaluator vs torch/sklearn (first {n} tn_val events): max |diff| = {np.max(np.abs(mine - lib[:n])):.2e}")
    d = np.abs(lib - applied)
    print(f"  retrained vs the stack applied to tn_val: median |diff| {np.median(d, axis=0).round(4)}, max {d.max(axis=0).round(3)}"
          f"  (targets {list(TARGETS)})")


if __name__ == "__main__":
    main()
