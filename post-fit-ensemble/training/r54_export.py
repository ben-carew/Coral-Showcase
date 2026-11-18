#!/usr/bin/env python
"""Export the frozen R54 v2 post-fit estimators (sklearn HistGradientBoosting) to plain arrays for Coral
(src/fitqun_tuning/data/post_fit_estimator_{e,mu}.npz), and check the array evaluator reproduces
sklearn's predictions on the fresh files.

    python r54_export.py --out Software/worktrees/a/src/fitqun_tuning/data
"""
import argparse, importlib.util, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def export(models, targets):
    out = {}
    for c in targets:
        m = models[c]
        nodes = [p[0].nodes for p in m._predictors]
        roots = np.cumsum([0] + [len(n) for n in nodes[:-1]]).astype(np.int64)
        allnodes = np.concatenate(nodes)
        assert not allnodes["is_categorical"].any()
        off = np.repeat(roots, [len(n) for n in nodes])
        out[f"{c}_feature"] = allnodes["feature_idx"].astype(np.int64)
        out[f"{c}_threshold"] = allnodes["num_threshold"].astype(np.float64)
        out[f"{c}_nan_left"] = allnodes["missing_go_to_left"].astype(np.bool_)
        out[f"{c}_left"] = (allnodes["left"].astype(np.int64) + off)
        out[f"{c}_right"] = (allnodes["right"].astype(np.int64) + off)
        out[f"{c}_leaf"] = allnodes["is_leaf"].astype(np.bool_)
        out[f"{c}_value"] = allnodes["value"].astype(np.float64)
        out[f"{c}_roots"] = roots
        out[f"{c}_baseline"] = np.float64(np.ravel(m._baseline_prediction)[0])
    return out


def predict(arr, c, X):
    """Array evaluator (the same walk Coral does)."""
    f, th, nl, le, ri, lf, va = (arr[f"{c}_{k}"] for k in ("feature", "threshold", "nan_left", "left", "right", "leaf", "value"))
    out = np.full(len(X), float(arr[f"{c}_baseline"]))
    for i, x in enumerate(X):
        s = 0.0
        for n in arr[f"{c}_roots"]:
            while not lf[n]:
                v = x[f[n]]
                n = (le[n] if nl[n] else ri[n]) if np.isnan(v) else (le[n] if v <= th[n] else ri[n])
            s += va[n]
        out[i] += s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Software/worktrees/a/src/fitqun_tuning/data"))
    ap.add_argument("--prefix", default="r54")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("v1", ROOT / "Analysis/Scripts/r54_heldout.py")
    v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
    for smp, hyp, sample in (("e", "e-", "electron"), ("mu", "mu-", "muon")):
        d = pickle.load(open(ROOT / f"Output/Sextant/{a.prefix}/post_fit_estimator_v2_{smp}.pkl", "rb"))
        arr = export(d["models"], v1.TARGETS)
        arr["features"] = np.array(d["features"])
        arr["targets"] = np.array(v1.TARGETS)
        arr["gate_towall_cm"] = np.float64(d["gate_towall_cm"])
        arr["wall_radius_cm"], arr["wall_half_height_cm"] = np.float64(3250.0), np.float64(3300.0)
        np.savez(Path(a.out) / f"post_fit_estimator_{smp}.npz", **arr)
        truth = v1.Truth.from_file(ROOT / f"Output/Truth/truth_{sample}_v1.12.29.root")
        Xb, _, _, _ = v1.table(f"{a.prefix}_fresh_{smp}", hyp, sample, truth)
        Xs = Xb.to_numpy(float)[:300]
        worst = max(np.max(np.abs(predict(arr, c, Xs) - d["models"][c].predict(Xb.iloc[:300]))) for c in v1.TARGETS)
        print(f"{smp}: exported {len(arr['logp_value'])} logp nodes, gate {d['gate_towall_cm']:.0f} cm; max |array - sklearn| = {worst:.2e}")


if __name__ == "__main__":
    main()
