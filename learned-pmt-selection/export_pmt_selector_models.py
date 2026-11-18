#!/usr/bin/env python3

"""Export the trained DNLL-predictor models to dependency-free .npz artifacts.

Coral has to run these models inside the reconstruction, on the laptop AND on
CC-IN2P3.  The cluster's Coral environment has no xgboost, and the 31-feature
booster ships as a 58 MB JSON, so neither model is usable in its native form.

Both are pure feed-forward functions of a fixed feature vector, so each is
flattened here into plain numpy arrays that `fitqun_fit.pmt_selection` evaluates
with numba:

  * XGBoost -> flat node arrays (feature index, threshold, children, leaf value)
    concatenated over trees, plus the tree offsets.
  * MLP     -> the MinMaxNorm running_min/running_max buffers and the four
    Linear layers' weights/biases.

Every export is verified against the original library's own prediction on random
feature vectors before it is written.

    ./export_pmt_selector_models.py [--outdir DIR]
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPTS_DIR / "DNLLPredictorTrainedModels"
DEFAULT_OUT = MODEL_DIR / "exported"

# Feature order used at training time; every model records its own subset of it.
FEATURE_NAMES = [
    "dtheta_cherenkov", "abs_dtheta_cherenkov", "ring_offset_cm", "abs_ring_offset_cm",
    "dist_vertex_cm", "cos_theta_pmt", "photon_incidence", "nearest_hit_tof_res",
    "nearest_hit_distance", "nearest_hit_charge", "distance_to_barycenter",
    "pmt_rho_cm", "pmt_z_cm", "dist_to_edge_cm", "pmt_ori_dot_inward",
    "loc_top", "loc_side",
    "hit_count_r100", "charge_sum_r100", "time_std_r100", "time_density_r100",
    "hit_count_r250", "charge_sum_r250", "time_std_r250", "time_density_r250",
    "hit_count_r500", "charge_sum_r500", "time_std_r500", "time_density_r500",
    "charge_shell_asym_in_mid", "charge_shell_asym_mid_out",
]

XGB_MODELS = {
    "xgb_simple": "XGBoost/dnll_selector_v01.json",
    "xgb_v02":    "XGBoost/dnll_selector_v02.json",
    "xgb_full":   "XGBoost/xgb_model_final_tuning.json",
}
MLP_MODEL = "MLP/mlp_prefit_final_model.pt"


def export_xgboost(src: Path, dst: Path) -> dict:
    """Flatten an XGBoost booster JSON into concatenated node arrays."""
    model = json.loads(src.read_text())
    learner = model["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    feature_names = learner.get("feature_names") or FEATURE_NAMES
    base_score = float(json.loads(learner["learner_model_param"]["base_score"])[0]) \
        if learner["learner_model_param"]["base_score"].startswith("[") \
        else float(learner["learner_model_param"]["base_score"])

    split_idx, thresh, left, right, default_left, tree_start = [], [], [], [], [], [0]
    for tree in trees:
        n = len(tree["left_children"])
        split_idx.append(np.asarray(tree["split_indices"], dtype=np.int32))
        thresh.append(np.asarray(tree["split_conditions"], dtype=np.float64))
        left.append(np.asarray(tree["left_children"], dtype=np.int32))
        right.append(np.asarray(tree["right_children"], dtype=np.int32))
        default_left.append(np.asarray(tree["default_left"], dtype=np.uint8))
        tree_start.append(tree_start[-1] + n)

    payload = dict(
        kind=np.array("xgboost"),
        feature_names=np.array(feature_names),
        base_score=np.float64(base_score),
        tree_start=np.asarray(tree_start, dtype=np.int64),
        split_idx=np.concatenate(split_idx),
        thresh=np.concatenate(thresh),
        left=np.concatenate(left),
        right=np.concatenate(right),
        default_left=np.concatenate(default_left),
    )
    np.savez_compressed(dst, **payload)
    return payload


def export_mlp(src: Path, dst: Path) -> dict:
    """Flatten the MLP checkpoint: MinMaxNorm buffers + four Linear layers."""
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    feature_names = list(ckpt["feature_cols"])

    payload = {
        "kind": np.array("mlp"),
        "feature_names": np.array(feature_names),
        "running_min": state["net.0.running_min"].numpy().astype(np.float64),
        "running_max": state["net.0.running_max"].numpy().astype(np.float64),
        "eps": np.float64(1e-6),
    }
    # net.1/2/3 are Sequential(Linear, ReLU); net.4 is the bare output Linear.
    for i, key in enumerate(["net.1.0", "net.2.0", "net.3.0", "net.4"]):
        payload[f"w{i}"] = state[f"{key}.weight"].numpy().astype(np.float64)
        payload[f"b{i}"] = state[f"{key}.bias"].numpy().astype(np.float64)
    np.savez_compressed(dst, **payload)
    return payload


# ---------------------------------------------------------------- verification

def _xgb_predict_ref(src: Path, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(src))
    return booster.inplace_predict(X.astype(np.float32))


def _mlp_predict_ref(src: Path, X: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn as nn

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    class MinMaxNorm(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.register_buffer("running_min", torch.zeros(n))
            self.register_buffer("running_max", torch.ones(n))

        def forward(self, x):
            return (x - self.running_min) / (self.running_max - self.running_min + 1e-6)

    net = nn.Sequential(
        MinMaxNorm(ckpt["in_features"]),
        nn.Sequential(nn.Linear(ckpt["in_features"], 256), nn.ReLU()),
        nn.Sequential(nn.Linear(256, 128), nn.ReLU()),
        nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
        nn.Linear(64, ckpt["out_features"]),
        nn.Sigmoid(),
    )
    net.load_state_dict({k[len("net."):]: v for k, v in state.items()})
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(X.astype(np.float32))).numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Import the Coral-side evaluator so the exported artifact is checked with
    # exactly the code that will run inside the reconstruction.
    import sys
    sys.path.insert(0, str(SCRIPTS_DIR.parents[1] / "Software/worktrees/c/src"))
    from fitqun_fit.pmt_selection import load_model

    # The libraries accumulate in float32 and we accumulate in float64, so values
    # differ by a near-constant ~2e-4 on a 550-tree sum. Only the ORDER is used
    # downstream, and the residual rank swaps sit inside that noise (predictions
    # within ~2e-6 of each other), so the acceptance test is on the ranking:
    # rank correlation and, more to the point, the top-fraction sets Coral keeps.
    from scipy.stats import spearmanr

    def check(name, dst, got, want):
        err = float(np.max(np.abs(got - want)))
        rho = float(spearmanr(got, want).statistic)
        order_got, order_want = np.argsort(-got), np.argsort(-want)
        overlaps = {f: len(set(order_got[:int(f * len(got))]) &
                           set(order_want[:int(f * len(got))])) / int(f * len(got))
                    for f in (0.05, 0.10, 0.20, 0.30)}
        worst = min(overlaps.values())
        print(f"{name:>11}: {dst.stat().st_size/1e6:6.2f} MB  max|err| = {err:.2e}  "
              f"spearman = {rho:.9f}  worst top-k overlap = {worst:.4f}")
        assert rho > 1 - 1e-6, f"{name} export does not reproduce the reference ranking"
        assert worst > 0.999, f"{name} export changes the kept PMT set ({overlaps})"
        assert err < 1e-3, f"{name} export values drift too far ({err:.2e})"

    # Test points are drawn inside the training feature ranges (the MLP's
    # MinMaxNorm buffers record them). Far outside, the MLP's output sigmoid
    # saturates and every prediction collapses to the same value, which makes a
    # ranking comparison meaningless rather than informative.
    src_mlp, dst_mlp = MODEL_DIR / MLP_MODEL, args.outdir / "mlp.npz"
    mlp_payload = export_mlp(src_mlp, dst_mlp)
    lo, hi = mlp_payload["running_min"], mlp_payload["running_max"]

    rng = np.random.default_rng(0)
    n_test = 20_000

    def sample(feature_names):
        cols = [FEATURE_NAMES.index(f) for f in feature_names]
        return rng.uniform(lo[cols], hi[cols],
                           size=(n_test, len(cols))).astype(np.float32)

    for name, rel in XGB_MODELS.items():
        src, dst = MODEL_DIR / rel, args.outdir / f"{name}.npz"
        export_xgboost(src, dst)
        model = load_model(dst)
        X = sample(model.feature_names)
        check(name, dst, model.predict(X.astype(np.float64)), _xgb_predict_ref(src, X))

    model = load_model(dst_mlp)
    X = sample(model.feature_names)
    check("mlp", dst_mlp, model.predict(X.astype(np.float64)), _mlp_predict_ref(src_mlp, X))

    print(f"\nExported to {args.outdir}")


if __name__ == "__main__":
    main()
