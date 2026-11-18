"""Rank unhit PMTs by how much they matter to the fit, so only the top slice is kept.

Coral evaluates the charge likelihood over every unhit PMT, which is by far the
largest term in the sum: the unhit PMTs outnumber the hit ones ~10:1 and their
contribution is recomputed at every likelihood call.  `unhit_keep_fraction`
already trims that set, but the historical selector is a *uniform* stride over
tube ids -- it throws away the ring-edge PMTs that actually constrain the vertex
and direction just as readily as the ones on the far side of the tank.

This module supplies importance-ordered alternatives.  All of them rank unhit
PMTs by a proxy for |dNLL|, the change in that PMT's unhit charge term between
the seed and the converged fit, which is what makes a PMT informative:

    ring        -|perpendicular distance from the prefit Cherenkov cone|
    model       an offline-trained gradient-boosted regressor over 4-31
                geometric features (the shipped one uses 4 ring features)

The prefit (vertex, direction, t0) runs before any unhit selection, so every
feature here is legitimately available at inference time.

Models are exported into plain .npz arrays and evaluated below with numba, so
neither xgboost nor torch is needed at run time.

Ranked subsets are for GEOMETRY steps only: the momentum is set by the total
predicted light, which only an unbiased (uniform) sample of the unhit PMTs
estimates, and ranking conditions on a single prefit ring, so it is wrong for
multi-ring events. A ranked subset also needs a calibrated unhit-term weight,
not n_total/n_kept (see likelihood.neg_loglikelihood_factory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from fitqun_tuning.corrections import resolve

C0_CM_PER_NS = 29.9792458
REFRACTION_INDEX = 1.38
CHERENKOV_ANGLE_DEG = float(np.degrees(np.arccos(1.0 / REFRACTION_INDEX)))
DENSITY_RADII_CM = (100.0, 250.0, 500.0)
DENSITY_TIME_WINDOW_NS = 20.0

# Training-time feature order. A model records the subset it uses; the indices
# into this list are resolved once at load time.
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
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Features that need the three cKDTree neighbourhood sweeps -- ~50 ms/event, so
# they are skipped entirely when the chosen model does not ask for them.
_KDTREE_FEATURES = frozenset(
    [f"{stem}_r{int(r)}" for r in DENSITY_RADII_CM
     for stem in ("hit_count", "charge_sum", "time_std", "time_density")]
    + ["charge_shell_asym_in_mid", "charge_shell_asym_mid_out"]
)


# --------------------------------------------------------------------- features

def compute_features(unhit_pos, unhit_ori, unhit_loc, hit_pos, hit_q, hit_t,
                     vertex, direction, t0, radius, length, wanted):
    """One row per unhit PMT, columns in FEATURE_NAMES order.

    `wanted` is the set of feature names the caller actually needs; anything else
    is left at zero.  Only the cKDTree neighbourhood block is genuinely expensive,
    and it is the block the cheap models do not use.
    """
    n = len(unhit_pos)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0 or len(hit_pos) == 0:
        return out

    rel = unhit_pos - vertex[None, :]
    dist_vtx = np.linalg.norm(rel, axis=1)
    rel_unit = rel / np.maximum(dist_vtx[:, None], 1e-12)
    cos_theta = np.clip(rel_unit @ direction, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    dtheta = theta_deg - CHERENKOV_ANGLE_DEG
    ring_offset = dist_vtx * np.sin(np.radians(dtheta))

    ori_unit = unhit_ori / np.maximum(
        np.linalg.norm(unhit_ori, axis=1, keepdims=True), 1e-12)
    photon_incidence = np.sum(rel_unit * (-ori_unit), axis=1)

    hit_dist_vtx = np.linalg.norm(hit_pos - vertex[None, :], axis=1)
    hit_tof_res = hit_t - t0 - hit_dist_vtx * REFRACTION_INDEX / C0_CM_PER_NS

    tree = cKDTree(hit_pos)
    nn_dist, nn_idx = tree.query(unhit_pos, k=1)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)

    total_q = float(hit_q.sum())
    bary = (np.average(hit_pos, axis=0, weights=hit_q) if total_q > 0
            else hit_pos.mean(axis=0))

    rho = np.hypot(unhit_pos[:, 0], unhit_pos[:, 1])
    pmt_unit = unhit_pos / np.maximum(
        np.linalg.norm(unhit_pos, axis=1)[:, None], 1e-12)
    dist_to_edge = np.where(unhit_loc == 1,
                            0.5 * length - np.abs(unhit_pos[:, 2]),
                            radius - rho)

    nearest_tof = hit_tof_res[nn_idx]
    nearest_tof = nearest_tof - float(np.median(nearest_tof))

    col = FEATURE_INDEX
    out[:, col["dtheta_cherenkov"]] = dtheta
    out[:, col["abs_dtheta_cherenkov"]] = np.abs(dtheta)
    out[:, col["ring_offset_cm"]] = ring_offset
    out[:, col["abs_ring_offset_cm"]] = np.abs(ring_offset)
    out[:, col["dist_vertex_cm"]] = dist_vtx
    out[:, col["cos_theta_pmt"]] = cos_theta
    out[:, col["photon_incidence"]] = photon_incidence
    out[:, col["nearest_hit_tof_res"]] = nearest_tof
    out[:, col["nearest_hit_distance"]] = nn_dist
    out[:, col["nearest_hit_charge"]] = hit_q[nn_idx]
    out[:, col["distance_to_barycenter"]] = np.linalg.norm(unhit_pos - bary[None, :], axis=1)
    out[:, col["pmt_rho_cm"]] = rho
    out[:, col["pmt_z_cm"]] = unhit_pos[:, 2]
    out[:, col["dist_to_edge_cm"]] = dist_to_edge
    out[:, col["pmt_ori_dot_inward"]] = np.sum(ori_unit * (-pmt_unit), axis=1)
    out[:, col["loc_top"]] = (unhit_loc == 0)
    out[:, col["loc_side"]] = (unhit_loc == 1)

    if not (wanted & _KDTREE_FEATURES):
        return out

    for r in DENSITY_RADII_CM:
        nbrs = tree.query_ball_point(unhit_pos, r)
        counts = np.fromiter((len(v) for v in nbrs), dtype=np.float64, count=n)
        charge_sum = np.fromiter(
            (float(hit_q[v].sum()) if v else 0.0 for v in nbrs),
            dtype=np.float64, count=n)
        time_std = np.fromiter(
            (float(hit_t[v].std()) if len(v) > 1 else 0.0 for v in nbrs),
            dtype=np.float64, count=n)
        t_density = np.fromiter(
            (float(np.sum(np.abs(hit_t[v] - hit_t[nn_idx[i]]) <= DENSITY_TIME_WINDOW_NS))
             if v else 0.0 for i, v in enumerate(nbrs)),
            dtype=np.float64, count=n)
        tag = int(r)
        out[:, col[f"hit_count_r{tag}"]] = counts
        out[:, col[f"charge_sum_r{tag}"]] = charge_sum
        out[:, col[f"time_std_r{tag}"]] = time_std
        out[:, col[f"time_density_r{tag}"]] = t_density

    inner, mid, outer = (int(r) for r in sorted(DENSITY_RADII_CM))
    q_in = out[:, col[f"charge_sum_r{inner}"]]
    q_mid = out[:, col[f"charge_sum_r{mid}"]] - q_in
    q_out = out[:, col[f"charge_sum_r{outer}"]] - out[:, col[f"charge_sum_r{mid}"]]
    out[:, col["charge_shell_asym_in_mid"]] = (q_in - q_mid) / np.maximum(q_in + q_mid, 1e-6)
    out[:, col["charge_shell_asym_mid_out"]] = (q_mid - q_out) / np.maximum(q_mid + q_out, 1e-6)
    return out


# ------------------------------------------------------------------- inference

@njit(cache=True, parallel=True, fastmath=True)
def _xgb_predict(X, tree_start, split_idx, thresh, left, right, default_left, base_score):
    n_rows = X.shape[0]
    n_trees = len(tree_start) - 1
    out = np.full(n_rows, base_score)
    for i in prange(n_rows):
        total = 0.0
        for t in range(n_trees):
            off = tree_start[t]
            node = 0
            while left[off + node] != -1:
                value = X[i, split_idx[off + node]]
                if np.isnan(value):
                    go_left = default_left[off + node] == 1
                else:
                    go_left = value < thresh[off + node]
                node = left[off + node] if go_left else right[off + node]
            total += thresh[off + node]
        out[i] += total
    return out


class _XGBModel:
    """Flattened gradient-boosted ensemble; leaf values live in `thresh`."""

    def __init__(self, data):
        self.feature_names = [str(s) for s in data["feature_names"]]
        self.base_score = float(data["base_score"])
        self.tree_start = np.ascontiguousarray(data["tree_start"], dtype=np.int64)
        self.split_idx = np.ascontiguousarray(data["split_idx"], dtype=np.int64)
        self.thresh = np.ascontiguousarray(data["thresh"], dtype=np.float64)
        self.left = np.ascontiguousarray(data["left"], dtype=np.int64)
        self.right = np.ascontiguousarray(data["right"], dtype=np.int64)
        self.default_left = np.ascontiguousarray(data["default_left"], dtype=np.uint8)

    def predict(self, X):
        return _xgb_predict(np.ascontiguousarray(X, dtype=np.float64),
                            self.tree_start, self.split_idx, self.thresh,
                            self.left, self.right, self.default_left, self.base_score)


def load_model(path):
    """Load an .npz exported by export_pmt_selector_models.py."""
    data = np.load(Path(path), allow_pickle=False)
    kind = str(data["kind"])
    if kind == "xgboost":
        return _XGBModel(data)
    raise ValueError(f"unknown PMT-selector model kind '{kind}' in {path}")


# -------------------------------------------------------------------- selector

class UnhitRanker:
    """Orders the unhit PMTs of one sub-event by predicted importance.

    `method` is one of:
        uniform      no ranking; the caller falls back to the historical stride
        ring         proximity to the prefit Cherenkov cone (no model, ~1 ms)
        model        the .npz regressor at `model_path`
    """

    METHODS = ("uniform", "ring", "model")

    def __init__(self, method="uniform", model_path=None):
        method = (method or "uniform").lower()
        if method not in self.METHODS:
            raise ValueError(f"[pmt_selection] method must be one of {self.METHODS}")
        self.method = method
        self.model = None
        self.feature_slots = None
        self.wanted = set()
        if method == "model":
            if not model_path:
                raise ValueError("select=model needs [pmt_selection] model")
            self.model = load_model(resolve(model_path))
            self.wanted = set(self.model.feature_names)
            unknown = self.wanted - set(FEATURE_NAMES)
            if unknown:
                raise ValueError(f"model {model_path} needs unknown features: {sorted(unknown)}")
            self.feature_slots = np.array(
                [FEATURE_INDEX[n] for n in self.model.feature_names], dtype=np.int64)
        elif method == "ring":
            self.wanted = {"abs_ring_offset_cm"}

    @property
    def active(self) -> bool:
        return self.method != "uniform"

    def score(self, features):
        """Predicted importance per unhit PMT, higher = keep."""
        if self.method == "ring":
            return -np.abs(features[:, FEATURE_INDEX["abs_ring_offset_cm"]])
        return self.model.predict(features[:, self.feature_slots])

    def order(self, unhit_pos, unhit_ori, unhit_loc, hit_pos, hit_q, hit_t,
              vertex, direction, t0, radius, length):
        """Positional indices of the unhit PMTs, most important first."""
        if not self.active or len(unhit_pos) == 0:
            return None
        features = compute_features(unhit_pos, unhit_ori, unhit_loc,
                                    hit_pos, hit_q, hit_t,
                                    vertex, direction, t0, radius, length, self.wanted)
        return np.argsort(-self.score(features), kind="stable")


class UnhitRankerBank:
    """The set of selectors one run can call on, sharing a single feature pass.

    A sequential fit can use a different selector per step -- ranked while the
    vertex and direction are floating, uniform while the momentum is, since the
    momentum is set by the TOTAL predicted light and so needs an unbiased sample.
    Computing the features once for the union of what those selectors ask for
    matters: the cKDTree neighbourhood block is ~0.4 s/event, so scoring two
    model-based selectors separately would pay it twice.
    """

    def __init__(self, methods, model_path=None):
        wanted_methods = [m for m in dict.fromkeys(methods) if m]
        self.rankers = {m: UnhitRanker(m, model_path) for m in wanted_methods}
        self.wanted = set()
        for ranker in self.rankers.values():
            self.wanted |= ranker.wanted

    @property
    def active(self) -> bool:
        return any(r.active for r in self.rankers.values())

    def orders(self, unhit_pos, unhit_ori, unhit_loc, hit_pos, hit_q, hit_t,
               vertex, direction, t0, radius, length):
        """{method: order}, with None for `uniform` (the caller's stride)."""
        orders = {m: None for m in self.rankers}
        if not self.active or len(unhit_pos) == 0:
            return orders
        features = compute_features(unhit_pos, unhit_ori, unhit_loc,
                                    hit_pos, hit_q, hit_t,
                                    vertex, direction, t0, radius, length, self.wanted)
        for method, ranker in self.rankers.items():
            if ranker.active:
                orders[method] = np.argsort(-ranker.score(features), kind="stable")
        return orders


def direction_from_angles(theta: float, phi: float) -> np.ndarray:
    sin_theta = np.sin(theta)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)])
