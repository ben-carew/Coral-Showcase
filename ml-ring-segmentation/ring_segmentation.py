"""
    ML ring segmentation: a sparse 3D U-Net (WatChMaL multi-ring inference)
    splits a sub-event's hits into rings, each of which Coral then fits as a
    single ring.

    Optional: needs WatChMaL's multi-ring inference package, torch and spconv
    (Linux only), and the trained model's files, all named in the [ml] section.
"""
import sys

import numpy as np
import pandas as pd

from fitqun_fit.subeventing import one_hit_per_pmt

# [ml] keys that configure Coral rather than the segmenter
_CORAL_KEYS = ("segmentation", "watchmal_path", "time_offset_ns")


def load_segmenter(ml_config):
    """(MultiRingSegmenter, time offset in ns) from the [ml] section's keys."""
    cfg = dict(ml_config)
    path = cfg.get("watchmal_path", "").strip()
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        from watchmal.engine.multiring.segmentation import MultiRingSegmenter
    except ImportError as e:
        raise ImportError("[ml] segmentation needs WatChMaL's multi-ring inference package "
                          "(set [ml] watchmal_path), torch and spconv") from e
    time_offset = float(cfg.get("time_offset_ns", 0.0))
    for key in _CORAL_KEYS:
        cfg.pop(key, None)
    cfg["use_optimized_lut"] = str(cfg.get("use_optimized_lut", "true")).lower() in ("1", "true", "yes", "on")
    device = cfg.pop("device", None) or None
    return MultiRingSegmenter(cfg["train_output_dir"], cfg, device=device), time_offset


def segment_rings(segmenter, hits, time_offset):
    """[(ring_id, ring hits)] for one sub-event. The network was trained on raw
    WCSim digit times, i.e. before the trigger offset Coral subtracts. A PMT
    shared by rings appears in each, with the charge the network assigns it."""
    rings = segmenter.segment(hits[["x", "y", "z"]].to_numpy(np.float32),
                              hits.q.to_numpy(np.float32),
                              hits.t.to_numpy(np.float32) + np.float32(time_offset),
                              hits.tubeid.to_numpy(np.int64))
    geometry = hits[["tubeid", "mPMTid", "x", "y", "z", "trigger"]].drop_duplicates("tubeid")
    out = []
    for ring in rings:
        ring_hits = pd.DataFrame({"tubeid": np.asarray(ring.tube_id, dtype=np.int64),
                                  "q": np.asarray(ring.charge_pe, dtype=np.float64),
                                  "t": np.asarray(ring.time_ns, dtype=np.float64) - time_offset})
        ring_hits = one_hit_per_pmt(ring_hits.merge(geometry, on="tubeid", how="inner"))
        if len(ring_hits):
            out.append((int(ring.ring_id), ring_hits))
    return out
