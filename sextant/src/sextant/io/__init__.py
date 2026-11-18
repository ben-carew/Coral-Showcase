"""Readers: every supported on-disk format into the canonical tables."""

from .dataset import Arm, Truth, detect_format, load_arms
from .schema import (HYP_ELECTRON, HYP_MUON, POINT_REF, POINT_SELF, POINT_TRUTH,
                     RING_LEADING, STAGE_FINAL, columns, describe, empty)

__all__ = [
    "Arm", "Truth", "detect_format", "load_arms",
    "HYP_ELECTRON", "HYP_MUON", "POINT_SELF", "POINT_TRUTH", "POINT_REF",
    "RING_LEADING", "STAGE_FINAL", "columns", "describe", "empty",
]
