"""Cramer-Rao lower bounds on the seven single-ring reconstruction parameters.

    model = CRBModel.load("Output/Sextant/crb_model/crb_model.npz")
    b = bound(model, "e-", vertex, direction, p)      # one event
    b["sigma_long"], b["sigma_angle"], b["sigma_p_rel"]

Build the model once with `sextant crb build` (see build.py). Everything here is
numpy; building additionally needs h5py and scikit-learn.
"""

from .fisher import PARAMS, bound, fisher, shower_sigma
from .model import CRBModel

__all__ = ["CRBModel", "PARAMS", "bound", "fisher", "shower_sigma"]
