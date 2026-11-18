"""Sextant -- validation and comparison of water-Cherenkov reconstruction.

Two halves that integrate but do not depend on each other:

* `sextant.io` / `metrics` / `plots` / `report` -- the validation half. Give it
  reconstruction files and a truth file and it produces residuals, resolutions,
  PID, ring counting, intermediate-quantity comparisons, a PDF and CSV tables.
  Works on hand-picked files, with no experiment machinery involved.
* `sextant.run` / `probe` -- the experiment half. Launches Coral and fiTQun arms
  locally or on the cluster from one spec, keeps their configurations in step,
  and collects the results into the arms the validation half consumes.
"""

__version__ = "0.1.0"

from .io import Arm, Truth, load_arms

__all__ = ["Arm", "Truth", "load_arms", "__version__"]
