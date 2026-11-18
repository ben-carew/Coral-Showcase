"""Read Coral reconstruction output into the canonical tables.

Coral writes per-job HDF5 which `merge_coral.py` consolidates into one ROOT file
holding up to four flat TTrees -- `fits`, `prefits`, and (when those fitters ran)
`pi0fits` and `multiringfits`. All four are read here; which of them exist is a
property of the run, not something the caller should have to know.

Two Coral conventions are undone at the door so that nothing downstream has to
carry them:

* the fit hypothesis is stored as an int8 `particle_code` (0 = e-, 1 = mu-) --
  decoded back to the 'e-'/'mu-' strings the rest of Sextant uses;
* direction is stored as `px, py, pz`, which despite the name is already a unit
  vector, not a momentum vector (the magnitude lives in `p`). It is normalised
  again here anyway, because a zero-length direction from a failed fit should
  become NaN rather than silently pass through as a valid axis.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd

from . import _root
from .schema import (CORAL_PARTICLE_CODES, HYP_ELECTRON, POINT_SELF,
                     RING_LEADING, STAGE_FINAL, conform, empty, unit_vector)

#: The tree `merge_coral.py` always writes, and the marker that a file is Coral's.
MAIN_TREE = "fits"
PREFIT_TREE = "prefits"
PI0_TREE = "pi0fits"
MULTIRING_TREE = "multiringfits"


def is_coral(path) -> bool:
    """Coral and fiTQun merges both call their main tree `fits`; the int8
    `particle_code` branch is what only Coral writes."""
    try:
        names = _root.tree_names(path)
        if MAIN_TREE not in names:
            return False
        return "particle_code" in _root.branches(path, MAIN_TREE)
    except Exception:
        return False


def _decode_hypothesis(codes) -> np.ndarray:
    codes = np.asarray(codes)
    out = np.full(codes.shape, "?", dtype=object)
    for code, label in CORAL_PARTICLE_CODES.items():
        out[codes == code] = label
    return out


def _direction(df: pd.DataFrame) -> tuple:
    return unit_vector(df["px"].to_numpy(float),
                       df["py"].to_numpy(float),
                       df["pz"].to_numpy(float))


def read_fits(path, *, only_success: bool = False) -> pd.DataFrame:
    """The single-ring fits: one row per (event, sub_event, hypothesis).

    `only_success` is off by default deliberately. Requiring success on *both*
    hypotheses is what silently removes most true muons from a PID sample (the
    e- fit routinely fails on them and vice versa), so the cut belongs in the
    metric that wants it, applied to the hypothesis it actually concerns.
    """
    df = _root.read_tree(path, MAIN_TREE)
    if df.empty:
        return empty("fits")
    dx, dy, dz = _direction(df)
    out = pd.DataFrame({
        "event": df["event"], "sub_event": df["sub_event"],
        "hypothesis": _decode_hypothesis(df["particle_code"]),
        "ring": RING_LEADING, "nrings": 1, "stage": STAGE_FINAL,
        "success": df["success"].astype(bool),
        "nll": df["nL"], "totmu": np.nan,
        "t": df["t"], "x": df["x"], "y": df["y"], "z": df["z"],
        "dx": dx, "dy": dy, "dz": dz, "p": df["p"], "dconv": np.nan,
    })
    if only_success:
        out = out[out["success"]]
    return conform(out, "fits")


def read_prefits(path) -> pd.DataFrame:
    """Coral's prefit (the fast vertex/time search before any likelihood fit).

    Returned in the `fits` schema with hypothesis '(prefit)': the prefit runs
    once per sub-event, before any particle hypothesis is chosen, so labelling
    it 'e-' would invite a meaningless comparison against fiTQun's e- fit.
    """
    if not _root.has_tree(path, PREFIT_TREE):
        return empty("fits")
    df = _root.read_tree(path, PREFIT_TREE)
    if df.empty:
        return empty("fits")
    dx, dy, dz = _direction(df)
    return conform(pd.DataFrame({
        "event": df["event"], "sub_event": df["sub_event"],
        "hypothesis": "(prefit)", "ring": RING_LEADING, "nrings": 1,
        "stage": STAGE_FINAL,
        "success": df["success"].astype(bool) if "success" in df else True,
        "nll": np.nan, "totmu": np.nan,
        "t": df["t"], "x": df["x"], "y": df["y"], "z": df["z"],
        "dx": dx, "dy": dy, "dz": dz, "p": df["p"], "dconv": np.nan,
    }), "fits")


def read_multiring(path) -> pd.DataFrame:
    """The multi-ring fit: one row per ring, sharing a vertex within a fit.

    Coral also writes the lower-ring-count stages of each fit (`final` False):
    they get stage = nrings, so the reported fit stays the STAGE_FINAL one and an
    offline ring-counting cut can pick the stage it would have stopped at."""
    if not _root.has_tree(path, MULTIRING_TREE):
        return empty("fits")
    df = _root.read_tree(path, MULTIRING_TREE)
    if df.empty:
        return empty("fits")
    dx, dy, dz = _direction(df)
    return conform(pd.DataFrame({
        "event": df["event"], "sub_event": df["sub_event"],
        "hypothesis": _decode_hypothesis(df["particle_code"]),
        "ring": df["ring"], "nrings": df["nrings"],
        "stage": (np.where(df["final"].astype(bool), STAGE_FINAL, df["nrings"])
                  if "final" in df else STAGE_FINAL),
        "success": df["success"].astype(bool),
        "nll": df["nL"], "totmu": np.nan,
        "t": df["t"], "x": df["x"], "y": df["y"], "z": df["z"],
        "dx": dx, "dy": dy, "dz": dz, "p": df["p"],
        "dconv": df["dconv"] if "dconv" in df else np.nan,
    }), "fits")


def read_pi0(path) -> pd.DataFrame:
    """The two-ring pi0 fit, unpacked to one row per photon ring.

    Coral saves both the free-mass and the pi0-mass-constrained fit for every
    event. They are different fits of the same data, so they are kept apart by
    hypothesis label rather than being averaged or silently deduplicated.
    """
    if not _root.has_tree(path, PI0_TREE):
        return empty("fits")
    df = _root.read_tree(path, PI0_TREE)
    if df.empty:
        return empty("fits")
    frames = []
    for ring, suffix in enumerate(("1", "2")):
        dx, dy, dz = unit_vector(df[f"px{suffix}"].to_numpy(float),
                                 df[f"py{suffix}"].to_numpy(float),
                                 df[f"pz{suffix}"].to_numpy(float))
        constrained = df["mass_constraint"].astype(bool).to_numpy()
        frames.append(pd.DataFrame({
            "event": df["event"], "sub_event": df["sub_event"],
            "hypothesis": np.where(constrained, "pi0-constrained", "pi0-free"),
            "ring": ring, "nrings": 2, "stage": STAGE_FINAL,
            "success": df["success"].astype(bool),
            "nll": df["nL"], "totmu": np.nan,
            "t": df["t"], "x": df["x"], "y": df["y"], "z": df["z"],
            "dx": dx, "dy": dy, "dz": dz,
            "p": df[f"p{suffix}"], "dconv": df[f"dconv{suffix}"],
        }))
    return conform(pd.concat(frames, ignore_index=True), "fits")


def read_pi0_masses(path) -> pd.DataFrame:
    """Per-event reconstructed pi0 mass and opening angle (not a `fits` table)."""
    if not _root.has_tree(path, PI0_TREE):
        return pd.DataFrame(columns=["event", "sub_event", "constrained",
                                     "mass", "opening_angle", "nll", "success"])
    df = _root.read_tree(path, PI0_TREE)
    return pd.DataFrame({
        "event": df["event"].astype("int64"),
        "sub_event": df["sub_event"].astype("int32"),
        "constrained": df["mass_constraint"].astype(bool),
        "mass": df["mass"].astype(float),
        "opening_angle": df["opening_angle"].astype(float),
        "nll": df["nL"].astype(float),
        "success": df["success"].astype(bool),
    })


def read_nll_terms(path, *, point: str = POINT_SELF) -> pd.DataFrame:
    """Coral's per-term NLL breakdown, written when `[in_out] dump_nll_terms`.

    `path` may be the `nll_terms.csv` itself or a run/job directory containing
    one. The file is appended to by every job, so a header line can appear more
    than once in a concatenated copy; those repeats are dropped.
    """
    csv_path = _find_job_csv(path, "nll_terms.csv")
    if csv_path is None:
        return empty("nll_terms")

    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("event") in (None, "", "event"):
                continue                      # repeated header from an append
            try:
                event = int(float(row["event"]))
                sub_event = int(float(row.get("sub_event", 0) or 0))
            except (TypeError, ValueError):
                continue
            for term in ("unhit", "phit", "cpdf", "tpdf"):
                if term not in row or row[term] in (None, ""):
                    continue
                rows.append({
                    "event": event, "sub_event": sub_event,
                    "hypothesis": row.get("particle", HYP_ELECTRON),
                    "point": point, "term": term, "value": float(row[term]),
                })
            if row.get("nll_fit") not in (None, ""):
                rows.append({
                    "event": event, "sub_event": sub_event,
                    "hypothesis": row.get("particle", HYP_ELECTRON),
                    "point": point, "term": "total", "value": float(row["nll_fit"]),
                })
    if not rows:
        return empty("nll_terms")
    return conform(pd.DataFrame(rows), "nll_terms")


def read_fit_stages(path) -> pd.DataFrame:
    """Coral's per-stage fit results, from `[in_out] dump_fit_stages`.

    Returned in the `fits` schema with a real `stage` index, so the stage
    trajectory metric treats Coral and fiTQun identically. `fit_vars` (which
    parameters that stage moved) is dropped: it is per-stage metadata, not a
    per-fit quantity, and the canonical schema has no column for it.
    """
    csv_path = _find_job_csv(path, "fit_stages.csv")
    if csv_path is None:
        return empty("fits")
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return empty("fits")
    if df.empty:
        return empty("fits")
    theta = df["theta"].to_numpy(float)
    phi = df["phi"].to_numpy(float)
    return conform(pd.DataFrame({
        "event": df["event"], "sub_event": df["sub_event"],
        "hypothesis": df["particle"].astype(str),
        "ring": RING_LEADING, "nrings": 1, "stage": df["stage"],
        "success": df["success"].astype(bool),
        "nll": df["nll"], "totmu": np.nan,
        "t": df["t"], "x": df["x"], "y": df["y"], "z": df["z"],
        "dx": np.sin(theta) * np.cos(phi), "dy": np.sin(theta) * np.sin(phi),
        "dz": np.cos(theta), "p": df["p"], "dconv": np.nan,
    }), "fits")


def read_event_totals(path) -> pd.DataFrame:
    """Predicted and observed charge per event, from the nll_terms dump.

    The ratio is the charge model's overall scale -- the quantity that
    distinguishes a momentum bias caused by a mis-scaled charge model from one
    caused by the fit.
    """
    csv_path = _find_job_csv(path, "nll_terms.csv")
    columns = ["event", "sub_event", "totq", "nhits", "nll_0r", "totmu_0r"]
    if csv_path is None:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=columns)
    if "totq" not in df.columns:
        return pd.DataFrame(columns=columns)          # dump predates the totals
    out = (df[["event", "sub_event", "totq"]]
           .drop_duplicates(["event", "sub_event"]).copy())
    out["nhits"] = np.nan
    # Coral has no zero-ring hypothesis to compare against.
    out["nll_0r"] = np.nan
    out["totmu_0r"] = np.nan
    return out.reset_index(drop=True)


def _find_job_csv(path, name: str):
    path = Path(path)
    if path.is_file() and path.suffix == ".csv":
        return path
    if path.is_dir():
        direct = path / name
        if direct.exists():
            return direct
        found = sorted(path.rglob(name))
        return found[0] if found else None
    # A merged .root was handed over. merge_coral.py writes the concatenated
    # per-term file as "<merged stem>_nll_terms.csv" beside it, so that two
    # merges in one Reco directory cannot overwrite each other's.
    candidates = [path.parent / f"{path.stem}_{name}", path.parent / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_all(path, *, only_success: bool = False) -> dict:
    """Every canonical table this Coral output can supply."""
    fits = read_fits(path, only_success=only_success)
    return {
        "fits": _attach_totmu(fits, path),
        "prefits": read_prefits(path),
        "multiring": read_multiring(path),
        "pi0": read_pi0(path),
        "nll_terms": read_nll_terms(path),
        "stages": read_fit_stages(path),
        "event_totals": read_event_totals(path),
    }


def _attach_totmu(fits: pd.DataFrame, path) -> pd.DataFrame:
    """Fold the dump's per-hypothesis predicted charge into the fits table.

    Coral's ROOT output has no totmu column; the value comes from the
    nll_terms dump, which is keyed on the same (event, sub_event, hypothesis).
    """
    csv_path = _find_job_csv(path, "nll_terms.csv")
    if csv_path is None or fits.empty:
        return fits
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return fits
    if "totmu" not in df.columns:
        return fits
    totals = (df[["event", "sub_event", "particle", "totmu"]]
              .rename(columns={"particle": "hypothesis"})
              .drop_duplicates(["event", "sub_event", "hypothesis"]))
    merged = fits.drop(columns=["totmu"]).merge(
        totals, on=["event", "sub_event", "hypothesis"], how="left")
    return conform(merged, "fits")
