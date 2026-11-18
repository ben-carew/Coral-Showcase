"""Drive the `fqprobe` binary and read its output into the canonical table.

`fqprobe` (see `cpp/fqprobe/`) is a small standalone program that links fiTQun's
public library and calls `Get1Rmudist` at parameter points it reads from stdin.
This module writes those points, runs it, and converts the result.

Building it is a one-off and does NOT rebuild fiTQun -- it links whatever
fiTQun install is already there:

    cpp/fqprobe/build.sh [FITQUN_SRC] [FITQUN_INSTALL]

and `SEXTANT_FQPROBE` or `--fqprobe` then points at the binary.

`fqprobe` needs FITQUN_ROOT set (it reads the tuning constants and the default
parameter file from there) and the fiTQun/WCSimRoot libraries on the loader
path; `probe_env` below builds that environment so callers do not have to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.schema import (FITQUN_HYPOTHESIS_INDEX, FITQUN_INDEX_TO_HYPOTHESIS,
                         conform, empty)

BINARY_ENV = "SEXTANT_FQPROBE"


def find_binary(explicit=None) -> Path:
    for candidate in (explicit, os.environ.get(BINARY_ENV)):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    found = shutil.which("fqprobe")
    if found:
        return Path(found)
    package_root = Path(__file__).resolve().parents[3]
    for relative in ("cpp/fqprobe/fqprobe", "build/fqprobe/fqprobe",
                     "cpp/fqprobe/build/fqprobe"):
        candidate = package_root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "fqprobe binary not found. Build it with:\n"
        "  Software/Sextant/cpp/fqprobe/build.sh\n"
        "(it links an existing fiTQun install; nothing is rebuilt)\n"
        f"then set {BINARY_ENV} or pass --fqprobe.")


def probe_env(fitqun_root=None) -> dict:
    """Environment fqprobe needs: FITQUN_ROOT plus the fiTQun/WCSimRoot libs.

    Left to the ambient environment this fails in a way that looks like a
    physics result rather than a setup error -- without FITQUN_ROOT the tuning
    constants are not found and fiTQun exits, and with the wrong one it silently
    predicts light from a different detector's tuning.
    """
    root = Path(fitqun_root or os.environ.get("FITQUN_ROOT") or "")
    env = dict(os.environ)
    if root:
        env["FITQUN_ROOT"] = str(root)
        libs = [str(p / "lib") for p in root.glob("install-*")]
        wcsim = list((root.parent / "Dependencies/WCSimRoot").glob("install-*"))
        libs += [str(p / "lib") for p in wcsim]
        if libs:
            key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" \
                else "LD_LIBRARY_PATH"
            env[key] = os.pathsep.join(libs + [p for p in (env.get(key),) if p])
    return env


def write_points(points: pd.DataFrame, path, *, stride: int) -> int:
    """Write the point list in the format fqprobe reads.

    fqprobe addresses events by WCSim *tree entry*, so the global event id is
    split back into (file index, entry) here -- and only points from one input
    file may be written at a time, because a single fqprobe run opens one file.
    Points are ordered by entry so the binary loads each event once.

    `twnd_start`/`twnd_end`, if the point table carries them, are fiTQun's own
    per-event gate (its `fqtwnd`). Passing them matters: fiTQun does not fit on
    its default window but on the hit cluster spliTChan found, so probing
    without them scores the model against a different set of hits than the fit
    used -- typically ~5% fewer -- and against a total charge that is therefore
    not the one fiTQun saw.
    """
    has_window = {"twnd_start", "twnd_end"}.issubset(points.columns)
    rows = []
    for _, point in points.iterrows():
        hypothesis = str(point["hypothesis"])
        pid = FITQUN_HYPOTHESIS_INDEX.get(hypothesis)
        if pid is None:
            continue
        _, entry = divmod(int(point["event"]), stride)
        dconv = float(point.get("dconv", 0.0) or 0.0)
        row = [entry, int(point["sub_event"]), pid,
               float(point["t"]), float(point["x"]), float(point["y"]),
               float(point["z"]), float(point["theta"]),
               float(point["phi"]), float(point["p"]), dconv]
        if has_window:
            row += [float(point["twnd_start"]), float(point["twnd_end"])]
        rows.append(tuple(row))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    header = "# entry sub_event pid t x y z theta phi p dconv"
    if has_window:
        header += " twnd_start twnd_end"
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(" ".join(str(v) for v in row) + "\n")
    return len(rows)


def run(input_file, points: pd.DataFrame, parameter_file, *, stride: int,
        file_index: int, binary=None, workdir=None, fitqun_root=None,
        with_pmts: bool = True, time_offset_ns: float = 0.0) -> dict:
    """Probe one WCSim input file.

    Returns {"pmt_predictions", "nll_terms"} -- the same pair `coral_probe`
    returns, so the two arms go through the same metrics unchanged.
    """
    import uproot

    binary = find_binary(binary)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="sextant-fqprobe-"))
    workdir.mkdir(parents=True, exist_ok=True)
    points_path = workdir / "points.txt"
    output_path = workdir / "predictions.root"

    n = write_points(points, points_path, stride=stride)
    if n == 0:
        return {"pmt_predictions": empty("pmt_predictions"),
                "nll_terms": empty("nll_terms")}

    command = [str(binary), str(input_file), str(output_path),
               str(parameter_file)]
    if not with_pmts:
        command.append("--no-pmt")
    with open(points_path) as stdin:
        result = subprocess.run(command, stdin=stdin, capture_output=True,
                                text=True, env=probe_env(fitqun_root))
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"fqprobe failed (exit {result.returncode}) on {input_file}:\n"
            f"{result.stderr.strip()[-2000:]}")

    point_label = str(points["point"].iloc[0])
    with uproot.open(output_path) as fh:
        totals = fh["points"].arrays(library="pd")
        geometry = fh["geometry"].arrays(library="pd")
        table = (fh["predictions"].arrays(library="pd")
                 if with_pmts and "predictions" in fh else None)

    def _events(frame):
        return file_index * stride + frame["entry"].to_numpy(np.int64)

    # fiTQun does not split its likelihood the way Coral does, so only the two
    # terms both codes can supply -- the total light and the observed charge --
    # plus the total NLL and the charge/time halves fiTQun does separate.
    term_rows = []
    for name, column in (("totmu", "totmu_all"), ("totq", "totq"),
                         ("total", "nll"), ("cpdf", "nll_charge"),
                         ("tpdf", "nll_time")):
        term_rows.append(pd.DataFrame({
            "event": _events(totals), "sub_event": totals["sub_event"],
            "hypothesis": [FITQUN_INDEX_TO_HYPOTHESIS.get(int(p), "?")
                           for p in totals["pid"]],
            "point": point_label, "term": name, "value": totals[column]}))
    terms = conform(pd.concat(term_rows, ignore_index=True), "nll_terms")

    if table is None or table.empty:
        return {"pmt_predictions": empty("pmt_predictions"), "nll_terms": terms}

    is_hit = table["is_hit"].to_numpy().astype(bool)
    # An unhit PMT has no observed time -- fiTQun leaves the slot at 0, and
    # carrying that through would report a ~-1100 ns residual for every unhit
    # tube and drag any residual distribution built from this table.
    t_obs = np.where(is_hit, table["t_obs"].to_numpy(float) - time_offset_ns,
                     np.nan)
    t_resid = np.where(is_hit,
                       table["t_obs"].to_numpy(float)
                       - table["t_pred"].to_numpy(float), np.nan)
    geometry_columns = _geometry_columns(table, totals, geometry)

    out = pd.DataFrame({
        "event": _events(table),
        "sub_event": table["sub_event"],
        "hypothesis": [FITQUN_INDEX_TO_HYPOTHESIS.get(int(p), "?")
                       for p in table["pid"]],
        "point": point_label,
        "pmt": table["pmt"],
        "mu_direct": table["mu_direct"], "mu_indirect": table["mu_indirect"],
        "q_obs": table["q_obs"],
        "t_obs": t_obs,
        # fiTQun predicts a hit time per PMT (its `tau`), so the residual is a
        # real number here rather than the NaN placeholder it used to be.
        "t_resid": t_resid,
        "cos_theta": table["cos_theta"],
        "is_hit": is_hit,
        **geometry_columns,
    })
    result_table = conform(out, "pmt_predictions")
    # Carried alongside rather than in the canonical columns: per-PMT likelihood
    # contributions are a fiTQun-only quantity, and inventing schema columns
    # only one code can fill would make every Coral table two-thirds empty.
    result_table.attrs["per_pmt_likelihood"] = pd.DataFrame({
        "event": out["event"], "sub_event": out["sub_event"],
        "hypothesis": out["hypothesis"], "pmt": out["pmt"],
        "nll_charge": table["nll_charge"], "nll_time": table["nll_time"]})
    return {"pmt_predictions": result_table, "nll_terms": terms}


def _geometry_columns(table, totals, geometry) -> dict:
    """`distance` and `cos_incidence` per row, from the probe's geometry tree.

    Coral's probe fills these and the binned light-model analyses use them --
    predicted-over-observed against distance is the attenuation-length test,
    against incidence the angular-response one. fiTQun does not report either,
    but it reports the PMT positions and the vertex it used, which is all they
    are: leaving them empty would silently restrict every such comparison to
    the Coral arm.
    """
    positions = np.column_stack([geometry["x"].to_numpy(float),
                                 geometry["y"].to_numpy(float),
                                 geometry["z"].to_numpy(float)])
    directions = np.column_stack([geometry["dir_x"].to_numpy(float),
                                  geometry["dir_y"].to_numpy(float),
                                  geometry["dir_z"].to_numpy(float)])
    key = list(zip(totals["entry"].to_numpy(), totals["sub_event"].to_numpy(),
                   totals["pid"].to_numpy()))
    vertices = {k: (x, y, z) for k, x, y, z in zip(
        key, totals["x"].to_numpy(float), totals["y"].to_numpy(float),
        totals["z"].to_numpy(float))}
    rows = list(zip(table["entry"].to_numpy(), table["sub_event"].to_numpy(),
                    table["pid"].to_numpy()))
    vertex = np.array([vertices.get(k, (np.nan,) * 3) for k in rows])
    pmt = positions[table["pmt"].to_numpy(int)]
    normal = directions[table["pmt"].to_numpy(int)]
    delta = pmt - vertex
    distance = np.linalg.norm(delta, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # The PMT normal points into the water, so light arriving head-on gives
        # a *negative* dot product with it; the sign is flipped to make
        # cos_incidence = 1 mean normal incidence, as Coral's probe reports it.
        cos_incidence = -np.sum(delta * normal, axis=1) / distance
    return {"distance": distance, "cos_incidence": cos_incidence,
            "pmt_loc": geometry["pmt_type"].to_numpy()[table["pmt"].to_numpy(int)]}


def probe_files(file_points: dict, parameter_file, *, stride: int, binary=None,
                workdir=None, fitqun_root=None, with_pmts: bool = True,
                time_offset_ns: float = 0.0) -> dict:
    """Probe several input files. `file_points` maps (path, file_index) -> points."""
    results = []
    for (path, file_index), points in file_points.items():
        results.append(run(path, points, parameter_file, stride=stride,
                           file_index=file_index, binary=binary,
                           workdir=(Path(workdir) / f"file_{file_index:04d}"
                                    if workdir else None),
                           fitqun_root=fitqun_root, with_pmts=with_pmts,
                           time_offset_ns=time_offset_ns))
    out = {}
    for key in ("pmt_predictions", "nll_terms"):
        frames = [r[key] for r in results if not r[key].empty]
        out[key] = (pd.concat(frames, ignore_index=True) if frames
                    else empty(key))
    return out


def group_points_by_file(points: pd.DataFrame, production: str, sample: str,
                         *, root=None) -> dict:
    """Split a point table into the input files the events live in."""
    from .. import productions

    prod = productions.get(production)
    spec = prod.sample(sample)
    grouped = {}
    for event, group in points.groupby("event"):
        file_index, _ = divmod(int(event), prod.stride)
        try:
            path = spec.input_file(file_index)
        except ValueError:
            continue
        if root is not None:
            path = Path(root) / Path(path).relative_to(Path(path).anchor)
        grouped.setdefault((str(path), file_index), []).append(group)
    return {key: pd.concat(frames, ignore_index=True)
            for key, frames in grouped.items()}
