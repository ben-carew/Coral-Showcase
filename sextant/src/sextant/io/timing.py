"""Per-event reconstruction timing.

Both merge scripts mirror `Reco/<sub>/x.root` to `Timing/<sub>/x.csv`, one row
per event with `time_s` (the algorithm's own reported reconstruction time) and
`events` (1 for a real per-event row). Older CSVs carry one row per *job*;
dividing by `events` normalises either shape, which is why that column is kept.

Wall time and reconstruction time are different questions and both are reported:
`time_s` is what an algorithm change moves, `wall_s` includes the interpreter,
JIT and tuning-file startup that packing events per job exists to amortise.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


def timing_path_for(reco_path) -> Path:
    """Locate the timing CSV belonging to a reconstruction file."""
    reco_path = Path(reco_path)
    parts = list(reco_path.parts)
    candidates = []
    if "Reco" in parts:
        i = len(parts) - 1 - parts[::-1].index("Reco")
        candidates.append(Path(*parts[:i]) / "Timing"
                          / Path(*parts[i + 1:]).with_suffix(".csv"))
    candidates += [
        reco_path.parent.parent / "Timing" / (reco_path.stem + ".csv"),
        reco_path.parent / "Timing" / (reco_path.stem + ".csv"),
        reco_path.with_suffix(".csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else reco_path.with_suffix(".csv")


def read(path, *, drop_first_event: bool = False) -> pd.DataFrame:
    """One row per event: event id, per-event time, and the job it came from.

    Coral's numba kernels cache their JIT to disk, so the first event of a job
    is *not* a warm-up outlier (measured 2.16 s against a 1.81 s median) and is
    kept by default; `drop_first_event` is there for a cold-cache run.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["job", "event", "time_s", "wall_s", "events"])

    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                n = float(row.get("events", 1) or 1)
            except ValueError:
                n = 1.0
            if n <= 0:
                continue
            event = row.get("event")
            rows.append({
                "job": row.get("job", ""),
                "event": float(event) if event not in (None, "") else np.nan,
                "time_s": _as_float(row.get("time_s")) / n,
                "wall_s": _as_float(row.get("wall_s")) / n,
                "events": n,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if drop_first_event and df["event"].notna().any():
        df = df.sort_values(["job", "event"])
        df = df[df.groupby("job").cumcount() > 0]
    return df.reset_index(drop=True)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def summarise(df: pd.DataFrame, label: str = "") -> dict:
    if df.empty:
        return {"label": label, "n_events": 0}
    t = df["time_s"].to_numpy(float)
    t = t[np.isfinite(t)]
    wall = df["wall_s"].to_numpy(float)
    wall = wall[np.isfinite(wall)]
    # A row is normally one JOB of `events` events, not one event: both merge
    # scripts write per-job rows and `read` divides by `events` to get a rate.
    # So the row count is the number of jobs, and reporting it as `n_events`
    # said 30 for a 3000-event run and 1000 for a 100,000-event one.
    n_events = int(df["events"].sum()) if "events" in df.columns else int(len(t))
    out = {
        "label": label, "n_events": n_events, "n_jobs": int(len(t)),
        "mean_s": float(np.mean(t)) if len(t) else np.nan,
        "median_s": float(np.median(t)) if len(t) else np.nan,
        "p90_s": float(np.percentile(t, 90)) if len(t) else np.nan,
        "min_s": float(np.min(t)) if len(t) else np.nan,
        "max_s": float(np.max(t)) if len(t) else np.nan,
    }
    # Absent when the run's timing CSV predates the `wall_s` column, which is
    # why a reference run can show a reconstruction time and no wall time. Left
    # as NaN rather than filled from time_s: they measure different things.
    out["median_wall_s"] = float(np.median(wall)) if len(wall) else np.nan
    return out


def event_ids_from_job(job: str, n_events: int, stride: int = 100):
    """Recover event ids for an old per-job row from its input filename."""
    m = re.search(r"-(\d+)_rwcs", job or "")
    if not m:
        return np.array([], dtype=np.int64)
    return int(m.group(1)) * stride + np.arange(int(n_events), dtype=np.int64)
