"""Coral's internals, evaluated at chosen parameter points.

Runs Coral's own charge model and likelihood on real events, at parameter points
Sextant supplies, and writes the canonical `pmt_predictions` and `nll_terms`
tables. Nothing is re-implemented: it calls `compute_predicted_photons` and
`nll_term_sums` from the installed Coral, so what it reports is what Coral's
fitter would see, and it moves automatically when Coral changes.

Needs an importable Coral (`fitqun_fit`, `fitqun_io`, `fitqun`) and the tuning
files its config points at, so it runs where Coral runs -- on the cluster, or on
a laptop with the local tuning paths. Importing this module does not require
Coral; only calling `probe_events` does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.schema import conform, empty


def _require_coral(coral_root=None):
    import sys

    if coral_root:
        source = str(Path(coral_root) / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        from fitqun.FQManager import FQManager                      # noqa: F401
        from fitqun_fit.predicted_charges import compute_predicted_photons  # noqa: F401
    except ImportError as exc:                              # pragma: no cover
        raise RuntimeError(
            "Coral is not importable. Pass --coral-root pointing at a Coral "
            "checkout (the directory containing src/), or run where Coral is "
            "installed. Reading and plotting Coral output needs none of this — "
            "only probing its internals does."
        ) from exc


def probe_events(config_path, input_files, points: pd.DataFrame, *,
                 coral_root=None, outdir=None, max_events=None,
                 store_unhit: bool = False, production=None, sample=None,
                 stride: int = 100) -> dict:
    """Per-PMT predictions and NLL terms at each supplied parameter point.

    `store_unhit` keeps the unhit PMTs in the per-PMT table. Off by default:
    ~50k PMTs per event against a few thousand hits, and the unhit ones carry
    only the predicted charge (there is no observed time to compare against),
    so the sum over them -- which is what the likelihood actually uses -- is
    reported in `nll_terms` instead.

    Coral's event loop yields the event's index *within its input file*, while
    `points` is keyed on the global event id. `production`/`sample` (or a bare
    `stride`) are what reconcile the two: the current file is read back off the
    manager and its index folded in. Without this the probe silently matches
    nothing and reports zero events.
    """
    _require_coral(coral_root)
    from fitqun.FQManager import FQManager
    from fitqun_fit.likelihood import nll_term_sums
    from fitqun_fit.predicted_charges import compute_predicted_photons

    input_files = [str(p) for p in input_files]
    outdir = Path(outdir or Path.cwd())
    # FQManager writes into outdir and expects it to exist already.
    outdir.mkdir(parents=True, exist_ok=True)
    manager = FQManager(str(config_path), input_files, str(outdir))

    pmts = manager.pmts
    pmt_pos = np.ascontiguousarray(pmts.loc[:, ("x", "y", "z")].values, dtype=np.float64)
    pmt_orient = np.ascontiguousarray(pmts.loc[:, ("ox", "oy", "oz")].values,
                                      dtype=np.float64)
    pmt_loc = np.ascontiguousarray(pmts["Loc"].values, dtype=np.int64)
    tubeids = pmts["tubeid"].to_numpy(dtype=np.int64)

    wanted = set(points["event"].astype(int))
    by_event = {int(e): table for e, table in points.groupby("event")}
    stride, index_of = _numbering(production, sample, stride)

    pmt_rows, term_rows = [], []
    seen = 0
    for next_event in manager.event_loop_iterator():
        if next_event is None:
            continue
        local, hits = next_event
        event = index_of(getattr(manager, "infile", "")) * stride + int(local)
        if event not in wanted or len(hits) == 0:
            continue
        seen += 1

        hit_tubeids = np.unique(hits.tubeid.to_numpy(dtype=int))
        is_hit = np.isin(tubeids, hit_tubeids)
        q_obs = np.zeros(len(tubeids), float)
        t_obs = np.full(len(tubeids), np.nan)
        order = {tid: i for i, tid in enumerate(tubeids)}
        for tid, q, t in zip(hits.tubeid.to_numpy(dtype=int),
                             hits.q.to_numpy(float), hits.t.to_numpy(float)):
            index = order.get(int(tid))
            if index is not None:
                q_obs[index] += q
                # Earliest hit: the time PDF is defined against the first
                # photon, and a PMT with several hits would otherwise report
                # whichever happened to come last in the file.
                if np.isnan(t_obs[index]) or t < t_obs[index]:
                    t_obs[index] = t

        for _, point in by_event[event].iterrows():
            hypothesis = str(point["hypothesis"])
            if hypothesis not in manager.cprofs:
                continue
            X = np.array([point["t"], point["x"], point["y"], point["z"],
                          point["theta"], point["phi"], point["p"]], float)

            n_direct, n_indirect = compute_predicted_photons(
                X, pmt_pos, pmt_orient, pmt_loc, manager.ang,
                manager.cprofs[hypothesis], manager.scat,
                manager.attenuation_length, None)
            qe = _qe_for(manager, hypothesis)
            mu_direct = qe * np.asarray(n_direct, float)
            mu_indirect = qe * np.asarray(n_indirect, float)

            keep = np.ones(len(tubeids), bool) if store_unhit else is_hit
            t_resid = _time_residual(manager, X, pmt_pos, t_obs)

            pmt_rows.append(pd.DataFrame({
                "event": event, "sub_event": int(point["sub_event"]),
                "hypothesis": hypothesis, "point": str(point["point"]),
                "pmt": tubeids[keep],
                "mu_direct": mu_direct[keep], "mu_indirect": mu_indirect[keep],
                "q_obs": q_obs[keep], "t_obs": t_obs[keep],
                "t_resid": t_resid[keep],
                "cos_theta": _cos_to_axis(X, pmt_pos)[keep],
                # Straight-line vertex-to-PMT distance. Attenuation makes the
                # predicted charge fall with it, so a drift of observed/predicted
                # against this is the signature of a wrong attenuation length --
                # which Coral's own tuning sets globally (9500 cm) to stand in
                # for scattered light it does not add to the charge.
                "distance": _distance(X, pmt_pos)[keep],
                # How obliquely the light arrives, and which surface the PMT is
                # on. The predicted charge folds in the PMT's angular acceptance,
                # so a drift of observed/predicted against incidence -- or a step
                # between barrel and endcaps -- points at the angular response
                # rather than at attenuation, which is flat in this sample.
                "cos_incidence": _cos_incidence(X, pmt_pos, pmt_orient)[keep],
                "pmt_loc": np.asarray(pmt_loc)[keep],
                "is_hit": is_hit[keep],
            }))

            try:
                unhit, phit, cpdf, tpdf = nll_term_sums(
                    pmts, hits, manager.ang, manager.cprofs[hypothesis],
                    manager.scat, manager.cpdf, manager.tpdfs[hypothesis],
                    manager.attenuation_length, manager.refraction_index, qe, X)
                for name, value in (("unhit", unhit), ("phit", phit),
                                    ("cpdf", cpdf), ("tpdf", tpdf),
                                    ("total", unhit + phit + cpdf + tpdf)):
                    term_rows.append({
                        "event": event, "sub_event": int(point["sub_event"]),
                        "hypothesis": hypothesis, "point": str(point["point"]),
                        "term": name, "value": float(value)})
            except Exception as exc:                        # pragma: no cover
                print(f"[warn] event {event} {hypothesis}: NLL terms failed "
                      f"({exc})")

            # Total predicted light, the event-level charge-normalisation number.
            term_rows.append({
                "event": event, "sub_event": int(point["sub_event"]),
                "hypothesis": hypothesis, "point": str(point["point"]),
                "term": "totmu", "value": float(mu_direct.sum()
                                                + mu_indirect.sum())})
            term_rows.append({
                "event": event, "sub_event": int(point["sub_event"]),
                "hypothesis": hypothesis, "point": str(point["point"]),
                "term": "totq", "value": float(q_obs.sum())})

        if max_events is not None and seen >= max_events:
            break

    predictions = (conform(pd.concat(pmt_rows, ignore_index=True),
                           "pmt_predictions") if pmt_rows
                   else empty("pmt_predictions"))
    terms = (conform(pd.DataFrame(term_rows), "nll_terms") if term_rows
             else empty("nll_terms"))
    return {"pmt_predictions": predictions, "nll_terms": terms,
            "n_events": seen}


def _numbering(production, sample, stride):
    """(stride, filename -> file index) for turning local ids into global ones."""
    import re

    if production is not None:
        from .. import productions as registry

        prod = registry.get(production)
        stride = prod.stride
        if sample is not None:
            spec = prod.sample(sample)
            return stride, lambda name: spec.file_index(name) if name else 0

    # Fallback for a hand-assembled probe: both naming schemes put the index
    # immediately before "_rwcs".
    pattern = re.compile(r"[-_](\d+)_rwcs")

    def index_of(name):
        matches = pattern.findall(str(name))
        return int(matches[-1]) if matches else 0

    return stride, index_of


def _qe_for(manager, hypothesis: str) -> float:
    """Quantum efficiency, including any per-hypothesis correction."""
    qe = float(manager.QE)
    corrections = getattr(manager, "qe_corrections", None)
    if isinstance(corrections, dict):
        qe *= float(corrections.get(hypothesis, 1.0))
    return qe


def _cos_incidence(X, pmt_pos, pmt_orient) -> np.ndarray:
    """cos of the angle between the arriving light and the PMT's own normal."""
    delta = np.asarray(X[1:4], float) - pmt_pos      # PMT -> vertex
    norm = np.sqrt((delta*delta).sum(axis=1))
    norm[norm == 0] = 1.0
    return (delta/norm[:, None] * np.asarray(pmt_orient, float)).sum(axis=1)


def _distance(X, pmt_pos) -> np.ndarray:
    """Vertex-to-PMT distance for every PMT, in cm."""
    delta = pmt_pos - np.asarray(X[1:4], float)
    return np.sqrt((delta*delta).sum(axis=1))


def _cos_to_axis(X, pmt_pos) -> np.ndarray:
    """cos of the angle between the ring axis and the direction to each PMT."""
    theta, phi = float(X[4]), float(X[5])
    axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)])
    delta = pmt_pos - np.asarray(X[1:4], float)
    norm = np.linalg.norm(delta, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (delta @ axis) / norm, np.nan)


def _time_residual(manager, X, pmt_pos, t_obs) -> np.ndarray:
    """Observed hit time minus the time-of-flight prediction from this vertex."""
    from fitqun_fit.likelihood import _compute_pdir, _compute_time_residuals

    try:
        pdir = _compute_pdir(X[4], X[5])
        maxd = manager.cprofs[next(iter(manager.cprofs))].get_maxdistance(X[6])
        return _compute_time_residuals(pmt_pos, X[0], X[1:4], pdir, maxd,
                                       manager.refraction_index, t_obs)
    except Exception:
        # A residual we cannot compute the same way Coral does is better absent
        # than approximated with a different convention.
        return np.full(len(pmt_pos), np.nan)


def write(tables: dict, directory) -> dict:
    """Persist probe output next to the rest of an experiment's results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("pmt_predictions", "nll_terms"):
        table = tables.get(name)
        if table is None or table.empty:
            continue
        path = directory / f"{name}.parquet"
        try:
            table.to_parquet(path, index=False)
        except Exception:
            path = directory / f"{name}.csv"
            table.to_csv(path, index=False)
        written[name] = str(path)
    return written
