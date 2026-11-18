"""Read MC truth into the canonical `truth_rings` / `truth_events` tables.

Three on-disk formats are understood, so existing truth files keep working
alongside newly built ones:

`sextant`
    What `sextant.io.truth_builder` writes: a `rings` tree (one row per true
    ring) and an `events` tree (topology, vertex, totals). The only format that
    can describe a neutrino interaction.

`legacy-tracks`
    The particle-gun truth files already in `Output/` -- a `Tracks` tree whose
    primary track is `Parenttype == 0 & Id == 1`. One ring per event.

`legacy-photon-pair`
    The pi0 truth files -- a `Tracks` tree with flat `g1_*`/`g2_*` branches for
    the two decay photons plus the true mass and opening angle.

Reading is pure uproot, so this module works anywhere. *Building* truth from
WCSim needs PyROOT and libWCSimRoot and lives in `truth_builder`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import _root
from .geometry import dwall as _dwall, towall as _towall
from .schema import (HYP_ELECTRON, PDG_TO_RING_TYPE, cherenkov_threshold_mev,
                     conform, empty, unit_vector)

FORMAT_SEXTANT = "sextant"
FORMAT_LEGACY_TRACKS = "legacy-tracks"
FORMAT_LEGACY_PHOTON_PAIR = "legacy-photon-pair"


def detect_format(path) -> str:
    names = _root.tree_names(path)
    if "rings" in names and "events" in names:
        return FORMAT_SEXTANT
    if "Tracks" in names:
        return (FORMAT_LEGACY_PHOTON_PAIR
                if "g1_P" in _root.branches(path, "Tracks")
                else FORMAT_LEGACY_TRACKS)
    raise ValueError(
        f"{Path(path).name}: not a recognised truth file. Expected a "
        "'rings'+'events' pair (sextant), or a 'Tracks' tree (legacy).")


def read(path, *, refraction_index: float = 1.33) -> tuple:
    """(truth_rings, truth_events) from any supported truth file."""
    fmt = detect_format(path)
    if fmt == FORMAT_SEXTANT:
        return _read_sextant(path)
    if fmt == FORMAT_LEGACY_PHOTON_PAIR:
        return _read_photon_pair(path, refraction_index)
    return _read_legacy_tracks(path, refraction_index)


def subevents_sidecar(path) -> Path:
    """Where a truth file's sub-event table lives when it is not inside it.

    The legacy particle-gun truth files predate sub-event truth and are read by
    other tools, so they are not rewritten: `sextant truth --subevents-only`
    writes `<stem>_subevents.root` beside them instead.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}_subevents{path.suffix}")


def read_subevents(path) -> pd.DataFrame:
    """The `truth_subevents` table: from the file itself, else from its sidecar.

    Empty when neither has one -- sub-event truth is optional, and a report
    without it simply has no sub-eventing section.
    """
    for candidate in (Path(path), subevents_sidecar(path)):
        if candidate.exists() and "subevents" in _root.tree_names(candidate):
            return conform(_root.read_tree(candidate, "subevents"),
                           "truth_subevents")
    return empty("truth_subevents")


def read_many(paths, **kwargs) -> tuple:
    rings, events = [], []
    for path in paths:
        r, e = read(path, **kwargs)
        rings.append(r)
        events.append(e)
    if not rings:
        return empty("truth_rings"), empty("truth_events")
    return (pd.concat(rings, ignore_index=True).drop_duplicates(["event", "ring"]),
            pd.concat(events, ignore_index=True).drop_duplicates("event"))


# ---------------------------------------------------------------------------

def _read_sextant(path) -> tuple:
    raw_rings = _root.read_tree(path, "rings")
    rings = conform(raw_rings, "truth_rings")
    events = conform(_root.read_tree(path, "events"), "truth_events")
    if "particle" in raw_rings.columns:
        rings["particle"] = _decode_strings(raw_rings["particle"])
    else:
        # Reconstructed from pdg -- see `write`.
        rings["particle"] = [PDG_TO_RING_TYPE.get(abs(int(p)), "?")
                             for p in rings["pdg"]]
    return rings, events


def _decode_strings(col) -> np.ndarray:
    """uproot gives std::string branches back as bytes; normalise to str."""
    return np.array([v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
                     for v in np.asarray(col, dtype=object)], dtype=object)


def _read_legacy_tracks(path, refraction_index) -> tuple:
    tracks = _root.read_tree(path, "Tracks")
    if "Parenttype" in tracks.columns and "Id" in tracks.columns:
        tracks = tracks[(tracks["Parenttype"] == 0) & (tracks["Id"] == 1)]
    sort_cols = [c for c in ("EvtNum", "SubEvtNumber") if c in tracks.columns]
    if sort_cols:
        tracks = tracks.sort_values(sort_cols)
    tracks = tracks.drop_duplicates(subset=["EvtNum"], keep="first")
    if tracks.empty:
        return empty("truth_rings"), empty("truth_events")

    dx, dy, dz = unit_vector(tracks["Dir_x0"], tracks["Dir_x1"], tracks["Dir_x2"])
    pdg = (tracks["Ipnu"].astype(int).to_numpy() if "Ipnu" in tracks.columns
           else np.full(len(tracks), 11))
    momentum = tracks["P"].astype(float).to_numpy()
    threshold = np.array([cherenkov_threshold_mev(c, refraction_index) for c in pdg])
    above = momentum > threshold

    # Reconstruction reports time on the TRIGGER clock: WCSim stores a digit
    # at (photon time - trigger-0 date + offset), and both codes remove the
    # offset. The generator clock puts the gun at Time = 0, so without the
    # trigger date a time residual is dominated by the 5 ns-quantised,
    # vertex-dependent trigger position (tens of ns), not by the fit.
    trigger_time = (tracks["TriggerTime"].astype(float).to_numpy()
                    if "TriggerTime" in tracks.columns
                    else np.full(len(tracks), np.nan))
    time = tracks["Time"].astype(float).to_numpy()
    rings = pd.DataFrame({
        "event": tracks["EvtNum"].astype(int).to_numpy(),
        "ring": 0, "pdg": pdg,
        "particle": [PDG_TO_RING_TYPE.get(int(c), "?") for c in pdg],
        "t": np.where(np.isfinite(trigger_time), time - trigger_time, time),
        "x": tracks["Start_x0"].astype(float).to_numpy(),
        "y": tracks["Start_x1"].astype(float).to_numpy(),
        "z": tracks["Start_x2"].astype(float).to_numpy(),
        "dx": dx, "dy": dy, "dz": dz, "p": momentum,
        "is_primary_lepton": True,
        "above_threshold": above, "visible": above,
    })
    events = _events_from_rings(
        rings,
        nhits=(tracks["nhits"].astype(int).to_numpy()
               if "nhits" in tracks.columns else None),
        trigger_time=trigger_time)
    return conform(rings, "truth_rings"), events


def _read_photon_pair(path, refraction_index) -> tuple:
    tracks = _root.read_tree(path, "Tracks")
    frames = []
    for ring, tag in enumerate(("g1", "g2")):
        dx, dy, dz = unit_vector(tracks[f"{tag}_dx"], tracks[f"{tag}_dy"],
                                 tracks[f"{tag}_dz"])
        momentum = tracks[f"{tag}_P"].astype(float).to_numpy()
        frames.append(pd.DataFrame({
            "event": tracks["EvtNum"].astype(int).to_numpy(),
            "ring": ring, "pdg": 22, "particle": HYP_ELECTRON,
            "t": tracks["vtx_t"].astype(float).to_numpy(),
            "x": tracks["vtx_x"].astype(float).to_numpy(),
            "y": tracks["vtx_y"].astype(float).to_numpy(),
            "z": tracks["vtx_z"].astype(float).to_numpy(),
            "dx": dx, "dy": dy, "dz": dz, "p": momentum,
            "is_primary_lepton": False,
            # A photon has no Cherenkov threshold of its own; it converts to an
            # e+e- pair that does. Below ~20 MeV the pair makes no usable ring,
            # which is exactly the sub-threshold second photon that shows up as
            # a permanently 'missed' ring in pi0 multi-ring metrics.
            "above_threshold": momentum > 20.0,
            "visible": momentum > 20.0,
        }))
    rings = pd.concat(frames, ignore_index=True)
    events = _events_from_rings(
        rings,
        nhits=(tracks["nhits"].astype(int).to_numpy()
               if "nhits" in tracks.columns else None),
        extra={"event": tracks["EvtNum"].astype(int).to_numpy()})
    return conform(rings, "truth_rings"), events


def _events_from_rings(rings: pd.DataFrame, nhits=None, extra=None,
                       trigger_time=None) -> pd.DataFrame:
    """Per-event truth from the ring table, for the legacy formats.

    The vertex is the leading ring's start point, which for a particle gun is
    the gun vertex and for a pi0 is the decay point -- in both cases the thing a
    vertex residual should be measured against.
    """
    lead = (rings.sort_values(["event", "p"], ascending=[True, False])
                 .drop_duplicates("event", keep="first"))
    counts = rings.groupby("event").size()
    visible = rings[rings["visible"]].groupby("event").size()
    events = pd.DataFrame({
        "event": lead["event"].to_numpy(),
        "nhits": -1 if nhits is None else nhits[:len(lead)],
        "totq": np.nan,
        "vtx_t": lead["t"].to_numpy(), "vtx_x": lead["x"].to_numpy(),
        "vtx_y": lead["y"].to_numpy(), "vtx_z": lead["z"].to_numpy(),
        "nu_pdg": 0, "nu_energy": np.nan, "mode": 0, "is_cc": False,
        "n_true_rings": lead["event"].map(counts).to_numpy(),
        "n_visible_rings": lead["event"].map(visible).fillna(0).to_numpy(),
        "n_sub_events": -1,
        "trigger_time": (np.nan if trigger_time is None
                         else pd.Series(trigger_time, index=rings.index)
                         .loc[lead.index].to_numpy()),
    })
    events["dwall"] = _dwall(events["vtx_x"], events["vtx_y"], events["vtx_z"])
    events["towall"] = _towall(events["vtx_x"], events["vtx_y"], events["vtx_z"],
                               lead["dx"].to_numpy(), lead["dy"].to_numpy(),
                               lead["dz"].to_numpy())
    return conform(events, "truth_events")


# ---------------------------------------------------------------------------

def write(path, rings: pd.DataFrame | None, events: pd.DataFrame | None,
          subevents: pd.DataFrame | None = None):
    """Write a `sextant`-format truth file (`rings`, `events`, `subevents`).

    Any table given as None is left out, which is how a sub-event sidecar
    (`subevents` only) is written.
    """
    import uproot

    tables = [(name, conform(table, schema_name))
              for name, table, schema_name in (
                  ("rings", rings, "truth_rings"),
                  ("events", events, "truth_events"),
                  ("subevents", subevents, "truth_subevents"))
              if table is not None]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as fout:
        for name, table in tables:
            columns = {}
            for col in table.columns:
                values = table[col].to_numpy()
                if values.dtype == object:
                    # A classic TTree branch cannot hold a numpy unicode dtype,
                    # and the only string column here (`particle`) is a pure
                    # function of `pdg` -- so it is dropped rather than encoded,
                    # and `read` reconstructs it. Storing derivable data is also
                    # how the two can disagree later.
                    continue
                columns[col] = values
            fout.mktree(name, {k: v.dtype for k, v in columns.items()})
            if len(table):
                fout[name].extend(columns)


def select_primary_rings(rings: pd.DataFrame) -> pd.DataFrame:
    """One row per event: the ring a single-ring fit should be judged against.

    The primary lepton where the truth marks one (beam CC events), otherwise the
    most energetic visible ring. This is what turns a beam sample into something
    the single-ring residual machinery can consume unchanged.
    """
    if rings.empty:
        return rings
    lepton = rings[rings["is_primary_lepton"] & rings["visible"]]
    leading = (rings[rings["visible"]]
               .sort_values(["event", "p"], ascending=[True, False])
               .drop_duplicates("event", keep="first"))
    if lepton.empty:
        return leading.reset_index(drop=True)
    fallback = leading[~leading["event"].isin(lepton["event"])]
    return (pd.concat([lepton, fallback], ignore_index=True)
              .sort_values("event").reset_index(drop=True))
