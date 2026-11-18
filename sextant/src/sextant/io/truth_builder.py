"""Build canonical truth tables directly from WCSim files.

Reading MC truth needs PyROOT and libWCSimRoot, which exist on the cluster and
in a full local HK install but not in a plain analysis environment. ROOT is
therefore imported lazily: `sextant.io.truth` (reading truth files) works
everywhere, and only *building* them requires the heavier stack.

## What a true ring is

For a particle gun the answer is trivial -- the gun particle. For a beam event
it is a choice, and this is the one made here:

* the **primary final-state particles** are the WCSim tracks with `flag == 0`
  and `parenttype == 0`; these are what NEUT put out of the interaction vertex;
* track `id == 1` is the outgoing lepton (verified across the production: it is
  the numu for NC events and the mu-/e- for CC ones), which is what the
  `is_primary_lepton` flag marks;
* a particle is **visible** if it is charged and above the Cherenkov threshold
  for its mass, or a photon above the pair-production-ring threshold;
* a primary **pi0** makes no ring itself -- it is expanded into its two decay
  photons, taken from the daughter tracks with `parenttype == 111`.

Neutrals below threshold (neutrons, low-energy protons) are kept in the ring
table with `visible = False` rather than dropped, so that "what did the fitter
miss" can be asked of the full final state, while every metric that counts rings
filters on `visible` and never sees them.

## Decay products (true sub-events)

Sub-eventing is scored against `truth_subevents`: the prompt interaction plus
every **decay electron** -- an e+/e- whose parent is a mu+/mu- and which was
created where that muon stopped (within `DECAY_AT_STOP_CM`), i.e. a decay at
rest, the Michel electron. Verified on v1.12.29 muneg files 80-82: WCSim keeps
every track, and the daughters created at the muon's stop point are the Michel
(p up to 53 MeV/c, ~87% of events) plus mu- capture debris -- neutrons,
protons, ~2.6 MeV de-excitation gammas and sub-MeV electrons. `DECAY_E_MIN_MEV`
removes the last of those; the others are not electrons. A pi+ -> mu+ -> e+
chain in a beam event yields its e+ by the same rule.

A sub-event is `recorded` when the readout window (`TRIGGER_WINDOW_NS` around
the date) of a WCSim trigger WITH DIGITS covers its creation time: one that
fired no trigger left no hits, and no reconstruction can be blamed for missing
it. The prompt needs the flag too. An event whose prompt fired nothing carries a
digit-less trigger 0 dated 0 (a 1-5 MeV/c or wall-exiting electron; ~1% of the
electron gun), and a dim muon's trigger 0 can be its decay electron.

## Interaction mode

`GetMode()` is the NEUT mode: |mode| < 30 is charged-current, >= 30 is neutral-
current, and the sign distinguishes neutrino from antineutrino.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import dwall as _dwall, towall as _towall
from .schema import (CHARGED_PDGS, PDG_TO_RING_TYPE, cherenkov_threshold_mev,
                     conform, empty, unit_vector)

#: NEUT modes below this are charged-current.
NC_MODE_THRESHOLD = 30

#: Momentum above which a photon's conversion makes a usable e-like ring.
PHOTON_RING_THRESHOLD_MEV = 20.0

#: A decay electron is created where its parent muon stopped; allow this much
#: for WCSim's step granularity.
DECAY_AT_STOP_CM = 5.0

#: Below this, an electron at the muon's stop point is capture debris (Auger /
#: binding electrons), not a decay electron -- the Michel spectrum is 0-53 MeV/c
#: but essentially nothing of it lies below 1 MeV/c.
DECAY_E_MIN_MEV = 1.0

#: WCSim digit times span (date - 400, date + 950) ns around each trigger.
TRIGGER_WINDOW_NS = (-400.0, 950.0)


def _load_root():
    try:
        import ROOT
    except ImportError as exc:                     # pragma: no cover
        raise RuntimeError(
            "Building truth from WCSim needs PyROOT. Run this on the cluster "
            "(source env_coral.zsh) or in an environment with ROOT and "
            "libWCSimRoot; reading an existing truth file needs neither."
        ) from exc
    if not getattr(_load_root, "_loaded", False):
        # gSystem.Load returns <0 on failure and only *prints* the reason, so an
        # unchecked call leaves you with a ROOT module that cannot see a single
        # WCSim class and an error much further downstream.
        if ROOT.gSystem.Load("libWCSimRoot") < 0:
            raise RuntimeError(
                "libWCSimRoot could not be loaded. On the cluster, source "
                "env_coral.zsh first; locally, point LD_LIBRARY_PATH (or "
                "DYLD_LIBRARY_PATH) at a WCSimRoot build.")
        _load_root._loaded = True
    return ROOT


def build_from_files(paths, *, production, sample, limit_events=None,
                     refraction_index: float = 1.33, progress=None) -> tuple:
    """(truth_rings, truth_events, truth_subevents) from a list of WCSim files."""
    from .. import productions as registry

    prod = registry.get(production)
    spec = prod.sample(sample)

    frames = ([], [], [])
    total = 0
    for path in paths:
        file_index = spec.file_index(path)
        tables = build_from_file(
            path, file_index=file_index, stride=prod.stride,
            kind=spec.kind, refraction_index=refraction_index)
        for frame, table in zip(frames, tables):
            frame.append(table)
        total += len(tables[1])
        if progress is not None:
            progress(path, len(tables[1]), total)
        if limit_events is not None and total >= limit_events:
            break

    names = ("truth_rings", "truth_events", "truth_subevents")
    if not frames[1]:
        return tuple(empty(name) for name in names)
    rings, events, subevents = (pd.concat(f, ignore_index=True) for f in frames)
    if limit_events is not None:
        keep = set(events["event"].to_numpy()[:limit_events])
        rings, events, subevents = (t[t["event"].isin(keep)]
                                    for t in (rings, events, subevents))
    return tuple(conform(t, name).reset_index(drop=True)
                 for t, name in zip((rings, events, subevents), names))


def build_from_file(path, *, file_index: int, stride: int, kind: str = "beam",
                    refraction_index: float = 1.33) -> tuple:
    ROOT = _load_root()

    fh = ROOT.TFile.Open(str(path))
    if not fh or fh.IsZombie():
        raise IOError(f"cannot open {path}")
    try:
        tree = fh.Get("wcsimT")
        if not tree:
            raise IOError(f"{path}: no wcsimT tree")
        event = ROOT.WCSimRootEvent()
        tree.SetBranchAddress("wcsimrootevent", event)

        ring_rows, event_rows, sub_rows = [], [], []
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            event_id = file_index * stride + entry
            rings, header, subevents = _one_event(event, event_id, kind,
                                                  refraction_index)
            ring_rows.extend(rings)
            event_rows.append(header)
            sub_rows.extend(subevents)
    finally:
        fh.Close()

    rings = pd.DataFrame(ring_rows) if ring_rows else empty("truth_rings")
    events = pd.DataFrame(event_rows) if event_rows else empty("truth_events")
    subevents = pd.DataFrame(sub_rows) if sub_rows else empty("truth_subevents")
    return (conform(rings, "truth_rings"), conform(events, "truth_events"),
            conform(subevents, "truth_subevents"))


def _one_event(event, event_id: int, kind: str, refraction_index: float):
    trigger = event.GetTrigger(0)
    n_sub_events = event.GetNumberOfEvents()

    vtx = [trigger.GetVtx(i) for i in range(3)]
    mode = int(trigger.GetMode())

    nhits, totq = _hit_totals(event)

    tracks = trigger.GetTracks()
    primaries, neutrino, daughters = [], None, []
    for i in range(tracks.GetEntries()):
        track = tracks.At(i)
        flag = int(track.GetFlag())
        if flag == -1 and neutrino is None:
            neutrino = track
        elif flag == 0:
            if int(track.GetParenttype()) == 0:
                primaries.append(track)
            elif int(track.GetParenttype()) == 111:
                daughters.append(track)

    # Photons from pi0 decay are allocated once per event, in descending
    # momentum, two to each primary pi0 -- see `_expand_pi0s`.
    photon_pool = sorted((t for t in daughters), key=lambda t: -float(t.GetP()))
    ring_rows = []
    for track in primaries:
        ring_rows.extend(
            _rings_for_track(track, photon_pool, refraction_index))

    # Rings are numbered by descending momentum so that ring 0 is always the
    # brightest one, matching how reconstructed rings are ordered.
    ring_rows.sort(key=lambda r: -r["p"])
    # onto the trigger clock the reconstruction reports (see truth._read_legacy_tracks)
    trigger_time = float(trigger.GetHeader().GetDate())
    for index, row in enumerate(ring_rows):
        row["event"] = event_id
        row["ring"] = index
        row["t"] -= trigger_time

    visible = [r for r in ring_rows if r["visible"]]
    # towall is measured along the ring that actually develops in the detector.
    # With no visible ring there is nothing to measure it along -- an NC elastic
    # event would otherwise report the to-wall distance of its outgoing
    # neutrino, a geometrically valid number describing a particle that leaves
    # no light at all.
    lead = visible[0] if visible else None

    header = {
        "event": event_id, "nhits": nhits, "totq": totq,
        "nhits_trigger0": int(trigger.GetNcherenkovdigihits()),
        "vtx_t": _interaction_time(ring_rows),
        "vtx_x": vtx[0], "vtx_y": vtx[1], "vtx_z": vtx[2],
        "nu_pdg": int(neutrino.GetIpnu()) if neutrino is not None else 0,
        "nu_energy": float(neutrino.GetE()) if neutrino is not None else np.nan,
        "mode": mode,
        "is_cc": abs(mode) < NC_MODE_THRESHOLD and neutrino is not None,
        "n_true_rings": len(ring_rows),
        "n_visible_rings": len(visible),
        "n_sub_events": int(n_sub_events),
        "trigger_time": trigger_time,
        "dwall": float(_dwall(vtx[0], vtx[1], vtx[2])),
        "towall": (float(_towall(vtx[0], vtx[1], vtx[2],
                                 lead["dx"], lead["dy"], lead["dz"]))
                   if lead else np.nan),
    }
    triggers = [event.GetTrigger(i) for i in range(n_sub_events)]
    dates = [float(t.GetHeader().GetDate()) for t in triggers]
    # Only a trigger that recorded digits can have recorded a sub-event.
    live = [date for date, t in zip(dates, triggers)
            if int(t.GetNcherenkovdigihits()) > 0]
    subevents = _subevent_rows(event, event_id, header, lead, dates, live,
                               refraction_index)
    return ring_rows, header, subevents


def _recorded(t_gen, live_dates) -> bool:
    return any(date + TRIGGER_WINDOW_NS[0] <= t_gen <= date + TRIGGER_WINDOW_NS[1]
               for date in live_dates)


def _subevent_rows(event, event_id, header, lead, dates, live_dates,
                   refraction_index):
    """The prompt interaction, then every decay electron in time order.

    Times go onto the trigger clock the same way as the rings' (minus trigger
    0's date). `recorded` is judged on the generator clock, where the trigger
    dates live.
    """
    trigger_time = dates[0] if dates else 0.0
    prompt = {
        "event": event_id, "sub_event": 0, "is_prompt": True,
        "pdg": lead["pdg"] if lead else 0, "parent_pdg": 0,
        "t": header["vtx_t"], "x": header["vtx_x"], "y": header["vtx_y"],
        "z": header["vtx_z"],
        "dx": lead["dx"] if lead else np.nan, "dy": lead["dy"] if lead else np.nan,
        "dz": lead["dz"] if lead else np.nan, "p": lead["p"] if lead else np.nan,
        "recorded": _recorded(header["vtx_t"] + trigger_time, live_dates),
        "visible": bool(lead),
    }
    decays = []
    for track in _decay_electrons(event):
        t_gen = float(track.GetTime())
        momentum = float(track.GetP())
        dx, dy, dz = unit_vector(*[np.array([track.GetDir(i)]) for i in range(3)])
        decays.append({
            "event": event_id, "is_prompt": False,
            "pdg": int(track.GetIpnu()), "parent_pdg": int(track.GetParenttype()),
            "t": t_gen - trigger_time,
            "x": float(track.GetStart(0)), "y": float(track.GetStart(1)),
            "z": float(track.GetStart(2)),
            "dx": float(dx[0]), "dy": float(dy[0]), "dz": float(dz[0]),
            "p": momentum,
            "recorded": _recorded(t_gen, live_dates),
            "visible": momentum > cherenkov_threshold_mev(int(track.GetIpnu()),
                                                          refraction_index),
        })
    decays.sort(key=lambda row: row["t"])
    for index, row in enumerate(decays, start=1):
        row["sub_event"] = index
    return [prompt] + decays


def _decay_electrons(event):
    """e+/e- tracks created where their parent muon stopped (see module doc).

    Every trigger's track list is scanned, de-duplicated on track id: this
    production stores them all in trigger 0, but nothing guarantees it.
    """
    tracks = {}
    for i in range(event.GetNumberOfEvents()):
        array = event.GetTrigger(i).GetTracks()
        for j in range(array.GetEntries()):
            track = array.At(j)
            tracks.setdefault(int(track.GetId()), track)
    out = []
    for track in tracks.values():
        if abs(int(track.GetIpnu())) != 11 or abs(int(track.GetParenttype())) != 13:
            continue
        if float(track.GetP()) < DECAY_E_MIN_MEV:
            continue
        parent = tracks.get(int(track.GetParentId()))
        if parent is None:
            continue
        gap = np.sqrt(sum((float(track.GetStart(i)) - float(parent.GetStop(i))) ** 2
                          for i in range(3)))
        if gap <= DECAY_AT_STOP_CM:
            out.append(track)
    return out


def _interaction_time(ring_rows):
    """Interaction time = creation time of the primary final-state particles.

    All primaries leave the vertex at the same instant, so any of them will do;
    the primary lepton is preferred purely for determinism. This is the same
    quantity a particle-gun truth file reports as the track time, which keeps a
    t residual meaning the same thing across both sample kinds.
    """
    if not ring_rows:
        return 0.0
    for row in ring_rows:
        if row["is_primary_lepton"]:
            return float(row["t"])
    return float(ring_rows[0]["t"])


def _rings_for_track(track, photon_pool, refraction_index):
    pdg = int(track.GetIpnu())
    momentum = float(track.GetP())
    is_lepton = int(track.GetId()) == 1

    if pdg == 111:
        return _pi0_rings(track, photon_pool, is_lepton)

    threshold = cherenkov_threshold_mev(pdg, refraction_index)
    charged = abs(pdg) in CHARGED_PDGS
    if pdg == 22:
        visible = momentum > PHOTON_RING_THRESHOLD_MEV
        above = visible
    else:
        above = momentum > threshold
        visible = bool(charged and above)

    return [_ring_row(track, pdg, momentum, is_lepton, above, visible)]


def _pi0_rings(track, photon_pool, is_lepton):
    """A primary pi0 makes two e-like rings, not one.

    The decay photons are the daughter tracks with `parenttype == 111`. With
    more than one primary pi0 in the event they cannot be attributed to a
    specific parent (WCSim records the parent's PDG, not its track id, and both
    pi0s decay at effectively the same point), so photons are taken two at a
    time in descending momentum -- exact for the common 0- or 1-pi0 event, an
    approximation for the rest.

    `photon_pool` is consumed in place, so each photon is claimed by exactly one
    pi0 within the event and the pool starts fresh for the next event. An
    earlier version kept the cursor on the function object, which leaked across
    events and silently starved every pi0 after the first file.
    """
    chosen = [photon_pool.pop(0) for _ in range(min(2, len(photon_pool)))]

    if len(chosen) < 2:
        # Photons not stored: fall back to the pi0 itself as one e-like ring,
        # flagged not-visible so it is never counted as a findable ring.
        momentum = float(track.GetP())
        return [_ring_row(track, 111, momentum, is_lepton,
                          above=False, visible=False)]
    return [_ring_row(photon, 22, float(photon.GetP()), False,
                      above=float(photon.GetP()) > PHOTON_RING_THRESHOLD_MEV,
                      visible=float(photon.GetP()) > PHOTON_RING_THRESHOLD_MEV)
            for photon in chosen]


def _ring_row(track, pdg, momentum, is_lepton, above, visible):
    direction = [track.GetDir(i) for i in range(3)]
    start = [track.GetStart(i) for i in range(3)]
    dx, dy, dz = unit_vector(*[np.array([v]) for v in direction])
    return {
        "event": -1, "ring": -1, "pdg": pdg,
        "particle": PDG_TO_RING_TYPE.get(abs(pdg), "?" if pdg != 111 else "pi0"),
        "t": float(track.GetTime()),
        "x": start[0], "y": start[1], "z": start[2],
        "dx": float(dx[0]), "dy": float(dy[0]), "dz": float(dz[0]),
        "p": momentum, "is_primary_lepton": bool(is_lepton),
        "above_threshold": bool(above), "visible": bool(visible),
    }


def _hit_totals(event):
    """True hit-PMT count and total charge, summed over every trigger.

    Summed rather than taken from trigger 0 because a muon-decay event puts its
    Michel electron in a second trigger, and an event-level 'how much light was
    there' should include it.
    """
    nhits, totq = 0, 0.0
    for i in range(event.GetNumberOfEvents()):
        trigger = event.GetTrigger(i)
        n = trigger.GetNcherenkovdigihits()
        nhits += int(n)
        try:
            hits = trigger.GetCherenkovDigiHits()
            for j in range(int(n)):
                totq += float(hits.At(j).GetQ())
        except Exception:
            totq = np.nan
    return nhits, totq

