"""Tests for the HK validation-spec additions (roadmap R1, 2026-09-28).

Sub-eventing efficiency/purity, PID efficiency against the binning variables,
the multi-track section and towall profiles. As in test_sextant.py, what is
pinned is the behaviour whose breakage would still produce plausible plots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sextant.io import schema, truth as truth_io
from sextant.io.dataset import Arm, Truth
from sextant.metrics import multitrack, pid, stats, subevents


def _sub(event, sub_event, t, prompt, recorded=True, p=40.0):
    return {"event": event, "sub_event": sub_event, "is_prompt": prompt,
            "pdg": 13 if prompt else 11, "parent_pdg": 0 if prompt else 13,
            "t": t, "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0,
            "p": 500.0 if prompt else p, "recorded": recorded, "visible": True}


def _prefits(rows):
    return schema.conform(pd.DataFrame(
        [{"event": e, "sub_event": s, "hypothesis": "(prefit)", "ring": 0,
          "nrings": 1, "stage": schema.STAGE_FINAL, "success": True, "t": t,
          "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": 1.0}
         for e, s, t in rows]), "fits")


def _truth(sub_rows):
    events = sorted({r["event"] for r in sub_rows})
    return Truth(rings=schema.empty("truth_rings"),
                 events=schema.conform(pd.DataFrame({"event": events}), "truth_events"),
                 subevents=schema.conform(pd.DataFrame(sub_rows), "truth_subevents"))


# ---------------------------------------------------------------------------
# Sub-eventing
# ---------------------------------------------------------------------------

def test_subevents_match_by_time_not_by_index():
    """fiTQun's staged fits come out Michel-first in ~40% of muon events; a
    match on the index would score every one of those as two failures."""
    truth = _truth([_sub(0, 0, 0.0, True), _sub(0, 1, 1000.0, False)])
    arm = Arm("a", "coral", fits=schema.empty("fits"),
              prefits=_prefits([(0, 0, 1003.0), (0, 1, 2.0)]))
    per_true, per_reco = subevents.build([arm], truth)["a"]
    assert per_true["matched"].all()
    rates = {k: v.mean() for k, v in subevents.rates(per_true, per_reco).items()}
    assert rates["prompt purity"] == 1.0 and rates["delayed purity"] == 1.0


def test_subevent_matching_is_one_to_one_and_windowed():
    """Two reco sub-events on one true one: the second is a fake, and a reco
    sub-event outside the window matches nothing."""
    truth = _truth([_sub(0, 0, 0.0, True), _sub(1, 0, 0.0, True)])
    arm = Arm("a", "coral", fits=schema.empty("fits"),
              prefits=_prefits([(0, 0, 1.0), (0, 1, 3.0), (1, 0, 80.0)]))
    per_true, per_reco = subevents.build([arm], truth, window_ns=50.0)["a"]
    assert per_true.set_index("event")["matched"].to_dict() == {0: True, 1: False}
    assert per_reco["matched"].sum() == 1


def test_the_prompt_claims_a_reco_sub_event_before_an_overlapping_decay():
    """A decay 20 ns after the prompt cannot be resolved: the lone reco sub-event,
    even when it sits nearer the decay's time, is the prompt found."""
    truth = _truth([_sub(0, 0, 0.0, True), _sub(0, 1, 20.0, False)])
    arm = Arm("a", "coral", fits=schema.empty("fits"), prefits=_prefits([(0, 0, 14.0)]))
    per_true, _ = subevents.build([arm], truth)["a"]
    assert per_true.set_index("kind")["matched"].to_dict() == {"prompt": True, "decay": False}


def test_an_unrecorded_prompt_removes_the_event_and_an_absent_arm_scores_zero():
    """A prompt that fired no trigger is not a sub-eventing failure. An event one
    code has no output for (a crash, or a job lost to the straggler cut) is
    reported as "events without output"; the rates are paired on the events
    every arm has."""
    truth = _truth([_sub(0, 0, 0.0, True), _sub(1, 0, 0.0, True, recorded=False),
                    _sub(2, 0, 0.0, True)])
    full = Arm("full", "fitqun", fits=schema.empty("fits"),
               prefits=_prefits([(0, 0, 0.0), (1, 0, 0.0), (2, 0, 0.0)]))
    partial = Arm("partial", "coral", fits=schema.empty("fits"),
                  prefits=_prefits([(0, 0, 0.0)]))
    tables = subevents.build([full, partial], truth)
    table = subevents.summary(tables).set_index(["arm", "metric"])
    assert table.loc[("full", "prompt efficiency"), "n"] == 1
    assert table.loc[("partial", "prompt efficiency"), "value"] == 100.0
    assert table.loc[("full", "events without output"), "value"] == 0.0
    assert table.loc[("partial", "events without output"), "value"] == 50.0


def test_unrecorded_decay_electrons_are_not_counted_as_missed():
    truth = _truth([_sub(0, 0, 0.0, True), _sub(0, 1, 900.0, False, recorded=False)])
    arm = Arm("a", "coral", fits=schema.empty("fits"), prefits=_prefits([(0, 0, 0.0)]))
    per_true, _ = subevents.build([arm], truth)["a"]
    assert (per_true["kind"] == subevents.DECAY).sum() == 0


def test_wilson_interval_stays_inside_zero_one_and_contains_the_rate():
    lo, hi = stats.wilson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert (lo >= 0).all() and (hi <= 1 + 1e-12).all()
    assert lo[1] < 0.5 < hi[1]


def test_merged_fitqun_stage_fallback_reindexes_sub_events_in_time_order(tmp_path):
    """Older merges have only the staged fits, whose sub-event index is not in
    time order."""
    import uproot
    from sextant.io.fitqun import read_merged_subevents

    path = tmp_path / "fq.root"
    columns = {"event": np.array([5, 5]), "sub_event": np.array([0, 1], np.int32),
               "stage": np.array([1, 1], np.int32), "pid": np.array([1, 1], np.int32),
               "t": np.array([980.0, -55.0])}
    with uproot.recreate(path) as f:
        # mktree, not f[name] = dict: uproot 5.7 writes the latter as an RNTuple
        f.mktree("stages", {k: v.dtype for k, v in columns.items()})
        f["stages"].extend(columns)
    out = read_merged_subevents(path)
    assert out.sort_values("sub_event")["t"].tolist() == [-55.0, 980.0]


def test_subevent_sidecar_round_trips_and_is_found_beside_the_truth(tmp_path):
    rings = schema.conform(pd.DataFrame([{
        "event": 0, "ring": 0, "pdg": 13, "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0,
        "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": 500.0, "is_primary_lepton": True,
        "above_threshold": True, "visible": True}]), "truth_rings")
    events = schema.conform(pd.DataFrame({"event": [0]}), "truth_events")
    main = tmp_path / "truth_muon.root"
    truth_io.write(main, rings, events)
    sub = pd.DataFrame([_sub(0, 0, 0.0, True), _sub(0, 1, 1200.0, False)])
    truth_io.write(truth_io.subevents_sidecar(main), None, None, sub)
    truth = Truth.from_file(main)
    assert len(truth.subevents) == 2
    assert truth.subevents["is_prompt"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# PID and multi-track
# ---------------------------------------------------------------------------

def test_pid_classification_follows_the_discriminant_sign():
    table = pd.DataFrame({"event": [0, 1], "dnll": [5.0, -5.0]})
    primary = pd.DataFrame({"event": [0, 1], "p": [300.0, 300.0], "x": 0.0,
                            "y": 0.0, "z": 0.0, "dx": 1.0, "dy": 0.0, "dz": 0.0})
    assert pid.classified(table, primary, "e-")["correct"].tolist() == [True, False]
    assert pid.classified(table, primary, "mu-")["correct"].tolist() == [False, True]


def _beam_truth():
    """Two-track events: a 480 MeV/c muon (E = 491 MeV) and a 485 MeV/c
    electron (E = 485 MeV). Ranked by momentum the electron leads; the spec asks
    for the most ENERGETIC particle, which is the muon."""
    rows = []
    for event in range(30):
        rows.append({"event": event, "ring": 0, "pdg": 13, "particle": "mu-",
                     "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0,
                     "dz": 1.0, "p": 480.0, "is_primary_lepton": True,
                     "above_threshold": True, "visible": True})
        rows.append({"event": event, "ring": 1, "pdg": 11, "particle": "e-",
                     "t": 0.0, "x": 50.0, "y": 0.0, "z": 0.0, "dx": 1.0, "dy": 0.0,
                     "dz": 0.0, "p": 485.0, "is_primary_lepton": False,
                     "above_threshold": True, "visible": True})
    events = pd.DataFrame({"event": range(30), "vtx_x": 10.0, "vtx_y": 0.0,
                           "vtx_z": 0.0, "vtx_t": 0.0})
    return Truth(rings=schema.conform(pd.DataFrame(rows), "truth_rings"),
                 events=schema.conform(events, "truth_events"))


def test_multitrack_uses_the_most_energetic_particle_at_the_interaction_vertex():
    truth = _beam_truth()
    lead = multitrack.most_energetic(truth)
    assert (lead["particle"] == "mu-").all()
    assert (lead["x"] == 10.0).all()          # the interaction vertex, not the ring start


def test_multitrack_scores_the_leading_ring_of_the_multiring_fit():
    truth = _beam_truth()
    rows = []
    for event in range(30):
        # Leading ring (highest p) is a mu- along +z; a weaker e- ring too.
        for ring, (hyp, p, d) in enumerate((("mu-", 499.2, (0, 0, 1)),
                                            ("e-", 400.0, (1, 0, 0)))):
            rows.append({"event": event, "sub_event": 0, "hypothesis": hyp,
                         "ring": ring, "nrings": 2, "stage": schema.STAGE_FINAL,
                         "success": True, "nll": 1.0, "t": 0.0, "x": 12.0,
                         "y": 0.0, "z": 0.0, "dx": d[0], "dy": d[1], "dz": d[2],
                         "p": p})
    arm = Arm("mr", "coral", fits=schema.empty("fits"),
              multiring=schema.conform(pd.DataFrame(rows), "fits"))
    table = multitrack.build([arm], truth)["mr"]
    assert table.attrs["source"] == multitrack.SOURCE_MULTIRING
    assert np.allclose(table["vertex_dist"], 2.0)
    assert np.allclose(table["angle_deg"], 0.0)
    assert np.allclose(table["p_res_pct"], 4.0)
    assert table["pid_correct"].all()


def test_a_single_track_sample_has_no_multitrack_section():
    rows = [{"event": e, "ring": 0, "pdg": 11, "particle": "e-", "t": 0.0,
             "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0,
             "p": 400.0, "is_primary_lepton": True, "above_threshold": True,
             "visible": True} for e in range(50)]
    truth = Truth(rings=schema.conform(pd.DataFrame(rows), "truth_rings"),
                  events=schema.conform(pd.DataFrame({"event": range(50)}), "truth_events"))
    assert multitrack.build([], truth) == {}


# ---------------------------------------------------------------------------
# The report end to end
# ---------------------------------------------------------------------------

def _gun_arms(n=300, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(150, 1500, n)
    x, y, z = (rng.uniform(-2500, 2500, n) for _ in range(3))
    truth_rings = pd.DataFrame({
        "event": np.arange(n), "ring": 0, "pdg": 13, "particle": "mu-", "t": 0.0,
        "x": x, "y": y, "z": z, "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": p,
        "is_primary_lepton": True, "above_threshold": True, "visible": True})
    sub = ([_sub(e, 0, 0.0, True) for e in range(n)]
           + [_sub(e, 1, 1500.0, False) for e in range(0, n, 2)])
    truth = Truth(rings=schema.conform(truth_rings, "truth_rings"),
                  events=schema.conform(pd.DataFrame({"event": np.arange(n)}),
                                        "truth_events"),
                  subevents=schema.conform(pd.DataFrame(sub), "truth_subevents"))

    def arm(label, noise):
        fits = []
        for hyp, dnll_sign in (("e-", 1.0), ("mu-", -1.0)):
            fits.append(pd.DataFrame({
                "event": np.arange(n), "sub_event": 0, "hypothesis": hyp,
                "ring": 0, "nrings": 1, "stage": schema.STAGE_FINAL,
                "success": True,
                "nll": 100.0 + dnll_sign * rng.normal(20, 10, n),
                "t": rng.normal(0, 1, n), "x": x + rng.normal(0, noise, n),
                "y": y, "z": z, "dx": 0.0, "dy": 0.0, "dz": 1.0,
                "p": p * (1 + rng.normal(0, 0.03, n))}))
        return Arm(label, "coral", fits=schema.conform(pd.concat(fits), "fits"),
                   prefits=_prefits([(e, 0, 1.0) for e in range(n)]
                                    + [(e, 1, 1501.0) for e in range(0, n, 4)]))
    return [arm("A", 20.0), arm("B", 30.0)], truth


def test_report_carries_every_validation_spec_section(tmp_path):
    from sextant import analysis

    arms, truth = _gun_arms()
    e_arms, e_truth = _gun_arms(seed=2)
    for a in e_arms:                       # an electron sample: flip the sign
        a.fits["nll"] = np.where(a.fits["hypothesis"] == "e-", 80.0, 120.0)
    analysis.standard_report(
        arms, truth, tmp_path, hypothesis="mu-", crb_model=None,
        pid_samples={"mu-": arms, "e-": e_arms},
        pid_truths={"mu-": truth, "e-": e_truth})
    tables = {p.stem for p in (tmp_path / "tables").glob("*.csv")}
    for name in ("resolution_p_true", "resolution_dwall", "resolution_towall",
                 "pid", "pid_efficiency_profiles", "subevents",
                 "subevent_profiles", "subevent_counts"):
        assert name in tables, name
    headings = [s["heading"] for s in
                json.loads((tmp_path / "manifest.json").read_text())["sections"]]
    assert "Resolution against towall" in headings
    summary = pd.read_csv(tmp_path / "tables" / "subevents.csv").set_index(["run", "metric"])
    # half the events have a decay electron; each arm finds half of those
    assert summary.loc[("A", "decay efficiency"), "value"] == pytest.approx(50.0)
    assert summary.loc[("A", "prompt efficiency"), "value"] == 100.0


def test_gaussian_fit_never_leaves_the_range_of_the_data():
    """A distribution with no Gaussian core (a leading-ring momentum residual
    piled against -100%) drove the fitted centre to -138%, which the quantity
    cannot take."""
    rng = np.random.default_rng(3)
    values = np.clip(np.r_[rng.uniform(-100, -20, 700), rng.normal(10, 30, 240)],
                     -100, None)
    mu, sigma = stats.gaussian_fit(values)
    assert values.min() <= mu <= values.max()
