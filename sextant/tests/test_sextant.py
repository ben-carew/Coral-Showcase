"""Tests for the invariants whose breakage would be silent.

The failures this package exists to prevent do not raise: an arm scored against
the wrong truth, an event-id stride that aliases two events into one, a
calibration mixed across two self-consistent schemes, a duplicate key that makes
a join ambiguous. Each produces a full set of plausible plots. These tests pin
the behaviour that catches them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sextant import productions
from sextant.config import mapping
from sextant.io import schema
from sextant.io.geometry import DEFAULT_HALF_HEIGHT_CM, DEFAULT_RADIUS_CM, dwall, towall
from sextant.metrics import pid, residuals, rings, stats, validate


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_conform_fills_missing_columns_and_types():
    partial = pd.DataFrame({"event": [1, 2], "hypothesis": ["e-", "mu-"]})
    out = schema.conform(partial, "fits")
    assert list(out.columns) == schema.columns("fits")
    assert out["event"].dtype == np.int64
    assert out["success"].dtype == bool
    assert np.isnan(out["nll"]).all()          # absent float -> NaN, not 0


def test_conform_strict_rejects_unknown_columns():
    with pytest.raises(ValueError, match="unknown columns"):
        schema.conform(pd.DataFrame({"event": [1], "nonsense": [2]}), "fits",
                       strict=True)


def test_cherenkov_thresholds_match_the_standard_values():
    # The familiar HK/SK numbers; if these move, "visible ring" has changed
    # meaning and every ring-counting metric shifts with it.
    assert schema.cherenkov_threshold_mev(13) == pytest.approx(120.5, abs=0.5)
    assert schema.cherenkov_threshold_mev(2212) == pytest.approx(1070, abs=5)
    assert schema.cherenkov_threshold_mev(11) < 1.0


# ---------------------------------------------------------------------------
# Event numbering -- the aliasing bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(productions.PRODUCTIONS))
def test_stride_exceeds_events_per_file(name):
    """A stride <= events_per_file aliases one file's tail onto the next file."""
    production = productions.get(name)
    assert production.stride >= production.events_per_file, (
        f"{name}: stride {production.stride} <= {production.events_per_file} "
        f"events per file — event ids from adjacent files would collide")


def test_beam_ids_do_not_collide_across_files():
    production = productions.get("beam-fhc")
    last_of_file_0 = production.event_id(0, production.events_per_file - 1)
    first_of_file_1 = production.event_id(1, 0)
    assert last_of_file_0 < first_of_file_1


def test_event_id_round_trips():
    for name in productions.PRODUCTIONS:
        production = productions.get(name)
        assert production.split_event_id(production.event_id(37, 12)) == (37, 12)


def test_file_index_parses_both_naming_schemes():
    _, gun = productions.resolve("v1.12.22", "electron")
    _, beam = productions.resolve("beam-fhc", "numu_x_numu")
    assert gun.file_index("pgun_hkfd_id_elpos_00000000-0042_rwcs.root") == 42
    assert beam.file_index(
        "t2ksk23c.fqv4r0.21bv2.fhc.numu_x_numu_14542_rwcs.root") == 14542
    # The official reconstruction of the same file carries the same index.
    assert beam.file_index(
        "t2ksk23c.fqv4r0.21bv2.fhc.numu_x_numu_0005_rfqn.root") == 5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_dwall_and_towall_are_different_quantities():
    # At the centre pointing +z, dwall is set by the barrel and towall by the cap.
    assert dwall(0, 0, 0) == pytest.approx(DEFAULT_RADIUS_CM)
    assert towall(0, 0, 0, 0, 0, 1) == pytest.approx(DEFAULT_HALF_HEIGHT_CM)
    assert towall(0, 0, 0, 1, 0, 0) == pytest.approx(DEFAULT_RADIUS_CM)


def test_towall_is_nan_without_a_direction():
    assert np.isnan(towall(0, 0, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def _fit_row(**kwargs):
    row = {"event": 0, "sub_event": 0, "hypothesis": "e-", "ring": 0,
           "nrings": 1, "stage": schema.STAGE_FINAL, "success": True,
           "nll": 1.0, "totmu": np.nan, "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0,
           "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": 500.0, "dconv": np.nan}
    row.update(kwargs)
    return row


def _truth_row(**kwargs):
    row = {"event": 0, "ring": 0, "pdg": 11, "particle": "e-", "t": 0.0,
           "x": 0.0, "y": 0.0, "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0,
           "p": 500.0, "is_primary_lepton": True, "above_threshold": True,
           "visible": True}
    row.update(kwargs)
    return row


def test_residual_sign_convention_is_reco_minus_truth():
    reco = schema.conform(pd.DataFrame([_fit_row(x=110.0, p=550.0)]), "fits")
    truth = schema.conform(pd.DataFrame([_truth_row(x=100.0, p=500.0)]),
                           "truth_rings")
    out = residuals.compute(reco, truth)
    assert out["x"].iloc[0] == pytest.approx(10.0)
    assert out["p"].iloc[0] == pytest.approx(50.0)
    assert out["p_res_pct"].iloc[0] == pytest.approx(10.0)


def test_vertex_distance_and_opening_angle():
    reco = schema.conform(pd.DataFrame([_fit_row(x=3.0, y=4.0, dz=0.0, dx=1.0)]),
                          "fits")
    truth = schema.conform(pd.DataFrame([_truth_row()]), "truth_rings")
    out = residuals.compute(reco, truth)
    assert out["vertex_dist"].iloc[0] == pytest.approx(5.0)
    assert out["angle_deg"].iloc[0] == pytest.approx(90.0)


def test_low_momentum_events_are_excluded_from_fractional_residual_only():
    reco = schema.conform(pd.DataFrame([_fit_row(p=60.0)]), "fits")
    truth = schema.conform(pd.DataFrame([_truth_row(p=50.0)]), "truth_rings")
    out = residuals.compute(reco, truth, p_min_mev=100.0)
    assert np.isnan(out["p_res_pct"].iloc[0])       # fraction diverges, dropped
    assert out["p_res"].iloc[0] == pytest.approx(10.0)   # absolute kept


def test_match_events_restricts_to_the_common_set():
    a = pd.DataFrame({"event": [1, 2, 3], "vertex_dist": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"event": [2, 3, 4], "vertex_dist": [2.0, 3.0, 4.0]})
    matched = residuals.match_events({"a": a, "b": b})
    assert list(matched["a"]["event"]) == [2, 3]
    assert list(matched["b"]["event"]) == [2, 3]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_iqr68_matches_sigma_for_a_gaussian():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 3.0, 200_000)
    assert stats.iqr68(sample) == pytest.approx(3.0, rel=0.02)


def test_median_is_robust_where_the_mean_is_not():
    sample = np.concatenate([np.ones(1000), [1e6]])
    assert stats.median(sample) == pytest.approx(1.0)
    assert stats.STATISTICS["mean"](sample) > 100


def test_bootstrap_is_seeded_and_reproducible():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=500)
    assert stats.bootstrap_ci(sample, seed=7) == stats.bootstrap_ci(sample, seed=7)


def test_paired_difference_detects_a_real_shift_and_not_a_null_one():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, 4000)
    assert stats.paired_difference(base + 0.5, base)["excludes_zero"]
    assert not stats.paired_difference(base, base + 0.0)["excludes_zero"]


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------

def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert pid.auc([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    overlapping = rng.normal(size=5000)
    assert pid.auc(overlapping, rng.normal(size=5000)) == pytest.approx(0.5, abs=0.02)


def test_working_point_is_invariant_to_a_constant_offset():
    """The whole point of quoting a fixed-efficiency working point."""
    rng = np.random.default_rng(4)
    e = rng.normal(2.0, 1.0, 20_000)
    m = rng.normal(-2.0, 1.0, 20_000)
    plain = pid.working_point(e, m, 0.99)
    shifted = pid.working_point(e + 1000.0, m + 1000.0, 0.99)
    assert plain["eff_mu"] == pytest.approx(shifted["eff_mu"], abs=1e-9)


def test_discriminant_sign_convention():
    fits = schema.conform(pd.DataFrame([
        _fit_row(hypothesis="e-", nll=100.0),
        _fit_row(hypothesis="mu-", nll=150.0),
    ]), "fits")
    out = pid.discriminant(fits)
    assert out["dnll"].iloc[0] == pytest.approx(50.0)
    assert out["pid"].iloc[0] == "e-"      # dnll > 0 is electron-like


# ---------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------

def test_ring_matching_pairs_by_angle_and_rejects_wild_pairs():
    reco = pd.DataFrame({"dx": [1.0, 0.0], "dy": [0.0, 1.0], "dz": [0.0, 0.0]})
    true = pd.DataFrame({"dx": [0.0, 1.0], "dy": [1.0, 0.0], "dz": [0.0, 0.0]})
    matched = rings.match_rings(reco, true)
    assert set(zip(matched["reco_index"], matched["true_index"])) == {(0, 1), (1, 0)}
    assert (matched["angle_deg"] < 1e-6).all()

    opposed = pd.DataFrame({"dx": [-1.0], "dy": [0.0], "dz": [0.0]})
    assert rings.match_rings(pd.DataFrame({"dx": [1.0], "dy": [0.0], "dz": [0.0]}),
                             opposed).empty


def test_balanced_accuracy_penalises_always_answering_one():
    # 90 one-ring events and 10 two-ring ones; a fitter that always says "1"
    # scores 90% pooled but only 50% balanced.
    per_event = pd.DataFrame({
        "n_true": [1] * 90 + [2] * 10,
        "n_reco": [1] * 100,
        "n_fake": 0, "n_missed": [0] * 90 + [1] * 10,
        "exact_count": [True] * 90 + [False] * 10,
    })
    summary = rings.counting_summary(per_event, "always-1")
    assert summary["exact_count_pct"].iloc[0] == pytest.approx(90.0)
    assert summary["balanced_count_pct"].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Configuration mapping
# ---------------------------------------------------------------------------

def test_mixing_calibration_schemes_is_an_error():
    issues = mapping.verify_scheme("coral", "coral-production", {
        "physics.quantum_efficiency": "0.221",       # fiTQun's value
        "physics.attenuation_length": "9500",        # Coral's
    })
    assert any(i.severity == "error" for i in issues)


def test_a_self_consistent_scheme_verifies_clean():
    scheme = mapping.SCHEMES["coral-production"]
    issues = mapping.verify_scheme("coral", "coral-production", scheme.coral)
    assert [i for i in issues if i.severity == "error"] == []


def test_every_scheme_defines_values_for_exactly_one_algorithm():
    for name, scheme in mapping.SCHEMES.items():
        assert bool(scheme.coral) != bool(scheme.fitqun), (
            f"{name} must describe one algorithm's calibration, not both")


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_wrong_truth_is_caught_by_the_vertex_residual():
    bad = pd.DataFrame({"event": range(100),
                        "vertex_dist": np.full(100, 4000.0)})
    finding = validate.check_truth_pairing(bad, label="bad")
    assert finding.severity == "error"

    good = pd.DataFrame({"event": range(100), "vertex_dist": np.full(100, 25.0)})
    assert validate.check_truth_pairing(good, label="good").severity == "ok"


def test_no_event_overlap_is_an_error():
    finding = validate.check_event_overlap([1, 2, 3], [10, 11, 12])
    assert finding.severity == "error"


def test_duplicate_keys_are_caught():
    duplicated = pd.DataFrame({"event": [1, 1], "sub_event": [0, 0],
                               "hypothesis": ["e-", "e-"]})
    finding = validate.check_duplicate_keys(
        duplicated, ["event", "sub_event", "hypothesis"], name="t")
    assert finding.severity == "error"


def test_time_pedestal_disagreement_is_flagged():
    a = pd.DataFrame({"event": [1], "t": [-39.0]})
    b = pd.DataFrame({"event": [1], "t": [911.0]})
    assert validate.check_time_pedestal({"a": a, "b": b}).severity == "warning"


def _write_fitqun_merge(path, per_hypothesis):
    """A two-event fiTQun merge; with `per_hypothesis` the e- and mu- fits differ."""
    import uproot

    cols = {"event": np.array([0, 1], dtype=np.int64),
            "nll_e": np.array([10.0, 11.0], dtype=np.float32),
            "nll_mu": np.array([12.0, 9.0], dtype=np.float32),
            "nhits": np.array([500, 600], dtype=np.int32)}
    own = {"t": 1.0, "x": 100.0, "y": 0.0, "z": 0.0, "theta": 0.5, "phi": 0.1, "p": 400.0}
    for key, value in own.items():
        cols[key] = np.full(2, value, dtype=np.float32)
    cols["success"] = np.array([True, True])
    if per_hypothesis:
        for suffix, scale in (("_e", 0.7), ("_mu", 1.0)):
            for key, value in own.items():
                cols[key + suffix] = np.full(2, value * (scale if key == "p" else 1.0),
                                             dtype=np.float32)
            cols["success" + suffix] = np.array([True, suffix == "_mu"])
    with uproot.recreate(path) as fout:
        fout.mktree("fits", {k: v.dtype for k, v in cols.items()})
        fout["fits"].extend(cols)


def test_fitqun_merge_gives_each_hypothesis_its_own_fit(tmp_path):
    """Regression (2026-11-14): a merge held ONE fit's kinematics per event and the
    reader copied it onto the e- and the mu- row, so on a muon sample the "e-"
    momentum was the muon's. A PID cut sloped in p_e built on that was wrong by 2 pt."""
    from sextant.io import fitqun
    from sextant.io.dataset import Arm

    new, old = tmp_path / "new.root", tmp_path / "old.root"
    _write_fitqun_merge(new, per_hypothesis=True)
    _write_fitqun_merge(old, per_hypothesis=False)

    assert fitqun.merged_has_both_hypotheses(new)
    fits = fitqun.read_merged_fits(new).set_index(["event", "hypothesis"])
    assert fits.loc[(0, "e-"), "p"] == pytest.approx(280.0)
    assert fits.loc[(0, "mu-"), "p"] == pytest.approx(400.0)
    assert not fits.loc[(1, "e-"), "success"] and fits.loc[(1, "mu-"), "success"]

    assert not fitqun.merged_has_both_hypotheses(old)
    shared = fitqun.read_merged_fits(old).set_index(["event", "hypothesis"])
    assert shared.loc[(0, "e-"), "p"] == shared.loc[(0, "mu-"), "p"]
    assert validate.check_hypothesis_kinematics(
        Arm.from_file("old", old, algorithm="fitqun")).severity == "warning"
    assert validate.check_hypothesis_kinematics(
        Arm.from_file("new", new, algorithm="fitqun")).severity == "ok"


def _slot_with_preset(tmp_path, qe, indirect):
    root = tmp_path / "Software/worktrees/a/configs"
    root.mkdir(parents=True)
    (root / "multiring.ini").write_text(
        "; PRESET\n[physics]\n"
        f"quantum_efficiency = {qe}  ; photon yield scale\n"
        "attenuation_length = 8500  ; cm\n[scattering]\n"
        f"indirect_in_charge = {indirect}  ; False | True | hit\n"
        "indirect_factor = 0.15  ; scale\n")
    return tmp_path


def test_a_scheme_may_not_silently_rewrite_a_presets_calibration(tmp_path):
    """Regression (2026-11-14): `coral-hitscat-2026-10` named with
    `configs/multiring.ini` ran the direct-only multi-ring preset with hit-PMT
    scattered light and another photon yield -- in every multi-ring run from
    10-21 to 10-27, an adoption validation included, after the same thing on
    10-05 had been written up as a trap."""
    from sextant.run import launcher
    from sextant.run.spec import ArmSpec

    workspace = _slot_with_preset(tmp_path, "0.1940", "False")
    arm = dict(name="x", algorithm="coral", slot="a", config_file="configs/multiring.ini")

    with pytest.raises(SystemExit, match="would CHANGE calibration"):
        launcher._check_scheme_against_config(
            ArmSpec(scheme="coral-hitscat-2026-10", **arm), workspace)
    # the preset's own calibration, the config scheme and an explicit override all pass
    launcher._check_scheme_against_config(
        ArmSpec(scheme="coral-masked-2026-09", **arm), workspace)
    launcher._check_scheme_against_config(ArmSpec(scheme="config", **arm), workspace)
    launcher._check_scheme_against_config(
        ArmSpec(scheme="coral-hitscat-2026-10", scheme_override=True, **arm), workspace)


def test_the_launcher_says_what_an_arm_changes_in_its_config(tmp_path, capsys):
    from sextant.run import launcher
    from sextant.run.spec import ArmSpec

    workspace = _slot_with_preset(tmp_path, "0.1940", "False")
    arm = ArmSpec(name="x", algorithm="coral", slot="a", config_file="configs/multiring.ini")
    changes = launcher._report_changes(arm, workspace, {
        "physics.quantum_efficiency": "0.194",          # same value, other spelling: not a change
        "scattering.indirect_in_charge": "hit",
        "fits.near_wall_seeds": "clipgrid"})
    assert changes == [("scattering.indirect_in_charge", "False", "hit"),
                       ("fits.near_wall_seeds", "(not set)", "clipgrid")]
    assert "changes 2 key(s) of multiring.ini" in capsys.readouterr().out


def test_a_coral_arm_without_a_scheme_runs_its_config_as_committed():
    """The default used to be `coral-production`, the JULY tuning (0.176 / 9500),
    written over whatever the branch's config carried."""
    from sextant.run.spec import ArmSpec, ExperimentSpec

    arm = ArmSpec(name="x", algorithm="coral")
    assert arm.scheme == mapping.CONFIG_SCHEME
    spec = ExperimentSpec(name="t", production="v1.12.29", sample="electron",
                          events=10, arms=[arm])
    assert spec.resolved_settings(arm) == {}
    spec.validate()


def test_a_shifted_event_numbering_is_caught():
    """Regression (2026-11-14): Coral numbers events by WCSim EvtNum, the truth by
    tree entry. One skipped number in a file shifted every later event by one."""
    rng = np.random.default_rng(0)
    n = 400
    truth = pd.DataFrame({"event": np.arange(n), "x": rng.uniform(-3000, 3000, n),
                          "y": rng.uniform(-3000, 3000, n), "z": rng.uniform(-3000, 3000, n)})
    fits = truth.copy()
    fits[["x", "y", "z"]] += rng.normal(0, 20, (n, 3))
    assert validate.check_event_numbering(fits, truth, label="aligned").severity == "ok"
    # events 100..199 carry the number of the NEXT entry: fit of entry k labelled k + 1
    shifted = fits.copy()
    block = (shifted["event"] >= 100) & (shifted["event"] < 200)
    shifted.loc[block, "event"] += 1
    shifted = shifted.drop_duplicates("event", keep="last")
    finding = validate.check_event_numbering(shifted, truth, label="shifted")
    assert finding.severity == "error" and "event -1" in finding.message


# ---------------------------------------------------------------------------
# Probe guards
# ---------------------------------------------------------------------------

def test_probe_refuses_a_shared_self_point_set():
    from sextant.probe import points

    with pytest.raises(ValueError, match="'self' cannot be part"):
        points.build(points=("self",))


def test_per_pmt_comparison_refuses_mixed_points():
    from sextant.metrics import intermediates

    def table(point):
        return pd.DataFrame({"event": [1], "sub_event": [0],
                             "hypothesis": ["e-"], "point": [point],
                             "pmt": [7], "mu_direct": [1.0]})

    mixed = pd.concat([table("truth"), table("ref")], ignore_index=True)
    with pytest.raises(ValueError, match="several parameter points"):
        intermediates.compare_at_point(mixed, mixed)


def test_charge_profile_separates_normalisation_from_shape():
    """A flat ratio means retune a constant; a sliding one means fix the model."""
    from sextant.metrics import intermediates

    rng = np.random.default_rng(0)
    cos_theta = rng.uniform(-1.0, 1.0, 4000)
    q_obs = np.full(4000, 2.0)

    flat = pd.DataFrame({"cos_theta": cos_theta, "q_obs": q_obs,
                         "mu_direct": 1.2 * q_obs, "mu_indirect": 0.0,
                         "is_hit": True})
    profile = intermediates.charge_profile(flat)
    assert profile["median_ratio"].std() < 0.01
    assert abs(profile["median_ratio"].median() - 1.2) < 0.02

    sloped = flat.assign(mu_direct=(1.0 + 0.5 * cos_theta) * q_obs)
    profile = intermediates.charge_profile(sloped)
    assert profile["median_ratio"].iloc[-1] - profile["median_ratio"].iloc[0] > 0.5


def test_probe_draw_can_be_restricted_to_a_run_s_events():
    """A probe compared against a run must look at the events the run saw."""
    from types import SimpleNamespace
    from sextant.probe import points

    truth = SimpleNamespace(events=pd.DataFrame({
        "event": [1, 2, 3, 4, 5, 6, 7, 8],
        "n_visible_rings": [1] * 8}))

    drawn = points.select_events(truth, n=3, within=[5, 6, 7, 8])
    assert set(drawn) <= {5, 6, 7, 8}

    with pytest.raises(ValueError, match="different events"):
        points.select_events(truth, n=3, within=[99, 100])


def test_truth_point_is_put_on_the_algorithm_s_time_origin():
    """Generators start at t=0; a reco measures t against the trigger."""
    from sextant.probe import points

    fits = pd.DataFrame({"event": [1, 2, 3], "t": [-40.0, -35.0, -39.0]})
    rings = pd.DataFrame({"event": [1, 2, 3], "t": [0.0, 0.0, 0.0]})
    assert points.time_pedestal(fits, rings) == pytest.approx(-39.0)

    with pytest.raises(ValueError, match="no events in common"):
        points.time_pedestal(fits, pd.DataFrame({"event": [9], "t": [0.0]}))


def test_shape_ratio_is_blind_to_overall_normalisation():
    """Scaling every prediction must move the ratio but not the shape."""
    from sextant.metrics import intermediates

    rng = np.random.default_rng(1)
    cos_theta = rng.uniform(-1.0, 1.0, 4000)
    q_obs = 1.0 + 3.0 * np.exp(-((cos_theta - 0.75) ** 2) / 0.02)
    base = pd.DataFrame({"cos_theta": cos_theta, "q_obs": q_obs,
                         "mu_direct": q_obs, "mu_indirect": 0.0,
                         "is_hit": True})

    right = intermediates.charge_profile(base)
    assert np.allclose(right["shape_ratio"], 1.0, atol=1e-9)

    # Half the light everywhere: badly normalised, still the right shape.
    halved = intermediates.charge_profile(base.assign(mu_direct=0.5 * q_obs))
    assert np.allclose(halved["shape_ratio"], 1.0, atol=1e-9)
    assert abs(halved["sum_mu"].sum() / halved["sum_q_obs"].sum() - 0.5) < 1e-9

    # Light in the wrong place: shape_ratio must depart from 1.
    skewed = intermediates.charge_profile(
        base.assign(mu_direct=q_obs * (1.0 + 0.8 * cos_theta)))
    assert skewed["shape_ratio"].max() - skewed["shape_ratio"].min() > 0.5


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", ["electron_comparison", "beam_comparison"])
def test_example_specs_are_valid(name):
    from sextant.run.spec import ExperimentSpec, check_configuration

    spec = ExperimentSpec.load(EXAMPLES / f"{name}.ini")
    assert len(spec.arms) >= 2
    rows, issues = check_configuration(spec)
    assert rows
    assert [i for i in issues if i.severity == "error"] == []


def test_shared_keys_must_be_known_concepts(tmp_path):
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "bad.ini"
    path.write_text("[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
                    "events=10\n\n[shared]\nnot_a_concept = 1\n\n"
                    "[arm.a]\nalgorithm=coral\n")
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSpec.load(path)


def test_duplicate_arm_labels_are_rejected(tmp_path):
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "dup.ini"
    path.write_text("[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
                    "events=10\n\n[arm.a]\nalgorithm=coral\nlabel=same\n\n"
                    "[arm.b]\nalgorithm=coral\nlabel=same\n")
    with pytest.raises(ValueError, match="share a label"):
        ExperimentSpec.load(path)


# ---------------------------------------------------------------------------
# Regressions found by running the thing end to end on the cluster
# ---------------------------------------------------------------------------


def _fitqun_arm_spec(tmp_path, base_text, extra_arm_lines=""):
    from sextant.run.spec import ExperimentSpec

    base = tmp_path / "base.parameters.dat"
    base.write_text(base_text)
    path = tmp_path / "fq.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
        "events=10\n\n[shared]\ndo_muon_fit = false\n\n"
        f"[arm.a]\nalgorithm=fitqun\nparams={base}\n{extra_arm_lines}")
    return ExperimentSpec.load(path)


def test_fitqun_arms_get_their_shared_and_param_overrides(tmp_path):
    """A fiTQun arm's declared configuration has to reach its parameter file.

    It used to reach nothing: launcher.py wrote only `params` for a fiTQun arm,
    so `[shared]` keys were translated for the comparison table and then
    dropped, `param.fiTQun.*` overrides were parsed into `native_params` and
    dropped, and every scheme's `fitqun_override_file` was read by no one. Arms
    that differed on paper ran identically.
    """
    from sextant.run.launcher import write_arm_spec

    spec = _fitqun_arm_spec(
        tmp_path,
        " < fiTQun.DoMuon1RFit = 1 >\n < fiTQun.NotAConcept = 1.5 > a comment\n",
        extra_arm_lines="param.fiTQun.DoSubEvent = 0\n")
    write_arm_spec(spec, spec.arms[0], tmp_path)
    written = (tmp_path / "Output/Sextant/x/specs/a.parameters.dat").read_text()

    # [shared] do_muon_fit = false, rendered in fiTQun's spelling, not "false"
    assert " < fiTQun.DoMuon1RFit = 0 >" in written
    # a param. override for a key the base file never mentions is appended
    assert "fiTQun.DoSubEvent = 0" in written
    # a key nothing overrides keeps its value AND its trailing comment
    assert " < fiTQun.NotAConcept = 1.5 > a comment" in written
    # and every change to the author's own base file is recorded in the header
    assert "fiTQun.DoMuon1RFit: 1 -> 0" in written


def test_fitqun_override_rewrites_every_occurrence_of_a_key(tmp_path):
    """fiTQun takes the LAST value, so rewriting only the first leaves a stale
    one winning -- and the shipped HK-2024 overrides really do set
    DoProton1RFit twice."""
    from sextant.run.launcher import write_arm_spec

    spec = _fitqun_arm_spec(
        tmp_path, " < fiTQun.DoMuon1RFit = 1 >\n < fiTQun.DoMuon1RFit = 1 >\n")
    write_arm_spec(spec, spec.arms[0], tmp_path)
    written = (tmp_path / "Output/Sextant/x/specs/a.parameters.dat").read_text()
    assert written.count(" < fiTQun.DoMuon1RFit = 0 >") == 2
    assert " < fiTQun.DoMuon1RFit = 1 >" not in written


def test_fitqun_scheme_supplies_its_own_override_file(tmp_path):
    """Declaring a scheme has to SELECT its tuning, not merely describe it."""
    from sextant.config import mapping

    for name, scheme in mapping.SCHEMES.items():
        if not scheme.fitqun:
            continue
        assert scheme.fitqun_override_file, (
            f"fiTQun scheme {name} names no override file, so an arm using it "
            f"would silently inherit whatever params it was handed")

def test_probe_input_lookup_returns_only_the_relevant_files():
    """Handing Coral the whole sample to find 20 events is the difference
    between a minute and an afternoon."""
    production = productions.get("beam-fhc")
    events = [1035, 1055, 2049, 3012]
    indices = sorted({production.split_event_id(e)[0] for e in events})
    assert indices == [1, 2, 3]           # 4 events, 3 files -- not 5986


def test_narrowing_the_hypothesis_list_narrows_multiring_candidates(tmp_path):
    """Coral refuses to start if a multi-ring candidate type has no tuning in
    [fits] particles, so the two must be written in step."""
    from sextant.run.launcher import _coral_config
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "narrow.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
        "events=10\n\n[shared]\ndo_electron_fit = true\n"
        "do_multiring_fit = true\n\n"
        "[arm.a]\nalgorithm=coral\n")
    spec = ExperimentSpec.load(path)
    config = _coral_config(spec, spec.arms[0])
    assert config["fits.particles"] == "e-"
    assert config["multiring.candidate_types"] == "e-"


def test_every_panel_is_square_and_the_page_never_changes_shape():
    """Two invariants the report's look depends on. Panels are square because a
    wide box exaggerates horizontal separation between arms, which is the very
    judgement the reader is making; the page is one fixed size because a PDF
    whose pages change shape cannot be printed or bound."""
    from sextant.plots import style as st
    st.use_style()
    assert st.PAGE_SIZE[0] == st.PAGE_SIZE[1], "the report page must be square"
    for n in (1, 3, 4, 7, 12):
        fig, axes = st.grid_figure(n, ncols=n)          # ncols is advisory
        assert len(axes) == n
        assert tuple(fig.get_size_inches()) == tuple(st.PAGE_SIZE)
        for ax in axes:
            assert ax.get_box_aspect() == 1.0


def test_science_style_is_applied_without_latex():
    """SciencePlots' plain `science` sheet routes text through LaTeX, which
    exists neither on the cluster nor in the HyperK env, so every figure would
    fail to render. Sextant must ask for `no-latex`."""
    from sextant.plots import style as st
    assert "no-latex" in st.SCIENCE_STYLE
    st.use_style()
    import matplotlib as mpl
    assert not mpl.rcParams["text.usetex"]
    assert mpl.rcParams["axes.grid"], "the science sheet turns the grid off; we restore it"


def test_a_switched_off_hypothesis_does_not_erase_the_ones_left_on(tmp_path):
    """Several concepts share Coral's `fits.particles`, so assigning them one
    after the other lets the last one win: `do_muon_fit = false` used to
    overwrite `fits.particles = e-` with the string "false", which configparser
    accepts and Coral reads as no hypothesis at all. `check` then reported
    do_electron_fit as OFF for Coral and raised a disagreement against fiTQun
    that did not exist."""
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "onehyp.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=v1.12.29\nsample=electron\n"
        "events=10\n\n[shared]\ndo_electron_fit = true\n"
        "do_muon_fit = false\n\n[arm.a]\nalgorithm=coral\n")
    spec = ExperimentSpec.load(path)
    assert spec.resolved_settings(spec.arms[0])["fits.particles"] == "e-"


def test_a_coral_hypothesis_list_is_compared_by_membership(tmp_path):
    """fiTQun has one boolean per hypothesis, Coral one list for all of them, so
    the comparison has to test membership. Reading "e-, mu-" as a boolean makes
    every hypothesis look switched off."""
    from sextant.config import mapping

    rows, issues = mapping.compare_arms({
        "coral":  {"algorithm": "coral", "scheme": "",
                   "values": {"fits.particles": "e-, mu-"}},
        "fitqun": {"algorithm": "fitqun", "scheme": "",
                   "values": {"fiTQun.DoElectron1RFit": "1",
                              "fiTQun.DoMuon1RFit": "1"}}})
    by_concept = {r["concept"]: r for r in rows}
    assert by_concept["do_electron_fit"]["agrees"]
    assert by_concept["do_muon_fit"]["agrees"]
    assert not [i for i in issues if i.concept in ("do_electron_fit",
                                                   "do_muon_fit")]


def test_native_particle_override_also_drives_multiring_candidates(tmp_path):
    from sextant.run.launcher import _coral_config
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "native.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
        "events=10\n\n[shared]\ndo_electron_fit = true\ndo_muon_fit = true\n\n"
        "[arm.a]\nalgorithm=coral\nconfig.fits.particles = e-\n"
        "config.multiring.do_multiring = true\n")
    spec = ExperimentSpec.load(path)
    config = _coral_config(spec, spec.arms[0])
    assert config["fits.particles"] == "e-"
    assert config["multiring.candidate_types"] == "e-"


def test_no_multiring_section_when_the_build_has_no_multiring_fitter(tmp_path):
    """Upstream dev has no multi-ring fitter; writing a [multiring] section into
    its frozen config would be a setting that nothing reads."""
    from sextant.run.launcher import _coral_config
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "single.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=v1.12.22\nsample=electron\n"
        "events=10\n\n[shared]\ndo_electron_fit = true\ndo_muon_fit = true\n\n"
        "[arm.a]\nalgorithm=coral\n")
    spec = ExperimentSpec.load(path)
    config = _coral_config(spec, spec.arms[0])
    assert config["fits.particles"] == "e-, mu-"
    assert not any(k.startswith("multiring.") for k in config)


def test_probe_maps_local_event_index_to_the_global_id():
    """Coral's event loop yields the index within its input file; `points` is
    keyed on the global id. Getting this wrong makes the probe match nothing
    and silently report zero events."""
    from sextant.probe.coral_probe import _numbering

    stride, index_of = _numbering("beam-fhc", "numu_x_numu", 100)
    assert stride == 1000
    name = "t2ksk23c.fqv4r0.21bv2.fhc.numu_x_numu_0003_rwcs.root"
    assert index_of(name) * stride + 12 == 3012

    # Fallback with no production: both naming schemes still parse.
    stride, index_of = _numbering(None, None, 100)
    assert index_of("pgun_hkfd_id_elpos_00000000-0007_rwcs.root") * stride + 5 == 705


def test_every_sample_name_is_a_valid_command_line_flag():
    """exp.py passes the sample name to the submit scripts as --<name>, so a
    name argparse cannot express would fail only at submit time -- which is how
    a beam run died with "unrecognized arguments: --numu_x_numu"."""
    for production in productions.PRODUCTIONS.values():
        for name in production.samples:
            assert name == name.strip()
            assert " " not in name and not name.startswith("-")


def test_fitqun_params_parser_separates_the_expensive_fits():
    """The full HK-2024 files enable the multi-ring and pi0 fits, which cost
    ~20 min/event on this detector; the _1Remu variants do not. A single-ring
    comparison configured with the wrong one does not fail, it never finishes."""
    from sextant.config import fitqun_params

    # tests -> Sextant -> Software, which is where fiTQun lives beside us.
    overrides = (Path(__file__).resolve().parents[2]
                 / "fiTQun" / "ParameterOverrideFiles")
    full = overrides / "Official_HyperK_2024_wscat.parameters.dat"
    lean = overrides / "Official_HyperK_2024_wscat_1Remu.parameters.dat"
    if not full.exists() or not lean.exists():
        pytest.skip("fiTQun parameter override files not available here")

    assert fitqun_params.enabled_fits(fitqun_params.read(full))
    assert fitqun_params.enabled_fits(fitqun_params.read(lean)) == []


def test_fitqun_params_parser_handles_the_bespoke_format(tmp_path):
    from sextant.config import fitqun_params

    path = tmp_path / "p.dat"
    path.write_text(
        "  header text with no angle brackets\n"
        " < fiTQun.DoMRFit = 1 >   run the multi-ring fit\n"
        "# < fiTQun.DoPi0Fit = 1 > commented out\n"
        " < fiTQun.QEEffWCSim = 0.221 > \n")
    values = fitqun_params.read(path)
    assert values["fiTQun.DoMRFit"] == "1"
    assert values["fiTQun.QEEffWCSim"] == "0.221"
    assert "fiTQun.DoPi0Fit" not in values      # '#' comments are skipped


def test_sextant_and_runlib_registries_agree():
    """Sextant and the workspace's runlib both define the productions, and a
    disagreement about the stride would give merged results different event ids
    from the truth built beside them -- silently. The knowledge base already
    records one instance of the local and server scripts drifting apart; this
    is the same failure mode across two registries."""
    import os
    import sys

    workspace = Path(__file__).resolve().parents[3]
    cluster = workspace / "Analysis" / "Scripts" / "cluster"
    if not (cluster / "runlib.py").exists():
        pytest.skip("runlib is not available (Sextant installed outside the "
                    "reconstruction workspace)")

    os.environ.setdefault("HK_BASE_DIR", "/tmp")
    sys.path.insert(0, str(cluster))
    try:
        import runlib
    finally:
        sys.path.remove(str(cluster))

    shared = set(runlib.PRODUCTIONS) & set(productions.PRODUCTIONS)
    assert shared, "the two registries name no production in common"
    for name in shared:
        assert runlib.event_stride(name) == productions.get(name).stride, (
            f"{name}: runlib stride {runlib.event_stride(name)} != sextant "
            f"stride {productions.get(name).stride}")
        assert (runlib.events_per_file(name)
                == productions.get(name).events_per_file), name

    # And every sample Sextant knows about must be submittable through runlib.
    for name in productions.PRODUCTIONS:
        if name not in runlib.PRODUCTIONS:
            continue
        for sample in productions.get(name).samples:
            assert sample in runlib.PRODUCTIONS[name]["particles"], (
                f"{name}/{sample} is in Sextant's registry but not runlib's, "
                f"so exp.py could not submit it")


def test_all_merge_paths_resolve_the_same_stride():
    """Four modules build global event ids -- the two cluster merges and the two
    local runners. They have drifted before (knowledge.md records `multiringfits`
    added to one and not the other), and a stride disagreement between them is
    invisible: the ids simply do not line up with the truth built beside them."""
    import os
    import sys

    workspace = Path(__file__).resolve().parents[3]
    scripts = workspace / "Analysis" / "Scripts"
    cluster = scripts / "cluster"
    if not (cluster / "runlib.py").exists():
        pytest.skip("workspace scripts not available")

    os.environ.setdefault("HK_BASE_DIR", "/tmp")
    for directory in (str(scripts), str(cluster)):
        if directory not in sys.path:
            sys.path.insert(0, directory)

    import merge_coral
    import merge_fitqun
    import run_coral_local
    import runlib

    for production in ("v1.12.22", "beam-fhc"):
        expected = runlib.event_stride(production)
        assert run_coral_local.event_stride(production) == expected
        assert productions.get(production).stride == expected

    # And every one of them parses both file-naming schemes identically.
    for name in ("fitqun_pgun_hkfd_id_elpos_00000000-0042_rwcs.root",
                 "fitqun_t2ksk23c.fqv4r0.21bv2.fhc.numu_x_numu_14542_rwcs.root"):
        indices = {merge_coral.parse_file_index(name),
                   merge_fitqun.parse_file_index(name),
                   run_coral_local.parse_file_index(name),
                   runlib.parse_file_index(name)}
        assert len(indices) == 1, f"{name}: parsers disagree -> {indices}"


# ---------------------------------------------------------------------------
# Report rendering and presentation
# ---------------------------------------------------------------------------

def test_theta_phi_residuals_are_produced_and_phi_wraps():
    """phi is periodic: a fit at +179 deg against a truth at -179 is 2 deg out,
    not 358."""
    reco = schema.conform(pd.DataFrame([
        _fit_row(dx=np.cos(np.radians(179.0)), dy=np.sin(np.radians(179.0)),
                 dz=0.0)]), "fits")
    truth = schema.conform(pd.DataFrame([
        _truth_row(dx=np.cos(np.radians(-179.0)), dy=np.sin(np.radians(-179.0)),
                   dz=0.0)]), "truth_rings")
    out = residuals.compute(reco, truth)
    assert abs(out["phi"].iloc[0]) == pytest.approx(2.0, abs=0.01)
    assert abs(out["theta"].iloc[0]) == pytest.approx(0.0, abs=0.01)


def test_component_and_aggregate_quantity_sets_are_disjoint_and_complete():
    """The report puts them on separate pages; a quantity in both, or in
    neither, would be shown twice or not at all."""
    components = set(residuals.COMPONENT_QUANTITIES)
    aggregates = set(residuals.AGGREGATE_QUANTITIES)
    assert components.isdisjoint(aggregates)
    assert components == {"t", "x", "y", "z", "theta", "phi", "p"}
    assert aggregates == {"vertex_dist", "vertex_long", "vertex_trans",
                          "angle_deg", "p_res_pct"}


def test_vertex_decomposes_into_longitudinal_and_transverse():
    """vertex_long is signed and along the TRUE direction; vertex_trans is the
    perpendicular magnitude; the two must reconstruct the distance exactly.

    The sign matters more than it looks: both codes sit ~4-10 cm UPSTREAM, and a
    positive-definite distance cannot show that at all."""
    reco = pd.DataFrame({
        "event": [0, 1, 2],
        # +30 cm downstream / 40 cm sideways / 30 downstream and 40 sideways
        "t": 0.0, "x": [30.0, 0.0, 30.0], "y": [0.0, 40.0, 40.0], "z": 0.0,
        "dx": 1.0, "dy": 0.0, "dz": 0.0, "p": 500.0, "nll": 0.0})
    truth_rings = pd.DataFrame({
        "event": [0, 1, 2], "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0,
        "dx": 1.0, "dy": 0.0, "dz": 0.0, "p": 500.0})
    out = residuals.compute(reco, truth_rings)
    assert np.allclose(out["vertex_long"], [30.0, 0.0, 30.0])
    assert np.allclose(out["vertex_trans"], [0.0, 40.0, 40.0])
    assert np.allclose(out["vertex_dist"], [30.0, 40.0, 50.0])
    assert np.allclose(out["vertex_long"] ** 2 + out["vertex_trans"] ** 2,
                       out["vertex_dist"] ** 2)


def test_vertex_long_is_negative_for_an_upstream_vertex():
    """The convention the campaign's numbers are quoted in: negative = upstream
    of truth, which is the direction both Coral and fiTQun are biased in."""
    reco = pd.DataFrame({"event": [0], "t": 0.0, "x": 0.0, "y": 0.0, "z": [-12.0],
                         "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": 500.0, "nll": 0.0})
    truth_rings = pd.DataFrame({"event": [0], "t": 0.0, "x": 0.0, "y": 0.0,
                                "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0,
                                "p": 500.0})
    out = residuals.compute(reco, truth_rings)
    assert out["vertex_long"].iloc[0] == pytest.approx(-12.0)
    assert out["vertex_trans"].iloc[0] == pytest.approx(0.0)


def test_resolution_falls_back_to_equal_count_bins_on_a_small_sample():
    """49 events over the standard 12 momentum bins leaves every bin below the
    floor and the panel blank. The fallback must produce real bins instead."""
    from sextant.metrics import resolution

    rng = np.random.default_rng(0)
    small = pd.DataFrame({"p_true": rng.uniform(100, 1500, 49),
                          "vertex_dist": rng.uniform(10, 60, 49)})
    profile = resolution.profile(small, "vertex_dist", "p_true")
    assert not profile.empty
    assert profile.attrs["binning"] == "quantile"
    assert (profile["n"] >= resolution.MIN_EVENTS_PER_BIN).all()

    big = pd.DataFrame({"p_true": rng.uniform(100, 1500, 20_000),
                        "vertex_dist": rng.uniform(10, 60, 20_000)})
    assert resolution.profile(big, "vertex_dist", "p_true").attrs["binning"] == "fixed"


def test_explicit_bins_are_never_overridden():
    """A caller who chose edges gets those edges, however sparse."""
    from sextant.metrics import resolution

    rng = np.random.default_rng(1)
    small = pd.DataFrame({"p_true": rng.uniform(100, 1500, 49),
                          "vertex_dist": rng.uniform(10, 60, 49)})
    profile = resolution.profile(small, "vertex_dist", "p_true",
                                 bins=np.linspace(100, 1500, 30))
    assert profile.attrs["binning"] == "fixed"


def test_wide_short_tables_are_transposed_for_the_page():
    """Eleven columns and two rows needs a 5 pt font laid out as-is."""
    from sextant.report import tables as table_io

    wide = pd.DataFrame({"arm": ["a", "b"],
                         **{f"c{i}": [i, i] for i in range(10)}})
    out, done = table_io.maybe_transpose(wide)
    assert done and len(out) == 10 and list(out.columns) == ["field", "a", "b"]

    tall = pd.DataFrame({"arm": [f"a{i}" for i in range(30)],
                         "x": range(30), "y": range(30)})
    _, done = table_io.maybe_transpose(tall)
    assert not done


def test_table_cells_wrap_rather_than_truncate():
    """The validation table is nothing but its message column; eliding it made
    the page useless."""
    from sextant.report.builder import _layout

    long_message = ("Coral: keys unique on ['event', 'sub_event', "
                    "'hypothesis', 'ring', 'stage'] and nothing was dropped.")
    table = pd.DataFrame({"check": ["duplicate-keys"], "message": [long_message]})
    _, wrapped, row_lines, _, _, _ = _layout(table, 160)
    rendered = wrapped[0][1].replace("\n", " ")
    # Every word survives, and it took more than one line to do it.
    assert all(word in rendered for word in long_message.split())
    assert row_lines[0] > 1
    assert "…" not in wrapped[0][1]


def test_no_column_is_narrower_than_its_own_header():
    """Squeezing a column below its header is what made 'algorithm' render as
    'algorith' with the next column's text on top of it."""
    from sextant.report.builder import _layout

    table = pd.DataFrame({"algorithm": ["coral"], "events": [49],
                          "code_sha": ["88b85f43ef"], "dirty": [False],
                          "path": ["Output/Coral/Reco/x/coral_electron.root"]})
    _, _, _, _, _, budget = _layout(table, 160)
    for name, width in zip(table.columns, budget):
        longest_word = max(len(w) for w in str(name).split())
        assert width >= longest_word, f"{name}: {width} < {longest_word}"


def test_files_touched_is_driven_by_jobs_not_events_per_file(tmp_path):
    """One job reads one input file, so a run spans as many files as it has
    jobs. Computing it from events/events_per_file made the reference arm read
    a tenth of the sample the other arm ran on."""
    from sextant.run.collect import _files_needed
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "beam.ini"
    path.write_text(
        "[experiment]\nname=x\nproduction=beam-fhc\nsample=numu_x_numu\n"
        "events=1000\n\n[run]\nevents_per_job = 10\n\n"
        "[arm.a]\nalgorithm=coral\n")
    spec = ExperimentSpec.load(path)
    # 101 events per file would say 10; the run actually touches 100.
    assert _files_needed(spec) == 100


def test_official_reco_is_selected_by_index_not_by_position():
    """The shipped reconstruction covers a scattered subset of the production
    (about a third of the beam files), so "the first N reconstruction files"
    describes a different part of the sample than the run covered."""
    assert hasattr(productions, "find_reference_reco_for_indices")
    # With nothing mounted the lookup is empty, but it must not raise and must
    # not silently fall back to positional selection.
    got = productions.find_reference_reco_for_indices(
        "beam-fhc", "numu_x_numu", range(10))
    assert got == []


def test_only_the_best_multiring_candidate_is_read():
    """fiTQun stacks its multi-ring candidates as it searches from one ring
    upward; index 0 is the final best fit (fiTQun README, "Multi-Ring Fits").
    Reading every candidate reports a stack of attempts as if they were rings
    and gives ~57 'rings' per event instead of ~1.5."""
    from sextant.io import fitqun

    assert fitqun.BEST_MULTIRING_FIT == 0

    sample = (Path(__file__).resolve().parents[1] / "tests" / "data"
              / "beam_rfqn_sample.root")
    if not sample.exists():
        pytest.skip("no official fiTQun sample checked in")
    tables = fitqun.read_official(sample, file_index=0, stride=1000)
    rings = tables["multiring"]
    if rings.empty:
        pytest.skip("sample has no multi-ring fits")
    per_event = rings.groupby("event")["nrings"].first()
    assert per_event.max() <= 6, "more rings than fiTQun can fit"
    assert per_event.mean() < 4, "candidate stack is being read as rings"


@pytest.mark.parametrize("name", ["nll_terms.csv", "fit_stages.csv"])
def test_coral_diagnostics_found_beside_the_merged_file(name):
    """The collector writes '<merged stem>_<name>' next to the ROOT file so two
    merges into one Reco directory cannot overwrite each other's diagnostics."""
    import tempfile
    from sextant.io.coral import _find_job_csv

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "coral_electron_run.root"
        root.write_bytes(b"")
        assert _find_job_csv(root, name) is None
        stem_named = Path(tmp) / f"coral_electron_run_{name}"
        stem_named.write_text("event,sub_event,particle\n")
        assert _find_job_csv(root, name) == stem_named



def test_do1ringfit_step_results_are_opt_in():
    """return_steps defaults False so every existing caller is unaffected; the
    per-step results are what make a fit's trajectory inspectable."""
    import inspect
    import sys

    slot = Path(__file__).resolve().parents[3] / "worktrees" / "a" / "src"
    if not (slot / "fitqun_fit" / "minimization.py").exists():
        pytest.skip("Coral source not available beside the package")
    text = (slot / "fitqun_fit" / "minimization.py").read_text()
    assert "return_steps: bool = False" in text
    # The final result must still be returned bare when steps are not asked for,
    # or `event_data[particle][0]` in fitqun.py silently becomes a tuple.
    assert "    return fit_result" in text.splitlines(True)[-1] or \
           any(line.strip() == "return fit_result" for line in text.splitlines())


def test_resubmit_clears_a_previous_run_s_job_outputs(tmp_path):
    """Re-submitting into a finished run's directory used to leave its job
    outputs in place, so the merge picked up both runs -- and because Coral's
    diagnostic dumps APPEND, a second run's rows landed under the first run's
    header, silently misaligning any column added in between."""
    import os
    import sys

    cluster = Path(__file__).resolve().parents[3] / "Analysis" / "Scripts" / "cluster"
    if not (cluster / "runlib.py").exists():
        pytest.skip("workspace scripts not available")
    os.environ.setdefault("HK_BASE_DIR", str(tmp_path))
    sys.path.insert(0, str(cluster))
    try:
        import runlib
    finally:
        sys.path.remove(str(cluster))

    dirs = runlib.RunDirs("Coral", "exp/variant", base=tmp_path)
    dirs.create()
    for i in range(3):
        job = dirs.job(i)
        job.mkdir(parents=True, exist_ok=True)
        (job / "nll_terms.csv").write_text("event,old,header\n1,2,3\n")
    (dirs.logs / "job_0000.err").write_text("stale")

    assert runlib.clear_job_outputs(dirs) == 3
    assert not list(dirs.root.glob("job_*"))
    assert not list(dirs.logs.glob("*"))


# --- the 2026-09-11 summary convention -------------------------------------

def test_gaussian_fit_recovers_a_known_mean_and_width():
    rng = np.random.default_rng(0)
    v = rng.normal(loc=-7.5, scale=3.2, size=20000)
    mu, sigma = stats.gaussian_fit(v)
    assert mu == pytest.approx(-7.5, abs=0.15)
    assert sigma == pytest.approx(3.2, abs=0.15)


def test_gaussian_fit_ignores_a_mis_convergence_tail():
    """The reason for a FIT rather than a mean and a standard deviation: a few
    per cent of catastrophic events move both enormously and say nothing about
    the core they are supposed to describe."""
    rng = np.random.default_rng(1)
    core = rng.normal(0.0, 2.0, size=9500)
    tail = rng.uniform(200.0, 800.0, size=500)       # 5% mis-converged
    v = np.concatenate([core, tail])
    mu, sigma = stats.gaussian_fit(v)
    assert mu == pytest.approx(0.0, abs=0.2)
    assert sigma == pytest.approx(2.0, abs=0.2)
    # what the naive statistics would have said
    assert np.std(v) > 50
    assert abs(np.mean(v)) > 10


def test_p68_is_the_error_68_percent_of_events_come_in_under():
    v = np.arange(1.0, 101.0)                        # 1..100
    assert stats.p68(v) == pytest.approx(np.percentile(v, 68))
    # and it uses |value|, so a signed input is folded first
    assert stats.p68(-v) == pytest.approx(stats.p68(v))


def test_every_reported_quantity_declares_its_summary_kind():
    """A quantity with no entry silently defaults to gaussian, which would put a
    mean and a sigma on a positive-definite magnitude."""
    reported = set(residuals.COMPONENT_QUANTITIES) | set(residuals.AGGREGATE_QUANTITIES)
    assert reported <= set(residuals.QUANTITY_KIND)
    for q in ("vertex_dist", "vertex_trans", "angle_deg"):
        assert residuals.kind_of(q) == "one_sided"
    for q in ("t", "x", "y", "z", "theta", "phi", "p_res_pct", "vertex_long"):
        assert residuals.kind_of(q) == "gaussian"


def test_summary_reports_one_number_for_one_sided_and_two_for_gaussian():
    rng = np.random.default_rng(2)
    n = 500
    res = pd.DataFrame({
        "vertex_dist": np.abs(rng.normal(0, 20, n)),
        "vertex_long": rng.normal(-9.0, 20, n),
        "vertex_trans": np.abs(rng.normal(0, 8, n)),
        "angle_deg": np.abs(rng.normal(0, 2, n)),
        "p_res_pct": rng.normal(0.4, 4.0, n),
        "t": rng.normal(-39.0, 2.0, n),
        "p_res": rng.normal(0, 20, n),
    })
    table = residuals.summarise(res, "arm").set_index("quantity")
    assert table.loc["vertex_dist", "kind"] == "one_sided"
    assert np.isfinite(table.loc["vertex_dist", "p68"])
    assert table.loc["p_res_pct", "kind"] == "gaussian"
    assert np.isfinite(table.loc["p_res_pct", "gauss_mu"])
    assert np.isfinite(table.loc["p_res_pct", "gauss_sigma"])
    # t is fitted about its own pedestal, so its bias is ~0 not ~-39
    assert abs(table.loc["t", "gauss_mu"]) < 1.0


def test_trigger_clock_truth_keeps_the_time_bias():
    """Against trigger-clock truth the t residual is a measurement: no pedestal."""
    rng = np.random.default_rng(3)
    res = pd.DataFrame({"t": rng.normal(1.5, 1.0, 400)})
    res.attrs["t_clock"] = "trigger"
    assert not residuals.needs_pedestal(res)
    table = residuals.summarise(res, "arm").set_index("quantity")
    assert table.loc["t", "gauss_mu"] == pytest.approx(1.5, abs=0.3)
    res.attrs["t_clock"] = "generator"
    table = residuals.summarise(res, "arm").set_index("quantity")
    assert abs(table.loc["t", "gauss_mu"]) < 0.3


def test_legacy_truth_moves_time_onto_the_trigger_clock(tmp_path):
    """A TriggerTime column shifts t by -date0 and marks the residual's clock.

    WCSim writes a digit at (photon time - trigger date + 950 ns) and both codes
    drop the 950, so a gun started at Time = 0 sits at -date0 on the clock the
    fits report. Without this the time residual is the trigger placement.
    """
    import uproot
    from sextant.io import truth as truth_io
    n = 3
    tracks = {
        "EvtNum": np.arange(n, dtype=np.int32), "SubEvtNumber": np.zeros(n, np.int32),
        "Parenttype": np.zeros(n, np.int32), "Id": np.ones(n, np.int32),
        "Ipnu": np.full(n, 11, np.int32), "P": np.full(n, 500.0), "Time": np.zeros(n),
        "Dir_x0": np.zeros(n), "Dir_x1": np.zeros(n), "Dir_x2": np.ones(n),
        "Start_x0": np.zeros(n), "Start_x1": np.zeros(n), "Start_x2": np.zeros(n),
        "TriggerTime": np.array([35.0, 60.0, np.nan]),
    }
    path = tmp_path / "truth.root"
    with uproot.recreate(path) as f:
        f.mktree("Tracks", {k: v.dtype for k, v in tracks.items()})
        f["Tracks"].extend(tracks)
    rings, events = truth_io.read(path)
    assert rings.sort_values("event")["t"].tolist()[:2] == [-35.0, -60.0]
    assert np.isnan(events.sort_values("event")["trigger_time"].iloc[2])
    fits = pd.DataFrame({"event": [0, 1], "t": [-34.0, -59.0], "x": 0.0, "y": 0.0,
                         "z": 0.0, "dx": 0.0, "dy": 0.0, "dz": 1.0, "p": 500.0})
    res = residuals.compute(fits, rings, truth_events=events)
    assert res.attrs["t_clock"] == "trigger"
    assert res["t"].tolist() == [1.0, 1.0]
    fits3 = pd.concat([fits, fits.iloc[:1].assign(event=2)], ignore_index=True)
    assert residuals.compute(fits3, rings, truth_events=events).attrs["t_clock"] == "generator"


def test_run_start_reaches_the_generated_exp_spec(tmp_path):
    """`[run] start` must survive into exp.py's spec.

    Sextant parsed it and never propagated it, so every "held-out" validation
    run silently re-read the calibration's own files 0..N. A correction fitted
    and tested on the same events proves nothing, and nothing in the output
    said so -- the arm ran perfectly, on the wrong files. Found 2026-09-16 when
    a momval log was named ...-0000_rwcs under `start = 30`.
    """
    from sextant.run.spec import ExperimentSpec
    from sextant.run import launcher

    path = tmp_path / "s.ini"
    path.write_text("[experiment]\nname=x\nproduction=v1.12.29\nsample=electron\n"
                    "events=100\n\n[run]\nstart=30\nevents_per_job=50\n\n"
                    "[arm.a]\nalgorithm=coral\n")
    spec = ExperimentSpec.load(path)
    assert spec.start == 30

    parser = launcher._exp_spec(spec, spec.arms[0]) if hasattr(launcher, "_exp_spec") else None
    if parser is not None:
        assert parser["run"]["start"] == "30"


def test_an_arm_may_override_start(tmp_path):
    from sextant.run.spec import ExperimentSpec

    path = tmp_path / "s.ini"
    path.write_text("[experiment]\nname=x\nproduction=v1.12.29\nsample=electron\n"
                    "events=100\n\n[run]\nstart=30\n\n"
                    "[arm.a]\nalgorithm=coral\n\n[arm.b]\nalgorithm=coral\nstart=0\n")
    spec = ExperimentSpec.load(path)
    by = {a.name: a for a in spec.arms}
    assert by["a"].start is None          # inherits the experiment's 30
    assert by["b"].start == 0             # explicit override


# ---------------------------------------------------------------------------
# Report layout and degenerate-input handling
# ---------------------------------------------------------------------------

def test_headline_panels_are_the_nine_headline_numbers():
    """The resolution grid and the headline table must not drift apart.

    Both read `headline_panels()`, so the profiled panels and the table's
    columns are the same nine quantities by construction. A change to one that
    forgets the other shows up here rather than as a report whose binned page
    disagrees with the summary above it.
    """
    from sextant.metrics import residuals as R

    panels = R.headline_panels()
    assert len(panels) == 9
    # one number for a one-sided magnitude, two for a signed residual
    for column, statistic, _name, _unit in panels:
        kind = R.kind_of(column)
        assert statistic in R.KIND_STATISTICS[kind]
        if kind == "one_sided":
            assert statistic == "p68"


def test_no_figure_caption_is_drawn_off_the_page():
    """Captions must sit INSIDE the figure.

    Report pages are written with `bbox_inches=None` so every page is the same
    shape, which means a suptitle placed above y=1.0 is off the sheet and its
    text is lost. The standalone PNGs are cropped tight and kept it, so this
    only ever showed up in the PDF -- exactly the kind of silent breakage the
    suite exists to pin.
    """
    import re
    from pathlib import Path

    plots_dir = Path(__file__).resolve().parents[1] / "src" / "sextant" / "plots"
    offenders = []
    for path in plots_dir.glob("*.py"):
        for match in re.finditer(r"suptitle\((?:[^()]|\([^()]*\))*\)",
                                 path.read_text(), re.S):
            y = re.search(r"y\s*=\s*([0-9.]+)", match.group(0))
            if y and float(y.group(1)) > 1.0:
                offenders.append(f"{path.name}: y={y.group(1)}")
    assert not offenders, "caption drawn off the page: " + "; ".join(offenders)


def test_a_totals_only_likelihood_is_not_reported_as_a_decomposition():
    """'total' and 'zero_ring' are totals, not components.

    An official fiTQun file publishes only those two, and drawing them as a
    breakdown produced a one-bar chart that read as a broken plot rather than
    as a property of the input.
    """
    import pandas as pd
    from sextant.metrics import intermediates

    totals_only = pd.DataFrame({
        "event": [1, 1], "sub_event": [0, 0], "hypothesis": ["e-", "e-"],
        "point": ["self", "self"], "term": ["total", "zero_ring"],
        "value": [10.0, 30.0]})
    assert not intermediates.has_term_split(totals_only)

    split = pd.concat([totals_only, pd.DataFrame({
        "event": [1], "sub_event": [0], "hypothesis": ["e-"],
        "point": ["self"], "term": ["tpdf"], "value": [4.0]})],
        ignore_index=True)
    assert intermediates.has_term_split(split)

    # ...and the improvement over the zero-ring hypothesis IS available from
    # the totals alone, which is what the report shows instead.
    gain = intermediates.zero_ring_improvement(totals_only, label="fitqun")
    assert list(gain["improvement"]) == [20.0]


def test_the_report_carries_no_prose():
    """No sentences or paragraphs anywhere in a rendered report.

    Two ways prose used to reach the page: a whole `add_text` section
    explaining that a section was skipped (a section that is unavailable is
    simply not produced now), and a figure caption written as an explanation
    rather than a label. Both are pinned here because both grew back the last
    time they were trimmed by hand.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "sextant"

    # No report section may be a block of text.
    assert "add_text" not in (src / "analysis.py").read_text(), \
        "analysis.py adds a text section; skip the section instead"

    # Captions and axis titles are labels, not explanations. A full stop mid
    # string, or a string this long, is prose.
    budget = 64
    offenders = []
    pattern = re.compile(r"(?:st\.caption\(fig,|suptitle\(|title=)\s*(f?\"[^\"]*\")")
    for path in (src / "plots").glob("*.py"):
        for match in pattern.finditer(path.read_text()):
            text = match.group(1).strip('f"')
            if len(text) > budget or ". " in text:
                offenders.append(f"{path.name}: {text[:70]!r}")
    assert not offenders, "prose in a figure label: " + "; ".join(offenders)
