"""Tests for sextant.crb -- each piece of the bound checked on its own.

The toy-MLE test at the end is the one that matters most: it fits simulated events
by maximum likelihood and checks their spread equals the inverse Fisher matrix,
which exercises the derivatives, the frame, the assembly and the inversion at once.
It needs the built model and is skipped without it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sextant.crb.response import (ChargeResponse, Q_THRESHOLD, SIGMA_1PE,
                                  exact_charge_information, location_information)

MODEL = (Path(__file__).resolve().parents[3] / "Output" / "Sextant" / "crb_model"
         / "crb_model.npz")


# -- charge channel ---------------------------------------------------------
def test_charge_information_never_exceeds_poisson_counting():
    resp = ChargeResponse()
    mu = np.logspace(-4, 3, 60)
    assert np.all(resp.weight(mu) * mu <= 1.0 + 1e-6)


def test_charge_information_limits():
    # Dim PMTs: a thinned Poisson counter, efficiency = P(one pe clears threshold).
    eff = 1.0 - norm.cdf((Q_THRESHOLD - 1.0) / SIGMA_1PE)
    info, phit = exact_charge_information(1e-3)
    assert info * 1e-3 == pytest.approx(eff, rel=0.01)
    assert phit / 1e-3 == pytest.approx(eff, rel=0.01)
    # Bright PMTs: the excess-noise factor F = 1 + sigma_1pe^2.
    info, _ = exact_charge_information(200.0)
    assert info * 200.0 == pytest.approx(1.0 / (1.0 + SIGMA_1PE ** 2), rel=0.01)


@pytest.mark.parametrize("mu", [0.7, 4.0])
def test_charge_information_matches_score_variance(mu):
    """Fisher information = variance of the score, checked by simulation."""
    rng = np.random.default_rng(3)
    n_obs = 40000

    def logp(q, hit, m):
        n = np.arange(1, 60)
        pn = np.exp(n * np.log(m) - m - np.cumsum(np.log(n)))
        sd = SIGMA_1PE * np.sqrt(n)
        p0 = np.exp(-m) + np.sum(pn * norm.cdf((Q_THRESHOLD - n) / sd))
        dens = norm.pdf((q[:, None] - n) / sd) / sd @ pn
        return np.where(hit, np.log(np.maximum(dens, 1e-300)), np.log(p0))

    n = rng.poisson(mu, n_obs)
    q = np.array([rng.normal(1.0, SIGMA_1PE, k).sum() if k else 0.0 for k in n])
    hit = (n > 0) & (q > Q_THRESHOLD)
    h = 1e-4
    score = (logp(q, hit, mu + h) - logp(q, hit, mu - h)) / (2 * h)
    expected, _ = exact_charge_information(mu)
    assert np.mean(score ** 2) == pytest.approx(expected, rel=0.04)


# -- time channel -----------------------------------------------------------
def test_location_information_of_a_gaussian():
    x = np.random.default_rng(1).normal(0.0, 1.5, 20000)
    info, n = location_information(x)
    assert n == 20000
    assert info == pytest.approx(1 / 1.5 ** 2, rel=0.05)


def test_location_information_of_a_known_mixture():
    """Against the exact integral for a core + broad-tail mixture like the data."""
    w, (m1, s1), (m2, s2) = 0.7, (0.0, 1.0), (3.0, 4.0)
    t = np.linspace(-40, 60, 200001)
    f = w * norm.pdf(t, m1, s1) + (1 - w) * norm.pdf(t, m2, s2)
    fp = (w * norm.pdf(t, m1, s1) * -(t - m1) / s1 ** 2
          + (1 - w) * norm.pdf(t, m2, s2) * -(t - m2) / s2 ** 2)
    exact = np.trapz(fp ** 2 / f, t)
    rng = np.random.default_rng(4)
    n = 20000
    core = rng.random(n) < w
    x = np.where(core, rng.normal(m1, s1, n), rng.normal(m2, s2, n))
    info, _ = location_information(x)
    assert info == pytest.approx(exact, rel=0.06)


def test_location_information_of_a_mixture_is_below_its_core():
    """A late tail dilutes timing information; it must not add to it."""
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1.0, 14000), rng.exponential(15.0, 6000)]
    info, _ = location_information(x)
    assert 0.3 < info < 1.0


# -- the smooth profile lookup ------------------------------------------------
def test_profile_is_exact_at_bin_centres_and_linear_between():
    from sextant.crb.profile import CherenkovProfile
    nr, nt, npars = 6, 5, 5
    pars = np.zeros((3, nr, nt, 2 * npars + 2))
    rng = np.random.default_rng(0)
    const = rng.uniform(1, 2, (3, nr, nt))
    pars[..., 0] = const                         # one section, constant in ln p
    pars[..., npars] = 0.0
    pars[..., npars + 1] = 20.0
    prof = CherenkovProfile(npars, pars, np.ones((3, nr, nt), int),
                            np.linspace(0, 600, nr + 1), np.linspace(-1, 1, nt + 1),
                            [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], (0, 10))
    r = prof.r0_ctr[[1, 3]]
    c = prof.th0_ctr[[2, 4]]
    np.testing.assert_allclose(prof.integrals(300.0, r, c), const[:, [1, 3], [2, 4]])
    mid = 0.5 * (prof.r0_ctr[1] + prof.r0_ctr[2])
    got = prof.integrals(300.0, np.array([mid]), prof.th0_ctr[[2]])[:, 0]
    np.testing.assert_allclose(got, 0.5 * (const[:, 1, 2] + const[:, 2, 2]))


# -- summarising the bound like an arm -----------------------------------------
def test_efficient_residuals_reproduce_known_statistics():
    from sextant.crb.compare import efficient_residuals
    from sextant.metrics import stats
    n = 400
    sig_v, sig_a, sig_p = 2.0, 0.003, 0.02
    cov = np.diag([0.1 ** 2, sig_v ** 2, sig_v ** 2, sig_v ** 2, sig_a ** 2, sig_a ** 2,
                   (sig_p * 500.0) ** 2])
    table = pd.DataFrame({"event": np.arange(n), "p_true": 500.0,
                          "dx_true": 1.0, "dy_true": 0.0, "dz_true": 0.0})
    res = pd.DataFrame({"event": np.arange(n), "p_true": 500.0, "dwall": 1000.0})
    out = efficient_residuals(table, np.repeat(cov[None], n, 0), res, n_draw=50)
    rayleigh68 = np.sqrt(-2 * np.log(0.32))                  # 68% of a 2-D |gaussian|
    assert stats.gauss_sigma(out["vertex_long"]) == pytest.approx(sig_v, rel=0.03)
    assert stats.p68(out["vertex_trans"]) == pytest.approx(rayleigh68 * sig_v, rel=0.03)
    assert stats.p68(out["angle"]) == pytest.approx(rayleigh68 * sig_a, rel=0.03)
    assert stats.gauss_sigma(out["p_res_pct"]) == pytest.approx(100 * sig_p, rel=0.03)


def test_true_particle_refuses_mixed_samples():
    from sextant.crb.compare import true_particle
    assert true_particle(pd.DataFrame({"particle": ["e-"] * 50})) == "e-"
    assert true_particle(pd.DataFrame({"particle": ["e-"] * 30 + ["mu-"] * 20})) is None


def test_cli_has_crb_commands():
    from sextant.cli import build_parser
    args = build_parser().parse_args(["crb", "event", "--particle", "mu-", "-p", "700"])
    assert args.crb_command == "event" and args.p == 700.0


# -- the end-to-end check -----------------------------------------------------
@pytest.mark.skipif(not MODEL.exists(), reason="built CRB model not present")
def test_maximum_likelihood_fits_scatter_by_the_inverse_fisher_matrix():
    """Toy events from the model, fitted by Fisher scoring from the truth.

    Poisson counts and Gaussian hit times, so the observation model's exact
    Fisher matrix is the one assembled from the same derivatives as the bound.
    """
    from sextant.crb import CRBModel
    from sextant.crb.fisher import STEPS, _direction, _inverse, frame
    m = CRBModel.load(MODEL)
    light = m.light["e-"]
    v0 = np.array([400.0, -900.0, 600.0])
    d0, e1, e2 = frame(np.array([0.2, 0.6, -0.77]))
    x0 = np.r_[0.0, v0, 0.0, 0.0, 600.0]

    def model(x):
        d = _direction(d0, e1, e2, x[4], x[5])
        return light.charge(x[1:4], d, x[6]), light.time(x[0], x[1:4], d, x[6])

    def derivs(x):
        D, E = [], []
        for k in range(7):
            h = STEPS[k] * (x[6] if k == 6 else 1.0)
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            (mp, tp), (mm, tm) = model(xp), model(xm)
            D.append((mp - mm) / (2 * h))
            E.append((tp - tm) / (2 * h))
        return np.array(D), np.array(E)

    mu0, t0 = model(x0)
    s = 1 / np.sqrt(m.timing["e-"](light.delta_alpha(v0, d0, 600.0), mu0))
    ph = 1 - np.exp(-mu0)

    def info(D, E, mu):
        return (np.einsum("aj,bj,j->ab", D, D, 1 / mu)
                + np.einsum("aj,bj,j->ab", E, E, ph / s ** 2))

    D, E = derivs(x0)
    crb = _inverse(info(D, E, mu0))
    rng = np.random.default_rng(11)
    fits = []
    for _ in range(120):
        n = rng.poisson(mu0)
        t = t0 + rng.normal(size=mu0.size) * s
        x = x0.copy()
        for _ in range(2):
            mu, T = model(x)
            D, E = derivs(x)
            score = D @ ((n - mu) / mu) + E @ np.where(n > 0, (t - T) / s ** 2, 0.0)
            x = x + _inverse(info(D, E, mu)) @ score
        fits.append(x - x0)
    ratio = np.sqrt(np.diag(np.cov(np.array(fits).T)) / np.diag(crb))
    # 120 toys pin a width to ~6.5%; 25% is a 4-sigma failure.
    assert np.all(np.abs(ratio - 1.0) < 0.25), ratio


def test_ratio_figure_survives_a_bin_the_bound_cannot_fill():
    """Muons below the model's lowest momentum leave the bound a bin short."""
    import matplotlib
    matplotlib.use("Agg")
    from sextant.plots import residual_plots, style as st
    rng = np.random.default_rng(0)
    n = 600
    arm = pd.DataFrame({"event": np.arange(n), "p_true": rng.uniform(100, 1500, n),
                        "vertex_dist": rng.gamma(2, 10, n), "vertex_long": rng.normal(0, 15, n),
                        "vertex_trans": rng.gamma(2, 4, n), "angle_deg": rng.gamma(2, 1, n),
                        "t": rng.normal(0, 1, n), "p_res_pct": rng.normal(0, 3, n)})
    bound = arm[arm.p_true > 300].copy()
    for c in ("vertex_dist", "vertex_long", "vertex_trans", "angle_deg", "p_res_pct"):
        bound[c] = bound[c] / 5
    fig = residual_plots.bound_ratio_profiles({"arm": arm}, bound, st.ArmStyle(["arm"]))
    assert fig is not None


# -- physics floors -------------------------------------------------------------
def _floors():
    from sextant.crb.floor import Floors
    lp = np.log([200.0, 1000.0])
    return Floors(lp, np.array([0.02, 0.01]), np.array([20.0, 30.0]), np.array([2.0, 3.0]),
                  np.array([np.nan, np.nan]), ("angle", "along", "perp"))


def test_floor_covariance_lies_along_the_track_frame():
    from sextant.crb.fisher import frame
    fl = _floors()
    d, e1, e2 = frame(np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]))
    cov = fl.covariance(d, e1, e2, 1000.0)
    assert np.isclose(d @ cov[1:4, 1:4] @ d, 30.0 ** 2)
    assert np.isclose(e1 @ cov[1:4, 1:4] @ e1, 3.0 ** 2) and np.isclose(d @ cov[1:4, 1:4] @ e1, 0.0)
    assert np.isclose(cov[4, 4], 0.01 ** 2) and cov[6, 6] == 0
    assert np.isclose(cov[0, 0], (30.0 / 29.98) ** 2)                       # the time moves with the vertex
    assert np.isclose(fl.sigmas(np.sqrt(200.0 * 1000.0))["along"], 25.0)    # interpolated in log p
    assert fl.sigmas(5000.0)["along"] == 30.0                               # clamped at the ends


def test_floors_survive_the_model_file():
    from sextant.crb.floor import Floors
    fl = _floors()
    back = Floors.from_arrays(fl.to_arrays("e"), "e")
    assert back.applies == fl.applies and np.allclose(back.along, fl.along)
    assert Floors.from_arrays({}, "e") is None


@pytest.mark.skipif(not MODEL.exists(), reason="built CRB model not present")
def test_bound_with_floors_adds_them_to_the_fisher_covariance():
    from sextant.crb import CRBModel, bound
    m = CRBModel.load(MODEL)
    m.floors = {"e-": _floors()}
    v, d = np.zeros(3), np.array([1.0, 0.0, 0.0])
    a, b = bound(m, "e-", v, d, 1000.0, floor=False), bound(m, "e-", v, d, 1000.0)
    assert np.isclose(b["sigma_long"] ** 2, a["sigma_long"] ** 2 + 30.0 ** 2, rtol=1e-6)
    assert np.isclose(b["sigma_p_rel"], a["sigma_p_rel"])
