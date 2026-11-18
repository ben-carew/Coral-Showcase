"""What one PMT tells you: the exact information in its hit/charge and its time.

Charge channel
--------------
A PMT expecting mu photoelectrons receives n ~ Poisson(mu). Each photoelectron gives a
charge g ~ N(1, sigma_1pe^2); the PMT records a hit only if the summed charge clears
a threshold q_th, and then reports that charge. The information about mu in the
observation (hit?, q) is computed EXACTLY on a grid of mu and tabulated:

    w(mu) = P0'(mu)^2 / P0(mu)  +  INTEGRAL_{q > q_th} (d p(q|mu)/d mu)^2 / p(q|mu) dq

so that a PMT contributes w(mu_j) (d mu_j/d x_a)(d mu_j/d x_b) to the Fisher matrix.

This replaces an interpolation (e^-mu/(1-e^-mu) + (1-e^-mu)/(F mu)) that had the right
limits but overshot the Poisson ceiling 1/mu by up to 10 % near mu ~ 1 -- impossible,
since reading a charge cannot tell you more about mu than counting the photoelectrons
would -- and that ignored the threshold. `w(mu) * mu <= 1` is asserted.

Measured on the production sample: a single photoelectron gives a digitised hit 79.0 %
of the time, two 97.1 %, three 99.5 %; sigma_1pe = 0.478 with q_th = 0.615 reproduces
0.790 / 0.980 / 0.998.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

SIGMA_1PE = 0.478
Q_THRESHOLD = 0.615


def _pois(n, mu):
    return np.exp(n * np.log(mu) - mu - gammaln(n + 1))


def exact_charge_information(mu, sigma1=SIGMA_1PE, q_th=Q_THRESHOLD, nmax=None):
    """Fisher information about mu per unit (d mu)^2, and P(hit), for one mu."""
    mu = float(mu)
    if nmax is None:
        nmax = int(mu + 12 * np.sqrt(mu + 1) + 15)
    n = np.arange(1, nmax + 1)
    pn = _pois(n, mu)
    dpn = pn * (n / mu - 1.0)                       # d Pois / d mu
    sd = sigma1 * np.sqrt(n)
    below = norm.cdf((q_th - n) / sd)
    p0 = np.exp(-mu) + np.sum(pn * below)
    dp0 = -np.exp(-mu) + np.sum(dpn * below)
    lo = q_th
    hi = nmax + 8 * sigma1 * np.sqrt(nmax)
    q = np.linspace(lo, hi, 4000)
    dens = norm.pdf((q[:, None] - n) / sd) / sd     # (nq, n)
    p = dens @ pn
    dp = dens @ dpn
    ok = p > 1e-300
    info = np.trapz(np.where(ok, dp ** 2 / np.where(ok, p, 1), 0), q)
    if p0 > 1e-300:
        info += dp0 ** 2 / p0
    return info, 1.0 - p0


class ChargeResponse:
    """w(mu) and P(hit|mu), tabulated once and interpolated in log mu."""

    def __init__(self, sigma1=SIGMA_1PE, q_th=Q_THRESHOLD, grid=None):
        self.sigma1, self.q_th = sigma1, q_th
        grid = np.logspace(-5, 3.3, 180) if grid is None else grid
        vals = np.array([exact_charge_information(m, sigma1, q_th) for m in grid])
        self.logmu = np.log(grid)
        self.mu_w = vals[:, 0] * grid            # w * mu: 1 = Poisson counting
        self.phit = vals[:, 1]
        assert np.all(self.mu_w <= 1.0 + 1e-6), "information exceeds the Poisson ceiling"

    def weight(self, mu):
        """w(mu): information about mu per unit (d mu)^2."""
        lm = np.log(np.clip(mu, 1e-5, 1.9e3))
        return np.interp(lm, self.logmu, self.mu_w) / np.clip(mu, 1e-5, None)

    def p_hit(self, mu):
        lm = np.log(np.clip(mu, 1e-5, 1.9e3))
        return np.interp(lm, self.logmu, self.phit)


def sigma_t_ski(q):
    """WCSim SKI single-PMT time resolution, ns (used only for the 'ideal' variant)."""
    return np.maximum(0.58, 0.33 + np.sqrt(2.0 / np.maximum(q, 0.05)))


def location_information(samples, window=(-40.0, 80.0), max_components=5, seed=0):
    """Fisher information for a SHIFT of the distribution these samples came from.

    I = INTEGRAL f'(t)^2 / f(t) dt -- for a Gaussian, 1/sigma^2. Estimated from a
    Gaussian-mixture fit rather than a histogram, because a histogram's bin-to-bin
    noise enters f'^2 and biases the information upward.

    The number of components is chosen by BIC. A fixed, generous number is NOT
    safe: f'^2/f rewards narrow structure, so a mixture that over-fits noise into
    narrow components overstates the information -- a fixed 4-component fit to a
    pure Gaussian returned 14.5% too much.
    """
    from sklearn.mixture import GaussianMixture
    x = np.asarray(samples, float)
    x = x[(x > window[0]) & (x < window[1])]
    if x.size < 200:
        return np.nan, x.size
    x = x - np.median(x)
    best, best_bic = None, np.inf
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full",
                             max_iter=500, reg_covar=1e-3, n_init=2)
        gm.fit(x[:, None])
        bic = gm.bic(x[:, None])
        if bic < best_bic:
            best, best_bic = gm, bic
    w = best.weights_
    m = best.means_[:, 0]
    s = np.sqrt(best.covariances_[:, 0, 0])
    t = np.linspace(x.min() - 10, x.max() + 10, 20000)
    comp = w * norm.pdf((t[:, None] - m) / s) / s
    f = comp.sum(axis=1)
    fp = (comp * (-(t[:, None] - m) / s ** 2)).sum(axis=1)
    ok = f > 1e-12
    return float(np.trapz(np.where(ok, fp ** 2 / np.where(ok, f, 1), 0), t)), x.size
