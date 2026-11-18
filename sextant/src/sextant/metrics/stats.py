"""Summary statistics, and honest uncertainties on them.

### Two families, and which one a quantity belongs to

Reconstruction residuals come in two shapes and they are not summarised the same
way (convention set 2026-09-11):

**One-sided** -- a magnitude that starts at zero and has only an upper tail:
vertex displacement, its transverse component, opening angle. There is no "bias"
to speak of, so these get a SINGLE number, the **68th percentile** `p68`: the
error 68% of events come in under. Reporting a median and a width for a quantity
that cannot go negative says less than one honest quantile.

**Two-sided** -- a signed residual expected to be Gaussian about its true value:
the raw t/x/y/z/theta/phi residuals, the momentum residual, and the signed
along-track vertex displacement. These get a **Gaussian fit**, reporting `mu`
(the bias) and `sigma` (the width). A fit rather than a mean and a standard
deviation, because a handful of mis-converged events move both enormously while
leaving the core they are supposed to describe untouched.

`iqr68` and the rest are kept: they are what the earlier campaign numbers are
quoted in, and the fit needs something robust to seed and to fall back on.

Reconstruction residuals are not Gaussian. Vertex displacement and opening angle
are one-sided with heavy tails, and a handful of mis-converged events inflate a
standard deviation enormously -- which is why the project's own summary tables
report median and inter-percentile ranges rather than mean and sigma. The
functions here are the vocabulary the rest of Sextant reports in.

`iqr68` is the half-width of the central 68% interval, (p84 - p16)/2. It is the
robust analogue of a Gaussian sigma and coincides with it for a true Gaussian,
which is what makes "resolution" comparable between a well-behaved arm and one
with a bad tail.

Every headline number comes with a bootstrap confidence interval. A comparison
of two arms on a few hundred events routinely produces differences well inside
their uncertainties, and a table that reports 26.7 cm against 25.8 cm without
saying that the interval straddles zero invites a conclusion the data does not
support.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP = 2000
GAUSS_BOOTSTRAP = 200
DEFAULT_CI = 68.0


def finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def median(values) -> float:
    v = finite(values)
    return float(np.median(v)) if v.size else np.nan


def iqr68(values) -> float:
    """Half the central 68% width -- the robust stand-in for a Gaussian sigma."""
    v = finite(values)
    if v.size < 2:
        return np.nan
    return float((np.percentile(v, 84) - np.percentile(v, 16)) / 2.0)


def quantile(values, q) -> float:
    v = finite(values)
    return float(np.percentile(v, q)) if v.size else np.nan


def rms68(values) -> float:
    """RMS of the central 68% of the distribution, by |value|.

    Reported alongside iqr68 because they answer slightly different questions:
    iqr68 is about the spread of the bulk, rms68 about the size of a typical
    error once the worst third is set aside.
    """
    v = finite(values)
    if v.size == 0:
        return np.nan
    cut = np.percentile(np.abs(v), 68)
    core = v[np.abs(v) <= cut]
    return float(np.sqrt(np.mean(core ** 2))) if core.size else np.nan


def p68(values) -> float:
    """The 68th percentile of |value| -- the one-sided resolution.

    For a one-sided magnitude this is the whole summary: the error that 68% of
    events come in under. It coincides with a Gaussian sigma for the magnitude
    of a 1-D Gaussian, so it stays comparable with the iqr68 the earlier
    campaign numbers are quoted in.
    """
    v = finite(values)
    return float(np.percentile(np.abs(v), 68)) if v.size else np.nan


def gaussian_fit(values, *, nsigma=2.5, iterations=3, bins=80):
    """(mu, sigma) from a binned Gaussian fit to the core of a distribution.

    Iteratively re-fits within +-nsigma of the current estimate, seeded from the
    median and iqr68. The iteration is what makes it a fit to the GAUSSIAN CORE
    rather than to the whole histogram: a few mis-converged events in a tail
    would otherwise drag a least-squares fit off the peak they have nothing to
    do with.

    Falls back to (median, iqr68) if there is too little data or the fit does
    not converge, so a caller always gets usable numbers.
    """
    v = finite(values)
    if v.size < 20:
        return (median(v), iqr68(v))
    mu, sigma = median(v), iqr68(v)
    if not np.isfinite(sigma) or sigma <= 0:
        return (mu, sigma)
    try:
        from scipy.optimize import curve_fit
    except ImportError:                                   # pragma: no cover
        return (mu, sigma)

    def _gauss(x, amp, m, sd):
        return amp*np.exp(-0.5*((x - m)/sd)**2)

    for _ in range(iterations):
        lo, hi = mu - nsigma*sigma, mu + nsigma*sigma
        core = v[(v >= lo) & (v <= hi)]
        if core.size < 20:
            break
        counts, edges = np.histogram(core, bins=bins)
        centres = 0.5*(edges[:-1] + edges[1:])
        try:
            with warnings.catch_warnings():
                # The covariance is discarded, so a singular one is not a
                # problem worth printing once per quantity per arm.
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_gauss, centres, counts,
                                    p0=[counts.max(), mu, sigma], maxfev=5000)
        except (RuntimeError, ValueError):                # pragma: no cover
            break
        if not np.isfinite(popt).all() or popt[2] == 0:
            break
        # A centre outside the window it was fitted in is an extrapolation, not
        # a core: on a distribution with no Gaussian peak (a leading-ring
        # momentum piled up at -100%) the fit ran off to -138%, a value the
        # quantity cannot take.
        if not lo <= popt[1] <= hi:
            break
        mu, sigma = float(popt[1]), float(abs(popt[2]))
    return (mu, sigma)


def gauss_mu(values) -> float:
    return gaussian_fit(values)[0]


def gauss_sigma(values) -> float:
    return gaussian_fit(values)[1]


def fraction_within(values, limit) -> float:
    v = finite(values)
    return float(np.mean(np.abs(v) <= limit)) if v.size else np.nan


STATISTICS = {
    "median": median,
    "iqr68": iqr68,
    "rms68": rms68,
    "mean": lambda v: float(np.mean(finite(v))) if finite(v).size else np.nan,
    "std": lambda v: float(np.std(finite(v), ddof=1)) if finite(v).size > 1 else np.nan,
    "p68": p68,
    "gauss_mu": gauss_mu, "gauss_sigma": gauss_sigma,
    "p16": lambda v: quantile(v, 16), "p84": lambda v: quantile(v, 84),
    "p01": lambda v: quantile(v, 1), "p99": lambda v: quantile(v, 99),
    "n": lambda v: int(finite(v).size),
}


def bootstrap_ci(values, statistic=median, *, n_resamples=DEFAULT_BOOTSTRAP,
                 confidence=DEFAULT_CI, seed=0):
    """(low, high) percentile bootstrap interval for a statistic.

    Seeded by default so a report regenerated from the same inputs produces the
    same intervals -- an error bar that moves when nothing changed is worse than
    no error bar.
    """
    v = finite(values)
    if v.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    stats = np.array([statistic(v[row]) for row in idx], dtype=float)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return (np.nan, np.nan)
    tail = (100.0 - confidence) / 2.0
    return (float(np.percentile(stats, tail)),
            float(np.percentile(stats, 100.0 - tail)))


def paired_difference(a_values, b_values, statistic=median, **kwargs) -> dict:
    """Compare two arms on the *same* events, with a bootstrap CI on the delta.

    Pairing matters: a per-event difference removes the event-to-event variation
    that both arms share, so it resolves a difference far smaller than either
    arm's own spread. `a` and `b` must already be aligned by event.
    """
    a = np.asarray(a_values, float)
    b = np.asarray(b_values, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    delta = a[mask] - b[mask]
    low, high = bootstrap_ci(delta, statistic, **kwargs)
    value = statistic(delta) if delta.size else np.nan
    return {
        "n_paired": int(delta.size), "delta": value,
        "ci_low": low, "ci_high": high,
        # "significant" here means only: the interval excludes zero. It is a
        # statement about sampling noise, not about systematic differences
        # between the two configurations, which no resampling can see.
        "excludes_zero": bool(np.isfinite(low) and np.isfinite(high)
                              and (low > 0 or high < 0)),
    }


def summarise(values, name="", *, statistics=("n", "median", "iqr68", "rms68",
                                              "p16", "p84"),
              ci_for=("median", "iqr68"), seed=0) -> dict:
    """One row of a summary table: the statistics plus CIs on the headline ones."""
    row = {"metric": name}
    for key in statistics:
        row[key] = STATISTICS[key](values)
    for key in ci_for:
        if key in STATISTICS:
            # a Gaussian-fit statistic is an iterated least-squares fit per resample:
            # 200 resamples resolve a 68% interval; percentile statistics are cheap
            n = GAUSS_BOOTSTRAP if key.startswith("gauss") else DEFAULT_BOOTSTRAP
            low, high = bootstrap_ci(values, STATISTICS[key], n_resamples=n, seed=seed)
            row[f"{key}_lo"], row[f"{key}_hi"] = low, high
    return row


def summary_table(named_values: dict, **kwargs) -> pd.DataFrame:
    return pd.DataFrame([summarise(v, name=k, **kwargs)
                         for k, v in named_values.items()])


# ---------------------------------------------------------------------------
# Rates (efficiencies, purities): binomial, with Wilson intervals
# ---------------------------------------------------------------------------

#: Bins with fewer trials than this are dropped from a rate profile.
MIN_RATE_EVENTS = 10


def wilson(k, n, z: float = 1.0) -> tuple:
    """Wilson score interval for k successes in n (z = 1: a 68% interval).

    Not the normal approximation, which gives an interval outside [0, 1] for
    the rates near 100% that efficiencies and purities mostly are.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1.0 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def rate(mask) -> dict:
    """A rate in %, its 68% Wilson interval and k/n, from a boolean array."""
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), int(mask.size)
    lo, hi = wilson(k, n)
    return {"value": 100.0 * k / n if n else np.nan, "k": k, "n": n,
            "ci_lo": 100.0 * float(lo) if n else np.nan,
            "ci_hi": 100.0 * float(hi) if n else np.nan}


def rate_profile(values, mask, edges, *,
                 min_events: int = MIN_RATE_EVENTS) -> pd.DataFrame:
    """Rate (%) per bin of `values`, with Wilson 68% intervals."""
    values = np.asarray(values, float)
    mask = np.asarray(mask, bool)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        last = index == len(edges) - 2
        sel = (values >= low) & ((values <= high) if last else (values < high))
        if sel.sum() < min_events:
            continue
        rows.append({"low": low, "high": high, "centre": 0.5 * (low + high),
                     **rate(mask[sel])})
    return pd.DataFrame(rows)
