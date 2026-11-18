"""Resolution as a function of the things it actually depends on.

A single median vertex error hides most of what distinguishes two algorithms.
Resolution varies strongly with how much light there is (true momentum) and how
much detector the ring had to develop in (towall) or how near the wall the
vertex sat (dwall) -- and two arms with the same overall median can have
completely different behaviour across those.

Bins with too few events are dropped rather than plotted with a huge error bar,
because a median of three events is not a measurement. `MIN_EVENTS_PER_BIN` is
the threshold and travels with the result so a plot can say so.

That floor is why a small run produced an entirely blank panel: 49 events spread
over the standard twelve momentum bins leaves 2-7 events in each, and every one
is correctly dropped. Correct, but useless. So when the fixed binning survives
nowhere, `profile` falls back to **quantile bins** sized to hold at least
`MIN_EVENTS_PER_BIN` events each -- the same data, binned to what it can
actually support. The result records which binning it used (`attrs['binning']`)
so the figure can say so rather than quietly implying the standard one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats

MIN_EVENTS_PER_BIN = 10

#: Below this many surviving bins a profile is not a trend, it is a point or
#: two. The adaptive fallback triggers on this rather than on "no bin at all":
#: a single fixed bin scraping past the event floor produced a panel with one
#: marker on it, which is no more use than a blank one.
MIN_USEFUL_BINS = 3

#: The standard binning variables: (column, label, unit, default bin edges).
BINNINGS = {
    "p_true": ("true momentum", "MeV/c", np.linspace(100, 1700, 13)),
    "dwall": ("distance to wall", "cm", np.linspace(0, 3000, 16)),
    # to 75 m: towall runs to the far corner of the detector (~92 m), and the
    # old 50 m ceiling silently dropped every event pointing across the tank.
    "towall": ("distance to wall along ring", "cm", np.linspace(0, 7500, 16)),
    "nhits": ("true hit PMTs", "", np.linspace(0, 8000, 17)),
    "nu_energy": ("neutrino energy", "MeV", np.linspace(0, 3000, 16)),
}

#: The statistic each profiled quantity is summarised by, per bin, following
#: the same convention as the summary tables: a 68th percentile for the
#: one-sided magnitudes, a fitted Gaussian mu for the signed ones (so a profile
#: of a signed residual reads as a BIAS against the binning variable, which is
#: how the near-wall light and vertex corrections were found).
QUANTITIES = {
    "vertex_dist": ("vertex resolution", "cm", "p68"),
    "vertex_long": ("vertex bias along track", "cm", "gauss_mu"),
    "vertex_trans": ("vertex resolution, transverse", "cm", "p68"),
    "angle_deg": ("angular resolution", "deg", "p68"),
    "p_res_pct": ("momentum bias", "%", "gauss_mu"),
}


def quantile_bins(values, min_events: int = MIN_EVENTS_PER_BIN,
                  max_bins: int = 12) -> np.ndarray:
    """Bin edges placing at least `min_events` finite values in every bin."""
    finite = np.asarray(values, float)
    finite = np.sort(finite[np.isfinite(finite)])
    if finite.size < min_events:
        return np.array([])
    n_bins = int(min(max_bins, finite.size // min_events))
    if n_bins < 1:
        return np.array([])
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    # The last bin is half-open in `profile`, so nudge the top edge to keep the
    # maximum value inside it.
    if edges.size >= 2:
        edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges if edges.size >= 2 else np.array([])


def profile(residuals: pd.DataFrame, quantity: str, versus: str, *,
            bins=None, statistic: str = "median",
            spread: str = "iqr68", min_events: int = MIN_EVENTS_PER_BIN,
            with_ci: bool = False, seed: int = 0,
            adaptive: bool = True) -> pd.DataFrame:
    """One row per bin: centre, the statistic, its spread and the count.

    `spread` is the *width of the distribution* in that bin, not the uncertainty
    on the statistic -- those are different quantities and conflating them is
    how a plot ends up claiming a precision it does not have. Set `with_ci` for
    a bootstrap interval on the statistic itself, reported separately.
    """
    if residuals.empty or quantity not in residuals or versus not in residuals:
        return pd.DataFrame(columns=["centre", "low", "high", "value", "spread",
                                     "p16", "p84", "sigma", "sigma_robust",
                                     "err_lo", "err_hi", "n"])
    requested = bins
    if bins is None:
        bins = BINNINGS.get(versus, (None, None, np.linspace(0, 1, 11)))[2]
    bins = np.asarray(bins, float)

    x = residuals[versus].to_numpy(float)
    y = residuals[quantity].to_numpy(float)
    stat_fn = stats.STATISTICS[statistic]
    spread_fn = stats.STATISTICS[spread]

    binning = "fixed"
    if adaptive and requested is None:
        usable = x[np.isfinite(x) & np.isfinite(y)]
        counts, _ = np.histogram(usable, bins=bins)
        if int((counts >= min_events).sum()) < MIN_USEFUL_BINS:
            adapted = quantile_bins(usable, min_events, max_bins=len(bins) - 1)
            if adapted.size >= 2:
                bins, binning = adapted, "quantile"

    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        sel = (x >= low) & (x < high) & np.isfinite(y)
        values = y[sel]
        if values.size < min_events:
            continue
        centre_value = stat_fn(values)
        # Asymmetric band from the actual percentiles, as well as the symmetric
        # spread. Vertex distance and opening angle are strictly positive, so a
        # symmetric +/- iqr68 bar drawn around a small median extends below zero
        # and claims a region the quantity cannot occupy.
        p16, p84 = stats.quantile(values, 16), stats.quantile(values, 84)
        # Error bars are ONE STANDARD DEVIATION of the values in the bin.
        # The percentile band is still computed and still written to the CSV --
        # it is the better descriptor of these skewed distributions -- but the
        # drawn bar is a sigma, which is what a reader of a physics plot expects
        # an error bar to mean.
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        # The bar is a ROBUST 1 sigma: 1.4826 x MAD, which equals the standard
        # deviation for a Gaussian and ignores the few per cent of events a
        # fitter loses outright. The plain sigma above is kept in the CSV, but
        # it is not what gets drawn -- on these distributions it runs about 12x
        # the median, so bars drawn from it are full-height vertical lines that
        # say nothing and bury the median curve they are attached to.
        mad = float(np.median(np.abs(values - np.median(values))))
        sigma_robust = 1.4826 * mad
        # A strictly positive quantity (a distance, an opening angle) cannot
        # extend below zero, so the lower bar is truncated there rather than
        # drawn into a region the quantity cannot occupy.
        lower = (min(sigma_robust, centre_value)
                 if float(np.min(values)) >= 0.0 else sigma_robust)
        # THE DRAWN BAR follows the summary convention (see
        # metrics.residuals.QUANTITY_KIND). A one-sided quantity is summarised
        # by its 68th percentile alone -- a single number with no bias and no
        # width beside it -- so it gets NO bar. A Gaussian one is summarised by
        # a fitted mu and sigma, so the point is the fitted mu and the bar is
        # its fitted +-1 sigma. Anything else puts a number on the page that
        # does not appear in the table above it.
        # A fitted WIDTH is drawn bare as well: the panel already *is* the
        # spread, and hanging the distribution's own spread on it as an error
        # bar would claim an uncertainty that is not what the bar measures.
        if statistic in ("p68", "gauss_sigma"):
            bar_lo = bar_hi = 0.0
        elif statistic == "gauss_mu":
            _, fit_sigma = stats.gaussian_fit(values)
            bar_lo = bar_hi = float(fit_sigma) if np.isfinite(fit_sigma) else 0.0
        else:
            bar_lo, bar_hi = max(lower, 0.0), max(sigma_robust, 0.0)
        row = {"centre": 0.5 * (low + high), "low": low, "high": high,
               "value": centre_value, "spread": spread_fn(values),
               "p16": p16, "p84": p84,
               "sigma": sigma, "sigma_robust": sigma_robust,
               "err_lo": bar_lo,
               "err_hi": bar_hi,
               "n": int(values.size)}
        if with_ci:
            row["ci_low"], row["ci_high"] = stats.bootstrap_ci(
                values, stat_fn, seed=seed)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["quantity"] = quantity
    out.attrs["versus"] = versus
    out.attrs["statistic"] = statistic
    out.attrs["min_events"] = min_events
    out.attrs["binning"] = binning
    out.attrs["n_events"] = int(np.isfinite(x).sum())
    return out


def profile_all(residuals_by_arm: dict, quantity: str, versus: str,
                **kwargs) -> pd.DataFrame:
    """Profiles for several arms, stacked with an `arm` column."""
    frames = []
    for label, residuals in residuals_by_arm.items():
        table = profile(residuals, quantity, versus, **kwargs)
        if table.empty:
            continue
        table = table.copy()
        table.insert(0, "arm", label)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolution_table(residuals_by_arm: dict, *, versus="p_true", bins=None,
                     quantities=None) -> pd.DataFrame:
    """A tidy table of every arm x quantity x bin, for the CSV output."""
    quantities = quantities or list(QUANTITIES)
    frames = []
    for quantity in quantities:
        table = profile_all(residuals_by_arm, quantity, versus, bins=bins)
        if table.empty:
            continue
        table = table.copy()
        table.insert(1, "quantity", quantity)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def headline_profile_table(residuals_by_arm: dict, versus: str) -> pd.DataFrame:
    """The binned numbers behind a "Resolution against <versus>" figure.

    Every arm x headline panel x bin, with the statistic each panel draws (a
    68th percentile, or a fitted bias and width) and the same time-pedestal
    handling, so the CSV and the figure cannot disagree.
    """
    from . import residuals as residual_metrics

    frames = []
    for label, residuals in residuals_by_arm.items():
        for quantity, statistic, name, unit in residual_metrics.headline_panels():
            if (quantity in residual_metrics.PEDESTAL_QUANTITIES
                    and residual_metrics.needs_pedestal(residuals)):
                residuals = residuals.assign(
                    **{quantity: residuals[quantity] - stats.median(residuals[quantity])})
            table = profile(residuals, quantity, versus, statistic=statistic)
            if table.empty:
                continue
            frames.append(table[["low", "high", "centre", "value", "n"]].assign(
                arm=label, quantity=quantity, statistic=statistic, unit=unit,
                versus=versus, binning=table.attrs.get("binning", "fixed")))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[["arm", "quantity", "statistic", "unit", "versus", "binning",
                "low", "high", "centre", "value", "n"]]
