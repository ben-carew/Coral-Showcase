"""Residual and resolution figures."""

from __future__ import annotations

import numpy as np

from ..metrics import resolution as resolution_metrics
from ..metrics import stats as stats_metrics
from ..metrics.residuals import PARAMETER_UNITS
from . import style as st

#: Quantiles used to set a shared x-range across arms. Fixed limits would clip
#: whichever arm has the wider tail; per-arm limits would make the histograms
#: incomparable.
#: Display range for a two-sided residual. NOT the old (0.005, 0.995): these
#: distributions have very heavy tails -- on 500 v1.12.29 electrons the vertex
#: displacement has a median of 21.6 cm and a 99.5th percentile of 1183 cm -- so
#: a 99.5% range compresses the entire bulk, and the comparison the reader is
#: making, into two pixels. Clipping is honest as long as it is declared, so
#: every panel annotates the fraction that falls outside its axis.
RANGE_QUANTILES = (0.02, 0.98)

#: One-sided quantities (a distance, an opening angle) start at zero and have
#: only an upper tail, so they get a tighter upper cut and no lower one.
ONE_SIDED_QUANTILES = (0.0, 0.95)
ONE_SIDED = ("vertex_dist", "vertex_trans", "angle_deg", "angle")

#: On trigger-clock truth the time residual is ~1 ns wide, but a few per cent of
#: muon events sit tens to thousands of ns out (a muon too dim to trigger, so
#: trigger 0 is its Michel electron). A 2-98% window would spend the whole axis
#: on them; this one shows the core and the tail is declared as "beyond axis".
QUANTITY_QUANTILES = {"t": (0.05, 0.95)}

#: A paired difference inherits the tails of BOTH arms, and on this sample the
#: tail is a cliff rather than a slope: the pooled vertex difference has a 90th
#: percentile of 20 cm and a 95th of 107 cm, because the last few per cent are
#: the events one code fails outright and the other does not. A percentile
#: window either lands before the cliff or swallows it whole.
#:
#: So the window is symmetric about zero at the 10-90% robust half-width. A
#: difference plot is read by asking which side of zero the mass sits on, and a
#: symmetric axis is what makes that answerable at a glance; the fraction
#: outside is annotated, as everywhere else.
PAIRED_QUANTILES = (0.10, 0.90)


def _symmetric_range(series_list, quantiles=PAIRED_QUANTILES):
    lo, hi = _shared_range(series_list, quantiles)
    half = max(abs(lo), abs(hi))
    return -half, half

PRETTY = {
    "vertex_dist": "vertex displacement",
    "vertex_long": "vertex displacement, along the track",
    "vertex_trans": "vertex displacement, transverse to the track",
    "angle_deg": "angular displacement",
    "p_res_pct": "momentum resolution", "p_res": "momentum residual",
    "t": "t residual", "x": "x residual", "y": "y residual",
    "z": "z residual", "p": "p residual",
    "theta": "θ residual", "phi": "φ residual",
}


def _pretty(key: str) -> str:
    return PRETTY.get(key, key)


def _outside_fraction(series_list, lo, hi):
    """Fraction of all plotted values that fall outside the display range."""
    arrays = [np.asarray(s, float) for s in series_list if len(s)]
    pooled = np.concatenate(arrays) if arrays else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 0.0
    return float(((pooled < lo) | (pooled > hi)).mean())


def _shared_range(series_list, quantiles=RANGE_QUANTILES):
    arrays = [np.asarray(s, float) for s in series_list if len(s)]
    pooled = np.concatenate(arrays) if arrays else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(pooled, quantiles)
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)


def residual_histograms(residuals_by_arm: dict, arm_style: st.ArmStyle, *,
                        quantities=None, bins: int = 60, density: bool = True,
                        ncols: int = 4, title: str = None):
    """One panel per residual quantity, all arms overlaid.

    Density rather than counts by default: arms routinely differ in sample size
    by two orders of magnitude (95k fiTQun events against 1k Coral ones), and
    raw counts make the smaller sample invisible.

    The report calls this twice -- once for the seven per-parameter residuals
    and once for the three aggregate measures -- because they answer different
    questions and mixing them on one page invites reading a coordinate
    component as if it were a performance number.
    """
    from ..metrics.residuals import COMPONENT_QUANTITIES

    if quantities is None:
        quantities = COMPONENT_QUANTITIES
    quantities = [q for q in quantities
                  if any(q in r.columns for r in residuals_by_arm.values())]
    if not quantities:
        return None
    fig, axes = st.grid_figure(len(quantities), ncols=ncols)

    for ax, quantity in zip(axes, quantities):
        series = [r[quantity].to_numpy(float) for r in residuals_by_arm.values()
                  if quantity in r.columns]
        quantiles = QUANTITY_QUANTILES.get(
            quantity, ONE_SIDED_QUANTILES if quantity in ONE_SIDED else RANGE_QUANTILES)
        lo, hi = _shared_range(series, quantiles)
        outside = _outside_fraction(series, lo, hi)
        edges = np.linspace(lo, hi, bins + 1)
        for label, residuals in residuals_by_arm.items():
            if quantity not in residuals.columns:
                continue
            values = residuals[quantity].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                # density=True on an empty series divides by a zero sum. A
                # quantity can legitimately be empty (a momentum residual with
                # every event below the p_min cut, say), so skip it rather than
                # emit a numpy warning and an invisible line.
                continue
            ax.hist(values, bins=edges, histtype="step", density=density,
                    linewidth=1.8, color=arm_style.color(label),
                    linestyle=arm_style.linestyle(label), label=label)
        if quantity not in ONE_SIDED:
            ax.axvline(0.0, color=st.TEXT_SECONDARY, lw=0.8, ls="--", alpha=0.7)
        if outside > 0.001:
            # Declared, not hidden: a clipped axis that does not say so is a lie
            # about the tail, and the tail is where the failures live.
            ax.text(0.98, 0.03, f"{outside:.1%} beyond axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=st.TEXT_SECONDARY)
        unit = PARAMETER_UNITS.get(quantity, "")
        st.finish(ax, title=_pretty(quantity),
                  xlabel=_pretty(quantity) + (f" [{unit}]" if unit else ""),
                  ylabel="density" if density else "events", legend=False)

    # one legend for the grid, under the title: arm labels are long and a legend
    # inside the first panel spilled over it and its axis label
    top = 0.96
    if axes:
        handles, labels = axes[0].get_legend_handles_labels()
        if len(labels) > 1:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 2),
                       bbox_to_anchor=(0.5, 0.955), frameon=False)
            top = 0.955 - 0.025 * ((len(labels) + 1) // 2)
    fig.suptitle(st.wrap_caption(
                     title or "Reconstruction residuals (reconstructed − true)"),
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def resolution_profiles(residuals_by_arm: dict, arm_style: st.ArmStyle, *,
                        versus="p_true", quantities=None, bins=None,
                        show_spread: bool = True, bound=None):
    """Median of each quantity against a binning variable, one panel each.

    `bound`, if given, is the residual table of an efficient estimator
    (`crb.compare.efficient_residuals`). On every WIDTH panel it is profiled on the
    same bins as the arms and drawn as a shaded floor: the region below it is out of
    reach of any unbiased reconstruction. Bias panels get nothing -- an efficient
    estimator is unbiased by construction -- and neither does time, whose residual
    here measures the trigger-time jitter rather than a resolution (see
    `BOUND_EXCLUDED`).

    The error bars are the *spread* of the distribution in the bin (iqr68), not
    the uncertainty on the median -- so they say how variable the performance
    is, which is what a resolution plot is for.
    """
    from ..metrics import residuals as residuals_metrics

    # The nine headline numbers, profiled: a 68th percentile for each one-sided
    # magnitude and a fitted bias AND width for each signed residual. Reading
    # the same definition the headline table does keeps the binned page and the
    # summary consistent by construction.
    panels = list(quantities or residuals_metrics.headline_panels())
    # A 3x3 grid, not a 1xN strip: nine panels in one row squeezed every title
    # off the page edge.
    fig, axes = st.grid_figure(len(panels), ncols=3, panel_size=(5.0, 3.6))
    label_x, unit_x, _default = resolution_metrics.BINNINGS.get(
        versus, (versus, "", np.linspace(0, 1, 11)))

    adapted = False
    empty_arms = []
    for ax, panel in zip(axes, panels):
        quantity, statistic, name, unit = panel
        plotted = 0
        first_edges = None
        centres, bar_lo, bar_hi = [], [], []
        for label, residuals in residuals_by_arm.items():
            # `bins` is passed straight through -- None when the caller did
            # not choose edges, which is what lets `profile` fall back to
            # equal-count bins. Resolving the default here instead defeated
            # that fallback entirely and left the panel blank.
            # The time residual carries a convention pedestal both codes
            # share (~-39 ns against the truth track time). Its BIAS panel must
            # describe the residual, not the convention, so the pedestal is
            # removed first -- exactly as the summary table does it.
            if (quantity in residuals_metrics.PEDESTAL_QUANTITIES
                    and residuals_metrics.needs_pedestal(residuals)):
                residuals = residuals.assign(
                    **{quantity: residuals[quantity]
                       - stats_metrics.median(residuals[quantity])})
            profile = resolution_metrics.profile(
                residuals, quantity, versus, bins=bins, statistic=statistic)
            if profile.empty:
                empty_arms.append(label)
                continue
            plotted += 1
            if first_edges is None:
                first_edges = np.r_[profile["low"].to_numpy(float),
                                    profile["high"].to_numpy(float)[-1:]]
            if profile.attrs.get("binning") == "quantile":
                adapted = True
            # The view is scaled from the bars that are ACTUALLY DRAWN. It used
            # to be scaled from the percentile half-span while the bars had
            # become the smaller 1.4826 x MAD, so every panel was framed wider
            # than anything in it.
            _v = profile["value"].to_numpy(float)
            centres.extend(_v.tolist())
            bar_lo.extend((_v - profile["err_lo"].to_numpy(float)).tolist())
            bar_hi.extend((_v + profile["err_hi"].to_numpy(float)).tolist())
            kwargs = arm_style.point_kwargs(label)
            # A one-sided quantity carries no bar at all (its summary is one
            # number, the 68th percentile); a Gaussian one carries its fitted
            # +-1 sigma. resolution.profile decides which, so that the bars and
            # the summary table cannot drift apart.
            yerr = np.vstack([profile["err_lo"].to_numpy(float),
                              profile["err_hi"].to_numpy(float)])
            if show_spread and float(np.nanmax(yerr)) > 0.0:
                ax.errorbar(profile["centre"], profile["value"], yerr=yerr,
                            capsize=2.5, elinewidth=1.0, markersize=4, **kwargs)
            else:
                ax.plot(profile["centre"], profile["value"], marker="o",
                        markersize=4, **{k: v for k, v in kwargs.items()
                                         if k != "marker"})
        if bound is not None and _bound_applies(quantity, statistic) and first_edges is not None:
            drawn = draw_bound(ax, bound, quantity, versus, statistic, first_edges)
            centres.extend(drawn)
            bar_lo.extend(drawn)
            bar_hi.extend(drawn)
        if statistic == "gauss_mu":
            ax.axhline(0.0, color=st.TEXT_SECONDARY, lw=0.8, ls="--", alpha=0.7)
        # Counted explicitly: `ax.has_data()` is true as soon as the zero
        # reference line is drawn, so the momentum panel never got the
        # explanation the other two did.
        if plotted == 0:
            # An empty panel with no explanation reads as a bug. Say why.
            n = min((len(r) for r in residuals_by_arm.values()), default=0)
            ax.text(0.5, 0.5, f"too few events ({n})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color=st.TEXT_SECONDARY)
        # Frame the drawn bars tightly. Matplotlib's autoscale reserves room
        # for the raw data range, which on these heavy-tailed quantities is set
        # by the few per cent of events a fitter loses outright and leaves the
        # median curves -- the actual content -- as a flat line near the axis.
        if show_spread and bar_lo:
            # Frame EVERY drawn bar end to end, with only a little air. Nothing
            # is clipped -- an error bar you cannot see the end of is worse than
            # a loose axis -- but matplotlib's own autoscale is not used either,
            # because it reserves room for the raw data range, which on these
            # heavy-tailed quantities is set by the few per cent of events a
            # fitter loses outright and leaves the median curves flat on the
            # axis. The tightest window that still shows everything is the ends
            # of the longest bars.
            # Frame the BULK of the drawn bars, not their extremes. One
            # near-wall bin of a failing configuration reaches +800% momentum
            # bias, and scaling to it flattens every other point in the panel
            # onto the axis -- the panel then shows one outlier and nothing
            # else. The window is the 5-95% range of the drawn ends, and
            # anything outside it is COUNTED ON THE PANEL rather than hidden.
            ends = np.array(bar_lo + bar_hi, dtype=float)
            ends = ends[np.isfinite(ends)]
            lo, hi = (float(np.percentile(ends, 5)),
                      float(np.percentile(ends, 95))) if ends.size else (0.0, 1.0)
            if hi <= lo:
                lo, hi = float(np.min(ends)), float(np.max(ends))
            span = hi - lo
            pad = 0.08 * span if span > 0 else max(abs(hi), 1.0) * 0.1
            lo, hi = lo - pad, hi + pad
            if all(v >= 0 for v in centres):
                lo = max(lo, 0.0)          # a distance cannot go below zero
            ax.set_ylim(lo, hi)
            off = int(np.sum((np.array(centres) < lo) | (np.array(centres) > hi)))
            if off:
                ax.text(0.98, 0.95, f"{off} beyond axis",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color=st.TEXT_SECONDARY)
        # The axis says which statistic is plotted, so a reader cannot mistake a
        # 68th percentile for a median or a fitted mean for either.
        stat_label = {"p68": "68th percentile", "gauss_mu": "bias",
                      "gauss_sigma": "1σ width",
                      "median": "median"}.get(statistic, statistic)
        # Short titles: "vs <binning variable>" was the same on all nine panels
        # and pushed the rightmost one off the sheet. The binning variable is
        # named once, on every x axis and in the caption.
        st.finish(ax, title=f"{name} — {stat_label}",
                  xlabel=label_x + (f" [{unit_x}]" if unit_x else ""),
                  ylabel=f"{stat_label} [{unit}]",
                  legend=False)

    subtitle = f"Resolution against {label_x}"
    if adapted:
        subtitle += "  (equal-count bins)"
    fig.suptitle(subtitle, y=0.98, fontsize=11)
    bottom = _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    return fig


#: Residuals the bound is NOT drawn against. The fitters report t relative to the
#: detector trigger and the particle-gun truth starts every particle at t = 0, so
#: the time residual carries the event-to-event trigger offset (iqr68 ~30 ns) --
#: it measures the trigger, not the reconstruction, and a 0.1 ns bound drawn under
#: it would be meaningless. Fixing that needs the trigger time in the truth file.
BOUND_EXCLUDED = ("t",)
BOUND_COLOUR = "#6b6a66"
BOUND_LABEL = "Cramér–Rao bound"


def _figure_legend(fig, axes):
    """One legend for the whole grid, below it; returns the height it takes.

    A legend inside the first panel covered its data as soon as a comparison had
    more than three runs with realistic labels (five runs + the bound did).
    """
    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    if not handles:
        return 0.0
    ncol = 1 if max(len(l) for l in labels) > 45 else 2
    rows = -(-len(labels) // ncol)
    height = min(0.03 + 0.024 * rows, 0.22)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    return height


def _bound_applies(quantity, statistic):
    return statistic in ("p68", "gauss_sigma") and quantity not in BOUND_EXCLUDED


def bound_profile(bound, quantity, versus, statistic, edges):
    """The bound's statistic on exactly the arms' bins (every bin kept)."""
    return resolution_metrics.profile(bound, quantity, versus, bins=edges,
                                      statistic=statistic, adaptive=False,
                                      min_events=1)


def draw_bound(ax, bound, quantity, versus, statistic, edges):
    """Shade the unreachable region under the bound; return the values drawn."""
    prof = bound_profile(bound, quantity, versus, statistic, edges)
    if prof.empty:
        return []
    lo = prof["low"].to_numpy(float)
    hi = prof["high"].to_numpy(float)
    val = prof["value"].to_numpy(float)
    xs = np.ravel(np.column_stack([lo, hi]))
    ys = np.repeat(val, 2)
    ax.fill_between(xs, 0.0, ys, color=BOUND_COLOUR, alpha=0.16, lw=0,
                    label=BOUND_LABEL, zorder=0)
    ax.plot(xs, ys, color=BOUND_COLOUR, lw=1.2, zorder=1)
    return val.tolist()


def bound_ratio_profiles(residuals_by_arm: dict, bound, arm_style: st.ArmStyle, *,
                         versus="p_true", bins=None):
    """How far each arm sits above the Cramér–Rao bound, bin by bin.

    One panel per width metric; the y axis is (arm statistic) / (bound statistic),
    on a log scale, so 1 is an efficient reconstruction and 10 means ten times the
    best any unbiased algorithm could do with this detector. Bars are 68% bootstrap
    intervals on the arm's statistic.
    """
    from ..metrics import residuals as residuals_metrics

    panels = [p for p in residuals_metrics.headline_panels()
              if _bound_applies(p[0], p[1])]
    fig, axes = st.grid_figure(len(panels), ncols=3)
    label_x, unit_x, _ = resolution_metrics.BINNINGS.get(
        versus, (versus, "", np.linspace(0, 1, 11)))
    for ax, (quantity, statistic, name, unit) in zip(axes, panels):
        drawn = []
        for label, residuals in residuals_by_arm.items():
            prof = resolution_metrics.profile(residuals, quantity, versus,
                                              bins=bins, statistic=statistic)
            if prof.empty:
                continue
            arm_edges = np.r_[prof["low"].to_numpy(float), prof["high"].to_numpy(float)[-1:]]
            ref = bound_profile(bound, quantity, versus, statistic, arm_edges)
            # Aligned on the bin, not the position: a bin the bound cannot fill
            # (muons below the model's lowest momentum) is simply absent from it.
            floor = (ref.set_index("low")["value"].reindex(prof["low"].to_numpy())
                     .to_numpy(float) if not ref.empty else np.full(len(prof), np.nan))
            ratio = prof["value"].to_numpy(float) / floor
            # A ratio from 20 events is not a measurement without its interval: a
            # bare point below 1 reads as a violated bound when it is a 1.5 sigma
            # wobble. 68% bootstrap interval on the ARM's statistic.
            fn = stats_metrics.STATISTICS[statistic]
            x = residuals[versus].to_numpy(float)
            y = residuals[quantity].to_numpy(float)
            ci = []
            for low, high in zip(prof["low"], prof["high"]):
                sel = (x >= low) & (x < high) & np.isfinite(y)
                ci.append(stats_metrics.bootstrap_ci(y[sel], fn, n_resamples=200,
                                                     confidence=68.0))
            ci = np.array(ci, float).reshape(-1, 2) / floor[:, None]
            yerr = np.vstack([np.clip(ratio - ci[:, 0], 0, None),
                              np.clip(ci[:, 1] - ratio, 0, None)])
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            kwargs = arm_style.point_kwargs(label)
            ax.errorbar(prof["centre"], ratio, yerr=yerr, capsize=2.0, elinewidth=0.9,
                        markersize=4, **kwargs)
            drawn.extend(ratio[np.isfinite(ratio)].tolist())
        ax.axhline(1.0, color=BOUND_COLOUR, lw=1.2)
        ax.axhspan(1e-3, 1.0, color=BOUND_COLOUR, alpha=0.16, lw=0, label=BOUND_LABEL)
        ax.set_yscale("log")
        # Frame every arm, not whichever was drawn first; a failing arm's outliers
        # are COUNTED on the panel rather than allowed to flatten everything else.
        # The bottom always shows the bound, and never hides a point below it.
        vals = np.array([v for v in drawn if v > 0])
        if vals.size:
            lo = min(0.8, 0.8 * float(vals.min()))
            hi = max(2.0, 1.3 * float(np.percentile(vals, 95)))
            ax.set_ylim(lo, hi)
            off = int(np.sum(vals > hi))
            if off:
                ax.text(0.98, 0.95, f"{off} beyond axis", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7, color=st.TEXT_SECONDARY)
        stat_label = {"p68": "68th percentile", "gauss_sigma": "1σ width"}[statistic]
        st.finish(ax, title=f"{name} — {stat_label}",
                  xlabel=label_x + (f" [{unit_x}]" if unit_x else ""),
                  ylabel="arm / bound", legend=False)
    fig.suptitle(f"Distance from the Cramér–Rao bound against {label_x}",
                 y=0.98, fontsize=11)
    bottom = _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    return fig


def paired_difference_plot(residuals_by_arm: dict, arm_style: st.ArmStyle,
                           reference: str, *,
                           quantities=("vertex_dist", "vertex_long",
                                       "vertex_trans", "angle_deg", "p_res_pct")):
    """Per-event difference of each arm against a reference arm.

    Pairing on the event removes the event-to-event variation the arms share, so
    a difference far smaller than either arm's own spread becomes visible. Only
    events reconstructed by both arms take part, and the count says how many.
    """
    from ..metrics.residuals import match_events

    others = [label for label in residuals_by_arm if label != reference]
    if not others or reference not in residuals_by_arm:
        return None
    matched = match_events(residuals_by_arm)
    ref = matched[reference]
    quantities = list(quantities)
    fig, axes = st.grid_figure(len(quantities), ncols=len(quantities),
                               panel_size=(4.8, 3.4))

    for ax, quantity in zip(axes, quantities):
        deltas = {}
        for label in others:
            other = matched[label]
            if quantity not in ref.columns or quantity not in other.columns:
                continue
            delta = (other[quantity].to_numpy(float)
                     - ref[quantity].to_numpy(float))
            delta = delta[np.isfinite(delta)]
            if delta.size == 0:
                continue
            deltas[label] = delta
        if not deltas:
            continue
        # One range shared by every arm on the panel, and a TIGHT one: a paired
        # difference inherits both arms' tails, so the 2-98% window used for a
        # single residual still leaves the central mass -- the whole point of
        # pairing -- squeezed into a couple of bins.
        lo, hi = _symmetric_range(list(deltas.values()))
        outside = _outside_fraction(list(deltas.values()), lo, hi)
        for label, delta in deltas.items():
            ax.hist(delta, bins=np.linspace(lo, hi, 51), histtype="step",
                    linewidth=1.8, color=arm_style.color(label),
                    linestyle=arm_style.linestyle(label),
                    label=f"{label} − {reference}")
            ax.axvline(float(np.median(delta)), color=arm_style.color(label),
                       lw=1.0, ls=":", alpha=0.9)
        ax.axvline(0.0, color=st.TEXT_SECONDARY, lw=0.9, ls="--")
        if outside > 0.001:
            ax.text(0.98, 0.03, f"{outside:.1%} beyond axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=st.TEXT_SECONDARY)
        unit = PARAMETER_UNITS.get(quantity, "")
        st.finish(ax, title=f"Δ {_pretty(quantity)}",
                  xlabel=f"difference [{unit}]" if unit else "difference",
                  ylabel="events", legend=(quantity == quantities[0]))
    # n goes in the suptitle, not on the axes: annotate_count writes to the
    # upper right, which is where the legend sits on these panels, and the two
    # were printing on top of each other.
    # y=1.02 put the title ABOVE the figure. The standalone PNG is saved with a
    # tight bbox so it survived there, but the PDF page is not cropped, so on
    # the page itself the title was simply off the top and the sheet had none.
    fig.suptitle(f"Per-event difference vs {reference} — {len(ref):,} matched",
                 y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def timing_histogram(arms, arm_style: st.ArmStyle, *, bins: int = 40):
    """Per-event reconstruction time.

    Never compare a laptop time against a cluster time -- different hardware.
    The report says which machine each arm ran on for exactly this reason.
    """
    data = [(arm.label, arm.timing["time_s"].to_numpy(float))
            for arm in arms if not arm.timing.empty]
    data = [(label, t[np.isfinite(t)]) for label, t in data]
    data = [(label, t) for label, t in data if t.size]
    if not data:
        return None
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(7.0, 4.0))
    ax = axes[0]
    pooled = np.concatenate([t for _, t in data])
    # Coral sits at ~5 s/event and fiTQun at ~128, so a linear axis spends 95%
    # of its width on empty space and renders Coral as a single spike. Log bins
    # when the arms span more than a decade, which is exactly the comparison
    # the figure exists to make.
    positive = pooled[pooled > 0]
    span = (positive.max() / positive.min()) if positive.size else 1.0
    if span > 10.0 and positive.size:
        lo = float(np.percentile(positive, 0.5))
        hi = float(np.percentile(positive, 99.5))
        edges = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), bins + 1)
        ax.set_xscale("log")
    else:
        edges = np.linspace(0, float(np.percentile(pooled, 99)), bins + 1)
    for label, values in data:
        ax.hist(values, bins=edges, histtype="step", linewidth=1.8,
                color=arm_style.color(label),
                linestyle=arm_style.linestyle(label),
                label=f"{label} — median {np.median(values):.2f} s/ev")
        ax.axvline(float(np.median(values)), color=arm_style.color(label),
                   lw=1.0, ls=":", alpha=0.8)
    st.finish(ax, title="Reconstruction time per event",
              xlabel="time per event [s]", ylabel="events")
    fig.tight_layout()
    return fig
