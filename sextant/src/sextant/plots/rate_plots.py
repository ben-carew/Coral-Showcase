"""Figures for rates -- efficiencies and purities -- with Wilson intervals.

One module for every binomial quantity the report shows (sub-eventing, PID
efficiency, multi-track PID), so they all read the same way: the rate in per
cent, its 68% interval as the error bar, one colour per arm.
"""

from __future__ import annotations

import numpy as np

from ..metrics import resolution as resolution_metrics
from . import style as st
from .residual_plots import _figure_legend

#: Axis label for each binning variable, as the rate tables name them.
VARIABLE_LABELS = {
    "p": ("true momentum", "MeV/c"),
    "p_true": ("true momentum", "MeV/c"),
    "dwall": resolution_metrics.BINNINGS["dwall"][:2],
    "towall": resolution_metrics.BINNINGS["towall"][:2],
}


def _axis_label(versus):
    name, unit = VARIABLE_LABELS.get(versus, (versus, ""))
    return name + (f" [{unit}]" if unit else "")


def _bars(values, low, high):
    """Error-bar lengths from an interval. Clipped at zero: at k = n the Wilson
    bound lands a rounding error below the rate itself."""
    return np.clip(np.vstack([values - np.asarray(low, float),
                              np.asarray(high, float) - values]), 0.0, None)


def summary_bars(summary, arm_style: st.ArmStyle, *, metrics=None, title=""):
    """Horizontal bars, one group per metric, one bar per arm, 68% intervals."""
    if summary is None or summary.empty:
        return None
    metrics = list(metrics or summary["metric"].drop_duplicates())
    arms = list(summary["arm"].drop_duplicates())
    fig, axes = st.grid_figure(1, ncols=1)
    ax = axes[0]
    height = 0.8 / max(len(arms), 1)
    lowest = 100.0
    for i, arm in enumerate(arms):
        rows = summary[summary["arm"] == arm].set_index("metric").reindex(metrics)
        y = np.arange(len(metrics)) + (i - (len(arms) - 1) / 2) * height
        values = rows["value"].to_numpy(float)
        err = _bars(values, rows["ci_lo"], rows["ci_hi"])
        ax.barh(y, values, height=height * 0.9, xerr=err, capsize=2,
                color=arm_style.color(arm), label=arm, alpha=0.85)
        finite = values[np.isfinite(values)]
        if finite.size:
            lowest = min(lowest, float(np.nanmin(rows["ci_lo"].to_numpy(float))))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    # Rates here sit in the 80-100% band; starting the axis at 0 would squash
    # every difference that matters into the last few millimetres.
    ax.set_xlim(max(0.0, np.floor(lowest / 5.0) * 5.0 - 5.0), 100.5)
    st.finish(ax, title=title, xlabel="rate [%]", legend=False)
    _below_grid_legend(fig, axes)
    return fig


def _below_grid_legend(fig, axes, top: float = 1.0):
    """One legend under the whole figure, with the layout making room for it."""
    bottom = _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, bottom, 1, top))


def rate_profiles(profiles, arm_style: st.ArmStyle, *, row_key: str,
                  rows=None, versus=("p", "dwall", "towall"), title=""):
    """A grid: one row per `row_key` value, one column per binning variable."""
    if profiles is None or profiles.empty:
        return None
    rows = [r for r in (rows or profiles[row_key].drop_duplicates())
            if (profiles[row_key] == r).any()]
    versus = [v for v in versus if (profiles["versus"] == v).any()]
    if not rows or not versus:
        return None
    fig, axes = st.grid_figure(len(rows) * len(versus), ncols=len(versus))
    arms = list(profiles["arm"].drop_duplicates())
    for index, (row, var) in enumerate((r, v) for r in rows for v in versus):
        ax = axes[index]
        sel = profiles[(profiles[row_key] == row) & (profiles["versus"] == var)]
        for arm in arms:
            prof = sel[sel["arm"] == arm]
            if prof.empty:
                continue
            values = prof["value"].to_numpy(float)
            yerr = _bars(values, prof["ci_lo"], prof["ci_hi"])
            ax.errorbar(prof["centre"], values, yerr=yerr, capsize=2,
                        markersize=4, elinewidth=1.0,
                        **arm_style.point_kwargs(arm))
        st.finish(ax, title=str(row), xlabel=_axis_label(var),
                  ylabel="rate [%]", legend=False)
    if title:
        fig.suptitle(title, y=0.98, fontsize=11)
    _below_grid_legend(fig, axes, top=0.95 if title else 1.0)
    return fig


def time_offsets(tables: dict, arm_style: st.ArmStyle, *, window_ns: float):
    """|t_reco - t_true| of matched sub-events, per class, with the window."""
    if not tables:
        return None
    kinds = ("prompt", "decay")
    fig, axes = st.grid_figure(len(kinds), ncols=2)
    edges = np.linspace(-window_ns, window_ns, 51)
    for ax, kind in zip(axes, kinds):
        for arm, (per_true, _) in tables.items():
            dt = per_true.loc[(per_true["kind"] == kind) & per_true["matched"],
                              "dt"].to_numpy(float)
            if dt.size:
                ax.hist(dt, bins=edges, histtype="step", linewidth=1.6,
                        **arm_style.line_kwargs(arm))
        st.finish(ax, title=f"{kind} sub-events",
                  xlabel="t reco − t true [ns]", ylabel="sub-events",
                  legend=False)
    fig.suptitle("Time offset of matched sub-events", y=0.98, fontsize=11)
    _below_grid_legend(fig, axes, top=0.95)
    return fig
