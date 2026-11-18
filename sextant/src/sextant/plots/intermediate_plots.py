"""Figures for the intermediate quantities -- the algorithm internals.

These are the ones that answer *why* one algorithm beats another, rather than
*whether* it does.
"""

from __future__ import annotations

import numpy as np

from . import style as st


def charge_normalisation(tables_by_arm: dict, arm_style: st.ArmStyle, *,
                         bins: int = 50):
    """Predicted total light over observed, per hypothesis.

    A ratio of 1 means the charge model reproduces the light actually seen. A
    hypothesis sitting systematically below 1 has to explain the missing light
    through the unhit and occupancy terms instead, which is how a constant
    likelihood offset between hypotheses -- a PID bias -- arises.
    """
    hypotheses = sorted(set().union(*[
        set(t["hypothesis"].unique()) for t in tables_by_arm.values()
        if t is not None and not t.empty]) or {"e-"})
    fig, axes = st.grid_figure(len(hypotheses), ncols=min(len(hypotheses), 2),
                              panel_size=(5.6, 3.9))
    for ax, hypothesis in zip(axes, hypotheses):
        edges = np.linspace(0.0, 2.0, bins + 1)
        for arm, table in tables_by_arm.items():
            if table is None or table.empty:
                continue
            values = table[table["hypothesis"] == hypothesis]["ratio"].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            ax.hist(values, bins=edges, histtype="step", density=True,
                    linewidth=1.8, color=arm_style.color(arm),
                    linestyle=arm_style.linestyle(arm),
                    label=f"{arm} — median {np.median(values):.3f}")
        ax.axvline(1.0, color=st.TEXT_SECONDARY, lw=1.2, ls="--")
        st.finish(ax, title=f"{hypothesis} hypothesis",
                  xlabel="predicted PE / observed charge",
                  ylabel="density")
    st.caption(fig, "Charge-model normalisation")
    return fig


def nll_term_breakdown(term_summaries: dict, arm_style: st.ArmStyle):
    """Median size of each likelihood component, grouped by hypothesis.

    Only Coral splits its NLL four ways; fiTQun reports a total. The figure
    shows what is available rather than inventing a split, and the missing
    fiTQun bars are the honest representation of that.
    """
    frames = [t for t in term_summaries.values() if t is not None and not t.empty]
    if not frames:
        return None
    import pandas as pd
    combined = pd.concat(frames, ignore_index=True)
    hypotheses = sorted(combined["hypothesis"].unique())
    fig, axes = st.grid_figure(len(hypotheses), ncols=min(len(hypotheses), 2),
                               panel_size=(5.8, 4.0))
    terms = [t for t in ("unhit", "phit", "cpdf", "tpdf", "total")
             if t in set(combined["term"])]
    for ax, hypothesis in zip(axes, hypotheses):
        subset = combined[combined["hypothesis"] == hypothesis]
        arms = [a for a in arm_style.labels if a in set(subset["arm"])]
        width = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            arm_rows = subset[subset["arm"] == arm].set_index("term")
            values = [float(arm_rows["median"].get(t, np.nan)) for t in terms]
            positions = np.arange(len(terms)) - 0.4 + width * (i + 0.5)
            ax.bar(positions, values, width=width, color=arm_style.color(arm),
                   edgecolor="white", linewidth=2.0, label=arm)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels(terms)
        st.finish(ax, title=f"{hypothesis} hypothesis",
                  xlabel="likelihood component", ylabel="median contribution to NLL")
    st.caption(fig, "Negative log-likelihood decomposition")
    return fig


def nll_improvement(improvement_by_arm: dict, arm_style: st.ArmStyle, *,
                    bins: int = 60):
    """Distribution of nLL(0-ring) - nLL(fit), per hypothesis.

    The cross-code-comparable likelihood quantity: absolute NLLs carry a
    per-code constant, the improvement over each code's own no-ring hypothesis
    does not. It is also the whole of what an official fiTQun file can say
    about its likelihood, which is why this figure exists beside the per-term
    breakdown rather than inside it.
    """
    frames = {a: t for a, t in improvement_by_arm.items()
              if t is not None and not t.empty}
    if not frames:
        return None
    import pandas as pd
    combined = pd.concat(frames.values(), ignore_index=True)
    hypotheses = sorted(combined["hypothesis"].unique())
    fig, axes = st.grid_figure(len(hypotheses), ncols=min(len(hypotheses), 2))
    for ax, hypothesis in zip(axes, hypotheses):
        subset = combined[combined["hypothesis"] == hypothesis]
        values_by_arm = {}
        for arm in arm_style.labels:
            v = subset[subset["arm"] == arm]["improvement"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                values_by_arm[arm] = v
        if not values_by_arm:
            continue
        # One shared binning across arms, framed on the bulk: these
        # distributions have a long tail of very bright events, and binning to
        # the maximum puts every arm in the first bin.
        allv = np.concatenate(list(values_by_arm.values()))
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(allv.min()), float(allv.max()) or 1.0
        edges = np.linspace(lo, hi, bins + 1)
        for arm, v in values_by_arm.items():
            ax.hist(v, bins=edges, histtype="step", density=True, linewidth=1.8,
                    color=arm_style.color(arm),
                    linestyle=arm_style.linestyle(arm),
                    label=f"{arm} — median {np.median(v):,.0f}")
        ax.axvline(0.0, color=st.TEXT_SECONDARY, lw=1.0, ls="--")
        # Always legend, even for a single arm: the median in the label is the
        # number a reader takes off this figure, and the default hides a
        # one-entry legend.
        st.finish(ax, title=f"{hypothesis} hypothesis",
                  xlabel="nLL(0-ring) − nLL(fit)", ylabel="density",
                  legend=True)
    st.caption(fig, "Likelihood improvement over zero-ring hypothesis")
    return fig


def hypothesis_term_difference(difference_by_arm: dict, arm_style: st.ArmStyle):
    """Per-term μ-minus-e difference: which component drives the PID."""
    frames = {a: t for a, t in difference_by_arm.items()
              if t is not None and not t.empty}
    if not frames:
        return None
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(7.4, 4.2))
    ax = axes[0]
    terms = list(next(iter(frames.values()))["term"])
    width = 0.8 / len(frames)
    for i, (arm, table) in enumerate(frames.items()):
        rows = table.set_index("term")
        medians = [float(rows["median_delta"].get(t, np.nan)) for t in terms]
        spreads = [float(rows["iqr68_delta"].get(t, np.nan)) for t in terms]
        positions = np.arange(len(terms)) - 0.4 + width * (i + 0.5)
        ax.bar(positions, medians, width=width, yerr=spreads, capsize=3,
               color=arm_style.color(arm), edgecolor="white", linewidth=2.0,
               error_kw={"elinewidth": 1.0, "ecolor": st.TEXT_SECONDARY},
               label=arm)
    ax.axhline(0.0, color=st.TEXT_SECONDARY, lw=1.0, ls="--")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms)
    st.finish(ax, title="μ − e by likelihood term",
              xlabel="likelihood component",
              ylabel="nLL(μ) − nLL(e) contribution")
    fig.tight_layout()
    return fig


def stage_trajectory(trajectories: dict, arm_style: st.ArmStyle):
    """How each fit's estimate improves stage by stage.

    Stage numbers are each code's *own* sequence and are not equivalent between
    codes, so this compares shapes -- where accuracy is won and whether a stage
    ever makes things worse -- not stage 1 against stage 1.
    """
    frames = {a: t for a, t in trajectories.items()
              if t is not None and not t.empty}
    if not frames:
        return None
    panels = [("median_vertex_dist", "vertex distance", "cm"),
              ("median_angle_deg", "angular error", "deg"),
              ("median_p_res_pct", "momentum bias", "%")]
    fig, axes = st.grid_figure(len(panels), ncols=len(panels),
                               panel_size=(4.8, 3.6))
    for ax, (column, title, unit) in zip(axes, panels):
        for arm, table in frames.items():
            ax.plot(table["stage"], table[column],
                    **arm_style.point_kwargs(arm))
        if column == "median_p_res_pct":
            ax.axhline(0.0, color=st.TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_xticks(sorted(set().union(*[set(t["stage"]) for t in frames.values()])))
        st.finish(ax, title=title, xlabel="fit stage",
                  ylabel=f"median {title} [{unit}]",
                  legend=(column == panels[0][0]))
    st.caption(fig, "Fit-stage trajectory")
    return fig


def pmt_prediction_comparison(comparison, labels=("a", "b"), *, bins: int = 60):
    """Per-PMT predicted charge of one code against the other, at a common point.

    A scatter of two magnitudes, so a 2-D density rather than overplotted points
    -- with tens of thousands of PMTs per event, a scatter is a solid block that
    hides exactly the structure being looked for. The diagonal is agreement.
    """
    if comparison is None or comparison.empty:
        return None
    ca, cb = f"mu_direct_{labels[0]}", f"mu_direct_{labels[1]}"
    if ca not in comparison or cb not in comparison:
        return None
    a = comparison[ca].to_numpy(float)
    b = comparison[cb].to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() == 0:
        return None
    a, b = a[mask], b[mask]

    fig, axes = st.grid_figure(2, ncols=2, panel_size=(5.2, 4.4))
    ax = axes[0]
    edges = np.logspace(np.log10(max(min(a.min(), b.min()), 1e-4)),
                        np.log10(max(a.max(), b.max())), bins)
    ax.hist2d(a, b, bins=[edges, edges], cmap="Blues", cmin=1)
    limits = (edges[0], edges[-1])
    ax.plot(limits, limits, color=st.TEXT_SECONDARY, lw=1.2, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(False)
    st.finish(ax, title="Per-PMT predicted charge",
              xlabel=f"predicted PE — {labels[0]}",
              ylabel=f"predicted PE — {labels[1]}", legend=False)

    ax2 = axes[1]
    ratio = a / b
    ax2.hist(ratio, bins=np.logspace(-1, 1, bins), histtype="step",
             linewidth=1.8, color=st.PALETTE[0])
    ax2.axvline(1.0, color=st.TEXT_SECONDARY, lw=1.2, ls="--")
    ax2.set_xscale("log")
    st.finish(ax2, title=f"Ratio (median {np.median(ratio):.3f})",
              xlabel=f"predicted PE  {labels[0]} / {labels[1]}",
              ylabel="PMTs", legend=False)
    point = comparison["point"].iloc[0] if "point" in comparison else "?"
    st.caption(fig, f"Per-PMT charge model — point '{point}'")
    return fig
