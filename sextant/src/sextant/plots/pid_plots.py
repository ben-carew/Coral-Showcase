"""Particle-identification figures."""

from __future__ import annotations

import numpy as np

from ..metrics import pid as pid_metrics
from . import style as st


def discriminant_distributions(pid_by_arm_and_sample: dict, arm_style: st.ArmStyle,
                               *, bins: int = 60):
    """dnLL for true electrons and true muons, one panel per arm.

    Both true samples on one axis per arm, because the separation *between* the
    two is the quantity of interest -- plotting them on separate axes makes the
    overlap, which is exactly what PID performance is, impossible to see.
    """
    arms = list(pid_by_arm_and_sample)
    if not arms:
        return None
    fig, axes = st.grid_figure(len(arms), ncols=min(len(arms), 3),
                               panel_size=(5.2, 3.8))
    for ax, arm in zip(axes, arms):
        samples = pid_by_arm_and_sample[arm]
        series = [s["dnll"].to_numpy(float) for s in samples.values() if len(s)]
        pooled = np.concatenate(series) if series else np.array([])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            ax.axis("off")
            continue
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        # Two fixed roles, not the arm palette: within a panel the identity that
        # matters is which true particle it is, and that must read the same way
        # in every panel.
        colours = {"e-": st.PALETTE[0], "mu-": st.PALETTE[1]}
        for true_particle, table in samples.items():
            values = table["dnll"].to_numpy(float)
            ax.hist(values, bins=edges, histtype="step", linewidth=1.8,
                    density=True, color=colours.get(true_particle, st.PALETTE[2]),
                    label=f"true {true_particle} (n={np.isfinite(values).sum():,})")
        ax.axvline(0.0, color=st.TEXT_SECONDARY, lw=1.0, ls="--")
        st.finish(ax, title=arm,
                  xlabel="dnLL = nLL(μ) − nLL(e)      →  electron-like",
                  ylabel="density")
    st.caption(fig, "PID discriminant")
    return fig


def roc_curves(pid_by_arm_and_sample: dict, arm_style: st.ArmStyle, *,
               electron="e-", muon="mu-", working_point=0.99):
    """ROC per arm, with the fixed-electron-efficiency working point marked.

    The working point is what makes two arms comparable when their discriminants
    have different offsets: "muon efficiency at 99% electron efficiency" is
    invariant to a constant shift, where "fraction correct at dnLL > 0" is not.
    """
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(6.2, 5.2))
    ax = axes[0]
    plotted = False
    for arm, samples in pid_by_arm_and_sample.items():
        if electron not in samples or muon not in samples:
            continue
        e = samples[electron]["dnll"].to_numpy(float)
        m = samples[muon]["dnll"].to_numpy(float)
        curve = pid_metrics.roc(e, m)
        if curve.empty:
            continue
        area = pid_metrics.auc(e, m)
        ax.plot(curve["eff_e"], curve["eff_mu"],
                **arm_style.line_kwargs(arm, label=f"{arm}  (AUC {area:.3f})"))
        point = pid_metrics.working_point(e, m, working_point)
        if np.isfinite(point["eff_mu"]):
            ax.plot([point["eff_e"]], [point["eff_mu"]],
                    marker=arm_style.marker(arm), markersize=9,
                    color=arm_style.color(arm), linestyle="none",
                    markeredgecolor="white", markeredgewidth=1.2)
        plotted = True
    if not plotted:
        return None
    ax.set_xlim(0.8, 1.005)
    ax.set_ylim(0.0, 1.02)
    st.finish(ax, title="PID ROC",
              xlabel="electron efficiency", ylabel="muon efficiency")
    fig.tight_layout()
    return fig


def pid_summary_bars(summary_by_arm: dict, arm_style: st.ArmStyle):
    """AUC and the working-point muon efficiency, side by side."""
    arms = [a for a, t in summary_by_arm.items() if t is not None and not t.empty]
    if not arms:
        return None
    metrics = [("auc", "AUC"), ("eff_mu_at_0", "μ efficiency at dnLL>0")]
    eff_columns = [c for c in summary_by_arm[arms[0]].columns
                   if c.startswith("eff_mu_at_") and c.endswith("_e")]
    if eff_columns:
        metrics.append((eff_columns[0], "μ eff. at fixed e eff."))

    fig, axes = st.grid_figure(len(metrics), ncols=len(metrics),
                               panel_size=(4.4, 3.6))
    for ax, (column, title) in zip(axes, metrics):
        values, labels, colours = [], [], []
        for arm in arms:
            table = summary_by_arm[arm]
            if column not in table.columns:
                continue
            values.append(float(table[column].iloc[0]))
            labels.append(arm)
            colours.append(arm_style.color(arm))
        positions = np.arange(len(values))
        # A 2px surface gap between adjacent bars, per the mark spec.
        ax.bar(positions, values, width=0.68, color=colours,
               edgecolor="white", linewidth=2.0)
        for pos, value in zip(positions, values):
            ax.text(pos, value, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=9, color=st.TEXT_PRIMARY)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, max(values) * 1.18 if values else 1)
        st.finish(ax, title=title, ylabel=title, legend=False)
    st.caption(fig, "PID summary")
    return fig
