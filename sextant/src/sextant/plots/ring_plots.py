"""Multi-ring figures: how many rings, and how good each one is."""

from __future__ import annotations

import numpy as np

from ..metrics import rings as ring_metrics
from . import style as st


def multiplicity(per_event_by_arm: dict, arm_style: st.ArmStyle, *,
                 max_rings: int = ring_metrics.DEFAULT_MAX_RINGS):
    """Reconstructed ring-count distribution, with the true one as reference."""
    arms = [a for a, t in per_event_by_arm.items() if not t.empty]
    if not arms:
        return None
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(8.0, 4.2))
    ax = axes[0]
    counts = np.arange(1, max_rings + 1)
    width = 0.8 / (len(arms) + 1)

    reference = per_event_by_arm[arms[0]]
    true_fraction = np.array([(reference["n_true_clipped"] == c).mean()
                              for c in counts]) * 100.0
    ax.bar(counts - 0.4 + width / 2, true_fraction, width=width,
           color=st.TRUTH_COLOUR, alpha=0.45, edgecolor="white", linewidth=2.0,
           label="true")
    for i, arm in enumerate(arms, start=1):
        table = per_event_by_arm[arm]
        fraction = np.array([(table["n_reco_clipped"] == c).mean()
                             for c in counts]) * 100.0
        ax.bar(counts - 0.4 + width * (i + 0.5), fraction, width=width,
               color=arm_style.color(arm), edgecolor="white", linewidth=2.0,
               label=arm)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) if c < max_rings else f"{max_rings}+"
                        for c in counts])
    st.finish(ax, title="Ring multiplicity",
              xlabel="rings per event", ylabel="% of events")
    fig.tight_layout()
    return fig


def counting_accuracy(per_event_by_arm: dict, arm_style: st.ArmStyle):
    """Exact-ring-count accuracy split by the true ring count.

    Split rather than pooled: pooled accuracy is dominated by whichever count
    the sample is mostly made of, so a fitter that always answers "1" scores
    well on a mostly-single-ring sample without being right about anything.
    """
    arms = [a for a, t in per_event_by_arm.items() if not t.empty]
    if not arms:
        return None
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(7.6, 4.2))
    ax = axes[0]
    true_counts = sorted(set().union(*[
        set(per_event_by_arm[a]["n_true_clipped"].unique()) for a in arms]))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        table = per_event_by_arm[arm]
        accuracy, positions = [], []
        for j, count in enumerate(true_counts):
            sel = table[table["n_true_clipped"] == count]
            if sel.empty:
                continue
            accuracy.append(100.0 * sel["exact_count"].mean())
            positions.append(j - 0.4 + width * (i + 0.5))
        ax.bar(positions, accuracy, width=width, color=arm_style.color(arm),
               edgecolor="white", linewidth=2.0, label=arm)
    ax.set_xticks(range(len(true_counts)))
    ax.set_xticklabels([str(int(c)) for c in true_counts])
    ax.set_ylim(0, 100)
    st.finish(ax, title="Exact ring-count accuracy, by true ring count",
              xlabel="true ring count", ylabel="% of events counted exactly")
    fig.tight_layout()
    return fig


def per_slot_quality(ring_summary_by_arm: dict, arm_style: st.ArmStyle):
    """Match rate, angular error and momentum bias for each ring slot."""
    arms = [a for a, t in ring_summary_by_arm.items()
            if t is not None and not t.empty]
    if not arms:
        return None
    panels = [("matched_pct", "matched to a true ring", "%"),
              ("median_angle_deg", "angular error of matched rings", "deg"),
              ("median_dp_pct", "momentum bias of matched rings", "%")]
    fig, axes = st.grid_figure(len(panels), ncols=len(panels),
                               panel_size=(4.8, 3.6))
    for ax, (column, title, unit) in zip(axes, panels):
        for arm in arms:
            table = ring_summary_by_arm[arm]
            if column not in table.columns:
                continue
            ax.plot(table["slot"], table[column],
                    **arm_style.point_kwargs(arm))
        if column == "median_dp_pct":
            ax.axhline(0.0, color=st.TEXT_SECONDARY, lw=0.8, ls="--")
        st.finish(ax, title=title, xlabel="ring slot (1 = leading)",
                  ylabel=f"{title} [{unit}]", legend=(column == panels[0][0]))
    st.caption(fig, "Per-ring quality by momentum-ordered slot")
    return fig


def confusion_heatmap(per_event, arm_label: str, *,
                      max_rings: int = ring_metrics.DEFAULT_MAX_RINGS):
    """True against reconstructed ring count, as a sequential heatmap.

    One hue, light to dark -- a magnitude, so a sequential ramp, never a
    rainbow. Cells are annotated because the numbers matter more than the shade.
    """
    table = ring_metrics.counting_confusion(per_event, max_rings)
    if table.empty:
        return None
    fig, axes = st.grid_figure(1, ncols=1, panel_size=(5.4, 4.6))
    ax = axes[0]
    values = table.to_numpy(float)
    fractions = values / np.clip(values.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            count = int(values[i, j])
            if count == 0:
                continue
            ax.text(j, i, f"{count}", ha="center", va="center", fontsize=9,
                    color="white" if fractions[i, j] > 0.55 else st.TEXT_PRIMARY)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.grid(False)
    st.finish(ax, title=f"Ring-count confusion — {arm_label}",
              xlabel="reconstructed rings", ylabel="true rings", legend=False)
    fig.tight_layout()
    return fig
