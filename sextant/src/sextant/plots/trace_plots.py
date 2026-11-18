"""Fit-trajectory views: what every iterate of a likelihood fit did (Coral `fit_trace.csv`).

The trace holds, per fit (the main fit and each refit from a rival axis or a low
momentum seed) and per fit step, every minimiser iterate: the evaluations used so far
in that step, the step's nLL and the full parameter vector. Iteration -1 is the step's
returned minimum. Each step minimises its own likelihood (its own PMT selection), so
nLL values are comparable within a step, not across steps: the views show each step's
nLL relative to that step's own minimum.

    from sextant.plots import trace_plots
    tr = trace_plots.read("…/coral_electron_base_fit_trace.csv")
    trace_plots.event_figure(tr, truth.primary, event=30012).savefig(…)
    trace_plots.summary_figure(tr, truth.primary).savefig(…)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import style as st

PARAMS = ["t", "x", "y", "z", "theta", "phi", "p"]
STEP_COLOURS = st.PALETTE
STEP_NAMES = {0: "step 0: x,y,z,θ,φ (simplex)", 1: "step 1: t,p (L-BFGS-B)",
              2: "step 2: all 7 (simplex)", 3: "step 3"}


def read(path) -> pd.DataFrame:
    tr = pd.read_csv(path)
    tr["fit"] = tr["fit"].astype(str)
    return tr


def _direction(theta, phi):
    return np.c_[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]


def with_truth(tr: pd.DataFrame, truth_rings: pd.DataFrame) -> pd.DataFrame:
    """Each iterate's distance to the true vertex (cm), angle to the true direction (deg)
    and momentum residual (%), plus its cumulative evaluation count within its fit."""
    t = truth_rings.drop_duplicates("event").set_index("event")
    tr = tr[tr.event.isin(t.index)].copy()
    tt = t.loc[tr.event]
    d = _direction(tr.theta.to_numpy(), tr.phi.to_numpy())
    td = tt[["dx", "dy", "dz"]].to_numpy()
    td = td / np.linalg.norm(td, axis=1)[:, None]
    tr["d_vertex"] = np.linalg.norm(tr[["x", "y", "z"]].to_numpy() - tt[["x", "y", "z"]].to_numpy(), axis=1)
    tr["d_angle"] = np.degrees(np.arccos(np.clip((d * td).sum(1), -1, 1)))
    tr["d_p"] = 100 * (tr.p.to_numpy() / tt.p.to_numpy() - 1)
    # cumulative evaluations within each fit: offsets from the earlier steps' totals
    last = tr[tr["iter"] == -1].groupby(["event", "sub_event", "particle", "fit", "step"]).nfev.max()
    offs = last.groupby(level=[0, 1, 2, 3]).cumsum() - last
    key = list(zip(tr.event, tr.sub_event, tr.particle, tr.fit, tr.step))
    tr["evals"] = tr.nfev.to_numpy() + offs.reindex(key).fillna(0).to_numpy()
    step_min = tr[tr["iter"] == -1].set_index(["event", "sub_event", "particle", "fit", "step"]).nll
    tr["dnll_step"] = tr.nll.to_numpy() - step_min.reindex(key).to_numpy()
    return tr


def winner(tr_ev: pd.DataFrame) -> str:
    """The fit whose last step returned the lowest nLL (what Coral keeps)."""
    ends = tr_ev[(tr_ev["iter"] == -1)]
    ends = ends[ends.step == ends.step.max()]
    return ends.loc[ends.nll.idxmin(), "fit"]


def event_figure(tr: pd.DataFrame, truth_rings: pd.DataFrame, event: int, *, sub_event=0,
                 particle="e-", title=None):
    """Four panels against the fit's cumulative evaluations: each step's nLL above its own
    minimum, and the distance to the true vertex, direction and momentum. The kept fit is
    drawn in colour by step, the other refits in grey."""
    import matplotlib.pyplot as plt
    st.use_style()
    tr = tr[(tr.event == event) & (tr.sub_event == sub_event) & (tr.particle == particle)]
    tr = with_truth(tr, truth_rings) if "evals" not in tr else tr
    win = winner(tr)
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    panels = [("dnll_step", "nLL − step minimum", "symlog"), ("d_vertex", "|vertex − true| [cm]", "linear"),
              ("d_angle", "angle to true direction [deg]", "linear"), ("d_p", "$p/p_{\\mathrm{true}} - 1$ [%]", "linear")]
    for ax, (col, lab, scale) in zip(axs.flat, panels):
        for fit, g in tr.groupby("fit"):
            if fit == win:
                continue
            ax.plot(g.evals, g[col].clip(lower=0) if col == "dnll_step" else g[col],
                    color="#b9b8b3", lw=0.9, zorder=1)
        g = tr[tr.fit == win]
        for s, gs in g.groupby("step"):
            y = gs[col].clip(lower=0) if col == "dnll_step" else gs[col]
            ax.plot(gs.evals, y, color=STEP_COLOURS[s % len(STEP_COLOURS)], lw=1.6, zorder=3,
                    label=STEP_NAMES.get(s, f"step {s}"))
            ax.axvline(gs.evals.max(), color=st.GRID_COLOUR, ls="--", lw=0.8, zorder=0)
        if col == "d_p":
            ax.axhline(0, color=st.TRUTH_COLOUR, lw=0.8)
        ax.set_yscale(scale, **({"linthresh": 1.0} if scale == "symlog" else {}))
        if scale == "symlog":
            ax.set_ylim(bottom=0)
        ax.set_ylabel(lab)
        end = g[(g.step == g.step.max()) & (g["iter"] == -1)][col]
        if len(end) and col != "dnll_step":
            ax.annotate(f"final {end.iloc[0]:.2f}", xy=(0.98, 0.92), xycoords="axes fraction",
                        ha="right", fontsize=9, color=st.TEXT_SECONDARY)
    for ax in axs[1]:
        ax.set_xlabel("likelihood evaluations (cumulative, kept fit)")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    nref = tr.fit.nunique() - 1
    fig.suptitle(title or f"event {event}: kept fit = {win}; {nref} other refit(s) in grey", fontsize=11)
    fig.tight_layout()
    return fig


def settling(tr: pd.DataFrame, *, tol=None) -> pd.DataFrame:
    """Per kept fit: total evaluations per step, nLL dropped per step (on its own likelihood),
    and the fraction of the fit's evaluations after which each quantity stays within `tol`
    of its final value (vertex cm, angle deg, momentum %)."""
    tol = tol or {"d_vertex": 2.0, "d_angle": 0.2, "d_p": 0.2}
    rows = []
    for (e, sub, part), g in tr.groupby(["event", "sub_event", "particle"]):
        win = winner(g)
        g = g[g.fit == win].sort_values(["step", "evals"])
        total = g.evals.max()
        row = dict(event=e, sub_event=sub, particle=part, fit=win, total_evals=total,
                   n_fits=g.fit.nunique())
        for s, gs in g.groupby("step"):
            row[f"evals_step{s}"] = gs[gs["iter"] == -1].nfev.max()
            row[f"dnll_step{s}"] = gs.nll.iloc[0] - gs.nll.min()
        xf = g.iloc[-1]
        pos = g[["x", "y", "z"]].to_numpy() - xf[["x", "y", "z"]].to_numpy(float)
        d = _direction(g.theta.to_numpy(), g.phi.to_numpy())
        df = _direction(np.array([xf.theta]), np.array([xf.phi]))[0]
        dev = {"d_vertex": np.linalg.norm(pos, axis=1),
               "d_angle": np.degrees(np.arccos(np.clip(d @ df, -1, 1))),
               "d_p": 100*np.abs(g.p.to_numpy()/xf.p - 1)}
        for k, v in dev.items():
            out = np.flatnonzero(v > tol[k])
            row[f"settle_{k}"] = (g.evals.to_numpy()[out[-1]] / total) if len(out) else 0.0
        for k in ("d_vertex", "d_angle", "d_p"):
            row[f"final_{k}"] = xf[k] if k in xf else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_figure(tr: pd.DataFrame, truth_rings: pd.DataFrame, *, all_fits: pd.DataFrame | None = None):
    """Stacked over every traced event: (a) evaluations per step of the kept fit and of all
    refits; (b) the median truth residual vs progress through each step; (c) when each
    quantity reaches its final value; (d) nLL dropped per step."""
    import matplotlib.pyplot as plt
    st.use_style()
    tr = tr if "evals" in tr else with_truth(tr, truth_rings)
    s = settling(tr)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    # (a) evaluations per step
    ax = axs[0, 0]
    steps = sorted(c for c in s.columns if c.startswith("evals_step"))
    data = [s[c].dropna() for c in steps]
    refit_evals = (tr[tr["iter"] == -1].groupby(["event", "sub_event", "particle", "fit"]).nfev.sum()
                   .groupby(level=[0, 1, 2]).sum() - s.set_index(["event", "sub_event", "particle"]).total_evals)
    data.append(refit_evals.dropna())
    ax.boxplot(data, showfliers=False, labels=[c.replace("evals_", "") for c in steps] + ["other\nrefits"])
    for i, dd in enumerate(data):
        ax.annotate(f"med {np.median(dd):.0f}\nmean {np.mean(dd):.0f}", xy=(i + 1, np.percentile(dd, 75)),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("likelihood evaluations per event")
    ax.set_title("(a) where the evaluations go", fontsize=10)
    # (b) median truth residual vs progress within step
    ax = axs[0, 1]
    win = s.set_index(["event", "sub_event", "particle"]).fit
    key = pd.MultiIndex.from_arrays([tr.event, tr.sub_event, tr.particle])
    kept = tr[tr.fit.to_numpy() == win.reindex(key).to_numpy()]
    grid = np.linspace(0, 1, 21)
    curves = {k: [] for k in ("d_vertex", "d_angle", "d_p")}
    for _, g in kept.groupby(["event", "sub_event", "particle"]):
        prev = None
        row = {k: [] for k in curves}
        for stp, gs in g.groupby("step"):
            frac = gs.nfev.to_numpy() / max(gs.nfev.max(), 1)
            for k in curves:
                v = np.abs(gs[k].to_numpy()) / (10 if k == "d_vertex" else 1)
                # value held at each grid point = the last iterate reached by then
                i = np.searchsorted(frac, grid, side="right") - 1
                start = v[0] if prev is None else prev[k]
                row[k] += list(np.where(i >= 0, v[np.clip(i, 0, None)], start))
            prev = {k: np.abs(gs[k].to_numpy()[-1]) / (10 if k == "d_vertex" else 1) for k in curves}
        for k in curves:
            curves[k].append(row[k])
    nsteps = kept.step.nunique()
    xs = np.concatenate([s_ + grid for s_ in range(nsteps)])
    for c, (k, lab) in enumerate((("d_vertex", "vertex [cm] / 10"), ("d_angle", "angle [deg]"), ("d_p", "|p| [%]"))):
        arr = np.array([r for r in curves[k] if len(r) == len(xs)])
        ax.plot(xs, np.median(arr, axis=0), color=st.PALETTE[c], lw=1.6, label=lab)
    for stp in sorted(kept.step.unique())[1:]:
        ax.axvline(stp, color=st.GRID_COLOUR, ls="--", lw=0.8)
    ax.set_xlabel("progress (step index + fraction of the step's evaluations); every event on every grid point")
    ax.set_ylabel("median distance to truth")
    ax.legend(fontsize=8)
    ax.set_title("(b) median residual to truth through the fit", fontsize=10)
    # (c) settling
    ax = axs[1, 0]
    for i, (k, lab) in enumerate((("d_vertex", "vertex within 2 cm"), ("d_angle", "direction within 0.2°"),
                                  ("d_p", "momentum within 0.2%"))):
        v = np.sort(s[f"settle_{k}"].to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=st.PALETTE[i], lw=1.6, label=lab)
    ax.set_xlabel("fraction of the kept fit's evaluations")
    ax.set_ylabel("fraction of events settled")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(c) when each quantity reaches its final value", fontsize=10)
    # (d) nLL dropped per step
    ax = axs[1, 1]
    cols = sorted(c for c in s.columns if c.startswith("dnll_step"))
    ax.boxplot([s[c].dropna().clip(lower=0.01) for c in cols], showfliers=False,
               labels=[c.replace("dnll_", "") for c in cols])
    ax.set_yscale("log")
    ax.set_ylabel("nLL dropped within the step (own likelihood)")
    ax.set_title("(d) what each step gains", fontsize=10)
    fig.suptitle(f"Fit trajectories, {len(s)} traced fits", fontsize=11)
    fig.tight_layout()
    return fig, s


def paired_scatter(res_a: pd.DataFrame, res_b: pd.DataFrame, column: str, *, label_a="Coral",
                   label_b="fiTQun", unit="", clip=None, ax=None):
    """R41 view (2): one point per shared event, arm A's residual against arm B's. Shared noise
    lies along the diagonal; extra core width in A widens the cloud vertically; A's failures sit
    far above / below the band. Marginal widths (iqr68-like half width of the central 68%) and
    the Pearson correlation of the clipped residuals are annotated."""
    import matplotlib.pyplot as plt
    st.use_style()
    ev = res_a.index.intersection(res_b.index)
    a = res_a.loc[ev, column].to_numpy(float); b = res_b.loc[ev, column].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    clip = clip or float(np.percentile(np.abs(np.r_[a, b]), 98))
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(np.clip(b, -clip, clip), np.clip(a, -clip, clip), s=3, alpha=0.35, color=st.PALETTE[0],
               rasterized=True, linewidths=0)
    ax.plot([-clip, clip], [-clip, clip], color=st.GRID_COLOUR, lw=1)
    w = lambda x: (np.percentile(x, 84) - np.percentile(x, 16)) / 2
    m = (np.abs(a) < clip) & (np.abs(b) < clip)
    r = np.corrcoef(a[m], b[m])[0, 1]
    ax.text(0.03, 0.97, f"n = {len(a)}\nwidth {label_a} {w(a):.2f}{unit}, {label_b} {w(b):.2f}{unit}\nr = {r:.2f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel(f"{label_b} residual [{unit}]" if unit else f"{label_b} residual")
    ax.set_ylabel(f"{label_a} residual [{unit}]" if unit else f"{label_a} residual")
    ax.set_xlim(-clip, clip); ax.set_ylim(-clip, clip)
    return ax
