"""Particle identification: the e-/mu- likelihood difference and what it buys.

The discriminant is

    dnll = nLL(mu hypothesis) - nLL(e hypothesis)

so **dnll > 0 means electron-like**. Both algorithms produce it the same way,
from the two hypotheses' total negative log-likelihoods.

Two things about scoring it are easy to get wrong and are handled here:

*Success cuts.* Requiring both hypotheses to have converged removes a large,
*biased* slice of the sample -- the e- fit routinely fails on a true muon and
vice versa, so the cut preferentially discards exactly the events the
discriminant finds hardest. The default is therefore to keep every event with a
finite dnll under both hypotheses, and to report how many were dropped.

*Working points.* A raw "fraction classified correctly at dnll > 0" is not
comparable between two arms whose discriminants have different offsets. The
honest comparison is the ROC curve, plus the muon efficiency at a *fixed*
electron efficiency, which is the number the HK sensitivity actually depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.schema import HYP_ELECTRON, HYP_MUON, STAGE_FINAL

#: The electron efficiency the mis-identification working point is quoted at.
DEFAULT_ELECTRON_EFFICIENCY = 0.99


def discriminant(fits: pd.DataFrame, *, electron=HYP_ELECTRON, muon=HYP_MUON,
                 sub_event: int | None = 0, require_success: bool = False) -> pd.DataFrame:
    """One row per event: nll_e, nll_mu, dnll and the resulting label."""
    if fits.empty:
        return pd.DataFrame(columns=["event", "nll_e", "nll_mu", "dnll", "pid"])
    df = fits[fits["stage"] == STAGE_FINAL]
    if sub_event is not None:
        df = df[df["sub_event"] == sub_event]
    if require_success:
        df = df[df["success"]]

    def side(hyp, name):
        return (df[df["hypothesis"] == hyp]
                .sort_values("event").drop_duplicates("event")
                [["event", "nll"]].rename(columns={"nll": name}))

    merged = side(electron, "nll_e").merge(side(muon, "nll_mu"), on="event")
    merged["dnll"] = merged["nll_mu"] - merged["nll_e"]
    n_before = len(merged)
    merged = merged[np.isfinite(merged["dnll"])].copy()
    merged["pid"] = np.where(merged["dnll"] > 0, electron, muon)
    merged.attrs["n_dropped_nonfinite"] = n_before - len(merged)
    return merged.reset_index(drop=True)


def roc(dnll_electron, dnll_muon, *, n_points: int = 512) -> pd.DataFrame:
    """ROC of the discriminant, from the two true-particle samples.

    `dnll_electron` are the discriminant values of events whose true particle is
    an electron, `dnll_muon` those of true muons. Returns thresholds with the
    electron efficiency (true positive rate) and muon rejection.
    """
    e = np.asarray(dnll_electron, float)
    m = np.asarray(dnll_muon, float)
    e, m = e[np.isfinite(e)], m[np.isfinite(m)]
    if e.size == 0 or m.size == 0:
        return pd.DataFrame(columns=["threshold", "eff_e", "eff_mu", "mis_mu"])

    pooled = np.concatenate([e, m])
    lo, hi = np.percentile(pooled, [0.1, 99.9])
    thresholds = np.unique(np.concatenate([
        np.linspace(lo, hi, n_points), [0.0]]))
    rows = []
    for threshold in thresholds:
        rows.append({
            "threshold": float(threshold),
            # An event is called electron-like when dnll > threshold.
            "eff_e": float(np.mean(e > threshold)),
            "mis_mu": float(np.mean(m > threshold)),
            "eff_mu": float(np.mean(m <= threshold)),
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def auc(dnll_electron, dnll_muon) -> float:
    """Area under the ROC, computed as the Mann-Whitney statistic.

    Rank-based rather than trapezoidal on a sampled curve, so it is exact and
    insensitive to the threshold grid.
    """
    e = np.asarray(dnll_electron, float)
    m = np.asarray(dnll_muon, float)
    e, m = e[np.isfinite(e)], m[np.isfinite(m)]
    if e.size == 0 or m.size == 0:
        return np.nan
    pooled = np.concatenate([e, m])
    ranks = pd.Series(pooled).rank().to_numpy()
    rank_sum_e = ranks[:e.size].sum()
    return float((rank_sum_e - e.size * (e.size + 1) / 2.0) / (e.size * m.size))


def working_point(dnll_electron, dnll_muon,
                  electron_efficiency: float = DEFAULT_ELECTRON_EFFICIENCY) -> dict:
    """Muon efficiency at a fixed electron efficiency.

    This is the comparison that survives a constant offset between two
    algorithms' discriminants, which a raw dnll > 0 count does not.
    """
    e = np.asarray(dnll_electron, float)
    m = np.asarray(dnll_muon, float)
    e, m = e[np.isfinite(e)], m[np.isfinite(m)]
    if e.size == 0 or m.size == 0:
        return {"threshold": np.nan, "eff_e": np.nan, "eff_mu": np.nan}
    # Keep the requested fraction of electrons: the threshold is the
    # (1 - eff) quantile of the electron discriminant.
    threshold = float(np.percentile(e, 100.0 * (1.0 - electron_efficiency)))
    return {
        "threshold": threshold,
        "eff_e": float(np.mean(e > threshold)),
        "eff_mu": float(np.mean(m <= threshold)),
        "target_eff_e": electron_efficiency,
    }


def summarise(pid_by_sample: dict, *,
              electron_efficiency: float = DEFAULT_ELECTRON_EFFICIENCY) -> pd.DataFrame:
    """PID summary for one arm across its true-electron and true-muon samples.

    `pid_by_sample` maps a true-particle label to that sample's discriminant
    table, e.g. {'e-': df_from_electron_run, 'mu-': df_from_muon_run}.
    """
    if HYP_ELECTRON not in pid_by_sample or HYP_MUON not in pid_by_sample:
        return pd.DataFrame()
    e = pid_by_sample[HYP_ELECTRON]["dnll"].to_numpy(float)
    m = pid_by_sample[HYP_MUON]["dnll"].to_numpy(float)
    point = working_point(e, m, electron_efficiency)
    return pd.DataFrame([{
        "n_electron": int(np.isfinite(e).sum()),
        "n_muon": int(np.isfinite(m).sum()),
        "auc": auc(e, m),
        "eff_e_at_0": float(np.mean(e[np.isfinite(e)] > 0)),
        "eff_mu_at_0": float(np.mean(m[np.isfinite(m)] <= 0)),
        f"eff_mu_at_{electron_efficiency:.0%}_e": point["eff_mu"],
        "threshold": point["threshold"],
        "median_dnll_electron": float(np.nanmedian(e)) if e.size else np.nan,
        "median_dnll_muon": float(np.nanmedian(m)) if m.size else np.nan,
    }])


def confusion(pid_table: pd.DataFrame, true_particle: str) -> pd.DataFrame:
    counts = pid_table["pid"].value_counts()
    total = int(counts.sum())
    return pd.DataFrame([{
        "true": true_particle, "n": total,
        "classified_e": int(counts.get(HYP_ELECTRON, 0)),
        "classified_mu": int(counts.get(HYP_MUON, 0)),
        "correct_fraction": (float(counts.get(true_particle, 0)) / total
                             if total else np.nan),
    }])


#: The variables the HK validation spec asks every quantity against.
PROFILE_VARIABLES = ("p_true", "dwall", "towall")


def classified(pid_table: pd.DataFrame, truth_primary: pd.DataFrame,
               true_particle: str, *, threshold: float = 0.0) -> pd.DataFrame:
    """Per event: correctly identified at `threshold`, with the binning variables.

    `truth_primary` is one true ring per event (`Truth.primary`); momentum,
    dwall and towall are those of that ring, the same as the resolution
    profiles use.
    """
    from ..io.geometry import dwall, towall

    if pid_table.empty or truth_primary.empty:
        return pd.DataFrame(columns=["event", "correct", *PROFILE_VARIABLES])
    merged = pid_table[["event", "dnll"]].merge(
        truth_primary[["event", "p", "x", "y", "z", "dx", "dy", "dz"]], on="event")
    electron_like = merged["dnll"].to_numpy(float) > threshold
    return pd.DataFrame({
        "event": merged["event"].to_numpy(),
        "correct": electron_like if true_particle == HYP_ELECTRON else ~electron_like,
        "p_true": merged["p"].to_numpy(float),
        "dwall": dwall(merged["x"], merged["y"], merged["z"]),
        "towall": towall(merged["x"], merged["y"], merged["z"],
                         merged["dx"], merged["dy"], merged["dz"]),
    })


def efficiency_profiles(classified_by_arm: dict) -> pd.DataFrame:
    """Identification efficiency against p, dwall and towall, long format.

    `classified_by_arm` maps an arm label to {true particle: `classified`
    table}. Bins follow the resolution profiles' standard edges, falling back
    to equal-count bins where those leave fewer than three usable bins.
    """
    from . import resolution, stats

    rows = []
    for label, by_particle in classified_by_arm.items():
        for particle, table in by_particle.items():
            if table.empty:
                continue
            for versus in PROFILE_VARIABLES:
                edges = resolution.BINNINGS[versus][2]
                counts, _ = np.histogram(table[versus].dropna(), bins=edges)
                if int((counts >= stats.MIN_RATE_EVENTS).sum()) < resolution.MIN_USEFUL_BINS:
                    edges = resolution.quantile_bins(table[versus],
                                                     stats.MIN_RATE_EVENTS)
                if len(edges) < 2:
                    continue
                prof = stats.rate_profile(table[versus], table["correct"], edges)
                if not prof.empty:
                    rows.append(prof.assign(arm=label, true_particle=particle,
                                            versus=versus))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["arm", "true_particle", "versus", "low", "high", "centre",
                "value", "ci_lo", "ci_hi", "k", "n"]]
