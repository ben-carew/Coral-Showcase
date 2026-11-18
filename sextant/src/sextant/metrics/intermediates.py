"""Comparing the algorithms' internals, not just their answers.

fiTQun still beats Coral on single-ring electrons. Comparing final vertices
cannot say why; comparing the quantities the fits are built from can. Four
levels are available, in increasing order of what they require:

1. **Charge normalisation** -- total predicted photoelectrons against total
   observed charge, per hypothesis. Needs nothing but the standard outputs
   (fiTQun writes `fq1rtotmu` and `fqtotq`; Coral's comes from a probe run).
   This is the direct test of the charge model's overall scale, and the
   documented next lever: the muon hypothesis under-predicts light.

2. **Likelihood decomposition** -- Coral splits its NLL into unhit / phit /
   cpdf / tpdf; fiTQun reports only a total. So the *split* is Coral-only, but
   the total, and the difference from each code's own zero-ring hypothesis,
   are comparable.

3. **Fit trajectory** -- both codes fit in stages, and both record the
   intermediate results. Where an estimate goes wrong in the sequence is a
   sharper question than whether the endpoint is wrong.

4. **Per-PMT predictions** -- predicted charge and time residual PMT by PMT.
   Produced by `sextant.probe`.

### Why a common parameter point matters

Levels 1 and 4 are only meaningful when both codes are evaluated at the *same*
parameters. Compared at each code's own fitted vertex, predicted charges differ
mostly because the vertices differ, which is what you were trying to explain.
Evaluated at a common point -- truth, or one code's fit -- what remains is the
model difference. That is why every table here carries a `point` column and why
`compare_at_point` refuses to join across different points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.schema import HYP_ELECTRON, HYP_MUON, POINT_SELF, STAGE_FINAL
from . import stats


# ---------------------------------------------------------------------------
# 1. Charge normalisation
# ---------------------------------------------------------------------------

def charge_normalisation(fits: pd.DataFrame, event_totals: pd.DataFrame, *,
                         label: str = "", sub_event: int | None = 0) -> pd.DataFrame:
    """Predicted total light against observed, per event and hypothesis.

    `ratio` is the quantity of interest: 1.0 means the charge model reproduces
    the light actually seen. A hypothesis that systematically predicts too
    little light cannot explain the off-ring hits except through the unhit and
    hit-occupancy terms, which is how a constant likelihood offset between
    hypotheses arises.
    """
    if fits.empty or event_totals is None or event_totals.empty:
        return pd.DataFrame(columns=["arm", "event", "sub_event", "hypothesis",
                                     "totmu", "totq", "ratio"])
    df = fits[fits["stage"] == STAGE_FINAL]
    if sub_event is not None:
        df = df[df["sub_event"] == sub_event]
    df = df[np.isfinite(df["totmu"])]
    if df.empty:
        return pd.DataFrame(columns=["arm", "event", "sub_event", "hypothesis",
                                     "totmu", "totq", "ratio"])

    merged = df.merge(event_totals[["event", "sub_event", "totq"]],
                      on=["event", "sub_event"], how="inner")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(merged["totq"].to_numpy(float) > 0,
                         merged["totmu"].to_numpy(float)
                         / merged["totq"].to_numpy(float), np.nan)
    out = merged[["event", "sub_event", "hypothesis", "totmu", "totq"]].copy()
    out["ratio"] = ratio
    out.insert(0, "arm", label)
    return out.reset_index(drop=True)


def charge_normalisation_summary(tables) -> pd.DataFrame:
    frames = [t for t in (tables if isinstance(tables, (list, tuple)) else [tables])
              if t is not None and not t.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for (arm, hyp), group in combined.groupby(["arm", "hypothesis"]):
        rows.append({
            "arm": arm, "hypothesis": hyp, "n": len(group),
            "median_ratio": stats.median(group["ratio"]),
            "iqr68_ratio": stats.iqr68(group["ratio"]),
            "median_totmu": stats.median(group["totmu"]),
            "median_totq": stats.median(group["totq"]),
            "under_predicting_pct": 100.0 * float(
                np.nanmean(group["ratio"].to_numpy(float) < 1.0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Likelihood decomposition
# ---------------------------------------------------------------------------

def term_table(nll_terms: pd.DataFrame, *, label: str = "",
               point: str = POINT_SELF) -> pd.DataFrame:
    """Wide per-event view of the NLL components at one parameter point."""
    if nll_terms.empty:
        return pd.DataFrame()
    df = nll_terms[nll_terms["point"] == point]
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index=["event", "sub_event", "hypothesis"],
                          columns="term", values="value",
                          aggfunc="first").reset_index()
    wide.columns.name = None
    wide.insert(0, "arm", label)
    return wide


def term_summary(nll_terms: pd.DataFrame, *, label: str = "",
                 point: str = POINT_SELF) -> pd.DataFrame:
    """Median size of each likelihood component, and its share of the total."""
    wide = term_table(nll_terms, label=label, point=point)
    if wide.empty:
        return pd.DataFrame()
    components = [c for c in ("unhit", "phit", "cpdf", "tpdf")
                  if c in wide.columns]
    rows = []
    for hyp, group in wide.groupby("hypothesis"):
        total = group[components].sum(axis=1) if components else group.get("total")
        for term in components + (["total"] if "total" in wide.columns else []):
            values = group[term].to_numpy(float)
            row = {"arm": label, "hypothesis": hyp, "term": term,
                   "n": int(np.isfinite(values).sum()),
                   "median": stats.median(values), "iqr68": stats.iqr68(values)}
            if term in components and total is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    share = values / total.to_numpy(float)
                row["median_share"] = stats.median(share)
            rows.append(row)
    return pd.DataFrame(rows)


def zero_ring_improvement(nll_terms: pd.DataFrame, *, label: str = "",
                          point: str = POINT_SELF) -> pd.DataFrame:
    """How much better each hypothesis fits than the zero-ring hypothesis.

    The ONE likelihood quantity every algorithm can supply and that means the
    same thing in each: an absolute NLL carries a per-code constant, but the
    improvement over that code's own no-ring hypothesis does not. fiTQun
    publishes no per-term split at all (its official output has `fq1rnll` and
    `fq0rnll` and nothing between them), so for a fiTQun-only report this is
    the likelihood decomposition -- the alternative is a bar chart with one bar.
    """
    wide = term_table(nll_terms, label=label, point=point)
    if wide.empty or "total" not in wide.columns or "zero_ring" not in wide.columns:
        return pd.DataFrame()
    out = wide[["event", "sub_event", "hypothesis"]].copy()
    out["arm"] = label
    # Positive = the ring hypothesis explains the event better than no ring.
    out["improvement"] = (wide["zero_ring"].to_numpy(float)
                          - wide["total"].to_numpy(float))
    return out[np.isfinite(out["improvement"].to_numpy(float))].reset_index(drop=True)


def has_term_split(nll_terms: pd.DataFrame) -> bool:
    """True when a real per-component decomposition is present.

    'total' and 'zero_ring' are totals, not components: an arm carrying only
    those has nothing to decompose, and drawing it as a breakdown produces a
    one-bar chart that reads as a broken plot rather than as a limitation of
    the input.
    """
    if nll_terms is None or nll_terms.empty or "term" not in nll_terms:
        return False
    return bool(set(nll_terms["term"].unique()) - {"total", "zero_ring"})


def hypothesis_term_difference(nll_terms: pd.DataFrame, *, label: str = "",
                               point: str = POINT_SELF,
                               electron=HYP_ELECTRON, muon=HYP_MUON) -> pd.DataFrame:
    """Per-term mu-minus-e difference: which component drives the PID.

    The discriminant is a difference of totals, so the question "what makes an
    event muon-like" is answered term by term. A component with a large,
    event-independent offset is a systematic tilt of the PID rather than
    information about the particle -- which is how the hit-occupancy term was
    found to carry a constant anti-muon offset.
    """
    wide = term_table(nll_terms, label=label, point=point)
    if wide.empty:
        return pd.DataFrame()
    components = [c for c in ("unhit", "phit", "cpdf", "tpdf", "total")
                  if c in wide.columns]
    e = wide[wide["hypothesis"] == electron].set_index(["event", "sub_event"])
    m = wide[wide["hypothesis"] == muon].set_index(["event", "sub_event"])
    common = e.index.intersection(m.index)
    if len(common) == 0:
        return pd.DataFrame()
    rows = []
    for term in components:
        delta = (m.loc[common, term].to_numpy(float)
                 - e.loc[common, term].to_numpy(float))
        rows.append({"arm": label, "term": term, "n": len(common),
                     "median_delta": stats.median(delta),
                     "iqr68_delta": stats.iqr68(delta),
                     # A term whose spread is small next to its offset is acting
                     # as a constant, not as a discriminant.
                     "offset_dominated": bool(
                         np.isfinite(stats.median(delta))
                         and abs(stats.median(delta)) > 2 * stats.iqr68(delta))})
    return pd.DataFrame(rows)


def delta_nll_vs_zero_ring(nll_terms: pd.DataFrame, *, label: str = "",
                           point: str = POINT_SELF) -> pd.DataFrame:
    """NLL improvement over the no-ring hypothesis -- comparable across codes.

    Absolute NLLs carry code-specific constants (different unhit subsampling,
    different offsets) and must never be compared directly between Coral and
    fiTQun. The improvement each code's fit achieves over its *own* zero-ring
    hypothesis has those constants cancel, and is a like-for-like measure of how
    much the fit explains.
    """
    wide = term_table(nll_terms, label=label, point=point)
    if wide.empty or "zero_ring" not in wide.columns or "total" not in wide.columns:
        return pd.DataFrame()
    out = wide[["arm", "event", "sub_event", "hypothesis"]].copy()
    out["delta_nll_0r"] = (wide["zero_ring"].to_numpy(float)
                           - wide["total"].to_numpy(float))
    return out


# ---------------------------------------------------------------------------
# 3. Fit trajectory
# ---------------------------------------------------------------------------

def stage_trajectory(stages: pd.DataFrame, truth_rings: pd.DataFrame, *,
                     label: str = "", hypothesis: str = HYP_ELECTRON,
                     sub_event: int | None = 0) -> pd.DataFrame:
    """Residuals after each fit stage: where in the sequence accuracy is won.

    Both codes fit in stages, so this puts Coral's [step_0..2] and fiTQun's
    recorded stages on one axis. The stage *numbers* are not equivalent between
    codes -- they are each code's own sequence -- so this compares shapes, not
    stage 1 against stage 1.
    """
    from .residuals import compute as compute_residuals

    if stages.empty:
        return pd.DataFrame()
    df = stages[stages["hypothesis"] == hypothesis]
    if sub_event is not None:
        df = df[df["sub_event"] == sub_event]
    frames = []
    for stage, group in df.groupby("stage"):
        per_event = (group.sort_values("event")
                          .drop_duplicates("event").reset_index(drop=True))
        residuals = compute_residuals(per_event, truth_rings)
        if residuals.empty:
            continue
        frames.append({
            "arm": label, "hypothesis": hypothesis, "stage": int(stage),
            "n": len(residuals),
            "median_vertex_dist": stats.median(residuals["vertex_dist"]),
            "median_angle_deg": stats.median(residuals["angle_deg"]),
            "median_p_res_pct": stats.median(residuals["p_res_pct"]),
            "iqr68_p_res_pct": stats.iqr68(residuals["p_res_pct"]),
            "median_nll": stats.median(per_event["nll"]),
        })
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# 4. Per-PMT predictions
# ---------------------------------------------------------------------------

def compare_at_point(a: pd.DataFrame, b: pd.DataFrame, *, labels=("a", "b"),
                     point: str | None = None) -> pd.DataFrame:
    """Join two arms' per-PMT predictions, PMT by PMT, at one parameter point.

    Refuses to join across different points: a per-PMT difference between
    predictions made at two different vertices is not a model comparison, and
    silently producing one is the mistake this whole module exists to avoid.
    """
    if a.empty or b.empty:
        return pd.DataFrame()
    points = set(a["point"].unique()) | set(b["point"].unique())
    if point is None:
        if len(points) != 1:
            raise ValueError(
                f"per-PMT tables span several parameter points {sorted(points)}; "
                f"pass point= to choose one. Comparing predictions made at "
                f"different parameters measures the parameter difference, not "
                f"the charge models.")
        point = points.pop()
    a = a[a["point"] == point]
    b = b[b["point"] == point]
    if a.empty or b.empty:
        raise ValueError(f"one side has no rows at point '{point}'")

    keys = ["event", "sub_event", "hypothesis", "pmt"]
    merged = a.merge(b, on=keys, suffixes=(f"_{labels[0]}", f"_{labels[1]}"))
    for column in ("mu_direct", "mu_indirect", "t_resid"):
        ca, cb = f"{column}_{labels[0]}", f"{column}_{labels[1]}"
        if ca in merged and cb in merged:
            merged[f"d_{column}"] = (merged[ca].to_numpy(float)
                                     - merged[cb].to_numpy(float))
    mu_a = merged.get(f"mu_direct_{labels[0]}")
    mu_b = merged.get(f"mu_direct_{labels[1]}")
    if mu_a is not None and mu_b is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            merged["ratio_mu_direct"] = np.where(
                mu_b.to_numpy(float) > 0,
                mu_a.to_numpy(float) / mu_b.to_numpy(float), np.nan)
    merged["point"] = point
    return merged


def pmt_summary(comparison: pd.DataFrame, *, labels=("a", "b")) -> pd.DataFrame:
    """How far apart two per-PMT charge models are, and where."""
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for hyp, group in comparison.groupby("hypothesis"):
        hit = group[group.get(f"is_hit_{labels[0]}", pd.Series(True, index=group.index))]
        rows.append({
            "hypothesis": hyp, "n_pmt": len(group), "n_hit": len(hit),
            "median_ratio_mu_direct": stats.median(group.get("ratio_mu_direct")),
            "iqr68_ratio_mu_direct": stats.iqr68(group.get("ratio_mu_direct")),
            "median_d_mu_direct": stats.median(group.get("d_mu_direct")),
            "median_d_t_resid": stats.median(group.get("d_t_resid")),
            "point": group["point"].iloc[0],
        })
    return pd.DataFrame(rows)


def charge_profile(predictions: pd.DataFrame, *, bins=None,
                   hits_only: bool = True) -> pd.DataFrame:
    """Predicted charge against *observed*, profiled along the ring axis.

    `pmt_summary` asks how two codes differ from each other; this asks how one
    code differs from the data, which is the question when only one code can be
    probed. A charge model that is merely mis-normalised gives a flat ratio
    across `cos_theta`; one whose *shape* is wrong -- the Cherenkov cone in the
    wrong place, or the wrong width -- gives a ratio that slides with angle.
    Distinguishing those two is what tells you whether to retune a constant or
    fix the light model.
    """
    if predictions.empty:
        return pd.DataFrame()
    if bins is None:
        bins = np.linspace(-1.0, 1.0, 21)
    df = predictions
    if hits_only:
        df = df[df["is_hit"].astype(bool)]
    df = df[df["q_obs"] > 0]
    if df.empty:
        return pd.DataFrame()

    mu = df["mu_direct"].to_numpy() + df["mu_indirect"].fillna(0.0).to_numpy()
    ratio = np.divide(mu, df["q_obs"].to_numpy(),
                      out=np.full(len(df), np.nan),
                      where=df["q_obs"].to_numpy() > 0)
    work = pd.DataFrame({"cos_theta": df["cos_theta"].to_numpy(),
                         "ratio": ratio, "mu": mu,
                         "q_obs": df["q_obs"].to_numpy()})
    work["bin"] = pd.cut(work["cos_theta"], bins)

    rows = []
    for interval, group in work.groupby("bin", observed=True):
        if len(group) < 10:
            continue
        rows.append({
            "cos_theta_lo": interval.left,
            "cos_theta_hi": interval.right,
            "cos_theta_mid": 0.5 * (interval.left + interval.right),
            "n_pmt": len(group),
            "sum_q_obs": float(group["q_obs"].sum()),
            "sum_mu": float(group["mu"].sum()),
            "median_q_obs": stats.median(group["q_obs"]),
            "median_mu": stats.median(group["mu"]),
            "median_ratio": stats.median(group["ratio"]),
            "iqr68_ratio": stats.iqr68(group["ratio"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # The shape statistic. A per-bin predicted/observed ratio conflates two
    # things: whether the light model is normalised right, and whether it puts
    # the light in the right places. Comparing each bin's *share* of the total
    # divides the normalisation out, so `shape_ratio` is 1.0 everywhere iff the
    # angular distribution is right -- however wrong the overall scale is.
    out["frac_mu"] = out["sum_mu"] / out["sum_mu"].sum()
    out["frac_q_obs"] = out["sum_q_obs"] / out["sum_q_obs"].sum()
    out["shape_ratio"] = out["frac_mu"] / out["frac_q_obs"]
    return out
