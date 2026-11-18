"""Tables: the same numbers as the figures, in a form you can read and diff.

Every quantitative claim a Sextant report makes exists twice -- once as a figure
and once as a table row -- for two reasons. Three slots of the categorical
palette fall below 3:1 contrast on white, and the accessibility rule for that is
that the figure must be accompanied by the values in text. And a CSV is what
lets the next run be compared against this one without re-reading a PDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(table: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def format_for_display(table: pd.DataFrame, *, max_rows: int = 40,
                       float_format: str = "{:.4g}",
                       drop_empty_columns: bool = True) -> pd.DataFrame:
    """Round floats and normalise types for rendering. Nothing is truncated.

    An earlier version elided anything over 42 characters, which made the
    validation table -- whose entire content is the message -- unreadable. Long
    values are *wrapped* at render time instead (see `layout` in the builder),
    so the page shows the whole value.
    """
    out = table.head(max_rows).copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(
                lambda v: "" if not np.isfinite(v) else float_format.format(v))
        elif pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].map({True: "yes", False: "no"})
        else:
            out[column] = out[column].map(lambda v: "" if v is None else str(v))
    if drop_empty_columns:
        # A column that is empty for every arm (no code SHA on a hand-loaded
        # file, say) is a column of nothing; dropping it leaves room for the
        # ones that carry information.
        keep = [c for c in out.columns
                if not (out[c].astype(str).str.strip() == "").all()]
        out = out[keep] if keep else out
    return out


#: Columns whose content is a long opaque string. They are allowed to wrap over
#: several lines but are given a smaller share of the page than their raw
#: length would suggest -- otherwise one path column squeezes out every column
#: a reader actually compares.
WIDE_COLUMNS = {"path", "message", "note", "input_dir", "description"}


def shorten_paths(table: pd.DataFrame, workspace=None) -> pd.DataFrame:
    """Make absolute paths workspace-relative so they fit without eliding."""
    if "path" not in table.columns or workspace is None:
        return table
    out = table.copy()
    prefix = str(workspace).rstrip("/") + "/"
    out["path"] = out["path"].astype(str).map(
        lambda v: v[len(prefix):] if v.startswith(prefix) else v)
    return out


def headline_table(residual_summaries: dict) -> pd.DataFrame:
    """The one table a reader looks at first: each arm's headline numbers.

    Wide (one row per arm) rather than long, because the comparison being made
    is between arms, and a reader should not have to pivot in their head.
    """
    from ..metrics import residuals as residuals_metrics

    rows = []
    for arm, summary in residual_summaries.items():
        if summary is None or summary.empty:
            continue
        indexed = summary.set_index("quantity")
        row = {"run": arm}
        # One number for a one-sided magnitude, two for a signed residual; the
        # split lives in metrics.residuals.headline_panels(), which the
        # resolution pages profile panel for panel, so the two cannot drift.
        for quantity, statistic, name, unit in residuals_metrics.headline_panels():
            if quantity not in indexed.index:
                continue
            word = residuals_metrics.STATISTIC_WORD[statistic]
            row[f"{word} {name} [{unit}]"] = float(indexed.loc[quantity][statistic])
        row["n"] = int(indexed["n"].max()) if "n" in indexed else 0
        rows.append(row)
    return pd.DataFrame(rows)


#: Signed residuals with a per-event pull against the bound: (residual column,
#: column of the per-event 1-sigma bound in crb_events, scale to the residual's unit).
PULLS = {"vertex_long": ("sigma_long", 1.0), "p_res_pct": ("sigma_p_rel", 100.0)}


def bound_table(residuals_by_arm: dict, bound, per_event=None) -> pd.DataFrame:
    """Each width metric: the Cramér–Rao bound, each arm's value, and the ratio.

    Every number is computed by the same statistic function from a residual table
    -- the bound's being draws from its per-event covariances -- so the ratio
    compares like with like.

    For the signed residuals, `per_event` (the crb_events table) adds a PULL: each
    event's residual divided by its own bound, summarised by a fitted Gaussian
    width. That is the sharper test -- 1.00 is an efficient estimator, event by
    event, with no averaging of easy events against hard ones.
    """
    from ..metrics import residuals as residuals_metrics, stats
    from ..plots.residual_plots import _bound_applies

    rows = []
    for quantity, statistic, name, unit in residuals_metrics.headline_panels():
        if not _bound_applies(quantity, statistic):
            continue
        fn = stats.STATISTICS[statistic]
        floor = float(fn(bound[quantity]))
        row = {"metric": f"{residuals_metrics.STATISTIC_WORD[statistic]} {name} [{unit}]",
               "bound": floor}
        for arm, residuals in residuals_by_arm.items():
            if residuals is None or residuals.empty or quantity not in residuals:
                continue
            value = float(fn(residuals[quantity]))
            row[arm] = value
            row[f"{arm} / bound"] = value / floor if floor > 0 else np.nan
            if per_event is not None and quantity in PULLS:
                column, scale = PULLS[quantity]
                m = residuals[["event", quantity]].merge(
                    per_event[["event", column]], on="event")
                row[f"{arm} pull σ"] = stats.gauss_sigma(
                    m[quantity] / (scale * m[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(residuals_by_arm: dict, reference: str, *,
                     quantities=("vertex_dist", "vertex_long", "vertex_trans",
                                 "angle_deg", "p_res_pct"),
                     seed: int = 0) -> pd.DataFrame:
    """Paired per-event differences against a reference arm, with CIs.

    The `excludes_zero` column is the honest verdict: with a few hundred events
    most differences do not exclude zero, and a table that reports the delta
    without saying so invites a conclusion the data cannot support.
    """
    from ..metrics import stats
    from ..metrics.residuals import match_events

    if reference not in residuals_by_arm:
        return pd.DataFrame()
    matched = match_events(residuals_by_arm)
    ref = matched[reference]
    rows = []
    for arm, residuals in matched.items():
        if arm == reference or residuals.empty:
            continue
        for quantity in quantities:
            if quantity not in residuals.columns or quantity not in ref.columns:
                continue
            result = stats.paired_difference(
                residuals[quantity].to_numpy(float),
                ref[quantity].to_numpy(float), seed=seed)
            rows.append({"run": arm, "reference": reference,
                         "quantity": quantity, **result})
    return pd.DataFrame(rows)


def provenance_table(arms) -> pd.DataFrame:
    """What produced each arm -- the thing that makes a result reproducible."""
    rows = []
    for arm in arms:
        meta = arm.meta or {}
        rows.append({
            "run": arm.label,
            "algorithm": arm.algorithm,
            "events": arm.n_events(),
            "format": meta.get("format", ""),
            "branch": meta.get("branch", ""),
            "code_sha": (meta.get("code_sha", "") or "")[:10],
            "dirty": meta.get("dirty", ""),
            "run": meta.get("run", ""),
            "where": meta.get("where", ""),
            "config": meta.get("config_digest", ""),
            "path": meta.get("path", ""),
        })
    return pd.DataFrame(rows)


def findings_table(findings) -> pd.DataFrame:
    from ..metrics.validate import to_frame
    return to_frame(findings)


def write_all(tables: dict, directory) -> dict:
    """Write every table to CSV; returns {name: path}."""
    directory = Path(directory)
    written = {}
    for name, table in tables.items():
        if table is None or (hasattr(table, "empty") and table.empty):
            continue
        written[name] = write_csv(table, directory / f"{name}.csv")
    return written


#: A table with many columns and few rows reads far better turned on its side:
#: "Runs compared" is naturally one row per *field* and one column per *run*,
#: not the reverse. Laid out the other way it needs a 5 pt font to fit a page.
TRANSPOSE_MIN_COLUMNS = 7
TRANSPOSE_MAX_ROWS = 6


def maybe_transpose(table: pd.DataFrame, label_column: str = None):
    """Transpose a wide, short table so it fits legibly. Returns (table, done)."""
    if (len(table.columns) < TRANSPOSE_MIN_COLUMNS
            or len(table) > TRANSPOSE_MAX_ROWS or len(table) == 0):
        return table, False
    label_column = label_column or table.columns[0]
    if label_column not in table.columns:
        return table, False
    out = table.set_index(label_column).T
    out.index.name = "field"
    return out.reset_index(), True


def compact_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Fold the summary into something a person can read on one page.

    The raw summary is 17 columns -- p68, gauss_mu, gauss_sigma, the legacy
    median and iqr68, and a low/high bound for each -- and every row leaves
    roughly half of them empty, because a one-sided quantity has no bias and a
    Gaussian one has no p68. Rendered as-is it needs an unreadable font and two
    pages.

    Each row is therefore reduced to the two numbers that quantity actually
    has: a BIAS (blank for a one-sided magnitude, which cannot have one) and a
    RESOLUTION -- the 68th percentile for one-sided quantities, the fitted sigma
    for Gaussian ones. That is exactly the convention the headline table uses,
    so the two agree by construction. The CSV keeps every column.
    """
    if table.empty:
        return table
    out = table.copy()

    def folded(stat):
        lo, hi = f"{stat}_lo", f"{stat}_hi"
        if stat not in out.columns:
            return [""] * len(out)
        if lo in out.columns and hi in out.columns:
            return [_with_interval(v, l, h)
                    for v, l, h in zip(out[stat], out[lo], out[hi])]
        return [_with_interval(v, np.nan, np.nan) for v in out[stat]]

    p68, mu, sigma = folded("p68"), folded("gauss_mu"), folded("gauss_sigma")
    kinds = out["kind"] if "kind" in out.columns else ["gaussian"] * len(out)
    one_sided = [str(k) == "one_sided" for k in kinds]

    out["bias"] = ["" if os else m for os, m in zip(one_sided, mu)]
    out["resolution"] = [p if os else sg
                         for os, p, sg in zip(one_sided, p68, sigma)]
    # `p_res` in MeV/c restates `p_res_pct`, which is what every number in this
    # campaign is quoted in.
    if "quantity" in out.columns:
        out = out[out["quantity"] != "p_res"]

    lead = [c for c in ("run", "quantity", "unit", "n") if c in out.columns]
    return out[lead + ["bias", "resolution"]]


def _with_interval(value, low, high) -> str:
    def fmt(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        return "" if not np.isfinite(v) else f"{v:.4g}"

    body = fmt(value)
    if not body:
        return ""
    lo, hi = fmt(low), fmt(high)
    return f"{body} [{lo}, {hi}]" if lo and hi else body
