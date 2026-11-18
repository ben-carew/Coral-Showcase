"""Sub-eventing: did the reconstruction find each prompt and delayed track?

The HK validation spec asks for the **efficiency and purity of sub-eventing for
the prompt and the secondary tracks**. A sub-event is a time cluster of hits
that the fitter then reconstructs on its own; the true ones are the prompt
interaction and every decay electron (`truth_subevents`).

### Matching

Reconstructed and true sub-events are matched **by time**, one to one, per
event: pairs are taken in order of increasing |t_reco - t_true| and accepted
while both are free and within `DEFAULT_WINDOW_NS` -- except that the true
prompt claims first. A muon that decays within ~30 ns leaves two true
sub-events that no finder can separate; the one reco sub-event is the prompt's
light, and a pure nearest-time match used to hand it to the decay electron and
score the prompt as missed (v1.12.29 held-out muons: 41 of Coral's 85 "missed"
prompts, 19 of fiTQun's 55). Such a decay now counts as not found. Time rather than index,
because neither code's index is a physical label -- fiTQun's staged fits come
out of order in ~40% of two-sub-event muon events, and a spurious split of the
prompt light shifts every later index by one. The window: on v1.12.29 held-out
muons ~90% of the nearest reconstructed sub-events lie within 10-60 ns of the
true time for both codes, with a long flat tail beyond it (a sub-event 100 ns
away is a different time cluster, not the same one badly timed).

### Definitions

    prompt efficiency   true prompts matched / true prompts
    decay efficiency    true decay electrons matched / true decay electrons
                        (only `recorded` ones: a decay that fired no WCSim
                        trigger left no hits for any fitter to find)
    prompt purity       earliest reco sub-events matched to the true prompt /
                        earliest reco sub-events
    delayed purity      later reco sub-events matched to a true decay electron /
                        later reco sub-events
    purity              matched reco sub-events / reco sub-events

"Earliest" and "later" are the reconstruction's own roles -- what an analysis
would take as the neutrino interaction and as its decay-electron tags -- so the
two purities are exactly the two mistakes an analysis would inherit.

### Which events

Only events whose prompt interaction fired a WCSim trigger (`recorded`) are
scored: in the others the first recorded light is a decay electron's or there is
none, and no sub-event finder can recover a prompt that left no hits. An event
absent from an arm's output counts as zero reconstructed sub-events for that
arm (Coral writes nothing for an event with no sub-event). The event set is
therefore the union over the arms, restricted to events with a recorded prompt,
and every arm is scored on that same set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.geometry import dwall as _dwall, towall as _towall
from ..io.schema import HYP_ELECTRON, STAGE_FINAL
from . import stats

#: |t_reco - t_true| above which a reconstructed sub-event is not that true one.
DEFAULT_WINDOW_NS = 50.0

#: Classes of true sub-event, as reported.
PROMPT, DECAY = "prompt", "decay"


def reco_subevents(arm) -> pd.DataFrame:
    """(event, sub_event, t): one row per reconstructed sub-event of an arm.

    A reader that knows the sub-event list directly (`arm.subevents`) wins.
    Otherwise the per-sub-event prefit, which is seeded at the time the
    sub-event finder chose and so IS the sub-eventing output; then the final e-
    single-ring fit (both codes run it on every sub-event) for any sub-event
    without a prefit.
    """
    explicit = getattr(arm, "subevents", None)
    if explicit is not None and not explicit.empty:
        return explicit[["event", "sub_event", "t"]].reset_index(drop=True)
    frames = []
    if not arm.prefits.empty:
        frames.append(arm.prefits[["event", "sub_event", "t"]])
    fits = arm.fits
    if not fits.empty:
        final = fits[fits["stage"] == STAGE_FINAL]
        e_fits = final[final["hypothesis"] == HYP_ELECTRON]
        frames.append((e_fits if not e_fits.empty else final)
                      [["event", "sub_event", "t"]])
    if not frames:
        return pd.DataFrame(columns=["event", "sub_event", "t"])
    out = (pd.concat(frames, ignore_index=True)
             .dropna(subset=["t"])
             .drop_duplicates(["event", "sub_event"], keep="first"))
    return out.sort_values(["event", "sub_event"]).reset_index(drop=True)


def arm_events(arm) -> set:
    """Events an arm has any output for."""
    events = set()
    for table in (arm.fits, arm.prefits, getattr(arm, "subevents", None)):
        if table is not None and not table.empty:
            events.update(table["event"].to_numpy().tolist())
    return events


def _geometry(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["dwall"] = _dwall(out["x"], out["y"], out["z"])
    out["towall"] = _towall(out["x"], out["y"], out["z"],
                            out["dx"], out["dy"], out["dz"])
    return out


def match(true: pd.DataFrame, reco: pd.DataFrame, events, *,
          window_ns: float = DEFAULT_WINDOW_NS) -> tuple:
    """(per_true, per_reco) tables for one arm on the given events.

    `true` is `truth_subevents` (already restricted to what counts), `reco` the
    arm's `reco_subevents`. Each true row gains `matched` and `dt`; each reco
    row gains its time-ordered `order` (0 = earliest), `matched` and the class
    of the true sub-event it matched.
    """
    events = np.asarray(sorted(events), dtype=np.int64)
    true = _geometry(true[true["event"].isin(events)]).reset_index(drop=True)
    reco = (reco[reco["event"].isin(events)]
            .sort_values(["event", "t"]).reset_index(drop=True))
    reco["order"] = reco.groupby("event").cumcount()

    true_matched = np.zeros(len(true), bool)
    true_dt = np.full(len(true), np.nan)
    reco_match = np.full(len(reco), -1, dtype=np.int64)

    true_by_event = true.groupby("event").indices
    reco_by_event = reco.groupby("event").indices
    t_true = true["t"].to_numpy(float)
    t_reco = reco["t"].to_numpy(float)
    is_prompt = true["is_prompt"].to_numpy(bool)
    for event, ti in true_by_event.items():
        ri = reco_by_event.get(event)
        if ri is None:
            continue
        dt = t_reco[ri][None, :] - t_true[ti][:, None]
        # prompt pairs first, each class in order of |dt|
        decay_rank = np.repeat(~is_prompt[ti], len(ri)).astype(int)
        order = np.lexsort((np.abs(dt).ravel(), decay_rank))
        used_t, used_r = set(), set()
        for flat in order:
            a, b = divmod(int(flat), len(ri))
            if abs(dt[a, b]) > window_ns:
                continue
            if a in used_t or b in used_r:
                continue
            used_t.add(a)
            used_r.add(b)
            true_matched[ti[a]] = True
            true_dt[ti[a]] = dt[a, b]
            reco_match[ri[b]] = ti[a]

    per_true = true.assign(matched=true_matched, dt=true_dt,
                           kind=np.where(true["is_prompt"], PROMPT, DECAY))
    matched_kind = np.where(reco_match >= 0,
                            per_true["kind"].to_numpy(object)[np.maximum(reco_match, 0)],
                            None)
    per_reco = reco.assign(matched=reco_match >= 0, matched_kind=matched_kind)
    return per_true, per_reco


def _prompt_context(true: pd.DataFrame) -> pd.DataFrame:
    """Per event, the prompt's momentum and geometry: what purity is binned by."""
    prompt = _geometry(true[true["is_prompt"]])
    return (prompt[["event", "p", "dwall", "towall"]]
            .drop_duplicates("event").set_index("event"))


def build(arms, truth, *, window_ns: float = DEFAULT_WINDOW_NS,
          recorded_only: bool = True) -> dict:
    """{arm label: (per_true, per_reco)} on the shared event set, or {}."""
    true = truth.subevents
    if true is None or true.empty:
        return {}
    n_unrecorded_prompt = 0
    if recorded_only:
        prompts = true[true["is_prompt"]]
        live = set(prompts.loc[prompts["recorded"], "event"].to_numpy().tolist())
        n_unrecorded_prompt = int((~prompts["recorded"]).sum())
        true = true[true["recorded"] & true["event"].isin(live)]
    # rates are paired: only events every arm has output for, so an arm that lost
    # a job to the straggler cut is not charged for its sub-events. An event an arm
    # has no output for is reported separately ("events without output").
    truth_events = set(true["event"].to_numpy().tolist())
    seen = {arm.label: arm_events(arm) & truth_events for arm in arms}
    any_output = set().union(*seen.values())
    universe = set.intersection(*seen.values()) if seen else set()
    if not universe:
        return {}
    context = _prompt_context(true)
    out = {}
    for arm in arms:
        per_true, per_reco = match(true, reco_subevents(arm), universe,
                                   window_ns=window_ns)
        per_reco = per_reco.join(context, on="event")
        per_true.attrs["window_ns"] = per_reco.attrs["window_ns"] = window_ns
        per_true.attrs["n_events"] = len(universe)
        per_true.attrs["n_unrecorded_prompt"] = n_unrecorded_prompt
        per_true.attrs["no_output"] = np.array([e not in seen[arm.label] for e in sorted(any_output)])
        out[arm.label] = (per_true, per_reco)
    return out


def rates(per_true: pd.DataFrame, per_reco: pd.DataFrame) -> dict:
    """{metric name: boolean array whose mean is that rate}."""
    prompt = per_true[per_true["kind"] == PROMPT]
    decay = per_true[per_true["kind"] == DECAY]
    first = per_reco[per_reco["order"] == 0]
    later = per_reco[per_reco["order"] > 0]
    return {
        "prompt efficiency": prompt["matched"].to_numpy(),
        "decay efficiency": decay["matched"].to_numpy(),
        "prompt purity": (first["matched_kind"] == PROMPT).to_numpy(),
        "delayed purity": (later["matched_kind"] == DECAY).to_numpy(),
        "purity": per_reco["matched"].to_numpy(),
    }


def summary(tables: dict) -> pd.DataFrame:
    """One row per (arm, metric): the rate in %, its 68% Wilson interval, k/n."""
    rows = []
    for label, (per_true, per_reco) in tables.items():
        for metric, mask in rates(per_true, per_reco).items():
            rows.append({"arm": label, "metric": metric, **stats.rate(mask)})
        n_true = per_true.groupby("event").size()
        n_reco = per_reco.groupby("event").size().reindex(n_true.index, fill_value=0)
        rows.append({"arm": label, "metric": "exact sub-event count",
                     **stats.rate((n_true == n_reco).to_numpy())})
        if "no_output" in per_true.attrs:
            rows.append({"arm": label, "metric": "events without output",
                         **stats.rate(per_true.attrs["no_output"])})
    table = pd.DataFrame(rows)
    return table[["arm", "metric", "value", "ci_lo", "ci_hi", "k", "n"]]


def count_table(tables: dict) -> pd.DataFrame:
    """Events by (true, reconstructed) sub-event count, per arm."""
    rows = []
    for label, (per_true, per_reco) in tables.items():
        n_true = per_true.groupby("event").size()
        n_reco = per_reco.groupby("event").size().reindex(n_true.index, fill_value=0)
        counts = pd.crosstab(n_true.clip(upper=4), n_reco.clip(upper=4))
        for t, row in counts.iterrows():
            for r, value in row.items():
                rows.append({"arm": label, "n_true": int(t), "n_reco": int(r),
                             "events": int(value)})
    return pd.DataFrame(rows)


#: Binning for the profiles: decay electrons span 0-53 MeV/c, the prompt the
#: sample's momentum range; geometry is shared.
PROFILE_BINS = {
    "p": {PROMPT: None, DECAY: np.linspace(0, 60, 7)},
    "dwall": np.linspace(0, 3000, 11),
    "towall": np.linspace(0, 7000, 11),
}
MIN_EVENTS_PER_BIN = stats.MIN_RATE_EVENTS


def _edges(values, preset):
    if preset is not None:
        return np.asarray(preset, float)
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 * MIN_EVENTS_PER_BIN:
        return np.array([])
    return np.unique(np.quantile(finite, np.linspace(0, 1, 9)))


def profiles(tables: dict) -> pd.DataFrame:
    """Every efficiency and purity against p, dwall and towall, long format.

    Efficiencies are binned by the true sub-event's own momentum and geometry
    (a decay electron's, not its muon's); purities by the event's prompt.
    """
    rows = []
    for label, (per_true, per_reco) in tables.items():
        series = [
            ("prompt efficiency", per_true[per_true["kind"] == PROMPT], "matched",
             PROMPT),
            ("decay efficiency", per_true[per_true["kind"] == DECAY], "matched",
             DECAY),
            ("purity", per_reco, "matched", PROMPT),
        ]
        for metric, table, column, kind in series:
            if table.empty:
                continue
            for versus, preset in PROFILE_BINS.items():
                preset = preset.get(kind) if isinstance(preset, dict) else preset
                edges = _edges(table[versus], preset)
                if edges.size < 2:
                    continue
                prof = stats.rate_profile(table[versus], table[column], edges)
                if not prof.empty:
                    rows.append(prof.assign(arm=label, metric=metric,
                                            versus=versus))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    lead = ["arm", "metric", "versus", "low", "high", "centre", "value",
            "ci_lo", "ci_hi", "k", "n"]
    return out[lead]
