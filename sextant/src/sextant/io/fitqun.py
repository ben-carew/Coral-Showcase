"""Read fiTQun reconstruction output into the canonical tables.

Two quite different files both count as "fiTQun output" here:

**Merged** -- what `merge_fitqun.py` produces for a run launched from this
workspace: flat TTrees already reduced to sub-event 0, with the e-/mu- NLLs
side by side. Cheap to read, but everything the merge did not carry through is
gone.

**Official** -- a `*_rfqn.root` straight from the production area, holding
fiTQun's own 137-branch `fiTQun` tree. Read directly, this gives every
sub-event, every hypothesis, the staged intermediate fits, the multi-ring and
pi0 fits, and the quantities that make an intermediate comparison possible at
all: `fq1rtotmu` (total predicted photoelectrons) against `fqtotq` (total
observed charge), and `fq0rnll` (the no-ring hypothesis).

That last pair is worth spelling out. Absolute NLL values are not comparable
between Coral and fiTQun -- different constant offsets, different unhit-PMT
subsampling -- but *differences* against each code's own zero-ring hypothesis
are, and so is the ratio of predicted to observed charge. Both are read here.

### Time convention

fiTQun reports `fq1rt0` in absolute WCSim time; Coral reports relative to the
trigger. The offset is ~950 ns for the HK productions. It is a parameter rather
than a constant because it is a property of the *sample*, and getting it wrong
shifts every t residual by a constant that is easy to mistake for a real bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import _root
from .schema import (FITQUN_HYPOTHESIS_INDEX, FITQUN_INDEX_TO_HYPOTHESIS,
                     HYP_ELECTRON, HYP_MUON, POINT_SELF, RING_LEADING,
                     STAGE_FINAL, conform, empty, unit_vector)

#: WCSim trigger offset. fiTQun t0 minus this gives Coral's convention.
DEFAULT_TIME_OFFSET_NS = 950.0

OFFICIAL_TREE = "fiTQun"
MERGED_TREE = "fits"


def is_fitqun_merged(path) -> bool:
    try:
        names = _root.tree_names(path)
        if MERGED_TREE not in names:
            return False
        return "nll_e" in _root.branches(path, MERGED_TREE)
    except Exception:
        return False


def merged_has_both_hypotheses(path) -> bool:
    """True when a merged file carries the e- and the mu- fit's kinematics separately
    (merge_fitqun.py since 2026-11-14); False = one hypothesis's kinematics on both rows."""
    try:
        return "p_e" in _root.branches(path, MERGED_TREE)
    except Exception:
        return False


def is_fitqun_official(path) -> bool:
    try:
        return OFFICIAL_TREE in _root.tree_names(path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Merged format (merge_fitqun.py)
# ---------------------------------------------------------------------------

def read_merged_fits(path, *, only_success: bool = False,
                     time_offset_ns: float = 0.0) -> pd.DataFrame:
    """The merged single-ring fits, one e- and one mu- row per event.

    Merges made since 2026-11-14 carry each hypothesis's own kinematics (the
    `*_e` / `*_mu` columns) and each row gets its own. OLDER merges hold ONE
    set of kinematics per event, those of the sample's own hypothesis (the muon
    fit on muon / numu samples, the electron fit elsewhere), and both rows
    share it: on such a file the e- row of a muon sample is the MUON fit
    labelled e-. `merged_has_both_hypotheses(path)` tells the two apart; for an
    old file, re-merge it or take the other hypothesis from the `stages` tree
    (seed values only) or from the official file.
    """
    df = _root.read_tree(path, MERGED_TREE)
    if df.empty:
        return empty("fits")
    per_hypothesis = "p_e" in df.columns and "p_mu" in df.columns

    def kinematics(suffix):
        theta = df["theta" + suffix].to_numpy(float)
        phi = df["phi" + suffix].to_numpy(float)
        success = df["success" + suffix].astype(bool).to_numpy()
        if "nhits" in df:
            success = success & (df["nhits"].to_numpy() > 0)
        return {
            "event": df["event"], "sub_event": 0, "ring": RING_LEADING,
            "nrings": 1, "stage": STAGE_FINAL, "success": success,
            "t": df["t" + suffix].to_numpy(float) - time_offset_ns,
            "x": df["x" + suffix], "y": df["y" + suffix], "z": df["z" + suffix],
            "dx": np.sin(theta) * np.cos(phi), "dy": np.sin(theta) * np.sin(phi),
            "dz": np.cos(theta), "p": df["p" + suffix], "dconv": np.nan,
        }

    # Present only in merges made after the intermediates were carried through
    # (merge_fitqun.py FIT_COLUMNS); older files simply have no totmu.
    def column(name):
        return df[name] if name in df.columns else np.nan

    frames = [
        pd.DataFrame({**kinematics("_e" if per_hypothesis else ""),
                      "hypothesis": HYP_ELECTRON, "nll": df["nll_e"],
                      "totmu": column("totmu_e")}),
        pd.DataFrame({**kinematics("_mu" if per_hypothesis else ""),
                      "hypothesis": HYP_MUON, "nll": df["nll_mu"],
                      "totmu": column("totmu_mu")}),
    ]
    out = pd.concat(frames, ignore_index=True)
    if only_success:
        out = out[out["success"]]
    return conform(out, "fits")


def read_merged_totals(path) -> pd.DataFrame:
    """Per-event observed totals and the zero-ring reference, from a merge.

    The same quantities `read_official` pulls from a production file. Empty for
    merges made before merge_fitqun.py carried them through.
    """
    columns = _root.branches(path, MERGED_TREE)
    if "totq" not in columns:
        return pd.DataFrame(columns=["event", "sub_event", "totq", "nhits",
                                     "nll_0r", "totmu_0r"])
    df = _root.read_tree(path, MERGED_TREE)
    return pd.DataFrame({
        "event": df["event"].astype("int64"), "sub_event": 0,
        "totq": df["totq"].astype(float),
        "nhits": df["nhits"].astype(float) if "nhits" in df else np.nan,
        "nll_0r": df["nll_0r"].astype(float) if "nll_0r" in df else np.nan,
        "totmu_0r": df["totmu_0r"].astype(float) if "totmu_0r" in df else np.nan,
    })


def read_merged_nll_terms(path) -> pd.DataFrame:
    """'total' and 'zero_ring' NLL per hypothesis, from a merge."""
    columns = _root.branches(path, MERGED_TREE)
    if "nll_0r" not in columns:
        return empty("nll_terms")
    df = _root.read_tree(path, MERGED_TREE)
    rows = []
    for hypothesis, key in ((HYP_ELECTRON, "nll_e"), (HYP_MUON, "nll_mu")):
        for name, values in (("total", df[key]), ("zero_ring", df["nll_0r"])):
            rows.append(pd.DataFrame({
                "event": df["event"], "sub_event": 0, "hypothesis": hypothesis,
                "point": POINT_SELF, "term": name, "value": values}))
    return conform(pd.concat(rows, ignore_index=True), "nll_terms")


def read_merged_stages(path, *, time_offset_ns: float = 0.0) -> pd.DataFrame:
    """The staged intermediate fits, from a merge's `stages` tree."""
    if not _root.has_tree(path, "stages"):
        return empty("fits")
    df = _root.read_tree(path, "stages")
    if df.empty:
        return empty("fits")
    dx, dy, dz = unit_vector(df["px"], df["py"], df["pz"])
    return conform(pd.DataFrame({
        "event": df["event"], "sub_event": df["sub_event"],
        "hypothesis": [FITQUN_INDEX_TO_HYPOTHESIS.get(int(v), "?")
                       for v in df["pid"]],
        "ring": RING_LEADING, "nrings": 1, "stage": df["stage"],
        "success": df["success"].astype(bool),
        "nll": df["nll"], "totmu": df["totmu"],
        "t": df["t"].to_numpy(float) - time_offset_ns,
        "x": df["x"], "y": df["y"], "z": df["z"],
        "dx": dx, "dy": dy, "dz": dz, "p": df["p"], "dconv": np.nan,
    }), "fits")


def read_merged_subevents(path, *, time_offset_ns: float = 0.0) -> pd.DataFrame:
    """(event, sub_event, t) for every sub-event fiTQun found, from a merge.

    The `subevents` tree (merge_fitqun.py since 2026-09-28) is exact. Older
    merges only have the staged fits, whose derived sub-event index is NOT in
    time order -- in ~40% of two-sub-event muon events the decay electron's fit
    comes first -- so from those only the set of times is used, re-indexed in
    time order.
    """
    columns = ["event", "sub_event", "t"]
    if _root.has_tree(path, "subevents"):
        df = _root.read_tree(path, "subevents")
        out = df[columns].copy()
        out["t"] = out["t"].astype(float) - time_offset_ns
        return out
    if not _root.has_tree(path, "stages"):
        return pd.DataFrame(columns=columns)
    df = _root.read_tree(path, "stages")
    df = df[df["pid"] == FITQUN_HYPOTHESIS_INDEX[HYP_ELECTRON]]
    if df.empty:
        return pd.DataFrame(columns=columns)
    last = (df.sort_values("stage")
              .drop_duplicates(["event", "sub_event"], keep="last")
              .sort_values(["event", "t"]))
    out = pd.DataFrame({"event": last["event"].astype("int64"),
                        "t": last["t"].astype(float) - time_offset_ns})
    out["sub_event"] = out.groupby("event").cumcount().astype("int32")
    return out[columns].reset_index(drop=True)


def read_merged_prefits(path, *, time_offset_ns: float = 0.0) -> pd.DataFrame:
    """fiTQun's prefit, present only in runs made with a *_prefitonly override."""
    cols = _root.branches(path, MERGED_TREE)
    if "prefit_x" not in cols:
        return empty("fits")
    df = _root.read_tree(path, MERGED_TREE)
    return conform(pd.DataFrame({
        "event": df["event"], "sub_event": 0, "hypothesis": "(prefit)",
        "ring": RING_LEADING, "nrings": 1, "stage": STAGE_FINAL,
        "success": df["nhits"].to_numpy() > 0 if "nhits" in df else True,
        "nll": np.nan, "totmu": np.nan,
        "t": df["prefit_t"].to_numpy(float) - time_offset_ns,
        "x": df["prefit_x"], "y": df["prefit_y"], "z": df["prefit_z"],
        "dx": np.nan, "dy": np.nan, "dz": np.nan, "p": np.nan, "dconv": np.nan,
    }), "fits")


def read_merged_multiring(path) -> pd.DataFrame:
    if not _root.has_tree(path, "multiringfits"):
        return empty("fits")
    from .coral import read_multiring          # identical schema by construction
    return read_multiring(path)


def read_merged_pi0(path) -> pd.DataFrame:
    if not _root.has_tree(path, "pi0fits"):
        return empty("fits")
    from .coral import read_pi0
    return read_pi0(path)


# ---------------------------------------------------------------------------
# Official format (*_rfqn.root, fiTQun's own tree)
# ---------------------------------------------------------------------------

def _flat(values):
    """Awkward -> flat numpy, tolerating both jagged and rectangular arrays."""
    arr = np.asarray(values.tolist(), dtype=object) if values.dtype == object else values
    return arr


def _iter_subevents(nse):
    for event_index, n in enumerate(nse):
        for sub in range(int(n)):
            yield event_index, sub


def read_official(path, *, file_index=None, stride=None, event_ids=None,
                  time_offset_ns: float = DEFAULT_TIME_OFFSET_NS,
                  hypotheses=(HYP_ELECTRON, HYP_MUON),
                  include_stages: bool = True) -> dict:
    """Every canonical table an official fiTQun file can supply.

    Returns a dict with 'fits', 'prefits', 'multiring', 'pi0', 'nll_terms' and
    'event_totals'. Global event ids come from `event_ids` if given, else from
    `file_index * stride + entry`.
    """
    import uproot

    with uproot.open(path) as fh:
        tree = fh[OFFICIAL_TREE]
        available = set(tree.keys())

        def get(name):
            return tree[name].array(library="np") if name in available else None

        nse = get("fqnse")
        if nse is None:
            return {k: empty("fits") for k in ("fits", "prefits", "multiring", "pi0")}
        n_events = len(nse)

        if event_ids is not None:
            ids = np.asarray(event_ids, dtype=np.int64)
        elif file_index is not None and stride is not None:
            ids = file_index * stride + np.arange(n_events, dtype=np.int64)
        else:
            ids = np.arange(n_events, dtype=np.int64)

        wanted = {h: FITQUN_HYPOTHESIS_INDEX[h] for h in hypotheses
                  if h in FITQUN_HYPOTHESIS_INDEX}

        out = {}
        out["fits"] = _official_fits(get, nse, ids, wanted, time_offset_ns)
        out["prefits"] = _official_prefits(get, nse, ids, time_offset_ns)
        out["multiring"] = _official_multiring(get, ids, time_offset_ns)
        out["pi0"] = _official_pi0(get, ids, time_offset_ns)
        out["event_totals"] = _official_totals(get, nse, ids)
        out["nll_terms"] = _official_nll_terms(get, nse, ids, wanted)
        out["stages"] = (_official_stages(get, ids, wanted, time_offset_ns)
                         if include_stages else empty("fits"))
    return out


def _official_fits(get, nse, ids, wanted, time_offset_ns) -> pd.DataFrame:
    pos, dirn = get("fq1rpos"), get("fq1rdir")
    mom, t0 = get("fq1rmom"), get("fq1rt0")
    nll, pcflg = get("fq1rnll"), get("fq1rpcflg")
    totmu, dconv = get("fq1rtotmu"), get("fq1rdconv")
    if pos is None or nll is None:
        return empty("fits")

    rows = []
    for ev, sub in _iter_subevents(nse):
        for label, hyp in wanted.items():
            try:
                p3 = pos[ev][sub][hyp]
                d3 = dirn[ev][sub][hyp]
            except (IndexError, KeyError):
                continue
            rows.append({
                "event": ids[ev], "sub_event": sub, "hypothesis": label,
                "ring": RING_LEADING, "nrings": 1, "stage": STAGE_FINAL,
                "success": bool(pcflg[ev][sub][hyp] >= 0) if pcflg is not None else True,
                "nll": float(nll[ev][sub][hyp]),
                "totmu": float(totmu[ev][sub][hyp]) if totmu is not None else np.nan,
                "t": float(t0[ev][sub][hyp]) - time_offset_ns,
                "x": float(p3[0]), "y": float(p3[1]), "z": float(p3[2]),
                "dx": float(d3[0]), "dy": float(d3[1]), "dz": float(d3[2]),
                "p": float(mom[ev][sub][hyp]),
                "dconv": float(dconv[ev][sub][hyp]) if dconv is not None else np.nan,
            })
    if not rows:
        return empty("fits")
    df = pd.DataFrame(rows)
    df["dx"], df["dy"], df["dz"] = unit_vector(df["dx"], df["dy"], df["dz"])
    return conform(df, "fits")


def _official_prefits(get, nse, ids, time_offset_ns) -> pd.DataFrame:
    pos, t0 = get("fqtwnd_prftpos"), get("fqtwnd_prftt0")
    if pos is None or t0 is None:
        return empty("fits")
    rows = []
    for ev, sub in _iter_subevents(nse):
        try:
            p3 = pos[ev][sub]
            tval = t0[ev][sub]
        except (IndexError, KeyError):
            continue
        rows.append({
            "event": ids[ev], "sub_event": sub, "hypothesis": "(prefit)",
            "ring": RING_LEADING, "nrings": 1, "stage": STAGE_FINAL,
            "success": True, "nll": np.nan, "totmu": np.nan,
            "t": float(tval) - time_offset_ns,
            "x": float(p3[0]), "y": float(p3[1]), "z": float(p3[2]),
            "dx": np.nan, "dy": np.nan, "dz": np.nan, "p": np.nan, "dconv": np.nan,
        })
    return conform(pd.DataFrame(rows), "fits") if rows else empty("fits")


#: fiTQun stacks its multi-ring candidates in the fqmr* branches as it searches
#: from one ring upward and refines. Per the fiTQun README ("Multi-Ring Fits")
#: index 0 is the FINAL best fit, so that is the only one to read: taking every
#: candidate reports a stack of attempts as if they were rings, and an event
#: comes out with 56 "rings" instead of two.
BEST_MULTIRING_FIT = 0


def _official_multiring(get, ids, time_offset_ns) -> pd.DataFrame:
    nfit = get("fqnmrfit")
    if nfit is None:
        return empty("fits")
    nring, pcflg, nll = get("fqmrnring"), get("fqmrpcflg"), get("fqmrnll")
    pid, mom, t0 = get("fqmrpid"), get("fqmrmom"), get("fqmrt0")
    pos, dirn = get("fqmrpos"), get("fqmrdir")
    totmu, dconv = get("fqmrtotmu"), get("fqmrdconv")
    best = BEST_MULTIRING_FIT
    rows = []
    for ev in range(len(nfit)):
        if int(nfit[ev]) <= 0:
            continue
        try:
            n = int(nring[ev][best])
        except (IndexError, KeyError):
            continue
        if n <= 0:
            continue
        # Slice to n: the per-ring arrays are fixed-width (6 slots) and the
        # entries past nrings are unset padding, not rings.
        for ring in range(n):
            p3, d3 = pos[ev][best][ring], dirn[ev][best][ring]
            rows.append({
                "event": ids[ev], "sub_event": 0,
                "hypothesis": FITQUN_INDEX_TO_HYPOTHESIS.get(
                    int(pid[ev][best][ring]), "?"),
                "ring": ring, "nrings": n, "stage": STAGE_FINAL,
                "success": bool(pcflg[ev][best] >= 0),
                "nll": float(nll[ev][best]),
                "totmu": float(totmu[ev][best]) if totmu is not None else np.nan,
                "t": float(t0[ev][best][ring]) - time_offset_ns,
                "x": float(p3[0]), "y": float(p3[1]), "z": float(p3[2]),
                "dx": float(d3[0]), "dy": float(d3[1]), "dz": float(d3[2]),
                "p": float(mom[ev][best][ring]),
                "dconv": float(dconv[ev][best][ring]) if dconv is not None else np.nan,
            })
    if not rows:
        return empty("fits")
    df = pd.DataFrame(rows)
    df["dx"], df["dy"], df["dz"] = unit_vector(df["dx"], df["dy"], df["dz"])
    return conform(df, "fits")


def _official_pi0(get, ids, time_offset_ns) -> pd.DataFrame:
    nll = get("fqpi0nll")
    if nll is None:
        return empty("fits")
    pos, t0, pcflg = get("fqpi0pos"), get("fqpi0t0"), get("fqpi0pcflg")
    rows = []
    for ev in range(len(nll)):
        for ifit in range(len(np.atleast_1d(nll[ev]))):
            for ring, tag in enumerate(("1", "2")):
                d3 = get(f"fqpi0dir{tag}")[ev][ifit]
                p3 = pos[ev][ifit]
                rows.append({
                    "event": ids[ev], "sub_event": 0,
                    # fit 0 = free mass, fit 1 = pi0-mass constrained, matching
                    # the DoPi0Fit/DoPi0Fit2 pair Coral mirrors.
                    "hypothesis": "pi0-free" if ifit == 0 else "pi0-constrained",
                    "ring": ring, "nrings": 2, "stage": STAGE_FINAL,
                    "success": bool(pcflg[ev][ifit] >= 0),
                    "nll": float(nll[ev][ifit]),
                    "totmu": float(get("fqpi0totmu")[ev][ifit]),
                    "t": float(t0[ev][ifit]) - time_offset_ns,
                    "x": float(p3[0]), "y": float(p3[1]), "z": float(p3[2]),
                    "dx": float(d3[0]), "dy": float(d3[1]), "dz": float(d3[2]),
                    "p": float(get(f"fqpi0mom{tag}")[ev][ifit]),
                    "dconv": float(get(f"fqpi0dconv{tag}")[ev][ifit]),
                })
    if not rows:
        return empty("fits")
    df = pd.DataFrame(rows)
    df["dx"], df["dy"], df["dz"] = unit_vector(df["dx"], df["dy"], df["dz"])
    return conform(df, "fits")


def _official_stages(get, ids, wanted, time_offset_ns) -> pd.DataFrame:
    """fiTQun's staged intermediate 1R fits (`fqtest1r*`).

    fiTQun records the result of each fit stage for each hypothesis it tried, so
    the trajectory of an estimate through the fit can be compared against
    Coral's [step_N] sequence rather than only the endpoints.

    The sub-event index is *derived* rather than read from `fqtest1rse`, which
    is wrong in the production files: for a 3-sub-event event it records
    [0,0,0, 0,0,0, 1,1,1] instead of [0,0,0, 1,1,1, 2,2,2], which would give two
    different fits the same (event, sub_event, hypothesis, stage) key and make
    any join against them ambiguous. The entries are laid out in blocks of one
    per hypothesis per sub-event, so the n-th repeat of a given (stage,
    hypothesis) pair within an event is its sub-event index. Checked against
    `fqnse` over a full file: derived is right for 101/101 events, the recorded
    branch for 78/101.
    """
    n = get("fqtestn1r")
    if n is None:
        return empty("fits")
    stage, pid = get("fqtest1rstage"), get("fqtest1rpid")
    pcflg, mom, t0 = get("fqtest1rpcflg"), get("fqtest1rmom"), get("fqtest1rt0")
    nll, pos, dirn = get("fqtest1rnll"), get("fqtest1rpos"), get("fqtest1rdir")
    totmu = get("fqtest1rtotmu")
    by_index = {v: k for k, v in FITQUN_HYPOTHESIS_INDEX.items()}
    keep = set(wanted.values())
    rows = []
    for ev in range(len(n)):
        seen = {}
        for i in range(int(n[ev])):
            hyp_index = int(pid[ev][i])
            key = (int(stage[ev][i]), hyp_index)
            sub_event = seen.get(key, 0)
            seen[key] = sub_event + 1
            if keep and hyp_index not in keep:
                continue
            p3, d3 = pos[ev][i], dirn[ev][i]
            rows.append({
                "event": ids[ev],
                "sub_event": sub_event,
                "hypothesis": by_index.get(hyp_index, "?"),
                "ring": RING_LEADING, "nrings": 1,
                "stage": int(stage[ev][i]),
                "success": bool(pcflg[ev][i] >= 0),
                "nll": float(nll[ev][i]),
                "totmu": float(totmu[ev][i]) if totmu is not None else np.nan,
                "t": float(t0[ev][i]) - time_offset_ns,
                "x": float(p3[0]), "y": float(p3[1]), "z": float(p3[2]),
                "dx": float(d3[0]), "dy": float(d3[1]), "dz": float(d3[2]),
                "p": float(mom[ev][i]), "dconv": np.nan,
            })
    if not rows:
        return empty("fits")
    df = pd.DataFrame(rows)
    df["dx"], df["dy"], df["dz"] = unit_vector(df["dx"], df["dy"], df["dz"])
    return conform(df, "fits")


def _official_totals(get, nse, ids) -> pd.DataFrame:
    """Per-sub-event observed totals and the zero-ring hypothesis.

    `totq` vs a fit's `totmu` is the charge-model normalisation check, and
    `nll_0r` is the only absolute reference point an NLL has: differences
    against it survive the constant offsets that make raw NLLs incomparable
    between codes.
    """
    totq, nhit = get("fqtotq"), get("fqnhitpmt")
    nll0, totmu0 = get("fq0rnll"), get("fq0rtotmu")
    rows = []
    for ev, sub in _iter_subevents(nse):
        def at(arr):
            try:
                return float(arr[ev][sub]) if arr is not None else np.nan
            except (IndexError, KeyError, TypeError):
                return np.nan
        rows.append({
            "event": int(ids[ev]), "sub_event": sub,
            "totq": at(totq), "nhits": at(nhit),
            "nll_0r": at(nll0), "totmu_0r": at(totmu0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["event", "sub_event", "totq", "nhits", "nll_0r", "totmu_0r"])


def _official_nll_terms(get, nse, ids, wanted) -> pd.DataFrame:
    """fiTQun reports one total NLL per hypothesis, plus the 0-ring reference.

    No four-way split exists on the fiTQun side, so the table carries 'total'
    and 'zero_ring' only. Sextant's per-term comparison therefore compares
    Coral's decomposition against fiTQun's total, and the *difference from the
    zero-ring hypothesis*, which both codes can supply.
    """
    nll, nll0 = get("fq1rnll"), get("fq0rnll")
    if nll is None:
        return empty("nll_terms")
    rows = []
    for ev, sub in _iter_subevents(nse):
        for label, hyp in wanted.items():
            try:
                value = float(nll[ev][sub][hyp])
            except (IndexError, KeyError):
                continue
            rows.append({"event": ids[ev], "sub_event": sub, "hypothesis": label,
                         "point": POINT_SELF, "term": "total", "value": value})
            if nll0 is not None:
                try:
                    rows.append({"event": ids[ev], "sub_event": sub,
                                 "hypothesis": label, "point": POINT_SELF,
                                 "term": "zero_ring", "value": float(nll0[ev][sub])})
                except (IndexError, KeyError):
                    pass
    return conform(pd.DataFrame(rows), "nll_terms") if rows else empty("nll_terms")


def read_official_many(paths, production=None, *, sample=None, limit=None,
                       **kwargs) -> dict:
    """Read and concatenate many official files, assigning global event ids.

    `production`/`sample` supply the file-index parse and the numbering stride,
    which is what keeps ids consistent with everything else in the workspace.
    """
    from .. import productions as prod_registry

    spec = None
    stride = None
    if production is not None:
        prod = prod_registry.get(production)
        stride = prod.stride
        if sample is not None:
            spec = prod.sample(sample)

    collected = {}
    for path in list(paths)[:limit] if limit else list(paths):
        index = spec.file_index(path) if spec is not None else 0
        tables = read_official(path, file_index=index, stride=stride, **kwargs)
        for key, table in tables.items():
            collected.setdefault(key, []).append(table)
    return {k: (pd.concat(v, ignore_index=True) if v else empty("fits"))
            for k, v in collected.items()}
