"""The canonical tables every reader produces, whatever the source format.

Sextant's one structural idea is that *nothing downstream of `io` knows which
algorithm produced a number*. A Coral HDF5-merged ROOT file, a fiTQun merge, an
official `*_rfqn.root` straight from the production area and a WCSim truth file
all get flattened into the same handful of DataFrames, and every metric, plot
and table is written once against those.

Six tables, all long-format, all keyed on a global integer `event`:

`fits`
    One row per (event, sub_event, hypothesis, ring, stage). Covers everything
    from a single-ring e- fit to ring 3 of a 6-ring fit to the intermediate
    result of fit stage 1 -- the distinction is in the columns, not in the table.
    A single-ring final fit is simply ring=0, nrings=1, stage=STAGE_FINAL.

`truth_rings`
    One row per true Cherenkov ring. For a particle gun that is one row per
    event; for a beam event it is every primary final-state particle, with
    `visible` marking the ones actually above Cherenkov threshold.

`truth_events`
    One row per event: interaction topology, true vertex, hit/charge totals and
    the geometry variables (towall/dwall) every resolution plot bins against.

`truth_subevents`
    One row per true sub-event: the prompt interaction, then every delayed
    decay product (a muon's decay electron) in time order. What sub-eventing
    efficiency and purity are scored against. Optional -- a truth file built
    before 2026-09-28 has none, and the sub-event section is then skipped.

`nll_terms`
    One row per (event, sub_event, hypothesis, point, term). The decomposition
    of a likelihood into its named pieces, evaluated at a named parameter point.
    Long format because Coral splits its NLL four ways and fiTQun does not split
    it at all -- a wide table would be mostly holes.

`pmt_predictions`
    One row per (event, sub_event, hypothesis, point, pmt). The per-PMT
    predicted light and time-residual behind those sums. This is the only table
    that is big, so it is always produced for an explicit, small event list.

The `point` column in the last two is what makes an intermediate comparison
mean anything. Predicted charges evaluated at each algorithm's *own* fitted
vertex differ mostly because the vertices differ; evaluated at a *common* point
(see `sextant.probe.points`) they differ only because the charge models differ,
which is the question actually being asked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

#: `stage` value for a converged, final fit result. Intermediate stages are
#: 0, 1, 2, ... in execution order (Coral's [step_N], fiTQun's fqtest1rstage).
STAGE_FINAL = -1

#: `ring` value used by single-ring fits and by the leading ring of a multi-ring
#: fit. Rings are ordered by descending reconstructed momentum.
RING_LEADING = 0

#: Named parameter points at which intermediates can be evaluated. Anything else
#: is allowed too (a `point` is just a label), but these three are what the
#: standard comparison uses.
POINT_TRUTH = "truth"       # the MC truth vertex/direction/momentum
POINT_SELF = "self"         # each algorithm's own fitted parameters
POINT_REF = "ref"           # the reference arm's fitted parameters

#: Particle-hypothesis labels. Both algorithms fit a *hypothesis*, which is not
#: necessarily the true particle; keeping them as strings avoids the recurring
#: confusion between Coral's int8 particle_code and fiTQun's 1R index.
HYP_ELECTRON = "e-"
HYP_MUON = "mu-"

#: Coral's int8 `particle_code` branch (see merge_coral.py).
CORAL_PARTICLE_CODES = {0: HYP_ELECTRON, 1: HYP_MUON, 3: "pi+"}

#: fiTQun's 1R hypothesis index (fq1r* arrays are [subevent][hypothesis][...]).
FITQUN_HYPOTHESIS_INDEX = {
    "gamma": 0, HYP_ELECTRON: 1, HYP_MUON: 2, "pi+": 3, "K+": 4, "p": 5,
    "conv-gamma": 6,
}
FITQUN_INDEX_TO_HYPOTHESIS = {v: k for k, v in FITQUN_HYPOTHESIS_INDEX.items()}

#: PDG code -> the Cherenkov ring a particle actually makes. Photons and pi0s
#: are reconstructed as electron-like rings, so that is how they are labelled.
PDG_TO_RING_TYPE = {
    11: HYP_ELECTRON, -11: HYP_ELECTRON, 22: HYP_ELECTRON,
    13: HYP_MUON, -13: HYP_MUON,
    211: "pi+", -211: "pi+", 2212: "p", 321: "K+", -321: "K+",
}

#: Rest masses in MeV, for the Cherenkov-threshold calculation.
PDG_MASS_MEV = {
    11: 0.511, -11: 0.511, 22: 0.0, 13: 105.658, -13: 105.658,
    211: 139.570, -211: 139.570, 111: 134.977,
    2212: 938.272, 2112: 939.565, 321: 493.677, -321: 493.677,
}

#: |PDG| of particles that carry electric charge (neutrals make no ring
#: themselves; pi0 and gamma are handled separately as they convert).
CHARGED_PDGS = {11, 13, 211, 321, 2212, 15}

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

# (name, dtype, description). The dtype is what `empty()` and `conform()` cast
# to, so a reader can build columns in whatever order is convenient.
FITS_SCHEMA = [
    ("event", "int64", "global event id (file_index * stride + entry)"),
    ("sub_event", "int32", "trigger / sub-event index within the event"),
    ("hypothesis", "object", "particle hypothesis fitted ('e-', 'mu-', ...)"),
    ("ring", "int32", "ring index within the fit, 0 = leading"),
    ("nrings", "int32", "number of rings this fit used"),
    ("stage", "int32", f"fit stage; {STAGE_FINAL} = final converged result"),
    ("success", "bool", "fit converged and is usable"),
    ("nll", "float64", "total negative log-likelihood of the fit"),
    ("totmu", "float64", "total predicted photoelectrons (NaN if unavailable)"),
    ("t", "float64", "vertex time [ns], trigger-relative"),
    ("x", "float64", "vertex x [cm]"),
    ("y", "float64", "vertex y [cm]"),
    ("z", "float64", "vertex z [cm]"),
    ("dx", "float64", "unit direction x"),
    ("dy", "float64", "unit direction y"),
    ("dz", "float64", "unit direction z"),
    ("p", "float64", "momentum [MeV/c]"),
    ("dconv", "float64", "photon conversion length [cm] (NaN if not fitted)"),
]

TRUTH_RINGS_SCHEMA = [
    ("event", "int64", "global event id"),
    ("ring", "int32", "index of this true ring within the event"),
    ("pdg", "int32", "PDG code of the particle"),
    ("particle", "object", "ring type it produces ('e-', 'mu-', ...)"),
    ("t", "float64", "creation time [ns], on the trigger clock when the event's "
                     "trigger_time is known (see truth_events)"),
    ("x", "float64", "start x [cm]"),
    ("y", "float64", "start y [cm]"),
    ("z", "float64", "start z [cm]"),
    ("dx", "float64", "unit direction x"),
    ("dy", "float64", "unit direction y"),
    ("dz", "float64", "unit direction z"),
    ("p", "float64", "momentum [MeV/c]"),
    ("is_primary_lepton", "bool", "outgoing lepton of the primary interaction"),
    ("above_threshold", "bool", "above the Cherenkov threshold for its mass"),
    ("visible", "bool", "expected to produce a reconstructable ring"),
]

TRUTH_EVENTS_SCHEMA = [
    ("event", "int64", "global event id"),
    ("nhits", "int32", "true number of hit PMTs (-1 if unknown)"),
    ("nhits_trigger0", "int32", "hit PMTs in the first trigger alone (-1 if unknown)"),
    ("totq", "float64", "total observed charge [p.e.] (NaN if unknown)"),
    ("vtx_t", "float64", "true interaction time [ns], same clock as truth_rings.t"),
    ("vtx_x", "float64", "true interaction vertex x [cm]"),
    ("vtx_y", "float64", "true interaction vertex y [cm]"),
    ("vtx_z", "float64", "true interaction vertex z [cm]"),
    ("nu_pdg", "int32", "incoming neutrino PDG (0 for a particle gun)"),
    ("nu_energy", "float64", "incoming neutrino energy [MeV] (NaN for a gun)"),
    ("mode", "int32", "NEUT interaction mode (0 for a particle gun)"),
    ("is_cc", "bool", "charged-current interaction"),
    ("n_true_rings", "int32", "primary final-state particles"),
    ("n_visible_rings", "int32", "of those, above Cherenkov threshold"),
    ("n_sub_events", "int32", "WCSim triggers in this event"),
    ("towall", "float64", "distance from vertex to the wall along the ring [cm]"),
    ("dwall", "float64", "distance from vertex to the nearest wall [cm]"),
    ("trigger_time", "float64", "WCSim trigger-0 time on the generator clock [ns] "
                                "(header GetDate); already subtracted from t and "
                                "vtx_t. NaN when unknown, and t is then generator time"),
]

TRUTH_SUBEVENTS_SCHEMA = [
    ("event", "int64", "global event id"),
    ("sub_event", "int32", "true sub-event index: 0 = the prompt interaction, "
                           "1.. = delayed decay products in time order"),
    ("is_prompt", "bool", "the prompt interaction rather than a delayed decay"),
    ("pdg", "int32", "PDG code: the leading visible ring's for the prompt, the "
                     "decay product's otherwise"),
    ("parent_pdg", "int32", "PDG code of the particle that decayed (0 for the prompt)"),
    ("t", "float64", "creation time [ns], on the trigger clock when the event's "
                     "trigger_time is known"),
    ("x", "float64", "start x [cm]"),
    ("y", "float64", "start y [cm]"),
    ("z", "float64", "start z [cm]"),
    ("dx", "float64", "unit direction x"),
    ("dy", "float64", "unit direction y"),
    ("dz", "float64", "unit direction z"),
    ("p", "float64", "momentum [MeV/c]"),
    ("recorded", "bool", "a WCSim trigger's readout window covers its creation "
                         "time, so its light is in the data at all"),
    ("visible", "bool", "above the Cherenkov threshold for its mass"),
]

NLL_TERMS_SCHEMA = [
    ("event", "int64", "global event id"),
    ("sub_event", "int32", "sub-event index"),
    ("hypothesis", "object", "particle hypothesis"),
    ("point", "object", "parameter point the terms were evaluated at"),
    ("term", "object", "name of the likelihood component"),
    ("value", "float64", "its contribution to the negative log-likelihood"),
]

PMT_PREDICTIONS_SCHEMA = [
    ("event", "int64", "global event id"),
    ("sub_event", "int32", "sub-event index"),
    ("hypothesis", "object", "particle hypothesis"),
    ("point", "object", "parameter point the prediction was made at"),
    ("pmt", "int32", "PMT tube id"),
    ("mu_direct", "float64", "predicted direct photoelectrons"),
    ("mu_indirect", "float64", "predicted scattered/indirect photoelectrons"),
    ("q_obs", "float64", "observed charge [p.e.], 0 if unhit"),
    ("t_obs", "float64", "observed hit time [ns], NaN if unhit"),
    ("t_resid", "float64", "time residual at this parameter point [ns]"),
    ("cos_theta", "float64", "cosine of the angle to the ring axis"),
    ("distance", "float64", "vertex-to-PMT distance [cm]"),
    ("cos_incidence", "float64", "cosine of the light's incidence on the PMT"),
    ("pmt_loc", "int32", "PMT surface: barrel / top / bottom"),
    ("is_hit", "bool", "PMT recorded a hit"),
]

SCHEMAS = {
    "fits": FITS_SCHEMA,
    "truth_rings": TRUTH_RINGS_SCHEMA,
    "truth_events": TRUTH_EVENTS_SCHEMA,
    "truth_subevents": TRUTH_SUBEVENTS_SCHEMA,
    "nll_terms": NLL_TERMS_SCHEMA,
    "pmt_predictions": PMT_PREDICTIONS_SCHEMA,
}

#: Likelihood-component names Coral reports (see fitqun.py dump_nll_terms).
#: fiTQun reports only the total, so a fiTQun row set is a single 'total' term.
NLL_TERM_NAMES = ("unhit", "phit", "cpdf", "tpdf")


def columns(table: str) -> list:
    return [name for name, _, _ in SCHEMAS[table]]


def dtypes(table: str) -> dict:
    return {name: dtype for name, dtype, _ in SCHEMAS[table]}


def describe(table: str) -> pd.DataFrame:
    """The schema itself, as a table -- used by `sextant schema` and the report."""
    return pd.DataFrame(SCHEMAS[table], columns=["column", "dtype", "description"])


def empty(table: str) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype)
                         for name, dtype, _ in SCHEMAS[table]})


def conform(df: pd.DataFrame, table: str, *, strict: bool = False) -> pd.DataFrame:
    """Coerce a reader's output to the canonical column set, order and dtypes.

    Missing optional columns are filled with a null appropriate to their dtype
    rather than rejected: a reader for a format that never had `totmu` should
    not have to know what to put there. `strict` turns unknown extra columns
    into an error instead of dropping them, which is what the test suite uses to
    catch a reader inventing a column name that no metric will ever read.
    """
    spec = SCHEMAS[table]
    known = {name for name, _, _ in spec}
    extra = [c for c in df.columns if c not in known]
    if extra and strict:
        raise ValueError(f"{table}: unknown columns {extra}; "
                         f"add them to SCHEMAS or drop them in the reader")

    out = {}
    n = len(df)
    for name, dtype, _ in spec:
        if name in df.columns:
            col = df[name]
        else:
            col = pd.Series([_null_for(dtype)] * n, dtype=object)
        out[name] = _cast(col, dtype, n)
    return pd.DataFrame(out, index=pd.RangeIndex(n))


def _null_for(dtype: str):
    if dtype == "bool":
        return False
    if dtype.startswith("int"):
        return -1
    if dtype.startswith("float"):
        return np.nan
    return None


def _cast(col: pd.Series, dtype: str, n: int) -> pd.Series:
    values = pd.Series(np.asarray(col), index=pd.RangeIndex(n)) if n else pd.Series(dtype=object)
    if dtype == "object":
        return values.astype(object)
    if dtype == "bool":
        if not n:
            return values.astype(bool)
        # `.fillna(False)` on an object column is deprecated (silent
        # downcasting); map to a real boolean array first.
        filled = values.where(values.notna(), False)
        return filled.infer_objects(copy=False).astype(bool)
    if dtype.startswith("int"):
        # Ints arrive as floats often enough (a NaN-carrying merge column, an
        # awkward-array conversion) that silently rounding is kinder than raising;
        # the null fill above already put -1 where a value was genuinely absent.
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.fillna(-1).astype(dtype) if n else numeric.astype(dtype)
    return pd.to_numeric(values, errors="coerce").astype(dtype)


# ---------------------------------------------------------------------------
# Small shared helpers on canonical tables
# ---------------------------------------------------------------------------

def unit_vector(dx, dy, dz):
    """Normalise a direction, leaving exact zeros as NaN rather than 0/0 noise."""
    v = np.stack([np.asarray(dx, float), np.asarray(dy, float), np.asarray(dz, float)], axis=-1)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(norm > 0, v / norm, np.nan)
    return out[..., 0], out[..., 1], out[..., 2]


def add_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Add `theta`/`phi` derived from the direction columns.

    Stored as a direction vector rather than angles because phi is discontinuous
    and theta degenerate at the poles; every residual that needs an angle wants
    the opening angle between two vectors anyway. These are for plotting axes.
    """
    out = df.copy()
    out["theta"] = np.arccos(np.clip(out["dz"].to_numpy(float), -1.0, 1.0))
    out["phi"] = np.arctan2(out["dy"].to_numpy(float), out["dx"].to_numpy(float))
    return out


def cherenkov_threshold_mev(pdg: int, refraction_index: float = 1.33) -> float:
    """Momentum above which a particle radiates: p > m / sqrt(n^2 - 1).

    n=1.33 (phase index at the PMT-sensitive wavelengths) is the value the HK/SK
    threshold convention uses -- it gives the familiar ~120 MeV/c for muons and
    ~1.07 GeV/c for protons. Coral's config carries 1.38, a group index used for
    photon *timing*; using that here would move the muon threshold to 111 MeV/c
    and quietly change which true rings count as visible, so the two are kept
    deliberately separate.
    """
    mass = PDG_MASS_MEV.get(abs(int(pdg)))
    if mass is None:
        return np.inf
    if mass == 0.0:
        return 0.0
    denom = refraction_index ** 2 - 1.0
    if denom <= 0:
        return np.inf
    return mass / np.sqrt(denom)
