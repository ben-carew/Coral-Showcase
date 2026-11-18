"""An `Arm` -- one algorithm's results, loaded and ready to compare.

An arm is the unit of comparison: *one algorithm, at one version, with one
configuration, on one sample*. "Coral ben/sandbox on 1000 electrons" is an arm;
so is "official fiTQun 6.4.4 on the beam sample". Everything the rest of Sextant
plots or tabulates is a function of a list of arms and one truth.

Construction is deliberately dumb -- a label and some paths -- because this is
the seam the user asked to be able to drive by hand:

    arm = Arm.from_file("Coral sandbox", "Output/Coral/Reco/x/coral_electron.root")
    truth = Truth.from_file("Output/Truth/truth_electron.root")

No experiment spec, no manifest, no cluster. `sextant.run` builds arms from a
finished experiment, but nothing in `io`, `metrics`, `plots` or `report` knows
that module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import coral, fitqun, timing as timing_io, truth as truth_io
from .schema import HYP_ELECTRON, STAGE_FINAL, empty

FORMAT_CORAL = "coral"
FORMAT_FITQUN_MERGED = "fitqun-merged"
FORMAT_FITQUN_OFFICIAL = "fitqun-official"


def detect_format(path) -> str:
    # Checked before sniffing, because every is_* probe swallows its exceptions:
    # without this a missing or unreadable file reports as "unrecognised
    # format", which sends you looking at the wrong problem entirely.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such reconstruction file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name}: file is empty")
    if coral.is_coral(path):
        return FORMAT_CORAL
    if fitqun.is_fitqun_official(path):
        return FORMAT_FITQUN_OFFICIAL
    if fitqun.is_fitqun_merged(path):
        return FORMAT_FITQUN_MERGED
    raise ValueError(
        f"{Path(path).name}: unrecognised reconstruction output. Expected a "
        "Coral merge (a 'fits' tree with 'particle_code'), a fiTQun merge (a "
        "'fits' tree with 'nll_e'), or an official fiTQun file (a 'fiTQun' tree).")


@dataclass
class Truth:
    """MC truth for one sample."""

    rings: pd.DataFrame
    events: pd.DataFrame
    label: str = "truth"
    source: str = ""
    #: Prompt + decay-electron sub-events; empty when the truth has none.
    subevents: pd.DataFrame = field(default_factory=lambda: empty("truth_subevents"))

    @classmethod
    def from_file(cls, path, label: str = "truth", *, subevents=None) -> "Truth":
        """`subevents` names a sub-event truth file; by default the truth file
        itself, then its `<stem>_subevents.root` sidecar, are tried."""
        rings, events = truth_io.read(path)
        return cls(rings=rings, events=events, label=label, source=str(path),
                   subevents=truth_io.read_subevents(subevents or path))

    @classmethod
    def from_files(cls, paths, label: str = "truth") -> "Truth":
        paths = list(paths)
        rings, events = truth_io.read_many(paths)
        subevents = [truth_io.read_subevents(p) for p in paths]
        subevents = [s for s in subevents if not s.empty]
        return cls(rings=rings, events=events, label=label,
                   source=", ".join(str(p) for p in paths),
                   subevents=(pd.concat(subevents, ignore_index=True)
                              .drop_duplicates(["event", "sub_event"])
                              if subevents else empty("truth_subevents")))

    @property
    def primary(self) -> pd.DataFrame:
        """One true ring per event -- what a single-ring fit is judged against."""
        if not hasattr(self, "_primary"):
            self._primary = truth_io.select_primary_rings(self.rings)
        return self._primary

    def __len__(self):
        return len(self.events)


@dataclass
class Arm:
    """One algorithm's output, in canonical form."""

    label: str
    algorithm: str                     # 'coral' | 'fitqun'
    fits: pd.DataFrame
    prefits: pd.DataFrame = field(default_factory=lambda: empty("fits"))
    multiring: pd.DataFrame = field(default_factory=lambda: empty("fits"))
    pi0: pd.DataFrame = field(default_factory=lambda: empty("fits"))
    stages: pd.DataFrame = field(default_factory=lambda: empty("fits"))
    nll_terms: pd.DataFrame = field(default_factory=lambda: empty("nll_terms"))
    event_totals: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: (event, sub_event, t) of every reconstructed sub-event, when a reader
    #: knows it better than the fits do (a fiTQun merge keeps sub-event 0 only).
    #: Empty means: derive it from the prefits and fits.
    subevents: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Free-form provenance: run id, code SHA, config digest, spec path.
    meta: dict = field(default_factory=dict)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_file(cls, label, path, *, algorithm=None, timing_path=None,
                  production=None, sample=None, time_offset_ns=None,
                  meta=None) -> "Arm":
        path = Path(path)
        fmt = detect_format(path)
        meta = dict(meta or {})
        meta.setdefault("path", str(path))
        meta.setdefault("format", fmt)

        if fmt == FORMAT_CORAL:
            tables = coral.read_all(path)
            arm = cls(label=label, algorithm=algorithm or "coral",
                      fits=tables["fits"], prefits=tables["prefits"],
                      multiring=tables["multiring"], pi0=tables["pi0"],
                      stages=tables["stages"],
                      nll_terms=tables["nll_terms"],
                      event_totals=tables["event_totals"], meta=meta)
        elif fmt == FORMAT_FITQUN_MERGED:
            offset = 0.0 if time_offset_ns is None else time_offset_ns
            arm = cls(label=label, algorithm=algorithm or "fitqun",
                      fits=fitqun.read_merged_fits(path, time_offset_ns=offset),
                      prefits=fitqun.read_merged_prefits(path, time_offset_ns=offset),
                      multiring=fitqun.read_merged_multiring(path),
                      pi0=fitqun.read_merged_pi0(path),
                      stages=fitqun.read_merged_stages(path, time_offset_ns=offset),
                      nll_terms=fitqun.read_merged_nll_terms(path),
                      event_totals=fitqun.read_merged_totals(path),
                      subevents=fitqun.read_merged_subevents(
                          path, time_offset_ns=offset), meta=meta)
        else:
            offset = (fitqun.DEFAULT_TIME_OFFSET_NS if time_offset_ns is None
                      else time_offset_ns)
            index, stride = _official_ids(path, production, sample)
            tables = fitqun.read_official(path, file_index=index, stride=stride,
                                          time_offset_ns=offset)
            arm = cls(label=label, algorithm=algorithm or "fitqun",
                      fits=tables["fits"], prefits=tables["prefits"],
                      multiring=tables["multiring"], pi0=tables["pi0"],
                      stages=tables["stages"], nll_terms=tables["nll_terms"],
                      event_totals=tables["event_totals"], meta=meta)

        tpath = (Path(timing_path) if timing_path
                 else timing_io.timing_path_for(path))
        arm.timing = timing_io.read(tpath)
        arm.meta.setdefault("timing_path", str(tpath))
        return arm

    @classmethod
    def from_official_files(cls, label, paths, *, production, sample,
                            time_offset_ns=None, meta=None) -> "Arm":
        """An arm built from many official fiTQun files (the beam reference)."""
        offset = (fitqun.DEFAULT_TIME_OFFSET_NS if time_offset_ns is None
                  else time_offset_ns)
        tables = fitqun.read_official_many(
            paths, production=production, sample=sample, time_offset_ns=offset)
        meta = dict(meta or {})
        meta.setdefault("format", FORMAT_FITQUN_OFFICIAL)
        meta.setdefault("n_files", len(list(paths)))
        return cls(label=label, algorithm="fitqun",
                   fits=tables.get("fits", empty("fits")),
                   prefits=tables.get("prefits", empty("fits")),
                   multiring=tables.get("multiring", empty("fits")),
                   pi0=tables.get("pi0", empty("fits")),
                   stages=tables.get("stages", empty("fits")),
                   nll_terms=tables.get("nll_terms", empty("nll_terms")),
                   event_totals=tables.get("event_totals", pd.DataFrame()),
                   meta=meta)

    # -- views ---------------------------------------------------------------

    def hypothesis(self, hyp: str = HYP_ELECTRON, *, only_success: bool = True,
                   sub_event: int | None = 0) -> pd.DataFrame:
        """The final single-ring fit under one hypothesis, one row per event.

        Restricted to `sub_event` (0 by default: the primary trigger; later
        sub-events are muon-decay Michel electrons, which have no true ring in
        the primary truth record and would otherwise be scored against it).
        """
        df = self.fits
        df = df[(df["hypothesis"] == hyp) & (df["stage"] == STAGE_FINAL)]
        if sub_event is not None:
            df = df[df["sub_event"] == sub_event]
        if only_success:
            df = df[df["success"]]
        return (df.sort_values(["event", "sub_event"])
                  .drop_duplicates("event", keep="first").reset_index(drop=True))

    def available_hypotheses(self) -> list:
        if self.fits.empty:
            return []
        return sorted(h for h in self.fits["hypothesis"].unique() if h != "(prefit)")

    def n_events(self) -> int:
        return int(self.fits["event"].nunique()) if not self.fits.empty else 0

    def describe(self) -> dict:
        return {
            "label": self.label, "algorithm": self.algorithm,
            "events": self.n_events(),
            "hypotheses": ",".join(self.available_hypotheses()),
            "rows_fits": len(self.fits), "rows_multiring": len(self.multiring),
            "rows_pi0": len(self.pi0), "rows_stages": len(self.stages),
            "has_nll_terms": not self.nll_terms.empty,
            "has_timing": not self.timing.empty,
            **{k: v for k, v in self.meta.items() if k in
               ("path", "format", "run", "code_sha", "branch")},
        }

    def __repr__(self):
        return (f"Arm({self.label!r}, {self.algorithm}, "
                f"{self.n_events()} events, "
                f"hyps={self.available_hypotheses()})")


def _official_ids(path, production, sample):
    """(file_index, stride) for an official file, from the production registry."""
    from .. import productions as registry

    if production is None:
        return 0, 1
    prod = registry.get(production)
    if sample is None:
        return 0, prod.stride
    return prod.sample(sample).file_index(path), prod.stride


def load_arms(entries, **defaults) -> list:
    """Build several arms from (label, path) pairs or dicts.

    Accepts the shape the plotting half is most often driven with by hand:

        load_arms([("Coral sandbox", p1), ("fiTQun", p2)], production="v1.12.22")
    """
    arms = []
    for entry in entries:
        if isinstance(entry, Arm):
            arms.append(entry)
        elif isinstance(entry, dict):
            arms.append(Arm.from_file(**{**defaults, **entry}))
        else:
            label, path = entry
            arms.append(Arm.from_file(label, path, **defaults))
    return arms
