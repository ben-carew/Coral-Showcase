"""The experiment spec: one file describing a whole comparison.

An `exp.py`-style INI extended in the one way it needed to be -- an experiment
is a set of **arms**, not a single run, so that "Coral on this branch against
fiTQun at its official tune, on 1000 electrons" is one file rather than three
files and a note about how they go together.

    [experiment]
    name       = e_comparison
    production = v1.12.22
    sample     = electron
    events     = 1000
    where      = server

    [shared]                       ; copied into every arm's own config
    use_time_likelihood = true
    do_muon_fit         = true

    [arm.coral_sandbox]
    algorithm = coral
    slot      = a
    scheme    = coral-production
    config.fits.truth_seed = False

    [arm.fitqun]
    algorithm = fitqun
    params    = ParameterOverrideFiles/Official_HyperK_2024_wscat.parameters.dat
    scheme    = fitqun-hk2024-wscat

`[shared]` keys are *concepts* from `sextant.config.mapping`, not either code's
native key names; each is translated into the right key for each arm. That is
the point: writing `use_time_likelihood = true` once and having it land as
`[likelihood] use_time_pdf` in Coral and `fiTQun.UseTimeLikelihood` in fiTQun is
what keeps the two configurations honestly in step.

Arm-specific native settings still go through `config.<section>.<key>` (Coral)
or `param.<Name>` (fiTQun), because not everything has a counterpart and
pretending otherwise would be worse than admitting it.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from ..config import mapping

#: Where an arm's results come from.
SOURCE_RUN = "run"            # launch it
SOURCE_OFFICIAL = "official"  # read the production's shipped reconstruction
SOURCE_FILE = "file"          # read an existing merged output


def _is_on(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _events_per_job(raw, path, name, where, production, sample, events, arms,
                    s_per_event="") -> int:
    """`[run] events_per_job`, with 'auto' (the default) resolved by `exp.py pack`.

    Every arm shares one packing -- it decides which input files, hence which
    events, a run reads -- so it is resolved for the slowest algorithm launched.
    The answer is pinned in Output/Sextant/<name>/packing.json on first use:
    exp.py learns from new Timing CSVs, and a later `status` or `report` must see
    the packing the arms were launched with, not today's re-estimate.
    """
    import json
    import re
    import subprocess
    import sys

    if raw.lower() != "auto":
        return int(raw)
    algorithms = sorted({a.algorithm for a in arms if a.source == SOURCE_RUN})
    if where != "server" or not events or not algorithms:
        return 25
    from .launcher import find_exp_py
    exp_py = find_exp_py(Path(path).resolve().parent)
    pin = exp_py.parents[2] / "Output" / "Sextant" / name / "packing.json"
    if pin.exists():
        return int(json.loads(pin.read_text())["events_per_job"])
    if (pin.parent / "experiment.json").exists():
        # launched before 'auto' existed (2026-09-29), at the old implicit default
        return 25
    chosen = None
    for algo in algorithms:
        out = subprocess.run(
            [sys.executable, str(exp_py), "pack", "--name", name, "--where", "server",
             "--algo", algo, "--particle", sample, "--production", production,
             "-n", str(events), "--events-per-job", "auto"]
            + (["--s-per-event", s_per_event] if s_per_event else []),
            capture_output=True, text=True)
        found = re.search(r"events_per_job=(\d+)", out.stdout)
        if not found:
            raise ValueError(f"exp.py pack failed for {algo}: {out.stderr.strip()[-400:]}")
        chosen = min(chosen or 10**9, int(found.group(1)))
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"events_per_job": chosen, "events": events,
                               "algorithms": algorithms}) + "\n")
    return chosen


@dataclass
class ArmSpec:
    name: str
    algorithm: str
    source: str = SOURCE_RUN
    label: str = ""
    slot: str = "a"
    branch: str = ""
    scheme: str = ""
    params: str = ""                       # fiTQun parameter-override file
    path: str = ""                         # source=file
    events: int | None = None
    events_per_job: int | None = None
    start: int | None = None               # first input FILE index
    config_file: str = ""                  # Coral preset, relative to the slot's checkout
    #: `scheme_override = true`: the scheme is MEANT to change calibration constants the
    #: arm's own config sets (a scan, a measurement scheme). Without it, such an arm is refused.
    scheme_override: bool = False
    native_config: dict = field(default_factory=dict)   # Coral section.key
    native_params: dict = field(default_factory=dict)   # fiTQun parameter names
    server: dict = field(default_factory=dict)          # server.<key>: this arm's [server] overrides

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ")
        if not self.scheme:
            self.scheme = mapping.DEFAULT_SCHEME.get(self.algorithm, "")

    @property
    def variant(self) -> str:
        return self.name


@dataclass
class ExperimentSpec:
    name: str
    production: str
    sample: str
    events: int
    arms: list
    where: str = "local"
    events_per_job: int = 25
    #: First input FILE index, passed through to exp.py's --start. Exists so a
    #: calibration run and its validation run can be drawn from DISJOINT files:
    #: a correction fitted and tested on the same events proves nothing. Until
    #: 2026-09-16 this was parsed from the spec and never propagated, so every
    #: "held-out" Sextant run silently re-read files 0..N.
    start: int = 0
    #: Input directory override, passed through to exp.py's [experiment]
    #: input_dir. Needed for samples outside PROD_BASE -- e.g. a pre-filtered
    #: copy of a production. Dropped silently before 2026-09-16, like `start`.
    input_dir: str = ""
    #: Where the report is written. Empty means the default,
    #: Output/Sextant/<name>/report -- see cli.cmd_report.
    outdir: str = ""
    truth: str = ""
    #: Sub-event truth file; empty means the truth file itself, then its
    #: <stem>_subevents.root sidecar.
    truth_subevents: str = ""
    #: Another spec, on the other particle's sample, whose arms pair with this
    #: one's BY ARM NAME to give the PID section its second true particle.
    pid_partner: str = ""
    hypothesis: str = "e-"
    reference: str = ""
    shared: dict = field(default_factory=dict)
    time: str = ""
    #: Seconds between sbatch calls, paced so a run's jobs do not all hit /sps
    #: at once. Blank means the cluster submitter's own default.
    submit_delay: str = ""
    #: --mem per job (e.g. 8G); blank means the cluster submitter's default.
    mem: str = ""
    #: percentage of a run's slowest jobs exp.py cancels once the rest are done
    #: ([server] straggler_cut); 'auto' = exp.py's 5-10% default, 0 keeps every job.
    straggler_cut: str = "auto"
    #: s/event estimate for auto packing ([run] s_per_event); blank = learnt by exp.py
    s_per_event: str = ""
    workers: int = 4
    threads: int = 2
    source_path: str = ""

    @classmethod
    def load(cls, path) -> "ExperimentSpec":
        path = Path(path)
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        parser.optionxform = str          # Coral and fiTQun keys are case-sensitive
        if not path.exists():
            raise FileNotFoundError(f"experiment spec not found: {path}")
        parser.read(path)

        if not parser.has_section("experiment"):
            raise ValueError(f"{path}: no [experiment] section")
        get = lambda key, fallback="": parser.get("experiment", key,
                                                  fallback=fallback).strip()

        shared = {k: v.strip() for k, v in parser.items("shared")} \
            if parser.has_section("shared") else {}
        unknown = [k for k in shared if k not in mapping.BY_CONCEPT]
        if unknown:
            raise ValueError(
                f"{path}: [shared] keys must be concepts from the parameter "
                f"mapping; unknown: {', '.join(unknown)}.\n"
                f"Known concepts: {', '.join(sorted(mapping.BY_CONCEPT))}")

        arms = [cls._parse_arm(parser, section)
                for section in parser.sections() if section.startswith("arm.")]
        if not arms:
            raise ValueError(f"{path}: no [arm.<name>] sections — an experiment "
                             f"needs at least one arm")

        events = int(get("events", "0") or 0)
        spec = cls(
            name=get("name") or path.stem,
            production=get("production", "v1.12.22"),
            sample=get("sample", "electron"),
            events=events,
            arms=arms,
            where=get("where", "local").lower(),
            events_per_job=_events_per_job(
                parser.get("run", "events_per_job", fallback="auto").strip() or "auto",
                path, get("name") or path.stem, get("where", "local").lower(),
                get("production", "v1.12.22"), get("sample", "electron"), events, arms,
                parser.get("run", "s_per_event", fallback="").strip()),
            s_per_event=parser.get("run", "s_per_event", fallback="").strip(),
            start=int(parser.get("run", "start", fallback="0")
                      or 0) if parser.has_section("run") else 0,
            input_dir=get("input_dir"),
            outdir=get("outdir"),
            truth=get("truth"),
            truth_subevents=get("truth_subevents"),
            pid_partner=get("pid_partner"),
            hypothesis=get("hypothesis", "e-"),
            reference=get("reference"),
            shared=shared,
            submit_delay=parser.get("server", "submit_delay", fallback="").strip(),
            mem=parser.get("server", "mem", fallback="").strip(),
            straggler_cut=parser.get("server", "straggler_cut", fallback="auto").strip() or "auto",
            time=parser.get("server", "time", fallback="").strip()
                 if parser.has_section("server") else "",
            workers=int(parser.get("local", "workers", fallback="4"))
                    if parser.has_section("local") else 4,
            threads=int(parser.get("local", "threads", fallback="2"))
                    if parser.has_section("local") else 2,
            source_path=str(path),
        )
        spec.validate()
        return spec

    @staticmethod
    def _parse_arm(parser, section) -> ArmSpec:
        name = section.split(".", 1)[1]
        get = lambda key, fallback="": parser.get(section, key,
                                                  fallback=fallback).strip()
        native_config, native_params, server = {}, {}, {}
        for key, value in parser.items(section):
            if key.startswith("server."):
                server[key[len("server."):]] = value.strip()
            elif key.startswith("config."):
                native_config[key[len("config."):]] = value.strip()
            elif key.startswith("param."):
                native_params[key[len("param."):]] = value.strip()
        events = get("events")
        per_job = get("events_per_job")
        start = get("start")
        return ArmSpec(
            name=name,
            algorithm=get("algorithm", "coral").lower(),
            source=get("source", SOURCE_RUN).lower(),
            label=get("label"),
            slot=get("slot", "a").lower(),
            branch=get("branch"),
            scheme=get("scheme"),
            params=get("params"),
            path=get("path"),
            events=int(events) if events else None,
            events_per_job=int(per_job) if per_job else None,
            start=int(start) if start else None,
            config_file=get("config_file"),
            scheme_override=_is_on(get("scheme_override", "false")),
            native_config=native_config,
            native_params=native_params,
            server=server,
        )

    # -- validation ----------------------------------------------------------

    def validate(self):
        from .. import productions

        problems = []
        try:
            prod = productions.get(self.production)
            prod.sample(self.sample)
        except (KeyError, SystemExit) as exc:
            problems.append(str(exc))

        names = [arm.name for arm in self.arms]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            problems.append(f"duplicate arm names: {', '.join(sorted(duplicates))}")

        labels = [arm.label for arm in self.arms]
        if len(set(labels)) != len(labels):
            problems.append("two arms share a label; labels identify arms in "
                            "every figure and table and must be unique")

        for arm in self.arms:
            if arm.algorithm not in ("coral", "fitqun"):
                problems.append(f"arm '{arm.name}': unknown algorithm "
                                f"'{arm.algorithm}'")
            if arm.source == SOURCE_FILE and not arm.path:
                problems.append(f"arm '{arm.name}': source=file needs a path")
            if arm.source == SOURCE_OFFICIAL:
                if arm.algorithm != "fitqun":
                    problems.append(f"arm '{arm.name}': source=official is only "
                                    f"available for fiTQun")
                else:
                    # Whether the sample *declares* an official reconstruction,
                    # not whether its files are visible from here: these live on
                    # the cluster, and a spec must validate on the laptop that
                    # wrote it.
                    try:
                        sample = productions.get(self.production).sample(self.sample)
                    except KeyError:
                        sample = None
                    if sample is not None and sample.reference_reco_dir is None:
                        problems.append(
                            f"arm '{arm.name}': production '{self.production}' "
                            f"sample '{self.sample}' ships no official "
                            f"reconstruction to read")
            if (arm.scheme and arm.scheme != mapping.CONFIG_SCHEME
                    and arm.scheme not in mapping.SCHEMES):
                problems.append(f"arm '{arm.name}': unknown calibration scheme "
                                f"'{arm.scheme}'")
        if self.where not in ("local", "server"):
            problems.append(f"where must be 'local' or 'server' (got "
                            f"'{self.where}')")
        if self.reference and self.reference not in labels + names:
            problems.append(f"reference '{self.reference}' is not one of the "
                            f"arms ({', '.join(names)})")
        if problems:
            raise ValueError("invalid experiment spec:\n  - "
                             + "\n  - ".join(problems))

    # -- derived -------------------------------------------------------------

    def arm(self, name: str) -> ArmSpec:
        for arm in self.arms:
            if arm.name == name or arm.label == name:
                return arm
        raise KeyError(f"no arm '{name}' in experiment '{self.name}'")

    def reference_label(self) -> str:
        if self.reference:
            return self.arm(self.reference).label
        return self.arms[0].label

    def resolved_settings(self, arm: ArmSpec) -> dict:
        """The arm's full settings: scheme + shared concepts + native overrides.

        Later sources win, so an explicit native override always beats a shared
        concept, and a shared concept always beats the scheme default -- and the
        report records all three layers so it is visible which won.
        """
        values = {}
        if arm.scheme and arm.scheme in mapping.SCHEMES:
            values.update(mapping.SCHEMES[arm.scheme].values_for(arm.algorithm))
        hypotheses = []
        for concept, value in self.shared.items():
            parameter = mapping.BY_CONCEPT[concept]
            key = parameter.coral if arm.algorithm == "coral" else parameter.fitqun
            if key is None:
                continue                     # no counterpart in this code
            if parameter.kind == mapping.CALIBRATION:
                # Calibration constants come from the scheme as a set; a
                # [shared] entry must not pick one out of it.
                continue
            if arm.algorithm == "coral" and concept in mapping.HYPOTHESIS_LABEL:
                # Several concepts share Coral's fits.particles, so they have to
                # be assembled into its list rather than assigned one after the
                # other -- plain assignment lets `do_muon_fit = false` overwrite
                # `fits.particles = e-` with the string "false", which reads as
                # no hypothesis at all.
                if _is_on(value):
                    hypotheses.append(mapping.HYPOTHESIS_LABEL[concept])
                continue
            values[key] = value
        if hypotheses:
            values["fits.particles"] = ", ".join(hypotheses)
        values.update(arm.native_config if arm.algorithm == "coral"
                      else arm.native_params)
        return values

    def describe(self) -> str:
        lines = [f"Experiment : {self.name}",
                 f"Sample     : {self.production} / {self.sample}, "
                 f"{self.events} events",
                 f"Where      : {self.where}",
                 f"Arms       : {len(self.arms)}"]
        for arm in self.arms:
            detail = {SOURCE_RUN: f"run ({arm.algorithm}, slot {arm.slot})",
                      SOURCE_OFFICIAL: "read official reconstruction",
                      SOURCE_FILE: f"read {arm.path}"}[arm.source]
            lines.append(f"  - {arm.label}: {detail}"
                         + (f", scheme {arm.scheme}" if arm.scheme else ""))
        return "\n".join(lines)


def check_configuration(spec: ExperimentSpec) -> "tuple[list, list]":
    """Compare the arms' resolved settings; returns (rows, issues)."""
    return mapping.compare_arms({
        arm.label: {"algorithm": arm.algorithm, "scheme": arm.scheme,
                    "values": spec.resolved_settings(arm)}
        for arm in spec.arms})
