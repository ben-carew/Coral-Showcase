#!/usr/bin/env python3

"""Unified driver for Coral/fiTQun reconstruction experiments.

    exp.py submit    start a run. Server: freeze code+config into the run
                     directory and sbatch, returns immediately. Local: run AND
                     merge synchronously.
    exp.py runs      one-line inventory of every known run, local and server.
    exp.py status    live dashboard of runs that are active or awaiting merge.
    exp.py logs      stream (tail -f) a run's raw job output.
    exp.py merge     consolidate a finished run's job outputs into one ROOT file.
    exp.py pull      rsync a server run's Reco/ + Timing/ back to the laptop.
    exp.py cancel    cancel one run's jobs (use --all for every job you own).

Only `submit` reads an experiment spec. Every other command takes a RUN ID and
reads that run's own manifest for algorithm, particle, production and paths:

    ./exp.py logs  mr_scan/nseed100
    ./exp.py merge coral:mr_scan/nseed100      # algo prefix if the id is ambiguous

That is deliberate. The old version derived everything from a single shared
experiment.ini, so whichever agent edited it last silently redirected every
other agent's status/logs/merge/pull/cancel — the reason `logs` showed fiTQun
output while Coral jobs were running.

Experiment specs live in Analysis/Experiments/<name>.ini (`-f` to pick one);
every field has a command-line override:

    ./exp.py submit -f mr_scan.ini -n 300 --particle muon --variant nseed100
"""

from __future__ import annotations

from pathlib import Path
import argparse
import configparser
import csv
import statistics
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time

SCRIPTS_DIR  = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPTS_DIR.parents[1]
CLUSTER_DIR  = SCRIPTS_DIR / "cluster"
SPEC_DIR     = RESEARCH_DIR / "Analysis/Experiments"
DEFAULT_SPEC = SPEC_DIR / "default.ini"

sys.path.insert(0, str(CLUSTER_DIR))
import runlib                                    # noqa: E402  (shared with the cluster scripts)
from runlib import MANIFEST_NAME, PRODUCTIONS, DEFAULT_PRODUCTION, EVENTS_PER_FILE

# PINNED 2026-09-16: `cca.in2p3.fr` is a round-robin alias and /sps/hyperk fails
# to automount on roughly half the login nodes behind it -- measured that day,
# cca011/013/015 fine, cca003/004/012 with no /sps/hyperk at all ("df: No such
# file or directory"). Landing on a bad one makes exp.py HANG rather than fail:
# a submit sat blocked on --sync for 25 minutes having created only an empty run
# directory. Revert to the alias once the automount is fixed; override here
# rather than in ~/.ssh/config so the choice is visible and versioned.
REMOTE_HOST = os.environ.get("CORAL_REMOTE_HOST", "cca013.in2p3.fr")
REMOTE_BASE = "/sps/hyperk/bcarew"

ALGOS = {
    "coral":  {"outdir": "Coral",  "env": "env_coral.zsh",  "submit": "submit_coral.py",
               "merge": "merge_coral.py",  "local_runner": "run_coral_local.py"},
    "fitqun": {"outdir": "fiTQun", "env": "env_fitqun.zsh", "submit": "submit_fitqun.py",
               "merge": "merge_fitqun.py", "local_runner": "run_fitqun_local.py"},
}
# Derived from the production registry rather than hardcoded, so adding a
# production (e.g. the NEUT beam samples, whose "particles" are neutrino
# channels) does not need an edit here as well. "multiring" is the CAVERNS
# sample, which lives outside PROD_BASE and is always given an explicit
# --input-dir.
PARTICLES = tuple(sorted({name
                          for prod in PRODUCTIONS.values()
                          for name in prod["particles"]} | {"multiring"}))

# Worktree slots for parallel agents. Each slot is an independent checkout, so
# three agents can hold three different branches at once; `main` is the primary
# checkout and is reserved for interactive work.
SLOTS = ("main", "a", "b", "c")

def coral_root_for(slot: str, base: Path) -> Path:
    return base / "Software/Coral" if slot == "main" else base / f"Software/worktrees/{slot}"

MANIFEST_LOOKBACK_DAYS = 14  # server: how far back to look for manifests worth reporting
MERGED_SHOW_HOURS = 12       # keep a finished+merged run on the dashboard this long

LOCAL_CORAL_CONFIG = RESEARCH_DIR / "Software/Coral/config.ini"
LOCAL_CORAL_CONFIG_LOCAL = RESEARCH_DIR / "Software/Coral/config_local.ini"


def coral_config_for(slot: str, local: bool = False) -> Path:
    """The config belonging to a SLOT, not to the primary checkout.

    config.ini is committed per branch precisely so each branch ships the
    settings its features need, so a run whose code comes from slot `a` must
    take slot `a`'s config -- taking slot `main`'s instead silently pairs one
    branch's code with another branch's physics. Falls back to the primary
    checkout only when the slot has no config of its own.
    """
    name = "config_local.ini" if local else "config.ini"
    candidate = coral_root_for(slot, RESEARCH_DIR) / name
    if candidate.exists():
        return candidate
    fallback = LOCAL_CORAL_CONFIG_LOCAL if local else LOCAL_CORAL_CONFIG
    return fallback

HYPERK_PYTHON = Path.home() / ".conda/envs/HyperK/bin/python"
LOCAL_PYTHON  = HYPERK_PYTHON if HYPERK_PYTHON.exists() else Path(sys.executable)

# Straggler reaping ([server] straggler_cut): submit ships this watchdog to the
# server and runs it as its own small SLURM job, which scancels the slowest tail
# of the run once the rest has finished.
REAPER_SCRIPT    = "reap_stragglers.py"
REAPER_PLAN_NAME = "reap_plan.json"
REAPER_POLL_S    = 60
REAPER_MARGIN_S  = 3600

CORAL_EVENT_RE = re.compile(r"Processing time for event #\s*\d+\s*=\s*[\d.]+\s*s")
CORAL_DONE_RE  = re.compile(r"PyfiTQun running time:\s*[\d.]+\s*s")


# -----------------------------
# Experiment spec (submit only)
# -----------------------------

class Experiment:
    """What to launch. Read by `submit` alone — never by the other commands."""

    def __init__(self, spec_path: Path, overrides):
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        if spec_path and spec_path.exists():
            cfg.read(spec_path)
        get = lambda sec, key, fb: cfg.get(sec, key, fallback=fb).strip()

        self.name     = overrides.name     or get("experiment", "name", "")
        self.variant  = (overrides.variant if overrides.variant is not None
                         else get("experiment", "variant", ""))
        self.algo     = (overrides.algo     or get("experiment", "algo", "coral")).lower()
        self.where    = (overrides.where    or get("experiment", "where", "local")).lower()
        self.particle = overrides.particle  or get("experiment", "particle", "electron")
        self.events   = (overrides.events   if overrides.events is not None
                         else int(get("experiment", "events", "0") or 0))
        self.production = (overrides.production or
                           get("experiment", "production", DEFAULT_PRODUCTION))
        self.params   = (overrides.params if getattr(overrides, "params", None) is not None
                         else get("experiment", "params", ""))
        # coral server submits only: overrides submit_coral.py's input-file directory
        # (naming/prefix unchanged) -- needed for samples outside PROD_BASE, e.g.
        # --particle multiring (see cluster/runlib.py PRODUCTIONS).
        self.input_dir = (overrides.input_dir if getattr(overrides, "input_dir", None) is not None
                          else get("experiment", "input_dir", ""))
        self.multiring = get("experiment", "multiring", "false").lower() in ("1", "true", "yes")
        self.slot     = (overrides.slot or get("experiment", "slot", "main")).lower()

        # 'auto' (the default) packs from measured rates -- see _resolve_packing.
        # An explicit number is kept: it also decides WHICH events run (one input
        # file per job), so established validation sets depend on it.
        _epj = (str(overrides.events_per_job) if overrides.events_per_job is not None
                else get("run", "events_per_job", "auto") or "auto")
        self.events_per_job_auto = _epj.lower() == "auto"
        self.events_per_job = 0 if self.events_per_job_auto else int(_epj)
        # optional s/event estimate for the auto packer (e.g. multiring ~3-4x single-ring)
        _spe = getattr(overrides, "s_per_event", None)
        self.s_per_event = (_spe if _spe is not None
                            else float(get("run", "s_per_event", "0") or 0))
        # First input FILE index. Exists so a calibration run and its validation
        # run can be drawn from disjoint files of the same production: a
        # correction fitted and tested on the same events proves nothing.
        self.start = (overrides.start if getattr(overrides, "start", None) is not None
                      else int(get("run", "start", "0") or 0))
        self.workers  = int(get("local", "workers", "4"))
        self.threads  = int(get("local", "threads", "2"))
        self.fetch    = get("local", "fetch", "true").lower() in ("1", "true", "yes")
        self.time     = (overrides.time if getattr(overrides, "time", None) is not None
                         else get("server", "time", ""))
        _cut = (str(overrides.straggler_cut) if overrides.straggler_cut is not None
                else get("server", "straggler_cut", "auto") or "auto")
        self.straggler_cut_auto = _cut.lower() == "auto"
        self.straggler_cut = 0.0 if self.straggler_cut_auto else float(_cut)
        # Seconds between sbatch calls. Blank in the spec means "the submitter's
        # own default", which is where the pacing policy lives -- exp.py should
        # not carry a second copy of it that can drift.
        _delay = getattr(overrides, "submit_delay", None)
        self.submit_delay = (_delay if _delay is not None
                             else get("server", "submit_delay", ""))
        # --mem per job; blank = the submitter's default (3G; 4G for GPU jobs)
        self.mem = get("server", "mem", "")
        # --cpus-per-task (= numba threads) per job; blank = the submitter's default (1; 4 for GPU jobs)
        self.cpus = get("server", "cpus", "")
        # GPU arms: SLURM partition, GPUs per job, extra exports ("K=V; K2=V2"); blank = htc, none
        self.partition = get("server", "partition", "")
        self.gpus = get("server", "gpus", "")
        self.env = [e.strip() for e in get("server", "env", "").split(";") if e.strip()]
        # a preset inside the slot's Coral checkout (e.g. configs/speed.ini) to
        # use instead of its config.ini; [coral_config] still patches it
        self.coral_config = get("experiment", "coral_config", "")

        # [coral_config] entries patch the Coral config on its way into the run
        # directory, as 'section.key = value'. A scan over a config knob then
        # needs one spec per variant instead of an edit to the shared
        # Software/Coral/config*.ini between submits — which would race any other
        # agent submitting at the same moment, and leave no record of what ran.
        # The patched file is what gets frozen, so manifest/config.ini still say
        # exactly what the jobs executed.
        self.config_overrides = {}
        if cfg.has_section("coral_config"):
            for dotted, value in cfg.items("coral_config"):
                if "." not in dotted:
                    raise SystemExit(f"[coral_config] keys are 'section.key' (got '{dotted}')")
                # rpartition, not partition: Coral has NESTED config sections
                # ([tuning_files.cprofiles], [fits.subeventing]), so the section
                # is everything before the LAST dot. Splitting on the first dot
                # turned "tuning_files.cprofiles.e-" into a key "cprofiles.e-"
                # inside [tuning_files], which Coral silently ignores.
                section, _, key = dotted.rpartition(".")
                self.config_overrides.setdefault(section, {})[key] = value.strip()
        for item in (getattr(overrides, "set_config", None) or []):
            key_part, sep, value = item.partition("=")
            section, dot, key = key_part.strip().rpartition(".")
            if not (sep and dot and section and key):
                raise SystemExit(f"--set expects 'section.key=value' (got '{item}')")
            self.config_overrides.setdefault(section, {})[key] = value.strip()

        if self.algo not in ALGOS:
            raise SystemExit(f"algo must be one of {sorted(ALGOS)} (got '{self.algo}')")
        if self.where not in ("server", "local"):
            raise SystemExit(f"where must be 'server' or 'local' (got '{self.where}')")
        if self.particle not in PARTICLES:
            raise SystemExit(f"particle must be one of {PARTICLES} (got '{self.particle}')")
        if self.production not in PRODUCTIONS:
            raise SystemExit(f"production must be one of {sorted(PRODUCTIONS)} "
                             f"(got '{self.production}')")
        if self.particle not in PRODUCTIONS[self.production]["particles"]:
            available = ", ".join(sorted(PRODUCTIONS[self.production]["particles"]))
            raise SystemExit(f"particle '{self.particle}' is not available in production "
                             f"'{self.production}' (available: {available})")
        if self.slot not in SLOTS:
            raise SystemExit(f"slot must be one of {SLOTS} (got '{self.slot}')")
        self._resolve_packing()
        if self.expected_jobs > runlib.MAX_JOBS:
            raise SystemExit(f"{self.expected_jobs} jobs exceeds the {runlib.MAX_JOBS}-job cap "
                             f"per submission -- raise events_per_job or split the run")
        if not 0 <= self.straggler_cut < 100:
            raise SystemExit(f"straggler_cut must be a percentage in [0, 100)")
        if self.submit_delay != "":
            try:
                if float(self.submit_delay) < 0:
                    raise ValueError
            except ValueError:
                raise SystemExit("submit_delay must be a non-negative number of seconds")
        if not 1 <= self.events_per_job <= EVENTS_PER_FILE:
            raise SystemExit(f"events_per_job must be in 1..{EVENTS_PER_FILE}")
        if not self.name:
            raise SystemExit("experiment name is empty — set [experiment] name in the spec "
                             "or pass --name")

    def _resolve_packing(self):
        """Fill in 'auto' events_per_job / straggler_cut (bcarew's rule, 2026-09-29):
        server runs are packed for the shortest wall time from MEASURED rates and
        always cut 5-10% of stragglers; local runs keep 25/job and every job
        (laptop runs are the ones used for real timing)."""
        self.pack_note = "explicit"
        if self.events_per_job_auto:
            if self.where == "server" and self.events and self.algo in ALGOS:
                s_ev, src = ((self.s_per_event, "spec") if self.s_per_event
                             else measured_s_per_event(self.algo, self.particle,
                                                       int(self.cpus or (4 if self.gpus else 1))))
                m, msrc = measured_merge_s_per_job(self.algo)
                pace = (float(self.submit_delay) if self.submit_delay != ""
                        else runlib.DEFAULT_SUBMIT_DELAY_S)
                self.events_per_job = runlib.auto_events_per_job(self.events, s_ev, pace, m)
                n = -(-self.events // self.events_per_job)
                est = n * (pace + m) + self.events_per_job * s_ev * runlib.PACK_TAIL + 20
                self.pack_note = (f"auto: {s_ev:.1f} s/event ({src}), merge {m:.2f} s/job "
                                  f"({msrc}); est. wall ~{est/60:.0f} min + queue")
            else:
                self.events_per_job = 25
                self.pack_note = "auto (local default)"
        if self.straggler_cut_auto:
            self.straggler_cut = (runlib.auto_straggler_cut(self.expected_jobs)
                                  if self.where == "server" else 0.0)

    @property
    def jobs_subdir(self) -> str:
        return f"{self.name}/{self.variant}" if self.variant else self.name

    @property
    def reco_filename(self) -> str:
        return runlib.reco_filename(self.algo, self.particle, self.variant)

    @property
    def algo_outdir(self) -> str:
        return ALGOS[self.algo]["outdir"]

    @property
    def job_name(self) -> str:
        return (self.variant or self.name)[:24]

    @property
    def local_jobs_dir(self) -> Path:
        return RESEARCH_DIR / "Output" / self.algo_outdir / "Jobs" / self.jobs_subdir

    @property
    def expected_jobs(self) -> int:
        return len(runlib.plan_jobs(self.events, self.events_per_job)) if self.events else 0

    def describe(self):
        loc = "CC-IN2P3 cluster" if self.where == "server" else "this laptop"
        print(f"Experiment : {self.name}" + (f" / {self.variant}" if self.variant else ""))
        print(f"Run        : {self.algo} on {loc}, {self.events} {self.particle} events "
              f"(production {self.production})")
        print(f"Packing    : {self.events_per_job} events/job -> {self.expected_jobs} jobs "
              f"[{self.pack_note}]")
        # one input file per job: this is the sample, keep it clear of calibration files
        print(f"Input files: {self.start}-{self.start + self.expected_jobs - 1} "
              f"(first {self.events_per_job} events of each)")
        if self.where == "server":
            print(f"Stragglers : cut {self.straggler_cut:g}%"
                  + (" (auto)" if self.straggler_cut_auto else ""))
        if self.algo == "coral":
            print(f"Code       : slot '{self.slot}'")
        print(f"Jobs dir   : Output/{self.algo_outdir}/Jobs/{self.jobs_subdir}")
        print(f"Result     : Output/{self.algo_outdir}/Reco/{self.name}/{self.reco_filename}")
        for section, entries in self.config_overrides.items():
            for key, value in entries.items():
                print(f"Config     : [{section}] {key} = {value}")
        print()

    def materialise_config(self, source: Path, workdir: Path) -> Path:
        """The Coral config this run will freeze, with [coral_config] applied.

        Returns `source` untouched when there is nothing to override, so the
        no-override path is byte-for-byte what it always was.
        """
        if not self.config_overrides:
            return source
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))   # keep comments out of values
        cfg.optionxform = str            # Coral keys are case-sensitive (e.g. 'e-', 'STable')
        cfg.read(source)
        for section, entries in self.config_overrides.items():
            if not cfg.has_section(section):
                cfg.add_section(section)
            for key, value in entries.items():
                # Coral's keys are case-sensitive here (optionxform = str), but
                # the spec files that supply overrides are read by a ConfigParser
                # that lower-cases them. Writing the lower-cased key blindly left
                # BOTH "Charge" and "charge" in [tuning_files], and Coral's own
                # ConfigParser then died with DuplicateOptionError. Replace any
                # existing key that differs only by case.
                for existing in list(cfg[section].keys()):
                    if existing != key and existing.lower() == key.lower():
                        cfg.remove_option(section, existing)
                        key = existing          # keep Coral's spelling
                cfg.set(section, key, value)
        workdir.mkdir(parents=True, exist_ok=True)
        patched = workdir / f"config_{self.jobs_subdir.replace('/', '_')}.ini"
        with patched.open("w") as fh:
            cfg.write(fh)
        return patched


# -----------------------------
# ssh plumbing
# -----------------------------

# Reuse one ssh connection across calls (and across `status` refreshes): a fresh
# CC-IN2P3 handshake costs seconds, which is most of the latency of a refresh.
SSH_OPTS = ["-o", "ControlMaster=auto",
            "-o", "ControlPath=/tmp/.exp-ssh-%r@%h:%p",
            "-o", "ControlPersist=600"]


PACK_FALLBACK_S_PER_EVENT = {"coral": 8.0, "fitqun": 150.0}
# first consolidate commit whose jobs fit the default 3G (older Coral code gets 5G)
MEM_FIX_SHA = "c6da583"
PACK_FALLBACK_MERGE_S = 0.3
MERGE_LOG_NAME = "merge_log.csv"


def measured_s_per_event(algo: str, particle: str, cpus: int = 4, newest: int = 8):
    """Reconstruction s/event learnt from the newest Timing CSVs of this
    algo+particle. The 75th percentile of their per-run medians, not the median:
    special configs (prefit-only, no time PDF) sit among them, and
    underestimating is the costly error (long jobs, wall-time kills). A config
    known to be slower should say so with [run] s_per_event. Rates are rescaled
    to `cpus` threads as core-seconds (rows without a cpus column = 4-thread
    jobs); that overestimates a 1-thread job by ~1.4x, the safe direction."""
    timing = RESEARCH_DIR / "Output" / ALGOS[algo]["outdir"] / "Timing"
    files = (sorted(timing.glob(f"**/*_{particle}*.csv"), key=lambda f: f.stat().st_mtime,
                    reverse=True) if timing.exists() else [])
    per_run = []
    for f in files:
        try:
            with open(f) as fh:
                rows = [r for r in csv.DictReader(fh)
                        if r.get("time_s") and r.get("events") and float(r["events"]) > 0]
        except (OSError, ValueError, KeyError):
            continue
        if len(rows) >= 3:
            per_run.append(statistics.median(float(r["time_s"]) / float(r["events"])
                                             * float(r.get("cpus") or 4) / cpus for r in rows))
        if len(per_run) == newest:
            break
    if not per_run:
        return PACK_FALLBACK_S_PER_EVENT[algo], "default"
    if len(per_run) == 1:
        return per_run[0], "1 recent run"
    return (statistics.quantiles(per_run, n=4)[2],
            f"p75 of {len(per_run)} recent runs")


def measured_merge_s_per_job(algo: str):
    """Median merge seconds per job from this algo's merge log (written by cmd_merge)."""
    log = RESEARCH_DIR / "Output" / ALGOS[algo]["outdir"] / "Timing" / MERGE_LOG_NAME
    try:
        with open(log) as fh:
            vals = [float(r["seconds"]) / int(r["jobs"]) for r in csv.DictReader(fh)
                    if int(r["jobs"]) >= 5]
    except (OSError, ValueError, KeyError):
        vals = []
    if not vals:
        return PACK_FALLBACK_MERGE_S, "default"
    return statistics.median(vals[-10:]), f"median of {len(vals[-10:])} merges"


def ssh(command: str, capture=False):
    result = subprocess.run(["ssh", *SSH_OPTS, REMOTE_HOST, command],
                            capture_output=capture, text=True)
    if capture:
        return result.returncode, result.stdout, result.stderr
    return result.returncode


def ssh_tee(command: str):
    """Run a remote command, echoing its output live AND returning it."""
    proc = subprocess.Popen(["ssh", *SSH_OPTS, REMOTE_HOST, command],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def run_local(cmd: list):
    print("+", " ".join(shlex.quote(str(c)) for c in cmd))
    # Flush before handing the terminal to the child: when output is redirected to
    # a file our own prints are block-buffered and would otherwise appear AFTER
    # the subprocess's, making a captured log read out of order.
    sys.stdout.flush()
    return subprocess.run([str(c) for c in cmd]).returncode


def particle_flag(particle: str) -> str:
    return f"--{particle}"


# -----------------------------
# Run resolution (every command except submit)
# -----------------------------

class RunRef:
    """A specific run, identified by its own manifest rather than by a spec file."""

    def __init__(self, algo: str, jobs_subdir: str, where: str, manifest: dict):
        self.algo = algo
        self.jobs_subdir = jobs_subdir
        self.where = where
        self.manifest = manifest or {}

    @property
    def outdir(self) -> str:      return ALGOS[self.algo]["outdir"]
    @property
    def particle(self) -> str:    return self.manifest.get("particle", "")
    @property
    def production(self) -> str:  return self.manifest.get("production", DEFAULT_PRODUCTION)
    @property
    def name(self) -> str:        return self.manifest.get("name") or self.jobs_subdir.split("/")[0]
    @property
    def variant(self) -> str:     return self.manifest.get("variant", "")

    @property
    def local_root(self) -> Path:
        return RESEARCH_DIR / "Output" / self.outdir / "Jobs" / self.jobs_subdir

    @property
    def remote_root(self) -> str:
        return f"{REMOTE_BASE}/Output/{self.outdir}/Jobs/{self.jobs_subdir}"

    @property
    def reco_filename(self) -> str:
        return runlib.reco_filename(self.algo, self.particle, self.variant)

    def describe(self):
        code = self.manifest.get("code") or {}
        where = "CC-IN2P3 cluster" if self.where == "server" else "this laptop"
        print(f"Run        : {self.algo} / {self.jobs_subdir} ({self.particle}) on {where}")
        if code.get("sha"):
            print(f"Code       : {code.get('branch')} {str(code['sha'])[:8]}"
                  f"{' (DIRTY)' if code.get('dirty') else ''}")
        print(f"Result     : Output/{self.outdir}/Reco/{self.name}/{self.reco_filename}")
        print()


def _local_manifest(algo: str, jobs_subdir: str):
    path = (RESEARCH_DIR / "Output" / ALGOS[algo]["outdir"] / "Jobs"
            / jobs_subdir / MANIFEST_NAME)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _remote_manifest(algo: str, jobs_subdir: str):
    path = f"{REMOTE_BASE}/Output/{ALGOS[algo]['outdir']}/Jobs/{jobs_subdir}/{MANIFEST_NAME}"
    rc, out, _ = ssh(f"cat {shlex.quote(path)} 2>/dev/null", capture=True)
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def resolve_run(target: str, where_hint: str | None = None,
                algo_hint: str | None = None) -> RunRef:
    """Find the run named by `target` and load its manifest.

    `target` is '<name>[/<variant>]', optionally prefixed 'coral:' or 'fitqun:'.
    Both algorithms and both machines are searched, so a run is found wherever
    it lives — and an ambiguous id is reported rather than silently resolved to
    whichever one a config file happened to name.
    """
    algo = algo_hint
    if ":" in target:
        algo, _, target = target.partition(":")
        algo = algo.lower()
        if algo not in ALGOS:
            raise SystemExit(f"Unknown algo prefix '{algo}:' (use {'/'.join(ALGOS)})")
    jobs_subdir = target.strip("/")

    algos = [algo] if algo else list(ALGOS)
    wheres = [where_hint] if where_hint else ["local", "server"]

    found = []
    for a in algos:
        for w in wheres:
            manifest = (_local_manifest(a, jobs_subdir) if w == "local"
                        else _remote_manifest(a, jobs_subdir))
            # A laptop copy of a SERVER run's manifest (Sextant fetches them for
            # provenance) is not a local run; resolving to it merges nothing.
            if w == "local" and manifest and manifest.get("where") == "server":
                continue
            if manifest:
                found.append(RunRef(a, jobs_subdir, w, manifest))

    if not found:
        hint = "" if algo else " (searched both algorithms, laptop and server)"
        raise SystemExit(
            f"No run '{jobs_subdir}' found{hint}.\n"
            f"List what exists with:  ./exp.py runs")
    if len(found) > 1:
        # Prefer the most recent submission, but say so — silently guessing is
        # what the old shared-config behaviour did.
        found.sort(key=lambda r: r.manifest.get("submitted_at", 0), reverse=True)
        others = ", ".join(f"{r.algo}:{r.jobs_subdir}@{r.where}" for r in found[1:])
        print(f"[NOTE] '{jobs_subdir}' matches several runs; using the most recent "
              f"({found[0].algo}@{found[0].where}). Also present: {others}")
        print(f"       Disambiguate with e.g. './exp.py <cmd> {found[0].algo}:{jobs_subdir} "
              f"--where {found[0].where}'\n")
    return found[0]


def run_target_from_args(args, spec_path: Path) -> RunRef:
    """The run a non-submit command should act on: the positional id if given,
    otherwise the spec's target (announced, so it is never a surprise)."""
    if getattr(args, "run", None):
        return resolve_run(args.run, args.where, args.algo)
    exp = Experiment(spec_path, args)
    print(f"[NOTE] No run id given — using the spec's target "
          f"'{exp.algo}:{exp.jobs_subdir}' from {spec_path.name}.\n"
          f"       Pass a run id explicitly to be sure: "
          f"./exp.py <cmd> {exp.algo}:{exp.jobs_subdir}\n")
    return resolve_run(exp.jobs_subdir, exp.where, exp.algo)


# -----------------------------
# submit
# -----------------------------

def _local_coral_root(slot: str) -> Path:
    root = coral_root_for(slot, RESEARCH_DIR)
    if not (root / "src").is_dir():
        raise SystemExit(
            f"Coral slot '{slot}' not found at {root}.\n"
            f"Create it with:  ./wt.py add {slot} <branch>")
    return root


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True).stdout.strip()


def _check_code_sync(exp: Experiment, remote_root: str):
    """Warn when the server's checkout is not what is being tested here.

    Compares commit SHAs, not branch names: two checkouts can sit on the same
    branch name and be many commits apart, which the old branch-name check
    reported as fine.
    """
    local_root = _local_coral_root(exp.slot)
    local_sha = _git(local_root, "rev-parse", "HEAD")
    local_branch = _git(local_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git(local_root, "status", "--porcelain"))

    rc, out, _ = ssh(f"git -C {shlex.quote(remote_root)} rev-parse HEAD 2>/dev/null; "
                     f"git -C {shlex.quote(remote_root)} rev-parse --abbrev-ref HEAD 2>/dev/null",
                     capture=True)
    lines = [l for l in out.splitlines() if l.strip()]
    remote_sha = lines[0] if lines else ""
    remote_branch = lines[1] if len(lines) > 1 else ""

    print(f"[CODE] laptop slot '{exp.slot}': {local_branch} {local_sha[:8]}"
          f"{' (DIRTY)' if dirty else ''}")
    print(f"[CODE] server slot '{exp.slot}': {remote_branch} {remote_sha[:8] or '(missing)'}")
    if not remote_sha:
        raise SystemExit(f"No Coral checkout at {remote_root} on the server.\n"
                         f"Create it with:  ./wt.py add {exp.slot} <branch> --server")
    if remote_sha != local_sha:
        # A hard stop, not a warning. A run submitted against the server's older
        # code still accepts the config you wrote for the newer one -- the keys
        # the old code does not read are simply ignored, so the arm runs, merges
        # and reports clean numbers for a feature that was never enabled. That
        # is the single most expensive failure mode this workspace has, and a
        # printed warning does not stop it: it scrolls past, or a grep filters
        # it out of the submit output.
        raise SystemExit(
            f"\n[STOP] the server is NOT at the commit you have here.\n"
            f"       laptop {local_sha[:8]}   server {remote_sha[:8]}\n"
            f"       The run would use the SERVER's code, and any config key\n"
            f"       only the newer code reads would be silently ignored.\n\n"
            f"       Sync it:            ./exp.py submit --sync ...\n"
            f"       Or, deliberately:   ./exp.py submit --force ...  "
            f"(runs the server's {remote_sha[:8]})\n")
    if dirty:
        print("[WARNING] the laptop checkout has uncommitted changes; they are NOT on the "
              "server and will not be in this run.\n")


def _sync_code(exp: Experiment, remote_root: str):
    """Push this slot's branch to origin and fast-forward the server's worktree.

    Kept explicit (--sync) rather than automatic: pushing is a visible action,
    and a silent push on every submit is exactly the kind of surprise this
    rework is trying to remove.
    """
    local_root = _local_coral_root(exp.slot)
    branch = _git(local_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch in ("", "HEAD"):
        raise SystemExit(f"slot '{exp.slot}' is in a detached HEAD — check out a branch first")
    if _git(local_root, "status", "--porcelain"):
        raise SystemExit(f"slot '{exp.slot}' has uncommitted changes — commit them before "
                         f"--sync (the server can only run what is pushed)")

    print(f"[SYNC] pushing {branch} to origin from slot '{exp.slot}'")
    if subprocess.run(["git", "-C", str(local_root), "push", "origin", branch]).returncode != 0:
        raise SystemExit("push to origin failed — resolve it and retry")

    print(f"[SYNC] fetching {branch} into the server's slot '{exp.slot}'")
    rc = ssh(f"git -C {shlex.quote(remote_root)} fetch origin {shlex.quote(branch)} && "
             f"git -C {shlex.quote(remote_root)} checkout -B {shlex.quote(branch)} "
             f"FETCH_HEAD")
    if rc != 0:
        raise SystemExit("server-side fetch/checkout failed")


def cmd_submit(exp: Experiment, args):
    exp.describe()
    if not exp.events:
        raise SystemExit("events is 0 — set [experiment] events in the spec, or pass -n")

    if exp.where == "local":
        return _submit_local(exp)
    return _submit_server(exp, args)


def _submit_local(exp: Experiment):
    runner = SCRIPTS_DIR / ALGOS[exp.algo]["local_runner"]
    cmd = [LOCAL_PYTHON, runner, "-n", exp.events, particle_flag(exp.particle),
           "--jobs-subdir", exp.jobs_subdir, "--workers", exp.workers,
           "--production", exp.production, "--events-per-job", exp.events_per_job,
           "--start", exp.start]
    if exp.algo == "coral":
        cmd += ["--threads", exp.threads, "--coral-root", _local_coral_root(exp.slot)]
        if exp.coral_config:
            raise SystemExit("coral_config presets carry server paths; run them with where = server")
        local_config = coral_config_for(exp.slot, local=True)
        if local_config.exists():
            cmd += ["--config", exp.materialise_config(
                local_config, exp.local_jobs_dir)]
    if exp.algo == "fitqun" and exp.params:
        cmd += ["--params", _resolve_params_path(exp)]
    if exp.fetch:
        cmd += ["--fetch"]
    print("Local run — this executes AND merges in one go; keep the terminal open.")
    print(f"Monitor from another terminal:  ./exp.py status   |   "
          f"./exp.py logs {exp.jobs_subdir}\n")
    raise SystemExit(run_local(cmd))


def _resolve_params_path(exp: Experiment) -> Path:
    params_path = Path(exp.params)
    if not params_path.is_absolute():
        params_path = RESEARCH_DIR / params_path
    if not params_path.exists():
        raise SystemExit(f"[experiment] params file not found: {params_path}")
    return params_path


def _deploy_cluster_scripts():
    """Push the cluster scripts to the server before using them.

    They are edited here and versioned with the rest of Analysis/Scripts, so the
    server always runs the current copy instead of whatever was last hand-edited
    over ssh.
    """
    files = (sorted(CLUSTER_DIR.glob("*.py")) + sorted(CLUSTER_DIR.glob("*.zsh"))
             + sorted(CLUSTER_DIR.glob("*.json")))      # evtnum_gaps.json: runlib's numbering registry
    print(f"[SHIP] {len(files)} cluster script(s) -> {REMOTE_HOST}:{REMOTE_BASE}/")
    result = subprocess.run(["rsync", "-az", *[str(f) for f in files],
                             f"{REMOTE_HOST}:{REMOTE_BASE}/"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"failed to ship cluster scripts: {result.stderr.strip()}")


def _submit_server(exp: Experiment, args):
    a = ALGOS[exp.algo]
    remote_coral = str(coral_root_for(exp.slot, Path(REMOTE_BASE)))

    _deploy_cluster_scripts()

    if exp.algo == "coral":
        if args.sync:
            _sync_code(exp, remote_coral)
        if args.force:
            print("[CODE] --force: not checking that the server matches this checkout")
        else:
            _check_code_sync(exp, remote_coral)

    # The config travels into the run directory itself, not over the server's
    # shared Software/Coral/config.ini — two submits racing through that shared
    # slot is how runs used to end up with each other's physics.
    remote_run = f"{REMOTE_BASE}/Output/{exp.algo_outdir}/Jobs/{exp.jobs_subdir}"
    submit = (f"source {REMOTE_BASE}/{a['env']} && python {REMOTE_BASE}/{a['submit']} "
              f"-n {exp.events} {particle_flag(exp.particle)} "
              f"--jobs-subdir {shlex.quote(exp.jobs_subdir)} "
              f"--job-name {shlex.quote(exp.job_name)} "
              f"--production {exp.production} "
              f"--events-per-job {exp.events_per_job} "
              f"--start {exp.start} "
              f"--straggler-cut {exp.straggler_cut} "
              f"--submitted-by exp.py")

    if exp.algo == "coral":
        staged = f"{remote_run}/_incoming_config.ini"
        ssh(f"mkdir -p {shlex.quote(remote_run)}")
        base = (coral_root_for(exp.slot, RESEARCH_DIR) / exp.coral_config
                if exp.coral_config else coral_config_for(exp.slot))
        if not base.exists():
            raise SystemExit(f"coral_config not found: {base}")
        local_config = exp.materialise_config(base, exp.local_jobs_dir)
        print(f"[SHIP] {local_config.name} -> {staged}")
        subprocess.run(["scp", str(local_config), f"{REMOTE_HOST}:{staged}"], check=True)
        submit += f" --config {shlex.quote(staged)} --coral-root {shlex.quote(remote_coral)}"
        if exp.input_dir:
            submit += f" --input-dir {shlex.quote(exp.input_dir)}"
    elif exp.params:
        if exp.input_dir:
            submit += f" --input-dir {shlex.quote(exp.input_dir)}"
        params_path = _resolve_params_path(exp)
        staged = f"{remote_run}/_incoming_params.dat"
        ssh(f"mkdir -p {shlex.quote(remote_run)}")
        print(f"[SHIP] {params_path.name} -> {staged}")
        subprocess.run(["scp", str(params_path), f"{REMOTE_HOST}:{staged}"], check=True)
        submit += f" --params {shlex.quote(staged)}"

    mem = exp.mem
    if (exp.algo == "coral" and not mem and not exp.gpus
            and subprocess.run(["git", "-C", str(_local_coral_root(exp.slot)), "merge-base",
                                "--is-ancestor", MEM_FIX_SHA, "HEAD"], capture_output=True).returncode):
        mem = "5G"
        print(f"[MEM] code without {MEM_FIX_SHA} (Cherenkov-profile memory fix) peaks at 4.3 GB: --mem 5G")
    if exp.time:
        submit += f" --time {exp.time}"
    if mem:
        submit += f" --mem {shlex.quote(mem)}"
    if exp.cpus:
        submit += f" --cpus {int(exp.cpus)}"
    if exp.partition:
        submit += f" --partition {shlex.quote(exp.partition)}"
    if exp.gpus:
        submit += f" --gpus {int(exp.gpus)}"
    for e in exp.env:
        submit += f" --env {shlex.quote(e)}"
    if exp.submit_delay != "":
        submit += f" --submit-delay {float(exp.submit_delay):g}"
    if args.force:
        submit += " --force"

    rc, out = ssh_tee(submit)
    if rc == 0 and exp.straggler_cut:
        _launch_reaper(exp, out)
    if rc == 0:
        print(f"\nMonitor with:  ./exp.py status")
        print(f"Merge later:   ./exp.py merge {exp.algo}:{exp.jobs_subdir}")
    raise SystemExit(rc)


# -----------------------------
# Straggler reaping
# -----------------------------

SBATCH_ID_RE = re.compile(r"Submitted batch job (\d+)")


def _fmt_walltime(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _parse_walltime(text: str):
    parts = text.strip().split(":")
    if not (2 <= len(parts) <= 3) or not all(p.strip().isdigit() for p in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def _launch_reaper(exp: Experiment, submit_out: str):
    """Ship reap_stragglers.py and run it as a watchdog job for this submission.

    With events packed per job there are far fewer, far longer jobs, so the long
    tail this was written for is mostly gone; it stays for wide scans.
    """
    job_ids = SBATCH_ID_RE.findall(submit_out)
    keep = runlib.reap_keep_count(len(job_ids), exp.straggler_cut)
    if not job_ids or keep >= len(job_ids):
        print(f"\n[REAP] straggler_cut = {exp.straggler_cut:g}% but the run is too small "
              f"to cut ({len(job_ids)} jobs) — no watchdog submitted.")
        return

    remote_dir = f"{REMOTE_BASE}/Output/{exp.algo_outdir}/Jobs/{exp.jobs_subdir}"
    plan_path  = f"{remote_dir}/{REAPER_PLAN_NAME}"
    log_stem   = f"{remote_dir}/logs/reap"

    print(f"\n[SHIP] {REAPER_SCRIPT} -> {REMOTE_HOST}")
    subprocess.run(["scp", str(SCRIPTS_DIR / REAPER_SCRIPT),
                    f"{REMOTE_HOST}:{REMOTE_BASE}/{REAPER_SCRIPT}"], check=True)

    plan = {"cut_percent": exp.straggler_cut, "poll_seconds": REAPER_POLL_S,
            "manifest": f"{remote_dir}/{MANIFEST_NAME}",
            "groups": {str(exp.events_per_job): job_ids}}
    ssh(f"printf '%s' {shlex.quote(json.dumps(plan))} > {shlex.quote(plan_path)}")

    job_seconds = _parse_walltime(exp.time) if exp.time else None
    if job_seconds is None:
        job_seconds = 900 + 240 * exp.events_per_job
    inner = (f"source {REMOTE_BASE}/{ALGOS[exp.algo]['env']} && "
             f"python {REMOTE_BASE}/{REAPER_SCRIPT} --plan {shlex.quote(plan_path)}")
    sbatch = ("sbatch --account=hyperk --partition=htc "
              f"--job-name={shlex.quote('reap_' + exp.job_name)} "
              f"--time={_fmt_walltime(job_seconds + REAPER_MARGIN_S)} "
              f"--mem=1G --cpus-per-task=1 "
              f"--output {shlex.quote(log_stem + '.out')} "
              f"--error {shlex.quote(log_stem + '.err')} "
              f"--wrap {shlex.quote(inner)}")
    rc, out, err = ssh(sbatch, capture=True)
    if rc != 0:
        print(f"[REAP] [WARNING] could not submit the watchdog: {err.strip() or out.strip()}")
        print("       The run is unaffected — it just keeps every straggler.")
        return
    print(out.strip())
    print(f"[REAP] cancelling the slowest {len(job_ids) - keep} job(s) once {keep} finish")
    print("[REAP] NOTE: the Timing CSV will then only cover the surviving jobs — "
          "use straggler_cut = 0 for timing runs.")


# -----------------------------
# Rendering helpers
# -----------------------------

def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def _bar(done: int, total, width: int = 20) -> str:
    if not total or total <= 0:
        return "[" + "?" * width + "]"
    frac = max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac*100:5.1f}%"


def _parse_log_cached(path: Path, cache: dict) -> tuple:
    """(events_seen, finished) for one Coral job log, memoised on mtime+size."""
    try:
        st = path.stat()
    except OSError:
        return 0, False
    key = (st.st_mtime, st.st_size)
    hit = cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    text = path.read_text(errors="replace")
    value = (len(CORAL_EVENT_RE.findall(text)), bool(CORAL_DONE_RE.search(text)))
    cache[path] = (key, value)
    return value


_LOG_CACHE: dict = {}


def _merged_at(reco_mtime, submitted_at):
    """When this run's own merged output was written, or None.

    A Reco file OLDER than the submission belongs to a previous run that reused
    the same name -- treating it as this run's output would hide the new run
    behind a stale result.
    """
    if reco_mtime is None:
        return None
    if submitted_at is not None and reco_mtime < submitted_at:
        return None
    return reco_mtime


def _terminal_status(is_live: bool, cancelled: bool, produced, expected, merged_at=None,
                     reaped: int = 0) -> str:
    if is_live:
        return "RUNNING"
    if merged_at is not None:
        return "MERGED"
    if expected and produced is not None and produced >= expected - reaped:
        return "DONE"
    return "CANCELLED" if cancelled else "STOPPED"


def _is_stale(rec: dict) -> bool:
    return (rec["status"] == "MERGED" and rec.get("merged_at") is not None
            and (time.time() - rec["merged_at"]) > MERGED_SHOW_HOURS * 3600)


# -----------------------------
# Local discovery
# -----------------------------

PROC_RE            = re.compile(r"run_(coral|fitqun)_local\.py")
JOBS_SUBDIR_ARG_RE = re.compile(r"--jobs-subdir\s+(\S+)")


def _local_live_processes() -> dict:
    """jobs_subdir -> {algo} for every running local driver."""
    try:
        out = subprocess.run(["ps", "-axo", "pid,command", "-ww"],
                             capture_output=True, text=True).stdout
    except Exception:
        return {}
    live = {}
    for line in out.splitlines():
        m = PROC_RE.search(line)
        sub = JOBS_SUBDIR_ARG_RE.search(line)
        if m and sub:
            live[sub.group(1)] = {"algo": m.group(1)}
    return live


def _local_run_dirs(jobs_root: Path):
    """Run directories at the two depths submit creates (<name>/, <name>/<variant>/).
    Bounded on purpose: a recursive glob also descends every job_*/ directory."""
    if not jobs_root.exists():
        return
    for level1 in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        if level1.name in ("logs", "sbatch"):
            continue
        yield level1
        for level2 in sorted(p for p in level1.iterdir() if p.is_dir()):
            if level2.name not in ("logs", "sbatch", "code", "numba_cache") \
                    and not level2.name.startswith("job_"):
                yield level2


def _build_local_record(algo: str, jobs_dir: Path, jobs_subdir: str, manifest, is_live: bool):
    if manifest is None:
        return None
    particle = manifest.get("particle")
    events   = manifest.get("events")
    expected = manifest.get("expected_jobs")
    prefix   = manifest.get("input_prefix", "")
    name     = manifest.get("name") or jobs_subdir.split("/")[0]
    variant  = manifest.get("variant", "")
    started  = manifest.get("submitted_at")
    cancelled = manifest.get("cancelled_at") is not None

    err_logs = sorted((jobs_dir / "logs").glob("*.err")) if (jobs_dir / "logs").is_dir() else []
    root_patterns = ((f"job_*/fitqun_{prefix}_00000000-*_rwcs.root", f"job_*/fitqun_{prefix}-*_rwcs.root")
                     if prefix else ("job_*/*.root",))
    produced = len({f for pattern in root_patterns for f in jobs_dir.glob(pattern)})

    if algo == "coral" and err_logs:
        parsed = [_parse_log_cached(lf, _LOG_CACHE) for lf in err_logs]
        done = sum(1 for _, finished in parsed if finished)
        events_done = sum(n for n, _ in parsed)
    else:
        done, events_done = produced, None

    reco = RESEARCH_DIR / "Output" / ALGOS[algo]["outdir"] / "Reco" / name / \
        runlib.reco_filename(algo, particle, variant)
    reco_mtime = reco.stat().st_mtime if reco.exists() else None
    merged_at = manifest.get("merged_at") or _merged_at(reco_mtime, started)

    started_n = min(len(err_logs), expected) if expected else len(err_logs)
    unfinished = max(started_n - done, 0)
    elapsed = time.time() - started if started else None
    rate = eta = None
    if events_done and elapsed and events and is_live and elapsed > 0:
        rate = events_done / elapsed
        eta = (events - events_done) / rate if rate > 0 else None

    return {
        "algo": algo, "jobs_subdir": jobs_subdir, "name": name, "variant": variant,
        "particle": particle, "where": "local",
        "status": _terminal_status(is_live, cancelled, produced, expected, merged_at),
        "done": done, "running": unfinished if is_live else 0,
        "stalled": 0 if is_live else unfinished,
        "pending": max(expected - started_n, 0) if expected else None,
        "expected": expected, "produced": produced,
        "events_total": events, "events_done": events_done, "rate": rate, "eta": eta,
        "elapsed": elapsed, "merged_at": merged_at,
        "code": manifest.get("code") or {},
        "reco": str(reco.relative_to(RESEARCH_DIR)),
    }


def _discover_local_runs() -> list:
    live = _local_live_processes()
    records = []
    for algo, meta in ALGOS.items():
        jobs_root = RESEARCH_DIR / "Output" / meta["outdir"] / "Jobs"
        for jobs_dir in _local_run_dirs(jobs_root):
            jobs_subdir = str(jobs_dir.relative_to(jobs_root))
            manifest = None
            try:
                manifest = json.loads((jobs_dir / MANIFEST_NAME).read_text())
            except (OSError, json.JSONDecodeError):
                pass
            live_info = live.get(jobs_subdir)
            is_live = bool(live_info and live_info["algo"] == algo)
            rec = _build_local_record(algo, jobs_dir, jobs_subdir, manifest, is_live)
            if rec is not None and not _is_stale(rec):
                records.append(rec)
    return records


# -----------------------------
# Server discovery
# -----------------------------

# One ssh round-trip gathers everything `status` needs. Manifests now live one
# per run directory with their own logs alongside, so the find is small — the
# old global Jobs/logs held 170k files and walking it was what made the first
# status call take about a minute.
_REMOTE_STATUS_CMD = f"""
squeue -u bcarew -h -o '%i|%j|%t' 2>/dev/null
echo '===RECO==='
find {REMOTE_BASE}/Output/Coral/Reco {REMOTE_BASE}/Output/fiTQun/Reco -mindepth 2 -maxdepth 2 -name '*.root' -printf '%T@ %p\\n' 2>/dev/null
echo '===MANIFESTS==='
for m in $(find {REMOTE_BASE}/Output/Coral/Jobs {REMOTE_BASE}/Output/fiTQun/Jobs \\
             -mindepth 1 -maxdepth 3 \\
             \\( -name logs -o -name sbatch -o -name code -o -name numba_cache \\
                -o -name 'job_*' \\) -prune -o \\
             -name {MANIFEST_NAME} -mtime -{MANIFEST_LOOKBACK_DAYS} -print 2>/dev/null); do
  d=$(dirname "$m")
  p=$(tr -d '\\n' < "$m" | sed -n 's/.*"input_prefix"[ ]*:[ ]*"\\([^"]*\\)".*/\\1/p')
  if [ -n "$p" ]; then
    n=$(find "$d" -mindepth 2 -maxdepth 2 \\( -name "fitqun_${{p}}_00000000-*_rwcs.root" -o -name "fitqun_${{p}}-*_rwcs.root" \\) 2>/dev/null | wc -l)
  else
    n=$(find "$d" -mindepth 2 -maxdepth 2 -name '*.root' 2>/dev/null | wc -l)
  fi
  echo '@@BEGIN@@'; cat "$m"; echo '@@PRODUCED@@'; echo "$n"; echo '@@END@@'
done
"""


def _parse_server_output(out: str):
    head, sep, rest = out.partition("===RECO===")
    if not sep:
        return [], {}, []
    reco_part, sep, rest = rest.partition("===MANIFESTS===")
    if not sep:
        return [], {}, []

    queue = []
    for line in head.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            # Array ids ('123_4') collapse to the parent id so they still match
            # what sbatch reported at submit time.
            queue.append((parts[0].split("_")[0], parts[1], parts[2]))

    # '<mtime> <path>' — how a run merged before manifests recorded merged_at is
    # still recognised as merged rather than reported as READY TO MERGE.
    merged = {}
    for ln in reco_part.splitlines():
        stamp, sep2, path = ln.strip().partition(" ")
        if sep2:
            try:
                merged[path] = float(stamp)
            except ValueError:
                continue

    manifests = []
    for block in rest.split("@@BEGIN@@")[1:]:
        try:
            json_part, rest_b = block.split("@@PRODUCED@@", 1)
            produced_part, _ = rest_b.split("@@END@@", 1)
            manifests.append((json.loads(json_part.strip()), int(produced_part.strip())))
        except (ValueError, json.JSONDecodeError):
            continue
    return queue, merged, manifests


def _discover_server_runs():
    """Returns (records, error) in a single ssh round-trip."""
    rc, out, err = ssh(_REMOTE_STATUS_CMD, capture=True)
    if rc != 0:
        return [], (err.strip() or f"ssh exited {rc}")

    queue, merged, manifests = _parse_server_output(out)
    records = []
    for manifest, produced in manifests:
        algo = manifest.get("algo")
        if algo not in ALGOS:
            continue
        reco_remote = (f"{REMOTE_BASE}/Output/{ALGOS[algo]['outdir']}/Reco/"
                       f"{manifest.get('name')}/"
                       f"{runlib.reco_filename(algo, manifest.get('particle', ''), manifest.get('variant', ''))}")
        job_ids = set(str(j) for j in (manifest.get("job_ids") or []))
        job_name = manifest.get("job_name", "")
        if job_ids:
            mine = [state for jid, _n, state in queue if jid in job_ids]
        else:
            mine = [state for _j, jname, state in queue if jname and jname == job_name]
        pending = sum(1 for s in mine if s == "PD")
        running = len(mine) - pending

        # SLURM is authoritative for what is still queued; everything else in the
        # submission has left the queue and is therefore finished. Deriving `done`
        # this way guarantees the buckets add up to the number of jobs submitted.
        expected = len(job_ids) or manifest.get("expected_jobs")
        done = max(expected - pending - running, 0) if expected else None
        reaped = (manifest.get("reaped") or {}).get("cancelled", 0)
        submitted_at = manifest.get("submitted_at")
        merged_at = manifest.get("merged_at") or _merged_at(merged.get(reco_remote),
                                                            submitted_at)

        records.append({
            "algo": algo, "jobs_subdir": manifest.get("jobs_subdir"),
            "name": manifest.get("name"), "variant": manifest.get("variant", ""),
            "particle": manifest.get("particle"), "where": "server",
            "status": _terminal_status(bool(mine), manifest.get("cancelled_at") is not None,
                                       produced, expected, merged_at, reaped),
            "done": done, "running": running, "stalled": 0, "pending": pending,
            "expected": expected, "produced": produced,
            "events_total": manifest.get("events"), "events_done": None,
            "rate": None, "eta": None,
            "elapsed": (time.time() - submitted_at) if submitted_at else None,
            "merged_at": merged_at, "reaped": manifest.get("reaped"),
            "code": manifest.get("code") or {},
            "events_per_job": manifest.get("events_per_job"),
            "reco": manifest.get("merged_to") or reco_remote.replace(REMOTE_BASE + "/", ""),
        })
    return [r for r in records if not _is_stale(r)], None


# -----------------------------
# status / runs
# -----------------------------

def _render_record(marker: str, rec: dict) -> list:
    pad = " " * len(marker)
    label = f"{rec['algo']} / {rec['jobs_subdir']}"
    if rec.get("particle"):
        label += f" ({rec['particle']})"
    lines = [f"{marker}[{rec['status']:>9s}] {label}"]

    code = rec.get("code") or {}
    if code.get("sha"):
        lines.append(f"{pad}  code   {code.get('branch', '?')} {str(code['sha'])[:8]}"
                     f"{' DIRTY' if code.get('dirty') else ''}")
    if rec.get("expected"):
        lines.append(f"{pad}  jobs   {rec['done']:>5d}/{rec['expected']:<5d} "
                     + _bar(rec["done"], rec["expected"])
                     + (f"  x{rec['events_per_job']} ev/job"
                        if rec.get("events_per_job") else ""))
    bits = [f"{rec['done']} done"]
    for key in ("running", "stalled", "pending"):
        if rec.get(key):
            bits.append(f"{rec[key]} {key}")
    lines.append(f"{pad}         " + ", ".join(bits)
                 + (f"  of {rec['expected']} submitted" if rec.get("expected") else ""))

    if rec.get("events_total") and rec.get("events_done") is not None:
        line = (f"{pad}  events {rec['events_done']:>5d}/{rec['events_total']:<5d} "
                + _bar(rec["events_done"], rec["events_total"]))
        if rec.get("rate"):
            line += f"  {rec['rate']*60:.1f}/min"
            if rec.get("eta") is not None:
                line += f"  ETA ~{_fmt_dur(rec['eta'])}"
        lines.append(line)

    reaped = rec.get("reaped") or {}
    n_reaped = reaped.get("cancelled", 0)
    if n_reaped:
        lines.append(f"{pad}         reaped {n_reaped} straggler(s) — merge uses the rest")
    produced, done = rec.get("produced"), rec.get("done")
    if produced is not None and done is not None and produced < done - n_reaped:
        lines.append(f"{pad}         {produced} output file(s) written — "
                     f"{done - n_reaped - produced} finished job(s) produced nothing "
                     f"(failed/timed out)")

    target = f"{rec['algo']}:{rec['jobs_subdir']}"
    if rec["status"] == "MERGED":
        lines.append(f"{pad}         merged {_fmt_dur(time.time() - rec['merged_at'])} ago"
                     + (f" -> {rec['reco']}" if rec.get("reco") else ""))
    elif rec["status"] == "DONE":
        lines.append(f"{pad}         READY TO MERGE — ./exp.py merge {target}")
    elif rec["status"] == "CANCELLED":
        lines.append(f"{pad}         cancelled — ./exp.py merge {target} salvages the "
                     f"{produced} completed job(s)")
    elif rec["status"] == "STOPPED":
        lines.append(f"{pad}         incomplete and not running — crashed or timed out; "
                     f"./exp.py merge {target} salvages the {produced} that finished")
    if rec.get("elapsed") is not None:
        lines.append(f"{pad}         submitted {_fmt_dur(rec['elapsed'])} ago")
    lines.append("")
    return lines


class _ServerPoller:
    """Fetches server state on a background thread so the local section stays
    on screen while the cluster query is in flight."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.records, self.error = [], None
        self.fetched_at = None
        self._busy = False
        self._lock = threading.Lock()
        if enabled:
            self.refresh()

    def refresh(self):
        with self._lock:
            if self._busy or not self.enabled:
                return
            self._busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        records, error = _discover_server_runs()
        with self._lock:
            self.records, self.error = records, error
            self.fetched_at = time.time()
            self._busy = False

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def wait(self, timeout: float):
        deadline = time.time() + timeout
        while self.busy and time.time() < deadline:
            time.sleep(0.05)


def _section(title: str, records, empty_msg: str, highlight: str | None) -> list:
    lines = [title]
    for rec in records:
        here = highlight is not None and f"{rec['algo']}:{rec['jobs_subdir']}" == highlight
        lines += _render_record("-> " if here else "   ", rec)
    if len(lines) == 1:
        lines.append(empty_msg + "\n")
    return lines


def _status_lines(poller: _ServerPoller, highlight: str | None) -> list:
    lines = _section("=== LOCAL (this laptop) ===", _discover_local_runs(),
                     "Nothing currently running or awaiting merge.", highlight)
    if not poller.enabled:
        lines += ["=== SERVER (CC-IN2P3) ===", "(skipped — --local)\n"]
        return lines
    if poller.fetched_at is None:
        lines += ["=== SERVER (CC-IN2P3) ===", "Querying the cluster…\n"]
    elif poller.error:
        lines += ["=== SERVER (CC-IN2P3) ===",
                  f"[WARNING] could not reach the server: {poller.error}\n"]
    else:
        age = time.time() - poller.fetched_at
        suffix = "  (refreshing…)" if poller.busy else f"  (as of {_fmt_dur(age)} ago)"
        lines += _section("=== SERVER (CC-IN2P3) ===" + suffix, poller.records,
                          "Nothing queued and nothing awaiting merge.", highlight)
    return lines


def cmd_status(args):
    highlight = None
    if args.run:
        target = args.run
        highlight = target if ":" in target else None
        if highlight is None:
            # No algo prefix: highlight whichever algo actually has this run.
            for algo in ALGOS:
                if _local_manifest(algo, target):
                    highlight = f"{algo}:{target}"
                    break
            else:
                highlight = f"coral:{target}"

    poller = _ServerPoller(enabled=not args.local)
    watch = not args.once and sys.stdout.isatty()
    interval = max(args.interval, 2)

    if not watch:
        poller.wait(timeout=90)
        print("\n".join(_status_lines(poller, highlight)))
        print("Act on a run with:  ./exp.py {logs|merge|pull|cancel} <algo>:<name>/<variant>")
        return

    try:
        while True:
            if poller.fetched_at and time.time() - poller.fetched_at >= interval:
                poller.refresh()
            out = ["\x1b[H\x1b[2J"]        # home + clear, so the view updates in place
            out.append(f"exp.py status — live, refreshing every {interval}s "
                       f"(Ctrl-C to stop)   [{time.strftime('%H:%M:%S')}]\n")
            out += _status_lines(poller, highlight)
            out.append("Act on a run with:  ./exp.py {logs|merge|pull|cancel} "
                       "<algo>:<name>/<variant>")
            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()
            time.sleep(min(interval, 2))
    except KeyboardInterrupt:
        print()


def cmd_runs(args):
    """Compact inventory — what exists, where, and what to type next."""
    rows = _discover_local_runs()
    if not args.local:
        server, error = _discover_server_runs()
        if error:
            print(f"[WARNING] server unreachable: {error}")
        rows += server
    if not rows:
        print("No runs found.")
        return
    rows.sort(key=lambda r: (r["where"], r["algo"], r["jobs_subdir"]))
    print(f"{'WHERE':<7} {'STATUS':>9}  {'ALGO':<7} {'RUN':<34} {'PARTICLE':<14} JOBS")
    for r in rows:
        jobs = (f"{r['done']}/{r['expected']}" if r.get("expected") else "-")
        print(f"{r['where']:<7} {r['status']:>9}  {r['algo']:<7} "
              f"{r['jobs_subdir']:<34} {str(r.get('particle') or '-'):<14} {jobs}")


# -----------------------------
# logs / merge / pull / cancel
# -----------------------------

def _initial_tail_lines(requested: int, nfiles: int) -> int:
    return requested if nfiles <= 8 else max(1, 200 // max(nfiles, 1))


def cmd_logs(run: RunRef, lines: int):
    """Stream a run's own job logs. The log directory comes from the run, so this
    can no longer show one algorithm's output while another is running."""
    run.describe()
    if run.where == "local":
        log_dir = run.local_root / "logs"
        err_logs = sorted(log_dir.glob("*.err")) if log_dir.exists() else []
        if not err_logs:
            print(f"No job logs yet under {log_dir}.")
            return
        init = _initial_tail_lines(lines, len(err_logs))
        print(f"Streaming {len(err_logs)} job log(s) under {log_dir} "
              f"(last {init} lines each, then following)")
        print("(Ctrl-C to stop; re-run to include jobs that start later)\n")
        try:
            subprocess.run(["tail", f"-n{init}", "-f", *(str(f) for f in err_logs)])
        except KeyboardInterrupt:
            print()
        return

    pattern = f"{run.remote_root}/logs/*.err"
    rc, out, _ = ssh(f"ls -1 {pattern} 2>/dev/null | wc -l", capture=True)
    try:
        count = int(out.strip()) if rc == 0 else 0
    except ValueError:
        count = 0
    if count == 0:
        print(f"No job logs yet on the server ({pattern}).")
        print("Jobs may still be queued — check ./exp.py status.")
        return
    init = _initial_tail_lines(lines, count)
    print(f"Streaming {count} server log(s) (last {init} lines each, then following)")
    print(f"  {pattern}")
    print("(Ctrl-C to stop; re-run to include jobs that start later)\n")
    subprocess.run(["ssh", "-t", REMOTE_HOST, f"tail -n{init} -f {pattern}"])


def cmd_merge(run: RunRef):
    run.describe()
    if run.manifest.get("submitting"):
        # a submit killed part-way (a timeout around `sextant run`) leaves this set; merging would
        # silently shrink the sample to the files that were submitted
        raise SystemExit(f"REFUSED: {run.jobs_subdir} was never fully submitted "
                         f"({len(run.manifest.get('job_ids') or [])} of {run.manifest.get('expected_jobs')} jobs). "
                         f"Submit the remaining files as a run with [run] start, or resubmit.")
    if run.where == "local":
        runner = SCRIPTS_DIR / ALGOS[run.algo]["local_runner"]
        raise SystemExit(run_local(
            [LOCAL_PYTHON, runner, "--merge-only", particle_flag(run.particle),
             "--jobs-subdir", run.jobs_subdir, "--production", run.production]))

    _deploy_cluster_scripts()
    a = ALGOS[run.algo]
    output = (f"Output/{run.outdir}/Reco/{run.name}/{run.reco_filename}")
    if run.algo == "coral":
        merge_cmd = (f"source {REMOTE_BASE}/{a['env']} && python {REMOTE_BASE}/{a['merge']} "
                     f"--run {shlex.quote(run.jobs_subdir)} --output {shlex.quote(output)}")
    else:
        merge_cmd = (f"source {REMOTE_BASE}/{a['env']} && python {REMOTE_BASE}/{a['merge']} "
                     f"{particle_flag(run.particle)} "
                     f"--input-dir {shlex.quote(run.remote_root)} "
                     f"--output {shlex.quote(output)} --production {run.production}")
        if run.manifest.get("multiring") or run.manifest.get("params"):
            merge_cmd += " --multiring"
    t0 = time.time()
    rc = ssh(merge_cmd)
    if rc == 0:
        # merge cost per job feeds the auto packer (measured_merge_s_per_job)
        jobs = (len(run.manifest.get("job_ids") or [])
                or int(run.manifest.get("expected_jobs") or 0))
        if jobs:
            log = RESEARCH_DIR / "Output" / run.outdir / "Timing" / MERGE_LOG_NAME
            log.parent.mkdir(parents=True, exist_ok=True)
            new = not log.exists()
            with open(log, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["run", "jobs", "seconds", "merged_at"])
                w.writerow([run.jobs_subdir, jobs, f"{time.time() - t0:.1f}", int(time.time())])
        print(f"\nMerged. Fetch the results with:  ./exp.py pull {run.algo}:{run.jobs_subdir}")
    raise SystemExit(rc)


def cmd_pull(run: RunRef):
    run.describe()
    if run.where == "local":
        print("Local run — results are already on this machine, nothing to pull.")
        return
    for sub in ("Reco", "Timing"):
        remote = f"{REMOTE_HOST}:{REMOTE_BASE}/Output/{run.outdir}/{sub}/{run.name}/"
        local  = RESEARCH_DIR / "Output" / run.outdir / sub / run.name
        local.mkdir(parents=True, exist_ok=True)
        print(f"[PULL] {sub}/{run.name}/")
        subprocess.run(["rsync", "-avz", remote, str(local) + "/"], check=True)
    reco = RESEARCH_DIR / "Output" / run.outdir / "Reco" / run.name / run.reco_filename
    print(f"\nDone. Point the comparison notebook at:\n  {reco}")


def cmd_cancel(args, spec_path: Path):
    """Cancel one run's jobs. `--all` is the old blanket behaviour and now has to
    be asked for: `scancel -u bcarew` also kills every other agent's jobs."""
    if args.all:
        print("[SERVER] scancel -u bcarew  (EVERY job you own)")
        rc = ssh("scancel -u bcarew")
        print("  ok" if rc == 0 else f"  ssh/scancel returned {rc}")
        _kill_local_processes()
        return

    run = run_target_from_args(args, spec_path)
    run.describe()
    if run.where == "server":
        job_ids = [str(j) for j in (run.manifest.get("job_ids") or [])]
        if not job_ids:
            raise SystemExit("This run's manifest has no SLURM job ids — nothing to cancel "
                             "safely. Use --all if you really mean every job you own.")
        print(f"[SERVER] scancel {len(job_ids)} job(s) belonging to {run.jobs_subdir}")
        ssh(f"scancel {' '.join(job_ids)}")
        ssh(f"python3 - <<'PY'\n"
            f"import json,pathlib,time\n"
            f"p=pathlib.Path({json.dumps(run.remote_root + '/' + MANIFEST_NAME)})\n"
            f"d=json.loads(p.read_text()); d['cancelled_at']=time.time()\n"
            f"p.write_text(json.dumps(d,indent=2))\nPY")
    else:
        _kill_local_processes(run.jobs_subdir)
        path = run.local_root / MANIFEST_NAME
        try:
            data = json.loads(path.read_text())
            data["cancelled_at"] = time.time()
            path.write_text(json.dumps(data, indent=2))
        except (OSError, json.JSONDecodeError):
            pass
    print("Cancelled.")


def _kill_local_processes(jobs_subdir: str | None = None):
    """Kill local reconstruction processes.

    The driver pools out worker subprocesses running the reconstruction binaries
    (Coral's `fitqun` CLI / fiTQun's `runfiTQun`); killing just the Python
    wrapper leaves those orphaned and still burning CPU.
    """
    print("[LOCAL] killing running local reconstruction processes"
          + (f" for {jobs_subdir}" if jobs_subdir else ""))
    if jobs_subdir:
        patterns = [f"run_coral_local.py.*{jobs_subdir}", f"run_fitqun_local.py.*{jobs_subdir}"]
    else:
        patterns = ["Analysis/Scripts/run_coral_local.py",
                    "Analysis/Scripts/run_fitqun_local.py",
                    "envs/HyperK/bin/fitqun", "bin/runfiTQun"]
    killed = False
    for pattern in patterns:
        if subprocess.run(["pkill", "-9", "-f", pattern]).returncode == 0:
            killed = True
            print(f"  killed process(es) matching: {pattern}")
    if not killed:
        print("  nothing local was running")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["submit", "status", "runs", "merge", "pull", "logs", "cancel",
                                 "pack"],
                        help="What to do (see above)")
    parser.add_argument("run", nargs="?", default=None,
                        help="Run id '<name>[/<variant>]', optionally 'coral:'/'fitqun:' "
                             "prefixed. Required in practice for logs/merge/pull/cancel; "
                             "ignored by submit.")
    parser.add_argument("-f", "--file", type=Path, default=DEFAULT_SPEC,
                        help=f"Experiment spec for `submit` (default: {DEFAULT_SPEC})")
    parser.add_argument("-n", "--events", type=int, default=None, help="Override events")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument("--variant", default=None, help="Override variant ('' to clear)")
    parser.add_argument("--algo", choices=sorted(ALGOS), default=None, help="Override algo")
    parser.add_argument("--where", choices=["server", "local"], default=None,
                        help="Override where (submit), or disambiguate a run id")
    parser.add_argument("--particle", choices=PARTICLES, default=None, help="Override particle")
    parser.add_argument("--production", choices=sorted(PRODUCTIONS), default=None,
                        help="Override WCSim production")
    parser.add_argument("--start", type=int, default=None, metavar="N",
                        help="First input file index (default 0). Use it to draw a "
                             "validation sample from files a calibration run did not see.")
    parser.add_argument("--events-per-job", default=None,
                        help=f"Events packed into each job, 1..{EVENTS_PER_FILE}, or 'auto' "
                             "(default: server runs packed for the shortest wall time "
                             "from measured s/event and merge cost)")
    parser.add_argument("--s-per-event", type=float, default=None,
                        help="s/event estimate for auto packing (default: learnt from "
                             "recent Timing CSVs)")
    parser.add_argument("--slot", choices=SLOTS, default=None,
                        help="Coral worktree slot to run from (default main)")
    parser.add_argument("--set", dest="set_config", action="append", metavar="SECTION.KEY=VALUE",
                        help="coral submit: patch the run's frozen config.ini (repeatable). "
                             "Same effect as a [coral_config] entry in the spec.")
    parser.add_argument("--sync", action="store_true",
                        help="submit: push this slot's branch to origin and fast-forward "
                             "the server's slot before submitting")
    parser.add_argument("--force", action="store_true",
                        help="submit: reuse a run directory even if it already holds a run")
    parser.add_argument("--params", default=None,
                        help="fitqun only: parameter override file (repo-relative or absolute)")
    parser.add_argument("--input-dir", default=None,
                        help="coral server submits only: remote input-file directory override "
                             "(naming/prefix unchanged) -- for samples outside PROD_BASE")
    parser.add_argument("--submit-delay", type=float, default=None,
                        help="seconds between sbatch calls, overriding the spec's "
                             "[server] submit_delay. Paces the start burst on /sps; "
                             "0 disables.")
    parser.add_argument("--time", default=None,
                        help="server submits: fixed SLURM wall-time (H:MM:SS) override "
                             "(default: auto-scaled from events/job)")
    parser.add_argument("--straggler-cut", default=None,
                        help="server only: percentage of the slowest jobs to cancel once "
                             "the rest have finished, or 'auto' (default: 5%%, 10%% at "
                             ">= 100 jobs; local runs never cut)")
    parser.add_argument("--all", action="store_true",
                        help="cancel: cancel EVERY job you own, not just this run's")
    parser.add_argument("--lines", type=int, default=40,
                        help="logs: initial lines per job before following (default 40)")
    parser.add_argument("--once", action="store_true",
                        help="status: one snapshot instead of the live view")
    parser.add_argument("--interval", type=int, default=15,
                        help="status: seconds between live refreshes (default 15)")
    parser.add_argument("--local", action="store_true",
                        help="status/runs: skip the cluster query entirely")
    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(Experiment(args.file, args), args)
    elif args.command == "pack":
        # machine-readable packing for a spec/CLI shape (Sextant resolves 'auto' with it)
        exp = Experiment(args.file, args)
        print(f"events_per_job={exp.events_per_job} jobs={exp.expected_jobs} "
              f"straggler_cut={exp.straggler_cut:g}")
        print(f"# {exp.pack_note}")
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "cancel":
        cmd_cancel(args, args.file)
    elif args.command == "logs":
        cmd_logs(run_target_from_args(args, args.file), args.lines)
    elif args.command == "merge":
        cmd_merge(run_target_from_args(args, args.file))
    elif args.command == "pull":
        cmd_pull(run_target_from_args(args, args.file))


if __name__ == "__main__":
    main()
