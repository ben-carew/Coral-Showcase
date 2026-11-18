"""Turning finished runs into the arms the validation half consumes.

Merge, pull, then load. The first two are `exp.py`'s job; the third is where an
arm acquires the provenance that makes its numbers attributable -- the run's
frozen code SHA, branch, dirty flag and config digest, read out of the run's own
manifest rather than out of whatever the working tree happens to say now.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from ..io.dataset import Arm
from ..io.schema import HYP_ELECTRON
from .launcher import exp_command, run_ids
from .spec import SOURCE_FILE, SOURCE_OFFICIAL, SOURCE_RUN, ExperimentSpec


def merge_and_pull(spec: ExperimentSpec, workspace: Path, *, pull=None) -> int:
    """Merge every runnable arm, and pull it back if the run was on the server."""
    pull = spec.where == "server" if pull is None else pull
    failed = []
    for run_id in run_ids(spec):
        print(f"\n=== merge {run_id} ===")
        if exp_command(spec, workspace, "merge", run_id).returncode:
            failed.append(f"merge {run_id}")
        if pull:
            print(f"=== pull {run_id} ===")
            if exp_command(spec, workspace, "pull", run_id).returncode:
                failed.append(f"pull {run_id}")
    if failed:
        # Named, not counted: a dropped ssh during one arm's pull leaves that
        # arm's results on the server, and a report built afterwards is a
        # comparison silently missing an arm.
        print("\n[FAILED] " + "; ".join(failed))
        print("Re-run `sextant collect` (already-collected arms are cheap to "
              "redo) before building the report.")
    return len(failed)


def reco_path(spec: ExperimentSpec, arm, workspace: Path) -> Path:
    """Where `exp.py`'s merge puts this arm's output."""
    from .. import productions

    algo_dir = "Coral" if arm.algorithm == "coral" else "fiTQun"
    prefix = "coral" if arm.algorithm == "coral" else "fitqun"
    # Must match runlib.reco_filename exactly -- exp.py's merge names the file
    # from the sample and the variant, and the variant is the arm name.
    filename = f"{prefix}_{spec.sample}_{arm.name}.root"
    return Path(workspace) / "Output" / algo_dir / "Reco" / spec.name / filename


#: Where the run directories live on the cluster.
REMOTE_BASE = "/sps/hyperk/bcarew"
REMOTE_HOST = "cca.in2p3.fr"


def _run_dir(spec: ExperimentSpec, arm, base) -> str:
    algo_dir = "Coral" if arm.algorithm == "coral" else "fiTQun"
    return f"{base}/Output/{algo_dir}/Jobs/{spec.name}/{arm.name}"


def manifest_for(spec: ExperimentSpec, arm, workspace: Path) -> dict:
    """The run's manifest -- fetched from the server if it is not local.

    `exp.py pull` brings back Reco/ and Timing/ but deliberately not Jobs/ (the
    per-job outputs are huge), so for a server run reported from the laptop the
    manifest is not here. Without it the report's provenance table is blank
    exactly where it matters most: the code SHA, branch and dirty flag that make
    a number attributable. It is a few hundred bytes, so it is fetched and
    cached beside the run instead of being reported as unknown.
    """
    path = Path(_run_dir(spec, arm, str(Path(workspace)))) / "manifest.json"
    if not path.exists() and spec.where == "server":
        _fetch_provenance(spec, arm, workspace)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _fetch_provenance(spec: ExperimentSpec, arm, workspace: Path):
    """Copy a server run's manifest and frozen config back, once.

    One rsync per file rather than one call listing all three: only Coral runs
    have a config.ini and only fiTQun runs a params.dat, and the rsync shipped
    with macOS is too old for --ignore-missing-args, so a single call naming a
    file that does not exist fails for all of them.
    """
    import shutil
    import subprocess

    if getattr(arm, "source", "run") == "file":
        return                      # a file arm has no run directory to fetch from
    local = Path(_run_dir(spec, arm, str(Path(workspace))))
    remote = _run_dir(spec, arm, REMOTE_BASE)
    if Path(remote).is_dir():       # running on the server: the run is right here
        if Path(remote).resolve() != local.resolve():
            local.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "config.ini", "params.dat"):
                if (Path(remote) / name).exists():
                    shutil.copy2(Path(remote) / name, local / name)
        return
    local.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "config.ini", "params.dat"):
        try:
            subprocess.run(
                ["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                 f"{REMOTE_HOST}:{remote}/{name}", str(local) + "/"],
                capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            # Provenance is worth trying for, not worth failing the report for.
            continue


def config_digest(spec: ExperimentSpec, arm, workspace: Path) -> str:
    """Short hash of the frozen config, so two arms can be told apart at a glance."""
    run_dir = Path(_run_dir(spec, arm, str(Path(workspace))))
    for name in ("config.ini", "params.dat"):
        candidate = run_dir / name
        if candidate.exists():
            return hashlib.sha1(candidate.read_bytes()).hexdigest()[:8]
    return ""


def load_arms(spec: ExperimentSpec, workspace: Path, *, limit_files=None) -> list:
    """Build an Arm per spec arm, from wherever that arm's results live."""
    from .. import productions

    arms = []
    for arm in spec.arms:
        meta = {"run": f"{spec.name}/{arm.name}", "scheme": arm.scheme,
                "where": spec.where, "source": arm.source}

        if arm.source == SOURCE_OFFICIAL:
            # Match the *indices* the run covered, not the first N files: the
            # shipped reconstruction is a scattered subset of the production,
            # so "first N" points at a different part of the sample.
            n_files = limit_files or _files_needed(spec)
            paths = productions.find_reference_reco_for_indices(
                spec.production, spec.sample, range(n_files))
            if not paths:
                print(f"[skip] {arm.label}: no official reconstruction files "
                      f"found (they live on the cluster — run this where "
                      f"/sps is mounted, or rsync them first)")
                continue
            meta["n_files"] = len(paths)
            arms.append(Arm.from_official_files(
                arm.label, paths, production=spec.production,
                sample=spec.sample, meta=meta))
            continue

        path = Path(arm.path) if arm.source == SOURCE_FILE else reco_path(
            spec, arm, workspace)
        if not path.is_absolute():
            path = Path(workspace) / path
        if not path.exists():
            print(f"[skip] {arm.label}: merged output not found at {path}")
            continue

        manifest = manifest_for(spec, arm, workspace)
        code = manifest.get("code", {})
        meta.update({"code_sha": code.get("sha", ""),
                     "branch": code.get("branch", ""),
                     "dirty": code.get("dirty", ""),
                     "config_digest": config_digest(spec, arm, workspace),
                     "events_requested": manifest.get("events", "")})
        arms.append(Arm.from_file(arm.label, path, algorithm=arm.algorithm,
                                  production=spec.production, sample=spec.sample,
                                  meta=meta))
    return arms


def _files_needed(spec: ExperimentSpec) -> int:
    """How many input files a run of this shape touches.

    One job reads ONE input file and takes `events_per_job` events from it, so
    a run spans as many files as it has jobs -- not `events / events_per_file`.
    At 1000 events and 10 per job that is 100 files, not 10, and getting it
    wrong makes the reference arm read a tenth of the sample the other arm ran
    on: the two barely share an event and every paired comparison collapses.
    """
    per_job = spec.events_per_job or 1
    return max(1, -(-spec.events // per_job))


def load_truth(spec: ExperimentSpec, workspace: Path):
    """The truth for this experiment, from the spec or the conventional path."""
    from ..io.dataset import Truth

    if spec.truth:
        path = Path(spec.truth)
        if not path.is_absolute():
            path = Path(workspace) / path
        if not path.exists():
            raise FileNotFoundError(
                f"truth file not found: {path}\n"
                f"Build it with:  sextant truth --production {spec.production} "
                f"--sample {spec.sample} --output {path}")
        subevents = None
        if spec.truth_subevents:
            subevents = Path(spec.truth_subevents)
            if not subevents.is_absolute():
                subevents = Path(workspace) / subevents
        return Truth.from_file(path, subevents=subevents)

    conventional = (Path(workspace) / "Output" / "Truth"
                    / f"truth_{spec.sample}.root")
    if conventional.exists():
        return Truth.from_file(conventional)
    raise FileNotFoundError(
        f"no truth file given in the spec and none at {conventional}. "
        f"Set [experiment] truth = <path>, or build one with `sextant truth`.")


def pid_samples(spec: ExperimentSpec, arms: list, truth, workspace: Path) -> tuple:
    """(pid_samples, pid_truths) for `analysis.standard_report`, or (None, None).

    The PID section needs a true-electron AND a true-muon sample; one
    experiment holds one. `[experiment] pid_partner` names the spec of the other
    sample, whose arms are paired with this one's by ARM NAME (the [arm.<name>]
    section), and are relabelled with this experiment's labels so each pair is
    one line in every PID figure. Unpaired arms are left out, with a note.
    """
    from .. import productions

    if not spec.pid_partner:
        return None, None
    partner_path = Path(spec.pid_partner)
    if not partner_path.is_absolute():
        partner_path = Path(workspace) / partner_path
    partner = ExperimentSpec.load(partner_path)
    mine_s = productions.get(spec.production).sample(spec.sample)
    theirs_s = productions.get(partner.production).sample(partner.sample)
    # A beam channel is scored by its CC lepton on CC single-ring events.
    mine, theirs = mine_s.particle or mine_s.lepton, theirs_s.particle or theirs_s.lepton
    if not mine or not theirs or mine == theirs:
        print(f"[PID skipped] pid_partner {partner.name} is not the other "
              f"particle ({spec.sample} vs {partner.sample})")
        return None, None
    partner_arms = {arm.meta.get("run", "").split("/")[-1]: arm
                    for arm in load_arms(partner, workspace)}
    by_name = {arm.meta.get("run", "").split("/")[-1]: arm for arm in arms}
    paired = [name for name in by_name if name in partner_arms]
    unpaired = sorted(set(by_name) - set(paired))
    if unpaired:
        print(f"[PID] no {partner.name} arm to pair with: {', '.join(unpaired)}")
    if not paired:
        return None, None
    for name in paired:
        partner_arms[name].label = by_name[name].label
    partner_truth = load_truth(partner, workspace)
    mine_arms = [by_name[n] for n in paired]
    theirs_arms = [partner_arms[n] for n in paired]
    if mine_s.kind == "beam":
        mine_arms = _only_events(mine_arms, cc_single_ring_events(truth))
    if theirs_s.kind == "beam":
        theirs_arms = _only_events(theirs_arms, cc_single_ring_events(partner_truth))
    return ({mine: mine_arms, theirs: theirs_arms},
            {mine: truth, theirs: partner_truth})


def cc_single_ring_events(truth):
    """Beam events whose only visible ring is the CC lepton: the PID sample."""
    ev = truth.events
    vis = truth.rings[truth.rings["visible"]]
    lepton = set(vis.loc[vis["is_primary_lepton"], "event"])
    one = ev.loc[ev["is_cc"].astype(bool) & (ev["n_visible_rings"] == 1), "event"]
    return np.array(sorted(set(one) & lepton))


def _only_events(arms, events):
    """Copies of `arms` whose fits keep only `events` (sub-event 0 is scored)."""
    return [dataclasses.replace(a, fits=a.fits[a.fits["event"].isin(events)]) for a in arms]
