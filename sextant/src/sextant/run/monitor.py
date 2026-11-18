"""Progress, errors, and stopping things.

Thin wrappers over `exp.py`'s own commands, scoped to one experiment's arms.
The value added is scoping: `exp.py cancel --all` kills every job the user owns,
including other agents' and other experiments'; `sextant cancel <spec>` cancels
exactly the arms of that experiment and nothing else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .launcher import exp_command, run_ids, workdir_for
from .spec import SOURCE_RUN, ExperimentSpec

#: Log lines worth surfacing when the user asks what went wrong.
ERROR_PATTERNS = [
    re.compile(r"^\s*Traceback", re.I),
    re.compile(r"\bError\b|\bERROR\b"),
    re.compile(r"\bCRITICAL\b"),
    re.compile(r"slurmstepd:.*(CANCELLED|TIMEOUT|OUT OF MEMORY)", re.I),
    re.compile(r"\bSegmentation fault\b", re.I),
    re.compile(r"\bKilled\b"),
]

#: Lines matching these are noise, not failures -- a warning about a missing
#: optional tuning file appears in every healthy run.
IGNORE_PATTERNS = [
    re.compile(r"error.*=\s*[\d.]+", re.I),        # "...error = 0.03" in fit output
    re.compile(r"errorbar", re.I),
]


def status(spec: ExperimentSpec, workspace: Path, *, arm=None, full=False):
    """Progress of this experiment's arms.

    A focused per-arm summary by default. Calling `exp.py status` once per arm
    printed the entire cluster-wide dashboard once per arm, which for a
    two-arm experiment meant reading every unrelated run in the workspace
    twice to find two lines. `--full` gets that dashboard back deliberately.
    """
    selected = [spec.arm(arm)] if arm else list(spec.arms)
    runnable = [a for a in selected if a.source == SOURCE_RUN]
    static = [a for a in selected if a.source != SOURCE_RUN]

    if full:
        if not runnable:
            print("No runnable arms: every arm reads existing files.")
            return 0
        run_id = f"{runnable[0].algorithm}:{spec.name}/{runnable[0].name}"
        return exp_command(spec, workspace, "status", run_id).returncode

    print(f"Experiment {spec.name}  ({spec.production}/{spec.sample}, "
          f"{spec.events} events, {spec.where})\n")
    rows = _progress(spec, workspace, runnable) if runnable else []
    # Arms that read existing files are listed too. Omitting them made a
    # two-arm experiment look like a one-arm experiment.
    for a in static:
        rows.append({"arm": a.label, "expected": 0, "done": 0,
                     "state": "reads", "note": {
                         "official": "the production's own reconstruction",
                     }.get(a.source, f"existing file: {a.path}")})
    if not rows:
        print("  (no arms)")
        return 0
    width = max(len(r["arm"]) for r in rows)
    for row in rows:
        if row["state"] == "reads":
            print(f"  {row['arm']:<{width}}  {'reads':<9} "
                  f"{'':<22} {row['note']}")
            continue
        bar = _bar(row["done"], row["expected"])
        print(f"  {row['arm']:<{width}}  {row['state']:<9} {bar} "
              f"{row['done']}/{row['expected']} jobs   {row['note']}")
    print(f"\n  logs:   sextant logs <spec> --arm <name>"
          f"\n  errors: sextant errors <spec>"
          f"\n  full cluster dashboard: sextant status <spec> --full")
    return 0


def _bar(done: int, expected: int, width: int = 20) -> str:
    if not expected:
        return "[" + "-" * width + "]"
    filled = int(width * done / expected)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _progress(spec: ExperimentSpec, workspace: Path, arms) -> list:
    """Per-arm job counts, read from each run's own directory.

    Server runs are counted over ssh in ONE call for the whole experiment
    rather than one per arm -- a CC-IN2P3 handshake is most of the latency.
    """
    import json
    import subprocess

    rows = []
    remote_dirs = {}
    for arm in arms:
        algo_dir = "Coral" if arm.algorithm == "coral" else "fiTQun"
        relative = f"Output/{algo_dir}/Jobs/{spec.name}/{arm.name}"
        local = Path(workspace) / relative
        expected = -(-(arm.events or spec.events)
                     // (arm.events_per_job or spec.events_per_job))
        row = {"arm": arm.label, "expected": expected, "planned": expected,
               "done": 0, "state": "unknown", "note": ""}
        if spec.where == "server":
            remote_dirs[arm.label] = relative
        else:
            row.update(_count_local(local, expected))
        rows.append(row)

    if remote_dirs:
        counts = _count_remote(remote_dirs)
        for row in rows:
            row.update(counts.get(row["arm"], {}))
    # An arm that has produced all its jobs is done, whatever else is still in
    # the queue -- the SLURM count is workspace-wide and says nothing about
    # which arm the remaining jobs belong to.
    for row in rows:
        planned = row.pop("planned", None)
        if row["state"] in ("INTERRUPTED", "SUBMITTING"):
            continue
        if row["expected"] and row["done"] >= row["expected"]:
            row["state"] = "done"
            row["note"] = "ready to merge"
        if planned and row["expected"] and planned != row["expected"]:
            missing = planned - row["expected"]
            row["note"] = (f"{row['note']}; {missing} input file(s) absent "
                           f"from the production").lstrip("; ")
    return rows


def _count_local(run_dir: Path, expected: int) -> dict:
    if not run_dir.is_dir():
        return {"state": "not found", "note": "no run directory"}
    done = len(list(run_dir.glob("job_*/timing.json")))
    submitted = len(list((run_dir / "sbatch").glob("*.sh")))
    expected = submitted or expected
    return {"done": done, "expected": expected,
            "state": "done" if done >= expected else "running",
            "note": ""}


def _count_remote(relative_by_arm: dict) -> dict:
    """One ssh call: per-arm finished-job counts plus the live SLURM count."""
    import subprocess

    base = "/sps/hyperk/bcarew"
    pieces = []
    for label, relative in relative_by_arm.items():
        # Count timing.json, which each job writes in its epilogue -- so it
        # appears only when the job has actually finished. Counting output
        # files instead overstates progress badly: both Coral and fiTQun create
        # their output file when they START, so every running job looks done.
        pieces.append(
            f'printf "%s\\t%s:%s:%s\\n" {_quote(label)} '
            f'"$(find {base}/{relative} -name timing.json 2>/dev/null | wc -l)" '
            f'"$(ls {base}/{relative}/sbatch/*.sh 2>/dev/null | wc -l)" '
            f'"$(grep -c \'"submitting": true\' {base}/{relative}/manifest.json 2>/dev/null)"')
    command = "; ".join(pieces) + f'; echo "__LIVE__\t$(squeue -u bcarew -h | wc -l)"'
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
             "-o", "ControlPath=/tmp/.exp-ssh-%r@%h:%p", "-o",
             "ControlPersist=600", "cca.in2p3.fr", command],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {label: {"state": "unreachable", "note": str(exc)[:40]}
                for label in relative_by_arm}

    if result.returncode != 0 and not result.stdout.strip():
        # Without this the arms silently keep their initial "unknown" state and
        # the dashboard looks like a run that has not started, rather than like
        # a machine we could not reach.
        reason = (result.stderr or "").strip().splitlines()
        return {label: {"state": "unreachable",
                        "note": reason[-1][:60] if reason else "ssh failed"}
                for label in relative_by_arm}

    live = 0
    out = {}
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        label, value = line.split("\t", 1)
        if label == "__LIVE__":
            live = int(value or 0)
            continue
        # "finished:submitted". The submitted count is not always the spec's
        # arithmetic: the production's input files are not perfectly
        # contiguous, so a run over 340 indices can legitimately launch 339
        # jobs, and a bar computed from the spec sits at 339/340 for ever.
        finished, submitted, partial = (value.strip().split(":") + ["", ""])[:3]
        row = {"done": int(finished or 0), "partial": partial.strip() not in ("", "0")}
        if submitted.strip().isdigit() and int(submitted) > 0:
            row["expected"] = int(submitted)
        out[label] = row
    for label, row in out.items():
        partial = row.pop("partial")
        row["state"] = "running" if live else (
            "done" if row["done"] else "no output")
        row["note"] = f"{live} job(s) in the queue" if live else ""
        if partial:
            # the manifest is still marked mid-submission: either a submit is running now, or one
            # was killed part-way and the arm holds only the files submitted before it died
            row["state"] = "SUBMITTING" if live else "INTERRUPTED"
            row["note"] = "submission incomplete (manifest marked submitting)"
    return out


def _quote(text: str) -> str:
    import shlex
    return shlex.quote(text)


def logs(spec: ExperimentSpec, workspace: Path, arm: str, *, lines: int = 40):
    arm_spec = spec.arm(arm)
    if arm_spec.source != SOURCE_RUN:
        print(f"Arm '{arm}' reads existing files; it has no job logs.")
        return 0
    run_id = f"{arm_spec.algorithm}:{spec.name}/{arm_spec.name}"
    return exp_command(spec, workspace, "logs", run_id,
                       extra=["--lines", str(lines)]).returncode


def cancel(spec: ExperimentSpec, workspace: Path, *, arm=None):
    """Cancel this experiment's arms -- and only those.

    Deliberately never offers a `--all`: cancelling everything you own is a
    workspace-wide action that belongs to `exp.py`, where it is at least
    obvious that is what you are doing.
    """
    targets = (run_ids(spec) if arm is None
               else [f"{spec.arm(arm).algorithm}:{spec.name}/{spec.arm(arm).name}"])
    failures = 0
    for run_id in targets:
        print(f"Cancelling {run_id}")
        failures += bool(exp_command(spec, workspace, "cancel", run_id).returncode)
    return failures


#: Regex handed to the remote grep. Kept close to ERROR_PATTERNS but written
#: for POSIX grep -E, since a server run's logs are filtered where they live
#: rather than dragged back over ssh.
REMOTE_ERROR_RE = (r"Traceback|ERROR|CRITICAL|Segmentation fault|"
                   r"slurmstepd.*(CANCELLED|TIMEOUT|OUT OF MEMORY)|Killed")


def errors(spec: ExperimentSpec, workspace: Path, *, max_lines: int = 25) -> dict:
    """Scan each arm's job logs for real failures.

    Server runs are grepped where the logs live -- one ssh call for the whole
    experiment. An earlier version only looked locally, which meant the default
    workflow (`where = server`) reported "no local log directory" and nothing
    else, exactly when the command is most wanted.
    """
    arms = [a for a in spec.arms if a.source == SOURCE_RUN]
    if spec.where == "server":
        return _errors_remote(spec, arms, max_lines)
    return _errors_local(spec, workspace, arms, max_lines)


def _log_dir_for(spec: ExperimentSpec, arm, base) -> str:
    algo_dir = "Coral" if arm.algorithm == "coral" else "fiTQun"
    return f"{base}/Output/{algo_dir}/Jobs/{spec.name}/{arm.name}/logs"


def _errors_local(spec: ExperimentSpec, workspace: Path, arms, max_lines) -> dict:
    found = {}
    for arm in arms:
        log_dir = Path(_log_dir_for(spec, arm, str(Path(workspace))))
        if not log_dir.is_dir():
            found[arm.label] = ["(no log directory yet — has the run started?)"]
            continue
        hits = []
        for log in sorted(log_dir.glob("*")):
            if not log.is_file():
                continue
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                if any(p.search(line) for p in IGNORE_PATTERNS):
                    continue
                if any(p.search(line) for p in ERROR_PATTERNS):
                    hits.append(f"{log.name}: {line.strip()}")
                    if len(hits) >= max_lines:
                        break
            if len(hits) >= max_lines:
                break
        found[arm.label] = hits
    return found


def _errors_remote(spec: ExperimentSpec, arms, max_lines) -> dict:
    import shlex
    import subprocess

    base = "/sps/hyperk/bcarew"
    pieces = []
    for arm in arms:
        log_dir = _log_dir_for(spec, arm, base)
        pieces.append(
            f'echo "@@{arm.label}"; '
            f'if [ -d {shlex.quote(log_dir)} ]; then '
            f'grep -HE {shlex.quote(REMOTE_ERROR_RE)} {log_dir}/* 2>/dev/null '
            f'| head -{max_lines}; '
            f'else echo "!!no log directory yet"; fi')
    command = "; ".join(pieces)
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
             "-o", "ControlPath=/tmp/.exp-ssh-%r@%h:%p",
             "-o", "ControlPersist=600", "cca.in2p3.fr", command],
            capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return {arm.label: [f"(could not reach the server: {exc})"]
                for arm in arms}

    found, current = {}, None
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            current = line[2:]
            found[current] = []
        elif current is None:
            continue
        elif line.startswith("!!"):
            found[current].append(f"({line[2:]})")
        elif line.strip():
            # Strip the long remote path; the basename identifies the job.
            path, _, text = line.partition(":")
            found[current].append(f"{Path(path).name}: "
                                  f"{_strip_ansi(text).strip()}")
    for arm in arms:
        found.setdefault(arm.label, [])
    return found


def _strip_ansi(text: str) -> str:
    """Coral colours its log output; the escapes are noise in a summary."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def save_state(spec: ExperimentSpec, workspace: Path, launched) -> Path:
    """Record what was launched, so later commands need no re-derivation."""
    directory = workdir_for(spec, workspace)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.json"
    path.write_text(json.dumps({
        "experiment": spec.name,
        "spec": spec.source_path,
        "production": spec.production,
        "sample": spec.sample,
        "events": spec.events,
        "where": spec.where,
        "arms": [run.as_dict() for run in launched],
    }, indent=2))
    return path


def load_state(spec: ExperimentSpec, workspace: Path) -> dict:
    path = workdir_for(spec, workspace) / "experiment.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
