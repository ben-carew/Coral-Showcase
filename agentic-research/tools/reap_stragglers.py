#!/usr/bin/env python3

"""Cancel the slowest tail of one experiment's SLURM jobs once most have finished.

Runs ON the cluster as a tiny watchdog job submitted by exp.py (see _launch_reaper
there), so the cut still happens with the laptop closed. It reads a plan file
written at submit time -- the run's SLURM ids grouped by how many events each job
reconstructs -- polls squeue, and as soon as (100 - cut)% of a group has left the
queue it scancels whatever is still going in that group.

Grouping matters because submit_{coral,fitqun}.py pack more than one event per job
above 1000 events: a two-event job is legitimately about twice as slow as a
one-event job, so a single global percentile would cancel the big jobs every time.
Each group gets its own cut.

Jobs are counted as finished when they leave the queue, whether they succeeded or
crashed -- the merge scripts already skip missing/corrupt job outputs, so a reaped
run merges exactly like a run that was cancelled by hand.

    reap_stragglers.py --plan Output/Coral/Jobs/<exp>/<variant>/reap_plan.json

Starting late is harmless: the watchdog only ever looks at the current queue, so a
reaper that gets scheduled after the fast jobs are already done reaps immediately.
"""

from pathlib import Path
import argparse
import getpass
import json
import math
import subprocess
import sys
import time

# squeue outages (a busy or restarting slurmctld) must never be read as "the jobs
# finished" -- that would scancel a whole run. Bail out instead after this many
# consecutive failures, leaving every job alone.
MAX_POLL_ERRORS = 10


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def alive_ids(user: str):
    """SLURM ids still queued or running for `user`, or None if squeue failed."""
    result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        log(f"squeue failed ({result.returncode}): {result.stderr.strip()}")
        return None
    # Array ids ('123_4') collapse to the parent, matching what sbatch reported.
    return {line.strip().split("_")[0] for line in result.stdout.splitlines() if line.strip()}


def record(manifest_path: Path, summary: dict):
    """Publish the outcome into the run's manifest so `exp.py status` can show it."""
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    data["reaped"] = summary
    try:
        manifest_path.write_text(json.dumps(data, indent=2))
    except OSError as exc:
        log(f"could not update manifest: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="Reap plan written by exp.py submit (reap_plan.json)")
    parser.add_argument("--poll", type=int, default=None,
                        help="Seconds between squeue polls (default: from the plan)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be cancelled without running scancel")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    cut = float(plan["cut_percent"])
    poll = args.poll or int(plan.get("poll_seconds", 60))
    manifest = Path(plan["manifest"]) if plan.get("manifest") else None
    user = plan.get("user") or getpass.getuser()

    # {events-per-job: [slurm ids]}. Keep counts round UP, so a group is only ever
    # cut when strictly more than `cut`% of it would survive -- small groups (and
    # any group of one) are simply never reaped.
    groups = {int(k): list(v) for k, v in plan["groups"].items()}
    keep = {n: math.ceil((1.0 - cut / 100.0) * len(ids)) for n, ids in groups.items()}

    log(f"watching {sum(len(v) for v in groups.values())} job(s), cutting the slowest {cut:g}%")
    for n in sorted(groups, reverse=True):
        log(f"  {len(groups[n]):>4d} job(s) x {n} event(s) — reap once {keep[n]} have finished")

    pending = {n: ids for n, ids in groups.items() if keep[n] < len(ids)}
    for n in sorted(set(groups) - set(pending)):
        log(f"  group of {len(groups[n])} x {n} event(s) is too small to cut — left alone")

    cancelled = {}
    errors = 0
    while pending:
        alive = alive_ids(user)
        if alive is None:
            errors += 1
            if errors >= MAX_POLL_ERRORS:
                log(f"giving up after {errors} squeue failures — no jobs were cancelled")
                return 1
            time.sleep(poll)
            continue
        errors = 0

        for n, ids in list(pending.items()):
            still = [j for j in ids if j in alive]
            finished = len(ids) - len(still)
            if not still:
                log(f"group {n} event(s)/job: all {len(ids)} finished on their own")
                del pending[n]
                cancelled[n] = 0
            elif finished >= keep[n]:
                log(f"group {n} event(s)/job: {finished}/{len(ids)} finished — "
                    f"cancelling the {len(still)} still going")
                if args.dry_run:
                    log(f"  (dry run) scancel {' '.join(still)}")
                else:
                    rc = subprocess.run(["scancel", *still]).returncode
                    if rc != 0:
                        log(f"  scancel returned {rc}")
                del pending[n]
                cancelled[n] = len(still)

        if pending:
            time.sleep(poll)

    total = sum(cancelled.values())
    log(f"done — cancelled {total} straggler(s); merge over the rest with ./exp.py merge")
    if manifest and not args.dry_run:
        record(manifest, {
            "at": time.time(),
            "cut_percent": cut,
            "cancelled": total,
            "by_events_per_job": {str(n): c for n, c in sorted(cancelled.items())},
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
