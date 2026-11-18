#!/usr/bin/env python3

"""Submit the unhit-PMT-selection scan: every selector x every keep fraction.

One run per (selector, keep fraction) pair plus a single all-PMT reference, for
one particle type. Everything goes through exp.py, so each run freezes its own
code + config and is independently mergeable.

Reconstruction time is one of the two axes of the comparison, so a particle type
must be scanned entirely on ONE machine -- laptop and cluster per-event times are
not comparable. Pick `--where` per particle, never per variant.

    ./run_pmt_selection_scan.py --particle electron --where local  -n 150
    ./run_pmt_selection_scan.py --particle muon     --where server -n 200
    ./run_pmt_selection_scan.py --particle pi0      --where server -n 200 --dry-run
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPTS_DIR.parents[1]
EXP = SCRIPTS_DIR / "exp.py"
SPEC = RESEARCH_DIR / "Analysis/Experiments/default.ini"

LOCAL_MODELS = SCRIPTS_DIR / "DNLLPredictorTrainedModels/exported"
SERVER_MODELS = Path("/sps/hyperk/bcarew/PMTSelectorModels")

# Selector -> the [pmt_selection] settings it needs. `model` is filled in from
# the machine-appropriate model directory.
SELECTORS = {
    "uniform":   {"method": "uniform"},
    "ring":      {"method": "ring"},          # distance from the prefit Cherenkov cone
    "ringctr":   {"method": "ring_center"},   # distance from the prefit ring axis
    "xgbsimple": {"method": "model", "model": "xgb_simple.npz"},
    "xgbv02":    {"method": "model", "model": "xgb_v02.npz"},
    "xgbfull":   {"method": "model", "model": "xgb_full.npz"},
    "mlp":       {"method": "model", "model": "mlp.npz"},
}
FRACTIONS = (0.30, 0.20, 0.10, 0.05)


def variants(models_dir: Path, only=None):
    """(variant name, [pmt_selection] settings) for every run in the scan."""
    plan = [("none", {"method": "uniform", "keep_fraction": "1.0"})]
    for selector, base in SELECTORS.items():
        for fraction in FRACTIONS:
            settings = dict(base, keep_fraction=f"{fraction}")
            if "model" in settings:
                settings["model"] = str(models_dir / settings["model"])
            plan.append((f"{selector}_{int(round(fraction * 100)):02d}", settings))
    if only:
        plan = [(name, s) for name, s in plan if any(p in name for p in only)]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle", required=True)
    ap.add_argument("--where", required=True, choices=["local", "server"])
    ap.add_argument("-n", "--events", type=int, required=True)
    ap.add_argument("--slot", default="c")
    ap.add_argument("--name", default=None, help="experiment name (default pmtsel_<particle>)")
    ap.add_argument("--events-per-job", type=int, default=25)
    ap.add_argument("--only", nargs="*", help="only variants whose name contains one of these")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name = args.name or f"pmtsel_{args.particle}"
    models_dir = LOCAL_MODELS if args.where == "local" else SERVER_MODELS
    plan = variants(models_dir, args.only)

    print(f"{len(plan)} runs: {name} / {args.particle} / {args.events} events "
          f"on {'this laptop' if args.where == 'local' else 'CC-IN2P3'}\n")

    failures = []
    for i, (variant, settings) in enumerate(plan, 1):
        cmd = [sys.executable, str(EXP), "submit", "-f", str(SPEC),
               "--name", name, "--variant", variant, "--slot", args.slot,
               "-n", str(args.events), "--particle", args.particle,
               "--where", args.where, "--events-per-job", str(args.events_per_job)]
        for key, value in settings.items():
            cmd += ["--set", f"pmt_selection.{key}={value}"]

        print(f"[{i}/{len(plan)}] {variant:<14} " +
              " ".join(f"{k}={Path(v).name if k == 'model' else v}"
                       for k, v in settings.items()))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(variant)
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])

    if failures:
        print(f"\n{len(failures)} submission(s) FAILED: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(plan)} runs submitted. Monitor with: ./exp.py status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
