#!/usr/bin/env python3

"""Submit the hybrid sequential-fit scan: a different unhit selector per fit step.

The uniform-vs-ranked scan showed the two selectors win on different axes:

  * ranked  keeps the ring-edge PMTs that constrain the VERTEX and DIRECTION
  * uniform keeps MOMENTUM honest, because momentum is set by the total predicted
            light and so needs an unbiased sample of the unhit population

Coral's fit already separates those variables across steps, so the selector can
follow them:

  step_0  x,y,z,theta,phi   Nelder-Mead, ~500 evaluations   EXPENSIVE -> rank
  step_1  t,p               L-BFGS-B,    ~60 evaluations    CHEAP     -> uniform
  step_2  t,x,y,z,...,p     Nelder-Mead, ~800 evaluations   EXPENSIVE -> rank

That asymmetry is the whole idea: PMTs are cheap in the momentum step (few
evaluations) and expensive in the geometry steps (many), so the sequence can
afford a *generous uniform* momentum step while ranking hard everywhere else.

    ./run_pmt_hybrid_scan.py --particle electron --where server -n 200
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
# xgbsimple: 4 ring-geometry features, 30 trees, ~27 ms/event -- the best ranker
# per unit cost in the first scan (the 31-feature models pay ~0.4 s/event for the
# cKDTree block and rank no better).
RANKED_MODEL = "xgb_simple.npz"

# (name, step sequence). Each entry is '<selector>:<keep fraction>' per fit step.
SEQUENCES = {
    # controls: one selector throughout, so the hybrids have something to beat
    "ctl_uniform": "uniform:0.10, uniform:0.10, uniform:0.10",   # production default
    "ctl_ranked":  "model:0.10, model:0.10, model:0.10",         # all-ranked
    # rank the geometry steps, keep momentum uniform
    "hyb_a": "model:0.10, uniform:0.20, uniform:0.20",           # only step_0 ranked
    "hyb_b": "model:0.10, uniform:0.20, model:0.10",             # both NM steps ranked
    "hyb_c": "model:0.10, uniform:0.30, model:0.10",             # generous momentum step
    "hyb_d": "model:0.05, uniform:0.30, model:0.05",             # aggressive + generous p
    "hyb_e": "model:0.20, uniform:0.20, model:0.20",             # conservative ranking
    "hyb_f": "model:0.05, uniform:0.20, uniform:0.10",           # cheap step_0, uniform tail
    "hyb_g": "ring:0.10, uniform:0.30, ring:0.10",               # analytic ranker, no model
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle", required=True)
    ap.add_argument("--where", required=True, choices=["local", "server"])
    ap.add_argument("-n", "--events", type=int, required=True)
    ap.add_argument("--slot", default="c")
    ap.add_argument("--name", default=None)
    ap.add_argument("--events-per-job", type=int, default=40)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name = args.name or f"pmthyb_{args.particle}"
    model = str((LOCAL_MODELS if args.where == "local" else SERVER_MODELS) / RANKED_MODEL)
    plan = [(k, v) for k, v in SEQUENCES.items()
            if not args.only or any(o in k for o in args.only)]

    print(f"{len(plan)} runs: {name} / {args.particle} / {args.events} events "
          f"on {'this laptop' if args.where == 'local' else 'CC-IN2P3'}\n")

    failures = []
    for i, (variant, steps) in enumerate(plan, 1):
        cmd = [sys.executable, str(EXP), "submit", "-f", str(SPEC),
               "--name", name, "--variant", variant, "--slot", args.slot,
               "-n", str(args.events), "--particle", args.particle,
               "--where", args.where, "--events-per-job", str(args.events_per_job),
               "--set", "pmt_selection.method=model",
               "--set", f"pmt_selection.model={model}",
               "--set", f"pmt_selection.steps={steps}"]
        print(f"[{i}/{len(plan)}] {variant:<12} {steps}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(variant)
            print(result.stdout[-1500:], result.stderr[-1500:])

    if failures:
        print(f"\n{len(failures)} submission(s) FAILED: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(plan)} runs submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
