"""`sextant` -- the command line.

Two families of command, matching the two halves of the package:

**Driven by an experiment spec** (needs the workspace and the cluster):
    check    validate a spec and show how the arms' configurations line up
    run      launch every runnable arm through exp.py
    status   progress of this experiment's runs
    logs     stream one arm's job output
    errors   filtered failure summary across the arms
    cancel   cancel this experiment's runs, and only those
    collect  merge and pull finished arms
    report   collect (if needed) and build the report
    probe    per-PMT intermediates at a common parameter point

**Standalone** (needs only files):
    compare  build a report from reconstruction files named on the command line
    truth    build a truth file from WCSim input
    productions / schema / mapping   what the package knows

`compare` is the one to reach for when the experiment machinery is not wanted:
it takes labelled paths and a truth file and produces the same report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _workspace(args) -> Path:
    if getattr(args, "workspace", None):
        return Path(args.workspace).resolve()
    here = Path.cwd().resolve()
    for directory in [here, *here.parents]:
        if (directory / "Analysis/Scripts/exp.py").exists():
            return directory
    return here


def _load_spec(args):
    from .run.spec import ExperimentSpec
    return ExperimentSpec.load(args.spec)


# ---------------------------------------------------------------------------
# spec-driven commands
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    from .run.spec import check_configuration
    import pandas as pd

    spec = _load_spec(args)
    print(spec.describe())
    print()
    rows, issues = check_configuration(spec)
    table = pd.DataFrame(rows)
    interesting = table[table["kind"].isin(["shared", "calibration"])]
    columns = [c for c in interesting.columns
               if c not in ("coral_key", "fitqun_key")]
    print(interesting[columns].to_string(index=False))
    print()
    _report_fitqun_cost(spec, _workspace(args))

    if not issues:
        print("No configuration issues.")
        return 0
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        print(f"\n{len(errors)} error(s): the arms are not comparably "
              f"configured.")
        return 1
    return 0


def _report_fitqun_cost(spec, workspace):
    """Say which expensive fiTQun fits each arm's parameter file switches on.

    Silent unless there is something to say. The full official parameter files
    enable the multi-ring and pi0 fits, which cost ~20 minutes per event on the
    HK detector -- a single-ring comparison configured with one of those does
    not fail, it simply never finishes, and nothing else in the workflow says so
    until the jobs hit their wall time.
    """
    from pathlib import Path as _Path

    from .config import fitqun_params

    for arm in spec.arms:
        if arm.algorithm != "fitqun" or not arm.params:
            continue
        path = _Path(arm.params)
        if not path.is_absolute():
            path = _Path(workspace) / path
        try:
            found = fitqun_params.enabled_fits(fitqun_params.read(path))
        except (OSError, FileNotFoundError) as exc:
            print(f"\n[WARNING] {arm.label}: cannot read the parameter file "
                  f"({exc})")
            continue
        if not found:
            continue
        print(f"\n[COST   ] {arm.label}: {path.name} enables "
              f"{len(found)} expensive fit(s):")
        for name, description in found:
            print(f"            {name} -- {description}")
        print("            These dominate the runtime (the full HK-2024 files "
              "cost ~20 min/event\n            on this detector). For a "
              "single-ring comparison use a *_1Remu file.")


def cmd_run(args) -> int:
    from .run import launcher, monitor

    spec = _load_spec(args)
    workspace = _workspace(args)
    if not args.skip_check and cmd_check(args) != 0 and not args.force:
        print("\nRefusing to launch. Fix the configuration, or pass --force.")
        return 1
    launched = launcher.launch(spec, workspace, arms=args.arm, sync=args.sync,
                               dry_run=args.dry_run)
    if not args.dry_run:
        state = monitor.save_state(spec, workspace, launched)
        print(f"\nLaunched {len([l for l in launched if l.run_id])} arm(s). "
              f"State: {state}")
        print(f"Follow with:  sextant status {args.spec}")
    return 0


def cmd_status(args) -> int:
    from .run import monitor
    return monitor.status(_load_spec(args), _workspace(args), arm=args.arm,
                          full=args.full)


def cmd_logs(args) -> int:
    from .run import monitor
    return monitor.logs(_load_spec(args), _workspace(args), args.arm,
                        lines=args.lines)


def cmd_errors(args) -> int:
    from .run import monitor

    found = monitor.errors(_load_spec(args), _workspace(args))
    total = 0
    for arm, hits in found.items():
        print(f"\n=== {arm} ===")
        if not hits:
            print("  no errors found in this run's logs")
            continue
        for line in hits:
            print(f"  {line}")
        total += len(hits)
    return 1 if total else 0


def cmd_cancel(args) -> int:
    from .run import monitor
    return monitor.cancel(_load_spec(args), _workspace(args), arm=args.arm)


def cmd_collect(args) -> int:
    from .run import collect
    return collect.merge_and_pull(_load_spec(args), _workspace(args))


def cmd_report(args) -> int:
    from . import analysis
    from .run import collect

    spec = _load_spec(args)
    workspace = _workspace(args)
    if args.collect:
        collect.merge_and_pull(spec, workspace)

    arms = collect.load_arms(spec, workspace)
    if not arms:
        print("No arms could be loaded. Have they been merged and pulled? "
              "Try:  sextant collect <spec>")
        return 1

    # A missing arm must not pass quietly. `load_arms` skips what it cannot
    # find so that a partial look is still possible, but a *comparison* report
    # built without one of its arms looks complete and is not -- which is how a
    # dropped pull turned a two-arm study into a one-arm one with no warning.
    loaded = {arm.label for arm in arms}
    missing = [a.label for a in spec.arms if a.label not in loaded]
    if missing and not args.allow_missing:
        print(f"\nerror: {len(missing)} of {len(spec.arms)} arms could not be "
              f"loaded: {', '.join(missing)}")
        print("Their results are probably still on the server. Run "
              "`sextant collect <spec>` and try again,")
        print("or pass --allow-missing to report on the arms that did load.")
        return 1
    if missing:
        print(f"[WARNING] reporting without {', '.join(missing)}")
    truth = collect.load_truth(spec, workspace)
    pid_samples, pid_truths = collect.pid_samples(spec, arms, truth, workspace)
    # Output/Sextant/<experiment>/report by default: a validation report is an
    # output of the experiment and belongs with the rest of what that run
    # produced. Analysis/Reports/ is reserved for bespoke write-ups about Coral
    # development, which are authored rather than generated.
    outdir = Path(args.outdir or spec.outdir
                  or workspace / "Output" / "Sextant" / spec.name / "report")
    if not outdir.is_absolute():
        outdir = workspace / outdir

    path = analysis.standard_report(
        arms, truth, outdir, hypothesis=spec.hypothesis,
        reference=spec.reference_label(),
        subtitle=f"{spec.production} / {spec.sample} · {spec.events} events",
        pid_samples=pid_samples, pid_truths=pid_truths,
        crb_model=_crb_model(args, workspace))
    print(f"\nReport: {path}")
    print(f"Figures: {outdir / 'figures'}")
    print(f"Tables:  {outdir / 'tables'}")
    return 0


def _event_ids_in(path, workspace):
    """Event ids present in a merged reco file or a diagnostic CSV."""
    import pandas as pd

    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace) / p
    if p.suffix == ".csv":
        return pd.read_csv(p, usecols=["event"])["event"].unique()
    import uproot
    with uproot.open(p) as f:
        for tree in ("fits", "prefits"):
            if tree in f:
                return f[tree]["event"].array(library="np")
    raise ValueError(f"no fits tree in {p}")


def cmd_probe(args) -> int:
    from .probe import coral_probe, points as point_builder
    from .run import collect

    spec = _load_spec(args)
    workspace = _workspace(args)
    truth = collect.load_truth(spec, workspace)
    within = None
    if getattr(args, "events_in", None):
        within = _event_ids_in(args.events_in, workspace)
        print(f"Restricting the draw to the {len(within)} events found in "
              f"{args.events_in}.")
    events = point_builder.select_events(truth, n=args.events, seed=args.seed,
                                         within=within)
    import pandas as pd

    offset = 0.0
    if args.time_pedestal == "auto":
        if not args.events_in:
            print("--time-pedestal auto needs --events-in to measure the offset "
                  "against; probing at the raw truth time instead.")
        else:
            ref = pd.read_csv(args.events_in) if str(args.events_in).endswith(".csv") \
                else None
            if ref is not None and "t" in ref.columns:
                rings = truth.rings if hasattr(truth, "rings") else truth.primary
                offset = point_builder.time_pedestal(ref, rings)
                print(f"Time pedestal (algorithm origin minus truth): "
                      f"{offset:+.2f} ns; the truth point is shifted onto it.")
            else:
                print("could not measure a time pedestal from --events-in")
    elif args.time_pedestal not in (None, "none"):
        offset = float(args.time_pedestal)
        print(f"Time pedestal fixed at {offset:+.2f} ns.")

    table = point_builder.build(truth=truth, events=events, points=("truth",),
                                time_offset=offset)
    print(f"Probing {len(events)} events at the truth parameter point "
          f"({len(table)} points).")

    outdir = Path(args.outdir or workspace / "Output/Sextant" / spec.name / "probe")
    # Coral's FQManager writes into this directory and does not create it.
    outdir.mkdir(parents=True, exist_ok=True)
    from . import productions as registry

    # Only the files that actually hold the selected events. Handing Coral the
    # whole sample would make it walk every one of the production's files --
    # hundreds for a particle gun, ~6000 for a beam channel -- to find twenty
    # events, which is the difference between a minute and an afternoon.
    inputs = registry.find_inputs_for_events(spec.production, spec.sample, events)
    if not inputs:
        print("Input WCSim files are not reachable from here; the probe must "
              "run where the production is mounted (the cluster).")
        return 1
    print(f"Reading {len(inputs)} input file(s) that hold those events.")
    result = coral_probe.probe_events(args.config, inputs, table,
                                      coral_root=args.coral_root,
                                      outdir=str(outdir),
                                      max_events=args.events,
                                      production=spec.production,
                                      sample=spec.sample)
    written = coral_probe.write(result, outdir)
    print(f"Coral: probed {result['n_events']} events; wrote {written}")

    if not args.fitqun_params:
        print("\nNo --fitqun-params given, so only the Coral side was probed. "
              "A per-PMT comparison needs both codes evaluated at the SAME "
              "parameter point.")
        return 0

    from .probe import fitqun_probe

    params = Path(args.fitqun_params)
    if not params.is_absolute():
        params = Path(workspace) / params
    try:
        binary = fitqun_probe.find_binary(args.fqprobe)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1

    grouped = fitqun_probe.group_points_by_file(table, spec.production,
                                                spec.sample)
    print(f"\nfiTQun: probing the same points with {binary} "
          f"over {len(grouped)} file(s).")
    fq_tables = fitqun_probe.probe_files(
        grouped, params, stride=registry.get(spec.production).stride,
        binary=binary, workdir=str(outdir / "fqprobe"),
        fitqun_root=args.fitqun_root)
    fq = fq_tables["pmt_predictions"]
    if fq.empty:
        print("fiTQun probe produced no rows.")
        return 1
    fq_path = outdir / "pmt_predictions_fitqun.csv"
    fq.to_csv(fq_path, index=False)
    fq_tables["nll_terms"].to_csv(outdir / "nll_terms_fitqun.csv", index=False)
    print(f"fiTQun: {len(fq)} per-PMT rows -> {fq_path}")

    # The comparison itself: same events, same PMTs, same parameter point.
    from .metrics import intermediates

    comparison = intermediates.compare_at_point(
        result["pmt_predictions"], fq, labels=("coral", "fitqun"),
        point=point_builder.POINT_TRUTH if hasattr(point_builder, "POINT_TRUTH")
        else "truth")
    comparison_path = outdir / "pmt_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nMatched {len(comparison)} PMTs on both sides -> {comparison_path}")
    print(intermediates.pmt_summary(comparison, labels=("coral", "fitqun"))
          .round(4).to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# standalone commands
# ---------------------------------------------------------------------------

def cmd_compare(args) -> int:
    from . import analysis
    from .io import Arm, Truth

    arms = []
    for entry in args.arm:
        if "=" not in entry:
            print(f"--arm expects LABEL=PATH (got '{entry}')")
            return 1
        label, path = entry.split("=", 1)
        arms.append(Arm.from_file(label, path, production=args.production,
                                  sample=args.sample))
    if not arms:
        print("Give at least one --arm LABEL=PATH")
        return 1
    truth = Truth.from_file(args.truth, subevents=args.truth_subevents)
    pid_samples = pid_truths = None
    if args.pid_arm:
        # The other particle's sample, paired with the main arms by LABEL.
        if not args.pid_truth:
            print("--pid-arm needs --pid-truth (the other sample's truth)")
            return 1
        mine = args.particle or args.hypothesis
        theirs = "mu-" if mine == "e-" else "e-"
        partner = {}
        for entry in args.pid_arm:
            label, path = entry.split("=", 1)
            partner[label] = Arm.from_file(label, path, production=args.production,
                                           sample=args.pid_sample)
        paired = [a for a in arms if a.label in partner]
        if paired:
            pid_samples = {mine: paired, theirs: [partner[a.label] for a in paired]}
            pid_truths = {mine: truth, theirs: Truth.from_file(args.pid_truth)}
    path = analysis.standard_report(
        arms, truth, args.outdir, hypothesis=args.hypothesis,
        reference=args.reference or arms[0].label,
        title=args.title or "Reconstruction comparison",
        pid_samples=pid_samples, pid_truths=pid_truths,
        subevent_window_ns=args.subevent_window,
        crb_model=_crb_model(args, _workspace(args)))
    print(f"\nReport: {path}")
    return 0


#: Where `sextant crb build` writes the model by default, relative to the workspace.
DEFAULT_CRB_MODEL = Path("Output") / "Sextant" / "crb_model" / "crb_model.npz"


def _crb_model(args, workspace):
    """The Cramér–Rao model to use: --crb-model, else the workspace default if it
    has been built, else none (`--crb-model none` switches it off)."""
    choice = getattr(args, "crb_model", None)
    if choice and choice.lower() == "none":
        return None
    path = Path(choice) if choice else Path(workspace) / DEFAULT_CRB_MODEL
    if not path.is_absolute():
        path = Path(workspace) / path
    if path.exists():
        return str(path)
    if choice:
        print(f"[WARNING] Cramér–Rao model {path} not found; bound skipped. "
              "Build it with `sextant crb build`.")
    return None


def cmd_crb(args) -> int:
    import numpy as np
    from .crb import CRBModel, bound
    workspace = _workspace(args)
    if args.crb_command == "build":
        from .crb.build import build
        out = Path(args.out or workspace / DEFAULT_CRB_MODEL)
        build(args.sample, args.tunings, str(out), max_files=args.max_files)
        return 0
    if args.crb_command == "floors":
        from .crb.floor import add_to_model
        out = add_to_model(args.model or workspace / DEFAULT_CRB_MODEL, args.sample, max_files=args.max_files)
        for particle, (_, fl) in out.items():
            print(particle)
            for i, lp in enumerate(fl["logp"]):
                print(f"  p {np.exp(lp):6.0f}  angle {np.degrees(fl['angle'][i]):5.2f} deg  along {fl['along'][i]:5.1f} cm"
                      f"  perp {fl['perp'][i]:4.1f} cm  yield {100 * fl['yld'][i]:4.1f} %  ({fl['n'][i]:.0f} events)")
        return 0
    model = CRBModel.load(args.model or workspace / DEFAULT_CRB_MODEL)
    if args.crb_command == "event":
        v = np.array([float(x) for x in args.vertex.split(",")])
        d = np.array([float(x) for x in args.direction.split(",")])
        rows = []
        for timing in ("measured", "ideal"):
            b = bound(model, args.particle, v, d, args.p, timing=timing, floor=not args.no_floor)
            rows.append((timing, b))
        print(f"{args.particle}  p = {args.p:g} MeV/c  vertex = {v}  dir = {d / np.linalg.norm(d)}")
        print(f"  detected light {rows[0][1]['npe']:.0f} pe on {rows[0][1]['nhit']:.0f} "
              f"expected hit PMTs; shower light fluctuation {100 * rows[0][1]['shower_sigma']:.2f} %")
        if rows[0][1]["floor"]:
            f = rows[0][1]["floor"]
            print(f"  physics floors: angle {np.degrees(f['angle']):.2f} deg/tilt, along {f['along']:.1f} cm, "
                  f"perp {f['perp']:.1f} cm, yield {100 * f['yld']:.1f} %")
        print(f"  {'':22s} {'measured timing':>16s} {'ideal timing':>14s}")
        for key, label, scale, unit in (("sigma_t", "vertex time", 1, "ns"),
                                        ("sigma_long", "vertex along track", 1, "cm"),
                                        ("sigma_perp", "vertex transverse/axis", 1, "cm"),
                                        ("sigma_angle", "direction", 1e3, "mrad"),
                                        ("sigma_p_rel", "momentum", 100, "%")):
            print(f"  {label:22s} {rows[0][1][key] * scale:13.3f} {unit:4s}"
                  f"{rows[1][1][key] * scale:10.3f} {unit}")
        return 0
    if args.crb_command == "scan":
        print(f"{'part':>4s} {'p':>6s} {'dwall':>6s} {'Npe':>7s} {'t[ns]':>7s} {'long[cm]':>9s} "
              f"{'perp[cm]':>9s} {'dir[mrad]':>10s} {'dp/p[%]':>8s}")
        for particle in ("e-", "mu-"):
            for p in (200, 400, 700, 1000, 1400):
                if p < np.exp(model.light[particle].logp_ctr[0]):
                    continue
                for dw in (100, 300, 700, 1500):
                    # vertex dw from the barrel, pointing out at it (towall = dwall)
                    v = np.array([model.det.radius - dw, 0.0, 0.0])
                    b = bound(model, particle, v, np.array([1.0, 0.0, 0.0]), float(p))
                    print(f"{particle:>4s} {p:6d} {dw:6d} {b['npe']:7.0f} {b['sigma_t']:7.3f} "
                          f"{b['sigma_long']:9.2f} {b['sigma_perp']:9.2f} "
                          f"{1e3 * b['sigma_angle']:10.2f} {100 * b['sigma_p_rel']:8.2f}")
        return 0
    return 1


def cmd_truth(args) -> int:
    from .io import truth as truth_io, truth_builder
    from . import productions

    inputs = productions.find_inputs(args.production, args.sample,
                                     limit=args.max_files, start=args.start)
    if not inputs:
        print(f"No input files found for {args.production}/{args.sample}. "
              f"This must run where the production is mounted.")
        return 1
    print(f"Building truth from {len(inputs)} file(s)...")

    def progress(path, n, total):
        print(f"  {Path(path).name}: {n} events ({total} total)", flush=True)

    rings, events, subevents = truth_builder.build_from_files(
        inputs, production=args.production, sample=args.sample,
        limit_events=args.events, progress=progress if args.verbose else None)
    decays = subevents[~subevents["is_prompt"]]
    prompts = subevents[subevents["is_prompt"]]
    summary = (f"{int(decays['recorded'].sum())} of {len(decays)} decay electrons "
               f"and {int(prompts['recorded'].sum())} of {len(prompts)} prompts "
               f"inside a trigger window")
    if args.subevents_only:
        # A sidecar for a truth file that already exists (see
        # truth.subevents_sidecar): the rings and events are not rewritten.
        truth_io.write(args.output, None, None, subevents)
        print(f"\nWrote {args.output}: sub-events of {len(events)} events; {summary}")
        return 0
    truth_io.write(args.output, rings, events, subevents)
    print(f"\nWrote {args.output}: {len(events)} events, {len(rings)} true rings "
          f"({int(events['n_visible_rings'].gt(0).sum())} with a visible ring); "
          f"{summary}")
    return 0


def cmd_productions(args) -> int:
    import pandas as pd
    from . import productions

    print(pd.DataFrame(productions.describe()).to_string(index=False))
    return 0


def cmd_schema(args) -> int:
    from .io import schema

    for name in ([args.table] if args.table else schema.SCHEMAS):
        print(f"\n=== {name} ===")
        print(schema.describe(name).to_string(index=False))
    return 0


def cmd_mapping(args) -> int:
    import pandas as pd
    from .config import mapping

    print(pd.DataFrame(mapping.describe_table()).to_string(index=False))
    print("\nCalibration schemes:")
    for name, scheme in mapping.SCHEMES.items():
        print(f"\n  {name}: {scheme.description}")
        for key, value in {**scheme.coral, **scheme.fitqun}.items():
            print(f"      {key} = {value}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sextant",
        description="Validation and comparison of water-Cherenkov "
                    "reconstruction (Coral, fiTQun).")
    parser.add_argument("--workspace", help="reconstruction workspace root "
                                            "(default: found from the cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_spec(name, help_text, function):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("spec", help="experiment spec (.ini)")
        p.set_defaults(func=function)
        return p

    with_spec("check", "validate a spec and compare the arms' configurations",
              cmd_check)

    p = with_spec("run", "launch every runnable arm", cmd_run)
    p.add_argument("--arm", action="append", help="launch only these arms")
    p.add_argument("--sync", action="store_true",
                   help="push and check out the code on the server first")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-check", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="launch even if the configuration check fails")

    p = with_spec("status", "progress of this experiment's runs", cmd_status)
    p.add_argument("--arm")
    p.add_argument("--full", action="store_true",
                   help="the whole cluster dashboard from exp.py")

    p = with_spec("logs", "stream one arm's job output", cmd_logs)
    p.add_argument("--arm", required=True)
    p.add_argument("--lines", type=int, default=40)

    with_spec("errors", "filtered failure summary across the arms", cmd_errors)

    p = with_spec("cancel", "cancel this experiment's runs (only these)",
                  cmd_cancel)
    p.add_argument("--arm")

    with_spec("collect", "merge and pull finished arms", cmd_collect)

    p = with_spec("report", "build the comparison report", cmd_report)
    p.add_argument("--outdir")
    p.add_argument("--collect", action="store_true",
                   help="merge and pull first")
    p.add_argument("--allow-missing", action="store_true",
                   help="build the report even if an arm could not be loaded")
    p.add_argument("--crb-model", help="Cramér–Rao model .npz (default: the "
                   "workspace's Output/Sextant/crb_model/crb_model.npz if built; "
                   "'none' to skip)")

    p = with_spec("probe", "per-PMT intermediates at a common parameter point",
                  cmd_probe)
    p.add_argument("--config", required=True, help="Coral config.ini to probe with")
    p.add_argument("--coral-root", help="Coral checkout (the dir holding src/)")
    p.add_argument("--fitqun-params",
                   help="fiTQun parameter file; probes the fiTQun side at the "
                        "SAME points and writes the per-PMT comparison")
    p.add_argument("--fitqun-root", help="fiTQun install fqprobe reads its "
                   "tuning constants from (default: $FITQUN_ROOT)")
    p.add_argument("--fqprobe", help="path to the fqprobe binary "
                                     "(default: $SEXTANT_FQPROBE or on PATH)")
    p.add_argument("--events", type=int, default=20)
    p.add_argument("--events-in", metavar="PATH",
                   help="restrict to events present in this merged reco file "
                        "or diagnostic CSV -- use it whenever the probe will "
                        "be compared against a run's own fitted results")
    p.add_argument("--time-pedestal", default="auto", metavar="auto|none|NS",
                   help="offset added to the truth vertex time so the truth "
                        "point sits on the algorithm's time origin. 'auto' "
                        "measures it from --events-in (default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir")

    p = sub.add_parser("compare",
                       help="report from files named on the command line")
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--truth", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--hypothesis", default="e-")
    p.add_argument("--reference")
    p.add_argument("--title")
    p.add_argument("--production", help="needed only for official fiTQun files")
    p.add_argument("--sample")
    p.add_argument("--truth-subevents", help="sub-event truth file (default: the "
                   "truth file, then its <stem>_subevents.root sidecar)")
    p.add_argument("--particle", help="true particle of the main sample, e-/mu- "
                   "(default: --hypothesis); needed with --pid-arm")
    p.add_argument("--pid-arm", action="append", metavar="LABEL=PATH",
                   help="the SAME arm run on the other particle's sample; "
                        "pairs with --arm by label and enables the PID section")
    p.add_argument("--pid-truth", help="truth file of the --pid-arm sample")
    p.add_argument("--pid-sample", help="sample of the --pid-arm files (official "
                   "fiTQun only)")
    p.add_argument("--subevent-window", type=float, default=50.0, metavar="NS",
                   help="time window for matching sub-events (default 50 ns)")
    p.add_argument("--crb-model", help="Cramér–Rao model .npz (default: the "
                   "workspace's Output/Sextant/crb_model/crb_model.npz if built; "
                   "'none' to skip)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("truth", help="build a truth file from WCSim input")
    p.add_argument("--production", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--events", type=int)
    p.add_argument("--max-files", type=int)
    p.add_argument("--start", type=int, default=0,
                   help="first input FILE index (the held-out validation files "
                        "start at 80)")
    p.add_argument("--subevents-only", action="store_true",
                   help="write only the sub-event table, as the sidecar of an "
                        "existing truth file: name the output "
                        "<truth stem>_subevents.root")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_truth)

    p = sub.add_parser("crb", help="Cramér–Rao lower bounds on the fitted parameters")
    crb_sub = p.add_subparsers(dest="crb_command", required=True)
    b = crb_sub.add_parser("build", help="measure the light and timing model from a "
                           "WCSim particle-gun sample (needs h5py, scikit-learn)")
    b.add_argument("--sample", required=True,
                   help="directory holding electrons/*.h5 and muons/*.h5")
    b.add_argument("--tunings", required=True,
                   help="directory holding CProf_{11,13}_fit_WCSim.root")
    b.add_argument("--out", help="default: Output/Sextant/crb_model/crb_model.npz")
    b.add_argument("--max-files", type=int)
    e = crb_sub.add_parser("event", help="the bound for one configuration")
    e.add_argument("--particle", default="e-", choices=["e-", "mu-"])
    e.add_argument("-p", type=float, default=500.0, help="momentum, MeV/c")
    e.add_argument("--vertex", default="0,0,0", help="cm, x,y,z")
    e.add_argument("--direction", default="1,0,0")
    e.add_argument("--model")
    e.add_argument("--no-floor", action="store_true", help="the pure Fisher bound, without the physics floors")
    fl = crb_sub.add_parser("floors", help="measure the physics floors (light-axis, emission-centroid and "
                            "muon-range fluctuations) from WCSim photon truth and add them to the model")
    fl.add_argument("--sample", required=True, help="directory holding electrons/*.h5 and muons/*.h5")
    fl.add_argument("--model")
    fl.add_argument("--max-files", type=int)
    sc = crb_sub.add_parser("scan", help="bound against momentum and dwall")
    sc.add_argument("--model")
    p.set_defaults(func=cmd_crb)

    sub.add_parser("productions", help="samples this package knows about"
                   ).set_defaults(func=cmd_productions)

    p = sub.add_parser("schema", help="the canonical tables")
    p.add_argument("table", nargs="?")
    p.set_defaults(func=cmd_schema)

    sub.add_parser("mapping", help="Coral <-> fiTQun parameter equivalence"
                   ).set_defaults(func=cmd_mapping)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
