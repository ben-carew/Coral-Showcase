"""Launching the arms of an experiment.

Sextant does not submit jobs itself. It writes one `exp.py` spec per arm and
shells out to `exp.py`, which is the workspace's only sanctioned entry point for
starting a run -- it freezes the code, the config and the parameter file into
the run directory, writes the manifest that `status`/`merge`/`pull`/`cancel`
read, and makes the run immutable. Reimplementing any of that here would
recreate exactly the class of bug that machinery exists to prevent (a run
nothing can see, or a config edit that changes a job already queued).

The specs Sextant generates are written into the experiment's own directory and
never shared between arms, for the same reason: two arms must not be able to
edit each other's inputs.
"""

from __future__ import annotations

import configparser
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import mapping
from .spec import SOURCE_FILE, SOURCE_OFFICIAL, SOURCE_RUN, ArmSpec, ExperimentSpec


def find_exp_py(start: Path | None = None) -> Path:
    """Locate `Analysis/Scripts/exp.py` by walking up from a starting point."""
    candidates = []
    here = Path(start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidates.append(directory / "Analysis/Scripts/exp.py")
    # Also relative to this package, for an installed copy sitting inside the
    # workspace (Software/Sextant/src/sextant/run/launcher.py -> workspace root).
    package_root = Path(__file__).resolve().parents[4]
    candidates.append(package_root / "Analysis/Scripts/exp.py")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not find Analysis/Scripts/exp.py. Run sextant from inside the "
        "reconstruction workspace, or pass --exp-py explicitly.")


@dataclass
class ArmRun:
    """The identity of one launched arm."""
    arm: str
    algorithm: str
    run_id: str                 # 'coral:<experiment>/<arm>'
    spec_path: str
    source: str = SOURCE_RUN

    def as_dict(self) -> dict:
        return {"arm": self.arm, "algorithm": self.algorithm,
                "run_id": self.run_id, "spec_path": self.spec_path,
                "source": self.source}


def workdir_for(spec: ExperimentSpec, workspace: Path) -> Path:
    """Everything one experiment produces: specs, launch state, probe output
    and the report.

    `Output/Sextant/<experiment>/`, alongside the algorithms' own Output
    directories. Analysis/Reports/ is deliberately not used -- that is for
    bespoke, authored write-ups about Coral development, not for generated
    validation output.
    """
    return Path(workspace) / "Output" / "Sextant" / spec.name


def write_arm_spec(spec: ExperimentSpec, arm: ArmSpec, workspace: Path) -> Path:
    """Write the `exp.py` spec for one arm into the experiment's own directory."""
    from .. import productions

    parser = configparser.ConfigParser()
    parser.optionxform = str

    prod = productions.get(spec.production)
    sample = prod.sample(spec.sample)
    # exp.py identifies a sample by its registry key -- 'electron', 'muon',
    # 'pi0' for the guns and the neutrino channel name for the beam -- NOT by
    # the ring type ('e-') that Sample.particle carries.
    particle = spec.sample

    # fiTQun's configuration is a parameter-override FILE, so the arm's scheme,
    # [shared] keys and param. overrides have to be baked into one before the
    # spec can point at it.
    directory = workdir_for(spec, workspace) / "specs"
    fitqun_params = (_fitqun_params(spec, arm, workspace, directory)
                     if arm.algorithm == "fitqun" else arm.params)

    parser["experiment"] = {
        "name": spec.name,
        "variant": arm.name,
        "algo": arm.algorithm,
        "where": spec.where,
        # The production is passed through as itself. An earlier version mapped
        # anything unrecognised onto v1.12.22 and leaned on an input-directory
        # override, which quietly gave beam runs the particle guns' event-id
        # stride of 100 -- and the beam files hold 101 events, so every file's
        # last event would have aliased onto the next file's first.
        "particle": particle,
        "events": str(arm.events or spec.events),
        "production": spec.production,
        "slot": arm.slot,
        "params": fitqun_params,
        "multiring": "true" if _wants_multiring(spec, arm) else "false",
    }
    if spec.input_dir:
        parser["experiment"]["input_dir"] = spec.input_dir
    if arm.algorithm == "coral" and arm.config_file:
        parser["experiment"]["coral_config"] = arm.config_file
    # `start` must reach exp.py or a held-out validation run silently re-reads
    # the calibration's own files. It was dropped here until 2026-09-16.
    parser["run"] = {"events_per_job": str(arm.events_per_job
                                           or spec.events_per_job),
                     "start": str(arm.start if arm.start is not None
                                  else spec.start)}
    if spec.s_per_event:
        parser["run"]["s_per_event"] = spec.s_per_event
    parser["local"] = {"workers": str(spec.workers), "threads": str(spec.threads),
                       "fetch": "true"}
    parser["server"] = {"time": spec.time, "straggler_cut": spec.straggler_cut,
                        "submit_delay": spec.submit_delay, "mem": spec.mem}
    # per-arm overrides (e.g. partition / gpus / env for a GPU arm)
    parser["server"].update(getattr(arm, "server", {}) or {})

    if arm.algorithm == "coral":
        _check_scheme_against_config(arm, workspace)
        parser["coral_config"] = _coral_config(spec, arm)
        _report_changes(arm, workspace, parser["coral_config"])

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm.name}.ini"
    with path.open("w") as fh:
        fh.write(f"; Generated by Sextant from {spec.source_path}\n")
        fh.write(f"; Arm '{arm.name}' of experiment '{spec.name}'.\n")
        if arm.scheme:
            scheme = mapping.SCHEMES.get(arm.scheme)
            fh.write(f"; Calibration scheme: {arm.scheme}"
                     + (f" — {scheme.description}\n" if scheme else "\n"))
        fh.write("; Do not edit: regenerated on every `sextant run`.\n\n")
        parser.write(fh)
    return path


def coral_base_config(arm: ArmSpec, workspace: Path) -> Path:
    """The committed config this Coral arm starts from, in its slot's checkout."""
    root = (Path(workspace) / "Software/Coral" if arm.slot == "main"
            else Path(workspace) / "Software/worktrees" / arm.slot)
    return root / (arm.config_file or "config.ini")


def _check_scheme_against_config(arm: ArmSpec, workspace: Path):
    """Stop a scheme from silently rewriting the calibration its config carries.

    A scheme's constants are written OVER the arm's config. That is the point of
    a scan, and a trap everywhere else: from 2026-10-21 to 10-27 every multi-ring
    run named `coral-hitscat-2026-10` with `configs/multiring.ini`, so the
    direct-only preset ran with hit-PMT scattered light and another photon
    yield, against tables derived without them -- the second time (10-05 was the
    first). A branch that re-derives its yield hits the same thing with
    config.ini. An arm whose scheme changes a constant its config sets must say
    so (`scheme_override = true`) or take the config as committed
    (`scheme = config`).
    """
    if not arm.scheme or arm.scheme == mapping.CONFIG_SCHEME:
        return
    base = coral_base_config(arm, workspace)
    if not base.exists():
        return                                # slot not checked out here: nothing to compare
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
    cfg.optionxform = str
    cfg.read(base)
    values = {f"{section}.{key}": value.strip()
              for section in cfg.sections() for key, value in cfg.items(section)}
    conflicts = mapping.scheme_conflicts(arm.scheme, values)
    if not conflicts:
        return
    listing = "\n".join(f"      {key}: {base.name} = {have}, scheme = {want}"
                        for key, have, want in conflicts)
    if arm.scheme_override:
        print(f"[SCHEME] arm '{arm.name}': scheme '{arm.scheme}' overrides "
              f"{base.name} (scheme_override = true):\n{listing}")
        return
    raise SystemExit(
        f"arm '{arm.name}': scheme '{arm.scheme}' would CHANGE calibration "
        f"constants that {base} sets:\n{listing}\n"
        f"    The tables in that config were derived for its own constants.\n"
        f"    - to run the config as committed:  scheme = config\n"
        f"    - if the change is the experiment: scheme_override = true")


def _base_values(arm: ArmSpec, workspace: Path):
    base = coral_base_config(arm, workspace)
    if not base.exists():
        return base, None
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False)
    cfg.optionxform = str
    cfg.read(base)
    return base, {f"{section}.{key}": value.strip()
                  for section in cfg.sections() for key, value in cfg.items(section)}


def _report_changes(arm: ArmSpec, workspace: Path, overrides) -> list:
    """Print what this arm changes in the config it starts from, key by key.

    Three layers write over a Coral config (the scheme, `[shared]` concepts,
    `config.` keys) and each has silently changed a preset at least once: the
    scheme (multi-ring presets, 10-05 and 10-21), `[shared] do_muon_fit`
    rewriting `fits.particles` over a preset with a pi+ hypothesis (10-20).
    The frozen config can be diffed after the fact; this says it at launch.
    """
    base, values = _base_values(arm, workspace)
    if values is None:
        return []
    changes = []
    for key, new in dict(overrides).items():
        old = values.get(key)
        if old is None:
            changes.append((key, "(not set)", str(new)))
        elif not mapping._equal(old, new) and [x.strip() for x in old.split(",")] != [
                x.strip() for x in str(new).split(",")]:
            changes.append((key, old, str(new)))
    if changes:
        print(f"[CONFIG] arm '{arm.name}' changes {len(changes)} key(s) of {base.name}:")
        for key, old, new in changes:
            print(f"      {key}: {old} -> {new}")
    else:
        print(f"[CONFIG] arm '{arm.name}' runs {base.name} as committed")
    return changes


#: Concepts that select a fit hypothesis. Coral names all of them in one
#: comma-separated key, so they combine into a list rather than each becoming a
#: boolean of its own -- writing `fits.particles = true` would be silently
#: accepted by configparser and leave Coral fitting nothing. One definition,
#: shared with ExperimentSpec.resolved_settings, which has the same problem.
from ..config.mapping import HYPOTHESIS_LABEL as HYPOTHESIS_CONCEPTS
from .spec import _is_on


_FQ_LINE = re.compile(
    r"^(?P<pre>\s*<\s*)(?P<key>[A-Za-z_][\w.]*)(?P<mid>\s*=\s*)"
    r"(?P<val>[^>]*?)(?P<post>\s*>)(?P<rest>.*)$")


def _fitqun_base_params(arm: ArmSpec, workspace: Path) -> Path | None:
    """The override file this arm starts from, or None to start from nothing.

    An explicit `params =` on the arm wins, because it is the more specific
    statement; otherwise the scheme's own `fitqun_override_file` is used, which
    is what makes declaring a scheme actually select a tuning rather than only
    describe one.
    """
    if arm.params:
        path = Path(arm.params)
        return path if path.is_absolute() else Path(workspace) / path
    if arm.scheme and arm.scheme in mapping.SCHEMES:
        name = mapping.SCHEMES[arm.scheme].fitqun_override_file
        if name:
            path = Path(name)
            if path.is_absolute():
                return path
            if path.parent != Path("."):
                return Path(workspace) / path
            return (Path(workspace) / "Software/fiTQun/ParameterOverrideFiles"
                    / path)
    return None


def _apply_fitqun_overrides(text: str, overrides: dict) -> str:
    """Set `overrides` in a fiTQun `.parameters.dat`, preserving its comments.

    Keys already present are rewritten in place -- EVERY occurrence, because
    these files repeat keys (the HK-2024 overrides set DoProton1RFit twice) and
    fiTQun takes the last one, so rewriting only the first would leave a stale
    value winning. Keys not present are appended, which is the case that
    matters most: DoSubEvent lives only in the base fiTQun.parameters.dat, so
    an override file that never mentions it silently inherits the
    decay-electron search.
    """
    applied, changed, out = set(), {}, []
    for line in text.splitlines():
        match = _FQ_LINE.match(line)
        if match and match.group("key") in overrides:
            key = match.group("key")
            value = mapping.render_fitqun_value(key, overrides[key])
            was = match.group("val").strip()
            if was != value:
                changed[key] = (was, value)
            out.append(f"{match.group('pre')}{key}{match.group('mid')}{value}"
                       f"{match.group('post')}{match.group('rest')}")
            applied.add(key)
        else:
            out.append(line)
    missing = [k for k in overrides if k not in applied]
    if missing:
        out += ["", "# --- set by Sextant from the arm's scheme, [shared] keys "
                    "and param. overrides ---"]
        for k in sorted(missing):
            value = mapping.render_fitqun_value(k, overrides[k])
            out.append(f" < {k} = {value} >")
            changed[k] = ("(absent)", value)
    return "\n".join(out) + "\n", changed


def _fitqun_params(spec: ExperimentSpec, arm: ArmSpec, workspace: Path,
                   directory: Path) -> str:
    """Write this arm's fiTQun override file; return the path for `params`.

    Before this existed, a fiTQun arm shipped `arm.params` verbatim and nothing
    else: the scheme's `fitqun_override_file` was set on every scheme and read
    by nothing, `[shared]` keys were translated only for the comparison table,
    and `param.fiTQun.*` overrides were parsed into `native_params` and then
    dropped. Every fiTQun arm therefore ran whatever its `params` file said,
    with the rest of its declared configuration silently inert.
    """
    overrides = {k: v for k, v in spec.resolved_settings(arm).items()
                 if "." in k and k.split(".")[0] in ("fiTQun", "spliTChan")}
    base = _fitqun_base_params(arm, workspace)
    if base is not None and not base.exists():
        raise SystemExit(f"arm '{arm.name}': parameter file not found: {base}")
    if not overrides:
        # Nothing to layer on: ship the file itself, so the run's frozen record
        # points at the tuning by its own name.
        return arm.params or (str(base.relative_to(Path(workspace)))
                              if base is not None else "")
    text = base.read_text() if base is not None else ""
    body, changed = _apply_fitqun_overrides(text, overrides)
    lines = [f"# Generated by Sextant for arm '{arm.name}' of '{spec.name}'.",
             f"# Base: {base if base is not None else '(none)'}",
             f"# Scheme: {arm.scheme or '(none)'}"]
    # An explicit `params =` file is the author's own statement of a tuning, so
    # anything the scheme or a [shared] key changes IN it is recorded here
    # rather than applied silently.
    if changed:
        lines.append("# Changed from the base file:")
        lines += [f"#   {k}: {was} -> {now}"
                  for k, (was, now) in sorted(changed.items())]
    else:
        lines.append("# Nothing differed from the base file.")
    lines.append("# Do not edit: regenerated on every `sextant run`.")
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{arm.name}.parameters.dat"
    out.write_text("\n".join(lines) + "\n" + body)
    try:
        return str(out.relative_to(Path(workspace)))
    except ValueError:
        return str(out)


def _coral_config(spec: ExperimentSpec, arm: ArmSpec) -> dict:
    """Coral's `[coral_config]` block for this arm, with list-valued keys built.

    `spec.resolved_settings` maps each concept to a single key, which is right
    for every setting except the fit-hypothesis list: several concepts share
    `fits.particles`, and the last one written would otherwise win.
    """
    settings = spec.resolved_settings(arm)
    out = {k: v for k, v in settings.items()
           if "." in k and not k.startswith("fiTQun") and k != "fits.particles"}

    hypotheses = []
    for concept, label in HYPOTHESIS_CONCEPTS.items():
        value = spec.shared.get(concept)
        if value is not None and _is_on(value):
            hypotheses.append(label)
    # An explicit native override always wins over the assembled list.
    native = arm.native_config.get("fits.particles")
    if native:
        out["fits.particles"] = native
        hypotheses = [h.strip() for h in native.split(",") if h.strip()]
    elif hypotheses:
        out["fits.particles"] = ", ".join(hypotheses)

    # Only when multi-ring is actually in play: not every Coral build has the
    # fitter (upstream dev does not), and writing a [multiring] section into a
    # config that has no use for one is noise in the frozen record.
    if hypotheses and _wants_multiring(spec, arm):
        # Coral's multi-ring fitter validates that every candidate ring type has
        # a tuning in [fits] particles, and refuses to start if one does not.
        # Narrowing the hypothesis list therefore has to narrow the candidate
        # types with it -- otherwise Sextant writes a config that Coral rejects
        # at load ("multi-ring candidate types ['mu-'] need a tuning in [fits]
        # particles"), which is a failure of exactly the job this generator has.
        candidates = arm.native_config.get("multiring.candidate_types")
        if candidates:
            kept = [c.strip() for c in candidates.split(",")
                    if c.strip() in hypotheses]
            out["multiring.candidate_types"] = ", ".join(kept or hypotheses)
        else:
            out["multiring.candidate_types"] = ", ".join(hypotheses)
    return out


def _wants_multiring(spec: ExperimentSpec, arm: ArmSpec) -> bool:
    settings = spec.resolved_settings(arm)
    for key in ("multiring.do_multiring", "fiTQun.DoMRFit"):
        value = settings.get(key)
        if value is not None and str(value).strip().lower() not in ("0", "false",
                                                                    "no", "off"):
            return True
    return str(spec.shared.get("do_multiring_fit", "")).lower() in ("1", "true",
                                                                    "yes")


def launch(spec: ExperimentSpec, workspace: Path, *, arms=None, sync=False,
           dry_run=False, exp_py=None) -> list:
    """Submit every runnable arm through exp.py. Returns the ArmRun records."""
    exp_py = Path(exp_py) if exp_py else find_exp_py(workspace)
    selected = [a for a in spec.arms
                if arms is None or a.name in arms or a.label in arms]
    launched = []

    for arm in selected:
        if arm.source != SOURCE_RUN:
            launched.append(ArmRun(arm.name, arm.algorithm,
                                   run_id="", spec_path="", source=arm.source))
            continue
        arm_spec_path = write_arm_spec(spec, arm, workspace)
        command = [sys.executable, str(exp_py), "submit", "-f", str(arm_spec_path)]
        if sync:
            command.append("--sync")
        print("+", " ".join(shlex.quote(c) for c in command))
        if not dry_run:
            result = subprocess.run(command, cwd=str(workspace))
            if result.returncode != 0:
                raise SystemExit(
                    f"arm '{arm.name}': exp.py submit failed with exit code "
                    f"{result.returncode}. The other arms were not started; "
                    f"fix this one and re-run (already-submitted arms are "
                    f"skipped by name).")
        launched.append(ArmRun(
            arm=arm.name, algorithm=arm.algorithm,
            run_id=f"{arm.algorithm}:{spec.name}/{arm.name}",
            spec_path=str(arm_spec_path), source=arm.source))
    return launched


def exp_command(spec: ExperimentSpec, workspace: Path, command: str,
                run_id: str = "", *, extra=(), exp_py=None, capture=False):
    """Run any exp.py subcommand for this experiment (status/logs/merge/...)."""
    exp_py = Path(exp_py) if exp_py else find_exp_py(workspace)
    argv = [sys.executable, str(exp_py), command]
    if run_id:
        argv.append(run_id)
    argv.extend(extra)
    if capture:
        return subprocess.run(argv, cwd=str(workspace), capture_output=True,
                              text=True)
    return subprocess.run(argv, cwd=str(workspace))


def run_ids(spec: ExperimentSpec) -> list:
    """The exp.py run ids this experiment owns, for status/merge/cancel."""
    return [f"{arm.algorithm}:{spec.name}/{arm.name}"
            for arm in spec.arms if arm.source == SOURCE_RUN]
