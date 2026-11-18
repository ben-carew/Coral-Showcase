"""The standard comparison: arms in, report directory out.

This is the whole validation half behind one call. It takes arms and truth --
however they were obtained, whether from a Sextant experiment or from files
named by hand -- and produces the report directory.

    from sextant import Arm, Truth, analysis

    arms = [Arm.from_file("Coral sandbox", "…/coral_electron.root"),
            Arm.from_file("fiTQun",        "…/fitqun_electron.root")]
    truth = Truth.from_file("…/truth_electron.root")
    analysis.standard_report(arms, truth, outdir="…/Reports/e-comparison")

Nothing here imports `sextant.run`. Sections are skipped, with a note in the
report, when the inputs cannot support them -- a Coral-only comparison has no
per-PMT section, a single-particle sample has no PID section -- so the same call
works for every combination rather than needing a different script per study.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from .io.schema import HYP_ELECTRON, HYP_MUON
from .metrics import (intermediates as intermediate_metrics,
                      multitrack as multitrack_metrics, pid as pid_metrics,
                      resolution as resolution_metrics,
                      residuals as residual_metrics, rings as ring_metrics,
                      subevents as subevent_metrics, validate)
from .plots import (intermediate_plots, pid_plots, rate_plots, residual_plots,
                    ring_plots, style as st)

#: Every kinematic quantity is shown against these three (HK validation spec).
PROFILE_VARIABLES = ("p_true", "dwall", "towall")
from .report.builder import Report
from .report import tables as table_io


def compute_residuals(arms, truth, *, hypothesis=HYP_ELECTRON,
                      only_success=True, sub_event=0) -> dict:
    """{arm label: residual table} for one fit hypothesis."""
    out = {}
    for arm in arms:
        fits = arm.hypothesis(hypothesis, only_success=only_success,
                              sub_event=sub_event)
        out[arm.label] = residual_metrics.compute(
            fits, truth.primary, truth_events=truth.events)
    return out


def standard_report(arms, truth, outdir, *, title=None, subtitle="",
                    hypothesis=HYP_ELECTRON, reference=None,
                    pid_samples=None, pid_truths=None, backend="Agg", seed=0,
                    align_time_pedestal=None, crb_model=None,
                    crb_timing="measured",
                    subevent_window_ns=subevent_metrics.DEFAULT_WINDOW_NS) -> Path:
    """Build the full comparison report.

    `pid_samples` enables the PID section, which needs *two* samples of known
    true particle: {'e-': [arms on an electron sample],
                    'mu-': [arms on a muon sample]}, keyed by arm label within
    each. Without it PID is skipped, because a discriminant measured on one
    particle alone has no separation to report. `pid_truths` ({'e-': Truth,
    'mu-': Truth}) adds the identification efficiency against momentum, dwall
    and towall.

    Sub-eventing efficiency and purity are reported when the truth carries
    sub-event truth (`Truth.subevents`), and the multi-track section when the
    sample has enough events with two or more visible tracks.

    `crb_model` (a path or a loaded `crb.CRBModel`) adds the Cramér–Rao bound:
    shaded under every width panel, a table of arm/bound ratios, and figures of
    that ratio against momentum and dwall. Skipped, with the reason in the report,
    for samples a single-ring bound does not describe.
    """
    matplotlib.use(backend, force=False)
    st.use_style()

    arms = list(arms)
    labels = [arm.label for arm in arms]
    reference = reference or labels[0]
    arm_style = st.ArmStyle(labels, reference=reference)

    report = Report(
        title=title or f"Reconstruction comparison — {hypothesis} hypothesis",
        subtitle=subtitle or f"{len(arms)} runs · truth {Path(truth.source).name}",
        outdir=outdir)
    # Paths in the rendered tables are shown relative to this; the CSVs keep the
    # absolute value.
    report.workspace = _common_root(arms)
    report.meta["arms"] = labels
    report.meta["reference"] = reference
    report.meta["hypothesis"] = hypothesis

    # -- provenance and sanity ----------------------------------------------
    report.add_table("Runs compared", table_io.provenance_table(arms),
                     name="provenance",
                     note="What produced each arm. A result whose code SHA and "
                          "config are not recorded is not reproducible.")

    residuals = compute_residuals(arms, truth, hypothesis=hypothesis)
    findings = validate.run_all(arms, truth, residuals)
    report.add_table("Validation checks", table_io.findings_table(findings),
                     name="validation",
                     note="Checks that catch a comparison against the wrong "
                          "truth, duplicate keys, or a mismatched time "
                          "convention. Read these before the numbers.")

    # Against generator-clock truth every fit sits at -(trigger date); that is
    # trigger placement, not a bias, so it is removed unless the truth is on the
    # trigger clock (None = decide per arm from the residuals themselves).
    residuals = {label: (residual_metrics.subtract_pedestal(r, "t")
                         if (align_time_pedestal if align_time_pedestal is not None
                             else residual_metrics.needs_pedestal(r)) else r)
                 for label, r in residuals.items()}

    # -- residuals and resolution -------------------------------------------
    # Two pages, deliberately: the per-parameter residuals say *how* a fit is
    # wrong, the aggregates say how well it did. Mixing them on one page invites
    # reading a coordinate component as if it were a performance number.
    report.add_figure(
        "Residual distributions — fitted parameters",
        residual_plots.residual_histograms(
            residuals, arm_style,
            quantities=residual_metrics.COMPONENT_QUANTITIES, ncols=4,
            title="Per-parameter residuals (reconstructed − true)"),
        name="residuals_parameters",
        note="θ and φ are diagnostics, not performance numbers: φ wraps and is "
             "degenerate near the poles, so a large φ residual on a near-polar "
             "track can mean very little. The angular displacement on the next "
             "page is the rotationally well-behaved measure.")
    report.add_figure(
        "Residual distributions — aggregate performance",
        residual_plots.residual_histograms(
            residuals, arm_style,
            quantities=residual_metrics.AGGREGATE_QUANTITIES, ncols=3,
            title="Aggregate performance (reconstructed vs true)"),
        name="residuals_aggregate")

    summaries = {label: residual_metrics.summarise(r, label, seed=seed)
                 for label, r in residuals.items()}
    report.add_table("Headline summary", table_io.headline_table(summaries),
                     name="headline",
                     note="Median and central-68% half-width per arm. Medians, "
                          "not means: these distributions have heavy tails and "
                          "a handful of mis-converged events dominate a mean.")
    stacked = pd.concat([s for s in summaries.values() if not s.empty],
                        ignore_index=True) if summaries else pd.DataFrame()
    report.add_table("Full summary statistics", stacked, name="summary_full",
                     compact=True,
                     note="Bootstrap 68% confidence intervals in brackets (2000 "
                          "resamples, seeded, so a regenerated report gives the "
                          "same intervals). The CSV keeps every bound as its own "
                          "column.")

    bound = _cramer_rao(report, crb_model, residuals, truth, hypothesis,
                        timing=crb_timing, seed=seed)

    for versus in PROFILE_VARIABLES:
        if not any(versus in r.columns for r in residuals.values()):
            continue
        report.add_csv(f"resolution_{versus}",
                       resolution_metrics.headline_profile_table(residuals, versus))
        report.add_figure(
            f"Resolution against {versus}",
            residual_plots.resolution_profiles(residuals, arm_style,
                                               versus=versus, bound=bound),
            name=f"resolution_{versus}",
            note=("Grey: the Cramér–Rao bound, the best resolution any unbiased "
                  "reconstruction could reach in that bin. Not drawn on bias "
                  "panels (an efficient estimator is unbiased) or on time (see "
                  "the bound section)." if bound is not None else ""))
    if bound is not None:
        for versus in PROFILE_VARIABLES:
            report.add_figure(
                f"Distance from the Cramér–Rao bound against {versus}",
                residual_plots.bound_ratio_profiles(residuals, bound, arm_style,
                                                    versus=versus),
                name=f"crb_ratio_{versus}",
                note="Arm divided by bound, bin by bin, on a log scale. 1 is an "
                     "efficient reconstruction. The bound includes the measured "
                     "shower light-yield fluctuation for electrons but not multiple "
                     "scattering or shower-development fluctuations, so for "
                     "electron direction and along-track vertex the achievable "
                     "floor sits well above it.")

    if len(arms) > 1:
        report.add_figure(
            f"Paired differences against {reference}",
            residual_plots.paired_difference_plot(residuals, arm_style,
                                                  reference),
            name="paired_differences")
        report.add_table(
            f"Paired differences against {reference}",
            table_io.comparison_table(residuals, reference, seed=seed),
            name="paired_differences",
            note="Per-event differences on matched events, with a bootstrap CI. "
                 "'excludes_zero' is the verdict: where it is 'no', the two arms "
                 "are not distinguishable on this sample size.")

    # -- PID -----------------------------------------------------------------
    if pid_samples:
        _add_pid_section(report, pid_samples, arm_style, truths=pid_truths)

    # -- sub-eventing --------------------------------------------------------
    _add_subevent_section(report, arms, truth, arm_style,
                          window_ns=subevent_window_ns)

    # -- multi-track events ---------------------------------------------------
    _add_multitrack_section(report, arms, truth, arm_style, seed=seed)

    # -- multi-ring ----------------------------------------------------------
    _add_ring_section(report, arms, truth, arm_style)

    # -- intermediates -------------------------------------------------------
    _add_intermediate_section(report, arms, truth, arm_style, hypothesis)

    # -- timing --------------------------------------------------------------
    timing_figure = residual_plots.timing_histogram(arms, arm_style)
    if timing_figure is not None:
        report.add_figure("Reconstruction time", timing_figure, name="timing",
                          note="Never compare a laptop time against a cluster "
                               "time — see the 'where' column of the arms table.")
        from .io import timing as timing_io
        report.add_table("Timing summary", pd.DataFrame(
            [timing_io.summarise(arm.timing, arm.label) for arm in arms
             if not arm.timing.empty]), name="timing",
            note="Per EVENT. Both codes report their own reconstruction time "
                 "per job, which is divided by that job's event count here, so "
                 "neither number includes job startup and the two are "
                 "comparable. n_jobs is the number of timing rows; mean/median/"
                 "min/max are over per-job rates, so 'min_s' is the fastest "
                 "job's average event, not the fastest single event. "
                 "median_wall_s is blank when a run's timing CSV predates the "
                 "wall_s column; it is not filled from time_s because wall "
                 "time includes startup and reconstruction time does not.")

    return report.write()


def _cramer_rao(report, crb_model, residuals, truth, hypothesis, *, timing, seed):
    """Compute the bound's residual table and add its table to the report."""
    if crb_model is None:
        return None
    from .crb import CRBModel
    from .crb.compare import bound_arm

    model = CRBModel.load(crb_model) if not isinstance(crb_model, CRBModel) else crb_model
    bound, info = bound_arm(model, residuals, truth, hypothesis, timing=timing,
                            seed=seed)
    if bound is None:
        # An unavailable section is not produced (see test_the_report_carries_no_prose);
        # say why on the console instead.
        print(f"[Cramér–Rao bound skipped: {info}]")
        return None
    report.add_table(
        "Cramér–Rao bound", table_io.bound_table(residuals, bound, info), name="crb",
        note=(f"The best any unbiased reconstruction could do with this detector, "
              f"computed per event at the true parameters for {len(info)} events "
              f"({timing} hit-time information; the shower light-yield fluctuation "
              "is included for electrons) and summarised by the same statistics as "
              "the runs. Time is not compared: the fitters report t relative to the "
              "trigger, the truth starts every particle at t = 0, and the time "
              "residual therefore measures the ~30 ns trigger jitter, not the "
              "reconstruction. 'pull σ' divides each event's residual by its own "
              "bound: 1.00 is an efficient estimator. Model: "
              + str(model.provenance.get("sample", "?"))))
    # One row per event: the bound's 1-sigma widths at the true parameters
    # (sigma_angle in rad, sigma_p_rel a fraction). CSV only -- too long to render.
    report.add_csv("crb_events", info)
    return bound


def _add_pid_section(report, pid_samples, arm_style, truths=None):
    by_arm = {}
    for true_particle, arms_for_sample in pid_samples.items():
        for arm in arms_for_sample:
            by_arm.setdefault(arm.label, {})[true_particle] = \
                pid_metrics.discriminant(arm.fits)
    usable = {a: s for a, s in by_arm.items()
              if HYP_ELECTRON in s and HYP_MUON in s
              and len(s[HYP_ELECTRON]) and len(s[HYP_MUON])}
    if not usable:
        return
    # Shared events only, per true particle, like every paired comparison here.
    # An event one code has no output for is not a PID failure of the other: a
    # fiTQun merge keeps the events that fired no trigger as dummy fits with
    # dnLL exactly 0, and those alone moved its 99%-e working point to dnLL = 0.
    for particle in (HYP_ELECTRON, HYP_MUON):
        common = set.intersection(*(set(s[particle]["event"]) for s in usable.values()))
        for samples in usable.values():
            samples[particle] = samples[particle][
                samples[particle]["event"].isin(common)].reset_index(drop=True)
    report.add_figure("PID discriminant",
                      pid_plots.discriminant_distributions(usable, arm_style),
                      name="pid_discriminant")
    report.add_figure("PID ROC", pid_plots.roc_curves(usable, arm_style),
                      name="pid_roc")
    summaries = {a: pid_metrics.summarise(s) for a, s in usable.items()}
    report.add_figure("PID summary",
                      pid_plots.pid_summary_bars(summaries, arm_style),
                      name="pid_summary")
    stacked = pd.concat(
        [t.assign(arm=a) for a, t in summaries.items() if not t.empty],
        ignore_index=True)
    report.add_table("PID summary", stacked, name="pid",
                     note="'eff_mu_at_..._e' is the muon efficiency at a fixed "
                          "electron efficiency — the comparison that survives a "
                          "constant offset between two arms' discriminants.")
    if not truths:
        return
    classified = {
        arm: {particle: pid_metrics.classified(table, truths[particle].primary,
                                               particle)
              for particle, table in samples.items() if particle in truths}
        for arm, samples in usable.items()}
    profiles = pid_metrics.efficiency_profiles(classified)
    if profiles.empty:
        return
    report.add_figure(
        "PID efficiency against momentum, dwall and towall",
        rate_plots.rate_profiles(
            profiles.assign(row=profiles["true_particle"].map(
                lambda p: f"true {p} identified as {p}")),
            arm_style, row_key="row", versus=PROFILE_VARIABLES,
            title="PID efficiency at dnLL = 0"),
        name="pid_efficiency_profiles",
        note="Fraction of each true particle identified correctly at dnLL = 0, "
             "binned by its true momentum and vertex geometry. Error bars: 68% "
             "Wilson intervals.")
    report.add_csv("pid_efficiency_profiles", profiles)


def _add_subevent_section(report, arms, truth, arm_style, *, window_ns):
    tables = subevent_metrics.build(arms, truth, window_ns=window_ns)
    if not tables:
        return
    summary = subevent_metrics.summary(tables)
    first = next(iter(tables.values()))[0]
    report.add_table(
        "Sub-eventing efficiency and purity", summary, name="subevents",
        note=(f"{first.attrs['n_events']} events with a recorded prompt "
              f"({first.attrs['n_unrecorded_prompt']} whose prompt fired no "
              "trigger are left out). Reconstructed and true sub-events "
              f"(prompt; decay electrons that fired a trigger) matched one to "
              f"one in time within ±{window_ns:g} ns. Efficiency = matched true "
              "/ true; prompt purity = earliest reco sub-events that are the "
              "prompt; delayed purity = later reco sub-events that are a decay "
              "electron. Rates in %, 68% Wilson intervals."))
    report.add_figure(
        "Sub-eventing efficiency and purity",
        rate_plots.summary_bars(summary, arm_style,
                                title="Sub-eventing efficiency and purity"),
        name="subevents")
    profiles = subevent_metrics.profiles(tables)
    if not profiles.empty:
        report.add_figure(
            "Sub-eventing against momentum, dwall and towall",
            rate_plots.rate_profiles(
                profiles, arm_style, row_key="metric",
                rows=["prompt efficiency", "decay efficiency", "purity"],
                title="Sub-eventing against p, dwall and towall"),
            name="subevent_profiles",
            note="Efficiencies are binned by the true sub-event's own momentum "
                 "and vertex (a decay electron's, not its muon's); purity by "
                 "the event's prompt.")
        report.add_csv("subevent_profiles", profiles)
    report.add_figure("Sub-event time offsets",
                      rate_plots.time_offsets(tables, arm_style,
                                              window_ns=window_ns),
                      name="subevent_time_offsets")
    report.add_table("Sub-event counts", subevent_metrics.count_table(tables)
                     .pivot_table(index=["arm", "n_true"], columns="n_reco",
                                  values="events", fill_value=0)
                     .rename(columns=lambda c: f"reco {c}").reset_index(),
                     name="subevent_counts",
                     note="Events by true and reconstructed sub-event count "
                          "(4 = four or more).")


def _add_multitrack_section(report, arms, truth, arm_style, *, seed):
    tables = multitrack_metrics.build(arms, truth)
    if not tables:
        return
    sources = {label: t.attrs["source"] for label, t in tables.items()}
    n = next(iter(tables.values())).attrs["n_multitrack"]
    summaries = {label: residual_metrics.summarise(t, label, seed=seed)
                 for label, t in tables.items()}
    report.add_table(
        "Multi-track events — interaction vertex and leading particle",
        table_io.headline_table(summaries), name="multitrack_headline",
        note=(f"{n} true events with two or more visible tracks. Vertex: the "
              "fit's vertex against the true interaction vertex, split along "
              "and across the most energetic particle. Angle and momentum: the "
              "leading reconstructed ring against the most energetic visible "
              "particle. Leading ring from: "
              + "; ".join(f"{k}: {v}" for k, v in sources.items())))
    for versus in PROFILE_VARIABLES:
        report.add_csv(f"multitrack_resolution_{versus}",
                       resolution_metrics.headline_profile_table(tables, versus))
        report.add_figure(
            f"Multi-track events — resolution against {versus}",
            residual_plots.resolution_profiles(tables, arm_style, versus=versus),
            name=f"multitrack_resolution_{versus}")
    pid = multitrack_metrics.pid_summary(tables)
    report.add_table("Multi-track events — leading-particle PID", pid,
                     name="multitrack_pid",
                     note="Leading reconstructed ring's type equals the most "
                          "energetic visible particle's ring type, per true "
                          "type. Rates in %, 68% Wilson intervals.")
    profiles = pid_metrics.efficiency_profiles(
        multitrack_metrics.pid_classified(tables))
    if not profiles.empty:
        report.add_figure(
            "Multi-track events — leading-particle PID against p, dwall, towall",
            rate_plots.rate_profiles(
                profiles.assign(row=profiles["true_particle"].map(
                    lambda p: f"leading true {p} identified")),
                arm_style, row_key="row", versus=PROFILE_VARIABLES,
                title="Multi-track leading-particle PID"),
            name="multitrack_pid_profiles")
        report.add_csv("multitrack_pid_profiles", profiles)


def _add_ring_section(report, arms, truth, arm_style):
    per_event, per_ring = {}, {}
    for arm in arms:
        if arm.multiring.empty:
            continue
        rings, events = ring_metrics.build_tables(arm.multiring, truth.rings)
        if not events.empty:
            per_event[arm.label] = events
            per_ring[arm.label] = rings
    if not per_event:
        return
    report.add_figure("Ring multiplicity",
                      ring_plots.multiplicity(per_event, arm_style),
                      name="ring_multiplicity")
    report.add_figure("Ring-counting accuracy",
                      ring_plots.counting_accuracy(per_event, arm_style),
                      name="ring_counting")
    summaries = {a: ring_metrics.ring_summary(r, a) for a, r in per_ring.items()}
    report.add_figure("Per-ring quality",
                      ring_plots.per_slot_quality(summaries, arm_style),
                      name="ring_quality")
    for label, events in per_event.items():
        report.add_figure(f"Ring-count confusion — {label}",
                          ring_plots.confusion_heatmap(events, label),
                          name=f"ring_confusion_{_slug(label)}")
    report.add_table("Ring counting", pd.concat(
        [ring_metrics.counting_summary(e, a) for a, e in per_event.items()],
        ignore_index=True), name="ring_counting",
        note="Only true rings above Cherenkov threshold are counted. "
             "'balanced' averages the per-true-count accuracies, so a fitter "
             "that always answers '1' cannot score well on a 1-ring-heavy "
             "sample.")
    report.add_table("Per-ring quality", pd.concat(
        [s for s in summaries.values() if not s.empty], ignore_index=True),
        name="ring_quality")


def _add_intermediate_section(report, arms, truth, arm_style, hypothesis):
    charge = {}
    for arm in arms:
        if arm.event_totals is None or arm.event_totals.empty:
            continue
        table = intermediate_metrics.charge_normalisation(
            arm.fits, arm.event_totals, label=arm.label)
        if not table.empty:
            charge[arm.label] = table
    if charge:
        report.add_figure("Charge-model normalisation",
                          intermediate_plots.charge_normalisation(charge, arm_style),
                          name="charge_normalisation")
        report.add_table(
            "Charge-model normalisation",
            intermediate_metrics.charge_normalisation_summary(list(charge.values())),
            name="charge_normalisation",
            note="Predicted total photoelectrons over observed charge. A "
                 "hypothesis systematically below 1 must explain the missing "
                 "light through the unhit and occupancy terms instead, which is "
                 "how a constant PID offset arises.")

    term_summaries, differences, improvements = {}, {}, {}
    split_arms = []
    for arm in arms:
        if arm.nll_terms.empty:
            continue
        term_summaries[arm.label] = intermediate_metrics.term_summary(
            arm.nll_terms, label=arm.label)
        differences[arm.label] = intermediate_metrics.hypothesis_term_difference(
            arm.nll_terms, label=arm.label)
        improvements[arm.label] = intermediate_metrics.zero_ring_improvement(
            arm.nll_terms, label=arm.label)
        if intermediate_metrics.has_term_split(arm.nll_terms):
            split_arms.append(arm.label)

    # The per-term breakdown needs an arm that actually HAS terms. An official
    # fiTQun file publishes fq1rnll and fq0rnll and nothing between them, so a
    # fiTQun-only report drew a chart with a single 'total' bar per hypothesis
    # that read as a broken plot rather than as a property of the input.
    if split_arms:
        report.add_figure("Likelihood decomposition",
                          intermediate_plots.nll_term_breakdown(term_summaries,
                                                                arm_style),
                          name="nll_terms")
    if split_arms:
        report.add_table("Likelihood decomposition", pd.concat(
            [t for t in term_summaries.values() if t is not None and not t.empty],
            ignore_index=True), name="nll_terms",
            note="Coral splits its NLL four ways; fiTQun reports a total only. "
                 "Absolute NLLs are NOT comparable between codes — compare the "
                 "improvement over each code's own zero-ring hypothesis.")
    if any(t is not None and not t.empty for t in improvements.values()):
        report.add_figure("Likelihood improvement over the zero-ring hypothesis",
                          intermediate_plots.nll_improvement(improvements,
                                                             arm_style),
                          name="nll_improvement")
    if any(d is not None and not d.empty for d in differences.values()):
        report.add_figure("What separates μ from e",
                          intermediate_plots.hypothesis_term_difference(
                              differences, arm_style),
                          name="nll_term_difference")

    trajectories = {}
    for arm in arms:
        if arm.stages.empty:
            continue
        trajectories[arm.label] = intermediate_metrics.stage_trajectory(
            arm.stages, truth.primary, label=arm.label, hypothesis=hypothesis)
    # A trajectory needs at least two stages to be a trajectory. This
    # production's fiTQun files record `fqtest1rstage == 1` for every entry --
    # only the final stage was kept -- so the section is simply not produced.
    staged = {a: t for a, t in trajectories.items()
              if t is not None and not t.empty and t["stage"].nunique() > 1}
    if staged:
        report.add_figure("Fit-stage trajectory",
                          intermediate_plots.stage_trajectory(staged, arm_style),
                          name="stage_trajectory")
        report.add_table("Fit-stage trajectory", pd.concat(
            list(staged.values()), ignore_index=True), name="stage_trajectory",
            note="Stage numbering is each algorithm's own sequence; compare the "
                 "shape of the improvement, not stage 1 against stage 1.")


def _slug(text):
    from .report.builder import _slug as slug
    return slug(text)


def _common_root(arms):
    """The workspace the arms share, for shortening paths in the tables.

    Walks up from a reconstruction file to the directory holding Output/, which
    is the root everything else in a report is quoted relative to.
    """
    for arm in arms:
        path = Path(arm.meta.get("path", ""))
        for parent in path.parents:
            if parent.name == "Output":
                return parent.parent
    return None
