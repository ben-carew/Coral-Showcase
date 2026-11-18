#!/usr/bin/env python3

"""Generate the Sextant user guide PDF.

    python Software/Sextant/docs/make_guide.py [--output PATH]

Kept with the package rather than in Analysis/Scripts so the guide travels with
the code it documents and a collaborator can regenerate it after a change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table,
                                TableStyle)

REPO = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d8d7d2")
CODE_BG = colors.HexColor("#f4f3f0")
ACCENT = colors.HexColor("#2a78d6")


def styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("title", parent=base["Title"], fontSize=30,
                                leading=34, textColor=INK, spaceAfter=6)
    s["subtitle"] = ParagraphStyle("subtitle", parent=base["Normal"], fontSize=13,
                                   leading=18, textColor=MUTED, spaceAfter=18)
    s["h1"] = ParagraphStyle("h1", parent=base["Heading1"], fontSize=17, leading=21,
                             textColor=INK, spaceBefore=16, spaceAfter=8)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5,
                             leading=16, textColor=INK, spaceBefore=12,
                             spaceAfter=5)
    s["body"] = ParagraphStyle("body", parent=base["BodyText"], fontSize=9.6,
                               leading=14, textColor=INK, alignment=TA_LEFT,
                               spaceAfter=7)
    s["note"] = ParagraphStyle("note", parent=s["body"], fontSize=9.0,
                               leading=13, textColor=MUTED, leftIndent=8,
                               borderPadding=0)
    s["code"] = ParagraphStyle("code", parent=base["Code"], fontSize=8.2,
                               leading=11.4, textColor=INK,
                               backColor=CODE_BG, borderPadding=6,
                               spaceBefore=4, spaceAfter=9)
    s["caption"] = ParagraphStyle("caption", parent=s["body"], fontSize=8.4,
                                  textColor=MUTED, spaceBefore=-4, spaceAfter=10)
    return s


S = styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


def C(text):
    return Preformatted(text.strip("\n"), S["code"])


def bullets(items):
    return ListFlowable([ListItem(P(t), leftIndent=12) for t in items],
                        bulletType="bullet", start="•", leftIndent=14,
                        bulletFontSize=8, spaceAfter=8)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["body"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                  ("BOTTOMPADDING", (0, 0), (-1, 0), 6)]
    t.setStyle(TableStyle(style))
    return t


class Guide(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4,
                         leftMargin=22 * mm, rightMargin=22 * mm,
                         topMargin=20 * mm, bottomMargin=18 * mm,
                         title="Sextant — User Guide", author="Sextant", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width,
                      self.height, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame]),
            PageTemplate(id="main", frames=[frame], onPage=self._chrome),
        ])

    def _chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        y = doc.bottomMargin - 6 * mm
        canvas.line(doc.leftMargin, y, doc.leftMargin + doc.width, y)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, y - 4.5 * mm,
                          "Sextant — water-Cherenkov reconstruction validation")
        canvas.drawRightString(doc.leftMargin + doc.width, y - 4.5 * mm,
                               f"{doc.page - 1}")
        canvas.restoreState()


def build(output: Path):
    story = []

    # ---- cover -----------------------------------------------------------
    story += [
        Spacer(1, 55 * mm),
        P("Sextant", "title"),
        P("Validation and comparison of water-Cherenkov event reconstruction "
          "for Hyper-Kamiokande", "subtitle"),
        Spacer(1, 4 * mm),
        P("A user guide and tutorial", "body"),
        Spacer(1, 90 * mm),
        P("Compares Coral, fiTQun and anything else that can be read into its "
          "canonical tables — against MC truth, against each other, and step "
          "by step through their internals.", "note"),
        NextPageTemplate("main"),
        PageBreak(),
    ]


    # ---- 0. the name -----------------------------------------------------
    story += [
        P("Why “Sextant”?", "h1"),
        P("A sextant measures the angle between an object and a fixed "
          "reference, and it is the reference that makes the measurement mean "
          "anything: the same instrument, pointed at the same star, tells you "
          "nothing until you also sight the horizon.", "body"),
        P("That is exactly the shape of this problem. A reconstructed vertex is "
          "not interesting on its own — it is interesting relative to MC truth, "
          "and relative to what another algorithm did with the same event. "
          "Every number this package produces is a comparison against a stated "
          "reference, and most of its machinery exists to make sure the "
          "reference is the right one: that an arm is scored against the truth "
          "it belongs to, that two algorithms' predicted charges are evaluated "
          "at the <i>same</i> parameter point, that two configurations really "
          "are comparable.", "body"),
        P("It is also a navigation instrument, which suits a package whose main "
          "job is telling you where you are relative to fiTQun — and it "
          "belongs to the same nautical family as Coral, the algorithm it was "
          "first built to measure.", "body"),
        P("The name is cosmetic. Nothing depends on it, and it is one rename "
          "away from being anything else.", "note"),
    ]

    # ---- 1. what it is ---------------------------------------------------
    story += [
        P("1. What Sextant is", "h1"),
        P("Sextant measures not only what a reconstruction algorithm "
          "<i>outputs</i> — vertex, direction, momentum, PID, ring counting — "
          "but what it computes on the way there: predicted charge, the "
          "components of its likelihood, and how its estimate moves through "
          "the fit stages. Comparing final vertices tells you <i>that</i> "
          "fiTQun is better on single-ring electrons; comparing intermediates "
          "is how you find out <i>why</i>.", "body"),
        P("Two halves", "h2"),
        table([
            ["Half", "Modules", "What it needs"],
            ["<b>Validation</b> — residuals, resolutions, PID, ring counting, "
             "intermediates, the PDF and CSV report",
             "<font face='Courier'>io, metrics, plots, report, analysis</font>",
             "numpy, pandas, matplotlib, uproot, scipy. Runs on a laptop with "
             "no HK software installed."],
            ["<b>Experiments</b> — launching arms, keeping their configs in "
             "step, collecting results, probing internals",
             "<font face='Courier'>run, probe</font>",
             "The reconstruction workspace and <font face='Courier'>exp.py</font>; "
             "for probing, Coral or a built fiTQun."],
        ], [46 * mm, 40 * mm, 62 * mm]),
        P("The validation half never imports the experiment half. You can point "
          "it at three ROOT files you made by hand and get the whole report — "
          "which is the mode most people will use most of the time.", "body"),
        P("An <b>arm</b> is the unit of comparison: one algorithm, at one "
          "version, with one configuration, on one sample. “Coral ben/sandbox "
          "on 1000 electrons” is an arm; so is “official fiTQun 6.4.4 on the "
          "beam sample”.", "body"),
    ]

    # ---- 2. install ------------------------------------------------------
    story += [
        P("2. Installing", "h1"),
        C("""
pip install -e Software/Sextant
sextant --help
"""),
        P("Two optional extras, neither needed for the validation half:", "body"),
        bullets([
            "<b>reportlab</b> — only to regenerate this guide.",
            "<b>fqprobe</b> — the per-PMT fiTQun probe; see section 9.",
        ]),
    ]

    # ---- 3. data model ---------------------------------------------------
    story += [
        P("3. The data model", "h1"),
        P("Everything downstream of <font face='Courier'>io</font> is "
          "algorithm-agnostic. Readers flatten every supported on-disk format "
          "into five canonical tables, and no metric, plot or table ever asks "
          "which code produced a number. Adding a third algorithm means "
          "writing one reader.", "body"),
        table([
            ["Table", "One row per", "Holds"],
            ["<font face='Courier'>fits</font>",
             "event × sub-event × hypothesis × ring × stage",
             "Every fit result, from a single-ring e− fit to ring 3 of a "
             "6-ring fit to the intermediate result of fit stage 1."],
            ["<font face='Courier'>truth_rings</font>", "true Cherenkov ring",
             "Kinematics, whether it is the primary lepton, whether it is "
             "above Cherenkov threshold."],
            ["<font face='Courier'>truth_events</font>", "event",
             "Interaction topology (CC/NC, flavour, NEUT mode), true vertex, "
             "hit and charge totals, dwall and towall."],
            ["<font face='Courier'>nll_terms</font>",
             "event × hypothesis × point × term",
             "A likelihood decomposed into named components, at a named "
             "parameter point."],
            ["<font face='Courier'>pmt_predictions</font>",
             "event × hypothesis × point × PMT",
             "Predicted direct and scattered light, observed charge and time, "
             "time residual."],
        ], [34 * mm, 42 * mm, 72 * mm]),
        C("sextant schema              # print all five\nsextant schema fits         # just one"),
        P("<b>Event numbering.</b> Every table is keyed on a global integer "
          "<font face='Courier'>event = file_index × stride + entry</font>. The "
          "stride is a per-production property and must exceed the number of "
          "events in one file, or ids from adjacent files alias onto each "
          "other silently. The particle guns hold 100 events per file and use "
          "a stride of 100; the beam production holds <b>101</b> and uses 1000. "
          "A test enforces this.", "note"),
    ]

    # ---- 4. tutorial 1 ---------------------------------------------------
    story += [
        PageBreak(),
        P("4. Tutorial: comparing files you already have", "h1"),
        P("The shortest path to a report. No experiment spec, no cluster — just "
          "labelled paths and a truth file.", "body"),
        P("Step 1 — check what you have", "h2"),
        C("""
sextant productions          # samples Sextant knows about
sextant schema               # the canonical tables
"""),
        P("Step 2 — build the report", "h2"),
        C("""
sextant compare \\
    --arm "Coral sandbox=Output/Coral/Reco/fitval_electron/coral_electron_s00_prod.root" \\
    --arm "fiTQun=Output/fiTQun/Reco/fitqun_electron.root" \\
    --truth Output/Truth/truth_electron.root \\
    --reference "fiTQun" \\
    --outdir Output/Sextant/e_comparison/report
"""),
        P("The label before the <font face='Courier'>=</font> is what identifies "
          "that arm in every figure and table, so make it descriptive. "
          "<font face='Courier'>--reference</font> names the arm everything "
          "else is compared against; it defaults to the first.", "body"),
        P("Step 3 — read the output", "h2"),
        C("""
Output/Sextant/<experiment>/
    specs/*.ini         the exp.py spec generated for each arm
    experiment.json     what was launched
    probe/              per-PMT intermediates, if probed
    report/
        report.pdf      every figure and table, in reading order
        figures/*.png   the same figures individually, for slides
        tables/*.csv    every number in the report, machine-readable
        manifest.json   what was compared, and what produced it
"""),
        P("Everything one experiment produces lives under "
          "<font face='Courier'>Output/Sextant/&lt;experiment&gt;/</font>. "
          "<font face='Courier'>Analysis/Reports/</font> is deliberately not "
          "used: that is for bespoke, authored write-ups about Coral "
          "development, not for generated validation output.", "body"),
        P("Sections are skipped, with a note, when the inputs cannot support "
          "them — a Coral-only comparison has no per-PMT section, a "
          "single-particle sample has no PID section — so the same command "
          "works for every combination.", "body"),
        P("The same thing from Python", "h2"),
        C("""
from sextant import Arm, Truth, analysis

arms  = [Arm.from_file("Coral sandbox", ".../coral_electron.root"),
         Arm.from_file("fiTQun",        ".../fitqun_electron.root")]
truth = Truth.from_file(".../truth_electron.root")

analysis.standard_report(arms, truth, outdir=".../Reports/e_comparison",
                         reference="fiTQun")
"""),
        P("Or go straight to the numbers, skipping the report entirely:", "body"),
        C("""
from sextant.metrics import residuals, stats

res = residuals.compute(arms[0].hypothesis("e-"), truth.primary,
                        truth_events=truth.events)
print(stats.median(res["vertex_dist"]), stats.iqr68(res["p_res_pct"]))
print(residuals.summarise(res, "Coral sandbox"))
"""),
    ]

    # ---- 5. PID ----------------------------------------------------------
    story += [
        P("5. Adding PID", "h1"),
        P("PID needs two samples of known true particle, because the quantity "
          "of interest is the <i>separation</i> between them. Pass the arms "
          "grouped by true particle:", "body"),
        C("""
pid_samples = {
    "e-":  [Arm.from_file("fiTQun", ".../fitqun_electron.root")],
    "mu-": [Arm.from_file("fiTQun", ".../fitqun_muon.root")],
}
analysis.standard_report(arms, truth, outdir=..., pid_samples=pid_samples)
"""),
        P("The discriminant is <font face='Courier'>dnLL = nLL(μ) − nLL(e)</font>, "
          "so <b>dnLL &gt; 0 is electron-like</b>. Sextant reports the ROC, the "
          "AUC, and the muon efficiency at a <i>fixed</i> electron efficiency. "
          "That last number is the one to quote when comparing two arms: it is "
          "invariant to a constant offset between their discriminants, where "
          "“fraction correct at dnLL &gt; 0” is not.", "body"),
        P("Sextant does not require both hypotheses to have converged. "
          "Requiring it removes a large and <i>biased</i> slice of the sample — "
          "the e− fit routinely fails on a true muon and vice versa, so the cut "
          "preferentially discards exactly the events the discriminant finds "
          "hardest.", "note"),
    ]

    # ---- 6. tutorial 2: experiment ---------------------------------------
    story += [
        P("6. Tutorial: running a whole experiment", "h1"),
        P("One spec file describes a comparison: the sample, the arms, and the "
          "settings they must share. Copy an example and edit it.", "body"),
        C("""
cp Software/Sextant/examples/electron_comparison.ini \\
   Analysis/Experiments/my_comparison.ini
"""),
        P("Anatomy of a spec", "h2"),
        C("""
[experiment]
name       = e_comparison        ; run dirs and result names derive from this
production = v1.12.22
sample     = electron
events     = 1000
where      = server              ; server | local
hypothesis = e-                  ; which fit hypothesis the report is about
reference  = fitqun              ; the arm everything is compared against
truth      = Output/Truth/truth_electron.root

[run]
events_per_job = 25              ; packing; lower it for slow configs

[shared]                         ; concepts, translated into each code's own key
use_time_likelihood = true
do_electron_fit     = true
do_muon_fit         = true

[arm.coral_sandbox]
algorithm = coral
slot      = a                    ; worktree slot -- yours, not another agent's
branch    = ben/sandbox
scheme    = coral-production     ; calibration scheme (section 7)
label     = Coral ben/sandbox
config.fits.truth_seed = False   ; native Coral setting, no fiTQun counterpart

[arm.fitqun]
algorithm = fitqun
scheme    = fitqun-hk2024-wscat
label     = fiTQun HK-2024
params    = Software/fiTQun/ParameterOverrideFiles/Official_HyperK_2024_wscat.parameters.dat
"""),
        P("For a fiTQun arm, the parameter file is the single biggest lever on "
          "runtime. <font face='Courier'>sextant check</font> prints a COST "
          "line listing the expensive fits a file enables; the full official "
          "files run the multi-ring and pi0 fitters at roughly 20 minutes per "
          "event, where the <font face='Courier'>*_1Remu</font> variants run "
          "the single-ring e/mu fits alone.", "body"),
        P("Arms may also read results rather than produce them:", "body"),
        table([
            ["<font face='Courier'>source</font>", "Meaning"],
            ["<font face='Courier'>run</font> (default)", "Launch it."],
            ["<font face='Courier'>official</font>",
             "Read the reconstruction shipped with the production (fiTQun "
             "only; see section 8)."],
            ["<font face='Courier'>file</font>",
             "Read an existing merged output named by "
             "<font face='Courier'>path</font>."],
        ], [30 * mm, 118 * mm]),
        P("The workflow", "h2"),
        C("""
sextant check   Analysis/Experiments/my_comparison.ini   # do the arms agree?
sextant run     Analysis/Experiments/my_comparison.ini   # launch every arm
sextant status  Analysis/Experiments/my_comparison.ini   # progress
sextant errors  Analysis/Experiments/my_comparison.ini   # what failed
sextant logs    Analysis/Experiments/my_comparison.ini --arm coral_sandbox
sextant report  Analysis/Experiments/my_comparison.ini --collect
"""),
        P("<font face='Courier'>run</font> refuses to launch if "
          "<font face='Courier'>check</font> reports an error, unless you pass "
          "<font face='Courier'>--force</font>. Use "
          "<font face='Courier'>--dry-run</font> to see what would be submitted "
          "without submitting it.", "body"),
        P("Sextant does not submit jobs itself. It writes one "
          "<font face='Courier'>exp.py</font> spec per arm into "
          "<font face='Courier'>Output/Sextant/&lt;experiment&gt;/specs/</font> "
          "and calls <font face='Courier'>exp.py</font>, which freezes the "
          "code, config and parameter file into each run directory. Runs stay "
          "immutable and remain visible to "
          "<font face='Courier'>exp.py status</font> exactly as before — and "
          "<font face='Courier'>sextant cancel</font> scopes to this "
          "experiment's runs only, never to everything you own.", "note"),
        P("<font face='Courier'>report --collect</font> merges and pulls first. "
          "Without it, the report is built from whatever is already merged, "
          "which is what you want when iterating on plots.", "body"),
    ]

    # ---- 7. config -------------------------------------------------------
    story += [
        P("7. Keeping two configurations honest", "h1"),
        P("<font face='Courier'>sextant check</font> compares the arms setting "
          "by setting, using a mapping of <i>concepts</i> to each code's own "
          "key names. <font face='Courier'>sextant mapping</font> prints the "
          "whole table.", "body"),
        P("Most settings are genuinely shared and can be set once, in "
          "<font face='Courier'>[shared]</font>, using the concept name — "
          "Sextant writes it into each code's own key:", "body"),
        table([
            ["Concept", "Coral", "fiTQun"],
            ["<font face='Courier'>use_time_likelihood</font>",
             "<font face='Courier'>[likelihood] use_time_pdf</font>",
             "<font face='Courier'>fiTQun.UseTimeLikelihood</font>"],
            ["<font face='Courier'>rccut_a0</font>",
             "<font face='Courier'>[multiring] rccut_a0</font>",
             "<font face='Courier'>fiTQun.RCCuta0WCSim</font>"],
            ["<font face='Courier'>do_muon_fit</font>",
             "<font face='Courier'>[fits] particles</font> (membership)",
             "<font face='Courier'>fiTQun.DoMuon1RFit</font>"],
        ], [43 * mm, 43 * mm, 62 * mm]),
        P("Why some settings cannot be forced", "h2"),
        P("The photon yield, the water attenuation length and the "
          "scattered-light treatment are fitted <i>together</i> against "
          "observed light. They form a self-consistent <b>calibration scheme</b> "
          "whose members must not be mixed. fiTQun itself ships them as matched "
          "sets — <font face='Courier'>wscat</font> is 0.221 / 7000 / on, "
          "<font face='Courier'>noscat</font> is 0.240 / 7500 / off — and "
          "Coral's production tuning is a third, 0.176 / 9500 with scattered "
          "light routed to the time PDF only.", "body"),
        P("Copying fiTQun's quantum efficiency onto Coral while leaving Coral's "
          "attenuation length reintroduces a large momentum bias: a “matched” "
          "configuration that measures nothing but the mismatch.", "body"),
        P("So each arm <i>declares</i> a scheme. Sextant verifies the arm is "
          "internally consistent with it, records both schemes in the report, "
          "and refuses to interleave them. A "
          "<font face='Courier'>[shared]</font> entry naming a calibration "
          "constant is ignored by design — change the scheme instead.", "body"),
        C("""
$ sextant check my_comparison.ini
[ERROR  ] physics.quantum_efficiency: scheme 'coral-production' expects
          physics.quantum_efficiency=0.176, config has 0.221. These constants
          are fitted together — changing one alone reintroduces a momentum
          bias. Change the scheme, or declare a new one.
"""),
        P("Two arms using <i>different</i> schemes is normal and is reported as "
          "information, not an error: it means you are comparing two calibrated "
          "algorithms, which is usually the honest comparison.", "note"),
    ]

    # ---- 8. beam ---------------------------------------------------------
    story += [
        P("8. The neutrino beam production", "h1"),
        P("Alongside the particle guns, Sextant supports the NEUT beam "
          "production (<font face='Courier'>beam-fhc</font>, four flavour "
          "channels), which carries full neutrino final states and one or two "
          "triggers per event.", "body"),
        P("It ships an official fiTQun 6.4.4 reconstruction, which Sextant "
          "reads directly as a reference arm — no cluster time at all:", "body"),
        C("""
[arm.fitqun_official]
algorithm = fitqun
source    = official
label     = fiTQun 6.4.4 (official)
"""),
        P("Building truth", "h2"),
        C("""
# On the cluster, where the production is mounted:
sextant truth --production beam-fhc --sample numu_x_numu \\
              --output Output/Truth/truth_beam_numu.root --max-files 100
"""),
        P("<b>Size the truth by files, not by events.</b> One job reads one "
          "input file and takes <font face='Courier'>events_per_job</font> "
          "events from it, so a run spans as many files as it has <i>jobs</i>. "
          "1000 events at 10 per job touches 100 files — not the 10 that "
          "1000/101 would suggest. Truth built over 10 files then covers a "
          "tenth of the run, every paired comparison collapses to the overlap, "
          "and the validation page says so as an event-overlap warning. Build "
          "truth with <font face='Courier'>--max-files</font> equal to the job "
          "count.", "note"),
        P("True rings are the primary final-state particles "
          "(<font face='Courier'>flag == 0</font>, "
          "<font face='Courier'>parenttype == 0</font>), with the outgoing "
          "lepton flagged, <font face='Courier'>visible</font> set for those "
          "above Cherenkov threshold, and a primary π⁰ expanded into its two "
          "decay photons. Per-event labels record CC/NC, neutrino flavour and "
          "energy, NEUT interaction mode and the true ring count.", "body"),
        P("Sub-threshold neutrals are kept with "
          "<font face='Courier'>visible = False</font> rather than dropped, so "
          "“what did the fitter miss” can be asked of the full final state "
          "while every ring metric filters them out. Single-ring metrics still "
          "work: <font face='Courier'>Truth.primary</font> selects the primary "
          "lepton where there is one, else the most energetic visible ring.",
          "body"),
    ]

    # ---- 9. intermediates ------------------------------------------------
    story += [
        P("9. Comparing the internals", "h1"),
        P("Four levels, in increasing order of what they require.", "body"),
        table([
            ["Level", "What it shows", "Needs"],
            ["<b>1. Charge normalisation</b>",
             "Total predicted photoelectrons over total observed charge, per "
             "hypothesis. A hypothesis systematically below 1 must explain the "
             "missing light through the unhit and occupancy terms instead — "
             "which is how a constant PID offset arises.",
             "Nothing beyond the standard outputs."],
            ["<b>2. Likelihood decomposition</b>",
             "Coral's unhit / phit / cpdf / tpdf split, and each code's "
             "improvement over its own zero-ring hypothesis.",
             "Coral: <font face='Courier'>[in_out] dump_nll_terms</font>."],
            ["<b>3. Fit trajectory</b>",
             "Residuals after each fit stage — where in the sequence accuracy "
             "is won, and whether a stage ever makes things worse.",
             "Official fiTQun files; Coral stages via probe."],
            ["<b>4. Per-PMT predictions</b>",
             "Predicted direct and scattered light, PMT by PMT.",
             "The two probes below."],
        ], [34 * mm, 76 * mm, 38 * mm]),
        P("Absolute NLL values are never compared across codes — they carry "
          "different constant offsets and different unhit subsampling. What is "
          "comparable is each code's improvement over its <i>own</i> zero-ring "
          "hypothesis, which is what Sextant reports.", "note"),
        P("Why the parameter point matters", "h2"),
        P("Predicted charges compared at each code's <i>own</i> fitted vertex "
          "differ mostly because the vertices differ — which is the thing you "
          "were trying to explain. Evaluated at a <b>common</b> point, what "
          "remains is the model difference:", "body"),
        table([
            ["Point", "Meaning"],
            ["<font face='Courier'>truth</font>",
             "Both codes at the MC truth parameters. The cleanest comparison of "
             "the models themselves, and the only point at which “which code "
             "predicts the observed light better” is well posed."],
            ["<font face='Courier'>ref</font>",
             "Both codes at a chosen reference arm's fit. Separates a bad model "
             "from a bad minimisation: “given fiTQun's answer, what would Coral "
             "have predicted there?”"],
            ["<font face='Courier'>self</font>",
             "Each code at its own fit. Right for the likelihood decomposition; "
             "<b>never</b> for a cross-code per-PMT comparison — Sextant "
             "refuses that combination."],
        ], [26 * mm, 122 * mm]),
        P("Running the probes", "h2"),
        C("""
# Coral: calls Coral's own compute_predicted_photons and nll_term_sums
sextant probe my_comparison.ini --config Software/Coral/config.ini --events 20

# fiTQun: build the standalone probe once
cmake -S Software/Sextant/cpp/fqprobe -B build/fqprobe \\
      -DFITQUN_DIR=<fiTQun install>
cmake --build build/fqprobe
export SEXTANT_FQPROBE=$PWD/build/fqprobe/fqprobe
"""),
        P("<font face='Courier'>fqprobe</font> is a read-only program that links "
          "fiTQun's public library and calls "
          "<font face='Courier'>Get1Rmudist</font>. It changes nothing in "
          "fiTQun's fitting path. fiTQun's own "
          "<font face='Courier'>OutputQTdist</font> was the obvious alternative "
          "and is unusable here: the call that fills it sits inside "
          "<font face='Courier'>#ifndef NOSKLIBRARIES</font>, which the HyperK "
          "build defines, and it bins in (cos θ, time residual) rather than "
          "reporting per PMT.", "note"),
        P("Per-PMT tables are large — roughly 50k rows per event per "
          "hypothesis — so a probe run is a diagnostic over tens of events, not "
          "a production pass. Event selection is seeded, so a probe re-run "
          "looks at the same events and can actually be compared.", "body"),
    ]

    # ---- 10. reading the report -----------------------------------------
    story += [
        P("10. Reading the report", "h1"),
        P("Read the first two pages before the numbers.", "body"),
        P("Arms compared", "h2"),
        P("What produced each arm: algorithm, event count, code SHA, branch, "
          "dirty flag, config digest. A result whose code and config are not "
          "recorded is not reproducible.", "body"),
        P("Validation checks", "h2"),
        P("The expensive mistakes in this domain are not wrong formulas — they "
          "are correctly computed comparisons between unrelated things, and "
          "they produce a full set of plausible plots. These checks catch "
          "them:", "body"),
        bullets([
            "<b>truth pairing</b> — a median vertex residual larger than the "
            "detector means the arm is matched against the wrong truth sample. "
            "This catches what event-id overlap does not: two samples can share "
            "id ranges and still be completely unrelated.",
            "<b>event overlap</b> — no shared ids, or a low overlap fraction.",
            "<b>duplicate keys</b> — anything that would make a join ambiguous.",
            "<b>time pedestal</b> — arms disagreeing on the t0 convention.",
            "<b>success rate</b> — an arm most of whose fits failed.",
        ]),
        P("Residuals, on two pages", "h2"),
        P("The <b>per-parameter</b> page shows the seven fitted quantities "
          "(t, x, y, z, θ, φ, p) — these say <i>how</i> a fit is wrong. The "
          "<b>aggregate</b> page shows vertex displacement, angular "
          "displacement and momentum resolution — these are what gets quoted. "
          "They are separated deliberately: θ and φ carry a coordinate "
          "artefact (φ wraps and is degenerate near the poles) that the "
          "opening angle does not, and reading one as if it were the other is "
          "an easy mistake to make on a shared page.", "body"),
        P("Statistics", "h2"),
        P("Residual distributions here are not Gaussian: vertex displacement "
          "and opening angle are one-sided with heavy tails, and a handful of "
          "mis-converged events inflate a standard deviation enormously. So "
          "Sextant reports <b>median</b> and <b>iqr68</b> — half the central "
          "68% width, which coincides with σ for a true Gaussian and is what "
          "makes “resolution” comparable between a well-behaved arm and one "
          "with a bad tail.", "body"),
        P("Headline numbers carry bootstrap confidence intervals (2000 "
          "resamples, seeded so a regenerated report gives the same intervals). "
          "The paired-difference table's "
          "<font face='Courier'>excludes_zero</font> column is the honest "
          "verdict: with a few hundred events most differences do not exclude "
          "zero, and a table reporting the delta without saying so invites a "
          "conclusion the data cannot support.", "body"),
        P("A note on the time residual", "h2"),
        P("Both codes sit about −39 ns against the truth track time on the "
          "particle-gun samples, and they agree with each other to 0.2 ns. That "
          "is a convention offset between the trigger-relative time each fitter "
          "reports and the creation time truth records — not a bias in either. "
          "So the <i>absolute</i> t residual is not a performance number, while "
          "its spread and the difference between arms are. The report removes "
          "the common pedestal by default.", "note"),
        P("An empty resolution panel", "h2"),
        P("Resolution is profiled in bins, and a bin holding fewer than ten "
          "events is dropped — a median of three events is not a measurement. "
          "On a small run the standard fixed bins can leave nothing, so "
          "Sextant re-bins by equal counts instead and says so under the "
          "figure. A panel that is still empty carries the event count and the "
          "per-bin minimum, so it reads as a statement about the sample rather "
          "than as a broken plot.", "body"),
        P("Ring counting", "h2"),
        P("Only true rings above Cherenkov threshold are counted, and accuracy "
          "is reported both pooled and <b>balanced</b> across true ring counts. "
          "Balanced matters: pooled accuracy is dominated by whichever ring "
          "count the sample is mostly made of, so a fitter that always answers "
          "“1” scores 90% on a one-ring-heavy sample without being right about "
          "anything.", "body"),
    ]

    # ---- 11. extending ---------------------------------------------------
    story += [
        P("11. Extending Sextant", "h1"),
        P("Adding an algorithm", "h2"),
        P("Write one reader that returns the canonical tables, and register a "
          "format probe in <font face='Courier'>io/dataset.py</font>. Every "
          "metric, plot and table then works on it unchanged. "
          "<font face='Courier'>io/coral.py</font> is the shortest example.",
          "body"),
        P("Adding a production or sample", "h2"),
        P("Add an entry to <font face='Courier'>productions.py</font>: the "
          "input directory, a filename pattern with the file index as its one "
          "capture group, the events per file and the numbering stride. If runs "
          "are to be launched on it, mirror the entry in "
          "<font face='Courier'>Analysis/Scripts/cluster/runlib.py</font> so "
          "<font face='Courier'>exp.py</font> and the merge scripts agree.",
          "body"),
        P("Adding a metric or figure", "h2"),
        P("Metrics take canonical tables and return DataFrames; plots take "
          "those DataFrames plus an "
          "<font face='Courier'>ArmStyle</font>. Use the "
          "<font face='Courier'>ArmStyle</font> for colour — it fixes colour, "
          "linestyle and marker to the <i>arm</i>, so identity is consistent "
          "across every figure and adding an arm never repaints the others.",
          "body"),
        P("Tests", "h2"),
        C("pytest Software/Sextant/tests -q"),
        P("The suite pins the invariants whose breakage would otherwise be "
          "silent: stride ≥ events per file, residual sign convention, AUC "
          "under perfect and zero separation, working-point invariance to a "
          "constant offset, balanced ring accuracy, mixed-calibration "
          "detection, and each validation guard.", "body"),
    ]

    # ---- 12. reference ---------------------------------------------------
    story += [
        PageBreak(),
        P("12. Command reference", "h1"),
        table([
            ["Command", "What it does"],
            ["<font face='Courier'>sextant check &lt;spec&gt;</font>",
             "Validate a spec and show how the arms' configurations line up."],
            ["<font face='Courier'>sextant run &lt;spec&gt;</font>",
             "Launch every runnable arm. "
             "<font face='Courier'>--dry-run</font>, "
             "<font face='Courier'>--arm</font>, "
             "<font face='Courier'>--sync</font>, "
             "<font face='Courier'>--force</font>."],
            ["<font face='Courier'>sextant status &lt;spec&gt;</font>",
             "Per-arm progress. <font face='Courier'>--full</font> for the "
             "whole cluster dashboard."],
            ["<font face='Courier'>sextant logs &lt;spec&gt; --arm X</font>",
             "Stream one arm's job output."],
            ["<font face='Courier'>sextant errors &lt;spec&gt;</font>",
             "Filtered failure summary across the arms."],
            ["<font face='Courier'>sextant cancel &lt;spec&gt;</font>",
             "Cancel this experiment's runs — and only those."],
            ["<font face='Courier'>sextant collect &lt;spec&gt;</font>",
             "Merge and pull finished arms."],
            ["<font face='Courier'>sextant report &lt;spec&gt;</font>",
             "Build the report. <font face='Courier'>--collect</font> merges "
             "and pulls first."],
            ["<font face='Courier'>sextant probe &lt;spec&gt;</font>",
             "Per-PMT intermediates at a common parameter point."],
            ["<font face='Courier'>sextant compare</font>",
             "Report from files named on the command line — no spec needed."],
            ["<font face='Courier'>sextant truth</font>",
             "Build a truth file from WCSim input."],
            ["<font face='Courier'>sextant productions</font>",
             "Samples Sextant knows about."],
            ["<font face='Courier'>sextant schema [table]</font>",
             "The canonical tables."],
            ["<font face='Courier'>sextant mapping</font>",
             "Coral ↔ fiTQun parameter equivalence and the calibration schemes."],
        ], [58 * mm, 90 * mm]),
        P("13. Troubleshooting", "h1"),
        table([
            ["Symptom", "Cause and fix"],
            ["“unrecognised reconstruction output”",
             "The file is not a Coral merge, a fiTQun merge, or an official "
             "fiTQun file. A missing or empty file reports as such separately."],
            ["Validation says <i>truth pairing</i> error",
             "The arm is matched against a truth file from a different sample. "
             "Check <font face='Courier'>[experiment] truth</font>."],
            ["“no such reconstruction file”",
             "The arm has not been merged and pulled. Run "
             "<font face='Courier'>sextant collect</font>."],
            ["Report has no PID section",
             "PID needs the same arm on both a true-electron and a true-muon "
             "sample; pass <font face='Courier'>pid_samples</font>."],
            ["Report has no per-PMT section",
             "Run <font face='Courier'>sextant probe</font> first; the fiTQun "
             "side additionally needs "
             "<font face='Courier'>fqprobe</font> built."],
            ["“fqprobe binary not found”",
             "Build it (section 9) and set "
             "<font face='Courier'>SEXTANT_FQPROBE</font>."],
            ["Truth build fails on libWCSimRoot",
             "Run it where ROOT and WCSimRoot exist — on the cluster, after "
             "sourcing <font face='Courier'>env_coral.zsh</font>. Note that "
             "setting <font face='Courier'>PYTHONPATH</font> must "
             "<i>prepend</i>, never overwrite: ROOT lives there."],
            ["Arms disagree on a shared setting",
             "Expected if deliberate — put it in "
             "<font face='Courier'>[shared]</font> to force agreement, or "
             "accept the warning."],
            ["fiTQun jobs never finish",
             "Almost certainly the parameter file. The full "
             "<font face='Courier'>Official_HyperK_2024_*</font> files enable "
             "the multi-ring, sequential multi-ring, pi0 and pi+ fits, which "
             "cost <b>~20 minutes per event</b> on this detector — measured. "
             "For a single-ring comparison use a "
             "<font face='Courier'>*_1Remu</font> file. "
             "<font face='Courier'>sextant check</font> prints a COST line "
             "naming every expensive fit a parameter file switches on."],
        ], [46 * mm, 102 * mm]),
    ]

    doc = Guide(output)
    doc.build(story)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Beside the generator, inside the package: this is documentation that
    # travels with the code, not a report about a particular study.
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent / "Sextant_User_Guide.pdf"))
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    build(path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
