# Sextant

**Validation and comparison of water-Cherenkov event reconstruction for
Hyper-Kamiokande.**

Sextant compares reconstruction algorithms — Coral and fiTQun today, anything
that can be read into its canonical tables tomorrow — against MC truth and
against each other. It measures not only what the algorithms *output* (vertex,
direction, momentum, PID, ring counting) but what they compute *on the way
there*: predicted charge, likelihood components, and the trajectory of the fit
through its stages.

It has two halves that integrate but do not depend on each other:

| Half | Modules | Needs |
|---|---|---|
| **Validation** — residuals, resolutions, PID, ring counting, intermediates, PDF + CSV report | `io`, `metrics`, `plots`, `report`, `analysis` | Python, numpy/pandas/matplotlib/uproot/scipy. Runs on a laptop with no HK software installed. |
| **Experiments** — launching arms, keeping their configs in step, collecting results | `run`, `probe` | The reconstruction workspace, `exp.py`, and (for probing) Coral or a built fiTQun |

The validation half never imports the experiment half. You can point it at three
ROOT files you produced by hand and get the full report.

---

## Install

```bash
pip install -e Software/Sextant          # or: pip install -e .
```

## Two ways in

### 1. Files you already have

```bash
sextant compare \
    --arm "Coral sandbox=Output/Coral/Reco/x/coral_electron.root" \
    --arm "fiTQun=Output/fiTQun/Reco/fitqun_electron.root" \
    --truth Output/Truth/truth_electron.root \
    --outdir Output/Sextant/e_comparison/report
```

or from Python:

```python
from sextant import Arm, Truth, analysis

arms = [Arm.from_file("Coral sandbox", "…/coral_electron.root"),
        Arm.from_file("fiTQun",        "…/fitqun_electron.root")]
truth = Truth.from_file("…/truth_electron.root")
analysis.standard_report(arms, truth, outdir="…/Reports/e_comparison")
```

### 2. A whole experiment from one spec

```bash
sextant check   examples/electron_comparison.ini    # do the arms agree?
sextant run     examples/electron_comparison.ini    # launch every arm
sextant status  examples/electron_comparison.ini    # per-arm progress
sextant errors  examples/electron_comparison.ini    # what failed (greps the
                                                    #   server's logs in place)
sextant logs    examples/electron_comparison.ini --arm coral
sextant report  examples/electron_comparison.ini --collect
```

`status` prints a focused per-arm summary; `--full` gets `exp.py`'s
cluster-wide dashboard instead. Arms that read existing files rather than
running are listed too, so a two-arm experiment never looks like a one-arm one.

`sextant run` does not submit jobs itself: it writes one `exp.py` spec per arm
and calls `exp.py`, which freezes the code, config and parameters into each run
directory. Runs stay immutable and visible to `exp.py status` exactly as before.

---

## The output

```
Output/Sextant/<experiment>/
    specs/*.ini         the exp.py spec generated for each arm
    experiment.json     what was launched
    probe/              per-PMT intermediates, if probed
    report/
        report.pdf      every figure and table, in reading order
        figures/*.png   the same figures individually, for slides
        tables/*.csv    every number in the report, machine-readable
        manifest.json   what was compared, and what produced it
```

Everything one experiment produces lives under `Output/Sextant/<experiment>/`.
`Analysis/Reports/` is deliberately not used: that is for bespoke, authored
write-ups about Coral development, not for generated validation output.

Sections are skipped, with a note, when the inputs cannot support them — a
Coral-only comparison has no per-PMT section; a single-particle sample has no
PID section — so one command works for every combination.

---

## What it measures

The report covers the HK reconstruction validation spec (roadmap R1): for
single- and multiple-track events, sub-eventing efficiency and purity for the
prompt and secondary tracks, vertex / direction / momentum resolution and bias,
and PID efficiency, each also against true momentum, `dwall` and `towall`.
Every binned figure has its numbers in `tables/*.csv`.

**Outputs.** Residuals (reconstructed − true) in t, x, y, z, p; vertex distance,
opening angle, fractional momentum residual; resolution and bias profiled
against true momentum, `dwall` and `towall`; per-event paired differences
against a reference arm with bootstrap confidence intervals.

**PID.** The dnLL = nLL(μ) − nLL(e) discriminant, ROC, AUC, and muon efficiency
at a *fixed* electron efficiency — the comparison that survives a constant
offset between two codes' discriminants, which "fraction correct at dnLL > 0"
does not — and the identification efficiency against p, dwall and towall. PID
needs a true-e and a true-μ sample; an experiment spec names the other one with
`[experiment] pid_partner = <spec>` (arms pair by arm name), and `compare` takes
`--pid-arm LABEL=PATH --pid-truth <truth>`. Scored on the events every arm has.

**Sub-eventing.** Reconstructed sub-events are matched one to one in time
(±50 ns) to the true ones — the prompt interaction and every decay electron —
giving prompt and decay efficiency, prompt and delayed purity, and the
exact-count rate, with Wilson 68% intervals, against p, dwall and towall. Only
sub-events that fired a WCSim trigger count (a decay that fired none left no
hits). Needs sub-event truth, built from WCSim (below).

**Multiple-track events.** On events with two or more visible tracks: the fit's
vertex against the true interaction vertex, and the leading reconstructed
ring's direction, momentum and type against the most energetic visible particle
(by total energy), with the single-track statistics and profiles.

**Multi-ring.** Ring multiplicity, exact-count accuracy split by true ring count
(and balanced across them), fake and missed rings per event, and per-slot
angular error, momentum bias and type accuracy after a minimum-angle 1-1
assignment to the true rings.

**Intermediates** — the part that says *why*:

1. **Charge normalisation** — total predicted photoelectrons over total observed
   charge, per hypothesis. Needs nothing but the standard outputs.
2. **Likelihood decomposition** — Coral's unhit / phit / cpdf / tpdf split, and
   each code's improvement over its own zero-ring hypothesis. Absolute NLLs are
   never compared across codes; the zero-ring difference is what cancels their
   constant offsets.
3. **Fit trajectory** — residuals after each fit stage, for both codes.
4. **Per-PMT predictions** — predicted direct and scattered light PMT by PMT.

### Why the parameter point matters

Predicted charges compared at each code's *own* fitted vertex differ mostly
because the vertices differ. Evaluated at a **common** point — MC truth, or one
arm's fit — what remains is the model difference, which is the question actually
being asked. Every intermediate table carries a `point` column and the
comparison refuses to join across different points.

Per-PMT probing uses two additive tools, neither of which changes any fitting
path:

* Coral: `sextant.probe.coral_probe` calls Coral's own
  `compute_predicted_photons` and `nll_term_sums`.
* fiTQun: `cpp/fqprobe`, a standalone binary linking fiTQun's public library
  and calling `Get1Rmudist`.

```bash
cmake -S cpp/fqprobe -B build/fqprobe -DFITQUN_DIR=<fiTQun install>
cmake --build build/fqprobe
export SEXTANT_FQPROBE=$PWD/build/fqprobe/fqprobe
```

(fiTQun's own `OutputQTdist` was the obvious alternative and is unusable here:
the call that fills it is inside `#ifndef NOSKLIBRARIES`, which the HyperK build
defines, and it bins in (cos θ, time residual) rather than reporting per PMT.)

---

## Keeping two configurations honest

`sextant check` compares the arms setting by setting, using a mapping of
*concepts* to each code's own key names (`sextant mapping` prints it).

Most settings are genuinely shared and can be forced to match from one place:

```ini
[shared]
use_time_likelihood = true
do_muon_fit         = true
rccut_a0            = 530.
```

A few cannot. Photon yield, water attenuation length and the scattered-light
treatment are fitted *together* against observed light, so they form a
self-consistent **calibration scheme** whose members must not be mixed — fiTQun
itself ships them as matched pairs (`wscat`: 0.221 / 7000 / on, `noscat`:
0.240 / 7500 / off), and Coral's production tuning is a third (0.176 / 9500 /
time-PDF only). Copying one code's quantum efficiency onto the other while
leaving its attenuation length reintroduces a large momentum bias — a "matched"
configuration that measures nothing but the mismatch.

So each arm *declares* a scheme, Sextant verifies the arm is internally
consistent with it, records both in the report, and refuses to interleave them:

```ini
[arm.coral_sandbox]
algorithm = coral
scheme    = coral-production

[arm.fitqun]
algorithm = fitqun
scheme    = fitqun-hk2024-wscat
```

---

## Samples

`sextant productions` lists them. Alongside the particle guns, Sextant supports
the **NEUT beam production** (`beam-fhc`, four flavour channels), which carries
full neutrino final states and 1–2 triggers per event — and ships an **official
fiTQun 6.4.4 reconstruction** that Sextant reads directly as a reference arm,
with no cluster time spent at all:

```ini
[arm.fitqun_official]
algorithm = fitqun
source    = official
```

Truth for a beam sample is built with:

```bash
sextant truth --production beam-fhc --sample numu_x_numu \
              --output Output/Truth/truth_beam_numu.root --events 2000
```

True rings are the primary final-state particles (`flag == 0`,
`parenttype == 0`), with the outgoing lepton flagged, `visible` set for those
above Cherenkov threshold, and a primary π⁰ expanded into its two decay photons.
Per-event labels record CC/NC, neutrino flavour and energy, NEUT interaction
mode and the true ring count. Sub-threshold neutrals are kept with
`visible = False` rather than dropped, so "what did the fitter miss" can be
asked of the full final state while every ring metric filters them out.

**Sub-event truth.** `sextant truth` also records every decay electron (an e±
created where its parent μ± stopped, p > 1 MeV/c) and whether a trigger window
covers it. For a truth file that predates this, write a sidecar that
`Truth.from_file` picks up automatically:

```bash
sextant truth --production v1.12.29 --sample muon --start 80 --max-files 20 \
              --subevents-only --output Output/Truth/truth_muon_v1.12.29_subevents.root
```

> **Event numbering.** Global event id = `file_index * stride + entry`. The
> particle guns hold 100 events per file and use a stride of 100; the beam
> production holds **101**, so it uses 1000. A stride equal to the events per
> file would alias each file's last event onto the next file's first, silently.
> This is enforced by a test.

---

## Guards

The expensive mistakes in this domain are not wrong formulas, they are correctly
computed comparisons between unrelated things — and they produce a full set of
plausible plots. Every report opens with a validation table:

* **truth pairing** — a median vertex residual larger than the detector means
  the arm is matched against the wrong truth sample (this catches what event-id
  overlap does not: two samples can share id ranges and still be unrelated);
* **event overlap** — no shared ids, or a low overlap fraction;
* **duplicate keys** — anything that would make a join ambiguous;
* **time pedestal** — arms disagreeing on the t0 convention (fiTQun reports
  absolute WCSim time, Coral trigger-relative; both sit ~−39 ns against the
  truth track time, which is a convention, not a bias — so the *absolute* t
  residual is not a performance number while its spread and inter-arm
  differences are);
* **success rate** — an arm most of whose fits failed.

---

## Layout

```
src/sextant/
    io/          readers -> canonical tables (schema.py defines them)
    metrics/     residuals, resolution, pid, rings, subevents, multitrack,
                 intermediates, stats, validate
    plots/       figures; style.py fixes colour to the arm, never its position
    report/      PDF + CSV assembly
    run/         spec, launcher (drives exp.py), monitor, collect
    probe/       parameter points, Coral probe, fqprobe driver
    config/      Coral <-> fiTQun parameter mapping and calibration schemes
cpp/fqprobe/     standalone fiTQun per-PMT probe
examples/        experiment specs
tests/           invariants whose breakage would otherwise be silent
```

Everything downstream of `io` is algorithm-agnostic: it reads the six
canonical tables (`sextant schema`) and never asks which code produced a
number. Adding a third algorithm means writing one reader.

## Documentation

A fuller guide with worked tutorials is generated from the package:

```bash
python Software/Sextant/docs/make_guide.py   # -> docs/Sextant_User_Guide.pdf
```

## Tests

```bash
pytest Software/Sextant/tests -q
```

The suite pins the invariants whose breakage would otherwise be silent — the
event-id stride, the residual sign convention, PID working-point invariance,
balanced ring accuracy, mixed-calibration detection, every validation guard, and
the regressions found by running the whole thing on the cluster.
