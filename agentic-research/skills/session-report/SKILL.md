---
name: session-report
description: Write (or update) the session's report as a local Jupyter notebook in the Analysis repo - a complete, walked-through explanation of everything the session did and why (background, the idea in simple terms, the code inline where it is explained, plots that motivate each decision and plots of its result), in bcarew's Terraformer presentation theme. Only when bcarew explicitly asks (/session-report).
disable-model-invocation: true
---

# Session report

The report is bcarew's complete record of the session: reading it top to bottom must tell him
what was done, why, how it works and what it showed, **without the transcript**. It is NOT a
summary (bcarew 11-18: the outcome-bullets + captioned-plots + code-diff format was rejected).
Write it as an explanation a careful colleague would give: prose that walks him through the
work in the order it happened, with the reasoning at every step.

It is a **local notebook**, not a Claude artifact: bcarew edits the plots, sees the code that made
every number, and needs nothing from his Claude account to open it.

## What it must contain (all of it, for every piece of work in the session)

1. **Background and context.** What problem the session addressed, why it matters for Coral / HK,
   what existed before the session (earlier attempts, their numbers, why they fell short) and
   what was open. bcarew knows HK, Cherenkov detectors and Coral in general: do not explain those,
   but DO explain the specific mechanism, quantity or method the work is about.
2. **The idea, explained simply.** What is being computed, physically or geometrically, in plain
   words first and then precisely (the definition, the formula, the algorithm step by step). Use an
   illustrative figure (a single event, a schematic plot, the shape of a distribution) whenever it
   makes the idea easier to see. Say which alternatives were considered and why they were not used.
3. **The code, inline.** Show the real code (short excerpts from the commit, in markdown code
   blocks) at the point in the text where that step is explained: the function where the quantity
   is defined, the call site where it is wired in, the output change. Explain what each excerpt
   does. A final "code change" section then only lists commits, verification and how to revert.
4. **Why each decision was made, with evidence.** For every decision (including decisions NOT to do
   something, parked items and withdrawn claims): the question, the options, the pre-committed
   criterion if there was one, and a plot or table that shows WHY that choice — not only the
   result after it. Decisions taken by bcarew are labelled as his.
5. **How it was tested.** Sample, files (held out or not), n, how the run was made (offline
   screen, cluster run, smoke test), what was compared, the statistic and its uncertainty.
6. **Results**, with figures of the result of the change, and their interpretation: what the plot
   shows, what is good, what is not, the limitations and the regions where it fails.
7. **Roadmap and next steps.** Items touched (old -> new status, one-line reason) and what follows.

Start with a short outcome block (3-5 bullets: what is now true, headline numbers with 95% CI and n,
shipped / measured / refuted / parked), then the full narrative.

## Writing rules

- **Prose, not fragments.** Each section is paragraphs of explanation; bullets only for genuine
  lists. A figure caption never replaces the text: say before a figure what to look for, and after
  it what it shows and what follows from it.
- **Explain simply first, then precisely.** Define every quantity before using it; spell out units,
  sign conventions and what "good" looks like.
- **Numbers come from the code.** Headline numbers are printed by code cells (`display(Markdown(f...))`
  or a printed table); narrative text may quote a number only if a cell above it prints it. Never
  type numbers from memory or from the transcript.
- **Complete, not padded.** Cover everything the session did, including failed attempts, crashed or
  re-submitted runs that changed what was measured, and anything that remains unverified (say so).
  Leave out tool chatter that changed nothing (ssh retries, typos).
- Figures: typically 5-10 — motivation / mechanism figures as well as result figures.

## Where it goes

```
Analysis/Reports/Sessions/<YYYY-MM-DD>_<slug>/
    report.ipynb      the report: markdown + the code of every table and figure, executed
    data/             small extracted tables (CSV / parquet, < ~20 MB total) the notebook reads
    data/extract.py   how data/ was made from Output/ or the server (run once, not by the notebook)
    figs/             <name>.svg (slides) .pdf (Keynote, papers) .png, written by the notebook
```

- Re-running the notebook must take **seconds on the laptop** and must not need the server: heavy
  work (fits, merges, bootstraps over large files) runs on the server and lands in `data/` as
  per-event or per-bin tables. The notebook computes the reported numbers and CIs from `data/`.
- If the session already has a report, UPDATE that folder (rewrite it); never a second report for one session.
- Commit the folder in the Analysis repo (`git -C Analysis add ... && git commit`), one-line subject.

## Notebook skeleton

1. **Title cell** — `# <claim-style title>`, date, one paragraph of scope (samples, n, held out or not,
   code version).
2. **Setup cell** — imports, `bcplot`, paths (`HERE = Path.cwd()`, `DATA`, `FIGS`), a `PROV` string
   (Coral commit / sample / n) used in every figure's stamp. No hidden state between cells.
3. **Outcome** — printed from code.
4. **Background and context.**
5. **One section per piece of work, in the order done** — question, idea (simple, then precise),
   alternatives, code inline, criterion, test, results (figures + text), interpretation, decision.
6. **Decisions not tied to one section** (parked items, scope decisions), with the evidence.
7. **Code change** — commits + branch, files touched, verification (smoke arms, bit-identity,
   replay), cost if measured (or "not timed"), how to revert.
8. **Roadmap and next steps.**

## Figures: the Terraformer theme

Every figure uses bcarew's design kit so it drops into his slides unchanged
(`/Users/bcarew/Documents/PhD/Design/README.md`, `python/bcplot`):

```python
import sys; sys.path.insert(0, "/Users/bcarew/Documents/PhD/Design/python")
import bcplot as bp
bp.use("terraformer")                    # slide context: sizes in slide-grid cells
fig, ax = bp.figure("half")              # or (cols, rows) of the 12 x 6 slide grid
...
bp.legend(ax); bp.tag(ax, r"$\mathrm{e}^{-}$ · held out"); bp.note(fig, PROV)
bp.save(fig, FIGS / "name")              # svg + pdf + png
fig                                      # LAST line of the cell: renders the plot in the notebook
```

- **Every figure cell ends with a bare `fig`** (`bp.save` closes the figure, so without it nothing
  renders in the notebook). A cell that also shows a table uses `display(table)` before it, so
  `fig` stays the last line. After executing, check that every figure cell has an image output.

- **Size = the slide box it will fill** (`"half"`, `"wide"`, `"third"`, or `(cols, rows)`), so plot
  text is slide text at 1:1. Never rescale afterwards.
- **Series order is fixed and means something**: series 0 = sky-teal accent (ours / the point, filled, solid),
  1 = molten orange (the reference: fiTQun, previous result; open, dashed), 2 = continent green, 3 = crust blue. For bars use
  `bp.series(i)["color"]`; give the reference an open (white-filled, orange-edged) bar. Bounds and theory
  in ink (`bp.bound`, or `color=bp.INK, marker=""`). Never colour alone: markers and line styles differ too.
- No figure titles (the slide title carries the claim); `bp.tag` for panel labels, `bp.legend` above
  the axes, `bp.note` provenance stamp on every figure. Unicode superscripts (⁻, ⁺) are missing from
  the theme font: use mathtext (`$\mu^{-}$`) in every figure string.
- Real terminology on axes (`$d_{\mathrm{wall}}$ [cm]`, `68% vertex error [cm]`); the error-bar
  definition and n in the text around the figure.
- **Figures are shareable as they are** (bcarew shows them to his supervisor): standard technical
  language only. Never roadmap ids (`R54`), run / spec / arm names, branch or commit names (except in
  the provenance stamp), internal nicknames. The notebook TEXT is for bcarew and may name runs,
  items and commits where that helps him find things.

## Build and check

1. Write `data/extract.py` and run it (server work through the usual job scripts; pull the small
   tables back). Generate from the session's own `Output/` files, never from remembered numbers.
2. Write the notebook with `nbformat` (or edit cells), kernel `hyperk`, then execute in place:
   `~/.conda/envs/HyperK/bin/python -m jupyter nbconvert --to notebook --execute --inplace
   --ExecutePreprocessor.kernel_name=hyperk report.ipynb`. Fix every error and warning first.
3. Check every figure cell has an image output (plots render inline: `fig` last), then look at each
   PNG once; fix all faults in one pass (overlaps, clipped labels, wrong order/colour).
4. Re-read the markdown against the printed outputs: every number quoted in text must appear in an
   output above it, and every claim about a figure must be visible in it.
5. Commit; tell bcarew the folder path in one line and what is shippable.

## Traps

- Writing a summary instead of an explanation (bullets + captions + diff): rejected 11-18.
- Figure cells that only save: `bp.save` closes the figure, so the notebook shows no plot unless the
  cell ends with `fig` (11-18).
- Numbers from conversation memory instead of `Output/` files; numbers typed into markdown that the
  code does not print; claims about a figure that the figure does not show.
- A notebook that reads `/sps` or `Output/Coral/Jobs/...` directly (breaks off the server, slow).
- Bundling independent improvements into one headline.
- Presenting a diagnostic (offline, truth-geometry) result as a delivered one.
- A p_fit-indexed or fit-binned statistic presented as a mechanism (method-traps.md).
- Speed numbers from production job wall times presented as a benchmark: label them as such, or use
  the `bench` skill.
