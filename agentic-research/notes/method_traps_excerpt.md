# Method traps — instrument failures and the rules from them

Every one of these produced a CONFIDENT, PLAUSIBLE, WRONG number. None crashed.

## Index by what you are about to do (added 11-15: two recorded traps were walked into again because nobody re-read 600 lines)
Search the quoted words to reach the entry. [T] = now enforced by a tool, so it fails loudly instead of quietly.
- **Join Coral or fiTQun to truth, quote a failure / tail rate**: "event id is not truth's" [T: merges renumber,
  Sextant `event-numbering`], "File/event bookkeeping", "WRONG-PRODUCTION", "`electron` gun sample is POSITRONS",
  "Truth's hit count", "Sample names vs particles", "Partial ROOT files".
- **Write or copy a Sextant spec**: "`scheme` OVERRIDES" [T: refused unless `scheme_override`], "[shared] fit flags
  override" [T: printed at launch], "arm's `branch` is not what runs", "`branch =` does not switch the slot",
  "exit 0 having launched only some arms", "`timeout` around `sextant run`" [T], "mu-only run crashes on decay electrons", "Auto packing breaks", "`events_per_job = auto` changes WHICH events",
  "events_per_job < 100", "Held-out files get spent", "frozen configs have no inline comments".
- **Compare with fiTQun**: "ONE momentum per event" [T: merges carry both, old ones flagged], "staged fits are not in
  sub-event order" (and they are seeds: audit B25), "picks the 1R hypothesis by sample-name prefix",
  "dnLL between hypotheses", "headline ratio quoted on one width statistic".
- **Decide a verdict**: "No metric worse beyond its 95% CI", "comparison table is not the verdict statistic",
  "binned Gaussian-core width", "PID efficiency at 99% e-eff needs the CUT's uncertainty", "PID gain on the flat
  dnLL cut", "raw vertices when production applies a bias correction", "Co-tuned constants", "uncalibrated arm's
  whole-spectrum width", "null that kills your own hypothesis", "number that moves every time you tighten".
- **Change the hit set or the gate**: "absolute time gate is a distance cut", "Rebuilding a run's hit set".
- **Build an offline diagnostic at the true track or on a hit set**: "Diagnostic vs fit mismatch", "Rebuilding a
  run's hit set" (twice), "table looked up at the wrong argument convention", "`totmu/totq` spread",
  "truncated tail-only fit range", "control can fail because the QUESTION is wrong", "covariance share",
  "Edge-truncated per-event statistics", "Cutting in absolute units", "Binning by a quantity computed from the
  fitted value", "post-fit regression on p_fit".
- **Time anything**: "LOCAL runs used a stale config" [T: `local_config.py`], "Same-node timing needs an exclusive
  node" (a time bar is judged by the `bench` skill only; walked into again 10-31), "first event can be a JIT compile", "not bit-reproducible across CPU classes", "scipy BFGS ignores `eps`".
- **Run tooling**: "Silent-drop bugs", "half-written event", "tuning file overridden by path", "JAX persistent
  cache", "uproot 5.7", "pandas attribute collisions", "report run ON the server", "straggler-cut events as misses".

## Diagnostic vs fit mismatch (hit TWICE in one day)
## Rebuilding a run's hit set by hand (the third control failure, CAUSE FOUND)

`bias_attribution.py` rebuilt each event's sub-event with
`subevent_hit_splitting(hits, [fitted_t], ..., gate)`. Measured against the run's
OWN `nll_terms` dump: **my totq was 1.77x the run's** (median 6766 against 4877).
The likelihood was being evaluated on a 77%-larger hit set than the fit ever saw.

Coral's sub-eventing works from the PREFIT, can split an event into several
sub-events, and gates on the SUB-EVENT time -- not on the final fitted t. So
rebuilding it from the fitted parameters does not reproduce it.

**ACCEPTANCE TEST for any truth-geometry diagnostic: its totq must match the
run's dumped totq** (turn on `[in_out] dump_nll_terms` and compare). Cheap,
decisive, and it would have caught this before three failed attribution runs.

**Rule: a truth-geometry diagnostic must load the run's OWN frozen config**
(`Jobs/<exp>/<arm>/config.ini` from the server), never a config that merely looks
similar. And it must carry a CONTROL that reproduces the delivered answer, with
an explicit PASS/FAIL, before any row of it is read.

