# R54: working notes (moved verbatim from roadmap v8 on 2026-10-22; Claude's file, the roadmap keeps a summary)

**18 · R54 [H]. Learned post-fit estimator: both hypotheses vote — M** (accuracy in every bin; the lever for R28)
- Every event already has an e- and a mu- fit (PID). Their disagreement (p ratio, vertex shift, nLL difference) + fitted
  geometry + observed totals as ratios to p predict each fit's signed error (gradient boosting; no p magnitude input).
- Feasibility 10-20 (ledger Momentum 10-20, `Analysis/Archive/2026-11_tidy/Scripts/post_fit_estimator_cv.py`, 5-fold by file, sum10k):
  e- p core 3.38 -> 2.93% (fiTQun 3.12 same events), mu- 3.82 -> 2.50%, e- vertex -11% (transverse -27%), mu- vertex
  -13%, direction -3%; no gain at true towall < 6 m. Replaces the vertex-bias tables and their re-derivation.
- PASS (pre-registered 10-21): trained on files 0-299, frozen, applied to files 300-699 (sum10k) paired vs production:
  e- p core <= 0.90x production (CI), no (p, towall) bin of any width worse beyond CI, per-bin median bias not worse,
  PID untouched; then beam 1R events and one WCSim variation (R32) not worse than production.
- 10-21 run (operational, written before any result): fresh production runs on consolidate a7acac3, `r54_train_{e,mu}`
  (files 0-299 x 25) and `r54_test_{e,mu}` (files 300-699 x 25); `r54_heldout.py` trains once, freezes, applies. Strict
  reading: e- p-core ratio CI upper bound <= 0.90; bins = true p x true towall (< 3, 3-6, 6-15, > 15 m), >= 40
  events; widths p core / vertex p68 / angle p68 and |median p bias| per bin, paired bootstrap; PID untouched by
  construction (nLL unchanged).
- 10-21 v1 RESULT: FAIL (strict). e- p core 3.39 -> 2.63% (x0.77 [0.75,0.80]), vertex -11%, angle -3.5%; mu- p 3.80 ->
  2.54% (x0.67), vertex -12%; but bins worse beyond CI, all near the wall: e- |bias| at true towall < 3 m, p < 400; mu-
  p core x4.6 and |bias| up to 34% at towall < 3 m (+ two marginal mu- low-p bins). v2 (pre-reg): correction applied
  only where FITTED towall >= T, T = the smallest of {0,1,2,3,4,6} m with no (p, towall) bin worse beyond CI in 5-fold
  CV (by file) on files 0-299, per hypothesis; then the same PASS on FRESH files 700-999 (`r54_fresh_{e,mu}`), since
  300-699 has been seen.
- 10-21 v2 RESULT on fresh 700-999: FAIL by one bin per sample (ledger Momentum 10-21), otherwise every headline
  passes: e- p core x0.775 [0.747,0.806] (3.43 -> 2.65%, fiTQun-level), vertex -10%, angle -2%; mu- p x0.69, vertex
  -11%; the failing bins are near-wall |bias| (e- p 200-400 towall < 3 m; mu- p > 1.5 GeV towall 3-6 m), consistent
  with chance over ~96 CIs. ADOPTED 10-21 (bcarew): consolidate 457d5bc + abf70f6, default on in config / single_ring /
  full; in-code run reproduces offline exactly (ledger). Next: beam 1R events and one WCSim variation (R32) not worse;
  retrain whenever the fit changes (R56 etc.); Sextant tracker snapshot.

## 10-23 four-arm comparison on the same events (ledger Momentum 10-23)
e- (6640): p core A 3.45 / B 2.66 / C 2.94 / D 2.67%; R54 -22.9% before the table, -9.3% after; vertex -10%, transverse -23%
either way. mu- (6738): R54 -31 / -33% p core either way. Production (D) vs old base (A): e- p core -22.6%, mu- -32.4%.
Watch: e- true towall 1-3 m, the table's core +28% [+5,+50] with iqr68 -18% (non-Gaussian region).
