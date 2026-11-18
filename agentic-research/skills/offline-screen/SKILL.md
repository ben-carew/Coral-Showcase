---
name: offline-screen
description: Test an idea offline before any cluster physics run - on existing dumps, or by hooking production main() without running fits (the r75_seed_screen pattern) - with a pre-registered GO bar. Use for every idea whose first question can be answered without a fit campaign: seeds, gates, stop rules, minimisers, pdf closure, ring acceptance.
---

# Offline screen (step 0)

A screen answers one question in minutes and decides whether the cluster test is worth running. Write the GO bar in
`items/Rxx.md` BEFORE looking; a screen that fails its bar ends the idea or is reported as a labelled lead.

## A. On data that already exists (laptop, seconds)
| question | data | precedent |
|---|---|---|
| time pdf vs hits, per brightness | `Output/Coral/Reco/r28_tcal/*hit_times.csv` (2.07M e- hits at the TRUE track: tres, nd, mean, sigma, w, q), `r27_e/*hits_hit_times.csv` (fitted point) | audit 11-13 (R76), `r28_hit_times.py`, `r46_step0.py` |
| what each fit step buys | `r27_e/coral_electron_base_fit_trace.csv` (every evaluation, 480 e-), `r68_s0_{e,mu}` (per-step nLL, time, evaluations, both hypotheses) | `r40_worldline_trace.py`, `r68_step0.py` |
| refit / gate replay | `r11_refits_{e,mu}`, `r72_val` per-fit dumps | `r11_refit_replay.py`, `r72_rival_replay.py` |
| ring acceptance | open ladders `r63_s0_{nue,numu}`, `r4_open_*` | `r63_step0.py`, `r2_rccut_beam.py` |
| charge pattern at truth | `r47_maps_{e,mu}` | `r47_step0.py`, `r38_mu_table.py` |
| PID in any cut form, by p / dwall / containment; A/B re-scores | `tn_val_{e,mu}` Coral + fiTQun; any paired run's `<exp>_{e,mu}` | audit 11-14: `pid_cut_forms.py`, `pid_anatomy.py`, `pid_rescore.py <e_dir> <mu_dir> <A> <B> <truth>`, `pid_exiting.py` |
| resolution vs anything, 100k events | `Output/Coral/Reco/tn_val_{e,mu}` + `Output/Truth/truth_{elneg,muneg}-r1_v1.12.29.root` | audit 11-13 (axis check) |
Truth join: event = file x stride + tree ENTRY for the truth (`sextant.io.truth.read`, primary = `is_primary_lepton`),
for fiTQun merges and for Coral merges made since 11-15 (they carry a `numbering` tree). An OLDER Coral merge is
numbered by WCSim EvtNum and mis-joins every event after a skipped number (about 1% of gun files; items/R89): run
`Analysis/Scripts/evtnum/coral_renumber.py --production .. --sample .. <file>` first, or map ids with
`evtnum_gap_map.gap_map`. Every affected LOCAL merge was converted on 11-15 (originals in
`Output/Coral/Archive/<exp>_evtnum/`); a file pulled again from the server is EvtNum-numbered unless its run is
re-merged. Every truth file is entry-numbered. Silence warnings (`python -W ignore ... 2>/dev/null`).
**fiTQun merges made before 11-14 carry ONE momentum and vertex per event: the sample's own hypothesis**, labelled on
both hypothesis rows by Sextant; newer ones have `p_e` / `p_mu` (`fitqun_pe.momenta` picks the right one; tn_val is
re-merged). The fiTQun `stages` tree is the SEED of each fit, not a fit.
Before believing "code A gains from variable X, code B does not", check X means the same thing in both files.

## B. Hooking production main() without fits (server, one small job)
Pattern: `Analysis/Scripts/studies/speed/r75_seed_screen.py`. Import the production modules from a FROZEN run's `code/src`
(PYTHONPATH), monkeypatch `fitqun.fitqun.fit_single_ring` (or the function under study) to compute the candidates and
write one JSON line per event, then call production `main()` with the frozen config: reader, prefit, sub-eventing and
hit selection are production's by construction.
- One job per input file, 1 CPU, node-local numba cache (copy the header of `cluster/prof_job.sh`).
- Count likelihood evaluations and seconds per candidate: the screen also prices the idea.
- The sub-event prefit returns p = 0: seed p = q / conversion factor yourself (R75's first run was void for this).
- Rebuilding a run's hit set by hand is not possible (method-traps): always go through `main()`.

## C. Capturing likelihoods for minimiser tests
R71 C / D pattern: hook `Do1RingFit`, keep each step's `(nll callable, seed, bounds, production result)`, then run the
candidate minimiser on the same callable in the same process. Report evaluations to reach production's minimum
+ 0.1 nLL and the fraction of fits ending > 0.1 / 0.5 nLL above it.

## Reporting
Ledger row with the verdict words GO / NO-GO / LEAD and the bar as written. Post hoc findings are labelled post hoc.
Use files that the later cluster test will NOT use (R75: screen on 960-999, test on 900-939).
