---
name: experiment
description: Run any Coral or fiTQun test, scan, smoke test or benchmark end-to-end - Sextant spec, cluster submission via exp.py, liveness check, wait, merge/pull, paired comparison, and recording the result in the roadmap/ledger. Use whenever a change needs measuring, a config needs comparing, or reference outputs need regenerating.
---

# Experiment loop

Every test serves a roadmap item. Before anything runs, write its pass/kill criterion
into `.claude/Project Notes/items/Rxx.md` (never the roadmap: it is bcarew's summary), and grep
`ledger.md` for the idea — do not re-run what is already closed. If the first question can be
answered without fits, do that first (skill `offline-screen`). Speed claims: skill `bench`.

**The criterion (audit 11-13 proposal, method-traps; bcarew may overrule per item):** name ONE primary metric and its bar, judged on its
95% CI. Every other metric is a guard that fails only beyond its 99% CI. "No metric worse beyond
95%" over ~14 intervals rejects half of the neutral changes. A model change (anything that alters
predicted charge or time) is judged only after the light table and vertex curve are re-derived for
it, with the post-fit estimator off.

## 1. Spec — Sextant, one file per experiment

`Analysis/Experiments/<exp>.ini` (copy a recent one, e.g. `r0_towall_smoke.ini`):

```ini
[experiment]
name = <exp>          ; production = v1.12.29 ; sample = electron|muon ; events = N
where = server        ; hypothesis = e-|mu- ; reference = <arm> ; truth = Output/Truth/truth_<sample>_v1.12.29.root
[run]
start = 80            ; first input FILE: calibration on 0-59/0-29, validation on 80-99
                      ; events_per_job: omit (= auto) unless reproducing an established sample
[shared]              ; concepts from sextant.config.mapping, translated per code
use_time_likelihood = true
do_electron_fit = true
do_muon_fit = true
[arm.<name>]
algorithm = coral     ; slot = b ; branch = ben/dev ; scheme = config   (config_file = configs/<preset>.ini for a preset)
config.<section>.<key> = value     ; native Coral override (nested sections are fine)
```
**`scheme = config` is the rule** (also the default since 11-14): the arm runs its branch's config.ini / preset as
committed. A named scheme (`coral-hitscat-2026-10`, ...) writes ITS constants over the config; Sextant now stops
when that would change a value the config sets (`scheme ... would CHANGE calibration constants`). Only an experiment
whose subject IS those constants (a yield scan, a measurement scheme) names a scheme, with `scheme_override = true`.
Why: `coral-hitscat-2026-10` over `configs/multiring.ini` ran the direct-only multi-ring preset with hit-scattered
light in every multi-ring run from 10-21 to 10-27 (audit B26); a branch that re-derives its yield (R76, R86) would
have its new constants silently reset the same way.
fiTQun arms: `algorithm = fitqun`, `params = Software/fiTQun/ParameterOverrideFiles/...`,
`param.<Name> = v`. To read existing outputs instead of running: `source = file`, `path = ...`.

**Packing is automatic.** `events_per_job = auto` (default) sizes jobs for the shortest
wall time from measured s/event (newest Timing CSVs, p75) and merge cost
(`Output/<Algo>/Timing/merge_log.csv`): N ≈ sqrt(1.35·E·s/(pace+merge)) (1.35 = measured p95 job/median). Preview with
`./exp.py pack --name x --where server --algo coral --particle muon -n 2000 --production v1.12.29`.
A slower config says so with `[run] s_per_event = 30`. An explicit `events_per_job` is
kept; it also decides WHICH events run (one input file per job: files
`start .. start+jobs-1`, first `events_per_job` events of each — exp.py prints the range).
So: every arm of a paired comparison must share it (Sextant pins it in
`Output/Sextant/<exp>/packing.json`); to pair with an EXISTING run (e.g. `cons_val_*`,
files 80-99 × 100) reuse that run's value; and check the printed file range stays clear
of the calibration files (0-59).

**Job shape (adopted 10-29, R71).** Coral server jobs default to **1 CPU (1 numba thread),
`--mem 3G`**: whole fits in parallel give 1.4x more events per core-hour than 4 threads per
fit (full-node benchmark), and peak RSS is 2.7 GB since c6da583 on the production branch. Leave it.
Overrides in the spec: `[server] cpus = N`, `[server] mem = 5G` (also per arm as
`server.cpus` / `server.mem`). GPU arms (`server.gpus`) default to 4 CPUs / 4G (max 5 CPUs
per V100). A branch WITHOUT c6da583 peaks at 4.3 GB: exp.py detects it (`[MEM]` line) and
submits 5G; rebase old branches on ben/dev rather than relying on it. Timing CSVs carry a `cpus`
column; the packer rescales learnt rates to the job's CPUs as core-seconds (rows without
it = 4 threads), conservative by ~1.4x for 1-CPU jobs. Compare timings only at equal
CPUs, and quote core-seconds per event when shapes differ.

**Limits.** ≤ 1000 jobs per submission (exp.py refuses more); prefer a few hundred
(/sps rules in CLAUDE.md). Server runs always cut stragglers (`straggler_cut = auto`:
5%, 10% at ≥ 100 jobs). Server timing is preliminary; real timing runs are local, uncut.
A job-layout or memory change is timed on a WHOLE exclusive node (4 cores of an idle node
overstated R71's gain: 1.6-2.0x there, 1.4x loaded).

## 2. Launch and verify

```
sextant check Analysis/Experiments/<exp>.ini          # arms agree where they should
sextant run   Analysis/Experiments/<exp>.ini --sync    # pushes the slot, submits every arm
```
`--sync` is required after any commit (exp.py refuses when server code differs). Then
confirm the frozen code: `manifest.json` `code.sha` of each run = your commit, clean.
Then diff the arms' FROZEN configs (`ssh ... 'diff .../Jobs/<exp>/<armA>/config.ini .../<armB>/config.ini'`):
the diff must be exactly the switch under test. Frozen configs have no inline comments, so a
mistyped key passes silently; R74's first bench was a dead switch found only this way.

**Within ~3 minutes, verify the run is alive**: `./exp.py logs coral:<exp>/<arm>` or grep
`logs/*.err` for `Traceback|Error`. A crashed Coral job still leaves a valid-looking EMPTY
ROOT file (it is written before fitting); a 400-job batch was once lost this way.

**Code changes: smoke first.** 2 arms × ~4 events (`events = 4`, `events_per_job = 4`),
then check the arms DIFFER where they should and are bit-identical where they should
not (e.g. `r0_towall_smoke`: e- identical, mu- only x,y,z moved). Identical numbers =
dead switch (`do_subeventing`, `indirect_unhit` precedents). Bit-identity only holds on
the same CPU class: Coral differs by ~0.3% per event across CC-IN2P3 node types — check
`sacct -j <ids> -o JobID,NodeList` before reading a difference (or its absence).

## 3. Wait without spending tokens

One background Bash call, not repeated polling (zsh-safe; `grep -c` exits 1 on 0 matches,
so never chain it with `||`):
```
for i in $(seq 1 60); do n=$(timeout 60 ssh -o BatchMode=yes cca.in2p3.fr "q=\$(squeue -u bcarew -h -o %o) || exit 1; echo \"\$q\" | grep -c '/Jobs/<exp>/'" | tr -d ' '); [ "$n" = "0" ] && { echo done; exit 0; }; sleep 30; done; echo timeout
```
Match the run by its job script path (`%o` holds `.../Jobs/<exp>/<arm>/sbatch/...`), never by job
name: job names are the arm names, shared by every experiment (`base`, `delta`...), so a name wait
counts other runs' jobs (10-26: an orphan `noconv` of a cancelled run held a smoke wait open).
Cancel a run the same way: `squeue -u bcarew -h -o "%i %o" | grep /Jobs/<exp>/ | awk '{print $1}' | xargs -r scancel`
(`%Z`, the working directory, is /sps/hyperk/bcarew for every job and matches nothing).
A failed squeue (or ssh) must not read as zero: `grep -c` prints 0 on empty input, so a transient
squeue error ended a wait while 39 jobs still ran (10-26) - keep the `|| exit 1` guard (n is then
empty, not 0). Give the user the ETA exp.py printed.
A job 10-30x slower than its siblings is /sps contention: stop submitting, tell bcarew.
`PENDING` + `AssocMaxJobsLimit` + `MaxJobs=0` is an admin block: stop, tell bcarew.

## 4. Merge, pull, compare

```
./exp.py merge coral:<exp>/<arm>      # prints "With fits (ok): N / Empty: M" — Empty > 0 = crashes
./exp.py pull  coral:<exp>/<arm>
sextant report Analysis/Experiments/<exp>.ini --collect   # -> Output/Sextant/<exp>/report/
```
Judge on **paired bootstrap over shared events**, scale-removed widths, converged
subset (fit < 20° from truth), held-out files. Vertex/angle: p68; momentum: iqr68 of the
scale-removed residual. A paired width gap < ~0.5 pt is unreadable at n < 1000.
Read the verdict from `Analysis/Scripts/verdict/paired_verdict.py --exp <exp> --ref base --arms <arm> --primary <metric>
--bar <percent>` (R79, 11-15; older `r*_pair.py` scripts are history), not from the Sextant report: the report's
comparison table is the MEDIAN per-event difference with a 68% interval. The script applies the rule (primary at 95%,
guards at 99%), says separately whether the bar is MET (estimate) and ESTABLISHED (interval), prints every width
definition side by side with the transverse vertex and the failures (R90: a gain of "-9%" on one core definition was
-3% on the others, and the transverse vertex was missing from two reports), refuses a merge whose events are not
entry-numbered, diffs the frozen configs, and never prints seconds (time bars: `bench` skill). Check the arms' likelihood evaluation counts differ when they
should (identical = dead switch). The learned post-fit estimator left config.ini on 11-14 (617f8df,
not part of [L]); a branch cut before that commit still applies it: empty
`config.tuning_files.post_fit_estimators.e- =` / `mu- =` in its arms and say so in the ledger row.
A fiTQun arm read from an old merge carries ONE hypothesis's kinematics (the report's validation
table says so): re-merge it (`exp.py merge fitqun:<exp>/<arm>`) before using p_e on muons.
PID guard for a change to the charge / light / time model: flat AND sloped cut, and muons with true
p > 200 MeV/c (`Analysis/Scripts/studies/pid/pid_rescore.py`; method-traps 11-14).
Output-only changes (vertex-bias curves) can be validated exactly offline with
`Analysis/Scripts/calibration/validate_vertex_bias.py` — no run needed.

Read the report's `validation` table first: `event-numbering` (a Coral merge numbered by WCSim EvtNum joins
later events of a gapped file to the wrong truth; `exp.py merge` now writes tree-entry ids and says so,
"Event ids are file * N + tree entry"), `hypothesis-kinematics` (old fiTQun merges), `truth-pairing`.
A failure-rate or tail difference against truth or fiTQun is only read after listing the failing event
ids: consecutive ids or a file full of them is a join error (method-traps 11-14).

## 5. Record

Ledger row + roadmap item in one edit; re-rank the queue if the result moves priorities.
A method mistake that produced a confident wrong number goes into `method-traps.md`.

## fiTQun specifics

~120-190 s/event (HK-2024 1R params); reuse existing references first
(`Output/fiTQun/Reco/cons_val_*`, `fitqun_v129_electron_full/`). Params are frozen per run.
Local fiTQun builds (see memory `fitqun-local-build`) exist but cluster is the default.
Full HK-2024 params (not `*_1Remu`) cost ~20 min/event.

## Gotchas

- zsh does not word-split `$var`: use `${=var}` or arrays in loops.
- Numba cache portability (10-30): jobs export `NUMBA_CPU_NAME=x86-64-v3` AND `NUMBA_CPU_FEATURES=""`
  (submit_coral.py). With the features unset numba keys its cache on the host's feature string, so a job on a
  node type other than the warm-up's recompiled everything (30-130 s per job; 132 s for the inlined kernel).
  Hand-written benchmark jobs must set both too, and benchmarks drop each run's first event.
- Silence numpy warnings in analysis one-liners (`python -W ignore ... 2>/dev/null`);
  a warning flood once cost 94 KB of context.
- A diagnostic bin must not be computed from the residual under study (method-traps).
- Truth join: Coral `event` = file×100 + EvtNum = truth `EvtNum` (v1.12.29 guns).
- The sub-event prefit leaves p = 0 (the momentum is seeded per hypothesis later): any code that
  uses the prefit seed's p directly runs below threshold (R75's first screen was void).
- Profile and bench jobs can land on a slow or loaded node: compare the kernel's ns per PMT with
  the last profile on the same CPU model before believing a timing (skill `profile`).
- File ranges get spent: say in the ledger row which input files decided the result, and take
  final numbers from the r1 100k validation sets (`tn_val_*`), never used to choose between arms.
- Local runs: `config_local.ini` is git-ignored and drifts. Run
  `Analysis/Scripts/local_config.py --slot c` first; it prints every non-path difference from
  `config.ini` (on 11-13: no slabs, no mu- nodes, rival refits on).
