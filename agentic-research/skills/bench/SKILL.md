---
name: bench
description: Same-node A/B timing of Coral code or config versions on one exclusive CC-IN2P3 node (the R72-R75 protocol) - fitting s/event, evaluations and core-s per event with per-file CIs. Use for any speed claim, before adopting a speed change, or to re-measure a combined speed-up. Not for physics verdicts (experiment skill).
---

# Same-node bench

Speed ratios are only valid inside ONE job on ONE exclusive node: CC-IN2P3 node types differ by up to 3.7x per event
(EPYC 7302 vs 9335) and a shared node by 2x between passes. Never multiply ratios from different benches in a report
without saying so; never quote server seconds as absolute.

## 1. Arms = frozen exp.py runs
Each arm needs a frozen run directory `Output/Coral/Jobs/<exp>/<arm>/` on the server holding `code/` (the source at
that commit), `config.ini` and `numba_cache.tar` (its warm-up cache). Any 4-event smoke run made by `exp.py` gives one.
BEFORE submitting, for every pair of arms:
- `ssh ... 'diff <runA>/config.ini <runB>/config.ini'`: the diff must be exactly the switch under test. Frozen
  configs carry no inline comments, so read values, not comments (R74's first bench was a dead switch).
- the smoke runs' `Likelihood cost` lines must differ where the arms should differ and be identical where the change
  is meant to be exact.

## 2. Submit (one job, <= 6 h, 64 cores)
```
ssh -o BatchMode=yes cca.in2p3.fr 'mkdir -p /sps/hyperk/bcarew/Output/Coral/Jobs/<bench> && cd /sps/hyperk/bcarew && \
  sbatch --export=ALL,BENCH=/sps/hyperk/bcarew/Output/Coral/Jobs/<bench>,NEV=20,PHASES="base:<runA>:<runA>/config.ini new:<runB>:<runB>/config.ini new:<runB>:<runB>/config.ini base:<runA>:<runA>/config.ini" \
  -o /sps/hyperk/bcarew/Output/Coral/Jobs/<bench>/job.out /sps/hyperk/bcarew/Scripts/bench/bench_node.sh'
```
(`scp Analysis/Scripts/cluster/bench_node.sh Analysis/Scripts/profiling/bench_summary.py cca.in2p3.fr:/sps/hyperk/bcarew/Scripts/bench/` first.)
- Phases in ABBA order (base new new base); each phase = 64 one-core workers (32 e- + 32 mu- runs, files 701-732, NEV
  events each, pinned with taskset). A phase takes ~NEV x s/event + start-up; the whole bench 10-25 min.
- An exclusive node can pend for a while: wait with ONE background loop on the job's script path (experiment skill).
- Give bcarew the monitor command: `ssh -o BatchMode=yes cca.in2p3.fr 'tail -3 /sps/hyperk/bcarew/Output/Coral/Jobs/<bench>/job.out'`.

## 3. Read
```
ssh -o BatchMode=yes cca.in2p3.fr 'source /sps/hyperk/bcarew/env_coral.zsh && python /sps/hyperk/bcarew/Scripts/bench/bench_summary.py collect /sps/hyperk/bcarew/Output/Coral/Jobs/<bench>'
scp cca.in2p3.fr:/sps/hyperk/bcarew/Output/Coral/Jobs/<bench>/summary.json Output/Coral/Timing/<bench>/
/Users/bcarew/.conda/envs/HyperK/bin/python Analysis/Scripts/profiling/bench_summary.py report Output/Coral/Timing/<bench>/summary.json --ref base
```
Quote: fitting s/event ratio (first event of every run dropped), the per-file median with its 95% CI, e- and mu-
separately, the evaluation ratio, and the node model from `job.out`. Core-s (start-up included) per phase pair: the
first phase of a job is often cold (page cache), so read the second pair.

## 4. A bench is void when
- any traceback, or runs with very different event counts;
- evaluation counts identical in arms that should differ (dead switch) or different in arms meant to be exact;
- the two passes of one arm differ by > 5% (the node was disturbed): rerun;
- the numba cache was not portable (event 0 of each run > 10 s): both `NUMBA_CPU_NAME=x86-64-v3` and
  `NUMBA_CPU_FEATURES=""` must be set (the script does).

## 5. Record
Ledger Speed row: bench name, node model, arms and order, events per arm, the numbers of step 3. Adoption bars for
speed live in the item (R73's standing bars: exact changes >= 1.02x, approximate >= 1.05x, + a stage >= 1.08x).
Reference: `r75_bench` re-read with `bench_summary.py` gives 1.267x, per-file 1.265 [1.233,1.310], evaluations 1.32x.
