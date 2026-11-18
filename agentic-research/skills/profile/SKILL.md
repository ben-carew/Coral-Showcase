---
name: profile
description: Profile where one Coral single-ring event's time goes (per hypothesis, fit, step; kernel split by operation) with prof_event.py, and draw the flame graph. Use before any speed work, after a speed change is adopted, or when asked "where does the time go".
---

# One-core profile and flame graph

Tools (`Analysis/Scripts/`): `prof_event.py` (production `main()` monkeypatched: T timing, S staged kernel, C cProfile),
`prof_event_report.py` (tables), `prof_flamegraph.py` (folded stacks + SVG), `cluster/prof_job.sh` (the job).
Last profile: `Output/Coral/Timing/prof_r4/` (78b3d96 behaviour, EPYC 7302): 3.5 s / event; mu- hypothesis 39%, e- 28%,
decay-e sub-events 11%, prefit 10%, near-wall seed 5%; kernel 68%.

## Run (server, 1 core per job, ~10 min)
1. A frozen run of the code to profile: `R=/sps/hyperk/bcarew/Output/Coral/Jobs/<exp>/<arm>` (needs `code/`,
   `config.ini`, `numba_cache.tar`).
2. `scp Analysis/Scripts/profiling/prof_event.py Analysis/Scripts/profiling/prof_event_report.py Analysis/Scripts/cluster/prof_job.sh cca.in2p3.fr:/sps/hyperk/bcarew/Output/Coral/Jobs/<prof>/`
3. Six jobs: timing on two files per sample, staged kernel on a third:
   `sbatch --export=ALL,J=<prof dir>,R=$R,MODE=T,SMP=e,FILE=740,N=40,TAG=f740 <prof dir>/prof_job.sh` (and FILE=741; SMP=mu;
   MODE=S FILE=742 N=40 for each sample). Timing and staged runs are separate on purpose (staging slows the run).
4. Pull the `prof.jsonl` files into `Output/Coral/Timing/<prof>/`; then
   `python Analysis/Scripts/profiling/prof_event.py report <T_e...> <T_mu...> --stages <S_e> <S_mu>` and
   `python Analysis/Scripts/profiling/prof_flamegraph.py --timing T1.jsonl T2.jsonl --stages S.jsonl --label "mu- sample" --out flame_mu`.

## Traps
- **Slow node.** Each job prints its CPU model. Compare the kernel's ns per PMT (hit and unhit) with the last profile
  on the same model; a job 1.5x off landed on a loaded or older node: discard it (prof_r4's e- profile was).
- The first event of a run holds JIT and cache warm-up: the report drops it; do the same in any hand analysis.
- A profile describes ONE config. Check `diff` of its frozen config against production before quoting it.
- The staged kernel is a serial COPY of `_nll_fused`: when the kernel changes, update `build_staged()` first and check
  the staged total against the real kernel time (within 10%), or the operation split is fiction.
- Profiles rank ideas; only the same-node bench (skill `bench`) measures a gain.
