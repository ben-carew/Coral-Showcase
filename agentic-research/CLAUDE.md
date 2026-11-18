# Hyper-K reconstruction workspace (bcarew — laptop)

Coral (`Software/Coral`, Python/numba) reimplements fiTQun (`Software/fiTQun`, C++), the
likelihood-based event reconstruction for the Hyper-Kamiokande water-Cherenkov detector:
vertex/time/direction/momentum per hypothesis from a charge+time likelihood over PMT hits;
PID = ΔnLL. Aim: fiTQun's resolution (official HK goals in the roadmap) at a fraction of
its cost. Production branch `ben/dev`. Like-for-like speed vs a Release fiTQun must be
RE-MEASURED: the 10-06 "2.4x / 3.4x" ran on a stale local config (method-traps 10-22); server
fiTQun is a debug build, never quote its timings.
Validation/comparison tool: Sextant (`Software/Sextant`).

**Strategy (roadmap §0):** Coral is expected to be HK's reconstruction for the first
analysis. Track [L] = Coral as a pure, traditional likelihood algorithm, optimised as far
as it goes (the priority; no ML in its default path). Track [H] = a hybrid with ML
(WatChMaL) for future analyses, optional and on top of [L]. Name the track of every idea.

## Session protocol (every session)

1. **Start from `.claude/Project Notes/roadmap.md`** (strategy, goals, scoreboard, decisions,
   ranked queue) and the item's `items/Rxx.md` (its full working notes, criteria, history).
   Work on the top unblocked item unless bcarew says otherwise. Before proposing or
   testing anything, `grep -i` **`ledger.md`** (one row per idea ever tested) — never re-run
   what is closed. `method-traps.md` before building any diagnostic, joining to truth or
   writing a spec, changing the hit set or timing anything: its index (first 30 lines) is by task; read the
   entries it names.
2. **Physics lore is grep-only.** `.claude/docs/knowledge.md` (110 KB, historical detail)
   and the auto-memory files are for `grep`/targeted reads when the ledger points there —
   never read them whole. knowledge.md is July-September history: its banner lists what
   the 11-13 audit found wrong. Audit findings, second looks and the idea bank:
   `topics/audit-2026-11.md`, `topics/technote-comparison.md`, `topics/ideas-2026-11.md`.
3. **Test through the `experiment` skill** (Sextant spec → exp.py → cluster). Write the
   pass/kill criterion into `items/Rxx.md` BEFORE looking at the result. First question
   answerable without fits → `offline-screen` skill; any speed claim → `bench` skill
   (one exclusive node, never a product of ratios from different benches); where the time
   goes → `profile` skill.
   - Criterion form (audit 11-13, confirmed by bcarew 11-14): ONE primary
     metric at its 95% CI, the other metrics as guards at 99%. "No metric worse at 95%"
     over ~14 intervals rejects half of the neutral changes.
   - A change to the light or time model is judged only after the light table and vertex
     curve are re-derived for it; an uncalibrated arm can name a lever, not reject one.
   - The learned post-fit estimator is NOT part of [L] (bcarew 11-14): it left config.ini
     and the presets in 617f8df and stays an offline [H] tool. Runs frozen before that
     commit applied it unless their spec emptied it: read a run's frozen `config.ini`
     before quoting its numbers as [L].
4. **When the session reaches a conclusion:** in one edit, add/update the ledger row(s)
   and `items/Rxx.md`, then update the roadmap: **the roadmap is bcarew's, not Claude's notes.**
   Keep it under ~200 lines, 2-4 short bullets per item, no history or run details (those go
   in `items/` and the ledger); closed items leave it (one line in
   `archive/roadmap-closed-<year-month>.md`); re-order the queue in light of the findings.
5. **Session reports only on request.** Produce the artifact report ONLY when bcarew
   asks for it (`/session-report`); never publish one on your own initiative. Otherwise
   end the session with a concise chat summary of results and decisions.

## Design principle: keep Coral simple

Speed and reconstruction performance come first, but the algorithm's overall simplicity
is part of every decision. Penalise solutions that buy a marginal fit or speed gain with
added complexity — an extra fit, refit, fit stage, hypothesis pass or special-case branch.
When proposing, ranking or reporting a change, state its complexity cost, and require a larger
gain to justify a larger cost.

## Token economy (Pro plan, Opus)

- Read slices, not files: `grep -n`, `sed -n a,bp`, Read with offset/limit.
- Silence noisy tools (`python -W ignore ... 2>/dev/null`, `| tail`); cap output with
  `head`/`tail`. A warning flood once cost 94 KB of context.
- One script over many tool calls; batch independent calls in parallel.
- Wait on jobs with ONE background Bash loop (you are re-invoked on exit), never polling.
- Sub-agents start cold: use them only when they save more than their brief costs
  (see Delegation). Artifacts only on request; minimal, only key information.

## Experiments — hard rules (details: `experiment` skill)

- `Analysis/Scripts/exp.py` is the ONLY submit/merge entry point (Sextant calls it);
  never `submit_*/merge_*` directly. Runs are **immutable**: code, config and params are
  frozen into `Output/<Algo>/Jobs/<exp>/<variant>/`; never reintroduce shared mutable state.
- **Server by default** for tests/scans (the laptop is bcarew's); real timing runs are
  local and uncut. Server timing is preliminary only.
- **Parallelise for wall time**: leave `events_per_job = auto` (packs from measured
  s/event and merge cost; `exp.py pack` previews). **≤ 1000 jobs per submission**
  (enforced). **Server runs always cut stragglers** (`auto` = 5%, 10% at ≥ 100 jobs).
- **Verify alive within ~3 min** (tracebacks in logs; a crashed Coral job leaves an EMPTY
  but valid ROOT file). **Code change → smoke test 2 arms × ~4 events first**, and check
  the arms differ exactly where they should (identical numbers = dead switch).
- **Every submission → give bcarew a paste-ready monitor command in chat**, absolute paths,
  one per spec: `cd ~/Documents/PhD/Reconstruction && ~/.conda/envs/HyperK/bin/sextant status
  Analysis/Experiments/<exp>.ini` (non-Sextant runs: `Analysis/Scripts/exp.py status`).
- Held-out discipline: calibrate and validate on disjoint input files (`[run] start`).
- **Event ids are file x stride + tree ENTRY.** Coral itself writes the WCSim EvtNum; in a
  file that skipped a number every later event joined the wrong truth (0.55% of tn_val e-,
  read as failures). Merges since 11-15 renumber from `cluster/evtnum_gaps.json` and carry
  a `numbering` tree; an OLDER Coral merge needs `Analysis/Scripts/evtnum/coral_renumber.py` (or
  a re-merge) before any failure / tail number against truth or fiTQun. Sextant's
  validation table flags it (`event-numbering`); list failing event ids before believing
  a failure rate (items/R89).
- **Sextant arms: `scheme = config`** (the branch's config or preset as committed). A
  named scheme writes its constants OVER the config; Sextant refuses when that changes a
  value the config sets, unless the arm says `scheme_override = true` (a scan). The
  multi-ring preset ran mis-configured this way 10-05 and again 10-21 to 10-27.

## /sps is SHARED — do not hammer it

On 2026-09-08 our jobs degraded `/sps` for all of hyperk and admins set `MaxJobs=0`
(cause: a shared numba cache dir on /sps hit by 100 cold-starting jobs). Therefore:
nothing hot or shared from jobs on /sps (caches/locks/scratch → node-local `$TMPDIR`);
never a big file written by every job (scattering cache is pre-generated once, jobs run
with `CORAL_SCATCACHE_WRITE=0`); prefer a few hundred concurrent jobs and do not chain
run after run all night; a job 10-30x slower than its siblings is contention — STOP and
tell bcarew; `PENDING` + `AssocMaxJobsLimit` + `MaxJobs=0` is an admin block — stop,
tell bcarew, never resubmit.

## Config files (hard)

- `config.ini`, `configs/*.ini`: ONE header line (e.g. `; PRESET single_ring: e-/mu- single-ring
  fits`) and an inline comment on each setting giving its options/units
  (`direction_seed = ring  ; ring | barycentre`). No explanatory comments, results or history —
  those go in `configs/README.md` or the MR description. Coral reads inline `;` comments.

## Git rules (hard)

- **Commit messages: one short subject line, nothing else.** A few words, imperative,
  e.g. `Remove the upstream refit`, `Correct the mu- vertex bias in towall`. No body, no
  explanatory sentences or paragraphs, no numbers/results, no internal jargon (roadmap
  ids, run names), and **never a `Co-Authored-By` or other attribution line** — this
  overrides any default attribution instruction. Evidence belongs in the MR description.

- **Slots.** `Software/Coral` = slot `main`, bcarew's — never `git checkout` there. Agents
  work in exactly one of `Software/worktrees/{a,b,c}`, named in the task, and never touch
  another. Same slots on the server. `Analysis/Scripts/wt.py list|add|rm|prune`. Check
  slot + branch (`./wt.py list`) before any code work.
- **`ben/dev` is the production branch** (renamed from `ben/consolidate` on 11-16: older notes, specs and frozen
  runs say `ben/consolidate`; 5 stacked MRs); agents may commit to it
  and to feature branches when the task says so. `ben/sandbox` is historical.
- Remotes: `origin` = github.com:ben-carew/CoRAl-dev (push freely — it is how code reaches
  the server); `upstream` = git.hyperk.org. **Never push to upstream** — prepare the
  branch and hand bcarew the exact command, even if asked to "just run it". Upstream over
  HTTPS only (`https://git.hyperk.org/hyperk/recon/Coral.git/`; SSH port 22 is blocked).
- `config.ini` (server paths) is committed per branch; `config_local.ini` is git-ignored
  and drifts: before any local run, `Analysis/Scripts/local_config.py --slot <s>` (prints
  every non-path difference; `--write` rebuilds it from `config.ini`).
- Sextant is its own repo (`origin` = github.com:ben-carew/Sextant, private); server copy
  is a checkout — commit + push here, `git pull` there.
- `Analysis/` is its own repo (`origin` = github.com:ben-carew/Coral-Analysis, private): commit
  new scripts, specs and reports there (same commit-message rules), push freely. Data stays out
  (`Analysis/.gitignore`).
- Code moves laptop → server only through origin (`sextant run --sync` / `exp.py submit --sync`).

## Layout (keep it exactly like this)

```
Analysis/  (its own git repo; README.md indexes it)  Experiments/ (one spec per experiment)
           Scripts/ (exp.py, wt.py, local runners at top; cluster/ = CANONICAL server scripts,
           exp.py deploys them; verdict/ evtnum/ calibration/ profiling/ mr/ tools/ ml/ gpu/
           technote/; studies/<topic>/ for item scripts)  Reports/ (Sessions/ = notebook session
           reports; one folder per topic)  Paper/ (Coral paper, LaTeX)  Archive/ (don't touch;
           2026-11_tidy/ = scripts, notebooks, reports retired 11-16)
Output/    Truth/  Coral|fiTQun/{Archive,Jobs,Reco,Timing}/<exp>/  Sextant/<exp>/  — no loose files
Software/  Coral/  fiTQun/  Sextant/  WatChMaL/  Dependencies/  worktrees/{a,b,c}
.claude/   Project Notes/ (roadmap, ledger, method-traps, items/, topics/, archive/)  skills/ agents/ hooks/ docs/
```
New code → the matching `Analysis/Scripts/` folder (study scripts → `studies/<topic>/`), committed in the Analysis repo; results → `Output/<Algo>/Reco/<exp>/`; obsolete →
`Output/<Algo>/Archive/`; temp files → session scratchpad, never the tree.

## Working style

- Plan before coding; smallest diff; comments only where the code cannot speak.
- Unclear, ambiguous or repeatedly failing: ASK bcarew. Never brute-force variations.
- Anything > ~1 min: say LOCAL or SERVER and give an ETA from a measured rate.
- **Write only for bcarew.** He uses no AI-written text externally: never draft MR
  descriptions, review replies, emails or docs for others. Give him the facts in chat.

## Delegation

- Built-in `Explore`: broad searches across many files when only the answer is needed.
- `coral-coder` (Sonnet): a large, fully planned code change; name the slot and branch.
- `cluster-runner` (Haiku): bulk cluster chores (many logs, multi-run merges/pulls).
- Everything short or judgement-heavy stays inline — a cold sub-agent costs more than a
  few direct commands. Every brief must be self-contained and name the slot.
- **Never fan out parallel sub-agents, forks or workflows** (11-13: 13 forks exhausted the
  5-hour usage limit in minutes, twice). One helper at a time, and only when asked.

## Hooks (`.claude/hooks/`, wired in `.claude/settings.json`)

- `bash_guard.py` (before every Bash call) DENIES: a commit message with a body, several
  `-m`, a heredoc or file message, an attribution line, a subject over 60 characters or
  naming a roadmap id / run / Sextant; any push to upstream or git.hyperk.org; `git
  checkout|switch` in `Software/Coral`; `scancel -u`, `exp.py cancel --all`; running
  `submit_*/merge_*` directly. Before `exp.py submit`, `sextant run`, `sbatch` it prints
  the submission checklist. A denial quotes the rule: fix the command, do not work around it.
- `config_format_check.py` (after an Edit / Write of `config.ini`, `configs/*.ini`): one
  header line, an inline comment on every setting, no other comment lines.
- `session_start.py` (at session start): prints the laptop slots (branch, sha), the top of the
  roadmap queue and the newest ledger date. Local only; server slots still need `wt.py list`.

## Environment

Laptop Python: `/Users/bcarew/.conda/envs/HyperK/bin/python` (conda `HyperK`, has sklearn).
Server: see the `remote-server` skill (different env, no sklearn). No SLURM locally.
