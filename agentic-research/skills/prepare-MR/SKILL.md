---
name: prepare-MR
description: Prepare a series of stacked merge requests for upstream review - organise the commits into MRs (one branch each), validate every MR head against the one it builds on with Sextant (before-vs-after PDFs, paired stats, recommended figures), then let bcarew hand-edit every commit message and code comment in two text files and rewrite the history with them (code verified identical). Use when bcarew runs /prepare-MR or asks to prepare / split / validate MRs for GitLab.
---

# /prepare-MR

Everything lives in the **pack folder** `Analysis/Reports/Merge_Requests/`, defined by its
`series.ini` (base, MR branches, baseline runs). Tools, all taking `--pack DIR`:
`Analysis/Scripts/mr/mr_validate.py specs|submit|collect|build` and
`Analysis/Scripts/mr/mr_edit.py export|check|apply`. Worked example: the 2026-09-30 pack
(7 MRs onto `gdiazlopez_efficiency`; ledger row "MR review pack").

Commit messages, always: **one short imperative subject line, a few words; no body, no
explanation, no numbers, no jargon, no Co-Authored-By/attribution** (CLAUDE.md). Evidence
goes in the MR description, never in commits.

## 0. Start
- `grep -i "MR review pack\|R9" ledger.md roadmap.md` for the last pack's state.
- If `Merge_Requests/` holds a finished pack (its MRs merged or abandoned), move it whole to
  `Merge_Requests/archive/<date>/` first. Never overwrite `commit_messages.txt` /
  `code_comments.txt` bcarew may be editing (`export` refuses anyway).
- Ask bcarew for the target (upstream branch MR 1 goes onto, e.g. `upstream/dev`) and the
  tip (normally `origin/ben/dev`). `git fetch origin` and upstream over HTTPS.

## 1. Organise the commits into MRs
1. `git log --reverse --format='%h %s' <base>..<tip>`; the series must be linear on base.
2. Propose a grouping as a table (MR, theme, commits, files, what it changes). Rules:
   - one theme per MR, reviewable size (~2-20 commits); order: fixes that make the target
     run -> fit/likelihood features -> pure speed -> off-by-default features -> config and
     tuning -> one MR per later physics change;
   - MRs are **stacked**: MR N targets MR N-1's branch, so a later MR can use earlier ones;
   - prefer cutting the existing order (branch = the last commit of each group). Reordering
     needs cherry-picks into a slot (`git rebase -i` is unavailable) and every commit must
     still build and run: say what it costs before proposing it;
   - flag: a feature added in one MR but switched on in `config.ini` only in a later one
     (its gain then shows up in the later MR's report — say so in both descriptions);
     code added and later removed in the series; messages not in house style.
   - config files not in house style (CLAUDE.md "Config files": one header line + inline
     options per setting): fix them in the MR that shapes the configs, replaying later MRs'
     config hunks through the same restyle, and prove parsed values identical per commit.
3. Get bcarew's approval of the table (AskUserQuestion when there is a real choice).
4. `git branch ben/mr<N>-<slug> <sha>` for each MR (refs only, no checkout); push them to
   origin (`git push origin ben/mr<N>-<slug>`). Never push to upstream.
5. Write `series.ini` (copy the archived one): `base`, `also_move = ben/dev`,
   spec names (`experiment`, `report`: new names per pack), `slot` (a clean worktree
   slot), held-out `start`/`events`/`events_per_job` (default files 80-99 x 100);
   `[before]` = what MR 1 is compared against (existing reco `path.e`/`path.mu`, or
   `branch` to run it — if the target does not run as is, a branch with only the minimal
   fixes, like `val/gdiaz-base`); `[fitqun]` = existing fiTQun reco on the same events
   (reuse, never rerun); `[mr.N] branch, title` (+ `arm` only to reuse an existing run).
6. New commits on the tip later? If they belong to the last MR, fast-forward its branch
   (`git merge --ff-only`, push to origin), add an arm for the new head, and re-export.

## 2. Validate every MR (the `experiment` skill's rules apply: server, /sps limits)
1. Write the pass criterion into the roadmap item first, e.g. "no MR worsens a headline
   metric beyond its paired 95% CI; MRs meant to change nothing are identical".
2. `mr_validate.py specs` -> `Analysis/Experiments/<experiment>_{e,mu}.ini` (runs) and
   `<report><N>_{e,mu}.ini` (MR N vs N-1). Each arm runs its branch's own `config.ini`; the
   scheme is auto-detected to match it; event displays and truth copy are forced off.
   `sextant check` the run specs.
3. `mr_validate.py submit` (cycles the slot, restores it). ~20 jobs per arm per sample.
   Within ~3 min: `mr_validate.py collect --check-only` — any `tb`/`oom`/`disp` != 0 means
   stop. Then ONE background wait loop on the job names it prints (ETA ~30-60 min).
4. `mr_validate.py collect` (merge + pull; OOM check again), then `mr_validate.py build`
   -> `MR<N>.pdf` (summary page + e- and mu- Sextant reports) and `MR<N>_summary.csv`.
5. Read every summary and write `README.md` in the pack: the common method paragraph once,
   the whole-series change (base -> last MR, `r15_pair.py`), and per MR the paste-ready
   stats, 1-3 figures copied into `MR<N>_figures/` (usual picks: `residuals_aggregate`
   for tails, `resolution_p_true` for momentum width, `pid_roc` /
   `pid_efficiency_profiles` for PID, `timing` for speed MRs; none for an identical MR),
   and the caveats a reviewer will ask about. Tell bcarew plainly where speed or physics
   is NOT unchanged-or-better.

Reading traps: Sextant's `paired_differences` is the median of per-event SIGNED
differences (a p_res_pct delta is a scale shift, not a width) — quote the summary page's
widths instead. Server s/event varies ~10-15% by node: quote likelihood work (PMT evals,
logged since MR 5 of the first pack) where it exists. Bit-identity only on unchanged
arithmetic (summation-order changes give statistically equal, not identical, fits).

## 3. Hand-edit messages and comments, rewrite history
1. `mr_edit.py export` -> `commit_messages.txt` (every commit, grouped by MR) and
   `code_comments.txt` (every comment block, docstring and long trailing comment the
   series added that survives to its MR's head; identical text across files = one item).
   Fix any message not in house style in the file first, then hand both to bcarew.
2. When bcarew is done: `mr_edit.py check` (dry run: counts edits, proves every rebuilt
   commit code-identical — Python AST without docstrings, .ini/.sh without comments —
   and refuses on any code change). Show the counts; ask about any WARNING/NOTE.
3. `mr_edit.py apply`: rebuilds the chain with git plumbing (authors and dates kept),
   moves `ben/mr*` and `also_move` refs, keeps old heads as `refs/backup/pre-edit/*`,
   writes `sha_map.json`. The worktree holding `ben/dev` then needs
   `git reset --hard` (only if it was clean). Nothing is pushed.
4. Ask bcarew before force-pushing the rewritten branches to origin
   (`git push --force-with-lease origin ben/mr1-... ben/dev`); then the server
   slots need `git fetch && git reset --hard origin/<branch>` (only if clean). Branches
   built on the old commits must be rebased (`git branch --contains <old sha>`).
5. Upstream: never push — give bcarew the exact commands and the GitLab steps.

## 4. Record
Ledger row + roadmap item (R9-style) in one edit: pack path, verdict per MR, caveats.
