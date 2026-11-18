# R48: working notes (moved verbatim from roadmap v8 on 2026-10-22; Claude's file, the roadmap keeps a summary)

**19 · R48 [H]. Learned per-PMT likelihood (FreeDOM) — L** (long bet: accuracy + speed, GPU)
- Step 1 (M): train on ~1e6 e- guns; pulls and nLL at the true track vs Coral before any fit.
- PASS: beats Coral's e- momentum and vertex at equal PID on held-out events.
- 10-21 test (pre-reg): per-PMT likelihood-ratio classifier (FreeDOM-style: true vs prior-shuffled e- track per
  PMT, inputs = PMT geometry relative to the track, log p, towall/dwall, observation hit / log q / time residual;
  every hit PMT + a weighted unhit sample), trained on Coral's own sub-event-0 hit sets (`dump_hit_sets`, ben/r48-diag)
  of e- files 0-599; Nelder-Mead on the summed NN log-likelihood from Coral's production fit point, held-out e- files
  700-739. Step-1 PASS: NN-fit e- p core AND vertex p68 narrower than Coral production beyond CI (paired, same
  events); then the mu- network and PID (full PASS: equal PID). FAIL -> stop and report.
- 10-21 v1 RESULT: FAIL (876 held-out e-): p core +143%, p bias -28%, vertex +17%, angle +42% vs Coral; the truth-seeded
  fit lands at the same minimum; the network prefers Coral's point to the truth in 68%. Likelihood-ratio errors per PMT
  accumulate over ~20k PMTs. v2 (pre-reg, same data, same PASS): per-PMT DENSITY model trained by maximum likelihood
  (hit probability + 3-Gaussian mixtures for log q and the Cherenkov time residual, given the track geometry), the fit
  minimises the sum of these normalised -log p. If v2 fails too: R48 closed for this data size.
- 10-21 v2 (4 epochs) RESULT: Step-1 FAIL on momentum only: vertex -8.9% [-13.2,-3.9], transverse -24.5%, angle
  -18.7% [-22.3,-14.9] vs Coral (979 e-), but p core -5% [-19,+9] (ns), p half68 +10% [+0.5,+21]. Pre-reg v2b = same
  model trained to convergence (20 epochs); v3 = v2 + log(Coral's predicted direct light at the track) as an input
  (hybrid: Coral's analytic yield, learned shapes/corrections), 12 epochs. Same PASS for both.
- 10-21 v2b RESULT (979 e-): vertex -14.1% [-18.0,-10.1], transverse -29%, angle -23%, p half68 -7.8% [-14.9,-0.3],
  p core -10.2% [-20.8,+3.2] -> strict FAIL by the p-core CI only. Pre-reg (written before any new data): the FINAL
  verdict for v2b and v3 is on files 700-799 combined (~2500 e-, `r48_test2_e` adds 740-799), same frozen models and
  PASS, whatever the outcome; no further test data.
- 10-21 FINAL step 1 (2432 e-): PASS for v2b (p core -11.1% [-19.0,-3.1], vertex -11.4%, transverse -29%, angle -22%)
  and v3 (weaker); ledger Momentum 10-21. Step 2 (pre-reg): mu- density network (same recipe, mu- files 0-599), both
  hypotheses fitted with both likelihoods on held-out e- and mu- files 700-739 (seeds: Coral's own e-/mu- fits);
  full PASS = mu-eff at 99% e-eff of the network's dnLL not worse than Coral's beyond CI (joint bootstrap, cut
  re-derived per resample), with the e- result above; mu- widths reported.
- 10-22 RESULT: pre-registered PASS (step 1 + PID equal: +0.21 [-1.65,+1.54] pt). Against PRODUCTION (Coral + R54)
  the learned likelihood wins direction (e- -21%, mu- -22%) and transverse vertex (-11% / -9%) but loses momentum
  (e- half68 +10%, mu- +25%) and mu- vertex (+16%). Not adoptable as is. Next options (bcarew): R54 retrained on the
  network's fits (both gains may add); a hybrid using the network only for direction; GPU evaluation (R43).
- 10-22 R54n (bcarew: option 1), pre-reg: training set = Coral production + network fits (both hypotheses) on e-/mu-
  files 600-699 x 50 (disjoint from network training 0-599 and from tests); configurations (a) R54 features from the
  network's fits, (b) + Coral's fits as inputs, (c) Coral base + network inputs; then gate and GBM settings; chosen by
  5-fold CV by file on 600-699 only. ONE final test on files 700-739 (980 e- / 971 mu-) vs PRODUCTION (Coral + R54):
  report every width paired; 'better' = e- and mu- p core, vertex, angle each not worse beyond CI and >= 2 narrower.
- 10-22 CV on 600-699 (vs production): config b/d >> a (the network alone loses momentum); best = d (network fits +
  Coral's fits as inputs), GBM 800 trees depth 5, gate 1 m: e- p half68 -7.7%, vertex -10.9%, angle -20.7%, p core -2.8%
  (ns); mu- p core -11.1%, half68 -11.9%, vertex -18.5%, angle -32.9%. Learning curve steep (half the files: 2-5 pt
  worse) -> training set extended with files 800-999 (unused by the networks and the test); config fixed = d / GBM 1 /
  gate 1 m (re-checked by CV on the larger set); then the single final test on 700-739.
- 10-22 FINAL (ledger Momentum 10-22): PASS. Optimal = config d (network fits + Coral's fits stacked), gate 1 m,
  GBM e- 1500x6 / mu- 800x5, trained on 600-699 + 800-999 (CV on the larger set preferred the deeper GBM for e-).
  vs production: e- p core -9.6% (2.09%), half68 -9.4%, vertex -12.3% (18.8 cm), angle -24.1% (1.99 deg); mu- p core
  -10.3% (1.83%), half68 -14.8%, vertex -19.9% (15.6 cm), angle -32.0% (1.39 deg); PID = network's (equal to Coral).
  Cost ~7x Coral CPU (two network fits); adoption = bcarew's call (torch in production, GPU via R43 would cut it).

## Speed-up options for R54n (10-23, bcarew: record, do NOT implement yet)
Today: two Nelder-Mead fits per event (e- and mu- hypotheses) on the summed per-PMT network -log p, ~15 s each on 4
CPU threads (~7x Coral's CPU). Options, gains are estimates until measured:
1. Gradient fits (L-BFGS / Newton) from Coral's fit, which is already near the network's minimum; the network is
   differentiable (autodiff). R55's smoke: exact-gradient fits needed 10-20x fewer evaluations at >= 8-D. Est. 5-20x.
2. GPU evaluation (one MLP over ~20k PMTs is ideal batch work; V100, or JAX next to R39). Est. >= 10x per evaluation.
3. Fewer PMTs per evaluation: ring core + 10% stride for the unhit terms, as Coral does. Est. 2-3x; check gains survive.
4. Fit less: geometry only (p fixed at Coral's; the network's gains are direction / vertex) and / or only the better
   hypothesis (PID equal either way). ~2x each; retrain R54n on those fits.
5. Distil the 3x256 MLP to a smaller one. Est. 2-4x; possible accuracy loss.
6. Replace the network fits by R61's direct network estimate (ms/event) if it carries the same information.
Best CPU bet: 1 + 3 + 4, plausibly ~1x Coral's own cost extra. First: R61 step 0 (cheapest test of whether fits are needed).

## 10-27 Round 2 (bcarew: ideas 2 + 4, single ring) — pre-registered before any result
Data (all current production = ben/consolidate 8d2387e + hit-set dump, branch ben/r48-diag2):
network training = files 0-599 x 100 (r48v4_train_{e,mu}); stacking training = 600-699 + 800-999 x 100
(r54n2_{a,b}); DEV = 700-739 events 0-24 (the old R54n final test, now spent); FINAL = 700-739 events 25-99
(r54n2_test, 3000 / sample), used once.
- L-BFGS fit (autograd, seed frame, + Hessian) replaces Nelder-Mead iff on DEV (v2b): median |dnLL| < 0.1, < 5% of
  events worse by > 1, no width worse beyond CI. Smoke (3 e-): equal nLL to 0.1, 3-4x fewer s.
- Idea 2, network v4 [H]: 4x data, emission-point geometry (+4 inputs), time density conditioned on log q, K = 5,
  3-member ensemble (mean log p). Ablation: v2 architecture on the 4x data (separates data from architecture).
  PASS (stage 1, DEV, network alone from Coral's seed, vs v2b): e- and mu- each better on >= 2 of {p core, vertex 68,
  angle 68} and none worse beyond CI. Only then are the stacking sets refitted with v4.
- Idea 4, stacking GBM [H]: on r54n2_{a,b} with 5-fold GroupKFold by file, candidates (i) R54n as is (config d, GBM
  1500x6 / 800x5), (ii) + network Hessian widths h_*, (iii) random search over GBM settings, (iv) 5-seed bagging,
  (v) MLP instead of GBM. Choice by CV only (mean ratio of p core, p half68, vertex 68, angle 68 vs (i)).
  FINAL: winner vs (i) retrained on the same data: PASS = e- and mu- each better on >= 2 of the 4 widths, none worse
  beyond CI; also reported vs production (Coral + R54).
- 10-27 L-BFGS check (DEV, v2b): PASS. median |dnLL| 0.001, worse by > 1: 0.5% (e-, mu-); no width worse beyond CI;
  3.8-4.4 s vs 15-18 s per fit. L-BFGS (+ Hessian) used for every later fit.
- 10-27 DEVIATION from pre-reg (before any fit result): v4 seeds converge to very different optima (held-out -log L at
  the true track on DEV, per event vs v2b: e- s0 -709, s1 -889; mu- s0 -502, s1 -406, s2 -237); the mean-log-p
  ensemble is worse than its best member (e- -799, mu- -454). Use the best single member by DEV likelihood
  (e- s1, mu- s0) instead of the 3-member mean. Data alone (v2 net, 4x data): mu- -9 -> the gain is the architecture.
- 10-27 Idea 2 stage 1 (DEV, network alone from Coral's seed, v4 best member vs v2b): FAIL (e- angle).
  e-: p core +0.1% [-14.7,+10.3], half68 -8.1%, vertex -1.9% (ns), angle +10.2% [+3.9,+15.6] (worse beyond CI).
  mu-: p core -1.7% (ns), half68 -8.4% [-13.9,-1.2], vertex -5.5% [-9.6,-0.8], angle +2.2% (ns).
  Not an optimiser failure (truth point below the fit minimum in 0.6% / 4.2%, as v2b): the minimum moves although
  -log L at truth drops by 500-900 per event; median nLL(truth) - nLL(fit) ~70 for both (expected ~3.5 if calibrated)
  -> both networks are still far from calibrated; better density != better minimum location. Per pre-reg the final
  stacking uses v2b fits; v4 fits on the stacking sets (already submitted) only for an exploratory CV.
- 10-28 Idea 4 CV choice (v2b fits, 5-fold by file, mean ratio of the 4 widths vs (i)). Same recipe best for both:
  blend = mean of 3 MLPs (seeds 0-2) + 5-seed bagged GBM (rs00: 1000 it, lr .03, 31 leaves, min leaf 50, l2 3,
  max_features .7), all with Hessian widths. e- -3.39% (p core -2.0, half68 -1.5, vtx -8.4, ang -1.7, all CI < 0);
  mu- -5.04% (-4.6/-4.3/-6.9/-4.4). Hessian widths help every learner; MLP > GBM; random search alone ~0.
  (i) vs production (CV): e- -5.7/-9.0/-14.2/-24.5, mu- -14.5/-17.7/-22.6/-34.6. FINAL run once with this blend.
- 10-28 FINAL (r54n2_test, files 700-739 events 25-99; e- 2905 / mu- 2735 converged): PASS (pre-reg).
  Blend vs R54n as is (same data, retrained): e- p core -3.1 [-7.5,+1.1], half68 -1.5 [-3.7,+2.0], vertex -9.2
  [-11.7,-7.4], angle -3.2 [-5.3,-1.5]; mu- -3.8 [-8.6,+0.2], -5.2 [-7.9,-2.0], -6.3 [-9.3,-4.5], -4.4 [-6.9,-2.7].
  Vs production (Coral + R54): e- p core 2.01 -> 1.81%, half68 -9.8%, vertex 22.1 -> 17.1 cm, angle 2.55 -> 1.85 deg;
  mu- p core 1.86 -> 1.50%, half68 -24.5%, vertex 19.4 -> 13.7 cm, angle 1.97 -> 1.25 deg.
  Cost on top of R54n: 3 MLPs + 5 GBMs per target set (training minutes; inference negligible); the network fits
  need the Hessian (L-BFGS gives it for free, and L-BFGS is 4x faster than Nelder-Mead).
  Script: Analysis/Scripts/ml/r48/r54n2.py final --hess --blend '<json>' (exact call in logs/final_v2b_{e,mu}.log dir).
- 10-28 R54n3 (`Analysis/Scripts/ml/r48/r54n3.py`; server r48_nn/{oof3,resnet,tn}/). Two studies, same blend:
  (1) tn: Coral-only stack on raw 1527e03 (tech-note runs) replaces vertex-bias + R54: vs production e- -3.4/-4.2/
  -4.7/-4.2%, mu- -9.6/-10.0/-8.1/-5.2% (p core/half68/vtx68/ang68, 97.6k/92.8k events, all CI < 0). Corrected
  tn_val files (stack on the sample's own hypothesis, sub-event 0; rest = vb only; no gate; nhits = truth
  nhits_trigger0) -> Output/Coral/Reco/tn_val_{e,mu}/coral_*_coral_stack.root, for bcarew's validation session.
  Trained models are NOT saved (train-on-cal, predict-val in one job): save before shipping.
  (2) stack: sources C (Coral 8d2387e as written), N (v2b), N4 (v4), R (ResNet). ResNet alone >> all Coral/NN
  stacks; C+N+N4+R best (see ledger). ResNet inputs: trigger-0 digits, raw T (models subtract 950), tube-1,
  validated vs model card (e+ vertex mean 13.0 cm on 100 events). TRAP hit: registry `electron` = id_elpos (e+),
  `muon` = id_muneg (mu-); first mu run used id_mupos (vertices = the e+ files' -> nonsense), redone.
- Ideas to improve the NN-likelihood fits (untested): seed from ResNet; event-level score-at-truth penalty in
  training (fit minimum ~70 nLL below truth = correlated per-PMT errors); per-PMT densities conditioned on an
  event context; network nLL at Coral's / ResNet's point as stacking features.
- 10-28 Coral-only stack SAVED: `Output/Sextant/r54n3/stack_coral_{e,mu}.npz` (numpy-only: 3 MLPs as weight
  arrays + 5-seed GBM bag as R54-format tree arrays, 20 R54 features; trained on all converged tn_cal events,
  1527e03 raw). `r54n3_export.py` (train + export; numpy evaluator vs torch/sklearn <= 1.2e-4; retrain = applied
  corrections exactly), `r54n3_apply.py --reco <raw> --truth|--nhits --models <dir> [--base <vb file>] --out`
  (both hypotheses, sub-event 0, every event, like R54; ~7 min / 100k events / hypothesis pair on 4 CPU;
  reproduces the delivered tn_val files to 2e-4 cm). Delivered tn_val *_stack.root now = this tool's output.
