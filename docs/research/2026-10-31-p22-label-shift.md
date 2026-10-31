# P22(a) and P21 step 0: research notes (2026-10-31)

Research step for lane C of `.superpowers/sdd/run-2026-10-31/advisor-01.md` (binding): P22(a), an EM label-shift prior on
`shape_possession`'s home probability (R020), then P21 step 0, whether wrong-team slots cluster at the ball. These notes
feed the plan `docs/superpowers/plans/2026-10-31-p22-label-shift.md` (worktree `BarcaHackathon-wt/p22-label-shift`).
They hold no figure of BAR-ATH. The numbers quoted are DFL benchmark aggregates already in `docs/results.md` (R011, R013,
R014) and T32's published share.

## 1. Sources read

| source | link | what it says that matters here |
|---|---|---|
| Saerens, Latinne and Decaestecker 2002, *Neural Computation* 14(1):21-41 | [pdf](https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf), DOI 10.1162/089976602753284446 | EM for new class priors from a fixed classifier's posteriors: the E-step reweights each posterior by π_new / π_train, the M-step averages them. It is an instance of EM and maximises the likelihood of the new data. It needs the training prior and calibrated posteriors. |
| Lipton, Wang and Smola 2018 (BBSE), ICML | [arXiv 1802.03916](https://arxiv.org/abs/1802.03916), [PMLR](https://proceedings.mlr.press/v80/lipton18a.html) | Moment matching on a held-out confusion matrix. It stays consistent when the classifier is uncalibrated, provided the confusion matrix is invertible and does not change between domains. |
| Alexandari, Kundaje and Shrikumar 2020, ICML | [arXiv 1901.06852](https://arxiv.org/abs/1901.06852) | EM (maximum likelihood) with bias-corrected calibration (BCTS: a temperature plus class-specific biases, fitted by NLL on held-out *source* data) beats BBSE and RLLS. Plain temperature scaling was sometimes not enough because of systematic per-class bias. For a binary task BCTS reduces to Platt scaling (a·z + b). The ML objective is concave. Lemma A: on unshifted data, EM returns the source prior if and only if that prior equals the mean calibrated prediction, so they set the source prior to the mean prediction. |
| Garg, Wu, Balakrishnan and Lipton 2020, NeurIPS | [arXiv 2003.07554](https://arxiv.org/abs/2003.07554) | First theory for MLLS (EM). It is consistent under calibration plus the same invertibility condition as BBSE. Its error splits into a miscalibration term and a finite-sample term. BBSE is MLLS with a coarse calibration, which costs it statistical efficiency. |
| Garg, Balakrishnan and Lipton 2022 (open-set label shift), NeurIPS | [arXiv 2207.13048](https://arxiv.org/abs/2207.13048) | A target that also holds a class the source never had breaks plain label-shift estimation. Here that class is the dead-ball frames, below. |
| Varin, Reid and Firth 2011, *Statistica Sinica* 21:5-42 | [journal page](https://www3.stat.sinica.edu.tw/statistica/j21n1/J21N11/J21N11.html) | Composite (independence) likelihood: maximising a likelihood that wrongly treats dependent observations as independent still gives a consistent estimator under the usual regularity and ergodicity conditions. Only its variance needs a sandwich correction. |
| Bourlard and Morgan, hybrid HMM/ANN (NIPS 1989; *Connectionist Speech Recognition: A Hybrid Approach*, 1994) | [NIPS 1989](https://proceedings.nips.cc/paper_files/paper/1989/file/bcbe3365e6ac95ea2c0343a2395834dd-Paper.pdf), [Edinburgh ASR notes](https://www.inf.ed.ac.uk/teaching/courses/asr/2024-25/asr10-hybrid_hmm_nn.pdf) | A classifier's posteriors divided by the class priors are scaled likelihoods, and those are what the HMM's emissions need. The class prior belongs to the HMM's state model. `shape_possession.py`'s docstring already cites this for its prior of 0.5. |
| Wojke, Bewley and Paulus 2017 (DeepSORT) | [arXiv 1703.07402](https://arxiv.org/abs/1703.07402) | Identity switches come mostly from occlusions. Its appearance term exists to carry tracks through occlusions and so cut identity switches. |
| Cui et al. 2023 (SportsMOT), ICCV | [arXiv 2304.05170](https://arxiv.org/abs/2304.05170) | In team sports, association rather than detection is the bottleneck: fast, variable-speed motion and players who look alike. |
| Khanna et al. 2025 (SportMamba), CVPRW | [arXiv 2506.03335](https://arxiv.org/abs/2506.03335) | Team-sport identity switches come from frequent occlusions, ambiguous appearance and non-linear motion. Its examples show IDs reassigned after players overlap. |

## 2. Is EM sensible here, and does it need a calibration step?

**The binary, mirrored, antisymmetric model is the easy case for EM.**
- The training prior is 0.5 exactly, by construction (each row and its mirror). The source prior is therefore known and
  not estimated, and the E-step is one line: posterior = σ(z + logit π), where z is the symmetrised logit.
- Alexandari's Lemma A asks that the source prior equal the mean calibrated prediction on source data. On the mirrored
  source this holds exactly: a frame and its mirror give p and 1 - p, so the mean is 0.5. EM on balanced, unshifted
  data therefore returns 0.5.
- The part of BCTS that Alexandari found matters most is the class-specific bias. Fitted on mirrored source data it is
  exactly 0: the NLL is convex and symmetric in b. Only the slope a, Platt's temperature, is left.
- EM is antisymmetric: on -z it returns 1 - π̂ up to rounding. Shifting by s·logit(π̂) therefore keeps the source
  antisymmetric: relabelling the teams still gives 1 - home.

**Which way miscalibration pushes π̂** (the derivation behind the decision below):
- *Overconfident* (|z| too large, fitted slope a* < 1): the posteriors are near 0 or 1, and EM tends towards
  classify-and-count. Under symmetric errors classify-and-count gives 0.5 + (π - 0.5)(2·acc - 1). With R011's pooled
  held-out accuracy of 0.84 that is a shrink of about a third towards 0.5. This is under-correction, the safe side:
  even pure classify-and-count would meet the advisor's gate if it were the only error (0.32 ≤ 0.75).
- *Underconfident* (a* > 1): the likelihood ratios are flattened, so the evidence for a share is spread over weak
  frames, and EM over-corrects, in the limit towards 0 or 1. This is the unsafe side. A synthetic test in the plan
  shows the slow drift.

**Decision: no calibration step in the primary run.**
1. The advisor's feasibility gate measures the consequence directly: π̂ against π_true, out of fold, on the seven DFL
   matches. A miscalibration large enough to matter fails it.
2. Temperature scaling would add a fitted constant per fold. To keep the benchmark out of sample it would need nested
   fitting (T for target m fitted on the other six matches' out-of-fold predictions), plus a second artifact under
   `--verify`. That is more machinery than one EM line, and it should come only once a failure names it.
3. The bias term, which Alexandari found to matter most, is zero here by construction.

**Proposed deviation D1 (diagnostics only, so that the one retry can name a cause).** The feasibility report also logs,
per match and report only:
- (a) the calibration slope a*, the slope-only logistic fit of the truth on z over the truth's alive frames. No intercept
  is needed, since on mirrored rows it is 0.
- (b) π̂ over the truth's alive frames only (section 5).

If the gate fails, these two numbers name the cause, and the pre-stated retry follows it (advisor-01 §3.3: one retry,
with a cause named from logged aggregates and its own part 1):
- a* outside [0.8, 1.25]: EM on σ(z / T), with T fitted per target match on the other six matches' out-of-fold
  predictions (Alexandari 2020; Garg 2020).
- π̂ over alive frames inside the gate while π̂ over all frames is not: section 5's retry.

## 3. The frames are not i.i.d.: is the simplest form enough?

**Yes, for the estimate.** EM's fixed point maximises the independence likelihood of the frames. Under dependence this is
a composite likelihood, and its maximiser stays consistent for a stationary, ergodic process (Varin, Reid and Firth
2011); only its variance is understated, and a sandwich would correct that. The plan uses the point estimate and no
interval, so nothing changes. The effective sample is roughly the number of possessions, a few hundred per match, not
the 100k+ frames, so π̂ carries a few points of sampling noise. The feasibility gate measures that together with
everything else. Thinning the frames (the trainer's stride 5) would leave the expectation unchanged and throw away
information, so it is not done.

**Where dependence does matter is downstream, in the decoder** (section 4).

## 4. A prior inside an HMM: what s = 1 means

- In the hybrid HMM view (Bourlard and Morgan), the shape source is a scaled likelihood: its prior is 0.5, and
  `shape_possession.py` refuses any other.
- Under label shift p(x | y) is unchanged, so the scaled likelihood is unchanged too. What changes is the prior over the
  state sequence, and in an HMM that lives in the transition model. A two-state chain whose stationary distribution is
  π̂ pays about logit(π̂) per change of team.
- Adding s·logit(π̂) to every frame's logit instead places the prior in every emission, so the decoder counts it once a
  frame, scaled by the source's weight w = 0.2.
- The order of magnitude, on the benchmark's own figures for `main.toml` (R014: 3.13 possessions per graded minute, `D`
  44.6%, so about 831 alive frames a minute and about 265 frames a possession): take π̂ = 0.62, so logit 0.49.
  - At s = 1 the tilt is 0.2 × 0.49 = 0.098 nats a frame, about 26 nats over a mean possession, against `switch_team`'s
    4 nats a change.
  - At s = 0.5 it is half that.
  - For scale, `controlled_touch`'s hold at 0.60 is 0.41 nats a frame (weight 1.0), and the shape source at its most
    confident gives 0.2 × 3.89 = 0.78.
- So s = 1 is the frame-level Bayes value (Saerens' adjusted posterior), not the decoder's correct prior. Both grid
  points are strong tilts towards the estimated majority team. The F1 test still decides honestly whether either helps,
  and R020 part 1 should say what s = 1 is and not call it optimal.

**Proposed deviation D2 (not for tonight).** Put π̂ in the decoder as asymmetric switch costs, t ± logit(π̂)/2, which is
where the hybrid HMM puts a class prior. It is left out tonight:
- the decoder would need a number that a source estimates, which is a contract change, and the contracts change "only
  with the team";
- it needs its own decoder grid.

Adding a smaller strength such as s = 0.1 to tonight's grid was considered and not proposed. At about 2.6 nats a
possession its effect would very likely fall inside the 0.005 band, so it could not pass the bar, and the advisor
fixed the grid.

## 5. The dead-ball frames in EM's sample

- EM runs over every graded frame with an opinion (advisor-01). That includes the dead-ball frames: about 45% of DFL's
  graded frames and about 36% of ours by `main.toml`'s labels.
- Neither of the model's class-conditionals describes them. The model was trained on alive frames only, so the target
  is a mixture with a third component, which is open-set label shift (Garg, Balakrishnan and Lipton 2022).
- At a restart the shapes probably mimic the restarting or attacking side, so this may bias π̂ either way.
- The gate tests the procedure as designed: π̂ over all opinion frames against π_true over alive frames. A
  contamination that matters therefore fails it.
- D1(b) logs π̂ over the truth's alive frames only, which separates this cause from the others.
- The pre-stated retry for this cause runs EM only over the frames `dead_ball_model` calls alive (P(alive) ≥ 0.5). It
  is left out of the primary run because it couples two sources inside one evidence function.

## 6. What this cannot fix

- Label shift assumes p(x | y) is the same in source and target. Bundesliga and LaLiga styles differ, so for BAR-ATH
  it is not.
- EM then estimates a share under a model of x that is partly wrong. The advisor's BAR-ATH band [0.39, 0.68] (outside
  it, sign-off with `label_diff` aggregates) and the `touch_agreement` check (shifted ≥ unshifted - 0.005) are the
  guards.
- The share may never come from published statistics of this match: those are human-produced data about it (RULES.md
  rule 2, T26).

## 7. P21 step 0: do identity swaps cluster at the ball?

- **What the literature says.** Identity switches in multi-object tracking come mostly from occlusions and close
  interactions (DeepSORT). In team sports, frequent occlusions, near-identical appearance and non-linear motion make
  association the bottleneck (SportsMOT, SportMamba).
- **Why our case points at the ball.** A wrong-team slot (T32) is a swap *across* teams. Jersey colours make that rarer
  than a swap within a team, so it needs a near-total overlap of two players of opposite sides. In football that is a
  challenge for the ball: players converge on it, and duels happen there.
- **Expected direction: ratio > 1.** The share of frames with a team over 11 slots should be higher on
  `controlled_touch`'s candidate frames (ball seen, nearest slot within 1.5 m, moving with the ball at 6 m/s or less)
  than on all graded frames.
- **Two effects pull the ratio towards 1**, so a ratio below 1.5 does not prove that swaps never start at the ball:
  - The measure is a state, not an event. A surplus persists after its swap (T32: median 2.7 s, p90 16 s) while play
    and the ball move on.
  - Set-piece crowding (corners, free kicks into the box) overlaps players away from the ball, which is with the taker.
- The advisor's rule stands as written: below 1.5, P21 is dropped; at 1.5 or more, report only.

**Proposed deviation D3 (optional, report only).** Beside the ratio, report an onset ratio: the share of surplus-run
first frames that fall within ±1 s of a candidate frame, over the share of graded frames within ±1 s of one. It tells
the event from the state. It carries no rule, and no decision depends on it.

## 8. Taken, and left out

| | taken | left out, and why |
|---|---|---|
| estimator | Saerens EM on the symmetrised, unclipped logit; source prior 0.5 exactly; start 0.5; stop at \|Δπ\| < 1e-6 or 200 iterations (advisor) | BBSE: its confusion matrix comes from DFL, and whether that transfers to LaLiga is exactly the doubt; Garg 2020 shows it is the less efficient member of the same family |
| calibration | none in the primary run; the slope a* as a report-only diagnostic (D1) | BCTS/temperature scaling up front: the bias is 0 by construction, the slope's effect is measured by the gate, and nesting T per fold adds machinery; it is the pre-stated retry if a* names it |
| dependence | the point estimate, which is consistent | intervals, sandwich variance, thinning: nothing reads them |
| placement of the prior | the advisor's per-frame logit shift, s ∈ {0.5, 1.0}; s = 1 described as the frame-level Bayes value | asymmetric switch costs in the decoder (D2): a contract change and a new grid |
| sample | every graded frame with an opinion (advisor); π̂ over the truth's alive frames as a diagnostic (D1) | EM over P05-alive frames: couples two sources; the pre-stated retry if D1 names dead frames |
| P21 step 0 | the advisor's state ratio and its 1.5 rule; T32's stricter surplus definition reported beside it | the ceiling run (advisor: not run either way); the onset ratio only if D3 is accepted |

## 9. Proposed deviations, for the advisor

- **D1** (diagnostics only): the feasibility report also logs, per match, the calibration slope a* and π̂ over the
  truth's alive frames. Neither is a gate. Each names one cause for the one retry: temperature-scaled EM if a* is
  outside [0.8, 1.25], EM over P05-alive frames if dead frames are the cause. Alexandari 2020; Garg 2020; Garg,
  Balakrishnan and Lipton 2022.
- **D2** (not tonight): put π̂ in the decoder's switch costs (t ± logit(π̂)/2) instead of in every frame's emission;
  per-frame placement counts the prior once a frame. Bourlard and Morgan's hybrid HMM.
- **D3** (optional, report only): an onset ratio beside P21's state ratio, because a surplus persists after its swap.
  DeepSORT, SportsMOT.
- **D4** (procedural): the feasibility read is its own short heavy run before the grid, not a section of the grid's
  run. π̂ is deterministic, so the values are the ones the grid's configs use, and a failed gate then costs no grid run
  and leaves no F1 to be unread.
