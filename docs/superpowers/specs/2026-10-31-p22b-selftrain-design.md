# P22(b): P07 recalibrated per match from the touch rule's own labels (design)

Date: 2026-10-31. Proposal: `docs/proposals.md` P22 (b). Results entry to come: R027 (R027b if its one retry runs).
Status: revised after an independent review (Fable 5.1, xhigh; `.superpowers/sdd/2026-10-31-p22b-selftrain/design-review.md`,
verdict "proceed with changes", every finding resolved in the text below and listed in the appendix). Tian's answers to
section 9 are still needed before part 1 of R027 is written.

## 1. Why

P07 (`shape_possession`) gives `home` on every graded frame from the two teams' shape. Its trees were fitted on the
seven DFL matches, and BAR-ATH is scored by `full`, a model that has seen neither Barcelona nor Athletic (R011).

**What hints at a per-match offset in P07's logit.**
- **R020.** EM's label-shift estimate assumes that P07's likelihoods transfer to the match. It missed the home share by
  0.0538 on average, by 0.18 on J03WN1 and by 0.09 the other way on J03WQQ, while P07's calibration slope stayed at 0.94
  to 1.16 per match (a geometric mean of 1.071). A slope near 1 with misses of that size is what a per-match offset in
  P07's likelihood would produce. That is a reading, not a measurement: R020 fitted a slope only.
- **R011.** The benchmark's gain is not a forecast for BAR-ATH: the teams are outside training, and the count of
  possessions moved the other way on our match.
- **No measurement of such an offset exists yet.** A19's pooled β of +0.13 is the class prior, the mean of
  logit(π_true) over the seven matches (+0.130), which the estimator below removes by construction. It is no evidence
  for the offset. The gate's oracle b\* (section 6.1) is the first measurement of it.
- **Mild counter-evidence.** A19's β is +0.16 with the ball seen and +0.07 with it missing. The frames the teacher labels
  (ball seen) and the frames the correction is for (gaps) may not share one offset. The gate measures that transfer
  (b\*_touch against b\*_gap) and does not assume it.

**Why it would matter.** P07 enters the decoder at weight 0.2 on every graded frame (`main.toml`). In a ball gap (41%
of the graded frames, median 10.7 s, A01) it is the main team evidence besides Smart Tagging. An offset b in its logit
moves the difference between the two teams' accumulated evidence by **at most** 0.2·|b| nats a frame (less where the
clip at [0.02, 0.98] binds). At b = 0.5 that bound is 25 nats over a 10 s gap, against `switch_team`'s 4.

**Belief.** The touch rule reads the ball and the players next to it, and never the teams' shape. On DFL its episode
frames name the right team 93.1% of the time (R013's floor, `main.toml`'s confidence 0.93). Fit a noise-aware logistic
intercept on one half's episode frames, with P07's logit as the offset and the class prior of those frames removed, and
it estimates the match's offset. Applied to the other half, it makes P07's evidence in the gaps more right, so the
decoder puts fewer wrong changes there.

**Falsified if:**
1. on DFL, the touch-fitted intercept does not lower P07's log-loss on the held-out half's alive frames with the ball
   missing, by the declared minimum and without harming any match (the gate, 6.1); or
2. the pipeline with it does not pass the bar on the mean of the two truths (6.3).

## 2. What: rung 1, one intercept per half

**The teacher.** `controlled_touch` at `main.toml`'s rule: radius 1.5 m, max_rel_speed 6.0 m/s, min_frames 1,
duel_radius 1.0, called with confidence 0.99 and hold 0.5, so that its `home` is a number exactly on the kept episode
frames.
- **The rows** are those episode frames, labelled with the episode's team, where:
  - P07's logit z is finite (someone is present);
  - `dead_ball_model`'s P(alive), at `main.toml`'s model, is at least 0.5 (review M8). A keeper holding the ball or a
    player placing it for a restart is a touch whose truth is D. The mask is label-free, the same on DFL and BAR-ATH, and
    it removes BAR-ATH's 2,934 frames before the second half's kick-off, where that model is forced dead.
- **Every frame is used**, with no stride (review m5). A stride only costs precision in a one-parameter fit.
- **Not the decoded labels.** In the gaps they carry P07 itself and the hold, which makes the labels circular. On BAR-ATH
  they also carry Smart Tagging, which DFL lacks.
- **Not Smart Tagging,** which DFL lacks. It is used instead as an independent check on BAR-ATH's gaps (6.4).

**The student.** logit′ = z + b_k on graded block k, where:
- z is P07's unclipped, symmetrised logit (`shape_possession`'s `_logit_home`, the fold model on DFL, `full` on
  BAR-ATH);
- b_k is fitted on the rows of the **other** graded blocks.
- **What the cross-fit buys** (review m10). It makes the held-out checks honest (6.1, 6.4: every number about block k uses
  a b that never saw block k). The double counting it also prevents is small: at most 0.2·|b| nats on frames where the
  teacher itself speaks at 2.6.

**The estimator (E2; review M5).** Two steps, with ε = 1 − 0.93 = 0.07, `main.toml`'s own constant for the teacher:
1. s is the maximum-likelihood offset under symmetric label noise ε: it maximises
   Σᵢ [ỹᵢ log qᵢ + (1 − ỹᵢ) log(1 − qᵢ)], where qᵢ = ε + (1 − 2ε)·σ(zᵢ + s). This is a forward correction, as in
   Patrini et al. 2017. The score is positive at s = −10 and negative at s = +10 on any realistic sample, so s is the
   root of the score, found by bisection to 1e-12.
2. b = s − logit(π̂), where π̂ = clip((ȳ − ε)/(1 − 2ε), 0.05, 0.95) is the teacher's home share corrected for the noise.
   s is the posterior offset on the teacher's frames. Subtracting the log prior odds of those frames leaves the
   likelihood offset, the quantity the decoder wants: a likelihood with prior 0.5, as P07 and the decoder require (R011;
   R020's D2).

**Why not the class-balanced intercept (E1).** With weights balanced on the noisy classes, the minority class's rows are
the dirtier ones, so E1 is pulled towards the teacher's minority class. Synthetic data, 4,000 rows, 20 repetitions, z
the exact log-likelihood ratio missed by b_true:

| teacher's home share | noise H / A | b_true | E1 | E2 |
|---|---|---|---|---|
| 0.50 | 7% / 7% | 0 | −0.009 | −0.009 |
| 0.65 | 7% / 7% | 0 | −0.107 | +0.008 |
| 0.35 | 7% / 7% | 0 | +0.118 | +0.007 |
| 0.65 | 7% / 7% | +0.5 | +0.385 | +0.496 |
| 0.65 | 4% / 10% | 0 | −0.025 | +0.056 |
| 0.35 | 4% / 10% | +0.5 | +0.691 | +0.569 |

E2 is unbiased under symmetric noise at any share. Both drift by about +0.06 when one team's labels are noisier than the
other's (review m7), which the benchmark's symmetric slot swaps do not simulate. The share of teacher rows next to a team
showing more than 11 slots is therefore reported per team on BAR-ATH. E1 is reported beside E2 in the gate and decides
nothing. The SD of both is about 0.03 at 4,000 independent rows; the real rows are autocorrelated, so expect more.

**Determinism.**
- The sums use `math.fsum` (exact, independent of the order), and bisection has no data-dependent branching beyond the
  sign of an exact sum.
- b is rounded to 1e-9 before use, so arithmetic that differs at the level of a ULP across machines cannot reach the CSV
  unless b sits within about 1e-16 of a rounding boundary.
- The remaining cross-machine exposure is z itself, which P07 already carries today (T20, T21).

**Minimum rows.** At least 500 rows of each teacher class among the other blocks' rows (20 s of controlled touches a
team). Otherwise b_k = 0, the source equals P07 exactly on block k, and the report says so. The synthetic self-test
inputs have two blocks of 300 frames (`SYNTHETIC_GRADED`), so they fall back to b = 0 through this floor; the checks of
the fit use a larger scene (review m2).

**The output.** home = clip(σ(z + b_k), 0.02, 0.98) where P07 speaks under `scope` (`all` in `main.toml`); `alive` is
NaN everywhere. Where every b is 0 it is P07 to the last bit (a check).

**Serving.** A new evidence kind `shape_selftrain`, in `pipeline/evidence/shape_selftrain.py`, with the estimators in a
helper module `pipeline/evidence/recalibration.py` (pure numpy and `math`, like `trees.py`). The tool imports the same
functions, so the gate measures the code that would be served.

```toml
[[evidence]]
kind = "shape_selftrain"
weight = 0.2
model = "pipeline/models/shape_possession_hgb.json"
scope = "all"
min_prob = 0.02
alive_model = "pipeline/models/dead_ball_hgb.json"     # the teacher's dead-ball mask
teacher = { radius = 1.5, max_rel_speed = 6.0, min_frames = 1, duel_radius = 1.0 }
```

- **The teacher table is checked strictly** (review m8). Its keys must be `radius`, `max_rel_speed` and `min_frames`, with
  `duel_radius` optional; `confidence` and `hold` are refused, since the source fixes them.
- **Shared work:** it reuses `shape_possession.load` and `_logit_home` (the memo P07 shares per Inputs, model file and
  key, review m4), `controlled_touch.compute` and `dead_ball_model.compute` (both memoised per Inputs).
- **No dependency, no artifact.** The fit runs at run time from `data/`, so the one command still goes from `data/` to
  the CSV. The benchmark runs it per (match, seed) like any source, so the simulation is the procedure itself.
- **No benchmark import:** it imports nothing from `pipeline.benchmark`, and the import scan passes.
- **A module of its own**, so that `main.toml`'s P07 is untouched and a drop removes two files and one registry line.

**The candidate.** `main.toml` with its `shape_possession` entry replaced by the entry above, and nothing else
re-chosen: not the touch rule, the dead-ball model, Smart Tagging or the decoder 4/32/35/25. The shift is re-derived
(6.2).

## 3. Considered and left out: the ladder

From the fewest parameters fitted on the match to the most:

| rung | what is fitted per half | status |
|---|---|---|
| 1: intercept | b (E2) | **First.** |
| 2: recalibration | b and a slope on z | **Left out.** The slope has nothing to fix: 1.07 to 1.10 (A19), a geometric mean of 1.071 (R020), and a retry conditioned on it could never fire (review M3). Label noise also attenuates a fitted slope, which would weaken P07 on every match. |
| 3: team-aware head | b plus β on the 26 standardised shape features | **Measured in the gate run as a report-only column; it is the one retry, R027b, only if rung 1 fails the gate and rung 3 passes it in the same run (review M3).** It is the last-layer retraining of Kirichenko et al. (2023): it can learn that one of these two teams' cues misleads, which an intercept cannot. When clinical prediction models were updated for new patients, updates of that size overfit the updating set (Janssen et al. 2008), so it is not first. |
| 4: retrain the trees | P07's 300 trees on DFL rows plus the match's labelled rows | **Left out.** This is the literal "self-training of P07". It means 44 models served per (match, seed, half), about 20 MB of trees or a cache outside the repo, a `--verify` for a committed pair, and hours of throttled training. The mirrored rows that keep P07 symmetric would also erase the team-specific shift. |

**Rung 3, declared now** (report-only in the gate; served only if R027b fires):
1. The 26 `shape_features` on the rows are standardised by their mean and SD over those rows. NaN becomes 0 after
   standardising, and a column with no spread becomes 0.
2. β is fitted by class-balanced logistic regression with offset z, a free intercept and an L2 penalty of 0.01 a row on
   β (a coefficient of one SD must buy 0.005 nats a row). Newton's method runs to |Δ| < 1e-10, at most 50 steps.
3. The intercept is then re-fitted by E2 with offset z + β·x̃, so that it carries no minority-class pull.

**Also left out:**
- **Iterated rounds.** One round, as P22 declared. Each round feeds the student's own errors back (Arazo et al. 2020).
- **P07's own confidence as the selector**, which is classic self-training. The confidence of boosted trees picks
  pseudo-labels badly (Kim et al. 2023; CAST), and it would feed P07's errors back to it. The teacher is another view.

## 4. Research

Sources, what was taken, and what was left out:
- **Confirmation bias.** Arazo et al. ([arXiv 1908.02983](https://arxiv.org/abs/1908.02983)): pseudo-labels reinforce a
  model's own errors. Taken: cross-fitting, one round, and a teacher that is not the student.
- **Class balance.** CBST (Zou et al. 2018, [arXiv 1810.07911](https://arxiv.org/abs/1810.07911)): class-balanced
  selection, so that the majority class does not take over. Taken: the class prior of the teacher's frames is removed,
  by E2 instead of by balanced weights, which noise biases (section 2).
- **Label noise.** Natarajan et al. 2013 and Patrini et al. 2017 (the forward correction): a model of the noise in the
  likelihood. Taken: E2, with `main.toml`'s ε.
- **Regularisation.** Kumar, Ma and Liang, ICML 2020 ([arXiv 2002.11361](https://arxiv.org/abs/2002.11361)):
  self-training under shift needs regularisation even with unlimited data. Taken: one parameter first.
- **Co-training.** Blum and Mitchell, COLT 1998
  ([PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/blum_a_1998_1/blum_a_1998_1.pdf)), and
  Abney 2002 (weak dependence is enough): one view labels another.
  - The touch rule and P07 are different views of one frame, but both see the players.
  - So the gate measures the transfer from the teacher's frames to the gaps, and does not assume it.
- **Updating a prediction model for a new population.** Steyerberg et al., *Stat Med* 23:2567–2586, 2004; Janssen et
  al., *J Clin Epidemiol* 61(1):76–86, 2008
  ([record](https://researchinformation.umcutrecht.nl/en/publications/updating-methods-improved-the-performance-of-a-clinical-predictio/)):
  - the updates run intercept, recalibration, revision, extension;
  - simple updates held on new patients, and revisions overfit.
  - Taken: rung 1 first, rung 3 only when measured.
- **Cross-fitting.** Chernozhukov et al. 2018, double/debiased machine learning. Taken: leave one block out.
- **Last-layer retraining.** Kirichenko, Izmailov and Wilson, ICLR 2023
  ([arXiv 2204.02937](https://arxiv.org/abs/2204.02937)): refitting the last layer of a fixed model undoes a shift in how
  its features are weighted. Taken: rung 3's form.
- **Tabular self-training with boosted trees.** Kim et al. 2023 ([arXiv 2302.14013](https://arxiv.org/abs/2302.14013))
  and CAST ([arXiv 2310.06380](https://arxiv.org/abs/2310.06380)): the trees' confidence is unreliable for picking
  pseudo-labels. Taken: the teacher's rule picks the labels. Their curricula are left out.

## 5. Rules

- **The labels.** They are made by an algorithm, and no person decided any of them, which RULES.md allows on the
  provided game ("pseudo-labels produced by an algorithm, provided no person decided any of them"). There are no hand
  labels on the match and no human-produced data about it.
- **The truth.** DFL's truth (other games) is used only to validate: the gate, the bar and the reports.
- **The public window** is a correctness check after the decision. No constant comes from it.
- **Where every constant comes from:**
  - the teacher's rule, the dead-ball model and ε are `main.toml`'s (R009, R013, R024);
  - the 500-row floor, the clip of π̂, the tolerances, the rounding, the gate's two thresholds, rung 3's penalty and the
    BAR-ATH margins are fixed here, before any run, and none is chosen from a score.
- **Reproducibility.** The fit runs inside `pipeline/run.py` from `data/`, deterministically, with no new pins and no
  new artifact.
- **Aggregates only.** The tool prints and writes per-block numbers, counts and log-losses, and nothing per frame of
  BAR-ATH.
- **The technical document** describes the self-training (the teacher, the fit, that no person decided any label) if it
  is adopted (review m11). The organisers will look for it there.

## 6. Validation (R027)

Part 1 of R027 states this section before any run, is read by an independent reviewer and is committed. Runs are
throttled, one at a time.

### 6.1 The gate: `pipeline/benchmark/selftrain.py gate`, run before the source exists (review M7)

The tool computes everything from the estimators, the fold models and the benchmark's own loaders, so the cheapest
falsification comes first. The source, its configs and its checks are built while the gate runs, in their own
worktree, and merged only if it passes.

**The run.** `tracking` v3, 3 seeds, all seven matches. For each (match m, seed, block k): z from m's fold model, b_k
from the other block's teacher rows, and the served `home` of P07 (b = 0) and of each variant, clipped as served.

**The gate's frames:** block k's alive frames (truth H or A) where the served ball is missing. Those are the frames P07
exists for, as in R011. The log-loss of each variant on them is pooled by frames, per match over its seeds and blocks,
and over all.

**Go iff both hold** (review M2), a miss by any margin being a miss:
- the pooled log-loss of rung 1 is at most the pooled base log-loss **minus 0.002 nats**. That is about a third of what
  correcting the offsets R020's three misses suggest (0.73, 0.35 and 0.22 in logit) at 70% would gain;
- **no match** has a rung-1 log-loss above its base log-loss **plus 0.002 nats**. That is the cost of a spurious offset
  of about 0.16 on a match that has none.

**Reported, deciding nothing:**
- **The base and every variant on the gate's frames:** the accuracy and log-loss of P07 (base) and rung 1 (E2); E1,
  E2's companion; the single b per match, fitted on both blocks (no cross-fit, review m10); and rung 3, the retry's
  column (6.5).
- **The oracles,** fitted with the truth's labels by class-balanced weights. These are clean labels, so balancing is
  exact:
  - b\*_gap on block k's gate frames;
  - b\*_touch on the rows the teacher gave b_k, the other block's.
- **The decomposition of b − b\*_gap** into teacher noise (b − b\*_touch) and transfer (b\*_touch − b\*_gap), per match
  (review M4a).
- **B and E/B:** B is the mean |b\*_gap|, and E/B is R020's form (means over seeds and blocks within a match, then over
  matches).
- **The STOP band of 6.4:** the largest |b\*_gap| over the 42 cells.
- **The teacher:** its precision against the truth on its rows; its home share; π̂; the share masked as dead; the rows per
  class; the blocks that fell back to b = 0.
- **The hold stratum** (ball seen, alive, outside the teacher's episodes): base and rung 1 log-loss (review M4c).
- **The touch agreement on the held-out block**, base and rung 1, at R011's rule (`conventions.TOUCH`, 2.0 / 6.0 / 3). It
  is a sanity check, not a transfer check (review m1).
- **A second run on `context`** (T38: the gaps fall where the play is, as on BAR-ATH): the same statistics, deciding
  nothing (review M4b).

### 6.2 The shift: R012's rule in R024's form (review B2)

1. The lag of `r027_preshift.toml` (the candidate without `[[refine]]`) is measured with
   `conventions.py latency … dfl --profile tracking --seeds 3 --truth raw --drop-match-only`.
2. s_b is minus the rounded median lag per type, and 0 below 30 matched changes.
3. `r027_final.toml` is `r027_preshift` with that s_b. It is `r027_post` if s_b is `main.toml`'s +1 / −2 / 0.
4. s_b is kept only if `r027_final` passes the guard against `r027_preshift` (the objective's Δ > 0 on `tracking`, not
   below −0.005 on `context`). Otherwise the candidate is `r027_preshift` itself.

The runs of 6.3 therefore carry all three configs.

### 6.3 The bar (on `tracking`, the mean of the two truths, R023)

**The runs.** `run.py pipeline/configs/main.toml r027_final.toml r027_preshift.toml` with `--profile tracking --seeds 3
--truth both --drop-match-only`, then the same three on `--profile context`.
- `main.toml` must reproduce 0.5363 raw and 0.5764 coarse to the last digit (review m12).
- If it does not, the lane **stops for the advisor**: something in the environment changed, and nothing about P22(b) is
  decided until it is explained.

**The bar.** One form, chosen by Tian in section 9 and written into part 1 before any run (review M6):
- **(i) the house bar for a new mechanism (R025):** the objective's pooled Δ > 0 on `tracking` with an exact sign-flip
  p ≤ 0.05 over the seven per-match deltas.
- **(ii) a magnitude bar with a no-harm veto, for an effect the belief expects on a few matches:** the objective's pooled
  Δ ≥ +0.005 (the house's tie band) on `tracking`, and no match's Δ below −0.005. The sign-flip p is reported.

**The vetoes,** in both forms. The candidate is not adopted if either is below −0.005:
- the objective's Δ on `context`;
- the pooled `tracking` alive/dead F1 minus `main.toml`'s (review B1: the decoder moves D edges jointly, so equality
  is not expected; R013's Stage A varied it from 0.9730 to 0.9731).

**Reported:** each truth's Δ, and A19's ladder for the candidate. `diagnose.py`'s `SHAPE_KINDS` gains `shape_selftrain`
(review m3).

**A failed bar or veto is a failure of the bar,** which means DROP (review m12b).

### 6.4 BAR-ATH, label-free, before adopting

**The STOP band.** Each |b_k| must be within the largest |b\*_gap| over the gate run's 42 cells (review m9), or the lane
**stops for the advisor**: an offset outside the benchmark's range is an extrapolation.

**Reported:** b per block, |b₁ − b₂|, π̂ and the masked share per block, and the share of teacher rows next to a team
showing more than 11 slots, per team (A16's definition, review m7).

**The check in the gaps** (review M4d). The frames are BAR-ATH's graded frames with the ball missing,
`dead_ball_model`'s P(alive) ≥ 0.5 and a Smart Tagging team vote. On them, the adapted source's agreement with the votes
must be at least P07's minus 0.005. Every block is read with the b fitted on the other block. Smart Tagging is in neither
the teacher nor P07, and it is the only independent view of our gaps (R021 used it this way).

**Determinism:** the same sha256 on two runs of the final config, and no `X` in a graded block.

**The D share:** within 1 point of `main.toml`'s 36.00%.

**Possessions per half:** reported, with a WARN beyond ±25% of `main.toml`'s (T33). The share of graded frames relabelled
is reported too (`label_diff.py`).

**Reported only:** the held-out touch agreement, base and adapted (a sanity check), and the one command's runtime.

**Failures:** a failed check here is a DROP. A b outside the band is a STOP.

### 6.5 The one retry: R027b (rung 3)

R027b runs only if rung 1 fails the gate and rung 3's column passes the same two conditions in the same gate run. It
gets its own part 1, its source (rung 3 served, with the same determinism rules) and the same 6.2 to 6.4. If it fails
anywhere, the lane is DROPPED. No other retry is pre-approved. After a failed bar the advisor may approve one only from a
cause named in the logged aggregates (R020's rule), and only if R027b has not run.

### 6.6 The public window

After the decision only, as a correctness check (T01, T27).

## 7. The decision flow (runs without Tian; the advisor takes what this does not settle)

```
estimators + tool + their checks → task reviews → R027 part 1 (with Tian's choices) → independent review → commit
  → GATE run on tracking (+ the report-only context run)      [the source is built meanwhile, in its own worktree]
      rung 1 passes → merge the source → SHIFT lag → BAR runs (tracking, context)
              main.toml not reproduced → STOP (advisor)
              bar and vetoes pass → BAR-ATH checks
                      b outside the band → STOP (advisor: adopt or drop, from the aggregates)
                      a check fails → DROP
                      all pass → ADOPT
              bar or a veto fails → DROP (or one advisor-approved retry, if R027b has not run)
      rung 1 fails, rung 3's column passes → R027b: part 1, rung 3 served, then SHIFT, BAR, BAR-ATH as above
      both fail → DROP
```

**ADOPT.**
- `main.toml` becomes the final candidate, kept as `pipeline/configs/r027_selftrain.toml`.
- `main.csv` is regenerated twice with the same sha256 and logged in the submission log. It is **not uploaded**.
- The pipeline suite, the benchmark suite and `selftest.py --data` must pass on the merged tree, and the merge is
  `--no-ff`.
- `docs/technical_document.md` and the submission document describe the self-training. The docs, CLAUDE.md's
  architecture line and the site are updated.

**DROP.**
- The estimators, the source, its registry line, the tool, the checks and the configs are removed in one commit that
  names the commit to recover them from. R027 keeps its results.
- `main.toml` is never touched.
- The source's worktree is deleted unmerged if the gate failed.

**The advisor** (an opus agent standing in for Tian, every ruling logged) decides only the three STOPs above, a
cause-named retry after a failed bar, and anything the flow does not foresee.

## 8. Cost

- **Estimators, tool and checks:** half a day. **The source, configs and checks:** half a day, in parallel with the gate
  run.
- **Heavy runs,** one at a time and throttled:
  - gate on `tracking` about 30 min, then `context` about 30 min;
  - lag about 20 min;
  - bar on `tracking` with three configs about 40 min, then `context` about 40 min;
  - BAR-ATH a few minutes.
- **End:** about two days, well before the freeze (13 to 14 Nov).

## 9. Questions for Tian (before implementation)

1. **The rung.** Start at rung 1 (one intercept per half), with rung 3 measured in the same gate run as the one retry,
   or go straight to retraining the trees (rung 4)?
2. **The bar.** (i) R025's house bar (sign-flip p ≤ 0.05, about 6 consistent wins in 7), or (ii) the magnitude bar with
   a no-harm veto (pooled Δ ≥ +0.005, no match below −0.005)? The belief expects a gain on the few matches with an
   offset and next to nothing elsewhere, which (i) can hardly see.
3. **After the gate.** Continue to the benchmark automatically when the gate passes (the research bar's pause is the gate
   itself)?
4. **Adoption.** Switch `main.toml` automatically when everything passes, or stop before switching it?

## Appendix: the review's findings and where they went

| finding | resolution |
|---|---|
| B1 alive/dead F1 "equal" | 6.3: a −0.005 veto, as R020 and R021 |
| B2 shift guard needs the preshift config | 6.2 and 6.3: `r027_preshift` in both runs; no shift if its guard fails |
| M1 A19's β is the prior | 1: struck as evidence, restated; its seen/missing split kept as counter-evidence |
| M2 the gate's 5 of 7 | 6.1: a pooled minimum (0.002) and a per-match no-harm limit (0.002) |
| M3 the retry could not fire | 3, 6.5: rung 2 dropped; rung 3 measured in the gate run as the retry's condition |
| M4 touch → gap transfer | 6.1: b\*_touch beside b\*_gap, the `context` run, the hold stratum; 6.4: the Smart Tagging check in the gaps |
| M5 noise pushes E1 towards the minority class | 2: estimator E2 (forward correction, the noise-corrected share removed), synthetic table; E1 reported |
| M6 the bar's power | 6.3, 9: (i) or (ii), Tian's choice, declared before any run |
| M7 the cheapest falsification last | 6.1, 7: the gate runs from the tool before the source exists |
| M8 dead-ball touches | 2: masked by `dead_ball_model` P(alive) ≥ 0.5 (this also covers the pre-kick-off frames); the share reported |
| m1 the touch agreement is circular | 6.1, 6.4: reported as a sanity check, at R011's rule |
| m2 the synthetic inputs have two blocks | 2: b = 0 there through the floor; the fit is checked on a larger scene |
| m3 `diagnose.py`'s `SHAPE_KINDS` | 6.3 |
| m4 the memo is private | 2: named, `shape_possession._logit_home` |
| m5 the stride | 2: dropped; the floor is restated in frames |
| m6 25 nats is a bound | 1 |
| m7 asymmetric teacher error | 2: synthetic +0.06; 6.4: the over-11 share per team reported |
| m8 the teacher table unchecked | 2: strict keys |
| m9 the band's definition | 6.4: the largest |b\*_gap| over the 42 cells |
| m10 the reason for the cross-fit | 2; 6.1: the single-b variant reported |
| m11 the technical document | 5, 7 (ADOPT) |
| m12 the reference run and failed checks | 6.3: a STOP if not reproduced; 6.3 and 6.4: what a failure means |
