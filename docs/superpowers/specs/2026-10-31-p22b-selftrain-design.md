# P22(b): P07 recalibrated per match from the touch rule's own labels (design)

Date: 2026-10-31. Proposal: `docs/proposals.md` P22 (b). Results entry to come: R027 (R027b if its one retry runs).
Status: draft, for an independent review (Fable 5.1, xhigh), then Tian's questions, then the plan.

## 1. Why

P07 (`shape_possession`) gives `home` on every graded frame from the two teams' shape. Its trees were fitted on the seven
DFL matches, and BAR-ATH is scored by `full`, a model that has seen neither Barcelona nor Athletic (R011). Three
measurements say that its error on a new match includes an **offset that belongs to that match's two teams**:

- **R020.** EM's label-shift estimate, which assumes that P07's likelihoods transfer to the match, missed the home share
  by 0.0538 on average, by 0.18 on J03WN1 and by 0.09 the other way on J03WQQ, while P07's calibration slope stayed at
  0.94 to 1.16 per match. A slope near 1 together with misses of that size is what an offset in P07's logit produces.
  That is a reading, not a measurement.
- **A19.** Pooled over the benchmark, P07's α is 1.08 (1.07 to 1.10 with the ball seen or missing alike) and its β is
  +0.13. Its slope is right, and even pooled it carries an offset.
- **R011.** The benchmark's gain is not a forecast for BAR-ATH: the teams are outside training, and the count of
  possessions moved the other way on our match.

It matters most in the gaps. P07 enters the decoder at weight 0.2 on every graded frame (`main.toml`). In a ball gap,
which covers 41% of the graded frames with a median of 10.7 s (A01), it is the main team evidence besides the tags. An
offset b in its logit adds 0.2·b nats a frame towards one team: at b = 0.5, a 10 s gap carries 25 nats, six times
`switch_team` (4).

**Belief.** The touch rule reads the ball and the players next to it, and never the teams' shape. On DFL its episode
frames name the right team 93.1% of the time (R013's floor, `main.toml`'s confidence 0.93). Fit a class-balanced
logistic intercept on one half's episode frames, with P07's logit as the offset, and it estimates the match's offset.
Applied to the other half, it makes P07's evidence in the gaps more right, so the decoder puts fewer wrong changes
there.

**Falsified if:**
1. on DFL, the touch-fitted intercept does not lower P07's held-out log-loss on the alive frames with the ball missing
   (the gate, section 6.1); or
2. the pipeline with it does not score higher on the mean of the two truths (the bar, section 6.3).

## 2. What: rung 1, one intercept per half

**The teacher.** `controlled_touch` at `main.toml`'s rule (radius 1.5 m, max_rel_speed 6.0 m/s, min_frames 1,
duel_radius 1.0). The labels are its kept episode frames, each labelled with the episode's team.
- It is the same code with the same parameters on DFL and on BAR-ATH, so the benchmark simulates exactly what runs on
  our match.
- **Not the decoded labels**, which P22's first wording named ("a confident touch inside a long decoded possession").
  In the gaps they carry P07 itself and the hold, which makes the labels circular. On BAR-ATH they also carry Smart
  Tagging, which DFL lacks, so the teacher would differ between the benchmark and our match.
- **Not Smart Tagging:** DFL has none, so the benchmark could not validate it.

**The student.** logit′ = z + b_k on graded block k, where:
- z is P07's unclipped, symmetrised logit, as `shape_possession` computes it (the fold model on a DFL match, `full` on
  BAR-ATH);
- b_k is fitted on the rows of the **other** graded blocks (leave-one-block-out). With the two halves, half 1's labels
  correct half 2 and the reverse. That is P22's cross-fitting: no frame's own teacher label enters its correction, so the
  touch evidence is not counted twice on a frame.

**The rows.** Every 5th frame of the other block from its start (P07's stride: it keeps the rows deterministic and cuts
the autocorrelation), kept where the teacher has a label and z is finite (someone is present).

**The fit.** b solves Σᵢ wᵢ (σ(zᵢ + b) − yᵢ) = 0.
- **Weights:** class-balanced, so that each class's weights sum to one half. This is CBST's balance. It also keeps the
  student a likelihood with prior 0.5, which P07 and the decoder require (R011; R020's D2): b is an offset of the
  likelihood, not the class prior that P22(a) estimated.
- **Solving:** the left side strictly increases in b, so the root is unique when both classes have rows. It is found by
  Newton with a bisection safeguard (tolerance 1e-12, at most 100 steps).
- **Determinism:** the sums use `math.fsum`, which is exact and independent of the order, and b is rounded to 1e-9
  before use. Arithmetic that differs across machines at the level of a ULP cannot reach the CSV unless b sits within
  about 1e-16 of a rounding boundary.
- **Minimum rows:** at least 200 rows of each class among the other blocks' rows. Otherwise b = 0 for that block, the
  source equals P07 exactly there, and the report says so. With one graded block (the synthetic self-test inputs) there
  are no other rows, so b = 0.

**The output.** home = clip(σ(z + b_k), 0.02, 0.98) where P07 speaks under `scope` (`all` in `main.toml`); `alive` is
NaN everywhere. Where every b is 0 it is P07 to the last bit (a check).

**Serving.** A new evidence kind `shape_selftrain`, in `pipeline/evidence/shape_selftrain.py`:

```toml
[[evidence]]
kind = "shape_selftrain"
weight = 0.2
model = "pipeline/models/shape_possession_hgb.json"
scope = "all"
min_prob = 0.02
teacher = { radius = 1.5, max_rel_speed = 6.0, min_frames = 1, duel_radius = 1.0 }
```

- It reuses `shape_possession`'s model loading and its logit memo, and `controlled_touch.compute` (confidence 0.99 and
  hold 0.5, so that its `home` is a number exactly on the kept frames, as `train_shape.touch_agreement` already does).
- It adds no dependency and no artifact. The fit runs at run time from `data/`, so the one command still goes from
  `data/` to the CSV. The benchmark runs it per (match, seed) like any source, so the simulation is the procedure itself.
- It imports nothing from `pipeline.benchmark`, and the existing check holds.
- A separate module rather than an option of `shape_possession`, so that `main.toml`'s P07 is untouched and a drop
  removes one file and one registry line.

**The candidate.** `main.toml` with its `shape_possession` entry replaced by the entry above: the same model, weight 0.2,
scope `all` and `min_prob` 0.02, and the teacher at `main.toml`'s own touch rule. Nothing else is re-chosen: not the touch
rule, the dead-ball model, Smart Tagging or the decoder 4/32/35/25. The shift is re-derived by R012's rule (section 6.2).

## 3. Considered and left out: the ladder

The options run from the fewest parameters fitted on the match to the most. Rung 1 is the first.

| rung | what is fitted per half | why not first |
|---|---|---|
| 2: recalibration | b and a slope a on z | The slope has little to fix: 1.07 to 1.10 (A19) and 0.94 to 1.16 (R020). About 7% of the teacher's labels are wrong, and that noise flattens a fitted slope (attenuation), which would weaken P07 on every match. Correcting for it (forward correction, Patrini et al. 2017) needs the noise rate on BAR-ATH, which is unknown. **This is the one retry, for one named cause (section 7).** |
| 3: extension | b plus L2-penalised terms on the 26 standardised features, team-aware | It would learn which of these teams' shape cues mislead. But that is about 27 parameters from noisy labels on ball-seen frames, asked about the gaps. When clinical prediction models were updated for new patients, the revisions that refit more of the model looked better on the updating set and lost on new patients, while the simple updates held (Janssen et al. 2008). |
| 4: retrain the trees | P07's 300 trees on DFL rows plus the match's pseudo-labelled rows | This is the literal "self-training of P07". It means 44 models (7 matches × 3 seeds × 2 halves, plus BAR-ATH's 2), served per (match, seed, half), about 20 MB of trees or a cache outside the repo, a `--verify` for the committed BAR-ATH pair, and hours of throttled training. The mirrored rows that keep P07 symmetric would also erase what is team-specific, which is the shift we are after; without them the loader's symmetrisation must change. M to L effort for an unknown gain. It makes sense only after rung 1 shows whether there is an offset to fix at all. |

Also left out:
- **Iterated rounds.** One round, as P22 declared. Each round feeds the student's errors back to it (confirmation bias,
  Arazo et al. 2020).
- **A noise correction of b.** Symmetric label noise shrinks b towards 0, which errs on the side of doing nothing, and a
  correction would need the noise rate on BAR-ATH.
- **P07's own confidence as the selector** (classic self-training). The confidence of boosted trees is a poor selector
  of pseudo-labels (Kim et al. 2023), and P07's own errors would be fed back to it. The teacher is another view.

## 4. Research

Sources, what was taken, and what was left out:
- **Self-training and confirmation bias.**
  - Arazo et al. ([arXiv 1908.02983](https://arxiv.org/abs/1908.02983)) found that pseudo-labels reinforce a model's own
    errors. Taken: cross-fitting, one round, and a teacher other than the student.
  - CBST (Zou et al. 2018, [arXiv 1810.07911](https://arxiv.org/abs/1810.07911)): class-balanced selection, so that the
    majority class does not take over. Taken: balanced weights.
- **Regularisation.** Kumar, Ma and Liang, *Understanding Self-Training for Gradual Domain Adaptation*, ICML 2020
  ([arXiv 2002.11361](https://arxiv.org/abs/2002.11361)), show that regularisation is necessary for self-training under
  shift even with unlimited data. Taken: the most regularised student there is, one parameter.
- **Co-training.** Blum and Mitchell, COLT 1998
  ([PDF](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/blum_a_1998_1/blum_a_1998_1.pdf)): one view
  labels the other. It works when the two views are close to conditionally independent given the label, and weak
  dependence is enough (Abney 2002).
  - Here the touch rule reads the ball and its nearest players, and P07 reads team aggregates and never the ball. They
    are different views of one frame, but not independent: both see the players.
  - That is why the gate measures the transfer and does not assume it.
- **Updating a prediction model for a new population.**
  - Steyerberg et al., *Stat Med* 23:2567–2586, 2004, and Janssen et al., *J Clin Epidemiol* 61(1):76–86, 2008
    ([record](https://researchinformation.umcutrecht.nl/en/publications/updating-methods-improved-the-performance-of-a-clinical-predictio/)),
    order the updates as intercept, recalibration, revision, extension.
  - Simple recalibration fixed poor calibration and held on new patients. Revisions overfit.
  - Taken: start at the intercept.
- **Cross-fitting.** Chernozhukov et al. 2018 (double/debiased machine learning): fit on one fold and apply on another, so
  that a correction never sees its own data. Taken: leave one block out.
- **Last-layer retraining.** Kirichenko, Izmailov and Wilson, ICLR 2023
  ([arXiv 2204.02937](https://arxiv.org/abs/2204.02937)): refitting only the last layer of a fixed model on a small set
  is enough to undo a shift in how its features are weighted. It is the deep-learning cousin of model updating, and
  supports adapting a small head over a frozen model. Rung 3 would be that head; rung 1 is its intercept.
- **Self-training with boosted trees on tabular data.** Kim et al. 2023
  ([arXiv 2302.14013](https://arxiv.org/abs/2302.14013)) and CAST ([arXiv 2310.06380](https://arxiv.org/abs/2310.06380))
  find the trees' confidence unreliable for picking pseudo-labels. Taken: the teacher's rule picks the labels, not P07's
  confidence. Their curricula and cluster-based confidence are left out as too much for one round.

## 5. Rules

- **The labels.** They are made by an algorithm and no person decided any of them, which `RULES.md` allows on the provided
  game ("pseudo-labels produced by an algorithm, provided no person decided any of them"). There are no hand labels on
  the match and no human-produced data about it.
- **The truth.** DFL's truth (other games) is used only to validate: the gate, the bar and the reports.
- **The public window** is a correctness check after the decision. No constant comes from it.
- **Where every constant comes from.** The teacher's rule is `main.toml`'s (R013, R024); the stride is P07's. The
  200-row floor, the tolerance and the rounding are fixed here, before any run, and none is chosen from a score.
- **Reproducibility.** The fit runs inside `pipeline/run.py` from `data/`, deterministically, with no new pins and no
  new artifact.
- **Aggregates only.** The tool prints and writes per-block b's, counts and log-losses, and nothing per frame of
  BAR-ATH.

## 6. Validation (R027)

Part 1 of R027 states all of this before any run. An independent reviewer reads it, and it is committed. Runs are
throttled, one at a time, `--drop-match-only` wherever `main.toml`'s Smart Tagging is in a config.

### 6.1 The gate (feasibility, DFL): `pipeline/benchmark/selftrain.py feasibility`

The run covers `tracking` v3, 3 seeds and all seven matches. For each (match, seed, block k), with the fold model, the
tool computes the base and the adapted log-loss of the served `home` (clipped) on block k's alive frames (truth H or A)
where the served ball is missing. Those are the frames P07 exists for, and R011's figures are of the same kind.

- **Go iff** the adapted log-loss, pooled by frames over all of them, is lower than the base, **and** it is lower in at
  least 5 of the 7 matches, each match pooled over its seeds and blocks.
- A miss by any margin is a miss.

**Reported, deciding nothing:**
- the accuracy, base and adapted;
- the fitted b_k;
- **the oracle b\*_k:** the same balanced intercept fitted on block k's own alive gap frames with the truth's labels and
  the same offset. It is the offset the correction is trying to estimate;
- E/B, where E is the mean |b − b\*| and B the mean |b\*|, means over seeds and blocks within a match, then over matches
  (R020's form);
- the oracle slope (intercept and slope fitted on the truth), as a geometric mean over matches;
- the teacher's precision against the truth on its rows;
- the held-out touch agreement, base and adapted;
- the row counts and the blocks that fell back to b = 0.

### 6.2 The shift

By R012's rule (R024's form): the lag of `r027_preshift.toml`, the candidate without `[[refine]]`, measured with
`conventions.py latency … dfl --profile tracking --seeds 3 --truth raw --drop-match-only`. The final candidate is
`r027_preshift` with the shift the rule gives. That is `r027_post.toml` only if the shift is `main.toml`'s +1 / −2 / 0.

### 6.3 The bar (R025's form: a new mechanism)

**The runs.** `run.py pipeline/configs/main.toml <final candidate>`, with `--profile tracking --seeds 3 --truth both
--drop-match-only`. `main.toml` must reproduce 0.5363 raw and 0.5764 coarse to the last digit, or the lane stops. Then
the same two configs on `--profile context`.

**The tests:**
- **The guard:** the objective (the mean of the two truths' pooled spellF1, R023) has a pooled Δ > 0 on `tracking`, and
  a Δ not below −0.005 on `context`.
- **The bar:** an exact sign-flip p ≤ 0.05 over the seven per-match deltas of the objective on `tracking`
  (`choose.py --objective mean` reports it).
- **Reported:** each truth's Δ, and A19's ladder for the candidate (`diagnose.py`).
- **A check:** the alive/dead F1 must be equal to `main.toml`'s, because the source touches `home` only.

### 6.4 BAR-ATH, label-free, before adopting

- **b per half.** It must lie within the largest |b\*| seen on DFL, or the lane **stops for the advisor**: an offset
  outside the benchmark's range is an extrapolation. |b₁ − b₂| is reported.
- **Determinism:** the same sha256 on two runs, and no `X` in a graded block.
- **The D share:** within 1 point of `main.toml`'s 36.00%.
- **Possessions per half:** reported, with a WARN beyond ±25% of `main.toml`'s (T33). The share of graded frames
  relabelled is reported too (`label_diff`).
- **Touch agreement:** the held-out touch agreement of the adapted source must be at least the base's minus 0.005. The
  label-free look at the transfer is R011's statistic on the half that the correction did not see.
- **Runtime:** the one command's runtime is reported. The fit is a root-find over a few thousand rows.

### 6.5 The public window

After the decision only, as a correctness check (T01, T27).

## 7. The decision flow (runs without Tian; the advisor takes what this section does not settle)

```
implement → checks green → task reviews → R027 part 1 written → independent review → committed
  → GATE (6.1)
      pass → SHIFT (6.2) → BAR (6.3)
                 pass → BAR-ATH CHECKS (6.4)
                          pass → ADOPT
                          b out of range → STOP for the advisor (adopt or drop, from the aggregates)
                          any other check fails → DROP
                 fail → DROP (a retry only by the advisor, from a cause named in the logged aggregates; R020's rule)
      fail → if the oracle slope (geometric mean) is outside [0.8, 1.25] → RETRY R027b (rung 2), its own part 1 first
             else → DROP
  RETRY R027b: the same gate, shift, bar and checks; any failure → DROP
```

**ADOPT.**
- `main.toml` becomes the final candidate, kept as `pipeline/configs/r027_selftrain.toml`.
- `main.csv` is regenerated twice with the same sha256 and logged in the submission log. It is **not uploaded**.
- The pipeline suite, the benchmark suite and `selftest.py --data` must pass on the merged tree, and the merge is
  `--no-ff`.
- The docs and the site are updated.

**DROP.**
- `shape_selftrain.py`, its registry line, `selftrain.py`, the checks and the configs are removed in one commit that
  names the commit to recover them from. R027 keeps its results.
- `main.toml` is never touched.

**The advisor** (an opus agent standing in for Tian, every ruling logged) decides only:
- the STOP above;
- a cause-named retry after a failed bar;
- anything this flow does not foresee.

## 8. Cost

- **Implementation:** about a day. That covers the solver, the source, the tool, the configs and the checks, each with a
  task review.
- **Heavy runs:** about 2 hours, one at a time and throttled. The gate takes about 30 min; the lag, the `tracking` bar and
  `context` about 20 to 40 min each; BAR-ATH a few minutes.
- **End:** well before the freeze (13 to 14 Nov).

## 9. Open questions for Tian (asked before implementation)

1. Start at rung 1 (one intercept per half), with rung 2 as the one retry, or go straight to rung 4 (retraining the trees)?
2. The house bar needs about 6 consistent wins in 7. An intercept helps the matches with an offset and does next to
   nothing elsewhere, so it can fail the bar while being useful. Keep the house bar as it is?
3. Continue to the benchmark automatically when the gate passes (the research bar's pause is the gate itself)?
4. Adopt into `main.toml` automatically when everything passes, or stop before switching it?
