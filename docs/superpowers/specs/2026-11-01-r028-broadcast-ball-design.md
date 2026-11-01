# R028: the broadcast's ball in our gaps (P15, use 1) — design

**Status:** designed 2026-11-01, after a builder check (appendix) and with Tian's four answers (section 10). Nothing is
built. Tian asked for it after A20 ("write up R028 and plan it out"). The plan that implements it is
`docs/superpowers/plans/2026-11-01-r028-broadcast-ball.md`. R028 part 1 in
`docs/results.md` will be the binding declaration once it is committed. Where this document and that declaration
differ, the declaration wins.

**In one paragraph.** Our ball is missing on 41% of the graded frames. The TV broadcast tracks the same match from
another angle, and A20 found it aligned to our feed by two constants that code can find. Where ours is missing and the
broadcast's is not, R028 puts the broadcast's ball in, before any evidence source reads the tracking. So
`controlled_touch` and `dead_ball_model` see more ball, unchanged. Nothing else changes: no source, no model, no
decoder parameter. DFL has no broadcast, so the benchmark gets a stand-in: a simulated broadcast ball on each DFL match,
transplanted from what the real one does in our gaps. R028 measures that stand-in against `main.toml` by a bar declared
first.

## 1. Why

- **The ball in our gaps is worth a lot, if it can be had.** On the benchmark a perfect ball in our gaps is worth
  +0.1016 raw and +0.0197 coarse (A14), +0.061 on the objective (the mean of the two truths, R023/R024).
- **The broadcast has some of it** (A20):
  - it is a second, independent tracker (a person steers the camera; the tactical camera adds nothing in the gaps);
  - two offsets align it, +4,014 frames in the first half and +26,919 in the second, steady to ±1 frame;
  - its players confirm the alignment on 91.6% of the valid frames;
  - its usable ball would fill 29.6% of the gap frames our decoder calls alive and 4.2% of those it calls `D`;
  - where both feeds have the ball, they are a median 0.77 m apart (1.00 m near our gaps);
  - 15% of the balls it would give us in the gaps are beyond a line, a ball out of play that ours lost.
- **The belief, and why it is modest:**
  - expected objective gain +0.005 to +0.012 (range −0.002 to +0.015); about 45% that it clears +0.005;
  - covering 30% of the alive gap frames would be +0.018 if the value scaled linearly, but four things cut it:
    - the fill clings to the gaps' edges (A20 part 6: 30% of a gap's first second, 7% of its inside), where the
      decoder already knows who had the ball;
    - its fragments are short (median 14 frames in our alive), shorter than the decoder's 35-frame minimum possession;
    - about a third of its positions are more than 2 m off near our gaps;
    - the coarse truth barely rewards more ball (A14).
- **T38 makes the benchmark's number generous.** `tracking` v3 places our gaps blind to the play, so a larger share of
  its gap frames are alive than ours (our decoder calls 20% of our gap frames alive). The simulated fill therefore
  covers more of each graded minute on DFL than the real one does on BAR-ATH. The run reports both (W1, W2, section 7.1)
  so the gap is seen. The bar decides as declared; it is the same caveat as R011's ("not a forecast for our match").
- **It is allowed.** "Everything is fair game, in any combination, as input to an algorithm" (DATA.md), and the
  broadcast is a provided asset. RULES.md rule 2 is about human-produced data from outside (events, corrected tracking,
  annotations); the broadcast's own tracking is the organisers' automatic output. The offsets are found by code, from
  aggregates of the two tracks; no person sets them. The technical document must say the broadcast is used and how.

## 2. What: rung 1

### 2.1 The switch

A new optional table in a run's TOML, checked as strictly as the rest of the config (pipeline/config.py):

```toml
[inputs]
ball_fill = "broadcast"   # default "none": the tactical ball only, as every config so far
```

`config.py` gains `inputs` among the top-level keys and `Config.ball_fill`. Any other key, or any other value, stops
the run before data loads. `pipeline/README.md` documents it. Every existing config is unchanged and writes the same
bytes.

### 2.2 The broadcast's assets

- **The ATD.** Parsed once into `data/cache/atd_broadcast/` by the tactical cache's own code (`atd_cache.load_or_build`
  with a folder name), 164,488 rows, 51 slots, the same checks.
- **The homography.** Only its `valid` column is used (`starter.load_homography.load(source="broadcast")`).
- **Lazy loading.** Neither is loaded unless a config asks for the fill.

### 2.3 The alignment, found by code on every run

For each graded half h, the offset d_h such that tactical frame t pairs with broadcast frame t − d_h:
1. **Coarse:** every 25th graded tactical frame of the half, against every offset in steps of 25 frames. The score is
   the share of pairs within 3 m among those where both balls exist and the broadcast homography is valid. An offset
   counts only with at least 300 such pairs.
2. **Fine:** every graded frame of the half, offsets within ±40 frames of the coarse best. The score is the share
   within 1.5 m.
3. **Acceptance,** or an `AlignmentError` that stops the run:
   - the fine score is at least 0.5;
   - the coarse best is at least twice the best coarse score more than 10 s away.

   A misaligned feed must never be filled from silently.
4. **Ties** go to the smaller |d|. The search is plain numpy, with no randomness, and takes about 4 s for the two
   halves (A20 part 6).

Expected: d_1 = 4,014, d_2 = 26,919 (the 25 fps windows gave 26,918 to 26,920). The run's manifest records both and
the scores. The second half's 2,750 frames before its kick-off map into the broadcast's first half. The live check
(2.4) rejects them, and they are all `D` anyway (T02).

### 2.4 Live frames

A mapped frame is **live** when:
- the broadcast homography is valid;
- at least 4 broadcast players are present;
- the median, over the broadcast's present players, of the distance to the nearest present tactical player is below
  2.0 m.

Replays, close-ups and misaligned stretches fail this. The live share is 91.6% of valid graded frames (A20).

### 2.5 Quality: no run shorter than 13 frames

A **usable** broadcast ball is present on a live frame. Runs of usable frames shorter than 13 frames (0.52 s) are
removed, because A20 part 6 shows the short runs are mostly wrong:

| run length (frames) | median distance from ours | within 2 m |
|---|---|---|
| 1 to 4 | 26.9 m | 13% |
| 5 to 12 | 15.9 m | 33% |
| 13 to 49 | 1.6 m | 68% |
| 250 or more | 0.7 m | 77% |

The run is the broadcast ball's own run, which often reaches outside our gap. 60% of the fill fragments start on the
gap's first frame: the broadcast still had the ball when ours was lost.

### 2.6 The fill

- **Where:** on graded frames where the tactical ball is missing and a usable broadcast ball exists.
- **What:** `ball_xy` becomes the broadcast's position (both ATDs use the same normalised pitch, and A20 confirmed the
  orientation and the team convention). `ball_speed` stays NaN; no source reads it (T31).
- **How it reaches the sources:** `Inputs.atd` returns the filled tracking, so every source reads it with no change.
  `Inputs.ball_filled` is a read-only boolean mask of the filled frames, for the reports (and for R028b).
- **The rest stays:** the players are the tactical ones. No model is retrained, and no decoder or refiner parameter
  changes.
- **Expected on BAR-ATH:** about 5,970 filled frames, 3.8% of the graded frames.

### 2.7 Determinism

- The alignment, the live test, the quality filter and the fill are pure numpy, with no randomness. The same files give
  the same mask and the same bytes.
- The fill adds the broadcast files to what regenerates `main.csv`: `data/broadcast/BAR-ATH_broadcast_atd.{xml,txt}`
  and `_homography.csv`. The run manifest records their sizes and hashes, as it does the tactical files'.

## 3. The benchmark's stand-in: a simulated broadcast on DFL

### 3.1 Why a simulation, and what R019 taught

- DFL has no broadcast, so the real fill cannot run there.
- A simulated broadcast ball, built from what ours does in our gaps, can. P14 already planned the same kind of
  stand-in for a video detector.
- R019's camera simulator failed because a parametric follower could not reproduce our camera's stillness. So this one
  is a **transplant** (P01's way: "their pattern, not their clock"). Fill masks and position errors are copied from
  BAR-ATH, not modelled.

### 3.2 The driver: the decoder, not the truth (my call)

Where the simulated broadcast has the ball depends on the play, as the real one does. 29.6% of our decoder's alive gap
frames are filled against 4.2% of its `D` ones. On DFL the state of each gap is read from **the candidate's own labels
without the fill**: the same config with `ball_fill = "none"`, decoded on the served match. It is not read from DFL's
truth.
- **It keeps the benchmark's rule that no truth reaches the inputs** (`checks/inputs.py`, `test_match_inputs_hold_no_
  truth`). A ball placed by the truth's alive flag would carry the answer.
- **It mirrors how A20 measured the rates:** against our decoder's labels, which are not a truth either.
- **It is slightly pessimistic.** Where the decoder wrongly says `D` in live play, the stand-in fills at the `D` rate,
  so it cannot show the fill fixing those frames.

The cost is one more decode per (match, seed) for each config that fills.

### 3.3 The donor pools (BAR-ATH, built once, cached, never printed)

`pipeline/benchmark/broadcast_sim.py pools` loads BAR-ATH and builds, with the pipeline's own functions of section 2:
- **The labels:** decoded in-process from `pipeline/configs/r024_both.toml`, not read from whatever `main.csv` is on disk
  (the builder check's question 2; R027's way).
  - Their CSV must have sha256 `e6496a9b…`, the file `main.toml` writes today. Otherwise the step stops.
  - Pinning a record config keeps the stand-in fixed after an adoption, when `main.toml`, and so `main.csv`, change. A
    stale or dirty `main.csv` cannot creep in.
- **The gaps:** every maximal run of graded frames without the tactical ball, inside a graded half (150 gaps).
  - Each has its fill mask (2.6) and its class: **alive** if at least half of its frames are `H` or `A` under those
    labels, else **D**. A20 counted 55 alive and 95 `D`.
- **The errors:** the maximal runs of graded frames where both balls exist, the broadcast's is usable, and the frame is
  within 2 s of one of our gaps (the near-gap class, median 1.00 m). Each frame's offset is broadcast minus tactical,
  in metres.
- **The cache:** `data/cache/benchmark/broadcast_sim/<digest>/`, like `calibration.py`'s.
  - The key holds the ATD caches' source records, the broadcast files' hashes, `r024_both.toml`'s text, its CSV's
    sha256 and the pools' version.
  - The digest goes into **every** report of a run that loaded the pools, including `main`'s, which does not fill. That
    is what lets `choose.py` compare `main` with the candidate once its SAME gains `broadcast_sim`. A run in which no
    config fills writes null.
  - These are frame-level arrays of our match: they stay in `data/cache/`, are never printed and never committed.

### 3.4 The transplant

For every DFL gap g (a maximal run of served-missing ball frames inside a graded half, length m):
1. **Its class:** alive if at least half of its frames are `H` or `A` under the driver, else `D`.
2. **The donor:** among the donor gaps of that class, the 5 whose lengths are closest to m on a log scale (ties to the
   shorter, then the earlier). One is picked with `numpy.random.default_rng((seed, match index, 28))`, in gap order.
3. **The mask:** fill[g.start + i] = donor_fill[⌊i · k / m⌋] for i = 0 … m − 1, with k the donor's length. This keeps
   the fill at the gap's two edges where the donor has it.
4. **The position:** each maximal fill fragment of length f draws an error run from the pool with the same generator,
   uniformly among runs at least f long (or the longest, repeated, if none is). Its first f offsets go onto DFL's
   clean ball (in metres, then normalised by the match's pitch).
5. **No ball on DFL there:** a fill frame where DFL's clean ball is NaN stays empty.

`MatchInputs` takes the result as `broadcast_ball=`: an (n, 2) normalised array, NaN where there is no fill. The fill
of 2.6 then applies to it unchanged, as it does to the real one on BAR-ATH. The decision of which frames to fill lives
in one place.

### 3.5 Fidelity: gates, checks and warnings (run first, before any bar run)

`broadcast_sim.py fidelity --profile tracking --seeds 3 CONFIG` applies one function, `measure_fill()`, to BAR-ATH (the
real fill, `main.csv`'s labels) and to the 21 simulated copies (the driver's labels), pooled by frames.

**Gates** (placement). They decide whether the stand-in is ours:

| | statistic | expected on BAR-ATH (A20) | pass if |
|---|---|---|---|
| F1 | fill share of the gap frames the labels call alive | 0.296 | \|sim − ours\| ≤ 0.05 |
| F2 | fill share of the gap frames the labels call `D` | 0.042 | \|sim − ours\| ≤ 0.03 |
| F3 | fill share in the first 1 s / last 1 s / inside of gaps ≥ 2 s | 0.297 / 0.218 / 0.073 | each \|sim − ours\| ≤ 0.05 |

**Checks** (the code does what it says):

| | statistic | pass if |
|---|---|---|
| F4 | median fill fragment, by class | within 10 frames |
| F5 | median position error of fill frames | within 0.2 m |

**Warnings**, reported and deciding nothing; they size T38's generosity:

| | statistic | expected on BAR-ATH |
|---|---|---|
| W1 | fill frames per graded frame | 0.038 |
| W2 | share of gap frames the labels call alive | 0.20 |
| W3 | share of fill frames beyond a line by > 1 m | 0.151 |

A failed gate or check is a **STOP for the advisor**, not for the controller who wrote the code (the builder check's
question 4):
- The advisor compares the code with this section's text. It does not look at the number the gate missed.
- **If the code departs from the text,** the one fix allowed brings it back to the text. No constant, threshold, donor
  rule or tolerance may change; that would tune the stand-in towards the gate, which T26 and T37 forbid.
  - The fix is logged in part 2 as a deviation, reviewed, and the fidelity run is repeated once.
- **If the code matches the text,** the failure is a verdict. The stand-in is not ours, there is no bar run, and the
  result is **DROP**, as R019 dropped.

The BAR-ATH numbers in the tables are A20's and are expected, not targets. The gates compare the code's own
measurement of both sides.

### 3.6 What the stand-in cannot reproduce

- A misaligned or replayed broadcast frame that slips past the live test. On BAR-ATH the live test and the alignment
  checks (7.4) are the guard.
- Failure modes of the broadcast's tracker beyond A20's statistics.
- T38's blind gaps.

None of these can be tested on DFL. So two things stand: section 7.4's independent check on BAR-ATH, and the bar's no-harm
rule.

## 4. Considered and left out

- **A truth-driven placement** (fill by DFL's alive flag at A20's rates). It is more generous and breaks the no-truth
  rule (3.2).
- **Copying the donor gap that `tracking` v3 transplanted.** It is blind to DFL's play, so the alive gaps would be
  filled at the `D` rate.
- **A parametric fill process.** It is R019's trap; the transplant keeps the edge structure for free.
- **Averaging the two balls where both exist.** Ours is already there, and A20 cannot show the broadcast's is better.
- **The broadcast's shots off the pitch as dead-ball evidence** (P15 use 2), and **its aim as a ball zone** (use 3).
  They are separate experiments with different tests (P15).
- **Re-choosing the decoder, or retraining `dead_ball_model` or P07 with fills.** Rung 1 changes the input only. A
  re-choice would be a new proposal.

## 5. Research

- **Prior art** (the survey of 1 Nov, in P15):
  - nothing on filling one tracker's ball gaps from a second broadcast tracker for possession labelling;
  - multi-camera ball fusion exists (Maksai et al., CVPR 2016);
  - single-view ball height is metres off (SoccerNet-v3D, 4.2 m), one more reason not to trust the far tail of the
    error distribution;
  - aligning two recordings by matching their tracking in pitch coordinates is the field's own route (Crang et al.
    2025). Ours is the simplest form of it.
- **Our measurements:** A20, parts 1 to 6. Every constant above is one of them or a round value:
  - 3 m and 1.5 m (the alignment's tolerances);
  - 2.0 m and 4 players (the live test);
  - 13 frames (part 6's table);
  - 5 donors (enough to vary the pick; the pools hold 55 and 95 gaps);
  - 0.5 (the class threshold).

  None comes from the public window, and none was chosen by a score.

## 6. Rules

- Match data never reaches an LLM. The pools, the masks and the simulated balls are frame-level and stay under
  `data/cache/`. Reports and prints hold aggregates only.
- No truth through the inputs (3.2). The pools read `main.csv`, which is our own output, not a truth. The public window
  is never used for anything here.
- `evaluation/` is not touched.
- Until an adoption commit, `main.toml` writes `e6496a9b…`.
- One heavy process at a time, throttled. Never relaunch a reaped run without Tian.
- No AI attribution in commits. Push only to the private origin.

## 7. Validation (R028)

### 7.1 Fidelity first

Section 3.5's command on `tracking` (21 copies) decides. The same on `context` is reported.

### 7.2 The shift: R012's rule in R024's form (as R027's 6.2)

1. `r028_preshift.toml` is `main.toml` with `ball_fill = "broadcast"` and without `[[refine]]`. Its lag on DFL comes
   from `conventions.py latency pipeline/configs/r028_preshift.toml dfl --profile tracking --seeds 3 --truth raw
   --drop-match-only`, with the simulated fill.
2. s_b is minus the rounded median lag per type, and 0 below 30 matched changes.
3. `r028_final.toml` is the preshift config with that s_b.
4. s_b is kept only if `r028_final` passes the guard against `r028_preshift`: the objective's Δ > 0 on `tracking`, and
   not below −0.005 on `context`. Otherwise the candidate is `r028_preshift`.

### 7.3 The bar (Tian's answer, section 10)

**The runs.** `run.py pipeline/configs/main.toml r028_final.toml r028_preshift.toml --profile tracking --seeds 3 --truth
both --drop-match-only`, then the same on `--profile context`.
- `main.toml` must reproduce 0.5363 raw and 0.5764 coarse on `tracking` to the last digit (R024). Otherwise the lane
  **stops for the advisor**.
- `--drop-match-only` removes Smart Tagging from all three configs, as always. The fill is not an evidence source and
  stays.

**Declared: (ii), as R027** (Tian's answer 1):
- On `tracking`, the objective's pooled Δ is at least +0.005, and no match's Δ is below −0.005.
- **Vetoes:** neither may be below −0.005:
  - the objective's Δ on `context`;
  - the pooled `tracking` alive/dead F1 minus `main.toml`'s.
- **Reported:** each truth's Δ, the sign-flip p, the possessions per graded minute and the fill's W1 and W2.

### 7.4 BAR-ATH, label-free, before adopting

- **The alignment** found by the run: d_1 and d_2 within ±2 frames of 4,014 and 26,919, and live frames at least 0.85 of
  the valid graded frames. Otherwise **STOP** for the advisor: the real fill would not be the one the stand-in copied.
- **The independent check in our gaps.**
  - **The teacher:** the house's form of the touch rule, `conventions.TOUCH` (R010's rule with confidence 0.99 and hold
    0.5). Its `home` is a number exactly on the kept episodes, so no hold vote is counted (the builder check's question
    3). R017 measured Smart Tagging this way (`tag_stats.py`), and R021 and R027 used it as a teacher.
  - **The frames:** the graded frames where the teacher speaks on the filled inputs and not on the plain ones.
  - **The measure:** `tag_stats.team_agreement` of the teacher's team with Smart Tagging's team votes, on the frames
    where a tag votes.
  - **The bar:** at least the agreement on the teacher's frames on the plain inputs, minus 0.05. With fewer than 300
    fill-only frames that a tag votes on, it is reported and decides nothing. A failure is **DROP**.
  - Smart Tagging is in neither the fill nor the touch rule, so it is the one independent view of our gaps.
- **Determinism and sanity** (R024's): valid and the same sha256 twice; no `X` in a graded block; `D` within [30%, 62%]
  of the graded frames. A failure is **DROP**.
- **Reported:**
  - the filled frames, by our decoded state;
  - W3's share on BAR-ATH;
  - the possessions per half, with a WARN beyond ±25% of `main.toml`'s (T33);
  - the graded frames relabelled (`label_diff.py`);
  - the run's added time.

### 7.5 The one retry: R028b (Tian's answer, section 10)

**Declared (Tian's answer 2).** R028b runs only if rung 1 fails the bar or a veto while its `tracking` objective Δ is
above 0: there is a gain to protect, and the likely harm is wrong touches from mislocated balls.
- **The single change:** `controlled_touch` gains `filled_confidence`, the vote's confidence on frames whose ball was
  filled. It defaults to `confidence`, so `main.toml` is unchanged, and R028b sets it to **0.79**.
- **Where 0.79 comes from:** near our gaps, 67.4% of the broadcast's balls are within 2 m of ours (A20 part 6). A
  mislocated ball gives a coin flip for the team, so 0.674 × 0.93 + 0.326 × 0.5 = 0.79.
- **What it goes through:** its own part 1b, then the same 7.1 to 7.4.

If R028b fails anywhere, the lane is **DROPPED**. No other retry is pre-approved.

### 7.6 The public window

After the decision only, as a correctness check (T01, T27).

## 8. The decision flow (runs without Tian; the advisor takes what it does not settle)

```
code + checks → task reviews → R028 part 1 (with Tian's answers) → independent review → commit
  → POOLS on BAR-ATH (heavy, once) → FIDELITY on tracking (+ context, reported)
        a gate or check fails: an implementation error → fix once, rerun; otherwise → DROP
  → SHIFT lag (r028_preshift) → BAR runs (tracking, then context: main, r028_final, r028_preshift)
        main.toml not reproduced → STOP (advisor)
        bar and vetoes pass → BAR-ATH checks
              alignment outside ±2 frames or live < 0.85 → STOP (advisor)
              a check fails → DROP
              all pass → ADOPT
        bar or a veto fails, tracking Δ > 0, R028b not run → R028b (part 1b, then FIDELITY, SHIFT, BAR, BAR-ATH)
        otherwise → DROP
```

**The errors rule** (as R027's):
- an exit 2, a crash or a log without a valid EXIT line is not a verdict;
- a clear cause outside the declared code: fix it and rerun once;
- a reaped run waits for Tian;
- a traceback in the declared code goes to the advisor.

**ADOPT.**
- `main.toml` becomes the final candidate, kept as `pipeline/configs/r028_broadcast_ball.toml`.
- `main.csv` is regenerated twice with the same sha256 and logged in the submission log. It is **not uploaded**.
- The suites, `selftest.py --data`, a `--no-ff` merge, and the push.
- Docs:
  - R028 part 2;
  - the scoreboard;
  - P15;
  - both technical documents (the broadcast is now an asset used, and regenerating needs `data/broadcast/`);
  - CLAUDE.md's broadcast lines ("no mapping provided" becomes "aligned by code, R028");
  - the site.

**DROP.**
- The fill, the schema key, the simulation, the configs and their checks are removed in one commit that names the
  commit to recover them from. R028 keeps its results.
- `main.toml` is never touched.

**The advisor** (an opus agent standing in for Tian, every ruling logged) decides only:
- the two STOPs;
- a crash or a reaped run whose cause is not clear;
- anything the flow does not foresee.

## 9. Cost

- **Code:** about a day, in two parallel worktrees (the pipeline side, 2.x; the benchmark side, 3.x), with task
  reviews.
- **Heavy runs,** one at a time and throttled:
  - pools a few minutes;
  - fidelity about 30 min (a driver decode of each of 21 copies);
  - lag about 25 min;
  - bar about 60 min per profile (three configs, two of them with a driver decode);
  - BAR-ATH a few minutes.
- **End:** about two and a half days, before the freeze (13 Nov). R028b would add about a day.

## 10. Tian's answers (2026-11-01, before implementation)

1. **The bar:** (ii), the size bar with a no-harm veto, declared before any run.
   - On `tracking`, the objective's pooled Δ is at least +0.005 and no match's Δ is below −0.005.
   - The vetoes are the `context` objective and the `tracking` alive/dead F1, neither below −0.005.
   - The sign-flip p is reported and decides nothing.
2. **The retry:** R028b, and only if rung 1 fails while its `tracking` objective Δ is above 0. The single change is
   `filled_confidence` 0.79. No other retry is pre-approved.
3. **After the fidelity gates:** a pass continues to the lag, the bar and the BAR-ATH checks automatically.
4. **Adoption:** if everything passes, `main.toml` is switched automatically. The CSV is not uploaded, and regenerating
   it then needs `data/broadcast/`.

Tian did not rule on the driver (3.2: the decoder's labels, not the truth). It stands as the designer's call, and
Tian may overturn it before part 1 is committed.

## Appendix: the builder check's questions and where they went

A fresh reviewer read this design as a builder would, and asked four questions:

1. **Are the bar and the retry declared or only proposed?** That is Tian's call, asked before implementation. The
   answers are in section 10, and part 1 copies them.
2. **Which labels feed the pools, and what happens after an adoption?** They are `r024_both.toml`'s, decoded
   in-process, with their sha256 checked against `e6496a9b…` (3.3, R027's way). The report field is shared by every
   config of a run, so that `choose.py` can compare `main` with the candidate.
3. **Does the tag check count the hold's votes?** No: the teacher form `conventions.TOUCH`, on kept episodes only (7.4).
4. **Who rules a failed fidelity gate an implementation error?** The advisor, by the code against the text, never by the
   number. A fix may only restore the text (3.5).

Settled by the builder from the code, and kept as they read:
- the match index is `degrade.index_of`;
- the lag step's plumbing goes through `train_dead_ball.served_inputs`;
- `MatchInputs` becomes per config for a config that fills;
- F5's "ours" is the near-gap pool's median.
