# [Team name] — More than a Hack 2027, Phase 1: technical document

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `[team name].csv`, the file `pipeline/configs/main.toml` writes, sha256
  `7ba575c0982aacb9765b39c406e4ff5c9ddb82ca37cfa2b7f101417431fc3d7c` (R017)
- **Full record:** [`technical_document.md`](technical_document.md). IDs: A.. and R.. in [`results.md`](results.md),
  T.. in [`traps.md`](traps.md), P.. in [`proposals.md`](proposals.md), all under `docs/`.

## 1. Approach

**The problem as measured** (A01). The ball is missing on 40.7% of the graded frames, in 150 gaps with a median of
10.7 s. With the ball visible, the nearest player's team is right on only 77.8% of the in-play frames, and the ball is
visible on 66% of the public window's dead frames, so a missing ball is not a dead ball. The organisers' baselines
fragment: the unsmoothed one reports 131 possessions in the public window against 17 (R000). The question is how to
hold a team through long stretches without a ball and change it only on real evidence.

**The pipeline** (`pipeline/README.md`): inputs → evidence → decode → refine → output. Each *evidence* source gives per
frame P(ball in play) (`alive`) and P(home has it | in play) (`home`), or no opinion. A *decoder* turns them into labels
per graded half, a *refiner* adjusts the edges, and the *output* stage writes the CSV and runs the organisers' validator
on it. The organisers' baseline runs through the same code and is rebuilt byte for byte as a check (R001).

**The decoder** (P06) is a three-state hidden Markov model over `H`, `A` and `D`, decoded by Viterbi. Each source
adds its log-likelihood to the states it speaks for; silence adds nothing. A change of team costs 4 nats, a change into
or out of `D` 32, a run of `H` or `A` lasts at least 35 frames (1.4 s) and a run of `D` at least 25 (R010, R013). A
possession is then an interval whose two edges must be paid for, which is what spellF1 scores.

**Four sources:**
- `controlled_touch` (P04), `home`: a control rule where the ball is seen (section 4), voting 0.93 inside a control
  episode and, while the ball is visible, 0.60 for the last controlling team outside one (R013).
- `dead_ball_model` (P05), `alive`: boosted trees on 30 play features (the ball against the lines and restart spots, the
  players' collective slowdown, kick-off and set-piece shapes), trained on DFL's `BallStatus` (R009).
- `shape_possession` (P07), `home`: boosted trees on 26 features of how the two teams stand and move, none reading the
  ball, trained on DFL's `BallPossession`, symmetric in the two teams, weight 0.2 on every graded frame (R011, R013).
- `smart_tagging` (P08), `home`: the automatic phase tags as team votes, confidence 0.76, weight 0.2 (R017).

After the decoder, `shift_edges` (P02) moves a change of team 1 frame later and a change into `D` 2 frames earlier: the
pipeline's own median lag on the benchmark, negated (R012).

**Why.** No labels of this match may be used, so every learned part is trained on other matches and every choice is
made on a benchmark of other matches (section 5). Boosted trees on hand-built features suit that: they train on a CPU,
are committed as JSON and read with numpy, and each feature can be checked against how our tracking differs from the
training data. A sequence model, PathCRF's published checkpoint (Kim et al. 2026), did not beat P07 on our degraded
tracking (R022). Each adopted step on the benchmark (pooled spellF1, seven DFL matches degraded as `tracking` v3, three
seeds, two truths):

| step | entry | raw | coarse |
|---|---|---|---|
| organisers' `smooth50` decoded per half (yardstick) | R004c | 0.2018 | 0.1842 |
| organisers' `smooth125` (yardstick) | R004c | 0.1665 | 0.2226 |
| touch rule + dead-ball model + HMM | R010 | 0.4217 | 0.4556 |
| + shape model | R011 | 0.5159 | 0.5591 |
| + edge shift | R012 | 0.5164 | 0.5592 |
| touch rule and decoder re-chosen | R013 | 0.5430 | 0.5593 |
| + Smart Tagging (DFL has no tags: run without it) | R017 | 0.5430 | 0.5593 |

## 2. Assets used and dropped

| asset | used | how, or why not |
|---|---|---|
| Tactical ATD | yes | Every source. Speeds from successive positions, never the speed channel (T31). |
| Smart Tagging | team votes | Agreement 0.761 with our touch-rule pseudo-labels, bar 0.75; the dead votes failed (0.409, bar 0.60) and were removed (R017). |
| Tactical homography | no | Analysed only: the camera's aim has a median along-pitch error of 12.1 m (A03). P09 dropped (below). |
| Tactical video, audio | no | The audio was decoded locally for P10, dropped. No LLM saw video or audio. |
| Broadcast assets | no | Own 164,488-frame timeline, no mapping to the tactical frames (A04). |
| Public 5-minute GT | check only | Section 5. |
| DFL/IDSSE open data | yes | Trains P05 and P07; the validation benchmark. |

**Tried and dropped**, each by a bar declared before its run:
- **P17, the restarting team** (R014, R014b): +0.0040 raw at best (p 0.0938), bar +0.005 with p ≤ 0.05.
- **P18, DFL's out-of-play ball hidden** (R015, R015b): failed the median-gap gate (15.04 s against our 10.74 s); the
  second rule could not reach our share past a line (0.258 to 0.262 against 0.0407).
- **P12, free `X`** (R016): `X` where the decoder's posterior is low; macroF1 fell at every threshold (raw -0.0003 to
  -0.0008).
- **P10, the referee's whistle** (R018, R018b): detections 1.52, then 1.51 times as likely near a stoppage as by chance,
  against the declared 2.
- **P09, the camera's aim** (R019): the simulated camera on DFL failed two of four fidelity gates (peak correlation
  0.9115 against our 0.843; dead/alive speed ratio 0.575 against our 0.128).
- **P22(a), a label-shift prior on P07** (R020): EM's home share (Saerens et al. 2002) missed the truth by 0.0538 on
  average against 0.0623 for guessing 0.5, a ratio of 0.863 where 0.75 was needed.
- **P21, wrong-team slot repair** (A16): worth it only if such slots cluster at the ball; a team is over 11 on 0.1097
  of the touch candidates against 0.1286 of all graded frames, a ratio of 0.85 where 1.5 was needed.
- **P20 step 0, PathCRF** (R022): on the one benchmark match it never trained on, +0.0049 over P07 on the gap frames
  where +0.04 was needed.
- **P19, a learned team model where the ball is seen** (R021): out-of-fold per-frame accuracy +0.016 over `main.toml`'s
  labels where +0.020 was needed in 6 of the 7 matches; it reached +0.020 in 3.

Not built: pass events (P11), more labelled tracking (P13), the ball from the video (P14), the broadcast (P15).

### Labelled data

We labelled nothing by hand, on any game. Two labelled datasets were used, both of other games:

| dataset | source, licence | who labelled it, how | how much | used for |
|---|---|---|---|---|
| DFL/IDSSE `BallStatus` (alive or dead) and `BallPossession` (the team), every frame | Bassek et al. 2025, figshare, CC BY 4.0 | The provider, DFL, in its official match data. The paper does not say whether a person or the system sets the flags, and says all data are visually controlled by live operators and quality-checked after the game: we treat them as human-checked labels. | 7 full matches (2 Bundesliga, 5 2. Bundesliga, R002), 25 Hz, 668.4 graded minutes (A06). Rows: 601,602 for P05 (R009); 331,881 for P07, 663,762 with team-swapped mirrors (R011). | Training P05 and P07 (and the dropped P19); validation (the benchmark's truth); the pipeline's lag (R012) |
| DFL/IDSSE restart events | the same release | DFL's trained human operators | 681 events (A12) | A diagnostic of P17 only; nothing trained or chosen from it |

**About this match.** No frame was hand-labelled, no tracking corrected by hand, and no human-produced data about it
used. The public 5-minute GT only checked the pipeline: alignment, a valid and deterministic file, each source's
direction. Nothing was trained, set or chosen on it (section 5). Smart Tagging is automatic, without manual review
(FAQ.md); its one constant measured here, 0.76, is its agreement with our touch-rule pseudo-labels, which an algorithm
made and no person decided (R017). We degrade the DFL tracking and say so, as CC BY 4.0 asks; `sources.py` accepts a
DFL file only if its size and md5 match the pinned values.

## 3. Missing and noisy tracking

Measured on our match (aggregates only), and what the pipeline does about each.

- **Long ball gaps.** Gaps under 2 s hold 0.1% of the missing frames, the p90 gap is 34 s (A01), and coverage falls
  from 12% missing in the first five graded minutes to 80% in a later block (A02). *Response:* no interpolation. The
  touch rule is silent without the ball and the decoder keeps its state. The shape model reads the players on every
  frame: out of fold on DFL it is right on 0.8349 of the alive frames without a ball (log-loss 0.3759 against 0.6931
  for a coin, R011). The tag votes cover 0.499 of the graded frames without a ball (R017).
- **Gaps start where stoppages start.** 28.6% of gap onsets lie within 2 m of a line against 8.5% of visible frames, on
  slow balls (A08). *Response:* the dead-ball model reads where the ball vanished and reappeared, the players' slowdown
  and set-piece shapes, and never splits on whether the ball is missing (R009). `D` never comes from ball presence.
- **The ball-speed channel** is NaN on 92.2% of graded frames with a ball and otherwise behaves like a player's speed
  (T31, A07). *Response:* every speed comes from successive positions.
- **The nearest player is not the possessor** (77.8% right, A01; 77.5% on DFL, A06). *Response:* the touch rule also
  needs no opponent close and the ball moving with the player.
- **Slots are tracklets.** 46 slots for 22 players, 20.52 present per frame, 1,144 tracklets with a median of 41.3 s
  (A07, A09). *Response:* no feature follows a slot over time; shape features are per-frame team aggregates.
- **Wrong-team slots.** A team shows more than 11 slots on 10.8% of graded frames (17,156 of 158,857), a lower bound
  (T32, A07). They show no sign of clustering at the ball (A16), so no repair was built.
- **The benchmark imitates this.** `tracking` v3 gives the DFL matches our ball-gap masks, tracklet lengths, wrong-team
  slots, NaN-speed runs and low-count frames. All eight fidelity gates pass, e.g. ball missing 0.4189 against our 0.4110
  (R004c). The whole player degradation costs `main.toml` 0.0129 raw (A14).

## 4. Possession rules implemented

Some answers to CHALLENGE.md are explicit rules, some follow from the decoder's costs, and some are implemented by
nothing.

**Explicit rules:**
- **Control.** A team controls the ball when the nearest tracked player of either team is its own and within 1.5 m, no
  opponent is within 1.0 m (the duel zone of Vidal-Codina et al. 2022), and the ball moves with that player (relative
  speed ≤ 6 m/s): a proximity rule, as in Link and Hoernig (2017). A run of such frames is an episode and votes for its
  team at 0.93, the floor of the episodes' agreement with DFL's flag; the constants come from a declared 192-config
  grid (R013). The vote starts on the episode's first frame, so a change lands where control began.
- **Loose balls.** While the ball is visible outside an episode (a pass in flight, a rebound, a duel), the rule votes
  0.60 for the last controlling team; without the ball it is silent. Possession stays with the last team through loose
  and contested phases, as CHALLENGE.md asks, unless shape or tags outweigh the hold.
- **Dead ball.** P(alive) is the model's. Before each half's documented kick-off a graded frame is `D` (2,934 frames
  before the second half's, A07), from the organisers' documented kick-offs, not from any ground truth.
- **Edge timing.** The shift of section 1.

**Implied by the decoder, with no rule of their own:**
- **Touches, deflections, challenges.** One control frame is an episode, so only the decoder stops a touch or deflection
  from becoming a possession: a change of team costs 4 nats each way and the new run must last 35 frames, so a touch of
  a frame or two changes nothing unless the evidence around it agrees. A challenge with opponents within 1 m gives no
  episode, so the hold continues. There is no "few seconds" or "two passes" test; the 35-frame minimum and the switch
  cost, chosen on DFL's flag (R013), are the only persistence requirement.
- **After a restart.** The hold favours the team that had the ball before the stoppage until a new episode names a team;
  shape and tags give their own view.
- **A change inside a ball gap.** The touch rule is silent there, so the team changes where the shape and tag evidence
  for the other team pays the switch cost.

**Implemented by none:**
- **Aerial duels.** No ball height: nothing tells winning a header from controlling the ball after it.
- **The restarting team by restart type** (corner to the attackers, goal kick to the defenders): P17, dropped.
- **Passes, receptions, interceptions** as events (P11): a completed pass is only a new episode of the same team.
- **Abstention.** No `X` on a graded frame; `X` only where nothing is graded. Free `X` was dropped (R016).
- **Dead-ball ownership.** Not asked: `D`.

## 5. Validation

**Not the public window.** The organisers' rule makes it a correctness check, "not training data and not a tuning set".
It would also mislead: 17 possessions, so one moves spellF1 by about 0.06; the best-covered block (A02); and it ranks
`smooth125` above `smooth50`, the reverse of the full match (T01). Today's file scores macroF1 0.6986 and spellF1 0.4000
on it (R017), figures that chose nothing.

**The benchmark** (P01): seven DFL/IDSSE matches, the truth built as ours is (`D` where `BallStatus` is dead, else
`BallPossession`'s team), served to our components with no truth reachable and graded by the organisers' scorer per
half. The truth served as a prediction scores 1.0000 on all seven (R002). The degradation was fitted on label-free
statistics of our tracking and frozen, never tuned towards a score.

**Two truths.** DFL's flag gives 4.58 possessions per graded minute. Our grain is unknown, so every choice is also
ranked under a coarse truth that absorbs possessions under 5 s between two of the other team, 2.48 a minute (A06).
Disagreement is a warning (T33).

**Statistics.** Pooled spellF1; each change against its reference by the seven matches' paired deltas: a 90%
t-interval, an exact sign-flip test (smallest possible p 0.0156) and matches won. Three seeds, since one seed moves a
pooled score by up to about 0.015 (R007).

**Every adopted step met a bar declared before its runs:** R010 +0.2199 raw over `smooth50` and +0.2330 coarse over
`smooth125`, 7 wins in 7 each; R011 +0.0942 raw, 7 in 7; R013 +0.0267 raw (p 0.0156, 7 in 7), level coarse (+0.0001).
R012's shift, +0.0005 raw (p 0.5000), was adopted because its rule says so, not because it helps.

**No constant comes from the public window.** The full record's section 5 traces each constant of `main.toml` to its
entry. The two once measured on the window (the nearest player's 0.78, `ball_presence`'s values) were never in
`main.toml` (A01).

**On our match, without labels:** `D` 35.93% of the graded frames, 413 possessions, no `X` in a graded block, the same
sha256 twice (R017). The shape model agrees with the touch episodes on 0.8251 of 41,052 frames, above its declared gate
of 0.7486 (R011).

**Where the remaining score is** (A13): a perfect team where the ball is seen would add +0.1942 raw, in the gaps
+0.1764, a perfect dead ball only +0.0290. What is left is who has the ball, which is why P19 to P22 attacked it.

## 6. Limitations

- **Our truth's grain is unknown** (T33). DFL's flag gives 437 possessions a match, median 3.7 s (A06). The public
  window cannot say how ours compares: it is alive on 80.4% of its frames, livelier than 96% of DFL's 5-minute windows.
  R013 gained +0.0267 raw and +0.0001 coarse: if our truth is coarse, it may not help.
- **The benchmark overstates the dead ball** (T38, T39). DFL tracks the ball out of play and ours loses it near the
  lines (beyond a line on 17.68% of DFL's at-risk rows against 4.07% of ours, R009), so P05's 0.9725 alive/dead F1 is an
  upper bound; our predicted `D` share is 36.0% against 44.6% on the benchmark (R010). The transplanted gaps ignore the
  play, so features about where gaps fall go unvalued, and hiding DFL's out-of-play ball failed (P18).
- **Transfer to LaLiga.** Both models learned from seven German matches, five of them one club's home games. The shape
  model's transfer gate passes, but it lowers the possession count by 2.4% on our match where it raises it by 16% on the
  benchmark (R011). The 1.5 m touch rule finds control on 36.9% of our ball-visible frames against 43.6% for the 2.0 m
  rule it replaced (R013); what that costs here is not measured.
- **Smart Tagging's weight is untested.** No benchmark has tags. Weight 0.2 and the per-frame spread were Tian's pick of
  two declared options; the source changes 2.28% of graded frames, 56.3% of them in gaps, with no score measured (R017).
- **Seven matches**, not independent; R013 selected from 192 and then 144 configs on them, though its
  leave-one-match-out estimates agree.
- **Timing.** On the public window our truth flips a median 24 frames (0.96 s) before the new team's first controlled
  touch, where DFL's flag flips 1 frame after it (A11). Nothing may be fitted to that, so the shift is DFL's lag alone.

**With more time:** our provider's grain and timing from its other games (P16); edges from pass events (P11); PathCRF
retrained on degraded copies (P20 step 1); the ball from the video (P14).

## 7. Reproduction

```bash
py -3.11 -m venv .venv                         # macOS/Linux: python3.11 -m venv .venv, then .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ as data/README.md describes
.venv/Scripts/python pipeline/run.py pipeline/configs/main.toml       # writes evaluation/results/main.csv
.venv/Scripts/python evaluation/validate_submission.py evaluation/results/main.csv
```

- **One command** from `data/` to the CSV, written with LF line endings and validated by the official checks. Expected
  sha256 `7ba575c0…` (R017), the same on two runs (R017).
- **Pins:** Python 3.11.9; every package the pipeline imports at an exact version in `requirements-pipeline.txt` (numpy
  2.4.6, pandas 3.0.6). Both models are JSON under `pipeline/models/`, LF by `.gitattributes`, read with numpy only.
- **`--verify`:** with `requirements-benchmark.txt` and the DFL files (`pipeline/benchmark/sources.py --verify`),
  `train_dead_ball.py --model hgb --verify` and `train_shape.py --verify` retrain each model and confirm the committed
  file (exit 0, R009, R011).
- **Benchmark:** `pipeline/benchmark/run.py pipeline/configs/main.toml --drop-match-only --truth both` gives 0.5430 raw,
  0.5593 coarse (R017); `pipeline/selftest.py --data` rebuilds the organisers' four baselines byte for byte.
- [A17: reproduction from a fresh clone, filled in by the controller]

## References

- Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in
  elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- Kim, H. et al. (2026). PathCRF. KDD 2026, arXiv 2602.12080.
- Link, D. and Hoernig, M. (2017). *PLOS ONE*, PMC5503225.
- Saerens, M., Latinne, P. and Decaestecker, C. (2002). *Neural Computation* 14(1).
- Vidal-Codina, F. et al. (2022). *Sports Engineering*, arXiv 2202.00804.
