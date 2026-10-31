# [Team name] — More than a Hack 2027, Phase 1: technical document

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `[team name].csv`, the file `pipeline/configs/main.toml` writes, sha256
  `e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170` (R024)
- **IDs** refer to files in the repository: A.. (analyses) and R.. (experiments) in docs/results.md, T.. (pitfalls) in
  docs/traps.md, P.. (proposals) in docs/proposals.md. The full record of every choice is docs/technical_document.md.

**Terms.** The *benchmark* is seven open DFL/IDSSE matches of other games, their tracking degraded to look like ours
(our ball-gap pattern, tracklet lengths, wrong-team slots). Its *fidelity gates* are statistics the copy must match ours
within tolerances fixed in advance. *Pooled spellF1* is the official scorer's spellF1 over all seven matches' halves as
one file, scored against a *raw* and a *coarse* truth (section 5).

## 1. Approach

**The problem.** The ball is missing on 40.7% of the graded frames, in 150 gaps with a median of 10.7 s (A01); the
nearest player is a poor guess at the possessor, and a missing ball is not a dead ball (section 3). The organisers'
baselines fragment: the unsmoothed one reports 131 possessions in the public window against 17 (R000). The question is
how to hold a team through long stretches without a ball and change it only on real evidence.

**The pipeline** (pipeline/README.md): inputs → evidence → decode → refine → output. Each *evidence* source gives per
frame P(ball in play) (`alive`) and P(home has it | in play) (`home`), or no opinion. A *decoder* turns them into labels
per graded half, a *refiner* adjusts the edges, and the *output* stage writes and validates the CSV.

**The decoder** (P06) is a three-state hidden Markov model over `H`, `A` and `D`, decoded by Viterbi. Each source
adds its log-likelihood to the states it speaks for; silence adds nothing. A change of team costs 4 nats (units of
log-likelihood), a change into or out of `D` 32, a run of `H` or `A` lasts at least 35 frames (1.4 s) and a run of `D`
at least 25 (R010, R013). A possession is then an interval whose two edges must be paid for, which is what spellF1
scores.

**Four sources:** `controlled_touch` (P04, `home`), a control rule where the ball is seen (section 4);
`dead_ball_model` (P05, `alive`), boosted trees on 30 play features (R009); `shape_possession` (P07, `home`), boosted
trees on 26 features of how the two teams stand and move, none reading the ball, weight 0.2 (R011, R013); and
`smart_tagging` (P08, `home`), the automatic phase tags as team votes, weight 0.2 (R017). After the decoder,
`shift_edges` (P02) moves a change of team 1 frame later and a change into `D` 2 frames earlier: the pipeline's own
median lag on the benchmark, negated (R012).

**Why.** No labels of this match may be used, so every learned part is trained on other matches. Every choice is made
on the benchmark (section 5), except Smart Tagging's strength (section 6). Boosted trees on hand-built features suit
that: they train on a CPU, and each feature can be checked against how our tracking differs from the training data.

## 2. Assets used and dropped

| asset | used | how, or why not |
|---|---|---|
| Tactical ATD | yes | Every source. Speeds from successive positions, never the speed channel (T31). |
| Smart Tagging | team votes | The team votes passed their bar (section 5); the dead votes failed (0.409, bar 0.60) and were removed (R017). |
| Tactical homography | no | Analysed only: the camera's aim has a median along-pitch error of 12.1 m (A03). P09 dropped (below). |
| Tactical video, audio | no | The audio was decoded locally for P10, dropped. No LLM saw video or audio. |
| Broadcast assets | no | Own 164,488-frame timeline, no mapping to the tactical frames (A04). |
| Public 5-minute GT | check, inspection | See "About this match" below. |
| DFL/IDSSE open data | yes | Trains P05 and P07; the benchmark (below). |
| Metrica's sample games (events) | measurement | Our provider's grain (R023, section 5). |

**Tried and dropped**, each by a bar declared before its run:
- **P17, the restarting team** (R014, R014b): +0.0040 raw at best (p 0.0938), bar +0.005 with p ≤ 0.05.
- **P18, DFL's out-of-play ball hidden** (R015, R015b): a fidelity gate failed (median ball gap 15.04 s against our
  10.74 s), and a second rule could not reach our share of balls seen past a line.
- **P10, the referee's whistle** (R018, R018b): detections near our decoded stoppages were 1.52, then 1.51 times a
  control share, against the declared 2.
- **P09, the camera's aim** (R019): a simulated camera on DFL failed two of its four fidelity gates (peak correlation,
  dead/alive speed ratio).
- **P22(a), a label-shift prior on P07** (R020): EM's home share (Saerens et al. 2002) missed the truth by 0.0538 on
  average against 0.0623 for guessing 0.5, a ratio of 0.863 where 0.75 was needed.
- **P21, wrong-team slot repair** (A16): worth it only if such slots cluster at the ball. A team had over 11 slots 0.85
  times as often on the touch rule's candidate frames as on all graded frames, where 1.5 was needed (a measure that may
  partly reflect ball visibility).
- **P20 step 0, PathCRF** (R022): on J03WR9, the benchmark match it never trained on, it beat P07 on the gap frames by
  +0.0049 where +0.04 was needed.
- **P19, a learned team model where the ball is seen** (R021): out-of-fold per-frame accuracy +0.0160 over `main.toml`'s
  decoded labels, where +0.020 was needed pooled and in 6 of the 7 matches; 3 reached it.
- **P23 C, edges placed by the decoder's posterior** (R025, R025b). A19 found most of what spellF1 loses from IoU 0.3 to
  0.7 in edges on otherwise right possessions (0.070), but the posterior is four to five times too sharp: its median
  moved edges by 1 frame, and tempered, by 3 the wrong way (mean of the two truths -0.0010, then -0.0282).
- **P23 E, P07 weighted apart in the gaps** (R026): every heavier gap weight added possessions and lost on the mean
  (-0.010 to -0.020).

### Labelled data

We labelled nothing by hand, on any game. The labelled data used are all about other games:

| data | source, licence | who labelled it, how | how much | used for |
|---|---|---|---|---|
| DFL/IDSSE `BallStatus` (alive or dead) and `BallPossession` (the team), every frame | Bassek et al. 2025, figshare, CC BY 4.0 | DFL, in its official match data. The paper does not say whether a person or the system sets the flags, but says all data are visually controlled by live operators and checked after the game: we treat them as human-checked labels. | 7 full matches (2 Bundesliga, 5 2. Bundesliga, R002), 25 Hz, 668.4 graded minutes (A06). Rows: 601,602 for P05 (R009); 331,881 for P07, 663,762 with team-swapped mirrors (R011). | Training P05 and P07 (and P19, dropped, R021); the touch rule's confidence 0.93, measured against the flag (R013); validation, as the benchmark's truth; the pipeline's lag (R012) |
| DFL/IDSSE events | DFL's raw event files, the same release; for J03WR9, the copy synchronised in PathCRF's repository | DFL's data collectors, trained human operators according to the paper | 681 restart events over the 7 matches (A12); one match's events (R022) | A diagnostic of P17's restart types (A12); R022's correctness check of PathCRF on the clean J03WR9. Nothing trained or chosen from them |
| PathCRF checkpoint (a model trained on labels) | Kim et al. 2026, MPL-2.0 | Its authors, on possession derived from DFL's events: five of the seven matches for training, a sixth for validation | one checkpoint | R022's measurement only; dropped |
| Metrica's sample games' events | metrica-sports/sample-data at `e706dd5`; no licence, its README asks for acknowledgement (given here); fetched at run time, not redistributed | Metrica Sports, our ground truth's provider: event types, per its published definitions | 3 anonymised games, 7,300 events | R023's grain only; nothing trained |

**About this match.** No tracking was corrected by hand, and no human-produced data about the match were used. No
model was trained on the public 5-minute ground truth and no constant of `main.toml` was set on it. It was used in
three ways:
- **A correctness check** (alignment, a valid and deterministic file, each source's direction); section 5.
- **An inspection to describe the truth,** which RULES.md allows. The window figures quoted here (77.8% and 66%, A01;
  80.4%, A06; the timing of its changes, A11) describe it and set nothing.
- **Two inputs, both disclosed.** R009 chose the dead-ball source and its weight 1.0 in configs that all carried
  `nearest_player` at A01's window-derived 0.78 (not in `main.toml`). R013 re-chose the touch rule and the decoder
  without it, and declared why R009's choice stands: it rests on the out-of-fold log-loss (lower on 7 of 7 matches),
  which no other source enters, and weight 1.0 takes the probability as it is. R012's shift
  rule read the window as an input to g, a gap to our truth's convention; g came out 0 for every type and was retired
  under the organisers' rule of 23 Oct 2026.

Smart Tagging is automatic, without manual review (FAQ.md), and the pseudo-labels behind its constant 0.76 came from an
algorithm, decided by no person. Its strength (per frame, weight 0.2) was Tian's pick of two declared options, made
after seeing label-free aggregates of our own predictions (the `D` share, possessions per half, frames changed); no
score or label entered it (R017). We degrade the DFL tracking and say so, as CC BY 4.0 asks.

## 3. Missing and noisy tracking

- **Long ball gaps.** Gaps under 2 s hold 0.1% of the missing frames and the p90 gap is 34 s (A01); coverage falls
  through each half (section 6). *Response:* no interpolation. The decoder keeps its state, and the shape model and the
  tags speak: out of fold on DFL the shape model is right on 0.8349 of the alive frames without a ball (log-loss 0.3759
  against 0.6931 for a coin, R011), and the tag votes cover 0.499 of the graded frames without a ball (R017).
- **Gaps start near the lines and on slow balls more often than chance.** 28.6% of gap onsets lie within 2 m of a line
  against 8.5% of visible frames, and a hazard model places the gaps only coarsely (A08). *Response:* the dead-ball
  model reads where the ball vanished and reappeared, the players' slowdown and set-piece shapes, and never splits on
  whether the ball is missing (R009). The ball is visible on 66% of the public window's dead frames (A01), so `D` never
  comes from ball presence.
- **The ball-speed channel** is NaN on 92.2% of graded frames with a ball and otherwise behaves like a player's speed
  (T31, A07). *Response:* every speed comes from successive positions.
- **The nearest player is not the possessor** (right on 77.8% in the public window, A01; 77.5% on DFL, A06).
  *Response:* the touch rule's further conditions (section 4).
- **Slots are tracklets.** 46 slots for 22 players, 20.52 present per frame, 1,144 tracklets with a median of 41.3 s
  (A07, A09). *Response:* no feature follows a slot over time; shape features are per-frame team aggregates.
- **Wrong-team slots.** One team shows more than 11 slots while the other shows fewer on 10.8% of graded frames
  (17,156 of 158,857), a lower bound (T32, A07). They show no sign of clustering at the ball (A16, section 2).

## 4. Possession rules implemented

**Explicit rules:**
- **Control.** A team controls the ball when the nearest tracked player of either team is its own and within 1.5 m, no
  opponent is within 1.0 m (the duel zone of Vidal-Codina et al. 2022), and the ball moves with that player (relative
  speed ≤ 6 m/s): a proximity rule, as in Link and Hoernig (2017). A run of such frames is an episode and votes for its
  team at 0.93, the floor of the episodes' agreement with DFL's flag; the constants come from a declared 192-config
  grid (R013, R024). The vote starts on the episode's first frame, so a change lands where control began.
- **Loose balls.** While the ball is visible outside an episode (a pass in flight, a rebound, a duel), the rule votes
  0.55 for the last controlling team (R024); without the ball it is silent. Possession stays with the last team
  through loose and contested phases, as CHALLENGE.md asks, unless shape or tags outweigh the hold.
- **Dead ball.** P(alive) is the model's. Before each half's documented kick-off a graded frame is `D` (2,934 frames
  before the second half's, A07): a rule from the organisers' kick-offs, not from any ground truth.
- **Edge timing.** The shift of section 1 (R012).

**Implied by the decoder, with no rule of their own:**
- **Touches, deflections, challenges.** One control frame is an episode, so only the decoder's switch cost and 35-frame
  minimum (section 1) stop a touch of a frame or two from becoming a possession, unless the evidence around it agrees.
  A challenge with opponents within 1 m gives no episode, so the hold continues. There is no "few seconds" or "two
  passes" test; those two, chosen on DFL's flag (R013), are the only persistence requirement.
- **After a restart.** The hold favours the team that had the ball before the stoppage until a new episode names a team;
  shape and tags give their own view.
- **A change inside a ball gap** falls where the shape and tag evidence for the other team pays the switch cost.

**Implemented by none:**
- **Aerial duels.** No ball height: nothing tells winning a header from controlling the ball after it.
- **The restarting team by restart type** (corner to the attackers, goal kick to the defenders): P17, dropped.
- **Passes, receptions, interceptions** as events (P11): a completed pass is only a new episode of the same team.
- **Abstention.** No `X` on a graded frame; `X` only where nothing is graded (free `X`: P12, dropped).
- **Dead-ball ownership.** Not asked: `D`.

## 5. Validation

**Not the public window** (section 2 lists its uses). Beyond the rule, it would mislead: with 17 possessions one moves
spellF1 by about 0.06, it is the best-covered block (A02), and it ranks `smooth125` above `smooth50`, the reverse of the
full match (T01). Today's file scores macroF1 0.7007 and spellF1 0.4000 on it (R024), which chose nothing.

**The benchmark** (P01) builds the truth as ours is (`D` where `BallStatus` is dead, else `BallPossession`'s team);
our components see no truth, and the organisers' scorer grades each half. The truth served as a prediction scores
1.0000 on all seven (R002). The degradation was fitted on label-free statistics of our tracking and frozen, never tuned
towards a score. All eight fidelity gates pass, e.g. ball missing 0.4189 against our 0.4110 (R004c), and the whole
player degradation costs `main.toml` 0.0129 raw (A16, from A14's table).

**Two truths.** DFL's flag gives 4.58 possessions per graded minute. Our grain is unknown, so every choice is also
ranked under a coarse truth that absorbs possessions under 5 s between two of the other team: 2.48 a minute (R002; the
5 s rule was written before the numbers were seen, A06). Disagreement is a warning (T33). On Metrica's own sample
games, possession derived from their events has 6.20 possessions per alive minute (raw 8.30, coarse 4.50) and a median
of 4.72 s; R023's declared outcome, "both", ranks later choices on the mean of the two truths.

**Statistics.** Each change is compared with its reference through the seven matches' paired deltas: a 90%
t-interval, an exact sign-flip test (smallest possible p 0.0156) and matches won. Three seeds, since one moves a pooled
score by up to about 0.015 (R007).

**Every adopted step met a bar declared before its runs** (pooled spellF1 raw / coarse; the organisers'
baselines on the benchmark score `smooth50` 0.2018 / 0.1842 and `smooth125` 0.1665 / 0.2226, R004c):
- R010, touch rule + dead-ball model + HMM, 0.4217 / 0.4556: +0.2199 raw over `smooth50` and +0.2330 coarse over
  `smooth125`, 7 wins in 7 each.
- R011, + shape model, 0.5159 / 0.5591: +0.0942 raw, 7 in 7.
- R012, + edge shift, 0.5164 / 0.5592: +0.0005 raw (p 0.5000), adopted because its rule says so, not because it helps.
- R013, touch rule and decoder re-chosen, 0.5430 / 0.5593: +0.0267 raw (p 0.0156, 7 in 7), level coarse (+0.0001).
- R017, + Smart Tagging: the tag votes' agreement 0.761 against a bar of 0.75; their strength is untested (section 6).
- R024, R013's grids re-chosen on the mean of the two truths: only the hold moved (0.60 to 0.55), 0.5363 / 0.5764,
  mean +0.0052 (p 0.0469, 6 in 7), raw -0.0067.

**On our match, without labels:** `D` 36.00% of the graded frames, 399 possessions, no `X` in a graded block, the same
sha256 twice (R024). The shape model agrees with the touch episodes on 0.8251 of 41,052 frames, above its declared gate
of 0.7486 (R011).

**Where the remaining score is** (A13). On the benchmark a perfect team would add +0.1942 raw where the ball is seen
and +0.1764 in the gaps, a perfect dead ball only +0.0290 (the benchmark overstates the dead ball, section 6).

## 6. Limitations

- **Where it is weakest on our match.** Late in each half the ball is missing on up to 80% of a 5-minute block (A02).
  There the team rests on the shape model, trained on other teams, and on the tags. Where the ball is seen, the touch
  rule finds control on 36.9% of our frames, against 43.6% for the 2.0 m rule it replaced (R013); what the tighter rule
  loses here is not measured.
- **Our truth's grain is only partly known** (T33). DFL's flag gives 437 possessions a match, median 3.7 s, and the
  public window cannot say how ours compares: it is alive on 80.4% of its frames, livelier than 96% of DFL's 5-minute
  windows (A06). R023 places our provider between the two, but from events, not the flag our truth comes from. R024's
  gain is coarse (+0.0171) at a raw cost (-0.0067): if our truth is as fine as DFL's, it costs a little.
- **The benchmark overstates the dead ball** (T38, T39). DFL tracks the ball out of play, and ours loses it near the
  lines: among the at-risk rows (frames where a gap could begin) the ball is beyond a line on 17.68% of DFL's against
  4.07% of ours (R009). So P05's 0.9725 alive/dead F1 is optimistic for our match by an amount the benchmark cannot
  measure. Our predicted `D` share, 36.00% (section 5), is below the benchmark's 44.6% (R013). The transplanted gaps
  ignore the play, so features about where gaps fall go unvalued (P18 did not fix this).
- **Transfer to LaLiga, on seven matches.** Both models learned from seven German matches, five of them one club's
  home games, so not independent. The shape model's transfer gate passes, but it lowers the possession count by 2.4% on
  our match where it raises it by 16% on the benchmark (R011). R013 and R024 each selected from 336 configs on these
  matches; R024's rule holds out weaker than its gain (+0.0035, p 0.1562).
- **Smart Tagging's strength is untested** (Tian's pick, section 2). It changes 2.28% of the graded frames, 56.3% of
  them in gaps, and no score was measured (R017).
- **Timing.** On 10 of the public window's 12 team changes, our truth flips a median 0.96 s before the new team's first
  controlled touch, where DFL's flag flips 1 frame after it; the gap's 90% interval holds 0, so no difference is shown
  (A11), and nothing may be fitted to it.

**With more time:**
- learn our provider's switch timing from its other games (P16; its grain is R023's);
- place the edges from pass events (P11);
- retrain PathCRF on degraded copies (P20 step 1);
- recover the ball from the video (P14).

## 7. Reproduction

```bash
py -3.11 -m venv .venv                         # macOS/Linux: python3.11 -m venv .venv, then .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ as data/README.md describes
.venv/Scripts/python pipeline/run.py pipeline/configs/main.toml --out "../[team name].csv"
.venv/Scripts/python evaluation/validate_submission.py "../[team name].csv"
```

- **One command** goes from `data/` to the CSV, written with LF line endings and validated by the official checks
  (without `--out`: `evaluation/results/main.csv`).
- **Pins:** Python 3.11.9, and every package the pipeline imports at an exact version in `requirements-pipeline.txt`
  (numpy 2.4.6, pandas 3.0.6). Both models are JSON under `pipeline/models/`, kept LF by `.gitattributes` and read with
  numpy only.
- **The models and the benchmark** (these need `data/` as well, since the degradation is calibrated on our tracking):

```bash
.venv/Scripts/python -m pip install -r requirements-benchmark.txt
.venv/Scripts/python pipeline/benchmark/sources.py                    # fetch the DFL files; --verify only checks them
.venv/Scripts/python pipeline/benchmark/train_dead_ball.py --model hgb --verify  # retrains P05's model, compares (R009)
.venv/Scripts/python pipeline/benchmark/train_shape.py --verify                  # the same for P07's (R011)
.venv/Scripts/python pipeline/benchmark/run.py pipeline/configs/main.toml --drop-match-only --truth both
.venv/Scripts/python pipeline/selftest.py --data       # rebuilds the organisers' four baselines byte for byte
```

**From a fresh clone** (A17, A18). R024's file, `e6496a9b…`, came out the same on every run in our working repository
but has not yet been regenerated from a fresh clone or on a second machine. R017's file (`7ba575c0…`, hold 0.60; the
and the pipeline's code outside the benchmark and the configs is the same at R024's merge) has: the pushed repository (`5f0145c`), cloned into a new Python 3.11.9 venv with
only the raw bundle, gave it on three runs (one at the default BLAS threads), the self-tests passed, both models
retrained identical and the benchmark returned its scores to the last digit (AMD Ryzen 5 5500U, Windows 11, CPU only).
Under Linux (A18: WSL 2, the same laptop) the file was the same; another processor is untried.

## References

- Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in
  elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y. Data: figshare,
  doi:10.6084/m9.figshare.28196177 (CC BY 4.0).
- Kim, Lee, Seo, Ko, Yoon and Park (2026). PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference
  from Player Trajectories. KDD 2026, arXiv:2602.12080. Code: github.com/hyunsungkim-ds/pathcrf (MPL-2.0).
- Link and Hoernig (2017). Individual ball possession in soccer. *PLOS ONE*, PMC5503225.
- Metrica Sports. Sample data. github.com/metrica-sports/sample-data, commit `e706dd5`.
- Saerens, Latinne and Decaestecker (2002). Adjusting the outputs of a classifier to new a priori probabilities: a
  simple procedure. *Neural Computation* 14(1), 21-41. doi:10.1162/089976602753284446.
- Vidal-Codina et al. (2022). Automatic event detection in football using tracking data. *Sports Engineering*.
  arXiv:2202.00804.
