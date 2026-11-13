# Un Dia de Partit — More than a Hack 2027, Phase 1: technical document

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `Un Dia de Partit.csv`, the file `pipeline/configs/main.toml` writes, sha256
  `aae79a0446dabdf565d86b95f03acf0a53b5698cf3d80033c25bf7d49cbd3bdb` (R042)
- **IDs** refer to files in the repository: A.. (analyses) and R.. (experiments) in docs/results.md, T.. (pitfalls) in
  docs/traps.md, P.. (proposals) in docs/proposals.md. The full record of every choice is docs/technical_document.md.

**Terms.** The *benchmark* is seven open DFL/IDSSE matches of other games, their tracking degraded by a declared
profile, `generic` (section 5): the ball goes missing in long gaps and players drop out, at the two rates the organisers
publish, and nothing we measured on our match enters it. *Pooled spellF1* is the official scorer's spellF1 over all
seven matches' halves as one file, scored against a *raw* and a *coarse* truth (section 5). The profile used until
9 Nov 2026 was fitted to our own tracking; its *fidelity gates* were statistics that copy had to match ours (section 5).

## 1. Approach

**The problem.** The ball is missing on 40.7% of the graded frames, in 150 gaps with a median of 10.7 s (A01); the
nearest player is a poor guess at the possessor, and a missing ball is not a dead ball (section 3). The organisers'
baselines fragment: the unsmoothed one reports 131 possessions in the public window against 17 (R000). The question is
how to hold a team through long stretches without a ball and change it only on real evidence.

**The pipeline** (pipeline/README.md): inputs → evidence → decode → refine → output. Each *evidence* source gives per
frame P(ball in play) (`alive`) and P(home has it | in play) (`home`), or no opinion. A *decoder* turns them into labels
per graded half, a *refiner* adjusts the edges, and the *output* stage writes and validates the CSV.

**The decoder** (P06) is a three-state hidden Markov model over `H`, `A` and `D`, decoded by Viterbi. Each source
adds its log-likelihood to the states it speaks for; silence adds nothing. A change of team costs 6 nats (units of
log-likelihood), a change into or out of `D` 32, a run of `H` or `A` lasts at least 35 frames (1.4 s) and a run of `D`
at least 25 (R030). A possession is then an interval whose two edges must be paid for, which is what spellF1
scores.

**Four sources:** `controlled_touch` (P04, `home`), a control rule where the ball is seen (section 4);
`dead_ball_model` (P05, `alive`), boosted trees on 30 play features; `shape_possession` (P07 and P30, `home`),
boosted trees on 42 features of how the two teams stand and move, none reading the ball: version 1's 26 plus, per team,
its length, its spread and the change of its advance one-sided in time over 2, 5 and 10 s, so that a hand-over inside a
ball gap shows earlier (weight 0.3, R033); and `team_model` (P32, `home`), boosted trees on 72 features of the ball
against each team (distances, closing, time to reach it, control timing, direction towards each goal) plus the shape
features, only where the ball is seen, weight 0.3 (R037). After
the decoder, `shift_edges` (P02) moves a change of team 1 frame later and a change into `D` 2 frames earlier: the
pipeline's own median lag on the benchmark, negated (R012's rule, measured again in R030).

**Why.** The organisers' rules (their clarification of November 2026) allow training and tuning on external data only,
and their reply of 5 Nov adds that anything not explicitly allowed is not permitted. So both models are trained on
other matches, and every numeric constant is chosen on the benchmark, whose degradation measures nothing of our match
(R030, section 5). The architecture (which components exist, the trees' settings, the shape source's clip, the
decoder's structure) was inherited from our earlier benchmark, since retired, and re-verified on the new one. The
rules are also why the Smart Tagging is not used (section 2). Boosted trees on hand-built features suit them: they
train on a CPU, and each feature can be checked against how our tracking differs from the training data.

## 2. Assets used and dropped

| asset | used | how, or why not |
|---|---|---|
| Tactical ATD | yes | Every source. |
| Smart Tagging | no | Its team votes were a source from R017 to R029, then dropped for the rules (below, "About this match"); its dead votes failed their bar (0.409, bar 0.60, R017). |
| Tactical homography | no | Analysed only: the camera's aim has a median along-pitch error of 12.1 m (A03). |
| Tactical video, audio | no | Audio decoded locally for P10 only; no LLM saw video or audio. |
| Broadcast assets | no | Own 164,488-frame timeline, no mapping to the tactical frames (A04). |
| Public 5-minute GT | check, inspection | See "About this match" below. |
| DFL/IDSSE open data | yes | Trains P05 and P07; the benchmark (below). |
| SkillCorner open data | check only | MIT licence. The ball's gaps in three A-League matches were read once, after the benchmark's gap law was written; no number was taken from it (R030, section 5). |
| Metrica's sample games (events) | measurement, withdrawn | Our provider's grain (R023, section 5); used once, withdrawn on 5 Nov 2026: the data has no licence. |

**Tried and dropped**, each by a bar declared before its run (those measured on the benchmark used its retired
profile, section 5):
- **P17, the restarting team** (R014, R014b): +0.0040 raw at best (p 0.0938), bar +0.005 with p ≤ 0.05.
- **P18, DFL's out-of-play ball hidden** (R015, R015b): a fidelity gate failed (median ball gap 15.04 s against our
  10.74 s), and a second rule could not reach our share of balls seen past a line.
- **P10, the referee's whistle** (R018, R018b): detections near our decoded stoppages were 1.52, then 1.51 times a
  control share, against the declared 2.
- **P09, the camera's aim** (R019): a simulated camera on DFL failed two of its four fidelity gates (peak correlation,
  dead/alive speed ratio).
- **P22(a), a label-shift prior on P07** (R020): EM's home share (Saerens et al. 2002) missed the truth by 0.0538 on
  average against 0.0623 for guessing 0.5, a ratio of 0.863 where 0.75 was needed.
- **P21, wrong-team slot repair** (A16): worth it only if such slots cluster at the ball: a team had over 11 slots 0.85
  times as often on the touch rule's candidate frames as on all graded frames, where 1.5 was needed (which may partly
  reflect ball visibility).
- **P20 step 0, PathCRF** (R022): on J03WR9, the benchmark match it never trained on, it beat P07 on the gap frames by
  +0.0049 where +0.04 was needed.
- **P19, a learned team model where the ball is seen** (R021): out-of-fold per-frame accuracy +0.0160 over `main.toml`'s
  decoded labels, where +0.020 was needed pooled and in 6 of 7 matches; 3 reached it.
- **P23 C, edges placed by the decoder's posterior** (R025, R025b): most of what spellF1 loses from IoU 0.3 to 0.7 is
  edges on otherwise right possessions (0.070, A19), but the posterior is four to five times too sharp: its median
  moved edges 1 frame, and tempered, 3 the wrong way (mean of the two truths -0.0010, then -0.0282).
- **P23 E, P07 weighted apart in the gaps** (R026): every heavier gap weight added possessions and lost on the mean
  (-0.010 to -0.020).
- **P22(b), P07 recalibrated per half from the touch rule's labels** (R027): on DFL its log-loss in the other half's
  gaps rose 0.0061 where a 0.002 fall was needed; the touches' offset was not the gaps' (0.296 apart, label noise
  0.107).
- **The broadcast's ball in our gaps** (R028, P15): the broadcast was aligned by code, but its stand-in on DFL (a
  transplant of our real fill) failed its fidelity gates. The measured difference is in the gaps' states: under the
  decoder, 54% of DFL's served gap frames are alive against our 20%. That is our reading of why it failed. The fill was
  never scored, and the code was removed.

### Labelled data

We labelled nothing by hand, on any game. The labelled data used are all about other games:

| data | source, licence | who labelled it, how | how much | used for |
|---|---|---|---|---|
| DFL/IDSSE `BallStatus` (alive or dead) and `BallPossession` (the team), every frame | Bassek et al. 2025, figshare, CC BY 4.0 | DFL, in its official match data. The paper does not say whether a person or the system sets the flags, but says all data are visually controlled by live operators and checked after the game: we treat them as human-checked labels. | 7 full matches (2 Bundesliga, 5 2. Bundesliga, R002), 25 Hz, 668.4 graded minutes (A06). Rows: 601,602 for P05 (R009, R030); 331,881 for P07, 663,762 with team-swapped mirrors (R011). | Training P05 and P07 (and P19, dropped, R021); the touch rule's confidence 0.93, measured against the flag (R030); validation, as the benchmark's truth; the pipeline's lag (R030) |
| DFL/IDSSE events | DFL's raw event files, the same release; for J03WR9, the copy synchronised in PathCRF's repository | DFL's data collectors, trained human operators according to the paper | 681 restart events over the 7 matches (A12); one match's events (R022) | A diagnostic of P17's restart types (A12); R022's correctness check of PathCRF on the clean J03WR9. Nothing trained or chosen from them |
| PathCRF checkpoint (a model trained on labels) | Kim et al. 2026, MPL-2.0 | Its authors, on possession derived from DFL's events: five of the seven matches for training, a sixth for validation | one checkpoint | R022's measurement only; dropped |
| Metrica's sample games' events | metrica-sports/sample-data at `e706dd5`; no licence, its README asks for acknowledgement (given here); fetched at run time, not redistributed | Metrica Sports, our ground truth's provider: event types, per its published definitions | 3 anonymised games, 7,300 events | R023's grain only; nothing trained. Withdrawn on 5 Nov 2026: with no licence we cannot show the rights the organisers ask for |

**About this match.** No tracking was corrected by hand, and no human-produced data about the match were used. No
model was trained on the public 5-minute ground truth and no constant of `main.toml` was set on it. Its three uses:
- **A correctness check** (alignment, a valid and deterministic file, each source's direction); section 5.
- **An inspection to describe the truth,** which the rules allow ("inspecting the provided data ... to better
  understand what counts as possession"). The window figures quoted here (77.8% and 66%, A01; 80.4%, A06; the timing
  of its changes, A11) set nothing. No output of ours was checked against the video [team: confirm].
- **Two inputs, both disclosed, and neither stands.** R009 first chose the dead-ball source and its weight 1.0 in
  configs that all carried `nearest_player` at A01's window-derived 0.78 (never in `main.toml`). R030 chose the weight
  again, and kept the trees over the logistic reference, with no such source in any config. So no choice that stands
  had a window-derived value among its candidates. R012's shift rule read the window as an input to g, a gap to our
  truth's convention; g came out 0 for every type and was retired under the organisers' rule of 23 Oct 2026, and R030
  reads the shift from DFL's lag alone.

**The tracking's published figures.** The benchmark's profile quotes two statistics of this match from the
organisers' DATA.md (section 5). Until 9 Nov 2026 it was fitted on label-free statistics of our tracking instead. We
retired that profile after the organisers' reply of 5 Nov, and nothing in the submission stands on it.

**The Smart Tagging is not used.** It is automatic, without manual review (FAQ.md). From R017 to R029 its team votes
were a source. Their confidence, 0.76, was their agreement on this match with our own touch rule's output, which an
algorithm made and no person decided. Their strength (per frame, weight 0.2) was Tian's pick of two declared options,
made after seeing label-free aggregates of our own predictions. The organisers' clarification of November 2026 has
parameters tuned on external data only, and no external data has the tags. So R029 took the source out before
submission, and the submitted file does not read them. We degrade the DFL tracking and say so, as CC BY 4.0 asks.

## 3. Missing and noisy tracking

- **Long ball gaps.** Gaps under 2 s hold 0.1% of the missing frames and the p90 gap is 34 s (A01); coverage falls
  through each half (section 6). *Response:* no interpolation. The decoder keeps its state, and the shape model
  speaks: out of fold on DFL it is right on 0.8458 of the alive frames without a ball (log-loss 0.3560 against 0.6931
  for a coin, R030).
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
  team at 0.93, the floor of the episodes' agreement with DFL's flag (93.5%); the constants come from a declared
  192-config grid (R030). The vote starts on the episode's first frame, so a change lands where control began.
- **Loose balls.** While the ball is visible outside an episode (a pass in flight, a rebound, a duel), the rule votes
  0.60 for the last controlling team (R030); without the ball it is silent. Possession stays with the last team
  through loose and contested phases, as CHALLENGE.md asks, unless the shape evidence outweighs the hold.
- **Dead ball.** P(alive) is the model's. Before each half's documented kick-off a graded frame is `D` (2,934 frames
  before the second half's, A07): a rule from the organisers' kick-offs, not from any ground truth.
- **Edge timing.** The shift of section 1 (R012).

**Implied by the decoder, with no rule of their own:**
- **Touches, deflections, challenges.** One control frame is an episode, so only the decoder's switch cost and 35-frame
  minimum (section 1) stop a touch of a frame or two from becoming a possession, unless the evidence around it agrees.
  A challenge with opponents within 1 m gives no episode, so the hold continues. There is no "few seconds" or "two
  passes" test; those two, chosen on DFL's flag (R030), are the only persistence requirement.
- **After a restart.** The hold favours the team that had the ball before the stoppage until a new episode names a team;
  the shape source gives its own view.
- **A change inside a ball gap** falls where the shape evidence for the other team pays the switch cost.

**Implemented by none:**
- **Aerial duels.** No ball height: nothing tells winning a header from controlling the ball after it.
- **The restarting team by restart type** (corner to the attackers, goal kick to the defenders): P17, dropped.
- **Passes, receptions, interceptions** as events (P11): a completed pass is only a new episode of the same team.
- **Abstention.** No `X` on a graded frame; `X` only where nothing is graded (free `X`: P12, dropped).
- **Dead-ball ownership.** Not asked: `D`.

## 5. Validation

**Not the public window** (section 2 lists its uses). Beyond the rule, it would mislead: with 17 possessions one moves
spellF1 by about 0.06, it is the best-covered block (A02), and it ranks `smooth125` above `smooth50`, the reverse of the
full match (T01). Today's file scores macroF1 0.7199 and spellF1 0.4737 on it (R042; R030's file 0.7238 and 0.5029): a
check for gross errors, which chose nothing.

**The benchmark** (P01) builds the truth as ours is (`D` where `BallStatus` is dead, else `BallPossession`'s team);
our components see no truth, and the organisers' scorer grades each half. The truth served as a prediction scores
1.0000 on all seven (R002). The tracking is degraded by a declared profile, `generic` v1 (R030), and nothing we
measured on our match enters it. It quotes two figures from the organisers' DATA.md: the ball is missing on 41% of the
graded frames, and 20.5 objects are tracked on average, so each player is lost on 1 - 20.5 / 22 = 6.8% of the frames.
The rest was declared before any run: gap lengths log-uniform between 1 s and 60 s, placed by a Poisson process blind
to the play; no ball speed; players lost independently, in tracked spans log-uniform between 2 s and 300 s; a 7-frame
moving average; no wrong-team slots, NaN-speed runs or low-count frames. One point we state openly: the two figures are
statistics of this match, and no text allows quoting them by name. We read them as public information about the task,
which the organisers publish to every team, so quoting them puts none of the match's data into the benchmark.
A check on external data, made after the gap law
was written and taking no number from it: in SkillCorner's open data (MIT licence, three A-League matches) the ball is
undetected on 39% to 46% of the frames, in gaps with a median of 5.4 to 5.8 s (the declared law gives 7.7 s). Until
9 Nov 2026 the benchmark's profile was fitted on label-free statistics of our own tracking and checked by eight
fidelity gates (`tracking` v3, R004c). We retired it after the organisers' reply of 5 Nov ("anything that is not
explicitly allowed is not permitted"); `generic` has no such gate against our match, by design.

**Two truths.** DFL's flag gives 4.58 possessions per graded minute. Our grain is unknown, so every choice is also
ranked under a coarse truth that absorbs possessions under 5 s between two of the other team: 2.48 a minute (R002; the
5 s rule was written before the numbers were seen, A06). Disagreement is a warning (T33). Since R023, whose data is
now withdrawn (section 2), choices rank on the mean of the two truths' scores: the neutral ranking while neither is
known to be right (kept on 9 Nov). R030 chose every constant under it.

**Statistics.** Each change is compared with its reference through the seven matches' paired deltas: a 90%
t-interval, an exact sign-flip test (smallest possible p 0.0156) and matches won. Three seeds, since one moves a pooled
score by up to about 0.015 (R007).

**How each component was adopted, on the retired profile.** Every step met a bar declared before its runs. These
scores are history and nothing stands on them today (pooled spellF1 raw / coarse; the organisers' baselines scored
`smooth50` 0.2018 / 0.1842 and `smooth125` 0.1665 / 0.2226 there, R004c):
- R010, touch rule + dead-ball model + HMM, 0.4217 / 0.4556: +0.2199 raw over `smooth50`, +0.2330 coarse over
  `smooth125`, 7 wins in 7 each.
- R011, + shape model, 0.5159 / 0.5591: +0.0942 raw, 7 in 7.
- R012, + edge shift, 0.5164 / 0.5592: +0.0005 raw (p 0.5000), adopted because its rule says so, not because it helps.
- R013, touch rule and decoder re-chosen, 0.5430 / 0.5593: +0.0267 raw (p 0.0156, 7 in 7), level coarse (+0.0001).
- R017, + Smart Tagging (agreement on our match 0.761, bar 0.75; strength untested); taken out by R029 (section 2).
- R024, re-chosen on the mean of the two truths: the hold 0.60 to 0.55, 0.5363 / 0.5764, mean +0.0052 (p 0.0469).

**What stands today: R030's constants plus R042's two models, on `generic`** (seven matches, three seeds). Both models were retrained: the dead-ball trees
have a pooled held-out log-loss of 0.0788 (AUC 0.9942 to 0.9973 by match) and beat the logistic reference's 0.1524 on
7 of 7 matches; the shape model's is 0.3560 on alive frames without a ball (accuracy 0.8458). The touch rule's
confidence was measured again (93.5% agreement, so 0.93). Every constant was then chosen again by a rule declared
before the runs: the best mean of the two truths' scores and, among the configs within 0.005 of it, the one whose
worse truth scores higher. The touch rule kept 1.5 m / 6.0 m/s / 1 frame and its hold went from 0.55 to 0.60 (192
configs); `switch_team` went from 4 to 6 and the shape weight from 0.2 to 0.3, with `min_possession` 35 kept (144
configs); `switch_dead` 32, `min_dead` 25, the dead-ball weight 1.0, the shape source on every graded frame and the
shift +1 / -2 / 0 came back unchanged. Without the shape source the mean falls by 0.1037, to 0.4584.

**R042 (13 Nov 2026, Tian's decision) added two models on top.** P30's version 2 shape model beat R030's file by
+0.0143 on the mean of the two truths (0.5629 raw / 0.5901 coarse; all seven matches, p 0.016; +0.0131 on three fresh
seeds), and P32's team model beat P30 by a further +0.0162 (all seven matches, p 0.016; +0.0160 on fresh seeds):
**0.5891 raw / 0.5963 coarse (mean 0.5927)**, +0.0305 over R030's file (R033, R037). Their gains held under declared
wrong-team slots, but the slots that check realised were at most 2.3% of the graded frames, far fewer than ours (A27).
Where the ball is seen the edges of the possessions are within a few frames; the remaining error is at hand-overs
inside the ball's gaps (A26). Every other constant is R030's, and R030's rule gives the same shift with the new sources.

R030's final config scored **0.5414 raw / 0.5829 coarse (mean 0.5622)**; the constants it replaced scored 0.5388 / 0.5779
(mean 0.5584) on the same profile and models. The organisers' baselines decoded per half score 0.1885 / 0.1698 (`w50`)
and 0.1616 / 0.2092 (`w125`). R030's bar passed: +0.3529 raw over `w50` and +0.3738 coarse over `w125`, p 0.0156 and 7
wins in 7 each. Two of the choices were knife-edges, decided by the declared rule (section 6).

**On our match, pass or fail only (R042).** The checks were declared before the runs, and all pass. The file
validates, with no `X` in a graded block and the same sha256 on two runs, each from a rebuilt cache. Its `D` share is
34.4% of the graded frames, inside [30%, 62%], a band set on DFL (`H` 38.2%, `A` 27.4%; 386 possessions, against R030's
393). The version 2 shape model agrees with the touch episodes on 0.8317 of 41,052 frames; its gate is 0.7656 (R030's
version 1 model: 0.8199 against 0.7426). The team model has no such check on our match (section 6). On the public
window each source agrees with the truth on more than half of its frames and the macroF1 beats every constant file's.
docs/results.md, R042, has the rest.

**Where the remaining score is** (A25, on `generic`). A perfect team would add +0.1989 raw (+0.2151 coarse) where the
ball is seen and +0.1800 (+0.1035) in the gaps, a perfect dead ball only +0.0278 (+0.0389): what is left is who has
the ball. Each source earns its place: without the shape source the mean of the two truths falls by 0.1037, and
without the touch rule by 0.0304 (R030, before the two new models).

## 6. Limitations

- **Where it is weakest on our match.** Late in each half the ball is missing on up to 80% of a 5-minute block (A02).
  There the team rests on the shape model, trained on other teams, and the decoder learns of a change of team late (a
  median of +6 frames at a start and +11 at an end, A26).
- **The team model has no transfer check on our match, and it is the source most exposed to wrong-team slots.** It
  reads distances per team, and our tracking shows a team with more than 11 slots on at least 10.8% of the graded frames
  (T32); the benchmark's check of this (A27) realised at most 2.3%. Its gain is the benchmark's, where the ball's gaps
  ignore the play, so its size on our match is not known.
- **The touch rule where the ball is seen.** Where the ball is seen, the touch
  rule finds control on 36.9% of our frames, against 43.6% for the 2.0 m rule it replaced (R013); what the tighter rule
  loses here is not measured.
- **Our truth's grain is unknown, and two choices sit on it** (T33). DFL's flag gives 437 possessions a match, median
  3.7 s, and the public window cannot say how ours compares: it is alive on 80.4% of its frames, livelier than 96% of
  DFL's 5-minute windows (A06). The raw and the coarse truth pull the hold and the decoder in opposite directions, and
  R030's rule decided both on a knife-edge. The hold 0.60 is inside the 0.005 tie band by 0.000004; on the mean alone
  0.55 is 0.0050 better (0.60 gains 0.0076 raw and loses 0.0176 coarse). The decoder is the tie-break's choice over
  the best-mean config (`switch_team` 8, `min_possession` 50), which is +0.0069 on the mean over the constants it
  would replace but -0.0146 under the raw truth. If our truth is coarse, both choices cost a little. A run at six
  seeds, made after the choice and choosing nothing, tells no option from another: the declared constants, the hold
  0.55 with the decoder chosen again from it, the best-mean decoder and the constants before R030 lie within 0.0054 on
  the mean. On its three fresh seeds the hold 0.60 alone is 0.0084 below 0.55, so the tie did not repeat (R030). We kept the
  declared rule's outcome: the benchmark cannot tell the options apart, and nothing was picked after a result.
- **The benchmark is not our tracking** (T38, T39). `generic`'s gaps ignore the play, and ours fall where the tracker
  loses the ball; DFL also tracks the ball out of play, which ours loses near the lines (among the rows where a gap
  could begin, the ball is beyond a line on 17.68% of DFL's against 4.07% of ours, R009). So the benchmark still
  overstates the dead-ball model (alive/dead F1 0.9741, R030) by an amount it cannot measure, and cannot value a
  feature about where the ball is lost. It has no wrong-team slots and no long NaN-speed runs, which ours has (T31,
  T32; the retired benchmark put the whole player degradation at 0.013 spellF1, A14). Its gap law is one declared
  range: another range would give other constants.
- **Transfer to LaLiga, on seven matches.** Both models learned from seven German matches, five of them one club's
  home games, so not independent. R030 selected from 192 and then 144 configs on them; held out one match at a time,
  its rules' picks are no better than the constants they replaced (mean -0.0048 for the touch rule's stage, -0.0009
  for the decoder's). In sample the re-choice is worth +0.0038 on the mean (p 0.0312, 6 wins in 7): what it buys is
  that every constant stands on external data only.
- **The tags are left out.** Their team votes covered 0.499 of the graded frames without a ball (R017), but nothing
  external could set their confidence, so they were dropped under the rules (R029, section 2). Taking them out changed
  2.94% of the graded frames; what they were worth was never measured.
- **Timing.** On 10 of the public window's 12 team changes, our truth flips a median 0.96 s before the new team's first
  controlled touch, where DFL's flag flips 1 frame after it; the gap's 90% interval holds 0, so no difference is shown
  (A11), and nothing may be fitted to it. Where the truth's `D` begins and ends (at the crossing, the whistle or the
  restart) is undocumented; the question was drafted (A22) but not sent, since the organisers answered our others with
  their clarification alone. The dead-ball model carries DFL's timing.

**With more time:**
- learn our provider's switch timing from its other games (P16), from a source with a licence;
- place the edges from pass events (P11);
- retrain PathCRF on degraded copies (P20 step 1);
- recover the ball from the video (P14), with a detector trained on other matches only.

## 7. Reproduction

```bash
py -3.11 -m venv .venv                         # macOS/Linux: python3.11 -m venv .venv, then .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ as data/README.md describes
.venv/Scripts/python pipeline/run.py pipeline/configs/main.toml --out "../Un Dia de Partit.csv"
.venv/Scripts/python evaluation/validate_submission.py "../Un Dia de Partit.csv"
```

- **One command** goes from `data/` to the CSV, written with LF line endings and validated by the official checks
  (without `--out`: `evaluation/results/main.csv`).
- **Pins:** Python 3.11.9, and every package the pipeline imports at an exact version in `requirements-pipeline.txt`
  (numpy 2.4.6, pandas 3.0.6). Both models are JSON under `pipeline/models/`, kept LF by `.gitattributes` and read with
  numpy only.
- **The models and the benchmark** (these need the DFL files only: the profile reads nothing of our match):

```bash
.venv/Scripts/python -m pip install -r requirements-benchmark.txt
.venv/Scripts/python pipeline/benchmark/sources.py                    # fetch the DFL files; --verify only checks them
.venv/Scripts/python pipeline/benchmark/train_dead_ball.py --model hgb --profile generic --seeds 3 --verify
.venv/Scripts/python pipeline/benchmark/train_shape.py --features 2 --verify   # each retrains, compares
.venv/Scripts/python pipeline/benchmark/train_team.py --verify
.venv/Scripts/python pipeline/benchmark/p27_profile_check.py          # the profile against its two declared shares
.venv/Scripts/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/p27/p27_final.toml \
    --profile generic --seeds 3 --truth both           # R030's file and the two yardsticks (R030)
.venv/Scripts/python pipeline/benchmark/p27_choose.py bar p27_final   # the bar, from those reports
.venv/Scripts/python pipeline/selftest.py --data       # needs data/: the organisers' four baselines byte for byte
```

- **The re-choice** (R030) runs from the repository: `p27_grid.py` writes each stage's configs, `run.py` scores them,
  `p27_choose.py` applies the declared rules and `p27_shift.py` reads the shift; docs/results.md, R030, has the
  commands stage by stage. The retired profiles stay in the code for the record of R002 to R029, marked as such.

**From a fresh clone.** R042's file (`aae79a04…`) came out the same on two runs, each from a rebuilt cache, and from a
clean clone of `33d5ad7` (the same machine and environment); the three models' `--verify` pass. R030's file
(`2dd51320…`) had come out the same on two runs in the working repository and from a
clean clone of `c8d596e` in a fresh Python 3.11.9 environment built from `requirements-pipeline.txt`; again with the
BLAS thread variables unset, and twice on Linux (WSL 2, Ubuntu 24.04, LF configs). That
was one machine, so another processor is untried. Earlier, A17 and A18 regenerated R017's file
(`7ba575c0…`, an earlier config) from a fresh clone under Windows and under Linux (WSL 2; one laptop, AMD Ryzen 5
5500U, CPU only), with the self-tests passing.

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
- Shaw. *Friends of Tracking*, `Metrica_Velocities`: a 7-frame moving average for the speeds of 25 Hz tracking.
- SkillCorner. Open data: automatic broadcast tracking of A-League matches (MIT licence).
- Vidal-Codina et al. (2022). Automatic event detection in football using tracking data. *Sports Engineering*.
  arXiv:2202.00804.
