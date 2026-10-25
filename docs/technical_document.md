# [Team name] — More than a Hack 2027, Phase 1: technical document

> **Draft.** This is deliverable 3 of SUBMISSION.md: a few pages, uploaded as `<team_name>.pdf` or `<team_name>.md`.
> Fill each section as decisions land, so that the last week is editing rather than writing. Delete these notes
> before submitting.

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `<team_name>.csv`, sha256 `<…>`
- **Scores we can state:** public window macroF1 … / spellF1 …; our validation …. The full-match score is the
  organisers'.

Entry IDs: A.. and R.. are analyses and experiments in `docs/results.md`, T.. pitfalls in `docs/traps.md`, P..
proposals in `docs/proposals.md`.

## 1. Approach

<!-- What we built, and why this rather than the alternatives. Link the proposals we tried (docs/proposals.md). -->

**Pipeline** (`pipeline/README.md`). Five stages: inputs → evidence → decode → refine → output. *Evidence* sources
look at the whole match and give, per frame, the probability that the ball is in play (`alive`) and that the home
team has it given play (`home`), or abstain. A *decoder* turns them into labels one graded half at a time, so nothing
is smoothed across halftime. *Refiners* apply metric-aware clean-ups, and the *output* stage writes the CSV with LF
line endings and validates it with the official checks. Splitting "what the data says" from "what the label sequence
should be" lets each source be measured on its own and lets one decoder weigh them all. The organisers' baseline
runs through the same pipeline and is rebuilt byte for byte as a regression check (R001).

**The decoder** (P06, `pipeline/decode/hmm.py`). A three-state hidden Markov model over `H`, `A` and `D`, decoded by
Viterbi once per graded half. Each source adds a log-likelihood ratio to the states it speaks for, and a source with no
opinion adds nothing (a missing `alive` is not a coin flip). A path pays `switch_team` nats for every change of team and
`switch_dead` for every change into or out of `D`, and a run of `H` or `A` has to last `min_possession` frames. That
makes a possession an interval whose edges have to be paid for, which the baseline's ±2 s vote (`majority_vote`) is
not: it ignores `alive`, gives `D` only where the ball is missing, and keeps whatever survives the vote. The evidence it
decodes is `nearest_player` (the team of the nearest tracked slot, at a confidence of 0.78: the share of in-play,
ball-visible public frames on which it is right, A01) and `ball_presence` (`alive` from whether the tracker sees the
ball); both are fixed by rule from A01 and were not tuned. Its four parameters were chosen on the benchmark of section 5
by a protocol declared before the runs, plus one capped edge extension ruled after Stage 2's results and before its own
(R006, R007). **It does not clear its bar there, so the file we upload today is still `main.toml`** (R007). It stays
because it is the backbone the evidence sources of P04, P05 and P07 plug into, and A10 measures what each of them could
deliver to it.

## 2. Assets used and dropped

| asset | used? | how — or why it was dropped |
|---|---|---|
| Tactical ATD (players, ball, speeds) | | |
| Smart Tagging | | |
| Tactical homography | | |
| Tactical video (frames, audio) | | |
| Broadcast video, ATD, homography | | |
| Public 5-minute ground truth | | |
| Labelled tracking of other matches: the DFL/IDSSE open data, 7 matches (P01) | yes, to validate only | Bassek et al. 2025 (reference below), CC BY 4.0. We use the 25 Hz positions of players and ball, the ball's alive/dead flag and the team in possession on every frame (they give the truth: `D` where dead, otherwise the team), and the match information (teams, roles, pitch, halves). The tracking is degraded to look like ours and run through our pipeline. Nothing is trained on it so far (P13 is an idea) |
| The DFL/IDSSE event files; kloppy's DFL loader and download | dropped | The raw events, one of the three files of a match, are never read. kloppy's loader drops every dead-ball frame by default and its download fails (T30), so our own parser (`pipeline/benchmark/dfl.py`) reads the positions, and kloppy only cross-checks it: 0 disagreements (A05) |
| Other external datasets or pretrained models | | |

**The DFL/IDSSE data: sources, citation, licence.** Seven full matches, two Bundesliga and five 2. Bundesliga, 25 Hz.
Reference: Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event
data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. The data are released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) "with the authorization from the DFL" and are on
figshare (DOI 10.6084/m9.figshare.28196177, as the paper's reference list gives it). We change the data (we degrade
the tracking) and say so. The data are public and outside the competition bundle, so their own licence governs them and
not the bundle's rules. `pipeline/benchmark/sources.py` downloads the 21 files (2.6 GB) into `data/external/dfl/` (not
committed) and accepts a file only if its size and md5 equal the pinned values; a Hugging Face mirror of the same data,
pinned to one commit and checked the same way, is the fallback. Five of the seven matches are one club's home games.

No frame of this match was hand-labelled, for training, validation or the submission (RULES.md, as clarified on
22 Oct 2026).

## 3. Missing and noisy tracking

<!-- What we did about each: fill as the evidence sources land (P04, P05, P07). This section holds what we measured. -->

Measured on our match's ATD and the public GT. Aggregates only; each figure cites its entry.

**The ball.**

- It is missing on 40.7% of the graded frames, in only 150 gaps: median 10.7 s, p90 34 s, longest 221 s. Gaps under 2 s
  hold 0.1% of the missing frames, so interpolation cannot help (A01).
- Coverage worsens through each half: the ball is missing on 12% of the first five graded minutes and on up to 80% of
  a later block (A02). The public window, with 11.8% missing, is the best-covered block of the match (A01).
- A missing ball is not a dead ball: the ball is visible on 66% of the public window's dead-ball frames, and it is
  missing on 33.6% of them against 11.8% of all frames (A01, A07).
- Gaps start where our tracker loses the ball: 28.6% of their onsets are within 2 m of a line against 8.5% of the
  visible frames, the median ball speed at an onset is 2.2 m/s against 5.1 m/s, and they come later in the half. A
  hazard fitted on the 143 onsets ranks them with a cross-half AUC of 0.7110 and 0.6475: a coarse placement, not a
  predictor (A08). Near-line gaps last longer, medians of 12 to 19 s against 7 to 11 s (A08).
- The ATD's ball-speed channel is NaN on 92.2% of the ball-present graded frames and, where given, behaves like a
  player's speed (median 1.3 m/s against 4.8 m/s from the ball's own positions, correlation at most 0.11). We never
  read it and compute the speed from successive positions (T31, A07).
- The tactical camera follows the ball's along-pitch position about 35 frames (1.4 s) late, with a median error of
  12.1 m, and says nothing across the pitch (correlation 0.07), so it is a coarse zone prior for the gaps (A03).
- With the ball visible, the nearest player's team is the true one on 77.8% of 5,640 in-play public frames (A01),
  against 77.5% on the DFL matches (A06).

**The players** (46 slots for 22 players; measured on each half from its kick-off, 155,923 frames, unless said).

- On average 20.52 slots are present per frame; all 22 on 50.2% of the frames, 14 or fewer on 5.25% and 23 or more
  on 2.2%, which is more slots than players (A07, A09).
- A slot is a tracklet, not a player: 1,144 tracklets, median 41.3 s (35.5 s on the graded blocks), p10 2.7 s, p90
  297 s, mean 111.9 s, shortest 8 frames (0.32 s), longest 2,937 s (A07, A09). A slot idles between two tracklets for
  a median of 7.4 s, and 15.0% of those gaps last over a minute (A09).
- A slot can carry the other team's player: a team shows more than 11 slots while the other shows fewer on 10.8% of
  the graded frames (17,156 of 158,857), in 108 episodes (63 with the home team in surplus, 45 away), median 2.7 s, p90
  16 s, longest 50 s. This is a lower bound, because a swap inside 11 against 11 cannot be seen (T32, A07).
- The speed is NaN on 7.8% of the present slot-frames, in 2,843 runs of a median of 21 frames (p90 69, longest 9,268)
  (A07). The position jitter, the RMS of the second difference, is 3.3 mm (A07).

**What the open data tells us, and what it cannot** (A05, A06, T36). We use it to rank methods, so its conventions
matter.

- Our parser and kloppy read the same ball, flags and players (0 disagreements, A05). `D` is DFL's own ball-in-play
  clock to within 2 frames in every match: 55.2% of the graded frames are alive, 48.3% to 59.3% by match, with 92 to
  113 stoppages a match (A06).
- DFL's flag is fine-grained: 437 possessions a match on average (370 to 511), median 3.7 s, 60.5% under 5 s, at the
  every-touch end of the literature's range (A06).
- How far our GT's grain differs is not known. The public window has 17 possessions with a median of 7.56 s, but it is
  alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows. Against DFL's windows that lively, it is 1.34
  times slower, which DFL's own conventions can produce. EVALUATION.md's lag table does not reproduce on the public GT
  (A06, T33).
- DFL's flag turns dead a median 0.16 s after the ball is first seen beyond a line. Ours is unmeasured (A06, P02).
- Two of the seven matches have a team down to 10 players: from minute 7.1 of the first half in one, from minute 13.4
  of the second half in the other. 18.3% of the seven matches' graded frames have 21 players, so a statistic on how
  many players are tracked is measured on the frames at full strength only (T36).
- DFL's players jitter by a few jumps on a 1 cm grid: the RMS of the second difference is 9.0 cm against our 3.3 mm,
  and a 5-frame average inside a tracklet takes it down to a fifth, 1.8 cm (T35, R004c).

## 4. Possession rules implemented

What the submission does today is the baseline's rule in substance (`main.toml`): the nearest tracked player's team, a
±2 s majority vote per half, and `D` only where the ±2 s window holds no ball. The rules below are those of the `hmm`
decoder (P06, `pipeline/configs/p06_hmm.toml`), which does not clear its bar on the benchmark (R007) and is not what we
upload today.

- **Change of possession:** A possession is a run of `H` or of `A`, and the decoder holds the team until the evidence
  for a change pays for it. A change of team costs `switch_team` (8 nats), a blip out and back twice that, and a run
  has to last `min_possession` (75 frames, 3 s) except where the half's edge cuts it short, so no possession is
  shorter. The team evidence is the nearest tracked player's, at a confidence of 0.78 (A01), and there is none where
  the ball is missing. A minimum run turned out to matter more than the penalties (the best config with one scores
  0.2083 raw, the best that switch penalties reached at a minimum of 1 scores 0.1959) and takes the place of a large
  `switch_team` (R007).
- **Loose balls and contested phases:** No rule of its own. A loose ball is read as the nearest player's team, which is
  wrong on 22% of in-play, ball-visible public frames (T14); what stops it flickering is the cost of changing and the
  minimum run. Across a stretch without ball evidence the team changes on the stretch's first frame, by the decoder's
  fixed tie rule (`pipeline/decode/README.md`).
- **Touches, deflections, aerial duels:** Not modelled (P04 would).
- **Dead ball — when it starts and ends:** From ball presence only. A frame without the ball speaks for `D` (a
  likelihood ratio of 4) and a frame with it weakly for play (1.41 for alive), both from A01 by rule, and a change
  into or out of `D` costs `switch_dead` (32 nats). So a gap in the ball decodes as `D`, its edges included, once it is
  longer than about 2 × `switch_dead` / 1.39 frames (46 at 32) and nothing else speaks, while a shorter gap or a stretch
  with the ball can still overrule it. That is broader than the baseline's `D`: `main.toml`'s vote gives it only more
  than 2 s inside a gap, so a gap of 47 to 100 frames, and the first and last 2 s of a longer one, are `D` here and keep
  their team there. R007's `margin` 50 diagnostic, which leaves those edges to the other evidence, measures the
  difference (predicted `D` share 43.1% to 38.3%, raw spellF1 +0.0071). A missing ball is not a dead ball (T12), which
  makes this the weakest part: alive/dead F1 0.5676 on the benchmark against 0.6094 for `nearest_w50` (R007). Nothing
  else says `D` until P05.
- **Restarts — who gets the ball:** No rule. After a stoppage the team is whichever the evidence names once the ball is
  seen (a restart detector is part of P05).
- **Abstentions (`X`):** The decoder never abstains inside a graded half, and the runner writes `X` on the ungraded
  frames, where it is free. It can return the posterior of each state per frame, which shows where an `X` would be
  free (inside a possession), but no refiner reads it yet.

## 5. Validation

**Why not the public window.** It holds 17 possessions, so one possession moves spellF1 by about 0.06. It is the
best-covered 5 minutes of the match (the ball is missing on 12% of its frames, against 40.7% overall), and it ranks
`smooth125` above `smooth50`, the reverse of the full match (T01). We may not hand-label this match, even to validate
(T27). So the window checks that the pipeline is correct and shows this ground truth's timing conventions (P02), and we
choose between versions on other matches.

**The benchmark (P01).** The seven DFL/IDSSE matches of section 2, 668.4 graded minutes, with the truth built as ours
is: `D` where `BallStatus` is dead, otherwise the team `BallPossession` names. Each match is served to our own
components as one of ours (`MatchInputs`, with nothing of the truth reachable), so a config runs on it unchanged, and
the organisers' own `possession_metrics` scores it half by half. An evidence source equal to the truth scores exactly
1.0000 on all seven matches, which anchors the chain from the loaded match to the score (R002). The overview is
`pipeline/benchmark/README.md`.

**Two truths.** The raw truth is DFL's flags as they are (3,061 possessions, 4.58 per graded minute). The grain of our
GT is not known (section 3), so a choice that depends on possession length is ranked under a second, coarse truth too,
which absorbs every possession under 5 s that sits between two possessions of the other team (1,659 possessions, 2.48
a minute; the 5 s rule was written before the numbers were seen and blind to our match, A06). It is a sensitivity
check, not a correction, and a ranking that differs between the two truths is a warning.

**Profiles.** A profile says how a DFL match's tracking is made before a config sees it. Each of the first three adds
one part of ours to the one before, `context` keeps `tracking`'s players and replaces its ball, and the version of a
profile goes into every report.

| profile | the ball | the players | entry |
|---|---|---|---|
| `clean` v1 | DFL's, on every frame | DFL's, one slot per player | R002, the floor |
| `gaps` v1 | missing as ours was: our own missing-ball masks, transplanted onto each half (their pattern, not their clock); its speed NaN (T31) | untouched | R003 |
| `tracking` v3, **the default as of R004c** | `gaps`' ball, byte for byte | our tracklet lengths in 46 slots, wrong-team slots (T32), NaN-speed runs, 5-frame smoothing, and our low-count frames (only the players nearest the ball) | R004 to R004c |
| `context` v1, **sensitivity only** | the gaps placed where our tracker loses the ball: a hazard fitted on our match (A08) | `tracking`'s | R005: fails two of its gates |

Version 3 (R004c) changed one rule of version 2, so that a slot is taken by a new tracklet only from the row after its
last one stopped, and derived the scale of the tracked spans again by the same one-shot rule, because the slot rule
moved its measurement (2.042 to 2.072; T37). All eight gates pass, no check warns and the ranking does not move. What
moved is the account of the jitter, 1.80 cm per slot after smoothing, where R004b's 8.5 cm was slot handovers (T35).
`context` fails two of its gates (R005, section 6): the near-line share badly, 0.5586 of its gaps starting within 2 m
of a line against our 0.2797 (+-0.05), because DFL tracks the ball out of play, and the share missing narrowly, 0.4313
against 0.4110 (+-0.02). The near-line gate binds only `context`: the default's transplanted gaps measure 36.0% on the
same statistic (R003). `context` is a sensitivity check, the default stays `tracking`, and the
four yardsticks keep their order on it.

**Fidelity, as of R004c** (`tracking` v3: 7 matches x 3 seeds, `fidelity.py --profile tracking`, which exits 0). A gate
is our own value, measured the same way on each half from its kick-off, plus or minus a tolerance fixed before anything
was measured. The three count gates are measured where the clean match has all 22 players, since a team down to 10
cannot show 22 slots (T36). All eight pass:

| gate | benchmark | our match | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, share of full-strength frames | 0.4925 | 0.5019 | +-0.06 | pass |
| mean slots present, full-strength frames | 20.596 | 20.522 | +-0.5 | pass |
| 14 or fewer slots present, share of full-strength frames | 0.0469 | 0.0525 | +-0.02 | pass |
| wrong-team share of frames | 0.0918 | 0.1054 | +-0.03 | pass |
| tracklet median | 26.52 s | 41.28 s | +-50% | pass (36% below ours) |
| NaN-speed share of the present slot-frames | 0.0788 | 0.0783 | +-0.01 | pass |

Ten WARN checks are printed and never tuned towards, and none warns: the four yardsticks predict 18.7% to 24.3% fewer
possessions a minute than on ours (band 25%), and the ball is visible on 0.5684 of the benchmark's dead frames against
0.6637 in our public window (band 0.10; the ball is missing there on 12% of frames against 40.7% over the match, so
ours over the whole match is unmeasured).

**The rule on degradations.** A degradation is fitted only on label-free statistics of our tracking (A07, A08), with
constants from rules written before any score was seen, rescaled once per version from one label-free measurement,
frozen with their measurement in the code, and never tuned towards a published score, a yardstick or a WARN (T26, T37).
When a gate missed (the first low-count cap, on the tracklet median) we stopped, said so and made two corrections, each
forced by its measurement: no piece under 8 frames, and one rescale of the tracked spans, applied whether or not the
gate would then pass (R004b, T37). The degrader never reads `ball_status` or `ball_possession`, and a change to a
profile bumps its version.

**Statistics on seven clusters.** The pooled spellF1 is the official scorer's number on all halves as one file. An
interval is the 90% Student t interval of the seven matches' paired deltas against a reference config, beside the exact
sign-flip test (smallest p 2/128 = 0.0156) and the matches won out of seven. It is never a bootstrap: seven matches,
five of them one club's home games, are not a sample to resample. A degradation seed moves a pooled spellF1 by up to
about 0.015 (R003 to R004c), which the intervals do not show, so three seeds are the default.

**Does it rank the four baselines as the full match does?** We run the organisers' four baselines decoded per half
(`nearest_w{0,25,50,125}`) as a check, not a target. Pooled spellF1 on the default profile (R004c; the other profiles'
are in [the scoreboard summary of `docs/results.md`](results.md#the-four-yardsticks-on-every-profile-in-one-table)):

| baseline | published, full match (R000) | benchmark, raw truth | benchmark, coarse truth |
|---|---|---|---|
| smooth50 (`w50`) | 0.2433 | 0.2018 | 0.1842 |
| smooth125 (`w125`) | 0.2242 | 0.1665 | 0.2226 |
| smooth25 (`w25`) | 0.1996 | 0.1900 | 0.1432 |
| raw (`w0`) | 0.0486 | 0.0777 | 0.0342 |

`w0` is last under both truths, losing to `w50` in 7 matches of 7: the robust part of the check. The rest follows the
truth. Raw gives `w50 > w25 > w125 > w0`, coarse `w125 > w50 > w25 > w0`, and neither is the published
`w50 > w125 > w25 > w0` (Kendall tau +0.667 under each). `w125` against `w25` is -0.0228 [-0.0346, -0.0109], 0 wins of
7, under the raw truth and +0.0795 [+0.0636, +0.0954], 7 of 7, under the coarse one. So we decide between smoothing
levels only where the two truths agree, and absolute spellF1 on the benchmark is not our leaderboard's. What the
benchmark cannot show is in section 6.

**The P06 decoder on the benchmark** (R006, R007, A10). Its four parameters were chosen by a protocol declared before
the first run, over Stage 1 and Stage 2: profile `tracking` v3, 3 seeds, the objective the pooled spellF1 under the raw
truth, configs within 0.005 of the best counted as tied, then the best coarse score, then the larger `switch_team`,
`min_possession`, `switch_dead` and `min_dead`; both truths are read side by side (T33), and the bar for calling it a
win is recorded and not used to choose. The evidence is fixed by rule from A01 and was not tuned. Stage 1 crossed the
two switch penalties (24 configs, R006) and Stage 2 the minimum runs (90). Stage 2b (15 configs) was one edge extension,
capped at one, ruled after Stage 2's results and before its own, and the chosen config comes from it: the declared rule
over Stages 1 and 2 alone picks `p06s2_t016_d064_mp050_md025`, 0.0012 under `nearest_w50` raw and 0.0290 under
`nearest_w125` coarse (R007). In all, 123 distinct configs, 78 once the `min_dead` 25 twins are dropped (R007). The
choice is `switch_team` 8, `switch_dead` 32, `min_possession` 75 frames and `min_dead` 1 (which moves nothing in Stage
2). Pooled spellF1 on `tracking`, 3 seeds:

| config | raw spellF1 | coarse spellF1 | alive/dead F1 | possessions per minute | `D` share |
|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | 0.1842 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` | 0.1665 | 0.2226 | 0.6428 | 3.05 | 23.3% |
| `hmm`, chosen (8, 32, 75, 1) | 0.2053 | 0.1967 | 0.5676 | 3.10 | 43.1% |
| `hmm`, without the `seen` term (control) | 0.1990 | 0.1971 | 0.5422 | 2.73 | 46.4% |
| `hmm`, `margin` 50 (diagnostic, not a candidate) | 0.2124 | 0.1990 | 0.5876 | 3.29 | 38.3% |
| the truth | | | | 4.58 raw, 2.48 coarse | 44.8% |

**The bar is not met under either truth.** Raw, the chosen config is +0.0036 over `nearest_w50` (+0.0041 per match
[-0.0022, +0.0104], sign-flip p 0.2500, 5 wins in 7); coarse, it is 0.0259 below `nearest_w125` (0 wins in 7). No config
of the three stages passes either half. A minimum run mattered more than the penalties: the best config with one scores
0.2083 raw, the best that switch penalties reached at a minimum of 1 scores 0.1959. The HMM's predicted `D` share,
43.1%, is near the truth's 44.8%, and its alive/dead F1 is still below both yardsticks'. The `seen` term is worth 0.0063
raw on `tracking` (without it the chosen config loses in 7 matches of 7, p 0.0156) and nothing coarse (+0.0003). The
`margin` 50 diagnostic, which leaves the edges of a gap to the other evidence, is above the chosen config in all four
profile and truth cells (+0.0023 to +0.0071, two of them on `context`) and clears the raw half of the bar on `tracking`
(+0.0107 over `nearest_w50`, p 0.0312, 6 wins in 7) and not the coarse half. It
was one look at one value, so it is not a candidate: a sweep with its own protocol is the follow-up
(`docs/proposals.md`, "After P06"). On `context` (sensitivity only, R005) every score is higher and the chosen config
clears the raw half (+0.0140 over `nearest_w50`, 7 wins in 7) and not the coarse half (-0.0182, p 0.0781).

**What better evidence would be worth** (A10). The chosen decoder run with the benchmark's own truth as evidence gives
an upper bound, since the oracles are exact: a perfect dead-ball detector takes pooled spellF1 to 0.3678 raw and 0.4417
coarse, a perfect possessor where the ball is seen to 0.3586 and 0.3651, and both everywhere to 0.7627 and 0.9226.
Under the raw truth the first two are level (+0.0092 apart, p 0.2500); under the coarse truth the dead-ball detector is
ahead by +0.0766 (p 0.0156). A perfect possessor inside the gaps adds +0.0008 raw until `D` is right and +0.1971 once
it is, so the possession-without-the-ball work (P07) is worth nothing to this decoder before a dead-ball source (P05)
exists. The full-oracle row is a cap on this config and not on the approach: 0.7627 is below 1.0 because the penalties
and the 75-frame minimum, chosen under noisy evidence, merge the shortest of DFL's possessions (the logs cannot say how
much each costs). Apart from A01's three evidence values, fixed before P06, the public window was used only to check
that the pipeline is correct (the file is valid and identical across runs), never to choose (T01).

## 6. Limitations

<!-- Where it fails, and what we would fix with more time. The pipeline's own limits: fill as decisions land. -->

**Of the validation benchmark (P01)**, on which every choice between versions rests:

- **Conventions.** DFL's flags may flip at other moments than ours do, and the grain of our ground truth is not known:
  the public window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, so it cannot say how far
  ours differs from DFL's 437 possessions a match (A06, T33). The coarse truth stands in for the difference and does not
  correct it, and `D` timing against ours is unmeasured. The timing parameters come from the public window (P02).
- **Seven matches, five of them Fortuna Düsseldorf's home games.** The sample is small, so the intervals are wide, and
  not independent, so even they may be too narrow; the smallest sign-flip p is 0.0156. Two matches have a team down to
  10 players, so statistics on how many players are tracked are measured at full strength only (T36).
- **No video, audio, homography or Smart Tagging.** Whatever uses them (P08, P09, P10, P14, P15) is checked on the
  public window only, or simulated from the benchmark's truth with noise measured on our match (A03).
- **Aerial projection.** A camera-based tracker projects an airborne ball onto the pitch plane, which displaces it.
  DFL's positions carry no such error and the benchmark adds none, and ours has no ball height, so the error cannot be
  measured on ours. Nor can we tell how many of our ball gaps come from airborne balls: our first plan said "crowded
  and aerial", and measured it is "lines and slow balls" (A08).
- **Position bias.** None is added, because none can be measured on ours (there is no reference to measure it against),
  so a method that reads distances is validated without whatever positional error ours carries.
- **Rougher positions.** The jitter after smoothing is 1.80 cm per slot against our 3.3 mm, 5.5 times ours in RMS (T35,
  R004c), which matters to a method that reads positions, speeds or accelerations.
- **The default's ball gaps are transplanted and blind to the play.** They keep the pattern of our masks, not their
  clock (the offset is drawn uniformly, so the pattern lags by up to 7.6 minutes in a 68,000-frame half, `degrade.py`;
  R003's draws reached 11,222 frames) and not where our tracker loses the ball (near the lines, on slow balls, with
  more players around, A08). 36.0% of them start within 2 m of a line against 28.0% of ours, because DFL's ball is
  tracked out of play and so spends more time near the lines (R003, R005), and the ball is visible on 0.5684 of the
  dead frames against 0.6637 in the public window (R003). A method that depends on where the gaps fall is validated on
  gaps that ignore the play (T38).
- **`context` does not fix that, and why.** It places the gaps by the play and fails two of its gates: the near-line
  one (the default has none) on every seed, 0.5586 against 0.2797 (+-0.05), and the share missing narrowly (0.4313
  against 0.4110, +-0.02). The first is the data's doing: DFL tracks the ball out of play, so 25.2% of its
  at-risk rows are within 2 m of a line against 9.0% of ours, while the model's probability rises near a line by a
  similar factor on both (3.9 on DFL's rows, 2.9 on ours). No scale of the hazard moves it and no remedy was applied.
  It stays a sensitivity profile, and its numbers are not set beside `tracking`'s without saying so (R005).
- **Tracklets and slots are not ours.** The tracklet median is 26.52 s against our 41.28 s, 36% below and inside its
  +-50% band (R004c, T37). A player returns to his own slot 47.4% of the time (ours has no identities), and a slot
  idles a median of 4.3 s where ours idles 7.4 s (both on J03WMX, seed 0, R004c, A09). A method that follows a slot
  across a loss would be validated optimistically.
- **The cap keeps the players nearest the ball.** Where ours has 18 or fewer slots present, the profile keeps the
  players nearest the true ball. A nearest-player method barely notices (the four baselines move by 0.4 points of
  spellF1 at most, R004b), and one that needs the players far from the ball sees fewer of them than the real match gave
  it, as it does on ours. The cap's counts come from the frames of the ball's gaps and not from the same play.
- **The halves.** A benchmark half starts at its kick-off, while our graded second half starts 117 s before it, all
  `D` (A06, T02).

**Of the P06 decoder** (R006, R007, A10; it does not clear its bar and is not the submission):

- **`D` comes only from ball presence until P05.** The ball is missing on 40.7% of the graded frames in long gaps, and
  a missing ball is not a dead ball (T12). The decoder's `D` has an alive/dead F1 of 0.5676 on the benchmark against
  0.6094 for `nearest_w50` and 0.6428 for `nearest_w125`, and the two `ball_presence` values rest on four stoppages of
  the public window (A01). A10 puts the price at +0.1625 raw and +0.2450 coarse for a perfect detector, so this is the
  decoder's weakest point and P05's job.
- **The grain of our GT is unknown, and the choice rests on the raw truth.** The two truths want different decoders:
  raw peaks at a minimum run of 75 frames and `switch_team` 16 at minimum runs of 1, coarse keeps rising to 150 frames
  (0.2310 coarse for the coarse best, against 0.1967 for the chosen config) and to `switch_team` 64 at minimum runs of 1
  (R006, R007). The chosen config predicts 3.10 possessions a minute, between the raw truth's 4.58 and the coarse
  truth's 2.48. Which of them our GT resembles is not known (section 3, T33), and a decoder too fine or too coarse for
  it would not show on the public window.
- **Selecting from 123 configs is optimistic, and by how much is only partly known.** The leave-one-match-out check
  chooses the config on six matches and scores it on the seventh. With the whole rule it picks the chosen config in all
  seven folds, so its held-out estimate is the config's own, +0.0041 over `nearest_w50` (p 0.2500, 5 wins in 7). By the
  objective alone the six-fold picks score 0.2090 held out against 0.2093 in sample for the raw best, so selecting
  costs nothing measurable across matches. That is less than it looks: the check varies the match and not the
  degradation draws, which the three seeds share across folds, a seed moves a pooled spellF1 by up to about 0.015, three
  times the 0.005 band that defined the tied six, it holds the grid fixed although Stage 2 was centred, and Stage 2b
  placed, on results from all seven matches, and it says nothing of how DFL's degraded tracking transfers to ours.
  The chosen config is the raw fourth and the coarse 29th of 78 score profiles, and it sits on an edge of the last grid
  in `switch_team` and `switch_dead`, where by the protocol's ruling nothing was extended further.
- **`switch_dead` was chosen where the benchmark cannot value ball presence.** On `tracking` the gaps are transplanted
  blind to the play, so a missing ball says almost nothing about `D` there (R005, T38), and `switch_dead`, which sets
  how far the decoder believes it, was chosen on a profile that cannot value it and is specific to it. The check on
  `context` (sensitivity only) shows how far under the raw truth: 8, 32 and 128 score 0.2323, 0.2332 and 0.2297, so the
  chosen 32 holds there within noise; the coarse truth prefers more (0.2327 at 128, +0.0067 over 32, 6 wins in 7, p
  0.0781). A10's `__alive` ceiling also overstates what a perfect detector would add on our match, since it starts from
  almost no `D` information. Any choice about the gaps is read on `context` too (T38).

## 7. Reproduction

```bash
python3.11 -m venv .venv                     # Windows: py -3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-pipeline.txt
# unpack the data bundle into data/ (see data/README.md)
.venv/bin/python pipeline/selftest.py --data                    # optional: rebuilds the organisers' 4 baselines byte for byte
.venv/bin/python pipeline/run.py pipeline/configs/main.toml     # writes evaluation/results/main.csv and validates it
.venv/bin/python evaluation/validate_submission.py evaluation/results/main.csv
```

Expected sha256 of `main.csv`: `<…>` (the run prints it). Runtime: about 20 s the first time, which includes parsing
the tracking into `data/cache/` once, then a few seconds per run, on [hardware].

The validation benchmark of section 5 is reproduced separately and is not needed for the CSV (the
[benchmark README](../pipeline/benchmark/README.md) has the rest; reports go to `data/runs/benchmark/`):

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt          # the pipeline's pins + lxml
.venv/bin/python pipeline/benchmark/sources.py --verify               # the 21 DFL files, md5 (drop --verify to fetch)
.venv/bin/python pipeline/benchmark/fidelity.py --profile tracking --seeds 3     # the gates and warnings of section 5
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml \
    pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3    # R004c; again with --truth coarse
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/configs/p06_hmm.toml --profile tracking --seeds 3     # P06's candidate (R007); --truth coarse too,
                                                                    # with nearest_w125.toml first: its bar there
.venv/bin/python pipeline/benchmark/headroom.py pipeline/configs/p06_hmm.toml \
    --profile tracking --seeds 3                                    # A10; --truth coarse too
```
