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

- **Change of possession:** <!-- what counts as controlled possession -->
- **Loose balls and contested phases:**
- **Touches, deflections, aerial duels:**
- **Dead ball — when it starts and ends:**
- **Restarts — who gets the ball:**
- **Abstentions (`X`):**

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

**Profiles.** A profile says how a DFL match's tracking is made before a config sees it. Each adds one part of ours to
the one before, and its version goes into every report.

| profile | the ball | the players | entry |
|---|---|---|---|
| `clean` v1 | DFL's, on every frame | DFL's, one slot per player | R002, the floor |
| `gaps` v1 | missing where ours was: our own missing-ball masks, transplanted onto each half; its speed NaN (T31) | untouched | R003 |
| `tracking` v3, **the default as of R004c** | `gaps`' ball, byte for byte | our tracklet lengths in 46 slots, wrong-team slots (T32), NaN-speed runs, 5-frame smoothing, and our low-count frames (only the players nearest the ball) | R004 to R004c |
| `context` v1, **sensitivity only** | the gaps placed where our tracker loses the ball: a hazard fitted on our match (A08) | `tracking`'s | R005: fails the near-line gate |

Version 3 (R004c) changed one rule of version 2, so that a slot is taken by a new tracklet only from the row after its
last one stopped, and rescaled the tracked spans once more (2.072). All eight gates pass, no check warns and the ranking
does not move. What moved is the account of the jitter, 1.80 cm per slot after smoothing, where R004b's 8.5 cm was slot
handovers (T35). `context` fails its near-line gate, 0.5586 of its gaps starting within 2 m of a line against our 0.2797
(+-0.05), because DFL tracks the ball out of play (R005, section 6). It is a sensitivity check, the default stays
`tracking`, and the four yardsticks keep their order on it.

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
0.6637 (band 0.10).

**The rule on degradations.** A degradation is fitted only on label-free statistics of our tracking (A07, A08), with
constants from rules written before any score was seen, rescaled at most once, frozen with their measurement in the
code, and never tuned towards a published score, a yardstick or a WARN (T26, T37). When a gate missed (the first
low-count cap, on the tracklet median) we stopped, said so and made two corrections, each forced by its measurement
(R004b). The degrader never reads `ball_status` or `ball_possession`, and a change to a profile bumps its version.

**Statistics on seven clusters.** The pooled spellF1 is the official scorer's number on all halves as one file. An
interval is the 90% Student t interval of the seven matches' paired deltas against a reference config, beside the exact
sign-flip test (smallest p 2/128 = 0.0156) and the matches won out of seven. It is never a bootstrap: seven matches,
five of them one club's home games, are not a sample to resample. A degradation seed moves a pooled spellF1 by up to
about 0.015 (R003 to R004c), which the intervals do not show, so three seeds are the default.

**Does it rank the four baselines as the full match does?** We run the organisers' four baselines decoded per half
(`nearest_w{0,25,50,125}`) as a check, not a target. Pooled spellF1 on the default profile (R004c; the other profiles'
are in the scoreboard summary of `docs/results.md`):

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

## 6. Limitations

<!-- Where it fails, and what we would fix with more time. The pipeline's own limits: fill as decisions land. -->

**Of the validation benchmark (P01)**, on which every choice between versions rests:

- **Conventions.** DFL's flags may flip at other moments than ours do, and the grain of our ground truth is not known:
  the public window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, so it cannot say how far
  ours differs from DFL's 437 possessions a match (A06, T33). The coarse truth stands in for the difference and does not
  correct it, and `D` timing against ours is unmeasured. The timing parameters come from the public window (P02).
- **Seven matches, five of them Fortuna Düsseldorf's home games.** A small and not independent sample: the intervals
  are wide for that reason, and the smallest sign-flip p is 0.0156. Two matches have a team down to 10 players, so
  statistics on how many players are tracked are measured at full strength only (T36).
- **No video, audio, homography or Smart Tagging.** Whatever uses them (P08, P09, P10, P14, P15) is checked on the
  public window only, or simulated from the benchmark's truth with noise measured on our match (A03).
- **Aerial play.** The ATD has no ball height, so how many of our ball gaps and tracker errors come from airborne balls
  cannot be measured on ours, and the profiles do not model it: our first plan said "crowded and aerial", and measured
  it is "lines and slow balls" (A08).
- **Position bias.** None is added, because none can be measured on ours (there is no reference to measure it against).
  The jitter after smoothing is 1.80 cm per slot against our 3.3 mm, 5.5 times ours (T35, R004c), so a method that
  reads distances is validated without whatever positional error ours carries.
- **The default's ball gaps are transplanted and blind to the play.** They sit where our own masks were in time, not
  where a tracker loses the ball (near the lines, on slow balls, with more players around, A08): 36.0% of them start
  within 2 m of a line against 28.0% of ours, and the ball is visible on 0.5684 of the dead frames against 0.6637 in
  the public window (R003). A method that depends on where the gaps fall is validated on gaps that ignore the play.
- **`context` does not fix that, and why.** It places the gaps by the play and fails its near-line gate on every seed,
  0.5586 against 0.2797 (+-0.05): DFL tracks the ball out of play, so 25.2% of its at-risk rows are within 2 m of a line
  against 9.0% of ours, while the model's probability rises near a line by a similar factor on both (3.9 on DFL's rows,
  2.9 on ours). No scale of the hazard moves it and no remedy was applied. It stays a sensitivity profile, and its
  numbers are not set beside `tracking`'s without saying so (R005).
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
