# [Team name] — More than a Hack 2027, Phase 1: technical document

> **Draft.** This is deliverable 3 of SUBMISSION.md: a few pages, uploaded as `<team_name>.pdf` or `<team_name>.md`.
> Fill each section as decisions land, so that the last week is editing rather than writing. Delete these notes
> before submitting.

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `<team_name>.csv`, sha256 `<…>`
- **Scores we can state:** public window macroF1 … / spellF1 …; our validation …. The full-match score is the
  organisers'.

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
| Labelled tracking of other matches: the DFL/IDSSE open data, 7 matches (P01) | yes, to validate only | Bassek et al., *Scientific Data* 12, 195 (2025), CC BY 4.0. We use the 25 Hz positions of players and ball, the ball's alive/dead flag and the team in possession on every frame (they give the truth: `D` where dead, otherwise the team), and the match information (teams, roles, pitch, halves). The tracking is degraded to look like ours and run through our pipeline. Nothing is trained on it so far (P13 is an idea) |
| The DFL/IDSSE event files; kloppy's DFL loader and download | dropped | The raw events, one of the three files of a match, are never read. kloppy's loader drops every dead-ball frame by default and its download fails (T30), so our own parser (`pipeline/benchmark/dfl.py`) reads the positions, and kloppy only cross-checks it: 0 disagreements (A05) |
| Other external datasets or pretrained models | | |

**The DFL/IDSSE data: sources, citation, licence.** Seven full matches, two Bundesliga and five 2. Bundesliga, 25 Hz,
from Bassek et al., *Scientific Data* 12, 195 (2025), published under CC BY 4.0, and cited here as that licence asks.
We change the data (we degrade the tracking), and say so. They are public and outside the competition bundle, so their
own licence governs them and not the bundle's rules. `pipeline/benchmark/sources.py` downloads the 21 files (2.6 GB) into `data/external/dfl/` (not committed) and
accepts a file only if its size and md5 equal the pinned values; a Hugging Face mirror of the same data, pinned to one
commit and checked the same way, is the fallback. Five of the seven matches are one club's home games.

No frame of this match was hand-labelled, for training, validation or the submission (RULES.md, as clarified on
22 Oct 2026).

## 3. Missing and noisy tracking

<!-- What we did about each: fill as the evidence sources land (P04, P05, P07). This section holds what we measured.
Entries A.. and R.. are in docs/results.md, T.. in docs/traps.md, P.. in docs/proposals.md. -->

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
  and a moving average removes little of it (T35).

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
(T27). So we use the window to check that the pipeline is correct and to read this ground truth's timing conventions
(P02), and we choose between versions on other matches.

**The benchmark (P01).** The seven DFL/IDSSE matches of section 2, 668.4 graded minutes. The truth is built as ours is:
`D` where `BallStatus` is dead, otherwise the team `BallPossession` names. A match is served to our own components as
one of ours (`MatchInputs`, from which nothing of the truth is reachable, and a check walks everything the inputs
reach), so a config runs on it unchanged, and its labels are scored with the organisers' own `possession_metrics`, half
by half and pooled from summed counts. An evidence source equal to the truth scores exactly 1.0000 on spellF1, macroF1
and alive/dead F1 on all seven matches, which anchors the chain from the loaded match to the score (R002).

**Two truths.** The raw truth is DFL's flags as they are: 3,061 possessions, 4.58 per graded minute. The grain of our GT
is not known (section 3), so every choice that depends on possession length is ranked under a second, coarse truth too:
it absorbs every possession under 5 s that sits between two possessions of the other team (1,659 possessions, 2.48 per
graded minute). `COARSE_MIN_S` = 5 s came from a rule written before the numbers were seen and blind to our match
(A06, R002). The coarse truth is a sensitivity check, not a correction, and a ranking that differs between the two is a
warning.

**Profiles.** A profile says how the tracking of a DFL match is made before a config sees it. Each adds one part of
our tracking to the one before, and has a version that goes into every report.

| profile | the ball | the players | entry |
|---|---|---|---|
| `clean` v1 | DFL's, on every frame | DFL's, one slot per player | R002, the floor |
| `gaps` v1 | missing where ours was: our own missing-ball masks, transplanted onto each half; its speed NaN (T31) | untouched | R003 |
| `tracking` v2, the default as of R004b | `gaps`' ball, byte for byte | our tracklet lengths in 46 slots (24 home, 22 away); wrong-team slots (T32); NaN-speed runs; 5-frame smoothing; and since v2 only the players nearest the ball where ours has 18 or fewer slots present, no piece under 8 frames, and the tracked spans rescaled once | R004 (v1), R004b (v2) |
| `context` | the gaps placed where our tracker loses the ball: a hazard fitted on our match (A08), not a transplanted mask | `tracking`'s | R005, in progress |

R005 adds the `context` profile, with its own gates and table; the default and the tables below are "as of R004b" until
then.

**Fidelity, as of R004b** (`tracking` v2: 7 matches x 3 seeds, `fidelity.py --profile tracking`). Gates are our own
values, measured the same way on each half from its kick-off, plus or minus a tolerance fixed before anything was
measured. The profile is frozen only when all of them pass, and all eight do:

| gate | benchmark | our match | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, share of full-strength frames | 0.4938 | 0.5019 | +-0.06 | pass |
| mean slots present, full-strength frames | 20.599 | 20.522 | +-0.5 | pass |
| 14 or fewer slots present, share of full-strength frames | 0.0469 | 0.0525 | +-0.02 | pass |
| wrong-team share of frames | 0.0942 | 0.1054 | +-0.03 | pass |
| tracklet median | 27.08 s | 41.28 s | +-50% | pass (34% below ours) |
| NaN-speed share of the present slot-frames | 0.0788 | 0.0783 | +-0.01 | pass |

The three count gates are measured on the frames where the clean DFL match has all 22 players, because a team down to
10 cannot show 22 slots whatever the degradation does (T36). WARN checks are printed and discussed, and never tuned
towards (none warns in R004b):

| check | benchmark | our match | band |
|---|---|---|---|
| nearest player agrees with the truth, alive and ball visible | 0.7586 | 0.7784 (public window) | +-0.05 |
| ball visible on dead frames | 0.5684 | 0.6637 (public window) | +-0.10 |
| possessions per graded minute of `w0`, `w25`, `w50`, `w125` | 13.91, 5.39, 4.32, 3.06 | 18.40, 7.04, 5.61, 3.76 | +-25% (-24.4%, -23.5%, -23.0%, -18.7%) |
| D share of `w0`, `w25`, `w50`, `w125` | 0.4189, 0.3764, 0.3341, 0.2329 | 0.4069, 0.3649, 0.3234, 0.2233 | +-0.05 |

**The rule on degradations.** A degradation is fitted only on label-free statistics of our tracking (A07, A08): the
gaps, the tracklets, the slots, the wrong-team and NaN-speed shares, the jitter. Its constants come from rules written
before any score was seen, are rescaled at most once, are frozen with their measurement in the code, and are never
tuned towards a published score, a yardstick or a WARN (T26, T37). When a gate missed, we stopped and said so: the first
version of the low-count cap missed the tracklet median by 0.96 s, and the fix was one rescale (2.042) of the tracked
spans, applied whether or not the gate would then pass (R004b). The degrader never reads `ball_status` or
`ball_possession`. A change to a profile bumps its version.

**Statistics on seven clusters.** The pooled spellF1 is the official scorer's number on all halves of all matches as
one file. An interval is the 90% Student t interval of the seven matches' paired deltas against a reference config. We
add the exact sign-flip test over all 2^7 sign patterns (its smallest p is 2/128 = 0.0156) and the number of matches
won out of seven, and report them by competition too. It is never a bootstrap: seven matches are not a sample to
resample, and five of them are one club's home games. A degradation seed moves a pooled spellF1 by up to about 0.015,
and the intervals say nothing of that spread, so three seeds are the default and a gap of that size is read with care.

**Does it rank the four baselines as the full match does?** We run the organisers' four baselines decoded per half
(`nearest_w{0,25,50,125}`; frame for frame the published files on the graded frames, A07), and ask it as a check, not a
target. Pooled spellF1, R004b:

| baseline | published, full match (R000) | benchmark, raw truth | benchmark, coarse truth |
|---|---|---|---|
| smooth50 (`w50`) | 0.2433 | 0.1989 | 0.1811 |
| smooth125 (`w125`) | 0.2242 | 0.1624 | 0.2173 |
| smooth25 (`w25`) | 0.1996 | 0.1897 | 0.1422 |
| raw (`w0`) | 0.0486 | 0.0779 | 0.0341 |

The robust part holds: `w0` is last under both truths, losing to `w50` in 7 matches of 7 (R002 to R004b). The rest of
the order follows the truth. Raw gives `w50 > w25 > w125 > w0` and coarse `w125 > w50 > w25 > w0`; neither is the
published order `w50 > w125 > w25 > w0` (Kendall tau +0.667 under each from R003 on). `w125` against `w25` is
-0.0264 [-0.0389, -0.0139], 1 win of 7, under the raw truth and +0.0752 [+0.0613, +0.0892], 7 of 7, under the coarse
one. So we decide between smoothing levels only where the two truths agree, and absolute spellF1 on the benchmark is
not our leaderboard's under either.

**What the benchmark could not measure, and how we check it instead.**

- Whatever uses the homography, the Smart Tagging, the camera or the audio: DFL has no video. Those sources (P08, P09,
  P10, P14, P15) are checked on the public window only, or simulated from the benchmark's truth with noise measured on
  our match (A03).
- Our ground truth's conventions: DFL's flags may flip at other moments than ours do, and the grain of our GT is not
  known. The timing parameters come from the public window (P02), and `D` timing against ours is unmeasured (A06).
- Slot identity. The profile's slots are the lowest free ones, so a player returns to his own slot 48.1% of the time
  (ours has no identities) and a slot idles a median of 4.1 s where ours idles 7.4 s (R004b, A09). A method that follows
  a slot across a loss would be validated optimistically.
- Players far from the ball: the cap keeps those nearest the true ball, so a nearest-player method barely notices it
  (it moves the four baselines by 0.4 points of spellF1 at most, R004b).
- Positional error: the jitter after smoothing is 8.5 cm against our 3.3 mm (T35), and no positional bias is added,
  since none can be measured on ours.
- The halves: a benchmark half starts at its kick-off, while our graded second half starts 117 s before it, all `D`
  (A06, T02).

## 6. Limitations

<!-- Where it fails, and what we would fix with more time. -->

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
