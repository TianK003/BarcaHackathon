# Un Dia de Partit — More than a Hack 2027, Phase 1: technical document

> **The full record, not the file to upload.** Deliverable 3 (SUBMISSION.md: a few pages, uploaded as
> `Un Dia de Partit.pdf` or `Un Dia de Partit.md`) is [`technical_document_submission.md`](technical_document_submission.md): the
> same seven sections at deliverable length, every number copied from `docs/results.md` with its entry. This file keeps
> every choice, number and caveat in full. When a decision changes, update both.

- **Team:** [five names]
- **Code:** [public repository URL], commit `<hash>`
- **Submission:** `Un Dia de Partit.csv`, sha256 `2dd51320b1d32f8be6d06e63f833e6d8e196ddcaad294b606db3914c351c1660`
  (R030)
- **Scores we can state:** public window macroF1 0.7238 / spellF1 0.5029 (a check for gross errors only); our
  validation, pooled spellF1 on the benchmark of other matches, 0.5414 raw / 0.5829 coarse (R030). The full-match
  score is the organisers'.

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
not: it ignores `alive`, gives `D` only where the ball is missing, and keeps whatever survives the vote.

Its four parameters were chosen on the benchmark of section 5 by protocols declared before the runs. The first
choice, 8, 32, 75 and 1 for `nearest_player` and `ball_presence`, did not clear its bar (R006, R007). R010 chose 4, 32,
25 and 25 for the first two sources below, and R011 added P07's source with the decoder as it was. R013 chose
`switch_team` and `min_possession` again with the three sources of that day and the touch rule it re-chose first: `switch_team` 4,
`switch_dead` 32, `min_possession` 35 and `min_dead` 25 (the last two kept from R010, where the score was flat along
them). R024 chose `switch_team` and `min_possession` again under R023's objective, the mean of the raw and the coarse
truth's scores, and kept them. R030 chose all four again on the benchmark's new profile, `generic`, which measures
nothing of our match (section 5): `switch_team` went to 6 and the other three stayed. **Today: 6, 32, 35 and 25.**

**The evidence** (P04, P05 and P07, `pipeline/evidence/`). Three sources. `controlled_touch` (P04) gives `home`: it
votes for a team at 0.93 inside the episodes where that team controls the ball, and weakly (0.60) for the last
controlling team on the other frames where the ball is visible. R013 chose the rule's constants on the benchmark with
none of the window-derived values in any candidate, R024 moved its hold to 0.55, and R030 chose them all again on
`generic`: the rule stayed and the hold went to 0.60. It says nothing where the ball is missing, and so a
pass in flight or a duel keeps the last team and does not flip it. `dead_ball_model` (P05) gives `alive`: a
gradient-boosted tree model on 30 features of the play (where the ball is against the lines and the restart spots, how
the players move, the set-piece shapes), trained on the seven benchmark matches as the benchmark's profile degrades
them (`generic` since R030, section 5). It reads
the ball where it is visible and the players where it is not. `shape_possession` (P07) also gives `home`, from the
players alone: boosted trees on 26 features of how the two teams stand and move (how far up the pitch each is and how
wide, its velocity along its attack, its shares in the two end thirds), none of which reads the ball, trained on the
seven benchmark matches with the ball hidden and read symmetrically in the two teams. It is the only source of `home`
that reads no ball (R011), and at weight 0.3 (R030; 0.2 until then) it speaks on every graded frame; where the ball is
missing (40.7% of the graded frames) it is the only source of `home`. From R017 to R029 it shared that vote with
`smart_tagging` (P08), Smart
Tagging's automatic phase tags as team votes, until the organisers' clarification of November 2026 ruled out the
source's confidence, which had been measured on this match (section 2).
`nearest_player` (the baselines' team source) and `ball_presence` (P06's first `alive` source) stay in the pipeline and
are not in the submission. **`main.toml` is these three sources, the decoder and the shift below** (R010 to R013,
R024, R029, R030): P04 and P05 together, P07 on top of them, and R013's re-chosen touch rule and decoder each met the
benchmark's bar of their day; R024, re-choosing R013's two stages on the mean of the two truths, moved only the hold,
from 0.60 to 0.55, and met its bar on that mean; R029 took out P08, which R017 had adopted at Tian's pick of its
strength after its team votes passed a label-free bar on this match; and R030 retrained both models and chose every
numeric constant again on `generic`, a profile that measures nothing of our match (the hold to 0.60, `switch_team` to
6, P07's weight to 0.3, the rest unchanged), and met the bar there. Section 6 says what that does and does not show.
The file has not been uploaded.

**The shift** (P02, `pipeline/refine/shift_edges.py`). After the decoder, each kind of edge of a half is moved by a
whole number of frames: a change of team 1 frame later, a change into `D` 2 frames earlier, a change out of `D` not at
all. They are the pipeline's own lag on the seven benchmark matches, negated, kept by a guard on the benchmark. The rule
R012 declared had a second term, a gap to our truth's own convention read from the public window. It came out 0 for
every type (it rests on 1 to 4 matched changes per type, R012, A11), and since the organisers' rule of 23 Oct 2026 it is
retired: reading a constant off the public window is tuning on a window that is for checking the pipeline only, so the
term stays 0 by rule and the shift is the DFL lag alone. The shift is adopted by a rule declared before the runs, and it
is not shown to help: its benchmark gains were +0.0000 to +0.0009 pooled spellF1 (R012, on `tracking` and `context`,
raw and coarse truth), inside the noise. R030 measured the lag again on `generic` and got the same three values, and
its guard kept them by a hair: +0.000029 on the mean of the two truths (section 5).

## 2. Assets used and dropped

| asset | used? | how — or why it was dropped |
|---|---|---|
| Tactical ATD (players, ball, speeds) | yes | The input of every source: the players' positions, the team of each slot and the ball, with every speed computed from successive positions (T31). For P02 it also gives the label-free events that A11 times our truth's changes against: a team's first controlled touch, the ball first seen beyond a line, where the ball is lost and where it moves after standing still, where the dead-ball model's `alive` crosses 0.5. The same events are found on DFL's degraded copy |
| Smart Tagging | not in the submission: team votes from R017 to R029 (P08); also in A11 | The `smart_tagging` source turns the automatic phase tags into a team vote on `home` (a possession-phase tag votes for its team, a defending one for the other, no vote where tags disagree or say nothing), cut at the graded blocks. Its dead votes (`SET PIECES`, `GOALS`) were measured in R017 against `dead_ball_model` (0.409 agreement, bar 0.60) and dropped, and the set-piece tags are not used as a restart type either. Earlier, A11 timed `SET PIECES` intervals against our truth's stoppages (an interval within 5 s of the start of none of the 4 and of the end of 1 of the 4), and they said nothing. The source is `MATCH_ONLY` (DFL has no tags), so the benchmark ran `main.toml` with `--drop-match-only` while the source was in it. R029 took it out on 3 Nov 2026: its `c_team` had been measured on this match (below), and the organisers' clarification of November 2026 has parameters tuned on external data only, which has no tags |
| Tactical homography | only in A03 | The pitch point under the image centre (the inverse homography of the centre pixel) was compared with the ATD ball over the graded frames: the camera's aim is a coarse zone prior for long ball gaps, not a position (A03). No source of the pipeline reads it: P09 planned the aim as features of the shape model, to be validated with a simulated camera on DFL, and was dropped before any model was trained when the simulated camera failed two of its four fidelity gates (R019) |
| Tactical video (frames, audio) | only its structure (A04) | `ffprobe` read the stream information (H.264, 1920×1080, 25 fps, 199,319 frames; AAC audio from t = 0), and no value of the file was read (A04). The audio was decoded locally by our code (never seen by an LLM; aggregates only) for P10's band-pass whistle detector, which R018 and R018b dropped and whose code was removed. No source of the pipeline reads it (P14, the ball in the frames, is an idea, and since the organisers' clarification of November 2026 only a detector trained on other matches may run on it), and no LLM may see it, so what needs it is left to a person (A11), who may watch it only "to better understand what counts as possession". Nobody checks our output against it, which that clarification forbids |
| Broadcast video, ATD, homography | not used | The broadcast runs on its own 164,488-frame timeline (cuts, replays) and no mapping to the tactical frames is provided (T09), so the pipeline loads none of it. A04 read its structure only: the mp4 is one frame shorter than its ATD, and 34.2% of its homography rows are NaN. A20 aligned it to the tactical feed by code (two constants, one per half), and R028 tried filling our ball's gaps with its ball (P15, use 1). R028 was dropped at its stand-in's fidelity gate, so the broadcast stays unused |
| Public 5-minute ground truth | only to check the pipeline, and to inspect | RULES.md rule 2 (23 Oct 2026): "not training data and not a tuning set"; the organisers' clarification of November 2026 keeps it (no "training, fitting or calibrating your algorithm on the ground-truth possession data"; "running automatic checks using only the provided data" is allowed). It checks every file (valid, deterministic, each source's agreement; the score is not used to choose, T01) and it supplies the 20 changes (12 of team, 4 into `D`, 4 out of it) that A11 inspects, to understand how our truth flips against label-free events. Nothing is trained, calibrated or chosen on it, and no constant of the submission config comes from it (the audit in section 5). The one place the pipeline read it as an input was R012's g, which came out 0 for every type and is now retired, so the shift is s_b alone, from DFL. Nothing of this match was hand-labelled |
| Labelled tracking of other matches: the DFL/IDSSE open data, 7 matches (P01) | yes, to validate, to train P05's dead-ball model and P07's shape model, and to measure P02's lag | Bassek et al. 2025 (reference below), CC BY 4.0. We use the 25 Hz positions of players and ball, the ball's alive/dead flag and the team in possession on every frame (they give the truth: `D` where dead, otherwise the team), and the match information (teams, roles, pitch, halves). The tracking is degraded by the benchmark's profile and run through our pipeline: `generic` v1 since R030, which measures nothing of our match; until then `tracking` v3, fitted on label-free statistics of our tracking and retired after the organisers' reply of 5 Nov 2026 (section 5). The ball's alive/dead flag trains the dead-ball model (every fifth graded frame of the degraded matches, and each benchmark match is read by the model that never saw it). The team in possession trains the shape model of P07 (every fifth alive graded frame, each also as its team-swapped mirror, from 26 features that do not read the ball, and again each benchmark match is read by the model fitted without it); it is also used for validation and to measure the touch rule's confidence. For P02 the same matches give the pipeline's own latency against DFL's flag (R012) and DFL's flag timed against the same label-free events (A11). The only other model trained on it is P19's team model, dropped at its feasibility gate (R021); more labelled tracking (P13) is an idea |
| The DFL/IDSSE event files (the restart events); kloppy's DFL loader and download | the restart events only, as a diagnostic (P17); the loader and the download dropped | The raw events, one of the three files of a match, are read for the restart events only (below, and the labelled-data list), never by the pipeline and never to choose or fit anything. kloppy's loader drops every dead-ball frame by default and its download fails (T30), so our own parser (`pipeline/benchmark/dfl.py`) reads the positions, and kloppy only cross-checks it: 0 disagreements (A05) |
| Metrica's public sample games: the event files of 3 anonymised games (P16) | used once, as a measurement (R023, 31 Oct 2026); withdrawn on 5 Nov 2026 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) at commit `e706dd5` (2021), from the provider of our ground truth. The repository has no licence file; its README asks users to "be responsible" and to "acknowledge the source", which we do here. We asked the organisers whether that use was fine; they replied with their clarification ("any dataset you have the rights to use. Please check the license") and "anything that is not explicitly allowed is not permitted". A README's request is not a grant of rights, so we do not use the data again (docs/rules.md, question 3). `pipeline/benchmark/metrica.py` fetches the four event files (two CSVs, one JSON and its metadata) at run time into `data/external/metrica/`, checks each by its git blob SHA and never commits or redistributes them; no tracking file is fetched. Possession is derived from the events by a rule taken from Metrica's own event definitions and declared before the fetch, and its grain set which truth ranks later choices (below, section 5). Nothing is trained, fitted or chosen from them beyond that |
| SkillCorner's open data: automatic broadcast tracking of A-League matches | once, as a check (R030) | MIT licence. After `generic`'s gap law was written, the ball's gaps in three of its matches were read, to see whether the law looks like a real automatic tracker's gaps on other games: the ball is undetected on 39% to 46% of the frames, in gaps with a median of 5.4 to 5.8 s, a mean of 11.5 to 14.0 s and a 90th percentile of 27 to 36 s, against the law's 7.7 s, 14.4 s and 40 s (section 5). No number was taken from it, no label of it is read, and nothing is trained or chosen on it |
| Other external datasets or pretrained models | no | Nothing beyond the DFL/IDSSE matches, the SkillCorner check and Metrica's sample events above is used: P13 (more labelled tracking) and P14 (a pretrained soccer-ball detector) are ideas, not done; PathCRF's published checkpoint (P20, MPL-2.0; trained by its authors on five of the seven DFL matches) was run only on the held-out DFL match J03WR9, degraded like ours, where it beat P07 on the gap frames by +0.0049, short of the declared +0.04 (R022); it was never run on our match and was dropped |

**The DFL/IDSSE data: sources, citation, licence.** Seven full matches, two Bundesliga and five 2. Bundesliga, 25 Hz.
Reference: Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of spatiotemporal and event
data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. The data are released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) "with the authorization from the DFL" and are on
figshare (DOI 10.6084/m9.figshare.28196177, as the paper's reference list gives it). We change the data (we degrade
the tracking) and say so. The data are public and outside the competition bundle, so their own licence governs them and
not the bundle's rules. `pipeline/benchmark/sources.py` downloads the 21 files (2.6 GB) into `data/external/dfl/` (not
committed) and accepts a file only if its size and md5 equal the pinned values; a Hugging Face mirror of the same data,
pinned to one commit and checked the same way, is the fallback. Five of the seven matches are one club's home games.

**Labelled data (SUBMISSION.md: every labelled dataset, its source, who labelled it and how much; none of it from the
provided game).** Three kinds of label are used, all about other games: DFL's ball flags in the first row, DFL's restart events in
the second, and Metrica's sample events in the third.

| dataset | source | who labelled it | how much | used for |
|---|---|---|---|---|
| DFL/IDSSE open data: `BallStatus` (alive or dead) and `BallPossession` (the team) on every frame | Bassek et al. 2025, figshare, CC BY 4.0 (above) | DFL's official data (Bassek et al. 2025). The paper defines `BallStatus` and `BallPossession` but does not say who or what sets them. It says that the positions come from TRACAB Gen5 (ChyronHego), the match information from Sportec Solutions and the events from trained human operators, and that "all data are visually controlled by live operators for plausibility and undergo a final quality check after the game". So we treat the flags as human-checked labels | 7 full matches of other games (2 Bundesliga, 5 2. Bundesliga), 25 Hz: 668.4 graded minutes (about 1.0 million frames) per degradation seed, 437 possessions a match on average and 92 to 113 stoppages a match (A06) | the benchmark's truth (validation); the dead-ball model's labels (601,602 rows: every fifth graded frame, 3 seeds); the shape model's labels (331,881 rows, 663,762 with the team-swapped mirrors); the touch rule's confidence; the pipeline's own lag |
| DFL/IDSSE open data: the restart events (kick-off, throw-in, corner, goal kick, free kick, penalty, with the team that takes each) | DFL's raw event files, Bassek et al. 2025, figshare, CC BY 4.0 (above) | DFL's data collectors: the paper says the events come from trained human operators and that all data are visually controlled by live operators and pass a final quality check | 681 restart events over the 7 matches of other games (KickOff 22, ThrowIn 308, CornerKick 57, GoalKick 119, FreeKick 172, Penalty 3), placed on the benchmark's timeline by the position file's frame clock (`pipeline/benchmark/dfl_events.py`) | a diagnostic of the restart source's typing only (A12 part 2: recall per kind, a confusion of the called type against the event's kind); nothing is trained, fitted or chosen from it |
| Metrica's public sample games: the event files (type, subtypes, team, start and end frame of every event) | metrica-sports/sample-data at `e706dd5`; no licence file, its README asks users to acknowledge the source (done here); fetched at run time, never redistributed | Metrica Sports, the provider of our ground truth, tagging events by its own published definitions (`documentation/events-definitions.pdf`); the games are anonymised | 3 games of other teams: 7,300 events in periods 1 and 2, 1,117 derived possessions and 307 stoppages (R023) | R023 only: the grain of possession derived from them, which set the objective of later choices (section 5); nothing is trained, fitted or chosen from them beyond that. Withdrawn on 5 Nov 2026, as the data has no licence (above) |

Not in the list, and why: the public 5 minutes are published by the organisers, derived from Metrica's corrected
tracking, and are used to check only (never to train, fit or choose); the Smart Tagging is generated automatically by
Metrica "without manual review" (FAQ.md), so nobody labelled it. It was read from R017 to R029: the `smart_tagging`
source turned its tags into team votes (P08), and its one constant measured on this match, `c_team` 0.76, was the
tags' agreement with our own touch-rule pseudo-labels (an algorithm made them and no person decided any, so allowed
under RULES.md as it stood when R017 ran). The organisers' clarification of November 2026 requires parameters to be
tuned "on external data", so R029 took the source out, and the submission does not read the tags.
We produce no pseudo-labels that a model is trained on (the shape model learns from DFL's flag, not from the touch rule). We
labelled nothing by hand on any game.

No frame of this match was hand-labelled, none of its tracking was corrected by hand, and no human-produced data about
it (event data, corrected tracking, possession data, annotations) was used, for training, validation or the submission
(RULES.md, as clarified on 22 and 23 Oct 2026, and the organisers' emailed clarification of November 2026, which takes
precedence; docs/rules.md). The analyses of this document read its tracking and the public GT to
understand the data and to check the pipeline, and no label, correction or constant tied to its frames came out of them
(the audit in section 5).

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
- Where the ball is missing, the players are what is left to say who has it, and that is where P07 speaks. The shape
  model, scored out of fold on DFL's degraded matches (`generic` v1, raw truth, 7 matches x 3 seeds, pooled over
  frames; R030), has a log-loss of 0.3560 and an accuracy of 0.8458 on the alive frames whose served ball is missing
  (685,247 frames), and 0.3567 and 0.8454 on all alive frames (saying 0.5 on every frame scores ln 2 = 0.6931; the stop
  rule, declared before any config ran, was 0.6831). On the retired profile, where the model was first trained, the
  four figures were 0.3759, 0.8349, 0.3633 and 0.8422 (R011). It gives an opinion on `home` on every graded frame with
  a player present, ball or no ball. Its agreement with the public window is a correctness check (T01): R011's model
  agreed with the GT on `home` on 90.9% of the 6,031 frames it judged, and R030's on 91.2%.
  None of these figures says it is as right on our match, which it never saw (section 6, R011).

**The players** (46 slots for 22 players; measured on each half from its kick-off, 155,923 frames, unless said).

- On average 20.52 slots are present per frame; all 22 on 50.2% of the frames, 14 or fewer on 5.25% and 23 or more
  on 2.2%, which is more slots than players (A07, A09).
- A slot is a tracklet, not a player: 1,144 tracklets, median 41.3 s (35.5 s on the graded blocks), p10 2.7 s, p90
  297 s, mean 111.9 s, shortest 8 frames (0.32 s), longest 2,937 s (A07, A09). A slot idles between two tracklets for
  a median of 7.4 s, and 15.0% of those gaps last over a minute (A09).
- A slot can carry the other team's player: a team shows more than 11 slots while the other shows fewer on 10.8% of
  the graded frames (17,156 of 158,857), in 108 episodes (63 with the home team in surplus, 45 away), median 2.7 s, p90
  16 s, longest 50 s. This is a lower bound, because a swap inside 11 against 11 cannot be seen (T32, A07). They do
  not show up more at the ball: the share of `controlled_touch`-style candidate frames with a team over 11 slots is
  0.1097 against 0.1286 of all graded frames, a ratio of 0.85 against a declared 1.5, so the repair (P21) was dropped
  and A14's bound on the player degradation (0.013) applies. The comparison is against all graded frames while the
  candidates need the ball seen, so the ratio may partly reflect ball visibility (A16).
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
  (A06, T33). R023 measured it on other games of our provider instead (data withdrawn on 5 Nov 2026 for want of a
  licence, section 3): on Metrica's three sample games, possession
  derived from their events gives 6.20 possessions per alive minute (DFL's raw truth 8.30, the coarse 4.50) and a
  median of 4.72 s (3.68 and 8.40), between the two truths by its declared rule. It is event-tagged possession, not the
  tracking flag our GT comes from, and how far those two differ is not measured.
- DFL's flag turns dead a median 0.16 s after the ball is first seen beyond a line (DFL's own positions, A06). Ours is
  measured on 2 stoppages only: our tracking first sees the ball beyond a line a median 1.48 s after our `D` begins,
  where DFL's `tracking` v3 copy has it 0.12 s before its flag, a gap of +40 frames (1.6 s) under the raw and the
  coarse truth. That is our truth's convention or our tracker's loss of the ball near the lines, not separated (A11,
  T39).
- Two of the seven matches have a team down to 10 players: from minute 7.1 of the first half in one, from minute 13.4
  of the second half in the other. 18.3% of the seven matches' graded frames have 21 players, so a statistic on how
  many players are tracked is measured on the frames at full strength only (T36).
- DFL's players jitter by a few jumps on a 1 cm grid: the RMS of the second difference is 9.0 cm against our 3.3 mm,
  and a 5-frame average inside a tracklet takes it down to a fifth, 1.8 cm (T35, R004c; the retired default's
  smoothing). `generic` averages over 7 frames, the window of Shaw's velocity filter for 25 Hz tracking, and what that
  leaves is not measured against ours, by design.

## 4. Possession rules implemented

What the submission does today is `main.toml` as R030 left it: the components are R013's (R017's Smart Tagging team
votes were taken out by R029), and every numeric constant was chosen again on the benchmark's `generic` profile.
`controlled_touch` and `shape_possession` give the team,
`dead_ball_model` the ball in play, decoded by the `hmm` per half, then `shift_edges`. The rules below are those it
implements.

- **Control, who has the ball:** A team controls the ball when the nearest tracked player, of either team, is one of its
  own and within 1.5 m of the ball, no opponent is within 1.0 m, and the two move together, a relative speed of at most
  6 m/s, on a frame (R013's rule; R008's was 2.0 m and 3 frames). Both speeds come from successive positions, never from
  the ATD's speed channel (T31). The zones are Vidal-Codina et al.'s (2022, *Sports Engineering*): a zone around a
  player, and a 1 m duel zone around the ball where two opponents make a contested ball. Link and Hoernig (2017, *PLOS
  ONE*) likewise start an individual possession from the nearest player within a distance, and end it when the ball has
  left for a time. A run of control frames of one team is an episode, and it votes for that team at 0.93, a measurement:
  the share of episode frames on which the team is the truth's, floored to a hundredth, on the seven benchmark matches
  (93.5% on `generic`, R030; 93.1% on the retired profile, R013).
- **Back-dating:** The vote starts on the episode's first frame, not when a minimum of it has been seen, so a change
  of team lands where control began and confirmation costs no latency (T03). With R013's minimum of one frame there is
  nothing left to confirm; the decoder's minimum run is what asks for persistence (below).
- **Loose balls and contested phases:** In flight and in a duel no episode holds, and the source gives a weak hold
  instead: on every frame with the ball visible outside an episode, after the half's first episode, it votes 0.60 for
  the last controlling team, against the episodes' 0.93. The last team is never reset, so the hold also carries it
  through a stoppage. Where the ball is missing, and before the half's first episode, the source says nothing and the
  decoder keeps its state. R008 reads DFL's flag as turning at the new team's control and not at the old team's last
  touch (a reading and not a test: only 49.7% of its changes have an episode of the new team in the window). Without
  the hold the change lands on the first frame after the old team's last control frame (the decoder's tie rule), and
  with it on the new team's first. Without the hold no config of R008's grid beats either yardstick, and R013 found
  0.60 better than 0.55 under the raw truth and worse under the coarse one (T33). R024, ranking on the mean of the two
  (R023), chose 0.55. R030, on `generic` under the same objective, chose 0.60: 0.55 has the better mean by 0.0050, but
  0.60 is inside the 0.005 tie band (by 0.000004) and has the larger min(raw, coarse), which the declared tie-break
  takes (sections 5 and 6).
- **Possession where the ball is missing (P07):** `controlled_touch` is silent without the ball, so on its own the
  decoder would keep the last team through a gap, and the median gap lasts 10.7 s (section 3). `shape_possession` gives
  `home` there from where the two teams stand and how they move: how far up the pitch each is, how wide, its velocity
  along its attack and its shares in the two end thirds, over the frame and over the second or two around it. It is a
  learned probability and not a rule, read symmetrically in the two teams so that nothing in it says which one is home,
  clipped to [0.02, 0.98] and weighted 0.3 (R030; R011's 0.2 was chosen again by R013 and R024 on the retired
  profile). That weight lets it move
  the decoder by at most 1.17 nats a frame (3.9 x 0.3), more than the hold's 0.41 and less than an episode's 2.59
  (arithmetic from those constants, not a measurement),
  so it can turn a held team over a gap and cannot outvote a control episode frame for frame. It also speaks where the
  ball is visible, which R011 found better than speaking only in the gaps (+0.0113 pooled spellF1 raw at the chosen
  weight, `tracking` v3), and R030 found again on `generic`: speaking only in the gaps scores 0.5380 on the mean of the
  two truths against 0.5621, and loses 0.0440 coarse with 0 wins in 7.
- **P07 recalibrated per half from the touch rule's own labels (P22(b), tried and dropped):** an intercept added to
  P07's logit on each half, fitted on the other half's controlled-touch frames, allowing for the rule's 7% label noise
  and removing their class share (R027). It failed its gate on the seven DFL matches: in the held-out half's ball gaps
  P07's pooled log-loss rose from 0.3759 to 0.3820, where a fall of 0.002 was needed, and it rose on 5 of 7 matches;
  the retry's column, a team-aware linear head, did worse (0.4597). The offsets exist (0.28 in logit on average in the
  gaps), but the one the touches see is not the gaps' (0.30 apart, against 0.11 of label noise). It never ran on our
  match, and the code was removed.
- **The broadcast's ball in our gaps (P15, use 1; tried and dropped):** R028 aligned the broadcast by code and filled
  our missing ball from its usable, live-tested ball.
  - **The stand-in:** to value it on DFL, which has no broadcast, a simulated broadcast transplanted BAR-ATH's real fill
    gap by gap.
  - **Why it was dropped:** the stand-in failed its declared fidelity gates (fill on alive gap frames 0.218 against our
    0.295, on `D` frames 0.089 against 0.041). The one difference measured is the gaps' states: under the decoder, 54%
    of DFL's served gap frames are alive against our 20% (T38). We read that as the cause (T53). The code did what was
    declared (an identity test gave our placement statistics back exactly, F5 within 0.02 m), so the failure was the
    verdict.
  - **What is left:** the fill never ran into a score. Its value is untested, not refuted, and the code was removed.
- **Touches, deflections, aerial duels:** Since R013 a single control frame is an episode, so what keeps a deflection
  from being a possession is the decoder: an episode of the other team earns 2.59 nats a frame, a change of team costs
  6 nats each way, and the new team's run must last 35 frames, so a touch of a frame or two does not turn the
  possession unless the evidence around it agrees. The 1.5 m radius and the 6 m/s relative speed keep a ball that only
  passes a player out of the episode. An aerial duel has two teams within 1 m of the ball and so no vote. Neither is
  modelled further.
- **Change of possession:** A possession is a run of `H` or of `A`. The decoder holds the team until the evidence
  for a change pays for it: a change of team costs `switch_team` (6 nats) and a run lasts at least `min_possession` (35
  frames, 1.4 s) except where the half's edge cuts it short. Both are inside R013's grid: raw spellF1 peaks at
  `switch_team` 3 to 4 and `min_possession` 25, the coarse truth prefers fewer possessions, and the tie-break between
  the two set the minimum at 35 (R013, section 6, T33). R024 chose both again on the mean of the two truths and kept
  them: the mean alone climbs towards the coarse truth's corner, and the declared tie-break, the larger min(raw,
  coarse), held the choice at 4 and 35. R030 ran the same grid on `generic`: the mean alone again leans to the coarse
  corner (`switch_team` 8, `min_possession` 50), eleven configs tie with it, and the tie-break chose `switch_team` 6 and
  `min_possession` 35 (section 5).
- **Dead ball, what the detector sees:** Thirty features: the ball beyond a touchline or goal line, now, next and when
  last seen; the ball still at a restart spot (the centre spot, a corner, a penalty spot, a goal area); a collective
  slowdown of the players over windows before and after the frame; the kick-off shape (every player in their own half,
  the nearest player to the centre spot); the set-piece shapes (a crowd in a penalty area, a player at a corner point, a
  player beyond a touchline, players in a goal area); the gap between the teams and how spread out they are; and where
  the ball vanished and reappeared, which is what the tracker leaves of a stoppage inside a gap. R009's trees split
  first on where the ball is (beyond a line, next or last seen) and on whether a player stands beyond a line, and then
  on the restart spots and the players' speeds, and never on whether the ball is missing; the trees retrained in R030
  do not split on it either (the committed `full` model holds no such split). The model's probability
  is corrected for its training prior and clipped to [0.001, 0.999] before the decoder reads it.
- **Dead ball, before the kick-off:** A graded frame before its half's documented kick-off is `D`, whatever the model
  says. The rule comes from the organisers' documented kick-offs and from no ground truth, and on our match it makes
  the 2,934 graded frames before the second half's kick-off `D` (T02).
- **Dead ball, the decoder:** A change into or out of `D` costs `switch_dead` (32 nats), and a run of `D` lasts at
  least `min_dead` (25 frames). The second moves nothing: the two values differ by at most 0.0003 in spellF1 (R010). R030
  offered {16, 32, 64} x {1, 25} again on `generic`: the six score 0.5597 to 0.5623 on the mean of the two truths, and
  the declared rule kept 32 and 25 (Stage D).
- **A learned team model where the ball is seen (P19, tried and dropped):** boosted trees on the ball's distances to
  each team's nearest slots, closing and relative speeds, a time to reach (a 0.7 s reaction, then 5 m/s), the ball's speed,
  acceleration and turn, the touch episodes on either side of the frame and P07's shape, trained on the seven DFL matches
  (degraded as `tracking` v3), would have named the team on the frames where the ball is seen. It was dropped at its
  feasibility gate: out of fold it agreed with the per-frame truth on 0.9087 of the frames against 0.8927 for `main.toml`'s
  decoded labels, +0.0160 pooled against the declared +0.020 and in 3 of 7 matches against the 6 needed (R021). All seven
  matches were positive, mostly on the hold frames; the grid was not run and no spellF1 was produced.
- **Restarts (P17, not in the submission):** `restart_team` is a label-free source for who takes the restart after a
  stoppage. It was built and measured, and it does not clear the benchmark's bar, so `main.toml` does not use it. It
  finds the stoppages from the dead-ball model (a run of P(alive) below the alive level of at least 25 frames, its
  restart the frame after it), reads the players' arrangement on the stoppage's last 50 frames before the estimated
  restart (clipped to the stoppage), takes the type called on most of them, and votes for the team that type names.
  The Laws of the Game decide who restarts and how the players stand while it is set up; Vidal-Codina et al. (2022)
  detect set pieces from that configuration. The five rules, in order, each needing both teams on the frame; the
  opponents' clearance is **half** the law's, because players encroach:

  | type | where | who restarts | the law | opponent clearance used |
  |---|---|---|---|---|
  | kick-off | everyone in their own half, one team in the centre circle | that team | Law 8: 9.15 m | the other team outside it |
  | corner | a player within 3 m of a corner point | his team | Law 17: 9.15 m | 4.5 m |
  | goal kick | a player of the defending team in his goal area, at most one opponent in the penalty area | the defending team | Law 16: opponents outside the penalty area | one opponent may still be in the penalty area |
  | throw-in | a player within 1.5 m of a touchline or beyond | his team | Law 15: 2 m | 1.0 m |
  | free kick | with the ball only: the ball still, a player at it | his team | Law 13: 9.15 m | 4.5 m |

  It reads the players only, never the ball, unless `use_ball` (which only free kicks need, and which then restricts
  each rule to the ball's place). On the seven DFL matches the confidences (agreement floored to a hundredth) are,
  players only, at the 1 s and the 2 s tail: throw-in 0.92 and 0.88, corner 0.93 and 0.81, goal kick 0.72 and 0.72;
  kick-offs and free kicks are called too rarely and do not vote (a floor of 0.60 agreement and 30 calls a seed). The
  goal-kick rule is not specific (a keeper stands in his goal area in most stoppages: 134 of 608 matched throw-ins were
  called goal kicks), and its rules were not revised after that measurement (A12 part 2).

  A perfect 2 s vote was worth +0.0198 raw spellF1 to `main.toml` on the benchmark (A12), and two declared tests of
  the real source were run against a bar of +0.005 raw with a sign-flip p of at most 0.05, and neither met it:
  - **R014**, a vote for a fixed tail after the restart (25 to 150 frames, 8 configs): the best, 25 frames and players
    only, was +0.0040 raw (p 0.0938, 6 wins in 7) and -0.0029 coarse; every longer tail scored lower, down to -0.0187.
  - **R014b**, restart states in the `hmm` decoder (`restart_source`, off by default; the observation spread over the
    stoppage, 8 configs): +0.0003 raw (p 0.8125) and -0.0004 coarse. The four `switch_restart` values gave identical
    labels by construction: one stoppage's observation is at most about 2.6 nats (logit 0.93; a throw-in 2.44), and a
    cross transition costs `switch_restart`, so any value above about 2.6 decodes like `inf` (T44). Only the +0.0003
    says the observation is weak against the episodes that follow.

  The source and the decoder's restart states stay in the code, off by default, and `main.toml` is unchanged.
- **Restarts, who gets the ball (`main.toml` today):** No rule. After a stoppage the hold weakly favours the team that
  controlled the ball before it, until the first new episode, which names the team, and the shape source gives a view of
  its own from where the teams stand. The restart-team channel above (a corner to the attackers, a goal kick to the
  defenders) is not in `main.toml`. Its restart types would have to come from code (Smart Tagging's set-piece tags, P08,
  or other games' flags, P16): a type that a person read off this match's video for each stoppage and fed to the
  pipeline would be a hand label, while a person watching to understand what counts as possession at a restart is
  allowed (A11, section 6; the organisers' clarification of November 2026).
- **Dead ball, the truth's convention (A22):** The ground truth is `D` where Metrica's corrected tracking marks the
  ball dead, and no organiser text says whether that flag is set when the ball crosses the line, at the whistle or at
  the restart kick; the public question that asks (issue #31, 3 Nov) is unanswered, as is the one on whether a touch
  without control is a possession (#17). The Laws of the Game, Metrica's own event definitions ("out of play: the ball
  went out of bounds or the referee has stopped play"; a set piece "brings the ball back into play") and our timing of
  the four public stoppages (A11) point the same way: dead when the ball leaves play or the referee stops it, alive when
  the restart is taken. The dead-ball model carries DFL's timing, which turns dead 0.16 s after the ball is seen beyond
  a line and alive 0.12 s after it moves; if our provider's flag sits earlier and later than that, our stoppages are
  short at both ends (section 6).
- **Abstentions (`X`):** The decoder never abstains inside a graded half, and the runner writes `X` on the ungraded
  frames, where it is free: the chosen file has none inside a graded block (R010 to R024). It can return the posterior
  of each state per frame. A free-`X` refiner built on it was measured in R016 and dropped (macroF1 fell at every threshold).
- **Edges placed by the posterior (P23 C, tried and dropped):** each change moved to the median of the decoder's
  posterior within its window, then on a posterior tempered at T = 5 (R025, R025b). Neither passed its guard (mean of
  the two truths -0.0010, then -0.0282): the posterior is four to five times too sharp (A19, section 5), so its median
  moved an edge by a median of 1 frame, and tempered it moved them 3 frames and widened their errors. The code was
  removed.
- **Edge timing, where each rule flips against our ground truth, and the shift (A11, R012, T40):** The back-dating,
  the hold and the `D` onset follow DFL's flag, so we timed every change of the public truth (12 of team, 4 into `D`, 4
  out of it) against label-free events on our tracking, and the same events against the seven DFL matches' flags on
  `tracking` v3 (tracking degraded to look like ours; each DFL figure is raw truth, and the coarse truth's is given
  where it differs). Our truth flips a median 24 frames (0.96 s) before the new team's first controlled touch, where
  DFL's flag flips 1 frame after it (on it under the coarse truth); 10 changes found, and the gap's 90% interval holds 0
  under both truths. Our `D` begins a median 1.5 s (37 frames, on 2 of 4 stoppages) before the ball is first seen beyond
  a line, where DFL's flag on the copy turns 0.12 s after it, under both truths (A06's 0.16 s is DFL's own positions).
  Our `D` ends 1.0 to 1.5 s later than DFL's flag would, against the restart's first controlled touch (a gap of -26 raw,
  -28 coarse) and against the dead-ball model's alive crossing 0.5 (-36.5 under both). Taken at face value, our
  stoppages are 2.5 to 3 s longer than DFL's flag draws them. Three explanations fit and the data do not separate them:
  a convention of our truth; chance, at 2 to 4 changes in 4 stoppages; and tracker coverage, which predicts all four
  signs by itself (our tracker loses the ball near the lines, T39, so `line_out` and the model fire late and the model's
  alive rises early on ours, and the copy's ball gaps ignore the play, R003). So it is a lead, to understand and not to
  fit: a person may watch the video to see which it is, to understand what counts as possession, and what is seen sets
  no constant (T27); nobody checks our own output against the video (the organisers' clarification of November 2026).
  The questions on it (5 and 6 in docs/rules.md, A22) were drafted but not sent: the organisers answered our earlier
  four on 5 Nov with their clarification alone. Metrica's sample games (P16) were the one lawful-looking source of a
  timing constant, and they carry no licence, so since 5 Nov no timing constant has a source we may use, and the
  convention stays a limitation (section 6).
  The `shift_edges` rule is declared, was applied in R012 and was applied again in R013 to the re-chosen pipeline,
  whose lag on DFL came out the same. R024 and R030 applied it again, R030 on `generic` with a tool that reads DFL's
  lag and nothing of our match (`p27_shift.py`), and each got the same three values. The shift of a type of edge is s_b, the pipeline's own
  lag on the seven DFL matches (the median of predicted minus true over the changes it matches within 50 frames,
  negated; team +1, into `D` -2, out of `D` 0 frames; kept by a guard on the benchmark), plus g, a gap to our truth's
  convention read from the public window under conditions declared beforehand (a median lag of at least 3 frames in
  absolute value, a 90% bootstrap interval that excludes 0, every leave-one-out median with its sign). g came out 0 for
  every type, at 4, 1 and 1 matched changes of the three types, in an unusually lively 5 minutes (A06). **g is retired**
  since the organisers' rule of 23 Oct 2026: it reads the public window as an input, which is a tuning use, so it stays
  0 by rule, the rule's table still prints it for reference, and any future shift comes from other games alone (P16). So
  the shift is s_b alone: a change of team 1 frame later and a change into `D` 2 frames earlier. It is within noise on
  the benchmark (section 5), and it is here because the rule says so.

## 5. Validation

**Why not the public window.** The organisers' rule (RULES.md, as clarified on 23 Oct 2026, and kept by their
clarification of November 2026) is that its 5 minutes are for checking that the pipeline is correct and are "not
training data and not a tuning set". It would also mislead. It
holds 17 possessions, so one possession moves spellF1 by about 0.06. It is the best-covered 5 minutes of the match (the
ball is missing on 12% of its frames, against 40.7% overall), and it ranks `smooth125` above `smooth50`, the reverse of
the full match (T01). We may not hand-label this match, even to validate (T27). So the window checks that the pipeline
is correct (alignment, the file, determinism, each source's agreement) and is inspected to understand this ground
truth's timing conventions (P02), never to set a constant, and we choose between versions on other matches.

**What set each constant of `main.toml` (the audit, rewritten for R030 on 10 Nov 2026).** Every constant is traced to
its entry in `docs/results.md`, and three tests apply. None may be set from the public window (the 23 Oct rule). None
may be measured on this match at all, labels or not: the organisers' clarification of 3 Nov has parameters tuned "on
external data" only, and that test took `smart_tagging` out (R029, the last row). And since the organisers' reply of
5 Nov ("anything that is not explicitly allowed is not permitted"), the data a constant is chosen on must itself owe
nothing to our match. Until then the benchmark's degradation, `tracking` v3, was fitted on label-free statistics of
our tracking. We had read that as allowed and asked (question 1 in docs/rules.md); after the reply, by our reading, it
is not permitted, and neither is R023's use of Metrica's unlicensed sample data, which had set the objective of R024's
choice. On 9 Nov Tian decided to redo the degradation (P27) and to keep the objective, the unweighted mean of the
raw-truth and coarse-truth pooled spellF1, on A06's ground alone: the grain of our truth is unknown, so neither truth
is preferred. R030 is that redo. Both models were retrained and every numeric constant was chosen again on `generic`
v1, a profile that no measurement of ours enters ("Profiles", below). The table says what sets each constant today;
the entries that set it before R030, on the retired profile, are the history further down this section. What R030 did
not choose again is the architecture: which components exist, the trees' settings, `min_prob` 0.02 and the hmm's
structure were adopted on the retired benchmark, and on `generic` the bar and Stage D's ablation re-verify them. The
honest wording of the claim is: every numeric constant was chosen, and both models were trained, on `generic`; the
architecture was inherited and re-verified. docs/rules.md has the audit against the clarification and the questions.

| constant | value today | chosen or measured in | on what data |
|---|---|---|---|
| `controlled_touch` radius, max_rel_speed, min_frames | 1.5 m, 6.0 m/s, 1 | R030 Stage A: R013's 192-config grid, by the rule declared before the runs (unchanged from R013) | the seven DFL matches, `generic` v1, 3 seeds, both truths; no window-derived value in any config; the zone follows Vidal-Codina et al. (2022) |
| `controlled_touch` confidence | 0.93 | R030 Stage 1, `touch_stats.py --grid r013 --profile generic --seeds 3`: a measurement, not a choice | the floor, to a hundredth, of the pooled agreement of the kept episodes with DFL's flag over the seven matches on `generic` (93.5%; 90.6% to 95.2% per match): in sample there (section 6), never on the window or on our match |
| `controlled_touch` hold | 0.60 | R030 Stage A (0.55 from R024 to R029) | the same grid's {0.52, 0.55, 0.60, 0.70} on `generic`: 0.55 has the best mean, 0.60 is inside the 0.005 tie band (by 0.000004) and has the larger min(raw, coarse), so the declared tie-break takes it |
| `controlled_touch` duel_radius | not set (default 1.0 m) | the source's default | a fixed rule (Vidal-Codina et al.'s 1 m duel zone), never varied |
| `dead_ball_model` model | `dead_ball_hgb.json` | R030 Stage 0, `train_dead_ball.py --model hgb --profile generic --seeds 3` | trees trained on DFL's `BallStatus` over the seven matches as `generic` degrades them (601,602 rows; settings fixed beforehand and not tuned, as in R009), each benchmark match read by the fold that never saw it and BAR-ATH by `full`; kept over the logistic reference retrained the same way (pooled held-out log-loss 0.0788 against 0.1524, lower in 7 of 7 matches); its feature thresholds are fixed rules (the laws of the game, the literature) |
| `dead_ball_model` weight | 1.0 | R030 Stage D, among {0.1, 0.25, 0.5, 1.0} (unchanged from R009) | DFL on `generic`: the best mean and the larger minimum; no window-derived source in any config |
| `dead_ball_model` min_prob | not set (default 0.001) | the source's default | a fixed default, a clip to [0.001, 0.999] (also P(alive) before a kick-off); never varied |
| pre-kick-off rule | `D` before a half's kick-off (P(alive) 0.001) | `play_features`, R009 | a fixed rule from the organisers' documented kick-offs (FAQ.md: the second half is graded from 110,808 and kicks off at 113,742), from no ground truth |
| `shape_possession` model | `shape_possession_hgb.json` | R030 Stage 0, `train_shape.py --profile generic --seeds 3` | trees trained on DFL's `BallPossession` with the ball hidden, on `generic` (settings fixed beforehand, as in R011), each benchmark match read by the fold that never saw it and BAR-ATH by `full`; the stop rule (held-out log-loss on ball-missing frames below 0.6831) holds at 0.3560 |
| `shape_possession` weight | 0.3 | R030 Stage C: R013's 144-config grid (0.2 from R011 to R029) | DFL on `generic`: the weight axis {0.1, 0.2, 0.3, 0.5} crossed with the decoder, by the declared rule |
| `shape_possession` scope | `all` | R030 Stage D, against `gaps` (unchanged from R011) | DFL on `generic`: `gaps` scores 0.5380 on the mean of the two truths against 0.5621 |
| `shape_possession` min_prob | 0.02 | R011, part 1; **not chosen again** | fixed beforehand, a clip that caps a frame at 3.9 nats; never varied in any grid, so it is a fixed rule inherited from the retired benchmark's day, not a tuned value |
| `hmm` switch_team, min_possession | 6, 35 | R030 Stage C (4 and 35 from R013 to R029) | DFL on `generic`: the grid spans `switch_team` 1 to 8 and `min_possession` 1 to 50, and the choice is on no edge; the best mean is at 8 and 50, eleven configs are within 0.005 of it, and the tie-break (the larger min(raw, coarse)) takes 6 and 35 |
| `hmm` switch_dead, min_dead | 32, 25 | R030 Stage D, among {16, 32, 64} x {1, 25} (unchanged from R010) | DFL on `generic`: six tied within the band, and 32 / 25 has the larger min(raw, coarse) (0.5410 against 0.5407) |
| `hmm` eps | not set (default 1e-6) | the decoder's default | a fixed default, a numerical guard on the clip and not a knob |
| `shift_edges` team, into_dead, out_of_dead | +1, -2, 0 frames | R030 Stage E, by R012's rule (the values R012, R013 and R024 found) | s_b alone: minus the rounded median lag of the pre-shift config on DFL (`generic`, raw truth, 3 seeds; 2,967, 1,114 and 1,645 matched changes), read by `p27_shift.py`, which reads nothing of our match; kept by the declared guard; g, from the public window, is 0 and retired |
| `seed`, `scope` | 0, `halves` | fixed | a seed for reproducibility; the scorer grades halves |
| `smart_tagging` c_team; weight, spread (until R029) | 0.76; 0.2, `frame` | R017; out of `main.toml` since R029 | kept for the record. c_team was `towards_half` of the measured agreement (0.761, 90% interval [0.723, 0.795]) of the tags' team votes with our own controlled-touch pseudo-labels on BAR-ATH; the strength was Tian's pick (2026-10-30) of one of two declared options, untested on any benchmark because DFL has no tags. **Both conflicted with the organisers' clarification of November 2026** (tuning on external data only; docs/rules.md), so R029 took the source out |

Two sources do carry constants measured on the public window, and neither is in `main.toml`: `nearest_player`'s
confidence 0.78 and `ball_presence`'s 0.2 and 0.585 (A01, P06; R006 to R010 and `p06_hmm.toml`). Since the 23 Oct rule
they may not enter a submission config unless re-derived from other games (A06 measured the nearest player's agreement
on the DFL matches, 77.5%).

**Where the window-derived values sat in the choices.** In R006 to R009 every candidate config (the yardsticks aside,
which carry neither) carried at least one of those two sources: all 24 configs of R008 held `ball_presence` at A01's
values, with R007's decoder (8 / 32 / 75 / 1, itself chosen with both sources), and all 24 of R009 held `nearest_player`
at 0.78. So the touch rule's radius, relative speed, minimum frames and hold (R008) and the `hgb` source and its weight
1.0 (R009) were chosen on DFL with window-derived values present in every candidate. R010 chose the decoder again (4 /
32 / 25 / 25) with neither source present, but R008's and R009's constants were not chosen again, and 4 and 25 sat on
the lower edge of R010's grid. We judged this compliant: the window's labels never scored a candidate, and the
window-derived values were part of the fixed context of selections scored on DFL's flags. **R013 removed the question
for the touch rule and the decoder**: it chose the touch rule again (192 configs) and then `switch_team`,
`min_possession` and P07's weight (144 configs) on `main.toml`'s own evidence, which has neither source, by rules
declared before its runs. The touch rule moved (R008's setting ranked 83rd of 192 under the raw truth), and the decoder
moved by one value (`min_possession` 25 to 35). R013 did not choose R009's source and weight again, for reasons it
declared: the source rests on the out-of-fold log-loss, which no other source enters, and a weight of 1.0 takes the
model's probability as it is. **R030 closed that too.** Its configs carry `main.toml`'s three sources and no other;
Stage D chose the weight again among {0.1, 0.25, 0.5, 1.0}; and the trees were kept over the logistic reference
retrained on `generic` by their held-out log-loss (0.0788 against 0.1524, lower in 7 of 7 matches). So no choice that
stands today had a window-derived value among its candidates.

Two further uses of the window are not constants. The agreement figures that `--score` prints are outputs and never
inputs (86.8%, 86.5% and 91.2% for R030's file). And the fidelity report of the retired profiles set two statistics of
the window (the nearest player's agreement, the share of dead frames with the ball visible) beside the degraded copy
as warnings, with no profile constant fitted to them (`pipeline/benchmark/README.md`); `generic` is not run through
`fidelity.py` and reads nothing of the window.

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
check, not a correction, and a ranking that differs between the two truths is a warning. R023 (P16, section 3) found
our provider's grain between them on Metrica's sample games, so its declared outcome, "both", makes the unweighted mean
of the two pooled spellF1s the objective of every choice declared after it (31 Oct 2026), each truth reported. R024
re-ranked R013's two stages under it (below); no other earlier choice was re-ranked. Since 5 Nov the sample data is
withdrawn (section 3: it has no licence), so the grain of our truth is unknown again, as A06 left it; the mean of the
two truths is still the neutral way to rank when neither is known to be right. Tian kept it as the objective on
9 Nov 2026, and R030 chose every constant again under it.

**Profiles.** A profile says how a DFL match's tracking is made before a config sees it, and its version goes into
every report. `generic` is the default since R030, and the only profile anything is trained or chosen on. `gaps`,
`tracking` and `context` read a `Calibration` measured on our match; they stay in the code for the record of R002 to
R029, marked as reading our match (`degrade.NEEDS_CALIBRATION`), and nothing is trained or chosen on them again.

| profile | the ball | the players | entry |
|---|---|---|---|
| `clean` v1 | DFL's, on every frame | DFL's, one slot per player | R002, the floor |
| `generic` v1, **the default since R030** | missing on 41% of the frames, in gaps whose lengths are log-uniform between 1 s and 60 s and whose onsets are a Poisson process blind to the play; its speed NaN | each on-pitch player lost on 6.8% of the frames, independently, in 46 slots; a 7-frame moving average | R030 |
| `gaps` v1, retired | missing as ours was: our own missing-ball masks, transplanted onto each half (their pattern, not their clock); its speed NaN (T31) | untouched | R003 |
| `tracking` v3, the default from R004c to R029, retired | `gaps`' ball, byte for byte | our tracklet lengths in 46 slots, wrong-team slots (T32), NaN-speed runs, 5-frame smoothing, and our low-count frames (only the players nearest the ball) | R004 to R004c |
| `context` v1, sensitivity only, retired | the gaps placed where our tracker loses the ball: a hazard fitted on our match (A08) | `tracking`'s | R005: fails two of its gates |

**`generic`, version 1** (R030; `pipeline/benchmark/degrade.py`; the spec, written before any run, is
`docs/superpowers/specs/2026-11-09-p27-degradation-design.md`). No measurement of ours enters it, and no `Calibration`
is loaded for it: `degrade.needs_calibration("generic")` is False, the runner and the trainers refuse a `Calibration`
for it, and a check proves that `calibration.load` is not called. Three sources shape it and nothing else: a fact the
organisers publish in the challenge documents, a rule declared before any run with its reason, and the file format.

| part | setting | source |
|---|---|---|
| ball missing, share of the frames | 0.41 | DATA.md: the ball is missing on 41% of the graded frames |
| length of a gap | log-uniform between 1 s and 60 s | declared. DATA.md names the causes (occlusions, crowded boxes, balls in the air) and no length. A ball in the air is gone for about a second, a crowded box or a stoppage for up to a minute, and log-uniform makes every scale between them equally likely |
| where the gaps fall | a Poisson process, blind to the play | declared: the simplest process with that share (a limitation, section 6) |
| ball speed | NaN on every frame | declared: the profile promises no ball speed, so no component can lean on one |
| slots | 46: 24 home, 22 away | the file format |
| players lost | each on-pitch player on 1 - 20.5 / 22 = 6.8% of the frames, each on its own | the rate is DATA.md's (a mean of 20.5 tracked objects over the graded frames); that the players are lost independently is declared |
| length of a tracked span | log-uniform between 2 s and 300 s | declared. No component reads a slot's identity, so this sets only how often a player drops out |
| length of a lost span | geometric, mean 4.4 s | follows from the two rows above |
| smoothing | a centred 7-frame moving average over each tracked piece | declared, from the literature: the window of Shaw's velocity filter for 25 Hz tracking (Friends of Tracking, `Metrica_Velocities`) |
| wrong-team slots, NaN-speed runs, frames with few players, a shortest tracklet | none | DATA.md documents none of them |

- **The residual risk.** The two figures quoted from DATA.md (41% and 20.5) are statistics of this match. The
  organisers publish them in the task statement, as they publish 25 fps and 46 slots. But no text allows quoting them
  by name, so a risk remains, and we state it.
- **What the declared gap law gives:** a median gap of 7.7 s, a mean of 14.4 s and a 90th percentile of 40 s, by
  arithmetic.
- **A check on external data, made after the law was written.** In SkillCorner's open data (MIT licence, automatic
  broadcast tracking, three A-League matches read) the ball is undetected on 39% to 46% of the frames, in gaps with a
  median of 5.4 to 5.8 s, a mean of 11.5 to 14.0 s and a 90th percentile of 27 to 36 s. So the declared law looks like
  a real automatic tracker's gaps on other games. No number was taken from the check.
- **The check of the profile itself** (`p27_profile_check.py`, counts only, the 21 match-seeds degraded again). The
  pooled share of the ball's masks is 0.4109 (1,236,063 of 3,007,932 rows) against the declared 0.41 +-0.02, and the
  pooled share of on-pitch player-frames lost is 0.0681 (4,467,232 of 65,622,702) against 0.0682 +-0.01. Both pass.
  Per match and seed the mask share runs from 0.374 to 0.474.
- **No fidelity gate against our match, by design.** `fidelity.py` measures a profile against the `Calibration`, and
  `generic` is never run through it. Version 1 is final for this submission: a version 2 would be made only on a rules
  ground, never after a score.

**The retired profiles (history).** Version 3 of `tracking` (R004c) changed one rule of version 2, so that a slot is taken by a new tracklet only from the row after its
last one stopped, and derived the scale of the tracked spans again by the same one-shot rule, because the slot rule
moved its measurement (2.042 to 2.072; T37). All eight gates pass, no check warns and the ranking does not move. What
moved is the account of the jitter, 1.80 cm per slot after smoothing, where R004b's 8.5 cm was slot handovers (T35).
`context` fails two of its gates (R005, section 6): the near-line share badly, 0.5586 of its gaps starting within 2 m
of a line against our 0.2797 (+-0.05), because DFL tracks the ball out of play, and the share missing narrowly, 0.4313
against 0.4110 (+-0.02). The near-line gate binds only `context`: the default's transplanted gaps measure 36.0% on the
same statistic (R003). `context` was a sensitivity check, the default stayed `tracking` until R030, and the
four yardsticks kept their order on it.

**Fidelity of the retired default, as of R004c** (history; `tracking` v3: 7 matches x 3 seeds, `fidelity.py --profile
tracking`, which exited 0). These gates belong to the retired profiles only: `generic` has none against our match. A gate
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

**The rule on degradations until 9 Nov 2026 (history).** A degradation was fitted only on label-free statistics of
our tracking (A07, A08), with constants from rules written before any score was seen, rescaled once per version from
one label-free measurement, frozen with their measurement in the code, and never tuned towards a published score, a
yardstick or a WARN (T26, T37). When a gate missed (the first low-count cap, on the tracklet median) we stopped, said
so and made two corrections, each forced by its measurement: no piece under 8 frames, and one rescale of the tracked
spans, applied whether or not the gate would then pass (R004b, T37). The degrader never read `ball_status` or
`ball_possession`, and a change to a profile bumped its version. We read this rule as allowed until 5 Nov 2026 and
asked the organisers to confirm it. Their reply ("anything that is not explicitly allowed is not permitted") makes it,
by our reading, a use of the provided game in the data our models learn from, and nothing allows that explicitly
(docs/rules.md, question 1). So the rule, and the three profiles made under it, are retired.

**The rule on degradations since R030.** A profile takes its settings from a fact the organisers publish in the
challenge documents, from a rule declared before any run, or from the file format, and from nothing we measured on our
match: not A01 to A03, not A07 to A09, not the `Calibration`, and no number picked because it lands near one of those.
The last part cannot be proved from the inside, since we have seen those measurements. So each declared rule is a
round range with a stated reason, and none is tuned. The degrader still reads no flag of the truth, and a change to a
profile still bumps its version.

**Statistics on seven clusters.** The pooled spellF1 is the official scorer's number on all halves as one file. An
interval is the 90% Student t interval of the seven matches' paired deltas against a reference config, beside the exact
sign-flip test (smallest p 2/128 = 0.0156) and the matches won out of seven. It is never a bootstrap: seven matches,
five of them one club's home games, are not a sample to resample. A degradation seed moves a pooled spellF1 by up to
about 0.015 (R003 to R004c), which the intervals do not show, so three seeds are the default.

**Does it rank the four baselines as the full match does?** We run the organisers' four baselines decoded per half
(`nearest_w{0,25,50,125}`) as a check, not a target. Pooled spellF1 on the retired default, `tracking` v3 (R004c; the other profiles'
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

On `generic` (R030) the two yardsticks that the bar reads score 0.1885 raw and 0.1698 coarse (`w50`) and 0.1616 raw
and 0.2092 coarse (`w125`). The raw truth puts `w50` first and the coarse one `w125`, as on every earlier profile.

**From here to A19: the record on the retired profiles.** The paragraphs below, from P06's decoder to R024 and A19,
tell how each component was adopted. Every score in them is on `tracking` v3 (some also on `context`), profiles that
read our match. They are history: since R030 nothing in `main.toml` stands on them. R030's re-choice on `generic`
closes this section, before "Why these choices".

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

**What better evidence would be worth** (A10). The chosen decoder run with the benchmark's own truth as evidence
gives an upper bound, since the oracles are exact: a perfect dead-ball detector takes pooled spellF1 to 0.3678 raw and
0.4417 coarse, a perfect possessor where the ball is seen to 0.3586 and 0.3651, and both everywhere to 0.7627 and
0.9226. Under the raw truth the first two are level (+0.0092 apart, p 0.2500); under the coarse truth the dead-ball
detector is ahead by +0.0766 (p 0.0156). A perfect possessor inside the gaps adds +0.0008 raw until `D` is right and
+0.1971 once it is, so the possession-without-the-ball work (P07) was worth nothing to this decoder before a dead-ball
source (P05) existed; with P05 in, R011 measures it below. The full-oracle row is a cap on this config and not on the
approach: 0.7627 is below 1.0 because the penalties and the 75-frame minimum, chosen under noisy evidence, merge the
shortest of DFL's possessions (the logs cannot say how much each costs). Apart from A01's three evidence values, fixed
before P06, the public window was used only to check that the pipeline is correct (the file is valid and identical
across runs), never to choose (T01).

**P04 and P05 on the benchmark** (R008, R009, R010). Each source was chosen on its own grid by the same kind of
protocol (profile `tracking` v3, 3 seeds, the pooled raw spellF1 as the objective, a 0.005 tie band, then the coarse
score), and the two were then crossed with the decoder's penalties. The bar for adopting a config was declared before
the runs: its raw gain over `nearest_w50` and its coarse gain over `nearest_w125` both above 0 with a sign-flip p of
0.05 or less, and, declared before Stage C ran, a label-free check on our match that the file is not
degenerate.

- **P04** (R008): 24 configs, radius {1.0, 1.5, 2.0} m x relative speed {3, 6} m/s x `min_frames` {3, 8} x hold {off,
  0.55}. Chosen: 2.0 m, 6 m/s, 3 frames, hold 0.55, confidence 0.91, with R007's decoder and `ball_presence`: 0.2345
  raw and 0.2433 coarse, +0.0328 over `nearest_w50` raw (p 0.0156, 7 wins in 7) and +0.0206 over `nearest_w125` coarse
  (p 0.0156, 7 in 7). At R007's decoder it adds +0.0292 raw and +0.0466 coarse over `nearest_player`.
- **P05** (R009): 24 configs, source {fixed rules, logistic regression, gradient boosting} x weight {0.1, 0.25, 0.5,
  1.0} x `ball_presence` {off, on}. Chosen: the boosting model at weight 1.0 without `ball_presence`: 0.3503 raw and
  0.4150 coarse, alive/dead F1 0.9725 (the yardsticks' 0.6094 and 0.6428; an upper bound for our match, since DFL
  tracks the ball out of play, T39). Boosting led logistic regression by
  0.0274 raw and had the lower out-of-fold log-loss on 7 of 7 matches (0.0837 against 0.1526 pooled).
- **Both, with the decoder re-tuned** (R010): 72 configs, `switch_team` {4, 8, 16} x `switch_dead` {16, 32, 64} x
  `min_possession` {25, 50, 75, 100} x `min_dead` {1, 25}. Chosen: 4, 32, 25, 25.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | alive/dead F1 | possessions per minute |
|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | 0.1842 | 0.4592 | 0.6094 | 4.30 |
| `nearest_w125` | 0.1665 | 0.2226 | 0.4407 | 0.6428 | 3.05 |
| R007's `hmm` (8, 32, 75, 1) | 0.2053 | 0.1967 | 0.4569 | 0.5676 | 3.10 |
| P04 + P05, chosen (4, 32, 25, 25) | 0.4217 | 0.4556 | 0.8275 | 0.9726 | 2.74 |
| without P04 (`nearest_player` instead) | 0.2796 | 0.2424 | 0.7796 | 0.9726 | 5.09 |
| without P05 (`ball_presence` instead) | 0.2556 | 0.2170 | 0.4851 | 0.5749 | 3.05 |

**The bar is met under both truths**: raw +0.2199 over `nearest_w50` (mean per match +0.2217 [+0.2019, +0.2415]), coarse
+0.2330 over `nearest_w125` ([+0.2202, +0.2480]), sign-flip p 0.0156 and 7 wins in 7 each. The leave-one-match-out
selection picks the chosen config in all seven folds, and on `context` (sensitivity only) it is above both yardsticks by
+0.2130 and +0.2211. At the chosen decoder each source is needed: without P04 the coarse half of the bar fails (+0.0198,
p 0.1406) and without P05 it fails (-0.0056, p 0.1094), while the raw half holds without either; alone, at R007's
decoder, each clears both halves (R008, R009). At R007's decoder the two gains add, and a little more (0.3873 raw
against 0.3795 for the sum). Predicted `D` on our match is 36.0%, inside the label-free check's band [30%, 62%], with no
`X` inside a graded block and the same sha256 on two runs. On the public window, a correctness check only (T01), the
file has macroF1 0.6996, spellF1 0.4686 and 18 possessions against 17.

These benchmark figures are upper bounds for our match, not predictions: DFL tracks the ball out of play, so the
benchmark overstates P05 by an amount it cannot measure (section 6, T38, T39).

**P07 on the benchmark** (R011). The shape model was scored out of fold first (section 3), and then 12 configs added it
to R010's config with the decoder unchanged: weight {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} x scope {the frames with no ball,
every graded frame}. The protocol was declared before any of them ran: the objective is the pooled raw spellF1 on
`tracking` v3 (3 seeds), configs within 0.005 of the best count as tied, the bar is a raw gain over R010's config with a
sign-flip p of 0.05 or less, `context` (sensitivity only) is a veto and the coarse truth a warning, and a label-free
transfer gate on our match must pass (section 6). Chosen: weight 0.2, every graded frame, the only config within 0.005
of the best (the next, weight 0.1, is 0.0102 below). Pooled spellF1 on `tracking`, 3 seeds:

| config | raw spellF1 | coarse spellF1 | raw macroF1 | alive/dead F1 | possessions per minute |
|---|---|---|---|---|---|
| P04 + P05, R010's `main` (4, 32, 25, 25) | 0.4217 | 0.4556 | 0.8275 | 0.9726 | 2.74 |
| + P07 at weight 0.2, frames with no ball only | 0.5046 | 0.5262 | 0.8799 | 0.9729 | 3.33 |
| + P07 at weight 0.2, every graded frame (chosen) | 0.5159 | 0.5591 | 0.8861 | 0.9729 | 3.17 |
| the truth | | | | | 4.58 raw, 2.48 coarse |

**The bar is met**: +0.0942 raw over R010's config (mean per match +0.0940 [+0.0775, +0.1105], sign-flip p 0.0156, 7
wins in 7), +0.1035 coarse, and on `context` +0.0831 raw and +0.0890 coarse, 7 wins in 7 in each cell. All 12 configs
beat R010's in all four cells (the smallest +0.0290), with scope `all` (every graded frame) the gain grows with the
weight up to 0.2 and falls at 0.5 in all four cells, and a leave-one-match-out selection picks the chosen config in
all seven folds.
The label-free checks on our match pass: the transfer gate at 0.8251 on 41,052 frames against its declared 0.7486
(section 6), no `X` in a graded block, the same sha256 twice, predicted `D` 35.8% against R010's 36.0%, and 439
possessions against 450 (4.1452 per graded minute against 4.2491), the labels differing from R010's on 5.30% of the
graded frames. On the public window, a correctness check only (T01), spellF1 is 0.4000 against R010's 0.4686 (20
possessions against 18; one possession moves it about 0.06) and macroF1 0.7151 against 0.6996. The benchmark's gain is
not a forecast for our match (section 6).

**P02, the shift on the benchmark** (A11, R012). R012 measured the pipeline's own latency on the seven DFL matches
(`tracking` v3, raw truth, 3 seeds): a predicted change is matched to a true one of the same pair of labels within 50
frames, and the lag is predicted minus true, positive when our label changed late. Pooled over matches and seeds, a
change of team has a median lag of -1 frame (quartiles -10 and +20; 43% matched, 3,011 matched changes), a change into
`D` +2 (0 and +2; 50%; 1,069) and a change out of `D` 0 (-3 and 0; 79%; 1,657). The shift is minus the rounded median:
team +1, into `D` -2, out of `D` 0. A guard kept it, as one config (R011's `main.toml` plus these three values) against
R011's `main.toml`, pooled spellF1 in four cells:

| profile | truth | `main` | with the shift | pooled Δ | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|
| `tracking` | raw: the guard's | 0.5159 | 0.5164 | +0.0005 | 0.5000 | 4 / 3 |
| `tracking` | coarse | 0.5591 | 0.5592 | +0.0000 | 1.0000 | 3 / 2 |
| `context` | raw: sensitivity only | 0.5153 | 0.5162 | +0.0009 | 0.2500 | 4 / 3 |
| `context` | coarse: sensitivity only | 0.5544 | 0.5544 | +0.0001 | 0.9531 | 4 / 3 |

Every 90% interval of the mean per-match gain includes 0 (the raw one on `tracking` is [-0.0007, +0.0015]), and the only
consistent effect is +0.0003 to +0.0004 macroF1 in all four cells. The rule asked for a positive raw gain and no cell
below -0.005, which holds, so the shift is adopted: by the rule, and not because it is shown to help. The gap to our own
truth's convention (g) that the rule read from the public window came out 0 for every type, at 4, 1 and 1 matched
changes, and is retired (section 4). On the public window, a correctness check only, spellF1 is 0.4000 with the shift in
sample and out of fold and without it (the same 20 possessions), and macroF1 0.7159 in sample, 0.7125 out of fold and
0.7151 without. On our match the shift moves the labels of 0.34% of the graded frames, exactly the shifted boundaries,
and leaves 439 possessions.

**The touch rule and the decoder chosen again** (R013). A protocol declared before any run re-chose, in two stages on
`tracking` v3 (3 seeds, both truths from one decode, the same rule as R008 to R011: the best pooled raw spellF1, a 0.005
tie band, then the best coarse), the constants that had been chosen with the window-derived sources present or
against an older decoder. Stage A crossed the touch rule's radius {1.5, 2.0, 2.5, 3.0} m, relative speed {6, 9, 12}
m/s, minimum episode {1, 2, 3, 5} frames and hold {0.52, 0.55, 0.60, 0.70} on R012's `main.toml` (192 configs, each at
its measured confidence); Stage C crossed `switch_team` {1, 2, 3, 4, 6, 8}, `min_possession` {1, 10, 15, 25, 35, 50}
and P07's weight {0.1, 0.2, 0.3, 0.5} at Stage A's choice (144 configs). The shift was then re-derived by R012's rule
(the same +1, -2, 0) and the result guarded against R012's `main.toml`. Pooled spellF1:

| config | `tracking` raw | `tracking` coarse | `context` raw | `context` coarse | possessions per minute |
|---|---|---|---|---|---|
| R012's `main.toml` | 0.5164 | 0.5592 | 0.5162 | 0.5544 | 3.17 |
| Stage A's choice (1.5 m, 6 m/s, 1 frame, hold 0.60; decoder 4 / 32 / 25 / 25) | 0.5473 | 0.5485 | | | 3.27 |
| R013's final config (the same touch rule; decoder 4 / 32 / 35 / 25; P07 at 0.2) | 0.5430 | 0.5593 | 0.5489 | 0.5642 | 3.13 |
| the truth | | | | | 4.58 raw, 2.48 coarse |

The final config is +0.0267 raw over R012's on `tracking` (mean per match +0.0264 [+0.0191, +0.0338], sign-flip p
0.0156, 7 wins in 7) and level coarse (+0.0001, 4 wins and 3 losses), and on `context` (sensitivity only) +0.0327 raw
and +0.0097 coarse. The guard (a raw gain on `tracking`, no cell below -0.005) passes and the bar is met, and the
leave-one-match-out estimates agree (+0.0306 raw for Stage A's rule, +0.0263 for Stage C's, 7 wins in 7 each). The
truths disagreed again (T33): Stage A's touch rule alone gained 0.0310 raw and lost 0.0107 coarse (0 wins in 7), and the
longer minimum possession that Stage C's tie-break chose won the coarse cell back. On our match the label-free checks
pass: `D` 35.9% of the graded frames, no `X` in a graded block, the same sha256 twice (and under one BLAS thread), 410
possessions against 439, 2.4% of the graded frames relabelled. On the public window, a correctness check only (T01),
spellF1 is 0.4649 and macroF1 0.7152 with 20 possessions (R012's file: 0.4000, 0.7159, 20).

**The restart team, tested and dropped** (R014, R014b). A12 measured what a perfect 2 s vote for the restarting team
would add to `main.toml` (+0.0198 raw, 7 wins in 7), and two protocols declared before their runs tested the label-free
source against a bar of +0.005 raw with a sign-flip p of at most 0.05, on `tracking` v3 with both truths. R014 voted for
a fixed tail after the restart (8 configs): the best, 25 frames, was +0.0040 raw (p 0.0938, 6 wins in 7) and -0.0029
coarse, and every longer tail scored lower. R014b put the call into the decoder as restart states (8 configs, all within
0.0005 of each other, identical labels by construction, T44): +0.0003 raw (p 0.8125). Neither met the bar, nothing was
adopted, and the oracle's gap was partly information a fixed-length vote does not have (it stopped where the truth next
changed, T43) and partly the rest of the distance to a real source: coverage (73.2% of true restarts called),
agreement (72 to 93% by type), the error in the estimated restart frame, and the vote's strength (the oracle's about
13.8 nats a frame against about 2.4 for a 0.92 vote).

**Where the remaining score is** (A25, on `generic`, with today's `main.toml`). A perfect team where the ball is seen
would add +0.199 raw spellF1 (+0.215 coarse) and in the gaps +0.180 (+0.104), against +0.028 (+0.039) for a perfect
dead-ball source, so what is left is who has the ball, and not whether it is dead, on the benchmark (T39). With both
perfect the decoder reaches 0.917 raw and 0.969 coarse. A13 measured the same on the retired `tracking` v3 (+0.194,
+0.176 and +0.029 raw). The ladder's loss is mostly late edges on possessions that are otherwise right (0.068 of
spellF1 on the mean, against 0.026 each for merges and splits), and the decoder's posterior is four to six times too
sharp (A25; A19 on the retired profile).

**R013 chosen again on the mean of the two truths** (R024, P23 A). A protocol declared before the selection re-ranked
R013's two stages under R023's objective: the mean of the pooled raw and coarse spellF1s on `tracking` v3 (3 seeds), a
0.005 tie band, then the larger min(raw, coarse) among the tied, then R013's declared orders (`choose.py --objective
mean`). Stage A′ re-ranked R013's 192 Stage A reports and moved only the hold, from 0.60 to 0.55 (two settings tied, and
0.55 had the larger minimum). Stage C′ ran R013's 144 Stage C configs at that hold: the best mean was `switch_team` 6,
`min_possession` 50 (0.5601, but 0.5178 raw), 14 configs were tied with it, and the tie-break chose `switch_team` 4,
`min_possession` 35 and P07 at 0.2, `main.toml`'s own, inside the grid on every axis. The shift came out +1, -2, 0
again by R012's rule, and its guard kept it. Pooled spellF1:

| config | `tracking` raw | `tracking` coarse | mean | `context` raw | `context` coarse | possessions per minute |
|---|---|---|---|---|---|---|
| R017's `main.toml` (with `--drop-match-only`: R013's config) | 0.5430 | 0.5593 | 0.5511 | 0.5489 | 0.5642 | 3.13 |
| R024's final config (hold 0.55; decoder 4 / 32 / 35 / 25; P07 at 0.2; shift +1, -2, 0) | 0.5363 | 0.5764 | 0.5563 | 0.5394 | 0.5756 | 3.06 |
| the truth | | | | | | 4.58 raw, 2.48 coarse |

The final config is +0.0052 on the mean on `tracking` (mean per match +0.0048 [+0.0010, +0.0087], sign-flip p 0.0469,
6 wins in 7), which is -0.0067 raw (1 win) and +0.0171 coarse (7 wins), and +0.0010 on the mean on `context`
(sensitivity only). The guard (the mean's gain above 0 on `tracking`, not below -0.005 on `context`) passes, and the
bar, reported and not required, is met. The rule's leave-one-match-out estimate is weaker: +0.0035 [-0.0004, +0.0074],
p 0.1562, 5 wins. On our match the label-free checks pass: `D` 36.00% of the graded frames (35.93% before), no `X` in a
graded block, the same sha256 twice, 399 possessions against 413 (mean 10.2 s), 1.66% of the graded frames relabelled.
On the public window, a correctness check only (T01), spellF1 is 0.4000 and macroF1 0.7007 with 19 possessions (R017's
file: 0.4000 and 0.6986).

**Where the decoder loses, and two ideas that did not recover it** (A19, R025, R025b, R026; P23 B to E). A19 split the
fall of `main.toml`'s spellF1 from IoU 0.3 to 0.7 with the scorer's own matching (`diagnose.py`, `tracking` v3, 3
seeds). Most of it is edges placed off on possessions that are otherwise right, not merges or splits: if every clean
pair matched at 0.3 were also matched at 0.7, pooled spellF1 would gain 0.0722 raw and 0.0686 coarse (0.0704 on the
mean), against 0.0254 for the merges and 0.0273 for the splits. A wrong edge is most often late (the clean pairs' end
errors have quartiles of -6 and +22 frames, raw). Against the raw truth, the sources are close to calibrated as
likelihoods (Platt α 1.00 for `dead_ball_model`, 1.07 to 1.10 for P07 with the ball seen or missing; the touch rule's
two fixed votes a little conservative, 1.22), and the decoder's posterior is four to five times too sharp (α 0.27 for
alive, 0.19 for the team), which a product of per-frame terms over correlated frames gives (T50). Two declared tests followed, each with a
guard on the mean of the two truths and one retry at most, and both were dropped:
- **C, edges from the posterior** (R025, R025b): moving each change to the posterior's median within its window moved
  55% of the changes, by a median of 1 frame (mean -0.0010, 2 wins in 7); the retry, the posterior tempered at T = 5
  from A19's α, moved them 3 frames and widened the edges' errors (mean -0.0282, 0 wins). Code removed, recoverable
  from `171d0bb`.
- **E, P07 weighted apart in the gaps** (R026, 6 configs): the rule kept `main.toml`'s weights, and every heavier gap
  weight cut more possessions (3.22 to 3.35 a minute against 3.06) and lost on the mean (-0.010 to -0.020), the coarse
  truth most. No retry, by the advisor's ruling for Tian (A19 shows P07 calibrated alike in both ball states). Code
  removed, recoverable from `57013a1`.

What is left of the edges' loss is in where the evidence changes, not in where the decoder puts a change given the
evidence (P23).

**Everything chosen again on `generic`** (R030, P27; declared 9 Nov 2026, run 9 to 10 Nov). The spec was written
before any run and not changed after a result (`docs/superpowers/specs/2026-11-09-p27-degradation-design.md`). Tian was
away for the night, so the spec's open points went to an advisor agent before the runs, and the rules were applied by
code (`p27_choose.py`), not by a reading of the tables. Every run is `--profile generic --seeds 3`, the seven DFL
matches, both truths. The objective is R024's, which Tian kept: the unweighted mean of the two truths' pooled
spellF1s, a tie band of 0.005, then the larger min(raw, coarse), then the stage's own tie order.

- **Stage 0, both models retrained** with the settings they had (seeds 0 to 2, every fifth graded frame, the trees'
  settings unchanged). *Dead-ball model:* held out per match, log-loss 0.0654, 0.0901, 0.0725, 0.0852, 0.0650, 0.0918
  and 0.0823 and AUC 0.9942 to 0.9973; pooled log-loss 0.0788 and Brier 0.0227 over 601,602 held-out rows; the
  reliability table is flat. The logistic reference retrained the same way has a pooled held-out log-loss of 0.1524
  (Brier 0.0427, AUC 0.9831 to 0.9890), and the trees are lower in 7 of 7 matches, so they stay. That was the
  advisor's condition; it ran after the grids, since only one heavy process runs at a time, and had it failed the
  grids would have been void. *Shape model:* pooled held-out log-loss on alive frames with the ball missing 0.3560
  (accuracy 0.8458, 685,247 frames) against the stop rule's 0.6831, so the rule does not fire; on all alive frames
  0.3567 and 0.8454. Its `touch_agreement` per held-out match is 0.8586, 0.7426, 0.8313, 0.8486, 0.8462, 0.8351 and
  0.8679 (pooled 0.8334), so the transfer gate on our match is at least 0.7426.
- **Stage 1, the touch rule's confidence**, measured and not tuned (48 settings). The incumbent 1.5 m / 6.0 m/s / 1
  frame covers 41.9% of the alive frames and agrees with DFL's flag on 93.5% of its episode frames (90.6% to 95.2% per
  match), so the confidence is 0.93, as on the retired profile (93.1%). It finds a new-team episode at 53.1% of the
  truth's changes of team, with a median lag of -1 frame (quartiles -6 and +4).
- **The reference** (`p27_ref`), the incumbent constants before the shift, on the retrained models: 0.5388 raw, 0.5779
  coarse, mean 0.5584; macroF1 0.893; alive/dead F1 0.9740; 3.03 possessions per graded minute; `D` 44.6%. R029's
  `main.toml` scored 0.5363 and 0.5764 on `tracking` v3. The two profiles are different benchmarks, so that is no
  comparison of them.
- **Stage A, the touch rule** (R013's grid, 192 configs). The best mean is the incumbent itself, and four configs are
  within 0.005 of it. Among them the rule takes the larger min(raw, coarse): the hold 0.60's 0.5463 against the
  incumbent's 0.5388. So the rule stays 1.5 m / 6.0 m/s / 1 frame and the hold goes from 0.55 to 0.60. Against the
  reference that is raw +0.0076 (p 0.0156, 7 wins in 7), coarse -0.0176 (p 0.0156, 0 wins) and mean -0.0050 (p
  0.1406). **A knife-edge:** 0.60 is inside the band by 0.000004 (its mean is 0.004996 below the best, on the
  unrounded scores), and 0.52, at 0.005054 below, is outside. Leave one match out, the whole rule picks 0.55 in four
  folds and 0.60 in three: mean -0.0048 [-0.0099, +0.0002] against the reference.
- **Stage C, the decoder and the shape weight** (R013's grid, 144 configs, at Stage A's choice). The best mean is
  `switch_team` 8, `min_possession` 50, weight 0.3: 0.5653 (0.5242 raw, 0.6064 coarse), which is +0.0069 on the mean
  over the reference (p 0.0156, 7 wins) and -0.0146 raw. Eleven configs are within 0.005 of it, and the larger
  min(raw, coarse) among them is `switch_team` 6, `min_possession` 35, weight 0.3: 0.5621 (0.5410 raw, 0.5833 coarse),
  0.0032 below the best mean and on no edge of the grid. Against the reference: raw +0.0022 (p 0.2969, 5 wins), coarse
  +0.0054 (p 0.0469, 6 wins), mean +0.0038 (p 0.0781, 6 wins). **A second knife-edge, on the other side:**
  `switch_team` 6 with `min_possession` 25 has a larger minimum (0.5437) and is outside the band by 0.000034. Leave one
  match out, the whole rule scores -0.0009 [-0.0047, +0.0030] against the reference: out of sample its picks are no
  better than the incumbent constants. The objective alone picks the best-mean config in all seven folds (mean
  +0.0071, 7 wins in 7; raw -0.0142, 1 win; coarse +0.0284).
- **Stage D, what R009, R010 and R011 chose** (11 configs: the incumbent and ten variants), one group at a time at
  Stage C's choice (mean 0.5621), each by the same
  rule, the incumbent being one candidate among the others. `switch_dead` x `min_dead`: 64/25 0.5623, 64/1 0.5623,
  32/25 0.5621, 32/1 0.5620, 16/25 0.5601, 16/1 0.5597; six tied, and 32 / 25 has the larger min(raw, coarse) (0.5410
  against 0.5407). The dead-ball weight: 1.0 0.5621, 0.5 0.5614, 0.25 0.5581, 0.1 0.5196; 1.0 has the best mean and
  the larger minimum. The shape scope: `all` 0.5621, `gaps` 0.5380 (coarse -0.0440, 0 wins in 7). The shape source:
  with it 0.5621, without it 0.4584 (-0.1037; 0 wins in 7 under both truths), so it stays. No group changes.
- **Stage E, the shift** (R012's rule: the lag of the pre-shift config on `generic`, raw truth). A change of team:
  7,053 changes of the truth, 2,967 matched, median lag -1 frame, so +1. Into `D`: 2,130, 1,114, +2 frames, so -2.
  Out of `D`: 2,097, 1,645, 0, so 0. They are the three values R012 and R013 found on the retired profile. The guard,
  the shifted config against the unshifted one: mean +0.000029 (3 wins, 4 losses), raw +0.0004, coarse -0.0004. The
  declared guard keeps a shift whose mean is above the unshifted config's with neither truth more than 0.005 below, so
  the shift is kept, by a hair. On spellF1 it is worth nothing either way (macroF1 +0.0003, 6 wins in 7).

| config (`generic` v1, 3 seeds) | raw spellF1 | coarse spellF1 | mean | possessions per minute | `D` share |
|---|---|---|---|---|---|
| `nearest_w50` | 0.1885 | 0.1698 | | | |
| `nearest_w125` | 0.1616 | 0.2092 | | | |
| the incumbent constants (`p27_ref`) | 0.5388 | 0.5779 | 0.5584 | 3.03 | 44.6% |
| R030's final config, `main.toml` today | 0.5414 | 0.5829 | 0.5622 | 2.98 | 44.7% |
| the truth | | | | 4.58 raw, 2.48 coarse | 44.8% |

**The bar (R010's) is met on `generic`**: raw 0.5414 against `nearest_w50`'s 0.1885, +0.3529, and coarse 0.5829 against
`nearest_w125`'s 0.2092, +0.3738, each with a sign-flip p of 0.0156 and 7 wins in 7. Per match the final config scores
0.4865, 0.4784, 0.5713, 0.5729, 0.5437, 0.5769 and 0.5785 raw (mean 0.5440, 90% interval [0.5120, 0.5761]) and 0.5926,
0.5375, 0.5830, 0.5685, 0.6262, 0.5961 and 0.5826 coarse (0.5838, [0.5639, 0.6037]); macroF1 is 0.8934 and 0.8941 and
alive/dead F1 0.9741. Against the incumbent constants, for information: mean +0.0038 (p 0.0312, 6 wins in 7), raw
+0.0026 (p 0.2188, 5 wins), coarse +0.0050 (p 0.0469, 6 wins). The re-choice is worth a third of a point on its own
benchmark; what it buys is that every constant now stands on external data only.

**The checks on our match are pass or fail only and choose nothing** (T01; declared in the spec): that the file
validates, has no `X` inside a graded block and has the same sha256 on two runs; that its predicted `D` share lies in
[30%, 62%] (R010's band, set on DFL); that the retrained shape model's `touch_agreement` on our match is at least
0.7426; and that the public window, scored once, shows no gross error (every source's agreement above 0.5, and the
file's macroF1 above each constant file's). All pass (R030). The gate is at 0.8199 on
41,052 frames (R011's model: 0.8251 against 0.7486). `D` is on 34.5% of the graded frames (`H` 38.5%, `A` 27.0%), in
393 possessions with a mean of 10.6 s. The sha256 is `2dd51320…` on both runs, each from a rebuilt ATD cache, and
again from a clean clone (section 7). On the public window macroF1 is 0.7238 against the constant files' 0.2102,
0.1705 and 0.1092, and each source agrees on more than half of its frames (`controlled_touch` 86.8% of 5,484,
`dead_ball_model` 86.5% of 7,500, `shape_possession` 91.2% of 6,031); for the record only, its spellF1 is 0.5029 with
18 possessions against 17. Both models' `--verify` pass, and so do the self-tests (192 of 192, 194 with `--data`; the
benchmark's 472).

**Why these choices.** Lang et al. (2022, *Scientific Reports*) predict the in-play status from tracking and find
boosting the best of four classical families (92.0% frame accuracy, against 86.7% for logistic regression), from the
players alone and so in the regime of our frames without a ball. Their features are shifted values of the players'
positions, speeds and accelerations, and ours are windows of the same kind. Vidal-Codina et al. (2022, *Sports
Engineering*) give the set-piece triggers that became features and the zone sizes for control, and report that the
best radius depends on the provider, which is why the radius is a grid and the confidence a measurement. Link and
Hoernig (2017, *PLOS ONE*) show control as a proximity held over time, with the possession ended by the ball leaving.
The model's settings (learning rate 0.05, 300 trees, 15 leaves, at least 1,000 training rows a leaf, no early
stopping, no class weights) come from general boosted-tree practice and not from these papers (Lang et al. used
AdaBoost): a leaf of at least 1,000 rows (every fifth frame over three seeds, so about 200 s of sampled frames and at
least about 67 s of distinct play), no early stopping on autocorrelated frames, the log-loss and no class
weights, so that the probabilities stay probabilities. They were fixed and not tuned on seven matches. The evidence
reaches the decoder as probabilities that are corrected for the training prior and clipped, so that the decoder's own
penalties set the prior and no frame is certain. P07's shape model takes the same settings from the same code, with no
monotone constraint, and each training row is also shown as its team-swapped mirror with the label flipped, so that the
classes balance exactly and the model has no prior to subtract and no notion of which team is home (R011).

## 6. Limitations

<!-- Where it fails, and what we would fix with more time. The pipeline's own limits: fill as decisions land. -->

**Of the benchmark's profile today** (`generic` v1, R030), on which both models and every numeric constant now rest:

- **Its gaps ignore the play, and ours do not.** They are a Poisson process, and DATA.md says our ball goes missing
  "disproportionately on the frames that matter". So the benchmark still overstates the dead-ball model (T38, T39) and
  cannot value a feature about where the ball is lost.
- **It has no wrong-team slots and no long NaN-speed runs**, which our tracking has (T31, T32), because DATA.md
  documents neither. The retired benchmark put the whole player degradation at 0.013 spellF1 (A14), so the cost is
  expected to be small, and it is not measured on our match.
- **The gap law is one declared range.** Another range would give other constants, and no second range was run.
- **Two figures are quoted from DATA.md**, and no text allows quoting them by name (section 5, the residual risk).
- **No fidelity gate says how far it is from our tracking**, by design: a profile may not be changed to pass one
  (docs/rules.md), and `generic` is never run through `fidelity.py`. So nothing measured says whether it is nearer to
  our match than `tracking` v3 was.
- **The two truths still pull the hold and the decoder in opposite directions (T33), and both of R030's choices sit
  on that trade-off.** The hold 0.60 came in through a tie band that it is inside by 0.000004: another set of seeds
  could leave it out, and on the objective alone 0.55 is 0.0050 better. The decoder is the tie-break's choice over a
  best-mean config that is +0.0069 on the mean and -0.0146 under the raw truth, against the incumbent constants. Both
  were decided by the rule as declared, and both are listed for Tian. Three runs made after the choice, declared
  first and choosing nothing, size it (R030, "After the choice"). At six seeds the hold 0.60 alone is 0.0084 below
  0.55 on the three fresh seeds, so the tie did not repeat. The whole protocol re-run from the hold 0.55 ends at
  0.5374 raw and 0.5903 coarse (mean 0.5639) against the declared 0.5414 and 0.5829 (0.5622): the same score, leaning
  to the coarse truth. And the declared constants, that alternative, the best-mean decoder and the constants before
  R030 lie within 0.0054 on the mean at six seeds, with no difference clearing its test.
- **Out of sample the re-choice is no better than the constants it replaced.** Leave one match out: -0.0048
  [-0.0099, +0.0002] for Stage A's rule and -0.0009 [-0.0047, +0.0030] for Stage C's. What R030 buys is the ground
  the constants stand on, not a score.
- **The architecture was inherited.** Which components exist, the trees' settings, `min_prob` 0.02 and the hmm's
  structure were each adopted on the retired benchmark by a bar. On `generic`, Stage D's ablation re-verifies the
  shape source and the bar the whole; nothing re-verifies the rest one by one.
- **The reports' commit fields carry `+dirty`**: configs and docs were being written while the runs went. No pipeline
  code changed between `95c75ce` and the last run.

**Of the validation benchmark (P01) in general, and of its retired profiles** (a bullet marked retired describes
`tracking` v3 or `context`, on which R002 to R029 ran):

- **Conventions.** DFL's flags may flip at other moments than ours do, and the grain of our ground truth is not known:
  the public window is alive on 80.4% of its frames, more than 96% of DFL's 5-minute windows, so it cannot say how far
  ours differs from DFL's 437 possessions a match (A06, T33). The coarse truth stands in for the difference and does not
  correct it, and `D` timing against ours is measured on four stoppages only (A11, T40). The pipeline's own lag on DFL
  is adopted as a shift of 1 to 2 frames (R012), and a gap to our convention read from the public window is not: it
  came out 0 and the term is retired, because the window is no tuning set (P02, section 4).
- **Seven matches, five of them Fortuna Düsseldorf's home games.** The sample is small, so the intervals are wide, and
  not independent, so even they may be too narrow; the smallest sign-flip p is 0.0156. Two matches have a team down to
  10 players, so statistics on how many players are tracked are measured at full strength only (T36).
- **No video, audio, homography or Smart Tagging.** Whatever uses them (P08, P14, P15) is checked on the
  public window for correctness only, with its strength fixed by a declared constant. P09 tried to simulate the camera
  on DFL from noise measured on our match and was dropped when the simulation failed two of its four fidelity gates
  (R019). P22(a) shifted the shape model's logit by the home share that EM estimates from its own outputs, and was
  dropped at its feasibility gate on the seven DFL matches (the share's mean error 0.0538 against 0.0623 for guessing
  0.5, needing 0.75 of it; R020): the model's outputs are not a calibrated estimate of a match's possession share, and
  no score was produced. P08 was not checked on the public window before its
  adoption, and its strength (weight 0.2, spread `frame`) was Tian's pick between two declared options: no constant came
  from the window. After the adoption `selftest.py --data` scored the window as a correctness check (macroF1 0.6986,
  spellF1 0.4000, against R013's 0.7152 and 0.4649; one possession moves spellF1 by about 0.06 there), and those figures
  chose nothing (T01). R029 took P08 out again under the organisers' clarification of November 2026.
- **The tags are left out (R029).** Their team votes covered 49.9% of the ball-gap frames (R017), but nothing external
  could set their confidence, so they were dropped under the rules. Against R024's file, taking them out changed 2.94%
  of the graded frames (42.3% of them where the ball is not seen) and moved the possessions from 399 to 413; what the
  tags were worth was never measured, because DFL has none. The benchmark always ran `main.toml` without them, so its
  scores are those of the submitted config.
- **DFL tracks the ball out of play, and ours does not.** We tried two profiles that hide it (R015, R015b) and dropped
  both. One matched our share of visible frames past a line (0.0414 against 0.0407) but made the gaps too long (median
  15.04 s against 10.74); the other could not reach the share at all (0.258 to 0.262). With the ball hidden by the first,
  today's dead-ball model lost 0.013 alive/dead F1 (0.9731 to 0.9600), a lead and not a measurement (T39, T45).
- **Aerial projection.** A camera-based tracker projects an airborne ball onto the pitch plane, which displaces it.
  DFL's positions carry no such error and the benchmark adds none, and ours has no ball height, so the error cannot be
  measured on ours. Nor can we tell how many of our ball gaps come from airborne balls: our first plan said "crowded
  and aerial", and measured it is "lines and slow balls" (A08).
- **Position bias.** None is added, because none can be measured on ours (there is no reference to measure it against),
  so a method that reads distances is validated without whatever positional error ours carries.
- **Rougher positions (retired default).** The jitter after smoothing was 1.80 cm per slot against our 3.3 mm, 5.5 times ours in RMS (T35,
  R004c), which matters to a method that reads positions, speeds or accelerations.
- **The retired default's ball gaps were transplanted and blind to the play.** They keep the pattern of our masks, not their
  clock (the offset is drawn uniformly, so the pattern lags by up to 7.6 minutes in a 68,000-frame half, `degrade.py`;
  R003's draws reached 11,222 frames) and not where our tracker loses the ball (near the lines, on slow balls, with
  more players around, A08). 36.0% of them start within 2 m of a line against 28.0% of ours, because DFL's ball is
  tracked out of play and so spends more time near the lines (R003, R005), and the ball is visible on 0.5684 of the
  dead frames against 0.6637 in the public window (R003). A method that depends on where the gaps fall is validated on
  gaps that ignore the play (T38).
- **`context` did not fix that, and why (retired).** It places the gaps by the play and fails two of its gates: the near-line
  one (the default has none) on every seed, 0.5586 against 0.2797 (+-0.05), and the share missing narrowly (0.4313
  against 0.4110, +-0.02). The first is the data's doing: DFL tracks the ball out of play, so 25.2% of its
  at-risk rows are within 2 m of a line against 9.0% of ours, while the model's probability rises near a line by a
  similar factor on both (3.9 on DFL's rows, 2.9 on ours). No scale of the hazard moves it and no remedy was applied.
  It stays a sensitivity profile, and its numbers are not set beside `tracking`'s without saying so (R005).
- **Tracklets and slots were not ours (retired default).** The tracklet median is 26.52 s against our 41.28 s, 36% below and inside its
  +-50% band (R004c, T37). A player returns to his own slot 47.4% of the time (ours has no identities), and a slot
  idles a median of 4.3 s where ours idles 7.4 s (both on J03WMX, seed 0, R004c, A09). A method that follows a slot
  across a loss would be validated optimistically.
- **The cap kept the players nearest the ball (retired default).** Where ours has 18 or fewer slots present, the profile keeps the
  players nearest the true ball. A nearest-player method barely notices (the four baselines move by 0.4 points of
  spellF1 at most, R004b), and one that needs the players far from the ball sees fewer of them than the real match gave
  it, as it does on ours. The cap's counts come from the frames of the ball's gaps and not from the same play.
- **The halves.** A benchmark half starts at its kick-off, while our graded second half starts 117 s before it, all
  `D` (A06, T02).

**Of P04, P05 and the decoder** (R008 to R010; all three are still in the submission, and the figures are R010's
unless a bullet cites another entry: they were measured on the retired profile, and R030's on `generic` are in
section 5):

- **The domain shift: DFL tracks the ball out of play.** The trees read where the ball is, and in DFL's matches it is
  tracked beyond the lines and at restart spots during stoppages, which our tracker loses (among rows where a gap could
  begin, the ball is beyond a line on 17.68% of DFL's against 4.07% of ours, R009). R010's benchmark alive/dead F1 of
  0.9726 (0.9729 with P07, R011) is therefore not our match's, and the gain over the yardsticks is overstated by an
  amount the benchmark cannot measure (T38, T39). Neither `tracking` nor `context` removed it, and `generic` does not either: all three keep DFL's
  out-of-play ball (alive/dead F1 0.9741 on `generic`, R030). The raw half of the bar does not rest on P05 at R010's decoder (the config without it still beats
  `nearest_w50` raw by +0.0539), and the coarse half does. The two looks at our match are the public window and the predicted `D` share. For R010's file they
  were an alive agreement of 86.6% of 7,500 frames, and 36.0% against 44.6% on the benchmark and 40.7% to 51.7% in
  DFL's matches; R030's are pass-or-fail checks (section 5). How much `D` our match has is not known beyond the public window.
- **The two truths disagree on the decoder (T33).** The raw truth chose the shortest minimum possession and smallest
  switch penalty of the grid and the coarse truth the longest and the largest (`switch_team` 16, `min_possession` 100:
  0.5049 coarse and 0.3532 raw, against R010's choice at 0.4556 coarse and 0.4217 raw). R010's config predicted 2.74
  possessions a minute (3.17 with P07, R011), between the raw truth's 4.58 and the coarse truth's 2.48. R013 met the
  same split: the raw truth wanted a stronger hold and more possessions, the coarse truth fewer (its best Stage C config
  predicts 2.79 a minute), and the final config is +0.0267 raw and level coarse against R012's. If our truth is as fine
  as DFL's, R013 helps; if it is coarse, it is no better. R023 put our provider's event-derived grain between the two
  truths, so later choices rank on their mean, and R024 re-ranked R013's grids on it: only the hold moved (0.60 to
  0.55), +0.0052 on the mean, which is -0.0067 raw (1 win in 7) and +0.0171 coarse. If our truth is as fine as DFL's,
  R024 costs a little; the mean alone would have gone further towards the coarse truth (`switch_team` 6,
  `min_possession` 50, 2.74 possessions a minute), and the declared tie-break kept the choice near the raw side. R030 met the same
  split on `generic` and resolved it the same way, twice on a knife-edge (the first group of this section).
- **R013 selected from 336 configs on seven matches.** The leave-one-match-out estimates agree with the in-sample
  gains, but Stage C's coarse tie-break picked the final config in five folds of seven, and the two stages ran in order
  (Stage C only at the full-data Stage A choice). R024 selected again from the same 192 reports and 144 new runs, and
  its rule's leave-one-match-out estimate is weaker than its own gain (+0.0035, p 0.1562, against +0.0052, p 0.0469):
  Stage A′'s whole rule picked R013's hold of 0.60 in two folds of seven. R030 selected again from the same two
  grids on `generic`, and its rules hold out no better than the incumbent constants (the first group of this section).
- **The decoder's posterior is not a probability.** A19 measured it four to five times too sharp (Platt α 0.27 for
  alive, 0.19 for the team), while each source alone is near calibrated, and the IoU ladder's loss is mostly edges on
  otherwise right possessions (0.0704 of spellF1 on the mean of the two truths). Neither moving the edges by the
  posterior (R025, R025b) nor weighting P07 apart in the gaps (R026) recovered any of it, so that loss stays (T50).
- **The touch rule sits on edges.** The decoder (6 / 32 / 35 / 25 since R030) and P07's weight (0.3) are inside their
  grid, as R013's were, but the touch rule is on the lower edge in radius (1.5 m), relative speed (6 m/s) and minimum episode (1 frame), the last
  being the smallest there is; R008 found 1.0 m and 3 m/s worse in its context, and neither was run again. P05's
  weight sits on the upper edge of its grid, in R009 and in R030's Stage D. By the protocol none was extended after the
  results, and R030 ran R013's grids as they were.
- **The tighter touch rule transfers less well, by an amount nobody can measure.** On BAR-ATH it finds control
  episodes on 36.9% of the ball-visible graded frames, against 43.6% for R008's rule; on the benchmark the two cover
  43.4% and 47.6% of the alive ball-visible frames, so the 1.5 m radius loses more of its episodes on our match (a ratio
  of 0.85 to the benchmark's, against 0.92). The two figures are not quite the same statistic (ours counts the dead
  frames with the ball visible too), and what it costs in spellF1 is not measured (R013).
- **Some measured values are in sample.** The touch confidence (0.93), the fixed rules' values and the selection of the
  configs use the seven matches the grids are scored on. They were never tuned towards a score, but they are not out
  of sample, and the leave-one-match-out check does not remove that. The trees are different: each benchmark match is
  scored by the model fitted without it. Seven matches, five from one club's home games, are not seven independent
  draws.
- **DFL's switch timing is not ours.** The back-dating, the hold and the `D` onset follow DFL's flag, which turns at
  the new team's control and a median 0.16 s after the ball is first seen beyond a line. Our GT's timing may be
  inspected on the public window (P02) and cannot be fitted to it (RULES.md, 23 Oct 2026), so the one shift made is the
  pipeline's own lag on DFL; the only way to learn our provider's conventions is other games (P16). R023 measured
  their grain from Metrica's sample events, which we withdrew on 5 Nov 2026 for want of a licence; their switch timing
  is not measured, and no licensed source of our provider's games is known to us.
- **The restart team is not modelled beyond the hold.** After a stoppage the hold weakly favours the team that
  controlled the ball before it, until the first new episode names the team, so the team after a stoppage is not decided
  by who restarts. A label-free restart source exists and was measured (R014, R014b), but it did not clear the
  benchmark's bar. Which restart it is cannot come from a person's reading of this match (below).
- **No profile can value the gap features (T38).** The trees' features about the ball's last and next position
  around a gap are validated on gaps that ignore the play, on `tracking` (transplanted) and on `generic` (a Poisson
  process) alike, so what they are worth on our match is unmeasured.

**Of P07 and the shift** (R011, R012, A11; the figures are those entries', on the retired profile and with the models
of their day, unless a bullet cites R030):

- **The shape model learned from other teams' matches, and the benchmark's gain is not a forecast for ours.** It was
  fitted on the seven DFL matches (two competitions, five of them Fortuna Düsseldorf home games) with the ball hidden,
  and each benchmark match is scored by the model that never saw it. BAR-ATH is a LaLiga match with neither team among
  them, the flag the model learned from is DFL's, which is fine-grained (437 possessions a match, A06), and DFL tracks
  the ball out of play. The gain, +0.0942 raw pooled spellF1 over P04 + P05 on `tracking` with 7 wins in 7, is a gain
  over a config that already carries P05's overstated gain (T38, T39). On `generic` the config without the shape source
  loses 0.1037 on the mean of the two truths, with 0 wins in 7 under both (R030, Stage D). The seven matches are not seven independent
  draws, and p 0.0156 is the smallest an exact test over seven can give.
- **The transfer gate is one proxy, and not an accuracy.** The one look at our match that needs no truth was declared
  before any config ran: among the frames inside kept controlled-touch episodes, the share where the shape source names
  the episode's team. For R011's model it was 0.8251 on our match (41,052 frames) against 0.7486, the lowest
  of the seven held-out matches, and 0.8290 pooled over them. For the model retrained in R030 the gate is 0.7426
  (0.8334 pooled), and it is 0.8199 on our match (41,052 frames): a pass. The episode's team is itself right on about 92% of its frames, so the figure
  says the shape agrees with the touches about as often as on the matches it never trained on, and not that its team
  is right that often in a gap.
- **The possession count moves the other way on our match.** On the benchmark P07 raises the predicted possessions per
  graded minute from 2.74 to 3.17 (+16%). On BAR-ATH it lowers them, from 4.2491 to 4.1452 (-2.4%), and the 4.15 is
  far above the benchmark's 3.17 for the same config. We report it and do not explain it; what our truth's count is
  remains unknown (section 3).
- **The public window fell by about one possession.** spellF1 is 0.4000 under P07 against 0.4686 before (20
  possessions against 18, one possession moving it about 0.06), while macroF1 rose from 0.6996 to 0.7151. It is a
  correctness check (T01), it chose nothing, and it is the one place where the chosen config looks worse than the one
  before it.
- **The shift rests on 20 transitions.** The public truth has 20 changes (12 of team, 4 into `D`, 4 out of it), in the
  liveliest 5 minutes of the match (A06), and the pipeline's own latency is matched at 4, 1 and 1 of them. So g = 0
  means no evidence, not no gap: the rule cannot tell a convention from chance at that n, and it is right not to move
  an edge on it. Against our truth the pipeline is a median 22 frames late at a team flip (4 matched changes), where
  against DFL's it is 1 early, and A11's `D` edges look 2.5 to 3 s longer than DFL's flag would draw them. That is a
  convention of our truth, chance or our tracker's coverage, and the data do not separate them. More transitions need
  our provider's conventions on other games (P16); a person watching the 20 can understand them and can set no constant,
  and looks at the truth's changes, never our output.
- **The restart type is not available.** Which stoppage each of the 4 is (a throw-in, a goal kick, a corner, a free
  kick, a foul) and whether `D` begins at the line, the whistle or when play stops cannot be read by code here: the
  Smart Tagging intervals find none of the 4 starts and 1 of the 4 ends, and no LLM may see the video. A person may
  watch the video to understand them, which is understanding what counts as possession and is allowed (A11 gives the
  command and what to look at, `pipeline/benchmark/conventions.py public`). A restart
  type written down per stoppage and fed to the pipeline would be a hand label (RULES.md, T27), so any such channel
  must take its types from code, and none is built.
- **The shift is adopted by its rule, not shown to help.** It is 1 to 2 frames at 439 boundaries, its benchmark gains
  were +0.0000 to +0.0009 pooled spellF1 (R012, on `tracking` and `context`, raw and coarse truth) with every interval
  including 0, and +0.000029 on the mean of the two truths on `generic` (R030; 3 wins, 4 losses), it is one whole number per type of edge (not a function of the ball's visibility or of the stoppage),
  and the team shift depends on the truth: +1 under the raw truth and 0 under the coarse one (T33). Dropping it
  returns the file without it.

**Of the P06 decoder's first choice** (R006, R007, A10; it did not clear its bar on `nearest_player` and
`ball_presence`, whose constants came from the public window, A01, and may not be used in a submission config since the
23 Oct rule; R010's retune replaced it; kept as the record of why):

- **`D` came only from ball presence until P05.** The ball is missing on 40.7% of the graded frames in long gaps, and
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

Expected sha256 of `main.csv` (R030): `2dd51320b1d32f8be6d06e63f833e6d8e196ddcaad294b606db3914c351c1660` (the run
prints it). R030 wrote it twice, each from a rebuilt ATD cache, and a clean clone of `c8d596e` in a fresh Python 3.11.9
environment from `requirements-pipeline.txt` gave the same file. So did a clean clone run with the BLAS thread
variables unset, and two runs on Linux (WSL 2, Ubuntu 24.04, LF configs, the ATD parsed again). All of that was on one
laptop, so a second processor is still untried (T21).

The files before it, for the record: R029's was `762f9073e0d0c56bf8fd4e0b801b80d79defe87246b4f0a8c201d9b5337cdf6e`
(written twice, never regenerated from a fresh clone), R024's
`e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170` (`r024_both.toml`) and R017's
`7ba575c0982aacb9765b39c406e4ff5c9ddb82ca37cfa2b7f101417431fc3d7c` (`r017_frame.toml`). The record configs read the
two model files under `pipeline/models/`, and R030 retrained both (`b30e0f2`). So they wrote those files with the
models of their day and do not write them now: regenerating an old file needs the model files from before that commit.
A17 and A18 checked R017's file while it was current. A17
checked it from a fresh clone of the pushed repository (`5f0145c`), a new venv from the requirements files and a data
folder with no caches: the same sha256 on two runs at one BLAS thread and on a third at the default threads, and both
trained models and the benchmark's 0.5430 / 0.5593 (on `tracking` v3) came back exactly. The first run parses the
tracking into `data/cache/` once
(7.4 s in A17). Every run then computes the play features and the shape features and walks the trees: 13 s with the cache
warm and 20 s cold in A17, on an AMD Ryzen 5 5500U (6 cores), 15.3 GB RAM, Windows 11, CPU only. A18 ran the same on
Linux (WSL 2, Ubuntu 24.04, same laptop) with LF configs: the same sha256 on two runs, and `selftest.py --data` passed. Another
processor (an Intel CPU) has not been tried yet (T20, T21).

The two trained models that `main.toml` reads, `dead_ball_hgb.json` and `shape_possession_hgb.json`, are committed as
JSON under `pipeline/models/` and read with numpy only, so the CSV needs no scikit-learn. Two commands in
`pipeline/benchmark/` retrain one and check that it is the same artefact: `train_dead_ball.py --model hgb --verify`
and `train_shape.py --verify`, on the default profile, `generic` (R030 trained both with `--profile generic --seeds 3`).
They are benchmark commands and sit with them below.

The validation benchmark of section 5 is reproduced separately and is not needed for the CSV (the
[benchmark README](../pipeline/benchmark/README.md) has the rest; reports go to `data/runs/benchmark/`). On `generic`
it needs the DFL files only, and nothing of our match is read.

R030, stage by stage. `FILE` is the JSON that `touch_stats.py` writes. Each `p27_choose.py` call ends with a `NEXT`
line that holds the arguments of the next `p27_grid.py` call. The grids ran in shares, one heavy process at a time;
docs/results.md, R030, has the runs as they were made.

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt          # the pipeline's pins + lxml + scikit-learn
.venv/bin/python pipeline/benchmark/sources.py --verify               # the 21 DFL files, md5 (drop --verify to fetch)
.venv/bin/python pipeline/benchmark/p27_profile_check.py              # the profile against its two declared shares
.venv/bin/python pipeline/benchmark/train_dead_ball.py --model hgb --profile generic --seeds 3   # Stage 0: P05's
                                                                       # model; add --verify to retrain and compare
.venv/bin/python pipeline/benchmark/train_shape.py --profile generic --seeds 3      # P07's model; --verify likewise
.venv/bin/python pipeline/benchmark/train_shape.py --report                         # its held-out scores, its gate
.venv/bin/python pipeline/benchmark/touch_stats.py --grid r013 --profile generic --seeds 3 --out FILE   # Stage 1
.venv/bin/python pipeline/benchmark/p27_grid.py ref --confidences FILE              # the incumbent, p27_ref
.venv/bin/python pipeline/benchmark/p27_grid.py a --confidences FILE                # Stage A's 192 configs
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/p27/p27_ref.toml \
    pipeline/benchmark/configs/p27/p27a_*.toml --profile generic --seeds 3 --truth both       # (in shares)
.venv/bin/python pipeline/benchmark/p27_choose.py a                                 # NEXT --touch 1.5,6.0,1,0.6
.venv/bin/python pipeline/benchmark/p27_grid.py c --confidences FILE --touch 1.5,6.0,1,0.6    # Stage C's 144 configs
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/p27/p27_ref.toml \
    pipeline/benchmark/configs/p27/p27c_*.toml --profile generic --seeds 3 --truth both       # (in shares)
.venv/bin/python pipeline/benchmark/p27_choose.py c                                 # NEXT --decoder 6,35 --weight 0.3
.venv/bin/python pipeline/benchmark/p27_grid.py d --confidences FILE --touch 1.5,6.0,1,0.6 --decoder 6,35 --weight 0.3
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/p27/p27d_*.toml \
    --profile generic --seeds 3 --truth both                                        # Stage D; p27d_base comes first
.venv/bin/python pipeline/benchmark/p27_choose.py d                                 # no group changes
.venv/bin/python pipeline/benchmark/p27_grid.py final --confidences FILE --touch 1.5,6.0,1,0.6 --decoder 6,35 \
    --weight 0.3 --name p27_preshift                                                # Stage C's choice, no shift
.venv/bin/python pipeline/benchmark/conventions.py latency pipeline/benchmark/configs/p27/p27_preshift.toml dfl \
    --profile generic --seeds 3 --truth raw                                         # Stage E: the lag
.venv/bin/python pipeline/benchmark/p27_shift.py p27_preshift                       # NEXT --shift 1,-2,0
.venv/bin/python pipeline/benchmark/p27_grid.py final --confidences FILE --touch 1.5,6.0,1,0.6 --decoder 6,35 \
    --weight 0.3 --shift 1,-2,0                                                     # p27_final: main.toml's constants
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/p27/p27_preshift.toml \
    pipeline/benchmark/configs/p27/p27_final.toml --profile generic --seeds 3 --truth both
.venv/bin/python pipeline/benchmark/p27_choose.py guard p27_final p27_preshift      # the shift's guard
.venv/bin/python pipeline/benchmark/p27_choose.py bar p27_final                     # the bar
```

The record of R002 to R029 on the retired profiles follows, as the commands were run. They read our match's
calibration cache, so they need `data/`, and they are kept for the record only: nothing is trained or chosen on those
profiles again. With today's code the default profile is `generic`, so a line below that names no profile needs
`--profile tracking`; `conventions.py shift` now reads the `generic` latency; and the scores of that day also need the
model files from before `b30e0f2`.

```bash
.venv/bin/python pipeline/benchmark/fidelity.py --profile tracking --seeds 3     # the retired default's gates, warnings
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml \
    pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3    # R004c; again with --truth coarse
.venv/bin/python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml \
    pipeline/configs/p06_hmm.toml --profile tracking --seeds 3     # P06's candidate (R007); --truth coarse too,
                                                                    # with nearest_w125.toml first: its bar there
.venv/bin/python pipeline/benchmark/headroom.py pipeline/configs/p06_hmm.toml \
    --profile tracking --seeds 3                                    # A10; --truth coarse too
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/p45_hmm.toml pipeline/benchmark/configs/p07/*.toml \
    --profile tracking --seeds 3                                    # R011's grid, with R010's main.toml (p45_hmm.toml)
                                                                    # as the reference; --truth coarse, --profile
                                                                    # context too
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/p07_shape.toml \
    pipeline/benchmark/configs/p02/p02_sb.toml --profile tracking --seeds 3     # R012's guard; the other cells too
.venv/bin/python pipeline/benchmark/conventions.py public                       # A11: the public window's 20 changes
.venv/bin/python pipeline/benchmark/conventions.py dfl --profile tracking --seeds 3   # A11, DFL's; --truth coarse too
.venv/bin/python pipeline/benchmark/conventions.py compare --profile tracking   # A11: the gaps, after the two above
.venv/bin/python pipeline/run.py pipeline/configs/p07_shape.toml    # writes evaluation/results/p07_shape.csv, which
                                                                    # `latency ... public` and `shift` read
.venv/bin/python pipeline/benchmark/conventions.py latency pipeline/configs/p07_shape.toml dfl   # R012: the lag; also
                                                                                # `public`, and with --truth coarse
.venv/bin/python pipeline/benchmark/conventions.py shift --config pipeline/configs/p07_shape.toml  # R012's rule
```

R013's grids and runs (each process lists `pipeline/configs/main.toml` first, as R012's `main.toml` when they ran; to
rerun them now, use `pipeline/configs/p02_shift.toml` as the reference, which writes R012's file):

```bash
.venv/bin/python pipeline/benchmark/touch_stats.py --grid r013 --profile tracking --seeds 3 \
    --out data/runs/r013/touch_stats_r013.json                                   # Stage A's 48 confidences
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/p02_shift.toml \
    pipeline/benchmark/configs/r013/r013a_*.toml --profile tracking --seeds 3 --truth both   # Stage A (in shares)
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/p02_shift.toml \
    pipeline/benchmark/configs/r013/r013c_*.toml --profile tracking --seeds 3 --truth both   # Stage C (in shares)
.venv/bin/python pipeline/benchmark/choose.py --reference p02_shift <the stems in the declared tie order>   # each stage
.venv/bin/python pipeline/benchmark/conventions.py latency pipeline/benchmark/configs/r013/r013_preshift.toml dfl
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/p02_shift.toml pipeline/configs/r013_retune.toml \
    pipeline/benchmark/configs/r013/r013_preshift.toml --profile context --seeds 3 --truth both   # the guard's cells
```

R024's selection and runs, and A19 (R024's processes list `pipeline/configs/main.toml` first, as R017's `main.toml` when
they ran; to rerun them now, use `pipeline/configs/r017_frame.toml` as the reference, which writes R017's file, and pass
its stem to `--reference`; Stage A′ reads R013's Stage A reports, written by the block above):

```bash
.venv/bin/python pipeline/benchmark/choose.py --reference main --objective mean <R013's 192 Stage A stems in the declared order>
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/r024/r024c_*.toml \
    --profile tracking --seeds 3 --truth both --drop-match-only        # Stage C′ (in shares, one per switch_team)
.venv/bin/python pipeline/benchmark/choose.py --reference main --objective mean <the 144 stems in the declared order>
.venv/bin/python pipeline/benchmark/conventions.py latency pipeline/benchmark/configs/r024/r024_preshift.toml dfl \
    --profile tracking --seeds 3 --truth raw                            # the lag, by R012's rule
.venv/bin/python pipeline/benchmark/run.py pipeline/configs/main.toml \
    pipeline/benchmark/configs/r024/r024c_t04_mp035_w200.toml pipeline/benchmark/configs/r024/r024_preshift.toml \
    --profile tracking --seeds 3 --truth both --drop-match-only        # the guard; --profile context too
.venv/bin/python pipeline/benchmark/diagnose.py pipeline/configs/main.toml --profile tracking --seeds 3 \
    --drop-match-only                                                   # A19, on R024's main.toml
```

`shift` reads the CSV of the config it is given and that config's latency files, and refuses a latency file made for
another content of the config (or written before the files carried a hash of it), naming the `latency` command to run
again. R012's own files, written when `main.toml` was R011's, are refused for `main.toml` and are written again for
`p07_shape.toml` by the commands above.
