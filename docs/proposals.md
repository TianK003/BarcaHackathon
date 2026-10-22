# Proposals

What we could try, ordered by how much `spellF1` we expect per day of work. Pick one, put your name in the status,
and log what happened in [results.md](results.md) — a dropped idea with a reason is as useful as a win.

**Status:** `idea` → `doing (name)` → `done → R0xx` or `dropped → R0xx (why)`.
**Effort:** S ≤ 1 day · M 2–4 days · L ≥ 1 week. **Impact** is our guess for the full match; revise it as results
come in.

## What the measurements say

Measured on 2026-10-21; details in [results.md](results.md) A01–A03. These five facts drive the ordering below.

1. **The ball goes missing in long gaps.** 40.7% of graded frames have no ball, in only 150 gaps with a median of
   10.7 s (p90 34 s, max 3.7 min). Gaps under 2 s hold 0.1% of the missing frames, so interpolation cannot help —
   possession changes *inside* the gaps.
2. **The public window flatters ball-based methods.** Its ball is missing on 12% of frames; later 5-minute blocks
   reach 80%. It is the best-covered block of the whole match.
3. **The nearest player is not the possessor.** With the ball visible and in play, the nearest player's team is wrong
   on 22% of frames.
4. **A missing ball is not a dead ball.** The ball is visible on 66% of dead-ball frames (placed, held, thrown in), and
   the baseline hands those frames to a team.
5. **Possessions are short and the slack is tiny.** Median 7.6 s; 6 of the 17 public possessions last under 5 s, and
   the six shortest tolerate only 0.3–1.5 s of *total* boundary error at IoU 0.7. Meanwhile the baselines report
   398–1,949 possessions over the match: fragmentation is their main failure.

## Overview

| ID | proposal | attacks | effort | impact | status |
|---|---|---|---|---|---|
| [P01](#p01--a-validation-set-we-can-trust) | Validation set we can trust: other matches, degraded to look like ours | deciding anything at all | M | enabling | doing (Tian) |
| [P02](#p02--learn-the-ground-truths-conventions-from-the-public-window) | Learn the GT's conventions from the public window | boundary timing | S | high | idea |
| [P03](#p03--one-command-deterministic-cached) | One-command, deterministic, cached pipeline | reproducibility | S–M | enabling | done → R001 |
| [P04](#p04--sticky-ownership-from-controlled-touches) | Sticky ownership from controlled touches | facts 3, 5 | S | high | idea |
| [P05](#p05--a-real-dead-ball-detector) | Dead-ball detector | fact 4 | M | high | idea |
| [P06](#p06--global-decoding-per-half-hmm--hidden-semi-markov) | Global decoder (HMM / semi-Markov) per half | fact 5 | M | high | idea |
| [P07](#p07--possession-without-the-ball-team-shape-and-movement) | Possession without the ball: team shape | fact 1 | M | high | idea |
| [P08](#p08--smart-tagging-as-soft-evidence) | Smart Tagging as soft evidence | facts 1, 4 | S | medium | idea |
| [P09](#p09--camera-aim-as-a-coarse-ball-zone-prior) | Camera aim as a coarse ball-zone prior | fact 1 | S | low–medium | idea |
| [P10](#p10--referee-whistle-from-the-audio-track) | Referee whistle from the audio track | fact 4 | S–M | medium (audio confirmed, A04) | idea |
| [P11](#p11--touch-and-pass-events-to-place-boundaries) | Touch and pass events to place boundaries | fact 5 | M–L | medium–high | idea |
| [P12](#p12--metric-aware-post-processing) | Metric-aware post-processing | fact 5 | S | medium | idea |
| [P13](#p13--learn-from-external-labelled-tracking-data) | Pre-train on external labelled tracking data | facts 1, 2 | L (M on top of P01) | medium–high | idea |
| [P14](#p14--recover-the-ball-from-the-video) | Recover the ball from the video | facts 1, 2 | L | medium–high, uncertain | idea |
| [P15](#p15--broadcast-tracking-as-a-second-opinion) | Broadcast tracking as a second opinion | facts 1, 4 | L | medium | idea |

## Suggested plan — 27 days to 17 November

One track per person; swap freely.

| track | week 1 (→ Oct 27) | week 2 (→ Nov 3) | week 3 (→ Nov 10) |
|---|---|---|---|
| Validation | P01 adapter and degraded benchmark, P02 | P12 | re-validate every candidate |
| Pipeline and decoder | P03 | P06 | tune P06 on P01 |
| Ball visible | P04 | P11 | P11 |
| Dead ball | P05 rules | P05 classifier, P08 | P10 |
| Ball missing | P07 | P07, P09 | P13 or P14 spike → go / no-go |

Week 4: feature freeze **Fri 13 Nov**; regenerate the file from a clean clone on a second machine; finish the
[technical document](technical_document.md); submit **Mon 16 Nov** and keep Tuesday as slack.

---

## Foundations

### P01 — A validation set we can trust

**Rules changed on 22 Oct 2026.** The organisers' clarification (upstream `c073f29`, RULES.md and OVERVIEW.md) allows
hand-labelled data from **other matches**, in any amount, and forbids hand labels on the competition match itself —
for validation and calibration too ([T27](traps.md)). The first plan for P01 (label about 15 minutes of BAR-ATH from
the video, after labelling the public window blind to calibrate) is therefore out. The goal is unchanged; the labels
now come from other matches.

**Why.** The public window holds 17 possessions — each worth about 0.06 spellF1 — and it is the easiest 5 minutes of
the match (fact 2). It already picked the wrong baseline. Until we have a better yardstick, every other proposal is
a coin flip.

**What.** A benchmark of other matches whose tracking carries labels of the same kind as our ground truth, degraded
to look like our ATD and run through our own pipeline.
1. **Source: the DFL/IDSSE open data** (Bassek et al., *Sci Data* 12, 195, 2025; CC BY 4.0): seven full Bundesliga
   matches at 25 Hz. Each frame carries `BallStatus` (alive/dead) and `BallPossession` (which team): the same two
   things our GT encodes. Their truth is therefore `D` where `BallStatus` is dead and otherwise the team
   `BallPossession` names, exactly how our GT is built. The loader is `kloppy.sportec.load_open_tracking_data(match_id=...)`
   in our pinned kloppy 3.18.0; it downloads the files, and they go under `data/external/` (gitignored).
   Two traps sit in the loader ([T30](traps.md)):
   - it drops every dead-ball frame unless you pass `only_alive=False`;
   - it reads `BallPossession` as "1 is home, anything else is away", so a frame with no possession value comes out
     as away.
2. **An adapter** turns each match into an `Inputs`-compatible object, plus a `frame,label` truth for it. The object
   carries the same arrays as `atd_cache.Atd` (players, the team of each slot, ball, speeds), its own `graded`
   blocks per half, pitch size and fps. Every evidence source, decoder and refiner then runs on it unchanged.
   A benchmark runner runs one config over every match and reports scores per match and pooled.
3. **Degrade it to look like our ATD**, with the degradation seeded ([T22](traps.md)):
   - Drop the ball in gaps drawn from A01's distribution (median 10.7 s; 75% of the missing frames in gaps of 15 s
     or more), at a rate that climbs through each half as in A02. Our gaps are not random: they cluster in crowded
     and aerial play. Random gaps would flatter ball-free methods (P07), so where the data allows, also build a
     variant that places gaps in crowded play, and report both.
   - Break identities into anonymous, team-labelled tracklets that appear, vanish and swap ([T15](traps.md)).
   - Add positional noise.
   - Before trusting the copy, check that it matches our data on A01's numbers: ball-missing share, gap lengths,
     the nearest player agreeing with the truth on about 78% of in-play ball-visible frames, and the ball visible on
     about 66% of dead-ball frames.
4. **Score with the official metric** (`possession_metrics.evaluate`), per match and half, never stitched across
   halves, with bootstrap intervals. Seven full matches replace 17 possessions.
5. **Check the yardstick.** The benchmark should rank the four baselines in their published full-match order
   (`smooth50` > `smooth125` > `smooth25` > `raw`, EVALUATION.md), the order the public window gets wrong. That is a
   check, not a target. Never tune the degradation towards the published numbers ([T26](traps.md)).

**What it cannot settle.**
- **Conventions.** The DFL flags may flip at different moments from Metrica's corrected tracking. Use the benchmark
  to rank methods and to fit what transfers (switch penalties, duration priors, model weights). Fit timing (P04's
  back-dating, P05's `D` onset, P12's shift) on the public window through P02.
- **Our other assets.** DFL comes without video, audio, homography or Smart Tagging, so P08, P09, P10, P14 and P15
  cannot be benchmarked directly. Either simulate them from the benchmark's truth with the noise measured on our match
  (A03 for P09), or check them on the public window only.

If a convention cannot be read from the DFL flags, hand-labelling other matches is allowed, in any amount. Never
label BAR-ATH. The DFL data is public and outside the competition bundle, so its own licence governs it, not the
bundle's rules. Cite it in the technical document.

### P02 — Learn the ground truth's conventions from the public window

**Why.** The GT is mechanical: `D` where the corrected tracking says the ball is dead, otherwise the team it names.
*When* it flips matters more than anything. Does `D` start when the ball crosses the line, or at the whistle? Does
possession change at the first touch, or once control is established? With 0.3–1.5 s of slack on short possessions
(fact 5), a systematic half-second offset is fatal.

**What.** Around each of the 20 public transitions (12 team→team, 4 into `D`, 4 out of `D`), measure locally:
- whether the ball is visible, and its position and speed;
- the nearest player of each team and their distance;
- where the ball is relative to the lines, and the time from the ball leaving the pitch to `D`;
- the restart type.

Watch each transition on video. Write the conventions into [traps.md](traps.md) and the technical document, using
aggregates and frame numbers only.

With no hand labels allowed on this match, this is the only place we can learn *this* GT's timing: P01's benchmark
follows the DFL's conventions, not Metrica's. Watching the video and measuring around the published transitions is
fine. Writing labels for frames of this match is not ([T27](traps.md)).

### P03 — One command, deterministic, cached

**Status: done → [R001](results.md).** The pipeline lives in [`pipeline/`](../pipeline/README.md); every item below is in
place, and `python pipeline/selftest.py --data` proves the four baselines come out of it byte for byte.

**Why.** The deliverable is *the same file* from one command, and an experiment we cannot re-run cannot be compared.

**What.**
- One entry point that goes from `data/` to `evaluation/results/<name>.csv` and runs the validator.
- Stream the raw ATD once and cache per-frame arrays (ball, players, teams, speeds) under `data/cache/`. That folder
  is gitignored; put caches nowhere else.
- One config per experiment, and fixed seeds.
- Write CSVs with `\n` line endings ([T21](traps.md)).
- Log the commit, config and scores in [results.md](results.md).

## Core: fix what the baseline gets wrong

### P04 — Sticky ownership from controlled touches

**Why.** Fact 3, and the definition itself: possession changes only when the opponent gains *controlled* possession,
and loose or contested balls stay with the last team. Nearest-player-per-frame flips on every pass in flight and
every duel.

**What.**
- Per frame with the ball: the nearest home and away distances in metres, the ball's speed, and the ball–player
  relative velocity.
- A team *controls* the ball when one of its players stays within r of it at low relative speed for ≥ k frames, or
  receives it and plays on.
- Keep the current team until the other team controls. Then **back-date** the change to the start of that control
  episode, so that confirmation does not turn into latency ([T03](traps.md)).
- No ball → keep the current state.
- Calibrate r and k with P02/P01, not with the public score.

### P05 — A real dead-ball detector

**Why.** Fact 4. Every stoppage we miss merges two possessions, and every spurious one splits a possession.

**What.** Rules first, then a classifier on the same features. Train it on P01's matches (their `BallStatus`), never
on labels of this match ([T27](traps.md)). The features:
- the ball beyond the touch or goal line, or vanishing right at it;
- the ball still at a restart spot — corner arc, six-yard box, touchline, centre spot;
- a collective slowdown — median player speed from the ATD speed channel;
- set-piece shapes — kick-off (everyone in their own half), corner (crowd in the box), free-kick wall;
- `SET PIECES` and `GOALS` tags as soft evidence (P08).

Each restart also tells us which team gets the ball next:

| restart | next team |
|---|---|
| corner | the attackers |
| goal kick | the defenders |
| kick-off | the side that conceded, or the half's kicker |
| throw-in | the team of the player at the touchline (Smart Tagging leaves throw-ins `N/A`) |

The graded stretch before the second-half kick-off (2,934 frames) must come out `D`, never `X` ([T02](traps.md)).
Watch `alive_dead_f1` and the `D` row in the scorer's detail block.

### P06 — Global decoding per half: HMM / hidden semi-Markov

**Why.** Majority voting is local. Possessions are intervals and spellF1 pays for every spurious split; a decoder
weighs the evidence against the cost of each change, over a whole half at once.

**What.**
- States `H`, `A` and `D` (optionally with `D` split by which side restarts).
- Per-frame log-likelihoods from calibrated classifiers fed by P04, P05, P07, P08 and P09.
- A switch penalty λ, plus semi-Markov duration priors with minimum durations.
- Viterbi per half via `games.half_video_blocks` — never across halftime.
- The posterior also shows where an `X` is free, namely inside a possession.
- Tune λ and the priors on P01. Sanity-check possession counts against football sense, but never tune to them.

This is the backbone that everything else plugs into.

### P07 — Possession without the ball: team shape and movement

**Why.** Fact 1: four frames in ten have no ball, in gaps long enough to hold several possessions.

**What.**
- Per-frame, permutation-invariant features of all tracked players: team centroids, length, width, hull area,
  defensive-line heights, mean velocity toward the opponent's goal, players per third, and pressure near the ball
  zone. Add how each of these changes over ±1–2 s.
- Derive each team's attacking direction per half from the data.
- Train H-vs-A on ball-visible frames where P04 is confident (pseudo-labels, tens of thousands of frames), on P01's
  matches with the ball masked (P13), and on the public GT, predicted out of fold ([T27](traps.md)).
- Measure on P01's **simulated ball gaps**, where the truth is known, and check against the public window's
  ball-missing frames (few: 6.5% of its in-play frames). Real gaps are harder than simulated ones: crowded, aerial,
  and late in halves.
- Feed the probabilities into P06.

### P08 — Smart Tagging as soft evidence

**Why.** There are 638 automatic tags, each naming a team: free evidence exactly where the ball is missing.

**What.** Turn each tag into a per-frame vote:

| tag codes | vote |
|---|---|
| `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `ATTACKING TRANSITION`, `LONG BALL` | the tagged team has the ball |
| `DEFENDING IN … THIRD`, `DEFENSIVE TRANSITION` | the other team has the ball |
| `SET PIECES` | a stoppage ending in a restart by the tagged team |
| `GOALS` | a stoppage until the kick-off |

Measure each code's agreement with the public GT first, then feed the votes to P06 as features. P01's matches have no
Smart Tagging, so the public window is the only place to measure this source. Never copy
intervals across as labels. Map `Barcelona` → H, `Atletic Club` → A and `N/A` → no vote. Tag windows are seconds wide,
and phases are not possessions ([T17](traps.md)).

## Cheap signals worth checking

### P09 — Camera aim as a coarse ball-zone prior

**Why.** The tactical camera pans to follow play on every frame, including the long ball gaps. Measured in A03: the
pitch point under the image centre tracks the ball's along-pitch position with corr 0.84, once the camera is allowed
its ~1.4 s (35-frame) lag. The median error is 12 m, and across the pitch it says nothing (corr 0.07).

**What.** Use these as features:
- the aim's x at t + 35 frames (we process offline, so looking ahead is fine);
- the aim's velocity, since counter-attacks swing the camera;
- the zoom (homography scale) around set pieces.

This gives a zone, not a position: feed it to P07 and P06, never to nearest-player logic.

### P10 — Referee whistle from the audio track

**Why.** Fouls and offsides stop play without the ball leaving the pitch, and the tracking only sees players slow
down a second later. A whistle marks a sharp `D` onset, and it also starts free kicks and kick-offs.

**What.**
1. ~~Check whether the tactical mp4 has an audio stream at all.~~ It does: AAC stereo 48 kHz, starting at 0.000 s
   like the video ([A04](results.md)).
2. Band-pass around 2–4 kHz and detect sustained tonal peaks.
3. Map audio time to video frames: frame = `round(t * 25)`, with no start offset.
4. Feed the whistle probability to P05 and P06.

The broadcast audio has whistles too, but it is not aligned (P15).

## Precision

### P11 — Touch and pass events to place boundaries

**Why.** One second of lag takes spellF1 from 1.0 to 0.65 (EVALUATION.md), and possession changes happen at touches.

**What.**
- Find touch candidates from ball kinematics (changes of direction and speed) and attribute each to the nearest
  player within r.
- Chain the touches into passes, receptions, interceptions and deflections.
- Apply CHALLENGE.md literally. A touch without control, a deflection, or an aerial duel without follow-up control
  changes nothing; a controlled reception or a completed pass does.
- Snap P06's boundaries to the matching touch, and check that `lat_f` on the board sits near 0.

### P12 — Metric-aware post-processing

**Why.** Cheap last-mile gains on the ranking metric.

**What.**
- Drop possession blips shorter than m frames, and `D` blips inside possessions. Validate m: the public GT has
  genuine 0.76 s and 1.32 s possessions.
- Shift transitions by the measured mean latency.
- Write explicit `D`, never `X`, before kick-offs and once the ball is out ([T02](traps.md)).
- Use `X` only inside possessions where the label is genuinely uncertain. It is free on spellF1 and only touches the
  macroF1 tie-break.

## Bigger bets

### P13 — Learn from external labelled tracking data

**Why.** The 7,500 publicly labelled frames of this match can train almost nothing, and since 22 Oct we may not add
hand labels of our own on it ([T27](traps.md)). Public datasets of other matches carry our kind of labels, in any
amount. The benchmark half of this idea moved to [P01](#p01--a-validation-set-we-can-trust); what stays here is
training on them.

**What.** Three sources, all with kloppy loaders in our pinned 3.18.0:
- **DFL/IDSSE open data** (Bassek et al., *Sci Data* 12, 195, 2025): 7 Bundesliga matches. Their 25 Hz positional
  data carries per-frame `BallStatus` (alive/dead) and `BallPossession` — the same two things our GT encodes. Loader:
  `kloppy.sportec.load_open_tracking_data(match_id=..., only_alive=False)` ([T30](traps.md)). P01 builds the
  adapter and the degradation for it.
- **Metrica sample games 1–3** (`kloppy.metrica.load_open_data`; game 3 is in our FIFA/EPTS format), with event data
  to derive possession from.
- **SkillCorner open data** (`kloppy.skillcorner.load_open_data`): 10 fps broadcast tracking. kloppy derives its
  alive/dead flag from the possession field, so it is weaker.

Degrade them the way P01 does, then pre-train P07 or a sequence model on them: a per-frame set encoder over the
players plus the ball with a missing mask → TCN/BiGRU → per-frame logits → P06. Calibrate it afterwards on the public
window, predicted out of fold. That is the only labelled data of this match we have.

Hold out P01's benchmark matches from training, or train and score across folds of matches, so that the benchmark
still measures something.

**Rules.** Conventions differ between providers, so use external data to rank methods and pre-train, then calibrate
on the public window. Check every licence, keep the sources public (the organisers must be able to access them), and
explain them in the technical document. **Never use anything about this fixture**: no event feeds, stats or match
reports ([T26](traps.md)).

### P14 — Recover the ball from the video

**Why.** The tracker's ball gaps are long and pile up late in each half (A02); a dedicated detector may see what the
tracker lost.

**What.**
- Run a pretrained soccer-ball detector only on the graded ATD-gap frames (~65k), searching around the zone from P09
  and P07.
- Project the detections to the pitch with the homography (`load_homography.unproject`, valid on 100% of tactical
  frames), and clean them with a kinematic filter.
- An airborne ball projects to the wrong ground point, so give it low confidence.
- Fine-tune only on frames of other matches. Drawing ball boxes on this match's frames is hand-labelling it
  ([T27](traps.md)).
- Decode frame-exactly ([T23](traps.md)), budget the GPU time, and keep the video on our own machines unless the T&C
  clearly allow otherwise.

### P15 — Broadcast tracking as a second opinion

**Why.** The broadcast has its own ball track (50.5% of its frames) from a different, often zoomed, angle. And TV cuts
to replays and close-ups during stoppages, so its 122 invalid-homography stretches are themselves a dead-ball cue.

**What.**
- Align the broadcast to the tactical feed by matching player configurations. Both ATDs use the same normalised
  pitch coordinates, so team point clouds compare directly (an assignment cost per frame pair).
- Solve the alignment per broadcast shot with a monotone time warp; replays match earlier times or nothing.
- Fuse the broadcast ball into P06 and P07, and use "broadcast off the pitch" as stoppage evidence.
