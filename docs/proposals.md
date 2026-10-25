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
   possession changes *inside* the gaps. Our gaps start at the lines and on slow balls, not at random (A07, A08: 28% of
   their onsets are within 2 m of a line, against 9% of the frames at risk); on DFL's open data, which tracks the ball
   out of play, the same hazard alone puts twice as many gap starts near a line (56% against 28%, R005).
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
| [P01](#p01--a-validation-set-we-can-trust) | Validation set we can trust: other matches, degraded to look like ours | deciding anything at all | M | enabling | done → R002–R005 |
| [P02](#p02--learn-the-ground-truths-conventions-from-the-public-window) | Learn the GT's conventions from the public window | boundary timing | S | high | idea |
| [P16](#p16--metricas-sample-games-in-our-ground-truth-providers-conventions) | Metrica's sample games, in our GT provider's conventions: derive H/A/D from their events, study the switch timing | boundary timing, grain | M | medium | idea |
| [P03](#p03--one-command-deterministic-cached) | One-command, deterministic, cached pipeline | reproducibility | S–M | enabling | done → R001 |
| [P04](#p04--sticky-ownership-from-controlled-touches) | Sticky ownership from controlled touches | facts 3, 5 | S | high | idea |
| [P05](#p05--a-real-dead-ball-detector) | Dead-ball detector | fact 4 | M | high | idea |
| [P06](#p06--global-decoding-per-half-hmm--hidden-semi-markov) | Global decoder (HMM / semi-Markov) per half | fact 5 | M | high | done → R006–R007, A10 |
| [P07](#p07--possession-without-the-ball-team-shape-and-movement) | Possession without the ball: team shape | fact 1 | M | high, but only after P05: ~0 to this decoder without it at `margin` 0 (A10) | idea |
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
| Validation | P01 (done 23 Oct, R002–R005), P02 | P12 | re-validate every candidate on P01 |
| Pipeline and decoder | P03 | P06 (done 25 Oct, R006–R007, A10) | P06 is tuned on P01 (R007); then [After P06](#after-p06) |
| Ball visible | P04 | P11 | P11 |
| Dead ball | P05 rules | P05 classifier, P08 | P10 |
| Ball missing | P07 | P07, P09 | P13 or P14 spike → go / no-go |

Week 4: feature freeze **Fri 13 Nov**; regenerate the file from a clean clone on a second machine; finish the
[technical document](technical_document.md); submit **Mon 16 Nov** and keep Tuesday as slack.

### After P06

A recommendation from
[A10](results.md#a10--headroom-what-perfect-evidence-would-be-worth-to-the-chosen-decoder-2026-10-25) (the chosen
decoder run with the truth as evidence) and from the controls of [R007](results.md), written on 25 Oct. Nobody has
started any of it. P06's decoder has not cleared its bar on the benchmark (R007), so `main.toml` is
unchanged, and what follows is about where the next day of work goes. The A10 figures are pooled spellF1 on `tracking`,
raw truth first and coarse second, and are upper bounds: the oracles are exact.

1. **Next: P05, the dead-ball detector.** The case rests on its ceiling under the coarse truth, on its gating P07, on
   `D` being the chosen config's weakest point, and on P04 running alongside it (item 2). It does not rest on a lead
   over P04 under the raw truth, which A10 does not show.
   - Its ceiling is the largest under the coarse truth and level with P04's under the raw one. A perfect dead-ball
     source adds +0.1625 raw and +0.2450 coarse (to 0.3678 and 0.4417); a perfect possessor where the ball is seen,
     P04's ceiling, adds +0.1533 and +0.1684. Raw: +0.0092 apart (mean per match +0.0100 [-0.0063, +0.0263], p 0.2500,
     4 wins and 3 losses), which is no difference. Coarse: +0.0766 apart ([+0.0594, +0.0952], p 0.0156, 7 wins). The
     `tracking` caveat below tilts the comparison towards P05, so even the coarse lead is an upper bound.
   - It gates P07. A perfect possessor inside the gaps adds only +0.0008 and +0.0007 on top of a perfect possessor where
     the ball is seen, and +0.1971 and +0.1810 once `D` is right: `ball_presence` calls every gap longer than about 3.7
     s dead, whatever the team evidence says.
   - It is where the chosen config is weakest and where it responds. Its alive/dead F1 is 0.5676 against the
     yardsticks' 0.6094 and 0.6428, and the controls move spellF1 with it (`margin` 50: +0.0200 alive/dead F1 and
     +0.0071 raw; `seen` off: -0.0254 and -0.0063). It is also the lever for the tiebreaker: macroF1 +0.36 against
     +0.07 to +0.09 for the possessor.
   - The features that need only the ATD and the pitch (the ball beyond a line or at a restart spot, a collective
     slowdown, set-piece shapes) can be benchmarked on the DFL matches, whose `BallStatus` is their `D`. The whistle
     (P10) and the tags (P08) cannot, and are checked on the public window only. The new source replaces
     `ball_presence` as `alive` evidence in the same decoder.
   - Do not expect +0.16. On `tracking` the gaps are blind to the play, so `ball_presence` carries almost no `D`
     information there and A10 measures from almost nothing to perfect; on our match it is better than that (A01).
     Report P05's gain on `tracking` and on `context` (sensitivity only), and rank it under both truths.
2. **P04 alongside it, because it is S.** Its ceiling is level with P05's under the raw truth (above), and the two
   together are worth more than apart (+0.3602 raw, against +0.3158 if they added). A real one will capture a part of
   its ceiling and no more.
3. **P07 only after P05.** Without a source for `D` inside the gaps it is worth nothing to this decoder (+0.0008 raw).
   That holds at `margin` 0, where the gaps' edges are `D` too; if the `margin` sweep (item 5) raises `margin`, the
   edges become P07's again.
4. **After any new source, re-run the minimum-run part of P06's protocol.** With the truth as its evidence the chosen
   config scores 0.7627 raw and 0.9226 coarse and not 1.0: its penalties and its 75-frame minimum, chosen under noisy
   evidence, cap it at DFL's fine grain.
5. **A separate, small item: a sweep of `ball_presence`'s `margin`.** R007's diagnostic (`margin` 50, `D` evidence only
   inside the gaps' interiors) is above the chosen config in all four profile and truth cells (+0.0023 to +0.0071;
   `context` is sensitivity only), and clears the raw half of the bar on `tracking` (+0.0107 over `nearest_w50`, p
   0.0312, 6 wins in 7) but not the coarse half (-0.0236 against `nearest_w125`). It was one look at one value, so it
   is not a candidate. A sweep (a few dozen runs of about 20 s) would need a protocol declared before it runs: the
   values of `margin`, the objective (pooled raw spellF1 on `tracking`, 3 seeds, the 0.005 tie band, then the coarse
   score), both truths side by side, and the rule that no value is adopted unless it also beats `margin` 0 on
   `context` (sensitivity only) under both truths. `tracking` cannot value `ball_presence`: its gaps are blind to the
   play, so its missing frames say almost nothing about `D` there, and a gain on `tracking` alone could be an artefact
   of where the transplanted masks fall. `context` places the gaps by the play, so its direction is the check. P05
   would replace this source, so the sweep is worth running only as a hedge in case P05 does not land before the
   freeze on 13 Nov.

---

## Foundations

### P01 — A validation set we can trust

**Status: done → [R002](results.md) to [R005](results.md).** The benchmark's default profile is `tracking` version 3
([R004c](results.md)). `context` fails two of its gates and is a sensitivity profile only ([R005](results.md)).
The overview is the [benchmark README](../pipeline/benchmark/README.md), and the technical document's section 5 has
the design and the fidelity table.

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
1. **Source: the DFL/IDSSE open data** (Bassek et al., *Sci Data* 12, 195, 2025; CC BY 4.0): seven full matches (two
   Bundesliga, five 2. Bundesliga) at 25 Hz. Each frame carries `BallStatus` (alive/dead) and `BallPossession` (which
   team): the same two things our GT encodes. Their truth is therefore `D` where `BallStatus` is dead and otherwise the
   team `BallPossession` names, exactly how our GT is built (`pipeline/benchmark/truth.py`).
   **Our own download and parser read them, not kloppy's loader** (`sources.py` and `dfl.py`): kloppy 3.18's download
   fails (figshare sits behind a bot challenge), and its loader has two traps ([T30](traps.md)):
   - it drops every dead-ball frame unless you pass `only_alive=False`, which is 41% to 52% of every match;
   - it reads `BallPossession` as "1 is home, anything else is away", so a frame with no possession value comes out
     as away (it never happens in these files, and our parser refuses any code other than 1 or 2).

   `sources.py` fetches the 21 files (2.6 GB) into `data/external/dfl/` (gitignored), checks every size and md5 against
   pinned values and falls back to a Hugging Face mirror of the same data. `dfl.load_match` parses a match once and
   caches it under `data/cache/dfl/`. kloppy stays only as a cross-check of the parser, with 0 disagreements
   ([A05](results.md)).
2. **An adapter** (`pipeline/benchmark/inputs.py`) serves each match as an `Inputs`-compatible object, `MatchInputs`,
   which holds no flag of the truth, and `truth.py` builds a `frame,label` truth for it. The object carries the same
   arrays as `atd_cache.Atd` (players, the team of each slot, ball, speeds), its own `graded` blocks per half, pitch
   size and fps. Every evidence source, decoder and refiner then runs on it unchanged. `run.py` runs one config over
   every match and reports scores per match and pooled.
3. **Degrade it to look like our ATD**, with the degradation seeded ([T22](traps.md)), in profiles that add one part
   of our tracking at a time (`degrade.py`: `clean`, `gaps` and `tracking`, the default, in its version 3 since R004c;
   `context`, with R005, is a sensitivity profile only):
   - **Drop the ball as ours was dropped** (`gaps`, R003): our own missing-ball masks are transplanted onto each half,
     which keeps the gap lengths (A01: median 10.7 s; 75% of the missing frames in gaps of 15 s or more) and the pattern
     of a stretch of our half, but not its clock. The offset is drawn uniformly, so the pattern lags by up to M - n
     frames (M the length of our half, n the benchmark half's: up to 7.6 minutes for a 68,000-frame half), and most of
     the climb through each half (A02) comes shifted. Our gaps are not random, and random ones would flatter ball-free
     methods (P07), so the masks are transplanted and not drawn. The plan's "crowded and aerial" became **lines and slow
     balls** once measured: 28.6% of their onsets are within 2 m of a line against 8.5% of the visible frames, and the
     median ball speed at an onset is 2.2 m/s against 5.1 m/s (A07, A08). The ATD has no ball height, so aerial play
     cannot be measured on ours, and a crowd around the ball raises the hazard too, less than the lines and slow balls
     do (A08). A transplanted mask is still placed without looking at the play: 36.0% of its gaps start within 2 m of a
     line against our 28.0% (R003). The `context` profile (R005) places the gaps with a hazard fitted on our own tracker
     (A08) and **fails two of its gates**: the near-line share badly, a gate that binds only it (0.5586 against our
     0.2797 +-0.05, because DFL tracks the ball out of play, so a quarter of its at-risk rows are near a line against 9%
     of ours), and the missing share narrowly (0.4313 against 0.4110 +-0.02). It stays a
     sensitivity check, the default stays `tracking`, and the four yardsticks keep their order on `context` under both
     truths (R005).
   - **Break identities** into anonymous, team-labelled tracklets that appear, vanish and swap ([T15](traps.md); the
     `tracking` profile, R004 to R004c): our own tracklet lengths in 46 slots, wrong-team slots ([T32](traps.md)), NaN
     speeds in runs, and our low-count frames.
   - **Smooth the positions as the rule says, add no noise**: DFL's players' second difference has an RMS of 9.0 cm
     against 3.3 mm in ours ([T35](traps.md)), so each tracklet piece is smoothed with a 5-frame average by a rule set
     before measuring. Per slot the jitter is then 1.80 cm, 5.5 times ours, and equal to the jitter inside tracklets:
     the average takes DFL's clean 9.0 cm down to a fifth, and no jump filter is needed (R004c). The 6.5 to 8.5 cm
     that versions 1 and 2 showed (8.5 cm, R004b) came from slot handovers: their rule let a slot change player
     between two adjacent rows, and version 3 frees a slot only from the row after its tracklet. No positional bias
     is added: it cannot be measured on ours.
   - **Before trusting the copy**, `fidelity.py` checks it against our match: gates fixed before anything was measured
     (the ball-missing share, the median gap, the slots present, the wrong-team share, the tracklet median and the
     NaN-speed share; eight for `tracking`) and WARN checks on A01's numbers (the nearest player agreeing with the
     truth on about 78% of in-play ball-visible frames, the ball visible on about 66% of dead-ball frames) and on what
     the four yardsticks output on our match (A07: possessions per minute and D shares). A degradation is fitted once
     per version, on label-free statistics of our tracking, then frozen, and never tuned towards a WARN or a score
     ([T37](traps.md)).
4. **Score with the official metric** (`possession_metrics.evaluate`), per match and half, never stitched across
   halves. Uncertainty is a 90% t-interval over the matches' paired deltas against a reference config, the exact
   sign-flip test and wins out of seven, never a bootstrap: seven matches are not a sample to resample, and five of them
   are one club's home games. Seven full matches replace 17 possessions.
5. **Check the yardstick.** The benchmark should rank the four baselines in their published full-match order
   (`smooth50` > `smooth125` > `smooth25` > `raw`, EVALUATION.md), the order the public window gets wrong. That is a
   check, not a target. Never tune the degradation towards the published numbers ([T26](traps.md)). R002 to R004c: the
   check holds in its robust part (`nearest_w0`, the raw baseline, is last under both truths), the rest of the order
   follows the truth, and neither truth gives the published order.

**What it cannot settle.**
- **Conventions.** The DFL flags may flip at different moments from Metrica's corrected tracking. Use the benchmark
  to rank methods and to fit what transfers (switch penalties, duration priors, model weights). Fit timing (P04's
  back-dating, P05's `D` onset, P12's shift) on the public window through P02. A06 measured DFL's grain (437
  possessions a match, median 3.7 s) and found that the public window cannot say how far ours differs, because it is
  livelier than 96% of DFL's windows:
  [A06](results.md#a06--dfls-conventions-against-the-public-window-and-the-literature-2026-10-23).
  So rank the choices that depend on possession length under the raw and the coarse truth (`truth.coarse_codes`,
  `--truth coarse`: a sensitivity check, not a correction) side by side, and treat a disagreement as a warning
  ([T33](traps.md)).
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

### P16 — Metrica's sample games, in our ground truth provider's conventions

**Why.** P01's truth follows DFL's conventions, and P02 learns Metrica's from 20 transitions in one 5-minute window,
which is livelier than 96% of DFL's windows (A06). When the flags flip and how fine our GT's possessions are remain the
two things the benchmark cannot settle ([T33](traps.md)). Metrica's three public sample games come from the provider
of our GT, and ship with event data to derive possession from (P13 lists them). Whether possession derived from events
follows the conventions of our GT is what P16 tests: our GT comes from corrected tracking flags, and labels derived
from events may not follow them. If they do, they give far more transitions than the public window's 20.

**What.** Do it alongside P02.

- Test first whether events reproduce our GT's conventions: compare the derived grain and `D` timing with what the
  public window shows (P02), before trusting them for anything else.
- Derive `H`, `A` and `D` per frame from each game's events and tracking, the way our GT is built: `D` where the ball
  is dead, otherwise the team that has it. The events tell where a possession starts and where play stops.
- Study the switch timing as P02 does around the public transitions, but on many more of them: the lag between the
  touch that wins the ball and the flip, the `D` onset against the ball crossing a line, and the grain (possessions
  per alive minute, median length), set beside DFL's in A06.
- Feed P04's back-dating, P05's `D` onset and P12's shift, and check each against the public window through P02. If
  the grain lands between DFL's raw truth and the coarse one, add a truth of that grain next to the two, and never
  retune a degradation towards it ([T37](traps.md)).

**Rules.** Hand-labelled data from other matches is allowed, in any amount; BAR-ATH is never labelled
([T27](traps.md)). Check each source's licence first, keep it public so that the organisers can access it, and cite it
in the technical document (P13's rule). Use nothing about this fixture ([T26](traps.md)).

### P03 — One command, deterministic, cached

**Status: done → [R001](results.md).** The pipeline lives in [`pipeline/`](../pipeline/README.md); every item below
is in place, and `python pipeline/selftest.py --data` proves the four baselines come out of it byte for byte.

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

**Status: done →
[R006](results.md#r006--the-hmm-decoders-first-grid-on-the-benchmark-the-two-switch-penalties-2026-10-25-tian),
[R007](results.md#r007--the-hmm-decoders-second-grid-the-minimum-run-lengths-and-the-choice-of-its-parameters-2026-10-25-tian)
and
[A10](results.md#a10--headroom-what-perfect-evidence-would-be-worth-to-the-chosen-decoder-2026-10-25).** The `hmm`
decoder is in [`pipeline/decode/hmm.py`](../pipeline/decode/hmm.py), with its posterior (off by default) and the first
`alive` evidence, `ball_presence`. Its parameters were chosen on P01's benchmark and are kept in
[`pipeline/configs/p06_hmm.toml`](../pipeline/configs/p06_hmm.toml), which is not the submission config.

**Result.** Switch penalties of 8 and 32 nats (`switch_team`, `switch_dead`) and minimum runs of 75 frames and 1 give
pooled spellF1 0.2053 under the raw truth and 0.1967 under the coarse one, against `nearest_w50`'s 0.2018 raw and
`nearest_w125`'s 0.2226 coarse (R007, profile `tracking`). **The bar is not met**: +0.0036 over `nearest_w50` raw
(sign-flip p 0.2500, 5 wins in 7) and -0.0259 against `nearest_w125` coarse (0 wins in 7), so `main.toml` is
unchanged. Its `D` is worse than the yardsticks' (alive/dead F1 0.5676 against 0.6094 and 0.6428), and A10 sizes what
better evidence would be worth: 0.3678 raw and 0.4417 coarse with a perfect dead-ball detector, 0.3586 and 0.3651 with
a perfect possessor where the ball is seen, 0.7627 and 0.9226 with both everywhere. The code stays because P04, P05 and
P07 plug into it; what to do next is [After P06](#after-p06).

**Why.** Majority voting is local. Possessions are intervals and spellF1 pays for every spurious split; a decoder
weighs the evidence against the cost of each change, over a whole half at once.

**What.**
- States `H`, `A` and `D` (optionally with `D` split by which side restarts).
- Per-frame log-likelihoods from calibrated classifiers fed by P04, P05, P07, P08 and P09.
- A switch penalty λ, plus semi-Markov duration priors with minimum durations.
- Viterbi per half via `games.half_video_blocks` — never across halftime.
- The posterior also shows where an `X` is free, namely inside a possession.
- Tune λ and the priors on P01, ranking the decoders under the raw and under the coarse truth side by side: DFL's
  possessions are fine-grained and ours may not be ([T33](traps.md); the evidence is in
  [A06](results.md#a06--dfls-conventions-against-the-public-window-and-the-literature-2026-10-23)).
  Where the two truths disagree, treat it as a warning. Sanity-check possession counts against football sense, but
  never tune to them.

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
  ball-missing frames (few: 6.5% of its in-play frames). Real gaps are harder than the `gaps` profile's, which are
  transplanted with their pattern and not their clock, and not placed by the play: ours start near the lines and on
  slow balls, with more players around the ball (A08). The `context` profile (R005) places them that way, fails
  two of its gates and is a sensitivity check only, so the benchmark's default still has gaps that are blind to the
  play.
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

**What.** Three sources (kloppy 3.18.0 has loaders for the last two; the first has our own, below):
- **DFL/IDSSE open data** (Bassek et al., *Sci Data* 12, 195, 2025): 7 matches, two Bundesliga and five 2. Bundesliga.
  Their 25 Hz positional data carries per-frame `BallStatus` (alive/dead) and `BallPossession` — the same two things
  our GT encodes. kloppy's download fails and its loaders drop the dead ball by default ([T30](traps.md)), so P01
  downloads and parses them itself (`pipeline/benchmark/sources.py` and `dfl.load_match`), and kloppy's
  `load_tracking` only cross-checks the parser ([A05](results.md)). P01 builds the adapter and the degradation for it.
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
