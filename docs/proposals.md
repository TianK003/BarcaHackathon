# Proposals

What we could try, ordered by how much `spellF1` we expect per day of work. Pick one, put your name in the status,
and log what happened in [results.md](results.md) — a dropped idea with a reason is as useful as a win.

**Status:** `idea` → `doing (name)` → `done → R0xx`, `partly done → R0xx (what is left)` or `dropped → R0xx (why)`.
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
| [P01](#p01--a-validation-set-we-can-trust) | Validation set we can trust: other matches, degraded to look like ours | deciding anything at all | M | enabling | done → R002–R005; **since 5 Nov its degradation is, by our reading, not permitted** (fitted on our tracking; the organisers' reply: "anything that is not explicitly allowed is not permitted", [rules.md](rules.md)) → P27, done → R030: since 10 Nov the default profile is `generic` v1, which measures nothing of our match |
| [P27](#p27--a-degradation-that-measures-nothing-of-our-match) | A degradation that measures nothing of our match; both models retrained and every choice re-run on it | the rules (question 1); everything P01 decided | M–L | enabling: keeps every learned part and constant on external data only; measured on `generic`: 0.5414 raw and 0.5829 coarse (mean 0.5622), against 0.5584 for the incumbent constants | **done → R030 (10 Nov 2026)**: the profile `generic` v1 is the benchmark's default, both models are retrained on it and every numeric constant is re-chosen on it (hold 0.55 → 0.60, `switch_team` 4 → 6, shape weight 0.2 → 0.3; the rest unchanged); the bar passes; two knife-edge choices, which Tian reviewed on 10 Nov, keeping the declared outcome; not uploaded |
| [P28](#p28--a-possession-back-dated-to-the-teams-first-touch) | A possession back-dated to the team's first touch: a kink of the ball's velocity inside a player's zone, then control | boundary timing at a change of team (a first-time pass or a header before the control) | S–M | none: the gate failed | **dropped at its gate → R031** (10 Nov): the frames it would back-date name the touching team 33.8% and 46.5% of the time against the 70% needed; nothing built into the pipeline |
| [P02](#p02--inspect-the-ground-truths-conventions-on-the-public-window-to-understand-and-never-to-tune) | Inspect the GT's conventions on the public window, to understand and never to tune | boundary timing | S | bounded since the 23 Oct rule: understanding only; so far the pipeline's own lag on DFL, 1 to 2 frames, within noise (R012) | done → A11, R012, A15 (inspection only; a person may watch to understand what counts as possession, never to check our output, 3 Nov) |
| [P16](#p16--metricas-sample-games-in-our-ground-truth-providers-conventions) | Metrica's sample games, in our GT provider's conventions: derive H/A/D from their events, study the switch timing | boundary timing, grain | M | medium | partly done → R023 (the grain: outcome both, so choices rank on the mean of the two truths); the switch timing and the feeds are not done; A22 (4 Nov): the timing part was the one lawful route to a `D`-edge constant; **on hold since 5 Nov: the sample data has no licence, so by our reading we may not use it** ([rules.md](rules.md), question 3); R024's objective, which R023 set, stays (Tian, 9 Nov) |
| [P17](#p17--the-restart-team-channel) | Restart-team channel: who restarts after a stoppage, from the players' arrangement | facts 3, 5 (the hold points at the old team after most stoppages) | S–M | ceiling +0.0198 raw (A12); delivered +0.0040 raw at best, bar not met (R014) | dropped → R014, R014b (the source and the decoder's restart states stay, off by default) |
| [P18](#p18--the-offpitch-profile-dfls-ball-hidden-off-the-pitch-as-ours-is-and-p05-retrained-on-it) | offpitch profile: DFL's ball hidden off the pitch as ours is, and P05 retrained on it | T39 (P05 overstated by the visible out-of-play ball) | M | A13: a perfect dead-ball source adds +0.029 raw on tracking; main.toml on offpitch 0.5509 raw / 0.5296 coarse; hiding the out-of-play ball cost today's P05 0.013 alive/dead F1 (R015) | dropped → R015, R015b (our tracker's off-pitch ball cannot be reproduced by a simple rule; code removed) |
| [P19](#p19--a-learned-team-model-where-the-ball-is-seen) | A learned team model where the ball is seen: P05's recipe applied to the team | fact 3; A13's largest lever (+0.194 raw where the ball is seen) | M (feasibility S) | measured: +0.0160 pooled per-frame accuracy against `main.toml`'s decoded labels, gate +0.020 (R021) | dropped → R021 (feasibility: +0.0160 pooled against +0.020, 3 of 7 matches against 6; no spellF1) |
| [P20](#p20--ball-free-possession-paths-pathcrf-on-our-slots) | Ball-free possession paths: PathCRF (KDD 2026), trained on the same DFL matches | fact 1; A13's gaps (+0.176 raw) | S test, L to adopt | measured: +0.0049 pooled on the gap frames (needed +0.04); PathCRF's own protocol on the clean match reached 0.9172, against P07's 0.8522 on degraded tracking (not like for like) | dropped → R022 (step 0 no-go: +0.0049 under our degradation and the adapter, needed +0.04; code removed, recoverable at `2163406`) |
| [P21](#p21--wrong-team-slots-measure-the-ceiling-then-repair-them) | Wrong-team slots: measure the ceiling, then repair from tracking or jersey colours | T32 | S test, M | low on the benchmark: all of `tracking`'s player degradation costs `main.toml` 0.013 (A14) | dropped → A16 (step 0's ratio 0.85 < 1.5; the tool is kept for P19's transfer report) |
| [P22](#p22--the-shape-model-adapted-to-our-two-teams-without-labels) | The shape model adapted to our two teams without labels | fact 1 | S–M | low: guess +0.00 to +0.02 | (a) dropped → R020 (feasibility gate failed: 0.0538 against 0.0623, ratio 0.863 > 0.75); (b) dropped → R027 (the gate on DFL: a per-half intercept on P07 from the touch rule's labels made its pooled log-loss in the held-out half's gaps +0.0061 worse, needing −0.002; the transfer from touch frames to gaps 0.296 against the teacher's noise 0.107; rung 3 worse too, +0.0838; code removed, recoverable from `1e5a5d2` and the local tag `archive/r027-source`); (c) not tried; none of (a) to (c) allowed since 3 Nov ([rules.md](rules.md)) |
| [P23](#p23--the-hmm-backbone-reviewed-nine-ideas-five-taken) | The HMM backbone reviewed: re-choose under R023's objective (A), decompose the IoU ladder (B), posterior edges (C), calibration (D), a gap weight for P07 (E) | fact 5; the decoder's placement of changes; fact 1 (E) | S each | low: +0.01 to +0.03 for the lot (the advisor's estimate); measured: A +0.0052 on the mean (R024) | done: A adopted → R024; B, D measured → A19; C dropped → R025, R025b; E dropped → R026; F deferred, then run as P22(b) and dropped → R027; G to I not taken |
| [P24](#p24--the-balls-direction-and-the-touch-timing-as-features-of-a-learned-team-source-p19s-one-retry) | The ball's direction against each team's attacking direction, and the time since and until each team's touches, as features of a learned team source where the ball is seen: P19's one retry | fact 3; A13's largest lever (+0.194 raw where the ball is seen) | S–M | medium: P19's first try reached +0.0160 of the +0.020 gate; False9 calls these its largest gain | idea (added 4 Nov, A22) |
| [P25](#p25--a-tempo-check-a-normalised-speed-cue-and-a-sparse-tracking-guard-for-the-dead-ball-model) | A tempo check (is our match slower than the degraded DFL?), normalised copies of the speed features, and a guard where fewer than a handful of slots are present | fact 4; T32; coverage that worsens through each half | S check, M retrain | low–medium: two teams found a fixed speed cut flags half the match dead | idea (added 4 Nov, A22) |
| [P26](#p26--metricas-sample-games-tracking-as-a-second-benchmark-and-training-set) | Metrica's sample games' public tracking (3 games) as a second benchmark and training set, in our provider's grain, degraded as ours | T33, T40, the grain and the timing of our truth | M | medium: the one external data in our provider's conventions; P16's timing part runs on it | on hold (5 Nov): the sample data has no licence ([rules.md](rules.md), question 3) |
| [P03](#p03--one-command-deterministic-cached) | One-command, deterministic, cached pipeline | reproducibility | S–M | enabling | done → R001 |
| [P04](#p04--sticky-ownership-from-controlled-touches) | Sticky ownership from controlled touches | facts 3, 5 | S | high: +0.1421 raw at the chosen decoder (R010) | done → R008, R010; rule re-chosen → R013; hold re-chosen → R024 (0.55); both re-chosen on `generic` → R030 (rule unchanged, hold 0.60) |
| [P05](#p05--a-real-dead-ball-detector) | Dead-ball detector | fact 4 | M | high: +0.1661 raw at the chosen decoder, on a benchmark that overstates it (R010, T38, T39) | done → R009, R010; model retrained on `generic`, weight kept → R030 |
| [P06](#p06--global-decoding-per-half-hmm--hidden-semi-markov) | Global decoder (HMM / semi-Markov) per half | fact 5 | M | high | done → R006–R007, A10; decoder re-chosen → R010, R013; re-chosen on `generic` → R030 (`switch_team` 6) |
| [P07](#p07--possession-without-the-ball-team-shape-and-movement) | Possession without the ball: team shape | fact 1 | M | high: +0.0942 raw at the chosen decoder, on a benchmark whose gain is not a forecast for our match (R011) | done → R011; weight re-chosen → R013; model retrained and weight re-chosen on `generic` → R030 (0.3) |
| [P08](#p08--smart-tagging-as-soft-evidence) | Smart Tagging as soft evidence | facts 1, 4 | S | medium | done → R017 (team votes adopted per frame, weight 0.2; dead votes measured and dropped); **taken out of `main.toml` → R029** (3 Nov rules: `c_team` was measured on our match, [rules.md](rules.md)) |
| [P09](#p09--camera-aim-as-a-coarse-ball-zone-prior) | Camera aim as a coarse ball-zone prior | fact 1 | S | low–medium | dropped → R019 (the simulated camera failed two fidelity gates; code removed); A20: in the gaps the camera only moves with the players |
| [P10](#p10--referee-whistle-from-the-audio-track) | Referee whistle from the audio track | fact 4 | S–M | medium (audio confirmed, A04) | dropped → R018, R018b (whistles near stoppages 1.5× chance, below the declared 2×; code removed) |
| [P11](#p11--touch-and-pass-events-to-place-boundaries) | Touch and pass events to place boundaries | fact 5 | M–L | medium–high | idea |
| [P12](#p12--metric-aware-post-processing) | Metric-aware post-processing | fact 5 | S | medium | done → R012 (shift adopted), R016 (free X measured and dropped); shift re-derived on `generic` → R030 (unchanged) |
| [P13](#p13--learn-from-external-labelled-tracking-data) | Pre-train on external labelled tracking data | facts 1, 2 | L (M on top of P01) | medium–high | idea |
| [P14](#p14--recover-the-ball-from-the-video) | Recover the ball from the video | facts 1, 2 | L | A14: a perfect ball in our gaps is worth +0.10 raw but +0.02 coarse; a real detector perhaps +0.025 to +0.04 raw | not built → A14 (not a ViT: the ball is 6 px, T46; a half-day zero-shot test may run, [After A14](#after-a14)); A20 lowers the prior (the broadcast's tracker has the ball on 28% of the live frames where ours is missing); since 3 Nov no training on our video, only weights trained elsewhere ([rules.md](rules.md)) |
| [P15](#p15--broadcast-tracking-as-a-second-opinion) | Broadcast tracking as a second opinion | facts 1, 4 | S–M per use (the alignment is measured) | A20: its ball fills 30% of the gap frames we call alive, in fragments; its shots off the pitch cover 75% of our `D` | measured → A20; use 1 dropped as R028 (2 Nov): the stand-in on DFL failed its fidelity gates (`tracking`'s gaps are not ours, T38, T53), so the fill is untested, not refuted; uses 2 and 3 ideas |

## Suggested plan — 27 days to 17 November

One track per person; swap freely.

| track | week 1 (→ Oct 27) | week 2 (→ Nov 3) | week 3 (→ Nov 10) |
|---|---|---|---|
| Validation | P01 (done 23 Oct, R002–R005), P02 (done 30 Oct, A11, R012, A15) | P12 (done 30 Oct: R012 shift adopted; free X dropped by R016) | re-validate every candidate on P01; P27, the benchmark's degradation redone and everything re-chosen on it (done 10 Nov, R030) |
| Pipeline and decoder | P03 | P06 (done 25 Oct, R006–R007, A10) | P06 is tuned on P01 (R007); then [After P06](#after-p06); `main.toml` is P04 + P05 (R010), then P07 (R011) and P02's shift (R012); [After P04 and P05](#after-p04-and-p05), then [After P07 and P02](#after-p07-and-p02); its item 1 done 26 Oct (R013: the touch rule and the decoder re-chosen); then [After R013](#after-r013) |
| Ball visible | P04 (done 25 Oct, R008, R010) | P11 | P11 |
| Dead ball | P05 (done 25 Oct, R009, R010) | P08 (done 30 Oct, R017: team votes adopted per frame, dead votes dropped) | P10 (dropped 30 Oct, R018, R018b) |
| Ball missing | P07 (done 25 Oct, R011) | P09 (dropped 31 Oct, R019) | P13 or P14 spike → go / no-go |

Week 4: feature freeze **Fri 13 Nov**; regenerate the file from a clean clone on a second machine; finish the
[technical document](technical_document.md); submit **Mon 16 Nov** and keep Tuesday as slack.

*Where the plan stands on 10 Nov ([R030](results.md)).* The redo the organisers' reply of 5 Nov made necessary (P27)
is done: `main.toml` is the config re-chosen on the `generic` profile, and both models are retrained on it. What is
left before the upload:
- Tian's review of R030's two knife-edge choices: **done 10 Nov, the declared outcome kept** (the hold 0.60; the
  decoder at `switch_team` 6 with the shape weight 0.3; [P27](#p27--a-degradation-that-measures-nothing-of-our-match));
- the checks on our match (pass or fail only) and the clean-clone regeneration of R030's file: their results are in
  R030, and a failed one stops the upload;
- the technical document brought to R030, with the residual risk of the two quoted figures stated
  ([rules.md](rules.md)): **decided 10 Nov, Tian keeps the two figures** (published statistics are public
  information);
- a run of the file on a second processor (T21): so far one laptop, under Windows and Linux;
- the technical document read and trimmed for the PDF;
- the upload and making the repository public, which are Tian's.

The next-step lane after R030, P28 (a possession back-dated to the first touch), was dropped at its gate (R031), and
no other lane was opened. A25 puts the remaining score where A13 did, in who has the ball where the ball is seen
(+0.20 for a perfect source). P24 would address that, and it is not started.

From here on, anything trained or chosen on the benchmark runs on `generic` (`run.py`'s default). The numbers
measured on `gaps`, `tracking` or `context` (R002 to R029) stay as history: nothing is trained or chosen on those
profiles again.

### After P06

A recommendation from
[A10](results.md#a10--headroom-what-perfect-evidence-would-be-worth-to-the-chosen-decoder-2026-10-25) (the chosen
decoder run with the truth as evidence) and from the controls of [R007](results.md), written on 25 Oct. Nobody has
started any of it. P06's decoder has not cleared its bar on the benchmark (R007), so `main.toml` is
unchanged, and what follows is about where the next day of work goes. The A10 figures are pooled spellF1 on `tracking`,
raw truth first and coarse second, and are upper bounds: the oracles are exact. *Later (R008 to R012):* items 1 to 3
were done (P05, P04, P07), item 4 is carried into [After P07 and P02](#after-p07-and-p02), and item 5 is moot, since
P05 replaced `ball_presence` as the `alive` source.

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
     (P10) and the tags (P08) cannot, and are checked on the public window for correctness only. The new source replaces
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

### After P04 and P05

A recommendation from [R010](results.md), written on 25 Oct. `main.toml` was then P04 + P05 (the bar is met under both
truths and the label-free check passes), and the file has not been uploaded. Pooled spellF1 on `tracking`, raw truth
first and coarse second. *Later:* R011 did item 3 and A11 with R012 did part of item 1, and the next step is in
[After P07 and P02](#after-p07-and-p02).

1. **Understand our GT's timing (P02), with a person watching the video around the published transitions, to design
   features and never to derive a constant.** The back-dating, the 0.55 hold and the `D` onset follow DFL's flag, and
   nothing about them was fitted to ours. Where a possession's edges fall decides whether it matches at the higher IoU
   thresholds, so it is worth understanding. *Changed 23 Oct 2026 (`f8f2629`):* the public window is no tuning set, so
   it cannot be where our timing is fitted: what a person sees may shape a feature or a question for other games
   (P16), and any constant comes from other games by a declared rule.
2. **Re-check the decoder's edges under a new protocol.** The chosen `switch_team` 4 and `min_possession` 25 are the
   lowest values tried and raw spellF1 falls from them (R010). The coarse truth wants the opposite end (0.5049 coarse
   and 0.3532 raw at 16 and 100), so the grain of our GT (T33, A06) decides it. Declare the values and the rule before
   running, and read both truths.
3. **P07 for the gaps, now unlocked.** A10 found a perfect possessor inside the gaps worth +0.0008 raw until `D` was
   right and +0.1971 once it was. `D` now has a source, so P07 can be tried, but `tracking` cannot value gap features
   (T38): read it on `context`, and on the public window as a correctness check only.
4. **A restart-team channel (P05's table).** After a stoppage `controlled_touch`'s hold weakly favours the team that
   controlled the ball before it, until the first new episode names the team. A source for the team that restarts
   would speak sooner. R010 does not measure its worth, so it comes after the first three.
5. **Before the upload, decide how much of the gain to believe.** The benchmark's alive/dead F1 of 0.9726 rests on
   DFL's ball being tracked out of play (T39). Our match's predicted `D` share is 36.0% against the benchmark's 44.6%.
   The decision rests on the benchmark and on label-free checks such as that share. The public window's alive agreement
   (86.6% of 7,500 frames) checks correctness and may not weigh in (T01). Neither the benchmark nor the share says how
   the full match scores.

### After P07 and P02

A recommendation from [R011](results.md), [R012](results.md) and
[A11](results.md#a11--our-ground-truths-timing-against-dfls-from-tracking-events-2026-10-25-tian), written on 25 Oct.
`main.toml` is now P04 + P05 + P07 with P02's shift, and the file has not been uploaded. On the benchmark (`tracking`
v3, raw truth, pooled spellF1) P07 took it from 0.4217 to 0.5159 and the shift added +0.0005, inside noise. Neither
figure is a forecast for our match: the shape model never saw our teams, DFL tracks the ball out of play, and the count
of possessions moved the other way on BAR-ATH (R011). *Items 2 and 4 are rewritten for the organisers' 23 Oct rule
(`f8f2629`): the public window checks the pipeline and never tunes it, and a person watches to understand.*
*Later (R013, 26 Oct):* item 1 is done, and the next step is in [After R013](#after-r013).

1. **Next: re-check the decoder's edges with the new evidence ("After P04 and P05", item 2).** The reasons are in the
   results:
   - R011 kept the decoder at R010's values (4, 32, 25, 25) in all 12 configs, so P07's weight was chosen against a
     decoder tuned for the other two sources. The weight's peak is interior (0.2; 0.1 and 0.5 score 0.5057 and 0.5051
     raw against 0.5159), which is what a weight trading off against the switch penalties could look like, and nothing
     in R011 tells the two apart.
   - The chosen `switch_team` 4 and `min_possession` 25 are the lowest values R010 tried, and "After P06" item 4 already
     says to re-run the minimum-run part of the protocol after any new source. P07 is one: at weight 0.2 it can move a
     frame by up to 0.78 nats (3.9 x 0.2, the clip and the weight), on every graded frame.
   - The grain is live again. P07 moved the predicted possessions per graded minute from 2.74 to 3.17 on the benchmark
     (the truth's 4.58 raw and 2.48 coarse) and from 4.25 to 4.15 on BAR-ATH, and the two truths still disagreed on the
     decoder in R010 (T33). Read both.
   - It needs no new data: only benchmark runs, which R011 logged at 25.6 to 27.5 s a config on `tracking` once the
     evidence was memoised, after 143 to 156 s for the first config of each process (the runner's `seconds`). It needs
     a protocol declared before the first run (the values, the objective, the rule), the weight as a second axis only
     if that protocol says so, and nothing from the public window.
   - It is also where the choices made with window-derived values present get re-confirmed without them (the 23 Oct
     rule). In R006 to R009 every candidate config (the yardsticks aside) carried `nearest_player` at 0.78 or
     `ball_presence` at 0.2 / 0.585, both from the public window (A01), so R008's touch constants (radius,
     `max_rel_speed`, `min_frames`, hold) and R009's source and weight were chosen with them present, and R010 re-chose
     only the decoder. `main.toml` has neither source, so this protocol runs without them, and may add those constants
     as axes if it declares so first.
   - *Done (R013, 26 Oct):* the touch rule first (192 configs), then `switch_team`, `min_possession` and P07's weight
     (144 configs), the shift re-derived by R012's rule, all on `main.toml`'s own evidence. The touch rule moved (1.5 m,
     6 m/s, 1 frame, hold 0.60; R008's setting ranked 83rd of 192), `min_possession` went from 25 to 35, `switch_team`
     (4), the weight (0.2) and the shift stayed. Against R012's `main.toml`: +0.0267 raw on `tracking` (p 0.0156, 7 wins
     in 7) and +0.0001 coarse; adopted. R009's source and weight were declared out of scope (the source rests on the
     out-of-fold log-loss, the weight 1.0 on taking the model's probability as it is).
2. **In parallel, and off the machine: a person watching P02's transitions, to understand.** R012's rule found g = 0
   for every type because 1 to 4 matched public changes cannot carry a gap, and A11's `D` edges (2.5 to 3 s longer than
   DFL's flag would draw them) are a lead and not a measurement: our truth's convention, chance and our tracker's
   coverage all fit. A person watching the video to see where `D` begins and ends (the line, the whistle, the stoppage)
   and what kind of restart follows can understand which of them it is (A11, "How a person can extend it"). That
   understanding can shape a feature. It cannot set a number: g is retired (T40), and a constant read off these 20
   changes, or a label for a stoppage, is tuning on the public window or a hand label (T27). It costs hours and not
   compute, so it can run beside item 1; what comes out of it is a note in the docs, in aggregates. *3 Nov 2026
   ([rules.md](rules.md)): still allowed, as understanding "what counts as possession"; the person looks at the
   truth's changes only, never at our output.*
3. **Before any upload, decide how much of the gain to believe ("After P04 and P05", item 5).** It is a decision and
   not a run. R011's transfer gate (0.8251 against 0.7486) is a proxy and not an accuracy, and the possession count
   moved the other way on BAR-ATH. The public window's spellF1 fell from 0.4686 to 0.4000, about one possession, but it
   is a correctness check and may not weigh in on this decision (T01): the decision rests on the benchmark, which is
   still the best evidence we have, with the caveats of T38, T39 and R011, and on the label-free checks. It has to be
   settled before the freeze on 13 Nov.
4. **The restart-team channel, after those.** R010 and R011 do not measure its worth, and A11 gives it no signal: it
   needs the restart type of each stoppage. A type that a person read off this match's video for each stoppage and fed
   to the pipeline would be a hand label (T27), so the types have to come from code (Smart Tagging's set-piece tags,
   P08, which are automatic) or from other games' flags (P16). A person watching to judge whether the channel is worth
   building is design, which is allowed (since 3 Nov 2026 only as understanding what counts as possession at a
   restart, [rules.md](rules.md)). If the stoppages are mostly of a few types with a clear next team (a corner, a
   goal kick), it is a small source for `home` after a stoppage; if not, drop it. *Started 26 Oct as P17 (A12: a
   perfect 2 s restart vote is worth +0.0198 raw and +0.0197 coarse, 7 wins in 7).*

What the evidence does not support is a second round on P07's features, on the shift's three integers or on P07's
weight alone (item 1 may revisit the weight, as a second axis of the decoder's re-check, under a protocol declared
first): R011 chose by its rule and added nothing after its results, and each gain is measured on DFL alone, at a size
the benchmark cannot carry over to our match.

### After R013

A recommendation from [R013](results.md), written on 26 Oct. `main.toml` is now R013's final config (sha256
`0ebdfbec...`, not uploaded): every constant of it comes from a choice whose candidates carried no window-derived value,
except R009's dead-ball source and weight, which R013 declared out of scope.

1. **The grain of our truth is now the question that decides the most (P16).** Every re-tune since R007 has split the
   two truths: R010's decoder and R013's touch rule each gained under the raw truth and lost under the coarse one, and
   R013's final config is +0.0267 raw and only level coarse. Which of the two our ground truth resembles is what would
   say whether R013 helps on our match. The public window cannot answer it (A06), so it has to come from our
   provider's conventions on other games: Metrica's sample games (P16), by a rule declared on them. **Answered by R023
   (2026-10-31): both.** The rate falls between the truths and the median on the raw side of the middle, so choices
   declared from now on rank on the mean of the two pooled spellF1s. Whether to re-choose what the raw truth decided
   alone (R013) is the team's call.
2. **Before the upload, decide how much of the gain to believe** ("After P07 and P02", item 3, unchanged). R013 adds two
   caveats to T38, T39 and R011's: the gain is a raw-truth gain, and the tighter touch rule covers relatively fewer
   ball-visible frames on BAR-ATH than on DFL (36.9% against R008's rule's 43.6%, where the benchmark has 43.4% against
   47.6%). Settle it before the freeze on 13 Nov.
3. **Not a second round on the touch rule.** Its three edges (1.5 m, 6 m/s, 1 frame) point at a smaller zone, but
   R008 found 1.0 m and 3 m/s worse, the minimum episode cannot go below one frame, and the coverage on BAR-ATH already
   argues against tightening it further on DFL's evidence alone. A grid there would be tuning to DFL's fine grain.
4. **In parallel, as before:** a person watching P02's transitions to understand ("After P07 and P02", item 2), and the
   restart-team channel from code (item 4). R013 changes neither. *Later (26 Oct):* the channel is P17, started; it
   closed as dropped (R014, R014b), see [After P17](#after-p17).

### After P17

A recommendation from [R014](results.md) and [R014b](results.md), written on 26 Oct. Neither met its bar, `main.toml`
is unchanged (sha256 `0ebdfbec...`, not uploaded), and the source and the decoder's restart states stay in the code, off
by default. None of the four has been started, and each needs its own protocol declared before its runs.

1. **A shorter vote.** R014's choice sat on its grid's lower edge (25 frames), every longer tail scored lower, and the
   leave-one-match-out picked it in 7 of 7 folds (+0.0040, p 0.0938). The cheapest next test is a declared grid below it,
   for example 5, 10, 15 and 25 frames, with the confidences measured at each tail and the same bar. It is a second look
   at the same benchmark, so the bar must be the same and declared before it runs. Two cautions: the chosen `t025_b0`
   was negative under the coarse truth everywhere (pooled -0.0029, 2 wins in 7; leave-one-match-out, whole rule,
   -0.0039, p 0.0312, 1 in 7; context coarse -0.0042), a raw/coarse disagreement that T33 and CLAUDE.md treat as a
   warning; and this would be the third declared test of P17 on the same seven matches, not the second.
2. **The goal-kick rule's specificity.** In A12 part 2, 134 of 608 matched throw-ins were called goal kicks, because the
   keeper stands in their goal area in most stoppages. A rule that tells the two apart would raise a type's confidence,
   and the goal kick is the weakest (0.72).
3. **The observation's weight as a declared axis in the restart states.** One observation is at most about 2.6 nats,
   and R014b's +0.0003 says that is too weak to matter; its identical labels across `switch_restart` {4, 8, 16, inf}
   were by construction (T44), not evidence. A weight axis w brings `switch_restart` into play only where w * logit(c)
   exceeds it, so the two must be declared together, with `switch_restart` values on both sides of the largest w *
   logit(c), or the next grid repeats the problem. Run this only if item 1 does not settle it.
4. **Smart Tagging's set-piece tags as a type hint (P08).** Left out of P17 on purpose. They are automatic, but P01's
   matches have no Smart Tagging, so there is no benchmark to choose their weight on.

*Added 29 Oct:* [A13](results.md) part 1 sizes where `main.toml`'s remaining spellF1 is on the benchmark, and it points
at the team side: a perfect team where the ball is seen is worth +0.19 raw, and in the gaps +0.18 raw, against +0.029 for
a perfect dead-ball source (`tracking`, 7 wins in 7 each). P18 was started on 29 Oct, for the dead-ball side's transfer
gap (T39), and it was dropped (R015, R015b), see [After P18](#after-p18).

### After P18

Written on 29 Oct from [A13](results.md), [R015](results.md) and [R015b](results.md). `main.toml` is unchanged (sha256
`0ebdfbec...`).

1. **A13: on the benchmark the team side holds the remaining score.** A perfect team where the ball is seen is worth
   +0.194 raw and +0.231 coarse, where it is missing +0.176 and +0.108, everywhere +0.340 and +0.355, against +0.029 and
   +0.039 for a perfect dead-ball source (`tracking` v3; `context` agrees). The decoder's own ceiling is 0.918 raw and
   0.969 coarse.
2. **R015 is a lead, not a measurement,** that the visible out-of-play ball overstates P05 only a little: alive/dead F1
   goes from 0.9731 to 0.9600 with it hidden, and the profile failed a gate. R015b closed the profile.
3. **So the next work goes to who has the ball:** where it is seen (P04's touch rule is chosen; P11, touch and pass
   events) and in the gaps (P07, which never saw our teams). Each needs its own declared protocol before its runs.

### After A14

Written on 29 Oct. Three inputs: a debate on P14 (a video ball detector, a ViT or anything better), held between an
advocate and a critic who each researched their case and then rebutted the other's; [A14](results.md), the measurements
that debate called for; and a survey of the literature for the team side, which produced P19 to P22. `main.toml` is
unchanged (sha256 `0ebdfbec...`).

1. **P14 is not built now, and would not be a ViT.**
   - The ball is about 6 px wide, smaller than one ViT patch (T46). If anything, the candidate is a stride-4 heatmap net
     at full resolution with stacked frames, trained on the ATD's own ball as algorithm-made pseudo-labels. *Since
     3 Nov 2026 that training set is not allowed (training on our video, [rules.md](rules.md)); only weights trained on
     other data may run on our video.*
   - A perfect ball in our gaps is worth +0.10 raw but only +0.02 coarse on the benchmark (A14). A real detector covering
     35 to 50% of the gap frames would deliver perhaps +0.025 to +0.04 raw and +0.005 coarse, below the full match's
     noise.
   - The cost is about 9 person-days. The machine is a CPU-only laptop that crashes under sustained load, and a neural
     network's output would sit in the path that has to regenerate the CSV byte for byte.
   - Both sides declared their rule before A14 ran. The advocate's own rule drops it, and the critic's says no go.
   - **What may still run** is P14's test B: zero-shot, half a day, one person, CPU only. Run it only if someone is free
     after items 2 and 3. A full build follows only if test B passes *and* P16 finds that our truth is closer to DFL's raw
     grain than to the coarse one.
2. **P19's feasibility test: done, no-go (R021).** The team where the ball is seen holds the largest headroom (+0.194
   raw, A13), but a learned team model beat `main.toml`'s decoded labels per frame by only +0.0160 pooled against the
   +0.020 declared, and in 3 of 7 matches against the 6 needed (all seven positive, mostly on hold frames). P19 is
   dropped and its code removed (recoverable at `15e7e14`). What per-frame accuracy says about a decoder vote is not
   measured, in either direction, and a spellF1 grid was never run: whether to run one is Tian's call, as a new proposal,
   conceived after seeing this result, and the result predicts a small effect.
3. **P20's step 0: done, no-go (R022).** PathCRF's published checkpoint on the one DFL match it never trained on,
   degraded to `tracking` v3, against P07 on the same gap frames: +0.0049 pooled against the +0.04 declared (PathCRF's
   own protocol on the clean match reached 0.9172, against P07's 0.8522 on degraded tracking: not like for like). The retry condition did not hold, so P20 is dropped and its code removed (recoverable at
   `2163406`). A GPU retraining on degraded copies was not pursued; that is Tian's call.
4. **Cheap, only if a person is free:**
   - ~~P21's step 0: do our identity swaps cluster at the ball? A14 already bounds all of the benchmark's player
     degradation at 0.013 for `main.toml`.~~ Dropped (A16): ratio 0.85, below the declared 1.5.
   - ~~P22(a), the label-shift prior.~~ Dropped (R020): the feasibility gate failed, ratio 0.863 against 0.75.
   - ~~P10 with a pretrained audio tagger.~~ Dropped (R018, R018b): only the hand-built band-pass was tried, the tagger was not.
5. **P16 still decides more than any of these.** Every ball-rich change so far has split the two truths: R010's decoder,
   R013's touch rule, and now A14's ball. Knowing our truth's grain says which of them helps on our match. **R023
   answered "both" (2026-10-31):** each of them is now read on the mean of the two truths.

---

## Foundations

### P01 — A validation set we can trust

**Status: done → [R002](results.md) to [R005](results.md).** The benchmark's default profile was `tracking` version 3
([R004c](results.md)) until 10 Nov 2026. Since [R030](results.md) it is `generic` version 1
([P27](#p27--a-degradation-that-measures-nothing-of-our-match)), which measures nothing of our match. `gaps`,
`tracking` and `context` read our match and stay in the code for the record of R002 to R029 only; what this section
says of them below is that record. `context` fails two of its gates and was a sensitivity profile only ([R005](results.md)).
The overview is the [benchmark README](../pipeline/benchmark/README.md), and the technical document's section 5 has
the design and the fidelity table.

**Rules changed on 22 and 23 Oct 2026.** The organisers' clarification of 22 Oct (upstream `c073f29`, RULES.md and
OVERVIEW.md) allows hand-labelled data from **other matches**, in any amount, and forbids hand labels on the competition
match itself — for validation and calibration too ([T27](traps.md)). The first plan for P01 (label about 15 minutes of
BAR-ATH from the video, after labelling the public window blind to calibrate) is therefore out. The goal is unchanged;
the labels now come from other matches. The clarification of 23 Oct (upstream `f8f2629`, RULES.md rules 1 and 2,
EVALUATION.md, FAQ.md) tightens it three ways. The published 5 minutes are for checking that the pipeline is correct and
are "not training data and not a tuning set", so the old reading (use them, predicted out of fold) is gone. No
human-produced data about the game from any source (event data, corrected tracking, possession data, annotations) may
be used, and external data must be about other games. And correcting the game's tracking by hand is hand annotation:
corrections must come out of code. Learning without human labels (unsupervised, self-supervised, pseudo-labels an
algorithm made) and watching the video or inspecting the data to understand them stay fine.

**Rules changed again on 3 Nov 2026** ([rules.md](rules.md), which now takes precedence). The organisers' emailed
clarification says "If you need to train (or retrain), you MUST use only external data" and that parameters are tuned
"on external data". So learning on our match without labels is out as well ([T54](traps.md)). Watching the video is
now only "to better understand what counts as possession", and "checking your output against the video" is not
allowed. That makes P01 the only place anything is trained or chosen, as it already was. The open question is the
degradation: `tracking` v3 is fitted on label-free statistics of our tracking (A07 to A09), and whether that is
allowed is question 1 for the organisers in [rules.md](rules.md).

**The organisers' reply of 5 Nov** ([rules.md](rules.md)) answered question 1 with the clarification and one more
sentence: "anything that is not explicitly allowed is not permitted". Nothing allows the degradation explicitly, and
the training data it makes carries our match's missing-ball masks and rates, so by our reading it is not permitted.
P27 is the redo, which Tian decided on 9 Nov. It was done on 10 Nov ([R030](results.md)).

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
   of our tracking at a time (`degrade.py`: `clean`, `gaps` and `tracking`, the default until R030, in its version 3
   since R004c; `context`, with R005, is a sensitivity profile only; since R030 the default is P27's `generic`, which
   is not built this way):
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
     sensitivity check, the default stayed `tracking` (until R030), and the four yardsticks keep their order on
     `context` under both truths (R005).
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
     the four yardsticks output on our match (A07: possessions per minute and D shares). Each of these three
     degradations was fitted once per version, on label-free statistics of our tracking, then frozen, and never tuned
     towards a WARN or a score ([T37](traps.md)). That fit is what the reply of 5 Nov closed; `generic` is fitted on
     nothing and has no fidelity gate (P27).
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
  to rank methods and to fit what transfers (switch penalties, duration priors, model weights). Timing (P04's
  back-dating, P05's `D` onset, P12's shift) may be inspected on the public window through P02, to understand it and
  never to fit it (T01, T27; `f8f2629`), and what is not settled stays with other games (P16). A06 measured DFL's grain
  (437 possessions a match, median 3.7 s) and found that the public window cannot say how far ours differs, because it
  is livelier than 96% of DFL's windows:
  [A06](results.md#a06--dfls-conventions-against-the-public-window-and-the-literature-2026-10-23).
  So rank the choices that depend on possession length under the raw and the coarse truth (`truth.coarse_codes`,
  `--truth coarse`: a sensitivity check, not a correction) side by side, and treat a disagreement as a warning
  ([T33](traps.md)).
- **Our other assets.** DFL comes without video, audio, homography or Smart Tagging, so P08, P09, P10, P14 and P15
  cannot be benchmarked directly. Either simulate them from the benchmark's truth with the noise measured on our match
  (A03 for P09), or read them on the public window as a correctness check only, with their strength fixed by a declared
  constant and never by the window.

If a convention cannot be read from the DFL flags, hand-labelling other matches is allowed, in any amount. Never
label BAR-ATH. The DFL data is public and outside the competition bundle, so its own licence governs it, not the
bundle's rules. Cite it in the technical document.

### P02 — Inspect the ground truth's conventions on the public window, to understand and never to tune

**Status: done → A11, R012, A15 (inspection only; a person may watch to understand):
[A11](results.md#a11--our-ground-truths-timing-against-dfls-from-tracking-events-2026-10-25-tian),
[R012](results.md),
[A15](results.md#a15--descriptors-of-our-truths-changes-against-dfls-2026-10-30-tian).**
*Changed 23 Oct 2026 (`f8f2629`):* this proposal began as "learn our GT's timing conventions from
the public window". The organisers' rule now says those 5 minutes are for checking that the pipeline is correct and are
"not training data and not a tuning set", so P02 is inspection: it measures and watches to understand how our truth
flips and to design features, and it fixes no constant. Its two findings stand as findings (A11 was inspection and
adopted nothing); a number derived from them would be tuning. The instrument,
[`pipeline/benchmark/conventions.py`](../pipeline/benchmark/conventions.py), times each of the 20 public changes against
label-free events on our tracking and sets them beside DFL's (its `tracking` v3 copy, raw and coarse truth). A11 reads
it as a lead and not a measurement (2 to 4 changes behind each `D` cell): the events our tracking sees put our `D` 2.5
to 3 s longer than DFL's flag would be, and our team flips about a second before the new team's first controlled touch,
where DFL's flag flips at it (the interval holds 0). Each is our truth's convention, or chance, or our tracker's
coverage (it loses the ball near the lines), and A11 does not separate them. The pipeline's own latency and the
`shift_edges` values are R012's: the rule adopted the pipeline's own lag on DFL (team +1, into `D` -2, out of `D` 0
frames) and no gap to our convention (g = 0, from 1 to 4 matched public changes per type); its gain is inside noise.
The rule's second half, g, read from the public window's latencies, is **retired**: it is a tuning use of the window, it
stays 0 by rule, and any future shift comes from other games only (P16). What is left needs a person who may watch the
video, to understand what the restart types are and whether `D` begins at the line, the whistle or the stoppage.
*Changed 3 Nov 2026 ([rules.md](rules.md)):* watching is now only "to better understand what counts as possession",
and "checking your output against the video" is not allowed. So the person watches the truth's changes and ignores the
file's detector and descriptor columns, which are our own components' events: comparing those with the video would be
checking our output. The person's command, which regenerates the file to watch, is

```bash
python pipeline/benchmark/conventions.py public     # writes data/runs/conventions/public_transitions.csv, gitignored
```

with one row per change (its frame in the tactical video, the label before and after, the type, and each detector's
lag, and now the descriptor columns of A15). No LLM reads that file, and nothing is written from it but aggregates (A11, "How a person can extend it"; T27
allows watching and inspecting to understand, and never turning what was seen into labels, corrections, or rules and
constants tied to specific frames).

**Why.** The GT is mechanical: `D` where the corrected tracking says the ball is dead, otherwise the team it names.
*When* it flips matters more than anything. Does `D` start when the ball crosses the line, or at the whistle? Does
possession change at the first touch, or once control is established? With 0.3–1.5 s of slack on short possessions
(fact 5), a systematic half-second offset is fatal.

**What.** Around each of the 20 public transitions (12 team→team, 4 into `D`, 4 out of `D`), measure locally:
- whether the ball is visible, and its position and speed;
- the nearest player of each team and their distance;
- where the ball is relative to the lines, and the time from the ball leaving the pitch to `D`;
- the restart type, for the person's understanding and not as a record for those frames.

Watch each transition on video to understand what counts as possession there, never to judge our own output against
it (3 Nov, [rules.md](rules.md)). What goes into [traps.md](traps.md) and the technical document is what was
understood, in aggregates, and never a constant or a rule tied to those frames.

P01's benchmark follows DFL's conventions, not Metrica's, and the public window is not a place to fit ours: it can be
inspected to understand them, and read as a check. A convention of our provider has to come from other games (P16), and
a constant from it is fixed by a declared rule on those games, never read off these 20 changes. Watching the video and
measuring around the published transitions is fine. Writing labels for frames of this match, or turning what was seen
into a number for a config, is not ([T27](traps.md)).

### P16 — Metrica's sample games, in our ground truth provider's conventions

**On hold since 5 Nov 2026.** The sample-data repository has no licence (GitHub reports none, checked 9 Nov). The
organisers' reply to question 3 was the clarification ("any dataset you have the rights to use. Please check the
license") and "anything that is not explicitly allowed is not permitted". So by our reading we do not have the rights,
and nothing below runs again. R023 stays in the log as used once, and the technical document says it was withdrawn.
R024's objective stays without it, on A06's ground (Tian, 9 Nov; [rules.md](rules.md), "Question 3 in detail").

**Status: partly done → [R023](results.md) (signed by Tian and run on 2026-10-31). The three games' event-derived
labels give 6.20 possessions per alive minute (class both) and a median possession of 4.72 s (class raw, 0.11 s under the
band). The outcome is **both**: choices declared from now on rank on the unweighted mean of the raw and coarse pooled
spellF1s, each reported. Nothing past is reopened and `main.toml` is untouched. Not done: the switch timing and the feeds
below (P04's back-dating, P05's `D` onset, P12's shift).** *4 Nov ([A22](results.md)):* the timing part is now the one
lawful route to a constant for where our provider's `D` begins and ends. The organisers left issue #31 (when `D` starts)
unanswered, no public game carries Metrica's tracking flag (game 3's EPTS metadata declares x and y ball channels and
nothing else, checked 4 Nov), and the window may not set a constant (T01). What is measurable on the sample games is the
ball's own track against the events: the lag from the ball crossing a line to the end of the BALL OUT, and from the SET
PIECE's start to the ball moving; it would say how Metrica's operators time a stop and a restart, which is the
convention the corrected tracking's flag most likely follows. Effort S to M; a declared rule before the run, as R023.

**Why.** P01's truth follows DFL's conventions, and P02 inspects Metrica's in 20 transitions in one 5-minute window,
which is livelier than 96% of DFL's windows (A06) and, since `f8f2629`, no place to fit them. When the flags flip and
how fine our GT's possessions are remain the two things the benchmark cannot settle ([T33](traps.md)). Metrica's three
public sample games come from the provider of our GT, and ship with event data to derive possession from (P13 lists
them). Whether possession derived from events follows the conventions of our GT is the question P16 puts: our GT comes
from corrected tracking flags, and labels derived from events may not follow them. Whether to adopt them is decided by
a rule declared on the sample games themselves, never by their agreement with the public window, which is no
validation set. If they pass, they give far more transitions than the public window's 20.

**What.** Do it alongside P02.

- Declare first, before any result, the rule that decides whether the derived labels are used, and apply it on the
  sample games alone. A comparison of the derived grain and `D` timing with what the public window shows (P02) may be
  read as a correctness check and to understand how they differ; it cannot gate adoption, and a constant taken from
  P16 is fixed by the declared rule on the sample games alone.
- Derive `H`, `A` and `D` per frame from each game's events and tracking, the way our GT is built: `D` where the ball
  is dead, otherwise the team that has it. The events tell where a possession starts and where play stops.
- Study the switch timing as P02 does around the public transitions, but on many more of them: the lag between the
  touch that wins the ball and the flip, the `D` onset against the ball crossing a line, and the grain (possessions
  per alive minute, median length), set beside DFL's in A06.
- Feed P04's back-dating, P05's `D` onset and P12's shift from the sample games, and read each against the public
  window through P02 as a correctness check only. If the grain lands between DFL's raw truth and the coarse one, add a
  truth of that grain next to the two, and never retune a degradation towards it ([T37](traps.md)).

**Rules.** Hand-labelled data and event data of other games is allowed, in any amount; BAR-ATH is never labelled, and
nothing human-produced about it is used from any source ([T27](traps.md), `f8f2629`). Check each source's licence
first, keep it public so that the organisers can access it, and cite it in the technical document with who labelled it
and how much (P13's rule). Use nothing about this fixture ([T26](traps.md)).

**Draft protocol (2026-10-31, advisor for Tian, unsigned).** [R023](results.md#r023--p16-part-1-metricas-sample-games-and-the-grain-of-our-truth-draft-2026-10-31) part 1
fixes, before anything is fetched:
- one derivation rule from Metrica's event definitions;
- A06's grain statistics;
- the rule that maps the grain to the objective (raw, coarse or both);
- a licence stop;
- what voids it.

It covers the grain only, not the switch timing. Its "both, with equal weight" replaces the fourth bullet's "add a
truth of that grain" if Tian signs it. Nothing runs until Tian signs or amends it.

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

**Status: done → R008, R010; the rule re-chosen → R013.** The source is `controlled_touch`
([`pipeline/evidence/`](../pipeline/evidence/README.md)). Chosen on a 24-config grid (R008): radius 2.0 m, relative
speed 6 m/s, 3 frames, hold 0.55, confidence 0.91. Alone it clears both halves of P06's bar (+0.0328 raw, +0.0206
coarse); in the chosen config removing it costs +0.1421 raw and +0.2132 coarse (R010). R013 chose it again on
`main.toml`'s own evidence, without the window-derived `ball_presence` that every R008 config carried: radius 1.5 m,
6 m/s, 1 frame, hold 0.60, confidence 0.93 (192 configs; on three lower edges of its grid).

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
- Calibrate r and k on P01 (other games), never with the public window or its score (T01, `f8f2629`).

### P05 — A real dead-ball detector

**Status: done → R009, R010.** The 30 play features, a fixed-rule source and a gradient-boosted model are in
(`dead_ball_rules`, `dead_ball_model`, the trained JSON in `pipeline/models/`). Boosting was chosen over logistic
regression and the rules (R009). Alone it clears both halves of P06's bar by a wide margin (+0.1486 raw, +0.1924
coarse), but the benchmark overstates it for our match by an amount it cannot measure (T38, T39). The restart-team table
below is not built: no source names the team that restarts.

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
[`pipeline/configs/p06_hmm.toml`](../pipeline/configs/p06_hmm.toml), which is not the submission config. The
submission's decoder parameters are R010's and then R013's (`switch_team` 4, `switch_dead` 32, `min_possession` 35,
`min_dead` 25), chosen with the sources that `main.toml` carries.

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

**Status: done → [R011](results.md).** The source is `shape_possession`
([`pipeline/evidence/`](../pipeline/evidence/README.md)): boosted trees on 26 features of how the two teams stand and
move, none of which reads the ball, trained on the seven DFL matches with the ball hidden and read symmetrically in
the two teams (`pipeline/benchmark/train_shape.py`; the model is `pipeline/models/shape_possession_hgb.json`). Out of
fold on the benchmark (`tracking` v3, raw truth), on the alive frames whose ball is missing it scores a log-loss of
0.3759 and an accuracy of 0.8349 (ln 2 = 0.6931). The 12-config grid chose weight 0.2 and every graded frame: on
`tracking` v3 it adds +0.0942 pooled spellF1 under the raw truth over P04 + P05 (7 wins in 7, p 0.0156; +0.1035 under
the coarse one), and the label-free transfer gate on our match passes (0.8251 against 0.7486). `main.toml` carries it,
and R013 chose the weight again with the decoder (0.2 once more, 144 configs).
The limits: the model never saw our teams, so the benchmark's gain is not a forecast for them; the count of
possessions moves the other way on BAR-ATH (down 2.4%, where the benchmark's goes up 16%); and the public window's
spellF1 fell by about one possession, 0.4686 to 0.4000 (a correctness check, T01). What was built differs from the
plan below: the model trained on the DFL matches with the ball hidden (P13's route), not on pseudo-labels from P04 or
on the public GT, and its features are R011's 26 (no hull area, and none near the ball zone, since none reads the
ball). What comes next is in [After P07 and P02](#after-p07-and-p02).

**Why.** Fact 1: four frames in ten have no ball, in gaps long enough to hold several possessions.

**What.**
- Per-frame, permutation-invariant features of all tracked players: team centroids, length, width, hull area,
  defensive-line heights, mean velocity toward the opponent's goal, players per third, and pressure near the ball
  zone. Add how each of these changes over ±1–2 s.
- Derive each team's attacking direction per half from the data.
- Train H-vs-A on ball-visible frames where P04 is confident (pseudo-labels, tens of thousands of frames) and on P01's
  matches with the ball masked (P13). Not on the public GT, out of fold or not: the organisers' 23 Oct rule (`f8f2629`)
  makes it a check and no training data ([T27](traps.md)). What was built used P01's matches alone (R011). The
  pseudo-label half of this plan has not been allowed since 3 Nov 2026: training uses external data only
  ([rules.md](rules.md), [T54](traps.md)).
- Measure on P01's **simulated ball gaps**, where the truth is known, and check against the public window's
  ball-missing frames (few: 6.5% of its in-play frames). Real gaps are harder than the `gaps` profile's, which are
  transplanted with their pattern and not their clock, and not placed by the play: ours start near the lines and on
  slow balls, with more players around the ball (A08). The `context` profile (R005) places them that way, fails
  two of its gates and is a sensitivity check only, so the benchmark's default still has gaps that are blind to the
  play.
- Feed the probabilities into P06.

### P08 — Smart Tagging as soft evidence

**Status: done → [R017](results.md) (team votes adopted per frame, weight 0.2; dead votes measured and dropped); taken
out of `main.toml` → [R029](results.md) (3 Nov 2026: `c_team` was measured on our match, [rules.md](rules.md)).** The team
votes passed their declared bar (0.761 agreement with our own controlled-touch pseudo-labels) and entered `main.toml` at
Tian's pick of strength (2026-10-30); the dead votes failed theirs (0.409) and were removed from the code.

*Changed 3 Nov 2026:* the organisers' clarification ([rules.md](rules.md)) requires parameters to be tuned "on external
data". `c_team` 0.76 is a measurement on our own match, so it conflicts ([T54](traps.md)). The weight was picked after
seeing our match's output and is doubtful. DFL has no tags, so no external data can set either. Recommended: take
`smart_tagging` out of `main.toml` (R017 measured no score for it), ask the organisers (question 2), and put it back
only on a yes. The plan below had a compliant route, "a declared constant argued from what the tags mean". That route is
not clean any more, because any number declared now is chosen knowing 0.761. *Decided 3 Nov (Tian):* the source is
out of `main.toml` (R029, sha256 `762f9073…`). The organisers were not asked; the code stays, off.

**Why.** There are 638 automatic tags, each naming a team: free evidence exactly where the ball is missing.

**What.** Turn each tag into a per-frame vote:

| tag codes | vote |
|---|---|
| `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `ATTACKING TRANSITION`, `LONG BALL` | the tagged team has the ball |
| `DEFENDING IN … THIRD`, `DEFENSIVE TRANSITION` | the other team has the ball |
| `SET PIECES` | a stoppage ending in a restart by the tagged team |
| `GOALS` | a stoppage until the kick-off |

Fix each code's vote beforehand, by the table above and not by its agreement with the public GT, then feed the votes to
P06 as features. P01's matches have no Smart Tagging, so there is no benchmark for this source: its weight is a declared
constant argued from what the tags mean, and the public window only checks that the votes are read the right way round
(a correctness check, never a tuning set, `f8f2629`). Never copy intervals across as labels. Map `Barcelona` → H,
`Atletic Club` → A and `N/A` → no vote. Tag windows are seconds wide, and phases are not possessions ([T17](traps.md)).

### P17 — The restart-team channel

**Status: dropped → [R014](results.md), [R014b](results.md) (the source `restart_team` and the `hmm` decoder's restart
states stay in the code, off by default; `main.toml` is unchanged).** Phase 0 ([A12](results.md)) measured a ceiling of
+0.0198 pooled raw spellF1 (+0.0197 coarse, 7 wins in 7), and the decision was GO against a bar of +0.01. The source was
built and its confidences measured (A12 part 2), but neither declared test met its bar of +0.005 raw with p at most
0.05: the vote after the restart (R014) gave +0.0040 raw (p 0.0938) and -0.0029 coarse at best, and restart states in
the decoder (R014b) +0.0003 raw (p 0.8125). The follow-ups are in [After P17](#after-p17).

**Why.** Facts 3 and 5. After a stoppage `controlled_touch`'s hold weakly favours the team that controlled the ball
before it, until the first new episode names the team ("After P04 and P05" item 4, "After P07 and P02" item 4, "After
R013" item 4, and P05's restart table, which no source built). A12 sizes it: the restarting team differs from the
previous one after 70.3% of the benchmark's stoppages, `main.toml` names the restarting team right on 78.7% of the 50
frames after a restart, wrong on 14.1% and still `D` on 7.2%, and the ceiling is the +0.0198 above. On our match the ball
is missing at 39.8% of the predicted stoppages' ends and 20.2% of their restarts, so the rule has to lean on the players.

**What.** A label-free source, `restart_team`, returning only `home`:
- The stoppages come from the dead-ball model, the same runs the decoder sees (P(alive) below 0.5 for at least 25
  frames), and the restart is the first frame after one.
- The players' arrangement in the 2 s before the restart says the type and the team: kick-off, corner, goal kick,
  throw-in and, when the ball is visible, free kick. Five rules from the Laws of the Game (8, 13, 15, 16, 17), with the
  opponent distances halved because real players encroach, declared and not tuned.
- A vote only from the restart onwards, for a tail of frames, never into the stoppage for span "after" (the dead-ball
  edges are not the source's job; R014b's span "stoppage_end" later spread one observation over the stoppage instead)
  and never across a block edge.
- The confidence of each type is measured on DFL, as `controlled_touch`'s is, and a type that cannot be measured does not
  vote.
- R014 chooses the tail and whether the ball is read, and adopts or drops the source by a rule declared before the run,
  reading both truths.

**What it cannot settle.**
- The ceiling is an upper bound: the oracle knows the true restart frame and team. The real source estimates both.
- The benchmark's ball is tracked out of play (T39), so anything that reads the ball looks better on DFL than on ours:
  the rules are players-only, and free kicks, which need a visible ball, are the part that may not transfer.
- Our match has no restart types to measure against: a type read off its video by a person would be a hand label (T27).
- Smart Tagging's set-piece tags are left out on purpose: they would be a P08 follow-up, with no benchmark to choose
  their weight on.

### P18 — The offpitch profile: DFL's ball hidden off the pitch as ours is, and P05 retrained on it

**Status: dropped → [R015](results.md), [R015b](results.md) (our tracker's off-pitch ball cannot be reproduced by a simple
rule; code removed).** Tian's rule at the pause of 29 Oct: if the v2 profile failed, P18 would be dropped. It failed R015b's
feasibility check, and the offpitch code was removed in `7b555eb` (recoverable from `fb33b38`). [A13](results.md) part 1
stays; part 2 and R016 never ran. What it leaves is in [After P18](#after-p18).

**Why.** T39. P05's boosted trees are the largest single contributor to `main.toml` (+0.1661 raw at R010's decoder), and
the docs say the benchmark overstates them for our match by an amount it cannot measure. The trees split first on where
the ball is: past a line, last or next seen past a line, at a restart spot. DFL's tracker keeps the ball while it is out
of play and ours loses it there: the ball is past a line on 17.68% of DFL's at-risk rows against 4.07% of ours (R005,
R009). The model learned a cue that is mostly absent on our match. Belief (a) was that the current P05 relies heavily on the
visible out-of-play ball, so it should lose clearly when that ball is hidden. A13 adds the other half: on the benchmark a perfect
dead-ball source is worth only +0.029 raw to `main.toml` (alive/dead F1 is already 0.973), which is what a near-perfect
P05 on a ball that is visible out of play would produce.

**What.**
- **A13** (done, part 1): the headroom of `main.toml` with oracles added to its own evidence (`plus_` variants), on
  `tracking` and `context`. Part 2, on `offpitch`, was not run.
- **R015** (done): a benchmark profile, `offpitch` v1, in which `tracking` v3's ball is also hidden once it is
  more than m past a line (until it is back on the pitch), and a share q of the transplanted gaps is dropped. m is fitted
  once so that the share of visible ball frames past a line equals ours (0.0407), and q so that the missing share equals
  ours; both are frozen (m 0.2 m, q 0.430). Tian reopened R005's ruling against a mask for the ball off the pitch on
  2026-10-29 for this separate profile, because the parameter is fitted to a statistic of visible frames. `main.toml` on
  it scores 0.5509 raw and 0.5296 coarse (0.5430 and 0.5593 on `tracking`; alive/dead F1 0.9731 to 0.9600).
- **R015b** (done): a v2 rule, the ball hidden while more than m past a line and moving, found again at rest, swept once on
  label-free statistics. Its share past a line stays at 0.258 to 0.262 at every margin, against ours 0.0407. P18 dropped.
- **R016** (never ran): retrain P05 on `offpitch`. The P18 retrain was never run and its number was reassigned to P12's free X on 30 Oct.

**What it cannot settle.**
- What stays unobservable on our match is how long our ball spends out of play in total. `offpitch` assumes it is like
  DFL's; the assumption is written in R015.
- A profile is a model of our tracking, not our tracking: the gates and checks say where it differs (R015: median gap,
  the near-line onset share and the ball visible on dead frames).
- What R015 and R015b found: two profiles that hide the out-of-play ball were tried. v1 matched our share past a line and
  made the gaps too long (median 15.04 s against 10.74); v2 could not reach the share at any margin (0.258 to 0.262
  against 0.0407). Our tracker's off-pitch ball is not reproduced by a simple rule, and where it is during our own gaps
  is not observable. With the ball hidden by v1, today's P05 lost 0.013 alive/dead F1 (0.9731 to 0.9600), a lead, not a
  measurement, that belief (a) is weak, since the profile failed a gate.

## Who has the ball: the next layer (added 29 Oct)

A13 puts the remaining score on the team side: +0.194 raw where the ball is seen, +0.176 in the gaps, against +0.029 for
the dead ball. These four came out of a literature survey written for that finding ([After A14](#after-a14)). Each states
its belief and what would falsify it, starts with a feasibility test that reads aggregates only, and needs its own
protocol and bar declared before its runs. Every impact figure below is a guess until that test has run.

### P19 — A learned team model where the ball is seen

**Status: dropped → [R021](results.md) (the feasibility gate failed: out of fold, P19 was right on 0.9087 of the
frames against 0.8927 for `main.toml`'s decoded labels, +0.0160 pooled against the declared +0.020, and 3 of 7 matches
cleared it against the 6 needed; all seven deltas are positive, mostly on the hold frames, +0.0257 against +0.0035 in
episodes). The grid, the bar and the transfer were not run and no spellF1 was produced; the code is removed, recoverable
at `15e7e14`.** A grid without the per-frame gate would be a new proposal for Tian, conceived after seeing this result, and the result
predicts a small effect.

**Why.** `main.toml` has no learned team source where the ball is seen. `controlled_touch` is a fixed rule with four
constants, three of them on the edges of R013's grid, and it speaks on only 36.9% of the ball-visible frames on BAR-ATH;
everywhere else the decoder holds the last team. P05 showed what boosted trees trained on DFL's flags did for the dead
ball (+0.1661 raw at R010's decoder). P19 applies the same recipe to the team, which is the larger lever (A13).
- **Belief:** a model that reads the ball, the nearest players of both teams and the touch episodes on either side of a
  frame names the team more often than the rule and its hold do.
- **Falsified if:** out of fold, it is not better per frame than the rule and its hold on the same frames.

**What.**
- Boosted trees, sklearn's `HistGradientBoostingClassifier`, seeded and saved as JSON as P05 and P07 are, trained on the
  alive, ball-visible frames of the degraded DFL copies (`tracking` v3). The target is `BallPossession`, and the trees
  read the two teams symmetrically, as P07 does.
- Features:
  - per team, the two nearest slots to the ball: distance, closing speed and relative speed, and each team's time to
    reach the ball;
  - the ball's speed, acceleration and turn angle over ±0.2, 0.5 and 1 s, from its positions (T31);
  - a "team over 11 slots" flag (T32);
  - the time since and until each team's last and next touch episode (offline, so both directions are allowed);
  - P07's 26 shape features.
- An optional second head for the gaps' edges: time into the gap and time left in it, the team in control at each edge,
  and where the ball was last and next seen.
- It enters the HMM as a `home` source, beside `controlled_touch` or instead of it. A declared grid chooses its weight
  together with `switch_team` and `min_possession`, since a learned source and the hold can count the same evidence twice.

**Feasibility first (S, half a day).** Leave-one-match-out on the benchmark: per-frame accuracy and log-loss on the
alive, ball-visible frames, against what `main.toml`'s own labels give on the same frames. Go if it is at least 2 points
better in 6 of the 7 matches.

**Then the bar,** as R014's: +0.005 pooled raw spellF1 over `main.toml` with sign-flip p at most 0.05 on `tracking`; the
coarse truth not below -0.005; `context` in the same direction (sensitivity only); and a label-free transfer check on
BAR-ATH, as R011 had.

**What it cannot settle.** It learns DFL's flag, which is fine-grained (T33): read the coarse truth beside the raw one.
The literature puts a rule-based model of individual ball possession at F = 0.83, and of individual ball actions at
0.88, on 60 Bundesliga matches (Link and Hoernig 2017,
[PLoS ONE, CC BY](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/)). That is a reference for the rule, not a forecast for
our match.

- **Research (2026-10-31).** [docs/research/2026-10-31-p19-team-model.md](research/2026-10-31-p19-team-model.md) checked the
  frozen design against the literature before any code. A learned model beats a fixed rule in every source that compares
  them, and boosted trees on per-frame features with temporal context, decoded by our HMM, is the sound middle form; the
  sequence models (TCN, Bi-LSTM, set-transformer, CRF) and Spearman's full pitch-control integral were left out as too
  heavy for a byte-for-byte reproduction. Spearman's time to reach (a 0.7 s reaction, then a run at the top speed) is the
  one feature taken from it, at 5 m/s, not the 7 m/s first written (R021, Advisor (for Tian), §5.1).


### P20 — Ball-free possession paths: PathCRF on our slots

**Why.** In the gaps, P07 classifies one frame at a time from the teams' shape (83.5% out of fold on gap frames, R011);
the HMM alone links frames over time. PathCRF (Kim et al., KDD 2026,
[arXiv 2602.12080](https://arxiv.org/abs/2602.12080); [code](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0, five
trained checkpoints) infers who has the ball from player trajectories alone. Set-attention encoders read the players at
5 fps in 10 s windows. A CRF then chooses, at every step, which player holds the ball, or which of four out-of-play
nodes, one per line, it has gone to. Viterbi decodes the path, and impossible transitions are forbidden.
- It was trained on the **same seven Sportec/DFL open matches as our benchmark**: five for training, one for validation,
  one for testing.
- On its test match it reports 92.40% frame-level team accuracy, without the ball. That is on clean tracking with known
  identities, and it is not like for like with P07's 83.5%.
- It also gives the possessor's changes, which time the boundaries, and out-of-play events, which are evidence for `D`.
- Its group's earlier work inferred the ball's trajectory from players alone (Kim et al., KDD 2023,
  [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)).

- **Belief:** a model that reasons over a sequence of hand-overs between players gets the gaps right more often than
  shape read frame by frame.
- **Falsified if:** under our degradation it does not beat P07 on the same gap frames.

**What.**
1. **Step 0 (S, one to two days).** Run the published checkpoint on its own test match, the only one of our seven it
   never trained or validated on.
   - Run it on the clean match, and on the match degraded to `tracking` v3.
   - The degraded copy needs an adapter: 46 anonymous slots become at most 22 nodes per 10 s window, by team, with slots
     that end and begin joined by proximity.
   - Compare its team accuracy on the gap frames with P07's out-of-fold on the same frames.
   - Go if it is at least 4 points better on the degraded copy.
2. **Step 1 (L).** Retrain leave-one-match-out on degraded copies: the checkpoints saw six of our seven matches, so the
   benchmark cannot score them anywhere else.
   - Training needs a GPU. A cloud GPU is acceptable here because it sees only DFL data (public, CC BY 4.0); nothing of
     BAR-ATH ever leaves our machines.
   - The weights are committed, inference on BAR-ATH runs on our CPU, and the per-frame team probabilities feed the HMM
     as a `home` source.
   - Its scope (gaps or all frames) and weight are chosen by a declared grid, as R011's were. Its out-of-play nodes may
     later feed `alive`.

**What it cannot settle.**
- How far 92.4% falls under our identity breaks: tracklets shorter than a window, and wrong-team slots (T32).
- Byte-for-byte reproduction with a neural network in the path. As P14: one thread, committed weights, and the CSV
  hashed on an AMD and an Intel machine early.
- MPL-2.0 is file-level copyleft. It is fine in a public repo, but its files keep their licence headers.

**Step 0, measured (R022, 2026-10-31).**
- **Licence and split:** MPL-2.0 for the code, with no separate licence for the weights (read as the repository's). The
  checkpoint's test match is J03WR9 (validation J03WQQ, training the other five), shown by its `args.json`
  (`lomocv`, fold 6), the tutorial and the paper's test counts.
- **What the 92.4% is:** event-split possession on in-play frames after their own preprocessing, not DFL's flag. Their
  protocol reproduced on the clean match (team accuracy 0.9164, edge accuracy 0.695, event F1 0.756).
- **Pins:** commit `1b7c6f85…`, 27 files by blob SHA-1 and sha256; torch 2.11.0, numpy 2.2.6, pandas 2.2.3 in a
  separate venv.
- **The adapter:** windows on the maximal H or A runs of `main.toml`, pieces joined within 2 s and 5 m, at most 11
  nodes a team, gaps filled linearly (D1 to D4, advisor-01 section 6).
- **Result:** on 90,449 primary frames PathCRF 0.8571 against P07 0.8522, Δ +0.0049 (needed +0.04 pooled and +0.02 per
  seed); PathCRF by its own protocol on the clean match 0.9172, not like for like (the truth's in-play episodes, true
  identities); −0.0182 under the coarse truth. The retry condition was false. Dropped.

### P21 — Wrong-team slots: measure the ceiling, then repair them

**Status: dropped → [A16](results.md) (step 0: over-11 frames are 0.1097 of `controlled_touch`-style candidate frames against 0.1286 of graded frames, ratio 0.85, below the declared 1.5; the onset ratio, 0.79, is report only and shares the ratio's denominator; a ratio below 1 may partly reflect ball visibility, as the candidates need the ball seen and the comparison is all graded frames). A14's bound stands. `pipeline/benchmark/wrong_team.py` and its checks are kept, because P19's transfer check reports its ratio; no repair was built.**

**Why.** T32: on at least 10.8% of the graded frames a team shows more than 11 slots, because a slot has followed a
player of the other side. `controlled_touch` takes the team of the nearest slot at confidence 0.93, so a swap next to the
ball injects a confident wrong team. The benchmark places its wrong-team episodes at the closest home and away pair
anywhere on the pitch, not at the ball, so it may understate the cost. But A14 bounds it there: all of `tracking`'s
player degradation (tracklets, wrong-team slots, NaN speeds, smoothing) costs `main.toml` only 0.0129 raw and 0.0126
coarse (`gaps` 0.5559 and 0.5719 against `tracking` 0.5430 and 0.5593).
- **Belief:** our swaps cluster at the ball, which the benchmark does not simulate.
- **Falsified if:** they do not cluster there.

**What.**
1. **Step 0 (S, an hour).** On BAR-ATH, measure the share of `controlled_touch`'s candidate frames that have a team over
   11 slots, against the 10.8% of all graded frames.
   - If it is not well above 10.8%, drop P21: the benchmark's bound applies.
   - If it is, measure the ceiling: `main.toml` on `tracking` with its wrong-team episodes switched off.
2. **(a) From the tracking (S).** At each encounter under 2 m, test whether the two slots' continuing trajectories fit
   better swapped (velocity continuity), under a rule of at most 11 slots per team. The benchmark can score it, but its
   degradation mirrors the repair, so the result partly confirms itself (T37).
3. **(b) From the video (M).** Cut each slot's track at its close encounters, project each piece's foot point through the
   homography, crop the torso, and cluster colour histograms per half without labels (KMeans, two teams plus goalkeepers
   and the referee). A piece's majority overrides the metadata's team when it is confident.
   - Unsupervised, so allowed until 3 Nov 2026. Since then it is not: clusters fitted on our frames are "fitting ... on
     the provided video" ([rules.md](rules.md), [T54](traps.md)). A colour or team classifier trained on other games
     and run on our frames would be allowed. The video is read by code on our machines only.
   - The benchmark cannot score it (DFL has no video). The label-free checks on our match: the share of frames with a
     team over 11 slots should fall, and away from encounters the colour team should agree with the metadata's on at
     least 98% of the pieces.
   - Sources: Koshkina et al., unsupervised team classification
     ([CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/papers/Koshkina_Contrastive_Learning_for_Sports_Video_Unsupervised_Player_Classification_CVPRW_2021_paper.pdf));
     [roboflow/sports](https://github.com/roboflow/sports) (MIT) has a team classifier.
   - Decode to YUV planes with a pinned PyAV, as P14 would, so that the output reproduces.

### P22 — The shape model adapted to our two teams without labels

**Status: (a) dropped → [R020](results.md) (the feasibility gate failed: EM's share was 0.0538 from the true one on average against 0.0623 for guessing 0.5, a ratio of 0.863 against the declared 0.75; neither retry condition held, so no grid ran and no F1 was produced). The code was removed in one commit, recoverable from `e303365`. (b) dropped → [R027](results.md) (the gate on DFL failed: rung 1, an intercept on P07's logit fitted on the other half's touch-rule frames, made P07's pooled log-loss in the held-out half's gaps 0.0061 worse, 0.3759 to 0.3820, where 0.002 better was needed, and worse on 5 of 7 matches. The offsets exist, 0.276 on average in the gaps, but the one the touches see is not the gaps': the transfer is 0.296 against the teacher's noise of 0.107. Rung 3, the retry's column, was worse too, +0.0838, so R027b did not run, and no spellF1 was produced). The code was removed in one commit (`9ee4d1f`), recoverable from `1e5a5d2` and the local tag `archive/r027-source`. (c) not tried.** **Since 3 Nov 2026 none of (a), (b) or (c) is allowed:** each estimates something on our own match (a share by EM, a recalibration from our pseudo-labels, a prior), and training and tuning now use external data only ([rules.md](rules.md), [T54](traps.md)). Do not revive P22.

**Why.** P07 was trained on German teams only, and R011 warns that the benchmark's gain is not a forecast for Barça and
Athletic: the count of possessions moved the other way on BAR-ATH.
- **Belief:** part of that gap is a shift in priors and features that can be corrected without labels.
- **Falsified if:** the correction does not help in the leave-one-match-out simulation below.

**What.**
- **(a) S:** estimate our match's home share from P07's own outputs by EM label-shift (Saerens, Latinne and Decaestecker,
  *Neural Computation* 14(1), 2002), and shift its logit by the result. Never take the share from published statistics
  of this match: they are human-produced data about it (rule 2).
- **(b) M:** one round of class-balanced self-training of P07 on frames that `main.toml` labels with confidence (a
  confident touch inside a long decoded possession).
  - Cross-fit by half: half 1's pseudo-labels adapt the model for half 2, and the reverse. This limits confirmation bias
    ([Arazo et al. 2019](https://arxiv.org/abs/1908.02983); CBST, [Zou et al. 2018](https://arxiv.org/abs/1810.07911)).
  - The pseudo-labels are made by an algorithm and decided by no person, so they were allowed until 3 Nov 2026 (not
    since: [rules.md](rules.md)).
  - Train on one half and check on the other. Without that split, R011's transfer gate (agreement with the touch rule)
    would check the model on its own training labels.
  - **Expanded 2026-10-31** (Tian: "start working on P22(b)"; design in
    `docs/superpowers/specs/2026-10-31-p22b-selftrain-design.md`, revised after an independent review, run R027). The
    design starts at the simplest rung of a ladder of updates (intercept, recalibration, team-aware head, retrained
    trees):
    - **Rung 1:** one logistic intercept on P07's logit per half, fitted on the other half's frames. The estimator is
      noise-aware: a forward correction at `main.toml`'s ε = 0.07, with the noise-corrected class share of those frames
      removed. On synthetic data a class-balanced fit is pulled about 0.1 towards the teacher's minority class.
    - **The labels:** `controlled_touch`'s episode frames at `main.toml`'s rule, without the frames `dead_ball_model`
      calls dead, computed by the same code on DFL and on BAR-ATH. The decoded labels are not used: in the gaps they
      carry P07 itself, and on BAR-ATH Smart Tagging too.
    - **Where it runs:** at run time in a new source `shape_selftrain`, with no new artifact.
    - **Why the intercept:** R020's misses hint at a per-match offset, and P07's slope is about right (α 1.07 to 1.10).
      When a prediction model is updated for a new population, the simplest update holds and revisions overfit (Janssen
      et al. 2008), and self-training under shift needs strong regularisation (Kumar, Ma and Liang 2020). No offset has
      been measured yet: the gate's oracle is the first measurement.
    - **The tests:**
      1. A gate on DFL, run from the tool before the source is built. It requires a pooled log-loss gain of at least
         0.002 nats in the held-out half's gaps, and no match worse by more than 0.002.
      2. The bar on the mean of the two truths: R025's form or a magnitude bar, Tian's choice.
      3. Label-free checks on BAR-ATH, among them Smart Tagging's votes in the gaps.
    - **One retry:** rung 3 (a team-aware linear head), only if rung 1 fails the gate and rung 3 passes it in the same
      run.
    - **Outcome (R027):** dropped at the gate. P07's pooled log-loss in the gaps rose 0.0061 under rung 1 and 0.0838
      under rung 3; the touches' offset is not the gaps' (transfer 0.296, noise 0.107). Code removed.
- **(c) M, recorded 2026-10-31 (advisor, for the team; not pursued tonight):** the prior as asymmetric switch costs in the
  decoder, the decoder's own place for a class prior. It is a contract change and needs its own decoder grid, so the team
  decides. R020 reads s = 1 as the frame-level Bayes value, not the decoder's prior. It would use the same share as (a),
  which failed R020's feasibility gate (0.0538 from the truth against 0.0623 for guessing 0.5), so it is untested but not
  promising as it stands.
- **Research (2026-10-31).** The estimator is Saerens, Latinne and Decaestecker's EM
  ([Neural Computation 14(1), 2002](https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf)),
  taken with the training prior fixed at 0.5 (our rows are mirrored). Alexandari, Kundaje and Shrikumar
  ([arXiv 1901.06852](https://arxiv.org/abs/1901.06852)) show EM beats the moment estimators once the model is calibrated,
  and Garg et al. ([arXiv 2003.07554](https://arxiv.org/abs/2003.07554)) that it is consistent under calibration: from
  them we took the calibration slope as a report-only diagnostic and temperature scaling as a pre-stated retry. Lipton,
  Wang and Smola's BBSE ([arXiv 1802.03916](https://arxiv.org/abs/1802.03916)) was left out: its confusion matrix would
  come from DFL, and whether it transfers is the doubt. Bourlard and Morgan's hybrid HMM (1994) says where a class prior
  belongs, in the decoder, which is why (c) was recorded. R020 tested only the estimator's feasibility, and it failed.
- **Validate:** simulate the procedure on DFL, leave one match out, each match in turn as the unlabelled target. A
  Bundesliga-to-Bundesliga shift is small, so expect "no harm" rather than a gain.

## The decoder revisited (added 31 Oct)

### P23 — The HMM backbone reviewed: nine ideas, five taken

**Status: done (Claude, for Tian, 31 Oct).** Tian took A to E and deferred F; G, H and I were not taken.
- **A → [R024](results.md): adopted.** The touch hold moves from 0.60 to 0.55. The mean of the two truths is +0.0052
  (p 0.0469, 6 wins in 7); `main.csv` was `e6496a9b…` (until R029 took Smart Tagging out on 3 Nov: `762f9073…`;
  R030 then re-chose the hold on `generic`, 0.60 by the declared tie-break, and replaced that file).
- **B and D → [A19](results.md): measured.** The IoU ladder is mostly edges on otherwise right pairs (0.070 of
  spellF1 on the mean). The sources are near calibrated; the decoder's posterior is about five times too sharp.
- **C → [R025](results.md), [R025b](results.md): dropped.** The gate passed. R025 moved edges by 1 frame (mean
  -0.0010), and its one retry, a posterior tempered at T = 5 (the advisor's ruling), moved them by 3 and made them
  worse (mean -0.0282, 0 wins). The code was removed, recoverable from `171d0bb`.
- **E → [R026](results.md): dropped.** Every heavier gap weight lost on the mean (-0.010 to -0.020): it cut more
  possessions, which the coarse truth punishes. The rule kept `main.toml`'s weights, and the advisor ruled no retry
  (A19 shows P07 calibrated alike with the ball seen or missing). The code was removed, recoverable from `57013a1`.
- **What P23 leaves:** `main.toml` gains R024's hold. The decoder's remaining loss is in where the evidence changes,
  not in where the decoder puts the change given the evidence. F (P22(b), the evidence in the gaps) is the one idea
  left aimed there; it ran as [R027](results.md) and was dropped at its gate.

**Why.** On 31 Oct Tian asked for a review of the decoder against the literature:
- is the HMM the right model?
- should we ensemble?
- where is it weak, and what would improve it?

An advisor (Fable 5.1, extra-high effort) checked the ideas and commented on each. This section is the review's record,
and the four entries above are its declared tests.

**The backbone: keep it.**
- **It is the standard design.** A per-frame classifier is decoded by a model that enforces sensible segments. This is
  the hybrid HMM of speech recognition, which reads scaled likelihoods (Bourlard and Morgan 1994), and the design of
  temporal action segmentation, which is scored by a segment F1 at IoU thresholds close to spellF1 (Ding, Sener and Yao's
  survey). A learned possession model still sits under such a decoder (Borghesi et al. 2023, a TCN).
- **The decoder is not the limit.** With perfect evidence it scores 0.918 raw and 0.969 coarse (A13). Real evidence
  leaves about 0.38 on the table.
- **Its knobs are spent.** R013's Stage C had ten configs within 0.005 of each other, and its choice is interior on
  every axis.
- **Learned replacements fell short here.** P19 gained +0.016 per frame (R021) and P20 +0.005 in the gaps (R022).

**Its weak points.**
- The hard minimum possession, 35 frames (1.4 s), rules out every shorter possession, however strong the evidence.
- A change of team costs the same everywhere: in a dribble, in a 50-50 duel, during a pass in flight.
- Each frame's evidence is counted as if it were independent of its neighbours. The weights correct for this in part.
- Viterbi returns the single best path, which is not the path that maximises the expected spellF1. In a stretch with no
  evidence, its tie rule puts a change on the stretch's first frame.
- One P07 weight serves both kinds of frame, ball-visible and ball-missing. Where the ball is missing, P07 is the only
  team source (`controlled_touch`'s hold votes only where the ball is visible, outside its episodes). At weight 0.2 it
  moves the H-against-A log-odds by 0.2 x logit(0.83) ≈ 0.32 nats a frame when it is right, against switch_team 4. A
  lasting change inside a gap needs about 13 frames of consistent evidence, and a contrary blip about 25. P07 is right
  on only part of the gap frames, so in practice it takes longer. (The review first said 40 frames, from the one-sided
  0.10 nats of log(2h). The declarations' reviewer corrected it.)

**Ensembles: not of decoders.** Several decoder settings on the same evidence share the same errors, so their medoid lands
in the middle of the plateau, which can be chosen directly. The sources are already combined as a product of experts,
which is the ensemble this design supports. The only ensemble with a mechanism behind it is bagging P07 (G).

**A correction made during the review.** Claude first put most of the ladder loss on badly placed edges: spellF1 falls
from 0.670 at IoU 0.3 to 0.401 at 0.7 (raw). The advisor's count says most of it is merged and split possessions instead.
`main.toml` predicts 6,279 possessions, against 9,183 in the raw truth and 4,977 in the coarse. Under the raw truth we
merge (recall 0.56), and under the coarse truth we split (precision 0.61), and moving an edge fixes neither. B measures the
split before C is built. *A19's answer:* the review's first reading was nearer. On the mean of the two
truths, the clean pairs lose 0.070 up the ladder, against 0.025 for merges and 0.027 for splits.

**A finding made while declaring D.** The review said the decoder's `log(2a)` assumes a 50/50 dead-ball prior, a bias of
logit(0.55) ≈ 0.2 nats a frame towards play. That bias is not there:
- `dead_ball_model` has subtracted logit(prior_alive) since P05 (its step 4), as scaled likelihoods require;
- P07's prior is 0.5 by construction (mirrored rows; the file is refused otherwise).

So D is the calibration check alone, and it adopts nothing. The declarations' reviewer found one 50/50 assumption that
does remain: team evidence under `D`. `hmm` adds a team source's terms to H and A and nothing to D, so wherever a team
source speaks, play is favoured over a stoppage by w · log(2 · max(h, 1 − h)) a frame (0.62 nats for a touch episode
at 0.93). That is wrong exactly where a player handles a dead ball. A19 measures it, and changes nothing.

| | idea | effort | expected gain | what it brings | decision |
|---|---|---|---|---|---|
| **A** | Re-choose R013's touch rule, decoder and P07 weight under R023's objective (the mean of the two truths), from R013's 336 existing reports | S | +0.005 to +0.015 | R023 changed the target, and every config was scored under both truths. The advisor called it the largest lever left | **do** → R024 |
| **B** | Decompose the IoU ladder into merged, split and edge losses | S | none by itself | Sets C's ceiling and gate; says whether merging or splitting costs more | **do** → A19 |
| **C** | Posterior edges: move each change to the median of the posterior over where it happened | S–M | +0.003 to +0.015 | The posterior is computed by `marginals` and not used yet; matters most in gaps, where P07 alone speaks | **do**, after B's gate → R025 |
| **D** | The dead-ball prior and every source's calibration | S | 0 | The prior is already corrected (above); the calibration of the sources and of the posterior tells C what its posterior is worth | **do**, as a check → A19 |
| **E** | P07 weighted apart in the gaps and where the ball is seen | S | 0 to +0.02 | In the gaps P07 is the only team signal, at the same weight as where touches speak | **do** → R026 |
| **F** | P22(b): self-train P07 on our match's own touch pseudo-labels, cross-fit by half, the procedure tested on DFL first | M–L | 0 to +0.03 | The one idea aimed at the gaps, where a perfect team signal is worth +0.176 raw (A13) | **deferred** (Tian, 31 Oct), then run as P22(b) → R027, dropped at its gate; the original condition: only if 4 days remain before the freeze; not allowed since 3 Nov 2026 (training on our match, [rules.md](rules.md)) |
| **G** | Bag P07's seven fold models | M | ≤ +0.005 | Less variance, better-calibrated gap probabilities; a fair benchmark needs 42 small models | not taken |
| **H** | A soft duration prior in place of the hard 1.4 s minimum (semi-Markov) | M | −0.005 to +0.01 | The raw-against-coarse knob in another form, which A settles | not taken |
| **I** | A learned switch hazard (an IOHMM: cheaper changes at loose balls, duels and passes in flight) | L | 0 to +0.015 | The best-supported idea in the literature, and the largest transfer risk from DFL to our match | not taken |

**Not recommended**, by both the advisor and Claude:
- **Minimum Bayes risk decoding over sampled paths.** It has the right objective but the wrong posterior: ours is a
  product of grid-chosen exponents, not a calibrated one. A diffuse posterior favours fewer, longer runs, which drifts
  towards the coarse truth, so a gain could not be told apart from a change of grain.
- **A medoid of the plateau's configs.** No diversity, and R023 settles the hedge it would provide.
- **A CRF or structured perceptron.** It trains on a per-frame surrogate of a segment metric, for five weights that the
  grid already fits directly.
- **A deep sequence model.** Seven labelled matches and degraded tracking; P19 and P20 already fell short.

**Order.** A, then B and D, then C, then E. One heavy process at a time. Every Δ is taken against `main.toml` as it
stands when the entry runs. F comes after E, only if at least four days remain before the freeze.

**Sources.**
- Yu, [Hidden semi-Markov models](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf) (2010).
- Lea et al., [Segmental spatiotemporal CNNs](https://arxiv.org/pdf/1602.02995).
- Ding, Sener and Yao, [Temporal action segmentation: a survey](https://arxiv.org/pdf/2210.10352).
- Ishikawa et al., [ASRF](https://arxiv.org/pdf/2007.06866), action boundaries against over-segmentation.
- Kumar and Byrne, [segmental minimum Bayes risk decoding](https://www.isca-archive.org/icslp_2002/kumar02_icslp.html);
  Freitag et al., [sampling for MBR](https://arxiv.org/pdf/2305.09860).
- [hmmTMB](https://arxiv.org/pdf/2211.14139) and
  [LaMa](https://cran.csail.mit.edu/web/packages/LaMa/vignettes/Inhomogeneous_HMMs.html), covariate-dependent transitions.
- [Hybrid HMM/NN scaled likelihoods](https://www.inf.ed.ac.uk/teaching/courses/asr/2024-25/asr10-hybrid_hmm_nn.pdf)
  (Edinburgh ASR notes).
- Link and Hoernig, [individual ball possession in soccer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5503225/);
  Borghesi et al., [TCNs for ball possession](https://iris.polito.it/handle/11583/2976470) (2023).

## From the field (added 4 Nov, A22)

Three ideas other teams use and we do not, read from their public repositories on 4 Nov ([A22](results.md)); each is
lawful because it trains and tunes on external data only, and each gets a declared rule before any run, as every
proposal does.

### P24 — The ball's direction and the touch timing as features of a learned team source: P19's one retry

**Status: done as P32 → [R037](results.md) (13 Nov 2026): P19's one retry ran in this slot, with the ball's direction
added and the gate moved to log-loss (declared); it cleared the bar against `p30_w030`.** Was: **idea.** P19 (a learned team model where the ball is seen) was dropped at its feasibility gate: +0.0160
pooled per-frame accuracy against `main.toml`'s decoded labels, +0.020 needed, 3 of 7 matches against 6 (R021), with
no retry run; its code is recoverable at `15e7e14`. False9 reports the ball's velocity projected on each team's
attacking direction (the direction from where each team's slots stand, per block, as `play_features.own_half_is_left`
already decides it) as its single largest gain, with the time since and until each team's last and next on-ball
frame and the on-ball share over ±0.5, 1, 2 and 5 s. Neither group was in P19's features. **What:** P19's one retry
under the research bar (one retry, then drop): the same model and gate with these two feature groups added, trained
on the seven DFL matches, out of fold; then the grid and R014's bar only if the gate passes. **Why it could work:** a
ball flying towards a team's own goal was played by the other team, which reads possession during flight and right
after kicks, where the nearest player is worst (T14) and where A13 puts the largest remaining headroom (+0.194 raw
where the ball is seen). **Effort** S–M.

### P25 — A tempo check, a normalised speed cue and a sparse-tracking guard for the dead-ball model

**Status: idea.** Two teams report that the players' median speed falls from about 2 m/s in the opening minutes to 1
to 1.7 m/s later in this match, so that a fixed speed cut flags half the match dead, and one of them normalises every
speed by a rolling median over ±7.5 minutes. Our `dead_ball_model` reads absolute speeds (`speed_mean`,
`slow_share` at 2.0 m/s) learned on DFL; its label-free checks on BAR-ATH passed (predicted `D` 36%, R010 to R029),
so the effect, if any, is small, and the public window is the fastest stretch of the match (A02), so it cannot show
it. **What, in order:** (1) a label-free check, allowed as a check and never a value (T01, T54): the distribution of
`speed_mean` per 5-minute block on BAR-ATH beside the degraded DFL matches, and the predicted `D` share per block; a
WARN band declared before the run; (2) only if the check warns, normalised copies of the speed features (each divided
by its block's rolling median; a run-time transform, its window a constant set on DFL) added to `play_features`, the
model retrained on DFL and read on the benchmark under both truths by the usual rule; (3) a guard that returns NaN from
every source on a frame with fewer than n present slots, n set on DFL, for the 750-frame blackout another team reports
near the end of the first graded block (no home tracks). **Effort** S for the check, M for the retrain.

### P26 — Metrica's sample games' tracking as a second benchmark and training set

**On hold since 5 Nov 2026:** the sample data has no licence, so by our reading we may not use it (P16, question 3 in
[rules.md](rules.md)). **Status before: idea.** R023 (P16) read the three sample games' events only. Their tracking is public too (games 1 and 2 as
CSV, game 3 in the EPTS format of our own ATD), and two teams use it: False9 trains its models on the three games with
the tracking roughened (ball gaps to 59% visibility, 46 slots), and InkSight calibrates on games 1 and 2 with the ball
dropped in long blocks. The games carry no ball-status flag (checked 4 Nov), so the labels are R023's event-derived
ones (`D` from the stops, the team from the events). **What:** serve the three games to the benchmark as the DFL
matches are served (`MatchInputs`), degrade them with the `tracking` profile, score `main.toml` under R023's labels
beside the DFL numbers, and run P16's timing part on them (the ball's own track against the BALL OUT and SET PIECE
events: the lag from the ball crossing a line to the stop, and from the set piece's start to the ball moving). **Why:**
it is the one external data in our provider's conventions, so it answers T33 and T40 (the grain and the timing of
our truth) directly, and False9's experience that another provider's data (SkillCorner) hurt says the provider's
convention matters. Nothing of BAR-ATH enters it. **Effort** M (a reader for the two CSV formats and the EPTS game,
R023's labels, the degradation as it stands). A decision rule before the run, as R023 had.

## After the organisers' reply (added 9 Nov)

### P27 — A degradation that measures nothing of our match

**Status: done → [R030](results.md) (10 Nov 2026).** Declared on 9 Nov
([the spec](superpowers/specs/2026-11-09-p27-degradation-design.md)) and run overnight, 9 to 10 Nov. Tian decided the
redo that evening and was away for the night; the spec's open points went to an advisor before the runs (R030, "Who
decided what").

**Why.** The organisers' reply of 5 Nov to question 1 was the
clarification and "anything that is not explicitly allowed is not permitted" ([rules.md](rules.md)). Nothing allows
fitting the benchmark's degradation on our tracking. `gaps` transplants our missing-ball masks byte for byte, and
`tracking` v3 adds rates measured on our tracking (A07 to A09). By our reading, then, both models and every constant
chosen on the benchmark stand on training data that is not "only external data". The other way is to keep them and
disclose this as a risk in the technical document. Tian chose the redo on 9 Nov, with the mean of the two truths
kept as the objective.

**Outcome (R030).**
- *The profile is `generic`, version 1,* now the benchmark's default (`run.DEFAULT_PROFILE`). It is candidate (a)
  below with a generic fragmentation of the players. No measurement of ours enters it, and no `Calibration` is loaded
  for it. It quotes two figures from the organisers' DATA.md (the ball missing on 41% of the graded frames; a mean of
  20.5 tracked objects) and declares round rules for the rest: gap lengths log-uniform between 1 s and 60 s, onsets a
  Poisson process blind to the play, no ball speed, players lost independently on 6.8% of the frames in tracked spans
  log-uniform between 2 s and 300 s, a 7-frame moving average, 46 slots as 24 home and 22 away, and no wrong-team
  slots, NaN-speed runs or low-count frames. *The residual risk:* no text allows quoting the two figures by name
  ([rules.md](rules.md)); Tian decided on 10 Nov to keep them, as public information of the task statement. Candidates (b) and (c) were not taken. SkillCorner's open data (MIT licence) was read once,
  after the law was written, as a check on external data; no number was taken from it. `generic` v1 is final for
  this submission: a version 2 is made only on a rules ground, never after a score.
- *The check of the profile itself* passes on the 21 match-seeds: a pooled mask share of 0.4109 against the declared
  0.41 ± 0.02, and 0.0681 of the on-pitch player-frames lost against 0.0682 ± 0.01.
- *Both models are retrained on it* (`pipeline/models/dead_ball_hgb.json`, `shape_possession_hgb.json`), with the
  settings they had. The dead-ball trees' pooled held-out log-loss is 0.0788, against 0.1524 for the logistic
  reference retrained on `generic` (lower in 7 of 7 matches, so the trees stay). The shape model's is 0.3560 on the
  alive frames with the ball missing, against the stop rule's 0.6831.
- *The re-chosen constants* (the new `main.toml`):

  | part | value | against R029's `main.toml` |
  |---|---|---|
  | `controlled_touch` | 1.5 m / 6.0 m/s / 1 frame, confidence 0.93 | unchanged |
  | its hold | 0.60 | was 0.55 |
  | `dead_ball_model` weight | 1.0 | unchanged |
  | `shape_possession` | weight 0.3, scope `all` | the weight was 0.2 |
  | `hmm` | `switch_team` 6, `switch_dead` 32, `min_possession` 35, `min_dead` 25 | `switch_team` was 4 |
  | `shift_edges` | +1 / −2 / 0 | re-derived, unchanged |

- *On `generic`* (seven DFL matches, three seeds, pooled spellF1) the final config scores **0.5414 raw and 0.5829
  coarse, mean 0.5622**. The incumbent constants on the same profile and the same retrained models score 0.5388 and
  0.5779, mean 0.5584. The yardsticks: `nearest_w50` 0.1885 raw and 0.1698 coarse, `nearest_w125` 0.1616 raw and
  0.2092 coarse. The bar passes, with 7 wins in 7 under each truth. The re-choice is worth a third of a point on its
  own benchmark; what it buys is that every constant now stands on external data only.
- *Two choices were knife-edges; Tian reviewed them on 10 Nov and kept the declared outcome.* The rule declared
  before the runs decided both (within 0.005 of the best mean, the larger min(raw, coarse)). For the review the
  protocol was re-run from the hold 0.55 and the finalists at six seeds, each declared first and choosing nothing:
  the benchmark could not tell the options apart (R030, "After the choice").
  - The hold, 0.60: it is inside the tie band by 0.000004, and on the mean alone 0.55 is 0.0050 better.
  - The decoder: the tie-break chose `switch_team` 6, `min_possession` 35 and the shape weight 0.3 (mean 0.5621, raw
    0.5410) over the best-mean config, `switch_team` 8, `min_possession` 50 and weight 0.3 (mean 0.5653, raw 0.5242).
- *The claim, in its honest wording:* every numeric constant was chosen, and both models were trained, on `generic`.
  The architecture (which components exist, the trees' settings, `min_prob`, the hmm's structure) was inherited from
  the old benchmark and re-verified on `generic`.
- *What the new benchmark still cannot settle:* its gaps ignore the play and ours do not, so it still overstates the
  dead-ball model (T38, T39); it has no wrong-team slots and no long NaN-speed runs (T31, T32); the gap law is one
  declared range; and the two truths still pull the hold and the decoder in opposite directions (T33).
- *Tools, so that the re-choice can be re-run from the repository:* `pipeline/benchmark/p27_grid.py` (the configs of
  each stage, in `pipeline/benchmark/configs/p27/`), `p27_choose.py` (the declared rules), `p27_shift.py` (the shift)
  and `p27_profile_check.py` (the profile against its declared shares).
- *The old profiles* `gaps`, `tracking` and `context` stay in the code for the record of R002 to R029, marked as
  reading our match. Nothing is trained or chosen on them again.
- *Not uploaded.* The checks on our match (pass or fail only) and the regenerated file's sha256 are in R030. The
  upload and making the repository public are Tian's.

**What (the plan, as written on 9 Nov).** A new profile whose every setting comes from a rule declared before any run
and from nothing measured on BAR-ATH, followed by the work that already exists, re-run on it:
1. *The profile.* Research first, then the simplest version that serves; none is chosen yet. The candidates:
   - (a) the ball hidden in blocks by a generic masking rule (block lengths and a missing share from the literature on
     optical ball tracking, or a declared range);
   - (b) a family of such profiles over a wide declared range, with every choice ranked on their mean, so that no
     single setting stands in for our match;
   - (c) the gaps of another provider's tracker on other games (SkillCorner's open broadcast tracking, if its licence
     allows; Metrica's sample tracking is out, P26).

   The players: A14 puts all of `tracking`'s player degradation at 0.013 of `main.toml`'s spellF1. So `clean` players,
   or a generic fragmentation with declared rates, would do.
2. *What we measured on BAR-ATH is not a setting.* A median gap of 10.7 s, the tracklet lengths and the wrong-team
   share may not become the new profile's numbers, or "rules" that land on them. That would be the same fit, made by
   hand. The figures the organisers publish in DATA.md (41% of the graded frames without a ball, a mean of 20.5
   tracked objects) are part of the task statement and may be quoted. `fidelity.py` is not run on the new profile.
3. *Retrain:* `train_dead_ball.py` and `train_shape.py` on the new profile, the folds as before.
4. *Re-choose:* R010's, R011's, R013's and R024's grids by their declared rules (`run.py`, `choose.py`), under the
   objective declared for P27. Without R023 (question 3) that is A06's: both truths, a disagreement a warning, or
   their mean.
5. *Regenerate* `main.csv`, run the selftests and the label-free checks on BAR-ATH (pass or fail only), and update
   the technical document's audit.

**Effort** M–L: steps 3 to 5 are scripts that exist, and step 1 is the new work. One process at a time on this
machine, so the grids take most of a day. The feature freeze is Fri 13 Nov. *As run (R030):* one night; Stage A's 192
configs took 48 minutes and Stage C's 144 took 32.

### P28 — A possession back-dated to the team's first touch

**Status: dropped at its gate → [R031](results.md) (10 Nov 2026).** Declared on 9 Nov, before any run
([the spec](superpowers/specs/2026-11-10-p28-touch-backdating-design.md)), as the next step after P27. Tian was away
for the night; the lane, its gate and its bar were an advisor's rulings, made after a research pass ranked what had
not been tried. The advisor put a pass of the bar at about one in ten.

**The belief.** `controlled_touch` speaks only where the ball moves with the nearest player. A first-time pass, a
header or a deflection that the same team then controls is not an episode, so the decoder's change of team lands on
the control, one pass flight after the truth's flag. This is P11's first bullet, made a rule: a touch is a change of
the ball's velocity of at least τ (3 or 6 m/s) with one player in reach and no rival within the duel radius
(Vidal-Codina et al. 2022's zone test), and a team's episode is back-dated to its first touch after the other team's
last episode.

**The gate, declared first** (on `generic`, raw truth, 21 runs): (i) at least 10% of the ball-visible changes of team
have a new-team touch and no new-team episode within 12 frames; (ii) the truth names the touching team on at least 70%
of the frames the rule would back-date. Both, for one τ.

**The outcome.** (i) 12.7% at τ = 3 and 9.8% at τ = 6; (ii) 33.8% and 46.5%. The gate fails for both, so nothing was
built into the pipeline and no config was run. The touch exists near one flip in eight, but the frames between a
team's first touch and its next episode are the other team's or dead more often than not: the first touch is usually
not the start of the possession. The code (the detector, the off-by-default rule, the diagnostic) is on the branch
`p28-touch-backdating`, not merged.

**What it leaves.** P11's other bullets (pass events as boundaries) rest on the same touch detector and would meet the
same table. The ball-state shift, the lane's fallback, was struck before any run: `generic`'s gaps are blind to the
play, so a lag measured inside them does not transfer.

## From the review of 12 Nov

A code and logic review on 12 Nov found no bug that changes a label (A23 holds) and three places where the evidence
is weaker than it could be. Each became a proposal with a gate and a bar declared before any run
([the spec](superpowers/specs/2026-11-12-p29-p31-design.md), [the plan](superpowers/plans/2026-11-12-p29-p31.md)),
built overnight on the branch `p29-p31` in the order P29, P30, P31. Everything is measured on `generic`, three seeds,
both truths, against `p27_final`; `main.toml` and the file are not touched until Tian decides.

**The bar (P29, P30):** the mean of the two truths' pooled spellF1 at least +0.005, the sign-flip p of that difference
at most 0.05 (seven matches: all seven won, or six with a small loss on the seventh), neither truth more than 0.005
lower. One retry at most, then the lane's code is removed.

### P29 — A learned hold

**Status: dropped 12 Nov 2026 → [R032](results.md): the gate passed (deviation 0.098), the pick `p29_s025` gained
+0.0054 on the mean with both truths up and missed the sign-flip test (p 0.078, six of seven matches); the advisor
found no change the evidence points to, so no retry. Code removed in `57279e9`, recoverable from `3eee56e`. A near
miss: Tian can reopen it as `p29_s025` exactly as run.** Effort S, then M. Odds put at about one in three.

**The belief.** Between two control episodes `controlled_touch` votes 0.60 for the last controlling team on every
frame with the ball visible, however long that lasts: through ball gaps and through stoppages, after which the other
team restarts in 57 to 90% of the cases per DFL match (A12). How likely the last controller still has the ball falls
with the time since their last touch and falls further once the ball has been out of sight. A vote that follows that
curve is stronger where the constant is too timid and silent or reversed where it is wrong.

**Step 0, the gate.** On the hold frames where DFL's truth is alive, the agreement of the truth's team with the last
controller's, in ten bins of time since the last kept frame and by whether the ball was hidden in between: the 20
cells differ from the pooled agreement by at least 0.05 on average, weighted by frames.

**The mechanism.** The cells, smoothed, as `pipeline/models/hold_curve.json` (a curve per held-out match and one from
all seven); `controlled_touch` takes `hold_curve` and `hold_scale`, and the vote for the last controller is
`0.5 + hold_scale * (p - 0.5)`. Candidates: the scale at 0.25, 0.5, 0.75 and 1.0; within 0.005 of the best mean, the
smallest.

### P30 — Timing-aware shape features

**Status: cleared the bar 12 Nov 2026 → [R033](results.md): gate G2 passed (held-out log-loss in the gaps 0.3189
against 0.3560, lower on all seven matches); the pick `p30_w030` (the version 2 model at today's weight 0.3) gained
+0.0143 on the mean of the two truths (raw +0.0215, coarse +0.0072; all seven matches won, p 0.016). It is the
night's candidate, `pipeline/configs/candidates/p30_shape_v2.toml`: not adopted, Tian decides.** Effort M. Odds put
at about one in four.

**The belief.** Clean pairs have a median edge error of 0 but an upper quartile of +14 to +27 frames (A25), 42% of
DFL's changes of team happen with the ball hidden (R031), and P07's time features are centred windows of ±1 s and
±2 s, which see the average of before and after. One-sided windows see the change itself: the team that has just won
the ball starts to advance and to spread.

**The mechanism.** A second feature set beside the first (42 columns): the 26 of today plus, per team, `length`,
`spread` and the advance over the last and the next 2, 5 and 10 s. The model file names its feature version, so
today's file gives today's numbers; the version 2 model is a second file.

**The gate.** The pooled held-out log-loss on alive frames with the ball missing at most 0.3510 (0.3560 today).
Candidates: the new model at weight 0.3, 0.2 and 0.45; within 0.005 of the best mean, the weight nearest 0.3.

### P31 — Team evidence that does not vote on the dead ball

**Status: measured 12 Nov 2026 → [R034](results.md): the guard holds (raw −0.0014, coarse −0.0018; six of seven
matches a little lower under both truths). The flag stays off by default; adopting it is Tian's call.** Effort S. A
matter of principle; expected neutral on the benchmark.

**The belief.** A `home` term goes to `H` and `A` and not to `D`, so a control episode adds +0.62 nats per frame for
play over `D`, the hold +0.18 and P07 up to +0.20 (A19). Who has the ball should not say whether it is alive.

**The mechanism.** A decoder flag `team_neutral_dead`: each `home` term also gives `D` the larger of the two team
terms. One candidate, and a guard instead of a bar: neither truth more than 0.005 lower. The decoder's other constants
were chosen with the push in place and are not re-chosen. Adoption is Tian's call.

## The night's second round (12 to 13 Nov)

With P29 to P31 measured by 23:20, Tian asked for as many more proposals as the night allows. A research agent
proposed six (C0 and P32 to P36); the advisor, in Tian's place, ruled that C0, P32, P33 and P34 run tonight and
amended their gates ([the spec](superpowers/specs/2026-11-13-p32-p34-design.md), declared at 23:50 before any run;
[the plan](superpowers/plans/2026-11-13-p32-p34.md)). **All three are measured against the night's candidate
`p30_w030` (P30), not `main.toml`'s copy**: each replaces or reads the version 2 shape source. If Tian does not adopt
P30, none of them carries to `main.toml` without a run against `p27_final`. The bar is the night's. Not run tonight:
P35 (a turnover hazard as a per-frame switch cost, about 6 h with a decoder change: a lane for another night) and P36
(the decoder re-chosen with version 2 in place: reopens a choice Tian closed on 10 Nov, odds about 1 in 10).

### P32 — The team model where the ball is seen, on `generic`, measured on spellF1 (P19's one retry, in P24's slot)

**Status: cleared the bar 13 Nov 2026 → [R037](results.md): the gate passed (feasibility log-loss 0.2088
against 0.2700, below the rule's on 7 of 7); the pick `p32_w030` gained +0.0162 on the mean against `p30_w030`
(raw +0.0262, coarse +0.0062; all seven matches). The night's candidate, stacked on P30:
`pipeline/configs/candidates/p32_team_model.toml`; not adopted, Tian decides. No transfer check on our match.**
Effort about 3.5 h. Odds about one in three.

**The belief.** Where the ball is seen and no control episode runs, the evidence is the constant hold 0.60 and the
shape model. P19's tree model, which reads the ball against both teams, was better than the touch rule and its hold
in 7 of 7 matches (R021) but was dropped at a per-frame gate against the decoded output of all sources, and no spellF1
was ever measured. P32 restores it from `15e7e14`, retrains it on `generic` with P24's change (the ball's advance and
velocity towards each team's attack; `over11` and the gap timers dropped, the version 2 shape block), and lets the
bar on spellF1 decide; R021's per-frame statistic is printed and gates nothing.

**The gate.** The dry run's rows within ±20% of a declared estimate; then the held-out log-loss on the feasibility
frames at most 0.2700 and below the rule-with-hold's on all seven matches. Candidates: weight 0.075, 0.15, 0.30, 0.60;
ties to the smallest. No transfer check on our match (stated).

### P33 — Shape features version 3

**Status: dropped 13 Nov 2026 → [R035](results.md): the gate passed by 0.0002 (gap log-loss 0.3137),
the benchmark did not move against `p30_w030` (mean +0.0010, p 0.44); no retry (the advisor); code removed.
Version 2 seems to have taken what the teams' shape says in time.** Effort about 2.5 h. Odds about one in four.

**The belief.** P30 showed that one-sided time is where the shape signal is. Version 3 adds the difference of the two
teams' one-sided advance changes, one-sided changes of spread and width over 5 s, and a 20 s horizon (60 columns).
**The gate:** held-out log-loss in the gaps at most 0.3139 (version 2's 0.3189 − 0.005). Candidates: weight 0.3 and
0.45; ties to 0.3.

### P34 — The version 2 shape model trained on both truths' labels

**Status: dropped at step 0, 13 Nov 2026 → [R036](results.md): the sanity gate missed (the mean of the raw
and coarse gap log-losses fell 0.0007 against 0.003; raw +0.0075, coarse −0.0088, a trade along the grain); no
retry (the advisor); code removed, the report's coarse cells kept.** Effort about 2 h. Odds about one in seven.

**The belief.** The objective is the mean of the two truths and the model learns the raw one only; the teams' shape
cannot see a 2-second possession, which the coarse truth merges. Training on both labels (each weighted half where they
differ) trains on the objective's own target. **A sanity gate:** the mean of the raw and coarse held-out log-losses in
the gaps at most version 2's − 0.003, the raw one at most +0.010 worse. One config at weight 0.3.

## The night's third round (13 Nov)

With P32 kept and P33 and P34 dropped by 00:40, a research agent proposed a third round (C1 and P35 to P39); the
advisor, in Tian's place, ruled at 01:05 that all five run and amended them ([the
spec](superpowers/specs/2026-11-13-p35-p39-design.md), declared at 01:07 before any run; [the
plan](superpowers/plans/2026-11-13-p35-p39.md)). **The reference is the night's candidate `p32_w030`** (P30 + P32,
raw 0.5891, coarse 0.5963), not `main.toml`'s copy; the bar is the night's. New for this round: a pick, or the
combination, becomes a candidate only if it also holds on fresh seeds (3 to 5: mean at least +0.005, neither truth
below −0.005). The combination of kept picks must beat the better single pick by +0.005. P36 (the decoder re-chosen
with version 2 in place) stays not recommended.

### C1 — The edge errors by boundary kind and ball state (a check)

**Status: in progress (13 Nov).** A report that chooses nothing: `diagnose.py` splits the clean pairs' start and end
errors by the kind of boundary (a change of team, into `D`, out of `D`) and by whether the ball is seen at the true
edge. It opens P38's gate: the into-`D` end error's upper quartile at least +10 frames (raw truth) and above the team
ends.

### P35 — A turnover hazard: the decoder's switch cost per frame

**Status: in progress (13 Nov).** Effort about 2.5 h (A1 the decoder, A2 the model). Odds about one in five.

**The belief.** `switch_team` is one constant for every frame, so a change of team costs as much in the middle of a
controlled run as at a loose ball after a long pass. A model of where changes of team happen (trained on DFL, the ball
seen) lowers the cost where a change is likely and raises it where it is not. **Amended:** the hazard speaks only where
the ball is seen; the gaps decode as today. **The gate** (a screen): held-out AUC at least 0.80 and log-loss at most
0.90 × the base rate's. Scales 0.5, 1.0, 2.0; ties to the smallest. `HAZARD_FLOOR` 1.0 nat, the cost capped at
2 × `switch_team`. Safeguards on `hmm.py`: off by default it decodes bit for bit as today, checked by brute force.

### P37 — Shape features version 4: each team's time since and until control, in the gaps only

**Status: in progress (13 Nov).** Effort about 1.5 h. Odds about one in six. An **alternative** for Tian at most: no
transfer check is possible (T38), so it never enters the night's candidate.

**The belief.** Version 2 knows the teams' shape but not how long since each last controlled the ball; in a gap that
timing says who is likely to have it. Version 4 adds it, with a scope `"visible"` (R026's, restored) so the seen frames
keep version 2. **Gates:** gap log-loss at most 0.3059, and on the hand-over gaps at most version 2's − 0.013.

### P38 — The dead-ball onset: one-sided speed changes

**Status: waiting on C1's gate (13 Nov).** Effort about 1.5 h. Odds about one in eight, times a half for C1's gate. Built only if the into-`D` end
error is the late tail C1 would show. **Gates:** the onset log-loss at most 0.90 × version 1's, on all frames at most
version 1's + 0.002.

### P39 — The shape vote silent where P32's model speaks

**Status: dropped 13 Nov 2026 → [R038](results.md): the bar missed (mean −0.0084; raw −0.0013, coarse −0.0156 and
lower on all seven matches), the direction R026 predicted; no retry; config and check removed.** A config only.

**The belief.** P32's model reads the 42 shape columns itself, so the shape source counts them twice where the ball is
seen; with the shape source silent there (`scope = "gaps"`), each input votes once. The measurement says the second
vote smooths who has the ball: without it the decoder makes more, shorter possessions, which the coarse truth punishes.

## Cheap signals worth checking

### P09 — Camera aim as a coarse ball-zone prior

**Status: dropped → [R019](results.md) (the simulated camera failed two of the four fidelity gates on `tracking`: peak
correlation and dead/alive speed ratio; Tian chose the drop over a retry, T47; code removed).** The code was removed in
`8c4c959`, where it is recoverable from `13c36c6`.

*A20 (1 Nov): the premise fails too, not only the simulator.* In the ball gaps the tactical camera moves with the
players (r 0.845 between its move and their centroid's). Adding its move to theirs explains no more of where the ball
reappears (leave-one-out median error 5.43 m with the players, 5.50 m with both). The broadcast camera is the one that
carries independent information (P15).

**Why.** The tactical camera pans to follow play on every frame, including the long ball gaps. Measured in A03: the
pitch point under the image centre tracks the ball's along-pitch position with corr 0.84, once the camera is allowed
its ~1.4 s (35-frame) lag. The median error is 12 m, and across the pitch it says nothing (corr 0.07).

**What.** Use these as features:
- the aim's x at t + 35 frames (we process offline, so looking ahead is fine);
- the aim's velocity, since counter-attacks swing the camera;
- the zoom (homography scale) around set pieces.

This gives a zone, not a position: feed it to P07 and P06, never to nearest-player logic.

### P10 — Referee whistle from the audio track

**Status: dropped → [R018](results.md), [R018b](results.md) (whistles near stoppages 1.5× chance, below the declared 2×;
code removed).** Both feasibility gates failed (ratio 1.52, then 1.51); the code was removed in `fc9dbed`, where it is recoverable from `a08e381`.

**Why.** Fouls and offsides stop play without the ball leaving the pitch, and the tracking only sees players slow
down a second later. A whistle marks a sharp `D` onset, and it also starts free kicks and kick-offs.

**What.**
1. ~~Check whether the tactical mp4 has an audio stream at all.~~ It does: AAC stereo 48 kHz, starting at 0.000 s
   like the video ([A04](results.md)).
2. Band-pass around 2–4 kHz and detect sustained tonal peaks.
3. Map audio time to video frames: frame = `round(t * 25)`, with no start offset.
4. Feed the whistle probability to P05 and P06.

The broadcast audio has whistles too, but it is not aligned (P15).

*Added 29 Oct:* instead of a hand-built band-pass, use a pretrained audio tagger for step 2.
- PANNs CNN14 ([code and weights](https://github.com/qiuqiangkong/audioset_tagging_cnn), MIT) gives frame-level scores
  for AudioSet's "Whistle" class, and for "Cheering" and "Crowd". It runs in minutes on the CPU for 2.2 h of audio.
- AudioSet's whistle class is generic, and no one has checked it on football.
- DFL has no audio, so its weight is a declared constant. The label-free checks on our match are the whistle count per
  half and the lag from each whistle to P05's predicted `D` onsets.
- The benchmark's dead-ball headroom is small (+0.029, A13), though T39 says ours may be larger, so this stays low
  priority.

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

*Added 29 Oct:* the DFL open data ships event data, so the touch detector can be measured on the benchmark before it is
used.
- `pipeline/benchmark/dfl_events.py` already reads the restart events.
- PathCRF's repository (P20) has the events synchronised to the tracking with ELASTIC
  ([arXiv 2508.09238](https://arxiv.org/abs/2508.09238)).
- Measure the precision, recall and timing of receptions and interceptions on the degraded copies.
- Start from the rule-based detector of Vidal-Codina et al. 2022 ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)).
- Event timestamps are imprecise, and DFL's conventions are not ours (P16).

### P12 — Metric-aware post-processing

**Status: done → [R012](results.md) (shift adopted), [R016](results.md) (free X measured and dropped).** R016's bar was not met: every threshold of `free_x` lowered macroF1 (raw -0.0003 to -0.0008, coarse -0.0009 to -0.0020), no retry, code removed (recoverable from `c909e2f`).

**Why.** Cheap last-mile gains on the ranking metric.

**What.**
- Drop possession blips shorter than m frames, and `D` blips inside possessions. *Covered elsewhere: blips by the hmm decoder's `min_possession` 35 and `min_dead` 25 (R007, R013).* Validate m on P01's benchmark: the
  public GT's genuine 0.76 s and 1.32 s possessions show that short ones exist, and do not set m.
- Shift transitions by the measured mean latency. *Done as `shift_edges` (P02, R012): minus the pipeline's own median
  lag on DFL, team +1, into `D` -2, out of `D` 0 frames.*
- Write explicit `D`, never `X`, before kick-offs and once the ball is out ([T02](traps.md)). *Covered elsewhere: before kick-offs by the pre-kick-off rule (P(alive) 0.001, R009).*
- Use `X` only inside possessions where the label is genuinely uncertain. It is free on spellF1 and only touches the
  macroF1 tie-break. *Measured as `free_x` (R016) and dropped: the decoder's least confident frames are still right more often than F1/2.*

## Bigger bets

### P13 — Learn from external labelled tracking data

**Why.** The 7,500 publicly labelled frames of this match can train almost nothing, and since 22 and 23 Oct we may
neither add hand labels of our own on it nor train or tune on the published ones ([T27](traps.md)). Public datasets of
other matches carry our kind of labels, in any amount. The benchmark half of this idea moved to
[P01](#p01--a-validation-set-we-can-trust); what stays here is training on them.

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
players plus the ball with a missing mask → TCN/BiGRU → per-frame logits → P06. Calibrate it afterwards on other games
only (held-out DFL matches, P16's sample games). The public window may check that the result is correct and is never a
calibration set, out of fold or not (`f8f2629`, [T27](traps.md)); we have no labelled data of this match to train on.

Hold out P01's benchmark matches from training, or train and score across folds of matches, so that the benchmark
still measures something.

**Rules.** Conventions differ between providers, so use external data to rank methods and pre-train, and calibrate on
other games. Since 3 Nov 2026 that is the only place training and tuning may happen ([rules.md](rules.md)): no
fine-tuning or recalibration on our match, with labels or without. External data must be about other games, and we
must have the rights to use it. Check every licence, keep the sources public (the organisers
must be able to access them), and explain them in the technical document: what was labelled, how, by whom and how
much. **Never use anything about this fixture**: no event feeds, stats or match reports, nothing human-produced about
it from any source ([T26](traps.md)).

### P14 — Recover the ball from the video

**Status: not built → [A14](results.md) (the debate of 29 Oct).**
- It would not be a vision transformer. The ball is about 6 px wide, smaller than one patch ([T46](traps.md)).
- On the benchmark a perfect ball in our gaps is worth +0.10 raw but +0.02 coarse (A14), and a real detector would
  deliver a fraction of that.
- Both sides of the debate declared their rule before that result, and both rules say no full build.
- A half-day zero-shot test (test B, below) may still run: [After A14](#after-a14), item 1.

**Why.** The tracker's ball gaps are long and pile up late in each half (A02), and a dedicated detector may see what the
tracker lost. A14 adds that the ball is lost inside the image: the median onset is 285 px from the frame's edge, none is
within 40 px of it, and the camera keeps play in view through the gaps. So the ball was in view when the track was lost.
Either the tracker lost its lock or the ball was occluded, and only a detector run on the gaps can tell which.

*A20 (1 Nov) lowers the prior.* The broadcast's own tracker sees the ball from a second angle, at about 10 px wide, yet
has it on only 28% of the live frames where ours is missing (and on 94% where ours is present). The moments we lose
seem hard to see from both cameras, so A14's assumption of a detector reaching 35 to 50% of the gap frames looks
optimistic. Test B's figures would still settle it.

**What, if it is ever built.** This is the design that both sides of the debate accepted.
- **Model:** a stride-4 heatmap network at full resolution, with three stacked frames for motion. Two candidates, both
  MIT-licensed:
  - FootAndBall (Komorowski et al. 2020, [code and weights](https://github.com/jac99/FootAndBall), trained on ISSIA-CNR),
    with its first layer widened to take three frames;
  - WASB-SBDT (Tarashima et al., BMVC 2023, [code](https://github.com/nttcom/WASB-SBDT)), without its downscaling.
- **Rejected models:** DINOv2, RF-DETR, OWLv2 and Grounding DINO, because of the ball's size (T46); Ultralytics YOLO,
  because of its AGPL licence and its stride of 8; SAM 2 and CoTracker, for drift and licence.
- **Training data:** ~~the ATD's own ball, projected through the homography to pixels, as pseudo-labels made by an
  algorithm~~. *Not allowed since 3 Nov 2026* ([rules.md](rules.md), [T54](traps.md)): no "training, fitting or
  calibrating your algorithm ... on the provided video", and training uses "only external data". That removes this
  plan's training set, its oversampling of our gap edges and its round of self-training. What stays allowed: the stock
  weights, or weights trained or fine-tuned on other matches' video (hand labels of other matches included, licence
  checked), run on our video as an input.
  - The public window is left out of every test.
  - Boxes drawn on this match are not allowed ([T27](traps.md)).
  - A person may watch gap moments to understand what counts as possession there, but never to judge the detector's
    output against the video. A tally of what they saw is not a gate.
- **Training compute:** about 12k crops of 256 px, which is 10 to 25 h on the laptop's CPU: one or two throttled
  nights.
- **Inference:** on the CPU, at 5 fps inside the graded gaps (about 16k frames), in the zone given by the camera's aim
  (A03; P09, which computed it, was dropped, R019, and its code is recoverable from 13c36c6) and the players.
  - Link the detections into tracklets with a motion gate, and keep those of at least 0.4 s whose speeds are
    plausible.
  - Project them to the pitch. An airborne ball projects to the wrong ground point, so give it low confidence.
  - Fill the ATD's ball with them, flagged as coming from the video, so that `controlled_touch` and `dead_ball_model`
    read it unchanged.
- **Reproduction:**
  - Decode to YUV planes with a pinned PyAV. H.264 decoding is bit-exact; swscale is not ([T23](traps.md)).
  - Commit the weights (about 1 MB) with a `--verify` path, as `train_shape.py` has.
  - Keep the detections in `data/cache/`. They are frame-level data and are never committed.
  - Run ONNX Runtime on one thread.
  - Hash the CSV on an AMD and an Intel machine as soon as the first model exists. Snapping the outputs to a grid
    makes a threshold flip rarer, but does not prevent one.
  - Keep the video on our own machines.

**The gates, declared on 29 Oct before any of them ran.**
- **Test A (done, A14): a perfect ball in our gaps on the benchmark.**
  - Drop if Δraw < +0.04, or Δcoarse ≤ +0.01, or raw is high while coarse is below +0.02 (the advocate's rule).
  - Go at Δraw ≥ +0.07 with Δcoarse ≥ +0.03 (the critic's rule) or at +0.08 and +0.04 (the advocate's).
  - Result: +0.1016 and +0.0197, so not go.
- **Test B: zero-shot, half a day, one person, CPU only.** Run the stock weights at full resolution, and 1.5 times
  upscaled, on seeded samples: about 1,500 tracked frames, the last 2 s before each gap, and about 1,500 gap-interior
  frames that `main.toml` predicts alive. Measure:
  - R_t: the share of tracked frames with a detection within 1 m of the ATD ball;
  - R_g: the share of gap-interior frames with a plausible detection (on the pitch, consistent over ±2 frames, below
    35 m/s, not at a fixed spot such as the penalty spot);
  - a false-positive proxy: detections near players that never link to the ATD ball at either edge of the gap.

  Kill if R_g < 0.25 R_t or the proxy exceeds 30% of R_g. Build only if R_g ≥ 0.5 R_t and the proxy is under 15% of
  R_g. In between, drop. A model fine-tuned on easy frames will not make an occluded ball visible.
  Test B is an automatic check of a model trained elsewhere, run on our video and scored against the ATD's ball, so the
  3 Nov rules allow it. Its numbers may stop or start a build. They may not tune the detector (a threshold, a zone) on
  our match.
- **A build would add a simulation on DFL first.** Give back a share c of the gaps with position noise, and inject
  false balls near players at f ∈ {0.02, 0.05, 0.1} of the gap frames, then apply the kill rule at the worst f. Ship
  only if the gain is at least +0.015 raw with its 90% interval above 0, with the coarse truth not below -0.005. One retry
  at most, then drop and remove the code.

**What it cannot settle.**
- DFL has no video, so the detector itself cannot be scored on the benchmark.
- How well it finds the ball deep inside a gap cannot be observed on our match.
- The frames that would matter most, slow balls at players' feet where the gaps start, are the most occluded.

### P15 — Broadcast tracking as a second opinion

**Status: measured → [A20](results.md); use 1 tried as [R028](results.md) and dropped on 2 Nov 2026, at its fidelity
gate.** The design is `docs/superpowers/specs/2026-11-01-r028-broadcast-ball-design.md` and the plan
`docs/superpowers/plans/2026-11-01-r028-broadcast-ball.md`.
- **Fidelity failed.** The simulated broadcast on DFL, a transplant of BAR-ATH's real fill, did not reproduce our fill
  rates: alive 0.218 against 0.295, `D` 0.089 against 0.041.
- **The code was right.** The advisor found no departure from part 1, and an identity world gives back our statistics
  exactly.
- **The cause is the two worlds' gaps.** `tracking`'s gaps fall blind to the play (T38): 54% of their frames are alive
  against our 20%.
- **What R028 leaves.** The broadcast's value in our gaps is untested, not refuted; another validation would be a new
  proposal (T53). Its code is in `659a6c2` (local tag `archive/r028-source`).
- **Uses 2 and 3 remain ideas.** *3 Nov 2026 ([rules.md](rules.md)):* the broadcast is allowed as an input. If a use
  returns, find the alignment by code at run time, the way A20 found it, rather than keeping the two offsets as
  constants measured on our match. Calibrate a use on external data only. And watch the broadcast only to understand
  what counts as possession, never to check our output.
- The feeds align by two constants.
- The broadcast's ball fills 30% of the gap frames we call alive, in fragments.
- Its shots off the pitch cover 75% of our `D`.

**Why.** The broadcast has its own ball track (50.5% of its frames) from a different, often zoomed, angle. And TV cuts
to replays and close-ups during stoppages, so its 122 invalid-homography stretches are themselves a dead-ball cue.
Unlike the tactical camera, a person steers it: in the gaps the tactical camera only moves with the players (A20, P09).

**Measured (A20, 1 Nov).**
- **The alignment is two constants, found by code from the two ATDs' balls.** Tactical = broadcast + 4,014 frames in the
  first half and + 26,920 in the second (the broadcast leaves out halftime), steady to ±1 frame. The players confirm it
  on 91.6% of the valid frames; the rest are replays and partial views. No time warp per shot is needed.
- **Its ball.** On live frames it has the ball on 28% of those where ours is missing. That fills:
  - 29.6% of the gap frames our decoder calls alive;
  - 4.2% of those it calls `D`;
  - in fragments, median 0.6 s.

  Where both feeds have the ball, the two are a median 0.77 m apart.
- **Its camera** follows the ball at lag 0 (r 0.905, median error 8.0 m), more closely than the tactical camera (0.843 at
  1.4 s, 11.65 m). It is live on 33% of our gap frames.
- **Its shots off the pitch.**
  - 74.9% of our `D` frames are not live on the broadcast, against 16.1% of our `H`/`A`.
  - The director cuts away from stoppages in open play as often as from those at the lines.
  - 22 cut-aways (198 s) sit inside our possessions, mostly in midfield, where play slows. They are either missed fouls
    or replays over slow play, and A20 cannot tell which.

**What, revised.** Every use is MATCH_ONLY (DFL has no broadcast), so each needs its own declared test:
1. **The broadcast ball where ours is missing.**
   - Fused in `inputs` and flagged, so that `controlled_touch` and `dead_ball_model` read it unchanged; live-consistent
     frames only.
   - Test: a simulation on DFL that gives back the ball in A20's shares of alive and dead gap frames, in A20's burst
     lengths and with A20's position error.
   - Fidelity gates on those three, then R024's bar.
2. **Not live (invalid, or a replay) as `alive` evidence.**
   - A soft prior over seconds, not an edge. The literature puts broadcast play/break at 83 to 99% per segment or shot
     (Xie et al., ICASSP 2002 and PRL 2004; Fani et al., IEEE Access 2017), and video spotting of "ball out of play" at
     5 to 60 s tolerance (Cioppa et al. 2021).
   - Calibration: SoccerNet-v2's camera-shot and action labels of other games
     ([arXiv 2011.13367](https://arxiv.org/abs/2011.13367); camera classes on 200 games). The label files download
     without the video NDA, but their licence is not stated, so check it first.
   - Otherwise R017's route: agreement with our own pseudo-labels, and a weight that is Tian's pick and untested. R027
     showed that such an agreement can fail to transfer.
3. **The aim as a ball zone in the gaps** (P07 v2).
   - Needs a simulated aim on DFL and a retrained P07, with R019's fidelity risk. After the deadline at the earliest.
   - SkillCorner's open broadcast tracking may carry the camera's footprint next to possession labels (unverified),
     which would test it on other games.

Not needed: clock OCR or audio to align the feeds, or a time warp per shot.
