# Results

Everything we run that is worth remembering: experiments *and* analyses, including the ones that failed. A number
without the commit and command that produced it is not a result.

Public-window scores alone decide nothing ([T01](traps.md)). Run `--bootstrap 1000` before calling a public-window gap
real, and add the [P01](proposals.md#p01--a-validation-set-we-can-trust) validation score: the benchmark's
uncertainty is per-match t-intervals and an exact sign-flip test, never a bootstrap
([benchmark README](../pipeline/benchmark/README.md)). Record aggregates only — never raw data rows
([CLAUDE.md](../CLAUDE.md)).

## Scoreboard

Our runs, best first. *Public* means the 5-minute public GT (17 possessions, mean 14.2 s); *val* means the P01
benchmark: other matches (DFL open data), degraded to look like ours. Hand labels on this match are not allowed
([T27](traps.md)).

| ID | date | what | public macroF1 | public spellF1 | poss / dur | val spellF1 | verdict |
|---|---|---|---|---|---|---|---|
| R001 | 2026-10-22 | `main`: pipeline skeleton, i.e. R000c per half | 0.6207 | 0.3256 | 26 / 10.8 s | 0.2018 raw, 0.1842 coarse (`main` is the `nearest_w50` yardstick, on `tracking` v3, R004c) | keep as the base; the floor, not a candidate |
| R002 | 2026-10-23 | the four yardsticks `nearest_w{0,25,50,125}` on the benchmark's `clean` profile | — | — | — | 0.3073 raw (`w50`), 0.3257 coarse (`w125`); clean, so not our match's | the benchmark's floor, not a candidate; the truth reorders them |
| R003 | 2026-10-23 | the same four yardsticks on the benchmark's `gaps` profile: our own ball gaps transplanted onto the DFL matches | — | — | — | 0.2102 raw (`w50`), 0.2302 coarse (`w125`); `w0` last under both | `gaps` v1 frozen, the first benchmark that looks like our match; the yardstick check holds (`w0` last), the rest of the order follows the truth |
| R004 | 2026-10-23 | the same four yardsticks on the benchmark's `tracking` profile: our players' tracklets and slots, wrong-team slots, NaN-speed runs and smoothing on top of `gaps`' ball | — | — | — | 0.2000 raw (`w50`), 0.2195 coarse (`w125`); `w0` last under both | `tracking` v1 frozen as it is: the five counted gates pass and `fidelity.py` exits 0; its three count gates stay pending because they depend on the low-count tail, which v2 adds (exactly 22 present: 0.5631 of the full-strength frames against 0.5019 +-0.06; about 0.503 after the tail, an estimate that assumes its cap is placed by our counts), no WARN; the yardstick check holds, no order changes from `gaps` |
| R004b | 2026-10-23 | the same four yardsticks on `tracking` version 2: R004's players plus our low-count frames (a cap that keeps the players nearest the ball), no piece shorter than 8 frames and the walk's tracked spans rescaled once | — | — | — | 0.1989 raw (`w50`), 0.2173 coarse (`w125`); `w0` last under both | `tracking` v2 frozen, `fidelity.py` exits 0: all eight gates pass (the three on how many slots are present are gates now: exactly 22 present 0.4938 against 0.5019 +-0.06, mean 20.60 against 20.52 +-0.5, 14 or fewer 0.0469 against 0.0525 +-0.02), no WARN; the yardstick check holds and the ranking does not move from R004 (the four configs move by 0.4 points of spellF1 at most) |
| R004c | 2026-10-23 | the same four yardsticks on `tracking` version 3: R004b's players, but a slot is taken by a new tracklet only from the row after its last one stopped (versions 1 and 2 let it take the slot on the row it was freed), and the walk's tracked spans rescaled again, 2.042 to 2.072 | — | — | — | 0.2018 raw (`w50`), 0.2226 coarse (`w125`); `w0` last under both | `tracking` v3 frozen, `fidelity.py` exits 0: all eight gates pass (exactly 22 present 0.4925 against 0.5019 +-0.06, mean 20.60 against 20.52 +-0.5, 14 or fewer 0.0469 against 0.0525 +-0.02, tracklet median 26.52 s against 41.28 s, -36%), no WARN; the yardstick check holds and the ranking does not move from R004b (no config moves by more than 0.6 points of spellF1); the jitter per slot after smoothing is 1.80 cm and not R004b's 8.5 cm: that was the slot handovers and not DFL's jumps (T35) |
| R005 | 2026-10-23 | the same four yardsticks on the `context` profile: `tracking` v3's players, and the ball's holes placed by a hazard fitted on our own match (A08: the line, the ball's speed, the players near it, the minute) instead of transplanted from our masks with their pattern and not their clock | — | — | — | 0.2193 raw (`w50`), 0.2443 coarse (`w125`); `w0` last under both | **`context` v1 fails two gates**, so `DEFAULT_PROFILE` stays `tracking`: the gaps that start within 2 m of a line 0.5586 against our 0.2797 +-0.05 (on each seed: DFL tracks the ball out of play, so a quarter of its at-risk rows are near a line against 9% of ours) and the share missing 0.4313 against 0.4110 +-0.02 (pooled, by 0.0003, and on seed 2), and one WARN (the ball visible on dead frames 0.4622 against the public window's 0.6637, which is not the match's; `tracking` v3 0.5684); no fix, no second rescale; the order of the four configs is `tracking` v3's under both truths (Kendall tau +1.000 between the two profiles' orders on pooled spellF1, +0.667 against the published order under each), and every config scores 0.3 to 2.2 points higher; sensitivity only; fails two of its gates |
| R006 | 2026-10-25 | the `hmm` decoder's first grid on `tracking` v3 (P06): switch_dead d in {4, 8, 16, 32, 64, 128} x switch_team t in {d/4, d/2, d, 2d}, 24 configs with both minimum runs at 1; `nearest_player` at confidence 0.78 and `ball_presence` at A01's values, both fixed | — | — | — | 0.1952 raw and 0.1700 coarse for the chosen `p06s1_t016_d064` (the best of the grid: 0.1959 raw, `t016_d032`, and 0.2228 coarse, `t064_d064`), against `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse | **Stage 1 chooses (switch_team, switch_dead) = (16, 64)** by the pre-declared rule (4 configs within 0.005 of the best raw score, all at t = 16, and the best coarse score among them is d = 64); **no config clears the bar under either truth** (the best raw pooled Δ against `nearest_w50` is -0.0059, p 0.1719; the best coarse pooled Δ against `nearest_w125` is +0.0002, p 1.0000); alive/dead F1 is 0.4916 to 0.5761 for all 24, below both yardsticks (0.6094, 0.6428); the truths disagree on t (raw 16, coarse 64); Stage 2 (R007) adds the minimum run lengths |
| R007 | 2026-10-25 | the `hmm` decoder's second grid on `tracking` v3 (P06): the minimum run lengths (`min_possession` 1 to 150 frames, `min_dead` 1 and 25) around Stage 1's choice, a one-step edge extension (Stage 2b), and the choice over the three stages; the chosen `pipeline/configs/p06_hmm.toml` is not the submission config | 0.6393 | 0.3500 | 23 / 11.4 s | 0.2053 raw and 0.1967 coarse for the chosen `p06s2b_t008_d032_mp075_md001` (the best of 123 configs: 0.2083 raw, `p06s2b_t004_d016_mp075_md001`, and 0.2310 coarse, `p06s2b_t008_d032_mp150_md001`), against `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse | **(switch_team, switch_dead, min_possession, min_dead) = (8, 32, 75, 1)** by the pre-declared rule over the set Stage 2b's ruling extended (6 configs within 0.005 of the best raw score, and the best coarse score among them is 8 / 32 / 75, by 0.0007; over Stage 1 and Stage 2 alone the rule picks `p06s2_t016_d064_mp050_md025`, 0.0012 under `nearest_w50` raw); **the bar is not met under either truth** (the chosen config's raw pooled Δ against `nearest_w50` is +0.0036, p 0.2500; its coarse pooled Δ against `nearest_w125` is -0.0259, p 0.0156) and no config of the three stages meets it; the truths disagree on the minimum run (raw 75 frames, coarse 150 or more, the grid's upper edge); the winner is on an edge of Stage 2b's grid in t and d and is not extended; alive/dead F1 0.5676, below both yardsticks (0.6094, 0.6428); the public window is a correctness check only (valid, deterministic; in-sample for the three evidence values A01 measured on it); controls (diagnostics, none a candidate): `seen` off costs 0.0063 raw on `tracking` (0 wins in 7) and moves nothing coarse; `margin` 50 is above the chosen config in all four profile and truth cells (two of them on `context`, sensitivity only; +0.0023 to +0.0071) and clears the raw half of the bar on `tracking` (+0.0107 over `nearest_w50`, p 0.0312, 6 wins in 7) but not the coarse half (-0.0236 against `nearest_w125`), so a `margin` sweep by its own protocol is the follow-up; `switch_dead` on `context` (sensitivity only) is flat under the raw truth (a span of 0.0035 over 8 to 128) and the coarse truth wants more; the headroom is A10 |
| R008 | 2026-10-25 | P04's controlled-touch evidence on `tracking` v3: the 12 rule settings (radius, `max_rel_speed`, `min_frames`, each `confidence` measured on all seven matches) x `hold` off or 0.55, 24 configs with `ball_presence` and the `hmm` (8, 32, 75, 1); the chosen `p45a_r20_v6_k03_h55` (not the submission config) | 0.6532 | 0.3889 | 19 / 13.9 s | 0.2345 raw and 0.2433 coarse for the chosen `p45a_r20_v6_k03_h55` (the best raw is `r15_v6_k03_h55` 0.2348; the best coarse is the chosen one), against `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse, and R007's 0.2053 raw and 0.1967 coarse at the same decoder | **Stage A's choice is radius 2.0 m, `max_rel_speed` 6.0 m/s, `min_frames` 3, `hold` 0.55** (2 configs within 0.005 of the best raw, the best coarse among them is the chosen one by 0.0023); alone it clears both halves of P06's bar (raw Δ +0.0328 over `nearest_w50`, p 0.0156, 7 wins in 7; coarse Δ +0.0206 over `nearest_w125`, p 0.0156, 7 wins in 7) and the leave-one-match-out estimates agree (raw objective alone +0.0316 raw and +0.0176 coarse, the whole rule +0.0334 and +0.0209, 7 wins in 7 each); the winner is on the grid's edge in radius, speed, minimum frames and hold and nothing is added; with the hold off no config beats either yardstick; alive/dead F1 0.5731, no better than R007's 0.5676 by more than 0.006; the public window is a correctness check only; controls (diagnostics): adding `nearest_player` 0.78 costs 0.0161 raw and 0.0352 coarse (`p45actl_nearest`, 0.2184 and 0.2081), and on `context` (sensitivity only) the chosen config is +0.0568 raw over `nearest_w50` and +0.0534 coarse over `nearest_w125`, 7 wins in 7 each, so the four profile and truth cells agree; the combined bar is Stage C's |
| R009 | 2026-10-25 | P05's dead-ball evidence on `tracking` v3: source (`rules`, `logit`, `hgb`) x weight (0.1 to 1.0) x `ball_presence` off or on, 24 configs with `nearest_player` 0.78 and the `hmm` (8, 32, 75, 1); the chosen `p45b_hgb_w100_bp0` (not the submission config) | 0.6482 | 0.3024 | 24 / 11.8 s | 0.3503 raw and 0.4150 coarse for the chosen `p45b_hgb_w100_bp0` (the best raw is `hgb_w100_bp1` 0.3529; the best coarse is the chosen one), against `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse, and R007's 0.2053 raw and 0.1967 coarse at the same decoder | **Stage B's choice is the boosted trees at weight 1.0 without `ball_presence`** (2 configs within 0.005 of the best raw, the best coarse among them is the chosen one by 0.0029); alone it clears both halves of P06's bar (raw Δ +0.1486 over `nearest_w50`, p 0.0156, 7 wins in 7; coarse Δ +0.1924 over `nearest_w125`, p 0.0156, 7 wins in 7) and the leave-one-match-out estimates agree (+0.1498 and +0.1926 for the whole rule, 7 wins in 7 each); Stage B2 does not run (`hgb` leads `logit` by 0.0274 raw, and has the lower out-of-fold log-loss on 7 of 7 matches); alive/dead F1 0.9725 against 0.6094 and 0.6428; **the benchmark overstates P05 for BAR-ATH** (DFL tracks the ball out of play; our tracker loses the ball near the lines: beyond a line on 4.07% of our at-risk rows against DFL's 17.68%), by an amount it cannot measure, and the public window's alive agreement is 86.6% of 7,500 frames; the winner is on the grid's upper edge in weight and nothing is added; no evidence that `ball_presence` helps (costs up to 0.0917 raw below weight 1.0); the `context` control (sensitivity only) is +0.1359 raw and +0.1695 coarse over the yardsticks, 7 wins in 7 each, so the four profile and truth cells agree; the combined bar is Stage C's |
| R010 | 2026-10-25 | P04 and P05 together on `tracking` v3: `controlled_touch` (R008) for `home` and the `hgb` `dead_ball_model` (R009) for `alive`, the `hmm` re-tuned over switch_team {4, 8, 16} x switch_dead {16, 32, 64} x `min_possession` {25, 50, 75, 100} x `min_dead` {1, 25}, 72 configs; the chosen `p45c_t004_d032_mp025_md025` is `pipeline/configs/p45_hmm.toml`, and was `main.toml` from R010 to R011 | 0.6996 | 0.4686 | 18 / 15.7 s | 0.4217 raw and 0.4556 coarse for the chosen `p45c_t004_d032_mp025_md025` (the best raw; the best coarse is `p45c_t016_d032_mp100_md025` 0.5049, raw 0.3532), against `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse, and R009's 0.3503 raw and 0.4150 coarse (P05 alone, R007's decoder) | **Chosen (switch_team, switch_dead, min_possession, min_dead) = (4, 32, 25, 25)** by the pre-declared rule (6 configs within 0.005 of the best raw score, all at switch_team 4 and min_possession 25, and the best coarse among them is the chosen one by 0.0002); **the bar is met under both truths** (raw Δ +0.2199 over `nearest_w50`, p 0.0156, 7 wins in 7; coarse Δ +0.2330 over `nearest_w125`, p 0.0156, 7 wins in 7) and the leave-one-match-out estimates agree (+0.2217 and +0.2341, 7 wins in 7 each); at the chosen decoder each source is needed (alone, at R007's decoder, each clears both halves; without `controlled_touch` the coarse Δ over `nearest_w125` is +0.0198, p 0.1406; without `dead_ball_model` it is -0.0056, p 0.1094; raw both hold); the truths disagree on the decoder (coarse prefers switch_team 16 and min_possession 100, T33) and the winner is on the grid's lower edge in switch_team and min_possession, nothing added; alive/dead F1 0.9726; the `context` control (sensitivity only) is +0.2130 raw and +0.2211 coarse over the yardsticks, 7 wins in 7 each; **the benchmark overstates P05 for BAR-ATH** (T38, T39), by an amount it cannot measure; the label-free check on our match passes (predicted `D` 36.0% of the graded frames, no `X` in a graded block, the same sha256 twice), so **`main.toml` is switched** (sha256 `a706e7f7...`, not uploaded); the public window is a correctness check only |
| R011 | 2026-10-25 | P07's shape-possession evidence on `tracking` v3: the held-out shape model (7 DFL matches, ball hidden, part 1) and a 12-config grid, `shape_possession` weight {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} x scope {`gaps`, `all`} on top of R010's `main.toml`, the `hmm` (4, 32, 25, 25) unchanged; the chosen `p07_w200_all` is `pipeline/configs/p07_shape.toml`, and was `main.toml` from R011 to R012 | 0.7151 | 0.4000 | 20 / 14.1 s | 0.5159 raw and 0.5591 coarse for the chosen `p07_w200_all` (the best of the 12 under both truths), against R010's `main` 0.4217 raw and 0.4556 coarse (the same two sources and decoder without P07), and `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse | **Chosen weight 0.2, scope `all`** by the pre-declared rule (the only config within 0.005 of the best raw score; the next is `p07_w100_all` 0.5057); **the bar is met** (`tracking` raw Δ +0.0942 over `main`, p 0.0156, 7 wins in 7) and the `context` veto does not fire (raw Δ +0.0831, 7 wins in 7); the coarse cells agree (+0.1035 on `tracking`, +0.0890 on `context`, 7 wins in 7 each), the leave-one-match-out estimate picks the same config in all seven folds (+0.0940 raw, 7 wins in 7), and all 12 configs beat `main` in all four cells (the smallest Δ +0.0290); alive/dead F1 0.9729; **the label-free checks on BAR-ATH pass** (`touch_agreement` with the `full` model 0.8251 on 41,052 frames against the gate's 0.7486, predicted `D` 35.8% against R010's 36.0%, 439 possessions = 4.1452 per graded minute against R010's 4.2491, no `X` in a graded block, the same sha256 twice), so **`main.toml` is switched** (sha256 `8810afce...`, not uploaded); the labels differ from R010's on 5.30% of the graded frames; the shape model was trained on other teams' matches, so the benchmark's gain is not a forecast for BAR-ATH, and the benchmark's count of possessions moves the other way (up 16%, BAR-ATH's down 2.4%); the public window is a correctness check only (spellF1 0.4000 against R010's 0.4686: 20 possessions against 18, one possession moves it about 0.06, T01; macroF1 0.7151 against 0.6996) |
| R012 | 2026-10-25 | P02's edge shift on `tracking` v3: the pipeline's own lag on DFL (`conventions.py latency`, raw truth, 7 matches x 3 seeds), guarded by one config (`p02_sb`: R011's `main.toml` plus `shift_edges` team +1, into_dead -2, out_of_dead 0) in four cells, then the gap to our truth's convention read from the public window by the declared rule (g); the total is `pipeline/configs/p02_shift.toml` and now `main.toml` | 0.7159 in sample, 0.7125 out of fold | 0.4000 in sample and out of fold | 20 / 14.1 s | 0.5164 raw and 0.5592 coarse for `p02_sb`, against R011's `main` 0.5159 raw and 0.5591 coarse (pooled Δ +0.0005 raw, p 0.5000, 4 wins in 7; +0.0000 coarse); on `context` (sensitivity only) +0.0009 raw and +0.0001 coarse | **s_b is kept** by the guard (`tracking` raw Δ +0.0005 > 0, no cell below -0.005) and **g is 0 for every type** (1 to 4 matched public changes each, no type passes all three conditions), so the total is **team +1, into `D` -2, out of `D` 0 frames** and **`main.toml` is switched** (sha256 `3fd06da0...`, not uploaded); the shift is 1 to 2 frames and every benchmark gain is inside noise (all four intervals include 0, +0.0003 to +0.0004 macroF1 is the only consistent effect): adopted because the declared rule says so, not because it is shown to help; the labels differ from R011's on 0.34% of the graded frames, 439 possessions as before; the public window is a correctness check only (spellF1 0.4000 as R011's) |
| R013 | 2026-10-26 | the touch rule and the decoder re-chosen without the window's values (P04, P06, P07): Stage A, `controlled_touch` radius x speed x frames x hold, 192 configs on R012's `main.toml`; Stage C, switch_team x min_possession x P07's weight, 144 configs at Stage A's choice; the shift re-derived by R012's rule; the final config is `pipeline/configs/r013_retune.toml` and now `main.toml` | 0.7152 | 0.4649 | 20 / 14.1 s | 0.5430 raw and 0.5593 coarse for the final config (radius 1.5 m, 6 m/s, 1 frame, hold 0.60; hmm 4/32/35/25; P07 at 0.2; shift +1/-2/0), against R012's `main` 0.5164 raw and 0.5592 coarse; on `context` (sensitivity only) 0.5489 and 0.5642 against 0.5162 and 0.5544 | **adopted by the declared rule**: the guard passes (`tracking` raw +0.0267, p 0.0156, 7 wins in 7; `tracking` coarse +0.0001; `context` +0.0327 raw, +0.0097 coarse) and the bar is met; Stage A alone was +0.0310 raw and -0.0107 coarse (T33), and Stage C's longer minimum possession (35 frames) won the coarse cell back; leave-one-match-out +0.0306 (Stage A) and +0.0263 (Stage C) raw, 7 wins in 7; the touch rule is on three edges of its grid and covers relatively fewer ball-visible frames on BAR-ATH than on DFL (36.9% against R008's rule's 43.6%); s_b re-derived as R012's; label-free checks pass (`D` 35.9%, no `X`, the same sha256 twice, 410 possessions against 439); `main.toml` writes sha256 `0ebdfbec...` (not uploaded); the public window is a correctness check only (spellF1 0.4649, 20 possessions against R012's 0.4000, 20) |
| R014 | 2026-10-26 | the restart vote (P17) on `main.toml`: `restart_team` over tail {25, 50, 100, 150} x `use_ball` {false, true}, 8 configs at their measured confidences | — | — | — | 0.5470 raw and 0.5564 coarse for `r014_t025_b0`, against `main.toml`'s 0.5430 and 0.5593 | bar not met (+0.0040 raw, p 0.0938); not adopted; Phase 2b followed |
| R014b | 2026-10-26 | restart states in the decoder (P17, Phase 2b): `restart_team` span `stoppage_end` with `hmm` `restart_source`, `switch_restart` {4, 8, 16, inf} x `use_ball` {false, true}, 8 configs | — | — | — | 0.5433 raw and 0.5589 coarse for `r014b_s04_b1`, against `main.toml`'s 0.5430 and 0.5593 | bar not met (+0.0003 raw, p 0.8125); not adopted; P17 dropped |
| R015 | 2026-10-29 | the `offpitch` profile (P18): `tracking` v3 with DFL's ball hidden once it is more than m past a line and a share q of the transplanted gaps dropped, m and q fitted once on label-free statistics of our match, its fidelity, and `main.toml` on it | — | — | — | 0.5509 raw and 0.5296 coarse for `main.toml` on `offpitch`, against 0.5430 and 0.5593 on `tracking` (a difference between two profiles) | `offpitch` v1 frozen (m 0.2 m, q 0.430); 8 of 9 gates pass, median gap fails (15.04 s against 10.74); main.toml 0.5509 raw / 0.5296 coarse on it; P18 dropped after R015b |
| R015b | 2026-10-29 | the `offpitch` v2 rule (the ball hidden while it is more than m past a line and moving, found again at rest), swept once on the benchmark's label-free fidelity statistics before anything was built | — | — | — | — | v2 cannot reach our share past a line at any margin (0.258 to 0.262 against 0.0407); P18 dropped, offpitch code removed (7b555eb) |
| R016 | 2026-10-30 | `free_x` (P12) on `main.toml`: `X` inside `H` and `A` runs where the decoded label's posterior is below a threshold, over 0.25, 0.35, 0.45, 0.55, 4 configs | — | — | — | — | bar not met; free_x dropped |
| R017 | 2026-10-30 | Smart Tagging's votes (P08) against our own pseudo-labels on BAR-ATH: team agreement with `controlled_touch` episodes and dead agreement with `dead_ball_model`, the two parts' bars, and two strengths (`frame` 0.2, `tag` 1.0) for Tian's pick | 0.6986 | 0.4000 | — | — (DFL has no tags; `main.toml --drop-match-only` reproduces R013's 0.5430 raw, 0.5593 coarse) | team votes adopted per frame (0.761); dead votes dropped (0.409) |
| R018 | 2026-10-30 | the referee's whistle (P10): a hand-built 2 to 4 kHz band-pass detector on the tactical audio, its feasibility gate and its bar against `main.toml`'s own decoded stoppages | — | — | — | — | dropped (R018, R018b: gate failed, ratio 1.52 / 1.51) |
| R018b | 2026-10-30 | the whistle (P10), the one retry R018 allowed: the tonal threshold set to the match's own 98th percentile of tonal_db, the same gate and bar | — | — | — | — | dropped, code removed (fc9dbed): gate failed, ratio 1.51 (55 detections) |
| R019 | 2026-10-30 | P09: the camera's aim in the shape model; part 1a: the camera's fit and fidelity gates (peak lag +-10 frames, peak correlation +-0.05, median error +-3.0 m, dead/alive speed ratio +-25%; the gap/visible ratio a WARN), declared before the first simulation; part 2a: the gates run | - | - | - | - | `tracking` FAIL: peak correlation 0.9115 vs ours 0.843, dead/alive speed ratio 0.575 vs 0.128 (the simulated camera never stands still); dropped 2026-10-31, code removed (8c4c959): gates 2 and 4 failed, no retry |

### The four yardsticks on every profile, in one table

The organisers' four baselines decoded per graded half (`nearest_w{0,25,50,125}`, the benchmark's yardsticks), as pooled
spellF1 on the seven DFL matches, per profile and under each truth ([the two
truths](../pipeline/benchmark/README.md#the-two-truths)). Every number is copied from the entry in its first column,
where the intervals, the paired deltas and the sign-flip tests are. The profiles differ in what they take from our
tracking. Read each row for its order, and a difference between two rows as that profile's effect on the same four
configs; absolute spellF1 on the benchmark is not our leaderboard's.

| profile (entry) | raw: `w0` | `w25` | `w50` | `w125` | coarse: `w0` | `w25` | `w50` | `w125` |
|---|---|---|---|---|---|---|---|---|
| `clean` v1 (R002) | 0.0983 | 0.2958 | **0.3073** | 0.2166 | 0.0434 | 0.2280 | 0.3037 | **0.3257** |
| `gaps` v1 (R003) | 0.0784 | 0.2021 | **0.2102** | 0.1712 | 0.0342 | 0.1541 | 0.1960 | **0.2302** |
| `tracking` v1 (R004) | 0.0768 | 0.1895 | **0.2000** | 0.1660 | 0.0332 | 0.1426 | 0.1826 | **0.2195** |
| `tracking` v2 (R004b) | 0.0779 | 0.1897 | **0.1989** | 0.1624 | 0.0341 | 0.1422 | 0.1811 | **0.2173** |
| **`tracking` v3 (R004c), the default** | 0.0777 | 0.1900 | **0.2018** | 0.1665 | 0.0342 | 0.1432 | 0.1842 | **0.2226** |
| `context` v1 (R005), sensitivity only | 0.0812 | 0.2097 | **0.2193** | 0.1774 | 0.0369 | 0.1608 | 0.2057 | **0.2443** |

Bold is the first of the four. For comparison, the published full-match scores (R000) are `w50` 0.2433, `w125` 0.2242,
`w25` 0.1996 and `w0` 0.0486. One line of verdict for each row:

- **`clean` (R002):** the benchmark's floor, with the ball on every frame, so `majority_vote` predicts `D` on 0.0% of
  the graded frames (alive/dead F1 0.7110 for all four, "always alive"); the truth reorders the yardsticks, and nothing
  here ranks a dead-ball detector.
- **`gaps` (R003):** frozen at v1, the first benchmark that looks like our match (our own ball gaps, transplanted with
  their pattern and not their clock, and not placed by the play); the yardstick check holds in its robust part (`w0`
  last under both truths), the rest of the order follows the truth.
- **`tracking` v1 (R004):** superseded by R004b; the five counted gates passed and three count gates stayed pending
  until the low-count tail existed; no order changes from `gaps`.
- **`tracking` v2 (R004b):** all eight gates pass and no WARN; no order moves from R004 (the four move by 0.4 points of
  spellF1 at most); superseded by R004c, which corrected its account of the jitter (T35).
- **`tracking` v3 (R004c), the default:** all eight gates and all ten checks pass; no order moves from R004b (no config
  by more than 0.6 points); the jitter per slot is 1.80 cm and not R004b's 8.5 cm, which was slot handovers (T35).
- **`context` v1 (R005):** fails two of its gates, the near-line share badly (0.5586 against 0.2797 +-0.05; `tracking`
  v3's transplanted gaps measure 36.0% on it, R003, but have no gate) and the share missing narrowly (0.4313 against
  0.4110 +-0.02, pooled), so it is a sensitivity check and not the default; it gives `tracking` v3's order under both
  truths (Kendall tau +1.000 between the two profiles' orders) with every config 0.3 to 2.2 points higher.

What every row says: `w0` is last under both truths, and the rest of the order follows the truth. The raw truth puts
`w50` first and the coarse one puts `w125` first, and neither gives the published order `w50 > w125 > w25 > w0` (R000).
So rank any choice that depends on possession length under both truths, and treat a disagreement as a warning (T33).

## Reference: the organisers' baselines (R000)

| ID | file | public macroF1 | public spellF1 | public poss / dur | full macroF1 | full spellF1 | full poss (own) |
|---|---|---|---|---|---|---|---|
| R000a | `baseline_raw` | 0.5972 | 0.0378 | 131 / 2.0 s | 0.6532 | 0.0486 | 1,949 |
| R000b | `baseline_smooth25` | 0.6200 | 0.2868 | 36 / 7.6 s | 0.6752 | 0.1996 | 746 |
| R000c | `baseline_smooth50` | 0.6207 | 0.3256 | 26 / 10.8 s | 0.6724 | **0.2433** ← floor | 594 |
| R000d | `baseline_smooth125` | 0.5812 | 0.4242 | 16 / 18.5 s | 0.6356 | 0.2242 | 398 |

The *full* columns are the organisers' full-match scores (EVALUATION.md), the only full-match numbers we will ever
see. *Full poss (own)* is each file's own possession count over the graded halves, not the GT's.

Reproduced on 2026-10-21:
- `python starter/baseline.py --out <scratch dir>` rebuilt all four baselines from the extracted data. They are
  identical to the committed files up to line endings.
- `python evaluation/run_evaluation.py` reproduces the public board under Python 3.11.9 (`.venv`: numpy 2.4.6,
  pandas 3.0.6, kloppy 3.18.0) and under 3.12.

## Analyses

### A01 — Anatomy of the public window and of the ball gaps (2026-10-21)

Computed from the public GT and `baseline_raw`, whose `D` marks exactly the frames where the ATD has no ball.

**Public window, frames 11,185–18,684.**
- 21 segments: 17 possessions plus 4 stoppages of 5.5, 20.4, 18.3 and 14.5 s.
- Possessions: median 7.6 s, mean 14.2 s, range 0.76–42.4 s. Six last under 5 s, and two under 2 s.
- 20 transitions: 12 team→team (6 each way), 4 into `D` (3 from A, 1 from H), and 4 out of `D` (2 each way).
- The ball is missing on 11.8% of frames: 6.5% of in-play frames, but 33.6% of dead-ball frames. So the ball is
  *visible* on 66% of dead-ball frames.
- The nearest player's team equals the GT team on 77.8% of in-play, ball-visible frames (5,640 frames).

Slack for IoU ≥ 0.7, as the total boundary error over both ends, per possession, shortest first:

| possession rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slack (s) | 0.3 | 0.6 | 1.2 | 1.4 | 1.5 | 1.5 | 2.8 | 3.2 | 3.2 | 3.5 | 5.1 | 6.9 | 8.6 | 12.3 | 15.3 | 17.8 | 18.2 |

**Ball gaps over the graded halves.** The ball is missing on 40.7% of frames, in only 150 gaps: median 10.7 s, p75
21 s, p90 34 s, p99 105 s, max 221 s.

| gap length | < 2 s | 2–5 s | 5–15 s | ≥ 15 s |
|---|---|---|---|---|
| share of missing frames | 0.1% | 0.2% | 24.9% | 74.8% |

**So:**
- Ball-missing stretches are long and hold whole possessions → P07, P09, P13, P14. Interpolation won't help.
- A dead-ball detector cannot lean on ball visibility → P05.
- Even with the ball visible, ownership needs control logic → P04.

*Later (23 Oct rule, `f8f2629`):* A01 describes the public window and stands as a description. Two constants were
derived from its statistics: `nearest_player`'s confidence 0.78 (77.8% of 5,640 frames) and `ball_presence`'s `missing`
0.2 and `seen` 0.585 (the ball missing on 33.6% of the dead frames and 6.5% of the alive ones). They were used in R006
to R010's research configs and in `p06_hmm.toml`. Setting a constant on the public window is the use that the
organisers' second clarification forbids ("not training data and not a tuning set"), so neither value is usable in a
submission config unless re-derived from other games, for instance from DFL (A06: 77.5% for the nearest player). Neither
source is in `main.toml`, whose constants come from DFL, the benchmark, fixed rules or the organisers' documented
kick-offs (the audit in the technical document, section 5).

### A02 — Ball coverage across the match (2026-10-21)

Share of frames with no ball, per 5-minute block of graded play, counted from each half's first graded frame (from
`baseline_raw`):

| half | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | 45–50 | 50–53 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1st | 12% | 17% | 24% | 28% | 44% | 35% | 63% | 43% | 71% | 46% | 80% |
| 2nd | 12% | 33% | 26% | 55% | 26% | 39% | 80% | 51% | 52% | 40% | 28% |

**So:** the public window is the best-covered block of the match, and ball-based methods will look better on it
than they really are ([T01](traps.md)). When building P01, degrade the benchmark to the late blocks' coverage
too, not only to the match average.

### A03 — Does the tactical camera follow the ball? (2026-10-21)

We took the pitch point under the image centre (the inverse homography of pixel 960, 540) and compared it with the
ATD ball over graded frames:
- The aim moves along the pitch (sd 19.2 m) and hardly across it (sd 3.6 m).
- On ball-visible frames, corr x is 0.83 at zero lag, and best at 0.84 with the aim taken 35 frames (1.4 s) *later*.
  Corr y is 0.07.
- Along-pitch error: median 12.1 m, p75 19.1 m, p90 25.6 m.
- The aim moves at 1.22 m/s while the ball is visible and 0.58 m/s during ball gaps.

**So:** the aim is a coarse zone prior for long ball gaps, not a position (→ P09). Look about 35 frames ahead.

### A04 — What the files contain: streams and shapes (2026-10-22)

Structure only: stream info, element paths, column names, counts. No values were read. The video figures come from
`ffprobe -show_entries format=…:stream=…` (FFmpeg 9.0.2, `winget install Gyan.FFmpeg`); keyframe spacing is measured
over the first 60 s.

| | tactical | broadcast |
|---|---|---|
| video | H.264 High, 1920×1080, 25/1 CFR, no B-frames, 199,319 frames, 7,972.76 s | H.264 Main, 1920×1080, 25/1 CFR, no B-frames, **164,487** frames, 6,579.48 s |
| keyframes | every 50 frames (2 s) | every 30 frames (1.2 s) |
| audio | AAC stereo 48 kHz, starts at 0.000 s, ends 0.07 s before the video | AAC stereo 48 kHz, starts at 0.000 s |
| ATD txt | 199,319 lines, `frame_id` 1…199,319 with no gaps; all 46 slots (`x,y,speed`) on every line, plus the ball (`x,y,speed`) | 164,488 lines, all 51 slots on every line |
| ATD xml | 2 teams, 46 players with `teamId`, 3 channels | 2 teams, 51 players |
| homography | 199,319 × 12 (`frame`, `h00…h22`, `reprojection_error`, `valid`), no NaN | 164,488 × 12; 56,240 rows (34.2%) NaN |
| Smart Tagging | 638 instances (`ID`, `start`, `end`, `code`, 1,762 `label` group/text pairs); 14 codes | — |

**So:**
- P10 can go ahead. The tactical mp4 has audio from t = 0, so audio second t is frame `round(t * 25)` with no
  offset to correct.
- The broadcast mp4 is one frame shorter than its ATD and homography (164,487 vs 164,488). We don't know which end
  is missing; find out before using broadcast pixels (P14, P15).
- The ATD writes every slot on every line, so an absent player shows up as a placeholder value in its slot, not as a
  missing entry.
- A frame-exact decode of the tactical video seeks back to a keyframe, at most 49 frames earlier ([T23](traps.md)).

### A05 — The DFL parser against kloppy and by frame number (2026-10-22)

Public data only (the DFL/IDSSE matches of P01). Command: `python pipeline/benchmark/selftest.py --dfl`, checks
`test_dfl_parser_agrees_with_kloppy`, `test_dfl_second_half_spot_check`, `test_dfl_t30_facts`,
`test_dfl_cache_equals_fresh_parse` and `test_dfl_all_matches_load`; the first parse of a match takes 42 to 64 s. The
timeline is `pipeline/benchmark/dfl.py`'s: the first half, 25 separator rows, then the second half (J03WMX: 145,992
rows, the second half starts at row 70,733).

- **kloppy, J03WMX, `only_alive=False, limit=5000`** (N 10,000 to 14,999, the first half only; 4,176 alive, 1,359
  with the home team in possession): the pitch length and width (2), ball x, y and z (5,000 each), alive or dead
  (5,000), team in possession (5,000), the team of the 22 players in its first frame, presence of 3 players (15,000),
  their x and y (30,000 values) and the number of players per frame (5,000). Positions agree to 1e-6 m. **0
  disagreements** in every count. The counts quoted here (4,176 alive; 1,359 home; 22 players; 30,000 values; and the
  default's 15,823, 5,824 and 824) are asserted by the check.
- **kloppy's default, `only_alive=True`, same limit:** 5,000 frames over N 10,000 to 15,823 (5,824 rows). None is dead
  in our parse, no alive row of ours in that range is left out, and the 824 dead rows are the ones it drops.
- **By frame number:** a separate lxml pass, keyed by N, read 200 ball frames from each half (second half drawn with
  `default_rng(0)`, first with `default_rng(1)`) and compared N, X, Y, Z, S, BallStatus and BallPossession with the
  timeline's row: N - 10000 in the first half, n1 + 25 + (N - 100000) in the second, n1 being the rows of the first
  half as that pass counts them (largest ball N less the smallest, plus one), which `graded` must equal.
  **0 disagreements.**
- **Fields that are missing:** none of X, Y, S, Z, BallPossession or BallStatus is missing on any ball or player frame
  of J03WMX, and no row lacks a ball frame; the same holds for all seven matches (below), and in all seven no player
  frame lies outside the two halves (0 dropped).
- **Bench players** (2026-10-23, aggregates only). The position arrays have a column for every squad member (40, 40, 39,
  40, 40, 40 and 40 for J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9), and a column of a player who never
  had a position is NaN on every row: 8, 11, 7, 8, 9, 9 and 9 of them (home and away: 4 and 4, 4 and 7, 4 and 3, 4 and
  4, 5 and 4, 5 and 4, 5 and 4), so 7 to 11 a match. Method: `np.load` of `players_xy.npy` and `player_team.npy` of each
  match's folder under `data/cache/dfl/`, and a count of the columns whose x is NaN on all rows. `nearest_player`
  ignores an absent slot, and the degrader walks only the players who have positions (`degrade.py`).

Ball frames, by match, from the same parse (`rows w/o ball`, `no possession` and `no status` are 0 everywhere, and the
parser refuses a possession that is not 1 or 2, so a clean parse is a 0 for that too):

| match | ball frames | dead | dead share |
|---|---|---|---|
| J03WMX | 145,967 | 60,598 | 41.5% |
| J03WN1 | 141,561 | 73,184 | 51.7% |
| J03WOH | 137,214 | 65,269 | 47.6% |
| J03WOY | 142,536 | 66,250 | 46.5% |
| J03WPY | 146,211 | 64,751 | 44.3% |
| J03WQQ | 142,345 | 57,993 | 40.7% |
| J03WR9 | 146,810 | 61,494 | 41.9% |

**So:**
- Our timeline and kloppy read the same ball, alive flag, possession and players, and the halves are laid out where
  the file says. Home and away agree between kloppy's reading of `Role` and ours.
- Dead ball is 41% to 52% of every benchmark match. kloppy's default drops all of it ([T30](traps.md)), which would
  leave the benchmark with no `D`. Its fallbacks for a missing possession or ball entry never apply to these files.

**Cost** (2026-10-23; Windows 11, local disk, Python 3.11.9). Every later stage reads a match through
`dfl.load_match`, which keeps it as nine `.npy` files and a `manifest.json` in `data/cache/dfl/<match id>/`, keyed on
the sha256 of both source files, the numpy and lxml versions, `CACHE_VERSION` and the sha256 of `dfl.py` itself, taken
when the module is imported. `test_dfl_all_matches_load` loads all seven twice, and `test_dfl_cache_equals_fresh_parse`
compares J03WMX from the cache with a fresh parse (0 differences in all nine arrays, the scalars and the report):

| | no cache (built) | cache (reused) |
|---|---|---|
| One match, first measurement | 41.7 to 49.1 s | 0.7 to 1.9 s |
| One match, after the parser's hash changed | 21.6 to 29.4 s | 1.4 to 1.8 s |
| All seven, first measurement | 325.8 s | 5.5 to 6.7 s |
| All seven, after the hash changed | 174.4 s | 10.7 s |
| Whole `--dfl` run, 59 checks (first measurement) | 532 s | 221 s |
| `test_dfl_t30_facts` (first measurement) | 297.8 s before the cache (one parse per match) | 4.7 s |

The two measurements are the same code on a differently loaded machine, so the times depend on the machine's load:
quote the ranges, never one figure.

- A reused match is read in under 2 s, almost all of it the sha256 of the two source files (the positions file is up
  to 420 MB), which is what keeps a cache from being read after its sources change. The caches take 129 to 141 MB per
  match, 956 MB for the seven.
- What the reused `--dfl` run still spends: kloppy's own loads (about 100 s), the fresh parse of the cache check (about
  50 s), the spot check's own pass over the file (about 34 s) and the 2.6 GB md5 check (about 10 s).
- **Suite times at the end of P01** (2026-10-23, `fdc5eae`, the same machine; its load decides, so ranges): the
  synthetic run is 272 checks and took 49 to 115 s over six runs (R005 measured 46 s for 264, on a quiet machine);
  `--dfl --data` is 295 checks (272 synthetic, 16 with `--dfl`, 7 with `--data`) and took 949 s (15 min 49 s, the sum of
  the checks' own times 945 s) with the seven DFL caches and the Calibration rebuilt after the edits to `dfl.py` and
  `calibration.py` (the caches are keyed on their hashes); `pipeline/selftest.py --data` passed 37 of 37, the four
  baselines byte for byte, in 9 s.

### A06 — DFL's conventions against the public window and the literature (2026-10-23)

The seven DFL matches (public data, CC BY 4.0) and our public GT; no BAR-ATH tracking is read. Command:
`python pipeline/benchmark/selftest.py --dfl`, checks `test_dfl_truth_conventions`, `test_dfl_window_confound` and
`test_dfl_coarse_truth` (they print every number below and take 8 s together with the caches warm). They assert every
integer count quoted here (possessions, short runs, stoppages, window counts and that the windows are drawn from all
seven matches, changes, the three matches whose upper quartile of the lead is the whole 10 s window, the coarse truth's
possessions), the 60% agreement and DFL's playing time to within 2 frames; the rates, shares, medians, lags and line
timings are printed, not asserted. The rules are checked on hand-built matches by `test_truth_*`
(`pipeline/benchmark/checks/truth.py`). The truth is `pipeline/benchmark/truth.py`'s, built as our GT is: `D` where
`BallStatus` is 0, otherwise the team `BallPossession` names (set on dead frames too, and ignored there), `X` outside
the two halves. All seven matches have both flags on every frame, so the truth carried 0 frames and guessed none. The
literature ranges are the ones collected for P01: ball in play 55–60% of match time, 100–115 stoppages per match,
175–220 team possessions per match under Metrica-like rules and up to about 420 if every touch counts.

**Possessions and stoppages.** Possession lengths come from `pm.possession_spells`, stoppages are `D` runs, both
inside each half. *pooled* is one summary of the seven matches laid end to end, never an average; *first 5′* pools the
first 5 minutes of each first half, as long as the public window.

| | minutes | alive | possessions | per alive min | median | mean | under 5 s | ≤ 2 frames | ≤ 12 frames | stoppages | stoppage median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | 97.3 | 58.5% | 511 | 8.98 | 3.00 s | 6.68 s | 64.2% | 3 | 29 | 103 | 15.5 s |
| J03WN1 | 94.4 | 48.3% | 500 | 10.97 | 3.12 s | 5.47 s | 66.2% | 2 | 32 | 107 | 23.8 s |
| J03WOH | 91.5 | 52.4% | 402 | 8.38 | 3.94 s | 7.16 s | 58.2% | 2 | 14 | 102 | 19.1 s |
| J03WOY | 95.0 | 53.5% | 456 | 8.97 | 3.60 s | 6.69 s | 63.8% | 1 | 17 | 113 | 19.2 s |
| J03WPY | 97.5 | 55.7% | 421 | 7.75 | 3.84 s | 7.74 s | 60.1% | 3 | 27 | 97 | 22.8 s |
| J03WQQ | 94.9 | 59.3% | 401 | 7.13 | 4.68 s | 8.41 s | 52.9% | 3 | 14 | 99 | 19.6 s |
| J03WR9 | 97.9 | 58.1% | 370 | 6.51 | 4.24 s | 9.22 s | 55.1% | 1 | 23 | 92 | 20.0 s |
| **pooled** | 668.4 | 55.2% | 3,061 | **8.30** | **3.68 s** | 7.23 s | **60.5%** | 15 | 156 | 713 | 19.6 s |
| **public window** | 5.0 | 80.4% | 17 | **4.23** | **7.56 s** | 14.19 s | **35.3%** | 0 | 0 | 4 | 16.4 s |
| DFL first 5′ | 35.0 | 64.8% | 211 | 9.30 | 3.12 s | 6.45 s | 64.5% | 3 | 11 | 44 | 13.5 s |
| literature | | 55–60% | 175–220 (up to ~420) | | | | | | | 100–115 | |

- **Ball in play:** 48.3% to 59.3% of the graded frames, 55.2% pooled; four matches are inside 55–60%, three below.
  The first halves are livelier than the second (54.5–68.0% against 41.0–52.5%). The public window, at 80.4%, is
  livelier than 96% of DFL's 5-minute windows.
- **Stoppages:** 92 to 113 per match, 101.9 on average; four matches are inside 100–115, three below.
- **Possessions:** 370 to 511 per match, 437 on average, all seven above the Metrica-like range and four of them above
  420. DFL's flag is fine-grained, at the every-touch end of the literature's range, with a median possession of 3.7 s.
  That is a measurement of DFL. It says nothing yet about how far our GT is from it (next section).
- **Short runs:** 15 of 3,061 pooled possessions (0.49%) last 2 frames or less, and 156 (5.1%) last 12 frames or less
  (0.48 s); the shortest public possession is 0.76 s. The flicker is not material (the 2% rule): the truth is the raw
  one, and the de-flickered variant (`min_run`) is not built.

**The public window is unusually lively, and that confounds its rate.** The public window is alive on 80.4% of its
frames: 124 of DFL's 129 whole 5-minute windows (96.1%; each half cut from its first frame) are at or below that. In
DFL the possession rate per alive minute falls as the alive share rises (r = −0.71 over the 129 windows), partly because
every stoppage splits a possession. So the public window's 4.23 per alive minute is set against the DFL windows that
are as lively, and against the line fitted to all of them:

| DFL windows | n | median rate | rate ≤ 4.23 | median of the medians | median ≥ 7.56 s | under 5 s ≤ 35.3% |
|---|---|---|---|---|---|---|
| all | 129 | 8.91 | 7 (5.4%) | 3.76 s | 7 (5.4%) | 9 (7.0%) |
| alive ≥ 70% | 32 | 6.28 | 7 (21.9%) | 5.44 s | 5 (15.6%) | 7 (21.9%) |
| **alive ≥ 75%** | 19 | **5.67** | 5 (26.3%) | 5.52 s | 4 (21.1%) | 5 (26.3%) |
| public window | 1 | 4.23 | | 7.56 s | | |

- The comparable windows (≥ 75% alive, all seven matches) have a median rate of 5.67, so the public window is 1.34 times
  slower, not 1.96 times (the pooled 8.30). A quarter of them (5 of 19) are as slow as the window and a fifth (4 of 19)
  have a median possession as long as its 7.56 s.
- The fit over all 129 windows is rate = 17.27 − 14.62 × alive share (residual sd 2.38). At 80.4% alive it predicts 5.51
  per alive minute; the public window's 4.23 is 0.54 residual sd below it.
- 17 possessions carry their own noise (Poisson alone is ±24%), on top of this. The window therefore sits inside what
  DFL's own conventions give on a lively stretch, slower than most and not outside it. It neither shows nor excludes a
  difference in grain between DFL's flag and our GT, and no estimate of our match's possession count is made from it.

**DFL's own playing time.** `OtherGameInformation@PlayingTime…` over `@TotalTime…` from each match information file,
against the alive share of the truth, and the alive frames less PlayingTime (converted to frames) in all and per half:

| match | playing / total | of half 1 | of half 2 | alive frames | less PlayingTime | half 1 | half 2 |
|---|---|---|---|---|---|---|---|
| J03WMX | 58.5% | 66.8% | 50.7% | 85,369 | +2 | +1 | +1 |
| J03WN1 | 48.3% | 55.9% | 41.0% | 68,377 | +1 | +1 | 0 |
| J03WOH | 52.4% | 54.5% | 50.3% | 71,945 | +1 | +1 | 0 |
| J03WOY | 53.5% | 54.6% | 52.5% | 76,286 | +1 | 0 | +1 |
| J03WPY | 55.7% | 66.8% | 45.8% | 81,460 | +2 | +1 | +1 |
| J03WQQ | 59.3% | 68.0% | 51.0% | 84,352 | +2 | +1 | +1 |
| J03WR9 | 58.1% | 64.8% | 52.3% | 85,316 | +2 | +1 | +1 |

The alive frames of the flags are DFL's official ball-in-play time to within 2 frames in every match and 1 in every
half, so `D` is DFL's own stoppage clock. The matches range from 48% to 59% in play, around the literature's
55–60%.

**The cost of being late.** spellF1 of the truth against itself shifted k frames later within each half (the first k
frames `X`, which the scorer back-fills), through `pm.evaluate`:

| | k = 3 | k = 8 | k = 25 |
|---|---|---|---|
| J03WMX | 0.9503 | 0.8184 | 0.4943 |
| J03WN1 | 0.9476 | 0.8352 | 0.5048 |
| J03WOH | 0.9706 | 0.8746 | 0.5816 |
| J03WOY | 0.9732 | 0.8829 | 0.5395 |
| J03WPY | 0.9492 | 0.8518 | 0.5639 |
| J03WQQ | 0.9731 | 0.8933 | 0.6125 |
| J03WR9 | 0.9524 | 0.8751 | 0.6011 |
| **pooled** | **0.9590** | **0.8594** | **0.5522** |
| public GT, same function | 1.0000 | 0.9529 | 0.7765 |
| EVALUATION.md | 0.9818 | 0.8933 | 0.6539 |

- **EVALUATION.md's table does not reproduce on the public GT.** k = 1 does (1.0000), k = 3, 8 and 25 do not: the public
  GT gives 1.0000, 0.9529 and 0.7765 against the table's 0.9818, 0.8933 and 0.6539. Shifting the other way, or keeping
  the head instead of `X`, changes nothing, so the method is not what differs. EVALUATION.md does not say what the table
  was computed on, and nothing is inferred from it here ([T33](traps.md)).
- What a lag costs depends on how long the possessions are, so it is a property of a truth. On DFL's, being 1 s late
  costs 0.45 and being 8 frames late 0.14; on the public window, 0.22 and 0.05. Neither is the cost on our full match.

**The flip to dead against the ball leaving the pitch.** At each alive → dead change (710 in all), the ball's normalised
centre is checked against the lines (x or y outside 0–1) over the 10 s before the change; the lead is the time from the
first such frame to the change. In the first measurement the window reaches back into an earlier stoppage (a throw-in
taken less than 10 s before the next stop), which puts the ball beyond a line on dead frames: that is why its upper
quartile is 7.6 s (10.00 s, the window's whole length, in three matches: J03WMX, J03WN1 and J03WOY). The second
measurement looks only at the frames the truth calls alive.

| | changes | ball beyond a line | median lead | p25 | p75 |
|---|---|---|---|---|---|
| any frame of the window, pooled | 710 | 518 (73.0%) | 0.16 s | 0.12 s | 7.61 s |
| alive frames of the window, pooled | 710 | 490 (69.0%) | **0.16 s** | 0.08 s | **0.28 s** |
| alive frames, per match | 92–112 | 41.4%–79.6% | 0.12–0.16 s | 0.08–0.12 s | 0.20–0.32 s |

- DFL's flag turns dead 4 frames (0.16 s) after the ball is first seen beyond a line while the flag still says alive,
  and within 0.28 s in three cases out of four. The other 31% of the changes have no alive frame with the ball beyond a
  line in the 10 s before (fouls, offsides and the like).
- J03WQQ has the lowest share (41.4% of its changes), and its ball is tracked beyond a line on 30% of its dead
  frames, against 59–74% in the others. The two are seen together and not explained (not investigated further). The
  tracked ball is never more than about 4 m outside a line (x from −0.022 to 1.029 and y from −0.059 to 1.046 of the
  pitch over the seven matches; J03WQQ goes furthest).
- Our match has no counterpart yet: it needs BAR-ATH's ball against the public GT's `D` onsets, which is P02.
  *Later:* A11 has one, from 2 of the 4 public stoppages (the other 2 have no first sighting of the ball beyond a line
  within 5 s): our tracking sees it a median 37 frames (1.48 s) after our `D` begins, where on DFL's `tracking` v3 copy
  (raw and coarse truth alike) the flag turns 3 frames after the ball is seen, beside the 4 frames above, measured on
  DFL's own positions. A lead, not a measurement: our truth's convention, or our tracker's loss of the ball near the
  lines (T39).

**The nearest player against the truth** (A01: 77.8% of 5,640 alive ball-visible frames on the public window). Brute
force, in metres, present players only:

| match | all | half 1 | half 2 | frames |
|---|---|---|---|---|
| J03WMX | 77.9% | 80.4% | 74.9% | 85,369 |
| J03WN1 | 75.0% | 78.3% | 70.6% | 68,377 |
| J03WOH | 76.5% | 76.3% | 76.8% | 71,945 |
| J03WOY | 75.3% | 77.0% | 73.6% | 76,286 |
| J03WPY | 78.6% | 81.2% | 75.3% | 81,460 |
| J03WQQ | 77.4% | 79.2% | 75.3% | 84,352 |
| J03WR9 | 80.8% | 82.2% | 79.2% | 85,316 |
| **pooled** | **77.5%** | | | 553,105 |

Every match and every half is above 60% (the check asserts it), so home is team 0 everywhere and no half is
misaligned. The second halves are 4 points lower on average (six of the seven).

The pipeline's own `nearest_player`, served through `MatchInputs` (`test_dfl_match_inputs_serve_the_real_matches`),
names a team on every graded row of the seven matches. The check asserts that the rows it cannot answer are exactly
the 25 gap rows between the halves of each (`dfl.GAP_ROWS`), that the seven timelines have 1,002,819 rows together, and
that the rows where it differs from the brute-force team above are the one tie, J03WMX row 58523, a tie between the
best home and the best away player (1 row of the 1,002,819); it prints 100.0% answered for each match. In
`nearest_player`'s arithmetic their squared distances are exactly equal and `argmin` takes the lower slot, which is
home; in the brute force's they are 5e-15 m² apart, with the away player nearer.

**The coarse truth, a side-by-side check.** `truth.coarse_codes` absorbs a possession shorter than `min_s` into the
other team when that team has the ball on both sides of it with no `D` between (A·B·A becomes A·A·A), the shortest
first, to a fixed point; `D`, `X`, the halves and every spell next to a `D` are untouched. It exists because the grain
of our GT is unknown (above): a truth much finer than ours could rank a smoothing level, or a switch penalty,
differently from one like ours. It is a **sensitivity check, not a correction**: it is scored beside the raw truth,
which stays the default, and a disagreement between the two is a warning about that choice. `min_s` was fixed by a
rule written before the numbers were looked at and blind to our match: the smallest of 1, 2, 3, 4 and 5 s whose pooled
rate is at or below the literature's upper bound, 220 possessions a match, which is 4.18 per alive minute at DFL's own
52.7 alive minutes a match (175 is 3.32), and the largest candidate if none is. The public window's rate is not used.

| min_s | possessions per match | per alive minute | median spell | mean spell | raw spells absorbed | frames changed |
|---|---|---|---|---|---|---|
| raw | 437.3 | 8.30 | 3.68 s | 7.23 s | 0 | 0 |
| 1 s | 386.7 | 7.34 | 4.40 s | 8.17 s | 5.8% | 0.22% |
| 2 s | 324.4 | 6.16 | 5.76 s | 9.74 s | 12.9% | 1.03% |
| 3 s | 282.7 | 5.37 | 6.76 s | 11.18 s | 17.7% | 1.90% |
| 4 s | 257.0 | 4.88 | 7.72 s | 12.30 s | 20.6% | 2.66% |
| **5 s** | **237.0** | **4.50** | 8.40 s | 13.34 s | 22.9% | 3.43% |

- **No candidate reaches 4.18.** The rule then takes the largest, so `COARSE_MIN_S` is **5 s**, and that is flagged:
  even there the coarse truth has 237 possessions a match, above the 175–220 range. (The public window's 4.23 happens
  to be close to 4.50. That is not why 5 was taken and must not become the reason to keep it.)
- The coarse truth, pooled, next to the raw one (`stoppages` do not change, so `D` is identical):

| | possessions | per alive min | median | mean | under 5 s | ≤ 2 frames | ≤ 12 frames |
|---|---|---|---|---|---|---|---|
| raw | 3,061 | 8.30 | 3.68 s | 7.23 s | 60.5% | 15 | 156 |
| coarse (5 s) | 1,659 | 4.50 | 8.40 s | 13.34 s | 30.3% | 3 | 41 |

  Per match it has 220–255 possessions (3.90–5.57 per alive minute); its spellF1 against itself lagged 3, 8 and 25
  frames, pooled, is 0.9834, 0.9236 and 0.7720.
- Short spells remain (41 of at most 12 frames): those next to a `D` or at an end of their half, which the rule
  leaves alone.

**So:**
- **The only like-for-like comparison with our match is the nearest player's team:** 77.5% of DFL's alive ball-visible
  frames against 77.8% of the public window's (A01). That is one aggregate, from different tracking, and ours comes
  from a single 5-minute stretch of 5,640 frames, so even this one is a public-window figure. It shows that one simple
  source names the team about as often in the two; it says nothing yet about any other evidence. Every other
  comparison here is with the public window, which cannot carry a verdict, or with the literature. Evidence sources
  and degradation handling are ranked on the benchmark on its own terms; a decoder choice that does not depend on
  how long a possession is can be taken from it too, and one that does (smoothing, a switch penalty, duration priors)
  needs both truths (the **Use** bullet below).
- **DFL's flag is fine-grained, and that is measured:** 370–511 possessions a match (437 on average), far above the
  literature's 175–220 for control-based rules; the mean is past the roughly 420 that counting every touch gives, and
  three matches (370, 401 and 402) are below it; a median of 3.7 s; `D` is DFL's own
  ball-in-play clock to within 2 frames, with a stoppage count (102 a match) in the literature's range and the flag
  flipping a median 0.16 s after the ball is seen beyond a line.
- **How far our GT differs in grain is not known.** The public window is livelier than 96% of DFL's, and the rate falls
  with liveliness (r = −0.71). Against the 19 comparable DFL windows it is 1.34 times slower (5.67 against 4.23), 26% of
  them are as slow, 21% have a median possession as long, and DFL's own fit predicts 5.51 where the window has 4.23
  (0.54 residual sd below). That is within what DFL's conventions produce, and 17 possessions cannot separate a
  different grain from noise. The 175–220 of the literature is what a control-based provider gives; whether ours is
  one is not shown here.
- **D timing against ours is unmeasured.** DFL's counterpart needs BAR-ATH's ball, which is P02. *Later:* A11 measured
  it on the public window's 4 stoppages and 12 team changes, weakly: against DFL's flag on the `tracking` v3 copy (raw
  and coarse truth), the events our tracking sees put our `D` 2.5 to 3 s longer. That is our truth's convention, chance
  or our tracker's coverage, and A11 does not separate them.
- **Use:** hyperparameters are picked on the benchmark, never on the public window (CLAUDE.md, T01). Where a choice
  depends on how long a possession is (smoothing, a switch penalty λ, duration priors), rank it under the raw truth and
  under the coarse one side by side, and treat a disagreement between them as a warning, not as a licence to pick the
  one that suits. Fit timing (P04's back-dating, P05's `D` onset, P12's shift) on the public window through P02.
  Absolute spellF1 on the benchmark is not our leaderboard's under either truth. *Later (23 Oct rule, `f8f2629`):* the
  sentence on fitting timing on the public window is withdrawn. The window is for checking the pipeline and is "not a
  tuning set": P02 may inspect it to understand how our truth flips, and no timing constant is fitted on it. A shift
  comes from other games only (P16).
- **What the benchmark cannot test:** DFL's halves start at the kick-off (11 of 14 start alive; the other three have 1,
  1 and 48 leading `D` frames) and all 14 end in `D`, whereas our graded second half starts 117 s before the kick-off,
  all `D` (T02).

### A07 — Our tracking, measured for the benchmark's calibration (2026-10-23)

BAR-ATH's ATD and the public GT, read by `pipeline/benchmark/calibration.py` (`load()`, ATD cache present: 1.9 s to
measure, 0.4 s to read back, 0.5 MB under `data/cache/benchmark/calibration/`); no DFL data. Command:
`MTAH_DATA_DIR=<data dir> python pipeline/benchmark/selftest.py --data`, check `test_data_calibration_replicates_a01`.
The definitions are in the module's docstring and are checked on hand-built objects by `test_calibration_*`: a gap is a
maximal run of missing ball inside a block, **interior** if a visible frame lies on both sides of it; a tracklet is a
maximal run of one slot being present inside a block; a frame is *wrong-team* if a team shows more than 11 slots while
the other shows fewer than 11; a ball or a slot is present only if both coordinates are finite (on BAR-ATH none is
half-missing: the count of dropped frames and slot-frames is 0).

**Which span each figure is measured on.** A benchmark match starts at its kick-off, so everything that drives the
degradation and centres the fidelity gates is measured on **each half from its kick-off to the end of its graded block**
(155,923 frames: 79,467 and 76,456), and that is what the `Calibration` stores: the gap, tracklet, wrong-team,
NaN-speed and jitter statistics, the slot histogram, `ball_masks` and `present_counts`, and every onset minute, counted
from the kick-off (none is negative). Our second half is graded from 117 s (2,934 frames) before its kick-off, all of
it `D`, and those frames distort the graded-block figures. Three things stay on other spans: `missing_share_5min` is
A02's table, from each half's first graded frame (it reproduces a published table); `baseline_outputs` are over the
graded halves, as the published baselines are; the `public_window` figures are the public GT's 5 minutes. A01, A02,
R001 and the plan published **graded-block** figures: the `--data` check computes them by calling the same two functions
on the graded blocks (158,857 frames: 79,467 and 79,390) and asserts them, and does not store them.

**What was already published, re-measured by the same definitions** (✓ is asserted):

| statistic | published | graded blocks | from the kick-offs (stored) |
|---|---|---|---|
| ball missing, share ✓ | 0.4069 (A01) | 0.4069 (64,643 of 158,857) | 0.4110 (64,079 of 155,923) |
| gaps / interior gaps / near-line onsets ✓ | 150 / 147 / 42 (plan) | 150 / 147 / 42 | 146 / 143 / 40 |
| near-line share of the interior gaps ✓ | 28.57% | 28.57% | 27.97% |
| median gap ✓ | 10.7 s (A01) | 10.72 s | 10.74 s |
| share missing per 5 minutes, both halves ✓ | A02's 22 blocks | all 22 equal, in whole percent | (stored from the graded blocks) |
| tracklets ✓ | 1,219 (plan) | 1,219 | 1,144 |
| tracklet median | 35.5 s (plan) | 35.52 s | 41.28 s |
| wrong-team frames ✓ | 0.108 (plan) | 0.1080 (17,156) | 0.1054 (16,433) |
| wrong-team episodes, home / away in surplus ✓ | 63 / 45 (plan) | 63 / 45 | 60 / 44 |
| NaN speeds over present slot-frames ✓ | 0.0796 (plan) | 0.0796 (258,995 of 3,254,570) | 0.0783 (250,466 of 3,199,919) |
| slots present per frame, mean ✓ | 20.5 (plan) | 20.49 | 20.52 |
| exactly 22 present ✓ | 49.5% (plan) | 49.48% | 50.19% |
| 14 or fewer present | 6.6% (plan, **not reproduced**) | 5.26% | 5.25% |
| jitter, RMS of the second difference ✓ | 3.3 mm (plan) | 3.31 mm | 3.28 mm |
| ball-speed channel NaN, ball-present frames ✓ | 92.2% (plan) | 92.17% of 94,214 ([T31](traps.md)) | 91.97% |
| nearest player = GT team, public window ✓ | 77.8% of 5,640 (A01, R001) | 77.84% of 5,640 | the same |
| ball visible on dead frames, public window | 66% (A01) | 66.4% of 1,469 dead frames | the same |

The nearest-player agreement is R001's own method: `pipeline.evidence.describe` of the `nearest_player` source against
the public GT, on the frames both have an opinion on. The tracklet median is the one figure that moves much: 41.3 s from
the kick-offs against 35.5 s on the graded blocks, because the 117 s before our second half's kick-off hold short
tracklets (75 more tracklets, 162 more NaN-speed runs on the graded blocks than from the kick-offs).

**Ball** (from the kick-offs). 146 gaps (143 interior), in seconds: p10 2.68, p25 6.18, median 10.74, p75 21.15, p90
35.3, p99 107.5, max 221.5; the interior ones: median 10.76, p90 36.4. Interior onsets per 10 minutes of a half: 22, 27,
28, 29, 32, 5 (the spans last 52.98 and 50.97 minutes, so the last bin holds about 3.0 minutes of half 1 and 1.0 minute
of half 2). The ball is missing on 12% to 80% of each 5-minute stretch of the
graded blocks (A02), and on 33.6% of the dead frames of the public window. `gap_durations_s` holds every gap, and
`interior_durations_s`, `gap_onset_minute` and `gap_onset_near_line` hold the 143 interior ones, entry by entry, so a
duration can be read with its class and its minute of the half.

**Players** (46 slots, from the kick-offs; graded blocks in brackets where they differ).

| | |
|---|---|
| present per frame | mean 20.52 (20.49); p5 14, p25 21, median 22; exactly 22 on 50.19% (49.48%); 14 or fewer on 5.25% (5.26%) |
| tracklets (s) | 1,144 (1,219): p10 2.7, p25 8.3, median 41.3 (35.5), p75 152.3, p90 297.4 (284.6), p99 776.0, max 2,937 |
| wrong team ([T32](traps.md)) | 10.54% of frames (10.80%) in 104 episodes (108): 60 home in surplus, 44 away; median 2.7 s, p90 15.8 s (16.1), max 50.2 s |
| NaN speed, runs of one present slot | 2,843 runs (3,005): median 21 frames, p75 21, p90 69 (68), p99 2,323 (2,305), max 9,268 |
| jitter | RMS of the second difference 3.28 mm (3.31) over 3,197,631 present triples |

**The four yardsticks** (`nearest_w{0,25,50,125}.toml`, through `run.predict` and `pipeline.output.summarise`; no CSV is
written), over the graded halves (asserted to ±5 possessions):

| | w0 (raw) | w25 | w50 | w125 |
|---|---|---|---|---|
| possessions | 1,949 | 746 | 594 | 398 |
| per graded minute (105.9 min) | 18.40 | 7.04 | 5.61 | 3.76 |
| D share of graded frames | 0.4069 | 0.3649 | 0.3234 | 0.2233 |

Their codes are **identical, frame for frame, to the published `baseline_raw`, `baseline_smooth25`, `baseline_smooth50`
and `baseline_smooth125` on all 158,857 graded frames** (the check asserts 0 differing frames at each width), although
ours are decoded per half and the published ones per file.

Two figures of the plan did **not** reproduce, and the check does not assert them:
- **Frames with 14 or fewer slots present: 6.6% in the plan, 5.26% here** (5.25% from the kick-offs). The rest of the
  plan's table does (22: 49.5%, 21: 25.6%, 20: 7.7%, ≤ 9: 3.0%, p5 14); no cut-off from 14 to 16 gives 6.6% (5.26%,
  6.38%, 7.66%), and the whole file gives 16.4%. The `tracking` gate (0.066 ± 0.02) holds for BAR-ATH's own 0.0525,
  0.014 from its centre.
- **p99 of the NaN-speed runs: 1,209 frames in the plan, 2,305 on the graded blocks (2,323 from the kick-offs).** The
  plan's figure is the whole file (all 199,319 frames: 5,296 runs, median 21, p90 66, p99 1,209.9, max 14,509). The
  median agrees in all three.

**So:**
- Every figure of A01, A02 and R001 that the calibration re-measures is reproduced by one set of definitions, so the
  benchmark can be held to BAR-ATH by the same two functions (`ball_stats`, `player_stats`). The plan's fidelity gates
  were centred on the graded-block figures; every one still holds for the figures from the kick-offs (missing share
  0.411 against 0.407 ± 0.02, median gap 10.7 s, 22 present 0.502 against 0.495 ± 0.06, wrong team 0.105 against 0.108
  ± 0.03, tracklet median 41.3 s against 35.5 s ± 50%, NaN speed 0.078 against 0.080 ± 0.01, near-line onsets 0.280
  against 0.286 ± 0.05), and the span of ours that the benchmark's degraded halves should be held to is the stored one.
- The per-half yardsticks reproduce the organisers' four baselines on every graded frame, so they are a faithful
  stand-in for the baselines on the benchmark, and their outputs here are the published files' (R000).
- The ball-speed channel is unusable ([T31](traps.md)) and slots do carry the wrong team ([T32](traps.md), which answers
  [T15](traps.md)). The wrong-team share is a lower bound: a swap inside 11 against 11 is invisible.
- The cache key holds the hashes of `calibration.py` and `hazard.py`, the ATD parse version and the four configs, so a
  change of definition, of parse or of a config rebuilds it. A change to the code that the yardsticks run through
  (`pipeline/inputs/`, `evidence/`, `decode/`, `refine/`, `output/`, `benchmark/run.py`) is not seen by the key and
  needs a bump of `calibration.CALIBRATION_VERSION`.

### A08 — Where our tracker loses the ball: a hazard fitted on BAR-ATH, and the go decision (2026-10-23)

BAR-ATH's ATD (ball and slots), read by `pipeline/benchmark/calibration.py` (`fit_hazard`, on `hazard.py`'s features,
logistic fit and AUC); no DFL data. `load()` measures it in 2.3 s and reads it back in 0.1 to 0.4 s; the Calibration
is version 2 and sits in a folder of its own per cache key, `data/cache/benchmark/calibration/<digest>/` (0.5 MB).
Command: `MTAH_DATA_DIR=<data dir> python pipeline/benchmark/selftest.py --data`, check `test_data_hazard_fit`. It
prints the counts, the penalties, the intercept, the coefficients with each feature's mean and SD, the four AUCs, c, the
decision and each cell's count and median; it asserts that they are finite, that a second fit and the cache give the
same bytes, and that the counts add up (it does not assert how large the AUCs are, because a no-go would have been a
result too). **Worked out by hand from those and not printed:** the rates per row and "1 in 642", the odds columns, the
Hanley-McNeil standard errors, the AUC differences ("what the features add") and the SE multiples under "So"; the frames
per half are A07's spans, and the timings and the cache's size come from timing `load()` apart from the check. The code
is checked on hand-built objects by `test_calibration_v2_synthetic` and the `test_calibration_hazard_*` checks.

**What was fitted.**
- **At risk:** every ball-visible frame of the spans from each kick-off to the end of its graded block (the spans of
  A07, and its clock: the minute counts from the kick-off), as the frame `r` whose features say how likely the *next*
  frame is to start a gap. **Event:** frame `r + 1` starts an *interior* gap (A07's definition: a visible frame follows
  it inside the span).
- **Not at risk, counted:** the frame before a gap that runs to the span's end (an edge gap: neither an onset nor a
  non-onset), the last frame of a span if the ball is seen on it, and a frame with an undefined feature (the ball's
  speed before its first step; nobody present).
- **Features** (`hazard.features`, in metres, from BAR-ATH's own ball and 46 slots): distance to the nearest line
  (negative beyond it), the beyond-the-line flag, the ball's speed (mean of its last 5 step speeds), the distance to the
  nearest player, the players within 3 m, and the minute of the half. Standardised by their mean and SD on the rows
  fitted.
- **Fit:** L2-penalised logistic regression, `l2 = 1` on the standardised features, the intercept unpenalised; a fit
  that does not converge would be refitted with `l2` times 10, at most 3 times, and the `l2` used recorded. **None
  needed it: all five fits (the pooled one and the four cross-half ones) converged at `l2 = 1`.**

| | frames | rows used | onsets | rate per row |
|---|---|---|---|---|
| half 1 (from the kick-off) | 79,467 | 47,161 | 67 | 0.00142 |
| half 2 (from the kick-off) | 76,456 | 44,679 | 76 | 0.00170 |
| both | 155,923 (91,844 with the ball) | 91,840 | 143 | 0.001557 (1 in 642) |

The 4 visible frames not used: 2 before an edge gap (the ball is missing at the end of both halves' graded blocks), 0
last frames, and 2 with an undefined feature (the first visible frame of each half has no step yet, so no ball speed;
neither is an onset). So the 143 onsets are the 143 interior gaps of A07.

**The model** (both halves pooled; coefficients are on the standardised features; intercept -6.894113980):

| feature | coefficient | odds x per SD | per unit | mean | SD (scale) |
|---|---|---|---|---|---|
| `line_distance_m` | -0.394332679 | 0.674 | x0.959 per m | 13.81 | 9.51 |
| `beyond_line` | +0.120581120 | 1.128 | x1.84 if beyond | 0.041 | 0.198 |
| `ball_speed_mps` | -0.692766907 | 0.500 | x0.883 per m/s | 6.58 | 5.58 |
| `nearest_player_m` | -0.026116903 | 0.974 | x0.993 per m | 3.04 | 3.74 |
| `players_within_3m` | +0.220885663 | 1.247 | x1.339 per player | 0.848 | 0.756 |
| `minute_of_half` | +0.329593166 | 1.390 | x1.022 per minute | 22.76 | 15.18 |

The line and slow-ball signs are those of the plan's measurement of the gap onsets (A01 does not report it): 28.6% of
the onsets (42 of 147) have the ball within 2 m of a line or beyond it, against 8.5% of all visible frames, and the
median ball speed at an onset is 2.2 m/s against 5.1 m/s (A07 re-measures the near-line share: 28.57% of the interior
gaps on the graded blocks, 27.97% from the kick-offs). The sign of the minute is A02's: the ball is missing more
through each half. A crowd around the ball (more players within 3 m) raises the hazard too; the distance to the nearest
player adds nothing once the others are in.

**Cross-half AUCs** (fitted on one half, scored on the other; the same procedure for a model of `minute_of_half` alone;
standard errors by Hanley-McNeil, approximate):

| fitted on, scored on | onsets scored | full model | time only | what the features add |
|---|---|---|---|---|
| half 1, half 2 | 76 | **0.7110** (SE 0.034) | 0.5612 (0.034) | +0.150 |
| half 2, half 1 | 67 | **0.6475** (SE 0.037) | 0.6260 (0.037) | +0.022 |

**c = 1.000000000.** It is the onsets per row (143 / 91,840) over the model's mean probability on the same rows, and
that is 1 up to the 1e-9 rounding for any penalty, because the intercept is not penalised: the model's mean probability
is the base rate. The L2 shrinkage changes who gets the probability, not how much there is. So `c` only does something
on a match whose rows have another rate, which is what the `context` profile's one-time rescale is for.

**Gap durations by onset class and third of the half** (interior gaps, seconds; class 1 = the onset is within 2 m of a
line, 40 of the 143; thirds are minutes 0-15, 15-30 and 30 on, from the kick-off). The rule that a cell with fewer than
5 durations uses its class's durations pooled over the thirds was fixed in advance. **No cell fell back: the smallest
has 11.**

| | third 0 | third 1 | third 2 | class |
|---|---|---|---|---|
| class 0 (away from the lines), n / median | 23 / 6.76 | 30 / 8.42 | 50 / 11.16 | 103 |
| class 1 (near a line), n / median | 13 / 14.64 | 11 / 19.40 | 16 / 12.24 | 40 |

**Decision: go.** The rule, set in the plan before the fit, is both full-model cross-half AUCs at least 0.60: 0.7110 and
0.6475 are. Nothing was tuned to pass it (one penalty, no feature chosen). So the `context` profile goes ahead.

**So:**
- The signal is real but modest, and it is not equally clear in both directions. 0.7110 is 3.3 SE above 0.60; 0.6475
  only 1.3 SE above it, on 67 onsets. On that split the minute alone scores 0.626, so the five other features add 0.02
  there, inside the noise (and 0.15 on the other split, beyond it). The model is a coarse placement of the gaps (near
  lines, on slow balls, later in the half), not a predictor of the next one: AUC 0.65 to 0.71.
- Near-line gaps last longer (median 12 to 19 s against 7 to 11 s), and class 0 gaps lengthen through the half (6.8,
  8.4, 11.2 s), as A02's coverage worsens; the sampler draws a gap's length from the cell of its onset's class and
  third.
- Everything the `context` profile needs is in the Calibration: `hazard_model`, `hazard_l2`, `hazard_c`, `hazard_auc`,
  `hazard_go`, `hazard_durations_s` (seconds; convert with the DFL fps), `hazard_fallback_cells` (none), and the counts
  as `hazard_*` scalars. The model is applied to the features of the *degraded* benchmark players and ball, at the
  at-risk frame, and `sample_mask` takes its probability unshifted.

### A09 — The tracking profile's first part: the parameters from our match, and its distance from ours (2026-10-23)

Proposal: P01 · Commit: `66cae73` (the commit that adds this entry) · Command: `python
pipeline/benchmark/selftest.py --dfl --data`, checks
`test_data_tracking_profile_on_a_real_match_with_our_real_calibration` (prints d, the mean spans, the second table and
the lost share and the slot persistence below, and asserts that the lost share pooled over the matches is d to
within 0.003) and `test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches`; they take 16 s and 8 s once the
caches are warm. The Calibration is version 2 (A07, A08), read from its cache; the profile is `tracking` version 1
(`degrade.py`), **work in progress: no gates, no yardstick**. Aggregates only; nothing here is a row of BAR-ATH. Worked
out apart from the checks and not printed by them, by throwaway scripts that read the Calibration, the ATD cache and
the seven matches and print aggregates: the frame counts and the lengths of the first table, the spread of the tracklet
median over matches and seeds, and the slot usage of ours.

**What it draws from, measured on our kick-off spans** (both halves, 155,923 frames):

| | value | how |
|---|---|---|
| tracklets of ours | 1,144 | `Calibration.tracklet_lengths_s`: maximal runs of one slot being present |
| their lengths | median 41.3 s (1,032 frames), mean **111.9 s (2,797.1 frames)**, shortest 0.32 s (8 frames), longest 2,937 s (73,427 frames) | the shortest is 8 frames, so the rule that a tracked span is at least 1 frame is never used on ours |
| frames with more than 18 slots present | 139,360 (89.4%) | the tail, 10.6% of the frames with 18 or fewer, is left out of d: it is the profile's next part |
| mean slots present over them | 21.4500 | |
| **d**, the share of an on-pitch player's frames that are lost | **0.0250** (0.025000) | 1 - 21.4500 / 22 |
| **mu**, the mean lost span | **71.7 frames (2.87 s)** | 2,797.1 * d / (1 - d) |

d counts the frames of ours with 23 or more slots present (more slots than players: 3,460 frames, 2.2% of the 155,923,
2.5% of the 139,360 above) at their full count, so it is a little smaller than the share lost would be if the counts
were capped at 22: **d is 0.0250 as computed and 0.0262 with the counts capped**. That is immaterial next to the draw's
own spread (the lost share is 2.27% to 2.80% by match and seed, below), and d is kept as it is.

**What it makes** (J03WMX, seed 0; `calibration.player_stats` on the degraded match, from each kick-off to the end of
its graded block, beside the Calibration's; not gated):

| | degraded | ours |
|---|---|---|
| slots present, mean | 21.38 | 20.52 |
| 23 or more present | 0.0% | 2.2% |
| all 22 present | 54.0% | 50.2% |
| 19 to 21 present | 45.5% | 37.0% |
| 15 to 18 present | 0.5% | 5.4% |
| 14 or fewer present | 0.0% | 5.2% |
| tracklets (maximal runs of a slot) | 1,289 | 1,144 |
| tracklet median | 35.3 s | 41.3 s |
| NaN speeds, of the present slot-frames | 0.04% | 7.83% |
| wrong-team frames | 0.0% | 10.5% |

- **The present histogram is ours where the profile has made it ours, and not elsewhere.** Over the frames of ours with
  more than 18 present the mean is 21.45; the degraded match has 21.38 over all its frames, 0.07 below that, because
  this seed lost 2.80% of the on-pitch frames and not 2.50% (below). What is missing is the tail: 10.6% of our frames
  have 18 or fewer present, 5.2% have 14 or fewer, and the degraded match has 0.5% and 0.0%. That is the next part's,
  not a flaw of this one. It is most of why the two upper buckets (22, and 19 to 21) are fuller than ours, 99.5% of
  the frames against 87.2%, a gap of 12.3 points: the 10.6% that ours has below them (5.4% and 5.2%, against 0.5% and
  0.0% here) explains 10.1 of them, and the other 2.2 points are the 2.2% of our frames with 23 or more present, which
  the degraded match does not have (0.0%: each player is tracked once at a time).
- **The lost share is d, a little over, and the draw moves it.** The share lost of the on-pitch frames, pooled over the
  7 matches and seeds 0 to 4, is **2.59%** against d = 2.50%; by match 2.47% to 2.66%, and by match and seed 2.27% to
  2.80% (J03WMX's seed 0 is the top of that). A DFL half is 68,000 to 78,000 frames and a tracked span of ours has a
  mean of 2,797, so a run has about 25 cycles and the cut at its end is not negligible: the profile loses 0.09 points
  more than d (4% of d), and that is not corrected.
- **The tracklets are as many as ours, and a little shorter.** 1,026 to 1,301 per match and seed (35 runs) against
  1,144. The median of their lengths is 39.4 s pooled (41,181 tracklets) against 41.3 s, and 34.8 s to 49.8 s over the
  35 runs (SD 3.9 s). The 2 s under ours may be the cuts at the ends of the runs (not isolated). The 35.3 s of the table
  is one seed, and of slot runs: the profile's tracklets are 1,301 on that run and the table's 1,289 are slot runs,
  because where one tracklet ends on the frame that another takes its slot the two read as one run. It is inside the
  spread.
- **NaN speeds** are on the first frame of each tracklet and nowhere else (0.04% of the slot-frames); ours come in runs
  (A07), 7.83%. **Wrong-team frames** are none. Both are the next part's.
- **Slots, and what a slot's index still says.** The lowest free slot keeps a team in its first 11 slots all match (11
  + 11 slots used on J03WMX), where ours has 21 of its 24 home slots and 20 of its 22 away slots present on at least 10%
  of the frames (22 and 21 on at least 1%; the median slot is present on 43.5% of the frames, home, and 46.1%, away):
  ours rotates through nearly all of them as the tracker hands slots on. A slot's index is not as empty of the player
  as in ours, which the first version of this entry said it was. Measured on J03WMX, seed 0, by the same two
  functions on both sides (an *idle gap* is a maximal run of absent frames of one slot with a present frame of that
  slot on both sides, inside a half from its kick-off to the end of its graded block):

  | | degraded | ours |
  |---|---|---|
  | slot idle gaps, count | 1,245 | 1,065 |
  | median | 1.96 s | 7.40 s |
  | at most 3 s | 67.1% | 17.7% |
  | over 60 s | 0.0% | 15.0% |

  A *return* is a player's tracklet that starts after a lost span of his (not on the frame his last one stopped, which
  is a handover). Of the degraded match's 1,247 returns, **948, 76.0%, land back in the slot he had**; ours has no
  identities, so there is no such figure on its side. So no evidence source reads a slot's identity today, beyond
  `nearest_player`'s `argmin` on exact ties, which goes to the lower slot; but **a method that follows a slot across a
  loss would be validated optimistically on this profile**: here a player comes back mostly to his own slot, and a
  slot is idle for a median of 2.0 s where ours idles for 7.4 s, nearly four times as long, and 15.0% of ours run over
  a minute (what is measured is the player's side, 76.0%, and the gaps' lengths; how often the next tracklet in an
  idle slot is the same player is not). The tail's part of the profile re-measures this.

**No pool runs out.** On each of the 7 matches at most 11 players of a team have a position on any frame of the graded
halves, and under parameters that make a slot change hands as often as it can (tracked spans of 1, 2, 5, 25 or 250
frames, 30% of the time lost; 36,594 to 40,056 tracklets per match) and under those that lose nobody, 3 seeds, no team
uses more than 11 slots. A pool of 24 or 22 slots has room for about twice that.

**What this does not say.** It is the first of three parts of the profile, so none of these figures is a gate, a miss or
a pass; the gates come with the last part, centred on the Calibration's own values, and the yardsticks' scores on
`tracking` are not run. The shape of the tail (frames with few players tracked), the wrong-team episodes and the
NaN-speed runs are what the next two parts bring. Its figures here are a baseline for them: they say what the profile
looks like before those parts are added, and by how much each is short of ours.

*Later:* the wrong-team episodes, the NaN-speed runs and the smoothing were added in Task 13, which also ran the gates
and the yardsticks (R004); the tables above are of the profile as it was at this entry, before them.

### A10 — Headroom: what perfect evidence would be worth to the chosen decoder (2026-10-25)

Proposal: P06 (it sizes P04, P05 and P07) · Commit: `bdd699b` (the code both runs were made at, from a detached
checkout; the runner's header says `bdd699b082e6`; the entry is in the commit that adds it) · Commands, with
`MTAH_DATA_DIR` at the main checkout's `data/`:
`python pipeline/benchmark/headroom.py pipeline/configs/p06_hmm.toml --profile tracking --seeds 3 --truth raw`
and the same with `--truth coarse`, side by side (the coarse run started 20 s after the raw one; the five variant
configs are the same files under both truths, which the runner allows). Neither passed `--write`, so no report was
written and the two logs are the record; both end `EXIT=0`. Public: not applicable, no BAR-ATH frame is decoded or
scored. Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4
graded minutes per seed. Aggregates only; every figure is on a log line, follows from two of them, or is a computation
said so where it is made.

**What was run.** The reference is `pipeline/configs/p06_hmm.toml` (R007's chosen config: switch_team 8, switch_dead 32,
min_possession 75, min_dead 1, `nearest_player` at confidence 0.78 and `ball_presence` at 0.2 and 0.585), and five
variants of it whose evidence is the benchmark's own truth (`pipeline/benchmark/headroom.py`, its README's "Headroom"
section; the truth reaches the oracles by closure and never through the inputs):

| variant | evidence |
|---|---|
| `p06_hmm__alive` | `p06_hmm`'s own, plus `oracle_alive`: a perfect dead-ball detector (P05's ceiling) |
| `p06_hmm__home_visible` | `nearest_player` replaced by `oracle_home_visible`: the true possessor on the frames where the degraded tracking has the ball, nothing where it does not (P04's ceiling) |
| `p06_hmm__home_all` | `nearest_player` replaced by `oracle_home_all`: the true possessor on every live frame, the ball seen or not (P04 and P07) |
| `p06_hmm__alive_home_visible` | both of the above: what remains once `D` is right and the possessor is right where the ball is seen |
| `p06_hmm__alive_home_all` | both, whole: the truth itself as evidence |

The oracles are exact (`oracle_alive` is 1 on the truth's `H` and `A` and 0 on its `D`; the home oracles are 1 on `H`, 0
on `A` and NaN elsewhere, dead frames included) and of weight 1, so to the decoder a perfect source is about 13.8 nats a
frame at `eps` 1e-6. The decoder, its penalties, its minimum run and the config's own `ball_presence` are unchanged
(next to `oracle_alive` the `ball_presence` counts for 1.4 nats at most against 13.8; in the home variants it is the
only evidence for `D`). The headroom is that of this config, not of the benchmark.

**Checks on the runs.** Both logs end `EXIT=0`. The `p06_hmm` block of each is the same as the one in the control logs
of R007 (every printed field, both truths) and so R007's chosen config. On the 10 comparison lines the runner printed
(each variant against `p06_hmm`, per truth), a recomputation from the 4-decimal per-match scores with the benchmark's
own `t_interval` and `sign_flip_p` reproduces the pooled Δ to 0.0001, the mean and the interval ends to 0.0001, the p,
and the wins and losses exactly. The paired steps below are recomputed the same way.

**The scores.** Δ against `p06_hmm` is the runner's own line: the pooled Δ, the mean per-match Δ with its 90%
t-interval, the sign-flip p, wins and losses over the 7 matches. Possessions per minute, `D` share and alive/dead F1
are the config's, so they depend on the truth here only through the oracle. The true `D` share is 44.8% of the graded
frames and the true possessions per graded minute 4.58 raw and 2.48 coarse.

Raw truth:

| config | pooled spellF1 | macroF1 | alive/dead F1 | Δ spellF1, pooled | mean per-match Δ [90% t] | sign-flip p | wins / losses | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` (the chosen config) | 0.2053 | 0.4569 | 0.5676 | | | | | 3.10 | 43.1% |
| `__alive` | 0.3678 | 0.8147 | 0.9997 | +0.1625 | +0.1632 [+0.1526, +0.1738] | 0.0156 | 7 / 0 | 2.59 | 44.8% |
| `__home_visible` | 0.3586 | 0.5230 | 0.5558 | +0.1533 | +0.1532 [+0.1437, +0.1627] | 0.0156 | 7 / 0 | 2.41 | 45.1% |
| `__home_all` | 0.3594 | 0.5242 | 0.5572 | +0.1541 | +0.1540 [+0.1445, +0.1634] | 0.0156 | 7 / 0 | 2.43 | 45.0% |
| `__alive_home_visible` | 0.5655 | 0.8905 | 0.9980 | +0.3602 | +0.3620 [+0.3404, +0.3837] | 0.0156 | 7 / 0 | 2.46 | 44.6% |
| `__alive_home_all` | 0.7627 | 0.9763 | 0.9957 | +0.5573 | +0.5592 [+0.5329, +0.5856] | 0.0156 | 7 / 0 | 3.19 | 44.8% |

Coarse truth:

| config | pooled spellF1 | macroF1 | alive/dead F1 | Δ spellF1, pooled | mean per-match Δ [90% t] | sign-flip p | wins / losses | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` (the chosen config) | 0.1967 | 0.4557 | 0.5676 | | | | | 3.10 | 43.1% |
| `__alive` | 0.4417 | 0.8153 | 0.9997 | +0.2450 | +0.2449 [+0.2309, +0.2589] | 0.0156 | 7 / 0 | 2.59 | 44.8% |
| `__home_visible` | 0.3651 | 0.5406 | 0.5775 | +0.1684 | +0.1676 [+0.1585, +0.1767] | 0.0156 | 7 / 0 | 1.95 | 43.4% |
| `__home_all` | 0.3658 | 0.5421 | 0.5793 | +0.1691 | +0.1683 [+0.1598, +0.1769] | 0.0156 | 7 / 0 | 1.98 | 43.2% |
| `__alive_home_visible` | 0.7416 | 0.9126 | 0.9984 | +0.5449 | +0.5448 [+0.5322, +0.5575] | 0.0156 | 7 / 0 | 1.84 | 44.7% |
| `__alive_home_all` | 0.9226 | 0.9925 | 0.9968 | +0.7259 | +0.7256 [+0.7092, +0.7421] | 0.0156 | 7 / 0 | 2.20 | 44.8% |

The macroF1 Δ against `p06_hmm` (all with p 0.0156 and 7 wins) is, raw / coarse: `__alive` +0.3578 / +0.3596,
`__home_visible` +0.0661 / +0.0849, `__home_all` +0.0673 / +0.0864, `__alive_home_visible` +0.4336 / +0.4568,
`__alive_home_all` +0.5195 / +0.5368.

**The steps between the variants.** The runner prints only the comparisons against `p06_hmm`, so these are **recomputed
here from the 4-decimal per-match scores of the logs** with the benchmark's own `t_interval` and `sign_flip_p`; a pooled
Δ is the difference of the two variants' pooled Δ against `p06_hmm`, good to ±0.0001 (R006 explains the rounding), and
a p can differ by a step from one on unrounded scores. A positive Δ means the second variant is better.

| step | raw: pooled Δ | mean per-match Δ [90% t] | p | wins / losses | coarse: pooled Δ | mean per-match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `__alive` over `__home_visible`: a perfect `D` against a perfect visible possessor (P05's ceiling against P04's) | +0.0092 | +0.0100 [-0.0063, +0.0263] | 0.2500 | 4 / 3 | +0.0766 | +0.0773 [+0.0594, +0.0952] | 0.0156 | 7 / 0 |
| `__home_all` over `__home_visible` (no `alive` in either) | +0.0008 | +0.0008 [-0.0004, +0.0019] | 0.3125 | 4 / 2 | +0.0007 | +0.0008 [-0.0007, +0.0023] | 0.3594 | 4 / 3 |
| `__alive_home_all` over `__alive_home_visible` (`alive` in both) | +0.1971 | +0.1972 [+0.1879, +0.2066] | 0.0156 | 7 / 0 | +0.1810 | +0.1808 [+0.1702, +0.1914] | 0.0156 | 7 / 0 |
| `__alive_home_visible` over `__home_visible`: a perfect `D` added to a perfect visible possessor | +0.2069 | +0.2088 [+0.1863, +0.2313] | 0.0156 | 7 / 0 | +0.3765 | +0.3773 [+0.3584, +0.3961] | 0.0156 | 7 / 0 |
| `__alive_home_visible` over `__alive`: a perfect visible possessor added to a perfect `D` | +0.1977 | +0.1988 [+0.1835, +0.2142] | 0.0156 | 7 / 0 | +0.2999 | +0.3000 [+0.2913, +0.3087] | 0.0156 | 7 / 0 |
| `__alive_home_all` over `__home_all`: a perfect `D` added to the possessor everywhere | +0.4032 | +0.4052 [+0.3760, +0.4345] | 0.0156 | 7 / 0 | +0.5568 | +0.5573 [+0.5360, +0.5786] | 0.0156 | 7 / 0 |
| `__alive_home_all` over `__alive`: the possessor everywhere added to a perfect `D` | +0.3948 | +0.3960 [+0.3769, +0.4152] | 0.0156 | 7 / 0 | +0.4809 | +0.4808 [+0.4697, +0.4918] | 0.0156 | 7 / 0 |

**What the tables say**:
- **Every variant beats the chosen config in all 7 matches under both truths** (p 0.0156, the smallest 7 matches allow;
  the interval of each Δ is far from 0), so the question is size and not whether.
- **A perfect dead-ball detector (P05's ceiling) is the largest single lever under the coarse truth, and level with a
  perfect possessor where the ball is seen (P04's ceiling) under the raw one.** Raw: `__alive` +0.1625 (0.3678)
  against `__home_visible` +0.1533 (0.3586), 0.0092 apart; the paired step in the table above (`__alive` over
  `__home_visible`) is +0.0100 per match [-0.0063, +0.0263], p 0.2500, 4 wins and 3 losses: no difference. Coarse:
  +0.2450 (0.4417) against +0.1684 (0.3651), 0.0766 apart; the step is +0.0773 per match [+0.0594, +0.0952], p 0.0156,
  7 wins and 0 losses. Both comparisons are tilted towards P05: on `tracking` `ball_presence` carries almost no `D`
  information (caveat below), so `__alive` has more to gain here than it would on our match, and `__home_visible`'s
  gain does not depend on it. Each of the three single sources is worth 27.5% to 29.2% of the full oracle's +0.5573
  raw and 23.2% to 33.8% of its +0.7259 coarse. On the tiebreaker, `D` is the lever: macroF1 0.8147 and 0.8153 with a
  perfect detector (+0.36), against +0.066 to +0.086 for the possessor.
- **A perfect possessor alone does not touch `D`, and the gaps stay `D`.** `__home_visible` and `__home_all` have
  alive/dead F1 0.5558 and 0.5572 raw, 0.5775 and 0.5793 coarse, against the chosen config's 0.5676 (-0.0118 and -0.0104
  raw; the oracle differs by truth, so the figure does too), and 2.41 and 2.43 possessions per minute raw against the
  true 4.58 (1.95 and 1.98 coarse, against 2.48). The decoder reads the gaps as `D` and cannot place a possession in
  them, whoever has the ball.
- **A perfect possessor in the gaps adds nothing until `D` is right, and then it adds a great deal.** `__home_all` over
  `__home_visible` is +0.0008 raw (+0.0007 coarse; p 0.3125 and 0.3594, 4 wins and 2 losses raw, 4 and 3 coarse:
  nothing), the possessor on the frames with no ball being worth nothing. With a perfect `D` in both, the same step is
  +0.1971 raw and +0.1810 coarse, 7 wins in 7. The arithmetic of the evidence says why. On a missing frame
  `ball_presence` (0.2) puts `D` ahead of either team by log 4 = 1.386 nats; a perfect possessor adds log 2 = 0.693 to
  the true team (and about -13.1 to the other) and nothing to `D`, so `D` stays ahead by 0.693 nats a frame, and a gap
  longer than about 92 frames (3.7 s; 2 x `switch_dead` of 32 at 0.693 nats a frame) decodes as `D` whatever the team
  evidence says, the gaps' median being 10.7 s (A01). The team evidence inside the gaps, which is all of P07, is never
  read, unless an `alive` source overrules `ball_presence`: the oracle's 13.8 nats a frame does, and then the possessor
  in the gaps is worth +0.1971.
- **The two sources are worth more together than apart.** Raw: `__alive` and `__home_visible` alone are +0.1625 and
  +0.1533, +0.3158 together if they added; both give +0.3602, +0.0444 more. With the possessor everywhere, +0.1625 and
  +0.1541 would be +0.3166, and both give +0.5573, +0.2407 more. Coarse: +0.4134 against +0.5449 (+0.1315 more), and
  +0.4141 against +0.7259 (+0.3118 more). Without a possessor in the live stretches of the gaps the decoder has nothing
  to place a change of team on there, and puts it on the stretch's first frame (the design's tie rule), so a perfect
  `D` still leaves the hand-overs inside the gaps wrong (`__alive_home_visible`, 0.5655 raw); the possessor everywhere
  is what fixes them.
- **The decoder's own cap, at these parameters, is below the truth.** With the truth as its evidence the chosen config
  scores 0.7627 raw and 0.9226 coarse (alive/dead F1 0.9957 and 0.9968), predicting 3.19 possessions per minute against
  the true 4.58 raw, and 2.20 against 2.48 coarse. The README's "exactly 1.0" is for penalties near 0 and minimums of
  1; this config has switch_team 8, switch_dead 32 and a minimum possession of 75 frames (3 s), which merges the
  shortest of DFL's possessions (median 3.7 s, A06), and costs 0.2373 raw and 0.0774 coarse. The logs cannot split it
  between the penalties and the minimum run. R007 chose those parameters under noisy evidence, to suppress its
  flicker, so the full-oracle row is a cap on this config and not on the approach; how much a decoder retuned to
  better evidence would add on the partial oracles is not measured. A better source calls for re-running at least the
  minimum-run part of P06's protocol.

**Caveats.**
- **The oracles are noise-free and exact.** They are a bound on what the evidence could give this config and its
  penalties, and not an estimate of what a real source will: no real detector or possessor is perfect, and these two
  need no ball, which a real one would. Only the endpoints are measured. What a detector at an alive/dead F1 of 0.8, or
  a possessor right on 90% of the visible frames, would be worth is not on any line here; a graded oracle (the truth
  flipped at random to a target F1) would give the slope, and nobody has built it.
- **`tracking`'s gaps are blind to the play.** The masks are transplanted from our match with their pattern and not
  their clock (R003), so `ball_presence` carries nearly no information about `D` on `tracking`: the ball is visible on
  56.8% of its dead frames against about 59% of all graded frames (R005's WARN line, and our match's 41.1% missing),
  where the public window gives a ratio of 5.2 for dead (A01; the source uses 4 by rule). The config's `D` therefore
  starts from almost nothing on the benchmark, and `__alive`'s +0.1625 overstates what a perfect detector would add to
  the real decoder, on our match. The same blindness says nothing about P07's real gaps, which start at the lines and
  on slow balls (A07, A08). `context` places the gaps by the play and is where this would be read; A10 was not run on
  it (it fails two gates, R005, and would be a sensitivity figure).
- **DFL's grain (T33).** Both truths are shown because the headroom depends on it: the raw truth has 4.58 possessions a
  minute and the coarse one 2.48, the dead-ball detector's ceiling is +0.1625 against +0.2450, and the full oracle's cap
  is 0.7627 against 0.9226. How fine our GT is is not known, so no row is a forecast for the leaderboard.
- **Seven matches, five of them one club's home games, three seeds shared by all the variants.** The intervals are over
  the matches. The headroom is that of one config at one set of parameters.

**What it changes.** P05's ceiling is the largest under the coarse truth and level with P04's under the raw one
(+0.0092, p 0.2500), and P05 is the one that makes the gaps readable to P07; a perfect P07 on top of a perfect P04 is
worth +0.0008 raw to this decoder, and +0.1971 once `D` is right; and any new source calls for re-running the
minimum-run choice. The recommendation that follows is in [proposals.md](proposals.md#after-p06). It stands on the
coarse-truth ceiling, on P05 gating P07, on `D` being the chosen config's weakest point and on P04 running alongside
it, and not on a raw-truth lead over P04.

### A11 — Our ground truth's timing against DFL's, from tracking events (2026-10-25, Tian)

Proposal: P02 · Commit: `9f06349` (the code all five runs were made at, from a detached checkout; the entry is in the
commit that adds it) · Commands, with the data directory at the main checkout's `data/`; the instrument is
`pipeline/benchmark/conventions.py`:

```bash
python pipeline/benchmark/conventions.py public
python pipeline/benchmark/conventions.py dfl --profile tracking --seeds 3 --truth raw       # and --truth coarse
python pipeline/benchmark/conventions.py compare --profile tracking --truth raw             # and --truth coarse
```

Public: the 20 changes of the public truth (12 of team, 4 into `D`, 4 out of it), timed on our real tracking.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, under the raw
and under the coarse truth, pooled over matches and seeds by the lags themselves, never as a median of medians. All
five logs end `EXIT=0`. This is a measurement and not a score: nothing is decoded, no configuration is compared and
nothing is tuned or shifted from it (T01). Aggregates only; every figure is on a log line, or follows from two of them
by arithmetic said where it is made.

**What was measured.** A06 measured DFL's flag against the ball leaving the pitch and left the same question on our
match open ("D timing against ours is unmeasured"). The instrument times every change of a truth's label against the
nearest of eight events that the tracking shows without any label, so that no LLM has to see the match. A change is
typed `team` (H to A or A to H), `into_dead` (H or A to D) or `out_of_dead` (D to H or A); a detector speaks only for
the types it applies to:

| detector | applies to | the event |
|---|---|---|
| `touch_start` | `team`, `out_of_dead` | the first frame of a kept `controlled_touch` episode of the new team (`main.toml`'s rule: 2.0 m, 6.0 m/s, 3 frames) |
| `line_out` | `into_dead` | the first frame of a run where the ball is visible and beyond a line (`outside_m` above 0) |
| `ball_lost` | `into_dead` | the first missing frame of a ball gap that opens inside the block |
| `ball_moves` | `out_of_dead` | the first frame at or above `STILL_SPEED` after at least 25 frames below it, from positions (T31) |
| `model_dead` / `model_alive` | `into_dead` / `out_of_dead` | where P05's `dead_ball_model` (R009, the file `main.toml` uses) alive crosses 0.5 down / up |
| `tag_start` / `tag_end` | `into_dead` / `out_of_dead` | the start / end of a Smart Tagging `SET PIECES` interval; BAR-ATH only |

The lag is `event - change` in frames (25 per second), so **a negative lag means the event came first**, and a positive
one that the label flipped first. The event taken is the nearest to the change, the earlier on a tie, within 5 s either
side, bounded by the neighbouring changes of any type (or the block's edges) so that a change is never matched to
another one's event; none in that range is "not found". The detectors run over the graded halves of each match and the
lags are read at the changes only: on BAR-ATH they find 1,189 `touch_start` episodes, 81 `line_out`, 149 `ball_lost`, 71
`ball_moves`, 248 `model_dead` and 247 `model_alive` events in the graded halves, and 59 `tag_start` and 59 `tag_end`
(every `SET PIECES` interval of the match's Smart Tagging: the tags are not clipped to the graded halves), of which the
public window's changes see only those within 5 s. On DFL the same code runs on the tracking as the pipeline sees it
under `tracking` v3 and no flag (`MatchInputs`), and the truth is read beside it only to find the changes. `model_dead`
and `model_alive` are scored on DFL by the model fitted without that match (out of fold, by the code's rule), and on
BAR-ATH by the one fitted on all seven.

**The public window, on our tracking.** Found is the share of the changes with an event in the window. A cell with
fewer than 5 changes found prints no quartile and no bound (the instrument's `MIN_SHOWN`: with so few they would be the
lags of single stoppages), so only the median is given, and for the gap only whether its interval excludes 0:

| type | detector | n | found | median | quartiles |
|---|---|---|---|---|---|
| `team` | `touch_start` | 12 | 10 (83%) | +24.0 frames (+0.96 s) | -4.5, +37.2 |
| `into_dead` | `line_out` | 4 | 2 (50%) | +37.0 (+1.48 s) | not shown |
| `into_dead` | `ball_lost` | 4 | 2 (50%) | +30.0 (+1.20 s) | not shown |
| `into_dead` | `model_dead` | 4 | 3 (75%) | +39.0 (+1.56 s) | not shown |
| `into_dead` | `tag_start` | 4 | 0 (0%) | none | |
| `out_of_dead` | `touch_start` | 4 | 4 (100%) | +9.0 (+0.36 s) | not shown |
| `out_of_dead` | `ball_moves` | 4 | 0 (0%) | none | |
| `out_of_dead` | `model_alive` | 4 | 2 (50%) | -36.5 (-1.46 s) | not shown |
| `out_of_dead` | `tag_end` | 4 | 1 (25%) | +19.0 (+0.76 s), one change | not shown |

**DFL, on the degraded copy, under each truth.** Frames, with the median in seconds. `n` counts each change once per
seed, so it is three times the changes: the raw truth has 3,760 (2,351 team, 710 into `D`, 699 out of it) and the coarse
one 2,358 (949 team; it absorbs short possessions and leaves `D` as it is, A06), as the per-match lines of the logs add
up and the pooled `n` divide by 3. Raw truth:

| type | detector | n | found | median | p25 | p75 | median of each match |
|---|---|---|---|---|---|---|---|
| `team` | `touch_start` | 7,053 | 50% | -1.0 (-0.04 s) | -8.0 | +12.0 | -2.0 to -1.0 |
| `into_dead` | `line_out` | 2,130 | 47% | -3.0 (-0.12 s) | -5.0 | -1.0 | -4.0 to -3.0 |
| `into_dead` | `ball_lost` | 2,130 | 18% | +28.0 (+1.12 s) | -25.0 | +70.5 | +21.0 to +38.5 |
| `into_dead` | `model_dead` | 2,130 | 93% | +1.0 (+0.04 s) | -1.0 | +2.0 | +1.0 to +2.0 |
| `out_of_dead` | `touch_start` | 2,097 | 44% | +35.0 (+1.40 s) | +19.0 | +51.2 | +27.0 to +38.0 |
| `out_of_dead` | `ball_moves` | 2,097 | 56% | -3.0 (-0.12 s) | -3.0 | -3.0 | -3.0 in all seven |
| `out_of_dead` | `model_alive` | 2,097 | 96% | 0.0 (0.00 s) | -3.0 | +2.0 | 0.0 in six, -3.0 in J03WQQ |

Coarse truth (the same matches, seeds and profile):

| type | detector | n | found | median | p25 | p75 | median of each match |
|---|---|---|---|---|---|---|---|
| `team` | `touch_start` | 2,847 | 55% | 0.0 (0.00 s) | -7.0 | +25.0 | -1.0 to 0.0 |
| `into_dead` | `line_out` | 2,130 | 47% | -3.0 (-0.12 s) | -5.0 | -1.0 | -4.0 to -3.0 |
| `into_dead` | `ball_lost` | 2,130 | 19% | +23.0 (+0.92 s) | -36.0 | +67.0 | +4.0 to +36.0 |
| `into_dead` | `model_dead` | 2,130 | 93% | +1.0 (+0.04 s) | -1.0 | +2.0 | +1.0 to +2.0 |
| `out_of_dead` | `touch_start` | 2,097 | 48% | +37.0 (+1.48 s) | +21.0 | +59.0 | +34.0 to +39.0 |
| `out_of_dead` | `ball_moves` | 2,097 | 56% | -3.0 (-0.12 s) | -3.0 | -3.0 | -3.0 in all seven |
| `out_of_dead` | `model_alive` | 2,097 | 96% | 0.0 (0.00 s) | -3.0 | +2.0 | 0.0 in six, -3.0 in J03WQQ |

The coarse truth removes team changes (the `team` n falls from 7,053 to 2,847), so a window bounded by the neighbouring
changes reaches further. The cells whose events sit within a few frames of the change do not move (`line_out`,
`model_dead`, `ball_moves` and `model_alive` are the same to the digit under both truths). Those whose events sit tens
of frames away find more of them, a share 1 to 5 points higher, and move by 1 to 5 frames (`ball_lost` 5 earlier,
`touch_start` after a restart 2 later, `team` 1 later). A check on the copy: its `line_out` median, -3 frames, is A06's
4 frames (DFL's flag turning 0.16 s after the ball is first seen beyond a line, measured on its own positions) to within
a frame, so the instrument reproduces A06 on the copy.

**The gap: the public median minus DFL's**, with a 90% bootstrap interval of the public lags (2,000 draws; DFL's median
is taken as it is, so the interval carries the public side's sampling only). Where the public side has fewer than 5
changes found only whether the interval excludes 0 is printed:

| type | detector | raw truth: gap | coarse truth: gap |
|---|---|---|---|
| `team` | `touch_start` | +25.0 (+1.00 s), [-9.0, +40.0] | +24.0 (+0.96 s), [-10.0, +39.0] |
| `into_dead` | `line_out` | +40.0 (+1.60 s), excludes 0: yes | +40.0 (+1.60 s), excludes 0: yes |
| `into_dead` | `ball_lost` | +2.0 (+0.08 s), excludes 0: no | +7.0 (+0.28 s), excludes 0: no |
| `into_dead` | `model_dead` | +38.0 (+1.52 s), excludes 0: no | +38.0 (+1.52 s), excludes 0: no |
| `out_of_dead` | `touch_start` | -26.0 (-1.04 s), excludes 0: yes | -28.0 (-1.12 s), excludes 0: yes |
| `out_of_dead` | `model_alive` | -36.5 (-1.46 s), excludes 0: yes | -36.5 (-1.46 s), excludes 0: yes |

`tag_start` and `tag_end` have no DFL counterpart, and `ball_moves` found nothing on the public side, so those three
have no gap.

**What it says about when our GT flips** (each against DFL's flag, on the events our tracking sees):

- **Team changes: ours flips about a second before the new team's first controlled touch; DFL's flips at it.** On our
  tracking the new team's first controlled touch comes a median 24 frames (0.96 s) after our truth flips, in 10 of the
  12 changes (quartiles -4.5 and +37.2). On DFL's copy the touch comes 1 frame before the flag flips under the raw
  truth and on the same frame under the coarse one (quartiles -8 and +12, -7 and +25), so DFL's flag turns where
  control begins. The gap is +25 frames [-9, +40] raw and +24 [-10, +39] coarse. Both intervals hold 0, so **a
  difference is not shown**; it is the best-supported cell on the public side (10 changes found) and it points one way,
  with a spread (42 frames between the quartiles, against DFL's 20) wider than the gap itself. A flip at the moment the
  ball is won, before the controlled touch is confirmed on our tracking, would give this; so would a later start of
  controlled episodes on our tracker than on the copy. Nothing here separates them.
- **`D` starts early, or our tracker sees the stoppage late: the ball is seen beyond a line about 1.5 s after our truth
  turns to `D`, where DFL's flag turns just after it.** `line_out` is +37 frames (1.48 s) on the 2 of 4 stoppages where
  our tracking sees the ball beyond a line within 5 s, against -3 (-0.12 s) on DFL, and the model's alive crosses 0.5
  downward 39 frames (1.56 s) after our `D` begins on 3 of 4, against +1 frame on DFL, out of fold. The gaps are +40 and
  +38 frames, the same under both truths. They are on 2 and 3 changes, and one interval excludes 0 where the other does
  not, which is what 2 and 3 observations do and not a difference between the detectors. `ball_lost` agrees with DFL
  (+30 against +28 raw, gap +2 and +7) and carries nothing: the copy's ball gaps are our masks transplanted with their
  pattern and not their clock (R003), so where they fall is not placed by the stoppage.
- **`D` ends late, or the two trackings' cues differ: the restart's first controlled touch comes 9 frames after our
  truth turns alive, where DFL's comes 35 to 37 after its flag.** A first controlled touch is found at all 4 restarts, a
  median 9 frames (0.36 s) after our flip; on DFL's copy it follows the flag by 35 (raw) or 37 (coarse) frames (1.40 s,
  1.48 s), a gap of -26 and -28 frames. The model's alive crosses 0.5 upward 36.5 frames (1.46 s) before our truth turns
  alive on 2 of 4, against 0 frames on DFL (a gap of -36.5). Both detectors put our `D`'s end later than DFL's flag
  would be on the same events, by 1.0 to 1.5 s, and both intervals exclude 0 under both truths, on 4 and 2 changes.
  `ball_moves` finds nothing on any of the 4 (on DFL its flag turns alive 3 frames after the ball starts to move in at
  least half of the changes, p25 = p75 = -3.0, and every match's median is -3.0), so the sharper question, whether our
  `D` ends at the kick, is not answered. Why our tracking shows no ball that is still for 25 frames and then moves near
  these restarts is not explained here (a ball that is not visible, or not still).
- **Taken together, the four `D`-edge gaps put our stoppages longer than DFL's flag would draw them.** Relative to
  DFL, on the same kind of tracking events, `D` starts 1.5 s earlier and ends 1.0 to 1.5 s later: about 2.5 to 3 s a
  stoppage (the sum of the gaps above, 64 to 76.5 frames, both truths). The signs differ between the start and the end,
  so one constant offset between our tracking and the truth's frames (T08, T29) cannot be all of it, since it would
  move every gap the same way. That does not leave a difference in convention as the explanation: tracker coverage
  also gives opposite signs at the two ends (see "How strong", below).
- **Smart Tagging says nothing yet.** `tag_start` finds no `SET PIECES` interval within 5 s of any of the 4 stoppages
  and `tag_end` one of 4 (+19 frames, 0.76 s, a single change). A tag marks a window of a few seconds around an event
  (T17), so it is not an instant, and it is absent from most of these edges.

How strong: only the team cell has more than 4 changes, and its interval holds 0. The `D` findings rest on 2 to 4
changes each. The four `D`-edge gaps that carry anything (`line_out` +40 and `model_dead` +38 at the start,
`touch_start` after a restart -26 and `model_alive` -36.5 at the end, raw truth) all point to a longer `D`, and three of
them exclude 0. Three explanations fit, and these data do not separate them:
- **A difference in convention**: our truth turns `D` on earlier and off later than DFL's flag does.
- **Chance**: four stoppages, found on 2 to 4 changes each, from one lively stretch of the match.
- **Tracker coverage**, which predicts all four signs by itself. Our tracker loses the ball near the lines (T39: beyond
  a line on 4.07% of our at-risk rows against DFL's 17.68%), so on ours `line_out` and `model_dead`, which lean on the
  ball beyond a line, fire late (a positive gap at the start of `D`). With fewer cues that the ball is dead on ours, the
  model's alive rises early (a negative `model_alive` gap at the end). And the copy's ball gaps are our masks placed
  blind to the play (R003), which can delay the copy's first visible touch after a restart and inflate DFL's +35 (a
  negative `touch_start` gap at the end). It gives no sizes, so how much of 2.5 to 3 s it could carry is not known.

So they are **a lead and not a measurement**. The model-based pair mixes our truth's convention with the model's
transfer to our tracking (below); the model-free pair (`line_out`, `touch_start` after a restart) avoids the model's
transfer but not the coverage difference, since they are the two detectors most exposed to it, and rests on 2 and 4
changes. What would separate convention from coverage is a person's check on the video of where `D` begins and ends
against the ball and the whistle (below).

**Caveats.**
- **Four changes into `D`, four out of it, 12 of team.** Several `D` cells rest on 1 to 3 found changes (`line_out` and
  `ball_lost` 2, `model_dead` 3, `model_alive` 2, `tag_end` 1), so a median is close to a single observation (over two
  changes it is the midpoint of the pair), and an interval over so few says little. The quartiles of those cells and
  every bound of their intervals are withheld on purpose.
- **Two different trackings.** The detectors run on our real tracking for BAR-ATH and on DFL's positions degraded by
  `tracking` v3 for the benchmark, so a gap mixes our truth's convention with any difference between our tracker and
  the copy. For example, controlled-touch episodes may start later on our tracking than on the copy, which would show
  as a positive team gap with no convention behind it. And the copy's ball is missing on 41.9% of the graded frames
  (R004c's gate, 0.4189) while the public window's is missing on 11.8% (A01; 33.6% of its dead frames), so the copy's
  first visible touch after a restart may come later than it really does, which would inflate DFL's +35 and with it the
  negative `out_of_dead` gap. Found shares are partly that coverage (83% against 50% for team changes), and a median is
  only over the changes found.
- **The public window is unusually lively** (A06): alive on 80.4% of its frames, more than 96% of DFL's 5-minute
  windows, and the best-covered stretch of the match. Its 4 stoppages and 12 team changes are not a sample of the
  match's, and the restarts of a whole half (throw-ins, goal kicks, corners, free kicks in different shares) are not
  represented.
- **DFL tracks the ball out of play** (T39), which helps `line_out` and `model_dead` on the copy (the ball is beyond a
  line on 17.68% of DFL's at-risk rows against 4.07% of ours, R009), and our tracker loses it near the lines. A late
  `line_out` or `model_dead` on our tracking can be the tracker's, not the truth's. The model was fitted on DFL's flag,
  and its lag there, +1 and 0 frames out of fold, shows it reproduces DFL's timing; whether it does so on ours is what
  its found shares (75% and 50% against 93% and 96%) put in doubt (T39).
- **`tag_start` and `tag_end` exist only on BAR-ATH**, so they have no gap; Smart Tagging's tag values are match data,
  which no LLM may read, so no restart type is classified here (that stays with a person, below).
- **The DFL side is pooled over three seeds**, so its `n` is three times the changes, and its medians are taken as fixed
  numbers in the gap's interval.

**How a person can extend it.** `python pipeline/benchmark/conventions.py public` regenerates
`data/runs/conventions/public_transitions.csv` (gitignored, on our machines only): one row per change of the public
truth, with its frame (the tactical video's frame, row i = frame i), the label before and after, its type and each
detector's lag there, blank where the detector does not apply or found nothing. No LLM reads it, and no row of it is
quoted anywhere. Watching the video around those frames, and measuring around the published transitions, is allowed;
labelling frames of this match is not (T27). Things to watch at each change, and to write into the docs as aggregates
and frame counts only:
- at a change into `D`: whether `D` begins at the ball crossing the line, at the whistle or when play stops, and what
  kind of stoppage it is (a throw-in, a goal kick, a corner, a free kick, a foul, a goal, an offside);
- at a change out of `D`: the restart type, and whether `D` ends at the whistle, at the kick or at the first touch;
- at a team change: what happens at the flip (a tackle, an interception, a deflection, the old team's last touch or the
  new team's first), and whether the ball is visible there;
- the four restarts where `ball_moves` found nothing, and the stoppages where `line_out` and `tag_start` found nothing.

A frame-exact decode of the tactical video seeks back to a keyframe, at most 49 frames earlier (T23). More than 20
transitions would come from P16's sample games in our truth's provider's conventions, not from the public window.

**What it changes.** Nothing is adopted and no shift is derived from A11: its gaps are between detectors on two
sources, and the pipeline's own latency at these 20 changes is what the `shift_edges` rule reads (R012: it found none
of the three types passing the rule, so g = 0 and only the pipeline's lag on DFL is adopted, team +1, into `D` -2, out
of `D` 0). It says where to look first, the two `D` edges before the team flips, and in which direction (our `D`
earlier at the start and later at the end, our team flips earlier), which is T40: a component tuned on the benchmark
may carry DFL's timing.

*Later (23 Oct rule, `f8f2629`):* A11 is inspection. It measured how our truth's changes sit against events our tracking
shows, beside DFL's, and it adopted nothing, so its findings stand as understanding. Under the rule they may not set a
constant: no back-dating, `D` onset or shift is to be tuned to them. "How a person can extend it" stays a way to
understand: a person may watch those 20 changes to see how our truth flips and what kind of stoppage each is, and
nothing seen may become a label, a restart type or a number written down for those frames.

### A12 — What knowing who restarts is worth to main.toml, and how stoppages end (2026-10-26, Tian)

Proposal: P17 · Commit: `a4d1349` (the code all five runs were made at, on branch `p17-restart-team`; the oracles in
`headroom.py` are from `0bca76c`, `restarts.py` from `a4d1349`; the entry is in the commit that adds it) · Commands,
from the repo root, one process at a time on one BLAS thread at below-normal priority:

```bash
python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants restart restart_run --truth raw --write
python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants restart restart_run --truth coarse --write
python pipeline/benchmark/restarts.py anatomy pipeline/configs/main.toml --truth raw --out data/runs/benchmark/restarts/anatomy.raw.json
python pipeline/benchmark/restarts.py anatomy pipeline/configs/main.toml --truth coarse --out data/runs/benchmark/restarts/anatomy.coarse.json
python pipeline/benchmark/restarts.py bar-ath pipeline/configs/main.toml --out data/runs/restarts/bar_ath.json
```

Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, under the raw
and the coarse truth. All five logs end `EXIT=0`. Before them a one-thread check (`run.py main.toml --truth both
--no-write`) reproduced R013 exactly (0.5430 raw, 0.5593 coarse, alive/dead F1 0.9731), so the throttled runner changes
nothing. Part 1 of two: the ceiling and the anatomy. Part 2, the measured confidences of the real source, comes with it
(Phase 1b). Aggregates only: the benchmark is public DFL data, and the one number from our match is a count or a share of
`main.toml`'s own output (no truth for it exists). This is a measurement of a bound and nothing is adopted.

**What was measured.** P05's table and "After R013" item 4 say that after a stoppage `controlled_touch`'s hold favours
the team that had the ball before it, until the first new control episode names the team, and that a source for the
restarting team "would speak sooner". Nothing sized that. Two questions, before any source is built:

1. **The ceiling.** `headroom.py` gained `--variants` and two oracles that read the truth by closure (as the others do,
   registered only inside a run). A restart is every true `D` run in a graded half followed by `H` or `A` in the same
   half, `e` its first live frame and T the truth's team at `e`.
   - `oracle_restart` (variant `restart`, `main__restart`) is `main.toml` plus a source of `home` = 1 (T = H) or 0 (T =
     A) on `[e, min(e + 50, the truth's next change, the block's end))`, at weight 1, and NaN everywhere else, `alive`
     included: a perfect "which team restarts" vote for 2 s. It is the go/no-go number.
   - `oracle_restart_run` (variant `restart_run`) is the same to the truth's next change, which is the whole possession.
     Context only: it also hands the decoder where the first possession ends, which is not "who restarts".
2. **The anatomy** (`restarts.py anatomy`). The same restarts, with `main.toml`'s decoded labels on the benchmark's
   degraded copy. The **flip** is the share of restarts whose team differs from the truth's team on the frame before the
   stoppage (restarts with no team before it excluded). The **outcome** is read on the 50 frames after the true restart:
   *right* if `main.toml` labels more of them with the restarting team than with the other, *wrong* otherwise but with some
   `H`/`A` (a tie is wrong), *still D* if it labels none. On BAR-ATH (`restarts.py bar-ath`) it counts `main.toml`'s own
   predicted stoppages (a `D` run followed by `H` or `A` in a graded half) and whether the ball is missing on more than
   half of the 50 frames before and after.

**The ceiling**, pooled spellF1 on `tracking`, with the deltas against `main` (the interval is a 90% t-interval over
the 7 matches):

| config | raw spellF1 | coarse spellF1 | raw macroF1 | alive/dead F1 | possessions / min (raw) |
|---|---|---|---|---|---|
| main | 0.5430 | 0.5593 | 0.8895 | 0.9731 | 3.13 |
| main__restart | 0.5628 | 0.5790 | 0.8962 | 0.9742 | 3.19 |
| main__restart_run | 0.5839 | 0.6950 | 0.9037 | 0.9749 | 3.05 |

| delta against main | raw | coarse |
|---|---|---|
| `restart`, pooled | **+0.0198** | +0.0197 |
| `restart`, mean per match | +0.0196 | +0.0197 |
| `restart`, 90% t interval | [+0.0161, +0.0230] | [+0.0156, +0.0238] |
| `restart`, sign-flip p, wins-losses | 0.0156, 7-0 | 0.0156, 7-0 |
| `restart`, macroF1 | +0.0066 | +0.0074 |
| `restart_run`, pooled (context) | +0.0409 [+0.0356, +0.0461], 7-0 | +0.1357 [+0.1197, +0.1505], 7-0 |

Per match under the raw truth, `restart`: J03WMX +0.0235, J03WN1 +0.0255, J03WOH +0.0151, J03WOY +0.0182, J03WPY
+0.0241, J03WQQ +0.0161, J03WR9 +0.0143; by competition DFL-COM-000001 +0.0245, DFL-COM-000002 +0.0176.

**The anatomy**, raw truth (the coarse truth gave the same figures to the last digit: it absorbs short possessions and
leaves the runs next to `D` alone, A06). `restarts` counts each restart once per seed, so it is three times the
distinct ones (101, 105, 100, 111, 95, 97 and 90 per match, in the table's order). Right, wrong and still D are shares
of all restarts, then of the flips and of the non-flips:

| match | restarts | median stoppage (s) | flip | right | wrong | still D | right / flip | wrong / flip | right / same | wrong / same |
|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | 303 | 16.0 | 90.1% | 78.2% | 14.2% | 7.6% | 76.6% | 15.0% | 93.3% | 6.7% |
| J03WN1 | 315 | 24.5 | 70.2% | 69.2% | 19.7% | 11.1% | 68.5% | 21.0% | 69.9% | 17.2% |
| J03WOH | 300 | 18.9 | 65.7% | 81.3% | 11.7% | 7.0% | 83.1% | 11.3% | 77.5% | 12.7% |
| J03WOY | 333 | 19.5 | 65.5% | 85.0% | 9.9% | 5.1% | 81.9% | 11.6% | 90.4% | 7.0% |
| J03WPY | 285 | 22.8 | 68.4% | 75.1% | 16.8% | 8.1% | 69.7% | 20.0% | 86.7% | 10.0% |
| J03WQQ | 291 | 20.0 | 56.7% | 82.1% | 12.7% | 5.2% | 84.2% | 10.9% | 79.4% | 15.1% |
| J03WR9 | 270 | 20.0 | 75.6% | 79.6% | 13.7% | 6.7% | 78.9% | 14.2% | 81.8% | 12.1% |
| pooled | 2,097 | 19.8 | 70.3% | 78.7% | 14.1% | 7.2% | 77.3% | 15.0% | 81.6% | 12.1% |

**BAR-ATH, label-free.** `main.toml`'s output has 104 predicted stoppages followed by `H` or `A` (44 in the first half,
60 in the second). The ball is missing on more than half of the 50 frames before the restart for 39.8% of them, and on
more than half of the first 50 frames after it for 20.2%.

So:

- **The ceiling clears the declared bar.** A perfect 2 s vote for the restarting team is worth +0.0198 pooled raw
  spellF1 to `main.toml` (90% interval [+0.0161, +0.0230], 7 wins in 7, sign-flip p 0.0156, every match between +0.0143
  and +0.0255) and +0.0197 under the coarse truth (7 in 7), so the two truths agree, which they did not for R010's and
  R013's re-tunes. It also lifts macroF1 (+0.0066, +0.0074) and alive/dead F1 (0.9731 to 0.9742).
- **Go/no-go: GO.** The rule was declared in the design before any run, Tian's: stop if `main__restart`'s pooled raw
  delta is below +0.01, with the coarse figure reported beside it and not deciding. It is +0.0198, twice the bar, so
  Phase 1 (the source `restart_team`, from the players' arrangement) goes ahead.
- **The hold points the wrong way after most stoppages.** Seven in ten stoppages hand the ball to a different team than
  the one that had it (70.3%, from 56.7% to 90.1% a match), so a decoder that carries the team across a stoppage is
  wrong at the start of the next possession whenever the team flips. That is the "clear next team" condition that "After P07 and
  P02" item 4 asked for, and it holds in the sense that matters.
- **The pipeline already names the restarting team on most restarts, which is why the ceiling is +0.02 and not more.**
  On the 50 frames after a true restart `main.toml` is right on 78.7%, wrong on 14.1% and still in `D` on 7.2%, because
  the first control episode speaks quickly. The mistakes sit a little more on the flips (15.0% wrong against 12.1%
  without a flip), so the carried team is only part of them, and the stoppages' ends that `D` has not left are another.
  What a vote can earn is therefore the wrong 14.1% and some of the late 7.2%.
- **The 2 s oracle is the honest ceiling and `restart_run` is not.** To the next change it adds +0.0409 raw and +0.1357
  coarse, but it hands the decoder where the first possession ends. Under the coarse truth, whose possessions are
  longer, that is worth more than three times as much as under the raw one, so the difference between the two figures is a statement
  about the possession's end and not about who restarts. It is context only and decides nothing.
- **On our match the ball is missing at 39.8% of the predicted stoppages' ends and 20.2% of their restarts**, which is
  where a ball-free rule has to carry the vote (the arrangement of the players, not the ball's position): the source's
  design reads players only unless `use_ball`, for free kicks.
- **This is an upper bound, and a loose one.** The oracle knows the true restart frame and the true team, with no error
  in either, at full weight (an exact 1.0 or 0.0 reads as about 13.8 nats a frame to the hmm at `eps` 1e-6, `headroom.py`'s
  documented reading of an oracle, not a measurement). The real source estimates the
  restart from the dead-ball model and the team from the players' arrangement, and says nothing for the types it cannot
  read, so a real source cannot be expected to exceed +0.0198, and the bar for adopting it is set by R014's declared rule, not by
  this figure. The benchmark also tracks the ball out of play (T39), which leaves the stoppages' anatomy there easier to
  read than on our match.
- **No scoreboard row**: A12 is an analysis, and nothing is submitted from it. The scoreboard lists experiments.

**Part 2: the restart vote's measured confidences, and the DFL restart events (2026-10-26)**

Proposal: P17 · Code: `restart_team.py` at `e186d1a` (the setup window clipped to the stoppage), `restarts.py stats` at
`4304213`, `restarts.py events` and `dfl_events.py` at `57d0a8f` · Commands, from the repo root, one process at a time on
one BLAS thread at below-normal priority:

```bash
python pipeline/benchmark/restarts.py stats --model pipeline/models/dead_ball_hgb.json --out data/runs/benchmark/restarts/stats.json
python pipeline/benchmark/restarts.py events --out data/runs/benchmark/restarts/events.json
```

Validation (P01 benchmark): all 7 DFL matches, profile `tracking` version 3, degradation seeds 0, 1 and 2. Both logs end
`EXIT=0`. The first `events` run failed on a half-edited `restarts.py` and was run again; nothing of it was used (T42).
Aggregates only: the benchmark is public DFL data. Nothing is adopted here, and this part has no scoreboard row.

**What was measured.**

1. **Agreement and confidence** (`restarts.py stats`). The source `restart_team` calls a type of restart and a team after
   each predicted stoppage and votes `home` for that team on `[restart, end)`. For each type and each (tail, `use_ball`)
   setting, the **agreement** is the number of voted frames inside `[restart, end)` whose truth is `H` or `A` and equals
   the called team, over the voted frames whose truth is `H` or `A`, pooled over matches and seeds. The **confidence** is
   that rate floored to a hundredth (`touch_stats.confidence_from_agreement`, as for `controlled_touch`).
2. **The drop rule**, declared in the design before any measurement: a type does not vote if its agreement is below 0.60
   or it has fewer than 30 calls per seed.
3. **Coverage**: the true restarts with a call of the type within 50 frames of them, over all true restarts.
4. **DFL's restart events against the calls** (`restarts.py events`, setting `t050_b0`; a diagnostic that chooses
   nothing). DFL's own event files name the restarts (kick-off, throw-in, corner, goal kick, free kick, penalty); the
   events are placed on the benchmark's timeline by `dfl_events.py` (681 restart events over the 7 matches: KickOff 22,
   ThrowIn 308, CornerKick 57, GoalKick 119, FreeKick 172, Penalty 3; 476 of the 681, 69.9%, fall within 2 s of a true
   `D` to `H`/`A` change; 10 events outside the graded halves were dropped). Each event is matched to the nearest call
   within 50 frames. The counts pool the 3 seeds. The events are human-collected data about other matches (CC BY 4.0), and
   no constant comes from them.

**The confidences, players only** (`use_ball` false). Coverage of all true restarts, with a call of any type: 73.2% in
every tail. The cell is the agreement and its confidence; "dropped" is the rule's:

| type | calls / seed | coverage | tail 25 | tail 50 | tail 100 | tail 150 |
|---|---|---|---|---|---|---|
| kick_off | 5.7 | 0.8% | 94.1% dropped | 94.1% dropped | 94.1% dropped | 93.6% dropped |
| corner | 35.3 | 4.9% | 93.0% -> 0.93 | 81.1% -> 0.81 | 60.3% -> 0.60 | 61.2% -> 0.61 |
| goal_kick | 224.3 | 26.7% | 72.3% -> 0.72 | 72.5% -> 0.72 | 71.1% -> 0.71 | 69.2% -> 0.69 |
| throw_in | 333.3 | 40.8% | 92.0% -> 0.92 | 88.5% -> 0.88 | 80.0% -> 0.79 | 76.9% -> 0.76 |
| free_kick | 0.0 | 0.0% | n/a dropped | n/a | n/a | n/a |

`kick_off` is dropped by the 30-calls-per-seed floor: it is called about once a match outside the halves' starts.

**The confidences, with the ball** (`use_ball` true). Coverage of all true restarts: 39.5% in every tail.

| type | calls / seed | coverage | tail 25 | tail 50 | tail 100 | tail 150 |
|---|---|---|---|---|---|---|
| kick_off | 1.7 | 0.2% | 79.7% dropped | 79.8% dropped | 79.9% dropped | 79.9% dropped |
| corner | 18.3 | 2.5% | 92.6% dropped | 79.7% dropped | 62.2% dropped | 62.9% dropped |
| goal_kick | 104.0 | 9.5% | 72.2% -> 0.72 | 71.7% -> 0.71 | 68.4% -> 0.68 | 66.9% -> 0.66 |
| throw_in | 222.7 | 25.2% | 90.8% -> 0.90 | 87.7% -> 0.87 | 79.2% -> 0.79 | 75.5% -> 0.75 |
| free_kick | 17.3 | 2.1% | 84.3% dropped | 84.0% dropped | 82.7% dropped | 80.3% dropped |

`corner` and `free_kick` are dropped by the 30-calls-per-seed floor when the ball is used. Free kicks are only called with
the ball, and then too rarely to vote.

**The events, recall** (setting `t050_b0`; matched = a call of any type within 50 frames; the 3 seeds pooled):

| DFL kind | events | matched | recall |
|---|---|---|---|
| KickOff | 66 | 11 | 16.7% |
| ThrowIn | 924 | 608 | 65.8% |
| CornerKick | 171 | 88 | 51.5% |
| GoalKick | 357 | 209 | 58.5% |
| FreeKick | 516 | 153 | 29.7% |
| Penalty | 9 | 5 | 55.6% |

**The events, confusion** (rows: the call's type; columns: the event's kind; matched pairs):

| call \ event | KickOff | ThrowIn | CornerKick | GoalKick | FreeKick | Penalty |
|---|---|---|---|---|---|---|
| kick_off | 3 | 0 | 0 | 0 | 0 | 0 |
| corner | 0 | 0 | 45 | 0 | 3 | 2 |
| goal_kick | 8 | 134 | 2 | 205 | 53 | 3 |
| throw_in | 0 | 474 | 41 | 4 | 97 | 0 |
| free_kick | 0 | 0 | 0 | 0 | 0 | 0 |

So:

- **Throw-ins are the source's backbone.** They are the most calls (333 a seed), at 92% agreement on a 1 s vote (tail 25),
  falling to 77% on a 6 s vote (tail 150) as the restarting team's first possession ends.
- **Corners are precise when called, and few.** 45 of the 50 matched corner calls are corners, and agreement is 93% at
  1 s, but there are 35 calls a seed and 4.9% coverage.
- **The goal-kick rule is not specific, and its measured 72% prices that in.** The keeper stands in his goal area in most
  stoppages and the rule comes before the throw-in in the design's priority order, so 134 of the 608 matched throw-ins were called
  goal kicks. **The rules were not revised after this measurement**: they were declared in the design from the Laws of the
  Game before any run, and a revision made after seeing these tables would be tuning on what was seen (T26, T37). The
  goal-kick rule's specificity is a follow-up.
- **Agreement falls with the tail for every type**, as the design expected, because the first possession after a restart
  ends. That is why the confidence is measured per setting and not once.
- **Using the ball halves the coverage** (73.2% to 39.5%) and drops corners and free kicks under the floor: with
  `use_ball` the rules are restricted to the ball's place on the frames where the ball is visible (each rule must fire
  where the ball is, and its taker must be the player nearest it), so a stoppage that does not meet both is no longer
  called. A rule that leans on the ball is also helped by the benchmark's tracked ball out of play (T39).
- **Kick-offs are called rarely and dropped by the floor**; the halves' kick-offs are known anyway, and goals are rare.
- **Recall is partial and is not a score.** 69.9% of the events fall within 2 s of a true change, so the events and the
  truth's flag agree on most restarts and not all; the events' recall (66% of throw-ins, 52% of corners, 59% of goal
  kicks, 30% of free kicks) says what the source sees, and the free kicks' 30% says why they are the weakest type.
  Nothing is chosen from this table.

### A13 — The headroom of main.toml: where the remaining spellF1 is (2026-10-29, Tian)

Proposal: P18 · Commit: `5ab0e89` (branch `p18-offpitch`; `headroom.py`'s `plus_` variants and `oracle_home_gaps`) ·
Commands, from the repo root, one process at a time on one BLAS thread at below-normal priority (four runs, one per
profile and truth; the `context` ones add `--profile context`):

```bash
python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants alive plus_home_visible plus_home_gaps plus_home_all alive_plus_home_visible alive_plus_home_gaps alive_plus_home_all --truth raw --write
python pipeline/benchmark/headroom.py pipeline/configs/main.toml --variants alive plus_home_visible plus_home_gaps plus_home_all alive_plus_home_visible alive_plus_home_gaps alive_plus_home_all --truth coarse --write
# and the same two with --profile context
```

Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), `main.toml` as R013 left it (sha256
`0ebdfbec...`), on `tracking` version 3 and, as a sensitivity check, `context` version 1 (R005), under the raw and the
coarse truth. All four logs end `EXIT=0` (2026-10-29). The `main` row reproduces R013: `tracking` 0.5430 raw and 0.5593
coarse, `context` 0.5489 and 0.5642. **Part 1 of two**: this is `tracking` and `context`; part 2, on `offpitch`, was not run,
since P18 was dropped ([R015b](#r015b--offpitch-v2-found-again-at-rest-a-feasibility-check-and-p18-dropped-2026-10-29-tian)). Aggregates only: the benchmark is public DFL data. This is a measurement of bounds
and nothing is adopted.

**What was measured.** A10 sized what perfect evidence would add for P06's old decoder and its old sources
(`nearest_player`, `ball_presence`). `headroom.py` refused `main.toml` for every variant that gives the decoder the true
team, because it looks for exactly one `nearest_player` source to replace and `main.toml` has two other `home` sources
(`controlled_touch`, `shape_possession`). So A13 adds oracles to the config's own evidence instead of replacing a
source, under new `plus_` names, each at weight 1.0 (the classic variants keep their meaning):

- `alive` adds `oracle_alive`: a perfect dead-ball detector.
- `plus_home_visible` adds the true team on the frames where the tracking has the ball.
- `plus_home_gaps` adds the true team on the live frames where the tracking has no ball (NaN elsewhere).
- `plus_home_all` adds the true team on every live frame.
- `alive_plus_home_visible`, `alive_plus_home_gaps` and `alive_plus_home_all` add `oracle_alive` as well. The last is the
  decoder's own ceiling with perfect evidence.

An exact oracle reads as about 13.8 nats a frame, so where it speaks it outweighs the config's sources, and where it is
silent they speak as before ("what if we also knew X").

**Pooled spellF1** (macroF1, alive/dead F1 in brackets):

| variant | tracking raw | tracking coarse | context raw | context coarse |
|---|---|---|---|---|
| main | 0.5430 (0.8895, 0.9731) | 0.5593 (0.8878, 0.9731) | 0.5489 (0.8902, 0.9700) | 0.5642 (0.8883, 0.9700) |
| alive | 0.5720 (0.9134, 0.9987) | 0.5983 (0.9118, 0.9987) | 0.5823 (0.9175, 0.9987) | 0.6088 (0.9160, 0.9987) |
| plus_home_visible | 0.7372 (0.9289, 0.9720) | 0.7906 (0.9357, 0.9724) | 0.7435 (0.9303, 0.9688) | 0.7996 (0.9371, 0.9692) |
| plus_home_gaps | 0.7194 (0.9313, 0.9756) | 0.6669 (0.9276, 0.9768) | 0.7110 (0.9272, 0.9730) | 0.6622 (0.9238, 0.9743) |
| plus_home_all | 0.8832 (0.9682, 0.9745) | 0.9138 (0.9730, 0.9761) | 0.8778 (0.9652, 0.9716) | 0.9079 (0.9703, 0.9736) |
| alive_plus_home_visible | 0.7663 (0.9532, 0.9987) | 0.8353 (0.9600, 0.9987) | 0.7743 (0.9576, 0.9987) | 0.8486 (0.9649, 0.9987) |
| alive_plus_home_gaps | 0.7546 (0.9545, 0.9984) | 0.7130 (0.9503, 0.9985) | 0.7534 (0.9535, 0.9984) | 0.7161 (0.9495, 0.9985) |
| alive_plus_home_all | 0.9178 (0.9916, 0.9984) | 0.9688 (0.9960, 0.9984) | 0.9173 (0.9915, 0.9984) | 0.9686 (0.9960, 0.9985) |

**Pooled delta over `main`** and the 90% t-interval of the per-match deltas. Every delta in the table has an interval that
excludes 0, an exact sign-flip p of 0.0156 and 7 wins and 0 losses:

| variant | tracking raw | tracking coarse | context raw | context coarse |
|---|---|---|---|---|
| alive | +0.0290 [+0.0259, +0.0321] | +0.0390 [+0.0332, +0.0446] | +0.0334 [+0.0309, +0.0363] | +0.0446 [+0.0385, +0.0507] |
| plus_home_visible | +0.1942 [+0.1821, +0.2046] | +0.2313 [+0.2132, +0.2472] | +0.1946 [+0.1890, +0.2002] | +0.2354 [+0.2200, +0.2500] |
| plus_home_gaps | +0.1764 [+0.1687, +0.1835] | +0.1076 [+0.0993, +0.1154] | +0.1621 [+0.1513, +0.1725] | +0.0980 [+0.0934, +0.1026] |
| plus_home_all | +0.3402 [+0.3273, +0.3514] | +0.3546 [+0.3318, +0.3751] | +0.3289 [+0.3163, +0.3407] | +0.3437 [+0.3259, +0.3607] |
| alive_plus_home_visible | +0.2233 [+0.2107, +0.2344] | +0.2760 [+0.2521, +0.2975] | +0.2254 [+0.2191, +0.2317] | +0.2844 [+0.2670, +0.3006] |
| alive_plus_home_gaps | +0.2116 [+0.2048, +0.2184] | +0.1538 [+0.1415, +0.1660] | +0.2045 [+0.1934, +0.2150] | +0.1519 [+0.1427, +0.1610] |
| alive_plus_home_all | +0.3748 [+0.3636, +0.3850] | +0.4095 [+0.3847, +0.4324] | +0.3684 [+0.3550, +0.3809] | +0.4045 [+0.3840, +0.4239] |

For `context` coarse `alive_plus_home_all` the pooled delta is +0.4045, the mean of the per-match deltas +0.4040, and
the interval is that of the per-match deltas.

**Reading.**
- **On the benchmark the dead-ball side is nearly exhausted.** A perfect `alive` adds only +0.029 to +0.045 spellF1, and
  alive/dead F1 is already 0.970 to 0.973. This is the T39 caveat in numbers: the benchmark's P05 is near-perfect partly
  because DFL tracks the ball out of play. R015 tried it: with the out-of-play ball hidden, today's P05 lost 0.013
  alive/dead F1 (0.9731 to 0.9600), a lead only since the profile failed a gate; R016 never ran (R015b; the P18 retrain's number, reassigned on 30 Oct).
- **The team is where the score is.** A perfect team where the ball is seen adds +0.19 to +0.24; in the gaps +0.10 to
  +0.18 (more under the raw truth than the coarse one); on every live frame +0.33 to +0.35. A10, with the old decoder,
  had the possessor where seen worth +0.15 raw.
- **The interaction is small.** Taken as deltas over the `alive` variant, with the dead ball perfect the team in the gaps
  is worth 0.7546 - 0.5720 = +0.183 (against +0.176 over `main`) and the team where the ball is seen 0.7663 - 0.5720 =
  +0.194 (against +0.194) on `tracking` raw. Over all 8 cells (gaps and seen, four columns) `alive_plus_X` minus `alive`
  differs from `plus_X` minus `main` by at most 0.0093 (gaps, `context` coarse).
- **The decoder's own ceiling with perfect evidence is 0.918 raw and 0.969 coarse** (A10's old decoder: 0.7627 and
  0.9226): `min_possession` 35 and the penalties cap the score far less than they did.
- **`tracking` and `context` agree in every variant**: the same order, and deltas within 0.02.
- **Limits.** Each oracle is exact at full weight, so these are upper bounds, loose ones; what a real source can reach is
  a separate question, and A12's restart ceiling (+0.0198) turned into +0.0040 at best (R014). Nothing is adopted and
  there is no scoreboard row: A13 is an analysis.

### A14 — Before a video ball detector (P14): the ball in pixels, where the tracker loses it, and a perfect ball in our gaps (2026-10-29, Tian)

Proposal: P14 · Commit: `d47ec20` (no code change) · Commands, from the repo root, one process at a time on one BLAS
thread at below-normal priority:

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml --profile clean --truth both --no-write
python pipeline/benchmark/run.py pipeline/configs/main.toml --profile gaps --truth both --no-write
```

Parts 1 and 2 come from two throwaway local scripts (not committed) that read the ATD cache and the tactical homography
through `pipeline.inputs.Inputs` and printed aggregates only. Part 3 is the benchmark (public DFL data). It was run as the
kill test of a debate on P14 between an advocate and a critic, whose decision rules were declared before the result
(the end of this entry). Nothing is adopted and there is no scoreboard row: A14 is an analysis.

**Part 1: the ball in pixels** (BAR-ATH, graded frames).
- The ball's diameter, the ATD ball ± 0.11 m along the pitch projected through the homography: p10 4.7, median 6.0, p90
  8.2 px on the 94,214 ball-present graded frames. Across the pitch the ground footprint is foreshortened to a median of
  2.9 px, but a sphere looks round, so the ball is about 6 px wide. At the image centre the scale is 29.7 px/m (p10 25.3,
  p90 33.8): the frame shows about 65 m of pitch across 1,920 px.
- The ATD ball projects inside the 1920 × 1080 frame on 99.91% of the frames where it exists, and within 40 px of an edge
  on 0.6%: the provider's ball exists only while it is in view.
- 95.3% of the present player slots project inside the frame: 95.5% on ball-visible frames and 95.0% in the gaps. The
  median number of present slots is 22 (21 in the gaps). The camera keeps play in view during the gaps.

**Part 2: where the tracker loses the ball.** The last ball seen before each gap of at least 2 s inside a graded block
(133 gaps) and the first one after it (131):

| | within 40 px of the frame's edge | within 100 px | median distance to the edge | past a line | within 2 m of a line |
|---|---|---|---|---|---|
| last seen before a gap | 0.0% | 6.0% | 285 px | 16.5% | 31.6% |
| first seen after a gap | 3.8% | 5.3% | 266 px | 21.4% | 29.0% |
| every 20th ball-visible frame (4,711) | 0.7% | 2.0% | 318 px | — | — |

**Part 3: a perfect ball in our gaps, on the benchmark.** `gaps` is DFL's clean players with our ball mask transplanted
(R003); `clean` is the same players with DFL's whole ball. Only the ball differs (its speed channel aside, which no
source reads, T31), so `clean` minus `gaps` is what a perfect ball in our gaps would be worth to `main.toml`.

| profile | raw spellF1 (macroF1, alive/dead F1) | coarse spellF1 | possessions per graded minute |
|---|---|---|---|
| `clean` (1 seed) | 0.6575 (0.9351, 0.9976) | 0.5916 | 3.50 |
| `gaps` (3 seeds) | 0.5559 (0.8948, 0.9727) | 0.5719 | 3.07 |
| `tracking` v3 (A13, for reference) | 0.5430 (0.8895, 0.9731) | 0.5593 | — |
| the truth | — | — | 4.58 raw, 2.48 coarse |

Pooled delta `clean` minus `gaps`: **+0.1016 raw and +0.0197 coarse**. The per-match deltas give +0.1014 raw (90%
t-interval [+0.0944, +0.1084], sign-flip p 0.0156, 7 wins in 7; from +0.0848 to +0.1163) and +0.0191 coarse
([+0.0064, +0.0319], p 0.0469, 6 wins in 7; from -0.0049 to +0.0475).

**Reading.**
- **A ViT is the wrong tool here.** A 6 px ball is smaller than one 14 to 16 px patch of DINOv2, RF-DETR or OWLv2, and
  any detector that shrinks the frame loses it ([T46](traps.md)). The candidate is a stride-4 heatmap network at full
  resolution with stacked frames (FootAndBall, WASB-SBDT, both MIT).
- **The ball is lost inside the image, not at its edge.** No onset is within 40 px of the edge, against 0.7% of all
  visible frames, and the median onset is 285 px from it. The ball was in view when the track was lost, so the camera
  does not explain the gaps. A lost lock and an occlusion both fit, and only a detector run on the gaps can tell them
  apart. 16.5% of the onsets are already past a line (the ball out of play), and 21.4% of the ends are found again past a
  line (a throw-in or a corner being taken).
- **The truths disagree on what the ball is worth.** Under the raw truth it is worth +0.10, in every match. Under the
  coarse truth it is worth +0.02, and one match loses. The ball lets the pipeline cut at more of DFL's changes (3.07 to
  3.50 possessions a minute), and most of those are short possessions that the coarse truth absorbs. This is T33's
  warning, and which truth our match resembles is P16's open question.
- **It is generous.** The players are clean in both runs, DFL's ball is visible out of play (T39: alive/dead F1 0.9727
  to 0.9976), and the oracle has perfect recall and no false positives. Against that, the decoder was not re-chosen for
  a ball-rich input. A real detector reaching 35 to 50% of the gap frames, as both sides of the debate assume,
  converts perhaps a quarter to two fifths of this: about +0.025 to +0.04 raw and +0.005 to +0.008 coarse. That is below
  the full match's noise floor of about 0.04 ([T07](traps.md)).
- **The decision rules declared before the result.** The advocate: drop P14 if Δraw < +0.04, or if Δcoarse ≤ +0.01, or
  if raw is high while coarse is below +0.02; go at Δraw ≥ +0.08 with Δcoarse ≥ +0.04. The critic: drop if Δraw < +0.04
  or Δcoarse ≤ +0.01; test B only in the grey zone; go at Δraw ≥ +0.07 with Δcoarse ≥ +0.03. At +0.1016 and +0.0197, the
  advocate's own rule says drop (coarse is below +0.02, by 0.0003), and the critic's says no go. P14 is therefore not
  built now: [After A14](proposals.md#after-a14) and P14's status say what may still run.

### A15 — Descriptors of our truth's changes against DFL's (2026-10-30, Tian)

Proposal: P02 · Commit: `88ce602` (branch `p02-measurements`) · Extends [A11](#a11--our-ground-truths-timing-against-dfls-from-tracking-events-2026-10-25-tian) · Commands, with the data directory at the main checkout's `data/`:

```bash
python pipeline/benchmark/conventions.py public
python pipeline/benchmark/conventions.py dfl --profile tracking --seeds 3 --truth raw       # and --truth coarse
python pipeline/benchmark/conventions.py compare --profile tracking --truth raw             # and --truth coarse
```

Public: the 20 changes of the public truth (12 of `team`, 4 `into_dead`, 4 `out_of_dead`), on our real tracking.
Validation: 7 DFL matches x 3 degradation seeds, profile `tracking` version 3, under the raw and the coarse truth.
A11 timed each change against detector events. This entry describes what the tracking shows around each change (ball
seen, ball speed, the nearest slots' distances), from `Inputs` alone, and compares the public window with DFL. It is a
measurement and not a score: nothing is decoded, tuned or shifted. Aggregates only, copied from the `compare` logs.

Descriptors: `ball_seen` and `ball_seen_1s` are the shares of changes with the ball visible at the change and within
1 s; `ball_speed` (m/s) and `line_m` (m, signed distance to the line) are medians; `old_near_m` and `new_near_m` are the
median distances (m) of the old and the new team's nearest slot to the ball; `new_nearer` is the share of changes where
the new team's slot is the nearer. Gap = public minus DFL; intervals are 90% bootstrap intervals of the public values,
and with fewer than 5 public values only whether the interval excludes 0 is printed.

**Team changes** (12 public; DFL n = 7,053 raw, 2,847 coarse, counting changes times seeds: 2,351 changes times 3 seeds; 4,167 and 1,646 for the distances):

| descriptor | public | DFL raw | gap (raw) | DFL coarse | gap (coarse) |
|---|---|---|---|---|---|
| `ball_seen` | 100% | 59% | 41% [41%, 41%] | 58% | 42% [42%, 42%] |
| `ball_seen_1s` | 100% | 63% | 37% [37%, 37%] | 63% | 37% [37%, 37%] |
| `ball_speed` | 6.54 | 5.43 | 1.10 [-1.73, 3.02] | 5.86 | 0.67 [-2.16, 2.59] |
| `line_m` | -7.48 | -12.20 | 4.72 [-0.27, 7.12] | -11.39 | 3.91 [-1.09, 6.31] |
| `old_near_m` | 3.69 | 2.14 | 1.56 [0.09, 2.65] | 2.22 | 1.47 [-0.00, 2.56] |
| `new_near_m` | 2.68 | 0.64 | 2.05 [1.16, 2.74] | 0.63 | 2.06 [1.17, 2.75] |
| `new_nearer` | 58% | 87% | -29% [-54%, -4%] | 89% | -30% [-55%, -5%] |

**Into `D`** (4 public; DFL n = 2,130; 1,247 for speed and line; 3 public changes have a ball speed and line; the
nearest-slot descriptors have no value on either side's public changes):

| descriptor | public | DFL (raw and coarse) | gap | interval |
|---|---|---|---|---|
| `ball_seen` | 75% | 59% | 16% | excludes 0: no |
| `ball_seen_1s` | 100% | 63% | 37% | excludes 0: yes |
| `ball_speed` | 9.42 | 1.57 | 7.86 | excludes 0: yes |
| `line_m` | -6.45 | 0.28 | -6.73 | excludes 0: no |

**Out of `D`** (4 public; DFL n = 2,097; 1,233 for speed and line):

| descriptor | public | DFL (raw and coarse) | gap | interval |
|---|---|---|---|---|
| `ball_seen` | 100% | 59% | 41% | excludes 0: yes |
| `ball_seen_1s` | 100% | 62% | 38% | excludes 0: yes |
| `ball_speed` | 5.33 | 38.01 | -32.68 | excludes 0: yes |
| `line_m` | -3.67 | -2.01 | -1.66 | excludes 0: no |

The zero-width intervals on the all-True public cells (12 of 12 and 4 of 4 changes seen) and their "excludes 0: yes"
are artefacts: a percentile bootstrap of a 100% share has no spread, so read those cells as the gap itself, not as a
test.

The DFL figures for `into_dead` and `out_of_dead` are the same under both truths (the coarse truth only merges
possessions, so those changes do not move). Only the team rows differ between the two truths.

**What they suggest (leads, not measurements; 12, 4 and 4 public changes).**
- At team changes the new team's nearest slot is further from the ball than on DFL (2.68 m against 0.64 m, interval
  excludes 0 under both truths) and is less often nearer than the old team's (58% against 87 to 89%). That is
  compatible with A11's lead that our truth flips before the new team's first controlled touch, when the ball has not
  yet reached the new team. It is the same direction as A11's `touch_start` lag, though `touch_start` is also built on
  ball-slot distances, so it is not independent evidence.
- The old team's slot is also further on ours (3.69 m against 2.14 m), which suggests the ball is further from
  everybody at our changes, possibly a loose ball in transit rather than a challenge. The interval is thin: the
  coarse `old_near_m` interval [-0.00, 2.56] touches 0 and the raw one only just excludes it ([0.09, 2.65]).
- The into and out of `D` rows hold too few changes to say more than that the ball may be seen more often on ours (the
  `into_dead` `ball_seen` interval holds 0).

Confounds, which keep these from being read as a convention:
- The public window is the best-covered block of the match, while `tracking` v3 places its gaps blind to play (T38):
  `ball_seen` at 100% against 59% is most likely coverage, not convention.
- `tracking` v3 hides DFL's ball but keeps DFL's positions exact, so our own (unmeasured) ball and slot position error
  is absent from DFL's copy. It would inflate both `new_near_m` and `old_near_m` and pull `new_nearer` towards 50%,
  which matches the team observations with no convention at all.
- DFL tracks the ball out of play (T39), so its ball-speed and line descriptors behave differently at restarts. A DFL
  `out_of_dead` median ball speed of 38.01 m/s suggests that DFL's restart placement jumps (the ball is moved to the
  spot), so that row says more about the provider than about the convention.
- Four changes are not a distribution: the intervals for `into_dead` and `out_of_dead` are of values, not statistics.

Inspection only (RULES.md, `f8f2629`, T27): nothing is set from these numbers; a timing constant comes from other games (P16).

## Experiments

Copy this template for each run, and add a scoreboard row for anything worth comparing.

```
### R0xx — <short name> (YYYY-MM-DD, <who>)
Proposal: P0x · Commit: `abc1234` · Command: `…`
Public: macroF1 … · spellF1 … (bootstrap … – …) · poss … · dur … · lat_f …
Validation (P01 benchmark): spellF1 … (… matches, profile …, truth …; 90% t-interval … – …, sign-flip p …)
Changed vs the previous best: …
Verdict: keep / drop / inconclusive — because …
```

### R001 — Pipeline skeleton (2026-10-22, Tian)
Proposal: P03 · Commit: `787c6e4` · Command: `python pipeline/run.py pipeline/configs/main.toml --score`
Public: macroF1 0.6207 (bootstrap 0.460 – 0.742) · spellF1 0.3256 (bootstrap 0.214 – 0.447) · poss 26 · dur 10.8 s ·
lat_f +4.0
Validation (P01): filled in after the benchmark existed. `main.toml` is the benchmark's `nearest_w50` yardstick (the
same evidence source, decoder and half width, per graded half), so its val spellF1 is `nearest_w50` on the default
profile, `tracking` version 3 (R004c, 7 DFL matches x 3 seeds): pooled 0.2018 under the raw truth and 0.1842 under the
coarse one. That is the floor a change has to beat on the benchmark, not our match's score.
Changed vs the previous best: nothing in substance. `main.toml` is R000c run through the pipeline: nearest player
(`nearest_player`) and a ±50-frame vote (`majority_vote`), decoded per half with `X` outside the graded frames. The
public scores are identical and it reports the same 594 possessions over the graded halves. Its `D` is still "the
tracking lost the ball", wrong on 66% of dead-ball frames.
Checks: `python pipeline/selftest.py --data` rebuilds all four committed baselines byte for byte through
`pipeline/configs/smooth50_repro.toml` (scope "file"). `--score` reports `nearest_player` agreeing with the public GT
on 77.8% of 5,640 frames, A01's figure, which confirms the alignment. Runtime: ~15 s once to cache the ATD, then a
few seconds per run.
Verdict: keep as the base everyone builds on (see [pipeline/README.md](../pipeline/README.md)). It is the floor,
R000c, not a candidate.

### R002 — The four yardsticks on the benchmark's clean profile (2026-10-23, Tian)
Proposal: P01 · Commit: `e479764` (the commit that adds this entry; run on its clean tree, so the reports say
`dirty: false`, and an earlier run on the working tree gave the same tables to the last digit) · Command:
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml`
(the first config is the reference; run once more with `--truth coarse`)
Public: not applicable; no BAR-ATH data is read.
Validation (P01 benchmark): 7 DFL matches, profile `clean`, 668.4 graded minutes. Raw truth: pooled spellF1 0.3073
(`nearest_w50`), 0.2958 (`w25`), 0.2166 (`w125`), 0.0983 (`w0`). Coarse truth: 0.3257 (`w125`), 0.3037 (`w50`), 0.2280
(`w25`), 0.0434 (`w0`). Intervals and sign-flip p below.
Changed vs the previous best: first benchmark entry. These are the organisers' four baselines decoded per graded half
instead of per file (`configs/nearest_w*.toml`; R000's published files are per file), so they are the benchmark's
floor and not R000's numbers. Reports: `data/runs/benchmark/clean/<stem>.json` and `<stem>.coarse.json` (not
committed).
Checks: the oracle (the truth as the prediction) scores spellF1 = macroF1 = alive/dead F1 = 1.0000 on all seven
matches. That checks the chain from the loaded match to the score (runner, inputs, the half time, decoder path,
scoring), and not the link from the DFL file to the truth, which the oracle shares with the runner; see below.

**Raw truth** (3,061 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config
minus `nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas, and every
interval below clears 0 by far more than the 1e-12 tie tolerance.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.3073** | 0.3957 | 0.7110 | 0.3081 [0.2871, 0.3291] | 2,646 (3.96) | | | |
| `nearest_w25` | 0.2958 | 0.3893 | 0.7110 | 0.2962 [0.2813, 0.3111] | 4,034 (6.04) | −0.0119 [−0.0212, −0.0026] | 0.0156 | 0 / 7 |
| `nearest_w125` | 0.2166 | 0.3912 | 0.7110 | 0.2178 [0.2011, 0.2345] | 1,534 (2.29) | −0.0903 [−0.1026, −0.0780] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0983 | 0.3667 | 0.7110 | 0.0984 [0.0900, 0.1068] | 14,439 (21.60) | −0.2097 [−0.2377, −0.1818] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 1,659 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.3257** | 0.4044 | 0.7110 | 0.3261 [0.3037, 0.3484] | 1,534 (2.29) | +0.0216 [+0.0073, +0.0358] | 0.0469 | 6 / 1 |
| `nearest_w50` (reference) | 0.3037 | 0.3967 | 0.7110 | 0.3045 [0.2862, 0.3228] | 2,646 (3.96) | | | |
| `nearest_w25` | 0.2280 | 0.3879 | 0.7110 | 0.2281 [0.2165, 0.2397] | 4,034 (6.04) | −0.0764 [−0.0887, −0.0641] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0434 | 0.3645 | 0.7110 | 0.0437 [0.0387, 0.0488] | 14,439 (21.60) | −0.2608 [−0.2812, −0.2403] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The ranking differs between the two truths.** Under the raw truth `w50` beats `w125` on all seven matches (the
  mean delta of `w125` is −0.0903); under the coarse truth `w125` beats `w50` on six of seven (+0.0216, p 0.047, the
  one loss −0.0052), and `w0` is last under both. The raw order is `w50 > w25 > w125 > w0` and the coarse one
  `w125 > w50 > w25 > w0`. This is the warning that A06 asks for. It is not resolved here: a smoothing
  width taken from one truth alone is not safe to pick. In both tables the order of spellF1 is the order of how close
  the predicted possessions per minute are to the truth's (raw, truth 4.58: `w50` 3.96, `w25` 6.04, `w125` 2.29, `w0`
  21.60; coarse, truth 2.48: `w125` 2.29, `w50` 3.96, `w25` 6.04, `w0` 21.60). That describes the table; it does not
  test why.
- **Neither truth gives the published full-match order** (R000: `smooth50` 0.2433 > `smooth125` 0.2242 > `smooth25`
  0.1996 > `raw` 0.0486). Raw puts `w25` above `w125` (Δ of `w125` against `w25`: −0.0784, [−0.0866, −0.0702], 0 wins
  of 7; printed by `python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w25.toml
  pipeline/benchmark/configs/nearest_w125.toml --no-write`, with `w25` first, raw truth), and coarse puts `w125` above
  `w50`: `w125` is third under the raw truth, second in the published order and first under the coarse one. This is the
  clean profile, where the ball is tracked on every frame (on ours it is missing on 41% of the graded frames, A02), so
  the published order is not expected here. Whether the degraded profiles reproduce it is the fidelity check of the
  later tasks.
- **There is almost no `D` on `clean`, as expected.** The ball is visible on every frame, so `nearest_player` votes
  everywhere and `majority_vote`, which takes `D` only where nobody voted, predicts `D` on 0.0% of the graded frames
  against the truth's 44.8%. The `D` class therefore scores 0 and macroF1 stays near 0.4, and the alive/dead F1 is
  0.7110 for every config: it is the F1 of "always alive" (the scorer takes ALIVE as positive, and 55.2% of the graded
  frames are alive). Nothing here ranks a dead-ball detector; that needs the `gaps` profile. The configs are not
  "fixed" for it.
- **Under the raw truth every difference to `w50` is as clear as seven matches can make it**: 0 wins and 7 losses give
  the smallest sign-flip p there is, 2/128 = 0.0156, and the intervals clear 0. It holds in both competitions (the mean
  Δ of `w25`: −0.0039 over the two Bundesliga matches and −0.0151 over the five 2. Bundesliga ones). It is a statement
  about these seven matches, five of them one club's home games, under this truth and this profile; it does not say
  that the gap is real on ours.
- **Pooled and mean per match differ by at most 0.002** (spellF1 0.3073 against 0.3081 for `w50`). The pooled value is
  the organisers' scorer's on all halves as one file (`score.py`); the mean per match is the one the interval is of.
- **The oracle.** An evidence source equal to the raw truth, decoded with `majority_vote(half_width=0)`, scores exactly
  1.0000 on spellF1, macroF1 and alive/dead F1 on each of the seven matches (3,061 possessions: 511, 500, 402, 456,
  421, 401 and 370; both halves of each). **What that checks** is the chain from the loaded match to the score: the
  runner, `MatchInputs`, the separator rows and each half's ends, the decoder path and the scoring. **What it cannot
  check** is the link from the DFL file to the truth: its answers and the runner's truth are the same
  `truth.truth_codes` of the same loaded match, so a mistake in the parse or in `truth_codes` is on both sides and
  cancels (a review swapped H and A inside `truth_codes`, and all seven matches still scored 1.0000). That link rests
  on A05's checks of the parse against kloppy and by frame number (`test_dfl_parser_agrees_with_kloppy`,
  `test_dfl_second_half_spot_check`) and, for the teams, on `MIN_AGREEMENT` in `test_dfl_truth_conventions` (the
  nearest player names the truth's team on 71% to 82% of the alive ball-visible frames of every match and half, and
  would on 18% to 29% with the teams swapped). All three scores are asserted because they are blind to different
  things: macroF1 counts frames and sees a single wrong frame; spellF1 matches possessions by IoU and forgives a
  boundary a frame out; alive/dead F1 cannot see a team. A review's mutant that left the first row of the second half
  undecoded (X) was seen by macroF1 and not by spellF1, because the scorer back-fills an X at a half's start; that is
  not the slip of the controls, which shift every label of a half by one row. Controls on J03WMX (spellF1 / macroF1 /
  alive-dead F1): one row late in both halves 0.9898 / 0.9954 / 0.9988, in the second half only 0.9941 / 0.9976 /
  0.9993, in the first half only 0.9957 / 0.9978 / 0.9995; the teams swapped in the second half 0.4892 / 0.6992 /
  1.0000; the raw oracle against the coarse truth 0.5843 / 0.9581 / 1.0000 (511 predicted possessions against 255),
  so the coarse truth is applied. These control figures are those of `pipeline/benchmark/selftest.py --dfl` re-run at
  `c65c252`, which added the controls to the check; the entry's own commit, `e479764`, is the one that made the
  runs.

Runtime: 8 to 10 s for the four configs on all seven matches with the DFL caches warm (about 1 s per config, nearly all
the rest a cached load per match). The same command twice, and in two processes with different `PYTHONHASHSEED`, writes
the same bytes (`test_runner_json_identical_across_processes`).
Verdict: keep as the benchmark's floor on `clean`, not a candidate. It decides nothing about smoothing width until the
degraded profiles exist and both truths agree. It does show that the runner and the scoring line up with the loaded
match's truth (the oracle; the file-to-truth link is A05's and A06's), and that the choice of truth can reorder the
yardsticks.

### R003 — The four yardsticks on the benchmark's `gaps` profile (2026-10-23, Tian)
Proposal: P01 · Commit: `a8804bf` (the commit that adds this entry; the reports were first made on the working tree it
was committed from, and the fidelity report and the eight run reports re-made on its clean tree, `dirty: false`, are the
same to the last digit apart from the commit they name; the commit "Correct R003's account of the WARNs and tighten the
gaps checks" after it rewords the explanation of the WARNs and a few sentences, and changes no result) ·
Commands: `python pipeline/benchmark/fidelity.py --profile gaps --seeds 3` first, whose gates passed, and then
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile gaps --seeds 3`
(the first config is the reference; run once more with `--truth coarse`, and `w25` first with `w125` second for the
paired comparison of those two)
Public: not applicable. The scores read no BAR-ATH data; the profile does: the gaps are our match's own missing-ball
masks (the Calibration, A07, digest `2415f32e29ea…`), and two of the warning checks are compared with its public
window's figures (77.8%, 66.4%).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `gaps` version 1, 2,005.3 graded
minutes (668.4 per seed). Raw truth: pooled spellF1 0.2102 (`nearest_w50`), 0.2021 (`w25`), 0.1712 (`w125`), 0.0784
(`w0`). Coarse truth: 0.2302 (`w125`), 0.1960 (`w50`), 0.1541 (`w25`), 0.0342 (`w0`). Intervals, fidelity and
sign-flip p below.
Changed vs the previous best: the second benchmark entry, and the first with our gaps in the ball: R002's `clean`
profile has the ball on every frame. The same four configs lose spellF1 when the ball goes (raw `w50` 0.3073 to 0.2102,
`w25` 0.2958 to 0.2021, `w125` 0.2166 to 0.1712, `w0` 0.0983 to 0.0784) and macroF1 rises (raw `w50` 0.3957 to
0.4658) because `majority_vote` has `D` to give where nobody voted (33.4% of the graded frames for `w50`, none on
`clean`). Reports: `data/runs/benchmark/gaps/<stem>.json` and `<stem>.coarse.json`, and
`data/runs/benchmark/fidelity_gaps.json` (not committed; each carries the Calibration's digest).
Checks: `pipeline/benchmark/selftest.py --dfl --data` passes all 189 checks (173 synthetic, 14 with `--dfl` and 2 with
`--data`), and `pipeline/selftest.py` 33 of 33. The checks of the degrader, the fidelity report, the runner's wiring
and Kendall's tau were run against 23 mutants of the code (the last five after the review: an offset that never
reaches its top value, a `mask_share` that counts DFL's gaps, an array back in the repr of an `Atd`, a calibration
that is not type-checked, a stream keyed by a string hash), and each is caught by a check: the cycle of a short mask
started at its first frame, an `rng_for` key without the half, a ball speed left in, a mask applied to x only, the
players nudged, an offset that ignores the match length, a source half that is always the first, a match index that
depends on the order the matches were seen in, a missing share that is a mean of shares, a gap gate ten times as wide,
a D-share band opened, a median gap of the first match only (this one was not caught at first, and a check of the
pooling on hand-made parts was added), an agreement over frames that were not judged, a seed count that is always in
the file name, a dot allowed in a config name, concordant pairs counted as discordant, `np.load` given a path again in
`dfl._read_array`, and histograms of slots that are not padded to one length (the first real fidelity run stopped on
that one: a squad is 39 or 40 people in the DFL matches, and a check of it was added). The real-match check
`test_data_gaps_profile_serves_a_real_match_with_our_real_masks` runs the profile on J03WMX with our real Calibration:
the players are DFL's own arrays, the ball is missing exactly where the transplanted window says (with DFL's own gaps
added), and nothing of the match or its flags is reachable from the inputs it is served with.

**Fidelity first** (`fidelity.py`, 7 matches x 3 seeds; centres are the Calibration's own values on our kick-off
spans, tolerances the plan's; the `gaps` profile was frozen at version 1 once both gates passed, and nothing of it was
changed after, whatever the warnings said). The two gates:

| gate | benchmark | our match (centre) | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |

The share per (match, seed) runs from 0.3927 to 0.4402. The 42 half-draws (7 matches x 3 seeds x 2 halves) took 21 masks
from our first half and 21 from our second, at offsets from 0 to 11,222 frames. In 3 of the 42 the benchmark half is
longer than the mask it drew (J03WPY's second half under seed 2, J03WR9's under seeds 1 and 2: 77,080 and 78,456 frames
against our second half's 76,456), so its offset is 0 and the mask goes on round its last 7,500 frames from its end
(`degrade.transplant`); counted again on 2026-10-23 from the draws and the lengths of the DFL halves, aggregates only.

The ten warning checks (WARN is printed and discussed here; none was tuned towards):

| check | benchmark | our match | band | |
|---|---|---|---|---|
| nearest player agrees with the truth, alive and ball visible | 0.7741 | 0.7784 (public window) | +-0.05 | ok |
| ball visible on dead frames | 0.5684 | 0.6637 (public window) | +-0.10 | ok |
| `w0` possessions per graded minute | 14.14 | 18.40 | +-25% | ok (-23.2%) |
| `w0` D share | 0.4189 | 0.4069 | +-0.05 | ok |
| `w25` possessions per graded minute | 5.21 | 7.04 | +-25% | **WARN** (-26.0%) |
| `w25` D share | 0.3764 | 0.3649 | +-0.05 | ok |
| `w50` possessions per graded minute | 4.16 | 5.61 | +-25% | **WARN** (-25.8%) |
| `w50` D share | 0.3341 | 0.3234 | +-0.05 | ok |
| `w125` possessions per graded minute | 2.97 | 3.76 | +-25% | ok (-21.0%) |
| `w125` D share | 0.2329 | 0.2233 | +-0.05 | ok |

What the warnings say. (1) **Possessions per minute.** All four yardsticks output 21% to 26% fewer possessions per
minute on the benchmark than on ours, and the two that miss the band miss it by under 1.1 points. The direction is the
same for all four, which a band's edge hides: the same decoders flicker more on our tracking than on this one. It is not
that the nearest player is wrong less often here, as far as the public window shows: it names the truth's team on 77.4%
of the alive, ball-visible frames against 77.8% there (check 3 below), so it is wrong on about 22.6% of them here and
22% on ours. What `gaps` lacks is the **structure** of our errors, not their rate: our slots swap and carry the other
team for stretches (T15, T32: a slot of the wrong team on 10.5% of our frames, against none here, and tracklets of a
median 41 s against one slot per player for a whole half), where DFL's players have no swap and no wrong-team slot. That
the arrangement and not the amount of the errors makes the decoders flicker is a reading of these figures, not something
measured here. It belongs to the `tracking` profile, and the two WARNs are left as they are. **What this implies for the
next profile**: the wrong-team episodes of the `tracking` profile will land on top of an agreement that already matches
ours, so they may push `nearest_agreement` below its band (0.778 - 0.05: the benchmark is 4.6 points above the edge).
That would be a WARN to report and discuss, not a miss to tune away; the possessions-per-minute WARNs are the ones the
`tracking` profile is expected to move, reported and not tuned for. (2) **Dead-ball visibility.** 56.8% of the
benchmark's dead frames have a ball, against 58.1% of all its frames: the transplant puts gaps where it likes, with no
regard for the ball being dead. In the public window the ball is visible on 66% of dead frames and on 88% of all frames,
so on ours a dead ball is about three times as likely to be missing as a frame of the window in general (33.6% of the
dead frames against 11.8% of all of them, A01); the benchmark has none of that association, by construction (it is what
the `context` profile would have to bring, if it is built). It is why the alive/dead F1 of the four configs here (0.5761
to 0.6428) is under the 0.7110 of "always alive" that `clean` gave every one of them: a `D` that comes from the ball's
absence is right on ours more often than it can be here. **Do not read the alive/dead F1 of this profile as a measure of
dead-ball detection on ours; the ranking of the possession metrics is what it supports.** (3) **Agreement.** 77.4%
against 77.8%, inside the band, and no more than a sanity check: it compares all frames of seven matches with a window
of 5 minutes (T01).

The statistics `gaps` does not model, on the page so that the later profiles know what they have to fix (information, no
band; the benchmark first, ours second): tracklet median 2,765 s against 41.3 s (one slot per player, against 46 slots
for 22 players), wrong-team frames 0.0% against 10.5%, NaN player speeds 0.0% against 7.8%, 22 players tracked on 81.7%
of the frames against 50.2% (mean 21.8 against 20.5, at most 14 tracked on 0.0% against 5.2%), the players' jitter 9.0
cm against 3.3 mm. Of the gaps: 9.41 gaps per 10,000 frames against 9.36, the 90th percentile 39.0 s against 35.3 s, the
10th 4.9 s against 2.7 s (a pooled figure of 42 transplanted windows, no gate), and **36.0% of the gaps start within 2 m
of a line against 28.0% of ours**: a gap placed without regard to the play starts near a line more often in a DFL match
than ours do. What a placement by the play would change (the `context` profile, if it is built) cannot be told from
this, and there is no base rate of ours here to say more.

**Raw truth** (9,183 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config minus
`nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.2102** | 0.4658 | 0.6094 | 0.2108 [0.2010, 0.2205] | 8,348 (4.16) | | | |
| `nearest_w25` | 0.2021 | 0.4633 | 0.5935 | 0.2023 [0.1950, 0.2097] | 10,446 (5.21) | -0.0084 [-0.0150, -0.0019] | 0.0469 | 1 / 6 |
| `nearest_w125` | 0.1712 | 0.4461 | 0.6428 | 0.1724 [0.1550, 0.1898] | 5,957 (2.97) | -0.0384 [-0.0483, -0.0285] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0784 | 0.4456 | 0.5761 | 0.0784 [0.0719, 0.0849] | 28,352 (14.14) | -0.1324 [-0.1475, -0.1173] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 4,977 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.2302** | 0.4564 | 0.6428 | 0.2305 [0.2109, 0.2501] | 5,957 (2.97) | +0.0340 [+0.0228, +0.0452] | 0.0156 | 7 / 0 |
| `nearest_w50` (reference) | 0.1960 | 0.4665 | 0.6094 | 0.1965 [0.1837, 0.2093] | 8,348 (4.16) | | | |
| `nearest_w25` | 0.1541 | 0.4618 | 0.5935 | 0.1542 [0.1482, 0.1601] | 10,446 (5.21) | -0.0423 [-0.0520, -0.0326] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0342 | 0.4437 | 0.5761 | 0.0344 [0.0301, 0.0388] | 28,352 (14.14) | -0.1621 [-0.1758, -0.1484] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The yardstick check holds in its robust part: `nearest_w0` is last under both truths**, by far more than the noise
  (7 losses of 7 against `w50` under each, Δ -0.1324 raw and -0.1621 coarse, and the gap from `w0` to the next config is
  0.0928 raw and 0.1199 coarse), and in each of the three seeds. Nothing had to be fixed or changed to get it; the check
  is a check, not a goal.
- **The rest of the order follows the truth, as it did on `clean`, and neither truth gives the published order.** Raw:
  `w50 > w25 > w125 > w0`; coarse: `w125 > w50 > w25 > w0`; published (R000, full match): `w50 > w125 > w25 > w0`.
  Kendall tau against the published order is **+0.667 under each truth** (5 of the 6 pairs concordant, 1 discordant:
  `w125` below `w25` under raw, above `w50` under coarse). Each truth keeps the same order in each of the three seeds.
- **The published gaps are inside the noise (about 0.04), and two of the three lie between the raw and the coarse
  ones.** Pooled spellF1 gaps, published against raw against coarse: `w50` - `w25` +0.044 / +0.008 / +0.042; `w125` -
  `w25` +0.025 / -0.031 / +0.076; `w50` - `w125` +0.019 / +0.039 / -0.034. Paired over the matches, `w125` against `w25`
  is -0.0299 [-0.0425, -0.0174], 0 wins of 7, under the raw truth, and +0.0763 [+0.0587, +0.0939], 7 wins of 7, under
  the coarse one
  (`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile gaps --seeds 3 --no-write`,
  with `--truth coarse` for the second). That the published value of each of the two gaps that involve `w125` lies
  between the raw and the coarse one (the third, `w50` - `w25`, does not: +0.044 against +0.008 and +0.042) fits A06's
  reading (DFL's grain is finer than ours, and the coarse truth stands in for how much); it is two numbers, not a
  measurement of ours, and it is not used to choose anything (T26).
- **The choice of truth decides `w125` against `w50` and `w25`, not the profile.** The sign of every paired delta
  against `w50` is the same on both competitions: raw `w125` -0.0478 on the two Bundesliga matches and -0.0346 on the
  five 2. Bundesliga ones, coarse +0.0342 and +0.0339; `w25` -0.0059 and -0.0094 raw, -0.0370 and -0.0445 coarse. Mean
  spellF1 per competition (BL1, BL2): raw `w50` 0.2064, 0.2125; `w25` 0.2005, 0.2031; `w125` 0.1586, 0.1779; `w0`
  0.0866, 0.0751; coarse `w50` 0.1970, 0.1963; `w25` 0.1600, 0.1519; `w125` 0.2313, 0.2302; `w0` 0.0326, 0.0351. Five of
  the seven matches are one club's home games and the Bundesliga mean is of two matches.
- **`w25` against `w50` is the one raw gap that is not decided.** -0.0084 [-0.0150, -0.0019], p 0.0469, 1 win in 7
  (macroF1: -0.0025 with an interval whose upper end is -0.000005, 3 wins in 7, p 0.125): a difference of a hundredth of
  a point that the interval only just excludes; it is not a result, and the published `w50` - `w25` of 0.044 is not
  reproduced under the raw truth but nearly is under the coarse (+0.042).
- **The seeds move a pooled score by up to about 0.015.** Pooled spellF1 by seed (0 / 1 / 2): raw `w50` 0.2063 / 0.2082
  / 0.2162, `w25` 0.1953 / 0.2003 / 0.2106, `w125` 0.1697 / 0.1711 / 0.1729, `w0` 0.0764 / 0.0774 / 0.0815; coarse `w50`
  0.1903 / 0.1957 / 0.2019, `w25` 0.1499 / 0.1521 / 0.1604, `w125` 0.2239 / 0.2347 / 0.2320, `w0` 0.0347 / 0.0316 /
  0.0363. The intervals are over the matches, with the seeds pooled inside each match, so they say nothing of the
  seed-to-seed spread; three seeds are what the default runs.
- **Each yardstick predicts a share of `D` that is nearer to the truth's 44.8% than on `clean` (0.0%)** and still under
  it (`w0` 41.9%, `w25` 37.6%, `w50` 33.4%, `w125` 23.3%), because `majority_vote` takes `D` only where nobody voted
  inside its window: the wider the window, the longer a gap has to be to get a `D` (251 frames, 10 s, for `w125`). That
  follows from the configs, not from the profile.
- **What the benchmark is for after this.** `gaps` v1 is frozen: it resembles our match in the ball's missing share and
  gap lengths (the gates), in the nearest player's agreement and in the `D` shares of the four yardsticks (the checks),
  and in nothing else of the tracking (the statistics above). A conclusion about a change that rests on the flicker of
  the nearest player or on slots that swap needs the `tracking` profile, and one that rests on the dead-ball association
  (a ball that goes missing when it is dead) needs the `context` profile, if that is built; what rests on how a decoder
  smooths over a ball that is gone for ten seconds is what this profile can answer.

Runtime: 20 s for `fidelity.py` on all seven matches and three seeds, and 18 s for each run of the four configs (7
matches x 3 seeds x 4 configs), with the DFL caches and the Calibration cached; the same command twice writes the same
bytes.
Verdict: keep `gaps` v1 as the benchmark's profile, and the four yardsticks as its floor, not as candidates. The
yardstick check passes in its robust part under both truths, and the profile reproduces what it was calibrated on (both
gates). It does not make the truth's grain a non-issue: choose between `w125` and the others only where the raw and the
coarse truth agree.

### R004 — The four yardsticks on the benchmark's `tracking` profile (2026-10-23, Tian)
Proposal: P01 · Commit: `39d86cf` (the commit that adds this entry; the reports were first made on the working tree it
was committed from, and the fidelity report and the run reports re-made on its clean tree, `dirty: false`, are the same
to the last digit apart from the commit they name) · Commands:
`python pipeline/benchmark/fidelity.py --profile tracking --seeds 3` first, whose required gates passed, and then
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3`
(the first config is the reference; run once more with `--truth coarse`, and `w25` first with `w125` second for the
paired comparison)
Public: not applicable. The scores read no BAR-ATH data; the profile does: the gaps, the tracklets, the episodes, the
NaN-speed runs and the jitter are our match's own measurements (the Calibration, A07 to A09, digest `65b0c1915194…`,
version 2), and the fidelity checks are compared with its public window's figures (77.8%, 66.4%).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 1, 2,005.3 graded
minutes (668.4 per seed). Raw truth: pooled spellF1 0.2000 (`nearest_w50`), 0.1895 (`w25`), 0.1660 (`w125`), 0.0768
(`w0`). Coarse truth: 0.2195 (`w125`), 0.1826 (`w50`), 0.1426 (`w25`), 0.0332 (`w0`). Intervals, fidelity and
sign-flip p below.
Changed vs the previous best: the third benchmark entry, and the first with **our players**. The ball is R003's, byte
for byte (`test_profiles_nest`); what changed is the players: broken into tracklets in 46 slots, with a wrong-team slot
now and then, NaN speeds in runs and smoothed positions. Not our frames with few players tracked (the tail, which is
version 2), so three of the eight fidelity gates are pending, and version 1 is frozen as it is with them pending.
Reports: `data/runs/benchmark/tracking/<stem>.json` and `<stem>.coarse.json`, and
`data/runs/benchmark/fidelity_tracking.json` (not committed; each carries the Calibration's digest).
**Corrected after review**, in `47bde9f` (the code, the checks and the README) and `ee0959e` (this entry and T36), which
follow `c89fe8c`; the degradation, the yardsticks' reports and their numbers are unchanged. The three fidelity gates on
how many slots are present were measured on all frames and are now measured on full-strength frames (below): the first
version of this entry said that six required gates passed; measured on full-strength frames the one on exactly 22
present misses by 0.0012, and it is now pending with the other two (`0d12a4e`). And the slot-persistence figures that
the README and the degrader's docstring quoted for this profile were A09's, from before the wrong-team episodes: "The
degraded match against ours" below has them with the episodes on, and the number of slots a team uses (13, not 11). A
second round of review corrected, in the commit after `0d12a4e`, the citation of those figures (A09, not R003), the
printout of the full-strength shares, the statement that `tracking` v1 is frozen (with its three count gates pending),
and the assumption behind the estimate of 0.503.
**Erratum, added with R004c** (2026-10-23): the per-slot jitter that this entry reports after smoothing (6.78 cm on seed
0, 6.47 cm on three seeds, "still 20 x ours") and the reading of it below ("The jitter is not noise that an average
removes", T35) came from same-frame slot handovers, not from DFL's jumps. Version 1 let a tracklet take a slot on the
very row it was freed: 236 times in these 21 runs (11.2 a match and seed, a jump of 5.3 m at the median and 43.0 m at
the 90th percentile), and the same draws measured inside the tracklets have 1.80 cm after the average (R004c). What this
entry says of DFL's own data stands (8.996 cm over the clean players, 98% of the squares in rare jumps in the five 2.
Bundesliga matches, the 1 cm grid), and its scores are version 1's as they were.
Checks: `pipeline/benchmark/selftest.py --dfl --data` passes all 238 checks (218 synthetic, 20 with `--dfl` and
`--data`; 235 and 215 before the correction), `pipeline/selftest.py` 35 of 35 and its `--data` run 37 of 37 (the four
baselines identical byte for byte). The new checks were written first and failed first (16 of 29 checks of the degrader
and 6 of 10 of the fidelity report, for the missing parts). The new code was run against 18 mutants, each caught by a
check: a tie that goes to the highest columns, 2.0 m not counted as within, an overlap not skipped, the draws of picks
and durations swapped, the pick the other way, an episode left in its own pool, a NaN run that goes on past its
tracklet, the first rows not counted towards the NaN target, a smoothing window that does not shrink, onsets drawn from
the walk's stream, an onset rate without the scale, lengths rounded down, calibrated parameters with no smoothing, a
relative tolerance read as an absolute one, a pending gate that is counted, a failed gate that does not fail the run, a
median of medians, a NaN share over all frames. With the new parts off the players are Task 12's byte for byte, and that
was checked against the code of `ac940b2` itself, not only by the reference in the check: the same digest of positions,
speeds and records on two synthetic matches under three seeds and on J03WMX under seeds 0 and 1. The correction's checks
were written first as well and failed first (4 of the 13 checks of the fidelity report, for the missing full-strength
filter and gate kinds), and the filter was run against 6 more mutants, each caught by a check: the count gates on all
frames, full strength read as 22 or more, a gate with no kind read as relative, the full-strength wrong-team share over
all frames, a match with no full-strength frame counted as full strength, and the wrong-team gate on full-strength
frames (a seventh, counting a match's frames once per seed, cannot differ from the code, which keeps one entry per
match).

**The two measurements that fixed a constant** (label-free, once, before any yardstick score was looked at; the rules
were set in the plan):

| | measured | rule | decision |
|---|---|---|---|
| wrong-team share seen by `calibration.player_stats`, s = 1.0, seed 0, 7 matches pooled by sums | 0.03965 (39,756 of 1,002,644 frames) | ours 0.10539 +-0.03 | misses: s <- 1.0 * 0.10539 / 0.03965 = **2.658**, frozen (`degrade.WRONG_TEAM_SCALE`) |
| the same at s = 2.658 | 0.09828 (98,537 frames) | ours 0.10539 +-0.03 | inside (-0.0071); not touched again |
| DFL's second-difference RMS of the clean players, 7 matches pooled by sums of squares | 8.996 cm (per match 1.44, 1.68, 9.27, 10.36, 11.30, 10.45, 11.31) | smooth if more than 2 x ours = 6.56 mm; ours 3.280 mm | 27.4 x: **smooth**, a centred 5-frame moving average per tracklet piece (`degrade.SMOOTH_WINDOW`) |
| the degraded players' jitter after smoothing: seed 0 on 7 matches / the fidelity report (3 seeds) | 6.78 cm / 6.47 cm | information | still 20 x ours |

Both rules are applied again by `test_data_tracking_constants_follow_from_the_measurements`, which prints these figures.

The jitter is not noise that an average removes (T35). In the five 2. Bundesliga matches 98% of the squares of DFL's
second differences are in the largest 0.05% of the triples (rare jumps: 61 of the 1,520,419 steps of J03WOY's first half
exceed 25 m/s, and the largest second difference is 29.7 m), in the two Bundesliga matches 5% and 29%, against 24% in
ours. The bulk is a 1 cm grid: the median second difference is 1.00 cm and the 99th percentile 2.83 cm in all seven
matches (ours 1.4 mm and 9.6 mm). A 5-frame average lowers white noise to 16.3% (checked); here 8.996 cm goes to 6.78
cm. Nothing was done about the rest: the rule was applied as set, and the benchmark's players keep DFL's jumps and are
rougher than ours, which matters to a method that reads positions or speeds more than to the nearest-player vote that
the four yardsticks use.

**Fidelity first** (`fidelity.py --profile tracking`, 7 matches x 3 seeds, 109 s with the caches warm; the centres are
the Calibration's own values on our kick-off spans, the tolerances the plan's; the constants of the degradation were
fixed before any of this was looked at, and nothing of it was changed after, whatever the checks said). **`tracking`
v1 is frozen as it is**: its five counted gates pass, and its three count gates stay pending because they depend on the
low-count tail, which is version 2 (a pending gate is not counted, and the tail is a new version, so v1's pending list
is never emptied). **The three gates on how many slots are present are measured on full-strength frames only**, those
where the clean DFL match has exactly 22 players with positions (the paragraph on full strength below says why; the
first version of this entry measured them on all frames, and that was wrong):

| gate | benchmark | our match (centre) | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, share of full-strength frames | 0.5631 | 0.5019 | +-0.06 | **pending** (off by 0.0612: 0.0012 over the ceiling, 0.5619; would FAIL if counted) |
| mean slots present, full-strength frames | 21.429 | 20.522 | +-0.5 | **pending** (off by 0.907) |
| 14 or fewer slots present, share of full-strength frames | 0.0000 | 0.0525 | +-0.02 | **pending** (off by 0.0525) |
| wrong-team share of frames | 0.1005 | 0.1054 | +-0.03 | pass |
| tracklet median | 29.26 s | 41.28 s | +-50% (+-20.64 s) | pass (-29%) |
| NaN-speed share of the present slot-frames | 0.0787 | 0.0783 | +-0.01 | pass |

All five counted gates pass and `fidelity.py --profile tracking` exits 0, with three gates pending: the three about how
many slots are present, which depend on the low-count tail that the profile does not have yet (10.6% of our frames have
18 or fewer slots present, and the cap of the tail takes frames out of the upper buckets). Ours has such frames and the
degradation has none, so it keeps too many frames at 22: a player is lost 2.59% of the time, and 0.9741^22 = 0.561 of
the frames would have all 22 if the players were lost independently, which is nearly what the profile does. The share of
full-strength frames with all 22 present is 0.5631 against ours 0.5019, 0.0012 over its ceiling, and would FAIL if it
were counted. If the cap of the tail is placed by our counts, independently of the benchmark's own, each upper bucket
loses about 10.6% of its frames and the share becomes 0.5631 x (1 - 0.106) = 0.503, against ours 0.5019 (an estimate
under that assumption, not a measurement; Task 14 measures it). The mean (21.429 against 20.522 +-0.5) and the share
with 14 or fewer (0.0000 against 0.0525 +-0.02) are the same story. The three are printed and not counted
(`fidelity.PENDING_GATES`, whose comment says why each is pending), and all three would miss if they were counted, by
the amounts in the table. The tail is version 2 of the profile: version 1 is frozen as it is, with those three pending,
and the task that builds version 2 empties the list for it. The five gates that are not about the count use 40% (the
ball's missing share), 1% (the median gap), 16% (the wrong-team share) and 4% (the NaN-speed share) of their bands, and
the tracklet median 58% of its band: 29.3 s is 29% below ours, because the episodes cut tracked spans into more and
shorter tracklets (31,398 tracklets over the 21 (match, seed) runs, 1,495 each, against 1,144 in ours), and it is the
passing gate nearest its edge. The 42 half-draws took 21 masks from our first half and 21 from our second, as in R003.

**Full strength, and what the first version of this entry got wrong.** Two of the seven DFL matches have a team down to
10 players for much of the match: in J03WN1 the away team from minute 7.1 of the first half to the end of the match
(92.5% of its graded frames), in J03WQQ the away team from minute 13.4 of the second half to the end (37.2% of its
frames). The other five have 22 players throughout, and no frame has more than 22. Of the 1,002,644 graded frames of the
seven matches, 81.7% (818,710) are at full strength and 18.3% have 21 players, whatever the degradation draws. Those
frames can never show 22 slots and they bring the mean down, so a gate on how many slots are present, measured on all
frames, is half about the degradation and half about the missing player. The first version of this entry gated on all
frames: 0.4598 of them with 22 present against ours 0.5019, a pass at 70% of its band, and it said that the tracklet
median was the gate nearest its edge, which it was not. That pass was two effects that cancel: the profile has too many
frames at 22 (0.5631 on full-strength frames, 0.061 above ours) and 18.3% of the frames cannot have 22 (0.5631 x 0.8166
= 0.4598). The pass would not have outlived the tail: 0.4598 x 0.894 is about 0.41, under the floor of 0.442, for a
reason that has nothing to do with the tail (also an estimate, under the same assumption, not a measurement).

The filter is the frames where the clean match has exactly 22 players with positions, inside the same spans
(`fidelity.full_strength_blocks`, `fidelity.COUNT_GATES`). Ours is measured on all its frames, which **assumes that ours
has 22 players on the pitch throughout**: it is one match, its tracking has no identities, and nothing in the
Calibration can say otherwise. Every other gate and every check stays on all frames. Both values of the three
statistics, and the wrong-team share (a gate on all frames, with the full-strength value as information):

| statistic (benchmark, 7 matches x 3 seeds) | full-strength frames | all frames | our match |
|---|---|---|---|
| exactly 22 slots present, share | 0.5631 | 0.4598 | 0.5019 |
| mean slots present | 21.429 | 21.252 | 20.522 |
| 14 or fewer slots present, share | 0.0000 | 0.0000 | 0.0525 |
| wrong-team share of frames | 0.1100 | 0.1005 | 0.1054 |

The report says how many frames are at full strength (`full_strength`: 818,710 of 1,002,644, 81.7%; J03WN1 7.5%,
J03WQQ 62.8%, the other five 100%), counted once per match whatever the seeds. **s stays at 2.658**: it was rescaled
once from the observable share at s = 1.0 and the filter does not touch the degradation. On the frames at full strength
the share is 0.1100, 0.0046 above ours and inside the gate's band of 0.03. On a frame with 10 players against 11 about
half of the episodes cannot be seen (a player who moves into the team of 10 leaves it with 11, not more than 11), so the
share there is lower, and the shares per match at the frozen scale (seed 0, all frames) show it: 0.1006 (J03WMX),
**0.0499** (J03WN1), 0.1069 (J03WOH), 0.1136 (J03WOY), 0.1048 (J03WPY), **0.0884** (J03WQQ), 0.1228 (J03WR9). That is
information, and no reason to touch s.

The degradation is unchanged, so the yardsticks' numbers below are unchanged: the fidelity report was re-made on the
same code with the filter added, and its other gates (0.4189, 10.76 s, 0.1005, 29.26 s, 0.0787), its ten checks and its
draws are the same to the last digit as in the first version of this entry. The filter has its own check, with a match
that has a player off the pitch for part of a half: those frames are left out of the count gates and counted in every
other statistic (`test_fidelity_count_gates_leave_out_frames_below_full_strength`).

What was drawn, pooled over the 21 runs and both halves: 5,330 onsets made 3,445 episodes (64.6%); 1,759 were skipped
because no opposing pair of tracked players was within 2 m (33.0%) and 126 because the player was already in an episode
(2.4%); the episodes carried 498,167 slot-frames; 86,686 NaN-speed runs were placed. The episodes that `player_stats`
sees in the degraded matches (seed 0) are 105 to 200 a match (mean 163), with medians of 1.4 to 2.2 s and 90th
percentiles of 6.2 to 9.4 s, against ours 104, 2.7 s and 15.8 s: the share is matched and the episodes are more numerous
and shorter (an episode that starts while another runs the other way, or while a player is lost, is seen in pieces or
not at all).

The ten warning checks (WARN is printed and discussed here; none was tuned towards): **none warns**.

| check | benchmark | our match | band | |
|---|---|---|---|---|
| nearest player agrees with the truth, alive and ball visible | 0.7611 | 0.7784 (public window) | +-0.05 | ok |
| ball visible on dead frames | 0.5684 | 0.6637 (public window) | +-0.10 | ok |
| `w0` possessions per graded minute | 14.11 | 18.40 | +-25% | ok (-23.3%) |
| `w0` D share | 0.4189 | 0.4069 | +-0.05 | ok |
| `w25` possessions per graded minute | 5.42 | 7.04 | +-25% | ok (-23.0%) |
| `w25` D share | 0.3764 | 0.3649 | +-0.05 | ok |
| `w50` possessions per graded minute | 4.33 | 5.61 | +-25% | ok (-22.7%) |
| `w50` D share | 0.3341 | 0.3234 | +-0.05 | ok |
| `w125` possessions per graded minute | 3.07 | 3.76 | +-25% | ok (-18.4%) |
| `w125` D share | 0.2329 | 0.2233 | +-0.05 | ok |

The WARN on the nearest player's agreement that R003 expected did not come: the wrong-team episodes lowered it from
0.7741 (`gaps`) to 0.7611, 1.7 points under the public window's 0.7784 and inside the band (0.05). The two
possessions-per-minute WARNs of R003 (`w25` -26.0%, `w50` -25.8%) are gone as well: three of the four yardsticks now
output 4% to 5% more possessions per minute than on `gaps` (`w50` 4.16 to 4.33, `w25` 5.21 to 5.42, `w125` 2.97 to 3.07;
`w0` 14.14 to 14.11), still 18% to 23% fewer than on ours. That the arrangement of the errors and not their amount
drives the flicker is, as in R003, a reading of these figures and not something measured. The
ball-visible-on-dead-frames figure is `gaps`' (0.5684), and so are the D shares and the alive/dead F1 of all four
configs (0.6094, 0.5935, 0.6428, 0.5761): the ball is the same, and these depend on it alone.

**The degraded match against ours, with the episodes on (information; J03WMX, seed 0, our real Calibration).** This is
`test_data_tracking_profile_on_a_real_match_with_our_real_calibration` as it runs at `47bde9f`, the commit that
corrected this entry; only the layout, the determinism, the leaks and the lost share are asserted. d is 0.0250 (ours has
more than 18 slots present on 89.4% of its frames), the mean tracked span 2,797 frames (111.9 s) and the mean lost span
71.7 frames (2.87 s); 2.80% of the 3,211,274 on-pitch frames are lost, in 1,605 tracklets. **Slots used per team**
(home / away; the first version of these figures came from a throwaway script, and the check has printed them itself
since `6144685`, which added the print, `slot_use`, the same function for both sides): of the pools of
24 / 22 slots the degraded match has 13 / 13 ever present, 11 / 12 present on at least a tenth of the frames and 12 / 12
on at least a hundredth, against ours 22 / 21, 21 / 20 and 22 / 21 (A09 had 11 + 11 on a tenth before the episodes,
which take a slot in the other team's pool). The lost
share over the 7 matches and seeds 0 to 4 is 2.59% against d = 2.50% (the check allows 0.003; per match 2.47% to 2.66%).

| | degraded | ours |
|---|---|---|
| slots present, mean | 21.38 | 20.52 |
| 23 or more present | 0.0% | 2.2% |
| all 22 present | 54.0% | 50.2% |
| 19 to 21 present | 45.5% | 37.0% |
| 15 to 18 present | 0.5% | 5.4% |
| 14 or fewer present | 0.0% | 5.2% |
| tracklets | 1,589 | 1,144 |
| tracklet median | 25.8 s | 41.3 s |
| speeds NaN of the present | 7.90% | 7.83% |
| wrong-team frames | 10.1% | 10.5% |

(All frames of the match, which has 22 players throughout; one match and one seed, which is why the gates are measured
on the pooled benchmark and not here.) **Slot persistence with the episodes on.** A09's table of the idle gaps and of
the returns (1,245 gaps, 76.0% of the returns to the same slot) is of the profile as it was before the wrong-team
episodes, and the README and the degrader's docstring quoted it for version 1 by mistake. At this commit, for the same
match and seed, by the same two functions on both sides (`scale = 0` is the same code with the episodes off, and gives
A09's figures again to the last digit):

| | v1, episodes on | `scale = 0` (A09) | ours |
|---|---|---|---|
| slot idle gaps, count | 1,537 | 1,245 | 1,065 |
| median | 2.28 s | 1.96 s | 7.40 s |
| at most 3 s | 59.7% | 67.1% | 17.7% |
| over 60 s | 3.8% | 0.0% | 15.0% |
| returns to the same slot | 860 of 1,247 (69.0%) | 948 of 1,247 (76.0%) | no identities |

A piece of a track that goes into the other team's pool takes a slot there, and the figures move the way one would
expect: a player comes back to his own slot less often (69.0% against 76.0%) and some slots sit idle for a minute or
more (3.8% of the gaps against none), but the profile is still far from ours (a median of 2.3 s against 7.4 s, and 15.0%
of ours over a minute). The warning of A09 stands, and is a little weaker: **a method that follows a slot across a loss
would be validated optimistically on this profile**.

**Raw truth** (9,183 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config minus
`nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.2000** | 0.4598 | 0.6094 | 0.2004 [0.1931, 0.2076] | 8,692 (4.33) | | | |
| `nearest_w25` | 0.1895 | 0.4571 | 0.5935 | 0.1898 [0.1834, 0.1962] | 10,872 (5.42) | -0.0106 [-0.0156, -0.0055] | 0.0156 | 0 / 7 |
| `nearest_w125` | 0.1660 | 0.4414 | 0.6428 | 0.1671 [0.1505, 0.1837] | 6,148 (3.07) | -0.0332 [-0.0441, -0.0224] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0768 | 0.4406 | 0.5761 | 0.0768 [0.0711, 0.0825] | 28,301 (14.11) | -0.1236 [-0.1357, -0.1115] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 4,977 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.2195** | 0.4516 | 0.6428 | 0.2198 [0.2026, 0.2369] | 6,148 (3.07) | +0.0368 [+0.0253, +0.0483] | 0.0156 | 7 / 0 |
| `nearest_w50` (reference) | 0.1826 | 0.4606 | 0.6094 | 0.1829 [0.1725, 0.1933] | 8,692 (4.33) | | | |
| `nearest_w25` | 0.1426 | 0.4556 | 0.5935 | 0.1428 [0.1363, 0.1492] | 10,872 (5.42) | -0.0402 [-0.0472, -0.0331] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0332 | 0.4389 | 0.5761 | 0.0334 [0.0291, 0.0377] | 28,301 (14.11) | -0.1495 [-0.1601, -0.1389] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The yardstick check holds in its robust part: `nearest_w0` is last under both truths**, by far more than the noise
  (7 losses of 7 against `w50` under each, Δ -0.1236 raw and -0.1495 coarse, and the gap from `w0` to the next config is
  0.0892 raw and 0.1094 coarse), and in each of the three seeds. Nothing had to be fixed or changed to get it; the check
  is a check, not a goal.
- **The rest of the order follows the truth, as on `clean` and on `gaps`, and neither truth gives the published
  order.** Raw: `w50 > w25 > w125 > w0`; coarse: `w125 > w50 > w25 > w0`; published (R000, full match):
  `w50 > w125 > w25 > w0`. Kendall tau against the published order is **+0.667 under each truth** (5 of the 6 pairs
  concordant, 1 discordant: `w125` below `w25` under raw, above `w50` under coarse). Each truth keeps the same order in
  each of the three seeds (pooled spellF1 by seed, 0 / 1 / 2: raw `w50` 0.1984 / 0.1985 / 0.2030, `w25` 0.1819 / 0.1904
  / 0.1964, `w125` 0.1634 / 0.1672 / 0.1674, `w0` 0.0741 / 0.0759 / 0.0805; coarse `w50` 0.1746 / 0.1876 / 0.1857, `w25`
  0.1355 / 0.1426 / 0.1497, `w125` 0.2113 / 0.2242 / 0.2231, `w0` 0.0326 / 0.0312 / 0.0357). The seeds move a pooled
  score by up to about 0.015, and the intervals above are over the matches and say nothing of that spread.
- **The published gaps are inside the noise (about 0.04), and two of the three lie between the raw and the coarse
  ones.** Pooled spellF1 gaps, published against raw against coarse: `w50` - `w25` +0.044 / +0.011 / +0.040; `w125` -
  `w25` +0.025 / -0.024 / +0.077; `w50` - `w125` +0.019 / +0.034 / -0.037. Paired over the matches, `w125` against `w25`
  is -0.0227 [-0.0357, -0.0097], 0 wins of 7, under the raw truth, and +0.0770 [+0.0607, +0.0934], 7 wins of 7, under
  the coarse one
  (`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3 --no-write`,
  with `--truth coarse` for the second). That the published value of each of the two gaps that involve `w125` lies
  between the raw and the coarse one fits A06's reading (DFL's grain is finer than ours, and the coarse truth stands in
  for how much); it is two numbers, not a measurement of ours, and it is not used to choose anything (T26). The third
  gap, `w50` - `w25`, is not between them (published +0.044, coarse +0.040).
- **The choice of truth decides `w125` against `w50` and `w25`, not the profile.** The sign of every paired delta
  against `w50` is the same on both competitions: raw `w125` -0.0431 on the two Bundesliga matches and -0.0293 on the
  five 2. Bundesliga ones, coarse +0.0368 and +0.0369; `w25` -0.0080 and -0.0116 raw, -0.0362 and -0.0418 coarse. Mean
  spellF1 per competition (BL1, BL2): raw `w50` 0.1951, 0.2025; `w25` 0.1871, 0.1909; `w125` 0.1520, 0.1732; `w0`
  0.0841, 0.0738; coarse `w50` 0.1832, 0.1828; `w25` 0.1470, 0.1411; `w125` 0.2200, 0.2197; `w0` 0.0317, 0.0341. Five of
  the seven matches are one club's home games and the Bundesliga mean is of two matches.
- **`w25` against `w50` is the one raw gap that is not decided.** -0.0106 [-0.0156, -0.0055], p 0.0156, 0 wins in 7
  (macroF1: -0.0027 with an interval whose upper end is -0.0002, 2 wins in 7, p 0.1094): about a hundredth of a point,
  which the interval excludes by a little; it is not a result, and the published `w50` - `w25` of 0.044 is not
  reproduced under the raw truth (+0.011) and nearly is under the coarse (+0.040).
- **Each yardstick predicts the share of `D` that it did on `gaps`**, because the ball is `gaps`' and `majority_vote`
  takes `D` only where nobody voted inside its window (`w0` 41.9%, `w25` 37.6%, `w50` 33.4%, `w125` 23.3%, against the
  truth's 44.8%).

**How R004 differs from R003** (only the players changed: the ball, the seeds, the matches, the configs and the scorer
are the same, so the two are paired; `tracking` minus `gaps`, from the two sets of reports, over the matches):

| config | raw: pooled spellF1 `gaps` to `tracking` | Δ mean per match [90% t] | wins / losses | coarse: pooled spellF1 | Δ mean per match [90% t] | wins / losses |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2102 to 0.2000 | -0.0104 [-0.0135, -0.0073] | 0 / 7 | 0.1960 to 0.1826 | -0.0136 [-0.0179, -0.0093] | 0 / 7 |
| `nearest_w25` | 0.2021 to 0.1895 | -0.0125 [-0.0155, -0.0096] | 0 / 7 | 0.1541 to 0.1426 | -0.0114 [-0.0145, -0.0084] | 0 / 7 |
| `nearest_w125` | 0.1712 to 0.1660 | -0.0053 [-0.0075, -0.0030] | 0 / 7 | 0.2302 to 0.2195 | -0.0107 [-0.0154, -0.0061] | 0 / 7 |
| `nearest_w0` | 0.0784 to 0.0768 | -0.0016 [-0.0026, -0.0006] | 1 / 6 | 0.0342 to 0.0332 | -0.0010 [-0.0012, -0.0008] | 0 / 7 |

- **Our players cost the four yardsticks 0.1 to 1.4 points of spellF1 and about half a point of macroF1** (macroF1 by
  config -0.0047 to -0.0062 under both truths), in every match for three of the four configs (`w0` under the raw truth:
  6 of 7), and **change no order**: both truths rank the four configs exactly as on `gaps`, with the same Kendall tau,
  and the paired deltas against `w50` move by at most 0.005 (`w125` raw -0.0384 to -0.0332, coarse +0.0340 to +0.0368;
  `w25` raw -0.0084 to -0.0106, coarse -0.0423 to -0.0402). The two truths still disagree about `w125`. What `tracking`
  adds to the benchmark is therefore not a different ranking of these four configs but a harder place to run a method
  that follows a slot, an identity, a speed or a position; its size here is about one point of spellF1 for a decoder
  that reads the nearest player's team and nothing else.
- The nearest player is wrong more often than on `gaps` (agreement 0.7611 against 0.7741, 1.3 points) and the
  yardsticks flicker a little more (possessions per minute up by 4% to 5% for three of the four), which is the direction
  R003 expected, and small.
- The `gaps` figures of R003's reports carry the Calibration digest `2415f32e29ea…` (version 1) and the `tracking` ones
  the digest `65b0c1915194…` (version 2): the Calibration gained its hazard fit, and the gaps profile reads only the
  masks, which did not change. Re-running the `gaps` profile on the version-2 Calibration gives R003's pooled spellF1,
  macroF1, alive/dead F1 and per-match means and intervals to the last digit under both truths (raw 0.2102, 0.2021,
  0.1712, 0.0784 and coarse 0.2302, 0.1960, 0.1541, 0.0342 for `w50`, `w25`, `w125`, `w0`), so the pairing above is of
  the same gaps.

Runtime: 109 s for `fidelity.py` on all seven matches and three seeds (78 s in the first version of this entry), and 52
s for a run of the four configs (7 matches x 3 seeds x 4 configs, against 31 s on `gaps`), with the DFL caches and the
Calibration cached; the same command twice writes the same bytes (the figures of the working-tree run and of a second
run are identical).
Verdict: keep `tracking` v1 as the benchmark's profile for what depends on identities, wrong-team slots and speeds, and
the four yardsticks as its floor, not as candidates; `DEFAULT_PROFILE` stays `gaps` until the tail is built and its
gates count. The yardstick check passes in its robust part under both truths. The profile reproduces what it was
calibrated on in all five counted gates and in all ten checks (no WARN); the three gates about how many slots are
present are pending, because it is the profile before its tail (the share of full-strength frames with all 22 present is
0.5631 against ours 0.5019 +-0.06, and about 0.503 after the tail, an estimate that assumes its cap is placed by our
counts), so `fidelity.py --profile tracking` exits 0 and `tracking` v1 is frozen as it is, with those three pending: the
README's rule does not count a pending gate, and the tail is version 2, so v1's list is never emptied. It
does not make the truth's grain a non-issue: choose between `w125` and the others only where the raw and the coarse
truth agree. The profile's players are rougher than ours (jumps, T35) and have no positional bias (the README's
limitations), and it has no tail yet. **Superseded by R004b** (`tracking` v2: the tail, the shortest
piece and the scale of the tracked spans; the ranking is the same).

### R004b — The four yardsticks on `tracking` version 2 (2026-10-23, Tian)
Proposal: P01 · Commit: `95583d9` (the code every report below was made at, on a clean tree, `dirty: false`;
the cap alone, before the rule and the scale, is `b640752`; the entry, the README and T37 are `16b7e50`) · Commands:
`python pipeline/benchmark/fidelity.py --profile tracking --seeds 3` first, whose gates all passed, and then
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3`
(the first config is the reference; run once more with `--truth coarse`, and `w25` first with `w125` second for the
paired comparison; `--profile tracking` is the default now)
Public: not applicable. The scores read no BAR-ATH data; the profile does: the gaps, the tracklets, the episodes, the
NaN-speed runs, the jitter and now the slot counts are our match's own measurements (the Calibration, A07 to A09, digest
`65b0c1915194…`, version 2), and the fidelity checks are compared with its public window's figures (77.8%, 66.4%).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 2, 2,005.3 graded
minutes (668.4 per seed). Raw truth: pooled spellF1 0.1989 (`nearest_w50`), 0.1897 (`w25`), 0.1624 (`w125`), 0.0779
(`w0`). Coarse truth: 0.2173 (`w125`), 0.1811 (`w50`), 0.1422 (`w25`), 0.0341 (`w0`). Intervals, fidelity and sign-flip
p below.
Changed vs the previous best: the fifth benchmark entry, and the second with our players: version 2 of the `tracking`
profile. R004's players (version 1) had no frames with few players tracked, so three of its eight fidelity gates were
pending; version 2 adds those frames, and two corrections that the first measurement of them forced (below). The ball is
R003's and R004's, byte for byte (`test_profiles_nest`). **The ranking does not move**: both truths order the four
configs as they did in R004 and as they did on `gaps`, with the same Kendall tau, and no config's pooled spellF1 moves
by more than 0.4 points; what changed is the players, and the checks below say how.
Reports: `data/runs/benchmark/tracking/<stem>.json` and `<stem>.coarse.json` (R004's, version 1, were kept in
`tracking_v1/` for the paired comparison below), and `data/runs/benchmark/fidelity_tracking.json` (not committed; each
carries the Calibration's digest).
**Corrected after review** (review round 1 of Task 14, in `97b62b5` and `59bc7d6`; the degradation, the reports and
every number of this entry are unchanged: `fidelity.py` on the corrected code prints the same figures to the last digit,
and a run of the four yardsticks under both truths with `--no-write` prints the same tables and Kendall's tau, apart
from the commit it names). Four things changed. (1) The assertion that smoothing lowers the jitter compared the degraded
players, cut by the walk, the episodes, the cap and the rule, with DFL's clean ones, so which jumps survive the cuts
decided its sign (T35); it ran before the assertion on lambda, and 2.042 was asserted nowhere else, so a moved lambda
was reported as "smoothing lowers the jitter". It now compares the same draws with and without the average (smooth=0:
0.1201 m to 0.0873 m, seed 0, the 7 matches), the assertion on lambda comes first and uses recorded constants only, and
the default run pins it too (`test_tracked_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements`: 41.28 /
20.22 = 2.042; with lambda at 2.0 that check fails alone, and in the data check the first assertion does). (2) The order
of the cap, the episodes and the smoothing was pinned by no check: a cap taken before the episodes, and a smoothing and
speeds taken over the uncapped piece across the cuts, passed all 41 older synthetic checks of checks/tracking.py. Two
scenes pin them, and each fails on its mutant and on no other check
(`test_cap_is_applied_to_the_pieces_after_the_episodes_and_leaves_their_counts_alone`, where the half has 23 onsets and
no episode when the cap comes first, every onset on a capped row finding no pair; and
`test_cap_cuts_the_smoothing_and_the_speeds_at_the_drops`). (3) `leaks` and its helpers moved from checks/inputs.py to
checks/common.py, so that checks/tracking.py and checks/degrade.py import no other check module. (4)
`calibrated_parameters` read TRACKED_SCALE and MIN_PIECE_FRAMES as default arguments, bound when the module loaded; it
now reads them when it is called, as WRONG_TEAM_SCALE and SMOOTH_WINDOW are
(`test_calibrated_parameters_read_the_scale_and_the_shortest_piece_when_called`), and the docstrings and the README's
jitter (it quoted version 1's 6.47 cm, 20 times ours, for the profile) were brought up to date.
**Erratum, added with R004c** (2026-10-23): the jitter after the 5-frame smoothing that this entry reports (8.5 cm;
0.1201 m to 0.0873 m on seed 0 in the corrected assertion) and what it says of it ("98% rare jumps in the 2. Bundesliga
matches", "which the cuts can do", T35) came from same-frame slot handovers, not from DFL's jumps: version 2 let a
tracklet take a slot on the very row it was freed, 372 times in these 21 runs (17.7 a match and seed, 13.8 m at the
median), and the same draws measured inside the tracklets have 1.75 cm after the average and 8.65 cm without it (R004c).
The scale of the tracked spans, lambda = 2.042, was derived with that rule; version 3's is 2.072 (R004c). The scores,
the gates and the other figures of this entry stand for version 2 as it was.
Checks: `pipeline/benchmark/selftest.py --dfl --data` passes all 261 checks (241 synthetic, 20 with `--dfl` and
`--data`; 257 and 237 before the review, 238 and 218 in R004), `pipeline/selftest.py` 35 of 35 and its `--data` run 37
of 37 (the four baselines identical byte for byte). The checks of each part were written first and failed first (the
cap's 12 for a missing `cap_pieces`, `cap_limits` and `transplant_counts` and a `choose_half` that took no limit; the 5
of the shortest piece, the scale and the draws' record for a missing `MIN_PIECE_FRAMES` and missing `min_piece` and
`tracked_scale` parameters; the 4 of the fidelity gates, the report, the pending list and the runner for a gate still
pending and a default still `gaps`). The code was run against 34 mutants (21 of the cap and what it touches, 13 of the
shortest piece, lambda and their record), each caught by a check in the end: for the cap, `CAP` itself not capped, a tie
to the higher column, the farthest kept, membership that is not sticky, the nearest dropped when there are too many, the
farthest added when there are too few, a centre spot that is a corner, no last place for a missing ball, another
Transplant, a window of counts that goes round the whole source and not its tail, a drop counted once per piece, a
profile that does not cap, a float count accepted, pieces returned unsorted, counts and masks of different lengths, a
ball lost on every third row, rows before the kick-off capped and a piece that is not there yet pre-marked as kept (it
survived until a hand-built scene was added); for the shortest piece, the minimum itself dropped, one frame short kept,
an episode piece not counted as one, the frames counted as one, a rule that does not run, a profile with no rule; for
lambda, not applied, rounded down, zero accepted, a bool accepted, the record forgetting it, and the counts not pooled
by the report; and the default profile, the pending list and a moved lambda. Three survived at first, none of them for
the same reason: a window of counts that did not go round (a no-op, replaced by a sharper mutant that goes round the
whole source), a piece that is not present kept (a hand-built scene with spare room was added) and the rule's counts not
pooled by the report (the check now names the pooled counts explicitly). The moved lambda was at first caught only by
`--data`, and there by the assertion of the smoothing (fragile) before it reached the one on lambda; review round 1
fixed both (above).

**What happened, in order** (the first two steps were stops: the notes say to stop and report when a gate misses).
1. The cap alone (the first version of v2) passed the three count gates it was built for and failed one gate it was not
   about: exactly 22 present 0.5068 against 0.5019 +-0.06, mean 20.62, 14 or fewer 0.0468, wrong-team 0.0910, and **the
   tracklet median 19.68 s against ours 41.28 s, 52% below, 0.96 s past the +-50% floor** (v1: 29.26 s). The three count
   gates came out as the estimate of R004 said (0.503 for the first), so the cap and its placement by our counts work.
2. Where the median went (information; the same 7 matches and 3 seeds, the lengths laid end to end, tracklets as
   `player_stats` sees them):

   | variant | tracklets | median | quartiles | under 8 frames |
   |---|---|---|---|---|
   | neither the episodes nor the cap (the walk alone) | 24,459 | 40.88 s | 8.32, 145.24 s | 2 (0.01%) |
   | episodes only (version 1) | 31,162 | 29.26 s | 6.40, 111.40 s | 206 (0.66%) |
   | cap only | 33,759 | 24.44 s | 5.56, 97.48 s | 671 (1.99%) |
   | both | 40,231 | 19.68 s | 4.56, 79.40 s | 875 (2.17%) |
   | ours | 1,144 (one match) | 41.28 s | 8.28, 152.26 s | 0 (the shortest is 8 frames, 0.32 s) |

   The walk alone reproduces ours, as it should: it draws its tracked spans from our tracklets. And every part that cuts
   takes the median further down, because those tracklets are our **final** ones and already hold the cuts that the
   episodes and the cap add: the profile counted our cuts twice (T37).
3. The shortest pieces. 906 of 40,866 pieces (2.2%) were shorter than 8 frames, where ours has none: 410 of them
   between two runs of low counts (the count crosses 18 on 155 frames of ours, 147 of them by one slot, 18 and 19, and
   the dropped players come back for a few frames and go again; 9.2% of the gaps between our 78 runs are under 8
   frames), 22 at one edge of a run, 195 short before the cap (the walk's run ends and the episodes) and 279 made inside
   the runs by membership changes. The ruling was a rule and no machinery: `degrade.MIN_PIECE_FRAMES = 8` (A09's
   shortest tracklet of ours), and after the cap and before the slots every piece shorter than that is dropped, whatever
   made it (the player is absent there). Over the 21 runs it dropped 794 pieces holding 3,177 slot-frames, 172 of them
   episode pieces.
4. The scale of the tracked spans, rescaled once, label-free (`degrade.TRACKED_SCALE`, lambda). The lengths the walk
   draws are multiplied by lambda and rounded (at least 1 frame); the lost spans follow through mu, so the lost share
   stays d. **One measurement fixed it**: the pooled tracklet median of this profile at lambda = 1 with the
   shortest-piece rule in, seed 0, over the 7 matches laid end to end as the fidelity report pools, was **20.22 s**
   (13,192 tracklets), against ours **41.28 s**, so lambda = 41.28 / 20.22 = 2.0415, **2.042** (2026-10-23). It was
   applied whether or not the gate would pass, and frozen: it is not to be touched again because of a gate, a WARN or a
   score. The median at lambda 2.042 is 26.24 s on seed 0 and 27.08 s on the three seeds (the ratio is a one-shot on a
   statistic that does not grow linearly with lambda, so it does not reach ours: -34%). Both measurements are in the
   comment of the constant, and `test_data_tracking_constants_follow_from_the_measurements` re-derives lambda from the
   recorded 20.22 s and makes the measurement again (it must reproduce it).
5. Lost share against d after lambda: the walk, the episodes and the rule (no cap), pooled over the 7 matches and
   seeds 0 to 4, lose **2.71% against d = 2.50%**, per match 2.73%, 2.76%, 2.49%, 2.76%, 2.61%, 2.71%, 2.90% (R004:
   2.59%): longer spans mean fewer cycles per run, so the cut at the end of a run weighs more, as A09 says, and the rule
   drops a little. The existing check allows 0.3 points and holds (0.21); it was kept and not loosened. With the cap,
   J03WMX on seed 0 loses 7.76% of its on-pitch frames (the cap takes the rest).

**Fidelity** (`fidelity.py --profile tracking`, 7 matches x 3 seeds, 93 to 110 s with the caches warm; the same digits
in the last two runs; the centres are the Calibration's own values on our kick-off spans, the
tolerances the plan's; the constants were fixed before any yardstick score was looked at, and nothing was changed after
because of one). **`tracking` v2 is frozen: all eight gates are counted and pass, `fidelity.py` exits 0**:

| gate | benchmark | our match (centre) | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, share of full-strength frames | 0.4938 | 0.5019 | +-0.06 | pass (-0.0081, 14% of the band) |
| mean slots present, full-strength frames | 20.599 | 20.522 | +-0.5 | pass (+0.076, 15%) |
| 14 or fewer slots present, share of full-strength frames | 0.0469 | 0.0525 | +-0.02 | pass (-0.0056, 28%) |
| wrong-team share of frames | 0.0942 | 0.1054 | +-0.03 | pass (-0.0112, 37%) |
| tracklet median | 27.08 s | 41.28 s | +-50% (+-20.64 s) | pass (-34%, 69% of the band) |
| NaN-speed share of the present slot-frames | 0.0788 | 0.0783 | +-0.01 | pass |

The count gates are measured on full-strength frames as R004 says (T36); on all frames the same statistics are 0.4032
(22 present), 20.436 (mean) and 0.0467 (14 or fewer), against ours on all its frames. The wrong-team share on the
full-strength frames is 0.1038. The tracklet median is the gate nearest its edge, as it was in R004. The 42 half-draws
took 21 masks from our first half and 21 from our second, and the draws' record pooled over the 21 runs and both halves
says: 5,330 onsets made 3,447 episodes (64.7%), 1,757 skipped for no opposing pair within 2 m (33.0%) and 126 for a
player already in an episode (2.4%); 79,277 NaN-speed runs; **the cap: 303,259 frames at or below 18 of our count, 10.1%
of the 3,007,932 frames of the spans (ours 10.6% and 5.2% at or below 14), 2,363,692 slot-frames dropped and 4,463
pieces cut; the shortest-piece rule: 794 pieces dropped, 3,177 slot-frames, 172 episode pieces**; 28,880 pieces, 1,375 a
run (ours 1,144 tracklets in a match). The jitter after the 5-frame smoothing is 8.5 cm (ours 3.3 mm; T35).

The ten warning checks (WARN is printed and discussed here; none was tuned towards): **none warns**, as in R004.

| check | benchmark | our match | band | |
|---|---|---|---|---|
| nearest player agrees with the truth, alive and ball visible | 0.7586 | 0.7784 (public window) | +-0.05 | ok |
| ball visible on dead frames | 0.5684 | 0.6637 (public window) | +-0.10 | ok |
| `w0` possessions per graded minute | 13.91 | 18.40 | +-25% | ok (-24.4%) |
| `w0` D share | 0.4189 | 0.4069 | +-0.05 | ok |
| `w25` possessions per graded minute | 5.39 | 7.04 | +-25% | ok (-23.5%) |
| `w25` D share | 0.3764 | 0.3649 | +-0.05 | ok |
| `w50` possessions per graded minute | 4.32 | 5.61 | +-25% | ok (-23.0%) |
| `w50` D share | 0.3341 | 0.3234 | +-0.05 | ok |
| `w125` possessions per graded minute | 3.06 | 3.76 | +-25% | ok (-18.7%) |
| `w125` D share | 0.2329 | 0.2233 | +-0.05 | ok |

**What version 2 did to the WARNs: almost nothing** (the cap, the rule and the scale together: they were not separated).
The cap keeps the players nearest the true ball, and the nearest player is what the four yardsticks read, so the nearest
player's agreement moves only from 0.7611 (R004) to 0.7586 and the possessions per minute from 14.11, 5.42, 4.33 and
3.07 to 13.91, 5.39, 4.32 and 3.06; the D shares and the alive/dead F1 of all four configs are R004's to the last digit,
since they depend on the ball alone. `w0` is the one near its edge (-24.4% of ours against a band of 25%), as it was
(-23.3%). The ball-visible-on-dead-frames figure is `gaps`' (0.5684).

**The degraded match against ours, with the cap (information; J03WMX, seed 0, our real Calibration).** This is
`test_data_tracking_profile_on_a_real_match_with_our_real_calibration` as it runs at `95583d9`; only the layout, the
determinism, the leaks, the Transplant of the cap, the count of players on every row and the lost share are asserted.
d is 0.0250, the mean tracked span 5,712 frames (228.5 s) after lambda and the mean lost span 146.5 frames (5.86 s); the
cap reads 20,031 of the 145,967 frames from the kick-offs (13.7%: the window of our counts that this half got) and drops
162,263 slot-frames in 236 pieces.

| | degraded | ours |
|---|---|---|
| slots present, mean | 20.29 | 20.52 |
| 23 or more present | 0.0% | 2.2% |
| all 22 present | 48.4% | 50.2% |
| 19 to 21 present | 37.5% | 37.0% |
| 15 to 18 present | 7.3% | 5.4% |
| 14 or fewer present | 6.7% | 5.2% |
| tracklets | 1,454 | 1,144 |
| tracklet median | 22.8 s | 41.3 s |
| tracklets under 8 frames | 0.00% | 0.00% |
| speeds NaN of the present | 7.85% | 7.83% |
| wrong-team frames | 8.7% | 10.5% |

**Slot persistence with the cap** (§B6 of the notes; the same two functions on both sides, J03WMX, seed 0; "no cap" is
the same code with lambda and the rule and without the cap):

| | v2 | v2 without the cap | v1 (R004) | ours |
|---|---|---|---|---|
| slot idle gaps, count | 1,402 | | 1,537 | 1,065 |
| median | 4.12 s | | 2.28 s | 7.40 s |
| at most 3 s | 42.0% | | 59.7% | 17.7% |
| over 60 s | 5.3% | | 3.8% | 15.0% |
| returns to the same slot | 552 of 1,147 (48.1%) | 384 of 598 (64.2%) | 860 of 1,247 (69.0%) | no identities |
| slots ever present, home / away of 24 / 22 | 14 / 13 | | 13 / 13 | 22 / 21 |
| on at least a tenth of the frames | 12 / 12 | | 11 / 12 | 21 / 20 |
| on at least a hundredth | 12 / 13 | | 12 / 12 | 22 / 21 |

The cap does what the notes expected of it, a drop breaks a slot's identity: a player who comes back takes the lowest
free slot, which is now often not his own (48.1% against 69.0%), and the slots idle longer (4.1 s against 2.3 s) and
more of them idle for a minute (5.3%). It is still far from ours (7.4 s and 15.0%; 17.7% of ours at most 3 s against
42.0%): the lowest-free rule gives a team 12 slots that carry most of the frames where ours has 21. The warning of A09
and R004 stands, a little weaker: **a method that follows a slot across a loss would be validated optimistically on this
profile**. (The final review decides whether a slot-anonymising variant is needed.)

**Raw truth** (9,183 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config minus
`nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.1989** | 0.4579 | 0.6094 | 0.1993 [0.1918, 0.2068] | 8,660 (4.32) | | | |
| `nearest_w25` | 0.1897 | 0.4557 | 0.5935 | 0.1898 [0.1853, 0.1943] | 10,800 (5.39) | -0.0094 [-0.0147, -0.0042] | 0.0312 | 1 / 6 |
| `nearest_w125` | 0.1624 | 0.4395 | 0.6428 | 0.1634 [0.1485, 0.1784] | 6,127 (3.06) | -0.0358 [-0.0458, -0.0259] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0779 | 0.4397 | 0.5761 | 0.0778 [0.0721, 0.0834] | 27,895 (13.91) | -0.1215 [-0.1328, -0.1101] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 4,977 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.2173** | 0.4494 | 0.6428 | 0.2176 [0.2017, 0.2334] | 6,127 (3.06) | +0.0360 [+0.0279, +0.0441] | 0.0156 | 7 / 0 |
| `nearest_w50` (reference) | 0.1811 | 0.4584 | 0.6094 | 0.1815 [0.1705, 0.1926] | 8,660 (4.32) | | | |
| `nearest_w25` | 0.1422 | 0.4541 | 0.5935 | 0.1423 [0.1374, 0.1472] | 10,800 (5.39) | -0.0392 [-0.0475, -0.0309] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0341 | 0.4378 | 0.5761 | 0.0343 [0.0309, 0.0376] | 27,895 (13.91) | -0.1473 [-0.1588, -0.1357] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The yardstick check holds in its robust part: `nearest_w0` is last under both truths**, by far more than the noise
  (7 losses of 7 against `w50` under each, Δ -0.1215 raw and -0.1473 coarse, and the gap from `w0` to the next config is
  0.0845 raw and 0.1081 coarse), and in each of the three seeds. Nothing had to be fixed or changed to get it.
- **The rest of the order follows the truth, as on `clean`, `gaps` and R004, and neither truth gives the published
  order.** Raw: `w50 > w25 > w125 > w0`; coarse: `w125 > w50 > w25 > w0`; published (R000, full match): `w50 > w125 >
  w25 > w0`. **Kendall tau against the published order is +0.667 under each truth** (5 of the 6 pairs concordant, 1
  discordant), as in R004: **the ranking did not move**. Each truth keeps the same order in each of the three seeds
  (pooled spellF1 by seed, 0 / 1 / 2: raw `w50` 0.1973 / 0.1970 / 0.2026, `w25` 0.1830 / 0.1874 / 0.1990, `w125` 0.1616
  / 0.1615 / 0.1640, `w0` 0.0751 / 0.0770 / 0.0817; coarse `w50` 0.1768 / 0.1804 / 0.1861, `w25` 0.1381 / 0.1393 /
  0.1494, `w125` 0.2092 / 0.2225 / 0.2202, `w0` 0.0345 / 0.0316 / 0.0362). The seeds move a pooled score by up to about
  0.015, and the intervals above are over the matches and say nothing of that spread.
- **The published gaps are inside the noise (about 0.04), and two of the three lie between the raw and the coarse
  ones.** Pooled spellF1 gaps, published against raw against coarse: `w50` - `w25` +0.044 / +0.009 / +0.039; `w125` -
  `w25` +0.025 / -0.027 / +0.075; `w50` - `w125` +0.019 / +0.037 / -0.036. Paired over the matches, `w125` against `w25`
  is -0.0264 [-0.0389, -0.0139], 1 win of 7, under the raw truth, and +0.0752 [+0.0613, +0.0892], 7 wins of 7, under the
  coarse one (`... nearest_w25.toml ... nearest_w125.toml --profile tracking --seeds 3 --no-write`, with `--truth
  coarse` for the second). That the published value of each of the gaps that involve `w125` lies between the raw and the
  coarse one fits A06's reading (DFL's grain is finer than ours, and the coarse truth stands in for how much); it is two
  numbers, not a measurement of ours, and it is not used to choose anything (T26).
- **The choice of truth decides `w125` against `w50` and `w25`, not the profile.** The sign of every paired delta
  against `w50` is the same on both competitions: raw `w125` -0.0486 on the two Bundesliga matches and -0.0307 on the
  five 2. Bundesliga ones, coarse +0.0381 and +0.0352; `w25` -0.0079 and -0.0101 raw, -0.0354 and -0.0408 coarse. Mean
  spellF1 per competition (BL1, BL2): raw `w50` 0.1978, 0.1999; `w25` 0.1899, 0.1898; `w125` 0.1492, 0.1691; `w0`
  0.0865, 0.0743; coarse `w50` 0.1823, 0.1812; `w25` 0.1469, 0.1405; `w125` 0.2203, 0.2164; `w0` 0.0333, 0.0347.
- **`w25` against `w50` is the one raw gap that is not decided**: -0.0094 [-0.0147, -0.0042], p 0.0312, 1 win in 7
  (macroF1: -0.0021 [-0.0043, +0.0002], 3 wins in 7, p 0.1250): about a hundredth of a point, which the interval
  excludes by a little; it is not a result.

**How R004b differs from R004** (only the players changed: the ball, the seeds, the matches, the configs and the scorer
are the same, so the two are paired; `tracking` v2 minus v1, from the two sets of reports, over the matches):

| config | raw: pooled spellF1 v1 to v2 | Δ mean per match [90% t] | wins / losses | coarse: pooled spellF1 | Δ mean per match [90% t] | wins / losses |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2000 to 0.1989 | -0.0011 [-0.0038, +0.0016] | 3 / 4 | 0.1826 to 0.1811 | -0.0014 [-0.0047, +0.0019] | 3 / 4 |
| `nearest_w25` | 0.1895 to 0.1897 | +0.0000 [-0.0031, +0.0031] | 4 / 3 | 0.1426 to 0.1422 | -0.0004 [-0.0028, +0.0019] | 3 / 4 |
| `nearest_w125` | 0.1660 to 0.1624 | -0.0037 [-0.0063, -0.0011] | 2 / 5 | 0.2195 to 0.2173 | -0.0022 [-0.0051, +0.0006] | 2 / 5 |
| `nearest_w0` | 0.0768 to 0.0779 | +0.0010 [-0.0002, +0.0023] | 4 / 3 | 0.0332 to 0.0341 | +0.0009 [-0.0001, +0.0018] | 4 / 3 |

- **The low-count frames cost the four yardsticks no more than 0.4 points of spellF1 and about 0.1 to 0.2 points of
  macroF1** (macroF1 by config -0.0009 to -0.0022 under both truths; its 90% interval excludes 0 for three of the four
  configs under the raw truth and for all four under the coarse one), and **change no order**: both truths rank the four
  configs exactly as in R004, with the same Kendall tau. Most spellF1 deltas have an interval that includes 0 (`w125`
  under the raw truth is the exception, -0.0037 [-0.0063, -0.0011], 5 losses of 7). What `tracking` v2 adds to the
  benchmark is therefore not a different ranking of these four configs but a harder place for a method that reads the
  players far from the ball, follows a slot through a loss or a drop, or needs the count of players to be 22: the
  nearest player to the ball is always kept, so a decoder that reads only that is nearly blind to the cap, as expected.
- The `tracking` v1 figures of this table are R004's reports (the Calibration digest `65b0c1915194…`, version 2, commit
  `39d86cf`); the v2 ones are at `95583d9`.
- The assertion that smoothing lowers the jitter was fragile and is corrected (review round 1, above):
  `test_data_tracking_constants_follow_from_the_measurements` compared the degraded players, cut and smoothed (0.0873 m,
  seed 0), with DFL's clean ones (0.0900 m), a margin of 3% that lambda had used up (with lambda at 2.0 it was gone); it
  now compares the same draws with and without the average, 0.1201 m to 0.0873 m, a fall of 27%. The RMS without the
  average is more than DFL's clean one, which the cuts can do: it is 98% rare jumps in the 2. Bundesliga matches (T35),
  and the pieces and slots change which triples are counted.

Runtime: 93 to 110 s for `fidelity.py` on all seven matches and three seeds (three runs; it was 109 s for v1), and the
runner's own line for the yardstick says 5.4 to 6.0 s per config (the whole run was not timed), with the DFL caches and
the Calibration cached. The last two runs of `fidelity.py`, which are of the final code, print the same figures to
the last digit.
Verdict: keep `tracking` v2 as the benchmark's profile (it is `DEFAULT_PROFILE` now) and the four yardsticks as its
floor, not as candidates. All eight gates and all ten checks hold, with the three count gates counted for the first
time, and the yardstick check passes in its robust part under both truths. Two figures to keep in mind: the tracklet
median is 34% below ours (inside the gate's 50%, after a single rescale that was applied whether or not it would have
passed), and slot identity is better broken than in R004 (48% of the returns to the same slot, against 69%) but still
not ours. Do not choose between `w125` and the others where the raw and the coarse truth disagree. The profile's players
are rougher than ours (jumps, T35), have no positional bias (the README's limitations), take our counts from the frames
of the ball's gaps and not from the same play, and keep the players nearest the true ball: a method that reads only the
nearest player sees little of what this profile adds to `gaps`.

### R004c — The four yardsticks on `tracking` version 3 (2026-10-23, Tian)
Proposal: P01 · Commit: `7fb0a52` (the code every report below was made at, on a clean tree: the eight yardstick reports
say `dirty: false`, and `fidelity.py`, whose report has no `git` field, was run on the same tree with `git status`
empty; the fix itself is `188f578` and `7fb0a52`, and this entry and the docs it corrects are the commits after them) ·
Commands: `python pipeline/benchmark/fidelity.py --profile tracking --seeds 3` first, whose gates all passed, and then
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile tracking --seeds 3`
(the first config is the reference; run once more with `--truth coarse`, and `w25` first with `w125` second and
`--no-write` for the paired comparison, under each truth)
Public: not applicable. The scores read no BAR-ATH data; the profile does: the gaps, the tracklets, the episodes, the
NaN-speed runs, the jitter and the slot counts are our match's own measurements (the Calibration, A07 to A09, digest
`65b0c1915194…`, version 2), and the fidelity checks are compared with its public window's figures (77.8%, 66.4%).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 2,005.3 graded
minutes (668.4 per seed). Raw truth: pooled spellF1 0.2018 (`nearest_w50`), 0.1900 (`w25`), 0.1665 (`w125`), 0.0777
(`w0`). Coarse truth: 0.2226 (`w125`), 0.1842 (`w50`), 0.1432 (`w25`), 0.0342 (`w0`). Intervals, fidelity and sign-flip
p below.
Changed vs the previous best: a benchmark entry, the third with our players: version 3 of the `tracking` profile. One
thing changed, and one constant that followed from it: **a slot is taken by a new tracklet only from the row after its
last tracklet stopped** (versions 1 and 2 let it take the slot on the very row it was freed), and the scale of the
tracked spans was derived again, 2.042 to 2.072. The ball is R004b's, byte for byte (`test_profiles_nest`); the
episodes, the cap, the shortest-piece rule, the smoothing and `WRONG_TEAM_SCALE` are R004b's code and constants, and
their draws follow the pieces, which lambda changed. **The ranking does not move**: both truths order the four configs
as in R004 and R004b, with the same Kendall tau, and no config's pooled spellF1 moves by more than 0.6 points (the
largest are `w125`, +0.4 raw and +0.5 coarse, and `w50`, +0.3 under each). **What does move is the jitter**: 1.80 cm per
slot after the average, not R004b's 8.5 cm, and the docs that blamed the rest on DFL's jumps were wrong (T35, corrected
with this entry).
Reports: `data/runs/benchmark/tracking/<stem>.json` and `<stem>.coarse.json` (R004b's, version 2, were copied to
`tracking_v2/` before this run, and R004's to `tracking_v1/`, for the paired comparison below), and
`data/runs/benchmark/fidelity_tracking.json` (not committed; each carries the Calibration's digest).
Checks (on `7fb0a52`, before the doc edits of this entry; the branch also holds the first part of the `context` profile,
`daf0783`, which this entry does not use): `pipeline/benchmark/selftest.py --dfl --data` passes all 284 checks (262
synthetic, 16 with `--dfl`, 6 with `--data`) in 589 s, `pipeline/selftest.py` 35 of 35 (9 s) and its `--data` run 37 of
37 in 18 s, the four baselines identical byte for byte; the default run of the benchmark's checks was repeated after the
comment and doc edits of this entry (262 passed, 45 s). The checks that changed with the rule:
`test_each_tracklet_takes_the_lowest_free_slot_of_its_pool` and
`test_tracking_speeds_are_finite_differences_within_a_tracklet`, which required same-row takeovers, now require none;
new are `test_a_slot_never_changes_player_between_adjacent_present_rows`,
`test_a_handover_on_the_row_a_slot_is_freed_is_what_the_check_would_catch`,
`test_dfl_a_slot_never_changes_player_on_a_real_match` and `test_jitter_inside_tracklets_counts_what_it_says`. With
version 2's rule patched back into the committed code (`degrade.assign_slots` replaced in a throwaway script), the check
of adjacent rows fails on a made-up match, the real-match one fails on J03WMX seed 0, and the two inverted ones fail
(the check of the check builds its handover by hand and passes with either rule); with the committed rule all four pass,
and J03WMX has no slot that changes player on any of 11,526,850 adjacent present pairs of 4 seeds.

**Why: a same-row handover is a position jump in a slot.** The final review of the benchmark's first half (reviewer B)
found that `assign_slots` freed a slot at the half-open stop of its tracklet and let a piece that starts on that row
take it: the slot holds one player up to row t - 1 and another from row t. Ours has none (3.28 mm of jitter over 3.2
million triples, A07, leaves no room: one jump of 12 m adds 288 m2 to the 34 m2 of squares of all of ours). The finding
is measured here, not taken from the review: version 1's and 2's rule patched back into the committed code (a throwaway
script, not committed: `degrade.assign_slots` replaced by the old three lines, `degraded_players` called as the profile
calls it, with version 2's scale 2.042 for version 2, so that it is R004b's players, 28,880 pieces as R004b's record
has; and with scale 1, no cap and no shortest piece for version 1), the 7 matches x 3 seeds, and the same draws measured
per slot (`calibration.player_stats`) and then only inside tracklets:

| 21 runs: 7 matches x 3 seeds | version 1 (R004) | version 2 (R004b) | version 3 |
|---|---|---|---|
| pieces (tracklets) | 31,398 | 28,880 | 28,743 |
| slot runs that `player_stats` sees (a handover joins two pieces) | 31,162 | 28,508 | 28,743 |
| same-row slot handovers | 236 (11.2 a run, 6 to 16) | 372 (17.7 a run, 9 to 30) | 0 |
| of them one player's own two pieces | 3 | 2 | 0 |
| jump between the two players: median / 90th percentile / largest | 5.3 / 43.0 / 76.5 m | 13.8 / 40.1 / 76.5 m | |
| jitter per slot, without the average | 10.90 cm | 11.99 cm | 8.90 cm |
| jitter per slot, with the 5-frame average | 6.47 cm | 8.49 cm | 1.80 cm |
| jitter inside tracklets, without / with | 8.95 / 1.80 cm | 8.65 / 1.75 cm | 8.90 / 1.80 cm |
| share of the per-slot squares in the triples that straddle a handover, without / with | 33% / 92% | 48% / 96% | 0 |
| triples that straddle a handover (2 each) | 472 of 63,862,354 | 744 of 61,412,002 | 0 |

The old rule reproduces what the earlier entries reported (6.47 cm in R004; in R004b 0.1201 m without the average and
0.0873 m with it on seed 0, 8.5 cm with it on three seeds, against 0.1199 m and 0.0849 m on three seeds here), so the
patched code is their code. **The per-slot jitter of versions 1 and 2 was the handovers**: 744 triples of 61 million
(0.0012%) held 96% of the squares of version 2's jitter after the average, because a jump of 13.8 m between two rows,
spread by the average over five frames, is still metres and not centimetres. Inside the tracklets, version 2 had 1.75 cm
after the average and version 1 1.80 cm, which is what version 3 has per slot. DFL's own jumps are real (T35: 8.90 cm
without the average is DFL's clean 9.00 cm again) and the average takes them down to a fifth, as the arithmetic of a
jump says (a step of J is two triples of +-J, and after a centred 5-frame average two of +-J/5); they are not what R004
and R004b measured, and **no jump filter is needed**.

**The scale of the tracked spans, derived again** (the one constant that the rule reached; `degrade.TRACKED_SCALE`,
once, label-free, with the rule that R004b set: lambda = ours' tracklet median over the pooled median of the profile at
lambda = 1, shortest-piece rule in, seed 0, the 7 matches laid end to end as `fidelity.py` pools them; the measurement
made again here on the committed code, version 3's with `test_data_tracking_constants_follow_from_the_measurements` (it
prints 19.92 s) and version 2's with the old rule patched in, in a throwaway script):

| slot rule | pooled median at lambda = 1 | tracklets | lambda |
|---|---|---|---|
| version 2's (reproduces R004b) | 20.22 s | 13,192 | 41.28 / 20.22 = 2.0415, 2.042 |
| version 3's | 19.92 s | 13,396 | 41.28 / 19.92 = 2.0723, **2.072** |

Ours is 41.28 s over 1,144 tracklets. The old rule joined 204 pairs of pieces into one run of a slot (13,396 pieces,
13,192 runs), longer tracklets in `player_stats`'s eyes, so the measurement moved, and lambda with it: a bug fix
reopened the constant once, and it was applied whether or not the tracklet-median gate would pass (it does) and frozen
on 2026-10-23, with both measurements in the comment of the constant and `RECORDED_MEDIAN_S` 19.92 s beside
`RECORDED_MEDIAN_V2_S` 20.22 s. The median at lambda 2.072 is 26.08 s on seed 0 (the data check) and 26.52 s on the
three seeds (the fidelity report), against R004b's 26.24 s and 27.08 s at 2.042: the ratio is a one-shot on a statistic
that does not grow linearly with lambda, so it does not reach ours, and this one is a little further (-36% against
-34%). `WRONG_TEAM_SCALE` (2.658), the 5-frame average and `MIN_PIECE_FRAMES` (8) stay frozen, and no check said
otherwise. The lost share of the walk, the episodes and the rule, seeds 0 to 4, pooled: 2.72% against d = 2.50% (per
match 2.66%, 2.79%, 2.61%, 2.77%, 2.67%, 2.76%, 2.81%; R004b 2.71%; the check allows 0.3 points).

**Fidelity** (`fidelity.py --profile tracking`, 7 matches x 3 seeds, 98 s with the caches warm; the centres are the
Calibration's own values on our kick-off spans, the tolerances the plan's; nothing was changed after looking at it).
**`tracking` v3 is frozen: all eight gates are counted and pass, `fidelity.py` exits 0**, and **none of the ten checks
warns**:

| gate | benchmark | our match (centre) | tolerance | | R004b (v2) |
|---|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass (+0.0079, 40% of the band) | 0.4189 |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass (+0.02 s, 1%) | 10.76 s |
| exactly 22 slots present, share of full-strength frames | 0.4925 | 0.5019 | +-0.06 | pass (-0.0094, 16%) | 0.4938 |
| mean slots present, full-strength frames | 20.596 | 20.522 | +-0.5 | pass (+0.074, 15%) | 20.599 |
| 14 or fewer slots present, share of full-strength frames | 0.0469 | 0.0525 | +-0.02 | pass (-0.0056, 28%) | 0.0469 |
| wrong-team share of frames | 0.0918 | 0.1054 | +-0.03 | pass (-0.0136, 45%) | 0.0942 |
| tracklet median | 26.52 s | 41.28 s | +-50% (+-20.64 s) | pass (-36%, 72% of the band) | 27.08 s |
| NaN-speed share of the present slot-frames | 0.0788 | 0.0783 | +-0.01 | pass (+0.0005, 5%) | 0.0788 |

The count gates are measured on full-strength frames (T36: 818,710 of 1,002,644 frames, 81.7%; J03WN1 is at full
strength on 7.5% of its frames and J03WQQ on 62.8%). The wrong-team share is the one that moved most (-0.0024; lambda
and the rule both changed the pieces, and which of them moved it was not separated) and the tracklet median is the gate
nearest its edge, as in R004 and R004b. The 42 half-draws took 21 masks from our first half and 21 from our second, and
the draws' record pooled over the 21 runs and both halves says: 5,330 onsets made 3,453 episodes (64.8%), 1,746 skipped
for no opposing pair within 2 m (32.8%) and 131 for a player already in an episode (2.5%) (R004b: 3,447, 1,757, 126);
79,703 NaN-speed runs (R004b: 79,277); the cap: 303,259 frames at or below 18 of our count and 2,363,692 slot-frames
dropped, both as in R004b, in 4,513 pieces cut (R004b: 4,463); the shortest-piece rule: 823 pieces dropped, 3,268
slot-frames, 169 episode pieces (R004b: 794, 3,177, 172); 28,743 tracklets, 1,369 a run (R004b: 1,375; ours 1,144 in a
match). The jitter after the 5-frame smoothing is 1.80 cm (ours 3.3 mm; 5.5 times; T35).

The ten warning checks (none tuned towards): none warns.

| check | benchmark | our match | band | | R004b (v2) |
|---|---|---|---|---|---|
| nearest player agrees with the truth, alive and ball visible | 0.7609 | 0.7784 (public window) | +-0.05 | ok | 0.7586 |
| ball visible on dead frames | 0.5684 | 0.6637 (public window) | +-0.10 | ok | 0.5684 |
| `w0` possessions per graded minute | 13.92 | 18.40 | +-25% | ok (-24.3%) | 13.91 |
| `w0` D share | 0.4189 | 0.4069 | +-0.05 | ok | 0.4189 |
| `w25` possessions per graded minute | 5.36 | 7.04 | +-25% | ok (-23.9%) | 5.39 |
| `w25` D share | 0.3764 | 0.3649 | +-0.05 | ok | 0.3764 |
| `w50` possessions per graded minute | 4.30 | 5.61 | +-25% | ok (-23.3%) | 4.32 |
| `w50` D share | 0.3341 | 0.3234 | +-0.05 | ok | 0.3341 |
| `w125` possessions per graded minute | 3.05 | 3.76 | +-25% | ok (-18.7%) | 3.06 |
| `w125` D share | 0.2329 | 0.2233 | +-0.05 | ok | 0.2329 |

The WARN rows moved by almost nothing (the D shares and the ball-visible figure depend on the ball alone, and are
R004b's to the last digit); `w0` is the nearest its edge (-24.3% of ours against a band of 25%), as before (-24.4%).

**Slot persistence and the degraded match against ours (information; J03WMX, seed 0, our real Calibration, from the data
check at `7fb0a52`; version 2's column is R004b's).** d is 0.0250, the mean tracked span 5,796 frames (231.8 s) after
lambda and the mean lost span 148.6 frames (5.94 s); the cap reads 20,031 of the 145,967 frames from the kick-offs
(13.7%) and drops 164,572 slot-frames in 234 pieces; the match loses 7.66% of its 3,211,274 on-pitch frames.

| | v3 | v2 (R004b) | ours |
|---|---|---|---|
| slots present, mean | 20.31 | 20.29 | 20.52 |
| all 22 present | 48.7% | 48.4% | 50.2% |
| 14 or fewer present | 6.7% | 6.7% | 5.2% |
| tracklets | 1,448 | 1,454 | 1,144 |
| tracklet median | 21.8 s | 22.8 s | 41.3 s |
| speeds NaN of the present | 7.91% | 7.85% | 7.83% |
| wrong-team frames | 8.5% | 8.7% | 10.5% |
| slot idle gaps, count / median | 1,397 / 4.32 s | 1,402 / 4.12 s | 1,065 / 7.40 s |
| idle at most 3 s / over 60 s | 39.4% / 5.3% | 42.0% / 5.3% | 17.7% / 15.0% |
| returns to the same slot | 529 of 1,117 (47.4%) | 552 of 1,147 (48.1%) | no identities |
| the same without the cap | 380 of 562 (67.6%) | 384 of 598 (64.2%) | |
| slots ever present, home / away of 24 / 22 | 13 / 13 | 14 / 13 | 22 / 21 |
| present on at least a tenth of the frames | 12 / 12 | 12 / 12 | 21 / 20 |
| present on at least a hundredth | 13 / 12 | 12 / 13 | 22 / 21 |

Slot identity is no more ours than it was (the returns to the same slot, 47% against 48%), as the warning of A09, R004
and R004b says: **a method that follows a slot across a loss would be validated optimistically on this profile**.

**Wrong-team episodes as `player_stats` sees them (seed 0, at the frozen scale; the data check at `7fb0a52`, and the
same measurement with version 2's rule patched in).** Version 3: 128, 75, 115, 135, 148, 112 and 159 episodes in J03WMX,
J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9: **75 to 159 a match, mean 124.6**, medians 2.0 to 2.8 s and 90th
percentiles 8.2 to 14.5 s, against ours **104**, 2.68 s and 15.79 s. Version 2: 119, 82, 118, 133, 134, 114, 151: 82 to
151, mean 121.6, medians 1.8 to 2.7 s, 90th percentiles 9.4 to 14.4 s: the figures that the README quoted for the
profile until now (82 to 151, mean 122), which no entry logged and which were version 2's.

**Raw truth** (9,183 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config minus
`nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.2018** | 0.4592 | 0.6094 | 0.2023 [0.1927, 0.2118] | 8,623 (4.30) | | | |
| `nearest_w25` | 0.1900 | 0.4567 | 0.5935 | 0.1903 [0.1839, 0.1967] | 10,755 (5.36) | -0.0120 [-0.0176, -0.0063] | 0.0312 | 1 / 6 |
| `nearest_w125` | 0.1665 | 0.4407 | 0.6428 | 0.1676 [0.1516, 0.1835] | 6,124 (3.05) | -0.0347 [-0.0450, -0.0244] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0777 | 0.4405 | 0.5761 | 0.0777 [0.0722, 0.0831] | 27,921 (13.92) | -0.1246 [-0.1383, -0.1110] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 4,977 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.2226** | 0.4509 | 0.6428 | 0.2228 [0.2062, 0.2395] | 6,124 (3.05) | +0.0382 [+0.0279, +0.0486] | 0.0156 | 7 / 0 |
| `nearest_w50` (reference) | 0.1842 | 0.4602 | 0.6094 | 0.1846 [0.1747, 0.1945] | 8,623 (4.30) | | | |
| `nearest_w25` | 0.1432 | 0.4555 | 0.5935 | 0.1434 [0.1385, 0.1482] | 10,755 (5.36) | -0.0412 [-0.0490, -0.0335] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0342 | 0.4390 | 0.5761 | 0.0344 [0.0307, 0.0382] | 27,921 (13.92) | -0.1502 [-0.1607, -0.1396] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The yardstick check holds in its robust part: `nearest_w0` is last under both truths**, by far more than the noise
  (7 losses of 7 against `w50` under each, Δ -0.1246 raw and -0.1502 coarse, and the gap from `w0` to the next config is
  0.0888 raw and 0.1090 coarse), and in each of the three seeds (the gap by seed, 0 / 1 / 2: 0.0912 / 0.0887 / 0.0863
  raw, 0.1052 / 0.1083 / 0.1136 coarse). Nothing had to be fixed or changed to get it.
- **The rest of the order follows the truth, as on `clean`, `gaps`, R004 and R004b, and neither truth gives the
  published order.** Raw: `w50 > w25 > w125 > w0`; coarse: `w125 > w50 > w25 > w0`; published (R000, full match): `w50 >
  w125 > w25 > w0`. **Kendall tau against the published order is +0.667 under each truth** (5 of the 6 pairs concordant,
  1 discordant), as in R004b. Each truth keeps the same order in each of the three seeds (pooled spellF1 by seed, 0 / 1
  / 2: raw `w50` 0.2006 / 0.1988 / 0.2058, `w25` 0.1847 / 0.1879 / 0.1976, `w125` 0.1657 / 0.1655 / 0.1681, `w0` 0.0745
  / 0.0769 / 0.0818; coarse `w50` 0.1788 / 0.1831 / 0.1907, `w25` 0.1394 / 0.1403 / 0.1501, `w125` 0.2148 / 0.2267 /
  0.2263, `w0` 0.0342 / 0.0320 / 0.0365). The seeds move a pooled score by up to about 0.013, and the intervals above
  are over the matches and say nothing of that spread.
- **The published gaps are inside the noise (about 0.04), and the two that involve `w125` lie between the raw and the
  coarse ones.** Pooled spellF1 gaps, published against raw against coarse: `w50` - `w25` +0.044 / +0.012 / +0.041;
  `w125` - `w25` +0.025 / -0.024 / +0.079; `w50` - `w125` +0.019 / +0.035 / -0.038. Paired over the matches, `w125`
  against `w25` is -0.0228 [-0.0346, -0.0109], 0 wins of 7 (7 losses), under the raw truth (macroF1 -0.0165 [-0.0240,
  -0.0090], 1 win), and +0.0795 [+0.0636, +0.0954], 7 wins of 7, under the coarse one (macroF1 -0.0052 [-0.0135,
  +0.0032], 2 wins, p 0.2656) (`... nearest_w25.toml ... nearest_w125.toml --profile tracking --seeds 3 --no-write`,
  with `--truth coarse` for the second). That the published gaps that involve `w125` lie between the raw and the coarse
  one fits A06's reading (DFL's grain is finer than ours, and the coarse truth stands in for how much); it is two
  numbers, not a measurement of ours, and it is not used to choose anything (T26).
- **The choice of truth decides `w125` against `w50` and `w25`, not the profile.** The sign of every paired delta
  against `w50` is the same on both competitions (BL1: the two Bundesliga matches, BL2: the five 2. Bundesliga ones):
  raw `w125` -0.0440 on BL1 and -0.0310 on BL2, coarse +0.0389 and +0.0380; `w25` -0.0092 and -0.0131 raw, -0.0367 and
  -0.0431 coarse; `w0` -0.1113 and -0.1300 raw, -0.1505 and -0.1500 coarse. Mean spellF1 per competition (BL1, BL2): raw
  `w50` 0.1963, 0.2046; `w25` 0.1872, 0.1916; `w125` 0.1524, 0.1736; `w0` 0.0850, 0.0747; coarse `w50` 0.1833, 0.1851;
  `w25` 0.1467, 0.1420; `w125` 0.2222, 0.2231; `w0` 0.0329, 0.0351.
- **`w25` against `w50` is the one raw gap that is not decided**: -0.0120 [-0.0176, -0.0063], p 0.0312, 1 win in 7
  (macroF1: -0.0025 [-0.0047, -0.0003], 3 wins in 7, p 0.1250): about a hundredth of a point, which the interval
  excludes by a little; it is not a result.

**How R004c differs from R004b** (only the slot rule and lambda changed: the ball, the seeds, the matches, the configs
and the scorer are the same, so the two are paired by match; `tracking` v3 minus v2, from the two sets of reports, over
the matches; the seeds are the same but the draws are not, since lambda moved, so the pairing removes the match and not
the draw):

| config | raw: pooled spellF1 v2 to v3 | Δ mean per match [90% t] | wins / losses | coarse: pooled spellF1 | Δ mean per match [90% t] | wins / losses |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.1989 to 0.2018 | +0.0030 [-0.0005, +0.0065] | 5 / 2 | 0.1811 to 0.1842 | +0.0031 [+0.0010, +0.0051] | 6 / 1 |
| `nearest_w25` | 0.1897 to 0.1900 | +0.0005 [-0.0026, +0.0036] | 4 / 3 | 0.1422 to 0.1432 | +0.0011 [-0.0001, +0.0022] | 6 / 1 |
| `nearest_w125` | 0.1624 to 0.1665 | +0.0041 [+0.0011, +0.0071] | 6 / 1 | 0.2173 to 0.2226 | +0.0053 [+0.0024, +0.0082] | 6 / 1 |
| `nearest_w0` | 0.0779 to 0.0777 | -0.0001 [-0.0013, +0.0010] | 4 / 3 | 0.0341 to 0.0342 | +0.0002 [-0.0005, +0.0008] | 4 / 3 |

- **The rule changes the four yardsticks by 0.6 points of spellF1 at most and about 0.1 to 0.2 points of macroF1**
  (macroF1 by config +0.0009 to +0.0014 raw and +0.0012 to +0.0018 coarse, up for every config; the 90% interval
  excludes 0 for `w50`, `w25` and `w0` under the raw truth and for all but `w125` under the coarse one), and **changes
  no order**: both truths rank the four configs as in R004b, with the same Kendall tau. One spellF1 delta has an
  interval that excludes 0 under the raw truth (`w125`, +0.0041, 6 wins of 7) and two under the coarse one (`w50`
  +0.0031 and `w125` +0.0053; `w25`'s interval ends at -0.0001). The direction is what one expects of a profile with
  fewer position jumps and tracklets that are a little longer (28,743 pieces against 28,880), and the size is inside the
  seeds' spread (about 0.013): it is not a result about any config.
- The `tracking` v2 figures of this table are R004b's reports (`tracking_v2/`, the copy made before this run, byte for
  byte R004b's: pooled 0.1989, 0.1897, 0.1624 and 0.0779 raw, 0.1811, 0.1422, 0.2173 and 0.0341 coarse, as R004b says);
  the v3 ones are at `7fb0a52`.

Runtime: 98 s for `fidelity.py` on all seven matches and three seeds (R004b: 93 to 110 s); the runner's own line says
5.1 to 6.1 s per config, 62 s and 66 s for the four configs under the raw and the coarse truth and 55 s and 54 s for the
paired two-config runs, with the DFL caches and the Calibration cached; the suites 589 s, 9 s and 18 s (above).
Verdict: keep `tracking` v3 as the benchmark's profile (`DEFAULT_PROFILE` is still `tracking`, which now means version
3; whether `context` takes it over is R005's) and the four yardsticks as its floor, not as candidates. All eight gates
and all ten checks hold, and the yardstick check passes in its robust part under both truths. What version 3 corrects is
the account of the players, not the ranking: **the jitter per slot is 1.80 cm and not 8.5 cm, five and a half times ours
and not twenty-six, and every figure of R004 and R004b that blamed the rest on DFL's own jumps was a figure of slot
handovers** (errata added to both, T35 rewritten, the README corrected). Still true of the profile, and unchanged by
this entry: its slots are not ours (47% of the returns to the same slot, ours has no identities; a method that follows a
slot across a loss would be validated optimistically), its tracklet median is 36% below ours, it has no positional bias,
and it takes our counts from the frames of the ball's gaps and not from the same play. Do not choose between `w125` and
the others where the raw and the coarse truth disagree.

### R005 — The `context` profile fails two gates; its four yardsticks, as a sensitivity check only (2026-10-23, Tian)
Proposal: P01 · Commit: `a34e5f6` (the code every report below was made at, on a clean tree: the eight yardstick reports
say `dirty: false`, and `fidelity.py`, whose report has no `git` field, was run on the same commit with `git status`
empty; the later commits are `02c5cb8` (the README, proposals fact 1 and a pin check on the default), the merge
`28239f0` of tracking version 3's docs and `a75b7ad` (two comments in `degrade.py`), and none changes what the profile
runs: `git diff a34e5f6 HEAD` on `degrade.py` is those two comments, and `fidelity.py`, `hazard.py` and `calibration.py`
are untouched) · Commands: `python pipeline/benchmark/fidelity.py --profile context --seeds 3` first, which exits 1, and
then `python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w0.toml pipeline/benchmark/configs/nearest_w25.toml pipeline/benchmark/configs/nearest_w125.toml --profile context --seeds 3`
(the first config is the reference; run once more with `--truth coarse`, and `w25` first with `w125` second and
`--no-write` for the paired comparison, under each truth)
Public: not applicable. The scores read no BAR-ATH data; the profile does: the hazard that places the gaps is fitted on
our own match's gaps (A08), the players' tracklets, episodes, NaN-speed runs and slot counts are our measurements as in
R004c, and the fidelity checks are compared with its public window's figures (77.8%, 66.4%). Calibration digest
`65b0c1915194…` (version 2), as R004c.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `context` version 1, 2,005.3 graded
minutes (668.4 per seed), as R004c. Raw truth: pooled spellF1 0.2193 (`nearest_w50`), 0.2097 (`w25`), 0.1774 (`w125`),
0.0812 (`w0`). Coarse truth: 0.2443 (`w125`), 0.2057 (`w50`), 0.1608 (`w25`), 0.0369 (`w0`).
Changed vs the previous best: a benchmark entry against R004c, and **only the ball changed**. R004c's `tracking` v3
gives the DFL matches our players and puts the holes in the ball where our own match's masks are, transplanted with
their pattern and not their clock; `context` gives them the same players, draw for draw
(`test_context_players_equal_tracking`), and puts the holes where a hazard fitted on our match says our tracker loses
the ball (the distance to the nearest line, whether the ball is beyond it, its speed, the nearest player, the players
within 3 m and the minute of the half; A08). The seeds, the matches, the configs, the truths and the scorer are R004c's.
**`context` fails two of its nine gates and one warning check, and is not the benchmark's default: these tables are a
sensitivity check, and `DEFAULT_PROFILE` stays `tracking` (version 3).** The failure comes first, the scores after it.
Reports: `data/runs/benchmark/context/<stem>.json` and `<stem>.coarse.json`, and
`data/runs/benchmark/fidelity_context.json` (not committed; each carries the Calibration's digest). R004c's `tracking`
v3 reports (`data/runs/benchmark/tracking/`, at `7fb0a52`) are what the comparison at the end reads.
Checks (run on `a75b7ad`, the tree these docs sit on; the reports are at `a34e5f6`, and the code between them is the
two comments, the pin check and the `selftest.py` timing line named above): `pipeline/benchmark/selftest.py --dfl
--data` passes all 287 checks (264 synthetic, 16 with `--dfl`, 7 with `--data`) in 265 s, the default run 264 passed
(46 s), `pipeline/selftest.py` 35 of 35 (5 s) and its `--data` run 37 of 37 in 9 s, the four baselines identical byte
for byte. `checks/context.py` holds 16 of them (14 synthetic, 2 with `--data`), among them the pin on the default, the
frozen `HAZARD_SCALE` with its two recorded measurements and `test_context_players_equal_tracking`; the report's JSON is
the same under `PYTHONHASHSEED` 1, 2 and 3, as `tracking`'s is (`test_runner_context_json_identical_across_processes`,
in `checks/run.py`).

**The one constant, rescaled once** (`degrade.HAZARD_SCALE`, frozen at 0.699 on 2026-10-23 in `a34e5f6`, the commit
the runs are on; both measurements are in the comment of the constant and pinned by
`test_hazard_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements`, and
`test_data_hazard_scale_follows_from_the_measurement` measures them again). The chance that the next frame starts a gap
is `hazard_c` x `HAZARD_SCALE` x the model's probability. At the scale of 1.0, seed 0, the 7 matches laid end to end as
`fidelity.py` pools them, 501,013 of 1,002,644 frames lack the ball (0.4997) against our 0.4110 (the Calibration's
`missing_share`). The rescale is on the odds, scale x [0.4110 / (1 - 0.4110)] / [0.4997 / (1 - 0.4997)] = 0.6986,
rounded to 0.699 (gaps start only from visible frames, so the odds are about linear in the hazard, and a plain ratio,
0.822, would have undershot). At 0.699 the share on seed 0 is 0.4236. It was set on seed 0 alone, once, and not moved
after seeds 1 and 2 had been seen.

**The failure: two gates, on every seed or pooled.** `fidelity.py --profile context --seeds 3`, 7 matches x 3 seeds,
exits 1 ("9 gates DO NOT ALL PASS; 1 of 10 checks warn"). The centres are the Calibration's own values on our kick-off
spans and the tolerances the plan's; nothing was changed after looking at it. Pooled, and by seed (0, 1 and 2):

| gate | pooled | seed 0 | seed 1 | seed 2 | our match (centre) | tolerance | |
|---|---|---|---|---|---|---|---|
| ball missing, share of frames | 0.4313 | 0.4236 | 0.4275 | 0.4427 | 0.4110 | +-0.02 | **FAIL** pooled (+0.0203, by 0.0003) and on seed 2 (+0.0317); seeds 0 (+0.0126) and 1 (+0.0165) pass |
| median gap in the ball | 11.88 s | 11.78 s | 11.68 s | 12.04 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, full-strength frames | 0.4925 | 0.5090 | 0.4960 | 0.4724 | 0.5019 | +-0.06 | pass |
| mean slots present, full-strength frames | 20.596 | 20.553 | 20.662 | 20.572 | 20.522 | +-0.5 | pass |
| 14 or fewer slots present, full-strength frames | 0.0469 | 0.0514 | 0.0430 | 0.0462 | 0.0525 | +-0.02 | pass |
| wrong-team share of frames | 0.0918 | 0.0932 | 0.0978 | 0.0845 | 0.1054 | +-0.03 | pass |
| tracklet median | 26.52 s | 26.08 s | 28.08 s | 26.08 s | 41.28 s | +-50% | pass |
| NaN-speed share of the present slot-frames | 0.0788 | 0.0787 | 0.0791 | 0.0787 | 0.0783 | +-0.01 | pass |
| **gaps that start within 2 m of a line** (`near_line_share`) | **0.5586** | 0.5647 | 0.5644 | 0.5465 | 0.2797 | +-0.05 | **FAIL** pooled (+0.2789) and on each seed (+0.2850, +0.2847, +0.2668) |

The six gates on the players pass and equal R004c's `tracking` v3 figures to the last digit (the players are v3's, draw
for draw); of the two on the gaps, the median gap passes (11.88 s; v3's 10.76 s) and the share missing fails (v3's
0.4189 passed). `near_line_share` is the ninth gate, the one that `context` adds: the share of the gaps that begin
within 2 m of a line at the frame before them, counted the way A08 counts our own gaps (40 of our 143 onsets).
`tracking` has no such gate, and its own share is 0.3597 (the information row of its fidelity report; R003: 36.0% on the
same ball), outside +-0.05 of ours too: the gate was pre-registered for `context`, the profile that claims to place the
gaps by the play, and the transplant is blind to it by design. The
count gates are measured on full-strength frames (T36: 818,710 of 1,002,644 frames,
81.7%). The statistics beside the gates, benchmark against ours (information): gaps per 10,000 frames 9.57 against 9.36,
gap length p10 / median / p90 5.40 / 11.88 / 31.52 s against 2.68 / 10.74 / 35.30 s.

**The stop conditions did not trip** (the limits are the plan's: the profile is not to be run if a stop condition
holds):

| stop condition | limit | measured |
|---|---|---|
| share of the at-risk rows within five frames after a gap (the ball's speed there is the true one, where ours carries the speed from before the gap) | 2% | 0.83% pooled (0.84%, 0.81%, 0.84% by seed): 14,227 of 1,710,689 rows |
| rows where a feature is undefined, so that the chance is 0 | 1% | 42 of 3,007,932 rows (0.0014%) |
| rows whose chance reached the cap (1 - 1e-9) | a count; 0 means the cap never bit | 0 |

The draws' record pooled over the 21 runs and both halves: 2,880 onsets, 1,297,180 missing frames, 1,710,689 at-risk
rows.

**The warning, next to `tracking` v3's.** One of the ten warning checks warns: **the ball visible on dead frames is
0.4622** (0.4754, 0.4634 and 0.4478 by seed) against 0.6637 on our public window (band +-0.10: -0.2015 from the window's
figure, 0.1015 outside the band, a WARN), where `tracking` v3's was 0.5684 (R004c: -0.0953, inside the band). **The new
ball hides the ball on dead frames more than v3's does** (0.4622 against 0.5684). **What the row does not say is how
either compares with our match over the match.** Its centre is the public window's: the ball is missing on 11.8% of that
window's frames (A01) against about 41% of the graded frames of the match, the window is the best-covered block of our
match, and the dead frames of the rest of it have no labels (and, by the rules, cannot have), so our own visible-on-dead
share over the match is unmeasured. It is expected to be lower than the window's, by an amount that cannot be measured,
so the row shows the benchmark's share against the window's, and some of its -0.2015 is the window's (the README says
the same). The other nine checks are ok: nearest player agrees with the truth 0.7627 against 0.7784 (v3: 0.7609);
predicted possessions per graded minute `w0` / `w25` / `w50` / `w125` 14.66 / 5.49 / 4.37 / 3.07 against ours 18.40 /
7.04 / 5.61 / 3.76 (-20.4%, -22.1%, -22.1%, -18.4%, inside +-25%; v3: -24.3%, -23.9%, -23.3%, -18.7%); D shares 0.4313 /
0.3863 / 0.3416 / 0.2308 against 0.4069 / 0.3649 / 0.3234 / 0.2233 (+0.0243, +0.0214, +0.0182, +0.0075, inside +-0.05).

**Diagnosis: the model is fine, the rows it is applied to are not (`nearline_diag`, a throwaway script run on seed 0;
not committed).** The model's probability is higher near a line on both matches, by a similar factor, and the rows are
what differ: DFL tracks the ball out of play, ours does not.

| | at-risk rows within 2 m of a line | at-risk rows beyond a line | mean model probability p, near a line / elsewhere (ratio) | onsets within 2 m of a line |
|---|---|---|---|---|
| ours (the model's own fit: 91,840 at-risk rows, 143 onsets) | 9.01% | 4.07% | 0.00389 / 0.00133 (2.9) | 40 of 143 (0.2797) |
| DFL, 7 matches, seed 0, at 0.699 (577,949 at-risk rows, 986 onsets) | 25.23% | 17.68% | 0.00379 / 0.00098 (3.9) | 557 of 986 (0.565) |

With a similar p near a line, a match that has 25% of its at-risk rows there and 18% beyond a line puts 56% of its
gaps there, against our 28%: **the share of gaps near a line fails by the rows and not by the fit**, and no scale of
the hazard moves it (the ratios, 2.9 and 3.9, are the same at any scale).

**Two remedies were measured and not applied** (seed 0, 7 matches, at 0.699, with the change that the first column
names; to see what would move the gate, not to pass it):

| variant | ball missing | interior gaps | near-line share |
|---|---|---|---|
| as run (`context` v1) | 0.4236 | 981 | 0.5647 |
| chance 0 where the clean ball is beyond a line | 0.2408 | 578 | 0.1626 |
| the line distance clipped at 0 and `beyond_line` = 0 in the model's input | 0.3803 | 850 | 0.4894 |

Neither passes. The first overshoots the near-line gate the other way (0.1626 against 0.2797, -0.117, outside +-0.05)
and drops the share missing to 0.2408 (-0.170), outside its band: it takes the gaps away from the part of the ball
that is out of play altogether, and what a rescale that brought the missing share back would then do to the near-line
share was not measured. The second (0.4894, +0.210; missing 0.3803, -0.031) is still outside both bands. Two more were
considered and rejected before any run: thinning the chance beyond a line to our share of such rows (a factor of
0.23), which would pass the gate by leaving more of the out-of-play ball visible, so the visible rows would move
further from ours: tuning towards a gate; and a mask for the ball off the pitch, a new mechanism with a parameter that
cannot be measured on ours, since where our ball is during our gaps is unobservable. No second rescale of
`HAZARD_SCALE` either: the rule is once.

**The ruling, taken on seeing this (2026-10-23).** No fix: the profile stays as it is (version 1), `HAZARD_SCALE` stays
0.699, no gate is loosened or moved and no remedy is applied. `DEFAULT_PROFILE` stays `tracking`
(`test_default_profile_is_tracking_because_context_fails_a_gate` pins it, and that `near_line_share` stays a required
`context` gate, with the measured pair 0.5586 and 0.2797). The yardsticks were run on `context` for one purpose: to see
whether the ranking of the configs depends on how the gaps are placed. **They are a sensitivity check, not a validation
score: no number from `context` replaces one from `tracking` v3 or is compared with a number from another profile
without saying so.**

**Raw truth** (9,183 true possessions, 4.58 per graded minute; `D` is 44.8% of the graded frames). Δ is the config minus
`nearest_w50`, the first config; the interval is the 90% t interval of the seven matches' deltas.

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (reference) | **0.2193** | 0.5144 | 0.6575 | 0.2197 [0.2074, 0.2320] | 8,762 (4.37) | | | |
| `nearest_w25` | 0.2097 | 0.5198 | 0.6516 | 0.2099 [0.2002, 0.2195] | 11,010 (5.49) | -0.0098 [-0.0139, -0.0058] | 0.0156 | 0 / 7 |
| `nearest_w125` | 0.1774 | 0.4770 | 0.6711 | 0.1786 [0.1626, 0.1946] | 6,151 (3.07) | -0.0411 [-0.0515, -0.0307] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0812 | 0.5103 | 0.6453 | 0.0812 [0.0753, 0.0871] | 29,393 (14.66) | -0.1385 [-0.1563, -0.1208] | 0.0156 | 0 / 7 |

**Coarse truth** (`COARSE_MIN_S` = 5 s: 4,977 true possessions, 2.48 per graded minute; the same `D`). Δ is again the
config minus `nearest_w50`:

| config | pooled spellF1 | macroF1 | alive/dead F1 | mean per-match spellF1 [90% t] | possessions predicted (per min) | Δ spellF1, mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w125` | **0.2443** | 0.4877 | 0.6711 | 0.2447 [0.2286, 0.2608] | 6,151 (3.07) | +0.0388 [+0.0256, +0.0520] | 0.0156 | 7 / 0 |
| `nearest_w50` (reference) | 0.2057 | 0.5160 | 0.6575 | 0.2059 [0.1974, 0.2144] | 8,762 (4.37) | | | |
| `nearest_w25` | 0.1608 | 0.5191 | 0.6516 | 0.1608 [0.1542, 0.1675] | 11,010 (5.49) | -0.0451 [-0.0500, -0.0401] | 0.0156 | 0 / 7 |
| `nearest_w0` | 0.0369 | 0.5088 | 0.6453 | 0.0371 [0.0330, 0.0411] | 29,393 (14.66) | -0.1688 [-0.1793, -0.1583] | 0.0156 | 0 / 7 |

What the tables say, and what they do not:
- **The yardstick check holds in its robust part: `nearest_w0` is last under both truths**, by far more than the noise
  (7 losses of 7 against `w50` under each, Δ -0.1385 raw and -0.1688 coarse, and the gap from `w0` to the next config is
  0.0962 raw and 0.1239 coarse). Per-seed yardstick scores were not computed for `context`, so, unlike R004c, nothing is
  claimed about the order within each seed; the intervals above are over the matches and say nothing of the seeds'
  spread.
- **The rest of the order follows the truth, as on every profile before it, and neither truth gives the published
  order.** Raw: `w50 > w25 > w125 > w0`; coarse: `w125 > w50 > w25 > w0`; published (R000, full match): `w50 > w125 >
  w25 > w0`. **Kendall tau against the published order is +0.667 under each truth** (5 of the 6 pairs concordant, 1
  discordant), as in R004b and R004c.
- **The choice of truth decides `w125` against `w50` and `w25`, here as there.** Paired, `w125` against `w25` is -0.0313
  [-0.0432, -0.0193], 0 wins of 7, under the raw truth (macroF1 -0.0428 [-0.0473, -0.0383], 0 wins), and +0.0839
  [+0.0674, +0.1003], 7 wins of 7, under the coarse one (macroF1 -0.0312 [-0.0364, -0.0260], 0 wins) (`...
  nearest_w25.toml ... nearest_w125.toml --profile context --seeds 3 --no-write`, with `--truth coarse` for the
  second). The sign of every paired delta against `w50` is the same on both competitions (BL1: the two Bundesliga
  matches, BL2: the five 2. Bundesliga ones): raw `w125` -0.0489 on BL1 and -0.0380 on BL2, coarse +0.0341 and
  +0.0407; `w25` -0.0053 and -0.0117 raw, -0.0399 and -0.0471 coarse; `w0` -0.1221 and -0.1451 raw, -0.1702 and
  -0.1682 coarse.
- **`w25` against `w50` under the raw truth is 0 wins of 7 here** (-0.0098 [-0.0139, -0.0058], p 0.0156) against v3's
  1 win of 7 (-0.0120 [-0.0176, -0.0063], p 0.0312): about a hundredth of a point and the same sign, which, as in
  R004c, is not a result. macroF1 goes the other way for `w25` against `w50` (+0.0053 [+0.0025, +0.0080], 6 wins of 7,
  raw; +0.0029 [+0.0003, +0.0055], 6 wins, p 0.0781, coarse): the ranking metric and the tiebreaker disagree there.
- The published gaps are inside the noise (about 0.04 on the full match), and nothing here is used to choose a config
  (T26).

**How R005 differs from R004c, and the agreement between `context` and `tracking` v3** (only the ball's holes changed:
`context` minus v3 from the two sets of reports, over the 7 matches, the same matches and seeds and, by
`test_context_players_equal_tracking`, the same players; the pairing removes the match and the players but not the
ball's draw, which is the thing that changed). The two ranks of the four configs on pooled spellF1 agree completely
under each truth: **Kendall tau between the two profiles' orders is +1.000 under the raw truth and +1.000 under the
coarse one** (6 of the 6 pairs concordant; `w50 > w25 > w125 > w0` raw and `w125 > w50 > w25 > w0` coarse, on both
profiles), and +1.000 on alive/dead F1 (`w125 > w50 > w25 > w0`, both profiles). On macroF1, which is only the
tiebreaker, it is +0.333 under each truth (4 concordant, 2 discordant: `w25` passes `w50`, and `w0` passes `w125`, which
v3 had 0.0002 apart under the raw truth).

| config | raw: pooled spellF1 v3 to `context` | Δ mean per match [90% t] | sign-flip p | wins / losses | coarse: pooled spellF1 | Δ mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 to 0.2193 | +0.0174 [+0.0091, +0.0258] | 0.0156 | 7 / 0 | 0.1842 to 0.2057 | +0.0213 [+0.0127, +0.0299] | 0.0156 | 7 / 0 |
| `nearest_w25` | 0.1900 to 0.2097 | +0.0195 [+0.0143, +0.0248] | 0.0156 | 7 / 0 | 0.1432 to 0.1608 | +0.0175 [+0.0106, +0.0243] | 0.0156 | 7 / 0 |
| `nearest_w125` | 0.1665 to 0.1774 | +0.0110 [+0.0075, +0.0146] | 0.0156 | 7 / 0 | 0.2226 to 0.2443 | +0.0219 [+0.0155, +0.0282] | 0.0156 | 7 / 0 |
| `nearest_w0` | 0.0777 to 0.0812 | +0.0035 [+0.0010, +0.0061] | 0.0469 | 6 / 1 | 0.0342 to 0.0369 | +0.0026 [+0.0009, +0.0044] | 0.0312 | 6 / 1 |

| config | macroF1 v3 to `context` (raw truth) | Δ mean per match [90% t], wins / losses (7 / 0 each) | macroF1 (coarse truth) | Δ mean per match [90% t] | alive/dead F1 | predicted D share | predicted possessions per min |
|---|---|---|---|---|---|---|---|
| `nearest_w50` | 0.4592 to 0.5144 | +0.0545 [+0.0452, +0.0637] | 0.4602 to 0.5160 | +0.0551 [+0.0458, +0.0644] | 0.6094 to 0.6575 | 0.3341 to 0.3416 | 4.30 to 4.37 |
| `nearest_w25` | 0.4567 to 0.5198 | +0.0622 [+0.0524, +0.0720] | 0.4555 to 0.5191 | +0.0626 [+0.0529, +0.0722] | 0.5935 to 0.6516 | 0.3764 to 0.3863 | 5.36 to 5.49 |
| `nearest_w125` | 0.4407 to 0.4770 | +0.0359 [+0.0285, +0.0434] | 0.4509 to 0.4877 | +0.0365 [+0.0292, +0.0438] | 0.6428 to 0.6711 | 0.2329 to 0.2308 | 3.05 to 3.07 |
| `nearest_w0` | 0.4405 to 0.5103 | +0.0688 [+0.0584, +0.0792] | 0.4390 to 0.5088 | +0.0688 [+0.0584, +0.0792] | 0.5761 to 0.6453 | 0.4189 to 0.4313 | 13.92 to 14.66 |

- **The order does not depend on how the gaps are placed, under the two ways we have of placing them.** Transplanting
  our masks with their pattern and not their clock (R004c) and drawing the gaps from a hazard fitted on our match give
  the same order of the four yardsticks under each truth, with the same Kendall tau against the published order
  (+0.667), and the same `w0` last by a margin that is far outside the noise. That is what `context` was kept for, and
  it is all that it is evidence for.
- **The sizes are not the same, and they all move the same way.** Every config scores higher on `context`: spellF1 by
  +0.0027 (`w0`, coarse) to +0.0217 (`w125`, coarse) pooled, with `w50`, `w25` and `w125` winning in 7 matches of 7
  under both truths and `w0` in 6 of 7; macroF1 by +0.036 to +0.070 pooled, in 7 of 7 matches for every config under
  both truths; alive/dead F1 by +0.028 to +0.069. R004c's seeds moved a pooled spellF1 by up to about 0.013 (v3 only: no
  per-seed score exists for `context`): the deltas of `w0` are under that, and the other six, +0.011 to +0.022, are of
  its size or up to 1.7 times it, where R004c's v3-against-v2 differences (0.6 points at most) were inside it.
  **The mechanism was not measured.** What the numbers do say is that `context` hides the ball on dead frames more than
  v3 does (0.4622 visible against v3's 0.5684) and that the yardsticks fall back to `D` where the window of a frame
  holds no ball (`fallback = "D"` in the configs), so a profile that loses the ball more on dead frames may be one on
  which this rule does better. That is a reading and not a result: the predicted D share rose for `w0`, `w25` and `w50`,
  by +0.0124, +0.0100 and +0.0075, and fell by 0.0021 for `w125`, whose alive/dead F1 rose all the same. If it is the
  cause, **a method that reads the ball's absence as a sign of a dead ball would be validated more optimistically on
  `context` than on `tracking` v3**. Which of the two is nearer our match on this count is not known (the warning above:
  the 0.6637 is the public window's, and the match is worse covered than the window), so this says which profile
  flatters such a method more, and not how far either flatters it beyond our match.
- **`context` is not a better copy of our match than `tracking` v3, on the figures we can measure.** It was built to be
  nearer in one respect: its gaps follow the play in the model's six features, as ours do, which `tracking`'s
  transplanted masks cannot. The near-line gate says that the way it does so is not ours, and v3 is not nearer on that
  count either: its near-line share is 0.3597 against our 0.2797 (+0.0800, outside the gate's +-0.05; the information
  row `near_line_onset_share` of `fidelity.py --profile tracking --seeds 3`, and R003's 36.0% on the same ball), so
  `context`'s 0.5586 is further out, and **the default profile is outside the band too**. The gate is `context`'s alone,
  as pre-registered: only `context` claims to place the gaps by the play, the transplant is blind to it by design, and
  no gate was added to `tracking`. On the three whole-match figures that both profiles have, `context` is further from
  ours than v3: the share missing (0.4313 against 0.4110, where v3's 0.4189 passes), the median gap (11.88 s against our
  10.74 s, v3's 10.76 s) and the share of gaps that start within 2 m of a line (0.5586 against our 0.2797, v3's 0.3597).
  On the ball visible on dead frames it is further from the public window's figure (0.4622, against v3's 0.5684 and the
  window's 0.6637), and that is a distance from the window, not from the match (see the warning above). No profile of
  the benchmark is ours, and on the share missing, the median gap and the near-line share this one is not nearer.
- Per-seed yardstick scores were not computed for `context`, and the draws are not the same as v3's (the ball's mask is
  from another generator), so the pairing is over the matches and says nothing of the seeds.

Runtime: the runner's own line says 5.5 to 6.2 s per config for the raw truth and 5.2 to 5.7 s for the coarse one, 5.1
to 5.6 s for the paired two-config runs, with the DFL caches and the Calibration cached; the suites 265 s, 46 s, 5 s
and 9 s (above; the timings are load-dependent).
Verdict: **`context` v1 is kept in the code as a sensitivity profile and is not the benchmark's profile**: it fails the
near-line gate on every seed (0.5586 against 0.2797 +-0.05, because DFL tracks the ball out of play and the hazard that
fits ours is applied to those rows), fails the share missing pooled and on seed 2 (0.4313 against 0.4110 +-0.02, by
0.0003 pooled), and warns on the ball visible on dead frames (0.4622 against the public window's 0.6637 +-0.10; the
window is better covered than the match, so the row is a distance from the window and not from the match). No fix, no
second rescale of `HAZARD_SCALE`, no gate moved; `DEFAULT_PROFILE` stays `tracking` (version 3), so the validation of
any method is R004c's `tracking` v3, and a number from `context` is not compared with one from it. What the check says:
the order of the four yardsticks is the same under the two ways of placing the gaps (tau +1.000 between the profiles'
orders on each truth, +0.667 against the published order under each), and `w0` is last by far on both; every score is
higher on `context` (0.3 to 2.2 points of pooled spellF1), possibly because it hides the ball more on dead frames, which
the yardsticks read as dead (not measured). Do not choose between `w125` and the others where the raw and the coarse
truth disagree, on either profile.

### R006 — The `hmm` decoder's first grid on the benchmark: the two switch penalties (2026-10-25, Tian)
Proposal: P06 · Commit: `b6bcf8f` (the code and the 24 configs: the decoder is `41d9f09` and `0ee15e5`, the evidence
`82362dd`; the runner's header says `b6bcf8f45f93`, run from a detached checkout of it; this entry and Stage 2's configs
are the commit after it) · Commands: the 24 configs `pipeline/benchmark/configs/p06/p06s1_*.toml` in three shares of
eight, each share run twice, six processes at once, all with `--profile tracking --seeds 3`:
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml <share> --profile tracking --seeds 3 --truth raw`
and the same with `nearest_w125.toml` first, `nearest_w50.toml` second and `--truth coarse` (the first config of a run
is the reference of its paired comparison, so each truth compares with its strongest yardstick)
Public: not applicable. No BAR-ATH frame is decoded or scored; the profile reads our match's Calibration (R004c,
version 2), as every `tracking` run does.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4 graded
minutes per seed. The chosen config, `p06s1_t016_d064` (switch_team 16, switch_dead 64): pooled spellF1 0.1952 raw (mean
per match 0.1958 [0.1857, 0.2059]) and 0.1700 coarse (0.1701 [0.1654, 0.1748]). Against `nearest_w50` under the raw
truth: mean per-match Δ -0.0065 [-0.0124, -0.0006] (pooled -0.0066), sign-flip p 0.1094, 2 wins and 5 losses. Against
`nearest_w125` under the coarse truth: mean per-match Δ -0.0528 [-0.0678, -0.0377] (pooled -0.0527), p 0.0156, 0 wins
and 7 losses. The best of the grid: 0.1959 raw (`p06s1_t016_d032`, mean per-match Δ against `nearest_w50` -0.0057
[-0.0121, +0.0007], p 0.1719, 3 wins and 4 losses) and 0.2228 coarse (`p06s1_t064_d064`, mean per-match Δ against
`nearest_w125` +0.0008 [-0.0090, +0.0106], p 1.0000, 2 wins and 5 losses).
**No Stage-1 config clears the bar.**
Changed vs the previous best: the benchmark's best so far are the four yardsticks (R004c): `nearest_w50` 0.2018 raw and
`nearest_w125` 0.2226 coarse. This is the first entry with the `hmm` decoder, a Viterbi per half over `H`, `A` and `D`
with a penalty in nats for every change of team (`switch_team`) and every change into or out of `D` (`switch_dead`), and
with two evidence sources: `nearest_player` at `confidence = 0.78` and `ball_presence` (margin 0, missing 0.2, seen
0.585). Both are fixed by rule from A01 and were not tuned; both minimum run lengths are 1 here. Only the two penalties
vary.

Reports: `data/runs/benchmark/tracking/p06s1_t<TTT>_d<DDD>.json` and `.coarse.json` (not committed; each carries the
Calibration's digest). The six runs' stdout is the record of the numbers below, and every figure here is on a log line
or follows from two of them. The reference rows are the same in the three logs of each truth, to the last printed digit
(`seconds` aside), and are R004c's: `nearest_w50` 0.2018 and `nearest_w125` 0.1665 raw, 0.1842 and 0.2226 coarse.

**The grid.** switch_dead d in {4, 8, 16, 32, 64, 128} crossed with switch_team t in {d/4, d/2, d, 2d}: 24 configs, all
integers (above t = 2d + 1.3 the decoder never switches team directly, so larger t would repeat these). The names are
`p06s1_t<TTT>_d<DDD>`. The table has every config under both truths. Possessions per minute, `D` share and alive/dead F1
do not depend on the truth (the same predictions; the `D` truth is the same), so each has one column. Bold marks the
best of the 24 in a column of spellF1, and the chosen row. The true `D` share is 44.8% of the graded frames, and the
true possessions per graded minute are 4.58 under the raw truth and 2.48 under the coarse one.

| config | switch_team t | switch_dead d | raw spellF1 | raw macroF1 | coarse spellF1 | coarse macroF1 | alive/dead F1 | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (yardstick) | | | 0.2018 | 0.4592 | 0.1842 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` (yardstick) | | | 0.1665 | 0.4407 | 0.2226 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| `p06s1_t001_d004` | 1 | 4 | 0.0812 | 0.4406 | 0.0366 | 0.4390 | 0.5761 | 13.26 | 41.9% |
| `p06s1_t002_d004` | 2 | 4 | 0.0873 | 0.4409 | 0.0408 | 0.4394 | 0.5761 | 12.14 | 41.9% |
| `p06s1_t004_d004` | 4 | 4 | 0.1027 | 0.4421 | 0.0510 | 0.4405 | 0.5758 | 10.38 | 41.9% |
| `p06s1_t008_d004` | 8 | 4 | 0.1455 | 0.4473 | 0.0906 | 0.4457 | 0.5751 | 7.20 | 42.0% |
| `p06s1_t002_d008` | 2 | 8 | 0.0874 | 0.4409 | 0.0409 | 0.4393 | 0.5760 | 12.12 | 41.9% |
| `p06s1_t004_d008` | 4 | 8 | 0.1028 | 0.4420 | 0.0511 | 0.4405 | 0.5758 | 10.37 | 41.9% |
| `p06s1_t008_d008` | 8 | 8 | 0.1456 | 0.4473 | 0.0908 | 0.4456 | 0.5750 | 7.18 | 42.0% |
| `p06s1_t016_d008` | 16 | 8 | 0.1938 | 0.4568 | 0.1625 | 0.4561 | 0.5733 | 4.18 | 42.2% |
| `p06s1_t004_d016` | 4 | 16 | 0.1035 | 0.4418 | 0.0515 | 0.4403 | 0.5750 | 10.28 | 42.0% |
| `p06s1_t008_d016` | 8 | 16 | 0.1467 | 0.4471 | 0.0915 | 0.4454 | 0.5743 | 7.09 | 42.1% |
| `p06s1_t016_d016` | 16 | 16 | 0.1955 | 0.4566 | 0.1646 | 0.4559 | 0.5725 | 4.10 | 42.3% |
| `p06s1_t032_d016` | 32 | 16 | 0.1879 | 0.4603 | 0.2074 | 0.4631 | 0.5698 | 2.89 | 42.6% |
| `p06s1_t008_d032` | 8 | 32 | 0.1471 | 0.4468 | 0.0921 | 0.4451 | 0.5725 | 6.98 | 42.5% |
| `p06s1_t016_d032` | 16 | 32 | **0.1959** | 0.4561 | 0.1665 | 0.4555 | 0.5703 | 3.98 | 42.7% |
| `p06s1_t032_d032` | 32 | 32 | 0.1881 | 0.4596 | 0.2112 | 0.4626 | 0.5666 | 2.76 | 43.1% |
| `p06s1_t064_d032` | 64 | 32 | 0.1518 | 0.4576 | 0.2200 | 0.4663 | 0.5600 | 1.97 | 44.0% |
| **`p06s1_t016_d064` (chosen)** | 16 | 64 | 0.1952 | 0.4525 | 0.1700 | 0.4520 | 0.5581 | 3.64 | 44.9% |
| `p06s1_t032_d064` | 32 | 64 | 0.1854 | 0.4554 | 0.2147 | 0.4583 | 0.5537 | 2.48 | 45.4% |
| `p06s1_t064_d064` | 64 | 64 | 0.1470 | 0.4534 | **0.2228** | 0.4618 | 0.5462 | 1.70 | 46.5% |
| `p06s1_t128_d064` | 128 | 64 | 0.1044 | 0.4434 | 0.1785 | 0.4523 | 0.5351 | 1.23 | 47.9% |
| `p06s1_t032_d128` | 32 | 128 | 0.1789 | 0.4524 | 0.2150 | 0.4554 | 0.5389 | 2.10 | 48.4% |
| `p06s1_t064_d128` | 64 | 128 | 0.1371 | 0.4488 | 0.2174 | 0.4569 | 0.5289 | 1.39 | 49.7% |
| `p06s1_t128_d128` | 128 | 128 | 0.0913 | 0.4367 | 0.1645 | 0.4451 | 0.5121 | 0.94 | 51.9% |
| `p06s1_t256_d128` | 256 | 128 | 0.0617 | 0.4212 | 0.1157 | 0.4280 | 0.4916 | 0.68 | 54.8% |
| *the truth* | | | | | | | | 4.58 raw, 2.48 coarse | 44.8% |

**The choice**, by the objective and the tie rule of the design (pooled spellF1 under the raw truth; configs within
0.005 of the best are tied; among the tied the best coarse score; then the larger switch_team, min_possession,
switch_dead and min_dead):
- The best raw score is 0.1959 (`p06s1_t016_d032`), so the band is 0.1909 and above. Four configs are in it, all with
  t = 16: d = 8 (0.1938), 16 (0.1955), 32 (0.1959) and 64 (0.1952). The next is `p06s1_t032_d032` at 0.1881, 0.0028
  below the band.
- Their coarse scores are 0.1625 (d = 8), 0.1646 (d = 16), 0.1665 (d = 32) and 0.1700 (d = 64). The best is d = 64, and
  it decides alone, so no further tie-break is needed. **(t*, d*) = (16, 64).**
- The edge rule (add d = 2 or d = 256 if the chosen d is 4 or 128) does not fire: d* = 64.
- The raw-only winner is `p06s1_t016_d032` (0.1959) and the coarse-only winner is `p06s1_t064_d064` (0.2228). The
  chosen config is the raw third (0.1952, 0.0007 below the best) and the coarse ninth of 24 (0.1700, against the
  winner's 0.2228); the coarse winner is the raw eleventh (0.1470).
- The plateau: 4 of the 24 sit within 0.005 of the best raw score (all at t = 16, d from 8 to 64) and 2 within 0.005 of
  the best coarse one (t = 64, d = 32 and 64). A third coarse config, `p06s1_t064_d128` at 0.2174, is 0.0004 outside
  the band.
- **The two truths disagree on t**, and agree that neither small t nor very large t works. Raw ends at t = 16 and coarse
  at t = 64, and each truth's best sits at fewer possessions per minute than that truth's true rate (raw 3.64 to 4.18
  against 4.58, coarse 1.70 to 1.97 against 2.48). The objective is the raw one and Stage 2 follows it; the verdict
  says so (T33).

**Against the yardsticks.** The top three of the 24 under each truth, plus a fourth row: the coarse winner in the raw
table and the chosen config in the coarse one. The first Δ is the runner's, against that truth's first config
(`nearest_w50` raw, `nearest_w125` coarse; pooled spellF1 in the second column, Δ per match from the logs). The second
Δ is against the other yardstick, **recomputed here from the 4-decimal per-match scores of the logs** with the
benchmark's own `t_interval` and `sign_flip_p` (the runner printed no comparison for that pair). A pooled Δ against
the second yardstick is the difference of two 4-decimal figures, the config's pooled Δ and the second yardstick's, both
against the first config and as the runner printed them, so it is good to ±0.0001 and can differ in the last digit from
the difference of the two rounded pooled scores (-0.0194 here against -0.0195 for `p06s1_t064_d064` in the raw table;
-0.0143 against -0.0142 for `p06s1_t016_d064` in the coarse one). On the first-yardstick rows the same recomputation
reproduces the runner's mean and interval to 0.0001, and its p to within a step of the rounding (0.1406 against 0.1250
for one config). All the second-yardstick p are 0.0156 because the 7
matches go the same way, and that p does not depend on the rounding.

Raw truth (`nearest_w50` 0.2018, `nearest_w125` 0.1665):

| config | pooled spellF1 | Δ against `nearest_w50`, mean per match [90% t] | sign-flip p | wins / losses | Δ against `nearest_w125`, pooled | Δ mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `p06s1_t016_d032` | 0.1959 | -0.0057 [-0.0121, +0.0007] | 0.1719 | 3 / 4 | +0.0294 | +0.0290 [+0.0195, +0.0385] | 0.0156 | 7 / 0 |
| `p06s1_t016_d016` | 0.1955 | -0.0062 [-0.0127, +0.0002] | 0.1250 | 2 / 5 | +0.0290 | +0.0285 [+0.0185, +0.0384] | 0.0156 | 7 / 0 |
| `p06s1_t016_d064` | 0.1952 | -0.0065 [-0.0124, -0.0006] | 0.1094 | 2 / 5 | +0.0287 | +0.0282 [+0.0178, +0.0387] | 0.0156 | 7 / 0 |
| `p06s1_t064_d064` | 0.1470 | -0.0537 [-0.0634, -0.0440] | 0.0156 | 0 / 7 | -0.0194 | -0.0190 [-0.0248, -0.0131] | 0.0156 | 0 / 7 |

Coarse truth (`nearest_w125` 0.2226, `nearest_w50` 0.1842):

| config | pooled spellF1 | Δ against `nearest_w125`, mean per match [90% t] | sign-flip p | wins / losses | Δ against `nearest_w50`, pooled | Δ mean per match [90% t] | sign-flip p | wins / losses |
|---|---|---|---|---|---|---|---|---|
| `p06s1_t064_d064` | 0.2228 | +0.0008 [-0.0090, +0.0106] | 1.0000 | 2 / 5 | +0.0386 | +0.0391 [+0.0256, +0.0525] | 0.0156 | 7 / 0 |
| `p06s1_t064_d032` | 0.2200 | -0.0020 [-0.0100, +0.0060] | 0.6250 | 3 / 4 | +0.0358 | +0.0362 [+0.0230, +0.0494] | 0.0156 | 7 / 0 |
| `p06s1_t064_d128` | 0.2174 | -0.0046 [-0.0137, +0.0046] | 0.4062 | 3 / 4 | +0.0332 | +0.0337 [+0.0194, +0.0479] | 0.0156 | 7 / 0 |
| `p06s1_t016_d064` | 0.1700 | -0.0528 [-0.0678, -0.0377] | 0.0156 | 0 / 7 | -0.0143 | -0.0145 [-0.0215, -0.0075] | 0.0156 | 0 / 7 |

**What the tables say**:
- **At minimum runs of 1, no Stage-1 config beats the yardstick of its truth.** Raw: the best three are 0.0059 to
  0.0066 below `nearest_w50` pooled (p 0.1094 to 0.1719; the interval of the chosen config excludes 0 by 0.0006 while
  its p is 0.1094, and the other two reach +0.0007 and +0.0002; 2 to 3 wins in 7). Coarse: the best,
  `p06s1_t064_d064`, is 0.0002 above `nearest_w125` pooled (p 1.0000, 2 wins in 7) and the next two are 0.0026 and
  0.0052 below `nearest_w125`. The bar (Δ above 0 with p of 0.05 or less, against `nearest_w50` raw and
  `nearest_w125` coarse) is not met under either truth. It is recorded and was not used to choose.
- **Every top config beats the other yardstick in all 7 matches**: the raw top three beat `nearest_w125` by +0.0287 to
  +0.0294 pooled, and the coarse top three beat `nearest_w50` by +0.0332 to +0.0386. Against the first yardstick of
  each truth they are below it under the raw truth (-0.0059 to -0.0066) and within 0.0052 of it under the coarse one
  (+0.0002 to -0.0052).
- **The score is a ridge in t, in both truths, and d barely moves it where t is small.** Raw by t (all d, pooled
  spellF1): t = 1 to 4 0.0812 to 0.1035, t = 8 0.1455 to 0.1471, t = 16 0.1938 to 0.1959, t = 32 0.1789 to 0.1881,
  t = 64 0.1371 to 0.1518, t = 128 0.0913 and 0.1044, t = 256 0.0617. Coarse: t = 1 to 4 0.0366 to 0.0515, t = 8 0.0906
  to 0.0921, t = 16 0.1625 to 0.1700, t = 32 0.2074 to 0.2150, t = 64 0.2174 to 0.2228, t = 128 0.1645 and 0.1785,
  t = 256 0.1157. Possessions per minute fall with t, from 13.26 (t = 1, d = 4) to 0.68 (t = 256), and the true 4.58
  and 2.48 fall inside that range. At t = 16 the four d values span 0.0021 raw, at t = 8 0.0016, and the span grows
  with t up to t = 64 (0.0092 at t = 32, d = 16 to 128; 0.0147 at t = 64, d = 32 to 128), then falls (0.0131 at t = 128,
  d = 64 and 128). One degradation seed moves a pooled score by up to about 0.015 (R003, and 0.013 in R004c), so the
  steps inside the band are not ordered.
- **The `D` the HMM places is worse than the yardsticks', although its share of `D` is near the truth's.**
  Alive/dead F1 is 0.4916 to 0.5761 for all 24 configs, against 0.6094 for `nearest_w50` and 0.6428 for `nearest_w125`;
  the predicted `D` share is 41.9% to 54.8% against the true 44.8%, and the yardsticks' 33.4% and 23.3%. Within a t,
  a larger d lowers alive/dead F1 and raises the `D` share (t = 16: 0.5733 to 0.5581 and 42.2% to 44.9% from d = 8 to
  64). The three best raw configs have 3.64 to 4.10 possessions per minute, near `nearest_w50`'s 4.30 (the true value
  is 4.58), and they differ from it in alive/dead F1 (0.5581 to 0.5725 against 0.6094) and in the share of `D`
  (42.3% to 44.9% against 33.4%). The tables do not separate how much of the raw gap is `D` and how much is the team
  labelling. `tracking` v3 transplants our gaps without regard to the play (R003, R004c), which is the design's own
  caveat on valuing `ball_presence` as evidence for `D` here; `context` (R005, sensitivity only) places them by the
  play.
- **A small t fragments**: t = 1 to 4 predicts 10.28 to 13.26 possessions per minute against the true 4.58, and scores
  0.0812 to 0.1035 raw. Large t merges: t = 128 predicts 1.23 and 0.94, and t = 256 0.68.
- **macroF1, the tiebreaker**: the best HMM has 0.4603 raw (`p06s1_t032_d016`) against `nearest_w50`'s 0.4592, and
  0.4663 coarse (`p06s1_t064_d032`) against `nearest_w125`'s 0.4509 and `nearest_w50`'s 0.4602. The chosen config has
  0.4525 raw and 0.4520 coarse.
- **Selection, leave one match out (Stage 1 only).** For each match, take the config with the best mean per-match
  spellF1 raw on the other six (the runner's per-match scores stand in for the pooled objective, and the tie rule is not
  applied). It is `p06s1_t016_d032` in all seven folds, so the held-out estimate is that config's own: mean 0.1965
  against `nearest_w50`'s 0.2023 on the same matches, Δ -0.0057 [-0.0121, +0.0007], p 0.1719, 3 wins in 7. The pick
  does not change from fold to fold; the estimate for the final choice is over Stage 1 and Stage 2 (R007).

Runtime: the runner's `seconds` line says 16.7 to 24.5 s per HMM config (17.0 to 24.5 raw, 16.7 to 24.2 coarse) with
the six processes running side by side, against 10.9 to 13.0 s for the two yardsticks in the same runs.
Verdict: **(t*, d*) = (16, 64) for Stage 2**, by the objective and the tie rule above; the edge rule did not fire. Stage
1 as it stands does not clear the bar under either truth, and its `D` is worse than the yardsticks' (alive/dead F1
0.5581 for the chosen config against 0.6094 and 0.6428); the two truths disagree on t (raw 16, coarse 64), and the raw
objective decides. Stage 2 (R007) adds the minimum run lengths: `p06s2_t<TTT>_d<DDD>_mp<MMM>_md<NNN>.toml`, t in {8, 16,
32} x d in {32, 64, 128} (all nine pairs have t <= 2d) x min_possession in {1, 6, 12, 25, 50} x min_dead in {1, 25}: 90
configs, each of which `load_config` accepts. Six of the nine pairs are in Stage 1 (t = 8, 16 and 32 with d = 32, t = 16
and 32 with d = 64, and t = 32 with d = 128), so, as long as `hmm.py`'s labels and the evidence code are unchanged (the
posterior (`952b392`, merged at `8160627`) changes no label; its checks pin that), their minimums-1 configs must
score exactly as here: a free consistency check. **Two things Stage 2 does not test, by the protocol**: the coarse
winner's t = 64, which is outside {8, 16, 32}, and d below 32, where the raw score at t = 16 is as high (0.1938 at d =
8, 0.1955 at d = 16) and the coarse one lower. After Stage 2 come the controls the protocol names (`seen` = 0.5; d
scaled on `context`) and a diagnostic of where the HMM's `D` comes from (`ball_presence` margin 50: `D` only inside the
gaps' interiors, as `majority_vote`'s fallback places it), none of them a candidate.

### R007 — The `hmm` decoder's second grid: the minimum run lengths, and the choice of its parameters (2026-10-25, Tian)
Proposal: P06 · Commit: `5fafe0d` (Stage 2: the 90 configs, run from a detached checkout; the runner's header says
`5fafe0dbc175`), `9bbc3e3` (Stage 2b: the 15 configs of the edge extension, run from a detached checkout of it; header
`9bbc3e39d83d`) and `bdd699b` (the chosen config `pipeline/configs/p06_hmm.toml` and the four control configs; the
controls below were run at it, header `bdd699b082e6`; the entry itself, minus the controls and the final verdict, is the
commit after it, and those are the commit after that) · Commands: Stage 2 as in R006, the 90 configs
`pipeline/benchmark/configs/p06/p06s2_*.toml` in three shares of 30, each share run twice, six processes at once, all
with `--profile tracking --seeds 3`:
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml <share> --profile tracking --seeds 3 --truth raw`
and the same with `nearest_w125.toml` first, `nearest_w50.toml` second and `--truth coarse`. Stage 2b: the 15
`p06s2b_*.toml` sorted, one process per truth, the same flags and the same references first (the coarse process was run
again alone after a launch mistake, at the same commit and with the same command). Public window:
`python pipeline/run.py pipeline/configs/p06_hmm.toml --score`, twice, then `python evaluation/run_evaluation.py --no-detail`
Public (a correctness check only, T01): macroF1 0.6393 · spellF1 0.3500 · alive/dead F1 0.8895 · 23 possessions
against the truth's 17, mean 11.4 s against its 14.2 s · transF1 0.3043 · lat_f -10.1. The file is valid (199,319
rows) and both runs wrote the same bytes, sha256 `9705a88583d165fb67dca2a917c170e83dbf38749ec8a15af64a92c8816ba8fb`.
Over all the graded frames of the match it labels H 33.7%, A 24.0% and D 42.3%, and finds 393 possessions (3.71 per
graded minute, mean 9.3 s) against `main`'s 594 (5.61 per graded minute, R001). The run was made at `9bbc3e3` with
the file that `bdd699b` then committed, so the manifest (`data/runs/p06_hmm.json`, not committed) says `+dirty`.
`--score` reports `nearest_player` agreeing with the public GT on 77.8% of 5,640 frames (A01's figure) and
`ball_presence` on alive on 81.8% of 7,500 frames. No bootstrap was run, because no public-window gap is claimed.
The public scores are in-sample for the three evidence values A01 measured on those frames (`confidence` 0.78,
`missing` 0.2 and `seen` 0.585).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4 graded
minutes per seed. The chosen config, `p06s2b_t008_d032_mp075_md001` (switch_team 8, switch_dead 32, min_possession 75,
min_dead 1): pooled spellF1 0.2053 raw (mean per match 0.2064 [0.1937, 0.2191]) and 0.1967 coarse (0.1972 [0.1860,
0.2084]). Against `nearest_w50` under the raw truth: pooled Δ +0.0036, mean per match +0.0041 [-0.0022, +0.0104],
sign-flip p 0.2500, 5 wins and 2 losses. Against `nearest_w125` under the coarse truth: pooled Δ -0.0259, mean -0.0256
[-0.0363, -0.0150], p 0.0156, 0 wins and 7 losses. **The bar is not met under either truth**, and no config of the
three stages meets it.
Changed vs the previous best: the benchmark's best so far are the four yardsticks (R004c): `nearest_w50` 0.2018 raw and
`nearest_w125` 0.2226 coarse; R006's Stage 1 chose `p06s1_t016_d064` at 0.1952 raw and 0.1700 coarse, below both. This
entry adds the minimum run lengths to the same decoder (`min_possession` in {1, 6, 12, 25, 50} and `min_dead` in {1, 25}
in Stage 2, then `min_possession` in {50, 75, 100, 150} at t in {4, 8} and d in {16, 32} in Stage 2b) and chooses over
all three stages by the pre-declared rule (over the set Stage 2b's ruling extended: the choice below says what the rule
alone would have picked). The chosen config scores +0.0101 raw and +0.0267 coarse above R006's choice, and +0.0036 raw
above `nearest_w50`, but -0.0259 coarse below `nearest_w125`. The evidence sources are as in R006 and are not tuned:
`nearest_player` at `confidence = 0.78` and `ball_presence` (margin 0, missing 0.2, seen 0.585).

Reports: `data/runs/benchmark/tracking/p06s2_*.json`, `p06s2b_*.json` and the `.coarse.json` of each (not committed;
each carries the Calibration's digest). The eight runs' stdout (six of Stage 2, two of Stage 2b) is the record of the
numbers below, and every figure here is on a log line, follows from two of them, or is a computation said so where it is
made. **Checks on the runs.** All eight logs end `EXIT=0`. The references (`nearest_w50` and `nearest_w125`) are the
same in the fourteen logs of R006, Stage 2 and Stage 2b (seven per truth), to the last printed digit (`seconds` aside),
and are R004c's. The six Stage-2 points with min_possession = min_dead = 1 that Stage 1 also ran score exactly as in
R006 (every printed field, both truths, 0 mismatches), which is the consistency check R006 set. In all 45 pairs of Stage
2, the config with min_dead = 25 prints the same numbers as its twin with min_dead = 1, under both truths. Stage 2b ran
at `9bbc3e3`, which holds the posterior (`952b392`, merged at `8160627`, after Stage 2's commit); that change does not
touch a label and its checks pin that, but no config was run at both commits, so these logs cannot confirm it: the
references are the only repeat. The check was made afterwards, in the controls below.

**The grid.** Stage 2 crossed (t, d) with t in {8, 16, 32} and d in {32, 64, 128} (nine pairs, all with t <= 2d) with
`min_possession` in {1, 6, 12, 25, 50} and `min_dead` in {1, 25}: 90 configs, `p06s2_t<TTT>_d<DDD>_mp<MMM>_md<NNN>`.
**Stage 2b, with its ruling.** The best raw config of Stage 2, `p06s2_t008_d032_mp050_md*` (0.2043 raw, 0.1760 coarse),
sat on three edges of Stage 2's grid: the lowest t (8), the lowest d (32) and the highest `min_possession` (50). By a
ruling made after Stage 2's results and before Stage 2b's (the analogue of Stage 1's edge rule, capped at one
extension), 15 more configs were written, `p06s2b_t<TTT>_d<DDD>_mp<MMM>_md001`: t in {4, 8} x d in {16, 32} x
`min_possession` in {50, 75, 100, 150}, `min_dead` 1 (Stage 2 had just shown it moves nothing), minus (8, 32, 50), which
Stage 2 had run. The final choice is over Stage 1 (24 configs), Stage 2 (90) and Stage 2b (15): 123 distinct configs
once the six that Stage 1 and Stage 2 share are counted once, and 78 score profiles without the min_dead = 25 twins. The
table has every Stage-2 and Stage-2b config with `min_dead` 1, and Stage 1's at these (t, d), under both truths, as raw
/ coarse pooled spellF1 (62 of the 78 profiles; the other 16 Stage-1 configs are in R006's table); a blank cell was not
run. The `min_possession` = 1 cells are Stage 1's or Stage 2's, 6, 12, 25 and 50 are Stage 2's for its nine pairs, 50 is
also Stage 2b's for t in {4, 8} with d in {16, 32} (8 / 32 is Stage 2's), and 75, 100 and 150 are Stage 2b's. The bold
cell is the chosen config.

| t / d | mp 1 | mp 6 | mp 12 | mp 25 | mp 50 | mp 75 | mp 100 | mp 150 |
|---|---|---|---|---|---|---|---|---|
| 4 / 16 | 0.1035 / 0.0515 |  |  |  | 0.2018 / 0.1640 | 0.2083 / 0.1936 | 0.2003 / 0.2083 | 0.1805 / 0.2281 |
| 4 / 32 |  |  |  |  | 0.2025 / 0.1654 | 0.2080 / 0.1944 | 0.1992 / 0.2085 | 0.1801 / 0.2307 |
| 8 / 16 | 0.1467 / 0.0915 |  |  |  | 0.2037 / 0.1743 | 0.2056 / 0.1960 | 0.2000 / 0.2111 | 0.1789 / 0.2290 |
| 8 / 32 | 0.1471 / 0.0921 | 0.1471 / 0.0921 | 0.1473 / 0.0923 | 0.1817 / 0.1308 | 0.2043 / 0.1760 | **0.2053 / 0.1967** | 0.1988 / 0.2113 | 0.1782 / 0.2310 |
| 8 / 64 | 0.1471 / 0.0936 | 0.1471 / 0.0936 | 0.1472 / 0.0937 | 0.1810 / 0.1330 | 0.2032 / 0.1795 |  |  |  |
| 8 / 128 | 0.1467 / 0.0963 | 0.1467 / 0.0963 | 0.1469 / 0.0964 | 0.1797 / 0.1366 | 0.1990 / 0.1829 |  |  |  |
| 16 / 32 | 0.1959 / 0.1665 | 0.1959 / 0.1665 | 0.1959 / 0.1665 | 0.1965 / 0.1671 | 0.2021 / 0.1897 |  |  |  |
| 16 / 64 | 0.1952 / 0.1700 | 0.1952 / 0.1700 | 0.1952 / 0.1700 | 0.1957 / 0.1705 | 0.2005 / 0.1936 |  |  |  |
| 16 / 128 | 0.1918 / 0.1739 | 0.1918 / 0.1739 | 0.1918 / 0.1739 | 0.1922 / 0.1743 | 0.1956 / 0.1971 |  |  |  |
| 32 / 32 | 0.1881 / 0.2112 | 0.1881 / 0.2112 | 0.1881 / 0.2112 | 0.1881 / 0.2112 | 0.1883 / 0.2119 |  |  |  |
| 32 / 64 | 0.1854 / 0.2147 | 0.1854 / 0.2147 | 0.1854 / 0.2147 | 0.1854 / 0.2147 | 0.1855 / 0.2154 |  |  |  |
| 32 / 128 | 0.1789 / 0.2150 | 0.1789 / 0.2150 | 0.1789 / 0.2150 | 0.1789 / 0.2150 | 0.1791 / 0.2155 |  |  |  |

The second table has the references, the configs within 0.005 of the best raw score, the coarse best and R006's choice,
with the columns of R006's table plus `min_possession` (frames; 25 frames are 1 s). Possessions per minute, `D` share
and alive/dead F1 do not depend on the truth (the same predictions), so each has one column. The true `D` share is 44.8%
of the graded frames, and the true possessions per graded minute are 4.58 under the raw truth and 2.48 under the coarse
one.

| config | t | d | mp | raw spellF1 | raw macroF1 | coarse spellF1 | coarse macroF1 | alive/dead F1 | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| `nearest_w50` (yardstick) | | | | 0.2018 | 0.4592 | 0.1842 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` (yardstick) | | | | 0.1665 | 0.4407 | 0.2226 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| **`p06s2b_t008_d032_mp075_md001` (chosen)** | 8 | 32 | 75 | 0.2053 | 0.4569 | 0.1967 | 0.4557 | 0.5676 | 3.10 | 43.1% |
| `p06s2b_t004_d016_mp075_md001` (raw best) | 4 | 16 | 75 | 0.2083 | 0.4566 | 0.1936 | 0.4549 | 0.5699 | 3.26 | 42.8% |
| `p06s2b_t004_d032_mp075_md001` | 4 | 32 | 75 | 0.2080 | 0.4562 | 0.1944 | 0.4546 | 0.5680 | 3.20 | 43.1% |
| `p06s2b_t008_d016_mp075_md001` | 8 | 16 | 75 | 0.2056 | 0.4573 | 0.1960 | 0.4561 | 0.5697 | 3.16 | 42.8% |
| `p06s2_t008_d032_mp050_md001` | 8 | 32 | 50 | 0.2043 | 0.4549 | 0.1760 | 0.4537 | 0.5696 | 3.70 | 42.8% |
| `p06s2b_t008_d016_mp050_md001` | 8 | 16 | 50 | 0.2037 | 0.4554 | 0.1743 | 0.4541 | 0.5715 | 3.78 | 42.5% |
| `p06s2b_t008_d032_mp150_md001` (coarse best) | 8 | 32 | 150 | 0.1782 | 0.4559 | 0.2310 | 0.4600 | 0.5590 | 2.16 | 44.7% |
| `p06s1_t016_d064` (R006's choice) | 16 | 64 | 1 | 0.1952 | 0.4525 | 0.1700 | 0.4520 | 0.5581 | 3.64 | 44.9% |
| *the truth* | | | | | | | | | 4.58 raw, 2.48 coarse | 44.8% |

**The choice**, by the objective and the tie rule of the design (pooled spellF1 under the raw truth; configs within
0.005 of the best are tied; among the tied the best coarse score; then the larger switch_team, min_possession,
switch_dead and min_dead), over Stage 1 ∪ Stage 2 ∪ Stage 2b:
- The best raw score is 0.2083 (`p06s2b_t004_d016_mp075_md001`), so the band is 0.2033 and above. Six configs are in it
  (seven with the min_dead = 25 twin of `p06s2_t008_d032_mp050`), the four with min_possession 75 and the two with 50 in
  the second table. The next, `p06s2_t008_d064_mp050_md001`, is at 0.2032, 0.0051 below the best by the printed digits
  and so just outside; its coarse score is 0.1795, below the chosen config's, so counting it in would not change the
  choice.
- Their coarse scores are 0.1936 (4 / 16 / 75), 0.1944 (4 / 32 / 75), 0.1960 (8 / 16 / 75), 0.1967 (8 / 32 / 75), 0.1760
  (8 / 32 / 50) and 0.1743 (8 / 16 / 50). The best is 8 / 32 / 75 and it decides alone, 0.0007 ahead of 8 / 16 / 75, so
  no further tie-break is needed. **(switch_team, switch_dead, min_possession, min_dead) = (8, 32, 75, 1).**
- The raw-only winner is `p06s2b_t004_d016_mp075_md001` (0.2083) and the coarse-only winner is
  `p06s2b_t008_d032_mp150_md001` (0.2310). The chosen config is the raw fourth and the coarse 29th of the 78 score
  profiles (0.0030 below the raw best; 0.0343 below the coarse best).
- **What the declared rule alone would have chosen.** The protocol declared the choice over Stage 1 ∪ Stage 2; Stage 2b
  was a ruling made after Stage 2's results and before its own, capped at one extension, and the chosen config comes
  from it. Over Stage 1 ∪ Stage 2 the same rule (best raw 0.2043, a band of 0.1993 and above, four score profiles) picks
  `p06s2_t016_d064_mp050_md025` (its `min_dead` 1 twin scores the same), at 0.2005 raw and 0.1936 coarse: -0.0012 raw
  against `nearest_w50` (p 0.8281, 3 wins and 4 losses) and -0.0290 coarse against `nearest_w125` (p 0.0156, 0 wins and
  7 losses), the runner's own lines in the Stage-2 logs. So the ruling moved the choice, and the sign of the raw Δ
  (-0.0012 to +0.0036); both are level with `nearest_w50`, so the verdict does not change.
- **The two truths disagree on the minimum run, as they did on t (R006).** Raw peaks at 75 frames in all four Stage-2b
  pairs and falls after it (0.1988 to 0.2003 at 100 and 0.1782 to 0.1805 at 150); coarse rises through the whole grid
  (0.1936 to 0.1967 at 75, 0.2083 to 0.2113 at 100, 0.2281 to 0.2310 at 150). The raw band and the coarse band (four
  configs within 0.005 of 0.2310, all at min_possession 150) are disjoint: the coarse four score 0.1782 to 0.1805 raw,
  and the raw six 0.1743 to 0.1967 coarse. The raw objective decides, and the verdict says so (T33).
- **The winner is on an edge of Stage 2b's grid**, and by the ruling nothing is extended further. In t and d: the chosen
  config is at the upper corner of the (t, d) range of the 75-frame slice (t 8 of {4, 8}, d 32 of {16, 32}), and the raw
  best at its lower one (t 4, d 16); only `min_possession` is interior for the raw objective (75 of {50, 75, 100, 150}),
  while the coarse best is on its upper edge (150). What the grid shows about the edges: at 75 frames the four (t, d)
  points span 0.0030 raw (0.2053 to 0.2083) and 0.0031 coarse (0.1936 to 0.1967), with a lower t 0.0027 better raw at
  either d and d moving nothing beyond 0.0003; at 50 frames, where the grid is wider, raw falls with t above 8 (d = 32:
  0.2043 at t = 8, 0.2021 at t = 16, 0.1883 at t = 32) and with d above 32 (t = 8: 0.2043, 0.2032, 0.1990 at d = 32, 64,
  128) while coarse rises with both (d = 32: 0.1760, 0.1897, 0.2119; t = 8: 0.1760, 0.1795, 0.1829). Below t = 8 the
  order depends on the minimum: t = 4 is 0.0019 and 0.0018 below t = 8 at 50 frames (d = 16 and 32) and 0.0027 above it
  at 75, so the raw best's place on the lower edge does not show a slope towards smaller t. If 75 frames behave as 50
  do, a step beyond the upper corner would trade raw for coarse; it was not run at 75.
- The plateau: 6 of the 78 score profiles sit within 0.005 of the best raw score (the four at 75 frames and two at 50),
  and 4 within 0.005 of the best coarse one. One degradation seed moves a pooled score by up to about 0.015 (R003, and
  0.013 in R004c), so the order inside the raw band is not a ranking, and which of its members is chosen was decided by
  the coarse score with 0.0007 to spare.

**Against the yardsticks.** The chosen config, the raw and the coarse winners, Stage 2's best (the config that
prompted Stage 2b) and R006's choice, each under both truths against both yardsticks. In the tables of the first
yardstick of each truth (`nearest_w50` raw, `nearest_w125` coarse) the Δ columns are the runner's own lines for that
config: the pooled Δ, the mean per-match Δ with its 90% t-interval, the sign-flip p, wins and losses, and the `by
competition` line. In the tables of the second yardstick they are **recomputed here from the 4-decimal per-match scores
of the logs** with the benchmark's own `t_interval` and `sign_flip_p` (the runner prints no comparison for that pair);
the pooled Δ is the difference of the runner's pooled Δ of the config and of the second yardstick against the first,
good to ±0.0001 (R006 explains the rounding). On the ten first-yardstick rows the same recomputation reproduces the
runner's mean and interval ends to 0.0001, and its p, wins and losses exactly. A p of 0.0156 is the smallest that 7
matches allow. The competition columns are the mean per-match Δ of the 2 matches of DFL-COM-000001 and of the 5 of
DFL-COM-000002.

Raw truth (`nearest_w50` 0.2018, `nearest_w125` 0.1665):

Against `nearest_w50` (the first yardstick; pooled spellF1 0.2018):

| config | pooled spellF1 | pooled Δ | mean per-match Δ [90% t] | sign-flip p | wins / losses | by competition (2 matches / 5 matches) |
|---|---|---|---|---|---|---|
| `p06s2b_t008_d032_mp075_md001` (chosen) | 0.2053 | +0.0036 | +0.0041 [-0.0022, +0.0104] | 0.2500 | 5 / 2 | -0.0036 / +0.0073 |
| `p06s2b_t004_d016_mp075_md001` (raw best) | 0.2083 | +0.0065 | +0.0071 [+0.0008, +0.0134] | 0.0781 | 6 / 1 | -0.0019 / +0.0107 |
| `p06s2b_t008_d032_mp150_md001` (coarse best) | 0.1782 | -0.0236 | -0.0229 [-0.0320, -0.0138] | 0.0156 | 0 / 7 | -0.0357 / -0.0177 |
| `p06s2_t008_d032_mp050_md001` (Stage 2's best) | 0.2043 | +0.0026 | +0.0028 [-0.0028, +0.0084] | 0.3906 | 4 / 3 | -0.0024 / +0.0049 |
| `p06s1_t016_d064` (R006's choice) | 0.1952 | -0.0066 | -0.0065 [-0.0124, -0.0006] | 0.1094 | 2 / 5 | -0.0090 / -0.0055 |

Against `nearest_w125` (the second yardstick; pooled spellF1 0.1665):

| config | pooled spellF1 | pooled Δ | mean per-match Δ [90% t] | sign-flip p | wins / losses | by competition (2 matches / 5 matches) |
|---|---|---|---|---|---|---|
| `p06s2b_t008_d032_mp075_md001` (chosen) | 0.2053 | +0.0389 | +0.0389 [+0.0309, +0.0468] | 0.0156 | 7 / 0 | +0.0403 / +0.0383 |
| `p06s2b_t004_d016_mp075_md001` (raw best) | 0.2083 | +0.0418 | +0.0418 [+0.0336, +0.0499] | 0.0156 | 7 / 0 | +0.0420 / +0.0416 |
| `p06s2b_t008_d032_mp150_md001` (coarse best) | 0.1782 | +0.0117 | +0.0118 [+0.0048, +0.0189] | 0.0312 | 6 / 1 | +0.0083 / +0.0133 |
| `p06s2_t008_d032_mp050_md001` (Stage 2's best) | 0.2043 | +0.0379 | +0.0375 [+0.0281, +0.0470] | 0.0156 | 7 / 0 | +0.0416 / +0.0359 |
| `p06s1_t016_d064` (R006's choice) | 0.1952 | +0.0287 | +0.0282 [+0.0178, +0.0387] | 0.0156 | 7 / 0 | +0.0350 / +0.0255 |

Coarse truth (`nearest_w125` 0.2226, `nearest_w50` 0.1842):

Against `nearest_w125` (the first yardstick; pooled spellF1 0.2226):

| config | pooled spellF1 | pooled Δ | mean per-match Δ [90% t] | sign-flip p | wins / losses | by competition (2 matches / 5 matches) |
|---|---|---|---|---|---|---|
| `p06s2b_t008_d032_mp075_md001` (chosen) | 0.1967 | -0.0259 | -0.0256 [-0.0363, -0.0150] | 0.0156 | 0 / 7 | -0.0297 / -0.0240 |
| `p06s2b_t004_d016_mp075_md001` (raw best) | 0.1936 | -0.0290 | -0.0288 [-0.0390, -0.0185] | 0.0156 | 0 / 7 | -0.0341 / -0.0266 |
| `p06s2b_t008_d032_mp150_md001` (coarse best) | 0.2310 | +0.0084 | +0.0087 [-0.0006, +0.0181] | 0.1562 | 5 / 2 | -0.0002 / +0.0123 |
| `p06s2_t008_d032_mp050_md001` (Stage 2's best) | 0.1760 | -0.0466 | -0.0466 [-0.0599, -0.0334] | 0.0156 | 0 / 7 | -0.0498 / -0.0454 |
| `p06s1_t016_d064` (R006's choice) | 0.1700 | -0.0527 | -0.0528 [-0.0678, -0.0377] | 0.0156 | 0 / 7 | -0.0529 / -0.0527 |

Against `nearest_w50` (the second yardstick; pooled spellF1 0.1842):

| config | pooled spellF1 | pooled Δ | mean per-match Δ [90% t] | sign-flip p | wins / losses | by competition (2 matches / 5 matches) |
|---|---|---|---|---|---|---|
| `p06s2b_t008_d032_mp075_md001` (chosen) | 0.1967 | +0.0125 | +0.0126 [+0.0090, +0.0162] | 0.0156 | 7 / 0 | +0.0092 / +0.0140 |
| `p06s2b_t004_d016_mp075_md001` (raw best) | 0.1936 | +0.0094 | +0.0095 [+0.0063, +0.0127] | 0.0156 | 7 / 0 | +0.0048 / +0.0114 |
| `p06s2b_t008_d032_mp150_md001` (coarse best) | 0.2310 | +0.0468 | +0.0470 [+0.0377, +0.0562] | 0.0156 | 7 / 0 | +0.0387 / +0.0503 |
| `p06s2_t008_d032_mp050_md001` (Stage 2's best) | 0.1760 | -0.0082 | -0.0084 [-0.0139, -0.0029] | 0.0469 | 1 / 6 | -0.0108 / -0.0074 |
| `p06s1_t016_d064` (R006's choice) | 0.1700 | -0.0143 | -0.0145 [-0.0215, -0.0075] | 0.0156 | 0 / 7 | -0.0140 / -0.0147 |

**What the tables say**:
- **The bar is not met.** Raw: the chosen config is above `nearest_w50` by +0.0036 pooled (+0.0041 per match [-0.0022,
  +0.0104], p 0.2500, 5 wins in 7), and the raw best by +0.0065 (+0.0071 [+0.0008, +0.0134], p 0.0781, 6 wins in 7: the
  interval excludes 0 while the p does not reach 0.05). Coarse: the chosen config is 0.0259 below `nearest_w125` (0 wins
  in 7), and even the coarse best is above it by +0.0084 only (+0.0087 [-0.0006, +0.0181], p 0.1562, 5 wins in 7). Over
  the whole candidate set no config passes the raw bar or the coarse one: 10 of the 78 score profiles have a positive
  raw pooled Δ against `nearest_w50` (the best p is 0.0781, for the raw best and for 4 / 32 / 75), 5 have a positive
  coarse pooled Δ against `nearest_w125` (the four at 150 frames, p 0.1562 to 0.3125, and R006's `p06s1_t064_d064` at
  +0.0002, p 1.0000). The bar was recorded and was not used to choose.
- **Where the raw edge comes from.** The chosen config's raw gain over `nearest_w50` is in the 5 matches of
  DFL-COM-000002 (+0.0073) and not in the 2 of DFL-COM-000001 (-0.0036); per match it is +0.0024 (J03WMX), -0.0097
  (J03WN1), +0.0092 (J03WOH), +0.0109 (J03WOY), -0.0039 (J03WPY), +0.0056 (J03WQQ) and +0.0146 (J03WR9). Under the
  coarse truth the gap to `nearest_w125` is -0.0297 and -0.0240 in the two competitions, and is negative in every match.
- **A minimum run length lifts the best score above what the switch penalties reached, and it substitutes for a large
  `switch_team`.** At min_possession 1 R006's best was 0.1959 raw, and the best config with a minimum scores 0.2083. At
  t = 8 min_possession 6 and 12 move nothing (0.1467 to 0.1473 for all three d), 25 frames give 0.1797 to 0.1817 and 50
  give 0.1990 to 0.2043, +0.0523 to +0.0572 over min_possession 1. At t = 16 50 frames add +0.0038 to +0.0062, and at t
  = 32 nothing (at most +0.0002 raw, +0.0007 coarse). So with a minimum of 50 frames t = 8 (0.2043) beats t = 16
  (0.2021) and t = 32 (0.1883) at d = 32, which reorders R006's ranking at minimum 1 (t = 16 first at 0.1959, t = 32 at
  0.1881, t = 8 last at 0.1471). The chosen config is +0.0101 raw and +0.0267 coarse above R006's choice. `min_dead`
  changes nothing in Stage 2 (identical numbers in all 45 pairs); the likely reason is that no `D` run of the optimal
  paths is shorter than 25 frames at d >= 32, which the scores cannot confirm.
- **The chosen config sits between the two truths in possessions per minute**: 3.10 predicted (the four 75-frame
  configs 3.10 to 3.26, the four 150-frame ones 2.16 to 2.27) against 4.58 under the raw truth and 2.48 under the coarse
  one, and `nearest_w50`'s 4.30 and `nearest_w125`'s 3.05. Each truth's best configs sit at fewer possessions per minute
  than that truth has (the raw best at 75 frames 3.10 to 3.26 against 4.58, the coarse best 2.16 to 2.27 against 2.48),
  as in R006.
- **Its `D` is still worse than the yardsticks'**: alive/dead F1 0.5676 for the chosen config, 0.5590 to 0.5718 for the
  15 Stage-2b configs, 0.4916 to 0.5761 for all 123, against 0.6094 for `nearest_w50` and 0.6428 for `nearest_w125`; the
  predicted `D` share is 43.1% against the true 44.8% and the yardsticks' 33.4% and 23.3%. As in R006 the tables do not
  separate how much of the gap to a yardstick on alive/dead F1 is the stoppages the HMM places and how much is the team
  labelling, and `tracking` v3 transplants the gaps without regard to the play, which is the design's own caveat on
  valuing `ball_presence` here. Any gain over `nearest_w50` comes, if at all, from the possession structure; the tables
  cannot confirm it.
- **Against the other yardstick the chosen config and the raw best win in all 7 matches** (the chosen config beats
  `nearest_w125` raw by +0.0389 and `nearest_w50` coarse by +0.0125), as R006's best did; the coarse best beats
  `nearest_w125` raw in 6 of 7 (+0.0117, p 0.0312). Stage 2's best and R006's choice are below `nearest_w50` under the
  coarse truth (-0.0082, 1 win and 6 losses, p 0.0469; -0.0143, 0 wins).
- **macroF1, the tiebreaker**: the chosen config has 0.4569 raw against `nearest_w50`'s 0.4592 (pooled Δ -0.0024, mean
  per match -0.0022 [-0.0061, +0.0017], p 0.3594, 3 wins and 4 losses) and 0.4557 coarse against `nearest_w125`'s 0.4509
  (+0.0048, +0.0055 [-0.0052, +0.0162], p 0.3438, 5 wins and 2 losses). No Stage-2 or Stage-2b config beats R006's best
  macroF1 (0.4603 raw, `p06s1_t032_d016`; 0.4663 coarse, `p06s1_t064_d032`).
- **Selection, leave one match out (Stage 1 ∪ Stage 2 ∪ Stage 2b).** For each match, the config with the best raw score
  on the other six is scored on the held-out one. The logs print per-match spellF1 to 4 decimals and no counts, so the
  six are **averaged per match** (the mean of the six per-match spellF1, not a pooled score). By the objective alone
  (the best raw mean of six, no tie rule): `p06s2b_t004_d016_mp075_md001` in six folds and
  `p06s2b_t004_d032_mp075_md001` when J03WOY is held out, and the held-out mean is 0.2090 against `nearest_w50`'s 0.2023
  on the same matches, Δ +0.0067 [+0.0007, +0.0127], p 0.0781, 6 wins and 1 loss (recomputed from the 4-decimal scores,
  as above). With the whole rule applied to the six matches' means (the 0.005 band on the raw mean, then the best coarse
  mean, then the larger switch_team, min_possession, switch_dead and min_dead), **all seven folds pick the config chosen
  here**, so the held-out estimate is its own: 0.2064 against 0.2023, Δ +0.0041 [-0.0022, +0.0104], p 0.2500, 5 wins in
  7. The same two picks scored under the coarse truth give 0.1939 and 0.1972 against `nearest_w125`'s 0.2228, Δ -0.0290
  and -0.0256 (0 wins in 7, p 0.0156). Choosing among 123 configs costs nothing measurable across matches (the six-fold
  picks score 0.2090 held out, the raw best 0.2093 in sample), because the same config wins in six folds; the check
  varies the match and not the degradation draws, which the three seeds share across folds, it holds the grid fixed
  although Stage 2 was centred, and Stage 2b placed, on results from all seven matches, and it says nothing about how
  DFL's degraded tracking transfers to ours.

Runtime: the runner's `seconds` line says 17.1 to 21.1 s per HMM config in Stage 2 (17.1 to 21.1 raw, 17.9 to 21.0
coarse) with six processes side by side, against 7.8 to 9.4 s for the two yardsticks in the same runs; Stage 2b's say
8.8 to 8.9 s raw and 8.9 to 9.0 s coarse per HMM config against 3.1 to 3.3 s for the yardsticks, with two processes (the
coarse one a rerun). On our match the whole `pipeline/run.py` takes 2.4 s with the ATD cache warm, of which the decoder
0.3 s for the two graded halves.
**Controls.** None is a candidate, none was selected on a score, and none had run when the choice was made.
Each config is `pipeline/configs/p06_hmm.toml` with one thing changed (`pipeline/benchmark/configs/p06/p06ctl_*.toml`):
- `p06ctl_seen_off`: `ball_presence`'s `seen` at 0.5, which adds exactly 0 nats (log(2 x 0.5)), the design's control
  "without the `seen` term". The design expected that a visible stretch between two `D` stretches would then pay 2 x
  `switch_dead` out of a gain of about a sixth of a nat a frame, and possessions would be absorbed into the stoppages.
- `p06ctl_margin50`: `margin` 50 instead of 0, so that a missing frame speaks for `D` only when it is more than 50
  frames (2 s) from the nearest frame with a ball. A gap of 100 frames or less then carries no `D` evidence, and a
  longer one loses its first and last 50 frames: `D` only inside the gaps' interiors, where `majority_vote`'s fallback
  places it. A ruling made before the run, a diagnostic of where the HMM's `D` comes from.
- `p06ctl_d008` and `p06ctl_d128` (on `context` only): `switch_dead` 8 and 128, a quarter and four times the chosen 32.
  `switch_team` stays 8. It is not above 2 x 8 = 16, so at d = 8 a direct change of team (8 nats) is still cheaper than
  going through `D` (16 nats), and the case the protocol warned about does not arise.

Commands, from a detached checkout at `bdd699b` (the runner's header says `bdd699b082e6`) with `MTAH_DATA_DIR` at the
main checkout's `data/`, all with `--seeds 3`:
`python pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/configs/p06_hmm.toml pipeline/benchmark/configs/p06/p06ctl_seen_off.toml pipeline/benchmark/configs/p06/p06ctl_margin50.toml --profile tracking --seeds 3 --truth raw`,
the same with `nearest_w125.toml` first and `nearest_w50.toml` second under `--truth coarse`, and both again with
`--profile context` and `p06ctl_d008.toml` and `p06ctl_d128.toml` added at the end. `context` is a sensitivity profile
(R005: it fails two of its gates, the near-line share badly and the missing share narrowly), so every `context` number
below is read as that and no more. The headroom runs of the same commit are A10.

**Checks on the runs.** All four logs end `EXIT=0`. `p06_hmm` on `tracking` prints, under both truths, every field of
R007's chosen config as Stage 2b printed it at `9bbc3e3` (`p06s2b_t008_d032_mp075_md001`: pooled spellF1, macroF1 and
alive/dead F1, the per-match mean and interval, the possessions per minute, the `D` share and the seven per-match
scores), and its six lines of Δ against the first config (spellF1 and macroF1, with the competitions and the matches)
are the same text. So `p06_hmm.toml` is that config, and the run repeats exactly; between `9bbc3e3` and `bdd699b` only
the five config files were added. That part is a repeat of the same code, which in both runs holds the posterior
(`952b392`, merged at `8160627`) that Stage 2 lacked. The two yardsticks on `tracking` print every field as R006's logs
do (R004c's), under both truths. On `context` they match R005's table (pooled spellF1, macroF1 and alive/dead F1, the
per-match mean and interval, the possessions per minute, and `nearest_w125` against `nearest_w50`).

**The posterior's label check, closed.** The Stage-2b paragraph above left it open: the posterior lies between Stage 2
(`5fafe0d`) and Stage 2b, and no config had run on both sides of it. Two Stage-2 configs were rerun at `2bfbe11`, which
holds the posterior (a detached checkout; the runner's header says `2bfbe1112038`; the log ends `EXIT=0`):
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/p06/p06s2_t008_d032_mp050_md001.toml pipeline/benchmark/configs/p06/p06s2_t016_d064_mp012_md025.toml --profile tracking --seeds 3 --truth raw --no-write`.
Every printed field equals the same config's Stage-2 lines at `5fafe0d` (header `5fafe0dbc175`), the seven per-match
scores and the Δ blocks included: `p06s2_t008_d032_mp050_md001` 0.2043 / 0.4549 / 0.5696 / 0.2051 (pooled spellF1,
macroF1, alive/dead F1, mean per match) and `p06s2_t016_d064_mp012_md025` 0.1952 / 0.4525 / 0.5581 / 0.1958, and
`nearest_w50` as in every other log. So the posterior changed no label of these two configs that scores to four
decimals on 7 matches and 3 seeds can show. They cover a minimum possession of 50 (`min_dead` 1) and one of 12 with
`min_dead` 25, so the check is of two of the 90 and not of every config. Only the raw truth was run; the labels do not
depend on the truth, which enters only the scoring.

**The scores.** Possessions per minute, `D` share and alive/dead F1 do not depend on the truth, so each has one column.
The true `D` share is 44.8% of the graded frames, and the true possessions per graded minute are 4.58 raw and 2.48
coarse. On `tracking` v3, the default:

| config | raw spellF1 | raw macroF1 | coarse spellF1 | coarse macroF1 | alive/dead F1 | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|
| `nearest_w50` (yardstick) | 0.2018 | 0.4592 | 0.1842 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` (yardstick) | 0.1665 | 0.4407 | 0.2226 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| **`p06_hmm` (chosen)** | 0.2053 | 0.4569 | 0.1967 | 0.4557 | 0.5676 | 3.10 | 43.1% |
| `p06ctl_seen_off` | 0.1990 | 0.4468 | 0.1971 | 0.4455 | 0.5422 | 2.73 | 46.4% |
| `p06ctl_margin50` | 0.2124 | 0.4532 | 0.1990 | 0.4515 | 0.5876 | 3.29 | 38.3% |

On `context` v1 (sensitivity only):

| config | raw spellF1 | raw macroF1 | coarse spellF1 | coarse macroF1 | alive/dead F1 | possessions per min, predicted | D share, predicted |
|---|---|---|---|---|---|---|---|
| `nearest_w50` (yardstick) | 0.2193 | 0.5144 | 0.2057 | 0.5160 | 0.6575 | 4.37 | 34.2% |
| `nearest_w125` (yardstick) | 0.1774 | 0.4770 | 0.2443 | 0.4877 | 0.6711 | 3.07 | 23.1% |
| **`p06_hmm` (chosen)** | 0.2332 | 0.5304 | 0.2261 | 0.5295 | 0.6408 | 3.23 | 44.5% |
| `p06ctl_seen_off` | 0.2304 | 0.5274 | 0.2270 | 0.5263 | 0.6257 | 2.91 | 47.1% |
| `p06ctl_margin50` | 0.2389 | 0.5287 | 0.2319 | 0.5280 | 0.6560 | 3.44 | 40.0% |
| `p06ctl_d008` | 0.2323 | 0.5293 | 0.2196 | 0.5284 | 0.6404 | 3.38 | 44.2% |
| `p06ctl_d128` | 0.2297 | 0.5342 | 0.2327 | 0.5333 | 0.6302 | 2.65 | 48.6% |

**Each control against the chosen config.** The runner prints no comparison for these pairs (the first config of a run
is its reference), so they are **recomputed here from the 4-decimal per-match scores of the logs** with the benchmark's
own `t_interval` and `sign_flip_p`; the pooled Δ is the difference of the runner's pooled Δ of the control and of the
chosen config against the first config, good to ±0.0001 (R006 explains the rounding). On the 20 comparison lines the
runner did print (every config against the first), the same recomputation reproduces the pooled Δ to 0.0001, the mean,
the interval ends to 0.0001, and the wins and losses exactly, and the p on 19 of the 20: the one difference is a
rounding step (0.1406 recomputed against the runner's 0.1250, `p06ctl_margin50` against `nearest_w125`, coarse,
`context`). A positive Δ means the control is better. The last two columns are the pooled Δ of macroF1 and of
alive/dead F1.

| profile | truth | control | pooled Δ spellF1 | mean per-match Δ [90% t] | sign-flip p | wins / losses | Δ macroF1 | Δ alive/dead F1 |
|---|---|---|---|---|---|---|---|---|
| `tracking` | raw | `p06ctl_seen_off` | -0.0063 | -0.0063 [-0.0079, -0.0046] | 0.0156 | 0 / 7 | -0.0101 | -0.0254 |
| `tracking` | raw | `p06ctl_margin50` | +0.0071 | +0.0071 [+0.0050, +0.0091] | 0.0156 | 7 / 0 | -0.0036 | +0.0200 |
| `tracking` | coarse | `p06ctl_seen_off` | +0.0003 | +0.0003 [-0.0012, +0.0019] | 0.6875 | 4 / 3 | -0.0102 | -0.0254 |
| `tracking` | coarse | `p06ctl_margin50` | +0.0023 | +0.0023 [-0.0002, +0.0047] | 0.1250 | 5 / 2 | -0.0042 | +0.0200 |
| `context` | raw | `p06ctl_seen_off` | -0.0028 | -0.0026 [-0.0052, -0.0001] | 0.0938 | 2 / 5 | -0.0030 | -0.0151 |
| `context` | raw | `p06ctl_margin50` | +0.0057 | +0.0058 [+0.0030, +0.0086] | 0.0156 | 7 / 0 | -0.0017 | +0.0152 |
| `context` | raw | `p06ctl_d008` | -0.0009 | -0.0010 [-0.0026, +0.0007] | 0.2969 | 2 / 5 | -0.0011 | -0.0004 |
| `context` | raw | `p06ctl_d128` | -0.0035 | -0.0032 [-0.0075, +0.0011] | 0.2344 | 2 / 5 | +0.0038 | -0.0106 |
| `context` | coarse | `p06ctl_seen_off` | +0.0009 | +0.0009 [-0.0008, +0.0025] | 0.3594 | 4 / 3 | -0.0031 | -0.0151 |
| `context` | coarse | `p06ctl_margin50` | +0.0058 | +0.0057 [+0.0017, +0.0097] | 0.0312 | 6 / 1 | -0.0015 | +0.0152 |
| `context` | coarse | `p06ctl_d008` | -0.0065 | -0.0066 [-0.0088, -0.0043] | 0.0156 | 0 / 7 | -0.0010 | -0.0004 |
| `context` | coarse | `p06ctl_d128` | +0.0067 | +0.0067 [+0.0016, +0.0119] | 0.0781 | 6 / 1 | +0.0038 | -0.0106 |

**Against the yardsticks, the bar.** The bar (R007's, from the design): raw, Δ over `nearest_w50` above 0 with p of 0.05
or less; coarse, Δ over `nearest_w125` above 0 with p of 0.05 or less. The first yardstick of each truth is the
runner's own lines (pooled Δ, mean per-match Δ with its 90% t-interval, sign-flip p, wins and losses); the second is
recomputed as above.

`tracking`, raw (`nearest_w50` 0.2018, `nearest_w125` 0.1665):

| config | pooled spellF1 | Δ over `nearest_w50`, pooled | mean per-match Δ [90% t] | p | wins / losses | Δ over `nearest_w125`, pooled | mean per-match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` | 0.2053 | +0.0036 | +0.0041 [-0.0022, +0.0104] | 0.2500 | 5 / 2 | +0.0389 | +0.0389 [+0.0309, +0.0468] | 0.0156 | 7 / 0 |
| `p06ctl_seen_off` | 0.1990 | -0.0027 | -0.0021 [-0.0087, +0.0045] | 0.5312 | 4 / 3 | +0.0326 | +0.0326 [+0.0233, +0.0419] | 0.0156 | 7 / 0 |
| `p06ctl_margin50` | 0.2124 | +0.0107 | +0.0112 [+0.0050, +0.0174] | 0.0312 | 6 / 1 | +0.0460 | +0.0459 [+0.0389, +0.0529] | 0.0156 | 7 / 0 |

`tracking`, coarse (`nearest_w125` 0.2226, `nearest_w50` 0.1842):

| config | pooled spellF1 | Δ over `nearest_w125`, pooled | mean per-match Δ [90% t] | p | wins / losses | Δ over `nearest_w50`, pooled | mean per-match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` | 0.1967 | -0.0259 | -0.0256 [-0.0363, -0.0150] | 0.0156 | 0 / 7 | +0.0125 | +0.0126 [+0.0090, +0.0162] | 0.0156 | 7 / 0 |
| `p06ctl_seen_off` | 0.1971 | -0.0256 | -0.0253 [-0.0372, -0.0133] | 0.0156 | 0 / 7 | +0.0128 | +0.0129 [+0.0088, +0.0171] | 0.0156 | 7 / 0 |
| `p06ctl_margin50` | 0.1990 | -0.0236 | -0.0234 [-0.0326, -0.0142] | 0.0156 | 0 / 7 | +0.0148 | +0.0149 [+0.0122, +0.0175] | 0.0156 | 7 / 0 |

`context`, raw (`nearest_w50` 0.2193, `nearest_w125` 0.1774), sensitivity only:

| config | pooled spellF1 | Δ over `nearest_w50`, pooled | mean per-match Δ [90% t] | p | wins / losses | Δ over `nearest_w125`, pooled | mean per-match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` | 0.2332 | +0.0140 | +0.0144 [+0.0088, +0.0199] | 0.0156 | 7 / 0 | +0.0559 | +0.0555 [+0.0461, +0.0649] | 0.0156 | 7 / 0 |
| `p06ctl_seen_off` | 0.2304 | +0.0112 | +0.0118 [+0.0050, +0.0185] | 0.0312 | 6 / 1 | +0.0531 | +0.0529 [+0.0440, +0.0617] | 0.0156 | 7 / 0 |
| `p06ctl_margin50` | 0.2389 | +0.0197 | +0.0202 [+0.0144, +0.0260] | 0.0156 | 7 / 0 | +0.0616 | +0.0613 [+0.0540, +0.0686] | 0.0156 | 7 / 0 |
| `p06ctl_d008` | 0.2323 | +0.0131 | +0.0134 [+0.0079, +0.0190] | 0.0156 | 7 / 0 | +0.0550 | +0.0545 [+0.0440, +0.0651] | 0.0156 | 7 / 0 |
| `p06ctl_d128` | 0.2297 | +0.0105 | +0.0112 [+0.0029, +0.0194] | 0.0469 | 6 / 1 | +0.0524 | +0.0523 [+0.0448, +0.0598] | 0.0156 | 7 / 0 |

`context`, coarse (`nearest_w125` 0.2443, `nearest_w50` 0.2057), sensitivity only:

| config | pooled spellF1 | Δ over `nearest_w125`, pooled | mean per-match Δ [90% t] | p | wins / losses | Δ over `nearest_w50`, pooled | mean per-match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|---|---|
| `p06_hmm` | 0.2261 | -0.0182 | -0.0185 [-0.0329, -0.0040] | 0.0781 | 1 / 6 | +0.0204 | +0.0203 [+0.0155, +0.0251] | 0.0156 | 7 / 0 |
| `p06ctl_seen_off` | 0.2270 | -0.0173 | -0.0177 [-0.0319, -0.0034] | 0.0781 | 1 / 6 | +0.0213 | +0.0212 [+0.0172, +0.0252] | 0.0156 | 7 / 0 |
| `p06ctl_margin50` | 0.2319 | -0.0124 | -0.0128 [-0.0265, +0.0009] | 0.1250 | 1 / 6 | +0.0262 | +0.0260 [+0.0234, +0.0286] | 0.0156 | 7 / 0 |
| `p06ctl_d008` | 0.2196 | -0.0247 | -0.0251 [-0.0407, -0.0095] | 0.0469 | 1 / 6 | +0.0139 | +0.0137 [+0.0087, +0.0187] | 0.0156 | 7 / 0 |
| `p06ctl_d128` | 0.2327 | -0.0115 | -0.0118 [-0.0256, +0.0020] | 0.1719 | 1 / 6 | +0.0271 | +0.0270 [+0.0210, +0.0330] | 0.0156 | 7 / 0 |

**What the controls say**:
- **The `seen` term does what the design said, and is worth little.** Without it, on `tracking` under the raw truth, the
  chosen config loses 0.0063 (0 wins and 7 losses, p 0.0156), goes from +0.0036 above `nearest_w50` to 0.0027 below it
  (p 0.5312), and moves as the design expected: possessions per minute 3.10 to 2.73 (the true 4.58), `D` share 43.1% to
  46.4% (the true 44.8%), alive/dead F1 0.5676 to 0.5422, macroF1 -0.0101. Under the coarse truth it moves nothing
  (+0.0003, p 0.6875, 4 wins and 3 losses; macroF1 -0.0102). On `context` (sensitivity only) it costs 0.0028 raw (p
  0.0938, 2 wins in 7) and gains 0.0009 coarse (p 0.3594). So the term is worth up to 0.0063, under the raw truth,
  clearest on `tracking`, and its sign does not hold under the coarse truth. It stays, by rule from A01.
- **`margin` 50, a diagnostic: part of the HMM's `D` is evidence that costs it.** With `D` evidence only inside the
  gaps' interiors the predicted `D` share falls (43.1% to 38.3% on `tracking`, 44.5% to 40.0% on `context`, sensitivity
  only; the true 44.8%), alive/dead F1 rises by +0.0200 and +0.0152 (to 0.5876 and 0.6560, against 0.6094 and 0.6428 for
  the yardsticks on `tracking` and 0.6575 and 0.6711 on `context`), possessions per minute rise (3.10 to 3.29; 3.23 to
  3.44), and macroF1 falls in all four cells (-0.0036 and -0.0042 on `tracking`, -0.0017 and -0.0015 on `context`, raw
  and coarse). spellF1 is above the chosen config's in all four cells: +0.0071 raw on `tracking` (7 wins, p 0.0156) and
  +0.0057 on `context` (7 wins, p 0.0156); +0.0023 coarse on `tracking` (5 wins and 2 losses, the interval includes 0, p
  0.1250) and +0.0058 on `context` (6 wins, p 0.0312). **Against the yardsticks it clears the raw half of the bar on
  `tracking`**: +0.0107 over `nearest_w50` (mean per match +0.0112 [+0.0050, +0.0174], p 0.0312, 6 wins and 1 loss; the
  loss is J03WN1, -0.0053), and **not the coarse half**: 0.0236 below `nearest_w125` (0 wins in 7, p 0.0156; 0.0124
  below on `context`, 1 win in 7, p 0.1250). The logs do not separate whether the gain comes from the short gaps or from
  the edges of the long ones. It was run once, at one value, and by a ruling before the run and not by a score; it is
  not a candidate and `p06_hmm.toml` is unchanged. A pass of one half of the bar after one look at one value of an
  untuned parameter is what the selection rule exists to prevent, so what it supports is a follow-up with its own
  protocol (values, objective and tie rule declared before it runs): proposals.md, "After P06".
- **`switch_dead` on `context` (sensitivity only): d* is not specific to `tracking`'s gaps under the raw truth, and the
  coarse truth wants more.** Raw pooled spellF1 at d = 8, 32 and 128 is 0.2323, 0.2332 and 0.2297: the chosen value is
  the highest, by 0.0009 and 0.0035, neither interval excludes 0 (p 0.2969 and 0.2344, 2 wins and 5 losses each), a span
  of 0.0035 over a 16-fold range of d (R006's span on `tracking` at t = 16 over d = 8 to 64 is 0.0021). Coarse: 0.2196,
  0.2261 and 0.2327; a quarter of d* is 0.0065 lower (0 wins in 7, p 0.0156) and four times d* is 0.0067 higher (6 wins
  in 7, p 0.0781). Possessions per minute fall with d (3.38, 3.23, 2.65; the coarse truth has 2.48), `D` share 44.2%,
  44.5% and 48.6%, alive/dead F1 0.6404, 0.6408 and 0.6302. Placing the gaps by the play does not move the best d beyond
  noise under the raw objective, which decides; the coarse truth prefers more smoothing, as it preferred a larger t
  (R006) and a longer minimum run (above), the disagreement T33 describes and not a new one.
- **`context` (sensitivity only) against `tracking`, the chosen config.** Every score is higher on `context`: spellF1
  0.2053 to 0.2332 (+0.0279) raw and 0.1967 to 0.2261 (+0.0294) coarse, in all 7 matches under both truths; macroF1
  +0.0735 raw and +0.0738 coarse; alive/dead F1 0.5676 to 0.6408 (+0.0732). The yardsticks rise too, less than the
  chosen config, most clearly under the raw truth: `nearest_w50` +0.0175 and `nearest_w125` +0.0109 raw, +0.0215 and
  +0.0217 coarse. So the chosen config's position improves: raw, +0.0036 to +0.0140 over `nearest_w50` (mean per match
  +0.0144 [+0.0088, +0.0199], p 0.0156, 7 wins in 7), which would clear the raw half of the bar; coarse, -0.0259 to
  -0.0182 against `nearest_w125` (-0.0185 [-0.0329, -0.0040], p 0.0781, 1 win in 7), which would not clear the coarse
  half. Its alive/dead F1 is 0.0167 under `nearest_w50`'s (0.6408 against 0.6575), against 0.0418 on `tracking` (0.5676
  against 0.6094). That is the direction the design predicted, since `ball_presence` is evidence about `D` only where
  the gaps are related to the play: `tracking` v3 transplants them without regard to it (the ball is visible on 56.8% of
  its dead frames, against about 59% of all graded frames, R005), and `context` places them by a hazard fitted on our
  tracker (46.2% of its dead frames, against 56.9%). `context` fails its near-line gate by a wide margin, so these are
  sensitivity figures and not a second validation.

Runtime of the controls: the runner's `seconds` line says 16.7 to 18.4 s per HMM config on `tracking` (16.7 to 18.4 raw,
17.3 to 18.1 coarse; the yardsticks 7.8 to 8.4 s) and 14.9 to 16.0 s on `context` (the yardsticks 6.3 to 7.3 s).
Verdict: **(switch_team, switch_dead, min_possession, min_dead) = (8, 32, 75, 1) is P06's parameter set**, chosen by the
pre-declared rule over the set that Stage 2b's ruling extended after Stage 2's results (over the declared set alone the
rule picks `p06s2_t016_d064_mp050_md025`, -0.0012 raw against `nearest_w50`, and the verdict is the same), and
`pipeline/configs/p06_hmm.toml` holds it; the controls leave it as it is. It is not the
submission config, and `main.toml` is unchanged. On the benchmark's default profile it is level with the strongest
yardstick under the raw truth (+0.0036, p 0.2500) and 0.0259 below it under the coarse one, so **the bar is not met and
nothing here justifies replacing `main.toml`**. What the controls add. The `seen` term is worth 0.0063 raw on `tracking`
and nothing coarse. `margin` 50 is above the chosen config in all four profile and truth cells (two of them on
`context`, sensitivity only; +0.0023 to +0.0071, the interval of the smallest includes 0) and clears the raw half of the
bar on `tracking` (+0.0107 over `nearest_w50`, p 0.0312, 6 wins in 7) but not the coarse half (-0.0236 against
`nearest_w125`), so part of the HMM's `D` evidence costs it; as a diagnostic it changes nothing here, and a `margin`
sweep with its own pre-declared protocol is the follow-up (proposals.md, "After P06"). d* = 32 is not specific to
`tracking` under the raw truth (a span of 0.0035 over a 16-fold range of d on `context`, sensitivity only), and the
coarse truth wants a larger d. On `context`, a sensitivity profile, every score is higher and the chosen config clears
the raw half of the bar (+0.0140, 7 wins in 7) and not the coarse half (-0.0182, p 0.0781). Three things are not
settled: the two truths want different minimum runs (raw 75 frames, coarse 150 or more), and the coarse winner is on the
grid's upper edge, so the choice rests on the raw objective alone, which T33 warns about; the winner is on an edge of
Stage 2b's grid in t and d, and by the ruling that stays; and the HMM's `D` is worse than the yardsticks' on every
config (alive/dead F1 0.5676 against 0.6094 and 0.6428), while the `margin` control shows that its `D` evidence moves
spellF1. Where the gap to a clear win lies is what A10 measures: with a perfect source the chosen decoder scores 0.3678
raw and 0.4417 coarse (a perfect dead-ball detector), 0.3586 and 0.3651 (a perfect possessor where the ball is seen) and
0.7627 and 0.9226 (both, everywhere, so the evidence for `D` and for the team are both worth a lot, and the team
evidence inside the gaps is worth nothing until the `D` there is right). On the public window it is valid and
deterministic, and its spellF1 of 0.3500 sits between `baseline_smooth50`'s 0.3256 and `baseline_smooth125`'s 0.4242,
which is a correctness check and no evidence either way (T01).

### R008 — P04's controlled-touch evidence: the 24-config grid and the choice of its rule (2026-10-25, Tian)
Proposal: P04 · Commit: `71dc1c6` (the 24 configs `pipeline/benchmark/configs/p45/p45a_*.toml` and `touch_stats.py`, run
from a detached checkout of it; the runner's header says `71dc1c69e75f`; the configs are identical at the entry's parent
`a2c1eee`, where the public window was run) · Command: six processes at once, three shares of 8 configs, each share run
once per truth, all with `--profile tracking --seeds 3 --no-write`:
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml <8 configs> --profile tracking --seeds 3 --truth raw --no-write`
and the same with `nearest_w125.toml` first, `nearest_w50.toml` second and `--truth coarse`. Public window:
`python pipeline/run.py pipeline/benchmark/configs/p45/p45a_r20_v6_k03_h55.toml --score`
Public (a correctness check only, T01): macroF1 0.6532 · spellF1 0.3889 · alive/dead F1 0.8893 · 19 possessions against
the truth's 17, mean 13.9 s against its 14.2 s. The file is valid (199,319 rows), sha256
`713f86708a012ff4b8bbd71c1982a4e88ed17e8acf39424c1b849659923b1956`. Over all the graded frames of the match it labels H
34.7%, A 23.5% and D 41.8% and finds 312 possessions (mean 11.8 s). `--score` reports `controlled_touch` agreeing with
the public GT on 87.2% of 5,489 frames (the judged frames where it has an opinion) and `ball_presence` on alive on
81.8% of 7,500 frames. `controlled_touch` has an opinion on home on 59.2% of the graded frames (episodes plus the hold)
and none on alive. No bootstrap was run, because no public-window gap is claimed. Against R007's chosen config on the
same window (0.6393, 0.3500, 23 possessions) the figures are higher, which says nothing either way (T01); the
`ball_presence` values are A01's and in-sample there, and `controlled_touch`'s `confidence` (0.91) is a benchmark
measurement and not the window's.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4 graded
minutes per seed. The chosen config, `p45a_r20_v6_k03_h55` (controlled_touch radius 2.0 m, max_rel_speed 6.0 m/s,
min_frames 3, confidence 0.91, hold 0.55; `ball_presence` at its defaults; hmm 8 / 32 / 75 / 1): pooled spellF1 0.2345
raw (mean per match 0.2356 [0.2191, 0.2522]) and 0.2433 coarse (0.2437 [0.2297, 0.2577]). Against `nearest_w50` under
the raw truth: pooled Δ +0.0328, mean per match +0.0334 [+0.0203, +0.0464], sign-flip p 0.0156, 7 wins and 0 losses.
Against `nearest_w125` under the coarse truth: pooled Δ +0.0206, mean +0.0209 [+0.0156, +0.0261], p 0.0156, 7 wins and 0
losses. **Stage A alone meets both halves of P06's bar**; the bar is Stage C's to apply to the combined config.
Changed vs the previous best: the benchmark's best before this entry were the four yardsticks (R004c, `nearest_w50`
0.2018 raw and `nearest_w125` 0.2226 coarse) and R007's chosen decoder on `nearest_player` (0.2053 raw, 0.1967 coarse).
Stage A replaces `nearest_player` by `controlled_touch` with everything else equal (`ball_presence`, hmm 8 / 32 / 75 /
1), so the P04 gain at the same decoder is +0.0292 raw (0.2345 against 0.2053) and +0.0466 coarse (0.2433 against
0.1967), pooled; R007's per-match scores are not in this entry's logs, so no interval is given for it.

**Checks on the runs.** All six logs end `EXIT=0`. Each lists the two references first (raw: `nearest_w50`, then
`nearest_w125`; coarse the reverse) and 8 of the 24 configs; the three shares are disjoint and together are the 24, none
missing or repeated. The references are the same in all six logs to the last printed digit (`seconds` aside) and are
R004c's: `nearest_w50` 0.2018 raw (macroF1 0.4592, alive/dead 0.6094) and 0.1842 coarse, `nearest_w125` 0.1665 raw and
0.2226 coarse (macroF1 0.4509, alive/dead 0.6428). The runner's `seconds`, across the six logs: 8.2 to 10.8 s for
the yardsticks and 27.5 to 29.5 s for the grid's configs; the first config of every process took 62.6 to 65.2 s,
because it carries the one-off kinematics and features pass (the memo then serves the rest of that process). No grid
was shrunk: the 60 s per-config budget applies to steady-state configs, and a process's first config carries the
one-off pass (it guards runtime only; no result depends on it).

**What the rule measures (`touch_stats.py`; `tracking` v3, seeds 0 to 2, all seven matches).** Among the alive
graded frames with the ball visible: coverage is the share inside a kept episode, agreement the share of episode frames
whose team is the truth's, and `confidence` is floor(100 x pooled agreement) / 100. **The confidences are measurements
on all seven matches, the same matches the grid is then scored on.** They are a fixed rule and were never tuned towards
a score, but they are not out of sample either, and the leave-one-match-out estimate below does not remove that, because
the measurement is pooled over all seven. The lag is, per truth change between H and A with no D between, the number of
frames from the change to the nearest start of an episode of the new team within +-5 s, the neighbouring changes of the
truth bounding the window (the table is after that fix). Lags are in frames (25 per second).

| setting | coverage | agreement | per-match agreement | lag median | lag p25 | lag p75 | changes found | confidence |
|---|---|---|---|---|---|---|---|---|
| r1.0 v3.0 k3 | 28.8% | 95.6% | 93.3%-97.1% | 3 | -1 | 39 | 35.1% | 0.95 |
| r1.0 v3.0 k8 | 26.7% | 95.8% | 93.5%-97.3% | 17 | 0 | 48 | 29.2% | 0.95 |
| r1.0 v6.0 k3 | 33.7% | 94.6% | 91.8%-96.3% | -1 | -4 | 24 | 43.7% | 0.94 |
| r1.0 v6.0 k8 | 31.5% | 95.2% | 92.8%-96.8% | 3 | -4 | 39 | 33.6% | 0.95 |
| r1.5 v3.0 k3 | 36.4% | 94.8% | 92.5%-96.4% | 3 | -2 | 35 | 38.1% | 0.94 |
| r1.5 v3.0 k8 | 35.2% | 95.0% | 92.6%-96.6% | 5 | -1 | 40 | 34.0% | 0.95 |
| r1.5 v6.0 k3 | 43.1% | 93.2% | 90.3%-95.1% | -1 | -6 | 12 | 48.2% | 0.93 |
| r1.5 v6.0 k8 | 41.8% | 93.9% | 91.1%-95.7% | 0 | -7 | 31 | 39.7% | 0.93 |
| r2.0 v3.0 k3 | 39.4% | 94.0% | 91.7%-95.6% | 4 | -1 | 34 | 39.4% | 0.93 |
| r2.0 v3.0 k8 | 38.4% | 94.2% | 92.0%-95.8% | 6 | -1 | 36 | 35.7% | 0.94 |
| r2.0 v6.0 k3 | 47.6% | 91.9% | 88.9%-94.0% | -1 | -8 | 12 | 49.7% | 0.91 |
| r2.0 v6.0 k8 | 46.5% | 92.4% | 89.5%-94.4% | 0 | -9 | 26 | 42.6% | 0.92 |

Agreement falls with radius and `max_rel_speed` (95.8% to 91.9%) and coverage rises (26.7% to 47.6%). The agreement
column is printed rounded to a tenth: `r2.0 v3.0 k3` shows 94.0% with confidence 0.93, so its unrounded agreement is in
[93.95%, 94.00%), and the floor is taken on that.

**The grid.** The 12 settings (radius in {1.0, 1.5, 2.0} m x `max_rel_speed` in {3.0, 6.0} m/s x `min_frames` in {3, 8})
x `hold` in {0.5 (off), 0.55}: 24 configs `p45a_r{RR}_v{V}_k{KK}_h{HH}` (RR the radius in decimetres, V the speed,
KK the minimum frames, HH the hold in hundredths, 50 = off). Each holds `controlled_touch` at the setting's measured
`confidence`, `ball_presence` at its defaults and `hmm` at (8, 32, 75, 1). The table gives pooled spellF1 and macroF1
under each truth, alive/dead F1 (the same under both: the predicted labels do not depend on the truth), the predicted
possessions per graded minute (the truth has 4.58 raw and 2.48 coarse) and the predicted `D` share (the truth's is
44.8%). The bold row is the chosen config; the two references come first.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | 0.1842 | 0.4592 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` | 0.1665 | 0.2226 | 0.4407 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| `p45a_r10_v3_k03_h50` | 0.1489 | 0.1940 | 0.4606 | 0.4665 | 0.5730 | 1.77 | 43.8% |
| `p45a_r10_v3_k03_h55` | 0.1844 | 0.2182 | 0.4716 | 0.4757 | 0.5730 | 1.99 | 43.4% |
| `p45a_r10_v3_k08_h50` | 0.1359 | 0.1840 | 0.4546 | 0.4608 | 0.5724 | 1.67 | 43.9% |
| `p45a_r10_v3_k08_h55` | 0.1606 | 0.2015 | 0.4627 | 0.4680 | 0.5729 | 1.86 | 43.4% |
| `p45a_r10_v6_k03_h50` | 0.1685 | 0.2013 | 0.4651 | 0.4700 | 0.5735 | 1.94 | 43.8% |
| `p45a_r10_v6_k03_h55` | 0.2207 | 0.2318 | 0.4808 | 0.4804 | 0.5730 | 2.18 | 43.3% |
| `p45a_r10_v6_k08_h50` | 0.1494 | 0.1895 | 0.4576 | 0.4636 | 0.5732 | 1.81 | 43.8% |
| `p45a_r10_v6_k08_h55` | 0.1823 | 0.2138 | 0.4689 | 0.4728 | 0.5733 | 1.98 | 43.3% |
| `p45a_r15_v3_k03_h50` | 0.1605 | 0.2000 | 0.4626 | 0.4681 | 0.5738 | 1.93 | 43.7% |
| `p45a_r15_v3_k03_h55` | 0.1970 | 0.2278 | 0.4750 | 0.4784 | 0.5737 | 2.14 | 43.2% |
| `p45a_r15_v3_k08_h50` | 0.1550 | 0.1947 | 0.4605 | 0.4660 | 0.5741 | 1.89 | 43.6% |
| `p45a_r15_v3_k08_h55` | 0.1840 | 0.2202 | 0.4718 | 0.4760 | 0.5738 | 2.07 | 43.2% |
| `p45a_r15_v6_k03_h50` | 0.1822 | 0.2056 | 0.4669 | 0.4709 | 0.5739 | 2.11 | 43.6% |
| `p45a_r15_v6_k03_h55` | 0.2348 | 0.2410 | 0.4834 | 0.4823 | 0.5734 | 2.35 | 43.2% |
| `p45a_r15_v6_k08_h50` | 0.1718 | 0.1993 | 0.4633 | 0.4675 | 0.5741 | 2.06 | 43.6% |
| `p45a_r15_v6_k08_h55` | 0.2068 | 0.2306 | 0.4772 | 0.4793 | 0.5739 | 2.22 | 43.2% |
| `p45a_r20_v3_k03_h50` | 0.1642 | 0.2014 | 0.4624 | 0.4676 | 0.5738 | 2.01 | 43.6% |
| `p45a_r20_v3_k03_h55` | 0.2013 | 0.2315 | 0.4754 | 0.4787 | 0.5737 | 2.23 | 43.2% |
| `p45a_r20_v3_k08_h50` | 0.1617 | 0.1979 | 0.4614 | 0.4661 | 0.5742 | 2.01 | 43.5% |
| `p45a_r20_v3_k08_h55` | 0.1908 | 0.2248 | 0.4734 | 0.4771 | 0.5738 | 2.18 | 43.2% |
| `p45a_r20_v6_k03_h50` | 0.1843 | 0.2063 | 0.4659 | 0.4697 | 0.5740 | 2.22 | 43.5% |
| **`p45a_r20_v6_k03_h55`** | **0.2345** | **0.2433** | 0.4832 | 0.4824 | 0.5731 | 2.42 | 43.2% |
| `p45a_r20_v6_k08_h50` | 0.1808 | 0.2036 | 0.4646 | 0.4681 | 0.5741 | 2.19 | 43.5% |
| `p45a_r20_v6_k08_h55` | 0.2169 | 0.2380 | 0.4787 | 0.4803 | 0.5735 | 2.34 | 43.2% |

**The choice, by the pre-declared rule.** Objective: pooled spellF1 under the raw truth. The best raw config is
`p45a_r15_v6_k03_h55` at 0.2348. The tie band is 0.005, so the tied set is raw >= 0.2298. Two configs are in it:
`p45a_r15_v6_k03_h55` (0.2348 raw, 0.2410 coarse) and `p45a_r20_v6_k03_h55` (0.2345 raw, 0.2433 coarse). The next,
`r10_v6_k03_h55`, is at 0.2207, 0.0141 below the best. Among the tied, the best coarse score is `r20_v6_k03_h55` (0.2433
against 0.2410, by 0.0023), so the tie order (larger `min_frames`, smaller radius, smaller `max_rel_speed`, hold off) is
not reached: the two share `min_frames`, `max_rel_speed` and `hold`, and the radius would have preferred 1.5. **Chosen:
`p45a_r20_v6_k03_h55`**, 0.0003 below the raw best.
- **The plateau** (within 0.005 of the best raw): those two configs, both at `max_rel_speed` 6.0, `min_frames` 3 and
  `hold` 0.55, radius 1.5 and 2.0. It is narrow. The best config outside it with `min_frames` 8 is 0.2169
  (`r20_v6_k08_h55`), with radius 1.0 0.2207 (`r10_v6_k03_h55`), with `max_rel_speed` 3.0 0.2013 (`r20_v3_k03_h55`) and
  with the hold off 0.1843 (`r20_v6_k03_h50`).
- **The raw-only winner** is `p45a_r15_v6_k03_h55` (0.2348); **the coarse-only winner** is `p45a_r20_v6_k03_h55`
  (0.2433), which is the chosen config, so the coarse truth does not disagree with the choice.
- **At the edges.** The chosen config sits on the grid's upper edge in radius (2.0) and `max_rel_speed` (6.0), on its
  lower edge in `min_frames` (3), and at the only non-off `hold` (0.55). **Per the protocol nothing is added after the
  results**: a wider radius, a faster `max_rel_speed`, a shorter minimum episode and other holds were not tried, and the
  entry does not claim they would not score higher. Averaged over the other factors (hold 0.55), raw spellF1 rises with
  radius (1.0: 0.1870, 1.5: 0.2056, 2.0: 0.2109), with `max_rel_speed` (3.0: 0.1864, 6.0: 0.2160) and with a shorter
  minimum (3: 0.2121, 8: 0.1902), and the coarse truth goes the same way (radius 0.2163, 0.2299, 0.2344; speed 0.2207,
  0.2331; minimum 0.2323, 0.2215). The gain from radius flattens (+0.0186, then +0.0053), so that edge matters least.

**Δ for the top three under each truth** (spellF1; pooled Δ, mean per match with its 90% t-interval, sign-flip p,
wins in 7; against `nearest_w50` raw and `nearest_w125` coarse).

| truth | config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins |
|---|---|---|---|---|---|---|
| raw | `r15_v6_k03_h55` | 0.2348 | +0.0330 | +0.0337 [+0.0183, +0.0491] | 0.0156 | 7 |
| raw | `r20_v6_k03_h55` (chosen) | 0.2345 | +0.0328 | +0.0334 [+0.0203, +0.0464] | 0.0156 | 7 |
| raw | `r10_v6_k03_h55` | 0.2207 | +0.0189 | +0.0199 [+0.0031, +0.0368] | 0.0781 | 5 |
| coarse | `r20_v6_k03_h55` (chosen) | 0.2433 | +0.0206 | +0.0209 [+0.0156, +0.0261] | 0.0156 | 7 |
| coarse | `r15_v6_k03_h55` | 0.2410 | +0.0184 | +0.0186 [+0.0122, +0.0249] | 0.0156 | 7 |
| coarse | `r20_v6_k08_h55` | 0.2380 | +0.0154 | +0.0155 [+0.0080, +0.0231] | 0.0312 | 6 |

The two leaders' macroF1 Δ is +0.0240 to +0.0241 raw and +0.0313 to +0.0315 coarse, 7 wins in 7 under both truths. The
gap between the two leaders (0.0003 raw, 0.0023 coarse) is far inside the noise: the benchmark's own per-match interval
half-widths for these configs are about 0.013 to 0.017 raw and 0.014 coarse, and the full-match floor of our single
match (about 0.04, T07) is larger still. The choice between them is the rule's, not a finding.

**Leave-one-match-out estimate of the selection.** As R007 did, from the per-match spellF1 the runner prints (four
decimals): for each match, the selection is repeated on the other six matches and the chosen config's score on the
held-out match is compared with the reference's on it; the seven differences give the mean Δ with its 90% t-interval,
the exact sign-flip p over the 2^7 sign patterns, and the wins. The references are not candidates.
- **The raw objective alone** (the config of the 24 with the highest mean per-match raw spellF1 on the six, no tie
  band): `r15_v6_k03_h55` is picked for four matches (J03WMX, J03WN1, J03WOH, J03WPY) and `r20_v6_k03_h55` for three
  (J03WOY, J03WQQ, J03WR9). Scored under the raw truth against `nearest_w50`: mean Δ +0.0316 [+0.0179, +0.0452], p
  0.0156, 7 wins and 0 losses (per match +0.0301, +0.0099, +0.0310, +0.0527, +0.0047, +0.0478, +0.0447). Scored under
  the coarse truth against `nearest_w125`: mean Δ +0.0176 [+0.0117, +0.0236], p 0.0156, 7 wins and 0 losses.
- **The whole rule** (on each fold: the tied set is the configs within 0.005 of the best six-match raw mean, then the
  best six-match coarse mean among them): `r20_v6_k03_h55` is picked in all seven folds, so the held-out estimate is the
  chosen config's own: +0.0334 [+0.0203, +0.0464] raw and +0.0209 [+0.0156, +0.0261] coarse, p 0.0156, 7 wins and 0
  losses under each.

The selection is stable (the same two configs in every fold), so the choice is not what produces the gap. The estimate
does not cover the `confidence` measurement, which uses all seven matches, and seven matches from two competitions (five
of them Fortuna Düsseldorf home games) are not seven independent draws.

**`hold` against the flip lags.** `hold` 0.55 beats off in all 12 settings under both truths: raw +0.0247 to +0.0526
(mean +0.0376), coarse +0.0175 to +0.0370 (mean +0.0287); the chosen setting gains +0.0502 raw (0.2345 against 0.1843)
and +0.0370 coarse (0.2433 against 0.2063). With the hold off, all 12 configs are below `nearest_w50` raw (0.1359 to
0.1843 against 0.2018) and below `nearest_w125` coarse (0.1840 to 0.2063 against 0.2226). The lags suggest why, as a
reading and not a test: for the chosen setting the nearest episode start of the new team in the bounded window (+-5 s,
cut at the neighbouring changes of the truth) is a median of 1 frame before the flag changes (p25 -8, p75 +12), so when
one is found it is on time, but only 49.7% of changes have one in that window. Not found means only that no new-team
episode starts in the window; the source can speak in it (for the old team, or through the hold), and this entry does
not say why. A weak vote (0.55, a likelihood ratio of 1.22) for the last controlling team gives the decoder continuity
between episodes. Predicted possessions per graded minute are 1.67 to 2.22 with the hold off and 1.86 to 2.42 with it
on, so the hold also lengthens the count towards the truth. How strong the hold should be is not known: 0.55 is the only
value the grid has, and it is the edge.

**Alive/dead and possessions.** Alive/dead F1 is 0.5724 to 0.5742 for all 24, 0.0048 to 0.0066 above R007's decoder at
the same `ball_presence` (0.5676) and below both yardsticks (0.6094, 0.6428). The Stage-A source has no opinion on
alive, so the grid barely moves it (the small shifts are the decoder's, as the team evidence changes where `D` is
cheapest); the gap to the yardsticks is R007's and A10's `D` problem, which Stage B (P05) addresses. Predicted
possessions are 1.67 to 2.42 per graded minute against 4.58 under the raw truth (2.48 coarse): fewer than the truth for
every config, and the chosen config's 2.42 is close to the coarse truth's 2.48. `D` share is 43.2% to 43.9% against
44.8%.

**Controls** (diagnostics, never candidates; run from a detached checkout at `32a8405`, header `32a84055019f`,
`--profile tracking` or `--profile context`, `--seeds 3 --no-write`, the references first as in the grid; all four logs
end `EXIT=0`, and the references and the chosen config score on `tracking` exactly as in the grid's logs, to the last
printed digit).
- **`p45actl_nearest`** (the chosen config plus `nearest_player` at 0.78) **on `tracking`: adding it costs both
  truths.** Raw pooled spellF1 0.2184 against the chosen config's 0.2345 (-0.0161): the chosen config is better by
  +0.0160 [+0.0088, +0.0231] per match, p 0.0312, 6 wins in 7. Coarse 0.2081 against 0.2433 (-0.0352): the chosen config
  is better by +0.0352 [+0.0280, +0.0423], p 0.0156, 7 wins in 7. Against the yardsticks, the control is +0.0166 over
  `nearest_w50` raw (mean per match +0.0174 [+0.0085, +0.0263], p 0.0312, 6 wins) and -0.0145 under `nearest_w125`
  coarse (-0.0143 [-0.0248, -0.0037], p 0.0156, 0 wins in 7). It predicts 3.08 possessions per graded minute (the chosen
  config 2.42), a `D` share of 43.0%, alive/dead F1 0.5691 (0.0040 below the chosen config) and macroF1 0.4649 raw,
  0.4637 coarse (the chosen config 0.4832, 0.4824). So the nearest-player vote does not complement the controlled-touch
  evidence here: with it, the config is no better than the references coarse and clears `nearest_w50` raw by half the
  margin. The control does not say why (more hand-overs, 3.08 against 2.42 per minute, is what the count shows).
- **The chosen config on `context`** (sensitivity only, T38; `context` v1 fails two of its gates, R005). Pooled spellF1
  0.2761 raw (mean per match 0.2775 [0.2596, 0.2954]) and 0.2977 coarse (0.2978 [0.2822, 0.3133]); alive/dead F1 0.6456
  (`nearest_w50` 0.6575, `nearest_w125` 0.6711), 2.46 possessions per graded minute, `D` share 44.5%. Against
  `nearest_w50` raw: pooled Δ +0.0568, mean +0.0578 [+0.0474, +0.0681], p 0.0156, 7 wins in 7. Against `nearest_w125`
  coarse: +0.0534, mean +0.0531 [+0.0353, +0.0709], p 0.0156, 7 wins in 7. The yardsticks on `context` are R005's
  (0.2193 raw `nearest_w50`, 0.2443 coarse `nearest_w125`). At the same decoder, R007's chosen config on `context`
  scored 0.2332 raw and 0.2261 coarse (R007), so the P04 gain there is +0.0429 raw and +0.0716 coarse, pooled.
- **The four profile x truth cells agree.** The chosen config is above its truth's strongest yardstick in all four, in 7
  matches of 7 each, p 0.0156: `tracking` raw +0.0328, `tracking` coarse +0.0206, `context` raw +0.0568, `context`
  coarse +0.0534. There is no warning to write (T38); the gain is larger on `context`, as every score is (R005, R007),
  and `context` is a sensitivity check and not a second validation.

Runtime of the controls: the chosen config's first config in each process took 48.7 to 50.6 s (the one-off pass), the
control 29.2 and 29.6 s, the yardsticks 7.1 to 7.8 s.

Verdict: **`p45a_r20_v6_k03_h55` (controlled_touch radius 2.0 m, `max_rel_speed` 6.0 m/s, `min_frames` 3, `confidence`
0.91, `hold` 0.55, with `ball_presence` and the hmm (8, 32, 75, 1)) is P04's chosen rule**, by the pre-declared rule
over the 24 configs; it is not the submission config, and `main.toml` and `p06_hmm.toml` are unchanged. For P04 alone,
on `tracking` v3, it clears both halves of the bar P06 set (raw +0.0328 over `nearest_w50`, p 0.0156, 7 wins in 7;
coarse +0.0206 over `nearest_w125`, p 0.0156, 7 wins in 7), the leave-one-match-out estimates agree (raw objective alone
+0.0316 raw and +0.0176 coarse, the whole rule +0.0334 and +0.0209, each 7 wins in 7), and the `context` sensitivity
profile is above both yardsticks by more (+0.0568, +0.0534). Whether the combined config meets the bar is Stage C's to
apply (R010), not this entry's. The limits: the choice is on the grid's upper edge in radius and `max_rel_speed`, its
lower edge in `min_frames` and at the only non-off `hold`, and by the protocol none is extended; the confidences are
measurements on the seven matches the grid is scored on; the result depends on the hold (off, no config beats either
yardstick); alive/dead F1 (0.5731) is below both yardsticks (0.6094, 0.6428) and no better than R007's by more than
0.006, which is Stage B's to address; and adding `nearest_player` costs it (-0.0161 raw, -0.0352 coarse), so Stage C's
evidence for the team carries `controlled_touch` alone. The public window is a correctness check only (valid,
deterministic, 19 possessions against 17).

### R009 — P05's dead-ball evidence: the 24-config grid and the choice of its source (2026-10-25, Tian)
Proposal: P05 · Commit: `5e1d4ef` (the two trained artefacts, the 24 configs
`pipeline/benchmark/configs/p45/p45b_*.toml` and the code they run on; the grid and the control were run from a
detached checkout of it, and the runner's header says `5e1d4ef8de6b`) · Command: six processes at once, three shares
of 8 configs (one source each: `hgb`, `logit`, `rules`), each share run once per truth, all with
`--profile tracking --seeds 3 --no-write`:
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml <8 configs> --profile tracking --seeds 3 --truth raw --no-write`
and the same with `nearest_w125.toml` first, `nearest_w50.toml` second and `--truth coarse`. The control is the same two
yardsticks and `p45b_hgb_w100_bp0` with `--profile context`, both truths. Public window:
`python pipeline/run.py pipeline/benchmark/configs/p45/p45b_hgb_w100_bp0.toml --score`
Public (a correctness check only, T01): macroF1 0.6482 · spellF1 0.3024 · alive/dead F1 0.9216 · 24 possessions against
the truth's 17, mean 11.8 s against its 14.2 s. The file is valid (199,319 rows), sha256
`e79647f1651ff93223c82cc59617a35c7aae38f37de449d789e396e746fcac25`. Over all the graded frames of the match it labels H
36.2%, A 28.1% and D 35.7% and finds 425 possessions (mean 9.6 s). Run twice, it gives the same sha256 both times.
`--score` reports `dead_ball_model` (the `hgb` source, scored by its `full` model) agreeing with the public GT's
alive/dead on 86.6% of 7,500 frames (the judged frames, where it has an opinion on all of them) and `nearest_player`
agreeing on home on 77.8% of 5,640 frames. For comparison R008's `ball_presence` agreed on alive on 81.8% of the same
7,500 frames. These public figures are in-sample for `nearest_player`'s 0.78, which is A01's measurement on this window,
as in R007 and R008. No bootstrap was run, because no public-window gap is claimed. Against R007's chosen config on the
same window (0.6393, 0.3500, 23 possessions) and `baseline_smooth50` (0.6207, 0.3256, 26 possessions) the spellF1 is
lower and the macroF1 higher, which says nothing either way (T01: one possession moves spellF1 about 0.06 here).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4 graded
minutes per seed. The chosen config, `p45b_hgb_w100_bp0` (`nearest_player` 0.78 and `dead_ball_model` with the `hgb`
artefact at weight 1.0, no `ball_presence`, hmm 8 / 32 / 75 / 1): pooled spellF1 0.3503 raw (mean per match 0.3521
[0.3322, 0.3720]) and 0.4150 coarse (0.4155 [0.3985, 0.4325]). Against `nearest_w50` under the raw truth: pooled Δ
+0.1486, mean per match +0.1498 [+0.1353, +0.1643], sign-flip p 0.0156, 7 wins and 0 losses. Against `nearest_w125`
under the coarse truth: pooled Δ +0.1924, mean +0.1926 [+0.1700, +0.2152], p 0.0156, 7 wins and 0 losses. **Stage B
alone meets both halves of P06's bar by a wide margin**; the bar is Stage C's to apply to the combined config.
Changed vs the previous best: the benchmark's best before this entry were R008's `p45a_r20_v6_k03_h55` (0.2345 raw,
0.2433 coarse), R007's chosen decoder on `nearest_player` (0.2053 raw, 0.1967 coarse) and the yardsticks (R004c). Stage
B replaces `ball_presence` by the P05 source at the same decoder (the weight-1.0 configs with `bp1` keep it too), so the
P05 gain at the same decoder is +0.1450 raw (0.3503 against 0.2053) and +0.2183 coarse (0.4150 against 0.1967), pooled;
R007's per-match scores are not in this entry's logs, so no interval is given for it.

**Read this before the numbers: the benchmark overstates P05 for BAR-ATH (T38).** On the benchmark the chosen source's
alive/dead F1 is 0.9725 (the yardsticks' 0.6094 and 0.6428) and its spellF1 reaches 95.2% raw and 94.0% coarse of
A10's perfect-`D` ceiling (0.3503 against 0.3678 raw, 0.4150 against 0.4417 coarse). That is not a result about our
match. The trees split on where the ball is (beyond a line, next or last seen beyond a line or at a restart spot) and
on what the players do, and DFL tracks the ball while it is out of play, so those features are visible during DFL's
stoppages far more often than on our match. The difference is not how often the ball is missing: on `tracking` it is
missing on 43.2% of the dead frames (visible on 56.8%, R005), more than the public window's 33.6% (A01). It is where
the visible ball is. Among the at-risk rows of R005's diagnosis (rows where a gap could begin; DFL's 7 matches at seed
0 against our match's own fit, A08), the ball is beyond a line on 17.68% of DFL's against 4.07% of ours, and within 2
m of a line on 25.23% against 9.01%: our tracker loses the ball near the lines, and DFL's keeps it there. The
`tracking` profile transplants our ball gaps, which are blind to play (R003); `context` places them by our hazard but
still keeps DFL's out-of-play ball tracking (R005), so it does not remove this either. **The benchmark therefore
overstates P05 for BAR-ATH by an amount it cannot measure.** The public window's alive agreement (86.6% of 7,500
frames for the chosen source, and an alive/dead F1 of 0.9216) is the only look at our match, at the level of
correctness, and its 5 minutes are the best-covered block of the match (A01). A label-free sign of the shift is the
chosen config's predicted `D` share over the graded frames of the whole match: 35.7%, against 42.3% for R007's chosen
config and 41.8% for R008's on the same match, and 44.6% for the chosen config on the benchmark (whose truth is
44.8%). A source that sees the dead ball on the benchmark finds less of it on our match; how much `D` our match has is
not known beyond the public window.

**Checks on the runs.** All six grid logs and both control logs end `EXIT=0`. Each grid log lists the two references
first (raw: `nearest_w50`, then `nearest_w125`; coarse the reverse) and 8 of the 24 configs; the three shares are
disjoint and together are the 24, none missing or repeated (process 1 the 8 `hgb` configs, 2 the `logit`, 3 the
`rules`). The references are the same in all six logs to the last printed digit (`seconds` aside) and are R004c's:
`nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse. On `context` they are R005's (0.2193 raw `nearest_w50`,
0.2443 coarse `nearest_w125`). The runner's `seconds`, across the six grid logs: 8.1 to 9.4 s for the yardsticks and
24.0 to 27.6 s for the grid's configs; the first config of every process took 160.3 and 162.4 s (`hgb`), 109.8 and
110.4 s (`logit`) and 102.0 and 101.8 s (`rules`), because it carries the one-off kinematics and features pass (the
memo then serves the rest of that process). No grid was shrunk: the 60 s per-config budget applies to steady-state
configs, and a process's first config carries the one-off pass (it guards runtime only; no result depends on it).

**What the sources are** (`dead_ball_model.py` and `train_dead_ball.py`; all trained on `tracking` v3, seeds 0 to 2,
every fifth graded frame, the label 1 where the truth is not `D`). `rules` is four fixed triggers (ball more than 0.5 m
outside; ball still 2 s at a restart spot, a line or a goal area; 80% of the players slow over +-4 s; the kick-off
set-up) and a value for no trigger, each with the value the next table gives. `logit` is a ridge-penalised logistic
regression on 30 play features, and `hgb` is scikit-learn's histogram gradient boosting (learning rate 0.05, 300
iterations, 15 leaves, minimum 1,000 per leaf, L2 1.0, monotone in the ball's distance outside and the slow share, no
early stopping, no class weights, **fixed and not tuned**). Both are corrected for the training prior, clipped to
[0.001, 0.999] and read in each benchmark match by the model fitted without it (`folds`); BAR-ATH is read by `full`. The
weight is the emission exponent on the logit.

**The rule values (`train_dead_ball.py --rules`).** The value of each trigger is P(t | alive) / (P(t | alive) + P(t |
dead)), rounded towards 0.5 to two decimals, counted on the graded frames of the seven matches, `tracking` v3, seeds 0
to 2: 1,659,315 alive and 1,348,617 dead. The counts are an exact recount (the log's four-decimal print cannot audit
`alive_kickoff`). **The values are measured on all seven matches, the same matches the grid is scored on, which is a
small optimism**; they are fixed rules and were never tuned towards a score.

| rule | alive frames | dead frames | P(t\|alive)/(P(t\|alive)+P(t\|dead)) | value |
|---|---|---|---|---|
| out | 1,925 | 267,425 | 0.0058 | 0.01 |
| still | 770 | 343,905 | 0.0018 | 0.01 |
| slow | 97,210 | 1,073,205 | 0.0686 | 0.07 |
| kickoff | 888 | 13,181 | 0.0519 | 0.06 |
| none | 1,559,771 | 168,252 | 0.8828 | 0.88 |

**Out-of-fold diagnostics of the two models (the trainer's own print).** Per held-out match, the model fitted on the
other six, scored on every fifth graded frame of its three seeds (rows are the match's rows over the three seeds); AUC
is of alive against dead.

| held out | rows | `hgb` log-loss | `logit` log-loss | `hgb` Brier | `logit` Brier | `hgb` AUC | `logit` AUC |
|---|---|---|---|---|---|---|---|
| J03WMX | 87,582 | 0.0721 | 0.1328 | 0.0207 | 0.0385 | 0.9966 | 0.9887 |
| J03WN1 | 84,939 | 0.0911 | 0.1578 | 0.0270 | 0.0467 | 0.9953 | 0.9861 |
| J03WOH | 82,329 | 0.0832 | 0.1485 | 0.0247 | 0.0415 | 0.9959 | 0.9872 |
| J03WOY | 85,524 | 0.0870 | 0.1584 | 0.0241 | 0.0443 | 0.9950 | 0.9847 |
| J03WPY | 87,729 | 0.0651 | 0.1340 | 0.0184 | 0.0362 | 0.9973 | 0.9891 |
| J03WQQ | 85,410 | 0.1028 | 0.1693 | 0.0287 | 0.0468 | 0.9932 | 0.9826 |
| J03WR9 | 88,089 | 0.0855 | 0.1677 | 0.0244 | 0.0456 | 0.9953 | 0.9842 |

`hgb` is better on log-loss, Brier and AUC in all 7 held-out matches. The row-weighted mean log-loss is 0.0837 against
0.1526 (a computation from the four-decimal rows). The reliability table, pooled over the held-out matches (counts and
means only):

| predicted bin | `hgb` rows | `hgb` predicted | `hgb` observed | `logit` rows | `logit` predicted | `logit` observed |
|---|---|---|---|---|---|---|
| 0.0-0.1 | 243,650 | 0.0074 | 0.0073 | 215,035 | 0.0152 | 0.0109 |
| 0.1-0.2 | 9,487 | 0.1428 | 0.1598 | 21,455 | 0.1429 | 0.1335 |
| 0.2-0.3 | 5,421 | 0.2473 | 0.2773 | 12,776 | 0.2472 | 0.2527 |
| 0.3-0.4 | 4,417 | 0.3491 | 0.3724 | 10,017 | 0.3492 | 0.3568 |
| 0.4-0.5 | 4,232 | 0.4503 | 0.4336 | 9,001 | 0.4494 | 0.4829 |
| 0.5-0.6 | 4,368 | 0.5525 | 0.5602 | 9,135 | 0.5510 | 0.6011 |
| 0.6-0.7 | 5,202 | 0.6523 | 0.6415 | 10,811 | 0.6514 | 0.6899 |
| 0.7-0.8 | 7,081 | 0.7531 | 0.7376 | 14,357 | 0.7525 | 0.7841 |
| 0.8-0.9 | 13,815 | 0.8573 | 0.8481 | 24,803 | 0.8560 | 0.8728 |
| 0.9-1.0 | 303,929 | 0.9905 | 0.9899 | 274,212 | 0.9859 | 0.9834 |

Both are calibrated at the ends, where most rows are (547,579 of 601,602 for `hgb`). `hgb` is within 0.001 in its two
extreme bins and its misses elsewhere are small: it predicts below the observed rate in 0.1-0.4 (by 0.017 to 0.030) and
in 0.5-0.6 (by 0.008), and above it in 0.4-0.5 (by 0.017) and in 0.6-0.9 (by 0.009 to 0.016). `logit` predicts below
the observed rate from 0.4 to 0.9 (by 0.017 to 0.050) and above it in the lowest bin (0.0152 predicted against 0.0109)
and the highest (0.9859 against 0.9834). A read of the exported trees also found that `hgb`'s top splits are
`next_outside_m`, `outside_player_m`, `next_spot_m`, `last_outside_m` and the past and future player speeds; it never
splits on `ball_missing` (as expected, when the gaps are blind to play); and the finfo.max stand-in for +inf thresholds
appears in none of the 8 models.

**The grid.** source in {`rules`, `logit`, `hgb`} x `weight` in {0.1, 0.25, 0.5, 1.0} x `ball_presence` in {off (`bp0`),
on (`bp1`)}: 24 configs `p45b_{src}_w{WWW}_bp{B}` (WWW the weight in thousandths, 010 to 100). Each holds
`nearest_player` at 0.78, the P05 source at that weight, `ball_presence` at its defaults if on, and `hmm` at (8, 32, 75,
1). The table gives pooled spellF1 and macroF1 under each truth, alive/dead F1 (the same under both: the predicted
labels do not depend on the truth), the predicted possessions per graded minute (the truth has 4.58 raw and 2.48
coarse) and the predicted `D` share (the truth's is 44.8%). The bold row is the chosen config; the two references come
first.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | 0.1842 | 0.4592 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` | 0.1665 | 0.2226 | 0.4407 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| `p45b_rules_w010_bp0` | 0.2420 | 0.2708 | 0.6064 | 0.6065 | 0.8155 | 2.77 | 23.0% |
| `p45b_rules_w025_bp0` | 0.2981 | 0.3423 | 0.7430 | 0.7429 | 0.9216 | 2.67 | 41.8% |
| `p45b_rules_w050_bp0` | 0.2963 | 0.3407 | 0.7406 | 0.7407 | 0.9217 | 2.77 | 42.4% |
| `p45b_rules_w100_bp0` | 0.2947 | 0.3377 | 0.7410 | 0.7416 | 0.9211 | 2.80 | 42.6% |
| `p45b_rules_w010_bp1` | 0.2090 | 0.2005 | 0.4653 | 0.4641 | 0.5750 | 3.04 | 43.6% |
| `p45b_rules_w025_bp1` | 0.2590 | 0.2605 | 0.5953 | 0.5940 | 0.6694 | 2.61 | 60.0% |
| `p45b_rules_w050_bp1` | 0.2729 | 0.2792 | 0.6387 | 0.6372 | 0.7144 | 2.55 | 66.2% |
| `p45b_rules_w100_bp1` | 0.2956 | 0.3392 | 0.7407 | 0.7414 | 0.9219 | 2.78 | 42.7% |
| `p45b_logit_w010_bp0` | 0.2916 | 0.3341 | 0.7469 | 0.7465 | 0.9218 | 2.54 | 41.2% |
| `p45b_logit_w025_bp0` | 0.3184 | 0.3691 | 0.7706 | 0.7707 | 0.9456 | 2.56 | 44.2% |
| `p45b_logit_w050_bp0` | 0.3248 | 0.3766 | 0.7736 | 0.7739 | 0.9489 | 2.60 | 44.7% |
| `p45b_logit_w100_bp0` | 0.3255 | 0.3796 | 0.7738 | 0.7741 | 0.9498 | 2.63 | 44.9% |
| `p45b_logit_w010_bp1` | 0.2160 | 0.2086 | 0.4925 | 0.4911 | 0.5950 | 2.88 | 46.2% |
| `p45b_logit_w025_bp1` | 0.2921 | 0.3072 | 0.6725 | 0.6713 | 0.7679 | 2.60 | 62.7% |
| `p45b_logit_w050_bp1` | 0.3151 | 0.3519 | 0.7354 | 0.7353 | 0.8879 | 2.66 | 53.3% |
| `p45b_logit_w100_bp1` | 0.3248 | 0.3721 | 0.7615 | 0.7620 | 0.9295 | 2.69 | 48.9% |
| `p45b_hgb_w010_bp0` | 0.3091 | 0.3581 | 0.7661 | 0.7656 | 0.9441 | 2.54 | 42.3% |
| `p45b_hgb_w025_bp0` | 0.3445 | 0.4052 | 0.7911 | 0.7914 | 0.9694 | 2.54 | 44.3% |
| `p45b_hgb_w050_bp0` | 0.3477 | 0.4115 | 0.7928 | 0.7935 | 0.9722 | 2.55 | 44.5% |
| **`p45b_hgb_w100_bp0`** | **0.3503** | **0.4150** | **0.7935** | **0.7948** | **0.9725** | **2.59** | **44.6%** |
| `p45b_hgb_w010_bp1` | 0.2174 | 0.2102 | 0.5000 | 0.4987 | 0.6009 | 2.85 | 47.0% |
| `p45b_hgb_w025_bp1` | 0.3166 | 0.3368 | 0.6878 | 0.6862 | 0.7847 | 2.49 | 63.4% |
| `p45b_hgb_w050_bp1` | 0.3466 | 0.3987 | 0.7690 | 0.7698 | 0.9280 | 2.54 | 51.5% |
| `p45b_hgb_w100_bp1` | 0.3529 | 0.4121 | 0.7900 | 0.7908 | 0.9629 | 2.60 | 47.3% |

**The choice, by the pre-declared rule.** Objective: pooled spellF1 under the raw truth. The best raw config is
`p45b_hgb_w100_bp1` at 0.3529. The tie band is 0.005, so the tied set is raw >= 0.3479. Two configs are in it:
`p45b_hgb_w100_bp1` (0.3529 raw, 0.4121 coarse) and `p45b_hgb_w100_bp0` (0.3503 raw, 0.4150 coarse). The next,
`hgb_w050_bp0`, is at 0.3477, 0.0002 below the band's edge and 0.0052 below the best; the best non-`hgb` config is
`logit_w100_bp0` at 0.3255, 0.0274 below. Among the tied, the best coarse score is `hgb_w100_bp0` (0.4150 against
0.4121, by 0.0029), so the tie order (the simpler source, then `ball_presence` off, then the smaller weight) is not
reached: the two share the source and the weight, and `ball_presence` off would have chosen the same. **Chosen:
`p45b_hgb_w100_bp0`**, 0.0026 below the raw best.
- **The plateau** (within 0.005 of the best raw): those two configs, both `hgb` at weight 1.0, with and without
  `ball_presence`. It is narrow at the top, but the family is flat: `hgb` without `ball_presence` scores 0.3445 to
  0.3503 at the three weights from 0.25 up, within 0.0084 of the best.
- **The raw-only winner** is `p45b_hgb_w100_bp1` (0.3529); **the coarse-only winner** is `p45b_hgb_w100_bp0` (0.4150),
  which is the chosen config, so the coarse truth does not disagree with the choice and the raw winner is the chosen
  config plus `ball_presence`, 0.0026 higher.
- **Each source's best weight** (raw spellF1; coarse in brackets), without `ball_presence`: `hgb` 1.0 (0.3503, 0.4150),
  `logit` 1.0 (0.3255, 0.3796), `rules` 0.25 (0.2981, 0.3423; 0.5 and 1.0 are 0.0018 and 0.0034 lower, so the rules'
  weight is flat above 0.25). With `ball_presence`: `hgb` 1.0 (0.3529, 0.4121), `logit` 1.0 (0.3248, 0.3721), `rules`
  1.0 (0.2956, 0.3392).
- **Does `ball_presence` still help? There is no evidence that it does.** Adding it to a P05 source changes raw
  spellF1 as below (coarse in brackets). Below weight 1.0 it costs, by up to 0.0917 raw and 0.1479 coarse at weight 0.1.
  That is a reading, not a test: `ball_presence` is a poor `D` detector on this benchmark (R007's chosen config, which
  has it, has an alive/dead F1 of 0.5676), and a weak P05 source cannot outvote it.
- **At weight 1.0 the change is inside the noise:** +0.0026 raw for `hgb` (inside the tie band) and -0.0029 coarse;
  +0.0009 and +0.0015 for `rules`; -0.0007 and -0.0075 for `logit`. For `hgb`, the per-match paired difference (`bp1`
  minus `bp0`) is +0.0024 [-0.0006, +0.0053] raw, sign-flip p 0.1719, 6 wins and 1 loss, and -0.0031 [-0.0078, +0.0016]
  coarse, p 0.2500, 3 wins and 4 losses. So the tie order's `ball_presence` off is the simpler answer, and the one the
  coarse truth prefers for `hgb` and `logit`.

| source | w 0.1 | w 0.25 | w 0.5 | w 1.0 |
|---|---|---|---|---|
| `rules` | -0.0330 (-0.0703) | -0.0391 (-0.0818) | -0.0234 (-0.0615) | +0.0009 (+0.0015) |
| `logit` | -0.0756 (-0.1255) | -0.0263 (-0.0619) | -0.0097 (-0.0247) | -0.0007 (-0.0075) |
| `hgb` | -0.0917 (-0.1479) | -0.0279 (-0.0684) | -0.0011 (-0.0128) | +0.0026 (-0.0029) |

- **At the edges.** The chosen config sits on the grid's upper edge in weight (1.0), for `hgb` and for `logit`, and its
  score rises with the weight in both (`hgb` 0.3091, 0.3445, 0.3477, 0.3503). **Per the protocol nothing is added after
  the results**: a weight above 1.0 (an emission exponent above the calibrated probability) was not tried, and the entry
  does not claim it would not score higher. The gain flattens: from weight 0.5 to 1.0 `hgb` gains +0.0026 raw and
  `logit` +0.0007.
- **Source order.** At every weight without `ball_presence` the order is `hgb` > `logit` > `rules` under both truths:
  at weight 1.0, 0.3503 > 0.3255 > 0.2947 raw and 0.4150 > 0.3796 > 0.3377 coarse. The models earn their keep over the
  rules, and the trees over the regression.

**Stage B2's trigger, evaluated.** Declared in the protocol before any run; it fires if either holds.
- **(a)** The best `hgb` config's pooled raw spellF1 is 0.3529 (`hgb_w100_bp1`) and the best `logit` config's is 0.3255
  (`logit_w100_bp0`): the gap is +0.0274, against the 0.005 band. `hgb` is more than 0.005 above `logit`, so (a) does
  not hold. (Like with like, `hgb_w100_bp0` against `logit_w100_bp0` is +0.0248, and with `ball_presence` +0.0281.)
- **(b)** `hgb`'s out-of-fold log-loss is worse than `logistic`'s on 0 of the 7 held-out matches (it is better on all
  7), against the trigger's 4 or more. So (b) does not hold.

**Neither holds: the literature's settings stand, and Stage B2 does not run.** `pipeline/models/dead_ball_hgb.json`
(3,585,829 bytes, sha256 `666d8ee534c6ebf9dd5393e60f2d5f9520376142ba63b6aec7afaa0d00506497`) is the `hgb` the choice
uses, fixed as designed; `dead_ball_logistic.json` (23,503 bytes, sha256
`079cd23ca7ec76074f162a08e4758d1a1c250398720ed9e0358e5c5c3092f398`) is the reference. Both were trained at `5e1d4ef`'s
code and `--verify` exits 0 for both.
**Δ for the top three under each truth** (spellF1; pooled Δ, mean per match with its 90% t-interval, sign-flip p, wins
in 7; against `nearest_w50` raw and `nearest_w125` coarse).

| truth | config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins |
|---|---|---|---|---|---|---|
| raw | `hgb_w100_bp1` | 0.3529 | +0.1511 | +0.1522 [+0.1378, +0.1666] | 0.0156 | 7 |
| raw | `hgb_w100_bp0` (chosen) | 0.3503 | +0.1486 | +0.1498 [+0.1353, +0.1643] | 0.0156 | 7 |
| raw | `hgb_w050_bp0` | 0.3477 | +0.1459 | +0.1472 [+0.1320, +0.1624] | 0.0156 | 7 |
| coarse | `hgb_w100_bp0` (chosen) | 0.4150 | +0.1924 | +0.1926 [+0.1700, +0.2152] | 0.0156 | 7 |
| coarse | `hgb_w100_bp1` | 0.4121 | +0.1894 | +0.1896 [+0.1657, +0.2135] | 0.0156 | 7 |
| coarse | `hgb_w050_bp0` | 0.4115 | +0.1888 | +0.1890 [+0.1669, +0.2111] | 0.0156 | 7 |

The gap between the two leaders (0.0026 raw, 0.0029 coarse) is inside the noise: the paired differences between them
are in the `ball_presence` item above (both intervals contain 0), the benchmark's own per-match interval half-widths
for these configs are about 0.020 raw and 0.017 coarse, and the full-match floor of our single match (about 0.04, T07)
is larger still. The choice between them is the rule's, not a finding. Every Δ in the table is more than 0.14 and 7 wins
in 7, so the question this benchmark can answer is not whether P05 beats the yardsticks but how much of it transfers,
which the caveat above
says it cannot measure.

**Alive/dead and possessions.** Alive/dead F1 is 0.9725 for the chosen config against 0.6094 (`nearest_w50`), 0.6428
(`nearest_w125`) and 0.5676 (R007's chosen config): +0.3631, +0.3297 and +0.4049. Predicted possessions are 2.59 per
graded minute against 4.58 under the raw truth and 2.48 under the coarse truth (R008's chosen config 2.42); the
predicted `D` share is 44.6% against 44.8% (the yardsticks 33.4% and 23.3%). So on the benchmark the P05 source fixes
the `D` problem that R007 and R008 named and A10 sized (the caveat above is why that is not a claim about our match):
the chosen config is 0.0175 raw and 0.0267 coarse below the oracle-alive variant (0.3678, 0.4417; the oracle there is
added to R007's evidence, so the comparison is of like decoders and not of like evidence). The count of possessions is
still below the raw truth's 4.58, and the coarse truth is the one it matches.

**Leave-one-match-out estimate of the selection.** As R008 did, from the per-match spellF1 the runner prints (four
decimals): for each match, the selection is repeated on the other six matches and the chosen config's score on the
held-out match is compared with the reference's on it; the seven differences give the mean Δ with its 90% t-interval,
the exact sign-flip p over the 2^7 sign patterns, and the wins. The references are not candidates.
- **The raw objective alone** (the config of the 24 with the highest mean per-match raw spellF1 on the six, no tie
  band): `hgb_w100_bp1` is picked for all seven matches. Scored under the raw truth against `nearest_w50`: mean Δ
  +0.1522 [+0.1378, +0.1666], p 0.0156, 7 wins and 0 losses (per match +0.1277, +0.1227, +0.1693, +0.1647, +0.1682,
  +0.1630, +0.1498). Scored under the coarse truth against `nearest_w125`: mean Δ +0.1896 [+0.1657, +0.2135], p 0.0156,
  7 wins and 0 losses.
- **The whole rule** (on each fold: the tied set is the configs within 0.005 of the best six-match raw mean, then the
  best six-match coarse mean among them, then the tie order): `hgb_w100_bp0` is picked in all seven folds, so the
  held-out estimate is the chosen config's own: +0.1498 [+0.1353, +0.1643] raw (per match +0.1210, +0.1216, +0.1652,
  +0.1597, +0.1638, +0.1623, +0.1551) and +0.1926 [+0.1700, +0.2152] coarse, p 0.0156, 7 wins and 0 losses under each.

The selection is stable (one config in every fold under each rule, and the same source and weight under both), so the
choice is not what produces the gap. The estimate does not cover the rule values, which use all seven matches, nor the
two models' training, which holds out the scored match only through `folds`, and seven matches from two competitions
(five of them Fortuna Düsseldorf home games) are not seven independent draws.

**Controls** (sensitivity only, T38; `context` v1 fails two of its gates, R005; run from a detached checkout at
`5e1d4ef`, `--profile context`, `--seeds 3 --no-write`, the references first as in the grid; both logs end `EXIT=0`,
and the references score on `context` exactly as in R005: 0.2193 raw and 0.2057 coarse for `nearest_w50`, 0.1774 raw and
0.2443 coarse for `nearest_w125`, with R005's macroF1 and alive/dead F1).
- **The chosen config on `context`.** Pooled spellF1 0.3551 raw (mean per match 0.3567 [0.3360, 0.3774]) and 0.4138
  coarse (0.4138 [0.3947, 0.4328]); macroF1 0.8016 raw and 0.8026 coarse; alive/dead F1 0.9698 (`nearest_w50` 0.6575,
  `nearest_w125` 0.6711); 2.69 possessions per graded minute; `D` share 44.8%. Against `nearest_w50` raw: pooled Δ
  +0.1359, mean +0.1370 [+0.1230, +0.1510], p 0.0156, 7 wins in 7. Against `nearest_w125` coarse: pooled Δ +0.1695, mean
  +0.1691 [+0.1441, +0.1940], p 0.0156, 7 wins in 7. At the same decoder, R007's chosen config on `context`
  scored 0.2332 raw and 0.2261 coarse (R007), so the P05 gain there is +0.1219 raw and +0.1877 coarse, pooled.
- **The four profile x truth cells agree.** The chosen config is above its truth's strongest yardstick in all four, 7
  matches of 7 each, p 0.0156: `tracking` raw +0.1486, `tracking` coarse +0.1924, `context` raw +0.1359, `context`
  coarse +0.1695. There is no warning to write (T38). The agreement is not evidence that the gain transfers: `context`
  places the gaps by our hazard but keeps DFL's out-of-play ball tracking, so it shares the caveat above, and neither
  profile is a second validation.

Runtime of the control: the chosen config's first config in each process took 119.7 and 119.9 s (the one-off pass), the
yardsticks 5.8 to 6.4 s.

Verdict: **`p45b_hgb_w100_bp0` (`nearest_player` 0.78 and `dead_ball_model` with `pipeline/models/dead_ball_hgb.json`
at weight 1.0, no `ball_presence`, and the hmm (8, 32, 75, 1)) is P05's chosen source**, by the pre-declared rule over
the 24 configs; it is not the submission config, and `main.toml` and `p06_hmm.toml` are unchanged. Stage B2 does not
run: `hgb` leads `logit` by 0.0274 raw spellF1 (the trigger's band is 0.005) and has the lower out-of-fold log-loss on
7 of 7 held-out matches, so the literature's settings stand. For P05 alone, on `tracking` v3, it clears both halves of
the bar P06 set (raw +0.1486 over `nearest_w50`, p 0.0156, 7 wins in 7; coarse +0.1924 over `nearest_w125`, p 0.0156, 7
wins in 7), the leave-one-match-out estimates agree (raw objective alone +0.1522 raw and +0.1896 coarse, the whole rule
+0.1498 and +0.1926, each 7 wins in 7), and the `context` sensitivity profile is above both yardsticks by less but by
a wide margin (+0.1359, +0.1695). Whether the combined config meets the bar is Stage C's to apply (R010), not this
entry's. The limits: **the size of these gains rests on DFL's out-of-play ball tracking and does not transfer in full
to BAR-ATH, by an amount the benchmark cannot measure (T38)**; the choice is on the grid's upper edge in weight and by
the protocol none is extended; the rule values are measurements on the seven matches the grid is scored on;
there is no evidence that `ball_presence` helps (it costs up to 0.0917 raw below weight 1.0), and Stage C's alive
evidence is the P05 source alone; and the public window is a correctness check only (valid, deterministic with the same
sha256 twice, 24 possessions against 17, alive agreement 86.6%).

### R010 — P04 and P05 together: the 72-config grid, the choice and the bar (2026-10-25, Tian)
Proposal: P04, P05 · Commit: `7cc99f3` (Stage C's code and its 72 configs `pipeline/benchmark/configs/p45/p45c_*.toml`;
the grid was run from a detached checkout of it, the runner's header says `7cc99f3db152`); the controls were run from
a detached checkout of `15580c1` (the chosen config and the two ablations; header `15580c15235b`); the public window and
`main.toml` were run at `f276ae7` (the new `main.toml` was not yet committed, so the manifest says `+dirty`; it was
committed at `dd25113`) · Command: six processes at once, three shares of 24 configs (the t = 4, 8 and 16
rows of the grid), each share run once per truth, all with `--profile tracking --seeds 3 --no-write`:
`python -u pipeline/benchmark/run.py pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml <24 configs> --profile tracking --seeds 3 --truth raw --no-write`
and the same with `nearest_w125.toml` first, `nearest_w50.toml` second and `--truth coarse`. The controls are the same
two yardsticks, the chosen config and the two ablations with `--profile tracking` (both truths), and the two yardsticks
and the chosen config with `--profile context` (both truths). Public window:
`python pipeline/run.py pipeline/configs/p45_hmm.toml --score`
Public (a correctness check only, T01): macroF1 0.6996 · spellF1 0.4686 · alive/dead F1 0.9222 · 18 possessions against
the truth's 17, mean 15.7 s against its 14.2 s. The file is valid (199,319 rows), sha256
`a706e7f711b2d3faacc2f071b6e28963ed8e2f2326a7880e6d0776cce4295f1f`. Over all the graded frames of the match it labels H
37.5%, A 26.5% and D 36.0% (X 0.0%) and finds 450 possessions (mean 9.0 s). Run twice, it gives the same sha256 both
times. `--score` reports `controlled_touch` agreeing with the public GT on home on 87.2% of 5,489 frames (the judged
frames where it has an opinion; the same figure as R008's, for the same rule and frames) and `dead_ball_model` agreeing
on alive on 86.6% of 7,500 frames (the same as R009's, for the same model). `controlled_touch` has an opinion on home on
59.2% of the graded frames and none on alive; `dead_ball_model` has an opinion on alive on all of them and none on
home. No bootstrap was run, because no public-window gap is claimed. The two agreement lines are measurements on the
public window, and they are not inputs: no constant of the chosen config comes from that window (the `controlled_touch`
confidence 0.91 is touch_stats.py's floor over the seven benchmark matches, the trees were trained on those matches, and
the decoder was chosen on them; this config has no `nearest_player`, and so none of A01's 0.78). Against R009's chosen
config on the same window (0.6482, 0.3024, 24 possessions), R008's (0.6532, 0.3889, 19) and `baseline_smooth50`
(0.6207, 0.3256, 26 possessions) both F1 scores are higher, while the possession count (18) is lower than all three
(24, 19 and 26; the truth's is 17). That says nothing either way (T01: one possession moves spellF1 about 0.06 here,
and the window ranks `smooth125` above `smooth50`, the reverse of the full match).
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), profile `tracking` version 3, 668.4 graded
minutes per seed. The chosen config, `p45c_t004_d032_mp025_md025` (`controlled_touch` 2.0 m / 6.0 m/s / 3 frames / hold
0.55 at confidence 0.91 and `dead_ball_model` with the `hgb` artefact at weight 1.0, no `nearest_player`, no
`ball_presence`, hmm 4 / 32 / 25 / 25): pooled spellF1 0.4217 raw (mean per match 0.4240 [0.3991, 0.4488]) and 0.4556
coarse (0.4569 [0.4337, 0.4802]); macroF1 0.8275 raw and 0.8288 coarse; alive/dead F1 0.9726; 2.74 possessions per
graded minute; `D` share 44.6% (the truth's 44.8%).

| truth | against | pooled Δ | mean per match Δ [90% t] | p | wins |
|---|---|---|---|---|---|
| raw | `nearest_w50` (the bar's) | +0.2199 | +0.2217 [+0.2019, +0.2415] | 0.0156 | 7 |
| raw | `nearest_w125` | +0.2552 | +0.2564 [+0.2396, +0.2732] | 0.0156 | 7 |
| coarse | `nearest_w125` (the bar's) | +0.2330 | +0.2341 [+0.2202, +0.2480] | 0.0156 | 7 |
| coarse | `nearest_w50` | +0.2714 | +0.2724 [+0.2537, +0.2910] | 0.0156 | 7 |

Changed vs the previous best: the file `main.toml` wrote until now was R001's, nearest player plus a +-50-frame majority
vote (`nearest_w50` on the benchmark: 0.2018 raw, 0.1842 coarse), public macroF1 0.6207, spellF1 0.3256, 26
possessions, sha256 `678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da` (R001 does not record the sha256;
this is the one the file in `evaluation/results/` had before R010 rewrote it). The benchmark's previous best was R009's
`p45b_hgb_w100_bp0`, 0.3503 raw and 0.4150 coarse; this config is +0.0714 and +0.0406 above it (pooled).

**The bar is met under both truths, by a wide margin.** The label-free check on BAR-ATH (below) also passes, so
`main.toml` is switched.

**Read this before the numbers: the benchmark overstates P05 for BAR-ATH (T38), and with it the combined gain.** R009
gives the reason and the measurements: DFL tracks the ball out of play and our tracker loses it near the lines (beyond a
line on 4.07% of our at-risk rows against DFL's 17.68%), so the trees see the dead ball far more often on the benchmark
than they will on our match, by an amount the benchmark cannot measure. Everything here that rests on the alive/dead F1
(0.9726 on the benchmark) inherits that, and so does the size of the Δs above. The sign of the raw one does not rest on
it, since the config without P05 still beats `nearest_w50` raw (+0.0539, below); the coarse half of the bar does
(without P05: -0.0056 against `nearest_w125`, p 0.1094, below), so the domain shift threatens exactly that half. The
public window's alive agreement (86.6% of 7,500 frames) and the predicted `D` share over the graded frames of our match,
**36.0%** against 44.6% on the benchmark, are the two looks at our match. The share is close to R009's alone (35.7%) and
below R007's 42.3% and R008's 41.8%, but not off the scale of what a football match has (DFL's seven matches have 40.7%
to 51.7%).

**The grid.** switch_team t in {4, 8, 16} x switch_dead d in {16, 32, 64} x `min_possession` in {25, 50, 75, 100}
frames x `min_dead` in {1, 25} frames: 72 configs `p45c_t{TTT}_d{DDD}_mp{MMM}_md{NNN}`. The evidence is fixed and the
union of the two stages' choices: `controlled_touch` (R008) for `home` and the `hgb` `dead_ball_model` (R009) for
`alive`; no `nearest_player`, no `ball_presence`. The table gives pooled spellF1 and macroF1 under each truth,
alive/dead F1 (the same under both: the predicted labels do not depend on the truth), the predicted possessions per
graded minute (the truth has 4.58 raw and 2.48 coarse) and the predicted `D` share (the truth's is 44.8%). The bold row
is the chosen config; the two references come first.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | 0.1842 | 0.4592 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` | 0.1665 | 0.2226 | 0.4407 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| `p45c_t004_d016_mp025_md001` | 0.4208 | 0.4534 | 0.8275 | 0.8288 | 0.9722 | 2.78 | 44.7% |
| `p45c_t004_d016_mp025_md025` | 0.4210 | 0.4536 | 0.8274 | 0.8288 | 0.9722 | 2.78 | 44.7% |
| `p45c_t004_d016_mp050_md001` | 0.4085 | 0.4735 | 0.8264 | 0.8280 | 0.9722 | 2.44 | 44.6% |
| `p45c_t004_d016_mp050_md025` | 0.4087 | 0.4737 | 0.8264 | 0.8280 | 0.9722 | 2.44 | 44.6% |
| `p45c_t004_d016_mp075_md001` | 0.3904 | 0.4878 | 0.8241 | 0.8260 | 0.9721 | 2.21 | 44.6% |
| `p45c_t004_d016_mp075_md025` | 0.3905 | 0.4880 | 0.8241 | 0.8260 | 0.9721 | 2.21 | 44.6% |
| `p45c_t004_d016_mp100_md001` | 0.3712 | 0.4956 | 0.8222 | 0.8256 | 0.9718 | 2.05 | 44.6% |
| `p45c_t004_d016_mp100_md025` | 0.3713 | 0.4958 | 0.8222 | 0.8256 | 0.9718 | 2.05 | 44.6% |
| `p45c_t004_d032_mp025_md001` | 0.4215 | 0.4554 | 0.8275 | 0.8288 | 0.9726 | 2.74 | 44.6% |
| **`p45c_t004_d032_mp025_md025`** | 0.4217 | 0.4556 | 0.8275 | 0.8288 | 0.9726 | 2.74 | 44.6% |
| `p45c_t004_d032_mp050_md001` | 0.4092 | 0.4759 | 0.8266 | 0.8282 | 0.9726 | 2.40 | 44.6% |
| `p45c_t004_d032_mp050_md025` | 0.4093 | 0.4761 | 0.8266 | 0.8282 | 0.9726 | 2.40 | 44.6% |
| `p45c_t004_d032_mp075_md001` | 0.3909 | 0.4904 | 0.8244 | 0.8262 | 0.9725 | 2.17 | 44.6% |
| `p45c_t004_d032_mp075_md025` | 0.3910 | 0.4906 | 0.8244 | 0.8263 | 0.9725 | 2.17 | 44.6% |
| `p45c_t004_d032_mp100_md001` | 0.3709 | 0.4965 | 0.8223 | 0.8257 | 0.9721 | 2.02 | 44.6% |
| `p45c_t004_d032_mp100_md025` | 0.3710 | 0.4966 | 0.8223 | 0.8257 | 0.9721 | 2.02 | 44.6% |
| `p45c_t004_d064_mp025_md001` | 0.4190 | 0.4535 | 0.8278 | 0.8288 | 0.9726 | 2.69 | 44.6% |
| `p45c_t004_d064_mp025_md025` | 0.4190 | 0.4535 | 0.8278 | 0.8288 | 0.9726 | 2.69 | 44.6% |
| `p45c_t004_d064_mp050_md001` | 0.4063 | 0.4737 | 0.8268 | 0.8281 | 0.9726 | 2.36 | 44.5% |
| `p45c_t004_d064_mp050_md025` | 0.4063 | 0.4737 | 0.8268 | 0.8281 | 0.9726 | 2.36 | 44.5% |
| `p45c_t004_d064_mp075_md001` | 0.3879 | 0.4879 | 0.8245 | 0.8262 | 0.9725 | 2.13 | 44.5% |
| `p45c_t004_d064_mp075_md025` | 0.3879 | 0.4879 | 0.8245 | 0.8262 | 0.9725 | 2.13 | 44.5% |
| `p45c_t004_d064_mp100_md001` | 0.3682 | 0.4945 | 0.8225 | 0.8257 | 0.9720 | 1.98 | 44.5% |
| `p45c_t004_d064_mp100_md025` | 0.3682 | 0.4945 | 0.8225 | 0.8257 | 0.9720 | 1.98 | 44.5% |
| `p45c_t008_d016_mp025_md001` | 0.4138 | 0.4691 | 0.8284 | 0.8302 | 0.9721 | 2.57 | 44.7% |
| `p45c_t008_d016_mp025_md025` | 0.4140 | 0.4694 | 0.8284 | 0.8302 | 0.9721 | 2.57 | 44.7% |
| `p45c_t008_d016_mp050_md001` | 0.4023 | 0.4828 | 0.8272 | 0.8290 | 0.9721 | 2.34 | 44.7% |
| `p45c_t008_d016_mp050_md025` | 0.4025 | 0.4830 | 0.8272 | 0.8290 | 0.9721 | 2.34 | 44.7% |
| `p45c_t008_d016_mp075_md001` | 0.3865 | 0.4946 | 0.8254 | 0.8279 | 0.9721 | 2.15 | 44.6% |
| `p45c_t008_d016_mp075_md025` | 0.3867 | 0.4946 | 0.8254 | 0.8279 | 0.9721 | 2.15 | 44.6% |
| `p45c_t008_d016_mp100_md001` | 0.3692 | 0.5008 | 0.8235 | 0.8274 | 0.9717 | 2.00 | 44.6% |
| `p45c_t008_d016_mp100_md025` | 0.3693 | 0.5010 | 0.8235 | 0.8274 | 0.9717 | 2.00 | 44.6% |
| `p45c_t008_d032_mp025_md001` | 0.4145 | 0.4717 | 0.8288 | 0.8305 | 0.9726 | 2.53 | 44.6% |
| `p45c_t008_d032_mp025_md025` | 0.4146 | 0.4719 | 0.8288 | 0.8306 | 0.9726 | 2.53 | 44.6% |
| `p45c_t008_d032_mp050_md001` | 0.4030 | 0.4854 | 0.8276 | 0.8294 | 0.9726 | 2.30 | 44.6% |
| `p45c_t008_d032_mp050_md025` | 0.4031 | 0.4856 | 0.8276 | 0.8294 | 0.9726 | 2.30 | 44.6% |
| `p45c_t008_d032_mp075_md001` | 0.3873 | 0.4973 | 0.8259 | 0.8283 | 0.9725 | 2.11 | 44.6% |
| `p45c_t008_d032_mp075_md025` | 0.3874 | 0.4974 | 0.8259 | 0.8283 | 0.9725 | 2.11 | 44.6% |
| `p45c_t008_d032_mp100_md001` | 0.3691 | 0.5020 | 0.8239 | 0.8277 | 0.9721 | 1.97 | 44.6% |
| `p45c_t008_d032_mp100_md025` | 0.3692 | 0.5021 | 0.8239 | 0.8277 | 0.9721 | 1.97 | 44.6% |
| `p45c_t008_d064_mp025_md001` | 0.4120 | 0.4697 | 0.8288 | 0.8303 | 0.9726 | 2.48 | 44.6% |
| `p45c_t008_d064_mp025_md025` | 0.4120 | 0.4697 | 0.8288 | 0.8303 | 0.9726 | 2.48 | 44.6% |
| `p45c_t008_d064_mp050_md001` | 0.4001 | 0.4830 | 0.8276 | 0.8291 | 0.9726 | 2.25 | 44.5% |
| `p45c_t008_d064_mp050_md025` | 0.4001 | 0.4830 | 0.8276 | 0.8291 | 0.9726 | 2.25 | 44.5% |
| `p45c_t008_d064_mp075_md001` | 0.3843 | 0.4945 | 0.8258 | 0.8280 | 0.9725 | 2.07 | 44.5% |
| `p45c_t008_d064_mp075_md025` | 0.3843 | 0.4945 | 0.8258 | 0.8280 | 0.9725 | 2.07 | 44.5% |
| `p45c_t008_d064_mp100_md001` | 0.3661 | 0.4997 | 0.8238 | 0.8274 | 0.9720 | 1.92 | 44.5% |
| `p45c_t008_d064_mp100_md025` | 0.3661 | 0.4997 | 0.8238 | 0.8274 | 0.9720 | 1.92 | 44.5% |
| `p45c_t016_d016_mp025_md001` | 0.3872 | 0.4963 | 0.8265 | 0.8314 | 0.9721 | 2.20 | 44.7% |
| `p45c_t016_d016_mp025_md025` | 0.3874 | 0.4966 | 0.8265 | 0.8314 | 0.9721 | 2.20 | 44.7% |
| `p45c_t016_d016_mp050_md001` | 0.3791 | 0.5008 | 0.8254 | 0.8305 | 0.9721 | 2.11 | 44.7% |
| `p45c_t016_d016_mp050_md025` | 0.3792 | 0.5011 | 0.8254 | 0.8305 | 0.9721 | 2.11 | 44.7% |
| `p45c_t016_d016_mp075_md001` | 0.3688 | 0.4998 | 0.8234 | 0.8283 | 0.9720 | 2.01 | 44.6% |
| `p45c_t016_d016_mp075_md025` | 0.3688 | 0.5001 | 0.8234 | 0.8283 | 0.9721 | 2.01 | 44.6% |
| `p45c_t016_d016_mp100_md001` | 0.3534 | 0.5038 | 0.8218 | 0.8277 | 0.9717 | 1.89 | 44.6% |
| `p45c_t016_d016_mp100_md025` | 0.3535 | 0.5040 | 0.8218 | 0.8277 | 0.9717 | 1.88 | 44.6% |
| `p45c_t016_d032_mp025_md001` | 0.3877 | 0.4992 | 0.8270 | 0.8320 | 0.9726 | 2.16 | 44.6% |
| `p45c_t016_d032_mp025_md025` | 0.3878 | 0.4994 | 0.8270 | 0.8320 | 0.9726 | 2.16 | 44.6% |
| `p45c_t016_d032_mp050_md001` | 0.3797 | 0.5039 | 0.8261 | 0.8312 | 0.9726 | 2.07 | 44.6% |
| `p45c_t016_d032_mp050_md025` | 0.3799 | 0.5041 | 0.8262 | 0.8312 | 0.9726 | 2.07 | 44.6% |
| `p45c_t016_d032_mp075_md001` | 0.3693 | 0.5031 | 0.8242 | 0.8290 | 0.9725 | 1.97 | 44.6% |
| `p45c_t016_d032_mp075_md025` | 0.3694 | 0.5032 | 0.8242 | 0.8290 | 0.9725 | 1.97 | 44.6% |
| `p45c_t016_d032_mp100_md001` | 0.3532 | 0.5047 | 0.8224 | 0.8282 | 0.9721 | 1.85 | 44.6% |
| `p45c_t016_d032_mp100_md025` | 0.3532 | 0.5049 | 0.8224 | 0.8282 | 0.9721 | 1.85 | 44.6% |
| `p45c_t016_d064_mp025_md001` | 0.3843 | 0.4975 | 0.8272 | 0.8320 | 0.9726 | 2.11 | 44.6% |
| `p45c_t016_d064_mp025_md025` | 0.3843 | 0.4975 | 0.8272 | 0.8320 | 0.9726 | 2.11 | 44.6% |
| `p45c_t016_d064_mp050_md001` | 0.3761 | 0.5015 | 0.8262 | 0.8310 | 0.9726 | 2.02 | 44.6% |
| `p45c_t016_d064_mp050_md025` | 0.3761 | 0.5015 | 0.8262 | 0.8310 | 0.9726 | 2.02 | 44.6% |
| `p45c_t016_d064_mp075_md001` | 0.3654 | 0.5001 | 0.8242 | 0.8288 | 0.9724 | 1.92 | 44.5% |
| `p45c_t016_d064_mp075_md025` | 0.3654 | 0.5001 | 0.8242 | 0.8288 | 0.9724 | 1.92 | 44.5% |
| `p45c_t016_d064_mp100_md001` | 0.3496 | 0.5028 | 0.8225 | 0.8281 | 0.9720 | 1.81 | 44.5% |
| `p45c_t016_d064_mp100_md025` | 0.3496 | 0.5028 | 0.8225 | 0.8281 | 0.9720 | 1.81 | 44.5% |

**The choice, by the pre-declared rule.** Objective: pooled spellF1 under the raw truth. The best raw config is the
chosen one itself, `p45c_t004_d032_mp025_md025`, at 0.4217. The tie band is 0.005, so the tied set is raw >= 0.4167.
Six configs are in it, all at switch_team 4 and min_possession 25 (switch_dead 16, 32 and 64, min_dead 1 and 25):

| config | raw spellF1 | coarse spellF1 |
|---|---|---|
| `p45c_t004_d032_mp025_md025` | 0.4217 | 0.4556 |
| `p45c_t004_d032_mp025_md001` | 0.4215 | 0.4554 |
| `p45c_t004_d016_mp025_md025` | 0.4210 | 0.4536 |
| `p45c_t004_d064_mp025_md001` | 0.4190 | 0.4535 |
| `p45c_t004_d064_mp025_md025` | 0.4190 | 0.4535 |
| `p45c_t004_d016_mp025_md001` | 0.4208 | 0.4534 |

The next config, `t008_d032_mp025_md025`, is at 0.4146, 0.0071 below the best. Among the tied, the best coarse score is
`p45c_t004_d032_mp025_md025` (0.4556, against 0.4554 for its `md001` twin, by 0.0002), so the tie order (larger
switch_team, then min_possession, then switch_dead, then min_dead) is not reached. **Chosen:
`p45c_t004_d032_mp025_md025`**, which is also the raw best.
- **The plateau** (within 0.005 of the best raw): those six configs. It is flat along switch_dead (0.4190 to 0.4217
  across 16, 32 and 64) and along min_dead (the two values differ by at most 0.0002 raw and 0.0003 coarse in every one
  of the 36 pairs), but narrow along the other two axes: switch_team 8 is 0.0071 below, and min_possession 50 is 0.0124
  below.
- **The raw-only winner** is the chosen config (0.4217); **the coarse-only winner** is `p45c_t016_d032_mp100_md025`
  (0.5049 coarse, 0.3532 raw). The truths disagree, as in R007 (T33): raw wants the shortest minimum possession and the
  smallest switch penalty of the grid, coarse the longest and the largest. By axis, the best raw and coarse scores at
  each value are: switch_team 4 (0.4217, 0.4966), 8 (0.4146, 0.5021), 16 (0.3878, 0.5049); min_possession 25 (0.4217,
  0.4994), 50 (0.4093, 0.5041), 75 (0.3910, 0.5032), 100 (0.3713, 0.5049). Picking by the coarse truth would cost 0.0685
  raw (0.3532 against 0.4217), and the rule's choice costs 0.0493 coarse (0.4556 against 0.5049). The raw truth is the
  objective and the coarse truth the warning: the possession length is not known (A06). The chosen config predicts 2.74
  possessions per graded minute against 4.58 under the raw truth and 2.48 under the coarse one; the coarse-only winner
  predicts 1.85.
- **At the edges.** The chosen config is at the grid's lower edge in switch_team (4) and in min_possession (25), at its
  upper edge in min_dead (25, where it makes no difference), and inside it in switch_dead (32). Raw spellF1 falls
  monotonically with both switch_team and min_possession from the edge. **Per the protocol nothing is added after the
  results**: a switch_team below 4 or a minimum possession below 25 frames (1 s) was not tried, and the entry does not
  claim it would not score higher on the raw truth, nor that it would be sound on the coarse one.

**Δ for the other leaders.** The coarse-only winner's raw Δ over `nearest_w50` is +0.1514 (0.3532 against 0.2018), so
it too clears the bar's raw half. No config of the 72 fails either half: the lowest pooled raw score is 0.3496 and the
lowest coarse 0.4534, against the yardsticks' 0.2018 and 0.2226.

**Controls: what each source adds at the chosen decoder** (hmm 4 / 32 / 25 / 25; `tracking` v3, 3 seeds; run from
`15580c1`; both logs end `EXIT=0`; the chosen config and the references score as in the grid, to the last digit, and per
match too). Each ablation changes one source: `p45cctl_no_p04` replaces `controlled_touch` by `nearest_player` at
confidence 0.78, and `p45cctl_no_p05` replaces `dead_ball_model` by `ball_presence` at its defaults (R007's `alive`
evidence).

| config | raw spellF1 | coarse spellF1 | raw macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|
| chosen | 0.4217 | 0.4556 | 0.8275 | 0.9726 | 2.74 | 44.6% |
| `p45cctl_no_p04` | 0.2796 | 0.2424 | 0.7796 | 0.9726 | 5.09 | 44.6% |
| `p45cctl_no_p05` | 0.2556 | 0.2170 | 0.4851 | 0.5749 | 3.05 | 43.0% |

Paired against the chosen config (chosen minus the ablation; mean per match [90% t], exact sign-flip p, wins in 7):
- **What P04 adds** (chosen vs `no_p04`): +0.1421 pooled raw, mean +0.1441 [+0.1150, +0.1732], p 0.0156, 7 wins;
  +0.2132 pooled coarse, mean +0.2142 [+0.1876, +0.2408], p 0.0156, 7 wins.
- **What P05 adds** (chosen vs `no_p05`): +0.1661 pooled raw, mean +0.1670 [+0.1571, +0.1770], p 0.0156, 7 wins;
  +0.2386 pooled coarse, mean +0.2393 [+0.2275, +0.2511], p 0.0156, 7 wins.
- **Each is needed.** Neither ablation clears the bar. Against `nearest_w50` raw, `no_p04` is +0.0778 (p 0.0156, 7
  wins; its predicted possessions are 5.09 per minute, `nearest_player`'s flicker) and `no_p05` +0.0539 (p 0.0156, 7
  wins). Against `nearest_w125` coarse, `no_p04` is +0.0198 (mean interval [-0.0026, +0.0425], p 0.1406, 5 wins and 2
  losses) and `no_p05` is -0.0056 (mean interval [-0.0105, +0.0001], p 0.1094, 1 win and 6 losses). So the raw half of
  the bar holds without either source and the coarse half fails without either: at this decoder each source alone is
  worth little under the coarse truth, and the two together a lot.

**Comparison with the earlier entries** (pooled spellF1, raw and coarse). The combined config scores 0.4217 and 0.4556.
- **R009, P05 alone** (R007's decoder): 0.3503 and 0.4150. The combined config is +0.0714 raw and +0.0406 coarse above.
- **R008, P04 alone** (the same decoder): 0.2345 and 0.2433: +0.1872 and +0.2123 above.
- **R007's chosen config** (`nearest_player` and `ball_presence`, the same decoder): 0.2053 and 0.1967: +0.2164 and
  +0.2589 above.
- **The sum of the two gains.** At R007's decoder P04 gains +0.0292 raw and +0.0466 coarse, and P05 +0.1450 and +0.2183;
  the sum, +0.1742 and +0.2649, would give 0.3795 and 0.4616, and the combined config at the same decoder (the grid's
  `p45c_t008_d032_mp075_md001`) scores 0.3873 and 0.4973. So the two gains add, and a little more (pooled, no interval:
  +0.0078 raw, +0.0357 coarse). The re-tuned decoder then adds +0.0344 raw (0.4217 against 0.3873) and costs 0.0417
  coarse (0.4556 against 0.4973): the re-tuning follows the raw objective, which is what T33 is about.
- **A10's ceilings** (at R007's decoder, with `nearest_player` for the team): a perfect `D` would reach 0.3678 raw and
  0.4417 coarse, and both parts perfect everywhere 0.7627 and 0.9226. The chosen config is +0.0539 raw and +0.0139
  coarse above the first (its team evidence is better than `nearest_player`, and its decoder is tuned to it), and is
  55.3% raw and 49.4% coarse of the second.

**Leave-one-match-out estimate of the selection.** As R008 and R009 did, from the per-match spellF1 the runner prints
(four decimals): for each match, the selection is repeated on the other six matches and the chosen config's score on the
held-out match is compared with the yardstick's on it (`nearest_w50` raw, `nearest_w125` coarse); the seven differences
give the mean Δ with its 90% t-interval, the exact sign-flip p over the 2^7 sign patterns, and the wins. The references
are not candidates.
- **The raw objective alone** (the config of the 72 with the highest mean per-match raw spellF1 on the six, no tie
  band): `p45c_t004_d032_mp025_md025` is picked for all seven matches. Scored under the raw truth against `nearest_w50`:
  mean Δ +0.2217 [+0.2019, +0.2415], p 0.0156, 7 wins and 0 losses (per match +0.1956, +0.1847, +0.2297, +0.2375,
  +0.2032, +0.2499, +0.2512). Scored under the coarse truth against `nearest_w125`: mean Δ +0.2341 [+0.2202, +0.2480],
  p 0.0156, 7 wins and 0 losses (+0.2112, +0.2115, +0.2280, +0.2340, +0.2576, +0.2562, +0.2403).
- **The whole rule** (on each fold: the configs within 0.005 of the best six-match raw mean, then the best six-match
  coarse mean among them, then the tie order): the same config is picked in all seven folds, so the held-out estimate
  is the chosen config's own, the figures above, p 0.0156, 7 wins and 0 losses under each truth.

The selection is stable (one config in every fold under each rule), so the choice is not what produces the gap. The
estimate does not cover the rule values (`controlled_touch`'s confidence, measured on all seven matches) nor the
trees' training, which holds out the scored match only through `folds`, and seven matches from two competitions (five
of them Fortuna Düsseldorf home games) are not seven independent draws.

**Controls on `context`** (sensitivity only, T38; `context` v1 fails two gates, R005; run from `15580c1`,
`--profile context`, `--seeds 3 --no-write`; both logs end `EXIT=0`; the references score on `context` as in R005
and R009).
- **The chosen config.** Pooled spellF1 0.4323 raw (mean per match 0.4341 [0.4116, 0.4565]) and 0.4654 coarse (0.4660
  [0.4466, 0.4854]); macroF1 0.8353 raw and 0.8361 coarse; alive/dead F1 0.9700; 2.81 possessions per graded minute; `D`
  share 44.8%. Against `nearest_w50` raw: pooled Δ +0.2130, mean +0.2144 [+0.1963, +0.2324], p 0.0156, 7 wins in 7.
  Against `nearest_w125` coarse: pooled Δ +0.2211, mean +0.2213 [+0.2059, +0.2367], p 0.0156, 7 wins in 7.
- **The four profile x truth cells agree.** The chosen config is above its truth's strongest yardstick in all four, 7
  matches of 7 each, p 0.0156: `tracking` raw +0.2199, `tracking` coarse +0.2330, `context` raw +0.2130, `context`
  coarse +0.2211. There is no warning to write (T38). The agreement is not evidence that the gain transfers: `context`
  keeps DFL's out-of-play ball tracking, so it shares the caveat above.

**The label-free check on our match.** The benchmark cannot show that the file is not degenerate on BAR-ATH, so this
check is required in addition to the bar; it was declared before Stage C ran, when no Stage C number existed. It is
computed over the graded blocks of BAR-ATH (`games.half_video_blocks`, 158,857 frames) of
`evaluation/results/p45_hmm.csv`, with no ground truth:
- **(a) The predicted `D` share** is 36.0%, inside [30%, 62%] (DFL's seven matches have 40.7% to 51.7% of their ball
  frames dead, A05 and T30; the band is that range widened by 10 points each way and rounded outward). It passes.
- **(b) The `X` labels inside the graded blocks:** 0. It passes.
- **(c) Determinism:** two runs of `pipeline/run.py pipeline/configs/p45_hmm.toml --score` give the same sha256
  (`a706e7f7...`), and the file is valid. It passes.

It cannot say the labels are right; it says the file is not degenerate on our match.

**Adoption.** The bar (the chosen config's raw Δ over `nearest_w50` and coarse Δ over `nearest_w125` both above 0 with
sign-flip p <= 0.05, on `tracking` v3 with 3 seeds) is met, and the check passes, so `pipeline/configs/main.toml` takes
this config's evidence and decoder, under its own header. `pipeline/run.py pipeline/configs/main.toml` writes
`evaluation/results/main.csv` with sha256 `a706e7f711b2d3faacc2f071b6e28963ed8e2f2326a7880e6d0776cce4295f1f`, the same
as `p45_hmm.toml`'s (the two configs have the same evidence, decoder, seed and scope), and the file passes
`evaluation/validate_submission.py` (199,319 rows). `pipeline/selftest.py --data` passes (114 of 114 checks).

**Checks on the runs.** All six grid logs and all four control logs end `EXIT=0`. Each grid log lists the two
references first (raw: `nearest_w50`, then `nearest_w125`; coarse the reverse) and 24 of the 72 configs; the three
shares are disjoint and together are the 72, none missing or repeated, and the references are the same in all six logs
to the last printed digit (`seconds` aside) and are R004c's: `nearest_w50` 0.2018 raw and `nearest_w125` 0.2226
coarse. The chosen config scores 0.4217 raw and 0.4556 coarse in the grid and in the control, with the same per-match
scores. Runtime: the runner's `seconds` for a grid config are 24.5 to 26.6 s and for a yardstick 7.6 to 8.8 s; the first
config of each process took 157.6 to 164.0 s (the one-off kinematics and features pass; the memo serves the rest of the
process). In the controls the first config took 135.5 s and 135.6 s on `tracking` (raw, coarse) and 136.8 s and 136.9 s
on `context`; the ablations took 20.8 s and 22.5 s (raw) and 20.8 s and 22.7 s (coarse).

Verdict: **`p45c_t004_d032_mp025_md025` (`controlled_touch` and `dead_ball_model` with
`pipeline/models/dead_ball_hgb.json` at weight 1.0 and the hmm (4, 32, 25, 25)) is P04 and P05's chosen config**, by
the pre-declared rule over the 72 configs, and is written to `pipeline/configs/p45_hmm.toml`. It meets the bar under
both truths (raw +0.2199 over `nearest_w50`, p 0.0156, 7 wins in 7; coarse +0.2330 over `nearest_w125`, p 0.0156, 7
wins in 7), the leave-one-match-out estimates agree (+0.2217 and +0.2341, 7 wins in 7 each), the `context` profile is
above both yardsticks by a wide margin (+0.2130, +0.2211), at the chosen decoder each source is needed (without P04
the coarse half of the bar fails, +0.0198, p 0.1406; without P05 it fails, -0.0056, p 0.1094; alone, at R007's
decoder, each clears both halves, R008 and R009), and the label-free check on BAR-ATH passes (`D` 36.0%, no `X` in a
graded block, the same sha256 twice). **`main.toml` is switched**, and then produced sha256 `a706e7f7...` (above; R011
and R012 have changed it since); it has not been uploaded. The limits: **the size of the gain rests on DFL's
out-of-play ball tracking and does not transfer in full to BAR-ATH, by an amount the benchmark cannot measure (T38,
T39)**, and the predicted `D` share on our match (36.0%) is 8.6 points below the benchmark's; the choice is on the
grid's lower edge in switch_team and min_possession and by the protocol none is extended; the coarse truth prefers the
opposite end of both axes (switch_team 16, min_possession 100: 0.5049 coarse, 0.3532 raw), so the possession length is
not settled (T33); the `controlled_touch` confidence is a measurement on the seven matches the grid is scored on; and
the public window is a correctness check only (valid, deterministic, 18 possessions against 17, alive agreement
86.6%).

### R011 — P07's shape-possession evidence: the model's held-out scores, then the 12-config grid (2026-10-25, Tian)
Proposal: P07 · Commit: `dd362bf` (the code the model was trained at: the trainer, `pipeline/benchmark/train_shape.py`,
merged) and `7151e7a` (the trained model, `pipeline/models/shape_possession_hgb.json`, committed); the grid's 12 configs
`pipeline/benchmark/configs/p07/p07_*.toml` and its runner are those of `5245490`, from a detached checkout of which the
four grid processes were run (the runner's header says `52454905e0a3`; its tree is `5c2b37d`'s, which part 2 builds on);
the label-free checks, the public window and the new `main.toml` were run at `5c2b37d` with `p07_shape.toml` and the new
`main.toml` not yet committed (the manifest says `+dirty`), and the commit that holds this entry adds both · Command:
part 1: `python pipeline/benchmark/train_shape.py` (writes the model; EXIT=0), `python
pipeline/benchmark/train_shape.py --verify` (retrains and compares with the committed file: exit 0, "is what the code
trains") and `python pipeline/benchmark/train_shape.py --report` (the held-out scores below; EXIT=0), all with the
defaults `--profile tracking --seeds 3 --stride 5`. Part 2: four processes, one per profile and truth, each
`python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml
pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/p07/*.toml --profile tracking --seeds 3 --truth
raw`, and the same with `--profile context`; under `--truth coarse` the two yardsticks come in the order `nearest_w125`,
`nearest_w50`. The label-free gate and figures on BAR-ATH are the two commands in the section on that check below.
Public window and determinism: `python pipeline/run.py pipeline/configs/p07_shape.toml --score`, then the same without
`--score`; `python pipeline/run.py pipeline/configs/main.toml` twice after the switch.

Part 1 (the model and what it scores held out) was written before any config of the grid had run; part 2 (the grid, the
choice, the bar, the veto, the gate and the verdict) after the four grid logs ended `EXIT=0`. Both apply the rules
declared in part 1, as written.

Public (a correctness check only, T01): macroF1 0.7151 · spellF1 0.4000 · alive/dead F1 0.9218 · 20 possessions against
the truth's 17, mean 14.1 s against its 14.2 s. The file is valid (199,319 rows), sha256
`8810afce254e94fe159769f3b55e8eef65f41a25c84f4550034e2afbc20e8487`, the same twice. Over the graded frames of the
match it labels H 37.6%, A 26.6% and D 35.8% (X 0.0%) and finds 439 possessions (mean 9.3 s). `--score` reports
`controlled_touch` agreeing with the public GT on home on 87.2% of 5,489 frames and `dead_ball_model` on alive on 86.6%
of 7,500 frames (both as in R010: neither source changed), and `shape_possession` agreeing on home on 90.9% of 6,031
frames (the judged frames where it has an opinion), with an opinion on home on 100.0% of the graded frames and none on
alive. Against R010's `main.toml` on the same window (0.6996, 0.4686, 0.9222, 18 possessions) macroF1 is 0.0155 higher
and spellF1 0.0686 lower; that is about one possession's worth on this window (T01: one possession moves spellF1 about
0.06 here), with two more possessions than R010's 18, and it says nothing either way. No bootstrap was run, because no
public-window gap is claimed, and no constant of the config comes from that window: the weight and the scope come from
the grid, `min_prob` was fixed beforehand, and the trees were trained on the DFL matches.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), 668.4 graded minutes per seed. The chosen
config, `p07_w200_all` (R010's `main.toml` plus `shape_possession` at weight 0.2, scope `all`, `min_prob` 0.02; each
benchmark match scored by the fold model that never saw it): on `tracking` v3 pooled spellF1 0.5159 raw (mean per match
0.5179 [0.4877, 0.5481]) and 0.5591 coarse (0.5602 [0.5355, 0.5850]); macroF1 0.8861 raw and 0.8892 coarse; alive/dead
F1 0.9729; 3.17 possessions per graded minute; `D` share 44.5% (the truth's 44.8%). On `context` (sensitivity only,
T38) 0.5153 raw (0.5176 [0.4869, 0.5482]) and 0.5544 coarse (0.5550 [0.5360, 0.5739]).

| profile | truth | against | pooled Δ | mean per match Δ [90% t] | p | wins |
|---|---|---|---|---|---|---|
| `tracking` | raw | `main` (R010): the bar's | +0.0942 | +0.0940 [+0.0775, +0.1105] | 0.0156 | 7 |
| `tracking` | coarse | `main`: a warning if negative (T33) | +0.1035 | +0.1033 [+0.0869, +0.1197] | 0.0156 | 7 |
| `context` | raw | `main`: the veto's | +0.0831 | +0.0835 [+0.0737, +0.0932] | 0.0156 | 7 |
| `context` | coarse | `main`: a warning if negative | +0.0890 | +0.0890 [+0.0834, +0.0945] | 0.0156 | 7 |

Changed vs the previous best: the previous best is R010's `main.toml` (`p45c_t004_d032_mp025_md025`): 0.4217 raw and
0.4556 coarse on `tracking` v3, 0.4323 and 0.4654 on `context`; public 0.6996 macroF1, 0.4686 spellF1, 18 possessions;
sha256 `a706e7f711b2d3faacc2f071b6e28963ed8e2f2326a7880e6d0776cce4295f1f`. The chosen config is +0.0942 raw and +0.1035
coarse above it on `tracking` (pooled), +0.0586 raw macroF1 (0.8275 to 0.8861) and +0.0003 alive/dead F1; on BAR-ATH
it relabels 5.30% of the graded frames (below).
Verdict: **P07's chosen config is weight 0.2, scope `all` (`p07_w200_all`)**, by the pre-declared rule: it is the only
config of the 12 within 0.005 of the best raw score. **The bar is met** (`tracking` raw Δ +0.0942 over `main.toml`, p
0.0156, 7 wins in 7), **the `context` veto does not fire** (raw Δ +0.0831), the coarse cells agree and so does the
leave-one-match-out estimate, and the label-free checks on BAR-ATH pass (`touch_agreement` 0.8251 against the gate's
0.7486, predicted `D` 35.8% against R010's 36.0%, 4.1452 possessions per graded minute against R010's 4.2491, no `X`
in a graded block, the same sha256 twice). **`main.toml` is switched**, and then produced sha256 `8810afce...` (not
uploaded; R012 has since added the shift). The limits are below: the shape model was trained on other teams' matches,
the benchmark's size of the gain is not a forecast for BAR-ATH (the P05 caveat of R010 still holds), and the public
window is a correctness check only.

**The model.** A classifier of which team has the ball from the 26 `shape_features` of a frame (feature version 1: per
team the present slots, the mean and the 10th and 90th percentile of how far up the pitch the team is, its width, its
velocity along the attack and its mean speed, its shares in the attacking and the defending third, the change of its
mean advance over 1 s and 2 s and its velocity over a 1 s window; plus the gap between the two teams' advance and in
their present slots). No column reads the ball, so the model can be asked about the frames where the ball is missing,
which is 41% of the graded frames of our match. It is scikit-learn's `HistGradientBoostingClassifier` with P05's
settings (`train_dead_ball.HGB_SETTINGS`: learning rate 0.05, 300 iterations, 15 leaves, `min_samples_leaf` 1000, L2
1.0, no early stopping, `random_state` 0), fixed beforehand and never tuned (T26). The training rows are every 5th
alive graded frame (truth H or A) of the seven DFL matches, degraded as `tracking` v3, seeds 0 to 2: 331,881 rows
(41,016 to 51,213 a match, home 43.9% to 62.9% of a match's rows). Each row is added again as its team-swapped mirror
with the label flipped, so the classes balance exactly (`prior_home` 0.5) and `full` is fitted on 663,762 rows. There is
one fold model per held-out match, fitted on the other six, and `full`, fitted on all seven; the file is 3,668,856
bytes. `pipeline/evidence/shape_possession.py` runs the trees with numpy alone, symmetrises the output (the logit is
half of r(x) minus r(swap(x))) and clips `home` to [0.02, 0.98]. A benchmark match is scored by its own fold and BAR-ATH
by `full`.

**Held out.** Profile `tracking` version 3, 7 DFL matches x 3 degradation seeds (0, 1, 2), the seeds pooled by frames.
Each fold model on its own match, served as the pipeline serves it (`shape_possession` with scope `all` and the
default `min_prob` 0.02), against the raw truth (`D` where `BallStatus` is dead, else the team in `BallPossession`) on
the alive frames (truth H or A) of all the graded frames, not only the training stride; `home` >= 0.5 is the
accuracy's call. "Ball missing" is the alive frames where the served (degraded) ball has no position. The source had
an opinion on every alive frame of every match (0 silent: it is silent only where nobody is present). `touch
agreement` is the label-free cross-check described below. The pooled row sums the frames, and is not a mean of the
matches.

| match | alive frames | log-loss | accuracy | ball missing: frames | log-loss | accuracy | touch frames | touch agreement |
|---|---|---|---|---|---|---|---|---|
| J03WMX | 256,107 | 0.3759 | 0.8474 | 98,966 | 0.3928 | 0.8304 | 89,144 | 0.8546 |
| J03WN1 | 205,131 | 0.4739 | 0.7809 | 87,035 | 0.4612 | 0.7912 | 82,958 | **0.7486** |
| J03WOH | 215,835 | 0.3603 | 0.8430 | 90,369 | 0.3711 | 0.8373 | 80,249 | 0.8170 |
| J03WOY | 228,858 | 0.3676 | 0.8429 | 95,561 | 0.3624 | 0.8440 | 76,411 | 0.8476 |
| J03WPY | 244,380 | 0.3382 | 0.8496 | 101,397 | 0.3574 | 0.8389 | 85,679 | 0.8462 |
| J03WQQ | 253,056 | 0.3354 | 0.8530 | 108,130 | 0.3481 | 0.8475 | 89,155 | 0.8159 |
| J03WR9 | 255,948 | 0.3120 | 0.8669 | 96,442 | 0.3500 | 0.8495 | 100,223 | 0.8654 |
| **pooled** | 1,659,315 | 0.3633 | 0.8422 | 677,900 | **0.3759** | 0.8349 | 603,819 | 0.8290 |

`ln 2` = 0.6931 is the log-loss of saying 0.5 on every frame. The pooled alive-frame log-loss is 0.3633 and the pooled
log-loss with the ball missing is 0.3759 (those frames are 40.9% of the alive frames, close to our own match's 41% of
the graded ones): the frames the source is for are slightly harder than the rest, and still 0.3172 nats below chance.
J03WN1 is the hardest match on every column (log-loss 0.4739, 0.4612 with the ball missing; accuracy 0.7809 and 0.7912;
touch agreement 0.7486), J03WR9 the easiest by accuracy (0.8669, 0.8495 with the ball missing) and J03WQQ by the
ball-missing log-loss (0.3481).

**The stop rule does not fire.** P07's design declared that the pooled held-out log-loss on the alive frames with the
ball missing must be below ln 2 - 0.01 = 0.6831, or P07 stops here and the negative result is recorded. It is **0.3759
< 0.6831**, by 0.3072, and the worst match (J03WN1, 0.4612) is also 0.2219 under it. P07 goes on to the grid.

**What the numbers do not say.**
- They are frame-level scores of a probability. They show that the shape of the teams carries information about who has
  the ball where the ball is missing; they do not show what it adds to possession spellF1 where `controlled_touch` and
  the decoder already hold a view. That is the grid's question. Neighbouring frames are nearly the same frame, so no
  interval is given and 0.84 accuracy is not 84% of possessions.
- They are scores on DFL's degraded tracking under DFL's flag, which is fine-grained (437 possessions a match, A06).
  Seven matches from two competitions, five of them Fortuna Düsseldorf home games, are not seven independent draws (as
  in R010), and a fold that holds one of those out trains on four others. BAR-ATH is a LaLiga match with neither team in
  the training set. The held-out figures are not a forecast for it; the transfer gate below is the label-free look.

**`touch_agreement`, and the transfer gate.** The share of the frames inside kept `controlled_touch` episodes
(`conventions.TOUCH`: 2.0 m, 6.0 m/s, 3 frames, confidence 0.99 so that `home` is a number on exactly the kept frames,
hold off) where the shape source's `home` (scope `all`, >= 0.5) names the episode's team. It needs no truth, so it can
be asked of BAR-ATH with `full`. It is a proxy and not an accuracy: the episode's team is itself right on about 92% of
its frames (91.9% pooled over the seven matches, the source of `controlled_touch`'s 0.91 in `main.toml`). The pooled
value is 0.8290 over 603,819 frames; per match 0.8546, 0.7486, 0.8170, 0.8476, 0.8462, 0.8159 and 0.8654 (table). **The
lowest per match is 0.7486, J03WN1.** That is the threshold of the transfer gate declared below, and it is read from
this table and fixed here.

**Declared before the grid runs** (P07's design, 2026-10-25; part 2 applies it as written). The weights, the scopes and
the rules below are fixed before any config of the grid has run, and none is chosen from a grid score:
- *Selection:* the objective is the pooled spellF1 on `tracking`, raw truth, over the 12 configs; the tie band is 0.005;
  ties go to the best `tracking` coarse score, then the smaller weight, then `gaps` over `all`.
- *Bar,* against `main.toml` (R010): `tracking` raw Δ > 0 with sign-flip p <= 0.05; `context` raw Δ > 0, a veto and
  never a validation (T38); the coarse cells are reported as warnings (T33).
- *Label-free transfer gate:* on BAR-ATH, `touch_agreement` with the `full` model and scope `all` must be at least
  **0.7486**, the lowest per-match value above.
- *The rest:* no `X` in the graded blocks and the same sha256 twice; the `D` share and the possessions per graded
  minute are reported against R010's (36.0% and 4.2491, which is 450 possessions over the graded frames). *Note:* this
  bullet first counted possessions over the graded frames, and after the grid had run its wording was aligned to P07's
  design, which says per graded minute (the same information: 450 possessions in 105.9 minutes). It only names what is
  reported, and no choice depended on it.
- If adopted, `main.toml` becomes the chosen config. `pipeline/configs/p07_shape.toml` is written either way.

**The grid.** 12 configs `pipeline/benchmark/configs/p07/p07_w{WWW}_{gaps,all}.toml`: the weight w of `shape_possession`
in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} (`WWW` is 1000 w, zero-padded: 010, 020, 050, 100, 200, 500) x scope `gaps` (home
only where the ball is missing) or `all` (every graded frame). Each is `main.toml`'s evidence and decoder verbatim
(`controlled_touch` 2.0 m / 6.0 m/s / 3 frames / hold 0.55 at confidence 0.91, `dead_ball_model` with the `hgb` file at
weight 1.0, hmm (4, 32, 25, 25) decoded per half, seed 0) plus `shape_possession` with the committed model and
`min_prob` 0.02. The weight is an exponent on the source's likelihood, and `min_prob` caps a frame at 3.9 nats before
it. The runs (part 2): `main.toml`, `nearest_w50`, `nearest_w125` and the 12 configs, each on `tracking` v3 and on
`context` (a sensitivity check and the veto only, T38), under the raw and the coarse truth, 3 seeds.

**Part 2: the grid.** Four processes, one per profile and truth, each over `main.toml`, the two yardsticks and the 12
configs (7 matches x 3 seeds; 668.4 graded minutes per seed). `main.toml` is R010's (this is before the switch) and is
the reference of every Δ; the yardsticks are the references of R004c and R005 and are not candidates. The first table
is `tracking` v3, the profile of the selection (the same `alive/dead F1`, possessions per minute and `D` share under
both truths: the predicted labels do not depend on the truth); the second is `context`, sensitivity only (T38). The
truth has 4.58 possessions per graded minute raw and 2.48 coarse, and `D` on 44.8% of the graded frames. The bold row
is the chosen config.

`tracking` v3:

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|---|
| `main` | 0.4217 | 0.4556 | 0.8275 | 0.8288 | 0.9726 | 2.74 | 44.6% |
| `nearest_w50` | 0.2018 | 0.1842 | 0.4592 | 0.4602 | 0.6094 | 4.30 | 33.4% |
| `nearest_w125` | 0.1665 | 0.2226 | 0.4407 | 0.4509 | 0.6428 | 3.05 | 23.3% |
| `p07_w010_all` | 0.4537 | 0.4993 | 0.8636 | 0.8668 | 0.9726 | 2.80 | 44.6% |
| `p07_w010_gaps` | 0.4528 | 0.4988 | 0.8634 | 0.8665 | 0.9726 | 2.81 | 44.6% |
| `p07_w020_all` | 0.4694 | 0.5186 | 0.8711 | 0.8753 | 0.9726 | 2.88 | 44.6% |
| `p07_w020_gaps` | 0.4673 | 0.5161 | 0.8706 | 0.8748 | 0.9726 | 2.89 | 44.6% |
| `p07_w050_all` | 0.4923 | 0.5426 | 0.8791 | 0.8830 | 0.9726 | 2.98 | 44.6% |
| `p07_w050_gaps` | 0.4866 | 0.5319 | 0.8768 | 0.8804 | 0.9726 | 3.06 | 44.6% |
| `p07_w100_all` | 0.5057 | 0.5531 | 0.8829 | 0.8864 | 0.9727 | 3.05 | 44.6% |
| `p07_w100_gaps` | 0.4961 | 0.5335 | 0.8788 | 0.8813 | 0.9727 | 3.18 | 44.6% |
| **`p07_w200_all`** | 0.5159 | 0.5591 | 0.8861 | 0.8892 | 0.9729 | 3.17 | 44.5% |
| `p07_w200_gaps` | 0.5046 | 0.5262 | 0.8799 | 0.8806 | 0.9729 | 3.33 | 44.5% |
| `p07_w500_all` | 0.5051 | 0.5406 | 0.8839 | 0.8867 | 0.9727 | 3.35 | 44.3% |
| `p07_w500_gaps` | 0.5032 | 0.5057 | 0.8789 | 0.8783 | 0.9727 | 3.56 | 44.3% |

`context` v1 (sensitivity only):

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share |
|---|---|---|---|---|---|---|---|
| `main` | 0.4323 | 0.4654 | 0.8353 | 0.8361 | 0.9700 | 2.81 | 44.8% |
| `nearest_w50` | 0.2193 | 0.2057 | 0.5144 | 0.5160 | 0.6575 | 4.37 | 34.2% |
| `nearest_w125` | 0.1774 | 0.2443 | 0.4770 | 0.4877 | 0.6711 | 3.07 | 23.1% |
| `p07_w010_all` | 0.4626 | 0.5080 | 0.8660 | 0.8697 | 0.9700 | 2.86 | 44.8% |
| `p07_w010_gaps` | 0.4613 | 0.5071 | 0.8657 | 0.8693 | 0.9700 | 2.86 | 44.8% |
| `p07_w020_all` | 0.4737 | 0.5223 | 0.8718 | 0.8756 | 0.9700 | 2.91 | 44.8% |
| `p07_w020_gaps` | 0.4721 | 0.5206 | 0.8713 | 0.8752 | 0.9700 | 2.92 | 44.8% |
| `p07_w050_all` | 0.4969 | 0.5430 | 0.8790 | 0.8823 | 0.9699 | 3.01 | 44.8% |
| `p07_w050_gaps` | 0.4902 | 0.5332 | 0.8770 | 0.8803 | 0.9699 | 3.08 | 44.8% |
| `p07_w100_all` | 0.5082 | 0.5501 | 0.8825 | 0.8855 | 0.9699 | 3.07 | 44.8% |
| `p07_w100_gaps` | 0.4987 | 0.5323 | 0.8786 | 0.8809 | 0.9699 | 3.19 | 44.8% |
| **`p07_w200_all`** | 0.5153 | 0.5544 | 0.8847 | 0.8872 | 0.9699 | 3.15 | 44.7% |
| `p07_w200_gaps` | 0.5053 | 0.5251 | 0.8792 | 0.8802 | 0.9699 | 3.31 | 44.7% |
| `p07_w500_all` | 0.5088 | 0.5401 | 0.8831 | 0.8853 | 0.9697 | 3.33 | 44.4% |
| `p07_w500_gaps` | 0.5095 | 0.5104 | 0.8793 | 0.8787 | 0.9697 | 3.49 | 44.4% |

Paired against `main.toml` (the first config of each log; Δ = config minus `main`, so positive means the config is
better; pooled Δ, then the mean per-match Δ with its 90% t-interval, the exact sign-flip p over the 2^7 sign patterns,
and the wins in 7 matches; the yardsticks have 0 wins and 7 losses). With 7 matches the smallest p there can be is
0.0156, so neither it nor an interval makes a gap real.

`tracking` v3, raw truth (the bar's cell):

| config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins in 7 | macroF1 Δ |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2018 | -0.2199 | -0.2217 [-0.2415, -0.2019] | 0.0156 | 0 | -0.3683 |
| `nearest_w125` | 0.1665 | -0.2552 | -0.2564 [-0.2732, -0.2396] | 0.0156 | 0 | -0.3868 |
| `p07_w010_all` | 0.4537 | +0.0320 | +0.0321 [+0.0266, +0.0375] | 0.0156 | 7 | +0.0361 |
| `p07_w010_gaps` | 0.4528 | +0.0311 | +0.0312 [+0.0259, +0.0366] | 0.0156 | 7 | +0.0358 |
| `p07_w020_all` | 0.4694 | +0.0477 | +0.0480 [+0.0404, +0.0555] | 0.0156 | 7 | +0.0436 |
| `p07_w020_gaps` | 0.4673 | +0.0456 | +0.0458 [+0.0381, +0.0536] | 0.0156 | 7 | +0.0430 |
| `p07_w050_all` | 0.4923 | +0.0706 | +0.0709 [+0.0609, +0.0809] | 0.0156 | 7 | +0.0516 |
| `p07_w050_gaps` | 0.4866 | +0.0650 | +0.0651 [+0.0564, +0.0737] | 0.0156 | 7 | +0.0492 |
| `p07_w100_all` | 0.5057 | +0.0840 | +0.0841 [+0.0695, +0.0986] | 0.0156 | 7 | +0.0554 |
| `p07_w100_gaps` | 0.4961 | +0.0745 | +0.0743 [+0.0642, +0.0844] | 0.0156 | 7 | +0.0512 |
| **`p07_w200_all`** | 0.5159 | +0.0942 | +0.0940 [+0.0775, +0.1105] | 0.0156 | 7 | +0.0586 |
| `p07_w200_gaps` | 0.5046 | +0.0830 | +0.0827 [+0.0719, +0.0935] | 0.0156 | 7 | +0.0524 |
| `p07_w500_all` | 0.5051 | +0.0835 | +0.0830 [+0.0685, +0.0975] | 0.0156 | 7 | +0.0564 |
| `p07_w500_gaps` | 0.5032 | +0.0815 | +0.0810 [+0.0694, +0.0927] | 0.0156 | 7 | +0.0514 |

`tracking` v3, coarse truth (a warning, T33):

| config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins in 7 | macroF1 Δ |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.1842 | -0.2714 | -0.2724 [-0.2910, -0.2537] | 0.0156 | 0 | -0.3686 |
| `nearest_w125` | 0.2226 | -0.2330 | -0.2341 [-0.2480, -0.2202] | 0.0156 | 0 | -0.3779 |
| `p07_w010_all` | 0.4993 | +0.0437 | +0.0437 [+0.0342, +0.0532] | 0.0156 | 7 | +0.0380 |
| `p07_w010_gaps` | 0.4988 | +0.0432 | +0.0432 [+0.0342, +0.0522] | 0.0156 | 7 | +0.0377 |
| `p07_w020_all` | 0.5186 | +0.0630 | +0.0631 [+0.0525, +0.0738] | 0.0156 | 7 | +0.0465 |
| `p07_w020_gaps` | 0.5161 | +0.0605 | +0.0605 [+0.0504, +0.0706] | 0.0156 | 7 | +0.0459 |
| `p07_w050_all` | 0.5426 | +0.0870 | +0.0870 [+0.0712, +0.1027] | 0.0156 | 7 | +0.0541 |
| `p07_w050_gaps` | 0.5319 | +0.0762 | +0.0759 [+0.0628, +0.0891] | 0.0156 | 7 | +0.0515 |
| `p07_w100_all` | 0.5531 | +0.0974 | +0.0973 [+0.0805, +0.1141] | 0.0156 | 7 | +0.0576 |
| `p07_w100_gaps` | 0.5335 | +0.0779 | +0.0774 [+0.0652, +0.0897] | 0.0156 | 7 | +0.0525 |
| **`p07_w200_all`** | 0.5591 | +0.1035 | +0.1033 [+0.0869, +0.1197] | 0.0156 | 7 | +0.0604 |
| `p07_w200_gaps` | 0.5262 | +0.0706 | +0.0703 [+0.0582, +0.0824] | 0.0156 | 7 | +0.0518 |
| `p07_w500_all` | 0.5406 | +0.0850 | +0.0844 [+0.0702, +0.0986] | 0.0156 | 7 | +0.0578 |
| `p07_w500_gaps` | 0.5057 | +0.0501 | +0.0497 [+0.0400, +0.0594] | 0.0156 | 7 | +0.0494 |

`context` v1, raw truth (the veto's cell, T38):

| config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins in 7 | macroF1 Δ |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2193 | -0.2130 | -0.2144 [-0.2324, -0.1963] | 0.0156 | 0 | -0.3209 |
| `nearest_w125` | 0.1774 | -0.2549 | -0.2555 [-0.2741, -0.2368] | 0.0156 | 0 | -0.3583 |
| `p07_w010_all` | 0.4626 | +0.0303 | +0.0305 [+0.0271, +0.0339] | 0.0156 | 7 | +0.0307 |
| `p07_w010_gaps` | 0.4613 | +0.0290 | +0.0292 [+0.0251, +0.0332] | 0.0156 | 7 | +0.0304 |
| `p07_w020_all` | 0.4737 | +0.0414 | +0.0417 [+0.0380, +0.0454] | 0.0156 | 7 | +0.0365 |
| `p07_w020_gaps` | 0.4721 | +0.0398 | +0.0401 [+0.0355, +0.0446] | 0.0156 | 7 | +0.0361 |
| `p07_w050_all` | 0.4969 | +0.0646 | +0.0650 [+0.0579, +0.0721] | 0.0156 | 7 | +0.0437 |
| `p07_w050_gaps` | 0.4902 | +0.0579 | +0.0580 [+0.0509, +0.0652] | 0.0156 | 7 | +0.0417 |
| `p07_w100_all` | 0.5082 | +0.0760 | +0.0764 [+0.0691, +0.0837] | 0.0156 | 7 | +0.0472 |
| `p07_w100_gaps` | 0.4987 | +0.0664 | +0.0665 [+0.0610, +0.0721] | 0.0156 | 7 | +0.0433 |
| **`p07_w200_all`** | 0.5153 | +0.0831 | +0.0835 [+0.0737, +0.0932] | 0.0156 | 7 | +0.0494 |
| `p07_w200_gaps` | 0.5053 | +0.0730 | +0.0731 [+0.0672, +0.0789] | 0.0156 | 7 | +0.0439 |
| `p07_w500_all` | 0.5088 | +0.0766 | +0.0765 [+0.0655, +0.0875] | 0.0156 | 7 | +0.0479 |
| `p07_w500_gaps` | 0.5095 | +0.0772 | +0.0772 [+0.0690, +0.0853] | 0.0156 | 7 | +0.0440 |

`context` v1, coarse truth (a warning):

| config | pooled spellF1 | pooled Δ | mean per match Δ [90% t] | p | wins in 7 | macroF1 Δ |
|---|---|---|---|---|---|---|
| `nearest_w50` | 0.2057 | -0.2597 | -0.2601 [-0.2746, -0.2456] | 0.0156 | 0 | -0.3201 |
| `nearest_w125` | 0.2443 | -0.2211 | -0.2213 [-0.2367, -0.2059] | 0.0156 | 0 | -0.3485 |
| `p07_w010_all` | 0.5080 | +0.0426 | +0.0427 [+0.0400, +0.0453] | 0.0156 | 7 | +0.0336 |
| `p07_w010_gaps` | 0.5071 | +0.0418 | +0.0417 [+0.0392, +0.0442] | 0.0156 | 7 | +0.0332 |
| `p07_w020_all` | 0.5223 | +0.0570 | +0.0571 [+0.0525, +0.0617] | 0.0156 | 7 | +0.0395 |
| `p07_w020_gaps` | 0.5206 | +0.0553 | +0.0553 [+0.0519, +0.0586] | 0.0156 | 7 | +0.0391 |
| `p07_w050_all` | 0.5430 | +0.0776 | +0.0778 [+0.0693, +0.0863] | 0.0156 | 7 | +0.0462 |
| `p07_w050_gaps` | 0.5332 | +0.0678 | +0.0677 [+0.0624, +0.0730] | 0.0156 | 7 | +0.0442 |
| `p07_w100_all` | 0.5501 | +0.0848 | +0.0849 [+0.0757, +0.0940] | 0.0156 | 7 | +0.0494 |
| `p07_w100_gaps` | 0.5323 | +0.0669 | +0.0668 [+0.0624, +0.0711] | 0.0156 | 7 | +0.0448 |
| **`p07_w200_all`** | 0.5544 | +0.0890 | +0.0890 [+0.0834, +0.0945] | 0.0156 | 7 | +0.0511 |
| `p07_w200_gaps` | 0.5251 | +0.0597 | +0.0596 [+0.0574, +0.0617] | 0.0156 | 7 | +0.0441 |
| `p07_w500_all` | 0.5401 | +0.0747 | +0.0746 [+0.0645, +0.0847] | 0.0156 | 7 | +0.0492 |
| `p07_w500_gaps` | 0.5104 | +0.0450 | +0.0447 [+0.0389, +0.0506] | 0.0156 | 7 | +0.0426 |

**The choice, by the pre-declared rule.** Objective: pooled spellF1 under the raw truth on `tracking`, over the 12
configs. The best is `p07_w200_all`, 0.5159. The tie band is 0.005, so the tied set is raw >= 0.5109, and it holds that
one config: the next, `p07_w100_all`, is at 0.5057 (0.0102 below), then `p07_w500_all` at 0.5051 (0.0108) and
`p07_w200_gaps` at 0.5046 (0.0113). There is no tie, so the tie order (best coarse, then the smaller weight, then `gaps`
over `all`) is not reached. **Chosen: `p07_w200_all`**, weight 0.2, scope `all`, which is also the raw best.
- **No plateau, a peak.** Along the weight, with scope `all`, raw spellF1 rises 0.4537, 0.4694, 0.4923, 0.5057, 0.5159
  (weights 0.01 to 0.2) and falls to 0.5051 at 0.5; with `gaps` it is 0.4528, 0.4673, 0.4866, 0.4961, 0.5046, 0.5032.
  The winner is inside the grid in weight, not on an edge: with `all` the steps are +0.0157, +0.0229, +0.0134 and
  +0.0102, then -0.0108 at 0.5, and the peak is at 0.2 for `all` in all four cells.
- **Scope.** `all` is above `gaps` at every weight under `tracking` raw (by 0.0009, 0.0021, 0.0057, 0.0096, 0.0113 and
  0.0019 for weights 0.01 to 0.5) and under `tracking` coarse (0.0005 to 0.0349); on `context` it is above at all six
  under the coarse truth and at five of six under the raw one (weight 0.5: `gaps` by 0.0007). So the shape source helps
  on the frames where `controlled_touch` also speaks, and not only where the ball is missing, which is where it was
  designed to be used; at the chosen weight the difference is 0.0113 raw and 0.0329 coarse.
- **The truths agree.** The raw winner on `tracking` is also the coarse winner (0.5591), and `p07_w200_all` is the best
  config of both `context` cells too (0.5153 raw, 0.5544 coarse). There is no T33 disagreement to read: four cells, one
  winner. The possessions per graded minute it predicts, 3.17 on `tracking`, are 0.43 above `main`'s 2.74 and sit
  between the two truths' (2.48 coarse, 4.58 raw), as `main`'s do.
- **Every config beats `main`.** The smallest Δ of the 48 (12 configs x 4 cells) is +0.0290 (`p07_w010_gaps` on
  `context` raw), the smallest on `tracking` raw +0.0311, and each has p 0.0156 and 7 wins in 7. With scope `all` the
  gain grows with the weight up to 0.2 in all four cells, so it is not a threshold effect of the first small weight,
  and the rule's choice is the largest gain, not the only one.
- **At the edges.** The chosen config is inside the weight axis and `all` is one of the two scopes. **Per the protocol
  nothing is added after the results**: a weight between 0.1 and 0.5, or above 0.5, was not tried, and the entry does
  not claim it would not score higher, nor that it would be sound.

**Leave-one-match-out estimate of the selection.** As R008 to R010 did, from the per-match spellF1 the runner prints
(four decimals): for each match, the selection is repeated on the other six matches and the chosen config's score on
the held-out match is compared with `main.toml`'s on it; the seven differences give the mean Δ with its 90%
t-interval, the exact sign-flip p and the wins. Two rules over the 12 configs: the raw objective alone (the highest
mean per-match `tracking` raw spellF1 on the six, no tie band) and the whole rule (the configs within 0.005 of the
best six-match raw mean, then the best six-match coarse mean among them, then the smaller weight, then `gaps` over
`all`). **Both pick `p07_w200_all` in all seven folds**, so the held-out estimate is the chosen config's own, computed
from the rounded printed scores (the last digit can differ from the runner's by 0.0001):
- `tracking` raw: mean Δ +0.0940 [+0.0775, +0.1105], p 0.0156, 7 wins and 0 losses (per match +0.0730, +0.0805,
  +0.1009, +0.1388, +0.0983, +0.0901, +0.0764, in the order of the table above);
- `tracking` coarse: +0.1033 [+0.0869, +0.1197], p 0.0156, 7 wins (+0.1119, +0.0788, +0.1242, +0.1317, +0.1136,
  +0.0846, +0.0782);
- `context` raw +0.0835 [+0.0737, +0.0933] and coarse +0.0890 [+0.0834, +0.0945], each p 0.0156 and 7 wins.

The selection is stable (one config in every fold under each rule), so the choice is not what produces the gap. The
estimate does not cover the model's training, which holds a match out only through `folds` (each fold model never saw
the match it scores), and seven matches from two competitions, five of them Fortuna Düsseldorf home games, are not
seven independent draws. By competition the raw Δ is +0.0767 on DFL-COM-000001 and +0.1009 on DFL-COM-000002.

**The bar, the veto and the warnings** (declared in part 1, applied as written).
- *Bar,* `tracking` raw, against `main.toml`: Δ +0.0942 > 0, sign-flip p 0.0156 <= 0.05, 7 wins in 7. **Met.**
- *Veto,* `context` raw: Δ +0.0831 > 0. **It does not fire.** It is a veto and never a validation (T38): `context`
  keeps DFL's out-of-play ball tracking and shares the caveats below.
- *Warnings,* the coarse cells (T33): `tracking` +0.1035 and `context` +0.0890, both positive, p 0.0156, 7 wins in 7.
  There is no warning to write.

**The transfer gate and the label-free check on BAR-ATH.** The benchmark cannot show that the file is not degenerate on
our match, and the shape model was trained on seven other matches and has never seen a frame of it, so these checks are
required in addition to the bar. They were declared in part 1, before any config of the grid had run. They need no
ground truth, and print aggregates only. The gate (the model `full`, since BAR-ATH is no fold):

```
python -c "from pipeline.inputs import Inputs; from pipeline.benchmark import train_shape; r = train_shape.touch_agreement(Inputs(), model='pipeline/models/shape_possession_hgb.json'); print('frames {:,}  agreement {:.4f}  gate >= 0.7486  {}'.format(r['frames'], r['agreement'], 'PASS' if r['agreement'] >= 0.7486 else 'FAIL'))"
```

It prints `frames 41,052  agreement 0.8251  gate >= 0.7486  PASS`. The figures from a written file, over the graded
blocks of BAR-ATH (`games.half_video_blocks`, 158,857 frames, 105.90 minutes), run on
`evaluation/results/p07_shape.csv` and again on `main.csv` after the switch (the same output, the same file), and on
R010's file before it:

```
python -c "import sys; sys.path[:0]=['.', 'evaluation']
import numpy as np, games, possession_metrics as pm
c = pm.label_codes(pm.read_label_csv(sys.argv[1]))
runs = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
m = np.zeros(len(c), bool)
for a, b in runs: m[a:b] = True
g = c[m]
s = pm.possession_spells(pm.fill_abstentions(c, runs), runs)
print({l: round(float((g == i).mean()), 4) for i, l in enumerate(pm.LABELS)}, 'X frames', int((g == 3).sum()))
print('possessions', len(s), 'per graded minute', round(len(s) / (m.sum() / 25 / 60), 4))" evaluation/results/p07_shape.csv
```

| | R010's `main.toml` (`a706e7f7`) | the chosen config (`8810afce`) | declared rule |
|---|---|---|---|
| gate: `touch_agreement`, `full`, scope `all` | n/a | **0.8251** on 41,052 frames | at least 0.7486 |
| `D` share of the graded frames | 36.0% | 35.8% | reported |
| H, A of the graded frames | 37.5%, 26.5% | 37.6%, 26.6% | reported |
| `X` frames inside the graded blocks | 0 | 0 | none |
| possessions | 450 | 439 | reported |
| possessions per graded minute | 4.2491 | 4.1452 | reported |
| mean possession | 9.0 s | 9.3 s | reported |
| the same sha256 twice | yes | yes | yes |

- **The gate passes**: 0.8251 against 0.7486, by 0.0765, and 0.0039 under the benchmark's pooled held-out value
  (0.8290). It is a proxy and not an accuracy (the episode's team is right on about 92% of its frames, part 1). BAR-ATH
  has 41,052 frames in kept episodes in one pass, against 25,470 to 33,408 per seed on the DFL matches (the held-out
  table's frames over 3 seeds), so the statistic is read on a larger sample.
- **The `D` share** moves by 0.2 points, 35.8% against R010's 36.0%, and stays inside the band R010 declared
  ([30%, 62%], from DFL's 40.7% to 51.7% widened by 10 points). The alive/dead decision is P05's and the shape source
  has no opinion on `alive`; the labels' alive/dead status differs on 0.22% of the graded frames between the two files.
- **The possessions** are 439 against R010's 450 (4.1452 per graded minute against 4.2491, 2.4% fewer, and the mean
  one 0.25 s longer). **The benchmark predicts the opposite direction**: on `tracking` v3 the chosen config has 3.17
  per graded minute against `main`'s 2.74 (about 16% more). The two are different matches, with different ball
  coverage and a count of 4.15 per minute where the benchmark has 3.17 for this config (R010's config: 4.25 against
  2.74), and nobody here knows what our match's true count is; the entry reports the difference and does not explain
  it.
- **How much of the file it changes**: the chosen config's labels differ from R010's on 5.30% of the graded frames, 0.22
  points of them a change of alive/dead and 5.08 points a change of team (7.94% of the frames that are alive in both
  files).
- **No `X`, and the same sha256 twice**: `pipeline/run.py pipeline/configs/p07_shape.toml` gave
  `8810afce254e94fe159769f3b55e8eef65f41a25c84f4550034e2afbc20e8487` twice (once with `--score`), and
  `pipeline/configs/main.toml`, switched, gave the same sha256 twice, which is the sha256 of `p07_shape.toml`'s file.

It cannot say the labels are right; it says the file is not degenerate on our match and that the shape source agrees
with the touch episodes on it about as often as on the held-out matches (82.5% of their frames, against 74.9% to
86.5% for the seven folds).

**Adoption.** The bar is met, the veto does not fire, and the gate and the checks pass, so `pipeline/configs/main.toml`
takes this config's evidence and decoder, under a header that says what it is now. `pipeline/run.py
pipeline/configs/main.toml` writes `evaluation/results/main.csv` with sha256
`8810afce254e94fe159769f3b55e8eef65f41a25c84f4550034e2afbc20e8487`, the same as `p07_shape.toml`'s (the two configs have
the same evidence, decoder, seed and scope), and the file passes `evaluation/validate_submission.py` (199,319 rows).
`pipeline/selftest.py --data` passes (153 of 153 checks; it runs `main.toml` twice and compares the bytes, and rebuilds
the four baselines byte for byte) and `pipeline/benchmark/selftest.py` passes (369 checks). `p07_shape.toml` is written,
and writes the same labels file under another name, as `p45_hmm.toml` does for R010's.

**Checks on the runs.** All four logs end `EXIT=0`. Each lists `main`, the two yardsticks (raw: `nearest_w50` then
`nearest_w125`; coarse the reverse) and the 12 configs, none missing or repeated. The `main` rows are R010's to the
last digit in all four cells (0.4217 raw and 0.4556 coarse on `tracking`, 0.4323 and 0.4654 on `context`), and the
yardsticks are R004c's and R005's (`nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse on `tracking`, 0.2193 and
0.2443 on `context`). The runner's check on the yardsticks' order prints Kendall tau +1.0000 under the raw truth and
-1.0000 under the coarse one on both profiles, as for the same two configs in R004c and R005 (the truth reorders them;
a check of the profile, not a goal). Runtime, the runner's `seconds`: the first config of each process (`main`, with
the one-off kinematics and features pass) took 144.9 and 143.3 s on `tracking` (raw, coarse) and 130.9 and 130.7 s on
`context`; the first `p07` config 156.1 and 155.7 s (the shape logits are walked once and the memo serves the rest)
and 141.8 and 140.9 s; the other 11 took 25.6 to 27.5 s on `tracking` and 23.4 to 25.1 s on `context`; a yardstick
took 5.7 to 6.5 s.

**The limits, and what the numbers do not say.**
- **The model never saw our teams.** It was trained on seven matches of two competitions, five of them Fortuna
  Düsseldorf home games, and BAR-ATH is a LaLiga match with neither team among them. The gate is the one look at
  transfer, and a proxy: 0.8251 says the shape agrees with the touch episodes about as often as on the held-out matches,
  not that its team is right that often where the ball is missing.
- **The gain is the benchmark's.** The benchmark overstates P05 for BAR-ATH (T38, T39, R010), and the size of the
  combined gain over R010 rests on that source as well; this entry's Δ is a gain over a `main.toml` that already
  has it. The model's own held-out accuracy (part 1) is 0.8422 on the alive frames and 0.8349 with the ball missing,
  both against DFL's fine-grained flag (A06).
- **The count of possessions does not transfer.** The benchmark moves it up (2.74 to 3.17 a minute) and BAR-ATH moves
  down (4.25 to 4.15 a minute), by 2.4%. The truth's count on our match is unknown.
- **The public window** is a correctness check only: valid, deterministic, 20 possessions against 17, agreement of the
  shape source on home 90.9% of 6,031 frames; its spellF1 is lower than R010's (0.4000 against 0.4686) by about one
  possession's worth. It was not used to choose anything.
- **Nothing was added after the results**: the weight axis is inside its grid and the scope is binary; neither a weight
  between the grid's points nor above 0.5 was tried, and `min_prob` was fixed beforehand (T26).
- **Seven matches from two competitions** are not seven independent draws; p 0.0156 is the smallest an exact test over
  7 matches can give, and every one of the 48 cells has it.

### R012 — P02's edge shift: the pipeline's own lag, the gap to our convention, and the rule's total (2026-10-25, Tian)
Proposal: P02 · Commit: `1ccaeb5` (the latency tables, the shift table and the first run of `main.toml`, all on R011's
`main.toml`, whose sha256 is `8810afce...`), `f104a0a` (the guard's config and its four runs) and the commit that
holds this entry (it adds `pipeline/configs/p02_shift.toml`, switches `main.toml` and logs the runs after the rule) ·
Command: `python pipeline/run.py pipeline/configs/main.toml` (at `1ccaeb5`), `python pipeline/benchmark/conventions.py
latency pipeline/configs/main.toml public`, `... latency pipeline/configs/main.toml dfl` (profile `tracking`, truth
raw, seeds 0, 1, 2) and `... dfl --truth coarse`, `... shift --config pipeline/configs/main.toml`; the guard, four
processes, each `python pipeline/benchmark/run.py pipeline/configs/main.toml
pipeline/benchmark/configs/p02/p02_sb.toml --profile tracking --seeds 3 --truth raw`, and the same with `--truth
coarse`, with `--profile context --truth raw` and with `--profile context --truth coarse`; after the rule, `python
pipeline/run.py pipeline/configs/p02_shift.toml --score` and again without `--score`, `python
evaluation/run_evaluation.py --results <a folder outside the repository holding three files: the in-sample file, the
out-of-fold file and R011's `main.csv`> --no-detail` (they are scored side by side), and `python pipeline/run.py
pipeline/configs/main.toml` twice, after the switch. Every log ends `EXIT=0`.

A later change of the tooling (the final review of P07 and P02): the latency files that these commands wrote are for
R011's `main.toml`, which is now `p07_shape.toml`, and `main.toml` itself carries the shift. A latency file records a
sha256 of the config's content (scope, seed, evidence, decoder and refine), and `conventions.py shift` refuses a file
of another content, or one written before the files carried it, with the `latency` command to run again. So R012's own
files are refused as they stand, and the rule can be reproduced with `--config pipeline/configs/p07_shape.toml` (its
commands are in the technical document, section 7).

The rule was declared before any of these runs and is applied as written. For each type of change, s_b is minus the
rounded median lag of the current `main.toml` on DFL (`tracking`, raw truth), and 0 below 30 matched changes. The
benchmark guard keeps it only if `main.toml` plus `shift_edges(s_b)` beats `main.toml` on `tracking` with the raw truth
(Δ > 0) and no cell of {`tracking`, `context`} x {raw, coarse} is below -0.005; one config holds all three values, so
they are kept or dropped together. Then g, the gap to our truth's convention, is minus the rounded median of the public
latencies after s_b, for a type only if that median is at least 3 frames in absolute value, its 90% bootstrap interval
(2,000 draws) excludes 0 and every leave-one-out median has its sign; the shift is s_b + g. Nothing in it is tuned.

Public (a correctness check only, T01), `p02_shift.toml` in sample (every boundary moved by the shift): macroF1 0.7159
· spellF1 0.4000 · alive/dead F1 0.9221 · transF1 0.1463 · lat_f +5.0 · 20 possessions against the truth's 17, mean
14.1 s. The same file out of fold (T27: a matched boundary moves by s_b plus the g that the other changes of its type
give, an unmatched one by the shift; sha256 `1078b527d4eae9dda5d4bcb2da77c249a699fb3d62928cb815a7a524f3193d7e`, scored
by the official scorer): macroF1 0.7125 · spellF1 0.4000 · transF1 0.1463 · lat_f +5.0 · 20 possessions, mean 14.1 s.
R011's `main.toml` on the same window: macroF1 0.7151 · spellF1 0.4000 · alive/dead F1 0.9218 · transF1 0.1463 · lat_f
+5.3 · 20 possessions. The three have the same spellF1: one possession moves it about 0.06 here (T01), and these are
the same 20. `p02_shift.toml` is valid (199,319 rows), sha256
`3fd06da011bde39f771ddd22f00da2bcbcf5205720d60f87662ab6fc0f76a702`, the same on two runs, and `main.toml` after the
switch writes the same file. No bootstrap was run, because no public-window gap is claimed. The in-sample and the
out-of-fold scores differ (macroF1 0.7159 and 0.7125), so leaving out one public change gives, for some type, a g
other than 0, though the full set does not: that is T27's reason to read a shift out of fold, and one more reason not
to trust a g from this window.
Validation (P01 benchmark): 7 DFL matches x 3 degradation seeds (0, 1, 2), `p02_sb` (`main.toml` plus `shift_edges`
team +1, into_dead -2, out_of_dead 0) against `main.toml`, pooled spellF1, in four cells:

| profile | truth | `main` | `p02_sb` | pooled Δ | mean per match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|
| `tracking` | raw: the guard's | 0.5159 | 0.5164 | +0.0005 | +0.0004 [-0.0007, +0.0015] | 0.5000 | 4 / 3 |
| `tracking` | coarse | 0.5591 | 0.5592 | +0.0000 | +0.0000 [-0.0004, +0.0004] | 1.0000 | 3 / 2 |
| `context` | raw: sensitivity only | 0.5153 | 0.5162 | +0.0009 | +0.0008 [-0.0004, +0.0021] | 0.2500 | 4 / 3 |
| `context` | coarse: sensitivity only | 0.5544 | 0.5544 | +0.0001 | +0.0001 [-0.0007, +0.0008] | 0.9531 | 4 / 3 |

Pooled macroF1 is +0.0004, +0.0003, +0.0004 and +0.0003 in the same four cells (the 90% interval is above 0 in each,
with 7, 6, 7 and 6 wins in 7 matches) and pooled alive/dead F1 goes from 0.9729 to 0.9730 on `tracking` and from
0.9699 to 0.9700 on `context`. The shift moves a team flip by 1 frame and a change into `D` by 2: the per-frame metric
sees that and the ranking metric (possessions matched at IoU 0.3 to 0.7) hardly does. The predicted `D` share goes
from 44.5% to 44.6% on `tracking` and from 44.7% to 44.9% on `context` (the truth's 44.8%); possessions per graded
minute do not change (3.17 and 3.15). The `main` rows are R011's to the last digit in all four cells (0.5159 and
0.5591 on `tracking`, 0.5153 and 0.5544 on `context`).
Changed vs the previous best: the previous best is R011's `main.toml`: 0.5159 raw and 0.5591 coarse on `tracking` v3,
0.5153 and 0.5544 on `context`; public 0.7151 macroF1, 0.4000 spellF1, 20 possessions; sha256
`8810afce254e94fe159769f3b55e8eef65f41a25c84f4550034e2afbc20e8487`. The shifted config is +0.0005 raw and +0.0000
coarse above it on `tracking` (pooled), +0.0004 raw macroF1, and on BAR-ATH it relabels 0.34% of the graded frames
(below).
Verdict: **the rule's total is team +1, into `D` -2, out of `D` 0 frames, which is s_b alone, and it is adopted**, by
the declared rule and not because it is shown to help: the guard keeps s_b (the `tracking` raw Δ is +0.0005 > 0 and no
cell is below -0.005), g is 0 for every type, and the label-free checks pass. **`main.toml` now carries `[[refine]] kind
= "shift_edges"` with team 1, into_dead -2, out_of_dead 0**, and writes sha256
`3fd06da011bde39f771ddd22f00da2bcbcf5205720d60f87662ab6fc0f76a702` (R011's was `8810afce...`). It has NOT been
uploaded.

**The latency of the pipeline on DFL** (`conventions.py latency`; `main.toml` as of R011; profile `tracking` v3, the
raw truth in the first three rows and the coarse one beneath; pooled over 7 matches and 3 seeds; a predicted change
matches a true one of the same pair of labels within 50 frames; predicted - true in frames, positive = our label
changed late; 25 frames = 1 s):

| truth | type | true changes | matched | median | p25 | p75 | median, s |
|---|---|---|---|---|---|---|---|
| raw | team | 7,053 | 43% | -1.0 | -10.0 | +20.0 | -0.04 |
| raw | into `D` | 2,130 | 50% | +2.0 | +0.0 | +2.0 | +0.08 |
| raw | out of `D` | 2,097 | 79% | +0.0 | -3.0 | +0.0 | +0.00 |
| coarse (reported only) | team | 2,847 | 58% | +0.0 | -10.0 | +22.0 | +0.00 |
| coarse | into `D` | 2,130 | 50% | +2.0 | +0.0 | +2.0 | +0.08 |
| coarse | out of `D` | 2,097 | 79% | +0.0 | -3.0 | +0.0 | +0.00 |

Per match, the median lag in frames (matches in the order J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9).
Raw: team -1, -1, -1, -1, +2, -1, -1; into `D` +2, +1, +2, +2, +1, +2, +1; out of `D` 0, 0, -2, 0, 0, -3, 0. Coarse:
team -1, -1, 0, -1, +3, 0, 0, and the two `D` rows are the raw ones. The matched changes behind each raw median are
3,011 (team), 1,069 (into `D`) and 1,657 (out of `D`), all above the rule's 30. A team flip's lag is wide (an
interquartile range of 30 frames, and only 43% of the flips find a partner within 50 frames), so its median (-1) is
the middle of a spread-out lag, and the shift of +1 that undoes it moves the middle and not the spread; the `D` edges
are tight (our `D` begins 2 frames, 0.08 s, after DFL's flag, and ends on it).

**On the public window** (`conventions.py latency ... public`; the 20 changes of the public truth; with fewer than 5
matched, n and the median only, no quartile): team 12 changes, 33% matched (4), median +22.0 frames; into `D` 4 changes,
25% matched (1), +16.0; out of `D` 4 changes, 25% matched (1), -12.0. These are the pipeline's lags against our truth at
4, 1 and 1 matched changes, where DFL's are -1, +2 and 0.

**The rule's table** (`conventions.py shift`; L' = the public latency after s_b):

| type | DFL matched | DFL median | s_b | public matched | median L' | interval excludes 0 | size / interval / leave-one-out | g | shift |
|---|---|---|---|---|---|---|---|---|---|
| team | 3,011 | -1.0 | +1 | 4 | +23.0 | no | yes / no / yes | 0 | +1 |
| into `D` | 1,069 | +2.0 | -2 | 1 | +14.0 | yes | yes / yes / no | 0 | -2 |
| out of `D` | 1,657 | +0.0 | 0 | 1 | -12.0 | yes | yes / yes / no | 0 | 0 |

Under the coarse truth, beside the raw one and never used, s_b would be team 0, into `D` -2, out of `D` 0: the one place
the two truths disagree is the team flip, +1 against 0 (T33). For the two `D` types the interval "excludes 0" is the
bootstrap of a single lag and says nothing, and the leave-one-out condition fails there (one change has no leave-one-out
median); for the team flip the interval includes 0. **g is 0 for every type**: no type of change has enough public
transitions behind it for the rule to move a boundary on the public window's evidence.

**Label-free checks on BAR-ATH** (`main.toml` after the switch, `--score` on `p02_shift.toml`): no `X` in a graded
block (X 0.0%; H 37.5%, A 26.6%, D 36.0%, against R011's 37.6%, 26.6%, 35.8%), the same sha256 on two runs, and 439
possessions in 105.905 graded minutes (4.1452 per graded minute, mean 9.3 s), as R011's `main.toml` (439; 4.1452). The
labels differ from R011's on 545 of the 158,857 graded frames (0.34%), exactly the moved boundaries (333 team changes
by 1 frame and 106 changes into `D` by 2 frames; 333 + 212 = 545); the graded halves still hold 333, 106 and 105 such
changes, so no boundary crossed another. `controlled_touch` agrees with the public GT on home on 87.2% of 5,489
frames, `dead_ball_model` on alive on 86.6% of 7,500 and `shape_possession` on home on 90.9% of 6,031, as in R011:
neither source changed. `pipeline/selftest.py --data` passes (153 of 153 checks, `main.toml` run twice and compared
byte for byte, the four baselines rebuilt byte for byte) and `pipeline/benchmark/selftest.py` passes (369 checks);
`evaluation/validate_submission.py` accepts `evaluation/results/main.csv`.

**Be plain about the size.** The shift is 1 to 2 frames at 439 boundaries (333 team flips and 106 changes into `D`) of
a match of about 106 graded minutes, and its benchmark gains are +0.0000 to +0.0009 pooled spellF1 (on `tracking` and
`context`, raw and coarse truth), with every 90% interval including 0 and the raw `tracking` cell's sign-flip p at
0.5000 (4 wins and 3 losses in 7 matches): inside the noise (0.04 on a full match, and the paired intervals here are
0.0008 to 0.0025 wide). It is adopted because the declared rule says so (a positive raw Δ on `tracking`, no cell under
-0.005), not because it is shown to help. The only consistent effect is on the per-frame metrics, +0.0003 to +0.0004
macroF1 in all four cells, and macroF1 breaks ties; it does not rank. Dropping the `[[refine]]` entry returns R011's
file (sha256 `8810afce...`).

**The limits, and what the numbers do not say.**
- **s_b is DFL's lag, not ours.** It measures the pipeline on a degraded copy of DFL's tracking against DFL's flag, and
  it says our `D` onset is 2 frames late there and our team flip 1 frame early. Our own truth's convention is what g was
  for, and the public window cannot say it (A11, T40).
- **The public lags are large and unused.** Against our truth the pipeline is 22 frames late at a team flip (4 matched
  changes), 16 late into `D` (1) and 12 early out of `D` (1). At 4, 1 and 1 changes the rule cannot tell a convention
  from chance, and it is right not to. A larger gap needs more transitions (P16's sample games) or a person watching the
  20 changes (A11), and not a looser rule.
- **The team shift depends on the truth.** +1 on the raw truth, 0 on the coarse one (T33), and the guard's two truths
  agree on it only in being within noise.
- **One integer per type.** The shift is not a function of anything (the ball's visibility, the stoppage type), and it
  does not narrow the lag's spread (an interquartile range of 30 frames at a team flip).
- **The public window is a correctness check only.** It chose nothing here: s_b comes from DFL and g is 0 everywhere.
  Its spellF1 is R011's (0.4000) in sample and out of fold.

*Later (23 Oct rule, `f8f2629`):* the declared rule read the public window as an input in one place, g, the gap to our
convention taken from the public latencies. That is a tuning use of a window that is "not training data and not a tuning
set", so **g is retired**: it stays 0 by rule, and `conventions.py shift` prints it for reference only. Nothing in the
adopted file changes: `main.toml`'s shift (team +1, into `D` -2, out of `D` 0) is s_b alone, the pipeline's own lag on
the seven DFL matches (other games), kept by the benchmark guard, and g was 0 for every type, so sha256 `3fd06da0...` is
the same. The public latencies and the out-of-fold public score above stay as the record, read as an inspection and a
check, and nothing is chosen from them. Any future shift comes from other games only (P16).

### R013 — The touch rule and the decoder re-chosen without the window's values, with P07's weight (2026-10-26, Tian)
Proposal: P04, P06, P07 · Commit: part 1 (this declaration) is committed before any run of it; the commits of the runs
are in part 2.

**Part 1, declared before any run** (2026-10-26). Nothing below was chosen from a score of this entry.

**Why.** Two items of [After P07 and P02](proposals.md#after-p07-and-p02):
- *The decoder with the new evidence* (item 1). R011 kept R010's decoder (switch_team 4, switch_dead 32,
  min_possession 25, min_dead 25) and chose P07's weight against it, and R010's switch_team 4 and min_possession 25
  are the lowest values it tried (raw spellF1 falls from both). P07's weight peaked inside its grid (0.2), which is
  what a weight trading off against the switch penalties looks like, and R011 cannot tell the two apart.
- *The touch constants without the window's values* (item 1, its last bullet, and the 23 Oct rule, `f8f2629`).
  R008 chose `controlled_touch`'s radius, `max_rel_speed`, `min_frames` and `hold` in configs that carried
  `ball_presence` at A01's values, which were read off the public window, and the hmm (8, 32, 75, 1). `main.toml` has no
  such source, so this protocol re-chooses them on `main.toml`'s own evidence, where every constant comes from DFL, the
  benchmark or a fixed rule.

**What is not re-chosen, and why.**
- *R009's source and weight* (the `hgb` `dead_ball_model` at weight 1.0; `nearest_player` at A01's 0.78 was present when
  it was chosen). The source rests on the out-of-fold log-loss, lower for `hgb` on 7 of 7 matches, which no other
  source enters. The weight 1.0 takes the model's probability as it is (an exponent of 1), and the benchmark overstates
  this source for BAR-ATH (T38, T39), so a grid over its weight would lean towards a larger one for a reason that does
  not transfer. Both stay.
- *`switch_dead` and `min_dead`*, fixed at 32 and 25: R010 found raw spellF1 flat along both (switch_dead 16 to 64
  within 0.0027 at switch_team 4 and min_possession 25; the two `min_dead` values within 0.0003 in all 36 pairs).
- *Smaller things*: `controlled_touch`'s `duel_radius` (1.0 m, never tuned), `shape_possession`'s scope `all` (above
  `gaps` at every weight on `tracking`, R011) and `min_prob` 0.02, and the trained models, all as in `main.toml`.
- *`controlled_touch`'s `confidence`* is not an axis: for each setting it is `touch_stats.py`'s rule, the floor of the
  pooled agreement to a hundredth on `tracking` v3, 3 seeds, all seven matches (R008), measured once before Stage A
  runs and written into its configs. R008's setting (2.0 m, 6.0 m/s, 3 frames) should measure 0.91 again, a check.

**Stage A, the touch constants.** `main.toml` verbatim (`dead_ball_model` `hgb` at 1.0, `shape_possession` at 0.2,
scope `all`, the hmm 4 / 32 / 25 / 25, `shift_edges` team +1, into_dead -2, out_of_dead 0), with `controlled_touch`
over radius {1.5, 2.0, 2.5, 3.0} m x `max_rel_speed` {6, 9, 12} m/s x `min_frames` {1, 2, 3, 5} x `hold` {0.52, 0.55,
0.60, 0.70}: 192 configs `pipeline/benchmark/configs/r013/r013a_r{RR}_v{VV}_k{K}_h{HH}.toml` (RR the radius in
decimetres, VV the speed, K the frames, HH the hold in hundredths), each at its setting's measured `confidence`. The
values extend R008's edges: its choice sat on the upper edge in radius (2.0) and speed (6.0), on the lower edge in
`min_frames` (3) and at the only `hold` other than off (0.55). Radius 1.5 stays (R008's raw best); the values R008 found
worse by a margin (radius 1.0, speed 3.0, 8 frames, hold off) are not run again. A setting whose `confidence` is below a
`hold` drops that config (the source refuses it), and the entry says so. `r013a_r20_v06_k3_h55` is `main.toml`'s own
config and must score as `main.toml` does, to the last digit.

**Stage C, the decoder and P07's weight**, run after Stage A is chosen: `main.toml` with Stage A's `controlled_touch`,
over switch_team {1, 2, 3, 4, 6, 8} x min_possession {1, 10, 15, 25, 35, 50} frames x `shape_possession`'s weight
{0.1, 0.2, 0.3, 0.5}: 144 configs `pipeline/benchmark/configs/r013/r013c_t{TT}_mp{MMM}_w{WWW}.toml` (WWW the weight
in thousandths, as R011's names). Both sides of R010's lower edges are covered (switch_team below 4 and min_possession
below 25), and the upper side too, because P07 raised the benchmark's predicted possessions (2.74 to 3.17 a minute,
R011). The weight's axis is R011's peak and its neighbours, with 0.3 between 0.2 and 0.5.

**The runs.** Profile `tracking` v3, 3 seeds, all seven matches; every config decoded once and scored under the raw and
the coarse truth (`run.py --truth both`, added for this entry: it scores the same codes against both truths and writes
the two reports that two runs would, which a check holds it to). Each process lists `main.toml` first (the reference of
every Δ), then `nearest_w50` and `nearest_w125` (they and `main.toml` must score as in R004c and R012, to the last
digit), then a share of the grid; processes run one at a time (one heavy process on this machine). The reports are
written, and the selection is computed from them by `pipeline/benchmark/select.py` (added for this entry: the rule
below and the leave-one-match-out estimate, from the reports alone).

**The selection, the same in both stages** (R008 to R011's rule). The objective is pooled spellF1 on `tracking`, raw
truth. The tie band is 0.005: every config within it of the best is tied, and among the tied the best pooled `tracking`
coarse spellF1 wins. Further ties: in Stage A the larger `min_frames`, then the smaller radius, the smaller
`max_rel_speed` and the smaller `hold`; in Stage C the larger switch_team, then the larger min_possession and the
smaller weight. **Nothing is added after the results**: a choice on an edge of its grid is reported as one, and the
grid is not extended in this entry.
- *Leave-one-match-out*, per stage, as R008 to R011: for each match, the selection repeated on the other six (the raw
  objective alone, and the whole rule on six-match means), and the chosen config's score on the held-out match against
  `main.toml`'s; mean Δ, 90% t-interval, exact sign-flip p, wins in 7. It does not cover the confidences (measured on
  all seven) or the two stages run in order (Stage C runs only at the full-data Stage A choice).

**Stage D, a diagnostic that chooses nothing.** The five best Stage A settings by raw spellF1 are run again at Stage C's
decoder and weight, under both truths: whether the touch choice depends on the decoder it was made at. It is reported
and not acted on.

**The shift, by R012's rule, applied to the new pipeline.** If the final config differs from `main.toml` anywhere before
the refiner, the lag of its form without `shift_edges` is measured (`conventions.py latency <config> dfl`, `tracking`,
raw truth, 3 seeds), s_b is minus the rounded median lag per type and 0 below 30 matched changes, and the guard keeps
s_b only if the config with it beats the config without it on `tracking` raw (Δ > 0) and no cell of {`tracking`,
`context`} x {raw, coarse} is below -0.005; otherwise there is no shift. g stays 0 (retired, `f8f2629`).

**Adoption**, the final config (Stage C's choice with the shift that rule gives) against `main.toml` as it is now (R012,
sha256 `3fd06da0...`):
- *The guard* (R012's form): pooled Δ > 0 on `tracking` raw, and no cell of {`tracking`, `context`} x {raw, coarse}
  below -0.005. `context` is a sensitivity profile and a veto here, never a validation (T38).
- *The bar*, reported and not required: `tracking` raw Δ > 0 with sign-flip p <= 0.05 says the change is shown to help.
  The entry re-chooses existing constants by a clean protocol, and as in R012 an adoption inside the noise is the rule's
  and is called that.
- *Label-free checks on BAR-ATH* (no ground truth): the predicted `D` share of the graded frames within [30%, 62%]
  (R010's band), no `X` in a graded block, the file valid and the same sha256 twice. Reported beside them: the
  possessions per graded minute (`main.toml`: 4.1452), the H / A / D shares, the share of graded frames relabelled and
  `controlled_touch`'s opinion share of the graded frames (R008's rule: 59.2%).
- If everything passes, `main.toml` becomes the final config and `pipeline/configs/r013_retune.toml` is written with it.
  If the final config is `main.toml` itself, or the guard or a check fails, `main.toml` is unchanged, and the entry says
  whether `main.toml`'s own values sit inside each stage's tie band, which is the re-confirmation the 23 Oct rule asks
  for either way.

**The public window** is a correctness check only (the file valid, deterministic, `--score`'s agreements). It chooses
nothing, and it is run once, after the adoption is decided.

**What the entry cannot settle.** The grain of our truth (T33: the coarse truth is the tie-break and the guard's
warning, not a second objective); the benchmark's overstatement of P05 (T38, T39); the shape model never having seen our
teams (R011). A gain here is the benchmark's, and none of it is a forecast for BAR-ATH.

**Cost**, from R010 and R011's timings (about 25 s a config once the first config of a process has paid its one-off
pass of about 150 s): Stage A about 85 minutes, Stage C about 65, the shift's runs and the `context` cells about 15.

**Part 2, after the runs** (2026-10-26). Everything below applies part 1 as written.

**The confidences** (`python pipeline/benchmark/touch_stats.py --grid r013 --profile tracking --seeds 3 --out
data/runs/r013/touch_stats_r013.json`, EXIT=0, 7 min; the code is `c086d23`'s, run before it was committed and not
changed since). They run from 0.86 (radius 3.0 m, 12 m/s) to 0.93 (radius 1.5 m, 6 m/s): agreement falls and coverage
rises with the radius and the speed, as in R008, and a shorter minimum episode changes neither by more than a point.
**R008's setting (2.0 m, 6 m/s, 3 frames) measures 0.91 again (91.9%), the check part 1 asked for.** Every `hold` of
the grid is below every confidence, so no config was dropped. The flip lags are as R008's: an episode of the new team,
when there is one within the window, starts a median of 0 to 5 frames before the flag (R008's sign convention), found
for 44% to 63% of the changes.

**Stage A's runs.** Commit `c086d23` (the tooling and the 192 configs), run from a detached checkout of it; the
runner's header says `c086d23db873` and no `+dirty`. Four processes, one per radius, one at a time:
`python -u pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml
pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r013/r013a_r{RR}_*.toml --profile tracking
--seeds 3 --truth both --data-dir <the repository's data/>`. All four logs end `EXIT=0` (20 to 23 minutes each, 16:12
to 17:38), and the 384 reports are there. Claude Code's memory-pressure reaper stopped the shell that chained the
processes at about 17:35, during the fourth; the process itself went on and finished, and nothing was rerun.
**The checks on the runs pass**: in every log `main` scores 0.5164 raw and 0.5592 coarse (R012's, to the last digit),
`nearest_w50` 0.2018 raw and `nearest_w125` 0.2226 coarse (R004c's), and `r013a_r20_v06_k3_h55`, `main.toml`'s own
setting, scores exactly as `main` under both truths (so `--truth both` reproduces two single-truth runs on the real
benchmark as well as in its check). The choice: `python pipeline/benchmark/choose.py --reference main <the 192 stems in
the declared tie order> --top 25 --out data/runs/r013/stage_a_choice.json`.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | possessions / min | raw Δ vs `main` [p, wins] | coarse Δ [p, wins] |
|---|---|---|---|---|---|---|
| `main` (R012) | 0.5164 | 0.5592 | 0.8865 | 3.17 | | |
| **`r013a_r15_v06_k1_h60`** | **0.5473** | 0.5485 | 0.8900 | 3.27 | +0.0310 [0.0156, 7] | -0.0107 [0.0156, 0] |
| `r013a_r15_v09_k1_h60` | 0.5443 | 0.5214 | 0.8889 | 3.52 | +0.0279 [0.0156, 7] | -0.0377 [0.0156, 0] |
| `r013a_r20_v06_k1_h60` | 0.5441 | 0.5454 | 0.8892 | 3.32 | +0.0277 [0.0156, 7] | -0.0137 [0.0156, 0] |
| `r013a_r15_v09_k2_h60` | 0.5425 | 0.5214 | 0.8887 | 3.51 | +0.0261 [0.0156, 7] | -0.0378 [0.0156, 0] |
| `r013a_r15_v06_k1_h70` | 0.5415 | 0.5251 | 0.8865 | 3.43 | +0.0252 [0.0156, 7] | -0.0341 [0.0156, 0] |
| `r013a_r20_v06_k1_h70` | 0.5406 | 0.5244 | 0.8861 | 3.46 | +0.0242 [0.0156, 7] | -0.0347 [0.0156, 0] |
| `r013a_r15_v06_k1_h55` | 0.5403 | 0.5662 | 0.8908 | 3.19 | +0.0240 [0.0156, 7] | +0.0070 [0.1250, 5] |
| `r013a_r25_v06_k1_h60` | 0.5399 | 0.5386 | 0.8884 | 3.37 | +0.0235 [0.0156, 7] | -0.0206 [0.0156, 0] |
| `r013a_r15_v06_k2_h60` | 0.5393 | 0.5474 | 0.8885 | 3.24 | +0.0230 [0.0156, 7] | -0.0118 [0.0156, 0] |
| `r013a_r20_v06_k3_h55` (`main`'s setting, rank 83) | 0.5164 | 0.5592 | 0.8865 | 3.17 | +0.0000 [1.0000, 0] | +0.0000 [1.0000, 0] |
| `r013a_r15_v06_k5_h55` (the best coarse) | 0.4986 | 0.5751 | 0.8836 | 2.97 | -0.0178 [0.0156, 0] | +0.0159 [0.0156, 7] |

The full table (192 rows, both truths, macroF1, alive/dead F1, `D` share) is what the command prints; alive/dead F1 is
0.9730 to 0.9731 and the `D` share 44.6% to 44.7% for every config (the touch rule has no opinion on `alive`), and the
predicted possessions per graded minute run from 2.96 to 4.15 (the truth's 4.58 raw, 2.48 coarse).

**The choice, by the rule.** The best raw config is `r013a_r15_v06_k1_h60`, 0.5473; the tie band takes four configs
(raw >= 0.5423): it, `r15_v09_k1_h60` (0.5443 raw, 0.5214 coarse), `r20_v06_k1_h60` (0.5441, 0.5454) and
`r15_v09_k2_h60` (0.5425, 0.5214). The best coarse among them is the raw best itself, so **Stage A's choice is radius
1.5 m, `max_rel_speed` 6.0 m/s, `min_frames` 1, `hold` 0.60, at confidence 0.93.** Against `main`: +0.0310 pooled raw
(mean per match +0.0306 [+0.0242, +0.0370], p 0.0156, 7 wins), and **-0.0107 pooled coarse (mean -0.0107 [-0.0178,
-0.0036], p 0.0156, 0 wins in 7)**. By competition the raw gain is +0.0380 (DFL-COM-000001) and +0.0276
(DFL-COM-000002).
- *Leave-one-match-out.* The raw objective alone picks the choice in six folds and `r15_v09_k1_h60` in one (J03WOY):
  held out +0.0267 [+0.0161, +0.0373] raw, p 0.0156, 7 wins, and -0.0157 coarse, 0 wins. The whole rule picks the
  choice in all seven folds, so its estimate is the choice's own (+0.0306 raw, -0.0107 coarse). The selection is
  stable; the estimate does not cover the confidences, measured on all seven matches.
- *The truths disagree (T33), and the hold is where.* At radius 1.5 m, 6 m/s and 1 frame the raw score peaks at hold
  0.60 (0.5256, 0.5403, 0.5473, 0.5415 for 0.52, 0.55, 0.60, 0.70) and the coarse score at 0.52 to 0.55 (0.5677,
  0.5662, 0.5485, 0.5251): a stronger hold cuts the possessions finer (3.15, 3.19, 3.27, 3.43 a minute). Across the
  grid the raw truth wants more possessions and the coarse truth fewer: the coarse winner (`r15_v06_k5_h55`, 0.5751) is
  0.0487 under the raw best, and the raw winner is 0.0266 under the coarse best. `r15_v06_k1_h55` is above `main` under
  both truths (+0.0240 raw, +0.0070 coarse, p 0.1250), but 0.0070 under the raw best, outside the band, so the rule
  does not reach it.
- *At the edges.* The choice is on the grid's lower edge in radius (1.5), in `max_rel_speed` (6.0) and in `min_frames`
  (1, the smallest there is), and inside it in `hold`. By axis, the best raw score falls from the lower edge: radius
  0.5473, 0.5441, 0.5399, 0.5360 (1.5 to 3.0 m); speed 0.5473, 0.5443, 0.5320 (6, 9, 12 m/s); minimum frames 0.5473,
  0.5425, 0.5362, 0.5166 (1, 2, 3, 5). R008 found radius 1.0 and speed 3.0 worse than 1.5 and 6.0 in its context; they
  were not run again, and per part 1 nothing is added.
- *`main.toml`'s own setting ranks 83rd of 192 by raw spellF1 (16th by coarse)*, 0.0310 under the best and far outside
  the band: **re-chosen without the window's values, the touch rule would not be R008's.** 82 configs beat it raw and
  15 coarse.

**Stage C's runs.** Commit `7000847` (the 144 configs at Stage A's choice; `r013c_t04_mp025_w200` is Stage A's choice
itself), from a detached checkout of it (header `7000847...`, no `+dirty`). The first process (switch_team 1 and 2, 96
configs and the three references) ended `EXIT=0` at 18:01. **The computer crashed at 18:04 during the second** (a
Windows bugcheck 0x133, DPC_WATCHDOG_VIOLATION, a driver stalling; the same bugcheck at the same code offset was logged
on 13 Oct, before the benchmark existed, and at 00:51 on 26 Oct; no low-memory event preceded it). The second process
wrote no report and was rerun from the start as four throttled processes, one switch_team value each (3, 4, 6, 8):
`OPENBLAS_NUM_THREADS=1` (and OMP, MKL), below-normal priority, a free-memory guard and a 90 s pause between processes.
All four ended `EXIT=0` (18:22 to 18:54). **The one-thread runs give the same numbers**: `main.json` and
`main.coarse.json` written by them are byte-identical to the first process's (every half's counts), and
`r013c_t04_mp025_w200` scores as Stage A's `r013a_r15_v06_k1_h60` to the last digit (0.5473 raw, 0.5485 coarse). The
references score as in Stage A in every log. The choice: `python pipeline/benchmark/choose.py --reference main <the 144
stems in the declared tie order> --top 30 --out data/runs/r013/stage_c_choice.json`.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | possessions / min | raw Δ vs `main` [p, wins] | coarse Δ [p, wins] |
|---|---|---|---|---|---|---|
| `main` (R012) | 0.5164 | 0.5592 | 0.8865 | 3.17 | | |
| `r013c_t03_mp025_w200` (the best raw) | 0.5477 | 0.5336 | 0.8895 | 3.40 | +0.0313 [0.0156, 7] | -0.0256 [0.0156, 0] |
| `r013c_t04_mp025_w200` (Stage A's choice) | 0.5473 | 0.5485 | 0.8900 | 3.27 | +0.0310 [0.0156, 7] | -0.0107 [0.0156, 0] |
| `r013c_t02_mp025_w200` | 0.5462 | 0.5185 | 0.8890 | 3.55 | +0.0299 [0.0156, 7] | -0.0407 [0.0156, 0] |
| `r013c_t04_mp015_w200` | 0.5453 | 0.5357 | 0.8899 | 3.42 | +0.0289 [0.0156, 7] | -0.0235 [0.0156, 0] |
| `r013c_t03_mp015_w200` | 0.5452 | 0.5215 | 0.8895 | 3.55 | +0.0288 [0.0156, 7] | -0.0377 [0.0156, 0] |
| `r013c_t02_mp035_w200` | 0.5445 | 0.5320 | 0.8888 | 3.38 | +0.0282 [0.0156, 7] | -0.0272 [0.0156, 0] |
| `r013c_t03_mp035_w200` | 0.5443 | 0.5456 | 0.8892 | 3.24 | +0.0279 [0.0156, 7] | -0.0135 [0.0156, 0] |
| `r013c_t02_mp015_w200` | 0.5440 | 0.5058 | 0.8892 | 3.72 | +0.0276 [0.0156, 7] | -0.0534 [0.0156, 0] |
| **`r013c_t04_mp035_w200`** | **0.5430** | **0.5593** | 0.8895 | 3.13 | +0.0267 [0.0156, 7] | +0.0001 [0.9844, 4] |
| `r013c_t04_mp010_w200` | 0.5428 | 0.5310 | 0.8901 | 3.47 | +0.0265 [0.0156, 7] | -0.0282 [0.0156, 0] |
| `r013c_t06_mp025_w200` | 0.5399 | 0.5634 | 0.8897 | 3.09 | +0.0236 [0.0156, 7] | +0.0042 [0.4531, 5] |
| `r013c_t08_mp050_w500` (the best coarse) | 0.5134 | 0.6017 | 0.8873 | 2.79 | -0.0030 [0.5000, 2] | +0.0425 [0.0156, 7] |

The first ten rows are the tie band (raw >= 0.5427) in raw order. Alive/dead F1 is 0.9728 to 0.9731 and the `D` share
44.6% to 44.7% throughout; 130 of the 144 configs are above `main` raw, and 39 are above it under both truths.

**The choice, by the rule.** The best raw config is `r013c_t03_mp025_w200`, 0.5477; ten configs are within 0.005 of
it, all at weight 0.2, switch_team 2 to 4 and min_possession 10 to 35. The best coarse among them is
`r013c_t04_mp035_w200` (0.5593; the next, `t04_mp025`, 0.5485), so **Stage C's choice is switch_team 4, min_possession
35, weight 0.2** (switch_dead 32 and min_dead 25 fixed). Against `main`: **+0.0267 pooled raw** (mean per match +0.0264
[+0.0191, +0.0338], p 0.0156, 7 wins; by competition +0.0319 and +0.0243) and **+0.0001 pooled coarse** (mean +0.0001
[-0.0075, +0.0077], p 0.9844, 4 wins and 3 losses). It is 0.0047 under the raw best and 0.0257 above it coarse.
- *Leave-one-match-out.* The whole rule picks the choice in five folds and `t04_mp025_w200` in two (J03WPY, J03WR9):
  held out +0.0263 [+0.0187, +0.0339] raw, p 0.0156, 7 wins, and -0.0028 [-0.0119, +0.0064] coarse, p 0.5469, 3 wins
  and 4 losses. The raw objective alone picks `t03_mp025` or `t04_mp025` (four and three folds): +0.0296 raw, -0.0194
  coarse (0 wins). The coarse tie-break is what keeps the coarse score level with `main`'s, and it is less stable than
  Stage A's (five folds of seven).
- *Inside the grid.* The choice is interior on all three axes: switch_team 4 (1 to 8), min_possession 35 (1 to 50),
  weight 0.2 (0.1 to 0.5). Along each axis at the other two's values, raw spellF1 peaks at or next to the choice and
  the coarse score rises steadily towards fewer possessions: min_possession at switch_team 4, weight 0.2 gives raw
  0.5403, 0.5428, 0.5453, 0.5473, 0.5430, 0.5304 and coarse 0.5262, 0.5310, 0.5357, 0.5485, 0.5593, 0.5774 (1 to 50);
  switch_team at min_possession 35 gives raw 0.5409, 0.5445, 0.5443, 0.5430, 0.5335, 0.5220 and coarse 0.5156 to
  0.5837 (1 to 8); the weight at 4 / 35 gives raw 0.5328, 0.5430, 0.5358, 0.5283 (0.1 to 0.5), R011's peak again.
- *The truths still pull apart* (T33). The coarse best, `t08_mp050_w500` (0.6017), is the grid's corner towards the
  fewest possessions (2.79 a minute) and 0.0343 under the raw best; the raw best predicts 3.40 a minute. The choice's
  3.13 is `main`'s 3.17, between the truths' 4.58 (raw) and 2.48 (coarse): with Stage A's touch rule cutting
  possessions finer, the decoder's longer minimum (35 frames, 1.4 s) puts the count back where `main` had it.
- *Below R010's edge.* R010's choice (switch_team 4, min_possession 25) sat on its grid's lower edge. With the new
  evidence the raw score is flat below it (the raw best, switch_team 3 at min_possession 25, is 0.0004 above switch_team
  4 there) and falls at switch_team 1 and min_possession 1, while the coarse score falls at every value below the
  choice: the edge hid no clearly better decoder.

**The shift, by R012's rule.** Commit `05a57af` (Stage D's configs and `r013_preshift.toml`, Stage C's choice without
`shift_edges`), run throttled from a detached checkout of it: `python pipeline/benchmark/conventions.py latency
pipeline/benchmark/configs/r013/r013_preshift.toml dfl --profile tracking --seeds 3 --truth raw` (EXIT=0). The new
pipeline's lag on DFL, pooled over 7 matches and 3 seeds (predicted - true, in frames): team median -1.0 (3,116 matched
of 7,053 changes; -1 on every match), into `D` +2.0 (1,095 of 2,130), out of `D` +0.0 (1,541 of 2,097). So **s_b is
team +1, into `D` -2, out of `D` 0** (`conventions.pipeline_lag` on the file's lags), exactly R012's, and the final
config is Stage C's choice as it stands. The guard (with the shift against `r013_preshift`): `tracking` raw +0.0005
(mean per match +0.0005 [-0.0003, +0.0013], p 0.2500, 5 wins), `tracking` coarse +0.0000, `context` raw +0.0007,
`context` coarse -0.0004: Δ > 0 on `tracking` raw and no cell below -0.005, so **s_b is kept**, inside the noise as in
R012.

**The guard runs.** Two throttled processes from `05a57af` (header `05a57af01222`): `python pipeline/benchmark/run.py
pipeline/configs/main.toml <r013c_t04_mp035_w200, r013_preshift and the five r013d configs> --profile tracking --seeds 3
--truth both` and `python pipeline/benchmark/run.py pipeline/configs/main.toml <r013c_t04_mp035_w200, r013_preshift>
--profile context --seeds 3 --truth both`; both EXIT=0. `main` scores as R012's in all four cells (0.5164 and 0.5592 on
`tracking`, 0.5162 and 0.5544 on `context`), and `r013c_t04_mp035_w200` as in Stage C.

| profile | truth | `main` (R012) | final (`r013c_t04_mp035_w200`) | pooled Δ | mean per match Δ [90% t] | p | wins / losses |
|---|---|---|---|---|---|---|---|
| `tracking` | raw: the guard's and the bar's | 0.5164 | 0.5430 | +0.0267 | +0.0264 [+0.0191, +0.0338] | 0.0156 | 7 / 0 |
| `tracking` | coarse | 0.5592 | 0.5593 | +0.0001 | +0.0001 [-0.0075, +0.0077] | 0.9844 | 4 / 3 |
| `context` | raw: sensitivity only | 0.5162 | 0.5489 | +0.0327 | +0.0323 [+0.0227, +0.0419] | 0.0156 | 7 / 0 |
| `context` | coarse: sensitivity only | 0.5544 | 0.5642 | +0.0097 | +0.0097 [+0.0039, +0.0155] | 0.0469 | 6 / 1 |

Pooled macroF1 on `tracking`: 0.8895 raw (`main` 0.8865) and 0.8878 coarse (0.8895); alive/dead F1 0.9731 (0.9730); 3.13
possessions per graded minute (3.17); `D` share 44.6%.

**Stage D, the diagnostic.** Stage A's five best touch settings at Stage C's decoder and weight (`tracking`, both
truths): radius 1.5 m, 9 m/s, 2 frames, hold 0.60: 0.5433 raw, 0.5408 coarse; 1.5 m, 9 m/s, 1 frame: 0.5431, 0.5419;
**1.5 m, 6 m/s, 1 frame (the choice): 0.5430, 0.5593**; 2.0 m, 6 m/s, 1 frame: 0.5396, 0.5596; 1.5 m, 6 m/s, 1 frame,
hold 0.70: 0.5392, 0.5398. The five are within 0.0041 raw of each other at the new decoder, the choice 0.0003 under
the best, and the 9 m/s settings lose 0.017 to 0.018 coarse where the choice holds `main`'s level. So the touch choice
does not hang on the decoder it was made at; it chooses nothing.

**The label-free checks on BAR-ATH** (`python pipeline/run.py pipeline/configs/r013_retune.toml`, twice; aggregates
over the 158,857 graded frames, 105.905 minutes):

| | R012's `main.toml` (`3fd06da0`) | the final config (`0ebdfbec`) | declared rule |
|---|---|---|---|
| the file valid, the same sha256 twice | yes | yes (and the same under one BLAS thread) | yes |
| `X` frames inside the graded blocks | 0 | 0 | none |
| `D` share of the graded frames | 35.95% | 35.92% | within [30%, 62%] |
| H, A of the graded frames | 37.5%, 26.6% | 37.4%, 26.7% | reported |
| possessions (per graded minute) | 439 (4.1452) | 410 (3.8714), mean 9.9 s | reported |
| graded frames relabelled | | 2.39% (0.18 points alive/dead, 2.20 points team: 3.45% of the frames alive in both) | reported |
| `controlled_touch`'s opinion share of the graded frames | 59.2% | 59.2% | reported |

All three checks pass. The opinion share is the same because with a hold the source speaks on every frame with the
ball visible (59.3% of the graded frames) after the half's first episode, so it says nothing of the rule; what does is
the share of the ball-visible graded frames inside a kept episode: **36.9% for the new rule against 43.6% for R008's**
on BAR-ATH. On the benchmark the two cover 43.4% and 47.6% of the alive ball-visible frames (touch_stats), so the
tighter rule loses more of its episodes on our match (a ratio of 0.85 to the benchmark's coverage, against 0.92 for
R008's rule). BAR-ATH's figure counts dead-ball frames too (the ball is visible on 66% of them, A01), so the two are not
the same statistic; it is a label-free sign that the 1.5 m radius transfers less well than 2.0 m did, and the benchmark
cannot say what that costs. The possessions move the same way on both (fewer: 3.17 to 3.13 a minute on the benchmark,
4.15 to 3.87 on BAR-ATH), unlike R011's.

**The public window** (a correctness check only, T01; run once, after the adoption was decided: `python pipeline/run.py
pipeline/configs/r013_retune.toml --score`): macroF1 0.7152 · spellF1 0.4649 · alive/dead F1 0.9225 · 20 possessions
against the truth's 17 (R012's file: 0.7159, 0.4000, 20). `controlled_touch` agrees with the public GT on home on 86.8%
of 5,484 frames, `dead_ball_model` on alive on 86.6% of 7,500 and `shape_possession` on home on 90.9% of 6,031. It
chose nothing; one possession moves its spellF1 about 0.06.

Verdict: **adopted. `main.toml` is R013's final config**: `controlled_touch` radius 1.5 m, `max_rel_speed` 6.0 m/s,
`min_frames` 1, `hold` 0.60 at confidence 0.93; `dead_ball_model` `hgb` at 1.0; `shape_possession` at 0.2, scope
`all`; the hmm 4 / 32 / 35 / 25; `shift_edges` +1 / -2 / 0. It is `pipeline/configs/r013_retune.toml`, and `main.toml`
writes sha256 `0ebdfbeca72ae2d09eb7aaa377037415696a5dacb95e29003741e0f57d5dbc5d` (not uploaded). The guard passes (a
gain on `tracking` raw, no cell below -0.005), the bar is met as well (`tracking` raw +0.0267, p 0.0156, 7 wins in 7,
so on the benchmark the change is shown to help), the label-free checks pass, and every constant of it now comes from a
choice whose candidates carried none of the window-derived values (the 23 Oct rule's re-confirmation: the touch rule
moved, the decoder moved by one value, P07's weight and the shift stayed).

**The limits, and what the numbers do not say.**
- **The touch rule sits on three edges of its grid** (radius 1.5, 6 m/s, 1 frame). Nothing was added: a smaller radius
  or a slower relative speed was worse in R008's context and was not run here.
- **The gain is the benchmark's.** The P05 caveat (T38, T39) and the shape model's (R011) still hold, and the tighter
  touch rule covers relatively fewer frames on BAR-ATH than on DFL (above). +0.0267 on DFL's raw truth is not a
  forecast for the full match, whose noise floor is about 0.04 (T07).
- **The truths still disagree (T33).** Stage A's choice alone lost 0.0107 coarse (0 wins in 7); the decoder's longer
  minimum (35 frames) won it back, so the coarse cell is level with `main` (+0.0001, 4 wins and 3 losses) and not a
  gain. The coarse truth prefers fewer possessions than any choice so far (its best Stage C config predicts 2.79 a
  minute); if our truth is coarse, this config is no better than R012's, and if it is DFL-fine, it is better.
- **The selection was made 192 + 144 times on seven matches.** The leave-one-match-out estimates agree (+0.0306 raw for
  Stage A's rule, +0.0263 for Stage C's), but Stage C's coarse tie-break picked the choice in five folds of seven, and
  the two stages ran in order (Stage C only at the full-data Stage A choice).
- **Seven matches from two competitions**, five of them Fortuna Düsseldorf home games, are not seven independent draws.

### R014 — The restart vote on main.toml: 8 configs, the choice and the bar (2026-10-26, Tian)
Proposal: P17 · Commit: part 1 (this declaration and the 8 configs) is committed before any run of it; the commits of the
runs are in part 2.

**Part 1, declared before any run** (2026-10-26). Nothing below was chosen from a score of this entry.

**Why.** A12: a perfect 2 s restart vote is worth +0.0198 raw and +0.0197 coarse to `main.toml` (7 wins in 7), above the
declared +0.01; the label-free source `restart_team` (P17) is built and its confidences are measured (A12 part 2).

**Candidates.** `main.toml` as it is (R013, sha256 `0ebdfbec...`) plus one `[[evidence]]` `restart_team` at weight 1.0 (an
exponent of 1 takes the measured probability as it is, R013's argument for `dead_ball_model`), model
`pipeline/models/dead_ball_hgb.json`, `setup` 50 and `min_stoppage` 25 (the design's), over `tail` {25, 50, 100, 150}
frames x `use_ball` {false, true}: 8 configs `pipeline/benchmark/configs/r014/r014_t{TTT}_b{0|1}.toml`, generated by
`restarts.py configs data/runs/benchmark/restarts/stats.json --out-dir pipeline/benchmark/configs/r014`, each with its own
setting's measured confidences (A12 part 2; a type without one does not vote):

| config | corner | goal_kick | throw_in |
|---|---|---|---|
| r014_t025_b0 | 0.93 | 0.72 | 0.92 |
| r014_t050_b0 | 0.81 | 0.72 | 0.88 |
| r014_t100_b0 | 0.60 | 0.71 | 0.79 |
| r014_t150_b0 | 0.61 | 0.69 | 0.76 |
| r014_t025_b1 | — | 0.72 | 0.90 |
| r014_t050_b1 | — | 0.71 | 0.87 |
| r014_t100_b1 | — | 0.68 | 0.79 |
| r014_t150_b1 | — | 0.66 | 0.75 |

(`kick_off` and `free_kick` do not vote in any config: the 30-calls-per-seed floor.)

**What is not re-chosen.** Everything else in `main.toml` (the touch rule, the dead-ball model, P07's weight, the decoder
4/32/35/25; R013 chose them); the restart rules' constants (declared in the design from the Laws of the Game, not tuned;
the goal-kick rule's lack of specificity seen in A12 part 2 is left as measured); weight, `setup`, `min_stoppage`.

**The runs.** Profile `tracking` v3, 3 seeds, all seven matches, one process:

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r014/r014_t025_b0.toml ... (all 8) --truth both
```

`main.toml`, `nearest_w50` and `nearest_w125` must reproduce R013's and R004c's figures to the last digit (`main` 0.5430
raw and 0.5593 coarse; `nearest_w50` 0.2018 raw; `nearest_w125` 0.2226 coarse). Then, for the chosen config and
`main.toml` only: the same on `--profile context` (a sensitivity check, R005).

**The selection** (R008 to R013's rule, `choose.py --reference main` on the reports). The objective is pooled spellF1 on
`tracking`, raw truth. The tie band is 0.005: every config within it of the best is tied, and among the tied the best
pooled `tracking` coarse spellF1 wins. Further ties: the smaller tail, then `use_ball` false; so the stems are given to
`choose.py` in that order: `r014_t025_b0 r014_t025_b1 r014_t050_b0 r014_t050_b1 r014_t100_b0 r014_t100_b1 r014_t150_b0
r014_t150_b1`. Leave-one-match-out as `choose.py` computes it (the raw objective alone and the whole rule). **Nothing is
added after the results**: a choice on an edge of the grid is reported as one, and the grid is not extended in this entry.

**The bar.** The chosen config is adopted only if its `tracking` raw pooled delta over `main.toml` is at least +0.005 (the
tie band) with an exact sign-flip p of at most 0.05.

**The veto.** It is not adopted if any of these is below -0.005: its `tracking` coarse pooled delta over `main.toml`, its
`context` raw delta, its `context` coarse delta, or its pooled alive/dead F1 on `tracking` (raw report) minus
`main.toml`'s (a vote after a restart must not move the dead-ball edges).

**The shift, by R012's rule.** If the chosen config passes, its form without `[[refine]]` has its lag measured as R013 did
(`conventions.py latency`, `tracking`, 3 seeds) and R012's rule gives the shift; the total is adopted as the rule gives it.

**Label-free checks on BAR-ATH, before adopting.** Two runs of the final config give the same sha256; the predicted `D`
share is within 1 point of R013's 35.9%; no `X` in a graded block; the number of possessions and the per-type vote counts
are reported as aggregates; `restarts.py bar-ath` on the final config. The public window is a correctness check only (T01)
and may not weigh in.

**Adoption** (Tian's standing answer of 26 Oct: like R010 to R013). If the bar is met, the veto does not fire and the
checks pass, the final config becomes `pipeline/configs/main.toml` and is kept as `pipeline/configs/r014_restart.toml`; its
sha256 goes to the submission log; it is not uploaded.

**If the bar is not met.** A12's ceiling is at least +0.01, so the design's Phase 2b follows: restart states in the
decoder behind a toggle (off by default, every existing config byte-identical), with its own declaration (R014b) before
its runs, and the same bar and veto.

**Part 2, the runs and the verdict** (2026-10-26). Everything below applies part 1 as written. The public window was not
scored: nothing was adopted, and it chooses nothing.

Commit hashes: the branch's commit messages were reworded on 27 Oct 2026 before the merge (the trees are unchanged), so
the run logs and the benchmark reports stamp the earlier hashes: f11e93d is 58fc542 and ccb5fc8 is c639e23 (the two
declarations), 0fbc216 is 4e0a650 and 8aef670 is 15949c4 (the decoder and the spread vote), 736c2c0 is a4d1349, 1e277e2
is 0bca76c, c89555d is 4304213, 404d980 is 57d0a8f.

**The runs.** Commit `58fc542` (part 1 and the 8 configs, committed before any run). Profile `tracking` v3, 3 seeds, all
seven matches, `--truth both`, throttled (one BLAS thread, below-normal priority); the log ends `EXIT=0`.

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r014/r014_t025_b0.toml ... (all 8) --truth both
python pipeline/benchmark/choose.py --reference main r014_t025_b0 r014_t025_b1 r014_t050_b0 r014_t050_b1 r014_t100_b0 r014_t100_b1 r014_t150_b0 r014_t150_b1 --out data/runs/benchmark/restarts/r014_choice.json
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/r014/r014_t025_b0.toml --profile context --truth both
```

The references reproduce to the last digit: `main` 0.5430 raw and 0.5593 coarse with alive/dead F1 0.9731,
`nearest_w50` 0.2018 raw, `nearest_w125` 0.2226 coarse.

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | poss/min | D share | raw Δ | p | wins | coarse Δ | p | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 0.5430 | 0.5593 | 0.8895 | 0.8878 | 0.9731 | 3.13 | 44.6% | | | | | | |
| r014_t025_b0 (chosen) | 0.5470 | 0.5564 | 0.8915 | 0.8897 | 0.9733 | 3.24 | 44.6% | +0.0040 | 0.0938 | 6 | -0.0029 | 0.0625 | 2 |
| r014_t050_b0 | 0.5438 | 0.5561 | 0.8906 | 0.8893 | 0.9733 | 3.22 | 44.6% | +0.0008 | 0.8594 | 5 | -0.0032 | 0.1719 | 1 |
| r014_t025_b1 | 0.5425 | 0.5545 | 0.8897 | 0.8878 | 0.9732 | 3.19 | 44.6% | -0.0005 | 0.6094 | 2 | -0.0048 | 0.0156 | 0 |
| r014_t050_b1 | 0.5416 | 0.5557 | 0.8892 | 0.8876 | 0.9733 | 3.17 | 44.6% | -0.0014 | 0.2031 | 2 | -0.0036 | 0.0312 | 1 |
| r014_t100_b1 | 0.5351 | 0.5558 | 0.8878 | 0.8868 | 0.9733 | 3.13 | 44.6% | -0.0079 | 0.0156 | 0 | -0.0035 | 0.0469 | 1 |
| r014_t100_b0 | 0.5349 | 0.5637 | 0.8888 | 0.8892 | 0.9733 | 3.12 | 44.6% | -0.0081 | 0.0625 | 1 | +0.0044 | 0.4375 | 4 |
| r014_t150_b1 | 0.5300 | 0.5535 | 0.8863 | 0.8859 | 0.9733 | 3.11 | 44.6% | -0.0130 | 0.0156 | 0 | -0.0058 | 0.0156 | 0 |
| r014_t150_b0 | 0.5243 | 0.5614 | 0.8870 | 0.8884 | 0.9734 | 3.08 | 44.6% | -0.0187 | 0.0156 | 0 | +0.0021 | 0.5312 | 3 |

Wins are the matches out of 7 where the config beats `main`.

**The selection, by the rule.** The best raw is `r014_t025_b0` at 0.5470; tied within 0.005 are it, `r014_t025_b1`
(0.5425) and `r014_t050_b0` (0.5438). The best coarse among them is `r014_t025_b0` (0.5564), so it is chosen. It sits on
the grid's lower edge in tail (25 frames), and the grid was not extended, as declared.

**Chosen against `main`.**
- `tracking` raw: pooled +0.0040, mean per match +0.0038 [+0.0001, +0.0075] (90% t), sign-flip p 0.0938, 6 wins and 1
  loss. Per match: J03WMX +0.0065, J03WN1 +0.0114, J03WOH +0.0029, J03WOY +0.0014, J03WPY +0.0054, J03WQQ +0.0036,
  J03WR9 -0.0049.
- `tracking` coarse: pooled -0.0029, mean -0.0029 [-0.0052, -0.0006], p 0.0625, 2 wins and 5 losses.

**Leave-one-match-out.**
- The raw objective alone picks `r014_t025_b0` in 7 of 7 folds: raw +0.0038 [+0.0001, +0.0075], p 0.0938, 6 of 7; coarse
  -0.0029, p 0.0625, 2 of 7.
- The whole rule picks `r014_t025_b0` in six folds and `r014_t050_b0` in one (J03WQQ): raw +0.0029 [-0.0013, +0.0070],
  p 0.2188, 5 of 7; coarse -0.0039 [-0.0059, -0.0019], p 0.0312, 1 of 7.

**Context (a sensitivity check, R005).** `main` 0.5489 raw and 0.5642 coarse, alive/dead F1 0.9700;
`r014_t025_b0` 0.5518 and 0.5600, alive/dead F1 0.9701.
- Raw +0.0029 (mean +0.0027 [-0.0004, +0.0059], p 0.0781, 5 wins and 2 losses).
- Coarse -0.0042 (mean -0.0043 [-0.0076, -0.0010], p 0.0781, 1 win and 6 losses).

**The bar** (at least +0.005 raw with p at most 0.05): **not met** (+0.0040, p 0.0938). **The veto** was not needed:
no quantity is below -0.005 (`tracking` coarse -0.0029, `context` raw +0.0029, `context` coarse -0.0042, alive/dead F1
+0.0002).

**Verdict.** Not adopted; `main.toml` is unchanged. Phase 2b follows, as declared (R014b).

**Reading.**
- Raw spellF1 falls with every longer tail, from +0.0040 at 25 frames to -0.0187 at 150. A vote of fixed length keeps
  pushing after the restarting team's first possession has ended: DFL's possessions are short (median 3.7 s), and the
  agreement falls with the tail (A12 part 2).
- A12's 2 s oracle stopped at the truth's next change, which a real fixed-length vote cannot know. The ceiling it
  measured (+0.0198) is partly information that no source of this kind has; R014 delivered a fifth of it
  ([T43](traps.md)).

### R014b — Restart states in the decoder: 8 configs, the choice and the bar (2026-10-26, Tian)
Proposal: P17 (Phase 2b) · Commit: part 1 (this declaration and the 8 configs) is committed before any run of it. The
commits of the runs are in part 2.

**Part 1, declared before any run** (2026-10-26). Nothing below was chosen from a score of this entry.

**Why.** R014's bar was not met. The chosen vote, `r014_t025_b0`, was +0.0040 raw (p 0.0938, 6 wins in 7) and -0.0029
coarse. R014 part 1 declared that, with A12's ceiling at least +0.01, the design's Phase 2b follows: restart states in
the decoder, behind a toggle. Two things in R014 point to the mechanism.
- Raw spellF1 fell with every longer tail, from 25 frames (+0.0040) to 150 frames (-0.0187). A vote of fixed length
  keeps pushing after the restarting team's first possession has ended.
- The vote lands on the source's own restart estimate and not on the decoder's D edge. A12's oracle stopped at the
  truth's next change, which a real fixed-length vote cannot do.

With restart states, the restart team is read inside the stoppage and carried by the decoder to wherever it ends D. It
decides only the first team after the stoppage, and later evidence ends that possession as it ends any other.

**What was built** (commits 4e0a650 and 15949c4, reviewed; `selftest.py --data` 185/185 and `main.toml`'s sha256
unchanged with the toggle off):
- The `hmm` decoder takes `restart_source` (the name of an evidence source) and `switch_restart` (nats).
  - Without `restart_source` it is byte-identical to before.
  - With it, it decodes four states: H, A, D_H ("dead, home restarts") and D_A.
  - The named source's `home` channel speaks only to D_H and D_A, and every other source is read as before.
  - Transitions:
    - Into either D state costs `switch_dead`.
    - Out of D_H to H, or D_A to A, costs `switch_dead`.
    - Out of D_H to A, or D_A to H, costs `switch_dead + switch_restart` (inf forbids it).
    - D_H to D_A is forbidden.
  - With no restart evidence it gives the three-state labels exactly.
- The source `restart_team` takes `span = "stoppage_end"`: one observation of who restarts, at the type's measured
  confidence c, spread evenly over the stoppage's frames [first, r), with sigmoid(logit(c)/n) on each of its n frames.
  The decoded D run therefore collects it wherever its edge falls.

**Candidates.** `main.toml` as it is (R013, sha256 `0ebdfbec...`), plus:
- one `[[evidence]]` `restart_team` at weight 1.0, with model `pipeline/models/dead_ball_hgb.json`,
  `span = "stoppage_end"`, `setup` 50, `min_stoppage` 25, and its `use_ball` setting's tail-25 confidences (the 1 s
  agreement of A12 part 2, as R014's t025 configs);
- the decoder's `restart_source = "restart_team"` and `switch_restart`.

The grid is `use_ball` {false, true} x `switch_restart` {4, 8, 16, inf}: 8 configs,
`pipeline/benchmark/configs/r014b/r014b_s{04,08,16,in}_b{0,1}.toml`, generated by:

```bash
python pipeline/benchmark/restarts.py configs data/runs/benchmark/restarts/stats.json --out-dir pipeline/benchmark/configs/r014b --span stoppage_end --from-tail 25 --switch-restart 4 8 16 inf
```

| config | use_ball | switch_restart | corner | goal_kick | throw_in |
|---|---|---|---|---|---|
| r014b_s04_b0 | false | 4 | 0.93 | 0.72 | 0.92 |
| r014b_s08_b0 | false | 8 | 0.93 | 0.72 | 0.92 |
| r014b_s16_b0 | false | 16 | 0.93 | 0.72 | 0.92 |
| r014b_sin_b0 | false | inf | 0.93 | 0.72 | 0.92 |
| r014b_s04_b1 | true | 4 | — | 0.72 | 0.90 |
| r014b_s08_b1 | true | 8 | — | 0.72 | 0.90 |
| r014b_s16_b1 | true | 16 | — | 0.72 | 0.90 |
| r014b_sin_b1 | true | inf | — | 0.72 | 0.90 |

- `switch_restart` is the cost of a first team other than the called one. At inf the first team after a stoppage is
  always the decoded restart team (true of the truth by definition), and the observation and the following evidence
  decide which team that is.
- The finite values let strong contrary evidence after the restart override a call.
- A call's confidence is a likelihood. One observation at 0.92 is about 2.4 nats, the size of one frame of a control
  episode.

**What is not re-chosen.**
- Everything else in `main.toml`.
- The restart rules (the design's).
- The confidences (A12 part 2's tail-25 measurement; no new measurement for the stoppage span).
- `weight`, `setup`, `min_stoppage`.

**The runs.** Profile `tracking` v3, 3 seeds, all seven matches, in one process:

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r014b/r014b_s04_b0.toml ... (all 8) --truth both
```

The three references must reproduce R014's figures to the last digit: `main` 0.5430 raw and 0.5593 coarse,
`nearest_w50` 0.2018 raw, `nearest_w125` 0.2226 coarse. Then the chosen config and `main.toml`, on `--profile context
--truth both` (a sensitivity check, R005).

**The selection.** R014's rule, `choose.py --reference main` on the reports:
- The objective is pooled spellF1 on `tracking` under the raw truth. With a tie band of 0.005, the best coarse among
  the tied wins.
- Further ties go to the smaller `switch_restart` (the weaker intervention), then to `use_ball` false. The stems are
  given in that order: `r014b_s04_b0 r014b_s04_b1 r014b_s08_b0 r014b_s08_b1 r014b_s16_b0 r014b_s16_b1 r014b_sin_b0
  r014b_sin_b1`.
- Leave-one-match-out as `choose.py` computes it.
- Nothing is added after the results: a choice on an edge of the grid is reported as one, and the grid is not extended.

**The bar.** R014's: a `tracking` raw pooled delta over `main.toml` of at least +0.005, with exact sign-flip p at most
0.05.

**The veto.** R014's: no adoption if any of the following is below -0.005:
- the `tracking` coarse delta;
- the `context` raw delta;
- the `context` coarse delta;
- pooled alive/dead F1 on `tracking` minus `main.toml`'s.

**Then.** If the bar is met and the veto does not fire:
- the shift by R012's rule;
- the label-free checks on BAR-ATH, as declared in R014: the same sha256 twice; the D share within 1 point of 35.9%;
  no `X` in a graded block; possessions and per-type vote counts as aggregates; `restarts.py bar-ath`;
- adoption into `main.toml`, also kept as `pipeline/configs/r014_restart.toml`, with its sha256 in the submission
  log. It is not uploaded.

If the bar is not met, P17 closes as "dropped -> R014, R014b". The source and the decoder's toggle stay in the code,
off by default.

**Part 2, the runs and the verdict** (2026-10-26). Everything below applies part 1 as written. The public window was not
scored: nothing was adopted, and it chooses nothing.

**The runs.** Commit `c639e23` (code `4e0a650` and `15949c4`; part 1 and the 8 configs, committed before any run).
Profile `tracking` v3, 3 seeds, all seven matches, `--truth both`, throttled; the log ends `EXIT=0`.

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r014b/r014b_s04_b0.toml ... (all 8) --truth both
python pipeline/benchmark/choose.py --reference main r014b_s04_b0 r014b_s04_b1 r014b_s08_b0 r014b_s08_b1 r014b_s16_b0 r014b_s16_b1 r014b_sin_b0 r014b_sin_b1 --out data/runs/benchmark/restarts/r014b_choice.json
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/r014b/r014b_s04_b1.toml --profile context --truth both
```

The references reproduce R014's figures to the last digit (`main` 0.5430 raw and 0.5593 coarse, alive/dead F1 0.9731;
`nearest_w50` 0.2018 raw; `nearest_w125` 0.2226 coarse).

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | poss/min | D share | raw Δ | p | wins | coarse Δ | p | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 0.5430 | 0.5593 | 0.8895 | 0.8878 | 0.9731 | 3.13 | 44.6% | | | | | | |
| r014b_s04_b0, s08_b0, s16_b0, sin_b0 (identical) | 0.5435 | 0.5587 | 0.8898 | 0.8880 | 0.9731 | 3.14 | 44.6% | +0.0005 | 0.5156 | 4 | -0.0006 | 0.2969 | 1 |
| r014b_s04_b1 (chosen), s08_b1, s16_b1, sin_b1 (identical) | 0.5433 | 0.5589 | 0.8897 | 0.8880 | 0.9731 | 3.14 | 44.6% | +0.0003 | 0.8125 | 2 | -0.0004 | 0.4688 | 1 |

**The selection, by the rule.** All 8 are within 0.005 of the best raw (`r014b_s04_b0`, 0.5435). The best coarse among
them is the `b1` configs' (0.5589), and the tie order (the smaller `switch_restart`, then `use_ball` false) gives
`r014b_s04_b1`.

**Chosen against `main`.**
- `tracking` raw: pooled +0.0003, mean +0.0002 [-0.0008, +0.0012], p 0.8125, 2 wins, 4 losses and one tie. Per match:
  -0.0003, +0.0024, -0.0003, +0.0016, -0.0009, +0.0000, -0.0014.
- `tracking` coarse: pooled -0.0004, mean -0.0004 [-0.0015, +0.0006], p 0.4688, 1 win and 5 losses.

**Leave-one-match-out.**
- The raw objective alone picks `r014b_s04_b0` in 7 of 7 folds: raw +0.0004 [-0.0007, +0.0016], p 0.5156, 4 of 7.
- The whole rule picks `r014b_s04_b1` in 7 of 7: raw +0.0002, p 0.8125; coarse -0.0004, p 0.4688.

**Context (a sensitivity check).** `main` 0.5489 raw and 0.5642 coarse; `r014b_s04_b1` 0.5497 and 0.5645; alive/dead F1
0.9700 for both.
- Raw +0.0008 (mean +0.0008 [+0.0002, +0.0014], p 0.0625, 6 wins and 1 loss).
- Coarse +0.0003 ([-0.0002, +0.0009], p 0.2812, 5 of 7).

**The bar:** **not met** (+0.0003, p 0.8125). **The veto:** no quantity is below -0.005.

**Verdict.** Not adopted. **P17 closes as dropped, R014 and R014b**, as declared. `main.toml` is unchanged (sha256
`0ebdfbec...`). The source `restart_team` and the decoder's restart states stay in the code, off by default; no existing
config's output changed (`selftest.py --data` passes 185/185, `main.toml`'s sha256 is unchanged, and the benchmark
reproduced `main`).

**Reading.**
- The four `switch_restart` values give identical labels on every match and seed, **by construction, not by
  observation**. Choosing `D_H` or `D_A` for a decoded `D` run changes its score by at most the restart log-odds that
  run collects: one stoppage's single observation, at most logit(0.93), about 2.59 nats (a throw-in, logit(0.92), 2.44),
  while a cross transition (a first team other than the run's restart team) costs `switch_restart` on top. So any
  `switch_restart` above about 2.6 decodes like `inf`, except for a `D` run that collects two or more same-team calls,
  which only the value 4 could ever reach (two calls make about 5 nats), and the grid {4, 8, 16, inf} could not
  differ. Only the +0.0003 delta says the observation is weak; the identical labels say nothing about it (T44).
- The single observation almost never tipped the first team. For a throw-in it is logit 0.92, 2.4 nats, the size of one
  frame of a control episode. R014's 25-frame vote carried about 61 nats and moved the score more (+0.0040).
- Restart states read the call where it belongs, inside the stoppage, but the call alone is weak evidence next to the
  episodes that follow it. The follow-ups are in [After P17](proposals.md#after-p17).

### R015 — The offpitch profile: its fit, its fidelity, and main.toml on it (2026-10-29, Tian)

Proposal: P18 · Commits: the profile `4f1766a`, the ruler and the target `906c588` and `530b0bb`, the fit and the
fidelity gates `dacd1d5`, the constants pinned `fb33b38` · Commands, from the repo root, one process at a time on one
BLAS thread at below-normal priority, every log ending `EXIT=<code>`:

```bash
python pipeline/benchmark/offpitch.py target                                              # at dacd1d5, EXIT=0
python pipeline/benchmark/offpitch.py fit --seeds 3 --out data/runs/offpitch/fit.json     # at dacd1d5, EXIT=0
python pipeline/benchmark/fidelity.py --profile offpitch --seeds 3                        # at fb33b38, EXIT=1: one gate fails
python pipeline/benchmark/run.py pipeline/configs/main.toml --profile offpitch --truth both   # at fb33b38, EXIT=0
```

Validation (P01 benchmark): 7 DFL matches x 3 seeds (0, 1, 2). At `dacd1d5` the benchmark's `selftest.py` passed 436
checks and `pipeline/selftest.py --data` 186 of 186 (`main.toml`'s sha256 `0ebdfbec...` unchanged), so no existing
profile or config moved. Aggregates only: the benchmark is public DFL data, and the one figure from our match is a count
and a share of its tracking (a label-free statistic).

**What the profile is** (`degrade.py`, version 1). `tracking` v3 exactly (the same players, the same transplant draws),
with two changes to the ball, in this order on each half's span:
1. **Thinning (q).** Each interior gap of the transplanted mask is dropped whole, its frames given DFL's ball back, with
   probability q (a new random component, `"thinning"`, one draw per gap in time order).
2. **Off the pitch (m), with hysteresis.** The ball is hidden from the first frame on which DFL's clean ball is more than
   m metres past a touch or goal line until the first later frame on which it is back on the pitch (line distance at
   least 0). Deterministic. A half that ends with the ball off the pitch stays hidden to its end.

The speed is NaN everywhere (the profile's shared `_remove_ball`). Any difference between a config's score on `tracking`
and on `offpitch` is therefore the ball's doing.

**The reopened ruling.** R005 (23 Oct) rejected "a mask for the ball off the pitch" as "a new mechanism with a parameter
that cannot be measured on ours". Tian reopened that ruling on 2026-10-29 for this separate profile, because here the
parameter is fitted to a statistic of the frames where our ball **is** seen (the share of them past a line), which is
measurable. R005's other rejection, thinning the gap hazard beyond a line towards our share to pass `context`'s near-line
gate, is not reopened: `offpitch` has no hazard, and the near-line onset share is a check here, not a gate.
`tracking` v3 stays the default.

**The assumption.** What stays unobservable is how long our ball spends out of play in total. The profile assumes it is
like DFL's: the stretches DFL's ball spends past a line, hidden, stand in for ours.

**Our target** (`offpitch.py target`, our kick-off spans, cache key `55d73a4cdb45983a`): 3,740 of 91,844 visible ball
frames are past a line, a share of **0.040721223**. R005's 4.07% was taken on the hazard's at-risk rows; the two agree.
On those 3,740 frames the ball is past the line by (metres) p10 0.355, p25 0.674, median 1.744, p75 4.408, p90 8.094.

**The fit**, once, pooled by sums over 7 matches x seeds 0 to 2 (3,007,932 rows from the kick-offs). The rule was fixed in
the design before any run: m is the grid value whose share past a line, at q = 0, is closest to the target; q then fixes
the missing-ball share; the share is measured once more at (m, q) and m refitted once only if it moved by more than 0.005.

| m (metres) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|---|
| share past a line at q = 0 | 0.0000 | 0.0182 | 0.0444 | 0.0800 | 0.1062 | 0.1259 | 0.1454 |

- **m = 0.2 m** (0.0444 against the target 0.0407 is the closest of the grid).
- **q = 0.430**: the missing-ball share 0.411666886 against the Calibration's 0.410965669.
- The share past a line at (0.2, 0.430) is 0.041424766 (73,308 of 1,769,666 visible frames), within 0.005 of the target:
  **no refit**.
- The draws at the fit: 1,326 off-pitch stretches hidden (703,713 frames, 23.4% of the rows); 1,225 of 2,802 interior
  transplanted gaps dropped (556,797 frames given back).

**Frozen** in `degrade.py` at `fb33b38`: `OFFPITCH_MARGIN_M` 0.2, `OFFPITCH_DROP_PROB` 0.43 and `OFFPITCH_FIT` (the
measurements that fixed them). They are not refitted because of a gate, a WARN or a score (T37).

**Fidelity** (`fidelity.py --profile offpitch`, 7 matches x seeds 0 to 2): the eight gates of `tracking` plus
`beyond_line_share`. **8 of 9 gates pass, and one FAILS.**

| gate | offpitch | ours | band | |
|---|---|---|---|---|
| median_gap_s | 15.04 s | 10.74 s | +-1.5 | **FAIL** |
| missing_share | 0.4117 | 0.4110 | | pass |
| present_share_22 | 0.4925 | 0.5019 | | pass |
| present_mean | 20.5959 | 20.5224 | | pass |
| present_share_le14 | 0.0469 | 0.0525 | | pass |
| wrong_team_share | 0.0918 | 0.1054 | | pass |
| tracklet_median_s | 26.52 | 41.28 | -36%, +-50% | pass |
| speed_nan_share | 0.0788 | 0.0783 | | pass |
| beyond_line_share | 0.0414 | 0.0407 | +-0.01 | pass |

**Checks:** 2 of 11 are **WARN**.
- `dead_ball_visible_share` is 0.3667 against the public window's 0.6637 (+-0.10). `tracking` v3's ball was 0.5684, which
  was inside the band, so the profile moves this share away from the window's.
- `near_line_share` (the share of gaps starting within 2 m of a line) is 0.5400 against ours 0.2797 (+-0.05). `tracking`
  v3 had 36.0% (R003). The design expected a rise, because a gap now starts where the ball leaves the margin.
- The other nine are ok, among them the nearest-player agreement (0.7612 against 0.7784) and the yardsticks' possessions
  per minute and D shares inside their bands.

**Information** (no band):
- Gaps per 10,000 frames 7.35 against ours 9.36. Gap length: p10 3.28 s against 2.68, median 15.04 against 10.74, p90
  45.64 against 35.30.
- Gaps per match and seed, `offpitch` against `tracking`'s ball: J03WMX 108.7 / 131.7, J03WN1 112.3 / 137.7, J03WOH 101.7
  / 133.3, J03WOY 111.0 / 138.0, J03WPY 107.7 / 132.3, J03WQQ 91.7 / 135.0, J03WR9 104.3 / 135.0.
- Metres past the line on the visible frames that are past it: the benchmark's p10 0.04, p25 0.07, median 0.13, p75 0.15,
  p90 0.17, against ours 0.35, 0.67, 1.74, 4.41, 8.09. The margin rule caps the benchmark at 0.2 m, while ours sees the
  ball far out, rarely.
- Gaps shorter than 2 s: the benchmark 200 of 2,212 (9.04%, 0.08% of the missing time), ours 15 of 146 (10.27%, 0.08%).
  The hysteresis leaves no flicker.
- The missing share per match and seed ranges from 0.265 (J03WQQ) to 0.493.

**`main.toml` (R013's) on `offpitch` beside on `tracking` v3.** This is a difference between two profiles, not a paired
comparison on one:

| cell | spellF1 | macroF1 | alive/dead F1 | possessions per graded min (true) | predicted D (true) |
|---|---|---|---|---|---|
| tracking raw (A13, R013) | 0.5430 | 0.8895 | 0.9731 | 3.13 (4.58) | — |
| offpitch raw | 0.5509 | 0.8800 | 0.9600 | 3.46 (4.58) | 41.7% (44.8%) |
| tracking coarse | 0.5593 | 0.8878 | 0.9731 | — | — |
| offpitch coarse | 0.5296 | 0.8746 | 0.9600 | 3.46 (2.48) | 41.7% (44.8%) |

Per match on `offpitch`, raw: J03WMX 0.4855, J03WN1 0.5285, J03WOH 0.5582, J03WOY 0.5759, J03WPY 0.5575, J03WQQ 0.6006,
J03WR9 0.5622. Coarse: J03WMX 0.5362, J03WN1 0.4841, J03WOH 0.5292, J03WOY 0.5206, J03WPY 0.5574, J03WQQ 0.5605, J03WR9
0.5254.

**Reading.**
- With DFL's out-of-play ball hidden, today's P05 loses 0.013 alive/dead F1 (0.9731 to 0.9600), and spellF1 moves +0.008
  raw and -0.030 coarse.
- Part (a) of the design's belief (the current P05 relies heavily on the visible out-of-play ball) is weak on this
  evidence: the loss is small.
- The off-pitch stretches are themselves stoppage-tied gaps (the dead-ball visible share falls from 0.57 to 0.37), which a
  model reading `since_seen` or `last_outside` can use.
- The profile fails one gate: its median gap is 15.04 s against ours 10.74 (+-1.5), and two checks WARN. The gap counts
  are lower than `tracking`'s in every match, and the distances past a line stop at 0.2 m by construction.

At the pause Tian chose to try a v2 profile designed from these lessons; R015b records it and P18's drop.

### R015b — offpitch v2 (found again at rest): a feasibility check, and P18 dropped (2026-10-29, Tian)

Proposal: P18 · Commits: the check was run at `cac9849` (code as `fb33b38`); the removal is `7b555eb` · Command: a
throwaway script, not committed (like R005's `nearline_diag`), run once from the repo root, one process, one BLAS thread,
below-normal priority. It printed label-free fidelity statistics only and scored nothing.

**Why a v2.** R015's v1 matched our share past a line (0.0414 against 0.0407) and failed the median gap (15.04 s against
10.74), with two WARNs. Our tracker sees the ball off the pitch rarely (4.07% of its visible frames) but well outside (median
1.74 m past the line, p90 8.09 m), which looked like a ball at rest or held for a restart, and v1 hid the restart
preparation too. At the pause (2026-10-29) Tian chose to try a v2 designed from v1's lessons, approved its rule and a quick
label-free feasibility check first, and set the rule: **if v2 fails, P18 is dropped and its code removed.** One shot.

**The v2 rule.** DFL's clean ball is hidden from the first frame on which it is more than m past a line and moving (speed
at least 1.0 m/s, `play_features.STILL_SPEED`, the dead-ball features' own "still" threshold), until the first frame on
which it is back on the pitch or at rest (speed below 1.0 m/s). The same two fitted constants as v1 (m, q).

**The check.** The 7 DFL matches x seeds 0 to 2, on the kick-off spans. For each m the script also found the q whose
missing share is closest to ours (0.410966). Share past a line is over all visible frames on the kick-off spans, which is
why `tracking` v3's ball shows 0.2608 here; R005's 17.68% was over the hazard's at-risk rows of `context`, a different
set. The window's dead-ball visible share is 0.6637.

| ball | m | q | share past a line | median gap (s) | gaps / 10k frames | gaps < 2 s | near-line onsets | dead visible |
|---|---|---|---|---|---|---|---|---|
| ours (target) | | | 0.0407 | 10.74 | 9.36 | 0.1027 | 0.2797 | (window 0.6637) |
| `tracking` v3's ball | | 0 | 0.2608 | 10.76 | 9.41 | 0.0979 | 0.3597 | 0.5684 |
| v1 (frozen) | 0.2 | 0.430 | 0.0414 | 15.04 | 7.35 | 0.0904 | 0.5400 | 0.3667 |
| v2 | 0.0 | 0.030 | 0.2579 | 5.76 | 14.91 | 0.4481 | 0.6091 | 0.5764 |
| v2 | 0.2 | 0.030 | 0.2586 | 6.08 | 14.03 | 0.4133 | 0.5848 | 0.5767 |
| v2 | 0.5 | 0.025 | 0.2601 | 7.32 | 12.24 | 0.3229 | 0.5198 | 0.5760 |
| v2 | 1.0 | 0.025 | 0.2610 | 9.48 | 10.35 | 0.1994 | 0.4315 | 0.5769 |
| v2 | 2.0 | 0.025 | 0.2614 | 10.68 | 9.53 | 0.1301 | 0.3818 | 0.5773 |
| v2 | 5.0 / 10.0 | 0.025 | 0.2616 | 10.76 | 9.18 | 0.0971 | 0.3581 | 0.5774 |

**Reading.**
- DFL's ball, when off the pitch, is almost always slow or at rest (held for a throw-in, waiting), so "found again at
  rest" brings almost all of it back: the share past a line stays at `tracking` v3's level (0.258 to 0.262) at every
  margin, and `beyond_line_share` (+-0.01 around 0.0407) cannot pass at any m.
- Small margins also make the ball flicker: up to 45% of the gaps are under 2 s (m 0.0: 0.4481), against ours 10%.
- v1 and v2 bracket what our tracker does: hiding the off-pitch ball until play resumes matches our share past a line and
  makes the gaps too long; re-finding it at rest keeps the gaps' shape and shows far too much of it. Where our ball is
  during our own gaps is not observable, which is R005's original objection, and it stands.
- v2's mechanism was designed after v1's gates had been seen, so the gates are less independent of it than they were of v1.

**Decision (Tian's rule): P18 is dropped.** The offpitch code was removed in `7b555eb`: `degrade.py`, `fidelity.py`,
`run.py` and the checks are restored to their state at `5ab0e89`, and `offpitch.py` and its two check modules are deleted.
It is recoverable from `fb33b38`. After it, the benchmark's `selftest.py` passed 402 checks, `pipeline/selftest.py --data`
186 of 186, and `main.toml`'s sha256 `0ebdfbec...` is unchanged. What stays: `headroom.py`'s `plus_` variants and the
gaps-only team oracle (A13, `5ab0e89`). R016 never ran (its number was reassigned to P12's free X on 30 Oct), A13 part 2 (on `offpitch`) never ran, and `main.toml` is unchanged.
What it leaves is in [After P18](proposals.md#after-p18) and T45.

### R016 — The free_x refiner (P12): 4 thresholds, the choice, the bar and the veto (2026-10-30, Tian)
Proposal: P12 · Commit: part 1 (this declaration and the 4 configs) is committed before any run of it; the commits of the
runs are in part 2.

**Part 1, declared before any run** (2026-10-30). Nothing below was chosen from a score of this entry. R016 was once
declared in outline and never ran (R015b); this is its declaration.

**Why.** Where the HMM's posterior for the decoded label is low, the label is often wrong. Writing `X` there turns a false
positive into nothing, at the cost of a true positive where the label was right. To first order, abstaining on a frame
predicted as class k changes F1_k by (F1_k - 2p) / (2TP + FP + FN), where p is the chance the label is right, so it
gains only where p < F1_k / 2: about 0.3 to 0.45 for our classes. The grid sits there, provided the posterior is roughly
calibrated; if it is overconfident a higher threshold may be needed, so the grid reaches 0.55. spellF1 cannot change if
`X` never sits within a margin of a run's edges, because spellF1 forward-fills `X` with the label before it, which is the
run's own label; only macroF1 can move.

**Expectation, stated now.** The decoded label's posterior is rarely below 0.45, so the gain is likely small. macroF1 only
breaks exact spellF1 ties, so even a pass is worth little to the ranking. Tian chose to build it anyway, and the bar
decides. **Falsified if** no threshold gives a pooled macroF1 gain on the benchmark above the bar.

**Candidates.** `main.toml` (sha256 of the committed file `0970f2de4b97ceec27660559ebc37b1a0dcbfcf1bfced08d016f1f9d4503181a`;
the Windows working copy with CRLF line ends hashes to `ad8d8b6c2f25a1d0713eb3c8e3bbf7f14eb5928d9fa6bf12ee5b455c106284bd`)
with `posterior = true` in `[decoder]` and one `[[refine]]` `free_x`, `margin` 3, before `shift_edges`, over `threshold`
in {0.25, 0.35, 0.45, 0.55}: four configs, `pipeline/benchmark/configs/r016/r016_t025.toml`, `r016_t035.toml`,
`r016_t045.toml`, `r016_t055.toml`. `margin` 3 is a fixed rule: more than `shift_edges`' largest move (2), so a shifted
edge never reaches an `X`.

**What is not re-chosen.** Everything else in `main.toml` (the touch rule, the dead-ball model, P07's weight, the decoder
4/32/35/25, the shift +1 / -2 / 0; R013 and R012 chose them); `margin`; `free_x` acts on `H` and `A` runs only.

**The runs.** The reference is `main.toml` as it is (with `--drop-match-only` if C or D has merged by then). Profile
`tracking` v3, 3 seeds, all seven matches, `--truth both`, one process:

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/r016/r016_t025.toml pipeline/benchmark/configs/r016/r016_t035.toml pipeline/benchmark/configs/r016/r016_t045.toml pipeline/benchmark/configs/r016/r016_t055.toml --truth both
python pipeline/benchmark/choose.py --reference main --metric macro --band 0.002 r016_t025 r016_t035 r016_t045 r016_t055
```

`main` must reproduce R013's figures to the last digit (0.5430 raw and 0.5593 coarse spellF1, alive/dead F1 0.9731).

**The selection** (R008 to R014's rule, `choose.py --reference main --metric macro` on the reports). The objective is pooled
raw macroF1 on `tracking`, with a tie band of 0.002. Among the tied configs the best pooled coarse macroF1 wins. Any tie
left over goes to the lowest threshold, which writes the fewest `X`; so the stems are given to `choose.py` in the order
`r016_t025 r016_t035 r016_t045 r016_t055`. The band must be passed explicitly (`--band 0.002`), because the tool's default is 0.005. **Nothing is added after the results**: a choice on an edge of the grid is
reported as one, and the grid is not extended in this entry.

**The bar.** The chosen config is adopted only if its `tracking` raw pooled macroF1 delta over `main.toml` is at least
+0.002 with an exact sign-flip p of at most 0.05.

**The veto.** It is not adopted if either of these holds:
- any pooled spellF1 delta is non-zero (raw or coarse). It is zero by construction, so a non-zero one means a bug;
- the pooled coarse macroF1 delta is below -0.002.

Alive/dead F1 is not a veto here: it counts an `X` on an alive frame as a miss by definition, and it does not rank. Its
delta is reported.

**Label-free checks on BAR-ATH, before adopting.**
- Two runs of the final config give the same sha256.
- The share of graded frames written `X` is reported.
- The `D` share is unchanged.
- R014's standing check "no `X` in a graded block" is replaced, for this config only, by "`X` only inside `H` and `A`
  runs, beyond the margin".

The public window is a correctness check only (T01) and may not weigh in.

**Adoption.** If the bar is met, the veto does not fire and the checks pass, the final config becomes
`pipeline/configs/main.toml` and is kept as `pipeline/configs/r016_free_x.toml`; its sha256 goes to the submission log; it
is not uploaded.

**If the bar is not met.** One knowledge-informed retry, R016b, is declared before its runs (a different grid, chosen from
what R016 shows about the posterior's calibration, never from a window score). If that fails too, P12's `free_x` code is
removed.

**Part 2, the runs and the verdict** (2026-10-30). Commit of the runs: `c909e2f` (the declaration and the configs, unchanged
since part 1). The controller ran, in one throttled process (profile `tracking` v3, 3 seeds 0 to 2, seven matches):

```bash
python pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r016/r016_t025.toml pipeline/benchmark/configs/r016/r016_t035.toml pipeline/benchmark/configs/r016/r016_t045.toml pipeline/benchmark/configs/r016/r016_t055.toml --truth both
python pipeline/benchmark/choose.py --reference main --metric macro --band 0.002 r016_t025 r016_t035 r016_t045 r016_t055
```

**Reference check.** `main` reproduced R013: spellF1 0.5430 raw and 0.5593 coarse, alive/dead F1 0.9731. The two nearest-player
baselines came out at `nearest_w50` 0.2018 raw (0.1842 coarse) and `nearest_w125` 0.2226 coarse,
and the coarse `nearest_w125` figure is R004c's 0.2226 to the last digit.

**The choice table** (`choose.py`, objective pooled raw macroF1, tie band 0.002; deltas are over `main`, positive is better):

| config | raw spellF1 | coarse spellF1 | raw macroF1 | coarse macroF1 | alive/dead F1 | possessions / min | D share | raw Δ | p | wins | coarse Δ | p | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `main` | 0.5430 | 0.5593 | 0.8895 | 0.8878 | 0.9731 | 3.13 | 44.6% |  |  |  |  |  |  |
| **`r016_t025`** | 0.5430 | 0.5593 | 0.8893 | 0.8868 | 0.9702 | 3.13 | 44.6% | -0.0003 | 0.0625 | 1 | -0.0009 | 0.0156 | 0 |
| `r016_t035` | 0.5430 | 0.5593 | 0.8891 | 0.8865 | 0.9692 | 3.13 | 44.6% | -0.0004 | 0.0312 | 1 | -0.0013 | 0.0156 | 0 |
| `r016_t045` | 0.5430 | 0.5593 | 0.8889 | 0.8862 | 0.9680 | 3.13 | 44.6% | -0.0006 | 0.0312 | 1 | -0.0016 | 0.0156 | 0 |
| `r016_t055` | 0.5430 | 0.5593 | 0.8887 | 0.8858 | 0.9663 | 3.13 | 44.6% | -0.0008 | 0.0312 | 1 | -0.0020 | 0.0156 | 0 |

The rule tied all four within 0.002 and, by its tie order, chose `r016_t025` (the lowest threshold, the fewest `X`).
Against `main`: raw macroF1 -0.0003 (mean per match -0.0003, 90% interval [-0.0005, -0.0000], p 0.0625, 1 win, 6 losses),
coarse -0.0009 (p 0.0156, 0 wins, 7 losses). Leave-one-match-out picked `r016_t025` in all seven folds.

**The bar** (at least +0.002 raw pooled macroF1 with p at most 0.05): **not met.** The chosen config is 0.0003 below `main`,
not 0.002 above it.

**The veto.** It did not fire: every spellF1 delta is exactly +0.0000 under both truths, in all seven matches, as
constructed (no bug), and the largest coarse macroF1 delta, -0.00197 at t055, is not below -0.002. Alive/dead F1 fell from
0.9731 to 0.9702, 0.9692, 0.9680 and 0.9663: reported as declared, not a veto. Possessions per minute (3.13) and the D share
(44.6%) are unchanged.

**Verdict: bar not met; `free_x` dropped, `main.toml` unchanged.** All four thresholds lose macroF1, and the loss grows
with the threshold (raw -0.0003 to -0.0008, coarse -0.0009 to -0.0020), so it shrinks monotonically toward 0 as fewer `X`
are written. The expectation stated in part 1 holds in its strong form: the HMM's least confident frames are still right
more often than F1/2 of the time, so no threshold of this signal can reach the bar (nothing below 0.25 was tested; the trend is monotone towards 0).

**Rulings (the controller's, binding).**
1. **No R016b.** The retry that part 1 allowed is not run. Every threshold lowers macroF1 and the loss falls to 0 only as
   `X` disappears, so a different grid of the same signal would rest on hope, not on knowledge.
2. **The idea's code is removed completely** (Tian's rule), in the commit after this entry: `pipeline/refine/free_x.py`, its
   registry line, its checks (with `FREE_X_ABSENT_SHA256`) and `pipeline/benchmark/configs/r016/`. It is recoverable from
   `c909e2f`. `choose.py --metric` stays, as a general benchmark option.

### R017 — Smart Tagging's votes against our own pseudo-labels: the bar, and the strength in the decoder (2026-10-30, Tian)
Proposal: P08 · Commit: part 1 (this declaration) is committed before any measurement it governs; the measurement's
commits are in part 2.

**Part 1, declared before any measurement** (2026-10-30). Nothing below was chosen from a score of this entry, and no
number of this match has been seen: the source (`pipeline/evidence/smart_tagging.py`) and the measurement
(`pipeline/benchmark/tag_stats.py`) are built and unit-tested on synthetic data only.

**Why.** Smart Tagging carries 638 automatic phase intervals of our match (T17): not possessions and not ground truth.
Their codes say who attacks or defends, and that play is stopped (set pieces, goals), which is information about the
ball-missing stretches where `controlled_touch` is blind. There is no truth to score them against and no benchmark (DFL
has no tags), so the only honest bar is agreement with the two pseudo-labels of our own pipeline that are already
trusted. The measured agreements become the votes' confidences, so the bar decides whether the source enters at all.

**The votes**, fixed from what the codes mean (P08's table), never from agreement with any truth. Codes are upper-cased
and tag values matched case-insensitively (T17):

| codes | vote |
|---|---|
| `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `ATTACKING TRANSITION`, `LONG BALL` | `home`: the tagged team has it |
| `DEFENDING IN ... THIRD`, `DEFENSIVE TRANSITION` | `home`: the other team has it |
| `SET PIECES`, `GOALS` | `alive`: dead |
| any other code | no vote |

`Team` `Barcelona` is home, `Atletic Club` (sic) is away, `N/A` gives no team vote (a set-piece tag with `N/A` still gives
its dead vote). Where tags disagree on a frame's team, that frame gets no team vote; agreeing tags give one vote, not
two. Outside the graded blocks the source says NaN. The source declares `MATCH_ONLY`: a DFL match raises
`AssetUnavailable`, and it is never run on the benchmark.

**The pseudo-labels** (no human label of this match, none decided by a person):
- **Team:** the kept episodes of `controlled_touch` with `conventions.TOUCH`'s parameters (radius 2.0 m, max_rel_speed
  6.0 m/s, min_frames 3, confidence 0.99, hold 0.5), so `home` is a number exactly on the kept frames, as
  `train_shape.touch_agreement` takes them. An episode is a home episode when `home` >= 0.5.
- **Dead:** `dead_ball_model` with `main.toml`'s own `[[evidence]]` entry's parameters, read from the config and not
  copied (the model only: that entry has no `min_prob`, so the source's default applies; weight 1.0). Dead is P(alive)
  < 0.5.

**The measurement.** `python pipeline/benchmark/tag_stats.py` (HEAVY: it loads BAR-ATH), run once by the controller on
this declaration's commit; it writes aggregates only (`data/runs/tag_stats/stats.json`).
- **Team agreement:** among the graded frames inside a kept episode where the tags give a team vote (no vote and
  disagreement excluded), the share where the vote names the episode's team. Also the coverage of graded frames and of
  ball-missing graded frames (`coverage_gaps`).
- **Dead agreement:** among the frames of `SET PIECES` and `GOALS` tags where the model has an opinion, the share where
  P(alive) < 0.5; the same share over all graded frames with an opinion is the base rate, and the lift is their ratio.
- **The intervals:** a 90% bootstrap (5th and 95th percentile) that resamples whole tags (`smart_tagging.tags`), never
  frames, since a tag's frames are one phase and strongly autocorrelated; 2000 draws, `default_rng` seeds (17, 0) for
  the team part and (17, 1) for the dead part. A frame covered by several tags is given to the first, so nothing is
  counted twice.
- **Per code:** code name, n frames and share, as aggregates (the retry's information).
- **The values:** c_team and c_dead are `towards_half` (P05's rounding, two decimals towards 0.5) of the measured point
  shares. They are required parameters of the source, so its code holds no number of its own.

**The bar.** Two independent parts, each adopted or not on its own:
- **Team part:** point agreement >= 0.75, the lower 90% bound >= 0.70, and `coverage_gaps` >= 0.05 (team votes on at
  least 5% of ball-missing graded frames; otherwise it adds nothing where it matters).
- **Dead part:** point dead share >= 0.60, the lower 90% bound >= 0.50, and a lift >= 1.5 over the base rate.

A part whose point share, lower bound, coverage or lift is undefined (no eligible frame, or a base rate of 0) fails its
bar; a bootstrap draw with no eligible frame is dropped (`tag_stats.py`), and `per_code` counts a frame covered by two tags
of a code twice (it is diagnosis for R017b, not the bar).

**The strength in the decoder.** No benchmark can decide it, since DFL has no tags. If a part passes, two configs are
run on BAR-ATH: `main.toml` plus the passing parts at `spread = "frame"`, weight 0.2 (`home` = c_team or 1 - c_team on
every voted frame, `alive` = 1 - c_dead on every dead-voted frame; 0.2 by analogy with P07, the other team source that
spans long windows), and at `spread = "tag"`, weight 1.0 (one observation per tag, each frame getting sigmoid(logit(c) /
n), so the decoder sums them back to logit(c)). Tian sees, for each, the aggregates of `label_diff` against `main.toml`:
the `D` share, the possessions per half, the share of graded frames whose label changed, and how many of those lie in
ball gaps. **Tian picks one, or neither, before `main.toml` changes. Nothing is chosen from a score.**

**Label-free checks on BAR-ATH** for the final config, before adopting: two runs give the same sha256; the `D` share is
within 1 point of `main.toml`'s; the number of possessions per half is reported; no `X` in a graded block. The public
window is scored only as a correctness check (are the votes the right way round?), never to choose, and may not weigh in
(T01).

**The reference.** `pipeline/configs/main.toml` is unchanged by this entry; its sha256 on the committed (LF) file,
`git show HEAD:pipeline/configs/main.toml | sha256sum`, at this declaration is
`0970f2de4b97ceec27660559ebc37b1a0dcbfcf1bfced08d016f1f9d4503181a` (R013's last change; the submission log's
`0ebdfbec...` is the sha256 of the CSV it produces, not of the config).

**If a part fails.** One retry is allowed (R017b), informed by which codes disagree (`per_code`), with its own
declaration before its measurement; if it fails again the source is removed from the code, as P18 was, and P08 closes as
dropped. A part that passes is not retried.

**Part 2, the measurement and the verdict** (2026-10-30). Everything below applies part 1 as written. The public window
was not scored for the verdict, and it chooses nothing. After the adoption, `selftest.py --data` scored it as a correctness
check: macroF1 0.6986, spellF1 0.4000 (R013: 0.7152, 0.4649). They were seen after Tian's pick and chose nothing (T01;
one possession moves spellF1 about 0.06 on that window). Logs hold aggregates only.

**The runs.**
- Measurement: `python pipeline/benchmark/tag_stats.py`, at commit `cefb8b0` (part 1 and the source, committed before the
  run); the log ends `EXIT=0`.
- Strength comparison: `python pipeline/benchmark/label_diff.py pipeline/configs/main.toml pipeline/configs/r017_frame.toml
  pipeline/configs/r017_tag.toml`, at commit `1125ad9`; the log ends `EXIT=0`. Both configs are `main.toml` plus the team
  vote only (the dead part failed).

**The measurement.**

| part | n frames | point share | 90% interval | other |
|---|---|---|---|---|
| team | 26,383 | 0.761 | [0.723, 0.795] | coverage 0.589 of graded frames, 0.499 of ball-missing graded frames |
| dead | 56,898 | 0.409 | [0.280, 0.508] | base rate 0.364, lift 1.124 |

Per code (frames covered, and the share that agrees with the part's pseudo-label: team agreement for the team codes, dead
agreement for `SET PIECES` and `GOALS`; a frame covered by two tags of a code counts twice here, as declared):

| code | frames | share |
|---|---|---|
| `ATTACKING TRANSITION` | 9,291 | 0.729 |
| `BALL IN FINAL THIRD` | 67,095 | 0.697 |
| `BALL IN THE BOX` | 31,654 | 0.634 |
| `BUILD UP` | 11,011 | 0.867 |
| `CREATING CHANCES` | 22,500 | 0.630 |
| `DEFENDING IN ATTACKING THIRD` | 11,011 | 0.867 |
| `DEFENDING IN DEFENSIVE THIRD` | 22,500 | 0.630 |
| `DEFENDING IN MIDDLE THIRD` | 31,118 | 0.839 |
| `DEFENSIVE TRANSITION` | 9,291 | 0.729 |
| `GOALS` | 16,503 | 0.785 |
| `LONG BALL` | 9,433 | 0.649 |
| `PLAYERS IN THE BOX` | 75,736 | 0.596 |
| `PROGRESSION` | 31,118 | 0.839 |
| `SET PIECES` | 47,037 | 0.263 |

The would-be values: c_team 0.76 and c_dead 0.41 (`towards_half`, two decimals).

**The bar.**
- **Team part: passes.** Point 0.761 >= 0.75, lower bound 0.723 >= 0.70, `coverage_gaps` 0.499 >= 0.05. c_team = 0.76.
- **Dead part: fails.** Point 0.409 < 0.60, lower bound 0.280 < 0.50, lift 1.124 < 1.5. All three conditions fail.

**The strength comparison** (against `main.toml`, graded frames):

| config | frames changed | of them in ball gaps | D share | possessions half 1 | half 2 | D onsets |
|---|---|---|---|---|---|---|
| main | | | 35.92% | 192 | 218 | 106 |
| `r017_frame.toml` (`frame`, weight 0.2) | 2.28% | 56.3% | 35.93% | 187 | 226 | 108 (16 moved, 1 lost, median move +1.0 frames) |
| `r017_tag.toml` (`tag`, weight 1.0) | 0.03% | 2.1% | 35.92% | 191 | 218 | 106 (0 moved) |

No `X` appears in either config (0.00% before and after).

**Verdict.**
- **Team part:** passes the bar, so the source may enter. The strength is Tian's to pick: `frame` at weight 0.2, `tag` at
  weight 1.0, or neither. `main.toml` is unchanged until Tian does, and nothing is chosen from a score.
- **Dead part: no R017b.** Only 0.263 of the `SET PIECES` frames are dead, which is consistent with those tags spanning the
  restart and the play after it (a reading, not a measurement); of the two dead codes it is the larger (47,037 frames,
  against 16,503 for `GOALS`) and it drags the share down. A `GOALS`-only retry would add almost
  nothing: P05 already calls 78.5% of those frames dead, over the log's 16,503 `GOALS` frames. A code subset chosen after
  this measurement would also be judged on the data that chose it. The dead-vote path was removed from the code (commit
  `668b868`; recoverable from `cefb8b0`).

**Reading.**
- `frame` is not small. It votes only on the team channel, yet it changes 2.28% of graded frames (56% of them in ball
  gaps, where it is meant to act), moves 16 `D` onsets (median +1 frame) and shifts the possession counts (half 1:
  192 to 187, half 2: 218 to 226). A team vote that moves dead-ball onsets does so presumably through the decoder's joint states,
  not through its own evidence.
- `tag` changes 0.03% of graded frames: at one observation per tag it is close to a no-op, which is its lesson too: a
  weight-1.0 sum of one observation per tag barely competes with the evidence on every frame.
- The paired attack and defence codes (`BUILD UP` and `DEFENDING IN ATTACKING THIRD`, and so on) share frames and
  shares by construction: one tag's vote is read from each side, so the per-code table counts each such tag twice.
- Neither config has a score here. There is no benchmark for tags (DFL has none), so what `frame` adds or loses in
  `spellF1` is unmeasured, and the team agreement of 0.761 is against our own pseudo-labels, not a truth.

**Tian's pick (2026-10-30): per frame (`r017_frame`), adopted into `main.toml`**

Adoption checks (commit 56916f2, 2026-10-30):

- `python pipeline/run.py pipeline/configs/main.toml`: valid, 199,319 rows, sha256
  `7ba575c0982aacb9765b39c406e4ff5c9ddb82ca37cfa2b7f101417431fc3d7c`, 413 possessions in the graded halves.
- `python evaluation/validate_submission.py evaluation/results/main.csv`: valid.
- `python pipeline/selftest.py --data`: 193 of 193 passed, including `main.toml` run twice to the same sha256 and the
  four committed baselines rebuilt byte for byte.
- `python pipeline/benchmark/run.py pipeline/configs/main.toml --drop-match-only --truth both`: dropped
  `smart_tagging` (the benchmark has no tags) and reproduced R013's 0.5430 raw and 0.5593 coarse.
- The declared label-free checks pass: `D` 35.93% against R013's 35.92%, no `X`, 187 and 226 possessions in the two
  halves (413 in all, against R013's 192 and 218), the same sha256 on two runs.
- The public window, a correctness check seen after the pick (T01): macroF1 0.6986, spellF1 0.4000 (R013: 0.7152,
  0.4649); it chose nothing.

### R018 — The referee's whistle: a hand-built band-pass detector, its feasibility gate and its bar (2026-10-30, Tian)

Proposal: P10 · Commit: part 1 (this declaration, the detector `pipeline/evidence/whistle.py` and the statistics
`pipeline/benchmark/whistle_stats.py`) is committed before any run of the detector on the match; the runs are in part 2.

**Part 1, declared before any run** (2026-10-30). Nothing below was chosen from a score, from a stoppage or from the
audio of this entry; the audio has not been decoded when this is written.

**Belief.** A referee's whistle is a sustained tone at 2 to 4 kHz, well above most crowd noise. Play is dead in the moment
after a whistle: either it has just been stopped, or it is about to be restarted by a kick that takes at least half a
second to follow the whistle. So detected whistles should sit near a stoppage far more often than random times do.
**Falsified if** the detections are near a decoded stoppage no more often than the same detections shifted in time (for
example crowd whistling).

**The detector** (`whistle.detect`, constants in its `# Params`, argued in its docstring). Features per video frame
(`pipeline/inputs/audio.py`, rounded to 0.01 dB): `tonal_db` and `band_db`. A frame is tonal when `tonal_db >= 15.0` and
`band_db >= 3.0`; a whistle is a run of at least 5 graded frames (0.2 s); w is the run's first frame; no run crosses a
block's edge. The source's dead window after a whistle is 12 frames (0.5 s), declared here and used in part 2's source. No
constant is changed after a run, except by the retry below.

**The pseudo-labels.** `main.toml`'s own decoded `D` on BAR-ATH, as `main.toml` stands at this declaration: sha256
`ad8d8b6c2f25a1d0713eb3c8e3bbf7f14eb5928d9fa6bf12ee5b455c106284bd` (`sha256sum pipeline/configs/main.toml`; the command
prints it, and a different hash at the run means the run does not apply this declaration). P05 sees a foul only when the
players slow down, about a second after the whistle, so a detection is **near a stoppage** when a decoded `D` frame lies in
[w - 25, w + 50] frames (-1 s to +2 s) inside w's graded block. It is never measured on the exact frames after the whistle,
which would count the most useful whistles (the ones P05 is late for) against it. No ground truth, no event data, no hand
label: the only reference is our own decoder.

**The control.** The same detections shifted circularly inside their own half by 5, 10 ... 60 s (12 offsets); the control's
share is the mean over the offsets.

**The feasibility gate** (`whistle_stats.py feasibility`). Proceed to the source only if all three hold:
1. detections per half are between 10 and 120;
2. at least 50% of the detections are near a decoded stoppage;
3. that share is at least 2 times the control's.

**The bar** for adoption, on the built source (`whistle_stats.py bar`): the share near a decoded stoppage is at least 0.70,
its lower 90% bound (2,000 bootstrap draws over the detections, `default_rng([18, 0])`) is at least 0.60, and it is at least
1.5 times the control's.

**The strength in the decoder.** If the bar passes, the source's two strengths (`spread = "frame"` at weight 1.0, and
`spread = "event"`) are run on BAR-ATH, the aggregates (`label_diff`: the share of labels changed, how many in gaps, `D`
onsets moved and their median move) go to Tian, and Tian picks before `main.toml` changes. The source's confidence c_whistle
is the precision measured here (the share near a stoppage), rounded towards 0.5 (`train_dead_ball.towards_half`).

**Label-free checks on BAR-ATH.** As P08's: the audio decode is deterministic across two runs (the same feature-cache
sha256); the number of detections with a feature within 0.5 dB of its threshold is reported (the ones a different ffmpeg
build could flip), as an aggregate. Only counts, shares, quartiles, hashes and pass/fail leave the machine; no sample, spectrum
or per-frame feature is printed or written.

**The retry.** If the gate fails, one retry is allowed with one changed detector (R018b), declared in its own part before it
runs. If R018b also fails, P10 is dropped and its code removed, as P18 was (R015b). The public window chooses nothing.

**The public window.** It is not used here at all.

| ID | status |
|---|---|
| R018 part 1 | declared, run (gate failed) |

**Part 2, the feasibility result** (2026-10-30, `whistle_stats.py feasibility`, aggregates only). The gate **FAILED**.
- 340 detections, 185 in half 1 and 155 in half 2, where the gate allows 10 to 120 per half (gate 1 fails);
- share near a decoded stoppage 0.624 against a control of 0.410, ratio 1.52, where the gate needs at least 2 (gate 3 fails;
  gate 2, at least 50%, passes);
- 240 of the 340 detections have a feature within 0.5 dB of a threshold;
- the lag from each detection to the nearest decoded stoppage edge (signed, edge minus whistle) has quartiles -253 / +16 / +193 frames.

The run was at commit `6594b31`. `main.toml` sha256 at the run: raw `ad8d8b6c…`, LF `0970f2de…`, matching the declaration; the
features' sha256 `a3d80f77…`. Determinism check (`p10_decode_check.log`): the features were decoded twice without the cache, with
the same sha256, shape (199319, 2) and 0 NaN, and the cache built equal to both decodes. That check ran 22:39 to 22:40, before part
1 was committed at 22:46; it printed only the shape, the NaN count and a hash, so no value shaped a constant.

Reading: the fixed `TONAL_DB = 15` dB sits inside the crowd noise, so the detector fires on it. The retry R018 part 1 allowed
is R018b, declared next. Nothing else is read into the failure.

### R018b — The whistle's tonal threshold from the match's own duty cycle (2026-10-30, Tian)

Proposal: P10 · The one retry allowed by R018 part 1.

**Part 1, declared before any run** (2026-10-30), before any code of it is written.

**The single change.** The tonal threshold is no longer a fixed 15 dB. It is the 98th percentile of `tonal_db` over the graded
frames of the match, computed at run time from the features alone (`np.percentile(features[graded, 0], 98.0)`); no label, no
stoppage and no score enters it.

**The duty-cycle argument.** A referee whistles roughly 30 to 50 times a half, each for under a second, so at most about 2% of
graded frames carry a whistle. A threshold that passes 2% of the frames passes about as many frames as whistles can occupy, and
no more; R018's fixed threshold passed far more and so fired on crowd noise.

**Everything else is unchanged**, by reference to R018 part 1: `BAND_DB` 3.0, `MIN_FRAMES` 5, `WINDOW_FRAMES` 12, the
pseudo-labels (`main.toml`, sha256 as declared there), the near-a-stoppage window [w - 25, w + 50], the control, the
feasibility gate (10 to 120 detections per half; at least 50% near a decoded stoppage; at least 2 times the control's share),
the bar and the strength in the decoder. The comparison is `>=` against the computed threshold, which is rounded as the features
are (0.01 dB). The public window is not used.

**The gate** is R018 part 1's, unchanged: all three conditions must hold to proceed to the source.

**Reported.** The computed threshold value (dB) is reported as an aggregate, with the number of detections near it (within
0.5 dB of the tonal threshold or of `BAND_DB`) as in R018.

**If R018b's gate fails, P10 is dropped and its code removed**, as P18 was (R015b). There is no further retry.

**Part 2, the feasibility result** (2026-10-30, `python pipeline/benchmark/whistle_stats.py feasibility` at commit `a08e381`,
aggregates only). The gate **FAILED**.
- `main.toml` sha256 raw `ad8d8b6c…`, LF `0970f2de…`, both matching the declaration; the features' sha256 `a3d80f77…`, the same
  as the earlier decodes;
- tonal threshold 24.76 dB (the match's 98th percentile);
- 55 detections, 25 in half 1 and 30 in half 2 (gate 1 passes, 10 to 120 per half);
- share near a decoded stoppage 0.564 against a control of 0.373, ratio 1.51, where the gate needs at least 2 (gate 3 fails;
  gate 2, at least 50%, passes);
- 29 of the 55 detections have a feature within 0.5 dB of a threshold (a rough indicator, not a tested quantity);
- the lag to the nearest stoppage edge has quartiles -171.5 / +20 / +292.5 frames.

The run was at commit `a08e381`.

Reading: with 340 marginal detections (R018) or 55 stronger ones (R018b; 29 of the 55 are within 0.5 dB of a threshold), detected
whistles sit near a decoded stoppage about 1.5 times as often as the same detections shifted in time. That is consistent with a
real but weak signal (no test of "real" was declared), too weak for the declared 2 times, and a stricter threshold does not
sharpen it.

**Verdict: P10 dropped; code removed in `fc9dbed`, recoverable from `a08e381`.** R018 part 1's rule applies: R018b was the one
retry. After the removal: `pipeline/selftest.py` 185/185, `pipeline/benchmark/selftest.py` 410 checks, `main.toml` unchanged.

| ID | status |
|---|---|
| R018 part 2 | feasibility gate failed (340 detections; 0.624 vs 0.410, ratio 1.52) |
| R018b part 1 | declared, run |
| R018b part 2 | feasibility gate failed (55 detections; 0.564 vs 0.373, ratio 1.51); P10 dropped, code removed |

### R019 — The camera's aim in the shape model, with a simulated camera on DFL (2026-10-30, Tian)

Proposal: P09 · Design: `docs/superpowers/specs/2026-10-30-p02-p08-p09-p10-p12-design.md` section 5.E · Branch
`p09-camera-aim`. This entry grows in parts; each part is committed before the run it governs and is never edited after
a run.

#### Part 1a — the camera's gates (declared before the first simulation)

Status: **declared, not run.** Nothing below has been measured on a DFL match. The fit of our camera and its statistics
on BAR-ATH are label-free aggregates of our own tracking and homography; the DFL side is public data.

**What is simulated.** A camera like ours on each DFL match, so that a feature built on the aim can be measured with a
real F1 test. The aim is the pitch point under the image centre (`inputs.camera_aim`, metres along the pitch), and our
camera is modelled as a follower of our ball, plus noise:

    aim(t) = a + b * lowpass(ball_x, tau)(t - lag) + e(t),     e(t) = phi * e(t - 1) + sigma * z(t)

**The fit's procedure (`camera.fit`, `pipeline/benchmark/camera.py`).**
- lowpass is a first-order exponential smoother, alpha = 1 / (1 + tau), holding its last value across a missing ball and
  starting again at each graded block.
- Search lag in 0 to 80 frames and tau in {0, 5, 10, 20, 35, 50} frames; for each pair `a` and `b` by least squares of
  the aim at t on the smoothed ball at t - lag. The pair with the smallest residual variance is kept (a tie goes to the
  smaller lag, then the smaller tau); phi is the residuals' lag-1 autocorrelation and sigma their standard deviation
  times sqrt(1 - phi^2); lo and hi are the 0.5th and 99.5th percentiles of our aim on the graded frames, to which the
  simulated aim is clipped. Nothing is random in the fit.
- **Settled frames.** Our smoother holds its last value through our tracker's gaps while the real camera kept following
  the real ball, so just after a gap our smoothed ball is stale. A frame's regressor counts only once the smoother has
  seen at least ceil(3 * tau) visible frames in a row since the last gap (the block's start counts as one; tau 0 needs
  only the frame itself). The stale part left is then about e^-3 = 5%. Candidates are compared on their own settled
  frames. Without the rule the fit would be biased towards a small tau and a larger sigma.
- Cached by the code and data it depends on (`load_fit`), digest of the eight constants (`camera.digest`).
- The simulator drives the fit with DFL's clean ball (the real camera follows the real ball even where ours loses it),
  seeded by (seed, match id), NaN off the graded blocks.

**The gates (`python pipeline/benchmark/camera.py gates --profile P --seeds 3`).** The same `measure()` on both sides,
on the `tracking` v3 copies (7 matches x seeds 0 to 2, rows pooled by frames, never medians of medians), against
BAR-ATH's own. `dead` is what `main.toml` decodes as D (pipeline functions: `pipeline.benchmark.run.predict` on each
served DFL match, the same decoding on BAR-ATH), never the truth. Tolerances, declared here and not tuned towards:

| gate | statistic | pass if |
|---|---|---|
| 1 | peak lag (lag of the peak correlation of the aim with the ball seen) | \|sim - ours\| <= 10 frames |
| 2 | peak correlation | \|sim - ours\| <= 0.05 |
| 3 | median along-pitch error at the peak lag | \|sim - ours\| <= 3.0 m |
| 4 | speed ratio dead / alive (median aim speed on frames decoded D over frames decoded H or A) | \|sim / ours - 1\| <= 0.25 |

A value exactly on a tolerance passes (differences are rounded to 9 decimals); a statistic that cannot be measured on
either side fails its gate. Exit code 1 if a gate fails.

**The WARN (not a gate).** `speed_ratio_gap_visible`: the aim's median speed on frames without the ball seen over
frames with it (ours 0.58 / 1.22, A03), on `tracking` and on `context` (`--profile context`). It is printed with the
same +-25% band and never fails the command. `tracking` v3 places its gaps blind to the play (T38), so the simulated
camera keeps moving through them and the ratio will sit near 1 by construction; a large gap between the two profiles is
the T38 warning that the benchmark may overstate P09. R019's veto already includes `context`.

**The retry rule.** If a gate fails, one retry with a single change, declared (what and why) before it runs, and
proposed to Tian for a decision first. If that also fails, P09 is dropped and its code removed. The pause (the research
bar) applies in any case: the gates' results go to Tian before any retraining.

**Amended before any run (2026-10-30).** Nothing had been run when these were written (the first run is Task E.3's
step 7, after this commit); they add to the declaration above and change none of its tolerances:
- **The decoder of `dead`.** The gates' D is `main.toml` decoded with its match-only sources dropped
  (`run.drop_match_only`: any source that declares MATCH_ONLY, since a DFL match lacks the asset it needs; today Smart
  Tagging (P08, adopted into `main.toml` on main) is the only one, and the whistle (P10) was dropped), on both sides,
  BAR-ATH and DFL, so that one decoder and the same code serve both. The names dropped are printed and kept in the
  report (`dropped_match_only`).
- **Only `tracking` decides.** Under `--profile tracking` a failed gate gives exit code 1. Under `--profile context`
  (a sensitivity profile, R005) the four gates are printed as information and never set the exit code; what is asked of
  `context` is the WARN.
- **One camera per selection.** Reports in one `choose.py` selection must share the camera digest (`camera` is in
  choose's SAME), so that one selection never compares reports of two camera fits.
- **Pooling.** DFL is pooled by frames across matches x seeds (the pooled rows are measured once), so a longer match
  weighs more; it is not a mean over matches.
- **Which models decode DFL.** On DFL the decoded D uses each match's held-out (out-of-fold) models, `folds[inputs.game]`,
  as the benchmark's runs do; BAR-ATH uses `full`. The two sides' decoders differ by that, as everywhere in the benchmark
  (T38, T39), and the gate's tolerance is not changed for it.

**What follows, not governed here.** Parts 2 and 3 (the shape model v2 and the benchmark's bar over the candidates at
weights {0.2, 0.3, 0.5}) are declared when Task E.4 starts, before their runs.

#### Part 2a — the gates' results (run 2026-10-30)

Run at `bc35178` on branch `p09-camera-aim`: `python pipeline/benchmark/camera.py gates --profile tracking --seeds 3`,
then the same with `--profile context`, 7 matches x seeds 0 to 2. The branch's `main.toml` decoded `dead` on both sides
and had no match-only source to drop (it predates P08's adoption). Reports: `data/runs/camera/gates_<profile>.json`.

**The fit** (camera digest `e4adfed3e482`): lag 0 frames, tau 50 frames, a 23.314 m, b 0.5793, phi 0.999971,
sigma 0.0674 m, lo 9.53 m, hi 93.88 m. 74,694 settled frames, R squared 0.719, residual sd 8.78 m. Both lag and tau are
on an edge of the declared grid: lag 0 is the smallest lag searched and tau 50 the largest smoother.

**The gates** (ours: 94,214 pairs; simulated: 1,747,071 pairs on `tracking`, 1,709,934 on `context`):

| gate | ours | `tracking` sim | delta | result | `context` sim | delta | result (information) |
|---|---|---|---|---|---|---|---|
| 1 peak lag (frames) | 37 | 40 | +3 | pass | 41 | +4 | pass |
| 2 peak correlation | 0.843 | 0.9115 | +0.0685 | **FAIL** | 0.9002 | +0.0572 | FAIL |
| 3 median error (m) | 11.65 | 12.69 | +1.04 | pass | 11.84 | +0.19 | pass |
| 4 speed ratio dead / alive | 0.1277 | 0.5754 | +351% | **FAIL** | 0.5786 | +353% | FAIL |
| WARN speed ratio gap / visible | 0.1671 | 0.9829 | +488% | WARN | 0.8751 | +424% | WARN |

The aim's median speed (m/s), the four numbers behind the two ratios:

| | alive | dead | ball seen | ball not seen |
|---|---|---|---|---|
| ours | 0.298 | 0.038 | 0.254 | 0.043 |
| simulated, `tracking` | 1.951 | 1.123 | 1.516 | 1.490 |
| simulated, `context` | 1.947 | 1.127 | 1.595 | 1.396 |

Part 1a quoted A03's gap/visible speeds (0.58 / 1.22); those came from A03's own measurement, and the gates compare
`measure()` with `measure()` on both sides.

**What the numbers say** (readings, not measured causes):
- The simulated camera never stands still. While play is alive it moves at 6.5 times our camera's median speed, while it
  is dead at 30 times. With phi this near 1 the noise's innovation is the residual's frame-to-frame change, and its
  median step alone is 0.6745 x 0.0674 m a frame, 1.14 m/s, about four times our camera's median speed in play. Our
  camera nearly stops whenever play is dead (0.04 m/s) or our ball is unseen (0.04 m/s); the follower plus that noise
  cannot.
- The simulated aim correlates 0.07 more with its ball than ours does. Two plausible readings: the served ball is a copy
  of the very ball the simulated camera follows, while ours is a tracker's estimate of the ball our camera follows; and
  with phi at 0.99997 the noise's correlation time (about 34,000 frames, 23 minutes) is half a graded block, so within a
  block it covers less than its stationary variance.
- For P09's premise: on frames without our ball the aim's position still says where our camera points, but the camera
  barely moves there, so a velocity of the aim (`aim_vel`, Task E.4) would be near zero on most of them.

**Status: `tracking` FAIL on gates 2 and 4** (`context`, information only: the same two). Under part 1a's retry rule, Tian
decides between one retry with a single declared change and the drop. No shape model has been retrained.

#### Verdict — dropped (2026-10-31)

At the research bar's pause, Tian decided between one retry and the drop: **drop**. The two retries put to Tian, with
the expected outcome of each, were:
- widen the fit's grid (lag -80 to 80 frames, tau up to 250 frames, as the fit sat on both edges, lag 0 and tau 50):
  might fix the correlation, expected not to fix the dead/alive ratio;
- hold the simulated camera through DFL's dead balls: expected to push the ratio to about 0 (the median over mostly
  frozen frames), also a fail, and to leave the correlation as it is.

Reasons: no single declared change was expected to pass both failed gates, so a retry would have been a guess, which
the research bar rules out; and our camera barely moves where our ball is missing (0.04 m/s), so the aim's velocity
carries little in the very frames P09 was meant to help (T47).

Code removed in `8c4c959` ("P09 dropped: the camera's aim and its simulator removed (recover from 13c36c6)"), which
puts every file under `pipeline/` back to `bfeee45` (main before P08 and P10): `pipeline/inputs/camera.py`,
`Inputs.camera_aim`, `MatchInputs(camera_aim=)`, `pipeline/benchmark/camera.py` and its checks, `camera` in
`choose.py`'s SAME, and the camera's plumbing through `run.py`, `headroom.py`, `train_shape.py`, `train_dead_ball.py`,
`conventions.py` and `benchmark/inputs.py`. Recover them from `13c36c6`. No shape model was retrained, and R019's parts
2 and 3 were never declared. Gitignored leftovers on the machine that ran it: `data/runs/camera/` and
`data/cache/benchmark/camera/`.

## Submission log

The form keeps only the last response before the deadline, so log every upload.

| date | file | sha256 | commit | command | notes |
|---|---|---|---|---|---|

`main.toml` was rewritten by R010 (2026-10-25): it then produced sha256
`a706e7f711b2d3faacc2f071b6e28963ed8e2f2326a7880e6d0776cce4295f1f` (P04 `controlled_touch` + P05
`dead_ball_model` hgb + the `hmm` (4, 32, 25, 25)), identical to `p45_hmm.toml`'s file. It was NOT uploaded.

`main.toml` was rewritten again by R011 (2026-10-25): it then produced sha256
`8810afce254e94fe159769f3b55e8eef65f41a25c84f4550034e2afbc20e8487` (P04 `controlled_touch` + P05
`dead_ball_model` hgb + P07 `shape_possession` at weight 0.2, scope `all` + the `hmm` (4, 32, 25, 25)), identical
to `p07_shape.toml`'s file; R010's file is still what `p45_hmm.toml` writes. It was NOT uploaded.

`main.toml` was rewritten a third time by R012 (2026-10-25): it then produced sha256
`3fd06da011bde39f771ddd22f00da2bcbcf5205720d60f87662ab6fc0f76a702` (P04 `controlled_touch` + P05
`dead_ball_model` hgb + P07 `shape_possession` at weight 0.2, scope `all` + the `hmm` (4, 32, 25, 25) + P02
`shift_edges` team +1, into_dead -2, out_of_dead 0), identical to `p02_shift.toml`'s file; R011's file is still what
`p07_shape.toml` writes. It was NOT uploaded.

`main.toml` was rewritten a fourth time by R013 (2026-10-26): it now produces sha256
`0ebdfbeca72ae2d09eb7aaa377037415696a5dacb95e29003741e0f57d5dbc5d` (P04 `controlled_touch` 1.5 m / 6.0 m/s / 1
frame / hold 0.60 + P05 `dead_ball_model` hgb + P07 `shape_possession` at 0.2, scope `all` + the `hmm` (4, 32, 35,
25) + P02 `shift_edges` +1, -2, 0), identical to `r013_retune.toml`'s file; R012's file is still what
`p02_shift.toml` writes. It has NOT been uploaded.

R014 and R014b (2026-10-26) left `main.toml` unchanged: neither met its bar, so it still produces sha256
`0ebdfbeca72ae2d09eb7aaa377037415696a5dacb95e29003741e0f57d5dbc5d` (R013's). It has NOT been uploaded.

`main.toml` was rewritten a fifth time by R017 (2026-10-30): P08 `smart_tagging` (per frame, `r017_frame`) adopted.
It now produces sha256 `7ba575c0982aacb9765b39c406e4ff5c9ddb82ca37cfa2b7f101417431fc3d7c` (commit 56916f2). It was
NOT uploaded.

Before each upload:
1. Regenerate the file from a clean clone, ideally on a second machine, and compare the sha256 ([T21](traps.md)).
2. Run `validate_submission.py` on the exact file.
3. Name it `<team_name>.csv`.
