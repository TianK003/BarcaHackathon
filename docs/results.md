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
- **D timing against ours is unmeasured.** DFL's counterpart needs BAR-ATH's ball, which is P02.
- **Use:** hyperparameters are picked on the benchmark, never on the public window (CLAUDE.md, T01). Where a choice
  depends on how long a possession is (smoothing, a switch penalty λ, duration priors), rank it under the raw truth and
  under the coarse one side by side, and treat a disagreement between them as a warning, not as a licence to pick the
  one that suits. Fit timing (P04's back-dating, P05's `D` onset, P12's shift) on the public window through P02.
  Absolute spellF1 on the benchmark is not our leaderboard's under either truth.
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
Public (a correctness check only, T01): macroF1 0.6393 · spellF1 0.3500 · alive/dead F1 0.8895 · 23 possessions against
the truth's 17, mean 11.4 s against its 14.2 s · transF1 0.3043 · lat_f -10.1. The file is valid (199,319 rows) and
both runs wrote the same bytes, sha256 `9705a88583d165fb67dca2a917c170e83dbf38749ec8a15af64a92c8816ba8fb`. Over all the
graded frames of the match it labels H 33.7%, A 24.0% and D 42.3%, and finds 393 possessions (3.71 per graded minute,
mean 9.3 s) against `main`'s 594 (5.61 per graded minute, R001). The run was made at `9bbc3e3` with the file that
`bdd699b` then committed, so the manifest (`data/runs/p06_hmm.json`, not committed) says `+dirty`. `--score` reports
`nearest_player` agreeing with the public GT on 77.8% of 5,640 frames (A01's figure) and `ball_presence` on alive on
81.8% of 7,500 frames. No bootstrap was run, because no public-window gap is claimed. The public scores are in-sample
for the three evidence values A01 measured on those frames (`confidence` 0.78, `missing` 0.2 and `seen` 0.585).
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

## Submission log

The form keeps only the last response before the deadline, so log every upload.

| date | file | sha256 | commit | command | notes |
|---|---|---|---|---|---|

Before each upload:
1. Regenerate the file from a clean clone, ideally on a second machine, and compare the sha256 ([T21](traps.md)).
2. Run `validate_submission.py` on the exact file.
3. Name it `<team_name>.csv`.
