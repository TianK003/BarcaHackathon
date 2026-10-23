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
| R001 | 2026-10-22 | `main`: pipeline skeleton, i.e. R000c per half | 0.6207 | 0.3256 | 26 / 10.8 s | — | keep as the base; the floor, not a candidate |
| R002 | 2026-10-23 | the four yardsticks `nearest_w{0,25,50,125}` on the benchmark's `clean` profile | — | — | — | 0.3073 raw (`w50`), 0.3257 coarse (`w125`); clean, so not our match's | the benchmark's floor, not a candidate; the truth reorders them |
| R003 | 2026-10-23 | the same four yardsticks on the benchmark's `gaps` profile: our own ball gaps transplanted onto the DFL matches | — | — | — | 0.2102 raw (`w50`), 0.2302 coarse (`w125`); `w0` last under both | `gaps` v1 frozen, the first benchmark that looks like our match; the yardstick check holds (`w0` last), the rest of the order follows the truth |
| R004 | 2026-10-23 | the same four yardsticks on the benchmark's `tracking` profile: our players' tracklets and slots, wrong-team slots, NaN-speed runs and smoothing on top of `gaps`' ball | — | — | — | 0.2000 raw (`w50`), 0.2195 coarse (`w125`); `w0` last under both | `tracking` v1 frozen (six gates pass, two pending until the low-count tail, no WARN); the yardstick check holds, no order changes from `gaps` |

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
Validation (P01): none yet.
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
from our first half and 21 from our second, at offsets from 0 to 11,222 frames.

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
now and then, NaN speeds in runs and smoothed positions. Not yet our frames with few players tracked (the tail, the next
task), so two of the eight fidelity gates are pending. Reports: `data/runs/benchmark/tracking/<stem>.json` and
`<stem>.coarse.json`, and `data/runs/benchmark/fidelity_tracking.json` (not committed; each carries the Calibration's
digest).
Checks: `pipeline/benchmark/selftest.py --dfl --data` passes all 235 checks (215 synthetic, 20 with `--dfl` and
`--data`), `pipeline/selftest.py` 35 of 35 and its `--data` run 37 of 37 (the four baselines identical byte for byte).
The new checks were written first and failed first (16 of 29 checks of the degrader and 6 of 10 of the fidelity report,
for the missing parts). The new code was run against 18 mutants, each caught by a check: a tie that goes to the highest
columns, 2.0 m not counted as within, an overlap not skipped, the draws of picks and durations swapped, the pick the
other way, an episode left in its own pool, a NaN run that goes on past its tracklet, the first rows not counted towards
the NaN target, a smoothing window that does not shrink, onsets drawn from the walk's stream, an onset rate without the
scale, lengths rounded down, calibrated parameters with no smoothing, a relative tolerance read as an absolute one, a
pending gate that is counted, a failed gate that does not fail the run, a median of medians, a NaN share over all
frames. With the new parts off the players are Task 12's byte for byte, and that was checked against the code of
`ac940b2` itself, not only by the reference in the check: the same digest of positions, speeds and records on two
synthetic matches under three seeds and on J03WMX under seeds 0 and 1.

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

**Fidelity first** (`fidelity.py --profile tracking`, 7 matches x 3 seeds, 78 s with the caches warm; the centres are
the Calibration's own values on our kick-off spans, the tolerances the plan's; `tracking` was frozen at version 1 once
the six required gates passed, and nothing of it was changed after, whatever the checks said):

| gate | benchmark | our match (centre) | tolerance | |
|---|---|---|---|---|
| ball missing, share of frames | 0.4189 | 0.4110 | +-0.02 | pass |
| median gap in the ball | 10.76 s | 10.74 s | +-1.5 s | pass |
| exactly 22 slots present, share of frames | 0.4598 | 0.5019 | +-0.06 | pass |
| mean slots present | 21.252 | 20.522 | +-0.5 | **pending** (off by 0.73) |
| 14 or fewer slots present, share | 0.0000 | 0.0525 | +-0.02 | **pending** (off by 0.0525) |
| wrong-team share of frames | 0.1005 | 0.1054 | +-0.03 | pass |
| tracklet median | 29.26 s | 41.28 s | +-50% (+-20.64 s) | pass (-29%) |
| NaN-speed share of the present slot-frames | 0.0787 | 0.0783 | +-0.01 | pass |

The two pending gates depend on the low-count tail (our frames with 18 or fewer slots present, 10.6%), which the profile
does not have yet: they are printed and not counted, and `fidelity.PENDING_GATES` is the list that the tail's task
empties. Both would miss if they were counted, by the amounts in the table, which is what the tail is for. The tracklet
median is the nearest of the others to its edge: 29.3 s is 29% below ours (the gate allows 50%), because the episodes
cut tracked spans into more and shorter tracklets (31,398 tracklets over the 21 (match, seed) runs, 1,495 each, against
1,144 in ours). The 42 half-draws took 21 masks from our first half and 21 from our second, as in R003.

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

Runtime: 78 s for `fidelity.py` on all seven matches and three seeds, and 52 s for a run of the four configs (7 matches
x 3 seeds x 4 configs, against 31 s on `gaps`), with the DFL caches and the Calibration cached; the same command twice
writes the same bytes (the figures of the working-tree run and of a second run are identical).
Verdict: keep `tracking` v1 as the benchmark's profile for what depends on identities, wrong-team slots and speeds, and
the four yardsticks as its floor, not as candidates; `DEFAULT_PROFILE` stays `gaps` until the tail is built and its two
gates count. The yardstick check passes in its robust part under both truths, the profile reproduces what it was
calibrated on (six required gates, no WARN), and it does not make the truth's grain a non-issue: choose between `w125`
and the others only where the raw and the coarse truth agree. The profile's players are rougher than ours (jumps, T35)
and have no positional bias (the README's limitations), and it has no tail yet.

## Submission log

The form keeps only the last response before the deadline, so log every upload.

| date | file | sha256 | commit | command | notes |
|---|---|---|---|---|---|

Before each upload:
1. Regenerate the file from a clean clone, ideally on a second machine, and compare the sha256 ([T21](traps.md)).
2. Run `validate_submission.py` on the exact file.
3. Name it `<team_name>.csv`.
