# Results

Everything we run that is worth remembering: experiments *and* analyses, including the ones that failed. A number
without the commit and command that produced it is not a result.

Public-window scores alone decide nothing ([T01](traps.md)). Run `--bootstrap 1000` before calling a gap real, and
add the [P01](proposals.md#p01--a-validation-set-we-can-trust) validation score once it exists. Record aggregates
only — never raw data rows ([CLAUDE.md](../CLAUDE.md)).

## Scoreboard

Our runs, best first. *Public* means the 5-minute public GT (17 possessions, mean 14.2 s); *val* means the P01
benchmark: other matches (DFL open data), degraded to look like ours. Hand labels on this match are not allowed
([T27](traps.md)).

| ID | date | what | public macroF1 | public spellF1 | poss / dur | val spellF1 | verdict |
|---|---|---|---|---|---|---|---|
| R001 | 2026-10-22 | `main`: pipeline skeleton, i.e. R000c per half | 0.6207 | 0.3256 | 26 / 10.8 s | — | keep as the base; the floor, not a candidate |

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
integer count quoted here (possessions, short runs, stoppages, window counts, changes, the coarse truth's possessions),
the 60% agreement and DFL's playing time to within 2 frames; the rates, shares, medians, lags and line timings are
printed, not asserted. The rules are checked on hand-built matches by `test_truth_*`
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

## Experiments

Copy this template for each run, and add a scoreboard row for anything worth comparing.

```
### R0xx — <short name> (YYYY-MM-DD, <who>)
Proposal: P0x · Commit: `abc1234` · Command: `…`
Public: macroF1 … · spellF1 … (bootstrap … – …) · poss … · dur … · lat_f …
Validation (P01 benchmark): spellF1 … (… matches, degradation seed …; bootstrap … – …)
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

## Submission log

The form keeps only the last response before the deadline, so log every upload.

| date | file | sha256 | commit | command | notes |
|---|---|---|---|---|---|

Before each upload:
1. Regenerate the file from a clean clone, ideally on a second machine, and compare the sha256 ([T21](traps.md)).
2. Run `validate_submission.py` on the exact file.
3. Name it `<team_name>.csv`.
