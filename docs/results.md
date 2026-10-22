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
timeline is `pipeline/benchmark/dfl.py`'s: the first half, 25 separator rows, then the
second half (J03WMX: 145,992 rows, the second half starts at row 70,733).

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
the sha256 of both source files, the numpy and lxml versions, `CACHE_VERSION` and the sha256 of `dfl.py` itself.
`test_dfl_all_matches_load` loads all seven twice, and `test_dfl_cache_equals_fresh_parse` compares J03WMX from the
cache with a fresh parse (0 differences in all nine arrays, the scalars and the report):

| | no cache (built) | cache (reused) |
|---|---|---|
| One match | 41.7 to 49.1 s | 0.7 to 1.9 s |
| All seven | 325.8 s | 5.5 to 6.7 s |
| Whole `--dfl` run, 59 checks | 532 s | 221 s |
| `test_dfl_t30_facts` | 297.8 s before the cache (one parse per match) | 4.7 s |

- A reused match is read in under 2 s, almost all of it the sha256 of the two source files (the positions file is up
  to 420 MB), which is what makes a stale cache impossible. The caches take 129 to 141 MB per match, 956 MB for the
  seven.
- What the reused `--dfl` run still spends: kloppy's own loads (about 100 s), the fresh parse of the cache check (about
  50 s), the spot check's own pass over the file (about 34 s) and the 2.6 GB md5 check (about 10 s).

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
