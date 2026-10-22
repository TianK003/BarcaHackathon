# Traps

Things that cost us time or would cost us points. Add one whenever something takes more than half an hour to figure
out: what you saw, why it happens, what to do instead. Give it the next free ID and put it in the right section.

## Scoring

**T01 — Tuning on the public window.** The public window holds 17 possessions, so one possession is worth about
0.06 spellF1. It is also the best-covered 5 minutes of the match: the ball is missing on 12% of its frames, against
40.7% overall and up to 80% per 5-minute block ([A02](results.md#a02--ball-coverage-across-the-match-2026-10-21)).
And it ranks `smooth125` above `smooth50`, the reverse of the full match. Use it to check correctness, and decide on
P01 validation with `--bootstrap 1000`.

**T02 — `X` means "repeat my last label" — and at the start of a half, "copy my first label".** Before computing
spellF1, the scorer forward-fills `X` with the last non-`X` label in the half, and back-fills the head of the half.
- The second half has 2,934 graded pre-kick-off frames (110,808–113,741), all of them `D`. An `X` across them becomes
  whatever we say at kick-off: a long spurious possession.
- An `X` after the ball goes out extends the possession into the stoppage.

Write `D` in both cases.

**T03 — Latency is expensive.** Perfect labels 1 s late score spellF1 0.65, and 8 frames late score 0.89
(EVALUATION.md). The shortest public possessions tolerate only 0.3–1.5 s of total boundary error at IoU 0.7 (A01).
Hysteresis and "k frames of control" add lag unless the change is back-dated to where the evidence starts. Check
`lat_f` on the board and the latencies in `evaluation/reports/<name>_transitions.csv`.

**T04 — A blip costs three possessions.** Five frames of `A` inside an `H` possession split it in two and add a
spurious `A`; a short `D` blip splits it too. But real short possessions exist — 0.76 s and 1.32 s in the public GT —
so minimum-duration filters need validating, not guessing.

**T05 — `H H D D H H` is two possessions.** Missing a stoppage merges the possessions on either side of it, and
inventing one splits a possession.

**T06 — Possession counts differ between tools.** `validate_submission.py` treats `X` as a gap and counts over the
whole file. The scorer bridges `X` and counts inside the graded halves only. Compare like with like: A01 counts the
baselines over the graded halves.

**T07 — Noise floors.** On the full match, spellF1 gaps below ~0.04 and macroF1 gaps below ~0.01 are not real. On the
public window, one possession is worth ~0.06.

## Alignment

**T08 — Off by one.** ATD `frame_id` runs 1…199,319, while submission rows, homography `frame` and video frames all
start at 0. Smart Tagging uses seconds: frame = `round(s * 25)`.

**T09 — ~164k rows means you indexed the broadcast.** The broadcast has its own timeline and no mapping to the
tactical one.

**T10 — The `offsets` in `games.py` are not for the ATD.** They map to the internal IdFrame counter the GT was derived
from.

**T29 — Two answers for each kick-off frame.** DATA.md, FAQ.md and `starter/load_smart_tagging.py` put the kick-offs
at frames 11,185 and 113,742. The `(minute, second, frame)` anchors in `games.py` convert, through
`games.anchor_video_frame`, to 11,186 and 113,741: one frame off, in opposite directions. Use the documented pair,
which the graded windows agree with (the first half is graded from its kick-off). The pipeline exposes it as
`inputs.kickoffs`.

**T11 — Temporal code must run per half.** The baseline's majority vote ignores half boundaries; ours must not. Use
`games.half_video_blocks("BAR-ATH")`, which gives 11,185–90,651 and 110,808–190,197. The first half is graded from its
kick-off frame; the second is graded from 2,934 frames before its kick-off at 113,742.

## Data

**T12 — A missing ball is not a dead ball, in either direction.** In the public window the ball is visible on 66% of
dead-ball frames (placed, held, thrown in), and over the match, in-play gaps of 10 s or more are normal (A01).

**T13 — Ball gaps are long, not flickers.** There are 150 gaps over the graded halves: median 10.7 s, p90 34 s,
max 3.7 min. Gaps under 2 s hold 0.1% of the missing frames, so interpolation buys almost nothing (A01). Coverage also
gets worse later in each half (A02).

**T14 — The nearest player is not the possessor.** The nearest player's team is wrong on 22% of in-play,
ball-visible public frames (A01): passes in flight, duels, deflections.

**T15 — Track slots are tracklets, not players.** There are 46 slots for 22 players; they appear, vanish, swap, and
occasionally duplicate someone (more than 22 objects). `ShirtNumber` is just the slot index. Build collective,
permutation-invariant features. Not yet checked: does a slot ever carry the wrong team after an identity switch?

**T16 — Units and orientation.** Coordinates are normalised 0–1, and the pitch is not square.
- Scale by 105.3 × 68.0 m (`games.py`; the metadata says 105 × 68) before computing distances.
- Feed the homography normalised coordinates, not metres.
- Teams swap ends at halftime, so derive each team's attacking direction per half before building directional
  features.

**T17 — Smart Tagging quirks.**
- Codes are upper case, values are title case, and kloppy's tag columns are title case, so match case-insensitively.
- The away team is spelled `Atletic Club`.
- 36 tags have Team `N/A`, including every throw-in.
- Windows open seconds early (kick-off tags by ~3.5 s), and phases are not possessions.
- `DEFENDING …` and `DEFENSIVE TRANSITION` name the team *without* the ball.

**T18 — The camera lags the ball.** The tactical camera's aim follows the ball's along-pitch position about 35 frames
late, with a median error of 12 m, and carries no information about the across-pitch position (A03).

**T28 — Speed is not presence.** The ATD's speed channel is NaN on 370,297 slot-frames and 103,579 ball frames where
the position exists (measured 2026-10-22). Test presence on x, which is always missing together with y; the
pipeline's `atd.players_present` and `atd.ball_present` do exactly that.

**T19 — kloppy is slow on the full ATD.** Loading all 199,319 frames takes minutes and gigabytes. Stream the raw file
once, as `starter/baseline.py` does, and cache arrays under `data/cache/`.

**T30 — kloppy's DFL loader hides the dead ball.** Both of kloppy's DFL loaders,
`kloppy.sportec.load_open_tracking_data` and `kloppy.sportec.load_tracking`, default to `only_alive=True` and silently
drop every dead-ball frame. That leaves no `D` to learn or score. Pass `only_alive=False`.

Measured on the seven DFL matches on 2026-10-22 (kloppy 3.18.0; `python pipeline/benchmark/selftest.py --dfl`):
- **The default drops the dead ball.** On J03WMX, `only_alive=True, limit=5000` returned 5,000 frames over N 10,000 to
  15,823: none is dead in our parse, no alive row in that range is missing, and the 824 dead rows are the ones it
  dropped. Dead frames are 40.7% to 51.7% of the ball frames of a match (J03WQQ the least, J03WN1 the most), so the
  default deletes about half of every match.
- **`limit` is per object and counts what the loader keeps.** With `only_alive=False` it stops after the first 5,000
  frames of the first half (N 10,000 to 14,999, period 1 only). With the default it counts alive frames, so the same
  limit reaches further.
- **With `only_alive=False` it agrees with our parser** on those 5,000 frames: ball x, y and z, alive or dead, the
  team in possession and three players, with 0 disagreements. Our frame numbers and its `frame_id` are the same N.
- **The other two behaviours never fire on these files.** The loader reads `BallPossession` as home when it is 1 and
  away otherwise, so a frame with no possession value would come out as away, and it skips a frame with no ball
  entry. In all seven matches every ball frame has a `BallPossession` of 1 or 2 and a `BallStatus`, and no frame
  number in a half is without a ball frame (0 in each count). Our own parser (`pipeline/benchmark/dfl.py`) counts the
  first two and refuses any other code, so a file that changes this stops the parse.

Its own download does not work: kloppy 3.18's DFL loader fetches from `springernature.figshare.com`, which sits behind a
bot challenge, so `load_open_tracking_data` cannot get the files. P01 downloads them with
`pipeline/benchmark/sources.py` and parses them with `pipeline/benchmark/dfl.py`. kloppy still reads local files
(`kloppy.sportec.load_tracking(meta_data=..., raw_data=...)`), which is how the cross-check runs.

If you do use kloppy on these files, index frames by `frame_id` rather than by position, and read `BallPossession`
yourself for the away team.

## Reproducibility and environment

**T20 — Use the team venv.** It runs Python 3.11.9 in `.venv`, with numpy 2.4.6, pandas 3.0.6 and kloppy 3.18.0.
System Pythons differ — a machine's default may be 3.12 with numpy 2.5 — and different library versions can change
floats, sort orders, and therefore the file. Pin the final versions in the submission repo.

**T21 — Line endings change the file.** `open(path, "w")` writes `\r\n` on Windows and `\n` elsewhere, so the same
code produces different bytes on different machines — and "the same file" may be checked byte for byte. Write with
`newline="\n"` (in pandas, `lineterminator="\n"`). Git's `core.autocrlf` also rewrites CSVs on checkout, so compare
hashes of freshly regenerated files, not of the working tree.

**T22 — Nondeterminism.** Seed everything. Set deterministic flags for GBDT and PyTorch (thread counts, GPU kernels),
or run final inference on CPU. Sort directory listings.

**T23 — Video decoding.** A decode must yield exactly 199,319 frames; don't pad or trim (FAQ). OpenCV's
`CAP_PROP_POS_FRAMES` seeking is not frame-accurate, so decode sequentially or verify the indices.

**T24 — The scorer reads every CSV in `evaluation/results/`.**
- Stale files clutter the board, so use `--results <dir>`.
- Don't name our files `baseline_*.csv`: the gitignore exception for that pattern would commit them.
- `python starter/baseline.py` without `--out` overwrites the committed baselines.

## Rules and data handling

**T25 — Data must never leave our machines the wrong way.**
- Never give match data to any LLM, Claude included.
- Nothing from `data/` goes into git, and neither does derived frame-level data. Keep caches and labels under
  `data/`, which is ignored. A new folder outside `data/` is *not* ignored, so check `git status` before every commit.
- Don't move the data to cloud GPUs or notebooks without checking the T&C, and never with AI assistant features
  switched on.

**T26 — Nothing about this fixture from outside the bundle.** Event feeds, possession stats, match reports and goal
times all count as "any other source of corrected possession data for this fixture", or close enough. Inferring the
private GT's statistics from the published baseline scores is off-limits in spirit too: tune on our own validation.

**T27 — No hand labels on this match, not even to validate.** The organisers changed the rule on 22 Oct 2026
(upstream `c073f29`, RULES.md). Hand-labelled data from other matches is fine for training or validation, in any
amount. Hand labels on BAR-ATH are not, whatever they are for.

Not allowed on this match:
- labelling possession or stoppages from its video, including "blind" calibration on the public window;
- drawing ball boxes on its frames to fine-tune a detector (P14);
- hand-correcting any output.

Still fine:
- the published public GT;
- watching the video and measuring around its published transitions (P02);
- pseudo-labels our own code produces (P07);
- labelled data of other matches (P01, P13).

A model trained on the public GT must predict the public window out of fold, so that no ground-truth label is copied
into the file. Disclose every labelled dataset in the technical document.
