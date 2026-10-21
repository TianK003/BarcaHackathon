# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Our team's fork of the organisers' starter repo for **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports × FC Barcelona × Mobile World Capital Barcelona). The `upstream` remote is
`metrica-sports/more-than-a-hack-2027`; organiser fixes to docs, scorer or alignment land there.

- **Team:** 5 members (rules allow 3–5).
- **Deadline: November 17, 2026.** Results before November 26, 2026.
- **This stage is tested automatically.** Organisers run this repo's own `evaluation/run_evaluation.py` against a
  private full-match ground truth. No jury, no presentation, no live leaderboard, no feedback before results.

**Task:** for every frame of the *tactical* video of FC Barcelona (home) vs Athletic Club (away), LaLiga 2024-25 MD2,
output `H` (home has the ball, alive), `A` (away, alive), `D` (dead ball) or `X` (abstain). One `frame,label` CSV,
exactly 199,319 rows, `frame` = 0…199318 in order, no blank cells. Ranked on **`spellF1`**, ties broken by `macroF1`.
Floor to beat: `baseline_smooth50`, full-match **spellF1 0.2433**.

## Hard rules

- **Match data must never reach an LLM — including you.** The T&C forbid making any video, dataset or extract of
  them available to a third-party/cloud LLM. Do not Read, `cat`, `head` or `grep` files under `data/`; do not print
  raw tracking/tagging/homography rows (`df.head()`, record dumps); do not view video frames or executed notebook
  outputs that may contain them. Write code that runs locally and inspect only aggregates — scores, counts, shapes,
  pass/fail. `evaluation/gt/` and `evaluation/results/baseline_*.csv` are public (MIT) and fine to read.
- **Never commit match data** or derived data that could reconstruct it (frame-level tracking exports, clipped
  video); the submission repo must be public. `.gitignore` covers `data/*`, `*.mp4` and `evaluation/results/*.csv`
  (except `baseline_*.csv`) — keep it that way.
- **Submitted labels must come from an algorithm.** Hand-labelling or hand-correcting the output disqualifies;
  hand-labelled data used only to train or validate a model is fine.
- **No private ground truth**, or anything derived from it, beyond the public 5 minutes.
- **The repo must regenerate the submitted CSV exactly** — the same file, not approximately. Seed all randomness,
  pin dependencies, keep one command that goes from `data/` to the CSV.
- **Don't modify `evaluation/`.** It is the official scorer; local edits only make our numbers drift from the
  official ones. Where the docs and `possession_metrics.py` disagree, the code is right.

## Commands

No test suite, linter or CI. The checks are the validator, the scorer, and the alignment self-checks in the loaders.

```bash
pip install -r requirements.txt            # numpy + pandas: all the scorer and baseline need
pip install -r requirements-starter.txt    # + kloppy==3.18.0, matplotlib, notebook

python evaluation/validate_submission.py path/to.csv            # official scorer's checks; run on the exact upload
python evaluation/run_evaluation.py                             # score every *.csv in evaluation/results/ vs public GT
python evaluation/run_evaluation.py --bootstrap 1000            # + intervals; do this before believing any gap
python evaluation/run_evaluation.py --results DIR --no-detail   # score another folder (e.g. one CSV), table only

python starter/make_dummy_submission.py             # constant D/H/X files, no data needed
python starter/baseline.py                          # rebuild the 4 baselines from the raw ATD (needs data/, 1–2 min)
python starter/baseline.py --window 40 60 --out DIR # other smoothing half-widths
python starter/load_smart_tagging.py                # stdlib only; re-checks the tagging clock against both kickoffs
python starter/load_homography.py                   # validity summary + row/frame alignment check, both sources
python starter/load_atd.py                          # full kloppy load (minutes, GBs of RAM);
                                                    # load_atd.check_alignment_raw() is the fast check
```

The scorer writes `evaluation/scoreboard.csv` and `evaluation/reports/<name>.json` + `<name>_transitions.csv`
(gitignored) and exits 1 if any submission was skipped as invalid. `--gt` takes `public` (default), `private`
(organisers only) or a CSV path.

Python 3.8+; scorer output is identical across versions (it also reproduces the README board on 3.12 / numpy 2.5 /
pandas 3.0). Use 3.9+ for the notebooks on Windows — Jupyter on 3.8 hits a `pywinpty` build failure.

## Architecture

There is no package: scripts put the repo root (and `evaluation/`) on `sys.path` and import `games` /
`possession_metrics` as top-level modules, so each runs by path from any working directory.

- **`games.py`** (stdlib only) — single source of truth for match constants (199,319 frames, 25 fps, pitch
  105.3 × 68.0 m, half boundaries) and every data path. `asset_paths(source="tactical"|"broadcast")` builds file
  names, `require([...])` returns paths or one readable error, `MTAH_DATA_DIR` relocates the bundle. Never hard-code
  a data filename elsewhere.
- **`evaluation/possession_metrics.py`** — the metric spec, with the reasoning in its docstrings;
  `run_evaluation.py` is the official scorer on top of it. The per-submission `_transitions.csv` lists every
  possession change as matched / missed / spurious with its video frame — the main debugging artefact.
- **`starter/`** — loaders, `baseline.py` (nearest tracked player to the ball, then a centred majority vote; numpy
  only, streams the raw ATD rather than loading it through kloppy) and three guided notebooks
  (`03_first_submission.ipynb` is the end-to-end walk-through).

### Frame alignment — an off-by-one shifts every transition

Submission row *i* = frame *i* of the **tactical** video.

| source | index | → submission row |
|---|---|---|
| ATD (tactical) | `frame_id` 1…199319 | `frame_id - 1` |
| Smart Tagging | seconds from the first tactical frame | `round(seconds * 25)` |
| Homography | `frame` 0…199318 | as is |
| Anything broadcast | its own 164,488-frame timeline (cuts, replays) | **no mapping provided** |

The per-half `offsets` in `games.py` (`video_to_id`, `id_to_video`) map to the internal IdFrame counter the ground
truth was derived from — not needed for ATD alignment. A ~164k-row output means the broadcast was indexed by mistake.

Graded frames: 1st half 11,185–90,651, 2nd half 110,808–190,197 (`games.half_video_blocks("BAR-ATH")`); everything
else is ungraded, so `X` there is free. The 2nd half is graded from 110,808 but kicks off at 113,742 — those 2,934
frames are all `D`. The public GT labels only 11,185–18,684 (5 minutes, 17 possessions); the rest of it is `X`.

### How scoring should shape the design

- Temporal metrics run inside each graded half, never across halftime. A **possession** is a maximal run of `H` or
  of `A`; `D` runs are stoppages, so `H H D D H H` is two possessions.
- `spellF1` matches predicted possessions one-to-one to true ones (same team, greedy by temporal IoU) and averages F1
  over IoU 0.3/0.4/0.5/0.6/0.7. Flicker is fatal: the raw baseline reports 131 possessions in the public window
  against 17 real ones.
- An `X` on a graded frame is a miss for `macroF1`, but for `spellF1` it is forward-filled with our last non-`X`
  label in that half (back-filled at the half's start). Abstaining inside a possession is free on the ranking
  metric; abstaining across a hand-over drags the old label over it — at transitions, guess.
- Noise: full-match `spellF1` gaps below ~0.04 (`macroF1`: ~0.01) are not real. On the public window one missed
  possession moves `spellF1` ~0.06, and it ranks `smooth125` above `smooth50` — the reverse of the full match. Use it
  to check correctness, never to pick hyperparameters.

### Data facts that bite

- The ball is missing (`NaN`) on 41% of graded frames, disproportionately at contested moments.
- ATD carries no identities: 46 tactical track slots ("Track 1…46", `ShirtNumber` = slot index) for 22 players;
  slots appear, vanish, swap and occasionally duplicate. A slot's team comes from the metadata XML, whose first
  `<Team>` is home/Barcelona (`baseline.team_of_each_track`).
- Raw ATD line: `<frame_id>:<x,y,speed>;<x,y,speed>;…:<ball x,y,speed>`. Coordinates are normalised 0–1 — scale by
  the pitch before computing distances (it is not square).
- Smart Tagging: 638 *automatic* phase intervals, not ground truth, and phases are not possessions. Codes are upper
  case (`SET PIECES`), tag values title case (`Kick Off (Start)`), kloppy tag columns title case (`Team`) — match
  case-insensitively. `Team` is `Barcelona`, `Atletic Club` (sic) or `N/A` (36 instances, including every
  throw-in). Tags mark a window of a few seconds around an event (kickoff tags open ~3.5 s early), not the instant.
- Homography maps normalised pitch (0–1) → pixels, so ATD coordinates go in unconverted. Check `valid` first
  (tactical 100%, broadcast 65.8%); `reprojection_error` is lower-is-better.
- The ground truth is derived mechanically from corrected tracking: `D` where it marks the ball dead, otherwise the
  team it names. Dead-ball ownership is never asked for.

## Data setup

Gated behind enrolment and T&C. `games.py` expects:

```
data/tactical/BAR-ATH_tactical{.mp4,_atd.xml,_atd.txt,_homography.csv,_smart_tagging.xml}
data/broadcast/BAR-ATH_broadcast{.mp4,_atd.xml,_atd.txt,_homography.csv}
```

The Google Drive download arrives as several zips whose paths start with
`More Than A Hack 2027 - Initial Challenge - Dataset/` (strip that prefix, and skip its `README.md` — `data/README.md`
is tracked) plus the two mp4s served unzipped with `-00N` suffixes (rename them to the names above). The scorer needs
none of this; it reads only CSVs.

## Deliverables

One Google Form (URL emailed to registrants, Google account needed); one response per team, editable, the last one
before the deadline counts.

1. **`<team_name>.csv`** — validate the exact file being uploaded.
2. **Public repo URL** — source, pinned dependencies, install/run instructions, and the command that regenerates the
   CSV. The `requirements*.txt` here are mostly unpinned; pin ours.
3. **`<team_name>.pdf` or `.md` technical doc**, a few pages: approach and why; assets used and dropped; handling of
   missing/noisy tracking; possession rules implemented (loose balls, challenges, restarts); limitations;
   reproduction. `*.pdf` is gitignored.
