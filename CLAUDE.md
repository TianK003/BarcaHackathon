# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Our team's fork of the organisers' starter repo for **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports × FC Barcelona × Mobile World Capital Barcelona). The `upstream` remote is
`metrica-sports/more-than-a-hack-2027`; organiser fixes to docs, scorer or alignment land there.

- **Team:** **Un Dia de Partit**, 5 members (rules allow 3–5). Questions to the organisers go by email to
  `hackathon.talentarena@mobileworldcapital.com`, not as GitHub issues (since 3 Nov; issues and pull requests are switched off on
  their repo by 9 Nov; `docs/rules.md`). They answered our questions 1–4 (5 Nov) with the clarification alone, so plan as if
  any question gets it back (`docs/traps.md` T56). The upload files are named after it:
  `Un Dia de Partit.csv` and `Un Dia de Partit.pdf` (or `.md`); the scorer names the board's row after the file.
- **Deadline: November 17, 2026.** Results before November 26, 2026.
- **This stage is tested automatically.** Organisers run this repo's own `evaluation/run_evaluation.py` against a
  private full-match ground truth. No jury, no presentation, no live leaderboard, no feedback before results.

**Task:** for every frame of the *tactical* video of FC Barcelona (home) vs Athletic Club (away), LaLiga 2024-25 MD2,
output `H` (home has the ball, alive), `A` (away, alive), `D` (dead ball) or `X` (abstain). One `frame,label` CSV,
exactly 199,319 rows, `frame` = 0…199318 in order, no blank cells. Ranked on **`spellF1`**, ties broken by `macroF1`.
Floor to beat: `baseline_smooth50`, full-match **spellF1 0.2433**.

## Hard rules

**The rules are the organisers' emailed "Clarification of the rules" (recorded 3 Nov 2026), in `docs/rules.md`
verbatim with our reading.** It takes precedence over RULES.md, FAQ.md, the clarifications of 22 and 23 Oct (upstream
`c073f29`, `f8f2629`) and anything written before it; where it says nothing, those still hold. Their reply to our
questions (5 Nov 2026) added: **"anything that is not explicitly allowed is not permitted"**. So if no text allows
something by name, treat it as forbidden from the start. Read `docs/rules.md` before building anything that learns,
tunes or uses the video.

- **Match data must never reach an LLM — including you.** The T&C forbid making any video, dataset or extract of
  them available to a third-party/cloud LLM, and the clarification forbids "uploading the provided data to AI
  services". Do not Read, `cat`, `head` or `grep` files under `data/`; do not print raw tracking/tagging/homography
  rows (`df.head()`, record dumps); do not view video frames or executed notebook outputs that may contain them. Write
  code that runs locally and inspect only aggregates — scores, counts, shapes, pass/fail. The public ground truth
  (`evaluation/gt/`) and `evaluation/results/baseline_*.csv` are labels of this match's frames, so since 3 Nov we treat
  them as provided data too: read their scores and counts, not their rows. The public datasets of *other* matches
  under `data/external/` (the DFL/IDSSE open data, CC BY 4.0; P01, P13) are fine to read. They are not competition
  data, and their own licences govern them. Keep everything else in `data/` closed.
- **Never commit match data** or derived data that could reconstruct it (frame-level tracking exports, clipped
  video); the submission repo must be public. `.gitignore` covers `data/*`, `*.mp4` and `evaluation/results/*.csv`
  (except `baseline_*.csv`) — keep it that way. "Providing files of intermediate steps" is allowed. A small intermediate output
  (event times and scores, no tracking, video or audio) may be committed with the script that regenerates it (issue
  #14); anything frame-level goes to the organisers directly, never into the repo (`docs/rules.md`).
- **Submitted labels must come from an algorithm.** Hand-labelling or hand-correcting the output disqualifies.
- **Train and tune on external data only.** Nothing is trained, retrained, fitted, calibrated, adapted or tuned on
  this match: not on its ground truth or video (named in the clarification), and, by our reading of "you MUST use only
  external data", not on its tracking, tagging or homography either, with labels or without. That rules out
  self-supervised or unsupervised learning on it, pseudo-labels our own code made from it, priors re-estimated on it,
  and any constant measured on it. A fixed rule that reads the match at run time is the algorithm running, not fitting;
  a number measured on the match and written into a config or model is fitting (`docs/rules.md`, "How we read it",
  `docs/traps.md` T54). *Changed 3 Nov 2026:* until then label-free learning on this match was allowed. R029 took
  `smart_tagging` out of `main.toml` for this, because its c_team 0.76 had been measured on this match (R017).
  **Not permitted by our reading since 5 Nov:** the benchmark's degradation is fitted on our tracking's label-free
  statistics (`gaps` transplants our missing-ball masks), so both models and every constant chosen on it stand on data
  that is not only external. The organisers answered question 1 with the clarification alone. Redo on a degradation that
  measures nothing of our match (`docs/proposals.md` P27), or keep and disclose: **decided 9 Nov (Tian): redo, P27**
  (`docs/rules.md`).
- **No hand labels on this match at all**, in any amount and for any purpose, and no "human-produced data or any other
  data created manually on the provided game": not to train, validate or calibrate, and no correcting its tracking by
  hand (corrections must come out of code). Hand-labelled data from *other* games is fine, in any amount, if we have
  the rights to use it (licence checked), the organisers can access it and the technical document says what was
  labelled, how, by whom and how much (so validation comes from other games, `docs/proposals.md` P01).
- **Nobody checks our output against the video.** A person may watch the video and inspect the data only "to better
  understand what counts as possession" (Claude still sees aggregates only, the first rule above). Not fine: comparing
  our labels, transitions, the scorer's `_transitions.csv` or our detectors' events with the video, and turning what
  was seen into labels, corrections, or rules and constants tied to specific frames (`docs/traps.md` T27). *Changed
  3 Nov 2026:* until then watching to debug the pipeline was allowed.
- **Automatic checks on the provided data are allowed, and never choose.** Validation, determinism, alignment, the
  label-free statistics of our own output and a score against the public window may pass or fail a candidate; they
  never rank candidates or set a value (`docs/traps.md` T01).
- **No private ground truth**, or anything derived from it, **and no human-produced data about this match from any
  source** (event data, corrected tracking, possession data, annotations; RULES.md rule 2, `f8f2629`). External data
  must be about *other* games. The public 5 minutes are for **checking that the pipeline is correct**, "not training
  data and not a tuning set": never train, fit, calibrate or choose a constant on them, out of fold or not
  (`docs/traps.md` T01, T27).
- **The repo must regenerate the submitted CSV exactly** — the same file, not approximately. Seed all randomness,
  pin dependencies, keep one command that goes from `data/` to the CSV.
- **Don't modify `evaluation/`.** It is the official scorer; local edits only make our numbers drift from the
  official ones. Where the docs and `possession_metrics.py` disagree, the code is right.

## Team docs (`docs/`)

Read them before starting work and update them when you finish. They are public and Claude reads them, so they hold
aggregates only, never raw data rows.

- `docs/rules.md` — the organisers' clarification of 3 Nov 2026 verbatim (it overrides RULES.md), what it changed, our
  reading, the audit of `main.toml` against it, and the questions for the organisers. Record their answers there.
- `docs/proposals.md` — prioritised ideas `P01…` with effort, impact, owner and status, plus a suggested plan to the
  deadline. Update the status when you start or finish one.
- `docs/results.md` — every experiment (`R…`) and analysis (`A…`): commit, command, public-window and validation
  scores, verdict. Log failures too, and every upload in its submission log. A01–A03 hold measurements that should
  shape any design; A05–A09 and R002–R005 are the benchmark's (P01, done; the default is R004c's); A13 says where
  `main.toml`'s remaining spellF1 is (who has the ball, not the dead ball); A21 audits `main.toml` against the 3 Nov
  rules.
- `docs/traps.md` — pitfalls `T01…`. Add one whenever something costs real time.
- `docs/technical_document.md` — the draft of deliverable 3. Update it when a pipeline decision lands.

## Commands

No CI or linter. The checks are `pipeline/selftest.py`, `pipeline/benchmark/selftest.py` and `fidelity.py`, the
validator, the scorer, and the alignment self-checks in the loaders.

```bash
py -3.11 -m venv .venv                                            # macOS/Linux: python3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-starter.txt   # numpy, pandas + kloppy==3.18.0, matplotlib, notebook
.venv/Scripts/python -m pip install -r requirements-pipeline.txt  # exact pins of what pipeline/ imports
.venv/Scripts/python -m pip install -r requirements-benchmark.txt # the same + lxml, scikit-learn: pipeline/benchmark/

python pipeline/run.py pipeline/configs/main.toml       # data/ -> evaluation/results/main.csv, validated + manifest
python pipeline/run.py CONFIG --score [GT]              # + scores vs public GT (or a labels CSV), per-source agreement
python pipeline/run.py CONFIG --out "../Un Dia de Partit.csv"   # CSVs only to evaluation/results/, data/, or outside the repo
python pipeline/selftest.py                             # contract + equivalence checks on synthetic data (seconds)
python pipeline/selftest.py --data                      # + the 4 committed baselines rebuilt byte for byte

python pipeline/benchmark/sources.py [--verify] [--data-dir D]   # P01: fetch (or only check) the 7 DFL matches, md5
python pipeline/benchmark/run.py CONFIG [CONFIG ...]    # configs on the benchmark; the first is the reference
python pipeline/benchmark/run.py CONFIG ... --truth coarse   # the same under the coarse truth; read both together
python pipeline/benchmark/run.py CONFIG ... --truth both     # both truths from one decode: the two runs' reports
python pipeline/benchmark/run.py CONFIG ... --drop-match-only   # drops (and reports) match-only sources (Smart Tagging:
                                                        # r017_*, r024_both), which DFL cannot serve; main.toml has none since R029
python pipeline/benchmark/choose.py --reference main STEM ...   # a grid's declared rule on its reports, + LOMO
python pipeline/benchmark/choose.py --reference main --objective mean STEM ...   # R024 on: the mean of the two truths
python pipeline/benchmark/diagnose.py CONFIG --drop-match-only   # A19: the IoU ladder decomposed, sources' and posterior's calibration
python pipeline/benchmark/fidelity.py --profile tracking --seeds 3   # gates + WARNs: the copy against our match
python pipeline/benchmark/train_shape.py [--verify|--report]   # P07: train the shape model as JSON; --verify, --report
python pipeline/benchmark/conventions.py public|dfl|compare|latency|shift ...   # P02: timing, inspected; DFL-lag shift
python pipeline/benchmark/metrica.py [--offline]        # P16: Metrica's sample games' grain against the two truths (R023);
                                                        # not to be run since 5 Nov: the data has no licence (docs/rules.md)
python pipeline/benchmark/selftest.py [--dfl] [--data]  # benchmark checks (seconds); --dfl: the 7 matches, --data: ours

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

The benchmark's `run.py` takes `--profile clean|gaps|tracking|context` (default `tracking`, version 3 as of R004c;
`context` is a sensitivity check only, it fails two of its gates, R005) and `--truth raw|coarse|both`; `fidelity.py`
takes `--profile gaps|tracking|context` (a profile that degrades nothing has no gates). Both take `--seeds N` (default
3), `--matches ID ...`, `--data-dir D` and `--no-write`; `sources.py` takes `--data-dir D` too. Reports go to
`data/runs/benchmark/`, caches to `data/cache/`, and `fidelity.py` exits 1 if a gate fails.

Run everything with the venv's interpreter — `.venv/Scripts/python` on Windows, `.venv/bin/python` on macOS/Linux —
because activation doesn't persist between tool calls. The team standard is Python 3.11.9 (numpy 2.4.6, pandas 3.0.6,
kloppy 3.18.0); `requirements-pipeline.txt` pins what the pipeline imports, and the pipeline needs ≥ 3.11
(`tomllib`). The scorer itself gives identical output on 3.8+, but different library versions can change our own
output (`docs/traps.md` T20–T21).

## Architecture

Apart from `pipeline/`, there is no package: scripts put the repo root (and `evaluation/`) on `sys.path` and import
`games` / `possession_metrics` as top-level modules, so each runs by path from any working directory. `pipeline/`
keeps that convention: its scripts swap their own folder on `sys.path` for the repo root, and importing it adds
`evaluation/`.

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
- **`pipeline/`** — our submission pipeline; `pipeline/README.md` is the overview (mermaid diagram, contracts,
  configs). Five stages, one folder each with its own README: `inputs` (alignment; the ATD parsed once into
  `data/cache/atd_tactical/`) → `evidence` (per-frame `alive`/`home` probabilities, NaN = no opinion) → `decode` (one
  block per graded half) → `refine` → `output` (LF CSV, official validation, manifest in `data/runs/`). Components are
  plain functions in each stage's `REGISTRY` (`evidence` holds `nearest_player`, `ball_presence`, `controlled_touch`,
  `dead_ball_rules`, `smart_tagging`, `dead_ball_model`, `shape_possession` and `restart_team`, the last three reading
  the trained JSON models in `pipeline/models/`; `smart_tagging` is Smart Tagging's team votes, in `main.toml` from
  R017 (P08) until R029 took it out under the 3 Nov rules (its `c_team` was measured on this match, `docs/rules.md`),
  and declares `MATCH_ONLY`, so the benchmark runs a config that has it with `--drop-match-only`; `restart_team` is
  a restart-team vote, measured but not in `main.toml`, P17; `decode`
  holds `majority_vote` and `hmm` (which can decode restart states, `restart_source`, off by default);
  `refine` holds `shift_edges`), chosen by TOML configs in `pipeline/configs/` and checked against their signatures
  before any data loads. `configs/smooth50_repro.toml` must rebuild `baseline_smooth50.csv` byte for byte — run
  `selftest.py --data` after any change upstream of the decoder. Contracts live in `pipeline/contracts.py`; change
  them only with the team.
- **`pipeline/benchmark/`** — P01's validation: the seven open DFL/IDSSE matches (Bassek et al. 2025, CC BY 4.0), the
  truth built as ours is (`D` where `BallStatus` is dead, else the team in `BallPossession`) and the tracking degraded
  to look like ours, run through the pipeline's own components and scored by the official scorer per half.
  `pipeline/benchmark/README.md` is its overview. Modules: `sources` (pinned download, size + md5, into
  `data/external/dfl/`), `dfl` (our own parser, cached in `data/cache/dfl/`; not kloppy's loader or download, T30),
  `truth`, `inputs` (a match served as `Inputs`, no truth reachable), `calibration` + `hazard` (our own tracking
  measured once, cached in `data/cache/benchmark/calibration/`), `degrade` (profiles `clean`, `gaps`, `tracking` v3,
  the default as of R004c, and `context`, a sensitivity check only, R005), `fidelity` (a profile's gates and WARNs
  against our match), `score` + `run` (pooled spellF1, 90% t-intervals, sign-flip test, reports in
  `data/runs/benchmark/`), `choose` (a grid's declared rule and its leave-one-match-out estimate, from the reports),
  `restarts` (the stoppages' anatomy, the restart vote's stats, its configs), `dfl_events` (the DFL restart events,
  diagnostics), `headroom` (a config beside variants whose evidence is the truth: what a perfect source would add;
  its `plus_` variants measure `main.toml` itself, A13), `tag_stats` (Smart Tagging's votes against our own
  pseudo-labels on BAR-ATH, R017; since 3 Nov a check at most, never a source of a constant, `docs/rules.md`), `label_diff` (how a config changes BAR-ATH's labels, aggregates only), `wrong_team` (P21 step 0, dropped in A16: the ratio of over-11 slot frames at the ball to all graded frames, kept for P19's transfer check), `metrica` (P16, R023: H/A/D derived from Metrica's three public sample games' events, fetched at run time and never committed, and their grain against A06's two truths; not to be run again since 5 Nov: the data has no licence, `docs/rules.md` question 3), `diagnose` (A19, P23: the IoU ladder decomposed into clean, merge and split pairs, and the calibration of the sources and of the hmm's posterior) and `checks/`
  (what `selftest.py` runs). Degradations are fitted once per version on
  label-free statistics of our tracking, then frozen, never tuned towards a score (T26, T37); since the organisers'
  reply of 5 Nov that fit is, by our reading, not permitted (question 1 in `docs/rules.md`; the redo is P27). The Calibration and every
  degraded `Atd` hold BAR-ATH's masks and counts: print none of it, and write no CSV of BAR-ATH labels.

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
  to check correctness, never to pick or tune anything (a rule since `f8f2629`, not just advice); pick hyperparameters
  on P01's benchmark of other matches. This GT's *timing* conventions (P02) may be inspected on the public window to
  understand what counts as possession, but no constant is set from it, and our own output is never checked against
  the video. Other providers flip their flags at different moments, so DFL cannot
  fix our convention either: a constant for it must come from other games of our provider (P16), never from the window
  (the pipeline's own lag on DFL, R012's s_b, is a different thing).

### Data facts that bite

- The ball is missing (`NaN`) on 41% of graded frames, in long gaps: median 10.7 s, and gaps under 2 s hold only
  0.1% of missing frames. Coverage worsens through each half, so the public window is the best-covered block of the
  match. With the ball visible, the nearest player's team is still wrong 22% of the time, and the ball is visible on
  66% of dead-ball frames (measured: `docs/results.md` A01–A03).
- ATD carries no identities: 46 tactical track slots ("Track 1…46", `ShirtNumber` = slot index) for 22 players;
  slots appear, vanish, swap and occasionally duplicate. A slot's team comes from the metadata XML, whose first
  `<Team>` is home/Barcelona (`baseline.team_of_each_track`).
- Raw ATD line: `<frame_id>:<x,y,speed>;<x,y,speed>;…:<ball x,y,speed>`. Coordinates are normalised 0–1 — scale by
  the pitch before computing distances (it is not square).
- The ball-speed channel is NaN on 92% of the graded frames where the ball has a position, and where it is given it
  behaves like a player's speed (median 1.3 m/s against 4.8 m/s from the ball's own positions): derive the speed from
  successive positions, never read `atd.ball_speed` (T31, A07).
- A slot's team is the slot's, not the player's: after an identity switch a team shows more than 11 slots while the
  other shows fewer, on 10.8% of graded frames (a lower bound), so a feature that trusts one slot's team is wrong there
  (T32, A07).
- Smart Tagging: 638 *automatic* phase intervals, not ground truth, and phases are not possessions. Codes are upper
  case (`SET PIECES`), tag values title case (`Kick Off (Start)`), kloppy tag columns title case (`Team`) — match
  case-insensitively. `Team` is `Barcelona`, `Atletic Club` (sic) or `N/A` (36 instances, including every
  throw-in). Every `SET PIECES` tag is a fixed 875-frame (35 s) clip and every kick-off tag 125 frames (5 s), opening before the
  restart (kick-offs ~3.5 s early): a tag's start is an offset from the event and its end means nothing (T55).
- Homography maps normalised pitch (0–1) → pixels, so ATD coordinates go in unconverted. Check `valid` first
  (tactical 100%, broadcast 65.8%); `reprojection_error` is lower-is-better.
- The ground truth is derived mechanically from corrected tracking: `D` where it marks the ball dead, otherwise the
  team it names. Dead-ball ownership is never asked for.
- DFL's flag is fine-grained (437 possessions a match, median 3.7 s) and how fine ours is is not known: rank anything
  that depends on possession length under the raw and the coarse truth (`--truth coarse`), and treat a disagreement
  as a warning (T33, A06). R023 (P16, Metrica's own sample games) found **both**: a choice declared since then
  ranks on the unweighted mean of the two truths' pooled spellF1s, each reported; `choose.py --objective mean` applies it (R024, which re-chose
  R013 under it: the touch hold moved to 0.55). Since 5 Nov R023's data is withdrawn (Metrica's sample data has no
  licence), so the grain is unknown again; the objective stays (Tian, 9 Nov) and P27 re-chooses the hold under it (`docs/rules.md`).
- Two benchmark matches have a team down to 10, so a gate on how many slots are present is measured on full-strength
  frames only (T36).
- DFL's own jitter is rare jumps on a 1 cm grid (RMS of the second difference 9.0 cm against our 3.3 mm): compare
  medians, not an RMS. The profile's 5-frame average leaves 1.80 cm per slot, 5.5 times ours, and no jump filter is
  needed; the 6.5 to 8.5 cm that R004 and R004b showed came from the profile's own slot handovers, fixed in v3 (T35).

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

1. **`Un Dia de Partit.csv`** — validate the exact file being uploaded, and log it in the submission log in
   `docs/results.md`.
2. **Public repo URL** — source, pinned dependencies, install/run instructions, and the command that regenerates the
   CSV. The `requirements*.txt` here are mostly unpinned; pin ours.
3. **`Un Dia de Partit.pdf` or `.md` technical doc**, a few pages: approach and why; assets used and dropped; handling of
   missing/noisy tracking; possession rules implemented (loose balls, challenges, restarts); limitations;
   reproduction. Draft it in `docs/technical_document.md`. `*.pdf` is gitignored.
