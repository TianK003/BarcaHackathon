# R028 the broadcast's ball in our gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the code R028 needs, and the controller's runbook from the fidelity run to ADOPT or DROP.
- On BAR-ATH: fill our ball's gaps with the broadcast's usable ball, behind one config switch.
- On DFL: a simulated broadcast transplanted from ours, so that the benchmark can judge the switch.

**Architecture:**
- **The pipeline side:**
  - `pipeline/inputs/broadcast.py` holds the alignment, the live test, the quality filter and the fill, as pure functions
    on arrays;
  - `Inputs` loads the broadcast lazily and offers `with_ball_fill(kind)`, which returns a view whose `atd` is filled;
  - a `[inputs] ball_fill` key in the config selects it, and `pipeline/run.py` and the benchmark's `predict()` apply it
    in one place (`prepare_inputs`).
- **The benchmark side:** `pipeline/benchmark/broadcast_sim.py` holds:
  - the donor pools (built once on BAR-ATH);
  - the transplant (`simulate`), which `MatchInputs` serves as `broadcast_ball`;
  - the fidelity measure, the wiring into `run.py` and `conventions.py`, and the `bar-ath` checks.
- **Nothing changes `main.toml`'s output** until the controller adopts, by the design's section 8.

**Tech Stack:** Python 3.11.9 and numpy 2.4.6 in `pipeline/`. `pipeline/benchmark/` may also use the pins of
`requirements-benchmark.txt`. Checks run under `pipeline/selftest.py` and `pipeline/benchmark/selftest.py`.

**Spec:** `docs/superpowers/specs/2026-11-01-r028-broadcast-ball-design.md`, plus R028 part 1 in `docs/results.md` once
it is written. Where this plan and they differ, the spec and the declaration win.

## Global Constraints

- **Match data never reaches an LLM.**
  - Never Read, `cat`, `head` or `grep` a file under `data/`.
  - Print and write aggregates only, never a frame-level array or row.
  - The pools and the simulated balls stay under `data/cache/`. JSON reports hold per-match numbers and the pools'
    digest, nothing longer.
- **Don't modify `evaluation/`.** Import `possession_metrics` as `pm`, as `pipeline/benchmark/score.py` does.
- **`main.toml` keeps its output.** Until the controller's adoption commit, `python pipeline/run.py
  pipeline/configs/main.toml` must still write sha256
  `e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170`, and every config without `[inputs]` must write the
  same bytes as before.
- **Files not edited:** `controlled_touch.py`, `dead_ball_model.py`, `shape_possession.py`, `smart_tagging.py`, the
  decoders and the refiners. Task 5b (R028b, conditional) is the one declared exception.
- **Dependencies:** numpy and the standard library in `pipeline/`. Nothing under `pipeline/evidence`, `decode` or
  `refine` imports `pipeline.benchmark` (the import scan in `checks/inputs.py`).
- **No truth through the inputs.**
  - `simulate()` takes arrays: the served ball's presence, DFL's clean ball, the driver's codes, the graded blocks, the
    pitch, the pools and a generator. It takes no match object.
  - The driver's codes come from `predict()` on the served inputs.
  - `test_match_inputs_hold_no_truth` must pass with a `broadcast_ball` given.
- **Determinism.**
  - `pipeline/inputs/broadcast.py` uses no randomness.
  - The simulation uses only `numpy.random.default_rng([seed, match_index, 28])`, with `match_index` as `degrade.py`
    numbers the match.
  - The same inputs give the same bytes whatever the thread count.
- **The constants are the design's,** fixed before any run:
  - the alignment: a coarse step of 25 frames, 3 m, at least 300 pairs; fine ±40 frames, 1.5 m; accepted at a fine score
    ≥ 0.5 and a coarse peak ≥ 2 × the best score more than 250 frames away; ties to the smaller |d|;
  - the live test: valid homography, ≥ 4 broadcast players, a median nearest-player distance < 2.0 m;
  - the quality filter: no usable run shorter than 13 frames; the fill on graded frames only;
  - the simulation: 5 donors, class alive at ≥ 0.5 `H`/`A`, near-gap errors within 50 frames of a gap edge, edges of 25
    frames, long gaps ≥ 50 frames, salt 28;
  - the fidelity tolerances of the design's 3.5;
  - R028b's `filled_confidence` 0.79.
- **Style:** match the surrounding code.
  - Module docstrings explain the why, and constants go in `# Params` blocks.
  - A ValueError message names the parameter and the fix.
  - No array in a repr (the `Atd` rule).
- **No AI attribution** in commit messages (no Co-Authored-By, no "Generated with", no session trailer).
- **Running things:** tests run with `.venv/Scripts/python`.
  - Tasks never start any of these on real data: `pipeline/run.py`, `selftest.py --data`, `broadcast_sim.py pools`,
    `fidelity`, `bar-ath`, `run.py` or `conventions.py` of the benchmark. The controller runs those, one heavy process at
    a time.

## Review Focus

- **Every config without `[inputs]` runs bit-identically.**
  - `with_ball_fill("none")` returns the inputs object itself.
  - `main.toml`'s sha256 is unchanged.
  - The benchmark's existing report fields are unchanged; there is one new field, `broadcast_sim: null`.
- **The alignment never answers silently when it should refuse.** `AlignmentError` is raised for:
  - a broadcast that agrees nowhere;
  - two peaks of nearly the same height;
  - a half with fewer than 300 pairs.
- **The fill writes only where it may.**
  - It writes only on graded frames where the tactical ball is missing, and never over a present ball.
  - It never fills from a frame that is not live, or from a run shorter than 13 frames.
  - `ball_speed` stays NaN there, and the players are untouched.
- **No truth reaches `MatchInputs` or `simulate()`.**
  - `test_match_inputs_hold_no_truth` passes with a `broadcast_ball`, and `simulate`'s signature has no match.
- **Determinism and cache keys.**
  - The same (seed, match index) gives the same simulated ball.
  - A changed `r024_both.toml`, broadcast file or `POOLS_VERSION` gives a new pools digest. Labels whose CSV is not
    `e6496a9b…` stop the pools step.
  - Every report of a run carries the same `broadcast_sim` value, so `main` and the candidate compare, and `choose.py`
    refuses to mix runs with different digests.

## Execution topology (parallel waves, worktrees, commits)

| item | where |
|---|---|
| lane branch `r028-broadcast-ball` | worktree `C:\Users\Tian\Desktop\BarcaHackathon-r028`: Tasks 2b, 3 and 4, the merges, part 2 and the docs |
| inputs branch `r028-inputs` | worktree `C:\Users\Tian\Desktop\BarcaHackathon-r028-inputs`, from the lane head: Task 1 |
| sim branch `r028-sim` | worktree `C:\Users\Tian\Desktop\BarcaHackathon-r028-sim`, from the lane head: Task 2a |
| run worktree | `C:\Users\Tian\Desktop\BarcaHackathon-r028run`, detached at the commit a run uses, with `MTAH_DATA_DIR` the main checkout's `data/` |
| ledger | `.superpowers/sdd/2026-11-01-r028-broadcast-ball/progress.md` in the main checkout (its first line names this plan); logs in `…/logs/` |
| handoff | `.superpowers/sdd/run-2026-10-31/handoff.md` (the hooks' path), kept current after every task and run |

```
Wave 1:  Task 1 (implementer, inputs worktree)  ‖  Task 2a (implementer, sim worktree)  ‖  Task 4a: part 1 drafted (controller)
Wave 2:  merge Tasks 1 and 2a into the lane → Task 2b (implementer, lane) → Task 3 (controller, lane)
Wave 3:  Task 4b: part 1's independent review, then its commit
Wave 4:  the runs (Task 5) → ADOPT or DROP (Task 6) → Task 7
```

**Commits.** Each task commits after its failing tests and again after its implementation. The controller commits:
- the declaration, before any run it governs;
- part 2a after the fidelity run, and part 2b after the bar and the BAR-ATH checks;
- the adoption or the drop, then the docs.

Merges are `--no-ff`. `main` is pushed to the private origin after the final merge.

---

### Task 0: Controller setup (no subagent)

- [ ] Create the ledger `.superpowers/sdd/2026-11-01-r028-broadcast-ball/progress.md`. Its first line is `# SDD ledger
  — plan: docs/superpowers/plans/2026-11-01-r028-broadcast-ball.md`. Record Tian's answers in it (design, section 10):
  - bar (ii);
  - R028b only if the `tracking` Δ is above 0;
  - continue automatically after fidelity;
  - adopt automatically.
- [ ] Copy `.superpowers/bin/heavy-p22b.ps1` to `heavy-r028.ps1`, with `$logs` set to
  `…\.superpowers\sdd\2026-11-01-r028-broadcast-ball\logs`. Keep everything else, including T51's cached handle and
  `EXIT=UNKNOWN`.
- [ ] Create the lane: `git worktree add -b r028-broadcast-ball C:/Users/Tian/Desktop/BarcaHackathon-r028 main`. Then
  `-b r028-inputs …-r028-inputs r028-broadcast-ball` and `-b r028-sim …-r028-sim r028-broadcast-ball`.
- [ ] Create the run worktree: `git worktree add --detach C:/Users/Tian/Desktop/BarcaHackathon-r028run
  r028-broadcast-ball`.

### Task 1: the pipeline side, `pipeline/inputs/broadcast.py` and the switch (worktree `r028-inputs`)

**Files:**
- Create: `pipeline/inputs/broadcast.py`
- Modify:
  - `pipeline/inputs/atd_cache.py` (`load_or_build` gains `folder=FOLDER`);
  - `pipeline/inputs/__init__.py`;
  - `pipeline/config.py`;
  - `pipeline/run.py`;
  - `pipeline/benchmark/run.py` (`predict` only);
  - `pipeline/benchmark/inputs.py` (`MatchInputs.broadcast_ball`);
  - `pipeline/README.md`, `pipeline/inputs/README.md`.
- Test: `pipeline/selftest.py` (synthetic data only).

**Interfaces:**
- Produces:
  - `broadcast.AlignmentError(ValueError)`.
  - `broadcast.Alignment`, a frozen dataclass: `offsets` {half: int}, `fine_scores`, `coarse_scores` and `runner_up`
    {half: float}, `pairs` {half: int}.
  - `broadcast.align(tactical_ball_m, broadcast_ball_m, broadcast_valid, graded) -> Alignment`:
    - `tactical_ball_m` is (n, 2) metres and `broadcast_ball_m` (nb, 2) metres, both NaN where absent;
    - `broadcast_valid` is (nb,) bool, and `graded` the `[(half, start, stop)]` of `Inputs.graded`.
  - `broadcast.frame_map(alignment, graded, n, nb) -> (n,) int64`: the broadcast frame paired with each graded tactical
    frame, −1 elsewhere or out of range.
  - `broadcast.live(tactical_players_m, broadcast_players_m, broadcast_valid, fmap) -> (n,) bool`.
  - `broadcast.usable(broadcast_ball_m, fmap, live_mask, min_run=13) -> (n,) bool`.
  - `broadcast.fill(atd, ball_norm, where) -> (Atd, filled (n,) bool)`. It writes `ball_norm` (normalised) where
    `where` holds and the tactical ball is missing; it never touches a present ball, the players or `ball_speed`.
  - `Inputs` gains:
    - `broadcast_atd` (cached; folder `atd_broadcast`; `games.source_frames("broadcast")` rows);
    - `broadcast_valid`;
    - `broadcast_alignment`;
    - `broadcast_ball`: (n, 2) normalised, NaN where not usable on a graded frame;
    - `ball_filled`: all False on the base;
    - `with_ball_fill(kind)`: the object itself for `"none"`, a `FilledInputs` for `"broadcast"`. A `FilledInputs`
      subclasses `Inputs`, holds every field of the base, and overrides `atd`, `ball_filled` and `fill_summary` (a
      dict of aggregates: offsets, scores, live share, filled frames).
  - `config.Config.ball_fill`: `"none"` | `"broadcast"`, from an optional top-level `[inputs]` table whose only key is
    `ball_fill`.
  - `pipeline.run.prepare_inputs(config, inputs) -> inputs.with_ball_fill(config.ball_fill)`. `run()` and the
    benchmark's `predict()` call it before any source runs.
  - `MatchInputs(..., broadcast_ball=None)`: `broadcast_ball` returns the array given, or raises `AssetUnavailable`
    with the sentence of the other assets.

- [ ] **Step 1: write the failing checks** in `pipeline/selftest.py`, all on synthetic arrays:
  - `test_config_inputs_table`: no `[inputs]` gives `ball_fill == "none"`; `[inputs] ball_fill = "broadcast"` loads;
    `ball_fill = "video"` and an unknown key under `[inputs]` raise `ConfigError` naming the key.
  - `test_with_ball_fill_none_is_identity`: `inputs.with_ball_fill("none") is inputs`.
  - `test_align_recovers_known_offsets`: a synthetic tactical ball (a random walk on the pitch, 20% NaN in runs) and a
    broadcast that is the same walk shifted by 4,014 frames in half 1 and 26,919 in half 2. It has 0.5 m Gaussian
    noise, 35% invalid in runs and its own 10% NaN. `align` returns exactly those offsets, and each fine score is above
    0.5.
  - `test_align_refuses_no_agreement`: an independent broadcast walk raises `AlignmentError`.
  - `test_align_refuses_two_peaks`: a broadcast holding the tactical walk at two offsets 1,000 frames apart, equally
    often, raises `AlignmentError`.
  - `test_align_refuses_too_few_pairs`: a half with under 300 coarse pairs raises `AlignmentError` naming the half.
  - `test_live_needs_valid_players_and_distance`, one frame for each condition:
    - 3 broadcast players → False;
    - invalid homography → False;
    - median nearest distance 2.5 m → False;
    - 6 players at 0.3 m → True.
  - `test_usable_drops_runs_shorter_than_13`: runs of 12 and 13 usable frames give False and True.
  - `test_fill_writes_only_missing_graded_frames`:
    - a present tactical ball is unchanged, and so is an ungraded frame;
    - a filled frame's `ball_xy` equals the broadcast's, and its `ball_speed` stays NaN;
    - `players_xy` is identical;
    - `filled` is True exactly on the written frames.
  - `test_fill_is_deterministic`: two calls give identical bytes.
  - `test_filled_inputs_is_an_inputs`: `isinstance(view, Inputs)`; `view.pitch`, `view.graded`, `view.n_frames` and
    `view.metres` match the base's; `repr(view)` holds no array.
- [ ] **Step 2:** run `.venv/Scripts/python pipeline/selftest.py`. The new checks fail; the 193 existing ones pass.
  Commit `R028 Task 1: failing checks`.
- [ ] **Step 3: implement.**
  - `align` is the design's 2.3, exactly: coarse, then fine, then acceptance.
  - `live` takes the nearest present tactical player in metres per present broadcast player; a frame with fewer than 4
    broadcast players or an invalid homography is not live.
  - `usable` = mapped ∧ live ∧ broadcast ball present, then runs < 13 removed.
  - Only the `broadcast_*` properties load the broadcast files, and only `with_ball_fill("broadcast")` needs them.
  - `pipeline/run.py` logs one `fill` line (offsets, fine scores, live share, filled frames).
  - When the fill is on, the manifest's `inputs` gains `broadcast`: the alignment, the counts, and the broadcast ATD
    cache's source record plus the homography file's size and sha256.
- [ ] **Step 4:** the whole pipeline suite passes. Commit `R028 Task 1: the broadcast's ball and the fill switch`.

### Task 2a: the simulation's core, pure functions (worktree `r028-sim`, in parallel with Task 1)

**Files:**
- Create: `pipeline/benchmark/broadcast_sim.py` (core only: no BAR-ATH loading, no CLI yet).
- Create: `pipeline/benchmark/checks/broadcast_sim.py`, and list it in `checks/__init__.py` MODULES.

**Interfaces:**
- Produces:
  - `Pools`, a frozen dataclass, no array in its repr:
    - `gap_lengths` (k,) int and `gap_alive` (k,) bool;
    - `gap_fill`, the concatenated bool masks, with `gap_starts` (k + 1,) offsets into it;
    - `error_xy` (m, 2) metres, concatenated, with `error_starts` (r + 1,).
  - `gaps_of(ball_present, graded) -> [(start, stop)]`: the maximal missing runs, cut at the graded blocks.
  - `gap_alive(gaps, codes) -> (k,) bool`: alive if at least 0.5 of the gap's codes are `H` or `A` (pm codes).
  - `simulate(ball_present, clean_ball_m, driver_codes, graded, pitch, pools, rng) -> (n, 2)` normalised, NaN where
    unfilled. It follows the design's 3.4, steps 1 to 5.
  - `measure_fill(ball_present, filled, codes, graded, fill_xy_m, reference_xy_m, pitch) -> dict`: F1 to F5 and W1 to
    W3 as raw numbers, with their counts. F5 and W3 read only the filled frames.
- Consumes: nothing from Task 1; arrays only.

- [ ] **Step 1: write the failing checks**, all synthetic:
  - `test_gaps_are_cut_at_graded_blocks`;
  - `test_gap_alive_threshold`: exactly 0.5 `H`/`A` → alive, 0.49 → D;
  - `test_simulate_copies_a_donor_of_equal_length`: one donor of length m in the class, and a DFL gap of length m,
    give the donor's mask exactly;
  - `test_simulate_keeps_both_edges_when_stretching`: a donor of length 50 with fill only on its first 10 and last 5
    frames, mapped onto 100 frames, fills the first 20 and the last 10;
  - `test_simulate_picks_by_class`: a D gap never takes an alive donor;
  - `test_simulate_draws_among_the_5_nearest_lengths`, over many seeds;
  - `test_simulate_adds_the_error_runs`: the filled positions equal the clean ball plus an error run's first f offsets;
  - `test_simulate_leaves_nan_where_dfl_has_no_ball`;
  - `test_simulate_is_seeded`: the same generator seed gives the same bytes, and different seeds give different
    donors;
  - `test_simulate_takes_no_match`: `inspect.signature(simulate)` has exactly the parameters above;
  - `test_measure_fill_on_known_masks`: hand-built masks give the F and W values computed by hand.
- [ ] **Step 2:** run `.venv/Scripts/python pipeline/benchmark/selftest.py`. The new checks fail and the rest pass.
  Commit.
- [ ] **Step 3: implement.** Commit `R028 Task 2a: the simulated broadcast's core`.

### Task 2b: the simulation's wiring (lane, after Tasks 1 and 2a are merged)

**Files:**
- Modify:
  - `pipeline/benchmark/broadcast_sim.py`;
  - `pipeline/benchmark/run.py`;
  - `pipeline/benchmark/conventions.py`;
  - `pipeline/benchmark/choose.py`;
  - `pipeline/benchmark/README.md`;
  - `pipeline/benchmark/checks/broadcast_sim.py`, `checks/inputs.py`, `checks/run.py`, `checks/choose.py` (or the
    modules that hold those topics).

**Interfaces:**
- Consumes: Task 1's `Inputs.broadcast_ball`, `broadcast.usable` and `with_ball_fill`, and Task 2a's core.
- Produces:
  - `build_pools(inputs) -> Pools`, from BAR-ATH's `Inputs` (design 3.3):
    - it decodes `pipeline/configs/r024_both.toml` in-process with `pipeline.run.run` into a file under the pools'
      cache folder, and raises unless that CSV's sha256 is `e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170`;
    - then the gaps' real fill masks and classes, and the near-gap error runs.
  - `load_pools(data_dir=None) -> Pools`, cached under `<data>/cache/benchmark/broadcast_sim/<digest>/`.
    - The key holds the tactical and broadcast ATD caches' source records, the broadcast homography's size and sha256,
      `r024_both.toml`'s text, its CSV's sha256 and `POOLS_VERSION = 1`.
    - A broken cache is absent, never an error, as in `calibration.load`.
  - `digest(pools) -> str`.
  - `served_inputs_for(config, match, atd, profile, seed, data_dir, pools, plain=None) -> MatchInputs`. With
    `config.ball_fill == "none"` it is the plain `MatchInputs`. Otherwise:
    - the driver's codes are `predict(dataclasses.replace(config, ball_fill="none"), plain)`;
    - `simulate` builds the ball;
    - the result is `MatchInputs(…, broadcast_ball=…)`.
  - `run.evaluate_truths` serves each fill config through it (the driver's codes cached per twin and seed). Every
    report of a run gains `broadcast_sim`: the digest of the pools that run loaded, the same for every config of the
    run (`main` included), or null when no config of the run fills.
  - `choose.SAME` gains `"broadcast_sim"`.
  - `conventions.py latency … dfl` serves a fill config the same way.
  - The CLI:
    - `broadcast_sim.py pools`;
    - `broadcast_sim.py fidelity --profile P --seeds N CONFIG [--data-dir D]`: the design's 3.5 tables and the
      verdict;
    - `broadcast_sim.py bar-ath CONFIG [--data-dir D]`: the design's 7.4 alignment band, the live share, the tag
      check, the `D` and `X` sanity and the reported figures.
  - **Exit codes:**
    - fidelity: 0 pass, 1 a gate fails, 3 a check fails, 2 an error;
    - bar-ath: 0 pass, 1 a check fails (DROP), 3 the alignment band or the live share (STOP), 2 an error.

- [ ] **Step 1: write the failing checks**, all synthetic or with the harness's fake matches:
  - `test_served_inputs_for_none_is_plain`;
  - `test_served_inputs_for_uses_the_twin_driver`: a stub predict records the config it was given, and `ball_fill`
    is `"none"`;
  - `test_match_inputs_hold_no_truth` is extended with a `broadcast_ball`;
  - `test_every_report_of_a_run_carries_the_same_broadcast_sim`: a run of a fill config beside one without carries the
    digest in both reports, and a run with no fill config carries null;
  - `test_choose_refuses_mixed_broadcast_sim`;
  - `test_pools_refuse_labels_that_are_not_e6496a9b`: a stub decode with another sha256 raises;
  - `test_pools_cache_key_changes_with_r024_text`;
  - `test_fidelity_exit_codes` and `test_bar_ath_exit_codes`, on stubbed measurements;
  - `test_tag_check_counts_fill_only_teacher_frames`: the teacher is `conventions.TOUCH`. The frames are those where
    its `home` is a number on the filled inputs and NaN on the plain ones.
- [ ] **Step 2:** the checks fail. Commit.
- [ ] **Step 3: implement.**
  - The tag check is `tag_stats.team_agreement`, on the teacher `controlled_touch.compute(..., **conventions.TOUCH)`,
    once on the plain inputs and once on the filled ones (design 7.4).
  - Commit `R028 Task 2b: the simulated broadcast wired into the benchmark`.
- [ ] **Step 4:** both suites pass. `main.toml` still writes `e6496a9b…` (the controller checks this on the lane, with
  one heavy run).

### Task 3: the configs (controller, lane)

- [ ] `pipeline/benchmark/configs/r028/r028_preshift.toml` is `main.toml` with `[inputs] ball_fill = "broadcast"` and
  without `[[refine]]`. Its header names R028 and the design.
- [ ] `pipeline/benchmark/configs/r028/r028_post.toml` is the preshift config plus `main.toml`'s `[[refine]]` (+1, −2,
  0).
- [ ] The configs check and both suites pass. Commit `R028 Task 3: the configs`.

### Task 4: R028 part 1 (controller, with an independent reviewer)

**Files:** `docs/results.md`: a new `### R028` after R027, before `## Submission log`, and a scoreboard row reading
"declared".

- [ ] **4a.** Write part 1 from the design's sections 2, 3 and 7 to 8, with Tian's answers in place. It states:
  - the constants;
  - the commands (the scripts below) and the exit codes;
  - the STOPs, the retry, ADOPT and DROP;
  - "what it cannot settle":
    - T38's blind gaps, so the benchmark is generous (W1, W2);
    - the broadcast's failure modes beyond A20's statistics;
    - the driver's pessimism (3.2);
    - T33's grain.
- [ ] **4b.** Dispatch an independent reviewer (opus) on part 1, against the design, the code of Tasks 1 to 3, and R024
  and R027. Resolve every finding in the text. Then commit `R028 part 1: declared before any run` **before** any run on
  data, and record the commit in the ledger.

### Task 5: the runs (controller; one heavy process at a time, through `heavy-r028.ps1`, from the run worktree)

Scripts go in `.superpowers/bin/`, in the style of `r027-gate.ps1`:
- before each step, stamp HEAD and an empty `git status --porcelain`;
- a 90 s cool-down between steps;
- stop at the first non-zero exit, except where an exit code is a verdict (fidelity's 1 and 3, `bar-ath`'s 1 and 3);
- every log ends with `EXIT=<code>`.

`<data>` is `C:/Users/Tian/Desktop/BarcaHackathon/data`, written in full. `<cfg>` is `pipeline/benchmark/configs/r028`.

| script | steps |
|---|---|
| `r028-fidelity.ps1` | `broadcast_sim.py pools`, then `broadcast_sim.py fidelity --profile tracking --seeds 3 <cfg>/r028_preshift.toml`, then the same with `--profile context` (report only) |
| `r028-lag.ps1` | `conventions.py latency <cfg>/r028_preshift.toml dfl --profile tracking --seeds 3 --truth raw --drop-match-only` |
| `r028-bar.ps1` | `run.py pipeline/configs/main.toml <cfg>/r028_final.toml <cfg>/r028_preshift.toml --profile tracking --seeds 3 --truth both --drop-match-only`, then the same on `--profile context` |
| `r028-choose` (light, run directly) | `choose.py --reference r028_preshift --objective mean r028_final --profile tracking --data-dir <data> --out <data>/runs/r028/guard-tracking.json`, the same on `context`; then `choose.py --reference main --objective mean <candidate> …` into `bar-tracking.json` and `bar-context.json` |
| `r028-barath.ps1` | `broadcast_sim.py bar-ath <candidate config>`, then `pipeline/run.py <candidate config>` twice, then `label_diff.py pipeline/configs/main.toml <candidate config>` |

**Reading the numbers.** From `choose.py`'s JSON (`--out`, 9 decimals), as R027:
- the objective's pooled Δ is `versus.mean.pooled_delta`, with the per-match deltas and the p beside it;
- the alive/dead F1 is `rows.<stem>.alive_dead`;
- `main`'s 4-decimal reproduction is 0.5363 and 0.5764.

**Errors:** the design's section 8 rule, exactly.

- [ ] **5.1 Fidelity.** Detach the run worktree at part 1's commit and run `r028-fidelity.ps1`. The pools step decodes
  `r024_both.toml` itself and stops unless its CSV is `e6496a9b…`.
  - **Exit 0:** write part 2a and continue automatically (Tian's answer 3).
  - **Exit 1 or 3:** STOP for the advisor (design 3.5). The advisor compares the code with the design's section 3 text,
    never with the number that missed.
    - If the code departs from the text, one fix restores the text, with no constant, threshold, donor rule or
      tolerance changed. It is logged as a deviation, reviewed, and the fidelity run is repeated once.
    - Otherwise, part 2a, then DROP.
- [ ] **5.2 The lag.** Run `r028-lag.ps1`. Compute s_b by R012's rule. Write `r028_final.toml` (a copy of `r028_post`
  under that stem if s_b is +1/−2/0). Commit.
- [ ] **5.3 The bar.** Run `r028-bar.ps1`, then `r028-choose`.
  - `main` not reproduced → **STOP**, advisor.
  - The shift's guard picks the candidate.
  - Apply the bar and the vetoes:
    - pass → 5.4;
    - fail with the `tracking` objective Δ > 0 and R028b not run → **Task 5b**;
    - otherwise → DROP.
- [ ] **5.4 BAR-ATH.** Run `r028-barath.ps1`.
  - `bar-ath` exit 3 → **STOP**, advisor.
  - Exit 1, a different sha256 twice, `D` outside [30%, 62%], or an `X` in a graded block → DROP.
  - Otherwise → ADOPT (automatically, if Tian said so).
- [ ] **5.5** Write part 2b, with every table, in R027's style. Commit.

**Task 5b (R028b, only from 5.3):**
1. A task that adds `filled_confidence` to `controlled_touch`, defaulting to `confidence` and read with
   `inputs.ball_filled`.
   - Checks: the default gives identical output; on the filled frames the vote uses 0.79.
   - Review, then merge.
2. `r028b_preshift.toml`, part 1b reviewed and committed.
3. Then 5.1 to 5.5 for R028b.

Any failure is DROP.

### Task 6: ADOPT or DROP (controller)

- [ ] **ADOPT.**
  1. Copy the candidate to `pipeline/configs/r028_broadcast_ball.toml` and make `main.toml` equal to it, with the header
     comments as R024 wrote them.
  2. `pipeline/run.py pipeline/configs/main.toml` twice: the same sha256. Log it in the submission log as not uploaded.
  3. The pipeline suite, the benchmark suite and `selftest.py --data` (heavy, alone) pass.
  4. Commit `R028 adopted: main.toml fills the ball from the broadcast (sha256 …)`.
  5. Only then, the public window as a correctness check (T01, T27): `pipeline/run.py pipeline/configs/main.toml
     --score`, reported in part 2b. It chooses nothing.
- [ ] **DROP.**
  1. One commit removes `pipeline/inputs/broadcast.py`, the `Inputs` and `MatchInputs` additions, the config key, the
     `atd_cache` folder parameter, `broadcast_sim.py` and its checks and registration, the wiring in `run.py`,
     `conventions.py` and `choose.py`, the configs and the README lines. It names the last commit that holds them.
  2. Both suites pass, and `main.toml` still writes `e6496a9b…`.

### Task 7: docs, final review, merge (controller)

- [ ] **The docs:**
  - `docs/proposals.md`: P15's status and row;
  - `docs/results.md`: R028 (and R028b), the scoreboard and the submission log;
  - `docs/traps.md`, if something cost real time;
  - CLAUDE.md: the broadcast rows of the alignment table and the architecture sentence, if adopted;
  - `pipeline/README.md` and `pipeline/benchmark/README.md`;
  - both technical documents: the broadcast as an asset used, and `data/broadcast/` needed to regenerate, if adopted;
  - the site.
- [ ] **The final review** on the most capable model: the whole branch against the design, R028 and this plan. One fix
  round, then adjudicate what is left.
- [ ] **The merge.** `git merge --no-ff r028-broadcast-ball` into `main` after both suites (and `selftest.py --data` if
  adopted) pass on the merged tree. Push to origin. Delete the worktrees and the branches. Update the handoff and the
  ledger.

## The decision flow at a glance

| event | action |
|---|---|
| fidelity exit 0 | lag, bar, BAR-ATH (automatically, if Tian said so) |
| fidelity exit 1 or 3 | STOP, advisor: code against text. A fix only restores the text, logged, reviewed and rerun once; otherwise DROP |
| `main.toml` not reproduced in the bar run | STOP, advisor |
| the bar or a veto fails, `tracking` Δ > 0, R028b not run | R028b (Task 5b) |
| the bar or a veto fails otherwise | DROP |
| `bar-ath` exit 3 | STOP, advisor |
| `bar-ath` exit 1, a sha256 differing between two runs, `D` outside [30%, 62%], an `X` in a graded block | DROP |
| everything passes | ADOPT (automatically, if Tian said so) |
| an exit 2, a crash or a log without `EXIT=` | the errors rule: a clear outside cause is removed and the step rerun once; a reaped run waits for Tian; a traceback in the declared code goes to the advisor |

**The advisor** is an opus agent briefed with the design, R028 and the aggregates. It rules on the STOPs and the
unforeseen. Each ruling goes into the ledger as `Advisor (for Tian): <decision> — <reason>`, and into the handoff for
Tian to overturn.
