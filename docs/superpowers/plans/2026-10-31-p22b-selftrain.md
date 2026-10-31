# P22(b) self-training (R027) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the code R027 needs, and the controller's runbook from the gate to ADOPT or DROP. The code is a noise-aware
per-half intercept on P07's logit, fitted on the other half's touch-episode frames: the estimators, the gate tool, the
`shape_selftrain` source and its configs.

**Architecture:**
- **One helper module,** `pipeline/evidence/recalibration.py`, holds the teacher's rows, the estimators and the
  leave-one-block-out fit. Both consumers import it, so the gate measures exactly what would be served:
  - the benchmark tool `pipeline/benchmark/selftrain.py`, which computes the gate before any source exists;
  - the source `pipeline/evidence/shape_selftrain.py`, built while the gate runs and merged only if it passes.
- **Nothing changes `main.toml`'s output** until the controller adopts, by the flow of the design's section 7.

**Tech Stack:** Python 3.11.9, numpy 2.4.6 and `math` in `pipeline/` (the pipeline's pins). `pipeline/benchmark/` may
also use what `requirements-benchmark.txt` pins (scikit-learn, scipy). The checks run under `pipeline/selftest.py` and
`pipeline/benchmark/selftest.py`.

**Spec:** `docs/superpowers/specs/2026-10-31-p22b-selftrain-design.md` (revised after the independent review), plus
R027 part 1 in `docs/results.md` once it is written. Where this plan and they differ, the spec and the declaration win.

## Global Constraints

- **Match data never reaches an LLM.** Do not Read, `cat`, `head` or `grep` any file under `data/`. Print and write
  aggregates only, never frame-level arrays or rows. JSON reports hold per-block and per-match numbers only.
- **Don't modify `evaluation/`.** Import `possession_metrics` as `pm`, the way `pipeline/benchmark/score.py` does.
- **`main.toml` keeps its output.** Until the controller's adoption commit, `python pipeline/run.py
  pipeline/configs/main.toml` must still write `main.csv` sha256
  `e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170`. `pipeline/evidence/shape_possession.py`,
  `controlled_touch.py` and `dead_ball_model.py` are not edited.
- **Dependencies:** numpy and the standard library only in `pipeline/` (no scikit-learn, no scipy).
  `pipeline/benchmark/` may use what `requirements-benchmark.txt` pins. Nothing under `pipeline/evidence`, `decode` or
  `refine` imports `pipeline.benchmark` (the import scan in `checks/inputs.py`).
- **Determinism.** Sums that decide a fitted offset use `math.fsum`. Offsets are rounded to 9 decimals before use, and
  roots are found by bisection on [−10, 10] to 1e-12. The same inputs give the same bytes whatever the row order or the
  thread count.
- **The constants are the design's,** fixed before any run:
  - ε is 0.07, which is 1 − `controlled_touch`'s confidence 0.93 in `main.toml`;
  - the floor is 500 rows a class, π̂ is clipped to [0.05, 0.95], and the clip of `home` is [0.02, 0.98];
  - the gate's limits are 0.002 nats of pooled gain and 0.002 nats of harm per match;
  - rung 3's L2 penalty is 0.01 a row;
  - the Smart Tagging margin is 0.005;
  - the teacher is `main.toml`'s touch rule: 1.5 m, 6.0 m/s, 1 frame, duel 1.0, called with confidence 0.99 and hold
    0.5. The dead-ball mask is `dead_ball_model` at `main.toml`'s model, P(alive) ≥ 0.5.
- **Style:** match the surrounding code. Module docstrings explain the why; constants go in `# Params` blocks; a
  ValueError message names the parameter and the fix; comment density as in the file being edited.
- **No AI attribution** in commit messages (no Co-Authored-By, no "Generated with", no session trailer).
- **Running things:** run tests with `.venv/Scripts/python`. One heavy process at a time on this machine. The suites
  are light. Tasks never start `run.py`, `selftest.py --data`, `selftrain.py gate`/`bar-ath` on real data or any
  `train_*`; the controller runs those.

## Review Focus

- **A block whose other blocks give under 500 rows of a class,** or one class only, or no root in the bracket: b = 0
  there, a reason in the report, no exception. A red card or a short block can do it on real data.
- **Relabelling the two teams gives `home` → 1 − `home`** within 1e-9, through the fit (b → −b) and the clip: the source
  keeps P07's antisymmetry.
- **Frames where nobody is present** (z NaN) never become teacher rows and stay NaN in `home`. The `scope` rule (`gaps`:
  only where the served ball is missing) is `shape_possession`'s, exactly.
- **With every b equal to 0, the source equals `shape_possession` to the bit.** Existing configs and `main.toml`'s CSV
  are unchanged.
- **No truth reaches the source or the teacher:** the tool reads the truth only from `truth.truth_codes(match)`, beside
  the inputs. The tool's JSON holds no array longer than the seven matches' table.

## Execution topology (parallel waves, worktrees, commits)

| item | where |
|---|---|
| lane branch `p22b-selftrain` | worktree `C:\Users\Tian\Desktop\BarcaHackathon-p22b` (exists): the design, this plan, R027's declarations, Tasks 1 and 2, the merges, part 2 and the docs |
| source branch `p22b-source` | worktree `C:\Users\Tian\Desktop\BarcaHackathon-p22b-source`, created from the lane head after Task 2: Task 3, merged into the lane only after a GO |
| run worktree | `C:\Users\Tian\Desktop\BarcaHackathon-p22brun`, detached at the commit a run uses (`git -C … checkout --detach <sha>`), `--data-dir C:\Users\Tian\Desktop\BarcaHackathon\data` |
| ledger | `.superpowers/sdd/2026-10-31-p22b-selftrain/progress.md` in the main checkout (first line names this plan); logs in `…/logs/` |
| handoff | `.superpowers/sdd/run-2026-10-31/handoff.md` (the hooks' path), kept current after every task and run |

```
Wave 1:  Task 1 (implementer, lane)            ‖  Task 4a: R027 part 1 drafted (controller)
Wave 2:  Task 2 (implementer, lane)            ‖  Task 4b: the declaration's independent review, then its commit
Wave 3:  GATE runs (controller, run worktree)  ‖  Task 3 (implementer, source worktree; light suites only)
Wave 4:  by the gate's exit code: merge the source → lag → bar → BAR-ATH (Task 5) → ADOPT or DROP (Task 6) → Task 7
```

**Commits.** Each task commits after its failing tests and again after its implementation. The controller commits:
- the declaration, before any run it governs;
- part 2a after the gate, and part 2b after the bar and the BAR-ATH checks;
- the adoption or the drop;
- the docs.

Merges are `--no-ff`. `main` is pushed to the private origin (TianK003/BarcaHackathon) after the final merge, as in the
earlier lanes.

---

### Task 0: Controller setup (no subagent)

- [ ] Create the ledger `.superpowers/sdd/2026-10-31-p22b-selftrain/progress.md`, first line
  `# SDD ledger — plan: docs/superpowers/plans/2026-10-31-p22b-selftrain.md`, and record Tian's four answers.
- [ ] Copy `.superpowers/bin/heavy-p23.ps1` to `.superpowers/bin/heavy-p22b.ps1`, with `$logs` set to
  `…\.superpowers\sdd\2026-10-31-p22b-selftrain\logs`; everything else unchanged (1 BLAS thread, below-normal priority,
  a ≥ 3 GB guard, `MTAH_DATA_DIR`).
- [ ] `git worktree add --detach C:/Users/Tian/Desktop/BarcaHackathon-p22brun p22b-selftrain`.

### Task 1: `pipeline/evidence/recalibration.py`: the teacher, the estimators, the leave-one-block-out fit

**Files:**
- Create: `pipeline/evidence/recalibration.py`
- Modify: `pipeline/selftest.py` (new checks, in a section after P07's)

**Interfaces:**
- Consumes: `controlled_touch.compute(run, *, radius, max_rel_speed, min_frames, confidence, duel_radius=1.0,
  hold=0.5)`; `dead_ball_model.compute(run, *, model, min_prob=0.001)`; `inputs.graded`, a list of `(half, start, stop)`.
- Produces:
  - `TEACHER_CONFIDENCE = 0.99`, `TEACHER_HOLD = 0.5`, `ALIVE_LEVEL = 0.5`, `MIN_ROWS = 500`, `PI_CLIP = (0.05, 0.95)`,
    `BRACKET = (-10.0, 10.0)`, `TOLERANCE = 1e-12`, `OFFSET_DECIMALS = 9`.
  - `check_teacher(teacher) -> dict`: the table with `duel_radius` defaulted to 1.0. A ValueError for a non-dict, a
    missing `radius`/`max_rel_speed`/`min_frames`, any other key, or `confidence`/`hold` (the source fixes them).
  - `teacher_labels(run, *, alive_model, teacher) -> (y, rows)`:
    - `y` is an (n,) float64 array: 1.0 on a kept home episode frame, 0.0 on an away one, NaN elsewhere;
    - `rows` is an (n,) bool array: `y` known **and** the dead-ball model's alive ≥ 0.5.
    - z's finiteness is not tested here; `block_offsets` takes the z mask.
  - `class NoRoot(ValueError)`.
  - `Offset = namedtuple("Offset", "offset s pi_hat share")`.
  - `noise_aware_offset(z, y, *, epsilon) -> Offset` (E2). `y` is 0/1 int, both classes present, else ValueError.
    `offset` is rounded to `OFFSET_DECIMALS`; `share` is the mean of y. `NoRoot` if the score does not change sign on
    `BRACKET`.
  - `balanced_offset(z, y) -> Offset` (E1, the class-balanced intercept with offset z; the gate's oracle on clean
    labels): `s` equals `offset`, `pi_hat` is None, rounded the same way.
  - `block_offsets(z, y, rows, blocks, *, fit, min_rows=MIN_ROWS) -> list[BlockOffset]`, one per block in `blocks`
    order:
    - `blocks` is a list of `(start, stop)`;
    - `fit` is a callable `(z, y) -> Offset`; for E2, `functools.partial(noise_aware_offset, epsilon=ε)`;
    - `y` is `teacher_labels`' float array, read as 0/1 on the selected rows;
    - `BlockOffset = namedtuple("BlockOffset", "offset rows_home rows_away fallback fitted")`, where `fallback` is
      `None`, `"few rows"` or `"no root"`, and `fitted` is the `Offset` or None;
    - block k's offset is `fit` on the frames of every *other* block where `rows` and `isfinite(z)` hold;
    - with fewer than `min_rows` rows of either class there, the offset is 0.0 with `"few rows"`; a `NoRoot` gives 0.0
      with `"no root"`.
  - `adapted_logit(z, blocks, offsets) -> np.ndarray`: z plus block k's offset on block k's frames, NaN kept, frames
    outside every block unchanged.

**The estimators** (the design's section 2):
- **E2.** g(s) = Σ ∂/∂s [ỹ log q + (1 − ỹ) log(1 − q)], with q = ε + (1 − 2ε)σ(z + s), summed with `math.fsum`. s is
  found by bisection on `BRACKET` to `TOLERANCE`. ȳ is the mean of y. Then
  π̂ = clip((ȳ − ε)/(1 − 2ε), *PI_CLIP), and the offset is s − log(π̂/(1 − π̂)).
- **E1.** g(b) = Σ wᵢ (yᵢ − σ(zᵢ + b)) with wᵢ = 0.5/n_class(yᵢ), by the same bisection.

- [ ] **Step 1: Write the failing tests** in `pipeline/selftest.py`, on synthetic arrays only. The generator: z | H ~
  N(1.2, 2.4), z | A ~ N(−1.2, 2.4) (z is then the exact log-likelihood ratio), the model's logit is z − b_true, and
  labels are flipped per class.
  - `test_recalibration_e2_recovers_offsets`: shares 0.35, 0.5 and 0.65; b_true −0.5, 0 and +0.5; 7% symmetric noise;
    n 20,000; seeds 0 to 4. The mean |E2 − b_true| is ≤ 0.03 in every cell.
  - `test_recalibration_e1_leans_to_the_minority`: at share 0.65, b_true 0 and 7% noise, E1's `.offset` is below −0.05
    and E2's is within ±0.03. This documents why E2 is served.
  - `test_recalibration_relabelling_negates_the_offset`: `noise_aware_offset(-z, 1 - y).offset` equals
    −`noise_aware_offset(z, y).offset` within 1e-9, and the same for `balanced_offset`.
  - `test_recalibration_ignores_row_order`: a permutation of the rows gives the same offset bit for bit.
  - `test_recalibration_refuses_one_class_and_reports_no_root`: one class gives ValueError. A sample where every home row
    has z = +50 and every away row −50 with ε = 0.07 still has a root. A case built to have none gives `NoRoot`.
  - `test_recalibration_block_offsets_leave_one_block_out`: three blocks with planted per-block offsets. Block k's offset
    equals `fit` on the other two blocks' rows. Changing block k's own labels leaves its offset unchanged. 499 rows of one
    class in the other blocks give 0.0 with `"few rows"`, and 500 fit. Frames with NaN z never count.
  - `test_recalibration_adapted_logit`: the offset lands on its block only, NaN stays NaN, and frames outside the blocks
    are untouched.
  - `test_recalibration_teacher_labels`: on a `touch_scene` with known episodes and a stub dead-ball model file that
    makes one stretch dead, `y` is 1/0 exactly on the kept episodes, `rows` excludes the dead stretch, and
    `check_teacher` refuses `{"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "hold": 0.6}` and a misspelt key.
    Write the stub model with the existing helpers for P05's file, or monkeypatch `dead_ball_model.compute` inside the
    test.
- [ ] **Step 2: Run them to see them fail:** `.venv/Scripts/python pipeline/selftest.py`. The new checks fail and the
  rest pass.
- [ ] **Step 3: Implement** `pipeline/evidence/recalibration.py` with the interfaces above. Its module docstring
  explains E2, why not E1 (the synthetic table of the design, in two sentences) and the leave-one-block-out rule.
- [ ] **Step 4: Run** `.venv/Scripts/python pipeline/selftest.py`: all pass. Run `.venv/Scripts/python
  pipeline/benchmark/selftest.py`: all pass (the import scan).
- [ ] **Step 5: Commit** `git add pipeline/evidence/recalibration.py pipeline/selftest.py && git commit -m "P22(b):
  recalibration.py, the touch teacher's rows and the noise-aware per-block intercept (E2), with E1 for the oracle"`

### Task 2: `pipeline/benchmark/selftrain.py`: the gate on DFL and the label-free checks on BAR-ATH

**Files:**
- Create: `pipeline/benchmark/selftrain.py` and `pipeline/benchmark/checks/selftrain.py`
- Modify: `pipeline/benchmark/checks/__init__.py` (register the module), `pipeline/benchmark/README.md` (one paragraph)

**Interfaces:**
- **Consumes, from Task 1:** `recalibration.*`. **From the benchmark:**
  - `dfl.load_match`, `truth.truth_codes`, `train_dead_ball.served_inputs`;
  - `train_shape._calibration_for` and `train_shape._seed_list` (import them; do not copy);
  - `run.resolve_match_ids`, `conventions.TOUCH`, `pipeline.config.load_config`.
- **Consumes, from the pipeline's sources:** `shape_possession.load` and `shape_possession._logit_home(inputs, path,
  key, model)` (the fold `inputs.game` if the artifact has it, else `full`); `smart_tagging.compute(run, *, c_team,
  spread)`; `wrong_team._counts(present, slot_team)`.
- **Produces:**
  - `settings(config_path="pipeline/configs/main.toml") -> Settings(teacher, epsilon, alive_model, shape_model,
    min_prob, c_team)`, read from `main.toml`'s entries:
    - `controlled_touch`'s radius, max_rel_speed, min_frames and duel_radius (default 1.0);
    - ε = round(1 − confidence, 12);
    - `dead_ball_model`'s model; `shape_possession`'s model and min_prob; `smart_tagging`'s c_team.
  - `head_fit(z, y, X, *, epsilon, penalty=HEAD_PENALTY) -> (offset, beta)`, rung 3 as the design's section 3 declares
    it:
    1. standardise X on the rows (NaN → 0 afterwards; a column with no spread → 0);
    2. β by class-balanced, L2-penalised (0.01 a row) Newton with offset z and a free intercept, to |Δ| < 1e-10, at most
       50 steps;
    3. the intercept re-fitted by `recalibration.noise_aware_offset` with offset z + X̃β.

    It also returns the standardisation, so the head can be applied to the held-out block.
  - `gate(profile="tracking", seeds=3, match_ids=None, matches=None, data_dir=None, settings=None) -> dict`, the gate's
    report. `matches` is `{id: DflMatch}` to use instead of loading (the checks pass synthetic ones), and `settings`
    replaces `settings()` (the checks pass small synthetic model files). A match whose fold is missing from the shape
    artifact is an error: `full` saw it, so it would be scored in sample.
  - `gate_verdict(per_match, pooled) -> (rung1_go, rung3_go)`, the two conditions on the report's numbers, and the exit
    code from them.
  - `bar_ath(gate_report, data_dir=None, inputs=None, settings=None) -> dict`, the BAR-ATH report. `inputs` replaces
    BAR-ATH's `Inputs` in the checks.
  - The CLI: `selftrain.py gate [--profile P] [--seeds N] [--matches ID ...] [--data-dir D] [--no-write]` and
    `selftrain.py bar-ath --gate-report PATH [--data-dir D] [--no-write]`.
  - Reports go to `<data dir>/runs/selftrain/gate-<profile>.json` and `…/bar-ath.json`, floats at 9 decimals.
  - **Exit codes:**

    | subcommand | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | `gate` | rung 1 GO | both NO-GO | error | | rung 1 NO-GO, rung 3 GO |
    | `bar-ath` | every check passes | a check fails | error | an offset outside the band (STOP) | |

**`gate`, per (match, seed),** one Inputs at a time, deleted after use:
1. **The arrays:** z from the fold; the teacher's `y` and `rows`; the blocks `[(start, stop) for _, start, stop in
   inputs.graded]`; codes from the truth.
2. **The variants' logits:**
   - base: z;
   - rung 1: E2, cross-fitted;
   - E1, cross-fitted;
   - single: E2 fitted on all blocks' rows, applied to every block;
   - rung 3: `head_fit`, cross-fitted per block.
3. **The oracles,** on clean labels (`balanced_offset`), per block k: b\*_gap on block k's gate frames (truth H/A, served
   ball missing, z finite), and b\*_touch on the other blocks' teacher rows with the truth's labels.
4. **Per block, accumulated per match:**
   - the frames, the sum of log-loss and the right calls of each variant's served home (`clip(σ(·), min_prob,
     1 − min_prob)`) on the gate frames;
   - the same on the hold stratum (truth alive, ball present, not a teacher row);
   - the held-out touch agreement at `conventions.TOUCH`, for base and rung 1;
   - the offsets and the oracles;
   - π̂, the teacher's home share, the rows per class, the masked share (teacher frames with alive < 0.5 over teacher
     frames) and the fallbacks;
   - the teacher's precision against the truth on its rows.
5. **The report:**
   - per match and pooled: each variant's log-loss and accuracy;
   - B = mean over matches of the mean over seeds and blocks of |b\*_gap|, and E/B (R020's form);
   - the decomposition: the mean over matches of |b − b\*_touch| and of |b\*_touch − b\*_gap|;
   - `band` = the largest |b\*_gap| over all (match, seed, block) cells;
   - the gate for rung 1 and for rung 3: `pooled_gain = base − variant ≥ 0.002`, and per match `variant − base ≤ 0.002`.

**`bar_ath`,** on BAR-ATH's `Inputs` (`pipeline.inputs.Inputs(data_dir=...)`, the `full` model):
- **Per block:** E2's b with its fallback, π̂, the teacher's share, the masked share, and the share of teacher rows on
  frames where home shows over 11 slots, and where away does.
- **The band:** read from the gate report. Any |b_k| above it → exit 3.
- **The gap check:** on graded frames with the ball missing, alive ≥ 0.5 and a Smart Tagging vote (`home` not NaN from
  `smart_tagging.compute(run, c_team=Settings.c_team, spread="frame")`), the agreement of base and adapted `home` (≥ 0.5)
  with the vote (≥ 0.5). It passes iff adapted ≥ base − 0.005.
- **Reported only:** the held-out touch agreement at `conventions.TOUCH`.
- **The printout:** a short table per block and the verdict line.

- [ ] **Step 1: Write the failing tests** in `checks/selftrain.py`, on synthetic matches (`checks/common.random_match`,
  the helpers `checks/shape.py` uses for a small artifact) and on arrays:
  - `test_selftrain_settings_read_main`: `settings()` on the real `main.toml` gives the teacher (1.5, 6.0, 1, 1.0),
    ε == 0.07, the two model paths and c_team 0.76. On a temporary config without `controlled_touch` it raises with a
    message naming it.
  - `test_selftrain_head_fit_matches_scipy`: on synthetic (z, y, X), `head_fit`'s β equals
    `scipy.optimize.minimize` (L-BFGS-B, tol 1e-12) of the same penalised balanced objective within 1e-6. Its offset
    equals `noise_aware_offset(z + X̃β, y)`. With every feature column constant, β = 0 and the offset is E2's.
  - `test_selftrain_gate_on_synthetic_matches`, with `gate(..., matches={...})` on two random matches, profile `clean`,
    one seed (fast):
    - the report has every key above;
    - the base variant's log-loss equals a plain recomputation from `shape_possession.compute(..., scope="all")` on
      the gate frames within 1e-12;
    - rung 1 with every offset forced to 0 (monkeypatch `recalibration.noise_aware_offset`) equals base exactly;
    - the exit code follows the two conditions on hand-set numbers (test the decision function `gate_verdict(per_match,
      pooled)` directly).
  - `test_selftrain_never_reads_truth_through_inputs`: run `gate` on a match whose truth is swapped for its mirror
    (H ↔ A). Every teacher row count, offset and π̂ is identical; the oracles and the log-losses change.
  - `test_selftrain_report_is_aggregates_only`: walk the JSON. No list or array is longer than 64, and no key holds
    per-frame data.
  - `test_selftrain_bar_ath_band_and_tag_check`: monkeypatch the BAR-ATH inputs with a synthetic `Inputs` (as
    `checks/tag_stats` does) and the gate report's band. |b| above the band gives exit 3. The tag check's arithmetic
    matches a plain loop. Adapted at base − 0.004 passes and base − 0.006 fails.
- [ ] **Step 2: Run** `.venv/Scripts/python pipeline/benchmark/selftest.py`: the new checks fail.
- [ ] **Step 3: Implement** `selftrain.py` and register the checks. Its docstring cites the design and R027, and states
  the exit codes.
- [ ] **Step 4: Run** both suites: all pass.
- [ ] **Step 5: Commit** `git add pipeline/benchmark/selftrain.py pipeline/benchmark/checks/selftrain.py
  pipeline/benchmark/checks/__init__.py pipeline/benchmark/README.md && git commit -m "P22(b): selftrain.py, R027's
  gate on DFL (rung 1, E1, single, rung 3, oracles) and the label-free checks on BAR-ATH"`

### Task 3: the source `shape_selftrain`, its configs and `diagnose.py` (worktree `p22b-source`, in parallel with the gate)

**Files:**
- Create: `pipeline/evidence/shape_selftrain.py`, `pipeline/benchmark/configs/r027/r027_post.toml`,
  `pipeline/benchmark/configs/r027/r027_preshift.toml`
- Modify: `pipeline/evidence/__init__.py` (one REGISTRY line, alphabetical), `pipeline/evidence/README.md` (one row),
  `pipeline/benchmark/diagnose.py` (`SHAPE_KINDS = ("shape_possession", "shape_selftrain")`), `pipeline/selftest.py`,
  `pipeline/benchmark/checks/selftrain.py`, `pipeline/benchmark/checks/diagnose.py`

**Interfaces:**
- Consumes: `recalibration.*`; `shape_possession.load`, `shape_possession._logit_home`, `shape_possession.SCOPES`;
  `trees.sigmoid`, `trees.check_min_prob`, `trees.model_path`.
- Produces:
  - `compute(run, *, model, alive_model, teacher, epsilon, scope="gaps", min_prob=0.02) -> Evidence`: `alive` NaN;
    `home = clip(σ(z + b_k), min_prob, 1 − min_prob)` where P07 speaks under `scope`, NaN elsewhere; both read-only.
  - `fit(run, *, model, alive_model, teacher, epsilon) -> list[BlockOffset]`: the per-block offsets that `compute`
    applies. They are memoised per Inputs and per (model file stamp, alive model stamp, teacher, epsilon).
  - `epsilon` is a number in (0, 0.5), else a ValueError naming it.
- The configs:
  - `r027_post.toml` is `main.toml`'s text with the `shape_possession` entry replaced by the design's `shape_selftrain`
    entry. That includes `epsilon = 0.07 # 1 - controlled_touch's confidence 0.93 (R013)` and
    `alive_model = "pipeline/models/dead_ball_hgb.json"`. It has a header comment naming R027, and `description` updated.
  - `r027_preshift.toml` is the same without the `[[refine]]` block.

- [ ] **Step 1: Write the failing tests:**
  - **In `pipeline/selftest.py`:**
    - `test_shape_selftrain_parameters`: the teacher table strictness via `check_teacher`, epsilon bounds, and scope and
      min_prob as `shape_possession`'s.
    - `test_shape_selftrain_equals_p07_when_every_offset_is_zero`: on the standard synthetic inputs (two blocks of 300
      frames, so every block falls back to 0 through the floor), the output equals `shape_possession.compute` with the
      same model, scope and min_prob, bit for bit.
    - `test_shape_selftrain_applies_the_other_blocks_offset`: monkeypatch `recalibration.teacher_labels` to return
      designed (y, rows) with ≥ 500 rows a class per block. `home` equals `clip(σ(z + b_k))` with b_k computed by
      `block_offsets(..., fit=partial(noise_aware_offset, epsilon=0.07))`, within 1e-12.
    - `test_shape_selftrain_is_antisymmetric`: on a large `touch_scene`-based scene with a stump model, relabelling the
      teams gives 1 − home within 1e-9.
    - `test_shape_selftrain_registry_and_scope`: the kind is registered and not match-only. `gaps` speaks only where the
      served ball is missing.
  - **In `checks/selftrain.py`:** `test_selftrain_r027_configs_are_main_with_the_source_swapped`. `load_config` accepts
    both configs. `r027_post`'s text equals `main.toml`'s with the `[[evidence]] kind = "shape_possession"` block
    replaced by the declared block and the header and description lines removed. `r027_preshift` equals `r027_post`
    without `[[refine]]`. The teacher equals `settings().teacher`.
  - **In `checks/diagnose.py`:** `shape_selftrain` is in `SHAPE_KINDS`, and a synthetic config with it gets the
    ball-seen/missing split.
- [ ] **Step 2: Run** both suites: the new checks fail.
- [ ] **Step 3: Implement** the source, the registry line, the README row, the configs and the `SHAPE_KINDS` change.
  The source's docstring takes the config block from the design, says what the teacher is, why it is cross-fitted, why
  E2, and that no person decides any label (RULES.md).
- [ ] **Step 4: Run** both suites, then `.venv/Scripts/python pipeline/run.py pipeline/configs/main.toml --out
  C:/Users/Tian/AppData/Local/Temp/p22b_main.csv` and check the sha256 is `e6496a9b…`. That is one pipeline run of
  about 2 minutes, light enough for a task; tell the controller before starting it, and do not start it while a heavy
  run is going.
- [ ] **Step 5: Commit** `git add -A pipeline && git commit -m "P22(b): the shape_selftrain source (P07 plus the other
  half's E2 intercept), R027's configs, diagnose's SHAPE_KINDS"`

### Task 4: R027 part 1 (controller, with an independent reviewer)

**Files:** `docs/results.md` (a new `### R027` after R026, before `## Submission log`), and a scoreboard row reading
"declared".

- [ ] **4a.** Write part 1 from the design's sections 2 and 6 to 7, with Tian's answers in place:
  - the bar's form (i) or (ii);
  - automatic continuation after the gate;
  - automatic adoption.

  It also states the commands (the scripts below), the exit-code meanings, the STOPs, the retry, ADOPT and DROP, and
  "what it cannot settle": the asymmetric teacher error; `tracking`'s placement of the gaps (T38); the grain (T33); the
  rung's capacity.
- [ ] **4b.** Dispatch an independent reviewer (opus) on part 1 against the design, the code of Tasks 1 and 2 and
  R024–R026. Resolve every finding in the text, then commit `R027 part 1: declared before any run` **before** any gate
  run. Record the commit in the ledger.

### Task 5: The runs (controller; one heavy process at a time, through `heavy-p22b.ps1`, from the run worktree)

Scripts in `.superpowers/bin/`, written in the style of `r025-guard.ps1`: steps as `@(name, args)` with a 90 s
cool-down, and the chain stops at the first non-zero exit **except where an exit code is a verdict** (the gate's 1 and
4, and `bar-ath`'s 1 and 3 are verdicts, not errors). Every log ends with `EXIT=<code>`.

| script | steps |
|---|---|
| `r027-gate.ps1` | `selftrain.py gate --profile tracking --seeds 3 --data-dir <data>`, then `selftrain.py gate --profile context --seeds 3 --data-dir <data>` (report only) |
| `r027-lag.ps1` | `conventions.py latency pipeline/benchmark/configs/r027/r027_preshift.toml dfl --profile tracking --seeds 3 --truth raw --drop-match-only --data-dir <data>` |
| `r027-bar.ps1` | `run.py pipeline/configs/main.toml <cfg>/r027_final.toml <cfg>/r027_preshift.toml --profile tracking --seeds 3 --truth both --drop-match-only --data-dir <data>`, the same on `--profile context`, then `diagnose.py <cfg>/<candidate>.toml --profile tracking --seeds 3 --drop-match-only --data-dir <data>` |
| `r027-barath.ps1` | `selftrain.py bar-ath --gate-report <data>/runs/selftrain/gate-tracking.json --data-dir <data>`, then `pipeline/run.py <candidate>` twice, then `label_diff.py pipeline/configs/main.toml <candidate>` |

`<candidate>` is the bar's verdict on the shift: `r027_final`, or `r027_preshift` if the shift's guard failed.

- [ ] **5.1 The gate.** Detach the run worktree at the commit holding Tasks 1, 2 and 4, and run `r027-gate.ps1`.
  Meanwhile Task 3 runs in its worktree.
  - **Gate exit 0:** review Task 3 and merge `p22b-source` into the lane. Write and commit part 2a (the gate's tables).
    Continue at 5.2, automatically if Tian said so in question 3, else STOP for Tian.
  - **Gate exit 4:** part 2a, then R027b (Task 5b).
  - **Gate exit 1:** part 2a, then DROP (Task 6). The `p22b-source` branch and worktree are deleted unmerged.
- [ ] **5.2 The lag.** Detach the run worktree at the lane head (the source merged), and run `r027-lag.ps1`.
  - Compute s_b by R012's rule.
  - Write `r027_final.toml`: `r027_preshift` plus `[[refine]]` with that s_b, or a copy of `r027_post` if s_b is
    +1 / −2 / 0.
  - Commit `R027: the lag and r027_final`.
- [ ] **5.3 The bar.** Run `r027-bar.ps1`.
  - `main.toml` must reproduce 0.5363 / 0.5764; if not, **STOP for the advisor**.
  - The shift guard picks the candidate: `r027_final` if it beats `r027_preshift` on the objective on `tracking` and is
    not below −0.005 on `context`, else `r027_preshift`.
  - Then `choose.py --reference main --objective mean <candidate>` for the deltas, the per-match deltas and the p.
  - Apply the declared bar and the vetoes.
  - **A pass** continues at 5.4. **A fail** writes part 2b and goes to DROP, unless the advisor approves one cause-named
    retry and R027b has not run.
- [ ] **5.4 BAR-ATH.** Run `r027-barath.ps1`.
  - **`bar-ath` exit 3:** STOP for the advisor (adopt or drop, from the aggregates).
  - **Exit 1:** DROP.
  - **Exit 0:** the sha256 must be the same twice, the D share within 1 point of 36.00% and no `X` in a graded block.
    Possessions per half get a WARN beyond ±25%.
  - **All pass:** ADOPT, automatically if Tian said so in question 4, else STOP for Tian.
- [ ] **5.5** Write part 2b, with every table, in the R025 and R026 style, and commit.

**Task 5b (R027b, only after gate exit 4):** rung 3 served.
1. A part 1 for R027b, reviewed and committed.
2. A task that moves `head_fit` into `recalibration.py` and gives the source `head = true`. Its checks: equality with
   the tool's head on synthetic data, and antisymmetry.
3. Then 5.2 to 5.5 with R027b's configs (`r027b_post`, `r027b_preshift`).

Any failure is DROP.

### Task 6: ADOPT or DROP (controller)

- [ ] **ADOPT.**
  1. Copy the candidate to `pipeline/configs/r027_selftrain.toml` and make `pipeline/configs/main.toml` equal to it
     (header comments as R024 wrote them).
  2. `pipeline/run.py pipeline/configs/main.toml` twice: the same sha256. Log it in the submission log as not uploaded.
  3. The pipeline suite, the benchmark suite and `selftest.py --data` (heavy, alone) pass.
  4. Commit `R027 adopted: main.toml takes shape_selftrain (sha256 …)`.
- [ ] **DROP.**
  1. One commit removes `pipeline/evidence/recalibration.py`, `shape_selftrain.py` if merged, the registry line, the
     README rows, `pipeline/benchmark/selftrain.py`, its checks and registration, the configs, `SHAPE_KINDS`' addition
     and the selftest checks. It names the last commit that holds them.
  2. Both suites pass, and `main.toml`'s sha256 is still `e6496a9b…`.

### Task 7: Docs, final review, merge (controller)

- [ ] **The docs.**
  - `docs/proposals.md`: P22's status and its overview row.
  - `docs/results.md`: R027 (and R027b), the scoreboard and the submission log.
  - `docs/traps.md`: a new trap if something cost real time.
  - `CLAUDE.md`'s architecture sentence (adopted: the source; dropped: nothing).
  - `pipeline/README.md` and `pipeline/benchmark/README.md`.
  - `docs/technical_document.md` and `docs/technical_document_submission.md`, which must describe the self-training if
    it is adopted (the teacher, the fit, that no person decides a label; RULES.md's second condition).
  - The site (`site/progress.html`, `site/index.html`, footer).
- [ ] **The final review** on the most capable model: the whole branch against the design, R027 and this plan. One fix
  round, then adjudicate the residuals.
- [ ] **The merge.** `git merge --no-ff p22b-selftrain` into `main` after both suites (and `selftest.py --data` if
  adopted) pass on the merged tree; push to origin. Delete the worktrees and the branches. Update the handoff and the
  ledger.

## The decision flow at a glance (from the design's section 7, with the exit codes)

| event | action |
|---|---|
| gate exit 0 | merge the source, then lag and bar (automatically if Q3 is yes) |
| gate exit 4 | R027b (rung 3): part 1, then serve it, then lag, bar, BAR-ATH |
| gate exit 1 | DROP; the source branch is deleted unmerged |
| `main.toml` not reproduced in the bar run | STOP, advisor |
| bar or a veto fails | DROP (an advisor-approved cause-named retry only if R027b has not run) |
| `bar-ath` exit 3 (b outside the band) | STOP, advisor (adopt or drop) |
| `bar-ath` exit 1, a different sha256 twice, D share > 1 point off, or an `X` in a graded block | DROP |
| everything passes | ADOPT (automatically if Q4 is yes) |
| a crash, a reaped run or anything unforeseen | rerun once if the cause is clear (a reaped run is rerun only if Tian allowed it), else STOP for the advisor |

**The advisor** is an opus agent briefed with the design, R027 and the aggregates. It rules on the STOPs and the
unforeseen. Each ruling goes into the ledger as `Advisor (for Tian): <decision> — <reason>` and into the handoff for
Tian to overturn.
