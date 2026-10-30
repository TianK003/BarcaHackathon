# Task 0: benchmarking a config that has match-only sources

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.
> Read the index plan's Global Constraints first: `docs/superpowers/plans/2026-10-30-p02-p08-p09-p10-p12-index.md`.

**Goal:** a source can declare itself match-only, and the benchmark runner can drop such sources explicitly, saying so
in its printout and report. Then `main.toml` stays benchmarkable after P08 or P10 joins it.

**Spec:** design section 4, "Task 0".

**Branch:** `task0-match-only`, made from `main` in the main checkout (not a worktree). It also carries the commit of
the spec and these plans. It is merged into `main` (fast-forward) and pushed before any part branches.

## Task 0.1: commit the design and the plans

**Files:** `docs/superpowers/specs/2026-10-30-p02-p08-p09-p10-p12-design.md`, `docs/superpowers/plans/*.md`
(already written).

- [ ] **Step 1:** `git checkout -b task0-match-only`
- [ ] **Step 2:** `git add docs/superpowers && git commit -m "Design and plans for P02, P08, P09, P10 and P12"`

## Task 0.2: `MATCH_ONLY` and `match_only(kind)`

**Files:**
- Modify: `pipeline/evidence/__init__.py` (add the function below `REGISTRY`)
- Test: `pipeline/selftest.py`

**Interfaces:**
- Produces: `pipeline.evidence.match_only(kind: str) -> bool`. It is True when the module that defines
  `REGISTRY[kind]` has a module-level `MATCH_ONLY = True`, and False when the attribute is absent. It raises `KeyError`
  naming the kind when the kind is unknown. P08 and P10 set `MATCH_ONLY = True` in their modules.

- [ ] **Step 1: Write the failing test** `test_match_only_reads_the_module_flag` in `pipeline/selftest.py`:
  - every kind in today's `REGISTRY` gives `match_only(kind) is False`;
  - a module object with `MATCH_ONLY = True`, registered under a temporary kind (restore `REGISTRY` in `finally`),
    gives True;
  - an unknown kind raises `KeyError` with the kind in the message (`expect_error`).
- [ ] **Step 2:** run `$PY pipeline/selftest.py`, expect FAIL for that check.
- [ ] **Step 3:** implement `match_only(kind)` with `sys.modules[REGISTRY[kind].__module__]`.
- [ ] **Step 4:** run `$PY pipeline/selftest.py`, expect all PASS.
- [ ] **Step 5:** commit: `git commit -am "evidence: match_only(kind) reads a source's MATCH_ONLY flag"`

## Task 0.3: `--drop-match-only` in the benchmark runner and in `conventions.py latency`

**Files:**
- Modify: `pipeline/benchmark/run.py`:
  - `_load_configs` and its callers `evaluate_configs` and `evaluate_truths` gain `drop_match_only=False`;
  - `_report` records the dropped names;
  - `_unavailable`'s sentence gains a hint;
  - `main` gains `--drop-match-only`;
  - `render` prints the dropped names.
- Modify: `pipeline/benchmark/conventions.py` (the `latency` subcommand: the flag, applied after `load_config`).
- Test: `pipeline/benchmark/checks/run.py`.

**Interfaces:**
- Produces: `pipeline.benchmark.run.drop_match_only(config: Config) -> tuple[Config, tuple[str, ...]]`. It returns the
  config with every `[[evidence]]` whose kind is `match_only` removed (via `dataclasses.replace`, keeping
  `config.text` and `config.path` unchanged), plus the removed sources' names in config order.
- Report field: `report["dropped_match_only"]` is a list of names, `[]` when the flag is off or nothing was dropped.
  It is written under the existing `config` key's sibling level, so reports from before the flag stay readable:
  `choose.py` treats a missing key as `[]`.
- `_unavailable`'s sentence ends with: ` Run with --drop-match-only to benchmark the config without its match-only
  sources (they are listed in the report).`

- [ ] **Step 1: Write the failing tests** in `pipeline/benchmark/checks/run.py`, synthetic, following
  `test_runner_end_to_end_synthetic`'s harness:
  - `test_drop_match_only_removes_exactly_the_flagged_sources`: a config with two sources, one of a temporary kind
    whose module has `MATCH_ONLY = True`. `drop_match_only` returns the other source alone, `("<its name>",)` as
    dropped, and the same `text` and `path`. A config with no flagged source comes back with `dropped == ()` and its
    evidence unchanged.
  - `test_runner_without_the_flag_names_it`: a temporary match-only source that raises `AssetUnavailable` makes
    `evaluate_truths` raise `BenchmarkError` whose message contains `--drop-match-only`.
  - `test_runner_with_the_flag_records_the_dropped`: the same config with `drop_match_only=True` runs, the report has
    `dropped_match_only == ["<name>"]`, and the reference config's report has `[]`.
- [ ] **Step 2:** run `$PY pipeline/benchmark/selftest.py`, expect the three to FAIL.
- [ ] **Step 3:** implement as in Interfaces. In `main`, add
  `parser.add_argument("--drop-match-only", action="store_true", help="drop the evidence sources that need an asset a
  benchmark match does not have (MATCH_ONLY), and record them in the report")`. Do the same on `conventions.py`'s
  `latency` parser.
- [ ] **Step 4:** run both selftests, expect all PASS.
- [ ] **Step 5:** add one paragraph to `pipeline/benchmark/README.md` ("Running a config") on the flag, and one
  sentence to `pipeline/evidence/README.md` ("Writing a source") on `MATCH_ONLY`.
- [ ] **Step 6:** commit: `git commit -am "benchmark: --drop-match-only for sources a DFL match cannot serve"`

## Task 0.4: `label_diff`, how a config changes BAR-ATH's labels (aggregates only)

P08's and P10's strength reviews (design 5.C and 5.D) and P12's and P09's label-free checks all need the same
aggregates, so they are built once here.

**Files:**
- Create: `pipeline/benchmark/label_diff.py`
- Test: `pipeline/benchmark/checks/label_diff.py` (register it the way the other `checks/*.py` modules are collected
  by `pipeline/benchmark/selftest.py`)

**Interfaces:**
- Produces: `diff(reference_codes, codes, inputs) -> dict`, with both arrays the full match's int8 codes. Over the
  graded frames it gives:
  - `{"d_share": [ref, new], "x_share": [ref, new], "possessions": {half: [ref, new]}}`;
  - `"changed_share"`: the share of graded frames whose label differs;
  - `"changed_in_gaps_share"`: the share of the changed frames where `inputs.atd.ball_xy` is NaN;
  - `"d_onsets": {"ref", "new", "moved", "median_move"}`. A D onset is the first frame of a D run inside a block. An
    onset moved if the nearest new onset within ±50 frames is at a different frame. `median_move` is in frames,
    signed new − ref, and `None` if none moved.
- CLI: `$PY pipeline/benchmark/label_diff.py REFERENCE.toml CONFIG.toml [CONFIG.toml ...]`.
  - It runs each config on BAR-ATH through `pipeline.run`'s own functions (no CSV written) and prints the aggregates.
  - It prints no frame number and no per-frame value.
  - It is HEAVY: the controller runs it.

- [ ] **Step 1: Write the failing tests:**
  - `test_label_diff_on_hand_built_codes`: two short code arrays with known differences in a synthetic `Inputs` give
    the exact shares, the possessions per half, and the moved onsets with their median move.
  - `test_label_diff_prints_aggregates_only`: the rendered lines contain no frame index of the scene.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement. **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit `"benchmark: label_diff, a config's change to BAR-ATH's labels in aggregates"`

## Task 0.5: merge and push

- [ ] **Step 1:** `git checkout main && git merge --ff-only task0-match-only && git push origin main`
- [ ] **Step 2:** `git push origin task0-match-only`
