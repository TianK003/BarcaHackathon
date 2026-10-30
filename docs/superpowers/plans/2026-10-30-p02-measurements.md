# Part A, P02: descriptors around each change of the truth (A15)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.
> Read the index plan's Global Constraints first: `docs/superpowers/plans/2026-10-30-p02-p08-p09-p10-p12-index.md`.

**Goal:** at every change of the truth, on the public window and on DFL, measure what the tracking shows, and report
aggregates that a person (and P16) can compare.

**Spec:** design section 5.A.

**Branch / worktree:** `git worktree add ../BarcaHackathon-p02 -b p02-measurements main` (after Task 0 is on `main`).

**Architecture:**
- The descriptors are a new layer in `pipeline/benchmark/conventions.py`, beside the detectors and the lags.
- They are computed by `describe(inputs, changes)` from `Inputs` alone (no truth: the change frames are the only input
  from the truth, exactly as for the lags).
- They are summarised by type, added to the `public` and `dfl` documents, and compared by `compare`.
- The person's CSV gains columns.

## Task A.1: `describe` and its summary

**Files:**
- Modify: `pipeline/benchmark/conventions.py`
- Test: `pipeline/benchmark/checks/conventions.py`

**Interfaces:**
- Consumes: `transitions_of(codes, runs) -> [(frame, from, to, type)]`, the `kinematics` tracks
  (`ball_track(inputs)`, `player_track(inputs)`, `outside_m`), and `inputs.atd.slot_team`.
- Produces:
  - `DESCRIPTORS = ("ball_seen", "ball_seen_1s", "ball_speed", "line_m", "old_near_m", "new_near_m", "new_nearer")`.
  - `describe(inputs, changes) -> list[dict]`: one dict per change, keyed by `DESCRIPTORS`. A value is `None` where
    the descriptor does not apply or cannot be computed. `old_near_m`, `new_near_m` and `new_nearer` apply to `team`
    changes only, and need the ball seen on the change frame.
  - `summarise_descriptors(changes, rows) -> {type: {descriptor: {"n", "share" | "median", "p25", "p75"}}}`. Boolean
    descriptors give `share`, numeric ones `median` and quartiles. `n` counts the rows where the value is not `None`.

**Definitions** (frame c = the change's frame, the first frame of the new label):
- `ball_seen`: ball position present on c.
- `ball_seen_1s`: present on any frame of [c − 25, c + 25] inside the change's block.
- `ball_speed`: the `kinematics.ball_track` speed on c, in m/s (from positions, never the ATD's channel, T31).
- `line_m`: `kinematics.outside_m` on c (signed metres beyond the nearest line; negative inside).
- `old_near_m` / `new_near_m`: the distance from the ball to the nearest present slot of the old / new team, in
  metres, by `slot_team`.
- `new_nearer`: `new_near_m < old_near_m`.

- [ ] **Step 1: Write the failing tests** (synthetic, on the existing scene helpers in `checks/conventions.py`):
  - `test_conventions_descriptors_on_a_hand_built_scene`:
    - a scene with the ball at a known point and one slot of each team at known distances;
    - a `team` change at c gives exact `old_near_m`, `new_near_m`, `new_nearer`, `ball_speed` and `line_m`;
    - an `into_dead` change gives `None` for the three team descriptors;
    - a change on a frame with no ball gives `ball_seen False`, `ball_speed None`, `line_m None`, and `ball_seen_1s`
      True when the ball is seen 20 frames later;
    - `ball_seen_1s` never reads across the block's edge.
  - `test_conventions_descriptor_summary`: `summarise_descriptors` gives `share` for booleans and median and quartiles
    for numbers, per type, with `n` excluding `None`.
- [ ] **Step 2:** `$PY pipeline/benchmark/selftest.py`, expect FAIL.
- [ ] **Step 3:** implement `describe` and `summarise_descriptors`.
- [ ] **Step 4:** `$PY pipeline/benchmark/selftest.py`, expect PASS.
- [ ] **Step 5:** commit `"conventions: descriptors of the tracking at each change of the truth (P02)"`, then push.

## Task A.2: into the documents, the person's CSV, `compare` and the printout

**Files:**
- Modify: `pipeline/benchmark/conventions.py`: `public`, `_write_transitions`, `measure` / `dfl_document`,
  `compare`, `render_public`, `render_dfl`.
- Test: `pipeline/benchmark/checks/conventions.py`.

**Interfaces:**
- Produces:
  - Document key `"descriptors"`: `summarise_descriptors`' output, in both the public and the DFL document. DFL pools
    the rows of every match and seed before summarising (pool by rows, never a median of medians, the file's rule).
  - `compare`'s result gains `"descriptor_gaps"`:
    `{type: {descriptor: {"public", "dfl", "gap", "low", "high"}}}`.
    - The gap is public minus DFL: of the share for booleans, of the median for numbers.
    - It comes with a 90% bootstrap interval from `bootstrap_gap`'s procedure (`BOOTSTRAP_DRAWS`, `BOOTSTRAP_SEED`)
      over the rows.
    - The bounds are hidden below `MIN_SHOWN` public rows, as for lags.
- The CSV for the person gains the `DESCRIPTORS` columns after the detectors' (blank for `None`). Still gitignored,
  still read by no LLM.
- The printouts add one line per (type, descriptor) with n, and the share or the median and quartiles. Quartiles are
  hidden below `MIN_SHOWN`, as `_summary_lines` does.

- [ ] **Step 1: Write the failing tests:**
  - `test_conventions_public_prints_aggregates_only` (extend the existing check): the new lines carry no frame number
    and no per-change value. The CSV header has the descriptor columns.
  - `test_conventions_compare_descriptor_gaps`: two hand-made documents give the expected gap and hide the bounds of a
    small cell.
  - `test_conventions_dfl_pools_descriptors_by_rows`: two matches' rows pooled equal one summary of their union.
- [ ] **Step 2:** run, expect FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** both selftests PASS.
- [ ] **Step 5:** update `pipeline/benchmark/README.md` ("Conventions") and the module docstring: what the descriptors
  are, and that they are inspection only.
- [ ] **Step 6:** commit `"conventions: descriptors in the documents, the person's file and compare (P02)"`, then push.

## Task A.3: the runs and A15 (HEAVY steps go to the controller)

- [ ] **Step 1 (HEAVY):** `$PY pipeline/benchmark/conventions.py public`
- [ ] **Step 2 (HEAVY):** `$PY pipeline/benchmark/conventions.py dfl --truth raw`, then `--truth coarse`
- [ ] **Step 3 (light):** `$PY pipeline/benchmark/conventions.py compare --truth raw` (and coarse)
- [ ] **Step 4:** write **A15** in `docs/results.md`, after A14: "Descriptors of our truth's changes against DFL's
  (2026-10-30, Tian)". It holds:
  - the commands and the commit;
  - per type, the descriptor table, public against DFL raw and coarse, with gaps and intervals (aggregates only);
  - what they suggest about the convention;
  - the sentence: "Inspection only (RULES.md, `f8f2629`, T27): nothing is set from these numbers; a timing constant
    comes from other games (P16)."
- [ ] **Step 5:** in `docs/proposals.md`, set P02's status to "done → A11, R012, A15 (inspection only)", in its
  section and in the overview row.
- [ ] **Step 6:** commit `"A15: descriptors of the truth's changes, public against DFL (P02 done)"`, then push.
