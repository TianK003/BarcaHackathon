# P22(a): an EM label-shift prior for the shape model (R020); then P21 step 0 (A16)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Give `shape_possession` an optional per-match prior. The prior is the home share π̂ that Saerens' EM estimates from
  the model's own outputs, added to the logit as s·logit(π̂).
- Check first, on the seven DFL matches, that π̂ beats guessing 0.5. If it does, choose s from {0.5, 1.0} by the house
  rule and adopt it only if R014's bar passes and the BAR-ATH checks hold.
- Then measure, on BAR-ATH, whether the wrong-team slots (T32) cluster at the ball (P21 step 0), with the advisor's
  ratio rule.

**Architecture:**
- **The prior.** `prior` and `prior_strength` on `shape_possession`, off by default (`prior = "none"`). The logit path
  of every existing config is untouched, so they write the same bytes.
- **The tools.** `pipeline/benchmark/label_shift.py` measures the feasibility on DFL and runs the BAR-ATH checks.
  `pipeline/benchmark/wrong_team.py` is P21 step 0; it stays after this lane, because P19's transfer check reports its
  ratio.
- **The grid.** Two configs, each `main.toml` plus the two keys.
- **No** contract change, new dependency, retraining or decoder change.

**Tech stack:** Python 3.11.9, numpy (the pipeline); the benchmark's pins. No new import anywhere.

**Spec (read all three before starting):**
- `.superpowers/sdd/run-2026-10-31/advisor-01.md`, §1 lane C, §2 "Lane C" and §3. These decisions are binding: they
  were made for Tian, who is away overnight.
- `.superpowers/sdd/2026-10-31-p22-label-shift/research.md`, the literature and the proposed deviations D1 to D4. Both
  files live in the main checkout, `C:/Users/Tian/Desktop/BarcaHackathon/`.
- `docs/proposals.md`, "### P22" and "### P21".

Also read before starting:
- `docs/results.md`: R011 (P07, the model and `touch_agreement`), R014 (the bar this lane reuses) and R017 (the latest
  part 1 / part 2 entry, for its form);
- `docs/traps.md`: T01, T26, T27, T32, T33, T38, T41 and T42.

**Branch / worktree:** `p22-label-shift` at `C:/Users/Tian/Desktop/BarcaHackathon-wt/p22-label-shift`, created from
`0716640`.

**Entries:**
- R020 is P22(a).
- A16 is P21 step 0. If another lane has taken A16 by the time you write it, take the next free A and say so in the
  ledger.
- R021 and R022 belong to the P19 and P20 lanes; do not use them.

**Deviations pending the advisor** (research.md section 9). The controller sends them to the advisor before Task 2
starts.
- **D1:** the feasibility diagnostics.
- **D2:** not tonight. Only the wording of R020 part 1 reflects it.
- **D3:** P21's onset ratio.
- **D4:** feasibility as its own heavy run.

Steps marked **[D1]** or **[D3]** are built only if the advisor accepts that deviation; otherwise skip them. The plan
follows D4. If the advisor rejects D4, the controller queues Tasks 6 and 7 back to back and reads Task 6's output
before any of Task 7's; nothing else changes.

## Global constraints (every task's requirements include these)

- **No match data to an LLM, including you.**
  - Never Read, cat, head or grep anything under `data/`.
  - Never print rows of tracking, per-frame outputs of a source on BAR-ATH, or labels of its frames.
  - Print, write and commit aggregates only: counts, shares, π̂, medians, hashes, pass/fail.
  - `evaluation/gt/` and `evaluation/results/baseline_*.csv` are public and may be read.
- **Never commit** match data or per-frame derived data. Run outputs go under `data/runs/`, caches under `data/cache/`
  (both gitignored).
- **Don't modify `evaluation/`.**
- **The public window chooses nothing.** It may be scored only as a correctness check after a decision (T01, T27).
  There are no hand labels of this match.
- **The share is never taken from outside the bundle.** Published possession statistics of this match are
  human-produced data about it (RULES.md rule 2, T26). π̂ comes from EM on our own model's outputs, and from nothing
  else.
- **Run everything with the main checkout's venv**, with the data directory pointed at the main checkout:

  ```bash
  export MTAH_DATA_DIR=C:/Users/Tian/Desktop/BarcaHackathon/data
  PY=C:/Users/Tian/Desktop/BarcaHackathon/.venv/Scripts/python
  $PY pipeline/selftest.py                 # synthetic, seconds: run after every change
  $PY pipeline/benchmark/selftest.py       # synthetic benchmark checks, seconds
  ```

- **HEAVY steps are not yours to start.** Write the exact command in your report and stop at that step.
  - The controller runs each one alone (one heavy process at a time, on a crash-prone CPU laptop: T41) through:

    ```powershell
    pwsh -File C:/Users/Tian/Desktop/BarcaHackathon/.superpowers/bin/heavy.ps1 -Name <name> `
         -WorkDir C:/Users/Tian/Desktop/BarcaHackathon-wt/p22-label-shift -PyArgs "<script.py args>"
    ```

    The script sets one BLAS thread, below-normal priority, a free-memory guard and the data directory. The controller
    waits 90 s between runs.
  - HEAVY means: `pipeline/run.py` on the real match; `selftest.py --data`; `benchmark/selftest.py --dfl` or `--data`;
    `benchmark/run.py`; `label_shift.py`; `wrong_team.py`; `label_diff.py`; anything that loads BAR-ATH or DFL data.
  - No new heavy work starts after 08:30. Nothing may be half-adopted at 08:30: either the adoption commit is in, or
    `main.toml` is untouched.
- **The references must reproduce to the last digit, or the lane stops.** On the benchmark: `main` 0.5430 raw and
  0.5593 coarse; `nearest_w50` 0.2018 raw; `nearest_w125` 0.2226 coarse. Every benchmark run of `main.toml` uses
  `--profile tracking --seeds 3 --truth both --drop-match-only` (smart_tagging is match-only).
- **`main.toml`: one change at a time.** P22 may merge only before P19's grid configs have been generated. After that,
  `main.toml` is frozen until P19's verdict. The controller confirms which holds before Task 9.
- **Thresholds are mechanical.** A miss by any margin is a miss (A14's -0.0003).
- **One retry.** It is allowed only with a cause named from logged aggregates, a change aimed at that cause, its own
  part 1 (R020b) and the same bar. More values on the s axis are not a retry. After a second failure: drop, remove the
  code, and record the commit to recover it from.
- **A declaration (part 1) is committed and pushed before any run it governs, and is never edited after.** Part 2
  applies it as written.
- **Commits.**
  - One per tested step. Messages are short and imperative, like `git log --oneline -20`.
  - No AI attribution and no `Claude-Session` trailer (managed policy). Never a gendered pronoun for Tian.
  - Push after each commit: `git push -u origin p22-label-shift`. Check `git remote -v` first: `origin` must be
    `TianK003/BarcaHackathon`.
- **Style.** Match the neighbours:
  - module docstrings that argue the why, and a `# Params` block for constants;
  - keyword-only parameters validated with a sentence that names the parameter;
  - plain `test_*` functions with `assert ..., "message"` and `expect_error(...)`;
  - `np.random.default_rng([seed, <a constant>])` for anything random in a test (T22).

## Review focus

1. **Byte identity.**
   - With `prior` absent or `"none"`, `compute` must run exactly today's arithmetic: no `+ 0.0 * ...` and no reordered
     clip.
   - `main.toml` must keep writing sha256 `7ba575c0...`, and `smooth50_repro.toml` the baseline byte for byte
     (Task 4).
2. **What EM reads.**
   - It reads the memoised, symmetrised, unclipped logit over every graded frame with an opinion.
   - It is the same for both scopes and every `min_prob`.
   - It never reads the clipped `home`.
3. **Fold choice.**
   - On DFL, EM runs on the fold of `inputs.game`, never on `full`.
   - `label_shift.feasibility` refuses a model file that lacks a match's fold, as `train_shape.report` does.
4. **No truth through the inputs.** π̂ comes from the `MatchInputs` alone; π_true comes from `truth_codes` read beside
   them. Reuse `checks/common.leaks`.
5. **Antisymmetry after the shift.** Relabelling the teams must still give 1 - home to 1e-12 with `prior = "em"`.
6. **EM at a boundary.** A π̂ of exactly 0 or 1 raises a sentence; no infinite shift is ever added. No frame with an
   opinion means π̂ = 0.5 and no shift.
7. **Config drift.** `r020_s050.toml` and `r020_s100.toml` are `main.toml` plus the two keys and nothing else
   (checked).
8. **Aggregates only.** π̂ on BAR-ATH is an aggregate and may be printed. No array of frames is printed or written.
9. **P21.**
   - The candidate frames are the touch rule *without* its duel test (the advisor's wording).
   - A team is "over 11" on its present slots.
   - Only graded frames count.
   - The rule is `ratio < 1.5`, strict.
10. **The order.** Part 1's commit hash precedes the feasibility run's start time in its log. The feasibility verdict
    is read before the grid runs (D4).

## Task 0: Setup

- [ ] **Step 1:** bring the branch up to date with `main`, which now holds `site/progress.html`:
  `git fetch origin && git merge --no-edit origin/main`. Expect no conflict: the branch has only this plan. Then push.
- [ ] **Step 2:** run both synthetic suites. Expect PASS, and record the check counts.
- [ ] **Step 3:** record the two references in the SDD ledger
  (`C:/Users/Tian/Desktop/BarcaHackathon/.superpowers/sdd/2026-10-31-p22-label-shift/progress.md`):
  - `git show HEAD:pipeline/configs/main.toml | sha256sum`, the config;
  - the CSV sha256 from the submission log, `7ba575c0982aacb9765b39c406e4ff5c9ddb82ca37cfa2b7f101417431fc3d7c`.

## Task 1: the prior on `shape_possession`

**Files:**
- Modify: `pipeline/evidence/shape_possession.py`. Rewrite step 4 of the module docstring ("No prior correction") and
  add the config keys to the docstring's example.
- Modify: `pipeline/evidence/README.md`, the `shape_possession` row: the two keys and one sentence.
- Test: `pipeline/selftest.py`, in the shape-possession section, beside `test_shape_possession_*`.

**Interfaces.** Everything below is produced by this task.

`# Params`:
- `PRIORS = ("none", "em")`;
- `EM_TOLERANCE = 1e-6`, stop when |Δπ| falls below this (advisor-01);
- `EM_MAX_ITERATIONS = 200`.

`logit_home(inputs, model) -> np.ndarray`:
- The public form of today's `_logit_home`, with the path resolved and the fold chosen exactly as `compute` does
  (`folds[inputs.game]`, else `full`).
- It returns the memoised read-only (n,) array: (r(x) - r(swap(x))) / 2, NaN outside the graded blocks and where
  nobody is present.
- `compute` calls it, so there is one code path.

`em_prior(logits, *, tolerance=EM_TOLERANCE, max_iterations=EM_MAX_ITERATIONS) -> {"prior_home", "iterations",
"converged", "frames"}`:
- Saerens' EM with the training prior 0.5.
- Over the finite entries z: start π = 0.5, then repeat π_new = mean(sigmoid(z + logit(π))) (`trees.sigmoid`, float64)
  until |π_new - π| < tolerance or `max_iterations`. `iterations` counts the updates.
- No finite entry: `{"prior_home": 0.5, "iterations": 0, "converged": True, "frames": 0}`.
- If π leaves (0, 1) open, raise a `ValueError` that says a share of 0 or 1 cannot be shifted by and names
  `shape_possession` and `prior`.

`compute(run, *, model, scope="gaps", min_prob=0.02, prior="none", prior_strength=None)`:
- `prior` must be a str in `PRIORS`, else `ValueError` naming `prior`, "none" and "em".
- With `"em"`, `prior_strength` is required: a real number in (0, 1] that is not a bool, NaN or inf. Above 1 would
  shift past Bayes' posterior.
- With `"none"`, `prior_strength` must be `None`, else `ValueError`: "prior_strength is read only with prior = "em"".
- `"none"`: today's code, unchanged.
- `"em"`: z = `logit_home(...)`, π̂ = `em_prior(z)["prior_home"]`, z' = z + s·logit(π̂) (a new array), then today's
  scope mask, sigmoid and clip on z'.

The new docstring paragraph says:
- s = 1 is the frame-level Bayes posterior (Saerens); the decoder counts a per-frame prior once a frame, scaled by the
  weight (research.md section 4).
- EM's sample is every graded frame with an opinion, whatever the scope.
- π̂ is never taken from outside the bundle (T26).

- [ ] **Step 1: write the failing tests.** Every one runs on synthetic data, with the seeds fixed.
  - `test_shape_possession_em_prior_recovers_a_known_share`:
    - Setup: `rng = np.random.default_rng([22, 1])`, n = 200,000. For π_true in (0.3, 0.5, 0.62, 0.7), draw y ~
      Bernoulli(π_true) and x ~ N(+1 if y else -1, 1); z = 2x is the calibrated logit under a balanced prior.
    - Expect `em_prior(z)` within 0.01 of the sample's share of y, and converged.
    - Teeth: for π_true ≠ 0.5, mean(sigmoid(z)) is more than 0.02 from the share (EM is not the plain average).
  - `test_shape_possession_em_prior_edge_cases`:
    - An all-NaN array gives (0.5, 0 iterations).
    - All zeros give exactly 0.5 after 1 iteration.
    - Inserting NaNs changes nothing (bitwise).
    - `em_prior(-z)` is 1 - `em_prior(z)` to 1e-12.
    - Two calls are bitwise equal.
    - Underconfidence: z = 1e-4 × (+1 on 90% of entries, -1 on the rest) reaches 200 iterations with
      `converged False` and π still moving away from 0.5. This documents research.md section 2's unsafe side.
    - z = 50 everywhere raises `ValueError` mentioning "prior".
  - `test_shape_possession_prior_none_is_byte_identical`: over `shape_scene` seeds (0, 3, 5), scopes "gaps" and "all",
    and `min_prob` 0.02 and 0.3, `compute` without the keys and with `prior="none"` give
    `home.tobytes()` equal, NaNs included.
  - `test_shape_possession_prior_em_shifts_the_logit`:
    - Build a scene where the home block is ahead on most graded frames (move the home players up the pitch), and
      assert the precondition |π̂ - 0.5| > 0.1.
    - With the stump model, expected home = clip(sigmoid(z + s·logit(π̂)), min_prob, 1 - min_prob), where z is worked
      out from `stump_home`'s logic and π̂ from `em_prior` on z's graded opinion frames. Check s ∈ (0.5, 1.0) and both
      scopes to 1e-12.
    - π̂ is the same for both scopes and for `min_prob` 0.02 and 0.4. Record `em_prior`'s input by wrapping the module
      attribute, and assert it equals `logit_home(inputs, path)` minus its NaNs, which is unclipped.
  - `test_shape_possession_prior_keeps_antisymmetry`: with the lopsided three-tree model of
    `test_shape_possession_is_antisymmetric`, `prior="em"`, s = 1, relabelling the teams gives 1 - home to 1e-12.
  - `test_shape_possession_prior_parameters`:
    - Refused `prior` values: "EM", "", None, True, 1.
    - Refused with `"em"`: no strength; strengths 0, -0.5, 1.5, NaN, inf, True and "0.5".
    - Refused with `"none"`: a strength of 0.5.
    - Accepted: `np.float64(0.5)`.
    - Messages name `shape_possession` and `prior` or `prior_strength`.
  - `test_shape_possession_prior_walks_the_trees_once`: as `test_shape_possession_remembers_its_scores_per_inputs`, a
    config with `prior="em"` after one without adds no tree walk.
  - In `test_run_end_to_end_synthetic_shape_possession`: the trial config with `prior = "em"` and
    `prior_strength = 0.5` runs config → registry → hmm → CSV. A config with `prior_strength` and no `prior` raises
    `ValueError` naming `prior_strength`.
- [ ] **Step 2:** run `$PY pipeline/selftest.py`. Expect the new tests to FAIL.
- [ ] **Step 3:** implement, numpy only. `compute`'s `"none"` path must keep today's statements verbatim.
- [ ] **Step 4:** run `$PY pipeline/selftest.py`, then `$PY pipeline/benchmark/selftest.py`. Expect PASS.
- [ ] **Step 5:** update the docstring and the README row. Commit
  `"evidence: shape_possession's optional EM label-shift prior (P22a)"` and push.

## Task 2: `label_shift.py`, the feasibility on DFL and the checks on BAR-ATH

**Files:**
- Create: `pipeline/benchmark/label_shift.py`.
- Modify: `pipeline/benchmark/train_shape.py`: `touch_agreement(inputs, model=..., **shape_values)` passes `prior` and
  `prior_strength` to `shape_possession.compute`. The default call is unchanged.
- Create: `pipeline/benchmark/checks/label_shift.py`. Modify `pipeline/benchmark/checks/__init__.py`: add it to
  `MODULES` and its topic to the docstring's list.
- Test: `pipeline/benchmark/checks/label_shift.py`, plus one check in `checks/shape.py` for `touch_agreement`.

**Interfaces.** Everything below is produced by this task.

`# Params`:
- `GATE_FACTOR = 0.75`, the advisor's: the mean error must be at most this × the mean error of guessing 0.5;
- `BAND = (0.39, 0.68)`, the advisor's BAR-ATH band (outside it, sign-off with `label_diff` aggregates);
- `TOUCH_SLACK = 0.005`, shifted `touch_agreement` must be ≥ unshifted - this;
- [D1] `SLOPE_BAND = (0.8, 1.25)`, the band outside which the slope names miscalibration, for the retry;
- `OUT_DIR = ("runs", "label_shift")`.

Functions:
- `home_share(codes, graded) -> float | None`: the share of H among the H and A codes inside the graded blocks (π_true);
  None with no alive frame.
- `gate(rows) -> {"error", "baseline", "ratio", "factor", "passes"}`: pure.
  - `rows` is `[{"pi_true": float, "pi_hat": [float per seed]}]`, one per match.
  - error = the mean over matches of (the mean over seeds of |π̂ - π_true|), which is the mean over all match-seeds
    when every match has the same seeds.
  - baseline = the mean over matches of |0.5 - π_true|; ratio = error / baseline (None if the baseline is 0).
  - `passes` = error <= 0.75 × baseline.
- [D1] `calibration_slope(z, y) -> float`: the slope-only logistic fit of y ∈ {0, 1} on z. Newton from a = 1, stopping
  at |Δa| < 1e-10 or 50 steps, deterministic. No intercept: on mirrored rows it is 0 (research.md section 2).
- `feasibility(model=DEFAULT_MODEL_PATH, *, profile=run.DEFAULT_PROFILE, seeds=run.DEFAULT_SEEDS, matches=None,
  calibration=None, data_dir=None, log=None) -> dict`:
  - The loop is `train_shape.report`'s: one match at a time, `truths.truth_codes(match)` beside the inputs,
    `served_inputs(match, profile, seed, calibration, data_dir)` per seed. A missing fold in the artifact is a
    `BenchmarkError`.
  - Per seed: `em_prior(shape_possession.logit_home(inputs, path))`.
  - Returns `{profile, seeds, model, per_match: {id: {pi_true, alive_frames, seeds: [{pi_hat, iterations, converged,
    frames}], abs_error, abs_baseline}}, gate: gate(...)}`.
  - [D1] per match also: `slope`, from z and the truth on the truth's alive opinion frames, all seeds pooled; and
    `pi_hat_alive`, the mean over seeds of `em_prior` on z masked to the truth's alive frames. Report only; the gate
    never reads them.
- `bar_ath(strength, *, model=DEFAULT_MODEL_PATH, data_dir=None) -> dict`:
  - On `Inputs(data_dir=data_dir)`, the `full` model.
  - Returns `{prior_home, iterations, converged, frames, band, inside_band, touch: {unshifted: {frames, agreement},
    shifted: {frames, agreement}, difference, passes}}`.
  - The touch figures are `train_shape.touch_agreement(inputs, model)`, and the same with `prior="em",
    prior_strength=strength`.
- `render_feasibility(result)` and `render_bar_ath(result)`: the printed lines, aggregates only.
- CLI, HEAVY, with `--out` defaulting to `<data dir>/runs/label_shift/feasibility.json` or `bar_ath.json`. The JSON is
  LF, sorted keys, rounded to 9 decimals as `run.py` does. Exit 0 whatever the verdict, 1 on a `BenchmarkError`:

  ```bash
  $PY pipeline/benchmark/label_shift.py feasibility [--profile tracking] [--seeds 3] [--model PATH] [--data-dir D] [--out PATH]
  $PY pipeline/benchmark/label_shift.py bar-ath --strength S [--model PATH] [--data-dir D] [--out PATH]
  ```

- [ ] **Step 1: write the failing checks.**
  - `test_label_shift_home_share_exact`: hand-built codes and two graded blocks, with D, X and frames outside the
    blocks, give the exact share; no alive frame gives None.
  - `test_label_shift_gate_is_the_advisors_rule`: hand rows with known errors. At error = 0.75 × baseline exactly it
    passes; at 0.75 × baseline + 1e-12 it fails. The mean is over matches first, then seeds.
  - `test_label_shift_feasibility_reads_each_matchs_fold_and_no_truth`:
    - On `checks/shape.py`'s `planted()` matches and artifact, each match's `pi_hat` equals `em_prior` on that fold's
      `logit_home`, which differs from `full`'s (assert it differs, so the check has teeth).
    - `leaks` shows no truth inside the inputs.
    - An artifact with one fold removed raises `BenchmarkError` naming the match.
  - `test_label_shift_feasibility_is_aggregates`: the result round-trips through `strict_json`, and no list in it is
    longer than the seeds.
  - [D1] `test_label_shift_calibration_slope`: z = c × (calibrated logit) on synthetic data, for c ∈ (0.5, 2), gives a
    slope within 0.03 of 1 / c.
  - `test_touch_agreement_prior_none_is_unchanged` (in `checks/shape.py`): on the hand-built scene of
    `test_shape_touch_agreement_counts_kept_frames`, the default call and `prior="none"` return equal dicts, and
    `prior="em", prior_strength=1.0` runs.
  - `test_label_shift_bar_ath_on_synthetic_inputs`: `bar_ath` on a synthetic `Inputs`, built through
    `pipeline/selftest.py`'s scene helpers with a stub `full` model. It gives the keys above, `inside_band` matches
    `BAND`, and `passes` matches the slack.
  - `test_label_shift_prints_aggregates_only`: the rendered lines hold no sequence of more than the seeds' numbers.
- [ ] **Step 2:** run `$PY pipeline/benchmark/selftest.py`. Expect FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run both synthetic suites. Expect PASS.
- [ ] **Step 5:** commit `"label_shift: P22(a)'s feasibility on DFL and its checks on BAR-ATH"` and push.

## Task 3: R020's two configs

**Files:**
- Create: `pipeline/benchmark/configs/r020/r020_s050.toml` and `r020_s100.toml`.
- Test: `pipeline/benchmark/checks/label_shift.py`.

Each config is `pipeline/configs/main.toml` verbatim, with a header that says what it is (R020's candidate at s), plus
two keys in the `shape_possession` entry: `prior = "em"` and `prior_strength = 0.5` (or `1.0`). Their `smart_tagging`
entries stay; the benchmark drops them with `--drop-match-only`.

- [ ] **Step 1:** write `test_label_shift_r020_configs_are_main_plus_the_prior`. `load_config` on `main.toml` and on
  each config gives the same seed, scope, decoder, refiners, evidence names, kinds and weights. Every evidence
  parameter is equal, except that `shape_possession` has exactly `prior` "em" and `prior_strength` 0.5 or 1.0 in
  addition. Run it and expect FAIL.
- [ ] **Step 2:** write the two files. Run both synthetic suites and expect PASS.
- [ ] **Step 3:** commit `"R020's two configs: main.toml plus the EM prior at s 0.5 and 1.0"` and push.

## Task 4: the existing configs write the same bytes on the real match (HEAVY steps go to the controller)

- [ ] **Step 1 (HEAVY):** `pipeline/run.py pipeline/configs/main.toml`. Its sha256 must be the submission log's
  `7ba575c0...`, and `evaluation/validate_submission.py evaluation/results/main.csv` must pass.
- [ ] **Step 2 (HEAVY):** `pipeline/selftest.py --data`. It runs `main.toml` twice and rebuilds the four baselines byte
  for byte, `smooth50_repro.toml` included. Expect PASS.
- [ ] **Step 3:** if either differs, STOP: Task 1 touched the default path; fix it there. Record both results in the
  SDD ledger (no commit needed).

## Task 5: R020 part 1, declared before any run it governs

- [ ] **Step 1:** write `### R020 — P22(a): an EM label-shift prior on the shape model, its feasibility, the grid and
  the bar (2026-10-31, <who>)` in `docs/results.md`, after the last R entry and before `## Submission log`. Start
  "Proposal: P22 · Commit: part 1 (this declaration, the source's prior, `label_shift.py` and the two configs) is
  committed before any run it governs". Then "**Part 1, declared before any run**" and the paragraphs below, written
  in the house style. Nothing in them is chosen from a score, and no number of this entry has been seen.
  - **Why.**
    - The shape model was trained on seven DFL matches, mirrored, so its prior is exactly 0.5.
    - Our match's home share is unknown. A per-match share of possession is a class prior, and the model's own
      outputs estimate it without labels (Saerens, Latinne and Decaestecker 2002).
    - The DFL matches' home shares of alive training rows span 43.9% to 62.9% (R011), so there is a shift to correct.
    - Never from published statistics (rule 2, T26).
    - Cite research.md's sources as links, not the file.
  - **The estimator.**
    - Per served match and seed (an `Inputs`), Saerens' EM runs on the symmetrised, unclipped logit z over every
      graded frame with an opinion, whatever the scope.
    - Training prior 0.5, start 0.5, update π ← mean(σ(z + logit π)), stop at |Δπ| < 1e-6 or 200 iterations.
    - Then home = clip(σ(z + s·logit(π̂)), 0.02, 0.98) where the scope speaks.
    - On DFL each match is read by its fold, BAR-ATH by `full`.
    - s = 1 is the frame-level Bayes posterior. The decoder counts a per-frame prior once a frame (×0.2), so it is not
      claimed to be the decoder's correct prior (research.md section 4; D2 not pursued tonight).
  - **The feasibility gate**, read first, from `label_shift.py feasibility` on `tracking` v3, 3 seeds, before the grid
    runs:
    - π_true per match is the raw truth's home share of the alive graded frames.
    - error = the mean over matches of the mean over seeds of |π̂ - π_true|; baseline = the mean over matches of
      |0.5 - π_true|.
    - **It passes iff error ≤ 0.75 × baseline.** If it fails, P22(a) is dropped, the grid does not run, and the F1 is
      never produced.
    - [D1] The diagnostics (the calibration slope and π̂ over the truth's alive frames) are reported and decide
      nothing. Each names one cause for the one retry: temperature-scaled EM if the slope is outside [0.8, 1.25], EM
      over frames P05 calls alive if the dead frames are the cause.
  - **Candidates.**
    - `main.toml` (its config sha256 recorded here, as R017 did) plus `prior = "em"` at s ∈ {0.5, 1.0}:
      `pipeline/benchmark/configs/r020/r020_s050.toml` and `r020_s100.toml`.
    - Not re-chosen: the touch rule, the dead-ball model, P07's weight 0.2, scope, `min_prob`, the decoder 4/32/35/25,
      the shift +1/-2/0, Smart Tagging.
    - The shift is not re-derived. s_b measures the pipeline's lag at changes of team, and a constant tilt delays
      changes towards the minority team as much as it advances those towards the majority. This is a reading, not a
      measurement; the advisor's lane names no shift step.
  - **The runs.**

    ```bash
    $PY pipeline/benchmark/run.py pipeline/configs/main.toml pipeline/benchmark/configs/nearest_w50.toml \
        pipeline/benchmark/configs/nearest_w125.toml pipeline/benchmark/configs/r020/r020_s050.toml \
        pipeline/benchmark/configs/r020/r020_s100.toml --profile tracking --seeds 3 --truth both --drop-match-only
    ```

    The references must reproduce to the last digit (`main` 0.5430 / 0.5593, `nearest_w50` 0.2018 raw,
    `nearest_w125` 0.2226 coarse), or the lane stops. After the choice: the chosen config and `main.toml` on
    `--profile context --truth both --drop-match-only`.
  - **The selection.**
    - `choose.py --reference main r020_s050 r020_s100`: pooled `tracking` raw spellF1, a tie band of 0.005, the best
      coarse score among the tied, then s 0.5 (the stems' order).
    - Leave-one-match-out is reported and decides nothing.
    - Nothing is added after the results.
  - **The bar** (R014's).
    - The chosen config's `tracking` raw pooled Δ over `main.toml` must be ≥ +0.005, with sign-flip p ≤ 0.05.
    - **The veto:** not adopted if any of these is below -0.005: the `tracking` coarse Δ, the `context` raw Δ, the
      `context` coarse Δ, or the pooled `tracking` alive/dead F1 minus `main.toml`'s.
  - **The BAR-ATH checks, before adopting.**
    - π̂ (with `full`) is reported. Outside [0.39, 0.68] the lane stops for the advisor's sign-off, with `label_diff`
      aggregates.
    - R011's `touch_agreement`, shifted at the chosen s, must be ≥ the unshifted value - 0.005.
    - The same sha256 on two runs; `D` within 1 point of `main.toml`'s; no `X` in a graded block.
    - The possessions per half are reported.
    - The public window may be scored after the decision, as a correctness check only.
  - **Adoption.**
    - If everything passes, and P19's grid configs do not exist yet (advisor-01 §1), the chosen config becomes
      `main.toml` and is kept as `pipeline/configs/r020_label_shift.toml`.
    - Its sha256 goes to the submission log; it is not uploaded.
    - The merge is `--no-ff`, after both synthetic suites and `selftest.py --data` pass on the merged tree.
  - **If the gate or the bar fails.** One retry (R020b), only with a cause named from the logged aggregates, with its
    own part 1 and the same gate and bar. Otherwise, or after a second failure, P22(a) is dropped, and the prior,
    `label_shift.py`, its checks and the two configs are removed in one commit that names the commit to recover them
    from. The advisor decides whether a retry is warranted.
- [ ] **Step 2:** commit `"R020 part 1: the EM prior's feasibility gate, grid and bar, declared before any run"` and
  push. Give the controller the commit hash: no governed run may start before it.

## Task 6: the feasibility read (HEAVY steps go to the controller)

- [ ] **Step 1 (HEAVY):** `label_shift.py feasibility`, name `r020-feasibility`. It writes
  `data/runs/label_shift/feasibility.json`.
- [ ] **Step 2:** apply the gate mechanically, from the printed aggregates. Read nothing else.
- [ ] **Step 3:** write R020 part 2's first section:
  - the command, the commit and that the log ends `EXIT=0`;
  - the per-match table: π_true, π̂ per seed, iterations, |error|, |0.5 - π_true|, and [D1] the slope and π̂ over alive
    frames;
  - the gate's error, baseline, ratio and verdict.

  Commit `"R020 part 2: the feasibility gate <passes|fails>"` and push.
- [ ] **Step 4, if the gate fails:** STOP. The controller sends the table to the advisor, who decides on the retry.
  With no retry: P22(a) is dropped (see Task 5's last bullet: remove the code), and the lane goes on to Task 10.

## Task 7: the grid, the choice and the bar (HEAVY steps go to the controller)

- [ ] **Step 1 (HEAVY):** the run from part 1, name `r020-grid`.
- [ ] **Step 2:** check the references to the last digit. A mismatch STOPs the lane: report it to the controller.
- [ ] **Step 3:** `$PY pipeline/benchmark/choose.py --reference main r020_s050 r020_s100 --out
  data/runs/benchmark/label_shift/r020_choice.json`. This reads the reports and is not heavy.
- [ ] **Step 4 (HEAVY):** `run.py pipeline/configs/main.toml pipeline/benchmark/configs/r020/<chosen>.toml --profile
  context --seeds 3 --truth both --drop-match-only`, name `r020-context`. `main` must reproduce 0.5489 raw and 0.5642
  coarse (A13).
- [ ] **Step 5:** apply the bar and the veto mechanically, in R014's table form, and write part 2's grid section:
  - the runs;
  - the table of the five configs (raw and coarse spellF1, macroF1, alive/dead F1, possessions per minute, `D` share,
    the deltas, p and wins);
  - the selection;
  - the leave-one-match-out estimate;
  - the context run;
  - the bar;
  - a reading. It may compare the per-match deltas with each match's |π̂ - π_true| from Task 6, as a reading only.

  Commit `"R020 part 2: the grid, the choice and the bar <met|not met>"` and push.
- [ ] **Step 6, if the bar is not met:** STOP, as in Task 6 step 4: the advisor decides on R020b. Then go to Task 10.

## Task 8: the BAR-ATH checks (HEAVY steps go to the controller; only if the bar is met)

- [ ] **Step 1 (HEAVY):** `label_shift.py bar-ath --strength <s>`, name `r020-bar-ath`. It prints π̂, the iterations,
  the frames and the two `touch_agreement`s.
- [ ] **Step 2 (HEAVY):** `label_diff.py pipeline/configs/main.toml pipeline/benchmark/configs/r020/<chosen>.toml`,
  name `r020-label-diff`. It gives the `D` share, the `X` share, the possessions per half, the changed share and the
  changed share in ball gaps.
- [ ] **Step 3:** if π̂ is outside [0.39, 0.68], STOP: the controller sends π̂ and Step 2's aggregates to the advisor,
  and nothing is adopted without the advisor's sign-off.
- [ ] **Step 4 (HEAVY):** `pipeline/run.py pipeline/benchmark/configs/r020/<chosen>.toml` twice, names `r020-run-1` and
  `r020-run-2`. The sha256 must match, and `validate_submission.py` must pass on the file.
- [ ] **Step 5:** apply the checks (touch ≥ unshifted - 0.005; `D` within 1 point; no `X`) and write part 2's BAR-ATH
  section. Commit `"R020 part 2: the BAR-ATH checks <pass|fail>"` and push.

## Task 9: adoption (only if the bar is met, the checks pass, and the controller confirms that P19's grid configs do not exist yet)

- [ ] **Step 1:** copy the chosen config to `pipeline/configs/r020_label_shift.toml`, with a header saying what it is.
  `main.toml` gets the two keys, its header paragraph gets one sentence on R020, and its `description` string gets
  "+ EM prior (R020, s <s>)".
- [ ] **Step 2 (HEAVY):** `pipeline/run.py pipeline/configs/main.toml`. Its sha256 must equal Task 8's, and
  `validate_submission.py` must pass.
- [ ] **Step 3:** run both synthetic suites. Then **(HEAVY)** `pipeline/selftest.py --data`.
- [ ] **Step 4 (HEAVY):** `run.py pipeline/configs/main.toml --profile tracking --seeds 3 --truth both
  --drop-match-only`. It must reproduce the chosen config's benchmark scores, since the two are the same config.
- [ ] **Step 5:** add a paragraph to the submission log ("`main.toml` was rewritten a sixth time by R020 ...", sha256,
  commit, NOT uploaded) and part 2's verdict. Commit `"main.toml: R020's EM label-shift prior at s <s> (P22a)"` and
  push.
- [ ] **If adoption is held** (P19's configs exist): write "adoption held for the advisor (advisor-01 §1: main.toml
  frozen until P19's verdict)" in part 2 and touch nothing else.

## Task 10: P21 step 0, wrong-team slots at the ball (A16)

The code can be written at any time after Task 0. Its heavy run is queued after P22's heavy runs and before P19's
(advisor-01's heavy order).

**Files:**
- Create: `pipeline/benchmark/wrong_team.py`.
- Create: `pipeline/benchmark/checks/wrong_team.py`. Add it to `checks.MODULES` and its docstring.

**Interfaces.** Everything below is produced by this task.

`# Params`:
- `RADIUS = 1.5` and `MAX_REL_SPEED = 6.0`: the advisor's figures, which are `main.toml`'s `controlled_touch` (a check
  holds them equal);
- `TEAM_SIZE = 11`;
- `RULE = 1.5`: below it, P21 is dropped;
- [D3] `ONSET_WINDOW_S = 1.0`.

Functions:
- `over_eleven(present, slot_team) -> (n,) bool`: either team has more than 11 present slots.
- `surplus(present, slot_team) -> (n,) bool`: T32's stricter definition, one team over 11 and the other under 11
  (`calibration.player_stats`' rule). Report only.
- `candidates(inputs, *, radius=RADIUS, max_rel_speed=MAX_REL_SPEED, duel_radius=None) -> (n,) bool`:
  - graded, ball visible, the nearest present slot within `radius`, |v_ball - v_nearest| ≤ `max_rel_speed`;
  - velocities from positions (`kinematics.ball_track`, `player_track`, `ball_distances`, as `controlled_touch`; T31);
  - `duel_radius=None` skips the duel test (the advisor's definition); a number applies `controlled_touch`'s test.
- `ratio(over, candidate, graded) -> {"graded_frames", "candidate_frames", "share_candidates", "share_graded", "ratio",
  "verdict"}`: pure. `verdict` is "P21 dropped" if ratio < 1.5, else "report only".
- `measure(inputs) -> dict`: `ratio` on `over_eleven`, plus the same on `surplus` under `"surplus"` (report only).
  [D3] It also gives `onset`: the share of first frames of over-11 runs (inside the graded blocks) that lie within
  ±1 s of a candidate frame, over the share of graded frames within ±1 s of one. Report only.
- CLI, HEAVY: `$PY pipeline/benchmark/wrong_team.py [--data-dir D] [--out PATH]`. It runs on BAR-ATH
  (`Inputs(data_dir=...)`), prints aggregates only, and writes `data/runs/wrong_team/ratio.json`.

- [ ] **Step 1: write the failing checks.**
  - `test_wrong_team_over_eleven_and_surplus_exact`: six hand-built frames with (home, away) present slots (12, 10),
    (11, 11), (12, 11), (13, 9), (10, 12), (11, 12). Expected `over_eleven` T F T T T T; `surplus` T F F T T F.
  - `test_wrong_team_candidates_are_the_touch_rule_without_the_duel_test`:
    - On a `pipeline/selftest.py` kinematics scene with duels near the ball (assert the precondition),
      `candidates(duel_radius=1.0)` equals `controlled_touch`'s kept frames at radius 1.5, 6.0 m/s, `min_frames` 1,
      confidence 0.99, hold 0.5, duel_radius 1.0.
    - `candidates()` is a strict superset of that.
  - `test_wrong_team_ratio_exact`:
    - 1,000 graded frames, 100 of them candidates, 30 of the candidates over 11, 50 over 11 in all: ratio 6.0.
    - The verdict at a ratio of exactly 1.5 is "report only", and at 1.4999 "P21 dropped".
    - Frames outside the graded blocks never count.
  - `test_wrong_team_constants_are_main_tomls`: `RADIUS` and `MAX_REL_SPEED` equal `main.toml`'s `controlled_touch`
    entry, read with `load_config`.
  - [D3] `test_wrong_team_onset_ratio_exact`: a hand-built mask.
  - `test_wrong_team_prints_aggregates_only`.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement. **Step 4:** run both synthetic suites, expect PASS.
- [ ] **Step 5:** commit `"wrong_team: P21 step 0, wrong-team slots at the ball against all graded frames"` and push.
- [ ] **Step 6:** write A16 part 1 in `docs/results.md`, after A15:
  `### A16 — P21 step 0: do wrong-team slots cluster at the ball? (2026-10-31, <who>)`.
  - The definitions above.
  - The rule: **ratio < 1.5: P21 dropped; ≥ 1.5: report only, and the proposal's ceiling run is not run**, because it
    would not measure swaps at the ball.
  - The expected direction, with research.md section 7's sources: above 1, pulled towards 1 by a surplus that persists
    after its swap and by set-piece crowding away from the ball.
  - That the tool stays for P19's transfer report.

  Commit `"A16 part 1: P21 step 0's ratio and rule, declared before the run"` and push.
- [ ] **Step 7 (HEAVY):** `wrong_team.py`, name `a16-wrong-team`.
- [ ] **Step 8:** write A16 part 2 (the counts, the shares, the ratio, the `surplus` line, [D3] the onset ratio, the
  verdict). Set P21's status in `docs/proposals.md`: "dropped → A16 (ratio x.xx < 1.5)", or "step 0 done → A16 (ratio
  x.xx, report only; ceiling not run)". Commit `"A16: P21 step 0, ratio <x.xx>, <verdict>"` and push.

## Task 11: the docs

- [ ] **Step 1, `docs/results.md`:**
  - R020's scoreboard row: the date, what, public `—` unless scored after the decision, the val spellF1 of the chosen
    config (or `—` if the grid never ran), and the verdict.
  - Make part 2 complete.
  - A16 gets no scoreboard row: it is an analysis.
- [ ] **Step 2, `docs/proposals.md`:**
  - P22's overview row and status: "partly done → R020 ((a) <adopted|dropped>; (b) left)".
  - Under "### P22", a "**Research (2026-10-31)**" paragraph of five or six lines from research.md (Tian's
    instruction: findings go in the proposal's section): calibration and why none was added; dependence; the
    per-frame prior in an HMM (D2 as the next step if (a) helps); the dead frames in the sample. Then (a)'s outcome.
  - P21's overview row and status, from Task 10.
- [ ] **Step 3, `docs/technical_document.md`:**
  - If adopted: the evidence section (one paragraph on the prior), the audit table of constants (`prior_strength`
    from R020's grid; the EM tolerance and iterations fixed beforehand; π̂ estimated per match, never taken from
    outside), and the shape model's paragraph on transfer.
  - If dropped: one line in the dropped list.
  - Either way, in the section on noisy tracking: A16's ratio and verdict.
- [ ] **Step 4, `pipeline/evidence/README.md`:** the `shape_possession` row says which configs use the prior (`main.toml`
  if adopted). If (a) was dropped and its code removed, revert the row.
- [ ] **Step 5, `pipeline/benchmark/README.md` and `CLAUDE.md`:**
  - one line each for `label_shift.py` (if kept) and `wrong_team.py`, in the module list;
  - their commands in CLAUDE.md's command block, as the last CLAUDE.md commit did for `tag_stats` and `label_diff`.
- [ ] **Step 6, `site/progress.html`** (read `site/README.md` first; edit `DATA` only):
  - `points`: if the grid ran, append `{ id: "R020", date: "2026-10-31", raw, coarse, adopted, dr, dc, prop: "P22",
    what, note }`. raw and coarse are the chosen config's pooled scores; dr and dc are its pooled Δ over `main.toml`
    (R013's 0.5430 / 0.5593, the last adopted point), as part 2 states them. No point if the grid never ran.
  - `ledger`: append a P22(a) row (entry R020, its deltas or `—`, status adopted or dropped, one line of why) and a
    P21 step 0 row (entry A16, `—`, `dropped` or `unbenchmarked`, the ratio in the why).
  - `features`: P22 and P21 if their status is final.
  - If R020 is adopted, `chartCaption`'s sentence on today's `main.toml`.
  - Bump `asOf` and `footer`.
  - Check every number against `docs/results.md`. Open the page; the console must stay empty.
- [ ] **Step 7, `site/index.html`** (`DATA` only):
  - the `proposals` entries for P21 and P22 (status, impact);
  - if adopted: the `components` and `models` `shape_possession` params (`prior em · prior_strength <s>`), `today`'s
    text, a `timeline` entry and the `cards` as they move;
  - bump `asOf`, `commit` and `footer`. Open it and click through the tabs; the console stays empty.
- [ ] **Step 8, `docs/traps.md`:** a new trap (T48 or the next free) only if something cost real time.
- [ ] **Step 9:** commit `"P22(a) and P21 step 0 docs: results, proposals, technical document, READMEs, site"` and
  push.

## Task 12: final checks and the merge (the controller)

- [ ] **Step 1:** both synthetic suites on the branch. Expect PASS, with the counts recorded in the ledger.
- [ ] **Step 2:** the controller merges into `main` with `--no-ff`, only after both synthetic suites pass on the
  merged tree. If `main.toml` changed: **(HEAVY)** `selftest.py --data` on the merged tree, and the submission log's
  sha256 checked once more.
- [ ] **Step 3:** push `origin main`. Update the SDD ledger and the run ledger (`lanes.md`) with the verdicts, the
  commits and anything the advisor decided.
