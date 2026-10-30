# Part C, P08: the `smart_tagging` evidence source (R017)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.
> Read the index plan's Global Constraints first: `docs/superpowers/plans/2026-10-30-p02-p08-p09-p10-p12-index.md`.

**Goal:** turn Smart Tagging's automatic tags into team votes and dead-ball votes, measure their agreement with our
own pipeline's pseudo-labels on BAR-ATH, and apply R017's bar. If it passes, Tian picks the strength before
`main.toml` changes.

**Spec:** design section 5.C (the vote table, overlaps, the two strengths, the bar).

**Branch / worktree:** `git worktree add ../BarcaHackathon-p08 -b p08-smart-tagging main` (after Task 0).

**Data rule for this part:** Smart Tagging is match data.
- The distinct code names with their counts may be printed: they are schema-level aggregates, as
  `starter/load_smart_tagging.py`'s summary already prints them.
- Tag rows, their frames and their team values per instance may not.

## Task C.1: the source

**Files:**
- Create: `pipeline/evidence/smart_tagging.py`
- Modify: `pipeline/evidence/__init__.py` (register `"smart_tagging"`), `pipeline/evidence/README.md` (the table row)
- Test: `pipeline/selftest.py`

**Interfaces:**
- Consumes: `inputs.smart_tagging`, a list of `{code, start_frame, end_frame, start_s, end_s, tags: {group: text}}`
  with `end_frame` exclusive; `inputs.graded`. Task 0's `MATCH_ONLY`.
- Produces:
  - `MATCH_ONLY = True`.
  - `# Params`:
    - `TEAM_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
      "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL")`;
    - `OTHER_TEAM_PATTERN = r"^DEFENDING IN .+ THIRD$"` plus `OTHER_TEAM_CODES = ("DEFENSIVE TRANSITION",)`;
    - `DEAD_CODES = ("SET PIECES", "GOALS")`;
    - `TEAMS = {"barcelona": "home", "atletic club": "away"}`.

    Codes are matched after `.strip().upper()`, team values after `.strip().lower()`, and the `Team` group key
    case-insensitively (T17). Anything else (`N/A`, missing) gives no team vote.
  - `votes(inputs) -> (team, dead)`: per-frame arrays from the tags alone.
    - `team` is int8: 1 home, 0 away, −1 no vote, −2 disagreement.
    - `dead` is bool.
    - Both are restricted to the graded blocks, with tags cut at block edges (half-open).
    - Public: the measurement in Task C.2 uses it.
  - `tag_spans(inputs) -> list[(kind, side, start, stop)]`: the per-tag spans after cutting at the graded blocks.
    `kind` is "team" or "dead"; `side` is 1, 0 or None. Public, for spread "tag" and the bootstrap.
  - `compute(run, *, c_team=None, c_dead=None, spread="frame") -> Evidence`.
    - At least one of `c_team` and `c_dead` is given, each a float in (0.5, 1]. `spread` is "frame" or "tag".
    - **"frame":** `home` = c_team (team 1) or 1 − c_team (team 0) on voted frames, NaN on −1 and −2. `alive` =
      1 − c_dead on dead frames.
    - **"tag":** per frame, the sum over the agreeing tags covering it of logit(c) / n_tag (n_tag = the tag's frame
      count after cutting), turned back by sigmoid. Frames with disagreeing team tags are NaN. Dead is the same with
      c_dead, for `alive` = 1 − that.
    - NaN outside the graded blocks, always.

- [ ] **Step 1: Write the failing tests** in `pipeline/selftest.py`. Use synthetic `Inputs` whose `smart_tagging`
  attribute is a hand-built list (override the cached property on a `synthetic_inputs` object):
  - `test_smart_tagging_vote_table`:
    - each `TEAM_CODES` code with `Barcelona` gives home;
    - `DEFENDING IN OWN THIRD` with `Barcelona` gives away;
    - `DEFENSIVE TRANSITION` with `Atletic Club` gives home;
    - `SET PIECES` with `N/A` gives dead and no team vote;
    - an unknown code gives nothing;
    - lower-case codes and a `team` group key still work.
  - `test_smart_tagging_overlaps`: two agreeing overlapping tags give one vote (frame mode: value c, not more), and
    two disagreeing ones give NaN on their overlap only.
  - `test_smart_tagging_cut_at_graded_blocks`:
    - a tag crossing a block's start and one inside the halftime gap leave NaN outside the blocks;
    - `end_frame` is exclusive;
    - spread "tag"'s n_tag counts the frames inside the blocks only.
  - `test_smart_tagging_tag_spread_sums_to_one_observation`: on a single tag, the sum over its frames of
    logit(home) equals logit(c_team) to 1e-12.
  - `test_smart_tagging_rejects_bad_parameters`: no c given; c values 0.5, 1.2, True and "0.8"; spread "x".
  - `test_smart_tagging_is_match_only`: `match_only("smart_tagging") is True`.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement, numpy only. **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit `"evidence: smart_tagging, team and dead-ball votes from the tags (P08)"`, then push.

## Task C.2: the measurement

**Files:**
- Create: `pipeline/benchmark/tag_stats.py`
- Test: `pipeline/benchmark/checks/tag_stats.py` (collected like the others)

**Interfaces:**
- Consumes: `smart_tagging.votes` and `tag_spans`; `controlled_touch.compute` with `conventions.TOUCH`'s parameters
  (kept episodes, the way `train_shape.touch_agreement` takes them); `dead_ball_model.compute` with `main.toml`'s
  `model` and `min_prob` (read from `pipeline/configs/main.toml` with `load_config`, not copied).
- Produces:
  - `team_agreement(team, touch_home, spans, *, draws=2000, seed=(17, 0)) -> {"n", "share", "low", "high",
    "coverage", "coverage_gaps"}`.
    - The frames are those inside kept touch episodes with a team vote of 0 or 1. The share is where the vote names
      the episode's team.
    - `low` and `high` are a 90% bootstrap that resamples whole team tags with replacement.
    - `coverage` is the share of graded frames with a team vote; `coverage_gaps` the same among graded frames whose
      ball is missing.
  - `dead_agreement(dead, alive_p, spans, graded_mask, *, draws=2000, seed=(17, 1)) -> {"n", "share", "low", "high",
    "base", "lift"}`.
    - The share is over dead-voted frames where `alive_p < 0.5`.
    - The bootstrap resamples whole dead tags.
    - `base` is the same share on all graded frames; `lift = share / base`.
  - `per_code(...) -> [{"code", "frames", "share"}]`: aggregates only, for the retry's diagnosis.
  - `rounded_towards_half(value) -> float`: reuse P05's `--rules` rounding helper from `train_dead_ball.py` (find it;
    do not write a second one).
  - CLI: `$PY pipeline/benchmark/tag_stats.py [--data-dir D]` (HEAVY).
    - It prints the two results, the per-code table, and the would-be `c_team` and `c_dead`.
    - It writes `data/runs/tag_stats/stats.json`.
    - It prints no frame and no row.

- [ ] **Step 1: Write the failing tests** (synthetic):
  - `test_tag_stats_team_agreement_exact`: hand-built votes, touch and spans give the exact share and coverages.
  - `test_tag_stats_bootstrap_resamples_whole_tags`: with one tag all right and one all wrong, every draw's share is
    0, 0.5 or 1.
  - `test_tag_stats_dead_agreement_and_lift`.
  - `test_tag_stats_prints_aggregates_only`.
- [ ] **Step 2:** run, expect FAIL. **Step 3:** implement. **Step 4:** run, expect PASS.
- [ ] **Step 5:** commit `"tag_stats: the tags' agreement with our own pseudo-labels on BAR-ATH (P08)"`, then push.

## Task C.3: R017 part 1, declared before any measurement

- [ ] **Step 1:** write R017 part 1 in `docs/results.md`, copying the spec's 5.C "Declared" block:
  - the vote table;
  - the pseudo-labels (`conventions.TOUCH` episodes; `main.toml`'s dead-ball model, P(alive) < 0.5);
  - the two parts' bars:
    - team: point ≥ 0.75, lower 90% bound ≥ 0.70, `coverage_gaps` ≥ 0.05;
    - dead: point ≥ 0.60, lower bound ≥ 0.50, lift ≥ 1.5;
  - the two strengths and Tian's pick;
  - the label-free checks;
  - the retry rule (R017b: one retry, informed by `per_code`; then removal).

  Record `main.toml`'s sha256.
- [ ] **Step 2:** commit `"R017 part 1: the tags' votes, pseudo-labels and bars, declared before any measurement"`,
  then push.

## Task C.4: measure, apply the bar, prepare the strengths (HEAVY steps go to the controller)

- [ ] **Step 1 (HEAVY):** `$PY pipeline/benchmark/tag_stats.py`
- [ ] **Step 2:** apply each part's bar. A part that fails gets `None` for its c.
- [ ] **Step 3:** if at least one part passes, write `pipeline/configs/r017_frame.toml` and `r017_tag.toml`.
  - Each is `main.toml` plus an `[[evidence]]` `kind = "smart_tagging"` with the passing parts' rounded c values.
  - `r017_frame` uses `spread = "frame"`, `weight = 0.2`; `r017_tag` uses `spread = "tag"`, `weight = 1.0`.
- [ ] **Step 4 (HEAVY):** `$PY pipeline/benchmark/label_diff.py pipeline/configs/main.toml
  pipeline/configs/r017_frame.toml pipeline/configs/r017_tag.toml`
- [ ] **Step 5:** R017 part 2 in `docs/results.md`: the measurements, the bar's verdict per part, and the `label_diff`
  aggregates.
- [ ] **Step 6:** commit `"R017 part 2: measured, bar <verdict>"`, then push.
- [ ] **Step 7: STOP for Tian.** The controller shows Tian the two `label_diff` summaries and asks: frame, tag, or
  neither.
- [ ] **Step 8, after Tian's pick (HEAVY):**
  - The picked config becomes `main.toml`, kept as `pipeline/configs/r017_smart_tagging.toml`.
  - Run `main.toml` twice (same sha256), then `validate_submission.py` and `selftest.py --data`.
  - Run `$PY pipeline/benchmark/run.py pipeline/configs/main.toml --drop-match-only --truth both`; it must reproduce
    R013's 0.5430 / 0.5593, since the benchmark sees no tags.
  - Update `docs/technical_document.md`, the submission log, P08's status in `docs/proposals.md`, and the Models tab
    in `site/index.html` (a source row).
  - Commit, then the controller merges into `main` (rebase first) and pushes.
- [ ] **If both parts fail:** declare R017b with one change informed by `per_code` (e.g. a code removed from the table,
  with its reason). If R017b fails too, remove `smart_tagging.py`, `tag_stats.py` and their checks in a commit that
  names the commit to recover them from, and set P08 to "dropped → R017, R017b".
