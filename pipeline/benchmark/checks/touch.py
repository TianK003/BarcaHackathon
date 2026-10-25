"""Checks of touch_stats (pipeline/benchmark/touch_stats.py): P04's control rule measured on the benchmark.

touch_stats says how much of the alive, ball-visible frames a control rule covers and how often its team is the truth's,
and from the agreement it sets the confidence that Stage A's configs give the rule. A mistake in it would put a wrong
confidence into every config, so what is held against it here is a plain loop over the frames (coverage, agreement and
the flip lags worked out one frame at a time from the source's own evidence and the truth), the confidence rule
(floor to a hundredth, clipped to [0.51, 0.99]), the default grid and its labels, and that the source is never handed
the truth: the inputs it is called on go through the deep walk of common.leaks.

The matches are the synthetic ones of checks/common.py (random flags and tracking, figures meaningless), under the
gaps profile and the synthetic Calibration.
"""

import math

import numpy as np

from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import touch_stats, truth
from pipeline.benchmark.checks.common import leaks, synthetic_calibration, synthetic_matches
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics
from pipeline.selftest import expect_error, quiet

# Params
WIDE = {"radius": 40.0, "max_rel_speed": math.inf, "min_frames": 1}   # on random tracking nearly every frame qualifies
SHORT = {"radius": 40.0, "max_rel_speed": math.inf, "min_frames": 2}
LAG_SECONDS = 5


def measured(settings, **more):
    return touch_stats.measure(settings, profile="gaps", seeds=1, matches=synthetic_matches(),
                               calibration=synthetic_calibration(), log=quiet, **more)


def plain_loop(match, setting, seed=0):
    """(population, kept, agree, lags, changes) of one match and seed, one frame at a time from its evidence."""
    atd = benchmark_run.make_atd(match, "gaps", seed, synthetic_calibration())
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
    home = controlled_touch.compute(RunContext(inputs=inputs, seed=0), confidence=0.99, **setting).home
    visible = kinematics.ball_track(inputs).visible
    codes, _ = truth.truth_codes(match)
    population = kept = agree = 0
    for start, stop in kinematics.blocks(inputs):
        for i in range(start, stop):
            if codes[i] not in (0, 1) or not visible[i]:
                continue
            population += 1
            if not np.isnan(home[i]):
                kept += 1
                agree += int((home[i] > 0.5) == (codes[i] == 0))
    lags, changes = [], 0
    for start, stop in kinematics.blocks(inputs):
        for c in range(start + 1, stop):
            if codes[c] in (0, 1) and codes[c - 1] in (0, 1) and codes[c] != codes[c - 1]:
                changes += 1
                for s in range(max(start, c - LAG_SECONDS * match.fps), min(stop, c + LAG_SECONDS * match.fps + 1)):
                    if np.isnan(home[s]) or (home[s] > 0.5) != (codes[c] == 0):
                        continue
                    if s > start and not np.isnan(home[s - 1]) and (home[s - 1] > 0.5) == (home[s] > 0.5):
                        continue                                  # not the first frame of its episode
                    lags.append(s - c)
                    break
    return population, kept, agree, lags, changes


def test_touch_stats_matches_a_loop():
    """Coverage, agreement and the flip lags equal a frame-by-frame loop, per match and pooled, for two settings."""
    matches = synthetic_matches()
    result = measured([WIDE, SHORT])
    assert list(result["settings"]) == ["r40.0_vinf_k1", "r40.0_vinf_k2"], list(result["settings"])
    for setting in (WIDE, SHORT):
        entry = result["settings"]["r40.0_vinf_k{}".format(setting["min_frames"])]
        total = np.zeros(3, dtype=int)
        pooled_lags, pooled_changes = [], 0
        for match_id, match in matches.items():
            population, kept, agree, lags, changes = plain_loop(match, setting)
            assert kept > 0 and agree < kept, "a loop that finds nothing, or everything right, proves little"
            row = entry["per_match"][match_id]
            assert row["frames"] == population and row["kept"] == kept, (match_id, row)
            assert row["coverage"] == kept / population and row["agreement"] == agree / kept, (match_id, row)
            total += (population, kept, agree)
            pooled_lags += lags
            pooled_changes += changes
        pooled = entry["pooled"]
        assert (pooled["frames"], pooled["kept"]) == (total[0], total[1]), pooled
        assert pooled["coverage"] == total[1] / total[0] and pooled["agreement"] == total[2] / total[1], pooled
        flip = entry["flip_lag"]
        assert flip["changes"] == pooled_changes and flip["found"] == len(pooled_lags), flip
        assert flip["share_found"] == len(pooled_lags) / pooled_changes, flip
        assert pooled_lags, "no flip was found, so no lag was compared"
        expected = np.percentile(pooled_lags, [25, 50, 75])
        assert (flip["p25"], flip["median"], flip["p75"]) == tuple(float(v) for v in expected), flip
        assert entry["confidence"] == touch_stats.confidence_from_agreement(pooled["agreement"])


def test_touch_stats_confidence_rule():
    """floor(100 a) / 100 clipped to [0.51, 0.99], and 0.57 stays 0.57 (100 * 0.57 is 56.99999999999999)."""
    rule = touch_stats.confidence_from_agreement
    assert rule(0.8765) == 0.87 and rule(0.57) == 0.57 and rule(0.58) == 0.58 and rule(0.7) == 0.7
    assert rule(0.5) == 0.51 and rule(0.0) == 0.51 and rule(0.509) == 0.51
    assert rule(0.9999) == 0.99 and rule(1.0) == 0.99
    assert isinstance(rule(0.8765), float)
    for bad in (-0.1, 1.1, float("nan"), True, "0.9"):
        expect_error(ValueError, lambda bad=bad: rule(bad), "agreement")


def test_touch_stats_default_grid():
    """The twelve settings of Stage A, radius outermost then max_rel_speed then min_frames, each once."""
    grid = touch_stats.DEFAULT_GRID
    assert len(grid) == 12 and len({tuple(sorted(s.items())) for s in grid}) == 12
    assert [(s["radius"], s["max_rel_speed"], s["min_frames"]) for s in grid] == [
        (r, v, k) for r in (1.0, 1.5, 2.0) for v in (3.0, 6.0) for k in (3, 8)]
    assert touch_stats.label(grid[0]) == "r1.0_v3.0_k3" and touch_stats.label(grid[-1]) == "r2.0_v6.0_k8"
    assert touch_stats.label(WIDE) == "r40.0_vinf_k1"


def test_touch_stats_never_passes_truth():
    """The inputs the source is called on hold neither the match nor its truth, however deep leaks() looks."""
    matches = synthetic_matches()
    seen = []
    original = controlled_touch.compute

    def spy(run, **parameters):
        seen.append(run.inputs)
        return original(run, **parameters)

    controlled_touch.compute = spy
    try:
        measured([WIDE])
    finally:
        controlled_touch.compute = original
    assert len(seen) == len(matches), "one call per match, seed and setting"
    for inputs, match in zip(seen, matches.values()):
        assert leaks(inputs, match) == [], leaks(inputs, match)


def test_touch_stats_refuses_what_it_cannot_measure():
    """No setting, a setting without a key or with a key too many, and a repeated setting each name touch_stats."""
    for settings, message in (([], "setting"), ([{"radius": 1.0}], "min_frames"),
                              ([dict(WIDE, extra=1)], "extra"), ([WIDE, dict(WIDE)], "twice")):
        expect_error(ValueError, lambda settings=settings: measured(settings), "touch_stats", message)
