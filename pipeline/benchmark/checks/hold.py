"""Checks of P29's measurement (pipeline/benchmark/hold_stats.py): the hold frames' agreement with the truth by time since
the last controlled touch and by whether the ball was hidden since, its gate, and the curve file made from it.

Everything is synthetic. Filled by the lane of P29 (docs/superpowers/plans/2026-11-12-p29-p31.md, Tasks 1 and 2).
"""

import contextlib
import inspect
import io
import json
import math
import os
import tempfile
import tomllib

import numpy as np

from pipeline.benchmark import hold_stats, sources
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import truth
from pipeline.benchmark.checks.common import loading, random_match, synthetic_matches
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics
from pipeline.selftest import expect_error, quiet

# Params
MAIN_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs",
                           "main.toml")
# On random tracking nearly every frame qualifies and the nearest team changes at random: a wide rule with a few frames
# of minimum gives episodes of both teams, and the ball that comes and goes gives hold frames with and without a gap.
SPARSE = {"radius": 40.0, "max_rel_speed": math.inf, "min_frames": 3, "duel_radius": 1.0}


def test_hold_stats_touch_is_mains():
    """TOUCH is main.toml's control rule (radius, max_rel_speed, min_frames) and compute's own duel_radius."""
    with open(MAIN_CONFIG, "rb") as handle:
        config = tomllib.load(handle)
    entries = [entry for entry in config["evidence"] if entry["kind"] == "controlled_touch"]
    assert len(entries) == 1, entries
    for key in ("radius", "max_rel_speed", "min_frames"):
        assert hold_stats.TOUCH[key] == entries[0][key], (key, hold_stats.TOUCH[key], entries[0][key])
    default = inspect.signature(controlled_touch.compute).parameters["duel_radius"].default
    assert hold_stats.TOUCH["duel_radius"] == default, (hold_stats.TOUCH["duel_radius"], default)
    assert set(hold_stats.TOUCH) == {"radius", "max_rel_speed", "min_frames", "duel_radius"}, hold_stats.TOUCH


def recount(match, touch):
    """{(row, bin): [frames, agree]} of one match under the clean profile, in a plain loop from hold_state."""
    atd = benchmark_run.make_atd(match, "clean", 0, None)
    inputs = match_inputs.MatchInputs(match, atd, profile="clean", seed=0)
    state = controlled_touch.hold_state(RunContext(inputs=inputs, seed=0), **touch)
    bins = controlled_touch.hold_bin(state.elapsed, match.fps)
    codes, _ = truth.truth_codes(match)
    cells = {}
    for start, stop in kinematics.blocks(inputs):
        for frame in range(start, stop):
            if state.elapsed[frame] < 0 or codes[frame] not in (0, 1):
                continue
            cell = cells.setdefault(("gap" if state.gap[frame] else "no_gap", int(bins[frame])), [0, 0])
            cell[0] += 1
            cell[1] += int(state.last_is_home[frame] == (codes[frame] == 0))
    return cells


def test_hold_stats_matches_a_loop():
    """Each per-match cell equals a frame-by-frame recount and the pooled cell is the sum of the matches'."""
    matches = synthetic_matches()
    result = hold_stats.measure(touch=SPARSE, profile="clean", seeds=1, matches=matches, log=quiet)
    assert result["profile"] == "clean" and result["seeds"] == [0] and result["matches"] == sorted(matches)
    assert result["touch"] == SPARSE and result["bins_s"] == list(controlled_touch.HOLD_BINS_S)
    pooled = result["pooled"]
    filled = [index for index in range(10) if pooled["no_gap"][index][0] + pooled["gap"][index][0] > 0]
    assert len(filled) >= 2, "the synthetic matches fill fewer than two bins: {}".format(pooled)
    for name in ("no_gap", "gap"):
        assert sum(cell[0] for cell in pooled[name]) > 0, "no frame in the {} row: {}".format(name, pooled)
    for name in ("no_gap", "gap"):
        assert len(pooled[name]) == 10, name
        for index in range(10):
            total = [sum(result["per_match"][match_id][name][index][part] for match_id in matches)
                     for part in (0, 1)]
            assert pooled[name][index] == total, (name, index, pooled[name][index], total)
    for match_id, match in matches.items():
        cells = recount(match, SPARSE)
        for name in ("no_gap", "gap"):
            for index in range(10):
                assert result["per_match"][match_id][name][index] == cells.get((name, index), [0, 0]), (
                    match_id, name, index)


def test_hold_stats_reads_touch_at_call_time():
    """touch=None means whatever hold_stats.TOUCH is when measure is called, not when it was defined."""
    matches = synthetic_matches()
    original = hold_stats.TOUCH
    hold_stats.TOUCH = SPARSE
    try:
        replaced = hold_stats.measure(profile="clean", seeds=1, matches=matches, log=quiet)
    finally:
        hold_stats.TOUCH = original
    assert replaced["touch"] == SPARSE
    explicit = hold_stats.measure(touch=SPARSE, profile="clean", seeds=1, matches=matches, log=quiet)
    assert replaced["pooled"] == explicit["pooled"]


def literal(*cells):
    """A result whose first cells are the given [frames, agree] (no_gap bin 0, gap bin 0, no_gap bin 1), rest empty."""
    result = {"pooled": {"no_gap": [[0, 0] for _ in range(10)], "gap": [[0, 0] for _ in range(10)]}}
    for (name, index), cell in zip((("no_gap", 0), ("gap", 0), ("no_gap", 1)), cells):
        result["pooled"][name][index] = list(cell)
    return result


def test_hold_stats_gate():
    """The gate is the frame-weighted deviation from the pooled agreement, passing at 0.05 and above."""
    verdict = hold_stats.gate(literal([1000, 900], [1000, 700]))
    assert verdict["frames"] == 2000 and abs(verdict["agreement"] - 0.8) < 1e-12, verdict
    assert abs(verdict["deviation"] - 0.1) < 1e-12 and verdict["passes"] is True, verdict
    verdict = hold_stats.gate(literal([1000, 850], [1000, 750]))
    assert abs(verdict["deviation"] - 0.05) < 1e-12 and verdict["passes"] is True, verdict
    verdict = hold_stats.gate(literal([1000, 800], [1000, 740], [5, 0]))
    assert 0.031 < verdict["deviation"] < 0.033 and verdict["passes"] is False, verdict
    verdict = hold_stats.gate(literal())
    assert verdict["deviation"] is None and verdict["agreement"] is None and verdict["passes"] is False, verdict
    assert hold_stats.GATE_DEVIATION == 0.05


def test_hold_stats_measure_refuses_a_bad_touch():
    """A missing, extra or misspelled key, or a value controlled_touch refuses, is a ValueError before any match loads."""
    def never(match_id):
        raise AssertionError("a match was loaded for a bad touch: {}".format(match_id))

    good = dict(SPARSE)
    bads = ({key: value for key, value in good.items() if key != "radius"}, dict(good, extra=1),
            {("radious" if key == "radius" else key): value for key, value in good.items()},
            dict(good, radius=-1.0), dict(good, min_frames=1.5), dict(good, max_rel_speed=0), dict(good, radius=True))
    with loading(never):
        for bad in bads:
            expect_error(ValueError, lambda: hold_stats.measure(touch=bad, profile="clean", seeds=1), "hold_stats",
                         "touch")


def result_of(pooled_cell, per_match_cells):
    """A measure()-shaped result whose only filled cell is no_gap bin 0: `pooled_cell` and each match's."""
    def cells(cell):
        return {"no_gap": [list(cell)] + [[0, 0] for _ in range(9)], "gap": [[0, 0] for _ in range(10)]}

    return {"profile": "generic", "seeds": [0, 1], "matches": sorted(per_match_cells),
            "bins_s": list(controlled_touch.HOLD_BINS_S), "touch": dict(SPARSE), "pooled": cells(pooled_cell),
            "per_match": {match_id: cells(cell) for match_id, cell in per_match_cells.items()}}


def test_hold_stats_curves():
    """Each number is (agree + 1) / (frames + 2); a fold leaves its match out of the counts."""
    result = result_of([8, 6], {"a": [5, 5], "b": [3, 1]})
    curves = hold_stats.curves(result)
    assert curves["format"] == 1 and curves["kind"] == "hold_curve" and curves["bins_s"] == result["bins_s"]
    assert curves["profile"] == "generic" and curves["seeds"] == [0, 1] and curves["matches"] == ["a", "b"]
    assert curves["counts"] == result["pooled"] and sorted(curves["folds"]) == ["a", "b"]
    assert abs(curves["full"]["no_gap"][0] - 0.7) < 1e-12
    assert abs(curves["folds"]["a"]["no_gap"][0] - 0.4) < 1e-12
    assert abs(curves["folds"]["b"]["no_gap"][0] - 6 / 7) < 1e-12
    assert curves["full"]["no_gap"][1] == 0.5 and curves["full"]["gap"] == [0.5] * 10
    assert all(len(row) == 10 for curve in [curves["full"]] + list(curves["folds"].values()) for row in curve.values())
    json.loads(hold_stats.dumps(curves))                 # strict JSON: the infinity of a touch is written as "inf"
    wide = hold_stats.curves(result)["touch"]
    assert wide == dict(SPARSE, max_rel_speed="inf") and curves["touch"] == wide


def quietly(function, *arguments):
    """(result, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments)
    return result, buffer.getvalue()


def test_hold_stats_main_writes_and_verifies():
    """The commands the controller runs, on seven synthetic matches under the DFL ids and the clean profile."""
    matches = {match_id: random_match(match_id, seed=seed, rows=(700, 600)) for seed, match_id
               in enumerate(sources.MATCH_IDS)}
    original = hold_stats.TOUCH
    hold_stats.TOUCH = SPARSE
    try:
        with tempfile.TemporaryDirectory() as root, loading(lambda match_id: matches[match_id]):
            path = os.path.join(root, "models", "hold.json")
            flags = ["--profile", "clean", "--seeds", "1", "--out", path]
            code, text = quietly(hold_stats.main, flags + ["--write"])
            assert code == 0 and "wrote" in text and "gate" in text and os.path.exists(path), text
            with open(path, encoding="utf-8") as handle:
                artifact = json.load(handle)
            assert sorted(artifact["folds"]) == sorted(sources.MATCH_IDS) and artifact["profile"] == "clean"
            assert artifact["touch"] == dict(SPARSE, max_rel_speed="inf"), artifact["touch"]
            assert any(cell[0] for row in artifact["counts"].values() for cell in row), artifact["counts"]
            from pipeline.selftest import touch_scene
            run = touch_scene(20)
            run.inputs.game = sources.MATCH_IDS[0]
            controlled_touch.compute(run, confidence=0.9, hold_curve=path, **SPARSE)
            code, text = quietly(hold_stats.main, flags + ["--verify"])
            assert code == 0 and "verified" in text, text
            artifact["full"]["no_gap"][0] += 1e-9
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(hold_stats.dumps(artifact))
            code, text = quietly(hold_stats.main, flags + ["--verify"])
            assert code == 1 and "FAILED" in text and "full" in text, text
            code, text = quietly(hold_stats.main, flags + ["--verify", "--out", os.path.join(root, "missing.json")])
            assert code == 1 and "FAILED" in text and "missing.json" in text and "hold_stats.py" in text, text
    finally:
        hold_stats.TOUCH = original
