"""Checks of whistle_stats (pipeline/benchmark/whistle_stats.py): P10's label-free statistics, on hand-built arrays.

The shares, the circular control, the lags to the edge of a stoppage, the bootstrap and the gate are worked out here on
detections and decoded stoppages whose every figure is known by construction, and the printout is held to its rule:
figures and no frame (CLAUDE.md, hard rules). No check reads data.
"""

import numpy as np

from pipeline.benchmark import whistle_stats

# Params: four-digit edges, so that a printed frame would be found
BLOCKS = [(1, 1000, 1600), (2, 2000, 2600)]
N = 3000
DEAD_RUNS = [(1100, 1200), (1500, 1600), (2300, 2400)]


def dead_of(runs):
    dead = np.zeros(N, dtype=bool)
    for start, stop in runs:
        dead[start:stop] = True
    return dead


def test_whistle_stats_near_and_control():
    dead = dead_of(DEAD_RUNS)
    # window [w - 25, w + 50]: 1050 reaches 1100; 1049 does not; 1224 sees the run's last frame, 1199; 1250 starts at
    # 1225 and sees nothing; 1400 sees nothing; 1580 is inside a run; 2260 reaches 2300; 2430's window starts at 2405
    w = np.array([1050, 1049, 1224, 1250, 1400, 1580, 2260, 2299, 2430], dtype=np.int64)
    near = whistle_stats.near_stoppage(w, dead, BLOCKS)
    assert near.tolist() == [True, False, True, False, False, True, True, True, False], near.tolist()
    # a stoppage past the block's edge is not near: 1595 + 50 reaches 1645, where a D run starts, outside the block
    edge = dead_of([(1610, 1650)])
    assert not whistle_stats.near_stoppage(np.array([1595]), edge, BLOCKS)[0], "a window stops at its block's edge"


def test_whistle_stats_control_shifts_inside_each_half():
    shifted = whistle_stats.shift_within_halves(np.array([1100, 1590, 2100, 2590]), BLOCKS, 250)
    assert shifted.tolist() == [1350, 1240, 2350, 2240], shifted.tolist()
    for offset in range(125, 1501, 125):
        out = whistle_stats.shift_within_halves(np.array([1000, 1599, 2000, 2599]), BLOCKS, offset)
        assert all(1000 <= x < 1600 for x in out[:2]) and all(2000 <= x < 2600 for x in out[2:]), "a half's own"


def test_whistle_stats_feasibility_worked_out():
    dead = dead_of(DEAD_RUNS)
    w = np.array([1050, 1224, 1400, 1580, 2260, 2430], dtype=np.int64)
    out = whistle_stats.feasibility(w, dead, BLOCKS, offsets_s=(10, 20))
    assert out["per_half"] == {1: 4, 2: 2}, out["per_half"]
    assert out["share_near"] == 4 / 6, out["share_near"]
    expected = []
    for off in (10, 20):
        moved = whistle_stats.shift_within_halves(w, BLOCKS, off * 25)
        expected.append(float(np.mean([bool(dead[max(x - 25, s):min(x + 51, e)].any())
                                       for x in moved for _, s, e in BLOCKS if s <= x < e])))
    assert abs(out["control_share_near"] - np.mean(expected)) < 1e-12, out["control_share_near"]
    assert abs(out["ratio"] - out["share_near"] / out["control_share_near"]) < 1e-12
    # lags: signed frames to the nearest edge of a D run (its first frame, or the first frame after it)
    lags = sorted(out["lags"])
    assert lags == [-80, -30, -24, 40, 50, 100], lags
    assert set(out["lag_to_edge"]) == {"q1", "median", "q3"}


def test_whistle_stats_bar_bootstrap_is_seeded_and_ordered():
    dead = dead_of(DEAD_RUNS)
    w = np.array([1050, 1224, 1400, 1580, 2260, 2430] * 5, dtype=np.int64)
    one, two = whistle_stats.bar(w, dead, BLOCKS, draws=200), whistle_stats.bar(w, dead, BLOCKS, draws=200)
    assert one == two, "seeded"
    assert one["low"] <= one["share_near"] <= one["high"], one
    assert one["share_near"] == 4 / 6


def test_whistle_stats_gate_and_bar_rules():
    ok = {"per_half": {1: 30, 2: 40}, "share_near": 0.6, "ratio": 2.5}
    assert whistle_stats.gate(ok)["pass"] is True
    for change in ({"per_half": {1: 9, 2: 40}}, {"per_half": {1: 30, 2: 121}}, {"share_near": 0.49}, {"ratio": 1.9}):
        assert whistle_stats.gate({**ok, **change})["pass"] is False, change
    assert whistle_stats.gate({**ok, "ratio": None})["pass"] is False, "no control share: no ratio, no pass"
    good = {"share_near": 0.71, "low": 0.61, "high": 0.8, "ratio": 1.6}
    assert whistle_stats.bar_verdict(good)["pass"] is True
    for change in ({"share_near": 0.69}, {"low": 0.59}, {"ratio": 1.4}):
        assert whistle_stats.bar_verdict({**good, **change})["pass"] is False, change


def test_whistle_stats_prints_aggregates_only():
    dead = dead_of(DEAD_RUNS)
    w = np.array([1050, 1224, 1400, 1580, 2260, 2430], dtype=np.int64)
    out = whistle_stats.feasibility(w, dead, BLOCKS)
    text = "\n".join(whistle_stats.render_feasibility(out, whistle_stats.gate(out)))
    numbers = [token for token in text.replace("\n", " ").replace(",", " ").split() if token.isdigit()]
    scene_frames = {str(frame) for frame in range(1000, N)}
    assert not scene_frames & set(numbers), "a frame of the scene is printed: {}".format(scene_frames & set(numbers))
    assert "share" in text, "the figures are printed"
    assert "lags" not in whistle_stats.public(out), "the per-detection lags are not written to the report"


def test_whistle_stats_undefined_ratio_and_no_detection_are_recorded_fails():
    assert whistle_stats._ratio(0.5, 0.0) is None, "a zero control has no ratio"
    assert whistle_stats._ratio(0.5, 0.25) == 2.0
    good = {"share_near": 0.9, "low": 0.8, "high": 0.95, "ratio": None}
    assert whistle_stats.bar_verdict(good)["pass"] is False, "no ratio, no pass"
    dead = dead_of(DEAD_RUNS)
    none = np.array([], dtype=np.int64)
    out = whistle_stats.feasibility(none, dead, BLOCKS)
    assert out["per_half"] == {1: 0, 2: 0} and out["n"] == 0, out
    assert out["share_near"] is None and out["ratio"] is None and out["lag_to_edge"]["median"] is None
    verdict = whistle_stats.gate(out)
    assert verdict["pass"] is False and verdict["checks"]["per_half"] is False, verdict
    text = " ".join(whistle_stats.render_feasibility(out, verdict))
    assert "FAIL" in text and "n/a" in text, text
    out = whistle_stats.bar(none, dead, BLOCKS)
    assert out["n"] == 0 and out["low"] is None, out
    assert whistle_stats.bar_verdict(out)["pass"] is False
    assert "FAIL" in " ".join(whistle_stats.render_bar(out, whistle_stats.bar_verdict(out)))
