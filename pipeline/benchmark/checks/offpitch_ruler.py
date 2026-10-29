"""Checks of the past-the-line ruler and our match's target (pipeline/benchmark/offpitch.py, P18).

line_stats is worked out on frames built by hand, with the figures written down from the construction. The kick-off
spans are compared with calibration.measure's documented rule on a stub. test_data_offpitch_target_is_cached_and_
aggregate_only runs on our match with --data and prints the share and the visible frames, nothing else.
"""

import json
import math
import os
import types

import numpy as np

from pipeline.benchmark import calibration, offpitch
from pipeline.selftest import expect_error

# Params
PITCH = (100.0, 60.0)
DOCUMENTED_KEYS = {"version", "key", "visible_frames", "beyond_frames", "beyond_line_share", "beyond_m_quantiles"}


def _frames():
    """(present, xy, blocks): 10 frames in one block on a 100 x 60 m pitch.

    frame 0  inside (50, 30)              line distance 30
    frame 1  x = -0.5                     past a line by 0.5
    frame 2  x = 0.0                      exactly on a line: visible, not beyond
    frame 3  y = 61.0                     past the far line by 1.0
    frame 4  flag False, NaN position     not visible
    frame 5  inside (20, 10)              line distance 10
    frame 6  x = 104.0                    past the far end by 4.0
    frame 7  flag True, y NaN             not visible (a position with a coordinate missing is absent)
    frame 8  inside (60, 40)              line distance 20
    frame 9  inside (5, 5)                line distance 5
    Visible: 0, 1, 2, 3, 5, 6, 8, 9 = 8; beyond: 1, 3, 6 = 3; share 0.375; distances 0.5, 1.0, 4.0.
    """
    xy = np.array([(50, 30), (-0.5, 30), (0.0, 30), (50, 61.0), (np.nan, np.nan), (20, 10), (104.0, 30),
                   (50, np.nan), (60, 40), (5, 5)], dtype=np.float64)
    present = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    return present, xy, [(1, 0, 10)]


def test_offpitch_line_stats_on_hand_built_frames():
    present, xy, blocks = _frames()
    stats = offpitch.line_stats(present, blocks, xy, PITCH)
    assert stats["visible_frames"] == 8, stats
    assert stats["beyond_frames"] == 3, stats
    assert stats["beyond_line_share"] == 0.375, stats
    q = stats["beyond_m_quantiles"]
    assert list(q) == [name for name, _ in offpitch.QUANTILES], q
    # distances 0.5, 1.0, 4.0: linear interpolation at 0.10, 0.25, 0.5, 0.75, 0.90
    assert q["median"] == 1.0 and q["p25"] == 0.75 and q["p75"] == 2.5, q
    assert math.isclose(q["p10"], 0.6) and math.isclose(q["p90"], 3.4), q
    inside = np.where(np.isfinite(xy), 30.0, xy)
    nothing = offpitch.line_stats(present, blocks, inside, PITCH)
    assert nothing["beyond_frames"] == 0 and nothing["beyond_line_share"] == 0.0, nothing
    assert nothing["beyond_m_quantiles"] == {}, nothing
    none_visible = offpitch.line_stats(np.zeros(10, dtype=bool), blocks, np.full((10, 2), np.nan), PITCH)
    assert none_visible["visible_frames"] == 0 and none_visible["beyond_frames"] == 0, none_visible
    assert math.isnan(none_visible["beyond_line_share"]) and none_visible["beyond_m_quantiles"] == {}


def test_offpitch_line_stats_only_counts_inside_the_blocks():
    present, xy, _ = _frames()
    # Blocks 0-3 (visible 0, 1, 2: 3, beyond 1) and 6-10 (visible 6, 8, 9: 3, beyond 1). Frame 3 (beyond), 5 and 4
    # are outside both. Given out of order: the blocks are sorted.
    stats = offpitch.line_stats(present, [(2, 6, 10), (1, 0, 3)], xy, PITCH)
    assert stats["visible_frames"] == 6 and stats["beyond_frames"] == 2, stats
    assert stats["beyond_line_share"] == 2 / 6, stats
    assert stats["beyond_m_quantiles"]["median"] == 2.25, stats     # distances 0.5 and 4.0
    only_outside = offpitch.line_stats(present, [(1, 4, 6)], xy, PITCH)
    assert only_outside["visible_frames"] == 1 and only_outside["beyond_frames"] == 0, only_outside


def test_offpitch_line_stats_refuses_what_would_make_it_meaningless():
    present, xy, blocks = _frames()
    expect_error(ValueError, lambda: offpitch.line_stats(present[:9], blocks, xy, PITCH), "ball_present", "ball_xy_m")
    expect_error(ValueError, lambda: offpitch.line_stats(present, blocks, xy[:, :1], PITCH), "ball_xy_m")
    expect_error(ValueError, lambda: offpitch.line_stats(present.astype(int), blocks, xy, PITCH), "ball_present")
    stray = present.copy()
    stray[0] = False                                  # a flag of absent over a finite position
    expect_error(ValueError, lambda: offpitch.line_stats(stray, blocks, xy, PITCH), "ball_present", "ball_xy_m")
    expect_error(ValueError, lambda: offpitch.line_stats(present, [(1, 0, 11)], xy, PITCH), "blocks")
    expect_error(ValueError, lambda: offpitch.line_stats(present, [(1, 4, 4)], xy, PITCH), "blocks")
    expect_error(ValueError, lambda: offpitch.line_stats(present, [(1, 0, 6), (2, 5, 10)], xy, PITCH), "blocks")
    expect_error(ValueError, lambda: offpitch.line_stats(present, [], xy, PITCH), "blocks")
    expect_error(ValueError, lambda: offpitch.line_stats(present, blocks, xy, (100.0, -1.0)), "pitch")


def test_offpitch_kickoff_spans_match_calibrations():
    # calibration.measure's rule (its spans are a local variable): one span per graded block, (half, kickoff, stop).
    inputs = types.SimpleNamespace(graded=[(1, 50, 280), (2, 320, 560)], kickoffs={1: 60, 2: 340})
    expected = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
    assert offpitch.kickoff_spans(inputs) == expected == [(1, 60, 280), (2, 340, 560)]
    assert calibration.measure.__doc__ and "kick-off" in calibration.measure.__doc__
    missing = types.SimpleNamespace(graded=[(1, 50, 280)], kickoffs={})
    expect_error(ValueError, lambda: offpitch.kickoff_spans(missing), "kickoffs", "half 1")
    outside = types.SimpleNamespace(graded=[(1, 50, 280)], kickoffs={1: 280})
    expect_error(ValueError, lambda: offpitch.kickoff_spans(outside), "kick-off", "half 1")


def test_offpitch_key_follows_the_kickoff_spans():
    def stub(graded, kickoffs):
        return types.SimpleNamespace(graded=graded, kickoffs=kickoffs, pitch=PITCH, fps=25.0,
                                     atd_manifest={"sources": {"atd": "a"}})
    base = offpitch._key(stub([(1, 50, 280), (2, 320, 560)], {1: 60, 2: 340}))
    assert base == offpitch._key(stub([(1, 50, 280), (2, 320, 560)], {1: 60, 2: 340}))
    other_kickoff = offpitch._key(stub([(1, 50, 280), (2, 320, 560)], {1: 61, 2: 340}))
    other_block = offpitch._key(stub([(1, 50, 281), (2, 320, 560)], {1: 60, 2: 340}))
    assert len({base, other_kickoff, other_block}) == 3, (base, other_kickoff, other_block)
    assert offpitch.target_path("d", base) != offpitch.target_path("d", other_kickoff)


def test_offpitch_line_stats_errors_name_the_function():
    present, xy, blocks = _frames()
    expect_error(ValueError, lambda: offpitch.line_stats(present.astype(int), blocks, xy, PITCH),
                 "offpitch.line_stats", "ball_present")
    expect_error(ValueError, lambda: offpitch.line_stats(present, [(1, 0, 11)], xy, PITCH),
                 "offpitch.line_stats", "blocks")
    expect_error(ValueError, lambda: offpitch.line_stats(present, blocks, xy, (1.0, -1.0)),
                 "offpitch.line_stats", "pitch")


def test_data_offpitch_target_is_cached_and_aggregate_only():
    """On our match: built once, then read from the cache; the file holds the documented keys and no frame."""
    from pipeline.inputs import Inputs
    inputs = Inputs()
    first = offpitch.target(inputs=inputs, rebuild=True)
    path = offpitch.target_path(inputs.data_dir, first["key"])
    assert os.path.isfile(path), path
    real = offpitch.line_stats

    def refuse(*args, **kwargs):
        raise AssertionError("the second call measured again instead of reading the cache")
    offpitch.line_stats = refuse
    try:
        second = offpitch.target(inputs=inputs)
    finally:
        offpitch.line_stats = real
    assert second == first, "the cache did not read back as it was written"
    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert set(saved) == DOCUMENTED_KEYS, sorted(saved)
    assert all(isinstance(v, (int, float, str, type(None))) for k, v in saved.items() if k != "beyond_m_quantiles")
    assert set(saved["beyond_m_quantiles"]) <= {name for name, _ in offpitch.QUANTILES}
    assert saved["visible_frames"] > 0 and 0.0 <= saved["beyond_line_share"] <= 1.0, saved
    print("    our match: {} visible ball frames on the spans, share past a line {:.4f}".format(
        saved["visible_frames"], saved["beyond_line_share"]))
