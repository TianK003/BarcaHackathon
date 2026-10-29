"""Checks of the past-the-line ruler, our match's target, the one-time fit of the offpitch profile's two constants
(pipeline/benchmark/offpitch.py, P18, R015), and the offpitch profile's fidelity (pipeline/benchmark/fidelity.py).

line_stats is worked out on frames built by hand, with the figures written down from the construction. The kick-off
spans are compared with calibration.measure's documented rule on a stub. test_data_offpitch_target_is_cached_and_
aggregate_only runs on our match with --data and prints the share and the visible frames, nothing else.

The fit is run on a match built by hand (hand_built_match): three excursions past a line at known depths, our gaps at
known rows and their thinning uniforms set by the check (fixed_uniforms), so that the share past a line at every margin
and the missing share at every drop probability can be written down, and the record's m, q and refit with them. The
sweeps are then held, with the real uniforms, against the profile's own masks measured by the ruler one grid value at
a time. The fidelity checks run the offpitch report on synthetic matches with the two constants set for the check's
duration (pinned) and a target of ours made up (MADE_UP_TARGET): the centre of the new gate is offpitch.target()'s,
injected as the Calibration is, so that no check reads a data directory.
"""

import contextlib
import dataclasses
import io
import json
import math
import os
import tempfile
import types

import numpy as np

from pipeline.benchmark import calibration, degrade, dfl, fidelity, hazard, offpitch, run
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import truth as truths
from pipeline.benchmark.checks.common import (loading, random_match, read, steady_match, strict_json,
                                              synthetic_calibration, tracking_calibration)
from pipeline.contracts import D
from pipeline.selftest import expect_error, quiet

# Params
PITCH = (100.0, 60.0)
DOCUMENTED_KEYS = {"version", "key", "visible_frames", "beyond_frames", "beyond_line_share", "beyond_m_quantiles"}
# The hand-built match of the fit (hand_built_match): two halves of FIT_ROWS rows on a FIT_PITCH pitch, each from its
# kick-off with the ball in the middle (25 m from a line) except three excursions past a line, (first row, stop row,
# axis, metres past), rows NO_BALL without a ball, and OUR_GAPS as our mask of both halves: an edge gap (never dropped)
# and two interior ones, G1 and G2, whose thinning uniforms are UNIFORMS.
FIT_PITCH = (100.0, 50.0)
FIT_ROWS = 200
EXCURSIONS = ((30, 40, 0, 0.55), (50, 70, 0, 1.25), (80, 110, 1, 3.05))
NO_BALL = (195, 200)
OUR_GAPS = ((0, 10), (120, 140), (150, 180))
UNIFORMS = (0.0123, 0.2468)
# The record of a fit, written out here so that a key the fit forgets, or adds, is seen.
FIT_KEYS = {"target_share", "missing_target", "margin_m", "drop_prob", "share_at_fit", "missing_at_fit", "refit",
            "first_margin_m", "share_at_first_margin", "sweep_m", "sweep_q", "sweep_m_refit", "counts_at_fit",
            "draws_at_fit", "seeds", "matches", "calibration"}
# The gates of tracking and context as they are, written out (fidelity.GATES must not move them).
TRACKING_GATES = ["missing_share", "median_gap_s", "present_share_22", "present_mean", "present_share_le14",
                  "wrong_team_share", "tracklet_median_s", "speed_nan_share"]
CONTEXT_GATES = TRACKING_GATES + ["near_line_share"]
REPORT_KEYS = ["profile", "seeds", "matches", "calibration", "git", "gates", "checks", "statistics", "draws",
               "players_draws", "ball_draws", "full_strength", "passed", "warnings", "notes"]
OFFPITCH_INFO = {"target", "line", "beyond_m_quantiles", "short_gaps", "gaps_per_match", "dead_ball_visible_share",
                 "draws"}
# Our match's target as offpitch.target() returns it, made up: the fidelity checks inject it.
MADE_UP_TARGET = {"version": 1, "key": "0123456789abcdef", "visible_frames": 1000, "beyond_frames": 40,
                  "beyond_line_share": 0.04,
                  "beyond_m_quantiles": {"p10": 0.1, "p25": 0.3, "median": 0.7, "p75": 1.5, "p90": 2.6}}
TEAMS_22 = (0,) * 11 + (1,) * 11       # a full squad: the count gates of tracking need frames at full strength


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


# -- the distances past a line, and one line distance everywhere ------------------------------------------------------

def test_offpitch_beyond_distances_pool_to_line_stats():
    """line_stats is beyond_distances summarised (line_summary), and parts measured apart pool by sums and by laying
    their distances end to end: what fidelity.py does over matches and seeds.

    On the hand-built frames: 8 visible, and the distances 0.5, 1.0 and 4.0 past a line, in time order, read-only. The
    blocks 0-3 and 6-10 measured apart and pooled give line_stats of the two together, quantiles included."""
    present, xy, blocks = _frames()
    visible, past = offpitch.beyond_distances(present, blocks, xy, PITCH)
    assert visible == 8 and past.tolist() == [0.5, 1.0, 4.0] and not past.flags.writeable
    assert offpitch.line_summary(visible, past) == offpitch.line_stats(present, blocks, xy, PITCH)
    first = offpitch.beyond_distances(present, [(1, 0, 3)], xy, PITCH)
    second = offpitch.beyond_distances(present, [(2, 6, 10)], xy, PITCH)
    pooled = offpitch.line_summary(first[0] + second[0], np.concatenate([first[1], second[1]]))
    assert pooled == offpitch.line_stats(present, [(1, 0, 3), (2, 6, 10)], xy, PITCH), pooled
    nothing = offpitch.line_summary(0, np.zeros(0))
    assert nothing["beyond_frames"] == 0 and math.isnan(nothing["beyond_line_share"])
    assert nothing["beyond_m_quantiles"] == {}
    expect_error(ValueError, lambda: offpitch.beyond_distances(present.astype(int), blocks, xy, PITCH),
                 "offpitch.beyond_distances", "ball_present")


def test_offpitch_line_distance_is_one_quantity_in_four_places():
    """degrade.line_distance_m (the profile's), hazard.features' line_distance_m and beyond_line columns (the context
    profile's and the Calibration's hazard), calibration.ball_stats' gap_onset_line_m (the near-line share) and
    offpitch's ruler are the same numbers, to the bit, on the same frames.

    The fit sets a constant of the profile by comparing one of its statistics with one of the Calibration's, so the
    formula, which is written in four places, must not drift in any of them. On a random match whose ball wanders up to
    2% of the pitch past its lines, with a fifth of its rows without a ball, on its own 105 x 68 m pitch and again on
    90 x 45 m: the metres follow the match's pitch in all four."""
    line_column = hazard.FEATURES.index("line_distance_m")
    beyond_column = hazard.FEATURES.index("beyond_line")
    for pitch in ((105.0, 68.0), (90.0, 45.0)):
        match = dataclasses.replace(random_match("RND001", seed=3), pitch=pitch)
        scale = np.array(pitch)
        present = ~np.isnan(match.ball_xy[:, 0])
        onsets_seen = beyond_seen = 0
        for half, _, stop in match.graded:
            kickoff = match.kickoffs[half]
            ball = match.ball_xy[kickoff:stop]
            line = degrade.line_distance_m(ball, pitch)
            table = hazard.features(ball * scale, match.players_xy[kickoff:stop] * scale, pitch, match.fps, kickoff,
                                    kickoff)
            assert np.array_equal(table[:, line_column], line, equal_nan=True), (pitch, half)
            seen = ~np.isnan(line)
            assert table[seen, beyond_column].tolist() == (line[seen] < 0).astype(np.float64).tolist()
            # The onset of an interior gap is the last frame with a ball before it, found here one row at a time.
            onsets, row = [], 0
            while row < seen.size:
                if seen[row]:
                    row += 1
                    continue
                first = row
                while row < seen.size and not seen[row]:
                    row += 1
                if first > 0 and row < seen.size:
                    onsets.append(first - 1)
            stats = calibration.ball_stats(present, [(half, kickoff, stop)], match.ball_xy * scale, pitch, match.fps)
            assert stats["gap_onset_line_m"].tolist() == line[onsets].tolist(), (pitch, half)
            visible, past = offpitch.beyond_distances(present, [(half, kickoff, stop)], match.ball_xy * scale, pitch)
            assert visible == int(seen.sum()) and past.tolist() == (-line[seen & (line < 0)]).tolist()
            onsets_seen += len(onsets)
            beyond_seen += past.size
        assert onsets_seen > 20 and beyond_seen > 20, "a check of nothing: {} onsets, {} past a line".format(
            onsets_seen, beyond_seen)


# -- the one-time fit of the margin and the drop probability ----------------------------------------------------------

def hand_built_match():
    """The fit's match: steady_match's players and a ball built by hand on a 100 x 50 m pitch, alike in both halves.

    From each half's kick-off (its first row) the ball is in the middle, 25 m from the nearest line, except on the
    excursions of EXCURSIONS: E1, rows 30-39, 0.55 m past a goal line; E2, rows 50-69, 1.25 m past it; E3, rows 80-109,
    3.05 m past a touch line; and rows 195-199 have no ball. An excursion starts past a margin below its depth and the
    next row on the pitch ends it, so it is hidden whole at a margin below its depth and shown whole at one at or above
    it."""
    base = steady_match("RND001", 0, rows=(FIT_ROWS, FIT_ROWS), substitutions=False)
    ball = np.array(base.ball_xy)
    for _, start, stop in base.graded:
        ball[start:stop] = 0.5
        for first, last, axis, depth in EXCURSIONS:
            ball[start + first:start + last, axis] = -depth / FIT_PITCH[axis]
        ball[start + NO_BALL[0]:start + NO_BALL[1]] = np.nan
    ball.setflags(write=False)
    return dataclasses.replace(base, ball_xy=ball, pitch=FIT_PITCH)


def hand_built_calibration(missing_share):
    """A Calibration whose mask of both halves is OUR_GAPS over FIT_ROWS rows, as long as a half, so that the transplant
    reads it from row 0 whichever half it draws; and whose missing share, the target of q, is `missing_share`."""
    mask = np.zeros(FIT_ROWS, dtype=bool)
    for first, last in OUR_GAPS:
        mask[first:last] = True
    mask.setflags(write=False)
    cal = synthetic_calibration()
    return dataclasses.replace(cal, ball_masks={1: mask, 2: mask},
                               scalars={**cal.scalars, "missing_share": missing_share})


@contextlib.contextmanager
def fixed_uniforms(values=UNIFORMS):
    """degrade.thinning_uniforms giving `values`, one per interior gap in time order, to every half and seed, for the
    duration; put back afterwards. The fit and the profile's masks both read it, so they go on agreeing."""
    real = degrade.thinning_uniforms

    def drawn(view, half, seed, n_gaps):
        assert n_gaps == len(values), (n_gaps, values)
        out = np.array(values, dtype=np.float64)
        out.setflags(write=False)
        return out
    degrade.thinning_uniforms = drawn
    try:
        yield
    finally:
        degrade.thinning_uniforms = real


@contextlib.contextmanager
def pinned(margin_m, drop_prob, fit=None):
    """degrade's OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB and OFFPITCH_FIT set for the duration (the profile refuses to run
    while they are None), then put back: every check here holds whether they are pinned or not."""
    saved = degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT
    record = {"set_by": "a check"} if fit is None and margin_m is not None else fit
    degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT = margin_m, drop_prob, record
    try:
        yield
    finally:
        degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT = saved


def hand_fit(target_share, missing_share, seeds=2):
    """offpitch.fit on the hand-built match and Calibration, with the uniforms of UNIFORMS."""
    with fixed_uniforms():
        return offpitch.fit(hand_built_calibration(missing_share), target_share, seeds=seeds,
                            matches={"RND001": hand_built_match()}, log=quiet)


def plateaus(grid, steps):
    """[[value, share]] over a grid whose share is `share` from each (start, share) of `steps` on (ascending), rounded
    as a record's are."""
    return [[value, round([share for start, share in steps if value >= start][-1], 9)] for value in grid.tolist()]


def measured_at(matches, cal, margin_m, drop_prob, seeds):
    """{"share", "missing_share"} of the profile's ball at (margin_m, drop_prob), pooled over matches and seeds by sums.

    The ball as degrade() makes it for offpitch (offpitch_masks taken out of DFL's by _remove_ball) and measured by the
    ruler (line_stats) and calibration.ball_stats on each half from its kick-off, one grid value at a time."""
    visible = beyond = frames = missing = 0
    for match in matches.values():
        view = degrade.tracking_view(match)
        clean = match_inputs.clean_atd(view)
        spans = [(half, match.kickoffs[half], stop) for half, _, stop in match.graded]
        metres = np.array(match.pitch)
        for seed in range(seeds):
            masks, _ = degrade.offpitch_masks(view, cal, seed, margin_m, drop_prob)
            ball_xy, _ = degrade._remove_ball(view, clean, masks)
            present = ~np.isnan(ball_xy[:, 0])
            line = offpitch.line_stats(present, spans, ball_xy * metres, match.pitch)
            ball = calibration.ball_stats(present, spans, ball_xy * metres, match.pitch, match.fps)
            visible, beyond = visible + line["visible_frames"], beyond + line["beyond_frames"]
            frames, missing = frames + ball["frames"], missing + ball["missing_frames"]
    return {"share": round(beyond / visible, 9), "missing_share": round(missing / frames, 9)}


def wave_match(match_id="RND001", seed=0, rows=(240, 200), teams=(0, 0, 0, 1, 1, 1)):
    """steady_match whose ball goes past the lines and comes back, with one row in 20 of each half without a ball.

    x = 0.5 + 0.53 sin(2 pi t / 130 + 1) and y = 0.5 + 0.56 sin(2 pi t / 80), plus a jitter of 0.004 (normalised) a
    row: on a 105 x 68 m pitch up to about 3 m past a goal line and 4 m past a touch line, several times a half."""
    match = steady_match(match_id, seed, rows, teams=teams, substitutions=False)
    rng = np.random.default_rng([41, seed])
    ball = np.array(match.ball_xy)
    for _, start, stop in match.graded:
        t = np.arange(stop - start)
        ball[start:stop, 0] = 0.5 + 0.53 * np.sin(2 * np.pi * t / 130.0 + 1.0) + rng.normal(0, 0.004, t.size)
        ball[start:stop, 1] = 0.5 + 0.56 * np.sin(2 * np.pi * t / 80.0) + rng.normal(0, 0.004, t.size)
        ball[start + rng.choice(t.size, size=t.size // 20, replace=False)] = np.nan
    ball.setflags(write=False)
    return dataclasses.replace(match, ball_xy=ball)


def wave_matches(teams=(0, 0, 0, 1, 1, 1)):
    return {"RND001": wave_match("RND001", 0, teams=teams), "RND002": wave_match("RND002", 1, (260, 190), teams)}


def test_offpitch_fit_on_a_hand_built_match_lands_on_the_documented_values():
    """The share past a line at every margin and the missing share at every drop probability, written down from the
    construction, and the record's m, q and refit with them.

    Per half (both alike, and the seeds alike: the uniforms are fixed) 195 rows have a ball. At q = 0 our gaps hide 60
    of them (10 + 20 + 30) and the margin the excursions deeper than it: all three below 0.55 m (60 rows), E2 and E3
    from 0.6 to 1.2 m (50), E3 from 1.3 to 3.0 m (30), none from 3.1 m. So 75, 85, 105 and 135 rows are visible, of
    which 0, 10, 30 and 60 are past a line: the shares 0, 10/85, 2/7 and 4/9. A target of 0.118 is closest to 10/85
    (0.11765), on every margin from 0.6 to 1.2 m: m = 0.6. At m = 0.6 the missing rows are 200 - (195 - 50 - our gaps'
    rows): 115 while G1 (uniform 0.0123) is kept, up to q = 0.010; 95 from q = 0.015 to 0.245; 65 from q = 0.250,
    G2's uniform 0.2468 being below it too. A missing target of 0.58 is closest to 115/200 = 0.575: q = 0. The share at
    (0.6, 0) is 10/85, 0.00035 from the target: no refit. The shares the record reports are the ruler's, on the
    degraded balls, and its counts four times one half's (two halves, two seeds)."""
    record = hand_fit(0.118, 0.58)
    assert set(record) == FIT_KEYS, sorted(set(record) ^ FIT_KEYS)
    assert (record["margin_m"], record["drop_prob"], record["refit"]) == (0.6, 0.0, False), record
    assert record["first_margin_m"] == 0.6 and record["sweep_m_refit"] == []
    assert record["share_at_fit"] == record["share_at_first_margin"] == round(10 / 85, 9)
    assert record["missing_at_fit"] == 0.575
    assert (record["target_share"], record["missing_target"]) == (0.118, 0.58)
    assert record["sweep_m"] == plateaus(offpitch.M_GRID, [(0.0, 0.0), (0.6, 10 / 85), (1.3, 30 / 105),
                                                           (3.1, 60 / 135)])
    assert record["sweep_q"] == plateaus(offpitch.Q_GRID, [(0.0, 115 / 200), (0.015, 95 / 200), (0.25, 65 / 200)])
    assert record["counts_at_fit"] == {"frames": 800, "missing_frames": 460, "visible_frames": 340,
                                       "beyond_frames": 40}, record["counts_at_fit"]
    assert record["draws_at_fit"] == {"interior_gaps": 8, "gaps_dropped": 0, "frames_restored": 0,
                                      "offpitch_stretches": 8, "offpitch_frames": 200, "missing_frames": 440,
                                      "rows": 800}, record["draws_at_fit"]
    assert record["seeds"] == [0, 1] and record["matches"] == ["RND001"]
    assert record["calibration"] == {"sha256": calibration.digest(hand_built_calibration(0.58)),
                                     "version": calibration.CALIBRATION_VERSION}
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_offpitch_fit_breaks_ties_to_the_smaller_value():
    """Where several grid values give the same share, or shares exactly as far from the target, the smaller value wins.

    On the hand-built match every margin from 0.6 to 1.2 m gives 10/85, and the fit takes 0.6 (the previous check);
    every drop probability from 0.015 to 0.245 gives 95/200, and a missing target of 0.48 takes 0.015. The rule itself,
    on counts written by hand and compared exactly (fractions): equal shares take the first; 1/4 and 3/4 are exactly
    1/4 from a target of 1/2, and whichever comes first is taken; a share closer by 2^-52 wins over an earlier one."""
    first = hand_fit(0.118, 0.58)
    assert first["margin_m"] == 0.6 and first["sweep_m"][6][1] == first["sweep_m"][12][1], first["sweep_m"][5:14]
    second = hand_fit(0.2, 0.48)
    assert second["first_margin_m"] == 0.6 and second["drop_prob"] == 0.015
    assert second["sweep_q"][3][1] == second["sweep_q"][49][1] == 0.475, second["sweep_q"][2:51]
    assert offpitch._closest([(1, 2), (1, 2), (1, 2)], 0.5) == 0
    assert offpitch._closest([(1, 4), (3, 4)], 0.5) == 0
    assert offpitch._closest([(3, 4), (1, 4)], 0.5) == 0
    assert offpitch._closest([(0, 4), (1, 4), (3, 4)], 0.5) == 1, "1/4 and 3/4 tie, and 1/4 comes first"
    assert offpitch._closest([(1, 4), (3 * 2 ** 50 - 1, 2 ** 52)], 0.5) == 1, "closer by the least amount"


def test_offpitch_fit_refits_m_at_most_once():
    """m is refitted once, at the fitted q, when the share there is more than 0.005 off the target, and never again.

    Target 0.2, missing target 0.48. At q = 0 the closest share is 10/85 (0.0824 off; 2/7 is 0.0857 off): m = 0.6. At
    m = 0.6 q = 0.015 (the missing share 95/200 = 0.475). With G1 dropped 20 more rows are visible and none of them past
    a line: the share is 10/105 = 0.0952, 0.105 off the target, so m is refitted at q = 0.015, where the same counts
    give 0, 10/105, 30/125 and 60/155, and 30/125 = 0.24 is the closest: m = 1.3. There it stops. At (1.3, 0.015) the
    missing share is 75/200 = 0.375, and a second fit of q would take q = 0 (95/200 = 0.475, closer to 0.48), where the
    share is 2/7, 0.0857 off, which would send m back to 0.6: the two would go round. The record keeps (1.3, 0.015),
    says that it refitted, and holds the refit's sweep and the share that called for it. A fit whose share at (m, q) is
    within 0.005 of the target does not refit."""
    record = hand_fit(0.2, 0.48)
    assert (record["margin_m"], record["drop_prob"], record["refit"]) == (1.3, 0.015, True), record
    assert record["first_margin_m"] == 0.6 and record["share_at_first_margin"] == round(10 / 105, 9)
    assert record["share_at_fit"] == 0.24 and record["missing_at_fit"] == 0.375
    assert record["sweep_m_refit"] == plateaus(offpitch.M_GRID, [(0.0, 0.0), (0.6, 10 / 105), (1.3, 30 / 125),
                                                                 (3.1, 60 / 155)])
    assert record["sweep_q"] == plateaus(offpitch.Q_GRID, [(0.0, 115 / 200), (0.015, 95 / 200), (0.25, 65 / 200)])
    # Not twice: q is not fitted again at the new m, where q = 0 would have been closer.
    with fixed_uniforms():
        again = measured_at({"RND001": hand_built_match()}, hand_built_calibration(0.48), 1.3, 0.0, seeds=2)
    assert again == {"share": round(30 / 105, 9), "missing_share": 0.475}, again
    assert abs(again["missing_share"] - 0.48) < abs(record["missing_at_fit"] - 0.48)
    assert hand_fit(0.118, 0.58)["refit"] is False


def test_offpitch_fit_sweeps_measure_what_the_profile_builds():
    """Every entry of the sweeps is the profile's ball measured by the ruler at that grid value, with the real draws.

    Two matches whose ball goes past the lines, our made-up masks (a transplant from a drawn offset) and the real
    thinning uniforms, two seeds. The share of sweep_m at each margin (q = 0), the missing share of sweep_q at each drop
    probability (at the first margin) and the share of sweep_m_refit at each margin (at the fitted q) are measured_at's,
    one grid value at a time. At the fitted (m, q) degrade() itself, with the two constants set to them, gives the
    record's shares and counts. The fit is not a trivial one: some gaps are dropped and some stretches hidden."""
    matches, cal = wave_matches(), tracking_calibration()
    record = offpitch.fit(cal, 0.03, seeds=2, matches=matches, log=quiet)
    for margin, share in record["sweep_m"]:
        assert share == measured_at(matches, cal, margin, 0.0, 2)["share"], margin
    for drop_prob, share in record["sweep_q"]:
        assert share == measured_at(matches, cal, record["first_margin_m"], drop_prob, 2)["missing_share"], drop_prob
    for margin, share in record["sweep_m_refit"]:
        assert share == measured_at(matches, cal, margin, record["drop_prob"], 2)["share"], margin
    assert record["draws_at_fit"]["gaps_dropped"] > 0 and record["draws_at_fit"]["offpitch_stretches"] > 0, record
    assert 0.0 < record["margin_m"] < 10.0 and 0.0 < record["drop_prob"], record
    visible = beyond = frames = missing = 0
    with pinned(*offpitch.constants_from(record)):
        for match in matches.values():
            spans = [(half, match.kickoffs[half], stop) for half, _, stop in match.graded]
            for seed in range(2):
                atd, _ = degrade.degrade(degrade.tracking_view(match), "offpitch", cal, seed)
                metres = atd.ball_xy * np.array(match.pitch)
                line = offpitch.line_stats(atd.ball_present, spans, metres, match.pitch)
                ball = calibration.ball_stats(atd.ball_present, spans, metres, match.pitch, match.fps)
                visible, beyond = visible + line["visible_frames"], beyond + line["beyond_frames"]
                frames, missing = frames + ball["frames"], missing + ball["missing_frames"]
    assert record["counts_at_fit"] == {"frames": frames, "missing_frames": missing, "visible_frames": visible,
                                       "beyond_frames": beyond}
    assert record["share_at_fit"] == round(beyond / visible, 9)
    assert record["missing_at_fit"] == round(missing / frames, 9)


def test_offpitch_fit_refuses_what_it_cannot_measure():
    """No NaN and no infinity ever reaches a record: what would make one is refused, naming offpitch.fit and the
    argument. A target that is not a share, a Calibration that is not one or whose missing share is not a share, a
    number of seeds that is not a whole number of at least 1, matches that are not as asked, and a match on which no
    ball frame is visible (the share past a line would be 0 / 0)."""
    cal = hand_built_calibration(0.5)

    def fit(**changes):
        arguments = {"calibration": cal, "target_share": 0.1, "seeds": 1, "matches": {"RND001": hand_built_match()},
                     "log": quiet}
        arguments.update(changes)
        return offpitch.fit(**arguments)

    for bad in (math.nan, math.inf, None, 1.5, -0.1, True, "0.1"):
        expect_error(run.BenchmarkError, lambda: fit(target_share=bad), "offpitch.fit", "target_share")
    expect_error(run.BenchmarkError, lambda: fit(calibration=object()), "offpitch.fit", "calibration", "Calibration")
    for bad in (math.nan, None, 2.0):
        broken = dataclasses.replace(cal, scalars={**cal.scalars, "missing_share": bad})
        expect_error(run.BenchmarkError, lambda: fit(calibration=broken), "offpitch.fit", "missing_share")
    for bad in (0, -1, True, 1.5):
        expect_error(run.BenchmarkError, lambda: fit(seeds=bad), "offpitch.fit", "seeds")
    expect_error(run.BenchmarkError, lambda: fit(match_ids=["RND009"]), "RND009")
    expect_error(run.BenchmarkError, lambda: fit(matches={"RND001": wave_match("RND002")}), "RND001", "RND002")
    blank = hand_built_match()
    blank = dataclasses.replace(blank, ball_xy=np.full(blank.ball_xy.shape, np.nan))
    expect_error(run.BenchmarkError, lambda: fit(matches={"RND001": blank}), "offpitch.fit", "no visible ball")


def test_offpitch_constants_follow_from_the_recorded_fit():
    """degrade.py's two constants are those of the fit's record that it pins (OFFPITCH_FIT), or all three are None.

    constants_from reads the margin and the drop probability of a record (fit's whole record, or the scalars that
    degrade.py pins of it) and refuses a record without them, or with a value that is not on its grid, which no fit
    makes. The grids are the design's: 0.0 to 10.0 m by 0.1 m and 0.000 to 0.500 by 0.005, read-only."""
    if degrade.OFFPITCH_FIT is None:
        assert degrade.OFFPITCH_MARGIN_M is None and degrade.OFFPITCH_DROP_PROB is None
    else:
        assert offpitch.constants_from(degrade.OFFPITCH_FIT) == (degrade.OFFPITCH_MARGIN_M,
                                                                 degrade.OFFPITCH_DROP_PROB)
    assert offpitch.constants_from({"margin_m": 1.3, "drop_prob": 0.015, "refit": True}) == (1.3, 0.015)
    assert offpitch.constants_from(hand_fit(0.2, 0.48)) == (1.3, 0.015)
    for bad in ({}, {"margin_m": 1.3}, {"drop_prob": 0.015}, {"margin_m": 1.33, "drop_prob": 0.015},
                {"margin_m": 1.3, "drop_prob": 0.0151}, {"margin_m": math.inf, "drop_prob": 0.0},
                {"margin_m": True, "drop_prob": 0.0}, {"margin_m": 10.1, "drop_prob": 0.0}, None, [1.3, 0.015]):
        expect_error(ValueError, lambda: offpitch.constants_from(bad), "offpitch.constants_from", "record")
    assert offpitch.M_GRID.size == offpitch.Q_GRID.size == 101
    assert offpitch.M_GRID.tolist()[:3] == [0.0, 0.1, 0.2] and offpitch.M_GRID[-1] == 10.0
    assert offpitch.Q_GRID.tolist()[:3] == [0.0, 0.005, 0.01] and offpitch.Q_GRID[-1] == 0.5
    assert offpitch.REFIT_TOLERANCE == 0.005
    assert not offpitch.M_GRID.flags.writeable and not offpitch.Q_GRID.flags.writeable


def test_offpitch_fit_command_prints_and_writes_the_record():
    """`offpitch.py fit` reads the Calibration and our target (both stood in for here), fits on the matches asked for,
    prints the record and, with --out, writes it: strict JSON (no NaN, no infinity), sorted, LF, fit's record and the
    key of the target it was fitted to. A run it cannot do prints FAILED and a sentence and exits 1, no traceback."""
    served = dataclasses.replace(hand_built_match(), match_id="J03WMX")
    cal = hand_built_calibration(0.58)
    real_load, real_target = calibration.load, offpitch.target
    try:
        calibration.load = lambda data_dir=None, **keywords: cal
        offpitch.target = lambda data_dir=None, **keywords: {**MADE_UP_TARGET, "beyond_line_share": 0.118}
        with tempfile.TemporaryDirectory() as root, fixed_uniforms(), loading(lambda match_id: served):
            path = os.path.join(root, "fit.json")
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                code = offpitch.main(["fit", "--seeds", "2", "--matches", "J03WMX", "--data-dir", root, "--out", path])
            assert code == 0, shown.getvalue()
            raw = read(path, "rb")
            written = strict_json(raw.decode("utf-8"))
            expected = offpitch.fit(cal, 0.118, seeds=2, match_ids=["J03WMX"], data_dir=root, log=quiet)
            assert written == {**expected, "target_key": MADE_UP_TARGET["key"]}, written
            assert (written["margin_m"], written["drop_prob"], written["refit"]) == (0.6, 0.0, False)
            assert raw.endswith(b"\n") and b"\r" not in raw
            assert raw.decode("utf-8") == json.dumps(written, indent=1, sort_keys=True) + "\n"
            assert json.dumps(written, indent=1, sort_keys=True) in shown.getvalue()
            offpitch.target = lambda data_dir=None, **keywords: {**MADE_UP_TARGET, "beyond_line_share": None}
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                code = offpitch.main(["fit", "--seeds", "1", "--matches", "J03WMX", "--data-dir", root])
            out = shown.getvalue()
            assert code == 1 and out.startswith("FAILED") and "target_share" in out and "Traceback" not in out, out
    finally:
        calibration.load, offpitch.target = real_load, real_target


# -- the offpitch profile's fidelity ----------------------------------------------------------------------------------

def lists_in(value):
    """Every list anywhere inside a JSON-ready structure."""
    if isinstance(value, dict):
        return [found for item in value.values() for found in lists_in(item)]
    if isinstance(value, list):
        return [value] + [found for item in value for found in lists_in(item)]
    return []


def test_fidelity_tracking_and_context_gates_are_unchanged():
    """The gates of gaps, tracking and context are as they were before offpitch came: the same names, in order."""
    assert [gate["name"] for gate in fidelity.GATES["gaps"]] == TRACKING_GATES[:2]
    assert [gate["name"] for gate in fidelity.GATES["tracking"]] == TRACKING_GATES
    assert [gate["name"] for gate in fidelity.GATES["context"]] == CONTEXT_GATES
    assert sorted(fidelity.GATES) == ["context", "gaps", "offpitch", "tracking"]


def test_fidelity_offpitch_has_trackings_gates_and_the_beyond_line_gate():
    """offpitch's gates are tracking's eight, the same specs, and the share of the visible ball frames past a line,
    +-0.01 absolute.

    Its centre is our match's, offpitch.target()'s (never the Calibration's, which does not hold it), given to
    centre_of and _gate as the Calibration is; its figure is pooled by sums, beyond frames over visible frames, and not
    a mean of the parts' shares: 1 of 10 and 9 of 30 are 10 of 40 = 0.25, where the mean of 0.1 and 0.3 would be 0.2.
    No visible frame is no share, and a FAIL."""
    gates = fidelity.GATES["offpitch"]
    assert [gate["name"] for gate in gates] == TRACKING_GATES + ["beyond_line_share"]
    assert gates[:-1] == fidelity.GATES["tracking"]
    assert gates[-1] == {"name": "beyond_line_share", "tolerance": 0.01, "kind": "absolute",
                         "text": "share of the visible ball frames past a line"}
    cal = synthetic_calibration()
    assert fidelity.centre_of("beyond_line_share", cal, MADE_UP_TARGET) == 0.04
    assert fidelity.centre_of("beyond_line_share", cal, {**MADE_UP_TARGET, "beyond_line_share": 0.07}) == 0.07
    expect_error(fidelity.FidelityError, lambda: fidelity.centre_of("beyond_line_share", cal), "beyond_line_share",
                 "offpitch.target()")
    assert fidelity.centre_of("missing_share", cal, MADE_UP_TARGET) == cal.scalars["missing_share"]
    pool = fidelity.Pool()
    pool.add_line(10, np.array([0.5]))
    pool.add_line(30, np.linspace(0.1, 3.0, 9))
    assert fidelity.measured("beyond_line_share", pool) == 10 / 40 == 0.25
    spec = gates[-1]
    assert fidelity._gate(spec, pool, cal, {**MADE_UP_TARGET, "beyond_line_share": 0.255})["status"] == "pass"
    assert fidelity._gate(spec, pool, cal, {**MADE_UP_TARGET, "beyond_line_share": 0.2399})["status"] == "FAIL"
    assert fidelity._gate(spec, pool, cal, {**MADE_UP_TARGET, "beyond_line_share": 0.2601})["status"] == "FAIL"
    empty = fidelity.Pool()
    assert np.isnan(fidelity.measured("beyond_line_share", empty))
    assert fidelity._gate(spec, empty, cal, MADE_UP_TARGET)["status"] == "FAIL"


def worked_out_offpitch(matches, cal, seeds):
    """The offpitch report's own figures by other routes: every (match, seed) degraded by degrade() (offpitch, and gaps
    for tracking's ball, which is the gaps profile's), the line distance of each frame of the spans (line_distance_m),
    calibration.ball_stats for the gaps, the truth's dead frames, and the records of the ball added up."""
    out = {"visible": 0, "beyond": 0, "past": [], "frames": 0, "missing": 0, "near": 0, "interior": 0, "gaps_s": [],
           "gaps": {}, "tracking_gaps": {}, "draws": {name: 0 for name in degrade.OFFPITCH_COUNTS}, "dead": 0,
           "dead_seen": 0, "dead_seen_tracking": 0}
    for match_id, match in matches.items():
        dead = truths.truth_codes(match)[0] == D
        spans = [(half, match.kickoffs[half], stop) for half, _, stop in match.graded]
        metres = np.array(match.pitch)
        out["gaps"][match_id] = out["tracking_gaps"][match_id] = 0
        for seed in range(seeds):
            atd, info = degrade.degrade(degrade.tracking_view(match), "offpitch", cal, seed)
            gaps_atd, _ = degrade.degrade(degrade.tracking_view(match), "gaps", cal, seed)
            for _, start, stop in spans:
                line = degrade.line_distance_m(atd.ball_xy[start:stop], match.pitch)
                seen = ~np.isnan(line)
                out["visible"] += int(seen.sum())
                out["beyond"] += int((seen & (line < 0)).sum())
                out["past"].append(-line[seen & (line < 0)])
            ball = calibration.ball_stats(atd.ball_present, spans, atd.ball_xy * metres, match.pitch, match.fps)
            tracking = calibration.ball_stats(gaps_atd.ball_present, spans, gaps_atd.ball_xy * metres, match.pitch,
                                              match.fps)
            out["frames"] += ball["frames"]
            out["missing"] += ball["missing_frames"]
            out["near"] += ball["near_line_onsets"]
            out["interior"] += ball["interior_gaps"]
            out["gaps_s"].append(ball["gap_durations_s"])
            out["gaps"][match_id] += ball["gaps"]
            out["tracking_gaps"][match_id] += tracking["gaps"]
            for name in degrade.OFFPITCH_COUNTS:
                out["draws"][name] += info["ball"][name]
            out["dead"] += int(dead.sum())
            out["dead_seen"] += int(atd.ball_present[dead].sum())
            out["dead_seen_tracking"] += int(gaps_atd.ball_present[dead].sum())
    return out


def short_gaps(durations):
    """{"gaps", "short", "share", "time_share"} of gap lengths in seconds: how many are shorter than 2 s, their share of
    the gaps and of the time without a ball, rounded as a report's."""
    durations = np.asarray(durations, dtype=np.float64)
    short = durations < 2.0
    return {"gaps": int(durations.size), "short": int(short.sum()),
            "share": round(float(short.sum()) / durations.size, 9) if durations.size else None,
            "time_share": round(float(durations[short].sum() / durations.sum()), 9) if durations.sum() > 0 else None}


def test_fidelity_offpitch_report_synthetic():
    """The offpitch report: tracking's gates and the past-the-line one, tracking's checks and the near-line onset share
    as a warning, and the information that R015 quotes, each figure worked out here by other code.

    Two matches of 22 players whose ball goes past the lines, our made-up masks and tracking figures, the constants set
    to m = 1.0 and q = 0.3 for the check (pinned), and our target made up and injected (MADE_UP_TARGET). The
    past-the-line gate is the visible frames past a line over the visible frames of every (match, seed), with the
    target's centre and +-0.01; the players' gates are tracking's on the same matches, seeds and Calibration; the
    near-line check has the Calibration's centre and +-0.05. The information: the quantiles of how far past the line
    the visible frames are, ours (the target's) beside the profile's pooled over every (match, seed); the gaps shorter
    than 2 s, the profile's and ours (the Calibration's gap lengths, counted, never listed); the gaps per match and seed
    beside tracking's ball (the gaps profile's, which tracking's is); the dead-ball visible share beside tracking's; and
    the ball records' counts added up. A tracking report on the same matches has none of it, and its keys are as they
    were."""
    matches, cal = wave_matches(TEAMS_22), tracking_calibration()
    with pinned(1.0, 0.3), tempfile.TemporaryDirectory() as root:
        result = fidelity.report("offpitch", seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet,
                                 target=MADE_UP_TARGET)
        again = fidelity.report("offpitch", seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet,
                                target=MADE_UP_TARGET)
        tracking = fidelity.report("tracking", seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet)
        ours = worked_out_offpitch(matches, cal, 2)
    assert again == result and json.loads(json.dumps(result, allow_nan=False)) == result
    assert sorted(result) == sorted(REPORT_KEYS + ["offpitch"]) and sorted(tracking) == sorted(REPORT_KEYS)
    assert result["profile"] == {"name": "offpitch", "version": 1, "tracking_version": 3}
    assert [gate["name"] for gate in result["gates"]] == TRACKING_GATES + ["beyond_line_share"]
    by_name = {gate["name"]: gate for gate in result["gates"]}
    gate = by_name["beyond_line_share"]
    assert abs(gate["measured"] - ours["beyond"] / ours["visible"]) < 1e-9 and gate["centre"] == 0.04, gate
    assert (gate["tolerance"], gate["kind"]) == (0.01, "absolute")
    assert gate["status"] == ("pass" if abs(gate["measured"] - 0.04) <= 0.01 else "FAIL")
    assert abs(by_name["missing_share"]["measured"] - ours["missing"] / ours["frames"]) < 1e-9
    tracking_gates = {gate["name"]: gate for gate in tracking["gates"]}
    for name in TRACKING_GATES[2:]:
        assert by_name[name] == tracking_gates[name], name
    assert result["players_draws"] == tracking["players_draws"] and result["ball_draws"] == {}
    assert result["passed"] == (not any(gate["status"] == "FAIL" for gate in result["gates"]))
    # The checks: tracking's, then the near-line onset share.
    assert [check["name"] for check in result["checks"]] == [check["name"] for check in tracking["checks"]] + [
        "near_line_share"]
    near = result["checks"][-1]
    assert (near["band"], near["kind"]) == (0.05, "absolute") and near["centre"] == round(
        cal.scalars["near_line_share"], 9)
    assert abs(near["measured"] - ours["near"] / ours["interior"]) < 1e-9 and ours["interior"] > 10
    assert near["status"] == ("ok" if abs(near["measured"] - near["centre"]) <= 0.05 else "WARN")
    assert result["warnings"] == sum(1 for check in result["checks"] if check["status"] == "WARN")
    # The information.
    info = result["offpitch"]
    assert set(info) == OFFPITCH_INFO, sorted(info)
    assert info["target"] == {"key": MADE_UP_TARGET["key"], "version": MADE_UP_TARGET["version"]}
    assert info["line"] == {"visible_frames": ours["visible"], "beyond_frames": ours["beyond"]}
    quantiles = info["beyond_m_quantiles"]
    assert quantiles["bar_ath"] == MADE_UP_TARGET["beyond_m_quantiles"]
    past = np.concatenate(ours["past"])
    assert past.size > 20 and list(quantiles["benchmark"]) == [name for name, _ in offpitch.QUANTILES]
    for name, level in offpitch.QUANTILES:
        assert abs(quantiles["benchmark"][name] - float(np.quantile(past, level))) < 1e-9, name
    assert info["short_gaps"] == {"benchmark": short_gaps(np.concatenate(ours["gaps_s"])),
                                  "bar_ath": short_gaps(cal.gap_durations_s)}, info["short_gaps"]
    assert info["gaps_per_match"] == {match_id: {"offpitch": round(ours["gaps"][match_id] / 2, 9),
                                                 "tracking": round(ours["tracking_gaps"][match_id] / 2, 9)}
                                      for match_id in matches}, info["gaps_per_match"]
    assert info["dead_ball_visible_share"] == {"offpitch": round(ours["dead_seen"] / ours["dead"], 9),
                                               "tracking": round(ours["dead_seen_tracking"] / ours["dead"], 9)}
    assert info["dead_ball_visible_share"]["offpitch"] == result["checks"][1]["measured"]
    assert info["draws"] == ours["draws"], (info["draws"], ours["draws"])
    assert ours["draws"]["offpitch_stretches"] > 0 and ours["draws"]["gaps_dropped"] > 0
    assert all(len(found) <= 64 for found in lists_in(result)), "no frame-sized list"
    assert any("offpitch.target()" in note for note in result["notes"]), result["notes"]
    assert not any("offpitch" in note for note in tracking["notes"])
    text = "\n".join(fidelity.render(result))
    for fragment in ("beyond_line_share", "near_line_share", "past a line", "shorter than 2 s", "gaps per match",
                     "off-pitch stretches", "dropped", "tracking", "p90"):
        assert fragment in text, (fragment, text)
    assert "past a line" not in "\n".join(fidelity.render(tracking))


def test_fidelity_offpitch_measures_what_the_fit_measured():
    """At the constants a fit chose, the fidelity report's two gates that the fit matched read the fit's own figures, to
    the last decimal: the missing share is missing_at_fit and the share past a line share_at_fit; the missing share's
    centre is the fit's missing target. The fit builds the ball as degrade() does and fidelity measures degrade()'s, so
    the two must not part."""
    matches, cal = wave_matches(TEAMS_22), tracking_calibration()
    record = offpitch.fit(cal, 0.03, seeds=2, matches=matches, log=quiet)
    with pinned(*offpitch.constants_from(record)), tempfile.TemporaryDirectory() as root:
        result = fidelity.report("offpitch", seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet,
                                 target={**MADE_UP_TARGET, "beyond_line_share": record["target_share"]})
    by_name = {gate["name"]: gate for gate in result["gates"]}
    assert by_name["missing_share"]["measured"] == record["missing_at_fit"], (by_name["missing_share"], record)
    assert by_name["beyond_line_share"]["measured"] == record["share_at_fit"], (by_name["beyond_line_share"], record)
    assert by_name["missing_share"]["centre"] == record["missing_target"]
    assert by_name["beyond_line_share"]["centre"] == record["target_share"]
    assert result["offpitch"]["line"] == {"visible_frames": record["counts_at_fit"]["visible_frames"],
                                          "beyond_frames": record["counts_at_fit"]["beyond_frames"]}
    assert result["offpitch"]["draws"] == record["draws_at_fit"]


def test_fidelity_offpitch_refuses_until_fitted_and_without_a_target():
    """Before anything is loaded: an offpitch report while the constants are not pinned is a FidelityError that names
    the fit (offpitch.py fit, R015), and one given a target that is not offpitch.target()'s (no finite share past a
    line, no quantiles) is one that names the target."""
    matches, cal = wave_matches(TEAMS_22), tracking_calibration()

    def report(target=MADE_UP_TARGET):
        return fidelity.report("offpitch", seeds=1, matches=matches, calibration=cal, log=quiet, target=target)

    with pinned(None, None, None):
        expect_error(fidelity.FidelityError, report, "offpitch", "offpitch.py fit", "R015")
    with pinned(1.0, 0.3):
        for bad in ({"key": "x"}, {**MADE_UP_TARGET, "beyond_line_share": None},
                    {**MADE_UP_TARGET, "beyond_line_share": math.nan}, {**MADE_UP_TARGET, "beyond_m_quantiles": None},
                    "0.04"):
            expect_error(fidelity.FidelityError, lambda: report(target=bad), "target", "offpitch.target()")
