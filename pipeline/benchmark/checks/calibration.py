"""Checks of the calibration (pipeline/benchmark/calibration.py): our own tracking, measured for the benchmark.

The statistics are checked on objects built here by hand, small enough that every figure in a check is worked out from
the objects' construction and written down with its working, never read off what the code printed. The two fixtures
(ball_fixture, player_fixture) are built so that a mistake of the kind that is easy to make changes a figure: a gap
that runs across the edge of a block, two blocks that touch (a naive run-length count would fuse the gaps either side
of the join), a ball that sits exactly 2.0 m from a line (the near-line test is strict), a slot that is present before
and after a block (a tracklet must not run over its edge) and a second difference that would be huge if a triple
straddled the edge. test_data_calibration_replicates_a01 then runs the same code on our match: on the graded blocks,
where A01, A02 and R001 published their figures, and on the spans from each kick-off that the Calibration stores. It
prints aggregates (counts, shares, quantiles) and never a frame or a row of the match.
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import calibration, hazard
from pipeline.benchmark.checks.common import NotRun
from pipeline.selftest import expect_error, make_synthetic_bundle, quiet, synthetic_inputs

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
PITCH = (105.3, 68.0)
MAX_SEQUENCE = 20     # describe() may hold no sequence longer than this: nothing that could rebuild a frame
# The figures the plan and the brief give for our match (A01, A02, R001; graded blocks), asserted by
# test_data_calibration_replicates_a01.
A01 = {"missing_share": (0.4069, 0.0001), "gaps": 150, "interior_gaps": 147, "near_line_onsets": 42,
       "median_gap_s": (10.7, 0.05), "tracklets": 1219, "wrong_team_share": (0.108, 0.001),
       "speed_nan_share": (0.0796, 0.0005), "nearest_agreement": (0.778, 0.001)}
# A02's table, in whole percent per 5 minutes counted from each half's first graded frame (the brief asks for blocks
# 1-3 of the first half; the whole table reproduces).
A02_PERCENT = {1: (12, 17, 24, 28, 44, 35, 63, 43, 71, 46, 80), 2: (12, 33, 26, 55, 26, 39, 80, 51, 52, 40, 28)}
BASELINE_POSSESSIONS = {0: 1949, 25: 746, 50: 594, 125: 398}
BASELINE_FILES = {0: "baseline_raw", 25: "baseline_smooth25", 50: "baseline_smooth50", 125: "baseline_smooth125"}
POSSESSION_TOLERANCE = 5
# Figures of the plan (measured 2026-10-22 by a read-only script, on the graded blocks) that the calibration
# reproduces; the two that it does not (5.26% of frames with 14 or fewer present against the plan's 6.6%, and p99 of the
# NaN-speed runs, 2,305 frames against 1,209, which the plan measured on the whole file) are in A07, not asserted here.
PLAN = {"wrong_team_episodes": (63, 45), "ball_speed_nan_share": (0.922, 0.0005), "present_mean": (20.5, 0.05),
        "present_share_22": (0.495, 0.0005), "tracklet_median_s": (35.5, 0.05), "jitter_rms_mm": (3.3, 0.05)}


# -- the ball, by hand ------------------------------------------------------------------------------------------------

def ball_fixture():
    """(ball_present, ball_xy_m, blocks, pitch, fps): 1,100 frames, two blocks that touch, one frame a second.

    Block 1 is frames 10-709 (700 frames), block 2 frames 710-1009 (300 frames). Missing runs, as (first, last) frames
    within the block (the frame number inside the block, counted from 0):

        block 1   g1  100-119   20 s   interior, last visible frame 99 at x = 1.5            (1.5 m from a line)
                  g2  150-154    5 s   interior, last visible frame 149 at x = 2.0            (exactly 2 m: not near)
                  g3  330-349   20 s   interior, last visible frame 329 at x = -0.5           (beyond the line)
                  g4  600-699  100 s   runs to the end of the block: no visible frame after it, not interior
        block 2   g5    0-9     10 s   starts with the block: no visible frame before it, not interior
                  g6  200-204    5 s   interior, last visible frame 199 at y = 67.0           (1.0 m from the far line)
                  g7  250-254    5 s   interior, last visible frame 249 at x = 104.0          (1.3 m from the far line)

    g4 and g5 are separated by nothing but the join of the two blocks, so a count over the whole array would make one
    gap of 110 frames. Frames 0-9 and 1010-1099 are missing too and belong to no block: they must not count.
    """
    n = 1100
    present = np.ones(n, dtype=bool)
    present[:10] = False
    present[1010:] = False
    for block_start, first, last in ((10, 100, 119), (10, 150, 154), (10, 330, 349), (10, 600, 699),
                                     (710, 0, 9), (710, 200, 204), (710, 250, 254)):
        present[block_start + first:block_start + last + 1] = False
    xy = np.full((n, 2), (52.65, 34.0))
    xy[10 + 99] = (1.5, 34.0)
    xy[10 + 149] = (2.0, 34.0)
    xy[10 + 329] = (-0.5, 34.0)
    xy[710 + 199] = (52.65, 67.0)
    xy[710 + 249] = (104.0, 34.0)
    xy[~present] = np.nan
    return present, xy, [(1, 10, 710), (2, 710, 1010)], PITCH, 1


def check_ball_stats():
    present, xy, blocks, pitch, fps = ball_fixture()
    got = calibration.ball_stats(present, blocks, xy, pitch, fps)

    # 20 + 5 + 20 + 100 + 10 + 5 + 5 = 165 missing frames of the 1,000 in the blocks.
    assert got["frames"] == 1000 and got["missing_frames"] == 165 and got["dropped_frames"] == 0, got
    assert got["missing_share"] == 0.165, got["missing_share"]
    assert got["gaps"] == 7 and got["interior_gaps"] == 5, (got["gaps"], got["interior_gaps"])
    assert got["gap_durations_s"].tolist() == [20.0, 5.0, 20.0, 100.0, 10.0, 5.0, 5.0], got["gap_durations_s"]
    # The interior gaps are g1, g2, g3, g6 and g7, and three arrays describe them in step: duration, minute, flag.
    assert got["interior_durations_s"].tolist() == [20.0, 5.0, 20.0, 5.0, 5.0], got["interior_durations_s"]
    # The line distance, min(x, L - x, y, W - y), at the last visible frame of each interior gap. Block 2's last
    # entry is 105.3 - 104.0, which a float spells 1.2999999999999972.
    assert np.allclose(got["gap_onset_line_m"], [1.5, 2.0, -0.5, 1.0, 105.3 - 104.0], rtol=0, atol=1e-12), (
        got["gap_onset_line_m"])
    # Near is strict: 2.0 exactly is not near, so g1, g3, g6 and g7 are, and g2 is not.
    assert got["gap_onset_near_line"].tolist() == [True, False, True, True, True], got["gap_onset_near_line"]
    assert got["near_line_onsets"] == 4 and got["near_line_share"] == 0.8, (got["near_line_onsets"],
                                                                            got["near_line_share"])
    # The minute of the last visible frame, from the start of its own block, at one frame a second.
    assert np.allclose(got["gap_onset_minute"], np.array([99, 149, 329, 199, 249]) / 60.0, rtol=0, atol=1e-12), (
        got["gap_onset_minute"])
    # 5-minute segments of 300 frames from each block's first frame: block 1 is 300, 300 and a last one of 100.
    assert sorted(got["missing_share_5min"]) == [1, 2], got["missing_share_5min"].keys()
    assert np.allclose(got["missing_share_5min"][1], [25 / 300, 20 / 300, 100 / 100], rtol=0, atol=1e-15), (
        got["missing_share_5min"][1])
    assert np.allclose(got["missing_share_5min"][2], [20 / 300], rtol=0, atol=1e-15), got["missing_share_5min"][2]
    # segment_shares is that table alone, from the visible frames.
    alone = calibration.segment_shares(present, blocks, fps)
    assert all(np.array_equal(alone[half], got["missing_share_5min"][half]) for half in (1, 2)), alone

    # The frame rate turns frames into seconds and minutes, and sets the segment: at 2 frames a second the same
    # gaps last half as long in seconds, and a 5-minute segment is 600 frames, so block 1 is a segment and 100 frames.
    twice = calibration.ball_stats(present, blocks, xy, pitch, 2)
    assert twice["gap_durations_s"].tolist() == [10.0, 2.5, 10.0, 50.0, 5.0, 2.5, 2.5], twice["gap_durations_s"]
    assert twice["interior_durations_s"].tolist() == [10.0, 2.5, 10.0, 2.5, 2.5], twice["interior_durations_s"]
    assert np.allclose(twice["gap_onset_minute"], np.array([99, 149, 329, 199, 249]) / 120.0, rtol=0, atol=1e-12)
    # Segment 1 of block 1 now holds g1, g2 and g3 (45 of 600 frames) and segment 2 holds g4 (100 of 100); block 2 is
    # still one segment of 300 frames, with 20 missing.
    assert np.allclose(twice["missing_share_5min"][1], [45 / 600, 1.0], rtol=0, atol=1e-15), twice["missing_share_5min"]
    assert np.allclose(twice["missing_share_5min"][2], [20 / 300], rtol=0, atol=1e-15), twice["missing_share_5min"]

    # No gap at all, and a block that is nothing but a gap.
    clear = calibration.ball_stats(np.ones(50, dtype=bool), [(1, 0, 50)], np.full((50, 2), 10.0), pitch, 25)
    assert clear["gaps"] == 0 and clear["interior_gaps"] == 0 and clear["missing_share"] == 0.0, clear
    assert clear["gap_durations_s"].size == 0 and clear["gap_onset_minute"].size == 0
    assert clear["interior_durations_s"].size == 0
    blind = calibration.ball_stats(np.zeros(50, dtype=bool), [(1, 5, 45)], np.full((50, 2), np.nan), pitch, 25)
    assert blind["gaps"] == 1 and blind["interior_gaps"] == 0 and blind["missing_share"] == 1.0, blind
    assert blind["gap_durations_s"].tolist() == [40 / 25], blind["gap_durations_s"]


def test_calibration_half_missing_ball_is_absent_and_counted():
    """A position with one coordinate NaN is no position: the frame is a gap of its own, and the count says so."""
    present, xy, blocks, pitch, fps = ball_fixture()
    xy[410, 1] = np.nan     # block 1, frame 400 of it: x = 52.65 is there, y is not, and the flag still says present
    assert present[410]
    got = calibration.ball_stats(present, blocks, xy, pitch, fps)
    # One more missing frame, one more gap (1 s, between visible frames: interior), 166 of 1,000 missing.
    assert got["dropped_frames"] == 1 and got["missing_frames"] == 166 and got["frames"] == 1000, got
    assert got["gaps"] == 8 and got["interior_gaps"] == 6, (got["gaps"], got["interior_gaps"])
    assert got["gap_durations_s"].tolist() == [20.0, 5.0, 20.0, 1.0, 100.0, 10.0, 5.0, 5.0], got["gap_durations_s"]
    # The new gap sits between g3 and g4, so between g3 and g6 among the interior ones; its onset is frame 399 of block
    # 1, at the centre of the pitch: 34 m from the nearest line, so not near.
    assert got["interior_durations_s"].tolist() == [20.0, 5.0, 20.0, 1.0, 5.0, 5.0], got["interior_durations_s"]
    assert np.allclose(got["gap_onset_minute"], np.array([99, 149, 329, 399, 199, 249]) / 60.0, rtol=0, atol=1e-12)
    assert got["gap_onset_near_line"].tolist() == [True, False, True, False, True, True], got["gap_onset_near_line"]
    assert np.isclose(got["gap_onset_line_m"][3], 34.0), got["gap_onset_line_m"]
    # Segment 2 of block 1 (frames 300-599 of it) holds g3 and the new frame: 21 of 300.
    assert np.allclose(got["missing_share_5min"][1], [25 / 300, 21 / 300, 1.0], rtol=0, atol=1e-15)

    # The reverse is not a half-missing position but two trackings that disagree: refused, by name.
    lying = present.copy()
    lying[300] = False
    expect_error(ValueError, lambda: calibration.ball_stats(lying, blocks, ball_fixture()[1], pitch, fps),
                 "ball_present", "ball_xy_m", "disagree")


# -- the players, by hand ---------------------------------------------------------------------------------------------

def player_fixture():
    """(present, slot_team, xy_m, speed, blocks, fps): 100 frames, 24 slots (0-11 home, 12-23 away), 4 frames a second.

    Block A is frames 5-44 (40 frames), block B frames 60-89 (30 frames). Frames 0-4, 45-59 and 90-99 belong to no
    block and have all 24 slots present, with wild positions for slot 0: nothing of them may count, and no tracklet or
    second difference may run over a block's edge. Within a block, r is the frame counted from the block's start.

    Block A: home slots 0-9 and away slots 12-20 present throughout; slot 10 for r 0-19, slot 11 for r 10-29; slot 21
    for r 0-9 and 30-39; slots 22 and 23 never. So home shows 11 (r 0-9), 12 (r 10-19), 11 (r 20-29), 10 (r 30-39) and
    away 10, 9, 9, 10. Present: 21 for r 0-19, 20 for r 20-39. Wrong team: home more than 11 and away fewer than 11 at
    r 10-19, 10 frames.

    Block B: home slots 0-9 and away slots 12-21 throughout; slot 22 for r 5-14 and 20-29, slot 23 for r 8-24; slots
    10, 11 never. Home shows 10 throughout, away 10 (r 0-4), 11 (5-7), 12 (8-14), 11 (15-19), 12 (20-24), 11 (25-29).
    Present: 20, 21, 22, 21, 22, 21. Wrong team: away more than 11 and home fewer than 11 at r 8-14 and r 20-24, 7 and 5
    frames.

    Positions: slot s is at x = 10 + s, y = 20 while present, except slot 0 in block A, x = 10 + r^2 / 64 (second
    difference 2/64 = 0.03125 m), and slot 5 in block B, y = 20 + r^2 / 32 (second difference 0.0625 m).
    Speeds: 1.5 where present and NaN where absent, and NaN while present for slot 3 (A: r 0-4, 20-21), slot 10 (A:
    r 15-19, its last present frames), slot 21 (A: r 8-9 and 30-31, the end of one tracklet and the start of the next)
    and slot 14 (B: r 10-19), and for slot 3 on frames 0-4, which no block holds.
    """
    n, slots = 100, 24
    present = np.zeros((n, slots), dtype=bool)
    present[:5] = present[45:60] = present[90:] = True

    def on(start, slot, first, last):
        present[start + first:start + last + 1, slot] = True

    a, b = 5, 60
    for slot in list(range(10)) + list(range(12, 21)):
        on(a, slot, 0, 39)
    on(a, 10, 0, 19)
    on(a, 11, 10, 29)
    on(a, 21, 0, 9)
    on(a, 21, 30, 39)
    for slot in list(range(10)) + list(range(12, 22)):
        on(b, slot, 0, 29)
    on(b, 22, 5, 14)
    on(b, 22, 20, 29)
    on(b, 23, 8, 24)

    xy = np.zeros((n, slots, 2))
    xy[:, :, 0] = 10.0 + np.arange(slots)
    xy[:, :, 1] = 20.0
    xy[0:5, 0, 0] = (500.0, 600.0, 800.0, 1100.0, 1500.0)     # wild, outside the blocks
    xy[45:60, 0, 0] = 2000.0 + np.arange(15) ** 3
    xy[90:, 0, 0] = 5000.0 + np.arange(10) ** 3
    for r in range(40):
        xy[a + r, 0, 0] = 10.0 + r * r / 64.0
    for r in range(30):
        xy[b + r, 5, 1] = 20.0 + r * r / 32.0
    xy[~present] = np.nan

    speed = np.where(present, 1.5, np.nan)
    for start, slot, first, last in ((a, 3, 0, 4), (a, 3, 20, 21), (a, 10, 15, 19), (a, 21, 8, 9), (a, 21, 30, 31),
                                     (b, 14, 10, 19), (0, 3, 0, 4)):
        speed[start + first:start + last + 1, slot] = np.nan
    slot_team = np.array([0] * 12 + [1] * 12, dtype=np.int8)
    return present, slot_team, xy, speed, [(1, 5, 45), (2, 60, 90)], 4


def check_player_stats():
    present, slot_team, xy, speed, blocks, fps = player_fixture()
    got = calibration.player_stats(present, slot_team, xy, speed, blocks, fps)

    # Present per frame is 21 (20 frames) and 20 (20 frames) in A, and 20 (5), 21 (13), 22 (12) in B: 25 frames of 20,
    # 33 of 21 and 12 of 22, 70 frames in the blocks and 20*25 + 21*33 + 22*12 = 1,457 present slot-frames.
    expected_hist = np.zeros(25, dtype=np.int64)
    expected_hist[[20, 21, 22]] = (25, 33, 12)
    assert got["present_hist"].dtype == np.int64 and got["present_hist"].tolist() == expected_hist.tolist(), (
        got["present_hist"])
    assert got["present_slot_frames"] == 1457 and got["slots"] == 24 and got["frames"] == 70, got
    assert got["dropped_slot_frames"] == 0, got["dropped_slot_frames"]
    assert got["present_mean"] == 1457 / 70, got["present_mean"]
    assert got["present_share_22"] == 12 / 70 and got["present_share_le14"] == 0.0, got

    # Tracklets: A has 19 of 40 frames (slots 0-9, 12-20), 2 of 20 (slots 10, 11), 2 of 10 (slot 21); B has 20 of 30
    # (slots 0-9, 12-21), 2 of 10 (slot 22) and 1 of 17 (slot 23). 46 in all, and 1,457 present slot-frames again.
    frames = [40] * 19 + [20] * 2 + [10] * 2 + [30] * 20 + [10] * 2 + [17]
    assert got["tracklets"] == 46 == len(frames) and sum(frames) == 1457
    assert sorted(got["tracklet_lengths_s"].tolist()) == sorted(length / 4 for length in frames), (
        got["tracklet_lengths_s"])

    # Wrong team: 10 frames in A and 7 + 5 in B, 22 of 70. Episodes in time order: home surplus 2.5 s, then away 1.75
    # s and 1.25 s.
    assert got["wrong_team_frames"] == 22 and got["wrong_team_share"] == 22 / 70, got
    assert got["wrong_team_episodes_s"].tolist() == [2.5, 1.75, 1.25], got["wrong_team_episodes_s"]
    assert (got["wrong_team_episodes_home"], got["wrong_team_episodes_away"]) == (1, 2), got

    # NaN speed while present: slot 3 5 + 2, slot 10 5, slot 21 2 + 2, slot 14 10 = 26 frames, in six runs. The five
    # frames of slot 3 outside the blocks do not count, and the two NaN frames of slot 21 either side of its absence
    # are two runs and not one.
    assert got["speed_nan_frames"] == 26 and got["speed_nan_share"] == 26 / 1457, got
    assert got["speed_nan_runs"].dtype == np.int64 and sorted(got["speed_nan_runs"].tolist()) == [2, 2, 2, 5, 5, 10], (
        got["speed_nan_runs"])

    # Jitter: 38 triples of slot 0 in A at 0.03125 m and 28 of slot 5 in B at 0.0625 m, of 1,365 present triples
    # (A: 380 + 18 + 18 + 342 + 16 = 774; B: 280 + 280 + 16 + 15 = 591; a tracklet of L frames has L - 2).
    assert got["jitter_triples"] == 1365, got["jitter_triples"]
    assert abs(got["jitter_rms_m"] - math.sqrt((38 * 0.03125 ** 2 + 28 * 0.0625 ** 2) / 1365)) < 1e-12, (
        got["jitter_rms_m"])

    # A team's size is 11, and the home and away of a block are told apart by slot_team, not by slot order.
    flipped = calibration.player_stats(present, 1 - slot_team, xy, speed, blocks, fps)
    assert flipped["wrong_team_episodes_s"].tolist() == got["wrong_team_episodes_s"].tolist()
    assert (flipped["wrong_team_episodes_home"], flipped["wrong_team_episodes_away"]) == (2, 1), flipped


def test_calibration_synthetic():
    """Every statistic exact on a small hand-built object: the ball's (A01, A02) and the players' (A07)."""
    check_ball_stats()
    check_player_stats()


def test_calibration_half_missing_slot_is_absent_and_counted():
    """A slot whose position has one coordinate NaN is absent on that frame, for every statistic, and is counted."""
    present, slot_team, xy, speed, blocks, fps = player_fixture()
    xy[5 + 15, 0, 1] = np.nan    # slot 0, frame 15 of block A: x is there, y is not, and the flag still says present
    assert present[5 + 15, 0]
    got = calibration.player_stats(present, slot_team, xy, speed, blocks, fps)
    assert got["dropped_slot_frames"] == 1, got["dropped_slot_frames"]

    # Frame 15 of A had 21 present and now has 20: 25 -> 26 frames of 20, 33 -> 32 of 21, 1,456 present slot-frames.
    expected_hist = np.zeros(25, dtype=np.int64)
    expected_hist[[20, 21, 22]] = (26, 32, 12)
    assert got["present_hist"].tolist() == expected_hist.tolist(), got["present_hist"]
    assert got["present_slot_frames"] == 1456 and got["present_mean"] == 1456 / 70, got

    # Slot 0's 40-frame tracklet in A is two now, frames 0-14 and 16-39 of the block: 15 and 24.
    frames = [40] * 18 + [15, 24] + [20] * 2 + [10] * 2 + [30] * 20 + [10] * 2 + [17]
    assert got["tracklets"] == 47 == len(frames) and sum(frames) == 1456
    assert sorted(got["tracklet_lengths_s"].tolist()) == sorted(length / 4 for length in frames)

    # The home surplus of r 10-19 loses r 15 (home shows 11 there now): two episodes, r 10-14 and r 16-19, 5 and 4
    # frames; 21 wrong-team frames of 70, and the away episodes of block B as before.
    assert got["wrong_team_frames"] == 21 and got["wrong_team_share"] == 21 / 70, got
    assert got["wrong_team_episodes_s"].tolist() == [1.25, 1.0, 1.75, 1.25], got["wrong_team_episodes_s"]
    assert (got["wrong_team_episodes_home"], got["wrong_team_episodes_away"]) == (2, 2), got

    # Speed: nothing of the 26 NaN frames is slot 0's, so only the denominator changes.
    assert got["speed_nan_frames"] == 26 and got["speed_nan_share"] == 26 / 1456, got

    # Jitter: three triples lose the dropped frame (those centred on r 14, 15 and 16): slot 0 has 13 + 22 = 35 left of
    # its 38, and the other slots' 1,327 are as before.
    assert got["jitter_triples"] == 1362, got["jitter_triples"]
    assert abs(got["jitter_rms_m"] - math.sqrt((35 * 0.03125 ** 2 + 28 * 0.0625 ** 2) / 1362)) < 1e-12, (
        got["jitter_rms_m"])

    # A position that is given where the flag says absent is two trackings that disagree: refused, by name.
    positioned = player_fixture()[2]
    positioned[10, 22, :] = 1.0   # slot 22 is never present in block A
    expect_error(ValueError, lambda: calibration.player_stats(present, slot_team, positioned, speed, blocks, fps),
                 "players_present", "players_xy_m", "disagree")


def test_calibration_stats_refuse_what_they_cannot_measure():
    present, xy, blocks, pitch, fps = ball_fixture()
    ball = calibration.ball_stats
    expect_error(ValueError, lambda: ball(present[:-1], blocks, xy, pitch, fps), "ball_present", "ball_xy_m", "1099")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 710), (2, 700, 1010)], xy, pitch, fps), "blocks", "overlap")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 710), (1, 710, 1010)], xy, pitch, fps), "half", "twice")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 1101)], xy, pitch, fps), "blocks", "1101", "1100")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 10)], xy, pitch, fps), "blocks", "empty")
    expect_error(ValueError, lambda: ball(present, [], xy, pitch, fps), "blocks", "empty")
    expect_error(ValueError, lambda: ball(present, blocks, xy, (0.0, 68.0), fps), "pitch")
    expect_error(ValueError, lambda: ball(present, blocks, xy, pitch, 0), "fps")
    expect_error(ValueError, lambda: ball(present.astype(int), blocks, xy, pitch, fps), "ball_present", "bool")
    expect_error(ValueError, lambda: calibration.segment_shares(present.astype(int), blocks, fps), "ball_visible",
                 "bool")

    have, slot_team, pxy, speed, pblocks, pfps = player_fixture()
    players = calibration.player_stats
    expect_error(ValueError, lambda: players(have, slot_team[:-1], pxy, speed, pblocks, pfps), "slot_team", "23")
    expect_error(ValueError, lambda: players(have, slot_team + 1, pxy, speed, pblocks, pfps), "slot_team",
                 "0 (home) or 1 (away)")
    expect_error(ValueError, lambda: players(have, slot_team, pxy[:, :, :1], speed, pblocks, pfps), "players_xy_m")
    expect_error(ValueError, lambda: players(have, slot_team, pxy, speed[:-1], pblocks, pfps), "players_speed")
    expect_error(ValueError, lambda: players(have, slot_team, pxy, speed, [(1, 5, 101)], pfps), "blocks", "101")


# -- a whole Calibration, on the synthetic bundle ---------------------------------------------------------------------

def synthetic_calibration_inputs(root):
    """(inputs, gt): the synthetic bundle with two graded blocks, kick-offs inside them and a made-up public truth."""
    make_synthetic_bundle(root)
    inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
    inputs.kickoffs = {1: 60, 2: 340}
    gt = np.full(inputs.n_frames, X, dtype=np.int8)
    gt[60:100], gt[100:120], gt[120:130], gt[130:150] = H, A, D, H
    gt[340:360], gt[360:370] = A, D
    return inputs, gt


def brute_force_agreement(inputs, gt):
    """(share, frames) of in-play frames, ball visible and somebody tracked, where the nearest slot's team is the GT's.

    One frame at a time, one slot at a time, in plain Python and in the baseline's own order of operations: the
    reference for pipeline.evidence.describe's number, with no vectorised code in common with it. A tie goes to the
    lowest slot, as starter/baseline.py's strict '<' keeps it.
    """
    atd = inputs.atd
    length, width = inputs.pitch
    agree = judged = 0
    for frame in range(inputs.n_frames):
        if gt[frame] not in (H, A) or not atd.ball_present[frame]:
            continue
        best, best_distance = None, None
        for slot in range(atd.n_slots):
            if not atd.players_present[frame, slot]:
                continue
            dx = atd.players_xy[frame, slot, 0] * length - atd.ball_xy[frame, 0] * length
            dy = atd.players_xy[frame, slot, 1] * width - atd.ball_xy[frame, 1] * width
            distance = dx * dx + dy * dy
            if best is None or distance < best_distance:
                best, best_distance = slot, distance
        if best is None:
            continue
        judged += 1
        agree += (atd.slot_team[best] == 0) == (gt[frame] == H)
    return agree / judged, judged


def test_calibration_measure_synthetic():
    """A Calibration of the synthetic bundle: spans, wiring of the statistics, the public window, the yardsticks."""
    from pipeline.config import load_config
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        atd = inputs.atd
        counts = atd.players_present.sum(axis=1)
        spans = [(1, 60, 280), (2, 340, 560)]    # from each kick-off to the end of the graded block

        # The ball's mask and the players' count run over the spans, per half: True where the ball is missing, int16
        # counts, 220 frames (60-279) and 220 (340-559) here.
        assert sorted(cal.ball_masks) == [1, 2] and sorted(cal.present_counts) == [1, 2]
        for half, kickoff, stop in spans:
            mask, per_frame = cal.ball_masks[half], cal.present_counts[half]
            assert mask.dtype == bool and per_frame.dtype == np.int16, (mask.dtype, per_frame.dtype)
            assert np.array_equal(mask, ~atd.ball_present[kickoff:stop]), half
            assert np.array_equal(per_frame, counts[kickoff:stop]), half
            assert not mask.flags.writeable and not per_frame.flags.writeable

        # So do the statistics (the graded blocks, frames 50-279 and 320-559, hold 470 frames, and are not used): 440.
        in_span = np.zeros(inputs.n_frames, dtype=bool)
        in_span[60:280] = in_span[340:560] = True
        assert cal.scalars["frames"] == 440 == int(in_span.sum()) and int(inputs.graded_mask.sum()) == 470
        assert cal.scalars["missing_frames"] == int((~atd.ball_present[in_span]).sum())
        assert cal.scalars["dropped_frames"] == cal.scalars["dropped_slot_frames"] == 0
        assert cal.present_hist.shape == (47,) and cal.present_hist.tolist() == np.bincount(
            counts[in_span], minlength=47).tolist()
        assert abs(cal.scalars["present_mean"] - float(counts[in_span].mean())) < 1e-9
        direct = calibration.ball_stats(atd.ball_present, spans, inputs.metres(atd.ball_xy), inputs.pitch, inputs.fps)
        assert cal.gap_durations_s.tolist() == direct["gap_durations_s"].tolist()
        assert cal.gap_onset_minute.tolist() == direct["gap_onset_minute"].tolist()
        assert cal.gap_onset_near_line.tolist() == direct["gap_onset_near_line"].tolist()
        # The interior gaps are three parallel arrays, and every gap is in the first.
        assert cal.interior_durations_s.tolist() == direct["interior_durations_s"].tolist()
        assert cal.interior_durations_s.shape == cal.gap_onset_minute.shape == cal.gap_onset_near_line.shape
        assert 0 < cal.interior_durations_s.size <= cal.gap_durations_s.size == cal.scalars["gaps"]
        assert cal.scalars["interior_gaps"] == cal.interior_durations_s.size
        assert cal.gap_onset_minute.min() >= 0.0, "the spans start at the kick-offs: no minute is before one"
        assert set(cal.interior_durations_s.tolist()) <= set(cal.gap_durations_s.tolist())

        # A02's table is the exception: the graded blocks, from their first frames (50 and 320), whole blocks of 230 and
        # 240 frames here, one segment each.
        for half, start, stop in inputs.graded:
            assert cal.missing_share_5min[half].shape == (1,)
            assert abs(cal.missing_share_5min[half][0] - float((~atd.ball_present[start:stop]).mean())) < 1e-12, half

        # The ball's speed channel is NaN on this share of the frames of the spans that have a ball (T31).
        seen = in_span & atd.ball_present
        assert abs(cal.scalars["ball_speed_nan_share"] - float(np.isnan(atd.ball_speed[seen]).mean())) < 1e-9

        # The public window: the nearest player's agreement by pipeline.evidence.describe, against a frame-by-frame
        # loop; the share of dead frames on which the ball is visible.
        share, judged = brute_force_agreement(inputs, gt)
        assert judged > 20, "the synthetic public truth must judge some frames"
        assert cal.public_window["judged_frames"] == judged, (cal.public_window, judged)
        assert abs(cal.public_window["nearest_agreement"] - share) < 1e-9, (cal.public_window, share)
        dead = gt == D
        assert cal.public_window["dead_frames"] == int(dead.sum()) == 20
        assert abs(cal.public_window["dead_ball_visible_share"] - float(atd.ball_present[dead].mean())) < 1e-9

        # The yardsticks: through run.predict and summarise, against pipeline.run.run on the same config, its CSV
        # written in the temporary folder and read nowhere else. Over the graded halves, as the published ones are.
        assert sorted(cal.baseline_outputs) == [0, 25, 50, 125], cal.baseline_outputs
        for width in sorted(cal.baseline_outputs):
            config = load_config(os.path.join(calibration.CONFIG_FOLDER, "nearest_w{}.toml".format(width)))
            result = run(config, inputs, os.path.join(root, "w{}.csv".format(width)), log=quiet)
            summary = result.summary
            minutes = summary["graded_frames"] / inputs.fps / 60
            got = cal.baseline_outputs[width]
            assert got["possessions"] == summary["possessions"], (width, got, summary["possessions"])
            assert abs(got["poss_per_min"] - summary["possessions"] / minutes) < 1e-8, (width, got)
            assert abs(got["d_share"] - summary["labels_graded"]["D"] / summary["graded_frames"]) < 1e-8, (width, got)

        # The hazard (A08) is what fit_hazard gives on the same arrays, spans and interior gaps, and its counts are the
        # scalars: every onset is either a row of the fit or counted as lost to an undefined feature.
        fitted = calibration.fit_hazard(
            atd.ball_present, inputs.metres(atd.ball_xy), atd.players_present, inputs.metres(atd.players_xy), spans,
            inputs.pitch, inputs.fps, direct["interior_durations_s"], direct["gap_onset_minute"],
            direct["gap_onset_near_line"])
        for name in ("coef", "mean", "scale"):
            assert getattr(cal.hazard_model, name).tobytes() == getattr(fitted["model"], name).tobytes(), name
            assert getattr(cal.hazard_model, name).shape == (len(hazard.FEATURES),)
        assert cal.hazard_model.intercept == fitted["model"].intercept
        assert (cal.hazard_l2, cal.hazard_c, cal.hazard_auc, cal.hazard_go) == (
            fitted["l2"], fitted["c"], fitted["auc"], fitted["go"])
        assert cal.hazard_fallback_cells == fitted["fallback_cells"] and cal.hazard_go in (True, False)
        assert sorted(cal.hazard_durations_s) == [0, 1] and all(sorted(row) == [0, 1, 2] for row in
                                                                cal.hazard_durations_s.values())
        for cls in (0, 1):
            for third in (0, 1, 2):
                cell = cal.hazard_durations_s[cls][third]
                assert cell.tolist() == fitted["durations"][cls][third].tolist() and cell.size > 0
                assert not cell.flags.writeable
        for name, value in fitted["counts"].items():
            assert cal.scalars["hazard_" + name] == value, name
        for name, value in fitted["cross_l2"].items():
            assert cal.scalars["hazard_l2_" + name] == value, name
        assert cal.scalars["hazard_events"] + cal.scalars["hazard_nan_events"] == cal.scalars["interior_gaps"]
        assert cal.scalars["hazard_visible"] == cal.scalars["frames"] - cal.scalars["missing_frames"]
        assert abs(cal.hazard_c - 1.0) < 1e-6, "the intercept is not penalised: the mean probability is the rate"

        # The same inputs give the same arrays and the same description, byte for byte.
        again = calibration.measure(inputs, gt)
        assert calibration.digest(again) == calibration.digest(cal)
        assert calibration.describe(again) == calibration.describe(cal)
        assert calibration.digest(cal) != calibration.digest(calibration.measure(inputs, np.where(gt == D, A, gt)
                                                                                 .astype(np.int8)))


def test_calibration_measure_refuses_what_it_cannot_measure():
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        expect_error(ValueError, lambda: calibration.measure(inputs, gt[:-1]), "public_gt_codes", "599", "600")
        inputs.kickoffs = {1: 60, 2: 560}
        expect_error(ValueError, lambda: calibration.measure(inputs, gt), "half 2", "kick-off", "560", "320")
        inputs.kickoffs = {1: 60}
        expect_error(ValueError, lambda: calibration.measure(inputs, gt), "kick-off", "half 2")


def test_calibration_names_the_yardstick_without_the_nearest_source():
    """measure() reads the public window from the nearest_player source: a config without it is named, not a crash."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        folder = os.path.join(root, "configs")
        shutil.copytree(calibration.CONFIG_FOLDER, folder)
        path = os.path.join(folder, "nearest_w0.toml")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert text.count('kind = "nearest_player"') == 1
        renamed = text.replace('kind = "nearest_player"', 'kind = "nearest_player"' + chr(10) + 'name = "nearest"')
        with open(path, "w", encoding="utf-8", newline=chr(10)) as handle:   # the same source, under another name
            handle.write(renamed)
        real = calibration.CONFIG_FOLDER
        calibration.CONFIG_FOLDER = folder
        try:
            expect_error(ValueError, lambda: calibration.measure(inputs, gt), "nearest_w0", "nearest_player")
        finally:
            calibration.CONFIG_FOLDER = real


def test_calibration_describe_survives_a_minute_before_the_kickoff():
    """A negative onset minute (a Calibration of graded blocks, say) is counted in the first bin, not a crash."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        odd = dataclasses.replace(cal, gap_onset_minute=np.array([-1.5, 3.0, 25.0, 1000.0]),
                                  interior_durations_s=np.ones(4), gap_onset_near_line=np.zeros(4, dtype=bool))
        bins = calibration.describe(odd)["onsets_per_10_min"]
        assert len(bins) == calibration.MAX_BINS and bins[0] == 2 and bins[2] == 1 and bins[-1] == 1, bins
        assert sum(bins) == 4, bins
        assert calibration.describe(dataclasses.replace(
            cal, gap_onset_minute=np.zeros(0), interior_durations_s=np.zeros(0),
            gap_onset_near_line=np.zeros(0, dtype=bool)))["onsets_per_10_min"] == []


# -- the hazard fit (A08) ---------------------------------------------------------------------------------------------

def hazard_fixture():
    """(ball_present, ball_xy_m, players_present, players_xy_m, blocks, pitch, fps): ball_fixture() and two players.

    The players stand still on the pitch, present on every frame (a slot of (60, 40) and one of (50, 30)), so that the
    only feature that is ever undefined is the ball's speed, on a frame with no defined step yet.
    """
    present, xy, blocks, pitch, fps = ball_fixture()
    n = len(present)
    return (present, xy, np.ones((n, 2), dtype=bool), np.tile(np.array([(60.0, 40.0), (50.0, 30.0)]), (n, 1, 1)),
            blocks, pitch, fps)


def test_calibration_v2_synthetic():
    """The hazard's rows, worked out by hand on the ball fixture, and its duration cells with their fallback."""
    present, xy, players_present, players_xy, blocks, pitch, fps = hazard_fixture()
    given = [array.copy() for array in (present, xy, players_present, players_xy)]
    rows, counts = calibration.hazard_rows(present, xy, players_present, players_xy, blocks, pitch, fps)

    # Block 1 (frames 10-709, r = 0-699 inside it) has 555 visible frames: 700 less g1 (20), g2 (5), g3 (20), g4 (100).
    # r = 599 is the visible frame before g4, which runs to the end of the block: neither an event nor a non-event, so it
    # is left out and counted (edge). The last frame of the block is in g4, so no frame is the last visible one. r = 0
    # has no step yet: its speed is NaN, so it is left out and counted. That leaves 553 rows, with 3 events: r = 99, 149
    # and 329, the visible frames before the interior gaps g1, g2 and g3.
    # Block 2 (710-1009, r = 0-299) has 280 visible frames: 300 less g5 (10), g6 (5) and g7 (5). g5 touches the start of
    # the block, so no visible frame precedes it. r = 299 is visible and the last frame: nothing follows it (last_frame).
    # r = 10 is the first visible frame, with no step before it: NaN speed. 278 rows, with 2 events: r = 199 and 249.
    assert counts == {"visible": 835, "edge": 1, "last_frame": 1, "nan_rows": 2, "nan_events": 0, "rows": 831,
                      "events": 5, "rows_1": 553, "events_1": 3, "rows_2": 278, "events_2": 2}, counts
    assert counts["visible"] == counts["edge"] + counts["last_frame"] + counts["nan_rows"] + counts["rows"]

    line, minute = hazard.FEATURES.index("line_distance_m"), hazard.FEATURES.index("minute_of_half")
    (x1, y1), (x2, y2) = rows[1], rows[2]
    assert sorted(rows) == [1, 2] and x1.shape == (553, 6) and x2.shape == (278, 6), (x1.shape, x2.shape)
    assert y1.shape == (553,) and y2.shape == (278,) and set(y1.tolist()) == set(y2.tolist()) == {0.0, 1.0}
    assert not np.isnan(x1).any() and not np.isnan(x2).any(), "a row with an undefined feature is left out"
    assert y1.sum() == 3 and y2.sum() == 2
    # The features are those of the at-risk frame r itself (the last visible frame), not of the frame after it: the
    # event rows carry the line distance and the minute of the onsets of the fixture (1.5 m, exactly 2 m and beyond the
    # line in block 1; 1.0 m and 105.3 - 104.0 in block 2), at one frame a second.
    assert np.allclose(x1[y1 == 1][:, line], [1.5, 2.0, -0.5], rtol=0, atol=1e-12), x1[y1 == 1][:, line]
    assert np.allclose(x1[y1 == 1][:, minute], np.array([99, 149, 329]) / 60.0, rtol=0, atol=1e-12)
    assert np.allclose(x2[y2 == 1][:, line], [1.0, 105.3 - 104.0], rtol=0, atol=1e-12), x2[y2 == 1][:, line]
    assert np.allclose(x2[y2 == 1][:, minute], np.array([199, 249]) / 60.0, rtol=0, atol=1e-12)
    # The rows themselves, by their minute (r / 60): r = 0 is out and r = 1 is the first; r = 598 is in and r = 599 (the
    # edge) is out; in block 2 r = 11 is the first and r = 298 the last (r = 299 is the last frame).
    assert np.isclose(x1[0, minute], 1 / 60) and np.isclose(x2[0, minute], 11 / 60) and np.isclose(
        x2[-1, minute], 298 / 60), (x1[0, minute], x2[0, minute], x2[-1, minute])
    assert np.isclose(x1[:, minute], 598 / 60).sum() == 1 and not np.isclose(x1[:, minute], 599 / 60).any()
    # The speed of the first frame after a gap is the one from before it (carried), so it is defined and the row stays:
    # r = 120 after g1 and r = 155 after g2.
    assert np.isclose(x1[:, minute], 120 / 60).sum() == 1 and np.isclose(x1[:, minute], 155 / 60).sum() == 1

    # The same rows come out of the same arrays, and the arrays given are not changed.
    again, again_counts = calibration.hazard_rows(present, xy, players_present, players_xy, blocks, pitch, fps)
    assert again_counts == counts and all(np.array_equal(again[h][0], rows[h][0]) and np.array_equal(
        again[h][1], rows[h][1]) for h in (1, 2))
    assert all(np.array_equal(before, after, equal_nan=True) for before, after in zip(
        given, (present, xy, players_present, players_xy))), "hazard_rows changed what it was given"

    # No gap at all: every frame but the last is a non-event; a ball that is never seen: no row at all.
    clear, clear_counts = calibration.hazard_rows(
        np.ones(50, dtype=bool), np.full((50, 2), 10.0), np.ones((50, 1), dtype=bool), np.full((50, 1, 2), 20.0),
        [(1, 0, 50)], pitch, fps)
    assert clear[1][0].shape == (48, 6) and clear[1][1].sum() == 0 and clear_counts["last_frame"] == 1
    assert clear_counts["nan_rows"] == 1 and clear_counts["events"] == 0, clear_counts   # r = 0: no step yet
    blind, blind_counts = calibration.hazard_rows(
        np.zeros(50, dtype=bool), np.full((50, 2), np.nan), np.ones((50, 1), dtype=bool), np.full((50, 1, 2), 20.0),
        [(1, 0, 50)], pitch, fps)
    assert blind[1][0].shape == (0, 6) and blind_counts["visible"] == 0 and blind_counts["rows"] == 0, blind_counts

    check_duration_cells()


def check_duration_cells():
    """duration_cells on hand-built arrays: classes by the near-line flag, thirds by the minute (cuts at 15 and 30), and
    a cell with fewer than 5 durations takes its class's durations over all thirds."""
    # (duration in s, onset minute, near a line), in the order the Calibration holds them: half 1 and then half 2, so
    # the minutes are not sorted. Class 0: 6, 4 and 5 durations in the thirds 0, 1 and 2 (minutes 15.0 and 30.0 begin the
    # second and the last third); class 1: none, 5 and 7.
    entries = [
        (1.0, 0.5, False), (2.0, 3.0, False), (3.0, 7.0, False), (4.0, 10.0, False), (5.0, 14.9, False),
        (6.0, 14.99, False),                                                                 # class 0, third 0: 6
        (11.0, 15.0, False), (12.0, 20.0, False), (13.0, 29.9, False), (14.0, 29.99, False),   # class 0, third 1: 4
        (21.0, 30.0, False), (22.0, 35.0, False), (23.0, 41.0, False), (24.0, 50.0, False),
        (25.0, 52.0, False),                                                                 # class 0, third 2: 5
        (31.0, 16.0, True), (32.0, 17.0, True), (33.0, 18.0, True), (34.0, 25.0, True),
        (35.0, 29.0, True),                                                                  # class 1, third 1: 5
        (41.0, 31.0, True), (42.0, 32.0, True), (43.0, 33.0, True), (44.0, 34.0, True), (45.0, 45.0, True),
        (46.0, 46.0, True), (47.0, 47.0, True),                                              # class 1, third 2: 7
    ]
    order = np.random.default_rng([16, 2]).permutation(len(entries))
    entries = [entries[i] for i in order]
    durations = np.array([e[0] for e in entries])
    minutes = np.array([e[1] for e in entries])
    near = np.array([e[2] for e in entries])

    table, fallback, own = calibration.duration_cells(durations, minutes, near)

    def inside(cls, third):
        """The durations of a cell, in the order they were given."""
        return [d for d, m, n in entries if n == bool(cls) and (0 if m < 15 else 1 if m < 30 else 2) == third]

    assert own == {(0, 0): 6, (0, 1): 4, (0, 2): 5, (1, 0): 0, (1, 1): 5, (1, 2): 7}, own
    # A cell with 5 or more uses its own; 4 and 0 fall back to their class pooled over the thirds (in the given order).
    assert fallback == ((0, 1), (1, 0)), fallback
    class0 = [d for d, _, n in entries if not n]
    class1 = [d for d, _, n in entries if n]
    assert len(class0) == 15 and len(class1) == 12
    expected = {(0, 0): inside(0, 0), (0, 1): class0, (0, 2): inside(0, 2),
                (1, 0): class1, (1, 1): inside(1, 1), (1, 2): inside(1, 2)}
    assert sorted(table) == [0, 1] and all(sorted(table[cls]) == [0, 1, 2] for cls in (0, 1))
    for (cls, third), values in expected.items():
        cell = table[cls][third]
        assert cell.dtype == np.float64 and cell.tolist() == values, (cls, third, cell, values)
        assert not cell.flags.writeable, "seconds, read-only"
    assert [len(inside(0, t)) for t in (0, 1, 2)] == [6, 4, 5] and [len(inside(1, t)) for t in (0, 1, 2)] == [0, 5, 7]

    # No gap of a class at all: there is nothing to draw its durations from, and the error says which class.
    expect_error(ValueError, lambda: calibration.duration_cells(durations[~near], minutes[~near], near[~near]),
                 "duration_cells", "onset class 1", "near")
    expect_error(ValueError, lambda: calibration.duration_cells(durations[near], minutes[near], near[near]),
                 "duration_cells", "onset class 0")
    expect_error(ValueError, lambda: calibration.duration_cells(durations[:-1], minutes, near), "duration_cells",
                 "interior_durations_s", "gap_onset_minute", "parallel")
    expect_error(ValueError, lambda: calibration.duration_cells(durations, np.where(near, np.nan, minutes), near),
                 "third_of_minute", "NaN")


def hazard_match(seed, near_rate, far_rate, frames=60_000):
    """(ball_present, ball_xy_m, players_present, players_xy_m, spans, pitch, fps): two halves of `frames` frames, 25 a
    second, with the ball scattered over the pitch (a metre beyond it too), whose gaps start with probability near_rate
    on a frame less than 2 m from a line and far_rate elsewhere, and last 5 to 39 frames. The hazard depends on the
    line distance and on nothing else, so the model has one thing to find (or nothing, when the two rates are equal)."""
    rng = np.random.default_rng([16, 3, seed])
    n = 2 * frames
    xy = rng.uniform((-1.0, -1.0), (PITCH[0] + 1.0, PITCH[1] + 1.0), size=(n, 2))
    line = np.minimum.reduce([xy[:, 0], PITCH[0] - xy[:, 0], xy[:, 1], PITCH[1] - xy[:, 1]])
    rate = np.where(line < calibration.NEAR_LINE_M, near_rate, far_rate)
    draw, length = rng.random(n), rng.integers(5, 40, size=n)
    present = np.ones(n, dtype=bool)
    for start, stop in ((0, frames), (frames, n)):
        r = start
        while r < stop - 1:
            if draw[r] < rate[r]:
                present[r + 1:min(r + 1 + length[r], stop)] = False
                r += length[r] + 1
            else:
                r += 1
    xy[~present] = np.nan
    players = rng.uniform((0.0, 0.0), PITCH, size=(n, 3, 2))
    return present, xy, np.ones((n, 3), dtype=bool), players, [(1, 0, frames), (2, frames, n)], PITCH, 25


def fit_of(match):
    """(fit_hazard's result, ball_stats') for a hazard_match; the interior-gap arrays come from the same spans."""
    present, xy, players_present, players, spans, pitch, fps = match
    ball = calibration.ball_stats(present, spans, xy, pitch, fps)
    return calibration.fit_hazard(present, xy, players_present, players, spans, pitch, fps,
                                  ball["interior_durations_s"], ball["gap_onset_minute"],
                                  ball["gap_onset_near_line"]), ball


def test_calibration_hazard_finds_a_signal_and_not_a_flat_hazard():
    """Gaps that start near the lines: both cross-half AUCs high, a negative line-distance coefficient, and go. Gaps
    at one flat rate: nothing to find, AUCs near 0.5, and no-go. Both on seeded synthetic matches."""
    result, ball = fit_of(hazard_match(1, near_rate=0.04, far_rate=0.002))
    auc = result["auc"]
    assert sorted(auc) == sorted(calibration.HAZARD_AUC_KEYS) == ["full_1to2", "full_2to1", "time_1to2", "time_2to1"]
    assert all(type(value) is float and 0.0 < value < 1.0 for value in auc.values()), auc
    assert auc["full_1to2"] >= 0.7 and auc["full_2to1"] >= 0.7, auc
    assert result["go"] is True
    # Time carries nothing here, and the full model beats it by a wide margin.
    assert abs(auc["time_1to2"] - 0.5) < 0.06 and abs(auc["time_2to1"] - 0.5) < 0.06, auc
    model = result["model"]
    assert model.coef.shape == model.mean.shape == model.scale.shape == (len(hazard.FEATURES),)
    assert model.coef[hazard.FEATURES.index("line_distance_m")] < -0.3, model.coef
    # c is 1 on the rows it was fitted on: the intercept is not penalised, so the model's mean probability is the rate.
    assert abs(result["c"] - 1.0) < 1e-6 and result["l2"] == calibration.HAZARD_L2 == 1.0, (result["c"], result["l2"])
    assert set(result["cross_l2"]) == set(calibration.HAZARD_AUC_KEYS) and set(result["cross_l2"].values()) == {1.0}
    # Every at-risk frame is somewhere, and so is every gap.
    counts = result["counts"]
    assert counts["visible"] == counts["edge"] + counts["last_frame"] + counts["nan_rows"] + counts["rows"], counts
    assert counts["events"] + counts["nan_events"] == ball["interior_gaps"], (counts, ball["interior_gaps"])
    assert counts["events_1"] + counts["events_2"] == counts["events"] and counts["rows_1"] + counts["rows_2"] == counts[
        "rows"]
    assert counts["events"] > 400 and counts["nan_rows"] >= 2, counts   # r = 0 of each half has no step yet
    # The durations: seconds, 5 to 39 frames at 25 a second, in six cells with no fallback (every cell is full).
    cells = result["durations"]
    assert all(cells[c][t].size >= calibration.MIN_CELL_DURATIONS and 0.2 <= cells[c][t].min()
               and cells[c][t].max() <= 39 / 25 for c in (0, 1) for t in (0, 1, 2)), cells
    assert result["fallback_cells"] == (), result["fallback_cells"]
    assert sum(counts["cell_{}_{}".format(c, t)] for c in (0, 1) for t in (0, 1, 2)) == ball["interior_gaps"]

    # The same arrays give the same fit, byte for byte.
    again, _ = fit_of(hazard_match(1, near_rate=0.04, far_rate=0.002))
    assert again["model"].coef.tobytes() == model.coef.tobytes() and again["model"].intercept == model.intercept
    assert again["auc"] == auc and again["c"] == result["c"] and again["go"] == result["go"]

    flat, _ = fit_of(hazard_match(2, near_rate=0.005, far_rate=0.005))
    assert max(flat["auc"].values()) < 0.6 and flat["go"] is False, flat["auc"]
    assert min(flat["auc"].values()) > 0.4, flat["auc"]


def test_calibration_rate_scale_is_the_rate_over_the_mean_probability():
    """c, worked out by hand: 1 onset in 4 frames at a mean probability of 0.1 is 2.5; and what it refuses."""
    assert calibration.rate_scale([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1]) == 2.5
    assert calibration.rate_scale([0, 1, 0, 1], [0.1, 0.4, 0.1, 0.4]) == 2.0     # 0.5 over a mean of 0.25
    assert calibration.rate_scale([1, 0], [0.25, 0.75]) == 1.0                    # the rate is the mean probability
    assert calibration.rate_scale([1, 0, 0, 0], [1 / 3] * 4) == 0.75
    assert type(calibration.rate_scale([1, 0], [0.5, 0.5])) is float
    expect_error(ValueError, lambda: calibration.rate_scale([1, 0], [0.5]), "rate_scale", "one length")
    expect_error(ValueError, lambda: calibration.rate_scale([], []), "rate_scale", "not empty")
    expect_error(ValueError, lambda: calibration.rate_scale([1, 0], [0.5, 0.0]), "rate_scale", "above 0")


def test_calibration_digest_covers_the_hazard():
    """Two Calibrations that differ in any hazard field, even in the ninth decimal of a coefficient, are not the same."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        same = calibration.digest(cal)
        model = cal.hazard_model
        nudged = np.array(model.coef) + np.array([1e-9] + [0.0] * (len(model.coef) - 1))
        durations = {cls: dict(thirds) for cls, thirds in cal.hazard_durations_s.items()}
        durations[1][2] = np.append(durations[1][2], 1.0)
        changes = {
            "coef": dataclasses.replace(model, coef=nudged), "intercept": dataclasses.replace(
                model, intercept=model.intercept + 1e-9),
            "mean": dataclasses.replace(model, mean=np.array(model.mean) + 1e-9),
            "scale": dataclasses.replace(model, scale=np.array(model.scale) + 1e-9)}
        for name, other in changes.items():
            assert calibration.digest(dataclasses.replace(cal, hazard_model=other)) != same, name
        for name, value in (("hazard_l2", cal.hazard_l2 * 10), ("hazard_c", cal.hazard_c + 1e-9),
                            ("hazard_auc", {**cal.hazard_auc, "full_1to2": cal.hazard_auc["full_1to2"] + 1e-9}),
                            ("hazard_go", not cal.hazard_go), ("hazard_durations_s", durations),
                            ("hazard_fallback_cells", cal.hazard_fallback_cells + ((1, 1),))):
            assert calibration.digest(dataclasses.replace(cal, **{name: value})) != same, name
        assert calibration.digest(dataclasses.replace(cal)) == same


def test_calibration_hazard_go_needs_both_full_aucs_at_the_threshold():
    """go is full_1to2 >= 0.60 and full_2to1 >= 0.60: the threshold counts, and one half is not enough."""
    match = hazard_match(4, near_rate=0.04, far_rate=0.002, frames=20_000)
    present, xy, players_present, players, spans, pitch, fps = match
    ball = calibration.ball_stats(present, spans, xy, pitch, fps)
    real = hazard.auc

    def fake_auc(first, second):
        """The two full-model AUCs are the values scripted; the time-only ones are 0.5."""
        scripted = iter([first, second, 0.5, 0.5])
        return lambda scores, labels: next(scripted)

    def go(first, second):
        hazard.auc = fake_auc(first, second)
        try:
            return calibration.fit_hazard(present, xy, players_present, players, spans, pitch, fps,
                                          ball["interior_durations_s"], ball["gap_onset_minute"],
                                          ball["gap_onset_near_line"])
        finally:
            hazard.auc = real

    assert go(0.60, 0.60)["go"] is True, "0.60 is enough"
    assert go(0.5999, 0.9)["go"] is False and go(0.9, 0.5999)["go"] is False
    assert go(0.9, 0.9)["go"] is True and go(0.4, 0.4)["go"] is False
    # The decision is on the stored (rounded to 1e-9) values: 0.5999999996 is stored as 0.6.
    assert go(0.5999999996, 0.5999999996)["go"] is True
    assert go(0.5999999994, 0.9)["go"] is False


def test_calibration_hazard_fits_one_half_and_scores_the_other():
    """The four AUCs are cross-half: a model fitted on half a is scored on half b, the full one on the six features and
    the time-only one on the minute alone, in the order of HAZARD_AUC_KEYS; the final model is fitted on both pooled."""
    match = hazard_match(6, near_rate=0.04, far_rate=0.002, frames=20_000)
    present, xy, players_present, players, spans, pitch, fps = match
    rows, counts = calibration.hazard_rows(present, xy, players_present, players, spans, pitch, fps)
    one, two = len(rows[1][1]), len(rows[2][1])
    assert one != two and counts["rows_1"] == one and counts["rows_2"] == two
    real_fit, real_auc = hazard.fit_logistic, hazard.auc
    fits, scored = [], []

    def fit(X, y, l2=1.0, iterations=50):
        fits.append((len(y), X.shape[1], float(y.sum())))
        return real_fit(X, y, l2=l2, iterations=iterations)

    def auc(scores, labels):
        scored.append((len(labels), int(np.sum(labels))))
        return real_auc(scores, labels)

    hazard.fit_logistic, hazard.auc = fit, auc
    try:
        result, _ = fit_of(match)
    finally:
        hazard.fit_logistic, hazard.auc = real_fit, real_auc
    events = {1: int(rows[1][1].sum()), 2: int(rows[2][1].sum())}
    # Fitted on: half 1, half 2 (the full model, six columns), half 1, half 2 (the minute, one column), both pooled.
    assert fits == [(one, 6, events[1]), (two, 6, events[2]), (one, 1, events[1]), (two, 1, events[2]),
                    (one + two, 6, events[1] + events[2])], fits
    # Scored on: half 2, half 1, half 2, half 1.
    assert scored == [(two, events[2]), (one, events[1]), (two, events[2]), (one, events[1])], scored
    # And the number stored under each key is that fit's AUC, so a swap of the two halves cannot hide: the two AUCs of a
    # model differ on this match, and each is recomputed here from its own half.
    assert result["auc"]["full_1to2"] != result["auc"]["full_2to1"] and result["auc"]["time_1to2"] != result["auc"][
        "time_2to1"], result["auc"]
    model_1 = real_fit(rows[1][0], rows[1][1])
    expected = real_auc(hazard.predict(model_1, rows[2][0]), rows[2][1])
    assert result["auc"]["full_1to2"] == round(expected, 9), (result["auc"]["full_1to2"], expected)
    model_2 = real_fit(rows[2][0][:, [5]], rows[2][1])
    expected = real_auc(hazard.predict(model_2, rows[1][0][:, [5]]), rows[1][1])
    assert result["auc"]["time_2to1"] == round(expected, 9), (result["auc"]["time_2to1"], expected)


def test_calibration_hazard_refits_with_a_larger_penalty_and_never_swallows():
    """A fit that does not converge is refitted with the penalty times 10, at most 3 times, and the l2 used is kept; when
    every penalty fails the error says so; any other error is raised at once."""
    match = hazard_match(3, near_rate=0.04, far_rate=0.002, frames=20_000)
    real = hazard.fit_logistic
    calls = []

    def needs(l2_needed):
        def fake(X, y, l2=1.0, iterations=50):
            calls.append(l2)
            if l2 < l2_needed:
                raise hazard.NotConverged("hazard.fit_logistic: no convergence in a stub")
            return real(X, y, l2=l2, iterations=iterations)
        return fake

    try:
        hazard.fit_logistic = needs(100.0)
        result, _ = fit_of(match)
        assert result["l2"] == 100.0 and result["cross_l2"] == {key: 100.0 for key in calibration.HAZARD_AUC_KEYS}
        assert calls == [1.0, 10.0, 100.0] * 5, calls   # four cross-half fits and the pooled one, each refitted twice
        # What comes out is the fit at that penalty, and not the fit at the first one.
        pooled = real(*pooled_rows(match), l2=100.0)
        assert result["model"].coef.tobytes() == pooled.coef.tobytes() and result["model"].intercept == pooled.intercept
        hazard.fit_logistic = real
        assert fit_of(match)[0]["model"].coef.tobytes() != pooled.coef.tobytes()

        calls.clear()
        hazard.fit_logistic = needs(1000.0)
        assert fit_of(match)[0]["l2"] == 1000.0 and calls == [1.0, 10.0, 100.0, 1000.0] * 5, calls

        calls.clear()
        hazard.fit_logistic = needs(1e4)    # 1, 10, 100 and 1,000 all fail: that is the end of it
        expect_error(ValueError, lambda: fit_of(match), "calibration", "hazard", "did not converge",
                     "l2 = 1, 10, 100, 1000", "no convergence in a stub")
        assert calls == [1.0, 10.0, 100.0, 1000.0], calls

        calls.clear()

        def broken(X, y, l2=1.0, iterations=50):
            calls.append(l2)
            raise ValueError("hazard.fit_logistic: X has 3 NaN")

        hazard.fit_logistic = broken
        expect_error(ValueError, lambda: fit_of(match), "X has 3 NaN")
        assert calls == [1.0], "a bad input is not a convergence problem, and no penalty cures it: {}".format(calls)
    finally:
        hazard.fit_logistic = real


def pooled_rows(match):
    """(X, y) of both halves of a hazard_match, in the order fit_hazard pools them."""
    present, xy, players_present, players, spans, pitch, fps = match
    rows, _ = calibration.hazard_rows(present, xy, players_present, players, spans, pitch, fps)
    return np.concatenate([rows[1][0], rows[2][0]]), np.concatenate([rows[1][1], rows[2][1]])


def test_calibration_hazard_refuses_what_it_cannot_fit():
    present, xy, players_present, players, spans, pitch, fps = hazard_match(5, 0.04, 0.002, frames=10_000)
    ball = calibration.ball_stats(present, spans, xy, pitch, fps)
    arrays = (ball["interior_durations_s"], ball["gap_onset_minute"], ball["gap_onset_near_line"])

    def fit(present=present, xy=xy, players_present=players_present, players=players, spans=spans, arrays=arrays):
        return lambda: calibration.fit_hazard(present, xy, players_present, players, spans, pitch, fps, *arrays)

    fit()()   # the arguments are good
    expect_error(ValueError, fit(present=present.astype(int)), "ball_present", "bool")
    expect_error(ValueError, fit(xy=xy[:-1]), "hazard_rows", "ball_present", "ball_xy_m")
    expect_error(ValueError, fit(players=players[:-1]), "hazard_rows", "players_xy_m")
    expect_error(ValueError, fit(players_present=players_present[:, :2]), "hazard_rows", "players_present",
                 "players_xy_m")
    # The hazard is fitted on one half and scored on the other: it needs halves 1 and 2.
    expect_error(ValueError, fit(spans=spans[:1]), "halves 1 and 2")
    expect_error(ValueError, fit(spans=[(1, 0, 10_000), (3, 10_000, 20_000)]), "halves 1 and 2")
    # The interior-gap arrays have to be those of the same spans: the counts of events must agree.
    expect_error(ValueError, fit(arrays=tuple(a[:-1] for a in arrays)), "interior", "onsets", "same spans")
    # A half without an onset (no gap in it) cannot be scored, and the error names the half.
    gapless = present.copy()
    gapless[10_000:] = True
    xy_gapless = np.where(gapless[:, None], np.where(np.isnan(xy), 30.0, xy), xy)
    ball_gapless = calibration.ball_stats(gapless, spans, xy_gapless, pitch, fps)
    expect_error(ValueError, lambda: calibration.fit_hazard(
        gapless, xy_gapless, players_present, players, spans, pitch, fps, ball_gapless["interior_durations_s"],
        ball_gapless["gap_onset_minute"], ball_gapless["gap_onset_near_line"]), "half 2", "onsets", "both")
    expect_error(ValueError, lambda: calibration.hazard_rows(present, xy, players_present, players, [], pitch, fps),
                 "blocks")


# -- the cache --------------------------------------------------------------------------------------------------------

def snapshot(root):
    """{relative path: (size, mtime_ns)} of every file under root."""
    found = {}
    for folder, _, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            stat = os.stat(path)
            found[os.path.relpath(path, root).replace(os.sep, "/")] = (stat.st_size, stat.st_mtime_ns)
    return found


def assert_no_frame_data(value, where="describe()"):
    """No array, nothing JSON cannot hold, and no sequence longer than MAX_SEQUENCE anywhere in `value`."""
    if isinstance(value, np.ndarray):
        raise AssertionError("{} holds an array of {} values".format(where, value.size))
    if isinstance(value, (list, tuple)):
        assert len(value) <= MAX_SEQUENCE, "{} holds a sequence of {} values".format(where, len(value))
        for index, item in enumerate(value):
            assert_no_frame_data(item, "{}[{}]".format(where, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, (str, int)) and not isinstance(key, bool), (where, key)
            assert_no_frame_data(item, "{}[{!r}]".format(where, key))
    else:
        assert value is None or type(value) in (bool, int, float, str), "{}: {} is not a plain scalar".format(
            where, type(value).__name__)


def expected_digest(key):
    """The folder name of a cache key, worked out here from its definition: the first 16 hex digits of the sha256 of the
    key's canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def test_calibration_cache_inside_data_dir():
    """load() writes only under <data dir>/cache/benchmark/calibration/<digest>/, and what it keeps there is reused."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        inputs.atd  # the ATD cache is the pipeline's, not this module's: build it first, so that it is not new below
        before = snapshot(root)
        cal = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        written = snapshot(root)
        # One folder per key: its name is the first 16 hex digits of the sha256 of the key's canonical JSON.
        digest = expected_digest(calibration._key(inputs, gt))
        folder = "cache/benchmark/calibration/{}/".format(digest)
        new = sorted(set(written) - set(before))
        assert new and all(path.startswith(folder) for path in new), new
        assert not [path for path in new if path.endswith((".csv", ".tmp"))], "no CSV, no leftover: {}".format(new)
        assert {path: stat for path, stat in written.items() if path in before} == before, "load() touched a file"
        assert folder + "manifest.json" in new
        assert os.path.join(root, *calibration.CACHE_FOLDER, digest) == calibration.cache_folder(
            root, calibration._key(inputs, gt))

        # Reused when fresh: nothing is rewritten, and what comes back is what was built.
        loaded = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert snapshot(root) == written, "a fresh cache must be read, not rewritten"
        assert calibration.digest(loaded) == calibration.digest(cal) == calibration.digest(
            calibration.measure(inputs, gt)), "the cache must give back exactly what measure() gives"
        assert calibration.describe(loaded) == calibration.describe(cal)
        for half in (1, 2):
            assert not loaded.ball_masks[half].flags.writeable and loaded.ball_masks[half].dtype == bool
            assert loaded.present_counts[half].dtype == np.int16
        # The hazard (A08) comes back as it went in: the model's arrays as read-only float64, the durations of the six
        # cells as read-only seconds, the AUCs, the decision and the cells that fell back as plain Python values.
        for name in ("coef", "mean", "scale"):
            got = getattr(loaded.hazard_model, name)
            assert got.dtype == np.float64 and not got.flags.writeable, name
            assert got.tobytes() == getattr(cal.hazard_model, name).tobytes(), name
        assert loaded.hazard_model.intercept == cal.hazard_model.intercept and type(
            loaded.hazard_model.intercept) is float
        assert (loaded.hazard_l2, loaded.hazard_c, loaded.hazard_auc, loaded.hazard_go) == (
            cal.hazard_l2, cal.hazard_c, cal.hazard_auc, cal.hazard_go)
        assert type(loaded.hazard_go) is bool and type(loaded.hazard_l2) is float and type(loaded.hazard_c) is float
        assert all(type(value) is float for value in loaded.hazard_auc.values())
        assert loaded.hazard_fallback_cells == cal.hazard_fallback_cells and type(loaded.hazard_fallback_cells) is tuple
        assert all(type(cell) is tuple for cell in loaded.hazard_fallback_cells)
        for cls in (0, 1):
            for third in (0, 1, 2):
                got = loaded.hazard_durations_s[cls][third]
                assert got.dtype == np.float64 and not got.flags.writeable and got.size > 0
                assert got.tobytes() == cal.hazard_durations_s[cls][third].tobytes()
        # rebuild=True parses again, into the same folder.
        calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt, rebuild=True)
        assert sorted(set(snapshot(root)) - set(before)) == new

        # describe() is for people: scalars and short lists, no frame of anything.
        description = calibration.describe(cal)
        assert_no_frame_data(description)
        json.dumps(description, sort_keys=True)   # and it is JSON, as a report will need it
        # The repr of a Calibration shows no array, so that a print or a traceback cannot leak a frame (CLAUDE.md).
        text = repr(cal)
        assert "array(" not in text and len(text) < 4000, text

        def holds_array(value):
            """An array, a dict that holds one at any depth, or the hazard's model (whose arrays a repr would show)."""
            if isinstance(value, dict):
                return any(holds_array(item) for item in value.values())
            return isinstance(value, (np.ndarray, hazard.LogisticModel))

        for field in dataclasses.fields(calibration.Calibration):
            value = getattr(cal, field.name)
            assert field.repr is (not holds_array(value)), "{}: repr must be {}".format(
                field.name, not holds_array(value))
        expect_error(dataclasses.FrozenInstanceError, lambda: setattr(cal, "wrong_team_share", 0.0), "wrong_team_share")


def test_calibration_cache_is_keyed_and_unreadable_means_absent():
    """A changed config, definition, version, truth or source gets a folder of its own; a damaged cache is rebuilt."""
    from pipeline.inputs import atd_cache
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        parent = os.path.join(root, "cache", "benchmark", "calibration")

        def folder(codes=gt, given=None):
            """The folder the current key (code, configs, versions, ... as they stand right now) is kept in."""
            return calibration.cache_folder(root, calibration._key(given or inputs, codes))

        def manifest(codes=gt, given=None):
            with open(os.path.join(folder(codes, given), "manifest.json"), encoding="utf-8") as handle:
                return json.load(handle)

        def load(codes=gt, given=None):
            return calibration.load(data_dir=root, inputs=given or inputs, public_gt_codes=codes)

        def folders():
            return sorted(os.listdir(parent))

        def trees():
            return {name: snapshot(os.path.join(parent, name)) for name in folders()}

        def adds_a_folder(action):
            """Run `action`; exactly one folder is new, and every old one is exactly as it was. Its name."""
            before = trees()
            action()
            after = trees()
            added = sorted(set(after) - set(before))
            assert len(added) == 1, (sorted(before), sorted(after))
            assert all(after[name] == files for name, files in before.items()), "an old key's folder was touched"
            return added[0]

        def adds_nothing(action):
            before = trees()
            action()
            assert trees() == before, "a key that is already cached must be read, not rebuilt or rewritten"

        cal = load()
        key = manifest()["key"]
        assert folders() == [expected_digest(key)], (folders(), key)
        assert key["calibration_version"] == calibration.CALIBRATION_VERSION
        assert key["code_sha256"] == calibration.CALIBRATION_SHA256
        assert key["hazard_sha256"] == calibration.HAZARD_SHA256, "hazard.py makes version 2's fields: it is in the key"
        assert key["atd_sources"] == inputs.atd_manifest["sources"], "the ATD cache's own record of its sources"
        assert key["atd_cache_version"] == inputs.atd_manifest["cache_version"] == atd_cache.CACHE_VERSION
        assert key["numpy"] == np.__version__ and sorted(key["configs"]) == ["0", "125", "25", "50"], key
        assert key["pitch"] == list(inputs.pitch) and key["fps"] == inputs.fps, "games.py's constants are in the key"

        # An edited config must not reuse the baseline outputs of the old one.
        real_folder = calibration.CONFIG_FOLDER
        edited = os.path.join(root, "configs")
        shutil.copytree(real_folder, edited)
        with open(os.path.join(edited, "nearest_w25.toml"), "a", encoding="utf-8", newline="\n") as handle:
            handle.write("# edited\n")
        calibration.CONFIG_FOLDER = edited
        try:
            name = adds_a_folder(load)
            assert name == expected_digest(manifest()["key"]) != expected_digest(key)
            changed = manifest()["key"]
            assert changed["configs"]["25"] != key["configs"]["25"]
            assert changed["configs"]["0"] == key["configs"]["0"] and changed["configs"]["50"] == key["configs"]["50"]
        finally:
            calibration.CONFIG_FOLDER = real_folder
        adds_nothing(load)   # the unedited configs are the first key again: its own folder, read as it is
        assert manifest()["key"] == key

        # A new version, another definition (the hash of calibration.py), another public truth, other kick-offs, and
        # another pitch or frame rate (games.py's constants, which every distance and every duration depends on).
        calibration.CALIBRATION_VERSION += 1
        try:
            adds_a_folder(load)
            assert manifest()["key"]["calibration_version"] == calibration.CALIBRATION_VERSION
        finally:
            calibration.CALIBRATION_VERSION -= 1
        adds_nothing(load)
        real_hash = calibration.CALIBRATION_SHA256
        calibration.CALIBRATION_SHA256 = "0" * 64
        try:
            adds_a_folder(load)
            assert manifest()["key"]["code_sha256"] == "0" * 64, "a different code hash must rebuild"
        finally:
            calibration.CALIBRATION_SHA256 = real_hash
        adds_nothing(load)
        # The same for hazard.py (its features, fit, AUC and thirds make the version-2 fields), with calibration.py as
        # it is: a checkout with another hazard.py must not read this one's fit.
        real_hazard_hash = calibration.HAZARD_SHA256
        calibration.HAZARD_SHA256 = "0" * 64
        try:
            adds_a_folder(load)
            changed = manifest()["key"]
            assert changed["hazard_sha256"] == "0" * 64, "a different hazard hash must rebuild"
            assert {k: v for k, v in changed.items() if k != "hazard_sha256"} == {
                k: v for k, v in key.items() if k != "hazard_sha256"}, "only the hazard hash differs"
        finally:
            calibration.HAZARD_SHA256 = real_hazard_hash
        adds_nothing(load)
        other = gt.copy()
        other[60:70] = A
        adds_a_folder(lambda: load(other))
        assert manifest(other)["key"]["public_gt_sha256"] != key["public_gt_sha256"]
        adds_nothing(load)
        inputs.kickoffs = {1: 61, 2: 340}
        adds_a_folder(load)
        assert manifest()["key"]["kickoffs"]["1"] == 61
        inputs.kickoffs = {1: 60, 2: 340}
        real_pitch, real_fps = inputs.pitch, inputs.fps
        inputs.pitch = (105.0, 68.0)
        adds_a_folder(load)
        assert manifest()["key"]["pitch"] == [105.0, 68.0]
        inputs.pitch, inputs.fps = real_pitch, 30
        adds_a_folder(load)
        assert manifest()["key"]["fps"] == 30
        inputs.fps = real_fps
        adds_nothing(load)
        assert manifest()["key"] == key

        # A new ATD parse (the ATD cache's cache_version) means new arrays under the same source files: rebuilt.
        atd_cache.CACHE_VERSION += 1
        try:
            reparsed, _ = synthetic_calibration_inputs(root)   # its ATD cache is stale, so it parses again
            adds_a_folder(lambda: load(given=reparsed))
            assert manifest(given=reparsed)["key"]["atd_cache_version"] == atd_cache.CACHE_VERSION == (
                key["atd_cache_version"] + 1)
        finally:
            atd_cache.CACHE_VERSION -= 1
        restored, _ = synthetic_calibration_inputs(root)
        adds_nothing(lambda: load(given=restored))
        assert manifest()["key"] == key and len(folders()) == 10, folders()

        # Unreadable is absent: an array cut short, an array gone, a manifest that is not JSON or lacks a field. The
        # damaged folder is the current key's own, and it is rebuilt in place.
        good = calibration.digest(cal)
        mine = folder()
        manifest_path = os.path.join(mine, "manifest.json")

        def edit_manifest(change):
            content = manifest()
            change(content)
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(content, handle)

        for damage in ("truncate", "delete", "garbage", "field", "empty", "not parallel", "no hazard", "short model",
                       "bad auc", "go not a bool", "zero scale", "NaN coef", "no cell", "empty cell", "cell not 1-D"):
            if damage == "truncate":
                with open(os.path.join(mine, "gap_durations_s.npy"), "wb") as handle:
                    handle.write(b"\x93NUMPY")
            elif damage == "delete":
                os.remove(os.path.join(mine, "ball_masks_1.npy"))
            elif damage == "garbage":
                with open(manifest_path, "wb") as handle:
                    handle.write(b"\x00 not json")
            elif damage == "field":
                content = manifest()
                del content["public_window"]
                with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(content, handle)
            elif damage == "not parallel":
                np.save(os.path.join(mine, "gap_onset_minute.npy"), np.zeros(1))   # valid array, wrong length
            elif damage == "no hazard":
                edit_manifest(lambda content: content.pop("hazard"))
            elif damage == "short model":
                edit_manifest(lambda content: content["hazard"]["model"].update(coef=[0.0]))
            elif damage == "bad auc":
                edit_manifest(lambda content: content["hazard"].update(auc={"full_1to2": 0.7}))
            elif damage == "go not a bool":
                edit_manifest(lambda content: content["hazard"].update(go="yes"))
            elif damage == "zero scale":
                edit_manifest(lambda content: content["hazard"]["model"].update(scale=[0.0] * len(hazard.FEATURES)))
            elif damage == "NaN coef":
                edit_manifest(lambda content: content["hazard"]["model"].update(coef=[math.nan] * len(hazard.FEATURES)))
            elif damage == "no cell":
                os.remove(os.path.join(mine, "hazard_durations_s_1_2.npy"))
            elif damage == "empty cell":
                np.save(os.path.join(mine, "hazard_durations_s_0_1.npy"), np.zeros(0))   # sample_mask needs a duration
            elif damage == "cell not 1-D":
                np.save(os.path.join(mine, "hazard_durations_s_0_0.npy"), np.zeros((2, 2)))
            else:
                os.remove(manifest_path)
            assert calibration.digest(load()) == good, damage
            assert os.path.getsize(os.path.join(mine, "gap_durations_s.npy")) > 10, damage
            assert len(folders()) == 10, damage
        assert calibration.digest(load()) == good


def test_calibration_each_key_has_its_own_folder():
    """Two checkouts with different code share one data dir: neither rebuilds the other's cache or touches its files."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        inputs.atd
        parent = os.path.join(root, *calibration.CACHE_FOLDER)

        def tree(name):
            return snapshot(os.path.join(parent, name))

        first = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        name_a = expected_digest(calibration._key(inputs, gt))
        assert os.listdir(parent) == [name_a] and len(name_a) == 16 and set(name_a) <= set("0123456789abcdef"), name_a
        files_a = tree(name_a)

        # The second checkout: other code, which is another key. It gets a folder, and leaves the first one as it was.
        real_hash = calibration.CALIBRATION_SHA256
        calibration.CALIBRATION_SHA256 = "f" * 64
        try:
            second = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
            name_b = expected_digest(calibration._key(inputs, gt))
            assert name_b != name_a and sorted(os.listdir(parent)) == sorted([name_a, name_b]), os.listdir(parent)
            assert tree(name_a) == files_a, "the first key's folder was touched"
            files_b = tree(name_b)
            assert files_b, "the second key has files of its own"

            # Both are reused: loading either again rewrites nothing, in either folder.
            calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
            assert tree(name_a) == files_a and tree(name_b) == files_b
        finally:
            calibration.CALIBRATION_SHA256 = real_hash
        again = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert tree(name_a) == files_a and tree(name_b) == files_b and len(os.listdir(parent)) == 2
        assert calibration.digest(again) == calibration.digest(first) == calibration.digest(second)

        # A key is the same key whatever order its entries come in: the folder name is of the canonical JSON.
        key = calibration._key(inputs, gt)
        shuffled = dict(reversed(list(key.items())))
        assert list(shuffled) != list(key) and expected_digest(shuffled) == expected_digest(key) == name_a
        assert calibration.cache_folder(root, shuffled) == calibration.cache_folder(root, key) == os.path.join(
            parent, name_a)
        # And a key that differs in one entry is another folder.
        assert calibration.cache_folder(root, {**key, "n_frames": key["n_frames"] + 1}) != calibration.cache_folder(
            root, key)


def test_calibration_manifest_is_written_whole_and_a_manifest_held_briefly_does_not_stop_a_build():
    """The manifest is moved into place complete, last, and an old one that cannot be removed is no reason to fail.

    On Windows a file that another process has open cannot be removed. Two checkouts with the same code share a key and
    so a folder; while one still had the manifest open to write it, the other's os.remove raised PermissionError out of
    load(). Now the manifest is written to a temp file of its own in the folder and moved over the old one (os.replace),
    after the arrays, and an old manifest that cannot be removed counts as absent. The refusal is simulated by an
    os.remove that refuses the manifest, which is the same on every platform. The move itself is retried while the OS
    refuses it, which the next two checks look at; a hold that lasts past the retries fails the build, and a rerun
    recovers."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        key = calibration._key(inputs, gt)
        folder = calibration.cache_folder(root, key)
        manifest_path = os.path.join(folder, "manifest.json")

        def read(path):
            with open(path, "rb") as handle:
                return handle.read()

        real_replace, real_remove, real_save = os.replace, os.remove, np.save
        calls = []

        def watching_replace(source, target, *args, **kwargs):
            # What a reader sees before the move: no manifest (the old one is gone), the arrays all there, and the new
            # manifest complete in a file of its own beside the target, so that the move is within one folder.
            have = sorted(os.listdir(folder))
            calls.append({"same_folder": os.path.dirname(os.path.abspath(source)) == os.path.dirname(
                os.path.abspath(target)), "target": os.path.basename(target), "manifest_there": "manifest.json" in have,
                "arrays": sum(name.endswith(".npy") for name in have), "key": json.loads(read(source))["key"]})
            return real_replace(source, target, *args, **kwargs)

        def saves_before_the_manifest(path, *args, **kwargs):
            calls.append("save " + os.path.basename(path))
            return real_save(path, *args, **kwargs)

        def refuses_the_manifest(path, *args, **kwargs):
            if os.path.basename(path) == "manifest.json":
                raise PermissionError(13, "The process cannot access the file: it is being used by another process")
            return real_remove(path, *args, **kwargs)

        def tidy():
            """Nothing but the arrays and the manifest is left in the folder: no temp file."""
            names = sorted(os.listdir(folder))
            assert "manifest.json" in names and not [name for name in names if name.endswith(".tmp")], names
            return names

        def written(*patches):
            """Run _write with the given (object, name, replacement) patched in, whatever happens; the error if any."""
            originals = [(target, name, getattr(target, name)) for target, name, _ in patches]
            try:
                for target, name, replacement in patches:
                    setattr(target, name, replacement)
                calibration._write(cal, folder, key)
            finally:
                for target, name, original in originals:
                    setattr(target, name, original)

        # The first write: arrays, then one move of a complete manifest from its own file into place.
        written((os, "replace", watching_replace), (np, "save", saves_before_the_manifest))
        moves = [call for call in calls if isinstance(call, dict)]
        assert len(moves) == 1, "the manifest must be moved into place once (os.replace): {}".format(calls)
        move = dict(moves[0])
        assert move.pop("arrays") == len(calibration._arrays(cal)) == sum(
            call.startswith("save ") for call in calls if isinstance(call, str)), "arrays first, manifest last"
        assert move == {"same_folder": True, "target": "manifest.json", "manifest_there": False, "key": key}, move
        assert calls.index(moves[0]) == len(calls) - 1, "the manifest is the last thing written"
        names = tidy()
        first = read(manifest_path)
        loaded = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert calibration.digest(loaded) == calibration.digest(cal), "and the cache reads back"

        # Another process holds the old manifest: it cannot be removed. That is "absent", the build goes on, and the
        # new manifest is moved over it, the same bytes (the key is the same).
        written((os, "remove", refuses_the_manifest))
        assert tidy() == names and read(manifest_path) == first

        # A manifest that cannot be moved into place is an error, and leaves neither a temp file nor a manifest that
        # claims the arrays: the folder reads as absent and load() measures again.
        def refuses_the_move(source, target, *args, **kwargs):
            raise PermissionError(13, "the manifest is in use")

        expect_error(OSError, lambda: written((os, "replace", refuses_the_move)), "in use")
        names = sorted(os.listdir(folder))
        assert "manifest.json" not in names and not [name for name in names if name.endswith(".tmp")], names
        assert calibration._read_manifest(folder) is None
        again = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert calibration.digest(again) == calibration.digest(cal)
        assert tidy() == sorted(names + ["manifest.json"]) and read(manifest_path) == first


def test_calibration_manifest_move_is_retried_briefly_and_then_fails():
    """A move that the OS refuses is tried again REPLACE_RETRIES times, REPLACE_WAIT_S apart, and then it is an error.

    On Windows os.replace raises PermissionError while another process has the target open, and a reader of the other
    checkout that shares this folder (same code, same key) has it open for milliseconds: waiting a little is enough.
    A hold that outlasts the retries fails the build, with no temp file left, and a rerun recovers. Simulated by an
    os.replace that refuses its first calls, which is the same on every platform; the next check holds a real handle.
    Only PermissionError is waited for: any other OSError is the caller's at once."""
    assert (calibration.REPLACE_RETRIES, calibration.REPLACE_WAIT_S) == (5, 0.05)
    retries = calibration.REPLACE_RETRIES
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        key = calibration._key(inputs, gt)
        folder = calibration.cache_folder(root, key)
        manifest_path = os.path.join(folder, "manifest.json")
        real_replace, real_sleep = os.replace, calibration.time.sleep

        def written(refusals, refusal=lambda: PermissionError(13, "the manifest is in use")):
            """_write with os.replace refusing its first `refusals` calls: (the calls made, the waits made, the error)."""
            calls, waits = [], []

            def refusing(source, target, *args, **kwargs):
                calls.append(os.path.basename(target))
                if len(calls) <= refusals:
                    raise refusal()
                return real_replace(source, target, *args, **kwargs)

            os.replace, calibration.time.sleep = refusing, waits.append
            try:
                calibration._write(cal, folder, key)
                failure = None
            except OSError as caught:
                failure = caught
            finally:
                os.replace, calibration.time.sleep = real_replace, real_sleep
            return calls, waits, failure

        def tidy():
            names = sorted(os.listdir(folder))
            assert not [name for name in names if name.endswith(".tmp")], names
            return names

        for refusals in (0, 1, 3, retries):    # the last one succeeds on the last retry
            calls, waits, failure = written(refusals)
            assert failure is None and len(calls) == refusals + 1 and waits == [calibration.REPLACE_WAIT_S] * refusals, (
                refusals, len(calls), waits, failure)
            assert "manifest.json" in tidy() and calibration._read_manifest(folder) is not None
        reference = calibration.digest(calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt))
        assert reference == calibration.digest(cal)

        # A hold that outlasts the retries: the error is the OS's own, after exactly retries + 1 tries and retries waits.
        calls, waits, failure = written(retries + 1)
        assert isinstance(failure, PermissionError) and "in use" in str(failure), failure
        assert len(calls) == retries + 1 and waits == [calibration.REPLACE_WAIT_S] * retries, (len(calls), waits)
        names = tidy()
        assert "manifest.json" not in names, "the old manifest was removed up front, and the new one never arrived"
        assert calibration._read_manifest(folder) is None, "the folder reads as absent"
        # The rerun recovers: it measures again and the cache is whole.
        again = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert calibration.digest(again) == reference and "manifest.json" in tidy()
        assert os.path.isfile(manifest_path)

        # Another kind of OSError is not a hold: one try, no waiting (errno 5 is EIO; OSError(13, ...) would be a
        # PermissionError, which Python picks by the errno).
        calls, waits, failure = written(1, refusal=lambda: OSError(5, "the disk failed"))
        assert type(failure) is OSError and len(calls) == 1 and waits == [], (calls, waits, failure)
        tidy()


def test_calibration_manifest_move_waits_for_a_real_handle_on_windows():
    """A real handle on the old manifest, released while the move is being retried: the build goes through (Windows).

    Windows refuses os.replace over a file that another handle has open (Python's open() does not share delete
    access), which is what a reader of a manifest does for a moment. Here the handle is real and the refusal is the
    OS's own: the first os.replace raises PermissionError, the handle is released on the second try (that is, during
    the retries), and the move completes. A handle held through every retry fails the build with the OS's error, no temp
    file is left, and once it is released a rerun recovers. Not run on other systems, which let a file be replaced
    whoever has it open: the retry is then checked by test_calibration_manifest_move_is_retried_briefly_and_then_fails
    alone."""
    if sys.platform != "win32":
        raise NotRun("only Windows refuses os.replace over a file that another handle has open ({}); the retries are "
                     "checked with a simulated refusal by "
                     "test_calibration_manifest_move_is_retried_briefly_and_then_fails".format(sys.platform))
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        cal = calibration.measure(inputs, gt)
        key = calibration._key(inputs, gt)
        folder = calibration.cache_folder(root, key)
        manifest_path = os.path.join(folder, "manifest.json")
        calibration._write(cal, folder, key)
        with open(manifest_path, "rb") as handle:
            first = handle.read()
        real_replace = os.replace

        def written(release_on):
            """_write while a reader holds the manifest; it lets go at the call of os.replace numbered `release_on`."""
            reader = open(manifest_path, "rb")
            outcomes = []

            def watching(source, target, *args, **kwargs):
                if len(outcomes) + 1 == release_on:
                    reader.close()
                try:
                    result = real_replace(source, target, *args, **kwargs)
                except PermissionError as refused:
                    outcomes.append(refused)
                    raise
                outcomes.append(None)
                return result

            os.replace = watching
            try:
                calibration._write(cal, folder, key)
                failure = None
            except PermissionError as caught:
                failure = caught
            finally:
                os.replace = real_replace
                reader.close()
            return outcomes, failure

        outcomes, failure = written(release_on=2)
        assert failure is None and len(outcomes) == 2, (len(outcomes), failure)
        assert isinstance(outcomes[0], PermissionError) and outcomes[1] is None, "the OS refused once, then let it in"
        with open(manifest_path, "rb") as handle:
            assert handle.read() == first, "the same key, the same bytes"
        assert not [name for name in os.listdir(folder) if name.endswith(".tmp")]

        # Released from inside the retry loop's own wait: the real handle makes the first os.replace raise the OS's
        # PermissionError, the first sleep (patched here to close the handle, and to return at once) lets go of it, and
        # the second os.replace goes through. No clock decides which comes first, so this cannot flake on a slow machine.
        reader = open(manifest_path, "rb")
        real_sleep, sleeps, refusals = calibration.time.sleep, [], []

        def letting_go(seconds):
            sleeps.append(seconds)
            reader.close()

        def refused_once(source, target, *args, **kwargs):
            try:
                return real_replace(source, target, *args, **kwargs)
            except PermissionError as refused:
                refusals.append(refused)
                raise

        os.replace, calibration.time.sleep = refused_once, letting_go
        try:
            calibration._write(cal, folder, key)
        finally:
            os.replace, calibration.time.sleep = real_replace, real_sleep
            reader.close()
        assert len(refusals) == 1 and sleeps == [calibration.REPLACE_WAIT_S], (len(refusals), sleeps)
        with open(manifest_path, "rb") as handle:
            assert handle.read() == first, "the same key, the same bytes"

        # Held through every retry: an error, no temp file, and a rerun recovers once the handle is released.
        outcomes, failure = written(release_on=calibration.REPLACE_RETRIES + 2)
        assert isinstance(failure, PermissionError), failure
        assert len(outcomes) == calibration.REPLACE_RETRIES + 1 and all(isinstance(o, PermissionError) for o in outcomes)
        assert not [name for name in os.listdir(folder) if name.endswith(".tmp")]
        again = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert calibration.digest(again) == calibration.digest(cal)
        with open(manifest_path, "rb") as handle:
            assert handle.read() == first


def import_copy(path, name):
    """The file at `path` imported as a module called `name`, which runs its code and takes its hash right now."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


def test_calibration_code_hash_is_the_files_and_ignores_line_endings():
    """CALIBRATION_SHA256 is the hash of calibration.py with CRLF read as LF: CRLF is the same definitions."""
    with open(calibration.__file__, "rb") as handle:
        raw = handle.read()
    assert calibration.CALIBRATION_SHA256 == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()

    names = ("calibration_copy_lf", "calibration_copy_crlf", "calibration_copy_edited")
    try:
        with tempfile.TemporaryDirectory() as folder:
            text = raw.replace(b"\r\n", b"\n").decode("utf-8")
            copies = {}
            for name, ending, extra in ((names[0], "\n", ""), (names[1], "\r\n", ""), (names[2], "\n", "# an edit\n")):
                path = os.path.join(folder, name + ".py")
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write((text + extra).replace("\n", ending))
                copies[name] = import_copy(path, name)
            with open(copies[names[1]].__file__, "rb") as handle:
                crlf = handle.read()
            assert crlf.count(b"\r\n") > 100 and crlf.count(b"\n") == crlf.count(b"\r\n"), "the copy is not all CRLF"
            assert copies[names[0]].CALIBRATION_SHA256 == copies[names[1]].CALIBRATION_SHA256 == \
                calibration.CALIBRATION_SHA256, "line endings changed the hash"
            assert copies[names[2]].CALIBRATION_SHA256 != calibration.CALIBRATION_SHA256, "an edit did not change it"
    finally:
        for name in names:
            sys.modules.pop(name, None)


def calibration_copy(folder, name, hazard_module):
    """An unedited copy of calibration.py, imported while `hazard_module` is pipeline.benchmark.hazard, so that its
    HAZARD_SHA256 is the hash of that module's file. Its configs are the real ones (the copy has no folder of its
    own)."""
    import pipeline.benchmark as package
    path = os.path.join(folder, name + ".py")
    shutil.copyfile(calibration.__file__, path)
    real_module, real_attribute = sys.modules["pipeline.benchmark.hazard"], package.hazard
    sys.modules["pipeline.benchmark.hazard"] = package.hazard = hazard_module   # what `from ... import hazard` finds
    try:
        copy = import_copy(path, name)
    finally:
        sys.modules["pipeline.benchmark.hazard"], package.hazard = real_module, real_attribute
    copy.CONFIG_FOLDER = calibration.CONFIG_FOLDER
    return copy


def test_calibration_key_holds_the_hazard_code_and_ignores_its_line_endings():
    """HAZARD_SHA256 is the hash of hazard.py with CRLF read as LF, and another hazard.py is another key and folder.

    hazard.py makes the version-2 fields (the features, the fit, the AUC, the thirds), so a checkout whose hazard.py
    differs must not read this one's fit: changing hazard.NEAR_PLAYER_M used to leave the key and the folder as they
    were, and load() gave back the first fit while measure() gave a different one."""
    with open(hazard.__file__, "rb") as handle:
        raw = handle.read()
    assert calibration.HAZARD_SHA256 == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert calibration.HAZARD_SHA256 != calibration.CALIBRATION_SHA256

    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    assert text.count("NEAR_PLAYER_M = 3.0 ") == 1, "the constant to edit is not where this check looks for it"
    names = ("hazard_copy_lf", "hazard_copy_crlf", "hazard_copy_edited", "calibration_copy_lf", "calibration_copy_crlf",
             "calibration_copy_edited")
    try:
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as root:
            inputs, gt = synthetic_calibration_inputs(root)
            hazards = {}
            edited_text = text.replace("NEAR_PLAYER_M = 3.0 ", "NEAR_PLAYER_M = 2.5 ")
            for name, ending, content in ((names[0], "\n", text), (names[1], "\r\n", text),
                                          (names[2], "\n", edited_text)):
                path = os.path.join(folder, name + ".py")
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    handle.write(content.replace("\n", ending))
                hazards[name] = import_copy(path, name)
            copies = {name: calibration_copy(folder, name, hazards[hazard_name])
                      for name, hazard_name in zip(names[3:], names[:3])}
            lf, crlf, edited = (copies[name] for name in names[3:])
            assert hazards[names[2]].NEAR_PLAYER_M == 2.5 and hazards[names[0]].NEAR_PLAYER_M == 3.0

            # Line endings are not a change of the file; an edit is. calibration.py is the same text in all three.
            assert lf.HAZARD_SHA256 == crlf.HAZARD_SHA256 == calibration.HAZARD_SHA256, "line endings changed the hash"
            assert edited.HAZARD_SHA256 != calibration.HAZARD_SHA256, "an edit to hazard.py did not change it"
            assert lf.CALIBRATION_SHA256 == edited.CALIBRATION_SHA256 == calibration.CALIBRATION_SHA256

            # So the key is the same for the same code, and for the edited one it differs in that hash alone: another
            # folder, which is a rebuild.
            key = calibration._key(inputs, gt)
            assert lf._key(inputs, gt) == crlf._key(inputs, gt) == key
            other = edited._key(inputs, gt)
            assert other != key and {k: v for k, v in other.items() if k != "hazard_sha256"} == {
                k: v for k, v in key.items() if k != "hazard_sha256"}, "only the hazard hash differs"
            assert other["hazard_sha256"] == edited.HAZARD_SHA256
            assert calibration.cache_folder(root, other) != calibration.cache_folder(root, key)
    finally:
        for name in names:
            sys.modules.pop(name, None)


# -- our match --------------------------------------------------------------------------------------------------------

def test_data_calibration_replicates_a01():
    """The calibration of BAR-ATH reproduces what A01, A02 and R001 published, by the same definitions.

    The published figures are those of the graded blocks, so they are measured here by calling ball_stats and
    player_stats on the graded blocks and asserted; the Calibration stores the spans from each kick-off, whose values
    are printed beside them. Prints aggregates only: counts, shares and quantiles of the match, never a frame, a
    position or a label.
    """
    from pipeline import ROOT
    from pipeline.benchmark.run import predict
    from pipeline.config import load_config
    from pipeline.inputs import Inputs
    inputs = Inputs()
    atd = inputs.atd
    gt = calibration.public_gt()
    cal = calibration.load()
    # Measured twice, the same: arrays, scalars and all (the cache is not what makes it so).
    assert calibration.digest(calibration.measure(inputs, gt)) == calibration.digest(cal), "not deterministic"
    assert_no_frame_data(calibration.describe(cal))

    # What is stored is on the spans from the kick-offs: their lengths, no minute before a kick-off, parallel gaps.
    for half, start, stop in inputs.graded:
        assert len(cal.ball_masks[half]) == len(cal.present_counts[half]) == stop - inputs.kickoffs[half], half
    assert cal.gap_onset_minute.min() >= 0.0
    assert cal.interior_durations_s.shape == cal.gap_onset_minute.shape == cal.gap_onset_near_line.shape
    stored = cal.scalars
    assert stored["dropped_frames"] == stored["dropped_slot_frames"] == 0, "x and y are missing together in the ATD"

    # The published figures: the same two functions, on the graded blocks.
    ball = calibration.ball_stats(atd.ball_present, inputs.graded, inputs.metres(atd.ball_xy), inputs.pitch, inputs.fps)
    players = calibration.player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy),
                                       atd.players_speed, inputs.graded, inputs.fps)

    def near(name, value, expected, tolerance, spanned):
        assert abs(value - expected) <= tolerance, "{}: {} is not {} +/- {}".format(name, value, expected, tolerance)
        print("      {:<30} graded {:>9.4f}  (expected {} +/- {})   from kick-off {:>9.4f}".format(
            name, value, expected, tolerance, spanned))

    def shown(name, value, spanned):
        """A figure that is printed and not asserted (the two that A07 says do not reproduce)."""
        print("      {:<30} graded {:>9.4f}  (not asserted)   from kick-off {:>9.4f}".format(name, value, spanned))

    near("missing share", ball["missing_share"], *A01["missing_share"], stored["missing_share"])
    assert (ball["gaps"], ball["interior_gaps"], ball["near_line_onsets"]) == (
        A01["gaps"], A01["interior_gaps"], A01["near_line_onsets"]), (
        ball["gaps"], ball["interior_gaps"], ball["near_line_onsets"])
    print("      gaps / interior / near-line    graded {} / {} / {}   from kick-off {} / {} / {}".format(
        ball["gaps"], ball["interior_gaps"], ball["near_line_onsets"],
        stored["gaps"], stored["interior_gaps"], stored["near_line_onsets"]))
    near("near-line share of interior", ball["near_line_share"], 42 / 147, 5e-5, stored["near_line_share"])
    near("median gap (s)", float(np.median(ball["gap_durations_s"])), *A01["median_gap_s"],
         float(np.median(cal.gap_durations_s)))
    for half, expected in A02_PERCENT.items():
        for table in (ball["missing_share_5min"], cal.missing_share_5min):   # the stored table is the graded one
            percent = tuple(int(round(100 * share)) for share in table[half])
            assert percent == expected, "A02, half {}: {} instead of {}".format(half, percent, expected)
    print("      A02 (%), half 1 {}".format(A02_PERCENT[1]))
    print("                half 2 {}".format(A02_PERCENT[2]))
    assert players["tracklets"] == A01["tracklets"] == len(players["tracklet_lengths_s"]), players["tracklets"]
    print("      tracklets                      graded {}   from kick-off {}".format(
        players["tracklets"], stored["tracklets"]))
    near("wrong-team share", players["wrong_team_share"], *A01["wrong_team_share"], cal.wrong_team_share)
    near("speed-NaN share", players["speed_nan_share"], *A01["speed_nan_share"], cal.speed_nan_share)
    episodes = (players["wrong_team_episodes_home"], players["wrong_team_episodes_away"])
    assert episodes == PLAN["wrong_team_episodes"], episodes
    print("      wrong-team episodes (h / a)    graded {} / {}   from kick-off {} / {}".format(
        *episodes, stored["wrong_team_episodes_home"], stored["wrong_team_episodes_away"]))
    near("present mean", players["present_mean"], *PLAN["present_mean"], stored["present_mean"])
    near("share with 22 present", players["present_share_22"], *PLAN["present_share_22"],
         stored["present_share_22"])
    shown("share with <= 14 present", players["present_share_le14"], stored["present_share_le14"])
    near("tracklet median (s)", float(np.median(players["tracklet_lengths_s"])), *PLAN["tracklet_median_s"],
         float(np.median(cal.tracklet_lengths_s)))
    near("jitter RMS (mm)", 1000 * players["jitter_rms_m"], *PLAN["jitter_rms_mm"], 1000 * cal.player_jitter_rms_m)
    graded_speed = float(np.isnan(atd.ball_speed[inputs.graded_mask & atd.ball_present]).mean())
    near("ball-speed NaN share (T31)", graded_speed, *PLAN["ball_speed_nan_share"], stored["ball_speed_nan_share"])
    shown("p99 of NaN-speed runs", float(np.quantile(players["speed_nan_runs"], 0.99)),
          float(np.quantile(cal.speed_nan_runs, 0.99)))

    # The public window and the yardsticks, over the graded halves.
    near("nearest-player agreement", cal.public_window["nearest_agreement"], *A01["nearest_agreement"],
         cal.public_window["nearest_agreement"])
    assert cal.public_window["judged_frames"] == 5640, cal.public_window
    print("      judged frames                  {}".format(cal.public_window["judged_frames"]))
    print("      dead-ball visible share        {:.4f}   of {} dead frames".format(
        cal.public_window["dead_ball_visible_share"], cal.public_window["dead_frames"]))
    graded = inputs.graded_mask
    for width, expected in BASELINE_POSSESSIONS.items():
        got = cal.baseline_outputs[width]
        assert abs(got["possessions"] - expected) <= POSSESSION_TOLERANCE, (
            "nearest_w{}: {} possessions, A07 expects {} +/- {}".format(width, got["possessions"], expected,
                                                                      POSSESSION_TOLERANCE))
        # Stronger than the count: the per-half yardstick's codes ARE the published baseline's on every graded frame.
        codes, _ = predict(load_config(os.path.join(calibration.CONFIG_FOLDER, "nearest_w{}.toml".format(width))),
                           inputs)
        published = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "results",
                                                                  BASELINE_FILES[width] + ".csv")))
        differ = int((codes[graded] != published[graded]).sum())
        assert differ == 0, "nearest_w{} differs from {} on {} graded frames".format(width, BASELINE_FILES[width],
                                                                                   differ)
        print("      nearest_w{:<4} possessions {:>5}  ({} expected)  per graded minute {:.2f}  D share {:.4f}  "
              "= {} on all {} graded frames".format(width, got["possessions"], expected, got["poss_per_min"],
                                                    got["d_share"], BASELINE_FILES[width], int(graded.sum())))


def test_data_hazard_fit():
    """A08: where our tracker loses the ball, fitted on BAR-ATH: the coefficients, the four cross-half AUCs, c, the
    duration cells and the go/no-go decision. Prints aggregates only (counts, coefficients, AUCs, medians), and asserts
    that they are finite, reproducible and consistent with A07's gaps; it does not assert the AUCs' size, since a no-go
    is a legitimate result and nothing here is tuned to a decision."""
    from pipeline.inputs import Inputs
    inputs = Inputs()
    atd = inputs.atd
    cal = calibration.load()
    spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
    ball = calibration.ball_stats(atd.ball_present, spans, inputs.metres(atd.ball_xy), inputs.pitch, inputs.fps)

    def fit():
        return calibration.fit_hazard(atd.ball_present, inputs.metres(atd.ball_xy), atd.players_present,
                                      inputs.metres(atd.players_xy), spans, inputs.pitch, inputs.fps,
                                      ball["interior_durations_s"], ball["gap_onset_minute"],
                                      ball["gap_onset_near_line"])

    result, second = fit(), fit()
    # Reproducible: fitted twice from the arrays, and the cache holds the same fit, to the byte.
    for other in (second, {"model": cal.hazard_model, "l2": cal.hazard_l2, "c": cal.hazard_c, "auc": cal.hazard_auc,
                           "go": cal.hazard_go, "fallback_cells": cal.hazard_fallback_cells}):
        for name in ("coef", "mean", "scale"):
            assert getattr(result["model"], name).tobytes() == getattr(other["model"], name).tobytes(), name
        assert result["model"].intercept == other["model"].intercept
        assert (result["l2"], result["c"], result["auc"], result["go"], result["fallback_cells"]) == (
            other["l2"], other["c"], other["auc"], other["go"], other["fallback_cells"])
    for cls in (0, 1):
        for third in (0, 1, 2):
            assert result["durations"][cls][third].tobytes() == cal.hazard_durations_s[cls][third].tobytes()

    model, auc, counts = result["model"], result["auc"], result["counts"]
    for name in ("coef", "mean", "scale"):
        assert np.isfinite(getattr(model, name)).all(), name
    assert np.isfinite([model.intercept, result["c"], result["l2"], *auc.values()]).all()
    assert (model.scale > 0).all() and 0.0 < min(auc.values()) and max(auc.values()) < 1.0, auc
    assert result["l2"] in (1.0, 10.0, 100.0, 1000.0), result["l2"]
    assert result["go"] == (min(auc["full_1to2"], auc["full_2to1"]) >= calibration.HAZARD_GO_AUC)
    # c is 1 on the rows it was fitted on (the intercept is not penalised), to the precision of the fit.
    assert 0.0 < result["c"] and abs(result["c"] - 1.0) < 1e-6, result["c"]

    # The spans' counts add up and agree with A07's gaps: every interior gap is an event or lost to an undefined feature.
    assert counts["visible"] == counts["edge"] + counts["last_frame"] + counts["nan_rows"] + counts["rows"], counts
    assert counts["events"] + counts["nan_events"] == ball["interior_gaps"] == cal.scalars["interior_gaps"] == 143
    assert counts["visible"] == cal.scalars["frames"] - cal.scalars["missing_frames"] == 91844, counts
    assert counts["rows_1"] + counts["rows_2"] == counts["rows"] and counts["events_1"] + counts["events_2"] == counts[
        "events"]
    assert counts["events_1"] > 0 and counts["events_2"] > 0, "each half needs onsets to be scored"
    own = {(cls, third): counts["cell_{}_{}".format(cls, third)] for cls in (0, 1) for third in (0, 1, 2)}
    assert sum(own.values()) == 143 and sum(n for (cls, _), n in own.items() if cls == 1) == cal.scalars[
        "near_line_onsets"] == 40, own
    for (cls, third), n in own.items():
        used = result["durations"][cls][third]
        assert used.size > 0 and ((cls, third) in result["fallback_cells"]) == (n < calibration.MIN_CELL_DURATIONS)
        assert used.size == (n if n >= calibration.MIN_CELL_DURATIONS else sum(
            m for (c, _), m in own.items() if c == cls))

    print("      spans from the kick-offs: {} frames, {} with the ball".format(cal.scalars["frames"], counts["visible"]))
    print("      at-risk rows {} ({} in half 1, {} in half 2), onsets {} ({} / {})".format(
        counts["rows"], counts["rows_1"], counts["rows_2"], counts["events"], counts["events_1"], counts["events_2"]))
    print("      left out: {} before an edge gap, {} last frames, {} with an undefined feature ({} of them onsets)"
          .format(counts["edge"], counts["last_frame"], counts["nan_rows"], counts["nan_events"]))
    print("      l2 used: pooled {:g}; cross-half {}".format(result["l2"], ", ".join(
        "{} {:g}".format(key, value) for key, value in sorted(result["cross_l2"].items()))))
    print("      intercept {:+.9f}".format(model.intercept))
    for index, name in enumerate(hazard.FEATURES):
        print("      {:<18} coef {:+.9f} (standardised)   mean {:>12.6f}   scale {:>12.6f}".format(
            name, model.coef[index], model.mean[index], model.scale[index]))
    print("      cross-half AUC   full  1to2 {:.4f}  2to1 {:.4f}    time-only  1to2 {:.4f}  2to1 {:.4f}".format(
        auc["full_1to2"], auc["full_2to1"], auc["time_1to2"], auc["time_2to1"]))
    print("      c = {:.9f}   go = {}  (both full AUCs >= {:.2f})".format(result["c"], result["go"],
                                                                         calibration.HAZARD_GO_AUC))
    for cls in (0, 1):
        for third in (0, 1, 2):
            used = result["durations"][cls][third]
            print("      class {} third {}: {:>3} durations, {:>3} used, median {:>6.2f} s{}".format(
                cls, third, own[(cls, third)], used.size, float(np.median(used)),
                "   (pooled over the thirds)" if (cls, third) in result["fallback_cells"] else ""))
