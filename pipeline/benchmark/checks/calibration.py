"""Checks of the calibration (pipeline/benchmark/calibration.py): our own tracking, measured for the benchmark.

The statistics are checked on objects built here by hand, small enough that every figure in a check is worked out from
the objects' construction and written down with its working, never read off what the code printed. The two fixtures
(ball_fixture, player_fixture) are built so that a mistake of the kind that is easy to make changes a figure: a gap
that runs across the edge of a block, two blocks that touch (a naive run-length count would fuse the gaps either side
of the join), a ball that sits exactly 2.0 m from a line (the near-line test is strict), a slot that is present before
and after a block (a tracklet must not run over its edge) and a second difference that would be huge if a triple
straddled the edge. test_data_calibration_replicates_a01 then runs the same code on our match, with the figures A01,
A02 and R001 published; it prints aggregates (counts, shares, quantiles) and never a frame or a row of the match.
"""

import dataclasses
import json
import math
import os
import shutil
import tempfile

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import calibration
from pipeline.selftest import expect_error, make_synthetic_bundle, quiet, synthetic_inputs

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
PITCH = (105.3, 68.0)
MAX_SEQUENCE = 20     # describe() may hold no sequence longer than this: nothing that could rebuild a frame
# The figures the plan and the brief give for our match (A01, A02, R001), asserted by
# test_data_calibration_replicates_a01.
A01 = {"missing_share": (0.4069, 0.0001), "gaps": 150, "interior_gaps": 147, "near_line_onsets": 42,
       "median_gap_s": (10.7, 0.05), "tracklets": 1219, "wrong_team_share": (0.108, 0.001),
       "speed_nan_share": (0.0796, 0.0005), "nearest_agreement": (0.778, 0.001)}
# A02's table, in whole percent per 5 minutes counted from each half's first graded frame (the brief asks for blocks
# 1-3 of the first half; the whole table reproduces).
A02_PERCENT = {1: (12, 17, 24, 28, 44, 35, 63, 43, 71, 46, 80), 2: (12, 33, 26, 55, 26, 39, 80, 51, 52, 40, 28)}
BASELINE_POSSESSIONS = {0: 1949, 25: 746, 50: 594, 125: 398}
POSSESSION_TOLERANCE = 5
# Figures of the plan (measured 2026-10-22 by a read-only script) that the calibration reproduces; the two that it does
# not (5.26% of frames with 14 or fewer present against the plan's 6.6%, and p99 of the NaN-speed runs, 2,305 frames
# against 1,209, which the plan measured on the whole file) are in A07 and not asserted here.
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


def test_calibration_ball_stats_exact():
    present, xy, blocks, pitch, fps = ball_fixture()
    got = calibration.ball_stats(present, blocks, xy, pitch, fps)

    # 20 + 5 + 20 + 100 + 10 + 5 + 5 = 165 missing frames of the 1,000 that are graded.
    assert got["graded_frames"] == 1000 and got["missing_frames"] == 165, got
    assert got["missing_share"] == 0.165, got["missing_share"]
    assert got["gaps"] == 7 and got["interior_gaps"] == 5, (got["gaps"], got["interior_gaps"])
    assert got["gap_durations_s"].tolist() == [20.0, 5.0, 20.0, 100.0, 10.0, 5.0, 5.0], got["gap_durations_s"]
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

    # The frame rate turns frames into seconds and minutes, and sets the segment: at 2 frames a second the same
    # gaps last half as long in seconds, and a 5-minute segment is 600 frames, so block 1 is a segment and 100 frames.
    twice = calibration.ball_stats(present, blocks, xy, pitch, 2)
    assert twice["gap_durations_s"].tolist() == [10.0, 2.5, 10.0, 50.0, 5.0, 2.5, 2.5], twice["gap_durations_s"]
    assert np.allclose(twice["gap_onset_minute"], np.array([99, 149, 329, 199, 249]) / 120.0, rtol=0, atol=1e-12)
    # Segment 1 of block 1 now holds g1, g2 and g3 (45 of 600 frames) and segment 2 holds g4 (100 of 100); block 2 is
    # still one segment of 300 frames, with 20 missing.
    assert np.allclose(twice["missing_share_5min"][1], [45 / 600, 1.0], rtol=0, atol=1e-15), twice["missing_share_5min"]
    assert np.allclose(twice["missing_share_5min"][2], [20 / 300], rtol=0, atol=1e-15), twice["missing_share_5min"]

    # No gap at all, and a block that is nothing but a gap.
    clear = calibration.ball_stats(np.ones(50, dtype=bool), [(1, 0, 50)], np.full((50, 2), 10.0), pitch, 25)
    assert clear["gaps"] == 0 and clear["interior_gaps"] == 0 and clear["missing_share"] == 0.0, clear
    assert clear["gap_durations_s"].size == 0 and clear["gap_onset_minute"].size == 0
    blind = calibration.ball_stats(np.zeros(50, dtype=bool), [(1, 5, 45)], np.full((50, 2), np.nan), pitch, 25)
    assert blind["gaps"] == 1 and blind["interior_gaps"] == 0 and blind["missing_share"] == 1.0, blind
    assert blind["gap_durations_s"].tolist() == [40 / 25], blind["gap_durations_s"]


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


def test_calibration_player_stats_exact():
    present, slot_team, xy, speed, blocks, fps = player_fixture()
    got = calibration.player_stats(present, slot_team, xy, speed, blocks, fps)

    # Present per frame is 21 (20 frames) and 20 (20 frames) in A, and 20 (5), 21 (13), 22 (12) in B: 25 frames of 20,
    # 33 of 21 and 12 of 22, 70 graded frames and 20*25 + 21*33 + 22*12 = 1,457 present slot-frames.
    expected_hist = np.zeros(25, dtype=np.int64)
    expected_hist[[20, 21, 22]] = (25, 33, 12)
    assert got["present_hist"].dtype == np.int64 and got["present_hist"].tolist() == expected_hist.tolist(), (
        got["present_hist"])
    assert got["present_slot_frames"] == 1457 and got["slots"] == 24, (got["present_slot_frames"], got["slots"])
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
    test_calibration_ball_stats_exact()
    test_calibration_player_stats_exact()


def test_calibration_stats_refuse_what_they_cannot_measure():
    present, xy, blocks, pitch, fps = ball_fixture()
    ball = calibration.ball_stats
    expect_error(ValueError, lambda: ball(present[:-1], blocks, xy, pitch, fps), "ball_present", "ball_xy_m", "1099")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 710), (2, 700, 1010)], xy, pitch, fps), "blocks", "overlap")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 710), (1, 710, 1010)], xy, pitch, fps), "half", "twice")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 1101)], xy, pitch, fps), "blocks", "1101", "1100")
    expect_error(ValueError, lambda: ball(present, [(1, 10, 10)], xy, pitch, fps), "blocks", "empty")
    expect_error(ValueError, lambda: ball(present, blocks, xy, (0.0, 68.0), fps), "pitch")
    expect_error(ValueError, lambda: ball(present, blocks, xy, pitch, 0), "fps")
    # A ball that is "present" with no position, or the reverse, would put a NaN into a line distance: refused by name.
    lying = present.copy()
    lying[300] = False
    expect_error(ValueError, lambda: ball(lying, blocks, xy, pitch, fps), "ball_present", "ball_xy_m", "disagree")
    expect_error(ValueError, lambda: ball(present.astype(int), blocks, xy, pitch, fps), "ball_present", "bool")

    have, slot_team, pxy, speed, pblocks, pfps = player_fixture()
    players = calibration.player_stats
    expect_error(ValueError, lambda: players(have, slot_team[:-1], pxy, speed, pblocks, pfps), "slot_team", "23")
    expect_error(ValueError, lambda: players(have, slot_team + 1, pxy, speed, pblocks, pfps), "slot_team",
                 "0 (home) or 1 (away)")
    expect_error(ValueError, lambda: players(have, slot_team, pxy[:, :, :1], speed, pblocks, pfps), "players_xy_m")
    expect_error(ValueError, lambda: players(have, slot_team, pxy, speed[:-1], pblocks, pfps), "players_speed")
    expect_error(ValueError, lambda: players(have, slot_team, pxy, speed, [(1, 5, 101)], pfps), "blocks", "101")
    positioned = np.where(have[:, :, None], pxy, np.nan)
    positioned[10, 3, 0] = np.nan   # present with no position: the second difference would be NaN
    expect_error(ValueError, lambda: players(have, slot_team, positioned, speed, pblocks, pfps),
                 "players_present", "players_xy_m", "disagree")


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

        # The ball's mask and the players' count run from the kick-off to the end of the graded block, per half: True
        # where the ball is missing, int16 counts, 220 frames (60-279) and 220 (340-559) here.
        assert sorted(cal.ball_masks) == [1, 2] and sorted(cal.present_counts) == [1, 2]
        for half, kickoff, stop in ((1, 60, 280), (2, 340, 560)):
            mask, per_frame = cal.ball_masks[half], cal.present_counts[half]
            assert mask.dtype == bool and per_frame.dtype == np.int16, (mask.dtype, per_frame.dtype)
            assert np.array_equal(mask, ~atd.ball_present[kickoff:stop]), half
            assert np.array_equal(per_frame, counts[kickoff:stop]), half
            assert not mask.flags.writeable and not per_frame.flags.writeable

        # The statistics are the graded blocks' (frames 50-279 and 320-559), 470 frames, not the kick-off spans'.
        graded = inputs.graded_mask
        assert cal.scalars["graded_frames"] == 470 == int(graded.sum())
        assert cal.scalars["missing_frames"] == int((~atd.ball_present[graded]).sum())
        assert cal.present_hist.shape == (47,) and cal.present_hist.tolist() == np.bincount(
            counts[graded], minlength=47).tolist()
        assert abs(cal.scalars["present_mean"] - float(counts[graded].mean())) < 1e-9
        direct = calibration.ball_stats(atd.ball_present, inputs.graded, inputs.metres(atd.ball_xy), inputs.pitch,
                                        inputs.fps)
        assert cal.gap_durations_s.tolist() == direct["gap_durations_s"].tolist()
        assert cal.gap_onset_near_line.tolist() == direct["gap_onset_near_line"].tolist()
        assert cal.gap_onset_minute.tolist() == direct["gap_onset_minute"].tolist()
        assert cal.missing_share_5min[1].tolist() == direct["missing_share_5min"][1].tolist()

        # The ball's speed channel is NaN on this share of the graded frames that have a ball (T31).
        seen = graded & atd.ball_present
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
        # written in the temporary folder and read nowhere else.
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


def test_calibration_cache_inside_data_dir():
    """load() writes only under <data dir>/cache/benchmark/calibration/, and what it keeps there is reused."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        inputs.atd  # the ATD cache is the pipeline's, not this module's: build it first, so that it is not new below
        before = snapshot(root)
        cal = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        written = snapshot(root)
        folder = "cache/benchmark/calibration/"
        new = sorted(set(written) - set(before))
        assert new and all(path.startswith(folder) for path in new), new
        assert not [path for path in new if path.endswith((".csv", ".tmp"))], "no CSV, no leftover: {}".format(new)
        assert {path: stat for path, stat in written.items() if path in before} == before, "load() touched a file"
        assert folder + "manifest.json" in new

        # Reused when fresh: nothing is rewritten, and what comes back is what was built.
        loaded = calibration.load(data_dir=root, inputs=inputs, public_gt_codes=gt)
        assert snapshot(root) == written, "a fresh cache must be read, not rewritten"
        assert calibration.digest(loaded) == calibration.digest(cal) == calibration.digest(
            calibration.measure(inputs, gt)), "the cache must give back exactly what measure() gives"
        assert calibration.describe(loaded) == calibration.describe(cal)
        for half in (1, 2):
            assert not loaded.ball_masks[half].flags.writeable and loaded.ball_masks[half].dtype == bool
            assert loaded.present_counts[half].dtype == np.int16
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
        for field in dataclasses.fields(calibration.Calibration):
            value = getattr(cal, field.name)
            holds_array = isinstance(value, np.ndarray) or (
                isinstance(value, dict) and any(isinstance(item, np.ndarray) for item in value.values()))
            assert field.repr is (not holds_array), "{}: repr must be {}".format(field.name, not holds_array)
        expect_error(dataclasses.FrozenInstanceError, lambda: setattr(cal, "wrong_team_share", 0.0), "wrong_team_share")


def test_calibration_cache_is_keyed_and_unreadable_means_absent():
    """A changed config, version, truth or source rebuilds; a damaged cache is rebuilt, never an error."""
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        folder = os.path.join(root, "cache", "benchmark", "calibration")
        manifest_path = os.path.join(folder, "manifest.json")

        def manifest():
            with open(manifest_path, encoding="utf-8") as handle:
                return json.load(handle)

        def load(codes=gt):
            return calibration.load(data_dir=root, inputs=inputs, public_gt_codes=codes)

        cal = load()
        key = manifest()["key"]
        assert key["calibration_version"] == calibration.CALIBRATION_VERSION
        assert key["atd_sources"] == inputs.atd_manifest["sources"], "the ATD cache's own record of its sources"
        assert key["numpy"] == np.__version__ and sorted(key["configs"]) == ["0", "125", "25", "50"], key

        # An edited config must not reuse the baseline outputs of the old one.
        real_folder = calibration.CONFIG_FOLDER
        edited = os.path.join(root, "configs")
        shutil.copytree(real_folder, edited)
        with open(os.path.join(edited, "nearest_w25.toml"), "a", encoding="utf-8", newline="\n") as handle:
            handle.write("# edited\n")
        calibration.CONFIG_FOLDER = edited
        try:
            load()
            changed = manifest()["key"]
            assert changed["configs"]["25"] != key["configs"]["25"]
            assert changed["configs"]["0"] == key["configs"]["0"] and changed["configs"]["50"] == key["configs"]["50"]
        finally:
            calibration.CONFIG_FOLDER = real_folder
        load()
        assert manifest()["key"] == key

        # A new version, another public truth, other kick-offs.
        calibration.CALIBRATION_VERSION += 1
        try:
            load()
            assert manifest()["key"]["calibration_version"] == calibration.CALIBRATION_VERSION
        finally:
            calibration.CALIBRATION_VERSION -= 1
        load()
        other = gt.copy()
        other[60:70] = A
        load(other)
        assert manifest()["key"]["public_gt_sha256"] != key["public_gt_sha256"]
        load()
        inputs.kickoffs = {1: 61, 2: 340}
        load()
        assert manifest()["key"]["kickoffs"]["1"] == 61
        inputs.kickoffs = {1: 60, 2: 340}
        load()

        # Unreadable is absent: an array cut short, an array gone, a manifest that is not JSON or lacks a field.
        good = calibration.digest(cal)
        for damage in ("truncate", "delete", "garbage", "field", "empty"):
            if damage == "truncate":
                with open(os.path.join(folder, "gap_durations_s.npy"), "wb") as handle:
                    handle.write(b"\x93NUMPY")
            elif damage == "delete":
                os.remove(os.path.join(folder, "ball_masks_1.npy"))
            elif damage == "garbage":
                with open(manifest_path, "wb") as handle:
                    handle.write(b"\x00 not json")
            elif damage == "field":
                content = manifest()
                del content["public_window"]
                with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(content, handle)
            else:
                os.remove(manifest_path)
            assert calibration.digest(load()) == good, damage
            assert os.path.getsize(os.path.join(folder, "gap_durations_s.npy")) > 10, damage
        assert calibration.digest(load()) == good


# -- our match --------------------------------------------------------------------------------------------------------

def test_data_calibration_replicates_a01():
    """The calibration of BAR-ATH reproduces what A01, A02 and R001 published, by the same definitions.

    Prints aggregates only: counts, shares and quantiles of the match, never a frame, a position or a label.
    """
    from pipeline.inputs import Inputs
    inputs = Inputs()
    gt = calibration.public_gt()
    cal = calibration.load()
    # Measured twice, the same: arrays, scalars and all (the cache is not what makes it so).
    assert calibration.digest(calibration.measure(inputs, gt)) == calibration.digest(cal), "not deterministic"
    description = calibration.describe(cal)
    assert_no_frame_data(description)

    scalars = cal.scalars

    def near(name, value, expected, tolerance):
        assert abs(value - expected) <= tolerance, "{}: {} is not {} +/- {}".format(name, value, expected, tolerance)
        print("      {:<26} {:>10.4f}   (expected {} +/- {})".format(name, value, expected, tolerance))

    near("missing share", scalars["missing_share"], *A01["missing_share"])
    assert (scalars["gaps"], scalars["interior_gaps"], scalars["near_line_onsets"]) == (
        A01["gaps"], A01["interior_gaps"], A01["near_line_onsets"]), (
        scalars["gaps"], scalars["interior_gaps"], scalars["near_line_onsets"])
    print("      gaps / interior / near-line onsets   {} / {} / {}".format(
        scalars["gaps"], scalars["interior_gaps"], scalars["near_line_onsets"]))
    near("near-line share of interior", scalars["near_line_share"], 42 / 147, 5e-5)
    near("median gap (s)", float(np.median(cal.gap_durations_s)), *A01["median_gap_s"])
    for half, expected in A02_PERCENT.items():
        percent = tuple(int(round(100 * share)) for share in cal.missing_share_5min[half])
        assert percent == expected, "A02, half {}: {} instead of {}".format(half, percent, expected)
    print("      A02 (%), half 1 {}".format(A02_PERCENT[1]))
    print("                half 2 {}".format(A02_PERCENT[2]))
    assert scalars["tracklets"] == A01["tracklets"] == len(cal.tracklet_lengths_s), scalars["tracklets"]
    print("      tracklets                           {}".format(scalars["tracklets"]))
    near("wrong-team share", cal.wrong_team_share, *A01["wrong_team_share"])
    near("speed-NaN share", cal.speed_nan_share, *A01["speed_nan_share"])
    assert (scalars["wrong_team_episodes_home"], scalars["wrong_team_episodes_away"]) == PLAN["wrong_team_episodes"], (
        scalars["wrong_team_episodes_home"], scalars["wrong_team_episodes_away"])
    print("      wrong-team episodes (home / away)   {} / {}".format(
        scalars["wrong_team_episodes_home"], scalars["wrong_team_episodes_away"]))
    near("present mean", scalars["present_mean"], *PLAN["present_mean"])
    near("share with 22 present", scalars["present_share_22"], *PLAN["present_share_22"])
    near("tracklet median (s)", float(np.median(cal.tracklet_lengths_s)), *PLAN["tracklet_median_s"])
    near("jitter RMS (mm)", 1000 * cal.player_jitter_rms_m, *PLAN["jitter_rms_mm"])
    near("nearest-player agreement", cal.public_window["nearest_agreement"], *A01["nearest_agreement"])
    assert cal.public_window["judged_frames"] == 5640, cal.public_window
    print("      judged frames                       {}".format(cal.public_window["judged_frames"]))
    print("      dead-ball visible share             {:.4f}   of {} dead frames".format(
        cal.public_window["dead_ball_visible_share"], cal.public_window["dead_frames"]))
    for width, expected in BASELINE_POSSESSIONS.items():
        got = cal.baseline_outputs[width]
        assert abs(got["possessions"] - expected) <= POSSESSION_TOLERANCE, (
            "nearest_w{}: {} possessions, A07 expects {} +/- {}".format(width, got["possessions"], expected,
                                                                      POSSESSION_TOLERANCE))
        print("      nearest_w{:<4} possessions {:>5}  ({} expected)  per graded minute {:.2f}  D share {:.4f}".format(
            width, got["possessions"], expected, got["poss_per_min"], got["d_share"]))
    near("ball-speed NaN share (T31)", scalars["ball_speed_nan_share"], *PLAN["ball_speed_nan_share"])
