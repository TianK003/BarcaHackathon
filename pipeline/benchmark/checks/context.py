"""Checks of the `context` profile of the degrader (pipeline/benchmark/degrade.py): the tracking profile's players, and a
ball whose gaps are placed by the hazard that was fitted on our own match (A08) instead of copied from it.

The profile is the part of P01 that ties the gaps to the play, so what is held against it here is each convention of how
the hazard is carried to another match, one by one: what it is given (the clean DFL ball and the degraded players, in
metres, from the half's kick-off), what comes out (c * HAZARD_SCALE * the model, capped, and 0 where a feature is
undefined), what the sampler is told (the class of the at-risk frame, the third of the half, the durations at the
match's rate) and what is counted. Most run on hand-built halves whose every count is worked out in the docstring: a
ball that is visible on every row, six players who never leave, a hazard so strong that a gap starts on every visible
frame (capped_match), so that the mask is a pattern that can be written down. reference_mask restates the whole
construction one row at a time, from the documentation and not from the code. What no check may see is the truth:
test_context_never_sees_truth gives two matches that differ only in their flags and asks for the same bytes.

The players are `tracking` version 2's (test_context_players_equal_tracking), and so are the draws that the cap reads
(the Transplant): the profile changes the ball's mask and nothing else. The checks that need a real match and our own
Calibration (the one measurement of the stop conditions, R005) are test_data_* (--data, NOT RUN where the DFL files are
not there) and print aggregates only. The frozen scale of the odds, and the fidelity gates, are checked where the other
constants and gates are: test_hazard_scale_* here and checks/fidelity.py.
"""

import dataclasses
import json
import os
import types

import numpy as np

from pipeline.benchmark import calibration, degrade, dfl, hazard
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (HAZARD_SECONDS, NotRun, context_namespace, leaks, made_up_calibration,
                                              made_up_hazard, real_match, steady_match, without_new_parts)
from pipeline.selftest import expect_error

# Params
# The record of the ball (info["ball"]): the scale and c, and counts per half and in all. No frame.
BALL_COUNTS = ("rows", "ball_missing_rows", "nan_feature_rows", "capped_rows", "onsets", "missing_frames",
               "at_risk_rows", "post_gap_rows")
BALL_KEYS = ("scale", "c") + BALL_COUNTS + ("halves",)
INFO_KEYS = ["ball", "mask_share", "match_id", "match_index", "players", "profile", "seed", "transplants", "version"]
NEAR_M = 2.0                     # a ball closer than this to a line is near it (calibration.NEAR_LINE_M, strict)
CAPPED_P = 1.0 - 1e-12           # a chance so close to 1 that c * scale * p is above the cap (1 - 1e-9) on every row
# What the halves of capped_match say of the hazard: gaps of exactly 3 frames whatever the class and the third (0.12 s at
# 25 Hz), and the chance CAPPED_P of an onset on every frame where all the features are defined.
CAPPED_SECONDS = {onset_class: {third: (0.12,) for third in range(3)} for onset_class in (0, 1)}


# -- helpers ----------------------------------------------------------------------------------------------------------

def everything(atd):
    """The bytes of every array of a degraded Atd: equal bytes are the same tracking, to the last bit."""
    names = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
    return b"".join(getattr(atd, name).tobytes() for name in names)


def nobody_lost(calibration):
    """The Calibration with every frame at 22 slots present, and no wrong-team episodes and no NaN-speed runs: with it
    each player is one tracklet for each stretch he is on the pitch, and the cap changes nothing."""
    return without_new_parts(calibration, present_counts={
        half: np.full(len(mask), 22, dtype=np.int16) for half, mask in calibration.ball_masks.items()})


def capped_match():
    """(match, calibration) of the hand-built halves: 240 and 200 rows, the ball seen on every row, six players on the
    pitch for all of both halves, nobody lost, and a hazard of CAPPED_P on every frame with gaps of 3 frames.

    A gap starts on every visible frame whose features are defined (the probability is capped at 1 - 1e-9, 20.7 of the
    hazard's running sum a frame, and the draw E of the exponential is under that unless it is 1 in 10^9), so the mask
    is a pattern. The ball's speed is the only feature that can be undefined here (the players never leave, so the
    nearest one is always there): it is NaN on a half's first row, whose step is not defined, so row 0 is not at risk,
    the first gap starts after row 1 (rows 2-4), the next after row 5 (rows 6-8), and so on every 4 rows: the frames seen
    are row 0 and the rows 1, 5, 9, ... and every other row is missing, up to the end of the half, which cuts the last
    gap."""
    match = steady_match(substitutions=False)
    calibration = nobody_lost(context_namespace(p=CAPPED_P, seconds=CAPPED_SECONDS))
    return match, calibration


def drawn_mask(atd, match, half):
    """(n,) bool: the rows of a half, from its kick-off, where the degraded ball is missing and DFL's ball is there."""
    kickoff, stop = match.kickoffs[half], next(last for number, _, last in match.graded if number == half)
    return np.isnan(atd.ball_xy[kickoff:stop, 0]) & ~np.isnan(match.ball_xy[kickoff:stop, 0])


def gaps_of(mask):
    """[(first, stop)] of the maximal runs of True in a 1-D bool array, half-open, found with np.diff."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def reference_mask(view, half, players_xy, calibration, seed, scale=None):
    """The ball's mask of a half as the documentation builds it, with plain loops: (n,) bool from the kick-off.

    The features of the clean ball and the given players (metres, from the kick-off), then row by row: 0 where a feature
    is NaN, else c * scale * the model's probability for the row, at most 1 - 1e-9; the class is 1 where the line
    distance of the row is below 2 m; the third is 0, 1 or 2 from the minute (the row over the rate over 60) of the row;
    the durations are max(1, round(seconds * fps)). hazard.sample_mask draws from rng_for(seed, match_index, "hazard",
    half)."""
    scale = degrade.HAZARD_SCALE if scale is None else scale
    kickoff, stop = view.kickoffs[half], next(last for number, _, last in view.graded if number == half)
    n = stop - kickoff
    pitch = np.array(view.pitch)
    table = hazard.features(np.asarray(view.ball_xy)[kickoff:stop] * pitch, np.asarray(players_xy)[kickoff:stop] * pitch,
                            view.pitch, view.fps, kickoff, kickoff)
    prob, onset_class, third = np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for row in range(n):
        if not np.isnan(table[row]).any():
            raw = calibration.hazard_c * scale * float(hazard.predict(calibration.hazard_model, table[row][None, :])[0])
            prob[row] = min(raw, 1.0 - 1e-9)
            onset_class[row] = 1 if table[row][0] < NEAR_M else 0
        minute = row / view.fps / 60.0
        third[row] = 0 if minute < 15 else 1 if minute < 30 else 2
    durations = [[np.array([max(1, round(seconds * view.fps)) for seconds in calibration.hazard_durations_s[c][t]])
                  for t in range(3)] for c in range(2)]
    return hazard.sample_mask(prob, onset_class, third, durations,
                              degrade.rng_for(seed, view.match_index, "hazard", half))


# -- what the profile is made of ---------------------------------------------------------------------------------------

def test_context_never_sees_truth():
    """Two matches that differ only in ball_status and ball_possession give byte-identical context Atds and records.

    A behavioural check: the flags of the second match are another random set (the first's are DFL's synthetic ones, as
    steady_match has them), so a degradation that read either, in the hazard's features, in the class, in the third or in
    the mask, would come out different. It is run through the degrader and through the runner's own path (make_atd takes
    the match and builds the view itself), for two seeds. The inputs that a context match is served with hold nothing of
    the truth either, to any depth (the deep walk of Task 8), and no field of the view is the truth."""
    match = steady_match()
    rng = np.random.default_rng([12])
    status = rng.integers(0, 2, size=match.n_frames).astype(np.int8)
    possession = rng.integers(0, 2, size=match.n_frames).astype(np.int8)
    for array in (status, possession):
        array.setflags(write=False)
    other = dataclasses.replace(match, ball_status=status, ball_possession=possession)
    assert not np.array_equal(match.ball_status, other.ball_status) and not np.array_equal(
        match.ball_possession, other.ball_possession), "the two matches must differ in their flags"
    assert np.array_equal(match.players_xy, other.players_xy, equal_nan=True) and np.array_equal(
        match.ball_xy, other.ball_xy, equal_nan=True), "and in nothing else"
    cal = context_namespace()
    for seed in (0, 1):
        first, info = degrade.degrade(degrade.tracking_view(match), "context", cal, seed)
        second, info_other = degrade.degrade(degrade.tracking_view(other), "context", cal, seed)
        assert everything(first) == everything(second) and info == info_other, seed
        assert everything(benchmark_run.make_atd(match, "context", seed, cal)) == everything(first)
        assert everything(benchmark_run.make_atd(other, "context", seed, cal)) == everything(first)
        assert info["ball"]["missing_frames"] > 0, "a degradation that drew nothing proves nothing"
    view = degrade.tracking_view(match)
    atd, _ = degrade.degrade(view, "context", cal, 0)
    inputs = match_inputs.MatchInputs(match, atd, profile="context", seed=0)
    assert leaks(inputs, match) == [] and leaks(view, match) == []
    assert inputs.profile == "context" and inputs.atd is atd and inputs.atd.n_slots == 46
    assert inputs.atd_manifest["profile"] == "context"
    assert "array" not in repr(atd), "no frame in a repr"


def test_context_players_equal_tracking():
    """Every player array of a context Atd and the players' record are tracking's, byte for byte, for the same seed.

    The slots' positions and speeds and the slot teams, the record of the players' draws (info["players"]) and the draws
    that the cap reads, the Transplant of each half (info["transplants"]): the profile changes the ball and nothing else,
    so that a difference between two reports is the ball's. On the steady match with the made-up Calibration, for three
    seeds, and on a match with another kick-off for the second half; the same on a real DFL match is
    test_data_context_profile_on_a_real_match_with_our_real_calibration. The ball is not tracking's: the masks differ
    (the other's is a transplant of ours), and it has no x without a y and no speed."""
    cal = context_namespace()
    for match in (steady_match(), steady_match("RND002", 1, rows=(260, 190))):
        view = degrade.tracking_view(match)
        for seed in (0, 1, 2):
            context, info = degrade.degrade(view, "context", cal, seed)
            tracking, tracking_info = degrade.degrade(view, "tracking", cal, seed)
            for name in ("players_xy", "players_speed", "slot_team"):
                assert getattr(context, name).tobytes() == getattr(tracking, name).tobytes(), (seed, name)
                assert getattr(context, name).dtype == getattr(tracking, name).dtype
            assert info["players"] == tracking_info["players"] and info["transplants"] == tracking_info["transplants"]
            assert context.ball_xy.tobytes() != tracking.ball_xy.tobytes(), "another mask: the ball is not tracking's"
            assert np.isnan(context.ball_speed).all() and context.ball_speed.shape == (match.n_frames,)
            gone = np.isnan(context.ball_xy)
            assert (gone[:, 0] == gone[:, 1]).all(), "x and y are missing together"
            assert info["profile"] == "context" and info["version"] == 1 == degrade.PROFILES["context"]
            assert info["players"]["tracklets"] > 0
            assert [info["transplants"][half]["frames"] for half in (1, 2)] == [
                stop - match.kickoffs[half] for half, _, stop in match.graded]
            for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
                assert not getattr(context, name).flags.writeable, name


def test_context_is_deterministic_and_independent_of_the_other_matches():
    """The same seed gives the same bytes and the same record; another seed, match_index or match gives another mask.

    A match gets the same degradation whichever matches are run with it and in whatever order, and the record of the
    ball is JSON-ready: ints and floats, no frame of anybody, and each total is the sum of its halves'."""
    matches = {"RND001": steady_match("RND001", 0), "RND002": steady_match("RND002", 1, rows=(260, 190)),
               "J03WMX": steady_match("J03WMX", 2)}
    cal = context_namespace()

    def keyed(order):
        return {(seed, match_id): everything(degrade.degrade(degrade.tracking_view(matches[match_id]), "context", cal,
                                                             seed)[0])
                for seed in (0, 1) for match_id in order}

    forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
    assert forward == backward and len(set(forward.values())) == len(forward)
    assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "alone, a match gets what it got in the run"
    view = degrade.tracking_view(matches["RND001"])
    first, info = degrade.degrade(view, "context", cal, 0)
    again, info_again = degrade.degrade(degrade.tracking_view(matches["RND001"]), "context", context_namespace(), 0)
    assert everything(first) == everything(again) and info == info_again
    elsewhere, _ = degrade.degrade(degrade.tracking_view(matches["RND001"], match_index=3), "context", cal, 0)
    assert elsewhere.ball_xy.tobytes() != first.ball_xy.tobytes(), "another match_index is another stream"
    drawn = {degrade.degrade(view, "context", cal, seed)[0].ball_xy.tobytes() for seed in range(12)}
    assert len(drawn) == 12, "another seed is another mask: the draws are not stuck"
    assert json.loads(json.dumps(info, sort_keys=True)) == json.loads(json.dumps(info_again, sort_keys=True))
    assert sorted(info) == INFO_KEYS, sorted(info)
    ball = info["ball"]
    assert sorted(ball) == sorted(BALL_KEYS) and sorted(ball["halves"]) == [1, 2]
    assert all(sorted(record) == sorted(BALL_COUNTS) for record in ball["halves"].values())
    for name in BALL_COUNTS:
        assert ball[name] == sum(record[name] for record in ball["halves"].values()), name
    assert all(isinstance(ball[name], int) for name in BALL_COUNTS)
    assert ball["scale"] == degrade.HAZARD_SCALE and ball["c"] == 1.0
    assert ball["rows"] == sum(stop - start for _, start, stop in view.graded)
    assert abs(info["mask_share"] - ball["missing_frames"] / ball["rows"]) < 1e-12, "the mask's share"
    assert ball["missing_frames"] == sum(int(drawn_mask(first, matches["RND001"], half).sum()) for half in (1, 2))
    assert ball["at_risk_rows"] + ball["missing_frames"] <= ball["rows"]
    assert 0 < ball["onsets"] <= ball["missing_frames"]


def test_context_counts_and_mask_on_hand_built_halves():
    """Every count of the record, and the mask itself, worked out by hand on capped_match.

    Half 1 has 240 rows: row 0 has no ball speed, so it is not at risk (one NaN-feature row); every other row has its
    features defined, and its chance is capped (239 capped rows); the visible frames are row 0 and rows 1, 5, ..., 237
    (60 rows after row 0), a gap of 3 frames follows each of the 60 events: rows 2-4, ..., 234-236 and 238-240, the last
    cut at row 239, so 59 * 3 + 2 = 179 frames are missing and 60 gaps started. The at-risk frames are the visible ones
    with defined features, not the last row of the half: rows 1, 5, ..., 237, that is 60. The first rows after a gap
    (5 frames from where it ends, the ones whose ball speed would carry a speed from before it in ours) are rows 5, 9,
    ..., 237, 59 of the 60 (row 1 follows no gap; the last gap has no end). Half 2 has 200 rows and the same pattern:
    1 NaN-feature row, 199 capped, 50 gaps, 149 missing frames, 50 at risk and 49 after a gap. In all 440 rows, 2
    NaN-feature rows, 438 capped, 110 onsets, 328 missing frames, 110 at-risk and 108 after a gap, and none of DFL's own
    rows without a ball. And the mask is the pattern: row k of a half is missing for k >= 2 where (k - 1) % 4 != 0."""
    match, cal = capped_match()
    view = degrade.tracking_view(match)
    for seed in (0, 3):
        atd, info = degrade.degrade(view, "context", cal, seed)
        ball = info["ball"]
        assert ball["halves"][1] == {"rows": 240, "ball_missing_rows": 0, "nan_feature_rows": 1, "capped_rows": 239,
                                     "onsets": 60, "missing_frames": 179, "at_risk_rows": 60, "post_gap_rows": 59}
        assert ball["halves"][2] == {"rows": 200, "ball_missing_rows": 0, "nan_feature_rows": 1, "capped_rows": 199,
                                     "onsets": 50, "missing_frames": 149, "at_risk_rows": 50, "post_gap_rows": 49}
        assert {name: ball[name] for name in BALL_COUNTS} == {
            "rows": 440, "ball_missing_rows": 0, "nan_feature_rows": 2, "capped_rows": 438, "onsets": 110,
            "missing_frames": 328, "at_risk_rows": 110, "post_gap_rows": 108}
        for half in (1, 2):
            mask = drawn_mask(atd, match, half)
            expected = np.array([k >= 2 and (k - 1) % 4 != 0 for k in range(mask.size)])
            assert mask.tolist() == expected.tolist(), (seed, half)
        assert abs(info["mask_share"] - 328 / 440) < 1e-12
    # Rows without a DFL ball are counted apart, stay missing and are never at risk: 10 rows of half 1 with no ball
    # (rows 1-10). The ball's speed has no step until it is seen on two rows in a row, which is rows 11 and 12, so row 11,
    # where the ball is back, is a NaN-feature row as well as row 0: two in the half.
    holed = np.array(match.ball_xy)
    holed[1:11] = np.nan
    holed.setflags(write=False)
    atd, info = degrade.degrade(degrade.tracking_view(dataclasses.replace(match, ball_xy=holed)), "context", cal, 0)
    assert info["ball"]["halves"][1]["ball_missing_rows"] == 10 and info["ball"]["halves"][1]["nan_feature_rows"] == 2
    assert np.isnan(atd.ball_xy[1:11]).all(), "a ball that DFL does not have stays missing"
    assert info["ball"]["halves"][2] == degrade.degrade(view, "context", cal, 0)[1]["ball"]["halves"][2], "the other half"


def test_context_mask_follows_prob_and_rows_without_features_are_never_at_risk():
    """Where the features are undefined the chance is 0, so every gap starts inside the stretch where they are defined.

    The six players are on the pitch only on rows 80-159 of the first half and rows 60-139 of the second (counted from
    each half's start), so that no player is there to be nearest to the ball anywhere else: the feature is NaN, and a row
    like that was never at risk in the fit. With a chance of 0.08 a frame inside, every gap's at-risk frame (the one
    before its first missing frame) is inside the stretch; the gap itself may run past its end. The rows outside are
    counted as NaN-feature rows, and only they: the half's rows less the stretch (row 0, which has no ball speed, is
    outside it, and every row inside has a step before it and a player)."""
    match = steady_match(substitutions=False)
    xy = np.array(match.players_xy)
    stretches = {}
    for half, start, stop in match.graded:
        first, last = start + (80 if half == 1 else 60), start + (160 if half == 1 else 140)
        stretches[half] = (first, last)
        xy[start:first] = np.nan
        xy[last:stop] = np.nan
    xy.setflags(write=False)
    match = dataclasses.replace(match, players_xy=xy)
    cal = nobody_lost(context_namespace(p=0.08))
    view = degrade.tracking_view(match)
    onsets = 0
    for seed in range(8):
        atd, info = degrade.degrade(view, "context", cal, seed)
        for half, start, stop in match.graded:
            mask = drawn_mask(atd, match, half)
            first, last = stretches[half]
            for gap_first, _ in gaps_of(mask):
                at_risk = start + gap_first - 1
                assert first <= at_risk < last, (seed, half, at_risk, (first, last))
                onsets += 1
            record = info["ball"]["halves"][half]
            assert record["nan_feature_rows"] == (stop - start) - (last - first), (seed, half, record)
            assert record["onsets"] == len(gaps_of(mask)) and record["capped_rows"] == 0
    assert onsets > 20, "a check with no gap proves nothing: {}".format(onsets)


def test_context_mask_is_the_hazards_draw_for_the_documented_inputs():
    """The ball's mask of every half is hazard.sample_mask on the inputs the documentation names, drawn from
    rng_for(seed, match_index, "hazard", half): reference_mask restates them one row at a time and the two agree.

    The features are those of the clean ball and the DEGRADED players (the Atd's own, lost spans and all, in metres from
    a corner), from the kick-off; the chance is c * HAZARD_SCALE * the model, capped, 0 where a feature is NaN; the
    class, the third and the durations are read at the at-risk frame. Two hazards (a flat one and one with weights on
    the nearest player, the players near the ball and the minute), both halves, three seeds, and a second half whose
    kick-off is 30 rows after its block starts: the mask covers from the kick-off and the rows before it keep the
    ball."""
    match = steady_match()
    window = dataclasses.replace(match, kickoffs={1: match.kickoffs[1], 2: match.kickoffs[2] + 30})
    weights = [0.0, 0.0, 0.0, -0.05, 0.3, 0.0]     # per metre to the nearest player, per player within 3 m
    for hazard_fields in ({"p": 0.03}, {"p": 0.03, "c": 1.7, "coef": weights}):
        cal = context_namespace(**hazard_fields)
        for case in (match, window):
            view = degrade.tracking_view(case)
            for seed in (0, 1, 2):
                atd, info = degrade.degrade(view, "context", cal, seed)
                missing = 0
                for half, start, stop in case.graded:
                    kickoff = case.kickoffs[half]
                    wanted = reference_mask(view, half, atd.players_xy, cal, seed)
                    assert wanted.size == stop - kickoff
                    got = drawn_mask(atd, case, half)
                    assert got.tolist() == wanted.tolist(), (hazard_fields, seed, half)
                    assert not np.isnan(atd.ball_xy[start:kickoff]).any(), "before the kick-off the ball is DFL's"
                    missing += int(wanted.sum())
                assert info["ball"]["missing_frames"] == missing > 0
    # The hazard's weights matter: with them the mask is another one (the check above could not tell otherwise).
    view = degrade.tracking_view(match)
    flat, _ = degrade.degrade(view, "context", context_namespace(p=0.03), 0)
    weighted, _ = degrade.degrade(view, "context", context_namespace(p=0.03, c=1.7, coef=weights), 0)
    assert flat.ball_xy.tobytes() != weighted.ball_xy.tobytes()


def long_match(rows, near=500):
    """steady_match of `rows` whose ball stands in the middle of the pitch, except on alternating stretches of `near`
    rows (the second, fourth, ...), when it is 1.0 m from the touch line: so every at-risk frame is of a known class."""
    match = steady_match(rows=rows, substitutions=False)
    ball = np.array(match.ball_xy)
    for _, start, stop in match.graded:
        near_rows = (np.arange(stop - start) // near) % 2 == 1
        ball[start:stop, 0] = 0.5
        ball[start:stop, 1] = np.where(near_rows, 1.0 / 68.0, 0.5)
    ball.setflags(write=False)
    return dataclasses.replace(match, ball_xy=ball)


def test_context_durations_follow_the_class_the_third_and_the_rate_of_the_at_risk_frame():
    """A gap lasts what the table says for the class (a ball within 2 m of a line) and the third of the half (minute 0-15,
    15-30, later) of the frame before it, at the match's rate.

    A first half of 46,000 rows (30.7 min at 25 Hz) whose ball is 1 m from the line in alternating stretches of 500
    rows and in the middle of the pitch in the others, a chance of 0.02, and a table where each of the six cells has a
    length of its own: 30, 31 and 32 frames for the far ones and 7, 8 and 9 for the near ones at 25 Hz (1.2, 1.24, 1.28
    and 0.28, 0.32, 0.36 s). Every gap that ends inside the half has the length of the cell of its at-risk frame (the one
    before it: class from the line distance there, third from the minute there), and every cell is used. At 50 Hz the
    same table is twice as long in frames and the thirds are twice as late, so the stretches are the same rows and the
    third ones are those of 45,000 rows or more: the conversion reads the match's rate, in both the durations and the
    minute."""
    seconds = {0: {0: (1.2,), 1: (1.24,), 2: (1.28,)}, 1: {0: (0.28,), 1: (0.32,), 2: (0.36,)}}
    cal = nobody_lost(context_namespace(p=0.02, seconds=seconds))
    base = long_match((46000, 300))
    for fps in (25, 50):
        match = dataclasses.replace(base, fps=fps)
        view = degrade.tracking_view(match)
        atd, info = degrade.degrade(view, "context", cal, 5)
        start, stop = match.graded[0][1:]
        mask = drawn_mask(atd, match, 1)
        used = {}
        for first, last in gaps_of(mask):
            if last == mask.size:
                continue                                    # cut by the end of the half
            row = first - 1
            line = min(0.5 * 105.0, 1.0 if (row // 500) % 2 == 1 else 34.0)
            onset_class = 1 if line < NEAR_M else 0
            minute = row / fps / 60.0
            third = 0 if minute < 15 else 1 if minute < 30 else 2
            frames = max(1, round(seconds[onset_class][third][0] * fps))
            assert last - first == frames, (fps, row, onset_class, third, last - first, frames)
            used[(onset_class, third)] = used.get((onset_class, third), 0) + 1
        cells = {(c, t) for c in (0, 1) for t in (0, 1, 2)} if fps == 25 else {(c, t) for c in (0, 1) for t in (0, 1)}
        assert set(used) == cells and min(used.values()) >= 1, (fps, used)
        assert info["ball"]["onsets"] >= sum(used.values())
    expected = degrade.context_durations(cal, 25)
    assert [[array.tolist() for array in row] for row in expected] == [[[30], [31], [32]], [[7], [8], [9]]]
    assert [[array.tolist() for array in row] for row in degrade.context_durations(cal, 50)] == [
        [[60], [62], [64]], [[14], [16], [18]]]
    tiny = context_namespace(seconds={c: {t: (0.002, 0.06, 0.1) for t in range(3)} for c in (0, 1)})
    assert degrade.context_durations(tiny, 25)[0][0].tolist() == [1, 2, 2], "at least 1 frame, and round half to even"
    assert all(array.dtype == np.int64 and not array.flags.writeable for row in expected for array in row)


def test_context_probability_is_c_times_the_scale_times_the_model_and_is_capped():
    """The chance of an onset is hazard_c * HAZARD_SCALE * the model's probability, at most 1 - 1e-9, and the rows where
    it reaches the cap are counted.

    The model gives 0.01 on every frame, and capped_match's halves have 438 rows with defined features. At c = 10 and the
    scale 1 the chance is 0.1 (no capped row); at c = 10 and a scale of 100 it is 10, so every one of the 438 is capped;
    at c = 100 and the scale 1 the same; and at c = 1 and the scale 1000 the same. The record says which scale and which
    c were used. The constant is read when the profile is made, not when the module loads: patched, the profile takes
    the patched one, and put back, the frozen one."""
    match, _ = capped_match()
    view = degrade.tracking_view(match)

    def made(c):
        return nobody_lost(context_namespace(p=0.01, c=c, seconds=CAPPED_SECONDS))

    saved = degrade.HAZARD_SCALE
    try:
        for c, scale, capped in ((10.0, 1.0, 0), (10.0, 100.0, 438), (100.0, 1.0, 438), (1.0, 1000.0, 438),
                                 (1.0, 1.0, 0), (50.0, 2.0, 438)):
            degrade.HAZARD_SCALE = scale
            _, info = degrade.degrade(view, "context", made(c), 0)
            assert info["ball"]["capped_rows"] == capped, (c, scale, info["ball"])
            assert info["ball"]["scale"] == scale and info["ball"]["c"] == c
    finally:
        degrade.HAZARD_SCALE = saved
    _, info = degrade.degrade(view, "context", made(1.0), 0)
    assert info["ball"]["scale"] == saved == degrade.HAZARD_SCALE, "put back, the frozen scale"
    assert degrade.MAX_HAZARD_PROB == 1.0 - 1e-9


def odds_rescale(scale, target, measured):
    """HAZARD_SCALE as the one-time rescale of the odds gives it: scale * [t / (1 - t)] / [m / (1 - m)]. Gaps start only
    from visible frames, so missing / visible, the odds of a frame being missing, is about linear in the hazard, and the
    share is not: a plain ratio of shares would undershoot."""
    return scale * ((target / (1.0 - target)) / (measured / (1.0 - measured)))


def test_hazard_scale_odds_rescale_arithmetic():
    """The rescale is on the odds, worked out by hand.

    A share of 0.5 is odds of 1 and one of 0.25 is odds of 1/3, so going from 0.25 to 0.5 takes a factor of 3 (a plain
    ratio of the shares would say 2); 0.4 from 0.2 is odds 2/3 against 1/4: 8/3 (not 2); equal shares give 1 and keep the
    scale; the factor carries the current scale; and the odds of the result are the target's, which is what the rescale
    is for. The odds are not symmetric: measuring above the target gives a factor below 1."""
    assert abs(odds_rescale(1.0, 0.5, 0.25) - 3.0) < 1e-12
    assert abs(odds_rescale(1.0, 0.4, 0.2) - 8.0 / 3.0) < 1e-12
    assert odds_rescale(1.0, 0.3, 0.3) == 1.0 and odds_rescale(2.5, 0.3, 0.3) == 2.5
    assert abs(odds_rescale(2.0, 0.5, 0.25) - 6.0) < 1e-12, "the current scale is a factor"
    assert abs(odds_rescale(1.0, 0.25, 0.5) - 1.0 / 3.0) < 1e-12 and odds_rescale(1.0, 0.25, 0.5) < 1.0
    for scale, target, measured in ((1.0, 0.411, 0.28), (1.7, 0.3, 0.45)):
        factor = odds_rescale(scale, target, measured) / scale
        after = measured / (1.0 - measured) * factor
        assert abs(after / (1.0 + after) - target) < 1e-12, "the odds of the result are the target's"
    assert odds_rescale(1.0, 0.4, 0.2) > 0.4 / 0.2, "not the plain ratio of the shares, which undershoots"


def test_context_ball_keeps_dfls_holes_and_the_rows_before_the_kickoff():
    """The ball is the clean DFL ball with the mask taken out: x and y together, its speed NaN everywhere, a row that DFL
    has no ball for stays missing, and the rows outside the span (before the kick-off, between the halves) are DFL's.

    Where the hazard's chance is 0 (a flat model with a tiny p) nothing is taken out, and the ball is the clean one with
    a NaN speed: the same bytes as the gaps profile with masks that say nothing is missing."""
    match = steady_match()
    window = dataclasses.replace(match, kickoffs={1: match.kickoffs[1] + 40, 2: match.kickoffs[2] + 30})
    nothing = context_namespace(p=1e-12, c=1e-6)
    atd, info = degrade.degrade(degrade.tracking_view(match), "context", nothing, 0)
    assert info["ball"]["onsets"] == 0 and info["ball"]["missing_frames"] == 0 and info["mask_share"] == 0.0
    silent = made_up_calibration()
    silent = types.SimpleNamespace(**{**vars(silent), "ball_masks": {
        half: np.zeros(len(mask), dtype=bool) for half, mask in silent.ball_masks.items()}})
    gaps, _ = degrade.degrade(degrade.tracking_view(match), "gaps", silent, 0)
    assert atd.ball_xy.tobytes() == gaps.ball_xy.tobytes() and atd.ball_speed.tobytes() == gaps.ball_speed.tobytes()
    assert np.isnan(atd.ball_speed).all()
    # With gaps: the rows before the kick-off and between the halves are not touched.
    busy = context_namespace(p=0.1)
    atd, info = degrade.degrade(degrade.tracking_view(window), "context", busy, 1)
    assert info["ball"]["onsets"] > 0
    inside = np.zeros(match.n_frames, dtype=bool)
    for half, _, stop in window.graded:
        inside[window.kickoffs[half]:stop] = True
    assert np.array_equal(atd.ball_xy[~inside], window.ball_xy[~inside], equal_nan=True)
    seen = ~np.isnan(atd.ball_xy[:, 0])
    assert np.array_equal(atd.ball_xy[seen], window.ball_xy[seen]), "what is seen is DFL's"
    gone = np.isnan(atd.ball_xy)
    assert (gone[:, 0] == gone[:, 1]).all() and info["ball"]["rows"] == sum(
        stop - window.kickoffs[half] for half, _, stop in window.graded)


def test_context_refuses_what_it_cannot_make():
    match = steady_match()
    view = degrade.tracking_view(match)
    expect_error(ValueError, lambda: degrade.degrade(view, "context", None, 0), "context", "Calibration",
                 "calibration.load()")
    expect_error(ValueError, lambda: degrade.degrade(view, "context", made_up_calibration(), 0), "context",
                 "hazard_model", "hazard_c", "hazard_durations_s", "calibration.load()")
    partial = types.SimpleNamespace(**{**vars(context_namespace()), "hazard_durations_s": {
        0: {0: np.array([1.0]), 1: np.array([1.0]), 2: np.array([1.0])}}})
    expect_error(ValueError, lambda: degrade.degrade(view, "context", partial, 0), "hazard_durations_s", "[1][0]",
                 "calibration.load()")
    expect_error(ValueError, lambda: degrade.degrade(view, "nonsense", context_namespace(), 0), "nonsense", "clean",
                 "context", "gaps", "tracking")
    expect_error(ValueError, lambda: degrade.degrade(view, "context", context_namespace(), -1), "seed", "at least 0")
    late = dataclasses.replace(match, kickoffs={1: 500, 2: match.kickoffs[2]})
    expect_error(ValueError, lambda: degrade.degrade(degrade.tracking_view(late), "context", context_namespace(), 0),
                 "RND001", "half 1", "kicks off")


def test_context_profile_is_registered_with_its_own_stream():
    """The profile is version 1, the hazard has the stream after the speed_nan one, and what the profile read of the
    Calibration is named: the profiles nest (context's players are tracking's, and the draws of the earlier parts stay)."""
    assert degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 2, "context": 1}
    assert degrade.CONTEXT == "context" and degrade.COMPONENTS["hazard"] == 4
    assert degrade.COMPONENTS == {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4}
    assert degrade.CONTEXT_FIELDS == ("hazard_model", "hazard_c", "hazard_durations_s")
    assert degrade.HAZARD_SCALE > 0 and calibration.NEAR_LINE_M == NEAR_M
    assert set(HAZARD_SECONDS) == {0, 1} and made_up_hazard()["hazard_c"] == 1.0


def test_data_context_profile_on_a_real_match_with_our_real_calibration():
    """Our own Calibration (the hazard fitted on our match, A08) on a downloaded DFL match: what the context profile
    makes, for information, and the real-match half of test_context_players_equal_tracking.

    Needs both our match and a downloaded DFL match: runs with --data and says NOT RUN where the DFL files are not there.
    Asserted: the players (the slots' positions and speeds, the slot teams) and their record, and the draws that the cap
    reads, are tracking's byte for byte; the draws are the same twice; x and y are missing together and the speed is NaN;
    nothing of the truth is reachable; the record's totals are the sums of its halves'. Printed (aggregates only, and what
    the stop conditions of R005 are read from): the share of the frames missing, the gaps, the NaN-feature, capped,
    at-risk and after-a-gap rows and their shares of the rows and of the at-risk rows."""
    from pipeline.benchmark import sources
    match_id = sources.MATCH_IDS[0]
    for kind in dfl.SOURCE_KINDS:
        if not os.path.isfile(sources.local_path(match_id, kind)):
            raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                match_id, kind, dfl.FETCH_ADVICE))
    cal = calibration.load()
    match = real_match(match_id)
    view = degrade.tracking_view(match)
    context, info = degrade.degrade(view, "context", cal, 0)
    again, info_again = degrade.degrade(view, "context", cal, 0)
    assert everything(context) == everything(again) and info == info_again
    tracking, tracking_info = degrade.degrade(view, "tracking", cal, 0)
    for name in ("players_xy", "players_speed", "slot_team"):
        assert getattr(context, name).tobytes() == getattr(tracking, name).tobytes(), name
    assert info["players"] == tracking_info["players"] and info["transplants"] == tracking_info["transplants"]
    assert np.isnan(context.ball_speed).all()
    gone = np.isnan(context.ball_xy)
    assert (gone[:, 0] == gone[:, 1]).all()
    inputs = match_inputs.MatchInputs(match, context, profile="context", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"
    ball = info["ball"]
    for name in BALL_COUNTS:
        assert ball[name] == sum(record[name] for record in ball["halves"].values()), name
    rows, at_risk = ball["rows"], ball["at_risk_rows"]
    print("      context on {} seed 0: {:,} rows, {:.1%} missing, {} gaps; NaN features {:,} ({:.3%}), no DFL ball {:,}, "
          "capped {:,}; at risk {:,}, {:,} within {} frames after a gap ({:.2%})".format(
              match_id, rows, ball["missing_frames"] / rows, ball["onsets"], ball["nan_feature_rows"],
              ball["nan_feature_rows"] / rows, ball["ball_missing_rows"], ball["capped_rows"], at_risk,
              ball["post_gap_rows"], hazard.SPEED_STEPS, ball["post_gap_rows"] / at_risk))

