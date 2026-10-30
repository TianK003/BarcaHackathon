"""Checks of label_diff (pipeline/benchmark/label_diff.py): how a config changes BAR-ATH's labels, in aggregates.

The aggregates are worked out here on two hand-built code arrays over a stand-in Inputs whose every figure is known by
construction: the shares, the possessions per half, the share of changes in ball gaps, the moved D onsets and their
median move. The printout is held to its rule: figures and no frame (CLAUDE.md, hard rules). No check reads data.
"""

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import label_diff
from pipeline.selftest import expect_error, kinematics_atd, kinematics_inputs

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
N = 3000
BLOCKS = [(1, 1000, 1600), (2, 2000, 2600)]     # four-digit edges, so that a printed frame would be found
REFERENCE = [(1000, 1200, H), (1200, 1300, D), (1300, 1500, A), (1500, 1600, H),
             (2000, 2100, A), (2100, 2150, D), (2150, 2400, A), (2400, 2600, D)]
NEW = [(1000, 1230, H), (1230, 1330, D), (1330, 1500, A), (1500, 1520, X), (1520, 1600, H),
       (2000, 2100, A), (2100, 2150, D), (2150, 2250, H), (2250, 2400, A), (2400, 2600, D)]


def codes_of(pieces, fill=X):
    codes = np.full(N, fill, dtype=np.int8)
    for start, stop, code in pieces:
        codes[start:stop] = code
    return codes


def scene(ball_missing=()):
    """A stand-in Inputs over N frames with two graded blocks; the ball is NaN on the given (start, stop) spans."""
    ball = np.full((N, 2), 0.5)
    for start, stop in ball_missing:
        ball[start:stop] = np.nan
    players = np.full((N, 2, 2), 0.5)
    inputs = kinematics_inputs(kinematics_atd(players, ball), list(BLOCKS))
    inputs.n_frames = N
    inputs.half_of_frame = np.zeros(N, dtype=np.int8)
    for half, start, stop in BLOCKS:
        inputs.half_of_frame[start:stop] = half
    inputs.graded_mask = inputs.half_of_frame > 0
    return inputs


def test_label_diff_on_hand_built_codes():
    # 1200 graded frames. Changed frames: 1200-1230 (ref D, new H: 30), 1300-1330 (ref A, new D: 30),
    # 1500-1520 (ref H, new X: 20), 2150-2250 (ref A, new H: 100).
    inputs = scene(ball_missing=[(1200, 1215), (2150, 2160)])
    out = label_diff.diff(codes_of(REFERENCE), codes_of(NEW), inputs)
    changed = 30 + 30 + 20 + 100
    assert abs(out["changed_share"] - changed / 1200) < 1e-12, "changed_share is {}".format(out["changed_share"])
    assert abs(out["changed_in_gaps_share"] - (15 + 10) / changed) < 1e-12, \
        "changed_in_gaps_share is {}".format(out["changed_in_gaps_share"])
    assert out["d_share"] == [(100 + 50 + 200) / 1200, (100 + 50 + 200) / 1200], "d_share is {}".format(out["d_share"])
    assert out["x_share"] == [0.0, 20 / 1200], "x_share is {}".format(out["x_share"])
    # Possessions, X filled: reference half 1 H A H = 3, half 2 A A = 2; new half 1 H A H = 3 (the X after the A is
    # filled with A), half 2 A H A = 3.
    assert out["possessions"] == {1: [3, 3], 2: [2, 3]}, "possessions are {}".format(out["possessions"])
    onsets = out["d_onsets"]
    # reference onsets 1200, 2100, 2400; new 1230, 2100, 2400: one moved, by +30
    assert (onsets["ref"], onsets["new"], onsets["moved"], onsets["median_move"]) == (3, 3, 1, 30.0), \
        "d_onsets are {}".format(onsets)


def test_label_diff_moves_are_matched_within_50_frames_and_signed():
    # reference onsets 1100, 1300, 1500; new onsets 1090 (-10), 1370 (70 away: lost), 1520 (+20)
    ref = codes_of([(1000, 1100, H), (1100, 1150, D), (1150, 1300, A), (1300, 1350, D), (1350, 1500, H),
                    (1500, 1550, D), (1550, 1600, A)])
    new = codes_of([(1000, 1090, H), (1090, 1150, D), (1150, 1370, A), (1370, 1400, D), (1400, 1520, H),
                    (1520, 1560, D), (1560, 1600, A)])
    onsets = label_diff.diff(ref, new, scene())["d_onsets"]
    assert (onsets["ref"], onsets["new"], onsets["moved"]) == (3, 3, 2), "d_onsets are {}".format(onsets)
    assert onsets["median_move"] == 5.0, "the median of -10 and +20 is 5, got {}".format(onsets["median_move"])
    same = label_diff.diff(ref, ref, scene())
    assert same["d_onsets"]["moved"] == 0 and same["d_onsets"]["median_move"] is None, "no move on equal codes"
    assert same["changed_share"] == 0.0 and same["changed_in_gaps_share"] == 0.0, "nothing changed"


def test_label_diff_refuses_codes_of_the_wrong_length():
    short = codes_of(NEW)[:-1]
    expect_error(ValueError, lambda: label_diff.diff(codes_of(REFERENCE), short, scene()),
                 "codes must have one code per frame")


def test_label_diff_prints_aggregates_only():
    inputs = scene(ball_missing=[(1200, 1215)])
    out = label_diff.diff(codes_of(REFERENCE), codes_of(NEW), inputs)
    text = "\n".join(label_diff.render("new.toml", out))
    numbers = [token for token in text.replace("\n", " ").replace(",", " ").split() if token.isdigit()]
    scene_frames = {str(frame) for frame in range(1000, N)}
    assert not scene_frames & set(numbers), "a frame of the scene is printed: {}".format(scene_frames & set(numbers))
    assert "moved" in text and "possessions" in text, "the figures are printed"


def test_label_diff_counts_as_the_scorer_does_with_x_filled():
    # An X inside a run does not split it: H X H is one possession and D X D one dead ball, as pm.fill_abstentions has
    # it. The raw shares still count the X as a label.
    ref = codes_of([(1000, 1300, H), (1300, 1400, D), (1400, 1600, A), (2000, 2600, H)])
    new = codes_of([(1000, 1100, H), (1100, 1120, X), (1120, 1300, H), (1300, 1340, D), (1340, 1360, X),
                    (1360, 1400, D), (1400, 1600, A), (2000, 2600, H)])
    out = label_diff.diff(ref, new, scene())
    assert out["possessions"] == {1: [2, 2], 2: [1, 1]}, "possessions are {}".format(out["possessions"])
    onsets = out["d_onsets"]
    assert (onsets["ref"], onsets["new"], onsets["moved"]) == (1, 1, 0), "D X D is one onset: {}".format(onsets)
    assert out["x_share"] == [0.0, 40 / 1200], "x_share stays raw: {}".format(out["x_share"])
    assert abs(out["changed_share"] - 40 / 1200) < 1e-12, "changed_share stays raw: {}".format(out["changed_share"])
