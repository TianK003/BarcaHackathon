"""Checks of wrong_team (pipeline/benchmark/wrong_team.py): P21 step 0, wrong-team slots at the ball against all frames.

Every count is worked out here on hand-built masks and scenes: the over-11 and surplus rules, the candidate frames as
`controlled_touch`'s kept frames without its duel test, the ratio at the rule's edge (from counts that make 1.5 exact),
the constants held equal to `main.toml`'s, D3's onset ratio against a plain loop, and the printout, which holds counts
and shares and no frame. No check reads data.
"""

import os
import re

import numpy as np

from pipeline.benchmark import wrong_team
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch
from pipeline.selftest import play_inputs, walkers


# -- the rules --------------------------------------------------------------------------------------------------------

def present_with(counts):
    """(frames, 32) present mask with (home, away) slots present per frame; slots 0-15 are home, 16-31 away."""
    present = np.zeros((len(counts), 32), dtype=bool)
    for frame, (home, away) in enumerate(counts):
        present[frame, :home] = True
        present[frame, 16:16 + away] = True
    return present, (np.arange(32) >= 16).astype(np.int8)


def test_wrong_team_over_eleven_and_surplus_exact():
    present, teams = present_with([(12, 10), (11, 11), (12, 11), (13, 9), (10, 12), (11, 12)])
    assert wrong_team.over_eleven(present, teams).tolist() == [True, False, True, True, True, True]
    assert wrong_team.surplus(present, teams).tolist() == [True, False, False, True, True, False]
    assert wrong_team.TEAM_SIZE == 11


# -- the candidates ---------------------------------------------------------------------------------------------------

def duel_scene():
    """A home player 0.5 m from a ball that all of them carry along at 3 m/s; an away player 0.8 m off it on frames
    40-80 (a duel) and 6 m off otherwise; the ball is lost on 100-110; the graded block is 10-140 of 150."""
    n = 150
    starts = [[20.0, 30.0], [20.0, 35.0], [60.0, 10.0], [60.0, 50.0]]
    players = walkers(n, 3.0, starts)
    ball = players[:, 0] + [0.5, 0.0]
    players[40:80, 1] = ball[40:80] + [0.0, 0.8]
    players[:, 1][np.r_[0:40, 80:n]] = ball[np.r_[0:40, 80:n]] + [0.0, 6.0]
    ball[100:110] = np.nan
    return play_inputs(players, ball, teams=np.array([0, 1, 0, 1]), graded=[(1, 10, 140)])


def test_wrong_team_candidates_are_the_touch_rule_without_the_duel_test():
    inputs = duel_scene()
    kept = ~np.isnan(controlled_touch.compute(
        RunContext(inputs=inputs, seed=0), radius=1.5, max_rel_speed=6.0, min_frames=1, confidence=0.99, hold=0.5,
        duel_radius=1.0).home)
    with_duel = wrong_team.candidates(inputs, duel_radius=1.0)
    assert with_duel.sum() > 50 and np.array_equal(with_duel, kept), "the duel test is controlled_touch's"
    plain = wrong_team.candidates(inputs)
    assert plain.sum() > with_duel.sum(), "the scene must hold duels, or the check has no teeth"
    assert np.array_equal(plain & with_duel, with_duel), "without the duel test is a superset"
    assert not plain[:10].any() and not plain[140:].any() and not plain[100:110].any(), "graded, ball seen"
    assert plain[40:80].all(), "the duel frames are the ones that the test removes"
    assert not with_duel[45:75].any()
    assert wrong_team.candidates(inputs, radius=0.4).sum() == 0, "the nearest slot is 0.5 m away"
    assert wrong_team.candidates(inputs, max_rel_speed=1.0).sum() == plain.sum(), "they all move together"


# -- the ratio --------------------------------------------------------------------------------------------------------

def test_wrong_team_ratio_exact():
    n = 1_100
    graded = np.arange(n) < 1_000                              # 1,000 graded frames, then 100 outside them
    candidate = np.zeros(n, dtype=bool)
    candidate[100:200] = True                                  # 100 candidates
    over = np.zeros(n, dtype=bool)
    over[100:130] = True                                       # 30 of them over 11
    over[300:320] = True                                       # 20 more: 50 over 11 in all
    over[1_020:1_090] = True                                   # outside the graded blocks: never counted
    candidate[1_020:1_030] = True
    out = wrong_team.ratio(over, candidate, graded)
    assert (out["graded_frames"], out["candidate_frames"], out["over_graded_frames"],
            out["over_candidate_frames"]) == (1_000, 100, 50, 30)
    assert out["share_candidates"] == 0.3 and out["share_graded"] == 0.05
    assert out["ratio"] == 6.0 and out["verdict"] == "report only", out
    # the rule's edge, from counts that make 1.5 exact: 15% of 100,000 candidates against 10% of a million frames
    big = 1_000_000
    everything = np.ones(big, dtype=bool)
    candidate = np.arange(big) < 100_000
    over = np.zeros(big, dtype=bool)
    over[:15_000] = True                                       # 15,000 over frames, all among the candidates
    over[500_000:585_000] = True                               # 100,000 over in all
    at = wrong_team.ratio(over, candidate, everything)
    assert at["over_graded_frames"] == 100_000 and at["ratio"] == 1.5 and at["verdict"] == "report only", at
    over[14_999], over[600_000] = False, True                  # one over frame moves from a candidate to outside
    below = wrong_team.ratio(over, candidate, everything)
    assert below["over_candidate_frames"] == 14_999 and below["over_graded_frames"] == 100_000
    assert below["ratio"] == 1.4999 and below["verdict"] == "P21 dropped", below
    assert wrong_team.ratio(np.zeros(big, dtype=bool), candidate, everything)["verdict"] == "undefined"
    assert wrong_team.ratio(over, np.zeros(big, dtype=bool), everything)["ratio"] is None
    assert wrong_team.RULE == 1.5


def test_wrong_team_constants_are_main_tomls():
    from pipeline.config import load_config
    main = load_config(os.path.join(wrong_team.ROOT, "pipeline", "configs", "main.toml"))
    touch = [c for c in main.evidence if c.kind == "controlled_touch"]
    assert len(touch) == 1
    assert touch[0].params["radius"] == wrong_team.RADIUS == 1.5
    assert touch[0].params["max_rel_speed"] == wrong_team.MAX_REL_SPEED == 6.0
    assert wrong_team.ONSET_WINDOW_S == 1.0


# -- D3's onset ratio and the measure ---------------------------------------------------------------------------------

def test_wrong_team_onset_ratio_exact():
    n = 200
    blocks = [(0, 100), (100, 200)]
    over = np.zeros(n, dtype=bool)
    for start, stop in ((0, 8), (20, 30), (60, 62), (100, 105), (140, 150), (190, 200)):
        over[start:stop] = True               # (0, 8) and (100, 105) begin on a block's first frame: carried in
    candidate = np.zeros(n, dtype=bool)
    candidate[[24, 70, 148, 160]] = True
    got = wrong_team.onset_ratio(over, candidate, blocks, 5)
    # plain loops: onsets at 20, 60, 140, 190; near a candidate within 5 frames of the same block: 20 (24), 140 (148: 8
    # away, no), 60 (70: 10 away, no), 190 (none)
    near = np.zeros(n, dtype=bool)
    for start, stop in blocks:
        for frame in range(start, stop):
            near[frame] = any(candidate[other] for other in range(max(start, frame - 5), min(stop, frame + 6)))
    onsets = [20, 60, 140, 190]
    assert got["onsets"] == 4 and got["onsets_near"] == sum(near[f] for f in onsets) == 1
    assert got["share_onsets_near"] == 0.25 and got["share_graded_near"] == near.sum() / n
    assert got["ratio"] == 0.25 / (near.sum() / n)
    assert wrong_team.onset_ratio(np.zeros(n, dtype=bool), candidate, blocks, 5)["ratio"] is None
    # a candidate in the other block is not near, whatever the distance
    across = np.zeros(n, dtype=bool)
    across[101] = True
    assert wrong_team.onset_ratio(over, across, [(0, 100), (100, 200)], 5)["onsets_near"] == 0


def synthetic_measure_inputs(n=600, seed=0):
    """Twenty-three slots (twelve home) wandering about a ball they hover near; some slots drop out; two blocks."""
    rng = np.random.default_rng([21, seed])
    teams = (np.arange(23) >= 12).astype(np.int8)
    ball = np.cumsum(rng.normal(size=(n, 2)) * 0.1, axis=0) + [50.0, 34.0]
    players = ball[:, None, :] + rng.normal(size=(n, 23, 2)) * 2.0
    players[rng.random((n, 23)) < 0.15] = np.nan
    ball[rng.random(n) < 0.1] = np.nan
    inputs = play_inputs(players, ball, teams=teams, graded=[(1, 20, 300), (2, 320, 580)])
    return inputs


def test_wrong_team_measure_against_plain_loops():
    inputs = synthetic_measure_inputs()
    out = wrong_team.measure(inputs)
    assert sorted(out) == ["onset", "ratio", "surplus"]
    present = ~np.isnan(inputs.atd.players_xy[:, :, 0])
    graded = [frame for _, start, stop in inputs.graded for frame in range(start, stop)]
    teams = inputs.atd.slot_team
    over = [frame for frame in graded if (present[frame] & (teams == 0)).sum() > 11
            or (present[frame] & (teams == 1)).sum() > 11]
    cell = out["ratio"]
    assert cell["graded_frames"] == len(graded) == 540 and cell["over_graded_frames"] == len(over) > 0
    assert 0 < cell["candidate_frames"] < 540 and cell["verdict"] in ("P21 dropped", "report only")
    candidate = wrong_team.candidates(inputs)
    assert cell["over_candidate_frames"] == sum(1 for frame in over if candidate[frame])
    assert out["surplus"]["candidate_frames"] == cell["candidate_frames"]
    assert out["surplus"]["verdict"] == "report only", "only the over-11 ratio carries the rule's verdict"
    assert out["onset"]["onsets"] > 0


def test_wrong_team_prints_aggregates_only():
    lines = wrong_team.render(wrong_team.measure(synthetic_measure_inputs()))
    assert len(lines) == 3 and all(len(line) <= 160 for line in lines)
    for line in lines:
        assert not re.search(r"\[|\]|\(\d+, \d+\)", line), line
        assert len(re.findall(r"\d+\.\d+", line)) <= 4, line
    assert lines[0].startswith("ratio") and lines[1].startswith("surplus") and "report only" in lines[2]
    empty = {"ratio": wrong_team.ratio(np.zeros(5, bool), np.zeros(5, bool), np.zeros(5, bool))}
    empty["surplus"] = empty["ratio"]
    empty["onset"] = wrong_team.onset_ratio(np.zeros(5, bool), np.zeros(5, bool), [], 5)
    assert all("-" in line for line in wrong_team.render(empty)), "an undefined share prints as a dash"
