"""Checks of restarts (pipeline/benchmark/restarts.py): the anatomy of the stoppages, on the benchmark and on BAR-ATH.

restarts says how the dead-ball runs of the truth end (which team restarts, how often it is not the team that had the
ball before, how often the pipeline names the restarting team in the 2 s after) and, label-free, what the predicted
stoppages of BAR-ATH look like to the ball. A mistake in it would size P17's restart-team channel wrongly, so what is
held against it here is the restarts found on a hand-built truth (two stoppages, one cut off by an X run), the three
outcomes worked out on hand-built predictions (a tie is wrong, the window never reaches into the next block), the ball
shares on a hand-built scene, the pooling over matches and seeds against the same figures taken one match and seed at a
time, and that the pipeline is never handed the truth: the inputs it is called on go through the deep walk of
common.leaks.

The matches are the synthetic ones of checks/common.py (random flags and tracking, figures meaningless), under the
gaps profile and the synthetic Calibration.
"""

import numpy as np

from pipeline.benchmark import restarts, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import A, D, H, X, leaks, synthetic_calibration, synthetic_matches, yardstick_path
from pipeline.config import load_config
from pipeline.selftest import quiet

# Params
SEEDS = 2


def anatomy_of_synthetic(**more):
    return restarts.anatomy(yardstick_path(50), profile="gaps", seeds=SEEDS, matches=synthetic_matches(),
                            calibration=synthetic_calibration(), log=quiet, **more)


def test_restarts_true_restarts_finds_each_stoppage():
    """Each D run of a block followed by H or A is one restart; one followed by X, or by the block's end, is not."""
    codes = np.array([D] * 3 + [H] * 7 + [D] * 5 + [A] * 5 + [D] * 5 + [X] * 5, dtype=np.int8)   # block [0, 30)
    found = restarts.true_restarts(codes, [(1, 0, 30)])
    assert found["start"].tolist() == [0, 10] and found["restart"].tolist() == [3, 15]
    assert found["before"].tolist() == [-1, H] and found["after"].tolist() == [H, A]
    assert all(found[key].dtype == np.int64 for key in ("start", "restart", "before", "after"))
    # a stoppage that runs to the end of its block has no restart, and the next block's first frame is no "after"
    codes = np.array([H] * 4 + [D] * 4 + [A] * 4 + [D] * 4, dtype=np.int8)
    found = restarts.true_restarts(codes, [(1, 0, 12), (2, 12, 16)])
    assert found["start"].tolist() == [4] and found["before"].tolist() == [H] and found["after"].tolist() == [A]
    # the frame before a stoppage that opens a block is outside it: before is -1 even if the code there is H
    codes = np.array([H] * 4 + [D] * 4 + [A] * 4, dtype=np.int8)
    found = restarts.true_restarts(codes, [(1, 4, 12)])
    assert found["before"].tolist() == [-1] and found["after"].tolist() == [A]


def test_restarts_outcomes_three_ways():
    """Right is more of the restarting team than of the other in the window; a tie is wrong; no H/A is still D."""
    predicted = np.full(80, X, dtype=np.int8)
    predicted[5:9] = (H, H, A, H)            # right
    predicted[15:19] = (A, A, H, D)          # wrong: more of the other team
    predicted[25:29] = D                     # still D
    predicted[38:40] = D
    predicted[40:44] = H                     # the next block's: never counted for a restart of the first
    predicted[50:54] = (H, H, A, A)          # a tie
    found = {"start": np.array([3, 13, 23, 36, 48], dtype=np.int64),
             "restart": np.array([5, 15, 25, 38, 50], dtype=np.int64),
             "before": np.array([-1, A, H, -1, A], dtype=np.int64),
             "after": np.array([H, H, A, H, H], dtype=np.int64)}
    out = restarts.outcomes(predicted, found, [(1, 0, 40), (2, 40, 80)], window=4)
    assert out.tolist() == [0, 1, 2, 2, 1], out.tolist()
    assert out.dtype == np.int8
    assert restarts.OUTCOME_WINDOW == 50 and restarts.BALL_WINDOW == 50


def test_restarts_stoppage_ball_shares():
    """Two predicted stoppages; the ball missing on 3 of the last 4 frames before the first only."""
    codes = np.array([H] * 8 + [D] * 4 + [A] * 8 + [D] * 4 + [H] * 4 + [D] * 4, dtype=np.int8)   # block [0, 32)
    visible = np.ones(32, dtype=bool)
    visible[[8, 9, 10]] = False              # inside the first stoppage: not in any window
    visible[[4, 5, 6]] = False               # 3 of the 4 frames before the first stoppage
    visible[[12, 13]] = False                # 2 of 4 after the first: not more than half
    visible[[24, 25, 26]] = False            # 3 of the 4 frames after the second
    shares = restarts.stoppage_ball_shares(codes, visible, [(1, 0, 32)], window=4)
    assert shares["stoppages"] == 2 and shares["per_half"] == {1: 2}, shares
    assert shares["no_ball_before"] == 0.5 and shares["no_ball_after"] == 0.5, shares
    # a stoppage that opens its block has no frame before it: it is left out of that share
    shares = restarts.stoppage_ball_shares(codes[8:], visible[8:], [(1, 0, 24)], window=4)
    assert shares["stoppages"] == 2 and shares["no_ball_before"] == 0.0, shares
    # none
    shares = restarts.stoppage_ball_shares(np.full(10, H, dtype=np.int8), np.ones(10, dtype=bool), [(1, 0, 10)])
    assert shares["stoppages"] == 0 and shares["no_ball_before"] is None and shares["per_half"] == {}, shares


def test_restarts_anatomy_pools_over_matches_and_seeds():
    """Pooled restarts are the sum over matches and seeds of the truth's restarts; the shares are those of the sums."""
    matches = synthetic_matches()
    result = anatomy_of_synthetic()
    assert result["matches"] == sorted(matches) and result["seeds"] == [0, 1] and result["truth"] == "raw"
    total = flips = known = 0
    outcome_totals = np.zeros(3, dtype=int)
    calibration = synthetic_calibration()
    config = load_config(yardstick_path(50))
    for match_id, match in matches.items():
        codes, _ = truth.truth_codes(match)
        found = restarts.true_restarts(codes, match.graded)
        assert len(found["start"]) > 3, "a match with no restart proves nothing"
        row = result["per_match"][match_id]
        assert row["restarts"] == SEEDS * len(found["start"]), row
        assert row["median_stoppage_s"] == float(np.median((found["restart"] - found["start"]) / match.fps)), row
        counts = np.zeros(3, dtype=int)
        for seed in range(SEEDS):
            atd = benchmark_run.make_atd(match, "gaps", seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
            predicted, _ = benchmark_run.predict(config, inputs)
            counts += np.bincount(restarts.outcomes(predicted, found, match.graded), minlength=3)
        assert counts.sum() == row["restarts"]
        for name, count in zip(("right", "wrong", "still_d"), counts):
            assert row[name] == count / row["restarts"], (match_id, name, row)
        known_here = found["before"] != -1
        known += SEEDS * int(known_here.sum())
        flips += SEEDS * int((found["after"][known_here] != found["before"][known_here]).sum())
        total += row["restarts"]
        outcome_totals += counts
    pooled = result["pooled"]
    assert pooled["restarts"] == total and 0 < flips < known
    assert pooled["flip_share"] == flips / known, pooled
    for name, count in zip(("right", "wrong", "still_d"), outcome_totals):
        assert pooled[name] == count / total, (name, pooled)
    assert pooled["flip"]["restarts"] == flips and pooled["flip"]["restarts"] + pooled["no_flip"]["restarts"] == known
    for part in ("flip", "no_flip"):
        shares = [pooled[part][name] for name in ("right", "wrong", "still_d")]
        assert abs(sum(shares) - 1.0) < 1e-12, (part, shares)


def test_restarts_coarse_truth_keeps_the_stoppages():
    """The coarse truth never moves a stoppage, so the restarts are the raw truth's; only the teams may differ."""
    raw, coarse = anatomy_of_synthetic(), anatomy_of_synthetic(truth="coarse")
    assert coarse["truth"] == "coarse"
    assert raw["pooled"]["restarts"] == coarse["pooled"]["restarts"]


def test_restarts_never_pass_truth():
    """The inputs the pipeline is called on hold neither the match nor its truth, however deep leaks() looks."""
    matches = synthetic_matches()
    seen = []
    original = restarts.run.predict

    def spy(config, inputs):
        seen.append(inputs)
        return original(config, inputs)

    restarts.run.predict = spy
    try:
        anatomy_of_synthetic()
    finally:
        restarts.run.predict = original
    assert len(seen) == len(matches) * SEEDS, "one call per match and seed"
    for inputs, match in zip(seen, [m for m in matches.values() for _ in range(SEEDS)]):
        assert leaks(inputs, match) == [], leaks(inputs, match)
