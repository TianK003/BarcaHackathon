"""Checks of label_shift (pipeline/benchmark/label_shift.py): P22(a)'s feasibility on DFL and its checks on BAR-ATH.

The share of home, the gate at its edge (on numbers that are exact in binary, so that "exactly 0.75 times" is exact),
the calibration slope and the touch verdict are worked out here by other code. The feasibility runs on the planted
matches (a truth that the shape features solve) with the artifact trained on them: each match must be read by its own
fold and not by `full`, no truth may reach the inputs it serves, and the result holds numbers and strings, one per
seed at most. `bar_ath` runs on a synthetic Inputs with a stub `full` model. Nothing here reads data.
"""

import copy
import functools
import json
import os
import re
import tempfile

import numpy as np

from pipeline.benchmark import label_shift, run
from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark import truth
from pipeline.benchmark.checks.common import (A, D, FOLD_IDS, H, PLANTED_SEEDS, X, leaks, planted, strict_json,
                                              synthetic_calibration)
from pipeline.benchmark.run import BenchmarkError
from pipeline.evidence import shape_possession, trees
from pipeline.selftest import expect_error, play_inputs, shape_model, write_shape_model

# Params
PROFILE = "gaps"                              # the planted artifact is trained with it; the players are untouched


# -- helpers ----------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def planted_feasibility():
    """(result, the artifact's text): label_shift.feasibility on the planted matches and the artifact trained on them."""
    matches, artifact = planted()
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "shape.json")
        trainer.write(path, artifact)
        result = label_shift.feasibility(path, profile=PROFILE, seeds=PLANTED_SEEDS, matches=matches,
                                         calibration=synthetic_calibration())
    return result


def touch_scene(stretches=6, length=100):
    """Six players still but for a jump at each stretch, and a ball half a metre from a home player on the even
    stretches and from an away player on the odd ones: home is the more advanced team exactly while it has the ball."""
    teams = np.array([0, 0, 0, 1, 1, 1])
    base = np.array([[15.0, 20.0], [25.0, 34.0], [35.0, 48.0], [65.0, 20.0], [75.0, 34.0], [85.0, 48.0]])
    n = stretches * length
    shift = np.where((np.arange(n) // length) % 2 == 0, 8.0, -8.0)
    players = np.repeat(base[None], n, axis=0)
    players[:, :, 0] += shift[:, None]
    ball = players[np.arange(n), np.where(shift > 0, 1, 4)] + [0.5, 0.0]
    return players, teams, ball, [(1, 0, n)]


def bar_ath_inputs():
    players, teams, ball, graded = touch_scene()
    inputs = play_inputs(players, ball, teams=teams, graded=graded)
    inputs.game = "BAR-ATH"
    return inputs


# -- the pieces -------------------------------------------------------------------------------------------------------

def test_label_shift_home_share_exact():
    codes = np.full(60, X, dtype=np.int8)
    codes[10:16] = H                          # 6 H, 2 A, 5 D inside the blocks; the rest is outside and ignored
    codes[16:18] = A
    codes[18:23] = D
    codes[30:34] = H                          # 4 more H: 10 H against 2 A
    codes[0:5] = A                            # outside the blocks: not counted
    codes[50:60] = H
    graded = [(1, 10, 25), (2, 30, 40)]
    assert label_shift.home_share(codes, graded) == 10 / 12
    assert label_shift.home_share(np.full(60, D, dtype=np.int8), graded) is None
    assert label_shift.home_share(codes, [(1, 18, 25)]) is None, "only dead frames and abstentions in the block"
    only_away = np.full(60, A, dtype=np.int8)
    assert label_shift.home_share(only_away, graded) == 0.0


def test_label_shift_gate_is_the_advisors_rule():
    assert label_shift.GATE_FACTOR == 0.75 and label_shift.BAND == (0.39, 0.68) and label_shift.TOUCH_SLACK == 0.005
    # the numbers are exact in binary: a baseline of 0.25 and an error of 0.1875 = 0.75 * 0.25
    rows = [{"pi_true": 0.25, "pi_hat": [0.4375]}, {"pi_true": 0.75, "pi_hat": [0.5625]}]
    out = label_shift.gate(rows)
    assert out["error"] == 0.1875 and out["baseline"] == 0.25 and out["ratio"] == 0.75 and out["factor"] == 0.75
    assert out["passes"] is True, "an error of exactly 0.75 x the baseline passes"
    worse = [{"pi_true": 0.25, "pi_hat": [0.4375 + 1e-12]}, {"pi_true": 0.75, "pi_hat": [0.5625 - 1e-12]}]
    assert label_shift.gate(worse)["passes"] is False, "0.75 x the baseline + 1e-12 fails"
    assert sorted(out) == ["baseline", "error", "factor", "passes", "ratio"]
    # the mean is over seeds within a match first, then over matches: one seed of the first match, three of the second
    rows = [{"pi_true": 0.5, "pi_hat": [0.75]}, {"pi_true": 0.25, "pi_hat": [0.25, 0.25, 0.5]}]
    out = label_shift.gate(rows)
    assert out["error"] == (0.25 + (0.0 + 0.0 + 0.25) / 3) / 2, out
    assert out["baseline"] == (0.0 + 0.25) / 2
    # a baseline of 0 has no ratio, and nothing passes it unless the error is 0 too
    flat = label_shift.gate([{"pi_true": 0.5, "pi_hat": [0.5]}])
    assert flat["ratio"] is None and flat["passes"] is True
    assert label_shift.gate([{"pi_true": 0.5, "pi_hat": [0.6]}])["passes"] is False


def test_label_shift_calibration_slope():
    rng = np.random.default_rng([22, 3])
    n = 200_000
    calibrated = rng.normal(0.0, 2.0, n)
    y = rng.random(n) < trees.sigmoid(calibrated)
    assert abs(label_shift.calibration_slope(calibrated, y) - 1.0) < 0.03
    for scale in (0.5, 2.0):                                         # overconfident or underconfident by `scale`
        slope = label_shift.calibration_slope(scale * calibrated, y)
        assert abs(slope - 1.0 / scale) < 0.03, (scale, slope)
    assert label_shift.calibration_slope(calibrated, y) == label_shift.calibration_slope(calibrated.copy(), y.copy())
    assert label_shift.calibration_slope(np.zeros(10), np.ones(10)) is None, "no spread, no slope"


def test_label_shift_touch_verdict():
    assert label_shift.touch_verdict(0.80, 0.80) and label_shift.touch_verdict(0.80, 0.796)
    assert not label_shift.touch_verdict(0.80, 0.794) and label_shift.touch_verdict(0.80, 0.9)
    assert not label_shift.touch_verdict(None, 0.8) and not label_shift.touch_verdict(0.8, None)


# -- the feasibility --------------------------------------------------------------------------------------------------

def test_label_shift_feasibility_reads_each_matchs_fold_and_no_truth():
    matches, artifact = planted()
    calibration = synthetic_calibration()
    result = planted_feasibility()
    assert sorted(result["per_match"]) == list(FOLD_IDS)
    seen, real = [], shape_possession.logit_home

    def spy(inputs, model):
        seen.append(inputs)
        return real(inputs, model)
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "shape.json")
        trainer.write(path, artifact)
        shape_possession.logit_home = spy
        try:
            again = label_shift.feasibility(path, profile=PROFILE, seeds=PLANTED_SEEDS, matches=matches,
                                            calibration=calibration)
        finally:
            shape_possession.logit_home = real
        assert again["per_match"] == result["per_match"] and again["gate"] == result["gate"], "not deterministic"
        for match_id, match in matches.items():
            cell = result["per_match"][match_id]
            codes, _ = truth.truth_codes(match)
            frames = [frame for _, start, stop in match.graded for frame in range(start, stop)]
            home = sum(1 for frame in frames if codes[frame] == H)
            away = sum(1 for frame in frames if codes[frame] == A)
            assert cell["pi_true"] == home / (home + away) and cell["alive_frames"] == home + away
            assert 0.0 < cell["pi_true"] < 1.0
            differs = False
            for seed, got in enumerate(cell["seeds"]):
                inputs = trainer.served_inputs(match, PROFILE, seed, calibration, None)
                assert inputs.game == match_id
                fold = shape_possession.em_prior(real(inputs, path))
                assert got == {"pi_hat": fold["prior_home"], "iterations": fold["iterations"],
                               "converged": fold["converged"], "frames": fold["frames"]}, (match_id, seed)
                inputs.game = "BAR-ATH"                                  # a game with no fold: the `full` model
                full = shape_possession.em_prior(real(inputs, path))
                differs |= full["prior_home"] != got["pi_hat"]
            assert differs, "{}: the fold and `full` give the same share, so the check has no teeth".format(match_id)
            hats = [s["pi_hat"] for s in cell["seeds"]]
            assert cell["abs_error"] == sum(abs(p - cell["pi_true"]) for p in hats) / len(hats)
            assert cell["abs_baseline"] == abs(0.5 - cell["pi_true"])
            assert cell["slope"] is not None and cell["slope"] > 0, "the planted truth is learnable: a positive slope"
            assert cell["pi_hat_alive"] is not None
        assert len(seen) == len(matches) * PLANTED_SEEDS and all(inputs.game in matches for inputs in seen)
        for inputs in seen:
            assert all(leaks(inputs, other) == [] for other in matches.values()), "the truth reached the inputs"
        # a model without one fold is refused, naming the match
        for held_out in FOLD_IDS:
            thinner = copy.deepcopy(artifact)
            del thinner["folds"][held_out]
            short = os.path.join(root, "short.json")
            trainer.write(short, thinner)
            expect_error(BenchmarkError, lambda: label_shift.feasibility(
                short, profile=PROFILE, seeds=PLANTED_SEEDS, matches=matches, calibration=calibration),
                held_out, "label_shift", "fold")
        expect_error(BenchmarkError, lambda: label_shift.feasibility(
            os.path.join(root, "nope.json"), profile=PROFILE, seeds=1, matches=matches, calibration=calibration),
            "label_shift", "nope.json")
        expect_error(BenchmarkError, lambda: label_shift.feasibility(
            path, profile="nonsense", seeds=1, matches=matches, calibration=calibration), "profile")


def test_label_shift_feasibility_is_aggregates():
    result = planted_feasibility()
    text = json.dumps(run.plain(result), allow_nan=False)
    assert strict_json(text) == run.plain(result)

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, "{}.{}".format(where, key))
        elif isinstance(value, list):
            assert where.endswith(".seeds") and len(value) == result["seeds"], "{} is a list of {}".format(
                where, len(value))
            for item in value:
                walk(item, where + "[]")
        else:
            assert value is None or isinstance(value, (bool, int, float, str)), (where, type(value))
    walk(result, "result")
    assert result["seeds"] == PLANTED_SEEDS and result["profile"] == PROFILE
    assert set(result["gate"]) == {"error", "baseline", "ratio", "factor", "passes"}
    rows = [{"pi_true": cell["pi_true"], "pi_hat": [s["pi_hat"] for s in cell["seeds"]]}
            for cell in result["per_match"].values()]
    assert result["gate"] == label_shift.gate(rows)


# -- BAR-ATH ----------------------------------------------------------------------------------------------------------

def test_label_shift_bar_ath_on_synthetic_inputs():
    inputs = bar_ath_inputs()
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root, name="full.json", full=shape_model(0.0, -1.0, 2.0))
        out = label_shift.bar_ath(0.5, model=path, inputs=inputs)
        assert sorted(out) == ["band", "converged", "frames", "inside_band", "iterations", "prior_home", "touch"]
        assert sorted(out["touch"]) == ["difference", "passes", "shifted", "unshifted"]
        found = shape_possession.em_prior(shape_possession.logit_home(inputs, path))
        assert out["prior_home"] == found["prior_home"] and out["frames"] == found["frames"] > 500
        assert out["band"] == list(label_shift.BAND) and out["inside_band"] is True, "the scene is balanced"
        assert out["converged"] is True and out["iterations"] == found["iterations"]
        plain = trainer.touch_agreement(inputs, model=path)
        assert out["touch"]["unshifted"] == plain and plain["frames"] > 0
        shifted = trainer.touch_agreement(inputs, model=path, prior="em", prior_strength=0.5)
        assert out["touch"]["shifted"] == shifted
        assert out["touch"]["difference"] == shifted["agreement"] - plain["agreement"]
        assert out["touch"]["passes"] is (shifted["agreement"] >= plain["agreement"] - label_shift.TOUCH_SLACK)
        # the band, from a share that EM is made to find: its edges are inside, outside is not
        real = shape_possession.em_prior
        try:
            for share, inside in ((0.2, False), (0.39, True), (0.68, True), (0.7, False), (0.55, True)):
                shape_possession.em_prior = lambda logits, share=share, **kw: {
                    "prior_home": share, "iterations": 3, "converged": True, "frames": 9}
                assert label_shift.bar_ath(1.0, model=path, inputs=inputs)["inside_band"] is inside, share
        finally:
            shape_possession.em_prior = real
        # the slack, from touch agreements that are made up: shifted calls are the ones with a prior
        real_touch = trainer.touch_agreement
        try:
            for delta, passes in ((0.0, True), (-0.004, True), (-0.006, False), (0.05, True)):
                trainer.touch_agreement = lambda inputs, model=None, delta=delta, **kw: {
                    "frames": 100, "agreement": 0.8 + (delta if kw else 0.0)}
                out = label_shift.bar_ath(0.5, model=path, inputs=inputs)
                assert out["touch"]["passes"] is passes, delta
            trainer.touch_agreement = lambda inputs, model=None, **kw: {"frames": 0, "agreement": None}
            out = label_shift.bar_ath(0.5, model=path, inputs=inputs)
            assert out["touch"]["passes"] is False and out["touch"]["difference"] is None
        finally:
            trainer.touch_agreement = real_touch
        json.dumps(run.plain(label_shift.bar_ath(0.5, model=path, inputs=inputs)), allow_nan=False)


def test_label_shift_bar_ath_refuses_an_extreme_share():
    # a model that says home, with certainty, on every frame (the symmetrised logit of a real model cannot, so it is
    # stubbed): no share can be shifted by, and the sentence is the source's, with the tool's name in front
    inputs = bar_ath_inputs()
    real = shape_possession.logit_home
    shape_possession.logit_home = lambda inputs, model: np.full(len(inputs.atd.ball_xy), 60.0)
    try:
        expect_error(BenchmarkError, lambda: label_shift.bar_ath(0.5, model="unused.json", inputs=inputs),
                     "label_shift", "shape_possession", "0 or 1")
    finally:
        shape_possession.logit_home = real


def test_label_shift_prints_aggregates_only():
    result = planted_feasibility()
    lines = label_shift.render_feasibility(result)
    assert len(lines) == len(FOLD_IDS) + 2 and all(len(line) <= 140 for line in lines)
    for line in lines:
        assert len(re.findall(r"\d+\.\d+", line)) <= PLANTED_SEEDS + 5, line
        assert not re.search(r"\d{4,}", line.replace(os.path.basename(result["model"]), "")), line
    assert all(match_id in "\n".join(lines) for match_id in FOLD_IDS) and ("PASS" in lines[-1] or "FAIL" in lines[-1])
    inputs = bar_ath_inputs()
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root, name="full.json", full=shape_model(0.0, -1.0, 2.0))
        shown = label_shift.render_bar_ath(label_shift.bar_ath(0.5, model=path, inputs=inputs))
    assert len(shown) == 2 and all(len(re.findall(r"\d+\.\d+", line)) <= 6 for line in shown), shown
    assert "inside" in shown[0] and ("PASS" in shown[1] or "FAIL" in shown[1])


def test_label_shift_command_line():
    parser = label_shift.build_parser()
    args = parser.parse_args(["bar-ath", "--strength", "0.5"])
    assert args.command == "bar-ath" and args.strength == 0.5 and args.model == label_shift.DEFAULT_MODEL_PATH
    args = parser.parse_args(["feasibility"])
    assert (args.command, args.profile, args.seeds) == ("feasibility", run.DEFAULT_PROFILE, run.DEFAULT_SEEDS)
    for bad in (["bar-ath"], ["feasibility", "--strength", "1"], []):
        try:
            with open(os.devnull, "w") as quiet, __import__("contextlib").redirect_stderr(quiet):
                parser.parse_args(bad)
        except SystemExit as stop:
            assert stop.code == 2, bad
        else:
            raise AssertionError("{} should be refused".format(bad))
