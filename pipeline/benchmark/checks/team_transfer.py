"""Checks of P19's transfer check (pipeline/benchmark/team_transfer.py): P19's agreement with Smart Tagging's team votes.

* the agreement counts a silent frame as wrong and names it (test_team_transfer_agreement_counts_silent_as_wrong);
* the margin is an integer test: exactly -0.01 passes and -0.0101 fails (test_team_transfer_margin_is_integer);
* the frames judged are the graded ones with the ball visible and a vote of 0 or 1, worked out here by a plain loop on a
  synthetic tag list, and nothing outside them is counted (test_team_transfer_uses_only_voted_visible_graded_frames);
* what is printed and written is aggregates: counts, shares, a verdict, and the command exits 0 either way
  (test_team_transfer_prints_aggregates_only).

Synthetic only: a stand-in Inputs with hand-built tracking and tags, a stump model and a two-line config.
"""

import json
import os
import tempfile

import numpy as np

from pipeline.benchmark import team_transfer as transfer
from pipeline.benchmark.run import BenchmarkError
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics, smart_tagging, team_features, team_model
from pipeline.selftest import (expect_error, play_inputs, tagged, team_model_stub, team_scene, write_team_model,
                               write_text)

CONFIG = """
[[evidence]]
kind = "controlled_touch"
radius = 3.0
max_rel_speed = inf
min_frames = 1
confidence = 0.9
hold = 0.6

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 32
min_possession = 35
min_dead = 25
"""
TAGS = [("BUILD UP", 5, 60, "Barcelona"), ("BUILD UP", 100, 160, "Atletic Club"), ("SET PIECES", 170, 200, "Barcelona")]


def scene_inputs(seed=0):
    """A stand-in Inputs with a ball that rides on a slot, two graded blocks and a hand-built tag list."""
    players, ball, teams, graded = team_scene(seed)
    n = len(players)
    inputs = play_inputs(players, ball, teams=teams, graded=graded)
    inputs.smart_tagging = tagged(TAGS, graded=graded, n_frames=n).smart_tagging
    inputs.n_frames, inputs.game = n, "BAR-ATH"
    return inputs


def test_team_transfer_agreement_counts_silent_as_wrong():
    home = np.array([0.9, 0.2, np.nan, 0.5, 0.4, 0.8, np.nan])
    vote = np.array([1, 0, 1, 1, 1, 0, 0])
    frames = np.array([True, True, True, True, True, True, False])
    cell = transfer.agreement(home, vote, frames)
    # right: 0.9 home / vote home, 0.2 away / vote away, 0.5 names home (>= 0.5) / vote home; wrong: NaN, 0.4, 0.8
    assert cell == {"frames": 6, "right": 3, "silent": 1, "agreement": 0.5}, cell
    assert transfer.agreement(home, vote, np.zeros(7, dtype=bool)) == {"frames": 0, "right": 0, "silent": 0,
                                                                       "agreement": None}
    everything_silent = transfer.agreement(np.full(7, np.nan), vote, np.ones(7, dtype=bool))
    assert everything_silent["right"] == 0 and everything_silent["silent"] == 7, "silent is wrong, however the votes go"
    expect_error(ValueError, lambda: transfer.agreement(home, vote[:-1], frames), "team_transfer", "one length")


def test_team_transfer_margin_is_integer():
    def cells(frames, gap):
        return {"frames": frames, "right": 5000}, {"frames": frames, "right": 5000 + gap}
    assert transfer.passes(*cells(10000, 100)) is True, "exactly -0.0100 passes"
    assert transfer.passes(*cells(10000, 101)) is False, "-0.0101 fails"
    assert transfer.passes(*cells(100, 1)) is True and transfer.passes(*cells(99, 1)) is False
    assert transfer.passes(*cells(1000, -50)) is True, "better than the rule passes"
    assert transfer.passes(*cells(1000, 0)) is True
    expect_error(ValueError, lambda: transfer.passes({"frames": 10, "right": 1}, {"frames": 11, "right": 1}),
                 "same frames")
    expect_error(ValueError, lambda: transfer.passes(*cells(10, 1), margin=0.03), "margin")
    expect_error(ValueError, lambda: transfer.passes(*cells(10, 1), margin=True), "margin")


def test_team_transfer_uses_only_voted_visible_graded_frames():
    inputs = scene_inputs()
    votes = smart_tagging.votes(inputs)
    assert (votes[12:60] == 1).all() and (votes[100:130] == 0).all() and (votes[150:160] == 0).all()
    assert (votes[:12] == -1).all() and (votes[60:100] == -1).all(), "no tag, no vote; a tag is cut at the blocks"
    assert (votes[170:200] == -1).all(), "a set piece does not vote"
    with tempfile.TemporaryDirectory() as root:
        model = write_team_model(root, full=team_model_stub(0.0, -1.0, 1.0))
        config = write_text(root, "rule.toml", CONFIG)
        result = transfer.measure(inputs, model, config)
        run_context = RunContext(inputs=inputs, seed=0)
        home = team_model.compute(run_context, model=model).home
        rule = controlled_touch.compute(run_context, radius=3.0, max_rel_speed=float("inf"), min_frames=1,
                                        confidence=0.9, hold=0.6).home
    visible = kinematics.ball_track(inputs).visible
    judged = [t for t in range(len(votes)) if visible[t] and votes[t] in (0, 1)]
    assert 0 < len(judged) < visible.sum() and all(any(a <= t < b for _, a, b in inputs.graded) for t in judged)
    assert result["frames"] == len(judged) == result["p19"]["frames"] == result["rule"]["frames"]
    for name, source in (("p19", home), ("rule", rule)):
        right = sum(1 for t in judged if not np.isnan(source[t]) and (source[t] >= 0.5) == (votes[t] == 1))
        silent = sum(1 for t in judged if np.isnan(source[t]))
        assert (result[name]["right"], result[name]["silent"]) == (right, silent), name
    assert result["visible_frames"] == int(visible.sum()) and result["gain_frames"] == (
        result["p19"]["right"] - result["rule"]["right"])
    assert result["passes"] == transfer.passes(result["p19"], result["rule"])
    shares = result["nan_shares"]
    assert list(shares) == list(team_features.FEATURE_NAMES) and all(0.0 <= v <= 1.0 for v in shares.values())
    X = team_features.features(inputs)[1]
    assert shares["ball_line_m"] == 0.0 and shares["h_ball_d1"] == float(np.isnan(X[visible][:, 0]).mean())
    # the rule is the config's: a config without a touch entry is refused
    with tempfile.TemporaryDirectory() as root:
        model = write_team_model(root)
        bare = write_text(root, "bare.toml", CONFIG.replace("controlled_touch", "nearest_player").replace(
            "radius = 3.0\n", "").replace("max_rel_speed = inf\n", "").replace("min_frames = 1\n", "").replace(
            "confidence = 0.9\n", "").replace("hold = 0.6\n", ""))
        expect_error(BenchmarkError, lambda: transfer.measure(inputs, model, bare), "controlled_touch")


def test_team_transfer_prints_aggregates_only():
    import contextlib
    import io
    inputs = scene_inputs(1)
    with tempfile.TemporaryDirectory() as root:
        model = write_team_model(root, full=team_model_stub(0.0, -1.0, 1.0))
        config = write_text(root, "rule.toml", CONFIG)
        result = transfer.measure(inputs, model, config)
        lines = transfer.render(result)
        text = "\n".join(lines)
        json.dumps(result, allow_nan=False)

        def leaves(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from leaves(item)
            else:
                yield value
        assert all(leaf is None or isinstance(leaf, (bool, int, float)) for leaf in leaves(result)), "no list, no array"
        assert "PASSES" in text or "FAILS" in text and "agreement" in text and len(lines) < 12
        # the command: a stand-in for the data load, exit 0 whatever the verdict, aggregates written
        real, buffer = transfer.load_inputs, io.StringIO()
        inputs.data_dir = root
        transfer.load_inputs = lambda data_dir: inputs
        try:
            with contextlib.redirect_stdout(buffer):
                code = transfer.main(["--model", model, "--config", config, "--data-dir", root])
        finally:
            transfer.load_inputs = real
        assert code == 0 and "wrote" in buffer.getvalue()
        path = os.path.join(root, "runs", "r021", "transfer.json")
        with open(path, encoding="utf-8") as handle:
            written = json.load(handle)
        assert written["frames"] == result["frames"] and written["p19"] == result["p19"]
        assert not any(isinstance(value, list) for value in written.values()), "no per-frame list in the file"
        # a failing verdict still exits 0: it is in the output
        failing = dict(result, passes=False)
        assert "FAILS" in "\n".join(transfer.render(failing))
        buffer = io.StringIO()
        transfer.load_inputs = lambda data_dir: inputs
        try:
            with contextlib.redirect_stdout(buffer):
                missing = transfer.main(["--model", os.path.join(root, "absent.json"), "--config", config,
                                         "--data-dir", root])
        finally:
            transfer.load_inputs = real
        assert missing == 1 and "FAILED" in buffer.getvalue() and "absent.json" in buffer.getvalue()
