"""Checks of P34 (the version 2 shape model trained on both truths' labels): train_shape.py --labels, its rows'
weights and report, and P34's config (pipeline/benchmark/configs/p34/), p30_w030 but for the shape model.

Synthetic matches only; no model file is read, so the checks hold before the model of P34 is trained.
"""

import contextlib
import copy
import io
import json
import math
import os
import tempfile
import tomllib

import numpy as np

from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark import truth
from pipeline.benchmark.checks.common import A, D, H, random_match
from pipeline.contracts import RunContext
from pipeline.evidence import shape_features, shape_possession
from pipeline.selftest import expect_error

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGS = os.path.join(ROOT, "pipeline", "benchmark", "configs")
MODEL_V2_BOTH = "pipeline/models/shape_possession_v2both_hgb.json"
IDS = ("RND001", "RND002", "RND003")
STRIDE = 5


def matches3():
    return {match_id: random_match(match_id, seed=seed, rows=(300 + 20 * seed, 260)) for seed, match_id
            in enumerate(IDS)}


def scene():
    """(inputs, codes, coarse): a served random match and two code arrays that differ on known frames.

    Per graded block: the first 10 frames are D in both; frames 20 to 29 are A in the raw and H in the coarse truth (a
    short spell absorbed); frames 60 to 69 are H in the raw and A in the coarse truth; the rest is H in both.
    """
    match = random_match("RND001", seed=4, rows=(300, 260))
    inputs = trainer.served_inputs(match, "clean", 0, None, None)
    codes = np.full(inputs.n_frames, -1, dtype=np.int8)
    for _, start, stop in inputs.graded:
        codes[start:stop] = H
        codes[start:start + 10] = D
        codes[start + 20:start + 30] = A
    coarse = codes.copy()
    for _, start, stop in inputs.graded:
        coarse[start + 20:start + 30] = H                 # the short A spell absorbed
        coarse[start + 60:start + 70] = A                 # a place where the coarse truth names the other team
    return inputs, codes, coarse


def quietly(function, *arguments, **keywords):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*arguments, **keywords)


def test_p34_raw_is_today():
    matches = matches3()
    for version in (1, 2):
        today = trainer.train("clean", 1, 9, matches=matches, version=version)
        raw = trainer.train("clean", 1, 9, matches=matches, version=version, labels="raw")
        assert trainer.dumps(today) == trainer.dumps(raw), version
        assert "labels" not in raw
    # and the fit is called as it always was: no weight, whatever it is replaced by
    seen = []
    real = trainer.hgb_estimator

    def spy(*args, **keywords):
        seen.append((len(args), tuple(sorted(keywords))))
        return real(*args, **keywords)
    trainer.hgb_estimator = spy
    try:
        trainer.train("clean", 1, 9, matches=matches, labels="raw")
        raw_calls, seen[:] = list(seen), []
        trainer.train("clean", 1, 9, matches=matches, labels="both")
        both_calls = list(seen)
    finally:
        trainer.hgb_estimator = real
    assert raw_calls and set(raw_calls) == {(2, ())}, raw_calls
    assert both_calls and set(both_calls) == {(3, ())}, both_calls
    expect_error(ValueError, lambda: trainer.train("clean", 1, 9, matches=matches, labels="coarse"), "train_shape",
                 "labels", "raw, both", "coarse")


def test_p34_both_weights_and_duplicates():
    inputs, codes, coarse = scene()
    X_all = shape_features.features(inputs)[1]
    X, y, w = trainer.rows_both(inputs, codes, coarse, STRIDE)
    wanted_frames, wanted_y, wanted_w = [], [], []
    for _, start, stop in inputs.graded:
        for frame in range(start, stop, STRIDE):
            if codes[frame] not in (H, A):
                continue
            raw_home, coarse_home = codes[frame] == H, coarse[frame] == H
            if raw_home == coarse_home:
                wanted_frames.append(frame), wanted_y.append(int(raw_home)), wanted_w.append(1.0)
            else:
                wanted_frames += [frame, frame]
                wanted_y += [int(raw_home), int(coarse_home)]
                wanted_w += [0.5, 0.5]
    assert len(wanted_frames) > 60 and sum(1 for value in wanted_w if value == 0.5) >= 4
    assert y.dtype == np.int64 and w.dtype == np.float64
    assert y.tolist() == wanted_y and w.tolist() == wanted_w
    assert np.array_equal(X, X_all[wanted_frames], equal_nan=True)
    assert set(w.tolist()) == {0.5, 1.0}
    # the raw rows are the frames taken once each: both's frames are theirs, some twice
    raw_X, raw_y = trainer.rows(inputs, codes, STRIDE)
    assert len(raw_y) == len(set(wanted_frames)) and len(y) > len(raw_y)
    # the mirrors carry the weights with the labels flipped, rows first and unchanged
    both_X, both_y, both_w = trainer.augment(X, y, weight=w)
    n = len(y)
    assert both_X.shape == (2 * n, X.shape[1]) and np.array_equal(both_X[:n], X, equal_nan=True)
    assert np.array_equal(both_X[n:], shape_features.swap(X), equal_nan=True)
    assert both_y.tolist() == y.tolist() + (1 - y).tolist()
    assert both_w.tolist() == w.tolist() + w.tolist()
    two = trainer.augment(X, y)
    assert len(two) == 2 and np.array_equal(two[1], both_y)                 # no weight: the pair, as always
    expect_error(ValueError, lambda: trainer.augment(X, y, weight=w[:-1]), "train_shape", "weight")
    expect_error(ValueError, lambda: trainer.augment(X, y, weight=np.zeros(n)), "train_shape", "weight")


def test_p34_both_refuses_a_frame_alive_in_one_truth_only():
    inputs, codes, coarse = scene()
    start = inputs.graded[0][1]
    for bad_code, label in ((D, "dead in the coarse truth"), (-1, "unlabelled in the coarse truth")):
        broken = coarse.copy()
        broken[start + 40] = bad_code                        # an alive frame at a stride frame of the block
        expect_error(ValueError, lambda: trainer.rows_both(inputs, codes, broken, STRIDE), "train_shape", "alive")
    broken = codes.copy()
    broken[start + 5] = H                                    # alive in the raw truth, dead in the coarse one
    expect_error(ValueError, lambda: trainer.rows_both(inputs, broken, coarse, STRIDE), "train_shape", "alive")
    expect_error(ValueError, lambda: trainer.rows_both(inputs, codes, coarse[:-1], STRIDE), "train_shape", "coarse")
    # a frame the rows do not take (off the stride) may differ: only the frames taken are compared
    quiet = coarse.copy()
    quiet[start + 41] = D
    trainer.rows_both(inputs, codes, quiet, STRIDE)


def test_p34_both_artifact():
    matches = matches3()
    artifact = trainer.train("clean", 1, 9, matches=matches, version=2, labels="both")
    assert artifact["labels"] == "both" and artifact["feature_version"] == 2
    assert artifact["features"] == list(shape_features.feature_names(2))
    assert sorted(artifact) == sorted(["format", "kind", "feature_version", "features", "profile", "seeds", "stride",
                                       "provenance", "folds", "full", "labels"])
    assert sorted(artifact["folds"]) == list(IDS)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "shape.json")
        trainer.write(path, artifact)
        assert shape_possession.load(path) == artifact
    # the coarse codes are built per match and seed from the raw ones, with the served blocks and fps
    gathered = trainer.gather("clean", 1, 9, matches=matches, labels="both")
    for match_id, match in matches.items():
        X, y, w = gathered.data[match_id]
        codes, _ = truth.truth_codes(match)
        inputs = trainer.served_inputs(match, "clean", 0, None, None)
        rx, ry, rw = trainer.rows_both(inputs, codes, truth.coarse_codes(codes, inputs.graded, inputs.fps), 9)
        assert np.array_equal(X, rx, equal_nan=True) and np.array_equal(y, ry) and np.array_equal(w, rw), match_id
    # the raw gather keeps its pair
    assert all(len(cell) == 2 for cell in trainer.gather("clean", 1, 9, matches=matches).data.values())


def test_p34_report_has_the_coarse_cells():
    matches = matches3()
    artifact = trainer.train("clean", 1, 9, matches=matches)
    report = trainer.report(artifact, "clean", 1, matches=matches)
    json.dumps(report, allow_nan=False)
    # today's cells keep their keys
    for cell in list(report["per_match"].values()) + [report["pooled"]]:
        assert set(cell) == {"alive", "missing", "touch"}
        assert set(cell["alive"]) == {"frames", "silent", "log_loss", "accuracy"}
        assert set(cell["missing"]) == {"frames", "log_loss", "accuracy"}
    coarse = report["coarse"]
    assert sorted(coarse) == ["per_match", "pooled"] and sorted(coarse["per_match"]) == list(IDS)
    for name, cell in list(coarse["per_match"].items()) + [("pooled", coarse["pooled"])]:
        raw_cell = report["pooled"] if name == "pooled" else report["per_match"][name]
        assert set(cell) == {"alive", "missing"} and set(cell["alive"]) == {"frames", "log_loss", "accuracy"}
        assert set(cell["missing"]) == {"frames", "log_loss", "accuracy"}
        for part in ("alive", "missing"):
            assert cell[part]["frames"] == raw_cell[part]["frames"] > 0, (name, part)    # the same frames
    # the numbers, against plain loops over the frames, for the raw and the coarse truth
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "shape.json")
        trainer.write(path, artifact)
        for match_id, match in sorted(matches.items()):
            codes, _ = truth.truth_codes(match)
            inputs = trainer.served_inputs(match, "clean", 0, None, None)
            coarse_codes = truth.coarse_codes(codes, inputs.graded, inputs.fps)
            home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=path, scope="all").home
            sums = {}
            for label, truth_codes in (("raw", codes), ("coarse", coarse_codes)):
                for part in ("alive", "missing"):
                    sums[label, part] = [0, 0.0, 0]
                for frame in range(len(codes)):
                    if truth_codes[frame] not in (H, A) or math.isnan(home[frame]):
                        continue
                    p, truly_home = float(home[frame]), bool(truth_codes[frame] == H)
                    for part, take in (("alive", True), ("missing", not inputs.atd.ball_present[frame])):
                        if take:
                            cell = sums[label, part]
                            cell[0] += 1
                            cell[1] += -math.log(p if truly_home else 1.0 - p)
                            cell[2] += (p >= 0.5) == truly_home
            for (label, part), (frames, loss, right) in sums.items():
                cell = (report["per_match"][match_id] if label == "raw" else coarse["per_match"][match_id])[part]
                assert cell["frames"] == frames and frames > 0, (match_id, label, part)
                assert abs(cell["log_loss"] - loss / frames) <= 1e-9, (match_id, label, part)
                assert cell["accuracy"] == right / frames, (match_id, label, part)
    # the rendering keeps today's lines byte for byte and adds the coarse ones after them
    today = trainer.render_report({key: value for key, value in report.items() if key != "coarse"})
    lines = trainer.render_report(report)
    assert lines[:len(today)] == today and len(lines) > len(today)
    assert all(len(line) <= 120 for line in lines)
    assert all(line.startswith("coarse") for line in lines[len(today):])
    assert all(match_id in "\n".join(lines[len(today):]) for match_id in IDS)


def test_p34_command_line_takes_labels():
    parser = trainer.build_parser()
    assert parser.parse_args([]).labels == "raw" and parser.parse_args(["--labels", "both"]).labels == "both"
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--labels", "coarse"])
        except SystemExit:
            pass
        else:
            raise AssertionError("--labels coarse was accepted")
        try:
            trainer.main(["--report", "--labels", "both"])
        except SystemExit:
            pass
        else:
            raise AssertionError("--report with --labels was accepted")


def test_p34_config_is_p30_w030_but_for_the_model():
    path = os.path.join(CONFIGS, "p34", "p34_both.toml")
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    with open(os.path.join(CONFIGS, "p30", "p30_w030.toml"), "rb") as handle:
        reference = tomllib.load(handle)
    tables = [table for table in config["evidence"] if table["kind"] == "shape_possession"]
    assert len(tables) == 1 and tables[0]["model"] == MODEL_V2_BOTH
    assert isinstance(config["description"], str) and config["description"] != reference["description"]
    left, right = copy.deepcopy(config), copy.deepcopy(reference)
    for each in (left, right):
        each.pop("description")
        for table in each["evidence"]:
            if table["kind"] == "shape_possession":
                table.pop("model")
    assert left == right
