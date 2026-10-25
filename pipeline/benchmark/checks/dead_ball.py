"""Checks of P05's dead-ball trainer (pipeline/benchmark/train_dead_ball.py) and what it writes.

The trainer fits a gradient-boosted classifier (and a logistic reference) on the benchmark's degraded tracking and
writes it as JSON, which pipeline/evidence/dead_ball_model.py then runs with numpy alone. What is held against it here:

* the export is the model: the trees that go into the file, walked by numpy, give scikit-learn's own probabilities to
  1e-12, NaNs included (test_exported_trees_match_sklearn), and the same fit on one thread and on four writes the same
  bytes (test_training_is_deterministic_across_threads), which is what `--verify` relies on;
* the folds are what they say: the model of a match never saw a row of that match, `full` saw them all, and a row is
  the stride-th graded frame of a block from its start with 1 for "not dead"
  (test_trainer_folds_exclude_the_held_out_match);
* no truth reaches the features (test_trainer_never_passes_truth), the file is JSON that reads back exactly, LF and
  without NaN, and `--verify` fails on anything but the three source hashes;
* the logistic reference keeps its own bookkeeping (missing indicators, imputation, dropped columns) through a round
  trip, and the rule values round towards 0.5, never past it.

Everything runs on arrays and synthetic matches made here (checks/common.py): none reads data, and none of the numbers
printed is a figure of any match.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile

import numpy as np

from pipeline.benchmark import hazard
from pipeline.benchmark.calibration import digest as calibration_digest
from pipeline.benchmark import train_dead_ball as trainer
from pipeline.benchmark.checks.common import (leaks, random_match, synthetic_calibration, tracked_match,
                                              tracking_calibration)
from pipeline.evidence import dead_ball_model, play_features
from pipeline.selftest import expect_error

# Params
NAMES = play_features.FEATURE_NAMES
FOLD_IDS = ("RND001", "RND002", "RND003")


# -- helpers ----------------------------------------------------------------------------------------------------------

def random_rows(n=4000, seed=0, nan_share=0.15):
    """(X, y) of n rows over the 30 features, with NaN in a share of the cells and a label that depends on a few."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES)))
    X[:, 1] = np.abs(X[:, 1])                                   # ball_outside_m, one of the two monotone columns
    X[rng.random(X.shape) < nan_share] = np.nan
    score = (np.nan_to_num(X[:, 0]) - 1.5 * np.nan_to_num(X[:, 1]) + np.nan_to_num(X[:, 3])
             - np.nan_to_num(X[:, 19]) + rng.normal(size=n))
    return X, (score > 0).astype(np.int64)


def three_matches():
    return {match_id: random_match(match_id, seed=seed, rows=(300 + 20 * seed, 260)) for seed, match_id
            in enumerate(FOLD_IDS)}


def recording_fit(calls):
    """A stand-in for trainer.fit that records what it was given and returns a model that holds just that."""
    def fit(kind, X, y, names):
        calls.append((kind, np.array(X), np.array(y)))
        return {"prior_alive": float(np.mean(y)), "rows": int(len(y))}
    return fit


def small_artifact():
    """A real, small, logistic artifact with a provenance made by hand."""
    X, y = random_rows(600, seed=3)
    model = trainer.fit("logistic", X, y, NAMES)
    return {"format": 1, "kind": "logistic", "feature_version": play_features.FEATURE_VERSION,
            "features": list(NAMES), "profile": "tracking", "seeds": 3, "stride": 5,
            "provenance": {"scikit_learn": "1.9.1", "calibration_sha256": "ab" * 32,
                           "dfl_sources": {"J03WMX": {"information": "01" * 32, "positions": "02" * 32}},
                           "source_sha256": {"train_dead_ball.py": "aa" * 32, "play_features.py": "bb" * 32,
                                             "kinematics.py": "cc" * 32}},
            "folds": {"J03WMX": model}, "full": model}


def quietly(function, *arguments):
    """(result, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments)
    return result, buffer.getvalue()


# -- the export -------------------------------------------------------------------------------------------------------

def test_exported_trees_match_sklearn():
    X, y = random_rows(6000, seed=1)
    estimator = trainer.hgb_estimator(X, y, NAMES)
    model = trainer.fit("hgb", X, y, NAMES)
    assert len(model["trees"]) == estimator.n_iter_ == trainer.HGB_SETTINGS["max_iter"]
    probe, _ = random_rows(5000, seed=2, nan_share=0.25)          # rows the model never saw, with more NaN
    expected = estimator.predict_proba(probe)[:, 1]
    for rows in (X, probe):
        got = dead_ball_model.predict(model, rows)
        wanted = estimator.predict_proba(rows)[:, 1]
        worst = float(np.abs(got - wanted).max())
        print("      exported trees against sklearn: max |difference| {:.3e} on {:,} rows".format(worst, len(rows)))
        assert worst <= 1e-12, worst
    assert 0.0 < expected.min() and expected.max() < 1.0
    assert model["prior_alive"] == float(np.mean(y))
    tree = model["trees"][0]
    assert sorted(tree) == ["feature", "leaf", "left", "missing_left", "right", "threshold", "value"]
    assert len({len(values) for values in tree.values()}) == 1
    json.dumps(model, allow_nan=False)
    # the monotone constraints are in the exported trees: along each of the two columns, with everything else held,
    # the probability never rises (a NaN in the column is a different question and is left out)
    grid = np.linspace(-3.0, 3.0, 61)
    for name in trainer.MONOTONIC:
        column = NAMES.index(name)
        for row in probe[:40]:
            held = np.tile(row, (len(grid), 1))
            held[:, column] = grid
            p = dead_ball_model.predict(model, held)
            assert (np.diff(p) <= 1e-12).all(), "{} is not non-increasing in the exported model".format(name)


def test_training_is_deterministic_across_threads():
    from threadpoolctl import threadpool_limits
    X, y = random_rows(5000, seed=4)
    texts = []
    for limit in (1, 4):
        with threadpool_limits(limits=limit):
            texts.append(trainer.dumps({"model": trainer.fit("hgb", X, y, NAMES)}))
    assert texts[0] == texts[1], "the same fit on 1 and on 4 threads wrote different bytes"
    again = trainer.dumps({"model": trainer.fit("hgb", X, y, NAMES)})
    assert again == texts[0]


def test_export_survives_splits_on_missingness():
    """A split that sends every seen value left and NaN right has threshold +inf in scikit-learn; it must export."""
    rng = np.random.default_rng(1)
    n = 30000
    X = rng.normal(size=(n, len(NAMES)))
    missing = rng.random(n) < 0.4                      # like a missing ball: several columns NaN together
    for column in (1, 2, 3, 4, 5):
        X[missing, column] = np.nan
    y = (np.where(missing, -1.5, 2.0) + 0.5 * np.nan_to_num(X[:, 6]) + rng.normal(size=n) * 0.5 > 0).astype(np.int64)
    estimator = trainer.hgb_estimator(X, y, NAMES)
    infinite = sum(int(np.isposinf(round_[0].nodes["num_threshold"][~round_[0].nodes["is_leaf"].astype(bool)]).sum())
                   for round_ in estimator._predictors)
    assert infinite > 0, "the data no longer produce a split on missingness; the check covers nothing"
    model = trainer.fit("hgb", X, y, NAMES)
    json.dumps(model, allow_nan=False)
    assert any(value == np.finfo(np.float64).max for tree in model["trees"] for value in tree["threshold"])
    worst = float(np.abs(dead_ball_model.predict(model, X) - estimator.predict_proba(X)[:, 1]).max())
    print("      {} splits on missingness exported: max |difference| {:.3e}".format(infinite, worst))
    assert worst <= 1e-12, worst


def test_trainer_refuses_a_match_filed_under_another_id():
    match = random_match("RND001")
    expect_error(ValueError, lambda: trainer.gather("clean", 1, 5, matches={"RND001": match, "RND009": match}),
                 "RND009", "RND001")


def test_hgb_settings_are_the_designs():
    assert trainer.HGB_SETTINGS == {
        "loss": "log_loss", "learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 1000,
        "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False, "random_state": 0, "class_weight": None}
    assert trainer.MONOTONIC == {"ball_outside_m": -1, "slow_share_c4": -1}
    X, y = random_rows(3000, seed=5)
    estimator = trainer.hgb_estimator(X, y, NAMES)
    expected = [trainer.MONOTONIC.get(name, 0) for name in NAMES]
    assert list(estimator.monotonic_cst) == expected and estimator.early_stopping is False


# -- rows, folds, provenance ------------------------------------------------------------------------------------------

def test_trainer_folds_exclude_the_held_out_match():
    matches = three_matches()
    calls = []
    real = trainer.fit
    trainer.fit = recording_fit(calls)
    try:
        artifact = trainer.train("hgb", profile="gaps", seeds=2, stride=7, matches=matches,
                                 calibration=synthetic_calibration())
    finally:
        trainer.fit = real
    data = trainer.gather("gaps", 2, 7, matches=matches, calibration=synthetic_calibration()).data
    assert sorted(data) == list(FOLD_IDS)
    assert sorted(artifact["folds"]) == list(FOLD_IDS) and artifact["full"]["rows"] == sum(
        len(y) for _, y in data.values())
    assert (artifact["format"], artifact["kind"], artifact["profile"], artifact["seeds"], artifact["stride"]) == (
        1, "hgb", "gaps", 2, 7)
    assert artifact["features"] == list(NAMES) and artifact["feature_version"] == play_features.FEATURE_VERSION
    assert len(calls) == len(FOLD_IDS) + 1
    for match_id in FOLD_IDS:
        others = [name for name in FOLD_IDS if name != match_id]
        wanted_X = np.concatenate([data[name][0] for name in others])
        wanted_y = np.concatenate([data[name][1] for name in others])
        fold = artifact["folds"][match_id]
        assert fold["rows"] == len(wanted_y) and fold["prior_alive"] == float(wanted_y.mean())
        held_out_X = data[match_id][0]
        hits = [call for call in calls if call[1].shape == wanted_X.shape
                and np.array_equal(call[1], wanted_X, equal_nan=True) and np.array_equal(call[2], wanted_y)]
        assert len(hits) == 1, "the fold of {} was not fitted on exactly the other matches".format(match_id)
        # none of the held-out rows is among the rows it was fitted on
        key = {row.tobytes() for row in np.nan_to_num(hits[0][1], nan=-9.99e9)}
        assert not any(row.tobytes() in key for row in np.nan_to_num(held_out_X, nan=-9.99e9))
    everything = np.concatenate([data[name][0] for name in FOLD_IDS])
    assert any(np.array_equal(call[1], everything, equal_nan=True) for call in calls), "full saw every match"


def test_rows_take_every_stride_th_graded_frame_from_each_blocks_start():
    from pipeline.benchmark import inputs as match_inputs, truth
    match = random_match("RND001", rows=(103, 60), gap=25)
    atd = trainer.make_atd(match, "clean", 0, None)
    inputs = match_inputs.MatchInputs(match, atd, profile="clean", seed=0)
    codes, _ = truth.truth_codes(match)
    X, y = trainer.rows(inputs, codes, 5)
    frames = np.concatenate([np.arange(start, stop, 5) for _, start, stop in match.graded])
    assert X.shape == (len(frames), len(NAMES)) and y.shape == (len(frames),)
    assert np.array_equal(X, play_features.features(inputs)[1][frames], equal_nan=True)
    assert np.array_equal(y, (codes[frames] != trainer.DEAD).astype(y.dtype))
    assert len(frames) == -(-103 // 5) + -(-60 // 5) and y.dtype.kind in "iu" and set(np.unique(y)) <= {0, 1}
    assert not np.isnan(X[:, play_features.COLUMN["n_present"]]).any(), "every row is a graded frame"
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, 0), "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, True), "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes[:-1], 5), "codes")


def test_trainer_never_passes_truth():
    match = tracked_match("RND001")
    seen, real = [], play_features.features

    def spy(inputs):
        seen.append(leaks(inputs, match))
        return real(inputs)
    play_features.features = spy
    try:
        calls = []
        real_fit = trainer.fit
        trainer.fit = recording_fit(calls)
        try:
            trainer.train("hgb", profile="tracking", seeds=1, stride=9, calibration=tracking_calibration(),
                          matches={"RND001": match, "RND002": tracked_match("RND002", 1)})
        finally:
            trainer.fit = real_fit
    finally:
        play_features.features = real
    assert seen, "the trainer never asked for features"
    assert all(found == [] for found in seen), seen
    inputs = trainer.served_inputs(match, "tracking", 0, tracking_calibration(), None)
    assert leaks(inputs, match) == []


def test_trainer_needs_two_matches_and_a_known_profile():
    calibration = synthetic_calibration()
    expect_error(ValueError, lambda: trainer.train("hgb", profile="gaps", seeds=1, stride=5, calibration=calibration,
                                                   matches={"RND001": random_match("RND001")}), "at least two")
    expect_error(ValueError, lambda: trainer.train("forest", profile="clean", seeds=1, stride=5,
                                                   matches=three_matches()), "forest", "hgb", "logistic")
    expect_error(ValueError, lambda: trainer.train("hgb", profile="nope", seeds=1, stride=5, matches=three_matches()),
                 "nope")
    for bad in (0, True, 1.5):
        expect_error(ValueError, lambda: trainer.train("hgb", profile="clean", seeds=bad, stride=5,
                                                       matches=three_matches()), "seeds")
        expect_error(ValueError, lambda: trainer.train("hgb", profile="clean", seeds=1, stride=bad,
                                                       matches=three_matches()), "stride")


def test_provenance_holds_versions_and_hashes_and_no_commit():
    import sklearn
    calls = []
    real = trainer.fit
    trainer.fit = recording_fit(calls)
    try:
        artifact = trainer.train("logistic", profile="clean", seeds=1, stride=11, matches=three_matches())
    finally:
        trainer.fit = real
    provenance = artifact["provenance"]
    assert provenance["scikit_learn"] == sklearn.__version__
    assert provenance["calibration_sha256"] is None, "the clean profile degrades nothing and has no calibration"
    assert sorted(provenance["dfl_sources"]) == list(FOLD_IDS)       # synthetic matches have no cache: None each
    assert all(value is None for value in provenance["dfl_sources"].values())
    assert sorted(provenance["source_sha256"]) == ["kinematics.py", "play_features.py", "train_dead_ball.py"]
    folder = os.path.dirname(os.path.abspath(trainer.__file__))
    evidence = os.path.join(os.path.dirname(folder), "evidence")
    for name, path in (("train_dead_ball.py", os.path.join(folder, "train_dead_ball.py")),
                       ("play_features.py", os.path.join(evidence, "play_features.py")),
                       ("kinematics.py", os.path.join(evidence, "kinematics.py"))):
        with open(path, "rb") as handle:
            text = handle.read().replace(b"\r\n", b"\n")
        assert provenance["source_sha256"][name] == hashlib.sha256(text).hexdigest(), name
    assert not any("commit" in key or "git" in key for key in provenance)
    real = trainer.fit
    trainer.fit = recording_fit([])
    try:
        calibration = synthetic_calibration()
        gapped = trainer.train("logistic", profile="gaps", seeds=1, stride=11, matches=three_matches(),
                               calibration=calibration)
    finally:
        trainer.fit = real
    assert gapped["provenance"]["calibration_sha256"] == calibration_digest(calibration)


# -- the file ---------------------------------------------------------------------------------------------------------

def test_artifact_json_is_exact_lf_and_bounded():
    artifact = small_artifact()
    text = trainer.dumps(artifact)
    assert text.endswith("\n") and "\r" not in text and text.count("\n") == 1
    assert json.loads(text) == artifact, "floats must read back exactly"
    assert text == json.dumps(artifact, sort_keys=True) + "\n"
    expect_error(ValueError, lambda: trainer.dumps({"x": float("nan")}), "NaN")
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "models", "dead_ball_logistic.json")
        trainer.write(path, artifact)
        with open(path, "rb") as handle:
            assert handle.read() == text.encode("utf-8")
        big = {"padding": ["x" * 1000] * (trainer.MAX_BYTES // 1000 + 10)}
        expect_error(ValueError, lambda: trainer.write(os.path.join(root, "big.json"), big), "10")
        assert not os.path.exists(os.path.join(root, "big.json"))


def test_logistic_artifact_round_trip():
    rng = np.random.default_rng(6)
    X, y = random_rows(3000, seed=6, nan_share=0.0)
    X[:, 5] = 3.0                                           # constant: dropped, and recorded as such
    X[:, 7] = np.where(rng.random(len(X)) < 0.2, np.nan, X[:, 7])      # a column with missing values: an indicator
    X[:, 9] = np.where(rng.random(len(X)) < 0.3, np.nan, 1.0)          # missing or constant: its mean is constant too
    X[:, 11] = np.nan                                       # never seen: imputed with 0, both columns constant
    model = trainer.fit("logistic", X, y, NAMES)
    F = len(NAMES)
    assert model["prior_alive"] == float(y.mean())
    assert model["indicators"] == [7, 9, 11]
    assert {5, 11, F + 11}.issubset(model["dropped"]) and not set(model["dropped"]) & set(model["columns"])
    assert sorted(model["columns"] + model["dropped"]) == sorted(list(range(F)) + [F + 7, F + 9, F + 11])
    assert len(model["coef"]) == len(model["mean"]) == len(model["scale"]) == len(model["columns"])
    assert model["impute"][11] == 0.0 and len(model["impute"]) == F
    assert np.isclose(model["impute"][7], np.nanmean(X[:, 7]))
    assert all(scale > 0 for scale in model["scale"])
    revived = json.loads(trainer.dumps(model))
    assert revived == model

    # the independent reading: build the design by hand, standardise with the stored numbers, apply the coefficients
    probe, _ = random_rows(500, seed=8, nan_share=0.2)
    design = []
    for column in model["columns"]:
        if column < F:
            design.append(np.where(np.isnan(probe[:, column]), model["impute"][column], probe[:, column]))
        else:
            design.append(np.isnan(probe[:, column - F]).astype(float))
    design = np.column_stack(design)
    expected = hazard.predict(hazard.LogisticModel(coef=np.array(model["coef"]), intercept=model["intercept"],
                                                   mean=np.array(model["mean"]), scale=np.array(model["scale"])),
                              design)
    got = dead_ball_model.predict(revived, probe)
    assert np.abs(got - expected).max() <= 1e-12
    assert np.array_equal(got, dead_ball_model.predict(model, probe)), "the JSON round trip changes nothing"
    # a model that is fitted on this X predicts its own mean well: the intercept and the indicators are in use
    assert abs(float(dead_ball_model.predict(model, X).mean()) - float(y.mean())) < 0.02


def test_rule_values_rule():
    towards = trainer.towards_half
    assert towards(0.75) == 0.75 and towards(0.7499) == 0.74 and towards(0.7599) == 0.75
    assert towards(0.2501) == 0.26 and towards(0.2599) == 0.26 and towards(0.2401) == 0.25
    assert towards(0.5) == 0.5 and towards(0.5004) == 0.5 and towards(0.4996) == 0.5
    assert towards(0.999) == 0.99 and towards(0.001) == 0.01 and towards(1.0) == 1.0 and towards(0.0) == 0.0
    assert towards(0.29) == 0.29 and towards(0.57) == 0.57 and towards(0.58) == 0.58, "no float noise across a cell"
    counts = {"alive": 1000, "dead": 500,
              "out": (300, 50),                 # P(t | alive) 0.3, P(t | dead) 0.1: 0.75
              "still": (5, 400),                # 0.005 / (0.005 + 0.8) = 0.00621..: 0.01
              "slow": (500, 250),               # 0.5 / (0.5 + 0.5) = 0.5
              "kickoff": (0, 0),                # never fires: no information, 0.5
              "none": (901, 50)}                # 0.901 / (0.901 + 0.1) = 0.9000..: 0.9
    values = trainer.rule_values(counts)
    assert sorted(values) == ["alive_kickoff", "alive_none", "alive_out", "alive_slow", "alive_still"]
    assert values["alive_out"]["value"] == 0.75 and values["alive_still"]["value"] == 0.01
    assert values["alive_slow"]["value"] == 0.5 and values["alive_kickoff"]["value"] == 0.5
    assert values["alive_none"]["value"] == 0.9
    assert values["alive_out"]["p_given_alive"] == 0.3 and values["alive_out"]["p_given_dead"] == 0.1
    assert values["alive_kickoff"]["p_given_alive"] == 0.0 and values["alive_kickoff"]["fired"] == 0
    assert values["alive_out"]["fired"] == 350
    # the value of a trigger that fires in one class only is 0 or 1 exactly, then pulled to the nearest cell
    assert trainer.rule_values(dict(counts, out=(10, 0)))["alive_out"]["value"] == 1.0
    assert trainer.rule_values(dict(counts, out=(0, 10)))["alive_out"]["value"] == 0.0
    expect_error(ValueError, lambda: trainer.rule_values(dict(counts, dead=0)), "dead")
    expect_error(ValueError, lambda: trainer.rule_values(dict(counts, out=(2000, 0))), "out")


def test_rule_counts_use_the_rules_own_triggers():
    """The triggers are read off dead_ball_rules by probing it, so a trigger changed there changes here."""
    from pipeline.benchmark import inputs as match_inputs, truth
    match = tracked_match("RND001")
    atd = trainer.make_atd(match, "clean", 0, None)
    inputs = match_inputs.MatchInputs(match, atd, profile="clean", seed=0)
    codes, _ = truth.truth_codes(match)
    masks = trainer.trigger_masks(inputs)
    assert sorted(masks) == ["kickoff", "none", "out", "slow", "still"]
    graded = ~np.isnan(play_features.features(inputs)[1][:, 0])
    union = masks["out"] | masks["still"] | masks["slow"] | masks["kickoff"]
    assert np.array_equal(masks["none"] & graded, graded & ~union) and not (masks["none"] & union).any()
    counts = trainer.count_triggers(inputs, codes)
    assert counts["alive"] + counts["dead"] == int(graded.sum())
    assert counts["dead"] == int(((codes == trainer.DEAD) & graded).sum())
    assert counts["none"][0] + counts["none"][1] == int((masks["none"] & graded).sum())


# -- --verify and the command line ------------------------------------------------------------------------------------

def test_verify_ignores_only_source_hashes():
    committed = small_artifact()
    fresh = json.loads(trainer.dumps(committed))
    assert trainer.compare(committed, fresh) == ([], [])
    fresh["provenance"]["source_sha256"]["play_features.py"] = "dd" * 32
    problems, notes = trainer.compare(committed, fresh)
    assert problems == [] and len(notes) == 1 and "play_features.py" in notes[0], (problems, notes)

    def changed(edit):
        other = json.loads(trainer.dumps(committed))
        edit(other)
        return trainer.compare(committed, other)[0]
    assert changed(lambda a: a["provenance"].update(scikit_learn="1.9.2"))
    assert changed(lambda a: a["provenance"].update(calibration_sha256="00" * 32))
    assert changed(lambda a: a["provenance"]["dfl_sources"]["J03WMX"].update(positions="03" * 32))
    assert changed(lambda a: a["folds"]["J03WMX"]["coef"].__setitem__(0, a["folds"]["J03WMX"]["coef"][0] + 1e-12))
    assert changed(lambda a: a["full"].update(prior_alive=0.5))
    assert changed(lambda a: a.update(stride=6)) and changed(lambda a: a.update(seeds=2))
    assert changed(lambda a: a.pop("full")) and changed(lambda a: a["provenance"].update(extra=1))
    assert changed(lambda a: a["provenance"]["source_sha256"].pop("kinematics.py")), "a missing hash is not a note"
    problems = changed(lambda a: a["folds"]["J03WMX"]["coef"].__setitem__(0, 99.0))
    assert any("folds" in problem and "coef" in problem for problem in problems), problems

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "dead_ball_logistic.json")
        trainer.write(path, committed)
        code, text = quietly(trainer.verify, path, fresh)
        assert code == 0 and "note" in text.lower() and "play_features.py" in text
        other = json.loads(trainer.dumps(committed))
        other["full"]["intercept"] += 1e-9
        code, text = quietly(trainer.verify, path, other)
        assert code == 1 and "full" in text and "intercept" in text
        code, text = quietly(trainer.verify, os.path.join(root, "missing.json"), fresh)
        assert code == 1 and "missing.json" in text and "pipeline/benchmark/train_dead_ball.py" in text


def test_command_line_is_the_designs():
    parser = trainer.build_parser()
    args = parser.parse_args(["--model", "hgb"])
    assert (args.model, args.profile, args.seeds, args.stride, args.out, args.verify, args.rules, args.data_dir) == (
        "hgb", "tracking", 3, 5, None, False, False, None)
    args = parser.parse_args(["--rules", "--profile", "gaps", "--seeds", "2", "--data-dir", "D"])
    assert args.rules and args.model is None and args.profile == "gaps" and args.seeds == 2 and args.data_dir == "D"
    assert trainer.default_path("hgb").replace("\\", "/").endswith("pipeline/models/dead_ball_hgb.json")
    for bad in (["--model", "forest"], [], ["--model", "hgb", "--rules"], ["--rules", "--verify"],
                ["--rules", "--out", "x"], ["--rules", "--stride", "3"]):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            try:
                trainer.main(bad)
            except SystemExit as stop:
                assert stop.code != 0, bad
            else:
                raise AssertionError("{} was accepted".format(bad))


def test_held_out_report_is_counts_and_means_only():
    X, y = random_rows(2400, seed=11)
    data = {"M1": (X[:1200], y[:1200]), "M2": (X[1200:], y[1200:])}
    model = {name: trainer.fit("logistic", np.concatenate([data[other][0] for other in data if other != name]),
                               np.concatenate([data[other][1] for other in data if other != name]), NAMES)
             for name in data}
    artifact = {"folds": model, "kind": "logistic"}
    report = trainer.held_out(artifact, data)
    assert sorted(report["per_match"]) == ["M1", "M2"]
    for scores in report["per_match"].values():
        assert set(scores) == {"rows", "log_loss", "brier", "auc"} and 0.5 < scores["auc"] <= 1.0
        assert 0.0 < scores["log_loss"] < 0.8 and 0.0 < scores["brier"] < 0.25
    table = report["reliability"]
    assert len(table) == 10 and sum(row["count"] for row in table) == 2400
    assert all(set(row) == {"bin", "count", "mean_predicted", "mean_observed"} for row in table)
    lines = trainer.render_report(report)
    assert any("M1" in line for line in lines) and any("reliability" in line.lower() for line in lines)


def test_export_refuses_what_it_cannot_read():
    """A preprocessor, a missing private attribute or a changed layout is a ValueError that says so, never a crash."""
    import types
    X, y = random_rows(1500, seed=12)
    estimator = trainer.hgb_estimator(X, y, NAMES)
    estimator._preprocessor = object()
    expect_error(ValueError, lambda: trainer.export_hgb(estimator), "export_hgb", "preprocessor", "reordered")
    estimator._preprocessor = None
    del estimator._preprocessor                       # a version without the attribute is read as having none
    assert trainer.export_hgb(estimator)["trees"]
    for name in ("_baseline_prediction", "_predictors", "n_trees_per_iteration_"):
        broken = trainer.hgb_estimator(X, y, NAMES)
        delattr(broken, name)
        expect_error(ValueError, lambda: trainer.export_hgb(broken), "export_hgb", "requirements-benchmark.txt")
    odd = trainer.hgb_estimator(X, y, NAMES)
    odd._predictors = [[types.SimpleNamespace(nodes=np.zeros(3, dtype=[("value", "f8")]))]]
    expect_error(ValueError, lambda: trainer.export_hgb(odd), "export_hgb", "is_categorical")
