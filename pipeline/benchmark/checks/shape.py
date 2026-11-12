"""Checks of P07's shape-possession trainer (pipeline/benchmark/train_shape.py) and what it writes.

The trainer fits P05's gradient-boosted classifier on the benchmark's degraded tracking to say which team has the ball
from the 26 `shape_features` of a frame, shows it every row twice (as it is, and with the two teams exchanged and the
label flipped), and writes one model per held-out match and `full` as JSON, which pipeline/evidence/shape_possession.py
then runs with numpy alone. What is held against it here:

* the rows are what the Design says: every `stride`-th frame of a graded block from its start, kept where the truth is
  H or A, 1 for H (test_shape_rows_take_every_stride_th_alive_frame_from_each_blocks_start); no truth reaches the
  features (test_shape_trainer_never_passes_truth); the folds never see the match they are scored on
  (test_shape_trainer_folds_exclude_the_held_out_match);
* the augmentation balances the classes exactly and every row has its mirror
  (test_shape_augmentation_is_balanced), and a model is the export of the fit on those rows with P05's settings and no
  monotone constraint (test_shape_fit_is_p05s_settings_on_the_augmented_rows);
* the export is the model: the trees that go into the file, walked by numpy, give scikit-learn's raw score to 1e-12,
  NaN included (test_shape_export_matches_sklearn), and the same fit on one thread and on four writes the same bytes
  (test_shape_training_is_deterministic_across_threads), which `--verify` relies on;
* the file is the whole chain: what the source reads back out of it is what scikit-learn says of the held-out match,
  retraining writes identical bytes, and the bytes are LF, bounded, NaN-free and in the layout the source expects
  (test_shape_artifact_round_trip); `--verify` fails on anything but the three source hashes;
* the report is aggregates, worked out here frame by frame, and it cannot score a match in sample: a missing fold is
  refused (test_shape_report_is_aggregates, test_shape_report_holds_the_held_out_numbers,
  test_shape_report_refuses_what_would_score_in_sample), the stop rule is the Design's number
  (test_shape_stop_rule_is_the_designs), and `touch_agreement` counts what it says on a hand-built scene
  (test_shape_touch_agreement_counts_kept_frames, test_shape_touch_agreement_uses_the_conventions_rule_and_the_fold).

Everything runs on arrays and synthetic matches made here (checks/common.py): none reads data, and none of the numbers
printed is a figure of any match. The one fixture that takes a few seconds, three matches whose truth is a function of
the players alone, is trained once and shared; a check that changes an artifact copies it first.
"""

import collections
import contextlib
import copy
import dataclasses
import functools
import hashlib
import io
import json
import math
import os
import tempfile

import numpy as np

from pipeline.benchmark import conventions, sources, train_dead_ball, truth
from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark.calibration import digest as calibration_digest
from pipeline.benchmark.checks.common import (A, H, NotRun, leaks, loading, random_match, strict_json,
                                              synthetic_calibration, synthetic_matches, tracked_match,
                                              tracking_calibration)
from pipeline.benchmark.run import BenchmarkError
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, shape_features, shape_possession, trees
from pipeline.selftest import expect_error, play_inputs, shape_model, write_shape_model

# Params
NAMES = shape_features.FEATURE_NAMES
COLUMN = shape_features.COLUMN
FOLD_IDS = ("RND001", "RND002", "RND003")
PLANTED_ROWS = (2500, 2000)                  # graded frames of a planted match: enough rows for a fold to split
PLANTED_SEEDS, PLANTED_STRIDE = 2, 5


# -- helpers ----------------------------------------------------------------------------------------------------------

def shape_rows(n=6000, seed=0, nan_share=0.15):
    """(X, y) of n rows over the 26 features, NaN in a share of the cells, and a label that follows the gap between
    the teams: the antisymmetric structure that the augmentation is there to teach."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES)))
    X[:, COLUMN["adv_gap"]] = X[:, COLUMN["h_adv_mean"]] - X[:, COLUMN["a_adv_mean"]]
    X[:, COLUMN["n_diff"]] = np.round(X[:, COLUMN["n_diff"]])
    X[rng.random(X.shape) < nan_share] = np.nan
    score = (np.nan_to_num(X[:, COLUMN["adv_gap"]]) + 0.4 * np.nan_to_num(X[:, COLUMN["n_diff"]])
             + 0.5 * np.nan_to_num(X[:, COLUMN["h_vel_adv"]]) - 0.5 * np.nan_to_num(X[:, COLUMN["a_vel_adv"]])
             + rng.normal(size=n))
    return X, (score > 0).astype(np.int64)


def three_matches():
    return {match_id: random_match(match_id, seed=seed, rows=(300 + 20 * seed, 260)) for seed, match_id
            in enumerate(FOLD_IDS)}


def recording_fit(calls):
    """A stand-in for trainer.fit that records what it was given and returns a model that holds just that."""
    def fit(X, y):
        calls.append((np.array(X), np.array(y)))
        return {"rows": int(len(y)), "home_share": float(np.mean(y))}
    return fit


def quietly(function, *arguments):
    """(result, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments)
    return result, buffer.getvalue()


def planted_match(match_id, seed):
    """A DflMatch whose truth is a function of its players alone: home has the ball when it is the more advanced team.

    Every graded frame is alive. The label is the sign of `adv_gap`, or, where a team is absent and the gap is NaN, the
    sign of `n_diff` (the team that is there has the ball): antisymmetric, so that a model can learn it from the rows
    and their mirrors. The `gaps` profile leaves the players as they are, so every seed of it has these features."""
    base = random_match(match_id, seed=seed, rows=PLANTED_ROWS, gap=25)
    inputs = trainer.served_inputs(base, "gaps", 0, synthetic_calibration(), None)
    X = shape_features.features(inputs)[1]
    gap, n_diff = X[:, COLUMN["adv_gap"]], X[:, COLUMN["n_diff"]]
    with np.errstate(invalid="ignore"):
        home = np.where(np.isnan(gap), n_diff > 0, gap > 0)
    inside = np.asarray(base.frame_number) >= 0
    possession = np.where(inside, np.where(home, 0, 1), -1).astype(np.int8)
    status = np.where(inside, 1, -1).astype(np.int8)
    possession.setflags(write=False)
    status.setflags(write=False)
    return dataclasses.replace(base, ball_status=status, ball_possession=possession)


@functools.lru_cache(maxsize=1)
def planted():
    """(matches, artifact): three planted matches, and the artifact trained on them with the `gaps` profile."""
    matches = {match_id: planted_match(match_id, seed) for seed, match_id in enumerate(FOLD_IDS)}
    artifact = trainer.train("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches,
                             calibration=synthetic_calibration())
    return matches, artifact


@functools.lru_cache(maxsize=1)
def planted_report():
    matches, artifact = planted()
    return trainer.report(artifact, "gaps", PLANTED_SEEDS, matches=matches, calibration=synthetic_calibration())


def small_artifact():
    """A small artifact of stumps with a provenance made by hand: what the verify and the file checks need."""
    model = shape_model(-0.2, 1.0, -1.0)
    return {"format": 1, "kind": "hgb", "feature_version": shape_features.FEATURE_VERSION, "features": list(NAMES),
            "profile": "tracking", "seeds": 3, "stride": 5,
            "provenance": {"scikit_learn": "1.9.1", "calibration_sha256": "ab" * 32,
                           "dfl_sources": {"J03WMX": {"information": "01" * 32, "positions": "02" * 32}},
                           "source_sha256": {"train_shape.py": "aa" * 32, "shape_features.py": "bb" * 32,
                                             "trees.py": "cc" * 32}},
            "folds": {"J03WMX": model}, "full": shape_model(0.7, -1.0, 2.0)}


# -- the rows and the augmentation ------------------------------------------------------------------------------------

def test_shape_rows_take_every_stride_th_alive_frame_from_each_blocks_start():
    match = random_match("RND001", rows=(103, 60), gap=25)
    inputs = trainer.served_inputs(match, "clean", 0, None, None)
    codes, _ = truth.truth_codes(match)
    X, y = trainer.rows(inputs, codes, 5)
    on_grid = np.concatenate([np.arange(start, stop, 5) for _, start, stop in match.graded])
    assert len(on_grid) == -(-103 // 5) + -(-60 // 5)
    alive = [frame for frame in on_grid if codes[frame] in (H, A)]
    assert 0 < len(alive) < len(on_grid), "the synthetic match must have dead frames among the grid, and alive ones"
    assert X.shape == (len(alive), len(NAMES)) and y.shape == (len(alive),)
    assert np.array_equal(X, shape_features.features(inputs)[1][alive], equal_nan=True)
    assert np.array_equal(y, np.array([codes[frame] == H for frame in alive], dtype=y.dtype))
    assert y.dtype.kind in "iu" and set(np.unique(y)) == {0, 1}
    assert not np.isnan(X[:, COLUMN["h_n"]]).any(), "every row is a graded frame"
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, 0), "train_shape", "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, True), "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes[:-1], 5), "codes")


def test_shape_augmentation_is_balanced():
    X, y = shape_rows(900, seed=3)
    y = (np.arange(len(y)) % 5 == 0).astype(np.int64)                      # 20% homes: the rows are not balanced
    n = len(X)
    Xa, ya = trainer.augment(X, y)
    assert Xa.shape == (2 * n, len(NAMES)) and ya.shape == (2 * n,) and ya.dtype == np.int64
    assert int(ya.sum()) * 2 == len(ya) and float(ya.mean()) == 0.5, "exactly half the labels are 1"
    assert np.array_equal(Xa[:n], X, equal_nan=True) and np.array_equal(ya[:n], y), "the rows come first, unchanged"
    for i in range(n):
        assert np.array_equal(Xa[n + i], shape_features.swap(Xa[i]), equal_nan=True), i
        assert np.array_equal(shape_features.swap(Xa[n + i]), Xa[i], equal_nan=True), i
        assert ya[n + i] == 1 - ya[i], i
    # every row has its swapped twin, with the other label, somewhere in the set (a multiset closed under the mirror)
    def key(row, label):                                          # + 0.0: a -0.0 and a 0.0 are the same feature
        return (np.nan_to_num(row, nan=-9.99e9) + 0.0).tobytes() + bytes([int(label)])
    held = collections.Counter(key(row, label) for row, label in zip(Xa, ya))
    for row, label in zip(Xa, ya):
        assert held[key(shape_features.swap(row), 1 - label)] >= 1
    expect_error(ValueError, lambda: trainer.augment(X[:, :25], y), "train_shape", "26")
    expect_error(ValueError, lambda: trainer.augment(X, y[:-1]), "train_shape")
    expect_error(ValueError, lambda: trainer.augment(X, y + 1), "train_shape", "0 or 1")


# -- the fit and the export -------------------------------------------------------------------------------------------

def test_shape_fit_is_p05s_settings_on_the_augmented_rows():
    assert trainer.HGB_SETTINGS is train_dead_ball.HGB_SETTINGS, "P05's settings, imported and not retyped"
    assert trainer.HGB_SETTINGS == {
        "loss": "log_loss", "learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 1000,
        "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False, "random_state": 0, "class_weight": None}
    X, y = shape_rows(3000, seed=5)
    Xa, ya = trainer.augment(X, y)
    estimator = trainer.hgb_estimator(Xa, ya)
    assert estimator.monotonic_cst is None, "the Design has no monotone constraints"
    assert estimator.early_stopping is False and estimator.n_iter_ == trainer.HGB_SETTINGS["max_iter"]
    assert estimator.n_features_in_ == len(NAMES) and len(estimator._predictors) == 300
    # fit is exactly the export of that estimator, plus the prior of rows and their mirrors
    model = trainer.fit(X, y)
    assert model == {**trainer.export_hgb(estimator), "prior_home": 0.5}
    assert sorted(model) == ["baseline", "prior_home", "trees"] and model["prior_home"] == 0.5
    assert abs(model["baseline"]) < 1e-12, "the rows and their mirrors are balanced: the baseline is log-odds 0"
    expect_error(ValueError, lambda: trainer.fit(X, np.ones(len(X), dtype=np.int64)), "train_shape", "both")
    expect_error(ValueError, lambda: trainer.fit(X[:, :20], y), "train_shape", "26")
    expect_error(ValueError, lambda: trainer.fit(X, y[:-1]), "train_shape")


def test_shape_export_matches_sklearn():
    X, y = shape_rows(6000, seed=1)
    Xa, ya = trainer.augment(X, y)
    estimator = trainer.hgb_estimator(Xa, ya)
    model = trainer.fit(X, y)
    assert len(model["trees"]) == 300
    assert any(len(tree["feature"]) > 3 for tree in model["trees"]), "the fit must grow trees for this to mean much"
    probe, _ = shape_rows(5000, seed=2, nan_share=0.25)                   # rows the model never saw, with more NaN
    twins = shape_features.swap(probe)
    for rows in (Xa, probe, twins):
        got = trees.score(model, rows)
        wanted = estimator.decision_function(rows)
        worst = float(np.abs(got - wanted).max())
        print("      exported trees against sklearn: max |difference| {:.3e} on {:,} rows".format(worst, len(rows)))
        assert worst <= 1e-12, worst
    trees.check_trees(model, len(NAMES), "a fold of m.json", "shape_possession")
    json.dumps(model, allow_nan=False)
    # the fit has learnt something: the held-out rows are separated better than chance
    p = 1.0 / (1.0 + np.exp(-estimator.decision_function(probe)))
    _, probe_y = shape_rows(5000, seed=2, nan_share=0.25)
    assert float(np.mean((p >= 0.5) == (probe_y == 1))) > 0.7


def test_shape_training_is_deterministic_across_threads():
    from threadpoolctl import threadpool_limits
    X, y = shape_rows(5000, seed=4)
    texts = []
    for limit in (1, 4):
        with threadpool_limits(limits=limit):
            texts.append(trainer.dumps({"model": trainer.fit(X, y)}))
    assert texts[0] == texts[1], "the same fit on 1 and on 4 threads wrote different bytes"
    assert trainer.dumps({"model": trainer.fit(X, y)}) == texts[0]


# -- gathering, folds, provenance -------------------------------------------------------------------------------------

def test_shape_trainer_folds_exclude_the_held_out_match():
    matches = three_matches()
    calls = []
    real = trainer.fit
    trainer.fit = recording_fit(calls)
    try:
        artifact = trainer.train("gaps", 2, 7, matches=matches, calibration=synthetic_calibration())
    finally:
        trainer.fit = real
    data = trainer.gather("gaps", 2, 7, matches=matches, calibration=synthetic_calibration()).data
    assert sorted(data) == list(FOLD_IDS)
    assert sorted(artifact["folds"]) == list(FOLD_IDS) and artifact["full"]["rows"] == sum(
        len(y) for _, y in data.values())
    assert (artifact["format"], artifact["kind"], artifact["profile"], artifact["seeds"], artifact["stride"]) == (
        1, "hgb", "gaps", 2, 7)
    assert artifact["features"] == list(NAMES) and artifact["feature_version"] == shape_features.FEATURE_VERSION
    assert len(calls) == len(FOLD_IDS) + 1
    for match_id in FOLD_IDS:
        others = [name for name in FOLD_IDS if name != match_id]
        wanted_X = np.concatenate([data[name][0] for name in others])
        wanted_y = np.concatenate([data[name][1] for name in others])
        fold = artifact["folds"][match_id]
        assert fold["rows"] == len(wanted_y) and fold["home_share"] == float(wanted_y.mean())
        hits = [call for call in calls if call[0].shape == wanted_X.shape
                and np.array_equal(call[0], wanted_X, equal_nan=True) and np.array_equal(call[1], wanted_y)]
        assert len(hits) == 1, "the fold of {} was not fitted on exactly the other matches".format(match_id)
        # none of the held-out rows is among the rows it was fitted on
        seen = {row.tobytes() for row in np.nan_to_num(hits[0][0], nan=-9.99e9)}
        assert not any(row.tobytes() in seen for row in np.nan_to_num(data[match_id][0], nan=-9.99e9))
    everything = np.concatenate([data[name][0] for name in FOLD_IDS])
    assert any(np.array_equal(call[0], everything, equal_nan=True) for call in calls), "full saw every match"
    # the rows are the un-augmented ones: the augmentation is the fit's, so a fold's model is the one that mirrors them
    assert all(len(call[1]) == len(call[0]) and set(np.unique(call[1])) <= {0, 1} for call in calls)


def test_shape_trainer_never_passes_truth():
    for profile, calibration, matches in (
            ("tracking", tracking_calibration(), {"RND001": tracked_match("RND001"),
                                                  "RND002": tracked_match("RND002", 1)}),
            ("gaps", synthetic_calibration(), synthetic_matches())):
        seen, real = [], shape_features.features
        match = matches["RND001"]

        def spy(inputs, version=1):
            seen.append([leaks(inputs, other) for other in matches.values()])      # against every match's truth
            return real(inputs, version)
        shape_features.features = spy
        real_fit = trainer.fit
        trainer.fit = recording_fit([])
        try:
            trainer.train(profile, 1, 9, calibration=calibration, matches=matches)
        finally:
            trainer.fit = real_fit
            shape_features.features = real
        assert seen, "the trainer never asked for features"
        assert all(found == [[] for _ in matches] for found in seen), (profile, seen)
        inputs = trainer.served_inputs(match, profile, 0, calibration, None)
        assert all(leaks(inputs, other) == [] for other in matches.values())


def test_shape_trainer_needs_two_matches_and_a_known_profile():
    calibration = synthetic_calibration()
    expect_error(ValueError, lambda: trainer.train("gaps", 1, 5, calibration=calibration,
                                                   matches={"RND001": random_match("RND001")}), "at least two")
    expect_error(ValueError, lambda: trainer.train("nope", 1, 5, matches=three_matches()), "nope")
    expect_error(ValueError, lambda: trainer.train("clean", 1, 5, matches=three_matches(), calibration=calibration),
                 "clean", "calibration")
    expect_error(ValueError, lambda: trainer.train("gaps", 1, 5, matches={"RND001": random_match("RND001"),
                                                                           "RND009": random_match("RND001")},
                                                   calibration=calibration), "RND009", "RND001")
    for bad in (0, True, 1.5):
        expect_error(ValueError, lambda: trainer.train("clean", bad, 5, matches=three_matches()), "train_shape",
                     "seeds")
        expect_error(ValueError, lambda: trainer.train("clean", 1, bad, matches=three_matches()), "train_shape",
                     "stride")


def test_shape_provenance_holds_versions_and_hashes_and_no_commit():
    import sklearn
    real = trainer.fit
    trainer.fit = recording_fit([])
    try:
        clean = trainer.train("clean", 1, 11, matches=three_matches())
        calibration = synthetic_calibration()
        gapped = trainer.train("gaps", 1, 11, matches=three_matches(), calibration=calibration)
    finally:
        trainer.fit = real
    provenance = clean["provenance"]
    assert provenance["scikit_learn"] == sklearn.__version__
    assert provenance["calibration_sha256"] is None, "the clean profile degrades nothing and has no calibration"
    assert sorted(provenance["dfl_sources"]) == list(FOLD_IDS)       # synthetic matches have no cache: None each
    assert all(value is None for value in provenance["dfl_sources"].values())
    assert sorted(provenance["source_sha256"]) == ["shape_features.py", "train_shape.py", "trees.py"]
    folder = os.path.dirname(os.path.abspath(trainer.__file__))
    evidence = os.path.join(os.path.dirname(folder), "evidence")
    for name, path in (("train_shape.py", os.path.join(folder, "train_shape.py")),
                       ("shape_features.py", os.path.join(evidence, "shape_features.py")),
                       ("trees.py", os.path.join(evidence, "trees.py"))):
        with open(path, "rb") as handle:
            text = handle.read().replace(b"\r\n", b"\n")
        assert provenance["source_sha256"][name] == hashlib.sha256(text).hexdigest(), name
    assert not any("commit" in key or "git" in key for key in provenance)
    assert gapped["provenance"]["calibration_sha256"] == calibration_digest(calibration)


# -- the file ---------------------------------------------------------------------------------------------------------

def test_shape_artifact_round_trip():
    matches, artifact = planted()
    calibration = synthetic_calibration()

    # the layout that shape_possession loads
    assert sorted(artifact) == sorted(["format", "kind", "feature_version", "features", "profile", "seeds", "stride",
                                       "provenance", "folds", "full"])
    assert (artifact["format"], artifact["kind"], artifact["profile"], artifact["seeds"], artifact["stride"]) == (
        trees.FORMAT, "hgb", "gaps", PLANTED_SEEDS, PLANTED_STRIDE)
    assert artifact["feature_version"] == shape_features.FEATURE_VERSION and artifact["features"] == list(NAMES)
    assert sorted(artifact["folds"]) == list(FOLD_IDS)
    for name, model in list(artifact["folds"].items()) + [("full", artifact["full"])]:
        assert sorted(model) == ["baseline", "prior_home", "trees"], name
        assert model["prior_home"] == 0.5 and len(model["trees"]) == 300, name
        assert any(len(tree["feature"]) > 1 for tree in model["trees"]), "{} grew no split".format(name)
        trees.check_trees(model, len(NAMES), name, "shape_possession")

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "models", "shape.json")
        trainer.write(path, artifact)
        with open(path, "rb") as handle:
            data = handle.read()
        text = data.decode("utf-8")
        assert text == trainer.dumps(artifact) and text.endswith("\n") and text.count("\n") == 1
        assert b"\r" not in data and len(data) < 10 * 1000 * 1000
        assert strict_json(text) == artifact and json.loads(text) == artifact, "floats must read back exactly"
        print("      the artifact: {:,} bytes, {} models of {} trees".format(
            len(data), len(artifact["folds"]) + 1, len(artifact["full"]["trees"])))

        # loaded, and scored by the source: the held-out match by its fold, which is scikit-learn's own answer
        loaded = shape_possession.load(path)
        assert loaded == artifact
        shape_possession.load(path)                                                      # and again, from its memo
        data_by_match = trainer.gather("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches,
                                       calibration=calibration).data
        for held_out in FOLD_IDS:
            others = [name for name in FOLD_IDS if name != held_out]
            Xa, ya = trainer.augment(np.concatenate([data_by_match[name][0] for name in others]),
                                     np.concatenate([data_by_match[name][1] for name in others]))
            estimator = trainer.hgb_estimator(Xa, ya)
            assert artifact["folds"][held_out] == {**trainer.export_hgb(estimator), "prior_home": 0.5}, held_out
            inputs = trainer.served_inputs(matches[held_out], "gaps", 0, calibration, None)
            assert inputs.game == held_out
            home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=path, scope="all").home
            X = shape_features.features(inputs)[1]
            speaks = (X[:, COLUMN["h_n"]] > 0) | (X[:, COLUMN["a_n"]] > 0)
            rows = X[speaks]
            logit = (estimator.decision_function(rows) - estimator.decision_function(shape_features.swap(rows))) / 2.0
            want = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98)
            assert np.isnan(home[~speaks]).all() and speaks.sum() > 1000
            worst = float(np.abs(home[speaks] - want).max())
            print("      {}: the source against scikit-learn's own fold: max |difference| {:.3e}".format(
                held_out, worst))
            assert worst <= 1e-12, worst

    # retraining gives identical bytes
    again = trainer.train("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches, calibration=calibration)
    assert trainer.dumps(again) == trainer.dumps(artifact), "retraining wrote different bytes"
    problems, notes = trainer.compare(artifact, again)
    assert problems == [] and notes == []
    # a file over 10 MB is refused
    with tempfile.TemporaryDirectory() as root:
        big = {"padding": ["x" * 1000] * (train_dead_ball.MAX_BYTES // 1000 + 10)}
        expect_error(ValueError, lambda: trainer.write(os.path.join(root, "big.json"), big), "10")
        assert not os.path.exists(os.path.join(root, "big.json"))
        expect_error(ValueError, lambda: trainer.dumps({"x": float("nan")}), "NaN")


def test_shape_trainer_version_2_names_its_features_and_version_1_is_unchanged():
    """P30: `version=2` trains on the 42 columns and the file says so; the default is version 1, byte for byte."""
    matches, artifact = planted()
    calibration = synthetic_calibration()
    names2 = shape_features.feature_names(2)
    assert len(names2) == 42
    artifact2 = trainer.train("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches, calibration=calibration,
                              version=2)
    assert artifact2["feature_version"] == 2 and artifact2["features"] == list(names2)
    assert sorted(artifact2["folds"]) == list(FOLD_IDS) and "full" in artifact2
    for name, model in list(artifact2["folds"].items()) + [("full", artifact2["full"])]:
        trees.check_trees(model, 42, name, "shape_possession")
        assert any(feature >= len(NAMES) for tree in model["trees"] for feature in tree["feature"]), (
            "{} never split on a column of version 2".format(name))
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "v2.json")
        trainer.write(path, artifact2)
        shape_possession.load(path)                       # the source accepts it: names, version and tree width agree
    # the artifact trained with version=1 given is the one trained without the keyword
    explicit = trainer.train("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches, calibration=calibration,
                             version=1)
    assert trainer.dumps(explicit) == trainer.dumps(artifact)
    assert artifact["feature_version"] == 1 and artifact["features"] == list(NAMES)
    # what is refused
    for bad in (4, 0, True, 2.0, None, "2"):
        expect_error(ValueError, lambda: trainer.rows(*_rows_args(matches), version=bad), "shape_features", "version")
    expect_error(ValueError, lambda: trainer.augment(np.zeros((4, 26)), np.zeros(4), version=2), "train_shape", "42")
    assert trainer.DEFAULT_MODEL_PATHS == {1: "pipeline/models/shape_possession_hgb.json",
                                           2: "pipeline/models/shape_possession_v2_hgb.json",
                                           3: "pipeline/models/shape_possession_v3_hgb.json"}
    assert trainer.DEFAULT_MODEL_PATH == os.path.join(trainer.ROOT, "pipeline", "models", "shape_possession_hgb.json")
    args = trainer.build_parser().parse_args([])
    assert args.features == 1
    assert trainer.build_parser().parse_args(["--features", "2"]).features == 2
    assert trainer.build_parser().parse_args(["--features", "3"]).features == 3
    for bad in (["--features", "4"], ["--features", "0"], ["--features", "x"]):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            try:
                trainer.build_parser().parse_args(bad)
            except SystemExit:
                continue
        raise AssertionError("{} was accepted".format(bad))


def _rows_args(matches):
    match = matches[FOLD_IDS[0]]
    inputs = trainer.served_inputs(match, "gaps", 0, synthetic_calibration(), None)
    codes, _ = truth.truth_codes(match)
    return inputs, codes, 5


def test_shape_verify_ignores_only_source_hashes():
    committed = small_artifact()
    fresh = json.loads(trainer.dumps(committed))
    assert trainer.compare(committed, fresh) == ([], [])
    fresh["provenance"]["source_sha256"]["trees.py"] = "dd" * 32
    problems, notes = trainer.compare(committed, fresh)
    assert problems == [] and len(notes) == 1 and "trees.py" in notes[0], (problems, notes)

    def changed(edit):
        other = json.loads(trainer.dumps(committed))
        edit(other)
        return trainer.compare(committed, other)[0]
    assert changed(lambda a: a["provenance"].update(scikit_learn="1.9.2"))
    assert changed(lambda a: a["provenance"].update(calibration_sha256="00" * 32))
    assert changed(lambda a: a["provenance"]["dfl_sources"]["J03WMX"].update(positions="03" * 32))
    assert changed(lambda a: a["folds"]["J03WMX"].update(baseline=a["folds"]["J03WMX"]["baseline"] + 1e-12))
    assert changed(lambda a: a["full"]["trees"][0]["value"].__setitem__(1, 0.123))
    assert changed(lambda a: a.update(stride=6)) and changed(lambda a: a.update(seeds=2))
    assert changed(lambda a: a.pop("full")) and changed(lambda a: a["provenance"].update(extra=1))
    assert changed(lambda a: a["provenance"]["source_sha256"].pop("trees.py")), "a missing hash is not a note"
    problems = changed(lambda a: a["folds"]["J03WMX"]["trees"][0]["value"].__setitem__(2, 99.0))
    assert any("folds" in problem and "value" in problem for problem in problems), problems

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "shape_possession_hgb.json")
        trainer.write(path, committed)
        code, text = quietly(trainer.verify, path, fresh)
        assert code == 0 and "note" in text.lower() and "trees.py" in text
        other = json.loads(trainer.dumps(committed))
        other["full"]["baseline"] += 1e-9
        code, text = quietly(trainer.verify, path, other)
        assert code == 1 and "full" in text and "baseline" in text
        code, text = quietly(trainer.verify, os.path.join(root, "missing.json"), fresh)
        assert code == 1 and "missing.json" in text and "pipeline/benchmark/train_shape.py" in text


def test_shape_command_line_is_the_designs():
    parser = trainer.build_parser()
    args = parser.parse_args([])
    # The default profile is the runner's: generic, which reads nothing of our match (P27), not tracking, which does.
    assert (args.profile, args.seeds, args.stride, args.out, args.verify, args.report, args.data_dir) == (
        "generic", 3, 5, None, False, False, None)
    assert args.profile == trainer.run.DEFAULT_PROFILE and not trainer.run.needs_calibration(args.profile)
    args = parser.parse_args(["--profile", "gaps", "--seeds", "2", "--stride", "7", "--out", "x.json", "--report",
                              "--data-dir", "D"])
    assert (args.profile, args.seeds, args.stride, args.out, args.report, args.data_dir) == (
        "gaps", 2, 7, "x.json", True, "D")
    assert trainer.DEFAULT_MODEL_PATH.replace("\\", "/").endswith("pipeline/models/shape_possession_hgb.json")
    assert os.path.isabs(trainer.DEFAULT_MODEL_PATH)
    for bad in (["--verify", "--report"], ["--report", "--stride", "3"], ["--seeds", "0"], ["--stride", "0"],
                ["--profile"], ["--model", "hgb"]):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            try:
                trainer.main(bad)
            except SystemExit as stop:
                assert stop.code != 0, bad
            else:
                raise AssertionError("{} was accepted".format(bad))


def test_shape_main_trains_writes_verifies_and_reports():
    """The three commands the controller runs, on seven synthetic matches under the DFL ids and the clean profile."""
    matches = {match_id: random_match(match_id, seed=seed, rows=(700, 600)) for seed, match_id
               in enumerate(sources.MATCH_IDS)}
    with tempfile.TemporaryDirectory() as root, loading(lambda match_id: matches[match_id]):
        path = os.path.join(root, "models", "shape.json")
        flags = ["--profile", "clean", "--seeds", "1", "--stride", "2", "--out", path]
        code, text = quietly(trainer.main, flags)
        assert code == 0 and "wrote" in text and os.path.exists(path), text
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        assert sorted(artifact["folds"]) == sorted(sources.MATCH_IDS) and artifact["profile"] == "clean"
        assert artifact["seeds"] == 1 and artifact["stride"] == 2 and "full" in artifact
        shape_possession.load(path)
        code, text = quietly(trainer.main, flags + ["--verify"])
        assert code == 0 and "verified" in text, text
        artifact["full"]["baseline"] += 1e-9
        trainer.write(path, artifact)
        code, text = quietly(trainer.main, flags + ["--verify"])
        assert code == 1 and "FAILED" in text and "baseline" in text, text
        artifact["full"]["baseline"] -= 1e-9
        trainer.write(path, artifact)
        code, text = quietly(trainer.main, ["--profile", "clean", "--seeds", "1", "--out", path, "--report"])
        assert code == 0 and "stop rule" in text and "ln 2" in text and all(
            match_id in text for match_id in sources.MATCH_IDS), text
        code, text = quietly(trainer.main, ["--profile", "clean", "--seeds", "1", "--out",
                                            os.path.join(root, "missing.json"), "--report"])
        assert code == 1 and "FAILED" in text and "missing.json" in text and "train_shape.py" in text, text


# -- the report -------------------------------------------------------------------------------------------------------

def test_shape_report_is_aggregates():
    report = planted_report()
    json.dumps(report, allow_nan=False)
    strict_json(json.dumps(report))

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), where
                walk(item, "{}.{}".format(where, key))
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            raise AssertionError("{} is a sequence: the report holds numbers and strings, no per-frame list"
                                 .format(where))
        elif isinstance(value, float):
            assert math.isfinite(value), where
        else:
            assert value is None or isinstance(value, (bool, int, str)), (where, type(value))
    walk(report, "report")
    assert sorted(report["per_match"]) == list(FOLD_IDS)
    cells = list(report["per_match"].values()) + [report["pooled"]]
    for cell in cells:
        assert set(cell) == {"alive", "missing", "touch"}
        assert set(cell["alive"]) == {"frames", "silent", "log_loss", "accuracy"}
        assert set(cell["missing"]) == {"frames", "log_loss", "accuracy"}
        assert set(cell["touch"]) == {"frames", "agreement"}
    assert report["ln2"] == math.log(2) and report["profile"] == "gaps" and report["seeds"] == PLANTED_SEEDS
    # pooled over the matches' frames, not a median of medians
    for part in ("alive", "missing"):
        assert report["pooled"][part]["frames"] == sum(cell[part]["frames"] for cell in report["per_match"].values())
        frames = [cell[part]["frames"] for cell in report["per_match"].values()]
        for stat in ("log_loss", "accuracy"):
            weighted = sum(f * cell[part][stat] for f, cell in zip(frames, report["per_match"].values())) / sum(frames)
            assert abs(report["pooled"][part][stat] - weighted) <= 1e-12, (part, stat)
    assert report["pooled"]["touch"]["frames"] == sum(c["touch"]["frames"] for c in report["per_match"].values())
    lines = trainer.render_report(report)
    assert all(len(line) <= 120 for line in lines) and any("ln 2" in line for line in lines)
    assert all(match_id in "\n".join(lines) for match_id in FOLD_IDS)


def test_shape_report_holds_the_held_out_numbers():
    """The report against plain loops over the frames, on a task the features solve: the stop rule must not fire."""
    matches, artifact = planted()
    calibration = synthetic_calibration()
    report = planted_report()
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "shape.json")
        trainer.write(path, artifact)
        for match_id, match in sorted(matches.items()):
            codes, _ = truth.truth_codes(match)
            sums = {"alive": [0, 0.0, 0], "missing": [0, 0.0, 0]}
            silent = touched = agreed = 0
            for seed in range(PLANTED_SEEDS):
                inputs = trainer.served_inputs(match, "gaps", seed, calibration, None)
                home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=path, scope="all").home
                present = inputs.atd.ball_present
                for frame in range(len(codes)):
                    if codes[frame] not in (H, A):
                        continue
                    if math.isnan(home[frame]):
                        silent += 1
                        continue
                    p, truly_home = float(home[frame]), bool(codes[frame] == H)
                    loss = -math.log(p if truly_home else 1.0 - p)
                    for name, take in (("alive", True), ("missing", not present[frame])):
                        if take:
                            sums[name][0] += 1
                            sums[name][1] += loss
                            sums[name][2] += (p >= 0.5) == truly_home
                touch = trainer.touch_agreement(inputs, model=path)
                touched += touch["frames"]
                agreed += round(touch["agreement"] * touch["frames"]) if touch["frames"] else 0
            cell = report["per_match"][match_id]
            for name, (frames, loss, correct) in sums.items():
                assert cell[name]["frames"] == frames and frames > 100, (match_id, name)
                assert abs(cell[name]["log_loss"] - loss / frames) <= 1e-9, (match_id, name)
                assert cell[name]["accuracy"] == correct / frames, (match_id, name)
            assert cell["alive"]["silent"] == silent
            assert cell["touch"]["frames"] == touched
            assert cell["touch"]["agreement"] == (agreed / touched if touched else None)
            # the ball-missing frames are a part of the alive ones, and not all of them
            assert 0 < cell["missing"]["frames"] < cell["alive"]["frames"]
    assert report["pooled"]["missing"]["log_loss"] < trainer.STOP_LOSS, report["pooled"]
    assert report["pooled"]["missing"]["accuracy"] > 0.9
    assert report["stop_rule"] == {"threshold": trainer.STOP_LOSS, "log_loss": report["pooled"]["missing"]["log_loss"],
                                   "fires": False}
    print("      planted matches, held out: log-loss {:.4f} and accuracy {:.4f} on {:,} alive frames with the ball "
          "missing".format(report["pooled"]["missing"]["log_loss"], report["pooled"]["missing"]["accuracy"],
                           report["pooled"]["missing"]["frames"]))


def test_shape_report_refuses_what_would_score_in_sample():
    matches, artifact = planted()
    calibration = synthetic_calibration()
    # a fold that is missing would fall back to `full`, which saw every match: refused, naming the match
    without = copy.deepcopy(artifact)
    del without["folds"]["RND002"]
    expect_error(BenchmarkError, lambda: trainer.report(without, "gaps", 1, matches=matches, calibration=calibration),
                 "train_shape", "RND002", "fold", "in sample")
    renamed = copy.deepcopy(artifact)
    renamed["folds"]["RND002-copy"] = renamed["folds"].pop("RND002")
    expect_error(BenchmarkError, lambda: trainer.report(renamed, "gaps", 1, matches=matches, calibration=calibration),
                 "RND002")
    expect_error(BenchmarkError, lambda: trainer.report(artifact, "nope", 1, matches=matches), "nope")
    for bad in (0, True, 1.5):
        expect_error(ValueError, lambda: trainer.report(artifact, "gaps", bad, matches=matches,
                                                        calibration=calibration), "train_shape", "seeds")
    # a ball that is on every frame leaves nothing to measure the stop rule on
    whole = {"RND001": tracked_match("RND001"), "RND002": tracked_match("RND002", 1)}
    stumps = dict(small_artifact(), folds={name: shape_model(0.0, -1.0, 2.0) for name in whole},
                  full=shape_model(0.0, -1.0, 2.0))
    expect_error(BenchmarkError, lambda: trainer.report(stumps, "clean", 1, matches=whole), "train_shape",
                 "ball", "stop rule")
    # the same hand-made model on matches whose ball is missing somewhere does report
    gappy = {name: random_match(name, seed=seed) for seed, name in enumerate(("RND001", "RND002"))}
    report = trainer.report(stumps, "clean", 1, matches=gappy)
    assert sorted(report["per_match"]) == ["RND001", "RND002"] and report["pooled"]["missing"]["frames"] > 0


def test_shape_stop_rule_is_the_designs():
    assert trainer.STOP_LOSS == math.log(2) - 0.01 and abs(trainer.STOP_LOSS - 0.6831) < 1e-4
    assert trainer.LN2 == math.log(2)
    assert trainer.stop_rule_fires(0.30) is False
    assert trainer.stop_rule_fires(trainer.STOP_LOSS - 1e-9) is False
    assert trainer.stop_rule_fires(trainer.STOP_LOSS) is True, "at the threshold is not below it"
    assert trainer.stop_rule_fires(math.log(2)) is True and trainer.stop_rule_fires(0.9) is True
    expect_error(ValueError, lambda: trainer.stop_rule_fires(None), "train_shape", "stop rule")
    expect_error(ValueError, lambda: trainer.stop_rule_fires(float("nan")), "train_shape", "stop rule")
    report = json.loads(json.dumps(planted_report()))
    quiet = "\n".join(trainer.render_report(report))
    assert "does not fire" in quiet and "FIRES" not in quiet
    report["stop_rule"] = {"threshold": trainer.STOP_LOSS, "log_loss": 0.69, "fires": True}
    loud = "\n".join(trainer.render_report(report))
    assert "FIRES" in loud and "does not fire" not in loud


# -- touch_agreement --------------------------------------------------------------------------------------------------

def touch_scene(stretches=6, length=100):
    """(players_m, teams, ball_m, graded): six players still but for a jump at each change of stretch, and a ball that
    sits half a metre from a home player on the even stretches and from an away player on the odd ones.

    On an even stretch both teams stand 8 m further up the pitch to the right than on an odd one, so home is the
    more advanced team (adv_gap above 0) exactly while it has the ball, and away exactly while it does."""
    teams = np.array([0, 0, 0, 1, 1, 1])
    base = np.array([[15.0, 20.0], [25.0, 34.0], [35.0, 48.0], [65.0, 20.0], [75.0, 34.0], [85.0, 48.0]])
    n = stretches * length
    shift = np.where((np.arange(n) // length) % 2 == 0, 8.0, -8.0)
    players = np.repeat(base[None], n, axis=0)
    players[:, :, 0] += shift[:, None]
    holder = np.where(shift > 0, 1, 4)
    ball = players[np.arange(n), holder] + [0.5, 0.0]
    return players, teams, ball, [(1, 0, n)]


def touch_model(folder, name, full_follows_the_gap=True, fold="FOLD1"):
    """A model file of stumps on adv_gap: `full` says home when the gap is above 0 (or away, if not
    `full_follows_the_gap`), and the fold of `fold` says the opposite."""
    follows, opposes = shape_model(0.0, -1.0, 2.0), shape_model(0.0, 2.0, -1.0)
    return write_shape_model(folder, name=name, full=follows if full_follows_the_gap else opposes,
                             folds={fold: opposes if full_follows_the_gap else follows})


def test_shape_touch_agreement_counts_kept_frames():
    players, teams, ball, graded = touch_scene()
    inputs = play_inputs(players, ball, teams=teams, graded=graded)
    inputs.game = "BAR-ATH"
    with tempfile.TemporaryDirectory() as root:
        aligned = touch_model(root, "aligned.json")
        opposed = touch_model(root, "opposed.json", full_follows_the_gap=False)
        run = RunContext(inputs=inputs, seed=0)
        episode = controlled_touch.compute(run, **conventions.TOUCH).home
        home = shape_possession.compute(run, model=aligned, scope="all").home
        kept = ~np.isnan(episode)
        assert 500 <= int(kept.sum()) < len(episode), "the scene must give episodes to every stretch, and gaps between"
        assert set((episode[kept] >= 0.5).tolist()) == {True, False}, "home episodes and away ones"
        assert conventions.TOUCH["hold"] == 0.5, "home is a number on the kept frames only, with this rule"

        got = trainer.touch_agreement(inputs, model=aligned)
        assert sorted(got) == ["agreement", "frames"] and isinstance(got["frames"], int)
        # plain loops: kept frames where the shape home is a number, and where it names the episode's team
        frames = agreed = 0
        for frame in range(len(episode)):
            if math.isnan(episode[frame]) or math.isnan(home[frame]):
                continue
            frames += 1
            agreed += (home[frame] >= 0.5) == (episode[frame] >= 0.5)
        # the ball is present on every frame of the scene, so the scope "gaps" would leave no frame to count
        assert got["frames"] == frames >= 500
        assert got["agreement"] == agreed / frames == 1.0, got
        assert trainer.touch_agreement(inputs, model=opposed) == {"frames": frames, "agreement": 0.0}
        assert trainer.touch_agreement.__defaults__ == (trainer.DEFAULT_MODEL_PATH,)

        # a ball that nobody controls leaves no frame to judge
        loose = play_inputs(players, ball + np.array([30.0, 0.0]), teams=teams, graded=graded)
        loose.game = "BAR-ATH"
        assert trainer.touch_agreement(loose, model=aligned) == {"frames": 0, "agreement": None}


def test_shape_touch_agreement_uses_the_conventions_rule_and_the_fold():
    players, teams, ball, graded = touch_scene()
    with tempfile.TemporaryDirectory() as root:
        path = touch_model(root, "m.json", fold="FOLD1")                          # FOLD1's stump says the opposite
        inputs = play_inputs(players, ball, teams=teams, graded=graded)
        inputs.game = "BAR-ATH"
        assert trainer.touch_agreement(inputs, model=path)["agreement"] == 1.0, "BAR-ATH is scored by full"
        inputs.game = "FOLD1"
        assert trainer.touch_agreement(inputs, model=path)["agreement"] == 0.0, "a benchmark match by its fold"
        # the episodes are the conventions' rule, whatever it is called with
        seen, real = [], controlled_touch.compute

        def spy(run, **values):
            seen.append(values)
            return real(run, **values)
        controlled_touch.compute = spy
        try:
            trainer.touch_agreement(inputs, model=path)
        finally:
            controlled_touch.compute = real
        assert seen == [conventions.TOUCH], seen
        assert conventions.TOUCH == {"radius": 2.0, "max_rel_speed": 6.0, "min_frames": 3, "confidence": 0.99,
                                     "hold": 0.5}
        expect_error(ValueError, lambda: trainer.touch_agreement(inputs, model=os.path.join(root, "nope.json")),
                     "shape_possession", "train_shape.py")


# -- the committed model ----------------------------------------------------------------------------------------------

def test_committed_shape_model_holds_a_fold_for_each_benchmark_match():
    """A missing or renamed fold would fall back to `full`, which saw every match, and score it in sample silently."""
    path = trainer.DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise NotRun("{} is not trained yet; run `python pipeline/benchmark/train_shape.py` and commit it".format(
            os.path.relpath(path, trainer.ROOT)))
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    name = os.path.basename(path)
    assert sorted(artifact["folds"]) == sorted(sources.MATCH_IDS), (
        "{}: folds {} are not the seven benchmark matches".format(name, sorted(artifact["folds"])))
    assert "full" in artifact, "{}: no `full` model".format(name)
    shape_possession.load(path)
