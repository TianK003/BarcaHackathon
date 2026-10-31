"""Checks of P19's team-model trainer (pipeline/benchmark/train_team.py) and what it writes.

The trainer is train_shape's with `team_features`' 56 columns and a row only where the served ball is visible: it fits
P05's classifier on the benchmark's degraded tracking to say which team has the ball, shows every row twice (as it is,
and with the teams exchanged and the label flipped), and writes one model per held-out match and `full` as JSON, which
pipeline/evidence/team_model.py runs with numpy alone. What is held against it here:

* the rows are what the Design says: every `stride`-th frame of a graded block from its start, kept where the truth is
  H or A and the served ball is visible, 1 for H (test_team_rows_take_every_stride_th_alive_visible_frame); no truth
  reaches the features (test_team_trainer_never_passes_truth); the folds never see the match they are scored on
  (test_team_trainer_folds_exclude_the_held_out_match);
* the augmentation balances the classes exactly (test_team_augmentation_is_balanced); a model is the export of the fit
  on those rows with P05's settings (test_team_fit_is_p05s_settings_on_the_augmented_rows); the export is the model, to
  1e-12 and NaN included (test_team_export_matches_sklearn); the same bytes on 1 and 4 threads
  (test_team_training_is_deterministic_across_threads);
* the file is the whole chain (test_team_artifact_round_trip) and `--verify` fails on anything but source hashes
  (test_team_verify_ignores_only_source_hashes);
* the dry run is aggregates and flags the columns that cannot carry information
  (test_team_dry_run_is_aggregates_and_flags_dead_columns); it exits 1 on a dead column, on a row count outside the
  declared window and on a projected peak over 8 GB (test_team_dry_run_exit_code_and_pause_gates); the peak-memory
  helper returns a positive int or None;
* the feasibility counts the declared frames, worked out on a hand-built scene with the truth's D frames, the decoder's
  D and X frames, a silent rule frame, a hold frame, an episode frame and an over-11 frame, the expected counts written
  as literals (test_team_feasibility_counts_the_declared_frames_by_hand), and again by plain loops on a planted scene
  (test_team_feasibility_counts_the_declared_frames); its go rule is an integer test
  (test_team_feasibility_go_rule_is_integer_and_mechanical); a match with no frame is no win; it refuses a missing
  fold, reads the rule from the config, records the hashes of what it scored, and returns aggregates.

Everything runs on arrays and synthetic matches made here (checks/common.py): none reads data, and none of the numbers
printed is a figure of any match. The one fixture that takes a few seconds, three matches whose truth is a function of
the ball's nearest team, is trained once and shared; a check that changes an artifact copies it first.
"""

import contextlib
import copy
import dataclasses
import functools
import hashlib
import io
import json
import os
import tempfile

import numpy as np

from pipeline.benchmark import run, sources, train_dead_ball
from pipeline.benchmark import train_team as trainer
from pipeline.benchmark import truth
from pipeline.benchmark.calibration import digest as calibration_digest
from pipeline.benchmark.checks.common import (A, H, leaks, random_match, strict_json, synthetic_calibration,
                                              synthetic_matches, tracked_match, tracking_calibration)
from pipeline.benchmark.run import BenchmarkError
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics, team_features, team_model, trees
from pipeline.selftest import expect_error, team_model_stub, write_text

# Params
NAMES = team_features.FEATURE_NAMES
COLUMN = team_features.COLUMN
FOLD_IDS = ("RND001", "RND002", "RND003")
PLANTED_ROWS = (3500, 3000)                  # graded frames of a planted match: enough rows for a fold to split
PLANTED_SEEDS, PLANTED_STRIDE = 3, 5
CONFIG_TEXT = """
description = "a rule and a decoder, for the feasibility's checks"
seed = 0
scope = "halves"

[[evidence]]
kind = "controlled_touch"
radius = 3.0
max_rel_speed = inf
min_frames = 1
confidence = 0.9
hold = {hold}

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 32
min_possession = 35
min_dead = 25
"""


# -- helpers ----------------------------------------------------------------------------------------------------------

def team_rows(n=6000, seed=0, nan_share=0.15):
    """(X, y) of n rows over the 56 features, NaN in a share of the cells, and a label that follows which team's
    nearest slot is closer to the ball: the antisymmetric structure that the augmentation is there to teach."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES))) * 5.0
    X[:, COLUMN["h_ball_d1"]], X[:, COLUMN["a_ball_d1"]] = rng.uniform(0, 20, n), rng.uniform(0, 20, n)
    X[rng.random(X.shape) < nan_share] = np.nan
    score = (np.nan_to_num(X[:, COLUMN["a_ball_d1"]], nan=10.0) - np.nan_to_num(X[:, COLUMN["h_ball_d1"]], nan=10.0)
             + np.nan_to_num(X[:, COLUMN["adv_gap"]]) * 0.2 + rng.normal(size=n))
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


def quietly(function, *arguments, **keywords):
    """(result, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments, **keywords)
    return result, buffer.getvalue()


def planted_match(match_id, seed):
    """A DflMatch whose truth is a function of the ball's nearest team: home has the ball when its nearest slot is the
    closer one (the home team when the away team has no slot on the frame). Every graded frame is alive. The `gaps`
    profile leaves the players alone and only hides the ball more, so every seed of it carries the same geometry."""
    base = random_match(match_id, seed=seed, rows=PLANTED_ROWS, gap=25)
    inputs = trainer.served_inputs(base, "clean", 0, None, None)
    X = team_features.features(inputs)[1]
    h, a = X[:, COLUMN["h_ball_d1"]], X[:, COLUMN["a_ball_d1"]]
    with np.errstate(invalid="ignore"):
        home = np.where(np.isnan(a), True, np.where(np.isnan(h), False, h < a))
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


def small_artifact():
    """A small artifact of stumps with a provenance made by hand: what the verify and the file checks need."""
    model = team_model_stub(-0.2, 1.0, -1.0)
    return {"format": 1, "kind": "hgb", "feature_version": team_features.FEATURE_VERSION, "features": list(NAMES),
            "profile": "tracking", "seeds": 3, "stride": 5,
            "provenance": {"scikit_learn": "1.9.1", "calibration_sha256": "ab" * 32,
                           "dfl_sources": {"J03WMX": {"information": "01" * 32, "positions": "02" * 32}},
                           "source_sha256": {name: "aa" * 32 for name, _ in trainer.SOURCE_FILES}},
            "folds": {"J03WMX": model}, "full": team_model_stub(0.7, -1.0, 2.0)}


def config_file(folder, hold, name=None):
    return write_text(folder, name or "rule_{}.toml".format(str(hold).replace(".", "")), CONFIG_TEXT.format(hold=hold))


# -- the rows and the augmentation ------------------------------------------------------------------------------------

def test_team_rows_take_every_stride_th_alive_visible_frame():
    match = random_match("RND001", rows=(103, 60), gap=25)
    inputs = trainer.served_inputs(match, "clean", 0, None, None)
    codes, _ = truth.truth_codes(match)
    X, y = trainer.rows(inputs, codes, 5)
    visible = kinematics.ball_track(inputs).visible
    on_grid = np.concatenate([np.arange(start, stop, 5) for _, start, stop in match.graded])
    assert len(on_grid) == -(-103 // 5) + -(-60 // 5)
    kept = [frame for frame in on_grid if codes[frame] in (H, A) and visible[frame]]
    unseen = [frame for frame in on_grid if codes[frame] in (H, A) and not visible[frame]]
    assert 0 < len(kept) and unseen, "the match must have alive frames with the ball seen, and alive ones without"
    assert X.shape == (len(kept), len(NAMES)) and y.shape == (len(kept),)
    assert np.array_equal(X, team_features.features(inputs)[1][kept], equal_nan=True)
    assert np.array_equal(y, np.array([codes[frame] == H for frame in kept], dtype=y.dtype))
    assert y.dtype.kind in "iu" and set(np.unique(y)) == {0, 1}
    assert not np.isnan(X[:, COLUMN["ball_line_m"]]).any(), "every row has a ball: a missing-ball frame is no row"
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, 0), "train_team", "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes, True), "stride")
    expect_error(ValueError, lambda: trainer.rows(inputs, codes[:-1], 5), "codes")


def test_team_augmentation_is_balanced():
    X, y = team_rows(900, seed=3)
    y = (np.arange(len(y)) % 5 == 0).astype(np.int64)                      # 20% homes: the rows are not balanced
    n = len(X)
    Xa, ya = trainer.augment(X, y)
    assert Xa.shape == (2 * n, len(NAMES)) and ya.shape == (2 * n,) and ya.dtype == np.int64
    assert int(ya.sum()) * 2 == len(ya) and float(ya.mean()) == 0.5, "exactly half the labels are 1"
    assert np.array_equal(Xa[:n], X, equal_nan=True) and np.array_equal(ya[:n], y), "the rows come first, unchanged"
    for i in range(n):
        assert np.array_equal(Xa[n + i], team_features.swap(Xa[i]), equal_nan=True), i
        assert ya[n + i] == 1 - ya[i], i
    expect_error(ValueError, lambda: trainer.augment(X[:, :26], y), "train_team", "56")
    expect_error(ValueError, lambda: trainer.augment(X, y[:-1]), "train_team")
    expect_error(ValueError, lambda: trainer.augment(X, y + 1), "train_team", "0 or 1")


# -- the fit and the export -------------------------------------------------------------------------------------------

def test_team_fit_is_p05s_settings_on_the_augmented_rows():
    assert trainer.HGB_SETTINGS is train_dead_ball.HGB_SETTINGS, "P05's settings, imported and not retyped"
    X, y = team_rows(3000, seed=5)
    Xa, ya = trainer.augment(X, y)
    estimator = trainer.hgb_estimator(Xa, ya)
    assert estimator.monotonic_cst is None and estimator.early_stopping is False
    assert estimator.n_iter_ == trainer.HGB_SETTINGS["max_iter"] and estimator.n_features_in_ == len(NAMES)
    model = trainer.fit(X, y)
    assert model == {**trainer.export_hgb(estimator), "prior_home": 0.5}
    assert sorted(model) == ["baseline", "prior_home", "trees"]
    assert abs(model["baseline"]) < 1e-12, "the rows and their mirrors are balanced: the baseline is log-odds 0"
    expect_error(ValueError, lambda: trainer.fit(X, np.ones(len(X), dtype=np.int64)), "train_team", "both")
    expect_error(ValueError, lambda: trainer.fit(X[:, :20], y), "train_team", "56")


def test_team_export_matches_sklearn():
    X, y = team_rows(6000, seed=1)
    Xa, ya = trainer.augment(X, y)
    estimator = trainer.hgb_estimator(Xa, ya)
    model = trainer.fit(X, y)
    assert len(model["trees"]) == 300 and any(len(tree["feature"]) > 3 for tree in model["trees"])
    probe, probe_y = team_rows(5000, seed=2, nan_share=0.25)                # rows the model never saw, with more NaN
    for rows in (Xa, probe, team_features.swap(probe)):
        worst = float(np.abs(trees.score(model, rows) - estimator.decision_function(rows)).max())
        print("      exported trees against sklearn: max |difference| {:.3e} on {:,} rows".format(worst, len(rows)))
        assert worst <= 1e-12, worst
    trees.check_trees(model, len(NAMES), "a fold of m.json", "team_model")
    json.dumps(model, allow_nan=False)
    p = 1.0 / (1.0 + np.exp(-estimator.decision_function(probe)))
    assert float(np.mean((p >= 0.5) == (probe_y == 1))) > 0.7, "the fit must have learnt something"


def test_team_training_is_deterministic_across_threads():
    from threadpoolctl import threadpool_limits
    X, y = team_rows(5000, seed=4)
    texts = []
    for limit in (1, 4):
        with threadpool_limits(limits=limit):
            texts.append(trainer.dumps({"model": trainer.fit(X, y)}))
    assert texts[0] == texts[1], "the same fit on 1 and on 4 threads wrote different bytes"


# -- gathering, folds, provenance -------------------------------------------------------------------------------------

def test_team_trainer_folds_exclude_the_held_out_match():
    matches = three_matches()
    calls = []
    real = trainer.fit
    trainer.fit = recording_fit(calls)
    try:
        artifact = trainer.train("gaps", 2, 7, matches=matches, calibration=synthetic_calibration())
    finally:
        trainer.fit = real
    data = trainer.gather("gaps", 2, 7, matches=matches, calibration=synthetic_calibration()).data
    assert sorted(data) == list(FOLD_IDS) and sorted(artifact["folds"]) == list(FOLD_IDS)
    assert artifact["full"]["rows"] == sum(len(y) for _, y in data.values())
    assert artifact["features"] == list(NAMES) and artifact["feature_version"] == team_features.FEATURE_VERSION
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
        seen = {row.tobytes() for row in np.nan_to_num(hits[0][0], nan=-9.99e9)}
        assert not any(row.tobytes() in seen for row in np.nan_to_num(data[match_id][0], nan=-9.99e9))
    everything = np.concatenate([data[name][0] for name in FOLD_IDS])
    assert any(np.array_equal(call[0], everything, equal_nan=True) for call in calls), "full saw every match"
    expect_error(ValueError, lambda: trainer.train("gaps", 1, 5, calibration=synthetic_calibration(),
                                                   matches={"RND001": random_match("RND001")}), "at least two")
    expect_error(ValueError, lambda: trainer.train("nope", 1, 5, matches=three_matches()), "nope")
    expect_error(ValueError, lambda: trainer.train("clean", 1, 5, matches=three_matches(),
                                                   calibration=synthetic_calibration()), "clean", "calibration")


def test_team_trainer_never_passes_truth():
    for profile, calibration, matches in (
            ("tracking", tracking_calibration(), {"RND001": tracked_match("RND001"),
                                                  "RND002": tracked_match("RND002", 1)}),
            ("gaps", synthetic_calibration(), synthetic_matches())):
        seen, real = [], team_features.features
        match = matches["RND001"]

        def spy(inputs):
            seen.append([leaks(inputs, other) for other in matches.values()])      # against every match's truth
            return real(inputs)
        team_features.features = spy
        real_fit = trainer.fit
        trainer.fit = recording_fit([])
        try:
            trainer.train(profile, 1, 9, calibration=calibration, matches=matches)
        finally:
            trainer.fit = real_fit
            team_features.features = real
        assert seen, "the trainer never asked for features"
        assert all(found == [[] for _ in matches] for found in seen), (profile, seen)
        inputs = trainer.served_inputs(match, profile, 0, calibration, None)
        assert all(leaks(inputs, other) == [] for other in matches.values())


def test_team_provenance_holds_versions_and_hashes_and_no_commit():
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
    assert provenance["scikit_learn"] == sklearn.__version__ and provenance["calibration_sha256"] is None
    assert sorted(provenance["dfl_sources"]) == list(FOLD_IDS)
    assert sorted(provenance["source_sha256"]) == sorted(
        ["train_team.py", "team_features.py", "shape_features.py", "kinematics.py", "controlled_touch.py", "trees.py"])
    folder = os.path.dirname(os.path.abspath(trainer.__file__))
    evidence = os.path.join(os.path.dirname(folder), "evidence")
    for name, path in trainer.SOURCE_FILES:
        assert os.path.dirname(path) in (folder, evidence), name
        with open(path, "rb") as handle:
            text = handle.read().replace(b"\r\n", b"\n")
        assert provenance["source_sha256"][name] == hashlib.sha256(text).hexdigest(), name
    assert not any("commit" in key or "git" in key for key in provenance)
    assert gapped["provenance"]["calibration_sha256"] == calibration_digest(calibration)


# -- the file ---------------------------------------------------------------------------------------------------------

def test_team_artifact_round_trip():
    matches, artifact = planted()
    calibration = synthetic_calibration()
    assert sorted(artifact) == sorted(["format", "kind", "feature_version", "features", "profile", "seeds", "stride",
                                       "provenance", "folds", "full"])
    assert (artifact["format"], artifact["kind"], artifact["profile"], artifact["seeds"], artifact["stride"]) == (
        trees.FORMAT, "hgb", "gaps", PLANTED_SEEDS, PLANTED_STRIDE)
    assert artifact["feature_version"] == team_features.FEATURE_VERSION and artifact["features"] == list(NAMES)
    assert sorted(artifact["folds"]) == list(FOLD_IDS)
    for name, model in list(artifact["folds"].items()) + [("full", artifact["full"])]:
        assert sorted(model) == ["baseline", "prior_home", "trees"] and model["prior_home"] == 0.5, name
        assert any(len(tree["feature"]) > 1 for tree in model["trees"]), "{} grew no split".format(name)
        trees.check_trees(model, len(NAMES), name, "team_model")

    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "models", "team.json")
        trainer.write(path, artifact)
        with open(path, "rb") as handle:
            data = handle.read()
        text = data.decode("utf-8")
        assert text == trainer.dumps(artifact) and text.endswith("\n") and text.count("\n") == 1
        assert b"\r" not in data and len(data) < 10 * 1000 * 1000
        assert strict_json(text) == artifact and json.loads(text) == artifact, "floats must read back exactly"
        assert team_model.load(path) == artifact
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
            home = team_model.compute(RunContext(inputs=inputs, seed=0), model=path).home
            X = team_features.features(inputs)[1]
            speaks = kinematics.ball_track(inputs).visible
            rows = X[speaks]
            logit = (estimator.decision_function(rows) - estimator.decision_function(team_features.swap(rows))) / 2.0
            want = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98)
            assert np.isnan(home[~speaks]).all() and speaks.sum() > 500
            worst = float(np.abs(home[speaks] - want).max())
            print("      {}: the source against scikit-learn's own fold: max |difference| {:.3e}".format(
                held_out, worst))
            assert worst <= 1e-12, worst
    again = trainer.train("gaps", PLANTED_SEEDS, PLANTED_STRIDE, matches=matches, calibration=calibration)
    assert trainer.dumps(again) == trainer.dumps(artifact), "retraining wrote different bytes"
    assert trainer.compare(artifact, again) == ([], [])
    with tempfile.TemporaryDirectory() as root:
        big = {"padding": ["x" * 1000] * (train_dead_ball.MAX_BYTES // 1000 + 10)}
        expect_error(ValueError, lambda: trainer.write(os.path.join(root, "big.json"), big), "10")


def test_team_verify_ignores_only_source_hashes():
    committed = small_artifact()
    fresh = json.loads(trainer.dumps(committed))
    assert trainer.compare(committed, fresh) == ([], [])
    fresh["provenance"]["source_sha256"]["kinematics.py"] = "dd" * 32
    problems, notes = trainer.compare(committed, fresh)
    assert problems == [] and len(notes) == 1 and "kinematics.py" in notes[0], (problems, notes)

    def changed(edit):
        other = json.loads(trainer.dumps(committed))
        edit(other)
        return trainer.compare(committed, other)[0]
    assert changed(lambda a: a["provenance"].update(scikit_learn="1.9.2"))
    assert changed(lambda a: a["provenance"].update(calibration_sha256="00" * 32))
    assert changed(lambda a: a["folds"]["J03WMX"].update(baseline=a["folds"]["J03WMX"]["baseline"] + 1e-12))
    assert changed(lambda a: a["full"]["trees"][0]["value"].__setitem__(1, 0.123))
    assert changed(lambda a: a.update(stride=6)) and changed(lambda a: a.update(seeds=2))
    assert changed(lambda a: a["provenance"]["source_sha256"].pop("trees.py")), "a missing hash is not a note"
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "team_model_hgb.json")
        trainer.write(path, committed)
        code, text = quietly(trainer.verify, path, fresh)
        assert code == 0 and "note" in text.lower() and "kinematics.py" in text
        other = json.loads(trainer.dumps(committed))
        other["full"]["baseline"] += 1e-9
        code, text = quietly(trainer.verify, path, other)
        assert code == 1 and "full" in text and "baseline" in text
        code, text = quietly(trainer.verify, os.path.join(root, "missing.json"), fresh)
        assert code == 1 and "missing.json" in text and "pipeline/benchmark/train_team.py" in text


def test_team_command_line_is_the_designs():
    parser = trainer.build_parser()
    args = parser.parse_args([])
    assert (args.profile, args.seeds, args.stride, args.out, args.verify, args.dry_run, args.feasibility,
            args.matches, args.data_dir) == ("tracking", 3, 5, None, False, False, False, None, None)
    assert args.config.replace("\\", "/").endswith("pipeline/configs/main.toml")
    args = parser.parse_args(["--dry-run", "--matches", "J03WMX", "J03WN1", "--seeds", "1"])
    assert args.dry_run and args.matches == ["J03WMX", "J03WN1"] and args.seeds == 1
    assert trainer.DEFAULT_MODEL_PATH.replace("\\", "/").endswith("pipeline/models/team_model_hgb.json")
    for bad in (["--verify", "--feasibility"], ["--feasibility", "--stride", "3"], ["--matches", "J03WMX"],
                ["--dry-run", "--verify"], ["--dry-run", "--feasibility"], ["--config", "x.toml"], ["--seeds", "0"],
                ["--stride", "0"], ["--profile"], ["--model", "hgb"]):
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            try:
                trainer.main(bad)
            except SystemExit as stop:
                assert stop.code != 0, bad
            else:
                raise AssertionError("{} was accepted".format(bad))


def test_team_main_trains_writes_and_verifies():
    """The two commands the controller runs after the dry run, on seven synthetic matches under the DFL ids."""
    from pipeline.benchmark.checks.common import loading
    matches = {match_id: random_match(match_id, seed=seed, rows=(700, 600)) for seed, match_id
               in enumerate(sources.MATCH_IDS)}
    with tempfile.TemporaryDirectory() as root, loading(lambda match_id: matches[match_id]):
        path = os.path.join(root, "models", "team.json")
        flags = ["--profile", "clean", "--seeds", "1", "--stride", "2", "--out", path]
        code, text = quietly(trainer.main, flags)
        assert code == 0 and "wrote" in text and os.path.exists(path), text
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        assert sorted(artifact["folds"]) == sorted(sources.MATCH_IDS) and "full" in artifact
        team_model.load(path)
        code, text = quietly(trainer.main, flags + ["--verify"])
        assert code == 0 and "verified" in text, text
        artifact["full"]["baseline"] += 1e-9
        trainer.write(path, artifact)
        code, text = quietly(trainer.main, flags + ["--verify"])
        assert code == 1 and "FAILED" in text and "baseline" in text, text


# -- the dry run and the memory ---------------------------------------------------------------------------------------

def test_team_peak_memory_is_a_positive_int_or_none():
    peak = trainer.peak_memory_bytes()
    assert peak is None or (isinstance(peak, int) and not isinstance(peak, bool) and peak > 10 ** 6), peak
    block = np.ones(4_000_000)                                                   # 32 MB: the peak can only rise
    again = trainer.peak_memory_bytes()
    assert again is None or peak is None or again >= peak
    del block


def test_team_dry_run_is_aggregates_and_flags_dead_columns():
    matches = three_matches()
    calibration = synthetic_calibration()
    report, text = quietly(trainer.dry_run, "gaps", 2, ["RND001"], 5, matches=matches, calibration=calibration,
                           log=print)
    json.dumps(report, allow_nan=False)
    assert sorted(report) == sorted(["profile", "seeds", "stride", "per_match", "rows", "nan_share", "all_nan",
                                     "constant", "constant_values", "peak_memory_bytes", "projected_peak_gb",
                                     "projection_basis", "checks", "ok"])
    gathered = trainer.gather("gaps", 2, 5, matches=matches, calibration=calibration, match_ids=["RND001",
                                                                                              "RND002"]).data
    assert sorted(report["per_match"]) == ["RND001"] and report["rows"] == len(gathered["RND001"][1])
    cell = report["per_match"]["RND001"]
    assert cell["rows"] == report["rows"] and cell["x_bytes"] == gathered["RND001"][0].nbytes
    assert cell["home_share"] == float(gathered["RND001"][1].mean())
    assert list(report["nan_share"]) == list(NAMES) and all(0.0 <= v <= 1.0 for v in report["nan_share"].values())
    X = gathered["RND001"][0]
    assert abs(report["nan_share"]["h_ball_d1"] - float(np.isnan(X[:, COLUMN["h_ball_d1"]]).mean())) < 1e-12
    assert "h_over11" in report["constant"] and "a_over11" in report["constant"], "3 slots a side are never over 11"
    extra = 196_283 * 2 * 56 * 8 * 4 / 1e9
    baseline = report["peak_memory_bytes"] if report["peak_memory_bytes"] is not None else X.nbytes
    assert abs(report["projected_peak_gb"] - (baseline / 1e9 + extra)) < 1e-9
    assert "RND001" in text, "the log line of the match"
    # an all-NaN column is named, and the command exits 1 on either kind
    real = team_features.features
    names, base = real(trainer.served_inputs(matches["RND001"], "clean", 0, None, None))
    team_features.features = lambda inputs: (names, np.where(np.arange(len(names)) == COLUMN["ball_line_m"], np.nan,
                                                              real(inputs)[1]))
    try:
        dead = trainer.dry_run("clean", 1, ["RND001"], 5, matches=matches)
    finally:
        team_features.features = real
    assert dead["all_nan"] == ["ball_line_m"] and dead["nan_share"]["ball_line_m"] == 1.0
    expect_error(ValueError, lambda: trainer.dry_run("gaps", 0, ["RND001"], 5, matches=matches), "seeds")
    lines = trainer.render_dry_run(report)
    assert any("constant columns" in line and "h_over11" in line for line in lines)


# -- the feasibility --------------------------------------------------------------------------------------------------

def feasibility_counts(config, matches, artifact, profile, seeds, calibration):
    """The cells again, frame by frame in plain loops from the same decoded codes, home and rule arrays."""
    expected = {}
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "team_model_hgb.json")
        trainer.write(path, artifact)
        touch = [c for c in config.evidence if c.kind == "controlled_touch"][0].params
        for match_id, match in sorted(matches.items()):
            codes, _ = truth.truth_codes(match)
            coarse = truth.coarse_codes(codes, match.graded, match.fps)
            cell = {"frames": 0, "p19": 0, "main": 0, "rule": 0, "silent": 0, "episode": 0, "hold": 0,
                    "rule_silent_class": 0, "coarse_frames": 0, "coarse_p19": 0, "coarse_main": 0, "over11": 0}
            for seed in range(seeds):
                inputs = trainer.served_inputs(match, profile, seed, calibration, None)
                run_context = RunContext(inputs=inputs, seed=0)
                decoded = run.predict(config, inputs)[0]
                home = team_model.compute(run_context, model=path).home
                rule = controlled_touch.compute(run_context, **touch).home
                X = team_features.features(inputs)[1]
                visible = kinematics.ball_track(inputs).visible
                for t in range(len(decoded)):
                    if not (visible[t] and decoded[t] in (H, A) and not np.isnan(home[t])):
                        continue
                    for name, source in (("", codes), ("coarse_", coarse)):
                        if source[t] not in (H, A):
                            continue
                        truly_home = source[t] == H
                        if name:
                            cell["coarse_frames"] += 1
                            cell["coarse_p19"] += int((home[t] >= 0.5) == truly_home)
                            cell["coarse_main"] += int((decoded[t] == H) == truly_home)
                            continue
                        cell["frames"] += 1
                        cell["p19"] += int((home[t] >= 0.5) == truly_home)
                        cell["main"] += int((decoded[t] == H) == truly_home)
                        if np.isnan(rule[t]):
                            cell["silent"] += 1
                        else:
                            cell["rule"] += int((rule[t] >= 0.5) == truly_home)
                            cell["episode" if abs(rule[t] - 0.5) > touch["hold"] - 0.5 + 1e-9 else "hold"] += 1
                        cell["over11"] += int(X[t, COLUMN["h_over11"]] == 1.0 or X[t, COLUMN["a_over11"]] == 1.0)
            expected[match_id] = cell
    return expected


def test_team_feasibility_counts_the_declared_frames():
    matches, artifact = planted()
    calibration = synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        path = config_file(root, 0.6)
        report = trainer.feasibility(artifact, path, "gaps", 2, matches=matches, calibration=calibration)
        from pipeline.config import load_config
        want = feasibility_counts(run.drop_match_only(load_config(path))[0], matches, artifact, "gaps", 2, calibration)
    assert sorted(report["per_match"]) == list(FOLD_IDS) and report["config"] == "rule_06"
    for match_id, cell in report["per_match"].items():
        w = want[match_id]
        assert cell["frames"] == w["frames"] > 100, match_id
        assert (cell["p19"]["right"], cell["main"]["right"], cell["rule"]["right"], cell["rule"]["silent"]) == (
            w["p19"], w["main"], w["rule"], w["silent"]), match_id
        assert cell["classes"]["episode"]["frames"] == w["episode"] and cell["classes"]["hold"]["frames"] == w["hold"]
        assert cell["classes"]["silent"]["frames"] == w["silent"]
        assert cell["classes"]["over11"]["frames"] == w["over11"]
        assert cell["classes"]["over11"]["frames"] + cell["classes"]["not_over11"]["frames"] == cell["frames"]
        assert (cell["coarse"]["frames"], cell["coarse"]["p19"]["right"], cell["coarse"]["main"]["right"]) == (
            w["coarse_frames"], w["coarse_p19"], w["coarse_main"]), match_id
        assert cell["p19"]["accuracy"] == cell["p19"]["right"] / cell["frames"]
        assert cell["rule"]["log_loss"] is not None and cell["p19"]["log_loss"] is not None
        assert cell["classes"]["episode"]["frames"] + cell["classes"]["hold"]["frames"] + cell["rule"]["silent"] \
            == cell["frames"], "the rule's three states partition the frames"
    assert report["pooled"]["frames"] == sum(c["frames"] for c in report["per_match"].values())
    assert report["go"] == trainer.go(report["per_match"])
    # the silent frame's log-loss is ln 2 each: a rule that is silent everywhere scores exactly ln 2
    silent = trainer._score(np.array([True, True, False]), np.array([True, False, True]), np.array([0.9, 0.2, 0.5]),
                            np.array([True, True, True]), np.array([np.nan, np.nan, 0.9]))
    assert silent["rule"][0] == 0 and silent["rule"][2] == 2 and abs(silent["rule"][1] - 2 * np.log(2.0)) < 1e-12
    assert silent["frames"] == 2 and silent["p19"][0] == 2 and silent["main"] == 1


def cells_of(*gains, frames=1000):
    return {"M{}".format(i): {"frames": frames, "p19": {"right": 500 + gain}, "main": {"right": 500}}
            for i, gain in enumerate(gains)}


def test_team_feasibility_go_rule_is_integer_and_mechanical():
    # exactly +0.020 passes and +0.0199 fails, per match and pooled
    assert trainer.go(cells_of(*[20] * 7)) == {"wins": 7, "pooled": True, "go": True}
    edge = cells_of(*[199] * 7, frames=10000)
    assert trainer.go(edge) == {"wins": 0, "pooled": False, "go": False}
    assert trainer.go(cells_of(*[200] * 7, frames=10000))["go"] is True
    # 5 wins fail even when the pool passes
    five = trainer.go(cells_of(100, 100, 100, 100, 100, 0, 0))
    assert five == {"wins": 5, "pooled": True, "go": False}, five
    # 6 wins with the pool failing fail
    six = trainer.go(cells_of(20, 20, 20, 20, 20, 20, -500))
    assert six == {"wins": 6, "pooled": False, "go": False}, six
    assert trainer.go(cells_of(40, 40, 40, 40, 40, 40, -100)) == {"wins": 6, "pooled": True, "go": True}
    assert trainer.go(cells_of(40, 40, 40, 40, 40, 40, -101)) == {"wins": 6, "pooled": False, "go": False}, \
        "the pool is the integer test as well: 50 * 140 = 7000 passes, 50 * 139 does not"
    assert trainer.GO_MATCHES == 6 and trainer.GO_NUMERATOR == 50


def test_team_feasibility_refuses_a_missing_fold():
    matches, artifact = planted()
    broken = copy.deepcopy(artifact)
    del broken["folds"]["RND002"]
    with tempfile.TemporaryDirectory() as root:
        path = config_file(root, 0.6)
        expect_error(BenchmarkError, lambda: trainer.feasibility(broken, path, "gaps", 1, matches=matches,
                                                                 calibration=synthetic_calibration()),
                     "RND002", "in sample", "full")
        expect_error(BenchmarkError, lambda: trainer.feasibility({"full": artifact["full"]}, path, "gaps", 1,
                                                                 matches=matches,
                                                                 calibration=synthetic_calibration()), "RND001")
        no_rule = write_text(root, "no_rule.toml", CONFIG_TEXT.replace("controlled_touch", "nearest_player")
                             .format(hold=0.6).replace("radius = 3.0\n", "").replace("max_rel_speed = inf\n", "")
                             .replace("min_frames = 1\n", "").replace("confidence = 0.9\n", "")
                             .replace("hold = 0.6\n", ""))
        expect_error(BenchmarkError, lambda: trainer.feasibility(artifact, no_rule, "gaps", 1, matches=matches,
                                                                 calibration=synthetic_calibration()),
                     "controlled_touch")


def test_team_feasibility_reads_the_rule_from_the_config():
    matches, artifact = planted()
    calibration = synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        strong = trainer.feasibility(artifact, config_file(root, 0.6), "gaps", 1, matches=matches,
                                     calibration=calibration)
        weak = trainer.feasibility(artifact, config_file(root, 0.55), "gaps", 1, matches=matches,
                                   calibration=calibration)
    # the hold is the config's own: the rule's value between episodes is 0.6 in one and 0.55 in the other, so its
    # log-loss differs, while a rule retyped in the trainer could not tell them apart
    assert strong["pooled"]["classes"]["hold"]["frames"] > 0 and weak["pooled"]["classes"]["hold"]["frames"] > 0
    assert strong["pooled"]["rule"]["log_loss"] != weak["pooled"]["rule"]["log_loss"]
    assert strong["config"] == "rule_06" and weak["config"] == "rule_055"
    # episodes are the confidence's frames, found by the confidence of the file (0.9), the hold's by its own value
    assert strong["pooled"]["classes"]["episode"]["frames"] > 0


def test_team_feasibility_is_aggregates():
    matches, artifact = planted()
    with tempfile.TemporaryDirectory() as root:
        report = trainer.feasibility(artifact, config_file(root, 0.6), "gaps", 1, matches=matches,
                                     calibration=synthetic_calibration())

    def leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from leaves(item)
        else:
            yield value
    for leaf in leaves(report):
        assert leaf is None or isinstance(leaf, (int, float, str, bool)), type(leaf)     # no list, no array
    json.dumps(report, allow_nan=False)
    assert max(cell["frames"] for cell in report["per_match"].values()) < 10 ** 6
    text = "\n".join(trainer.render_feasibility(report))
    assert "GO" in text and all(match_id in text for match_id in FOLD_IDS) and "pooled" in text


# -- the pause's gates, the exit code and the hand-built feasibility scene ---------------------------------------------

def varying_over11(real):
    """team_features.features with the columns that a small synthetic match leaves constant (3 slots a side never
    pass 11, and no episode is near enough) made to vary, so that no column is constant in it."""
    def features(inputs):
        names, X = real(inputs)
        X = np.array(X)
        for step, name in enumerate(("h_over11", "a_over11", "h_in_episode", "a_in_episode", "h_since_episode_s",
                                     "a_since_episode_s", "h_until_episode_s", "a_until_episode_s")):
            X[:, COLUMN[name]] = (np.arange(len(X)) // (step + 1)) % 3
        return names, X
    return features


def test_team_dry_run_exit_code_and_pause_gates():
    from pipeline.benchmark.checks.common import loading
    matches = {"J03WMX": random_match("J03WMX", rows=(700, 600)), "J03WN1": random_match("J03WN1", seed=1)}
    flags = ["--dry-run", "--profile", "clean", "--seeds", "1", "--matches", "J03WMX"]
    with loading(lambda match_id: matches[match_id]):
        code, text = quietly(trainer.main, flags)
        assert code == 1 and "FAILED" in text, text
        assert "h_over11 (always 0)" in text and "a_over11 (always 0)" in text, "a constant column's value is printed"
        # all-NaN and constant columns each fail the command on their own
        real = team_features.features
        team_features.features = varying_over11(real)
        try:
            code, text = quietly(trainer.main, flags)
            assert code == 0 and "constant columns: none" in text and "OK" in text, text
            varied = varying_over11(real)

            def blanked(inputs):
                names_, X_ = varied(inputs)
                X_ = np.array(X_)
                X_[:, COLUMN["ball_line_m"]] = np.nan
                return names_, X_
            team_features.features = blanked
            code, text = quietly(trainer.main, flags)
            assert code == 1 and "all-NaN columns: ball_line_m" in text, text
            team_features.features = varied
            # the gates: the row window and the projected peak are held, not only printed
            kwargs = dict(matches=matches, match_ids=["J03WMX"])
            base = trainer.dry_run("clean", 1, ["J03WMX"], 5, matches=matches)
            assert base["ok"] is True and base["checks"]["rows"] == {"window": None, "ok": None}
            rows = base["rows"]
            assert rows > 0
            inside = trainer.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, row_window=(rows, rows))
            assert inside["checks"]["rows"] == {"window": [rows, rows], "ok": True} and inside["ok"] is True
            for window in ((rows + 1, rows + 10), (0, rows - 1)):
                outside = trainer.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, row_window=window)
                assert outside["checks"]["rows"]["ok"] is False and outside["ok"] is False, window
            tight = trainer.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, peak_limit_gb=0.0)
            assert tight["checks"]["projection"] == {"limit_gb": 0.0, "ok": False} and tight["ok"] is False
            assert base["checks"]["projection"]["ok"] is True and base["projected_peak_gb"] < 8.0
            extra = trainer.TRAINING_ROWS * 2 * 56 * 8 * trainer.COPIES / 1e9
            floor = (base["peak_memory_bytes"] or base["per_match"]["J03WMX"]["x_bytes"]) / 1e9 + extra
            assert abs(base["projected_peak_gb"] - floor) < 1e-9, "the peak or the row bytes, plus the training set"
            assert base["projection_basis"] in ("peak", "row bytes")
            # the declared window is enforced for the declared run, whatever else is true of it
            declared = trainer.dry_run("tracking", 1, ["J03WMX"], 5, matches={"J03WMX": tracked_match("J03WMX")},
                                       calibration=tracking_calibration())
            assert declared["checks"]["rows"]["window"] == [8381, 12571] and declared["checks"]["rows"]["ok"] is False
            assert declared["ok"] is False, "a few hundred rows are not 10,476"
            lines = "\n".join(trainer.render_dry_run(declared))
            assert "FAIL, outside the window [8,381, 12,571]" in lines and "projected peak PASS" in lines
        finally:
            team_features.features = real
    assert trainer.ROW_WINDOW == (8381, 12571) and trainer.PEAK_LIMIT_GB == 8.0


class Stand:
    """What _seed_tally reads of an Inputs and of the sources, made of hand-built arrays."""


def hand_scene():
    """Twelve frames: (truth, coarse, decoded, ball visible, P19 home, rule value, over 11). See the test."""
    D, X = trainer.pm.D, trainer.pm.X
    rows = [
        (H, H, H, True, 0.9, 0.93, 0),            # 0 kept: P19 right, main right, rule right, an episode
        (A, H, H, True, 0.2, 0.07, 0),            # 1 kept: P19 right, main wrong, rule right, an episode (coarse H)
        (H, H, H, True, 0.3, 0.60, 0),            # 2 kept: P19 wrong, main right, rule right, the hold
        (A, A, A, True, 0.6, np.nan, 0),          # 3 kept: P19 wrong, main right, rule silent (wrong), silent
        (D, D, D, True, 0.5, 0.93, 0),            # 4 truth D and decoded D: out
        (H, H, D, True, 0.9, 0.93, 0),            # 5 decoded D: out (main's dead-ball error is not P19's credit)
        (D, D, H, True, 0.9, 0.93, 0),            # 6 truth D, decoded H: out
        (H, H, X, True, 0.9, 0.93, 0),            # 7 decoded X: out
        (H, H, H, False, np.nan, 0.93, 0),        # 8 the ball is not seen: out
        (A, A, A, True, 0.1, 0.07, 1),            # 9 kept: P19 right, main right, rule right, an episode, over 11
        (X, X, H, False, np.nan, np.nan, 0),      # 10 outside the graded blocks: out
        (H, H, H, True, np.nan, 0.93, 0)]         # 11 P19 has no opinion: out
    return [np.array([row[i] for row in rows]) for i in range(7)]


def test_team_feasibility_counts_the_declared_frames_by_hand():
    """The frames of the cell, worked out on twelve hand-built frames, the expected counts written as literals."""
    import math
    from types import SimpleNamespace
    truth, coarse, decoded, visible, home, rule, over = hand_scene()
    truth, coarse, decoded = (array.astype(np.int8) for array in (truth, coarse, decoded))
    X = np.zeros((12, len(NAMES)))
    X[:, COLUMN["h_over11"]] = over
    config = SimpleNamespace(stem="hand")
    inputs = Stand()
    patched = {(run, "predict"): lambda c, i: (decoded, None),
               (trainer.team_model, "compute"): lambda r, model: SimpleNamespace(home=home),
               (trainer.controlled_touch, "compute"): lambda r, **p: SimpleNamespace(home=rule),
               (trainer.kinematics, "ball_track"): lambda i: SimpleNamespace(visible=visible),
               (trainer.team_features, "features"): lambda i: (NAMES, X)}
    real = {key: getattr(*key) for key in patched}
    for (owner, name), function in patched.items():
        setattr(owner, name, function)
    try:
        part = trainer._seed_tally(config, {"confidence": 0.93, "hold": 0.6}, inputs, truth, coarse, "unused")
    finally:
        for (owner, name), function in real.items():
            setattr(owner, name, function)
    cell = trainer._cell({"raw": part["raw"], "coarse": part["coarse"], "classes": part["classes"]})
    # frames 0, 1, 2, 3 and 9 are the only ones with truth H or A, decoded H or A, the ball seen and P19 speaking
    assert cell["frames"] == 5
    assert (cell["p19"]["right"], cell["main"]["right"], cell["rule"]["right"], cell["rule"]["silent"]) == (3, 4, 4, 1)
    assert cell["rule"]["silent_share"] == 0.2
    assert abs(cell["rule"]["log_loss"] * 5 - (3 * -math.log(0.93) - math.log(0.6) + math.log(2.0))) < 1e-12
    assert cell["coarse"]["frames"] == 5 and (cell["coarse"]["p19"]["right"], cell["coarse"]["main"]["right"]) == (2, 5)
    classes = {name: (c["frames"], c["p19"]["right"], c["main"]["right"]) for name, c in cell["classes"].items()}
    assert classes == {"episode": (3, 3, 2), "hold": (1, 0, 1), "silent": (1, 0, 1), "over11": (1, 1, 1),
                       "not_over11": (4, 2, 3)}, classes
    # without the exclusions the same scene would count more: these are the frames the conditions remove
    assert int(((truth == H) | (truth == A)).sum()) == 9 and int(((decoded == H) | (decoded == A)).sum()) == 9


def test_team_feasibility_reports_silent_share_hashes_and_refuses_empty_wins():
    import hashlib
    matches, artifact = planted()
    with tempfile.TemporaryDirectory() as root:
        path = config_file(root, 0.6)
        report = trainer.feasibility(artifact, path, "gaps", 1, matches=matches, calibration=synthetic_calibration())
        with open(path, "rb") as handle:
            assert report["config_sha256"] == hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    assert report["model_sha256"] == hashlib.sha256(trainer.dumps(artifact).encode("utf-8")).hexdigest()
    for cell in list(report["per_match"].values()) + [report["pooled"]]:
        rule = cell["rule"]
        assert rule["silent_share"] == rule["silent"] / cell["frames"]
    assert "(" in "\n".join(trainer.render_feasibility(report)), "the silent share is printed beside the count"
    # a match with no frame is not a win, even though 50 * 0 >= 0
    empty = {"frames": 0, "p19": {"right": 0}, "main": {"right": 0}}
    assert trainer.go(dict(cells_of(*[20] * 6), E=empty)) == {"wins": 6, "pooled": True, "go": True}
    assert trainer.go(dict(cells_of(*[20] * 5), E=empty, F=empty))["wins"] == 5
    assert trainer._gain(empty) is False
