"""Checks of P35, a turnover hazard read by the hmm decoder as a per-frame switch cost: the decoder's checks (A1, the
per-frame team cost, named test_p35_decoder_...) and the model's (A2, named test_p35_model_...: the trainer
train_turnover.py, its report, the turnover_hazard source's artifact and the four configs of configs/p35/).

The two halves come from two lanes and are concatenated: the decoder's at the top, the model's below under their own
header and imports. Everything runs on arrays and synthetic matches made here (checks/common.py) and on config files;
none reads data (docs/superpowers/plans/2026-11-13-p35-p39.md, Tasks 13 and 14).
"""


# -- the turnover hazard's trainer, report and configs (Task 14, A2): test_p35_model_* ------------------------------------
#
# What is held against `train_turnover.py` (cloned from train_team.py) and the four configs, on arrays and synthetic
# matches made here (checks/common.py) and on config files: none reads data, and none of the numbers printed is a figure
# of any match. No check needs pipeline/models/turnover_hgb.json, which the controller trains. The source's own checks
# (values, swap invariance, NaN where the ball is hidden, the fold choice, bad files) are in pipeline/selftest.py.

import contextlib
import copy
import dataclasses
import functools
import glob
import hashlib
import io
import json
import os
import tempfile
import tomllib

import numpy as np

from pipeline.benchmark import train_dead_ball, train_team, truth
from pipeline.benchmark import train_turnover as turnover
from pipeline.benchmark.checks.common import leaks, loading, random_match, strict_json, synthetic_matches
from pipeline.contracts import RunContext
from pipeline.evidence import kinematics, team_features, trees, turnover_hazard
from pipeline.selftest import expect_error

TURNOVER_IDS = ("RND001", "RND002", "RND003")
TURNOVER_ROWS = (1300, 1100)                    # graded frames of a hazard match: rows enough for a fold to split
TURNOVER_CONFIGS = {"p35_s000": 0.0, "p35_s050": 0.5, "p35_s100": 1.0, "p35_s200": 2.0}
TURNOVER_MODEL = "pipeline/models/turnover_hgb.json"
CODE_H, CODE_A, CODE_D, CODE_X = turnover.HOME, turnover.AWAY, truth.pm.D, truth.pm.X
FEATURES = len(team_features.FEATURE_NAMES)


def turnover_quietly(function, *arguments, **keywords):
    """(result, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments, **keywords)
    return result, buffer.getvalue()


def hazard_match(match_id, seed):
    """A DflMatch whose truth follows the ball's nearest team (home when its nearest slot is the closer one, read every
    60 frames, so that changes are at least 60 frames apart), with a dead
    ball over 195..209 of each 400 frames (a change of the nearest team inside one is no direct change) and alive
    elsewhere. The truth is a function of the features, so a model has something to split on, and both classes exist."""
    base = random_match(match_id, seed=seed, rows=TURNOVER_ROWS, gap=25)
    served = turnover.served_inputs(base, "clean", 0, None, None)
    X = team_features.features(served)[1]
    h, a = X[:, team_features.COLUMN["h_ball_d1"]], X[:, team_features.COLUMN["a_ball_d1"]]
    with np.errstate(invalid="ignore"):
        home = np.where(np.isnan(a), True, np.where(np.isnan(h), False, h < a))
    frames = np.arange(base.n_frames)
    home = home[frames // 60 * 60]                      # held for 60 frames: a change at most every 2.4 s
    inside = np.asarray(base.frame_number) >= 0
    possession = np.where(inside, np.where(home, 0, 1), -1).astype(np.int8)
    status = np.where(inside, np.where((frames % 400 >= 195) & (frames % 400 < 210), 0, 1), -1).astype(np.int8)
    possession.setflags(write=False)
    status.setflags(write=False)
    return dataclasses.replace(base, ball_status=status, ball_possession=possession)


def hazard_matches():
    return {match_id: hazard_match(match_id, seed) for seed, match_id in enumerate(TURNOVER_IDS)}


@functools.lru_cache(maxsize=1)
def hazard_artifact():
    """(matches, artifact): three hazard matches and the artifact trained on them (profile clean, one seed)."""
    matches = hazard_matches()
    return matches, turnover.train("clean", 1, 5, matches=matches)


def plain_labels(codes, graded, window):
    """{frame: label} by plain loops, from the definition: 1 where some c in the same block, c > the block's start, has
    codes[c - 1] and codes[c] both H or A and different, and |t - c| <= window."""
    labels = {}
    for _, start, stop in graded:
        changes = [c for c in range(start + 1, stop)
                   if codes[c - 1] in (CODE_H, CODE_A) and codes[c] in (CODE_H, CODE_A) and codes[c] != codes[c - 1]]
        for t in range(start, stop):
            labels[t] = int(any(abs(t - c) <= window for c in changes))
    return labels


def test_p35_model_label_marks_direct_changes():
    codes = np.full(300, CODE_X, dtype=np.int8)
    graded = ((1, 10, 110), (2, 150, 250))
    codes[9], codes[10:40], codes[40:80] = CODE_A, CODE_H, CODE_A       # frame 9 is outside the block: no change at 10
    codes[80:90], codes[90:110] = CODE_D, CODE_H                          # A, D..., H: a change through D is no change
    codes[149], codes[150:245], codes[245:250] = CODE_A, CODE_H, CODE_A   # again outside at 149; a change at 245, by the edge
    assert turnover.direct_changes(codes, graded) == [(40, 10, 110), (245, 150, 250)]
    labels = turnover.change_labels(codes, graded)
    assert labels.dtype == bool and labels.shape == (300,)
    assert set(np.flatnonzero(labels)) == set(range(28, 53)) | set(range(233, 250)), "+-12 around 40, and 233 to 249"
    assert not labels[250:].any() and not labels[:10].any() and not labels[110:150].any(), "nothing outside a block"
    assert not labels[80:90].any() and not labels[90:100].any(), "no label for the change through D, nor at its end"
    # a window of 0 marks the change frame alone; the definition by plain loops agrees on a random truth
    assert set(np.flatnonzero(turnover.change_labels(codes, graded, window=0))) == {40, 245}
    rng = np.random.default_rng(3)
    random_codes = rng.choice(np.array([CODE_H, CODE_A, CODE_D], dtype=np.int8), size=400, p=[0.4, 0.4, 0.2])
    random_codes[:30] = CODE_X
    blocks = ((1, 30, 200), (2, 230, 400))
    for window in (0, 1, 5, 12):
        wanted = plain_labels(random_codes, blocks, window)
        got = turnover.change_labels(random_codes, blocks, window)
        assert all(int(got[t]) == label for t, label in wanted.items()) and not got[:30].any() and not got[200:230].any()
    assert turnover.WINDOW_FRAMES == turnover_hazard.WINDOW_FRAMES == 12
    for bad in (-1, 1.5, True, "12"):
        expect_error(ValueError, lambda: turnover.change_labels(codes, graded, window=bad), "train_turnover", "window")


def test_p35_model_rows_are_p32s_rows_with_the_label():
    match = hazard_match("RND001", 0)
    inputs = turnover.served_inputs(match, "clean", 0, None, None)
    codes, _ = truth.truth_codes(match)
    X, y = turnover.rows(inputs, codes, 5)
    visible = kinematics.ball_track(inputs).visible
    on_grid = np.concatenate([np.arange(start, stop, 5) for _, start, stop in match.graded])
    kept = [t for t in on_grid if codes[t] in (CODE_H, CODE_A) and visible[t]]
    hidden = [t for t in on_grid if codes[t] in (CODE_H, CODE_A) and not visible[t]]
    dead = [t for t in on_grid if codes[t] == CODE_D]
    assert kept and hidden and dead, "the match needs rows, alive frames with the ball hidden and dead frames"
    assert X.shape == (len(kept), FEATURES) and y.shape == (len(kept),) and y.dtype == np.int64
    assert np.array_equal(X, team_features.features(inputs)[1][kept], equal_nan=True)
    labels = plain_labels(codes, match.graded, 12)
    assert np.array_equal(y, np.array([labels[t] for t in kept])) and set(np.unique(y)) == {0, 1}
    # the same rows as P32's trainer gives (its label is the team: 1 where the truth is H)
    X32, y32 = train_team.rows(inputs, codes, 5)
    assert np.array_equal(X, X32, equal_nan=True) and np.array_equal(y32, np.array([codes[t] == CODE_H for t in kept]))
    expect_error(ValueError, lambda: turnover.rows(inputs, codes, 0), "train_turnover", "stride")
    expect_error(ValueError, lambda: turnover.rows(inputs, codes, True), "stride")
    expect_error(ValueError, lambda: turnover.rows(inputs, codes[:-1], 5), "codes")


def test_p35_model_mirror_keeps_the_label():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, FEATURES)) * 4.0
    X[rng.random(X.shape) < 0.2] = np.nan
    y = (np.arange(300) % 10 == 0).astype(np.int64)                        # 10% positives: the classes stay unbalanced
    Xa, ya = turnover.augment(X, y)
    assert Xa.shape == (600, FEATURES) and ya.dtype == np.int64
    assert np.array_equal(Xa[:300], X, equal_nan=True) and np.array_equal(ya[:300], y)
    assert np.array_equal(ya[300:], y), "a change is a change whichever team is which: the mirror keeps the label"
    assert float(ya.mean()) == 0.1, "the rows' own balance, not 0.5"
    for i in range(300):
        assert np.array_equal(Xa[300 + i], team_features.swap(X[i]), equal_nan=True), i
    expect_error(ValueError, lambda: turnover.augment(X[:, :56], y), "train_turnover", "72")
    expect_error(ValueError, lambda: turnover.augment(X, y[:-1]), "train_turnover")
    expect_error(ValueError, lambda: turnover.augment(X, y + 1), "train_turnover", "0 or 1")


def test_p35_model_fit_is_p05s_settings_and_stores_the_base_rate():
    assert turnover.HGB_SETTINGS is train_dead_ball.HGB_SETTINGS, "P05's settings, imported and not retyped"
    rng = np.random.default_rng(7)
    X = rng.normal(size=(1500, FEATURES)) * 3.0
    X[rng.random(X.shape) < 0.15] = np.nan
    y = ((np.nan_to_num(X[:, 0]) + rng.normal(size=1500)) > 1.2).astype(np.int64)
    model = turnover.fit(X, y)
    estimator = turnover.hgb_estimator(*turnover.augment(X, y))
    assert estimator.n_iter_ == turnover.HGB_SETTINGS["max_iter"] and estimator.monotonic_cst is None
    assert model == {**turnover.export_hgb(estimator), "base_rate": float(y.mean())}
    assert 0.0 < model["base_rate"] < 0.5 and "prior_home" not in model
    expect_error(ValueError, lambda: turnover.fit(X, np.zeros(1500, dtype=np.int64)), "train_turnover", "one class")
    expect_error(ValueError, lambda: turnover.fit(X, np.ones(1500, dtype=np.int64)), "one class")
    expect_error(ValueError, lambda: turnover.fit(X[:0], y[:0]), "one class")


def test_p35_model_trainer_never_passes_truth():
    matches = synthetic_matches()
    seen, real = [], team_features.features

    def spy(inputs):
        seen.append([leaks(inputs, other) for other in matches.values()])
        return real(inputs)
    team_features.features = spy
    real_fit = turnover.fit
    turnover.fit = lambda X, y: {"rows": int(len(y))}
    try:
        with tempfile.TemporaryDirectory() as root:
            turnover.train("generic", 1, 9, matches=matches, data_dir=root)
    finally:
        turnover.fit = real_fit
        team_features.features = real
    assert seen and all(found == [[] for _ in matches] for found in seen), seen
    # generic reads no Calibration, and one that is given is refused
    expect_error(ValueError, lambda: turnover.train("generic", 1, 5, matches=synthetic_matches(),
                                                    calibration=object()), "generic", "calibration")


def turnover_verify(path, fresh):
    return turnover_quietly(turnover.verify, path, fresh)[0]


def test_p35_model_artifact_layout_folds_and_round_trip():
    matches, artifact = hazard_artifact()
    assert sorted(artifact) == sorted(["format", "kind", "target", "window_frames", "feature_version", "features",
                                       "profile", "seeds", "stride", "provenance", "folds", "full"])
    assert (artifact["format"], artifact["kind"], artifact["target"], artifact["window_frames"]) == (
        trees.FORMAT, "hgb", "team_change_pm12", 12)
    assert artifact["feature_version"] == 2 and artifact["features"] == list(team_features.FEATURE_NAMES)
    assert sorted(artifact["folds"]) == list(TURNOVER_IDS)
    data = turnover.gather("clean", 1, 5, matches=matches).data
    for held_out in TURNOVER_IDS:
        others = [name for name in TURNOVER_IDS if name != held_out]
        X = np.concatenate([data[name][0] for name in others])
        y = np.concatenate([data[name][1] for name in others])
        fold = artifact["folds"][held_out]
        assert sorted(fold) == ["base_rate", "baseline", "trees"] and fold["base_rate"] == float(y.mean()), held_out
        assert fold == turnover.fit(X, y), "the fold of {} is the fit on the other two matches".format(held_out)
    everything_y = np.concatenate([data[name][1] for name in TURNOVER_IDS])
    assert artifact["full"]["base_rate"] == float(everything_y.mean())
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "models", "turnover.json")
        turnover.write(path, artifact)
        with open(path, "rb") as handle:
            raw = handle.read()
        text = raw.decode("utf-8")
        assert text == turnover.dumps(artifact) and text.count("\n") == 1 and b"\r" not in raw
        assert strict_json(text) == artifact, "floats must read back exactly"
        assert turnover_hazard.load(path) == artifact, "the source reads the file the trainer writes"
        assert turnover_verify(path, artifact) == 0
        changed = copy.deepcopy(artifact)
        changed["folds"][TURNOVER_IDS[0]]["base_rate"] += 0.01
        assert turnover_verify(path, changed) == 1
    again = turnover.train("clean", 1, 5, matches=matches)
    assert turnover.dumps(again) == turnover.dumps(artifact), "retraining wrote different bytes"
    assert turnover.compare(artifact, again) == ([], [])
    assert artifact["provenance"]["source_sha256"] == turnover.source_hashes()
    assert "train_turnover.py" in turnover.source_hashes() and "turnover_hazard.py" in turnover.source_hashes()


@functools.lru_cache(maxsize=1)
def learned_artifact():
    """(matches, artifact, fitted): the hazard matches' artifact with each fold replaced by a model of real splits,
    fitted on a label that is a function of a feature (the hazard matches' own labels are not learnable, so their
    trees stay a single leaf); fitted is {match: (X, y, estimator)} over the frames with the ball seen."""
    matches, artifact = hazard_artifact()
    column = team_features.COLUMN["ball_line_m"]
    learned, fitted = copy.deepcopy(artifact), {}
    for held_out in TURNOVER_IDS:
        inputs = turnover.served_inputs(matches[held_out], "clean", 0, None, None)
        X = team_features.features(inputs)[1][kinematics.ball_track(inputs).visible]
        y = (np.nan_to_num(X[:, column], nan=-1.0) > np.nanmedian(X[:, column])).astype(np.int64)
        fitted[held_out] = (X, y, turnover.hgb_estimator(*turnover.augment(X, y)))
        learned["folds"][held_out] = {**turnover.export_hgb(fitted[held_out][2]), "base_rate": float(y.mean())}
        assert any(len(tree["feature"]) > 1 for tree in learned["folds"][held_out]["trees"]), "no split grown"
    return matches, learned, fitted


def test_p35_model_source_runs_a_fitted_model_as_scikit_learn_does():
    """The source on a model with real splits is scikit-learn's own symmetrised score, p / (p + base_rate), on the seen
    frames."""
    matches, learned, fitted = learned_artifact()
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "turnover.json")
        turnover.write(path, learned)
        for held_out, (X, y, estimator) in fitted.items():
            inputs = turnover.served_inputs(matches[held_out], "clean", 0, None, None)
            home = turnover_hazard.compute(RunContext(inputs=inputs, seed=0), model=path).home
            speaks = kinematics.ball_track(inputs).visible
            logit = (estimator.decision_function(X) + estimator.decision_function(team_features.swap(X))) / 2.0
            p = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.001, 0.999)
            base = learned["folds"][held_out]["base_rate"]
            assert np.isnan(home[~speaks]).all() and speaks.sum() > 300 and np.ptp(p) > 0.1
            worst = float(np.abs(home[speaks] - p / (p + base)).max())
            print("      {}: the source against scikit-learn's own fold: max |difference| {:.3e}".format(held_out, worst))
            assert worst <= 1e-12, worst


# -- the dry run ------------------------------------------------------------------------------------------------------

def turnover_varying(real):
    """team_features.features with the columns that a small synthetic match may leave constant made to vary."""
    column = team_features.COLUMN

    def features(inputs):
        names, X = real(inputs)
        X = np.array(X)
        for step, name in enumerate(("h_in_episode", "a_in_episode", "h_since_episode_s", "a_since_episode_s",
                                     "h_until_episode_s", "a_until_episode_s")):
            X[:, column[name]] = (np.arange(len(X)) // (step + 1)) % 3
        return names, X
    return features


def test_p35_model_dry_run_windows_are_the_declared_ones():
    assert turnover.ROW_WINDOW == (8_290, 12_434) == train_team.ROW_WINDOW, "P32's rows, amendment 4"
    assert turnover.ROW_WINDOW_RUN == ("generic", 1, 5, ["J03WMX"]) and turnover.DEFAULT_STRIDE == 5
    assert turnover.POSITIVE_WINDOW == (0.06, 0.14)
    assert turnover.TRAINING_ROWS == train_team.TRAINING_ROWS and turnover.PEAK_LIMIT_GB == 8.0
    assert turnover.COPIES == train_team.COPIES
    assert (turnover.AUC_MIN, turnover.LOSS_RATIO_MAX) == (0.80, 0.90)
    with tempfile.TemporaryDirectory() as root:
        match = random_match("J03WMX", rows=(700, 600))
        declared = turnover.dry_run("generic", 1, ["J03WMX"], 5, matches={"J03WMX": match}, data_dir=root)
        assert declared["checks"]["rows"]["window"] == [8290, 12434] and declared["checks"]["rows"]["ok"] is False
        assert declared["checks"]["positive"]["window"] == [0.06, 0.14] and declared["ok"] is False
        text = "\n".join(turnover.render_dry_run(declared))
        assert "FAIL, outside the window [8,290, 12,434]" in text and "positive share" in text
        for profile, seeds, ids in (("clean", 1, ["J03WMX"]), ("generic", 2, ["J03WMX"])):
            other = turnover.dry_run(profile, seeds, ids, 5, matches={"J03WMX": match}, data_dir=root)
            assert other["checks"]["rows"] == {"window": None, "ok": None}, (profile, seeds)
            assert other["checks"]["positive"] == {"window": None, "ok": None}, (profile, seeds)


def test_p35_model_dry_run_is_aggregates_and_gates_the_pause():
    """Aggregates only; an all-NaN or constant column, rows or a positive share outside their windows and a projection
    over 8 GB each fail it, against a stubbed peak so that the check does not depend on the process it runs in."""
    matches = {"J03WMX": hazard_match("J03WMX", 0), "J03WN1": hazard_match("J03WN1", 1)}
    column = team_features.COLUMN
    real, real_peak = team_features.features, train_team.peak_memory_bytes
    stub_peak = 500_000_000
    team_features.features = turnover_varying(real)
    train_team.peak_memory_bytes = lambda: stub_peak
    try:
        base = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches)
        json.dumps(base, allow_nan=False)
        assert sorted(base) == sorted(["profile", "seeds", "stride", "per_match", "rows", "positive_share", "nan_share",
                                       "all_nan", "constant", "constant_values", "peak_memory_bytes",
                                       "projected_peak_gb", "projection_basis", "checks", "ok"])
        gathered = turnover.gather("clean", 1, 5, matches=matches).data
        X, y = gathered["J03WMX"]
        cell = base["per_match"]["J03WMX"]
        assert base["rows"] == cell["rows"] == len(y) and cell["x_bytes"] == X.nbytes
        assert base["positive_share"] == float(y.mean()) == cell["positive_share"]
        assert 0.0 < base["positive_share"] < 1.0
        assert list(base["nan_share"]) == list(team_features.FEATURE_NAMES) and base["ok"] is True
        extra = turnover.TRAINING_ROWS * 2 * FEATURES * 8 * turnover.COPIES / 1e9
        assert base["peak_memory_bytes"] == stub_peak
        assert abs(base["projected_peak_gb"] - (stub_peak / 1e9 + extra)) < 1e-9
        rows, share = base["rows"], base["positive_share"]
        inside = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, row_window=(rows, rows),
                                  positive_window=(share, share))
        assert inside["checks"]["rows"]["ok"] is True and inside["checks"]["positive"]["ok"] is True and inside["ok"]
        for window in ((rows + 1, rows + 10), (0, rows - 1)):
            out = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, row_window=window)
            assert out["checks"]["rows"]["ok"] is False and out["ok"] is False, window
        for window in ((share + 0.01, 1.0), (0.0, share - 0.01)):
            out = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, positive_window=window)
            assert out["checks"]["positive"]["ok"] is False and out["ok"] is False, window
        tight = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches, peak_limit_gb=0.0)
        assert tight["checks"]["projection"] == {"limit_gb": 0.0, "ok": False} and tight["ok"] is False
        # a dead column and a constant one are named, and fail the command on their own
        varied = team_features.features

        def edited(inputs):
            names, X_ = varied(inputs)
            X_ = np.array(X_)
            X_[:, column["ball_line_m"]] = np.nan
            X_[:, column["h_ball_adv"]] = 7.0
            return names, X_
        team_features.features = edited
        dead = turnover.dry_run("clean", 1, ["J03WMX"], 5, matches=matches)
        assert dead["all_nan"] == ["ball_line_m"] and dead["constant_values"]["h_ball_adv"] == 7.0 and not dead["ok"]
        lines = turnover.render_dry_run(dead)
        assert any("all-NaN columns: ball_line_m" in line for line in lines)
        assert any("h_ball_adv (always 7)" in line for line in lines)
        team_features.features = varied
        # the command line: exit codes
        with tempfile.TemporaryDirectory() as root, loading(lambda match_id: matches[match_id]):
            flags = ["--dry-run", "--profile", "clean", "--seeds", "1", "--matches", "J03WMX", "--data-dir", root]
            code, text = turnover_quietly(turnover.main, flags)
            assert code == 0 and "OK" in text and "positive share" in text, text
            train_team.peak_memory_bytes = lambda: int((8.0 - extra) * 1e9) + 1_000_000
            code, text = turnover_quietly(turnover.main, flags)
            assert code == 1 and "projected peak FAIL" in text, text
            train_team.peak_memory_bytes = lambda: stub_peak
            code, text = turnover_quietly(turnover.main, ["--dry-run", "--seeds", "1", "--matches", "J03WMX",
                                                          "--data-dir", root])
            assert code == 1 and "FAIL, outside the window [8,290, 12,434]" in text, text
    finally:
        team_features.features = real
        train_team.peak_memory_bytes = real_peak
    expect_error(ValueError, lambda: turnover.dry_run("clean", 0, ["J03WMX"], 5, matches=matches), "seeds")


# -- the report -------------------------------------------------------------------------------------------------------

def test_p35_model_auc_is_the_rank_formula():
    assert turnover.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75, "the textbook case"
    assert turnover.auc([1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 1, 0]) == 5 / 9, "pairs won: 1 + 2 + 2 of 9"
    assert turnover.auc([0.5, 0.5], [0, 1]) == 0.5 and turnover.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert turnover.auc([0.2, 0.9], [0, 1]) == 1.0 and turnover.auc([0.9, 0.2], [0, 1]) == 0.0
    assert turnover.auc([0.1, 0.2], [1, 1]) is None and turnover.auc([0.1, 0.2], [0, 0]) is None
    # ties count half: negatives 0.3, 0.7; positives 0.3, 0.9 -> (0.5 + 1 + 0 + 1) / 4
    assert turnover.auc([0.3, 0.7, 0.3, 0.9], [0, 0, 1, 1]) == 0.625
    rng = np.random.default_rng(11)
    for _ in range(5):
        p = np.round(rng.random(60), 1)                                      # many ties
        y = (rng.random(60) < 0.3).astype(int)
        wins = sum((a > b) + 0.5 * (a == b) for a in p[y == 1] for b in p[y == 0])
        assert abs(turnover.auc(p, y) - wins / ((y == 1).sum() * (y == 0).sum())) < 1e-12
    assert abs(turnover.log_loss_sum([0.5, 0.5], [0, 1]) - 2 * np.log(2)) < 1e-12
    assert abs(turnover.log_loss_sum([0.9, 0.2], [1, 0]) - (-np.log(0.9) - np.log(0.8))) < 1e-12
    assert np.isfinite(turnover.log_loss_sum([0.0, 1.0], [1, 0])), "a probability of 0 or 1 is held off the edge"


def test_p35_model_go_rule_at_its_edges():
    base = 0.30
    edge = 0.90 * base
    cases = ((0.80, edge, True, "both exactly at the gate"), (0.8001, edge - 0.01, True, "above both"),
             (0.7999, edge - 0.01, False, "the AUC just short"), (0.95, edge + 1e-9, False, "the loss just over"),
             (0.95, base, False, "no better than the base rate"), (None, 0.1, False, "no AUC"),
             (0.95, None, False, "no loss"))
    for auc_value, loss, wanted, why in cases:
        verdict = turnover.go(auc_value, loss, base)
        assert verdict["go"] is wanted, why
        assert verdict["auc_ok"] is (auc_value is not None and auc_value >= 0.80 - 1e-12), why
    verdict = turnover.go(0.85, 0.2, 0.4)
    assert verdict["loss_ratio"] == 0.5 and verdict["auc_min"] == 0.8 and verdict["loss_ratio_max"] == 0.9
    assert turnover.go(0.9, 0.2, None)["go"] is False
    place = turnover.localisation([-30, -8, -3, 0, 2, 5, 6, 40])
    assert place == {"changes": 8, "median": 1.0, "q1": -4.25, "q3": 5.25, "within5": 0.5}, place
    assert turnover.localisation([]) == {"changes": 0, "median": None, "q1": None, "q3": None, "within5": None}


def test_p35_model_report_scores_each_fold_on_its_own_match():
    matches, artifact = hazard_artifact()
    report_agrees_with_plain_loops(matches, artifact)
    report_agrees_with_plain_loops(*learned_artifact()[:2])               # and with trees that split


def report_agrees_with_plain_loops(matches, artifact):
    result = turnover.report(artifact, "clean", 1, 5, matches=matches)
    json.dumps(result, allow_nan=False)
    assert sorted(result) == sorted(["profile", "seeds", "stride", "model_sha256", "per_match", "pooled", "go",
                                     "transfer", "localisation", "match_seeds"])
    assert result["match_seeds"] == 3 * 1
    assert sorted(result["per_match"]) == list(TURNOVER_IDS)
    all_p, all_y, loss_sums, base_sums, means, offsets = [], [], 0.0, 0.0, [], []
    for match_id in TURNOVER_IDS:
        match = matches[match_id]
        codes, _ = truth.truth_codes(match)
        inputs = turnover.served_inputs(match, "clean", 0, None, None)
        labels = plain_labels(codes, match.graded, 12)
        visible = kinematics.ball_track(inputs).visible
        model = artifact["folds"][match_id]
        X = team_features.features(inputs)[1]
        logit = np.full(len(X), np.nan)
        logit[visible] = (trees.score(model, X[visible]) + trees.score(model, team_features.swap(X[visible]))) / 2.0
        p_all = 1.0 / (1.0 + np.exp(-logit))
        frames = [t for _, start, stop in match.graded for t in range(start, stop, 5)
                  if codes[t] in (CODE_H, CODE_A) and visible[t]]
        p = np.array([p_all[t] for t in frames])
        y = np.array([labels[t] for t in frames])
        base = model["base_rate"]
        loss = float(-sum(np.log(a) if b else np.log(1 - a) for a, b in zip(p, y)))
        base_loss = float(-sum(np.log(base) if b else np.log(1 - base) for b in y))
        wins = sum((a > b) + 0.5 * (a == b) for a in p[y == 1] for b in p[y == 0])
        cell = result["per_match"][match_id]
        assert cell["rows"] == len(y) and cell["positives"] == int(y.sum()) and cell["base_rate"] == base
        assert abs(cell["auc"] - wins / ((y == 1).sum() * (y == 0).sum())) < 1e-12
        assert abs(cell["log_loss"] - loss / len(y)) < 1e-9 and abs(cell["base_log_loss"] - base_loss / len(y)) < 1e-9
        graded_seen = [t for _, start, stop in match.graded for t in range(start, stop) if visible[t]]
        assert cell["seen_frames"] == len(graded_seen) and len(graded_seen) > len(frames), "alive and dead, every frame"
        assert abs(cell["mean_p_seen"] - np.mean(np.clip([p_all[t] for t in graded_seen], 0.001, 0.999))) < 1e-12
        for c, start, stop in turnover.direct_changes(codes, match.graded):
            if visible[c]:
                window = list(range(max(start, c - 25), min(stop, c + 26)))
                values = [logit[t] if visible[t] else -np.inf for t in window]
                offsets.append(window[int(np.argmax(values))] - c)
        all_p.append(p)
        all_y.append(y)
        loss_sums, base_sums = loss_sums + loss, base_sums + base_loss
        means.append(cell["mean_p_seen"])
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    pooled = result["pooled"]
    wins = sum((a > b) + 0.5 * (a == b) for a in p[y == 1] for b in p[y == 0])
    assert pooled["rows"] == len(y) and pooled["positives"] == int(y.sum())
    assert abs(pooled["auc"] - wins / ((y == 1).sum() * (y == 0).sum())) < 1e-12
    assert abs(pooled["log_loss"] - loss_sums / len(y)) < 1e-9
    assert abs(pooled["base_log_loss"] - base_sums / len(y)) < 1e-9
    assert result["go"] == turnover.go(pooled["auc"], pooled["log_loss"], pooled["base_log_loss"])
    assert result["transfer"] == {"low": 0.5 * min(means), "high": 2.0 * max(means)}
    assert offsets and result["localisation"] == turnover.localisation(offsets)
    assert result["model_sha256"] == hashlib.sha256(turnover.dumps(artifact).encode()).hexdigest()
    lines = turnover.render_report(result)
    assert any(line.startswith("report    GO") or line.startswith("report    NO GO") for line in lines)
    assert any("transfer" in line for line in lines)
    place = [line for line in lines if "localisation" in line]
    assert len(place) == 1 and "summed over seeds" in place[0] and "over {} match-seeds".format(
        result["match_seeds"]) in place[0] and "{:,} true direct changes".format(
        result["localisation"]["changes"]) in place[0], place
    assert sum(1 for line in lines if line.startswith("report    RND00")) == 3
    # a model that never saw the match is the only one that scores it: a missing fold is refused
    short = copy.deepcopy(artifact)
    del short["folds"]["RND002"]
    expect_error(ValueError, lambda: turnover.report(short, "clean", 1, 5, matches=matches), "RND002", "full")


def test_p35_model_report_refuses_a_file_that_is_not_its_own():
    """The report reads the file through the source's loader and holds it to what it is run with: a target, a window,
    a feature version, a profile or a stride that differs is one readable line (exit 2 on the command line)."""
    matches, artifact = hazard_artifact()
    with tempfile.TemporaryDirectory() as root:
        def written(**changes):
            changed = dict(copy.deepcopy(artifact), **changes)
            path = os.path.join(root, "model.json")
            turnover.write(path, changed)
            return path

        assert turnover.load_for_report(written(), "clean", 5) == artifact
        for name, value in (("target", "team_home"), ("window_frames", 13), ("feature_version", 1)):
            expect_error(ValueError, lambda: turnover.load_for_report(written(**{name: value}), "clean", 5),
                         "turnover_hazard", name)                               # the source's own loader refuses these
        for name, value in (("profile", "generic"), ("stride", 7)):
            message = expect_error(ValueError, lambda: turnover.load_for_report(written(**{name: value}), "clean", 5),
                                   "train_turnover", name, repr(value), "retrain")
            assert "\n" not in message, "one line"
        expect_error(ValueError, lambda: turnover.load_for_report(written(), "generic", 5), "profile", "'clean'")
        expect_error(ValueError, lambda: turnover.load_for_report(written(), "clean", 3), "stride")
        expect_error(ValueError, lambda: turnover.load_for_report(os.path.join(root, "absent.json"), "clean", 5),
                     "absent.json")
        # the command line: exit 2 and the line, before any match is loaded
        flags = ["--report", "--profile", "clean", "--out", written(stride=7), "--data-dir", root]
        code, text = turnover_quietly(turnover.main, flags)
        assert code == 2 and text.startswith("REFUSED") and text.count("\n") == 1 and "stride" in text, text


# -- the configs ------------------------------------------------------------------------------------------------------

def test_p35_model_configs_are_p32_w030_plus_the_hazard():
    """Each is p32_w030.toml plus one [[evidence]] table at the end (kind turnover_hazard, name hazard, the model file)
    and the decoder's hazard_source and hazard_scale; nothing else differs. TOML contents only: the decoder's
    parameters arrive with A1 and the model file with the controller's training."""
    configs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with open(os.path.join(configs, "p32", "p32_w030.toml"), "rb") as handle:
        reference = tomllib.load(handle)
    paths = sorted(glob.glob(os.path.join(configs, "p35", "*.toml")))
    assert [os.path.splitext(os.path.basename(path))[0] for path in paths] == sorted(TURNOVER_CONFIGS), paths
    descriptions = set()
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as handle:
            config = tomllib.load(handle)
        assert config["evidence"][-1] == {"kind": "turnover_hazard", "name": "hazard", "model": TURNOVER_MODEL}, name
        assert [table["kind"] for table in config["evidence"]].count("turnover_hazard") == 1, name
        decoder = config["decoder"]
        assert decoder["hazard_source"] == "hazard" and decoder["hazard_scale"] == TURNOVER_CONFIGS[name], name
        assert isinstance(decoder["hazard_scale"], float), "{}: a float, as the others are".format(name)
        assert isinstance(config["description"], str) and config["description"] != reference["description"], name
        descriptions.add(config["description"])
        left, right = copy.deepcopy(config), copy.deepcopy(reference)
        left.pop("description")
        right.pop("description")
        left["evidence"] = left["evidence"][:-1]
        del left["decoder"]["hazard_source"], left["decoder"]["hazard_scale"]
        assert left == right, "{} differs from p32_w030 in more than the hazard".format(name)
    assert len(descriptions) == len(TURNOVER_CONFIGS), "each config has its own description"
