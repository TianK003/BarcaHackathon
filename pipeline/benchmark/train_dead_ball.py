"""P05's trainer: fit the dead-ball model on the benchmark's matches and write it as JSON.

    python pipeline/benchmark/train_dead_ball.py --model hgb|logistic [--profile tracking] [--seeds 3] [--stride 5]
                                                 [--out PATH] [--verify] [--data-dir D]
    python pipeline/benchmark/train_dead_ball.py --rules [--profile tracking] [--seeds 3] [--data-dir D]

    artifact = train("hgb", profile="tracking", seeds=3, stride=5)        # the dict that is written, JSON-ready

What it learns, and from what
-----------------------------
The classifier answers one question, is the ball in play, from the thirty `play_features` of a frame. The seven open
DFL/IDSSE matches are the only ones with a flag for it (`BallStatus`), so those are the training data, degraded to look
like our tracking (profile `tracking`, version 3, seeds 0 to 2: P01), and served as `MatchInputs`, so that the features
are computed by the very code and from the very kind of inputs that the submission pipeline will give them. No truth
reaches the features: the label is read from `truth.truth_codes(match)` beside them, never through the inputs
(test_trainer_never_passes_truth). A row is every `stride`-th graded frame of a block from its start, 1 where the truth
is not D: stride 5 (0.2 s) keeps the rows deterministic and cuts the autocorrelation, since neighbouring frames carry
nearly the same features and the same label.

The models
----------
`hgb` is scikit-learn's `HistGradientBoostingClassifier` with settings that are fixed and never tuned (T26: a score
to be tuned towards would have to come from the public window or from the benchmark that also scores the submission):

    loss log_loss, learning_rate 0.05, max_iter 300, max_leaf_nodes 15, min_samples_leaf 1000,
    l2_regularization 1.0, max_bins 255, random_state 0, no class weights.

They are the literature survey's (docs/proposals.md P05). Boosting is the family that Lang et al. 2022, the one direct
study of in-play status from tracking (102 Bundesliga matches, players only), found best, ahead of a random forest,
logistic regression and a decision tree. The rest is the conservative end of the usual defaults: a small learning rate
with enough rounds, leaves of at least 1,000 rows (a stoppage is hundreds of neighbouring, nearly identical frames, so
a leaf must hold several of them), a little L2, and the 255 bins that are the histogram's most. There is no early
stopping: the default 'auto' holds out a random 10% of the frames, and a random split of autocorrelated
frames puts a frame's neighbours on both sides of it, which leaks; the rounds are fixed instead. Two monotone
constraints encode what the laws of the game say, that a ball further beyond a line (`ball_outside_m`) and slower
players (`slow_share_c4`) never make play likelier, and cost nothing where the data agree.

`logistic` is the reference a boosted model has to beat: each feature with a NaN in the training rows gets a missing
indicator and its NaN is imputed with the training mean, a column whose standard deviation rounds to 0 in a fold is
dropped (the artifact records which), and `hazard.fit_logistic(X, y, l2=1.0, iterations=100)` fits it.

Folds, the artifact, `--verify`
-------------------------------
For each match m there is a model fitted on the other six, and `full`, fitted on all seven: a benchmark match is scored
by the model that never saw it, and BAR-ATH by `full` (pipeline/evidence/dead_ball_model.py). The file is
pipeline/models/dead_ball_<kind>.json, `{format, kind, feature_version, features, profile, seeds, stride, provenance,
folds: {match_id: model}, full: model}`, written with sorted keys and Python's own float text, which reads back
exactly, LF, no NaN, under 10 MB, and nothing frame by frame. Each model has `prior_alive` (its training rows' alive
share, for the source's prior correction). An `hgb` model holds `baseline` and, per tree, the node arrays `feature`,
`threshold`, `missing_left`, `left`, `right`, `leaf` and `value`, read out of the fitted predictors (a private
structure of scikit-learn, which is why the version is pinned in requirements-benchmark.txt and
test_exported_trees_match_sklearn holds the export to 1e-12 of `predict_proba`). A `logistic` one holds its columns,
means, scales, coefficients, intercept and imputation values.

`provenance` is the scikit-learn version, the Calibration's digest, the DFL files' sha256 and the sha256 of the three
source files the model depends on. It has no git commit: that changes with every later commit and would make the file
differ from its own retraining for no reason. `--verify` retrains and compares everything but those three source
hashes and exits 1 if anything differs; a differing source hash is a note and not a failure, because a comment edited
after training is not a different model, and the features it was trained on are still pinned by `feature_version`.

It prints, per held-out match, on its out-of-fold predictions: log-loss, Brier score and AUC, and a pooled 10-bin
reliability table (counts and means only).

--rules
-------
`dead_ball_rules` takes five probabilities that are measurements. `--rules` makes them: for each of the four triggers
and for "no trigger", on all graded frames of the benchmark, P(trigger | alive) and P(trigger | dead), and the value
P(t | alive) / (P(t | alive) + P(t | dead)) rounded towards 0.5 to two decimals, the parameter. Rounding towards 0.5 is
the conservative direction: a rule is never trusted more than it was measured. They are measured on all seven matches,
which is a small optimism (a rule's trigger thresholds were fixed beforehand, but the seven matches are the ones the
rules are then scored on; R009 states it). The triggers are read off `dead_ball_rules` itself by running it with one
probability at 0 and the others at 1, never written a second time here.

Determinism: nothing random but HGB's random_state, no set iterated, folds in match-id order; the same inputs and
versions give the same bytes on any number of threads (test_training_is_deterministic_across_threads).
"""

import argparse
import hashlib
import json
import math
import numbers
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, hazard, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError, make_atd  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import dead_ball_model, dead_ball_rules, play_features  # noqa: E402

import possession_metrics as pm  # noqa: E402  (importing pipeline above put evaluation/ on sys.path)

# Params
FORMAT = 1
MODELS = ("hgb", "logistic")
DEAD = pm.D
DEFAULT_STRIDE = 5
MAX_BYTES = 10 * 1000 * 1000                  # one artifact stays under 10 MB
HGB_SETTINGS = {"loss": "log_loss", "learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
                "min_samples_leaf": 1000, "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False,
                "random_state": 0, "class_weight": None}
MONOTONIC = {"ball_outside_m": -1, "slow_share_c4": -1}      # the rest 0
LOGISTIC_L2, LOGISTIC_ITERATIONS = 1.0, 100
BINS = 10                                     # the reliability table's
SOURCE_FILES = (("train_dead_ball.py", os.path.join(HERE, "train_dead_ball.py")),
                ("play_features.py", os.path.join(ROOT, "pipeline", "evidence", "play_features.py")),
                ("kinematics.py", os.path.join(ROOT, "pipeline", "evidence", "kinematics.py")))
TRIGGERS = ("out", "still", "slow", "kickoff", "none")       # dead_ball_rules' parameters are alive_<trigger>
LOG_FLOOR = 1e-15                             # a probability is held this far from 0 and 1 for the log-loss
MAX_PROBLEMS = 20                             # --verify lists at most this many differences


# -- the rows ---------------------------------------------------------------------------------------------------------

def served_inputs(match, profile, seed, calibration, data_dir):
    """The MatchInputs that the pipeline's components are given for `match`: degraded tracking and no truth."""
    return match_inputs.MatchInputs(match, make_atd(match, profile, seed, calibration), profile=profile, seed=seed,
                                    data_dir=data_dir)


def rows(inputs, codes, stride):
    """(X, y): the features of every `stride`-th graded frame from each block's start, and 1 where `codes` is not D.

    `codes` is the truth's int8 pm codes per frame of the match, kept apart from the inputs, which have none.
    """
    stride = _whole("stride", stride, 1)
    codes = np.asarray(codes)
    if codes.shape != (inputs.n_frames,):
        raise ValueError("train_dead_ball: codes must hold one code per frame of the match ({}), got shape {}"
                         .format(inputs.n_frames, codes.shape))
    frames = np.concatenate([np.arange(start, stop, stride) for _, start, stop in inputs.graded])
    _, X = play_features.features(inputs)
    return X[frames], (codes[frames] != DEAD).astype(np.int64)


class Gathered:
    """What the matches gave: `data` {match_id: (X, y)} over all seeds, and the provenance parts they came with."""

    def __init__(self, data, calibration_sha256, dfl_sources, seeds):
        self.data, self.calibration_sha256, self.dfl_sources, self.seeds = data, calibration_sha256, dfl_sources, seeds


def gather(profile, seeds, stride, matches=None, calibration=None, data_dir=None, match_ids=None, log=None):
    """Gathered: the rows of every match under every seed of `profile`, served as the pipeline serves them.

    `matches` is {id: DflMatch} to use instead of loading the seven from `data_dir` (the checks pass synthetic ones);
    `calibration` is the Calibration that a degraded profile takes its masks from, measured or read from its cache when
    none is given. BenchmarkError for an unknown profile, fewer than two matches (a fold needs another match to be
    fitted on), a calibration given to the clean profile, which degrades nothing and so must not suggest it was used.
    """
    seeds = _whole("seeds", seeds, 1)
    _whole("stride", stride, 1)
    if profile not in run.PROFILES:
        raise BenchmarkError("train_dead_ball: profile {!r} is not one of: {}".format(profile,
                                                                                      ", ".join(sorted(run.PROFILES))))
    ids = run.resolve_match_ids(match_ids, matches)
    if len(ids) < 2:
        raise BenchmarkError("train_dead_ball: a fold is fitted on the other matches, so it needs at least two "
                             "matches, got {}".format(len(ids)))
    if profile == run.CLEAN and calibration is not None:
        raise BenchmarkError("train_dead_ball: profile {!r} takes no calibration, and one was given".format(profile))
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    log = log or (lambda text: None)

    data, sources = {}, {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise BenchmarkError("train_dead_ball: the match given as {} says it is {}; its fold would be filed under "
                                 "one id and looked up under the other, and scored by the model that saw it"
                                 .format(match_id, match.match_id))
        codes, _ = truths.truth_codes(match)
        parts = []
        for seed in seed_list:
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            parts.append(rows(inputs, codes, stride))
            if seed == seed_list[0]:
                recorded = inputs.atd_manifest["sources"]
                sources[match_id] = None if recorded is None else {name: recorded[name]["sha256"]
                                                                   for name in sorted(recorded)}
            del inputs
        data[match_id] = (np.concatenate([X for X, _ in parts]), np.concatenate([y for _, y in parts]))
        log("rows      {}   {:,} rows, {:.1%} alive".format(match_id, len(data[match_id][1]),
                                                           float(data[match_id][1].mean())))
        del match, codes
    digest = None if profile == run.CLEAN else calibration_module.digest(calibration)
    return Gathered(data, digest, sources, len(seed_list))


# -- the models -------------------------------------------------------------------------------------------------------

def fit(kind, X, y, names=play_features.FEATURE_NAMES):
    """One model of `kind`, as a JSON-ready dict with its `prior_alive`: fitted on the (n, 30) X and the 0/1 y."""
    if kind not in MODELS:
        raise ValueError("train_dead_ball: unknown model {!r}; use one of: {}".format(kind, ", ".join(MODELS)))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != len(names) or y.shape != (len(X),) or not np.isin(y, (0, 1)).all():
        raise ValueError("train_dead_ball: X must be (n, {}) and y one 0 or 1 per row, got X {} and y {}"
                         .format(len(names), X.shape, y.shape))
    if y.min() == y.max():
        raise ValueError("train_dead_ball: the rows hold one class only ({} of {}); a model needs both alive and dead "
                         "frames".format(int(y[0]), len(y)))
    model = export_hgb(hgb_estimator(X, y, names)) if kind == "hgb" else fit_logistic(X, y)
    model["prior_alive"] = float(y.mean())
    return model


def hgb_estimator(X, y, names=play_features.FEATURE_NAMES):
    """The fitted `HistGradientBoostingClassifier` with the fixed settings and the monotone constraints."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    constraints = [MONOTONIC.get(name, 0) for name in names]
    return HistGradientBoostingClassifier(monotonic_cst=constraints, **HGB_SETTINGS).fit(
        np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64))


def export_hgb(estimator):
    """The `baseline` and `trees` of a fitted HistGradientBoostingClassifier, as the artifact holds them.

    Reads scikit-learn's private `_baseline_prediction` and `_predictors[i][0].nodes`, which is what the pinned version
    stores, and refuses anything it does not recognise (a categorical split, several trees per round, a field that is
    gone): the export is then wrong, not merely different.
    """
    try:
        if estimator._preprocessor is not None:
            raise AttributeError("a preprocessor reorders the columns")
        baseline = float(np.asarray(estimator._baseline_prediction).ravel()[0])
        predictors = estimator._predictors
        single = estimator.n_trees_per_iteration_ == 1 and all(len(round_) == 1 for round_ in predictors)
        fields = {"value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf",
                  "is_categorical"}
        known = single and all(fields <= set(round_[0].nodes.dtype.names) for round_ in predictors)
    except (AttributeError, IndexError, TypeError):
        known = False
    if not known:
        raise ValueError("train_dead_ball: this scikit-learn does not store its trees as the exporter expects "
                         "(binary, one tree a round, nodes with {}); install the version pinned in "
                         "requirements-benchmark.txt".format(", ".join(sorted(fields))))
    trees = []
    for round_ in predictors:
        nodes = round_[0].nodes
        leaf = nodes["is_leaf"].astype(bool)
        if nodes["is_categorical"][~leaf].any():
            raise ValueError("train_dead_ball: a tree splits on a categorical feature; the features are all numeric")
        threshold = np.where(leaf, 0.0, nodes["num_threshold"])
        # A split on missingness (every seen value left, NaN right) has threshold +inf. The largest float stands in for
        # it: x <= max is x <= inf for every finite x, and a NaN still follows missing_left. dead_ball_model refuses
        # an infinite feature value, the one input for which the stand-in would differ.
        threshold = np.where(threshold == np.inf, np.finfo(np.float64).max, threshold)
        if not np.isfinite(threshold).all() or not np.isfinite(nodes["value"]).all():
            raise ValueError("train_dead_ball: a tree holds a NaN or -inf threshold or a non-finite leaf value")
        trees.append({"feature": np.where(leaf, -1, nodes["feature_idx"]).astype(np.int64).tolist(),
                      "threshold": threshold.tolist(),
                      "missing_left": nodes["missing_go_to_left"].astype(np.int64).tolist(),
                      "left": nodes["left"].astype(np.int64).tolist(),
                      "right": nodes["right"].astype(np.int64).tolist(),
                      "leaf": leaf.astype(np.int64).tolist(), "value": nodes["value"].astype(np.float64).tolist()})
    return {"baseline": baseline, "trees": trees}


def fit_logistic(X, y):
    """The logistic reference: missing indicators and mean imputation, constant columns dropped, then hazard's fit.

    A design column is a feature (id j) or the missing indicator of a feature that has a NaN in these rows (id
    F + j). `columns` are the ids kept, `dropped` those whose standard deviation rounds to 0 at the fit's own precision
    (a feature that is constant here, an indicator that is all ones for a feature that is never seen), in order.
    """
    features = X.shape[1]
    missing = np.isnan(X)
    seen = ~missing
    counts = seen.sum(axis=0)
    impute = np.where(counts > 0, np.where(seen, X, 0.0).sum(axis=0) / np.maximum(counts, 1), 0.0)
    indicators = np.flatnonzero(missing.any(axis=0))
    design = np.column_stack([np.where(missing, impute, X)] + [missing[:, j].astype(np.float64) for j in indicators])
    ids = list(range(features)) + [features + int(j) for j in indicators]
    flat = (design.max(axis=0) == design.min(axis=0)) | (np.round(design.std(axis=0), hazard.DECIMALS) == 0.0)
    if flat.all():
        raise ValueError("train_dead_ball: every column of the logistic design is constant")
    fitted = hazard.fit_logistic(design[:, ~flat], y, l2=LOGISTIC_L2, iterations=LOGISTIC_ITERATIONS)
    return {"impute": impute.tolist(), "indicators": [int(j) for j in indicators],
            "columns": [column for column, drop in zip(ids, flat) if not drop],
            "dropped": [column for column, drop in zip(ids, flat) if drop],
            "mean": [float(v) for v in fitted.mean], "scale": [float(v) for v in fitted.scale],
            "coef": [float(v) for v in fitted.coef], "intercept": float(fitted.intercept)}


def build(kind, gathered, profile, stride):
    """The artifact: one model per held-out match (fitted on the others), `full`, and the provenance."""
    if kind not in MODELS:
        raise ValueError("train_dead_ball: unknown model {!r}; use one of: {}".format(kind, ", ".join(MODELS)))
    ids = sorted(gathered.data)

    def fitted(match_ids):
        X = np.concatenate([gathered.data[match_id][0] for match_id in match_ids])
        y = np.concatenate([gathered.data[match_id][1] for match_id in match_ids])
        return fit(kind, X, y, play_features.FEATURE_NAMES)

    return {"format": FORMAT, "kind": kind, "feature_version": play_features.FEATURE_VERSION,
            "features": list(play_features.FEATURE_NAMES), "profile": profile, "seeds": gathered.seeds,
            "stride": stride, "provenance": provenance(gathered),
            "folds": {held_out: fitted([other for other in ids if other != held_out]) for held_out in ids},
            "full": fitted(ids)}


def train(kind, *, profile, seeds, stride, matches=None, calibration=None, data_dir=None, log=None):
    """The artifact for `kind`: gather the rows, then fit every fold and `full` (see the module docstring)."""
    if kind not in MODELS:
        raise ValueError("train_dead_ball: unknown model {!r}; use one of: {}".format(kind, ", ".join(MODELS)))
    gathered = gather(profile, seeds, stride, matches=matches, calibration=calibration, data_dir=data_dir, log=log)
    return build(kind, gathered, profile, stride)


def provenance(gathered):
    """What the model was made from: scikit-learn's version, the Calibration, the DFL files and our three sources."""
    import sklearn
    return {"scikit_learn": sklearn.__version__, "calibration_sha256": gathered.calibration_sha256,
            "dfl_sources": {match_id: gathered.dfl_sources[match_id] for match_id in sorted(gathered.dfl_sources)},
            "source_sha256": source_hashes()}


def source_hashes():
    """{file name: sha256} of the three files the model depends on, read with LF endings on every platform."""
    hashes = {}
    for name, path in SOURCE_FILES:
        with open(path, "rb") as handle:
            hashes[name] = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    return hashes


# -- the file ---------------------------------------------------------------------------------------------------------

def default_path(kind):
    return os.path.join(ROOT, "pipeline", "models", "dead_ball_{}.json".format(kind))


def dumps(artifact):
    """The artifact as it is written: sorted keys, Python's own float text (it reads back exactly), LF, no NaN."""
    try:
        return json.dumps(artifact, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise ValueError("train_dead_ball: the artifact holds a NaN or an infinity ({}); JSON cannot hold one, so "
                         "encode a missing value differently".format(error)) from error


def write(path, artifact):
    """Write the artifact to `path` (folders made), unless it would be over 10 MB, which is a ValueError."""
    data = dumps(artifact).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("train_dead_ball: the artifact is {:,} bytes and the limit is 10 MB ({:,}); fit fewer or "
                         "smaller trees".format(len(data), MAX_BYTES))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def compare(committed, fresh):
    """(problems, notes): what differs between the committed artifact and its retraining.

    Everything must be equal but the three source hashes of the provenance: a hash that differs in a file both name
    is a note, and a file named by one only is a problem like any other difference.
    """
    committed, fresh = json.loads(dumps(committed)), json.loads(dumps(fresh))
    notes, problems = [], []
    old = committed.get("provenance", {}).pop("source_sha256", None)
    new = fresh.get("provenance", {}).pop("source_sha256", None)
    if not isinstance(old, dict) or not isinstance(new, dict) or sorted(old) != sorted(new):
        problems.append("provenance.source_sha256: committed {} and retrained {} do not name the same files"
                        .format(sorted(old) if isinstance(old, dict) else old,
                                sorted(new) if isinstance(new, dict) else new))
    else:
        for name in sorted(old):
            if old[name] != new[name]:
                notes.append("the source hash of {} differs from the one the model was trained with: a comment edit "
                             "after training is not a different model, but retrain if the code changed".format(name))
    _differences(committed, fresh, "", problems)
    return problems[:MAX_PROBLEMS], notes


def _differences(old, new, path, out):
    if len(out) >= MAX_PROBLEMS:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            where = "{}.{}".format(path, key) if path else str(key)
            if key not in old or key not in new:
                out.append("{}: only in the {}".format(where, "committed file" if key in old else "retraining"))
            else:
                _differences(old[key], new[key], where, out)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append("{}: {} entries committed, {} retrained".format(path, len(old), len(new)))
            return
        for index, (a, b) in enumerate(zip(old, new)):
            _differences(a, b, "{}[{}]".format(path, index), out)
    elif old != new:
        out.append("{}: committed {!r}, retrained {!r}".format(path, old, new))


def verify(path, fresh):
    """Compare the committed file at `path` with the retrained `fresh`; print what differs; the exit code (0 or 1)."""
    try:
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
    except (OSError, ValueError) as error:
        print("FAILED    cannot read the committed model {} ({}); train it with `python {} --model {}`".format(
            path, error, "pipeline/benchmark/train_dead_ball.py", fresh["kind"]))
        return 1
    problems, notes = compare(committed, fresh)
    for note in notes:
        print("note      {}".format(note))
    for problem in problems:
        print("FAILED    {}".format(problem))
    if problems:
        print("FAILED    {} differs from its retraining in {} place(s){}".format(
            path, len(problems), " or more" if len(problems) >= MAX_PROBLEMS else ""))
        return 1
    print("verified  {} is what the code trains{}".format(path, " (source hashes aside)" if notes else ""))
    return 0


# -- the report -------------------------------------------------------------------------------------------------------

def held_out(artifact, data):
    """Per match, the out-of-fold log-loss, Brier score and AUC of the model that never saw it; the pooled reliability.

    {"per_match": {id: {rows, log_loss, brier, auc}}, "reliability": [{bin, count, mean_predicted, mean_observed}]}.
    The predictions are the models' own probabilities, before the source's prior correction.
    """
    per_match, counts, predicted, observed = {}, np.zeros(BINS, dtype=np.int64), np.zeros(BINS), np.zeros(BINS)
    for match_id in sorted(data):
        X, y = data[match_id]
        p = dead_ball_model.predict(artifact["folds"][match_id], X)
        clipped = np.clip(p, LOG_FLOOR, 1.0 - LOG_FLOOR)
        per_match[match_id] = {
            "rows": int(len(y)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))),
            "brier": float(np.mean((p - y) ** 2)), "auc": hazard.auc(p, y)}
        cell = np.minimum((p * BINS).astype(np.int64), BINS - 1)
        counts += np.bincount(cell, minlength=BINS)
        predicted += np.bincount(cell, weights=p, minlength=BINS)
        observed += np.bincount(cell, weights=y, minlength=BINS)
    table = [{"bin": "{:.1f}-{:.1f}".format(k / BINS, (k + 1) / BINS), "count": int(counts[k]),
              "mean_predicted": float(predicted[k] / counts[k]) if counts[k] else None,
              "mean_observed": float(observed[k] / counts[k]) if counts[k] else None} for k in range(BINS)]
    return {"per_match": per_match, "reliability": table}


def render_report(report):
    """The report's lines: one per match, then the pooled reliability table."""
    lines = ["held out  {:<8} {:>9} {:>9} {:>9} {:>9}".format("match", "rows", "log-loss", "Brier", "AUC")]
    for match_id, scores in sorted(report["per_match"].items()):
        lines.append("held out  {:<8} {:>9,} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            match_id, scores["rows"], scores["log_loss"], scores["brier"], scores["auc"]))
    lines.append("reliability, pooled over the held-out matches   {:<9} {:>9} {:>10} {:>10}".format(
        "bin", "rows", "predicted", "observed"))
    for cell in report["reliability"]:
        lines.append("reliability  {:<9} {:>9,} {:>10} {:>10}".format(
            cell["bin"], cell["count"],
            "-" if cell["mean_predicted"] is None else "{:.4f}".format(cell["mean_predicted"]),
            "-" if cell["mean_observed"] is None else "{:.4f}".format(cell["mean_observed"])))
    return lines


# -- the rules' values ------------------------------------------------------------------------------------------------

def trigger_masks(inputs):
    """{trigger: bool per frame}: where each of dead_ball_rules' triggers fires, read off the rules by probing them.

    With one probability at 0 and the others at 1, `alive` is 0 exactly where that trigger fires (the smallest value
    of the rules that fire wins), so the masks are the rules' own and a threshold changed there is changed here.
    """
    masks = {}
    for trigger in TRIGGERS:
        values = {"alive_" + name: 1.0 for name in TRIGGERS}
        values["alive_" + trigger] = 0.0
        alive = dead_ball_rules.compute(RunContext(inputs=inputs, seed=0), **values).alive
        masks[trigger] = alive == 0.0
    return masks


def count_triggers(inputs, codes):
    """{"alive", "dead", trigger: (frames alive with it, frames dead with it)} over the graded frames of `inputs`."""
    graded = np.asarray(codes) != pm.X
    alive, dead = graded & (np.asarray(codes) != DEAD), graded & (np.asarray(codes) == DEAD)
    counts = {"alive": int(alive.sum()), "dead": int(dead.sum())}
    for trigger, mask in trigger_masks(inputs).items():
        counts[trigger] = (int((mask & alive).sum()), int((mask & dead).sum()))
    return counts


def towards_half(value, decimals=2):
    """`value` rounded to `decimals` places towards 0.5: down from above it, up from below, never past it."""
    scale = 10 ** decimals
    scaled = round(value * scale, 9)               # 0.29 * 100 is 28.999999999999996: the cell is still 29
    rounded = math.floor(scaled) if value > 0.5 else math.ceil(scaled) if value < 0.5 else scaled
    return float(rounded) / scale


def rule_values(counts):
    """{"alive_<trigger>": {value, p_given_alive, p_given_dead, fired}} from the counts of `count_triggers`.

    value = P(t | alive) / (P(t | alive) + P(t | dead)) rounded towards 0.5, which is 0.5 where the trigger never fires
    (no information). ValueError for a class with no frames or a trigger that fires on more frames than its class has.
    """
    totals = {"alive": counts["alive"], "dead": counts["dead"]}
    for name, total in totals.items():
        if total <= 0:
            raise ValueError("train_dead_ball: no {} frame among the graded frames, so P(trigger | {}) is undefined"
                             .format(name, name))
    values = {}
    for trigger in TRIGGERS:
        with_alive, with_dead = counts[trigger]
        if not 0 <= with_alive <= totals["alive"] or not 0 <= with_dead <= totals["dead"]:
            raise ValueError("train_dead_ball: trigger {} fires on {} alive and {} dead frames, of {} and {}"
                             .format(trigger, with_alive, with_dead, totals["alive"], totals["dead"]))
        p_alive, p_dead = with_alive / totals["alive"], with_dead / totals["dead"]
        values["alive_" + trigger] = {
            "value": 0.5 if p_alive + p_dead == 0.0 else towards_half(p_alive / (p_alive + p_dead)),
            "p_given_alive": p_alive, "p_given_dead": p_dead, "fired": with_alive + with_dead}
    return values


def measure_rules(profile, seeds, matches=None, calibration=None, data_dir=None, match_ids=None, log=None):
    """The counts of `count_triggers`, summed over every match and seed of `profile`."""
    seeds = _whole("seeds", seeds, 1)
    if profile not in run.PROFILES:
        raise BenchmarkError("train_dead_ball: profile {!r} is not one of: {}".format(profile,
                                                                                      ", ".join(sorted(run.PROFILES))))
    ids = run.resolve_match_ids(match_ids, matches)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    log = log or (lambda text: None)
    total = {"alive": 0, "dead": 0, **{trigger: (0, 0) for trigger in TRIGGERS}}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)
        for seed in seed_list:
            counts = count_triggers(served_inputs(match, profile, seed, calibration, data_dir), codes)
            for key in total:
                total[key] = (tuple(a + b for a, b in zip(total[key], counts[key])) if key in TRIGGERS
                              else total[key] + counts[key])
        log("rules     {}".format(match_id))
        del match, codes
    return total


def render_rules(counts, values):
    lines = ["rules     {:,} alive and {:,} dead graded frames".format(counts["alive"], counts["dead"]),
             "rules     {:<14} {:>14} {:>14} {:>8}".format("trigger", "P(t | alive)", "P(t | dead)", "value")]
    for trigger in TRIGGERS:
        entry = values["alive_" + trigger]
        lines.append("rules     {:<14} {:>14.4f} {:>14.4f} {:>8.2f}".format(
            "alive_" + trigger, entry["p_given_alive"], entry["p_given_dead"], entry["value"]))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train P05's dead-ball model on the benchmark matches (--model), or measure the values of "
                    "dead_ball_rules' five parameters on them (--rules).")
    parser.add_argument("--model", choices=MODELS, help="the model to train and write: hgb, or the logistic reference")
    parser.add_argument("--rules", action="store_true",
                        help="print P(trigger | alive), P(trigger | dead) and the rule values instead of training")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="every this many graded frames is a row (default {})".format(DEFAULT_STRIDE))
    parser.add_argument("--out", help="where to write the artifact (default pipeline/models/dead_ball_<model>.json)")
    parser.add_argument("--verify", action="store_true",
                        help="retrain and compare with the committed file instead of writing it: exit 1 if "
                             "anything but the three source hashes differs")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    given = {text.split("=")[0] for text in (sys.argv[1:] if argv is None else argv)}
    if args.rules == (args.model is not None):
        parser.error("give exactly one of --model hgb|logistic and --rules")
    if args.rules and (args.verify or args.out or "--stride" in given):
        parser.error("--rules takes no --verify, --out or --stride: it measures and prints, and reads every graded "
                     "frame")
    try:
        if args.rules:
            counts = measure_rules(args.profile, args.seeds, data_dir=args.data_dir, log=print)
            for line in render_rules(counts, rule_values(counts)):
                print(line)
            return 0
        gathered = gather(args.profile, args.seeds, args.stride, data_dir=args.data_dir, log=print)
        artifact = build(args.model, gathered, args.profile, args.stride)
        for line in render_report(held_out(artifact, gathered.data)):
            print(line)
        path = args.out or default_path(args.model)
        if args.verify:
            return verify(path, artifact)
        write(path, artifact)
        print("wrote     {} ({:,} bytes)".format(path, os.path.getsize(path)))
        return 0
    except (BenchmarkError, OSError) as error:
        # The expected failures get a sentence, as in run.py: a profile or match that does not exist, data not
        # downloaded yet. Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1


def _whole(name, value, minimum):
    """`value` as an int, or ValueError unless it is a whole number of at least `minimum` (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError("train_dead_ball: {} must be a whole number of at least {}, got {!r}"
                         .format(name, minimum, value))
    return int(value)


if __name__ == "__main__":
    sys.exit(main())
