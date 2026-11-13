"""P07's trainer: fit the shape-possession model on the benchmark's matches and write it as JSON.

    python pipeline/benchmark/train_shape.py [--profile tracking] [--seeds 3] [--stride 5] [--features {1,2}]
                                             [--out PATH] [--verify] [--report] [--data-dir D]

    artifact = train(profile="tracking", seeds=3, stride=5)             # the dict that is written, JSON-ready
    held_out = report(artifact, profile="tracking", seeds=3)            # aggregates of each fold on its own match
    print(touch_agreement(inputs))                                      # {"frames": ..., "agreement": ...}

What it learns, and from what
-----------------------------
The classifier answers one question, which team has the ball, from the twenty-six `shape_features` of a frame
(version 1; version 2, P30, has 42 and goes in its own file, `--features 2`), none of which reads the ball
(pipeline/evidence/shape_features.py). The seven open DFL/IDSSE matches are the only ones with a flag for it
(`BallPossession`), so those are the training data, degraded to look like our tracking (profile `tracking`, version 3,
seeds 0 to 2: P01) and served as `MatchInputs`, so that the features are computed by the very code and from the very
kind of inputs that the submission pipeline will give them. No truth reaches the features: the label is read from
`truth.truth_codes(match)` beside them, never through the inputs (test_shape_trainer_never_passes_truth). A row is
every `stride`-th frame of a graded block from its start, kept where the truth is H or A (alive: who has the ball is
not asked of a dead ball), and 1 where it is H. Stride 5 (0.2 s) keeps the rows deterministic and cuts the
autocorrelation, since neighbouring frames carry nearly the same features and the same label.

Why P05's recipe
----------------
The model is scikit-learn's `HistGradientBoostingClassifier` with the settings that P05 fixed for the dead-ball model
(`train_dead_ball.HGB_SETTINGS`, imported and not retyped):

    loss log_loss, learning_rate 0.05, max_iter 300, max_leaf_nodes 15, min_samples_leaf 1000,
    l2_regularization 1.0, max_bins 255, random_state 0, no class weights, no early stopping.

They are the literature survey's (docs/proposals.md P05), fixed beforehand and never tuned (T26: a score to be tuned
towards would have to come from the public window or from the benchmark that also scores the submission). Boosting is
the family that Lang et al. 2022, the one direct study of ball status from tracking, found best; the rest is the
conservative end of the usual defaults, and `min_samples_leaf` 1000 keeps a leaf wide enough that it holds several
stretches of play and not one run of neighbouring, nearly identical frames. There is no early stopping, since a random
10% hold-out of autocorrelated frames leaks. Unlike P05's model there are no monotone constraints: nothing in the laws
of the game says that the team with the ball must be further up the pitch or wider, only that it usually is.

The augmentation
----------------
The two teams are interchangeable in what the features mean, and a model has no business knowing which of them is
home. So every row is shown twice: as it is, and as `shape_features.swap(row)`, the features of the same frame with the
two teams exchanged, with the label `1 - y`. This makes the classes balance exactly (half the labels are 1, so the
model's `prior_home` is 0.5, and a classifier that is an honest likelihood is what the decoder wants, with no prior to
subtract) and teaches the symmetry. It does not make the fitted classifier exactly antisymmetric; `shape_possession`
makes the output so, by (r(x) - r(swap(x))) / 2. The mirror rows follow the rows, in the same order, so the fit is
deterministic. A fold is fitted on the other matches' rows and their mirrors: the held-out match contributes neither.

The report's coarse cells (P34)
-------------------------------
The objective that ranks a choice is the mean of two truths, `raw` and `coarse` (`truth.coarse_codes`: short
possessions between two of the other team merged). `--report` scores the held-out folds against both: the raw cells as
always, and `report["coarse"]` with the coarse truth's log-loss and accuracy on the same frames (the coarse codes are
built from the raw ones with the served inputs' graded blocks and fps, never from the features). They only report; the
training uses the raw labels alone and nothing gates on them.

Folds, the artifact, `--verify`
-------------------------------
For each match m there is a model fitted on the other six, and `full`, fitted on all seven: a benchmark match is scored
by the model that never saw it, and BAR-ATH by `full` (pipeline/evidence/shape_possession.py). The file is
pipeline/models/shape_possession_hgb.json, `{format, kind, feature_version, features, profile, seeds, stride,
provenance, folds: {match_id: model}, full: model}`, each model `{baseline, trees, prior_home}`, written with sorted
keys and Python's own float text, which reads back exactly, LF, no NaN, under 10 MB. The trees are the node arrays that
`train_dead_ball.export_hgb` reads out of the fitted predictors (a private structure of scikit-learn, which is why the
version is pinned in requirements-benchmark.txt), and `trees.check_trees` validates. The keys of `folds` are the DFL
match ids, because that is `inputs.game` of a served match and how `shape_possession` picks the held-out model.

`provenance` is the scikit-learn version, the Calibration's digest, the DFL files' sha256 and the sha256 of the three
source files that the model depends on (shape_features.py, trees.py and this file). It has no git commit: that changes
with every later commit and would make the file differ from its own retraining for no reason. `--verify` retrains and
compares everything but those three source hashes and exits 1 if anything differs; a differing source hash is a note
and not a failure, because a comment edited after training is not a different model, and the features it was trained
on are still pinned by `feature_version`.

--report, and the stop rule
---------------------------
`report` scores each fold model on its own held-out match, served as the pipeline serves it, all seeds pooled per
match: `shape_possession` with scope "all", so that the numbers are of the source as the pipeline runs it (symmetrised,
clipped to [0.02, 0.98]), against the truth on the alive frames: log-loss and accuracy; the same on the alive frames
where the served ball is missing (`not atd.ball_present`), which are the frames P07 exists for; `ln 2`, the log-loss of
saying 0.5; and `touch_agreement`, below; then the same pooled over the matches by frames, never a median of medians.
A frame where the source has no opinion (nobody present) is counted apart (`silent`) and not scored. A fold that the
artifact lacks is an error, since the source would fall back to `full`, which saw the match, and score it in sample.

**The stop rule** (docs/proposals.md P07): if the pooled held-out log-loss on alive frames with the served ball
missing is not below ln 2 - 0.01 = 0.6831, the shape of the teams says too little about who has the ball for P07 to go
on. P07 then stops after this task, the negative result is recorded, and the rest of the lane is skipped. `report`
says whether it fires.

`touch_agreement(inputs, model)` is the statistic that `report` uses for a cross-check that needs no truth: among the
frames inside kept `controlled_touch` episodes (the rule of `conventions.TOUCH`, with confidence 0.99 and hold 0.5, so
that `home` is a number exactly on the kept frames and >= 0.5 means a home episode), the share where the shape home
(scope "all", >= 0.5) names the episode's team. It takes any Inputs, so it can be asked of our own match, where it is
the one label-free look at how the model's guess agrees with the touches that are certain enough to count.

Determinism: nothing random but HGB's random_state, no set iterated, folds in match-id order; the same inputs and
versions give the same bytes on any number of threads (test_shape_training_is_deterministic_across_threads).
"""

import argparse
import hashlib
import json
import math
import numbers
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import conventions, degrade, dfl, run  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
# `dumps` is imported to be re-exported: the checks and later tools write the artifact's text with the trainer's own.
from pipeline.benchmark.train_dead_ball import (  # noqa: E402
    HGB_SETTINGS, LOG_FLOOR, Gathered, compare, dumps, export_hgb, served_inputs, write)
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, shape_features, shape_possession, team_features, trees  # noqa: E402

import possession_metrics as pm  # noqa: E402  (importing pipeline above put evaluation/ on sys.path)

# Params
FORMAT = trees.FORMAT
KIND = "hgb"
HANDOVER, SAME_TEAM, UNKNOWN = 1, 2, 3        # gap_classes' codes (0: not a graded frame with the ball missing)
GAP_CELLS = ("gap_handover", "gap_same_team", "gap_unknown")
HOME, AWAY = pm.H, pm.A                       # the codes of the truth that are alive; a dead ball has no team to name
DEFAULT_STRIDE = 5
PRIOR_HOME = shape_possession.PRIOR_HOME      # the share of homes among the rows and their mirrors
# One file per feature version (P30): the file names its version, and `shape_possession` reads the features of it.
DEFAULT_MODEL_PATHS = {1: "pipeline/models/shape_possession_hgb.json",
                       2: "pipeline/models/shape_possession_v2_hgb.json"}     # relative to the repository root
DEFAULT_MODEL_PATH = os.path.join(ROOT, *DEFAULT_MODEL_PATHS[1].split("/"))
SOURCE_FILES = (("train_shape.py", os.path.join(HERE, "train_shape.py")),
                ("shape_features.py", os.path.join(ROOT, "pipeline", "evidence", "shape_features.py")),
                ("trees.py", os.path.join(ROOT, "pipeline", "evidence", "trees.py")))
HOME_LEVEL = 0.5                              # a home probability at or above this names home
LN2 = math.log(2.0)                           # the log-loss of saying 0.5 on every frame
STOP_MARGIN = 0.01                            # the Design's: the model must beat ln 2 by more than this to go on
STOP_LOSS = LN2 - STOP_MARGIN                 # 0.6831


# -- the rows ---------------------------------------------------------------------------------------------------------

def rows(inputs, codes, stride, version=1):
    """(X, y): the features (of `shape_features` `version`) of every `stride`-th graded frame from each block's start
    where the truth is H or A, and 1 where it is H.

    `codes` is the truth's int8 pm codes per frame of the match, kept apart from the inputs, which have none. A dead
    ball (D) and a frame outside the blocks (X) are no rows: nobody has the ball there.
    """
    stride = _whole("stride", stride, 1)
    codes = np.asarray(codes)
    if codes.shape != (inputs.n_frames,):
        raise ValueError("train_shape: codes must hold one code per frame of the match ({}), got shape {}"
                         .format(inputs.n_frames, codes.shape))
    frames = np.concatenate([np.arange(start, stop, stride) for _, start, stop in inputs.graded])
    frames = frames[(codes[frames] == HOME) | (codes[frames] == AWAY)]
    _, X = shape_features.features(inputs, version)
    return X[frames], (codes[frames] == HOME).astype(np.int64)


def gather(profile, seeds, stride, matches=None, calibration=None, data_dir=None, match_ids=None, log=None,
           version=1):
    """Gathered: the rows of every match under every seed of `profile`, served as the pipeline serves them.

    `version` is the feature version of the rows (`shape_features.VERSIONS`; ValueError for any other).

    `matches` is {id: DflMatch} to use instead of loading the seven from `data_dir` (the checks pass synthetic ones);
    `calibration` is the Calibration that a degraded profile takes its masks from, measured or read from its cache when
    none is given. BenchmarkError for an unknown profile, fewer than two matches (a fold needs another match to be
    fitted on), a match filed under another id than its own, and a calibration given to the clean profile, which
    degrades nothing and so must not suggest it was used.
    """
    seeds = _whole("seeds", seeds, 1)
    _whole("stride", stride, 1)
    shape_features.feature_names(version)               # refuses an unknown version before any match is loaded
    _known_profile(profile)
    ids = run.resolve_match_ids(match_ids, matches)
    if len(ids) < 2:
        raise BenchmarkError("train_shape: a fold is fitted on the other matches, so it needs at least two matches, "
                             "got {}".format(len(ids)))
    calibration = _calibration_for(profile, calibration, data_dir)
    seed_list = _seed_list(profile, seeds)
    log = log or (lambda text: None)

    data, sources = {}, {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        _own_id(match, match_id)
        codes, _ = truths.truth_codes(match)
        parts = []
        for seed in seed_list:
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            parts.append(rows(inputs, codes, stride, version))
            if seed == seed_list[0]:
                recorded = inputs.atd_manifest["sources"]
                sources[match_id] = None if recorded is None else {name: recorded[name]["sha256"]
                                                                   for name in sorted(recorded)}
            del inputs
        data[match_id] = (np.concatenate([X for X, _ in parts]), np.concatenate([y for _, y in parts]))
        log("rows      {}   {:,} rows, {:.1%} home".format(match_id, len(data[match_id][1]),
                                                         float(data[match_id][1].mean())))
        del match, codes
    digest = calibration_module.digest(calibration) if run.needs_calibration(profile) else None
    return Gathered(data, digest, sources, len(seed_list))


# -- the model --------------------------------------------------------------------------------------------------------

def augment(X, y, version=1):
    """(X', y'): the rows, then their mirror images (`shape_features.swap`) with the label flipped.

    The classes balance exactly, whatever the rows' own balance: half the labels of y' are 1. The rows come first and
    unchanged, so the order is the input's and the fit is deterministic. ValueError for X that is not (n, 26)
    (version 2: (n, 42)) or a y that is not one 0 or 1 per row.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    names = len(shape_features.feature_names(version))
    if X.ndim != 2 or X.shape[1] != names or y.shape != (len(X),) or not np.isin(y, (0, 1)).all():
        raise ValueError("train_shape: X must be (n, {}) and y one 0 or 1 per row, got X {} and y {}"
                         .format(names, X.shape, y.shape))
    return np.concatenate([X, shape_features.swap(X)]), np.concatenate([y, 1 - y]).astype(np.int64)


def hgb_estimator(X, y):
    """The fitted `HistGradientBoostingClassifier` with P05's fixed settings and no monotone constraint, on X and y as
    they are (the caller has augmented them)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**HGB_SETTINGS).fit(np.asarray(X, dtype=np.float64),
                                                              np.asarray(y, dtype=np.int64))


def fit(X, y, version=1):
    """One model, as a JSON-ready dict {baseline, trees, prior_home}: fitted on the rows X, y and their mirrors.

    ValueError for the shapes `augment` refuses and for rows of one class only: a model of who has the ball needs both
    homes and aways (the mirror rows would teach the symmetry alone, from a sample that says nothing of the game).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    augmented = augment(X, y, version)              # checks the shapes
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("train_shape: the rows hold one class only ({} home among {}); a model needs both home and "
                         "away frames".format(int(y.sum()), len(y)))
    model = export_hgb(hgb_estimator(*augmented))
    model["prior_home"] = PRIOR_HOME
    return model


def build(gathered, profile, stride, version=1):
    """The artifact: one model per held-out match (fitted on the others), `full`, and the provenance.

    `version` names the feature version of the rows in `gathered`; the artifact's `feature_version` and `features` are
    that version's, which is what `shape_possession` reads back to know which columns to compute."""
    names = shape_features.feature_names(version)
    ids = sorted(gathered.data)

    def fitted(match_ids):
        X = np.concatenate([gathered.data[match_id][0] for match_id in match_ids])
        y = np.concatenate([gathered.data[match_id][1] for match_id in match_ids])
        return fit(X, y) if version == 1 else fit(X, y, version)      # version 1 is the call that was always made

    return {"format": FORMAT, "kind": KIND, "feature_version": version, "features": list(names), "profile": profile,
            "seeds": gathered.seeds, "stride": stride, "provenance": provenance(gathered),
            "folds": {held_out: fitted([other for other in ids if other != held_out]) for held_out in ids},
            "full": fitted(ids)}


def train(profile, seeds, stride, matches=None, calibration=None, data_dir=None, log=None, version=1):
    """The artifact: gather the rows, then fit every fold and `full` (see the module docstring)."""
    gathered = gather(profile, seeds, stride, matches=matches, calibration=calibration, data_dir=data_dir, log=log,
                      version=version)
    return build(gathered, profile, stride, version)


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


# -- --verify ---------------------------------------------------------------------------------------------------------

def verify(path, fresh):
    """Compare the committed file at `path` with the retrained `fresh`; print what differs; the exit code (0 or 1)."""
    try:
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
    except (OSError, ValueError) as error:
        print("FAILED    cannot read the committed model {} ({}); train it with `python {}`".format(
            path, error, "pipeline/benchmark/train_shape.py"))
        return 1
    problems, notes = compare(committed, fresh)
    for note in notes:
        print("note      {}".format(note))
    for problem in problems:
        print("FAILED    {}".format(problem))
    if problems:
        print("FAILED    {} differs from its retraining in {} place(s){}".format(
            path, len(problems), " or more" if len(problems) >= 20 else ""))
        return 1
    print("verified  {} is what the code trains{}".format(path, " (source hashes aside)" if notes else ""))
    return 0


# -- the report -------------------------------------------------------------------------------------------------------

def touch_agreement(inputs, model=DEFAULT_MODEL_PATH):
    """{"frames", "agreement"}: how often the shape says the team that has an episode of control.

    Among the frames inside the kept `controlled_touch` episodes of `conventions.TOUCH` (the rule of main.toml, with
    confidence 0.99 and hold 0.5: `home` is then a number exactly on the kept frames, 0.99 on a home episode and 0.01
    on an away one) where `shape_possession` (scope "all", the model of `model`: the fold of `inputs.game` if it has
    one, else `full`) has an opinion, "frames" counts them and "agreement" is the share where (shape home >= 0.5)
    equals (the episode's team is home), or None if there is no such frame. Any Inputs will do, a benchmark match or our
    own; no truth is read, and only two numbers come back.
    """
    run_context = RunContext(inputs=inputs, seed=0)
    shape = shape_possession.compute(run_context, model=model, scope="all").home
    episode = controlled_touch.compute(run_context, **conventions.TOUCH).home
    kept = ~np.isnan(episode) & ~np.isnan(shape)
    frames = int(kept.sum())
    agreed = int(((shape[kept] >= HOME_LEVEL) == (episode[kept] >= HOME_LEVEL)).sum())
    return {"frames": frames, "agreement": agreed / frames if frames else None}


def stop_rule_fires(log_loss):
    """True unless `log_loss` (the pooled held-out log-loss on alive frames with the ball missing) is below ln 2 - 0.01.

    At the threshold the rule fires: the model must be shown to beat saying 0.5 by that much, not to match it.
    """
    if isinstance(log_loss, bool) or not isinstance(log_loss, numbers.Real) or math.isnan(log_loss):
        raise ValueError("train_shape: the stop rule needs the pooled held-out log-loss on alive frames with the ball "
                         "missing, a number, got {!r}".format(log_loss))
    return not log_loss < STOP_LOSS


def report(artifact, profile, seeds, matches=None, calibration=None, data_dir=None):
    """Aggregates of each fold model on its own held-out match, all seeds pooled, and the same pooled over the matches.

    {profile, seeds, ln2, per_match: {id: cell}, pooled: cell, stop_rule: {threshold, log_loss, fires}, coarse:
    {per_match: {id: ccell}, pooled: ccell}} where a cell is {alive: {frames, silent, log_loss, accuracy}, missing:
    {frames, log_loss, accuracy}, touch: {frames, agreement}}: the symmetrised, clipped `home` of `shape_possession`
    (scope "all") against the truth on the alive frames, on those where the served ball is missing, and
    `touch_agreement`. `coarse` (P34) is the same two scores against the coarse truth (`truth.coarse_codes` of the raw
    codes, per seed, with the served inputs' blocks and fps), a ccell being {alive: {frames, log_loss, accuracy},
    missing: {...}}; the alive frames are the raw truth's (ValueError if the coarse truth's differ). `gap_classes`
    (P37's cells, kept on `team_features`' touch rule, for every feature version, independent of the model's columns) is {per_match: {id: gcell}, pooled: gcell}, a
    gcell being {handover, same_team, unknown}, each {frames, log_loss, accuracy} of the alive frames with the ball
    missing whose gap, by the touch rule, has a different last controller before than next after, the same one, or
    none on a side (`gap_classes`). Numbers and strings only, no per-frame array.

    BenchmarkError for an unknown profile, for an artifact that lacks the fold of one of the matches (the source would
    score it with `full`, which saw it) and for matches with no alive frame whose ball is missing (the stop rule would
    have nothing to be measured on). The artifact is written to a temporary file first and read back by the source, so
    what is scored is the file as it would be committed.
    """
    seeds = _whole("seeds", seeds, 1)
    _known_profile(profile)
    ids = run.resolve_match_ids(None, matches)
    folds = artifact.get("folds") if isinstance(artifact, dict) else None
    absent = [match_id for match_id in sorted(ids) if not isinstance(folds, dict) or match_id not in folds]
    if absent:
        raise BenchmarkError("train_shape: the artifact has no fold for {}; the source would score that match with "
                             "`full`, which saw it, and the numbers would be in sample; retrain with `python {}`"
                             .format(", ".join(absent), "pipeline/benchmark/train_shape.py"))
    calibration = _calibration_for(profile, calibration, data_dir)
    seed_list = _seed_list(profile, seeds)

    tallies = {}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, os.path.basename(DEFAULT_MODEL_PATH))
        write(path, artifact)
        for match_id in sorted(ids):
            match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
            _own_id(match, match_id)
            codes, _ = truths.truth_codes(match)
            tallies[match_id] = _tally(match, codes, profile, seed_list, calibration, data_dir, path)
            del match, codes
    pooled = _empty_tally()
    for tally in tallies.values():
        _add(pooled, tally)
    pooled_cell = _cell(pooled)
    if pooled_cell["missing"]["frames"] == 0:
        raise BenchmarkError("train_shape: no alive frame of these matches has the ball missing under profile {!r}, so "
                             "the stop rule has nothing to be measured on; use a profile that hides the ball (gaps "
                             "or tracking)".format(profile))
    loss = pooled_cell["missing"]["log_loss"]
    return {"profile": profile, "seeds": len(seed_list), "ln2": LN2,
            "per_match": {match_id: _cell(tallies[match_id]) for match_id in sorted(tallies)},
            "pooled": pooled_cell,
            "stop_rule": {"threshold": STOP_LOSS, "log_loss": loss, "fires": stop_rule_fires(loss)},
            "coarse": {"per_match": {match_id: _coarse_cell(tallies[match_id]) for match_id in sorted(tallies)},
                       "pooled": _coarse_cell(pooled)},
            "gap_classes": {"per_match": {match_id: _gap_cell(tallies[match_id]) for match_id in sorted(tallies)},
                            "pooled": _gap_cell(pooled)}}


def gap_classes(inputs):
    """An int8 array over every frame: for the graded frames where the ball is missing, the kind of gap by the touch
    rule at `team_features.TOUCH_RULE` (P37's cells), whatever the model's columns are; 0 elsewhere.

    1 `HANDOVER`: the team that last had a kept control frame before the frame, in its block, is not the team that has
    the next one after it; 2 `SAME_TEAM`: it is; 3 `UNKNOWN`: there is none on one side (or both). A frame is classified
    by the kept frames on either side of it in its own block, never across half time. Reads the ball (the rule does),
    never the truth.
    """
    touch = controlled_touch.compute(RunContext(inputs=inputs, seed=0), **team_features.TOUCH_RULE, confidence=0.99,
                                     hold=0.5).home                 # the kept frames, as team_features reads them
    kept = np.isfinite(touch)
    home, away = kept & (touch >= 0.5), kept & (touch < 0.5)
    out = np.zeros(len(kept), dtype=np.int8)
    missing = ~inputs.atd.ball_present
    for _, start, stop in inputs.graded:
        if stop <= start:
            continue
        m = stop - start
        index = np.arange(m)
        mask = kept[start:stop]
        before = np.maximum.accumulate(np.where(mask, index, -1))
        after = np.minimum.accumulate(np.where(mask, index, m)[::-1])[::-1]
        is_home = home[start:stop]
        has = (before >= 0) & (after < m)
        differ = has & (is_home[np.maximum(before, 0)] != is_home[np.minimum(after, m - 1)])
        kind = np.where(~has, UNKNOWN, np.where(differ, HANDOVER, SAME_TEAM))
        out[start:stop] = np.where(missing[start:stop], kind, 0)
    return out


def _empty_tally():
    """The sums behind a cell: [frames, summed log-loss, frames right] for alive and for ball-missing frames, the
    alive frames the source was silent on, and [touch frames, frames that agree]; and, for the gap classes (P37), the
    same triple on the alive ball-missing frames of each class. Sums pool; ratios would not."""
    return {"alive": [0, 0.0, 0], "missing": [0, 0.0, 0], "silent": 0, "touch": [0, 0],
            "coarse_alive": [0, 0.0, 0], "coarse_missing": [0, 0.0, 0],
            "gap_handover": [0, 0.0, 0], "gap_same_team": [0, 0.0, 0], "gap_unknown": [0, 0.0, 0]}


def _add(total, part):
    for name in ("alive", "missing", "touch", "coarse_alive", "coarse_missing") + GAP_CELLS:
        total[name] = [a + b for a, b in zip(total[name], part[name])]
    total["silent"] += part["silent"]


def _tally(match, codes, profile, seed_list, calibration, data_dir, path):
    """The sums of one match under every seed, scored by the model at `path`, which is the fold of the match."""
    codes = np.asarray(codes)
    alive = (codes == HOME) | (codes == AWAY)
    is_home = codes == HOME
    tally = _empty_tally()
    for seed in seed_list:
        inputs = served_inputs(match, profile, seed, calibration, data_dir)
        coarse = truths.coarse_codes(codes, inputs.graded, inputs.fps)
        if ((coarse == HOME) | (coarse == AWAY) != alive).any():
            raise ValueError("train_shape: the coarse truth has other alive frames than the raw one; it only "
                             "relabels possessions, never a dead ball")
        coarse_home = coarse == HOME
        home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=path, scope="all").home
        spoken = ~np.isnan(home)
        part = _empty_tally()
        part["silent"] = int((alive & ~spoken).sum())
        kinds = gap_classes(inputs)
        for name, wanted in (("alive", alive), ("missing", alive & ~inputs.atd.ball_present),
                             ("gap_handover", alive & (kinds == HANDOVER)),
                             ("gap_same_team", alive & (kinds == SAME_TEAM)),
                             ("gap_unknown", alive & (kinds == UNKNOWN))):
            take = wanted & spoken
            p = np.clip(home[take], LOG_FLOOR, 1.0 - LOG_FLOOR)
            scored = [(name, is_home)] if name in GAP_CELLS else [(name, is_home), ("coarse_" + name, coarse_home)]
            for cell, homes in scored:                  # the gap classes are scored against the raw truth only
                truly_home = homes[take]
                loss = -np.where(truly_home, np.log(p), np.log(1.0 - p))
                part[cell] = [int(take.sum()), float(loss.sum()),
                              int(((home[take] >= HOME_LEVEL) == truly_home).sum())]
        touch = touch_agreement(inputs, model=path)
        if touch["frames"]:
            part["touch"] = [touch["frames"], int(round(touch["agreement"] * touch["frames"]))]
        _add(tally, part)
        del inputs, home
    return tally


def _cell(tally):
    """The JSON-ready cell of a tally: frames, mean log-loss and accuracy (None where there is no frame)."""
    def scored(name):
        frames, loss, right = tally[name]
        return {"frames": frames, "log_loss": loss / frames if frames else None,
                "accuracy": right / frames if frames else None}
    frames, agreed = tally["touch"]
    return {"alive": dict(scored("alive"), silent=tally["silent"]), "missing": scored("missing"),
            "touch": {"frames": frames, "agreement": agreed / frames if frames else None}}


def _gap_cell(tally):
    """The JSON-ready gap-class cell of a tally (P37): {handover, same_team, unknown}, each {frames, log_loss,
    accuracy} of the alive frames with the ball missing in a gap of that kind (`gap_classes`), the raw truth's."""
    def scored(name):
        frames, loss, right = tally[name]
        return {"frames": frames, "log_loss": loss / frames if frames else None,
                "accuracy": right / frames if frames else None}
    return {"handover": scored("gap_handover"), "same_team": scored("gap_same_team"),
            "unknown": scored("gap_unknown")}


def _coarse_cell(tally):
    """The JSON-ready coarse cell of a tally: {alive, missing}, each {frames, log_loss, accuracy}."""
    def scored(name):
        frames, loss, right = tally[name]
        return {"frames": frames, "log_loss": loss / frames if frames else None,
                "accuracy": right / frames if frames else None}
    return {"alive": scored("coarse_alive"), "missing": scored("coarse_missing")}


def render_report(report):
    """The report's lines: a table of the matches and the pool, then ln 2 and the stop rule."""
    def number(value, spec):
        return "-" if value is None else format(value, spec)

    lines = ["held out  profile {}, {} seed(s): each fold model on its own match, seeds pooled".format(
                 report["profile"], report["seeds"]),
             "held out  {:<8} | {:^36} | {:^28} | {:^19}".format(
                 "", "alive frames", "alive, ball missing", "touch episodes"),
             "held out  {:<8} | {:>10} {:>7} {:>8} {:>8} | {:>10} {:>8} {:>8} | {:>9} {:>9}".format(
                 "match", "frames", "silent", "log-loss", "accuracy", "frames", "log-loss", "accuracy", "frames",
                 "agreement")]
    for name, cell in list(sorted(report["per_match"].items())) + [("pooled", report["pooled"])]:
        alive, missing, touch = cell["alive"], cell["missing"], cell["touch"]
        lines.append("held out  {:<8} | {:>10,} {:>7,} {:>8} {:>8} | {:>10,} {:>8} {:>8} | {:>9,} {:>9}".format(
            name, alive["frames"], alive["silent"], number(alive["log_loss"], ".4f"), number(alive["accuracy"], ".4f"),
            missing["frames"], number(missing["log_loss"], ".4f"), number(missing["accuracy"], ".4f"),
            touch["frames"], number(touch["agreement"], ".4f")))
    rule = report["stop_rule"]
    lines.append("held out  ln 2 = {:.4f}, the log-loss of saying 0.5 on every frame".format(report["ln2"]))
    lines.append("stop rule  pooled log-loss on alive frames with the ball missing {:.4f}; the model must be below "
                 "{:.4f} (ln 2 - {})".format(rule["log_loss"], rule["threshold"], STOP_MARGIN))
    lines.append("stop rule  {}".format(
        "FIRES: P07 stops here, the result is a negative one and the rest of the lane is skipped" if rule["fires"]
        else "does not fire: P07 goes on"))
    if "gap_classes" in report:                         # P37: the gaps by the touch rule, before the coarse lines
        lines.append("gaps      alive frames, ball missing, by the touch rule: handover (controller before != after), "
                     "same team, unknown")
        lines.append("gaps      {:<8} | {:^28} | {:^28} | {:^28}".format("", "handover", "same team", "unknown"))
        lines.append("gaps      {:<8} | {:>10} {:>8} {:>8} | {:>10} {:>8} {:>8} | {:>10} {:>8} {:>8}".format(
            "match", *("frames", "log-loss", "accuracy") * 3))
        gaps = report["gap_classes"]
        for name, cell in list(sorted(gaps["per_match"].items())) + [("pooled", gaps["pooled"])]:
            parts = []
            for kind in ("handover", "same_team", "unknown"):
                part = cell[kind]
                parts += ["{:>10,}".format(part["frames"]), "{:>8}".format(number(part["log_loss"], ".4f")),
                          "{:>8}".format(number(part["accuracy"], ".4f"))]
            lines.append("gaps      {:<8} | {} {} {} | {} {} {} | {} {} {}".format(name, *parts))
    if "coarse" in report:                              # P34: the same scores against the coarse truth, after the rest
        lines.append("coarse    the coarse truth (short possessions between two of the other team merged), same frames")
        lines.append("coarse    {:<8} | {:>10} {:>8} {:>8} | {:>10} {:>8} {:>8}".format(
            "match", "frames", "log-loss", "accuracy", "frames", "log-loss", "accuracy"))
        for name, cell in list(sorted(report["coarse"]["per_match"].items())) + [("pooled", report["coarse"]["pooled"])]:
            alive, missing = cell["alive"], cell["missing"]
            lines.append("coarse    {:<8} | {:>10,} {:>8} {:>8} | {:>10,} {:>8} {:>8}".format(
                name, alive["frames"], number(alive["log_loss"], ".4f"), number(alive["accuracy"], ".4f"),
                missing["frames"], number(missing["log_loss"], ".4f"), number(missing["accuracy"], ".4f")))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train P07's shape-possession model on the benchmark matches (folds and full), write it as JSON, "
                    "verify it against a retraining, or report each fold on its own held-out match.")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="every this many graded frames is a row (default {})".format(DEFAULT_STRIDE))
    parser.add_argument("--features", type=int, choices=shape_features.VERSIONS, default=1,
                        help="the feature version to train, write or --verify (default 1: 26 columns, the committed "
                             "file; 2: 42 columns, P30; --report takes the version from the "
                             "file it reads")
    parser.add_argument("--out", help="where to write the artifact, or to read it from with --report (default "
                                      "pipeline/models/shape_possession_hgb.json, or shape_possession_v2_hgb.json "
                                      "with --features 2)")
    parser.add_argument("--verify", action="store_true",
                        help="retrain and compare with the committed file instead of writing it: exit 1 if "
                             "anything but the three source hashes differs")
    parser.add_argument("--report", action="store_true",
                        help="train nothing: read the artifact and print each fold's held-out log-loss and accuracy "
                             "(all alive frames, and those with the ball missing), touch agreement, ln 2 and whether "
                             "the stop rule fires, then the ball-missing frames by gap kind (handover, same team, "
                             "unknown) and the coarse truth's cells")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    given = {text.split("=")[0] for text in (sys.argv[1:] if argv is None else argv)}
    if args.seeds < 1 or args.stride < 1:
        parser.error("--seeds and --stride must be at least 1")
    if args.report and (args.verify or "--stride" in given):
        parser.error("--report takes no --verify or --stride: it reads the artifact and reports, and trains nothing")
    path = args.out or os.path.join(ROOT, *DEFAULT_MODEL_PATHS[args.features].split("/"))
    try:
        if args.report:
            try:
                with open(path, encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, ValueError) as error:
                raise BenchmarkError("cannot read the model {} ({}); train it with `python {}`".format(
                    path, error, "pipeline/benchmark/train_shape.py")) from error
            for line in render_report(report(artifact, args.profile, args.seeds, data_dir=args.data_dir)):
                print(line)
            return 0
        artifact = train(args.profile, args.seeds, args.stride, data_dir=args.data_dir, log=print,
                         version=args.features)
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


# -- small things -----------------------------------------------------------------------------------------------------

def _whole(name, value, minimum):
    """`value` as an int, or ValueError unless it is a whole number of at least `minimum` (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError("train_shape: {} must be a whole number of at least {}, got {!r}".format(name, minimum, value))
    return int(value)


def _known_profile(profile):
    refused = degrade.sensitivity_refusal(profile, "train_shape")
    if refused:
        raise BenchmarkError(refused)
    if profile not in run.PROFILES:
        raise BenchmarkError("train_shape: profile {!r} is not one of: {}".format(profile,
                                                                                  ", ".join(sorted(run.PROFILES))))


def _calibration_for(profile, calibration, data_dir):
    """The Calibration a degraded profile takes its masks from (measured or read from the cache if none is given); the
    clean profile degrades nothing, so a calibration given to it is refused rather than silently unused."""
    if not run.needs_calibration(profile) and calibration is not None:
        raise BenchmarkError("train_shape: profile {!r} takes no calibration, and one was given".format(profile))
    if run.needs_calibration(profile) and calibration is None:
        return calibration_module.load(data_dir=data_dir)
    return calibration


def _seed_list(profile, seeds):
    return [0] if profile == run.CLEAN else list(range(seeds))


def _own_id(match, match_id):
    if match.match_id != match_id:
        raise BenchmarkError("train_shape: the match given as {} says it is {}; its fold would be filed under one id "
                             "and looked up under the other, and scored by the model that saw it"
                             .format(match_id, match.match_id))


if __name__ == "__main__":
    sys.exit(main())
