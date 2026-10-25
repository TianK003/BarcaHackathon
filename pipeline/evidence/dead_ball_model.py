"""P05's learned opinion on a dead ball: a classifier on the play features, fitted on other matches, run with numpy.

    [[evidence]]
    kind = "dead_ball_model"
    model = "pipeline/models/dead_ball_hgb.json"   # relative to the repository root
    min_prob = 0.001                              # alive is clipped to [min_prob, 1 - min_prob]

`dead_ball_rules` states four fixed rules of the game; this source lets a classifier weigh the same thirty
`play_features` columns instead. The model is trained by `pipeline/benchmark/train_dead_ball.py` on the seven open
DFL/IDSSE matches, degraded to look like our tracking (docs/proposals.md P01, P05), and written as JSON under
pipeline/models/. Training needs scikit-learn and lives in pipeline/benchmark/; this module reads the file and runs the
trees with numpy alone, so the submission pipeline's pins stay numpy and pandas (requirements-pipeline.txt) and no
pickle is ever loaded. A pickle would run code from the file; JSON can only be numbers.

What it does, in order
-----------------------------------
1. Load. The file's `format`, `feature_version` and `features` must be play_features' own, or this raises, and says to
   retrain with `python pipeline/benchmark/train_dead_ball.py --model hgb` (or logistic): a model that reads column 7
   as a different feature than it was trained on would run and be wrong. The parsed file is remembered per path, size
   and modification time.
2. Choose the model. `folds[inputs.game]` if the file has that key, else `full`. A benchmark match is scored only by
   the model that never saw it, and BAR-ATH by `full`, so the benchmark's numbers are out of fold and the submission
   uses all seven matches.
3. Predict. For `hgb` a row goes left where x <= threshold, or where x is NaN and `missing_left`, and stops at a leaf;
   the raw score is the baseline plus the leaf values summed in tree order, which is scikit-learn's own order, so the
   probabilities agree with `predict_proba` to rounding (the trainer's check holds them to 1e-12). NaN features are
   handled where the trees were fitted to handle them: the tracker's missing ball is the commonest feature value of the
   match (A01). For `logistic` it is the linear predictor on the imputed, standardised features.
4. Correct for the prior. The classifier's p is calibrated to the training set's alive share (`prior_alive`, 0.9 or so
   on DFL, whose ball is dead less than ours). The decoder sets the class prior itself (pipeline/decode/), so the source
   gives it likelihood and not posterior, the recipe of hybrid HMMs (Bourlard and Morgan 1994): logit(alive) =
   logit(p) - logit(prior_alive). Without this, DFL's class balance would leak into the decoder's.
5. Clip to [min_prob, 1 - min_prob]: a tree can be far more certain than the matches it was fitted on deserve, and the
   decoder reads a probability as evidence of log(p / (1 - p)) nats a frame (ball_presence has the same reasoning).
6. NaN outside the graded blocks (no features there), `home` NaN everywhere (whether the ball is dead says nothing about
   who has it), and the pre-kick-off rule of every P05 source: a graded frame before its half's kick-off is
   `play_features.PRE_KICKOFF_ALIVE`, whatever the model says. It comes from the organisers' documented kick-offs and
   from no ground truth, only fires where the kick-off lies inside its block, and on BAR-ATH makes the 2,934 graded
   frames before the second half's kick-off D (T02).

Speed: with the features already computed, the first call on 140,000 frames and 300 trees of 15 leaves takes about 2 s
on the development machine (test_dead_ball_model_speed prints it and holds it under 3 s); the features take about 2.7 s
on top, cold. The walk (`_hgb_chunk`) reads a tree bottom up on all the rows of a 32,768-row chunk at once: each node
gives byte masks for "goes left" and "goes right", and a node's leaf id is the left child's or the right child's by
those masks, so after one pass from the last node to the root every row has its leaf. The raw scores are then
remembered per Inputs object, so the many configs of a benchmark stage that share a model walk the trees once.
A threshold of +inf (scikit-learn's split on missingness) is stored as the largest float, which is exact for finite
features; an infinite feature value is therefore refused.
"""

import json
import numbers
import os
import weakref

import numpy as np

import pipeline
from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import play_features

# Params
FORMAT = 1
KINDS = ("hgb", "logistic")
TRAINER = "pipeline/benchmark/train_dead_ball.py"      # path form: the benchmark-import scan reads dotted names
CHUNK = 32768                                          # rows per step of the tree walk
MAX_NODES = 255                                        # a leaf id is a node index held in one byte, and 255 is a mask
MAX_MIN_PROB = 0.5                                     # min_prob must stay below this, or the clip would invert

_MEMO = weakref.WeakKeyDictionary()                     # Inputs -> {(path, size, mtime, model key): raw scores}
_CACHE = {}                                            # absolute path -> ((size, mtime_ns), parsed artifact)


def compute(run, *, model, min_prob=0.001):
    """Evidence(alive = the model's likelihood of an alive ball, prior-corrected and clipped; home = NaN)."""
    min_prob = _min_prob(min_prob)
    path = _model_path(model)
    artifact = load(path)

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    score = _raw_scores(inputs, path, key, chosen)               # the model's own log-odds, NaN outside the blocks
    graded = ~np.isnan(score)
    alive = np.full(len(score), np.nan)
    alive[graded] = np.clip(_sigmoid(score[graded] - _logit(chosen["prior_alive"])), min_prob, 1.0 - min_prob)
    alive[play_features.pre_kickoff_mask(inputs) & graded] = play_features.PRE_KICKOFF_ALIVE
    alive.flags.writeable = False
    return Evidence(alive=alive, home=no_opinion(len(score)))


def _raw_scores(inputs, path, key, model):
    """The raw log-odds of `model` on every graded frame of `inputs` (NaN elsewhere), remembered per Inputs object.

    A benchmark stage runs dozens of configs that share one model and differ only in the decoder's weights, on the
    same Inputs; the walk of the trees is the expensive part and does not depend on them, so it is made once per
    (file, size, modification time, model of the file) and the prior correction, the clip and the kick-off rule, which
    are cheap, are applied after it. The memo is a WeakKeyDictionary on the Inputs object, as play_features' is, and
    dies with it; the arrays in it are read-only.
    """
    stat = os.stat(path)
    memo = _MEMO.setdefault(inputs, {})
    stamp = (path, stat.st_size, stat.st_mtime_ns, key)
    if stamp not in memo:
        _, X = play_features.features(inputs)
        graded = ~np.isnan(X[:, play_features.COLUMN["ball_missing"]])    # rows outside every graded block are NaN
        score = np.full(len(X), np.nan)
        if graded.any():
            if np.isinf(X[graded]).any():
                raise ValueError("dead_ball_model: a feature of {} is infinite; the trees are exact for finite values "
                                 "and NaN only, and play_features gives no other".format(inputs.game))
            score[graded] = raw_score(model, X[graded])
        score.flags.writeable = False
        memo.clear()                      # one model at a time per Inputs: a file that changed leaves nothing behind
        memo[stamp] = score
    return memo[stamp]


def predict(model, X):
    """P(alive) per row of the (n, 30) features X from one model of the file (a dict), before any prior correction.

    Numpy only. `model` is an entry of `folds` or `full`: the `hgb` ones hold `baseline` and `trees`, the `logistic`
    ones `coef` and its companions (train_dead_ball.py documents both).
    """
    return _sigmoid(raw_score(model, X))


def raw_score(model, X):
    """The model's log-odds of alive per row: the baseline plus its leaves, or the logistic's linear predictor."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(play_features.FEATURE_NAMES):
        raise ValueError("dead_ball_model: X must be (n, {}), one column per play feature, got shape {}"
                         .format(len(play_features.FEATURE_NAMES), X.shape))
    if "trees" in model:
        return _hgb_score(model, X)
    if "coef" in model:
        return _logistic_score(model, X)
    raise ValueError("dead_ball_model: a model holds `trees` (hgb) or `coef` (logistic); this has {}"
                     .format(sorted(model)))


def load(path):
    """The parsed, checked artifact at `path`, remembered while its size and modification time are the same."""
    try:
        stat = os.stat(path)
    except OSError as error:
        raise ValueError("dead_ball_model: cannot read the model file {} ({}); train it with `python {} --model "
                         "hgb` (or logistic), or fix `model` in the config".format(path, error.strerror, TRAINER)) \
            from error
    stamp = (stat.st_size, stat.st_mtime_ns)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("dead_ball_model: {} is not a JSON model file ({}); retrain it with `python {} --model "
                         "hgb`".format(path, error, TRAINER)) from error
    _check(artifact, path)
    _CACHE[path] = (stamp, artifact)
    return artifact


def _check(artifact, path):
    """ValueError naming the file, what differs and the trainer command, unless it is a model of these features."""
    retrain = "retrain it with `python {} --model hgb` (or logistic)".format(TRAINER)
    if not isinstance(artifact, dict):
        raise ValueError("dead_ball_model: {} holds no model object; {}".format(path, retrain))
    if artifact.get("format") != FORMAT:
        raise ValueError("dead_ball_model: {} has format {!r} and this reader's is {}; {}"
                         .format(path, artifact.get("format"), FORMAT, retrain))
    if artifact.get("feature_version") != play_features.FEATURE_VERSION:
        raise ValueError("dead_ball_model: {} was trained on feature_version {!r} and play_features is at {}; {}"
                         .format(path, artifact.get("feature_version"), play_features.FEATURE_VERSION, retrain))
    if artifact.get("features") != list(play_features.FEATURE_NAMES):
        raise ValueError("dead_ball_model: the features of {} are not play_features' ({} columns, in that order); {}"
                         .format(path, len(play_features.FEATURE_NAMES), retrain))
    if artifact.get("kind") not in KINDS:
        raise ValueError("dead_ball_model: the kind of {} is {!r}, not one of {}; {}"
                         .format(path, artifact.get("kind"), ", ".join(KINDS), retrain))
    if not isinstance(artifact.get("folds"), dict) or "full" not in artifact:
        raise ValueError("dead_ball_model: {} needs `folds` (a model per held-out match) and `full`; {}"
                         .format(path, retrain))
    for name, model in list(artifact["folds"].items()) + [("full", artifact["full"])]:
        _check_model(artifact["kind"], model, "{} of {}".format(name, path))


def _check_model(kind, model, where):
    if not isinstance(model, dict):
        raise ValueError("dead_ball_model: model {} is not an object".format(where))
    prior = model.get("prior_alive")
    if isinstance(prior, bool) or not isinstance(prior, numbers.Real) or not 0.0 < prior < 1.0:
        raise ValueError("dead_ball_model: prior_alive of {} must be a share strictly between 0 and 1, got {!r}"
                         .format(where, prior))
    if kind == "hgb":
        _check_trees(model, where)
    else:
        _check_logistic(model, where)


def _check_trees(model, where):
    if not isinstance(model.get("baseline"), numbers.Real) or not isinstance(model.get("trees"), list):
        raise ValueError("dead_ball_model: model {} needs a numeric `baseline` and a list of `trees`".format(where))
    features = len(play_features.FEATURE_NAMES)
    for number, tree in enumerate(model["trees"]):
        parts = [tree.get(key) for key in ("feature", "threshold", "missing_left", "left", "right", "leaf", "value")]
        if any(not isinstance(part, list) for part in parts) or len({len(part) for part in parts}) != 1 or not parts[0]:
            raise ValueError("dead_ball_model: tree {} of {} must hold seven node arrays of one length"
                             .format(number, where))
        feature, left, right, leaf = parts[0], parts[3], parts[4], parts[5]
        if len(feature) > MAX_NODES:
            raise ValueError("dead_ball_model: tree {} of {} has {} nodes and the walk reads at most {}; retrain with "
                             "fewer leaves".format(number, where, len(feature), MAX_NODES))
        for node in range(len(feature)):
            if leaf[node]:
                continue
            if not 0 <= feature[node] < features or not (node < left[node] < len(feature)
                                                         and node < right[node] < len(feature)):
                raise ValueError("dead_ball_model: tree {} of {} has a split at node {} that names no feature or "
                                 "no later child".format(number, where, node))


def _check_logistic(model, where):
    keys = ("columns", "mean", "scale", "coef", "impute")
    if any(not isinstance(model.get(key), list) for key in keys) or not isinstance(model.get("intercept"),
                                                                                    numbers.Real):
        raise ValueError("dead_ball_model: logistic model {} needs {} and `intercept`".format(where, ", ".join(keys)))
    if len({len(model[key]) for key in ("columns", "mean", "scale", "coef")}) != 1:
        raise ValueError("dead_ball_model: logistic model {} has `columns`, `mean`, `scale` and `coef` of different "
                         "lengths".format(where))
    if len(model["impute"]) != len(play_features.FEATURE_NAMES):
        raise ValueError("dead_ball_model: logistic model {} needs one `impute` value per feature".format(where))


def _hgb_score(model, X):
    """The raw score of every row of X, CHUNK rows at a time: the byte arrays of a step then stay in the cache."""
    return np.concatenate([_hgb_chunk(model, X[start:start + CHUNK]) for start in range(0, len(X), CHUNK)]
                          or [np.empty(0)])


def _hgb_chunk(model, X):
    """baseline + the leaf of every tree, added tree by tree, so a row's sum is made in scikit-learn's own order.

    Walking 300 trees over 140,000 rows is where the time goes, so the walk is made of whole-array operations on bytes
    and not of a loop over rows or of row subsets (an index-subset walk, tried first, cost about twice as much). A tree
    is read bottom up: a node's leaf id is its left child's where the row goes left and its right child's where not, and
    a child always has a larger index than its parent (checked on load), so one pass from the last node to the root
    gives every row its leaf. The choice between the two ids is made with the byte masks 255 (go left) and 0: the
    smaller of (left id, mask) is the id or 0, the smaller of (right id, 255 - mask) the other way round, and the larger
    of the two is the chosen one (`np.where` does the same and measured about 7% slower). A leaf's id is its node
    index, so a tree has at most 255 nodes (one of 15 leaves has 29; a bigger one is refused on load). The leaf
    values are then read from the tree's own table and added to the raw score of every row, in tree order, which is
    scikit-learn's own order, so the two agree to rounding.
    """
    raw = np.full(len(X), float(model["baseline"]))
    columns = np.asfortranarray(X)                       # a column is one contiguous run: x = columns[:, f]
    missing = {}                                         # feature -> its NaN mask, found only if a node needs it
    ids = {}                                             # leaf id -> that id on every row (np.minimum with a scalar is
                                                         # several times slower than with an array)
    with np.errstate(invalid="ignore"):                  # NaN <= threshold is False, which is what is wanted
        for tree in model["trees"]:
            feature, threshold, missing_left = tree["feature"], tree["threshold"], tree["missing_left"]
            left, right, leaf = tree["left"], tree["right"], tree["leaf"]
            if leaf[0]:
                raw += tree["value"][0]
                continue
            reach = [None] * len(feature)                # per node: the leaf id of every row that starts there
            for node in range(len(feature) - 1, -1, -1):
                if leaf[node]:
                    if node not in ids:
                        ids[node] = np.full(len(X), node, dtype=np.uint8)
                    reach[node] = ids[node]
                    continue
                goes_left = columns[:, feature[node]] <= threshold[node]
                if missing_left[node]:
                    if feature[node] not in missing:
                        missing[feature[node]] = np.isnan(columns[:, feature[node]])
                    goes_left |= missing[feature[node]]
                to_left = np.negative(goes_left.view(np.uint8))              # 255 where the row goes left, else 0
                to_right = np.invert(to_left)                                # and the other way round
                reach[node] = np.maximum(np.minimum(reach[left[node]], to_left),
                                         np.minimum(reach[right[node]], to_right))
            raw += np.array(tree["value"], dtype=np.float64)[reach[0]]
    return raw


def _logistic_score(model, X):
    features = len(play_features.FEATURE_NAMES)
    z = np.full(len(X), float(model["intercept"]))
    for position, column in enumerate(model["columns"]):
        if column < features:
            x = np.where(np.isnan(X[:, column]), model["impute"][column], X[:, column])
        else:
            x = np.isnan(X[:, column - features]).astype(np.float64)
        z = z + model["coef"][position] * ((x - model["mean"][position]) / model["scale"][position])
    return z


def _sigmoid(z):
    """1 / (1 + exp(-z)) without overflow: 0.5 at 0, and 0 or 1 (not NaN) far out on either side."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _min_prob(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 < value < MAX_MIN_PROB:
        raise ValueError("dead_ball_model: min_prob must be a number above 0 and below {}, got {!r}"
                         .format(MAX_MIN_PROB, value))
    return float(value)


def _model_path(model):
    """The absolute path of `model`, a string relative to the repository root (an absolute one is used as it is)."""
    if not isinstance(model, str) or not model:
        raise ValueError("dead_ball_model: model must be the path of a model file relative to the repository root, "
                         "such as \"pipeline/models/dead_ball_hgb.json\", got {!r}".format(model))
    return os.path.abspath(os.path.join(pipeline.ROOT, model))
