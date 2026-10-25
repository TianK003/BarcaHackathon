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
   probabilities agree with `predict_proba` to rounding (the trainer's check holds them to 1e-12). The walk is
   pipeline/evidence/trees.py's, which checks the trees on load and walks them, and which also reads the file, checks
   its header and keeps the raw scores per Inputs for every learned source. NaN features are
   handled where the trees were fitted to handle them: the tracker's missing ball is the commonest feature value of the
   match (A01). For `logistic` it is the linear predictor on the imputed, standardised features.
4. Correct for the prior. The classifier's p is calibrated to the training set's alive share (`prior_alive`, about 0.55
   on DFL, whose ball is dead in 41 to 52% of graded frames). The decoder sets the class prior itself
   (pipeline/decode/), so the source gives it likelihood and not posterior, the recipe of hybrid HMMs (Bourlard and
   Morgan 1994): logit(alive) = logit(p) - logit(prior_alive). Without this, DFL's class balance would leak into the
   decoder's.
5. Clip to [min_prob, 1 - min_prob]: a tree can be far more certain than the matches it was fitted on deserve, and the
   decoder reads a probability as evidence of log(p / (1 - p)) nats a frame (ball_presence has the same reasoning).
6. NaN outside the graded blocks (no features there), `home` NaN everywhere (whether the ball is dead says nothing about
   who has it), and the pre-kick-off rule of every P05 source: a graded frame before its half's kick-off is
   `play_features.PRE_KICKOFF_ALIVE`, whatever the model says. It comes from the organisers' documented kick-offs and
   from no ground truth, only fires where the kick-off lies inside its block, and on BAR-ATH makes the 2,934 graded
   frames before the second half's kick-off D (T02).

Speed: with the features already computed, the first call on 140,000 frames and 300 trees of 15 leaves takes about 2 s
on the development machine (test_dead_ball_model_speed prints it and holds it under 3 s); the features take about 2.7 s
on top, cold. The walk is `trees.score`, which the learned sources share: pipeline/evidence/trees.py explains how it
reads a tree and why it is fast. The raw scores are then remembered per Inputs object, so the many configs of a
benchmark stage that share a model walk the trees once. A threshold of +inf (scikit-learn's split on missingness) is
stored as the largest float, which is exact for finite features; an infinite feature value is therefore refused.
"""

import numbers
import weakref

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import play_features, trees

# Params
KINDS = ("hgb", "logistic")
TRAINER = "pipeline/benchmark/train_dead_ball.py"      # path form: the benchmark-import scan reads dotted names

_MEMO = weakref.WeakKeyDictionary()                     # Inputs -> {(path, size, mtime, model key): raw scores}
_CACHE = {}                                            # absolute path -> ((size, mtime_ns), parsed artifact)


def compute(run, *, model, min_prob=0.001):
    """Evidence(alive = the model's likelihood of an alive ball, prior-corrected and clipped; home = NaN)."""
    min_prob = trees.check_min_prob(min_prob, "dead_ball_model")
    path = trees.model_path(model, "dead_ball_model", "pipeline/models/dead_ball_hgb.json")
    artifact = load(path)

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    score = _raw_scores(inputs, path, key, chosen)               # the model's own log-odds, NaN outside the blocks
    graded = ~np.isnan(score)
    alive = np.full(len(score), np.nan)
    alive[graded] = np.clip(trees.sigmoid(score[graded] - _logit(chosen["prior_alive"])), min_prob,
                            1.0 - min_prob)
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
    def build():
        _, X = play_features.features(inputs)
        graded = ~np.isnan(X[:, play_features.COLUMN["ball_missing"]])    # rows outside every graded block are NaN
        score = np.full(len(X), np.nan)
        if graded.any():
            trees.check_finite(X[graded], "dead_ball_model", inputs.game, "play_features")
            score[graded] = raw_score(model, X[graded])
        return score

    return trees.remember(_MEMO, inputs, path, key, build)


def predict(model, X):
    """P(alive) per row of the (n, 30) features X from one model of the file (a dict), before any prior correction.

    Numpy only. `model` is an entry of `folds` or `full`: the `hgb` ones hold `baseline` and `trees`, the `logistic`
    ones `coef` and its companions (train_dead_ball.py documents both).
    """
    return trees.sigmoid(raw_score(model, X))


def raw_score(model, X):
    """The model's log-odds of alive per row: the baseline plus its leaves, or the logistic's linear predictor."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(play_features.FEATURE_NAMES):
        raise ValueError("dead_ball_model: X must be (n, {}), one column per play feature, got shape {}"
                         .format(len(play_features.FEATURE_NAMES), X.shape))
    if "trees" in model:
        return trees.score(model, X)
    if "coef" in model:
        return _logistic_score(model, X)
    raise ValueError("dead_ball_model: a model holds `trees` (hgb) or `coef` (logistic); this has {}"
                     .format(sorted(model)))


def load(path):
    """The parsed, checked artifact at `path`, remembered while its size and modification time are the same."""
    return trees.load_artifact(
        path, "dead_ball_model", _check, _CACHE,
        unreadable="train it with `python {} --model hgb` (or logistic), or fix `model` in the config".format(TRAINER),
        not_json="retrain it with `python {} --model hgb`".format(TRAINER))


def _check(artifact, path):
    """ValueError naming the file, what differs and the trainer command, unless it is a model of these features."""
    trees.check_artifact(artifact, path, "dead_ball_model",
                         "retrain it with `python {} --model hgb` (or logistic)".format(TRAINER),
                         source="play_features", version=play_features.FEATURE_VERSION,
                         names=play_features.FEATURE_NAMES, kinds=KINDS,
                         check_model=lambda model, where: _check_model(artifact["kind"], model, where))


def _check_model(kind, model, where):
    if not isinstance(model, dict):
        raise ValueError("dead_ball_model: model {} is not an object".format(where))
    prior = model.get("prior_alive")
    if isinstance(prior, bool) or not isinstance(prior, numbers.Real) or not 0.0 < prior < 1.0:
        raise ValueError("dead_ball_model: prior_alive of {} must be a share strictly between 0 and 1, got {!r}"
                         .format(where, prior))
    if kind == "hgb":
        trees.check_trees(model, len(play_features.FEATURE_NAMES), where, "dead_ball_model")
    else:
        _check_logistic(model, where)


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


def _logit(p):
    return float(np.log(p / (1.0 - p)))
