"""P07's opinion on who has the ball where nobody can see it: a classifier on the shape of the two teams, symmetric.

    [[evidence]]
    kind = "shape_possession"
    model = "pipeline/models/shape_possession_hgb.json"   # relative to the repository root
    scope = "gaps"                                        # "gaps": only where the ball is missing; "all": every frame
    min_prob = 0.02                                       # home is clipped to [min_prob, 1 - min_prob]

The ball is missing from our tracking on 41% of the graded frames, in long gaps (a median 10.7 s, docs/results.md A01),
and `nearest_player` and `controlled_touch` are silent exactly there: both read the ball. A team in possession is more
advanced, wider and moving up the pitch together; the other sits deeper and shifts to follow the ball
(`shape_features`). This source lets a classifier read that off the `shape_features` columns (26, 42 in version 2, or 51
in version 5), none of which touches
the ball, and so it can be trained on matches whose ball has been hidden and be asked about the gaps of ours. The model
is trained by `pipeline/benchmark/train_shape.py` on the seven open DFL/IDSSE matches, degraded to look like our
tracking (docs/proposals.md P01, P07), and written as JSON under pipeline/models/. Training needs scikit-learn and
lives in pipeline/benchmark/; this module reads the file and runs the trees with numpy alone, through the walk that
`dead_ball_model` shares (pipeline/evidence/trees.py), so the submission pipeline's pins stay numpy and pandas and no
pickle is ever loaded.

What it does, in order
-----------------------------------
1. Load. The file's `format`, `feature_version` (a plain int among `shape_features.VERSIONS`: 1 is the 26 columns, 2
   the 42 of P30, which add each team's length and spread and the advance one-sided in time, 5 the 51 of P40, which add
   the opponents' approach per team and as a gap; 3 and 4 are retired and refused),
   `features` (that version's names, in order) and `kind` ("hgb") must be shape_features' own, every model
   (`folds[match]` and `full`) must hold trees the walk can run and a `prior_home` of exactly 0.5, or this raises and
   says to retrain with `python pipeline/benchmark/train_shape.py`: a model that reads column 7 as a different feature
   than it was trained on would run and be wrong. The parsed file is remembered per path, size and modification time.
2. Choose the model. `folds[inputs.game]` if the file has that key, else `full`. A benchmark match is scored only by
   the model that never saw it, and BAR-ATH by `full`, so the benchmark's numbers are out of fold and the submission
   uses all seven matches.
3. Predict, symmetrised. With r(x) the trees' raw score (the baseline plus the leaf of every tree, `trees.score`), the
   source says

       logit(home) = (r(x) - r(swap(x))) / 2

   where `swap(x)` is the features of the same frame with the two teams relabelled (`shape_features.swap`, which is
   exact). The two teams are interchangeable in what the features mean, so a model has no business knowing which one
   is Barcelona; the trainer shows it every row twice, as it is and mirrored with the label flipped, and still the
   classifier it fits is only nearly symmetric. This makes it exactly so: relabelling the teams negates the logit, so
   `home` becomes `1 - home` to rounding, and whatever the trees learned that does not depend on which team is which
   (the baseline, a lean towards the first team) cancels. The cost is two walks of the trees a frame, which are
   remembered, below.
4. No prior correction. The trainer's rows and their mirrors have exactly as many homes as aways, so `prior_home` is
   0.5, the classifier's p is already a likelihood and not a posterior that carries a class balance (the decoder sets
   the class prior itself, pipeline/decode/, the recipe of hybrid HMMs, Bourlard and Morgan 1994); the file is refused
   if it says otherwise. `dead_ball_model` has to subtract logit(prior_alive); here there is nothing to subtract.
5. Clip `home` to [min_prob, 1 - min_prob]: a tree can be far more certain than seven matches deserve, and the decoder
   reads a probability as evidence of log(p / (1 - p)) nats a frame (ball_presence has the same reasoning). The default
   0.02 caps a frame at 3.9 nats; the clip is symmetric, so antisymmetry survives it.
6. Speak only where it is asked to. `alive` is NaN everywhere: that the shape of the teams says who has the ball is no
   evidence on whether the ball is in play (`dead_ball_model` has that). `home` is NaN outside the graded blocks (no
   features there) and on a frame where both teams have no present slot (`h_n` and `a_n` are both 0: nobody to read the
   shape of), and with `scope = "gaps"`, the default, also where the ball is present: those frames have better
   evidence, and P07 (docs/proposals.md) is a source for the gaps. `scope = "all"` speaks on those frames too, to
   measure the source on its own. A team that is present while the other is gone is not silent:
   the missing team's columns are NaN, which the trees were fitted to handle.

Speed: with the features already computed, the first call on 140,000 frames and 300 trees of 15 leaves, which is two
walks of the trees, takes 2 to 3.5 s on the development machine (test_shape_possession_speed prints it and holds it
under 6 s); the features take 1 to 1.7 s on top, cold. The logits are remembered per Inputs object, file size and
modification time and model of the file, for every graded frame with someone present, whatever the scope: the many
configs of a benchmark stage that share a model walk the trees once, and `scope` and `min_prob`, which are cheap, are
applied after it. A threshold of +inf (scikit-learn's split on missingness) is stored as the largest float, which is
exact for finite features; an infinite feature value is therefore refused.
"""

import numbers
import weakref

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import shape_features, trees

# Params
NAME = "shape_possession"
KINDS = ("hgb",)
PRIOR_HOME = 0.5                                       # what a model of rows and their mirrors must say
SCOPES = ("gaps", "all")
TRAINER = "pipeline/benchmark/train_shape.py"          # path form: the benchmark-import scan reads dotted names
RETRAIN = "retrain it with `python {}`".format(TRAINER)

_MEMO = weakref.WeakKeyDictionary()                     # Inputs -> {(path, size, mtime, model key): logit(home)}
_CACHE = {}                                            # absolute path -> ((size, mtime_ns), parsed artifact)


def compute(run, *, model, scope="gaps", min_prob=0.02):
    """Evidence(alive = NaN; home = the model's symmetrised P(home has the ball), clipped, where `scope` says)."""
    scope = _scope(scope)
    min_prob = trees.check_min_prob(min_prob, NAME)
    path = trees.model_path(model, NAME, "pipeline/models/shape_possession_hgb.json")
    artifact = load(path)

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    # NaN outside the blocks and where nobody is present
    logit_home = _logit_home(inputs, path, key, chosen, artifact["feature_version"])
    speaks = ~np.isnan(logit_home)
    if scope == "gaps":
        speaks &= ~inputs.atd.ball_present
    home = np.full(len(logit_home), np.nan)
    home[speaks] = np.clip(trees.sigmoid(logit_home[speaks]), min_prob, 1.0 - min_prob)
    home.flags.writeable = False
    alive = no_opinion(len(home))
    alive.flags.writeable = False
    return Evidence(alive=alive, home=home)


def _logit_home(inputs, path, key, model, version):
    """(r(x) - r(swap(x))) / 2 on every graded frame with a present slot of either team, NaN elsewhere; per Inputs.

    `version` is the file's `feature_version`: the features are `shape_features.features(inputs, version)`.

    The same walk serves every scope, clip and config that shares the model: see the module docstring. The array in the
    memo is read-only.
    """
    def build():
        _, X = shape_features.features(inputs, version)
        column = shape_features.columns(version)
        speaks = (X[:, column["h_n"]] > 0) | (X[:, column["a_n"]] > 0)   # NaN is False
        logit_home = np.full(len(X), np.nan)
        if speaks.any():
            rows = X[speaks]
            trees.check_finite(rows, NAME, inputs.game, "shape_features")
            logit_home[speaks] = (trees.score(model, rows) - trees.score(model, shape_features.swap(rows))) / 2.0
        return logit_home

    return trees.remember(_MEMO, inputs, path, key, build)


def load(path):
    """The parsed, checked artifact at `path`, remembered while its size and modification time are the same."""
    return trees.load_artifact(path, NAME, _check, _CACHE,
                               unreadable="train it with `python {}`, or fix `model` in the config".format(TRAINER),
                               not_json=RETRAIN)


def _check(artifact, path):
    """ValueError naming the file, what differs and the trainer command, unless it is a model of one of the versions.

    The file names its feature version (`shape_features.VERSIONS`); its `features` must be that version's names, in
    order, and its trees as wide. The version must be a plain int: JSON gives a bool or a float for `true` or `2.0`,
    which `in VERSIONS` would let through (`True == 1`, `2.0 == 2`).
    """
    version = artifact.get("feature_version") if isinstance(artifact, dict) else shape_features.FEATURE_VERSION
    if isinstance(version, bool) or not isinstance(version, int) or version not in shape_features.VERSIONS:
        raise ValueError("{}: the feature_version of {} must be a whole number among {}, got {!r}; {}"
                         .format(NAME, path, shape_features.VERSIONS, version, RETRAIN))
    width = len(shape_features.feature_names(version))
    trees.check_artifact(artifact, path, NAME, RETRAIN, source="shape_features", version=version,
                         names=shape_features.feature_names(version), kinds=KINDS,
                         check_model=lambda model, where: _check_model(model, where, width))


def _check_model(model, where, width):
    if not isinstance(model, dict):
        raise ValueError("{}: model {} is not an object; {}".format(NAME, where, RETRAIN))
    prior = model.get("prior_home")
    if isinstance(prior, bool) or not isinstance(prior, numbers.Real) or prior != PRIOR_HOME:
        raise ValueError("{}: prior_home of {} must be {}, the share of homes among the trainer's rows and their "
                         "mirrors, got {!r}; {}".format(NAME, where, PRIOR_HOME, prior, RETRAIN))
    try:
        trees.check_trees(model, width, where, NAME)
    except ValueError as error:
        raise ValueError("{}; {}".format(error, RETRAIN)) from error


def _scope(value):
    if not isinstance(value, str) or value not in SCOPES:
        raise ValueError("{}: scope must be \"gaps\" (only where the ball is missing) or \"all\" (every graded "
                         "frame), got {!r}".format(NAME, value))
    return value
