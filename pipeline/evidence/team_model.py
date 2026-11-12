"""P19's opinion on who has the ball where the ball is seen, restored for P32: a classifier on `team_features`
(version 2), symmetric.

    [[evidence]]
    kind = "team_model"
    model = "pipeline/models/team_model_hgb.json"   # relative to the repository root
    min_prob = 0.02                                 # home is clipped to [min_prob, 1 - min_prob]

`controlled_touch` names a team only inside a control episode, and `nearest_player` is wrong on 22% of the in-play
frames with the ball visible (T14, A02); between touches the decoder holds the last team. A boosted-tree classifier can
read more off the same frames: how close each team's nearest slots are, who is closing, who could get there first, how
the ball moves and which way it goes against each team's attack, when each team last and next controls it, and the
shape of the teams (`team_features` version 2, 72 columns). It is trained by `pipeline/benchmark/train_team.py` on the
seven open DFL/IDSSE matches, degraded by the benchmark's `generic` profile (docs/proposals.md P01, P19, P32), and
written as JSON under pipeline/models/. Training needs scikit-learn and lives in pipeline/benchmark/; this module reads
the file and runs the trees with numpy alone, through the walk that `dead_ball_model` and `shape_possession` share
(pipeline/evidence/trees.py), so the pipeline's pins stay numpy and pandas and no pickle is ever loaded. P19's version 1
(56 columns, the `tracking` profile) was dropped at R021's feasibility gate and is in git at commit 15e7e14; a file of
it is refused here.

It is `shape_possession`'s source without the `scope`, and the reasoning there applies unchanged:
- the file's `format`, `feature_version` (a plain int, `team_features.FEATURE_VERSION`: 2), `features` and `kind` must
  be `team_features`' own and every model (`folds[match]` and `full`) must hold runnable trees and a `prior_home` of
  exactly 0.5, or this raises and says to retrain;
- the model is `folds[inputs.game]` if the file has that key, else `full`: a benchmark match is scored only by the
  model that never saw it, and BAR-ATH by `full`;
- the score is symmetrised, logit(home) = (r(x) - r(swap(x))) / 2 with `team_features.swap`, so relabelling the slots'
  teams gives 1 - home to rounding, and what the trees learned about which team is which (the baseline, a lean towards
  the first team) cancels; no prior is corrected for (the trainer shows every row twice, mirrored);
- `home` is clipped to [min_prob, 1 - min_prob], so a tree cannot say more than 3.9 nats a frame at the default.

Where it speaks. `home` only on the graded frames with the ball visible (`kinematics.ball_track(...).visible`): those
are the frames the model was trained on, and the gaps are P07's. NaN elsewhere: outside the graded blocks and wherever
the ball is missing. `alive` is NaN everywhere (whether the ball is in play is `dead_ball_model`'s question). It runs on
a ball-visible frame whatever the players do: a NaN feature is something the trees were fitted to handle.

The logits are remembered per Inputs object, file size and modification time and model of the file (`trees.remember`),
so the configs of a benchmark stage that share a model walk the trees once, and `min_prob`, which is cheap, is applied
after it. An infinite feature value is refused (the walk is exact for finite values and NaN).
"""

import numbers
import weakref

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import kinematics, team_features, trees

# Params
NAME = "team_model"
KINDS = ("hgb",)
PRIOR_HOME = 0.5                                       # what a model of rows and their mirrors must say
TRAINER = "pipeline/benchmark/train_team.py"           # path form: the benchmark-import scan reads dotted names
RETRAIN = "retrain it with `python {}`".format(TRAINER)

_MEMO = weakref.WeakKeyDictionary()                     # Inputs -> {(path, size, mtime, model key): logit(home)}
_CACHE = {}                                            # absolute path -> ((size, mtime_ns), parsed artifact)


def compute(run, *, model, min_prob=0.02):
    """Evidence(alive = NaN; home = the model's symmetrised P(home has the ball), clipped, where the ball is seen)."""
    min_prob = trees.check_min_prob(min_prob, NAME)
    path = trees.model_path(model, NAME, "pipeline/models/team_model_hgb.json")
    artifact = load(path)

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    logit_home = _logit_home(inputs, path, key, chosen)          # NaN outside the blocks and where the ball is missing
    speaks = ~np.isnan(logit_home)
    home = np.full(len(logit_home), np.nan)
    home[speaks] = np.clip(trees.sigmoid(logit_home[speaks]), min_prob, 1.0 - min_prob)
    home.flags.writeable = False
    alive = no_opinion(len(home))
    alive.flags.writeable = False
    return Evidence(alive=alive, home=home)


def _logit_home(inputs, path, key, model):
    """(r(x) - r(swap(x))) / 2 on every graded frame with the ball visible, NaN elsewhere; per Inputs.

    The same walk serves every clip and config that shares the model. The array in the memo is read-only.
    """
    def build():
        _, X = team_features.features(inputs)
        speaks = kinematics.ball_track(inputs).visible
        logit_home = np.full(len(X), np.nan)
        if speaks.any():
            rows = X[speaks]
            trees.check_finite(rows, NAME, inputs.game, "team_features")
            logit_home[speaks] = (trees.score(model, rows) - trees.score(model, team_features.swap(rows))) / 2.0
        return logit_home

    return trees.remember(_MEMO, inputs, path, key, build)


def load(path):
    """The parsed, checked artifact at `path`, remembered while its size and modification time are the same."""
    return trees.load_artifact(path, NAME, _check, _CACHE,
                               unreadable="train it with `python {}`, or fix `model` in the config".format(TRAINER),
                               not_json=RETRAIN)


def _check(artifact, path):
    """ValueError naming the file, what differs and the trainer command, unless it is a model of these features.

    The version must be team_features' own as a plain int: JSON gives a bool or a float for `true` or `2.0`, and
    `2.0 == 2` would let a file through that no trainer of ours writes (shape_possession refuses them the same way). A
    file of version 1 (P19's 56 columns) is refused here too, by its version."""
    version = artifact.get("feature_version") if isinstance(artifact, dict) else team_features.FEATURE_VERSION
    if isinstance(version, bool) or not isinstance(version, int) or version != team_features.FEATURE_VERSION:
        raise ValueError("{}: the feature_version of {} must be {} (team_features' version, a whole number), got {!r}; "
                         "{}".format(NAME, path, team_features.FEATURE_VERSION, version, RETRAIN))
    trees.check_artifact(artifact, path, NAME, RETRAIN, source="team_features",
                         version=team_features.FEATURE_VERSION, names=team_features.FEATURE_NAMES, kinds=KINDS,
                         check_model=_check_model)


def _check_model(model, where):
    if not isinstance(model, dict):
        raise ValueError("{}: model {} is not an object; {}".format(NAME, where, RETRAIN))
    prior = model.get("prior_home")
    if isinstance(prior, bool) or not isinstance(prior, numbers.Real) or prior != PRIOR_HOME:
        raise ValueError("{}: prior_home of {} must be {}, the share of homes among the trainer's rows and their "
                         "mirrors, got {!r}; {}".format(NAME, where, PRIOR_HOME, prior, RETRAIN))
    try:
        trees.check_trees(model, len(team_features.FEATURE_NAMES), where, NAME)
    except ValueError as error:
        raise ValueError("{}; {}".format(error, RETRAIN)) from error
