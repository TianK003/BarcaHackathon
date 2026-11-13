"""P35's turnover hazard where the ball is seen: how likely a direct change of team is within half a second of a
frame, as a per-frame evidence value for the hmm decoder's switch cost.

    [[evidence]]
    kind = "turnover_hazard"
    name = "hazard"                                   # the decoder's hazard_source names it
    model = "pipeline/models/turnover_hgb.json"       # relative to the repository root
    min_prob = 0.001                                  # p is clipped to [min_prob, 1 - min_prob]

    [decoder]
    hazard_source = "hazard"
    hazard_scale = 1.0

This is a `team_model`'s clone with another question. The classifier reads `team_features` (version 2, 72 columns) and
answers: does a *direct* change of team (home to away or back, with no dead ball between) lie within 12 frames of this
one? It is trained by `pipeline/benchmark/train_turnover.py` on the seven open DFL/IDSSE matches degraded by `generic`,
written as JSON under pipeline/models/ and run here with numpy alone through the walk that the other learned sources
share (pipeline/evidence/trees.py).

What it returns is not a team vote. With p the model's probability and b the base rate that the model was trained on
(the share of its rows with a change near them, stored in the file as `base_rate`), `home` carries z = p / (p + b), so
that logit(z) = log(p / b): the log-ratio of the frame's hazard to the average one, which is what a per-frame switch
cost needs (the decoder charges `switch_team - hazard_scale * logit(z)`, bounded). z is a hazard in the guise of a
`home` probability only because that is the one channel an Evidence has; **a `turnover_hazard` source must never vote
as a team source.** The decoder takes the source out of its emissions when `hazard_source` names it, and a config that
has one the decoder does not name is refused (P35's A1; an unnamed source would pull every frame's team towards "home"
wherever z is above 0.5).

Swap-invariant. A change is a change whichever team is which, so the mirror rows of the trainer keep the label, and the
source's score is symmetrised the same way: logit(p) = (r(x) + r(swap(x))) / 2, which relabelling the slots' teams
leaves exactly unchanged. What the trees learned about which team is which (the baseline's lean) is averaged with its
own mirror image, not cancelled as in `team_model`.

Where it speaks (amendment 4 of P35's design). `home` only on the graded frames with the ball visible
(`kinematics.ball_track(...).visible`), the frames the model was trained on: NaN outside the graded blocks and wherever
the ball is missing, the gaps included, where the decoder's cost stays `switch_team`. `alive` is NaN everywhere.

The file's checks mirror `team_model`'s: `format`, `feature_version` (a plain int, `team_features.FEATURE_VERSION`),
`features`, `kind`, plus `target` "team_change_pm12" and `window_frames` 12 (a file trained for another question is
refused), and every model (`folds[match]` and `full`) holds runnable trees and a `base_rate` strictly between 0 and 1.
The model is `folds[inputs.game]` if the file has that key, else `full`. The logits are remembered per Inputs object,
file and model (`trees.remember`); `min_prob` and the base rate are applied after.
"""

import numbers
import weakref

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import kinematics, team_features, trees

# Params
NAME = "turnover_hazard"
KINDS = ("hgb",)
TARGET = "team_change_pm12"                            # the label: a direct change of team within +-WINDOW_FRAMES
WINDOW_FRAMES = 12
TRAINER = "pipeline/benchmark/train_turnover.py"       # path form: the benchmark-import scan reads dotted names
RETRAIN = "retrain it with `python {}`".format(TRAINER)

_MEMO = weakref.WeakKeyDictionary()                     # Inputs -> {(path, size, mtime, model key): logit(p)}
_CACHE = {}                                            # absolute path -> ((size, mtime_ns), parsed artifact)


def compute(run, *, model, min_prob=0.001):
    """Evidence(alive = NaN; home = z = p / (p + base_rate) where the ball is seen, NaN elsewhere), p clipped."""
    min_prob = trees.check_min_prob(min_prob, NAME)
    path = trees.model_path(model, NAME, "pipeline/models/turnover_hgb.json")
    artifact = load(path)

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    logit_p = _logit(inputs, path, key, chosen)               # NaN outside the blocks and where the ball is missing
    speaks = ~np.isnan(logit_p)
    base = float(chosen["base_rate"])
    p = np.clip(trees.sigmoid(logit_p[speaks]), min_prob, 1.0 - min_prob)
    home = np.full(len(logit_p), np.nan)
    home[speaks] = p / (p + base)
    home.flags.writeable = False
    alive = no_opinion(len(home))
    alive.flags.writeable = False
    return Evidence(alive=alive, home=home)


def symmetric_logit(model, X):
    """(r(x) + r(swap(x))) / 2 for each row of X, a (n, 72) float array: the model's log-odds of a change, unchanged by
    exchanging the teams. The trainer's report scores its folds with this function, so the report and the source agree."""
    X = np.asarray(X, dtype=np.float64)
    return (trees.score(model, X) + trees.score(model, team_features.swap(X))) / 2.0


def change_logit(inputs, model):
    """(n_frames,) the symmetrised log-odds of a change on every graded frame with the ball visible, NaN elsewhere.

    ValueError for an infinite feature (the walk is exact for finite values and NaN). Not remembered: `compute` does
    that; the trainer's report calls this one with a fold it holds in memory.
    """
    _, X = team_features.features(inputs)
    speaks = kinematics.ball_track(inputs).visible
    logit_p = np.full(len(X), np.nan)
    if speaks.any():
        rows = X[speaks]
        trees.check_finite(rows, NAME, inputs.game, "team_features")
        logit_p[speaks] = symmetric_logit(model, rows)
    return logit_p


def _logit(inputs, path, key, model):
    """`change_logit`, made once per Inputs and model file and shared by every clip and config that uses the model."""
    return trees.remember(_MEMO, inputs, path, key, lambda: change_logit(inputs, model))


def load(path):
    """The parsed, checked artifact at `path`, remembered while its size and modification time are the same."""
    return trees.load_artifact(path, NAME, _check, _CACHE,
                               unreadable="train it with `python {}`, or fix `model` in the config".format(TRAINER),
                               not_json=RETRAIN)


def _check(artifact, path):
    """ValueError naming the file, what differs and the trainer command, unless it is a model of this question on
    team_features' columns. The version, the window and (below) the target are plain values: JSON gives a bool or a
    float for `true` or `12.0`, and `12.0 == 12` would let a file through that no trainer of ours writes."""
    version = artifact.get("feature_version") if isinstance(artifact, dict) else team_features.FEATURE_VERSION
    if isinstance(version, bool) or not isinstance(version, int) or version != team_features.FEATURE_VERSION:
        raise ValueError("{}: the feature_version of {} must be {} (team_features' version, a whole number), got {!r}; "
                         "{}".format(NAME, path, team_features.FEATURE_VERSION, version, RETRAIN))
    if isinstance(artifact, dict):
        if artifact.get("target") != TARGET:
            raise ValueError("{}: the target of {} must be {!r}, got {!r}; {}"
                             .format(NAME, path, TARGET, artifact.get("target"), RETRAIN))
        window = artifact.get("window_frames")
        if isinstance(window, bool) or not isinstance(window, int) or window != WINDOW_FRAMES:
            raise ValueError("{}: window_frames of {} must be {} (a whole number), got {!r}; {}"
                             .format(NAME, path, WINDOW_FRAMES, window, RETRAIN))
    trees.check_artifact(artifact, path, NAME, RETRAIN, source="team_features",
                         version=team_features.FEATURE_VERSION, names=team_features.FEATURE_NAMES, kinds=KINDS,
                         check_model=_check_model)


def _check_model(model, where):
    if not isinstance(model, dict):
        raise ValueError("{}: model {} is not an object; {}".format(NAME, where, RETRAIN))
    base = model.get("base_rate")
    if isinstance(base, bool) or not isinstance(base, numbers.Real) or not 0.0 < base < 1.0:
        raise ValueError("{}: base_rate of {} must be a number above 0 and below 1, the share of the trainer's rows "
                         "with a change near them, got {!r}; {}".format(NAME, where, base, RETRAIN))
    try:
        trees.check_trees(model, len(team_features.FEATURE_NAMES), where, NAME)
    except ValueError as error:
        raise ValueError("{}; {}".format(error, RETRAIN)) from error
