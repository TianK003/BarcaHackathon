"""P22(b)'s gate on the DFL matches and its label-free checks on BAR-ATH: R027's tool (the design is
docs/superpowers/specs/2026-10-31-p22b-selftrain-design.md, sections 6.1 and 6.4; the declaration is R027 part 1 in
docs/results.md).

    python pipeline/benchmark/selftrain.py gate [--profile tracking] [--seeds 3] [--matches ID ...] [--data-dir D]
                                                [--no-write]
    python pipeline/benchmark/selftrain.py bar-ath --gate-report PATH [--data-dir D] [--no-write]

    report = gate(profile="tracking", seeds=3)                 # HEAVY: the seven DFL matches, one Inputs at a time
    rung1_go, rung3_go = gate_verdict(report["per_match"], report["pooled"])
    report = bar_ath(gate_report)                              # HEAVY: BAR-ATH, aggregates only
    head = head_fit(z, y, X, epsilon=0.07)                     # rung 3's last-layer head, on arrays

The belief. `shape_possession` (P07) gives a logit z of who has the ball from the teams' shape, and its trees never saw
the match; the touch rule reads the ball and the player beside it and never the shape, and on DFL it names the right
team 93% of the time. So its episode frames are noisy labels for z, and a per-half intercept fitted on them
(`recalibration.noise_aware_offset`, E2, with the other half's frames) should make P07's evidence in the ball gaps more
right. This tool asks that of the benchmark before any source exists, and then of our own match with no label at all.

`gate`. For each match, seed and graded block k it takes z from the match's own fold (a match whose fold is missing
from the shape model or from the dead-ball model is an error: `full` saw it, and the gate would be in sample), the
teacher's labels and rows (`recalibration.teacher_labels`, at `main.toml`'s rule and with its dead-ball mask), and
the truth beside the inputs (`truth.truth_codes`, never through them: test_selftrain_never_reads_truth_through_inputs).
Five logits are served, each as `clip(sigma(.), min_prob, 1 - min_prob)`:

    base     z itself, which is P07
    rung1    z + the E2 offset fitted on the other blocks' rows (`block_offsets`): the candidate, cross-fitted
    e1       the same with the class-balanced intercept, which noise pulls towards the minority class: reported only
    single   z + E2 fitted on every block's rows and applied to every block (no cross-fit; the design's m10)
    rung3    z + the head of `head_fit` fitted on the other blocks' rows: the retry's column

The gate's frames are block k's frames with truth H or A, the served ball missing and z finite, which are the frames
P07 exists for; the log-loss and the right calls (home at 0.5 or more) of each variant on them are summed per block,
per match over its seeds and blocks, and pooled by frames. The hold stratum (truth H or A, the ball seen, z finite,
outside every kept episode frame of the teacher, whether or not the dead-ball mask applies) is tallied for base and
rung 1, and so is the agreement with the kept episodes of `conventions.TOUCH` (R011's rule, a sanity check and not a
transfer check). The oracles are fitted with the truth's labels, which are clean, by `balanced_offset`: b*_gap on
block k's gate frames, and b*_touch on the other blocks' teacher rows that have an alive truth. From them:

    B        the mean over matches of the mean over seeds and blocks of |b*_gap|; E the same of |b - b*_gap|; E/B
    noise    |b - b*_touch|, the teacher's part of E; transfer |b*_touch - b*_gap|, from the teacher's frames to gaps
    band     the largest |b*_gap| over every (match, seed, block) cell: BAR-ATH's offsets must lie inside it

A cell whose oracle cannot be fitted (one class, no root) is left out of those means and counted. The teacher's own
figures are reported per match (precision against the truth on its rows with an alive truth, home share, pi_hat, the
share masked as dead, the rows per class) and so are the fallbacks (b = 0 and a reason).

The decision, for rung 1 and for rung 3, on the report's numbers at 9 decimals: GO iff the pooled gain over base is at
least GATE_MIN_GAIN (0.002 nats) and no match's log-loss is above its base by more than GATE_MAX_HARM (0.002); a miss by
any margin is a miss (`gate_verdict`). Exit codes of `gate`: 0 rung 1 GO; 1 both NO-GO; 4 rung 1 NO-GO and rung 3 GO
(the retry, R027b); 2 an error.

`bar-ath`. On BAR-ATH's Inputs, scored by the `full` model, the same cross-fitted E2: per block its offset (and the
fallback if it has one), pi_hat, the teacher's share, the share of the teacher's frames masked as dead, and the share of
its rows on frames where a team shows more than 11 slots, per team (A16's definition). The check in the gaps is the one
label-free look at them: on graded frames with the ball missing, P(alive) at least 0.5 and a Smart Tagging team vote
(`smart_tagging.compute` at `main.toml`'s c_team, spread "frame"), the adapted home must agree with the vote at least as
often as P07's minus TAG_MARGIN (0.005). The band is read from the gate report: an offset above it is a STOP, but only
if every check passes, and a failed check wins whatever the band says. Exit codes of `bar-ath`: 0 every check passes;
1 a check fails; 3 every check passes and an offset is outside the band; 2 an error. The held-out touch agreement is
reported only.

An error is 2 and never a crash that looks like a verdict: python's own status for an uncaught exception is 1, which
would read as "both NO-GO" or "a check failed", so `main` catches every exception, prints it and returns 2.

Constants (settings read from `main.toml`'s entries, ε = round(1 - confidence, 12) = 0.07, the floor, the clip of pi_hat
and the gate's limits) are the design's, fixed before any run and never chosen from a score. Determinism: every sum that
decides a fit is `math.fsum` (rung 3's Newton too, so that the head is the same to the bit whatever the order of the
rows and the number of threads), the offsets are `recalibration`'s, rounded to 9 decimals, and a report's floats are
written at 9 decimals. Nothing per frame is printed or written: the reports hold per-block, per-match and pooled
numbers, and no list longer than a match's seeds times its blocks.
"""

import argparse
import functools
import inspect
import json
import math
import numbers
import os
import sys
import traceback
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import conventions, degrade, dfl, run, wrong_team  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
from pipeline.benchmark.train_dead_ball import served_inputs  # noqa: E402
from pipeline.benchmark.train_shape import _calibration_for, _seed_list  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import (  # noqa: E402
    controlled_touch, dead_ball_model, recalibration, shape_features, shape_possession, smart_tagging, trees)
from pipeline.inputs import Inputs  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
DEFAULT_CONFIG = os.path.join(ROOT, "pipeline", "configs", "main.toml")
GATE_MIN_GAIN = 0.002           # nats: the pooled log-loss of a rung must be this far below the base's, at least
GATE_MAX_HARM = 0.002           # nats: and no match's may be above its base by more than this
TAG_MARGIN = 0.005              # BAR-ATH's Smart Tagging check: adapted agreement >= P07's minus this
HEAD_PENALTY = 0.01             # rung 3's L2 penalty on beta, a row: a coefficient of one SD must buy 0.005 nats a row
HEAD_TOLERANCE = 1e-10          # Newton stops when no coordinate of its step is this large
HEAD_MAX_STEPS = 50
HEAD_LINE_SEARCH = 1e-4         # a step with a coordinate above this is halved until it lowers the objective
HEAD_SMALLEST_STEP = 2.0 ** -30  # ... but not below this fraction of the Newton step
HOME_LEVEL = 0.5                # a served home at or above this names home
OVER_ELEVEN = wrong_team.TEAM_SIZE                  # a team with more present slots than this has one on the other side
DECIMALS = run.DECIMALS         # the reports' floats, and the numbers a decision is made on
FOLDER = ("runs", "selftrain")  # <data dir>/runs/selftrain/gate-<profile>.json and bar-ath.json
VARIANTS = ("base", "rung1", "e1", "single", "rung3")
OFFSET_VARIANTS = VARIANTS[1:]                      # the ones that move z by a per-block constant, or a head
HOLD_VARIANTS = ("base", "rung1")
RUNGS = ("rung1", "rung3")
FALLBACK_REASONS = ("few rows", "no root", "no convergence")
EXIT_ERROR = 2                  # both subcommands: python's own 1 for an uncaught exception would read as a verdict
KINDS = {"teacher": "controlled_touch", "alive": "dead_ball_model", "shape": "shape_possession",
         "tags": "smart_tagging"}

Settings = namedtuple("Settings", "teacher epsilon alive_model shape_model min_prob c_team")
Settings.__doc__ = """What the tool reads of main.toml. `teacher` is controlled_touch's radius, max_rel_speed,
    min_frames and duel_radius (a dict, checked by recalibration.check_teacher); `epsilon` is round(1 - its
    confidence, 12), the share of the teacher's labels that are wrong; `alive_model` and `shape_model` are the paths of
    the two models as the config writes them (relative to the repository root); `min_prob` is shape_possession's clip;
    `c_team` is smart_tagging's confidence."""
Head = namedtuple("Head", "offset beta mean scale steps")
Head.__doc__ = """Rung 3's fitted head: the intercept (E2 on z plus the head, rounded to 9 decimals), the coefficients
    on the standardised features (26 of them for the shape features), the standardisation (mean and scale per column,
    scale 0 for a column with no spread) and the number of Newton steps it took."""


class NoConvergence(ValueError):
    """The head's Newton iteration did not settle in HEAD_MAX_STEPS steps, or its system could not be solved."""


# -- the settings -----------------------------------------------------------------------------------------------------

def settings(config_path=DEFAULT_CONFIG):
    """Settings(teacher, epsilon, alive_model, shape_model, min_prob, c_team), read from a config's entries.

    `controlled_touch`: radius, max_rel_speed, min_frames and duel_radius (1.0 if the entry has none), and its
    confidence for epsilon = round(1 - confidence, 12); `dead_ball_model`'s model; `shape_possession`'s model and
    min_prob (its own default if the entry has none); `smart_tagging`'s c_team. The config needs exactly one entry of
    each of those four kinds, and BenchmarkError (a ValueError) names the kind or the key that is missing. The default
    is main.toml.
    """
    config = load_config(config_path)
    where = os.path.basename(config.path)
    touch = _entry(config, KINDS["teacher"], where)
    for key in ("radius", "max_rel_speed", "min_frames", "confidence"):
        _key(touch, key, KINDS["teacher"], where)
    teacher = recalibration.check_teacher({key: touch[key] for key in
                                           ("radius", "max_rel_speed", "min_frames", "duel_radius") if key in touch})
    confidence = touch["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real) or not 0.5 < confidence < 1.0:
        raise BenchmarkError("selftrain: {}'s {} has confidence {!r}, and epsilon = 1 - confidence has to be above 0 "
                             "and below 0.5 (the share of the teacher's labels that are wrong): use a confidence "
                             "strictly between 0.5 and 1.0".format(where, KINDS["teacher"], confidence))
    shape = _entry(config, KINDS["shape"], where)
    min_prob = shape.get("min_prob", inspect.signature(shape_possession.compute).parameters["min_prob"].default)
    return Settings(teacher=teacher, epsilon=round(1.0 - confidence, 12),
                    alive_model=_key(_entry(config, KINDS["alive"], where), "model", KINDS["alive"], where),
                    shape_model=_key(shape, "model", KINDS["shape"], where), min_prob=min_prob,
                    c_team=_key(_entry(config, KINDS["tags"], where), "c_team", KINDS["tags"], where))


read_settings = settings        # `gate` and `bar_ath` take a parameter of the same name, which hides the function


def _entry(config, kind, where):
    """The parameters of the config's one entry of `kind`; BenchmarkError naming the kind if it has none or several."""
    found = [component for component in config.evidence if component.kind == kind]
    if len(found) != 1:
        raise BenchmarkError("selftrain: {} has {} [[evidence]] entries of kind {!r}; settings() reads the teacher's "
                             "rule, the dead-ball mask, P07's model and Smart Tagging's c_team from the config's one "
                             "entry of each, so it needs exactly one {!r}".format(where, len(found), kind, kind))
    return found[0].params


def _key(params, key, kind, where):
    if key not in params:
        raise BenchmarkError("selftrain: the {!r} entry of {} has no `{}`, which settings() reads; add it".format(
            kind, where, key))
    return params[key]


def _checked(config):
    """`config` as a Settings that the tool can run on: a ValueError names what is wrong, before anything is loaded."""
    if not isinstance(config, Settings):
        raise BenchmarkError("selftrain: settings must be a selftrain.Settings (selftrain.settings() reads main.toml's)"
                             ", got {}".format(type(config).__name__))
    epsilon = config.epsilon
    if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real) or not 0.0 < epsilon < 0.5:
        raise ValueError("selftrain: epsilon must be a number above 0 and below 0.5, the share of the teacher's labels "
                         "that are wrong (1 - controlled_touch's confidence), got {!r}".format(epsilon))
    return config._replace(teacher=recalibration.check_teacher(config.teacher), epsilon=float(epsilon),
                           min_prob=trees.check_min_prob(config.min_prob, "selftrain"))


def _settings_record(config):
    record = config._asdict()
    record["teacher"] = dict(config.teacher)
    return record


def _constants():
    """The design's fixed numbers, written into every report so that a reader sees what it was made with."""
    return {"min_rows": recalibration.MIN_ROWS, "pi_clip": list(recalibration.PI_CLIP),
            "alive_level": recalibration.ALIVE_LEVEL, "teacher_confidence": recalibration.TEACHER_CONFIDENCE,
            "teacher_hold": recalibration.TEACHER_HOLD, "gate_min_gain": GATE_MIN_GAIN,
            "gate_max_harm": GATE_MAX_HARM, "head_penalty": HEAD_PENALTY, "tag_margin": TAG_MARGIN,
            "touch": dict(conventions.TOUCH)}


# -- rung 3's head ----------------------------------------------------------------------------------------------------

def standardisation(X):
    """(mean, scale): per column, the mean and the population SD of its finite entries, with scale 0 for a column that
    has no spread (constant, or no number in it). A column is constant when its largest and smallest entries are equal:
    the mean of a constant column need not be that constant to the last bit, and an SD made from it would not be 0."""
    X = _matrix(X)
    mean, scale = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        values = X[:, j][~np.isnan(X[:, j])]
        if len(values) == 0 or values.max() == values.min():
            mean[j] = values[0] if len(values) else 0.0
            continue
        mean[j] = math.fsum(values.tolist()) / len(values)
        scale[j] = math.sqrt(math.fsum(((values - mean[j]) ** 2).tolist()) / len(values))
    return mean, scale


def standardised(X, mean, scale):
    """X with each column's mean taken off and divided by its scale; NaN becomes 0 afterwards and a column whose
    scale is 0 becomes 0. A new (n, p) array."""
    X = _matrix(X)
    out = np.zeros(X.shape)
    for j in range(X.shape[1]):
        if scale[j] > 0.0:
            column = (X[:, j] - mean[j]) / scale[j]
            out[:, j] = np.where(np.isnan(column), 0.0, column)
    return out


def head_fit(z, y, X, *, epsilon, penalty=HEAD_PENALTY):
    """Rung 3 (the design's section 3): a team-aware head on the shape features, fitted on teacher labels y (0/1).

    1. X is standardised on the rows (`standardisation`: NaN becomes 0 afterwards, a column with no spread 0);
    2. beta by class-balanced, L2-penalised logistic regression with offset z and a free intercept: it minimises
       sum(w * log-loss) + (penalty / 2) |beta|^2, each class weighing a half, by Newton's method from 0 to a step
       under HEAD_TOLERANCE in every coordinate, in at most HEAD_MAX_STEPS steps (NoConvergence if not). The penalty
       is per row against the weighted mean: a coefficient of one SD must buy penalty / 2 nats a row. A step that is
       large (HEAD_LINE_SEARCH) is halved until it lowers the objective, because Newton's method from far away can
       overshoot on collinear features (adv_gap is h_adv_mean - a_adv_mean, and plain Newton failed on one random
       design in twelve); the steps near the optimum are never touched, so there it is Newton's own and the optimum
       is the same;
    3. the intercept is then E2 (`recalibration.noise_aware_offset`, with `epsilon`) on z + the head, so that it carries
       no pull towards the teacher's minority class.

    z must be finite and y 0 or 1 on every row, both classes present, X a 2-D array with a row for each (NaN is a
    missing feature, infinity is refused), else ValueError. `recalibration.NoRoot` if E2 has no root. Every sum is
    `math.fsum` and a row's linear predictor is added column by column, so the result is the same to the bit whatever
    the order of the rows and the number of threads. Returns a Head; `head_logit` applies it.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, numbers.Real) or not penalty > 0.0:
        raise ValueError("selftrain: head_fit: penalty must be a number above 0 (a column with no spread has no data "
                         "term, and only the penalty keeps the Newton system solvable), got {!r}".format(penalty))
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y)
    X = np.asarray(X, dtype=np.float64)
    if z.ndim != 1 or y.shape != z.shape or X.ndim != 2 or len(X) != len(z):
        raise ValueError("selftrain: head_fit: z and y must be 1-D arrays of one length and X a 2-D array with that "
                         "many rows, got shapes z {}, y {} and X {}".format(z.shape, y.shape, X.shape))
    if not np.isfinite(z).all():
        raise ValueError("selftrain: head_fit: z must be finite on every row; pass the rows where np.isfinite(z) holds")
    if np.isinf(X).any():
        raise ValueError("selftrain: head_fit: X holds an infinite value; a feature is finite or NaN (a missing one, "
                         "which is 0 once standardised)")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("selftrain: head_fit: y must be 0 or 1 on every row; select the rows first")
    y = y.astype(np.int64)
    homes = int(y.sum())
    if homes == 0 or homes == len(y):
        raise ValueError("selftrain: head_fit needs both classes among its rows, got {} home and {} away".format(
            homes, len(y) - homes))
    mean, scale = standardisation(X)
    Xs = standardised(X, mean, scale)
    theta, steps = _newton(z, y, Xs, float(penalty))
    beta = theta[1:]
    offset = recalibration.noise_aware_offset(z + _linear(Xs, beta), y, epsilon=epsilon).offset
    return Head(offset, beta, mean, scale, steps)


def head_score(head, X):
    """The head's part of the logit for the rows X: the standardised features times beta, without the intercept."""
    X = _matrix(X)
    if X.shape[1] != len(head.mean):
        raise ValueError("selftrain: head_logit: X has {} columns and the head was fitted on {}".format(
            X.shape[1], len(head.mean)))
    return _linear(standardised(X, head.mean, head.scale), head.beta)


def head_logit(head, z, X):
    """z + the head's score + its intercept, for the rows X: NaN where z is NaN, and nowhere else."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or len(np.asarray(X)) != len(z):
        raise ValueError("selftrain: head_logit: z must be 1-D with one value for each of the rows of X, got {} values "
                         "and {} rows".format(z.shape, np.shape(X)[0] if np.ndim(X) else 0))
    return z + head_score(head, X) + head.offset


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("selftrain: the features must be a 2-D array, (rows, features), got shape {}".format(X.shape))
    return X


def _linear(Xs, beta):
    """Xs times beta, one row at a time and in column order: elementwise sums that do not depend on the order of the
    rows (a matrix product does, through its blocking and its threads)."""
    total = np.zeros(len(Xs))
    for j in range(Xs.shape[1]):
        total += Xs[:, j] * beta[j]
    return total


def _newton(z, y, Xs, penalty):
    """(theta, steps): theta = (intercept, beta) minimising the weighted log-loss with offset z and the penalty on beta.

    Newton's method on exact sums. The gradient is sum(w (sigma - y) x) + penalty beta and the Hessian sum(w sigma
    (1 - sigma) x x') + penalty I over the beta block, with x = (1, standardised features) and w = 0.5 / (class size).
    The tails of sigma are taken from `trees.sigmoid` of t and of -t, each exact where the other would round to 1.
    A step with a coordinate above HEAD_LINE_SEARCH is halved while it does not lower the objective (exact sums, so that
    two evaluations are never confused by the order of the rows); the convergence test is on the Newton step itself.
    """
    n, p = Xs.shape
    homes = int(y.sum())
    weight = np.where(y == 1, 0.5 / homes, 0.5 / (n - homes))
    columns = [np.ones(n)] + [np.ascontiguousarray(Xs[:, j]) for j in range(p)]

    def objective(theta):
        t = (z + _linear(Xs, theta[1:])) + theta[0]
        loss = np.where(y == 1, np.logaddexp(0.0, -t), np.logaddexp(0.0, t))
        return math.fsum((weight * loss).tolist()) + 0.5 * penalty * math.fsum((theta[1:] ** 2).tolist())

    theta = np.zeros(p + 1)
    for step in range(1, HEAD_MAX_STEPS + 1):
        t = (z + _linear(Xs, theta[1:])) + theta[0]
        up, down = trees.sigmoid(t), trees.sigmoid(-t)
        residual = np.where(y == 1, -down, up) * weight                  # w * (sigma - y)
        curvature = up * down * weight
        gradient = np.array([math.fsum((column * residual).tolist()) for column in columns])
        gradient[1:] += penalty * theta[1:]
        hessian = np.empty((p + 1, p + 1))
        for j in range(p + 1):
            weighted = columns[j] * curvature
            for k in range(j, p + 1):
                hessian[j, k] = hessian[k, j] = math.fsum((weighted * columns[k]).tolist())
        hessian[np.arange(1, p + 1), np.arange(1, p + 1)] += penalty
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise NoConvergence("selftrain: head_fit: the Newton system is singular at step {} ({}); raise the "
                                "penalty".format(step, error)) from error
        size = 1.0
        if np.abs(delta).max() > HEAD_LINE_SEARCH:
            current = objective(theta)
            while objective(theta - size * delta) > current and size > HEAD_SMALLEST_STEP:
                size /= 2.0
        theta = theta - size * delta
        if np.abs(delta).max() < HEAD_TOLERANCE:
            return theta, step
    raise NoConvergence("selftrain: head_fit: Newton did not settle to a step under {} in {} steps; the labels may "
                        "separate the features, and a larger penalty (now {}) would hold the head back".format(
                            HEAD_TOLERANCE, HEAD_MAX_STEPS, penalty))


# -- the gate: what is measured on one match --------------------------------------------------------------------------

def gate(profile=run.DEFAULT_PROFILE, seeds=run.DEFAULT_SEEDS, match_ids=None, matches=None, data_dir=None,
         settings=None, calibration=None, log=None):
    """The gate's report on the benchmark matches (the module docstring has what it holds); HEAVY on the real seven.

    `profile` and `seeds` are the benchmark's (the clean profile has seed 0 only); `match_ids` the matches to run
    (default all seven); `matches` is {id: DflMatch} to use instead of loading them from `data_dir` (the checks pass
    synthetic ones); `settings` replaces `settings()` (main.toml's), and `calibration` is the Calibration a degraded
    profile takes its masks from (read from its cache if none is given); `log` receives a line per match. BenchmarkError
    for an unknown profile or match, a calibration given to the clean profile, and a match whose fold is missing from
    the shape model or the dead-ball model, all of it before the first match is loaded; ValueError for a bad `seeds` or
    settings. The report is JSON-ready but for the rounding of its floats (`run.plain` does that when it is written).
    """
    seeds = _whole("seeds", seeds, 1)
    _check_profile(profile)
    ids = run.resolve_match_ids(match_ids, matches)
    config = _checked(read_settings() if settings is None else settings)
    shape_path = trees.model_path(config.shape_model, "selftrain", "pipeline/models/shape_possession_hgb.json")
    alive_path = trees.model_path(config.alive_model, "selftrain", "pipeline/models/dead_ball_hgb.json")
    for who, artifact, trainer in (
            (KINDS["shape"], shape_possession.load(shape_path), "pipeline/benchmark/train_shape.py"),
            (KINDS["alive"], dead_ball_model.load(alive_path), "pipeline/benchmark/train_dead_ball.py")):
        absent = [match_id for match_id in sorted(ids) if match_id not in artifact["folds"]]
        if absent:
            raise BenchmarkError("selftrain: the {} model has no fold for {}; the source would use `full` for that "
                                 "match, which saw it, and the gate would be in sample; retrain with `python {}`"
                                 .format(who, ", ".join(absent), trainer))
    calibration = _calibration_for(profile, calibration, data_dir)
    seed_list = _seed_list(profile, seeds)
    log = log or (lambda text: None)

    cells = {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        _own_id(match, match_id)
        codes, _ = truths.truth_codes(match)
        cells[match_id] = []
        for seed in seed_list:
            cells[match_id] += _measure(match, codes, seed, profile, calibration, data_dir, config, shape_path)
        log("gate      {}   {} seed(s) x {} blocks measured".format(match_id, len(seed_list), len(match.graded)))
        del match, codes

    per_match = {match_id: _summary(cells[match_id], oracle=True) for match_id in sorted(cells)}
    pooled = _summary([cell for match_id in sorted(cells) for cell in cells[match_id]], oracle=False)
    for match_id, summary in per_match.items():
        if not summary["gate_frames"]:
            raise BenchmarkError("selftrain: {} has no alive frame with the ball missing and P07 speaking under "
                                 "profile {!r}, so the gate has nothing to be measured on there; use a profile that "
                                 "hides the ball (gaps, tracking or context)".format(match_id, profile))
    report = {"kind": "selftrain_gate", "profile": degrade.profile_record(profile), "seeds": list(seed_list),
              "matches": sorted(ids),
              "calibration": None if profile == run.CLEAN else {"sha256": calibration_module.digest(calibration),
                                                                "version": calibration_module.CALIBRATION_VERSION},
              "settings": _settings_record(config), "constants": _constants(), "git": pipeline_run.git_state(),
              "per_match": per_match, "pooled": pooled,
              "oracles": oracle_summary({match_id: summary["oracle"] for match_id, summary in per_match.items()})}
    rung1_go, rung3_go = gate_verdict(per_match, pooled)
    report["gate"] = {"rung1": gate_conditions(per_match, pooled, "rung1"),
                      "rung3": gate_conditions(per_match, pooled, "rung3"),
                      "exit_code": gate_exit_code(rung1_go, rung3_go),
                      "verdict": _gate_verdict_text(rung1_go, rung3_go)}
    return report


def _measure(match, codes, seed, profile, calibration, data_dir, config, shape_path):
    """The cells of one match under one seed, one per graded block: the raw tallies and figures that `_summary` adds up.

    One Inputs, served as the pipeline serves it (the degraded tracking and no flag), and gone when this returns: every
    array the cells are made of is taken out of it first, so that its memos die with it.
    """
    inputs = served_inputs(match, profile, seed, calibration, data_dir)
    run_context = RunContext(inputs=inputs, seed=0)
    z = shape_possession._logit_home(inputs, shape_path, match.match_id,
                                     shape_possession.load(shape_path)["folds"][match.match_id])
    y, rows = recalibration.teacher_labels(run_context, alive_model=config.alive_model, teacher=config.teacher)
    X = shape_features.features(inputs)[1]
    episode = controlled_touch.compute(run_context, **conventions.TOUCH).home
    blocks = [(start, stop) for _, start, stop in inputs.graded]
    halves = [half for half, _, _ in inputs.graded]
    return _cells(z, y, rows, X, episode, codes, ~inputs.atd.ball_present, blocks, halves, config, seed)


def _cells(z, y, rows, X, episode, codes, ball_missing, blocks, halves, config, seed):
    """One cell per block from the arrays of a match (every one of them has a value for each frame): the variants'
    logits, their tallies on the strata, the offsets and the oracles. `codes` is the truth, kept apart from the rest."""
    finite = np.isfinite(z)
    alive = (codes == pm.H) | (codes == pm.A)
    is_home = codes == pm.H
    known = ~np.isnan(y)
    usable = rows & finite
    gate_frames = alive & ball_missing & finite
    hold_frames = alive & ~ball_missing & finite & ~known
    kept = ~np.isnan(episode) & finite
    with np.errstate(invalid="ignore"):
        episode_home = episode >= HOME_LEVEL

    fit = functools.partial(recalibration.noise_aware_offset, epsilon=config.epsilon)
    found = {"rung1": recalibration.block_offsets(z, y, rows, blocks, fit=fit),
             "e1": recalibration.block_offsets(z, y, rows, blocks, fit=recalibration.balanced_offset)}
    # An empty block has every real block among "the other blocks": its fit is E2 on all of them, with the floor and
    # the no-root rule of the others, which is the single variant.
    everything = recalibration.block_offsets(z, y, rows, blocks + [(0, 0)], fit=fit)[-1]
    found["single"] = [everything] * len(blocks)
    heads = [_head(z, y, X, _pool(usable, blocks, k), found["rung1"][k], config.epsilon) for k in range(len(blocks))]
    logits = {"base": z}
    for name in ("rung1", "e1", "single"):
        logits[name] = recalibration.adapted_logit(z, blocks, found[name])
    logits["rung3"] = np.array(z)
    for (start, stop), (head, _) in zip(blocks, heads):
        if head is not None:
            logits["rung3"][start:stop] = head_logit(head, z[start:stop], X[start:stop])

    cells = []
    for k, ((start, stop), half) in enumerate(zip(blocks, halves)):
        own = slice(start, stop)
        pool = _pool(usable, blocks, k)
        in_pool = pool[alive[pool]]
        gate_k, hold_k, kept_k = gate_frames[own], hold_frames[own], kept[own]
        rows_k, known_k = usable[own], known[own]
        judged = rows_k & alive[own]
        e2, head = found["rung1"][k], heads[k][0]
        pooled_rows = e2.rows_home + e2.rows_away
        cells.append({
            "seed": seed, "half": half,
            "gate": {name: _tally(logits[name][own], gate_k, is_home[own], config.min_prob) for name in VARIANTS},
            "hold": {name: _tally(logits[name][own], hold_k, is_home[own], config.min_prob) for name in HOLD_VARIANTS},
            "touch": [int(kept_k.sum())] + [_agreed(logits[name][own], kept_k, episode_home[own], config.min_prob)
                                            for name in HOLD_VARIANTS],
            "b": {"rung1": e2.offset, "e1": found["e1"][k].offset, "single": found["single"][k].offset,
                  "rung3": head.offset if head is not None else 0.0},
            "fallback": {"rung1": e2.fallback, "e1": found["e1"][k].fallback, "single": found["single"][k].fallback,
                         "rung3": heads[k][1]},
            "steps": head.steps if head is not None else None,
            "b_gap": _oracle(z[own][gate_k], is_home[own][gate_k]),
            "b_touch": _oracle(z[in_pool], is_home[in_pool]),
            "rows_home": e2.rows_home, "rows_away": e2.rows_away,
            "share": e2.rows_home / pooled_rows if pooled_rows else None,
            "pi_hat": e2.fitted.pi_hat if e2.fitted is not None else None,
            "teacher": {"known": int(known_k.sum()), "masked": int((known_k & ~rows[own]).sum()),
                        "rows": int(rows_k.sum()), "rows_home": int((y[own][rows_k] == 1).sum()),
                        "judged": int(judged.sum()),
                        "right": int(((y[own][judged] == 1) == is_home[own][judged]).sum())}})
    return cells


def _pool(usable, blocks, skip):
    """The frames of every block but `skip` where `usable`, in order: the rows that block `skip`'s fit saw. It is
    `block_offsets`' own selection, which has to be found again here because its `fit` is given no frame numbers."""
    return np.concatenate([np.flatnonzero(usable[start:stop]) + start for k, (start, stop) in enumerate(blocks)
                           if k != skip] + [np.empty(0, dtype=np.intp)])


def _head(z, y, X, pool, e2, epsilon):
    """(Head or None, the reason it is None): rung 3 for one block, fitted on `pool`, under the floor E2 met."""
    if e2.fallback == "few rows":
        return None, "few rows"
    try:
        return head_fit(z[pool], y[pool].astype(np.int64), X[pool], epsilon=epsilon), None
    except recalibration.NoRoot:
        return None, "no root"
    except NoConvergence:
        return None, "no convergence"


def _oracle(z_rows, home_rows):
    """The class-balanced intercept on clean labels (E1 where there is no noise to lean on), or None if the frames hold
    one class only or have no root."""
    labels = home_rows.astype(np.int64)
    homes = int(labels.sum())
    if homes == 0 or homes == len(labels):
        return None
    try:
        return recalibration.balanced_offset(z_rows, labels).offset
    except recalibration.NoRoot:
        return None


def _served(logit, min_prob):
    """The home probability as the pipeline serves it: sigma of the logit, clipped to [min_prob, 1 - min_prob]."""
    return np.clip(trees.sigmoid(logit), min_prob, 1.0 - min_prob)


def _tally(logit, mask, is_home, min_prob):
    """[frames, summed log-loss, frames right] of the served home on the frames of `mask`, against is_home."""
    p = _served(logit[mask], min_prob)
    truly = is_home[mask]
    loss = -np.where(truly, np.log(p), np.log(1.0 - p))
    return [int(mask.sum()), math.fsum(loss.tolist()), int(((p >= HOME_LEVEL) == truly).sum())]


def _agreed(logit, mask, home_flags, min_prob):
    """The number of frames of `mask` where the served home (at 0.5 or more) says what `home_flags` says."""
    return int(((_served(logit[mask], min_prob) >= HOME_LEVEL) == home_flags[mask]).sum())


# -- the gate: what is reported ---------------------------------------------------------------------------------------

def _add(tallies):
    return [sum(t[0] for t in tallies), math.fsum(t[1] for t in tallies), sum(t[2] for t in tallies)]


def _score(tally):
    frames, loss, right = tally
    return {"frames": frames, "log_loss": loss / frames if frames else None,
            "accuracy": right / frames if frames else None}


def _ratio(top, bottom):
    return top / bottom if bottom else None


def _public(cell):
    """What a report keeps of a cell: its offsets, oracles and fallbacks and the teacher's shares: numbers, no frame."""
    return {key: cell[key] for key in ("seed", "half", "b", "fallback", "b_gap", "b_touch", "rows_home", "rows_away",
                                       "share", "pi_hat", "steps")}


def _summary(cells, oracle):
    """The report's block for a set of cells (one match's, or all of them): sums over seeds and blocks, as ratios.

    The log-loss and the accuracy are sums over the frames and never means of means; the teacher's figures are ratios of
    sums, and pi_hat the mean of the fits'. `oracle` adds the cells themselves and their oracle means (a match's).
    """
    teacher = {key: sum(cell["teacher"][key] for cell in cells)
               for key in ("known", "masked", "rows", "rows_home", "judged", "right")}
    pi_hats = [cell["pi_hat"] for cell in cells if cell["pi_hat"] is not None]
    touch = [sum(cell["touch"][slot] for cell in cells) for slot in range(3)]
    summary = {
        "gate_frames": sum(cell["gate"]["base"][0] for cell in cells),
        "variants": {name: _score(_add([cell["gate"][name] for cell in cells])) for name in VARIANTS},
        "hold": {name: _score(_add([cell["hold"][name] for cell in cells])) for name in HOLD_VARIANTS},
        "touch": {"frames": touch[0], "base": _ratio(touch[1], touch[0]), "rung1": _ratio(touch[2], touch[0])},
        "teacher": {"known_frames": teacher["known"], "masked_share": _ratio(teacher["masked"], teacher["known"]),
                    "rows": teacher["rows"], "rows_home": teacher["rows_home"],
                    "rows_away": teacher["rows"] - teacher["rows_home"],
                    "home_share": _ratio(teacher["rows_home"], teacher["rows"]),
                    "pi_hat": math.fsum(pi_hats) / len(pi_hats) if pi_hats else None, "fits": len(pi_hats),
                    "precision_frames": teacher["judged"], "precision": _ratio(teacher["right"], teacher["judged"]),
                    "rows_truth_dead": teacher["rows"] - teacher["judged"]},
        "fallbacks": {name: {reason: sum(1 for cell in cells if cell["fallback"][name] == reason)
                             for reason in FALLBACK_REASONS if name == "rung3" or reason != "no convergence"}
                      for name in OFFSET_VARIANTS}}
    if oracle:
        public = [_public(cell) for cell in cells]
        summary["oracle"], summary["cells"] = oracle_means(public), public
    else:
        del summary["gate_frames"]
    return summary


def oracle_means(cells):
    """One match's oracle figures from its cells (the public form: b, b_gap and b_touch of each): means over its seeds
    and blocks.

    {cells, cells_gap, cells_both, B, E, noise, transfer, band}. B is the mean |b*_gap| and E the mean |b - b*_gap|
    (b is rung 1's offset) over the cells whose gate oracle could be fitted; noise is the mean |b - b*_touch| and
    transfer the mean |b*_touch - b*_gap| over the cells where both oracles could (so that E <= noise + transfer holds
    on the same cells); band is the largest |b*_gap|. None for a mean over no cell.
    """
    with_gap = [cell for cell in cells if cell["b_gap"] is not None]
    with_both = [cell for cell in with_gap if cell["b_touch"] is not None]
    return {"cells": len(cells), "cells_gap": len(with_gap), "cells_both": len(with_both),
            "B": _mean([abs(cell["b_gap"]) for cell in with_gap]),
            "E": _mean([abs(cell["b"]["rung1"] - cell["b_gap"]) for cell in with_gap]),
            "noise": _mean([abs(cell["b"]["rung1"] - cell["b_touch"]) for cell in with_both]),
            "transfer": _mean([abs(cell["b_touch"] - cell["b_gap"]) for cell in with_both]),
            "band": max((abs(cell["b_gap"]) for cell in with_gap), default=None)}


def oracle_summary(by_match):
    """The oracle figures over the matches ({id: oracle_means}): the mean of the matches' means, so that each match
    weighs the same (R020's form), E/B from them (None if B is 0 or undefined), and the largest band. A match with no
    cell whose gate oracle could be fitted is left out, and `matches` counts those that are in."""
    used = [means for means in by_match.values() if means["B"] is not None]
    summary = {"matches": len(used), "B": _mean([means["B"] for means in used]),
               "E": _mean([means["E"] for means in used]),
               "noise": _mean([means["noise"] for means in used if means["noise"] is not None]),
               "transfer": _mean([means["transfer"] for means in used if means["transfer"] is not None]),
               "band": max((means["band"] for means in used), default=None)}
    summary["E_over_B"] = summary["E"] / summary["B"] if summary["B"] else None
    return summary


def _mean(values):
    return math.fsum(values) / len(values) if values else None


# -- the decision -----------------------------------------------------------------------------------------------------

def gate_conditions(per_match, pooled, variant):
    """The two conditions on one rung (`variant`: "rung1" or "rung3") and whether it passes, on the report's numbers.

    `per_match` is the report's {id: {"variants": {name: {"log_loss": ...}}}} and `pooled` its {"variants": ...}: the
    log-losses are taken at 9 decimals, as the report holds them, so that a reader of the file gets the same verdict.
    pooled_gain is base minus the rung's pooled log-loss and has to be at least GATE_MIN_GAIN; the harm of a match is
    the rung's log-loss minus its base's, and the largest has to be at most GATE_MAX_HARM. {pooled_gain, min_gain,
    gain_ok, worst_match, worst_harm, max_harm, harm_ok, per_match_harm, go}. ValueError for another variant and for a
    log-loss that is missing: a rung that cannot be compared is not a quiet NO-GO.
    """
    if variant not in RUNGS:
        raise ValueError("selftrain: the gate decides on {}, not on {!r}".format(" and ".join(RUNGS), variant))

    def loss(cell, name, where):
        value = cell["variants"][name]["log_loss"]
        if value is None:
            raise ValueError("selftrain: the gate needs a log-loss for {} and base {}, and has none (no gate frame)"
                             .format(variant, where))
        return round(float(value), DECIMALS)
    gain = round(loss(pooled, "base", "pooled") - loss(pooled, variant, "pooled"), DECIMALS)
    harm = {match_id: round(loss(per_match[match_id], variant, "on " + match_id)
                            - loss(per_match[match_id], "base", "on " + match_id), DECIMALS)
            for match_id in sorted(per_match)}
    worst = max(harm, key=lambda match_id: (harm[match_id], match_id), default=None)
    worst_harm = harm[worst] if worst is not None else 0.0
    gain_ok, harm_ok = gain >= GATE_MIN_GAIN, worst_harm <= GATE_MAX_HARM
    return {"pooled_gain": gain, "min_gain": GATE_MIN_GAIN, "gain_ok": gain_ok, "worst_match": worst,
            "worst_harm": worst_harm, "max_harm": GATE_MAX_HARM, "harm_ok": harm_ok, "per_match_harm": harm,
            "go": bool(gain_ok and harm_ok)}


def gate_verdict(per_match, pooled):
    """(rung1_go, rung3_go): the two rungs' `gate_conditions` on the report's per-match and pooled numbers."""
    return tuple(gate_conditions(per_match, pooled, variant)["go"] for variant in RUNGS)


def gate_exit_code(rung1_go, rung3_go):
    """0 if rung 1 goes; 4 if it does not and rung 3 does (the one retry, R027b); 1 if neither."""
    return 0 if rung1_go else 4 if rung3_go else 1


def _gate_verdict_text(rung1_go, rung3_go):
    if rung1_go:
        return "rung 1 GO"
    return "rung 1 NO-GO, rung 3 GO: the retry R027b" if rung3_go else "rung 1 NO-GO and rung 3 NO-GO: DROP"


# -- BAR-ATH ----------------------------------------------------------------------------------------------------------

def bar_ath(gate_report, data_dir=None, inputs=None, settings=None):
    """The label-free checks on BAR-ATH's Inputs, scored by the `full` model; HEAVY on the real match. Aggregates only.

    `gate_report` is the dict that `gate` wrote (its oracles' band is read from it); `inputs` replaces BAR-ATH's Inputs
    (the checks pass a synthetic one); `settings` replaces `settings()`. The module docstring has what is reported and
    what the exit code says: {kind, game, model, settings, band, gate_report, blocks, offset_gap, checks, touch,
    failed_checks, outside_band, exit_code, verdict}. BenchmarkError for a gate report without a usable band, and
    ValueError for settings that cannot run.
    """
    config = _checked(read_settings() if settings is None else settings)
    band = _band(gate_report)
    if inputs is None:
        inputs = Inputs(data_dir=data_dir)
    shape_path = trees.model_path(config.shape_model, "selftrain", "pipeline/models/shape_possession_hgb.json")
    artifact = shape_possession.load(shape_path)
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    run_context = RunContext(inputs=inputs, seed=0)
    z = shape_possession._logit_home(inputs, shape_path, key,
                                     artifact["full"] if key == "full" else artifact["folds"][key])
    y, rows = recalibration.teacher_labels(run_context, alive_model=config.alive_model, teacher=config.teacher)
    alive = dead_ball_model.compute(run_context, model=config.alive_model).alive
    vote = smart_tagging.compute(run_context, c_team=config.c_team, spread="frame").home
    episode = controlled_touch.compute(run_context, **conventions.TOUCH).home
    blocks = [(start, stop) for _, start, stop in inputs.graded]
    halves = [half for half, _, _ in inputs.graded]
    ball_missing = ~inputs.atd.ball_present
    over_home, over_away = (counts > OVER_ELEVEN for counts in wrong_team._counts(inputs.atd.players_present,
                                                                                  inputs.atd.slot_team))

    found = recalibration.block_offsets(z, y, rows, blocks,
                                        fit=functools.partial(recalibration.noise_aware_offset, epsilon=config.epsilon))
    adapted = recalibration.adapted_logit(z, blocks, found)
    usable, known = rows & np.isfinite(z), ~np.isnan(y)
    reported = []
    for (start, stop), half, offset in zip(blocks, halves, found):
        own = slice(start, stop)
        pooled_rows = offset.rows_home + offset.rows_away
        reported.append({
            "half": half, "b": offset.offset, "fallback": offset.fallback,
            "share": _ratio(offset.rows_home, pooled_rows),
            "pi_hat": offset.fitted.pi_hat if offset.fitted is not None else None,
            "rows_home": offset.rows_home, "rows_away": offset.rows_away, "rows": int(usable[own].sum()),
            "masked_share": _ratio(int((known[own] & ~rows[own]).sum()), int(known[own].sum())),
            "over_home": _ratio(int((usable[own] & over_home[own]).sum()), int(usable[own].sum())),
            "over_away": _ratio(int((usable[own] & over_away[own]).sum()), int(usable[own].sum()))})

    graded = np.zeros(len(z), dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True
    with np.errstate(invalid="ignore"):
        frames = graded & ball_missing & (alive >= recalibration.ALIVE_LEVEL) & ~np.isnan(vote) & np.isfinite(z)
    check = tag_check(vote, frames, z, adapted, config.min_prob)
    check["blocks"] = []
    for (start, stop), half in zip(blocks, halves):
        own = np.zeros(len(z), dtype=bool)
        own[start:stop] = True
        if (frames & own).any():
            part = tag_check(vote, frames & own, z, adapted, config.min_prob)
            check["blocks"].append({"half": half, "frames": part["frames"], "base": part["base"],
                                    "adapted": part["adapted"]})
    kept = ~np.isnan(episode) & np.isfinite(z)
    with np.errstate(invalid="ignore"):
        episode_home = episode >= HOME_LEVEL
    touch = [int(kept.sum())] + [_agreed(logit, kept, episode_home, config.min_prob) for logit in (z, adapted)]

    failed = [] if check["passed"] else ["smart_tagging_gaps"]
    outside = [block["half"] for block in reported if abs(block["b"]) > band]
    code, text = bar_ath_verdict(failed, outside)
    return {"kind": "selftrain_bar_ath", "game": inputs.game, "model": key, "settings": _settings_record(config),
            "band": band,
            "gate_report": {"profile": gate_report.get("profile"), "seeds": gate_report.get("seeds")},
            "blocks": reported,
            "offset_gap": abs(reported[0]["b"] - reported[1]["b"]) if len(reported) == 2 else None,
            "checks": {"smart_tagging_gaps": check},
            "touch": {"frames": touch[0], "base": _ratio(touch[1], touch[0]), "adapted": _ratio(touch[2], touch[0])},
            "failed_checks": failed, "outside_band": outside, "exit_code": code, "verdict": text}


def tag_check(vote, frames, base_logit, adapted_logit, min_prob, margin=None):
    """{frames, base, adapted, difference, margin, passed}: the adapted source's agreement with Smart Tagging's votes
    against P07's, on the frames of the bool mask `frames`.

    `vote` is smart_tagging's home array (at 0.5 or more it is a home vote; the frames must have one) and the two
    logits are z and the adapted z; each is served as the pipeline serves it (sigma, clipped to [min_prob, 1 -
    min_prob]) and agrees on a frame where it is at 0.5 or more exactly when the vote is. It passes iff adapted >=
    base - margin (TAG_MARGIN unless given), the difference taken at 9 decimals so that the edge itself passes.
    ValueError for arrays of different lengths and for no frame at all, which would make the check measure nothing.
    """
    margin = TAG_MARGIN if margin is None else margin
    vote, frames = np.asarray(vote), np.asarray(frames, dtype=bool)
    base_logit, adapted_logit = np.asarray(base_logit), np.asarray(adapted_logit)
    if not vote.shape == frames.shape == base_logit.shape == adapted_logit.shape or vote.ndim != 1:
        raise ValueError("selftrain: tag_check: the votes, the frames and the two logits must be 1-D with one value "
                         "per frame, got shapes {}, {}, {} and {}".format(vote.shape, frames.shape, base_logit.shape,
                                                                      adapted_logit.shape))
    n = int(frames.sum())
    if n == 0:
        raise ValueError("selftrain: tag_check: no frame has the ball missing, P(alive) of at least 0.5, a Smart "
                         "Tagging vote and a shape logit, so the check would measure nothing")
    named_home = vote[frames] >= HOME_LEVEL
    base = int(((_served(base_logit[frames], min_prob) >= HOME_LEVEL) == named_home).sum()) / n
    adapted = int(((_served(adapted_logit[frames], min_prob) >= HOME_LEVEL) == named_home).sum()) / n
    difference = adapted - base
    return {"frames": n, "base": base, "adapted": adapted, "difference": difference, "margin": margin,
            "passed": bool(round(difference, DECIMALS) >= -margin)}


def bar_ath_verdict(failed_checks, outside_band):
    """(exit code, sentence): 1 if a check failed, whatever the band says; else 3 if an offset is outside the band (a
    STOP for the advisor); else 0."""
    if failed_checks:
        return 1, "a check failed ({}): DROP, whatever the band says".format(", ".join(failed_checks))
    if outside_band:
        return 3, ("every check passes but the offset of half {} is outside the gate's band: STOP, an offset outside "
                   "the benchmark's range is an extrapolation".format(", ".join(str(half) for half in outside_band)))
    return 0, "every check passes and every offset is inside the band"


def bar_ath_exit_code(report):
    """The exit code of a BAR-ATH report: bar_ath_verdict on its failed checks and its blocks outside the band."""
    return bar_ath_verdict(report["failed_checks"], report["outside_band"])[0]


def _band(gate_report):
    """The gate report's band, the largest |b*_gap| of its cells, as a float; BenchmarkError if it has none."""
    oracles = gate_report.get("oracles") if isinstance(gate_report, dict) else None
    band = oracles.get("band") if isinstance(oracles, dict) else None
    if isinstance(band, bool) or not isinstance(band, numbers.Real) or not math.isfinite(band) or band < 0:
        raise BenchmarkError("selftrain: the gate report has no usable band (oracles.band, the largest |b*_gap| over "
                             "its cells; a number of 0 or more), got {!r}; give the report that `selftrain.py gate` "
                             "wrote".format(band))
    return float(band)


# -- the printout -----------------------------------------------------------------------------------------------------

def _number(value, spec="{:.4f}"):
    return "-" if value is None else spec.format(value)


def render_gate(report):
    """The gate's printed lines: the log-loss and the accuracy of every variant by match and pooled, the oracles, and
    the decision. Figures and match ids only."""
    profile, names = report["profile"], ("P07", "rung 1", "E1", "single", "rung 3")
    seeds = ",".join(str(s) for s in report["seeds"])
    lines = ["gate      profile {} (version {}), seeds {}: P07 and four intercepts or heads, each served as the "
             "pipeline serves it".format(profile["name"], profile["version"], seeds),
             "gate      on the alive frames with the served ball missing; a block's offset is fitted on the others"]
    rows = [(match_id, report["per_match"][match_id]) for match_id in sorted(report["per_match"])]
    rows.append(("pooled", report["pooled"]))
    for label, field, spec in (("log-loss", "log_loss", "{:.4f}"), ("accuracy", "accuracy", "{:.4f}")):
        lines.append("gate      {:<8} {:>10} | ".format(label, "frames") + " ".join("{:>7}".format(n) for n in names)
                     + " | rung 1 - P07")
        for name, summary in rows:
            cells = [summary["variants"][variant] for variant in VARIANTS]
            lines.append("gate      {:<8} {:>10,} | ".format(name, cells[0]["frames"])
                         + " ".join("{:>7}".format(_number(cell[field], spec)) for cell in cells)
                         + " | " + _number(None if cells[1][field] is None else cells[1][field] - cells[0][field],
                                           "{:+.4f}"))
    oracles = report["oracles"]
    lines += [
        "oracle    B (mean |b*_gap|) {}   E (mean |b - b*_gap|) {}   E/B {}   over {} matches".format(
            _number(oracles["B"]), _number(oracles["E"]), _number(oracles["E_over_B"], "{:.2f}"), oracles["matches"]),
        "oracle    E splits into the teacher's noise |b - b*_touch| {} and the transfer |b*_touch - b*_gap| {}".format(
            _number(oracles["noise"]), _number(oracles["transfer"])),
        "oracle    band, the largest |b*_gap| over the cells: {}".format(_number(oracles["band"]))]
    pooled = report["pooled"]
    lines += [
        "hold      ball seen, outside the teacher's episodes: log-loss P07 {}, rung 1 {} on {:,} frames".format(
            _number(pooled["hold"]["base"]["log_loss"]), _number(pooled["hold"]["rung1"]["log_loss"]),
            pooled["hold"]["base"]["frames"]),
        "touch     agreement with the kept episodes of conventions.TOUCH, held out: P07 {}, rung 1 {} (a sanity check)"
        .format(_number(pooled["touch"]["base"]), _number(pooled["touch"]["rung1"])),
        "teacher   precision {} on {:,} rows with an alive truth, home share {}, pi_hat {}, masked as dead {}".format(
            _number(pooled["teacher"]["precision"]), pooled["teacher"]["precision_frames"],
            _number(pooled["teacher"]["home_share"]), _number(pooled["teacher"]["pi_hat"]),
            _number(pooled["teacher"]["masked_share"])),
        "teacher   fallbacks (b = 0): " + "; ".join(
            "{} {}".format(name, ", ".join("{} {}".format(count, reason) for reason, count in counts.items() if count)
                           or "none") for name, counts in pooled["fallbacks"].items())]
    for rung, label in (("rung1", "rung 1"), ("rung3", "rung 3")):
        got = report["gate"][rung]
        lines.append("gate      {}: pooled gain {} (at least {}) {}; worst harm {} on {} (at most {}) {}: {}".format(
            label, _number(got["pooled_gain"], "{:+.4f}"), got["min_gain"], "ok" if got["gain_ok"] else "SHORT",
            _number(got["worst_harm"], "{:+.4f}"), got["worst_match"], got["max_harm"],
            "ok" if got["harm_ok"] else "OVER", "GO" if got["go"] else "NO-GO"))
    lines.append("gate      verdict: {} (exit {})".format(report["gate"]["verdict"], report["gate"]["exit_code"]))
    return lines


def render_bar_ath(report):
    """BAR-ATH's printed lines: a table per block, the band, the Smart Tagging check and the verdict. Figures only."""
    check = report["checks"]["smart_tagging_gaps"]
    lines = ["bar-ath   the gate's band {} (profile {}, seeds {}); every block is read with the offset fitted on the "
             "other".format(_number(report["band"]), (report["gate_report"]["profile"] or {}).get("name", "?"),
                            ",".join(str(s) for s in report["gate_report"]["seeds"] or [])),
             "bar-ath   half {:>8} {:<9} {:>7} {:>7} {:>7} {:>11} {:>11} {:>12}".format(
                 "b", "fallback", "pi_hat", "share", "masked", "over 11 H", "over 11 A", "rows")]
    for block in report["blocks"]:
        lines.append("bar-ath   {:>4} {:>8} {:<9} {:>7} {:>7} {:>7} {:>11} {:>11} {:>12}".format(
            block["half"], _number(block["b"], "{:+.4f}"), block["fallback"] or "-", _number(block["pi_hat"], "{:.3f}"),
            _number(block["share"], "{:.3f}"), _number(block["masked_share"], "{:.3f}"),
            _number(block["over_home"], "{:.3f}"), _number(block["over_away"], "{:.3f}"), "{:,}".format(block["rows"])))
    lines += [
        "bar-ath   |b1 - b2| {}".format(_number(report["offset_gap"])),
        "bar-ath   Smart Tagging in the gaps: {:,} frames, agreement P07 {}, adapted {} (difference {}, margin -{}): {}"
        .format(check["frames"], _number(check["base"]), _number(check["adapted"]),
                _number(check["difference"], "{:+.4f}"), check["margin"], "PASS" if check["passed"] else "FAIL"),
        "bar-ath   held-out touch agreement at conventions.TOUCH, {:,} frames: P07 {}, adapted {} (reported only)"
        .format(
            report["touch"]["frames"], _number(report["touch"]["base"]), _number(report["touch"]["adapted"])),
        "bar-ath   verdict: {} (exit {})".format(report["verdict"], report["exit_code"])]
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def report_path(data_dir, name, profile=None, match_ids=None, seeds=None):
    """<data dir>/runs/selftrain/<name>[-<profile>][.m-<ids>][.s<n>].json: where a report goes.

    The benchmark's own suffixes (`run.name_suffixes`) keep a quick look at some matches or one seed from replacing the
    report of all seven under the default seeds, which is the file `bar-ath --gate-report` is given; a report with no
    profile (bar-ath's) has neither."""
    stem = name if profile is None else "{}-{}".format(name, profile)
    suffixes = [] if profile is None else run.name_suffixes(profile, match_ids, seeds)
    return os.path.join(data_dir, *FOLDER, ".".join([stem] + suffixes) + ".json")


def build_parser():
    parser = argparse.ArgumentParser(
        description="P22(b)'s gate on the DFL matches (gate) and its label-free checks on BAR-ATH (bar-ath). Exit "
                    "codes: gate 0 rung 1 GO, 1 both NO-GO, 4 rung 1 NO-GO and rung 3 GO; bar-ath 0 every check "
                    "passes, 1 a check fails, 3 an offset is outside the band; 2 for an error in both.")
    commands = parser.add_subparsers(dest="command", required=True)
    gated = commands.add_parser("gate", help="R027's gate on the seven DFL matches (HEAVY)")
    gated.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    gated.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                       help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    gated.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run (default: all 7)")
    gated.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    gated.add_argument("--no-write", action="store_true", help="print only; write no report")
    barred = commands.add_parser("bar-ath", help="the label-free checks on BAR-ATH (HEAVY); aggregates only")
    barred.add_argument("--gate-report", required=True, metavar="PATH",
                        help="the report that `gate` wrote on the tracking profile: its band is read from it")
    barred.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    barred.add_argument("--no-write", action="store_true", help="print only; write no report")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gate" and args.seeds < 1:
        parser.error("--seeds must be at least 1")
    data_dir = args.data_dir or games.DATA_DIR
    try:
        if args.command == "gate":
            report = gate(args.profile, args.seeds, match_ids=args.matches, data_dir=args.data_dir, log=print)
            lines = render_gate(report)
            path = report_path(data_dir, "gate", args.profile, args.matches, args.seeds)
            code = report["gate"]["exit_code"]
        else:
            try:
                with open(args.gate_report, encoding="utf-8") as handle:
                    gate_report = json.load(handle)
            except (OSError, ValueError) as error:
                raise BenchmarkError("selftrain: cannot read the gate report {} ({}); run `python "
                                     "pipeline/benchmark/selftrain.py gate` first".format(
                                         args.gate_report, error)) from error
            report = bar_ath(gate_report, data_dir=args.data_dir)
            lines = render_bar_ath(report)
            path = report_path(data_dir, "bar-ath")
            code = report["exit_code"]
        for line in lines:
            print(line)
        if args.no_write:
            print("report    not written (--no-write)")
        else:
            run.write_json(path, run.plain(report))
            print("report    {}".format(run.shown(path)))
        return code
    except (BenchmarkError, ConfigError, OSError) as error:
        # The expected failures get a sentence, as in run.py: a profile or match that does not exist, a fold that is
        # missing, data not downloaded yet.
        print("FAILED    {}".format(error))
        return EXIT_ERROR
    except Exception:
        # Anything else is a bug and keeps its traceback, but it is exit 2 all the same: python's own status for an
        # uncaught exception is 1, which the controller reads as a verdict.
        traceback.print_exc()
        print("FAILED    an unexpected error (the traceback is above); exit {}, not a verdict".format(EXIT_ERROR))
        return EXIT_ERROR


# -- small things -----------------------------------------------------------------------------------------------------

def _whole(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise BenchmarkError("selftrain: {} must be a whole number of at least {}, got {!r}".format(
            name, minimum, value))
    return int(value)


def _check_profile(profile):
    if profile not in run.PROFILES:
        raise BenchmarkError("selftrain: profile {!r} is not one of: {}".format(
            profile, ", ".join(sorted(run.PROFILES))))


def _own_id(match, match_id):
    if match.match_id != match_id:
        raise BenchmarkError("selftrain: the match given as {} says it is {}; its fold would be filed under one id and "
                             "looked up under the other, and scored by the model that saw it".format(
                                 match_id, match.match_id))


if __name__ == "__main__":
    sys.exit(main())
