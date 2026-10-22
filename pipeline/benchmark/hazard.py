"""Where our tracker loses the ball, as a hazard: per-frame features, a small logistic fit, and a sampler of ball gaps.

    features(...)        what the tracking says about each frame of a graded half (line distance, ball speed, players)
    fit_logistic(...)    L2-penalised logistic regression by Newton/IRLS in numpy, deterministic to 1e-9
    predict, auc         the fitted probabilities, and how well they rank onsets above non-onsets
    sample_mask(...)     turns a per-frame onset probability into a ball-missing mask, gap by gap

Why this exists
---------------
P01 degrades the DFL matches until their tracking looks like ours (docs/proposals.md), and the ball is the part that
matters most: ours is missing on 40.7% of graded frames, in long gaps. Those gaps do not start at random. They cluster
at the lines and on slow balls (A01; the calibration measures them as A07). So the benchmark places its ball gaps where
our tracker loses the ball: a hazard fitted on our own match (A08) is carried to the benchmark matches, where it
gives each frame a probability that a gap starts there, and sample_mask draws the gaps from it. This module is the
maths of that and nothing else -- it reads no data and knows no match; the calibration fits it and the degrader uses it.

A model small enough for 147 onsets: about six features and a logistic link, the standard discrete-time event-history
model. numpy and nothing else (no scikit-learn, no scipy), so the pins stay as they are.

Conventions
-----------
* One block is one graded half. Row i of every array is frame block_start + i; positions are in metres, NaN where the
  tracker has nothing. Nothing here looks across the halftime break.
* prob[s] is the probability that a gap starts AT frame s, i.e. that the ball is missing at s given it was visible
  before. A hazard fitted on "the next frame starts a gap" (features at the last visible frame t, outcome at t + 1)
  must be shifted one frame before it goes to sample_mask.
* Gap lengths are frame counts (ints), never seconds: the caller converts.
* Fits are deterministic and their stored floats rounded to 1e-9, because BLAS summation order differs between
  machines (CLAUDE.md, determinism). predict adds its terms column by column rather than with a matrix product, so
  the probabilities do not depend on the BLAS either.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Params
FEATURES = ("line_distance_m", "beyond_line", "ball_speed_mps", "nearest_player_m", "players_within_3m",
            "minute_of_half")
SPEED_STEPS = 5          # the ball's speed is the mean of this many of its latest step speeds
NEAR_PLAYER_M = 3.0      # players_within_3m counts the players at most this far from the ball
THIRD_EDGES = (15, 30)   # minutes of the half where the second and the last third of a duration table begin
DECIMALS = 9             # a fitted model keeps this many decimals
NEWTON_TOLERANCE = 1e-10  # Newton stops once no parameter moves by more than this
MAX_CONDITION = 1e12      # a Hessian worse conditioned than this has no one solution (l2 = 0 and a repeated column)


# -- features ---------------------------------------------------------------------------------------------------------

def features(ball_xy_m, players_xy_m, pitch, fps, block_start, kickoff):
    """(n, 6) table of FEATURES for one block; a row where the ball is missing is NaN in every column.

    `ball_xy_m` (n, 2) and `players_xy_m` (n, P, 2) are in metres, NaN where missing (a position with one coordinate
    NaN counts as missing). `pitch` is (length, width) in metres, `fps` the frame rate, `block_start` the video frame
    of row 0 and `kickoff` the video frame of the half's kick-off.

    line_distance_m   min(x, L - x, y, W - y): the distance to the nearest line, negative beyond it.
    beyond_line       1.0 where line_distance_m < 0 (strictly: a ball on a line is still in play), else 0.0.
    ball_speed_mps    the mean of the last SPEED_STEPS defined step speeds at or before the row, within the block. A
                      step speed is |p_t - p_(t-1)| * fps where the ball is visible at t - 1 and at t, so after a gap
                      the speed is the one from before it, until new steps come in. NaN until one step is defined.
                      Always from positions: the ATD's own ball-speed channel is unusable (T31).
    nearest_player_m  the distance to the nearest present player; NaN if nobody is present.
    players_within_3m the number of present players at most NEAR_PLAYER_M from the ball (exactly 3 m counts).
    minute_of_half    (block_start + i - kickoff) / fps / 60, negative before the kick-off: our second half is graded
                      from 2,934 frames before it.
    """
    ball = np.asarray(ball_xy_m, dtype=np.float64)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("hazard.features: ball_xy_m must be (n, 2) in metres, got shape {}; pass one graded half"
                         .format(ball.shape))
    n = ball.shape[0]
    players = np.asarray(players_xy_m, dtype=np.float64)
    if players.ndim != 3 or players.shape[2] != 2:
        raise ValueError("hazard.features: players_xy_m must be (n, P, 2) in metres, got shape {}"
                         .format(players.shape))
    if players.shape[0] != n:
        raise ValueError("hazard.features: players_xy_m has {} rows but ball_xy_m has {}; both are one block"
                         .format(players.shape[0], n))
    try:
        length, width = (float(side) for side in pitch)
    except (TypeError, ValueError):
        raise ValueError("hazard.features: pitch must be (length, width) in metres, got {!r}".format(pitch)) from None
    if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
        raise ValueError("hazard.features: pitch must be positive and finite, got {!r}".format(pitch))
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise ValueError("hazard.features: fps must be a number, got {!r}".format(fps)) from None
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("hazard.features: fps must be positive and finite, got {!r}".format(fps))
    frames = []
    for name, value in (("block_start", block_start), ("kickoff", kickoff)):
        try:
            frames.append(operator.index(value))
        except TypeError:
            raise ValueError("hazard.features: {} must be an integer frame number, got {!r}"
                             .format(name, value)) from None
    block_start, kickoff = frames

    x, y = ball[:, 0], ball[:, 1]
    visible = ~np.isnan(ball).any(axis=1)
    line = np.minimum.reduce([x, length - x, y, width - y])
    beyond = (line < 0).astype(np.float64)

    # Step speeds are NaN unless the ball is visible at both ends of the step.
    step = np.full(n, np.nan)
    if n > 1:
        step[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
    defined = ~np.isnan(step)
    speed = np.full(n, np.nan)
    if defined.any():
        # Window j of the padded list ends at the j-th defined step and holds it and up to SPEED_STEPS - 1 before it.
        padded = np.concatenate([np.full(SPEED_STEPS - 1, np.nan), step[defined]])
        recent = np.nanmean(sliding_window_view(padded, SPEED_STEPS), axis=1)
        seen = np.cumsum(defined)   # how many steps are defined at or before each row
        speed = np.where(seen > 0, recent[np.maximum(seen - 1, 0)], np.nan)

    if players.shape[1]:
        gap = np.hypot(players[:, :, 0] - x[:, None], players[:, :, 1] - y[:, None])   # NaN: player or ball missing
        present = ~np.isnan(gap)
        nearest = np.where(present, gap, np.inf).min(axis=1)
        nearest[~present.any(axis=1)] = np.nan
        within = (gap <= NEAR_PLAYER_M).sum(axis=1).astype(np.float64)
    else:
        nearest = np.full(n, np.nan)
        within = np.zeros(n)

    minute = (block_start + np.arange(n) - kickoff) / fps / 60.0
    table = np.column_stack([line, beyond, speed, nearest, within, minute])
    table[~visible] = np.nan
    return table


def third_of_minute(minute):
    """0 for minute < 15, 1 for minute < 30, 2 otherwise (an int array): the third of the half that sample_mask's
    duration table is indexed by. A minute before the kick-off is in the first third; NaN is refused."""
    minute = np.asarray(minute, dtype=np.float64)
    if np.isnan(minute).any():
        raise ValueError("hazard.third_of_minute: minute has {} NaN; compute it from the frame numbers, which always "
                         "exist, not from a feature column that is NaN where the ball is missing"
                         .format(int(np.isnan(minute).sum())))
    return np.digitize(minute, THIRD_EDGES)


# -- the logistic fit -------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LogisticModel:
    """P(y = 1 | x) = sigmoid(intercept + sum_j coef[j] * (x[j] - mean[j]) / scale[j]).

    The coefficients are on the standardised features (mean and scale are what the fit standardised by); multiply by
    1 / scale to read them in each feature's own unit. fit_logistic stores them rounded to 1e-9 and read-only.
    """
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray


def fit_logistic(X, y, l2=1.0, iterations=50):
    """LogisticModel fitted to the (n, k) features X and the 0/1 outcomes y by Newton/IRLS.

    It minimises  -log-likelihood + l2 / 2 * |coef|^2  on features standardised by their mean and population SD
    (ddof 0; a constant column keeps its value as its mean and gets a scale of 1, and so a coefficient of 0). The
    intercept is not penalised. Newton runs from the base-rate intercept and zero coefficients for at most
    `iterations` steps, and stops early once no parameter moves by more than 1e-10. The result is rounded to 1e-9.

    X must have no NaN: the caller chooses the at-risk rows. ValueError, naming what to change, for a bad shape, a NaN
    or an outcome that is not 0 or 1, a single class, an l2 below 0, and (with l2 = 0) a Hessian that has no inverse.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("hazard.fit_logistic: X must be (n, k) with at least one row and one column, got shape {}"
                         .format(X.shape))
    n, k = X.shape
    if y.shape != (n,):
        raise ValueError("hazard.fit_logistic: X has {} rows but y has {} values; give one 0/1 outcome per row"
                         .format(n, y.size))
    if np.isnan(X).any():
        raise ValueError("hazard.fit_logistic: X has {} NaN; fit on the at-risk rows only, the ones where every "
                         "feature is defined".format(int(np.isnan(X).sum())))
    if not np.isfinite(X).all():
        raise ValueError("hazard.fit_logistic: X has infinite values; check how the features were computed")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("hazard.fit_logistic: y must be 0 or 1 in every row")
    if y.min() == y.max():
        raise ValueError("hazard.fit_logistic: y holds only {:g}; a fit needs both classes (some onsets and some "
                         "frames without one)".format(y[0]))
    if not l2 >= 0:
        raise ValueError("hazard.fit_logistic: l2 must be 0 or more, got {!r}".format(l2))
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("hazard.fit_logistic: iterations must be an integer of at least 1, got {!r}"
                         .format(iterations))

    constant = X.max(axis=0) == X.min(axis=0)
    mean = np.where(constant, X[0], X.mean(axis=0))
    scale = np.where(constant, 1.0, X.std(axis=0))
    design = np.column_stack([np.ones(n), (X - mean) / scale])
    penalty = np.full(k + 1, float(l2))
    penalty[0] = 0.0

    base_rate = y.mean()
    beta = np.zeros(k + 1)
    beta[0] = math.log(base_rate / (1.0 - base_rate))
    for _ in range(int(iterations)):
        p = _sigmoid(design @ beta)
        score = design.T @ (p - y) + penalty * beta
        curvature = (design.T * (p * (1.0 - p))) @ design + np.diag(penalty)
        if not np.linalg.cond(curvature) < MAX_CONDITION:
            raise ValueError("hazard.fit_logistic: the features are collinear (a column repeats another, or is a "
                             "combination of the others), so the fit has no one answer; use l2 > 0 or drop the column")
        step = np.linalg.solve(curvature, score)
        beta = beta - step
        if np.abs(step).max() < NEWTON_TOLERANCE:
            break

    def stored(values):
        values = np.round(values, DECIMALS) + 0.0   # + 0.0 turns a negative zero into 0.0
        values.setflags(write=False)
        return values

    return LogisticModel(coef=stored(beta[1:]), intercept=float(np.round(beta[0], DECIMALS) + 0.0), mean=stored(mean),
                         scale=stored(scale))


def predict(model, X):
    """The model's probability for each row of the (n, k) features X; a row with a NaN gives NaN."""
    X = np.asarray(X, dtype=np.float64)
    coef = np.asarray(model.coef, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("hazard.predict: X must be (n, k), got shape {}".format(X.shape))
    if X.shape[1] != coef.shape[0]:
        raise ValueError("hazard.predict: the model has {} coefficients but X has {} columns"
                         .format(coef.shape[0], X.shape[1]))
    mean = np.asarray(model.mean, dtype=np.float64)
    scale = np.asarray(model.scale, dtype=np.float64)
    z = np.full(X.shape[0], float(model.intercept))
    for j in range(coef.shape[0]):
        z = z + coef[j] * ((X[:, j] - mean[j]) / scale[j])
    return _sigmoid(z)


def _sigmoid(z):
    """1 / (1 + exp(-z)) without overflow: exactly 0.5 at 0, and 0 or 1 (not NaN) far out on either side."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def auc(scores, labels):
    """Area under the ROC curve: the Mann-Whitney U of the positives over all (positive, negative) pairs, with the
    average rank for tied scores (a tie counts as half a win). `labels` are 0/1 (or bool).

    ValueError for a NaN score, labels that are not 0 or 1, a length mismatch, or a class with no members: an AUC
    that cannot be computed fails loudly rather than returning NaN, which a later comparison would read as "false".
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("hazard.auc: scores and labels must be 1-D and of one length, got {} scores and {} labels"
                         .format(scores.size, labels.size))
    if np.isnan(scores).any():
        raise ValueError("hazard.auc: {} of the scores are NaN; score only the rows where the model has an opinion"
                         .format(int(np.isnan(scores).sum())))
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("hazard.auc: labels must be 0 or 1")
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = scores.size - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("hazard.auc: it needs both classes, got {} positives and {} negatives"
                         .format(n_positive, n_negative))

    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))   # first row of each run of ties
    stops = np.append(starts[1:], ordered.size)
    run_rank = (starts + 1 + stops) / 2.0   # a run covering ranks start + 1 ... stop has their mean as its rank
    ranks = np.empty(scores.size)
    ranks[order] = np.repeat(run_rank, stops - starts)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


# -- the sampler ------------------------------------------------------------------------------------------------------

def sample_mask(prob, onset_class, third, durations, rng):
    """Bool mask of one block (True = ball missing) with gaps drawn from the per-frame onset probability `prob`.

    prob         (n,) in [0, 1): the probability that a gap starts at each frame (see the module docstring).
    onset_class  (n,) ints: 1 where an onset would be near a line, 0 otherwise.
    third        (n,) ints: 0, 1 or 2 for the first, second and last third of the half (third_of_minute).
    durations    durations[onset_class][third]: a non-empty 1-D int array of gap lengths in frames, each at least 1.
    rng          a numpy Generator.

    From t = 0: draw E ~ Exponential(1); the onset is the first frame s >= t where L(s) - L(t - 1) >= E, with L the
    running sum of -log(1 - prob) (the exact inverse transform for a discrete-time hazard, by np.searchsorted); draw
    the length d of the gap from durations[onset_class[s]][third[s]]; mark frames s ... s + d - 1, cut at the end of
    the block; continue from t = s + d, so that no onset falls inside a gap. It stops once no onset falls in the block.
    The draws are exactly one rng.exponential(1.0) then one rng.choice(options) per gap, plus a last exponential that
    overshoots the block, so a seeded Generator gives the same gaps on every machine.
    """
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 1:
        raise ValueError("hazard.sample_mask: prob must be 1-D, one probability per frame, got shape {}"
                         .format(prob.shape))
    n = prob.size
    if np.isnan(prob).any():
        raise ValueError("hazard.sample_mask: prob has {} NaN; give 0 where no gap can start, for instance where the "
                         "features are undefined".format(int(np.isnan(prob).sum())))
    if prob.min(initial=0.0) < 0.0 or prob.max(initial=0.0) >= 1.0:
        raise ValueError("hazard.sample_mask: prob must lie in [0, 1); a probability of 1 has an infinite hazard "
                         "(cap it, e.g. at 1 - 1e-9)")
    onset_class = _labels("onset_class", onset_class, n, (0, 1), "0 or 1")
    third = _labels("third", third, n, (0, 1, 2), "0, 1 or 2")
    table = _duration_table(durations)

    mask = np.zeros(n, dtype=bool)
    cumulative = np.cumsum(-np.log1p(-prob))
    t = 0
    while t < n:
        base = cumulative[t - 1] if t else 0.0
        start = t + int(np.searchsorted(cumulative[t:], base + rng.exponential(1.0), side="left"))
        if start >= n:
            break
        length = int(rng.choice(table[onset_class[start]][third[start]]))
        mask[start:start + length] = True
        t = start + length
    return mask


def _labels(name, values, n, allowed, wording):
    """`values` as an int array of n entries drawn from `allowed`, or a ValueError that says which one is wrong."""
    values = np.asarray(values)
    if values.shape != (n,):
        raise ValueError("hazard.sample_mask: {} has {} values but prob has {}; give one per frame"
                         .format(name, values.size, n))
    if values.dtype.kind not in "iub":
        raise ValueError("hazard.sample_mask: {} must be an integer array, got dtype {}".format(name, values.dtype))
    values = values.astype(np.int64)
    if not np.isin(values, allowed).all():
        raise ValueError("hazard.sample_mask: {} must be {} in every frame".format(name, wording))
    return values


def _duration_table(durations):
    """durations as [[int64 array] x 3] x 2, every entry checked: lengths in frames, at least 1, none empty."""
    table = []
    for onset_class in range(2):
        row = []
        for third in range(3):
            name = "durations[{}][{}]".format(onset_class, third)
            try:
                options = np.asarray(durations[onset_class][third])
            except (IndexError, KeyError, TypeError):
                raise ValueError("hazard.sample_mask: {} is missing; give gap lengths for both onset classes and all "
                                 "three thirds, as a 2 x 3 table".format(name)) from None
            if options.ndim != 1 or options.size == 0:
                raise ValueError("hazard.sample_mask: {} must be a non-empty 1-D array of gap lengths, got shape {}"
                                 .format(name, options.shape))
            if options.dtype.kind not in "iu":
                raise ValueError("hazard.sample_mask: {} must hold whole frame counts (ints; convert seconds to "
                                 "frames first), got dtype {}".format(name, options.dtype))
            if options.min() < 1:
                raise ValueError("hazard.sample_mask: {} holds a length of {} frames; a gap lasts at least 1 frame"
                                 .format(name, int(options.min())))
            row.append(options.astype(np.int64))
        table.append(row)
    return table
