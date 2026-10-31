"""P22(b)'s estimators: the touch rule as a teacher for P07's logit, and an intercept per graded block, fitted
leave-one-block-out. Numpy and the standard library only, like `trees`; the source `shape_selftrain` and the gate tool
(pipeline/benchmark/selftrain.py) both import this module, so the gate measures the code that would be served.

The belief. `shape_possession` gives a logit z of who has the ball from the shape of the two teams, and its trees were
fitted on other teams' matches. `controlled_touch` reads the ball and the player next to it and never the shape, and on
the DFL matches its episode frames name the right team 93% of the time (`main.toml`'s confidence). So the episode frames
are labels for z, a noisy but independent teacher, and a per-match offset in z shows up as an intercept fitted on them
with z as the offset. Every label here is made by a rule applied to the tracking: no person decides one (RULES.md).

The teacher (`teacher_labels`). `controlled_touch` at the table given (`main.toml`'s is 1.5 m, 6.0 m/s, 1 frame), called
with `TEACHER_CONFIDENCE` and `TEACHER_HOLD` so that its `home` is 0.99 or 0.01 exactly on the kept episode frames and
NaN everywhere else, which is `y` (1.0 home, 0.0 away). A *row* is a frame where `y` is known and the dead-ball model's
P(alive) is at least `ALIVE_LEVEL`: a keeper holding the ball or a player placing it for a restart is a touch whose
truth is D, and the model needs no label of this match to say so. z's own finiteness (someone is present) is tested by
`block_offsets`, which takes the z of the match beside the rows.

E2, the estimator served (`noise_aware_offset`). Two steps, with epsilon the share of the teacher's labels that are
wrong (0.07 = 1 - the touch rule's confidence 0.93, which is measured, not tuned). First s, the maximum-likelihood
offset under symmetric label noise, a forward correction as in Patrini et al. (2017): the s that maximises
sum(y log q + (1 - y) log(1 - q)) with q = epsilon + (1 - 2 epsilon) sigma(z + s), found as the root of that sum's
derivative in s by bisection. s is then the *posterior* offset on the teacher's frames, which have a home share of their
own, so the second step removes that prior: offset = s - logit(pi), pi = clip((ybar - epsilon) / (1 - 2 epsilon),
`PI_CLIP`), the teacher's home share corrected for the noise. What is left is a likelihood offset, the quantity the
decoder wants from a source with prior 0.5 (R011, R020).

Why not E1, the class-balanced intercept (`balanced_offset`). With weights balanced on the noisy classes the minority
class's rows are the dirtier ones (at a home share of 0.65 and 7% noise, 12% of the labelled-away rows are truly home
against 4% of the labelled-home rows that are truly away), so E1 is pulled towards the teacher's minority class: on
synthetic data (4,000 rows, 20 repetitions, z the exact log-likelihood ratio, so that the offset to find is 0) it says
-0.107 at a share of 0.65 and +0.118 at 0.35, where E2 says +0.008 and +0.007. E2 is unbiased under symmetric noise at
any share, so it is the one served, and E1 stays for the gate as the oracle on clean labels, where there is no noise to
lean on.

Leave-one-block-out (`block_offsets`). The offset of graded block k is fitted on the rows of the *other* blocks only, so
every number about block k, a held-out check or the evidence the decoder reads on it, uses an offset that never saw
block k, and the teacher does not vote twice on the frames it labels. With two halves, each half is corrected by the
other half's fit. A block whose other blocks give fewer than `MIN_ROWS` rows of either class (20 s of controlled
touches a team), one class only, or no root in `BRACKET` gets an offset of 0.0 and its reason ("few rows", "no root"),
so that the source then equals P07 on that block, and nothing is raised: a red card or a short block can do this on real
data. `adapted_logit` adds each block's offset to z on that block's frames, NaN kept.

Determinism: the same inputs give the same offsets whatever the order of the rows and the number of threads. The sums
that decide an offset are `math.fsum`, exact and independent of the order; the bisection has no data-dependent branch
beyond the sign of such a sum; the offset is rounded to `OFFSET_DECIMALS` (9) before use, so that arithmetic which
differs by a ULP between machines cannot reach the CSV unless an offset sits within about 1e-16 of a rounding boundary.
The score is made of exp(-|t|) over (1 + exp(-|t|)) and never of 1 - sigma, which rounds to 0 for t above 37 and would
lose the root of a model that is very sure.
"""

import math
import numbers
from collections import namedtuple

import numpy as np

from pipeline.evidence import controlled_touch, dead_ball_model

# Params
TEACHER_CONFIDENCE = 0.99           # the teacher's home on a home episode frame (1 - it on an away one)
TEACHER_HOLD = 0.5                  # no hold: NaN between episodes, so home is a number on the kept episode frames only
ALIVE_LEVEL = 0.5                   # a row needs P(alive) of at least this from the dead-ball model
MIN_ROWS = 500                      # rows of each class among the other blocks': 20 s of controlled touches a team
PI_CLIP = (0.05, 0.95)              # the noise-corrected home share of the rows, clipped before its logit is taken
BRACKET = (-10.0, 10.0)             # the intercept is looked for in here, in nats
TOLERANCE = 1e-12                   # the bisection stops when its interval is narrower than this
OFFSET_DECIMALS = 9                 # an offset is rounded to this many decimals before use
NAME = "recalibration"
SOURCE = "shape_selftrain"          # the source whose `teacher` table check_teacher reads, for its messages
TEACHER_KEYS = ("radius", "max_rel_speed", "min_frames")     # a teacher table has these, and may add duel_radius
DUEL_RADIUS = 1.0                   # controlled_touch's own default
FIXED_KEYS = ("confidence", "hold")  # the source fixes these two (TEACHER_CONFIDENCE, TEACHER_HOLD)

Offset = namedtuple("Offset", "offset s pi_hat share")
Offset.__doc__ = """One fitted intercept. `offset` is the one to add to z (rounded to OFFSET_DECIMALS); `s` is E2's
    posterior offset before the prior's logit is taken off (E1's is its offset); `pi_hat` is E2's noise-corrected home
    share of the rows after the clip (None for E1); `share` is the mean of y."""
BlockOffset = namedtuple("BlockOffset", "offset rows_home rows_away fallback fitted")
BlockOffset.__doc__ = """Block k's offset: `offset` is 0.0 when `fallback` is "few rows" or "no root", else the
    `fitted` Offset's; `rows_home` and `rows_away` count the other blocks' rows that the fit saw, or would have seen."""


class NoRoot(ValueError):
    """The score of an intercept does not change sign on BRACKET: the model is more than 10 nats off on these rows."""


def check_teacher(teacher):
    """The teacher table (a dict) with `duel_radius` filled in, or a ValueError naming what is wrong with it.

    Its keys are `radius`, `max_rel_speed` and `min_frames`, and `duel_radius` if given (1.0 if not); `confidence` and
    `hold` are refused, since the source fixes them. The values are controlled_touch's to check.
    """
    keys = "radius, max_rel_speed and min_frames (and duel_radius, {} if left out)".format(DUEL_RADIUS)
    if not isinstance(teacher, dict):
        raise ValueError("{}: teacher must be a table with {}, such as {{ radius = 1.5, max_rel_speed = 6.0, "
                         "min_frames = 1 }}, got {!r}".format(SOURCE, keys, teacher))
    fixed = [key for key in FIXED_KEYS if key in teacher]
    if fixed:
        raise ValueError("{}: teacher has {}, which the source fixes (confidence {} and hold {}, so that the "
                         "teacher's home is a number exactly on the kept episode frames); remove {} from the table"
                         .format(SOURCE, " and ".join(fixed), TEACHER_CONFIDENCE, TEACHER_HOLD,
                                 "it" if len(fixed) == 1 else "them"))
    unknown = sorted(str(key) for key in teacher if key not in TEACHER_KEYS + ("duel_radius",))
    if unknown:
        raise ValueError("{}: teacher has the unknown key{} {}; the table has {}"
                         .format(SOURCE, "" if len(unknown) == 1 else "s", ", ".join(unknown), keys))
    missing = [key for key in TEACHER_KEYS if key not in teacher]
    if missing:
        raise ValueError("{}: teacher is missing {}; the table has {}".format(SOURCE, ", ".join(missing), keys))
    return dict(teacher, duel_radius=teacher.get("duel_radius", DUEL_RADIUS))


def teacher_labels(run, *, alive_model, teacher):
    """(y, rows): the touch rule's labels over the whole match, and the frames that may be used to fit.

    `y` is (n,) float64: 1.0 on a kept home episode frame, 0.0 on a kept away one, NaN everywhere else (with no hold
    there is nothing between the episodes). `rows` is (n,) bool: `y` is known and the dead-ball model `alive_model`
    (a path, as in the config) says P(alive) >= ALIVE_LEVEL. z's finiteness is not tested here: `block_offsets` takes
    the z mask. `teacher` is checked by `check_teacher` before anything is computed.
    """
    teacher = check_teacher(teacher)
    home = controlled_touch.compute(run, confidence=TEACHER_CONFIDENCE, hold=TEACHER_HOLD, **teacher).home
    alive = dead_ball_model.compute(run, model=alive_model).alive
    y = np.full(len(home), np.nan)
    with np.errstate(invalid="ignore"):                  # NaN compares False: no label, no row
        y[home > 0.5] = 1.0
        y[home < 0.5] = 0.0
        rows = ~np.isnan(y) & (alive >= ALIVE_LEVEL)
    return y, rows


def noise_aware_offset(z, y, *, epsilon):
    """E2: the offset to add to the logit z so that it is a likelihood, fitted on teacher labels y (0/1) that are wrong
    on a share `epsilon` of the rows, in (0, 0.5). The module docstring has the estimator.

    z must be finite and y 0 or 1 on every row, with both classes present, else ValueError; `NoRoot` if the score does
    not change sign on BRACKET. Returns Offset(offset, s, pi_hat, share).
    """
    epsilon = _epsilon(epsilon)
    z, y, n_home = _fit_inputs("noise_aware_offset", z, y)
    home = y == 1
    keep = 1.0 - 2.0 * epsilon

    def score(s):
        """d/ds of sum(y log q + (1 - y) log(1 - q)): keep * sigma * (1 - sigma) * (y / q - (1 - y) / (1 - q))."""
        up, down = _sigmoids(z + s)
        q, rest = epsilon + keep * up, epsilon + keep * down        # q = P(label home), rest = 1 - q
        return math.fsum((keep * up * down * np.where(home, 1.0 / q, -1.0 / rest)).tolist())

    s = _root(score, "noise_aware_offset")
    share = n_home / len(y)
    pi_hat = min(max((share - epsilon) / keep, PI_CLIP[0]), PI_CLIP[1])
    return Offset(_rounded(s - math.log(pi_hat / (1.0 - pi_hat))), s, pi_hat, share)


def balanced_offset(z, y):
    """E1: the class-balanced logistic intercept with offset z, the gate's oracle on clean labels. Not for noisy ones:
    it leans towards the minority class (the module docstring).

    The same arguments and errors as `noise_aware_offset` without epsilon. Returns Offset(offset, offset, None, share).
    """
    z, y, n_home = _fit_inputs("balanced_offset", z, y)
    home = y == 1
    w_home, w_away = 0.5 / n_home, 0.5 / (len(y) - n_home)

    def score(b):
        """d/db of the weighted log-likelihood: sum of w * (y - sigma(z + b)), each class weighing a half."""
        up, down = _sigmoids(z + b)
        return math.fsum(np.where(home, w_home * down, -w_away * up).tolist())

    b = _rounded(_root(score, "balanced_offset"))
    return Offset(b, b, None, n_home / len(y))


def block_offsets(z, y, rows, blocks, *, fit, min_rows=MIN_ROWS):
    """One BlockOffset per block of `blocks`, in order: block k's offset is `fit` on the other blocks' rows.

    z is the logit of every frame (NaN where P07 is silent), y `teacher_labels`' float array (read as 0/1 on the
    selected rows), rows its bool mask, `blocks` a list of (start, stop), half-open. The rows used are those of every
    block but k where `rows` and isfinite(z) hold; `fit` is a callable (z, y) -> Offset on them, y an int64 0/1 array
    (for E2, functools.partial(noise_aware_offset, epsilon=0.07)). With fewer than `min_rows` of either class the offset
    is 0.0 and `fallback` "few rows"; a NoRoot from `fit` gives 0.0 and "no root". `fitted` is the Offset or None.
    """
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rows = np.asarray(rows)
    if z.ndim != 1 or y.shape != z.shape or rows.shape != z.shape:
        raise ValueError("{}: block_offsets: z, y and rows must be 1-D arrays of one length, got shapes {}, {} and {}"
                         .format(NAME, z.shape, y.shape, rows.shape))
    if rows.dtype != np.bool_:
        raise ValueError("{}: block_offsets: rows must be a bool array (teacher_labels' second value), got {}"
                         .format(NAME, rows.dtype))
    blocks = _blocks(blocks, len(z))
    if isinstance(min_rows, bool) or not isinstance(min_rows, numbers.Integral) or min_rows < 1:
        raise ValueError("{}: min_rows must be a whole number of rows >= 1, got {!r}".format(NAME, min_rows))
    usable = rows & np.isfinite(z)
    if not np.isin(y[usable], (0.0, 1.0)).all():
        raise ValueError("{}: block_offsets: y must be 0.0 or 1.0 on every selected row, and is NaN or another value "
                         "on one: `rows` and `y` disagree".format(NAME))
    found = []
    for k in range(len(blocks)):
        pool = np.concatenate([np.flatnonzero(usable[start:stop]) + start
                               for other, (start, stop) in enumerate(blocks) if other != k] + [np.empty(0, np.intp)])
        labels = y[pool].astype(np.int64)
        rows_home = int(labels.sum())
        rows_away = len(labels) - rows_home
        if rows_home < min_rows or rows_away < min_rows:
            found.append(BlockOffset(0.0, rows_home, rows_away, "few rows", None))
            continue
        try:
            fitted = fit(z[pool], labels)
        except NoRoot:
            found.append(BlockOffset(0.0, rows_home, rows_away, "no root", None))
        else:
            found.append(BlockOffset(fitted.offset, rows_home, rows_away, None, fitted))
    return found


def adapted_logit(z, blocks, offsets):
    """z plus block k's offset on block k's frames: NaN stays NaN, a frame outside every block is unchanged.

    `offsets` holds one finite number per block, or the BlockOffsets that `block_offsets` returned. A new array.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("{}: adapted_logit: z must be a 1-D array, got shape {}".format(NAME, z.shape))
    blocks = _blocks(blocks, len(z))
    offsets = [item.offset if isinstance(item, BlockOffset) else item for item in offsets]
    if len(offsets) != len(blocks):
        raise ValueError("{}: adapted_logit needs one offset per block, got {} offsets for {} blocks"
                         .format(NAME, len(offsets), len(blocks)))
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, numbers.Real) or not math.isfinite(offset):
            raise ValueError("{}: the offsets must be finite numbers, got {!r}".format(NAME, offset))
    adapted = z.copy()
    for (start, stop), offset in zip(blocks, offsets):
        adapted[start:stop] += offset
    return adapted


def _sigmoids(t):
    """(sigma(t), 1 - sigma(t)) as two arrays, neither made by subtracting from 1: exact in both tails."""
    e = np.exp(-np.abs(t))
    d = 1.0 + e
    positive = t >= 0
    return np.where(positive, 1.0, e) / d, np.where(positive, e, 1.0) / d


def _root(score, who):
    """The root of `score` (a function of the intercept) on BRACKET, by bisection to TOLERANCE.

    The score must be positive at the bracket's low end and negative at its high end, else NoRoot. Each step keeps the
    half whose ends still bracket the sign change, and the midpoint of the last interval is the root.
    """
    low, high = BRACKET
    at_low, at_high = score(low), score(high)
    if not (at_low > 0.0 and at_high < 0.0):
        raise NoRoot("{}: {} has no root in [{}, {}]: its score is {:.3g} at {} and {:.3g} at {}, and has to be "
                     "positive at the first and negative at the second. The model's logit is more than {} nats off on "
                     "these rows, or the labels are not about it (block_offsets then uses an offset of 0)"
                     .format(NAME, who, low, high, at_low, low, at_high, high, high))
    while high - low >= TOLERANCE:
        middle = (low + high) / 2.0
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _rounded(value):
    """`value` as a Python float rounded to OFFSET_DECIMALS, with -0.0 written as 0.0."""
    return round(float(value), OFFSET_DECIMALS) + 0.0


def _epsilon(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 < value < 0.5:
        raise ValueError("{}: epsilon must be a number above 0 and below 0.5, the share of the teacher's labels that "
                         "are wrong (0.07 for the touch rule), got {!r}".format(NAME, value))
    return float(value)


def _fit_inputs(who, z, y):
    """(z as float64, y as int64, the number of home rows), after the checks that every fit needs."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y)
    if z.ndim != 1 or y.shape != z.shape:
        raise ValueError("{}: {}: z and y must be 1-D arrays of one length, got shapes {} and {}"
                         .format(NAME, who, z.shape, y.shape))
    if not np.isfinite(z).all():
        raise ValueError("{}: {}: z must be finite on every row; pass the rows where np.isfinite(z) holds "
                         "(block_offsets does)".format(NAME, who))
    if not np.isin(y, (0, 1)).all():
        raise ValueError("{}: {}: y must be 0 or 1 on every row; select the rows first (a NaN label is not a row)"
                         .format(NAME, who))
    y = y.astype(np.int64)
    n_home = int(y.sum())
    if n_home == 0 or n_home == len(y):
        raise ValueError("{}: {} needs both classes among its rows, got {} home and {} away".format(
            NAME, who, n_home, len(y) - n_home))
    return z, y, n_home


def _blocks(blocks, n):
    """`blocks` as a list of (start, stop) ints, or a ValueError naming the first that is not a pair of frames of n."""
    try:
        blocks = list(blocks)
    except TypeError:
        raise ValueError("{}: blocks must be a list of (start, stop) pairs of frames, got {!r}".format(NAME, blocks)) \
            from None
    checked = []
    for block in blocks:
        try:
            start, stop = block
            whole = all(isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in (start, stop))
        except (TypeError, ValueError):
            whole = False
        if not whole or not 0 <= start <= stop <= n:
            raise ValueError("{}: blocks must be (start, stop) pairs of whole frames with 0 <= start <= stop <= {}, "
                             "half-open as the graded blocks are, got {!r}".format(NAME, n, block))
        checked.append((int(start), int(stop)))
    return checked
