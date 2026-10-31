"""P22(b): P07's opinion on who has the ball, moved by an intercept per graded block that the other blocks' touch
episodes fit at run time.

    [[evidence]]
    kind = "shape_selftrain"
    weight = 0.2
    model = "pipeline/models/shape_possession_hgb.json"
    scope = "all"
    min_prob = 0.02
    alive_model = "pipeline/models/dead_ball_hgb.json"     # the teacher's dead-ball mask
    teacher = { radius = 1.5, max_rel_speed = 6.0, min_frames = 1, duel_radius = 1.0 }
    epsilon = 0.07                                         # 1 - controlled_touch's confidence 0.93 (R013)

`shape_possession` (P07) reads the shape of the two teams and gives a logit z of who has the ball, with trees fitted on
seven other matches; for two teams it never saw, z can be off by a constant, which in a gap of the ball (41% of the
graded frames) is the main team evidence besides Smart Tagging. `controlled_touch` reads the ball and the player next to
it and never the shape, and on the DFL matches its episode frames name the right team 93% of the time (`main.toml`'s
confidence), so those frames are labels for z, noisy but from another view of the frame. This source fits the constant
on them and adds it to z. The estimators are `recalibration.py`'s (the module docstring there has the maths); this
module puts them behind the evidence contract, and the benchmark's gate (pipeline/benchmark/selftrain.py) imports the
same functions, so what it measures is what is served.

What it does, in order
----------------------
1. z. `shape_possession`'s own: the symmetrised, unclipped logit, `shape_possession.load` and `_logit_home` (so the
   trees are walked once for both sources, per Inputs), from the fold of `inputs.game` if the file has one and else from
   `full`. It is NaN outside the graded blocks and where nobody is present.
2. The teacher. `controlled_touch` at the table given (`main.toml`'s is 1.5 m, 6.0 m/s, 1 frame, duel 1.0), called with
   the confidence 0.99 and the hold 0.5 that this source fixes, so that its `home` is a number exactly on the kept
   episode frames and NaN everywhere else: that is the label y, home 1 and away 0. A *row* is a frame where y is known,
   z is finite and the dead-ball model `alive_model` has alive evidence of at least 0.5 (it favours alive): a keeper
   holding the ball or a player placing it for a restart is a touch whose truth is D, and the model says so without a
   label of this match. `confidence` and `hold` are refused in the table, an unknown key and a missing one too
   (`check_teacher`).
3. The fit. Block k's offset b_k is E2 (`noise_aware_offset`, `epsilon` the share of the teacher's labels that are
   wrong) on the rows of the *other* blocks: with the two halves, each is corrected by the other's fit. A block whose
   other blocks give fewer than 500 rows of a class (20 s of touches a team), or one class, or whose likelihood has no
   maximum in [-10, 10] nats, gets b_k = 0 and a reason in `fit`'s result, and nothing is raised: there the source is
   `shape_possession` to the bit. A red card or a short half can do that on real data.
4. The output. `home` = clip(sigmoid(z + b_k), `min_prob`, 1 - `min_prob`) where P07 would speak under `scope`, NaN
   elsewhere (`scope` is P07's rule exactly: "gaps" is only where the served ball is missing, "all" every graded frame
   with someone present). `alive` is NaN everywhere: the shape of two teams is no evidence on whether the ball is in
   play. Both arrays are read-only.

Why cross-fitted. The offset of a block is fitted on the other blocks only, so that (i) the teacher does not vote twice
on the frames it labels: the decoder also reads `controlled_touch` itself on those frames, and an offset fitted on them
would hand that vote to P07 as well; and (ii) every number about block k, the evidence the decoder reads on it and the
held-out checks of the gate and of BAR-ATH, uses an offset that never saw block k. It costs one thing, that a block is
corrected by a fit of the other half of the match: the offset is taken to be a property of the match and not of a half.

Why E2 and not a balanced intercept. The teacher's rows have a home share of their own, and its labels are wrong on
`epsilon` of them. A class-balanced intercept is pulled towards the teacher's minority class, whose rows are the dirtier
ones; E2 corrects the noise and removes the share's prior, so that what is added is a likelihood offset, the quantity
the decoder wants from a source (recalibration.py has the synthetic comparison, and the balanced one stays for the gate
as the oracle on clean labels).

Labels. Every label here is made by a rule applied to the tracking of the match (`controlled_touch`), and the mask by a
model trained on other matches, so that no person decides any label (RULES.md allows a pseudo-label an algorithm made).
There is no hand label of this match, no ground truth and no human-produced data about it, and no constant here was set
from the public window. The fit runs inside `pipeline/run.py` from `data/`, with no artifact and no new dependency, so
the one command still goes from `data/` to the CSV. It is deterministic: its sums are `math.fsum`, the offsets are
rounded to 9 decimals before use, and the same inputs give the same bytes whatever the order of the rows.

What it costs and remembers. A `controlled_touch` pass and a `dead_ball_model` pass (whose walk of the trees is its own
memo), and a fit a block: 81 reads of the score on the grid, about 38 more for each maximum it finds (the bisection from
a cell of 0.25 nats down to 1e-12), and, where there are several maxima, a read of the likelihood at each. A read is a
pass over the rows of the other blocks, so a fit on 60,000 rows takes about 0.5 s. The offsets are remembered per Inputs
object and per (the two model files' sizes and modification times, the game, the table of the teacher, epsilon), so that
the configs of a benchmark stage that share them fit once; `scope` and `min_prob`, which are cheap, are applied after
it. A caller gets a list of its own. This module imports nothing from the benchmark.
"""

import functools
import numbers
import os
import weakref

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import dead_ball_model, kinematics, recalibration, shape_possession, trees

# Params
NAME = "shape_selftrain"
MODEL_EXAMPLE = "pipeline/models/shape_possession_hgb.json"
ALIVE_EXAMPLE = "pipeline/models/dead_ball_hgb.json"

_MEMO = weakref.WeakKeyDictionary()       # Inputs -> {stamp: the BlockOffsets of block_offsets, as a tuple}


def compute(run, *, model, alive_model, teacher, epsilon, scope="gaps", min_prob=0.02):
    """Evidence(alive = NaN; home = P07's logit plus the block's offset, as a clipped probability, where P07 speaks)."""
    scope = _scope(scope)
    min_prob = trees.check_min_prob(min_prob, NAME)
    z, offsets = _logit_and_offsets(run, model, alive_model, teacher, epsilon)

    inputs = run.inputs
    adapted = recalibration.adapted_logit(z, kinematics.blocks(inputs), offsets)      # NaN stays NaN
    speaks = ~np.isnan(adapted)
    if scope == "gaps":
        speaks &= ~inputs.atd.ball_present
    home = np.full(len(adapted), np.nan)
    home[speaks] = np.clip(trees.sigmoid(adapted[speaks]), min_prob, 1.0 - min_prob)
    home.flags.writeable = False
    alive = no_opinion(len(home))
    alive.flags.writeable = False
    return Evidence(alive=alive, home=home)


def fit(run, *, model, alive_model, teacher, epsilon):
    """The per-block offsets that `compute` adds: a list of `recalibration.BlockOffset`, one per graded block in order.

    A block's `offset` is 0.0 when its `fallback` is "few rows" or "no root" (the module docstring says when), and its
    `fitted` is the Offset otherwise; `rows_home` and `rows_away` count the other blocks' rows. The arguments are
    `compute`'s and are checked in the same way, before any file is read. Remembered per Inputs; the list is the
    caller's own.
    """
    return _logit_and_offsets(run, model, alive_model, teacher, epsilon)[1]


def _logit_and_offsets(run, model, alive_model, teacher, epsilon):
    """(z, offsets): P07's unclipped logit (read-only) and `block_offsets` for the graded blocks, as a list.

    The arguments are checked first (epsilon, the teacher's table, the two model paths and files), then z is taken from
    `shape_possession`'s memo and the offsets from this module's, made once per Inputs and stamp.
    """
    epsilon = _epsilon(epsilon)
    teacher = recalibration.check_teacher(teacher)
    path = trees.model_path(model, NAME, MODEL_EXAMPLE)
    alive_path = _alive_path(alive_model)
    artifact = shape_possession.load(path)
    try:
        dead_ball_model.load(alive_path)              # the file is there and is a dead-ball model, or this says why not
    except ValueError as error:                       # its hint says to fix `model`: in this entry it is `alive_model`
        raise ValueError("{}: alive_model: {}".format(NAME, error)) from error

    inputs = run.inputs
    key = inputs.game if inputs.game in artifact["folds"] else "full"
    chosen = artifact["folds"][key] if key != "full" else artifact["full"]
    z = shape_possession._logit_home(inputs, path, key, chosen)    # NaN outside the blocks and where nobody is present

    # repr: a table value of another kind (a list, say) is controlled_touch's to refuse, and must not fail in a hash
    table = tuple(sorted((name, repr(value)) for name, value in teacher.items()))
    stamp = (_stamp(path), _stamp(alive_path), inputs.game, table, epsilon)
    held = _MEMO.setdefault(inputs, {})
    if stamp not in held:
        y, rows = recalibration.teacher_labels(run, alive_model=alive_model, teacher=teacher)
        estimator = functools.partial(recalibration.noise_aware_offset, epsilon=epsilon)
        held[stamp] = tuple(recalibration.block_offsets(z, y, rows, kinematics.blocks(inputs), fit=estimator))
    return z, list(held[stamp])


def _stamp(path):
    """(path, size, modification time in ns): a model file that was rewritten is another file for the memo."""
    stat = os.stat(path)
    return path, stat.st_size, stat.st_mtime_ns


def _scope(value):
    if not isinstance(value, str) or value not in shape_possession.SCOPES:
        raise ValueError("{}: scope must be \"gaps\" (only where the ball is missing) or \"all\" (every graded "
                         "frame), got {!r}".format(NAME, value))
    return value


def _epsilon(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 < value < 0.5:
        raise ValueError("{}: epsilon must be a number above 0 and below 0.5, the share of the teacher's labels that "
                         "are wrong (1 - controlled_touch's confidence: 0.07 for main.toml's 0.93), got {!r}"
                         .format(NAME, value))
    return float(value)


def _alive_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("{}: alive_model must be the path of the dead-ball model file relative to the repository "
                         "root, such as \"{}\", got {!r}".format(NAME, ALIVE_EXAMPLE, value))
    return trees.model_path(value, NAME, ALIVE_EXAMPLE)
