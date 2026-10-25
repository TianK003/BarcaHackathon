"""P02's refiner: move each kind of edge of a decoded block by a fixed number of frames.

    [[refine]]
    kind = "shift_edges"
    team = 0           # frames: a change of team (H <-> A)
    into_dead = 0      # frames: a change from H or A into D
    out_of_dead = 0    # frames: a change from D into H or A

Why
---
A decoder's edges can sit early or late against the truth, and not by the same amount for every kind of edge: the
truth's flag changes when its provider's does, ours when the evidence turns, and a hand-over, a stoppage and a restart
need not agree. spellF1 matches possessions by their overlap, so an edge that is always a little off costs every short
possession something (docs/results.md A01). P02 measures that lag per kind of edge; this refiner applies a measured
value and has none of its own. Every default is 0, and 0 is the identity.

What moves
----------
A boundary is a change of label inside the block: position i (1 <= i < len) where labels[i] != labels[i - 1], the first
frame of the new run. Its kind comes from the pair of labels: H <-> A is `team`, H or A -> D is `into_dead`, D -> H or A
is `out_of_dead`. A positive value moves the boundary later (the new label starts later), a negative one earlier. A
value is a whole number of frames, |value| <= 250 (ten seconds at 25 fps), and bool is not a number here.

Two kinds of boundary never move:

* a change into or out of X. An X is the decoder's way of saying "unknown", and a boundary beside one is not an
  edge it has located.
* a change at a kick-off frame, a frame of `ctx.run.inputs.kickoffs`. The kick-off is known from Smart Tagging's clock
  and not estimated, so a lag measured on the other edges says nothing about it. The comparison is in frames of the
  match: ctx.block.start + position.

How
---
`apply(labels, offsets, fixed)` is the whole rule, a pure function that the measuring tool of P02 calls too, with its
own offsets for each boundary. Each boundary b_i gets the target t_i = b_i + offset_i, or t_i = b_i when it is fixed.
Then, in this order:

1. a target may not cross a fixed boundary. The j-th boundary after a fixed one at p is not earlier than p + j, and
   the j-th before one at q is not later than q - j, so that each keeps a frame of its own. Without a fixed
   boundary this step does nothing;
2. each target is clipped to [1, len - 1], so the block's first and last run keep a frame;
3. a pass from left to right sets t_i = max(t_i, t_(i-1) + 1), and one from right to left sets
   t_i = min(t_i, t_(i+1) - 1), with t = len beyond the last.

So the runs keep their labels and their order, and every run keeps at least one frame: nothing is added, removed or
merged, which also means that a shift cannot repair flicker. Where two boundaries would meet, the later one is pushed
later (the first pass), and a pile-up at the end of the block is pulled back (the second).

Step 1 is the one thing more than P02's clip and two passes. Without it a boundary that moves past a fixed one would
push it in the first pass, and "never moves" would hold only until two boundaries collide: a hand-over shifted onto
the edge of an X would drag that edge along. With it a fixed boundary is where it was in every case, and the result is
the plain clip and two passes to the frame whenever those would not have moved a fixed boundary (checked against the
algorithm written as loops, in pipeline/selftest.py).

The posterior
-------------
With all three values 0 the block is returned as it came, labels and posterior. Otherwise the labels are not the ones
the posterior was made for, and the posterior is None: the contract allows that, and a posterior that disagrees with the
labels would be worse than none. (The rule is about the setting and not about whether anything moved.)

What it does not do
-------------------
* It picks no value. P02's rule does, from the lag of the pipeline on the benchmark and of the public window's
  conventions, and a config carries them (docs/proposals.md P02).
* It does not look at the evidence. An edge at a kick-off is held and an edge beside X is held; the others move by
  their kind's shift whatever the data say there.
"""

import numbers

import numpy as np

from pipeline.contracts import A, D, Decoded, H, X

# Params
MAX_SHIFT = 250                                  # frames: the largest |value| of a parameter (10 s at 25 fps)
TYPES = ("team", "into_dead", "out_of_dead")     # the kinds of boundary that have an offset; the parameters' names
ABSTAIN = "x"                                    # the kind of a change into or out of X: no offset, never moved
KINDS = TYPES + (ABSTAIN,)


def _kind_table():
    """(4, 4) array: the index into KINDS of a change from the code on the row to the code on the column."""
    table = np.full((4, 4), KINDS.index(ABSTAIN), dtype=np.int64)
    for before, after, kind in ((H, A, "team"), (A, H, "team"), (H, D, "into_dead"), (A, D, "into_dead"),
                                (D, H, "out_of_dead"), (D, A, "out_of_dead")):
        table[before, after] = KINDS.index(kind)
    return table


KIND_OF_CHANGE = _kind_table()


def refine(ctx, decoded, *, team=0, into_dead=0, out_of_dead=0):
    """Decoded for ctx.block: the labels of `decoded` with each boundary moved by its kind's shift, in frames.

    See the module docstring for which boundaries move and how collisions are settled. All three zero returns
    `decoded` itself (labels and posterior); anything else returns new labels and no posterior. The labels that came in
    are read, never written: the runner may hand over a read-only array.
    """
    shifts = {name: _check_shift(name, value)
              for name, value in (("team", team), ("into_dead", into_dead), ("out_of_dead", out_of_dead))}
    if not any(shifts.values()):
        return decoded
    labels = decoded.labels
    positions, kinds = boundaries(labels)
    offsets = np.array([shifts.get(kind, 0) for kind in kinds], dtype=np.int64)
    kickoffs = np.fromiter(ctx.run.inputs.kickoffs.values(), dtype=np.int64)
    fixed = np.array([kind == ABSTAIN for kind in kinds], dtype=bool) | np.isin(ctx.block.start + positions, kickoffs)
    return Decoded(labels=apply(labels, offsets, fixed))


def boundaries(labels):
    """(positions, types): where `labels` changes, and of which kind.

    positions is int64, ascending, each the index of the first frame of a new run (so in [1, len - 1]); types is a
    tuple of the same length of "team", "into_dead", "out_of_dead", or ABSTAIN for a change into or out of X.
    `labels` is a 1-D integer array of codes 0-3 (contracts.H, A, D, X) and is only read.
    """
    labels = _check_labels(labels)
    positions = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).astype(np.int64)
    kinds = KIND_OF_CHANGE[labels[positions - 1], labels[positions]]
    return positions, tuple(KINDS[kind] for kind in kinds)


def apply(labels, offsets, fixed):
    """A new int8 array: `labels` with boundary i moved by offsets[i] frames, unless fixed[i], by the rule above.

    `offsets` is a 1-D integer array and `fixed` a 1-D bool array, each with one entry per boundary of `labels` in the
    order of `boundaries(labels)`: a positive offset moves its boundary later. `labels` is only read; the result is a
    copy even when nothing moves.
    """
    labels = _check_labels(labels)
    positions, _ = boundaries(labels)
    n, m = len(labels), len(positions)
    offsets, fixed = _check_per_boundary(offsets, fixed, m)
    moved = labels.astype(np.int8)
    if m == 0:
        return moved

    low, high = _room(positions, fixed, n)
    targets = np.clip(np.where(fixed, positions, positions + offsets), low, high).tolist()
    for i in range(1, m):                                        # left to right: a frame after the one before
        targets[i] = max(targets[i], targets[i - 1] + 1)
    for i in range(m - 1, -1, -1):                               # right to left: a frame before the one after
        targets[i] = min(targets[i], (targets[i + 1] if i + 1 < m else n) - 1)

    run_labels = labels[np.concatenate(([0], positions))]
    lengths = np.diff(np.concatenate(([0], targets, [n])))
    return np.repeat(run_labels, lengths).astype(np.int8)


def _room(positions, fixed, n):
    """(low, high), int64 arrays: the frames each boundary may be put on without crossing or pushing a fixed one.

    Between two fixed boundaries there are k others, each of which keeps a frame of its own, so the j-th after a fixed
    boundary at p cannot be earlier than p + j, and the j-th before one at q not later than q - j. A fixed boundary's
    room is its own position. Where no fixed boundary lies on a side, the block's own limits, 1 and n - 1, stay.
    """
    m = len(positions)
    low = np.ones(m, dtype=np.int64)
    high = np.full(m, n - 1, dtype=np.int64)
    last = None                                                  # index of the nearest fixed boundary on the left
    for i in range(m):
        if fixed[i]:
            last = i
            low[i] = positions[i]
        elif last is not None:
            low[i] = positions[last] + (i - last)
    following = None                                             # and on the right
    for i in range(m - 1, -1, -1):
        if fixed[i]:
            following = i
            high[i] = positions[i]
        elif following is not None:
            high[i] = positions[following] - (following - i)
    return low, high


def _check_shift(name, value):
    """`value` as an int, or ValueError naming the parameter: a whole number of frames, bool refused, |value| <= 250."""
    # numbers.Integral admits numpy integers too, for sweeps over np.arange(...) in a notebook; bool is excluded.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or abs(int(value)) > MAX_SHIFT:
        raise ValueError("shift_edges: {} must be a whole number of frames between -{} and {} (positive moves the edge "
                         "later), got {!r}".format(name, MAX_SHIFT, MAX_SHIFT, value))
    return int(value)


def _check_labels(labels):
    """`labels` itself, or ValueError: it must be a 1-D integer array of codes 0-3."""
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise ValueError("shift_edges: labels must be a 1-D numpy array of label codes, got {}".format(
            "shape {}".format(labels.shape) if isinstance(labels, np.ndarray) else type(labels).__name__))
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("shift_edges: labels are {}, expected integer codes (contracts.H, A, D, X)".format(
            labels.dtype))
    if labels.size and (labels.min() < 0 or labels.max() > X):
        raise ValueError("shift_edges: label codes must be 0-3 (contracts.H, A, D, X)")
    return labels


def _check_per_boundary(offsets, fixed, m):
    """(offsets as int64, fixed as bool), or ValueError: each a 1-D array with one entry for each of m boundaries."""
    for name, values, kind in (("offsets", offsets, "integer"), ("fixed", fixed, "bool")):
        if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) != m:
            raise ValueError("shift_edges: {} must be a 1-D numpy array of {} boundaries, one per entry of "
                             "boundaries(labels); got {}".format(
                                 name, m, "shape {}".format(values.shape) if isinstance(values, np.ndarray)
                                 else type(values).__name__))
        valid = np.issubdtype(values.dtype, np.integer) if kind == "integer" else values.dtype == np.bool_
        if not valid:
            raise ValueError("shift_edges: {} must be {} (frames), it is {}".format(
                name, "an integer array" if kind == "integer" else "a bool array (True: do not move)", values.dtype))
    return offsets.astype(np.int64), fixed
