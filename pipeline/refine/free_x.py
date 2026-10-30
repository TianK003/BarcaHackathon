"""P12's refiner: abstain (X) inside a possession where the decoder is unsure of its own label.

    [[refine]]
    kind = "free_x"
    threshold = 0.5    # an X where the posterior of the decoded label is below this
    margin = 3         # frames: never within this many frames of a run's edge (default 3)

Why
---
`spellF1` forward-fills an X with the last label before it (docs/results.md, `possession_metrics.fill_abstentions`), so
an X inside a run of H or A leaves every possession exactly as it was, and only `macroF1` sees it: it counts the frame
as a miss instead of a guess. A guess is worth making when it is right more often than not, and the decoder's posterior says which frames
are not (the design, section 5.B, argues the exact bar). So this refiner turns the frames below the
threshold into X, and the threshold is for a declared rule on the benchmark to set: the code ships with none.

Which frames
------------
A frame i of the block becomes X iff all of these hold:

* its run (the maximal stretch of one label in `decoded.labels`) is H or A. A `D` run is never touched, and an X
  that came in stays, so a run ends at it;
* it is at least `margin` frames from the run's first frame and from its last: i - start >= margin and
  stop - 1 - i >= margin. The edges are what spellF1 matches possessions on, and shift_edges moves them by up to 2
  frames in main.toml, so the margin (3) is more than the largest move: a shifted edge never reaches an X. A run of
  2 * margin frames or fewer has no frame to free;
* the posterior of the decoded label, `decoded.posterior[i, label]` with columns P(H), P(A), P(D), is below
  `threshold`.

Because the freed frames are inside a run and the frames at its edges keep the label, the forward fill puts the run's
own label back on every X: the scorer sees the labels the decoder made, so spellF1 is unchanged to the last digit.
Threshold 0 frees nothing (a posterior is never below 0) and returns the labels as they came.

The posterior
-------------
The decoder must be told to make one (`posterior = true` in `[decoder]`), and free_x must come before `shift_edges`,
which drops it. The posterior that comes out is the one that came in: the frames keep the numbers the decoder gave
them, and an X is no claim about them.

What it does not do
-------------------
* It picks no threshold. R016's rule on the benchmark does.
* It never abstains across a hand-over or a stoppage: the margin keeps it away from every edge.
"""

import numbers

import numpy as np

from pipeline.contracts import A, D, Decoded, H, X

# Params
# frames: more than shift_edges' largest move in main.toml (2), so a shifted edge never reaches an X (design 5.B)
MARGIN = 3


def refine(ctx, decoded, *, threshold, margin=MARGIN):
    """Decoded for ctx.block: `decoded`'s labels with X on the unsure interior frames of each H and A run.

    See the module docstring for the rule. The labels that came in are read, never written (the runner may hand over a
    read-only array), the result is a new int8 array, and the posterior is the same object.
    """
    threshold = _check_threshold(threshold)
    margin = _check_margin(margin)
    posterior = decoded.posterior
    if posterior is None:
        raise ValueError("free_x needs the decoder's posterior: set posterior = true in [decoder], and put free_x "
                         "before shift_edges, which drops it")
    labels = decoded.labels
    if posterior.shape != (len(labels), 3):
        raise ValueError("free_x: the posterior must have one row of P(H), P(A), P(D) per frame, shape {}, got {}".format(
            (len(labels), 3), posterior.shape))
    freed = freeable(labels, margin)
    freed &= posterior[np.arange(len(labels)), np.minimum(labels, D)] < threshold
    return Decoded(labels=np.where(freed, X, labels).astype(np.int8), posterior=posterior)


def freeable(labels, margin):
    """Bool array: the frames of `labels` that are H or A and at least `margin` frames from both edges of their run.

    A run is a maximal stretch of one label code; a frame is `i - start >= margin` and `stop - 1 - i >= margin`.
    Vectorised over runs: the start and the end of each frame's run come from the change positions by a running
    maximum and minimum.
    """
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=bool)
    index = np.arange(n)
    starts = np.where(np.concatenate(([True], labels[1:] != labels[:-1])), index, 0)
    last = np.where(np.concatenate((labels[1:] != labels[:-1], [True])), index, n)
    start = np.maximum.accumulate(starts)                       # first frame of each frame's run
    end = np.minimum.accumulate(last[::-1])[::-1]               # last frame of each frame's run
    return ((labels == H) | (labels == A)) & (index - start >= margin) & (end - index >= margin)


def _check_threshold(value):
    """`value` as a float, or ValueError naming the parameter: a real number in [0, 1], bool and NaN refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("free_x: threshold must be a number between 0 and 1 (frames whose posterior is below it "
                         "become X), got {!r}".format(value))
    return float(value)


def _check_margin(value):
    """`value` as an int, or ValueError naming the parameter: a whole number of frames, at least 1, bool refused."""
    # numbers.Integral admits numpy integers too, for sweeps over np.arange(...) in a notebook; bool is excluded.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise ValueError("free_x: margin must be a whole number of frames, at least 1, got {!r}".format(value))
    return int(value)
