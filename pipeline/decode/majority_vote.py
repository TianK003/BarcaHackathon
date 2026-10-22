"""Step 2 of the organisers' baseline, as a decoder: a centred majority vote over the evidence's `home`.

    [decoder]
    kind = "majority_vote"
    half_width = 50      # frames each side; 50 = +/- 2 s at 25 fps
    fallback = "D"       # label where no source had an opinion anywhere in the window

For each frame of the block, the `home` probabilities of the frames within +/- half_width -- clipped to the block,
never reaching across halftime -- are summed into a home score and an away score. The higher one wins, ties going to
home; a window without a single opinion gets the fallback label. half_width = 0 is the raw per-frame label.

With one source whose `home` is a hard 0 or 1, this is exactly starter/baseline.smooth(): the sums are counts, kept
exact by float64 cumulative sums of whole numbers, with the same tie rule (home >= away -> H), the same clipping and
the same D where nothing voted. That is what lets the pipeline rebuild the published baselines byte for byte.

It is a stand-in, kept for that regression and as the floor P06 has to beat. What it does not do:

* It ignores `alive` completely, so D only ever comes from the fallback -- wherever the tracking lost the ball --
  which is wrong on 66% of dead-ball frames, where the ball is visible (docs/traps.md T12).
* It has no notion of a possession as an interval, so it still fragments: a blip that survives the vote splits a
  possession into three (T04).
* It pools several sources with a weighted mean of `home`, which only approximates what `weight` means for a
  probabilistic decoder (an exponent on each source's likelihood -- pipeline/contracts.py). Ties between fractional
  scores are decided by float rounding; deterministic on one machine, but no longer a clean count.
"""

import numbers

import numpy as np

from pipeline.contracts import A, CODES, Decoded, H


def decode(ctx, *, half_width, fallback="D"):
    """Decoded(labels) for ctx.block: per frame, H or A by the windowed vote, `fallback` where nobody voted."""
    # numbers.Integral admits numpy integers too, for sweeps over np.arange(...) in a notebook; bool is excluded.
    if isinstance(half_width, bool) or not isinstance(half_width, numbers.Integral) or half_width < 0:
        raise ValueError("majority_vote: half_width must be a whole number of frames >= 0, got {!r}".format(
            half_width))
    half_width = int(half_width)
    if fallback not in CODES:
        raise ValueError("majority_vote: fallback must be one of {}, got {!r}".format(sorted(CODES), fallback))

    n_frames = len(ctx.block)
    home, has_vote = pooled_home(ctx.evidence.values(), n_frames)
    home_score = window_sum(np.where(has_vote, home, 0.0), half_width)
    away_score = window_sum(np.where(has_vote, 1.0 - home, 0.0), half_width)
    # Count the voters separately, in integers: "did anyone vote" must not depend on a float sum coming out at 0.0.
    voters = window_sum(has_vote.astype(np.int64), half_width)

    labels = np.full(n_frames, CODES[fallback], dtype=np.int8)
    seen = voters > 0
    labels[seen] = np.where(home_score[seen] >= away_score[seen], H, A)
    return Decoded(labels=labels)


def pooled_home(sources, n_frames):
    """(home, has_vote): the weighted mean of `home` over the sources with an opinion on each frame.

    A source with NaN on a frame is left out of that frame's mean, not counted as 0.5. With a single source of
    weight 1.0 the mean is that source's value exactly (h * 1.0 / 1.0), which the regression relies on.
    """
    numerator = np.zeros(n_frames)
    denominator = np.zeros(n_frames)
    for source in sources:
        known = ~np.isnan(source.home)
        numerator[known] += source.weight * source.home[known]
        denominator[known] += source.weight
    has_vote = denominator > 0
    home = np.zeros(n_frames)
    home[has_vote] = numerator[has_vote] / denominator[has_vote]
    return home, has_vote


def window_sum(values, half_width):
    """Sum of values over the centred window [i - half_width, i + half_width], clipped to the array, for every i.

    Two lookups into one cumulative sum per frame, so a 125-frame window costs the same as a 1-frame one.
    """
    n_frames = len(values)
    cumulative = np.concatenate(([0], np.cumsum(values)))
    index = np.arange(n_frames)
    return (cumulative[np.minimum(n_frames, index + half_width + 1)]
            - cumulative[np.maximum(0, index - half_width)])
