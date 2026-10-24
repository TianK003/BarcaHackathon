"""A first opinion on alive versus dead: does the tracker see the ball?

    [[evidence]]
    kind = "ball_presence"
    margin = 0        # frames: a missing frame this near a frame with a ball gets no opinion (0 = every missing frame)
    missing = 0.2     # alive, where the ball is missing
    seen = 0.585      # alive, where the ball is seen

The organisers' baseline already makes this call, silently: where the tracking has no ball it writes D, and nothing
else ever produces a D. This source states the same rule as evidence, so that a decoder can weigh it, pay for the
change into and out of D, and overrule it, which the baseline's fallback cannot. `home` is NaN on every frame: whether
the tracker sees the ball says nothing about who has it.

The two values come from A01 (docs/results.md), by rule, and are not tuned. In the public window the ball is missing on
33.6% of dead-ball frames and on 6.5% of in-play frames, so a missing ball is 5.2 times likelier in a stoppage. It is
rounded down to 4, which is P(alive) = 1 / (1 + 4) = 0.2, because that window is the best-covered block of the match
(A02) and the ratio over the whole match is lower. A visible ball goes the other way, weakly: 66.4% of dead frames
against 93.5% of in-play ones is a ratio of 1.41 for alive, P(alive) = 0.935 / (0.935 + 0.664) = 0.585. Both rest on a
handful of stoppages (four in that window), so they are a first estimate, and P05 replaces this source.

The `seen` term is not decoration. Without it a stretch with the ball, between two stretches without, would pay for two
changes of state out of what the home evidence earns there, about a sixth of a nat a frame for a nearest player right
78% of the time, and a heavy switch penalty would absorb possessions of many seconds into the stoppages around them.
With it each such frame earns log(0.585 / 0.415) = 0.34 nats more. seen = 0.5 adds exactly nothing to the decoder's
emissions (log(2 * 0.5) = 0), and is the control "without it".

What it cannot do is tell a missing ball from a dead one (docs/traps.md T12). The ball is missing on 41% of graded
frames, in 150 gaps with a median of 10.7 s, and most of them are live play; per frame the evidence is weak on purpose,
and only a long run of it, against the decoder's penalties, may become a D. `margin` leaves the edges of a gap to the
other evidence: with margin = 3 only the frames more than 3 frames from the nearest frame with a ball speak.
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion


def compute(run, *, margin=0, missing=0.2, seen=0.585):
    """Evidence(alive = `seen` with a ball, `missing` more than `margin` frames from one, NaN nearer; home = NaN)."""
    if isinstance(margin, bool) or not isinstance(margin, numbers.Integral) or margin < 0:
        raise ValueError("ball_presence: margin must be a whole number of frames >= 0, got {!r}".format(margin))
    missing = probability("missing", missing)
    seen = probability("seen", seen)

    inputs = run.inputs
    present = inputs.atd.ball_present
    alive = no_opinion(inputs.n_frames)
    alive[present] = seen
    alive[~present & (frames_to_ball(present) > margin)] = missing
    return Evidence(alive=alive, home=no_opinion(inputs.n_frames))


def probability(name, value):
    """`value` as a float, or ValueError unless it is a real number in [0, 1] (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 <= value <= 1.0:
        raise ValueError("ball_presence: {} must be a probability, a number in [0, 1], got {!r}".format(name, value))
    return float(value)


def frames_to_ball(present):
    """Per frame, how many frames to the nearest frame with a ball: 0 on one, inf everywhere if there is none."""
    index = np.arange(len(present))
    since_last = frames_since(present, index)                  # back to the last frame with a ball
    until_next = frames_since(present[::-1], index)[::-1]      # the same on the reversed match: on to the next one
    return np.minimum(since_last, until_next)


def frames_since(present, index):
    """Per frame, the distance back to the latest frame at or before it with a ball; inf before the first one."""
    latest = np.maximum.accumulate(np.where(present, index, -1))
    return np.where(latest >= 0, index - latest, np.inf)
