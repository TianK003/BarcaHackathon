"""Step 1 of the organisers' baseline, as evidence: the team of the tracked player nearest the ball.

    [[evidence]]
    kind = "nearest_player"
    confidence = 1.0     # optional, in [0.5, 1.0]; 1.0 is the baseline's hard 0 / 1

On every frame where the tracking sees the ball and at least one player, `home` is `confidence` if the nearest tracked
slot belongs to the home team and 1 - confidence if it belongs to the away team: 1.0 and 0.0 by default. Everywhere
else -- the ball missing on 41% of graded frames, or nobody tracked -- it is NaN: no opinion, rather than a guess.
`alive` is NaN on every frame, because distance says nothing about whether play is live; that is P05's job.

This is deliberately the baseline's exact rule, computed so that it reproduces starter/baseline.py bit for bit:

* distances in metres, with x scaled by the pitch length and y by its width, in the same float64 operations in the
  same order: (x * length - ball_x * length)^2 + (y * width - ball_y * width)^2;
* absent slots excluded, and a tie resolved to the lowest slot index -- np.argmin returns the first minimum, which
  is what the baseline's strict "<" keeps.

What it is not: the possessor. With the ball visible and in play, the nearest player's team is wrong on 22% of
frames -- passes in flight, duels, deflections (docs/traps.md T14). P04 (sticky ownership from controlled touches) is
for that; this source stays as the regression's first half and as a baseline to beat.

And by default it reports a hard 0 or 1 where a calibrated probability would be nearer 0.78, the share of in-play,
ball-visible frames on which the nearest player's team is the right one (docs/results.md A01). A probabilistic decoder
(P06) is given that number: confidence = 0.78 puts 0.78 or 0.22 on every frame the source answers, the same frames and
the same direction as the hard output. It is one constant for the whole match, set from a measurement, not a fit.
A majority vote sums the two sides, so it takes the same side as with the hard values, bar float rounding in an exact
tie. The default, 1.0, gives 1.0 and 0.0 exactly (the second as 1 - 1.0), which is what the four baselines are rebuilt
through, byte for byte.

The 0.78 is a public-window statistic (A01). Since the organisers' rule of 23 Oct 2026 (that window is not a tuning
set) no submission config may use it unless it is re-derived on other games (A06 gives 77.5% on DFL's matches).
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from starter.baseline import HOME


def compute(run, *, confidence=1.0):
    """Evidence(home = confidence / 1 - confidence by the nearest present slot's team, NaN without a ball or a
    player; alive = NaN). With the default confidence, home is the baseline's hard 1.0 / 0.0."""
    # numbers.Real admits numpy floats too; bool is excluded, and NaN fails the range test.
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real) or not 0.5 <= confidence <= 1.0:
        raise ValueError("nearest_player: confidence must be a number in [0.5, 1.0] (1.0 = the baseline's hard 0 / 1; "
                         "below 0.5 would back the farther team), got {!r}".format(confidence))
    confidence = float(confidence)

    inputs = run.inputs
    atd = inputs.atd
    length, width = inputs.pitch

    # Metres, operation for operation as starter/baseline.py computes them, so every distance is the same float64.
    ball_x = atd.ball_xy[:, 0] * length
    ball_y = atd.ball_xy[:, 1] * width
    dx = atd.players_xy[:, :, 0] * length - ball_x[:, None]
    dy = atd.players_xy[:, :, 1] * width - ball_y[:, None]
    distance = dx * dx + dy * dy

    present = atd.players_present
    distance[~present] = np.inf          # an absent slot can never be nearest
    nearest = np.argmin(distance, axis=1)  # first minimum on ties: the lowest slot index, as in the baseline

    # argmin of a row that is all inf is 0, a real slot -- so frames without a ball or without any player are masked
    # explicitly rather than trusted to come out as inf.
    answered = atd.ball_present & present.any(axis=1)
    home = no_opinion(inputs.n_frames)
    nearest_is_home = atd.slot_team[nearest[answered]] == HOME
    home[answered] = np.where(nearest_is_home, confidence, 1.0 - confidence)
    return Evidence(alive=no_opinion(inputs.n_frames), home=home)
