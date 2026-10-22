"""Step 1 of the organisers' baseline, as evidence: the team of the tracked player nearest the ball.

    [[evidence]]
    kind = "nearest_player"

On every frame where the tracking sees the ball and at least one player, `home` is 1.0 if the nearest tracked slot
belongs to the home team and 0.0 if it belongs to the away team. Everywhere else -- the ball missing on 41% of graded
frames, or nobody tracked -- it is NaN: no opinion, rather than a guess. `alive` is NaN on every frame, because
distance says nothing about whether play is live; that is P05's job.

This is deliberately the baseline's exact rule, computed so that it reproduces starter/baseline.py bit for bit:

* distances in metres, with x scaled by the pitch length and y by its width, in the same float64 operations in the
  same order: (x * length - ball_x * length)^2 + (y * width - ball_y * width)^2;
* absent slots excluded, and a tie resolved to the lowest slot index -- np.argmin returns the first minimum, which
  is what the baseline's strict "<" keeps.

What it is not: the possessor. With the ball visible and in play, the nearest player's team is wrong on 22% of
frames -- passes in flight, duels, deflections (docs/traps.md T14). And it reports a hard 0 or 1 where a calibrated
probability would be nearer 0.78. Both are what P04 (sticky ownership from controlled touches) is for; this source
stays as the regression's first half and as a baseline to beat.
"""

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from starter.baseline import HOME


def compute(run):
    """Evidence(home = 1.0 / 0.0 by the nearest present slot's team, NaN without a ball or a player; alive = NaN)."""
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
    home[answered] = (atd.slot_team[nearest[answered]] == HOME).astype(np.float64)
    return Evidence(alive=no_opinion(inputs.n_frames), home=home)
