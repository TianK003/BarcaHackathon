"""P05's rules for a dead ball: four traces of a stoppage in the play features, each a fixed rule of the game.

    [[evidence]]
    kind = "dead_ball_rules"
    alive_out = ...       # P(alive) where the ball is beyond a pitch line
    alive_still = ...     # ... where it has been still at a restart spot, on a line or in a goal area
    alive_slow = ...      # ... where the players have slowed together
    alive_kickoff = ...   # ... where the teams stand in a kick-off formation
    alive_none = ...      # ... where no rule fires

The ball is missing on 41% of graded frames and visible on 66% of dead ones (docs/results.md A01, T12), so
`ball_presence` is wrong in both directions. These rules read what the tracking shows instead, through
`play_features` (its docstring says where each feature comes from):

    out      ball_outside_m > 0.5 m                     the ball has crossed a line (the law says all of it must; the
                                                        half-metre is tracking noise)
    still    ball_still_s >= 2 s and (ball_spot_m <= 2 m, or |ball_outside_m| <= 1 m, or in a goal area)
                                                        waiting at a corner, a penalty or centre spot, a throw-in
                                                        line, a goal kick
    slow     slow_share_c4 >= 0.8                       four seconds either side, 80% of the players under 2 m/s
    kickoff  own_half_share >= 0.95 and centre_player_m <= 2 m
                                                        both teams in their own half, one slot at the centre spot

The thresholds come from the laws of the game and the literature and are never tuned. The five probabilities are the
parameters, required with no default: they are measurements (a later task's, on the benchmark), and a default here would
be a number nobody measured. `alive` is the smallest value of the rules that fire (the most confident call of a
stoppage wins) and `alive_none` where none does. NaN outside the graded blocks, and `home` is NaN everywhere: whether
the ball is dead says nothing about who has it.

The pre-kick-off rule holds here as in every P05 source: a graded frame before its half's kick-off is
`play_features.PRE_KICKOFF_ALIVE` (0.001), whatever the rules say. It comes from the organisers' documented kick-offs
and no ground truth, only fires where the kick-off lies inside its block, and on BAR-ATH makes the 2,934 graded frames
before the second half's kick-off D (T02).
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import play_features
from pipeline.evidence.play_features import COLUMN

# Params
OUT_M = 0.5                  # ball_outside_m above this: out
STILL_S = 2.0                # ball_still_s at least this: still ...
SPOT_M = 2.0                 # ... at a restart spot,
LINE_M = 1.0                 # ... or on a pitch line
SLOW_SHARE = 0.8             # slow_share_c4 at least this: slow
OWN_HALF_SHARE = 0.95        # own_half_share at least this, and
CENTRE_M = 2.0               # centre_player_m at most this: kick-off


def compute(run, *, alive_out, alive_still, alive_slow, alive_kickoff, alive_none):
    """Evidence(alive = the smallest of the values of the rules that fire, `alive_none` if none; home = NaN)."""
    values = {"alive_out": alive_out, "alive_still": alive_still, "alive_slow": alive_slow,
              "alive_kickoff": alive_kickoff, "alive_none": alive_none}
    values = {name: probability(name, value) for name, value in values.items()}

    inputs = run.inputs
    _, X = play_features.features(inputs)
    column = {name: X[:, index] for name, index in COLUMN.items()}
    with np.errstate(invalid="ignore"):      # NaN compares False: a rule without its inputs does not fire
        out = column["ball_outside_m"] > OUT_M
        still = (column["ball_still_s"] >= STILL_S) & ((column["ball_spot_m"] <= SPOT_M)
                                                       | (np.abs(column["ball_outside_m"]) <= LINE_M)
                                                       | (column["ball_in_goal_area"] == 1.0))
        slow = column["slow_share_c4"] >= SLOW_SHARE
        kickoff = (column["own_half_share"] >= OWN_HALF_SHARE) & (column["centre_player_m"] <= CENTRE_M)

    rules = ((out, "alive_out"), (still, "alive_still"), (slow, "alive_slow"), (kickoff, "alive_kickoff"))
    fired_any = out | still | slow | kickoff
    smallest = np.full(len(X), np.inf)
    for fired, name in rules:
        smallest = np.where(fired, np.minimum(smallest, values[name]), smallest)
    alive = np.where(fired_any, smallest, values["alive_none"])

    alive[play_features.pre_kickoff_mask(inputs)] = play_features.PRE_KICKOFF_ALIVE
    alive[np.isnan(column["ball_missing"])] = np.nan      # rows outside every graded block
    return Evidence(alive=alive, home=no_opinion(len(X)))


def probability(name, value):
    """`value` as a float, or ValueError unless it is a real number in [0, 1] (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 <= value <= 1.0:
        raise ValueError("dead_ball_rules: {} must be a probability, a number in [0, 1], got {!r}".format(name, value))
    return float(value)
