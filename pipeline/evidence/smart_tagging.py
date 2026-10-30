"""P08 -- Smart Tagging as evidence: team votes in `home`, dead-ball votes in `alive`, NaN elsewhere.

    [[evidence]]
    kind = "smart_tagging"
    c_team = 0.8           # confidence in (0.5, 1.0] of a team vote; left out, the source does not vote on the team
    c_dead = 0.9           # confidence in (0.5, 1.0] of a dead-ball vote; left out, it does not vote on the dead ball
    spread = "frame"       # optional: "frame" (the default) or "tag", how a tag's confidence is spent over its frames

Why. The organisers' 638 automatic phase intervals (BUILD UP, PROGRESSION, SET PIECES, ...) are an asset of our match
only and are not possessions (docs/traps.md T17). They are not ground truth either: this source turns them into votes
whose agreement with our own pseudo-labels is measured before any config trusts them (R017), so the confidences are
declared by that measurement and never tuned on a score (T26, T37).

The vote table. A code is read after .strip().upper(), a team value after .strip().lower(), and the `Team` group key
case-insensitively (the codes are upper case and the tag values title case, T17). `N/A` and a missing team give no team
vote.

    TEAM_CODES            the tag's team is the team with the ball: home for Barcelona, away for Atletic Club
    OTHER_TEAM_PATTERN,   the tag's team is the team defending: the ball is the other team's
    OTHER_TEAM_CODES
    DEAD_CODES            the ball is dead; no team vote, whatever the tag's team

Tags overlap, and two phases can name two teams. Two tags that agree are one opinion about the frame, not two: in
"frame" the frame carries c, however many agreeing tags cover it. Two that disagree leave the frame without an
opinion (NaN): the source does not guess which phase is the one that matters. The dead ball is a union.

Cut at the graded blocks. Everything is cut to the half-open graded blocks (inputs.graded), so a tag across a block's
start or a stretch in the halftime gap votes on nothing outside the blocks, and `end_frame` is exclusive.

The two spreads.
* "frame": `home` = c (home has it) or 1 - c (away) on every voted frame, `alive` = 1 - c_dead on a dead frame. A tag
  of 10 s counts 250 times, which makes c a per-frame hit rate, not a likelihood. That is what the decoder's
  per-frame votes already are for every other source.
* "tag": one observation per tag, spread over its n_tag frames (as restart_team's "stoppage_end" does): each frame
  takes sigmoid(sum over the agreeing tags covering it of logit(c) / n_tag), so a tag's log-odds sum over its frames
  back to logit(c) and a long tag does not out-vote a short one. Two agreeing tags on a frame add their shares. A tag
  across both blocks is one tag of its total n_tag. A frame covered by tags that disagree is NaN, as in "frame".

`votes`, `tag_spans` are public: the measurement (R017) and the bootstrap read them without a confidence.
"""

import math
import numbers
import re

import numpy as np

from pipeline.contracts import Evidence, no_opinion

MATCH_ONLY = True      # only our match has Smart Tagging: the benchmark's --drop-match-only reads this (match_only)

# Params
TEAM_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
              "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL")   # the team of the tag has the ball
OTHER_TEAM_PATTERN = r"^DEFENDING IN .+ THIRD$"                            # the team of the tag defends: the ball is
OTHER_TEAM_CODES = ("DEFENSIVE TRANSITION",)                               #   the other team's
DEAD_CODES = ("SET PIECES", "GOALS")                                       # the ball is dead
TEAMS = {"barcelona": "home", "atletic club": "away"}                      # "Atletic Club" (sic), as the tags spell it
SPREADS = ("frame", "tag")

_OTHER_TEAM = re.compile(OTHER_TEAM_PATTERN)
_HOME, _AWAY = 1, 0
NO_VOTE, DISAGREE = -1, -2           # votes' team codes besides 1 (home) and 0 (away)


def compute(run, *, c_team=None, c_dead=None, spread="frame"):
    """Evidence(home = the tags' team vote with confidence c_team, alive = 1 - their dead-ball vote's c_dead; NaN on a
    frame with no vote, with disagreeing team tags and outside the graded blocks; a channel with no c is NaN)."""
    if c_team is None and c_dead is None:
        raise ValueError("smart_tagging: give at least one of c_team and c_dead (a confidence in (0.5, 1.0]), or "
                         "nothing would vote")
    c_team = None if c_team is None else _confidence("c_team", c_team)
    c_dead = None if c_dead is None else _confidence("c_dead", c_dead)
    if spread not in SPREADS:
        raise ValueError("smart_tagging: spread must be \"frame\" or \"tag\", got {!r}".format(spread))
    inputs = run.inputs
    n_frames = inputs.n_frames
    home, alive = no_opinion(n_frames), no_opinion(n_frames)
    if spread == "frame":
        team, dead = votes(inputs)
        if c_team is not None:
            home[team == _HOME] = c_team
            home[team == _AWAY] = 1.0 - c_team
        if c_dead is not None:
            alive[dead] = 1.0 - c_dead
        return Evidence(alive=alive, home=home)
    # "tag": the log-odds each tag spends on a frame add up over its n_tag frames to logit(c), one observation
    towards = {_HOME: np.zeros(n_frames), _AWAY: np.zeros(n_frames), None: np.zeros(n_frames)}
    covered = {_HOME: np.zeros(n_frames, dtype=bool), _AWAY: np.zeros(n_frames, dtype=bool),
               None: np.zeros(n_frames, dtype=bool)}
    for kind, side, pieces in _tags(inputs):
        c = c_team if kind == "team" else c_dead
        if c is None:
            continue
        n_tag = sum(stop - start for start, stop in pieces)
        share = _logit(c) / n_tag
        for start, stop in pieces:
            towards[side][start:stop] += share
            covered[side][start:stop] = True
    if c_team is not None:
        only_home = covered[_HOME] & ~covered[_AWAY]
        only_away = covered[_AWAY] & ~covered[_HOME]
        home[only_home] = _sigmoid(towards[_HOME][only_home])
        home[only_away] = 1.0 - _sigmoid(towards[_AWAY][only_away])
    if c_dead is not None:
        alive[covered[None]] = 1.0 - _sigmoid(towards[None][covered[None]])
    return Evidence(alive=alive, home=home)


def votes(inputs):
    """(team, dead): per-frame arrays from the tags alone. team is int8: 1 home, 0 away, -1 no vote, -2 disagreement;
    dead is bool. Both say nothing (-1, False) outside inputs.graded, with tags cut at the blocks (half-open)."""
    n_frames = inputs.n_frames
    home_votes = np.zeros(n_frames, dtype=np.int32)
    away_votes = np.zeros(n_frames, dtype=np.int32)
    dead_votes = np.zeros(n_frames, dtype=np.int32)
    counts = {_HOME: home_votes, _AWAY: away_votes, None: dead_votes}
    for kind, side, start, stop in tag_spans(inputs):
        counts[side][start:stop] += 1
    team = np.full(n_frames, NO_VOTE, dtype=np.int8)
    team[(home_votes > 0) & (away_votes == 0)] = _HOME
    team[(away_votes > 0) & (home_votes == 0)] = _AWAY
    team[(home_votes > 0) & (away_votes > 0)] = DISAGREE
    return team, dead_votes > 0


def tag_spans(inputs):
    """[(kind, side, start, stop), ...]: every tag that votes, cut at the graded blocks (a tag across both blocks gives
    two spans). kind is "team" or "dead"; side is 1 (home), 0 (away) for a team vote and None for a dead one."""
    return [(kind, side, start, stop) for kind, side, pieces in _tags(inputs) for start, stop in pieces]


def _tags(inputs):
    """[(kind, side, [(start, stop), ...]), ...]: one entry per voting tag, its pieces inside the graded blocks."""
    found = []
    for tag in inputs.smart_tagging:
        kind, side = _vote(tag)
        if kind is None:
            continue
        pieces = []
        for _, block_start, block_stop in inputs.graded:
            start, stop = max(int(tag["start_frame"]), block_start), min(int(tag["end_frame"]), block_stop)
            if start < stop:
                pieces.append((start, stop))
        if pieces:
            found.append((kind, side, pieces))
    return found


def _vote(tag):
    """(kind, side) of one tag: ("dead", None), ("team", 1 or 0), or (None, None) where it says nothing."""
    code = str(tag["code"]).strip().upper()
    if code in DEAD_CODES:
        return "dead", None
    if code in TEAM_CODES:
        other = False
    elif code in OTHER_TEAM_CODES or _OTHER_TEAM.match(code):
        other = True
    else:
        return None, None
    named = None
    for group, value in tag["tags"].items():
        if str(group).strip().lower() == "team":
            named = TEAMS.get(str(value).strip().lower())
    if named is None:
        return None, None
    side = _HOME if named == "home" else _AWAY
    return "team", 1 - side if other else side


def _confidence(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not 0.5 < value <= 1.0:
        raise ValueError("smart_tagging: {} must be a confidence in (0.5, 1.0] (0.5 would say nothing; leave it out "
                         "and the channel does not vote), got {!r}".format(name, value))
    return float(value)


def _logit(c):
    return math.inf if c == 1.0 else math.log(c / (1.0 - c))


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))
