"""P17 -- who restarts after a stoppage: home evidence on the first frames after each predicted restart, NaN elsewhere.

    [[evidence]]
    kind = "restart_team"
    model = "pipeline/models/dead_ball_hgb.json"   # the dead-ball model main.toml uses: it says where the stoppages are
    tail = 50              # frames voted from the restart on
    use_ball = false       # optional: read the ball where the setup window sees it (free kicks need it)
    setup = 50             # optional: frames before the restart read for the arrangement (2 s)
    min_stoppage = 25      # optional: the shortest predicted stoppage, in frames (the decoder's min_dead)
    min_prob = 0.001       # optional: dead_ball_model's clip, as main.toml leaves it
    corner = 0.9           # one confidence in (0.5, 1.0] per type that votes; a type left out does not vote. The
    throw_in = 0.8         #   types: kick_off, corner, goal_kick, throw_in, free_kick (free_kick needs use_ball)

Why. The decoder carries the team it had across a stoppage until the first control episode after it names a team, and
seven stoppages in ten hand the ball to the other team (70.3% on the benchmark, docs/results.md A12). The Laws of the
Game decide who restarts, and they also decide how the players stand while the restart is set up; that arrangement is
in the tracking even where the ball is not (our tracker has lost it at 39.8% of the predicted stoppages' ends, A12). A
perfect vote for the restarting team on the 2 s after each true restart is worth +0.0198 pooled raw spellF1 to
main.toml (A12's ceiling). This source is the label-free estimate of that vote: no ground truth, no event data, no
tags; the players' positions, the laws' distances and the dead-ball model's P(alive).

What it does, per graded block (kinematics.blocks; nothing is read or voted across a block's edge)
-------------------------------------------------------------------------------------------------
1. Stoppages. P(alive) is dead_ball_model's, with its pre-kick-off rule, from the file `model` names and with
   `min_prob`. Given the config's own dead-ball source's (main.toml's file, the default clip), this source and the
   decoder agree on where the ball is dead, and since the model's raw scores are remembered per Inputs, the trees are
   walked once for both. A stoppage is a run of P(alive) < ALIVE_LEVEL of at least `min_stoppage` frames,
   and its restart is the frame after it, which must be >= ALIVE_LEVEL. A run that reaches the block's end has no
   restart, nor has one ended by a NaN (nobody knows when play resumed). A run from the block's first frame (the frames
   before the second half's kick-off on BAR-ATH) is a stoppage like any other.
2. The arrangement on [max(block start, restart - setup), restart). Each frame calls at most one (type, team): the first
   rule below that fires, in TYPES's order (`frame_calls`). The stoppage takes the (type, team) called on the most
   frames of the window, if that is at least MIN_WINDOW_SHARE of the window, and nothing otherwise; a tie goes to the
   lower type code, then to home (`arrangement`).
3. The vote. `home` = c (home restarts) or 1 - c (away) on [restart, end), with c the type's confidence and end =
   min(restart + tail, the next stoppage's first frame, the block's end). The next stoppage counts whether or not it has
   a restart of its own: a vote never runs into a stretch the dead-ball model calls dead. A type with no confidence
   says nothing; every other frame is NaN, and so is `alive`, everywhere (`votes`).

The rules, in metres, x along the pitch (0..L), y across it (0..W). Each team's defended end is decided per block by
play_features.own_half_is_left (the teams swap ends at half time, T16).

    kick-off    every present slot within OWN_HALF_TOLERANCE_M of its own half, both teams present, and exactly one
                team with a slot within the centre circle: that team (Law 8: the other team stays 9.15 m away).
    corner      a slot within CORNER_ZONE_M of a corner point, with no opponent within CORNER_CLEARANCE_M of him: his
                team (Law 17: 9.15 m).
    goal kick   at an end, a slot of the team defending it inside its goal area, and at most GOAL_KICK_MAX_INTRUDERS
                opponents inside that penalty area: the defending team (Law 16: opponents outside the penalty area).
    throw-in    a slot within TOUCHLINE_ZONE_M of a touchline or beyond it, with no opponent within
                THROW_IN_CLEARANCE_M of him: his team (Law 15: 2 m).
    free kick   use_ball only: the ball visible and still (play_features.STILL_SPEED), its nearest slot within
                TAKER_BALL_M, and no opponent of his within FREE_KICK_CLEARANCE_M of the ball: his team (Law 13:
                9.15 m).

The opponents' distances are half the law's: players encroach, and the full distance would rarely hold. The zones of
1.5 m and 3 m allow for the tracker's error (the corner arc is 1 m). Every constant is declared from the laws, and none
is tuned (T26, T37). This is the approach of Vidal-Codina et al. (2022), who detect set pieces deterministically from
the players' configuration during dead-ball intervals and find over 90% of them across providers; controlled_touch
cites them for its zones. Free kicks without the ball get no rule: with players alone, the spot could be anywhere.

Two takers. Where several slots meet one rule on a frame, the one nearest the rule's line or spot wins (the touchline,
the corner point, his own goal line, the centre spot), then the lowest slot, so the pick is deterministic. Every rule
needs both teams on the frame: an opponent clearance with no opponent present is vacuous, and a frame without a team is
the tracker's gap, not the game's. A slot is a tracklet (T15, T32): the rules read slot_team as given, and nothing
counts up to eleven.

The ball. Without `use_ball` it is never read. With it, on a frame of the window where the ball is visible, the ball
says *where*, never *what*. Each rule runs only at its own place nearest the ball, and its player is the taker, the slot
nearest the ball (the lowest on a tie), within TAKER_BALL_M, who must meet the rule as above:
    kick-off    if the ball is within KICK_OFF_BALL_M of the centre spot;
    corner      at the corner point nearest the ball, if the ball is within CORNER_ZONE_M of it;
    goal kick   at the end whose goal area, grown by GOAL_AREA_BALL_TOLERANCE_M, holds the ball;
    throw-in    if the ball is within TOUCHLINE_ZONE_M of a touchline or beyond it;
    free kick   at the ball, as above.
So the first four call a type on a ball frame only where the players alone would have called it, and only with the
taker's team. On a frame without the ball the players-only rules apply, and free kicks are never called. The benchmark
tracks the ball out of play and our tracker does not (T39), so a rule that leans on the ball looks better there than it
is here: `use_ball` is a grid axis, not a default.

Why the votes start at the restart (design anti-goal 3). A `home` vote on a frame raises the best alive state's score
against D by up to w * log(2c) nats (0.59 at c = 0.9), so a vote on the stoppage itself would pull the dead-ball edges
in. The source votes only from the dead-ball model's restart on, so where `D` ends stays that model's call (and P02's),
and it stops at the next stoppage. It does not stop at the first control episode: at controlled_touch's 0.93 an episode
out-votes any restart confidence below it anyway, and stopping there would tie this source to that one's parameters.

The confidences are measured per type and per (tail, use_ball) on the benchmark (restarts.py stats, P17), never tuned.
`frame_calls`, `stoppages`, `arrangement`, `calls_from_alive`, `calls` and `votes` are public: the tests drive them
directly, and restarts.py reads `calls`.

Speed. frame_calls takes the (frames, slots, slots) distances a CHUNK of frames at a time, so its memory is bounded for
any window; the source calls it once per stoppage on at most `setup` frames.
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import dead_ball_model, kinematics, play_features, trees

# Params
TYPES = ("kick_off", "corner", "goal_kick", "throw_in", "free_kick")   # priority order; a type's code is its index
KICK_OFF, CORNER, GOAL_KICK, THROW_IN, FREE_KICK = range(len(TYPES))
NO_CALL = -1                         # frame_calls' type and team on a frame where no rule fires
ALIVE_LEVEL = 0.5                    # P(alive) below this is a dead frame (the decoder's own reading of a probability)
MIN_WINDOW_SHARE = 0.25              # a stoppage's (type, team) must be called on this share of its setup window
CENTRE_CIRCLE_M = 9.15               # Law 1's centre circle, Law 8's distance at a kick-off
OWN_HALF_TOLERANCE_M = 1.0           # past the halfway line a slot still counts as in its own half (play_features')
CORNER_ZONE_M = 3.0                  # a corner taker is this close to the corner point (the arc is 1 m; tracking error)
CORNER_CLEARANCE_M = 4.5             # half Law 17's 9.15 m: no opponent this close to the corner taker
GOAL_AREA_DEPTH_M, GOAL_AREA_WIDTH_M = 5.5, 18.32         # Law 1
PENALTY_AREA_DEPTH_M, PENALTY_AREA_WIDTH_M = 16.5, 40.32  # Law 1
GOAL_KICK_MAX_INTRUDERS = 1          # Law 16 sends opponents out of the penalty area; one may still be leaving it
TOUCHLINE_ZONE_M = 1.5               # a thrower is this close to the touchline, or beyond it
THROW_IN_CLEARANCE_M = 1.0           # half Law 15's 2 m: no opponent this close to the thrower
TAKER_BALL_M = 2.0                   # with the ball: the taker is the slot nearest it, this close
FREE_KICK_CLEARANCE_M = 4.5          # half Law 13's 9.15 m: no opponent of the taker this close to the ball
KICK_OFF_BALL_M = 2.0                # with the ball: a kick-off's ball is this close to the centre spot
GOAL_AREA_BALL_TOLERANCE_M = 1.0     # with the ball: a goal kick's ball is in the goal area grown by this much
CHUNK = 256                          # frames per step of frame_calls: a (CHUNK, 46, 46) float64 array is 4.3 MB


def compute(run, *, model, tail, use_ball=False, setup=50, min_stoppage=25, min_prob=0.001, kick_off=None,
            corner=None, goal_kick=None, throw_in=None, free_kick=None):
    """Evidence(home = the restarting team's vote on [restart, end) of every called stoppage whose type has a
    confidence, else NaN; alive = NaN)."""
    given = (kick_off, corner, goal_kick, throw_in, free_kick)          # in TYPES's order
    confidences = {code: _confidence(TYPES[code], value) for code, value in enumerate(given) if value is not None}
    tail, use_ball, setup, min_stoppage = _settings(tail, use_ball, setup, min_stoppage)
    min_prob = trees.check_min_prob(min_prob, "restart_team")
    if not confidences:
        raise ValueError("restart_team: no type has a confidence, so nothing would vote; give at least one of {} a "
                         "confidence in (0.5, 1.0]".format(", ".join(TYPES)))
    if FREE_KICK in confidences and not use_ball:
        raise ValueError("restart_team: free_kick needs use_ball = true (a free kick is called from a still ball "
                         "only); set use_ball, or leave free_kick out")
    found = calls(run, model=model, tail=tail, use_ball=use_ball, setup=setup, min_stoppage=min_stoppage,
                  min_prob=min_prob)
    n_frames = run.inputs.atd.n_frames
    return Evidence(alive=no_opinion(n_frames), home=votes(found, n_frames, confidences))


def calls(run, *, model, tail, use_ball=False, setup=50, min_stoppage=25, min_prob=0.001):
    """calls_from_alive on dead_ball_model's P(alive) (`model`, `min_prob`, the pre-kick-off rule included)."""
    tail, use_ball, setup, min_stoppage = _settings(tail, use_ball, setup, min_stoppage)
    min_prob = trees.check_min_prob(min_prob, "restart_team")
    alive = dead_ball_model.compute(run, model=model, min_prob=min_prob).alive
    return calls_from_alive(run.inputs, alive, tail=tail, use_ball=use_ball, setup=setup, min_stoppage=min_stoppage)


def calls_from_alive(inputs, alive, *, tail, use_ball, setup, min_stoppage):
    """One call per stoppage with an arrangement, from any (n,) P(alive): a dict of read-only arrays in time order,
    `restart` and `end` (int64: the vote's [restart, end)), `type` (int8, a code of TYPES) and `team` (int8, 0 home,
    1 away)."""
    tail, use_ball, setup, min_stoppage = _settings(tail, use_ball, setup, min_stoppage)
    n_frames = inputs.atd.n_frames
    alive = np.asarray(alive, dtype=np.float64)
    if alive.shape != (n_frames,):
        raise ValueError("restart_team: alive must hold one P(alive) per frame, shape ({},), got {}".format(
            n_frames, alive.shape))
    players, ball = kinematics.player_track(inputs), kinematics.ball_track(inputs)
    teams = np.asarray(inputs.atd.slot_team)
    found = {"restart": [], "end": [], "type": [], "team": []}
    for start, stop in kinematics.blocks(inputs):
        if stop <= start:
            continue
        home_left = play_features.own_half_is_left(players, teams, start, stop, inputs.pitch[0])
        runs = _dead_runs(alive, start, stop, min_stoppage)
        for index, (first, restart) in enumerate(runs):
            if restart is None:
                continue
            window = slice(max(start, restart - setup), restart)
            types, named = frame_calls(players.xy_m[window], players.present[window], teams, ball.xy_m[window],
                                       ball.visible[window], ball.speed[window], inputs.pitch, home_left, use_ball)
            pair = arrangement(types, named, MIN_WINDOW_SHARE)
            if pair is None:
                continue
            following = runs[index + 1][0] if index + 1 < len(runs) else stop   # the next stoppage, restart or not
            found["restart"].append(restart)
            found["end"].append(min(restart + tail, following, stop))
            found["type"].append(pair[0])
            found["team"].append(pair[1])
    dtypes = {"restart": np.int64, "end": np.int64, "type": np.int8, "team": np.int8}
    return {key: _frozen(np.array(values, dtype=dtypes[key])) for key, values in found.items()}


def votes(calls, n_frames, confidences):
    """(n_frames,) float64 home: c (home restarts) or 1 - c on each call's [restart, end), with c = confidences[its
    type code]; NaN on every other frame and for a type with no confidence."""
    checked = {}
    for code, value in dict(confidences).items():
        if isinstance(code, (bool, np.bool_)) or not isinstance(code, numbers.Integral) or not 0 <= code < len(TYPES):
            raise ValueError("restart_team: confidences maps a type code (0 to {}, in TYPES's order) to a confidence, "
                             "got the key {!r}".format(len(TYPES) - 1, code))
        checked[int(code)] = _confidence("the confidence of " + TYPES[int(code)], value)
    home = no_opinion(n_frames)
    for restart, end, kind, team in zip(calls["restart"].tolist(), calls["end"].tolist(), calls["type"].tolist(),
                                        calls["team"].tolist()):
        if kind in checked:
            home[restart:end] = checked[kind] if team == 0 else 1.0 - checked[kind]
    return home


def stoppages(alive, start, stop, min_stoppage):
    """[(first, restart), ...]: every run of P(alive) < ALIVE_LEVEL inside [start, stop) of at least `min_stoppage`
    frames whose restart, the frame after it, is inside [start, stop) and >= ALIVE_LEVEL. NaN is not dead, so it ends
    a run, and is no restart."""
    return [(first, restart) for first, restart in _dead_runs(alive, start, stop, min_stoppage) if restart is not None]


def _dead_runs(alive, start, stop, min_stoppage):
    """[(first, restart or None), ...]: every run of at least min_stoppage dead frames in [start, stop), with its
    restart where it has one. The vote's end needs the runs without one too."""
    min_stoppage = _whole("min_stoppage", min_stoppage, "frames: the shortest stoppage")
    alive = np.asarray(alive, dtype=np.float64)
    if alive.ndim != 1 or not 0 <= start <= stop <= len(alive):
        raise ValueError("restart_team: alive must be 1-D and [start, stop) inside it, got shape {} and [{}, {})"
                         .format(alive.shape, start, stop))
    segment = alive[start:stop]
    with np.errstate(invalid="ignore"):                  # NaN compares False: neither dead nor a restart
        dead = segment < ALIVE_LEVEL
        up = segment >= ALIVE_LEVEL
    found = []
    for first, last in zip(*(edges.tolist() for edges in kinematics.runs(dead))):
        if last - first >= min_stoppage:
            found.append((start + first, start + last if last < len(segment) and up[last] else None))
    return found


def arrangement(types, teams, min_share):
    """(type, team): the pair called on the most frames, if that is at least min_share of all len(types) frames; None
    otherwise, and on a window with no call. A tie goes to the lower type code (the priority), then to team 0."""
    types, teams = np.asarray(types), np.asarray(teams)
    if types.shape != teams.shape or types.ndim != 1:
        raise ValueError("restart_team: types and teams must be 1-D and of one shape, got {} and {}".format(
            types.shape, teams.shape))
    called = types != NO_CALL
    if not called.any():
        return None
    counts = np.bincount(types[called].astype(np.int64) * 2 + teams[called].astype(np.int64),
                         minlength=2 * len(TYPES))
    best = int(np.argmax(counts))                      # the first of the largest: the lower type, then home
    if counts[best] < min_share * len(types):
        return None
    return best // 2, best % 2


def frame_calls(players_xy_m, present, slot_team, ball_xy_m, ball_visible, ball_speed, pitch, home_defends_left,
                use_ball):
    """(types, teams): the (type, team) each frame calls by the module's rules, two int8 (n,) arrays with NO_CALL (-1)
    where no rule fires. The frames of one block (one direction of play): players_xy_m (n, S, 2) metres, present (n, S),
    slot_team (S,) 0 home or 1 away, ball_xy_m (n, 2) metres, ball_visible (n,), ball_speed (n,) m/s, pitch (L, W)
    metres, home_defends_left whether home defends x < L/2, use_ball whether a visible ball is read. Pure: it reads
    only its arguments and writes none of them."""
    xy = np.asarray(players_xy_m, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    teams = np.asarray(slot_team)
    ball_xy = np.asarray(ball_xy_m, dtype=np.float64)
    visible = np.asarray(ball_visible, dtype=bool)
    speed = np.asarray(ball_speed, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[2] != 2 or present.shape != xy.shape[:2] or teams.shape != xy.shape[1:2] or (
            ball_xy.shape != (len(xy), 2) or visible.shape != (len(xy),) or speed.shape != (len(xy),)):
        raise ValueError("restart_team: frame_calls needs players (n, S, 2), present (n, S), slot_team (S,), ball "
                         "(n, 2), ball_visible (n,) and ball_speed (n,), got {}, {}, {}, {}, {} and {}".format(
                             xy.shape, present.shape, teams.shape, ball_xy.shape, visible.shape, speed.shape))
    if not np.isin(teams, (0, 1)).all():
        raise ValueError("restart_team: slot_team must be 0 (home) or 1 (away) for every slot")
    types, named = np.full(len(xy), NO_CALL, dtype=np.int8), np.full(len(xy), NO_CALL, dtype=np.int8)
    if xy.shape[1] == 0:
        return types, named                            # no slot, no call (and no argmin over an empty axis)
    for low in range(0, len(xy), CHUNK):
        high = min(low + CHUNK, len(xy))
        types[low:high], named[low:high] = _chunk_calls(xy[low:high], present[low:high], teams, ball_xy[low:high],
                                                        visible[low:high], speed[low:high], pitch,
                                                        bool(home_defends_left), bool(use_ball))
    return types, named


def _chunk_calls(xy, present, teams, ball_xy, visible, speed, pitch, home_left, use_ball):
    """frame_calls on a chunk of frames. Each rule gives, per frame and slot, whether the slot meets it and how far he
    is from its line or spot (the rank), and per frame whether the taker meets it at the ball's place; the frame takes
    the first rule that fires."""
    length, width = pitch
    m = len(xy)
    rows = np.arange(m)
    present = present & np.isfinite(xy).all(axis=2)       # a slot without a position is not there
    x = np.where(present, xy[..., 0], np.nan)              # NaN exactly where a slot is absent, so every comparison
    y = np.where(present, xy[..., 1], np.nan)              # with it below is False
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    home = teams == 0
    both = (present & home).any(axis=1) & (present & ~home).any(axis=1)        # every rule needs both teams
    ball_mode = (visible & np.isfinite(ball_xy).all(axis=1)) if use_ball else np.zeros(m, dtype=bool)

    with np.errstate(invalid="ignore"):
        # Every slot's nearest opponent (inf with none present), and the taker: the slot nearest the ball, within
        # TAKER_BALL_M (np.argmin keeps the lowest slot on a tie).
        apart = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        opponents = (teams[:, None] != teams[None, :])[None] & ~np.isnan(apart)
        rival = np.where(opponents, apart, np.inf).min(axis=2)
        to_ball = np.hypot(x - bx[:, None], y - by[:, None])
        to_ball = np.where(np.isnan(to_ball), np.inf, to_ball)
        taker = np.argmin(to_ball, axis=1)
        has_taker = to_ball[rows, taker] <= TAKER_BALL_M

        rules = []   # per type in TYPES's order: (meets (m, S), rank (m, S), the taker meets it at the ball (m,))

        # kick-off: everyone in their own half, both teams there, one team in the centre circle
        defends_left = home == home_left
        own = np.where(defends_left[None, :], x <= length / 2 + OWN_HALF_TOLERANCE_M,
                       x >= length / 2 - OWN_HALF_TOLERANCE_M)
        centre = np.hypot(x - length / 2, y - width / 2)
        circle = present & (centre <= CENTRE_CIRCLE_M)
        formation = (own | ~present).all(axis=1) & ((circle & home).any(axis=1) != (circle & ~home).any(axis=1))
        meets = circle & formation[:, None]
        rules.append((meets, centre,
                      (np.hypot(bx - length / 2, by - width / 2) <= KICK_OFF_BALL_M) & meets[rows, taker]))

        # corner: the nearest corner point is the nearest in x and in y apart, so fold the pitch onto one corner. With
        # the ball, the taker (within 2 m of a ball within 3 m of a corner point) has that corner point as his nearest.
        corner = np.hypot(np.minimum(x, length - x), np.minimum(y, width - y))
        meets = present & (corner <= CORNER_ZONE_M) & (rival > CORNER_CLEARANCE_M)
        ball_corner = np.hypot(np.minimum(bx, length - bx), np.minimum(by, width - by))
        rules.append((meets, corner, (ball_corner <= CORNER_ZONE_M) & meets[rows, taker]))

        # goal kick: each end in turn, measured from its goal line into the pitch
        axis = np.abs(y - width / 2)
        meets, rank = np.zeros((m, len(teams)), dtype=bool), np.full((m, len(teams)), np.inf)
        at_ball = np.zeros(m, dtype=bool)
        for left in (True, False):
            depth, ball_depth = (x, bx) if left else (length - x, length - bx)
            defender = teams == (0 if left == home_left else 1)
            box = (depth >= 0) & (depth <= PENALTY_AREA_DEPTH_M) & (axis <= PENALTY_AREA_WIDTH_M / 2)
            intruders = (present & box & ~defender[None, :]).sum(axis=1)
            keeper = (present & defender[None, :] & (depth >= 0) & (depth <= GOAL_AREA_DEPTH_M)
                      & (axis <= GOAL_AREA_WIDTH_M / 2) & (intruders <= GOAL_KICK_MAX_INTRUDERS)[:, None])
            ball_in = np.hypot(_beyond(ball_depth, 0.0, GOAL_AREA_DEPTH_M),
                               _beyond(np.abs(by - width / 2), 0.0, GOAL_AREA_WIDTH_M / 2))
            ball_in = ball_in <= GOAL_AREA_BALL_TOLERANCE_M         # the goal area grown by the tolerance holds it
            meets |= keeper
            rank = np.where(keeper, depth, rank)
            at_ball |= ball_in & keeper[rows, taker]
        rules.append((meets, rank, at_ball))

        # throw-in: the distance to the touchline, beyond it or inside
        line = np.minimum(np.abs(y), np.abs(width - y))
        meets = present & ((y <= TOUCHLINE_ZONE_M) | (y >= width - TOUCHLINE_ZONE_M)) & (rival > THROW_IN_CLEARANCE_M)
        rules.append((meets, line, ((by <= TOUCHLINE_ZONE_M) | (by >= width - TOUCHLINE_ZONE_M)) & meets[rows, taker]))

        # free kick: the ball alone, still, with no opponent of the taker near it; no players-only rule
        opponent_of_taker = present & (teams[None, :] != teams[taker][:, None])
        clear = np.where(opponent_of_taker, to_ball, np.inf).min(axis=1) > FREE_KICK_CLEARANCE_M
        rules.append((np.zeros((m, len(teams)), dtype=bool), np.full((m, len(teams)), np.inf),
                      (speed < play_features.STILL_SPEED) & clear))

    types, named = np.full(m, NO_CALL, dtype=np.int8), np.full(m, NO_CALL, dtype=np.int8)
    open_frames = both.copy()
    for code, (meets, rank, at_ball) in enumerate(rules):
        fires = open_frames & np.where(ball_mode, has_taker & at_ball, meets.any(axis=1))
        slot = np.where(ball_mode, taker, np.argmin(np.where(meets, rank, np.inf), axis=1))   # the nearest, then lowest
        types[fires] = code
        named[fires] = teams[slot[fires]]
        open_frames &= ~fires
    return types, named


def _beyond(value, low, high):
    """How far `value` lies outside [low, high]: 0 inside, NaN where it is NaN."""
    return np.maximum(np.maximum(low - value, value - high), 0.0)


def _settings(tail, use_ball, setup, min_stoppage):
    tail = _whole("tail", tail, "frames voted from the restart on")
    setup = _whole("setup", setup, "frames before the restart read for the arrangement")
    min_stoppage = _whole("min_stoppage", min_stoppage, "frames: the shortest stoppage")
    if not isinstance(use_ball, (bool, np.bool_)):
        raise ValueError("restart_team: use_ball must be true or false, got {!r}".format(use_ball))
    return tail, bool(use_ball), setup, min_stoppage


def _whole(name, value, unit):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError("restart_team: {} must be a whole number >= 1 ({}), got {!r}".format(name, unit, value))
    return int(value)


def _confidence(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not 0.5 < value <= 1.0:
        raise ValueError("restart_team: {} must be a confidence in (0.5, 1.0] (0.5 would say nothing; leave the type "
                         "out and it does not vote), got {!r}".format(name, value))
    return float(value)


def _frozen(array):
    array.flags.writeable = False
    return array
