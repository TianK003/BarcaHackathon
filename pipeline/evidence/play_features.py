"""P05's per-frame features of the play: is the ball dead? Thirty numbers a rule or a classifier can read.

A dead ball leaves four kinds of trace in the tracking, and the literature on detecting game interruptions from
positional data (Link and Hoernig 2017; Vidal-Codina et al. 2022; Sanford et al. 2020) works from the same four:

    the ball's own state      beyond a pitch line, motionless, on a restart spot (centre, penalty spot, corner), in a
                              goal area; and what it did just before it went missing and does just after it returns
                              (the tracker loses the ball in 150 gaps of a median 10.7 s, docs/results.md A01, T12);
    the players' speed        a stoppage is a collective slowdown, and so is the walk back after a goal, so the
                              speeds are given over windows both of them can be told apart by;
    the set-piece shapes      a kick-off (everyone in their own half, one pair at the centre spot), a corner (a crowd
                              in a penalty area, someone at a corner point), a goal kick or free kick (players in a
                              goal area, someone beyond the touchline, a wall);
    the teams' shape          the gap between the two centroids and how spread out each side is.

`features(inputs)` returns the names and an (n, 30) float64 matrix, row i = frame i; every row outside the graded
blocks is NaN. The definitions are in FEATURE_NAMES's order and docs in `_ball_columns`/`_player_columns`; all of
them are fixed, from the laws of the game, and never tuned (T26, T37). Every speed is the one `kinematics` derives from
positions, never the ATD's own speed channels (T28, T31). Windows are clipped to the frame's graded block and nothing
reads across half time (T11), and a team's half is decided per block (the teams swap ends, T16): the team whose present
slots have the smaller mean x over the block defends x < L/2. A slot is a tracklet and not a player (T15, T32), so the
counts are of present slots and the shares are of slots too; they are what the tracker gives, not the number of
players.

Missing values. NaN where a column needs a ball and the ball is not visible, or needs a slot or a team and the frame
has none. The counts (`n_present`, `box_crowd`, `goal_area_players`) are 0 on a frame with no slot, not NaN: a count of
nobody is nobody. A visible ball whose speed is NaN (a stretch of one frame, which has no neighbour to difference) is
not "still": it has no still run.

`features` remembers its result per Inputs object in a `weakref.WeakKeyDictionary`, like kinematics: the arrays are
read-only, a second call returns the very same objects, and the memo dies with the Inputs.

`pre_kickoff_mask` and `PRE_KICKOFF_ALIVE` are the other half of this module: a graded frame before its half's
kick-off, which the organisers document and which comes from no ground truth, is dead. On BAR-ATH those are the 2,934
graded frames before the second half's kick-off (T02). It fires only where the kick-off lies inside its block (start <=
kick-off < stop), so a block that starts at the kick-off, as the benchmark's do, has none.
"""

import weakref

import numpy as np

from pipeline.evidence import kinematics

# Params
FEATURE_VERSION = 1
FEATURE_NAMES = (
    "ball_missing", "ball_outside_m", "ball_speed", "ball_still_s", "ball_spot_m", "ball_in_goal_area",
    "since_seen_s", "until_seen_s", "last_outside_m", "last_speed", "next_outside_m", "next_spot_m",
    "n_present", "speed_mean", "speed_p90", "slow_share",
    "speed_mean_past4", "speed_mean_future4", "slow_share_c1", "slow_share_c4", "slow_share_c10", "speed_mean_c10",
    "own_half_share", "centre_player_m", "box_crowd", "corner_player_m", "outside_player_m", "goal_area_players",
    "team_gap_m", "spread_m")
PRE_KICKOFF_ALIVE = 0.001           # P(alive) before a half's kick-off: the stoppage is certain, not infinite
STILL_SPEED = 1.0                   # m/s: the ball counts as still under this
STILL_CAP_S = 30.0                  # seconds: ball_still_s is capped here
SEEN_CAP_S = 60.0                   # seconds: since_seen_s and until_seen_s are capped here
SLOW_SPEED = 2.0                    # m/s: a player is slow under this (a walk is about 1.5)
PENALTY_SPOT_M = 11.0               # from the goal line (laws of the game, law 1)
GOAL_AREA = (5.5, 18.32)            # deep, wide (laws of the game)
PENALTY_AREA = (16.5, 40.32)        # deep, wide
OWN_HALF_TOLERANCE_M = 1.0          # past the halfway line a slot still counts as in its own half
PERCENTILE = 90.0
PAST_S, FUTURE_S = 4.0, 4.0         # the one-sided windows of speed_mean
CENTRED_S = (1.0, 4.0, 10.0)        # the centred windows of slow_share
CENTRED_SPEED_S = 10.0              # the centred window of speed_mean
CHUNK = 1024                       # frames per step of the player columns: the temporaries stay in the cache

COLUMN = {name: index for index, name in enumerate(FEATURE_NAMES)}
_MEMO = weakref.WeakKeyDictionary()


def features(inputs):
    """(names, X): FEATURE_NAMES and the (n, 30) float64 feature matrix of `inputs`, read-only, remembered per object."""
    cached = _MEMO.get(inputs)
    if cached is None:
        cached = _MEMO[inputs] = (FEATURE_NAMES, _build(inputs))
    return cached


def pre_kickoff_mask(inputs):
    """(n,) bool: the graded frames before their half's kick-off, where the kick-off is inside the block."""
    mask = np.zeros(len(inputs.atd.ball_xy), dtype=bool)
    for half, start, stop in inputs.graded:
        kickoff = inputs.kickoffs.get(half)
        if kickoff is not None and start <= kickoff < stop:
            mask[start:int(kickoff)] = True
    return mask


def restart_spots(pitch):
    """(7, 2) metres: the centre spot, the two penalty spots, the four corner points."""
    length, width = pitch
    return np.array([(length / 2, width / 2), (PENALTY_SPOT_M, width / 2), (length - PENALTY_SPOT_M, width / 2),
                     (0.0, 0.0), (0.0, width), (length, 0.0), (length, width)])


def _build(inputs):
    ball, players = kinematics.ball_track(inputs), kinematics.player_track(inputs)
    blocks = kinematics.blocks(inputs)
    fps = inputs.fps
    pitch = inputs.pitch
    X = np.full((len(ball.visible), len(FEATURE_NAMES)), np.nan)
    for start, stop in blocks:
        if stop > start:
            _ball_columns(X[start:stop], ball, start, stop, pitch, fps)
            _player_columns(X[start:stop], players, inputs.atd.slot_team, start, stop, pitch)
    _windows(X, blocks, fps)
    X.flags.writeable = False
    return X


def _ball_columns(out, ball, start, stop, pitch, fps):
    """Columns 1-12 of one block: the ball's state, how long it has been still, how long unseen, and what it was doing
    at the last and the next frame it was seen."""
    length, width = pitch
    seen = ball.visible[start:stop]
    xy = ball.xy_m[start:stop]
    speed = np.where(seen, ball.speed[start:stop], np.nan)
    outside = kinematics.outside_m(xy, pitch)
    spot = _nearest_m(xy, restart_spots(pitch))
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        goal = ((np.abs(y - width / 2) <= GOAL_AREA[1] / 2)
                & (((x >= 0) & (x <= GOAL_AREA[0])) | ((x >= length - GOAL_AREA[0]) & (x <= length))))
        still = seen & (speed < STILL_SPEED)
    index = np.arange(stop - start)

    out[:, COLUMN["ball_missing"]] = ~seen
    out[:, COLUMN["ball_outside_m"]] = outside
    out[:, COLUMN["ball_speed"]] = speed
    out[:, COLUMN["ball_still_s"]] = np.where(
        seen, np.minimum((index - np.maximum.accumulate(np.where(still, -1, index))) / fps, STILL_CAP_S)
        * still, np.nan)
    out[:, COLUMN["ball_spot_m"]] = spot
    out[:, COLUMN["ball_in_goal_area"]] = np.where(seen, goal, np.nan)

    last = np.maximum.accumulate(np.where(seen, index, -1))                  # the last visible frame <= t, or -1
    reversed_seen = seen[::-1]
    following = np.maximum.accumulate(np.where(reversed_seen, index, -1))[::-1]   # counted from the end
    following = np.where(following >= 0, len(index) - 1 - following, -1)      # the next visible frame >= t, or -1
    has_last, has_next = last >= 0, following >= 0
    out[:, COLUMN["since_seen_s"]] = np.where(has_last, np.minimum((index - last) / fps, SEEN_CAP_S), np.nan)
    out[:, COLUMN["until_seen_s"]] = np.where(has_next, np.minimum((following - index) / fps, SEEN_CAP_S), np.nan)
    out[:, COLUMN["last_outside_m"]] = np.where(has_last, outside[last], np.nan)
    out[:, COLUMN["last_speed"]] = np.where(has_last, speed[last], np.nan)
    out[:, COLUMN["next_outside_m"]] = np.where(has_next, outside[following], np.nan)
    out[:, COLUMN["next_spot_m"]] = np.where(has_next, spot[following], np.nan)


def _nearest_m(xy, spots):
    """Distance from every row of xy (m, 2) to the nearest of `spots`; NaN where xy is NaN."""
    nearest = np.full(len(xy), np.inf)
    for spot_x, spot_y in spots:
        nearest = np.minimum(nearest, np.hypot(xy[:, 0] - spot_x, xy[:, 1] - spot_y))
    return np.where(np.isnan(xy[:, 0]), np.nan, nearest)


def _own_half_is_left(players, teams, start, stop, length):
    """Whether the home team (slot team 0) defends x < L/2 in this block: its present slots have the smaller mean x.
    A block where only one team is present compares that team's mean with the halfway line, and one with nobody
    present says True (every column that depends on it is NaN there anyway)."""
    x = players.xy_m[start:stop, :, 0]
    present = players.present[start:stop]
    means = []
    for team in (0, 1):
        mask = present & (teams == team)[None, :]
        means.append(x[mask].mean() if mask.any() else np.nan)
    if not np.isnan(means).any():
        return bool(means[0] < means[1])
    if not np.isnan(means[0]):
        return bool(means[0] < length / 2)
    if not np.isnan(means[1]):
        return bool(means[1] > length / 2)
    return True


def _player_columns(out, players, slot_team, start, stop, pitch):
    """Columns 13-16 and 23-30 of one block, from the present slots of each frame, a chunk of frames at a time."""
    length, width = pitch
    teams = np.asarray(slot_team)
    home_left = _own_half_is_left(players, teams, start, stop, length)
    defends_left = (teams == 0) == home_left                                       # (S,) per slot
    centre = restart_spots(pitch)[0]
    team_masks = [(teams == team).astype(np.float64) for team in (0, 1)]
    for low in range(0, stop - start, CHUNK):
        high = min(low + CHUNK, stop - start)
        rows = slice(low, high)
        present = players.present[start + low:start + high]
        speed = players.speed[start + low:start + high]
        x = np.ascontiguousarray(players.xy_m[start + low:start + high, :, 0])   # NaN exactly where a slot is absent,
        y = np.ascontiguousarray(players.xy_m[start + low:start + high, :, 1])   # so a comparison is False there
        count = present.sum(axis=1)
        out[rows, COLUMN["n_present"]] = count

        # speeds of the present slots that have one (NaN speed: absent, or a stretch of one frame)
        known = ~np.isnan(speed)
        n_known = known.sum(axis=1)
        out[rows, COLUMN["speed_mean"]] = _where_any(n_known, np.where(known, speed, 0.0).sum(axis=1)
                                                     / np.maximum(n_known, 1))
        out[rows, COLUMN["speed_p90"]] = _percentile(speed, n_known)
        with np.errstate(invalid="ignore"):
            slow = (speed < SLOW_SPEED).sum(axis=1)
            out[rows, COLUMN["slow_share"]] = _where_any(n_known, slow / np.maximum(n_known, 1))

            # the half
            own = np.where(defends_left[None, :], x <= length / 2 + OWN_HALF_TOLERANCE_M,
                           x >= length / 2 - OWN_HALF_TOLERANCE_M)
            out[rows, COLUMN["own_half_share"]] = _where_any(count, own.sum(axis=1) / np.maximum(count, 1))

            # set-piece shapes; fmin / fmax ignore the NaN of absent slots and give NaN on a row of nothing but
            out[rows, COLUMN["centre_player_m"]] = np.fmin.reduce(np.hypot(x - centre[0], y - centre[1]), axis=1)
            # the nearest corner point is the nearest in x and in y independently: fold the pitch onto one corner
            out[rows, COLUMN["corner_player_m"]] = np.fmin.reduce(
                np.hypot(np.minimum(x, length - x), np.minimum(y, width - y)), axis=1)
            out[rows, COLUMN["outside_player_m"]] = np.fmax.reduce(kinematics.outside_m(
                np.stack([x, y], axis=-1), pitch), axis=1)
            near_axis = np.abs(y - width / 2)
            in_pitch_x = (x >= 0) & (x <= length)
            in_penalty_y = near_axis <= PENALTY_AREA[1] / 2
            left_box = in_pitch_x & (x <= PENALTY_AREA[0]) & in_penalty_y
            right_box = in_pitch_x & (x >= length - PENALTY_AREA[0]) & in_penalty_y
            goal_area = (in_pitch_x & (near_axis <= GOAL_AREA[1] / 2)
                         & ((x <= GOAL_AREA[0]) | (x >= length - GOAL_AREA[0])))
        out[rows, COLUMN["box_crowd"]] = np.maximum(left_box.sum(axis=1), right_box.sum(axis=1))
        out[rows, COLUMN["goal_area_players"]] = goal_area.sum(axis=1)

        # the teams' shape; sums over a team's slots are products with the team's 0/1 mask
        xz, yz = np.where(present, x, 0.0), np.where(present, y, 0.0)
        presentf = present.astype(np.float64)
        centroids, spreads = [], []
        for mask in team_masks:
            n_mine = presentf @ mask
            safe = np.maximum(n_mine, 1.0)
            cx, cy = (xz @ mask) / safe, (yz @ mask) / safe
            square = np.where(present, (x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2, 0.0) @ mask
            centroids.append((_where_any(n_mine, cx), _where_any(n_mine, cy)))
            spreads.append(_where_any(n_mine, np.sqrt(square / safe)))
        out[rows, COLUMN["team_gap_m"]] = np.hypot(centroids[0][0] - centroids[1][0], centroids[0][1] - centroids[1][1])
        out[rows, COLUMN["spread_m"]] = (spreads[0] + spreads[1]) / 2


def _where_any(counts, values):
    """`values` where `counts` > 0, else NaN."""
    return np.where(counts > 0, values, np.nan)


def _percentile(values, counts):
    """The PERCENTILE of every row of `values` (m, S), NaN-padded, as np.percentile's linear method gives it.
    NaN where the row has no number."""
    ordered = np.sort(values, axis=1)                                         # NaN sorts last
    position = PERCENTILE / 100.0 * np.maximum(counts - 1, 0)
    low = np.floor(position).astype(np.intp)
    high = np.minimum(low + 1, np.maximum(counts - 1, 0))
    rows = np.arange(len(values))
    below, above = ordered[rows, low], ordered[rows, high]
    return np.where(counts > 0, below + (above - below) * (position - low), np.nan)


def _windows(X, blocks, fps):
    """Columns 17-22: window means of speed_mean and slow_share, clipped to the block."""
    def frames(seconds):
        return int(round(seconds * fps))

    speed_mean, slow_share = X[:, COLUMN["speed_mean"]], X[:, COLUMN["slow_share"]]
    mean_known, slow_known = np.isfinite(speed_mean), np.isfinite(slow_share)   # only finite values count
    X[:, COLUMN["speed_mean_past4"]] = kinematics.window_mean(speed_mean, mean_known, blocks, frames(PAST_S), 0)
    X[:, COLUMN["speed_mean_future4"]] = kinematics.window_mean(speed_mean, mean_known, blocks, 0, frames(FUTURE_S))
    for seconds, name in zip(CENTRED_S, ("slow_share_c1", "slow_share_c4", "slow_share_c10")):
        X[:, COLUMN[name]] = kinematics.window_mean(slow_share, slow_known, blocks, frames(seconds), frames(seconds))
    X[:, COLUMN["speed_mean_c10"]] = kinematics.window_mean(
        speed_mean, mean_known, blocks, frames(CENTRED_SPEED_S), frames(CENTRED_SPEED_S))
