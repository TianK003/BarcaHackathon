"""P19's per-frame features of which team has the ball where the ball is seen, version 2 (P32): seventy-two numbers,
all read from `Inputs`.

P04's touch rule says who has the ball only where one player is close and moves with it, and the nearest player's team
is wrong on 22% of the in-play frames with the ball visible (T14, A02). Between touches the evidence is in how the two
teams stand to the ball and how the ball moves: who is closer, who is closing, who can get there first, who touched it
last and how long ago, which way the ball is going against each team's attack, whether it was kicked, and which way
the teams are shaped. A boosted-tree classifier can read those off the tracking, and the same columns can be built on a
match whose positions have been degraded to look like ours (the benchmark's `generic` profile, P27) and on ours, from
the same `Inputs` both times. `features(inputs)` returns the names and an (n, 72) float64 matrix, row i = frame i, in
this order:

    h_*, a_*    ten columns for each team, from the team's slots and the team's own attacking direction only (so a
                column's value does not depend on the other team's slots, and `swap` is exact):
        ball_d1, ball_d2   metres from the ball to the team's nearest and second-nearest present slot
        closing            the nearest slot s's approach to the ball over 0.2 s either side, m/s: (d_s(t - 5) -
                           d_s(t + 5)) / 0.4 s; positive = approaching
        rel_speed          |v_ball - v_s|, the speed the ball and the nearest slot move apart at
        reach_s            time to reach (Spearman 2018; Shaw's implementation of it): a reaction time of 0.7 s, then a
                           run at the top speed REACH_SPEED, from where the slot will be after the reaction,
                           0.7 + |p + 0.7 v - b| / REACH_SPEED, the minimum over the team's slots with a velocity
        in_episode         1.0 inside one of the team's control episodes: the frames controlled_touch keeps at
                           TOUCH_RULE, by controlled_touch's own code (hold off, confidence 0.99, so kept = non-NaN)
        since_episode_s,   0 inside an episode; else the seconds back to the team's last kept frame in the block, and
        until_episode_s    forward to its next one, each capped at EPISODE_CAP_S (also where there is none), so both
                           saturate at 30 s: an episode 40 s away and no episode at all look the same
        ball_adv           metres the visible ball has advanced towards the goal the team attacks, in the team's own
                           frame exactly as `shape_features` sets it for the team's `adv_mean` in each block: x, or
                           L - x when the team attacks towards x = 0 (0 = the team's own goal line), the block's
                           direction from `play_features.own_half_is_left`, the helper shape_features calls
        ball_vel_adv       the ball's velocity along that direction, m/s, from `kinematics`' velocity of the ball
                           (never the ATD's speed channel): positive when the ball moves towards the goal the team
                           attacks. The two teams attack opposite goals, so a's is minus h's, and a's ball_adv is
                           L - h's; each is still computed from its own team's direction, so that `swap` is exact

    shared      ten columns of the ball alone: speed, acceleration and turning angle over 0.2, 0.5 and 1.0 s either
                side (`ball_speed_02`, ...) and the signed distance beyond the nearest pitch line (`ball_line_m`, minus
                inside)

    P30's 42    `shape_features.features(inputs, 2)`, unchanged (SHAPE_VERSION), so the model that reads the ball also
                reads the shape, and the one-sided changes of the teams' advance that cleared the bar in P30.

What version 2 changed from version 1 (P19's 56 columns, kept in git at commit 15e7e14; only version 2 is built here):
`over11` is gone from each team (the `generic` profile has no wrong-team slot, so the column would be constant in
training); `since_gap_s` and `until_gap_s` are gone from the shared columns (`generic`'s ball gaps are drawn blind to
the play and ours are not, T38, so a model would learn gap timing that does not transfer); `ball_adv` and
`ball_vel_adv` are new in each team block (P24: which way the ball goes against each team's attack); the shape block is
version 2's 42 columns, not version 1's 26.

Every speed, velocity and acceleration comes from positions through `kinematics` (T28, T31): the ATD's speed channels
are never read (test_team_features_ignore_speed_channels puts garbage in them). Nothing reads across a graded block's
edge (T11): a window, an episode, a direction or a closing speed that does not fit in the block is NaN or capped, and
every row outside the blocks is NaN in every column, the episode columns included. Where the ball is missing every
column that needs it is NaN, `ball_adv` and `ball_vel_adv` included; the shape columns and the episode columns are not.

`swap(X)` is the features of the same frames with the two teams relabelled: the two ten-column blocks trade places,
the shared ten stay, and the 42 go through `shape_features.swap`. It doubles a training set with its mirror image and
makes the model's output exactly antisymmetric, as P07's does. Like `shape_features`, it is exact except on a block
where the two teams' mean x are equal to the last bit (the direction then falls to the team labels), which does not
happen with real tracking.

The constants are the design's and are fixed with `FEATURE_VERSION`, never tuned (T26, T37). TOUCH_RULE is
`main.toml`'s touch rule as R013 left it and R030 kept it, frozen here (a check compares the two): a later retune of
the pipeline's rule must not move the model's inputs. `features` remembers its result per Inputs object in a
`weakref.WeakKeyDictionary`: the array is read-only and the memo dies with the Inputs.
"""

import weakref

import numpy as np

from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics, play_features, shape_features

# Params
FEATURE_VERSION = 2
TEAM_COLUMNS = ("ball_d1", "ball_d2", "closing", "rel_speed", "reach_s", "in_episode", "since_episode_s",
                "until_episode_s", "ball_adv", "ball_vel_adv")
SHARED_COLUMNS = ("ball_speed_02", "ball_speed_05", "ball_speed_10", "ball_acc_02", "ball_acc_05", "ball_acc_10",
                  "ball_turn_02", "ball_turn_05", "ball_turn_10", "ball_line_m")
SHAPE_VERSION = 2                       # the shape block: shape_features' version 2 (P30), 42 columns
FEATURE_NAMES = (tuple("h_" + name for name in TEAM_COLUMNS) + tuple("a_" + name for name in TEAM_COLUMNS)
                 + SHARED_COLUMNS + shape_features.feature_names(SHAPE_VERSION))
WINDOWS_S = (0.2, 0.5, 1.0)             # the ball's speed, acceleration and turn, seconds either side
CLOSING_S = 0.2                         # closing: the nearest slot's distance this far before and after
REACTION_S = 0.7                        # reach_s: the time before a slot starts to run
REACH_SPEED = 5.0                       # m/s, reach_s: the slot's top speed (Spearman 2018; Shaw's implementation)
EPISODE_CAP_S = 30.0
# main.toml's touch rule as of R013, kept by R030; duel_radius is written out at controlled_touch's default, so that a
# change of that default cannot move the model's inputs either
TOUCH_RULE = {"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "duel_radius": 1.0}

COLUMN = {name: index for index, name in enumerate(FEATURE_NAMES)}
TEAM_WIDTH = len(TEAM_COLUMNS)
SHARED_START = 2 * TEAM_WIDTH
SHAPE_START = SHARED_START + len(SHARED_COLUMNS)
_MEMO = weakref.WeakKeyDictionary()


def features(inputs):
    """(names, X): FEATURE_NAMES and the (n, 72) float64 feature matrix of `inputs`, read-only, memoised per object."""
    cached = _MEMO.get(inputs)
    if cached is None:
        cached = _MEMO[inputs] = (FEATURE_NAMES, _build(inputs))
    return cached


def swap(X):
    """A new array: the features of the same frames with the two teams relabelled, from the features X (..., 72).

    The two team blocks trade places, the shared columns are copied and the shape columns go through
    `shape_features.swap`. Applied twice it is the identity. ValueError for any other width (version 1's 56 included).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 0 or X.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("team_features.swap: X must have {} columns (FEATURE_NAMES of version {}), got shape {}"
                         .format(len(FEATURE_NAMES), FEATURE_VERSION, X.shape))
    out = np.empty(X.shape)
    out[..., :TEAM_WIDTH] = X[..., TEAM_WIDTH:SHARED_START]
    out[..., TEAM_WIDTH:SHARED_START] = X[..., :TEAM_WIDTH]
    out[..., SHARED_START:SHAPE_START] = X[..., SHARED_START:SHAPE_START]
    out[..., SHAPE_START:] = shape_features.swap(X[..., SHAPE_START:])
    return out


def _build(inputs):
    ball = kinematics.ball_track(inputs)
    players = kinematics.player_track(inputs)
    blocks = kinematics.blocks(inputs)
    teams = np.asarray(inputs.atd.slot_team)
    fps = inputs.fps
    length = inputs.pitch[0]
    n = len(ball.visible)
    X = np.full((n, len(FEATURE_NAMES)), np.nan)
    touch = controlled_touch.compute(RunContext(inputs=inputs, seed=0), **TOUCH_RULE, confidence=0.99, hold=0.5).home
    kept = np.isfinite(touch)
    for start, stop in blocks:
        if stop <= start:
            continue
        out = X[start:stop]                      # a view: the columns written here stay in X
        _ball_columns(out, ball, start, stop, fps, inputs.pitch)
        # the block's direction, by the very call shape_features makes for adv_mean: home defends x < L/2 or not
        home_left = play_features.own_half_is_left(players, teams, start, stop, length)
        for prefix, team, episode in (("h_", 0, kept & (touch >= 0.5)), ("a_", 1, kept & (touch < 0.5))):
            _team_columns(out, players, ball, np.flatnonzero(teams == team), episode[start:stop], start, stop, prefix,
                          fps)
            attacks_right = home_left if team == 0 else not home_left
            _direction_columns(out, ball, start, stop, prefix, attacks_right, length)
    X[:, SHAPE_START:] = shape_features.features(inputs, SHAPE_VERSION)[1]
    X.flags.writeable = False
    return X


def _team_columns(out, players, ball, slots, episode, start, stop, prefix, fps):
    """Columns ball_d1 to until_episode_s of one team in one block, from the team's slots only. `episode` is the
    block's bool mask of the team's kept control frames."""
    column = {name: COLUMN[prefix + name] for name in TEAM_COLUMNS}
    m = stop - start
    rows = slice(start, stop)
    out[:, column["in_episode"]] = episode
    since, until = _since_until(episode, fps, EPISODE_CAP_S)
    out[:, column["since_episode_s"]], out[:, column["until_episode_s"]] = since, until
    if len(slots) == 0:
        return
    present = players.present[rows][:, slots]
    xy, vel = players.xy_m[rows][:, slots], players.vel[rows][:, slots]
    b, b_vel = ball.xy_m[rows], ball.vel[rows]
    distance = np.where(present, kinematics.ball_distances(xy, b), np.nan)        # NaN: absent slot or no ball
    ordered = np.sort(distance, axis=1)                                           # NaN sorts last
    out[:, column["ball_d1"]] = ordered[:, 0]
    if len(slots) > 1:
        out[:, column["ball_d2"]] = ordered[:, 1]

    # the nearest slot s, the lowest index on a tie; frames with no finite distance have none
    nearest = np.argmin(np.where(np.isnan(distance), np.inf, distance), axis=1)
    has = np.isfinite(ordered[:, 0])
    delta = b_vel - vel[np.arange(m), nearest]
    out[:, column["rel_speed"]] = np.where(has, np.hypot(delta[:, 0], delta[:, 1]), np.nan)
    k = int(round(CLOSING_S * fps))
    if m > 2 * k:
        t = np.arange(k, m - k)
        closing = (distance[t - k, nearest[t]] - distance[t + k, nearest[t]]) / (2 * k / fps)
        out[k:m - k, column["closing"]] = np.where(has[t], closing, np.nan)

    # where each slot will be after the reaction time, and the time to cover the rest; a slot with no velocity is out
    ahead = xy + REACTION_S * vel - b[:, None, :]
    left = np.hypot(ahead[..., 0], ahead[..., 1])
    left = np.where(present & np.isfinite(left), left, np.inf).min(axis=1)
    out[:, column["reach_s"]] = np.where(np.isfinite(left), REACTION_S + left / REACH_SPEED, np.nan)


def _direction_columns(out, ball, start, stop, prefix, attacks_right, length):
    """Columns ball_adv and ball_vel_adv of one team in one block, from the team's own attacking direction alone.

    shape_features' rule for a slot's advance, applied to the ball: x when the team attacks towards x = L, else L - x,
    and the velocity along it, vx or 0.0 - vx (0.0 - v, not -v: a zero stays +0.0, as shape_features writes it). NaN
    where the ball is not visible, and where its velocity is NaN (a visible run of one frame)."""
    visible = ball.visible[start:stop]
    x, vx = ball.xy_m[start:stop, 0], ball.vel[start:stop, 0]
    advance = x if attacks_right else length - x
    along = vx if attacks_right else 0.0 - vx
    out[:, COLUMN[prefix + "ball_adv"]] = np.where(visible, advance, np.nan)
    out[:, COLUMN[prefix + "ball_vel_adv"]] = np.where(visible, along, np.nan)


def _ball_columns(out, ball, start, stop, fps, pitch):
    """The ten shared columns of one block, from the ball alone."""
    m = stop - start
    position = ball.xy_m[start:stop]
    for name, seconds in zip(("02", "05", "10"), WINDOWS_S):
        k = int(round(seconds * fps))
        if m <= 2 * k:
            continue
        before, here, after = position[:m - 2 * k], position[k:m - k], position[2 * k:]   # NaN in any is NaN out
        step = np.where(np.isnan(here), np.nan, after - before)   # three positions or none, as acc and turn need
        second = after - 2.0 * here + before
        u, w = here - before, after - here
        out[k:m - k, SHARED_START + SHARED_COLUMNS.index("ball_speed_" + name)] = np.hypot(
            step[:, 0], step[:, 1]) / (2 * k / fps)
        out[k:m - k, SHARED_START + SHARED_COLUMNS.index("ball_acc_" + name)] = np.hypot(
            second[:, 0], second[:, 1]) / (k / fps) ** 2
        # atan2(0, 0) is 0: a ball that did not move before or after turned by nothing
        out[k:m - k, SHARED_START + SHARED_COLUMNS.index("ball_turn_" + name)] = np.arctan2(
            np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]), u[:, 0] * w[:, 0] + u[:, 1] * w[:, 1])
    out[:, SHARED_START + SHARED_COLUMNS.index("ball_line_m")] = kinematics.outside_m(position, pitch)


def _since_until(mask, fps, cap):
    """(since, until): seconds back to the last True frame of `mask` and forward to the next, 0 on a True frame, `cap`
    where there is none and at most `cap`."""
    m = len(mask)
    index = np.arange(m)
    last = np.maximum.accumulate(np.where(mask, index, -1))
    following = np.minimum.accumulate(np.where(mask, index, m)[::-1])[::-1]
    since = np.where(last >= 0, np.minimum((index - last) / fps, cap), cap)
    until = np.where(following < m, np.minimum((following - index) / fps, cap), cap)
    return since, until
