"""P07's per-frame features of the two teams' shape and movement, none of which reads the ball. Twenty-six numbers
(version 1), or forty-two with P30's version 2.

The ball is missing from the tracking on 41% of the graded frames, in long gaps (a median 10.7 s, docs/results.md A01,
T12), and those are the frames nothing else says who has it on. A team in possession is more advanced, wider, and
moving up the pitch together; a team without it sits deeper and shifts to follow the ball. A classifier can read those
off the players alone, so every column here comes from the players and none from `atd.ball_*`: that is what lets it be
trained on matches whose ball has been hidden and be asked about the gaps of ours, from the same inputs both times
(test_shape_features_ignore_the_ball puts a ball in and sees nothing change).

`features(inputs)` returns the names and an (n, 26) float64 matrix, row i = frame i: twelve columns for the home team
(`h_*`), the same twelve for the away team (`a_*`), then `adv_gap` and `n_diff`.

    n                       present slots of the team
    adv_mean, adv_p10,      metres advanced towards the opponent's goal line (0 = the team's own goal line), of the
    adv_p90                 present slots: their mean, and their 10th and 90th percentiles (numpy's linear rule) -
                            the middle of the team and the lines it holds
    width                   p90 - p10 of the slots' y
    vel_adv                 mean velocity along the attacking direction, m/s, over the slots that have a velocity
    speed_mean              mean speed, m/s, over the slots that have one
    att_third, own_third    share of the present slots in the attacking third (adv > 2L/3) and in the defending
                            third (adv < L/3)
    d_adv_1s, d_adv_2s      adv_mean one or two seconds ahead minus the same behind: the team moving up or dropping
                            back, as a displacement and not as a velocity
    vel_adv_1s              vel_adv averaged over the second either side

    adv_gap                 h_adv_mean - a_adv_mean: who is further up the pitch
    n_diff                  h_n - a_n

Version 2 (P30) is version 1's matrix, its columns copied and not recomputed (a model trained on version 1 keeps
reading what it was trained on), with eight more columns per team after the twelve: twenty `h_*`, twenty `a_*`, then
`adv_gap` and `n_diff` (42).

    length                  adv_p90 - adv_p10: how stretched the team is between its lines
    spread                  mean distance, in metres, of the present slots to their centroid (x and y both)
    adv_past_2s, _5s, _10s  adv_mean now minus adv_mean k seconds earlier: the team's advance over the last k seconds
    adv_fut_2s, _5s, _10s   adv_mean k seconds later minus adv_mean now: its advance over the next k seconds

The one-sided changes are windows of the same block (nothing reads across half time) and NaN where either end is NaN,
a team's absence included; like `d_adv_ks` they are the trees' missing value there. `features(inputs, version)` takes
the version (1 by default) and `feature_names(version)` and `columns(version)` give its names and indices.

Versions 3 (P33, git history 411b881), 4 (P37, git history 963861b) and 5 (P40, git history 656fba3) were dropped at
their gates and are retired: they are never reused for another matrix (the next recipe is 6), and `feature_names`,
`features` and the model loader refuse them.

Each team is seen in its own frame. A team's `adv` is x or L - x, whichever points at the goal it attacks, so no column
carries "home is on the left" or "home is Barcelona": the classifier cannot learn a home bias from the geometry, and
relabelling the teams gives the features of the same frames with the two blocks exchanged. `swap` does that exchange,
and it is exact (the two teams' columns are computed by the same code from the same slots in the same order). It
doubles a training set with its mirror image at no cost and makes the model's output exactly antisymmetric. Each
block's direction comes from `play_features.own_half_is_left` (the teams swap ends at half time, T16): the team whose
present slots have the smaller mean x over the block defends x < L/2. The one thing `swap` cannot be exact on is a
block where the two teams' mean x are equal to the last bit: neither side is then the one on the left, and the
direction falls to the team labels. That does not happen with real tracking.

A slot is a tracklet and not a player: its team is the slot's, and after an identity switch a team can show more than
11 slots while the other shows fewer, on 10.8% of the graded frames (T32, A07). One wrong slot moves an extreme (the
deepest or the most advanced slot, the narrowest width) as far as it likes and a mean or a quantile by a tenth of
that, so the columns are means, 10th and 90th percentiles and shares, never a minimum or a maximum. The counts are of
present slots, which is what the tracker gives.

Missing values. Rows outside the graded blocks are NaN in every column, `n` and `n_diff` included. Inside a block, a
team with no present slot on a frame has `n = 0` and NaN in all its other columns, the windowed ones too (a team that
is gone for a second has no displacement of its own); `adv_gap` is then NaN and `n_diff` is not. A slot's velocity is
NaN on a stretch of one frame, so `vel_adv` and `speed_mean` are NaN on a frame where no present slot of the team has
one. Windows are clipped to the frame's block and nothing reads across half time (T11): `d_adv_ks` is NaN within k
seconds of a block's edge, and where either end of it is NaN. Speeds and velocities are the ones `kinematics` derives
from positions (T28, T31).

All of it is fixed, from the laws of the game and the shape of a pitch, and never tuned (T26, T37). `features` remembers
its result per Inputs object in a `weakref.WeakKeyDictionary`, like `play_features`: the array is read-only, a second
call returns the very same objects, and the memo dies with the Inputs.
"""

import weakref

import numpy as np

from pipeline.evidence import kinematics, play_features

# Params
FEATURE_VERSION = 1
VERSIONS = (1, 2)                                         # 3 (P33, 411b881), 4 (P37, 963861b) and 5 (P40, 656fba3)
                                                          # are retired and never reused; refused like any other
TEAM_COLUMNS = ("n", "adv_mean", "adv_p10", "adv_p90", "width", "vel_adv", "speed_mean", "att_third", "own_third",
                "d_adv_1s", "d_adv_2s", "vel_adv_1s")
CROSS_COLUMNS = ("adv_gap", "n_diff")
FEATURE_NAMES = (tuple("h_" + name for name in TEAM_COLUMNS) + tuple("a_" + name for name in TEAM_COLUMNS)
                 + CROSS_COLUMNS)
ONE_SIDED_S = (2.0, 5.0, 10.0)                            # version 2: seconds behind and ahead of the frame
TEAM_COLUMNS_V2 = TEAM_COLUMNS + ("length", "spread") + tuple(
    "adv_{}_{}s".format(side, int(seconds)) for side in ("past", "fut") for seconds in ONE_SIDED_S)
FEATURE_NAMES_V2 = (tuple("h_" + name for name in TEAM_COLUMNS_V2) + tuple("a_" + name for name in TEAM_COLUMNS_V2)
                    + CROSS_COLUMNS)
LOW_PERCENTILE, HIGH_PERCENTILE = 10.0, 90.0
DISPLACEMENT_S = (("d_adv_1s", 1.0), ("d_adv_2s", 2.0))   # seconds either side of the frame
VELOCITY_WINDOW_S = 1.0                                   # vel_adv_1s: seconds either side of the frame
CHUNK = 1024                                              # frames per step of the slot columns, for the cache

COLUMN = {name: index for index, name in enumerate(FEATURE_NAMES)}
TEAM_WIDTH = len(TEAM_COLUMNS)
_NAMES = {1: FEATURE_NAMES, 2: FEATURE_NAMES_V2}
_BLOCK_OF_WIDTH = {len(_NAMES[1]): TEAM_WIDTH, len(_NAMES[2]): len(TEAM_COLUMNS_V2)}   # matrix width -> a team's block
_MEMO = weakref.WeakKeyDictionary()                       # Inputs -> {version: (names, X)}


def feature_names(version):
    """The column names of a feature version (1: FEATURE_NAMES, 2: FEATURE_NAMES_V2); refuses any other version, 3, 4 and 5
    (retired) included."""
    if type(version) is not int or version not in VERSIONS:      # not True or 2.0, which `in` would let through
        raise ValueError("shape_features: version must be one of {} (3, 4 and 5 are retired), got {!r}".format(
            VERSIONS, version))
    return _NAMES[version]


def columns(version):
    """{name: column index} of a feature version; refuses any other version."""
    return {name: index for index, name in enumerate(feature_names(version))}


def features(inputs, version=1):
    """(names, X): the names of `version` and its (n, 26) or (n, 42) float64 feature matrix of `inputs`, read-only,
    memoised per object and version. Version 2 holds version 1's columns, unchanged. Refuses any other version."""
    feature_names(version)
    memo = _MEMO.setdefault(inputs, {})
    if version not in memo:
        if 1 not in memo:
            memo[1] = (FEATURE_NAMES, _build(inputs))
        if version == 2:
            memo[2] = (FEATURE_NAMES_V2, _build_v2(inputs, memo[1][1]))
    return memo[version]


def swap(X):
    """A new array: the features of the same frames with the two teams relabelled, from the features X (..., 26) of
    version 1 or (..., 42) of version 2; refuses any other width.

    The `h_*` and `a_*` blocks trade places and the two cross columns change sign. Applied twice it is the identity.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 0 or X.shape[-1] not in _BLOCK_OF_WIDTH:
        raise ValueError("shape_features.swap: X must have {} columns (a feature version's width), got shape {}".format(
            " or ".join(str(width) for width in _BLOCK_OF_WIDTH), X.shape))
    block = _BLOCK_OF_WIDTH[X.shape[-1]]
    out = np.empty(X.shape)
    out[..., :block] = X[..., block:2 * block]
    out[..., block:2 * block] = X[..., :block]
    out[..., 2 * block:] = 0.0 - X[..., 2 * block:]    # 0.0 - x, not -x: a zero stays +0.0, as a - h gives
    return out


def _build(inputs):
    players = kinematics.player_track(inputs)
    blocks = kinematics.blocks(inputs)
    length = inputs.pitch[0]
    teams = np.asarray(inputs.atd.slot_team)
    X = np.full((len(players.present), len(FEATURE_NAMES)), np.nan)
    for start, stop in blocks:
        if stop > start:
            home_left = play_features.own_half_is_left(players, teams, start, stop, length)
            for team, prefix in ((0, "h_"), (1, "a_")):
                attacks_right = home_left if team == 0 else not home_left
                _team_columns(X[start:stop], players, np.flatnonzero(teams == team), start, stop, prefix,
                              attacks_right, length)
    for prefix in ("h_", "a_"):
        _window_columns(X, blocks, prefix, inputs.fps)
    X[:, COLUMN["adv_gap"]] = X[:, COLUMN["h_adv_mean"]] - X[:, COLUMN["a_adv_mean"]]
    X[:, COLUMN["n_diff"]] = X[:, COLUMN["h_n"]] - X[:, COLUMN["a_n"]]
    X.flags.writeable = False
    return X


def _build_v2(inputs, X1):
    """The (n, 42) matrix of version 2: X1's columns copied into their places and the new ones filled in, read-only."""
    players = kinematics.player_track(inputs)
    blocks = kinematics.blocks(inputs)
    teams = np.asarray(inputs.atd.slot_team)
    column = columns(2)
    X = np.full((len(X1), len(FEATURE_NAMES_V2)), np.nan)
    for index, name in enumerate(FEATURE_NAMES):
        X[:, column[name]] = X1[:, index]
    for team, prefix in ((0, "h_"), (1, "a_")):
        adv_mean = X[:, column[prefix + "adv_mean"]]
        X[:, column[prefix + "length"]] = X[:, column[prefix + "adv_p90"]] - X[:, column[prefix + "adv_p10"]]
        for start, stop in blocks:
            if stop > start:
                _spread_column(X[start:stop, column[prefix + "spread"]], players, np.flatnonzero(teams == team), start)
            for seconds in ONE_SIDED_S:
                k = int(round(seconds * inputs.fps))
                if stop - start > k:
                    change = adv_mean[start + k:stop] - adv_mean[start:stop - k]    # NaN where either end is
                    X[start + k:stop, column["{}adv_past_{}s".format(prefix, int(seconds))]] = change
                    X[start:stop - k, column["{}adv_fut_{}s".format(prefix, int(seconds))]] = change
    X.flags.writeable = False
    return X


def _spread_column(out, players, slots, start):
    """Fills `out` (one block, a view) with the mean distance of the team's present slots to their centroid, NaN with
    no slot, a chunk of frames at a time over contiguous copies of the team's columns, as `_team_columns` does."""
    if len(slots) == 0:
        return
    for low in range(0, len(out), CHUNK):
        high = min(low + CHUNK, len(out))
        present = np.ascontiguousarray(players.present[start + low:start + high][:, slots])
        x = np.ascontiguousarray(players.xy_m[start + low:start + high][:, slots, 0])
        y = np.ascontiguousarray(players.xy_m[start + low:start + high][:, slots, 1])
        count = present.sum(axis=1)
        safe = np.maximum(count, 1)
        centre_x = np.where(present, x, 0.0).sum(axis=1) / safe
        centre_y = np.where(present, y, 0.0).sum(axis=1) / safe
        with np.errstate(invalid="ignore"):
            distance = np.hypot(x - centre_x[:, None], y - centre_y[:, None])
        out[low:high] = _where_any(count, np.where(present, distance, 0.0).sum(axis=1) / safe)


def _team_columns(out, players, slots, start, stop, prefix, attacks_right, length):
    """Columns n to own_third of one team in one block, from the team's present slots, a chunk of frames at a time.
    `slots` are the team's slot indices, in slot order; the sums run over them only, so what a team's columns are made
    of does not depend on the other team."""
    column = {name: COLUMN[prefix + name] for name in TEAM_COLUMNS}
    if len(slots) == 0:
        out[:, column["n"]] = 0.0                    # a team with no slot at all is never present
        return
    for low in range(0, stop - start, CHUNK):
        high = min(low + CHUNK, stop - start)
        rows = slice(low, high)
        # contiguous copies of the team's columns, so that a row's sum does not depend on how many rows are in the step
        present = np.ascontiguousarray(players.present[start + low:start + high][:, slots])
        x = np.ascontiguousarray(players.xy_m[start + low:start + high][:, slots, 0])
        y = np.ascontiguousarray(players.xy_m[start + low:start + high][:, slots, 1])
        speed = np.ascontiguousarray(players.speed[start + low:start + high][:, slots])
        vx = np.ascontiguousarray(players.vel[start + low:start + high][:, slots, 0])

        count = present.sum(axis=1)
        out[rows, column["n"]] = count
        safe = np.maximum(count, 1)
        adv = np.where(present, x if attacks_right else length - x, np.nan)     # NaN exactly where a slot is absent
        out[rows, column["adv_mean"]] = _where_any(count, np.where(present, adv, 0.0).sum(axis=1) / safe)
        out[rows, column["adv_p10"]], out[rows, column["adv_p90"]] = _percentiles(
            adv, count, (LOW_PERCENTILE, HIGH_PERCENTILE))
        low_y, high_y = _percentiles(np.where(present, y, np.nan), count, (LOW_PERCENTILE, HIGH_PERCENTILE))
        out[rows, column["width"]] = high_y - low_y

        # a slot has a velocity unless its run is one frame long (then its speed and both components are NaN)
        known = present & ~np.isnan(speed)
        n_known = known.sum(axis=1)
        safe_known = np.maximum(n_known, 1)
        along = vx if attacks_right else 0.0 - vx
        out[rows, column["vel_adv"]] = _where_any(n_known, np.where(known, along, 0.0).sum(axis=1) / safe_known)
        out[rows, column["speed_mean"]] = _where_any(n_known, np.where(known, speed, 0.0).sum(axis=1) / safe_known)

        with np.errstate(invalid="ignore"):
            out[rows, column["att_third"]] = _where_any(count, (adv > 2.0 * length / 3.0).sum(axis=1) / safe)
            out[rows, column["own_third"]] = _where_any(count, (adv < length / 3.0).sum(axis=1) / safe)


def _window_columns(X, blocks, prefix, fps):
    """Columns d_adv_1s, d_adv_2s and vel_adv_1s of one team, clipped to each block, NaN where the team has no slot."""
    count = X[:, COLUMN[prefix + "n"]]
    adv_mean = X[:, COLUMN[prefix + "adv_mean"]]
    for name, seconds in DISPLACEMENT_S:
        k = int(round(seconds * fps))
        displacement = X[:, COLUMN[prefix + name]]                              # a view: the column stays in X
        for start, stop in blocks:
            if stop - start > 2 * k:
                displacement[start + k:stop - k] = adv_mean[start + 2 * k:stop] - adv_mean[start:stop - 2 * k]
        displacement[~(count > 0)] = np.nan
    velocity = X[:, COLUMN[prefix + "vel_adv"]]
    k = int(round(VELOCITY_WINDOW_S * fps))
    smoothed = kinematics.window_mean(velocity, np.isfinite(velocity), blocks, k, k)
    X[:, COLUMN[prefix + "vel_adv_1s"]] = np.where(count > 0, smoothed, np.nan)


def _where_any(counts, values):
    """`values` where `counts` > 0, else NaN."""
    return np.where(counts > 0, values, np.nan)


def _percentiles(values, counts, percentiles):
    """The given percentiles of every row of `values` (m, K), NaN-padded, as numpy's linear method gives them: a list
    of (m,) arrays, NaN where the row has no number."""
    ordered = np.sort(values, axis=1)                                         # NaN sorts last
    last = np.maximum(counts - 1, 0)
    rows = np.arange(len(values))
    result = []
    for percentile in percentiles:
        position = percentile / 100.0 * last
        low = np.floor(position).astype(np.intp)
        high = np.minimum(low + 1, last)
        below, above = ordered[rows, low], ordered[rows, high]
        result.append(np.where(counts > 0, below + (above - below) * (position - low), np.nan))
    return result
