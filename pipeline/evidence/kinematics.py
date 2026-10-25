"""Shared kinematics for the evidence sources that look at motion: metres, smoothed velocities, windows, runs.

P04 (who controls the ball) and P05 (is it dead) both need the same few things from the tracking: positions in metres,
a velocity per slot and for the ball, the distance from every slot to the ball, the distance beyond a pitch line, and
means over a window of frames. Written once here so that the two agree to the last bit, and so that nothing in either
can look across half time (docs/traps.md T11) or read the ATD's own speed channels, which are NaN on 92% of the graded
frames with a ball and behave like a player's speed where given (T28, T31, A07): every speed here comes from positions.

Velocities are the 7-frame Savitzky-Golay first derivative (2nd order, weights (-3, -2, -1, 0, 1, 2, 3) / 28), applied
inside a *run*: a maximal stretch of valid frames inside one graded block, i.e. one slot's tracklet piece or one stretch
with the ball visible. It is exact on a straight line, and smooths the 1-2 cm jitter that a plain difference would
amplify. This is the smoothing Vidal-Codina et al. (2022) apply to tracking, here without scipy. Where the window does
not fit in the run the central difference is used (one neighbour each side), then the one-sided difference, then NaN on
a run of one frame. Nothing reads a frame outside its run, so a gap, a handover between slots or the half boundary
never leaks into a neighbour's velocity.

`ball_track` and `player_track` remember their result per Inputs object in a `weakref.WeakKeyDictionary`, so that a
grid of configs on one match computes them once. The memo lives in memory and dies with the object; it cannot outlive
the tracking it was made from, and nothing here writes to disk.
"""

import weakref
from dataclasses import dataclass, field

import numpy as np

# Params
CHUNK = 512                                  # frames per step of the derivative, for the cache
SG_HALF = 3                                  # the Savitzky-Golay window is 2 * SG_HALF + 1 = 7 frames
SG_WEIGHTS = np.arange(-SG_HALF, SG_HALF + 1, dtype=np.float64) / 28.0   # sum of k^2 over -3..3 is 28


@dataclass(frozen=True, eq=False)
class BallTrack:
    """The ball in metres. NaN outside the graded blocks and wherever the ball is absent; every array is read-only.

    No array is in the repr (repr=False): a BallTrack printed by a log line would put frames of the match in a log.
    """
    xy_m: np.ndarray = field(repr=False)      # (n, 2) float64
    vel: np.ndarray = field(repr=False)       # (n, 2) float64, metres per second
    speed: np.ndarray = field(repr=False)     # (n,) float64, metres per second
    visible: np.ndarray = field(repr=False)   # (n,) bool, False outside the graded blocks


@dataclass(frozen=True, eq=False)
class PlayerTrack:
    """Every slot in metres. NaN outside the graded blocks and wherever a slot is absent; every array is read-only."""
    xy_m: np.ndarray = field(repr=False)      # (n, S, 2) float64
    vel: np.ndarray = field(repr=False)       # (n, S, 2) float64, metres per second
    speed: np.ndarray = field(repr=False)     # (n, S) float64, metres per second
    present: np.ndarray = field(repr=False)   # (n, S) bool, False outside the graded blocks


_BALL_MEMO = weakref.WeakKeyDictionary()
_PLAYER_MEMO = weakref.WeakKeyDictionary()


def blocks(inputs):
    """The graded blocks as a list of (start, stop), half-open, from inputs.graded."""
    return [(int(start), int(stop)) for _, start, stop in inputs.graded]


def runs(mask):
    """(starts, stops): the maximal runs of True in a 1-D bool array, half-open, as int arrays."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("kinematics.runs: mask must be 1-D, got shape {}".format(mask.shape))
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def derivative(xy_m, valid, blocks, fps):
    """Velocity in units per second of xy_m (n, K, 2) or (n, 2), inside runs of `valid` (n, K) or (n,).

    Savitzky-Golay over 7 frames where the window lies in the run, else the central difference, else the one-sided one,
    else NaN. NaN wherever `valid` is False and outside every block. The result has xy_m's shape.
    """
    xy_m = np.asarray(xy_m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if xy_m.ndim not in (2, 3) or xy_m.shape[-1] != 2 or valid.shape != xy_m.shape[:-1]:
        raise ValueError("kinematics.derivative: xy_m must be (n, 2) or (n, K, 2) and valid its shape without the last "
                         "axis, got {} and {}".format(xy_m.shape, valid.shape))
    out = np.full(xy_m.shape, np.nan)
    for start, stop in blocks:
        if not 0 <= start <= stop <= len(xy_m):
            raise ValueError("kinematics.derivative: block ({}, {}) is outside the {} frames".format(
                start, stop, len(xy_m)))
        if stop - start:
            out[start:stop] = _block_derivative(xy_m[start:stop], valid[start:stop], fps)
    return out


def _block_derivative(xy, valid, fps):
    """One block: pad by the window's half-width with invalid frames, so that nothing reads past the block's edge, and
    work in chunks of CHUNK frames, which keeps every temporary in the cache (about twice as fast as whole arrays)."""
    m = len(xy)
    pad = ((SG_HALF, SG_HALF),) + ((0, 0),) * (xy.ndim - 1)
    xp = np.pad(np.where(valid[..., None], xy, 0.0), pad)
    vp = np.pad(valid, ((SG_HALF, SG_HALF),) + ((0, 0),) * (valid.ndim - 1))
    result = np.empty(xy.shape)
    for low in range(0, m, CHUNK):
        high = min(low + CHUNK, m)
        result[low:high] = _chunk_derivative(xp[low:high + 2 * SG_HALF], vp[low:high + 2 * SG_HALF], fps)
    return result


def _chunk_derivative(xp, vp, fps):
    """Velocities of the frames SG_HALF in from each end of xp (zeros where vp is False) and vp (padded the same)."""
    m = len(xp) - 2 * SG_HALF

    def at(array, k):   # array[t + k] for every t of the chunk
        return array[SG_HALF + k:SG_HALF + k + m]

    valid = at(vp, 0)
    ahead, behind = at(vp, 1), at(vp, -1)
    central = valid & ahead & behind
    window = central.copy()
    for k in range(2, SG_HALF + 1):
        window &= at(vp, k) & at(vp, -k)

    here = at(xp, 0)
    result = np.full(here.shape, np.nan)
    # Cheapest first, overwritten by the better estimate where the run allows it: one-sided, central, then the 7-frame
    # derivative.
    np.copyto(result, at(xp, 1) - here, where=(valid & ahead & ~behind)[..., None])
    np.copyto(result, here - at(xp, -1), where=(valid & behind & ~ahead)[..., None])
    np.copyto(result, (at(xp, 1) - at(xp, -1)) / 2.0, where=(central & ~window)[..., None])
    smooth = SG_WEIGHTS[SG_HALF + 1] * (at(xp, 1) - at(xp, -1))
    for k in range(2, SG_HALF + 1):
        smooth += SG_WEIGHTS[SG_HALF + k] * (at(xp, k) - at(xp, -k))
    np.copyto(result, smooth, where=window[..., None])
    result *= fps
    return result


def ball_track(inputs):
    """The ball's BallTrack, remembered per Inputs object (same read-only arrays on every call)."""
    track = _BALL_MEMO.get(inputs)
    if track is None:
        track = _BALL_MEMO[inputs] = _build_ball(inputs)
    return track


def player_track(inputs):
    """Every slot's PlayerTrack, remembered per Inputs object (same read-only arrays on every call)."""
    track = _PLAYER_MEMO.get(inputs)
    if track is None:
        track = _PLAYER_MEMO[inputs] = _build_players(inputs)
    return track


def _graded(inputs):
    mask = np.zeros(inputs.atd.ball_xy.shape[0], dtype=bool)
    for start, stop in blocks(inputs):
        mask[start:stop] = True
    return mask


def _build_ball(inputs):
    atd = inputs.atd
    graded = _graded(inputs)
    visible = atd.ball_present & graded
    xy_m = inputs.metres(atd.ball_xy)       # NaN where absent already; a fresh array, so it can be masked in place
    xy_m[~graded] = np.nan
    vel = derivative(xy_m, visible, blocks(inputs), inputs.fps)
    return BallTrack(xy_m=_frozen(xy_m), vel=_frozen(vel), speed=_frozen(np.hypot(vel[:, 0], vel[:, 1])),
                     visible=_frozen(visible))


def _build_players(inputs):
    atd = inputs.atd
    graded = _graded(inputs)
    present = atd.players_present & graded[:, None]
    xy_m = inputs.metres(atd.players_xy)
    xy_m[~graded] = np.nan
    vel = derivative(xy_m, present, blocks(inputs), inputs.fps)
    return PlayerTrack(xy_m=_frozen(xy_m), vel=_frozen(vel), speed=_frozen(np.hypot(vel[..., 0], vel[..., 1])),
                       present=_frozen(present))


def _frozen(array):
    array.flags.writeable = False
    return array


def ball_distances(player_xy_m, ball_xy_m):
    """(n, S) metres from every slot to the ball; NaN where either is absent."""
    delta = player_xy_m - ball_xy_m[:, None, :]
    return np.hypot(delta[..., 0], delta[..., 1])


def outside_m(xy_m, pitch):
    """Signed distance beyond the nearest pitch line, for xy_m (..., 2) in metres with the origin at a corner.

    Positive outside the pitch (the distance to it, to the corner when beyond two lines), minus the distance to the
    nearest line inside, zero on a line, NaN where xy_m is NaN. `pitch` is (length, width) in metres.
    """
    xy_m = np.asarray(xy_m, dtype=np.float64)
    length, width = pitch
    beyond_x = np.maximum(-xy_m[..., 0], xy_m[..., 0] - length)   # per axis: positive outside, minus inside
    beyond_y = np.maximum(-xy_m[..., 1], xy_m[..., 1] - width)
    outside = np.hypot(np.maximum(beyond_x, 0.0), np.maximum(beyond_y, 0.0))
    return np.where((beyond_x > 0) | (beyond_y > 0), outside, np.maximum(beyond_x, beyond_y))


def window_mean(values, valid, blocks, before, after):
    """(n,) mean of the valid, finite `values` over [t - before, t + after], clipped to t's block; NaN if none.

    NaN as well for every t outside all blocks. Cumulative sums per block, so the cost does not grow with the window.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("kinematics.window_mean: values and valid must be 1-D and of one shape, got {} and {}".format(
            values.shape, valid.shape))
    if before < 0 or after < 0:
        raise ValueError("kinematics.window_mean: before and after must be >= 0 frames, got {} and {}".format(
            before, after))
    out = np.full(len(values), np.nan)
    for start, stop in blocks:
        m = stop - start
        if m <= 0:
            continue
        ok = valid[start:stop] & np.isfinite(values[start:stop])   # a NaN would poison every later cumulative sum
        total = np.concatenate(([0.0], np.cumsum(np.where(ok, values[start:stop], 0.0))))
        count = np.concatenate(([0], np.cumsum(ok)))
        t = np.arange(m)
        low = np.maximum(t - before, 0)
        high = np.minimum(t + after + 1, m)
        n = count[high] - count[low]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)
    return out
