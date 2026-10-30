"""Our camera's aim, fitted as a follower of our ball, and a camera like it simulated on the DFL matches (P09).

    fit_ = load_fit()                                 # BAR-ATH, fitted once and cached: constants and aggregates only
    aim = simulated_aim(match, seed, fit_)            # (n,) metres, the aim a camera like ours would have had on DFL
    stats = measure(aim, ball_x, ball_seen, dead, blocks)   # the gates' statistics, one code for ours and DFL's
    digest(fit_)                                      # what a report names it by: the eight constants

Why a simulated camera. P09's belief is that the tactical camera's aim (pipeline/inputs/camera.py: the pitch point under
the image centre) says where the ball roughly is along the pitch even while our tracker has lost it, since the operator
kept following the real ball. The DFL open data has no video, so a benchmark match has no aim of its own, and a feature
built on the aim could not be measured with a real F1 test. This module gives each benchmark match one: a camera that
follows DFL's clean ball as ours follows our ball, with our camera's lag, smoothing, gain and noise.

The follower. The aim is modelled as a lagged, smoothed copy of the ball's position along the pitch, plus AR(1) noise:

    aim(t) = a + b * lowpass(ball_x)(t - lag) + e(t),     e(t) = phi * e(t - 1) + sigma * z(t)

lowpass is a first-order exponential smoother with alpha = 1 / (1 + tau), which holds its last value across a missing
ball (the follower cannot follow what it is not given, and our tracker's gaps are long: median 10.7 s, A02) and starts
again at each graded block. b below 1 is a camera that pans less far than the ball goes, staying short of the goal
lines, and a is the offset that centres it. fit() searches lag 0..80 frames (0 to 3.2 s) and tau in TAUS, and fits a
and b by least squares for each pair, on the frames where the aim at t is there and our ball at t - lag is seen and
settled (below). The pair with the smallest residual variance is kept (a tie goes to the smaller lag, then the smaller
tau), the residuals' lag-1 autocorrelation is phi and their standard deviation times sqrt(1 - phi^2) is sigma, so
that the noise has the residuals' variance. lo and hi are the 0.5th and 99.5th percentiles of our aim on the graded
frames: the simulated aim is clipped to them, since a real camera does not pan past where it ever pointed. Nothing is
random in the fit.

Settled frames: the fit after our ball's gaps. Our smoother holds its last value through each of our tracker's gaps,
but the real camera kept following the real ball, so for a while after a gap our smoothed ball still carries the stale
value and the camera does not: residuals that would bias the fit towards a small tau (whose smoother forgets the
stale value fastest) and a larger sigma. So for each candidate tau a frame's regressor counts only once the smoother
has seen at least ceil(SETTLE * tau) visible frames in a row since the last gap (the block's start counts as one,
since the smoother starts there from nothing while the camera was already following): what is left of the stale value
is then (tau / (1 + tau)) ** ceil(SETTLE * tau), about e^-3 = 5% of it. tau 0 has no memory and needs only the frame
itself to be seen. The candidates are then compared on their own settled frames. The simulator needs none of this:
DFL's clean ball has no gaps to settle from.

The simulator. simulated_aim drives the fitted follower with the match's clean ball, clean_atd(match).ball_xy[:, 0]
times the pitch's length: the real camera follows the real ball, even where our tracker loses it, so the degraded ball
a config is served is not what the camera follows. It reads the positions only (clean_atd holds no flag, inputs.py),
never ball_status or ball_possession: the aim it gives is served to the components as MatchInputs.camera_aim, and must
hold no truth (checks: test_camera_simulation_reads_no_truth). The noise is drawn from
np.random.default_rng([seed, SEED_TAG, crc32 of the match id]) (T22), block by block, from the noise's stationary
distribution at each block's start; before `lag` frames of a block have passed, the follower reads the block's first
smoothed ball (the camera waits where play starts). The aim is NaN outside the graded blocks, as no graded frame needs
it and our own is not fitted there.

The statistics (measure). The fidelity gates (Task E.3, spec 5.E) compare the simulated camera with ours by statistics
that one function computes on both sides: the lag of peak correlation between the aim and the ball seen, the peak
correlation, the median and p90 along-pitch error at that lag, and the aim's median speed on decoded dead frames over
alive ones (a gate) and in ball gaps over ball-visible frames (a WARN). A ratio of medians does not depend on the frame
rate, so two cameras at different rates can be compared by it.

The gates (Task E.3, `camera.py gates`). Declared in the design (section 5.E) and in R019 part 1a before the first
simulation, and never tuned towards: the simulated camera on the benchmark's degraded copies (`--profile`, default
tracking, 7 matches x seeds) against ours, each by measure(). Four gates, each a tolerance on the difference (GATES):
the peak lag within 10 frames, the peak correlation within 0.05, the median along-pitch error within 3.0 m, and the
dead-over-alive speed ratio within 25% of ours (sim / ours - 1). The ratio is the camera's behaviour by the state of
play, so `dead` is what main.toml decodes as D on both sides, by the pipeline's own functions (run.predict), never the
truth: ours on BAR-ATH, the simulated one on each served match. The DFL side pools the rows of every (match, seed)
into one measurement (a median of the pooled frames, not a median of medians), each in its own blocks so that no pair
crosses two of them. The WARN (WARNS) is the gap-over-visible speed ratio, with the same 25% band as information: it
never fails the command, since tracking places its gaps blind to the play (T38) and the simulated camera keeps moving
through them. A statistic that cannot be measured on either side fails its gate (nothing says it passed). The report
goes to <data dir>/runs/camera/gates_<profile>.json: aggregates only.

The digest. digest(fit) is what a report (and so choose.py's SAME) names the camera by: the sha256 of the eight
constants the simulator uses (lag, tau, a, b, phi, sigma, lo, hi), and nothing of `stats`, so that a change to what
measure() reports never makes two runs of the same simulated camera look different.

The cache. load_fit keeps the fit in <data dir>/cache/benchmark/camera/<key digest>/fit.json, with the key that made it:
the ATD's and the homography's source files (bytes and sha256), the fit's constants (FIT_VERSION, LAGS, TAUS, SETTLE,
CLIP_PERCENTILES, P90, DECIMALS and the image centre the aim is read at), the text sha256 of the code it runs
(SOURCES: this file, the aim's definition, the ball's kinematics, the homography's loader and the inputs that align
them), the numpy version, and the match's frames, rate, pitch and graded blocks. A cache is read only if the whole key
matches, calibration.load's rule, so an edited definition never reuses an old fit; an unreadable cache is absent, and is
fitted again. The folder name is the first calibration.KEY_DIGITS hex digits of the key's sha256, as the Calibration's
is.

What a fit holds is aggregates of BAR-ATH: eight constants and a short dict of statistics (scalars). Nothing per frame
is kept, cached, printed or returned by load_fit, so the fit and its file may be printed and committed as they are (the
data rule of P09). The arrays it is fitted from are the match's and are dropped once it is made.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import numbers
import os
import sys
import zlib
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark.inputs import clean_atd  # noqa: E402
from pipeline.inputs import camera as aim_module  # noqa: E402

# Params
FIT_VERSION = 1                 # bump to refit every cache at once; an edit to this file does that already (the key)
LAGS = tuple(range(81))         # frames the aim may trail the ball by: 0 to 3.2 s at 25 Hz (the fit and the peak lag)
TAUS = (0, 5, 10, 20, 35, 50)   # the smoother's time constants tried, in frames: alpha = 1 / (1 + tau)
SETTLE = 3                      # after a gap, a regressor counts from ceil(SETTLE * tau) visible frames in a row on
CONSTANTS = ("lag", "tau", "a", "b", "phi", "sigma", "lo", "hi")    # what the simulator uses, and what digest() names
CLIP_PERCENTILES = (0.5, 99.5)  # the simulated aim is clipped to these percentiles of ours: lo and hi
SEED_TAG = 9009                 # the simulator's own constant in its generator's seed (T22)
FPS = 25                        # frames per second of a speed, where none is given (ours and DFL's)
P90 = 90.0                      # the error percentile reported beside the median
DECIMALS = 9                    # the fit's floats are rounded to this, so that its digest is the same on any machine
CACHE_FOLDER = ("cache", "benchmark", "camera")     # <data dir>/cache/benchmark/camera/<key digest>/fit.json
FIT_FILE = "fit.json"
MIN_PAIRS = 3                   # a regression or a correlation needs at least this many frames
# The code a fit depends on, by its path from the repository's root: its sha256 (CRLF read as LF) is in the cache key.
SOURCES = ("pipeline/benchmark/camera.py", "pipeline/inputs/camera.py", "pipeline/evidence/kinematics.py",
           "starter/load_homography.py", "pipeline/inputs/__init__.py")


@dataclass(frozen=True)
class CameraFit:
    """The follower fitted on our camera: aim(t) = a + b * lowpass(ball_x, tau)(t - lag) + AR(1)(phi, sigma).

    lag and tau are in frames, a, sigma, lo and hi in metres, b and phi unitless. `stats` holds the aggregates the gates
    read (measure's statistics of our camera, and the fit's own: frames used, residual sd, R squared), scalars or None
    only. ValueError, naming the field, for a value the simulator could not use: a lag outside LAGS, a negative or
    non-finite tau, phi not strictly between -1 and 1 (the noise would not be stationary), a negative sigma, lo above
    hi.
    """
    lag: int
    tau: float
    a: float
    b: float
    phi: float
    sigma: float
    lo: float
    hi: float
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.lag, bool) or not isinstance(self.lag, numbers.Integral) or not 0 <= self.lag <= LAGS[-1]:
            raise ValueError("CameraFit: lag must be a whole number of frames from 0 to {}, got {!r}".format(
                LAGS[-1], self.lag))
        for name in ("tau", "a", "b", "phi", "sigma", "lo", "hi"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError("CameraFit: {} must be a finite number, got {!r}".format(name, value))
        if self.tau < 0:
            raise ValueError("CameraFit: tau must be at least 0 frames, got {!r}".format(self.tau))
        if not -1.0 < self.phi < 1.0:
            raise ValueError("CameraFit: phi must be strictly between -1 and 1, got {!r}".format(self.phi))
        if self.sigma < 0:
            raise ValueError("CameraFit: sigma must be at least 0 metres, got {!r}".format(self.sigma))
        if self.lo > self.hi:
            raise ValueError("CameraFit: lo ({!r}) must not be above hi ({!r})".format(self.lo, self.hi))
        if not isinstance(self.stats, dict) or not all(
                isinstance(key, str) and (value is None or isinstance(value, (bool, numbers.Real, str)))
                for key, value in self.stats.items()):
            raise ValueError("CameraFit: stats must be a dict of names to scalars (aggregates only)")


# -- the smoother -----------------------------------------------------------------------------------------------------

def lowpass(x, tau, blocks=None):
    """(n,) x smoothed by y(t) = y(t - 1) + alpha * (x(t) - y(t - 1)), alpha = 1 / (1 + tau); NaN where it has no value.

    A missing x (NaN or infinite) holds the last value; the first finite x of a block starts it, and the frames before
    that are NaN. `blocks` is a list of (half, start, stop), half-open: each is smoothed on its own, starting again at
    its first frame, and every frame outside them is NaN. None smooths the whole array as one block. tau 0 is the
    ball itself with its gaps held.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("lowpass: x must be one-dimensional, got shape {}".format(x.shape))
    if isinstance(tau, bool) or not isinstance(tau, numbers.Real) or not math.isfinite(tau) or tau < 0:
        raise ValueError("lowpass: tau must be a finite number of frames, at least 0, got {!r}".format(tau))
    alpha = 1.0 / (1.0 + float(tau))
    spans = [(0, len(x))] if blocks is None else [(start, stop) for _, start, stop in _blocks(blocks, len(x))]
    out = np.full(len(x), np.nan)
    for start, stop in spans:
        segment = x[start:stop]
        finite = np.isfinite(segment)
        if not finite.any():
            continue
        # The recursion runs over the block's finite values only (a missing one changes nothing), and every frame then
        # takes the value of the last finite frame at or before it: the hold. The first value starts it as it is.
        values = segment[finite].tolist()
        state, smoothed = values[0], []
        append = smoothed.append
        for value in values:
            state += alpha * (value - state)
            append(state)
        filled = np.full(len(segment), np.nan)
        filled[finite] = smoothed
        last = np.maximum.accumulate(np.where(finite, np.arange(len(segment)), -1))
        out[start:stop] = np.where(last >= 0, filled[np.maximum(last, 0)], np.nan)
    return out


# -- the fit ----------------------------------------------------------------------------------------------------------

def fit(aim, ball_x, blocks, fps=FPS):
    """The CameraFit of `aim` (n,) as a follower of `ball_x` (n,), metres, inside `blocks` [(half, start, stop)].

    See the module docstring for the model and the search. ball_x is NaN where the ball is not seen; aim may be NaN
    anywhere (an invalid homography). Frames outside the blocks are not used. Deterministic: no randomness, the grid in
    a fixed order. ValueError if no (lag, tau) has MIN_PAIRS settled frames with an aim, or the regressor never
    varies.
    """
    aim, ball_x = _vector("aim", aim), _vector("ball_x", ball_x)
    if ball_x.shape != aim.shape:
        raise ValueError("fit: aim and ball_x must have the same length, got {} and {}".format(aim.size, ball_x.size))
    spans = _blocks(blocks, aim.size)
    aim_ok, seen = np.isfinite(aim), np.isfinite(ball_x)
    smoothed = {tau: lowpass(ball_x, tau, blocks=spans) for tau in TAUS}
    runs = _visible_runs(seen, spans)
    settled = {tau: runs >= max(1, math.ceil(SETTLE * tau)) for tau in TAUS}      # the module docstring's rule

    # The scan reads sums over contiguous slices (each block's frames t and t - lag), weighted by the frames used: the
    # same least squares as on the gathered frames, and fast enough to try 486 pairs on a match. The values are centred
    # first (and NaN set to 0, where the weight is 0 anyway), so that the sums of squares do not cancel.
    y0 = _centred(aim)
    y0_squared = y0 * y0
    x0 = {tau: _centred(values) for tau, values in smoothed.items()}
    x0_squared = {tau: values * values for tau, values in x0.items()}
    best = None
    for lag in LAGS:
        parts = _slices(spans, lag)
        for tau in TAUS:
            n = sy = syy = sx = sxx = sxy = 0.0
            for now, then in parts:
                weight = (aim_ok[now] & settled[tau][then]).astype(np.float64)
                n += float(weight.sum())
                sy += float(weight @ y0[now])
                syy += float(weight @ y0_squared[now])
                sx += float(weight @ x0[tau][then])
                sxx += float(weight @ x0_squared[tau][then])
                sxy += float((weight * y0[now]) @ x0[tau][then])
            if n < MIN_PAIRS:
                continue
            syy -= sy * sy / n
            sxx -= sx * sx / n
            sxy -= sx * sy / n
            if not sxx > 0.0:
                continue
            variance = (syy - sxy * sxy / sxx) / n
            if best is None or variance < best[0]:
                best = (variance, lag, tau)
    if best is None:
        raise ValueError("fit: no lag in 0..{} and tau in {} has {} frames with an aim and a settled ball, with the "
                         "ball moving; nothing can be fitted".format(LAGS[-1], TAUS, MIN_PAIRS))

    _, lag, tau = best
    now, then = _pairs(spans, lag)
    keep = aim_ok[now] & settled[tau][then]
    now, then = now[keep], then[keep]
    x, y = smoothed[tau][then], aim[now]
    b = float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))
    a = float(y.mean() - b * x.mean())
    residuals = y - (a + b * x)
    following = np.flatnonzero(now[1:] == now[:-1] + 1)        # pairs of the fit's frames one frame apart, in a block
    phi = _correlation(residuals[following], residuals[following + 1])
    if phi is None:
        raise ValueError("fit: fewer than {} pairs of consecutive fitted frames; phi cannot be measured".format(
            MIN_PAIRS))
    phi = min(max(phi, -1.0 + 1e-9), 1.0 - 1e-9)                # a perfect correlation is not a stationary noise
    sd = float(residuals.std())
    in_blocks = _in_blocks(spans, aim.size)
    lo, hi = np.percentile(aim[in_blocks & aim_ok], CLIP_PERCENTILES)
    stats = {"fit_frames": int(y.size), "residual_sd": _round(sd),
             "r2": _round(1.0 - float(residuals @ residuals) / float((y - y.mean()) @ (y - y.mean())))}
    masked = np.where(in_blocks, aim, np.nan)
    stats.update(measure(masked, ball_x, seen & in_blocks, None, blocks=spans, fps=fps))
    return CameraFit(lag=int(lag), tau=float(tau), a=_round(a), b=_round(b), phi=_round(phi),
                     sigma=_round(sd * math.sqrt(1.0 - phi * phi)), lo=_round(lo), hi=_round(hi), stats=stats)


# -- the statistics ---------------------------------------------------------------------------------------------------

def measure(aim, ball_x, ball_seen, dead, blocks, fps=FPS):
    """The gates' statistics of a camera: {name: number or None}, the same code for ours and for a simulated one.

    `aim` and `ball_x` are (n,) metres along the pitch, `ball_seen` (n,) bool (the ball as the tracker serves it, not
    the clean one), `dead` (n,) bool of the frames decoded D, or None (the dead/alive entries are then None).
    `blocks` [(half, start, stop)], at least one, required: every pair and every difference is kept inside one block,
    and no frame outside them is ever counted, so that an aim off the graded frames (ours has one through half time)
    cannot reach a gate for want of a mask.

        peak_lag, peak_corr     the lag in LAGS maximising corr(aim(t + lag), ball_x(t)) over frames t with the ball
                                seen (a tie: the smaller lag), and that correlation; pairs, the frames it was on
        median_err, p90_err     the median and 90th percentile of |aim(t + peak_lag) - ball_x(t)| over the same
                                frames, in metres
        speed_*_mps             the aim's median |aim(t) - aim(t - 1)| * fps on dead frames, on the other frames of the
                                blocks (alive), on frames without the ball seen (gap) and with it (visible)
        speed_ratio_dead_alive  dead over alive (a gate); speed_ratio_gap_visible, gap over visible (the WARN)

    A statistic with no frame to stand on is None.
    """
    aim, ball_x = _vector("aim", aim), _vector("ball_x", ball_x)
    ball_seen = np.asarray(ball_seen, dtype=bool)
    if ball_x.shape != aim.shape or ball_seen.shape != aim.shape:
        raise ValueError("measure: aim, ball_x and ball_seen must have the same length, got {}, {} and {}".format(
            aim.size, ball_x.size, ball_seen.size))
    if dead is not None:
        dead = np.asarray(dead, dtype=bool)
        if dead.shape != aim.shape:
            raise ValueError("measure: dead must have one flag per frame ({}), got {}".format(aim.size, dead.size))
    spans = _blocks(blocks, aim.size)
    if not spans:
        raise ValueError("measure: blocks must name at least one graded block (half, start, stop); got none")
    seen = ball_seen & np.isfinite(ball_x)
    aim_ok = np.isfinite(aim)

    peak = None
    a0, b0 = _centred(aim), _centred(ball_x)
    a0_squared, b0_squared = a0 * a0, b0 * b0
    for lag in LAGS:                                  # the correlations by sums over slices, as fit() scans
        n = sa = sb = saa = sbb = sab = 0.0
        for now, then in _slices(spans, lag):
            weight = (aim_ok[now] & seen[then]).astype(np.float64)
            n += float(weight.sum())
            sa += float(weight @ a0[now])
            sb += float(weight @ b0[then])
            saa += float(weight @ a0_squared[now])
            sbb += float(weight @ b0_squared[then])
            sab += float((weight * a0[now]) @ b0[then])
        if n < MIN_PAIRS:
            continue
        caa, cbb, cab = saa - sa * sa / n, sbb - sb * sb / n, sab - sa * sb / n
        if not (caa > 0.0 and cbb > 0.0):
            continue
        corr = cab / math.sqrt(caa * cbb)
        if peak is None or corr > peak[1]:
            peak = (lag, corr)
    stats = {"peak_lag": None, "peak_corr": None, "pairs": 0, "median_err": None, "p90_err": None}
    if peak is not None:
        lag, corr = peak
        now, then = _pairs(spans, lag)
        keep = aim_ok[now] & seen[then]
        now, then = now[keep], then[keep]
        errors = np.abs(aim[now] - ball_x[then])
        stats.update(peak_lag=int(lag), peak_corr=_round(corr), pairs=int(now.size),
                     median_err=_round(np.median(errors)), p90_err=_round(np.percentile(errors, P90)))

    now, before = _pairs(spans, 1)
    speed = np.full(aim.size, np.nan)
    speed[now] = np.abs(aim[now] - aim[before]) * float(fps)
    moving = np.isfinite(speed)
    medians = {"dead": None, "alive": None}
    if dead is not None:
        medians = {"dead": _median(speed[moving & dead]), "alive": _median(speed[moving & ~dead])}
    medians.update(gap=_median(speed[moving & ~seen]), visible=_median(speed[moving & seen]))
    for name, value in medians.items():
        stats["speed_{}_mps".format(name)] = value
    stats["speed_ratio_dead_alive"] = _ratio(medians["dead"], medians["alive"])
    stats["speed_ratio_gap_visible"] = _ratio(medians["gap"], medians["visible"])
    return stats


# -- the simulator ----------------------------------------------------------------------------------------------------

def simulated_aim(match, seed, fit):
    """(n,) metres: the aim of a camera like ours on `match` (a DflMatch), for degradation seed `seed`.

    The fitted follower driven by the match's clean ball (clean_atd: positions, no flag), plus the fit's AR(1) noise
    drawn from np.random.default_rng([seed, SEED_TAG, crc32(match id)]), clipped to [fit.lo, fit.hi], NaN outside the
    graded blocks. The same (match, seed, fit) gives the same bytes. ValueError for a seed that is not a whole number
    of at least 0, or a fit that is not a CameraFit.
    """
    if not isinstance(fit, CameraFit):
        raise ValueError("simulated_aim: fit must be a CameraFit (load_fit()), got a {}".format(type(fit).__name__))
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("simulated_aim: seed must be a whole number of at least 0, got {!r}".format(seed))
    ball_x = clean_atd(match).ball_xy[:, 0] * float(match.pitch[0])
    blocks = _blocks(match.graded, match.n_frames)
    smoothed = lowpass(ball_x, fit.tau, blocks=blocks)
    rng = np.random.default_rng([int(seed), SEED_TAG, zlib.crc32(match.match_id.encode("utf-8"))])
    start_sd = fit.sigma / math.sqrt(1.0 - fit.phi * fit.phi)
    aim = np.full(match.n_frames, np.nan)
    for _, start, stop in blocks:
        followed = _backfilled(smoothed[start:stop])
        source = np.maximum(np.arange(stop - start) - fit.lag, 0)
        noise = _ar1(rng.standard_normal(stop - start), fit.phi, fit.sigma, start_sd)
        aim[start:stop] = np.clip(fit.a + fit.b * followed[source] + noise, fit.lo, fit.hi)
    return aim


def _ar1(z, phi, sigma, start_sd):
    """e(0) = start_sd * z(0), e(t) = phi * e(t - 1) + sigma * z(t): the noise, from the stationary distribution."""
    values = z.tolist()
    out = [0.0] * len(values)
    state = 0.0
    for t, value in enumerate(values):
        state = start_sd * value if t == 0 else phi * state + sigma * value
        out[t] = state
    return np.array(out, dtype=np.float64)


def _backfilled(values):
    """`values` with its leading NaNs set to its first finite value (the camera waits where play starts)."""
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size or finite[0] == 0:
        return values
    values = values.copy()
    values[:finite[0]] = values[finite[0]]
    return values


# -- the cache --------------------------------------------------------------------------------------------------------

# Taken as the module is imported, as calibration.py takes its own: a process that imported the code before it changed
# goes on running it, and keeps the cache that fits it.
SOURCE_SHA256 = {name: calibration_module._text_sha256(os.path.join(ROOT, *name.split("/"))) for name in SOURCES}


def load_fit(data_dir=None, rebuild=False, inputs=None):
    """The CameraFit of BAR-ATH, from the cache when its key matches, otherwise fitted and stored.

    `data_dir` defaults to games.DATA_DIR (MTAH_DATA_DIR); the cache is <data dir>/cache/benchmark/camera/<key digest>/
    fit.json and nothing else is written. The aim is Inputs().camera_aim, the ball kinematics.ball_track(inputs)'s x in
    metres (NaN where it is not seen, and off the graded halves), the blocks inputs.graded. rebuild=True fits again.
    `inputs` stands in for the real match (the self-checks pass a synthetic bundle). Prints nothing.
    """
    if inputs is None:
        from pipeline.inputs import Inputs  # here, not above: it pulls in pandas and the starter loaders
        inputs = Inputs(data_dir=data_dir)
    data_dir = data_dir or inputs.data_dir
    key = _key(inputs)
    path = os.path.join(cache_folder(data_dir, key), FIT_FILE)
    if not rebuild:
        cached = _read(path, key)
        if cached is not None:
            return cached
    from pipeline.evidence import kinematics
    made = fit(np.asarray(inputs.camera_aim), kinematics.ball_track(inputs).xy_m[:, 0], inputs.graded, fps=inputs.fps)
    _write(path, key, made)
    return made


def digest(fit):
    """The sha256 of the eight constants the simulator uses (CONSTANTS), as canonical JSON: what reports name the camera
    by. Two fits that would simulate the same aim have the same digest, whatever their `stats` say."""
    text = json.dumps({name: getattr(fit, name) for name in CONSTANTS}, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_folder(data_dir, key):
    """<data dir>/cache/benchmark/camera/<the first KEY_DIGITS hex digits of the key's canonical JSON's sha256>/."""
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return os.path.join(data_dir, *CACHE_FOLDER,
                        hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:calibration_module.KEY_DIGITS])


def _key(inputs):
    """Everything the fit depends on, as JSON-normal data: the sources, the constants, the code, the match's shape."""
    import games
    from pipeline.inputs import atd_cache
    homography = games.require(["homography"], inputs.game, inputs.data_dir)[0]
    key = {
        "fit_version": FIT_VERSION,
        "constants": {"lags": [LAGS[0], LAGS[-1]], "taus": list(TAUS), "settle": SETTLE,
                      "clip_percentiles": list(CLIP_PERCENTILES), "centre": list(aim_module.CENTRE), "p90": P90,
                      "decimals": DECIMALS},
        "source_sha256": dict(SOURCE_SHA256),
        "atd_sources": inputs.atd_manifest["sources"],
        "atd_cache_version": inputs.atd_manifest["cache_version"],
        "homography": atd_cache.describe_source(homography),
        "numpy": np.__version__,
        "n_frames": inputs.n_frames,
        "pitch": [float(side) for side in inputs.pitch],
        "fps": inputs.fps,
        "graded": [list(block) for block in inputs.graded],
    }
    return json.loads(json.dumps(key, sort_keys=True))


def _document(fit):
    """A CameraFit as JSON-ready data: its eight constants and its stats, sorted."""
    return {"lag": fit.lag, "tau": fit.tau, "a": fit.a, "b": fit.b, "phi": fit.phi, "sigma": fit.sigma,
            "lo": fit.lo, "hi": fit.hi, "stats": dict(sorted(fit.stats.items()))}


def _write(path, key, made):
    """fit.json, whole: written to a temp file of this process in the folder and moved over the old one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps({"key": key, "fit": _document(made)}, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temp_path = "{}.{}.tmp".format(path, os.getpid())
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        calibration_module._move_into_place(temp_path, path)
    except BaseException:   # no temp file is left behind, and the error is the caller's
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _read(path, key):
    """The CameraFit that fit.json holds under exactly `key`, or None if there is none or it cannot be read as one."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict) or document.get("key") != key:
            return None
        saved = document["fit"]
        names = [item.name for item in dataclasses.fields(CameraFit)]
        if not isinstance(saved, dict) or sorted(saved) != sorted(names):
            return None
        return CameraFit(**{name: saved[name] for name in names})
    except (OSError, ValueError, KeyError, TypeError):
        return None


# -- small things -----------------------------------------------------------------------------------------------------

def _vector(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("{} must be one-dimensional, got shape {}".format(name, values.shape))
    return values


def _blocks(blocks, n):
    """[(half, start, stop)] as ints, each inside 0..n with start < stop; ValueError otherwise."""
    out = []
    for block in blocks:
        half, start, stop = (int(value) for value in block)
        if not 0 <= start < stop <= n:
            raise ValueError("block {} is not a range inside the {} frames".format(tuple(block), n))
        out.append((half, start, stop))
    return out


def _in_blocks(spans, n):
    mask = np.zeros(n, dtype=bool)
    for _, start, stop in spans:
        mask[start:stop] = True
    return mask


def _pairs(spans, lag):
    """(now, then): every frame `now` of a block whose frame `then` = now - lag is in the same block, in time order."""
    now = [np.arange(start + lag, stop) for _, start, stop in spans if stop - start > lag]
    if not now:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    now = np.concatenate(now)
    return now, now - lag


def _visible_runs(seen, spans):
    """(n,) int: the frames in a row with the ball seen that end at each frame, counted from its block's start (a gap
    or the start resets it); 0 where the ball is not seen and outside the blocks."""
    runs = np.zeros(seen.size, dtype=np.int64)
    for _, start, stop in spans:
        segment = seen[start:stop]
        index = np.arange(stop - start)
        last_unseen = np.maximum.accumulate(np.where(segment, -1, index))
        runs[start:stop] = np.where(segment, index - last_unseen, 0)
    return runs


def _slices(spans, lag):
    """[(now, then)]: per block long enough, the slice of its frames t and the slice of its frames t - lag."""
    return [(slice(start + lag, stop), slice(start, stop - lag)) for _, start, stop in spans if stop - start > lag]


def _centred(values):
    """`values` less the mean of its finite values, with every non-finite value set to 0 (a weight of 0 skips it)."""
    finite = np.isfinite(values)
    centre = float(values[finite].mean()) if finite.any() else 0.0
    return np.where(finite, values - centre, 0.0)


def _correlation(x, y):
    """Pearson's correlation of two equal-length arrays, or None with fewer than MIN_PAIRS or no variance."""
    if x.size < MIN_PAIRS:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(xc @ xc) * float(yc @ yc))
    if not denominator > 0.0:
        return None
    return float(xc @ yc) / denominator


def _median(values):
    return _round(np.median(values)) if values.size else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _round(numerator / denominator)


def _round(value):
    """A summary number as a plain float rounded to DECIMALS (BLAS summation order varies by machine)."""
    return round(float(value), DECIMALS) + 0.0


# -- the fidelity gates -----------------------------------------------------------------------------------------------

# Params
# (name, statistic, kind, tolerance): "abs" is |simulated - ours| <= tolerance, "rel" is |simulated / ours - 1| <= it.
GATES = (
    ("peak_lag", "peak_lag", "abs", 10.0),                          # frames
    ("peak_corr", "peak_corr", "abs", 0.05),
    ("median_err", "median_err", "abs", 3.0),                       # metres
    ("speed_ratio_dead_alive", "speed_ratio_dead_alive", "rel", 0.25),
)
WARNS = (("speed_ratio_gap_visible", "speed_ratio_gap_visible", "rel", 0.25),)    # information: never a failure
NO_CAMERA_PROFILES = ("clean",)        # clean's ball is DFL's own, with no gap: there is nothing to hold against ours
DEFAULT_PROFILE = "tracking"
DEFAULT_SEEDS = 3
CONFIG = "pipeline/configs/main.toml"  # whose decoded D is the state of play on both sides
REPORT_FOLDER = ("runs", "camera")     # <data dir>/runs/camera/gates_<profile>.json


def compare(ours, simulated):
    """{"gates", "warns", "passed"}: the declared tolerances applied to two measure() dicts, ours and simulated.

    Each entry is {name, ours, simulated, delta, tolerance, kind, status}; delta is simulated - ours ("abs") or
    simulated / ours - 1 ("rel"), rounded to DECIMALS so that a value exactly on a tolerance passes whatever the
    floating point says. A gate's status is "pass" or "FAIL" (FAIL as well when either side is None, or ours is 0
    under "rel": nothing was shown to hold); a WARN's is "ok", "WARN" or "n/a" (not measurable), and never fails.
    `passed` is whether no gate failed.
    """
    def entry(spec, outside, inside):
        name, stat, kind, tolerance = spec
        mine, theirs = ours.get(stat), simulated.get(stat)
        delta = None
        if mine is not None and theirs is not None:
            if kind == "abs":
                delta = _round(theirs - mine)
            elif mine != 0:
                delta = _round(theirs / mine - 1.0)
        status = outside if delta is None or abs(delta) > tolerance else inside
        return {"name": name, "ours": mine, "simulated": theirs, "delta": delta, "tolerance": tolerance, "kind": kind,
                "status": status}
    gates = [entry(spec, "FAIL", "pass") for spec in GATES]
    warns = [entry(spec, "WARN", "ok") for spec in WARNS]
    for item in warns:
        if item["delta"] is None:
            item["status"] = "n/a"
    return {"gates": gates, "warns": warns, "passed": all(item["status"] == "pass" for item in gates)}


def pool(parts):
    """measure() of many (aim, ball_x, ball_seen, dead, blocks) parts as one: the rows pooled, not the medians.

    Every part is laid after the one before, with its blocks shifted by the rows before it, so no pair, difference or
    count crosses two parts (measure keeps all of them inside a block); `dead` must be given by each. ValueError for
    no part. Used for DFL's matches x seeds."""
    parts = list(parts)
    if not parts:
        raise ValueError("pool: there is no part to measure; give at least one (aim, ball_x, ball_seen, dead, blocks)")
    blocks, offset = [], 0
    for aim, _, _, _, spans in parts:
        blocks.extend((half, start + offset, stop + offset) for half, start, stop in spans)
        offset += len(aim)
    columns = [np.concatenate([np.asarray(part[index]) for part in parts]) for index in range(4)]
    return measure(columns[0], columns[1], columns[2], columns[3], blocks)


def decoded_part(config, inputs, aim, run_module):
    """One (aim, ball_x, ball_seen, dead, blocks) part: `inputs`' ball as the pipeline reads it, `aim`, and `dead`
    where `config` decodes D on `inputs` (run.predict: pseudo-labels, no truth is reachable from the inputs)."""
    from pipeline.evidence import kinematics
    import possession_metrics as pm
    codes, _ = run_module.predict(config, inputs)
    track = kinematics.ball_track(inputs)
    return (np.asarray(aim, dtype=np.float64), np.array(track.xy_m[:, 0]), np.array(track.visible),
            np.asarray(codes) == pm.D, list(inputs.graded))


def ours_stats(config, inputs):
    """measure() on BAR-ATH: its own aim, its ball as served, and `config`'s decoded D (aggregates only)."""
    from pipeline.benchmark import run
    aim, ball_x, seen, dead, blocks = decoded_part(config, inputs, inputs.camera_aim, run)
    return measure(aim, ball_x, seen, dead, blocks)


def simulated_stats(config, profile, seeds, matches, calibration, camera_fit, data_dir, log=print):
    """measure() pooled over `matches` ([(id, DflMatch or None to load)]) x seeds 0 .. seeds - 1 on `profile`'s copies.

    Each (match, seed) is served as the runner serves it (make_atd, simulated_aim, MatchInputs) and decoded by
    `config`; one at a time, keeping only the five arrays of the part."""
    from pipeline.benchmark import dfl, run
    from pipeline.benchmark import inputs as match_inputs
    parts = []
    for match_id, match in matches:
        if match is None:
            match = dfl.load_match(match_id, data_dir=data_dir)
        for seed in range(seeds):
            atd = run.make_atd(match, profile, seed, calibration)
            aim = simulated_aim(match, seed, camera_fit)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir,
                                              camera_aim=aim)
            parts.append(decoded_part(config, inputs, aim, run))
            log("simulated {} seed {}".format(match_id, seed))
    return pool(parts)


def report(profile=DEFAULT_PROFILE, seeds=DEFAULT_SEEDS, match_ids=None, matches=None, calibration=None,
           camera_fit=None, config_path=CONFIG, ours=None, data_dir=None, log=print):
    """The camera's gates on `profile`: a JSON-ready dict of the two measurements, the gates and the WARN.

    `ours` is BAR-ATH's measure() dict, or None to measure it (Inputs(data_dir), the config's decoded D); `matches` is
    {id: DflMatch} to use instead of loading, `calibration` and `camera_fit` stand in for the cached ones. Returns
    {"profile", "seeds", "matches", "config", "camera" (digest and the fit's constants), "ours", "simulated", "gates",
    "warns", "passed"}. run.BenchmarkError for a profile that has no camera to compare (clean) or is not one, and for
    seeds below 1.
    """
    from pipeline.benchmark import run
    from pipeline.config import load_config
    run._check_profile(profile)
    if profile in NO_CAMERA_PROFILES:
        raise run.BenchmarkError("profile {!r} degrades nothing, so its ball has no gaps for the camera to be held "
                                 "against ours; use one of: {}".format(
                                     profile, ", ".join(sorted(set(run.PROFILES) - set(NO_CAMERA_PROFILES)))))
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    config = load_config(config_path)
    if calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    camera_fit = run.camera_fit_for(profile, camera_fit, data_dir)
    if ours is None:
        from pipeline.inputs import Inputs
        ours = ours_stats(config, Inputs(data_dir=data_dir))
    served = [(match_id, None if matches is None else matches[match_id]) for match_id in ids]
    simulated = simulated_stats(config, profile, seeds, served, calibration, camera_fit, data_dir, log=log)
    result = compare(ours, simulated)
    result.update(profile=profile, seeds=seeds, matches=sorted(ids), config=config.stem,
                  camera={"sha256": digest(camera_fit), "fit": {name: getattr(camera_fit, name) for name in CONSTANTS}},
                  ours=ours, simulated=simulated)
    return result


def report_path(data_dir, profile):
    """<data dir>/runs/camera/gates_<profile>.json."""
    from games import DATA_DIR
    return os.path.join(data_dir or DATA_DIR, *REPORT_FOLDER, "gates_{}.json".format(profile))


def _figure(value, signed=False):
    return "n/a" if value is None else ("{:+.4g}" if signed else "{:.4g}").format(value)


def render(result):
    """The printed lines: one per gate and WARN, figures only."""
    lines = ["camera gates on {} ({} seeds, {} matches, {} decoded)".format(
        result["profile"], result["seeds"], len(result["matches"]), result["config"]),
        "camera    {}".format(result["camera"]["sha256"][:12])]
    for kind, items in (("gate", result["gates"]), ("warn", result["warns"])):
        for item in items:
            band = ("+-{}".format(item["tolerance"]) if item["kind"] == "abs"
                    else "+-{:.0%} of ours".format(item["tolerance"]))
            lines.append("{:<5} {:<24} ours {:>8}  sim {:>8}  delta {:>9}  {:<16} {}".format(
                kind, item["name"], _figure(item["ours"]), _figure(item["simulated"]),
                _figure(item["delta"], signed=True), band, item["status"]))
    lines.append("result    {}".format("PASS" if result["passed"] else "FAIL"))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="The simulated camera's fidelity against ours (HEAVY: loads BAR-ATH "
                                                 "and the 7 DFL matches; exit 1 if a gate fails).")
    commands = parser.add_subparsers(dest="command", required=True)
    gates = commands.add_parser("gates", help="the four gates and the WARN, written to runs/camera/")
    gates.add_argument("--profile", default=DEFAULT_PROFILE,
                       help="the profile to measure (default {})".format(DEFAULT_PROFILE))
    gates.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                       help="seeds per match (default {})".format(DEFAULT_SEEDS))
    gates.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    from pipeline.benchmark import run
    from pipeline.config import ConfigError
    try:
        result = report(args.profile, seeds=args.seeds, data_dir=args.data_dir)
        path = report_path(args.data_dir, args.profile)
        run.write_json(path, run.plain(result))
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    for line in render(result):
        print(line)
    print("report    {}".format(run.shown(path)))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
