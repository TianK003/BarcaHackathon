"""Checks of the camera's follower (pipeline/benchmark/camera.py): our aim's fit, its simulation on DFL, its serving.

P09 gives every benchmark match a camera like ours, so what is held against it is that the fit measures the camera it is
given, that the simulated camera is that fit and nothing else, and that it reaches the components only as the inputs'
camera_aim, with no truth in it:

* the smoother is the recursion it says, holding a missing ball and starting again at each block
  (test_camera_lowpass_on_a_hand_worked_series);
* the fit gives back the constants of data made by the follower it fits, of a ball it sees with gaps that the camera
  did not have, and its a, b, phi and sigma are the least squares and the residual statistics of the frames it says
  it used, recomputed here another way (test_camera_fit_recovers_known_constants);
* the digest that reports and selections name the camera by is the eight constants the simulator uses, and nothing of
  the statistics (test_camera_digest_names_the_simulation_constants_only);
* the simulated aim is the follower on the clean ball plus the seeded noise, the same bytes for the same (match, seed),
  NaN off the graded blocks, clipped, and blind to the truth (test_camera_simulation_is_deterministic_per_seed,
  test_camera_simulation_reads_no_truth);
* the gates' statistics are exact on a scene built by hand (test_camera_measure_on_a_hand_built_scene);
* the fidelity gates apply the declared tolerances, the DFL side pools rows and not medians, and the report is the
  pooled measurement of the served matches against ours (test_camera_gates_*);
* the fit, its cache and what it prints are aggregates only (test_camera_fit_prints_aggregates_only);
* the runner and the trainers serve the aim off the clean profile and not on it, the report names the fit, and no
  existing config's score moves (test_runner_serves_camera_aim_off_clean, test_trainer_serves_camera_aim_off_clean).

Everything runs on synthetic data made here and in checks/common.py; none reads data, and no number printed is a figure
of any match.
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import tempfile
import zlib

import numpy as np

from pipeline.benchmark import camera, degrade, score, train_dead_ball, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (leaks, random_match, synthetic_calibration, synthetic_camera_fit,
                                              synthetic_matches, tracking_calibration, yardstick_path)
from pipeline.config import load_config
from pipeline.inputs import Homography
from pipeline.selftest import expect_error, make_synthetic_bundle, quiet, synthetic_inputs, write_text
from starter import load_homography

# Params
# The follower the fit must give back (the brief's constants), and the scene it is drawn on: two blocks of BLOCK_ROWS
# frames. The intercept is the loosest of the four: it is the follower extrapolated to x = 0 from a ball that lives
# around the middle of the pitch, under noise whose long-run standard deviation is sigma / (1 - phi) = 40 m, so its
# error's spread is about 80 m / sqrt(frames): at 600,000 frames, with the fitter's ball gapped as ours is, its rms
# was 1.7% of a and its worst 2.3% over eight seeds (lag and tau exact on all eight), measured while writing this,
# against the 5% allowed. Fewer frames would make the 5% a matter of the seed.
TRUE = {"lag": 30, "tau": 10.0, "a": 5.0, "b": 0.6, "phi": 0.95, "sigma": 2.0}
TOLERANCE = 0.05
BLOCK_ROWS = 300_000
SEPARATOR = 100
PITCH_LENGTH = 105.0
REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# -- the smoother -----------------------------------------------------------------------------------------------------

def test_camera_lowpass_on_a_hand_worked_series():
    """y += alpha (x - y), alpha = 1 / (1 + tau): held across NaN, NaN before the first value, again at each block."""
    nan = math.nan
    got = camera.lowpass([nan, 2.0, nan, 4.0, 10.0, nan], 1.0)             # alpha 0.5
    assert np.array_equal(got, [nan, 2.0, 2.0, 3.0, 6.5, 6.5], equal_nan=True), got
    held = camera.lowpass([1.0, nan, 3.0, nan], 0)                          # tau 0: the values, held
    assert np.array_equal(held, [1.0, 1.0, 3.0, 3.0]), held
    assert np.isnan(camera.lowpass([nan, nan], 5.0)).all(), "no value yet: NaN"
    assert np.array_equal(camera.lowpass([1.0, np.inf, 3.0], 1.0), [1.0, 1.0, 2.0]), "an infinite x is missing"
    # Two blocks: the second starts again from its own first value, and frames outside both are NaN.
    x = [1.0, 3.0, 5.0, 99.0, nan, 8.0, 0.0]
    blocks = [(1, 0, 3), (2, 4, 7)]
    got = camera.lowpass(x, 1.0, blocks=blocks)
    assert np.array_equal(got, [1.0, 2.0, 3.5, nan, nan, 8.0, 4.0], equal_nan=True), got
    # The same as a plain loop, frame by frame, on a long random series with gaps.
    rng = np.random.default_rng([3, 1])
    series = rng.uniform(0, 100, 2000)
    series[rng.random(2000) < 0.3] = nan
    for tau in camera.TAUS:
        assert np.allclose(camera.lowpass(series, tau), plain_lowpass(series, tau), equal_nan=True, rtol=0,
                           atol=1e-9), tau
    for bad in (-1.0, math.nan, True):
        expect_error(ValueError, lambda bad=bad: camera.lowpass([1.0], bad), "lowpass", "tau")
    expect_error(ValueError, lambda: camera.lowpass(np.zeros((2, 2)), 1.0), "lowpass", "one-dimensional")


def plain_lowpass(values, tau, start=0, stop=None):
    """The smoother of the module docstring, one frame at a time over [start, stop): the reference."""
    alpha = 1.0 / (1.0 + tau)
    stop = len(values) if stop is None else stop
    out, state = [], None
    for value in np.asarray(values[start:stop], dtype=np.float64).tolist():
        if math.isfinite(value):
            state = value if state is None else state + alpha * (value - state)
        out.append(math.nan if state is None else state)
    return np.array(out)


# -- the fit ----------------------------------------------------------------------------------------------------------

def follower_scene(seed=0, rows=BLOCK_ROWS):
    """(aim, ball_x, blocks): the aim of TRUE's follower of a ball, and that ball as a tracker sees it, drawn here.

    The ball wanders the pitch as a reflected walk whose speed persists (and now and then a long pass). The camera
    follows all of it: the aim is a + b * the smoothed TRUE ball `lag` frames earlier (NaN for the first `lag` frames of
    a block) plus AR(1) noise that starts each block from its stationary distribution, all drawn one frame at a time.
    The ball returned is the one the fitter is given, as ours: missing in runs of 1 to 16 s, so that its own smoother
    holds a stale value through each gap that the camera never had."""
    rng = np.random.default_rng([seed, 30])
    blocks = [(1, 0, rows), (2, rows + SEPARATOR, 2 * rows + SEPARATOR)]
    n = 2 * rows + SEPARATOR
    ball, aim = np.full(n, np.nan), np.full(n, np.nan)
    for _, start, stop in blocks:
        m = stop - start
        kicks = (rng.random(m) < 0.003) * rng.normal(0.0, 3.0, m)
        steps = (0.12 * rng.standard_normal(m) + kicks).tolist()
        x, v, path = PITCH_LENGTH / 2, 0.0, []
        for step in steps:
            v = 0.97 * v + step
            x += v
            if x < 0:
                x, v = -x, -v
            if x > PITCH_LENGTH:
                x, v = 2 * PITCH_LENGTH - x, -v
            path.append(x)
        path = np.array(path)
        smoothed = plain_lowpass(path, TRUE["tau"])                  # the camera's: the true ball, no gap
        onsets, lengths = rng.random(m) < 0.002, rng.integers(25, 400, m)
        for onset in np.flatnonzero(onsets):
            path[onset:onset + lengths[onset]] = np.nan
        ball[start:stop] = path                                       # the fitter's: ours, with gaps
        z = rng.standard_normal(m).tolist()
        state = TRUE["sigma"] / math.sqrt(1 - TRUE["phi"] ** 2) * z[0]
        noise = [state]
        for value in z[1:]:
            state = TRUE["phi"] * state + TRUE["sigma"] * value
            noise.append(state)
        lag = TRUE["lag"]
        aim[start + lag:stop] = TRUE["a"] + TRUE["b"] * smoothed[:m - lag] + np.array(noise[lag:])
    return aim, ball, blocks


def visible_runs(ball, blocks):
    """(n,) the visible-ball frames in a row ending at each frame, counted inside its block (0 where it is not seen)."""
    runs = [0] * len(ball)
    values = np.asarray(ball).tolist()
    for _, start, stop in blocks:
        count = 0
        for t in range(start, stop):
            count = count + 1 if math.isfinite(values[t]) else 0
            runs[t] = count
    return np.array(runs)


def test_camera_fit_recovers_known_constants():
    """A camera that followed the true ball with lag 30, tau 10, a 5, b 0.6, phi 0.95 and sigma 2, fitted from a ball
    with our kind of gaps, gives back lag and tau exactly and the other four within 5%; a and b are the least squares,
    and phi and sigma the residuals', of the frames the fit used.

    The frames used are each block's t from `lag` on whose aim is there and whose ball at t - lag ends a run of at least
    ceil(3 tau) visible frames (the smoother's memory of the gap has faded; at least the one frame for tau 0). lo and hi
    are the aim's 0.5th and 99.5th percentiles, and the fit is the same on a second call (no randomness)."""
    aim, ball, blocks = follower_scene()
    fitted = camera.fit(aim, ball, blocks)
    assert (fitted.lag, fitted.tau) == (TRUE["lag"], TRUE["tau"]), (fitted.lag, fitted.tau)
    for name in ("a", "b", "phi", "sigma"):
        got = getattr(fitted, name)
        assert abs(got / TRUE[name] - 1.0) <= TOLERANCE, (name, got, TRUE[name])

    # The arithmetic, another way: np.polyfit on the gathered frames, np.corrcoef on consecutive residuals.
    now = np.concatenate([np.arange(start + fitted.lag, stop) for _, start, stop in blocks])
    then = now - fitted.lag
    settled = max(1, math.ceil(3 * fitted.tau))
    used = np.isfinite(aim[now]) & (visible_runs(ball, blocks)[then] >= settled)
    assert used.sum() < (np.isfinite(aim[now]) & np.isfinite(ball[then])).sum(), "the frames after a gap are left out"
    smoothed = np.full(len(ball), np.nan)
    for _, start, stop in blocks:
        smoothed[start:stop] = plain_lowpass(ball, fitted.tau, start, stop)
    x, y = smoothed[then[used]], aim[now[used]]
    slope, intercept = np.polyfit(x, y, 1)
    assert abs(fitted.b - slope) < 1e-8 and abs(fitted.a - intercept) < 1e-7, (fitted.a, fitted.b, intercept, slope)
    residuals = y - (intercept + slope * x)
    following = np.flatnonzero(now[used][1:] == now[used][:-1] + 1)
    phi = np.corrcoef(residuals[following], residuals[following + 1])[0, 1]
    assert abs(fitted.phi - phi) < 1e-8, (fitted.phi, phi)
    assert abs(fitted.sigma - residuals.std() * math.sqrt(1 - phi * phi)) < 1e-8
    inside = np.concatenate([aim[start:stop] for _, start, stop in blocks])
    lo, hi = np.percentile(inside[np.isfinite(inside)], [0.5, 99.5])
    assert abs(fitted.lo - lo) < 1e-8 and abs(fitted.hi - hi) < 1e-8, (fitted.lo, fitted.hi, lo, hi)
    assert fitted.stats["fit_frames"] == int(used.sum()), fitted.stats["fit_frames"]
    small = follower_scene(seed=1, rows=20_000)                 # determinism needs no long scene
    assert camera.fit(*small) == camera.fit(*small), "the fit is deterministic"


def test_camera_digest_names_the_simulation_constants_only():
    """digest(fit) is the sha256 of the eight constants the simulator uses, as canonical JSON: another constant is
    another digest, other statistics are not; and so a run with a fit that differs only in its stats writes the same
    reports, to the byte."""
    fit = synthetic_camera_fit()
    constants = {name: getattr(fit, name) for name in ("lag", "tau", "a", "b", "phi", "sigma", "lo", "hi")}
    canonical = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    assert camera.digest(fit) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    restated = dataclasses.replace(fit, stats={"peak_lag": 12, "median_err": 3.25, "another": None})
    assert camera.digest(restated) == camera.digest(fit), "the statistics are not the camera"
    for name, value in (("lag", 6), ("tau", 4.0), ("a", 4.5), ("b", 0.8), ("phi", 0.7), ("sigma", 1.25),
                        ("lo", 11.0), ("hi", 89.0)):
        assert camera.digest(dataclasses.replace(fit, **{name: value})) != camera.digest(fit), name
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        reports = [[entry["report"] for entry in benchmark_run.evaluate_configs(
            [yardstick_path(50)], profile="gaps", seeds=1, matches=matches, calibration=cal, camera_fit=one,
            data_dir=root, log=quiet)["configs"]] for one in (fit, restated)]
    assert json.dumps(reports[0], sort_keys=True) == json.dumps(reports[1], sort_keys=True)


# -- the simulator ----------------------------------------------------------------------------------------------------

def test_camera_simulation_is_deterministic_per_seed():
    """The same (match, seed, fit) gives the same bytes; another seed or match id other noise; NaN off the graded
    blocks, finite on them and inside [lo, hi]; without noise it is the follower on the clean ball, and its noise is
    the AR(1) of default_rng([seed, 9009, crc32(match id)])."""
    match, fit = random_match("RND001"), synthetic_camera_fit()
    first = camera.simulated_aim(match, 0, fit)
    assert first.dtype == np.float64 and first.shape == (match.n_frames,)
    assert camera.simulated_aim(match, 0, fit).tobytes() == first.tobytes(), "the same seed, the same bytes"
    assert camera.simulated_aim(match, 1, fit).tobytes() != first.tobytes(), "another seed, other noise"
    other = dataclasses.replace(match, match_id="RND777")
    assert camera.simulated_aim(other, 0, fit).tobytes() != first.tobytes(), "another match, other noise"
    graded = np.zeros(match.n_frames, dtype=bool)
    for _, start, stop in match.graded:
        graded[start:stop] = True
    assert np.isnan(first[~graded]).all() and np.isfinite(first[graded]).all()
    assert first[graded].min() >= fit.lo and first[graded].max() <= fit.hi, (first[graded].min(), first[graded].max())
    narrow = dataclasses.replace(fit, lo=45.0, hi=60.0)        # random_match's ball, smoothed, stays near the middle
    clipped = camera.simulated_aim(match, 0, narrow)[graded]
    assert clipped.min() == 45.0 and clipped.max() == 60.0, "the aim is clipped to [lo, hi], and reaches both here"

    # No noise and no clip: the follower, worked out here on the clean ball (the block's first smoothed ball before
    # `lag` frames have passed, and before its first ball is seen).
    quiet_fit = dataclasses.replace(fit, sigma=0.0, lo=-1e6, hi=1e6)
    follower = camera.simulated_aim(match, 0, quiet_fit)
    expected = np.full(match.n_frames, np.nan)
    ball = np.array(match.ball_xy[:, 0]) * match.pitch[0]
    for _, start, stop in match.graded:
        smoothed = plain_lowpass(ball, fit.tau, start, stop)
        first_seen = np.flatnonzero(np.isfinite(smoothed))[0]
        smoothed[:first_seen] = smoothed[first_seen]
        for t in range(stop - start):
            expected[start + t] = fit.a + fit.b * smoothed[max(t - fit.lag, 0)]
    assert np.allclose(follower, expected, equal_nan=True, rtol=0, atol=1e-9)
    # The noise: what the wide clip leaves of the aim beyond the follower is the AR(1) of the declared generator.
    wide = camera.simulated_aim(match, 4, dataclasses.replace(fit, lo=-1e6, hi=1e6))
    rng = np.random.default_rng([4, 9009, zlib.crc32(b"RND001")])
    for _, start, stop in match.graded:
        z = rng.standard_normal(stop - start)
        noise = [fit.sigma / math.sqrt(1 - fit.phi ** 2) * z[0]]
        for value in z[1:]:
            noise.append(fit.phi * noise[-1] + fit.sigma * value)
        assert np.allclose(wide[start:stop] - expected[start:stop], noise, rtol=0, atol=1e-9)
    for seed, fragment in ((-1, "seed"), (True, "seed"), (1.5, "seed")):
        expect_error(ValueError, lambda seed=seed: camera.simulated_aim(match, seed, fit), "simulated_aim", fragment)
    expect_error(ValueError, lambda: camera.simulated_aim(match, 0, {"lag": 5}), "simulated_aim", "CameraFit")
    for name, value in (("phi", 1.0), ("sigma", -1.0), ("lag", 81), ("tau", -2.0), ("lo", 95.0)):
        expect_error(ValueError, lambda name=name, value=value: dataclasses.replace(fit, **{name: value}),
                     "CameraFit", name)


def test_camera_simulation_reads_no_truth():
    """The aim depends on the ball's positions and on nothing of the flags: other flags, the same bytes; another ball,
    another aim; and the inputs it is served in hold no truth (common.leaks)."""
    match, fit = random_match("RND001"), synthetic_camera_fit()
    aim = camera.simulated_aim(match, 2, fit)
    flipped = dataclasses.replace(match, ball_status=np.ascontiguousarray(1 - np.array(match.ball_status)),
                                  ball_possession=np.ascontiguousarray(1 - np.array(match.ball_possession)))
    assert camera.simulated_aim(flipped, 2, fit).tobytes() == aim.tobytes(), "the flags changed the aim"
    moved = np.array(match.ball_xy)
    moved[:, 0] = 1.0 - moved[:, 0]
    moved.setflags(write=False)
    assert camera.simulated_aim(dataclasses.replace(match, ball_xy=moved), 2, fit).tobytes() != aim.tobytes()
    inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match), profile="gaps", seed=2, camera_aim=aim)
    inputs.camera_aim
    assert leaks(inputs, match) == [], leaks(inputs, match)


# -- the statistics ---------------------------------------------------------------------------------------------------

def hand_built_scene():
    """(aim, ball_x, seen, dead) of 120 frames, every value a multiple of 0.25 so that every difference is exact.

    The aim starts at 50 m and steps 0.25 m a frame on the dead frames 20-39 (6.25 m/s at 25 Hz), 1.0 m on 60-69
    (25 m/s) and 0.5 m on the others (12.5 m/s), up or down by a fixed coin, so that no stretch of it is a straight line
    that a long lag could correlate perfectly with. The ball is the aim 3 frames later less 2 m (frames 0-116; frames
    117-119 have a ball at 0 m), and it is not seen on 24, 25, 61, 62 and 63 (NaN there, as the served ball is). Worked
    out by hand: the peak lag is 3, every error there is 2 m, on 117 - 5 = 112 frames; the dead frames' median speed is
    6.25 and the alive ones' 12.5 (89 at 12.5, 10 at 25), a ratio of 0.5; the gap frames' speeds are 6.25, 6.25, 25, 25
    and 25 (median 25) and the visible ones' median 12.5 (18 at 6.25, 89 at 12.5, 7 at 25), a ratio of 2.0."""
    signs = np.where(np.random.default_rng([12, 0]).random(120) < 0.5, -1.0, 1.0)
    sizes = np.full(120, 0.5)
    sizes[20:40], sizes[60:70], sizes[0] = 0.25, 1.0, 0.0
    aim = 50.0 + np.cumsum(signs * sizes)
    ball = np.zeros(120)
    ball[:117] = aim[3:] - 2.0
    seen = np.ones(120, dtype=bool)
    seen[[24, 25, 61, 62, 63]] = False
    ball[~seen] = np.nan
    dead = np.zeros(120, dtype=bool)
    dead[20:40] = True
    return aim, ball, seen, dead


def test_camera_measure_on_a_hand_built_scene():
    """peak_lag, median_err and both ratios exactly as worked out by hand; the same with the scene in two blocks and
    other frames between them, which `blocks` keeps out; dead None leaves the dead/alive entries None; and `blocks` is
    required, so that no frame outside the graded ones can be counted."""
    aim, ball, seen, dead = hand_built_scene()
    whole = [(1, 0, 120)]
    stats = camera.measure(aim, ball, seen, dead, whole)
    expected = {"peak_lag": 3, "pairs": 112, "median_err": 2.0, "p90_err": 2.0, "speed_dead_mps": 6.25,
                "speed_alive_mps": 12.5, "speed_gap_mps": 25.0, "speed_visible_mps": 12.5,
                "speed_ratio_dead_alive": 0.5, "speed_ratio_gap_visible": 2.0}
    assert {name: stats[name] for name in expected} == expected, stats
    assert abs(stats["peak_corr"] - 1.0) < 1e-9, stats["peak_corr"]
    assert set(stats) == set(expected) | {"peak_corr"}, sorted(stats)

    # Twice, in two blocks with five frames between them that would spoil every statistic if they were read.
    between = 5
    doubled = [np.concatenate([part, filler, part]) for part, filler in (
        (aim, np.full(between, 500.0)), (ball, np.full(between, -400.0)), (seen, np.ones(between, dtype=bool)),
        (dead, np.ones(between, dtype=bool)))]
    blocks = [(1, 0, 120), (2, 120 + between, 240 + between)]
    twice = camera.measure(*doubled, blocks=blocks)
    assert {name: twice[name] for name in expected} == dict(expected, pairs=224), twice
    assert camera.measure(*doubled, blocks=[(1, 0, 245)]) != twice, "one block over all: the frames between are read"

    blind = camera.measure(aim, ball, seen, None, whole)
    assert blind["speed_dead_mps"] is None and blind["speed_alive_mps"] is None
    assert blind["speed_ratio_dead_alive"] is None and blind["speed_ratio_gap_visible"] == 2.0, blind
    empty = camera.measure(np.full(5, np.nan), np.full(5, np.nan), np.zeros(5, dtype=bool), np.zeros(5, dtype=bool),
                           [(1, 0, 5)])
    assert empty["peak_lag"] is None and empty["median_err"] is None and empty["speed_ratio_dead_alive"] is None
    expect_error(ValueError, lambda: camera.measure(aim, ball[:-1], seen, dead, whole), "measure", "same length")
    expect_error(ValueError, lambda: camera.measure(aim, ball, seen, dead[:-1], whole), "measure", "dead")
    # The blocks are required: an aim off the graded frames must never be counted for want of them.
    expect_error(TypeError, lambda: camera.measure(aim, ball, seen, dead), "blocks")
    expect_error(ValueError, lambda: camera.measure(aim, ball, seen, dead, []), "measure", "blocks")


# -- the cache, and what it says ---------------------------------------------------------------------------------------

def write_homography(root, aim, pitch_length):
    """root/tactical/BAR-ATH_tactical_homography.csv in the real format: per frame, a scale and a translation that put
    the image centre over pitch x = aim / pitch_length (and y = 0.5)."""
    rows = ["frame," + ",".join(load_homography.MATRIX_COLUMNS) + ",reprojection_error,valid"]
    for frame, value in enumerate(aim):
        u = value / pitch_length
        matrix = (1000.0, 0.0, 960.0 - 1000.0 * u, 0.0, 1000.0, 540.0 - 500.0, 0.0, 0.0, 1.0)
        rows.append("{},{},0.25,1".format(frame, ",".join(repr(float(number)) for number in matrix)))
    return write_text(os.path.join(root, "tactical"), "BAR-ATH_tactical_homography.csv", "\n".join(rows) + "\n")


def bundle_inputs(root):
    """The synthetic bundle (pipeline/selftest.py) with a homography whose aim follows its ball: an Inputs that
    load_fit can be given in place of ours. The homography is read without the 199,319-row check, which is BAR-ATH's."""
    make_synthetic_bundle(root)
    inputs = synthetic_inputs(root)
    ball = inputs.metres(inputs.atd.ball_xy)[:, 0]
    rng = np.random.default_rng([5, 9])
    aim = np.full(inputs.n_frames, np.nan)
    for _, start, stop in inputs.graded:
        smoothed = plain_lowpass(ball, 2.0, start, stop)
        smoothed[np.isnan(smoothed)] = 52.0
        for t in range(stop - start):
            aim[start + t] = 10.0 + 0.8 * smoothed[max(t - 4, 0)] + rng.normal(0.0, 0.5)
    write_homography(root, aim, inputs.pitch[0])
    matrices, error, valid = load_homography.load(data_dir=root)
    inputs.__dict__["homography"] = Homography(matrices=matrices, error=error, valid=valid)
    return inputs


def leaves(value, path="fit"):
    """[(path, leaf)] of a JSON document: every value that is not a dict."""
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in leaves(child, path + "." + key)]
    return [(path, value)]


def test_camera_fit_prints_aggregates_only():
    """load_fit prints nothing, and what it keeps is eight constants and a flat dict of scalars: no list, no array, no
    frame. A second load reads the cache (no fit); rebuild fits again to the same digest; an edited homography is
    another key; a damaged file is fitted again, not an error."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bundle_inputs(root)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            fitted = camera.load_fit(data_dir=root, inputs=inputs)
        assert shown.getvalue() == "", "load_fit printed: " + shown.getvalue()
        assert "array" not in repr(fitted) and len(repr(fitted)) < 2000, repr(fitted)
        assert all(value is None or isinstance(value, (bool, int, float, str)) for value in fitted.stats.values())
        folder = os.path.join(root, *camera.CACHE_FOLDER)
        keys = os.listdir(folder)
        assert len(keys) == 1 and os.listdir(os.path.join(folder, keys[0])) == [camera.FIT_FILE], keys
        path = os.path.join(folder, keys[0], camera.FIT_FILE)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        assert sorted(document) == ["fit", "key"], sorted(document)
        kept = leaves(document["fit"])
        assert all(value is None or isinstance(value, (bool, int, float, str)) for _, value in kept), kept
        assert len(kept) < 40, "{} values: a fit is aggregates, not frames".format(len(kept))
        assert sorted(document["fit"]) == sorted(item.name for item in dataclasses.fields(camera.CameraFit))
        key_leaves = leaves(document["key"], "key")
        assert not any(isinstance(value, list) and len(value) > 10 for _, value in key_leaves), key_leaves
        # The code the fit depends on is in the key, each file by its text with CRLF read as LF.
        sources = ("pipeline/benchmark/camera.py", "pipeline/inputs/camera.py", "pipeline/evidence/kinematics.py",
                   "starter/load_homography.py", "pipeline/inputs/__init__.py")
        assert sorted(document["key"]["source_sha256"]) == sorted(sources), sorted(document["key"]["source_sha256"])
        for name in sources:
            with open(os.path.join(REPOSITORY, *name.split("/")), "rb") as handle:
                text = handle.read().replace(b"\r\n", b"\n")
            assert document["key"]["source_sha256"][name] == hashlib.sha256(text).hexdigest(), name
        assert len(camera.digest(fitted)) == 64 and camera.digest(fitted) != camera.digest(
            dataclasses.replace(fitted, a=fitted.a + 1.0))

        # Read back, not fitted: a spy on fit() fails the check if it is called.
        real_fit = camera.fit

        def refused(*arguments, **keywords):
            raise AssertionError("the cache was not read")
        camera.fit = refused
        try:
            again = camera.load_fit(data_dir=root, inputs=inputs)
        finally:
            camera.fit = real_fit
        assert again == fitted
        rebuilt = camera.load_fit(data_dir=root, inputs=inputs, rebuild=True)
        assert rebuilt == fitted, "a refit of the same sources is the same fit"

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        assert camera.load_fit(data_dir=root, inputs=inputs) == fitted
        with open(os.path.join(root, "tactical", "BAR-ATH_tactical_homography.csv"), "a", encoding="utf-8") as handle:
            handle.write("\n")
        camera.load_fit(data_dir=root, inputs=inputs)
        assert len(os.listdir(folder)) == 2, "an edited source is another key, and another folder"


# -- the serving ------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def spying(module, name, replacement):
    """module.name replaced while the block runs, and restored afterwards."""
    real = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield real
    finally:
        setattr(module, name, real)


def served_aims(run):
    """Run `run()` with the runner's _codes spied on: [(match id, seed, the aim served or None)], in order."""
    served = []
    real = benchmark_run._codes

    def spy(config, inputs):
        try:
            aim = np.array(inputs.camera_aim)
        except match_inputs.AssetUnavailable:
            aim = None
        served.append((inputs.game, inputs.seed, aim))
        return real(config, inputs)
    with spying(benchmark_run, "_codes", spy):
        result = run()
    return result, served


def gaps_halves(match, config_path, cal, seed, data_dir):
    """[HalfScore] of one config on one match under gaps, served WITHOUT a camera: degrade, predict, score_halves."""
    atd, _ = degrade.degrade(degrade.tracking_view(match), "gaps", cal, seed)
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed, data_dir=data_dir)
    codes, _ = benchmark_run.predict(load_config(config_path), inputs)
    return score.score_halves(truth.truth_codes(match)[0], codes, match.graded, match.match_id, seed)


def test_runner_serves_camera_aim_off_clean():
    """Under gaps every served match has camera_aim = simulated_aim(match, seed, fit) and the report and printout name
    the fit's digest; under clean none has one and the report says None. The scores are the ones a run without a camera
    gives, byte for byte: no existing component reads it. None given, the fit is camera.load_fit(data_dir)'s; a fit
    given to clean, or something that is not a CameraFit, is refused before a match is loaded."""
    matches, cal, fit = synthetic_matches(), synthetic_calibration(), synthetic_camera_fit()
    paths = [yardstick_path(50), yardstick_path(0)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result, served = served_aims(lambda: benchmark_run.evaluate_configs(
            paths, profile="gaps", seeds=2, matches=matches, calibration=cal, camera_fit=fit, data_dir=root,
            log=lines.append))
        assert len(served) == len(paths) * len(matches) * 2, len(served)
        for match_id, seed, aim in served:
            assert aim is not None, (match_id, seed)
            assert aim.tobytes() == camera.simulated_aim(matches[match_id], seed, fit).tobytes(), (match_id, seed)
        assert result["camera"] == {"sha256": camera.digest(fit)}
        for entry, width in zip(result["configs"], (50, 0)):
            report = entry["report"]
            assert report["camera"] == {"sha256": camera.digest(fit)}, report["camera"]
            expected = [half for match in matches.values() for seed in (0, 1)
                        for half in gaps_halves(match, yardstick_path(width), cal, seed, root)]
            assert [benchmark_run.plain(dataclasses.asdict(half)) for half in expected] == report["halves"], width
        assert any(line.startswith("camera") and camera.digest(fit)[:12] in line for line in lines), lines

        # Another fit: the same reports but for the name of the fit.
        other = benchmark_run.evaluate_configs(paths, profile="gaps", seeds=2, matches=matches, calibration=cal,
                                               camera_fit=synthetic_camera_fit(lag=40, sigma=6.0), data_dir=root,
                                               log=quiet)
        for mine, theirs in zip(result["configs"], other["configs"]):
            assert mine["report"]["camera"] != theirs["report"]["camera"]
            assert dict(mine["report"], camera=None) == dict(theirs["report"], camera=None), mine["stem"]

        clean, served = served_aims(lambda: benchmark_run.evaluate_configs(
            paths[:1], profile="clean", matches=matches, data_dir=root, log=quiet))
        assert served and all(aim is None for _, _, aim in served), "clean serves no camera"
        assert clean["camera"] is None and clean["configs"][0]["report"]["camera"] is None

        # None given: load_fit(data_dir), once, before the first match.
        asked = []

        def loaded(data_dir=None, **keywords):
            asked.append(data_dir)
            return fit
        with spying(camera, "load_fit", loaded):
            loaded_run = benchmark_run.evaluate_configs(paths[:1], profile="gaps", seeds=1, matches=matches,
                                                        calibration=cal, data_dir=root, log=quiet)
        assert asked == [root] and loaded_run["camera"] == {"sha256": camera.digest(fit)}, asked

        refused = benchmark_run.BenchmarkError
        expect_error(refused, lambda: benchmark_run.evaluate_configs(
            paths[:1], profile="clean", matches=matches, camera_fit=fit, data_dir=root, log=quiet), "clean", "camera")
        expect_error(refused, lambda: benchmark_run.evaluate_configs(
            paths[:1], profile="gaps", matches=matches, calibration=cal, camera_fit=object(), data_dir=root,
            log=quiet), "camera_fit", "CameraFit")
        assert not os.path.exists(os.path.join(root, "runs")), "nothing is written by a run that is not written"


def test_trainer_serves_camera_aim_off_clean():
    """train_dead_ball.served_inputs (the trainers' and conventions.py's way to a served match) does as the runner does:
    the simulated aim off clean, none on it, load_fit(data_dir) when none is given; gather hands its fit down."""
    match, cal, fit = random_match("RND001"), synthetic_calibration(), synthetic_camera_fit()
    served = train_dead_ball.served_inputs(match, "gaps", 1, cal, None, camera_fit=fit)
    assert served.camera_aim.tobytes() == camera.simulated_aim(match, 1, fit).tobytes()
    bare = train_dead_ball.served_inputs(match, "clean", 0, None, None)
    expect_error(match_inputs.AssetUnavailable, lambda: bare.camera_aim, "camera_aim")
    asked = []

    def loaded(data_dir=None, **keywords):
        asked.append(data_dir)
        return fit
    with spying(camera, "load_fit", loaded):
        found = train_dead_ball.served_inputs(match, "gaps", 1, cal, "somewhere")
    assert asked == ["somewhere"] and found.camera_aim.tobytes() == served.camera_aim.tobytes(), asked
    expect_error(ValueError, lambda: train_dead_ball.served_inputs(match, "clean", 0, None, None, camera_fit=fit),
                 "clean", "camera")

    # gather loads nothing when it is given a fit, and serves each match and seed its own aim.
    seen = []
    real = train_dead_ball.served_inputs

    def spy(match_, profile, seed, calibration, data_dir, camera_fit=None):
        inputs = real(match_, profile, seed, calibration, data_dir, camera_fit=camera_fit)
        seen.append((match_.match_id, seed, inputs.camera_aim.tobytes()))
        return inputs

    def refused(*arguments, **keywords):
        raise AssertionError("gather loaded a fit although it was given one")
    with spying(camera, "load_fit", refused), spying(train_dead_ball, "served_inputs", spy):
        train_dead_ball.gather("gaps", 2, 7, matches=synthetic_matches(), calibration=cal, camera_fit=fit)
    expected = [(match_id, seed, camera.simulated_aim(other, seed, fit).tobytes())
                for match_id, other in sorted(synthetic_matches().items()) for seed in (0, 1)]
    assert seen == expected, [(match_id, seed) for match_id, seed, _ in seen]


# -- the fidelity gates -----------------------------------------------------------------------------------------------

def test_camera_gates_apply_the_declared_tolerances():
    """Each gate on either side of its tolerance (and exactly on it) gives pass / FAIL as declared: peak_lag +-10
    frames, peak_corr +-0.05, median_err +-3.0 m, speed_ratio_dead_alive +-25% of ours; an unmeasured statistic fails
    its gate; the WARN (speed_ratio_gap_visible) never fails, whatever its figures; and a failed gate sets passed to
    False."""
    ours = {"peak_lag": 40, "peak_corr": 0.50, "median_err": 8.0, "speed_ratio_dead_alive": 0.80,
            "speed_ratio_gap_visible": 0.58}
    base = dict(ours)

    def status(name, **changed):
        result = camera.compare(ours, {**base, **changed})
        return {item["name"]: item["status"] for item in result["gates"]}[name], result

    assert [(g[0], g[2], g[3]) for g in camera.GATES] == [
        ("peak_lag", "abs", 10.0), ("peak_corr", "abs", 0.05), ("median_err", "abs", 3.0),
        ("speed_ratio_dead_alive", "rel", 0.25)], camera.GATES
    for name, inside, edge, outside in (
            ("peak_lag", 49, 50, 51),
            ("peak_corr", 0.54, 0.55, 0.56),
            ("median_err", 10.9, 11.0, 11.1),
            ("speed_ratio_dead_alive", 0.99, 1.0, 1.01)):
        for value in (inside, edge):                       # on the tolerance passes: the rounding makes it exact
            assert status(name, **{name: value})[0] == "pass", (name, value)
        assert status(name, **{name: outside})[0] == "FAIL", (name, outside)
    for name, below in (("peak_lag", 29), ("peak_corr", 0.44), ("median_err", 4.9), ("speed_ratio_dead_alive", 0.59)):
        assert status(name, **{name: below})[0] == "FAIL", (name, below)       # either side of ours
    assert status("speed_ratio_dead_alive", speed_ratio_dead_alive=0.6)[0] == "pass"    # -25% exactly

    state, result = status("peak_lag")
    assert state == "pass" and result["passed"] and all(g["status"] == "pass" for g in result["gates"]), result
    assert result["warns"][0]["status"] == "ok" and result["warns"][0]["delta"] == 0.0, result["warns"]
    state, result = status("peak_lag", peak_lag=80)
    assert not result["passed"], "one failed gate fails the whole"

    # None on either side fails (nothing was shown to hold); the WARN is n/a and does not fail.
    for name in ("peak_lag", "peak_corr", "median_err", "speed_ratio_dead_alive"):
        assert status(name, **{name: None})[0] == "FAIL", name
        assert camera.compare({**ours, name: None}, base)["gates"][[g[0] for g in camera.GATES].index(name)][
            "status"] == "FAIL", name
    assert camera.compare({**ours, "speed_ratio_dead_alive": 0.0}, base)["passed"] is False, "ours 0: no ratio"
    # The WARN never fails: far outside its band, it is a WARN, and passed is untouched; not measurable, it is n/a.
    far = camera.compare(ours, {**base, "speed_ratio_gap_visible": 5.0})
    assert far["warns"][0]["status"] == "WARN" and far["passed"], far
    gone = camera.compare(ours, {**base, "speed_ratio_gap_visible": None})
    assert gone["warns"][0]["status"] == "n/a" and gone["passed"], gone


def test_camera_gates_pool_the_rows_of_every_match_and_seed():
    """pool measures one long series in which each part keeps its own blocks: the speed's median is that of all the rows
    (a part of many slow frames and a part of few fast ones give the slow one, not the middle of two medians), and a
    single part gives exactly measure()'s own figures; no part is refused."""
    n = 200
    slow = np.arange(n) * 0.04                 # 1 m/s at 25 Hz
    fast = np.arange(40) * 0.4                 # 10 m/s
    seen_slow, seen_fast = np.ones(n, dtype=bool), np.ones(40, dtype=bool)
    dead_slow, dead_fast = np.zeros(n, dtype=bool), np.zeros(40, dtype=bool)
    parts = [(slow, slow, seen_slow, dead_slow, [(1, 0, n)]), (fast, fast, seen_fast, dead_fast, [(1, 0, 40)])]
    pooled = camera.pool(parts)
    assert abs(pooled["speed_alive_mps"] - 1.0) < 1e-9, pooled["speed_alive_mps"]    # not the 5.5 of the two medians
    alone = camera.pool(parts[:1])
    assert alone == camera.measure(slow, slow, seen_slow, dead_slow, [(1, 0, n)]), alone
    expect_error(ValueError, lambda: camera.pool([]), "part")


def test_camera_gates_report_end_to_end_on_synthetic_matches():
    """report() on two synthetic matches: ours given, the simulated side pooled over matches x seeds from the served
    inputs, the gates and the WARN in the result, the file written by main's path, clean refused, no truth anywhere."""
    matches, cal, fit = synthetic_matches(), tracking_calibration(), synthetic_camera_fit()
    ours = {"peak_lag": 5, "peak_corr": 0.9, "median_err": 2.0, "speed_ratio_dead_alive": 1.0,
            "speed_ratio_gap_visible": 1.0}
    path = yardstick_path(50)
    lines = []
    result = camera.report("tracking", seeds=2, matches=matches, calibration=cal, camera_fit=fit, config_path=path,
                           ours=ours, log=lines.append)
    assert result["matches"] == sorted(matches) and result["seeds"] == 2 and result["profile"] == "tracking"
    assert result["camera"]["sha256"] == camera.digest(fit), result["camera"]
    assert result["ours"] == ours and result["gates"] == camera.compare(ours, result["simulated"])["gates"]
    assert len(lines) == len(matches) * 2, lines
    again = camera.report("tracking", seeds=2, matches=matches, calibration=cal, camera_fit=fit, config_path=path,
                          ours=ours, log=quiet)
    assert again == result, "the same bytes for the same inputs"
    one = camera.report("tracking", seeds=1, matches=matches, calibration=cal, camera_fit=fit, config_path=path,
                        ours=ours, log=quiet)
    assert one["simulated"] != result["simulated"], "the seeds are pooled in"
    text = "\n".join(camera.render(result))
    assert ("PASS" in text or "FAIL" in text) and "speed_ratio_gap_visible" in text, text
    with tempfile.TemporaryDirectory() as root:
        target = camera.report_path(root, "tracking")
        assert target.replace("\\", "/").endswith("/runs/camera/gates_tracking.json"), target
    expect_error(benchmark_run.BenchmarkError, lambda: camera.report(
        "clean", matches=matches, camera_fit=None, config_path=path, ours=ours, log=quiet), "clean")
    expect_error(benchmark_run.BenchmarkError, lambda: camera.report(
        "tracking", seeds=0, matches=matches, calibration=cal, camera_fit=fit, config_path=path, ours=ours,
        log=quiet), "seeds")
