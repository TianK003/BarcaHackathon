"""Checks of the gap-onset hazard (pipeline/benchmark/hazard.py): features, logistic fit, AUC and the mask sampler.

Every check runs on arrays made here, so none reads data and none prints a row of any match. The fit and the sampler are
checked against what the maths says, not against numbers the code printed once: the penalised optimum has a zero
gradient, a constant hazard makes a renewal process with a known rate, and the sampler's draws are scripted so that
every index of the inverse transform can be worked out by hand.
"""

import math

import numpy as np

from pipeline.benchmark import hazard
from pipeline.selftest import expect_error

# Params
PITCH = (105.0, 68.0)    # metres, length then width
FPS = 25
UNIT_HAZARD = -math.expm1(-1.0)   # the probability whose hazard, -log(1 - p), is exactly 1 per frame
# Constant-hazard run (test_sample_mask_deterministic_and_rate): 20 blocks of 100,000 frames, p = 0.002, d = 20 frames.
RATE_P, RATE_D, RATE_BLOCKS, RATE_FRAMES = 0.002, 20, 20, 100_000


# -- helpers ----------------------------------------------------------------------------------------------------------

def features_of(ball, players=None, pitch=PITCH, fps=FPS, block_start=0, kickoff=0):
    """hazard.features on a list of (x, y) ball points; one absent player unless `players` (n, P, 2) is given."""
    ball = np.array(ball, dtype=float).reshape(-1, 2)
    if players is None:
        players = np.full((len(ball), 1, 2), np.nan)
    return hazard.features(ball, np.asarray(players, dtype=float), pitch, fps, block_start, kickoff)


def column(table, name):
    return table[:, hazard.FEATURES.index(name)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gradient(model, X, y, l2):
    """The gradient of the L2-penalised negative log-likelihood at `model`, in the standardised space it was fitted in.

    The intercept is the first entry and carries no penalty, which is the convention hazard.fit_logistic documents.
    """
    Z = (X - model.mean) / model.scale
    miss = sigmoid(Z @ model.coef + model.intercept) - y
    return np.concatenate([[miss.sum()], Z.T @ miss + l2 * model.coef])


def synthetic_rows(seed, n, coef, intercept, means, spreads):
    """n rows of normal features and 0/1 outcomes from a known logistic model (in the features' own units)."""
    rng = np.random.default_rng(seed)
    means, spreads, coef = np.asarray(means), np.asarray(spreads), np.asarray(coef)
    X = means + spreads * rng.normal(size=(n, len(coef)))
    y = (rng.uniform(size=n) < sigmoid(intercept + (X - means) @ coef)).astype(float)
    return X, y


def rare_flag_rows(seed, n, prevalence, odds_ratio, events=147):
    """Two continuous features and one rare 0/1 flag with the given odds ratio, in n rows with about `events` onsets.

    The shape of our own data: 147 onsets in the ball-visible frames of one match, a few of them in a flagged sliver
    of the rows. The intercept is found by bisection so that the expected number of onsets is `events`. Returns the
    features (flag last), the outcomes, and the true probabilities, whose information gives the sampling error.
    """
    rng = np.random.default_rng(seed)
    flag = (rng.uniform(size=n) < prevalence).astype(float)
    continuous = rng.normal(size=(n, 2))
    linear = 0.5 * continuous[:, 0] - 0.3 * continuous[:, 1] + math.log(odds_ratio) * flag
    low, high = -20.0, 5.0
    for _ in range(80):   # the expected count rises with the intercept
        middle = (low + high) / 2.0
        low, high = (low, middle) if sigmoid(middle + linear).sum() > events else (middle, high)
    p = sigmoid(middle + linear)
    return np.column_stack([continuous, flag]), (rng.uniform(size=n) < p).astype(float), p


def sampling_errors(X, p):
    """Standard errors of the three raw coefficients at the TRUE probabilities p, from the information X'WX.

    Taken at the truth and not at the estimate on purpose: at a wild estimate (a coefficient of -88,000) the curvature
    vanishes and the errors explode, which would let exactly the failure this is meant to catch pass.
    """
    design = np.column_stack([np.ones(len(X)), X])
    information = (design.T * (p * (1.0 - p))) @ design
    return np.sqrt(np.diag(np.linalg.inv(information)))[1:]


class ScriptedRng:
    """Stands in for a Generator: exponential() returns the scripted draws (then 1e9, which ends any block), choice()
    takes the first option; every call is logged, so a check can see the order in which the sampler draws."""

    def __init__(self, draws):
        self.draws = list(draws)
        self.log = []

    def exponential(self, scale=1.0):
        self.log.append(("exponential", scale))
        return self.draws.pop(0) if self.draws else 1e9

    def choice(self, options):
        self.log.append(("choice", [int(value) for value in options]))
        return options[0]


class CountingRng:
    """Wraps a Generator and draws exactly what it would alone, counting the gap lengths chosen (one per onset)."""

    def __init__(self, rng):
        self.rng = rng
        self.onsets = 0

    def exponential(self, scale=1.0):
        return self.rng.exponential(scale)

    def choice(self, options):
        self.onsets += 1
        return self.rng.choice(options)


def runs_of_true(mask):
    """[(start, stop)] of every maximal run of True in a bool array."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def constant_durations(length):
    """A durations table, in frames, where every class and third has the one length."""
    return [[np.array([length]) for _ in range(3)] for _ in range(2)]


# -- features ---------------------------------------------------------------------------------------------------------

def test_hazard_features_are_the_six_named_columns():
    assert hazard.FEATURES == ("line_distance_m", "beyond_line", "ball_speed_mps", "nearest_player_m",
                               "players_within_3m", "minute_of_half"), hazard.FEATURES
    table = features_of([(50.0, 30.0), (51.0, 30.0), (52.0, 30.0)])
    assert table.shape == (3, 6) and table.dtype == np.float64, (table.shape, table.dtype)
    empty = features_of(np.empty((0, 2)), np.empty((0, 4, 2)))
    assert empty.shape == (0, 6), empty.shape


def test_features_line_distance_signs():
    # (x, y) and its line distance on the 105 x 68 pitch: min(x, L - x, y, W - y), negative beyond a line.
    cases = [
        ((52.5, 34.0), 34.0),                                                   # the middle: touchlines are nearest
        ((10.0, 60.0), 8.0),                                                    # W - y; a swapped L and W gives 10
        ((0.0, 30.0), 0.0), ((3.0, 30.0), 3.0), ((-1.5, 30.0), -1.5),           # the left goal line: on, in, beyond
        ((105.0, 30.0), 0.0), ((102.0, 30.0), 3.0), ((106.0, 30.0), -1.0),      # the right goal line
        ((50.0, 0.0), 0.0), ((50.0, 2.0), 2.0), ((50.0, -2.0), -2.0),           # the touchline at y = 0
        ((50.0, 68.0), 0.0), ((50.0, 66.0), 2.0), ((50.0, 70.5), -2.5),         # the touchline at y = W
        ((0.0, 0.0), 0.0), ((105.0, 68.0), 0.0),                                # corners, on both lines
        ((-1.0, -2.0), -2.0),                                                   # beyond two lines: the deeper one
    ]
    table = features_of([point for point, _ in cases])
    expected = np.array([distance for _, distance in cases])
    assert np.array_equal(column(table, "line_distance_m"), expected), column(table, "line_distance_m") - expected
    # beyond_line is 1.0 only strictly beyond a line: a ball exactly on one is still in play.
    assert np.array_equal(column(table, "beyond_line"), (expected < 0).astype(float)), column(table, "beyond_line")
    assert column(table, "beyond_line").sum() == 5.0, column(table, "beyond_line")   # the five negative distances


def test_hazard_features_ball_speed_is_the_mean_of_the_last_five_steps():
    # Hand case: steps of 1, 2, 3, 4, 5, 6 m per frame, a missing frame, a frame at 33 m (no step is defined: the frame
    # before it is missing) and one at 34 m (a step of 1 m). The feature at t is the mean of the last five defined
    # steps at or before t, times fps: 25 * (1 + 2 + ... ) / n, and across the gap the old speed carries on.
    x = [10.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, np.nan, 33.0, 34.0]
    table = features_of([(value, 34.0) for value in x])
    speed = column(table, "ball_speed_mps")
    expected = np.array([np.nan, 25.0, 37.5, 50.0, 62.5, 75.0, 100.0, np.nan, 100.0, 95.0])
    assert np.allclose(speed, expected, rtol=0, atol=1e-9, equal_nan=True), speed

    # Random tracks with gaps and half-missing rows, against a loop that follows the definition literally.
    rng = np.random.default_rng([15, 10])
    ball = np.array([50.0, 34.0]) + np.cumsum(rng.normal(0.0, 0.3, size=(400, 2)), axis=0)
    ball[rng.uniform(size=400) < 0.15] = np.nan
    ball[100:130] = np.nan
    ball[rng.uniform(size=400) < 0.05, 0] = np.nan      # one coordinate missing counts as the ball missing
    steps, expected = [], []
    for t in range(len(ball)):
        if t and not np.isnan(ball[t]).any() and not np.isnan(ball[t - 1]).any():
            steps.append(math.hypot(*(ball[t] - ball[t - 1])) * FPS)
        expected.append(np.mean(steps[-5:]) if steps and not np.isnan(ball[t]).any() else np.nan)
    speed = column(features_of(ball), "ball_speed_mps")
    assert np.allclose(speed, expected, rtol=1e-9, atol=0, equal_nan=True), np.nanmax(np.abs(speed - expected))
    assert np.isnan(speed).sum() == np.isnan(np.array(expected)).sum()


def test_hazard_features_players_around_the_ball():
    ball = [(50.0, 30.0)] * 6
    nan = np.nan
    players = np.array([
        [(53.0, 30.0), (50.0, 27.0), (50.0, 34.0), (nan, nan)],        # 3.0 and 3.0 exactly, 4.0, absent
        [(50.5, 30.0), (53.0001, 30.0), (nan, nan), (nan, nan)],       # 0.5, just over 3 m
        [(nan, nan), (nan, nan), (nan, nan), (nan, nan)],              # nobody on the pitch
        [(50.0, nan), (47.0, 30.0), (nan, nan), (nan, nan)],           # half a position is no position; 3.0 exactly
        [(50.0, 30.0), (51.0, 30.0), (nan, nan), (nan, nan)],          # the ball is missing on this row (below)
        [(50.0, 30.0), (50.0, 33.0), (49.0, 30.0), (52.0, 30.0)],      # on the ball, 3.0, 1.0, 2.0
    ])
    ball[4] = (nan, nan)
    table = features_of(ball, players)
    nearest, within = column(table, "nearest_player_m"), column(table, "players_within_3m")
    assert np.allclose(nearest, [3.0, 0.5, nan, 3.0, nan, 0.0], rtol=0, atol=1e-12, equal_nan=True), nearest
    assert np.allclose(within, [2, 1, 0, 1, nan, 4], rtol=0, atol=0, equal_nan=True), within
    # No players at all (zero columns) is the same as nobody present.
    bare = features_of([(50.0, 30.0)] * 2, np.empty((2, 0, 2)))
    assert np.isnan(column(bare, "nearest_player_m")).all() and (column(bare, "players_within_3m") == 0).all()


def test_hazard_features_minute_counts_from_the_kickoff():
    ball = [(50.0, 30.0)] * 4
    # Row i is frame block_start + i. At 25 fps, one minute is 1,500 frames.
    table = features_of(ball, block_start=1500, kickoff=0)
    assert np.allclose(column(table, "minute_of_half"), [1.0, 1.0 + 1 / 1500, 1.0 + 2 / 1500, 1.0 + 3 / 1500],
                       rtol=0, atol=1e-12), column(table, "minute_of_half")
    # A second half is graded from before its kick-off, so its first minutes are negative.
    table = features_of(ball, block_start=110_808, kickoff=113_742)
    expected = (110_808 + np.arange(4) - 113_742) / 25 / 60
    assert np.allclose(column(table, "minute_of_half"), expected, rtol=0, atol=1e-12)
    assert column(table, "minute_of_half")[0] < -1.9
    # A different fps changes the unit.
    table = features_of(ball, fps=50, block_start=3000, kickoff=0)
    assert abs(column(table, "minute_of_half")[0] - 1.0) < 1e-12


def test_hazard_features_are_nan_wherever_the_ball_is_missing():
    ball = [(50.0, 30.0), (np.nan, np.nan), (50.0, np.nan), (np.nan, 30.0), (51.0, 30.0)]
    players = np.full((5, 2, 2), 51.0)   # everyone is present on every row
    table = features_of(ball, players)
    assert np.isnan(table[[1, 2, 3]]).all(), table[[1, 2, 3]]   # every column, the minute included
    assert not np.isnan(table[[0, 4]][:, [0, 1, 3, 4, 5]]).any()


def test_hazard_features_refuse_what_they_cannot_read():
    good = dict(ball_xy_m=np.zeros((4, 2)), players_xy_m=np.zeros((4, 3, 2)), pitch=PITCH, fps=FPS, block_start=0,
                kickoff=0)

    def call(**changes):
        return lambda: hazard.features(**{**good, **changes})

    expect_error(ValueError, call(ball_xy_m=np.zeros((4, 3))), "hazard.features", "ball_xy_m", "(n, 2)")
    expect_error(ValueError, call(ball_xy_m=np.zeros(4)), "hazard.features", "ball_xy_m")
    expect_error(ValueError, call(players_xy_m=np.zeros((5, 3, 2))), "hazard.features", "players_xy_m", "4")
    expect_error(ValueError, call(players_xy_m=np.zeros((4, 2))), "hazard.features", "players_xy_m")
    expect_error(ValueError, call(pitch=(105.0,)), "hazard.features", "pitch")
    expect_error(ValueError, call(pitch=(105.0, 0.0)), "hazard.features", "pitch")
    expect_error(ValueError, call(fps=0), "hazard.features", "fps")
    expect_error(ValueError, call(block_start=1.5), "hazard.features", "block_start")


# -- the logistic fit -------------------------------------------------------------------------------------------------

def test_fit_logistic_recovers_coefficients():
    coef, intercept = [0.6, -1.1, 0.4, 0.0, 0.9], -0.7
    means, spreads = [10.0, -3.0, 2.5, 0.0, 0.5], [2.0, 0.5, 1.5, 3.0, 0.5]
    X, y = synthetic_rows([15, 1], 50_000, coef, intercept, means, spreads)
    model = hazard.fit_logistic(X, y)
    assert isinstance(model, hazard.LogisticModel), type(model)
    assert model.coef.shape == model.mean.shape == model.scale.shape == (5,)
    # The truth is stated for the features as given, so the standardisation the fit does inside must be undone: a
    # coefficient on the standardised feature divided by its scale is the coefficient per unit of the feature.
    found = model.coef / model.scale
    assert np.all(np.abs(found - coef) <= 0.1), found - coef
    # The stored intercept is the log-odds at the features' sample means, where the generating model has this value
    # (the raw intercept is no fair target: a mean of 10 turns a 0.01 error in a coefficient into 0.1).
    at_means = intercept + float((model.mean - means) @ np.asarray(coef))
    assert abs(model.intercept - at_means) <= 0.1, (model.intercept, at_means)


def test_hazard_fit_logistic_reaches_the_penalised_optimum():
    # The optimum of  -loglik + l2 / 2 * |coef|^2  (intercept not penalised) has a zero gradient. A penalty that is too
    # strong, too weak, on the intercept, or a Newton step that stopped early all leave one. The tolerance allows for
    # the 1e-9 rounding of the stored parameters.
    X, y = synthetic_rows([15, 2], 300, [1.0, -0.5, 0.0], 0.3, [1.0, 2.0, 0.0], [1.0, 3.0, 0.5])
    for l2 in (0.0, 1.0, 5.0, 40.0):
        model = hazard.fit_logistic(X, y, l2=l2)
        assert np.abs(gradient(model, X, y, l2)).max() < 1e-5, (l2, gradient(model, X, y, l2))
    # A feature in tiny units (SD 4e-7): its scale is stored rounded to 1e-9, a 0.1% change of the number, so a fit
    # made with the unrounded SD but stored with the rounded one has a gradient of about 5e-4 in the stored space.
    X, y = synthetic_rows([15, 13], 300, [1.5e6, -0.6], 0.2, [0.0, 1.0], [4e-7, 1.0])
    for l2 in (1.0, 40.0):
        model = hazard.fit_logistic(X, y, l2=l2)
        assert np.abs(gradient(model, X, y, l2)).max() < 1e-5, (l2, gradient(model, X, y, l2))
    # The rare-event case the hazard is really fitted on: about 0.4% of rows are onsets.
    X, y = synthetic_rows([15, 3], 20_000, [0.8, -0.6, 0.3], -5.5, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert 0.001 < y.mean() < 0.02, y.mean()
    model = hazard.fit_logistic(X, y)
    assert np.abs(gradient(model, X, y, 1.0)).max() < 1e-5, gradient(model, X, y, 1.0)
    # With an enormous penalty the coefficients vanish and the intercept is the base rate, which it can only be if the
    # intercept carries no penalty of its own.
    flat = hazard.fit_logistic(X, y, l2=1e9)
    assert np.abs(flat.coef).max() < 1e-4, flat.coef
    assert abs(flat.intercept - math.log(y.mean() / (1 - y.mean()))) < 1e-3, flat.intercept


def test_hazard_fit_logistic_converges_on_a_rare_strong_flag():
    # Cases that plain Newton mostly cannot do (it failed 3 of these 4): about 147 onsets in 90,000 rows and a 0/1 flag
    # on 0.3% to 4% of the rows whose odds ratio is 12 to 64 (the features of every other check are continuous). The
    # first Newton step overshoots, and the next ones cycle (the flag's coefficient went 3.7, -4,388, 549, ...) or
    # saturate the probabilities: the first version of this fit returned a coefficient of -88,000 on some of these and
    # refused the rest as "collinear". Halving each step until the penalised objective does not rise converges in at
    # most 8 steps on all of them, so 15 are allowed here, and a fit that has not converged would raise.
    for index, (prevalence, odds_ratio) in enumerate(((0.003, 20), (0.01, 12), (0.01, 64), (0.04, 20))):
        X, y, p = rare_flag_rows([15, 12, index], 90_000, prevalence, odds_ratio)
        model = hazard.fit_logistic(X, y, iterations=15)
        # It is the optimum: the penalised gradient vanishes (measured at most 2.3e-6 on these four; the 1e-9 rounding
        # of the stored parameters meets a curvature of up to about 1e4).
        assert np.abs(gradient(model, X, y, 1.0)).max() < 1e-5, (prevalence, odds_ratio, gradient(model, X, y, 1.0))
        # And it is the right optimum: each coefficient, per unit of its feature, is within 3 sampling standard errors
        # of the truth (the flag's is 0.2 to 0.4 here, i.e. a tolerance of 0.5 to 1.1 on the log odds ratio; the
        # measured errors are at most 1.9 standard errors).
        found = model.coef / model.scale
        truth = np.array([0.5, -0.3, math.log(odds_ratio)])
        errors = sampling_errors(X, p)
        assert np.all(np.abs(found - truth) <= 3 * errors), (prevalence, odds_ratio, found, truth, errors)


def test_hazard_fit_logistic_standardises_and_rounds():
    rng = np.random.default_rng([15, 4])
    X = np.column_stack([rng.normal(5.0, 2.0, 40), rng.uniform(-1.0, 1.0, 40), np.full(40, 0.1)])
    y = (rng.uniform(size=40) < sigmoid(0.8 * (X[:, 0] - 5.0))).astype(float)
    model = hazard.fit_logistic(X, y)
    # Mean and the population SD (ddof = 0); a constant column keeps its value as the mean and gets a scale of 1.
    assert np.allclose(model.mean, [X[:, 0].mean(), X[:, 1].mean(), 0.1], rtol=0, atol=1e-9), model.mean
    assert np.allclose(model.scale, [X[:, 0].std(), X[:, 1].std(), 1.0], rtol=0, atol=1e-9), model.scale
    assert model.mean[2] == 0.1 and model.scale[2] == 1.0 and model.coef[2] == 0.0, model
    # Every stored number is a multiple of 1e-9 and none is a negative zero.
    for values in (model.coef, model.mean, model.scale, np.array([model.intercept])):
        assert np.array_equal(np.round(values, 9), values), values
        assert not np.signbit(values[values == 0]).any(), values
    # It is deterministic, byte for byte.
    again = hazard.fit_logistic(X, y)
    assert all(a.tobytes() == b.tobytes() for a, b in ((model.coef, again.coef), (model.mean, again.mean),
                                                       (model.scale, again.scale)))
    assert model.intercept == again.intercept
    # An affine change of the units changes nothing the model says.
    moved = hazard.fit_logistic(X * [3.0, 0.01, 1.0] + [7.0, -2.0, 0.0], y)
    assert np.allclose(hazard.predict(model, X), hazard.predict(moved, X * [3.0, 0.01, 1.0] + [7.0, -2.0, 0.0]),
                       rtol=0, atol=1e-6)
    # `iterations` caps the Newton steps, and a fit that is not done by then is refused, never returned: one step is not
    # the optimum. The error gives the size of the last step and suggests a larger l2. By 50 steps this fit has
    # stopped moving, so more steps change nothing.
    # The size it gives is the undamped Newton step, which is not necessarily the step that was taken (that one is
    # halved until the objective falls), so it says "would move", not "moved".
    expect_error(ValueError, lambda: hazard.fit_logistic(X, y, iterations=1), "hazard.fit_logistic", "no convergence",
                 "1 Newton steps", "the last Newton step would move a parameter by", "larger l2")
    long = hazard.fit_logistic(X, y, iterations=200)
    assert np.array_equal(long.coef, model.coef) and long.intercept == model.intercept
    # The mean and the scale that are stored are the ones the fit used (the optimum check above has a column in tiny
    # units for that), so a column whose SD rounds to 0 at 1e-9 is refused: predict would divide by zero.
    tiny = X.copy()
    tiny[:, 1] = 1.0 + 1e-12 * rng.normal(size=40)
    expect_error(ValueError, lambda: hazard.fit_logistic(tiny, y), "hazard.fit_logistic", "column 1", "rounds to 0")


def test_hazard_fit_logistic_refuses_what_it_cannot_fit():
    X, y = synthetic_rows([15, 5], 60, [1.0, -1.0], 0.0, [0.0, 0.0], [1.0, 1.0])
    holed = X.copy()
    holed[3, 1] = np.nan
    expect_error(ValueError, lambda: hazard.fit_logistic(holed, y), "hazard.fit_logistic", "NaN", "at-risk")
    expect_error(ValueError, lambda: hazard.fit_logistic(X, y[:-1]), "hazard.fit_logistic", "60", "59")
    expect_error(ValueError, lambda: hazard.fit_logistic(X[:, 0], y), "hazard.fit_logistic", "(n, k)")
    expect_error(ValueError, lambda: hazard.fit_logistic(X, y + 1.0), "hazard.fit_logistic", "0 or 1")
    expect_error(ValueError, lambda: hazard.fit_logistic(X, np.zeros(60)), "hazard.fit_logistic", "both")
    expect_error(ValueError, lambda: hazard.fit_logistic(X, y, l2=-1.0), "hazard.fit_logistic", "l2")
    expect_error(ValueError, lambda: hazard.fit_logistic(X, y, iterations=0), "hazard.fit_logistic", "iterations")
    # Two identical columns and no penalty: the Hessian is singular, there is no one answer, and the error says so and
    # what to change. With a penalty the same columns are fine: it splits the weight between them.
    twin = np.column_stack([X[:, 0], X[:, 0]])
    expect_error(ValueError, lambda: hazard.fit_logistic(twin, y, l2=0.0), "hazard.fit_logistic", "singular",
                 "repeats another", "l2 > 0")
    assert np.allclose(hazard.fit_logistic(twin, y).coef[0], hazard.fit_logistic(twin, y).coef[1], atol=1e-6)
    # A repeated column is not the only way to a singular Hessian without a penalty: a constant column (nothing in the
    # design but zeros once it is centred) is another, and so is a feature that separates the outcomes (the
    # probabilities saturate at 0 and 1 and the curvature vanishes). The message names all three. With a penalty the
    # constant column is fine: its coefficient is 0.
    flat = np.column_stack([X[:, 0], np.full(60, 3.0)])
    message = expect_error(ValueError, lambda: hazard.fit_logistic(flat, y, l2=0.0), "hazard.fit_logistic",
                           "singular", "l2 > 0")
    assert "constant" in message and "separat" in message and "repeats another" in message, message
    assert hazard.fit_logistic(flat, y).coef[1] == 0.0
    separated = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 5)
    expect_error(ValueError, lambda: hazard.fit_logistic(separated, (separated[:, 0] > 0).astype(float), l2=0.0,
                                                         iterations=500),
                 "hazard.fit_logistic", "singular", "separat", "l2 > 0")


def test_hazard_predict_applies_the_stored_standardisation():
    model = hazard.LogisticModel(coef=np.array([1.0, -2.0]), intercept=0.5, mean=np.array([10.0, 0.0]),
                                 scale=np.array([2.0, 4.0]))
    # z = 0.5 + 1 * (12 - 10) / 2 - 2 * (4 - 0) / 4 = -0.5 ; z = 0.5 at the mean.
    got = hazard.predict(model, np.array([[12.0, 4.0], [10.0, 0.0], [10.0, np.nan], [18.0, -4.0]]))
    expected = [sigmoid(-0.5), sigmoid(0.5), np.nan, sigmoid(0.5 + 4.0 + 2.0)]
    assert np.allclose(got, expected, rtol=0, atol=1e-12, equal_nan=True), got
    assert got.shape == (4,)
    # Large scores neither overflow nor leave [0, 1].
    far = hazard.predict(model, np.array([[1e6, -1e6], [-1e6, 1e6]]))
    assert far[0] == 1.0 and far[1] == 0.0, far
    # A fitted model reproduces the base rate (the intercept's score equation), however hard the penalty pulls.
    X, y = synthetic_rows([15, 6], 500, [1.5, -0.5], -1.0, [0.0, 1.0], [1.0, 2.0])
    fitted = hazard.fit_logistic(X, y, l2=50.0)
    assert abs(hazard.predict(fitted, X).mean() - y.mean()) < 1e-6
    expect_error(ValueError, lambda: hazard.predict(model, np.zeros((3, 3))), "hazard.predict", "2", "3")
    expect_error(ValueError, lambda: hazard.predict(model, np.zeros(2)), "hazard.predict", "(n, k)")


# -- AUC --------------------------------------------------------------------------------------------------------------

def test_auc_known_cases():
    labels = [0, 0, 0, 1, 1]
    assert hazard.auc([0.1, 0.2, 0.3, 0.8, 0.9], labels) == 1.0
    assert hazard.auc([0.9, 0.8, 0.3, 0.2, 0.1], labels) == 0.0
    assert hazard.auc([0.5] * 5, labels) == 0.5                      # no information at all
    assert hazard.auc([1, 2, 2, 3], [0, 0, 1, 1]) == 0.875          # 3 wins and 1 tie of 4 pairs: (3 + 0.5) / 4
    assert hazard.auc([1, 1, 2, 2], [0, 1, 0, 1]) == 0.5
    assert hazard.auc([3, 2, 2, 1], [False, False, True, True]) == 0.125   # booleans are labels too: 1 tie, 3 losses
    assert isinstance(hazard.auc([0.1, 0.9], [0, 1]), float)


def test_hazard_auc_matches_the_pairwise_count_with_ties():
    rng = np.random.default_rng([15, 7])
    labels = (rng.uniform(size=300) < 0.2).astype(int)
    scores = np.round(rng.normal(size=300) + 0.8 * labels, 1)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    wins = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    assert len(np.unique(scores)) < 120   # plenty of ties
    assert abs(hazard.auc(scores, labels) - wins / (len(positives) * len(negatives))) < 1e-12
    # It does not matter how the rows are ordered.
    order = rng.permutation(300)
    assert hazard.auc(scores[order], labels[order]) == hazard.auc(scores, labels)


def test_hazard_auc_refuses_what_it_cannot_score():
    expect_error(ValueError, lambda: hazard.auc([0.1, np.nan, 0.9], [0, 1, 1]), "hazard.auc", "NaN")
    expect_error(ValueError, lambda: hazard.auc([0.1, 0.9], [1, 1]), "hazard.auc", "both")
    expect_error(ValueError, lambda: hazard.auc([0.1, 0.9], [0, 0]), "hazard.auc", "both")
    expect_error(ValueError, lambda: hazard.auc([0.1, 0.9, 0.5], [0, 1]), "hazard.auc", "3", "2")
    expect_error(ValueError, lambda: hazard.auc([0.1, 0.9], [0, 2]), "hazard.auc", "0 or 1")
    expect_error(ValueError, lambda: hazard.auc([], []), "hazard.auc", "both")


# -- the sampler ------------------------------------------------------------------------------------------------------

def test_hazard_third_of_minute_cuts_at_15_and_30():
    got = hazard.third_of_minute(np.array([-3.0, 0.0, 14.999, 15.0, 29.999, 30.0, 44.9, 100.0]))
    assert got.tolist() == [0, 0, 0, 1, 1, 2, 2, 2], got
    assert got.dtype.kind == "i"
    expect_error(ValueError, lambda: hazard.third_of_minute(np.array([1.0, np.nan])), "hazard.third_of_minute", "NaN")


def test_hazard_sample_mask_follows_the_cumulative_hazard():
    # prob = UNIT_HAZARD makes the cumulative hazard L(r) = r + 1. The event is the first at-risk frame r >= t with
    # L(r) - L(t - 1) >= E, t being the first visible frame after the last gap (0 at the start); the gap then occupies
    # r + 1 ... r + d, and its length is looked up at r. Every number below is worked out by hand.
    durations = [[np.array([3]), np.array([4]), np.array([5])],     # onset_class 0, the three thirds
                 [np.array([6]), np.array([7]), np.array([8])]]     # onset_class 1
    n = 30
    prob = np.full(n, UNIT_HAZARD)
    # Class and third differ on both neighbours of each event frame (2, 7 and 27), so a lookup made one frame early or
    # late (at the first missing frame, say) picks another length: the lengths at 1, 2, 3 are 8, 3, 7; at 6, 7, 8
    # 5, 7, 3; at 26, 27, 28 6, 5, 7.
    onset_class = np.zeros(n, dtype=np.int64)
    onset_class[[1, 3, 7, 26, 28]] = 1
    third = np.zeros(n, dtype=np.int64)
    third[[1, 6, 27]] = 2
    third[[3, 7, 28]] = 1
    rng = ScriptedRng([2.5, 1.2, 12.5])
    mask = hazard.sample_mask(prob, onset_class, third, durations, rng)
    # E = 2.5 from t = 0: L(2) = 3 >= 2.5 > L(1) = 2, so r = 2 (class 0, first third: 3 frames) -> missing 3, 4, 5.
    # E = 1.2 from t = 6: L(r) - L(5) = r - 5 >= 1.2 -> r = 7 (class 1, second third: 7 frames) -> missing 8 ... 14.
    # E = 12.5 from t = 15: L(r) - L(14) = r - 14 >= 12.5 -> r = 27 (class 0, last third: 5 frames) -> 28 ... 32, cut
    # at 30, so 28 and 29.
    expected = np.zeros(n, dtype=bool)
    expected[3:6] = expected[8:15] = expected[28:] = True
    assert mask.dtype == bool and mask.shape == (n,), (mask.dtype, mask.shape)
    assert np.array_equal(mask, expected), runs_of_true(mask)
    # The order of the draws is part of the contract (a stream seeded the same way must give the same gaps): an
    # exponential, then a length, per event, and nothing more once the last gap runs off the end of the block.
    assert rng.log == [("exponential", 1.0), ("choice", [3]), ("exponential", 1.0), ("choice", [7]),
                       ("exponential", 1.0), ("choice", [5])], rng.log

    # Gaps never fuse: an event on the very first frame after a gap leaves that frame visible. E = 2.5 gives r = 2 and
    # the gap 3, 4, 5; then E = 0.5 from t = 6: L(6) - L(5) = 1 >= 0.5, so r = 6 (class 0, last third: 5 frames) and
    # the gap 7 ... 11. Frame 6 stays visible between them.
    mask = hazard.sample_mask(prob[:20], onset_class[:20], third[:20], durations, ScriptedRng([2.5, 0.5]))
    assert runs_of_true(mask) == [(3, 6), (7, 12)], runs_of_true(mask)

    # A gap cannot start on frame 0, which is visible by construction: the earliest event is r = 0, the gap frame 1.
    mask = hazard.sample_mask(prob[:12], onset_class[:12], third[:12], durations, ScriptedRng([0.5]))
    assert runs_of_true(mask) == [(1, 4)], runs_of_true(mask)   # r = 0: class 0, first third -> 3 frames

    # An event on the last frame has nothing to mark, but still draws its length, and then nothing more is drawn.
    rng = ScriptedRng([11.5])   # L(11) = 12 >= 11.5 > L(10) = 11
    mask = hazard.sample_mask(prob[:12], onset_class[:12], third[:12], durations, rng)
    assert not mask.any() and rng.log == [("exponential", 1.0), ("choice", [3])], rng.log

    # A draw that overshoots the block ends the sampling: one event, then one exponential that finds no frame.
    rng = ScriptedRng([2.5])
    mask = hazard.sample_mask(prob[:12], onset_class[:12], third[:12], durations, rng)
    assert runs_of_true(mask) == [(3, 6)], runs_of_true(mask)
    assert [call for call, _ in rng.log] == ["exponential", "choice", "exponential"], rng.log

    # The exact rule for the draws (the docstring's): an exponential and a length per event, and ONE more exponential
    # if and only if r + 1 + d < n after the last event, or there was no event and n > 0. E = 2.5 puts the event at
    # r = 2 (class 0, first third: d = 3), so r + 1 + d = 6, and the block length n decides:
    #   n = 2  no frame reaches E: no event, one exponential
    #   n = 3  the event is on the last frame, which has nothing to mark: no more draws
    #   n = 5  the gap runs off the end of the block (3, 4, cut): no more draws
    #   n = 6  the gap ends exactly on the last frame: 6 is not < 6, so no more draws
    #   n = 7  one visible frame is left after the gap: one more exponential, which overshoots
    for n_block, marked, more in ((2, [], 0), (3, [], 0), (5, [(3, 5)], 0), (6, [(3, 6)], 0), (7, [(3, 6)], 1)):
        rng = ScriptedRng([2.5])
        mask = hazard.sample_mask(prob[:n_block], onset_class[:n_block], third[:n_block], durations, rng)
        assert runs_of_true(mask) == marked, (n_block, runs_of_true(mask))
        calls = [call for call, _ in rng.log]
        events = 0 if n_block == 2 else 1
        assert calls == ["exponential", "choice"] * events + ["exponential"] * (more + (n_block == 2)), (n_block, calls)
    # An empty block draws nothing at all.
    rng = ScriptedRng([2.5])
    assert hazard.sample_mask(np.zeros(0), onset_class[:0], third[:0], durations, rng).shape == (0,)
    assert rng.log == [], rng.log

    # The condition is "L(r) - L(t - 1) >= E": a draw that lands exactly on a cumulative value picks that frame, not the
    # next. (A continuous draw never does; the scripted one is the cumulative sum itself, at frame 4.)
    exact = float(np.cumsum(-np.log1p(-prob))[4])
    mask = hazard.sample_mask(prob[:12], onset_class[:12], third[:12], durations, ScriptedRng([exact]))
    assert runs_of_true(mask) == [(5, 8)], runs_of_true(mask)   # r = 4: class 0, first third -> 3 frames

    # Frames with no hazard are skipped: the first frame whose cumulative hazard reaches E is 10 (L = 1 there).
    quiet_start = np.concatenate([np.zeros(10), np.full(10, UNIT_HAZARD)])
    mask = hazard.sample_mask(quiet_start, onset_class[:20], third[:20], durations, ScriptedRng([0.5]))
    assert runs_of_true(mask) == [(11, 14)], runs_of_true(mask)   # r = 10: class 0, first third -> 3 frames
    # No hazard anywhere: no gap, and the one draw made finds nothing.
    rng = ScriptedRng([0.5])
    assert not hazard.sample_mask(np.zeros(20), onset_class[:20], third[:20], durations, rng).any()
    assert rng.log == [("exponential", 1.0)], rng.log
    assert hazard.sample_mask(np.zeros(0), onset_class[:0], third[:0], durations, ScriptedRng([])).shape == (0,)


def test_sample_mask_deterministic_and_rate():
    # With a constant hazard p and one gap length d, an event waits G ~ Geometric(p) at-risk frames (the first frame
    # counts, so E[G] = 1 / p and Var[G] = (1 - p) / p^2), and then the gap takes d frames, after which the next
    # at-risk frame comes at once. So the cycles are G + d, they renew, and over T frames the expected number of
    # events is T / mu, with variance T * sigma^2 / mu^3 (the standard renewal-count asymptotics), where
    #     mu = 1 / p + d,    sigma^2 = (1 - p) / p^2.
    # Edge effects are a count or two, far below the 3 sigma allowed.
    p, d, blocks, frames = RATE_P, RATE_D, RATE_BLOCKS, RATE_FRAMES
    mu = 1.0 / p + d
    total = blocks * frames
    expected = total / mu
    sigma = math.sqrt(total * ((1.0 - p) / p ** 2) / mu ** 3)
    prob = np.full(frames, p)
    zeros = np.zeros(frames, dtype=np.int64)
    durations = constant_durations(d)

    def draw(seed_tail):
        counter = CountingRng(np.random.default_rng([15, 8, seed_tail]))
        return hazard.sample_mask(prob, zeros, zeros, durations, counter), counter.onsets

    masks, onsets = zip(*(draw(block) for block in range(blocks)))
    count = sum(onsets)
    assert abs(count - expected) <= 3 * sigma, (count, expected, sigma)
    # Gaps never overlap, and only the last one in a block can be cut by the block's end (by up to d frames: an event
    # on the very last frame has all of its gap beyond the block).
    masked = sum(int(mask.sum()) for mask in masks)
    assert d * count - d * blocks <= masked <= d * count, (masked, count)
    assert not any(mask[0] for mask in masks)   # the first frame is visible by construction
    # The same seed gives the same mask, byte for byte; another seed gives another.
    again, _ = draw(0)
    assert again.tobytes() == masks[0].tobytes() and again.dtype == bool
    assert masks[1].tobytes() != masks[0].tobytes()


def test_hazard_sample_mask_starts_only_where_the_hazard_is_positive():
    n, d = 10_000, 10
    prob = np.zeros(n)
    prob[2000:3000] = 0.05
    zeros = np.zeros(n, dtype=np.int64)
    for seed in range(5):
        mask = hazard.sample_mask(prob, zeros, zeros, constant_durations(d), np.random.default_rng([15, 9, seed]))
        runs = runs_of_true(mask)
        # About 50 events fall in the window; each is an at-risk frame r in 2000 ... 2999 and its gap starts at r + 1.
        assert runs and runs[0][0] >= 2001 and runs[-1][1] <= 3000 + d, runs
        assert all(stop - start == d for start, stop in runs), runs
    # No event falls inside a gap and no two gaps fuse: with a hazard high enough to chain gaps, every run is exactly
    # one gap long (the last one may be cut by the end of the block). A multiple of d would mean fused gaps.
    n = 5000
    dense = np.full(n, 0.3)
    mask = hazard.sample_mask(dense, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), constant_durations(5),
                              np.random.default_rng([15, 9, 99]))
    runs = runs_of_true(mask)
    assert len(runs) > 100 and not mask[0], len(runs)
    assert all(stop - start == 5 for start, stop in runs[:-1]) and runs[-1][1] - runs[-1][0] <= 5, runs[:5]


def test_hazard_sample_mask_refuses_bad_input():
    n = 10
    prob, ok_class, ok_third = np.full(n, 0.01), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    durations = constant_durations(4)
    rng = np.random.default_rng([15, 11])

    def call(prob=prob, onset_class=ok_class, third=ok_third, durations=durations):
        return lambda: hazard.sample_mask(prob, onset_class, third, durations, rng)

    holed = prob.copy()
    holed[2] = np.nan
    expect_error(ValueError, call(prob=holed), "hazard.sample_mask", "prob", "NaN")
    expect_error(ValueError, call(prob=np.concatenate([prob[:-1], [1.0]])), "hazard.sample_mask", "prob", "[0, 1)")
    expect_error(ValueError, call(prob=np.concatenate([prob[:-1], [-0.1]])), "hazard.sample_mask", "prob", "[0, 1)")
    expect_error(ValueError, call(onset_class=ok_class[:-1]), "hazard.sample_mask", "onset_class", "10")
    expect_error(ValueError, call(onset_class=ok_class + 2), "hazard.sample_mask", "onset_class", "0 or 1")
    expect_error(ValueError, call(onset_class=ok_class + 0.0), "hazard.sample_mask", "onset_class", "integer")
    expect_error(ValueError, call(third=ok_third + 3), "hazard.sample_mask", "third", "0, 1 or 2")
    expect_error(ValueError, call(third=ok_third[:5]), "hazard.sample_mask", "third", "10")
    short = [[np.array([4])] * 3]
    expect_error(ValueError, call(durations=short), "hazard.sample_mask", "durations[1][0]")
    expect_error(ValueError, call(durations=[[np.array([4])] * 3, []]), "hazard.sample_mask", "durations[1][0]")
    empty = constant_durations(4)
    empty[1][2] = np.array([], dtype=np.int64)
    expect_error(ValueError, call(durations=empty), "hazard.sample_mask", "durations[1][2]", "empty")
    zero = constant_durations(4)
    zero[0][1] = np.array([3, 0])
    expect_error(ValueError, call(durations=zero), "hazard.sample_mask", "durations[0][1]", "at least 1")
    seconds = constant_durations(4)
    seconds[0][0] = np.array([4.5])
    expect_error(ValueError, call(durations=seconds), "hazard.sample_mask", "durations[0][0]", "frames")
