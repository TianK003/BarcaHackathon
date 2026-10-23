"""Checks of the benchmark's truth (pipeline/benchmark/truth.py): codes, summary, line timing, nearest team.

The synthetic checks run on matches made here out of a few rows, in the real DflMatch layout, with every answer worked
out by hand from the rules rather than from what the code printed. The ones that read the downloaded matches
(test_dfl_truth_conventions, test_dfl_window_confound and test_dfl_coarse_truth, --dfl) print the aggregates that A06 in
docs/results.md quotes, and share one pass over the 7 matches; the DFL files are public
(CC BY 4.0), so printing their aggregates is fine. Every integer count that A06 quotes is asserted against a constant
below (the EXPECTED_* ones); the rates, medians, shares, lags and timings are printed, not asserted. The public ground
truth of BAR-ATH (evaluation/gt/, MIT) is the only thing of our match that is read here, and only as the 7,500 labels
the organisers published.
"""

import functools
import math

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import dfl, truth
from pipeline.benchmark.checks.common import MIN_AGREEMENT
from pipeline.selftest import expect_error

# Params
ALIVE, DEAD, UNKNOWN = 1, 0, -1       # ball_status, as DflMatch holds it
HOME, AWAY = 0, 1                     # ball_possession and player_team, as DflMatch holds them
TOY_PITCH = (100.0, 50.0)             # metres, length then width: not square, so that normalised distances mislead
# EVALUATION.md's table of "perfect but k frames late" (spellF1 of the truth against itself shifted k frames later).
EVALUATION_LAG_TABLE = {1: 1.0000, 3: 0.9818, 8: 0.8933, 25: 0.6539}
# What the same shift gives on the 5-minute public GT (17 possessions, the shortest 0.76 s). It is not the table's
# (A06): the table does not reproduce on the public GT, and EVALUATION.md does not say what it was computed on.
PUBLIC_LAG = {1: 1.0000, 3: 1.0000, 8: 0.9529, 25: 0.7765}
LAGS = (3, 8, 25)                     # what A06 reports for DFL, next to the table's
PLAYING_TIME_ATTRIBUTES = {1: ("TotalTimeFirstHalf", "PlayingTimeFirstHalf"),
                           2: ("TotalTimeSecondHalf", "PlayingTimeSecondHalf")}
WINDOW_FRAMES = 7500                  # the public window: 5 minutes at 25 fps
# The plan's rule for a de-flickered variant of the truth (truth_codes with a minimum run length): it is built only if
# more than this share of the pooled possessions last 2 frames or less. They do not (A06), so the truth is the raw one.
FLICKER_SHARE = 0.02
# The integer counts that A06 quotes, per match: (possessions, stoppages, alive -> dead changes, of them with the ball
# beyond a line in the 10 s before, and on an alive frame of those, alive ball-visible frames). Pinned so that a change
# of the truth's rules, of the parser or of the data shows.
EXPECTED_COUNTS = {"J03WMX": (511, 103, 103, 84, 82, 85369), "J03WN1": (500, 107, 106, 83, 78, 68377),
                   "J03WOH": (402, 102, 101, 71, 69, 71945), "J03WOY": (456, 113, 112, 89, 81, 76286),
                   "J03WPY": (421, 97, 97, 75, 73, 81460), "J03WQQ": (401, 99, 99, 47, 41, 84352),
                   "J03WR9": (370, 92, 92, 69, 66, 85316)}
# Pooled over the 7 matches: possessions, stoppages, and possessions of at most 2 and at most 12 frames.
EXPECTED_POOLED = (3061, 713, 15, 156)
# Per match, the possessions of at most 2 and of at most 12 frames (the table's two short-run columns), the public
# window's (possessions, <= 2, <= 12, stoppages) and the first 5 minutes of each first half, pooled, in the same order.
EXPECTED_SHORT_RUNS = {"J03WMX": (3, 29), "J03WN1": (2, 32), "J03WOH": (2, 14), "J03WOY": (1, 17),
                       "J03WPY": (3, 27), "J03WQQ": (3, 14), "J03WR9": (1, 23)}
EXPECTED_PUBLIC = (17, 0, 0, 4)
EXPECTED_FIRST_5 = (211, 3, 11, 44)
# DFL's own PlayingTime against the alive frames, the most that they differ in frames: in a match, and in a half.
MAX_PLAYING_DIFF = (2, 1)
# The 5-minute windows of A06: how many there are, how many are at or below the public window's alive share, and per
# selection (n windows, how many with a rate at or below the public window's, how many with a median possession at least
# as long, how many with a share of possessions under 5 s at most as large). Filled in from the first run.
EXPECTED_WINDOWS = {"all": 129, "alive_at_or_below": 124, "matches": 7,
                    "counts": {"all": (129, 7, 7, 9), "alive >= 70%": (32, 7, 5, 7), "alive >= 75%": (19, 5, 4, 5)}}
# The matches whose first line-timing measurement (any frame of the 10 s before a flip to dead) has an upper quartile
# at the window's whole length, 10.00 s: the window reaches back into an earlier stoppage there (A06).
EXPECTED_LEAD_CAPPED = ("J03WMX", "J03WN1", "J03WOY")
# Pooled possessions of the coarse truth for each candidate shortest possession (s). Filled in from the first run.
EXPECTED_COARSE = {1: 2707, 2: 2271, 3: 1979, 4: 1799, 5: 1659}
# The coarse truth at COARSE_MIN_S: its pooled possessions of at most 2 and at most 12 frames, and its possessions per
# match (A06: 220-255).
EXPECTED_COARSE_SHORT = (3, 41)
EXPECTED_COARSE_PER_MATCH = {"J03WMX": 255, "J03WN1": 254, "J03WOH": 220, "J03WOY": 248, "J03WPY": 237, "J03WQQ": 223,
                             "J03WR9": 222}


# -- helpers ----------------------------------------------------------------------------------------------------------

def toy_match(status, possession, graded, ball_xy=None, players_xy=None, player_team=(HOME, AWAY),
              pitch=TOY_PITCH, fps=2):
    """A DflMatch of len(status) rows with the given flags; the rest is NaN (or, for the players, absent)."""
    n = len(status)
    players_xy = np.full((n, len(player_team), 2), np.nan) if players_xy is None else np.asarray(players_xy, float)
    ball_xy = np.full((n, 2), np.nan) if ball_xy is None else np.asarray(ball_xy, float)
    return dfl.DflMatch(
        match_id="TOY", pitch=pitch, fps=fps, graded=tuple(graded), kickoffs={half: start for half, start, _ in graded},
        frame_number=np.arange(n, dtype=np.int64), players_xy=players_xy,
        players_speed=np.full((n, len(player_team)), np.nan), player_team=np.array(player_team, dtype=np.int8),
        ball_xy=ball_xy, ball_z=np.full(n, np.nan), ball_speed=np.full(n, np.nan),
        ball_status=np.array(status, dtype=np.int8), ball_possession=np.array(possession, dtype=np.int8), report={})


def runs(*pairs):
    """int8 codes from (code, length) pairs: runs((pm.H, 3), (pm.D, 2)) is H H H D D."""
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in pairs])


def rules_match():
    """The toy that exercises every rule of truth_codes, with the answer below.

    Row 0 is outside the halves; half 1 is rows 1-12, rows 13-14 are the separator, half 2 is rows 15-21, rows 22-23
    are outside. (status, possession) per row:
    """
    rows = {  # row: (status, possession)
        0: (ALIVE, HOME),                                                         # ungraded: X whatever it says
        1: (UNKNOWN, UNKNOWN), 2: (ALIVE, UNKNOWN), 3: (ALIVE, AWAY),
        4: (DEAD, HOME),                                                          # a dead frame that names a team
        5: (ALIVE, UNKNOWN), 6: (ALIVE, HOME), 7: (UNKNOWN, UNKNOWN), 8: (ALIVE, AWAY),
        9: (DEAD, UNKNOWN), 10: (ALIVE, UNKNOWN), 11: (UNKNOWN, HOME), 12: (DEAD, HOME),
        13: (UNKNOWN, UNKNOWN), 14: (UNKNOWN, UNKNOWN),
        15: (ALIVE, UNKNOWN), 16: (ALIVE, AWAY), 17: (ALIVE, HOME), 18: (DEAD, UNKNOWN), 19: (UNKNOWN, UNKNOWN),
        20: (ALIVE, UNKNOWN), 21: (DEAD, AWAY),
        22: (ALIVE, HOME), 23: (DEAD, AWAY)}
    status = [rows[row][0] for row in range(24)]
    possession = [rows[row][1] for row in range(24)]
    return toy_match(status, possession, ((1, 1, 13), (2, 15, 22)))


# What truth_codes must make of rules_match(), row by row, and why:
RULES_CODES = [
    pm.X,                      # 0   outside the halves
    pm.A,                      # 1   status unknown, at the head: the next label (row 2's)
    pm.A,                      # 2   alive, possession unknown at the head: back-filled from the next known one (row 3)
    pm.A,                      # 3   alive, away
    pm.D,                      # 4   dead, whatever the possession says
    pm.H,                      # 5   alive, unknown: the last known possession is row 4's, which was set on a dead frame
    pm.H,                      # 6
    pm.H,                      # 7   status unknown: the previous label
    pm.A,                      # 8
    pm.D,                      # 9   dead, possession unknown: D needs none
    pm.A,                      # 10  alive, unknown: the last known is row 8's
    pm.A,                      # 11  status unknown (its possession is not used): the previous label
    pm.D,                      # 12
    pm.X, pm.X,                # 13, 14  the separator
    pm.A,                      # 15  second half's head: the next known (row 16), never the first half's last (home)
    pm.A,                      # 16
    pm.H,                      # 17
    pm.D,                      # 18
    pm.D,                      # 19  status unknown: the previous label, D
    pm.H,                      # 20  alive, unknown: the last known possession is row 17's
    pm.D,                      # 21
    pm.X, pm.X]                # 22, 23  outside the halves


# -- the codes --------------------------------------------------------------------------------------------------------

def test_truth_codes_rules():
    codes, report = truth.truth_codes(rules_match())
    assert codes.dtype == np.int8 and codes.shape == (24,), (codes.dtype, codes.shape)
    assert codes.tolist() == RULES_CODES, "{}\nvs\n{}".format(
        "".join(pm.LABELS[c] for c in codes), "".join(pm.LABELS[c] for c in RULES_CODES))
    # Alive frames whose team was carried: rows 2, 5, 10, 15, 20. Frames whose label was carried: 1, 7, 11, 19.
    assert report == {"carried_possession_frames": 5, "carried_status_frames": 4, "graded_frames": 19}, report

    # Nothing is carried when every flag is there.
    whole = toy_match([ALIVE, DEAD, ALIVE, ALIVE], [AWAY, HOME, HOME, AWAY], ((1, 0, 4),))
    codes, report = truth.truth_codes(whole)
    assert codes.tolist() == [pm.A, pm.D, pm.H, pm.A], codes
    assert report == {"carried_possession_frames": 0, "carried_status_frames": 0, "graded_frames": 4}, report
    # A half that is all dead has no possession to carry and needs none.
    dead = toy_match([DEAD, DEAD, UNKNOWN], [UNKNOWN, UNKNOWN, UNKNOWN], ((1, 0, 3),))
    assert truth.truth_codes(dead)[0].tolist() == [pm.D, pm.D, pm.D]


def test_truth_codes_refuses_what_it_cannot_label():
    # A half whose status is never known: there is nothing to carry, and nothing to say about it.
    blind = toy_match([ALIVE, ALIVE, UNKNOWN, UNKNOWN, UNKNOWN], [HOME] * 5, ((1, 0, 2), (2, 2, 5)))
    expect_error(ValueError, lambda: truth.truth_codes(blind), "TOY", "half 2", "status")
    # Alive frames, and not one possession in the half to name a team from.
    nobody = toy_match([ALIVE, DEAD, ALIVE], [UNKNOWN, UNKNOWN, UNKNOWN], ((1, 0, 3),))
    expect_error(ValueError, lambda: truth.truth_codes(nobody), "TOY", "half 1", "possession")
    # A code that is none of 0, 1, -1 is not DFL's: it is refused rather than read as a team.
    odd = toy_match([ALIVE, 2], [HOME, HOME], ((1, 0, 2),))
    expect_error(ValueError, lambda: truth.truth_codes(odd), "TOY", "ball_status", "2")
    odd = toy_match([ALIVE, ALIVE], [HOME, 2], ((1, 0, 2),))
    expect_error(ValueError, lambda: truth.truth_codes(odd), "TOY", "ball_possession", "2")


def test_truth_scores_perfectly_against_itself():
    """The scorer sees the truth as the organisers' GT: the graded halves are its valid runs, and it matches itself."""
    match = rules_match()
    codes, _ = truth.truth_codes(match)
    assert pm.valid_runs(codes) == [(start, stop) for _, start, stop in match.graded], pm.valid_runs(codes)
    result = pm.evaluate(codes, codes)
    assert result["spell_f1_mean"] == 1.0 and result["macro_f1"] == 1.0, (result["spell_f1_mean"], result["macro_f1"])
    # Half 1 has A (rows 1-3), H (5-7), A (8) and A (10-11); half 2 has A (15-16), H (17) and H (20): seven.
    assert result["n_gt_spells"] == 7 and result["n_pred_spells"] == 7, result["n_gt_spells"]


# -- the summary ------------------------------------------------------------------------------------------------------

def summary_toy():
    """170 graded frames at 10 fps in two halves (rows 0-99 and 110-179), X between and after them; answers by hand.

    Half 1: H30 A1 D9 A2 H28 D30; half 2: D5 H6 A59. Possessions (frames): 30, 1, 2, 28, 6, 59 -- six; stoppages: 9,
    30, 5 -- three; H 64 frames and A 62.
    """
    codes = np.concatenate([runs((pm.H, 30), (pm.A, 1), (pm.D, 9), (pm.A, 2), (pm.H, 28), (pm.D, 30)),
                            runs((pm.X, 10)),
                            runs((pm.D, 5), (pm.H, 6), (pm.A, 59)),
                            runs((pm.X, 5))])
    return codes, ((1, 0, 100), (2, 110, 180))


def test_truth_summary_definitions():
    codes, graded = summary_toy()
    summary = truth.summary(codes, graded, 10)
    assert set(summary) == {"minutes", "alive_share", "possessions", "per_alive_minute", "median_s", "mean_s",
                            "share_under_5s", "runs_le_2_frames", "runs_le_12_frames", "stoppages",
                            "stoppage_median_s", "home_share"}, sorted(summary)
    expected = {"minutes": 170 / 10 / 60, "alive_share": 126 / 170, "possessions": 6,
                "per_alive_minute": 6 / (126 / 10 / 60),   # alive minutes, not graded minutes
                "median_s": 1.7, "mean_s": 2.1,            # lengths 1 2 6 28 30 59: (6 + 28) / 2 and 126 / 6 frames
                "share_under_5s": 5 / 6,                   # strictly under 5 * fps = 50 frames: the 59 is not
                "runs_le_2_frames": 2, "runs_le_12_frames": 3, "stoppages": 3, "stoppage_median_s": 0.9,
                "home_share": 64 / 126}
    for key, value in expected.items():
        assert math.isclose(summary[key], value, rel_tol=1e-12), (key, summary[key], value)
    # The X frames between and after the halves are not graded frames and count for nothing.
    assert truth.summary(codes[:100], ((1, 0, 100),), 10)["possessions"] == 4
    # A possession must not run on over the halftime hole: the two halves' last and first frames are not neighbours.
    joined = np.concatenate([runs((pm.H, 5)), runs((pm.X, 3)), runs((pm.H, 5))])
    assert truth.summary(joined, ((1, 0, 5), (2, 8, 13)), 10)["possessions"] == 2
    # Nothing to say where there is nothing: no possessions leaves their statistics NaN and the counts 0.
    quiet = truth.summary(runs((pm.D, 20)), ((1, 0, 20),), 10)
    assert quiet["possessions"] == 0 and quiet["stoppages"] == 1 and quiet["alive_share"] == 0.0, quiet
    assert all(math.isnan(quiet[key]) for key in ("median_s", "mean_s", "share_under_5s", "per_alive_minute",
                                                  "home_share")), quiet


def test_truth_pooling_is_one_summary_of_the_concatenation():
    codes, graded = summary_toy()
    other = runs((pm.H, 100))
    pooled, pooled_graded = truth.concatenate([(codes, graded), (other, ((1, 0, 100),))])
    assert len(pooled) == len(codes) + 1 + len(other), len(pooled)
    assert pooled[len(codes)] == pm.X, "one X row between the matches"
    assert pooled_graded == ((1, 0, 100), (2, 110, 180), (1, 186, 286)), pooled_graded
    summary = truth.summary(pooled, pooled_graded, 10)
    assert summary["possessions"] == 7 and summary["stoppages"] == 3, summary
    # The median of all seven lengths (1 2 6 28 30 59 100) is 28 frames; the average of the two matches' medians (17
    # and 100 frames) would be 58.5.
    assert math.isclose(summary["median_s"], 2.8), summary["median_s"]
    assert math.isclose(summary["mean_s"], 226 / 7 / 10), summary["mean_s"]
    # The scorer reads the pooled truth as three graded runs: nothing merges across the X row.
    assert pm.valid_runs(pooled) == [(0, 100), (110, 180), (186, 286)], pm.valid_runs(pooled)
    assert truth.concatenate([(codes, graded)])[0].tolist() == codes.tolist()
    expect_error(ValueError, lambda: truth.concatenate([]), "nothing to concatenate")


# -- the line timing --------------------------------------------------------------------------------------------------

def test_truth_line_timing():
    """Four alive -> dead changes at 2 fps (a window of 20 frames), with the ball beyond a line at chosen frames."""
    codes = np.full(102, pm.X, dtype=np.int8)
    codes[0:60] = runs((pm.H, 25), (pm.D, 5), (pm.A, 20), (pm.D, 10))                         # changes at 25 and 50
    codes[62:102] = runs((pm.D, 2), (pm.A, 4), (pm.D, 8), (pm.H, 10), (pm.D, 16))             # at 68 and 86 (not 62)
    ball = np.full((102, 2), 0.5)
    ball[[22, 23, 24]] = np.nan                                  # no ball: neither beyond nor in the way
    ball[4] = (-0.01, 0.5)                                       # beyond, 21 frames before the change at 25: outside
    ball[20] = (0.5, -0.2)                                       # beyond (y < 0), 5 frames before 25: the first one
    ball[25] = (1.5, 0.5)                                        # beyond at the change itself: the window stops before
    ball[29] = (0.5, 1.2)                                        # beyond, 21 frames before 50: outside
    ball[30] = (1.01, 0.5)                                       # beyond (x > 1), 20 frames before 50: the lower edge
    ball[35] = (0.5, 1.2)                                        # a later one: the first counts, not this
    ball[55] = (0.5, 1.5)                                        # beyond in the first half's dead tail
    ball[61] = (0.5, 1.5)                                        # the separator row: before the second half, clipped
    ball[64] = (0.5, 1.2)                                        # beyond, 4 frames before 68 (22 before 86)
    # The ball is beyond a line at 64 and nowhere in [66, 86): the change at 86 has none.
    match = toy_match(np.full(102, ALIVE), np.full(102, HOME), ((1, 0, 60), (2, 62, 102)), ball_xy=ball)
    result = truth.line_timing(match, codes)
    # Changes 25 (lead (25 - 20) / 2 = 2.5 s), 50 ((50 - 30) / 2 = 10 s), 68 ((68 - 64) / 2 = 2 s), 86 (none).
    assert result["changes"] == 4 and result["with_beyond_line"] == 3, result
    assert math.isclose(result["share_beyond_line"], 0.75), result
    assert math.isclose(result["median_s"], 2.5), result
    assert math.isclose(result["p25_s"], 2.25) and math.isclose(result["p75_s"], 6.25), result
    assert result["leads_s"] == (2.5, 10.0, 2.0), result["leads_s"]
    # Pooled, the statistics are taken over all four leads (the median is (2 + 2.5) / 2), not averaged over the matches.
    other = truth.line_timing(toy_match(np.full(60, ALIVE), np.full(60, HOME), ((1, 0, 60),),
                                        ball_xy=np.where(np.arange(60)[:, None] == 18, (0.5, 1.2), (0.5, 0.5))),
                              runs((pm.H, 20), (pm.D, 40)))
    assert other["leads_s"] == (1.0,), other
    pooled = truth.pool_line_timing([result, other])
    assert pooled["changes"] == 5 and pooled["with_beyond_line"] == 4 and math.isclose(pooled["share_beyond_line"], 0.8)
    assert pooled["leads_s"] == (2.5, 10.0, 2.0, 1.0) and math.isclose(pooled["median_s"], 2.25), pooled
    # No change, or no change with a ball beyond a line, leaves the lead NaN and the counts honest.
    still = truth.line_timing(match, runs((pm.H, 102)))
    assert still["changes"] == 0 and still["with_beyond_line"] == 0 and math.isnan(still["median_s"]), still
    assert math.isnan(still["share_beyond_line"]), still
    inside = toy_match(np.full(102, ALIVE), np.full(102, HOME), ((1, 0, 60), (2, 62, 102)),
                       ball_xy=np.full((102, 2), 0.5))
    none = truth.line_timing(inside, codes)
    assert none["changes"] == 4 and none["with_beyond_line"] == 0 and math.isnan(none["median_s"]), none
    assert none["share_beyond_line"] == 0.0, none
    # The first ball beyond a line in each window of the toy above is on an alive frame, so alive_only changes nothing.
    assert truth.line_timing(match, codes, alive_only=True) == result


def test_truth_line_timing_alive_only_skips_an_earlier_stop():
    """A throw-in 10 s or less before the next stop: its ball is beyond a line, on dead frames."""
    codes = runs((pm.D, 10), (pm.H, 10), (pm.D, 40))                # at 2 fps: the change is at 20, the window [0, 20)
    ball = np.full((60, 2), 0.5)
    ball[5] = (0.5, 1.1)                                            # beyond, on a dead frame
    ball[15] = (-0.05, 0.5)                                         # beyond, on an alive frame
    match = toy_match(np.full(60, ALIVE), np.full(60, HOME), ((1, 0, 60),), ball_xy=ball)
    every = truth.line_timing(match, codes)
    assert every["changes"] == 1 and math.isclose(every["median_s"], 7.5), every          # (20 - 5) / 2: the stop's
    alive = truth.line_timing(match, codes, alive_only=True)
    assert alive["changes"] == 1 and math.isclose(alive["median_s"], 2.5), alive          # (20 - 15) / 2
    ball[15] = (0.5, 0.5)                                           # nothing beyond a line on an alive frame
    alive = truth.line_timing(match, codes, alive_only=True)
    assert alive["with_beyond_line"] == 0 and alive["share_beyond_line"] == 0.0, alive
    assert truth.line_timing(match, codes)["with_beyond_line"] == 1


# -- the nearest team -------------------------------------------------------------------------------------------------

def test_nearest_team_bruteforce():
    """A toy frame, then a loop of math.hypot against random frames with players missing."""
    # Four players: columns 0 and 2 are home, 1 and 3 are away. The pitch is 100 x 50 m.
    teams = (HOME, AWAY, HOME, AWAY)
    nan = (np.nan, np.nan)
    frames = np.array([
        # 0: the ball (0.5, 0.5) is 9 m from player 0 (home) and 7.5 m from player 1 (away): normalised units say
        #    0.09 against 0.15 and would pick home; metres pick away.
        [(0.59, 0.5), (0.5, 0.65), nan, nan],
        # 1: players 1 (away) and 2 (home) are equally far: the lower column, 1, wins -> away.
        [nan, (0.5, 0.6), (0.5, 0.4), nan],
        # 2: players 0 (home) and 3 (away) are equally far: column 0 -> home.
        [(0.4, 0.5), nan, nan, (0.6, 0.5)],
        # 3: the nearest player is absent (NaN) and does not count: the next nearest, player 3 (away), does.
        [nan, nan, nan, (0.9, 0.9)],
        # 4: nobody on the pitch: no answer.
        [nan, nan, nan, nan],
        # 5: no ball: no answer, whoever is there.
        [(0.5, 0.5), (0.5, 0.5), nan, nan],
        # 6: a row that is not a frame (the separator): no ball, no players.
        [nan, nan, nan, nan]])
    ball = np.array([(0.5, 0.5)] * 5 + [(np.nan, np.nan)] * 2)
    match = toy_match([ALIVE] * 7, [HOME] * 7, ((1, 0, 7),), ball_xy=ball, players_xy=frames, player_team=teams)
    nearest = truth.nearest_team(match)
    assert nearest.dtype == np.int8 and nearest.shape == (7,), (nearest.dtype, nearest.shape)
    assert nearest.tolist() == [AWAY, AWAY, HOME, AWAY, -1, -1, -1], nearest.tolist()

    rng = np.random.default_rng([6, 1])
    n, players = 400, 7
    xy = rng.uniform(0, 1, size=(n, players, 2))
    xy[rng.uniform(size=(n, players)) < 0.3] = np.nan
    ball = rng.uniform(0, 1, size=(n, 2))
    ball[rng.uniform(size=n) < 0.2] = np.nan
    xy[:20, :] = np.nan                                              # some frames with nobody at all
    team = (0, 1, 0, 1, 1, 0, 0)
    match = toy_match([ALIVE] * n, [HOME] * n, ((1, 0, n),), ball_xy=ball, players_xy=xy, player_team=team,
                      pitch=(105.0, 68.0))
    expected = []
    for frame in range(n):
        best, best_team = math.inf, -1
        if not math.isnan(ball[frame, 0]):
            for column in range(players):
                if math.isnan(xy[frame, column, 0]):
                    continue
                distance = math.hypot((xy[frame, column, 0] - ball[frame, 0]) * 105.0,
                                      (xy[frame, column, 1] - ball[frame, 1]) * 68.0)
                if distance < best:                                    # strict: a tie keeps the lower column
                    best, best_team = distance, team[column]
        expected.append(best_team)
    assert truth.nearest_team(match).tolist() == expected
    assert min(expected) == -1 and 0 in expected and 1 in expected


def test_truth_nearest_agreement_counts_alive_ball_visible_frames():
    codes = np.array([pm.H, pm.A, pm.D, pm.H, pm.X, pm.A, pm.H], dtype=np.int8)
    nearest = np.array([HOME, AWAY, HOME, AWAY, HOME, -1, HOME], dtype=np.int8)
    # Alive frames with a nearest team: 0, 1 and 6 agree, 3 does not; the D, the X and the frame without a nearest team
    # (5) are left out.
    assert truth.nearest_agreement(codes, nearest, [(0, 7)]) == (3, 4)
    assert truth.nearest_agreement(codes, nearest, [(0, 2), (3, 4)]) == (2, 3)
    assert truth.nearest_agreement(codes, nearest, [(0, 1)]) == (1, 1)
    assert truth.nearest_agreement(codes, nearest, [(2, 3)]) == (0, 0)


# -- the coarse truth -------------------------------------------------------------------------------------------------

def coarse_reference(codes, spans, min_frames):
    """coarse_codes by the plainest route: re-segment the whole array with the scorer's pm.segments after every step.

    Each step takes, of the spells shorter than min_frames that sit between two spells of the other team, the shortest
    (the earliest of equals), and gives its frames that team's label; it stops when there is none.
    """
    codes = codes.copy()
    while True:
        best = None
        pieces = pm.segments(codes, spans)
        for index in range(1, len(pieces) - 1):
            before, here, after = pieces[index - 1], pieces[index], pieces[index + 1]
            sandwiched = (here[2] in (pm.H, pm.A) and before[2] == after[2] and before[2] in (pm.H, pm.A)
                          and before[2] != here[2] and before[1] == here[0] and here[1] == after[0])
            length = here[1] - here[0]
            if sandwiched and length < min_frames and (best is None or length < best[0]):
                best = (length, here[0], here[1], before[2])
        if best is None:
            return codes
        codes[best[1]:best[2]] = best[3]


def random_halves(seed, frames=2000):
    """Two graded halves of random runs of H, A and D (short and long), X between and after: (codes, graded)."""
    rng = np.random.default_rng([6, seed])
    pieces = []
    for half in range(2):
        length = 0
        while length < frames:
            run = int(rng.integers(1, 40)) if rng.uniform() < 0.7 else int(rng.integers(40, 200))
            pieces.append(runs((int(rng.choice([pm.H, pm.A, pm.H, pm.A, pm.D])), run)))
            length += run
        pieces.append(runs((pm.X, 30)))
    codes = np.concatenate(pieces)
    return codes, tuple((1 + half, start, stop) for half, (start, stop) in enumerate(pm.valid_runs(codes)))


def test_truth_coarse_absorbs_sandwiched_short_spells():
    fps, min_s = 10, 0.8                                       # a spell shorter than 8 frames

    def coarse(codes, spans=None):
        return truth.coarse_codes(codes, ((1, 0, len(codes)),) if spans is None else spans, fps, min_s)

    # A sandwiched short spell takes its neighbours' team, and the three merge into one possession.
    codes = runs((pm.H, 20), (pm.A, 5), (pm.H, 20))
    assert coarse(codes).tolist() == runs((pm.H, 45)).tolist()
    assert codes.tolist() == runs((pm.H, 20), (pm.A, 5), (pm.H, 20)).tolist(), "the input is not changed"
    # A spell of exactly min_s (8 frames) is not short; one frame fewer is.
    kept = runs((pm.H, 20), (pm.A, 8), (pm.H, 20))
    assert coarse(kept).tolist() == kept.tolist()
    assert coarse(runs((pm.H, 20), (pm.A, 7), (pm.H, 20))).tolist() == runs((pm.H, 47)).tolist()
    # ... for every float, not only for 0.8 x 10 (which happens to be exact): at 25 fps, 0.28 x 25 is 7.000000000000001
    # and 2.2 x 25 is 55.00000000000001, and a plain `length < min_s * fps` would absorb a spell of exactly 7 or 55.
    for rate, seconds, frames in ((25, 0.28, 7), (25, 2.2, 55), (25, 0.8, 20), (10, 0.8, 8), (25, 5, 125)):
        edge = runs((pm.H, 200), (pm.A, frames), (pm.H, 200))
        assert truth.coarse_codes(edge, ((1, 0, len(edge)),), rate, seconds).tolist() == edge.tolist(), (
            "a spell of exactly {} s ({} frames at {} fps) must stay".format(seconds, frames, rate))
        short = runs((pm.H, 200), (pm.A, frames - 1), (pm.H, 200))
        assert truth.coarse_codes(short, ((1, 0, len(short)),), rate, seconds).tolist() == \
            runs((pm.H, 399 + frames)).tolist(), "one frame fewer than {} s at {} fps must go".format(seconds, rate)
    # A short spell next to D is not sandwiched by two spells, so it stays; D is never touched.
    for same in (runs((pm.H, 20), (pm.A, 5), (pm.D, 5), (pm.H, 20)), runs((pm.D, 5), (pm.A, 5), (pm.H, 20)),
                 runs((pm.H, 20), (pm.A, 5), (pm.D, 5)), runs((pm.H, 20), (pm.D, 5), (pm.A, 5), (pm.D, 5), (pm.H, 20))):
        assert coarse(same).tolist() == same.tolist()
    # Neighbours of the other team on both sides are needed: H A D A is not A A A.
    assert coarse(runs((pm.A, 20), (pm.H, 3), (pm.D, 3), (pm.A, 20))).tolist() == \
        runs((pm.A, 20), (pm.H, 3), (pm.D, 3), (pm.A, 20)).tolist()
    # The first and the last spell of a half have only one neighbour.
    ends = runs((pm.A, 3), (pm.H, 20), (pm.A, 3))
    assert coarse(ends).tolist() == ends.tolist()

    # A chain of short spells is absorbed shortest first, one at a time: H10 A6 H3 A5 H10 takes the H3 into the A's
    # (A14), and then nothing is sandwiched. Relabelling every candidate at once would give H16 A3 H15.
    chain = runs((pm.H, 10), (pm.A, 6), (pm.H, 3), (pm.A, 5), (pm.H, 10))
    assert coarse(chain).tolist() == runs((pm.H, 10), (pm.A, 14), (pm.H, 10)).tolist()
    # Equals go to the earliest: A10 B4 A4 B10 absorbs the first short spell (A18 B10), not the second (A10 B18).
    tie = runs((pm.A, 10), (pm.H, 4), (pm.A, 4), (pm.H, 10))
    assert coarse(tie).tolist() == runs((pm.A, 18), (pm.H, 10)).tolist()
    # Merging can make a spell long enough to stay, or leave one short enough to go in the next round.
    cascade = runs((pm.H, 10), (pm.A, 5), (pm.H, 2), (pm.A, 5), (pm.H, 10))
    assert coarse(cascade).tolist() == runs((pm.H, 10), (pm.A, 12), (pm.H, 10)).tolist()
    ladder = runs((pm.H, 3), (pm.A, 3), (pm.H, 3), (pm.A, 3), (pm.H, 3))
    assert coarse(ladder).tolist() == runs((pm.H, 15)).tolist(), "A3 goes, and then the second A3 is sandwiched in turn"

    # Halves are separate: a short spell at the head of the second half is not sandwiched by the first half's last one.
    halves = np.concatenate([runs((pm.A, 20)), runs((pm.X, 5)), runs((pm.A, 3), (pm.H, 20), (pm.A, 20))])
    spans = ((1, 0, 20), (2, 25, 68))
    assert coarse(halves, spans).tolist() == halves.tolist()
    # ... and nothing outside the halves changes.
    around = np.concatenate([runs((pm.X, 4)), runs((pm.H, 20), (pm.A, 5), (pm.H, 20)), runs((pm.X, 4))])
    result = coarse(around, ((1, 4, 49),))
    assert result.tolist() == np.concatenate([runs((pm.X, 4)), runs((pm.H, 45)), runs((pm.X, 4))]).tolist()
    assert pm.valid_runs(result) == pm.valid_runs(around)
    expect_error(ValueError, lambda: truth.coarse_codes(codes, ((1, 0, len(codes)),), fps, -1), "min_s", "-1")


def test_truth_coarse_is_idempotent_and_equals_the_plain_reference():
    fps, absorbed = 10, 0
    for seed in range(12):
        codes, graded = random_halves(seed)
        spans = [(start, stop) for _, start, stop in graded]
        for min_s in (0.5, 1.0, 2.5, 6.0):
            once = truth.coarse_codes(codes, graded, fps, min_s)
            assert once.dtype == np.int8 and once.shape == codes.shape
            assert once.tolist() == coarse_reference(codes, spans, min_s * fps).tolist(), (seed, min_s)
            assert truth.coarse_codes(once, graded, fps, min_s).tolist() == once.tolist(), "coarsening twice is once"
            assert pm.valid_runs(once) == pm.valid_runs(codes)
            assert ((codes == pm.D) == (once == pm.D)).all(), "no D is made or lost"
            assert ((codes == pm.X) == (once == pm.X)).all(), "no X is made or lost"
            absorbed += len(pm.possession_spells(codes, spans)) - len(pm.possession_spells(once, spans))
    assert absorbed > 250, "the random halves must contain spells to absorb, or the comparison above proves little"


# -- the lag ----------------------------------------------------------------------------------------------------------

def test_truth_lagged_shifts_each_half_and_scores_by_hand():
    codes = np.concatenate([runs((pm.H, 4), (pm.A, 3)), runs((pm.X, 2)), runs((pm.D, 2), (pm.H, 3))])
    # Within each half the truth moves k frames later; the first k frames of the half are X, and nothing crosses
    # the hole between the halves.
    assert truth.lagged(codes, 0).tolist() == codes.tolist()
    shifted = truth.lagged(codes, 2)
    expected = runs((pm.X, 2), (pm.H, 4), (pm.A, 1), (pm.X, 2), (pm.X, 2), (pm.D, 2), (pm.H, 1))
    assert shifted.tolist() == expected.tolist(), shifted.tolist()
    assert truth.lagged(codes, 9).tolist() == runs((pm.X, 14)).tolist(), "a lag longer than the half leaves only X"
    # Two possessions of 100 frames, lagged by 45: H 0-144 and A 145-199 against H 0-99 and A 100-199. The H pair has
    # IoU 100/145 = 0.690, the A pair 55/100 = 0.55, so per IoU threshold 0.3 0.4 0.5 0.6 0.7 the F1 is 1 1 1 0.5 0.
    two = runs((pm.H, 100), (pm.A, 100))
    assert truth.lag_spell_f1(two, 0) == 1.0
    assert math.isclose(truth.lag_spell_f1(two, 45), (1 + 1 + 1 + 0.5 + 0) / 5), truth.lag_spell_f1(two, 45)
    assert truth.lag_spell_f1(two, 1) == 1.0, "a one-frame lag is free"


def public_gt():
    """The organisers' public ground truth (evaluation/gt/, MIT) as codes, found as the scorer finds it."""
    import run_evaluation
    return pm.label_codes(pm.read_label_csv(run_evaluation.find_gt("public")))


def test_truth_lag_method_on_the_public_gt():
    """The shift of the lag table, applied to the public GT, gives the public GT's own figures, not EVALUATION.md's.

    The table's k = 1 (1.0000) is reproduced; k = 3, 8 and 25 are not (0.9818, 0.8933, 0.6539 there). EVALUATION.md does
    not say what the table was computed on, and nothing is inferred from it. The shift direction and whether the head is
    left X or kept make no difference on the public GT (asserted below), so the method is not what differs. The
    figures are pinned here so that a change to the scorer or to lagged() shows.
    """
    gt = public_gt()
    assert len(gt) == 199319 and pm.valid_runs(gt) == [(11185, 18685)], pm.valid_runs(gt)
    measured = {k: truth.lag_spell_f1(gt, k) for k in EVALUATION_LAG_TABLE}
    for k, value in PUBLIC_LAG.items():
        assert abs(measured[k] - value) < 5e-5, (k, measured[k], value)
    assert abs(measured[1] - EVALUATION_LAG_TABLE[1]) < 5e-5
    # Neither the direction of the shift nor what fills the head changes the public GT's figures.
    (start, stop), = pm.valid_runs(gt)
    for k in (3, 8, 25):
        for name, shifted in (("earlier", np.concatenate([gt[start + k:stop], gt[stop - 1:stop].repeat(k)])),
                              ("later, head kept",
                               np.concatenate([gt[start:start + 1].repeat(k), gt[start:stop - k]]))):
            pred = gt.copy()
            pred[start:stop] = shifted
            assert abs(pm.evaluate(gt, pred)["spell_f1_mean"] - measured[k]) < 1e-12, (name, k)


# -- the downloaded matches -------------------------------------------------------------------------------------------

def playing_time(match_id):
    """{half: (TotalTime, PlayingTime)} in ms from OtherGameInformation of a match's information file."""
    from lxml import etree
    from pipeline.benchmark import sources
    root = etree.parse(sources.local_path(match_id, "information")).getroot()
    other = root.find(".//OtherGameInformation")
    return {half: tuple(float(other.get(name)) for name in names) for half, names in PLAYING_TIME_ATTRIBUTES.items()}


def measure(match_id):
    """Everything the --dfl checks of this module report about one match, as arrays of codes and small numbers.

    test_dfl_truth_conventions, test_dfl_window_confound and test_dfl_coarse_truth all read this (through
    measurements()), so the 7 matches are loaded and measured once. The match itself (about 150 MB) is dropped on
    return, so that one match is held at a time.
    """
    match = dfl.load_match(match_id)
    codes, report = truth.truth_codes(match)
    spans = [(start, stop) for _, start, stop in match.graded]
    assert report["graded_frames"] == sum(stop - start for start, stop in spans), report
    assert (codes != pm.X).sum() == report["graded_frames"] and pm.valid_runs(codes) == spans, match_id
    itself = pm.evaluate(codes, codes)
    assert itself["spell_f1_mean"] == 1.0 and itself["macro_f1"] == 1.0, match_id

    nearest = truth.nearest_team(match)
    agree = {"all": truth.nearest_agreement(codes, nearest, spans)}
    for half, start, stop in match.graded:
        agree[half] = truth.nearest_agreement(codes, nearest, [(start, stop)])
    first = match.graded[0]
    x, y = match.ball_xy[:, 0], match.ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        beyond = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    return {
        "leading_dead": [int(np.argmax(codes[start:stop] != pm.D)) for _, start, stop in match.graded],
        "ends_dead": [bool(codes[stop - 1] == pm.D) for _, _, stop in match.graded],
        "dead_beyond": float(beyond[codes == pm.D].mean()),
        "ball_extent": (float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))),
        "codes": codes, "graded": match.graded, "report": report, "fps": match.fps,
        "summary": truth.summary(codes, match.graded, match.fps),
        "first_window": (codes, ((1, first[1], first[1] + WINDOW_FRAMES),)),
        "windows": [truth.summary(codes, ((half, begin, begin + WINDOW_FRAMES),), match.fps)    # every whole 5 minutes
                    for half, start, stop in match.graded
                    for begin in range(start, stop - WINDOW_FRAMES + 1, WINDOW_FRAMES)],
        "timing": truth.line_timing(match, codes), "alive_timing": truth.line_timing(match, codes, alive_only=True),
        "alive_frames": int(((codes == pm.H) | (codes == pm.A)).sum()),
        "alive_by_half": {half: int(((codes[start:stop] == pm.H) | (codes[start:stop] == pm.A)).sum())
                          for half, start, stop in match.graded},
        "agree": agree, "lag": {k: truth.lag_spell_f1(codes, k) for k in LAGS}, "playing": playing_time(match_id)}


@functools.lru_cache(maxsize=1)
def measurements():
    """measure() of all 7 matches, once: the --dfl checks of this module share it, and the first of them pays (8 s)."""
    from pipeline.benchmark import sources
    return {match_id: measure(match_id) for match_id in sources.MATCH_IDS}


def common_fps(found):
    """The one sampling rate of the matches (the checks divide by it), or an assertion that says they differ."""
    rates = {data["fps"] for data in found.values()}
    assert len(rates) == 1, "the matches are not all sampled at one rate: {}".format(sorted(rates))
    return rates.pop()


def print_summary_rows(rows):
    """Print the table of summaries [(name, summary)] that A06 quotes."""
    print("      {:<14} {:>7} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6} {:>5} {:>6} {:>6} {:>7}".format(
        "", "minutes", "alive", "poss", "per_min", "med_s", "mean_s", "<5 s", "<=2f", "<=12f", "stops", "stop_s"))
    for name, summary in rows:
        print(("      {:<14} {:>7.1f} {:>5.1%} {:>6} {:>7.2f} {:>7.2f} {:>6.2f} {:>5.1%} {:>5} {:>6} {:>6} "
               "{:>7.1f}").format(
            name, summary["minutes"], summary["alive_share"], summary["possessions"], summary["per_alive_minute"],
            summary["median_s"], summary["mean_s"], summary["share_under_5s"], summary["runs_le_2_frames"],
            summary["runs_le_12_frames"], summary["stoppages"], summary["stoppage_median_s"]))


def public_summary():
    gt = public_gt()
    return truth.summary(gt, tuple((1, start, stop) for start, stop in pm.valid_runs(gt)), pm.FPS)


def test_dfl_truth_conventions():
    """DFL's truth, per match and pooled, next to the public window: what A06 in docs/results.md quotes.

    Prints, for each of the 7 matches, the summary of the truth (and its pooled summary, the public GT's, and that of
    the first 5 minutes), DFL's own PlayingTime against the alive frames, the line timing, the nearest team's agreement
    and the cost of a lag. (The public window's place among DFL's 5-minute windows is test_dfl_window_confound's, and
    the coarse truth is test_dfl_coarse_truth's.) Asserts that the nearest player's team is the truth's on more than
    60% of alive ball-visible frames in every match and every half (which proves the home mapping), that every truth is
    a perfect score against itself, that the alive frames are DFL's PlayingTime to within MAX_PLAYING_DIFF, and the
    integer counts that A06 quotes for the 7 matches, the public window and the first 5 minutes, and the three matches
    whose first line-timing measurement has its upper quartile at the window's whole length (EXPECTED_LEAD_CAPPED).
    """
    from pipeline.benchmark import sources
    ids = sources.MATCH_IDS
    found = measurements()
    fps = common_fps(found)

    pooled_codes, pooled_graded = truth.concatenate([(found[m]["codes"], found[m]["graded"]) for m in ids])
    pooled = truth.summary(pooled_codes, pooled_graded, fps)
    gt = public_gt()
    public = public_summary()
    first = truth.summary(*truth.concatenate([found[m]["first_window"] for m in ids]), fps)

    print_summary_rows([(m, found[m]["summary"]) for m in ids] + [("pooled", pooled), ("public", public),
                                                                   ("first 5'", first)])
    print("      pooled: one summary of the 7 matches laid end to end; public: the 5-minute public GT; first 5': "
          "the first 5 minutes of each match's first half, pooled")
    print("      the pooled home share is {:.1%}, the public window's {:.1%}; the carried flags were: {}".format(
        pooled["home_share"], public["home_share"], ", ".join(
            "{} {}/{}".format(m, found[m]["report"]["carried_possession_frames"],
                              found[m]["report"]["carried_status_frames"]) for m in ids)))

    # DFL's own count of the time the ball is in play, against the frames the truth calls alive.
    print("\n      {:<8} {:>13} {:>10} {:>10} {:>13} {:>9} {:>9} {:>9}".format(
        "match", "playing/total", "of half 1", "of half 2", "alive frames", "minus PT", "half 1", "half 2"))
    for m in ids:
        times = found[m]["playing"]
        played = {half: times[half][1] * fps / 1000 for half in times}        # PlayingTime in frames, per half
        print("      {:<8} {:>13.1%} {:>10.1%} {:>10.1%} {:>13,} {:>+9.0f} {:>+9.0f} {:>+9.0f}".format(
            m, sum(times[half][1] for half in times) / sum(times[half][0] for half in times),
            times[1][1] / times[1][0], times[2][1] / times[2][0], found[m]["alive_frames"],
            found[m]["alive_frames"] - sum(played.values()),
            *(found[m]["alive_by_half"][half] - played[half] for half in (1, 2))))
    print("      (PlayingTime and TotalTime are OtherGameInformation attributes of the information file; 'minus PT' "
          "is the alive frames less PlayingTime converted to frames, in all and in each half)")
    for m in ids:
        times = found[m]["playing"]
        played = {half: times[half][1] * fps / 1000 for half in times}
        assert abs(found[m]["alive_frames"] - sum(played.values())) <= MAX_PLAYING_DIFF[0] and all(
            abs(found[m]["alive_by_half"][half] - played[half]) <= MAX_PLAYING_DIFF[1] for half in played), (
            "{}: the alive frames are no longer DFL's PlayingTime to within {} frames in all and {} in each "
            "half".format(m, *MAX_PLAYING_DIFF))

    for title, key in (("in the 10 s before the flip, whatever the truth calls those frames", "timing"),
                       ("looking only at the frames of that window that the truth calls alive", "alive_timing")):
        print("\n      alive -> dead changes with the ball beyond a line, {}:".format(title))
        print("      {:<8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}".format(
            "match", "changes", "beyond", "share", "median_s", "p25_s", "p75_s"))
        for m in ids:
            t = found[m][key]
            print("      {:<8} {:>8} {:>8} {:>7.1%} {:>9.2f} {:>8.2f} {:>8.2f}".format(
                m, t["changes"], t["with_beyond_line"], t["share_beyond_line"], t["median_s"], t["p25_s"], t["p75_s"]))
        t = truth.pool_line_timing([found[m][key] for m in ids])
        print("      {:<8} {:>8} {:>8} {:>7.1%} {:>9.2f} {:>8.2f} {:>8.2f}".format(
            "pooled", t["changes"], t["with_beyond_line"], t["share_beyond_line"], t["median_s"], t["p25_s"],
            t["p75_s"]))
    capped = tuple(m for m in ids if found[m]["timing"]["p75_s"] >= truth.LINE_WINDOW_S - 1e-9)
    assert capped == EXPECTED_LEAD_CAPPED, "the upper quartile is the window's whole length in {}".format(capped)

    print("\n      dead frames with the ball beyond a line, and how far the ball goes (x min, x max, y min, y max):")
    for m in ids:
        print("      {:<8} {:>6.1%}   {:+.3f} {:+.3f} {:+.3f} {:+.3f}".format(m, found[m]["dead_beyond"],
                                                                              *found[m]["ball_extent"]))
    leading = [frames for m in ids for frames in found[m]["leading_dead"]]
    print("      halves that start with D: {} of {}, with {} leading D frames; halves that end in D: {} of {}".format(
        sum(1 for frames in leading if frames), len(leading), sorted(frames for frames in leading if frames),
        sum(ends for m in ids for ends in found[m]["ends_dead"]), len(leading)))

    print("\n      the nearest player's team is the truth's, on alive ball-visible frames (A01: 77.8% of 5,640 on the "
          "public window):")
    print("      {:<8} {:>9} {:>9} {:>9} {:>10}".format("match", "all", "half 1", "half 2", "frames"))
    for m in ids:
        shares = {key: hit / total for key, (hit, total) in found[m]["agree"].items()}
        print("      {:<8} {:>9.1%} {:>9.1%} {:>9.1%} {:>10,}".format(
            m, shares["all"], shares[1], shares[2], found[m]["agree"]["all"][1]))
        for key, share in shares.items():
            assert share > MIN_AGREEMENT, (
                "{}: the nearest player names the truth's team on {:.1%} of {} frames; home and away may be inverted, "
                "or a half misaligned".format(m, share, key))
    hits = sum(found[m]["agree"]["all"][0] for m in ids)
    total = sum(found[m]["agree"]["all"][1] for m in ids)
    print("      {:<8} {:>9.1%} {:>29,}".format("pooled", hits / total, total))

    print("\n      spellF1 of the truth against itself lagged k frames:")
    print("      {:<8} ".format("match") + " ".join("{:>8}".format("k=" + str(k)) for k in LAGS))
    pooled_lag = {k: truth.lag_spell_f1(pooled_codes, k) for k in LAGS}
    public_lag = {k: truth.lag_spell_f1(gt, k) for k in (1,) + LAGS}
    for name, values in [(m, found[m]["lag"]) for m in ids] + [("pooled", pooled_lag), ("public", public_lag),
                                                               ("table", EVALUATION_LAG_TABLE)]:
        print("      {:<8} ".format(name) + " ".join("{:>8.4f}".format(values[k]) for k in LAGS))
    print("      public: the same shift on the public GT; table: EVALUATION.md ('perfect but k frames late'); k=1 is "
          "{:.4f} on the public GT and {:.4f} in the table".format(public_lag[1], EVALUATION_LAG_TABLE[1]))
    assert pooled_lag[3] > pooled_lag[8] > pooled_lag[25], pooled_lag

    # DFL's halves start at the kick-off and end with the final whistle.
    assert sorted(frames for frames in leading if frames) == [1, 1, 48] and len(leading) == 14, leading
    assert all(ends for m in ids for ends in found[m]["ends_dead"])

    # The counts that A06 quotes.
    counts = {m: (found[m]["summary"]["possessions"], found[m]["summary"]["stoppages"], found[m]["timing"]["changes"],
                  found[m]["timing"]["with_beyond_line"], found[m]["alive_timing"]["with_beyond_line"],
                  found[m]["agree"]["all"][1]) for m in ids}
    assert counts == EXPECTED_COUNTS, "the counts A06 quotes changed:\n{}\nvs\n{}".format(counts, EXPECTED_COUNTS)
    assert pooled["runs_le_2_frames"] <= FLICKER_SHARE * pooled["possessions"], (
        "{} of {} possessions last 2 frames or less: the flicker is material, so truth_codes needs the min_run variant "
        "that the plan describes (a run shorter than min_run, with the same other team on both sides, takes that "
        "team)".format(pooled["runs_le_2_frames"], pooled["possessions"]))
    assert (pooled["possessions"], pooled["stoppages"], pooled["runs_le_2_frames"],
            pooled["runs_le_12_frames"]) == EXPECTED_POOLED, (
        pooled["possessions"], pooled["stoppages"], pooled["runs_le_2_frames"], pooled["runs_le_12_frames"])
    short = {m: (found[m]["summary"]["runs_le_2_frames"], found[m]["summary"]["runs_le_12_frames"]) for m in ids}
    assert short == EXPECTED_SHORT_RUNS, "the short-run counts A06 quotes changed: {} vs {}".format(
        short, EXPECTED_SHORT_RUNS)
    for name, summary, expected in (("public window", public, EXPECTED_PUBLIC), ("first 5'", first, EXPECTED_FIRST_5)):
        got = (summary["possessions"], summary["runs_le_2_frames"], summary["runs_le_12_frames"],
               summary["stoppages"])
        assert got == expected, "the {} counts A06 quotes changed: {} vs {}".format(name, got, expected)


def test_dfl_window_confound():
    """How lively a 5-minute window is decides its possession rate, so the public window's rate cannot size a grain.

    The public window is alive on 80.4% of its frames, livelier than 96% of DFL's windows, and in DFL the possession
    rate per alive minute falls as the alive share rises (a possession is split by every stoppage and a lively stretch
    has fewer of them). So the public window is set against the DFL windows that are as lively, and a line fitted to
    all of DFL's windows says what rate that liveliness predicts. Prints the table that A06 quotes, and asserts its
    integer counts, and that every selection of windows is drawn from all 7 matches.
    """
    from pipeline.benchmark import sources
    found = measurements()
    ids = sources.MATCH_IDS
    public = public_summary()
    rows = [(m, window) for m in ids for window in found[m]["windows"]]
    alive, rate, median, under = (np.array([window[key] for _, window in rows]) for key in
                                  ("alive_share", "per_alive_minute", "median_s", "share_under_5s"))
    assert len(rows) == EXPECTED_WINDOWS["all"] and np.isfinite(rate).all() and np.isfinite(median).all()
    assert len({m for m, _ in rows}) == EXPECTED_WINDOWS["matches"] == len(ids), "the windows are of all the matches"
    print("      the public window: {} possessions, {:.1%} alive, {:.2f} per alive minute, median {:.2f} s, {:.1%} "
          "under 5 s".format(public["possessions"], public["alive_share"], public["per_alive_minute"],
                             public["median_s"], public["share_under_5s"]))
    print("      {:<14} {:>4} {:>8} {:>12} {:>14} {:>14} {:>14} {:>14}".format(
        "DFL windows", "n", "matches", "median rate", "rate <= pub", "median of med", "median >= pub", "<5 s <= pub"))
    selections = {"all": alive >= 0, "alive >= 70%": alive >= 0.70, "alive >= 75%": alive >= 0.75}
    counts = {}
    for name, chosen in selections.items():
        n = int(chosen.sum())
        slow = int((rate[chosen] <= public["per_alive_minute"]).sum())
        long_median = int((median[chosen] >= public["median_s"]).sum())
        few_short = int((under[chosen] <= public["share_under_5s"] + 1e-9).sum())
        counts[name] = (n, slow, long_median, few_short)
        assert len({m for (m, _), keep in zip(rows, chosen) if keep}) == EXPECTED_WINDOWS["matches"], (
            "{}: the windows are no longer drawn from all {} matches".format(name, EXPECTED_WINDOWS["matches"]))
        print("      {:<14} {:>4} {:>8} {:>12.2f} {:>14} {:>14.2f} {:>14} {:>14}".format(
            name, n, len({m for (m, _), keep in zip(rows, chosen) if keep}), np.median(rate[chosen]),
            "{} ({:.1%})".format(slow, slow / n), np.median(median[chosen]),
            "{} ({:.1%})".format(long_median, long_median / n), "{} ({:.1%})".format(few_short, few_short / n)))
    possessions = np.array([window["possessions"] for _, window in rows])
    print("      windows at or below the public window's {} possessions: {} ({:.1%}); at or below its {:.1%} alive: "
          "{} ({:.1%})".format(public["possessions"], int((possessions <= public["possessions"]).sum()),
                               (possessions <= public["possessions"]).mean(), public["alive_share"],
                               int((alive <= public["alive_share"]).sum()), (alive <= public["alive_share"]).mean()))
    comparable = selections["alive >= 75%"]
    print("      rate of the comparable windows, over the public window's: {:.2f}".format(
        np.median(rate[comparable]) / public["per_alive_minute"]))
    correlation = float(np.corrcoef(alive, rate)[0, 1])
    slope, intercept = np.polyfit(alive, rate, 1)
    residual_sd = float(np.std(rate - (intercept + slope * alive), ddof=2))
    predicted = intercept + slope * public["alive_share"]
    print("      over all {} windows: r = {:.2f} between alive share and rate; fit rate = {:.2f} {:+.2f} x alive "
          "share; at {:.1%} alive it gives {:.2f} per alive minute, the public window's {:.2f} is {:.2f} residual sd "
          "(sd {:.2f}) from it".format(
              len(rows), correlation, intercept, slope, public["alive_share"], predicted, public["per_alive_minute"],
              (public["per_alive_minute"] - predicted) / residual_sd, residual_sd))
    assert correlation < -0.5, "the rate no longer falls with the alive share ({:.2f})".format(correlation)
    assert public["alive_share"] > 0.75, "the comparable windows (alive >= 75%) no longer include the public window's"
    assert int((alive <= public["alive_share"]).sum()) == EXPECTED_WINDOWS["alive_at_or_below"], (
        "windows at or below the public window's alive share: {}".format(int((alive <= public["alive_share"]).sum())))
    assert counts == EXPECTED_WINDOWS["counts"], counts


def test_dfl_coarse_truth():
    """The coarse truth's pre-registered choice: the smallest candidate that brings DFL to the literature's rate.

    For each candidate shortest possession (1 to 5 s) the 7 matches are coarsened and pooled, and the table shows the
    possessions per match and per alive minute, the median length and the share of the raw spells absorbed. The
    literature's 175-220 possessions per match, divided by DFL's own mean alive minutes per match, gives a rate per
    alive minute; COARSE_MIN_S is the smallest candidate at or below its upper bound, or the largest candidate if none
    is. The public window's rate is printed for reference only: it rests on 17 possessions and is not used.
    """
    from pipeline.benchmark import sources
    found = measurements()
    ids = sources.MATCH_IDS
    fps = common_fps(found)
    raw_codes, graded = truth.concatenate([(found[m]["codes"], found[m]["graded"]) for m in ids])
    spans = [(start, stop) for _, start, stop in graded]
    raw = truth.summary(raw_codes, graded, fps)
    raw_spells = pm.possession_spells(raw_codes, spans)
    alive_minutes = raw["alive_share"] * raw["minutes"] / len(ids)               # a match's alive minutes, on average
    low, high = (count / alive_minutes for count in truth.LITERATURE_POSSESSIONS)
    print("      DFL has {:.1f} alive minutes a match; the literature's {}-{} possessions are {:.2f}-{:.2f} per alive "
          "minute".format(alive_minutes, *truth.LITERATURE_POSSESSIONS, low, high))
    print("      {:<8} {:>12} {:>10} {:>9} {:>8} {:>9} {:>10} {:>9}".format(
        "min_s", "poss/match", "per alive", "median s", "mean s", "absorbed", "frames", "stops"))
    coarse, table = {}, {}
    for min_s in (0,) + truth.COARSE_CANDIDATES_S:
        coarse[min_s] = raw_codes if min_s == 0 else truth.coarse_codes(raw_codes, graded, fps, min_s)
        summary = truth.summary(coarse[min_s], graded, fps)
        absorbed = sum(1 for start, stop, _ in raw_spells if coarse[min_s][start] != raw_codes[start])
        changed = sum(int((coarse[min_s][a:b] != raw_codes[a:b]).sum()) for a, b in spans)
        table[min_s] = (summary["possessions"], summary["per_alive_minute"])
        print("      {:<8} {:>12.1f} {:>10.2f} {:>9.2f} {:>8.2f} {:>9.1%} {:>10.2%} {:>9}".format(
            "raw" if min_s == 0 else min_s, summary["possessions"] / len(ids), summary["per_alive_minute"],
            summary["median_s"], summary["mean_s"], absorbed / len(raw_spells),
            changed / sum(b - a for a, b in spans), summary["stoppages"]))
        assert pm.valid_runs(coarse[min_s]) == pm.valid_runs(raw_codes) and \
            ((coarse[min_s] == pm.D) == (raw_codes == pm.D)).all(), "the coarse truth changed a D or the halves"
    chosen = next((s for s in truth.COARSE_CANDIDATES_S if table[s][1] <= high), truth.COARSE_CANDIDATES_S[-1])
    reached = table[chosen][1] <= high
    print("      {} {:.2f} per alive minute: {} s{} (the public window's {:.2f}, 17 possessions, is for reference "
          "only)".format("the smallest candidate at or below" if reached else "no candidate reaches", high, chosen,
                         "" if reached else ", the largest, is taken and flagged",
                         public_summary()["per_alive_minute"]))
    assert truth.COARSE_MIN_S == chosen, "COARSE_MIN_S is {}, and the pre-registered rule gives {}".format(
        truth.COARSE_MIN_S, chosen)
    assert table[chosen][1] <= high or chosen == truth.COARSE_CANDIDATES_S[-1], table

    print()
    per_match = []
    for m in ids:
        per_match.append((m, truth.summary(truth.coarse_codes(found[m]["codes"], found[m]["graded"], fps),
                                           found[m]["graded"], fps)))
    coarse_pooled = truth.summary(coarse[chosen], graded, fps)
    print_summary_rows([("raw pooled", raw)] + [(m + " coarse", summary) for m, summary in per_match] +
                       [("coarse pooled", coarse_pooled)])
    assert (coarse_pooled["runs_le_2_frames"], coarse_pooled["runs_le_12_frames"]) == EXPECTED_COARSE_SHORT, (
        coarse_pooled["runs_le_2_frames"], coarse_pooled["runs_le_12_frames"])
    assert {m: summary["possessions"] for m, summary in per_match} == EXPECTED_COARSE_PER_MATCH, per_match
    print("\n      spellF1 of the coarse truth against itself lagged k frames, pooled:   " + "   ".join(
        "k={}: {:.4f}".format(k, truth.lag_spell_f1(coarse[chosen], k)) for k in LAGS))
    assert {s: table[s][0] for s in truth.COARSE_CANDIDATES_S} == EXPECTED_COARSE, table
