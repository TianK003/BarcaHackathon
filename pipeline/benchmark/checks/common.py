"""What several check modules share: the synthetic DFL match, the real match's memo, the leak detector and the console
set-up.

The synthetic match (write_synthetic_dfl and the squads, tables and helpers it is written from) is one fixture for
every check that needs a DFL file: the parser checks read it, and so will the cache's and every later stage's. It lives
here, in the real file format with made-up values, so that no check needs a download. real_match is how the --dfl
checks that share a downloaded match get it (dfl.load_match behind a one-match memo); a check that wants the cache's
own answer, built or reused, calls dfl.load_match itself. leaks (with truth_forms, held_by and as_numbers) is the walk
that finds the match, or an array equal to its flags or its truth, anywhere that a component's inputs reach: the checks
of the inputs, of the degrader and of the tracking profile all use it, so it lives here, where the last two reach it
without importing another check module. So do the other fixtures that more than one module builds on: random_match (the
DflMatch of random numbers), the synthetic Calibration and the bundle it is measured from, what the tracking and the cap
checks both use (the pairs that make wrong-team episodes, reference_episodes, the record that lambda was worked out
from), and what the runner's and the fidelity report's checks both use (loading, read, strict_json, yardstick_path). A
check module imports from here and from production code, and never from another check module. Import from here with
absolute imports (from pipeline.benchmark.checks.common import ...): a module of this package named like a production
module would otherwise shadow it.
"""

import collections
import contextlib
import dataclasses
import functools
import heapq
import itertools
import json
import os
import sys
import time
import types
from xml.sax.saxutils import quoteattr

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import degrade, dfl, hazard, truth
from pipeline.selftest import make_synthetic_bundle, synthetic_inputs, write_text

# Params
HOME, AWAY = 0, 1       # the codes of the teams in DflMatch.ball_possession and player_team
H, A, D, X = pm.H, pm.A, pm.D, pm.X     # the codes of a label array, as possession_metrics has them
# The configs of the yardsticks (nearest_w0 ... nearest_w125), which score.py's, run.py's and fidelity.py's checks load.
CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
MIN_AGREEMENT = 0.6                     # the nearest player must name the truth's team on more than this (home mapping)
# The home and guest team of each DFL match, as Bassek, Rein, Weber & Memmert (2025), Sci Data 12:195, Table 1 lists
# them. Typed from the paper and never taken from the files' General@HomeTeamName / GuestTeamName: the parser reads
# the files, so a table "fixed" from them would make this check compare the files with themselves.
DFL_TEAMS = {"J03WMX": ("1. FC Köln", "FC Bayern München"), "J03WN1": ("VfL Bochum 1848", "Bayer 04 Leverkusen"),
             "J03WOH": ("Fortuna Düsseldorf", "SSV Jahn Regensburg"),
             "J03WOY": ("Fortuna Düsseldorf", "F.C. Hansa Rostock"),
             "J03WPY": ("Fortuna Düsseldorf", "1. FC Nürnberg"), "J03WQQ": ("Fortuna Düsseldorf", "FC St. Pauli"),
             "J03WR9": ("Fortuna Düsseldorf", "1. FC Kaiserslautern")}
# Squads of the synthetic match: (PersonId, ShirtNumber, FirstName, LastName, Shortname), None leaves the attribute
# out. Each is listed in the file in an order that is not PersonId order, so a parser that keeps file order fails.
SYNTHETIC_HOME = ("DFL-CLU-SYNHOM", "Synthetic Home FC",
                  (("DFL-OBJ-SYNH03", 4, "Hal", "Third", "H. Third"),
                   ("DFL-OBJ-SYNH01", 1, "Hanna", "Keeper", "H. Keeper"),
                   ("DFL-OBJ-SYNH02", 10, None, None, None)))  # no name at all: the PersonId stands in
SYNTHETIC_GUEST = ("DFL-CLU-SYNAWY", "Synthetic Away FC",
                   (("DFL-OBJ-SYNA02", 23, None, None, "B. Short"),  # only a short name: that is used
                    ("DFL-OBJ-SYNA03", 9, "Zoë Maria", "Müller", "Z. Müller"),  # non-ASCII, several given names
                    ("DFL-OBJ-SYNA01", 1, "Ann", "Goalie", "A. Goalie")))
# Positions of the synthetic match. The first half is N 10000-10009 and the second N 100000-100007, so the timeline
# has 10 + 25 + 8 = 43 rows and the second half starts on row 35. A coordinate is given as (jx, jy), a place on a grid
# with 8 intervals across and down the 104.5 x 67.5 m pitch: its normalised value is exactly jx / 8 and jy / 8 (x runs
# with X, y against Y), so a check can compare with == and no rounding hides an off-by-one or a missing flip. A None
# leaves that attribute out of the file. The tuples are in file order.
PERSON_A, PERSON_B, PERSON_C = "DFL-OBJ-SYNH01", "DFL-OBJ-SYNH02", "DFL-OBJ-SYNA01"
PERSON_REFEREE, PERSON_UNLISTED, PERSON_BALL = "DFL-OBJ-SYNR01", "DFL-OBJ-SYNZ01", "DFL-OBJ-SYNBAL"
SYNTHETIC_TIME = "2024-02-28T08:15:28.955+00:00"  # every frame says the same: the parser must not read T
# {(PersonId, GameSection): ((N, jx, jy, S in km/h as text), ...)}, in the order the file lists the FrameSets.
SYNTHETIC_PLAYER_FRAMES = {
    # A plays both halves, and her second half comes first in the file. Her first half has a frame before the half
    # (N 9999) and one after it (N 10010, which also lacks S): both are dropped, and neither may land on row -1 or on
    # a separator row.
    (PERSON_A, "secondHalf"): ((100000, 1, 1, "3.6"), (100001, 2, 3, "4.5"), (100002, 3, 5, None),
                               (100003, 4, 7, "6.3"), (100004, 5, 8, "7.2"), (100005, 6, 6, "8.1"),
                               (100006, 7, 4, "9.0"), (100007, 8, 2, "9.9")),
    (PERSON_A, "firstHalf"): ((9999, 0, 0, "1.0"),
                              (10000, 0, 2, "10.8"), (10001, 1, 3, "11.7"), (10002, 2, 2, "12.6"),
                              (10003, 3, 1, "13.5"), (10004, 4, 0, "14.4"), (10005, 5, 6, "15.3"),
                              (10006, None, 7, "16.2"), (10007, 7, 8, "17.1"), (10008, 8, 5, "18.0"),
                              (10009, 8, 1, "18.9"), (10010, 3, 3, None)),
    # B plays the first half only; his frames N 10004 and N 10003 are written in that order.
    (PERSON_B, "firstHalf"): ((10000, 8, 6, "5.4"), (10001, 7, 5, "6.3"), (10002, 6, 4, "7.2"),
                              (10004, 5, None, "9.0"), (10003, 5, 3, "8.1"), (10005, 4, 2, "9.9"),
                              (10006, 3, 1, "10.8"), (10007, 2, 0, None), (10008, 1, 7, "12.6"),
                              (10009, 0, 8, "13.5")),
    # C, of the away team, comes on at N 100003 (row 38).
    (PERSON_C, "secondHalf"): ((100003, 4, 4, "21.6"), (100004, 5, 5, "22.5"), (100005, None, 6, "23.4"),
                               (100006, 7, 7, "24.3"), (100007, 8, 8, "25.2")),
}
# {GameSection: ((N, jx, jy, Z in metres, S, BallPossession, BallStatus), ...)}; the text is what the file says.
# N 100004 has no ball frame, N 10007 is a dead ball that was last the away team's, N 100007 one that was the home
# team's, and each code is missing on a frame of its own.
SYNTHETIC_BALL_FRAMES = {
    "firstHalf": ((10000, 4, 4, "0.00", "7.2", "1", "1"), (10001, 5, 4, "0.25", "8.1", "1", "1"),
                  (10002, 6, 3, "0.50", "9.0", "1", "1"), (10003, 6, 2, "0.75", None, "1", "1"),
                  (10004, 5, 1, "1.00", "10.8", "1", "1"), (10005, 4, 1, "1.25", "11.7", "2", "1"),
                  (10006, 3, 2, "1.50", "12.6", "2", "1"), (10007, 3, 3, "1.75", "13.5", "2", "0"),
                  (10008, 2, 4, None, "14.4", "2", "1"), (10009, 2, 5, "2.25", "15.3", "2", None)),
    "secondHalf": ((100000, 1, 5, "0.00", "5.4", "2", "1"), (100001, 1, 6, None, "6.3", "2", "1"),
                   (100002, 2, 7, "0.50", "7.2", "2", "1"), (100003, 3, 7, "0.75", "8.1", "1", "1"),
                   (100005, 5, 6, "1.25", "9.9", "1", None), (100006, 6, 5, "1.50", "10.8", None, "1"),
                   (100007, 7, 4, "1.75", "11.7", "1", "0")),
}
# What parse_positions must report on the synthetic match: 17 ball frames, 33 stored player frames (A 10 + 8, B 10,
# C 5) and 2 dropped; X is missing on 2 stored frames, Y on 1, S on 3 (the dropped frame's missing S is not counted).
SYNTHETIC_REPORT = {
    "ball_frames": 17, "rows_without_ball": 1, "player_frames": 33, "player_frames_outside_halves": 2,
    "skipped_framesets": {"DFL-CLU-SYNAWY": 1, "referee": 1},
    "missing": {"X": 2, "Y": 1, "S": 3, "Z": 2, "BallPossession": 1, "BallStatus": 2}}


# -- the harness's contract with the checks ---------------------------------------------------------------------------

class NotRun(Exception):
    """Raised by a check that could not run on this machine, as opposed to one that ran and failed.

    The harness prints "NOT RUN  <name>: <reason>", counts it apart from the passes and does not fail the run for it:
    an optional comparison (kloppy) that is not installed has compared nothing, which a PASS would claim it had.
    It is defined here, and not in selftest.py, because the checks raise it and the harness catches it: when the
    harness runs by path as __main__, a check importing it from selftest.py would load that file a second time as a
    different module, and the harness would catch a different class.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# -- console ----------------------------------------------------------------------------------------------------------

def utf8_stdout():
    """Make stdout write UTF-8, whatever the console or the pipe thinks its code page is (team names have umlauts)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)  # a replaced stdout may not have it
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


# -- a random DflMatch ------------------------------------------------------------------------------------------------

def random_match(match_id="RND001", seed=0, rows=(240, 200), gap=25, teams=(0, 0, 0, 1, 1, 1), pitch=(105.0, 68.0),
                 fps=25, writeable=False):
    """A DflMatch of random positions and flags: the halves are rows[0] and rows[1] long, `gap` rows apart.

    Positions wander a little outside the pitch and about a quarter of the players, and a fifth of the balls, are
    missing, as they are in the real data; nothing is a tie. The gap rows are what DFL's separator is: no ball, no
    players, codes -1. The arrays are read-only, as dfl.load_match gives them, unless `writeable`.
    """
    rng = np.random.default_rng([9, seed])
    n, players = sum(rows) + gap, len(teams)
    graded = ((1, 0, rows[0]), (2, rows[0] + gap, n))
    inside = np.zeros(n, dtype=bool)
    for _, start, stop in graded:
        inside[start:stop] = True

    xy = rng.uniform(-0.02, 1.02, size=(n, players, 2))
    xy[rng.uniform(size=(n, players)) < 0.25] = np.nan
    ball = rng.uniform(-0.02, 1.02, size=(n, 2))
    ball[rng.uniform(size=n) < 0.2] = np.nan
    arrays = {
        "frame_number": np.where(inside, np.arange(n), -1).astype(np.int64),
        "players_xy": xy, "players_speed": rng.uniform(0, 9, size=(n, players)),
        "player_team": np.array(teams, dtype=np.int8), "ball_xy": ball, "ball_z": rng.uniform(0, 3, size=n),
        "ball_speed": rng.uniform(0, 30, size=n),
        "ball_status": rng.integers(0, 2, size=n).astype(np.int8),
        "ball_possession": rng.integers(0, 2, size=n).astype(np.int8)}
    for name in ("players_xy", "players_speed", "ball_xy", "ball_z", "ball_speed"):
        arrays[name][~inside] = np.nan
    for name in ("ball_status", "ball_possession"):
        arrays[name][~inside] = -1
    for array in arrays.values():
        array.setflags(write=writeable)
    return dfl.DflMatch(match_id=match_id, pitch=pitch, fps=fps, graded=graded,
                        kickoffs={half: start for half, start, _ in graded}, report={}, **arrays)


def whole_positions(match):
    """`match` with a position missing in both coordinates wherever it was missing in one.

    The synthetic DFL match keeps a lone X or a lone Y on a few frames, to test the parser; an Atd may not (clean_atd
    refuses it), so a check that serves that match through MatchInputs serves this one.
    """
    xy, ball = np.array(match.players_xy), np.array(match.ball_xy)
    xy[np.isnan(xy).any(axis=-1)] = np.nan
    ball[np.isnan(ball).any(axis=-1)] = np.nan
    xy.setflags(write=False)
    ball.setflags(write=False)
    return dataclasses.replace(match, players_xy=xy, ball_xy=ball)


# -- a synthetic tracking: the degrader's and the fidelity report's fixtures -----------------------------------------

# Params
ROWS = (240, 200)                       # the synthetic halves, in rows
MASK_ROWS = (330, 310)                  # our made-up masks: longer than the halves, so that an offset has a range
# What the made-up Calibration says of our tracking: tracklets of 10, 25, 50, 80 and 100 frames at 25 Hz (mean 53), and
# slots present per frame as {slots: frames} per half, each half as long as its made-up mask (MASK_ROWS), since a
# Calibration's counts are the frames of its masks. The frames with more than 18 present (CAP) are 290 with 22, 174
# with 20 and 116 with 19 over the two halves: their mean is (6380 + 3480 + 2204) / 580 = 20.8, so d = 1 - 20.8 / 22 =
# 3 / 55, and a lost span has the mean mu = 53 * d / (1 - d) = 53 * 3 / 52 frames. The 60 frames with 18 or fewer
# (counts 18, 10 and 7) are there to be left out of d, and they are the tail that the cap reads. The lowest count is 7,
# more than the 6 players of the steady matches, so that the cap changes nothing in them and the checks of the walk, the
# slots and the episodes see the same players as without a tail; the cap's own checks use their own counts, and the
# full squads of the fidelity checks (22 players) are capped by these. They come in runs (counts_runs), as ours do.
LENGTHS_S = (0.4, 1.0, 2.0, 3.2, 4.0)
PRESENT_COUNTS = {1: {22: 145, 20: 87, 19: 58, 18: 20, 10: 12, 7: 8}, 2: {22: 145, 20: 87, 19: 58, 18: 10, 10: 6, 7: 4}}
RUN_FRAMES = 15                         # the made-up counts come in runs of at most this many frames
MADE_UP_D, MADE_UP_MEAN, MADE_UP_MU = 3 / 55, 53.0, 53 * 3 / 52
# And of the wrong-team episodes and the NaN-speed runs: episodes of 0.4, 1.2, 2 and 4 s (10, 30, 50 and 100 frames at
# 25 Hz, mean 47.5), 10% of the frames wrong-team, and NaN-speed runs of 3, 8, 21 and 40 frames (mean 18), 8% of the
# present slot-frames.
EPISODES_S = (0.4, 1.2, 2.0, 4.0)
NAN_RUNS = (3, 8, 21, 40)
MADE_UP_WRONG_TEAM, MADE_UP_SPEED_NAN = 0.1, 0.08


def tracked_match(match_id="RND001", seed=0, rows=ROWS):
    """random_match with the ball on every frame of both halves (and none on the separator), x and y together."""
    match = random_match(match_id, seed=seed, rows=rows)
    ball = np.array(match.ball_xy)
    for _, start, stop in match.graded:
        ball[start:stop] = np.random.default_rng([4, seed]).uniform(0.05, 0.95, size=(stop - start, 2))
    ball.setflags(write=False)
    return dataclasses.replace(match, ball_xy=ball)


def steady_match(match_id="RND001", seed=0, rows=ROWS, teams=(0, 0, 0, 1, 1, 1), substitutions=True):
    """tracked_match whose players are on the pitch for whole stretches, as in a real match, and not at random.

    Everybody is on for the whole of both halves, except (with `substitutions`, in the first half, rows counted from
    its start): player 2 (home) comes on at 60, player 4 (away) is off from 150, and player 5 (away) is off from 100 up
    to 130 and back after it, so that he has two stretches. On the default teams and ROWS that makes 7 stretches in the
    first half (rows 0-239) and 6 in the second (rows 265-464). Positions are uniform over the pitch, all different, and
    NaN in both coordinates where a player is not on and on the rows between the halves.
    """
    match = tracked_match(match_id, seed, rows)
    xy = np.random.default_rng([5, seed]).uniform(0.02, 0.98, size=(match.n_frames, len(teams), 2))
    outside = np.ones(match.n_frames, dtype=bool)
    for _, start, stop in match.graded:
        outside[start:stop] = False
    xy[outside] = np.nan
    if substitutions:
        start, stop = match.graded[0][1:]
        xy[start:start + 60, 2] = np.nan
        xy[start + 150:stop, 4] = np.nan
        xy[start + 100:start + 130, 5] = np.nan
    xy.setflags(write=False)
    return dataclasses.replace(match, players_xy=xy, player_team=np.array(teams, dtype=np.int8))


def ball_bytes(atd):
    """The bytes of a degraded Atd's ball (position and speed): equal bytes are the same ball, to the last bit."""
    return atd.ball_xy.tobytes() + atd.ball_speed.tobytes()


def counts_runs(table, rng):
    """The frames of {slots: frames} in runs of at most RUN_FRAMES, the runs in an order drawn from `rng`: int16,
    read-only.

    Our counts are low for a stretch and not frame by frame, so the tail comes as runs; the multiset of counts is the
    table's whatever the order, which is all that d reads."""
    runs = []
    for slots, frames in sorted(table.items()):
        while frames > 0:
            runs.append(np.full(min(RUN_FRAMES, frames), slots, dtype=np.int16))
            frames -= RUN_FRAMES
    counts = np.concatenate([runs[index] for index in rng.permutation(len(runs))])
    counts.setflags(write=False)
    return counts


def made_up_calibration(seed=0, rows=MASK_ROWS):
    """What the degrader reads of a Calibration: `ball_masks`, with long runs of missing ball as ours has, and what the
    tracking profile reads: `tracklet_lengths_s` and `present_counts` (LENGTHS_S and PRESENT_COUNTS: as long as the
    masks, in runs), `wrong_team_episodes_s` and `wrong_team_share`, `speed_nan_runs` and `speed_nan_share` (EPISODES_S,
    NAN_RUNS and the two shares above). ValueError if a half of PRESENT_COUNTS is not as long as its mask (`rows`)."""
    rng = np.random.default_rng([7, seed])
    masks = {}
    for half, length in enumerate(rows, start=1):
        runs, first = [], bool(rng.integers(2))
        while sum(runs) < length:
            runs.append(int(rng.integers(5, 40)))
        flags = np.concatenate([np.full(count, first ^ (k % 2 == 1)) for k, count in enumerate(runs)])[:length]
        flags.setflags(write=False)
        masks[half] = flags
    counts = {}
    counts_rng = np.random.default_rng([7, seed, 1])
    for half, table in PRESENT_COUNTS.items():
        if sum(table.values()) != rows[half - 1]:
            raise ValueError("PRESENT_COUNTS of half {} has {} frames and its mask {}: a Calibration's counts are the "
                             "frames of its masks".format(half, sum(table.values()), rows[half - 1]))
        counts[half] = counts_runs(table, counts_rng)
    lengths, episodes, runs = np.array(LENGTHS_S), np.array(EPISODES_S), np.array(NAN_RUNS, dtype=np.int64)
    for array in (lengths, episodes, runs):
        array.setflags(write=False)
    return types.SimpleNamespace(ball_masks=masks, present_counts=counts, tracklet_lengths_s=lengths,
                                 wrong_team_episodes_s=episodes, wrong_team_share=MADE_UP_WRONG_TEAM,
                                 speed_nan_runs=runs, speed_nan_share=MADE_UP_SPEED_NAN)


# -- the synthetic DFL match ------------------------------------------------------------------------------------------

def xml_attributes(values):
    """name="value" for every pair whose value is not None, escaped, in the order given."""
    return " ".join("{}={}".format(name, quoteattr(str(value))) for name, value in values.items() if value is not None)


def synthetic_team(role, team_id, name, squad):
    """One <Team> as DFL writes it: the real attributes and children, with a trainer who is not a player."""
    players = "\n".join(
        "          <Player {} />".format(xml_attributes({
            "PersonId": person_id, "ShirtNumber": shirt, "FirstName": first, "LastName": last, "Shortname": short,
            "Starting": "true", "TeamLeader": "false"}))
        for person_id, shirt, first, last, short in squad)
    return """      <Team {}>
        <Players>
{}
        </Players>
        <TrainerStaff>
          <Trainer PersonId="DFL-OBJ-SYNT{}" Role="headcoach" FirstName="Tia" LastName="Coach" Shortname="T. Coach" />
        </TrainerStaff>
        <OfficialStaff>
        </OfficialStaff>
      </Team>""".format(xml_attributes({
        "TeamId": team_id, "TeamName": name, "Role": role, "PlayerShirtType": "Heimtrikot",
        "PlayerShirtMainColor": "#F4F5F0", "LineUp": "4-2-3-1"}), players, role[0].upper() + "00")


def grid_x(j):
    """X in metres, as text, that normalises to exactly j / 8 on the synthetic 104.5 m pitch (None stays None)."""
    return None if j is None else repr((j - 4) * 13.0625)


def grid_y(j):
    """Y in metres, as text, that normalises to exactly j / 8 on the synthetic 67.5 m pitch: Y runs against y."""
    return None if j is None else repr(33.75 - 8.4375 * j)


def synthetic_frameset(section, team_id, person_id, frames, match_id):
    """One <FrameSet> as DFL writes it, one <Frame> per line; `frames` are the rendered <Frame> lines."""
    return "<FrameSet {}>\n{}\n</FrameSet>".format(xml_attributes({
        "GameSection": section, "MatchId": "DFL-MAT-" + match_id, "TeamId": team_id, "PersonId": person_id}),
        "\n".join(frames))


def player_frame(n, jx, jy, speed):
    return "<Frame {} />".format(xml_attributes({
        "N": n, "T": SYNTHETIC_TIME, "X": grid_x(jx), "Y": grid_y(jy), "D": "0.00", "S": speed, "A": "0.00",
        "M": "1"}))


def ball_frame(n, jx, jy, z, speed, possession, status):
    return "<Frame {} />".format(xml_attributes({
        "N": n, "T": SYNTHETIC_TIME, "X": grid_x(jx), "Y": grid_y(jy), "Z": z, "D": "0.00", "S": speed, "A": "0.00",
        "M": "1", "BallPossession": possession, "BallStatus": status}))


def synthetic_positions_text(match_id):
    """The positions file of the synthetic match, with the layout that the parser must not depend on.

    The referee and a player who is not in the squad list come first (the real file starts with the referees), then
    the players, one of them with his second half before his first, one with two frames out of order, and the ball
    last, as in the real file. Every FrameSet carries the match id, as the real ones do.
    """
    home, guest = SYNTHETIC_HOME[0], SYNTHETIC_GUEST[0]
    club_of = {PERSON_A: home, PERSON_B: home, PERSON_C: guest}
    sets = [
        synthetic_frameset("firstHalf", "referee", PERSON_REFEREE,
                           [player_frame(10000 + k, 4, 4, "2.0") for k in range(3)], match_id),
        synthetic_frameset("firstHalf", guest, PERSON_UNLISTED,
                           [player_frame(10000 + k, 1, 1, "2.0") for k in range(3)], match_id)]
    sets += [synthetic_frameset(section, club_of[person], person, [player_frame(*frame) for frame in frames], match_id)
             for (person, section), frames in SYNTHETIC_PLAYER_FRAMES.items()]
    sets += [synthetic_frameset(section, "BALL", PERSON_BALL, [ball_frame(*frame) for frame in frames], match_id)
             for section, frames in SYNTHETIC_BALL_FRAMES.items()]
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest RequestId="00000000-0000-4000-8000-000000000001" MessageTime="{time}" TransmissionComplete="true" \
TransmissionSuspended="false">
<Positions EventTime="{time}">
<MetaData MatchId="DFL-MAT-{match_id}" Type="pitch-size">
<PitchSize X="104.50" Y="67.50" />
</MetaData>
{sets}
</Positions>
</PutDataRequest>
""".format(time=SYNTHETIC_TIME, match_id=match_id, sets="\n".join(sets))


def write_synthetic_dfl(folder, match_id="SYN001"):
    """Write a small DFL match in the real file format into `folder`; return {kind: path}, kinds as in sources.FILES.

    The match information and the positions (not the events, which no parser reads yet). The element and attribute
    names are the real ones (copied from J03WMX's files), the values are made up and chosen so that every parser rule
    shows. Information: the guest team is listed first, so Role decides and not the order; each squad is out of
    PersonId order; names exercise the fallbacks and non-ASCII text; a trainer and a referee carry PersonIds that
    are not players; the pitch is not the usual 105 x 68; the two half durations differ. Positions: see
    synthetic_positions_text and the SYNTHETIC_* tables.
    """
    home_id, home_name, home_squad = SYNTHETIC_HOME
    guest_id, guest_name, guest_squad = SYNTHETIC_GUEST
    text = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest RequestId="00000000-0000-4000-8000-000000000000" MessageTime="2024-02-28T08:15:28.955+00:00"
    TransmissionComplete="true" TransmissionSuspended="false">
  <MatchInformation>
    <General {general} />
    <Environment Country="Deutschland" StadiumName="Synthetic Arena" NeutralVenue="false" Roof="open"
        PitchX="104.50" PitchY="67.50" />
    <Teams>
{guest}
{home}
    </Teams>
    <Referees>
      <Referee PersonId="DFL-OBJ-SYNR01" Role="referee" FirstName="Rob" LastName="Whistle" Shortname="R. Whistle" />
    </Referees>
    <OtherGameInformation TotalTimeFirstHalf="2700040" TotalTimeSecondHalf="2820120"
        PlayingTimeFirstHalf="1888960" PlayingTimeSecondHalf="1525720" />
  </MatchInformation>
</PutDataRequest>
""".format(
        general=xml_attributes({
            "TypeOfSport": "Fußball", "CompetitionName": "Synthetic League", "CompetitionId": "DFL-COM-SYNTH",
            "MatchId": "DFL-MAT-" + match_id, "MatchTitle": "{}:{}".format(home_name, guest_name),
            "HomeTeamName": home_name, "HomeTeamId": home_id, "GuestTeamName": guest_name, "GuestTeamId": guest_id}),
        guest=synthetic_team("guest", guest_id, guest_name, guest_squad),
        home=synthetic_team("home", home_id, home_name, home_squad))
    name = "DFL_02_01_matchinformation_DFL-COM-SYNTH_DFL-MAT-{}.xml".format(match_id)
    positions = "DFL_04_03_positions_raw_observed_DFL-COM-SYNTH_DFL-MAT-{}.xml".format(match_id)
    return {"information": write_text(folder, name, text),
            "positions": write_text(folder, positions, synthetic_positions_text(match_id))}


def damaged(path, damage):
    """Rewrite the file at `path` as damage(its text), which must change it; return the path."""
    with open(path, encoding="utf-8", newline="") as handle:
        intact = handle.read()
    text = damage(intact)
    assert text != intact, "the damage left the synthetic file as it was"
    return write_text(os.path.dirname(path), os.path.basename(path), text)


# -- a Calibration without our data ------------------------------------------------------------------------------------

def synthetic_calibration_inputs(root):
    """(inputs, gt): the synthetic bundle with two graded blocks, kick-offs inside them and a made-up public truth."""
    make_synthetic_bundle(root)
    inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
    inputs.kickoffs = {1: 60, 2: 340}
    gt = np.full(inputs.n_frames, X, dtype=np.int8)
    gt[60:100], gt[100:120], gt[120:130], gt[130:150] = H, A, D, H
    gt[340:360], gt[360:370] = A, D
    return inputs, gt


def brute_force_agreement(inputs, gt):
    """(share, frames) of in-play frames, ball visible and somebody tracked, where the nearest slot's team is the GT's.

    One frame at a time, one slot at a time, in plain Python and in the baseline's own order of operations: the
    reference for pipeline.evidence.describe's number, with no vectorised code in common with it. A tie goes to the
    lowest slot, as starter/baseline.py's strict '<' keeps it.
    """
    atd = inputs.atd
    length, width = inputs.pitch
    agree = judged = 0
    for frame in range(inputs.n_frames):
        if gt[frame] not in (H, A) or not atd.ball_present[frame]:
            continue
        best, best_distance = None, None
        for slot in range(atd.n_slots):
            if not atd.players_present[frame, slot]:
                continue
            dx = atd.players_xy[frame, slot, 0] * length - atd.ball_xy[frame, 0] * length
            dy = atd.players_xy[frame, slot, 1] * width - atd.ball_xy[frame, 1] * width
            distance = dx * dx + dy * dy
            if best is None or distance < best_distance:
                best, best_distance = slot, distance
        if best is None:
            continue
        judged += 1
        agree += (atd.slot_team[best] == 0) == (gt[frame] == H)
    return agree / judged, judged


@functools.lru_cache(maxsize=1)
def synthetic_calibration():
    """A real Calibration: calibration.measure on the synthetic bundle of pipeline/selftest.py, measured once.

    Two halves of 220 frames from their kick-offs, a made-up public truth and the four yardsticks run on it; what the
    runner's and the fidelity report's checks need of a Calibration (masks, statistics, public window, baseline
    outputs, a digest) without the data directory. The arrays are in memory: nothing here refers to the folder it was
    measured in. Its numbers mean nothing about any match.
    """
    import tempfile
    from pipeline.benchmark import calibration
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        return calibration.measure(inputs, gt)


def tracking_calibration():
    """A Calibration with what the tracking profile reads: synthetic_calibration with the made-up masks, counts,
    tracklets, episodes and NaN-speed runs of this module (its other fields are the synthetic bundle's).

    The masks come with the counts, since the counts are the frames of the masks and the cap reads both through one
    Transplant; the figures that the bundle measured are not the made-up ones, which is what the checks that use it
    work out for themselves."""
    made = made_up_calibration()
    return dataclasses.replace(
        synthetic_calibration(), ball_masks=made.ball_masks, present_counts=made.present_counts,
        tracklet_lengths_s=made.tracklet_lengths_s, wrong_team_episodes_s=made.wrong_team_episodes_s,
        wrong_team_share=made.wrong_team_share, speed_nan_runs=made.speed_nan_runs,
        speed_nan_share=made.speed_nan_share)


# -- a Calibration with a hazard of our own making ---------------------------------------------------------------------

# What the made-up hazard says: a chance of P_ONSET that the next frame starts a gap, the same on every frame (a model
# with an intercept and coefficients of 0), and gaps of the durations below, in seconds, by onset class (0: far from a
# line, 1: near one) and third of the half. At 25 Hz a class-0 gap lasts 10, 20 or 30 frames and a class-1 gap 5 or 15.
P_ONSET = 0.02
HAZARD_SECONDS = {0: {third: (0.4, 0.8, 1.2) for third in range(3)}, 1: {third: (0.2, 0.6) for third in range(3)}}


def made_up_hazard(p=P_ONSET, c=1.0, seconds=None, coef=None):
    """The three fields of a Calibration that the context profile reads, made up: {hazard_model, hazard_c,
    hazard_durations_s}.

    The model gives every frame the probability `p` (intercept logit(p), coefficients 0 unless `coef` gives them, six
    numbers on the standardised features; mean 0 and scale 1, so a feature is itself), `c` is hazard_c and `seconds` is
    {class: {third: lengths in seconds}} (HAZARD_SECONDS). Every array is read-only, as a Calibration's are."""
    coefficients = np.zeros(len(hazard.FEATURES)) if coef is None else np.array(coef, dtype=np.float64)
    arrays = [coefficients, np.zeros(len(hazard.FEATURES)), np.ones(len(hazard.FEATURES))]
    for array in arrays:
        array.setflags(write=False)
    table = {}
    for onset_class, thirds in (HAZARD_SECONDS if seconds is None else seconds).items():
        table[onset_class] = {}
        for third, lengths in thirds.items():
            table[onset_class][third] = np.array(lengths, dtype=np.float64)
            table[onset_class][third].setflags(write=False)
    model = hazard.LogisticModel(coef=arrays[0], intercept=float(np.log(p / (1.0 - p))), mean=arrays[1],
                                 scale=arrays[2])
    return {"hazard_model": model, "hazard_c": c, "hazard_durations_s": table}


def context_namespace(**more):
    """made_up_calibration with a made-up hazard (made_up_hazard(**more)): what degrade.degrade reads for context."""
    return types.SimpleNamespace(**{**vars(made_up_calibration()), **made_up_hazard(**more)})


def context_calibration(**more):
    """A real Calibration with a made-up hazard: tracking_calibration, its hazard fields replaced by made_up_hazard's.

    The runner and the fidelity report need a real one (a digest, a public window, baseline outputs): the bundle's own
    hazard is fitted on a few hundred frames and means nothing, so the context profile's checks put theirs in. It is a
    go (hazard_go True): the profile refuses a Calibration whose hazard is not, and the bundle's own fit may not be."""
    return dataclasses.replace(tracking_calibration(), hazard_go=True, **made_up_hazard(**more))


# -- what the tracking profile's checks share with the cap's ----------------------------------------------------------

def without_new_parts(calibration, **more):
    """The Calibration with no wrong-team share and no NaN-speed share: its players are tracklets and slots alone."""
    return types.SimpleNamespace(**{**vars(calibration), "wrong_team_share": 0.0, "speed_nan_share": 0.0, **more})


def pairs_match(windows, rows=(1000, 1000), pitch=(128.0, 64.0)):
    """Six players (home 0-2, away 3-5) who stand far apart all through both halves, and come close in windows.

    Home k stands at x = 0.125 (k + 1), y = 0.25 and away k at x = 0.5 + 0.125 k, y = 0.75, so every opposing pair is at
    least 32 m apart. The pitch is 128 x 64 m, so that a distance in metres is a short dyadic number and comes out
    exactly: 1/64 of the pitch's length is 2.0 m, to the bit. `windows` is [(half, first row, stop row, pairs)] with
    the rows counted from the half's start; in a window, away player `away` of each (home, away, offset) pair stands at
    home player `home`'s place plus the normalised x `offset`, so the pair is offset * 128 m apart and no other opposing
    pair is within 16 m. Everybody is on the pitch all through both halves, so with nobody lost a player is one
    tracklet."""
    base = steady_match(rows=rows, substitutions=False)
    xy = np.array(base.players_xy)
    starts = {half: start for half, start, _ in base.graded}
    for _, start, stop in base.graded:
        xy[start:stop, 0:3, 0], xy[start:stop, 0:3, 1] = [0.125, 0.25, 0.375], 0.25
        xy[start:stop, 3:6, 0], xy[start:stop, 3:6, 1] = [0.5, 0.625, 0.75], 0.75
    for half, first, last, pairs in windows:
        lo, hi = starts[half] + first, starts[half] + last
        for home, away, offset in pairs:
            xy[lo:hi, away, 0], xy[lo:hi, away, 1] = xy[lo:hi, home, 0] + offset, xy[lo:hi, home, 1]
    xy.setflags(write=False)
    return dataclasses.replace(base, players_xy=xy, pitch=pitch)


def episode_parameters(share=0.45, scale=1.0, episodes=(10, 20), lengths=(10 ** 9,), d=0.0, **more):
    """Parameters with episodes of 10 and 20 frames (mean 15) at the rate scale * share / 15 = 0.03 per row, and nobody
    lost unless `lengths` and `d` say so."""
    return degrade.tracking_parameters(list(lengths), d, episodes=list(episodes), wrong_team_share=share, scale=scale,
                                       **more)


def is_carried(t, view):
    """A tracklet is an episode piece if its slot is in the pool of the team that its player is not in."""
    return (t.slot >= degrade.HOME_SLOTS) != (int(view.player_team[t.player]) == 1)


def reference_episodes(view, half, spans, parameters, seed):
    """(pieces, counts) of the wrong-team episodes as degrade.py documents them, written out one onset at a time.

    The draws are read from a generator made from the documented key, in the documented order (the onset rows as one
    uniform per row of the half's block, then one pick per onset, then one duration per onset), and every onset is
    judged by loops over players and pieces. pieces is [(first, player, last, pool)] sorted, pool being the team whose
    slots carry the piece."""
    start, stop = next((first, last) for number, first, last in view.graded if number == half)
    team, xy = list(view.player_team), view.players_xy
    length, width = view.pitch
    rate = parameters.scale * parameters.wrong_team_share / parameters.mean_episode
    rng = np.random.default_rng([seed, view.match_index, 2, half])
    uniform = rng.random(stop - start)
    onsets = [start + index for index in range(stop - start) if uniform[index] < rate]
    picks = rng.integers(2, size=len(onsets))
    durations = rng.integers(len(parameters.episodes), size=len(onsets))
    own = {}
    for first, player, last in spans:
        own.setdefault(player, []).append([first, last, False])
    counts = {"onsets": len(onsets), "episodes": 0, "skipped_far": 0, "skipped_overlap": 0, "episode_frames": 0}
    for k, t in enumerate(onsets):
        tracked = [player for player in range(len(team)) if any(f <= t < l for f, p, l in spans if p == player)]
        best = None
        for home in (p for p in tracked if team[p] == 0):
            for away in (p for p in tracked if team[p] == 1):
                distance = np.hypot((xy[t, home, 0] - xy[t, away, 0]) * length,
                                    (xy[t, home, 1] - xy[t, away, 1]) * width)
                if best is None or distance < best[0]:
                    best = (distance, home, away)
        if best is None or best[0] > 2.0:
            counts["skipped_far"] += 1
            continue
        player = (best[1], best[2])[int(picks[k])]
        index = next(i for i, (a, b, _) in enumerate(own[player]) if a <= t < b)
        a, b, inside = own[player][index]
        if inside:
            counts["skipped_overlap"] += 1
            continue
        end = min(t + int(parameters.episodes[int(durations[k])]), b)
        own[player][index:index + 1] = ([[a, t, False]] if t > a else []) + [[t, end, True]] + (
            [[end, b, False]] if end < b else [])
        counts["episodes"] += 1
        counts["episode_frames"] += end - t
    pieces = sorted((a, player, b, 1 - team[player] if inside else team[player])
                    for player, parts in own.items() for a, b, inside in parts)
    return pieces, counts


def smoothed_plain(run, window):
    """The centred moving average of `window` rows of a list of (x, y), the window shrinking to the rows inside."""
    half, out = window // 2, []
    for index in range(len(run)):
        rows = run[max(0, index - half):index + half + 1]
        out.append((sum(x for x, _ in rows) / len(rows), sum(y for _, y in rows) / len(rows)))
    return np.array(out)


# The measurements that TRACKED_SCALE was worked out from (docs/results.md): ours' tracklet median (the Calibration's,
# A09) and the pooled tracklet median, in seconds, of the tracking profile at lambda = 1 with the shortest-piece rule
# in, seed 0, over the 7 DFL matches laid end to end. With version 2's slot rule (a tracklet could take a slot on the
# row it was freed) it was 20.22 s, so lambda was 2.042 (R004b, 2026-10-23); with version 3's (a slot is free from the
# row after its tracklet stopped) it is 19.92 s, so lambda is 2.072 (R004c, 2026-10-23), which is the constant in force.
RECORDED_OURS_MEDIAN_S = 41.28
RECORDED_MEDIAN_V2_S = 20.22
RECORDED_MEDIAN_S = 19.92


def tracked_scale_from_the_record():
    """lambda as it was worked out once, with the slot rule of version 3: ours' tracklet median over the profile's at
    lambda = 1, to 3 decimals."""
    return round(RECORDED_OURS_MEDIAN_S / RECORDED_MEDIAN_S, 3)


# -- what the runner's and the fidelity report's checks share ---------------------------------------------------------

def yardstick_path(width):
    return os.path.join(CONFIG_FOLDER, "nearest_w{}.toml".format(width))


def synthetic_matches():
    """Two matches of random flags and tracking, of different lengths: the ids are not DFL's, so there is no cache."""
    return {"RND001": random_match("RND001"), "RND002": random_match("RND002", seed=1, rows=(300, 260))}


# -- a config that fills the ball, and the pools its simulated broadcast is made from (R028) ---------------------------

# A config whose ball is filled from the broadcast's ([inputs] ball_fill = "broadcast"): on a benchmark match it reads
# the simulated broadcast ball (broadcast_sim.served_inputs_for). FILL_TWIN is the same config without the fill, the
# driver whose labels class the gaps; its description differs, which nothing that a decode reads does.
FILL_CONFIG = """
description = "nearest_player's team and a majority vote, on our ball filled from the broadcast's"
seed = 0
scope = "halves"

[inputs]
ball_fill = "broadcast"

[[evidence]]
kind = "nearest_player"

[decoder]
kind = "majority_vote"
half_width = 3
"""
FILL_TWIN = FILL_CONFIG.replace('[inputs]\nball_fill = "broadcast"\n\n', "").replace(
    "on our ball filled from the broadcast's", "on our ball as it is")


def made_up_pools():
    """A broadcast_sim.Pools of made-up donors and errors, enough for any synthetic match.

    Donor gaps of 1 to 144 frames in both classes: the alive ones filled at their two edges (as ours mostly are), the D
    ones in stripes of 3 frames (so that a transplant gives fragments shorter than 13 frames, which DFL serves as they
    are). Three error runs, of 5, 40 and 300 frames, whose offsets say which step they are.
    """
    from pipeline.benchmark import broadcast_sim
    masks, alive = [], []
    for length in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144):
        masks.append([i < max(1, length // 4) or i >= length - max(1, length // 6) for i in range(length)])
        alive.append(True)
        masks.append([(i // 3) % 2 == 0 for i in range(length)])
        alive.append(False)
    runs = [[(0.5 + 0.01 * t, -0.25 - 0.02 * t) for t in range(length)] for length in (5, 40, 300)]
    return broadcast_sim.Pools.from_lists(masks, alive, runs)


@contextlib.contextmanager
def loading(serve):
    """dfl.load_match answered by serve(match_id), for the runner's own path to a match; restored afterwards."""
    real = dfl.load_match

    def served(match_id, data_dir=None, rebuild=False, paths=None):
        return serve(match_id)
    dfl.load_match = served
    try:
        yield
    finally:
        dfl.load_match = real


def read(path, mode="r"):
    """The contents of a file, closed again: text as UTF-8 by default, bytes with mode "rb"."""
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as handle:
        return handle.read()


def strict_json(text):
    """The JSON of `text`, refusing NaN and infinity, which JSON has no way to write."""
    def refuse(name):
        raise AssertionError("the report holds {}, which is not JSON".format(name))
    return json.loads(text, parse_constant=refuse)


# -- the leak detector: what the inputs of a component may not hold ----------------------------------------------------

def truth_forms(match):
    """Every array a component could rebuild the answer from, or is the answer: the flags and the codes made of them.

    Each of them whole, and each of its graded halves (a copy of a half's flags is as good a leak as all of them).
    """
    status, possession = np.asarray(match.ball_status), np.asarray(match.ball_possession)
    codes, _ = truth.truth_codes(match)
    whole = {"ball_status": status, "ball_possession": possession, "alive": status == 1, "dead": status == 0,
             "home": possession == HOME, "away": possession == AWAY, "codes": codes}
    forms = dict(whole)
    for half, start, stop in match.graded:
        forms.update({"{}[half {}]".format(name, half): array[start:stop] for name, array in whole.items()})
    return forms


def held_by(value):
    """[(how it is reached, what is reached)] for what `value` holds, one level down; [] for what holds nothing.

    A container's items (dict, mapping proxy, list, tuple, deque, set), a numpy array's base (a view is the array it
    looks into) and, for an array of objects, its elements; and what a function, a bound method and a
    functools.partial close over or are given. Then, for anything that is not a plain value, its attributes: the
    __dict__ and the __slots__ (those of a class that declares them, and of a dataclass with slots=True; a plain
    dataclass keeps its fields in __dict__). Classes, modules and plain values hold nothing. The members of a set are
    all named "{member}", with no position: their order is the hash's, and an object's hash is its id, so leaks() takes
    its routes in the order of their names and never in the order that this returns them.
    """
    if isinstance(value, (str, bytes, int, float, complex, range, type, types.ModuleType, np.generic, np.dtype)):
        return []
    held = []
    if isinstance(value, (dict, types.MappingProxyType)):
        held += [("[{!r}]".format(key), item) for key, item in value.items()]
    elif isinstance(value, (set, frozenset)):
        held += [("{member}", item) for item in value]
    elif isinstance(value, (list, tuple, collections.deque)):
        held += [("[{}]".format(number), item) for number, item in enumerate(value)]
    elif isinstance(value, np.ndarray):
        held += [(".base", value.base)] if isinstance(value.base, np.ndarray) else []
        held += ([("[{}]".format(number), item) for number, item in enumerate(value.flat)]
                 if value.dtype == object else [])
    elif isinstance(value, functools.partial):
        held += [(".func", value.func)] + [(".args[{}]".format(number), item) for number, item in enumerate(value.args)]
        held += [(".keywords[{!r}]".format(key), item) for key, item in (value.keywords or {}).items()]
    elif isinstance(value, types.MethodType):
        held += [(".__self__", value.__self__), (".__func__", value.__func__)]
    elif isinstance(value, types.FunctionType):
        held += [(".defaults[{}]".format(number), item) for number, item in enumerate(value.__defaults__ or ())]
        held += [(".kwdefaults[{!r}]".format(key), item) for key, item in (value.__kwdefaults__ or {}).items()]
        for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
            try:
                held.append((".closure[{!r}]".format(name), cell.cell_contents))
            except ValueError:                                       # a cell that was never filled
                pass
    held += [(".{}".format(name), item) for name, item in getattr(value, "__dict__", {}).items()]
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        for name in ((slots,) if isinstance(slots, str) else slots):
            if hasattr(value, name):
                held.append((".{}".format(name), getattr(value, name)))
    return held


def as_numbers(items):
    """`items` (a list or tuple) as an int array if it is made of ints and bools only (a tolist() copy), else None."""
    if all(isinstance(item, (bool, int, np.bool_, np.integer)) for item in items):
        return np.array(items)
    return None


def leaks(inputs, match):
    """Where `inputs` holds the match or an array equal to its flags or its truth, however deep: [what, ...], sorted.

    Walks everything reachable from `inputs` (held_by), once each: a visited set keeps a cycle from running for ever.
    A DflMatch is a finding by itself; an array of another dtype than object, and a list or tuple of ints (a tolist()
    copy) as long as one of the forms, are compared with every form of the truth (truth_forms); a view is found
    through its base: `inputs.window.base == ball_possession`. Findings are sorted and without duplicates.

    The label of a finding depends on the graph and never on the order in which a set happens to be iterated. The
    routes are taken in the order of their names (a heap on the path string: a parent's name is a prefix of its
    children's, so it is always taken first), and an object that two routes reach is named by the one that sorts
    first. The members of a set share one name, "{member}", so they are all taken before anything below them, and
    whichever of them is taken first, what they hold is named the same
    (test_leak_labels_do_not_depend_on_the_order_of_a_set).

    What it does not see, so that nobody takes it for a proof. It is for an accidental leak: a component's author who
    kept the match, or its flags, in a field, a cache, a closure or a view. It does not see:
      * a copy of a stretch that is not a whole array or a whole half;
      * a copy in another representation: bytes or a memoryview, letter codes such as "H" and "A", a column of shape
        (n, 1), a pandas Series, a list or tuple of floats (a float ndarray copy is found, and so is a list of ints
        or bools), a set of numbers;
      * a copy that is not equal to a form (a smoothed one, a rolling window);
      * an attribute computed by a property or a __getattr__ at the moment it is read;
      * a class-level attribute: type(inputs)._last = match, or an object whose class holds the match, because
        held_by looks into an instance's __dict__ and slots, and a class holds nothing for it
        (test_leak_detector_limits_are_as_documented);
      * anything built at run time that the inputs do not hold;
      * the match itself reached by an import or by name. test_components_import_nothing_from_the_benchmark closes the
        plain import in the three component folders, with its own limits: an import reached through another pipeline
        module is not seen, because the scan reads only those folders.
    """
    forms = truth_forms(match)
    lengths = {array.shape[0] for array in forms.values()}
    found, seen, kept = [], set(), []
    order = itertools.count()                 # breaks a tie between equal names, which are never compared by value
    pending = [("inputs", next(order), inputs)]
    while pending:
        path, _, value = heapq.heappop(pending)
        if id(value) in seen:
            continue
        seen.add(id(value))
        kept.append(value)                    # alive until the walk ends, so that no id is reused by a new object
        if isinstance(value, dfl.DflMatch):
            found.append(path)
            continue
        array = value if isinstance(value, np.ndarray) and value.dtype != object else None
        if array is None and isinstance(value, (list, tuple)) and len(value) in lengths:
            array = as_numbers(value)
        if array is not None:
            found += ["{} == {}".format(path, form) for form, other in forms.items()
                      if array.shape == other.shape and np.array_equal(array, other)]
        for how, item in held_by(value):
            heapq.heappush(pending, (path + how, next(order), item))
    return sorted(set(found))


# -- the downloaded DFL files (--dfl) ---------------------------------------------------------------------------------

LOAD_SECONDS = {}  # match id -> how long dfl.load_match took for it (real_match), for the check that prints it


@functools.lru_cache(maxsize=1)
def real_match(match_id):
    """The DflMatch of a downloaded match, through dfl.load_match and so through data/cache/dfl/.

    The first run parses the match (about a minute) and stores it; every later run, here and in any other script,
    reads it back in a second or two. One match is about 150 MB of arrays, so only the last match is kept in memory:
    the checks on J03WMX run back to back (their names sort that way) and share one load. Its report says whether
    that load was "built" or "reused".
    """
    started = time.perf_counter()
    match = dfl.load_match(match_id)
    LOAD_SECONDS[match_id] = time.perf_counter() - started
    return match


@contextlib.contextmanager
def registered(registry, **components):
    """The components in `registry` (a stage's REGISTRY) while the block runs, and the registry as it was after it."""
    before = dict(registry)
    registry.update(components)
    try:
        yield
    finally:
        registry.clear()
        registry.update(before)


@contextlib.contextmanager
def match_only_source(needs_homography=False):
    """A temporary evidence kind, "probe_match_only", whose module says MATCH_ONLY = True; the registry as it was after.

    It has no opinion anywhere, or, with `needs_homography`, asks for the video's homography (which a benchmark match
    does not have) and raises AssetUnavailable, as the Smart Tagging and the whistle would.
    """
    from pipeline import evidence as evidence_stage
    from pipeline.contracts import Evidence, no_opinion

    def compute(run):
        if needs_homography:
            run.inputs.homography
        return Evidence(alive=no_opinion(run.inputs.n_frames), home=no_opinion(run.inputs.n_frames))

    module = types.ModuleType("pipeline.evidence._probe_match_only")
    module.MATCH_ONLY = True
    compute.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        with registered(evidence_stage.REGISTRY, probe_match_only=compute):
            yield
    finally:
        del sys.modules[module.__name__]


PROBE_CONFIG = ('scope = "halves"\n\n[[evidence]]\nkind = "nearest_player"\n\n[[evidence]]\n'
                'kind = "probe_match_only"\nname = "{name}"\n\n[decoder]\nkind = "majority_vote"\nhalf_width = 2\n')
