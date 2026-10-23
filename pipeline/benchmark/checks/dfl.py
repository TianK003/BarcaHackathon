"""Checks of the DFL parsers and of the cache of parsed matches (pipeline/benchmark/dfl.py), synthetic and --dfl.

The synthetic checks (the two parsers, and load_match with its cache) read a match written by write_synthetic_dfl in
the real file format, so they need no download. The test_dfl_* checks need the 21 downloaded files and run only with
--dfl. Nothing here prints a row of the real match's data: the DFL files are public (CC BY 4.0), so their aggregates
may be printed freely.
"""

import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import types
import warnings

import numpy as np

from pipeline.benchmark.checks.common import (
    DFL_TEAMS, LOAD_SECONDS, NotRun, PERSON_A, PERSON_B, PERSON_BALL, PERSON_C, PERSON_REFEREE, PERSON_UNLISTED,
    SYNTHETIC_BALL_FRAMES, SYNTHETIC_PLAYER_FRAMES, SYNTHETIC_REPORT, damaged, grid_x, grid_y, real_match,
    write_synthetic_dfl)
from pipeline.selftest import expect_error, write_text

# Params
# The cross-checks on J03WMX (kloppy, and the spot check by frame number).
TOLERANCE_M = 1e-6                    # metres: two readers of one coordinate may not differ by more than this
KLOPPY_LIMIT = 5000                   # kloppy's limit= is per object: the first 5,000 frames, all in the first half
HALF_START_N = {1: 10000, 2: 100000}  # J03WMX's first frame number in each half (test_dfl_positions_one_match)
GAP_ROWS = 25                         # separator rows between the halves, as the plan fixes it (not read from dfl)
SPOT_FRAMES = 200                     # frame numbers drawn from each half for the spot check
# What docs/traps.md T30 and results.md A05 quote of kloppy's first KLOPPY_LIMIT frames of J03WMX (measured 2026-10-22):
# of the 5,000 frames with only_alive=False, how many are alive and how many have the home team in possession, how
# many players kloppy's first frame has (each one's team is compared) and how many x and y values of the 3 players
# that are followed (2 x 3 x 5,000) were compared; and with the default only_alive=True, the frames it returned, the
# range of N they cover, the rows in that range and the dead rows it dropped from it (5,000 + 824 = 5,824 =
# 15,823 - 10,000 + 1).
KLOPPY_FACTS = {"alive": 4176, "home": 1359, "players": 22, "player_xy_values": 30000, "default_frames": 5000,
                "default_first": 10000, "default_last": 15823, "default_rows": 5824, "default_dropped": 824}
# What docs/traps.md T30 and results.md A05 quote, measured through our parser on each match (2026-10-22): the ball
# frames, the rows with no ball frame (kloppy skips them), the ball frames with no BallPossession (it turns them into
# the away team), the ball frames with no BallStatus, the X, Y, S and Z that are missing on any frame, and the dead
# ball frames (its default only_alive=True drops them), and the player frames that lie outside both halves, which
# our parser drops (the docs say that none does). A BallPossession that is not 1 or 2 is not a count: our parser
# refuses such a file, so every match that parses has none.
T30_FACTS = {
    match_id: {"ball_frames": frames, "rows_without_ball": 0, "no_possession": 0, "no_status": 0, "no_value": 0,
               "dead": dead, "player_frames_outside_halves": 0}
    for match_id, frames, dead in (
        ("J03WMX", 145967, 60598), ("J03WN1", 141561, 73184), ("J03WOH", 137214, 65269), ("J03WOY", 142536, 66250),
        ("J03WPY", 146211, 64751), ("J03WQQ", 142345, 57993), ("J03WR9", 146810, 61494))}


# -- the match information parser -------------------------------------------------------------------------------------

def test_match_information_synthetic():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info = dfl.parse_match_information(write_synthetic_dfl(folder)["information"])
    assert info.match_id == "SYN001", "the short id, without the DFL-MAT- prefix: " + info.match_id
    assert (info.home_id, info.home_name) == ("DFL-CLU-SYNHOM", "Synthetic Home FC"), info
    # The guest team is listed first in the file, and is the away team: Role decides, not the order.
    assert (info.away_id, info.away_name) == ("DFL-CLU-SYNAWY", "Synthetic Away FC"), info
    assert info.pitch == (104.5, 67.5) and all(type(side) is float for side in info.pitch), info.pitch
    assert info.total_time_raw == {1: 2700040.0, 2: 2820120.0}, info.total_time_raw
    assert all(type(key) is int and type(value) is float for key, value in info.total_time_raw.items())
    # Home first, each team in PersonId order; the trainer and the referee are not players; a name is FirstName and
    # LastName, else Shortname, else the PersonId.
    assert info.players == (
        dfl.Player("DFL-OBJ-SYNH01", 0, 1, "Hanna Keeper"),
        dfl.Player("DFL-OBJ-SYNH02", 0, 10, "DFL-OBJ-SYNH02"),
        dfl.Player("DFL-OBJ-SYNH03", 0, 4, "Hal Third"),
        dfl.Player("DFL-OBJ-SYNA01", 1, 1, "Ann Goalie"),
        dfl.Player("DFL-OBJ-SYNA02", 1, 23, "B. Short"),
        dfl.Player("DFL-OBJ-SYNA03", 1, 9, "Zoë Maria Müller"),
    ), info.players
    assert type(info.players) is tuple, type(info.players)
    assert all(type(player.team) is int and type(player.shirt) is int for player in info.players), info.players
    assert info == dfl.MatchInfo(
        match_id="SYN001", home_id="DFL-CLU-SYNHOM", away_id="DFL-CLU-SYNAWY", home_name="Synthetic Home FC",
        away_name="Synthetic Away FC", pitch=(104.5, 67.5), players=info.players,
        total_time_raw={1: 2700040.0, 2: 2820120.0}), "MatchInfo must compare by value"
    try:
        info.pitch = (1.0, 1.0)
    except AttributeError:
        pass
    else:
        raise AssertionError("MatchInfo must be frozen")


def test_match_information_follows_the_match_id():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = write_synthetic_dfl(folder, match_id="SYN002")["information"]
        assert path.endswith("_DFL-MAT-SYN002.xml"), path
        assert dfl.parse_match_information(path).match_id == "SYN002"


def test_match_information_rejects_bad_role():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = damaged(write_synthetic_dfl(folder)["information"],
                       lambda text: text.replace('Role="guest"', 'Role="visitor"'))
        expect_error(ValueError, lambda: dfl.parse_match_information(path), os.path.basename(path), "Role", "visitor",
                     '"home"', '"guest"')


def test_match_information_rejects_damaged_files():
    from pipeline.benchmark import dfl

    def without_teams(text):
        return re.sub(r"<Teams>.*</Teams>", "<Teams></Teams>", text, flags=re.DOTALL)

    cases = [  # label, how to damage the file, what the message must name besides the file
        ("two home teams", lambda text: text.replace('Role="guest"', 'Role="home"'), ['Role="home"', "two <Team>"]),
        ("no teams", without_teams, ['Role="home"', 'Role="guest"']),
        # The guest team is the first one in the file, so count=1 empties its squad.
        ("a team with no players",
         lambda text: re.sub(r"<Players>.*?</Players>", "<Players></Players>", text, count=1, flags=re.DOTALL),
         ["Synthetic Away FC", 'Role="guest"', "lists no <Player>", "--verify"]),
        ("two <Environment> elements", lambda text: re.sub(r"<Environment[^>]*/>", lambda match: match.group(0) * 2,
                                                           text),
         ["<Environment>", "2 <Environment> elements", "--verify"]),
        ("no pitch length", lambda text: text.replace(' PitchX="104.50"', ""), ["PitchX", "Environment", "--verify"]),
        ("pitch that is no number", lambda text: text.replace('PitchY="67.50"', 'PitchY="wide"'),
         ["PitchY", "wide", "--verify"]),
        ("negative pitch", lambda text: text.replace('PitchX="104.50"', 'PitchX="-105"'), ["PitchX", "-105"]),
        ("shirt that is no number", lambda text: text.replace('ShirtNumber="9"', 'ShirtNumber="nine"'),
         ["ShirtNumber", "nine", "DFL-OBJ-SYNA03", "--verify"]),
        ("a shirt that is missing", lambda text: text.replace(' ShirtNumber="9"', ""),
         ["no ShirtNumber attribute", "DFL-OBJ-SYNA03", "--verify"]),
        ("a player in both teams", lambda text: text.replace("DFL-OBJ-SYNH01", "DFL-OBJ-SYNA01"),
         ["DFL-OBJ-SYNA01", "twice"]),
        ("no second half duration", lambda text: text.replace(' TotalTimeSecondHalf="2820120"', ""),
         ["TotalTimeSecondHalf", "--verify"]),
        ("match id without its prefix", lambda text: text.replace("DFL-MAT-SYN001", "SYN001"),
         ["MatchId", "DFL-MAT-"]),
        ("General naming the other home team",
         lambda text: text.replace('HomeTeamId="DFL-CLU-SYNHOM"', 'HomeTeamId="DFL-CLU-SYNAWY"'),
         ["HomeTeamId", "Role"]),
        ("a positions file instead", lambda text: "<PutDataRequest><Positions /></PutDataRequest>",
         ["MatchInformation", "Positions", "match information file"]),
        ("a file cut short", lambda text: text[:len(text) // 2], ["well-formed", "--verify"]),
    ]
    with tempfile.TemporaryDirectory() as folder:
        written = write_synthetic_dfl(folder)["information"]
        with open(written, encoding="utf-8", newline="") as handle:
            intact = handle.read()
        for label, damage, fragments in cases:
            write_text(folder, os.path.basename(written), intact)
            path = damaged(written, damage)
            try:
                expect_error(ValueError, lambda: dfl.parse_match_information(path), os.path.basename(path),
                             *fragments)
            except AssertionError as error:
                raise AssertionError("{}: {}".format(label, error)) from error

    # A duplicated element in an otherwise valid file is a damaged download, not another kind of file.
    with tempfile.TemporaryDirectory() as folder:
        path = damaged(write_synthetic_dfl(folder)["information"],
                       lambda text: re.sub(r"<Environment[^>]*/>", lambda match: match.group(0) * 2, text))
        message = expect_error(ValueError, lambda: dfl.parse_match_information(path), "--verify")
        assert "not a DFL match information file" not in message, message


def test_match_information_names_a_missing_file():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "DFL_02_01_matchinformation_DFL-COM-SYNTH_DFL-MAT-SYN001.xml")
        expect_error(FileNotFoundError, lambda: dfl.parse_match_information(path), os.path.basename(path),
                     "pipeline/benchmark/sources.py")


# -- the positions parser ---------------------------------------------------------------------------------------------

MATCH_ARRAYS = ("frame_number", "players_xy", "players_speed", "player_team", "ball_xy", "ball_z", "ball_speed",
                "ball_status", "ball_possession")


def positions_of(folder, damage=None):
    """(info, path) of a fresh synthetic match in `folder`; `damage`, if given, rewrites the positions file first."""
    from pipeline.benchmark import dfl
    paths = write_synthetic_dfl(folder)
    info = dfl.parse_match_information(paths["information"])
    return info, (paths["positions"] if damage is None else damaged(paths["positions"], damage))


def leaves(value):
    """Every number at the bottom of nested dicts."""
    return [leaf for item in value.values() for leaf in leaves(item)] if isinstance(value, dict) else [value]


def synthetic_row(n):
    """The timeline row of frame number N in the synthetic match, or None outside both halves."""
    if 10000 <= n <= 10009:
        return n - 10000
    if 100000 <= n <= 100007:
        return 35 + n - 100000  # 10 rows, 25 of separator, then the second half
    return None


def frameset_block(text, section, person):
    """The text of the one <FrameSet> of this person and half."""
    pattern = r'<FrameSet GameSection="{}" [^>]*PersonId="{}">.*?</FrameSet>'.format(section, person)
    found = re.findall(pattern, text, flags=re.DOTALL)
    assert len(found) == 1, "{} FrameSets for {} in {}".format(len(found), person, section)
    return found[0]


def edit_frameset(section, person, change):
    """A damage for `damaged`: the FrameSet of this person and half becomes change(its text)."""
    def damage(text):
        block = frameset_block(text, section, person)
        return text.replace(block, change(block))
    return damage


def edit_frame(n, change):
    """A change for edit_frameset: the line of the frame with this N becomes change(line)."""
    def edit(block):
        lines = block.split("\n")
        hits = [i for i, line in enumerate(lines) if '<Frame N="{}" '.format(n) in line]
        assert len(hits) == 1, "{} frames with N={}".format(len(hits), n)
        lines[hits[0]] = change(lines[hits[0]])
        return "\n".join(lines)
    return edit


def edit_ball_frame(n, old, new):
    """A damage: in the ball's frame N, the text `old` becomes `new`."""
    return edit_frameset("firstHalf" if n < 100000 else "secondHalf", PERSON_BALL,
                         edit_frame(n, lambda line: line.replace(old, new)))


def both(first, second):
    """A damage that applies two damages, one after the other."""
    return lambda text: second(first(text))


def insert_after(marker, addition):
    """A damage: `addition` goes right after the first `marker`."""
    return lambda text: text.replace(marker, marker + addition, 1)


def open_the_first_frameset(text):
    """A damage: the first FrameSet (the referee's) is never closed, so each one after it starts inside it.

    The closing tag that is missing is put back at the end, which keeps the file well-formed, so that the error is the
    parser's own and not the XML parser's.
    """
    opened = text.replace("</FrameSet>\n<FrameSet", "\n<FrameSet", 1)
    assert opened != text, "the synthetic file has no two FrameSets in a row"
    return opened.replace("</Positions>", "</FrameSet>\n</Positions>")


def expect_positions_errors(cases, error=ValueError):
    """Each case (label, damage, fragments): parse the synthetic match damaged that way; it must raise `error`."""
    from pipeline.benchmark import dfl
    for label, damage, fragments in cases:
        with tempfile.TemporaryDirectory() as folder:
            info, path = positions_of(folder, damage)
            try:
                expect_error(error, lambda: dfl.parse_positions(path, info), os.path.basename(path), *fragments)
            except AssertionError as problem:
                raise AssertionError("{}: {}".format(label, problem)) from problem


def test_positions_synthetic_exact():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
    assert dfl.GAP_ROWS == 25 and dfl.SECTIONS == {"firstHalf": 1, "secondHalf": 2}, (dfl.GAP_ROWS, dfl.SECTIONS)
    assert (match.match_id, match.pitch, match.fps) == ("SYN001", (104.5, 67.5), 25), match
    assert (match.n_frames, match.n_players) == (43, 6), (match.n_frames, match.n_players)
    assert match.graded == ((1, 0, 10), (2, 35, 43)), match.graded
    assert match.kickoffs == {1: 0, 2: 35}, match.kickoffs
    assert all(type(value) is int for block in match.graded for value in block), match.graded
    assert match.frame_number[10] == -1 and match.frame_number[35] == 100000, match.frame_number[[10, 35]]
    assert match.ball_status[39] == -1, "N 100004 has no ball frame: its codes are -1"
    assert list(match.player_team) == [0, 0, 0, 1, 1, 1], "home first, then away, as the squads list them"
    assert [player.person_id for player in info.players][::3] == [PERSON_A, PERSON_C], info.players

    shapes = {"frame_number": ((43,), np.int64), "players_xy": ((43, 6, 2), np.float64),
              "players_speed": ((43, 6), np.float64), "player_team": ((6,), np.int8), "ball_xy": ((43, 2), np.float64),
              "ball_z": ((43,), np.float64), "ball_speed": ((43,), np.float64), "ball_status": ((43,), np.int8),
              "ball_possession": ((43,), np.int8)}
    assert sorted(shapes) == sorted(MATCH_ARRAYS)
    for name, (shape, dtype) in shapes.items():
        array = getattr(match, name)
        assert (array.shape, array.dtype) == (shape, np.dtype(dtype)), (name, array.shape, array.dtype)
        assert not array.flags.writeable, "{} must be read-only: a cached match returns it that way".format(name)

    # The rows of C: NaN before he comes on at N 100003 (rows 35-37), then set.
    columns = {PERSON_A: 0, PERSON_B: 1, PERSON_C: 3}
    assert np.isnan(match.players_xy[35:38, 3]).all() and np.isnan(match.players_speed[35:38, 3]).all()
    assert tuple(match.players_xy[38, 3]) == (0.5, 0.5), match.players_xy[38, 3]
    assert not np.isnan(match.players_speed[38, 3])
    # Normalised exactly, and y flipped: A's frame N 10002 is X = -26.125, Y = 16.875 and gives (0.25, 0.25); her
    # frame N 10009 is X = 52.25, Y = 25.3125 and gives (1.0, 0.125), where an unflipped y would be 0.875.
    assert (grid_x(2), grid_y(2)) == ("-26.125", "16.875"), (grid_x(2), grid_y(2))
    assert tuple(match.players_xy[2, 0]) == (0.25, 0.25) and tuple(match.players_xy[9, 0]) == (1.0, 0.125), \
        (match.players_xy[2, 0], match.players_xy[9, 0])
    # Speed is S / 3.6 (km/h to m/s), NaN where S is missing; the same for the ball.
    assert match.ball_speed[0] == 7.2 / 3.6 and np.isnan(match.ball_speed[3]), match.ball_speed[:4]
    assert match.players_speed[0, 0] == 10.8 / 3.6 and np.isnan(match.players_speed[7, 1]), match.players_speed[:8, :2]
    assert match.ball_z[4] == 1.0 and np.isnan(match.ball_z[8]), "Z is metres as written, NaN where missing"
    assert (match.ball_status[7], match.ball_possession[7]) == (0, 1), "a dead frame, last the away team's"
    assert (match.ball_status[42], match.ball_possession[42]) == (0, 0), "a dead frame, last the home team's"
    assert (match.ball_status[9], match.ball_possession[9]) == (-1, 1), "a missing code is -1, the other stays"
    assert (match.ball_status[41], match.ball_possession[41]) == (1, -1), (match.ball_status[41],
                                                                        match.ball_possession[41])

    # Everything, built independently from the tables that wrote the file.
    want_xy, want_speed = np.full((43, 6, 2), np.nan), np.full((43, 6), np.nan)
    for (person, _), frames in SYNTHETIC_PLAYER_FRAMES.items():
        for n, jx, jy, speed in frames:
            row = synthetic_row(n)
            if row is None:
                continue  # before or after its half: dropped
            want_xy[row, columns[person]] = (np.nan if jx is None else jx / 8, np.nan if jy is None else jy / 8)
            want_speed[row, columns[person]] = np.nan if speed is None else float(speed) / 3.6
    want_ball_xy, want_ball_z, want_ball_speed = np.full((43, 2), np.nan), np.full(43, np.nan), np.full(43, np.nan)
    want_possession, want_status = np.full(43, -1, np.int8), np.full(43, -1, np.int8)
    for frames in SYNTHETIC_BALL_FRAMES.values():
        for n, jx, jy, z, speed, possession, status in frames:
            row = synthetic_row(n)
            want_ball_xy[row] = (jx / 8, jy / 8)
            want_ball_z[row] = np.nan if z is None else float(z)
            want_ball_speed[row] = np.nan if speed is None else float(speed) / 3.6
            want_possession[row] = -1 if possession is None else {"1": 0, "2": 1}[possession]
            want_status[row] = -1 if status is None else int(status)
    want_frames = np.concatenate([np.arange(10000, 10010), np.full(25, -1), np.arange(100000, 100008)])
    for name, want in (("frame_number", want_frames), ("players_xy", want_xy), ("players_speed", want_speed),
                       ("ball_xy", want_ball_xy), ("ball_z", want_ball_z), ("ball_speed", want_ball_speed),
                       ("ball_status", want_status), ("ball_possession", want_possession)):
        try:
            np.testing.assert_array_equal(getattr(match, name), want)
        except AssertionError as error:
            raise AssertionError("{} differs from the tables that wrote the file: {}".format(name, error)) from error

    assert match.report == SYNTHETIC_REPORT, match.report
    assert json.loads(json.dumps(match.report, sort_keys=True)) == match.report, "the report must be JSON-ready"
    assert all(type(leaf) is int for leaf in leaves(match.report)), "counts must be Python ints, not numpy's"
    try:
        match.fps = 30
    except AttributeError:
        pass
    else:
        raise AssertionError("DflMatch must be frozen")
    assert dfl.DflMatch.__eq__ is object.__eq__, "eq=False: a field-wise == over arrays is meaningless"


def test_positions_ignore_the_order_of_the_file():
    """The ball first and the FrameSets reversed: the same match, because rows come from N and never from position."""
    from pipeline.benchmark import dfl

    def reversed_framesets(text):
        blocks = re.findall(r"<FrameSet .*?</FrameSet>", text, flags=re.DOTALL)
        assert len(blocks) == 8 and "\n".join(blocks) in text, len(blocks)
        return text.replace("\n".join(blocks), "\n".join(reversed(blocks)))

    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        as_written = dfl.parse_positions(path, info)
        reordered = dfl.parse_positions(damaged(path, reversed_framesets), info)
    for name in MATCH_ARRAYS:
        np.testing.assert_array_equal(getattr(reordered, name), getattr(as_written, name), err_msg=name)
    assert reordered.report == as_written.report and reordered.graded == as_written.graded
    assert reordered != as_written, "two parses are two objects"


def test_positions_rejects_unknown_section():
    # The section is checked on every FrameSet, those the parser skips (the referee) included.
    cases = [(label, edit_frameset("firstHalf", person, lambda block: block.replace(
        'GameSection="firstHalf"', 'GameSection="firstHalfExtra"')), ["firstHalfExtra", "GameSection", "secondHalf"])
        for label, person in (("a referee's FrameSet", PERSON_REFEREE), ("a player's", PERSON_B),
                              ("the ball's", PERSON_BALL))]
    expect_positions_errors(cases)


def test_positions_rejects_bad_label():
    expect_positions_errors([
        ("BallStatus 3", edit_ball_frame(10002, 'BallStatus="1"', 'BallStatus="3"'),
         ["1 ball frame(s)", "N=10002", "firstHalf", "BallStatus='3'", "BallStatus that is not 0 or 1"]),
        ("BallPossession 0", edit_ball_frame(10001, 'BallPossession="1"', 'BallPossession="0"'),
         ["N=10001", "BallPossession='0'"]),
        ("a possession that is no integer", edit_ball_frame(10001, 'BallPossession="1"', 'BallPossession="1.5"'),
         ["BallPossession='1.5'"]),
        ("a possession that is text", edit_ball_frame(100001, 'BallPossession="2"', 'BallPossession="home"'),
         ["N=100001", "secondHalf", "BallPossession='home'"]),
        ("an empty status", edit_ball_frame(10003, 'BallStatus="1"', 'BallStatus=""'), ["N=10003", "BallStatus=''"]),
        ("two bad frames, in two halves", both(edit_ball_frame(100007, 'BallStatus="0"', 'BallStatus="2"'),
                                               edit_ball_frame(10004, 'BallPossession="1"', 'BallPossession="9"')),
         ["2 ball frame(s)", "N=10004", "firstHalf", "BallPossession='9'"])])


def test_positions_rejects_team_mismatch():
    def team(section, person, club, other):
        return edit_frameset(section, person, lambda block: block.replace(
            'TeamId="{}"'.format(club), 'TeamId="{}"'.format(other)))

    expect_positions_errors([
        ("a home player under the away club", team("secondHalf", PERSON_A, "DFL-CLU-SYNHOM", "DFL-CLU-SYNAWY"),
         [PERSON_A, "home", "DFL-CLU-SYNHOM", 'TeamId="DFL-CLU-SYNAWY"', "swap home and away"]),
        ("an away player under the home club", team("secondHalf", PERSON_C, "DFL-CLU-SYNAWY", "DFL-CLU-SYNHOM"),
         [PERSON_C, "away", "DFL-CLU-SYNAWY", 'TeamId="DFL-CLU-SYNHOM"']),
        ("a player under a club that is neither", team("firstHalf", PERSON_B, "DFL-CLU-SYNHOM", "DFL-CLU-NOBODY"),
         [PERSON_B, "DFL-CLU-SYNHOM", 'TeamId="DFL-CLU-NOBODY"']),
        # The club is checked before the skip and the ball rules, so neither a referee's TeamId nor the ball's lets a
        # rostered player through (as a referee he would be skipped silently, as the ball he would be read as it).
        ("a home player under the referees' TeamId", team("firstHalf", PERSON_B, "DFL-CLU-SYNHOM", "referee"),
         [PERSON_B, "home", "DFL-CLU-SYNHOM", 'TeamId="referee"', "swap home and away"]),
        ("an away player under the ball's TeamId", team("secondHalf", PERSON_C, "DFL-CLU-SYNAWY", "BALL"),
         [PERSON_C, "away", "DFL-CLU-SYNAWY", 'TeamId="BALL"', "swap home and away"])])


def test_positions_rejects_other_match():
    other = "DFL-MAT-OTHER9"
    fragments = [other, "DFL-MAT-SYN001", "different matches"]
    expect_positions_errors([
        ("every FrameSet", lambda text: text.replace('MatchId="DFL-MAT-SYN001"', 'MatchId="{}"'.format(other)),
         fragments),
        ("one skipped FrameSet", edit_frameset("firstHalf", PERSON_REFEREE, lambda block: block.replace(
            'MatchId="DFL-MAT-SYN001"', 'MatchId="{}"'.format(other))), fragments)])


def test_positions_rejects_duplicates():
    expect_positions_errors([
        ("a person's FrameSet twice", edit_frameset("firstHalf", PERSON_B, lambda block: block + "\n" + block),
         [PERSON_B, 'GameSection="firstHalf"', "appears twice"]),
        ("a frame number twice in a FrameSet", edit_frameset("firstHalf", PERSON_B, edit_frame(
            10003, lambda line: line.replace('N="10003"', 'N="10002"'))),
         [PERSON_B, 'GameSection="firstHalf"', "N=10002", "twice"]),
        ("a frame number twice in the ball's", edit_ball_frame(100003, 'N="100003"', 'N="100002"'),
         ['TeamId="BALL"', "N=100002", "twice"]),
        ("a second ball FrameSet", edit_frameset("firstHalf", PERSON_BALL, lambda block: block + "\n" + block),
         ["second ball FrameSet", "firstHalf", 'TeamId="BALL"'])])


def test_positions_rejects_a_match_without_a_ball_in_each_half():
    def without_frames(block):
        return "\n".join(line for line in block.split("\n") if not line.startswith("<Frame "))

    expect_positions_errors([
        ("no ball in the second half", edit_frameset("secondHalf", PERSON_BALL, lambda block: ""),
         ["no ball FrameSet", "secondHalf", 'TeamId="BALL"', "--verify"]),
        ("no ball in the first half", edit_frameset("firstHalf", PERSON_BALL, lambda block: ""),
         ["no ball FrameSet", "firstHalf"]),
        ("a ball FrameSet without frames", edit_frameset("secondHalf", PERSON_BALL, without_frames),
         ["ball FrameSet", "secondHalf", "no frames"]),
        ("a file without any FrameSet", lambda text: "<PutDataRequest><Positions /></PutDataRequest>",
         ["no ball FrameSet", "firstHalf", "positions file"])])


def test_positions_rejects_damaged_files():
    def frame(person, section, n, old, new):
        return edit_frameset(section, person, edit_frame(n, lambda line: line.replace(old, new)))

    def attribute_removed(person, section, name, value):
        return edit_frameset(section, person, lambda block: block.replace(' {}="{}"'.format(name, value), "", 1))

    expect_positions_errors([
        ("a file cut short", lambda text: text[:len(text) // 2], ["well-formed", "--verify"]),
        ("a frame without N", frame(PERSON_A, "secondHalf", 100001, ' N="100001"', ""),
         [PERSON_A, "secondHalf", "no N attribute", "--verify"]),
        ("an N that is no number", frame(PERSON_A, "secondHalf", 100001, 'N="100001"', 'N="ten"'),
         [PERSON_A, "N='ten'", "whole number"]),
        ("an X that is no number", frame(PERSON_A, "secondHalf", 100000, 'X="{}"'.format(grid_x(1)), 'X="abc"'),
         [PERSON_A, "X='abc'", "finite number"]),
        ("an infinite Y", frame(PERSON_B, "firstHalf", 10000, 'Y="{}"'.format(grid_y(6)), 'Y="inf"'),
         [PERSON_B, "Y='inf'", "finite number"]),
        ("an S that is no number", frame(PERSON_C, "secondHalf", 100003, 'S="21.6"', 'S="fast"'),
         [PERSON_C, "S='fast'"]),
        ("a Z that is no number", edit_ball_frame(10001, 'Z="0.25"', 'Z="high"'), ['TeamId="BALL"', "Z='high'"]),
        ("a FrameSet without PersonId", attribute_removed(PERSON_B, "firstHalf", "PersonId", PERSON_B),
         ["no PersonId attribute", "--verify"]),
        ("a FrameSet without TeamId", attribute_removed(PERSON_B, "firstHalf", "TeamId", "DFL-CLU-SYNHOM"),
         ["no TeamId attribute", PERSON_B]),
        ("a FrameSet without GameSection", attribute_removed(PERSON_B, "firstHalf", "GameSection", "firstHalf"),
         ["no GameSection attribute", PERSON_B]),
        ("a <Frame> before any FrameSet", insert_after("</MetaData>\n", '<Frame N="1" />\n'),
         ["<Frame> outside any <FrameSet>", "positions file", "--verify"]),
        ("a <Frame> between two FrameSets", insert_after("</FrameSet>\n", '<Frame N="1" />\n'),
         ["<Frame> outside any <FrameSet>", "positions file", "--verify"]),
        ("a FrameSet inside another", open_the_first_frameset,
         ["starts inside", PERSON_REFEREE, PERSON_UNLISTED, "positions file", "--verify"])])


def test_the_structural_refusals_give_the_advice_once():
    """A FrameSet in a FrameSet and a <Frame> outside any: what is wrong, then the one piece of advice, not two."""
    from pipeline.benchmark import dfl
    for damage in (insert_after("</MetaData>\n", '<Frame N="1" />\n'), open_the_first_frameset):
        with tempfile.TemporaryDirectory() as folder:
            info, path = positions_of(folder, damage)
            message = expect_error(ValueError, lambda: dfl.parse_positions(path, info), dfl.VERIFY_ADVICE)
            assert message.count("download is") == 1 and message.count("--verify") == 1, message


def test_positions_names_a_missing_file():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        os.remove(path)
        expect_error(FileNotFoundError, lambda: dfl.parse_positions(path, info), os.path.basename(path),
                     "pipeline/benchmark/sources.py")


# -- the cache: dfl.load_match ----------------------------------------------------------------------------------------

MANIFEST_KEYS = ["cache_version", "fps", "graded", "kickoffs", "match_id", "parsed_with", "parser_sha256", "pitch",
                 "report", "sources"]
CACHE_FILES = sorted(["{}.npy".format(name) for name in MATCH_ARRAYS] + ["manifest.json"])


def cache_setup(root):
    """(paths of a synthetic match written under root/source, the folder its cache must be in)."""
    source = os.path.join(root, "source")
    os.makedirs(source)
    return write_synthetic_dfl(source), os.path.join(root, "cache", "dfl", "SYN001")


def load_synthetic(root, paths, **options):
    from pipeline.benchmark import dfl
    return dfl.load_match("SYN001", data_dir=root, paths=paths, **options)


def type_tree(value):
    """The types inside a nested value, as text, so that 1, 1.0 and "1" differ, and so do a tuple and a list."""
    if isinstance(value, (tuple, list)):
        return "{}[{}]".format(type(value).__name__, ", ".join(type_tree(item) for item in value))
    if isinstance(value, dict):
        return "dict[{}]".format(", ".join("{}: {}".format(type_tree(key), type_tree(item))
                                           for key, item in sorted(value.items(), key=repr)))
    return type(value).__name__


def assert_same_match(first, second):
    """Both DflMatch hold the same: arrays equal (NaN equal to NaN) with one dtype, scalars equal with one type.

    The report is compared without its "cache" entry, which says how the match was reached and so differs by design.
    """
    from pipeline.benchmark import dfl
    for field in dataclasses.fields(dfl.DflMatch):  # every field of the class, so that a new one is never skipped
        one, other = getattr(first, field.name), getattr(second, field.name)
        if field.name == "report":
            one, other = [{key: value for key, value in report.items() if key != "cache"} for report in (one, other)]
        if isinstance(one, np.ndarray):
            assert isinstance(other, np.ndarray), (field.name, type(other))
            assert (one.dtype, one.shape) == (other.dtype, other.shape), (field.name, one.dtype, other.dtype, one.shape,
                                                                           other.shape)
            np.testing.assert_array_equal(one, other, err_msg=field.name)
        else:
            assert one == other and type_tree(one) == type_tree(other), (field.name, one, other, type_tree(one),
                                                                         type_tree(other))


def edit_manifest(folder, change):
    """Rewrite the cache's manifest.json as change(its parsed content) (which edits it in place)."""
    path = os.path.join(folder, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    change(manifest)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def manifest_of(folder):
    """The cache's manifest.json, parsed."""
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_the_cache_stores_every_array_field():
    """The arrays the cache writes are exactly the array fields of DflMatch: a new field cannot be left out of it."""
    from pipeline.benchmark import dfl
    array_fields = {field.name for field in dataclasses.fields(dfl.DflMatch) if field.type is np.ndarray}
    assert array_fields == set(dfl.ARRAY_DTYPES) == set(MATCH_ARRAYS), (sorted(array_fields), sorted(dfl.ARRAY_DTYPES))


def test_match_cache_built_then_reused():
    from lxml import etree
    from pipeline.benchmark import dfl
    from pipeline.inputs.atd_cache import describe_source
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        built = load_synthetic(root, paths)
        assert built.report["cache"] == "built", built.report
        assert sorted(os.listdir(root)) == ["cache", "source"], "nothing may be written outside <data dir>/cache"
        assert [os.listdir(os.path.join(root, "cache")), os.listdir(os.path.join(root, "cache", "dfl"))] == [
            ["dfl"], ["SYN001"]], "the cache is <data dir>/cache/dfl/<match id>/"
        assert sorted(os.listdir(folder)) == CACHE_FILES, sorted(os.listdir(folder))

        manifest_path = os.path.join(folder, "manifest.json")
        with open(manifest_path, "rb") as handle:
            manifest_bytes = handle.read()
        manifest = json.loads(manifest_bytes)
        assert sorted(manifest) == MANIFEST_KEYS, sorted(manifest)
        assert dfl.CACHE_VERSION == 1 == manifest["cache_version"]
        assert manifest["sources"] == {"information": describe_source(paths["information"]),
                                       "positions": describe_source(paths["positions"])}
        assert manifest["parsed_with"] == {"numpy": np.__version__, "lxml": etree.__version__}
        with open(dfl.__file__, "rb") as handle:  # the parser is this very file, read here by a pass of its own
            parser = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
        assert manifest["parser_sha256"] == parser == dfl.PARSER_SHA256, manifest["parser_sha256"]
        assert "cache" not in manifest["report"], "how a match was reached must not be stored with it"
        assert manifest_bytes.endswith(b"}\n") and b"\r" not in manifest_bytes, "LF line endings, one final newline"
        stamp = os.stat(manifest_path).st_mtime_ns

        reused = load_synthetic(root, paths)
        assert reused.report["cache"] == "reused", reused.report
        assert os.stat(manifest_path).st_mtime_ns == stamp, "a cache that was fresh was rewritten"
        assert_same_match(built, reused)
        direct = dfl.parse_positions(paths["positions"], dfl.parse_match_information(paths["information"]))
        assert_same_match(direct, reused)  # what the cache gives is what a parse gives, field for field
        assert "cache" not in direct.report, "parse_positions itself knows nothing of the cache"

        forced = load_synthetic(root, paths, rebuild=True)
        assert forced.report["cache"] == "built", forced.report
        with open(manifest_path, "rb") as handle:
            assert handle.read() == manifest_bytes, "the same inputs must give the same manifest, byte for byte"
        assert_same_match(built, forced)


def test_match_cache_rebuilt_when_source_changes():
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        first = load_synthetic(root, paths)
        assert (first.report["cache"], first.ball_status[2]) == ("built", 1)

        # The key is the bytes, not the file's date: the same text written again is still the same cache.
        with open(paths["positions"], encoding="utf-8", newline="") as handle:
            text = handle.read()
        write_text(os.path.dirname(paths["positions"]), os.path.basename(paths["positions"]), text)
        os.utime(paths["positions"], (4_000_000_000, 4_000_000_000))
        assert load_synthetic(root, paths).report["cache"] == "reused"

        damaged(paths["positions"], edit_ball_frame(10002, 'BallStatus="1"', 'BallStatus="0"'))
        second = load_synthetic(root, paths)
        assert (second.report["cache"], second.ball_status[2]) == ("built", 0), "the positions changed: rebuild"
        third = load_synthetic(root, paths)
        assert third.report["cache"] == "reused" and third.ball_status[2] == 0, "and the new cache is the one reused"
        assert_same_match(second, third)

        damaged(paths["information"], lambda text: text.replace('TotalTimeFirstHalf="2700040"',
                                                                 'TotalTimeFirstHalf="2700041"'))
        assert load_synthetic(root, paths).report["cache"] == "built", "the match information changed: rebuild"
        assert load_synthetic(root, paths).report["cache"] == "reused"

        # The rest of the key: the format version and the library versions that did the parsing.
        for label, change in (("cache_version", lambda manifest: manifest.update(cache_version=0)),
                              ("numpy", lambda manifest: manifest["parsed_with"].update(numpy="0.0.1")),
                              ("lxml", lambda manifest: manifest["parsed_with"].update(lxml="0.0.1"))):
            edit_manifest(folder, change)
            assert load_synthetic(root, paths).report["cache"] == "built", "{} changed: rebuild".format(label)
            assert load_synthetic(root, paths).report["cache"] == "reused", "{}: the new manifest".format(label)


def test_match_cache_rebuilt_when_the_parser_changes():
    """The parser is part of the key: a parser that is not the one that built a cache does not read it."""
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        first = load_synthetic(root, paths)
        assert first.report["cache"] == "built"
        real_hash = dfl.PARSER_SHA256
        assert real_hash == manifest_of(folder)["parser_sha256"], "the manifest must hold the parser's own hash"
        dfl.PARSER_SHA256 = "0" * 64  # as if this were another version of the parser
        try:
            second = load_synthetic(root, paths)
            assert second.report["cache"] == "built", "a parser that changed was read through the old cache"
            assert manifest_of(folder)["parser_sha256"] == "0" * 64
            assert load_synthetic(root, paths).report["cache"] == "reused", "and the new cache is fresh for it"
        finally:
            dfl.PARSER_SHA256 = real_hash
        third = load_synthetic(root, paths)
        assert third.report["cache"] == "built", "the parser changed back, which is another change: rebuild"
        assert load_synthetic(root, paths).report["cache"] == "reused"
        assert_same_match(first, third)


def test_the_parser_hash_is_the_hash_of_dfl_py():
    """The constant is the hash of dfl.py's text with its line endings made LF (see the check below for why)."""
    from pipeline.benchmark import dfl
    with open(dfl.__file__, "rb") as handle:
        assert dfl.PARSER_SHA256 == hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def import_parser(path, name):
    """The file at `path` imported as a module called `name`, which runs its code and takes its hash right now."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


def parser_copy(folder, name, ending, extra=""):
    """dfl.py's text plus `extra`, written to folder/name.py with this line ending and imported as module `name`."""
    from pipeline.benchmark import dfl
    with open(dfl.__file__, encoding="utf-8", newline="") as handle:
        text = handle.read().replace("\r\n", "\n") + extra
    path = os.path.join(folder, name + ".py")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text.replace("\n", ending))
    return import_parser(path, name)


def test_the_parser_hash_does_not_depend_on_line_endings():
    """dfl.py checked out with CRLF (git's core.autocrlf on Windows) is the same parser: it must not rebuild a cache.

    The same text as an LF file, as a CRLF file, and with one comment line added (a real edit), each imported as a
    module of its own, so that each hashes the file it was loaded from.
    """
    from pipeline.benchmark import dfl
    names = ("dfl_copy_lf", "dfl_copy_crlf", "dfl_copy_edited")
    try:
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as root:
            lf = parser_copy(folder, names[0], "\n")
            crlf = parser_copy(folder, names[1], "\r\n")
            edited = parser_copy(folder, names[2], "\n", extra="# an edit\n")
            with open(crlf.__file__, "rb") as handle:
                raw = handle.read()
            assert raw.count(b"\r\n") > 100 and raw.count(b"\n") == raw.count(b"\r\n"), "the copy is not all CRLF"
            assert lf.PARSER_SHA256 == crlf.PARSER_SHA256 == dfl.PARSER_SHA256, "line endings changed the hash"
            assert edited.PARSER_SHA256 != lf.PARSER_SHA256, "an edit did not change the hash"

            # What it is for: the cache one checkout built is fresh for the other, and not for the edited parser.
            paths, _ = cache_setup(root)
            args = ("SYN001",)
            options = dict(data_dir=root, paths=paths)
            statuses = [module.load_match(*args, **options).report["cache"] for module in (lf, crlf, lf, edited)]
            assert statuses == ["built", "reused", "reused", "built"], statuses
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_the_cache_is_keyed_on_the_parser_that_is_running():
    """The key is the parser's hash from when it was imported, not what dfl.py holds when load_match is called.

    A process that imported the parser before dfl.py changed (a notebook left open across a git pull, a long --dfl run,
    a lane merged meanwhile) keeps running the old code, so its cache must describe that code: it neither rebuilds
    because the file moved on, nor stamps the new file's hash on arrays that its old parser made.
    """
    names = ("dfl_copy_old", "dfl_copy_new")
    try:
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as root:
            paths, cache_folder = cache_setup(root)
            options = dict(data_dir=root, paths=paths)
            old = parser_copy(folder, names[0], "\n")
            assert old.load_match("SYN001", **options).report["cache"] == "built"

            with open(old.__file__, "a", encoding="utf-8", newline="") as handle:  # edited under the running process
                handle.write("FPS = 26\n")  # a change of behaviour: the edited parser reads 26 frames a second
            reused = old.load_match("SYN001", **options)
            assert (reused.report["cache"], reused.fps) == ("reused", 25), \
                "the running parser did not recognise the cache it built: it keyed on the file as it is now"

            new = import_parser(old.__file__, names[1])  # a process that starts now runs the edited file
            assert new.PARSER_SHA256 != old.PARSER_SHA256
            rebuilt = new.load_match("SYN001", **options)
            assert (rebuilt.report["cache"], rebuilt.fps) == ("built", 26), \
                "the edited parser read a cache that the old parser built"
            assert manifest_of(cache_folder)["parser_sha256"] == new.PARSER_SHA256

            # The old process meets the new parser's cache. It rebuilds with the code it is running and says so in the
            # manifest, with its own hash and not that of the file on disk: no later process is led to trust old arrays.
            assert old.load_match("SYN001", **options).report["cache"] == "built"
            assert manifest_of(cache_folder)["parser_sha256"] == old.PARSER_SHA256 != new.PARSER_SHA256

            # And it does not read the file again: with dfl.py gone from disk a session still works.
            os.remove(old.__file__)
            assert old.load_match("SYN001", **options).report["cache"] == "reused"
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_importing_the_parser_does_not_load_pandas():
    """dfl.load_match needs the ATD cache's hashing, but only when it is called: parsing alone must stay light."""
    from pipeline.benchmark import sources
    code = "import sys; sys.path.insert(0, {!r}); import pipeline.benchmark.dfl; print('pandas' in sys.modules)".format(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sources.__file__)))))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", "importing dfl pulled in pandas: {!r}".format(result.stdout)


def test_match_cache_arrays_read_only():
    with tempfile.TemporaryDirectory() as root:
        paths, _ = cache_setup(root)
        for expected in ("built", "reused"):
            match = load_synthetic(root, paths)
            assert match.report["cache"] == expected, match.report
            for name in MATCH_ARRAYS:
                assert not getattr(match, name).flags.writeable, "{} is writable ({})".format(name, expected)


def test_match_cache_rebuilt_when_array_missing():
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        original = load_synthetic(root, paths)
        for name in MATCH_ARRAYS:
            os.remove(os.path.join(folder, name + ".npy"))  # the manifest is left in place
            rebuilt = load_synthetic(root, paths)
            assert rebuilt.report["cache"] == "built", "{}.npy was gone but the cache was reused".format(name)
            assert sorted(os.listdir(folder)) == CACHE_FILES, sorted(os.listdir(folder))
            assert_same_match(original, rebuilt)


def test_match_cache_rebuilt_when_an_array_is_damaged():
    """A file that is cut short, is not an array, or is an array of another shape or type is as good as missing."""
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        original = load_synthetic(root, paths)
        target = os.path.join(folder, "ball_status.npy")
        with open(target, "rb") as handle:
            intact = handle.read()

        def write_bytes(body):
            with open(target, "wb") as handle:
                handle.write(body)

        def save(array, **options):
            np.save(target, array, **options)

        def save_as_npz():
            with open(target, "wb") as handle:  # a zip archive of arrays, which np.load opens as an NpzFile
                np.savez(handle, ball_status=np.zeros(original.n_frames, dtype=np.int8))

        def npz_cut_short():
            buffer = io.BytesIO()
            np.savez(buffer, ball_status=np.zeros(original.n_frames, dtype=np.int8))
            write_bytes(buffer.getvalue()[:len(buffer.getvalue()) // 2])  # still starts with the zip signature

        # A version 1.0 .npy file: magic, version, a 2-byte header length, the header, then one byte per frame (int8).
        assert intact[:8] == b"\x93NUMPY\x01\x00", intact[:8]
        header_end = 10 + int.from_bytes(intact[8:10], "little")
        assert len(intact) - header_end == original.n_frames, (len(intact), header_end, original.n_frames)
        cases = [("the last byte missing", lambda: write_bytes(intact[:-1])),
                 ("half the data missing", lambda: write_bytes(intact[:header_end + original.n_frames // 2])),
                 ("only the header", lambda: write_bytes(intact[:header_end])),
                 ("cut inside the header", lambda: write_bytes(intact[:20])),
                 ("an .npz saved under the .npy name", save_as_npz), ("an .npz cut short", npz_cut_short),
                 ("not an array", lambda: write_bytes(b"this is not a numpy file")),
                 ("empty", lambda: write_bytes(b"")),
                 ("too few rows", lambda: save(np.zeros(original.n_frames - 1, dtype=np.int8))),
                 ("too many dimensions", lambda: save(np.zeros((original.n_frames, 1), dtype=np.int8))),
                 ("floats", lambda: save(np.zeros(original.n_frames, dtype=np.float64))),
                 ("objects", lambda: save(np.array([None] * original.n_frames, dtype=object), allow_pickle=True))]
        for label, damage in cases:
            damage()
            with resource_warnings() as caught:
                rebuilt = load_synthetic(root, paths)
            assert rebuilt.report["cache"] == "built", "{}: a damaged array was loaded".format(label)
            assert not caught, "{}: the load left a file open: {}".format(label, caught)
            assert_same_match(original, rebuilt)
            assert load_synthetic(root, paths).report["cache"] == "reused", label


@contextlib.contextmanager
def resource_warnings():
    """The ResourceWarnings that the block, and a garbage collection after it, raise: a list that fills as they come.

    Warnings as the interpreter raises them for an object that dies with its file open (always shown, and recorded
    rather than turned into an exception, because a warning raised inside a destructor never reaches the caller: it is
    only printed as an ignored exception).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        found = []
        try:
            yield found
        finally:
            gc.collect()
            found += [str(item.message) for item in caught if issubclass(item.category, ResourceWarning)]


def test_match_cache_array_reader_closes_its_file_on_every_path():
    """dfl._read_array leaves no file open, whether it returns an array, or raises, and whatever it raises.

    np.load hands the file to an NpzFile when it starts with a zip signature, and an .npz cut short fails inside the
    NpzFile, after np.load has given up the file: nothing closed it (Task 9 saw the ResourceWarning). On Windows an
    open file cannot be removed or rewritten, which is what the rebuild that follows a refused cache does, so each case
    removes the file while the exception, and the frames it holds, are still alive: the failure that a collection
    would have hidden. The warning is checked as well, for the platforms that let the file go.
    """
    from pipeline.benchmark import dfl
    array = np.arange(50, dtype=np.int8)
    plain, as_npz = io.BytesIO(), io.BytesIO()
    np.save(plain, array)
    np.savez(as_npz, array=array)
    plain, as_npz = plain.getvalue(), as_npz.getvalue()
    cases = [("a plain array", plain, None), ("an .npz under the .npy name", as_npz, ValueError),
             ("an .npz cut short", as_npz[:len(as_npz) // 2], ValueError),
             ("an .npz cut after its signature", as_npz[:6], ValueError),
             ("the last byte missing", plain[:-1], ValueError), ("only the header", plain[:128], ValueError),
             ("not an array", b"this is not a numpy file", ValueError), ("empty", b"", EOFError)]
    with tempfile.TemporaryDirectory() as root:
        for label, body, refused in cases:
            path = os.path.join(root, "array.npy")
            with open(path, "wb") as handle:
                handle.write(body)
            with resource_warnings() as caught:
                held = None
                try:
                    loaded = dfl._read_array(path)
                except (ValueError, EOFError) as error:
                    assert refused is not None and isinstance(error, refused), (label, type(error), str(error))
                    held = error                       # alive while the file is removed: its frames are alive too
                    os.remove(path)                    # PermissionError on Windows if the file is still open
                else:
                    assert refused is None, "{}: expected {}, nothing was raised".format(label, refused.__name__)
                    assert loaded.tolist() == array.tolist() and loaded.dtype == array.dtype
                    os.remove(path)
                del held
            assert not caught, "{}: a file was left open: {}".format(label, caught)
            assert not os.path.exists(path), label


def test_match_cache_rebuilt_when_manifest_damaged():
    """A manifest that is not JSON, not an object, or lacks or garbles a field is no manifest: rebuild, never raise."""
    with tempfile.TemporaryDirectory() as root:
        paths, folder = cache_setup(root)
        original = load_synthetic(root, paths)
        target = os.path.join(folder, "manifest.json")

        def write_bytes(body):
            def damage():
                with open(target, "wb") as handle:
                    handle.write(body)
            return damage

        def without(key):
            return lambda: edit_manifest(folder, lambda manifest: manifest.pop(key))

        def garbled(key, value):
            return lambda: edit_manifest(folder, lambda manifest: manifest.update({key: value}))

        cases = [("cut short", write_bytes(b'{"cache_version": 1, "fps": 25, "gra')), ("empty", write_bytes(b"")),
                 ("not UTF-8", write_bytes(b"\xff\xfe\x00 not text")), ("a list", write_bytes(b"[1, 2, 3]")),
                 ("null", write_bytes(b"null")), ("a number", write_bytes(b"7"))]
        cases += [("no {}".format(key), without(key)) for key in MANIFEST_KEYS]
        cases += [("pitch a number", garbled("pitch", 105)), ("pitch of text", garbled("pitch", ["a", "b"])),
                  ("pitch of one value", garbled("pitch", [105.0])),
                  ("match_id of another match", garbled("match_id", "SYN999")),
                  ("match_id a number", garbled("match_id", 7)),
                  ("graded blocks too short", garbled("graded", [[1, 2], [2, 3]])),
                  ("graded a number", garbled("graded", 4)), ("graded of text", garbled("graded", [["a", 1, 2]])),
                  ("kickoffs a list", garbled("kickoffs", [0, 35])),
                  ("kickoffs of text", garbled("kickoffs", {"x": 1})),
                  ("fps text", garbled("fps", "fast")), ("report a list", garbled("report", [1]))]
        for label, damage in cases:
            damage()
            rebuilt = load_synthetic(root, paths)
            assert rebuilt.report["cache"] == "built", "{}: a damaged manifest was trusted".format(label)
            assert_same_match(original, rebuilt)
            assert load_synthetic(root, paths).report["cache"] == "reused", "{}: the rebuilt manifest".format(label)


def test_match_cache_leaves_no_manifest_when_a_save_is_interrupted():
    """The manifest is what marks a cache as complete: out first, written last, so a killed run leaves none."""
    class Interrupted(Exception):
        pass

    def failing_after(allowed):
        calls = []
        real_save = np.save

        def save(*arguments, **options):
            if len(calls) == allowed:
                raise Interrupted("the disk is gone")
            calls.append(arguments[0])
            return real_save(*arguments, **options)
        return save, real_save

    for allowed in (0, 3, len(MATCH_ARRAYS) - 1):  # before the first array, in the middle, before the last
        with tempfile.TemporaryDirectory() as root:
            paths, folder = cache_setup(root)
            save, real_save = failing_after(allowed)
            np.save = save
            try:
                expect_error(Interrupted, lambda: load_synthetic(root, paths), "the disk is gone")
            finally:
                np.save = real_save
            assert not os.path.exists(os.path.join(folder, "manifest.json")), "a half-written cache has a manifest"
            assert load_synthetic(root, paths).report["cache"] == "built"
            assert load_synthetic(root, paths).report["cache"] == "reused"

            # The same with a complete cache that is being rebuilt: its old manifest must go before any array does.
            edit_manifest(folder, lambda manifest: manifest.update(cache_version=0))
            save, real_save = failing_after(allowed)
            np.save = save
            try:
                expect_error(Interrupted, lambda: load_synthetic(root, paths), "the disk is gone")
            finally:
                np.save = real_save
            assert not os.path.exists(os.path.join(folder, "manifest.json")), \
                "an interrupted rebuild left the old manifest in front of a mixture of old and new arrays"
            assert load_synthetic(root, paths).report["cache"] == "built"


def test_match_cache_names_what_is_wrong():
    from pipeline.benchmark import dfl, sources
    with tempfile.TemporaryDirectory() as root:
        paths, _ = cache_setup(root)
        # An id that is not one of the seven, and no paths: say which ids there are.
        expect_error(ValueError, lambda: dfl.load_match("J03WXX", data_dir=root), "J03WXX", *sources.MATCH_IDS)
        # A real id whose files are not under this data dir: the file, and how to fetch it.
        wanted = sources.local_path("J03WMX", "information", root)
        assert wanted.startswith(root), wanted
        expect_error(FileNotFoundError, lambda: dfl.load_match("J03WMX", data_dir=root), wanted,
                     "pipeline/benchmark/sources.py")
        os.remove(paths["positions"])
        expect_error(FileNotFoundError, lambda: load_synthetic(root, paths), os.path.basename(paths["positions"]),
                     "pipeline/benchmark/sources.py")
        # Paths that are not the two kinds of file.
        expect_error(ValueError, lambda: load_synthetic(root, {"information": paths["information"]}), "information",
                     "positions")
        expect_error(ValueError, lambda: load_synthetic(root, dict(paths, events="x.xml")), "information", "positions",
                     "events")
        assert not os.path.exists(os.path.join(root, "cache")), "a failed load must not leave a cache folder behind"

    # A match id that is not the match the files are of: the cache would be filed under the wrong name.
    with tempfile.TemporaryDirectory() as root:
        paths, _ = cache_setup(root)
        expect_error(ValueError, lambda: dfl.load_match("SYN002", data_dir=root, paths=paths), "SYN002", "SYN001",
                     os.path.basename(paths["information"]))
        assert not os.path.exists(os.path.join(root, "cache")), "a refused match must not be cached"


# -- the downloaded DFL files (--dfl) ---------------------------------------------------------------------------------

def test_dfl_files_verified():
    """All 21 files are on disk with the pinned size and md5. Reads about 2.6 GB."""
    from pipeline.benchmark import sources
    statuses = sources.verify()
    bad = {os.path.basename(path): status for path, status in statuses.items() if status != "ok"}
    assert len(statuses) == 21, len(statuses)
    assert not bad, "{} of 21 DFL files are not ok (e.g. {}); run python pipeline/benchmark/sources.py".format(
        len(bad), sorted(bad.items())[0])
    print("      {} files ok, {:,} bytes under {}".format(
        len(statuses), sum(remote.size for remote in sources.select()), sources.external_dir()))


def test_dfl_all_matches_load():
    """All 7 matches load through the cache, and a second pass finds every cache fresh.

    The first pass builds the cache of a match that has none (about a minute each) and reads the others; it prints
    the seconds and "built" or "reused". The second pass must say "reused" for all seven. One match is held at a
    time (about 150 MB of arrays), and each is checked against the ball frame count that sources.py pins.
    """
    from pipeline.benchmark import dfl, sources
    print("      {:<7} {:>6} {:<7} {:>8} {:>11} {:>9}".format("match", "pass", "cache", "seconds", "rows", "players"))
    for load_pass in ("first", "second"):
        for match_id in sources.MATCH_IDS:
            started = time.perf_counter()
            match = dfl.load_match(match_id)
            seconds = time.perf_counter() - started
            status = match.report["cache"]
            print("      {:<7} {:>6} {:<7} {:>8.1f} {:>11,} {:>9}".format(
                match_id, load_pass, status, seconds, match.n_frames, match.n_players))
            expected = sources.EXPECTED_BALL_FRAMES[match_id]
            assert (match.match_id, match.report["ball_frames"]) == (match_id, expected), \
                (match_id, match.match_id, match.report["ball_frames"], expected)
            assert status in ("built", "reused"), status
            assert load_pass == "first" or status == "reused", \
                "{}: loaded a second time and it was {}, not reused".format(match_id, status)
            del match


def test_dfl_cache_equals_fresh_parse():
    """J03WMX read back from the cache is, field for field, what parse_positions gives on the file right now.

    Every other --dfl check reads the match through the cache, so this is the one that runs the parser itself on real
    data: if the parser changed and a stale cache were still being read, it would be seen here. (The cache key holds
    the hash of the parser that is running, which should make that impossible; this is the measurement that says it
    is.) The cache is built first if it is not there, so that the match compared is one read back from disk; the
    fresh parse (about a minute) is dropped as soon as the two have been compared, so two matches, about 300 MB, are
    held only while they are.
    """
    from pipeline.benchmark import dfl, sources
    match_id = "J03WMX"
    cached = dfl.load_match(match_id)
    if cached.report["cache"] != "reused":
        del cached
        cached = dfl.load_match(match_id)
    assert cached.report["cache"] == "reused", cached.report["cache"]
    started = time.perf_counter()
    fresh = dfl.parse_positions(sources.local_path(match_id, "positions"),
                                dfl.parse_match_information(sources.local_path(match_id, "information")))
    seconds = time.perf_counter() - started
    assert_same_match(fresh, cached)
    print("      {:,} rows x {} players, {} arrays and the report equal, fresh parse {:.1f} s".format(
        fresh.n_frames, fresh.n_players, len(MATCH_ARRAYS), seconds))
    del fresh, cached


def test_dfl_match_information_all():
    """All 7 match information files parse, and say what the paper says: teams, a real pitch, full squads."""
    from pipeline.benchmark import dfl, sources
    print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>21}".format(
        "match", "home", "guest", "pitch (m)", "players", "total (raw)"))
    for match_id in sources.MATCH_IDS:
        info = dfl.parse_match_information(sources.local_path(match_id, "information"))
        assert info.match_id == match_id, (info.match_id, match_id)
        assert (info.home_name, info.away_name) == DFL_TEAMS[match_id], (match_id, info.home_name, info.away_name)
        assert 100 <= info.pitch[0] <= 110 and 64 <= info.pitch[1] <= 75, (match_id, info.pitch)
        counts = [sum(player.team == team for player in info.players) for team in (0, 1)]
        assert min(counts) >= 14, "{}: players per team {}; a squad has at least 14".format(match_id, counts)
        assert [player.team for player in info.players] == sorted(player.team for player in info.players), \
            "{}: players must list home first".format(match_id)
        assert sorted(info.total_time_raw) == [1, 2] and min(info.total_time_raw.values()) > 0, info.total_time_raw
        print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>21}".format(
            match_id, info.home_name, info.away_name, "{:g} x {:g}".format(*info.pitch),
            "{} + {}".format(*counts), "{:,.0f} / {:,.0f}".format(*(info.total_time_raw[half] for half in (1, 2)))))


def test_dfl_positions_one_match():
    """J03WMX's positions parse into a timeline that agrees with the information file and with itself (418 MB)."""
    from pipeline.benchmark import dfl, sources
    match_id = "J03WMX"
    info = dfl.parse_match_information(sources.local_path(match_id, "information"))
    match = real_match(match_id)
    report = match.report
    print("      {} in {:.1f} s: {:,} rows x {} players, {:,} ball frames, {:,} player frames "
          "({} dropped outside the halves)".format(report["cache"], LOAD_SECONDS[match_id], match.n_frames,
                                                    match.n_players, report["ball_frames"], report["player_frames"],
                                                    report["player_frames_outside_halves"]))
    print("      skipped FrameSets {}; missing attributes {}".format(report["skipped_framesets"], report["missing"]))
    assert set(report["skipped_framesets"]) == {"referee"}, \
        "{}: FrameSets of {} were skipped; a player who is not in the match information would be missing".format(
            match_id, sorted(set(report["skipped_framesets"]) - {"referee"}))

    expected = sources.EXPECTED_BALL_FRAMES[match_id]
    assert report["ball_frames"] == expected, (report["ball_frames"], expected)
    (_, start1, stop1), (_, start2, stop2) = match.graded
    starts = (match.frame_number[start1], match.frame_number[start2])
    assert starts == (10000, 100000), "the halves start at N {} and {}, not 10000 and 100000".format(*starts)
    assert (start1, stop1 + dfl.GAP_ROWS, stop2) == (0, start2, match.n_frames), (match.graded, match.n_frames)
    print("      rows without a ball frame: {}".format(report["rows_without_ball"]))
    assert report["rows_without_ball"] == 0, report["rows_without_ball"]

    in_play = match.frame_number >= 0
    present = ~np.isnan(match.players_xy[:, :, 0])
    per_team = [present[:, match.player_team == team].sum(axis=1) for team in (0, 1)]
    over = float(((per_team[0] > 11) | (per_team[1] > 11))[in_play].mean())
    print("      rows with more than 11 players of a team: {:.5f} (home max {}, away max {})".format(
        over, per_team[0].max(), per_team[1].max()))
    assert over < 0.001, over

    length, width = match.pitch
    x_metres = (match.players_xy[:, :, 0][present] - 0.5) * length
    inside = float((np.abs(x_metres) <= length / 2 + 5).mean())
    print("      player frames no more than 5 m beyond a goal line: {:.5f}".format(inside))
    assert inside >= 0.999, inside

    # S is in km/h: against the distance a player really covers between frames, S / 3.6 is the same speed.
    metres = match.players_xy * np.array([length, width])
    step = np.hypot(*np.moveaxis(np.diff(metres, axis=0), -1, 0)) * match.fps  # frame r - 1 to r, in m/s
    stated = match.players_speed[1:]
    fast = np.isfinite(step) & np.isfinite(stated) & (step > 2.0)
    ratio = float(np.median(stated[fast] / step[fast]))
    print("      median (S / 3.6) / finite-difference speed, {:,} frames above 2 m/s: {:.3f}".format(
        int(fast.sum()), ratio))
    assert 0.9 <= ratio <= 1.1, ratio

    # TotalTime* is in milliseconds or in seconds: whichever reading makes the rows add up is the one that is right.
    def agrees(divisor):
        return all(abs((stop - start) / match.fps - info.total_time_raw[half] / divisor)
                   <= 0.02 * info.total_time_raw[half] / divisor for half, start, stop in match.graded)

    matched = [name for name, divisor in (("seconds", 1.0), ("ms/1000", 1000.0)) if agrees(divisor)]
    print("      rows / 25 per half: {}; TotalTime read as {}".format(
        " and ".join("{:,.1f} s".format((stop - start) / match.fps) for _, start, stop in match.graded),
        " and ".join(matched) or "neither"))
    assert matched, "rows / 25 is within 2% of TotalTime{First,Second}Half neither as seconds nor as ms / 1000"


# -- cross-checks of the parser on J03WMX: kloppy, and a second pass keyed by frame number (--dfl) ---------------------

def differing(ours, theirs):
    """How many entries differ by TOLERANCE_M or more; a NaN matches only a NaN."""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    nan_ours, nan_theirs = np.isnan(ours), np.isnan(theirs)
    both = ~nan_ours & ~nan_theirs
    return int((nan_ours != nan_theirs).sum() + (np.abs(ours - theirs)[both] >= TOLERANCE_M).sum())


def kloppy_load(match_id, only_alive):
    """kloppy's dataset of the first KLOPPY_LIMIT frames of a downloaded match, as the T30 advice would load it."""
    from kloppy import sportec
    from pipeline.benchmark import sources
    return sportec.load_tracking(
        meta_data=sources.local_path(match_id, "information"), raw_data=sources.local_path(match_id, "positions"),
        coordinates="sportec", only_alive=only_alive, limit=KLOPPY_LIMIT)


def kloppy_rows(dataset):
    """(frame numbers, timeline rows) of a dataset that covers the first half only, as the check must find it."""
    numbers = np.array([record.frame_id for record in dataset.records], dtype=np.int64)
    periods = {record.period.id for record in dataset.records}
    assert periods == {1}, "kloppy's limit is per object, so it should stop inside the first half; it read {}".format(
        sorted(periods))
    return numbers, numbers - HALF_START_N[1]


def kloppy_differences(dataset, match, info):
    """({what: how many values were compared}, {what: how many disagree}) for kloppy's frames against the timeline.

    The pitch is compared first, kloppy's length and width against ours. The row of kloppy's frame N is
    N - HALF_START_N[1]; the timeline's own frame numbers must agree with that. The ball
    is compared in metres (ours converted back from 0-1), ALIVE against status 1, and "the home team has the ball"
    against possession 0. Three players are compared by PersonId: the two lowest ids among the home players and the
    lowest among the away players of kloppy's first frame. Their positions must agree wherever both readers have one,
    and presence must agree too: our NaN is the player missing from kloppy's frame. Kloppy's own team for each player
    of its first frame must be ours, and the number of players in every frame must be the same.
    """
    from kloppy.domain import BallState, Ground
    teams = {team.ground: team for team in dataset.metadata.teams}
    assert (teams[Ground.HOME].team_id, teams[Ground.AWAY].team_id) == (info.home_id, info.away_id), \
        "kloppy and the parser disagree about which team is home: {} / {}".format(
            [team.team_id for team in dataset.metadata.teams], (info.home_id, info.away_id))
    numbers, rows = kloppy_rows(dataset)
    assert (match.frame_number[rows] == numbers).all(), "the timeline's frame numbers are not N - {}".format(
        HALF_START_N[1])
    records = dataset.records
    length, width = match.pitch

    ball = np.array([(r.ball_coordinates.x, r.ball_coordinates.y, r.ball_coordinates.z) for r in records])
    alive = np.array([r.ball_state == BallState.ALIVE for r in records])
    home_has_ball = np.array([r.ball_owning_team.team_id == teams[Ground.HOME].team_id for r in records])
    differences = {
        "pitch": kloppy_pitch_differences(dataset, match),
        "ball x": differing(match.ball_xy[rows, 0] * length - length / 2, ball[:, 0]),
        "ball y": differing(width / 2 - match.ball_xy[rows, 1] * width, ball[:, 1]),
        "ball z": differing(match.ball_z[rows], ball[:, 2]),
        "alive": int((alive != (match.ball_status[rows] == 1)).sum()),
        "home has the ball": int((home_has_ball != (match.ball_possession[rows] == 0)).sum())}
    compared = {"pitch": 2, "ball x": len(records), "ball y": len(records), "ball z": len(records),
                "alive": len(records), "home has the ball": len(records), "alive frames": int(alive.sum()),
                "home frames": int(home_has_ball.sum())}

    team_of = {player.person_id: player.team for player in info.players}
    kloppy_team = {player.player_id: 0 if player.team.ground == Ground.HOME else 1
                   for player in records[0].players_data}
    differences["player team"] = sum(team_of.get(person) != team for person, team in kloppy_team.items())
    compared["player team"] = len(kloppy_team)
    chosen = ([person for person, team in sorted(kloppy_team.items()) if team == 0][:2]
              + [person for person, team in sorted(kloppy_team.items()) if team == 1][:1])
    assert len(chosen) == 3 and {kloppy_team[person] for person in chosen} == {0, 1}, chosen
    column = {player.person_id: index for index, player in enumerate(info.players)}
    frames = [{player.player_id: data.coordinates for player, data in record.players_data.items()}
              for record in records]
    differences.update({"player presence": 0, "player x/y": 0})
    compared.update({"player presence": 0, "player x/y": 0})
    for person in chosen:
        theirs_present = np.array([person in frame for frame in frames])
        theirs = np.array([(frame[person].x, frame[person].y) if person in frame else (np.nan, np.nan)
                           for frame in frames])
        xy = match.players_xy[rows, column[person]]
        ours = np.column_stack([xy[:, 0] * length - length / 2, width / 2 - xy[:, 1] * width])
        ours_present = ~np.isnan(xy[:, 0])
        both = theirs_present & ours_present
        differences["player presence"] += int((theirs_present != ours_present).sum())
        differences["player x/y"] += differing(ours[both], theirs[both])
        compared["player presence"] += len(records)
        compared["player x/y"] += 2 * int(both.sum())
    ours_count = (~np.isnan(match.players_xy[rows, :, 0])).sum(axis=1)
    differences["players per frame"] = int((ours_count != np.array([len(frame) for frame in frames])).sum())
    compared["players per frame"] = len(records)
    return compared, differences


def kloppy_default_drops(dataset, match):
    """(returned that are dead in ours, alive rows of ours that kloppy left out, dead rows it dropped, N range, rows).

    The range is that of the frame numbers kloppy returned: with only_alive=True its limit counts alive frames, so it
    reaches further than the limit says. Inside the range, every alive row of ours must be among the frames returned
    and none of the frames returned may be dead.
    """
    numbers, rows = kloppy_rows(dataset)
    span = np.arange(numbers.min(), numbers.max() + 1) - HALF_START_N[1]
    returned = np.zeros(span.size, dtype=bool)
    returned[rows - span[0]] = True
    status = match.ball_status[span]
    return (int((returned & (status != 1)).sum()), int(((status == 1) & ~returned).sum()),
            int(((status == 0) & ~returned).sum()), (int(numbers.min()), int(numbers.max())), int(span.size))


def require_kloppy():
    """kloppy's sportec module; NotRun if kloppy is not installed, and an ImportError for anything else.

    Only a ModuleNotFoundError that names kloppy itself means "not installed". Any other import problem (a broken
    install, a dependency of kloppy that is missing, a version without sportec) is a failure of the check.
    """
    try:
        from kloppy import sportec
    except ModuleNotFoundError as error:
        if error.name != "kloppy":
            raise
        raise NotRun("kloppy is not installed (pip install -r requirements-starter.txt gives the kloppy==3.18.0 that "
                     "T30 was measured with): the parser was not compared with it") from None
    return sportec


def kloppy_pitch_differences(dataset, match):
    """How many of the pitch's length and width kloppy reads differently from the timeline's (0, 1 or 2).

    kloppy takes PitchX and PitchY from the match information file itself, so this is a reading of them that is
    independent of ours. The metre round trip of the coordinates cannot see a wrong pitch: a pitch read wrong in the
    parser would be wrong in both directions of that trip.
    """
    dimensions = dataset.metadata.pitch_dimensions
    return sum(theirs != ours for theirs, ours in zip((dimensions.pitch_length, dimensions.pitch_width), match.pitch))


@contextlib.contextmanager
def kloppy_as_if(folder=None):
    """Inside the block `import kloppy` finds nothing (no folder), or the package that `folder` holds instead."""
    def kloppy_modules():
        return [name for name in sys.modules if name == "kloppy" or name.startswith("kloppy.")]

    saved = {name: sys.modules.pop(name) for name in kloppy_modules()}
    if folder is None:
        sys.modules["kloppy"] = None  # what Python itself puts there for a module that must not be imported
    else:
        sys.path.insert(0, folder)
        importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in kloppy_modules():
            del sys.modules[name]
        if folder is not None:
            sys.path.remove(folder)
        sys.modules.update(saved)


def test_the_kloppy_check_is_not_run_only_when_kloppy_is_not_installed():
    """The comparison with kloppy is optional, so an absent kloppy is NOT RUN; any other import problem is a failure.

    A broken install, a missing dependency of kloppy and a kloppy without its sportec module all raise an ImportError,
    and none of them may let the comparison silently not happen: a check that passes without having compared anything
    is the worst outcome.
    """
    with kloppy_as_if():
        try:
            require_kloppy()
        except NotRun as stop:
            assert "kloppy is not installed" in stop.reason and "kloppy==3.18.0" in stop.reason \
                and "requirements-starter.txt" in stop.reason, stop.reason
        else:
            raise AssertionError("an absent kloppy must raise NotRun")

    broken = [("a broken install", 'raise ImportError("DLL load failed while importing the native library")',
               ImportError, "DLL load failed"),
              ("a dependency of kloppy that is missing", "import a_package_that_is_not_installed",
               ModuleNotFoundError, "a_package_that_is_not_installed"),
              ("a kloppy without sportec", "", ImportError, "cannot import name 'sportec'")]
    for label, source, error_type, fragment in broken:
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "kloppy"))
            write_text(os.path.join(folder, "kloppy"), "__init__.py", source)
            with kloppy_as_if(folder):
                try:
                    require_kloppy()
                except NotRun:
                    raise AssertionError("{} must fail the check, not read as NOT RUN".format(label)) from None
                except ImportError as error:
                    assert type(error) is error_type and fragment in str(error), (label, type(error), str(error))
                else:
                    raise AssertionError("{} went through require_kloppy".format(label))


def test_the_kloppy_check_compares_the_pitch():
    from pipeline.benchmark import dfl

    def dataset(length, width):
        dimensions = types.SimpleNamespace(pitch_length=length, pitch_width=width)
        return types.SimpleNamespace(metadata=types.SimpleNamespace(pitch_dimensions=dimensions))

    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
    assert match.pitch == (104.5, 67.5), match.pitch  # the synthetic pitch is not the usual 105 x 68
    for theirs, differences in (((104.5, 67.5), 0), ((105.0, 67.5), 1), ((104.5, 68.0), 1), ((105.0, 68.0), 2),
                                ((None, 67.5), 1), ((67.5, 104.5), 2)):
        assert kloppy_pitch_differences(dataset(*theirs), match) == differences, (theirs, differences)


def test_the_t30_measurement_has_every_figure_the_docs_quote():
    """t30_measured returns the figures of T30_FACTS (and no others); on the synthetic match they are known."""
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
    measured = t30_measured(match)
    assert set(measured) == set(T30_FACTS["J03WMX"]), (sorted(measured), sorted(T30_FACTS["J03WMX"]))
    assert "player_frames_outside_halves" in T30_FACTS["J03WMX"], "docs/traps.md T30 says none lies outside the halves"
    # The synthetic report: 17 ball frames, 1 row without one, X Y S Z missing 2 + 1 + 3 + 2, 1 BallPossession and 2
    # BallStatus missing; two dead ball frames (N 10007 and N 100007); 2 player frames dropped.
    assert measured == {"ball_frames": 17, "rows_without_ball": 1, "no_possession": 1, "no_status": 2, "no_value": 8,
                        "dead": 2, "player_frames_outside_halves": 2}, measured


def test_dfl_parser_agrees_with_kloppy():
    """On J03WMX's first 5,000 frames kloppy reads the same ball, alive flag, possession and players as we do.

    Then kloppy's default, only_alive=True, is run too, which makes T30's first claim a measurement: it returns no
    dead frame and leaves out no alive one inside the range it covers, so the dead frames are exactly what it drops.
    The counts that docs/traps.md T30 and docs/results.md A05 quote are asserted (KLOPPY_FACTS), not only printed. A
    kloppy that is not installed is NOT RUN, never a pass (require_kloppy).
    """
    require_kloppy()
    from pipeline.benchmark import dfl, sources
    match_id = "J03WMX"
    info = dfl.parse_match_information(sources.local_path(match_id, "information"))
    match = real_match(match_id)

    full = kloppy_load(match_id, only_alive=False)
    numbers, _ = kloppy_rows(full)
    assert (numbers == np.arange(HALF_START_N[1], HALF_START_N[1] + KLOPPY_LIMIT)).all(), \
        "limit={} should return the first {} frames of the first half, in order".format(KLOPPY_LIMIT, KLOPPY_LIMIT)
    compared, differences = kloppy_differences(full, match, info)
    print("      kloppy only_alive=False, limit={}: {:,} frames, N {}-{} (the first half only); {:,} alive, {:,} home "
          "possession".format(KLOPPY_LIMIT, len(numbers), numbers[0], numbers[-1], compared["alive frames"],
                              compared["home frames"]))
    for what in ("pitch", "ball x", "ball y", "ball z", "alive", "home has the ball", "player team",
                 "player presence", "player x/y", "players per frame"):
        print("        {:<18} {:>9,} compared, {} disagreements".format(what, compared[what], differences[what]))
    assert not any(differences.values()), "kloppy and the parser disagree: {}".format(
        {what: count for what, count in differences.items() if count})
    window = {"alive": compared["alive frames"], "home": compared["home frames"], "players": compared["player team"],
              "player_xy_values": compared["player x/y"]}
    quoted = {name: KLOPPY_FACTS[name] for name in window}
    assert window == quoted, "kloppy's first {} frames hold {} but docs/traps.md T30 and docs/results.md A05 quote " \
                             "{}".format(KLOPPY_LIMIT, window, quoted)

    default = kloppy_load(match_id, only_alive=True)
    returned_dead, left_out, dropped, (first, last), span = kloppy_default_drops(default, match)
    print("      kloppy default (only_alive=True), same limit: {:,} frames over N {}-{} ({:,} rows): {} returned that "
          "are dead in ours, {} alive rows left out, {:,} dead rows dropped".format(
              len(default.records), first, last, span, returned_dead, left_out, dropped))
    assert (returned_dead, left_out) == (0, 0), (returned_dead, left_out)
    assert dropped > 0 and len(default.records) + dropped == span, "the default should drop exactly the dead rows"
    reached = {"frames": len(default.records), "first": first, "last": last, "rows": span, "dropped": dropped}
    quoted = {name: KLOPPY_FACTS["default_" + name] for name in reached}
    assert reached == quoted, "kloppy's default read {} but docs/traps.md T30 and docs/results.md A05 quote " \
                              "{}".format(reached, quoted)


def read_ball_frames(path, wanted):
    """({(half, N): the attributes of the ball's <Frame>}, {half: (smallest N, largest N)}) over the ball's frames.

    `wanted` is {half: N values}: those frames are kept, as written. The range is that of every ball frame of the
    half, whether it was asked for or not, so that the caller can place a row without asking the parser how long the
    first half is.

    A pass of its own, on purpose: it calls nothing of dfl, so that a mistake the parser makes cannot be made here too.
    It looks only at the FrameSets of the ball (TeamId BALL).
    """
    from lxml import etree
    halves = {"firstHalf": 1, "secondHalf": 2}
    wanted = {half: set(numbers) for half, numbers in wanted.items()}
    found, ranges, half, is_ball = {}, {}, None, False
    for event, element in etree.iterparse(path, events=("start", "end"), tag=("FrameSet", "Frame"), huge_tree=True,
                                          resolve_entities=False, no_network=True):
        if element.tag == "FrameSet":
            if event == "start":
                half, is_ball = halves[element.get("GameSection")], element.get("TeamId") == "BALL"
        elif event == "end":
            if is_ball:
                n = int(element.get("N"))
                low, high = ranges.get(half, (n, n))
                ranges[half] = (min(low, n), max(high, n))
                if n in wanted[half]:
                    found[(half, n)] = dict(element.attrib)
            element.clear()
            while element.getprevious() is not None:  # free what is done: the file is 400 MB
                del element.getparent()[0]
    return found, ranges


def spot_differences(match, path, sample):
    """The (N, attribute) pairs where the timeline's ball differs from the ball frame the file has at N.

    `sample` is {half: N values}. The row of N is placed here and not by the parser: N - 10000 in the first half, and
    n1 + 25 + (N - 100000) in the second, n1 being the rows of the first half as this pass of the file counts them
    (the largest ball N less the smallest, plus one), and which `match.graded` must then agree with. Every ball field
    the timeline stores is compared: X, Y (in metres), Z, S (km/h as m/s), BallStatus and BallPossession (1 and 2 as 0
    and 1), and N itself. An N with no ball frame in the file must have the ball NaN and its codes -1.
    """
    found, ranges = read_ball_frames(path, sample)
    length, width = match.pitch
    n1 = ranges[1][1] - ranges[1][0] + 1
    differences = []
    if n1 != match.graded[0][2]:
        differences.append((None, "rows of the first half"))  # the parser and the file disagree about where the gap is
    for half, numbers in sorted(sample.items()):
        for n in numbers:
            row = n - HALF_START_N[1] if half == 1 else n1 + GAP_ROWS + (n - HALF_START_N[2])
            frame = found.get((half, n))  # None: the file has no ball frame at N

            def number(name):
                return math.nan if frame is None or name not in frame else float(frame[name])

            def code(name, codes):
                return -1 if frame is None or name not in frame else codes[frame[name]]

            expected = {"N": n, "X": number("X"), "Y": number("Y"), "Z": number("Z"), "S": number("S") / 3.6,
                        "BallStatus": code("BallStatus", {"0": 0, "1": 1}),
                        "BallPossession": code("BallPossession", {"1": 0, "2": 1})}
            actual = {"N": int(match.frame_number[row]), "X": match.ball_xy[row, 0] * length - length / 2,
                      "Y": width / 2 - match.ball_xy[row, 1] * width, "Z": match.ball_z[row],
                      "S": match.ball_speed[row], "BallStatus": int(match.ball_status[row]),
                      "BallPossession": int(match.ball_possession[row])}
            differences += [(n, name) for name in expected if differing(actual[name], expected[name])]
    return differences


def test_spot_check_reads_the_synthetic_match_and_can_fail():
    """The spot check finds no difference on the synthetic match, and one in each field that is damaged.

    The synthetic match has a frame with no ball and a missing value in every attribute, so those rules are exercised.
    The damage is what a bug in the parser would be: a field wrong, a half shifted by one row, a frame number off.
    """
    from pipeline.benchmark import dfl
    sample = {1: list(range(10000, 10010)), 2: list(range(100000, 100008))}

    def shifted_second_half(array):
        moved = array.copy()
        moved[35:] = np.roll(array[35:], 1, axis=0)
        return moved

    def first_half_one_row_too_long(match):
        """What a parser that counts one row too many in the first half would build, with `graded` agreeing with it.

        Everything after the first half moves down one row and the second half starts where `graded` says, so a check
        that took the length of the first half from `graded` would find every row of the second half where it looks.
        """
        at = match.graded[0][2]
        fills = {"frame_number": -1, "players_xy": np.nan, "players_speed": np.nan, "ball_xy": np.nan,
                 "ball_z": np.nan, "ball_speed": np.nan, "ball_status": -1, "ball_possession": -1}
        (_, _, stop1), (_, start2, stop2) = match.graded
        return dataclasses.replace(
            match, graded=((1, 0, stop1 + 1), (2, start2 + 1, stop2 + 1)), kickoffs={1: 0, 2: match.kickoffs[2] + 1},
            **{name: np.insert(getattr(match, name), at, fill, axis=0) for name, fill in fills.items()})

    damages = {  # field -> how to damage it
        "ball_xy": lambda array: array + 0.01, "ball_z": lambda array: array + 0.01,
        "ball_speed": lambda array: array + 0.01, "ball_status": lambda array: np.where(array == 1, 0, 1).astype("i1"),
        "ball_possession": lambda array: np.where(array == 1, 0, 1).astype("i1"),
        "frame_number": lambda array: array + 1}
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
        assert spot_differences(match, path, sample) == [], spot_differences(match, path, sample)
        for field, damage in damages.items():
            wrong = dataclasses.replace(match, **{field: damage(getattr(match, field))})
            assert spot_differences(wrong, path, sample), "damaging {} went unnoticed".format(field)
        late = dataclasses.replace(match, **{field: shifted_second_half(getattr(match, field)) for field in (
            "frame_number", "ball_xy", "ball_z", "ball_speed", "ball_status", "ball_possession")})
        assert spot_differences(late, path, sample), "a second half one row late went unnoticed"
        longer = first_half_one_row_too_long(match)
        assert spot_differences(longer, path, sample), \
            "a first half one row too long, with graded to match, went unnoticed: the length of the first half must " \
            "come from the file's own ball frames, not from the parser's graded"


def test_dfl_second_half_spot_check():
    """400 frames of J03WMX's ball, drawn from both halves, read by a second pass over the file, match the timeline.

    200 frame numbers come from the second half, which sits after the gap between the halves and so shows a wrong gap,
    and 200 from the first, which kloppy covers only in its first 5,000 frames. Each is looked up in the file by N.
    """
    from pipeline.benchmark import sources
    match_id = "J03WMX"
    match = real_match(match_id)
    sample = {}
    for half, seed in ((2, 0), (1, 1)):
        _, start, stop = match.graded[half - 1]
        sample[half] = sorted(int(n) for n in np.random.default_rng([seed]).choice(
            match.frame_number[start:stop], size=SPOT_FRAMES, replace=False))
    differences = spot_differences(match, sources.local_path(match_id, "positions"), sample)
    print("      {} frames per half (N {}-{} and {}-{}), 7 fields each (N, X, Y, Z, S, BallStatus, BallPossession): "
          "{} disagreements".format(SPOT_FRAMES, sample[1][0], sample[1][-1], sample[2][0], sample[2][-1],
                                    len(differences)))
    assert not differences, "the file and the timeline disagree at (N, attribute): {}".format(differences[:10])


def t30_measured(match):
    """What T30_FACTS holds for one match, measured on its parsed timeline (the report, and the dead ball frames)."""
    report = match.report
    return {"ball_frames": report["ball_frames"], "rows_without_ball": report["rows_without_ball"],
            "no_possession": report["missing"]["BallPossession"], "no_status": report["missing"]["BallStatus"],
            "no_value": sum(report["missing"][name] for name in ("X", "Y", "S", "Z")),
            "dead": int((match.ball_status == 0).sum()),
            "player_frames_outside_halves": report["player_frames_outside_halves"]}


def test_dfl_t30_facts():
    """The numbers behind docs/traps.md T30, on all 7 matches: how often kloppy's DFL loader would go wrong.

    kloppy turns a missing BallPossession into the away team, skips a frame that has no ball entry, and by default
    drops every dead-ball frame. The table counts each through our parser, which refuses a BallPossession that is not
    1 or 2 (so every match in it has 0 of those, the "other" column), and the counts are asserted exactly, as the docs
    quote them, with the player frames that lie outside both halves, which the parser drops (the "players out"
    column: the docs say that none does). Seconds with the cache, about 5 minutes for the one run that builds it.
    """
    from pipeline.benchmark import sources
    print("      {:<7} {:>11} {:>13} {:>13} {:>5} {:>9} {:>11} {:>16} {:>13}".format(
        "match", "ball frames", "rows w/o ball", "no possession", "other", "no status", "no x/y/s/z", "dead (share)",
        "players out"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        measured = t30_measured(match)
        print("      {:<7} {:>11,} {:>13,} {:>13,} {:>5} {:>9,} {:>11,} {:>8,} ({:.1%}) {:>13,}".format(
            match_id, measured["ball_frames"], measured["rows_without_ball"], measured["no_possession"], 0,
            measured["no_status"], measured["no_value"], measured["dead"], measured["dead"] / measured["ball_frames"],
            measured["player_frames_outside_halves"]))
        assert measured == T30_FACTS[match_id], "{}: measured {} but the docs say {}".format(
            match_id, measured, T30_FACTS[match_id])
