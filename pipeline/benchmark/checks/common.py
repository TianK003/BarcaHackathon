"""What several check modules share: the synthetic DFL match, the real match's memo, and the console set-up.

The synthetic match (write_synthetic_dfl and the squads, tables and helpers it is written from) is one fixture for
every check that needs a DFL file: the parser checks read it, and so will the cache's and every later stage's. It lives
here, in the real file format with made-up values, so that no check needs a download. real_match is how the --dfl
checks that share a downloaded match get it (dfl.load_match behind a one-match memo); a check that wants the cache's
own answer, built or reused, calls dfl.load_match itself. Import from here with absolute imports
(from pipeline.benchmark.checks.common import ...): a module of this package named like a production module would
otherwise shadow it.
"""

import functools
import os
import sys
import time
from xml.sax.saxutils import quoteattr

from pipeline.selftest import write_text

# Params
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
    from pipeline.benchmark.checks.calibration import synthetic_calibration_inputs
    with tempfile.TemporaryDirectory() as root:
        inputs, gt = synthetic_calibration_inputs(root)
        return calibration.measure(inputs, gt)


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
    from pipeline.benchmark import dfl
    started = time.perf_counter()
    match = dfl.load_match(match_id)
    LOAD_SECONDS[match_id] = time.perf_counter() - started
    return match
