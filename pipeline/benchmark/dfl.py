"""The DFL/IDSSE open data, read by our own parsers: the match information now; positions, truth and the cache later.

What the data are
-----------------
Seven Bundesliga and 2. Bundesliga matches (CC BY 4.0, downloaded by sources.py into data/external/dfl/), each as
three XML files: the match information (teams, squads, pitch size, the length of each half), the raw events, and the
observed positions of every player, the referees and the ball at 25 Hz, with the ball's own alive/dead status and the
team in possession. They stand in for a ground truth that we are not allowed to make for BAR-ATH (docs/proposals.md,
P01).

Why we parse them ourselves
---------------------------
kloppy (3.18.0, pinned) has a loader for this data, and it is wrong for us in three ways that nothing would announce:
it maps every BallPossession other than 1 to the away team, so a frame without a possession value turns into an
away frame; it skips frames that have no ball entry; and by default it drops every dead-ball frame, which leaves no
D to learn or to score (docs/traps.md T30, found by reading its source). Reading the XML ourselves lets each of
those be a decision we make and test, with the counts printed, instead of a default we inherit.

Home and away
-------------
DFL calls the away team "guest". The file lists the two <Team> elements in either order (the guest first in four of
the seven), so the team's Role attribute decides which is which, never its position; the file's own General
attributes are only used to check that they agree. Throughout the benchmark team 0 is home and team 1 is away, as in
the pipeline (Atd.slot_team).

Coordinates
-----------
Filled in with the positions parser. This module supplies the pitch size in metres, MatchInfo.pitch.
"""

import math
import os
from dataclasses import dataclass

from lxml import etree

# Params
MATCH_ID_PREFIX = "DFL-MAT-"             # General@MatchId reads "DFL-MAT-J03WMX"; we say "J03WMX" everywhere else
ROLE_TEAM = {"home": 0, "guest": 1}      # Team@Role -> Player.team; DFL's "guest" is our away
TOTAL_TIME_ATTRIBUTE = {1: "TotalTimeFirstHalf", 2: "TotalTimeSecondHalf"}  # half -> OtherGameInformation attribute
FETCH_ADVICE = "run python pipeline/benchmark/sources.py"
VERIFY_ADVICE = "the download is probably damaged, so {} --verify to check it".format(FETCH_ADVICE)


@dataclass(frozen=True)
class Player:
    """One player of the squad (substitutes included), as the match information lists him.

    `person_id` is the DFL id that the positions and the events use ("DFL-OBJ-0002DR"); team is 0 for home and 1 for
    away; `name` is FirstName and LastName joined by a space (FirstName holds every given name in the file), or the
    Shortname when there is no full name, or the person_id when there is nothing: it is for people to read, never
    for matching.
    """
    person_id: str
    team: int
    shirt: int
    name: str


@dataclass(frozen=True)
class MatchInfo:
    """What a match information file says, with home and away decided by Role.

    `players` lists the home squad first, then the away squad, each sorted by person_id, so that the order never
    depends on the file. `pitch` is (length, width) in metres. `total_time_raw` maps 1 and 2 to
    OtherGameInformation@TotalTimeFirstHalf and @TotalTimeSecondHalf as floats of exactly the numbers written: their
    unit is not interpreted here (the positions parser settles it against the frame numbers).
    """
    match_id: str          # the short id, "J03WMX"
    home_id: str           # DFL team ids, "DFL-CLU-000008"
    away_id: str
    home_name: str
    away_name: str
    pitch: tuple           # (PitchX, PitchY) in metres
    players: tuple         # of Player
    total_time_raw: dict   # {1: TotalTimeFirstHalf, 2: TotalTimeSecondHalf}


def parse_match_information(path):
    """MatchInfo from a DFL match information file (DFL_02_01_matchinformation_*.xml).

    ValueError, naming the file, for anything that is not a well-formed match information file with two teams (one
    "home", one "guest"), a pitch, squads with numeric shirts and distinct PersonIds, and both half durations.
    FileNotFoundError if there is no file. Nothing is repaired or guessed: the benchmark's labels hang on which team
    is home, so a file that does not say so clearly is refused.
    """
    root = _read(path)
    info = _child(root, path, "MatchInformation")
    general = _child(info, path, "General")
    environment = _child(info, path, "Environment")
    other = _child(info, path, "OtherGameInformation")

    match_id = _attribute(general, path, "MatchId")
    if not match_id.startswith(MATCH_ID_PREFIX) or len(match_id) == len(MATCH_ID_PREFIX):
        raise ValueError("{}: General@MatchId is {!r}, which does not start with {!r} and carry an id after it; is "
                         "this a DFL match information file?".format(path, match_id, MATCH_ID_PREFIX))
    pitch = tuple(_number(environment, path, name) for name in ("PitchX", "PitchY"))
    if min(pitch) <= 0:
        raise ValueError("{}: Environment has PitchX={} and PitchY={}; both must be positive (metres)".format(
            path, *(environment.get(name) for name in ("PitchX", "PitchY"))))
    total_time_raw = {half: _number(other, path, attribute) for half, attribute in TOTAL_TIME_ATTRIBUTE.items()}

    teams = _teams_by_role(_child(info, path, "Teams"), path)
    team_ids = {role: _attribute(team, path, "TeamId") for role, team in teams.items()}
    team_names = {role: _attribute(team, path, "TeamName") for role, team in teams.items()}
    for role, general_attribute in (("home", "HomeTeamId"), ("guest", "GuestTeamId")):
        stated = general.get(general_attribute)
        if stated is not None and stated != team_ids[role]:
            raise ValueError("{}: General@{} is {!r} but the <Team> with Role=\"{}\" has TeamId {!r}; the file "
                             "contradicts itself about who is home, so it cannot be trusted".format(
                                 path, general_attribute, stated, role, team_ids[role]))

    players = []
    role_of_person = {}
    for role, team in ROLE_TEAM.items():
        squad = [_player(element, path, team) for element in teams[role].findall("Players/Player")]
        if not squad:
            raise ValueError("{}: the <Team> with Role=\"{}\" ({}) lists no <Player> under <Players>; both "
                             "squads are expected; {}".format(path, role, team_names[role], VERIFY_ADVICE))
        for player in squad:
            if player.person_id in role_of_person:
                raise ValueError("{}: PersonId {} is listed twice (under Role=\"{}\", then under Role=\"{}\"); the "
                                 "positions are keyed by PersonId, so it must be unique".format(
                                     path, player.person_id, role_of_person[player.person_id], role))
            role_of_person[player.person_id] = role
        players += sorted(squad, key=lambda player: player.person_id)

    return MatchInfo(
        match_id=match_id[len(MATCH_ID_PREFIX):], home_id=team_ids["home"], away_id=team_ids["guest"],
        home_name=team_names["home"], away_name=team_names["guest"], pitch=pitch, players=tuple(players),
        total_time_raw=total_time_raw)


def _read(path):
    """The root element of the XML file; no entity is expanded and no network is touched."""
    if not os.path.isfile(path):
        raise FileNotFoundError("{}: no such file; {} to download the DFL data".format(path, FETCH_ADVICE))
    try:
        return etree.parse(path, etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except etree.XMLSyntaxError as error:
        raise ValueError("{}: not well-formed XML ({}); {}".format(path, error, VERIFY_ADVICE)) from error


def _child(parent, path, name):
    """The one child element called `name`, or a ValueError that says what the file has instead."""
    children = parent.findall(name)
    if not children:
        found = ", ".join("<{}>".format(element.tag) for element in parent if isinstance(element.tag, str))
        found = found or "nothing"  # a comment's tag is a function, not a name: left out
        raise ValueError("{}: <{}> has no <{}> element (it has {}); this is not a DFL match information file "
                         "(DFL_02_01_matchinformation_*.xml)".format(path, parent.tag, name, found))
    if len(children) > 1:
        raise ValueError("{}: <{}> has {} <{}> elements and exactly one is expected; this is not a DFL match "
                         "information file (DFL_02_01_matchinformation_*.xml)".format(
                             path, parent.tag, len(children), name))
    return children[0]


def _attribute(element, path, name):
    value = element.get(name)
    if not value:
        raise ValueError("{}: <{}> has no {} attribute; {}".format(path, element.tag, name, VERIFY_ADVICE))
    return value


def _number(element, path, name):
    """The attribute as a float, exactly as written; it must be a finite number."""
    text = _attribute(element, path, name)
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError("{}: <{}> has {}={!r}, which is not a number; {}".format(
            path, element.tag, name, text, VERIFY_ADVICE))
    return value


def _teams_by_role(teams_element, path):
    """{"home": <Team>, "guest": <Team>}; a Role that is neither, or that appears twice, is a ValueError."""
    found = {}
    for team in teams_element.findall("Team"):
        role = team.get("Role")
        if role not in ROLE_TEAM:
            raise ValueError("{}: <Team> {!r} has Role={!r}; Role must be \"home\" or \"guest\", it is what says "
                             "which team is home".format(path, team.get("TeamName"), role))
        if role in found:
            raise ValueError("{}: Role=\"{}\" is on two <Team> elements; exactly one \"home\" and one \"guest\" "
                             "are expected".format(path, role))
        found[role] = team
    missing = ["Role=\"{}\"".format(role) for role in ROLE_TEAM if role not in found]
    if missing:
        raise ValueError("{}: no <Team> with {}; exactly one \"home\" and one \"guest\" are expected".format(
            path, " or ".join(missing)))
    return found


def _player(element, path, team):
    person_id = _attribute(element, path, "PersonId")
    shirt = element.get("ShirtNumber")
    if shirt is None:
        raise ValueError("{}: <Player> {} has no ShirtNumber attribute; {}".format(path, person_id, VERIFY_ADVICE))
    try:
        shirt = int(shirt)
    except ValueError:
        raise ValueError("{}: <Player> {} has ShirtNumber={!r}, which is not a whole number; {}".format(
            path, person_id, shirt, VERIFY_ADVICE)) from None
    full_name = " ".join(part for part in (element.get("FirstName", "").strip(), element.get("LastName", "").strip())
                         if part)
    return Player(person_id=person_id, team=team, shirt=shirt,
                  name=full_name or element.get("Shortname", "").strip() or person_id)
