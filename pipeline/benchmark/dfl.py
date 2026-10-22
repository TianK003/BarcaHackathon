"""The DFL/IDSSE open data, read by our own parsers: the match information and the positions now; truth and the cache
later.

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

One timeline per match
----------------------
The positions come as one <FrameSet> per person and half, each a list of <Frame N=...>, where N is DFL's own frame
counter (the first half starts near 10000, the second near 100000). parse_positions lays a match out as ONE timeline,
which is what every later stage reads (the cache, the truth, the degrader, the scorer): the first half, one row per N
of the ball's FrameSet, then GAP_ROWS separator rows, then the second half. The separator puts the two halves in one
array without running them into each other: the truth says X on it, so the scorer's valid runs are exactly the two
halves, as for BAR-ATH, where nothing is scored across half time. A row's place comes only from its N, never from the
order of the file, because that order is not documented: the ball may be listed first or last, and a player's second
half before his first.

What is kept and what is refused is spelled out in parse_positions. The principle is the one of the match
information: nothing is repaired or guessed, and anything that could silently turn home into away, or shift every
label by a frame, stops the parse with a message that names the file.

Coordinates and units
---------------------
DFL gives X and Y in metres from the centre of the pitch, S in km/h and Z in metres. A DflMatch holds x = (X + L/2) / L
and y = (W/2 - Y) / W in 0-1 (L, W = MatchInfo.pitch). The y axis is flipped, as the plan fixes it; nothing in the
benchmark depends on which way it points, only on every consumer reading this one. Speeds are in m/s (S / 3.6) and Z in
metres. A missing X, Y, S or Z is NaN and a missing BallPossession or BallStatus is -1; both are counted in the report.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np
from lxml import etree

# Params
MATCH_ID_PREFIX = "DFL-MAT-"             # General@MatchId reads "DFL-MAT-J03WMX"; we say "J03WMX" everywhere else
ROLE_TEAM = {"home": 0, "guest": 1}      # Team@Role -> Player.team; DFL's "guest" is our away
TOTAL_TIME_ATTRIBUTE = {1: "TotalTimeFirstHalf", 2: "TotalTimeSecondHalf"}  # half -> OtherGameInformation attribute
FETCH_ADVICE = "run python pipeline/benchmark/sources.py"
VERIFY_ADVICE = "the download is probably damaged, so {} --verify to check it".format(FETCH_ADVICE)
GAP_ROWS = 25                            # separator rows between the halves of a timeline (one second at 25 Hz)
SECTIONS = {"firstHalf": 1, "secondHalf": 2}  # FrameSet@GameSection -> half, in time order
FPS = 25                                 # the positions are sampled at 25 Hz (consecutive T differ by 40 ms)
BALL_TEAM_ID = "BALL"                    # FrameSet@TeamId of the ball's FrameSet
REFEREE_TEAM_ID = "referee"              # ... of the referees', which are not read
KMH_PER_MS = 3.6                         # Frame@S is in km/h; speeds are kept in m/s
POSSESSION_TEAM = {"1": 0, "2": 1}       # Frame@BallPossession -> team: 1 is the home team (0), 2 the guest (1)
BALL_STATUS = {"0": 0, "1": 1}           # Frame@BallStatus: 0 the ball is dead, 1 alive; kept as it is
ABSENT = -1                              # no code, no ball frame, and the separator's frame number
PLAYER_ATTRIBUTES = ("N", "X", "Y", "S")  # what is read from a player's <Frame>
BALL_ATTRIBUTES = ("N", "X", "Y", "Z", "S", "BallPossession", "BallStatus")  # ... and from the ball's
MISSING_ATTRIBUTES = ("X", "Y", "S", "Z", "BallPossession", "BallStatus")    # the report counts these when absent


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


@dataclass(frozen=True, eq=False)
class DflMatch:
    """A match as one timeline of n rows: the first half, GAP_ROWS separator rows, then the second half.

    Row r is DFL frame `frame_number[r]`, and consecutive rows of a half are consecutive frames, 1 / fps seconds
    apart. A separator row has frame_number -1 and nothing else: NaN positions, speeds and ball, codes -1. Inside a
    half, a frame that has no ball entry has the ball NaN and its codes -1 (counted in the report; the players'
    columns are as usual), and a player who is not on the pitch (not yet on, off, or never played) is NaN in his
    column.

    Columns of players_xy and players_speed follow MatchInfo.players (home squad first, each sorted by PersonId), so
    `player_team` is that list's team: 0 home, 1 away. Every squad member has a column, the bench included.

    `graded` is ((1, start, stop), (2, start, stop)), the rows of each half as a half-open range; `kickoffs` maps the
    half to its first row, which is what the plan treats as the kick-off (the first frame DFL gives for the half,
    not necessarily the whistle). x, y are normalised 0-1 (see the module docstring), speeds are m/s, ball_z is
    metres. ball_status is 1 alive, 0 dead, -1 unknown; ball_possession is 0 home, 1 away, -1 unknown. `report`
    counts what the parse saw (parse_positions).

    The arrays are read-only, so that a match read from the cache and one just parsed behave the same. The class
    compares by identity (eq=False): a field-wise == over arrays would be a question with no useful answer.
    """
    match_id: str
    pitch: tuple                                    # (length, width) in metres
    fps: int
    graded: tuple                                   # ((half, start row, stop row), ...)
    kickoffs: dict                                  # {half: row}
    frame_number: np.ndarray = field(repr=False)    # int64 (n,): DFL's N, -1 on the separator
    players_xy: np.ndarray = field(repr=False)      # float64 (n, P, 2), normalised
    players_speed: np.ndarray = field(repr=False)   # float64 (n, P), m/s
    player_team: np.ndarray = field(repr=False)     # int8 (P,)
    ball_xy: np.ndarray = field(repr=False)         # float64 (n, 2), normalised
    ball_z: np.ndarray = field(repr=False)          # float64 (n,), metres
    ball_speed: np.ndarray = field(repr=False)      # float64 (n,), m/s
    ball_status: np.ndarray = field(repr=False)     # int8 (n,): 1 alive, 0 dead, -1 unknown
    ball_possession: np.ndarray = field(repr=False)  # int8 (n,): 0 home, 1 away, -1 unknown
    report: dict                                    # parse counts, JSON-ready (parse_positions)

    @property
    def n_frames(self):
        return self.frame_number.shape[0]

    @property
    def n_players(self):
        return self.player_team.shape[0]


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


def parse_positions(path, info):
    """DflMatch from a DFL positions file (DFL_04_03_positions_raw_observed_*.xml) and its match information.

    The file is streamed (350-420 MB) and each kept FrameSet is buffered as arrays, so that the timeline is built
    after the whole file is read and nothing depends on the order of the FrameSets or of the frames in them.

    Rows. A half's rows are the range of N from the smallest to the largest N of its ball FrameSet, so an N that has
    no ball frame becomes a row with the ball NaN and its codes -1 (report: rows_without_ball). A player's frame
    outside that range is dropped (player_frames_outside_halves). The rows of a frame are never found by position:
    row = start of the half + (N - first N of the half).

    FrameSets. Each has TeamId, PersonId and GameSection, a GameSection that is in SECTIONS (checked on every FrameSet,
    those that are skipped too) and, if it has one, the MatchId of `info`. TeamId "BALL" is the ball: exactly one per
    half, in both halves. TeamId "referee", and a PersonId that is not in info.players, are skipped and counted by
    TeamId. A rostered player's TeamId must be the club that `info` gives his team (a mismatch could turn home into
    away). A second FrameSet for one (PersonId, GameSection), and a repeated N inside a FrameSet, are errors: which to
    keep is not for us to guess.

    Values. X, Y, S and Z that are absent, or read nan, are NaN, and a BallPossession or BallStatus that is absent is
    -1; each is counted. Anything else that is not a finite number is a ValueError. A BallPossession outside {1, 2},
    a BallStatus outside {0, 1} (text that is not exactly one of those, such as 1.5 or an empty string, included) are
    ValueErrors too, raised once after the scan with the count and the first N, and BallPossession 1 becomes team 0.

    report (JSON-ready: str keys, int values, or {str: int}; no timings):
        ball_frames                   ball frames stored, both halves
        rows_without_ball             rows inside a half that have no ball frame
        player_frames                 player frames stored (inside a half)
        player_frames_outside_halves  player frames dropped
        skipped_framesets             {TeamId: number of FrameSets skipped}
        missing                       {attribute: count} on stored frames: X, Y and S over the players' and the ball's
                                      frames, Z, BallPossession and BallStatus over the ball's; all six are listed

    ValueError, naming the file, for anything that is not a positions file of this match; FileNotFoundError if there
    is no file.
    """
    balls, players, skipped, bad_codes = _scan(path, info)
    _require_ball_per_half(path, balls)
    _reject_bad_codes(path, bad_codes)
    return _timeline(info, balls, players, skipped)


def _require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("{}: no such file; {} to download the DFL data".format(path, FETCH_ADVICE))


def _syntax_error(path, error):
    return ValueError("{}: not well-formed XML ({}); {}".format(path, error, VERIFY_ADVICE))


def _read(path):
    """The root element of the XML file; no entity is expanded and no network is touched."""
    _require_file(path)
    try:
        return etree.parse(path, etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except etree.XMLSyntaxError as error:
        raise _syntax_error(path, error) from error


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


# -- positions --------------------------------------------------------------------------------------------------------

class _Reading:
    """A FrameSet being read: the text of the attributes kept from each frame, collected in one list per attribute."""

    def __init__(self, kind, team_id, person_id, section, names):
        self.kind = kind  # "ball", "player", or "skipped" (nothing is kept)
        self.person_id = person_id
        self.half = SECTIONS[section]
        self.label = _label(team_id, person_id, section)
        self.texts = {name: [] for name in names}
        self.appenders = tuple((name, texts.append) for name, texts in self.texts.items())


class _Series:
    """The frames of one kept FrameSet as arrays, in file order: frame numbers `n` and the values at those frames.

    For the ball, `possession` and `status` are already codes (ABSENT where missing) and `bad` lists the frames whose
    code is not allowed, as (N, BallPossession text, BallStatus text).
    """

    def __init__(self, n, x, y, s, z=None, possession=None, status=None, bad=()):
        self.n, self.x, self.y, self.s = n, x, y, s
        self.z, self.possession, self.status, self.bad = z, possession, status, bad


def _label(team_id, person_id, section):
    """How a FrameSet is named in a message; an attribute it lacks reads ?."""
    return '<FrameSet TeamId="{}" PersonId="{}" GameSection="{}">'.format(
        *("?" if value is None else value for value in (team_id, person_id, section)))


def _forget(element):
    """Free a finished element and the siblings before it: what keeps a 400 MB file in a few MB of memory."""
    element.clear()
    parent = element.getparent()
    while element.getprevious() is not None:
        del parent[0]


def _scan(path, info):
    """Read the file once: ({half: ball}, {(PersonId, half): player}, {TeamId: FrameSets skipped}, bad ball codes).

    The bad codes are (half, N, BallPossession text, BallStatus text), left for one error after the scan.
    """
    _require_file(path)
    roster = {player.person_id: player for player in info.players}
    club_of = {0: info.home_id, 1: info.away_id}
    balls, players, skipped, bad_codes = {}, {}, {}, []
    current = None
    try:
        with open(path, "rb") as handle:
            events = etree.iterparse(handle, events=("start", "end"), tag=("FrameSet", "Frame"), huge_tree=True,
                                     resolve_entities=False, no_network=True)
            for event, element in events:
                if element.tag == "Frame":
                    if event == "end":
                        if current is None:
                            raise ValueError("{}: a <Frame> outside any <FrameSet>; this is not a DFL positions file "
                                             "(DFL_04_03_positions_raw_observed_*.xml)".format(path))
                        get = element.get
                        for name, append in current.appenders:
                            append(get(name))
                        _forget(element)
                elif event == "start":
                    current = _open(element.attrib, path, info, roster, club_of, balls, players, skipped)
                else:
                    if current.kind == "ball":
                        series = balls[current.half] = _series(current, path)
                        bad_codes += [(current.half, *frame) for frame in series.bad]
                    elif current.kind == "player":
                        players[(current.person_id, current.half)] = _series(current, path)
                    current = None
                    _forget(element)
    except etree.XMLSyntaxError as error:
        raise _syntax_error(path, error) from error
    return balls, players, skipped, bad_codes


def _open(attributes, path, info, roster, club_of, balls, players, skipped):
    """The _Reading for a FrameSet that starts: what it is, and whether the file may have it."""
    team_id, person_id, section = (attributes.get(name) for name in ("TeamId", "PersonId", "GameSection"))
    label = _label(team_id, person_id, section)
    for name, value in (("TeamId", team_id), ("PersonId", person_id), ("GameSection", section)):
        if not value:
            raise ValueError("{}: {} has no {} attribute; {}".format(path, label, name, VERIFY_ADVICE))
    if section not in SECTIONS:
        raise ValueError("{}: {} has a GameSection that is not {}; {}".format(
            path, label, " or ".join('"{}"'.format(name) for name in SECTIONS), VERIFY_ADVICE))
    stated = attributes.get("MatchId")
    if stated is not None and stated != MATCH_ID_PREFIX + info.match_id:
        raise ValueError("{}: {} has MatchId=\"{}\" but the match information is of {}; the positions file and the "
                         "information file are of different matches, so pass the two files of one match "
                         "(sources.local_path)".format(path, label, stated, MATCH_ID_PREFIX + info.match_id))

    half = SECTIONS[section]
    if team_id == BALL_TEAM_ID:
        if half in balls:
            raise ValueError("{}: a second ball FrameSet for {} ({}); a positions file has exactly one per half, and "
                             "which to keep is not for us to guess; {}".format(path, section, label, VERIFY_ADVICE))
        return _Reading("ball", team_id, person_id, section, BALL_ATTRIBUTES)
    player = roster.get(person_id)
    if team_id == REFEREE_TEAM_ID or player is None:
        skipped[team_id] = skipped.get(team_id, 0) + 1
        return _Reading("skipped", team_id, person_id, section, ())
    if team_id != club_of[player.team]:
        raise ValueError("{}: {} is a player of the {} team ({}) in the match information, but its TeamId is not that "
                         "club; a mismatch could swap home and away, so the file is refused; {}".format(
                             path, label, "home" if player.team == 0 else "away", club_of[player.team],
                             VERIFY_ADVICE))
    if (person_id, half) in players:
        raise ValueError("{}: {} appears twice; each person has one FrameSet per half, and which of the two to keep "
                         "is not for us to guess; {}".format(path, label, VERIFY_ADVICE))
    return _Reading("player", team_id, person_id, section, PLAYER_ATTRIBUTES)


def _series(reading, path):
    """The arrays of a FrameSet that has ended; ValueError for a frame number or value that cannot be used."""
    texts, label = reading.texts, reading.label
    n = _integers(texts["N"], path, label)
    if reading.kind == "ball" and n.size == 0:
        raise ValueError("{}: the ball FrameSet {} has no frames; {}".format(path, label, VERIFY_ADVICE))
    ordered = np.sort(n)
    repeated = ordered[1:][ordered[1:] == ordered[:-1]]
    if repeated.size:
        raise ValueError("{}: {} has N={} twice ({} frames carry it); a frame number is unique in a FrameSet and "
                         "which frame to keep is not for us to guess; {}".format(
                             path, label, repeated[0], int((n == repeated[0]).sum()), VERIFY_ADVICE))
    values = {name: _decimals(texts[name], path, label, name) for name in texts
              if name not in ("N", "BallPossession", "BallStatus")}
    if reading.kind == "player":
        return _Series(n, values["X"], values["Y"], values["S"])
    possession, bad_possession = _codes(texts["BallPossession"], POSSESSION_TEAM)
    status, bad_status = _codes(texts["BallStatus"], BALL_STATUS)
    bad = [(int(n[i]), texts["BallPossession"][i], texts["BallStatus"][i])
           for i in np.flatnonzero(bad_possession | bad_status)]
    return _Series(n, values["X"], values["Y"], values["S"], values["Z"], possession, status, bad)


def _integers(texts, path, label):
    """The frame numbers as int64; a frame with none, or with one that is no whole number, is a ValueError."""
    try:
        return np.array(texts, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        for text in texts:  # numpy does not say which frame it refused
            try:
                np.int64(int(text))
            except TypeError:
                raise ValueError("{}: {} has a <Frame> with no N attribute; {}".format(
                    path, label, VERIFY_ADVICE)) from None
            except (ValueError, OverflowError):
                raise ValueError("{}: {} has a <Frame> with N={!r}, which is not a whole number; {}".format(
                    path, label, text, VERIFY_ADVICE)) from None
        raise


def _decimals(texts, path, label, name):
    """The attribute as float64, NaN where it is absent (or reads nan); anything else that is not finite is refused."""
    try:
        values = np.array(texts, dtype=np.float64)
    except ValueError:
        for text in texts:
            try:
                float(text)
            except TypeError:
                pass  # absent: NaN
            except ValueError:
                raise _not_finite(path, label, name, text) from None
        raise
    infinite = np.isinf(values)
    if infinite.any():
        raise _not_finite(path, label, name, texts[int(np.argmax(infinite))])
    return values


def _not_finite(path, label, name, text):
    return ValueError("{}: {} has a <Frame> with {}={!r}, which is not a finite number; {}".format(
        path, label, name, text, VERIFY_ADVICE))


def _codes(texts, allowed):
    """(codes, bad): each text's code under `allowed`, ABSENT where the text is absent; `bad` marks any other text."""
    codes = np.full(len(texts), ABSENT, dtype=np.int8)
    bad = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        if text is not None:
            code = allowed.get(text)
            if code is None:
                bad[i] = True
            else:
                codes[i] = code
    return codes, bad


def _require_ball_per_half(path, balls):
    for section, half in SECTIONS.items():
        if half not in balls:
            raise ValueError("{}: no ball FrameSet (TeamId=\"{}\") for GameSection=\"{}\"; a DFL positions file "
                             "(DFL_04_03_positions_raw_observed_*.xml) has exactly one per half; {}".format(
                                 path, BALL_TEAM_ID, section, VERIFY_ADVICE))


def _reject_bad_codes(path, bad_codes):
    """One ValueError for all the ball frames whose codes are not allowed, naming how many and the first."""
    if bad_codes:
        half, n, possession, status = min(bad_codes, key=lambda frame: frame[:2])
        section = next(name for name, number in SECTIONS.items() if number == half)
        raise ValueError("{}: {} ball frame(s) have a BallPossession that is not 1 or 2, or a BallStatus that is not "
                         "0 or 1; the first is N={} in {}, with BallPossession={!r} and BallStatus={!r}. The codes "
                         "decide what the benchmark's labels mean, so none is repaired; {}".format(
                             path, len(bad_codes), n, section, possession, status, VERIFY_ADVICE))


def _normalised(x, y, length, width):
    """DFL's metres from the centre as 0-1 across the pitch; y is flipped."""
    return (x + length / 2) / length, (width / 2 - y) / width


def _timeline(info, balls, players, skipped):
    """The DflMatch for the FrameSets that were read: first half, GAP_ROWS separator rows, second half."""
    length, width = info.pitch
    layout = {}  # half -> (first row, first N, rows); the rows of a half are the range of N of its ball
    row = 0
    for half in SECTIONS.values():
        first = int(balls[half].n.min())
        rows = int(balls[half].n.max()) - first + 1
        layout[half] = (row, first, rows)
        row += rows + GAP_ROWS
    n_rows = row - GAP_ROWS
    columns = {player.person_id: column for column, player in enumerate(info.players)}

    frame_number = np.full(n_rows, ABSENT, dtype=np.int64)
    players_xy = np.full((n_rows, len(info.players), 2), np.nan)
    players_speed = np.full((n_rows, len(info.players)), np.nan)
    ball_xy = np.full((n_rows, 2), np.nan)
    ball_z = np.full(n_rows, np.nan)
    ball_speed = np.full(n_rows, np.nan)
    ball_status = np.full(n_rows, ABSENT, dtype=np.int8)
    ball_possession = np.full(n_rows, ABSENT, dtype=np.int8)
    missing = dict.fromkeys(MISSING_ATTRIBUTES, 0)
    counts = dict.fromkeys(("ball_frames", "rows_without_ball", "player_frames", "player_frames_outside_halves"), 0)

    for half, (start, first, rows) in layout.items():
        frame_number[start:start + rows] = np.arange(first, first + rows, dtype=np.int64)
        ball = balls[half]
        at = start + (ball.n - first)
        ball_xy[at, 0], ball_xy[at, 1] = _normalised(ball.x, ball.y, length, width)
        ball_z[at] = ball.z
        ball_speed[at] = ball.s / KMH_PER_MS
        ball_possession[at] = ball.possession
        ball_status[at] = ball.status
        counts["ball_frames"] += ball.n.size
        counts["rows_without_ball"] += rows - ball.n.size
        for name, values in (("X", ball.x), ("Y", ball.y), ("S", ball.s), ("Z", ball.z)):
            missing[name] += int(np.isnan(values).sum())
        missing["BallPossession"] += int((ball.possession == ABSENT).sum())
        missing["BallStatus"] += int((ball.status == ABSENT).sum())

    for (person_id, half), series in players.items():
        start, first, rows = layout[half]
        inside = (series.n >= first) & (series.n < first + rows)
        kept = int(inside.sum())
        counts["player_frames"] += kept
        counts["player_frames_outside_halves"] += series.n.size - kept
        at = start + (series.n[inside] - first)
        x, y, s = series.x[inside], series.y[inside], series.s[inside]
        column = columns[person_id]
        players_xy[at, column, 0], players_xy[at, column, 1] = _normalised(x, y, length, width)
        players_speed[at, column] = s / KMH_PER_MS
        for name, values in (("X", x), ("Y", y), ("S", s)):
            missing[name] += int(np.isnan(values).sum())

    player_team = np.array([player.team for player in info.players], dtype=np.int8)
    for array in (frame_number, players_xy, players_speed, player_team, ball_xy, ball_z, ball_speed, ball_status,
                  ball_possession):
        array.setflags(write=False)
    return DflMatch(
        match_id=info.match_id, pitch=info.pitch, fps=FPS,
        graded=tuple((half, start, start + rows) for half, (start, _, rows) in layout.items()),
        kickoffs={half: start for half, (start, _, _) in layout.items()},
        frame_number=frame_number, players_xy=players_xy, players_speed=players_speed, player_team=player_team,
        ball_xy=ball_xy, ball_z=ball_z, ball_speed=ball_speed, ball_status=ball_status,
        ball_possession=ball_possession,
        report={**counts, "skipped_framesets": dict(sorted(skipped.items())), "missing": missing})
