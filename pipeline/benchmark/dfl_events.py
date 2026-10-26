"""The restarts that DFL's event files name, on the benchmark's timeline (P17's diagnostics; it chooses nothing).

What it is for
--------------
Each DFL match comes with a raw event file (DFL_03_02_events_raw_*.xml) that names every restart: KickOff, ThrowIn,
CornerKick, GoalKick, FreeKick and Penalty, each with the team that takes it. P17's restart detector is label-free, so
its calls can be set against these: per kind of restart, how often it fires, and what kind it says. For that the
restarts have to stand on the rows of dfl.py's timeline, which is what load_restarts gives. Nothing in here is read by
the pipeline, by a decoder or by a choice of a constant.

How an event time becomes a row
-------------------------------
An event carries a wall-clock time (EventTime, with a time zone: "2023-05-27T15:30:12.230+02:00"), and so does every
positions <Frame> (T, in UTC: "2023-05-27T13:30:12.600+00:00"). dfl.py lays a half out as one row per N of the ball's
FrameSet, N = DFL's frame counter, row = start of the half + (N - first N), and does not read T. The faithful mapping
is therefore through the frame clock: the row of a time t in half h is

    start_h + round((t - T_h) / 40 ms),   T_h = the T of the half's first ball frame (the one on row start_h).

This is exact, not an estimate, as long as the frame clock is regular, so scan_ball_times reads the T of the first and
the last ball frame of each half and refuses the file unless T_last - T_first is exactly 40 ms times N_last - N_first
(a dropped or doubled frame, or a clock that drifts, would shift every later restart without a word). The alternative,
anchoring each half on its KickOff event, was rejected: the kick-off's event time and the first frame DFL gives are not
the same instant (the first event of J03WMX is 0.37 s before its first frame, so it would sit on a row before the
half), and which of them is the better anchor is exactly what the frame clock settles. T is the one clock that the
positions and the events share, so it is the one used; round, not floor, because an event is stamped to the
millisecond and a frame stands for the 40 ms around it.

A time is given to a half by which side of the break it falls on (the midpoint between the first half's last frame and
the second half's first), and kept only if its row is inside that half's graded rows [start, stop). A restart that
falls outside (a kick-off stamped before the first frame, one in the break) is dropped and counted in "outside": a
row that is not graded can be neither right nor wrong, and the timeline has nothing there to be set against.

Which team and which restarts
-----------------------------
The six kinds are the child element of <Event>. ThrowIn, CornerKick, GoalKick, FreeKick and Penalty name the taking team
in Team. A KickOff names no team on itself (TeamLeft and TeamRight are the sides of the pitch), only on its <Play> child
(the first pass), whose Team is the team that takes it; a KickOff without one is refused. The club id is mapped to 0 home,
1 away through the match information's Role, as dfl.py does. An event of another kind is not a restart and is skipped;
a restart without an EventTime, with a time that is not an ISO time with a zone, or whose team is neither club is a
ValueError naming the file and the event.

The time cache
--------------
Reading T takes a pass over the 350-420 MB positions file (about a minute). load_restarts keeps the four numbers of
each half in <data dir>/cache/dfl_events/<match id>.json, keyed by the positions file's size and this file's sha256,
and rebuilds when either changes. dfl.py and its cache are untouched.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
from lxml import etree

import games
from pipeline.benchmark import dfl, sources

# Params
KINDS = ("KickOff", "ThrowIn", "CornerKick", "GoalKick", "FreeKick", "Penalty")   # the restarts, as DFL names them
FRAME_MS = 40                                  # the positions are sampled at 25 Hz
BALL_TEAM_ID = dfl.BALL_TEAM_ID                # FrameSet@TeamId of the ball
SECTIONS = dfl.SECTIONS                        # FrameSet@GameSection -> half
CACHE_FOLDER = ("cache", "dfl_events")         # <data dir>/cache/dfl_events/<match id>.json
CACHE_VERSION = 1                              # a manual switch: bump it to drop every cache


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


PARSER_SHA256 = _sha256(__file__)


def _ms(text, path, what):
    """Milliseconds since the epoch of an ISO time that names its zone; ValueError, naming the file, otherwise."""
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError("{}: {} is {!r}, which is not an ISO time with a time zone".format(path, what, text)) from None
    if moment.tzinfo is None:
        raise ValueError("{}: {} is {!r}, which names no time zone, so it cannot be put on the frame clock".format(
            path, what, text))
    return int(round(moment.astimezone(timezone.utc).timestamp() * 1000))


def scan_ball_times(path):
    """{half: (first N, T of it, last N, T of it)} of the ball FrameSets, T in ms since the epoch.

    One pass over the positions file, which keeps nothing but the smallest and the largest N of each ball FrameSet and
    their T. ValueError, naming the file, for a half with no ball, a frame without N or T, and a half whose clock is not
    FRAME_MS per frame between its first and its last frame.
    """
    best = {}
    current = None
    try:
        with open(path, "rb") as handle:
            for event, element in etree.iterparse(handle, events=("start", "end"), tag=("FrameSet", "Frame"),
                                                  huge_tree=True, resolve_entities=False, no_network=True):
                if element.tag == "Frame":
                    if event == "end":
                        if current is not None:
                            n, t = element.get("N"), element.get("T")
                            if n is None or t is None:
                                raise ValueError("{}: a ball <Frame> has no N or no T attribute".format(path))
                            n = int(n)
                            low, high = best.get(current, (None, None))
                            if low is None or n < low[0]:
                                low = (n, t)
                            if high is None or n > high[0]:
                                high = (n, t)
                            best[current] = (low, high)
                        element.clear()
                        parent = element.getparent()
                        while element.getprevious() is not None:
                            del parent[0]
                elif event == "start":
                    section = element.get("GameSection")
                    current = SECTIONS.get(section) if element.get("TeamId") == BALL_TEAM_ID else None
                    if element.get("TeamId") == BALL_TEAM_ID and current is None:
                        raise ValueError("{}: the ball FrameSet has GameSection={!r}, not {}".format(
                            path, section, " or ".join(SECTIONS)))
                else:
                    current = None
                    element.clear()
    except etree.XMLSyntaxError as error:
        raise ValueError("{}: not well-formed XML ({})".format(path, error)) from error
    except OSError as error:
        raise ValueError("{}: cannot be read ({})".format(path, error)) from error
    scanned = {}
    for section, half in SECTIONS.items():
        if half not in best:
            raise ValueError("{}: no ball FrameSet for GameSection=\"{}\" (secondHalf and firstHalf each need one)".format(
                path, section))
        (n0, text0), (n1, text1) = best[half]
        t0, t1 = _ms(text0, path, "the T of the first ball frame"), _ms(text1, path, "the T of the last ball frame")
        if t1 - t0 != FRAME_MS * (n1 - n0):
            raise ValueError("{}: the ball clock of {} is not {} ms a frame: N {} to {} is {} ms by T, {} by the frame "
                             "count; a restart's row would be off, so the file is refused".format(
                                 path, section, FRAME_MS, n0, n1, t1 - t0, FRAME_MS * (n1 - n0)))
        scanned[half] = (n0, t0, n1, t1)
    return scanned


def _restart_of(event_element, path, club_team):
    """(kind, team, EventTime text) of an <Event> that holds a restart, or None."""
    for child in event_element:
        if child.tag not in KINDS:
            continue
        kind = child.tag
        label = "<{}> of EventId={}".format(kind, event_element.get("EventId"))
        if kind == "KickOff":
            play = child.find("Play")
            club = None if play is None else play.get("Team")
            where = "its <Play> has no Team" if club is None else None
        else:
            club = child.get("Team")
            where = "it has no Team" if club is None else None
        if club is None:
            raise ValueError("{}: {}: {}, and a restart's team is what is wanted of it".format(path, label, where))
        if club not in club_team:
            raise ValueError("{}: {} has Team={!r}, which is neither the home club ({}) nor the away club ({}) of the "
                             "match information".format(path, label, club, *club_team["home_away"]))
        return kind, club_team[club], event_element.get("EventTime"), label
    return None


def build_restarts(events_path, positions_path, info, graded):
    """The restarts of one match on its timeline, from its two files and the layout of the timeline.

    `info` is the MatchInfo (for home and away) and `graded` is DflMatch.graded: ((half, start, stop), ...). Returns
    a dict of read-only arrays sorted by row (stable in the file's order): "row" int64, "kind" (str, one of KINDS) and
    "team" int8 (0 home, 1 away); and "outside", the number of restarts whose time is in neither half. ValueError,
    naming the file, for anything that cannot be placed faithfully (see the module docstring).
    """
    times = scan_ball_times(positions_path)
    return place_restarts(events_path, info, graded, times)


def place_restarts(events_path, info, graded, times):
    """build_restarts, given the scanned ball times (scan_ball_times) instead of the positions file."""
    club_team = {info.home_id: 0, info.away_id: 1, "home_away": (info.home_id, info.away_id)}
    start_of = {half: (start, stop) for half, start, stop in graded}
    if set(start_of) != set(times):
        raise ValueError("{}: the timeline has halves {} and the ball clock has {}".format(
            events_path, sorted(start_of), sorted(times)))
    split = (times[1][3] + times[2][1]) // 2   # the break's midpoint, in ms: before it is the first half
    try:
        root = etree.parse(events_path, etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
    except etree.XMLSyntaxError as error:
        raise ValueError("{}: not well-formed XML ({})".format(events_path, error)) from error
    rows, kinds, teams, outside = [], [], [], 0
    for element in root.iter("Event"):
        found = _restart_of(element, events_path, club_team)
        if found is None:
            continue
        kind, team, text, label = found
        if text is None:
            raise ValueError("{}: {} has no EventTime".format(events_path, label))
        t = _ms(text, events_path, "the EventTime of " + label)
        half = 1 if t < split else 2
        start, stop = start_of[half]
        # Round half up on integers: (offset + 20) // 40, the same for times before the first frame (a negative offset).
        row = start + (t - times[half][1] + FRAME_MS // 2) // FRAME_MS
        if not start <= row < stop:
            outside += 1
            continue
        rows.append(row)
        kinds.append(kind)
        teams.append(team)
    order = np.argsort(np.array(rows, dtype=np.int64), kind="stable")
    found = {"row": np.array(rows, dtype=np.int64)[order], "kind": np.array(kinds, dtype="<U10")[order],
             "team": np.array(teams, dtype=np.int8)[order]}
    for array in found.values():
        array.setflags(write=False)
    found["outside"] = outside
    return found


def _cache_path(match_id, data_dir):
    return os.path.join(data_dir or games.DATA_DIR, *CACHE_FOLDER, match_id + ".json")


def _times_of(match_id, positions_path, data_dir):
    """The ball times of a match: from its cache when the key fits, otherwise scanned (and stored)."""
    path = _cache_path(match_id, data_dir)
    key = {"cache_version": CACHE_VERSION, "positions_bytes": os.path.getsize(positions_path),
           "parser_sha256": PARSER_SHA256}
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        if all(stored.get(name) == value for name, value in key.items()):
            return {int(half): tuple(int(v) for v in values) for half, values in stored["times"].items()}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    times = scan_ball_times(positions_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(key, times={str(half): list(values) for half, values in times.items()}), handle, indent=2,
                  sort_keys=True)
        handle.write("\n")
    return times


def load_restarts(match_id, data_dir=None):
    """The restarts of one of the seven DFL matches as {"row", "kind", "team", "outside"} (see build_restarts).

    The rows are those of dfl.load_match(match_id)'s timeline. The first call reads the positions file's clock (about a
    minute) and caches it; later calls take a second or two. ValueError for an id outside sources.MATCH_IDS;
    whatever dfl and build_restarts raise otherwise.
    """
    if match_id not in sources.MATCH_IDS:
        raise ValueError("{!r} is not one of the DFL benchmark matches ({})".format(match_id, ", ".join(sources.MATCH_IDS)))
    paths = {kind: sources.local_path(match_id, kind, data_dir) for kind in sources.KINDS}
    for kind, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("{}: no such file; {}".format(path, dfl.FETCH_ADVICE))
    match = dfl.load_match(match_id, data_dir)
    info = dfl.parse_match_information(paths["information"])
    times = _times_of(match_id, paths["positions"], data_dir)
    return place_restarts(paths["events"], info, match.graded, times)
