"""P16, R023: Metrica's three public sample games, H/A/D derived from their events, and the grain of those labels.

    python pipeline/benchmark/metrica.py              # the licence stop, fetch what is missing, derive, print the verdict
    python pipeline/benchmark/metrica.py --offline    # no network: the files already on disk, checked by blob SHA
    python pipeline/benchmark/metrica.py --data-dir D --no-write

    games = load_games(data_dir)       # {game: [Event, ...]}, read by our own stdlib readers
    result = measure(games)            # integrity, labels, sanity bounds, the pooled grain and R023's outcome

Why it exists
-------------
Every re-tune since R007 has split the raw and the coarse truth (T33, A06), and which one our ground truth resembles
cannot be told from the public window. Metrica, the provider of our ground truth, publishes three anonymised sample
games with event data. R023 (docs/results.md, signed by Tian on 2026-10-31) derives H/A/D from those events by a rule
taken from Metrica's own event definitions, measures the grain (possessions per alive minute, the median possession) as
A06 measured DFL's, and sets which truth ranks later choices: raw, coarse, or both (the unweighted mean of the two).
This module is R023's part 2, and every rule below is R023's part 1 as written; the research behind it is
docs/research/2026-10-31-p16-metrica.md.

The files
---------
metrica-sports/sample-data has no licence file; its README asks to "be responsible" and to "acknowledge the source".
Tian ruled that this suffices for fetching at run time, never committing (R023's licence stop). So the event files are
fetched from one pinned commit into <data dir>/external/metrica/ (gitignored), each checked by its git blob SHA, and
the fetch first checks the stop: the README must still be the pinned blob at that commit and at HEAD, and the
repository must still have no licence file. No tracking file is fetched. kloppy 3.18.0 cannot read the CSV events and
drops SET PIECE events from the JSON, so the readers are ours; kloppy's constants only cross-check game 3's ids.

The derivation (R023 part 1, per game, periods 1 and 2, events in file order)
-----------------------------------------------------------------------------
0. Span: a period runs from its first event's start frame to its last event's end frame, inclusive; outside it, X.
1. Stops (D opens): the end frame of a BALL OUT; the end frame of an event with subtype GOAL; the end frame of a SHOT
   with subtype OUT; the start frame of a FAULT RECEIVED, or of a CHALLENGE with subtype FAULT; the end frame of an
   event with subtype FORCED, END HALF or OFFSIDE.
2. Restarts (D closes): at the start frame of the first later event of type SET PIECE, RECOVERY, PASS, SHOT, BALL LOST
   or CARRY whose start frame is after the stop's. D covers [stop, restart), or runs to the span's end with no
   restart; overlapping intervals merge.
3. Team on the alive frames: walk the events keeping the current team T (unset at the period's first event) and the
   frame t of the last change. For an event of team E: (a) a RECOVERY, or a SET PIECE, PASS, SHOT, BALL LOST, BALL OUT
   or CARRY with E != T, changes to E at max(start, t); (b) then a BALL LOST changes to the other team at max(end, t).
   A frame takes the team of the last change at or before it. CHALLENGE, FAULT RECEIVED and CARD never change it.
4. Labels: D on step 2's intervals; elsewhere Home (game 3: the metadata's first team) is H and the other team A.

Integrity, sanity, outcome
--------------------------
A game whose files fail an integrity check (integrity) drops out; with fewer than two games left R023 is invalid. A game
outside the sanity bounds (alive share 40% to 70%, 60 to 160 stoppages) makes R023 invalid, and drops nothing. The
grain is truth.summary of the passing games laid end to end (truth.concatenate), exactly as A06's figures were
computed. decide() turns it into the outcome. An invalid R023 decides nothing, and the raw truth stays the objective.
Everything printed and written is an aggregate of these public games; nothing of BAR-ATH is read.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline.benchmark import truth  # noqa: E402
from pipeline.benchmark.sources import TIMEOUT, USER_AGENT, DownloadError  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"      # metrica-sports/sample-data, "Added Sample Game 3", 2021-04-15
RAW_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{ref}/{path}"
LICENSE_API = "https://api.github.com/repos/metrica-sports/sample-data/license"
README_BLOB = "03fb97acc3545cc7428d0a07485921c5a1d405c1"
FPS = 25
PERIODS = (1, 2)
OUT_DIR = ("runs", "metrica")

# The PDF's nine event types, and CARRY (game 3's JSON only); for game 3, the ids kloppy 3.18.0's Metrica JSON
# deserializer gives them (MS_PASS_OUTCOME_COMPLETE 1, ..._INCOMPLETE 7, ..._OUT 6, MS_EVENT_TYPE_FOUL_COMMITTED 4, ...).
TYPE_IDS = {"PASS": 1, "SHOT": 2, "RECOVERY": 3, "FAULT RECEIVED": 4, "SET PIECE": 5, "BALL OUT": 6, "BALL LOST": 7,
            "CARD": 8, "CHALLENGE": 9, "CARRY": 10}

# Step 1: (kind, frame the stop is at, test); step 2's restart types; step 3's types that change team when E != T.
STOPS = (("ball out", "end", lambda e: e.type == "BALL OUT"),
         ("goal", "end", lambda e: "GOAL" in e.subtypes),
         ("shot out", "end", lambda e: e.type == "SHOT" and "OUT" in e.subtypes),
         ("fault received", "start", lambda e: e.type == "FAULT RECEIVED"),
         ("challenge fault", "start", lambda e: e.type == "CHALLENGE" and "FAULT" in e.subtypes),
         ("forced", "end", lambda e: "FORCED" in e.subtypes),
         ("end half", "end", lambda e: "END HALF" in e.subtypes),
         ("offside", "end", lambda e: "OFFSIDE" in e.subtypes))
RESTART_TYPES = frozenset({"SET PIECE", "RECOVERY", "PASS", "SHOT", "BALL LOST", "CARRY"})
ON_PLAY_TYPES = frozenset({"SET PIECE", "PASS", "SHOT", "BALL LOST", "BALL OUT", "CARRY"})
OTHER = {pm.H: pm.A, pm.A: pm.H}

# The decision (R023 part 1): A06's pooled figures on the 7 DFL matches, and the band around their geometric midpoint.
RAW_RATE, COARSE_RATE = 8.30, 4.50          # possessions per alive minute
RAW_MEDIAN_S, COARSE_MEDIAN_S = 3.68, 8.40  # median possession, seconds
BAND = 1.15                                 # "both" within this ratio of the midpoint, either way
ALIVE_BOUNDS = (0.40, 0.70)                 # sanity: every game's alive share
STOPPAGE_BOUNDS = (60, 160)                 # sanity: every game's stoppages
MIN_GAMES = 2


@dataclass(frozen=True)
class RemoteFile:
    """One pinned file of the sample-data repository and the git blob SHA it must have."""
    game: int
    kind: str
    path: str
    blob: str

    @property
    def name(self):
        return self.path.rsplit("/", 1)[1]

    def local(self, data_dir=None):
        return os.path.join(external_dir(data_dir), self.name)


FILES = (RemoteFile(1, "events", "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
                    "3c426dd4f442dd433bccf89e3634f91aa310cd8f"),
         RemoteFile(2, "events", "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
                    "decccd47c1d53960d290cb6679337eec83a8a97f"),
         RemoteFile(3, "events", "data/Sample_Game_3/Sample_Game_3_events.json",
                    "992b4cb502f0b0bc1244174f014f3f7ecb6d38ef"),
         RemoteFile(3, "metadata", "data/Sample_Game_3/Sample_Game_3_metadata.xml",
                    "74b55a80f6aeeb8f8df7fd26e4b3a94e2cc1738d"))


@dataclass(frozen=True)
class Event:
    """One event as R023 reads it. Names are upper case; a frame that cannot be read is None; ids are game 3's only."""
    period: int
    team: object         # pm.H, pm.A, or None when it cannot be read (integrity fails the game)
    type: str
    subtypes: tuple
    start_frame: object  # int or None
    start_time: object   # float or None
    end_frame: object    # int or None
    type_id: object = None
    subtype_ids: tuple = ()


# -- the files --------------------------------------------------------------------------------------------------------

def external_dir(data_dir=None):
    """<data dir>/external/metrica: gitignored, as everything under data/ is."""
    return os.path.join(data_dir or games.DATA_DIR, "external", "metrica")


def blob_sha(body):
    """The git blob SHA-1 of these bytes: what GitHub's tree lists, so a file is pinned without a separate md5."""
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def verify(data_dir=None):
    """{name: ok | missing | blob} for the four pinned files on disk."""
    status = {}
    for remote in FILES:
        path = remote.local(data_dir)
        if not os.path.isfile(path):
            status[remote.name] = "missing"
            continue
        with open(path, "rb") as handle:
            status[remote.name] = "ok" if blob_sha(handle.read()) == remote.blob else "blob"
    return status


def licence_stop(opener=None):
    """R023's licence stop: a sentence for each condition that fires, an empty list if none does. Reads the network.

    It fires when the README at the pinned commit or at HEAD is not the pinned blob, or when the repository has a
    licence file. R023 stops on one "that forbids this use"; any licence file stops here, since a person has to read its
    terms to tell (stricter than declared, and it can only halt the run, never change its outcome). Tian's signature already
    ruled the README's note sufficient, the third condition.
    """
    opener = opener or urllib.request.urlopen
    fired = []
    for ref in (COMMIT, "HEAD"):
        try:
            body = _get(RAW_URL.format(ref=ref, path="README.md"), opener)
        except (OSError, ValueError) as error:
            fired.append("the README at {} could not be read ({}), so the stop cannot be checked".format(ref, error))
            continue
        if blob_sha(body) != README_BLOB:
            fired.append("the README at {} is blob {}, not the pinned {}: read its terms again".format(
                ref, blob_sha(body), README_BLOB))
    try:
        _get(LICENSE_API, opener)
        fired.append("the repository now has a licence file ({}): read it before fetching".format(LICENSE_API))
    except urllib.error.HTTPError as error:
        if error.code != 404:
            fired.append("GitHub answered {} for {}, so the stop cannot be checked".format(error.code, LICENSE_API))
    except OSError as error:
        fired.append("{} could not be read ({}), so the stop cannot be checked".format(LICENSE_API, error))
    return fired


def fetch(data_dir=None, opener=None, log=print):
    """Make the four pinned files ok on disk, fetching only what is missing or damaged; DownloadError otherwise.

    A file is written only once its bytes have the pinned blob SHA (to <file>.part, then renamed), so a file on disk
    is either right or absent. The caller runs licence_stop first.
    """
    opener = opener or urllib.request.urlopen
    os.makedirs(external_dir(data_dir), exist_ok=True)
    for remote in FILES:
        path = remote.local(data_dir)
        if verify(data_dir)[remote.name] == "ok":
            log("ok          {}".format(remote.name))
            continue
        url = RAW_URL.format(ref=COMMIT, path=remote.path)
        try:
            body = _get(url, opener)
        except (OSError, ValueError) as error:
            raise DownloadError("{}: {} ({}); nothing was kept. Check the network and re-run.".format(
                remote.name, url, error)) from error
        if blob_sha(body) != remote.blob:
            raise DownloadError("{}: got blob {}, pinned {}; nothing was kept. The repository has changed under "
                                "the pinned commit, or the download is damaged.".format(
                                    remote.name, blob_sha(body), remote.blob))
        with open(path + ".part", "wb") as handle:
            handle.write(body)
        os.replace(path + ".part", path)
        log("fetched     {} ({:,} bytes)".format(remote.name, len(body)))


def _get(url, opener):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=TIMEOUT) as response:
        return response.read()


# -- the readers ------------------------------------------------------------------------------------------------------

def _number(cell, kind):
    """An int or float from a cell, None for a blank, NaN or unreadable one (integrity then reports it)."""
    if cell is None:
        return None
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    if kind is int:
        return int(value) if value.is_integer() else None
    return value


def read_csv_events(path):
    """Games 1 and 2: Metrica's RawEventsData.csv as Events, in file order. Subtypes are `Subtype` split on '-'."""
    teams = {"HOME": pm.H, "AWAY": pm.A}
    events = []
    with open(path, newline="", encoding="utf-8-sig") as handle:      # -sig: a byte-order mark is not part of `Team`
        for row in csv.DictReader(handle):
            team = teams.get((row["Team"] or "").strip().upper())       # None: integrity() fails the game
            subtypes = tuple(part.strip().upper() for part in row["Subtype"].split("-") if part.strip())
            events.append(Event(period=_number(row["Period"], int), team=team, type=row["Type"].strip().upper(),
                                subtypes=subtypes, start_frame=_number(row["Start Frame"], int),
                                start_time=_number(row["Start Time [s]"], float),
                                end_frame=_number(row["End Frame"], int)))
    return events


def read_json_events(path, metadata_path, fps=FPS):
    """Game 3: Metrica's events JSON as Events, in file order. H is the metadata's first <Team>.

    A start or end without a frame takes round(time x fps), as R023 says (game 3's file gives every frame: the format
    read before this code was written). Its first <Team> is also the metadata's `idLocalTeam`, as checked then.
    """
    teams = ET.parse(metadata_path).getroot().find("Metadata").find("Teams").findall("Team")
    if len(teams) != 2:
        raise ValueError("{}: {} <Team> elements, expected 2".format(metadata_path, len(teams)))
    codes = {teams[0].attrib["id"]: pm.H, teams[1].attrib["id"]: pm.A}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)["data"]
    events = []
    for raw in data:
        team = codes.get((raw.get("team") or {}).get("id"))           # None: integrity() fails the game
        subtypes = raw.get("subtypes")
        subtypes = [] if subtypes is None else subtypes if isinstance(subtypes, list) else [subtypes]
        start, end = raw.get("start") or {}, raw.get("end") or {}
        events.append(Event(period=_number(raw.get("period"), int), team=team, type=raw["type"]["name"].strip().upper(),
                            subtypes=tuple(s["name"].strip().upper() for s in subtypes),
                            start_frame=_frame(start, fps), start_time=_number(start.get("time"), float),
                            end_frame=_frame(end, fps), type_id=raw["type"]["id"],
                            subtype_ids=tuple(s["id"] for s in subtypes)))
    return events


def _frame(point, fps):
    """The point's frame; round(time x fps) only where the frame is missing (a frame that is there but unreadable
    stays None, and integrity() fails the game)."""
    if point.get("frame") is not None:
        return _number(point["frame"], int)
    time = _number(point.get("time"), float)
    return None if time is None else int(round(time * fps))


def load_games(data_dir=None):
    """{1: [Event, ...], 2: [...], 3: [...]} from the files on disk (fetch() first)."""
    paths = {(remote.game, remote.kind): remote.local(data_dir) for remote in FILES}
    return {1: read_csv_events(paths[1, "events"]), 2: read_csv_events(paths[2, "events"]),
            3: read_json_events(paths[3, "events"], paths[3, "metadata"])}


# -- integrity --------------------------------------------------------------------------------------------------------

def integrity(events, fps=FPS):
    """The integrity checks R023 declares that this game fails, one sentence each with a count; [] if it passes.

    Only periods 1 and 2 are read. They concern the files' format, never the grain: a start frame that is not
    round(start time x fps) to within 1 frame (also a check of 25 fps), start frames that decrease within a period, a
    period that does not begin with a SET PIECE KICK OFF (or has no event at all), a type outside the PDF's nine and
    CARRY, an event with no end frame, and for game 3 a type whose name disagrees with kloppy's id for it (R023: "kloppy
    only cross-checks game 3's type ids"). A team that cannot be read fails too: for game 3 that is R023's "names cannot
    be mapped", and for the CSV games it is a format failure of the same kind (a ruling made before the run).
    """
    events = [e for e in events if e.period in PERIODS]
    failures = []

    def count(test, sentence):
        n = sum(1 for e in events if test(e))
        if n:
            failures.append("{:,} {}".format(n, sentence))

    count(lambda e: e.start_frame is None or e.start_time is None
          or abs(e.start_frame - round(e.start_time * fps)) > 1,
          "events whose start frame is not round(start time x {}) to within 1".format(fps))
    count(lambda e: e.end_frame is None, "events with no end frame")
    count(lambda e: e.type not in TYPE_IDS, "events whose type is not one of the PDF's nine or CARRY")
    count(lambda e: e.type_id is not None and TYPE_IDS.get(e.type) != e.type_id,
          "events whose type name disagrees with kloppy's id for it")
    count(lambda e: e.team is None, "events whose team cannot be read (game 3: not one of the metadata's two)")
    for period in PERIODS:
        own = [e for e in events if e.period == period]
        if not own:
            failures.append("period {} has no event".format(period))
            continue
        if not (own[0].type == "SET PIECE" and "KICK OFF" in own[0].subtypes):
            failures.append("period {} does not begin with a SET PIECE KICK OFF".format(period))
        frames = [e.start_frame for e in own if e.start_frame is not None]
        drops = sum(1 for before, after in zip(frames, frames[1:]) if after < before)
        if drops:
            failures.append("period {}: start frames decrease {:,} times in file order".format(period, drops))
    return failures


# -- the derivation ---------------------------------------------------------------------------------------------------

def derive(events):
    """(codes, graded, stops, open_ended) of one game that passed integrity().

    `codes` is an int8 array of the two periods' spans laid end to end with one X between them, as truth.concatenate
    lays out matches, so a file whose second period numbers its frames from 0 again is read the same; `graded` is
    ((period, start, stop), ...) with half-open rows of `codes`, as truth.summary takes it; `stops` counts step 1's stops
    by kind (an event may make two), and `open_ended` how many stops found no restart in their period. A period whose
    last event ends before its first event starts has no span: ValueError (no file is expected to have one).
    """
    pieces, stops, open_ended = [], Counter(), 0
    for period in PERIODS:
        own = [e for e in events if e.period == period]
        start, stop = own[0].start_frame, own[-1].end_frame + 1
        if stop <= start:
            raise ValueError("period {}: its last event ends at frame {}, before its first starts ({})".format(
                period, stop - 1, start))
        labels = _teams(own, start, stop)
        dead, kinds, unmatched = _dead(own, start, stop)
        labels[dead] = pm.D
        pieces.append((labels, ((period, 0, stop - start),)))
        stops.update(kinds)
        open_ended += unmatched
    codes, graded = truth.concatenate(pieces)
    return codes, graded, dict(stops), open_ended


def _teams(events, start, stop):
    """Step 3: H or A on every frame of [start, stop), from the changes the events make."""
    frames, teams = [], []
    current, last = None, start
    for e in events:
        if e.type == "RECOVERY" or (e.type in ON_PLAY_TYPES and e.team != current):
            last, current = max(e.start_frame, last), e.team
            frames.append(last)
            teams.append(current)
        if e.type == "BALL LOST":
            last, current = max(e.end_frame, last), OTHER[e.team]
            frames.append(last)
            teams.append(current)
    if not frames or frames[0] > start:
        raise ValueError("the period's first event makes no team change at its start frame {}".format(start))
    which = np.searchsorted(np.asarray(frames), np.arange(start, stop), side="right") - 1
    return np.asarray(teams, dtype=np.int8)[which]


def _dead(events, start, stop):
    """Steps 1 and 2: the mask of D over [start, stop), the stops by kind, and how many found no restart."""
    dead = np.zeros(stop - start, dtype=bool)
    kinds, unmatched = Counter(), 0
    for index, e in enumerate(events):
        for kind, edge, test in STOPS:
            if not test(e):
                continue
            kinds[kind] += 1
            frame = e.end_frame if edge == "end" else e.start_frame
            restart = next((later.start_frame for later in events[index + 1:]
                            if later.type in RESTART_TYPES and later.start_frame > frame), None)
            if restart is None:
                unmatched += 1
                restart = stop
            dead[max(frame, start) - start:max(min(restart, stop), start) - start] = True
    return dead, kinds, unmatched


# -- the measure and the decision -------------------------------------------------------------------------------------

def classify(value, raw, coarse, larger_is_raw):
    """raw, coarse or both: both within BAND of the geometric midpoint of raw and coarse (in log ratio), else the
    nearer side. Possessions per alive minute are larger on the raw side; the median is smaller there."""
    middle = math.sqrt(raw * coarse)
    if abs(math.log(value / middle)) <= math.log(BAND):
        return "both"
    return "raw" if (value > middle) == larger_is_raw else "coarse"


def decide(rate, median_s):
    """R023's outcome from the pooled grain: the rate class, except that a raw / coarse split between the rate class
    and the median class gives both (T33: a disagreement is a warning)."""
    by_rate = classify(rate, RAW_RATE, COARSE_RATE, larger_is_raw=True)
    by_median = classify(median_s, RAW_MEDIAN_S, COARSE_MEDIAN_S, larger_is_raw=False)
    outcome = "both" if {by_rate, by_median} == {"raw", "coarse"} else by_rate
    return {"rate_class": by_rate, "median_class": by_median, "outcome": outcome}


def measure(games_events, fps=FPS):
    """R023 on {game: [Event, ...]}: per game its integrity, grain and stops; the pooled grain; the outcome.

    `valid` is False, and `outcome` "invalid", when fewer than MIN_GAMES games pass integrity or a passing game is
    outside the sanity bounds; the figures are still reported, and decide nothing.
    """
    result, parts, reasons, pooled_stops = {"games": {}}, [], [], Counter()
    for game in sorted(games_events):
        events = games_events[game]
        cell = {"events": sum(1 for e in events if e.period in PERIODS), "integrity": integrity(events, fps)}
        if not cell["integrity"]:
            codes, graded, stops, open_ended = derive([e for e in events if e.period in PERIODS])
            cell.update(summary=truth.summary(codes, graded, fps), stops=stops, open_ended=open_ended)
            parts.append((codes, graded))
            pooled_stops.update(stops)
            low, high = ALIVE_BOUNDS
            if not low <= cell["summary"]["alive_share"] <= high:
                reasons.append("game {}: alive share {:.3f} is outside {} to {}".format(
                    game, cell["summary"]["alive_share"], low, high))
            low, high = STOPPAGE_BOUNDS
            if not low <= cell["summary"]["stoppages"] <= high:
                reasons.append("game {}: {} stoppages, outside {} to {}".format(
                    game, cell["summary"]["stoppages"], low, high))
        result["games"]["game_{}".format(game)] = cell
    if len(parts) < MIN_GAMES:
        reasons.insert(0, "{} game(s) passed integrity, fewer than {}".format(len(parts), MIN_GAMES))
    if parts:
        result["pooled"] = truth.summary(*truth.concatenate(parts), fps)
        result["pooled_stops"] = dict(pooled_stops)
        result.update(decide(result["pooled"]["per_alive_minute"], result["pooled"]["median_s"]))
    result["valid"] = not reasons
    result["invalid_because"] = reasons
    if reasons:
        result["outcome"] = "invalid"
    return result


def render(result):
    """The printed lines: per game and pooled aggregates, the classes and the outcome."""
    lines = []
    for name, cell in result["games"].items():
        if cell["integrity"]:
            lines.append("{}  {:,} events: FAILS integrity, dropped: {}".format(
                name, cell["events"], "; ".join(cell["integrity"])))
            continue
        lines.append("{}  {:,} events: {}; stops {}, {} with no restart".format(
            name, cell["events"], _grain(cell["summary"]),
            ", ".join("{} {}".format(kind, n) for kind, n in sorted(cell["stops"].items())), cell["open_ended"]))
    if "pooled" in result:
        lines.append("pooled  {}; stops {}".format(_grain(result["pooled"]), ", ".join(
            "{} {}".format(kind, n) for kind, n in sorted(result["pooled_stops"].items()))))
        lines.append("classes rate {} (raw {}, coarse {}, both {:.2f} to {:.2f}), median {} (raw {} s, coarse {} s, "
                     "both {:.2f} to {:.2f} s){}".format(
                         result["rate_class"], RAW_RATE, COARSE_RATE, *_band(RAW_RATE, COARSE_RATE),
                         result["median_class"], RAW_MEDIAN_S, COARSE_MEDIAN_S, *_band(RAW_MEDIAN_S, COARSE_MEDIAN_S),
                         "" if result["valid"] else " -- R023 is invalid, so these decide nothing"))
    for reason in result["invalid_because"]:
        lines.append("INVALID {}".format(reason))
    lines.append("outcome {}".format(result["outcome"]))
    return lines


def _grain(summary):
    return ("{:.2f} possessions per alive minute, median {:.2f} s, mean {:.2f} s, {:,} possessions, alive {:.3f}, "
            "{:,} stoppages, under 5 s {:.3f}, <=2 frames {:,}, <=12 frames {:,}, home {:.3f}".format(
                summary["per_alive_minute"], summary["median_s"], summary["mean_s"], summary["possessions"],
                summary["alive_share"], summary["stoppages"], summary["share_under_5s"], summary["runs_le_2_frames"],
                summary["runs_le_12_frames"], summary["home_share"]))


def _band(raw, coarse):
    middle = math.sqrt(raw * coarse)
    return middle / BAND, middle * BAND


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="R023 (P16 part 2): the grain of H/A/D derived from Metrica's three sample games' events.")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--offline", action="store_true",
                        help="no network: no licence stop, no fetch; the files on disk must already be ok")
    parser.add_argument("--no-write", action="store_true", help="print only; write no JSON")
    parser.add_argument("--out", help="the JSON to write (default: <data dir>/runs/metrica/r023.json)")
    args = parser.parse_args(argv)

    from pipeline.benchmark import run
    try:
        if not args.offline:
            fired = licence_stop()
            if fired:
                for sentence in fired:
                    print("STOP      {}".format(sentence))
                return 1
            fetch(args.data_dir)
        bad = {name: status for name, status in verify(args.data_dir).items() if status != "ok"}
        if bad:
            raise DownloadError("not ok on disk: {}; run without --offline to fetch".format(bad))
        result = measure(load_games(args.data_dir))
        for line in render(result):
            print(line)
        if not args.no_write:
            out = args.out or os.path.join(args.data_dir or games.DATA_DIR, *OUT_DIR, "r023.json")
            run.write_json(out, run.plain(result))
            print("wrote     {}".format(run.shown(out)))
    except (DownloadError, ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
