"""Checks of metrica (pipeline/benchmark/metrica.py): R023, the grain of H/A/D derived from Metrica's sample games.

Everything here is synthetic: git's blob SHA on known bodies, the pins against R023's text, the fetch and the licence
stop against a fake opener (no network, nothing outside a temporary folder), both readers on small files written here,
each integrity failure, the derivation worked out frame by frame on a hand-built game, the decision at its edges, and
the measure's drop and invalid rules. kloppy's constants are compared with ours where kloppy is installed.
"""

import io
import json
import os
import tempfile
import urllib.error

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import metrica, truth
from pipeline.benchmark.checks.common import NotRun
from pipeline.benchmark.sources import DownloadError
from pipeline.selftest import expect_error, quiet

H, A, D, X = pm.H, pm.A, pm.D, pm.X


# -- helpers ----------------------------------------------------------------------------------------------------------

class FakeOpener:
    """Serves a body per URL, or raises the exception stored for it; remembers the URLs asked for."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.urls = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        body = self.bodies[request.full_url]
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body)


def not_found(url, code=404):
    return urllib.error.HTTPError(url, code, "status {}".format(code), {}, None)


def ev(period, team, kind, start, end, *subtypes, time=None, type_id=None, subtype_ids=()):
    """An Event; its start time agrees with its start frame at 25 fps unless given."""
    return metrica.Event(period=period, team=team, type=kind, subtypes=tuple(subtypes), start_frame=start,
                         start_time=start / 25 if time is None else time, end_frame=end, type_id=type_id,
                         subtype_ids=tuple(subtype_ids))


def hand_built_game():
    """Two periods whose labels are worked out by hand in test_metrica_derive_by_hand."""
    return [
        ev(1, H, "SET PIECE", 10, 12, "KICK OFF"),
        ev(1, H, "PASS", 12, 20),
        ev(1, A, "CHALLENGE", 20, 21, "TACKLE", "LOST"),       # never changes the team
        ev(1, H, "BALL LOST", 21, 25, "INTERCEPTION"),         # H already: no (a); (b) A at its end, 25
        ev(1, A, "RECOVERY", 26, 26, "INTERCEPTION"),          # A again at 26: no visible change
        ev(1, A, "PASS", 27, 35),
        ev(1, A, "BALL OUT", 35, 40),                          # stop at 40
        ev(1, H, "SET PIECE", 45, 46, "THROW IN"),             # restart at 45: D [40, 45); H at 45
        ev(1, H, "PASS", 46, 50),
        ev(1, H, "FAULT RECEIVED", 50, 50),                    # stop at its start, 50
        ev(1, A, "CHALLENGE", 50, 51, "FAULT"),                # stop at 50 again: the intervals merge
        ev(1, H, "SET PIECE", 55, 58, "FREE KICK"),            # restart at 55; H already
        ev(1, H, "SHOT", 58, 62, "ON TARGET", "GOAL"),         # stop at 62; no restart later in the period
        ev(1, A, "CARD", 63, 64, "YELLOW"),                    # the span ends at 64
        ev(2, A, "SET PIECE", 70, 72, "KICK OFF"),             # T unset: A at 70
        ev(2, A, "PASS", 72, 80),
        ev(2, H, "PASS", 80, 85),                              # the other team's first event: H at 80
        ev(2, H, "PASS", 85, 90, "OFFSIDE"),                   # stop at 90
        ev(2, A, "SET PIECE", 95, 96, "FREE KICK"),            # restart at 95; A at 95
        ev(2, A, "BALL LOST", 96, 99),                         # (b): H at 99
        ev(2, A, "PASS", 97, 101),                             # A != H: A at max(97, 99) = 99, the later change
        ev(2, A, "BALL LOST", 101, 104, "END HALF"),           # (b): H at 104; stop at 104, no restart
    ]


def cycle_period(period, first, cycles, alive=100, dead=50):
    """A period of `cycles` possessions, each a PASS then a BALL OUT and a SET PIECE of the other team `dead` frames
    later: alive/(alive+dead) of the span is alive and there are `cycles` stoppages."""
    events, team, frame = [ev(period, H, "SET PIECE", first, first + 1, "KICK OFF")], H, first + 1
    for _ in range(cycles):
        events.append(ev(period, team, "PASS", frame, frame + alive - 1))
        events.append(ev(period, team, "BALL OUT", frame + alive - 1, frame + alive))
        team, frame = metrica.OTHER[team], frame + alive + dead
        events.append(ev(period, team, "SET PIECE", frame, frame + 1, "THROW IN"))
        frame += 1
    return events


def cycle_game(cycles=40):
    return cycle_period(1, 0, cycles) + cycle_period(2, 7000, cycles)


# -- the files --------------------------------------------------------------------------------------------------------

def test_metrica_blob_sha_is_gits():
    # `git hash-object` of an empty file and of "hello\n".
    assert metrica.blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert metrica.blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_metrica_pins_are_r023s():
    assert metrica.COMMIT == "e706dd506b360d69d9d123d5b8026e7294b13996"
    assert metrica.README_BLOB == "03fb97acc3545cc7428d0a07485921c5a1d405c1"
    assert [(f.game, f.kind, f.name, f.blob) for f in metrica.FILES] == [
        (1, "events", "Sample_Game_1_RawEventsData.csv", "3c426dd4f442dd433bccf89e3634f91aa310cd8f"),
        (2, "events", "Sample_Game_2_RawEventsData.csv", "decccd47c1d53960d290cb6679337eec83a8a97f"),
        (3, "events", "Sample_Game_3_events.json", "992b4cb502f0b0bc1244174f014f3f7ecb6d38ef"),
        (3, "metadata", "Sample_Game_3_metadata.xml", "74b55a80f6aeeb8f8df7fd26e4b3a94e2cc1738d")]
    assert not any("tracking" in f.path.lower() for f in metrica.FILES)      # R023 fetches no tracking file


def test_metrica_fetch_keeps_only_pinned_bytes():
    bodies = {i: "file {}\n".format(i).encode() for i in range(4)}
    saved = metrica.FILES
    metrica.FILES = tuple(metrica.RemoteFile(f.game, f.kind, f.path, metrica.blob_sha(bodies[i]))
                          for i, f in enumerate(saved))
    try:
        url = {f.name: metrica.RAW_URL.format(ref=metrica.COMMIT, path=f.path) for f in metrica.FILES}
        with tempfile.TemporaryDirectory() as data_dir:
            opener = FakeOpener({url[f.name]: bodies[i] for i, f in enumerate(metrica.FILES)})
            metrica.fetch(data_dir, opener=opener, log=quiet)
            assert set(metrica.verify(data_dir).values()) == {"ok"} and len(opener.urls) == 4
            metrica.fetch(data_dir, opener=opener, log=quiet)                # all ok: no request
            assert len(opener.urls) == 4
            damaged = metrica.FILES[2].local(data_dir)
            with open(damaged, "wb") as handle:
                handle.write(b"changed")
            assert metrica.verify(data_dir)[metrica.FILES[2].name] == "blob"
            wrong = FakeOpener({u: b"not the pinned bytes" for u in url.values()})
            expect_error(DownloadError, lambda: metrica.fetch(data_dir, opener=wrong, log=quiet),
                         metrica.FILES[2].name, "nothing was kept")
            assert not os.path.exists(damaged + ".part")
            assert open(damaged, "rb").read() == b"changed"                  # untouched: never replaced by bad bytes
    finally:
        metrica.FILES = saved


def test_metrica_licence_stop():
    readme = b"# sample data\n"
    saved = metrica.README_BLOB
    metrica.README_BLOB = metrica.blob_sha(readme)
    try:
        at = {ref: metrica.RAW_URL.format(ref=ref, path="README.md") for ref in (metrica.COMMIT, "HEAD")}
        clean = {at[metrica.COMMIT]: readme, at["HEAD"]: readme, metrica.LICENSE_API: not_found(metrica.LICENSE_API)}
        assert metrica.licence_stop(FakeOpener(clean)) == []
        fired = metrica.licence_stop(FakeOpener(dict(clean, **{at["HEAD"]: b"new terms"})))
        assert len(fired) == 1 and "HEAD" in fired[0] and "read its terms again" in fired[0]
        fired = metrica.licence_stop(FakeOpener(dict(clean, **{metrica.LICENSE_API: b"{}"})))
        assert len(fired) == 1 and "licence file" in fired[0]
        fired = metrica.licence_stop(FakeOpener(dict(clean, **{metrica.LICENSE_API: not_found("x", 403)})))
        assert len(fired) == 1 and "403" in fired[0]
        fired = metrica.licence_stop(FakeOpener(dict(clean, **{at[metrica.COMMIT]: OSError("offline")})))
        assert len(fired) == 1 and "cannot be checked" in fired[0]
    finally:
        metrica.README_BLOB = saved


# -- the readers ------------------------------------------------------------------------------------------------------

CSV_TEXT = (
    "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
    "Away,SET PIECE,KICK OFF,1,1,0.04,1,0.04,Player19,,NaN,NaN,NaN,NaN\n"
    "Home,SHOT,HEAD-ON TARGET-GOAL,1,250,10.0,262,10.48,Player9,,0.9,0.5,1.0,0.5\n"
    "home,PASS,,2,300,12.0,,NaN,Player1,Player2,0.5,0.5,0.6,0.5\n")


def test_metrica_csv_reader():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "events.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write(CSV_TEXT)
        events = metrica.read_csv_events(path)
        assert events == [ev(1, A, "SET PIECE", 1, 1, "KICK OFF", time=0.04),
                          ev(1, H, "SHOT", 250, 262, "HEAD", "ON TARGET", "GOAL", time=10.0),
                          ev(2, H, "PASS", 300, None, time=12.0)]
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:          # a byte-order mark reads the same
            handle.write(CSV_TEXT)
        assert metrica.read_csv_events(path) == events
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write(CSV_TEXT.replace("home,PASS", "Referee,PASS"))
        assert metrica.read_csv_events(path)[2].team is None                       # integrity() fails the game


def test_metrica_json_reader():
    metadata = ("<main><Metadata><Teams><Team id=\"T2\"><Name>Second</Name></Team>"
                "<Team id=\"T1\"><Name>First</Name></Team></Teams></Metadata></main>")
    data = {"data": [
        {"index": 1, "team": {"id": "T2"}, "type": {"id": 5, "name": "SET PIECE"},
         "subtypes": {"id": 35, "name": "KICK OFF"}, "period": 1,
         "start": {"frame": 1, "time": 0.04}, "end": {"frame": 1, "time": 0.04}},
        {"index": 2, "team": {"id": "T1"}, "type": {"id": 2, "name": "shot"},
         "subtypes": [{"id": 28, "name": "ON TARGET"}, {"id": 30, "name": "GOAL"}], "period": 1,
         "start": {"time": 10.0}, "end": {"frame": None, "time": 10.5}},
        {"index": 3, "team": {"id": "T1"}, "type": {"id": 10, "name": "CARRY"}, "subtypes": None, "period": 2,
         "start": {"frame": 300, "time": 12.0}, "end": {"frame": 310, "time": 12.4}}]}
    with tempfile.TemporaryDirectory() as folder:
        events_path, meta_path = os.path.join(folder, "events.json"), os.path.join(folder, "meta.xml")
        with open(meta_path, "w", encoding="utf-8") as handle:
            handle.write(metadata)
        with open(events_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        events = metrica.read_json_events(events_path, meta_path)
        assert events == [
            ev(1, H, "SET PIECE", 1, 1, "KICK OFF", time=0.04, type_id=5, subtype_ids=(35,)),   # the first <Team> is H
            ev(1, A, "SHOT", 250, 262, "ON TARGET", "GOAL", time=10.0, type_id=2, subtype_ids=(28, 30)),
            ev(2, A, "CARRY", 300, 310, time=12.0, type_id=10)]
        data["data"][2]["team"] = {"id": "T9"}
        with open(events_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        assert metrica.read_json_events(events_path, meta_path)[2].team is None   # integrity() fails the game


def test_metrica_kloppy_ids_are_kloppys():
    try:
        from kloppy.infra.serializers.event.metrica import json_deserializer as k
    except ImportError as error:
        raise NotRun("kloppy is not installed ({})".format(error))
    assert metrica.TYPE_IDS == {
        "PASS": k.MS_PASS_OUTCOME_COMPLETE, "SHOT": k.MS_EVENT_TYPE_SHOT, "RECOVERY": k.MS_EVENT_TYPE_RECOVERY,
        "FAULT RECEIVED": k.MS_EVENT_TYPE_FOUL_COMMITTED, "SET PIECE": k.MS_SET_PIECE,
        "BALL OUT": k.MS_PASS_OUTCOME_OUT, "BALL LOST": k.MS_PASS_OUTCOME_INCOMPLETE, "CARD": k.MS_EVENT_TYPE_CARD,
        "CHALLENGE": k.MS_EVENT_TYPE_CHALLENGE, "CARRY": k.MS_EVENT_TYPE_CARRY}


# -- integrity --------------------------------------------------------------------------------------------------------

def test_metrica_integrity_each_failure():
    game = hand_built_game()
    assert metrica.integrity(game) == []
    assert metrica.integrity(game + [ev(3, H, "TACKLE", 1, None)]) == []       # periods other than 1 and 2 are not read

    def fails(changed, *fragments):
        failures = metrica.integrity(changed)
        assert len(failures) == 1 and all(f in failures[0] for f in fragments), failures

    def swap(index, event):
        return game[:index] + [event] + game[index + 1:]

    fails(swap(1, ev(1, H, "PASS", 12, 20, time=12 / 25 + 0.09)), "1 events", "round(start time x 25)")
    assert metrica.integrity(swap(1, ev(1, H, "PASS", 12, 20, time=12 / 25 + 0.05))) == []   # 1.25 frames rounds to 1
    fails(swap(1, ev(1, H, "PASS", 12, None)), "no end frame")
    fails(swap(1, ev(1, H, "TACKLE", 12, 20)), "not one of the PDF's nine or CARRY")
    fails(swap(1, ev(1, H, "PASS", 12, 20, type_id=7)), "type name disagrees with kloppy")
    # R023 cross-checks game 3's type ids only: a subtype's id is never compared.
    assert metrica.integrity(swap(0, ev(1, H, "SET PIECE", 10, 12, "KICK OFF", type_id=5, subtype_ids=(34,)))) == []
    fails(swap(1, ev(1, None, "PASS", 12, 20)), "team cannot be read")
    fails(swap(14, ev(2, A, "PASS", 70, 72)), "period 2 does not begin with a SET PIECE KICK OFF")
    fails(swap(2, ev(1, A, "CHALLENGE", 11, 21)), "period 1: start frames decrease 1 times")
    fails([e for e in game if e.period == 1], "period 2 has no event")


# -- the derivation ---------------------------------------------------------------------------------------------------

def test_metrica_derive_by_hand():
    codes, graded, stops, open_ended = metrica.derive(hand_built_game())
    by_frame = np.full(105, X, dtype=np.int8)                                   # in the game's own frame numbers
    for first, last, code in ((10, 24, H), (25, 39, A), (40, 44, D), (45, 49, H), (50, 54, D), (55, 61, H),
                              (62, 64, D),                                       # period 1: frames 10 to 64
                              (70, 79, A), (80, 89, H), (90, 94, D), (95, 103, A), (104, 104, D)):   # 70 to 104
        by_frame[first:last + 1] = code
    expected = by_frame[10:65].tolist() + [X] + by_frame[70:105].tolist()       # the spans end to end, one X between
    assert codes.dtype == np.int8 and codes.tolist() == expected
    assert graded == ((1, 0, 55), (2, 56, 91))
    assert stops == {"ball out": 1, "fault received": 1, "challenge fault": 1, "goal": 1, "offside": 1, "end half": 1}
    assert open_ended == 2                                                       # the goal, and the end of the half


def test_metrica_derive_flips_and_stops_by_rule():
    kick = ev(1, H, "SET PIECE", 0, 1, "KICK OFF")
    tail = [ev(2, H, "SET PIECE", 100, 101, "KICK OFF"), ev(2, H, "PASS", 101, 110)]

    def period_one(*events):
        codes, _, stops, _ = metrica.derive([kick, *events, *tail])
        return codes[:events[-1].end_frame + 1].tolist(), stops

    # The other team's SET PIECE, PASS, SHOT, BALL LOST, BALL OUT or CARRY changes the team at its start; CHALLENGE,
    # FAULT RECEIVED (a stop, though) and CARD do not.
    for kind in ("SET PIECE", "PASS", "SHOT", "BALL OUT", "CARRY"):
        codes, _ = period_one(ev(1, A, kind, 5, 9), ev(1, A, "PASS", 9, 12))
        assert codes[4:6] == [H, A], kind
    for kind in ("CHALLENGE", "CARD"):
        codes, _ = period_one(ev(1, A, kind, 5, 9), ev(1, H, "PASS", 9, 12))
        assert set(codes[:12]) == {H}, kind
    # A RECOVERY changes the team; a CHALLENGE with FAULT stops play at its start; a SHOT OUT stops at its end; GOAL
    # KICK is not GOAL; a stop with a CHALLENGE or CARD after it waits for a restart type.
    codes, stops = period_one(ev(1, H, "PASS", 1, 5), ev(1, A, "SHOT", 5, 8, "OFF TARGET", "OUT"),
                              ev(1, A, "CARD", 9, 9, "YELLOW"), ev(1, H, "PASS", 12, 14, "GOAL KICK"))
    assert codes[5:8] == [A, A, A] and codes[8:12] == [D] * 4 and codes[12] == H
    assert stops == {"shot out": 1}
    codes, stops = period_one(ev(1, H, "PASS", 1, 5), ev(1, A, "CHALLENGE", 5, 6, "FAULT"),
                              ev(1, A, "RECOVERY", 9, 9), ev(1, A, "PASS", 9, 12))
    assert codes[4] == H and codes[5:9] == [D] * 4 and codes[9:12] == [A] * 3
    assert stops == {"challenge fault": 1}
    # A restart must start after the stop: a PASS at the stop's own frame does not close it.
    codes, _ = period_one(ev(1, H, "BALL OUT", 1, 5), ev(1, A, "PASS", 5, 7), ev(1, A, "PASS", 8, 10))
    assert codes[5:8] == [D] * 3 and codes[8] == A


def test_metrica_derive_rules_the_first_checks_left_out():
    kick = ev(1, H, "SET PIECE", 0, 1, "KICK OFF")
    tail = [ev(2, H, "SET PIECE", 100, 101, "KICK OFF"), ev(2, H, "PASS", 101, 110)]

    def period_one(*events):
        codes, _, stops, open_ended = metrica.derive([kick, *events, *tail])
        return codes[:events[-1].end_frame + 1].tolist(), stops, open_ended

    # FORCED stops at its end frame; BALL OUT, CHALLENGE, FAULT RECEIVED and CARD never close D: the PASS does.
    codes, stops, _ = period_one(ev(1, H, "PASS", 1, 5, "FORCED"), ev(1, A, "BALL OUT", 6, 7),
                                 ev(1, A, "CHALLENGE", 7, 8), ev(1, H, "FAULT RECEIVED", 8, 8), ev(1, A, "CARD", 9, 9),
                                 ev(1, A, "PASS", 11, 14))
    assert codes[5:11] == [D] * 6 and codes[11:14] == [A] * 3
    assert stops == {"forced": 1, "ball out": 1, "fault received": 1}
    # SHOT, BALL LOST and CARRY close D like the declared restarts.
    for kind in ("SHOT", "BALL LOST", "CARRY"):
        codes, _, _ = period_one(ev(1, H, "BALL OUT", 1, 5), ev(1, A, kind, 8, 12))
        assert codes[5:8] == [D] * 3 and codes[8] == A, kind
    # A BALL LOST by the other team: (a) changes to it at its start, then (b) back at its end.
    codes, _, _ = period_one(ev(1, H, "PASS", 1, 5), ev(1, A, "BALL LOST", 5, 9), ev(1, H, "PASS", 9, 12))
    assert codes[4:10] == [H, A, A, A, A, H]
    # (b) with t later than the event's end: H's BALL LOST sets A at 9; A's, ending at 6, sets H at max(6, 9) = 9, the
    # later change at that frame, so A never shows; the PASS sets A at 10.
    codes, _, _ = period_one(ev(1, H, "BALL LOST", 2, 9), ev(1, A, "BALL LOST", 3, 6), ev(1, A, "PASS", 10, 12))
    assert codes[:10] == [H] * 10 and codes[10:13] == [A] * 3
    # A FAULT RECEIVED by the other team stops play and changes no team: the restart's team decides.
    codes, _, _ = period_one(ev(1, H, "PASS", 1, 5), ev(1, A, "FAULT RECEIVED", 5, 5), ev(1, H, "SET PIECE", 9, 12))
    assert codes[4] == H and codes[5:9] == [D] * 4 and codes[9:12] == [H] * 3
    # A stop with no restart runs to the span's end; one past the span's end gives no D (clipped).
    codes, stops, open_ended = period_one(ev(1, H, "PASS", 1, 30), ev(1, H, "PASS", 4, 6, "OFFSIDE"))
    assert stops == {"offside": 1} and open_ended == 1 and codes == [H] * 6 + [D]
    codes, _, open_ended = period_one(ev(1, H, "PASS", 1, 30, "OFFSIDE"), ev(1, H, "PASS", 4, 6))
    assert open_ended == 1 and codes == [H] * 7


def test_metrica_derive_reads_a_period_numbered_from_zero_the_same():
    game = hand_built_game()
    again = [e if e.period == 1 else ev(2, e.team, e.type, e.start_frame - 70, e.end_frame - 70, *e.subtypes)
             for e in game]
    assert metrica.derive(again)[0].tolist() == metrica.derive(game)[0].tolist()


# -- the decision -----------------------------------------------------------------------------------------------------

def test_metrica_decide_at_its_edges():
    assert [round(v, 2) for v in metrica._band(metrica.RAW_RATE, metrica.COARSE_RATE)] == [5.31, 7.03]
    assert [round(v, 2) for v in metrica._band(metrica.RAW_MEDIAN_S, metrica.COARSE_MEDIAN_S)] == [4.83, 6.39]
    rate_mid = (8.30 * 4.50) ** 0.5
    for value, expected in ((8.30, "raw"), (4.50, "coarse"), (rate_mid, "both"), (rate_mid * 1.149, "both"),
                            (rate_mid * 1.151, "raw"), (rate_mid / 1.149, "both"), (rate_mid / 1.151, "coarse")):
        assert metrica.classify(value, 8.30, 4.50, larger_is_raw=True) == expected, value
    for value, expected in ((3.68, "raw"), (8.40, "coarse"), (5.56, "both"), (4.80, "raw"), (6.42, "coarse")):
        assert metrica.classify(value, 3.68, 8.40, larger_is_raw=False) == expected, value
    for rate, median, outcome in ((8.0, 3.0, "raw"), (8.0, 6.0, "raw"), (8.0, 9.0, "both"), (4.0, 3.0, "both"),
                                  (4.0, 9.0, "coarse"), (4.0, 6.0, "coarse"), (6.0, 3.0, "both"), (6.0, 9.0, "both")):
        assert metrica.decide(rate, median)["outcome"] == outcome, (rate, median)


def test_metrica_measure_drops_and_invalidates():
    good = cycle_game()
    bad = [ev(e.period, e.team, e.type, e.start_frame, None, *e.subtypes) for e in good]   # no end frames
    result = metrica.measure({1: good, 2: good, 3: bad})
    assert result["valid"] and result["outcome"] != "invalid"
    assert result["games"]["game_3"]["integrity"] and "summary" not in result["games"]["game_3"]
    one = metrica.derive(good)
    assert result["pooled"] == truth.summary(*truth.concatenate([one[:2], one[:2]]), 25)
    assert result["games"]["game_1"]["summary"]["stoppages"] == 80
    assert abs(result["games"]["game_1"]["summary"]["alive_share"] - 2 / 3) < 0.01
    assert result["outcome"] == metrica.decide(result["pooled"]["per_alive_minute"], result["pooled"]["median_s"])[
        "outcome"]

    result = metrica.measure({1: good, 2: bad, 3: bad})
    assert not result["valid"] and result["outcome"] == "invalid"
    assert "1 game(s) passed integrity, fewer than 2" in result["invalid_because"][0]

    few = cycle_game(cycles=20)                                    # 40 stoppages: outside 60 to 160, not dropped
    result = metrica.measure({1: good, 2: few})
    assert result["outcome"] == "invalid" and "summary" in result["games"]["game_2"]
    assert result["invalid_because"] == ["game 2: 40 stoppages, outside 60 to 160"]
    lively = [e if e.type != "BALL OUT" else ev(e.period, e.team, "PASS", e.start_frame, e.end_frame) for e in good]
    result = metrica.measure({1: good, 2: lively})                 # no stop at all: always alive
    assert result["outcome"] == "invalid" and "alive share 1.000" in result["invalid_because"][0]
    # The bounds are inclusive: 60 stoppages (30 a period) pass.
    assert metrica.measure({1: good, 2: cycle_game(cycles=30)})["valid"]
    lines = metrica.render(metrica.measure({1: good, 2: few}))
    assert "decide nothing" in [line for line in lines if line.startswith("classes")][0]


def test_metrica_prints_aggregates_and_reads_nothing_of_ours():
    lines = metrica.render(metrica.measure({1: cycle_game(), 2: cycle_game()}))
    assert lines[-1].startswith("outcome ") and any(line.startswith("pooled ") for line in lines)
    assert all(line.split()[0] in ("game_1", "game_2", "pooled", "classes", "outcome") for line in lines)
    with open(metrica.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for fragment in ("asset_paths", "require(", "pipeline.inputs", "BAR-ATH\"", "evaluation/gt", "public_gt"):
        assert fragment not in source, fragment
