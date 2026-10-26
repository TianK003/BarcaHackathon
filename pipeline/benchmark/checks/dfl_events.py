"""Checks of dfl_events (pipeline/benchmark/dfl_events.py): the DFL restart events on the benchmark's timeline.

dfl_events puts each restart that DFL's event file names (kick-off, throw-in, corner, goal kick, free kick, penalty)
on a row of the timeline that dfl.py lays out, by the frame timestamps: the row of a time t is the half's first row
plus (t - T of the half's first ball frame) / 40 ms. A mistake in that mapping would shift or lose every restart, so
what is held against it is a small file in the real format whose timestamps are worked out by hand (rounding, the
kick-off's team read from its Play, an event in the gap and one before the half dropped, the output sorted by row, a
time zone that is not UTC), what it refuses (a team that is no club of the match, a frame clock that is not 40 ms
a frame, a missing ball), and, with --dfl, the seven matches: every restart inside a half, the counts per kind, and
that the restarts sit next to the truth's dead-to-alive changes, which is the sanity gate on the alignment.
"""

import tempfile
from datetime import datetime, timedelta

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import dfl, dfl_events, sources, truth
from pipeline.benchmark.checks.common import real_match, write_synthetic_dfl, xml_attributes
from pipeline.selftest import expect_error, write_text

# Params
FIRST_T = "2024-01-01T12:00:00.000+00:00"    # the synthetic first half's first ball frame (UTC)
SECOND_T = "2024-01-01T12:20:00.000+00:00"   # and the second half's
FIRST_EPOCH_MS = 1704110400000               # FIRST_T in ms since the epoch
SECOND_EPOCH_MS = 1704111600000              # SECOND_T
HALF_ROWS = 100                              # ball frames per half
GRADED = ((1, 0, HALF_ROWS), (2, HALF_ROWS + 25, 2 * HALF_ROWS + 25))
HOME_CLUB, AWAY_CLUB = "DFL-CLU-SYNHOM", "DFL-CLU-SYNAWY"
SANITY_SECONDS = 2                           # a restart is "next to" a dead-to-alive change within this many seconds
SANITY_MINIMUM = 0.5                         # and at least this share of the restarts must be


def frameset(section, team, person, n0, t0, step_ms):
    start = datetime.fromisoformat(t0)
    frames = "\n".join("<Frame {} />".format(xml_attributes({
        "N": n0 + k, "T": (start + timedelta(milliseconds=step_ms * k)).isoformat(timespec="milliseconds"),
        "X": "1.0", "Y": "1.0"})) for k in range(HALF_ROWS))
    return '<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">\n{}\n</FrameSet>'.format(
        section, team, person, frames)


def positions_text(step_ms=40, with_second_ball=True):
    """A positions file with only what the time scan reads: a referee's frames to skip, and the balls (second half first)."""
    sets = [frameset("firstHalf", "referee", "DFL-OBJ-SYNR01", 10000, "2024-01-01T11:59:00.000+00:00", step_ms)]
    if with_second_ball:
        sets.append(frameset("secondHalf", "BALL", "DFL-OBJ-SYNBAL", 100000, SECOND_T, step_ms))
    sets.append(frameset("firstHalf", "BALL", "DFL-OBJ-SYNBAL", 10000, FIRST_T, step_ms))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<PutDataRequest><Positions>\n{}\n'
            '</Positions></PutDataRequest>\n'.format("\n".join(sets)))


def event(time, kind, team, play_team=None, section=None):
    """One <Event> as DFL writes it; a kick-off names no team on itself, only in its Play."""
    if kind == "KickOff":
        body = '<KickOff TeamLeft="{}" TeamRight="{}"{}><Play Team="{}"/></KickOff>'.format(
            HOME_CLUB, AWAY_CLUB, "" if section is None else ' GameSection="{}"'.format(section), play_team)
    elif kind == "Foul":
        body = '<Foul Team="{}"/>'.format(team)
    else:
        body = '<{} Team="{}"><Play Team="{}"/></{}>'.format(kind, team, team, kind)
    return '<Event MatchId="DFL-MAT-SYN001" EventTime="{}" EventId="1">{}</Event>'.format(time, body)


def events_text(*events):
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n<PutDataRequest>\n{}\n</PutDataRequest>\n'.format(
        "\n".join(events))


def build(folder, events, **options):
    """dfl_events.build_restarts on the synthetic match's information and a positions file written here."""
    paths = write_synthetic_dfl(folder)
    positions = write_text(folder, "positions_with_time.xml", positions_text(**options))
    events_path = write_text(folder, "events.xml", events_text(*events))
    info = dfl.parse_match_information(paths["information"])
    return dfl_events.build_restarts(events_path, positions, info, GRADED)


def test_restart_events_parse_a_synthetic_file():
    # The clock is +02:00 here, the frames' UTC: 14:00:01.000+02:00 is 12:00:01.000 UTC, 25 frames after the first.
    events = (
        event("2024-01-01T14:00:02.040+02:00", "ThrowIn", HOME_CLUB),            # row 51, home
        event("2024-01-01T14:00:01.000+02:00", "KickOff", None, play_team=AWAY_CLUB, section="firstHalf"),  # row 25
        event("2024-01-01T14:00:01.510+02:00", "Foul", HOME_CLUB),               # not a restart: ignored
        event("2024-01-01T14:00:03.019+02:00", "GoalKick", AWAY_CLUB),           # 75.475 frames in: row 75, away
        event("2024-01-01T14:00:03.021+02:00", "CornerKick", HOME_CLUB),         # 75.525 frames in: row 76
        event("2024-01-01T14:20:00.080+02:00", "KickOff", None, play_team=HOME_CLUB),    # second half, row 127
        event("2024-01-01T14:20:03.800+02:00", "FreeKick", AWAY_CLUB),           # 95 frames in: row 220
        event("2024-01-01T14:20:04.000+02:00", "Penalty", HOME_CLUB),            # 100 frames in: row 225, outside
        event("2024-01-01T14:10:00.000+02:00", "ThrowIn", HOME_CLUB),            # half time: in no half
        event("2024-01-01T13:59:59.000+02:00", "ThrowIn", AWAY_CLUB))            # before the first frame
    with tempfile.TemporaryDirectory() as folder:
        found = build(folder, events)
    assert list(found["row"]) == [25, 51, 75, 76, 127, 220], found["row"]
    assert list(found["kind"]) == ["KickOff", "ThrowIn", "GoalKick", "CornerKick", "KickOff", "FreeKick"], found["kind"]
    assert list(found["team"]) == [1, 0, 1, 0, 0, 1], found["team"]
    assert found["row"].dtype == np.int64 and found["team"].dtype == np.int8
    assert found["outside"] == 3, found["outside"]
    for name in ("row", "kind", "team"):
        assert not found[name].flags.writeable, name


def test_restart_events_refuse_what_they_cannot_place():
    with tempfile.TemporaryDirectory() as folder:
        odd_team = (event("2024-01-01T14:00:02.000+02:00", "ThrowIn", "DFL-CLU-SOMEONE"),)
        expect_error(ValueError, lambda: build(folder, odd_team), "events.xml", "ThrowIn", "DFL-CLU-SOMEONE")
        no_time = ('<Event MatchId="DFL-MAT-SYN001" EventId="1"><ThrowIn Team="{}"/></Event>'.format(HOME_CLUB),)
        expect_error(ValueError, lambda: build(folder, no_time), "events.xml", "EventTime")
        fine = (event("2024-01-01T14:00:02.000+02:00", "ThrowIn", HOME_CLUB),)
        expect_error(ValueError, lambda: build(folder, fine, step_ms=41), "positions_with_time.xml", "40")
        expect_error(ValueError, lambda: build(folder, fine, with_second_ball=False), "positions_with_time.xml",
                     "secondHalf")


def test_restart_events_scan_the_ball_times_only():
    with tempfile.TemporaryDirectory() as folder:
        path = write_text(folder, "p.xml", positions_text())
        scanned = dfl_events.scan_ball_times(path)
    last = 40 * (HALF_ROWS - 1)
    assert scanned == {1: (10000, FIRST_EPOCH_MS, 10000 + HALF_ROWS - 1, FIRST_EPOCH_MS + last),
                       2: (100000, SECOND_EPOCH_MS, 100000 + HALF_ROWS - 1, SECOND_EPOCH_MS + last)}, scanned


def test_dfl_events_rows_fall_in_the_halves_dfl():
    totals = {}
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        found = dfl_events.load_restarts(match_id)
        rows = found["row"]
        assert rows.size > 0, match_id
        inside = np.zeros(rows.size, dtype=bool)
        for _, start, stop in match.graded:
            inside |= (rows >= start) & (rows < stop)
        assert inside.all(), "{}: {} restart rows lie outside both halves".format(match_id, int((~inside).sum()))
        assert (np.diff(rows) >= 0).all(), match_id
        assert set(found["team"].tolist()) <= {0, 1}, match_id
        for kind in dfl_events.KINDS:
            totals[kind] = totals.get(kind, 0) + int((found["kind"] == kind).sum())
        print("  {}: {} restarts ({} outside the halves, dropped)".format(match_id, rows.size, found["outside"]))
    print("  per kind over the 7 matches: " + ", ".join("{} {}".format(k, totals.get(k, 0)) for k in dfl_events.KINDS))
    assert totals["KickOff"] >= 14 and all(totals[kind] > 0 for kind in dfl_events.KINDS), totals


def test_dfl_events_restarts_meet_the_truths_stoppages_dfl():
    near, total = 0, 0
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        codes, _ = truth.truth_codes(match)
        ends = np.flatnonzero((codes[:-1] == pm.D) & ((codes[1:] == pm.H) | (codes[1:] == pm.A))) + 1
        rows = dfl_events.load_restarts(match_id)["row"]
        at = np.searchsorted(ends, rows)
        before = np.abs(rows - ends[np.clip(at - 1, 0, ends.size - 1)])
        after = np.abs(rows - ends[np.clip(at, 0, ends.size - 1)])
        near += int((np.minimum(before, after) <= SANITY_SECONDS * match.fps).sum())
        total += rows.size
    share = near / total
    print("  restarts within {} s of a true dead-to-alive change: {} of {} = {:.3f}".format(
        SANITY_SECONDS, near, total, share))
    assert share >= SANITY_MINIMUM, "the alignment looks off: only {:.3f} of the restarts are near the end of a " \
                                    "stoppage".format(share)
