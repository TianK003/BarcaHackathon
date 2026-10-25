"""Checks of the conventions instrument (pipeline/benchmark/conventions.py): when does the truth's label change?

The instrument times every change of a truth's label against label-free events that our own tracking can see, on the
public window of our match and on the seven DFL matches, with the same code. A mistake in it would put a wrong
convention into the shift rule that P02 may learn from it, so what is held against it here is a plain loop over the
frames (the window rule: nearest event, the earlier on a tie, never past a neighbouring change or a block edge), the
detectors on hand-built tracking whose answer is known by construction, that the DFL mode serves its inputs as the
trainer does and reads the truth beside them (the deep walk of common.leaks), that the public mode prints aggregates
and never a transition frame, and that the bootstrap of the gap is deterministic and holds a gap known in advance.

Everything runs on hand-built tracking and on the synthetic matches of checks/common.py: none reads data, and none of
the numbers printed is a figure of any match.
"""

import contextlib
import csv
import io
import json
import math
import os
import re
import tempfile

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import conventions, train_dead_ball, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import (leaks, synthetic_calibration, synthetic_matches, tracked_match,
                                              tracking_calibration)
from pipeline.contracts import RunContext
from pipeline.evidence import dead_ball_model, kinematics, play_features
from pipeline.selftest import beside_ball, expect_error, kinematics_atd, kinematics_inputs, quiet

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
REACH = 125                                     # 5 s at 25 Hz
PUBLIC_TRANSITIONS = (1300, 1500, 1620, 2000, 2050, 2400)     # four digits, so that a printed frame is found
MAX_PUBLIC_LINES = 20                           # what the public mode may print: a summary, never a table of rows


# -- helpers ----------------------------------------------------------------------------------------------------------

def codes_of(n, pieces):
    """An int8 code array of n frames: X, then [(start, stop, code)] pieces laid over it."""
    codes = np.full(n, X, dtype=np.int8)
    for start, stop, code in pieces:
        codes[start:stop] = code
    return codes


def found(frames, team=None):
    """An event entry as events() gives it."""
    return {"frames": np.asarray(frames, dtype=np.int64), "team": None if team is None else np.asarray(team, np.int8)}


def plain_lags(codes, runs, events, fps=25):
    """{(type, detector): (n, [lag, ...])} worked out one frame at a time, from the window rule alone.

    A change is a frame whose code differs from the one before inside a run. For the change at c, the window is
    [max(c - 5 s, previous change or the run's first frame), min(c + 5 s, next change or the run's last frame)]; of the
    event frames in it (and, for an event with a team, only those of the new team) the nearest to c is taken, the
    earlier one on a tie.
    """
    kinds = {(H, A): "team", (A, H): "team", (H, D): "into_dead", (A, D): "into_dead", (D, H): "out_of_dead",
             (D, A): "out_of_dead"}
    applies = {"touch_start": ("team", "out_of_dead"), "line_out": ("into_dead",), "ball_lost": ("into_dead",),
               "ball_moves": ("out_of_dead",), "model_dead": ("into_dead",), "model_alive": ("out_of_dead",),
               "tag_start": ("into_dead",), "tag_end": ("out_of_dead",)}
    out = {(kind, name): [0, []] for name in events for kind in applies[name]}
    for start, stop in runs:
        changes = [f for f in range(start + 1, stop) if codes[f] != codes[f - 1]]
        for number, c in enumerate(changes):
            kind = kinds[(int(codes[c - 1]), int(codes[c]))]
            low = max(c - round(5 * fps), changes[number - 1] if number else start)
            high = min(c + round(5 * fps), changes[number + 1] if number + 1 < len(changes) else stop - 1)
            for name, entry in events.items():
                if kind not in applies[name]:
                    continue
                cell = out[(kind, name)]
                cell[0] += 1
                best = None
                wanted = {int(f) for i, f in enumerate(entry["frames"])
                          if entry["team"] is None or int(entry["team"][i]) == int(codes[c])}
                for s in range(low, high + 1):
                    if s in wanted and (best is None or abs(s - c) < abs(best - c)):
                        best = s
                if best is not None:
                    cell[1].append(best - c)
    return {key: (n, lag) for key, (n, lag) in out.items()}


def scene(n, ball, slots=None, graded=None, tags=None, game="SCENE"):
    """A stand-in Inputs over n frames of hand-built tracking in metres (the shape of pipeline.selftest.touch_scene).

    `ball` is (n, 2) (NaN = missing), `slots` a dict slot -> (n, 2) (NaN = absent; slot k is the home team's when k is
    even), `graded` a list of (start, stop) blocks (default one over every frame), `tags` the Smart Tagging intervals.
    """
    ball = np.asarray(ball, dtype=np.float64)
    count = max(max(slots) + 1 if slots else 2, 2)
    players = np.full((n, count, 2), np.nan)
    for slot, track in (slots or {}).items():
        players[:, slot] = track
    graded = [(1, 0, n)] if graded is None else [(index + 1, start, stop) for index, (start, stop) in enumerate(graded)]
    scale = np.array([105.3, 68.0])
    inputs = kinematics_inputs(kinematics_atd(players / scale, ball / scale), graded)
    inputs.game, inputs.kickoffs, inputs.n_frames, inputs.data_dir = game, {}, n, None
    inputs.smart_tagging = [] if tags is None else tags
    return inputs


def still_ball(n, at=(50.0, 30.0)):
    return np.tile(at, (n, 1))


def numbers_in(text):
    return [int(token) for token in re.findall(r"\d+", text)]


# -- the window rule --------------------------------------------------------------------------------------------------

def test_conventions_transition_types():
    codes = codes_of(900, [(100, 200, H), (200, 300, A), (300, 400, D), (400, 500, A), (500, 600, H), (600, 700, D),
                           (700, 800, H), (800, 810, D)])
    runs = [(100, 810)]
    found_ = conventions.transitions_of(codes, runs)
    assert found_ == [(200, H, A, "team"), (300, A, D, "into_dead"), (400, D, A, "out_of_dead"),
                      (500, A, H, "team"), (600, H, D, "into_dead"), (700, D, H, "out_of_dead"),
                      (800, H, D, "into_dead")], found_
    assert conventions.TYPES == ("team", "into_dead", "out_of_dead")
    assert all(isinstance(frame, int) and isinstance(a, int) and isinstance(b, int) for frame, a, b, _ in found_)
    two = codes_of(900, [(100, 200, H), (200, 300, A), (600, 700, D), (700, 800, H)])
    assert [(f, kind) for f, _, _, kind in conventions.transitions_of(two, [(100, 300), (600, 800)])] == [
        (200, "team"), (700, "out_of_dead")], "a run's edge is never a change, and nothing spans two runs"
    inside = codes_of(900, [(100, 200, H), (200, 300, X), (300, 400, A)])
    expect_error(ValueError, lambda: conventions.transitions_of(inside, [(100, 400)]), "conventions", "X")


def test_conventions_lag_rule():
    """The sign, the nearest event, the earlier on a tie, the neighbours and the edges, the team, and not found."""
    codes = codes_of(1000, [(100, 300, H), (300, 500, A), (500, 620, D), (620, 900, H)])
    runs = [(100, 900)]
    transitions = conventions.transitions_of(codes, runs)
    assert [t[0] for t in transitions] == [300, 500, 620]

    def lag_of(name, frames, index, team=None):
        table = conventions.lags(transitions, {name: found(frames, team)}, runs, 25)
        kind = transitions[index][3]
        cell = table[(kind, name)]
        assert cell["n"] == sum(1 for t in transitions if t[3] == kind)
        return cell["found"], cell["lags"]

    # sign: an event after the change is positive, before it negative; the nearest of several
    assert lag_of("touch_start", [270, 305, 330], 0, team=[A, A, A]) == (1, [5])
    assert lag_of("touch_start", [296, 340], 0, team=[A, A]) == (1, [-4])
    # the earlier one on a tie
    assert lag_of("touch_start", [295, 305], 0, team=[A, A]) == (1, [-5])
    assert lag_of("touch_start", [305, 295], 0, team=[A, A]) == (1, [-5]), "the order of the list is not the tie rule"
    # only the new team's events count for an event that has a team
    assert lag_of("touch_start", [298, 310], 0, team=[H, A]) == (1, [10])
    assert lag_of("touch_start", [298, 310], 0, team=[H, H]) == (0, [])
    # the reach: 5 s either side, inclusive
    assert lag_of("touch_start", [300 + REACH], 0, team=[A]) == (1, [REACH])
    assert lag_of("touch_start", [300 + REACH + 1], 0, team=[A]) == (0, [])
    assert lag_of("touch_start", [300 - REACH, 300 - REACH - 1], 0, team=[A, A]) == (1, [-REACH])
    assert lag_of("touch_start", [300 - REACH - 1], 0, team=[A]) == (0, [])
    # never beyond the next change: the window of the change at 500 ends at the change at 620, which is inclusive, and
    # 621 would win (121 away) over 376 (124 away) without that rule
    assert lag_of("line_out", [621], 1) == (0, [])
    assert lag_of("line_out", [620], 1) == (1, [120])
    assert lag_of("line_out", [621, 376], 1) == (1, [-124])
    # never before the previous change: the window of the change at 620 starts at the change at 500, inclusive, and
    # 499 would win (121 away) over 745 (125 away) without that rule
    assert lag_of("ball_moves", [499], 2) == (0, [])
    assert lag_of("ball_moves", [500], 2) == (1, [-120])
    assert lag_of("ball_moves", [499, 745], 2) == (1, [REACH])
    # the run's edges: the first change's window starts at the run's first frame, the last one's ends at its last
    tight = codes_of(1000, [(100, 150, H), (150, 160, D), (160, 900, H)])
    edge = conventions.transitions_of(tight, [(100, 900)])
    assert [(t[0], t[3]) for t in edge] == [(150, "into_dead"), (160, "out_of_dead")]

    def at_edge(name, frames, kind, transitions_=edge, runs_=((100, 900),)):
        return conventions.lags(transitions_, {name: found(frames)}, list(runs_), 25)[(kind, name)]["lags"]
    assert at_edge("line_out", [99], "into_dead") == [], "99 is before the run, though 51 frames from the change"
    assert at_edge("line_out", [100], "into_dead") == [-50], "the run's first frame is in the window"
    assert at_edge("line_out", [99, 140], "into_dead") == [-10]
    last = codes_of(1000, [(100, 880, H), (880, 900, D)])
    end = conventions.transitions_of(last, [(100, 900)])
    assert at_edge("line_out", [899], "into_dead", end) == [19], "899 is the run's last frame"
    assert at_edge("line_out", [900], "into_dead", end) == [], "900 is outside the run"
    # not found counts in n and not in found; an empty detector finds nothing anywhere
    cells = conventions.lags(transitions, {"line_out": found([]), "touch_start": found([310], [A])}, runs, 25)
    assert cells[("into_dead", "line_out")] == {"n": 1, "found": 0, "lags": []}
    assert cells[("team", "touch_start")] == {"n": 1, "found": 1, "lags": [10]}
    assert cells[("out_of_dead", "touch_start")] == {"n": 1, "found": 0, "lags": []}, "310 is not near 620"
    # only the pairs that the design lists exist
    assert sorted(cells) == [("into_dead", "line_out"), ("out_of_dead", "touch_start"), ("team", "touch_start")]
    # a transition outside every run is refused, and a detector the design does not list
    expect_error(ValueError, lambda: conventions.lags([(950, H, A, "team")], {"line_out": found([])}, runs, 25),
                 "conventions", "run")
    expect_error(ValueError, lambda: conventions.lags(transitions, {"magic": found([])}, runs, 25), "magic")


def test_conventions_lag_rule_matches_a_plain_loop():
    """On random codes in three blocks and random events, every cell equals the frame-by-frame reading of the rule."""
    rng = np.random.default_rng(3)
    n = 3000
    runs = [(50, 900), (1000, 1900), (2000, 2900)]
    codes = np.full(n, X, dtype=np.int8)
    for start, stop in runs:
        position = start
        while position < stop:
            length = int(rng.integers(10, 130))
            codes[position:min(position + length, stop)] = rng.integers(0, 3)
            position += length
    transitions = conventions.transitions_of(codes, runs)
    assert len(transitions) > 12
    events = {name: found(np.sort(rng.choice(n, size=int(rng.integers(5, 120)), replace=False)))
              for name in conventions.DETECTORS if name != "touch_start"}
    events["touch_start"] = found(np.sort(rng.choice(n, size=150, replace=False)), rng.integers(0, 2, size=150))
    table = conventions.lags(transitions, events, runs, 25)
    wanted = plain_lags(codes, runs, events)
    assert sorted(table) == sorted(wanted)
    some_found = some_missed = False
    for key, (count, lag) in wanted.items():
        assert table[key]["n"] == count and table[key]["lags"] == lag and table[key]["found"] == len(lag), key
        some_found |= bool(lag)
        some_missed |= len(lag) < count
    assert some_found and some_missed, "the random events must exercise both the found and the not-found branch"
    per_transition = conventions.transition_lags(transitions, events, runs, 25)
    assert len(per_transition) == len(transitions)
    for (frame, _, _, kind), row in zip(transitions, per_transition):
        assert sorted(row) == sorted(name for name in events if kind in conventions.APPLIES[name]), (frame, kind)
    for key, cell in table.items():
        kind, name = key
        pooled = [row[name] for (_, _, _, k), row in zip(transitions, per_transition) if k == kind
                  and row[name] is not None]
        assert pooled == cell["lags"], key


def test_conventions_summary_and_pooling():
    table = {("team", "touch_start"): {"n": 5, "found": 4, "lags": [-10, -2, 4, 20]},
             ("into_dead", "line_out"): {"n": 3, "found": 0, "lags": []},
             ("out_of_dead", "ball_moves"): {"n": 0, "found": 0, "lags": []}}
    summary = conventions.summarise(table)
    cell = summary[("team", "touch_start")]
    assert (cell["n"], cell["found"], cell["share_found"]) == (5, 4, 0.8)
    assert (cell["median"], cell["p25"], cell["p75"]) == (1.0, -4.0, 8.0)
    assert (cell["median_s"], cell["p25_s"], cell["p75_s"]) == (1.0 / 25, -4.0 / 25, 8.0 / 25)
    nothing = summary[("into_dead", "line_out")]
    assert nothing["n"] == 3 and nothing["found"] == 0 and nothing["share_found"] == 0.0
    assert nothing["median"] is None and nothing["p25"] is None and nothing["median_s"] is None
    assert summary[("out_of_dead", "ball_moves")]["share_found"] is None, "no transition: no share"
    assert conventions.summarise(table, fps=50)[("team", "touch_start")]["median_s"] == 1.0 / 50
    # pooled by the lags themselves, never as a median of medians
    first = {("team", "touch_start"): {"n": 2, "found": 2, "lags": [1, 2]}}
    second = {("team", "touch_start"): {"n": 4, "found": 3, "lags": [100, 200, 300]},
              ("into_dead", "line_out"): {"n": 1, "found": 1, "lags": [7]}}
    pooled = conventions.pool([first, second])
    assert pooled[("team", "touch_start")] == {"n": 6, "found": 5, "lags": [1, 2, 100, 200, 300]}
    assert pooled[("into_dead", "line_out")] == {"n": 1, "found": 1, "lags": [7]}
    assert conventions.summarise(pooled)[("team", "touch_start")]["median"] == 100.0
    assert first[("team", "touch_start")]["lags"] == [1, 2], "pool must not change what it is given"


# -- the detectors ----------------------------------------------------------------------------------------------------

def test_conventions_detectors():
    n = 400
    # line_out: the first frame of each run of visible frames beyond a line; a line itself is not beyond it
    ball = still_ball(n)
    ball[100:120, 1] = -1.0                  # 1 m beyond the touchline
    ball[150:160] = np.nan                   # a gap: nothing is seen, so nothing is outside
    ball[200:210, 0] = 106.0                 # beyond the goal line
    ball[300:306, 1] = 69.0
    ball[303:305] = np.nan                   # a gap splits the run in two
    ball[350:360, 1] = 68.0                  # on the line, not beyond it
    events = conventions.events(scene(n, ball))
    assert events["line_out"]["frames"].tolist() == [100, 200, 300, 305], events["line_out"]["frames"]
    assert events["line_out"]["team"] is None
    assert events["line_out"]["frames"].dtype.kind == "i"

    # ball_lost: the first missing frame of each gap that opens inside a block
    ball = still_ball(n)
    ball[0:5] = np.nan                       # missing from the block's first frame: nothing was lost then
    ball[150:160] = np.nan
    ball[303:305] = np.nan
    ball[395:400] = np.nan                   # a gap that runs to the block's end still opens inside it
    events = conventions.events(scene(n, ball))
    assert events["ball_lost"]["frames"].tolist() == [150, 303, 395], events["ball_lost"]["frames"]
    two = conventions.events(scene(n, ball, graded=[(0, 200), (250, 400)]))
    assert two["ball_lost"]["frames"].tolist() == [150, 303, 395], "a block starting on a visible frame loses none"
    ball = still_ball(n)
    ball[250:260] = np.nan
    cut = conventions.events(scene(n, ball, graded=[(0, 200), (255, 400)]))
    assert cut["ball_lost"]["frames"].tolist() == [], "frames outside every block are not gaps"

    # ball_moves: the first frame at or above STILL_SPEED after at least 25 still frames, directly before it
    ball = still_ball(n)
    t = np.arange(90, 140)
    ball[t, 0] = 50.0 + 0.2 * (t - 89)       # 5 m/s for 50 frames
    ball[140:200, 0] = 60.0                  # then still for 60
    t = np.arange(200, 220)
    ball[t, 0] = 60.0 + 0.2 * (t - 199)
    ball[220:235, 0] = 64.0                  # still for 15 frames only, then on again
    t = np.arange(235, 260)
    ball[t, 0] = 64.0 + 0.2 * (t - 234)
    ball[260:, 0] = 69.0
    ball[:90, 0] = 50.0
    inputs = scene(n, ball)
    events = conventions.events(inputs)
    moves = events["ball_moves"]["frames"].tolist()
    assert moves == plain_ball_moves(inputs) and len(moves) == 2, moves
    assert 84 <= moves[0] <= 90 and 195 <= moves[1] <= 200, "the kick is found where the ball starts to move"
    assert events["ball_moves"]["team"] is None
    gap = still_ball(n)
    gap[60:70] = np.nan                      # still, then unseen, then moving: the ball was not seen to be still
    t = np.arange(70, 120)
    gap[t, 0] = 50.0 + 0.3 * (t - 69)
    assert conventions.events(scene(n, gap))["ball_moves"]["frames"].tolist() == []
    short = still_ball(n)
    t = np.arange(20, 60)
    short[t, 0] = 50.0 + 0.3 * (t - 19)      # moving from 20: only about 20 still frames before it
    assert conventions.events(scene(n, short))["ball_moves"]["frames"].tolist() == []

    # touch_start: the first frame of a kept control episode, with the episode's team; the two episodes of one team
    # with a frame between them are two; one shorter than min_frames is none
    ball = still_ball(n, (30.0, 30.0))
    slots = {0: np.full((n, 2), np.nan), 1: np.full((n, 2), np.nan)}
    slots[0] = beside_ball(n, range(10, 30))
    slots[1] = beside_ball(n, range(30, 50), offset=(-0.5, 0.0))
    slots[0][60:62] = [30.5, 30.0]           # two frames: under min_frames
    slots[0][90:100] = [30.5, 30.0]
    slots[0][101:110] = [30.5, 30.0]
    slots[0][120:123] = [30.5, 30.0]         # three frames: exactly min_frames
    events = conventions.events(scene(n, ball, slots=slots))
    touch = events["touch_start"]
    assert touch["frames"].tolist() == [10, 30, 90, 101, 120], touch["frames"]
    assert touch["team"].dtype == np.int8 and touch["team"].tolist() == [H, A, H, H, H], touch["team"]
    assert conventions.TOUCH == {"radius": 2.0, "max_rel_speed": 6.0, "min_frames": 3, "confidence": 0.99,
                                 "hold": 0.5}, "the rule is main.toml's, with the confidence and hold of the design"

    # tag_start / tag_end: SET PIECES intervals only, however the code is cased
    tags = [{"code": "SET PIECES", "start_frame": 120, "end_frame": 180, "tags": {}},
            {"code": "Set Pieces", "start_frame": 300, "end_frame": 330, "tags": {}},
            {"code": "SHOTS", "start_frame": 200, "end_frame": 230, "tags": {}},
            {"code": None, "start_frame": 250, "end_frame": 260, "tags": {}}]
    events = conventions.events(scene(n, still_ball(n), tags=tags))
    assert events["tag_start"]["frames"].tolist() == [120, 300] and events["tag_start"]["team"] is None
    assert events["tag_end"]["frames"].tolist() == [180, 330] and events["tag_end"]["team"] is None


def test_conventions_ball_moves_needs_25_still_frames():
    """A still run of 24 frames is no restart and one of 25 is, however the speed smooths the stop and the start."""
    outcomes = {}
    for hold in range(20, 41):
        n = 60 + hold + 60
        t = np.arange(n)
        ball = still_ball(n)
        ball[:, 0] = np.where(t < 60, 20.0 + 0.2 * t,
                              np.where(t < 60 + hold, 20.0 + 0.2 * 59, 20.0 + 0.2 * 59 + 0.2 * (t - 59 - hold)))
        inputs = scene(n, ball)
        speed = kinematics.ball_track(inputs).speed
        rise = next(s for s in range(61, n) if speed[s] >= play_features.STILL_SPEED > speed[s - 1])
        run = 0
        while speed[rise - 1 - run] < play_features.STILL_SPEED:
            run += 1
        moves = conventions.events(inputs)["ball_moves"]["frames"].tolist()
        assert moves == ([rise] if run >= 25 else []), (hold, run, rise, moves)
        outcomes[run] = bool(moves)
    assert outcomes[24] is False and outcomes[25] is True, "the sweep must reach both sides of the 25-frame line"


def plain_ball_moves(inputs):
    """ball_moves worked out frame by frame from the ball's own speed."""
    track = kinematics.ball_track(inputs)
    still_speed = play_features.STILL_SPEED
    out = []
    for start, stop in kinematics.blocks(inputs):
        for t in range(start + 25, stop):
            if not (track.visible[t] and track.speed[t] >= still_speed):
                continue
            if all(track.visible[s] and track.speed[s] < still_speed for s in range(t - 25, t)):
                out.append(t)
    return out


def plain_crossings(alive):
    down, up = [], []
    for t in range(1, len(alive)):
        before, now = alive[t - 1], alive[t]
        if np.isnan(before) or np.isnan(now):
            continue
        if before >= 0.5 and now < 0.5:
            down.append(t)
        if before < 0.5 and now >= 0.5:
            up.append(t)
    return down, up


def test_conventions_crossings_of_a_given_alive():
    """Downward: below 0.5 after at or above it. Upward: at or above it after below. NaN neither starts nor ends one."""
    nan = np.nan
    alive = np.array([0.9, 0.5, 0.4, 0.5, 0.6, nan, 0.2, 0.7, 0.3, nan, 0.8, 0.45, 0.55])
    assert conventions.crossings(alive, downward=True).tolist() == [2, 8, 11]
    assert conventions.crossings(alive, downward=False).tolist() == [3, 7, 12]
    assert conventions.crossings(alive, downward=True).dtype.kind == "i"
    assert conventions.crossings(np.array([nan, 0.9, nan, 0.1]), downward=True).tolist() == []
    assert conventions.crossings(np.array([0.2]), downward=False).tolist() == []
    assert conventions.crossings(np.array([0.2, 0.2, 0.2]), downward=True).tolist() == []
    assert conventions.crossings(np.array([0.2, 0.51]), downward=False).tolist() == [1]
    assert conventions.crossings(np.array([0.51, 0.49]), downward=True).tolist() == [1]
    assert conventions.ALIVE_LEVEL == 0.5


def test_conventions_model_events_are_the_models_crossings():
    """model_dead is where alive falls through 0.5 and model_alive where it rises through it, from the config's file."""
    assert conventions.MODEL == "pipeline/models/dead_ball_hgb.json"
    rng = np.random.default_rng(7)
    n = 900
    t = np.arange(n)
    ball = np.stack([52.0 + 20.0 * np.sin(t / 25.0), 34.0 + 12.0 * np.cos(t / 31.0)], axis=1)
    ball[300:500] = [30.0, -2.0]             # beyond the touchline and still: the model says dead
    slots = {}
    for k in range(8):
        slots[k] = np.stack([52.0 + 25.0 * np.sin(t / 40.0 + k), 34.0 + 15.0 * np.cos(t / 50.0 + 2 * k)], axis=1)
        slots[k][300:500] = slots[k][300] + rng.normal(0.0, 0.01, size=(200, 2))      # and everybody standing
    inputs = scene(n, ball, slots=slots, graded=[(0, 450), (460, 900)])
    alive = dead_ball_model.compute(RunContext(inputs=inputs, seed=0), model=conventions.MODEL).alive
    down, up = plain_crossings(alive)
    events = conventions.events(inputs)
    assert events["model_dead"]["frames"].tolist() == down and events["model_alive"]["frames"].tolist() == up
    assert events["model_dead"]["team"] is None and events["model_alive"]["team"] is None
    assert down and up, "the scene must make the model cross 0.5 both ways, or nothing was compared"
    assert 0 not in down + up and 460 not in down + up, "no crossing at a block's first frame"


def test_conventions_events_collects_every_detector_and_skips_the_unavailable():
    """A DFL match has no Smart Tagging: AssetUnavailable leaves the tag detectors out, and nothing else."""
    match = synthetic_matches()["RND001"]
    from pipeline.benchmark import run as benchmark_run
    atd = benchmark_run.make_atd(match, "gaps", 0, synthetic_calibration())
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=0)
    events = conventions.events(inputs)
    assert list(events) == ["touch_start", "line_out", "ball_lost", "ball_moves", "model_dead", "model_alive"]
    for name, entry in events.items():
        assert sorted(entry) == ["frames", "team"], name
        frames = entry["frames"]
        assert frames.dtype == np.int64 and (np.diff(frames) > 0).all(), "{}: sorted and distinct".format(name)
        assert all(any(start <= f < stop for _, start, stop in inputs.graded) for f in frames), name
        assert (entry["team"] is not None) == (name == "touch_start"), name
    assert len(events["touch_start"]["team"]) == len(events["touch_start"]["frames"])
    assert set(np.unique(events["touch_start"]["team"])) <= {H, A}
    # a stand-in with tags has all eight, in the design's order
    tagged = conventions.events(scene(400, still_ball(400), tags=[]))
    assert list(tagged) == list(conventions.DETECTORS)

    # a missing tagging file is a mistake to be seen, not an asset that this source does not have
    expect_error(FileNotFoundError, lambda: conventions.events(failing_scene(FileNotFoundError("no tagging file"))),
                 "tagging")
    unavailable = failing_scene(match_inputs.AssetUnavailable("smart_tagging: this source has none"))
    assert list(conventions.events(unavailable)) == ["touch_start", "line_out", "ball_lost", "ball_moves",
                                                     "model_dead", "model_alive"]


def failing_scene(error):
    """A stand-in whose smart_tagging raises `error`, as a source without tagging, or with a missing file, would."""
    inputs = scene(400, still_ball(400))

    class Failing(type(inputs)):
        @property
        def smart_tagging(self):
            raise error
    inputs.__class__ = Failing
    return inputs


# -- the DFL mode -----------------------------------------------------------------------------------------------------

def measured(**more):
    options = dict(profile="gaps", seeds=1, truth="raw", matches=synthetic_matches(),
                   calibration=synthetic_calibration(), log=quiet)
    options.update(more)
    return conventions.measure(**options)


def test_conventions_never_pass_truth():
    """The DFL mode serves its inputs as the trainer does, and neither they nor the events hold the match's truth."""
    matches = synthetic_matches()
    served, spied = [], []
    real_served, real_events = train_dead_ball.served_inputs, conventions.events

    def serving(match, profile, seed, calibration, data_dir):
        inputs = real_served(match, profile, seed, calibration, data_dir)
        served.append((match.match_id, profile, seed))
        return inputs

    def spy(inputs):
        spied.append([leaks(inputs, match) for match in matches.values()])
        return real_events(inputs)

    train_dead_ball.served_inputs, conventions.events = serving, spy
    try:
        measured(seeds=2)
    finally:
        train_dead_ball.served_inputs, conventions.events = real_served, real_events
    assert served == [(match_id, "gaps", seed) for match_id in matches for seed in (0, 1)], served
    assert len(spied) == len(served), "one set of events per match and seed"
    assert all(found_ == [[] for _ in matches] for found_ in spied), spied
    inputs = train_dead_ball.served_inputs(matches["RND001"], "gaps", 0, synthetic_calibration(), None)
    assert leaks(inputs, matches["RND001"]) == []


def test_conventions_runs_under_the_tracking_profile():
    """The default profile of the DFL mode: players broken into slots, NaN speeds, a cap; the events still come out."""
    matches = {"RND001": tracked_match("RND001"), "RND002": tracked_match("RND002", 1)}
    seen = []
    real = conventions.events

    def spy(inputs):
        seen.append(len(leaks(inputs, matches["RND001"])) + len(leaks(inputs, matches["RND002"])))
        return real(inputs)
    conventions.events = spy
    try:
        result = conventions.measure(profile="tracking", seeds=2, matches=matches, calibration=tracking_calibration(),
                                     log=quiet)
    finally:
        conventions.events = real
    assert seen == [0] * 4, seen
    assert result["profile"] == "tracking" and result["seeds"] == [0, 1]
    assert sorted(key for key in result["table"]) == sorted(
        (kind, name) for name in conventions.DETECTORS if not name.startswith("tag_")
        for kind in conventions.APPLIES[name])
    assert all(cell["n"] > 0 for cell in result["table"].values()), "random flags change often: every cell has changes"
    assert any(cell["found"] for cell in result["table"].values())
    assert "tracking" in conventions.render_dfl(result)[-1]


def test_conventions_dfl_pools_by_lags_and_reads_the_chosen_truth():
    matches = synthetic_matches()
    result = measured(seeds=2)
    assert result["profile"] == "gaps" and result["truth"] == "raw" and result["seeds"] == [0, 1]
    assert result["matches"] == ["RND001", "RND002"]
    expected, per_match = [], {}
    for match_id, match in matches.items():
        codes, _ = truth.truth_codes(match)
        runs = [(start, stop) for _, start, stop in match.graded]
        trans = conventions.transitions_of(codes, runs)
        tables = []
        for seed in (0, 1):
            inputs = train_dead_ball.served_inputs(match, "gaps", seed, synthetic_calibration(), None)
            tables.append(conventions.lags(trans, conventions.events(inputs), runs, match.fps))
        per_match[match_id] = conventions.pool(tables)
        expected.append(per_match[match_id])
    pooled = conventions.pool(expected)
    assert result["table"] == pooled
    assert sorted(result["per_match"]) == ["RND001", "RND002"]
    for match_id in per_match:
        assert result["per_match"][match_id] == per_match[match_id], match_id
    assert any(cell["found"] for cell in pooled.values()), "nothing was found anywhere: nothing was compared"
    assert all(len(cell["lags"]) == cell["found"] for cell in pooled.values())
    # the coarse truth absorbs short possessions and leaves every stoppage where it was
    coarse = measured(seeds=2, truth="coarse")
    assert coarse["truth"] == "coarse"
    raw_n = {key: cell["n"] for key, cell in result["table"].items()}
    for key, cell in coarse["table"].items():
        if key[0] == "team":
            assert cell["n"] < raw_n[key], key
        else:
            assert cell["n"] == raw_n[key], key
    # what it refuses, before a match is loaded
    from pipeline.benchmark import run as benchmark_run
    for options, word in (({"truth": "fine"}, "fine"), ({"seeds": 0}, "seeds"), ({"profile": "nope"}, "nope"),
                          ({"match_ids": ["NOPE"]}, "NOPE")):
        expect_error(benchmark_run.BenchmarkError, lambda options=options: measured(**options), word)


def test_conventions_dfl_document_and_file_name():
    result = measured(seeds=1)
    document = conventions.dfl_document(result)
    json.dumps(document, allow_nan=False)
    assert document["source"] == "dfl" and document["profile"] == "gaps" and document["truth"] == "raw"
    assert document["seeds"] == [0] and document["matches"] == ["RND001", "RND002"] and document["fps"] == 25
    key = conventions.cell_name("team", "touch_start")
    assert key == "team/touch_start" and conventions.parse_cell_name(key) == ("team", "touch_start")
    assert document["cells"][key]["lags"] == result["table"][("team", "touch_start")]["lags"]
    assert sorted(document["per_match"]) == ["RND001", "RND002"]
    assert "lags" not in document["per_match"]["RND001"][key], "per match: the summary only"
    summary = conventions.summarise(result["table"])
    assert document["summary"][key]["median"] == summary[("team", "touch_start")]["median"]
    assert os.path.basename(conventions.dfl_path("D", "tracking", "raw", None, 3)) == "dfl_tracking_raw.json"
    assert os.path.basename(conventions.dfl_path("D", "tracking", "coarse", None, 3)) == "dfl_tracking_coarse.json"
    assert os.path.basename(conventions.dfl_path("D", "clean", "raw", None, 3)) == "dfl_clean_raw.json"
    subset = os.path.basename(conventions.dfl_path("D", "tracking", "raw", ["J03WMX"], 1))
    assert subset == "dfl_tracking_raw.m-J03WMX.s1.json", "a subset never replaces the full report"
    folder = os.path.dirname(conventions.dfl_path("D", "tracking", "raw", None, 3)).replace("\\", "/")
    assert folder == "D/runs/conventions"
    lines = conventions.render_dfl(result)
    assert any("RND001" in line for line in lines) and any("touch_start" in line for line in lines)


# -- the public mode --------------------------------------------------------------------------------------------------

# What public_scene puts around the changes of write_public_gt's truth, by construction (frames):
#   1300 H to A      the away team (slot 1) takes the ball at 1306; the home team (slot 0) touched it at 1290-1297, the
#                    old team's, which is no start of the new one
#   1500 A to D      the ball leaves the pitch at 1492, is lost from view at 1503, and a SET PIECES interval starts at
#                    1495
#   1620 D to H      the ball is struck at 1628 (the speed shows it at 1625), the home team takes it at 1632, and the
#                    interval ends at 1625
#   2000 H to D      the ball leaves the pitch at 1995, and a SET PIECES interval starts at 1990
#   2050 D to A      the ball is struck at 2055 (the speed shows it at 2052), the away team (slot 3) takes it at 2062,
#                    and the interval ends at 2055
#   2400 A to H      the home team (slot 2) takes the ball at 2398; the away team (slot 1) touched it at 2380-2394
PUBLIC_EXACT = {1300: {"touch_start": 6},
                1500: {"line_out": -8, "ball_lost": 3, "tag_start": -5},
                1620: {"touch_start": 12, "ball_moves": 5, "tag_end": 5},
                2000: {"line_out": -5, "tag_start": -10},
                2050: {"touch_start": 12, "ball_moves": 2, "tag_end": 5},
                2400: {"touch_start": -2}}


def public_scene(n=3000, tags=True):
    """Hand-built tracking whose events fall at the frames above, and Smart Tagging intervals (SET PIECES, others)."""
    t = np.arange(n)
    path = np.stack([52.0 + 20.0 * np.sin(t / 25.0), 34.0 + 12.0 * np.cos(t / 31.0)], axis=1)   # in play, inside
    ball = path.copy()
    stops = ((1492, 1628), (1995, 2055))
    for start, stop in stops:
        ball[start:stop] = [30.0, -2.0]                         # beyond the touchline, and still
    ball[1503:1511] = np.nan
    rng = np.random.default_rng(11)
    slots = {}
    for slot in range(4, 10):                                   # far from the ball, and standing at a stoppage
        track = path + [12.0 + 2.0 * slot, 0.0]
        for start, stop in stops:
            track[start:stop] = track[start] + rng.normal(0.0, 0.01, size=(stop - start, 2))
        slots[slot] = track
    for slot in range(4):                                       # on the ball for a few frames, and nowhere otherwise
        slots[slot] = np.full((n, 2), np.nan)
    for slot, start, stop in ((1, 1306, 1340), (0, 1290, 1298), (0, 1632, 1660), (3, 2062, 2090), (2, 2398, 2430),
                              (1, 2380, 2395)):
        slots[slot][start:stop] = ball[start:stop] + [0.5, 0.0]
    intervals = [{"code": code, "start_frame": start, "end_frame": stop, "tags": {}} for code, start, stop in (
        ("SET PIECES", 1495, 1625), ("PHASE", 1200, 1290), ("SET PIECES", 1990, 2055), ("SHOTS", 2300, 2330))
        ] if tags else []
    return scene(n, ball, slots=slots, tags=intervals, game="PUBSCENE")


def write_public_gt(root, n=3000):
    codes = codes_of(n, [(1000, 1300, H), (1300, 1500, A), (1500, 1620, D), (1620, 2000, H), (2000, 2050, D),
                         (2050, 2400, A), (2400, 2600, H)])
    path = os.path.join(root, "gt.csv")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        for frame, code in enumerate(codes):
            handle.write("{},{}\n".format(frame, pm.LABELS[code]))
    return path


def test_conventions_public_prints_aggregates_only():
    with tempfile.TemporaryDirectory(prefix="conv") as root:
        gt = write_public_gt(root)
        inputs = public_scene()
        inputs.data_dir = os.path.join(root, "own")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            document = conventions.public(inputs, gt, data_dir=root)
        text = buffer.getvalue().replace(root, "<root>").replace(root.replace("\\", "/"), "<root>")
        lines = text.splitlines()
        assert 3 <= len(lines) <= MAX_PUBLIC_LINES, "{} lines:\n{}".format(len(lines), text)
        printed = set(numbers_in(text))
        assert not printed & set(PUBLIC_TRANSITIONS), sorted(printed & set(PUBLIC_TRANSITIONS))
        assert not printed & {1000, 2600}, "nor the window's edges"
        assert "touch_start" in text and "line_out" in text and "tag_start" in text
        # every cell of the public window has fewer than MIN_SHOWN changes found, so each line may show n, the share
        # found and the median (in frames and in seconds) and no other number: a quartile of one to three changes is
        # their lags, and the three figures of one change are its lag
        cell_lines = {}
        for line in lines:
            named = re.match(r"\s+(team|into_dead|out_of_dead)\s+(\w+)\s", line)
            if named:
                cell_lines[named.groups()] = line
        assert len(cell_lines) == 9, sorted(cell_lines)
        for (kind, name), line in cell_lines.items():
            cell = document["cells"][conventions.cell_name(kind, name)]
            row = document["summary"][conventions.cell_name(kind, name)]
            assert 0 < cell["found"] < conventions.MIN_SHOWN, (kind, name, cell["found"])
            printed = re.findall(r"[+-]\d+\.\d+", line)
            assert printed == ["{:+.1f}".format(row["median"]), "{:+.2f}".format(row["median_s"])], (kind, name, line)
            if cell["found"] > 1:
                assert row["p25"] is not None and row["p75"] is not None, "the document keeps the quartiles"
                for lag in cell["lags"]:
                    if lag != row["median"]:         # a lag that is the median is printed as the median, and may be
                        assert "{:+.1f}".format(lag) not in printed, (kind, name, lag)
        assert any(row["found"] == 1 for row in document["summary"].values()), "a one-change cell must be in the scene"
        assert any(row["found"] == 2 for row in document["summary"].values()), "and a two-change one"
        folder = os.path.join(root, "runs", "conventions")
        assert sorted(os.listdir(folder)) == ["public_lags.json", "public_transitions.csv"]
        assert os.path.join("runs", "conventions") in text.replace("/", os.sep)

        with open(os.path.join(folder, "public_transitions.csv"), encoding="utf-8", newline="") as handle:
            raw = handle.read()
        assert "\r" not in raw and raw.endswith("\n")
        rows = list(csv.DictReader(io.StringIO(raw)))
        assert [int(row["frame"]) for row in rows] == list(PUBLIC_TRANSITIONS)
        assert [(row["from"], row["to"], row["type"]) for row in rows] == [
            ("H", "A", "team"), ("A", "D", "into_dead"), ("D", "H", "out_of_dead"), ("H", "D", "into_dead"),
            ("D", "A", "out_of_dead"), ("A", "H", "team")]
        assert list(rows[0]) == ["frame", "from", "to", "type"] + list(conventions.DETECTORS)
        for row in rows:
            expected = PUBLIC_EXACT[int(row["frame"])]
            for name in conventions.DETECTORS:
                if row["type"] not in conventions.APPLIES[name]:
                    assert row[name] == "", (row["frame"], name, "does not apply to this type")
                elif name in expected:
                    assert row[name] != "" and int(row[name]) == expected[name], (row["frame"], name, row[name])
                elif name.startswith("model_"):
                    assert row[name] != "" and abs(int(row[name])) <= 15, (row["frame"], name, row[name])
                else:
                    assert row[name] == "", (row["frame"], name, "nothing is there to be found")

        with open(os.path.join(folder, "public_lags.json"), encoding="utf-8") as handle:
            loaded = json.load(handle)
        assert loaded == json.loads(json.dumps(document)), "the file is the document that public() returns"
        assert all(row["p25"] is not None for row in loaded["summary"].values() if row["found"]), (
            "the file keeps every value, whatever the printout shows")
        assert loaded["source"] == "public" and loaded["fps"] == 25
        cell = loaded["cells"][conventions.cell_name("into_dead", "line_out")]
        assert cell == {"n": 2, "found": 2, "lags": [-8, -5]}, cell
        assert loaded["cells"][conventions.cell_name("into_dead", "ball_lost")] == {"n": 2, "found": 1, "lags": [3]}
        assert loaded["cells"][conventions.cell_name("team", "touch_start")] == {"n": 2, "found": 2, "lags": [6, -2]}
        assert loaded["summary"][conventions.cell_name("team", "touch_start")]["median"] == 2.0
        assert loaded["transitions"] == {"team": 2, "into_dead": 2, "out_of_dead": 2}
        wanted = conventions.lags(conventions.transitions_of(pm.label_codes(pm.read_label_csv(gt)),
                                                             [(1000, 2600)]), conventions.events(inputs),
                                  [(1000, 2600)], 25)
        assert {conventions.parse_cell_name(key): cell["lags"] for key, cell in loaded["cells"].items()} == {
            key: cell["lags"] for key, cell in wanted.items()}
        assert not os.path.exists(inputs.data_dir), "the data dir that is given wins over the inputs' own"
        # without one, the files go under the inputs' own data dir
        with contextlib.redirect_stdout(io.StringIO()):
            conventions.public(inputs, gt)
        assert sorted(os.listdir(os.path.join(inputs.data_dir, "runs", "conventions"))) == [
            "public_lags.json", "public_transitions.csv"]
        # the second run rewrites the same files, byte for byte
        before = {name: open(os.path.join(folder, name), "rb").read() for name in os.listdir(folder)}
        with contextlib.redirect_stdout(io.StringIO()):
            conventions.public(inputs, gt, data_dir=root)
        assert {name: open(os.path.join(folder, name), "rb").read() for name in os.listdir(folder)} == before


def table_of(cells):
    """A lag table from {(type, detector): lags}: n is the found count plus two changes with nothing found."""
    return {key: {"n": len(lags) + 2, "found": len(lags), "lags": list(lags)} for key, lags in cells.items()}


SMALL_TABLE = table_of({("team", "touch_start"): [],
                        ("into_dead", "line_out"): [-9],
                        ("into_dead", "ball_lost"): [-8, -5],
                        ("into_dead", "model_dead"): [-20, -10, -4, -2],
                        ("out_of_dead", "ball_moves"): [1, 2, 3, 4, 10],
                        ("out_of_dead", "model_alive"): [-30, -12, -6, -2, -1, 0]})


def cell_line(lines, kind, name):
    found_ = [line for line in lines if re.match(r"\s*{}\s+{}\s".format(kind, name), line)]
    assert len(found_) == 1, (kind, name, lines)
    return found_[0]


def numbers_printed(line):
    return re.findall(r"[+-]\d+\.\d+", line)


def test_conventions_small_cells_show_the_median_only():
    """Below MIN_SHOWN found, a BAR-ATH printout gives n, the share found and the median: no quartile, nothing else."""
    assert conventions.MIN_SHOWN == 5
    summary = conventions.summarise(SMALL_TABLE)
    # the helper: a cell with 1 to 4 found is limited, with none (nothing to hide) or 5 and more it is not
    assert [conventions.limited(summary[key]) for key in SMALL_TABLE] == [False, True, True, True, False, False]
    row = summary[("into_dead", "model_dead")]
    hidden = conventions.shown(row)
    assert hidden["p25"] is None and hidden["p75"] is None and hidden["p25_s"] is None and hidden["p75_s"] is None
    assert hidden["median"] == row["median"] and hidden["median_s"] == row["median_s"] and hidden["limited"] is True
    assert (hidden["n"], hidden["found"], hidden["share_found"]) == (row["n"], row["found"], row["share_found"])
    assert row["p25"] == -12.5 and "limited" not in row, "shown does not change the row it is given"
    big = summary[("out_of_dead", "ball_moves")]
    assert conventions.shown(big) is big
    document = {"transitions": {"team": 2, "into_dead": 6, "out_of_dead": 4}, "events": {}, "fps": 25,
                "summary": {conventions.cell_name(*key): value for key, value in summary.items()}}
    lines = conventions.render_public(document)
    assert numbers_printed(cell_line(lines, "team", "touch_start")) == []
    one = cell_line(lines, "into_dead", "line_out")
    assert numbers_printed(one) == ["-9.0", "-0.36"] and "33%" in one, one
    two = cell_line(lines, "into_dead", "ball_lost")
    assert numbers_printed(two) == ["-6.5", "-0.26"] and "50%" in two, two     # not -8, -5, -7.2 or -5.8
    four = cell_line(lines, "into_dead", "model_dead")
    assert numbers_printed(four) == ["-7.0", "-0.28"] and "67%" in four, four   # not -12.5 or -3.5
    five = cell_line(lines, "out_of_dead", "ball_moves")
    assert numbers_printed(five) == ["+3.0", "+2.0", "+4.0", "+0.12"], five
    six = cell_line(lines, "out_of_dead", "model_alive")
    assert numbers_printed(six) == ["-4.0", "-10.5", "-1.2", "-0.16"], six
    assert any("fewer than 5" in line for line in lines), "the legend says why a quartile is missing"
    # DFL's own printout is not limited: the same cells show their quartiles in the pooled table, one line per cell
    result = {"profile": "tracking", "truth": "raw", "seeds": [0], "matches": ["M1"], "fps": 25,
              "table": SMALL_TABLE, "per_match": {"M1": SMALL_TABLE}}
    printout = conventions.render_dfl(result)
    pooled = printout[:next(i for i, line in enumerate(printout) if line.startswith("per match"))]
    for key in SMALL_TABLE:
        cell_line(pooled, *key)                                      # exactly one line each
    assert numbers_printed(cell_line(pooled, "into_dead", "ball_lost")) == ["-6.5", "-7.2", "-5.8", "-0.26"], pooled
    assert numbers_printed(cell_line(pooled, "into_dead", "model_dead")) == ["-7.0", "-12.5", "-3.5", "-0.28"], pooled
    assert numbers_printed(cell_line(pooled, "into_dead", "line_out")) == ["-9.0", "-9.0", "-9.0", "-0.36"], pooled
    assert not any("fewer than" in line for line in pooled), "no legend about a limit that DFL's printout has not"


def test_conventions_public_refuses_what_it_cannot_read():
    with tempfile.TemporaryDirectory(prefix="conv") as root:
        gt = write_public_gt(root, n=3000)
        short = scene(2000, still_ball(2000))
        expect_error(ValueError, lambda: conventions.public(short, gt, data_dir=root), "conventions", "3000", "2000")
        empty = os.path.join(root, "empty.csv")
        with open(empty, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("frame,label\n" + "".join("{},X\n".format(i) for i in range(3000)))
        expect_error(ValueError, lambda: conventions.public(public_scene(), empty, data_dir=root), "conventions",
                     "graded")
        assert not os.path.exists(os.path.join(root, "runs")), "nothing is written for a GT that was refused"


def test_conventions_main_dispatches_and_defaults():
    parser = conventions.build_parser()
    args = parser.parse_args(["public"])
    assert (args.command, args.gt, args.data_dir) == ("public", None, None)
    args = parser.parse_args(["dfl"])
    assert (args.command, args.profile, args.seeds, args.truth, args.matches, args.data_dir) == (
        "dfl", "tracking", 3, "raw", None, None)
    args = parser.parse_args(["dfl", "--profile", "gaps", "--seeds", "1", "--truth", "coarse", "--matches", "A", "B",
                              "--data-dir", "D"])
    assert (args.profile, args.seeds, args.truth, args.matches, args.data_dir) == ("gaps", 1, "coarse", ["A", "B"], "D")
    args = parser.parse_args(["compare"])
    assert (args.command, args.profile, args.truth, args.data_dir) == ("compare", "tracking", "raw", None)
    for bad in ([], ["nope"], ["dfl", "--truth", "fine"], ["compare", "--seeds", "2"], ["public", "--profile", "x"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                conventions.main(bad)
            except SystemExit as stop:
                assert stop.code != 0, bad
            else:
                raise AssertionError("{} was accepted".format(bad))
    # `public` builds BAR-ATH's Inputs through bar_ath_inputs and the public GT of the repository by default
    with tempfile.TemporaryDirectory(prefix="conv") as root:
        gt = write_public_gt(root)
        seen = []
        real = conventions.bar_ath_inputs

        def stand_in(data_dir):
            seen.append(data_dir)
            return public_scene()
        conventions.bar_ath_inputs = stand_in
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = conventions.main(["public", "--gt", gt, "--data-dir", root])
        finally:
            conventions.bar_ath_inputs = real
        assert code == 0 and seen == [root]
        assert os.path.isfile(os.path.join(root, "runs", "conventions", "public_lags.json"))
    assert conventions.default_gt().replace("\\", "/").endswith("evaluation/gt/BAR-ATH_possession_gt_public.csv")
    assert os.path.isfile(conventions.default_gt()), "the public GT is in the repository"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        code = conventions.main(["public", "--gt", os.path.join(tempfile.gettempdir(), "no-such-gt.csv"),
                                 "--data-dir", tempfile.gettempdir()])
    assert code == 1 and "FAILED" in printed.getvalue()


def test_conventions_main_runs_the_dfl_mode_and_writes_its_file():
    real = conventions.measure
    seen = []

    def synthetic(**options):
        seen.append(options)
        return real(**dict(options, matches=synthetic_matches(), calibration=synthetic_calibration(), log=quiet,
                           match_ids=None))
    with tempfile.TemporaryDirectory(prefix="conv") as root:
        conventions.measure = synthetic
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = conventions.main(["dfl", "--profile", "gaps", "--seeds", "1", "--truth", "coarse",
                                         "--data-dir", root])
        finally:
            conventions.measure = real
        assert code == 0 and seen == [{"profile": "gaps", "seeds": 1, "truth": "coarse", "match_ids": None,
                                       "data_dir": root}], seen
        path = os.path.join(root, "runs", "conventions", "dfl_gaps_coarse.s1.json")
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        assert document["truth"] == "coarse" and document["profile"] == "gaps" and document["seeds"] == [0]
        assert sorted(document["per_match"]) == ["RND001", "RND002"] and "team/touch_start" in document["cells"]
        text = buffer.getvalue()
        assert "touch_start" in text and "RND001" in text and "dfl_gaps_coarse.s1.json" in text
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            assert conventions.main(["dfl", "--profile", "nope", "--data-dir", root]) == 1
        assert "FAILED" in printed.getvalue() and "nope" in printed.getvalue()


# -- compare ----------------------------------------------------------------------------------------------------------

def test_conventions_compare_interval():
    public = [-12, -11, -10, -10, -9, -9, -8, -14, -10, -10]
    dfl = [-6, -5, -4, -4, -4, -3, -2, 0, -4, -7, -4]
    gap, low, high = conventions.bootstrap_gap(public, dfl)
    assert gap == -6.0 and low <= gap <= high, (gap, low, high)
    assert (low, high) == conventions.bootstrap_gap(public, dfl)[1:], "the same draws every time"
    # the draws are the design's: 2,000 of the public transitions with replacement, default_rng([2, 1]), 5 and 95
    rng = np.random.default_rng([2, 1])
    draws = np.asarray(public)[rng.integers(0, len(public), size=(2000, len(public)))]
    medians = np.median(draws, axis=1) - np.median(dfl)
    assert (low, high) == tuple(float(v) for v in np.percentile(medians, [5, 95]))
    assert conventions.BOOTSTRAP_DRAWS == 2000 and tuple(conventions.BOOTSTRAP_SEED) == (2, 1)
    assert conventions.PERCENTILES == (5, 95)
    # the generator is the design's, seeded afresh by each call, and the draws are 2,000 by default
    real, seeds = np.random.default_rng, []

    def spy(seed=None):
        seeds.append(seed)
        return real(seed)
    np.random.default_rng = spy
    try:
        conventions.bootstrap_gap([1, 2, 3], [0])
        conventions.bootstrap_gap([4, 5, 6], [0])
    finally:
        np.random.default_rng = real
    assert seeds == [[2, 1], [2, 1]], seeds
    import inspect
    assert inspect.signature(conventions.bootstrap_gap).parameters["draws"].default == 2000
    # a gap known in advance lies inside: public lags of median -13, DFL's of median -10, so the gap is -3
    gap, low, high = conventions.bootstrap_gap([-14, -13, -12] * 10, [-10] * 25)
    assert gap == -3.0 and low <= -3.0 <= high and high - low <= 2.0, (gap, low, high)
    # an interval that excludes zero when the lags are all the same side, and one that does not when they straddle it
    assert conventions.bootstrap_gap([-20] * 12, [0] * 12)[2] < 0
    straddling = conventions.bootstrap_gap([-5, 5, -4, 6, -3, 4, 3, -6, 2, -2], [0] * 5)
    assert straddling[1] < 0 < straddling[2]
    # nothing found on either side: no gap, no interval
    assert conventions.bootstrap_gap([], [1, 2]) == (None, None, None)
    assert conventions.bootstrap_gap([1, 2], []) == (None, None, None)
    # a fresh generator for each call: calling twice or in another order changes nothing
    assert conventions.bootstrap_gap([1, 2, 3], [0]) == conventions.bootstrap_gap([1, 2, 3], [0])


def test_conventions_compare_hides_the_bounds_of_small_cells():
    """Compare's public side shows an interval's bounds from MIN_SHOWN found; below it, whether it excludes 0."""
    dfl_lags = [-6, -5, -4, -4, -4, -3, -2]
    shape = {("into_dead", "ball_lost"): [-30, -25, -20],                  # 3 found: far from 0
             ("into_dead", "model_dead"): [-20, -10, -4, -2],              # 4 found: the line is at 5
             ("out_of_dead", "ball_moves"): [-5, 0, 5],                    # 3 found: straddles 0
             ("out_of_dead", "model_alive"): [-3, -2, -2, -1, 4],          # 5 found: shown
             ("into_dead", "line_out"): [-12, -11, -10, -10, -9, -8]}      # 6 found: shown
    public_document = {"source": "public", "fps": 25, "cells": {
        conventions.cell_name(*key): cell for key, cell in table_of(shape).items()}}
    dfl_document = {"source": "dfl", "profile": "tracking", "truth": "raw", "fps": 25, "cells": {
        conventions.cell_name(*key): cell for key, cell in table_of({key: dfl_lags for key in shape}).items()}}
    rows = {(row["type"], row["detector"]): row for row in conventions.compare(public_document, dfl_document)}
    lines = conventions.render_compare(list(rows.values()))
    for key, lags_ in shape.items():
        row, line = rows[key], cell_line(lines, *key)
        assert row["low"] is not None and row["high"] is not None
        medians = ["{:+.1f}".format(float(np.median(lags_))), "{:+.1f}".format(float(np.median(dfl_lags)))]
        gap = "{:+.1f}".format(row["gap"])
        bounds = ["{:+.1f}".format(row["low"]), "{:+.1f}".format(row["high"])]
        assert not set(bounds) & set(medians + [gap]), "the case must not hide a bound behind another number"
        if len(lags_) < conventions.MIN_SHOWN:
            assert numbers_printed(line) == medians + [gap], (key, line)
            excludes = row["low"] > 0 or row["high"] < 0
            assert "excludes 0: {}".format("yes" if excludes else "no") in line, (key, line)
        else:
            assert numbers_printed(line) == medians + [gap] + bounds, (key, line)
            assert "excludes" not in line
    assert "excludes 0: yes" in cell_line(lines, "into_dead", "ball_lost")
    assert "excludes 0: no" in cell_line(lines, "out_of_dead", "ball_moves")
    assert any("fewer than 5" in line for line in lines), "the legend says why a bound is missing"
    # the numbers are all still there for the code: nothing is lost from the rows that compare returns
    assert rows[("into_dead", "ball_lost")]["public"]["p25"] is not None
    assert conventions.interval_text(None, None, rows[("into_dead", "ball_lost")]["public"]) == "n/a"
    assert conventions.interval_text(-3.0, -1.0, {"found": 4}) == "excludes 0: yes"
    assert conventions.interval_text(-3.0, 1.0, {"found": 4}) == "excludes 0: no"
    assert conventions.interval_text(1.0, 3.0, {"found": 1}) == "excludes 0: yes"
    assert conventions.interval_text(-3.0, 1.0, {"found": 5}) == "[-3.0, +1.0]"


def test_conventions_compare_reads_the_two_files():
    key = conventions.cell_name("into_dead", "line_out")
    tag = conventions.cell_name("into_dead", "tag_start")
    public_document = {"source": "public", "fps": 25, "cells": {
        key: {"n": 6, "found": 6, "lags": [-12, -11, -10, -10, -9, -8]},
        tag: {"n": 6, "found": 3, "lags": [-30, -25, -20]}}}
    dfl_document = {"source": "dfl", "profile": "tracking", "truth": "raw", "fps": 25, "cells": {
        key: {"n": 400, "found": 300, "lags": [-6, -5, -4, -4, -4, -3, -2]}}}
    rows = conventions.compare(public_document, dfl_document)
    assert [(row["type"], row["detector"]) for row in rows] == [("into_dead", "line_out"), ("into_dead", "tag_start")]
    first = rows[0]
    assert first["public"]["median"] == -10.0 and first["dfl"]["median"] == -4.0 and first["gap"] == -6.0
    assert first["low"] <= first["gap"] <= first["high"] and first["public"]["n"] == 6 and first["dfl"]["n"] == 400
    assert rows[1]["dfl"] is None and rows[1]["gap"] is None, "a detector that DFL cannot run has no gap"
    lines = conventions.render_compare(rows)
    assert any("line_out" in line and "-6.0" in line for line in lines), lines
    assert any("tag_start" in line and "n/a" in line for line in lines), lines
    with tempfile.TemporaryDirectory(prefix="conv") as root:
        folder = os.path.join(root, "runs", "conventions")
        os.makedirs(folder)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert conventions.main(["compare", "--data-dir", root]) == 1
        assert "FAILED" in buffer.getvalue() and "conventions.py public" in buffer.getvalue()
        with open(os.path.join(folder, "public_lags.json"), "w", encoding="utf-8") as handle:
            json.dump(public_document, handle)
        with open(os.path.join(folder, "dfl_tracking_raw.json"), "w", encoding="utf-8") as handle:
            json.dump(dfl_document, handle)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert conventions.main(["compare", "--data-dir", root]) == 0
        assert "line_out" in buffer.getvalue() and "-6.0" in buffer.getvalue()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            assert conventions.main(["compare", "--data-dir", root, "--truth", "coarse"]) == 1
        assert "dfl_tracking_coarse.json" in printed.getvalue()
    odd = dict(dfl_document, fps=30)
    expect_error(ValueError, lambda: conventions.compare(public_document, odd), "conventions", "fps")
    assert math.isfinite(first["low"]) and math.isfinite(first["high"])
