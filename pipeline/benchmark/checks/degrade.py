"""Checks of the degrader (pipeline/benchmark/degrade.py): the `gaps` and `tracking` profiles, their draws, and what
the degrader may not see.

They run on a DflMatch of random positions (checks/inputs.py's random_match) with the ball put back on every frame of
both halves, so that the share of the ball that a degradation removes is the share of the mask it was given, and on
masks built here, so that every figure is worked out from how they are made. The tracking profile's checks run on a
steady match (steady_match), whose players are on the pitch for whole stretches, as in a real match, and on a made-up
Calibration whose tracklet lengths and slot counts give figures worked out by hand (d = 3/55, a mean tracked span of 53
frames). The checks that need real data print aggregates only:
test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches (--dfl: the 7 downloaded DFL matches, a made-up Calibration)
and the two that need our own match as well (--data, NOT RUN where the DFL files are not there),
test_data_gaps_profile_serves_a_real_match_with_our_real_masks and
test_data_tracking_profile_on_a_real_match_with_our_real_calibration. Every other check here reads nothing from the
data directory.
"""

import ast
import dataclasses
import hashlib
import json
import math
import os
import types

import numpy as np

from pipeline.benchmark import calibration, degrade, dfl
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import NotRun, real_match
from pipeline.benchmark.checks.inputs import leaks, random_match
from pipeline.selftest import expect_error

# Params
ROWS = (240, 200)                       # the synthetic halves, in rows
MASK_ROWS = (330, 310)                  # our made-up masks: longer than the halves, so that an offset has a range
PRODUCTION = degrade.__file__
# What the made-up Calibration says of our tracking: tracklets of 10, 25, 50, 80 and 100 frames at 25 Hz (mean 53), and
# slots present per frame as {slots: frames} per half. The frames with more than 18 present (CAP) are 50 with 22, 30
# with 20 and 20 with 19: their mean is (1100 + 600 + 380) / 100 = 20.8, so d = 1 - 20.8 / 22 = 3 / 55, and a lost span
# has the mean mu = 53 * d / (1 - d) = 53 * 3 / 52 frames. The frames with 18 or fewer are there to be left out.
LENGTHS_S = (0.4, 1.0, 2.0, 3.2, 4.0)
PRESENT_COUNTS = {1: {22: 30, 20: 20, 19: 10, 18: 4, 10: 2, 0: 1}, 2: {22: 20, 20: 10, 19: 10, 18: 3, 10: 1, 0: 1}}
MADE_UP_D, MADE_UP_MEAN, MADE_UP_MU = 3 / 55, 53.0, 53 * 3 / 52


# -- helpers ----------------------------------------------------------------------------------------------------------

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


def made_up_calibration(seed=0, rows=MASK_ROWS):
    """What the degrader reads of a Calibration: `ball_masks`, with long runs of missing ball as ours has, and what the
    tracking profile reads, `tracklet_lengths_s` and `present_counts` (LENGTHS_S and PRESENT_COUNTS)."""
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
    for half, table in PRESENT_COUNTS.items():
        counts[half] = np.array([slots for slots, frames in sorted(table.items()) for _ in range(frames)],
                                dtype=np.int16)
        counts[half].setflags(write=False)
    lengths = np.array(LENGTHS_S)
    lengths.setflags(write=False)
    return types.SimpleNamespace(ball_masks=masks, present_counts=counts, tracklet_lengths_s=lengths)


def ball_bytes(atd):
    return atd.ball_xy.tobytes() + atd.ball_speed.tobytes()


# -- the transplant ---------------------------------------------------------------------------------------------------

def test_transplant_exact():
    """A window of the source from the offset; past its end, its last TAIL_FRAMES frames over and over, to the frame."""
    source = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert degrade.transplant(source, 6, 3).tolist() == source[3:9].tolist() == [1, 1, 0, 1, 0, 0]
    assert degrade.transplant(source, 10, 0).tolist() == source.tolist()
    assert degrade.transplant(source, 0, 4).tolist() == []
    out = degrade.transplant(source, 6, 3)
    assert out.dtype == bool and out.shape == (6,)
    out[:] = ~out
    assert source.tolist() == [1, 0, 0, 1, 1, 0, 1, 0, 0, 1], "the result is a new array, not a view of the source"

    # A source shorter than the tail: the whole source goes round. Built by laying copies end to end, which is no
    # formula of the code's: [source, source, ...] from the offset, 25 frames of it.
    laid = np.concatenate([source] * 4)
    assert degrade.transplant(source, 25, 8).tolist() == laid[8:33].tolist()
    assert degrade.transplant(source, 5, 10).tolist() == source[:5].tolist(), "after the end: the tail's first"
    assert degrade.transplant(source, 5, 12).tolist() == source[2:7].tolist(), "an offset past the end is on the cycle"

    # A source longer than the tail: only its last 7,500 frames go round, from the frame after the end.
    rng = np.random.default_rng(5)
    long = rng.uniform(size=8000) < 0.4
    tail = long[-degrade.TAIL_FRAMES:]
    assert tail.size == 7500 == degrade.TAIL_FRAMES and long.tolist()[500:] == tail.tolist()
    n, offset = 15200, 7900
    expected = np.concatenate([long[offset:], tail, tail, tail])[:n]
    out = degrade.transplant(long, n, offset)
    assert out.tolist() == expected.tolist()
    assert out[:100].tolist() == long[7900:].tolist(), "the real frames first"
    assert out[100:7600].tolist() == long[500:].tolist(), "then the tail from its first frame, not the source's"
    assert out[7600:15100].tolist() == long[500:].tolist() and out[15100:].tolist() == long[500:600].tolist()
    # Inside the source nothing cycles, even where the source is longer than the tail.
    assert degrade.transplant(long, 7000, 900).tolist() == long[900:7900].tolist()

    expect_error(ValueError, lambda: degrade.transplant(source.astype(int), 3, 0), "source", "bool")
    expect_error(ValueError, lambda: degrade.transplant(source[:0], 3, 0), "source", "non-empty")
    expect_error(ValueError, lambda: degrade.transplant(source.reshape(2, 5), 3, 0), "source", "1-D")
    expect_error(ValueError, lambda: degrade.transplant(source, -1, 0), "n", "at least 0")
    expect_error(ValueError, lambda: degrade.transplant(source, 3, -2), "offset", "at least 0")
    expect_error(ValueError, lambda: degrade.transplant(source, 2.5, 0), "n")


def test_transplant_choice_is_drawn_from_the_match_and_the_half():
    """A source half 1 or 2 about equally often, an offset anywhere in 0..max(0, M - n), the same for the same key."""
    match = tracked_match()
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    n = {1: ROWS[0], 2: ROWS[1]}
    choices = {half: [degrade.choose_transplant(view, half, calibration, seed) for seed in range(400)]
               for half in (1, 2)}
    for half, drawn in choices.items():
        sources = [choice.source_half for choice in drawn]
        assert {1, 2} == set(sources) and 140 < sources.count(1) < 260, (half, sources.count(1))
        for source_half in (1, 2):
            offsets = np.array([choice.offset for choice in drawn if choice.source_half == source_half])
            top = max(0, MASK_ROWS[source_half - 1] - n[half])
            assert offsets.min() >= 0 and offsets.max() <= top, (half, source_half, offsets.min(), offsets.max(), top)
            assert offsets.min() < top * 0.1 and offsets.max() > top * 0.9, "the range is covered"
            assert abs(offsets.mean() - top / 2) < top * 0.12, (half, source_half, offsets.mean(), top)
        assert drawn[0] == degrade.choose_transplant(view, half, calibration, 0), "the same key, the same draw"
    assert choices[1] != choices[2], "the two halves of a match draw apart"
    # A mask shorter than the half has one place to start: 0 (max(0, M - n)), and the transplant cycles the rest.
    short = types.SimpleNamespace(ball_masks={1: np.zeros(50, dtype=bool), 2: np.zeros(60, dtype=bool)})
    assert {degrade.choose_transplant(view, 1, short, seed).offset for seed in range(30)} == {0}
    expect_error(ValueError, lambda: degrade.choose_transplant(view, 3, calibration, 0), "half 3", "graded")
    expect_error(ValueError, lambda: degrade.choose_transplant(view, 1, types.SimpleNamespace(ball_masks={}), 0),
                 "no ball masks", "calibration.load()")


def test_transplant_offset_reaches_the_top_of_its_range():
    """The offset is uniform over 0..max(0, M - n) with both ends in it: M - n itself is drawn, and nothing above it.

    A draw over 0..M - n - 1 would pass the statistical check above (its means and ranges are loose), so this one
    makes M - n small and enumerates the seeds: a mask 3 frames longer than the half has the offsets 0, 1, 2 and 3, one
    1 frame longer has 0 and 1, and one exactly as long has only 0. The last offset places the mask's last frame on the
    half's last frame, which is the one place that a draw one short would never use.
    """
    view = degrade.tracking_view(tracked_match())
    for half, length in ((1, ROWS[0]), (2, ROWS[1])):
        masks = {1: np.zeros(length + 3, dtype=bool), 2: np.zeros(length + 1, dtype=bool)}
        offsets = {1: set(), 2: set()}
        for seed in range(300):
            choice = degrade.choose_transplant(view, half, types.SimpleNamespace(ball_masks=masks), seed)
            offsets[choice.source_half].add(choice.offset)
        assert offsets == {1: {0, 1, 2, 3}, 2: {0, 1}}, (half, offsets)
        exact = types.SimpleNamespace(ball_masks={1: np.zeros(length, dtype=bool), 2: np.zeros(length, dtype=bool)})
        assert {degrade.choose_transplant(view, half, exact, seed).offset for seed in range(50)} == {0}
        last = degrade.transplant(np.arange(length + 3) % 2 == 0, length, 3)
        assert last.tolist() == (np.arange(3, length + 3) % 2 == 0).tolist(), "M - n ends on the mask's last frame"


def test_rng_for_is_the_documented_key_and_streams_are_apart():
    """np.random.default_rng([seed, match_index, COMPONENTS[component], half]), with no profile in it."""
    assert degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 1}
    assert degrade.COMPONENTS == {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4}
    for component, number in degrade.COMPONENTS.items():
        ours = degrade.rng_for(3, 5, component, 2).integers(1 << 62, size=4)
        theirs = np.random.default_rng([3, 5, number, 2]).integers(1 << 62, size=4)
        assert ours.tolist() == theirs.tolist(), component
    first = {key: degrade.rng_for(*key).integers(1 << 62, size=3).tolist() for key in (
        (0, 0, "transplant", 1), (1, 0, "transplant", 1), (0, 1, "transplant", 1), (0, 0, "tracklets", 1),
        (0, 0, "transplant", 2))}
    assert len({tuple(value) for value in first.values()}) == len(first), "every part of the key changes the stream"
    expect_error(ValueError, lambda: degrade.rng_for(0, 0, "gaps", 1), "gaps", "transplant", "COMPONENTS")
    expect_error(ValueError, lambda: degrade.rng_for(-1, 0, "transplant", 1), "seed", "at least 0")
    expect_error(ValueError, lambda: degrade.rng_for(0, True, "transplant", 1), "match_index")
    expect_error(ValueError, lambda: degrade.rng_for(0, 0, "transplant", 1.5), "half")


# -- the profile ------------------------------------------------------------------------------------------------------

def test_degrade_deterministic():
    """The same seed gives the same arrays; another seed, and another match_index, give another mask."""
    match, calibration = tracked_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    first, info = degrade.degrade(view, "gaps", calibration, 0)
    again, info_again = degrade.degrade(degrade.tracking_view(match), "gaps", made_up_calibration(), 0)
    assert ball_bytes(first) == ball_bytes(again) and info == info_again
    other_seed, _ = degrade.degrade(view, "gaps", calibration, 1)
    assert ball_bytes(other_seed) != ball_bytes(first), "seed 1 is not seed 0"
    elsewhere, _ = degrade.degrade(degrade.tracking_view(match, match_index=3), "gaps", calibration, 0)
    assert ball_bytes(elsewhere) != ball_bytes(first), "another match_index is another stream"
    assert ball_bytes(degrade.degrade(degrade.tracking_view(match, match_index=3), "gaps", calibration, 0)[0]) == \
        ball_bytes(elsewhere)
    # Over many seeds the masks are many: the draws are not stuck on a few places.
    drawn = {tuple((half, record["source_half"], record["offset"]) for half, record in
                   degrade.degrade(view, "gaps", calibration, seed)[1]["transplants"].items()) for seed in range(40)}
    assert len(drawn) > 30, len(drawn)
    # The record of the draws is JSON-ready: ints and floats, and no frame.
    assert sorted(info) == ["mask_share", "match_id", "match_index", "profile", "seed", "transplants", "version"]
    assert info["profile"] == "gaps" and info["version"] == 1 and info["match_id"] == "RND001"
    assert sorted(info["transplants"]) == [1, 2]
    assert all(sorted(record) == ["frames", "offset", "source_half"] for record in info["transplants"].values())
    assert [record["frames"] for record in info["transplants"].values()] == list(ROWS)


def test_degrade_gaps_only_touches_ball():
    """The players are bit for bit the clean ones, the ball is missing exactly where the mask says, its speed is NaN."""
    match, calibration = tracked_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    before = {name: getattr(match, name).tobytes() for name in ("players_xy", "players_speed", "ball_xy", "ball_speed",
                                                               "ball_z", "player_team")}
    clean = match_inputs.clean_atd(match)
    for seed in (0, 1, 2):
        atd, info = degrade.degrade(view, "gaps", calibration, seed)
        for name in ("players_xy", "players_speed", "slot_team"):
            assert getattr(atd, name).tobytes() == getattr(clean, name).tobytes(), (seed, name)
            assert getattr(atd, name).dtype == getattr(clean, name).dtype
        assert np.isnan(atd.ball_speed).all() and atd.ball_speed.shape == (match.n_frames,)
        assert atd.ball_xy.shape == (match.n_frames, 2) and atd.n_frames == match.n_frames
        covered = missing = 0
        for half, start, stop in match.graded:
            record = info["transplants"][half]
            mask = calibration.ball_masks[record["source_half"]]
            window = mask[record["offset"]:record["offset"] + (stop - start)]
            assert window.size == stop - start, "these masks are longer than the halves: nothing cycles here"
            gone = np.isnan(atd.ball_xy[start:stop])
            assert (gone[:, 0] == gone[:, 1]).all(), "x and y are missing together"
            assert gone[:, 0].tolist() == window.tolist(), (seed, half, "missing exactly where the mask says")
            kept = ~window
            assert np.array_equal(atd.ball_xy[start:stop][kept], match.ball_xy[start:stop][kept]), "the rest is DFL's"
            covered, missing = covered + (stop - start), missing + int(window.sum())
        assert abs(info["mask_share"] - missing / covered) < 1e-12, "the share of the mask, DFL's gaps not in it"
        separator = np.ones(match.n_frames, dtype=bool)
        for _, start, stop in match.graded:
            separator[start:stop] = False
        assert np.isnan(atd.ball_xy[separator]).all(), "between the halves there is no ball, as in DFL"
        for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
            assert not getattr(atd, name).flags.writeable, name
        assert atd.ball_present.tolist() == (~np.isnan(atd.ball_xy[:, 0])).tolist()
    # The match was not touched, and a DFL ball that was missing stays missing, whatever the mask says.
    assert {name: getattr(match, name).tobytes() for name in before} == before
    # info["mask_share"] is the share of the mask that was transplanted, and not of the ball that is missing after it:
    # DFL's own gaps are not in it. Ten frames of the first half that the mask leaves visible are missing in DFL's own
    # ball, so the ball is missing on exactly ten frames more than the mask says.
    start, stop = match.graded[0][1:]
    _, probe = degrade.degrade(view, "gaps", calibration, 0)
    record = probe["transplants"][1]
    window = calibration.ball_masks[record["source_half"]][record["offset"]:record["offset"] + (stop - start)]
    holes = start + np.flatnonzero(~window)[:10]
    assert holes.size == 10
    holed = np.array(match.ball_xy)
    holed[holes] = np.nan
    holed.setflags(write=False)
    holey = degrade.tracking_view(dataclasses.replace(match, ball_xy=holed))
    atd, info = degrade.degrade(holey, "gaps", calibration, 0)
    assert np.isnan(atd.ball_xy[holes]).all(), "a DFL ball that was missing stays missing, whatever the mask says"
    assert info["mask_share"] == probe["mask_share"], "the mask is the same: the holes are not in it"
    inside = np.zeros(match.n_frames, dtype=bool)
    for _, first, last in match.graded:
        inside[first:last] = True
    actual = np.isnan(atd.ball_xy[inside, 0]).sum() / inside.sum()
    assert abs(actual - info["mask_share"] - 10 / inside.sum()) < 1e-12, (actual, info["mask_share"])
    # clean is DFL's tracking as it is: the ball, its speed and every player.
    atd, info = degrade.degrade(view, "clean", None, 7)
    assert atd.ball_xy.tobytes() == clean.ball_xy.tobytes() and atd.ball_speed.tobytes() == clean.ball_speed.tobytes()
    assert atd.players_xy is clean.players_xy
    assert sorted(info) == ["match_id", "match_index", "profile", "seed", "version"]
    assert info["profile"] == "clean" and info["version"] == 1


def test_degrade_never_sees_truth():
    """No field of the view is the truth, and nothing in the degrader's code reaches for one."""
    names = [field.name for field in dataclasses.fields(degrade.TrackingView)]
    assert names == ["match_id", "match_index", "pitch", "fps", "graded", "kickoffs", "players_xy", "players_speed",
                     "player_team", "ball_xy", "ball_z", "ball_speed"], names
    for name in names:
        assert "status" not in name.lower() and "possession" not in name.lower(), name
    view = degrade.tracking_view(tracked_match())
    for held in vars(view):
        assert "status" not in held and "possession" not in held, held
    assert not hasattr(view, "ball_status") and not hasattr(view, "ball_possession")

    # And the source: apart from its docstrings, which say what is not read, no name, attribute, keyword or string of
    # degrade.py is status or possession.
    with open(PRODUCTION, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node, clean=False):
            docstrings.add(id(node.body[0].value))
    mentioned = []
    for node in ast.walk(tree):
        text = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            text = node.value
        elif isinstance(node, ast.Attribute):
            text = node.attr
        elif isinstance(node, ast.Name):
            text = node.id
        elif isinstance(node, (ast.arg, ast.keyword)):
            text = node.arg
        if text and ("status" in text.lower() or "possession" in text.lower()):
            mentioned.append(text)
    assert not mentioned, "degrade.py reaches for the truth: {}".format(mentioned)


def test_degrade_serves_inputs_without_the_truth():
    """The deep walk of Task 8 on a degraded match, as the runner serves it: nothing in it is the match or its flags."""
    match, calibration = tracked_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    atd, _ = degrade.degrade(view, "gaps", calibration, 0)
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=0)
    assert leaks(inputs, match) == []
    assert leaks(view, match) == [], "the view itself holds none either"
    assert inputs.profile == "gaps" and inputs.atd is atd


def test_degrade_independent_of_match_order():
    """A match gets the same degradation whichever matches are run with it, in whatever order, and alone."""
    matches = {"RND001": tracked_match("RND001", 0), "RND002": tracked_match("RND002", 1, rows=(260, 190)),
               "J03WMX": tracked_match("J03WMX", 2)}
    calibration = made_up_calibration()

    def keyed(order):
        out = {}
        for seed in (0, 1):
            for match_id in order:
                out[(seed, match_id)] = ball_bytes(degrade.degrade(degrade.tracking_view(matches[match_id]), "gaps",
                                                                  calibration, seed)[0])
        return out

    forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
    assert forward == backward, "the order of the matches changed a degradation"
    assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "run alone, a match gets what it got in the run"
    assert len(set(forward.values())) == len(forward), "the matches and the seeds draw apart"
    # match_index: the registry's position for a DFL id, a hash of the id for the others, never the loop's.
    assert degrade.tracking_view(matches["J03WMX"]).match_index == 0
    assert degrade.index_of("J03WR9") == 6
    for match_id in ("RND001", "RND002", "SYN001"):
        expected = int.from_bytes(hashlib.sha256(match_id.encode("utf-8")).digest()[:4], "little")
        assert degrade.index_of(match_id) == expected and expected > 6, match_id
    assert degrade.tracking_view(matches["RND001"]).match_index == degrade.index_of("RND001")
    assert degrade.tracking_view(matches["RND001"], match_index=9).match_index == 9


def test_data_gaps_profile_serves_a_real_match_with_our_real_masks():
    """Our own Calibration on a downloaded DFL match: the real masks, the real layout, nothing of the truth.

    Needs both our match (the Calibration is measured from it, or read from its cache) and a downloaded DFL match, so
    it runs with --data and says NOT RUN where the DFL files are not there. Aggregates only: the shares, and the
    frame-by-frame agreement of the degraded ball with the mask window that the draw says it was cut from, counted
    here and printed nowhere.
    """
    from pipeline.benchmark import sources
    match_id = sources.MATCH_IDS[0]
    for kind in dfl.SOURCE_KINDS:
        if not os.path.isfile(sources.local_path(match_id, kind)):
            raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                match_id, kind, dfl.FETCH_ADVICE))
    cal = calibration.load()
    match = real_match(match_id)
    view = degrade.tracking_view(match)
    clean = match_inputs.clean_atd(match)
    assert view.match_index == 0 and sorted(cal.ball_masks) == [1, 2]
    shares = []
    for seed in (0, 1):
        atd, info = degrade.degrade(view, "gaps", cal, seed)
        for name in ("players_xy", "players_speed", "slot_team"):
            assert getattr(atd, name) is getattr(clean, name), "{}: the players are DFL's own arrays".format(name)
        assert np.isnan(atd.ball_speed).all() and atd.ball_xy.shape == match.ball_xy.shape
        missing = masked = 0
        for half, start, stop in match.graded:
            record = info["transplants"][half]
            window = cal.ball_masks[record["source_half"]][record["offset"]:record["offset"] + (stop - start)]
            assert window.size == stop - start, "a DFL half fits inside one of ours: nothing cycles"
            gone = np.isnan(atd.ball_xy[start:stop, 0])
            assert np.array_equal(gone, window | np.isnan(match.ball_xy[start:stop, 0])), (seed, half)
            missing, masked = missing + int(gone.sum()), masked + int(window.sum())
        frames = sum(stop - start for _, start, stop in match.graded)
        shares.append(missing / frames)
        assert abs(masked / frames - info["mask_share"]) < 1e-9, "the mask's share, which is not the ball's"
        assert shares[-1] >= info["mask_share"] - 1e-12
        inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
        assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"
    ours = cal.scalars["missing_share"]
    assert all(abs(share - ours) < 0.05 for share in shares), (shares, ours)
    print("      {}: ball missing on {} of the frames under seeds 0 and 1, ours {:.1%}".format(
        match_id, " and ".join("{:.1%}".format(share) for share in shares), ours))


def test_degrade_refuses_what_it_cannot_make():
    match, calibration = tracked_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    expect_error(ValueError, lambda: degrade.degrade(view, "context", calibration, 0), "context", "clean", "gaps",
                 "tracking")
    expect_error(ValueError, lambda: degrade.degrade(view, "gaps", None, 0), "gaps", "Calibration",
                 "calibration.load()")
    expect_error(ValueError, lambda: degrade.degrade(view, "tracking", None, 0), "tracking", "Calibration",
                 "calibration.load()")
    # A Calibration without the tracklets is no Calibration for the tracking profile: it says what is missing.
    bare = types.SimpleNamespace(ball_masks=calibration.ball_masks)
    expect_error(ValueError, lambda: degrade.degrade(view, "tracking", bare, 0), "tracking", "present_counts",
                 "tracklet_lengths_s", "calibration.load()")
    expect_error(ValueError, lambda: degrade.degrade(view, "gaps", calibration, -1), "seed", "at least 0")
    expect_error(ValueError, lambda: degrade.degrade(view, "gaps", calibration, 1.0), "seed")
    expect_error(ValueError, lambda: degrade.tracking_view(match, match_index=-2), "match_index")
    # A kick-off outside its block is not a half to degrade.
    late = dataclasses.replace(match, kickoffs={1: 500, 2: match.kickoffs[2]})
    expect_error(ValueError, lambda: degrade.degrade(degrade.tracking_view(late), "gaps", calibration, 0),
                 "RND001", "half 1", "kicks off")


# -- the tracking profile ---------------------------------------------------------------------------------------------

def present_runs(column):
    """[(first, stop)] of the maximal runs of True in a 1-D bool list, written out one frame at a time."""
    runs, first = [], None
    for index, flag in enumerate(column):
        if flag and first is None:
            first = index
        elif not flag and first is not None:
            runs.append((first, index))
            first = None
    if first is not None:
        runs.append((first, len(column)))
    return runs


def test_tracking_parameters_are_worked_out_from_the_calibration():
    """d from the frames with more than 18 present, the lengths in frames at the match's rate, mu from both.

    The made-up Calibration has, over both halves, 50 frames with 22 present, 30 with 20, 20 with 19 and 12 with 18 or
    fewer: the last are left out (CAP is a strict bound, 18 itself is in the tail), so d = 1 - 20.8 / 22 = 3 / 55. Its
    tracklets are 0.4, 1, 2, 3.2 and 4 s: 10, 25, 50, 80 and 100 frames at 25 Hz, mean 53, and 20, 50, 100, 160 and 200
    at 50 Hz. A lost span has the mean mu = 53 * d / (1 - d) = 159 / 52 frames."""
    calibration = made_up_calibration()
    assert degrade.CAP == 18
    assert abs(degrade.lost_share(calibration.present_counts) - 3 / 55) < 1e-12
    parameters = degrade.calibrated_parameters(calibration, 25)
    assert parameters.lengths.tolist() == [10, 25, 50, 80, 100] and parameters.lengths.dtype == np.int64
    assert parameters.mean_tracked == MADE_UP_MEAN and abs(parameters.d - MADE_UP_D) < 1e-12
    assert abs(parameters.mu - MADE_UP_MU) < 1e-12 and abs(MADE_UP_MU - 159 / 52) < 1e-12
    assert not parameters.lengths.flags.writeable
    other_rate = degrade.calibrated_parameters(calibration, 50)
    assert other_rate.lengths.tolist() == [20, 50, 100, 160, 200] and other_rate.mean_tracked == 106.0
    assert abs(other_rate.mu - 106 * (3 / 55) / (52 / 55)) < 1e-12

    # A tracked span is at least a frame, and the seconds are rounded as round() does (halves to even).
    seconds = (0.0, 0.01, 0.02, 0.07, 0.1, 0.14, 4.0, 40.0)   # x 25: 0, 0.25, 0.5, 1.75, 2.5, 3.5, 100, 1000 frames
    short = types.SimpleNamespace(present_counts=calibration.present_counts, tracklet_lengths_s=np.array(seconds))
    assert degrade.calibrated_parameters(short, 25).lengths.tolist() == [max(1, round(s * 25)) for s in seconds] == [
        1, 1, 1, 2, 2, 4, 100, 1000]
    # Only the frames above CAP: a 19 is in and an 18 is not, and the halves are pooled.
    counts = {1: np.array([18, 18, 19], dtype=np.int16), 2: np.array([22, 3], dtype=np.int16)}
    assert abs(degrade.lost_share(counts) - (1 - 20.5 / 22)) < 1e-12
    # Nobody lost: d = 0, no lost span (mu = 0).
    nobody = degrade.calibrated_parameters(types.SimpleNamespace(
        present_counts={1: np.full(10, 22, dtype=np.int16)}, tracklet_lengths_s=np.array([1.0])), 25)
    assert (nobody.d, nobody.mu, nobody.mean_tracked) == (0.0, 0.0, 25.0)

    expect_error(ValueError, lambda: degrade.lost_share({1: np.array([18, 4, 0], dtype=np.int16)}),
                 "more than 18", "present_counts")
    expect_error(ValueError, lambda: degrade.lost_share({}), "present_counts")
    # A lost span is at least a frame, so a mean below 1 cannot be drawn: it is refused, and says why.
    nearly = {1: np.array([22] * 999 + [21], dtype=np.int16)}
    expect_error(ValueError, lambda: degrade.tracking_parameters([1], degrade.lost_share(nearly)), "lost span",
                 "at least 1 frame")
    expect_error(ValueError, lambda: degrade.tracking_parameters([], 0.1), "lengths")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3, 0], 0.1), "lengths", "at least 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3], 1.0), "d", "below 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3], -0.1), "d", "at least 0")
    expect_error(ValueError, lambda: degrade.calibrated_parameters(types.SimpleNamespace(), 25), "present_counts",
                 "tracklet_lengths_s", "calibration.load()")


def test_walk_alternates_tracked_and_lost_spans_and_is_cut_by_the_run():
    """A run starts with a tracked span, a lost span of at least a frame follows each, and the run's end cuts."""
    parameters = degrade.calibrated_parameters(made_up_calibration(), 25)
    lengths = set(parameters.lengths.tolist())
    seen, ends = set(), set()
    for seed in range(60):
        rng = degrade.rng_for(seed, 0, "tracklets", 1)
        for length in (1, 7, 60, 500, 3000):
            spans = degrade.walk_run(rng, length, parameters)
            assert spans and spans[0][0] == 0 and spans[-1][1] <= length, (seed, length, spans)
            assert all(isinstance(value, int) for span in spans for value in span), "plain ints, for JSON and slices"
            assert all(stop - start >= 1 for start, stop in spans)
            assert all(spans[k][1] + 1 <= spans[k + 1][0] for k in range(len(spans) - 1)), "a lost span between each"
            for start, stop in spans[:-1]:
                assert stop - start in lengths, "a span that is followed by another was drawn whole"
                seen.add(stop - start)
            start, stop = spans[-1]
            assert stop - start in lengths or stop == length, "the last is whole, or cut by the end of the run"
            ends.add(stop == length)
            assert stop - start <= max(lengths)
    assert seen == lengths, "every length is drawn"
    assert ends == {True, False}, "a run ends inside a tracked span, and inside a lost one"
    # A run shorter than its first tracked span is that span, cut: one tracklet, the whole run.
    long_first = degrade.tracking_parameters([100], 0.1)
    assert degrade.walk_run(degrade.rng_for(0, 0, "tracklets", 1), 40, long_first) == [(0, 40)]
    # Nobody is lost (mu = 0): the run is one tracklet, whatever the lengths, and nothing is drawn.
    never = degrade.tracking_parameters([5, 9], 0.0)
    rng = degrade.rng_for(0, 0, "tracklets", 1)
    before = rng.bit_generator.state
    assert degrade.walk_run(rng, 1000, never) == [(0, 1000)] and rng.bit_generator.state == before
    # The lost spans are at least 1 even when mu is exactly 1.
    sticky = degrade.tracking_parameters([1], 0.5)    # mu = 1 * 0.5 / 0.5 = 1: a frame tracked, a frame lost
    assert sticky.mu == 1.0
    spans = degrade.walk_run(degrade.rng_for(0, 0, "tracklets", 1), 100, sticky)
    assert spans == [(2 * k, 2 * k + 1) for k in range(50)], spans


def test_lost_share_on_long_synthetic():
    """On long runs the share lost is d, the tracked spans have the lengths' mean and the lost ones have mean mu.

    The mean tracked span is 100 frames (50 to 150, every length once) and d is 0.05 and 0.3, whose mu are 100 * d /
    (1 - d) = 5.26 and 42.86 frames: a mu that forgot the (1 - d) would give a lost share of 0.0476 and 0.2308, which
    the second d tells from d by 0.07. Ten runs of 300,000 frames each: the share's own spread is about 0.001, and the
    brief's tolerance of 0.01 is ten times it."""
    lengths = np.arange(50, 151)
    for d in (0.05, 0.3):
        parameters = degrade.tracking_parameters(lengths, d)
        assert parameters.mean_tracked == 100.0 and abs(parameters.mu - 100 * d / (1 - d)) < 1e-9
        total = tracked = 0
        whole, lost = [], []
        for run in range(10):
            spans = degrade.walk_run(degrade.rng_for(1, 0, "tracklets", run), 300_000, parameters)
            total += 300_000
            tracked += sum(stop - start for start, stop in spans)
            whole += [stop - start for start, stop in spans[:-1]]
            lost += [spans[k + 1][0] - spans[k][1] for k in range(len(spans) - 1)]
        share = 1 - tracked / total
        assert abs(share - d) < 0.01, (d, share)
        assert abs(np.mean(whole) - 100) < 3, (d, np.mean(whole))
        assert abs(np.mean(lost) / parameters.mu - 1) < 0.05, (d, np.mean(lost), parameters.mu)
        assert min(lost) >= 1 and max(lost) > 3 * parameters.mu, "a geometric tail, not a constant"

    # The same through the profile, on a steady match of 2 x 20,000 rows: what it reports and what the arrays show.
    match = steady_match(rows=(20000, 20000))
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    atd, info = degrade.degrade(view, "tracking", calibration, 0)
    clean_present = ~np.isnan(match.players_xy[:, :, 0])
    on_pitch = int(clean_present.sum())
    shown = int(atd.players_present.sum())
    assert abs(1 - shown / on_pitch - MADE_UP_D) < 0.01, (1 - shown / on_pitch, MADE_UP_D)
    assert abs(info["players"]["lost_share"] - (1 - shown / on_pitch)) < 1e-9
    assert info["players"]["on_pitch_frames"] == on_pitch and abs(info["players"]["d"] - MADE_UP_D) < 1e-9


def test_tracklets_never_overlap_in_a_slot():
    """A slot carries one tracklet at a time: no two overlap, and every slot-frame of the arrays has one writer.

    Many short tracklets (3 to 20 frames, 30% of the time lost) so that slots are freed and taken over again all the
    time. Checked on the records and on the arrays: the finite slot-frames are as many as the tracklets' frames, which
    they would not be if two tracklets had written the same one."""
    view = degrade.tracking_view(steady_match(rows=(400, 300)))
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    reused = jumped = 0
    for seed in range(6):
        xy, speed, tracklets = degrade.degraded_players(view, parameters, seed)
        frames = 0
        for half, records in tracklets.items():
            assert len(records) > 30, (seed, half, len(records))
            by_slot, by_player = {}, {}
            for t in records:
                by_slot.setdefault(t.slot, []).append((t.start, t.stop))
                by_player.setdefault(t.player, set()).add(t.slot)
                frames += t.stop - t.start
            for slot, spans in by_slot.items():
                spans.sort()
                assert all(first[1] <= second[0] for first, second in zip(spans, spans[1:])), (seed, half, slot)
                reused += len(spans) > 1
            jumped += any(len(slots) > 1 for slots in by_player.values())
        assert int(np.isfinite(xy[:, :, 0]).sum()) == frames, "two tracklets wrote one slot-frame"
        assert (np.isfinite(xy[:, :, 0]) == np.isfinite(xy[:, :, 1])).all(), "x and y are present together"
    assert reused > 20 and jumped > 5, "slots are taken over and players change slot, as in ours"


def test_each_tracklet_takes_the_lowest_free_slot_of_its_pool():
    """In the order of (start, player), a tracklet takes the lowest slot of its team's pool that is free at its start.

    Worked out from the records alone: the slots below the one taken were all busy at the start (an earlier tracklet,
    still tracked on that frame, had each of them), and the one taken was not. A slot is free for a tracklet starting at
    t when its previous tracklet ended before t, so one that ends at t (half-open: its last frame is t - 1) is free."""
    view = degrade.tracking_view(steady_match(rows=(400, 300)))
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    checked = taken_over = 0
    for seed in range(4):
        _, _, tracklets = degrade.degraded_players(view, parameters, seed)
        for records in tracklets.values():
            assert [(t.start, t.player) for t in records] == sorted((t.start, t.player) for t in records)
            for index, t in enumerate(records):
                pool = range(0, 24) if view.player_team[t.player] == 0 else range(24, 46)
                busy = {u.slot for u in records[:index] if u.stop > t.start}
                assert t.slot in pool and t.slot not in busy, (seed, t)
                assert all(slot in busy for slot in pool if slot < t.slot), (seed, t, sorted(busy))
                checked += 1
                taken_over += any(u.stop == t.start and u.slot == t.slot for u in records[:index])
    assert checked > 500 and taken_over > 0, "a slot freed on a frame is taken on that frame, by another player's"


def test_slot_team_matches_pool():
    """slot_team is 24 home slots then 22 away, and every slot of a pool carries only players of its team."""
    match = steady_match()
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    for seed in (0, 1, 2):
        atd, info = degrade.degrade(view, "tracking", calibration, seed)
        assert atd.slot_team.dtype == np.int8 and atd.slot_team.tolist() == [0] * 24 + [1] * 22
        assert (degrade.HOME_SLOTS, degrade.AWAY_SLOTS, degrade.N_SLOTS) == (24, 22, 46)
        assert atd.players_xy.shape == (match.n_frames, 46, 2) and atd.players_speed.shape == (match.n_frames, 46)
        assert atd.players_xy.dtype == atd.players_speed.dtype == np.float64 and atd.n_slots == 46
        for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
            assert not getattr(atd, name).flags.writeable, name
        # From the arrays alone: at every row, the positions in the home slots are positions of home players on the
        # pitch then, and the same for the away ones (the made-up positions are all different).
        for row in range(match.n_frames):
            for team, slots in ((0, slice(0, 24)), (1, slice(24, 46))):
                shown = {tuple(position) for position in atd.players_xy[row, slots] if not np.isnan(position).any()}
                own = {tuple(position) for position in match.players_xy[row][match.player_team == team]
                       if not np.isnan(position).any()}
                assert shown <= own, (seed, row, team)
    # No other slot count or team layout: the pools are the tracker's, whatever the match's squads.
    big = steady_match(teams=(0,) * 16 + (1,) * 15)
    atd, _ = degrade.degrade(degrade.tracking_view(big), "tracking", calibration, 0)
    assert atd.slot_team.tolist() == [0] * 24 + [1] * 22 and atd.players_xy.shape[1] == 46


def test_tracking_raises_when_a_pool_runs_out():
    """A pool with no free slot is an error that names the match, the half, the frame and the pool: never a guess."""
    parameters = degrade.tracking_parameters([3, 5], 0.2)
    exact = degrade.tracking_view(steady_match(teams=(0,) * 24 + (1,) * 22, substitutions=False))
    for seed in range(3):
        degrade.degraded_players(exact, parameters, seed)    # 24 and 22 on the pitch at once fit exactly
    too_many_home = degrade.tracking_view(steady_match(teams=(0,) * 25 + (1,) * 3, substitutions=False))
    message = expect_error(ValueError, lambda: degrade.degraded_players(too_many_home, parameters, 0),
                           "RND001", "half 1", "frame 0", "home", "24 slots", "players_xy")
    assert "player" in message
    too_many_away = degrade.tracking_view(steady_match("RND002", teams=(0,) * 3 + (1,) * 23, substitutions=False))
    expect_error(ValueError, lambda: degrade.degraded_players(too_many_away, parameters, 0), "RND002", "half 1",
                 "frame 0", "away", "22 slots")
    # Through the profile as well, and in the second half: 25 home players who are only on in the second half.
    late = steady_match(teams=(0,) * 25 + (1,) * 3, substitutions=False)
    xy = np.array(late.players_xy)
    xy[late.graded[0][1]:late.graded[0][2], 6:] = np.nan
    xy.setflags(write=False)
    expect_error(ValueError, lambda: degrade.degrade(degrade.tracking_view(dataclasses.replace(late, players_xy=xy)),
                                                     "tracking", made_up_calibration(), 0),
                 "RND001", "half 2", "frame {}".format(late.graded[1][1]), "home")


def test_tracking_degenerate_equals_gaps():
    """One endless tracklet per stretch of a player (d = 0, huge lengths): each player's positions are those of gaps.

    On the steady match: 7 stretches in the first half and 6 in the second, so 13 tracklets, each a whole stretch, and
    the slots are the lowest free ones, worked out by hand: the home players 0, 1 and 2 take slots 0, 1 and 2 (2 comes
    on at row 60 and finds slot 2 free), the away players 3, 4 and 5 take 24, 25 and 26, and player 5, who is off from
    100 to 130, gets 26 back, which nobody has taken. Then, whatever the mapping, every row has the same positions as
    gaps has, only moved to other slots: the same count per team and the same values."""
    match = steady_match()
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    endless = degrade.tracking_parameters([10 ** 9], 0.0)
    assert endless.mu == 0.0
    xy, speed, tracklets = degrade.degraded_players(view, endless, 3)
    expected = {
        1: [(0, 0, 240, 0), (1, 0, 240, 1), (3, 0, 240, 24), (4, 0, 150, 25), (5, 0, 100, 26), (2, 60, 240, 2),
            (5, 130, 240, 26)],
        2: [(0, 265, 465, 0), (1, 265, 465, 1), (2, 265, 465, 2), (3, 265, 465, 24), (4, 265, 465, 25),
            (5, 265, 465, 26)]}
    assert {half: [(t.player, t.start, t.stop, t.slot) for t in records]
            for half, records in tracklets.items()} == expected
    gaps, _ = degrade.degrade(view, "gaps", calibration, 3)
    assert gaps.players_xy.shape[1] == 6
    for records in tracklets.values():
        for t in records:
            assert np.array_equal(xy[t.start:t.stop, t.slot], gaps.players_xy[t.start:t.stop, t.player]), t
            assert np.isfinite(gaps.players_xy[t.start:t.stop, t.player]).all(), "a tracklet is a whole stretch"
    # Each stretch is a tracklet, found one frame at a time here and not by the code under test.
    for half, start, stop in match.graded:
        stretches = sorted((player, start + first, start + last) for player in range(6)
                           for first, last in present_runs(list(gaps.players_present[start:stop, player])))
        assert sorted((t.player, t.start, t.stop) for t in tracklets[half]) == stretches
    # Whatever the slots: every row has gaps' positions, nobody lost and nobody added.
    for _, start, stop in match.graded:
        shown = np.sort(xy[start:stop, :, 0], axis=1)
        assert np.isnan(shown[:, 6:]).all()
        assert np.array_equal(shown[:, :6], np.sort(gaps.players_xy[start:stop, :, 0], axis=1), equal_nan=True)
    assert np.isnan(xy[240:265]).all(), "the rows between the halves have no players"
    # The whole profile at d = 0 and through the Calibration: nobody lost in any frame.
    never = types.SimpleNamespace(**{**vars(calibration), "present_counts": {1: np.full(5, 22, dtype=np.int16)}})
    atd, info = degrade.degrade(view, "tracking", never, 3)
    assert info["players"]["d"] == 0.0 and info["players"]["lost_frames"] == 0
    assert np.array_equal(np.isfinite(atd.players_xy[:, :, 0]).sum(axis=1), gaps.players_present.sum(axis=1))
    assert ball_bytes(atd) == ball_bytes(gaps), "the ball is gaps' whatever the players do"


def test_tracking_draws_come_from_the_tracklets_stream_in_a_fixed_order():
    """The spans are what rng_for(seed, match_index, "tracklets", half) gives when it is read as documented.

    Players in column order, and for each one his stretches in time order: a tracked length (an index into the lengths),
    then, if the stretch goes on, a lost span (a geometric draw of mean mu), and so on. Written out here with plain
    loops and a generator made from the documented key, so that a change in the order of the draws shows. The other
    halves, seeds and components draw apart."""
    match = steady_match()
    view = degrade.tracking_view(match)
    parameters = degrade.calibrated_parameters(made_up_calibration(), 25)
    lengths = parameters.lengths.tolist()
    assert degrade.COMPONENTS["tracklets"] == 1
    for seed in (0, 4):
        for half, start, stop in match.graded:
            rng = np.random.default_rng([seed, view.match_index, 1, half])
            expected = []
            for player in range(6):
                for first, last in present_runs(list(~np.isnan(match.players_xy[start:stop, player, 0]))):
                    row = first
                    while True:
                        tracked = lengths[int(rng.integers(len(lengths)))]
                        expected.append((player, start + row, start + min(row + tracked, last)))
                        row += tracked
                        if row >= last:
                            break
                        row += int(rng.geometric(1 / parameters.mu))
                        if row >= last:
                            break
            got = [(t.player, t.start, t.stop) for t in degrade.choose_tracklets(view, half, parameters, seed)]
            assert sorted(got) == sorted(expected), (seed, half)
    spans = {seed: [(t.player, t.start, t.stop) for t in degrade.choose_tracklets(view, 1, parameters, seed)]
             for seed in (0, 1)}
    assert spans[0] != spans[1], "another seed draws other spans"
    expect_error(ValueError, lambda: degrade.choose_tracklets(view, 3, parameters, 0), "half 3", "graded")


def test_tracking_speeds_are_finite_differences_within_a_tracklet():
    """A slot's speed is the step from the frame before, in m/s on the pitch, within a tracklet; NaN on its first.

    Worked out with math.hypot, one frame at a time, from the match's own positions through each tracklet's records. A
    slot that another player's tracklet takes over on the very frame it is freed has NaN on that frame: a step from one
    player to another is no speed. The DFL speed channel is not used."""
    match = steady_match(rows=(400, 300))
    view = degrade.tracking_view(match)
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    length, width = match.pitch
    handed_over = checked = 0
    for seed in range(3):
        xy, speed, tracklets = degrade.degraded_players(view, parameters, seed)
        expected = np.full(speed.shape, np.nan)
        firsts = []
        for records in tracklets.values():
            for t in records:
                firsts.append((t.start, t.slot))
                for row in range(t.start + 1, t.stop):
                    dx = (match.players_xy[row, t.player, 0] - match.players_xy[row - 1, t.player, 0]) * length
                    dy = (match.players_xy[row, t.player, 1] - match.players_xy[row - 1, t.player, 1]) * width
                    expected[row, t.slot] = math.hypot(dx, dy) * match.fps
                    checked += 1
        assert np.allclose(speed, expected, rtol=1e-12, atol=0, equal_nan=True), seed
        assert all(np.isnan(speed[row, slot]) for row, slot in firsts), "NaN on the first frame of every tracklet"
        handed_over += sum(np.isfinite(xy[row - 1, slot, 0]) for row, slot in firsts if row > 0)
        assert np.isnan(speed[~np.isfinite(xy[:, :, 0])]).all(), "no speed where nobody is"
    assert checked > 1000 and handed_over > 0, (checked, handed_over)
    # DFL's own speed channel is not used: with another one, the same speeds.
    silent = np.zeros_like(match.players_speed)
    silent.setflags(write=False)
    other = degrade.tracking_view(dataclasses.replace(match, players_speed=silent))
    assert degrade.degraded_players(other, parameters, 2)[1].tobytes() == speed.tobytes()
    # A one-frame tracklet has a first frame and nothing else.
    one = degrade.tracking_parameters([1], 0.5)
    _, speed, _ = degrade.degraded_players(view, one, 0)
    assert np.isnan(speed).all()


def test_tracking_ball_is_exactly_gaps():
    """The ball of the tracking profile is the ball of the gaps profile: the same transplant draws, to the byte.

    The profiles nest: what tracking adds is the players, and the difference between two results is then one
    component. The record of the draws is gaps' as well, with the players' figures beside it."""
    match, calibration = steady_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    for seed in (0, 1, 2):
        tracking, tracking_info = degrade.degrade(view, "tracking", calibration, seed)
        gaps, gaps_info = degrade.degrade(view, "gaps", calibration, seed)
        assert ball_bytes(tracking) == ball_bytes(gaps) and np.isnan(tracking.ball_speed).all()
        assert tracking_info["transplants"] == gaps_info["transplants"]
        assert tracking_info["mask_share"] == gaps_info["mask_share"]
        assert sorted(tracking_info) == sorted(list(gaps_info) + ["players"]), sorted(tracking_info)
        assert (tracking_info["profile"], tracking_info["version"]) == ("tracking", 1)
        assert tracking.players_xy.shape[1] == 46 and gaps.players_xy.shape[1] == 6, "the players are what differs"


def test_tracking_is_deterministic_and_independent_of_the_other_matches():
    """The same seed gives the same bytes and the same record; another seed, match_index or match gives another.

    A match gets the same degradation whichever matches are run with it and in whatever order, and the record of what
    was drawn is JSON-ready: ints and floats only, no frame of anybody."""
    matches = {"RND001": steady_match("RND001", 0), "RND002": steady_match("RND002", 1, rows=(260, 190)),
               "J03WMX": steady_match("J03WMX", 2)}
    calibration = made_up_calibration()

    def everything(atd):
        names = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
        return b"".join(getattr(atd, name).tobytes() for name in names)

    def keyed(order):
        return {(seed, match_id): everything(degrade.degrade(degrade.tracking_view(matches[match_id]), "tracking",
                                                           calibration, seed)[0])
                for seed in (0, 1) for match_id in order}

    forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
    assert forward == backward and len(set(forward.values())) == len(forward)
    assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "alone, a match gets what it got in the run"

    view = degrade.tracking_view(matches["RND001"])
    first, info = degrade.degrade(view, "tracking", calibration, 0)
    again, info_again = degrade.degrade(degrade.tracking_view(matches["RND001"]), "tracking", made_up_calibration(), 0)
    assert everything(first) == everything(again) and info == info_again
    elsewhere, _ = degrade.degrade(degrade.tracking_view(matches["RND001"], match_index=3), "tracking", calibration, 0)
    assert everything(elsewhere) != everything(first) and elsewhere.players_xy.tobytes() != first.players_xy.tobytes()
    assert json.loads(json.dumps(info, sort_keys=True)) == json.loads(json.dumps(info_again, sort_keys=True))
    assert sorted(info["players"]) == ["d", "halves", "lost_frames", "lost_share", "mean_lost_frames",
                                       "mean_tracked_frames", "on_pitch_frames", "tracklets"]
    assert sorted(info["players"]["halves"]) == [1, 2]
    assert all(sorted(record) == ["lost_frames", "on_pitch_frames", "slots_used", "tracklets"]
               for record in info["players"]["halves"].values())
    assert info["players"]["tracklets"] == sum(record["tracklets"] for record in info["players"]["halves"].values())
    assert info["players"]["lost_frames"] == info["players"]["on_pitch_frames"] - int(
        np.isfinite(first.players_xy[:, :, 0]).sum())
    assert abs(info["players"]["mean_tracked_frames"] - MADE_UP_MEAN) < 1e-9
    assert abs(info["players"]["mean_lost_frames"] - MADE_UP_MU) < 1e-9
    assert all(isinstance(value, (int, float)) for value in info["players"].values() if not isinstance(value, dict))
    # Many seeds, many different spans: the draws are not stuck.
    drawn = {degrade.degrade(view, "tracking", calibration, seed)[0].players_xy.tobytes() for seed in range(12)}
    assert len(drawn) == 12


def test_tracking_serves_inputs_without_the_truth():
    """A tracking Atd served as MatchInputs holds nothing of the match or its flags, to any depth (the deep walk)."""
    match, calibration = steady_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    atd, _ = degrade.degrade(view, "tracking", calibration, 0)
    inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"
    assert leaks(view, match) == []
    assert inputs.profile == "tracking" and inputs.atd is atd and inputs.atd.n_slots == 46
    assert inputs.atd_manifest["profile"] == "tracking"
    assert "array" not in repr(atd), "no frame in a repr"


def test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches():
    """No pool of slots runs out on any of the 7 downloaded matches, under several seeds and losses: they fit.

    The slots are assigned from the players' stretches on the pitch (where DFL gives a position), with parameters that
    make a slot change hands as often as it can (tracked spans of 1 to 250 frames, 30% of the time lost) and with the
    ones that lose nobody. The most players of one team on the pitch at once on any frame of the graded halves, and the
    most slots a team uses, are printed per match (aggregates of public data); a pool has 24 and 22."""
    from pipeline.benchmark import sources
    stressing = degrade.tracking_parameters([1, 2, 5, 25, 250], 0.3)
    nobody = degrade.tracking_parameters([10 ** 9], 0.0)
    print("      {:<8} {:>10} {:>16} {:>14}".format("match", "tracklets", "most on, h / a", "slots, h / a"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        view = degrade.tracking_view(match)
        present = ~np.isnan(match.players_xy[:, :, 0])
        most = [int(present[:, match.player_team == team].sum(axis=1).max()) for team in (0, 1)]
        assert most[0] <= degrade.HOME_SLOTS and most[1] <= degrade.AWAY_SLOTS, (match_id, most)
        tracklets, slots = 0, [0, 0]
        for seed, parameters in ((0, stressing), (1, stressing), (2, stressing), (0, nobody)):
            for half, _, _ in view.graded:
                records = degrade.choose_tracklets(view, half, parameters, seed)     # raises if a pool runs out
                if parameters is stressing and seed == 0:
                    tracklets += len(records)
                for team, first in ((0, 0), (1, degrade.HOME_SLOTS)):
                    used = [t.slot - first + 1 for t in records if view.player_team[t.player] == team]
                    slots[team] = max([slots[team]] + used)
        print("      {:<8} {:>10,} {:>16} {:>14}".format(match_id, tracklets, "{} / {}".format(*most),
                                                         "{} / {}".format(*slots)))
        del match, view, present


def test_data_tracking_profile_on_a_real_match_with_our_real_calibration():
    """Our own Calibration on a downloaded DFL match: what the tracking profile's first part makes, for information.

    Needs both our match (the Calibration is measured from it, or read from its cache) and a downloaded DFL match, so
    it runs with --data and says NOT RUN where the DFL files are not there. It prints d, the mean tracked span and the
    mean lost span of ours, and the statistics that calibration.player_stats gives on the degraded match beside the
    Calibration's: the histogram of slots present, the tracklets' median and the NaN-speed share. Nothing of that
    table is asserted: this part of the profile has no wrong-team episodes, no NaN speeds and no low-count tail, and
    the table is there to say by how much; its gates come with the last part. What is asserted is the layout, that the
    ball is gaps', that the draws are the same twice, that nothing of the truth is reachable and that no pool runs out;
    and that the lost share, pooled over the 7 matches and 5 seeds, is d to within 0.01 (the tolerance for long runs; a
    DFL half is 68,000 to 78,000 frames, where a cut at the end of a run is not negligible). Aggregates only.
    """
    from pipeline.benchmark import sources
    match_id = sources.MATCH_IDS[0]
    for kind in dfl.SOURCE_KINDS:
        if not os.path.isfile(sources.local_path(match_id, kind)):
            raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                match_id, kind, dfl.FETCH_ADVICE))
    cal = calibration.load()
    match = real_match(match_id)
    view = degrade.tracking_view(match)
    parameters = degrade.calibrated_parameters(cal, view.fps)
    assert parameters.d == degrade.lost_share(cal.present_counts) and 0 < parameters.d < 0.2
    assert parameters.mu >= 1 and parameters.lengths.size == cal.tracklet_lengths_s.size
    atd, info = degrade.degrade(view, "tracking", cal, 0)
    again, info_again = degrade.degrade(view, "tracking", cal, 0)
    assert info == info_again and atd.players_xy.tobytes() == again.players_xy.tobytes()
    assert atd.players_speed.tobytes() == again.players_speed.tobytes()
    gaps, _ = degrade.degrade(view, "gaps", cal, 0)
    assert ball_bytes(atd) == ball_bytes(gaps)
    assert atd.players_xy.shape == (match.n_frames, 46, 2) and atd.slot_team.tolist() == [0] * 24 + [1] * 22
    inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"

    spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
    mine = calibration.player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy),
                                    atd.players_speed, spans, inputs.fps)

    def share(hist, low, high):
        return hist[low:high + 1].sum() / hist.sum()

    ours = cal.present_hist
    rows = (("slots present, mean", mine["present_mean"], cal.scalars["present_mean"], "{:.2f}"),
            ("all 22 present", share(mine["present_hist"], 22, 22), share(ours, 22, 22), "{:.1%}"),
            ("19 to 21 present", share(mine["present_hist"], 19, 21), share(ours, 19, 21), "{:.1%}"),
            ("15 to 18 present", share(mine["present_hist"], 15, 18), share(ours, 15, 18), "{:.1%}"),
            ("14 or fewer present", share(mine["present_hist"], 0, 14), share(ours, 0, 14), "{:.1%}"),
            ("tracklets", mine["tracklets"], cal.scalars["tracklets"], "{:,}"),
            ("tracklet median, s", float(np.median(mine["tracklet_lengths_s"])), float(np.median(
                cal.tracklet_lengths_s)), "{:.1f}"),
            ("speeds NaN of the present", mine["speed_nan_share"], cal.speed_nan_share, "{:.2%}"),
            ("wrong-team frames", mine["wrong_team_share"], cal.wrong_team_share, "{:.1%}"))
    counts = np.concatenate([cal.present_counts[half] for half in sorted(cal.present_counts)])
    used = np.flatnonzero(atd.players_present.any(axis=0))
    print("      {}: d {:.4f} (ours has more than {} slots present on {:.1%} of its frames), mean tracked span {:,.0f} "
          "frames ({:.1f} s), mean lost span {:.1f} frames ({:.2f} s)".format(
              match_id, parameters.d, degrade.CAP, (counts > degrade.CAP).mean(), parameters.mean_tracked,
              parameters.mean_tracked / view.fps, parameters.mu, parameters.mu / view.fps))
    print("      lost {:.2%} of the {:,} on-pitch frames, {:,} tracklets, {} + {} slots used".format(
        info["players"]["lost_share"], info["players"]["on_pitch_frames"], info["players"]["tracklets"],
        int((used < degrade.HOME_SLOTS).sum()), int((used >= degrade.HOME_SLOTS).sum())))
    print("      {:<26} {:>10} {:>10}".format("information, not gated", "degraded", "ours"))
    for name, degraded, measured, template in rows:
        print("      {:<26} {:>10} {:>10}".format(name, template.format(degraded), template.format(measured)))

    # The lost share over the 7 matches and 5 seeds: the walk and the slots only, no arrays.
    on_pitch = lost = 0
    by_match = []
    for other_id in sources.MATCH_IDS:
        other = degrade.tracking_view(match if other_id == match_id else real_match(other_id))
        present = ~np.isnan(other.players_xy[:, :, 0])
        on = gone = 0
        for seed in range(5):
            for half, start, stop in other.graded:
                records = degrade.choose_tracklets(other, half, parameters, seed)
                on += int(present[start:stop].sum())
                gone += int(present[start:stop].sum()) - sum(t.stop - t.start for t in records)
        by_match.append("{} {:.2%}".format(other_id, gone / on))
        on_pitch, lost = on_pitch + on, lost + gone
    print("      lost share, seeds 0-4: {}; pooled {:.2%} against d {:.2%}".format(
        ", ".join(by_match), lost / on_pitch, parameters.d))
    assert abs(lost / on_pitch - parameters.d) < 0.01, (lost / on_pitch, parameters.d)
