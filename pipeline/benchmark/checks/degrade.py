"""Checks of the degrader (pipeline/benchmark/degrade.py) that every profile shares: the transplant, the random draws,
what the `gaps` profile touches and does not, and that no truth reaches the degrader.

They run on a DflMatch of random positions (checks/common.py's random_match) with the ball put back on every frame of
both halves, so that the share of the ball that a degradation removes is the share of the mask it was given, and on
masks built here, so that every figure is worked out from how they are made. The `tracking` profile's checks (the
tracklets, the slots, the wrong-team episodes, the NaN-speed runs, the smoothing and how it nests in `gaps`) are
checks/tracking.py's. test_data_gaps_profile_serves_a_real_match_with_our_real_masks needs our own match and a
downloaded DFL match (--data, NOT RUN where the DFL files are not there); every other check here reads nothing from the
data directory.
"""

import ast
import dataclasses
import hashlib
import os
import types

import numpy as np

from pipeline.benchmark import calibration, degrade, dfl
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import (MASK_ROWS, ROWS, NotRun, ball_bytes, leaks, made_up_calibration,
                                              real_match, tracked_match)
from pipeline.selftest import expect_error

# Params
PRODUCTION = degrade.__file__


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


def test_transplant_counts_read_the_same_window_as_the_mask():
    """Our slot counts are read through the Transplant exactly as the missing-ball mask is: the same frames of the
    source.

    transplant_counts takes the window of a source of whole numbers that transplant takes of a bool one, so that the
    cap of the tracking profile and the ball's gaps come from the same frames of ours: from the offset, and past the
    source's end round and round its last TAIL_FRAMES frames. Worked out one frame at a time here, with the source's own
    index as its value (so that every value says which frame it is), on a source shorter than the tail and on one
    longer; the result keeps the source's dtype, is a new read-only array and is refused for what transplant refuses."""
    for length in (10, 8000):
        source = np.arange(length, dtype=np.int16)
        tail = min(degrade.TAIL_FRAMES, length)
        for n, offset in ((6, 3), (0, 4), (length, 0), (25, 8), (2 * length + 7, length - 3), (5, length + 2)):
            expected = []
            for j in range(n):
                index = offset + j
                if index >= length:
                    index = length - tail + (index - length) % tail
                expected.append(index)
            got = degrade.transplant_counts(source, n, offset)
            assert got.tolist() == expected, (length, n, offset)
            assert got.dtype == np.int16 and not got.flags.writeable
            # And the mask, from the same arguments, reads the very frames the counts do.
            mask = degrade.transplant(np.arange(length) % 3 == 0, n, offset)
            assert mask.tolist() == [index % 3 == 0 for index in expected], (length, n, offset)
    source = np.arange(10, dtype=np.int16)
    out = degrade.transplant_counts(source, 4, 2)
    assert source.tolist() == list(range(10)) and out.base is None, "a new array, not a view of the source"
    expect_error(ValueError, lambda: degrade.transplant_counts(source.astype(float), 3, 0), "source", "whole numbers")
    expect_error(ValueError, lambda: degrade.transplant_counts(source.astype(bool), 3, 0), "source", "whole numbers")
    expect_error(ValueError, lambda: degrade.transplant_counts(source[:0], 3, 0), "source", "non-empty")
    expect_error(ValueError, lambda: degrade.transplant_counts(source.reshape(2, 5), 3, 0), "source", "1-D")
    expect_error(ValueError, lambda: degrade.transplant_counts(source, -1, 0), "n", "at least 0")
    expect_error(ValueError, lambda: degrade.transplant_counts(source, 3, -2), "offset", "at least 0")


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
    assert degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 3, "context": 1, "generic": 1}
    assert degrade.COMPONENTS == {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4, "generic_gaps": 5}, (
        "the cap draws nothing, so version 2 has no stream of its own")
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
    expect_error(ValueError, lambda: degrade.degrade(view, "nonsense", calibration, 0), "nonsense", "clean", "gaps",
                 "tracking", "context", "generic")
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
