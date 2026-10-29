"""Checks of the `offpitch` profile of the degrader (pipeline/benchmark/degrade.py): the tracking profile exactly, with
its ball's transplanted gaps thinned and the ball hidden once it is more than a margin past a line, until it is back on
the pitch.

What is held against it here is each part of that sentence, one at a time. The off-pitch mask (offpitch_mask) on hand-
built line distances: the boundaries (a ball exactly on a line is on the pitch and ends a stretch, a ball exactly at -m
is not past the margin), a frame without a ball that neither starts nor ends a stretch, a half that ends off the pitch,
and a walk around -m that a frame-by-frame rule would make flicker. The thinning (thin_mask) on a hand-built mask: whole
interior gaps only, never an edge one, never a frame added. The line distance (line_distance_m) in metres on the
match's own pitch, which is not square and not the same for every match. Then the profile as a whole: its mask is the
thinned transplant ORed with the off-pitch stretches, restated here one frame at a time (reference_masks) from the
documentation and not from the code; its players are tracking's byte for byte; at no thinning and no margin its ball is
tracking's too; it never sees the truth; its draws depend on the seed, the match and the half and on nothing else.

The profile's two constants (OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB, with the fit's record OFFPITCH_FIT) are fitted once
by `offpitch.py fit` (R015) and refused while they are None. A check that needs the profile to run sets them for its
own duration (fitted) and puts them back in a finally, so that every check here passes whether they are pinned or not;
test_offpitch_refuses_until_fitted sets them to None itself. Everything here is synthetic: no data directory is read.
"""

import contextlib
import dataclasses
import json
import math
import types

import numpy as np

from pipeline.benchmark import degrade
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import ROWS, leaks, made_up_calibration, steady_match
from pipeline.selftest import expect_error

# Params
# The record of the ball (info["ball"]): the margin, the drop probability, the Transplant of each half, and counts per
# half and in all. No frame.
BALL_COUNTS = ("interior_gaps", "gaps_dropped", "frames_restored", "offpitch_stretches", "offpitch_frames",
               "missing_frames", "rows")
BALL_KEYS = ("margin_m", "drop_prob", "transplants") + BALL_COUNTS + ("halves",)
INFO_KEYS = ["ball", "mask_share", "match_id", "match_index", "players", "profile", "seed", "tracking_version",
             "transplants", "version"]
MADE_UP_FIT = {"made_up": True}   # what OFFPITCH_FIT is set to while a check sets the two constants: anything but None
THINNING = 5                      # the thinning's place in the key of rng_for: after the hazard's 4


# -- helpers ----------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def fitted(margin_m, drop_prob, fit=MADE_UP_FIT):
    """degrade's OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB and OFFPITCH_FIT set to these for the duration, then put back."""
    saved = degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT
    degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT = margin_m, drop_prob, fit
    try:
        yield
    finally:
        degrade.OFFPITCH_MARGIN_M, degrade.OFFPITCH_DROP_PROB, degrade.OFFPITCH_FIT = saved


def everything(atd):
    """The bytes of every array of a degraded Atd: equal bytes are the same tracking, to the last bit."""
    names = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
    return b"".join(getattr(atd, name).tobytes() for name in names)


def runs_of(flags):
    """[(first, stop)] of the maximal runs of True in a list or 1-D array of flags, half-open, by a plain loop."""
    flags, runs, row = [bool(flag) for flag in flags], [], 0
    while row < len(flags):
        if not flags[row]:
            row += 1
            continue
        stop = row
        while stop < len(flags) and flags[stop]:
            stop += 1
        runs.append((row, stop))
        row = stop
    return runs


def offpitch_match(match_id="RND001", seed=0, rows=ROWS, pitch=(105.0, 68.0), holes=True):
    """steady_match whose ball goes past the lines and comes back, with a few rows that DFL has no ball for.

    On each half the ball follows two slow waves, x = 0.5 + 0.53 sin(2 pi t / 130 + 1) and y = 0.5 + 0.56 sin(2 pi t /
    80), plus a jitter of 0.004 (normalised) a row: on a 105 x 68 m pitch it goes up to about 3 m past a goal line and
    4 m past a touch line, several times a half, and wobbles about every depth on the way, so that a frame-by-frame rule
    at a margin of 1 m would flicker. With `holes`, one row in 20 of each half has no ball (x and y together)."""
    match = steady_match(match_id, seed, rows)
    rng = np.random.default_rng([31, seed])
    ball = np.array(match.ball_xy)
    for _, start, stop in match.graded:
        t = np.arange(stop - start)
        ball[start:stop, 0] = 0.5 + 0.53 * np.sin(2 * np.pi * t / 130.0 + 1.0) + rng.normal(0, 0.004, t.size)
        ball[start:stop, 1] = 0.5 + 0.56 * np.sin(2 * np.pi * t / 80.0) + rng.normal(0, 0.004, t.size)
        if holes:
            ball[start + rng.choice(t.size, size=t.size // 20, replace=False)] = np.nan
    ball.setflags(write=False)
    return dataclasses.replace(match, ball_xy=ball, pitch=tuple(pitch))


def silent_calibration():
    """made_up_calibration with masks that say the ball is never missing: only the off-pitch mask can hide it."""
    base = made_up_calibration()
    return types.SimpleNamespace(**{**vars(base), "ball_masks": {
        half: np.zeros(len(mask), dtype=bool) for half, mask in base.ball_masks.items()}})


def drawn_mask(atd, match, half):
    """(n,) bool: the rows of a half, from its kick-off, where the degraded ball is missing and DFL's ball is there."""
    kickoff, stop = match.kickoffs[half], next(last for number, _, last in match.graded if number == half)
    return np.isnan(atd.ball_xy[kickoff:stop, 0]) & ~np.isnan(match.ball_xy[kickoff:stop, 0])


def reference_line(ball, pitch):
    """min(x, L - x, y, W - y) in metres of each row of a normalised ball, one row at a time; NaN where it is
    missing."""
    length, width = pitch
    out = []
    for x, y in np.asarray(ball, dtype=np.float64).tolist():
        if math.isnan(x) or math.isnan(y):
            out.append(math.nan)
            continue
        out.append(min(x * length, length - x * length, y * width, width - y * width))
    return np.array(out, dtype=np.float64)


def reference_hidden(line, margin):
    """The off-pitch rule one frame at a time: a frame back on the pitch (line distance at least 0) is visible and ends
    a stretch, a frame more than `margin` past a line is hidden and starts one, and any other frame (a ball between
    the line and the margin, or no ball) is as the frame before it; the first frame starts visible."""
    hidden, out = False, []
    for value in np.asarray(line, dtype=np.float64).tolist():
        if value >= 0:
            hidden = False
        elif value < -margin:
            hidden = True
        out.append(hidden)
    return out


def reference_thinned(mask, uniforms, drop_prob):
    """The mask with its interior gaps (runs of True touching neither end) dropped whole where their uniform, taken in
    time order, is below `drop_prob`; the edge gaps as they were."""
    flags = [bool(flag) for flag in mask]
    out, k = list(flags), 0
    for first, stop in runs_of(flags):
        if first > 0 and stop < len(flags):
            if uniforms[k] < drop_prob:
                out[first:stop] = [False] * (stop - first)
            k += 1
    assert k == len(uniforms), (k, len(uniforms))
    return out


def reference_masks(view, calibration, seed, margin, drop_prob):
    """{half: (mask, counts)} of the profile as the documentation builds it, from the half's kick-off.

    The transplant of the half (choose_transplant, transplant: checked in checks/degrade.py), its interior gaps dropped
    with the uniforms of np.random.default_rng([seed, match_index, 5, half]), one per interior gap in time order, then
    ORed with the off-pitch rule on the line distance of the CLEAN ball in metres on the view's pitch."""
    out = {}
    for half, _, stop in view.graded:
        kickoff = view.kickoffs[half]
        choice = degrade.choose_transplant(view, half, calibration, seed)
        base = degrade.transplant(calibration.ball_masks[choice.source_half], stop - kickoff, choice.offset).tolist()
        interior = [run for run in runs_of(base) if run[0] > 0 and run[1] < len(base)]
        uniforms = np.random.default_rng([seed, view.match_index, THINNING, half]).random(len(interior))
        thinned = reference_thinned(base, uniforms, drop_prob)
        off = reference_hidden(reference_line(np.asarray(view.ball_xy)[kickoff:stop], view.pitch), margin)
        mask = [a or b for a, b in zip(thinned, off)]
        counts = {"interior_gaps": len(interior), "gaps_dropped": int((uniforms < drop_prob).sum()),
                  "frames_restored": sum(base) - sum(thinned), "offpitch_stretches": len(runs_of(off)),
                  "offpitch_frames": sum(off), "missing_frames": sum(mask), "rows": stop - kickoff}
        out[half] = (mask, counts)
    return out


# -- the off-pitch mask -----------------------------------------------------------------------------------------------

def test_offpitch_mask_hysteresis_on_a_hand_built_line():
    """Hidden from the first frame more than m past a line to the first later frame back on the pitch, worked out by
    hand at m = 2.

    Rows 0-2 are visible (5 and 1 on the pitch, -0.5 past the line but not past the margin, and nothing to end). Row 3
    (-2.5) is past the margin: hidden. Rows 4-6 (-1.0, -3, -0.2) are still off the pitch: hidden, whatever their
    depth. Row 7 is exactly on a line (0.0), which is on the pitch: visible, and the stretch has ended. Row 9 is exactly
    at -2.0, which is not past -2: visible, and so is row 10 (-1.5), which follows it. One stretch, rows 3-6."""
    line = np.array([5, 1, -0.5, -2.5, -1.0, -3, -0.2, 0.0, 2, -2.0, -1.5, 1])
    mask = degrade.offpitch_mask(line, 2.0)
    assert mask.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]   # -2.0 is not past -2: visible
    assert mask.dtype == bool and mask.shape == line.shape and not mask.flags.writeable
    assert mask.tolist() == reference_hidden(line, 2.0)
    # At a margin of 0 every frame past a line starts a stretch, and the frame back on the line ends it.
    assert degrade.offpitch_mask(line, 0.0).tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    # At an infinite margin nothing is ever past it.
    assert not degrade.offpitch_mask(line, math.inf).any()
    assert degrade.offpitch_mask(np.zeros(0), 1.0).tolist() == []


def test_offpitch_mask_nan_neither_starts_nor_ends_a_stretch():
    """A frame without a ball (NaN) is as the frame before it: visible before a stretch, hidden inside one.

    At m = 2: row 1 (NaN) follows a visible row, so it stays visible; row 2 (-3) starts a stretch; rows 3 and 5 (NaN)
    and row 4 (-1) stay hidden; row 6 (0.5) ends it; row 7 (-1) is not past the margin."""
    line = np.array([1, np.nan, -3, np.nan, -1, np.nan, 0.5, -1])
    assert degrade.offpitch_mask(line, 2.0).tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert not degrade.offpitch_mask(np.full(6, np.nan), 0.0).any(), "no ball at all: nothing starts"
    # A NaN cannot start a stretch at the very first frame either, nor end one at the last.
    assert degrade.offpitch_mask(np.array([np.nan, -3, np.nan]), 2.0).tolist() == [0, 1, 1]


def test_offpitch_mask_stays_hidden_to_the_end():
    """A half that ends with the ball off the pitch stays hidden to its end, from its very first frame if it starts
    there."""
    assert degrade.offpitch_mask(np.array([-3, -1, -0.5]), 2.0).tolist() == [1, 1, 1]
    assert degrade.offpitch_mask(np.array([4, -2.5, -0.1, np.nan, -1.9]), 2.0).tolist() == [0, 1, 1, 1, 1]


def test_offpitch_mask_never_flickers():
    """On a random walk around -m, with frames without a ball in it, every stretch starts on a frame past -m, holds no
    frame back on the pitch and ends only on one (or at the end); no visible frame is past -m; and the stretches are as
    many as the times the ball, visible, went past -m (the plain loop's starts). A frame-by-frame rule would have made
    several times as many runs."""
    rng = np.random.default_rng(17)
    margin, value = 2.0, -2.0
    line = np.empty(6000)
    for t in range(line.size):
        value += 0.05 * (-margin - value) + rng.normal(0, 0.4)
        line[t] = value
    line[rng.uniform(size=line.size) < 0.05] = np.nan
    mask = degrade.offpitch_mask(line, margin)
    assert mask.tolist() == reference_hidden(line, margin)
    stretches = runs_of(mask)
    for first, stop in stretches:
        assert line[first] < -margin, (first, line[first])
        assert stop == line.size or line[stop] >= 0, (stop, line[stop])
        assert not (line[first:stop] >= 0).any(), "a frame back on the pitch inside a stretch"
    assert not (~mask & (line < -margin)).any(), "a visible frame past -m"
    plain = reference_hidden(line, margin)
    starts = sum(1 for t in range(line.size) if plain[t] and (t == 0 or not plain[t - 1]))
    assert len(stretches) == starts >= 10, (len(stretches), starts)
    assert len(runs_of(line < -margin)) > 3 * len(stretches), "the walk would have made a frame-by-frame rule flicker"


# -- the thinning -----------------------------------------------------------------------------------------------------

def test_offpitch_thinning_drops_whole_interior_gaps_only():
    """Whole interior gaps, each with its own uniform in time order; the edge gaps never; nothing added.

    The mask [1,1,0,1,1,0,0,1,0,1,1] has four gaps: rows 0-1 (touches row 0: an edge gap), rows 3-4 and row 7
    (interior), rows 9-10 (touches the last row: an edge gap). With uniforms [0.1, 0.9] at q = 0.5 only rows 3-4 go;
    at q = 0 nothing does; at q = 1 both interior gaps go and the edge ones stay. At a larger q the dropped gaps are a
    superset of a smaller q's (the same uniforms): the missing share falls with q, which the fit reads."""
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    uniforms = np.array([0.1, 0.9])
    once = degrade.thin_mask(mask, uniforms, 0.5)
    assert once.tolist() == [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    assert degrade.thin_mask(mask, uniforms, 0.0).tolist() == mask.tolist()
    assert degrade.thin_mask(mask, uniforms, 1.0).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert degrade.thin_mask(mask, uniforms, 0.1).tolist() == mask.tolist(), "a uniform equal to q is not below it"
    assert once.dtype == bool and not once.flags.writeable
    assert mask.tolist() == [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], "the input is not changed"
    expect_error(ValueError, lambda: degrade.thin_mask(mask, np.array([0.1]), 0.5), "thin_mask", "uniforms", "2")
    # On random masks: a subset of the input, every run kept whole or dropped whole, the edge runs kept, as the plain
    # loop says, and monotone in q.
    rng = np.random.default_rng(23)
    for _ in range(40):
        flags = rng.uniform(size=int(rng.integers(1, 120))) < rng.uniform(0.2, 0.8)
        interior = [run for run in runs_of(flags) if run[0] > 0 and run[1] < flags.size]
        draws = rng.uniform(size=len(interior))
        previous = flags
        for q in (0.0, 0.2, 0.5, 0.8, 1.0):
            thinned = degrade.thin_mask(flags, draws, q)
            assert thinned.tolist() == reference_thinned(flags, draws, q), q
            assert not (thinned & ~flags).any() and not (thinned & ~previous).any(), "a subset, and smaller as q grows"
            for first, stop in runs_of(flags):
                assert len(set(thinned[first:stop].tolist())) == 1, "a gap is dropped whole or kept whole"
                if first == 0 or stop == flags.size:
                    assert thinned[first:stop].all(), "an edge gap is never dropped"
            previous = thinned
    assert degrade.thin_mask(np.ones(5, dtype=bool), np.zeros(0), 1.0).all(), "one gap touching both ends: an edge one"


# -- the profile ------------------------------------------------------------------------------------------------------

def test_offpitch_mask_is_the_thinned_transplant_or_the_off_pitch_stretches():
    """The ball's mask of every half, and every count of the record, as reference_masks restates the construction.

    The transplant of tracking's draws, thinned with the "thinning" stream's uniforms, ORed with the off-pitch rule on
    the clean ball's line distance in metres. On a match whose ball goes past the lines with rows that DFL has no ball
    for, and on the same match with its second half kicking off 30 rows into its block (the mask covers from the
    kick-off and the rows before it keep DFL's ball); two settings of the constants and three seeds."""
    match = offpitch_match()
    window = dataclasses.replace(match, kickoffs={1: match.kickoffs[1], 2: match.kickoffs[2] + 30})
    cal = made_up_calibration()
    dropped = stretches = 0
    for margin, drop_prob in ((1.0, 0.3), (2.5, 0.15)):
        with fitted(margin, drop_prob):
            for case in (match, window):
                view = degrade.tracking_view(case)
                for seed in (0, 1, 2):
                    atd, info = degrade.degrade(view, "offpitch", cal, seed)
                    ball = info["ball"]
                    assert ball["margin_m"] == margin and ball["drop_prob"] == drop_prob
                    wanted = reference_masks(view, cal, seed, margin, drop_prob)
                    for half, start, stop in case.graded:
                        kickoff = case.kickoffs[half]
                        mask, counts = wanted[half]
                        seen = ~np.isnan(case.ball_xy[kickoff:stop, 0])
                        assert drawn_mask(atd, case, half).tolist() == (np.array(mask, dtype=bool) & seen).tolist(), (
                            margin, seed, half)
                        assert ball["halves"][half] == counts, (margin, seed, half, ball["halves"][half], counts)
                        assert np.array_equal(atd.ball_xy[start:kickoff], case.ball_xy[start:kickoff],
                                              equal_nan=True), "before the kick-off the ball is DFL's"
                        dropped += counts["gaps_dropped"]
                        stretches += counts["offpitch_stretches"]
                    assert ball["transplants"] == info["transplants"]
    assert dropped > 10 and stretches > 10, "a check with nothing dropped or hidden proves nothing: {} {}".format(
        dropped, stretches)


def test_offpitch_at_no_thinning_and_no_margin_is_trackings_ball():
    """With q = 0 and an infinite margin nothing is dropped and nothing is past the margin: the ball is tracking's, byte
    for byte (the same transplant draws, the same _remove_ball), on a ball that does go past the lines, for three seeds;
    so is the share of the mask, and the record says that nothing was dropped or hidden."""
    view = degrade.tracking_view(offpitch_match())
    cal = made_up_calibration()
    with fitted(math.inf, 0.0):
        for seed in (0, 1, 2):
            offpitch, info = degrade.degrade(view, "offpitch", cal, seed)
            tracking, tracking_info = degrade.degrade(view, "tracking", cal, seed)
            assert offpitch.ball_xy.tobytes() == tracking.ball_xy.tobytes(), seed
            assert offpitch.ball_speed.tobytes() == tracking.ball_speed.tobytes(), seed
            assert everything(offpitch) == everything(tracking), seed
            assert info["mask_share"] == tracking_info["mask_share"]
            ball = info["ball"]
            assert ball["gaps_dropped"] == ball["frames_restored"] == 0
            assert ball["offpitch_stretches"] == ball["offpitch_frames"] == 0
            assert ball["interior_gaps"] > 0 and ball["missing_frames"] > 0


def test_offpitch_players_equal_tracking():
    """Every player array of an offpitch Atd, the players' record and the draws that the cap reads are tracking's, byte
    for byte, for the same seed: the profile changes the ball and nothing else.

    On a match whose ball goes past the lines and on one whose ball never does (only the thinning acts there), three
    seeds each. The ball is not tracking's on any of them, has no x without a y and no speed, comes back only where a
    gap was dropped, and every array is read-only. The record names the profile, its version and the version of
    tracking's players it is built from."""
    cal = made_up_calibration()
    differs = []
    with fitted(1.0, 0.3):
        for match in (offpitch_match(), steady_match("RND002", 1, rows=(260, 190))):
            view = degrade.tracking_view(match)
            for seed in (0, 1, 2):
                offpitch, info = degrade.degrade(view, "offpitch", cal, seed)
                tracking, tracking_info = degrade.degrade(view, "tracking", cal, seed)
                for name in ("players_xy", "players_speed", "slot_team"):
                    assert getattr(offpitch, name).tobytes() == getattr(tracking, name).tobytes(), (seed, name)
                    assert getattr(offpitch, name).dtype == getattr(tracking, name).dtype
                assert info["players"] == tracking_info["players"]
                assert info["transplants"] == tracking_info["transplants"]
                differs.append(offpitch.ball_xy.tobytes() != tracking.ball_xy.tobytes())
                assert np.isnan(offpitch.ball_speed).all() and offpitch.ball_speed.shape == (match.n_frames,)
                gone = np.isnan(offpitch.ball_xy)
                assert (gone[:, 0] == gone[:, 1]).all(), "x and y are missing together"
                back = int((~gone[:, 0] & np.isnan(tracking.ball_xy[:, 0])).sum())
                assert back <= info["ball"]["frames_restored"], "a ball comes back only where a gap was dropped"
                assert (info["profile"], info["version"], info["tracking_version"]) == (
                    "offpitch", 1, degrade.PROFILES["tracking"])
                for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
                    assert not getattr(offpitch, name).flags.writeable, name
    assert all(differs), "the thinning or the margin changed the ball on every seed of both matches: {}".format(differs)


def test_offpitch_hides_every_frame_past_the_margin():
    """At m = 1.0 (and q = 0.1) no frame of a span shows a ball more than 1.0 m past a line, on a ball that goes up to
    about 4 m past them, for three seeds; and the frames it shows past a line are within the margin."""
    match = offpitch_match()
    view = degrade.tracking_view(match)
    cal = made_up_calibration()
    past = shown_beyond = 0
    with fitted(1.0, 0.1):
        for seed in (0, 1, 2):
            atd, info = degrade.degrade(view, "offpitch", cal, seed)
            for half, _, stop in match.graded:
                kickoff = match.kickoffs[half]
                line = degrade.line_distance_m(match.ball_xy[kickoff:stop], match.pitch)
                seen = ~np.isnan(atd.ball_xy[kickoff:stop, 0])
                assert not (seen & (line < -1.0)).any(), (seed, half)
                past += int((line < -1.0).sum())
                shown_beyond += int((seen & (line < 0)).sum())
            assert info["ball"]["offpitch_stretches"] > 0 and info["ball"]["offpitch_frames"] > 0
    assert past > 30, past
    assert shown_beyond > 0, "the ball is still shown just past a line, before it reaches the margin"


def test_offpitch_ball_reads_the_matchs_own_pitch_in_metres():
    """The line distance is in metres on the match's own pitch: x times the length, y times the width.

    By hand on a 100 x 60 m pitch, and through the profile: the ball stands in the middle except on rows 50-79 of each
    half (from its kick-off), where it is at x = -0.02, and rows 120-149, where it is at y = -0.03. At m = 1.5, with
    masks of ours that hide nothing: on 105 x 68 m the two are 2.1 and 2.04 m past a line, both hidden; on 60 x 68 m the
    first is only 1.2 m past, visible; on 105 x 40 m the second is only 1.2 m past, visible. The same normalised ball
    hides different frames, as line_distance_m in metres says."""
    line = degrade.line_distance_m(np.array([[0.5, 0.5], [-0.01, 0.5], [0.5, 1.02], [0.25, 0.1], [np.nan, 0.5],
                                             [0.5, np.nan]]), (100.0, 60.0))
    assert line[:4].tolist() == [30.0, -0.01 * 100.0, 60.0 - 1.02 * 60.0, 0.1 * 60.0], line[:4].tolist()
    assert np.isnan(line[4:]).all() and line.dtype == np.float64 and not line.flags.writeable
    assert degrade.line_distance_m(np.array([[0.5, -0.05]]), (100.0, 40.0)).tolist() == [-0.05 * 40.0]
    assert degrade.line_distance_m(np.zeros((0, 2)), (100.0, 60.0)).shape == (0,)
    base = steady_match(substitutions=False)
    ball = np.array(base.ball_xy)
    for half, _, stop in base.graded:
        kickoff = base.kickoffs[half]
        ball[kickoff:stop] = 0.5
        ball[kickoff + 50:kickoff + 80, 0] = -0.02
        ball[kickoff + 120:kickoff + 150, 1] = -0.03
    ball.setflags(write=False)
    expected = {(105.0, 68.0): (True, True), (60.0, 68.0): (False, True), (105.0, 40.0): (True, False)}
    cal = silent_calibration()
    hidden = set()
    with fitted(1.5, 0.0):
        for pitch, (by_x, by_y) in expected.items():
            match = dataclasses.replace(base, ball_xy=ball, pitch=pitch)
            atd, info = degrade.degrade(degrade.tracking_view(match), "offpitch", cal, 0)
            masks = []
            for half, _, stop in match.graded:
                kickoff = match.kickoffs[half]
                want = np.zeros(stop - kickoff, dtype=bool)
                want[50:80], want[120:150] = by_x, by_y
                got = drawn_mask(atd, match, half)
                assert got.tolist() == want.tolist(), (pitch, half)
                line = degrade.line_distance_m(match.ball_xy[kickoff:stop], pitch)
                assert got.tolist() == degrade.offpitch_mask(line, 1.5).tolist()
                masks.append(got.tobytes())
            hidden.add(tuple(masks))
            assert info["ball"]["offpitch_stretches"] == 2 * (by_x + by_y) and info["ball"]["interior_gaps"] == 0
    assert len(hidden) == 3, "three pitches, three masks"


def test_offpitch_never_sees_truth():
    """Two matches that differ only in ball_status and ball_possession give byte-identical offpitch Atds and records,
    through the degrader and through the runner's own path (make_atd), for two seeds; the inputs that an offpitch match
    is served with hold nothing of the truth, to any depth, and neither does the view."""
    match = offpitch_match()
    rng = np.random.default_rng([12])
    status = rng.integers(0, 2, size=match.n_frames).astype(np.int8)
    possession = rng.integers(0, 2, size=match.n_frames).astype(np.int8)
    for array in (status, possession):
        array.setflags(write=False)
    other = dataclasses.replace(match, ball_status=status, ball_possession=possession)
    assert not np.array_equal(match.ball_status, other.ball_status) and not np.array_equal(
        match.ball_possession, other.ball_possession), "the two matches must differ in their flags"
    cal = made_up_calibration()
    with fitted(1.0, 0.3):
        for seed in (0, 1):
            first, info = degrade.degrade(degrade.tracking_view(match), "offpitch", cal, seed)
            second, info_other = degrade.degrade(degrade.tracking_view(other), "offpitch", cal, seed)
            assert everything(first) == everything(second) and info == info_other, seed
            assert everything(benchmark_run.make_atd(match, "offpitch", seed, cal)) == everything(first)
            assert everything(benchmark_run.make_atd(other, "offpitch", seed, cal)) == everything(first)
            assert info["ball"]["missing_frames"] > 0 and info["ball"]["offpitch_frames"] > 0
        view = degrade.tracking_view(match)
        atd, _ = degrade.degrade(view, "offpitch", cal, 0)
    inputs = match_inputs.MatchInputs(match, atd, profile="offpitch", seed=0)
    assert leaks(inputs, match) == [] and leaks(view, match) == []
    assert inputs.profile == "offpitch" and inputs.atd is atd and inputs.atd.n_slots == 46
    assert "array" not in repr(atd), "no frame in a repr"


def test_offpitch_is_deterministic_and_independent_of_the_other_matches():
    """The same seed gives the same bytes and the same record; another seed, match_index or match gives another ball.

    A match gets the same degradation whichever matches are run with it and in whatever order, and the record of the
    ball is JSON-ready: ints and floats, no frame of anybody, and each total is the sum of its halves'."""
    matches = {"RND001": offpitch_match("RND001", 0), "RND002": offpitch_match("RND002", 1, rows=(260, 190)),
               "J03WMX": offpitch_match("J03WMX", 2)}
    cal = made_up_calibration()
    with fitted(1.0, 0.3):
        def keyed(order):
            return {(seed, match_id): everything(degrade.degrade(degrade.tracking_view(matches[match_id]), "offpitch",
                                                                 cal, seed)[0])
                    for seed in (0, 1) for match_id in order}

        forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
        assert forward == backward and len(set(forward.values())) == len(forward)
        assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "alone, a match gets what it got in the run"
        view = degrade.tracking_view(matches["RND001"])
        first, info = degrade.degrade(view, "offpitch", cal, 0)
        again, info_again = degrade.degrade(degrade.tracking_view(matches["RND001"]), "offpitch",
                                            made_up_calibration(), 0)
        assert everything(first) == everything(again) and info == info_again
        elsewhere, _ = degrade.degrade(degrade.tracking_view(matches["RND001"], match_index=3), "offpitch", cal, 0)
        assert elsewhere.ball_xy.tobytes() != first.ball_xy.tobytes(), "another match_index is another stream"
        drawn = {degrade.degrade(view, "offpitch", cal, seed)[0].ball_xy.tobytes() for seed in range(12)}
        assert len(drawn) == 12, "another seed is another ball: the draws are not stuck"
    assert json.loads(json.dumps(info, sort_keys=True, allow_nan=False)) == json.loads(
        json.dumps(info_again, sort_keys=True, allow_nan=False))
    assert sorted(info) == INFO_KEYS, sorted(info)
    ball = info["ball"]
    assert sorted(ball) == sorted(BALL_KEYS) and sorted(ball["halves"]) == [1, 2]
    assert all(sorted(record) == sorted(BALL_COUNTS) for record in ball["halves"].values())
    for name in BALL_COUNTS:
        assert ball[name] == sum(record[name] for record in ball["halves"].values()), name
        assert isinstance(ball[name], int), name
    assert ball["transplants"] == info["transplants"] and sorted(ball["transplants"]) == [1, 2]
    assert ball["rows"] == sum(stop - start for _, start, stop in view.graded)
    assert abs(info["mask_share"] - ball["missing_frames"] / ball["rows"]) < 1e-12, "the mask's share"
    # The mask's frames are the frames it took from DFL's ball, and at most the rows that DFL has no ball for besides.
    drawn = sum(int(drawn_mask(first, matches["RND001"], half).sum()) for half in (1, 2))
    holes = sum(int(np.isnan(matches["RND001"].ball_xy[view.kickoffs[half]:stop, 0]).sum())
                for half, _, stop in view.graded)
    assert drawn <= ball["missing_frames"] <= drawn + holes and holes > 0
    assert 0 < ball["gaps_dropped"] <= ball["interior_gaps"] and ball["frames_restored"] >= ball["gaps_dropped"]


def test_offpitch_is_registered_with_its_own_stream():
    """The profile is version 1 and built on tracking's version 3; the thinning has the stream after the hazard's, and
    the streams before it keep their numbers (the profiles nest: no earlier draw moves)."""
    assert degrade.OFFPITCH == "offpitch" and degrade.PROFILES["offpitch"] == 1
    assert degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 3, "context": 1, "offpitch": 1}
    assert degrade.COMPONENTS == {"transplant": 0, "tracklets": 1, "wrong_team": 2, "speed_nan": 3, "hazard": 4,
                                  "thinning": 5}
    assert degrade.COMPONENTS["thinning"] == THINNING
    assert degrade.OFFPITCH_TRACKING_VERSION == degrade.PROFILES["tracking"] == 3, (
        "offpitch reuses tracking's players: when tracking's version changes, bump PROFILES['offpitch'] and "
        "OFFPITCH_TRACKING_VERSION")
    assert degrade.profile_record("offpitch") == {"name": "offpitch", "version": 1, "tracking_version": 3}
    view = degrade.tracking_view(offpitch_match())
    for half in (1, 2):
        drawn = degrade.thinning_uniforms(view, half, 4, 7)
        assert drawn.tolist() == np.random.default_rng([4, view.match_index, THINNING, half]).random(7).tolist()
        assert not drawn.flags.writeable
    assert degrade.thinning_uniforms(view, 1, 4, 0).tolist() == []
    assert benchmark_run.PROFILES["offpitch"] == 1 and benchmark_run.DEFAULT_PROFILE == "tracking"


def test_offpitch_refuses_until_fitted():
    """While OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB or OFFPITCH_FIT is None the profile is not made: the error names the
    fit (offpitch.py fit) and R015. Set here to None, and put back, so that it holds whether they are pinned or not.
    The other refusals name the argument: no Calibration, a margin or a drop probability out of range, a mask that is
    not a 1-D bool array, uniforms that are not one per interior gap, a ball that is not (n, 2) or a pitch that is not
    two positive lengths."""
    view = degrade.tracking_view(offpitch_match())
    cal = made_up_calibration()
    with fitted(None, None, None):
        expect_error(ValueError, lambda: degrade.degrade(view, "offpitch", cal, 0), "offpitch", "offpitch.py fit",
                     "R015")
    for unfitted in ((None, 0.1, MADE_UP_FIT), (1.0, None, MADE_UP_FIT), (1.0, 0.1, None)):
        with fitted(*unfitted):
            expect_error(ValueError, lambda: degrade.degrade(view, "offpitch", cal, 0), "offpitch.py fit", "R015")
    with fitted(1.0, 0.1):
        degrade.degrade(view, "offpitch", cal, 0)
        expect_error(ValueError, lambda: degrade.degrade(view, "offpitch", None, 0), "offpitch", "Calibration",
                     "calibration.load()")
    with fitted(-1.0, 0.1):
        expect_error(ValueError, lambda: degrade.degrade(view, "offpitch", cal, 0), "margin_m", "-1.0")
    with fitted(1.0, 1.5):
        expect_error(ValueError, lambda: degrade.degrade(view, "offpitch", cal, 0), "drop_prob", "1.5")
    expect_error(ValueError, lambda: degrade.offpitch_mask(np.zeros(3), math.nan), "offpitch_mask", "margin_m")
    expect_error(ValueError, lambda: degrade.offpitch_mask(np.zeros((3, 2)), 1.0), "offpitch_mask", "line_m")
    expect_error(ValueError, lambda: degrade.thin_mask(np.zeros(3), np.zeros(0), 0.5), "thin_mask", "mask", "bool")
    expect_error(ValueError, lambda: degrade.thin_mask(np.zeros(3, dtype=bool), np.zeros(0), -0.1), "thin_mask",
                 "drop_prob")
    expect_error(ValueError, lambda: degrade.line_distance_m(np.zeros(4), (105.0, 68.0)), "line_distance_m",
                 "ball_xy")
    expect_error(ValueError, lambda: degrade.line_distance_m(np.zeros((4, 2)), (105.0, 0.0)), "line_distance_m",
                 "pitch")
    expect_error(ValueError, lambda: degrade.offpitch_masks(view, cal, 0, 1.0, True), "offpitch_masks", "drop_prob")
    expect_error(ValueError, lambda: degrade.thinning_uniforms(view, 1, 0, -1), "n_gaps")
