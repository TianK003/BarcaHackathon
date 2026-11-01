"""The broadcast's ball in our ball's gaps (R028): align the TV broadcast to our frames, keep its ball where it can be
trusted, and write it where ours is missing.

Why. Our ball is missing on 41% of the graded frames (docs/results.md A01), and a perfect ball in those gaps would be
worth a lot on the benchmark (A14). The broadcast tracks the same match from a camera that a person steers: a second,
independent tracker. On the live frames where ours is lost, A20 found the broadcast's ball on 28% of them. This module
holds the steps from the broadcast's tracking to our filled one, as plain functions on arrays. Inputs
(pipeline/inputs/__init__.py) loads the files and calls them in this order:

    align      the offset of each graded half, found by code on every run          (the design's 2.3)
    frame_map  our graded frame -> the broadcast frame that shows it
    live       the mapped frames whose broadcast shows the play we see             (2.4)
    usable     the broadcast's ball on live frames, in runs of 13 frames or more   (2.5)
    fill       our tracking, with that ball where ours is missing on a graded frame (2.6)

The design is docs/superpowers/specs/2026-11-01-r028-broadcast-ball-design.md, and every constant below is one of A20's
measurements or a round value (its section 5): none comes from the public window, and none was chosen by a score.

The alignment. The broadcast leaves out half time and starts at a moment of its own, so each half has its own offset
d: our frame t is the broadcast's frame t - d (A20: d = 4,014 and 26,919, steady to a frame). It is found ball
against ball, in metres, on the frames where both balls are present and the broadcast's homography is valid (a pair).
- Coarse: every 25th graded frame of the half, against every offset in steps of 25 frames. The score is the share of
  pairs within 3 m, and an offset counts only on 300 pairs or more.
- Fine: every graded frame of the half, against every offset within 40 frames of the coarse best. The score is the
  share within 1.5 m, again on 300 pairs or more.
- Acceptance: the fine score is at least 0.5, and the coarse best is at least twice the best coarse score more than
  250 frames (10 s) from it. A match whose ball moves would not agree at some other offset by chance, and a broadcast
  of something else agrees nowhere.
Otherwise AlignmentError stops the run, naming the half and the condition: a misaligned feed must never be filled from
silently. Ties go to the smaller |d|, then to the smaller d. Over BAR-ATH's two halves the search takes about 4 s.

The live test. A broadcast frame can be a replay, a close-up or the crowd, and its homography is sometimes valid even
then. A mapped frame is live when the homography is valid, the broadcast shows at least 4 players, and the median,
over those players, of the distance to our nearest player is below 2.0 m. Teams do not enter: a slot's team is the
slot's, not the player's (docs/traps.md T32).

The runs. Where both balls exist, A20 part 6 found the broadcast's runs of 1 to 12 usable frames a median 16 to 27 m
from ours, and its runs of 13 to 49 frames 1.6 m. So a usable ball (present on a live frame) is kept only in runs of
at least 13 frames (0.52 s). A run is the broadcast ball's own: it carries on across the edges of our gaps, and it
breaks wherever the broadcast's frames stop following each other.

The fill. Where a graded frame's ball is missing and a usable broadcast ball exists, ball_xy becomes the broadcast's
position, normalised as ours is: both ATDs use the same pitch, orientation and team convention (A20). ball_speed stays
NaN, which no source reads (T31), and the players stay ours.

Every distance here is in metres: the callers scale by the pitch first (Inputs.metres, T16). No randomness anywhere, so
the same arrays give the same mask and the same bytes (the design's 2.7).
"""

from dataclasses import dataclass

import numpy as np

from pipeline.inputs import atd_cache
from pipeline.inputs.atd_cache import Atd

# Params
COARSE_STEP = 25               # frames between the coarse search's samples, and between its offsets (1 s)
COARSE_TOL_M = 3.0             # a coarse pair agrees within this distance
MIN_PAIRS = 300                # an offset counts only on at least this many pairs, in either search
FINE_HALF_WIDTH = 40           # the fine search tries every offset within this many frames of the coarse best
FINE_TOL_M = 1.5               # a fine pair agrees within this distance
MIN_FINE_SCORE = 0.5           # the fine score an accepted offset needs
PEAK_RATIO = 2.0               # the coarse best must be this many times the best score far from it ...
PEAK_EXCLUSION_FRAMES = 250    # ... where far is more than this many frames away (10 s)
LIVE_MAX_MEDIAN_M = 2.0        # a live frame's median distance from a broadcast player to our nearest is below this
LIVE_MIN_PLAYERS = 4           # a live frame shows at least this many broadcast players
MIN_RUN = 13                   # usable runs shorter than this many frames are dropped (A20 part 6)
CHUNK = 1024                   # frames per step of the live test: bounds its (frames, slots, slots) temporaries


class AlignmentError(ValueError):
    """The broadcast cannot be aligned to our frames. The message names the half and the condition that failed."""


@dataclass(frozen=True)
class Alignment:
    """The offset that aligns the broadcast to each graded half, and the scores it was accepted on.

    Our frame t of half h is the broadcast's frame t - offsets[h]. Every field is a dict {half: value} of aggregates.
    """
    offsets: dict        # {half: int}
    fine_scores: dict    # {half: float}: the share of the fine pairs within FINE_TOL_M, at the offset
    coarse_scores: dict  # {half: float}: the coarse best's share of pairs within COARSE_TOL_M
    runner_up: dict      # {half: float}: the best coarse score more than PEAK_EXCLUSION_FRAMES from the coarse best,
                         # 0.0 if no offset there counts
    pairs: dict          # {half: int}: the fine search's pairs at the offset


def align(tactical_ball_m, broadcast_ball_m, broadcast_valid, graded):
    """The Alignment of the broadcast to our frames, half by half (the module docstring), or AlignmentError.

    tactical_ball_m is our ball, (n, 2), and broadcast_ball_m the broadcast's, (nb, 2), both in metres and NaN where
    absent; broadcast_valid, (nb,), is the broadcast homography's valid column; graded is Inputs.graded, the
    [(half, start, stop)] blocks (half-open). A broadcast ball on a frame whose homography is not valid is never paired.
    """
    ours = _ball(tactical_ball_m, "tactical_ball_m")
    theirs = _ball(broadcast_ball_m, "broadcast_ball_m")
    valid = np.asarray(broadcast_valid, dtype=bool)
    if valid.shape != (len(theirs),):
        raise ValueError("align: broadcast_valid has shape {} for {} broadcast frames; pass the broadcast homography's "
                         "valid column, one value per broadcast frame".format(valid.shape, len(theirs)))
    theirs = np.where(valid[:, None], theirs, np.nan)
    fields = {"offsets": {}, "fine_scores": {}, "coarse_scores": {}, "runner_up": {}, "pairs": {}}
    for half, start, stop in graded:
        coarse, coarse_score, runner, runner_score = _coarse(ours, theirs, half, start, stop)
        offset, fine_score, pairs = _fine(ours, theirs, half, start, stop, coarse)
        problems = []
        if fine_score < MIN_FINE_SCORE:
            problems.append("its fine score is {:.3f} (the share of {:,} ball pairs within {} m), below {}".format(
                fine_score, pairs, FINE_TOL_M, MIN_FINE_SCORE))
        if coarse_score < PEAK_RATIO * runner_score:
            problems.append("its coarse score {:.3f} is not {} times the best one more than {} frames away ({:.3f} at "
                            "{:+,})".format(coarse_score, PEAK_RATIO, PEAK_EXCLUSION_FRAMES, runner_score, runner))
        if problems:
            raise AlignmentError(_refusal(half, "at the best offset, {:+,} frames, {}".format(
                offset, "; and ".join(problems))))
        for name, value in (("offsets", offset), ("fine_scores", fine_score), ("coarse_scores", coarse_score),
                            ("runner_up", runner_score), ("pairs", pairs)):
            fields[name][half] = value
    return Alignment(**fields)


def frame_map(alignment, graded, n, nb):
    """(n,) int64: the broadcast frame that shows each of our graded frames, t - offsets[half]; -1 on every frame that
    is not graded, and where that broadcast frame does not exist (before its first frame or after its last)."""
    fmap = np.full(n, -1, dtype=np.int64)
    for half, start, stop in graded:
        if half not in alignment.offsets:
            raise ValueError("frame_map: the alignment has no offset for half {}; align every half of `graded` (it has "
                             "{})".format(half, sorted(alignment.offsets)))
        frames = np.arange(start, stop, dtype=np.int64)
        seen = frames - alignment.offsets[half]
        inside = (seen >= 0) & (seen < nb)
        fmap[frames[inside]] = seen[inside]
    return fmap


def live(tactical_players_m, broadcast_players_m, broadcast_valid, fmap):
    """(n,) bool: the mapped frames whose broadcast shows the play we see (the module docstring), in metres.

    tactical_players_m is our players, (n, S, 2), broadcast_players_m the broadcast's, (nb, Sb, 2), both NaN where a
    slot is absent; broadcast_valid, (nb,), the broadcast homography's valid column; fmap, (n,), from frame_map.
    """
    ours = np.asarray(tactical_players_m, dtype=np.float64)
    theirs = np.asarray(broadcast_players_m, dtype=np.float64)
    valid = np.asarray(broadcast_valid, dtype=bool)
    fmap = _fmap(fmap, len(theirs), "live")
    if ours.ndim != 3 or ours.shape[0] != len(fmap) or ours.shape[2] != 2:
        raise ValueError("live: tactical_players_m has shape {}; it must be (n, slots, 2) with n = {}, the frames of "
                         "fmap".format(ours.shape, len(fmap)))
    if theirs.ndim != 3 or theirs.shape[2] != 2 or valid.shape != (len(theirs),):
        raise ValueError("live: broadcast_players_m has shape {} and broadcast_valid {}; they must be (nb, slots, 2) "
                         "and (nb,), one row per broadcast frame".format(theirs.shape, valid.shape))
    result = np.zeros(len(fmap), dtype=bool)
    rows = np.flatnonzero(fmap >= 0)
    rows = rows[valid[fmap[rows]]]
    rows = rows[(~np.isnan(theirs[fmap[rows], :, 0])).sum(axis=1) >= LIVE_MIN_PLAYERS]
    for low in range(0, len(rows), CHUNK):
        chunk = rows[low:low + CHUNK]
        seen, mine = theirs[fmap[chunk]], ours[chunk]                      # (c, Sb, 2), (c, S, 2)
        squared = np.square(seen[:, :, None, 0] - mine[:, None, :, 0])
        squared += np.square(seen[:, :, None, 1] - mine[:, None, :, 1])    # (c, Sb, S), NaN where either is absent
        squared[np.isnan(squared)] = np.inf
        nearest = np.sqrt(squared.min(axis=2))                             # inf where none of ours is tracked
        present = ~np.isnan(seen[:, :, 0])
        nearest[~present] = np.inf                                         # absent slots sort after every present one
        nearest.sort(axis=1)
        count, index = present.sum(axis=1), np.arange(len(chunk))
        median = (nearest[index, (count - 1) // 2] + nearest[index, count // 2]) / 2     # np.median's, over the present
        result[chunk] = median < LIVE_MAX_MEDIAN_M
    return result


def usable(broadcast_ball_m, fmap, live_mask, min_run=MIN_RUN):
    """(n,) bool: our frames with a usable broadcast ball -- mapped, live, the broadcast's ball present -- in runs of at
    least min_run frames; a run breaks where the broadcast's frames stop following each other.

    broadcast_ball_m is the broadcast's ball, (nb, 2), NaN where absent (metres or not: only its presence is read);
    fmap, (n,), from frame_map; live_mask, (n,), from live.
    """
    if isinstance(min_run, bool) or not isinstance(min_run, (int, np.integer)) or min_run < 1:
        raise ValueError("usable: min_run must be a whole number of frames, 1 or more (the default is {}); got "
                         "{!r}".format(MIN_RUN, min_run))
    theirs = _ball(broadcast_ball_m, "broadcast_ball_m")
    fmap = _fmap(fmap, len(theirs), "usable")
    live_mask = np.asarray(live_mask, dtype=bool)
    if live_mask.shape != fmap.shape:
        raise ValueError("usable: live_mask has shape {} and fmap {}; both are one value per frame of ours".format(
            live_mask.shape, fmap.shape))
    keep = np.zeros(len(fmap), dtype=bool)
    mapped = np.flatnonzero(fmap >= 0)
    keep[mapped] = np.isfinite(theirs[fmap[mapped]]).all(axis=1)
    keep &= live_mask
    starts = keep.copy()
    starts[1:] &= ~keep[:-1] | (fmap[1:] != fmap[:-1] + 1)
    run = np.cumsum(starts) - 1                                            # on a kept frame, the number of its run
    lengths = np.bincount(run[keep], minlength=int(starts.sum()))
    keep[keep] = lengths[run[keep]] >= min_run
    return keep


def fill(atd, ball_norm, where):
    """(filled Atd, filled): `atd` with ball_xy taken from ball_norm on the frames where `where` holds, the tracking's
    ball is missing and ball_norm is finite; filled, (n,) bool and read-only, is True exactly on those frames.

    ball_norm is (n, 2), normalised as ball_xy is, and `where` (n,) bool (Inputs.graded_mask). A present ball is never
    replaced, ball_speed stays as it is (NaN where the ball was missing), and the players, their speeds and slot_team
    are the tracking's own arrays, not copies. The new Atd is read-only, like every Atd; `atd` itself is not changed,
    and its arrays keep the flags they had.
    """
    ball_norm = np.asarray(ball_norm, dtype=np.float64)
    where = np.asarray(where, dtype=bool)
    n = atd.n_frames
    if ball_norm.shape != (n, 2):
        raise ValueError("fill: ball_norm has shape {}, and the tracking has {} frames; it must be ({}, 2), "
                         "normalised as ball_xy is".format(ball_norm.shape, n, n))
    if where.shape != (n,):
        raise ValueError("fill: where has shape {}, and the tracking has {} frames; it must be ({},), True on the "
                         "frames that may be filled (the graded ones)".format(where.shape, n, n))
    filled = where & ~atd.ball_present & np.isfinite(ball_norm).all(axis=1)
    ball_xy = np.array(atd.ball_xy, dtype=np.float64)
    ball_xy[filled] = ball_norm[filled]
    filled.flags.writeable = False
    # Views of the arrays that do not change: _freeze makes the views read-only and leaves the tracking given as it was.
    return atd_cache._freeze(Atd(players_xy=atd.players_xy.view(), players_speed=atd.players_speed.view(),
                                 ball_xy=ball_xy, ball_speed=atd.ball_speed.view(),
                                 slot_team=atd.slot_team.view())), filled


def _coarse(ours, theirs, half, start, stop):
    """(offset, score, runner-up offset, runner-up score) of the coarse search over one half (runner-up None, 0.0 when
    no offset more than PEAK_EXCLUSION_FRAMES away counts).

    Our samples are frames start, start + 25, ... and the offsets multiples of 25, so a sample only ever meets the
    broadcast frames of its phase, start % 25: the search is a correlation of two arrays taken every 25th frame, sample
    i against broadcast frame phase + 25 j at offset 25 (first + i - j).
    """
    samples = ours[start:stop:COARSE_STEP]
    phase, first = start % COARSE_STEP, start // COARSE_STEP
    grid = theirs[phase::COARSE_STEP]
    steps = np.arange(first - len(grid) + 1, first + len(samples))        # every offset / 25 that pairs anything
    within, pairs = np.zeros(len(steps), dtype=np.int64), np.zeros(len(steps), dtype=np.int64)
    for index, step in enumerate(steps):
        low, high = max(0, step - first), min(len(samples), len(grid) + step - first)
        squared = _squared(samples[low:high], grid[first + low - step:first + high - step])
        pairs[index] = np.count_nonzero(~np.isnan(squared))
        within[index] = np.count_nonzero(squared <= COARSE_TOL_M ** 2)
    offsets = steps * COARSE_STEP
    counted = pairs >= MIN_PAIRS
    if not counted.any():
        raise AlignmentError(_refusal(half, "no offset has {} coarse pairs (every {}th graded frame where both balls "
                                            "are present and the broadcast's homography is valid); the most at any "
                                            "offset is {:,}".format(MIN_PAIRS, COARSE_STEP,
                                                                    int(pairs.max()) if len(pairs) else 0)))
    scores = within / np.maximum(pairs, 1)
    best = _best(offsets, scores, counted)
    far = counted & (np.abs(offsets - offsets[best]) > PEAK_EXCLUSION_FRAMES)
    if not far.any():
        return int(offsets[best]), float(scores[best]), None, 0.0
    runner = _best(offsets, scores, far)
    return int(offsets[best]), float(scores[best]), int(offsets[runner]), float(scores[runner])


def _fine(ours, theirs, half, start, stop, centre):
    """(offset, score, pairs) of the fine search: every offset within FINE_HALF_WIDTH of centre, every graded frame."""
    offsets = np.arange(centre - FINE_HALF_WIDTH, centre + FINE_HALF_WIDTH + 1, dtype=np.int64)
    within, pairs = np.zeros(len(offsets), dtype=np.int64), np.zeros(len(offsets), dtype=np.int64)
    for index, offset in enumerate(offsets):
        low, high = max(start, offset), min(stop, len(theirs) + offset)    # our frames whose t - offset exists
        if low < high:
            squared = _squared(ours[low:high], theirs[low - offset:high - offset])
            pairs[index] = np.count_nonzero(~np.isnan(squared))
            within[index] = np.count_nonzero(squared <= FINE_TOL_M ** 2)
    counted = pairs >= MIN_PAIRS
    if not counted.any():
        raise AlignmentError(_refusal(half, "no offset within {} frames of the coarse best ({:+,}) has {} fine pairs; "
                                            "the most is {:,}".format(FINE_HALF_WIDTH, centre, MIN_PAIRS,
                                                                      int(pairs.max()))))
    scores = within / np.maximum(pairs, 1)
    best = _best(offsets, scores, counted)
    return int(offsets[best]), float(scores[best]), int(pairs[best])


def _best(offsets, scores, among):
    """The index of the highest score where `among` holds; a tie goes to the smaller |offset|, then the smaller one."""
    candidates = np.flatnonzero(among)
    order = np.lexsort((offsets[candidates], np.abs(offsets[candidates]), -scores[candidates]))
    return candidates[order[0]]


def _squared(a, b):
    """Squared distances between two (k, 2) arrays of points, row by row; NaN where either point is absent."""
    difference = a - b
    return difference[:, 0] * difference[:, 0] + difference[:, 1] * difference[:, 1]


def _refusal(half, why):
    return ("half {}: the broadcast cannot be aligned to our frames: {}. A misaligned broadcast is never filled from: "
            "check that data/broadcast/ holds this match's broadcast files, or run without the fill ([inputs] "
            "ball_fill = \"none\")".format(half, why))


def _ball(array, name):
    """`array` as (k, 2) float64, or a ValueError that names it."""
    ball = np.asarray(array, dtype=np.float64)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("{} has shape {}; it must be (frames, 2), one ball position per frame, NaN where there is "
                         "none".format(name, ball.shape))
    return ball


def _fmap(fmap, nb, caller):
    """fmap as a 1-D integer array whose every value is -1 or a broadcast frame, or a ValueError naming the caller."""
    fmap = np.asarray(fmap)
    if fmap.ndim != 1 or not np.issubdtype(fmap.dtype, np.integer):
        raise ValueError("{}: fmap has shape {} and dtype {}; it must be frame_map's (n,) integers".format(
            caller, fmap.shape, fmap.dtype))
    if len(fmap) and (fmap.min() < -1 or fmap.max() >= nb):
        raise ValueError("{}: fmap holds frames outside -1 .. {} (the broadcast's last frame); it must be frame_map's, "
                         "made for these broadcast arrays".format(caller, nb - 1))
    return fmap
