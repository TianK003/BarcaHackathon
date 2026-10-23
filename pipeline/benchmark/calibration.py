"""Our own tracking, measured: what the benchmark's degradation has to reproduce (docs/results.md, A07, A08).

    ball = ball_stats(ball_present, blocks, ball_xy_m, pitch, fps)          # gaps, where they start, per 5 minutes
    players = player_stats(present, slot_team, xy_m, speed, blocks, fps)    # slots present, tracklets, wrong team, ...
    fit = fit_hazard(ball_present, ball_xy_m, players_present, players_xy_m, spans, pitch, fps, *ball[...])   # A08
    cal = load()                                                            # BAR-ATH, measured once and cached
    describe(cal)                                                           # scalars and quantiles, for reports

Why it exists. P01 validates pipeline choices on other matches, degraded until their tracking looks like ours: the ball
missing in long gaps, 46 slots for 22 players, slots that carry the wrong team. "Looks like ours" needs a ruler, and
the ruler has to be one set of functions: ball_stats and player_stats measure BAR-ATH here, and fidelity.py applies the
same two functions, unchanged, to a degraded benchmark match, so that a difference between the two numbers is a
difference between the matches and not between two ways of counting. They are pure, take plain arrays and know nothing
about either match.

Definitions (metres throughout: normalised coordinates times the pitch, 105.3 x 68.0; Inputs.metres)
    block           (half, start, stop), half-open: the stretch of one half that a statistic is taken on (see Spans).
    usable          a ball or a slot counts as present only if its flag says so and BOTH coordinates are finite. A
                    position with one coordinate missing is absent, and is counted (dropped_frames, dropped_slot_frames)
                    so that it is visible: the line distance and the jitter use y as well as x. A flag that says absent
                    over a finite position is a ValueError: the two come from different trackings.
    gap             a maximal run of frames without a ball inside a block. A gap that touches the block's first or last
                    frame is a gap (its duration counts, and so does its share of missing frames) but is not interior.
    interior gap    a gap with a visible frame on both sides; the frame before it is its onset. Only these have an
                    onset, a minute and a line distance, because only they can be asked where the tracker lost the ball.
                    interior_durations_s, gap_onset_minute and gap_onset_near_line are PARALLEL arrays, one entry per
                    interior gap in the same order, so that the duration of a gap can be read with its class and
                    minute; gap_durations_s holds every gap, interior or not.
    line distance   min(x, L - x, y, W - y) of the ball, negative beyond a line.
    near-line onset an interior gap whose onset frame has a line distance below 2 m (strictly: 2.0 is not near).
    onset minute    the onset frame's time in minutes from the start of its block. The spans that measure() passes start
                    at the kick-off, so it counts from the kick-off, as hazard.features does, and none is negative.
    5-minute share  the share of missing ball frames in each stretch of 5 minutes, counted from the block's first frame
                    (the last stretch is whatever is left): A02's table.
    tracklet        a maximal run of one slot being present inside a block (its length is in seconds).
    wrong-team frame
                    a frame on which a team shows more than 11 slots and the other fewer than 11: a player is being
                    carried by a slot of the other team (docs/traps.md T32). An episode is a maximal run of such
                    frames with the same team in surplus.
    speed-NaN share NaN speeds over present slot-frames; a run is a maximal run of one slot present with a NaN speed.
    jitter          the RMS of the norm of p[t+1] - 2 p[t] + p[t-1] over present triples: the second difference of a
                    slot's position in metres, one frame apart, which a tracker's smoothing makes small (3 mm here).

Spans. A benchmark match starts at its kick-off, so everything that drives the degradation and centres the fidelity
gates is measured on the same stretch of ours: each half from its KICK-OFF to the end of its graded block (measure()
passes those spans as the blocks of ball_stats and player_stats; ball_masks and present_counts run over them too). Our
second half is graded from 2,934 frames (117 s) before its kick-off, all of them D: short tracklets and NaN-speed runs
that no benchmark half has, which distort the graded-block figures (A07 gives both). Three things are not on the spans:
missing_share_5min is A02's published table, counted from each half's first graded frame; baseline_outputs are over the
graded halves, as the published baselines are, so that they can be compared with them; and the published figures
themselves (A01, A02, R001, the plan) are graded-block figures, which test_data_calibration_replicates_a01 computes by
calling these same two functions on the graded blocks and asserts, without storing them.

The hazard (A08, version 2). The benchmark places its ball gaps where our tracker loses the ball, so the Calibration
also holds a hazard fitted on BAR-ATH's own rows. fit_hazard takes the ball-visible frames of the spans as the frames at
risk (hazard_rows), the event "the next frame starts an interior gap", and the six features of hazard.py computed on our
own ball and players. The model is fitted on both halves pooled (hazard_model, with l2 = 1 and, if a fit does not
converge, ten times that, at most three times: hazard_l2 is the penalty that was used); on one half and scored on the
other, both ways, for the cross-half AUCs of the full model and of a model of the minute alone (hazard_auc); and the
decision for the `context` profile is that both full AUCs are at least 0.60 (hazard_go; the plan's rule, taken as it
comes). c (hazard_c) is the onsets per at-risk frame over the model's mean probability on the same rows, which is 1 on
BAR-ATH's own rows by construction: the intercept is not penalised, so the model's mean probability is the rate. It is
stored for completeness and carries nothing to another match; the rate transfer to the benchmark is the `context`
profile's own one-time, label-free rescale (the plan's Task 17). The interior gaps' durations are kept in seconds by
onset class (near a line or not) and third of the half (hazard_durations_s); a cell with fewer than 5 uses its class's
durations over all thirds, and hazard_fallback_cells lists those that did. Visible frames that are not at risk are
counted, never dropped silently: the frame before a gap that runs to the span's end (neither onset nor non-onset), the
last frame of a span, and a frame with a feature the tracking cannot give (the scalars hazard_edge, hazard_last_frame,
hazard_nan_rows and hazard_nan_events, beside hazard_rows, hazard_events, ...).

What a Calibration holds is frame-level data of BAR-ATH (the ball's mask and the slot counts of every frame), which must
never reach a log, a report, a CSV or git. Every array field has repr=False, describe() returns scalars and short
quantile lists only, and the cache sits under <data dir>/cache/benchmark/calibration/, which git ignores. The baselines
(baseline_outputs) go through run.predict and pipeline.output.summarise: no CSV of labels of this match is written.

The cache. load() keeps a Calibration as .npy arrays and a manifest.json whose key holds everything the numbers depend
on: the ATD cache's record of its source files and its cache_version (how they were parsed), CALIBRATION_VERSION, the
sha256 of this very file (CALIBRATION_SHA256, taken once as the module is imported, with CRLF read as LF, as dfl.py does
for its parser) and of hazard.py, which makes the hazard fields (HAZARD_SHA256, the same way), the numpy version, the
sha256 of the four yardstick configs, the public truth, the pitch and the frame rate (games.py's constants, which every
distance and duration is measured with), and the graded blocks and kick-offs. A cache is reused only if the whole key
matches, so an edited config or an edited definition cannot reuse old numbers.
Each key has a folder of its own, <data dir>/cache/benchmark/calibration/<digest>/ (cache_folder), where the digest is
the first 16 hex digits of the sha256 of the key's canonical JSON: checkouts that run different code, and so have
different keys, can share one data dir without rebuilding each other's cache in turn or interleaving their writes. A
folder of an old key is never removed (a few MB each); the key is what says whether one is still wanted. An unreadable
cache (a manifest that is not JSON or lacks a field, an array missing or cut short) is absent, never an error:
everything in it can be measured again from the sources. The arrays are written first and the manifest last, with the
old manifest removed up front, as pipeline/inputs/atd_cache.py and dfl.load_match do, so a run killed half-way leaves no
manifest. The manifest is written to a temp file of its own in the folder and moved into place (os.replace), so that no
reader sees half of it, and an old manifest that cannot be removed (Windows refuses to remove a file that another
process has open) counts as absent: two checkouts with the same code share a key and so a folder, and one must not
fail because the other is reading or writing it. Windows refuses the move over a manifest that is open as well, so the
move is tried again, REPLACE_RETRIES times and REPLACE_WAIT_S apart, which outlasts a reader that holds the file for a
moment. A hold that lasts past the retries fails the build with the OS's own error (PermissionError), and a rerun
recovers: nothing is left half-written, and the next load() reads or measures the cache again.
"""

import dataclasses
import hashlib
import json
import math
import operator
import os
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np

from pipeline.benchmark import hazard
from pipeline.contracts import D

# Params
# A manual switch, like dfl.CACHE_VERSION: bump it to invalidate every cache at once. Any edit to this file or to
# hazard.py does that already (CALIBRATION_SHA256 and HAZARD_SHA256, in the key), and so does a new ATD parse (the ATD
# cache's cache_version, in the key). What still needs a bump is a change to the code that the yardsticks run through,
# which the key hashes the configs of but not the code of: pipeline/inputs/ (other than the parse), pipeline/evidence/,
# pipeline/decode/, pipeline/refine/, pipeline/output/ (summarise) and pipeline/benchmark/run.py, for these decide
# baseline_outputs and public_window.
# 1: the measurements (A07). 2: and the hazard fit, the durations by class and third, and c (A08).
CALIBRATION_VERSION = 2
CACHE_FOLDER = ("cache", "benchmark", "calibration")   # <data dir>/cache/benchmark/calibration/<digest of the key>/
KEY_DIGITS = 16                                        # hex digits of the key's sha256 that name its folder
REPLACE_RETRIES = 5     # a manifest move that the OS refuses (PermissionError: a reader has the target open, on
                        # Windows)
REPLACE_WAIT_S = 0.05   # is tried again this many times, this long apart: 250 ms in all, a reader holds a file for less
MANIFEST = "manifest.json"
CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
YARDSTICK_WIDTHS = (0, 25, 50, 125)     # half-widths of nearest_w{width}.toml: the organisers' four baselines, per half
NEAR_LINE_M = 2.0                       # an onset closer than this to a line is a near-line onset (A01)
SEGMENT_MINUTES = 5                     # A02's blocks
TEAM_SIZE = 11                          # more than this many slots on one team, with fewer on the other: wrong team
FULL_COUNT = 2 * TEAM_SIZE              # slots present on a frame with everybody tracked
LOW_COUNT = 14                          # the benchmark's gate on frames with at most this many present (plan, Fidelity)
DECIMALS = 9                            # summary floats are rounded to this, because BLAS summation order varies
ONSET_BIN_MINUTES = 10                  # describe() counts onsets in bins of this many minutes
MAX_BINS = 20                           # and no list of describe() is longer than this
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99))
HIST_QUANTILES = (("p5", 0.05), ("p25", 0.25), ("median", 0.50))
# dtype of every array the cache keeps, by field; the arrays of a mapping are kept one file per half, name_half.npy.
FLAT_ARRAYS = {"gap_durations_s": np.float64, "interior_durations_s": np.float64, "gap_onset_minute": np.float64,
               "gap_onset_near_line": np.bool_, "tracklet_lengths_s": np.float64, "wrong_team_episodes_s": np.float64,
               "present_hist": np.int64, "speed_nan_runs": np.int64}
HALF_ARRAYS = {"ball_masks": np.bool_, "present_counts": np.int16, "missing_share_5min": np.float64}
# The three statistics that have a field of their own in the Calibration (wrong_team_share, speed_nan_share and
# player_jitter_rms_m); scalars holds the others.
OWN_FIELDS = ("wrong_team_share", "speed_nan_share", "jitter_rms_m")
# The source whose agreement with the public truth is A01's 77.8% (R001's method), looked up by name in every yardstick.
NEAREST_SOURCE = "nearest_player"
# The hazard fit (A08).
HAZARD_L2 = 1.0                 # the penalty of every fit; a fit that does not converge is refitted with it times 10...
HAZARD_L2_FACTOR = 10.0
HAZARD_REFITS = 3               # ...at most this many times, and the penalty that was used is kept
HAZARD_GO_AUC = 0.60            # go: both full-model cross-half AUCs at least this (the plan's rule, for `context`)
HAZARD_AUC_KEYS = ("full_1to2", "full_2to1", "time_1to2", "time_2to1")   # fitted on half a, scored on half b
CLASSES = (0, 1)                # onset class of a gap: 1 if its onset is near a line (< NEAR_LINE_M), else 0
THIRDS = (0, 1, 2)              # third of the half the onset falls in (hazard.third_of_minute)
MIN_CELL_DURATIONS = 5          # fixed in advance: a (class, third) cell with fewer durations uses its class's pool


# -- the statistics ---------------------------------------------------------------------------------------------------

def ball_stats(ball_present, blocks, ball_xy_m, pitch, fps):
    """The ball's gaps inside the blocks, as a dict; arrays are in time order, block by block.

    ball_present  (n,) bool: the tracker sees the ball.
    blocks        [(half, start, stop)], half-open, disjoint, one per half: the stretches to measure (measure() gives
                  the spans from each kick-off to the end of the graded block; the graded blocks reproduce A01).
    ball_xy_m     (n, 2) metres from a pitch corner, NaN where the ball is missing.
    pitch         (length, width) in metres; fps frames a second.

    The ball is present on a frame only if ball_present is True and both coordinates are finite; a frame that fails
    the second test is absent and counted (dropped_frames).

    Scalars: frames (in the blocks), missing_frames, missing_share, dropped_frames, gaps, interior_gaps,
    near_line_onsets and near_line_share (of the interior gaps; NaN when there are none). Arrays: gap_durations_s (one
    per gap, in seconds), then one per interior gap, in the same order: interior_durations_s, gap_onset_minute (from the
    block's first frame), gap_onset_line_m (the line distance at the onset frame) and gap_onset_near_line (bool). And
    missing_share_5min, {half: float64 array} of the missing share of each 5-minute stretch of the block.
    ValueError, naming the argument, for anything that would make a number meaningless.
    """
    flag = _bool_array("ball_present", ball_present, 1)
    n = len(flag)
    xy = np.asarray(ball_xy_m, dtype=np.float64)
    if xy.shape != (n, 2):
        raise ValueError("ball_present has {} frames but ball_xy_m has shape {}; both are one value per frame of the "
                         "match, the position in metres".format(n, xy.shape))
    length, width = _pitch(pitch)
    fps = _fps(fps)
    blocks = _blocks(blocks, n)
    finite = np.isfinite(xy).all(axis=1)
    stray = int((~flag & finite).sum())
    if stray:
        raise ValueError("ball_present and ball_xy_m disagree on {} frames: the ball is marked absent where a position "
                         "is given; build both from the same tracking".format(stray))
    present = flag & finite

    durations, interior_durations, minutes, lines = [], [], [], []
    graded = missing = dropped = 0
    for _, start, stop in blocks:
        gone = ~present[start:stop]
        graded += stop - start
        missing += int(gone.sum())
        dropped += int((flag[start:stop] & ~finite[start:stop]).sum())
        first, last = _runs(gone)
        block_durations = (last - first) / fps
        durations.append(block_durations)
        interior = (first > 0) & (last < stop - start)
        onset = first[interior] - 1                      # the last visible frame before each interior gap
        interior_durations.append(block_durations[interior])
        x, y = xy[start + onset, 0], xy[start + onset, 1]
        lines.append(np.minimum.reduce([x, length - x, y, width - y]))
        minutes.append(onset / fps / 60.0)
    durations, interior_durations, minutes, lines = (
        np.concatenate(part) for part in (durations, interior_durations, minutes, lines))
    near = lines < NEAR_LINE_M
    return {
        "frames": graded, "missing_frames": missing, "missing_share": missing / graded, "dropped_frames": dropped,
        "gaps": int(durations.size), "interior_gaps": int(minutes.size), "near_line_onsets": int(near.sum()),
        "near_line_share": float(near.sum()) / minutes.size if minutes.size else math.nan,
        "gap_durations_s": durations, "interior_durations_s": interior_durations, "gap_onset_minute": minutes,
        "gap_onset_line_m": lines, "gap_onset_near_line": near,
        "missing_share_5min": segment_shares(present, blocks, fps),
    }


def segment_shares(ball_visible, blocks, fps):
    """{half: float64 array}: the share of frames without a ball in each 5-minute stretch of each block.

    The stretches are counted from the block's first frame and the last one is whatever is left: A02's table, when the
    blocks are the graded halves. `ball_visible` is (n,) bool, already the ball being usable (a position with both
    coordinates finite).
    """
    visible = _bool_array("ball_visible", ball_visible, 1)
    fps = _fps(fps)
    segment = int(round(SEGMENT_MINUTES * 60 * fps))
    return {half: np.array([(~visible[lo:min(lo + segment, stop)]).mean() for lo in range(start, stop, segment)])
            for half, start, stop in _blocks(blocks, len(visible))}


def player_stats(players_present, slot_team, players_xy_m, players_speed, blocks, fps):
    """The slots' statistics inside the blocks, as a dict; arrays are in the order block, slot, time.

    players_present  (n, S) bool; slot_team (S,) 0 home, 1 away; players_xy_m (n, S, 2) metres, NaN where absent;
    players_speed (n, S), NaN where the speed is not given (always where absent); blocks and fps as in ball_stats.

    A slot is present on a frame only if players_present is True and both coordinates are finite; a slot-frame that
    fails the second test is absent and counted (dropped_slot_frames).

    Scalars: frames (in the blocks), slots, dropped_slot_frames, present_mean, present_share_22 (frames with 22
    present), present_share_le14, present_slot_frames, tracklets, wrong_team_frames, wrong_team_share,
    wrong_team_episodes_home and _away (episodes with that team in surplus), speed_nan_frames, speed_nan_share,
    jitter_triples and jitter_rms_m (NaN with no present triple). Arrays: present_hist (S + 1,) int64, the frames with
    0..S present; tracklet_lengths_s; wrong_team_episodes_s in time order, both teams together; speed_nan_runs, in
    frames, int64. ValueError, naming the argument, for anything that would make a number meaningless.
    """
    flag = _bool_array("players_present", players_present, 2)
    n, slots = flag.shape
    team = np.asarray(slot_team)
    if team.shape != (slots,):
        raise ValueError("slot_team has {} entries but players_present has {} slots; it says which team each slot "
                         "belongs to".format(team.size, slots))
    if not np.isin(team, (0, 1)).all():
        raise ValueError("slot_team must be 0 (home) or 1 (away) for every slot")
    xy = np.asarray(players_xy_m, dtype=np.float64)
    if xy.shape != (n, slots, 2):
        raise ValueError("players_xy_m must be ({}, {}, 2), the position of every slot in metres, but has shape {}"
                         .format(n, slots, xy.shape))
    speed = np.asarray(players_speed, dtype=np.float64)
    if speed.shape != (n, slots):
        raise ValueError("players_speed must be ({}, {}), the speed of every slot, but has shape {}"
                         .format(n, slots, speed.shape))
    fps = _fps(fps)
    blocks = _blocks(blocks, n)
    finite = np.isfinite(xy).all(axis=2)
    stray = int((~flag & finite).sum())
    if stray:
        raise ValueError("players_present and players_xy_m disagree on {} slot-frames: a slot is marked absent where "
                         "a position is given; build both from the same tracking".format(stray))
    present = flag & finite

    home, away = team == 0, team == 1
    hist = np.zeros(slots + 1, dtype=np.int64)
    frames = nan_frames = triples = dropped = 0
    squares = 0.0
    tracklets, episodes, runs = [], [], []
    surplus = {"home": 0, "away": 0}
    wrong = 0
    for _, start, stop in blocks:
        inside = present[start:stop]
        counts = inside.sum(axis=1)
        hist += np.bincount(counts, minlength=slots + 1)
        frames += stop - start
        dropped += int((flag[start:stop] & ~finite[start:stop]).sum())

        _, first, last = _column_runs(inside)
        tracklets.append((last - first) / fps)

        at_home, at_away = inside[:, home].sum(axis=1), inside[:, away].sum(axis=1)
        homes = (at_home > TEAM_SIZE) & (at_away < TEAM_SIZE)
        aways = (at_away > TEAM_SIZE) & (at_home < TEAM_SIZE)
        wrong += int(homes.sum() + aways.sum())
        firsts, stops = zip(_runs(homes), _runs(aways))
        surplus["home"] += firsts[0].size
        surplus["away"] += firsts[1].size
        order = np.argsort(np.concatenate(firsts), kind="stable")
        episodes.append((np.concatenate(stops) - np.concatenate(firsts))[order] / fps)

        silent = inside & np.isnan(speed[start:stop])
        nan_frames += int(silent.sum())
        _, first, last = _column_runs(silent)
        runs.append(last - first)

        # p[t+1] - 2 p[t] + p[t-1] where the slot is present at all three frames, all inside this block.
        triple = inside[2:] & inside[1:-1] & inside[:-2]
        if triple.any():
            second = xy[start + 2:stop] - 2.0 * xy[start + 1:stop - 1] + xy[start:stop - 2]
            squares += float((second * second).sum(axis=2)[triple].sum())
            triples += int(triple.sum())

    slot_frames = int((hist * np.arange(slots + 1)).sum())
    tracklet_lengths = np.concatenate(tracklets)
    return {
        "frames": frames, "slots": slots, "dropped_slot_frames": dropped, "present_hist": hist,
        "present_slot_frames": slot_frames, "present_mean": slot_frames / frames,
        "present_share_22": int(hist[FULL_COUNT:FULL_COUNT + 1].sum()) / frames,
        "present_share_le14": int(hist[:LOW_COUNT + 1].sum()) / frames,
        "tracklets": int(tracklet_lengths.size), "tracklet_lengths_s": tracklet_lengths,
        "wrong_team_frames": wrong, "wrong_team_share": wrong / frames,
        "wrong_team_episodes_s": np.concatenate(episodes),
        "wrong_team_episodes_home": surplus["home"], "wrong_team_episodes_away": surplus["away"],
        "speed_nan_frames": nan_frames, "speed_nan_share": nan_frames / slot_frames if slot_frames else math.nan,
        "speed_nan_runs": np.concatenate(runs).astype(np.int64),
        "jitter_triples": triples, "jitter_rms_m": math.sqrt(squares / triples) if triples else math.nan,
    }


def _usable(flag, xy):
    """The flag AND both coordinates finite: what ball_stats and player_stats count as present."""
    return flag & np.isfinite(xy).all(axis=-1)


def _runs(mask):
    """(first, stop) of the maximal runs of True in a 1-D bool array: int64 arrays, half-open, in time order."""
    padded = np.zeros(len(mask) + 2, dtype=np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _column_runs(mask):
    """(column, first, stop) of the maximal runs of True down each column of an (n, S) bool array.

    Ordered by column and then by time, so the k-th start and the k-th stop are the same run (np.nonzero lists a row
    of the padded transpose after another, and a run has exactly one rising and one falling edge).
    """
    n, columns = mask.shape
    padded = np.zeros((columns, n + 2), dtype=np.int8)
    padded[:, 1:-1] = mask.T
    edges = np.diff(padded, axis=1)
    column, first = np.nonzero(edges == 1)
    _, stop = np.nonzero(edges == -1)
    return column, first, stop


def _bool_array(name, value, ndim):
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != ndim:
        raise ValueError("{} must be a {}-D bool array, one value per frame{}; it is {} with shape {}".format(
            name, ndim, " and slot" if ndim == 2 else "", array.dtype, array.shape))
    return array


def _pitch(pitch):
    try:
        length, width = (float(side) for side in pitch)
    except (TypeError, ValueError):
        raise ValueError("pitch must be (length, width) in metres, got {!r}".format(pitch)) from None
    if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
        raise ValueError("pitch must be positive and finite, got {!r}".format(pitch))
    return length, width


def _fps(fps):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise ValueError("fps must be a number, got {!r}".format(fps)) from None
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be positive and finite, got {!r}".format(fps))
    return fps


def _blocks(blocks, n):
    """The blocks as a list of (half, start, stop) ints sorted by start; ValueError unless they fit n frames."""
    checked = []
    for block in blocks:
        try:
            half, start, stop = (operator.index(value) for value in block)
        except (TypeError, ValueError):
            raise ValueError("blocks must be (half, start, stop) triples of integers, got {!r}".format(block)) from None
        if start < 0 or stop > n:
            raise ValueError("blocks: half {} is frames {}-{}, outside the {} frames of the arrays".format(
                half, start, stop, n))
        if stop <= start:
            raise ValueError("blocks: half {} is frames {}-{}, an empty block".format(half, start, stop))
        checked.append((half, start, stop))
    if not checked:
        raise ValueError("blocks is empty: give the stretches to measure, [(half, start, stop), ...]")
    halves = [half for half, _, _ in checked]
    for half in sorted(set(halves)):
        if halves.count(half) > 1:
            raise ValueError("blocks: half {} is given twice; one block per half".format(half))
    checked.sort(key=lambda block: block[1])
    for (half, start, stop), (other, other_start, _) in zip(checked, checked[1:]):
        if other_start < stop:
            raise ValueError("blocks: half {} (frames {}-{}) and half {} (from frame {}) overlap".format(
                half, start, stop, other, other_start))
    return checked


# -- the hazard fit (A08) ---------------------------------------------------------------------------------------------

def hazard_rows(ball_present, ball_xy_m, players_present, players_xy_m, spans, pitch, fps):
    """({half: (X, y)}, counts): the at-risk rows of the spans and whether the next frame starts an interior gap.

    ball_present (n,) bool and ball_xy_m (n, 2) metres, players_present (n, S) bool and players_xy_m (n, S, 2) metres,
    as ball_stats and player_stats take them (a ball or a slot is present only if its flag says so and both coordinates
    are finite). spans is [(half, kickoff, stop)], half-open, each half from its KICK-OFF: hazard.features counts the
    minute from a span's first frame. pitch and fps are as everywhere.

    An at-risk frame r is a frame of a span on which the ball is seen: its row of X is hazard.features at r (the
    features of the last visible frame, which say how likely the NEXT frame is to start a gap), and y[r] is 1 if frame
    r + 1 starts an interior gap (ball_stats' definition: a visible frame follows it inside the span) and else 0. Three
    kinds of visible frame are not at risk and are counted instead of used:
        edge        the frame before a gap that runs to the span's end: whether that is an onset is not for the span
                    to say, so it is neither an event nor a non-event;
        last_frame  the last frame of a span, when the ball is seen on it: no frame follows it in the span;
        nan_rows    a frame with a feature the tracking cannot give (the ball's speed before its first step, no player
                    present), of which nan_events are onsets.
    counts: visible (frames of the spans with the ball), edge, last_frame, nan_rows, nan_events, rows and events (the
    rows used, and the onsets among them), and rows_<half> and events_<half>. visible = edge + last_frame + nan_rows +
    rows, and events + nan_events is the number of interior gaps.
    """
    flag = _bool_array("ball_present", ball_present, 1)
    n = len(flag)
    ball = np.asarray(ball_xy_m, dtype=np.float64)
    if ball.shape != (n, 2):
        raise ValueError("calibration.hazard_rows: ball_present has {} frames but ball_xy_m has shape {}; both are one "
                         "value per frame of the match, the position in metres".format(n, ball.shape))
    slots = _bool_array("players_present", players_present, 2)
    players = np.asarray(players_xy_m, dtype=np.float64)
    if slots.shape[0] != n or players.shape != slots.shape + (2,):
        raise ValueError("calibration.hazard_rows: players_present has shape {} and players_xy_m {}; players_present "
                         "is (n, slots) with one row per frame of the match, and players_xy_m (n, slots, 2), the "
                         "positions in metres".format(slots.shape, players.shape))
    spans = _blocks(spans, n)
    visible = _usable(flag, ball)
    here = _usable(slots, players)

    counts = dict.fromkeys(("visible", "edge", "last_frame", "nan_rows", "nan_events", "rows", "events"), 0)
    rows = {}
    for half, start, stop in spans:
        length = stop - start
        seen = visible[start:stop]
        table = hazard.features(np.where(seen[:, None], ball[start:stop], np.nan),
                                np.where(here[start:stop, :, None], players[start:stop], np.nan),
                                pitch, fps, start, start)
        first, last = _runs(~seen)
        after = last < length                               # a visible frame follows the gap inside the span
        event = np.zeros(length, dtype=bool)
        event[first[(first > 0) & after] - 1] = True        # the frame before an interior gap
        edge = np.zeros(length, dtype=bool)
        edge[first[(first > 0) & ~after] - 1] = True        # the frame before a gap that runs to the span's end
        final = np.zeros(length, dtype=bool)
        final[-1] = seen[-1]
        at_risk = seen & ~edge & ~final
        defined = ~np.isnan(table).any(axis=1)
        used = at_risk & defined
        rows[half] = (table[used], event[used].astype(np.float64))
        lost = at_risk & ~defined
        for name, value in (("visible", seen.sum()), ("edge", edge.sum()), ("last_frame", final.sum()),
                            ("nan_rows", lost.sum()), ("nan_events", (lost & event).sum()), ("rows", used.sum()),
                            ("events", (used & event).sum())):
            counts[name] += int(value)
        counts["rows_{}".format(half)] = int(used.sum())
        counts["events_{}".format(half)] = int((used & event).sum())
    return rows, counts


def duration_cells(interior_durations_s, gap_onset_minute, gap_onset_near_line):
    """(table, fallback, own): the interior gaps' durations in seconds by onset class and third of the half.

    The three arguments are the PARALLEL arrays of a Calibration (one entry per interior gap). The class is 1 for a
    near-line onset and 0 otherwise (the class sample_mask reads at the at-risk frame); the third is
    hazard.third_of_minute of the onset minute: 0 before minute 15, 1 before 30, 2 after.

    table[class][third] is a read-only float64 array of the cell's durations, in the order given (not rounded). The
    rule, fixed in advance: a cell with fewer than MIN_CELL_DURATIONS (5) durations uses its class's durations pooled
    over all three thirds; fallback lists those cells, as ((class, third), ...) sorted, and own[(class, third)] is the
    number of durations a cell had of its own. ValueError for arrays that are not parallel, a NaN minute, or a class
    that has no duration at all (then there is nothing to draw its gaps' lengths from).
    """
    durations = np.asarray(interior_durations_s, dtype=np.float64)
    minute = np.asarray(gap_onset_minute, dtype=np.float64)
    near = np.asarray(gap_onset_near_line)
    if durations.ndim != 1 or minute.shape != durations.shape or near.shape != durations.shape:
        raise ValueError("calibration.duration_cells: interior_durations_s, gap_onset_minute and gap_onset_near_line "
                         "are parallel, one entry per interior gap, but have shapes {}, {} and {}".format(
                             durations.shape, minute.shape, near.shape))
    if near.dtype != np.bool_:
        raise ValueError("calibration.duration_cells: gap_onset_near_line must be a bool array, got {}".format(
            near.dtype))
    third = hazard.third_of_minute(minute)
    table, fallback, own = {}, [], {}
    for cls in CLASSES:
        inside = near == bool(cls)
        pooled = durations[inside]
        if not pooled.size:
            raise ValueError("calibration.duration_cells: no interior gap has onset class {} ({}): the hazard needs "
                             "the lengths of the gaps of both classes to draw from".format(
                                 cls, "near a line" if cls else "away from the lines"))
        table[cls] = {}
        for part in THIRDS:
            cell = durations[inside & (third == part)]
            own[(cls, part)] = int(cell.size)
            if cell.size < MIN_CELL_DURATIONS:
                fallback.append((cls, part))
                cell = pooled
            table[cls][part] = _frozen(cell)
    return table, tuple(fallback), own


def fit_hazard(ball_present, ball_xy_m, players_present, players_xy_m, spans, pitch, fps,
               interior_durations_s, gap_onset_minute, gap_onset_near_line):
    """The hazard of our own ball gaps (A08), as a dict; the arguments are hazard_rows' and duration_cells'.

    The model is hazard.fit_logistic on the six hazard.FEATURES, with the penalty HAZARD_L2 and, for a fit that does not
    converge (hazard.NotConverged), the penalty times 10, at most 3 times (_fit; the l2 used is kept, and a failure at
    every penalty is an error, never a model). It is fitted on both halves pooled; and on one half and scored on the
    other, both ways, together with a model of the minute of the half alone, for the four AUCs. spans must be halves 1
    and 2, and each needs onsets and non-onsets. interior_durations_s, gap_onset_minute and gap_onset_near_line are
    ball_stats' arrays of the same spans.

    model, l2 (the pooled fit's), c, auc ({"full_1to2", "full_2to1", "time_1to2", "time_2to1"}, rounded to 1e-9),
    go (both full AUCs >= HAZARD_GO_AUC, on the stored values), durations / fallback_cells (duration_cells), counts
    (hazard_rows' and "cell_<class>_<third>", the durations a cell has of its own) and cross_l2 (the penalty used by
    each of the four cross-half fits). c is the onsets per at-risk frame divided by the model's mean probability over
    the same rows: 1, to the precision of the fit, by construction (the intercept is not penalised). It is stored for
    completeness; the rate transfer to the benchmark is the `context` profile's own one-time, label-free rescale.
    """
    rows, counts = hazard_rows(ball_present, ball_xy_m, players_present, players_xy_m, spans, pitch, fps)
    if sorted(rows) != [1, 2]:
        raise ValueError("calibration.fit_hazard: the spans must be the halves 1 and 2, one each (the hazard is fitted "
                         "on one half and scored on the other), got halves {}".format(sorted(rows)))
    interior = np.asarray(interior_durations_s)
    if interior.size != counts["events"] + counts["nan_events"]:
        raise ValueError("calibration.fit_hazard: the interior-gap arrays hold {} gaps but the spans have {} onsets; "
                         "build them from the same spans (ball_stats(...) on them)".format(
                             interior.size, counts["events"] + counts["nan_events"]))
    for half in (1, 2):
        events, size = int(rows[half][1].sum()), len(rows[half][1])
        if events == 0 or events == size:
            raise ValueError("calibration.fit_hazard: half {} has {} onsets among {} at-risk frames; each half is "
                             "scored in turn and needs both outcomes".format(half, events, size))

    minute = [hazard.FEATURES.index("minute_of_half")]
    auc, cross_l2 = {}, {}
    for kind, columns in (("full", slice(None)), ("time", minute)):
        for train, test in ((1, 2), (2, 1)):
            key = "{}_{}to{}".format(kind, train, test)
            model, cross_l2[key] = _fit(rows[train][0][:, columns], rows[train][1])
            auc[key] = _round(hazard.auc(hazard.predict(model, rows[test][0][:, columns]), rows[test][1]))

    X = np.concatenate([rows[1][0], rows[2][0]])
    y = np.concatenate([rows[1][1], rows[2][1]])
    model, l2 = _fit(X, y)
    c = rate_scale(y, hazard.predict(model, X))
    table, fallback, own = duration_cells(interior_durations_s, gap_onset_minute, gap_onset_near_line)
    counts.update({"cell_{}_{}".format(cls, third): number for (cls, third), number in own.items()})
    return {"model": model, "l2": float(l2), "c": c, "auc": auc,
            "go": bool(min(auc["full_1to2"], auc["full_2to1"]) >= HAZARD_GO_AUC),
            "durations": table, "fallback_cells": fallback, "counts": counts, "cross_l2": cross_l2}


def rate_scale(onsets, prob):
    """c: the onsets per at-risk frame divided by the model's mean probability over the same frames, rounded to 1e-9.

    onsets is the 0/1 outcome of each at-risk frame and prob the model's probability for it: a hazard with c times the
    probabilities expects the observed number of onsets on these frames. On the rows a model was fitted on it is 1 to
    the precision of the fit, by construction (the intercept is not penalised, so the mean probability is the rate),
    whatever the penalty, and fit_hazard takes it on those rows only; it differs from 1 only on rows with another rate.
    """
    onsets, prob = np.asarray(onsets, dtype=np.float64), np.asarray(prob, dtype=np.float64)
    if onsets.ndim != 1 or onsets.shape != prob.shape or not onsets.size:
        raise ValueError("calibration.rate_scale: onsets and prob must be 1-D, of one length and not empty, got {} "
                         "and {}".format(onsets.shape, prob.shape))
    if not (prob > 0).all():
        raise ValueError("calibration.rate_scale: every probability must be above 0, so that the mean is")
    return _round(float(onsets.mean()) / float(prob.mean()))


def _fit(X, y):
    """(model, l2 used): hazard.fit_logistic at HAZARD_L2, refitted with the penalty times 10 (at most 3 times) each
    time it does not converge. Only that failure is answered so; every other error is raised at once, and so is the
    last."""
    l2, tried = HAZARD_L2, []
    while True:
        tried.append(l2)
        try:
            return hazard.fit_logistic(X, y, l2=l2), l2
        except hazard.NotConverged as error:
            if len(tried) > HAZARD_REFITS:
                raise ValueError("calibration: the hazard fit did not converge at l2 = {} (the penalty of {:g}, then "
                                 "times {:g}, at most {} refits); the last error: {}".format(
                                     ", ".join("{:g}".format(value) for value in tried), HAZARD_L2,
                                     HAZARD_L2_FACTOR, HAZARD_REFITS, error)) from error
            l2 *= HAZARD_L2_FACTOR


# -- the calibration --------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Calibration:
    """What BAR-ATH's tracking looks like, for the degrader to reproduce and fidelity.py to check against.

    Frame-level data of the match: nothing here is printed or exported (every array field has repr=False; use
    describe() for a report). Compare two with digest(), not ==. Every array is read-only. Everything is measured on
    each half's span from its kick-off to the end of its graded block, except missing_share_5min (the graded blocks,
    A02's published table) and baseline_outputs (the graded halves, as the published baselines are).

    The hazard_* fields (A08, version 2) are where our tracker loses the ball, fitted on those spans: the logistic
    model of "the next frame starts an interior gap" (on the standardised hazard.FEATURES), the penalty it was fitted
    with, the scale c (1 on BAR-ATH's own rows by construction, stored for completeness), the four cross-half AUCs,
    the go decision for the `context` profile, the interior gaps' durations in SECONDS by onset class and third of the
    half ({class: {third: array}}, where a cell with fewer than 5 uses its class's pooled durations) and the cells that
    fell back. The counts behind them are scalars named hazard_*.
    """
    ball_masks: dict = field(repr=False)             # {half: bool array}, True where the ball is missing
    present_counts: dict = field(repr=False)         # {half: int16 array}, slots present, the same frames
    gap_durations_s: np.ndarray = field(repr=False)  # every gap, in seconds
    # The next three are PARALLEL: entry i of each is the same interior gap, in time order.
    interior_durations_s: np.ndarray = field(repr=False)  # its duration in seconds
    gap_onset_minute: np.ndarray = field(repr=False)      # the minute of its onset frame, from the kick-off
    gap_onset_near_line: np.ndarray = field(repr=False)   # its onset's line distance < 2 m
    missing_share_5min: dict = field(repr=False)     # {half: float64 array}, per 5 minutes from the first graded frame
    tracklet_lengths_s: np.ndarray = field(repr=False)
    wrong_team_episodes_s: np.ndarray = field(repr=False)
    wrong_team_share: float                          # of the frames of the spans
    present_hist: np.ndarray = field(repr=False)     # (47,) int64, frames of the spans with 0..46 slots present
    speed_nan_share: float                           # NaN speeds over present slot-frames
    speed_nan_runs: np.ndarray = field(repr=False)   # in frames
    player_jitter_rms_m: float
    public_window: dict      # nearest_agreement, judged_frames, dead_frames, dead_ball_visible_share (the public GT)
    baseline_outputs: dict   # {width: {possessions, poss_per_min, d_share}} of nearest_w{width}, over the graded halves
    scalars: dict            # every other scalar of ball_stats and player_stats, ball_speed_nan_share and hazard_*
    hazard_model: hazard.LogisticModel = field(repr=False)   # the pooled fit: coef on the standardised FEATURES
    hazard_l2: float                                         # the penalty it was fitted with (1, or 10 x that ...)
    hazard_c: float                                          # onsets per row / mean probability: 1 by construction
    hazard_auc: dict                                         # HAZARD_AUC_KEYS: full and time-only, cross-half
    hazard_go: bool                                          # both full-model cross-half AUCs >= HAZARD_GO_AUC
    hazard_durations_s: dict = field(repr=False)             # {class: {third: float64 array}}, seconds
    hazard_fallback_cells: tuple                             # ((class, third), ...): cells that used the class's pool


def measure(inputs, public_gt_codes):
    """The Calibration of a match (Inputs, with its ATD) and the public truth's codes, (n_frames,), X where unlabelled.

    The yardstick configs run on the inputs through run.predict and output.summarise: no file is written. ValueError
    for a truth of another length, a kick-off that is not inside its graded block, a yardstick without the
    nearest_player source, and a hazard that cannot be fitted (fit_hazard: a half without an onset, a fit that does not
    converge at any penalty, a class of gap with no duration).
    """
    # Here, not at the top: from Task 11 pipeline/benchmark/run.py imports this module, and a module-level import of
    # run would make a cycle.
    from pipeline import evidence, output
    from pipeline.benchmark.run import predict
    from pipeline.config import load_config

    gt = np.asarray(public_gt_codes)
    if gt.shape != (inputs.n_frames,):
        raise ValueError("public_gt_codes has {} frames but the match has {}; a truth covers every frame, X where it "
                         "has no label".format(gt.size, inputs.n_frames))
    spans = []
    for half, start, stop in inputs.graded:
        if half not in inputs.kickoffs:
            raise ValueError("inputs.kickoffs has no kick-off for half {}".format(half))
        if not start <= inputs.kickoffs[half] < stop:
            raise ValueError("half {}: the kick-off, frame {}, is not inside its graded block {}-{}".format(
                half, inputs.kickoffs[half], start, stop))
        spans.append((half, inputs.kickoffs[half], stop))

    atd = inputs.atd
    ball_xy_m, players_xy_m = inputs.metres(atd.ball_xy), inputs.metres(atd.players_xy)
    ball = ball_stats(atd.ball_present, spans, ball_xy_m, inputs.pitch, inputs.fps)
    players = player_stats(atd.players_present, atd.slot_team, players_xy_m, atd.players_speed, spans, inputs.fps)
    fit = fit_hazard(atd.ball_present, ball_xy_m, atd.players_present, players_xy_m, spans, inputs.pitch, inputs.fps,
                     ball["interior_durations_s"], ball["gap_onset_minute"], ball["gap_onset_near_line"])
    ball_ok = _usable(atd.ball_present, atd.ball_xy)
    counts = _usable(atd.players_present, atd.players_xy).sum(axis=1).astype(np.int16)
    in_span = np.zeros(inputs.n_frames, dtype=bool)
    for _, start, stop in spans:
        in_span[start:stop] = True
    seen = in_span & ball_ok   # the frames of the spans that have a ball: T31 asks about those

    baselines, nearest = {}, None
    for width in YARDSTICK_WIDTHS:
        config = load_config(_config_path(width))
        codes, sources = predict(config, inputs)
        summary = output.summarise(codes, inputs.graded, inputs.fps)
        baselines[width] = {
            "possessions": summary["possessions"],
            "poss_per_min": _round(summary["possessions"] / (summary["graded_frames"] / inputs.fps / 60.0)),
            "d_share": _round(summary["labels_graded"]["D"] / summary["graded_frames"])}
        if nearest is None:
            # The evidence is the same in all four configs (nearest_player has no parameter): the first one's is read.
            # This is R001's method for A01's 77.8%: pipeline.evidence.describe against the public truth, which counts
            # the frames on which the source and the truth both have an opinion.
            nearest = next((row for row in evidence.describe(sources, inputs.graded_mask, gt)
                            if row["name"] == NEAREST_SOURCE), None)
            if nearest is None:
                raise ValueError("{}: the config has no evidence source named {!r}, whose agreement with the public "
                                 "truth the calibration records; the yardsticks nearest_w*.toml all have it".format(
                                     config.stem, NEAREST_SOURCE))
    dead = gt == D
    window = {"nearest_agreement": _round(nearest["home_agreement"]),
              "judged_frames": int(nearest["home_judged_frames"]), "dead_frames": int(dead.sum()),
              "dead_ball_visible_share": _round(ball_ok[dead].mean()) if dead.any() else None}

    scalars = {}
    for stats in (ball, players):
        for name, value in stats.items():
            if not isinstance(value, (np.ndarray, dict)) and name not in OWN_FIELDS:
                scalars.setdefault(name, _round(value))   # "frames" is in both: the same number, from the same spans
    scalars["ball_speed_nan_share"] = _round(np.isnan(atd.ball_speed[seen]).mean()) if seen.any() else None
    scalars.update({"hazard_" + name: value for name, value in fit["counts"].items()})
    scalars.update({"hazard_l2_" + name: _round(value) for name, value in fit["cross_l2"].items()})
    return Calibration(
        ball_masks={half: _frozen(~ball_ok[kickoff:stop]) for half, kickoff, stop in spans},
        present_counts={half: _frozen(counts[kickoff:stop]) for half, kickoff, stop in spans},
        gap_durations_s=_frozen(ball["gap_durations_s"]), interior_durations_s=_frozen(ball["interior_durations_s"]),
        gap_onset_minute=_frozen(ball["gap_onset_minute"]), gap_onset_near_line=_frozen(ball["gap_onset_near_line"]),
        missing_share_5min={half: _frozen(array)
                            for half, array in segment_shares(ball_ok, inputs.graded, inputs.fps).items()},
        tracklet_lengths_s=_frozen(players["tracklet_lengths_s"]),
        wrong_team_episodes_s=_frozen(players["wrong_team_episodes_s"]),
        wrong_team_share=_round(players["wrong_team_share"]),
        present_hist=_frozen(players["present_hist"]), speed_nan_share=_round(players["speed_nan_share"]),
        speed_nan_runs=_frozen(players["speed_nan_runs"]), player_jitter_rms_m=_round(players["jitter_rms_m"]),
        public_window=window, baseline_outputs=baselines, scalars=scalars,
        hazard_model=fit["model"], hazard_l2=fit["l2"], hazard_c=fit["c"], hazard_auc=fit["auc"], hazard_go=fit["go"],
        hazard_durations_s=fit["durations"], hazard_fallback_cells=fit["fallback_cells"])


def public_gt():
    """The public ground truth as codes, (n_frames,) int8: the 5 minutes it labels, X everywhere else."""
    import possession_metrics as pm
    import run_evaluation  # the scorer's own resolver for "public"; evaluation/ is on sys.path (pipeline/__init__.py)
    return pm.label_codes(pm.read_label_csv(run_evaluation.find_gt("public")))


def describe(cal):
    """A Calibration as plain scalars, short quantile lists and small dicts: what a report may say, JSON-ready.

    Nothing in it is longer than 20 values, so nothing in it could rebuild a frame. Floats are rounded to 1e-9.
    """
    hist = cal.present_hist
    cumulative = np.cumsum(hist)
    present = {name: int(np.searchsorted(cumulative, level * cumulative[-1], side="left")) for name, level in
               HIST_QUANTILES}
    # A minute before the kick-off (none in a calibration, whose spans start there) goes into the first bin, and one
    # past the last bin into the last, so that no onset can break the count or make the list longer than MAX_BINS.
    bins = np.clip(np.floor(cal.gap_onset_minute / ONSET_BIN_MINUTES).astype(np.int64), 0, MAX_BINS - 1)
    model = cal.hazard_model
    cells = {}
    for cls in sorted(cal.hazard_durations_s):
        for third in sorted(cal.hazard_durations_s[cls]):
            used = cal.hazard_durations_s[cls][third]
            cells["class{}_third{}".format(cls, third)] = {
                "n": cal.scalars.get("hazard_cell_{}_{}".format(cls, third)), "n_used": int(used.size),
                "median_s": _round(np.median(used)), "pooled": (cls, third) in cal.hazard_fallback_cells}
    return {
        "version": CALIBRATION_VERSION,
        "scalars": dict(sorted(cal.scalars.items())),
        "wrong_team_share": cal.wrong_team_share, "speed_nan_share": cal.speed_nan_share,
        "player_jitter_rms_m": cal.player_jitter_rms_m,
        "gap_s": _quantiles(cal.gap_durations_s), "interior_gap_s": _quantiles(cal.interior_durations_s),
        "tracklet_s": _quantiles(cal.tracklet_lengths_s),
        "wrong_team_episode_s": _quantiles(cal.wrong_team_episodes_s),
        "speed_nan_run_frames": _quantiles(cal.speed_nan_runs),
        "present_slots": present,
        "onsets_per_10_min": np.bincount(bins).tolist() if bins.size else [],
        "missing_share_5min": {half: [_round(share) for share in shares]
                               for half, shares in sorted(cal.missing_share_5min.items())},
        "public_window": dict(cal.public_window),
        "baseline_outputs": {width: dict(row) for width, row in sorted(cal.baseline_outputs.items())},
        # A08: the model on the standardised features, the four cross-half AUCs and the decision; per (class, third)
        # cell the durations the cell had of its own, the number used and their median, and whether it used its class's
        # pool.
        "hazard": {
            "features": list(hazard.FEATURES), "coef": [_round(value) for value in model.coef],
            "intercept": _round(model.intercept), "mean": [_round(value) for value in model.mean],
            "scale": [_round(value) for value in model.scale], "l2": _round(cal.hazard_l2), "c": _round(cal.hazard_c),
            "auc": {key: _round(value) for key, value in sorted(cal.hazard_auc.items())}, "go": cal.hazard_go,
            "fallback_cells": [list(cell) for cell in cal.hazard_fallback_cells], "duration_cells": cells,
        },
    }


def digest(cal):
    """The sha256 of everything a Calibration holds: two are the same measurement if and only if their digests agree."""
    sha = hashlib.sha256()
    for item in dataclasses.fields(cal):
        _feed(sha, item.name, getattr(cal, item.name))
    return sha.hexdigest()


def _feed(sha, name, value):
    sha.update(name.encode("utf-8") + b"=")
    if isinstance(value, np.ndarray):
        sha.update("{}{}".format(value.dtype, value.shape).encode("ascii"))
        sha.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        sha.update(b"{")
        for key in sorted(value, key=str):
            _feed(sha, str(key), value[key])
        sha.update(b"}")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):   # the hazard's model: arrays, not their repr
        sha.update(b"<")
        for item in dataclasses.fields(value):
            _feed(sha, item.name, getattr(value, item.name))
        sha.update(b">")
    else:
        sha.update(repr(value).encode("utf-8"))   # repr round-trips a float exactly
    sha.update(b";")


def _quantiles(values):
    """{"n", "p10", ..., "p99", "max"} of an array, rounded; None for each when there are no values."""
    row = {"n": int(values.size)}
    if values.size:
        row.update({name: _round(np.quantile(values, level)) for name, level in QUANTILES})
        row["max"] = _round(values.max())
    else:
        row.update({name: None for name, _ in QUANTILES}, max=None)
    return row


def _round(value):
    """A summary number as a plain Python int or float, floats rounded to 1e-9 (NaN stays NaN)."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), DECIMALS)


def _frozen(array):
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


# -- the cache --------------------------------------------------------------------------------------------------------

def _text_sha256(path):
    """The sha256 of a text file's bytes with CRLF read as LF: a checkout that changes only the line endings (git does,
    on Windows, when core.autocrlf is on) is not a change of the file."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


# The hash of this file, taken now, as the module is imported, like dfl.PARSER_SHA256: the part of the cache key that
# says which definitions made a cache. A process that imported the module before the file changed goes on running the
# old definitions, and keeps using the cache that fits them; a process that starts later rebuilds.
CALIBRATION_SHA256 = _text_sha256(__file__)
# And of hazard.py, which makes the version-2 fields (the features, the fit, the AUC, the thirds): a checkout with
# another hazard.py has another key and so another folder, and cannot read a fit that this one's would not make.
HAZARD_SHA256 = _text_sha256(hazard.__file__)


def load(data_dir=None, rebuild=False, inputs=None, public_gt_codes=None):
    """The Calibration of BAR-ATH, from the cache when it is fresh, otherwise measured and stored.

    data_dir defaults to games.DATA_DIR (MTAH_DATA_DIR); the cache is <data dir>/cache/benchmark/calibration/<digest>/
    (cache_folder: one folder per key) and nothing else is written. The ATD comes from the pipeline's own cache and is
    never rebuilt here beyond what Inputs does when that cache is stale. rebuild=True measures again. `inputs` and
    `public_gt_codes` stand in for the real match and its public truth (the self-checks pass a synthetic bundle).
    """
    if inputs is None:
        from pipeline.inputs import Inputs  # here, not above: it pulls in pandas and the starter loaders
        inputs = Inputs(data_dir=data_dir)
    data_dir = data_dir or inputs.data_dir
    gt = public_gt() if public_gt_codes is None else np.asarray(public_gt_codes)
    key = _key(inputs, gt)
    folder = cache_folder(data_dir, key)
    manifest = None if rebuild else _read_manifest(folder)
    if manifest is not None and manifest.get("key") == key:
        cal = _read(folder, manifest)
        if cal is not None:
            return cal
    cal = measure(inputs, gt)
    _write(cal, folder, key)
    return cal


def cache_folder(data_dir, key):
    """The folder of one cache key: <data dir>/cache/benchmark/calibration/<digest>/.

    The digest is the first 16 hex digits of the sha256 of the key's canonical JSON (sorted keys, no spaces), so a key
    is the same folder whatever order its entries come in. Checkouts that run different code (their keys differ in
    code_sha256 or hazard_sha256) can share one data dir and never touch each other's files. Old folders stay where
    they are: each is a few MB, and the key is what says whether one is still wanted.
    """
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return os.path.join(data_dir, *CACHE_FOLDER, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:KEY_DIGITS])


def _config_path(width):
    return os.path.join(CONFIG_FOLDER, "nearest_w{}.toml".format(width))


def _key(inputs, gt):
    """Everything the numbers depend on, as JSON-normal data: a cache is read only if all of it is unchanged."""
    from pipeline.config import load_config
    key = {
        "calibration_version": CALIBRATION_VERSION,
        "code_sha256": CALIBRATION_SHA256,
        "hazard_sha256": HAZARD_SHA256,
        "atd_sources": inputs.atd_manifest["sources"],
        "atd_cache_version": inputs.atd_manifest["cache_version"],
        "numpy": np.__version__,
        "configs": {str(width): hashlib.sha256(load_config(_config_path(width)).text.encode("utf-8")).hexdigest()
                    for width in YARDSTICK_WIDTHS},
        "public_gt_sha256": hashlib.sha256(np.ascontiguousarray(gt, dtype=np.int8).tobytes()).hexdigest(),
        "n_frames": inputs.n_frames,
        "pitch": [float(side) for side in inputs.pitch],   # games.py's constants: every distance and duration uses them
        "fps": inputs.fps,
        "graded": [list(block) for block in inputs.graded],
        "kickoffs": {str(half): kickoff for half, kickoff in sorted(inputs.kickoffs.items())},
    }
    return json.loads(json.dumps(key, sort_keys=True))


def _read_manifest(folder):
    """The manifest as a dict, or None if there is none or it cannot be read as one: an unreadable cache is absent."""
    try:
        with open(os.path.join(folder, MANIFEST), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):   # no file, no access, not UTF-8, not JSON (JSONDecodeError is a ValueError)
        return None
    return manifest if isinstance(manifest, dict) else None


def _write(cal, folder, key):
    """Arrays first, manifest last, with the old manifest removed up front (atd_cache._save does the same).

    The manifest is written whole to a temp file of this process in the folder and moved over the old one (os.replace),
    so that no reader sees half of it. An old manifest that cannot be removed counts as absent: Windows refuses to
    remove a file that another process has open, and a checkout with the same code (the same key, so this folder) may
    be reading it or building it as well. Whatever it holds is the same measurement, and it is replaced below. Windows
    refuses the move over an open manifest too, so the move is retried (_move_into_place): a hold that lasts past the
    retries raises PermissionError out of here, with no temp file left, and a rerun recovers."""
    os.makedirs(folder, exist_ok=True)
    manifest_path = os.path.join(folder, MANIFEST)
    try:
        os.remove(manifest_path)
    except OSError:   # none (FileNotFoundError), or one that another process holds open (PermissionError on Windows)
        pass
    for name, array in _arrays(cal).items():
        np.save(os.path.join(folder, name + ".npy"), array, allow_pickle=False)
    manifest = {
        "key": key, "halves": sorted(cal.ball_masks), "scalars": cal.scalars,
        "wrong_team_share": cal.wrong_team_share, "speed_nan_share": cal.speed_nan_share,
        "player_jitter_rms_m": cal.player_jitter_rms_m, "public_window": cal.public_window,
        "baseline_outputs": {str(width): row for width, row in sorted(cal.baseline_outputs.items())},
        "hazard": {
            "model": {"coef": cal.hazard_model.coef.tolist(), "intercept": cal.hazard_model.intercept,
                      "mean": cal.hazard_model.mean.tolist(), "scale": cal.hazard_model.scale.tolist()},
            "l2": cal.hazard_l2, "c": cal.hazard_c, "auc": cal.hazard_auc, "go": cal.hazard_go,
            "fallback_cells": [list(cell) for cell in cal.hazard_fallback_cells],
        },
    }
    temp_path = "{}.{}.tmp".format(manifest_path, os.getpid())
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _move_into_place(temp_path, manifest_path)
    except BaseException:   # no temp file is left behind, and the error is the caller's
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _move_into_place(temp_path, manifest_path):
    """os.replace, tried again while the OS refuses it with PermissionError: REPLACE_RETRIES times, REPLACE_WAIT_S
    apart.

    On Windows the move fails while another process has the target open, and a reader of the other checkout that
    shares this folder has it open for milliseconds. After the last retry the PermissionError is the caller's; any other
    error is raised at once, as there is nothing to wait for."""
    for attempt in range(REPLACE_RETRIES + 1):
        try:
            os.replace(temp_path, manifest_path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES:
                raise
            time.sleep(REPLACE_WAIT_S)


def _arrays(cal):
    """{file name without .npy: array} of everything the cache keeps."""
    arrays = {name: getattr(cal, name) for name in FLAT_ARRAYS}
    for name in HALF_ARRAYS:
        arrays.update({"{}_{}".format(name, half): array for half, array in getattr(cal, name).items()})
    for cls, thirds in cal.hazard_durations_s.items():
        arrays.update({"hazard_durations_s_{}_{}".format(cls, third): array for third, array in thirds.items()})
    return arrays


def _read(folder, manifest):
    """The Calibration the cache holds, or None if any part of it cannot be read as one."""
    try:
        halves = [int(half) for half in manifest["halves"]]
        flat = {name: _read_array(folder, name, dtype) for name, dtype in FLAT_ARRAYS.items()}
        by_half = {name: {half: _read_array(folder, "{}_{}".format(name, half), dtype) for half in halves}
                   for name, dtype in HALF_ARRAYS.items()}
        for half in halves:
            if by_half["ball_masks"][half].shape != by_half["present_counts"][half].shape:
                return None
        parallel = (flat["interior_durations_s"], flat["gap_onset_minute"], flat["gap_onset_near_line"])
        if len({array.shape for array in parallel}) != 1:
            return None   # the three are parallel: one entry per interior gap
        return Calibration(
            wrong_team_share=manifest["wrong_team_share"], speed_nan_share=manifest["speed_nan_share"],
            player_jitter_rms_m=manifest["player_jitter_rms_m"], public_window=dict(manifest["public_window"]),
            baseline_outputs={int(width): dict(row) for width, row in manifest["baseline_outputs"].items()},
            scalars=dict(manifest["scalars"]), **_read_hazard(folder, manifest["hazard"]), **flat, **by_half)
    except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return None


def _read_hazard(folder, saved):
    """The hazard fields of a Calibration from the manifest's "hazard" entry and the six duration files, as keywords.

    ValueError (or KeyError, TypeError) for anything that is not what _write put there: a model of the wrong width or
    with a value that is not a finite number, a scale that is not positive, AUCs that are not the four of
    HAZARD_AUC_KEYS, a decision that is not a bool, a cell that is not a (class, third), a duration file that is
    missing, empty or not 1-D (sample_mask needs a length to draw in every cell).
    """
    width = len(hazard.FEATURES)
    arrays = {}
    for name in ("coef", "mean", "scale"):
        values = np.array(saved["model"][name], dtype=np.float64)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError("hazard model {} is not {} finite numbers".format(name, width))
        arrays[name] = _frozen(values)
    if (arrays["scale"] <= 0).any():
        raise ValueError("hazard model scale is not positive")
    saved_auc = saved["auc"]
    if not isinstance(saved_auc, dict) or sorted(saved_auc) != sorted(HAZARD_AUC_KEYS):
        raise ValueError("hazard auc does not have exactly the keys {}".format(HAZARD_AUC_KEYS))
    if type(saved["go"]) is not bool:
        raise ValueError("hazard go is not a bool")
    cells = tuple(tuple(operator.index(part) for part in cell) for cell in saved["fallback_cells"])
    if any(len(cell) != 2 or cell[0] not in CLASSES or cell[1] not in THIRDS for cell in cells):
        raise ValueError("hazard fallback_cells are not (class, third) pairs")
    durations = {cls: {third: _read_array(folder, "hazard_durations_s_{}_{}".format(cls, third), np.float64)
                       for third in THIRDS} for cls in CLASSES}
    if any(not array.size for thirds in durations.values() for array in thirds.values()):
        raise ValueError("a hazard duration cell is empty")
    return {"hazard_model": hazard.LogisticModel(coef=arrays["coef"], intercept=_number(saved["model"]["intercept"]),
                                                 mean=arrays["mean"], scale=arrays["scale"]),
            "hazard_l2": _number(saved["l2"]), "hazard_c": _number(saved["c"]),
            "hazard_auc": {key: _number(saved_auc[key]) for key in HAZARD_AUC_KEYS}, "hazard_go": saved["go"],
            "hazard_durations_s": durations, "hazard_fallback_cells": cells}


def _number(value):
    """`value` as a float if it is a finite JSON number (not a bool or a string); ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{!r} is not a finite number".format(value))
    return float(value)


def _read_array(folder, name, dtype):
    """One cached array as a read-only 1-D array of `dtype`; ValueError if it is not, or cannot be read as one.

    The file is opened here and np.load is given the handle, as dfl._read_array does (docs/traps.md T34): given a path,
    np.load hands a file that starts like a zip to an NpzFile, which fails on one cut short after np.load has given the
    file up, so that nothing closes it, and on Windows the rebuild that follows (it rewrites the file) is refused. A zip
    cut short under an .npy name is zipfile.BadZipFile, which is not a ValueError: it becomes one here, so that the
    cache that holds it is rebuilt like any other that cannot be read."""
    path = os.path.join(folder, name + ".npy")
    with open(path, "rb") as handle:
        try:
            array = np.load(handle, allow_pickle=False)
        except zipfile.BadZipFile as error:
            raise ValueError("{}.npy is not a readable numpy file ({})".format(name, error)) from error
        if not isinstance(array, np.ndarray):
            close = getattr(array, "close", None)
            if close is not None:
                close()   # an NpzFile of a file object leaves the file to its owner; this is its own bookkeeping
            raise ValueError("{}.npy is not a plain numpy array".format(name))
        if array.dtype != dtype or array.ndim != 1:
            raise ValueError("{}.npy is not a 1-D {} array".format(name, np.dtype(dtype)))
        return _frozen(array)
