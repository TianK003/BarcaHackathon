"""Our own tracking, measured: what the benchmark's degradation has to reproduce (docs/results.md, A07).

    ball = ball_stats(ball_present, blocks, ball_xy_m, pitch, fps)          # gaps, where they start, per 5 minutes
    players = player_stats(present, slot_team, xy_m, speed, blocks, fps)    # slots present, tracklets, wrong team, ...
    cal = load()                                                            # BAR-ATH, measured once and cached
    describe(cal)                                                           # scalars and quantiles, for reports

Why it exists. P01 validates pipeline choices on other matches, degraded until their tracking looks like ours: the ball
missing in long gaps, 46 slots for 22 players, slots that carry the wrong team. "Looks like ours" needs a ruler, and
the ruler has to be one set of functions: ball_stats and player_stats measure BAR-ATH here, and fidelity.py applies the
same two functions, unchanged, to a degraded benchmark match, so that a difference between the two numbers is a
difference between the matches and not between two ways of counting. They are pure, take plain arrays and know nothing
about either match.

Definitions (metres throughout: normalised coordinates times the pitch, 105.3 x 68.0; Inputs.metres)
    block           a graded half, (half, start, stop), half-open. Every statistic is taken inside the blocks only: the
                    second half is graded from 2,934 frames before its kick-off, and those frames count.
    gap             a maximal run of frames without a ball inside a block. A gap that touches the block's first or last
                    frame is a gap (its duration counts, and so does its share of missing frames) but is not interior.
    interior gap    a gap with a visible frame on both sides; the frame before it is its onset. Only these have an
                    onset, a minute and a line distance, because only they can be asked where the tracker lost the ball.
    line distance   min(x, L - x, y, W - y) of the ball, negative beyond a line.
    near-line onset an interior gap whose onset frame has a line distance below 2 m (strictly: 2.0 is not near).
    onset minute    the onset frame's time in minutes from the start of its block, so the second half's minutes start
                    117 s before its kick-off. (The hazard of hazard.py counts from the kick-off; that is the fit's
                    own feature, measured on the frames the fit uses.)
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

Two spans. The statistics above are those of the graded blocks, the frames that are scored. The two fields that the
benchmark transplants, Calibration.ball_masks and present_counts, run from each half's kick-off to the end of its graded
block instead: a benchmark match starts at its kick-off, and the second half's 117 s before ours are all D, not play.

What a Calibration holds is frame-level data of BAR-ATH (the ball's mask and the slot counts of every frame), which must
never reach a log, a report, a CSV or git. Every array field has repr=False, describe() returns scalars and short
quantile lists only, and the cache sits under <data dir>/cache/benchmark/calibration/, which git ignores. The baselines
(baseline_outputs) go through run.predict and pipeline.output.summarise: no CSV of labels of this match is written.

The cache. load() keeps a Calibration as .npy arrays and a manifest.json whose key holds everything the numbers depend
on: the ATD cache's record of its source files, CALIBRATION_VERSION, the numpy version, the sha256 of the four yardstick
configs, the public truth, and the graded blocks and kick-offs. A cache is reused only if the whole key matches, so an
edited config cannot reuse the old baseline outputs. An unreadable cache (a manifest that is not JSON or lacks a field,
an array missing or cut short) is absent, never an error: everything in it can be measured again from the sources. The
arrays are written first and the manifest last, with the old manifest removed up front, as pipeline/inputs/atd_cache.py
and dfl.load_match do, so a run killed half-way leaves no manifest.
"""

import dataclasses
import hashlib
import json
import math
import operator
import os
from dataclasses import dataclass, field

import numpy as np

from pipeline.contracts import D

# Params
# Bump it whenever the numbers change for a reason the key cannot see: a change to the statistics below, and above all a
# change to the pipeline code the baselines run through (pipeline/evidence/, pipeline/decode/, pipeline/refine/,
# pipeline/output/summarise, benchmark/run.py), because the key hashes the four configs and not the code they call.
CALIBRATION_VERSION = 1
CACHE_FOLDER = ("cache", "benchmark", "calibration")   # <data dir>/cache/benchmark/calibration/
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
FLAT_ARRAYS = {"gap_durations_s": np.float64, "gap_onset_minute": np.float64, "gap_onset_near_line": np.bool_,
               "tracklet_lengths_s": np.float64, "wrong_team_episodes_s": np.float64, "present_hist": np.int64,
               "speed_nan_runs": np.int64}
HALF_ARRAYS = {"ball_masks": np.bool_, "present_counts": np.int16, "missing_share_5min": np.float64}
# The three statistics that have a field of their own in the Calibration (wrong_team_share, speed_nan_share and
# player_jitter_rms_m); scalars holds the others.
OWN_FIELDS = ("wrong_team_share", "speed_nan_share", "jitter_rms_m")


# -- the statistics ---------------------------------------------------------------------------------------------------

def ball_stats(ball_present, blocks, ball_xy_m, pitch, fps):
    """The ball's gaps inside the blocks, as a dict; arrays are in time order, block by block.

    ball_present  (n,) bool: the tracker sees the ball.
    blocks        [(half, start, stop)], half-open, disjoint, one per half: the graded blocks (Inputs.graded).
    ball_xy_m     (n, 2) metres from a pitch corner, NaN exactly where ball_present is False.
    pitch         (length, width) in metres; fps frames a second.

    Scalars: graded_frames, missing_frames, missing_share, gaps, interior_gaps, near_line_onsets and near_line_share
    (of the interior gaps; NaN when there are none). Arrays: gap_durations_s (one per gap), then one per interior gap:
    gap_onset_minute, gap_onset_line_m (the line distance at the onset frame) and gap_onset_near_line (bool). And
    missing_share_5min, {half: float64 array} of the missing share of each 5-minute stretch of the block.
    ValueError, naming the argument, for anything that would make a number meaningless.
    """
    present = _bool_array("ball_present", ball_present, 1)
    n = len(present)
    xy = np.asarray(ball_xy_m, dtype=np.float64)
    if xy.shape != (n, 2):
        raise ValueError("ball_present has {} frames but ball_xy_m has shape {}; both are one value per frame of the "
                         "match, the position in metres".format(n, xy.shape))
    length, width = _pitch(pitch)
    fps = _fps(fps)
    blocks = _blocks(blocks, n)
    disagree = int((present != ~np.isnan(xy[:, 0])).sum())
    if disagree:
        raise ValueError("ball_present and ball_xy_m disagree on {} frames: the ball is present where its position is "
                         "NaN or the reverse; build both from the same tracking".format(disagree))

    segment = int(round(SEGMENT_MINUTES * 60 * fps))
    durations, minutes, lines, shares = [], [], [], {}
    graded = missing = 0
    for half, start, stop in blocks:
        gone = ~present[start:stop]
        graded += stop - start
        missing += int(gone.sum())
        first, last = _runs(gone)
        durations.append((last - first) / fps)
        interior = (first > 0) & (last < stop - start)
        onset = first[interior] - 1                      # the last visible frame before each interior gap
        x, y = xy[start + onset, 0], xy[start + onset, 1]
        lines.append(np.minimum.reduce([x, length - x, y, width - y]))
        minutes.append(onset / fps / 60.0)
        shares[half] = np.array([gone[lo:lo + segment].mean() for lo in range(0, stop - start, segment)])
    durations, minutes, lines = (np.concatenate(part) for part in (durations, minutes, lines))
    near = lines < NEAR_LINE_M
    return {
        "graded_frames": graded, "missing_frames": missing, "missing_share": missing / graded,
        "gaps": int(durations.size), "interior_gaps": int(minutes.size), "near_line_onsets": int(near.sum()),
        "near_line_share": float(near.sum()) / minutes.size if minutes.size else math.nan,
        "gap_durations_s": durations, "gap_onset_minute": minutes, "gap_onset_line_m": lines,
        "gap_onset_near_line": near, "missing_share_5min": shares,
    }


def player_stats(players_present, slot_team, players_xy_m, players_speed, blocks, fps):
    """The slots' statistics inside the blocks, as a dict; arrays are in the order block, slot, time.

    players_present  (n, S) bool; slot_team (S,) 0 home, 1 away; players_xy_m (n, S, 2) metres, NaN where absent;
    players_speed (n, S), NaN where the speed is not given (always where absent); blocks and fps as in ball_stats.

    Scalars: slots, present_mean, present_share_22 (frames with 22 present), present_share_le14, present_slot_frames,
    tracklets, wrong_team_frames, wrong_team_share, wrong_team_episodes_home and _away (episodes with that team in
    surplus), speed_nan_frames, speed_nan_share, jitter_triples and jitter_rms_m (NaN with no present triple). Arrays:
    present_hist (S + 1,) int64, the frames with 0..S present; tracklet_lengths_s; wrong_team_episodes_s in time order,
    both teams together; speed_nan_runs, in frames, int64. ValueError, naming the argument, for anything that would
    make a number meaningless.
    """
    present = _bool_array("players_present", players_present, 2)
    n, slots = present.shape
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
    disagree = int((present != ~np.isnan(xy[:, :, 0])).sum())
    if disagree:
        raise ValueError("players_present and players_xy_m disagree on {} slot-frames: a slot is present where its "
                         "position is NaN or the reverse; build both from the same tracking".format(disagree))

    home, away = team == 0, team == 1
    hist = np.zeros(slots + 1, dtype=np.int64)
    frames = nan_frames = triples = 0
    squares = 0.0
    tracklets, episodes, runs = [], [], []
    surplus = {"home": 0, "away": 0}
    wrong = 0
    for _, start, stop in blocks:
        inside = present[start:stop]
        counts = inside.sum(axis=1)
        hist += np.bincount(counts, minlength=slots + 1)
        frames += stop - start

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
        "slots": slots, "present_hist": hist, "present_slot_frames": slot_frames, "present_mean": slot_frames / frames,
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
        raise ValueError("blocks is empty: give the graded blocks, [(half, start, stop), ...]")
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


# -- the calibration --------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Calibration:
    """What BAR-ATH's tracking looks like, for the degrader to reproduce and fidelity.py to check against.

    Frame-level data of the match: nothing here is printed or exported (every array field has repr=False; use
    describe() for a report). Compare two with digest(), not ==. Every array is read-only. The masks and the counts of
    a half run from its kick-off to the end of its graded block; everything else is measured on the graded blocks.
    """
    ball_masks: dict = field(repr=False)             # {half: bool array}, True where the ball is missing
    present_counts: dict = field(repr=False)         # {half: int16 array}, slots present, the same frames
    gap_durations_s: np.ndarray = field(repr=False)  # every gap, in seconds
    gap_onset_minute: np.ndarray = field(repr=False)     # per interior gap, minutes from the block's start
    gap_onset_near_line: np.ndarray = field(repr=False)  # per interior gap, line distance of the onset < 2 m
    missing_share_5min: dict = field(repr=False)     # {half: float64 array}, one per 5 minutes from the block's start
    tracklet_lengths_s: np.ndarray = field(repr=False)
    wrong_team_episodes_s: np.ndarray = field(repr=False)
    wrong_team_share: float                          # of graded frames
    present_hist: np.ndarray = field(repr=False)     # (47,) int64, graded frames with 0..46 slots present
    speed_nan_share: float                           # NaN speeds over present slot-frames
    speed_nan_runs: np.ndarray = field(repr=False)   # in frames
    player_jitter_rms_m: float
    public_window: dict      # nearest_agreement, judged_frames, dead_frames, dead_ball_visible_share (the public GT)
    baseline_outputs: dict   # {width: {possessions, poss_per_min, d_share}} of nearest_w{width}, over the graded halves
    scalars: dict            # every other scalar of ball_stats and player_stats, and ball_speed_nan_share


def measure(inputs, public_gt_codes):
    """The Calibration of a match (Inputs, with its ATD) and the public truth's codes, (n_frames,), X where unlabelled.

    The yardstick configs run on the inputs through run.predict and output.summarise: no file is written. ValueError
    for a truth of another length, and for a kick-off that is not inside its graded block.
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
    spans = {}
    for half, start, stop in inputs.graded:
        if half not in inputs.kickoffs:
            raise ValueError("inputs.kickoffs has no kick-off for half {}".format(half))
        if not start <= inputs.kickoffs[half] < stop:
            raise ValueError("half {}: the kick-off, frame {}, is not inside its graded block {}-{}".format(
                half, inputs.kickoffs[half], start, stop))
        spans[half] = (inputs.kickoffs[half], stop)

    atd = inputs.atd
    ball = ball_stats(atd.ball_present, inputs.graded, inputs.metres(atd.ball_xy), inputs.pitch, inputs.fps)
    players = player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy), atd.players_speed,
                           inputs.graded, inputs.fps)
    counts = atd.players_present.sum(axis=1).astype(np.int16)
    seen = inputs.graded_mask & atd.ball_present   # the graded frames that have a ball: T31 asks about those

    baselines, nearest = {}, None
    for width in YARDSTICK_WIDTHS:
        codes, sources = predict(load_config(_config_path(width)), inputs)
        summary = output.summarise(codes, inputs.graded, inputs.fps)
        baselines[width] = {
            "possessions": summary["possessions"],
            "poss_per_min": _round(summary["possessions"] / (summary["graded_frames"] / inputs.fps / 60.0)),
            "d_share": _round(summary["labels_graded"]["D"] / summary["graded_frames"])}
        if nearest is None:
            # The evidence is the same in all four configs (nearest_player has no parameter): the first one's is read.
            # This is R001's method for A01's 77.8%: pipeline.evidence.describe against the public truth, which counts
            # the frames on which the source and the truth both have an opinion.
            nearest = next(row for row in evidence.describe(sources, inputs.graded_mask, gt)
                           if row["name"] == "nearest_player")
    dead = gt == D
    window = {"nearest_agreement": _round(nearest["home_agreement"]),
              "judged_frames": int(nearest["home_judged_frames"]), "dead_frames": int(dead.sum()),
              "dead_ball_visible_share": _round(atd.ball_present[dead].mean()) if dead.any() else None}

    scalars = {}
    for stats in (ball, players):
        for name, value in stats.items():
            if not isinstance(value, (np.ndarray, dict)) and name not in OWN_FIELDS:
                scalars[name] = _round(value)
    scalars["ball_speed_nan_share"] = _round(np.isnan(atd.ball_speed[seen]).mean()) if seen.any() else None
    return Calibration(
        ball_masks={half: _frozen(~atd.ball_present[kickoff:stop]) for half, (kickoff, stop) in spans.items()},
        present_counts={half: _frozen(counts[kickoff:stop]) for half, (kickoff, stop) in spans.items()},
        gap_durations_s=_frozen(ball["gap_durations_s"]), gap_onset_minute=_frozen(ball["gap_onset_minute"]),
        gap_onset_near_line=_frozen(ball["gap_onset_near_line"]),
        missing_share_5min={half: _frozen(array) for half, array in ball["missing_share_5min"].items()},
        tracklet_lengths_s=_frozen(players["tracklet_lengths_s"]),
        wrong_team_episodes_s=_frozen(players["wrong_team_episodes_s"]),
        wrong_team_share=_round(players["wrong_team_share"]),
        present_hist=_frozen(players["present_hist"]), speed_nan_share=_round(players["speed_nan_share"]),
        speed_nan_runs=_frozen(players["speed_nan_runs"]), player_jitter_rms_m=_round(players["jitter_rms_m"]),
        public_window=window, baseline_outputs=baselines, scalars=scalars)


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
    bins = np.minimum((cal.gap_onset_minute // ONSET_BIN_MINUTES).astype(np.int64), MAX_BINS - 1)
    return {
        "version": CALIBRATION_VERSION,
        "scalars": dict(sorted(cal.scalars.items())),
        "wrong_team_share": cal.wrong_team_share, "speed_nan_share": cal.speed_nan_share,
        "player_jitter_rms_m": cal.player_jitter_rms_m,
        "gap_s": _quantiles(cal.gap_durations_s), "tracklet_s": _quantiles(cal.tracklet_lengths_s),
        "wrong_team_episode_s": _quantiles(cal.wrong_team_episodes_s),
        "speed_nan_run_frames": _quantiles(cal.speed_nan_runs),
        "present_slots": present,
        "onsets_per_10_min": np.bincount(bins).tolist() if bins.size else [],
        "missing_share_5min": {half: [_round(share) for share in shares]
                               for half, shares in sorted(cal.missing_share_5min.items())},
        "public_window": dict(cal.public_window),
        "baseline_outputs": {width: dict(row) for width, row in sorted(cal.baseline_outputs.items())},
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

def load(data_dir=None, rebuild=False, inputs=None, public_gt_codes=None):
    """The Calibration of BAR-ATH, from the cache when it is fresh, otherwise measured and stored.

    data_dir defaults to games.DATA_DIR (MTAH_DATA_DIR); the cache is <data dir>/cache/benchmark/calibration/ and
    nothing else is written. The ATD comes from the pipeline's own cache and is never rebuilt here beyond what Inputs
    does when that cache is stale. rebuild=True measures again. `inputs` and `public_gt_codes` stand in for the real
    match and its public truth (the self-checks pass a synthetic bundle).
    """
    if inputs is None:
        from pipeline.inputs import Inputs  # here, not above: it pulls in pandas and the starter loaders
        inputs = Inputs(data_dir=data_dir)
    data_dir = data_dir or inputs.data_dir
    gt = public_gt() if public_gt_codes is None else np.asarray(public_gt_codes)
    key = _key(inputs, gt)
    folder = os.path.join(data_dir, *CACHE_FOLDER)
    manifest = None if rebuild else _read_manifest(folder)
    if manifest is not None and manifest.get("key") == key:
        cal = _read(folder, manifest)
        if cal is not None:
            return cal
    cal = measure(inputs, gt)
    _write(cal, folder, key)
    return cal


def _config_path(width):
    return os.path.join(CONFIG_FOLDER, "nearest_w{}.toml".format(width))


def _key(inputs, gt):
    """Everything the numbers depend on, as JSON-normal data: a cache is read only if all of it is unchanged."""
    from pipeline.config import load_config
    key = {
        "calibration_version": CALIBRATION_VERSION,
        "atd_sources": inputs.atd_manifest["sources"],
        "numpy": np.__version__,
        "configs": {str(width): hashlib.sha256(load_config(_config_path(width)).text.encode("utf-8")).hexdigest()
                    for width in YARDSTICK_WIDTHS},
        "public_gt_sha256": hashlib.sha256(np.ascontiguousarray(gt, dtype=np.int8).tobytes()).hexdigest(),
        "n_frames": inputs.n_frames,
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
    """Arrays first, manifest last, with the old manifest removed up front (atd_cache._save does the same)."""
    os.makedirs(folder, exist_ok=True)
    manifest_path = os.path.join(folder, MANIFEST)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    for name, array in _arrays(cal).items():
        np.save(os.path.join(folder, name + ".npy"), array, allow_pickle=False)
    manifest = {
        "key": key, "halves": sorted(cal.ball_masks), "scalars": cal.scalars,
        "wrong_team_share": cal.wrong_team_share, "speed_nan_share": cal.speed_nan_share,
        "player_jitter_rms_m": cal.player_jitter_rms_m, "public_window": cal.public_window,
        "baseline_outputs": {str(width): row for width, row in sorted(cal.baseline_outputs.items())},
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _arrays(cal):
    """{file name without .npy: array} of everything the cache keeps."""
    arrays = {name: getattr(cal, name) for name in FLAT_ARRAYS}
    for name in HALF_ARRAYS:
        arrays.update({"{}_{}".format(name, half): array for half, array in getattr(cal, name).items()})
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
        return Calibration(
            wrong_team_share=manifest["wrong_team_share"], speed_nan_share=manifest["speed_nan_share"],
            player_jitter_rms_m=manifest["player_jitter_rms_m"], public_window=dict(manifest["public_window"]),
            baseline_outputs={int(width): dict(row) for width, row in manifest["baseline_outputs"].items()},
            scalars=dict(manifest["scalars"]), **flat, **by_half)
    except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError):
        return None


def _read_array(folder, name, dtype):
    """One cached array as a read-only 1-D array of `dtype`; ValueError if it is not."""
    array = np.load(os.path.join(folder, name + ".npy"), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype != dtype or array.ndim != 1:
        raise ValueError("{}.npy is not a 1-D {} array".format(name, np.dtype(dtype)))
    return _frozen(array)
