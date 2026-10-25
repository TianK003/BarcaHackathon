"""conventions: when does the truth's label change? Every change of it, timed against events our tracking can see.

    python pipeline/benchmark/conventions.py public  [--gt PATH] [--data-dir D]
    python pipeline/benchmark/conventions.py dfl     [--profile tracking] [--seeds 3] [--truth raw|coarse]
                                                     [--matches ID ...] [--data-dir D]
    python pipeline/benchmark/conventions.py compare [--profile tracking] [--truth raw] [--data-dir D]

    trans = transitions_of(codes, runs)      # [(frame, from, to, type)], type in TYPES
    found = events(inputs)                   # {detector: {"frames": int array, "team": int8 array or None}}
    table = lags(trans, found, runs, fps)    # {(type, detector): {"n", "found", "lags"}}
    print(summarise(table))                  # n, share found, median, p25, p75 in frames and in seconds

Why it exists (docs/proposals.md P02). Nobody knows exactly when our ground truth flips its label: at the first touch of
the new team, at the line, at the whistle? A pipeline that decides a change a few frames away from where the truth puts
it loses every possession it touches (a possession matches only above an IoU), so the convention is worth measuring.
It cannot be measured against a video by an LLM, which may not see the match, so it is measured against events that
the tracking shows without any label: each change of the truth's label is timed against the nearest such event, and the
lag is the number. It runs with the same code on two sources, and their difference is the finding: the public 5 minutes
of our match (BAR-ATH, `public`) and the seven DFL matches of the benchmark (`dfl`), whose truth comes from DFL's own
flags. Where a detector's lag is the same on both, it is the detector's own bias (a ball that is lost from view 10
frames before it crosses the line does so in every match); where the public window's lag differs from DFL's, that is
a convention of our truth that DFL's does not share, which `compare` puts as a gap with a 90% bootstrap interval.

The changes of the truth (transitions_of). Inside each block of the truth (on the public window the runs of
pm.valid_runs, on a DFL match its graded halves), every frame whose code differs from the one before, typed by what
changed: `team` (H to A or A to H), `into_dead` (H or A to D) and `out_of_dead` (D to H or A). A change into or out of X
inside a block is refused: a truth that has X inside a block says nothing of when it changed.

The detectors (events). Label-free, computed from `Inputs` alone, and each only for the changes that it can speak about
(APPLIES). A frame is where the event is first seen, and an event is a position in time and nothing else, so that the
number a lag gives is the number of frames between the label and the thing the tracking saw.

    touch_start   a `team` or `out_of_dead` change: the first frame of a kept `controlled_touch` episode (P04), the
                  rule as main.toml has it (radius 2.0 m, max_rel_speed 6.0 m/s, min_frames 3), with confidence 0.99
                  and hold 0.5 so that `home` is a number exactly on the kept frames. It carries the episode's team,
                  and only an event of the new team can be the start of the new team's possession. Where the label
                  turns at the first touch, the lag is 0.
    line_out      an `into_dead` change: the first frame of each run of frames with the ball visible and
                  `kinematics.outside_m` above 0, so a ball that is on a line is not out. A gap in the ball cuts a run
                  in two, and the second run's first frame is an event as well.
    ball_lost     an `into_dead` change: the first missing frame of each gap in the ball that opens inside a block. A
                  gap that is there from the block's first frame has no onset (nothing was seen to be lost), and is
                  not an event; one that runs to the block's end still opens inside it. The tracker loses the ball in
                  long gaps (A01), often at a stoppage, so this is where the ball vanishes.
    ball_moves    an `out_of_dead` change: the first frame with the ball's own speed (from positions, never the ATD's
                  channel, T31) at or above `play_features.STILL_SPEED`, after at least 25 frames in a row below it.
                  A still ball that is then struck is a restart; a ball that was not seen to be still (a gap before it,
                  a run shorter than 25 frames) is not.
    model_dead    an `into_dead` change: where `dead_ball_model`'s alive (the file main.toml uses) crosses 0.5
    model_alive   downward, and an `out_of_dead` one: where it crosses 0.5 upward. The model has its own bias (a
                  window of features), and DFL's lag for it, which it is out of fold for, is the reference.
    tag_start     an `into_dead` change: the start of a Smart Tagging interval whose code, upper-cased, is `SET PIECES`;
    tag_end       an `out_of_dead` one: the end of such an interval. BAR-ATH only: a DFL match has no tagging, its
                  `smart_tagging` raises AssetUnavailable, and the detector is left out (skipped, not empty). Which set
                  piece it is stays with a person; Smart Tagging's tag values are our vocabulary and no LLM reads them.

The window rule (lags). The lag of a detector at a change at frame c is `event - c` in frames, so a negative lag means
the event came first. The event taken is the nearest to c, the earlier on a tie, among those in
[max(c - 5 s, p), min(c + 5 s, n)], where p and n are the neighbouring changes of the truth of any kind (or the block's
first and last frame), both ends included: the window never reaches past the neighbouring changes, nor out of the
block, so a change is never matched to an event that belongs to another one (touch_stats.py reads its flip lags by the
same rule). An event with a team counts only if it is of the new team. No event in the window: the change is not found,
and is counted in n and not in found. DFL is pooled over matches and seeds by the lags themselves, never as a median
of medians.

What it writes. `public` writes two files under `<data dir>/runs/conventions/`. `public_lags.json` is read by the code
(`compare`): the cells with their lags, which are numbers of frames and no frame of the match, and the summary.
`public_transitions.csv` is for a person: one row per change of the public truth, its frame and every detector's lag
there, so that somebody who may watch the video can look at exactly those moments. No LLM reads that file, and the code
prints no row of it and no frame in it: what `public` prints is the summary (n, the share found, the median, and
the quartiles of a cell with enough changes found, see below), at most a few lines, and what the check holds it to.
`dfl` writes `dfl_<profile>_<truth>.json` (suffixed as the runner's reports are when the matches are not all seven or
the seeds not the default) and prints the pooled summary and the per-match medians. `compare` prints, per
(type, detector), the public median beside DFL's and the gap with a 90% interval: 2,000 draws with replacement of the
public changes' lags, `default_rng([2, 1])` freshly seeded for each cell so that a cell's interval does not depend on
the others, DFL's median taken as it is.

What a printout of BAR-ATH may show (`MIN_SHOWN`, `limited`, `shown`, `interval_text`). The public window has a handful
of changes of each type, so a cell is small: with one change found its median, p25 and p75 are that change's lag, and
with two or three the linear quartiles fix every lag, which is a row of the person's file on the screen. So a cell of
a BAR-ATH printout (the `public` summary, and the public side of `compare`) with fewer than MIN_SHOWN = 5 changes found
prints n, the share found and the median only: no quartile, and in place of a bootstrap interval only whether it
excludes 0. A cell with at least MIN_SHOWN found prints everything. DFL's printouts are not limited (its data are
public), and the JSON files keep every value, since the code reads them. The rule lives in those three functions and
nowhere else, so that any later printout of BAR-ATH lags (the latency and the shift tables) calls them.

No truth reaches the detectors. In `dfl` every match is served through train_dead_ball.served_inputs (a MatchInputs,
which holds the degraded tracking and no flag, as the trainer's rows are), and the truth is read from the match beside
it, only to find the changes (test_conventions_never_pass_truth). Matches are loaded one at a time and released.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numbers  # noqa: E402

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run, train_dead_ball  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.inputs import AssetUnavailable  # noqa: E402
from pipeline.config import ConfigError  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, dead_ball_model, kinematics, play_features  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
TYPES = ("team", "into_dead", "out_of_dead")
DETECTORS = ("touch_start", "line_out", "ball_lost", "ball_moves", "model_dead", "model_alive", "tag_start", "tag_end")
APPLIES = {"touch_start": ("team", "out_of_dead"), "line_out": ("into_dead",), "ball_lost": ("into_dead",),
           "ball_moves": ("out_of_dead",), "model_dead": ("into_dead",), "model_alive": ("out_of_dead",),
           "tag_start": ("into_dead",), "tag_end": ("out_of_dead",)}
WINDOW_S = 5                        # an event is looked for this far either side of a change
TOUCH = {"radius": 2.0, "max_rel_speed": 6.0, "min_frames": 3, "confidence": 0.99, "hold": 0.5}   # main.toml's rule
MODEL = "pipeline/models/dead_ball_hgb.json"        # main.toml's file, relative to the repository root
ALIVE_LEVEL = 0.5                   # model_dead / model_alive: where alive crosses this
MIN_STILL_FRAMES = 25               # ball_moves: this many frames in a row below STILL_SPEED before the first above it
SET_PIECES = "SET PIECES"           # the Smart Tagging code of tag_start / tag_end, upper-cased
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = (2, 1)             # default_rng(BOOTSTRAP_SEED), the design's, for the gap's interval
PERCENTILES = (5, 95)               # the 90% interval
FOLDER = ("runs", "conventions")    # <data dir>/runs/conventions/
PUBLIC_LAGS, PUBLIC_TRANSITIONS = "public_lags.json", "public_transitions.csv"
GT_PUBLIC = os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TRUTHS = ("raw", "coarse")
DEFAULT_PROFILE = run.DEFAULT_PROFILE
DEFAULT_SEEDS = run.DEFAULT_SEEDS
LAG_WIDTH = 7                       # the printout's column of a lag
MIN_SHOWN = 5                       # a BAR-ATH printout shows quartiles and interval bounds from this many found

_TRANSITION_TYPE = {(pm.H, pm.A): "team", (pm.A, pm.H): "team", (pm.H, pm.D): "into_dead", (pm.A, pm.D): "into_dead",
                    (pm.D, pm.H): "out_of_dead", (pm.D, pm.A): "out_of_dead"}


class ConventionsError(ValueError):
    """What the instrument cannot do with what it was given. The message names the thing and says what to change."""


# -- the changes of the truth -----------------------------------------------------------------------------------------

def transitions_of(codes, runs):
    """[(frame, from, to, type)]: every change of `codes` inside the half-open `runs`, never across one.

    `frame` is the first frame of the new label, from and to are pm's codes (ints) and type is one of TYPES. This is
    pm.transitions with each change typed. ConventionsError for a change into or out of X, which has no type (no frame
    is in the message: the truth may be BAR-ATH's).
    """
    out = []
    for frame, before, after in pm.transitions(codes, runs):
        kind = _TRANSITION_TYPE.get((before, after))
        if kind is None:
            raise ConventionsError("conventions: a label changes into or out of X inside a block of the truth; a "
                                   "block must hold H, A and D only, so give the runs that pm.valid_runs (or the "
                                   "graded halves) make")
        out.append((frame, before, after, kind))
    return out


# -- the events -------------------------------------------------------------------------------------------------------

def events(inputs):
    """{detector: {"frames": sorted int64 array, "team": int8 array or None}} for the detectors that `inputs` supports.

    The keys are those of DETECTORS, in that order. `team` is the team of the event in pm's codes (H, A) for touch_start
    and None for the others. A detector whose asset the inputs do not have (AssetUnavailable) is left out; any other
    failure is not caught. Every event lies in a graded block of the inputs.
    """
    out = {}
    for name in DETECTORS:
        try:
            out[name] = _DETECTORS[name](inputs)
        except AssetUnavailable:
            continue
    return out


def _entry(frames, team=None):
    frames = np.asarray(frames, dtype=np.int64)
    return {"frames": frames, "team": None if team is None else np.asarray(team, dtype=np.int8)}


def _run_starts(mask, blocks, skip_first=False):
    """The first frame of every run of True in `mask` inside each block (a run never crosses a block's edge).

    With `skip_first` a run that begins on the block's first frame is left out: nothing before it was seen.
    """
    frames = []
    for start, stop in blocks:
        starts, _ = kinematics.runs(mask[start:stop])
        if skip_first:
            starts = starts[starts > 0]
        frames.append(starts + start)
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int64)


def _touch_start(inputs):
    """The first frame of each kept control episode: a kept frame whose predecessor is not kept, or of another team."""
    home = controlled_touch.compute(RunContext(inputs=inputs, seed=0), **TOUCH).home
    kept = ~np.isnan(home)
    team_home = home > 0.5
    start = kept.copy()
    start[1:] &= ~(kept[:-1] & (team_home[:-1] == team_home[1:]))
    for first, _ in kinematics.blocks(inputs):
        start[first] = kept[first]
    frames = np.flatnonzero(start)
    return _entry(frames, np.where(team_home[frames], pm.H, pm.A))


def _line_out(inputs):
    ball = kinematics.ball_track(inputs)
    with np.errstate(invalid="ignore"):
        outside = ball.visible & (kinematics.outside_m(ball.xy_m, inputs.pitch) > 0.0)
    return _entry(_run_starts(outside, kinematics.blocks(inputs)))


def _ball_lost(inputs):
    visible = kinematics.ball_track(inputs).visible
    return _entry(_run_starts(~visible, kinematics.blocks(inputs), skip_first=True))


def _ball_moves(inputs):
    ball = kinematics.ball_track(inputs)
    with np.errstate(invalid="ignore"):
        still = ball.visible & (ball.speed < play_features.STILL_SPEED)
        moving = ball.visible & (ball.speed >= play_features.STILL_SPEED)
    frames = []
    for start, stop in kinematics.blocks(inputs):
        firsts, stops = kinematics.runs(still[start:stop])
        for first, last in zip(firsts, stops):
            if last - first >= MIN_STILL_FRAMES and start + last < stop and moving[start + last]:
                frames.append(start + int(last))
    return _entry(frames)


def _alive(inputs):
    return dead_ball_model.compute(RunContext(inputs=inputs, seed=0), model=MODEL).alive


def crossings(alive, downward):
    """The frames where `alive` crosses ALIVE_LEVEL downward (or upward): both neighbours are numbers (NaN is none).

    Downward is a frame below the level whose predecessor is at or above it, upward one at or above whose predecessor
    is below. A frame at the level is therefore on the alive side, and a frame after a NaN is never a crossing.
    """
    before, now = alive[:-1], alive[1:]
    with np.errstate(invalid="ignore"):
        if downward:
            crossed = (before >= ALIVE_LEVEL) & (now < ALIVE_LEVEL)
        else:
            crossed = (before < ALIVE_LEVEL) & (now >= ALIVE_LEVEL)
    return np.flatnonzero(crossed) + 1


def _model_dead(inputs):
    return _entry(crossings(_alive(inputs), downward=True))


def _model_alive(inputs):
    return _entry(crossings(_alive(inputs), downward=False))


def _set_pieces(inputs, key):
    """The sorted, distinct `key` frames of the SET PIECES intervals; AssetUnavailable if there is no tagging."""
    return _entry(np.unique([interval[key] for interval in inputs.smart_tagging
                             if str(interval["code"] or "").strip().upper() == SET_PIECES]))


def _tag_start(inputs):
    return _set_pieces(inputs, "start_frame")


def _tag_end(inputs):
    return _set_pieces(inputs, "end_frame")


_DETECTORS = {"touch_start": _touch_start, "line_out": _line_out, "ball_lost": _ball_lost, "ball_moves": _ball_moves,
              "model_dead": _model_dead, "model_alive": _model_alive, "tag_start": _tag_start, "tag_end": _tag_end}


# -- the lags ---------------------------------------------------------------------------------------------------------

def transition_lags(transitions, events, runs, fps):
    """[{detector: lag or None}]: for each of `transitions` (in the order given), the lag of each detector that applies.

    `transitions` is transitions_of's list (all the changes of the truth: the neighbours that bound a window are read
    from it), `events` is events()'s dict, `runs` the half-open blocks and `fps` the frame rate. A detector that does
    not apply to the transition's type is not in its dict; one that applies and found nothing in the window is None.
    ConventionsError for a detector that DETECTORS does not name and for a change outside every run.
    """
    unknown = [name for name in events if name not in APPLIES]
    if unknown:
        raise ConventionsError("conventions: no detector called {}; the design's are {}".format(
            ", ".join(unknown), ", ".join(DETECTORS)))
    reach = int(round(WINDOW_S * fps))
    runs = [(int(start), int(stop)) for start, stop in runs]
    entries = {}
    for name, entry in events.items():
        frames = np.asarray(entry["frames"], dtype=np.int64)
        order = np.argsort(frames, kind="stable")             # events() gives them sorted; a hand-made list may not
        entries[name] = (frames[order], None if entry["team"] is None else np.asarray(entry["team"])[order])
    by_block = {}                       # run number -> the indices of its changes, in time order
    for index in sorted(range(len(transitions)), key=lambda index: transitions[index][0]):
        frame = transitions[index][0]
        holders = [number for number, (first, last) in enumerate(runs) if first <= frame < last]
        if not holders:
            raise ConventionsError("conventions: a change of the truth lies outside every run it was given; the runs "
                                   "must be the blocks the transitions were found in")
        by_block.setdefault(holders[0], []).append(index)
    result = [None] * len(transitions)
    for number, indices in by_block.items():
        first, last = runs[number]
        changes = [transitions[index][0] for index in indices]
        for position, index in enumerate(indices):
            frame, _, new, kind = transitions[index]
            # the window is bounded by the neighbouring changes of any type, or by the block's first and last frame
            before = changes[position - 1] if position else first
            after = changes[position + 1] if position + 1 < len(changes) else last - 1
            low, high = max(frame - reach, before), min(frame + reach, after)
            row = {}
            for name, (frames, team) in entries.items():
                if kind not in APPLIES[name]:
                    continue
                inside = slice(np.searchsorted(frames, low, side="left"), np.searchsorted(frames, high, side="right"))
                near = frames[inside] if team is None else frames[inside][team[inside] == new]
                # argmin takes the first of equals, and `near` is sorted: the earlier event on a tie
                row[name] = int(near[np.argmin(np.abs(near - frame))]) - frame if len(near) else None
            result[index] = row
    return result


def lags(transitions, events, runs, fps):
    """{(type, detector): {"n", "found", "lags"}}: the lag table, by the window rule (module docstring).

    One cell for each detector in `events` and each type it applies to (APPLIES), also where there is no change of
    that type (n is 0). n counts the changes of the type, found those with an event in their window, and lags holds
    those events' lags in the order of `transitions`, as ints. The same errors as transition_lags.
    """
    rows = transition_lags(transitions, events, runs, fps)
    table = {(kind, name): {"n": 0, "found": 0, "lags": []} for name in DETECTORS if name in events
             for kind in APPLIES[name]}
    for (_, _, _, kind), row in zip(transitions, rows):
        for name, lag in row.items():
            cell = table[(kind, name)]
            cell["n"] += 1
            if lag is not None:
                cell["found"] += 1
                cell["lags"].append(lag)
    return table


def pool(tables):
    """One lag table from several: n and found summed, the lags put one after the other (never a median of medians)."""
    out = {}
    for table in tables:
        for key, cell in table.items():
            pooled = out.setdefault(key, {"n": 0, "found": 0, "lags": []})
            pooled["n"] += cell["n"]
            pooled["found"] += cell["found"]
            pooled["lags"] += cell["lags"]
    return out


def summarise(lag_table, fps=pm.FPS):
    """{(type, detector): {n, found, share_found, median, p25, p75, median_s, p25_s, p75_s}} of a lag table.

    The percentiles are numpy's (linear), in frames and in seconds at `fps`; they are None where nothing was found, and
    share_found is None where there was no change of the type.
    """
    out = {}
    for key, cell in lag_table.items():
        row = {"n": cell["n"], "found": cell["found"], "share_found": cell["found"] / cell["n"] if cell["n"] else None}
        if cell["lags"]:
            p25, median, p75 = (float(value) for value in np.percentile(cell["lags"], [25, 50, 75]))
        else:
            p25 = median = p75 = None
        row.update({"median": median, "p25": p25, "p75": p75,
                    "median_s": None if median is None else median / fps,
                    "p25_s": None if p25 is None else p25 / fps, "p75_s": None if p75 is None else p75 / fps})
        out[key] = row
    return out


def cell_name(kind, detector):
    """The key of a cell in the JSON files: "type/detector"."""
    return "{}/{}".format(kind, detector)


def parse_cell_name(name):
    kind, detector = name.split("/")
    return kind, detector


# -- the public window ------------------------------------------------------------------------------------------------

def default_gt():
    return os.path.join(ROOT, GT_PUBLIC)


def bar_ath_inputs(data_dir=None):
    """BAR-ATH's Inputs (nothing is read until a component asks)."""
    from pipeline.inputs import Inputs
    return Inputs(data_dir=data_dir)


def public(inputs, gt_path, data_dir=None, log=print):
    """Time every change of the public truth at `gt_path` against `inputs`' events; write the two files; the document.

    `inputs` is BAR-ATH's (pipeline.inputs.Inputs); the GT is a frame,label CSV with X outside the public window, whose
    runs (pm.valid_runs) are the blocks. Writes public_transitions.csv and public_lags.json under
    `<data_dir or inputs.data_dir>/runs/conventions/` and logs the summary only: counts, shares and medians, and
    quartiles for a cell with MIN_SHOWN changes found or more, never a row of the CSV or a frame (module docstring).
    ConventionsError for a GT that is not one row per frame of the match or has no graded frame, before anything is
    written; the document is JSON-ready.
    """
    labels = pm.read_label_csv(gt_path)
    reason = pm.validate_submission(labels, inputs.n_frames)
    if reason is not None:
        raise ConventionsError("conventions: the GT {} is not a label per frame of the {} frames of the match ({}); "
                               "use the public GT, evaluation/gt/BAR-ATH_possession_gt_public.csv".format(
                                   os.path.basename(gt_path), inputs.n_frames, reason))
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    if not runs:
        raise ConventionsError("conventions: the GT {} has no graded frame (every label is X), so there is no change "
                               "of the truth to time".format(os.path.basename(gt_path)))
    fps = inputs.fps
    changes = transitions_of(codes, runs)
    found = events(inputs)
    rows = transition_lags(changes, found, runs, fps)
    table = lags(changes, found, runs, fps)
    document = {"source": "public", "gt": os.path.basename(gt_path), "fps": fps, "runs": len(runs),
                "transitions": {kind: sum(1 for change in changes if change[3] == kind) for kind in TYPES},
                "events": {name: int(len(entry["frames"])) for name, entry in found.items()},
                "cells": {cell_name(*key): cell for key, cell in table.items()},
                "summary": {cell_name(*key): row for key, row in summarise(table, fps).items()}}
    folder = os.path.join(data_dir or inputs.data_dir, *FOLDER)
    os.makedirs(folder, exist_ok=True)
    _write_transitions(os.path.join(folder, PUBLIC_TRANSITIONS), changes, rows, list(found))
    run.write_json(os.path.join(folder, PUBLIC_LAGS), document)
    for line in render_public(document):
        log(line)
    log("wrote     {}".format(os.path.join(folder, PUBLIC_LAGS)))
    log("wrote     {}   (one row per change, for a person; nobody else reads it)".format(
        os.path.join(folder, PUBLIC_TRANSITIONS)))
    return document


def _write_transitions(path, changes, rows, names):
    """One row per change: frame, from, to, type, then each detector's lag (blank: not applicable, or none found)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "from", "to", "type"] + names)
        for (frame, before, after, kind), row in zip(changes, rows):
            writer.writerow([frame, pm.LABELS[before], pm.LABELS[after], kind]
                            + ["" if row.get(name) is None else row[name] for name in names])


def render_public(document):
    """The summary of the public window: a line per change type and detector, with the small cells limited (`shown`)."""
    lines = ["public    {} changes of the truth ({}), events: {}".format(
        sum(document["transitions"].values()), ", ".join("{} {}".format(kind, document["transitions"][kind])
                                                         for kind in TYPES),
        ", ".join("{} {}".format(name, count) for name, count in document["events"].items()))]
    lines += _summary_lines("public", document["summary"], document["fps"], limit=True)
    return lines


def _summary_lines(label, summary, fps, limit=False):
    """The table of a summary keyed by cell_name: with `limit` (BAR-ATH's) each row goes through `shown`, and a hidden
    quartile is a "-"."""
    lines = ["{:<10}{:<13}{:<13}{:>5}{:>8}{:>{w}}{:>{w}}{:>{w}}{:>10}".format(
        label, "type", "detector", "n", "found", "median", "p25", "p75", "median s", w=LAG_WIDTH)]
    for kind in TYPES:
        for name in DETECTORS:
            row = summary.get(cell_name(kind, name))
            if row is None:
                continue
            if limit:
                row = shown(row)
            quartiles = ["-" if row.get("limited") else _frames(row[key]) for key in ("p25", "p75")]
            lines.append("{:<10}{:<13}{:<13}{:>5}{:>8}{:>{w}}{:>{w}}{:>{w}}{:>10}".format(
                "", kind, name, row["n"], _pct(row["share_found"]), _frames(row["median"]), *quartiles,
                "n/a" if row["median_s"] is None else "{:+.2f}".format(row["median_s"]), w=LAG_WIDTH))
    lines.append("{:<10}lags in frames ({} per second): event - change, negative = the event came first; found = "
                 "share of the changes with an event in the window".format("", fps))
    if limit:
        lines.append("{:<10}-: fewer than {} changes found, so only the median is printed (a quartile would be a lag)"
                     .format("", MIN_SHOWN))
    return lines


def _pct(value):
    return "n/a" if value is None else "{:.0f}%".format(100.0 * value)


def _frames(value):
    return "n/a" if value is None else "{:+.1f}".format(value)


def limited(row, minimum=MIN_SHOWN):
    """Whether a summarise row of BAR-ATH has too few changes found for its quartiles to be printed: 1 to minimum - 1.

    A cell with nothing found has no quartile to hide. `minimum` defaults to MIN_SHOWN; the rule is this and the two
    functions below, and every printout of BAR-ATH lags goes through them (module docstring).
    """
    return 0 < row["found"] < minimum


def shown(row, minimum=MIN_SHOWN):
    """The row as a printout of BAR-ATH may show it: itself, or a copy without its quartiles if it is limited.

    A limited row keeps n, found, share_found, median and median_s, has p25, p75, p25_s and p75_s None and `limited`
    True (a renderer prints "-" for those, not "n/a", which is "nothing found"). The row given is never changed.
    """
    if not limited(row, minimum):
        return row
    return {**row, "p25": None, "p75": None, "p25_s": None, "p75_s": None, "limited": True}


def interval_text(low, high, public_row, minimum=MIN_SHOWN):
    """A bootstrap interval as a printout of BAR-ATH may show it, given the public side's summarise row.

    "n/a" without an interval, "[low, high]" in frames when the public side has `minimum` changes found or more, and
    only "excludes 0: yes" or "excludes 0: no" below that: the bounds of a bootstrap of one to four lags are lags.
    """
    if low is None or high is None:
        return "n/a"
    if limited(public_row, minimum):
        return "excludes 0: {}".format("yes" if low > 0 or high < 0 else "no")
    return "[{}, {}]".format(_frames(low), _frames(high))


# -- the DFL matches --------------------------------------------------------------------------------------------------

def measure(*, profile=DEFAULT_PROFILE, seeds=None, truth="raw", match_ids=None, matches=None, calibration=None,
            data_dir=None, log=print):
    """The lag tables of the benchmark matches under `profile`, `seeds` and `truth`, pooled and per match.

    Each match is loaded (dfl.load_match, or taken from `matches`, {id: DflMatch}), its truth read beside the inputs
    (the raw truth, or the coarse one that absorbs short possessions) to find the changes, and for every seed served
    through train_dead_ball.served_inputs (the tracking as the pipeline sees it, no flag) the events are computed and
    timed. Returns {"profile", "truth", "seeds", "matches", "fps", "table", "per_match"}: `table` is the pooled lag
    table and `per_match` {id: table}, each pooled over seeds. The runner's BenchmarkError for a profile, seeds,
    truth or match it refuses, before a match is loaded.
    """
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, numbers.Integral) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    if truth not in TRUTHS:
        raise run.BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(TRUTHS)))
    ids = run.resolve_match_ids(match_ids, matches)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))

    per_match, rates = {}, []
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        rates.append(match.fps)
        codes, _ = truths.truth_codes(match)             # read here, to find the changes; never handed to the events
        if truth == "coarse":
            codes = truths.coarse_codes(codes, match.graded, match.fps)
        blocks = [(int(start), int(stop)) for _, start, stop in match.graded]
        changes = transitions_of(codes, blocks)
        tables = []
        for seed in seed_list:
            inputs = train_dead_ball.served_inputs(match, profile, seed, calibration, data_dir)
            tables.append(lags(changes, events(inputs), blocks, match.fps))
            del inputs
        per_match[match_id] = pool(tables)
        log("match     {}   {} changes, {} seeds".format(match_id, len(changes), len(seed_list)))
        del match, codes
    if any(rate != rates[0] for rate in rates):
        raise run.BenchmarkError("the matches have different frame rates ({}); lags in frames cannot be pooled".format(
            ", ".join(str(rate) for rate in rates)))
    return {"profile": profile, "truth": truth, "seeds": seed_list, "matches": sorted(ids), "fps": rates[0],
            "table": pool([per_match[match_id] for match_id in ids]), "per_match": per_match}


def dfl_document(result):
    """The JSON-ready document of measure()'s result: pooled cells with their lags, summaries, per-match summaries."""
    fps = result["fps"]
    return {"source": "dfl", "profile": result["profile"], "truth": result["truth"], "seeds": result["seeds"],
            "matches": result["matches"], "fps": fps,
            "cells": {cell_name(*key): cell for key, cell in result["table"].items()},
            "summary": {cell_name(*key): row for key, row in summarise(result["table"], fps).items()},
            "per_match": {match_id: {cell_name(*key): row for key, row in summarise(table, fps).items()}
                          for match_id, table in sorted(result["per_match"].items())}}


def dfl_path(data_dir, profile, truth, match_ids=None, seeds=None):
    """<data dir>/runs/conventions/dfl_<profile>_<truth>[.m-<ids>][.s<n>].json, suffixed as the runner's reports are."""
    parts = ["dfl_{}_{}".format(profile, truth)] + run.name_suffixes(profile, match_ids, seeds)
    return os.path.join(data_dir, *FOLDER, ".".join(parts) + ".json")


def render_dfl(result):
    """The pooled summary, then each match's medians: counts, shares, medians and quartiles only."""
    fps = result["fps"]
    pooled = {cell_name(*key): row for key, row in summarise(result["table"], fps).items()}
    lines = _summary_lines("pooled", pooled, fps)
    cells = [key for kind in TYPES for key in ((kind, name) for name in DETECTORS) if key in result["table"]]
    ids = sorted(result["per_match"])
    lines.append("{:<10}{:<13}{:<13}".format("per match", "type", "detector") + "".join(
        "{:>9}".format(match_id) for match_id in ids) + "   (median lag in frames)")
    summaries = {match_id: summarise(result["per_match"][match_id], fps) for match_id in ids}
    for key in cells:
        lines.append("{:<10}{:<13}{:<13}".format("", *key) + "".join(
            "{:>9}".format(_frames(summaries[match_id][key]["median"])) for match_id in ids))
    lines.append("{:<10}profile {}, truth {}, seeds {}".format("", result["profile"], result["truth"],
                                                               result["seeds"]))
    return lines


# -- compare ----------------------------------------------------------------------------------------------------------

def bootstrap_gap(public_lags, dfl_lags, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """(gap, low, high): the median of the public lags minus DFL's, and the 90% bootstrap interval of that gap.

    `draws` samples of the public lags, with replacement and as many as there are, from a generator seeded with `seed`
    (default_rng([2, 1]) by design), each with its median, minus DFL's median as it is; the interval is the 5th and the
    95th percentile of those. (None, None, None) if either list is empty. Seeded afresh on every call, so a result
    depends on its two lists and nothing else.
    """
    if not len(public_lags) or not len(dfl_lags):
        return None, None, None
    public_lags = np.asarray(public_lags, dtype=np.float64)
    reference = float(np.median(dfl_lags))
    rng = np.random.default_rng(list(seed))
    medians = np.median(public_lags[rng.integers(0, len(public_lags), size=(draws, len(public_lags)))], axis=1)
    low, high = (float(value) for value in np.percentile(medians - reference, PERCENTILES))
    return float(np.median(public_lags)) - reference, low, high


def compare(public_document, dfl_document):
    """[{type, detector, public, dfl, gap, low, high}]: each cell's public summary beside DFL's, in the design's order.

    `public` and `dfl` are summarise's row (None where that source has no such cell: a detector DFL cannot run), `gap`
    the public median minus DFL's with the interval of bootstrap_gap. ConventionsError if the two documents were made
    at different frame rates.
    """
    if public_document["fps"] != dfl_document["fps"]:
        raise ConventionsError("conventions: the public lags are at {} fps and DFL's at {} fps; lags in frames of "
                               "different lengths cannot be set side by side".format(public_document["fps"],
                                                                                     dfl_document["fps"]))
    fps = public_document["fps"]
    rows = []
    for kind in TYPES:
        for name in DETECTORS:
            key = cell_name(kind, name)
            cells = {"public": public_document["cells"].get(key), "dfl": dfl_document["cells"].get(key)}
            if cells["public"] is None and cells["dfl"] is None:
                continue
            gap, low, high = bootstrap_gap(cells["public"]["lags"] if cells["public"] else [],
                                           cells["dfl"]["lags"] if cells["dfl"] else [])
            rows.append({"type": kind, "detector": name, "gap": gap, "low": low, "high": high,
                         **{side: None if cell is None else summarise({(kind, name): cell}, fps)[(kind, name)]
                            for side, cell in cells.items()}})
    return rows


def render_compare(rows):
    """The comparison's lines: per cell the public median and DFL's, then the gap and its interval, in frames."""
    lines = ["{:<13}{:<13}{:>5}{:>8}{:>9}   {:>7}{:>8}{:>9}   {:>9}   {}".format(
        "type", "detector", "n", "found", "median", "dfl n", "found", "median", "gap", "90% interval of the gap")]
    for row in rows:
        public_row, dfl_row = row["public"], row["dfl"]
        lines.append("{:<13}{:<13}{:>5}{:>8}{:>9}   {:>7}{:>8}{:>9}   {:>9}   {}".format(
            row["type"], row["detector"],
            "n/a" if public_row is None else public_row["n"], "n/a" if public_row is None else _pct(
                public_row["share_found"]), "n/a" if public_row is None else _frames(public_row["median"]),
            "n/a" if dfl_row is None else dfl_row["n"], "n/a" if dfl_row is None else _pct(dfl_row["share_found"]),
            "n/a" if dfl_row is None else _frames(dfl_row["median"]), _frames(row["gap"]),
            interval_text(row["low"], row["high"], public_row)))
    lines.append("lags in frames, event - change (negative: the event came first); gap = the public median minus "
                 "DFL's, with a 90% bootstrap interval of the public lags ({} draws)".format(BOOTSTRAP_DRAWS))
    lines.append("with fewer than {} changes found on the public side the bounds would be lags, so only whether the "
                 "interval excludes 0 is printed".format(MIN_SHOWN))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Time every change of the truth's label against label-free events of our tracking: on the public "
                    "window of our match (public), on the DFL benchmark matches (dfl), and the two side by side "
                    "(compare).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("public", "the public 5 minutes of BAR-ATH: writes public_lags.json and "
                                  "public_transitions.csv"),
                       ("dfl", "the DFL matches, degraded as the pipeline's tracking is: writes "
                               "dfl_<profile>_<truth>.json"),
                       ("compare", "the public medians against DFL's, with a bootstrap interval of the gap")):
        command = commands.add_parser(name, help=text, description=text)
        command.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        if name == "public":
            command.add_argument("--gt", help="the public GT (default {})".format(GT_PUBLIC))
        if name in ("dfl", "compare"):
            command.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
            command.add_argument("--truth", choices=TRUTHS, default="raw",
                                 help="DFL's flags as they are (raw), or with short possessions absorbed (coarse)")
        if name == "dfl":
            command.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                                 help="degradation seeds per match (default {})".format(DEFAULT_SEEDS))
            command.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run (default: all 7)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or _default_data_dir()
    try:
        if args.command == "public":
            public(bar_ath_inputs(args.data_dir), args.gt or default_gt(), data_dir=data_dir)
        elif args.command == "dfl":
            result = measure(profile=args.profile, seeds=args.seeds, truth=args.truth, match_ids=args.matches,
                             data_dir=args.data_dir)
            path = dfl_path(data_dir, args.profile, args.truth, args.matches, args.seeds)
            run.write_json(path, dfl_document(result))
            print("")
            for line in render_dfl(result):
                print(line)
            print("wrote     {}".format(path))
        else:
            path = dfl_path(data_dir, args.profile, args.truth)
            for line in render_compare(compare(_read(os.path.join(data_dir, *FOLDER, PUBLIC_LAGS), "public"),
                                               _read(path, "dfl --profile {} --truth {}".format(args.profile,
                                                                                               args.truth)))):
                print(line)
    except (ConfigError, ConventionsError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


def _default_data_dir():
    import games
    return games.DATA_DIR


def _read(path, command):
    """The JSON document at `path`, or an OSError that says which command writes it."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise OSError("conventions: cannot read {} ({}); write it with `python pipeline/benchmark/conventions.py {}`"
                      .format(path, error.strerror, command)) from error


if __name__ == "__main__":
    sys.exit(main())
