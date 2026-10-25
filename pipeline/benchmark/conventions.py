"""conventions: when does the truth's label change? Every change of it, timed against events our tracking can see.

    python pipeline/benchmark/conventions.py public  [--gt PATH] [--data-dir D]
    python pipeline/benchmark/conventions.py dfl     [--profile tracking] [--seeds 3] [--truth raw|coarse]
                                                     [--matches ID ...] [--data-dir D]
    python pipeline/benchmark/conventions.py compare [--profile tracking] [--truth raw] [--data-dir D]
    python pipeline/benchmark/conventions.py latency CONFIG public [--gt PATH] [--data-dir D]
    python pipeline/benchmark/conventions.py latency CONFIG dfl    [--profile tracking] [--seeds 3] [--truth raw|coarse]
                                                                   [--matches ID ...] [--data-dir D]
    python pipeline/benchmark/conventions.py shift --config CONFIG [--s-b-rejected TYPE ...] [--gt PATH] [--data-dir D]

    trans = transitions_of(codes, runs)      # [(frame, from, to, type)], type in TYPES
    found = events(inputs)                   # {detector: {"frames": int array, "team": int8 array or None}}
    table = lags(trans, found, runs, fps)    # {(type, detector): {"n", "found", "lags"}}
    print(summarise(table))                  # n, share found, median, p25, p75 in frames and in seconds

    lag = latencies(gt_codes, pred_codes, runs)         # {type: [predicted - true, in frames]}, matched within 50
    rule = shift_rule(lag_public, lag_dfl)              # {type: {"s_b", "g", "shift", ...}}, P02's declared rule
    oof = oof_codes(pred_codes, gt_codes, runs, rule)   # the public window's prediction shifted out of fold

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
                  rule as P04 measured it, which main.toml had at R010 (radius 2.0 m, max_rel_speed 6.0 m/s,
                  min_frames 3), with confidence 0.99 and hold 0.5 so that `home` is a number exactly on the kept
                  frames. It carries the episode's team, and only an event of the new team can be the start of the
                  new team's possession. Where the label turns at the first touch, the lag is 0.
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
    model_dead    an `into_dead` change: where `dead_ball_model`'s alive (P05's file, as of R010) crosses 0.5
    model_alive   downward, and an `out_of_dead` one: where it crosses 0.5 upward. The model has its own bias (a
                  window of features), and DFL's lag for it, which it is out of fold for, is the reference.
    tag_start     an `into_dead` change: the start of a Smart Tagging interval whose code, upper-cased, is `SET PIECES`;
    tag_end       an `out_of_dead` one: the end of such an interval. BAR-ATH only: a DFL match has no tagging, its
                  `smart_tagging` raises AssetUnavailable, and the detector is left out (skipped, not empty). Which set
                  piece it is stays with a person; Smart Tagging's tag values are match data, which no LLM may read.

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

The public window is inspected here, never tuned on (the organisers' rule of 23 Oct 2026, upstream f8f2629, RULES.md
rule 2: the 5 published minutes are for checking that the pipeline is correct and are "not training data and not a
tuning set"). So `public`, and the public side of `compare` and `latency`, are for understanding how our truth flips and
for correctness checks. They set no constant, and nothing a person sees on the video may become a number in a config.
The half of the shift rule below that was read from the public window, g, is RETIRED: it must stay 0, the code still
computes and prints it for reference only, and the shift to adopt is s_b alone, from the DFL matches (other games). Any
future shift comes from other games only (docs/proposals.md P16).

The pipeline's latency and the shift rule (docs/proposals.md P02, Phase C). The detectors say when the tracking shows
a change; what costs score is when OUR labels change, so the same instrument is turned on the pipeline's own output.
`latencies` matches the predicted changes to the truth's with pm.match_transitions at tol = 50 frames (2 s), typed by
the pair of labels (a predicted H to D does not answer a true A to D), inside the runs and on the abstention-filled
prediction (the scorer's own view of it), and gives pred - true in frames, listed under the truth's type in the order
of the true changes. A positive number is a label that changed late. `latency CONFIG public` reads it from the CSV that
`pipeline/run.py CONFIG` wrote, against the public GT; `latency CONFIG dfl` runs `run.predict` on every benchmark match
and seed (the config must decode per half) against the truth, pooled by lags as everything above is.

The latency files are tied to the config they were measured for. Each records the sha256 of what the config runs
(`config_digest`: scope, seed, evidence, decoder and refine; no comment, no description), and `shift` refuses a file of
another content, or of none (a file written before they carried one is stale by rule), with the `latency` command to
run again: a config that has gained a refiner since is never read against a latency measured without it (main.toml
gained a shift_edges after R011's latency was measured). A public file also records the sha256 of the CSV it was read
from, which `shift` checks when that file is there. The digest does not read the files that a parameter names, so a
retrained model JSON needs the latency measured again by hand.

The shift rule is P02's declared one and nothing in it is tuned. For each type, s_b = -round(median of DFL's latencies)
(0 below MIN_DFL = 30 matched changes) is the pipeline's own lag, measured where the truth is DFL's and so owes nothing
to our convention; the benchmark then confirms or rejects it (`--s-b-rejected`), and the DFL latencies it is taken from
are those of the `tracking` profile under the raw truth, the benchmark's own default (a file under the coarse truth is
only reported beside it, as a warning if it disagrees: T33). What the public window shows beyond that is the convention
gap: L' = L_public + s_b, and g = -round(median L') only if |median L'| >= 3 frames, the 90% bootstrap interval of that
median excludes 0 (2,000 draws, default_rng([2, 0]), percentiles 5 and 95) and every leave-one-out median has the sign
of the full one, else 0; the shift is s_b + g. A single public change has no leave-one-out median (a median of nothing
has no sign), so it is not called stable and never sets a shift. A rounding is numpy's (half to even). g is retired (see
above): R012 found it 0 for every type, it is reported for reference only, and no g other than 0 may be adopted.

Out of fold (`oof_codes`). g was read from the public window (it is now retired), so the shifted prediction's public
score is read, as a correctness check only, both in sample (every boundary moved by shift) and out of fold: a
predicted boundary matched to public change i moves by s_b + g without i (the rule applied again to the other
changes), and one that matched nothing by shift. The moves are made by `shift_edges.apply`, so an X edge never
moves. (The refiner also holds a boundary at a kick-off frame; the public window's only kick-off is its first frame, a
block's edge and no boundary, so there is nothing to hold here.) The score of either is a check and never a choice:
adoption reads neither.

What a printout of BAR-ATH latencies may show is what `MIN_SHOWN` says above, by the same three functions:
`latency CONFIG public` goes through `shown` (a type with fewer than 5 changes matched prints n, the share and the
median, no quartile) and the shift table through `limited` and `interval_text` (the median of L', and of its interval
only whether it excludes 0, below 5). DFL's cells print in full. The JSON files keep every value.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Mapping

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numbers  # noqa: E402

import numpy as np  # noqa: E402

from pipeline import output  # noqa: E402
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run, train_dead_ball  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.inputs import AssetUnavailable  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, dead_ball_model, kinematics, play_features  # noqa: E402
from pipeline.refine import shift_edges  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
TYPES = shift_edges.TYPES              # the one list of the kinds of change: the refiner's parameters are named by it
DETECTORS = ("touch_start", "line_out", "ball_lost", "ball_moves", "model_dead", "model_alive", "tag_start", "tag_end")
APPLIES = {"touch_start": ("team", "out_of_dead"), "line_out": ("into_dead",), "ball_lost": ("into_dead",),
           "ball_moves": ("out_of_dead",), "model_dead": ("into_dead",), "model_alive": ("out_of_dead",),
           "tag_start": ("into_dead",), "tag_end": ("out_of_dead",)}
WINDOW_S = 5                        # an event is looked for this far either side of a change
TOUCH = {"radius": 2.0, "max_rel_speed": 6.0, "min_frames": 3, "confidence": 0.99, "hold": 0.5}   # P04's rule, R010
MODEL = "pipeline/models/dead_ball_hgb.json"        # P05's file as of R010, relative to the repository root
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
TOL = 50                            # frames (2 s): pm.match_transitions' tolerance for the pipeline's latency
MIN_DFL = 30                        # s_b is 0 where DFL has fewer matched changes of the type than this
MIN_GAP = 3                         # frames: a median of L' smaller in size than this is not a convention gap to act on
GAP_SEED = (2, 0)                   # default_rng(GAP_SEED), the design's, for the interval of the median of L'
SHIFT_DFL = ("tracking", "raw")     # the profile and truth that s_b is read from: the benchmark's default
SHIFT_COARSE = ("tracking", "coarse")   # reported beside it when its latency file is there, never used (T33)


def _typed_changes():
    """{(from, to): type} for the changes among H, A and D, read off the refiner's own table, so that the kinds of
    change are defined once, in shift_edges (a change into or out of X has no type and is left out)."""
    kinds = {}
    for before in (pm.H, pm.A, pm.D):
        for after in (pm.H, pm.A, pm.D):
            kind = shift_edges.KINDS[shift_edges.KIND_OF_CHANGE[before, after]]
            if kind != shift_edges.ABSTAIN:
                kinds[(before, after)] = kind
    return kinds


_TRANSITION_TYPE = _typed_changes()


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

def _plan(profile, seeds, truth, match_ids, matches, calibration, data_dir):
    """(ids, seed list, calibration) for a run over the benchmark matches, or the runner's BenchmarkError.

    Everything is checked before a match is loaded: the profile, the number of seeds, the truth and the match ids. The
    calibration is read (calibration_module.load) only for a profile that degrades, and only if none was given.
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
    return ids, ([0] if profile == run.CLEAN else list(range(seeds))), calibration


def _truth_of(match, truth):
    """(codes, blocks): the match's truth, as it is or with short possessions absorbed, and its graded halves."""
    codes, _ = truths.truth_codes(match)
    if truth == "coarse":
        codes = truths.coarse_codes(codes, match.graded, match.fps)
    return codes, [(int(start), int(stop)) for _, start, stop in match.graded]


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
    ids, seed_list, calibration = _plan(profile, seeds, truth, match_ids, matches, calibration, data_dir)

    per_match, rates = {}, []
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        rates.append(match.fps)
        codes, blocks = _truth_of(match, truth)          # read here, to find the changes; never handed to the events
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


# -- the pipeline's latency -------------------------------------------------------------------------------------------

def match_changes(gt_codes, pred_codes, runs, tol=TOL):
    """(changes, matches): the truth's typed changes, and those that a predicted change answers.

    `changes` is transitions_of(gt_codes, runs) (frame, from, to, type). `matches` is [(type, true, predicted)] for the
    pairs pm.match_transitions makes at +-tol frames, typed "pair" (both labels agree), inside `runs`, with the type of
    the true change and each side a pm transition (frame, from, to), in the order of the true changes. The prediction is
    the scorer's view of it: its X filled forward inside each run (pm.fill_abstentions), so a change is where the new
    label shows. ConventionsError for codes of different lengths, a tol that is not a whole number of frames of at
    least 0, and a truth with an X inside a run (transitions_of).
    """
    tol = _whole(tol, "tol", 0)
    gt_codes, pred_codes = np.asarray(gt_codes), np.asarray(pred_codes)
    if gt_codes.ndim != 1 or gt_codes.shape != pred_codes.shape:
        raise ConventionsError("conventions: the truth has {} frames and the prediction {}; they must be one code per "
                               "frame of the same match, of the same length".format(gt_codes.size, pred_codes.size))
    runs = sorted((int(start), int(stop)) for start, stop in runs)
    changes = transitions_of(gt_codes, runs)
    predicted = pm.transitions(pm.fill_abstentions(pred_codes, runs), runs)
    pairs, _, _ = pm.match_transitions([change[:3] for change in changes], predicted, tol, typed="pair")
    matches = [(_TRANSITION_TYPE[(true[1], true[2])], true, found) for true, found in pairs]
    matches.sort(key=lambda match: match[1][0])
    return changes, matches


def latency_cells(gt_codes, pred_codes, runs, tol=TOL):
    """{type: {"n", "found", "lags"}} of match_changes, a cell for each of TYPES (also one with no change: n is 0).

    n is the number of the truth's changes of the type, found those that a predicted change answered, and lags their
    `predicted - true` in frames (ints), in the order of the true changes.
    """
    changes, matches = match_changes(gt_codes, pred_codes, runs, tol)
    cells = {kind: {"n": sum(1 for change in changes if change[3] == kind), "found": 0, "lags": []} for kind in TYPES}
    for kind, true, found in matches:
        cells[kind]["found"] += 1
        cells[kind]["lags"].append(found[0] - true[0])
    return cells


def latencies(gt_codes, pred_codes, runs, tol=TOL):
    """{type: [predicted - true, in frames]} for each of TYPES: how late the pipeline's changes are, by the true type.

    A positive number is a label that changed after the truth's. Only changes that were matched (match_changes) have a
    latency; the lists are in the order of the true changes. See latency_cells for the counts beside them.
    """
    return {kind: cell["lags"] for kind, cell in latency_cells(gt_codes, pred_codes, runs, tol).items()}


def _whole(value, name, minimum):
    """`value` as an int, or a ConventionsError that names it: a whole number (bool refused) of at least `minimum`."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ConventionsError("conventions: {} must be a whole number of at least {}, got {!r}".format(
            name, minimum, value))
    return int(value)


def latency_path(data_dir, stem, source, profile=None, truth=None, match_ids=None, seeds=None):
    """<data dir>/runs/conventions/latency_<stem>_public.json, or latency_<stem>_dfl_<profile>_<truth>.json for DFL.

    The DFL name is suffixed (.m-<ids>, .s<n>) as dfl_path's and the runner's reports are, so that a subset of the
    matches or a one-seed look never replaces the full file that `shift` reads.
    """
    if source == "public":
        parts = ["latency_{}_public".format(stem)]
    else:
        parts = ["latency_{}_dfl_{}_{}".format(stem, profile, truth)] + run.name_suffixes(profile, match_ids, seeds)
    return os.path.join(data_dir, *FOLDER, ".".join(parts) + ".json")


def shift_path(data_dir, stem):
    """<data dir>/runs/conventions/shift_<stem>.json: the shift rule, with every value it was worked out from."""
    return os.path.join(data_dir, *FOLDER, "shift_{}.json".format(stem))


def config_digest(config):
    """The sha256 (hex) of what `config` runs: its scope, seed, evidence, decoder and refine, and nothing else.

    The parsed settings are serialised with sorted keys, so a comment, the description, the file's name and the order
    of its keys change nothing, and any parameter, weight, scope, seed or refiner does. A latency file records it
    because a latency measured for one content says nothing of another (main.toml gained a shift_edges after R011's
    was measured, and a shift read against it would have been applied twice). It does not read the files that a
    parameter names: a retrained model JSON leaves it as it was.
    """
    content = {"scope": config.scope, "seed": config.seed,
               "evidence": [dataclasses.asdict(component) for component in config.evidence],
               "decoder": dataclasses.asdict(config.decoder),
               "refine": [dataclasses.asdict(component) for component in config.refine]}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path):
    """The sha256 (hex) of the bytes of the file at `path`."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _read_labels(path):
    """The frame,label table at `path`, or a ConventionsError that names the file and not what is in it."""
    try:
        return pm.read_label_csv(path)
    except Exception as error:      # unreadable, empty, wrong columns: the scorer's own catch-all
        raise ConventionsError("conventions: cannot read {} as a frame,label CSV ({})".format(
            os.path.basename(path), type(error).__name__)) from error


def public_pair(config, gt_path):
    """(gt_codes, pred_codes, runs): the public truth, what `pipeline/run.py CONFIG` wrote for it, and its blocks.

    The prediction is evaluation/results/<stem>.csv (output.default_out_path). OSError that says which command writes
    it if it is not there; ConventionsError for a GT or a CSV that is not one row per frame of the same match, and for
    a GT with no graded frame. The runs are pm.valid_runs of the GT: the public window.
    """
    csv_path = output.default_out_path(config.stem)
    if not os.path.isfile(csv_path):
        raise OSError("conventions: {} does not exist; write it with `python pipeline/run.py {}` (the config on the "
                      "whole match), then run this again".format(run.shown(csv_path), run.shown(config.path)))
    gt = _read_labels(gt_path)
    reason = pm.validate_submission(gt, len(gt))
    if reason is not None:
        raise ConventionsError("conventions: the GT {} is not a frame,label file ({}); use the public GT, {}".format(
            os.path.basename(gt_path), reason, GT_PUBLIC))
    labels = _read_labels(csv_path)
    reason = pm.validate_submission(labels, len(gt))
    if reason is not None:
        raise ConventionsError("conventions: {} is not a submission of the GT's {} frames ({}); write it again with "
                               "`python pipeline/run.py {}`".format(os.path.basename(csv_path), len(gt), reason,
                                                                    run.shown(config.path)))
    gt_codes = pm.label_codes(gt)
    runs = pm.valid_runs(gt_codes)
    if not runs:
        raise ConventionsError("conventions: the GT {} has no graded frame (every label is X), so there is no change "
                               "of the truth to time".format(os.path.basename(gt_path)))
    return gt_codes, pm.label_codes(labels), runs


def latency_public(config, gt_path, data_dir, tol=TOL, log=print):
    """The pipeline's latency on the public window; writes latency_<stem>_public.json and logs the summary only.

    `config` is a pipeline.config.Config whose CSV `pipeline/run.py` wrote (public_pair), `gt_path` the public GT. The
    document is JSON-ready and holds every lag; what is logged is a line per type with its counts and median, and
    quartiles only for a type with MIN_SHOWN changes matched or more (render_latency, shown), never a frame. The
    document also records config_digest(config) and the sha256 of the CSV it was read from (`shift` checks both).
    """
    tol = _whole(tol, "tol", 0)
    gt_codes, pred_codes, runs = public_pair(config, gt_path)
    cells = latency_cells(gt_codes, pred_codes, runs, tol)
    csv_path = output.default_out_path(config.stem)
    document = {"source": "public", "config": config.stem, "config_sha256": config_digest(config),
                "gt": os.path.basename(gt_path), "csv": os.path.basename(csv_path),
                "csv_sha256": _file_sha256(csv_path), "fps": pm.FPS, "tol": tol, "runs": len(runs), "cells": cells,
                "summary": summarise(cells, pm.FPS)}
    path = latency_path(data_dir, config.stem, "public")
    run.write_json(path, document)
    for line in render_latency(document):
        log(line)
    log("wrote     {}".format(path))
    return document


def measure_latency(config, *, profile=DEFAULT_PROFILE, seeds=None, truth="raw", match_ids=None, matches=None,
                    calibration=None, tol=TOL, data_dir=None, log=print):
    """The pipeline's latency on the benchmark matches, pooled over matches and seeds, and per match.

    Each match is loaded (dfl.load_match, or taken from `matches`, {id: DflMatch}), its truth read beside the inputs
    (raw, or coarse), and for every seed the config is run on the match as the pipeline would see it
    (train_dead_ball.served_inputs: the degraded tracking and no flag) through run.predict, its codes matched to the
    truth. Returns {"config", "config_sha256", "profile", "truth", "seeds", "matches", "fps", "tol", "cells",
    "per_match"}: `config_sha256` is config_digest(config), `cells` the pooled {type: {"n", "found", "lags"}} and
    `per_match` {id: cells}. The runner's BenchmarkError for a config that decodes across half time, a profile, seeds,
    truth or match it refuses (before a match is loaded), or an asset a benchmark match does not have.
    """
    run.check_scope(config)
    tol = _whole(tol, "tol", 0)
    ids, seed_list, calibration = _plan(profile, seeds, truth, match_ids, matches, calibration, data_dir)
    per_match, rates = {}, []
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        rates.append(match.fps)
        codes, blocks = _truth_of(match, truth)
        tables = []
        for seed in seed_list:
            inputs = train_dead_ball.served_inputs(match, profile, seed, calibration, data_dir)
            tables.append(latency_cells(codes, run._codes(config, inputs), blocks, tol))
            del inputs
        per_match[match_id] = pool(tables)
        log("match     {}   {} changes, {} seeds".format(
            match_id, sum(cell["n"] for cell in per_match[match_id].values()) // len(seed_list), len(seed_list)))
        del match, codes
    if any(rate != rates[0] for rate in rates):
        raise run.BenchmarkError("the matches have different frame rates ({}); lags in frames cannot be pooled".format(
            ", ".join(str(rate) for rate in rates)))
    return {"config": config.stem, "config_sha256": config_digest(config), "profile": profile, "truth": truth,
            "seeds": seed_list, "matches": sorted(ids), "fps": rates[0], "tol": tol,
            "cells": pool([per_match[match_id] for match_id in ids]), "per_match": per_match}


def latency_dfl_document(result):
    """The JSON-ready document of measure_latency's result: pooled cells with their lags, and summaries."""
    fps = result["fps"]
    return {"source": "dfl", "config": result["config"], "config_sha256": result["config_sha256"],
            "profile": result["profile"], "truth": result["truth"], "seeds": result["seeds"],
            "matches": result["matches"], "fps": fps, "tol": result["tol"],
            "cells": result["cells"], "summary": summarise(result["cells"], fps),
            "per_match": {match_id: summarise(cells, fps) for match_id, cells in sorted(result["per_match"].items())}}


def render_latency(document):
    """The lines of a latency document: a row per type, then (DFL) each match's median.

    The public window's rows go through `shown`, so a type with fewer than MIN_SHOWN changes matched has no quartile
    printed ("-"), only its counts and median; DFL's are not limited. A document from latency_public or
    latency_dfl_document (or the same keys).
    """
    fps, summary = document["fps"], document["summary"]
    limit = document["source"] == "public"
    if limit:
        lines = ["latency   {}: {} changes of the public truth, the pipeline's matched within {} frames".format(
            document["config"], sum(row["n"] for row in summary.values()), document["tol"])]
    else:
        lines = ["latency   {}: pooled over {} matches and {} seeds, matched within {} frames".format(
            document["config"], len(document["matches"]), len(document["seeds"]), document["tol"])]
    lines.append("{:<10}{:<13}{:>5}{:>9}{:>{w}}{:>{w}}{:>{w}}{:>10}".format(
        "", "type", "n", "matched", "median", "p25", "p75", "median s", w=LAG_WIDTH))
    for kind in TYPES:
        row = summary.get(kind)
        if row is None:
            continue
        if limit:
            row = shown(row)
        quartiles = ["-" if row.get("limited") else _frames(row[key]) for key in ("p25", "p75")]
        lines.append("{:<10}{:<13}{:>5}{:>9}{:>{w}}{:>{w}}{:>{w}}{:>10}".format(
            "", kind, row["n"], _pct(row["share_found"]), _frames(row["median"]), *quartiles,
            "n/a" if row["median_s"] is None else "{:+.2f}".format(row["median_s"]), w=LAG_WIDTH))
    lines.append("{:<10}lags in frames ({} per second): predicted - true, positive = our label changed late; matched = "
                 "share of the true changes with a predicted one of the same pair of labels in {} frames".format(
                     "", fps, document["tol"]))
    if limit:
        lines.append("{:<10}-: fewer than {} changes matched, so only the median is printed (a quartile would be a "
                     "lag)".format("", MIN_SHOWN))
    else:
        ids = sorted(document["per_match"])
        lines.append("{:<10}{:<13}".format("per match", "type") + "".join("{:>9}".format(match_id) for match_id in ids)
                     + "   (median lag in frames)")
        for kind in TYPES:
            lines.append("{:<10}{:<13}".format("", kind) + "".join(
                "{:>9}".format(_frames(document["per_match"][match_id][kind]["median"])) for match_id in ids))
        lines.append("{:<10}profile {}, truth {}, seeds {}".format("", document["profile"], document["truth"],
                                                                   document["seeds"]))
    return lines


# -- the shift rule ---------------------------------------------------------------------------------------------------

def pipeline_lag(lags):
    """s_b: -round(median of `lags`), DFL's latencies of one type, or 0 below MIN_DFL matched changes (numpy rounds)."""
    return -int(np.round(np.median(lags))) if len(lags) >= MIN_DFL else 0


def convention_gap(shifted):
    """What the public latencies of one type, after s_b (L'), say about the convention: the design's three conditions.

    Returns {"median", "low", "high", "loo_signs", "conditions", "g"}. `median` is that of L', `low` and `high` the 5th
    and 95th percentiles of the medians of GAP_DRAWS resamples with replacement (a generator seeded afresh with GAP_SEED
    on every call, so a result depends on its list alone), `loo_signs` the sign of each leave-one-out median in the
    order of the list (none for fewer than two changes), and `conditions` {"size": |median| >= MIN_GAP, "interval": the
    interval excludes 0, "stable": there are leave-one-out medians and each has the sign of the median}. g is
    -round(median) (numpy, half to even) if all three hold, else 0. No lags: all None, False and 0.
    """
    lags = np.asarray(list(shifted), dtype=np.float64)
    count = len(lags)
    if count == 0:
        return {"median": None, "low": None, "high": None, "loo_signs": [],
                "conditions": {"size": False, "interval": False, "stable": False}, "g": 0}
    median = float(np.median(lags))
    rng = np.random.default_rng(list(GAP_SEED))
    medians = np.median(lags[rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))], axis=1)
    low, high = (float(value) for value in np.percentile(medians, PERCENTILES))
    signs = [int(np.sign(np.median(np.delete(lags, index)))) for index in range(count)] if count > 1 else []
    conditions = {"size": bool(abs(median) >= MIN_GAP), "interval": bool(low > 0 or high < 0),
                  "stable": bool(signs) and all(sign == int(np.sign(median)) for sign in signs)}
    return {"median": median, "low": low, "high": high, "loo_signs": signs, "conditions": conditions,
            "g": -int(np.round(median)) if all(conditions.values()) else 0}


def _lags_by_type(table, side):
    """{type: [int]} for each of TYPES from a mapping of type to lags (none if left out), or a ConventionsError."""
    if not isinstance(table, Mapping):
        raise ConventionsError("conventions: the {} latencies must be a mapping of type to lags, got {}".format(
            side, type(table).__name__))
    unknown = [kind for kind in table if kind not in TYPES]
    if unknown:
        raise ConventionsError("conventions: the {} latencies name a type {!r}, and the types are {}".format(
            side, unknown[0], ", ".join(TYPES)))
    out = {}
    for kind in TYPES:
        values = list(table.get(kind, []))
        for lag in values:
            if isinstance(lag, bool) or not isinstance(lag, numbers.Integral):
                raise ConventionsError("conventions: a {} latency of {} is {!r}; latencies are whole numbers of "
                                       "frames".format(side, kind, lag))
        out[kind] = [int(lag) for lag in values]
    return out


def _kept(s_b_kept):
    """{type: int} from the override of s_b ({} for None), or a ConventionsError naming the type or value."""
    if s_b_kept is None:
        return {}
    if not isinstance(s_b_kept, Mapping):
        raise ConventionsError("conventions: s_b_kept must be a mapping of type to frames, got {}".format(
            type(s_b_kept).__name__))
    for kind, value in s_b_kept.items():
        if kind not in TYPES:
            raise ConventionsError("conventions: s_b_kept names a type {!r}, and the types are {}".format(
                kind, ", ".join(TYPES)))
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ConventionsError("conventions: s_b_kept for {} must be a whole number of frames, got {!r}".format(
                kind, value))
    return {kind: int(value) for kind, value in s_b_kept.items()}


def shift_rule(public, dfl, s_b_kept=None):
    """{type: row} for each of TYPES: P02's declared rule from the public and DFL latencies (see the module docstring).

    `public` and `dfl` map a type to its latencies (`latencies`' result, or the lags of a latency file); `s_b_kept`
    maps a type to the s_b to use in place of DFL's (0 where the benchmark rejected it). A row holds `s_b` (the one
    used), `s_b_dfl` (the one DFL gave), `n_dfl` and `dfl_median`, `n_public`, then convention_gap's median, low, high,
    loo_signs and conditions for L' = public + s_b, `g`, and `shift` = s_b + g. JSON-ready. ConventionsError for a
    latency that is not a whole number, a type that is not one of TYPES, or an override that is not a whole number.
    """
    public, dfl, kept = _lags_by_type(public, "public"), _lags_by_type(dfl, "dfl"), _kept(s_b_kept)
    rule = {}
    for kind in TYPES:
        s_b_dfl = pipeline_lag(dfl[kind])
        s_b = kept.get(kind, s_b_dfl)
        gap = convention_gap([lag + s_b for lag in public[kind]])
        rule[kind] = {"s_b": s_b, "s_b_dfl": s_b_dfl, "n_dfl": len(dfl[kind]),
                      "dfl_median": float(np.median(dfl[kind])) if dfl[kind] else None,
                      "n_public": len(public[kind]), "median": gap["median"], "low": gap["low"], "high": gap["high"],
                      "loo_signs": gap["loo_signs"], "conditions": gap["conditions"], "g": gap["g"],
                      "shift": s_b + gap["g"]}
    return rule


def _checked_rule(rule):
    """`rule` itself, or a ConventionsError: it must be shift_rule's, a row with whole s_b and shift for each type."""
    for kind in TYPES:
        row = rule.get(kind) if isinstance(rule, Mapping) else None
        if not isinstance(row, Mapping) or any(
                isinstance(row.get(key), bool) or not isinstance(row.get(key), numbers.Integral)
                for key in ("s_b", "shift")):
            raise ConventionsError("conventions: oof_codes needs the shift rule that shift_rule returns, a row with a "
                                   "whole s_b and shift for each of {}; {} has none".format(", ".join(TYPES), kind))
    return rule


def oof_codes(pred_codes, gt_codes, runs, rule, tol=TOL):
    """A new int8 array: `pred_codes` with each boundary inside `runs` moved by the rule out of fold (T27).

    Equal to `pred_codes` outside the runs. Inside them a boundary that answers public change i (match_changes at `tol`)
    moves by s_b + g without i: the rule's s_b of the type, and g worked out again by convention_gap from the other
    matched changes' latencies after s_b. A boundary that answers none moves by the rule's `shift` of its type, and a
    boundary beside an X (shift_edges' "x") not at all. The moves are shift_edges.apply's, a positive one later. `rule`
    is shift_rule's, worked out from the latencies of this very prediction (a rule whose shift is not its s_b plus the g
    that those latencies give is refused, so that a stale rule is never applied); ConventionsError for that and for
    anything else that is not a rule, and the errors of match_changes. The arrays handed in are only read.
    """
    rule = _checked_rule(rule)
    pred_codes = np.asarray(pred_codes)
    _, matches = match_changes(gt_codes, pred_codes, runs, tol)
    runs = sorted((int(start), int(stop)) for start, stop in runs)
    moves = {}                                  # a predicted change (frame, from, to) -> frames it moves by
    for kind in TYPES:
        own = [(true, found) for match_kind, true, found in matches if match_kind == kind]
        s_b = int(rule[kind]["s_b"])
        shifted = [found[0] - true[0] + s_b for true, found in own]
        if int(rule[kind]["shift"]) != s_b + convention_gap(shifted)["g"]:
            raise ConventionsError("conventions: the rule's shift for {} is {:+d}, which is not its s_b plus the g "
                                   "that this prediction's own public latencies give ({:+d}); work the rule out from "
                                   "the same CSV and GT with shift_rule(latencies(...), ...)".format(
                                       kind, int(rule[kind]["shift"]), s_b + convention_gap(shifted)["g"]))
        for position, (_, found) in enumerate(own):
            moves[found] = s_b + convention_gap(shifted[:position] + shifted[position + 1:])["g"]
    out = pred_codes.astype(np.int8)
    for start, stop in runs:
        window = pred_codes[start:stop]
        positions, kinds = shift_edges.boundaries(window)
        offsets = np.zeros(len(positions), dtype=np.int64)
        fixed = np.zeros(len(positions), dtype=bool)
        for index, (position, kind) in enumerate(zip(positions.tolist(), kinds)):
            if kind == shift_edges.ABSTAIN:
                fixed[index] = True
            else:
                change = (start + position, int(window[position - 1]), int(window[position]))
                offsets[index] = moves.get(change, int(rule[kind]["shift"]))
        out[start:stop] = shift_edges.apply(window, offsets, fixed)
    return out


def render_shift(rule, coarse_s_b=None):
    """The shift table: a row per type, then what each column is. The public side is limited as BAR-ATH's printouts are.

    Per type: DFL's matched changes and their median latency, the s_b used, the public changes matched, the median of
    L' and its interval (`interval_text`: the bounds from MIN_SHOWN changes, below that only whether it excludes 0), the
    three conditions (size, interval, leave-one-out) as yes or no, g and the shift. `coarse_s_b` ({type: s_b}) adds one
    line for the coarse truth, which is reported beside the raw one and never used.
    """
    lines = ["{:<10}{:<13}{:>7}{:>11}{:>6}{:>10}{:>11}  {:<26}{:>13}{:>6}{:>7}".format(
        "shift", "type", "dfl n", "dfl median", "s_b", "public n", "median L'", "90% interval of it", "conditions",
        "g", "shift")]
    for kind in TYPES:
        row = rule[kind]
        interval = interval_text(row["low"], row["high"], {"found": row["n_public"]})
        held = "/".join("yes" if row["conditions"][name] else "no" for name in ("size", "interval", "stable"))
        lines.append("{:<10}{:<13}{:>7}{:>11}{:>6}{:>10}{:>11}  {:<26}{:>13}{:>6}{:>7}".format(
            "", kind, row["n_dfl"], _frames(row["dfl_median"]), "{:+d}".format(row["s_b"]), row["n_public"],
            _frames(row["median"]), interval, held, "{:+d}".format(row["g"]), "{:+d}".format(row["shift"])))
    for kind in TYPES:
        if rule[kind]["s_b"] != rule[kind]["s_b_dfl"]:
            lines.append("{:<10}s_b of {}: {:+d} in place of DFL's {:+d} (kept by the override: the benchmark rejected "
                         "it)".format("", kind, rule[kind]["s_b"], rule[kind]["s_b_dfl"]))
    if coarse_s_b is not None:
        lines.append("{:<10}coarse truth, beside the raw one and never used: s_b {}".format(
            "", ", ".join("{} {:+d}".format(kind, coarse_s_b[kind]) for kind in TYPES)))
    lines.append("{:<10}dfl n, dfl median: DFL's matched changes of the type, and their median latency in frames "
                 "(predicted - true); s_b = -round(median), 0 below {}".format("", MIN_DFL))
    lines.append("{:<10}median L' = the public latencies' median after s_b, and its 90% bootstrap interval ({} draws); "
                 "conditions: |median| >= {}, the interval excludes 0, every leave-one-out median has its sign; g = "
                 "-round(median) if all three hold, else 0; shift = s_b + g".format("", BOOTSTRAP_DRAWS, MIN_GAP))
    lines.append("{:<10}with fewer than {} public changes matched the bounds would be lags, so only whether the "
                 "interval excludes 0 is printed".format("", MIN_SHOWN))
    return lines


def _check_recorded(document, key, expected, path, command, because):
    """Nothing if `document` records `expected` under `key`, else a ConventionsError naming the command to run again.

    A document that records none (a file written before the latency files carried it) is refused as one of another
    content is. Only the first 12 characters of a hash are named, so that the two can be told apart.
    """
    recorded = document.get(key)
    if recorded == expected:
        return
    if not isinstance(recorded, str) or not recorded:
        said = "records no {}, so it cannot be told to belong to this one".format(key)
    else:
        said = "has {} {} and this one is {}: {}".format(key, recorded[:12], expected[:12], because)
    raise ConventionsError("conventions: {} {}; write it again with `python pipeline/benchmark/conventions.py {}`"
                           .format(os.path.basename(path), said, command))


def _check_latency_file(document, stem, tol, path, command, digest, csv_sha256=None):
    """`document` itself, or a ConventionsError: a latency file of this config's content, at this tolerance, with
    every type (and, for the public side, read from the CSV with this sha256).

    `digest` is config_digest of the config in hand and `csv_sha256` that of the CSV it wrote: a file that records
    another one, or none, is refused with the `latency` command (`command`) that writes it again.
    """
    if document.get("config") != stem:
        raise ConventionsError("conventions: {} was made for the config {!r}, not {!r}; write it again with `python "
                               "pipeline/benchmark/conventions.py {}`".format(os.path.basename(path),
                                                                              document.get("config"), stem, command))
    _check_recorded(document, "config_sha256", digest, path, command,
                    "an evidence, decoder, refine, scope or seed setting is not what it was measured for")
    if csv_sha256 is not None:
        _check_recorded(document, "csv_sha256", csv_sha256, path, command,
                        "the config was run again since, and this is not the CSV it was read from")
    if document.get("tol") != tol:
        raise ConventionsError("conventions: {} matched changes within {} frames and the public side within {}; write "
                               "it again with `python pipeline/benchmark/conventions.py {}`".format(
                                   os.path.basename(path), document.get("tol"), tol, command))
    cells = document.get("cells")
    if not isinstance(cells, Mapping) or any(not isinstance(cells.get(kind, {}).get("lags"), list) for kind in TYPES):
        raise ConventionsError("conventions: {} holds no lags for one of {}; write it again with `python "
                               "pipeline/benchmark/conventions.py {}`".format(os.path.basename(path),
                                                                              ", ".join(TYPES), command))
    return document


def run_shift(config, gt_path, data_dir, rejected=(), tol=TOL, log=print):
    """The shift rule of `config`, its table, shift_<stem>.json, and the out-of-fold prediction's CSV; the document.

    The public side is worked out from the CSV that `pipeline/run.py CONFIG` wrote (public_pair), so it cannot disagree
    with the prediction that is shifted; DFL's is read from latency_<stem>_dfl_tracking_raw.json (SHIFT_DFL), which
    `latency CONFIG dfl` writes, and refused if it is another config's content (config_digest; a file that records
    none is refused too) or at another tolerance. A latency file for the coarse truth (SHIFT_COARSE) is reported
    beside it when there is one and held to the same rule, and so is the public one (`latency CONFIG public`), which
    must also have been read from this very CSV (it is not needed, and when it is absent nothing is checked).
    `rejected` names the types whose s_b the benchmark rejected: it is 0 for them. Writes
    evaluation/results/<stem>_shift_oof.csv through the pipeline's own output stage (checked path, LF, read back and
    validated) and logs the table, no frame and no public value that `limited` would hide. Nothing is written if
    anything is refused.
    """
    tol = _whole(tol, "tol", 0)
    rejected = list(rejected)
    for kind in rejected:
        if kind not in TYPES:
            raise ConventionsError("conventions: no type called {!r}; the types are {}".format(kind, ", ".join(TYPES)))
    gt_codes, pred_codes, runs = public_pair(config, gt_path)
    public = latencies(gt_codes, pred_codes, runs, tol)
    digest, csv_path = config_digest(config), output.default_out_path(config.stem)
    csv_sha256 = _file_sha256(csv_path)

    def command(truth):
        return "latency {} dfl --profile {} --truth {}".format(run.shown(config.path), SHIFT_DFL[0], truth)

    dfl_path = latency_path(data_dir, config.stem, "dfl", *SHIFT_DFL)
    dfl_document = _check_latency_file(_read(dfl_path, command(SHIFT_DFL[1])), config.stem, tol, dfl_path,
                                       command(SHIFT_DFL[1]), digest)
    coarse = None
    coarse_path = latency_path(data_dir, config.stem, "dfl", *SHIFT_COARSE)
    if os.path.isfile(coarse_path):
        coarse_document = _check_latency_file(_read(coarse_path, command(SHIFT_COARSE[1])), config.stem, tol,
                                              coarse_path, command(SHIFT_COARSE[1]), digest)
        coarse = {kind: pipeline_lag(coarse_document["cells"][kind]["lags"]) for kind in TYPES}
    public_path, public_command = latency_path(data_dir, config.stem, "public"), "latency {} public".format(
        run.shown(config.path))
    if os.path.isfile(public_path):
        _check_latency_file(_read(public_path, public_command), config.stem, tol, public_path, public_command, digest,
                            csv_sha256)
    rule = shift_rule(public, {kind: dfl_document["cells"][kind]["lags"] for kind in TYPES},
                      {kind: 0 for kind in rejected})
    codes = oof_codes(pred_codes, gt_codes, runs, rule, tol)

    out_path = output.default_out_path(config.stem + "_shift_oof")
    output.check_out_path(out_path, data_dir)
    sha256 = output.write_submission(out_path, codes)
    output.validate(out_path, len(codes), codes)
    document = {"config": config.stem, "config_sha256": digest, "csv": os.path.basename(csv_path),
                "csv_sha256": csv_sha256, "tol": tol, "rule": rule, "s_b_rejected": rejected, "coarse_s_b": coarse,
                "dfl": {"file": os.path.basename(dfl_path), "profile": dfl_document.get("profile"),
                        "truth": dfl_document.get("truth"), "seeds": dfl_document.get("seeds"),
                        "matches": dfl_document.get("matches")},
                "oof": {"csv": os.path.basename(out_path), "sha256": sha256}}
    path = shift_path(data_dir, config.stem)
    run.write_json(path, document)
    for line in render_shift(rule, coarse):
        log(line)
    log("wrote     {}".format(path))
    log("wrote     {}   (the public window shifted out of fold; valid, sha256 {})".format(run.shown(out_path), sha256))
    return document


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
    latency = commands.add_parser(
        "latency", help="how late the pipeline's changes are: pred - true, matched within {} frames".format(TOL),
        description="The pipeline's latency: its predicted changes matched to the truth's, typed by the pair of "
                    "labels. public reads evaluation/results/<config>.csv (write it with pipeline/run.py CONFIG); "
                    "dfl runs the config on the benchmark matches. Writes latency_<config>_public.json or "
                    "latency_<config>_dfl_<profile>_<truth>.json.")
    latency.add_argument("config", help="the config's TOML file, e.g. pipeline/configs/main.toml")
    sources = latency.add_subparsers(dest="source", required=True, metavar="{public,dfl}")
    for name, text in (("public", "the public window: the config's CSV against the public GT"),
                       ("dfl", "the DFL matches, degraded as the pipeline's tracking is")):
        source = sources.add_parser(name, help=text, description=text)
        source.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        if name == "public":
            source.add_argument("--gt", help="the public GT (default {})".format(GT_PUBLIC))
        else:
            source.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
            source.add_argument("--truth", choices=TRUTHS, default="raw",
                                help="DFL's flags as they are (raw), or with short possessions absorbed (coarse)")
            source.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                                help="degradation seeds per match (default {})".format(DEFAULT_SEEDS))
            source.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run (default: all 7)")
    shift = commands.add_parser(
        "shift", help="P02's declared shift rule (g, read from the public window, is retired and shown for reference "
                      "only): the table, shift_<config>.json and the out-of-fold CSV",
        description="Work out the shift rule from the public latencies of the config's CSV and DFL's latency file "
                    "(latency CONFIG dfl, profile {}, truth {}), print its table, and write shift_<config>.json and "
                    "evaluation/results/<config>_shift_oof.csv, the public window shifted out of fold. g, the gap read "
                    "from the public window, is RETIRED since the organisers' rule of 23 Oct 2026 (that window is not "
                    "a tuning set): it is reported for reference only and must stay 0, and the shift to adopt is s_b "
                    "alone, from DFL.".format(*SHIFT_DFL))
    shift.add_argument("--config", required=True, help="the config's TOML file, e.g. pipeline/configs/main.toml")
    shift.add_argument("--s-b-rejected", nargs="+", choices=TYPES, default=[], metavar="TYPE",
                       help="the types whose s_b the benchmark guard rejected: 0 for them ({})".format(
                           ", ".join(TYPES)))
    shift.add_argument("--gt", help="the public GT (default {})".format(GT_PUBLIC))
    shift.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
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
        elif args.command == "compare":
            path = dfl_path(data_dir, args.profile, args.truth)
            for line in render_compare(compare(_read(os.path.join(data_dir, *FOLDER, PUBLIC_LAGS), "public"),
                                               _read(path, "dfl --profile {} --truth {}".format(args.profile,
                                                                                               args.truth)))):
                print(line)
        elif args.command == "latency":
            config = load_config(args.config)
            if args.source == "public":
                latency_public(config, args.gt or default_gt(), data_dir)
            else:
                result = measure_latency(config, profile=args.profile, seeds=args.seeds, truth=args.truth,
                                         match_ids=args.matches, data_dir=args.data_dir)
                path = latency_path(data_dir, config.stem, "dfl", args.profile, args.truth, args.matches, args.seeds)
                document = latency_dfl_document(result)
                run.write_json(path, document)
                print("")
                for line in render_latency(document):
                    print(line)
                print("wrote     {}".format(path))
        else:
            run_shift(load_config(args.config), args.gt or default_gt(), data_dir, rejected=args.s_b_rejected)
    except (ConfigError, ConventionsError, run.BenchmarkError, output.UnsafeOutPath, output.SubmissionInvalid,
            OSError) as error:
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
