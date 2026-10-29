"""The past-the-line ruler, our match's target for it, and the one-time fit of the offpitch profile (P18, R015).

    stats = line_stats(ball_present, blocks, ball_xy_m, pitch)       # the share of visible ball frames past a line
    visible, past = beyond_distances(ball_present, blocks, ball_xy_m, pitch)    # what line_stats summarises
    stats = line_summary(visible, past)                              # line_stats again, of parts pooled by sums
    spans = kickoff_spans(inputs)                                    # BAR-ATH's spans, as calibration.measure has them
    ours = target(data_dir=None, rebuild=False, inputs=None)         # line_stats of BAR-ATH, cached as JSON
    record = fit(calibration, ours["beyond_line_share"], seeds=3)    # m and q, fitted ONCE, with what fixed them
    margin_m, drop_prob = constants_from(record)                     # the two values degrade.py pins

    python pipeline/benchmark/offpitch.py target [--rebuild] [--data-dir D]     # prints the dict
    python pipeline/benchmark/offpitch.py fit [--seeds 3] [--matches ID ...] [--data-dir D] [--out PATH]

Why it exists. Our tracker sometimes reports the ball outside the pitch: a frame on which the ball has a position that
is past one of the four lines, and a dead-ball detector can read that as "out of play" for nothing but the position.
DFL's tracking reports the ball only while it is in play, so a benchmark match shows the pipeline no such frames. The
`offpitch` profile (degrade.py) therefore hides DFL's ball once it is more than m metres past a line, and m has to be
fitted so that the degraded DFL matches show as many of those frames as ours does. "As many" needs a ruler, and the
ruler has to be one function: line_stats measures BAR-ATH here (target) and the fit of m applies the same function,
unchanged, to the degraded matches, the way calibration.ball_stats is the shared ruler of the gaps. It takes plain
arrays and knows nothing of either match.

Definitions (metres, from a pitch corner; the line distance is calibration's, min(x, L - x, y, W - y), negative beyond a
line)
    visible         the flag says the ball is present and BOTH coordinates are finite (calibration._usable). A flag set
                    where the position is absent is a ValueError: the two come from different trackings.
    beyond          a visible frame whose line distance is below 0, strictly: a ball exactly on a line is on the pitch.
    beyond share    beyond frames / visible frames, over the blocks (NaN when no frame is visible), the number the fit
                    of m matches; beyond_m_quantiles are the quantiles of how far past the line the beyond frames are
                    (-line distance), for the report and for a look at the tail, not for the fit.
beyond_distances gives the two things that line_stats summarises (the visible frames and the distances past the line),
so that parts measured apart, the matches and seeds of the benchmark, pool into one line_stats by sums and by laying the
distances end to end (line_summary): the profile's quantiles in fidelity.py are pooled so.

The target. Ours is measured on the kick-off spans (each half from its kick-off to the end of its graded block), as
calibration.measure's spans are, because the degraded matches start at their kick-off. kickoff_spans repeats measure's
construction, since measure keeps its spans in a local variable and calibration.py is not edited (its hash keys the
Calibration's cache). The result is cached as JSON under <data dir>/cache/benchmark/offpitch/target-<16 hex>.json, the
key being the sha256 of the ATD cache's source record, this file's own sha256 (LF-normalised), numpy's version, the
pitch, the fps and the kick-off spans, so that an edit of this file, a new ATD or a new numpy is a new file and the old
ones are left alone.

The fit (fit, R015). The offpitch profile has two constants: m, the margin past a line beyond which DFL's ball is hidden
until it is back on the pitch, and q, the chance that an interior gap of the transplanted mask is dropped. They
calibrate two label-free statistics, and are fitted ONCE, by this function, then pinned in degrade.py with the record
(OFFPITCH_MARGIN_M, OFFPITCH_DROP_PROB, OFFPITCH_FIT) and frozen, as TRACKED_SCALE and HAZARD_SCALE were: never refitted
because of a gate, a WARN or a score (T37). The rule, fixed by the design before any run:

1. At q = 0, m is the value of M_GRID (0.0 to 10.0 m by 0.1 m) whose share of the visible ball frames past a line is
   closest to ours (target()'s beyond_line_share).
2. At that m, q is the value of Q_GRID (0.000 to 0.500 by 0.005) whose missing share is closest to the Calibration's
   (scalars["missing_share"], on the same kick-off spans).
3. The share past a line is measured at (m, q). If it is more than REFIT_TOLERANCE (0.005) off ours, m is refitted once,
   at that q, by step 1's rule. Nothing more: q is not fitted again at the new m, where the two could go round, each
   undoing the other.

Closest is compared exactly, as a fraction of the integer counts against the target's float, and a tie goes to the
smaller value (the first of the grid). Each share is pooled as the runner pools: the frames are summed over the matches
and the seeds, each half from its kick-off to the end of its graded block (the Calibration's spans), and the share is a
ratio of the sums. The missing share counts a frame that DFL itself has no ball for as missing, as
calibration.ball_stats does on the degraded ball. If no margin lands within the gate's +-0.01 of ours, the closest is
kept and the shortfall is in the record (share_at_fit against target_share), for the pause; the grids are not extended.

Why q is a threshold. The thinning draws one uniform per interior gap (degrade.thinning_uniforms: a stream of its own,
keyed by the seed, the match and the half and nothing else) and drops gap i iff its uniform is below q. For a given
(match, seed, half) the uniforms do not depend on q, so a larger q drops a superset of a smaller one's gaps, and the
sweep over Q_GRID reads the very gaps that the profile drops at each value. The sweeps build every mask from the
profile's own parts (offpitch_masks at an infinite margin and q = 0, which is the transplanted mask, and its count of
interior gaps; thin_mask with thinning_uniforms; offpitch_mask of line_distance_m), so the fit and the profile agree by
construction, and count with array sums. The shares that the record reports at the chosen (m, q) are not the sweep's:
they are the ruler's (calibration.ball_stats and line_stats) on the ball that offpitch_masks and degrade._remove_ball
make, as degrade() makes it, and the fit stops with an error if the two disagree by one frame.

Each match is read once. What the fit keeps of it is a view of its ball (no players) and, per half, the line distance,
the frames with a ball and, per seed, the transplanted mask and its uniforms: a few MB for the seven matches.

The record (FIT_KEYS) holds aggregates only: the two targets (target_share, missing_target), the two constants
(margin_m, drop_prob), the ruler's shares at them (share_at_fit, missing_at_fit) and its counts (counts_at_fit), the
profile's own counts there (draws_at_fit: offpitch_masks' records added up), whether m was refitted (refit), step 1's
margin and the share at it and q that decided the refit (first_margin_m, share_at_first_margin), the sweeps as [value,
share] (sweep_m at q = 0, sweep_q at the first margin, sweep_m_refit at q, or [] without a refit), the seeds, the
matches and the Calibration's digest. No NaN and no infinity: a share with no frame under it is a BenchmarkError, and
the record is written with allow_nan=False. The command adds the key of the target it was fitted to (target_key).
constants_from(record) gives the two values that a record fixes; degrade.py pins them with the record's scalars, and a
check ties them together.

What it assumes: that our out-of-play time is like DFL's, that is, that when the ball leaves the pitch the same share of
the match's visible frames is spent beyond a line in the two matches once DFL's ball is hidden from m metres out.

It holds no frame. The target has scalars and quantiles only (visible_frames, beyond_frames, beyond_line_share,
beyond_m_quantiles, a version and a key), and nothing about BAR-ATH is printed but what that file holds, and its
missing share in the fit's record.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import degrade, dfl, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402

# Params
VERSION = 1                                                    # of the target's definitions: bump on a change of them
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90))   # of the distance beyond
CACHE_FOLDER = ("cache", "benchmark", "offpitch")              # <data dir>/cache/benchmark/offpitch/target-<16 hex>.json
KEY_DIGITS = 16                                                # hex digits of the key's sha256 that name the file
DECIMALS = 9                                                   # a stored float is rounded to 1e-9, as calibration's are
# The fit (R015), as the design fixed it before any run: the grids that m and q are searched over, and when m is
# refitted.
M_GRID = np.round(np.arange(0, 100.5) / 10, 1)                 # the margins, metres: 0.0 to 10.0 by 0.1
Q_GRID = np.round(np.arange(0, 100.5) / 200, 3)                # the drop probabilities: 0.000 to 0.500 by 0.005
M_GRID.setflags(write=False)
Q_GRID.setflags(write=False)
REFIT_TOLERANCE = 0.005    # the share past a line at (m, q) more than this off ours: m is refitted once, at that q
DEFAULT_SEEDS = 3          # degradation seeds per match, as the runner and fidelity.py take them
FIT_KEYS = ("target_share", "missing_target", "margin_m", "drop_prob", "share_at_fit", "missing_at_fit", "refit",
            "first_margin_m", "share_at_first_margin", "sweep_m", "sweep_q", "sweep_m_refit", "counts_at_fit",
            "draws_at_fit", "seeds", "matches", "calibration")


# -- the ruler --------------------------------------------------------------------------------------------------------

def line_stats(ball_present, blocks, ball_xy_m, pitch):
    """The ball's frames past a line inside the blocks, as a dict.

    ball_present  (n,) bool: the tracker sees the ball.
    blocks        [(half, start, stop)], half-open, disjoint, one per half: the stretches to measure.
    ball_xy_m     (n, 2) metres from a pitch corner, NaN where the ball is missing.
    pitch         (length, width) in metres.

    Returns visible_frames, beyond_frames (visible with a line distance below 0, strictly), beyond_line_share
    (beyond / visible, NaN when none is visible) and beyond_m_quantiles, {name: float} of -line distance over the
    beyond frames (QUANTILES; {} when there is none). ValueError, naming the argument, for a flag that is not a (n,)
    bool array, a position that is not (n, 2), a flag set where the position is absent, a bad pitch and bad blocks.
    It is line_summary of beyond_distances.
    """
    return line_summary(*_beyond("offpitch.line_stats", ball_present, blocks, ball_xy_m, pitch))


def beyond_distances(ball_present, blocks, ball_xy_m, pitch):
    """(visible_frames, past): the two things that line_stats summarises, for pooling.

    visible_frames is how many frames inside the blocks are visible, and `past` is the (k,) float64 array, read-only,
    of how far past the line each beyond frame is (-line distance, in metres), in time order. Parts measured apart
    pool into line_stats of them all as line_summary(the sum of their visible_frames, their `past` laid end to end).
    The arguments and the errors are line_stats'."""
    return _beyond("offpitch.beyond_distances", ball_present, blocks, ball_xy_m, pitch)


def line_summary(visible_frames, past):
    """line_stats' dict of `visible_frames` and the distances `past` (beyond_distances'): the share is len(past) over
    the visible frames (NaN when there is none) and the quantiles are those of `past` ({} when it is empty)."""
    past = np.asarray(past, dtype=np.float64)
    visible_frames = int(visible_frames)
    return {
        "visible_frames": visible_frames,
        "beyond_frames": int(past.size),
        "beyond_line_share": past.size / visible_frames if visible_frames else math.nan,
        "beyond_m_quantiles": {name: float(np.quantile(past, level)) for name, level in QUANTILES} if past.size else {},
    }


def _beyond(function, ball_present, blocks, ball_xy_m, pitch):
    """(visible_frames, past) as beyond_distances documents them; `function` names the caller in the errors."""
    flag = _named(function, calibration_module._bool_array, "ball_present", ball_present, 1)
    n = len(flag)
    xy = np.asarray(ball_xy_m, dtype=np.float64)
    if xy.shape != (n, 2):
        raise ValueError("{}: ball_present has {} frames but ball_xy_m has shape {}; both are one value per frame of "
                         "the match, the position in metres".format(function, n, xy.shape))
    length, width = _named(function, calibration_module._pitch, pitch)
    blocks = _named(function, calibration_module._blocks, blocks, n)
    finite = np.isfinite(xy).all(axis=1)
    stray = int((~flag & finite).sum())
    if stray:
        raise ValueError("{}: ball_present and ball_xy_m disagree on {} frames: the ball is marked absent where a "
                         "position is given; build both from the same tracking".format(function, stray))
    visible = flag & finite
    in_blocks = np.zeros(n, dtype=bool)
    for _, start, stop in blocks:
        in_blocks[start:stop] = True
    visible &= in_blocks
    x, y = xy[visible, 0], xy[visible, 1]
    line = np.minimum.reduce([x, length - x, y, width - y])
    past = -line[line < 0.0]
    past.setflags(write=False)
    return int(visible.sum()), past


def _named(function, helper, *args):
    """helper(*args), a ValueError of calibration's re-raised with the caller's name in front of its message."""
    try:
        return helper(*args)
    except ValueError as error:
        raise ValueError("{}: {}".format(function, error)) from None


def kickoff_spans(inputs):
    """[(half, kickoff, stop)]: each graded half from its kick-off to the end of its block (calibration.measure's spans).

    inputs needs .graded and .kickoffs. ValueError for a half without a kick-off, or one that is not inside its graded
    block.
    """
    spans = []
    for half, start, stop in inputs.graded:
        if half not in inputs.kickoffs:
            raise ValueError("offpitch.kickoff_spans: inputs.kickoffs has no kick-off for half {}".format(half))
        if not start <= inputs.kickoffs[half] < stop:
            raise ValueError("offpitch.kickoff_spans: half {}: the kick-off, frame {}, is not inside its graded block "
                             "{}-{}".format(half, inputs.kickoffs[half], start, stop))
        spans.append((half, inputs.kickoffs[half], stop))
    return spans


# -- our match's target -----------------------------------------------------------------------------------------------

def target(data_dir=None, rebuild=False, inputs=None):
    """line_stats of BAR-ATH on its kick-off spans, with "version" and "key"; from the cache when it is fresh.

    data_dir defaults to games.DATA_DIR (MTAH_DATA_DIR) through Inputs; the cache is target_path(data_dir, key).
    rebuild=True measures again. `inputs` stands in for the real match (the self-checks pass a synthetic bundle). The
    dict holds scalars and quantiles only; a NaN share is stored as null and read back as NaN.
    """
    if inputs is None:
        from pipeline.inputs import Inputs  # here, not above: it pulls in pandas and the starter loaders
        inputs = Inputs(data_dir=data_dir)
    data_dir = data_dir or inputs.data_dir
    key = _key(inputs)
    path = target_path(data_dir, key)
    if not rebuild:
        saved = _read(path, key)
        if saved is not None:
            return saved
    stats = line_stats(inputs.atd.ball_present, kickoff_spans(inputs), inputs.metres(inputs.atd.ball_xy), inputs.pitch)
    result = {"version": VERSION, "key": key,
              "visible_frames": stats["visible_frames"], "beyond_frames": stats["beyond_frames"],
              "beyond_line_share": _round(stats["beyond_line_share"]),
              "beyond_m_quantiles": {name: _round(value) for name, value in stats["beyond_m_quantiles"].items()}}
    _write(path, result)
    return _read(path, key) or result


def target_path(data_dir, key):
    """<data dir>/cache/benchmark/offpitch/target-<key>.json."""
    return os.path.join(data_dir, *CACHE_FOLDER, "target-{}.json".format(key))


def _key(inputs):
    """The first 16 hex digits of the sha256 of everything the numbers depend on."""
    parts = {
        "atd_sources": inputs.atd_manifest["sources"],
        "offpitch_sha256": OFFPITCH_SHA256,
        "numpy": np.__version__,
        "pitch": [float(side) for side in inputs.pitch],
        "fps": inputs.fps,
        "spans": [list(span) for span in kickoff_spans(inputs)],   # the kick-offs and graded ends the share is taken on
        "version": VERSION,
    }
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:KEY_DIGITS]


# The hash of this file, taken as the module is imported (calibration.CALIBRATION_SHA256's rule).
OFFPITCH_SHA256 = calibration_module._text_sha256(__file__)


# -- the one-time fit of m and q --------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, eq=False)
class _Half:
    """What the fit keeps of one half of one match, on its span from the kick-off: the line distance of DFL's clean
    ball on each row (NaN without a ball), whether the row has a ball (seen), whether it has one past a line (below),
    and per seed the transplanted mask and the thinning's uniforms of its interior gaps (draws)."""
    line: np.ndarray
    seen: np.ndarray
    below: np.ndarray
    draws: tuple


def fit(calibration, target_share, seeds=DEFAULT_SEEDS, match_ids=None, data_dir=None, log=print, matches=None):
    """m and q of the offpitch profile, fitted once on label-free statistics by the module docstring's rule: the record.

    `calibration` is our match's Calibration (calibration.load()): its masks are transplanted and its missing share is
    q's target. `target_share` is our share of the visible ball frames past a line (target()'s beyond_line_share), m's
    target. Each match (all seven, or `match_ids`; `matches` is {id: DflMatch} to use instead of loading, as in
    fidelity.report) is read once from `data_dir` and degraded under seeds 0 .. seeds - 1. `log` receives a line per
    match and one for the result.

    Returns the record, FIT_KEYS: JSON-ready, finite, aggregates only (the module docstring). run.BenchmarkError, naming
    offpitch.fit and the argument, for a calibration that is not a Calibration or whose missing share is not a share, a
    target_share that is not a share (a finite number from 0 to 1), seeds that are not a whole number of at least 1,
    matches that are not as asked, a share with no frame under it (no visible ball), and a sweep and a ruler that
    disagree.
    """
    if not isinstance(calibration, calibration_module.Calibration):
        raise run.BenchmarkError("offpitch.fit: calibration must be our match's Calibration, as calibration.load() "
                                 "returns it, and it is a {}".format(type(calibration).__name__))
    goal = _share("target_share", target_share)
    missing_goal = _share("the Calibration's scalars['missing_share']", calibration.scalars.get("missing_share"))
    seeds = _seeds(seeds)
    ids = run.resolve_match_ids(match_ids, matches)
    views, halves = [], []
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise run.BenchmarkError("offpitch.fit: the match given as {} says it is {}; its figures would be filed "
                                     "under the wrong id".format(match_id, match.match_id))
        view = _ball_view(degrade.tracking_view(match))
        del match
        parts = _halves(view, calibration, seeds)
        views.append(view)
        halves += parts
        log("offpitch fit  {}  {:,} rows from the kick-offs, seeds 0-{}".format(
            match_id, sum(part.line.size for part in parts), seeds - 1))
    frames = sum(part.line.size * len(part.draws) for part in halves)

    sweep_m = _sweep_m(halves, 0.0)
    first = float(M_GRID[_closest(sweep_m, goal)])
    sweep_q = _sweep_q(halves, first, frames)
    drop_prob = float(Q_GRID[_closest(sweep_q, missing_goal)])
    checked = _sweep_m(halves, drop_prob, margins=(first,))[0]
    refit = abs(Fraction(*checked) - Fraction(goal)) > Fraction(REFIT_TOLERANCE)
    margin, final, sweep_refit = first, checked, []
    if refit:
        sweep_refit = _sweep_m(halves, drop_prob)
        index = _closest(sweep_refit, goal)
        margin, final = float(M_GRID[index]), sweep_refit[index]

    counts, draws = _ruler(views, calibration, seeds, margin, drop_prob)
    if (counts["beyond_frames"], counts["visible_frames"], counts["frames"]) != final + (frames,):
        raise run.BenchmarkError(
            "offpitch.fit: at m = {} and q = {} the sweep counted {} frames past a line of {} visible in {}, and the "
            "ruler on the profile's ball {} of {} in {}: the fit no longer measures what the profile builds".format(
                margin, drop_prob, final[0], final[1], frames, counts["beyond_frames"], counts["visible_frames"],
                counts["frames"]))
    record = {
        "target_share": _round(goal), "missing_target": _round(missing_goal),
        "margin_m": margin, "drop_prob": drop_prob,
        "share_at_fit": _round(counts["beyond_frames"] / counts["visible_frames"]),
        "missing_at_fit": _round(counts["missing_frames"] / counts["frames"]),
        "refit": bool(refit), "first_margin_m": first, "share_at_first_margin": _round(checked[0] / checked[1]),
        "sweep_m": _rows(M_GRID, sweep_m), "sweep_q": _rows(Q_GRID, sweep_q),
        "sweep_m_refit": _rows(M_GRID, sweep_refit) if refit else [],
        "counts_at_fit": counts, "draws_at_fit": draws,
        "seeds": list(range(seeds)), "matches": list(ids),
        "calibration": {"sha256": calibration_module.digest(calibration),
                        "version": calibration_module.CALIBRATION_VERSION},
    }
    try:
        json.dumps(record, allow_nan=False)
    except ValueError as error:
        raise run.BenchmarkError("offpitch.fit: the record holds a value that is not finite ({}); a fit's record "
                                 "never may".format(error)) from None
    log("offpitch fit  m = {:g} m (share past a line {:.4f}, ours {:.4f}), q = {:.3f} (missing share {:.4f}, ours "
        "{:.4f}), {}".format(margin, record["share_at_fit"], record["target_share"], drop_prob,
                             record["missing_at_fit"], record["missing_target"],
                             "m refitted once at q, from {:g} m".format(first) if refit else "no refit"))
    return record


def constants_from(record):
    """(margin_m, drop_prob): the two constants that a fit's record fixes, as degrade.py pins them.

    `record` is fit's record, or the scalars of it that degrade.py pins (OFFPITCH_FIT). ValueError, naming
    offpitch.constants_from and the record, for one that is not a dict, lacks either value, or has one that is not on
    its grid (M_GRID, Q_GRID): no fit makes such a record."""
    if not isinstance(record, dict):
        raise ValueError("offpitch.constants_from: the record must be a dict, offpitch.fit's record or the scalars "
                         "that degrade.py pins of it, and it is a {}".format(type(record).__name__))
    values = []
    for name, grid in (("margin_m", M_GRID), ("drop_prob", Q_GRID)):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in grid.tolist():
            raise ValueError("offpitch.constants_from: the record's {} is {!r}, which is not a value of its grid "
                             "({:g} to {:g}); a fit chooses one of them".format(name, value, grid[0], grid[-1]))
        values.append(float(value))
    return tuple(values)


def _ball_view(view):
    """The TrackingView with no player: everything the ball's degradation reads (offpitch_masks, inputs.clean_atd and
    degrade._remove_ball), at the cost of the ball, which is copied so that nothing of the match is held."""
    n = view.n_frames
    ball = {name: np.array(getattr(view, name)) for name in ("ball_xy", "ball_z", "ball_speed")}
    for array in ball.values():
        array.setflags(write=False)
    return dataclasses.replace(view, players_xy=np.zeros((n, 0, 2)), players_speed=np.zeros((n, 0)),
                               player_team=np.zeros(0, dtype=np.int8), **ball)


def _halves(view, calibration, seeds):
    """[_Half] of a view, one per graded half, built from the profile's own parts.

    The transplanted mask of a half and seed is offpitch_masks' at an infinite margin (nothing is past it) and q = 0
    (nothing is dropped), and its record counts the mask's interior gaps, whose uniforms are thinning_uniforms'."""
    plain = [degrade.offpitch_masks(view, calibration, seed, math.inf, 0.0) for seed in range(seeds)]
    ball = np.asarray(view.ball_xy, dtype=np.float64)
    out = []
    for half, _, _ in view.graded:
        kickoff, stop = degrade._span(view, half)
        line = degrade.line_distance_m(ball[kickoff:stop], view.pitch)
        seen = np.isfinite(ball[kickoff:stop]).all(axis=1)
        draws = tuple(
            (masks[half], degrade.thinning_uniforms(view, half, seed, record["halves"][half]["interior_gaps"]))
            for seed, (masks, record) in enumerate(plain))
        out.append(_Half(line=line, seen=seen, below=seen & (line < 0), draws=draws))
    return out


def _sweep_m(halves, drop_prob, margins=M_GRID):
    """[(beyond, visible)] at each margin and `drop_prob`, summed over the halves and seeds: the visible frames are
    those with a ball, outside the thinned mask and outside the off-pitch stretches. BenchmarkError where none is."""
    kept = [[half.seen & ~degrade.thin_mask(base, uniforms, drop_prob) for base, uniforms in half.draws]
            for half in halves]
    rows = []
    for margin in margins:
        beyond = visible = 0
        for half, shown in zip(halves, kept):
            on = ~degrade.offpitch_mask(half.line, margin)
            for seen in shown:
                frames = seen & on
                visible += int(np.count_nonzero(frames))
                beyond += int(np.count_nonzero(frames & half.below))
        if not visible:
            raise run.BenchmarkError(
                "offpitch.fit: no visible ball frame at m = {:g} and q = {:g}, so the share past a line is 0 / 0; the "
                "matches have no ball on their spans".format(margin, drop_prob))
        rows.append((beyond, visible))
    return rows


def _sweep_q(halves, margin, frames):
    """[(missing, frames)] at each drop probability of Q_GRID and `margin`, summed over the halves and seeds: a frame is
    missing unless it has a ball outside the thinned mask and the off-pitch stretches."""
    if not frames:
        raise run.BenchmarkError("offpitch.fit: the matches have no frame in their spans, so no missing share")
    shown = [half.seen & ~degrade.offpitch_mask(half.line, margin) for half in halves]
    rows = []
    for drop_prob in Q_GRID:
        visible = sum(int(np.count_nonzero(seen & ~degrade.thin_mask(base, uniforms, drop_prob)))
                      for half, seen in zip(halves, shown) for base, uniforms in half.draws)
        rows.append((frames - visible, frames))
    return rows


def _closest(rows, goal):
    """The index of the row whose share (numerator, denominator) is closest to `goal`; on a tie the first, which is the
    smaller value of an ascending grid. Compared exactly: the fraction of the counts against that of the float."""
    target_fraction = Fraction(goal)
    best = best_distance = None
    for index, (numerator, denominator) in enumerate(rows):
        distance = abs(Fraction(numerator, denominator) - target_fraction)
        if best is None or distance < best_distance:
            best, best_distance = index, distance
    return best


def _ruler(views, calibration, seeds, margin_m, drop_prob):
    """(counts, draws) of the profile's ball at (margin_m, drop_prob), measured as fidelity.py measures it.

    The ball is made as degrade() makes it for offpitch (offpitch_masks, then _remove_ball on inputs.clean_atd's ball)
    and measured on each half's span from its kick-off by calibration.ball_stats (frames, missing_frames) and line_stats
    (visible_frames, beyond_frames), summed over the matches and seeds; draws are offpitch_masks' records added up
    (degrade.OFFPITCH_COUNTS)."""
    counts = {"frames": 0, "missing_frames": 0, "visible_frames": 0, "beyond_frames": 0}
    draws = {name: 0 for name in degrade.OFFPITCH_COUNTS}
    for view in views:
        clean = match_inputs.clean_atd(view)
        spans = [(half,) + degrade._span(view, half) for half, _, _ in view.graded]
        metres = np.array(view.pitch, dtype=np.float64)
        for seed in range(seeds):
            masks, record = degrade.offpitch_masks(view, calibration, seed, margin_m, drop_prob)
            ball_xy, _ = degrade._remove_ball(view, clean, masks)
            present = ~np.isnan(ball_xy[:, 0])
            ball = calibration_module.ball_stats(present, spans, ball_xy * metres, view.pitch, view.fps)
            line = line_stats(present, spans, ball_xy * metres, view.pitch)
            counts["frames"] += ball["frames"]
            counts["missing_frames"] += ball["missing_frames"]
            counts["visible_frames"] += line["visible_frames"]
            counts["beyond_frames"] += line["beyond_frames"]
            for name in degrade.OFFPITCH_COUNTS:
                draws[name] += record[name]
    return counts, draws


def _rows(grid, rows):
    """[[value, share]] of a sweep, the share numerator / denominator rounded as a record's floats are."""
    return [[float(value), _round(numerator / denominator)] for value, (numerator, denominator) in zip(grid, rows)]


def _share(name, value):
    """`value` as a float share; run.BenchmarkError naming offpitch.fit and `name` unless it is a finite number in
    [0, 1]."""
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
            or not 0.0 <= value <= 1.0):
        raise run.BenchmarkError("offpitch.fit: {} must be a share, a finite number from 0 to 1, and it is {!r}".format(
            name, value))
    return float(value)


def _seeds(value):
    """`value` as an int of at least 1; run.BenchmarkError naming offpitch.fit and seeds otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise run.BenchmarkError("offpitch.fit: seeds must be a whole number of at least 1 (seeds 0 .. seeds - 1 are "
                                 "run), and it is {!r}".format(value))
    return int(value)


# -- the files --------------------------------------------------------------------------------------------------------

def _round(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), DECIMALS)


def _write(path, result):
    """The JSON, written whole to a temp file in the folder and moved over the old one, so no reader sees half."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            json.dump(result, out, indent=1, sort_keys=True, allow_nan=False)
            out.write("\n")
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def _read(path, key):
    """The cached target as a dict, or None if there is none, it cannot be read, or it is not this key's and version."""
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or saved.get("key") != key or saved.get("version") != VERSION:
        return None
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="P18's ruler (the share of visible ball frames past a line) and the "
                                                 "one-time fit of the offpitch profile's two constants (R015).")
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("target", help="print our match's target (aggregates only), measuring it if need be")
    cmd.add_argument("--rebuild", action="store_true", help="measure again instead of reading the cache")
    cmd.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    cmd = commands.add_parser("fit", help="fit the offpitch profile's margin and drop probability, once, and print "
                                          "the record (aggregates only)")
    cmd.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                     help="degradation seeds per match (default {})".format(DEFAULT_SEEDS))
    cmd.add_argument("--matches", nargs="+", metavar="ID", help="the matches to fit on (default: all 7)")
    cmd.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    cmd.add_argument("--out", metavar="PATH", help="also write the record there, as JSON")
    args = parser.parse_args(argv)
    if args.command == "target":
        result = target(data_dir=args.data_dir, rebuild=args.rebuild)
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    try:
        cal = calibration_module.load(data_dir=args.data_dir)
        ours = target(data_dir=args.data_dir)
        record = fit(cal, ours.get("beyond_line_share"), seeds=args.seeds, match_ids=args.matches,
                     data_dir=args.data_dir)
        record["target_key"] = ours.get("key")
        text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False)
        if args.out:
            _write(args.out, record)
    except (ValueError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print(text)
    if args.out:
        print("record    {}".format(run.shown(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
