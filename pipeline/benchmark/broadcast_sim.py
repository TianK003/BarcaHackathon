"""A simulated broadcast ball for matches that have none: what the TV's ball does in OUR gaps, transplanted (R028).

    gaps = gaps_of(ball_present, graded)                  # [(start, stop)]: the maximal runs of frames without the ball
    alive = gap_alive(gaps, codes)                        # (k,) bool: the play goes on in each gap, by the labels
    pools = Pools.from_lists(masks, alive_flags, error_runs)   # the donors: BAR-ATH's fill masks and position errors
    ball = simulate(ball_present, clean_ball_m, driver_codes, graded, pitch, pools, rng)   # (n, 2) normalised, NaN
    stats = measure_fill(ball_present, filled, codes, graded, fill_xy_m, reference_xy_m, pitch)   # F1-F5, W1-W3

Why it exists. R028 (docs/superpowers/specs/2026-11-01-r028-broadcast-ball-design.md) fills the gaps of our tactical
ball with the TV broadcast's ball. The benchmark's matches (DFL's seven) have no broadcast, so R028 gives each of them a
simulated one that does, in that match's gaps, what the real one does in ours: it has the ball in more of the gaps where
the play goes on than in the dead ones, mostly at their edges, and it is a metre or so off. R019's camera simulator
failed because a parametric follower could not reproduce our camera's stillness, so this is a transplant (P01's way:
"their pattern, not their clock"): the fill masks and the position errors are COPIED from BAR-ATH, not modelled. The
core below (Pools, gaps_of, gap_alive, simulate, measure_fill) works on arrays and nothing else: it loads no data and
knows no match. The wiring after it (Task 2b, last section of this docstring) builds the Pools from BAR-ATH, serves the
simulated ball to the benchmark's runs and holds the command line.

The codes and the blocks. A code is possession_metrics' (H, A, D, X are 0 to 3), and a frame is alive when its code is
H or A. `graded` is [(half, start, stop), ...], half-open, as DflMatch.graded and Inputs.graded give it. A gap is a
maximal run of graded frames without the ball, cut at its block's edges (a run that touches a block's first or last
frame is one), as (start, stop), in time order.

The transplant (design 3.4). For every gap of the served ball, of m frames, in time order:
  1. Its class: alive if at least ALIVE_SHARE of its frames' driver codes are H or A (exactly half is alive), else D.
  2. Its donor: among the pool's donor gaps of that class, the DONORS whose lengths k are nearest to m in |log(k / m)|,
     listed nearest first. The ranking is exact: the longer of k and m over the shorter, as a fraction, then the
     shorter k, then the earlier donor (a float log puts 40 and 90, both 1.5 times from 60, in the wrong order, and
     the ties are what the rule is about). With fewer than DONORS donors of the class, all of them; with none, the
     gap gets no fill and no draw.
  3. Its mask: fill[start + i] = donor_fill[(i * k) // m] for i = 0 ... m - 1, in integer arithmetic. The donor's two
     edges stay the gap's two edges, whatever the lengths.
  4. Its positions: each maximal fill fragment of f frames takes the first f offsets of an error run: one drawn
     uniformly among the runs at least f long or, if none is that long, the longest (the earlier of equals) repeated
     end to end to f frames. The offsets, in metres, are added to the clean ball.
  5. No ball on DFL there: a fill frame where the clean ball is NaN in either coordinate stays NaN. The fragments and
     the draws are the same with or without it.

The draws, exactly. For each gap that has a donor, one `rng.integers(c)`, c the DONORS or, if fewer, the donors of the
class: it names the donor in nearest-first order (even from a single candidate: a draw is made whenever there is
something to choose from). Then, for each fragment in time order that some error run is long enough for, one
`rng.integers(r)`, r the number of such runs: it names one of them in the pool's order. A fragment that no run is long
enough for makes no draw. Nothing else is random, and the generator is the caller's (Task 2b makes it with
numpy.random.default_rng((seed, match index, RNG_SALT))), so the same seed gives the same bytes.

No truth. simulate takes no match and nothing that carries DFL's ball flags: the class of a gap is read from the
driver's codes, which are the candidate's own labels without the fill (design 3.2), because a ball placed by the
truth's alive flag would carry the answer.

measure_fill (design 3.5) is the one function that both BAR-ATH's real fill and the simulated copies are measured by:
the gates F1 to F3 (where the fill falls), the checks F4 and F5 (how long its fragments are and how far off it is) and
the warnings W1 to W3 (how much of the match it covers, how much of the gaps is alive, how much is beyond a line). It
returns raw numbers with their counts, so that copies can be pooled by summing counts, and None where a share has no
frames.

The Pools are frame-level data of our match: they stay under data/cache/, are never printed and never committed. Every
field has repr=False, and no message here holds a value of an array, only shapes and counts.

The wiring (Task 2b): the pools, the served inputs, and the three commands.

    python pipeline/benchmark/broadcast_sim.py pools [--data-dir D]
    python pipeline/benchmark/broadcast_sim.py fidelity [--profile P] [--seeds N] CONFIG [--data-dir D]
    python pipeline/benchmark/broadcast_sim.py bar-ath CONFIG [--data-dir D]

    pools = load_pools()                       # BAR-ATH's donors and errors, read from the cache `pools` built
    inputs = served_inputs_for(config, match, atd, profile, seed, data_dir, pools)   # what the runner serves a config

* The pools (design 3.3) are built from BAR-ATH by the pipeline's own functions: the labels are r024_both.toml's,
  decoded in-process by pipeline.run.run into a temporary CSV in data/cache/, read and deleted as soon as its sha256
  is known (the labels are kept in memory only), and nothing is built unless that CSV is R024_SHA256, the file
  main.toml writes today (a stale or dirty main.csv cannot creep in, and the stand-in stays fixed after an
  adoption). Each gap of our ball inside a graded block gives its real fill mask (the filled view's
  ball_filled) and its class by those labels; the error runs are the maximal runs of graded frames where both balls
  exist within NEAR_GAP_FRAMES of a gap of their own block (near_gap_errors), each offset the broadcast's position minus
  ours, in metres. BAR-ATH's side of fidelity is measured at the same time and kept beside them as aggregates
  (load_bar_ath_measure), so that fidelity never decodes BAR-ATH: measure_fill on the ball before the fill, the fill's
  own mask, those labels and the broadcast ball in metres, with no reference (our ball is missing on every filled
  frame), and F5 the near-gap pool's median |offset| (the design's appendix). The cache,
  <data dir>/cache/benchmark/broadcast_sim/<digest of the key>/, follows calibration.py's rules: the key holds what the
  pools depend on (pools_key: the sha256 of the code that builds them, CODE_FILES, the two ATD caches' manifests, the
  broadcast homography file's sha256, r024_both.toml and the match's scalars; it is read from files, and no array of
  BAR-ATH's tracking is loaded to compute it), the arrays are written first and the manifest last. Only the `pools`
  command builds them (prepare_pools); the runner, latency, fidelity and bar-ath read them (load_pools), and an
  absent, broken or stale cache is an error there that says to run `pools` first.
* The served inputs (design 3.2 and 3.4). A config without the fill is served the plain MatchInputs. A config that
  fills is served its simulated ball: its twin, the same config with ball_fill "none", is decoded on the plain inputs
  (the driver: the candidate's own labels, never the truth), simulate builds the ball with the match's own generator,
  default_rng([seed, degrade.index_of(match), RNG_SALT]), and the ball is served as it comes, to the fill of
  Inputs.with_ball_fill unchanged: no usable test and no 13-frame filter on DFL (the donor masks carry BAR-ATH's), so a
  compressed transplant may give fragments shorter than 13 frames. The driver's codes can be kept in a dict per (twin,
  match, profile, seed), under driver_key, which holds everything predict reads of a config; the runner keeps one per
  run, and a config whose content is a twin's (main.toml's, say) shares its codes.
* `pools` builds the pools (or reads a whole cache of the key) and prints aggregates only. `fidelity` measures the
  copies (each match and seed of the profile, the config as it runs on DFL: its match-only sources dropped, as
  --drop-match-only does), laid end to end and measured once, against BAR-ATH's measure: the gates F1 to F3, the
  checks F4 and F5 and the warnings W1 to W3 of design 3.5; exit 0, 1 if a gate fails, 3 if only a check does.
  `bar-ath` is design 7.4 on BAR-ATH: the alignment band and the live share first (exit 3, a STOP, if either fails,
  after printing them), then the tag check on the fill-only frames of the teacher (conventions.TOUCH), no X on a graded
  frame and D within D_SHARE_BAND (exit 1, a DROP, if one fails), and the figures that are only reported. Every
  command exits 2 on an error, which is never a verdict.
  The reports go to <data dir>/runs/r028/ (fidelity-<config stem>-<profile>[.s<n>].json, bar-ath-<config stem>.json),
  aggregates only, with floats to 9 decimals and the pools' digest.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import operator
import os
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import conventions, degrade, dfl, label_diff, run, tag_stats  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.config import BALL_FILLS, ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, smart_tagging  # noqa: E402
from pipeline.inputs import BROADCAST_FOLDER, Inputs, atd_cache  # noqa: E402
from pipeline.inputs.broadcast import AlignmentError  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
POOLS_VERSION = 1       # part of the pools' cache key (Task 2b): change what a Pools holds or how it is built, bump it
DONORS = 5              # a gap's donor is one of this many donor gaps of its class, the nearest in length
ALIVE_SHARE = 0.5       # a gap, or a fill fragment, is alive if at least this share of its frames are H or A
NEAR_GAP_FRAMES = 50    # the error pool's frames lie within this far (2 s) of one of our gaps (Task 2b builds it)
EDGE_FRAMES = 25        # F3: the first and the last of these frames (1 s) of a gap are its edges
LONG_GAP_FRAMES = 50    # F3: only gaps of at least this many frames (2 s) are read
BEYOND_LINE_M = 1.0     # W3: a fill position more than this beyond a line (x < -1, x > L + 1, y < -1, y > W + 1)
RNG_SALT = 28           # the third element of the generator's key (Task 2b): default_rng((seed, match index, 28))
H, A, D, X = pm.H, pm.A, pm.D, pm.X
# The pools (design 3.3) and their cache
R024_CONFIG = os.path.join(ROOT, "pipeline", "configs", "r024_both.toml")    # the record config the labels come from
R024_SHA256 = "e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170"   # its CSV, what main.toml writes
CACHE_FOLDER = ("cache", "benchmark", "broadcast_sim")   # <data dir>/cache/benchmark/broadcast_sim/<digest of the key>/
KEY_DIGITS = 16                                          # hex digits of the key's sha256 that name its folder
MANIFEST = "manifest.json"                               # the key and the pools' digest, written last
MEASURE_FILE = "bar_ath_measure.json"                    # BAR-ATH's side of fidelity (measure_fill's result)
LABELS_STEM = "r024_both"   # the decode's CSV, r024_both.<pid>.csv beside the key folders: deleted once read
# The code that builds the pools and measures BAR-ATH's side, from the repository root: its sha256 is in the key
CODE_FILES = ("pipeline/benchmark/broadcast_sim.py", "pipeline/inputs/broadcast.py", "pipeline/inputs/__init__.py")
POOLS_FIRST = ("run `python pipeline/benchmark/broadcast_sim.py pools` first: it is the one command that builds "
               "the pools (and the ATD caches they are keyed on)")
POOL_ARRAYS = ("gap_lengths", "gap_alive", "gap_fill", "gap_starts", "error_xy", "error_starts")   # Pools, in order
MEASURE_KEYS = ("F1", "F2", "F3", "F4", "F5", "W1", "W2", "W3")
# Fidelity (design 3.5): |copies - ours| at most this
GATE_TOLERANCES = {"F1": 0.05, "F2": 0.03, "F3": 0.05}  # shares; F3 for each of its first, last and inside windows
CHECK_TOLERANCES = {"F4": 10.0, "F5": 0.2}              # frames (the median fragment, by class), metres (median error)
EXIT_ERROR = 2                                          # every command, on an error: never a verdict
FIDELITY_EXIT = {"pass": 0, "gate": 1, "error": EXIT_ERROR, "check": 3}
# BAR-ATH, label-free (design 7.4)
MAIN_CONFIG = os.path.join(ROOT, "pipeline", "configs", "main.toml")   # the reference of the reported figures
EXPECTED_OFFSETS = {1: 4014, 2: 26919}   # d_h of A20: tactical frame t is broadcast frame t - d_h
OFFSET_BAND = 2                          # frames: d_h must be this close to them
MIN_LIVE_SHARE = 0.85                    # live frames over valid graded frames, at least
TAG_MARGIN = 0.05                        # the fill-only agreement may be this far below the plain one
MIN_TAG_FRAMES = 300                     # fill-only teacher frames with a tag's vote, for the tag check to decide
D_SHARE_BAND = (0.30, 0.62)              # R024's sanity: D on this share of the graded frames, ends included
POSSESSIONS_WARN = 0.25                  # a half's possessions beyond this share of main.toml's are a WARN (T33)
BAR_ATH_EXIT = {"pass": 0, "drop": 1, "error": EXIT_ERROR, "stop": 3}
REPORT_FOLDER = ("runs", "r028")   # <data dir>/runs/r028/: fidelity_path's and bar_ath_path's reports


# -- the pools --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Pools:
    """The donors of the transplant: BAR-ATH's gaps with their fill masks and classes, and its broadcast ball's errors.

        gap_lengths   (k,) int64: the length in frames of each donor gap
        gap_alive     (k,) bool: its class, alive (True) or D (False), by the labels of the run that built the pools
        gap_fill      (sum of gap_lengths,) bool: the masks laid end to end, True where the broadcast ball filled
        gap_starts    (k + 1,) int64: gap j's mask is gap_fill[gap_starts[j]:gap_starts[j + 1]]; 0, then running totals
        error_xy      (m, 2) float64: broadcast minus tactical ball, in metres, of every error run laid end to end
        error_starts  (r + 1,) int64: error run j is error_xy[error_starts[j]:error_starts[j + 1]]

    Every field is validated (a ValueError that names it) and stored as a read-only copy of its own dtype. A donor gap
    and an error run have at least one frame, and the offsets are finite. A pool with no donor gap, or no error run, may
    exist (simulate refuses one with no error run). Frame-level data of our match: nothing here is printed (every field
    has repr=False, and no message holds a value), and eq=False, because a field-wise == over arrays means nothing:
    compare the arrays.
    """
    gap_lengths: np.ndarray = field(repr=False)
    gap_alive: np.ndarray = field(repr=False)
    gap_fill: np.ndarray = field(repr=False)
    gap_starts: np.ndarray = field(repr=False)
    error_xy: np.ndarray = field(repr=False)
    error_starts: np.ndarray = field(repr=False)

    def __post_init__(self):
        lengths = _stored("gap_lengths", self.gap_lengths, "int", 1, "one length in frames per donor gap")
        alive = _stored("gap_alive", self.gap_alive, "bool", 1, "one class per donor gap")
        fill = _stored("gap_fill", self.gap_fill, "bool", 1, "the donor gaps' masks laid end to end")
        starts = _stored("gap_starts", self.gap_starts, "int", 1, "k + 1 offsets into gap_fill")
        errors = _stored("error_xy", self.error_xy, "float", 2, "(m, 2): the offsets in metres")
        error_starts = _stored("error_starts", self.error_starts, "int", 1, "r + 1 offsets into error_xy")
        if (lengths < 1).any():
            raise ValueError("Pools.gap_lengths must be at least 1 in every gap; {} are below 1".format(
                int((lengths < 1).sum())))
        if alive.shape != lengths.shape:
            raise ValueError("Pools.gap_alive has {} flags for the {} gaps of Pools.gap_lengths; give one per gap"
                             .format(alive.size, lengths.size))
        if starts.shape != (lengths.size + 1,):
            raise ValueError("Pools.gap_starts must have k + 1 = {} offsets for the {} gaps of Pools.gap_lengths, "
                             "got {}".format(lengths.size + 1, lengths.size, starts.size))
        if starts[0] != 0:
            raise ValueError("Pools.gap_starts must start at 0: gap j's mask is "
                             "gap_fill[gap_starts[j]:gap_starts[j + 1]]")
        if (np.diff(starts) != lengths).any():
            raise ValueError("Pools.gap_starts must be the running totals of Pools.gap_lengths: gap j's mask is "
                             "gap_fill[gap_starts[j]:gap_starts[j + 1]], so its length is gap_lengths[j]")
        if fill.shape != (int(starts[-1]),):
            raise ValueError("Pools.gap_fill has {} frames but Pools.gap_starts ends at {}; the masks laid end to end "
                             "are as long as the gaps together".format(fill.size, int(starts[-1])))
        if errors.shape[1] != 2:
            raise ValueError("Pools.error_xy must have 2 columns, (m, 2): the x and y offsets in metres; it has "
                             "shape {}".format(errors.shape))
        if not np.isfinite(errors).all():
            raise ValueError("Pools.error_xy must be finite: {} offsets are not (an error is the broadcast ball minus "
                             "the tactical one, on frames where both exist)".format(
                                 int((~np.isfinite(errors)).sum())))
        if error_starts.size < 1:
            raise ValueError("Pools.error_starts must hold at least one offset: r + 1 of them for r runs, [0] for none")
        if error_starts[0] != 0:
            raise ValueError("Pools.error_starts must start at 0: error run j is "
                             "error_xy[error_starts[j]:error_starts[j + 1]]")
        if (np.diff(error_starts) < 1).any():
            raise ValueError("Pools.error_starts must grow by at least one frame per run; {} runs are empty or out of "
                             "order".format(int((np.diff(error_starts) < 1).sum())))
        if errors.shape[0] != int(error_starts[-1]):
            raise ValueError("Pools.error_xy has {} rows but Pools.error_starts ends at {}; the runs laid end to end "
                             "are as long as the runs together".format(errors.shape[0], int(error_starts[-1])))
        for name, array in (("gap_lengths", lengths), ("gap_alive", alive), ("gap_fill", fill), ("gap_starts", starts),
                            ("error_xy", errors), ("error_starts", error_starts)):
            object.__setattr__(self, name, array)

    @classmethod
    def from_lists(cls, masks, alive, error_runs):
        """Pools from Python lists, for the checks and for Task 2b's builder.

        `masks` is one list of bools per donor gap (its fill, frame by frame: its length is the gap's), `alive` one
        bool per gap (its class), and `error_runs` one list of (dx, dy) pairs, in metres, per error run. Every field
        is validated as the constructor does: a mask of ints is a ValueError, not a mask of the truthy.
        """
        masks = [list(mask) for mask in masks]
        flags = np.array(list(alive))
        runs = [list(run) for run in error_runs]
        if flags.size != len(masks):
            raise ValueError("Pools.from_lists: alive has {} flags for {} masks; give one per mask".format(
                flags.size, len(masks)))
        bits = np.array([bit for mask in masks for bit in mask])
        try:
            pairs = np.array([pair for run in runs for pair in run], dtype=np.float64)
        except (TypeError, ValueError):
            pairs = None
        if pairs is None or (pairs.size and (pairs.ndim != 2 or pairs.shape[1] != 2)):
            raise ValueError("Pools.from_lists: error_runs must be lists of (dx, dy) pairs in metres, one list per run")
        return cls(gap_lengths=np.array([len(mask) for mask in masks], dtype=np.int64),
                   gap_alive=flags if flags.size else flags.astype(bool),
                   gap_fill=bits if bits.size else bits.astype(bool),
                   gap_starts=np.concatenate([[0], np.cumsum([len(mask) for mask in masks], dtype=np.int64)]),
                   error_xy=pairs.reshape(-1, 2),
                   error_starts=np.concatenate([[0], np.cumsum([len(run) for run in runs], dtype=np.int64)]))


# -- reading the arguments --------------------------------------------------------------------------------------------

_KINDS = {"int": ("iu", np.int64, "integers"), "bool": ("b", np.bool_, "bool"), "float": ("fiu", np.float64, "numbers")}


def _stored(name, value, kind, ndim, wording):
    """`value` as a read-only copy of its own dtype (kind: int, bool or float); a ValueError that names Pools.<name>
    unless it is an array of that kind with `ndim` dimensions."""
    kinds, dtype, things = _KINDS[kind]
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in kinds:
        raise ValueError("Pools.{} must be a {}-D array of {} ({}); it is {} with shape {}".format(
            name, ndim, things, wording, array.dtype, array.shape))
    stored = np.array(array, dtype=dtype)           # a copy: a pool never shares memory with a caller's array
    stored.flags.writeable = False
    return stored


def _flags(who, name, value, n=None):
    """`value` as a 1-D bool array (of n flags, if n is given), else a ValueError that names `name`."""
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.ndim != 1 or (n is not None and array.shape[0] != n):
        raise ValueError("{}: {} must be a 1-D bool array{}, one flag per frame; it is {} with shape {}".format(
            who, name, "" if n is None else " of {} flags like ball_present".format(n), array.dtype, array.shape))
    return array


def _codes(who, name, value, n=None):
    """`value` as a 1-D integer array of pm's codes (n of them, if n is given), else a ValueError that names `name`."""
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or (n is not None and array.shape[0] != n):
        raise ValueError("{}: {} must be a 1-D integer array of {}codes, one per frame; it is {} with shape {}".format(
            who, name, "" if n is None else "{} ".format(n), array.dtype, array.shape))
    odd = np.unique(array[(array < 0) | (array >= pm.N_LABELS)])
    if odd.size:
        raise ValueError("{}: {} holds the codes {}, and the scorer's are 0-{} (H, A, D, X)".format(
            who, name, odd.tolist(), pm.N_LABELS - 1))
    return array


def _positions(who, name, value, n):
    """`value` as an (n, 2) float64 array of positions in metres, else a ValueError that names `name`."""
    array = np.asarray(value)
    if array.shape != (n, 2) or array.dtype.kind not in "fiu":
        raise ValueError("{}: {} must be (n, 2) with n = {} frames like ball_present, the positions in metres, NaN "
                         "where there is none; it is {} with shape {}".format(who, name, n, array.dtype, array.shape))
    return array.astype(np.float64, copy=False)


def _blocks(who, graded, n):
    """`graded` as a list of (half, start, stop) ints sorted by start; a ValueError unless they are n-frame blocks."""
    try:
        items = list(graded)
    except TypeError:
        raise ValueError("{}: graded must be a list of (half, start, stop) triples, got {!r}".format(
            who, graded)) from None
    blocks = []
    for block in items:
        try:
            half, start, stop = (operator.index(value) for value in block)
        except (TypeError, ValueError):
            raise ValueError("{}: graded must hold (half, start, stop) triples of integers, got {!r}; give the blocks "
                             "as DflMatch.graded and Inputs.graded do".format(who, block)) from None
        if start < 0 or stop > n:
            raise ValueError("{}: graded block of half {} is frames {}-{}, outside the {} frames of the arrays".format(
                who, half, start, stop, n))
        if stop <= start:
            raise ValueError("{}: graded block of half {} is frames {}-{}, an empty block".format(
                who, half, start, stop))
        blocks.append((half, start, stop))
    if not blocks:
        raise ValueError("{}: graded is empty; give the graded blocks, [(half, start, stop), ...], half-open".format(
            who))
    blocks.sort(key=lambda block: block[1])
    for (half, start, stop), (other, other_start, _) in zip(blocks, blocks[1:]):
        if other_start < stop:
            raise ValueError("{}: graded blocks of half {} (frames {}-{}) and half {} (from frame {}) overlap".format(
                who, half, start, stop, other, other_start))
    return blocks


def _spans(who, name, gaps, n):
    """`gaps` as a list of (start, stop) ints inside n frames, each with at least one frame."""
    spans = []
    for gap in gaps:
        try:
            start, stop = (operator.index(value) for value in gap)
        except (TypeError, ValueError):
            raise ValueError("{}: {} must hold (start, stop) pairs of integers, as gaps_of gives them, got {!r}".format(
                who, name, gap)) from None
        if start < 0 or stop > n:
            raise ValueError("{}: {} holds the gap {}-{}, outside the {} frames of the codes".format(
                who, name, start, stop, n))
        if stop <= start:
            raise ValueError("{}: {} holds the empty gap {}-{}; a gap has at least one frame".format(
                who, name, start, stop))
        spans.append((start, stop))
    return spans


def _pitch(who, pitch):
    try:
        length, width = (float(side) for side in pitch)
    except (TypeError, ValueError):
        raise ValueError("{}: pitch must be (length, width) in metres, got {!r}".format(who, pitch)) from None
    if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
        raise ValueError("{}: pitch must be (length, width) in metres, positive and finite, got {!r}".format(
            who, pitch))
    return length, width


def _runs(mask):
    """(first, stop) of the maximal runs of True in a 1-D bool array: int64 arrays, half-open, in time order."""
    padded = np.zeros(len(mask) + 2, dtype=np.int8)
    padded[1:-1] = mask
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# -- the gaps ---------------------------------------------------------------------------------------------------------

def gaps_of(ball_present, graded):
    """[(start, stop)]: the maximal runs of frames without the ball inside each graded block, cut at the block's edges.

    `ball_present` is a 1-D bool array, one flag per frame (False: no ball), and `graded` is [(half, start, stop), ...],
    half-open. A run that touches a block's first or last frame is included, one that runs out of a block is cut at
    its edge, and two blocks that touch do not merge a gap. The gaps come in time order, as plain ints, whatever the
    order of the blocks.
    """
    who = "broadcast_sim.gaps_of"
    present = _flags(who, "ball_present", ball_present)
    return _gaps(present, _blocks(who, graded, present.shape[0]))


def _gaps(present, blocks):
    gaps = []
    for _, start, stop in blocks:
        first, last = _runs(~present[start:stop])
        gaps += [(start + int(a), start + int(b)) for a, b in zip(first, last)]
    return gaps


def gap_alive(gaps, codes):
    """(k,) bool: whether each gap (start, stop) is alive: at least ALIVE_SHARE of its frames' codes are H or A.

    Exactly half is alive, 0.49 is not, and an X or a D is not alive. `codes` is one pm code per frame (0-3).
    """
    who = "broadcast_sim.gap_alive"
    codes = _codes(who, "codes", codes)
    spans = _spans(who, "gaps", gaps, codes.shape[0])
    total = np.concatenate([[0], np.cumsum((codes == H) | (codes == A), dtype=np.int64)])
    return np.array([total[stop] - total[start] >= ALIVE_SHARE * (stop - start) for start, stop in spans], dtype=bool)


# -- the transplant ---------------------------------------------------------------------------------------------------

def simulate(ball_present, clean_ball_m, driver_codes, graded, pitch, pools, rng):
    """(n, 2) normalised positions of the simulated broadcast ball, NaN where unfilled: design 3.4, steps 1 to 5.

    ball_present   (n,) bool: the ball as served to the pipeline, False where it is missing
    clean_ball_m   (n, 2) float: the match's own ball in metres from a pitch corner, NaN where it has none
    driver_codes   (n,) int: the candidate's own labels without the fill (pm codes); read on the gaps only
    graded         [(half, start, stop), ...], half-open; only the gaps inside the blocks are filled
    pitch          (length, width) in metres: the result is the position in metres over it
    pools          the Pools of donors
    rng            a numpy.random.Generator, the caller's: the module docstring says what is drawn and in what order

    Nothing about the match is a parameter, and nothing that carries its truth flags (test_simulate_takes_no_match).
    ValueError, naming the parameter, for any argument that is not as said, and for pools with no error run.
    """
    who = "broadcast_sim.simulate"
    present = _flags(who, "ball_present", ball_present)
    n = present.shape[0]
    clean = _positions(who, "clean_ball_m", clean_ball_m, n)
    codes = _codes(who, "driver_codes", driver_codes, n)
    blocks = _blocks(who, graded, n)
    length, width = _pitch(who, pitch)
    if not isinstance(pools, Pools):
        raise ValueError("{}: pools must be a Pools (Pools.from_lists builds one), got {}".format(
            who, type(pools).__name__))
    if not isinstance(rng, np.random.Generator):
        raise ValueError("{}: rng must be a numpy.random.Generator, got {}; make one with "
                         "numpy.random.default_rng(key), so that the draws are seeded".format(
                             who, type(rng).__name__))
    run_lengths = np.diff(pools.error_starts)
    if not run_lengths.size:
        raise ValueError("{}: pools has no error run, so a filled ball cannot be placed; build the pools from a match "
                         "where both balls exist near a gap".format(who))

    gaps = _gaps(present, blocks)
    lengths = pools.gap_lengths.tolist()
    donors = {True: np.flatnonzero(pools.gap_alive).tolist(), False: np.flatnonzero(~pools.gap_alive).tolist()}
    scale = np.array([length, width])
    filled = np.full((n, 2), np.nan)
    for (start, stop), alive in zip(gaps, gap_alive(gaps, codes).tolist()):
        m = stop - start
        candidates = donors[alive]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda j: (Fraction(max(lengths[j], m), min(lengths[j], m)), lengths[j], j))
        ranked = ranked[:DONORS]
        donor = ranked[int(rng.integers(len(ranked)))]
        k, first = lengths[donor], int(pools.gap_starts[donor])
        mask = pools.gap_fill[first:first + k][(np.arange(m, dtype=np.int64) * k) // m]
        begin, end = _runs(mask)
        for a, b in zip(begin.tolist(), end.tolist()):
            moved = clean[start + a:start + b] + _offsets(pools, run_lengths, b - a, rng)
            moved[np.isnan(moved).any(axis=1)] = np.nan
            filled[start + a:start + b] = moved / scale
    return filled


def _offsets(pools, run_lengths, f, rng):
    """The (f, 2) offsets of a fill fragment of f frames: the first f of an error run drawn uniformly among those at
    least f long or, if none is, of the longest (the earlier of equals), repeated end to end."""
    long_enough = np.flatnonzero(run_lengths >= f)
    if long_enough.size:
        first = int(pools.error_starts[long_enough[int(rng.integers(long_enough.size))]])
        return pools.error_xy[first:first + f]
    longest = int(np.argmax(run_lengths))
    first = int(pools.error_starts[longest])
    return pools.error_xy[first + np.arange(f) % int(run_lengths[longest])]


# -- the measure ------------------------------------------------------------------------------------------------------

def _tally(num, den):
    """{"share", "num", "den"}: the share is None over no frames, never an exception."""
    return {"share": num / den if den else None, "num": num, "den": den}


def _share(numerator, denominator):
    return _tally(int(numerator.sum()), int(denominator.sum()))


def measure_fill(ball_present, filled, codes, graded, fill_xy_m, reference_xy_m, pitch):
    """F1 to F5 and W1 to W3 of design 3.5, as raw numbers with their counts, over the graded frames.

    ball_present   (n,) bool: the ball before the fill (False: missing); its graded False frames are the gap frames
    filled         (n,) bool: where a fill ball was put in; on graded frames only on gap frames (a ValueError says so)
    codes          (n,) int: the labels that class the frames (pm codes): H or A is alive, D is dead, X is neither
    graded         [(half, start, stop), ...], half-open: the only frames read
    fill_xy_m      (n, 2): the fill's position in metres from a pitch corner; read on the filled frames only (finite)
    reference_xy_m (n, 2): the position to compare it with, in metres, NaN where none; read on the filled frames only
    pitch          (length, width) in metres

    The result is made of plain ints, floats and None. A share is {"share", "num", "den"}, the share None where den is
    0: to pool copies by frames, sum the counts and divide once. A median cannot be pooled from medians (it is the
    median of the frames laid end to end), so F4 and F5 carry the count that each is over.

        F1  fill frames over the gap frames whose code is H or A        num: filled and alive, den: gap and alive
        F2  the same over the gap frames whose code is D                num: filled and D, den: gap and D
        F3  {"gaps": the gaps of at least LONG_GAP_FRAMES, "first": ..., "last": ..., "inside": ...}: the fill share
            in their first EDGE_FRAMES frames, in their last EDGE_FRAMES and in the frames between (at least
            EDGE_FRAMES from both ends: none in a gap of exactly 50)
        F4  {"alive": {"median", "n"}, "dead": {"median", "n"}}: the median length in frames of the maximal fill
            fragments (cut at the blocks), alive if at least ALIVE_SHARE of their frames' codes are H or A, else dead
        F5  {"median_m", "n"}: the median |fill - reference| (Euclidean, metres) over the filled frames where both exist
        W1  fill frames over graded frames
        W2  gap frames whose code is H or A, over gap frames
        W3  the share of fill frames whose position is more than BEYOND_LINE_M beyond a line (x < -1, x > L + 1,
            y < -1 or y > W + 1)
    """
    who = "broadcast_sim.measure_fill"
    present = _flags(who, "ball_present", ball_present)
    n = present.shape[0]
    marked = _flags(who, "filled", filled, n)
    codes = _codes(who, "codes", codes, n)
    blocks = _blocks(who, graded, n)
    fill_xy = _positions(who, "fill_xy_m", fill_xy_m, n)
    reference = _positions(who, "reference_xy_m", reference_xy_m, n)
    length, width = _pitch(who, pitch)

    inside = np.zeros(n, dtype=bool)
    for _, start, stop in blocks:
        inside[start:stop] = True
    gap = inside & ~present
    fill = inside & marked
    stray = int((fill & present).sum())
    if stray:
        raise ValueError("{}: filled is True on {} graded frame(s) where ball_present is True; the fill is on gap "
                         "frames only, so ball_present must be the ball before the fill".format(who, stray))
    unplaced = int((fill & np.isnan(fill_xy).any(axis=1)).sum())
    if unplaced:
        raise ValueError("{}: fill_xy_m is NaN on {} frame(s) where filled is True; every filled frame needs the "
                         "position of its fill".format(who, unplaced))
    alive = (codes == H) | (codes == A)
    dead = codes == D

    total = np.concatenate([[0], np.cumsum(fill, dtype=np.int64)])
    long_gaps = [(a, b) for a, b in _gaps(present, blocks) if b - a >= LONG_GAP_FRAMES]
    windows = {"first": [(a, a + EDGE_FRAMES) for a, _ in long_gaps],
               "last": [(b - EDGE_FRAMES, b) for _, b in long_gaps],
               "inside": [(a + EDGE_FRAMES, b - EDGE_FRAMES) for a, b in long_gaps]}
    f3 = {"gaps": len(long_gaps)}
    for name, spans in windows.items():
        f3[name] = _tally(sum(int(total[b] - total[a]) for a, b in spans), sum(b - a for a, b in spans))

    alive_total = np.concatenate([[0], np.cumsum(alive, dtype=np.int64)])
    fragments = {True: [], False: []}
    for _, start, stop in blocks:
        first, last = _runs(fill[start:stop])
        for a, b in zip((first + start).tolist(), (last + start).tolist()):
            fragments[bool(alive_total[b] - alive_total[a] >= ALIVE_SHARE * (b - a))].append(b - a)
    f4 = {name: {"median": float(np.median(found)) if found else None, "n": len(found)}
          for name, found in (("alive", fragments[True]), ("dead", fragments[False]))}

    both = fill & np.isfinite(reference).all(axis=1)
    errors = np.hypot(fill_xy[both, 0] - reference[both, 0], fill_xy[both, 1] - reference[both, 1])
    x, y = fill_xy[:, 0], fill_xy[:, 1]
    beyond = (x < -BEYOND_LINE_M) | (x > length + BEYOND_LINE_M) | (y < -BEYOND_LINE_M) | (y > width + BEYOND_LINE_M)
    return {"F1": _share(fill & alive, gap & alive), "F2": _share(fill & dead, gap & dead), "F3": f3, "F4": f4,
            "F5": {"median_m": float(np.median(errors)) if errors.size else None, "n": int(errors.size)},
            "W1": _share(fill, inside), "W2": _share(gap & alive, gap), "W3": _share(fill & beyond, fill)}


# -- the served inputs (design 3.2 and 3.4) ---------------------------------------------------------------------------

class BroadcastSimError(ValueError):
    """What the wiring refuses: pools that cannot be built as the design says, a config it cannot measure, matches it
    cannot pool. The message says which, and what to change."""


def driver_key(config):
    """The sha256 (hex) of everything run.predict reads of `config`: scope, seed, ball_fill, evidence, decoder, refine.

    Two configs with one key decode to the same codes on the same inputs, whatever their names, paths, descriptions and
    comments, so the driver decoded for one is the driver of the other (ruling 9), and a config that is a twin's content
    (main.toml beside a candidate that is main.toml with the fill) shares its codes with it.
    """
    content = {"scope": config.scope, "seed": config.seed, "ball_fill": config.ball_fill,
               "evidence": [dataclasses.asdict(component) for component in config.evidence],
               "decoder": dataclasses.asdict(config.decoder),
               "refine": [dataclasses.asdict(component) for component in config.refine]}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def driver_slot(config, match_id, profile, seed):
    """Where a config's codes on one served tracking are kept in a `drivers` dict: (driver_key, match, profile,
    seed)."""
    return driver_key(config), match_id, profile, seed


def simulated_ball(config, match, plain, profile, seed, pools, drivers=None):
    """(ball, codes): the simulated broadcast ball of `config` on `plain` (design 3.4), and the driver's codes.

    The driver is the config's twin, dataclasses.replace(config, ball_fill="none"), decoded on `plain` (the MatchInputs
    of `match` without a ball) by run._codes, so an asset the twin cannot have on DFL is the runner's BenchmarkError.
    `drivers`, a dict or None, keeps the codes by driver_slot: a twin met again on the same (match, profile, seed) is
    not decoded again. The ball is simulate's, with the clean ball of the match in metres (DFL's own positions:
    tracking, never its flags) and the match's own generator, default_rng([seed, degrade.index_of(match id), RNG_SALT]).
    """
    twin = dataclasses.replace(config, ball_fill="none")
    slot = driver_slot(twin, match.match_id, profile, seed)
    codes = None if drivers is None else drivers.get(slot)
    if codes is None:
        codes = run._codes(twin, plain)
        if drivers is not None:
            drivers[slot] = codes
    pitch = np.array(match.pitch, dtype=np.float64)
    rng = np.random.default_rng([seed, degrade.index_of(match.match_id), RNG_SALT])
    ball = simulate(np.asarray(plain.atd.ball_present, dtype=bool), np.asarray(match.ball_xy, dtype=np.float64) * pitch,
                    codes, plain.graded, match.pitch, pools, rng)
    return ball, codes


def served_inputs_for(config, match, atd, profile, seed, data_dir, pools, plain=None, drivers=None):
    """The MatchInputs that `config` is served for `match`, its tracking `atd` (profile, seed), as the runner serves it.

    For a config without the fill ([inputs] ball_fill = "none") it is the plain MatchInputs: `plain` itself if given,
    else MatchInputs(match, atd, profile, seed, data_dir), and nothing is decoded. For a config that fills it is a
    MatchInputs of the same tracking with `broadcast_ball` the simulated ball (simulated_ball: the twin's codes on the
    plain inputs drive simulate), served as simulate made it; the config's own fill (Inputs.with_ball_fill, in predict)
    then applies to it unchanged. `pools` is the Pools (load_pools) and is read only for a config that fills. `drivers`
    (a dict, or None) keeps the twin's codes for the next config with the same twin on the same (match, profile, seed).

    ValueError, before anything is decoded, for a `plain` of another match or tracking, and for pools that are not a
    Pools when the config fills.
    """
    who = "broadcast_sim.served_inputs_for"
    if config.ball_fill not in BALL_FILLS:
        raise ValueError("{}: {} has [inputs] ball_fill {!r}, which is not one of {}".format(
            who, config.stem, config.ball_fill, ", ".join(BALL_FILLS)))
    if plain is None:
        plain = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
    elif plain.atd is not atd or plain.game != match.match_id:
        raise ValueError("{}: plain must be the MatchInputs of {} that serves this very atd (MatchInputs(match, atd, "
                         "...)), and it serves {} with another atd or match; pass it, or leave plain out".format(
                             who, match.match_id, plain.game))
    if config.ball_fill == "none":
        return plain
    if not isinstance(pools, Pools):
        raise ValueError("{}: pools must be a Pools to serve {}, which fills the ball ([inputs] ball_fill = {!r}); "
                         "load_pools() reads them, and it is a {}".format(who, config.stem, config.ball_fill,
                                                                          type(pools).__name__))
    ball, _ = simulated_ball(config, match, plain, profile, seed, pools, drivers)
    return match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir, broadcast_ball=ball)


# -- the pools of BAR-ATH (design 3.3) --------------------------------------------------------------------------------

def bar_ath_inputs(data_dir=None):
    """BAR-ATH's Inputs over `data_dir` (default games.DATA_DIR): what `pools`, fidelity's loaders and `bar-ath`
    read."""
    return Inputs(data_dir=data_dir)


def decode_labels(inputs, path, log=print):
    """(codes, sha256): r024_both.toml decoded on `inputs` by pipeline.run.run, in-process, its CSV written at `path`.

    The pipeline's own run, so the labels are exactly the file that `pipeline/run.py pipeline/configs/r024_both.toml`
    writes, and the sha256 is that of the CSV's bytes (RunResult.sha256). `log` takes the run's aggregate lines.
    """
    result = pipeline_run.run(load_config(R024_CONFIG), inputs, path, log=log)
    return result.codes, result.sha256


def near_gap_errors(ball_present, ball_xy_m, broadcast_xy_m, graded):
    """(error_xy, error_starts): the error runs of design 3.3, as Pools holds them (ruling 2).

    ball_present    (n,) bool: our ball (False where it is missing); its gaps are gaps_of(ball_present, graded)
    ball_xy_m       (n, 2): our ball in metres, NaN where it has none
    broadcast_xy_m  (n, 2): the broadcast's usable ball on our rows, in metres, NaN where it has none
    graded          [(half, start, stop), ...], half-open

    A frame qualifies when it is graded, both balls are there (ours present and finite, the broadcast's finite), and it
    is 1 to NEAR_GAP_FRAMES frames before the first frame of a gap of its own block, or after its last. The runs are the
    maximal runs of qualifying frames, cut at the blocks, in time order; each frame's offset is the broadcast's position
    minus ours. error_xy is (m, 2) float64, the runs laid end to end, and error_starts (r + 1,) int64, 0 then the
    running totals of their lengths. ValueError, naming the parameter, for an argument that is not as said.
    """
    who = "broadcast_sim.near_gap_errors"
    present = _flags(who, "ball_present", ball_present)
    n = present.shape[0]
    ours = _positions(who, "ball_xy_m", ball_xy_m, n)
    theirs = _positions(who, "broadcast_xy_m", broadcast_xy_m, n)
    blocks = _blocks(who, graded, n)
    both = present & np.isfinite(ours).all(axis=1) & np.isfinite(theirs).all(axis=1)
    pieces, lengths = [], []
    for half, start, stop in blocks:
        near = np.zeros(stop - start, dtype=bool)
        for first, end in _gaps(present, [(half, start, stop)]):
            near[max(start, first - NEAR_GAP_FRAMES) - start:first - start] = True       # 1 to 50 frames before
            near[end - start:min(stop, end + NEAR_GAP_FRAMES) - start] = True            # 1 to 50 frames after
        begins, ends = _runs(near & both[start:stop])
        for a, b in zip((begins + start).tolist(), (ends + start).tolist()):
            pieces.append(theirs[a:b] - ours[a:b])
            lengths.append(b - a)
    error_xy = np.concatenate(pieces) if pieces else np.zeros((0, 2))
    return error_xy, np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64)


def read_source(path):
    """The bytes of a source file with CRLF read as LF, so that a Windows checkout and a Linux one agree."""
    with open(path, "rb") as handle:
        return handle.read().replace(b"\r\n", b"\n")


def _text(path):
    """A text file's content, read by read_source: a checkout that changes only the line endings changes nothing."""
    return read_source(path).decode("utf-8")


def code_sha256():
    """{path: sha256} of CODE_FILES, read by read_source: the code that builds the pools and measures BAR-ATH's side
    (this file, and pipeline/inputs' broadcast.py and __init__.py, which find the broadcast ball and fill ours)."""
    return {name: hashlib.sha256(read_source(os.path.join(ROOT, *name.split("/")))).hexdigest() for name in CODE_FILES}


def pools_key(data_dir=None, game=games.GAME):
    """Everything the pools and BAR-ATH's measure depend on, as JSON-normal data: a cache is read only if all of it
    matches.

    POOLS_VERSION; the sha256 of the code that builds and measures them (code_sha256), so that a fix to measure_fill or
    to the fill is never measured against BAR-ATH's side as the old code measured it; the tactical and the broadcast
    ATD caches' records (their sources' sizes and sha256, and the cache version, which says how they were parsed);
    the broadcast homography file's size and sha256; r024_both.toml's text (LF) and the sha256 its CSV must have
    (R024_SHA256: the pools are never built from another); and the match's frames, pitch and graded blocks.

    It is read from files without loading BAR-ATH's tracking: the two ATD caches' manifests as they are on disk
    (<data dir>/cache/atd_tactical/ and atd_broadcast/), the homography file hashed, the scalars from games.py.
    BroadcastSimError, saying to run `pools` first (which builds them), if a cache's manifest cannot be read; the
    games.py error if the broadcast homography file is not there.
    """
    data_dir = data_dir or games.DATA_DIR
    cache = os.path.join(data_dir, "cache")
    tactical = _atd_record(os.path.join(cache, atd_cache.FOLDER))
    broadcast_atd = _atd_record(os.path.join(cache, BROADCAST_FOLDER))
    homography = games.require(["homography"], game, data_dir, source="broadcast")[0]
    key = {"pools_version": POOLS_VERSION, "code_sha256": code_sha256(),
           "tactical_atd": tactical, "broadcast_atd": broadcast_atd,
           "broadcast_homography": atd_cache.describe_source(homography),
           "r024_config": _text(R024_CONFIG), "r024_csv_sha256": R024_SHA256,
           "n_frames": int(games.source_frames("tactical", game)),
           "pitch": [float(side) for side in games.GAMES[game]["pitch"]],
           "graded": [[half, first, last + 1] for half, first, last in games.half_video_blocks(game)]}
    return json.loads(json.dumps(key, sort_keys=True))


def _atd_record(folder):
    """{"sources", "cache_version"} of the ATD cache in `folder`, from its manifest alone (no array is read)."""
    path = os.path.join(folder, atd_cache.MANIFEST)
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        return {"sources": manifest["sources"], "cache_version": manifest["cache_version"]}
    except (OSError, ValueError, KeyError, TypeError):
        raise BroadcastSimError("broadcast_sim: there is no readable ATD cache manifest at {}; {}".format(
            path, POOLS_FIRST)) from None


def cache_folder(data_dir, key):
    """<data dir>/cache/benchmark/broadcast_sim/<the first KEY_DIGITS hex digits of the sha256 of the key's canonical
    JSON>/: one folder per key, whatever the order of its entries, as calibration.cache_folder."""
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return os.path.join(data_dir, *CACHE_FOLDER, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:KEY_DIGITS])


def digest(pools):
    """The sha256 (hex) of everything a Pools holds, field by field in POOL_ARRAYS order (its name, dtype, shape and
    bytes): two pools are the same stand-in if and only if their digests agree. Every report of a run that loaded the
    pools holds it (`broadcast_sim`), and choose.py refuses reports of different ones together."""
    sha = hashlib.sha256()
    for name in POOL_ARRAYS:
        array = np.ascontiguousarray(getattr(pools, name))
        sha.update("{}={}{};".format(name, array.dtype, array.shape).encode("ascii"))
        sha.update(array.tobytes())
    return sha.hexdigest()


def build_pools(inputs, log=print):
    """The Pools of BAR-ATH's `inputs` (design 3.3), built and not cached: prepare_pools builds and caches them.

    The labels are decode_labels' (r024_both.toml), and BroadcastSimError, with nothing left behind, if their CSV is
    not R024_SHA256, if a class has no donor gap or if there is no error run.
    """
    return _build(inputs, inputs.data_dir, log)[0]


def prepare_pools(data_dir=None, inputs=None, log=print, rebuild=False):
    """(pools, measure): the cache's when it is whole and of the key, otherwise built from BAR-ATH and written.

    The one way the pools are built: the `pools` command. `inputs` stands in for BAR-ATH's (bar_ath_inputs(data_dir)
    by default; nothing of it is loaded when the cache is read), `data_dir` defaults to the inputs', `log` takes the
    decode's lines, and rebuild=True builds again. The key is read before the build, to find a cache, and again after
    it, since the build loads (and so refreshes) both ATD caches: the cache is written under the key that load_pools
    then reads. An ATD cache with no manifest yet is no cache: the build makes it.
    """
    inputs = bar_ath_inputs(data_dir) if inputs is None else inputs
    data_dir = data_dir or inputs.data_dir
    if not rebuild:
        try:
            key = pools_key(data_dir, inputs.game)
        except BroadcastSimError:
            key = None
        found = None if key is None else _read_cache(cache_folder(data_dir, key), key)
        if found is not None:
            return found
    pools, measure = _build(inputs, data_dir, log)
    key = pools_key(data_dir, inputs.game)
    _write_cache(cache_folder(data_dir, key), key, pools, measure)
    return pools, measure


def load_pools(data_dir=None):
    """The Pools of BAR-ATH, from <data dir>/cache/benchmark/broadcast_sim/<digest of the key>/: the runner's,
    latency's, fidelity's and bar-ath's way to them. It never builds them and never loads BAR-ATH's tracking (pools_key
    reads files): a cache that is absent, broken or of another key is a BroadcastSimError that says to run `pools`
    first.
    """
    return _load_cached(data_dir)[0]


def load_bar_ath_measure(data_dir=None):
    """BAR-ATH's side of fidelity (ruling 1): measure_fill's result on our real fill, kept beside the pools of the same
    key, as plain numbers, with load_pools' rules (read only; `pools` first).

    It is measured on the ball before the fill (ball_present), the fill's own mask (Inputs.with_ball_fill's
    ball_filled), r024_both.toml's labels and the broadcast ball in metres, with no reference (our ball is missing on
    every filled frame), and F5 is the near-gap error pool's median |offset| in metres over all its frames, with their
    count: the design's "F5's ours is the near-gap pool's median".
    """
    return _load_cached(data_dir)[1]


def _load_cached(data_dir):
    """(pools, measure) from the cache of the key, or BroadcastSimError (pools first)."""
    data_dir = data_dir or games.DATA_DIR
    key = pools_key(data_dir)
    folder = cache_folder(data_dir, key)
    found = _read_cache(folder, key)
    if found is None:
        raise BroadcastSimError("broadcast_sim: there is no whole cache of the pools for this data, this code and "
                                "r024_both.toml at {} (absent, broken, or built for another key); {}".format(
                                    folder, POOLS_FIRST))
    return found


def _build(inputs, data_dir, log):
    """(pools, measure) of BAR-ATH: the decode of r024_both.toml, then the pools and the measure.

    The decode writes its CSV (pipeline.run.run writes one) to a temporary name in the cache's folder, and the file is
    deleted as soon as its sha256 is known, whatever it is: the labels are kept in memory only, and no CSV of BAR-ATH's
    labels stays in data/cache/. A refusal or an error leaves nothing behind.
    """
    root = os.path.join(data_dir, *CACHE_FOLDER)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "{}.{}.csv".format(LABELS_STEM, os.getpid()))
    try:
        codes, sha256 = decode_labels(inputs, path, log)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)
    if sha256 != R024_SHA256:
        raise BroadcastSimError(
            "broadcast_sim: the labels of r024_both.toml decoded on this match have sha256 {}, not {}, the file "
            "main.toml writes today; the pools are built from those labels only. Find what changed (the config, the "
            "pipeline's code, the data) before building them".format(sha256, R024_SHA256))
    return _pools_and_measure(inputs, np.asarray(codes))


def _pools_and_measure(inputs, codes):
    """(pools, measure) of BAR-ATH's `inputs` and its decoded `codes`; BroadcastSimError for a class with no donor gap
    or no error run (ruling 10)."""
    present = np.asarray(inputs.atd.ball_present, dtype=bool)
    graded = [tuple(block) for block in inputs.graded]
    pitch = np.array(inputs.pitch, dtype=np.float64)
    gaps = gaps_of(present, graded)
    alive = gap_alive(gaps, codes)
    for flag, name in ((True, "alive"), (False, "D")):
        if not (alive == flag).any():
            raise BroadcastSimError(
                "broadcast_sim: r024_both.toml's labels make no gap of our ball {} (of {} gaps), so a DFL gap of that "
                "class would have no donor; the pools need donors of both classes".format(name, len(gaps)))
    filled = np.asarray(inputs.with_ball_fill("broadcast").ball_filled, dtype=bool)
    broadcast_m = np.asarray(inputs.broadcast_ball, dtype=np.float64) * pitch
    error_xy, error_starts = near_gap_errors(present, np.asarray(inputs.atd.ball_xy, dtype=np.float64) * pitch,
                                             broadcast_m, graded)
    if error_starts.size < 2:
        raise BroadcastSimError(
            "broadcast_sim: there is no error run (no graded frame within {} frames of a gap where both balls exist), "
            "so a simulated ball could not be placed; check the broadcast's alignment and live share".format(
                NEAR_GAP_FRAMES))
    lengths = np.array([stop - start for start, stop in gaps], dtype=np.int64)
    pools = Pools(gap_lengths=lengths, gap_alive=alive,
                  gap_fill=np.concatenate([filled[start:stop] for start, stop in gaps]),
                  gap_starts=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                  error_xy=error_xy, error_starts=error_starts)
    measure = measure_fill(present, filled, codes, graded, broadcast_m, np.full((len(present), 2), np.nan),
                           inputs.pitch)
    distances = np.hypot(error_xy[:, 0], error_xy[:, 1])
    measure["F5"] = {"median_m": float(np.median(distances)), "n": int(distances.size)}
    return pools, measure


def _write_cache(folder, key, pools, measure):
    """The arrays first, then the measure, the manifest last, with the old manifest removed up front (calibration.py's
    order): a run killed half-way leaves no manifest, so the next load builds again."""
    os.makedirs(folder, exist_ok=True)
    manifest = os.path.join(folder, MANIFEST)
    with contextlib.suppress(OSError):
        os.remove(manifest)
    for name in POOL_ARRAYS:
        np.save(os.path.join(folder, name + ".npy"), getattr(pools, name), allow_pickle=False)
    run.write_json(os.path.join(folder, MEASURE_FILE), measure)
    run.write_json(manifest, {"key": key, "digest": digest(pools), "pools_version": POOLS_VERSION})


def _read_cache(folder, key):
    """(pools, measure) from `folder`, or None if its manifest is not of `key`, or anything in it cannot be read whole:
    an unreadable cache is absent, never an error (calibration.load's rule)."""
    try:
        with open(os.path.join(folder, MANIFEST), encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("key") != key:
            return None
        pools = Pools(**{name: _read_array(folder, name) for name in POOL_ARRAYS})
        if digest(pools) != manifest.get("digest"):
            return None
        with open(os.path.join(folder, MEASURE_FILE), encoding="utf-8") as handle:
            measure = json.load(handle)
        if not isinstance(measure, dict) or sorted(measure) != sorted(MEASURE_KEYS):
            return None
        return pools, measure
    except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError, zipfile.BadZipFile):
        return None


def _read_array(folder, name):
    """One cached array, np.load given an open handle (docs/traps.md T34, as calibration._read_array): a file cut short
    is a ValueError here, and no handle is left open for the rebuild that rewrites it."""
    with open(os.path.join(folder, name + ".npy"), "rb") as handle:
        array = np.load(handle, allow_pickle=False)
        if not isinstance(array, np.ndarray):
            close = getattr(array, "close", None)
            if close is not None:
                close()
            raise ValueError("{}.npy is not a plain numpy array".format(name))
        return array


def render_pools(pools, measure):
    """The lines that `pools` prints: counts, shares and medians of the pools and of BAR-ATH's measure, and the digest.
    Aggregates only: no frame, no gap, no offset of its own."""
    lines = ["pools     BAR-ATH's gaps and the broadcast's errors near them (design 3.3), from r024_both.toml's labels "
             "({}...)".format(R024_SHA256[:8])]
    lengths, alive = pools.gap_lengths, pools.gap_alive
    lines.append("gaps      alive {}, D {} ({} in all)".format(int(alive.sum()), int((~alive).sum()), alive.size))
    filled_frames = np.add.reduceat(pools.gap_fill.astype(np.int64), pools.gap_starts[:-1]) if alive.size else alive
    parts = []
    for flag, name in ((True, "alive"), (False, "D")):
        total, filled = int(lengths[alive == flag].sum()), int(filled_frames[alive == flag].sum())
        parts.append("{} {:,} of {:,} gap frames filled ({})".format(name, filled, total, _percent(filled, total)))
    lines.append("donors    " + "; ".join(parts))
    distances = np.hypot(pools.error_xy[:, 0], pools.error_xy[:, 1])
    lines.append("errors    {} error runs, {:,} frames, median offset {} m".format(
        pools.error_starts.size - 1, pools.error_xy.shape[0],
        _number(float(np.median(distances)) if distances.size else None, 2)))
    lines.append("measure   BAR-ATH's own, fidelity's ours:")
    lines += ["  " + line for line in _measure_lines(measure)]
    lines.append("digest    {}".format(digest(pools)))
    return lines


def _measure_lines(measure):
    """One line per statistic of a measure_fill result: its value and what it is over."""
    f3, f4, f5 = measure["F3"], measure["F4"], measure["F5"]
    return ["F1  fill share of the gap frames the labels call alive       {}".format(_share_text(measure["F1"])),
            "F2  fill share of the gap frames the labels call D           {}".format(_share_text(measure["F2"])),
            "F3  fill share in the first 1 s {}, last 1 s {}, inside {}, of {} gaps of 2 s or more".format(
                _number(f3["first"]["share"]), _number(f3["last"]["share"]), _number(f3["inside"]["share"]),
                f3["gaps"]),
            "F4  median fill fragment: alive {} frames (n {}), D {} frames (n {})".format(
                _number(f4["alive"]["median"], 1), f4["alive"]["n"], _number(f4["dead"]["median"], 1),
                f4["dead"]["n"]),
            "F5  median position error {} m over {:,} frames".format(_number(f5["median_m"], 3), f5["n"]),
            "W1  fill frames per graded frame                             {}".format(_share_text(measure["W1"])),
            "W2  share of the gap frames the labels call alive            {}".format(_share_text(measure["W2"])),
            "W3  share of fill frames more than 1 m beyond a line         {}".format(_share_text(measure["W3"]))]


def _share_text(share):
    """A share of measure_fill's as the printouts show it: the share, then its count over what it is of."""
    return "{} ({:,} of {:,})".format(_number(share["share"]), share["num"], share["den"])


def _number(value, decimals=4):
    return "n/a" if value is None else "{:.{}f}".format(value, decimals)


def _percent(num, den):
    return "{:.1%}".format(num / den) if den else "n/a"


# -- fidelity (design 3.5) --------------------------------------------------------------------------------------------

def fidelity(config, profile=run.DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None, calibration=None,
             pools=None, ours=None, data_dir=None, log=print):
    """Design 3.5 for `config` (a pipeline.config.Config that fills): the copies' measure beside BAR-ATH's, judged.

    The config is taken as it runs on DFL: its match-only sources are dropped first (run.drop_match_only, as
    --drop-match-only does, and the names are logged), so the driver, its twin, is the one the runner's bar runs use
    (ruling 4). Each match (match_ids, default all seven, or `matches`, {id: DflMatch}) under each seed of `profile` is
    served as the runner serves it (run.make_atd, then simulated_ball and the MatchInputs that carries it), and a copy
    is its ball's presence before the fill, the filled view's mask, the driver's codes, its graded blocks, the simulated
    ball and DFL's clean ball in metres (ruling 3). The copies are laid end to end, their blocks offset
    (laid_end_to_end), and measured once by measure_fill, since F4 and F5 are medians of the frames, which no sum of
    medians gives. `calibration` defaults to calibration.load(data_dir) for a degraded profile; `pools` and `ours`
    (BAR-ATH's measure) default to the cache (load_pools, load_bar_ath_measure), which `pools` must have built:
    they are read, never built here, and nothing of BAR-ATH's tracking is loaded. Returns fidelity_result.

    BroadcastSimError for a config that does not fill and for matches of different pitches (one pitch measures W3);
    the runner's BenchmarkError for a scope, profile, number of seeds or match that it refuses.
    """
    config, dropped = run.drop_match_only(config)
    log("dropped   match-only sources: {}".format(", ".join(dropped) or "none"))
    run.check_scope(config)
    if config.ball_fill == "none":
        raise BroadcastSimError(
            "{}: [inputs] ball_fill is \"none\", so there is no simulated fill to measure; fidelity measures a config "
            "that fills ([inputs] ball_fill = \"broadcast\")".format(config.stem))
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    if profile == run.CLEAN and calibration is not None:
        raise run.BenchmarkError("profile {!r} takes no calibration, and one was given".format(profile))
    pools = load_pools(data_dir) if pools is None else pools            # read only: `pools` builds them
    ours = load_bar_ath_measure(data_dir) if ours is None else ours
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    if not isinstance(pools, Pools):
        raise BroadcastSimError("pools must be a Pools (load_pools() reads them), and it is a {}".format(
            type(pools).__name__))
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    copies, pitch = [], None
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        own = tuple(float(side) for side in match.pitch)
        if pitch is not None and own != pitch:
            raise BroadcastSimError("the matches have different pitches ({} and {}); W3 is measured against one pitch, "
                                    "so they cannot be pooled".format(pitch, own))
        pitch = own
        scale = np.array(pitch)
        reference = np.asarray(match.ball_xy, dtype=np.float64) * scale
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            plain = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            ball, codes = simulated_ball(config, match, plain, profile, seed, pools)
            served = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir,
                                              broadcast_ball=ball)
            copies.append({"ball_present": np.asarray(atd.ball_present, dtype=bool),
                           "filled": np.asarray(served.with_ball_fill("broadcast").ball_filled, dtype=bool),
                           "codes": np.asarray(codes), "graded": list(plain.graded), "fill_xy_m": ball * scale,
                           "reference_xy_m": reference})
            del atd, plain, served
        log("match     {}   {:,} rows, {} seed(s) simulated".format(match.match_id, match.n_frames, len(seed_list)))
        del match
    sim = measure_fill(*laid_end_to_end(copies), pitch)
    calibration_info = None if profile == run.CLEAN else {"sha256": calibration_module.digest(calibration),
                                                          "version": calibration_module.CALIBRATION_VERSION}
    return fidelity_result(config=config, dropped=dropped, profile=profile, seeds=seed_list, match_ids=ids,
                           calibration=calibration_info, digest=digest(pools), copies=len(copies), ours=ours, sim=sim)


def laid_end_to_end(copies):
    """(ball_present, filled, codes, graded, fill_xy_m, reference_xy_m) of `copies` laid one after the other: each
    copy's arrays in turn, and its graded blocks moved by the frames of the copies before it. measure_fill's arguments
    but the pitch, so that the copies are pooled by frames."""
    graded, offset = [], 0
    for copy in copies:
        graded += [(half, start + offset, stop + offset) for half, start, stop in copy["graded"]]
        offset += len(copy["ball_present"])
    return (np.concatenate([copy["ball_present"] for copy in copies]),
            np.concatenate([copy["filled"] for copy in copies]), np.concatenate([copy["codes"] for copy in copies]),
            graded, np.concatenate([copy["fill_xy_m"] for copy in copies]),
            np.concatenate([copy["reference_xy_m"] for copy in copies]))


def _compared(ours, sim, tolerance):
    """{"ours", "sim", "difference", "tolerance", "passed"}: passed when both exist and |sim - ours| <= tolerance."""
    known = ours is not None and sim is not None
    return {"ours": ours, "sim": sim, "difference": sim - ours if known else None, "tolerance": tolerance,
            "passed": bool(known and abs(sim - ours) <= tolerance)}


def judge_fidelity(ours, sim):
    """The design's 3.5 tables from two measure_fill results: {"gates", "checks", "warnings", "verdict", "exit"}.

    Gates: F1, F2 and F3's first, last and inside shares, each |sim - ours| within GATE_TOLERANCES. Checks: F4's median
    fragment by class within 10 frames, F5's median error within 0.2 m (CHECK_TOLERANCES). A statistic that is None on
    either side fails its gate or check (ruling 6). Warnings, W1 to W3, are reported and decide nothing. The verdict is
    "gate" (exit 1) if a gate fails, whatever the checks, else "check" (exit 3) if a check fails, else "pass" (0).
    """
    gates = {"F1": _compared(ours["F1"]["share"], sim["F1"]["share"], GATE_TOLERANCES["F1"]),
             "F2": _compared(ours["F2"]["share"], sim["F2"]["share"], GATE_TOLERANCES["F2"])}
    for part in ("first", "last", "inside"):
        gates["F3 " + part] = _compared(ours["F3"][part]["share"], sim["F3"][part]["share"], GATE_TOLERANCES["F3"])
    checks = {"F4 " + name: _compared(ours["F4"][name]["median"], sim["F4"][name]["median"], CHECK_TOLERANCES["F4"])
              for name in ("alive", "dead")}
    checks["F5"] = _compared(ours["F5"]["median_m"], sim["F5"]["median_m"], CHECK_TOLERANCES["F5"])
    warnings = {name: {"ours": ours[name]["share"], "sim": sim[name]["share"]} for name in ("W1", "W2", "W3")}
    if not all(row["passed"] for row in gates.values()):
        verdict = "gate"
    elif not all(row["passed"] for row in checks.values()):
        verdict = "check"
    else:
        verdict = "pass"
    return {"gates": gates, "checks": checks, "warnings": warnings, "verdict": verdict,
            "exit": FIDELITY_EXIT[verdict]}


def fidelity_result(config, dropped, profile, seeds, match_ids, calibration, digest, copies, ours, sim):
    """The JSON-ready result of a fidelity run (fidelity_path's report holds it, run.plain): the config as run, what
    was dropped, the profile's record, the seeds, the matches, the Calibration's {sha256, version} (None for
    clean), the pools' digest, the number of copies, the two measures and judge_fidelity's tables and exit."""
    return dict({"config": {"path": run.shown(config.path), "text": config.text.replace("\r\n", "\n")},
                 "stem": config.stem, "dropped_match_only": list(dropped), "profile": degrade.profile_record(profile),
                 "seeds": list(seeds), "matches": sorted(match_ids), "calibration": calibration,
                 "broadcast_sim": digest, "copies": copies, "ours": ours, "sim": sim}, **judge_fidelity(ours, sim))


def fidelity_path(data_dir, stem, profile, seeds=None):
    """<data dir>/runs/r028/fidelity-<config stem>-<profile>[.s<n>].json: the config's name, so that R028b's run
    never replaces rung 1's, and `.s<n>` for a number of seeds other than the default, as the runner's reports, so
    that a one-seed look never replaces the three-seed report."""
    name = ".".join(["fidelity-{}-{}".format(stem, profile)] + run.name_suffixes(profile, None, seeds)) + ".json"
    return os.path.join(data_dir, *REPORT_FOLDER, name)


FIDELITY_ROWS = {"F1": "fill share of the gap frames called alive", "F2": "fill share of the gap frames called D",
                 "F3 first": "fill share in a long gap's first 1 s", "F3 last": "fill share in a long gap's last 1 s",
                 "F3 inside": "fill share inside a long gap", "F4 alive": "median alive fill fragment, frames",
                 "F4 dead": "median D fill fragment, frames", "F5": "median position error, m",
                 "W1": "fill frames per graded frame", "W2": "share of the gap frames called alive",
                 "W3": "share of fill frames beyond a line by > 1 m"}
FIDELITY_VERDICTS = {"pass": "every gate and check passes: the stand-in is ours",
                     "gate": "a gate fails: STOP for the advisor (design 3.5)",
                     "check": "a check fails: STOP for the advisor (design 3.5)"}


def render_fidelity(result):
    """The lines that `fidelity` prints: the run, every gate, check and warning (ours, the copies', the difference, the
    tolerance) and the verdict. Aggregates only."""
    profile = result["profile"]
    lines = ["fidelity  {}   profile {} (version {})   seeds {}   {} matches, {} copies".format(
                 result["stem"], profile["name"], profile["version"], ",".join(str(s) for s in result["seeds"]),
                 len(result["matches"]), result["copies"]),
             "dropped   match-only sources: {}".format(", ".join(result["dropped_match_only"]) or "none"),
             "pools     broadcast_sim {}".format(result["broadcast_sim"]), "",
             "{:<6}{:<10}{:<44}{:>9}{:>9}{:>11}{:>11}".format("", "", "", "ours", "copies", "difference",
                                                              "tolerance")]
    for kind, rows in (("gate", result["gates"]), ("check", result["checks"])):
        for name, row in rows.items():
            lines.append("{:<6}{:<10}{:<44}{:>9}{:>9}{:>11}{:>11}  {}".format(
                kind, name, FIDELITY_ROWS[name], _number(row["ours"]), _number(row["sim"]),
                "n/a" if row["difference"] is None else "{:+.4f}".format(row["difference"]),
                "{:g}".format(row["tolerance"]), "pass" if row["passed"] else "FAIL"))
    for name, row in result["warnings"].items():
        lines.append("{:<6}{:<10}{:<44}{:>9}{:>9}".format("warn", name, FIDELITY_ROWS[name], _number(row["ours"]),
                                                          _number(row["sim"])))
    lines += ["", "verdict   {} (exit {})".format(FIDELITY_VERDICTS[result["verdict"]], result["exit"])]
    return lines


# -- BAR-ATH, label-free (design 7.4) ---------------------------------------------------------------------------------

def judge_alignment(summary, refused=None):
    """The alignment band and the live share of design 7.4, from Inputs.broadcast_summary (or None, with `refused` the
    AlignmentError's message): d_1 and d_2 within OFFSET_BAND of EXPECTED_OFFSETS, and the live share at least
    MIN_LIVE_SHARE. {"offsets", "expected", "band", "within", "live_share", "min_live_share", "live_passed",
    "refused", "summary", "passed"}: passed only if the alignment was found, every half is in its band and the share is
    high enough."""
    base = {"expected": dict(EXPECTED_OFFSETS), "band": OFFSET_BAND, "min_live_share": MIN_LIVE_SHARE,
            "refused": refused}
    if summary is None:
        return dict(base, offsets={}, within={half: False for half in EXPECTED_OFFSETS}, live_share=None,
                    live_passed=False, summary=None, passed=False)
    offsets = {int(half): int(value) for half, value in summary.get("offsets", {}).items()}
    within = {half: half in offsets and abs(offsets[half] - expected) <= OFFSET_BAND
              for half, expected in EXPECTED_OFFSETS.items()}
    live = summary.get("live_share")
    live_passed = live is not None and live >= MIN_LIVE_SHARE
    return dict(base, offsets=offsets, within=within, live_share=live, live_passed=live_passed,
                summary=run.plain(summary), passed=bool(refused is None and all(within.values()) and live_passed))


def tag_check(inputs, filled):
    """The tag check of design 7.4 (ruling 7): Smart Tagging's agreement on the frames that only the fill lets the
    teacher speak on, against its agreement on the plain inputs.

    The teacher is controlled_touch with conventions.TOUCH (kept episodes only, so no hold vote is counted), run once on
    `inputs` (BAR-ATH, plain) and once on `filled` (inputs.with_ball_fill("broadcast")). The fill-only frames are the
    graded ones where its `home` is a number on the filled inputs and NaN on the plain ones. Both agreements are
    tag_stats.team_agreement with the team votes and tags of smart_tagging: on all graded frames for the plain teacher,
    on the fill-only frames for the filled one. judge_tags says what that decides.
    """
    plain_home = np.asarray(controlled_touch.compute(RunContext(inputs=inputs, seed=0), **conventions.TOUCH).home,
                            dtype=np.float64)
    filled_home = np.asarray(controlled_touch.compute(RunContext(inputs=filled, seed=0), **conventions.TOUCH).home,
                             dtype=np.float64)
    graded = np.asarray(inputs.graded_mask, dtype=bool)
    fill_only = graded & np.isfinite(filled_home) & np.isnan(plain_home)
    team, tags = smart_tagging.votes(inputs), smart_tagging.tags(inputs)
    missing = ~np.asarray(inputs.atd.ball_present, dtype=bool)
    plain = tag_stats.team_agreement(team, plain_home, tags, graded=graded, ball_missing=missing)
    fill = tag_stats.team_agreement(team, filled_home, tags, graded=fill_only, ball_missing=missing)
    return judge_tags(plain, fill, int(fill_only.sum()))


def judge_tags(plain, fill_only, fill_only_frames):
    """{"plain", "fill_only", "fill_only_frames", "bar", "margin", "min_frames", "decisive", "passed"}: the bar is the
    plain agreement minus TAG_MARGIN; the check decides only when at least MIN_TAG_FRAMES fill-only frames have a tag's
    vote (fill_only["n"]), and then passes when the fill-only agreement is at or above the bar (an agreement or a bar
    that is None fails). Below that many frames it is reported and passes, deciding nothing."""
    bar = None if plain.get("share") is None else plain["share"] - TAG_MARGIN
    decisive = fill_only["n"] >= MIN_TAG_FRAMES
    share = fill_only.get("share")
    passed = (not decisive) or (bar is not None and share is not None and share >= bar)
    return {"plain": plain, "fill_only": fill_only, "fill_only_frames": int(fill_only_frames), "bar": bar,
            "margin": TAG_MARGIN, "min_frames": MIN_TAG_FRAMES, "decisive": bool(decisive), "passed": bool(passed)}


def judge_sanity(codes, graded):
    """R024's sanity of the candidate's labels: no X on a graded frame, and D within D_SHARE_BAND of the graded frames,
    both ends included. {"graded_frames", "x_graded", "d_share", "d_band", "passed"}."""
    codes = np.asarray(codes)
    mask = np.zeros(codes.shape[0], dtype=bool)
    for _, start, stop in graded:
        mask[start:stop] = True
    frames = int(mask.sum())
    x_graded = int((codes[mask] == X).sum())
    d_share = float((codes[mask] == D).sum()) / frames if frames else None
    low, high = D_SHARE_BAND
    passed = d_share is not None and x_graded == 0 and low <= d_share <= high
    return {"graded_frames": frames, "x_graded": x_graded, "d_share": d_share, "d_band": list(D_SHARE_BAND),
            "passed": bool(passed)}


def bar_ath_exit(alignment, tag, sanity):
    """3 (a STOP) if the alignment band or the live share fails, whatever else; else 1 (a DROP) if the tag check or the
    sanity fails (or is missing); else 0 (ruling 6)."""
    if not alignment["passed"]:
        return BAR_ATH_EXIT["stop"]
    if not (tag and tag["passed"]) or not (sanity and sanity["passed"]):
        return BAR_ATH_EXIT["drop"]
    return BAR_ATH_EXIT["pass"]


def bar_ath_result(config, alignment, digest, fill, tag, sanity, reported):
    """The JSON-ready result of bar-ath (bar-ath-<config stem>.json holds it): the parts as measured, the stage reached
    ("alignment" at a STOP, else "complete"), the verdict and bar_ath_exit's code."""
    code = bar_ath_exit(alignment, tag, sanity)
    verdict = {value: name for name, value in BAR_ATH_EXIT.items()}[code]
    return {"config": {"path": run.shown(config.path), "text": config.text.replace("\r\n", "\n")},
            "stem": config.stem, "stage": "complete" if alignment["passed"] else "alignment", "alignment": alignment,
            "broadcast_sim": digest, "fill": fill, "tag": tag, "sanity": sanity, "reported": reported,
            "verdict": verdict, "exit": code}


def bar_ath(config, inputs=None, data_dir=None, pools=None, log=print):
    """Design 7.4 on BAR-ATH for `config`, a config that fills: bar_ath_result of what was measured.

    1. The alignment that the run finds (Inputs.broadcast_summary: the broadcast is loaded, aligned and its live and
       usable frames found here) against the band, and the live share (judge_alignment). An AlignmentError is a
       refused alignment. If either fails, nothing else is run: a STOP (exit 3).
    2. The tag check (tag_check) on the plain inputs and their filled view.
    3. The candidate's labels, the config through the benchmark's predict (its own fill applied there), and their
       sanity (judge_sanity); main.toml's labels beside them for what is reported: the filled frames by the
       candidate's state, W1 to W3 on BAR-ATH, the possessions per half with a WARN beyond POSSESSIONS_WARN of
       main.toml's, and label_diff's figures (the graded frames relabelled among them).
    `inputs` defaults to bar_ath_inputs(data_dir), `pools` to load_pools (read only: `pools` first; its digest goes
    into the result). The fill's added time is logged, never kept. BroadcastSimError for a config that does not fill.
    """
    if config.ball_fill == "none":
        raise BroadcastSimError(
            "{}: [inputs] ball_fill is \"none\"; bar-ath checks a config that fills ([inputs] ball_fill = "
            "\"broadcast\") on BAR-ATH".format(config.stem))
    inputs = bar_ath_inputs(data_dir) if inputs is None else inputs
    inputs.atd                                       # the tactical tracking: every run's cost, not the fill's
    started = time.perf_counter()
    try:
        summary, refused = inputs.broadcast_summary, None
    except AlignmentError as error:
        summary, refused = None, str(error)
    seconds = time.perf_counter() - started
    alignment = judge_alignment(summary, refused)
    log("alignment {}".format("in the band, live share high enough" if alignment["passed"] else
                              "outside the band, refused or too few live frames: STOP"))
    if not alignment["passed"]:
        return bar_ath_result(config=config, alignment=alignment, digest=None, fill=None, tag=None, sanity=None,
                              reported=None)
    pools = load_pools(data_dir or inputs.data_dir) if pools is None else pools
    started = time.perf_counter()
    filled = inputs.with_ball_fill("broadcast")
    mask = np.asarray(filled.ball_filled, dtype=bool)
    seconds += time.perf_counter() - started
    fill = {"filled_frames": int(mask.sum()), "graded_frames": int(np.asarray(inputs.graded_mask).sum())}
    log("fill      {:,} frames filled of {:,} graded; added {:.1f} s (the broadcast loaded, aligned, its live frames "
        "found, the fill)".format(fill["filled_frames"], fill["graded_frames"], seconds))
    tag = tag_check(inputs, filled)
    codes = run.predict(config, inputs)[0]
    sanity = judge_sanity(codes, inputs.graded)
    main_codes = run.predict(load_config(MAIN_CONFIG), inputs)[0]
    reported = _reported(inputs, mask, codes, main_codes)
    return bar_ath_result(config=config, alignment=alignment, digest=digest(pools), fill=fill, tag=tag,
                          sanity=sanity, reported=reported)


def _reported(inputs, filled, codes, main_codes):
    """What design 7.4 reports and decides nothing with: the filled frames by the candidate's state, W1 to W3 of the
    real fill under its labels, the possessions per half beside main.toml's (a WARN beyond POSSESSIONS_WARN of them),
    and label_diff's figures of the candidate against main.toml."""
    present = np.asarray(inputs.atd.ball_present, dtype=bool)
    pitch = np.array(inputs.pitch, dtype=np.float64)
    state = measure_fill(present, filled, codes, inputs.graded, np.asarray(inputs.broadcast_ball) * pitch,
                         np.full((inputs.n_frames, 2), np.nan), inputs.pitch)
    diff = label_diff.diff(main_codes, codes, inputs)
    possessions = {half: {"main": before, "candidate": after,
                          "warn": bool(abs(after - before) > POSSESSIONS_WARN * before)}
                   for half, (before, after) in sorted(diff["possessions"].items())}
    total = int(filled.sum())
    alive, dead = state["F1"]["num"], state["F2"]["num"]
    return {"filled_by_state": {"alive": alive, "dead": dead, "other": total - alive - dead},
            "W1": state["W1"], "W2": state["W2"], "W3": state["W3"], "possessions": possessions,
            "relabelled_share": diff["changed_share"], "label_diff": diff}


def bar_ath_path(data_dir, stem):
    """<data dir>/runs/r028/bar-ath-<config stem>.json."""
    return os.path.join(data_dir, *REPORT_FOLDER, "bar-ath-{}.json".format(stem))


BAR_ATH_VERDICTS = {"pass": "every check passes", "drop": "a check fails: DROP (design 7.4)",
                    "stop": "the alignment band or the live share fails: STOP for the advisor (design 7.4)"}


def render_bar_ath(result):
    """The lines that `bar-ath` prints: the alignment and live share as measured, then (past them) the tag check, the
    sanity, the reported figures and the verdict. Aggregates only."""
    alignment = result["alignment"]
    lines = ["bar-ath   {}: design 7.4 on BAR-ATH, label-free".format(result["stem"])]
    if alignment["refused"] is not None:
        lines.append("alignment refused: {}".format(alignment["refused"]))
    for half, expected in sorted(alignment["expected"].items()):
        found = alignment["offsets"].get(half)
        lines.append("alignment d_{} {} (expected {:,} +- {}): {}".format(
            half, "none" if found is None else "{:,}".format(found), expected, alignment["band"],
            "in the band" if alignment["within"][half] else "OUTSIDE"))
    lines.append("live      {} of the valid graded frames (at least {}): {}".format(
        _number(alignment["live_share"]), alignment["min_live_share"], "pass" if alignment["live_passed"] else "FAIL"))
    if result["fill"] is not None:
        lines.append("fill      {:,} frames filled of {:,} graded".format(result["fill"]["filled_frames"],
                                                                         result["fill"]["graded_frames"]))
    tag = result["tag"]
    if tag is not None:
        lines.append("tag       plain teacher: n {:,}, agreement {}; fill-only: {:,} frames, {:,} with a vote, "
                     "agreement {}; bar {}: {}".format(
                         tag["plain"]["n"], _number(tag["plain"]["share"]), tag["fill_only_frames"],
                         tag["fill_only"]["n"], _number(tag["fill_only"]["share"]), _number(tag["bar"]),
                         ("pass" if tag["passed"] else "FAIL") if tag["decisive"] else
                         "reported only (fewer than {} frames with a vote)".format(tag["min_frames"])))
    sanity = result["sanity"]
    if sanity is not None:
        lines.append("sanity    X on {:,} graded frames; D {} of the graded frames (band {:.0%} to {:.0%}): {}".format(
            sanity["x_graded"], "n/a" if sanity["d_share"] is None else "{:.1%}".format(sanity["d_share"]),
            sanity["d_band"][0], sanity["d_band"][1], "pass" if sanity["passed"] else "FAIL"))
    reported = result["reported"]
    if reported is not None:
        state = reported["filled_by_state"]
        lines.append("reported  filled frames by the candidate's labels: alive {:,}, D {:,}, X {:,}; W1 {}, W2 {}, "
                     "W3 {}".format(state["alive"], state["dead"], state["other"], _number(reported["W1"]["share"]),
                                    _number(reported["W2"]["share"]), _number(reported["W3"]["share"])))
        for half, row in reported["possessions"].items():
            lines.append("          possessions half {}: main.toml {:,}, candidate {:,}{}".format(
                half, row["main"], row["candidate"],
                "   WARN: beyond {:.0%} of main.toml's (T33)".format(POSSESSIONS_WARN) if row["warn"] else ""))
        lines.append("          relabelled {:.2%} of the graded frames".format(reported["relabelled_share"]))
    if result["broadcast_sim"] is not None:
        lines.append("pools     broadcast_sim {}".format(result["broadcast_sim"]))
    lines.append("verdict   {} (exit {})".format(BAR_ATH_VERDICTS[result["verdict"]], result["exit"]))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="R028's simulated broadcast: build the pools from BAR-ATH (pools), measure the stand-in against "
                    "our match (fidelity), and check a config that fills on BAR-ATH (bar-ath).")
    commands = parser.add_subparsers(dest="command", required=True)
    pools = commands.add_parser("pools", help="build or read the pools of BAR-ATH and its measure; aggregates only")
    pools.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    measured = commands.add_parser(
        "fidelity", help="design 3.5: the copies against BAR-ATH; exit 0, 1 (a gate fails), 3 (a check fails), 2 "
                         "(an error)")
    measured.add_argument("config", metavar="CONFIG",
                          help="a config that fills, e.g. pipeline/benchmark/configs/r028/r028_preshift.toml (its "
                               "match-only sources are dropped)")
    measured.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    measured.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                          help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    measured.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    checked = commands.add_parser(
        "bar-ath", help="design 7.4 on BAR-ATH: exit 0, 1 (a check fails: DROP), 3 (the alignment band or the live "
                        "share: STOP), 2 (error)")
    checked.add_argument("config", metavar="CONFIG", help="the candidate, a config that fills")
    checked.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or games.DATA_DIR
    try:
        if args.command == "pools":
            pools, measure = prepare_pools(args.data_dir, inputs=bar_ath_inputs(args.data_dir))
            for line in render_pools(pools, measure):
                print(line)
            return 0
        config = load_config(args.config)
        if args.command == "fidelity":
            result = fidelity(config, profile=args.profile, seeds=args.seeds, data_dir=args.data_dir, log=print)
            path = fidelity_path(data_dir, config.stem, args.profile, args.seeds)
            lines = render_fidelity(result)
        else:
            result = bar_ath(config, inputs=bar_ath_inputs(args.data_dir), data_dir=args.data_dir, log=print)
            path = bar_ath_path(data_dir, config.stem)
            lines = render_bar_ath(result)
        run.write_json(path, run.plain(result))
    except (ConfigError, BroadcastSimError, run.BenchmarkError, AlignmentError, match_inputs.AssetUnavailable,
            OSError) as error:
        # The expected failures get a sentence: a config to fix, labels that are not R024's, data not there, a
        # broadcast that does not align. An error is never a verdict: exit 2, whatever it was.
        print("FAILED    {}".format(error))
        return EXIT_ERROR
    except Exception:          # a bug: its traceback, and still exit 2, so that it cannot be read as a gate's 1
        traceback.print_exc()
        print("FAILED    an unexpected error (its traceback is above); this is not a verdict")
        return EXIT_ERROR
    print("")
    for line in lines:
        print(line)
    print("report    {}".format(run.shown(path)))
    return result["exit"]


if __name__ == "__main__":
    sys.exit(main())
