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
"their pattern, not their clock"): the fill masks and the position errors are COPIED from BAR-ATH, not modelled. This
file is the core on arrays and nothing else: it loads no data and knows no match. Task 2b builds the Pools from
BAR-ATH, serves the simulated ball to the benchmark's runs and adds the command line.

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
"""

import math
import operator
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

import possession_metrics as pm

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
