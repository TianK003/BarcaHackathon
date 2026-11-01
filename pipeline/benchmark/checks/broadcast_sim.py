"""Checks of broadcast_sim (pipeline/benchmark/broadcast_sim.py): R028's simulated broadcast ball, on synthetic arrays.

The core is pure functions on arrays, so every check builds its own small world: a ball-present mask with gaps whose
places are known, a clean ball whose every position is different (so that an added offset is found again), codes, and a
Pools made from lists. What is checked, and how:

  * gaps_of and gap_alive: the runs worked out by hand on blocks that cut gaps, the 0.5 threshold at its edge, and what
    they refuse;
  * simulate: the donor's mask copied, stretched and compressed by the integer rule (against a plain loop); the donor
    drawn among the 5 nearest lengths by the exact log rule, with ties to the shorter and then the earlier (read through
    a generator that answers a chosen index, and over 300 seeds against a second generator of the same seed); by class;
    the error runs added, drawn as the text says and tiled when none is long enough; NaN where DFL has no ball; the
    same bytes for the same seed; and the whole of it against a plain-Python reference on random worlds;
  * measure_fill: every F and W value on hand-built masks, worked out by hand; None for what has no frames; what it
    refuses; and the lot against a plain loop on random worlds;
  * Pools: the arrays, the validation by field, read-only, and a repr with no array in it.

Nothing here reads data, and nothing prints an array.
"""

import functools
import inspect
import math
import statistics

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import broadcast_sim
from pipeline.selftest import expect_error

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
PITCH = (100.0, 50.0)           # (length, width) in metres: dyadic, so that a position over the pitch is exact
SCALE = np.array(PITCH)
SEEDS = range(300)              # the seeds that the donor draw is read over


# -- worlds -----------------------------------------------------------------------------------------------------------

def world(n, missing, alive_in=()):
    """(ball_present, clean_ball_m, codes) of n frames.

    The ball is missing on the (start, stop) spans of `missing`. Every clean position is different, and dyadic
    (10 + i / 4, 5 + i / 8), so that adding an offset and dividing by the pitch is exact. The codes are D, and H on the
    spans of `alive_in`."""
    present = np.ones(n, dtype=bool)
    for start, stop in missing:
        present[start:stop] = False
    steps = np.arange(n, dtype=np.float64)
    clean = np.column_stack([10.0 + 0.25 * steps, 5.0 + 0.125 * steps])
    codes = np.full(n, D, dtype=np.int8)
    for start, stop in alive_in:
        codes[start:stop] = H
    return present, clean, codes


def zero_run(length=2000):
    """An error run of `length` zero offsets: a fill that sits exactly on the clean ball."""
    return [(0.0, 0.0)] * length


def error_run(j, length):
    """Run j of a made-up error pool: offsets, in metres, that say which run they are from and which step."""
    return [(j + 1 + 0.25 * t, -(j + 1) - 0.125 * t) for t in range(length)]


def pools_of(donors, runs=None):
    """A Pools from [(alive, mask)] donors and error runs (default: one run of 2000 zero offsets, so no error)."""
    return broadcast_sim.Pools.from_lists([mask for _, mask in donors], [alive for alive, _ in donors],
                                          [zero_run()] if runs is None else runs)


def simulated(present, clean, codes, graded, pools, rng=None, seed=0):
    return broadcast_sim.simulate(present, clean, codes, graded, PITCH, pools,
                                  np.random.default_rng(seed) if rng is None else rng)


def filled_of(out):
    """The frames that carry a position (both coordinates finite); a position of one coordinate must not happen."""
    finite = np.isfinite(out)
    assert (finite.all(axis=1) == finite.any(axis=1)).all(), "a position with one coordinate missing"
    return finite.all(axis=1)


def mapped(mask, m):
    """A donor's mask on a gap of m frames by the rule fill[i] = mask[floor(i * k / m)], in plain integers."""
    k = len(mask)
    return [bool(mask[(i * k) // m]) for i in range(m)]


class Pick(np.random.Generator):
    """A generator whose integers(n) answers `index` (the last option, if there are fewer) and records every n asked."""

    def __init__(self, index):
        super().__init__(np.random.PCG64(0))
        self.index, self.asked = index, []

    def integers(self, high, *args, **kwargs):
        self.asked.append(int(high))
        return min(self.index, int(high) - 1)


def random_masks(lengths, seed):
    """One random mask per length, as lists of bools."""
    rng = np.random.default_rng(seed)
    return [[bool(flag) for flag in rng.random(k) < 0.5] for k in lengths]


def which(donor_masks, m, out, start):
    """The index of the donor whose mask, mapped onto m frames, is the fill that `out` has from `start`; exactly one."""
    seen = filled_of(out)[start:start + m].tolist()
    found = [j for j, mask in enumerate(donor_masks) if mapped(mask, m) == seen]
    assert len(found) == 1, "the fill is the mapped mask of {} donors: {}".format(len(found), found)
    return found[0]


def distinct_after_mapping(masks, m):
    assert len({tuple(mapped(mask, m)) for mask in masks}) == len(masks), \
        "the made-up donors must look different on the gap, or the check cannot tell them apart"


# -- gaps_of and gap_alive --------------------------------------------------------------------------------------------

def test_gaps_are_cut_at_graded_blocks():
    """The maximal runs without the ball inside each graded block: cut at the block's edges, in time order."""
    graded = [(1, 10, 40), (2, 60, 90)]
    present = np.ones(130, dtype=bool)
    # (0, 3) is before block 1 and (100, 110) after block 2: no gap. (5, 15) runs into block 1, (35, 65) runs out of it,
    # over the break, and into block 2, so it is two gaps; (85, 95) runs out of block 2.
    for start, stop in [(0, 3), (5, 15), (20, 25), (35, 65), (85, 95), (100, 110)]:
        present[start:stop] = False
    expected = [(10, 15), (20, 25), (35, 40), (60, 65), (85, 90)]
    found = broadcast_sim.gaps_of(present, graded)
    assert found == expected, "the gaps are {}".format(found)
    assert all(type(first) is int and type(stop) is int for first, stop in found), "plain ints, not numpy scalars"
    assert broadcast_sim.gaps_of(present, graded[::-1]) == expected, "in time order, whatever the order of the blocks"
    assert broadcast_sim.gaps_of(present, tuple(graded)) == expected

    # A block with no ball is one gap, one with the ball everywhere has none, and the first and last frames count.
    nothing = np.zeros(50, dtype=bool)
    assert broadcast_sim.gaps_of(nothing, [(1, 10, 20), (2, 30, 50)]) == [(10, 20), (30, 50)]
    assert broadcast_sim.gaps_of(np.ones(50, dtype=bool), [(1, 10, 20)]) == []
    edges = np.ones(50, dtype=bool)
    edges[10] = edges[19] = False
    assert broadcast_sim.gaps_of(edges, [(1, 10, 20)]) == [(10, 11), (19, 20)]
    # Two blocks that touch do not merge a gap that runs across them: a gap belongs to one block.
    assert broadcast_sim.gaps_of(nothing, [(1, 10, 20), (2, 20, 30)]) == [(10, 20), (20, 30)]


def test_gap_alive_threshold():
    """A gap is alive if at least half of its frames are H or A: exactly 0.5 is alive, 0.49 is not, and an X is not."""
    def one_gap(alive_frames, length=100, other=D, alive_code=H):
        codes = np.full(length, other, dtype=np.int8)
        codes[:alive_frames] = alive_code
        return broadcast_sim.gap_alive([(0, length)], codes).tolist()

    assert one_gap(50) == [True], "0.5 exactly counts as alive"
    assert one_gap(49) == [False], "0.49 does not"
    assert one_gap(51) == [True] and one_gap(0) == [False] and one_gap(100) == [True]
    assert one_gap(50, other=X) == [True] and one_gap(49, other=X) == [False], "an X is not alive"
    assert one_gap(50, alive_code=A) == [True] and one_gap(49, alive_code=A) == [False]
    codes = np.full(100, D, dtype=np.int8)
    codes[:30], codes[30:50] = H, A
    assert broadcast_sim.gap_alive([(0, 100)], codes).tolist() == [True], "H and A both count: 30 + 20 of 100"
    codes[49] = D
    assert broadcast_sim.gap_alive([(0, 100)], codes).tolist() == [False], "30 + 19 of 100"
    # Odd lengths, and the frames outside the gap do not count.
    assert one_gap(2, length=3) == [True] and one_gap(1, length=3) == [False]
    assert one_gap(1, length=2) == [True] and one_gap(1, length=1) == [True] and one_gap(0, length=1) == [False]

    codes = np.full(200, D, dtype=np.int8)
    codes[:10] = H                    # outside the gaps below
    codes[30:45] = A                  # the second gap: 15 of 30
    codes[70] = H                     # the third gap, one frame
    codes[150:160] = H                # outside
    alive = broadcast_sim.gap_alive([(10, 20), (30, 60), (70, 71), (100, 140)], codes)
    assert alive.tolist() == [False, True, True, False] and alive.dtype == np.bool_ and alive.shape == (4,)
    empty = broadcast_sim.gap_alive([], codes)
    assert empty.shape == (0,) and empty.dtype == np.bool_, "no gap, no flag"


def test_gaps_of_and_gap_alive_refuse_what_they_cannot_read():
    present = np.ones(100, dtype=bool)
    codes = np.full(100, D, dtype=np.int8)
    expect_error(ValueError, lambda: broadcast_sim.gaps_of(present.astype(np.int8), [(1, 0, 100)]), "ball_present",
                 "bool")
    expect_error(ValueError, lambda: broadcast_sim.gaps_of(present.reshape(10, 10), [(1, 0, 100)]), "ball_present")
    for graded, fragment in [([], "empty"), ([(1, 0, 101)], "outside"), ([(1, -1, 10)], "outside"),
                             ([(1, 10, 10)], "empty block"), ([(1, 0, 50), (2, 40, 90)], "overlap"),
                             ([(1, 0)], "triples"), ([(1, 0.5, 20)], "triples"), ([(1, "a", 20)], "triples")]:
        expect_error(ValueError, lambda: broadcast_sim.gaps_of(present, graded), "graded", fragment)
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0, 101)], codes), "gaps", "outside")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(-1, 5)], codes), "gaps", "outside")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(5, 5)], codes), "gaps", "empty")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0, 5, 6)], codes), "gaps", "(start, stop)")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0.5, 5)], codes), "gaps", "(start, stop)")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0, 5)], codes.astype(np.float64)), "codes", "integer")
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0, 5)], codes.reshape(10, 10)), "codes")
    odd = codes.copy()
    odd[3] = 7
    expect_error(ValueError, lambda: broadcast_sim.gap_alive([(0, 5)], odd), "codes", "7", "0-3")


# -- simulate: the mask -----------------------------------------------------------------------------------------------

def test_simulate_copies_a_donor_of_equal_length():
    donor = [True, True, False, False, True, False, True, True, True, False, False, True]
    m = len(donor)
    present, clean, codes = world(60, [(20, 20 + m)])
    pools = pools_of([(False, donor)])
    for seed in range(5):
        out = simulated(present, clean, codes, [(1, 5, 55)], pools, seed=seed)
        want = np.zeros(60, dtype=bool)
        want[20:20 + m] = donor
        assert out.shape == (60, 2)
        assert (filled_of(out) == want).all(), "the donor's mask, frame for frame, at the gap and nowhere else"
        assert (out[want] == clean[want] / SCALE).all(), "no error: the clean ball, normalised by the pitch"
        assert np.isnan(out[~want]).all()


def test_simulate_keeps_both_edges_when_stretching():
    donor = [True] * 10 + [False] * 35 + [True] * 5
    assert len(donor) == 50
    present, clean, codes = world(200, [(20, 120)])
    out = simulated(present, clean, codes, [(1, 0, 200)], pools_of([(False, donor)]))
    found = np.flatnonzero(filled_of(out)).tolist()
    assert found == list(range(20, 40)) + list(range(110, 120)), (
        "the donor's first 10 frames become the gap's first 20, its last 5 the last 10: {}".format(found))


def test_simulate_maps_a_donor_of_any_length_by_the_integer_rule():
    """fill[start + i] = donor[floor(i * k / m)], for a donor shorter than, as long as and longer than the gap."""
    rng = np.random.default_rng(3)
    for k, m in [(1, 5), (5, 1), (7, 3), (3, 7), (10, 10), (13, 4), (4, 13), (50, 100), (100, 50), (31, 97), (97, 31)]:
        donor = [bool(flag) for flag in rng.random(k) < 0.5]
        present, clean, codes = world(m + 20, [(10, 10 + m)])
        found = filled_of(simulated(present, clean, codes, [(1, 0, m + 20)], pools_of([(False, donor)])))
        assert found[10:10 + m].tolist() == mapped(donor, m), "donor of {} frames on a gap of {}".format(k, m)
        assert not found[:10].any() and not found[10 + m:].any(), "nothing outside the gap"


def test_simulate_picks_by_class():
    """A gap whose frames are mostly D takes a D donor, an alive gap an alive one, whatever the lengths."""
    d_donor = [True] * 40 + [False] * 160              # 200 frames: on 50 frames its first 40 give the first 10
    alive_donor = [True] * 50
    present, clean, codes = world(200, [(20, 70), (100, 150)], alive_in=[(100, 150)])
    graded = [(1, 0, 200)]
    both = pools_of([(True, alive_donor), (False, d_donor)])
    for seed in range(30):
        found = filled_of(simulated(present, clean, codes, graded, both, seed=seed))
        assert np.flatnonzero(found[:100]).tolist() == list(range(20, 30)), (
            "the D gap took the D donor (the alive one, all of 50 frames and the nearer in length, would fill it all)")
        assert np.flatnonzero(found[100:]).tolist() == list(range(0, 50)), "the alive gap took the alive donor"
    # A class with no donor gets no fill, and the other class still does.
    only_alive = filled_of(simulated(present, clean, codes, graded, pools_of([(True, alive_donor)])))
    assert not only_alive[:100].any() and only_alive[100:150].all(), "a D gap with no D donor is left empty"
    only_d = filled_of(simulated(present, clean, codes, graded, pools_of([(False, d_donor)])))
    assert only_d[:100].sum() == 10 and not only_d[100:].any(), "an alive gap with no alive donor is left empty"
    nothing = filled_of(simulated(present, clean, codes, graded, pools_of([])))
    assert not nothing.any(), "a pool with no donor at all fills nothing"


# -- simulate: the donor ---------------------------------------------------------------------------------------------

def nearest_five_case():
    """A gap of 100 frames, nine alive donors of different lengths, and which five are nearest.

    The lengths are in the pool's order. By |log(k / 100)| the nearest are 100, 110, 90 and, tied (each 1.25 times
    from 100), 80 then 125; the rest are 160, 60, and 40 and 250 (tied). The masks are random, so that each donor's fill
    on the gap tells which one it was.
    """
    m = 100
    lengths = [160, 90, 250, 100, 60, 125, 80, 110, 40]
    masks = random_masks(lengths, seed=5)
    distinct_after_mapping(masks, m)
    ranked = [lengths.index(k) for k in (100, 110, 90, 80, 125)]
    present, clean, codes = world(300, [(100, 200)], alive_in=[(100, 200)])
    pools = pools_of([(True, mask) for mask in masks])
    return m, masks, ranked, (present, clean, codes, [(1, 0, 300)], pools)


def test_simulate_draws_among_the_5_nearest_lengths():
    """Over 300 seeds the donor is always one of the five nearest, each of them turns up, and the donor is the one
    that `rng.integers(5)`, drawn first from a generator of the same seed, names in nearest-first order."""
    m, masks, ranked, args = nearest_five_case()
    seen = set()
    for seed in SEEDS:
        donor = which(masks, m, simulated(*args, seed=seed), 100)
        assert donor == ranked[int(np.random.default_rng(seed).integers(5))], "seed {}".format(seed)
        seen.add(donor)
    assert seen == set(ranked), "every one of the five and no other: {}".format(sorted(seen))


def test_donors_are_ranked_by_the_exact_log_distance_ties_to_the_shorter_then_the_earlier():
    """The ranking is |log(k / m)| worked out exactly: the longer length over the shorter, as a fraction, then the
    shorter length, then the earlier donor. A generator that answers a chosen index reads the ranking off, rank by
    rank."""
    m = 60
    # (the pool's lengths, the pool indices of the five nearest, nearest first)
    cases = [
        # 40 and 90 are both 1.5 times from 60; in floating point |log(40 / 60)| comes out larger than |log(90 / 60)|,
        # so a float ranking would take 90 for the fifth place. Exactly they tie, and the shorter, 40, comes first.
        ([75, 40, 60, 90, 66, 55], [2, 5, 4, 0, 1]),
        # 45 and 80 are both 4/3 times from 60 (floating point puts 80 first): 45, the shorter, is ranked before it.
        ([80, 60, 45, 66, 55, 100], [1, 4, 3, 2, 0]),
        # equal lengths tie, and the earlier donor (index 0) is ranked before the later (index 1).
        ([66, 66, 60, 55, 75, 90], [2, 3, 0, 1, 4]),
        # fewer than 5 donors of the class: all of them, and the draw is among 3. The other class never shows up.
        ([30, 100, 45], [2, 1, 0]),
    ]
    for lengths, expected in cases:
        masks = random_masks(lengths, seed=len(lengths) * 7 + lengths[0])
        distinct_after_mapping(masks, m)
        others = [(False, [True] * 60), (False, [True] * 59)] if len(lengths) == 3 else []   # the D class
        present, clean, codes = world(100, [(20, 80)], alive_in=[(20, 80)])
        pools = pools_of([(True, mask) for mask in masks] + others)
        for rank, pool_index in enumerate(expected):
            spy = Pick(rank)
            out = simulated(present, clean, codes, [(1, 0, 100)], pools, rng=spy)
            assert which(masks, m, out, 20) == pool_index, "lengths {}: rank {} is donor {}".format(
                lengths, rank, pool_index)
            assert spy.asked[0] == len(expected), "the donor is drawn from {} candidates".format(len(expected))


def test_simulate_is_seeded():
    """The same generator seed gives the same bytes, different seeds give different donors, and the generator moves."""
    m, masks, ranked, args = nearest_five_case()
    outs = [simulated(*args, seed=seed) for seed in range(20)]
    again = [simulated(*args, seed=seed) for seed in range(20)]
    assert all(first.tobytes() == second.tobytes() for first, second in zip(outs, again)), "same seed, same bytes"
    assert len({out.tobytes() for out in outs}) > 1, "different seeds do not all give the same ball"
    donors = {which(masks, m, out, 100) for out in outs}
    assert len(donors) >= 3, "different seeds give different donors: {}".format(sorted(donors))
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    simulated(*args, rng=rng)
    assert rng.bit_generator.state != before, "the caller's generator is the one that is drawn from"


def test_simulate_takes_no_match():
    """No match, no inputs, nothing that carries DFL's truth flags: arrays, a pitch, the pools and a generator."""
    parameters = inspect.signature(broadcast_sim.simulate).parameters
    assert list(parameters) == ["ball_present", "clean_ball_m", "driver_codes", "graded", "pitch", "pools", "rng"], (
        "simulate's parameters are {}".format(list(parameters)))
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty for p in parameters.values())
    for name in parameters:
        assert not any(word in name for word in ("match", "truth", "status", "possession", "inputs")), name


# -- simulate: the position -------------------------------------------------------------------------------------------

def test_simulate_adds_the_error_runs():
    """Each fragment of fill takes the first f offsets of an error run at least f long, drawn uniformly among those, in
    time order, after the donor's draw: the positions are the clean ball plus those offsets."""
    donor = [False] * 2 + [True] * 4 + [False] * 14 + [True] * 7 + [False] * 13          # 40 frames: fragments of 4, 7
    lengths = [3, 6, 10, 5]
    runs = [error_run(j, k) for j, k in enumerate(lengths)]
    present, clean, codes = world(60, [(10, 50)])
    pools = pools_of([(False, donor)], runs)
    first_runs = set()
    for seed in range(60):
        out = simulated(present, clean, codes, [(1, 0, 60)], pools, seed=seed)
        replay = np.random.default_rng(seed)
        replay.integers(1)                                  # the donor: the pool has one
        first = [1, 2, 3][int(replay.integers(3))]          # runs of 4 frames or more: 6, 10 and 5 long (indices 1-3)
        replay.integers(1)                                  # the second fragment (7 frames): only run 2 is long enough
        want = np.full((60, 2), np.nan)
        want[12:16] = (clean[12:16] + np.array(runs[first][:4])) / SCALE
        want[30:37] = (clean[30:37] + np.array(runs[2][:7])) / SCALE
        assert np.array_equal(out, want, equal_nan=True), "seed {}".format(seed)
        first_runs.add(first)
    assert first_runs == {1, 2, 3}, "uniform among the runs long enough, and never the run of 3: {}".format(first_runs)


def test_simulate_tiles_the_longest_error_run_when_none_is_long_enough():
    runs = [error_run(j, k) for j, k in enumerate([3, 5, 5, 2])]          # the longest is 5: the earlier one, index 1
    present, clean, codes = world(60, [(10, 30)])
    pools = pools_of([(False, [True] * 12 + [False] * 8)], runs)           # one fragment of 12 frames
    spy = Pick(0)
    out = simulated(present, clean, codes, [(1, 0, 60)], pools, rng=spy)
    want = np.full((60, 2), np.nan)
    want[10:22] = (clean[10:22] + np.array([runs[1][t % 5] for t in range(12)])) / SCALE
    assert np.array_equal(out, want, equal_nan=True), "the first 5 offsets of run 1, over and over, to 12 frames"
    assert spy.asked == [1], "one draw for the donor and none for the fragment, which has nothing to choose from"
    # A run exactly as long as the fragment is long enough: it is drawn among the runs of 5 and is not tiled.
    pools = pools_of([(False, [True] * 5 + [False] * 15)], runs)
    spy = Pick(1)
    out = simulated(present, clean, codes, [(1, 0, 60)], pools, rng=spy)
    want = np.full((60, 2), np.nan)
    want[10:15] = (clean[10:15] + np.array(runs[2][:5])) / SCALE
    assert np.array_equal(out, want, equal_nan=True), "answer 1 of the two runs of 5 is run 2, its first 5 offsets"
    assert spy.asked == [1, 2]


def test_simulate_leaves_nan_where_dfl_has_no_ball():
    donor = [True] * 10 + [False] * 10 + [True] * 10 + [False] * 10
    present, clean, codes = world(80, [(20, 60)])
    pools = pools_of([(False, donor)])
    graded = [(1, 0, 80)]
    holes = clean.copy()
    holes[22:24] = np.nan                   # both coordinates
    holes[26, 0] = np.nan                   # one coordinate only: still no position
    holes[45, 1] = np.nan
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    whole = simulated(present, clean, codes, graded, pools, rng=rng_a)
    out = simulated(present, holes, codes, graded, pools, rng=rng_b)
    want = filled_of(whole)
    assert want.sum() == 20 and np.flatnonzero(want).tolist() == list(range(20, 30)) + list(range(40, 50))
    for frame in (22, 23, 26, 45):
        assert np.isnan(out[frame]).all(), "frame {}: no ball on DFL, no fill".format(frame)
    want[[22, 23, 26, 45]] = False
    assert (filled_of(out) == want).all(), "every other frame that the donor fills still is"
    assert np.array_equal(out[want], whole[want]), "and with the same position"
    assert rng_a.bit_generator.state == rng_b.bit_generator.state, "a hole in DFL's ball changes none of the draws"


def test_simulate_fills_only_the_gaps_of_the_graded_blocks():
    """The output is NaN on every frame with the ball, and outside the graded blocks even where the ball is missing."""
    present, clean, codes = world(200, [(0, 30), (50, 80), (95, 110), (150, 200)])
    graded = [(1, 20, 100), (2, 140, 180)]
    out = simulated(present, clean, codes, graded, pools_of([(False, [True] * 40)]))    # a donor that fills every frame
    expected = list(range(20, 30)) + list(range(50, 80)) + list(range(95, 100)) + list(range(150, 180))
    assert np.flatnonzero(filled_of(out)).tolist() == expected, "the graded part of each missing run, and nothing else"


# -- simulate: the whole of it, against a plain reference -------------------------------------------------------------

def plain_simulate(present, clean, codes, graded, pitch, donors, runs, seed):
    """simulate written out one frame at a time in plain Python, from the text of design 3.4 and nothing else.

    `donors` is [(alive, mask)] and `runs` [[(dx, dy)]], as lists. The nearest donors are ordered by a comparator that
    cross-multiplies the two ratios (no Fraction, no log), the generator is drawn from in the stated order, and the
    result is a list of [x, y]."""
    length, width = pitch
    rng = np.random.default_rng(seed)
    out = [[math.nan, math.nan] for _ in present]
    gaps = []
    for _, first, last in sorted(graded, key=lambda block: block[1]):
        t = first
        while t < last:
            if present[t]:
                t += 1
                continue
            stop = t
            while stop < last and not present[stop]:
                stop += 1
            gaps.append((t, stop))
            t = stop
    for first, last in gaps:
        m = last - first
        alive_frames = sum(1 for t in range(first, last) if codes[t] in (H, A))
        wanted = 2 * alive_frames >= m
        candidates = [j for j, (flag, _) in enumerate(donors) if flag == wanted]
        if not candidates:
            continue

        def compare(i, j):
            ki, kj = len(donors[i][1]), len(donors[j][1])
            left, right = max(ki, m) * min(kj, m), max(kj, m) * min(ki, m)     # ratio_i against ratio_j, crossed
            for a, b in ((left, right), (ki, kj), (i, j)):
                if a != b:
                    return -1 if a < b else 1
            return 0

        ranked = sorted(candidates, key=functools.cmp_to_key(compare))[:5]
        mask = donors[ranked[int(rng.integers(len(ranked)))]][1]
        k = len(mask)
        fill = [mask[(i * k) // m] for i in range(m)]
        i = 0
        while i < m:
            if not fill[i]:
                i += 1
                continue
            j = i
            while j < m and fill[j]:
                j += 1
            f = j - i
            long_enough = [r for r, one in enumerate(runs) if len(one) >= f]
            if long_enough:
                offsets = runs[long_enough[int(rng.integers(len(long_enough)))]][:f]
            else:
                longest = max(len(one) for one in runs)
                one = runs[min(r for r, other in enumerate(runs) if len(other) == longest)]
                offsets = [one[t % len(one)] for t in range(f)]
            for t in range(f):
                x = clean[first + i + t][0] + offsets[t][0]
                y = clean[first + i + t][1] + offsets[t][1]
                if not (math.isnan(x) or math.isnan(y)):
                    out[first + i + t] = [x / length, y / width]
            i = j
    return out


def random_world(seed):
    """(present, clean, codes, graded, donors, runs): gaps of one frame to a hundred, codes in runs (so that gaps take
    both classes and sit near the 0.5 edge now and then), a clean ball with holes, a random pool (sometimes of one class
    only, sometimes empty) and random error runs, some shorter than the fragments."""
    r = np.random.default_rng([28, seed])
    n = int(r.integers(450, 600))
    first_stop = int(r.integers(120, 200))
    second_start = first_stop + int(r.integers(10, 40))
    graded = [(1, int(r.integers(0, 30)), first_stop), (2, second_start, int(r.integers(second_start + 100, n)))]
    present = np.ones(n, dtype=bool)
    t = 0
    while t < n:
        t += int(r.integers(1, 40))
        stop = t + int(r.integers(1, 100))
        present[t:stop] = False
        t = stop
    codes = np.empty(n, dtype=np.int8)
    t = 0
    while t < n:
        stop = t + int(r.integers(5, 45))
        codes[t:stop] = r.choice([H, A, D, X], p=[0.3, 0.2, 0.4, 0.1])
        t = stop
    clean = r.uniform(0.0, 100.0, size=(n, 2))
    clean[r.random(n) < 0.1] = np.nan
    clean[r.random(n) < 0.03, 0] = np.nan
    donors = []
    for _ in range(int(r.integers(0, 13)) if seed % 7 else 0):
        k = int(r.integers(1, 90))
        flags, t, flag = [], 0, bool(r.integers(2))
        while t < k:
            step = int(r.integers(1, 16))
            flags += [flag] * step
            flag = not flag
            t += step
        donors.append((bool(r.integers(2)) if seed % 4 else True, flags[:k]))
    runs = [[(float(dx), float(dy)) for dx, dy in r.normal(0.0, 2.0, size=(int(r.integers(1, 30)), 2))]
            for _ in range(int(r.integers(1, 7)))]
    return present, clean, codes, graded, donors, runs


def test_simulate_matches_a_plain_reference_on_random_worlds():
    classes = set()
    for seed in range(60):
        present, clean, codes, graded, donors, runs = random_world(seed)
        pools = broadcast_sim.Pools.from_lists([mask for _, mask in donors], [flag for flag, _ in donors], runs)
        out = broadcast_sim.simulate(present, clean, codes, graded, PITCH, pools, np.random.default_rng(seed))
        want = np.array(plain_simulate(present, clean, codes, graded, PITCH, donors, runs, seed))
        assert np.array_equal(out, want, equal_nan=True), "world {}".format(seed)
        classes.add(bool(np.isfinite(out).any()))
    assert classes == {True, False}, "some worlds fill something and some (an empty pool) nothing"


def test_simulate_refuses_what_it_cannot_serve():
    present, clean, codes = world(100, [(20, 40)])
    pools = pools_of([(False, [True] * 10)])
    graded = [(1, 0, 100)]
    rng = np.random.default_rng(0)

    def call(**changes):
        arguments = dict(ball_present=present, clean_ball_m=clean, driver_codes=codes, graded=graded, pitch=PITCH,
                         pools=pools, rng=rng)
        arguments.update(changes)
        return lambda: broadcast_sim.simulate(**arguments)

    call()()
    expect_error(ValueError, call(ball_present=present.astype(np.int8)), "ball_present", "bool")
    expect_error(ValueError, call(ball_present=present[:99]), "ball_present", "clean_ball_m", "100")
    expect_error(ValueError, call(clean_ball_m=clean[:, 0]), "clean_ball_m", "(n, 2)")
    expect_error(ValueError, call(clean_ball_m=clean[:99]), "clean_ball_m", "100")
    expect_error(ValueError, call(clean_ball_m=clean.astype(np.int8).astype(bool)), "clean_ball_m")
    expect_error(ValueError, call(driver_codes=codes[:99]), "driver_codes", "100")
    expect_error(ValueError, call(driver_codes=codes.astype(np.float64)), "driver_codes", "integer")
    expect_error(ValueError, call(driver_codes=np.full(100, 4, dtype=np.int8)), "driver_codes", "0-3")
    expect_error(ValueError, call(graded=[]), "graded", "empty")
    expect_error(ValueError, call(graded=[(1, 0, 101)]), "graded", "outside")
    expect_error(ValueError, call(graded=[(1, 0, 60), (2, 50, 100)]), "graded", "overlap")
    for pitch in [(0.0, 50.0), (100.0, -1.0), (100.0,), (math.nan, 50.0), (math.inf, 50.0), "ab", 100.0]:
        expect_error(ValueError, call(pitch=pitch), "pitch", "(length, width)")
    expect_error(ValueError, call(pools=None), "pools", "Pools")
    expect_error(ValueError, call(pools=[(False, [True])]), "pools", "Pools")
    for bad in (0, np.random.RandomState(0), None):
        expect_error(ValueError, call(rng=bad), "rng", "numpy.random.Generator", "default_rng")
    no_errors = broadcast_sim.Pools.from_lists([[True]], [False], [])
    expect_error(ValueError, call(pools=no_errors), "pools", "error run")


# -- measure_fill -----------------------------------------------------------------------------------------------------

def fill_scene():
    """A scene of 240 frames, its gaps and fills built by hand: (ball_present, filled, codes, graded, fill_xy, ref_xy).

    Blocks 10-110 and 130-230. The ball is missing on 0-40, 50-120, 140-150 and 160-235, so that the graded gaps are
    G1 = 10-40 (30 frames), G2 = 50-110 (60), G3 = 140-150 (10) and G4 = 160-230 (70): a missing run that reaches out
    of a block is cut by it. Codes: G1 is H on 10-25 and A on 25-40; G2 is A on 50-80 and D on 80-110; G3 is D; G4 is
    H on 160-200, X on 200-205 and D on 205-230; the graded frames with the ball are H; the rest is X.

    The fill (nine fragments, 60 frames) is on 10-20, 25-28, 50-62, 76-84, 95-100, 160-170, 190-195, 200-203 and
    226-230, and, outside the blocks, on 2-6, 112-116 and 231-234, which nothing may count. Each fragment has one
    offset (reference = fill - offset), as in `fragments` below, and they are Pythagorean (3-4-5 and the like), so that
    every distance is exact and a Euclidean distance cannot be taken for a Manhattan one; the reference is NaN on 95-100
    and on frame 160.
    """
    n = 240
    graded = [(1, 10, 110), (2, 130, 230)]
    present = np.ones(n, dtype=bool)
    for start, stop in [(0, 40), (50, 120), (140, 150), (160, 235)]:
        present[start:stop] = False
    codes = np.full(n, X, dtype=np.int8)
    for _, start, stop in graded:
        codes[start:stop] = H
    for start, stop, code in [(25, 40, A), (50, 80, A), (80, 110, D), (140, 150, D), (160, 200, H), (200, 205, X),
                              (205, 230, D)]:
        codes[start:stop] = code
    # fragment: (start, stop, position of the fill (x, y), offset (dx, dy))
    fragments = [(10, 20, (50.0, 25.0), (0, 0)), (25, 28, (-1.5, 25.0), (6, 8)), (50, 62, (60.0, 10.0), (0, 1)),
                 (76, 84, (101.0, 25.0), (5, 12)), (95, 100, (30.0, -1.0), (1, 1)), (160, 170, (30.0, 51.0), (8, 15)),
                 (190, 195, (30.0, 40.0), (20, 21)), (200, 203, (105.0, 20.0), (0, 2)),
                 (226, 230, (70.0, 20.0), (3, 4)),
                 (2, 6, (500.0, 25.0), (0, 0)), (112, 116, (500.0, 25.0), (0, 0)), (231, 234, (500.0, 25.0), (0, 0))]
    filled = np.zeros(n, dtype=bool)
    fill_xy = np.full((n, 2), np.nan)
    offset_xy = np.zeros((n, 2))
    for start, stop, position, offset in fragments:
        filled[start:stop] = True
        fill_xy[start:stop] = position
        offset_xy[start:stop] = offset
    fill_xy[83] = (101.5, 25.0)           # one frame of 76-84 is beyond the line, the others are exactly 1 m beyond it
    fill_xy[99] = (30.0, -2.0)            # the same at y: 95-99 are on the 1 m line, frame 99 is beyond it
    fill_xy[165:170] = (30.0, 52.0)       # five frames of 160-170 are beyond, five are on the line
    fill_xy[190:192] = (-1.0, 40.0)       # and two frames of 190-195 are on the line at x = -1 (all four are covered)
    reference = fill_xy - offset_xy
    reference[95:100] = np.nan
    reference[160] = np.nan
    return present, filled, codes, graded, fill_xy, reference


def test_measure_fill_on_known_masks():
    present, filled, codes, graded, fill_xy, reference = fill_scene()
    out = broadcast_sim.measure_fill(present, filled, codes, graded, fill_xy, reference, PITCH)

    def share(name, num, den):
        assert out[name] == {"share": num / den, "num": num, "den": den}, "{}: {}".format(name, out[name])

    # Gap frames: 30 + 60 + 10 + 70 = 170, of which 100 are H or A (30 + 30 + 0 + 40), 65 are D (0 + 30 + 10 + 25) and 5
    # are X. The 60 fill frames: 44 on H or A frames, 13 on D frames, 3 on X frames.
    share("F1", 44, 100)                 # 10 + 3 + 12 + 4 (76-80, A) + 10 + 5
    share("F2", 13, 65)                  # 4 (80-84) + 5 + 4 (226-230)
    share("W1", 60, 200)                 # the fill over the 100 + 100 graded frames
    share("W2", 100, 170)
    share("W3", 13, 60)                  # 3 (x < -1) + 1 (frame 83) + 1 (frame 99) + 5 + 3 (x > 101); none on a line
    # F3: gaps of 50 frames or more are G2 (60) and G4 (70). First 25 frames: 12 (50-62) + 10 (160-170); last 25: 5
    # (95-100) + 4 (226-230); inside, 10 + 20 frames: 8 (76-84) + 5 + 3.
    assert out["F3"] == {"gaps": 2, "first": {"share": 22 / 50, "num": 22, "den": 50},
                         "last": {"share": 9 / 50, "num": 9, "den": 50},
                         "inside": {"share": 16 / 30, "num": 16, "den": 30}}, out["F3"]
    # F4: alive fragments (at least half H or A) 10, 3, 12, 8 (76-84: exactly half), 10, 5; dead 5, 3 (all X), 4.
    assert out["F4"] == {"alive": {"median": 9.0, "n": 6}, "dead": {"median": 4.0, "n": 3}}, out["F4"]
    # F5: 54 frames with both positions (the 60, less 5 on 95-100 and frame 160): 0 x10 (10-20), 1 x12 (50-62), 2 x3
    # (200-203), 5 x4 (226-230), 10 x3 (25-28), 13 x8 (76-84), 17 x9 (160-170), 29 x5 (190-195): the median is the mean
    # of the 27th and 28th, both 5. (Manhattan distances would make it 7.)
    assert out["F5"] == {"median_m": 5.0, "n": 54}, out["F5"]
    assert set(out) == {"F1", "F2", "F3", "F4", "F5", "W1", "W2", "W3"}, sorted(out)
    for value in (out["F1"]["num"], out["F5"]["n"], out["F4"]["alive"]["n"]):
        assert type(value) is int
    for value in (out["F1"]["share"], out["F5"]["median_m"], out["F4"]["alive"]["median"]):
        assert type(value) is float, "plain floats, which a report can write"


def test_measure_fill_has_no_value_where_it_has_no_frames():
    """A share over no frames is None, a median over no fragments and no frames too, and none of it raises."""
    n = 100
    graded = [(1, 10, 90)]
    nothing = np.full((n, 2), np.nan)
    codes = np.full(n, H, dtype=np.int8)
    off = np.zeros(n, dtype=bool)

    def measure(present, filled, codes=codes):
        return broadcast_sim.measure_fill(present, filled, codes, graded, nothing, nothing, PITCH)

    out = measure(np.ones(n, dtype=bool), off)                    # the ball everywhere: no gap frame at all
    for name in ("F1", "F2", "W2", "W3"):
        assert out[name]["share"] is None, name
    assert out["F1"]["den"] == out["F2"]["den"] == out["W2"]["den"] == 0 and out["W3"]["den"] == 0
    assert out["W1"] == {"share": 0.0, "num": 0, "den": 80}, "no fill over 80 graded frames is 0, not None"
    assert out["F3"] == {"gaps": 0, "first": {"share": None, "num": 0, "den": 0},
                         "last": {"share": None, "num": 0, "den": 0}, "inside": {"share": None, "num": 0, "den": 0}}
    assert out["F4"] == {"alive": {"median": None, "n": 0}, "dead": {"median": None, "n": 0}}
    assert out["F5"] == {"median_m": None, "n": 0}

    # Gaps of exactly 50 frames have a first and a last 25 and nothing inside; no D frame: F2 is None, F1 is not.
    present = np.ones(n, dtype=bool)
    present[20:70] = False
    out = measure(present, off)
    assert out["F1"]["share"] == 0.0 and out["F1"]["den"] == 50 and out["F2"]["share"] is None
    assert out["F3"]["gaps"] == 1 and out["F3"]["first"]["den"] == 25 and out["F3"]["last"]["den"] == 25
    assert out["F3"]["inside"] == {"share": None, "num": 0, "den": 0}
    present[20:69] = True                                          # a gap of 1 frame: no gap of 50
    assert measure(present, off)["F3"]["first"]["share"] is None
    # No alive frame in the gaps (the ball is missing on frames 0-49, all D): F1 is None, F2 and W2 are 0.
    early = np.arange(n) >= 50
    out = measure(early, off, np.full(n, D, dtype=np.int8))
    assert out["F1"]["share"] is None and out["F2"]["share"] == 0.0 and out["W2"]["share"] == 0.0
    # One fill frame, exactly on the 1 m line and with no reference: W3 is 0 over 1 frame, F5 has no frame.
    only = np.zeros(n, dtype=bool)
    only[30] = True
    position = np.full((n, 2), np.nan)
    position[30] = (-1.0, 25.0)
    out = broadcast_sim.measure_fill(early, only, codes, graded, position, nothing, PITCH)
    assert out["W3"] == {"share": 0.0, "num": 0, "den": 1} and out["F5"] == {"median_m": None, "n": 0}


def test_measure_fill_refuses_what_it_cannot_read():
    present, filled, codes, graded, fill_xy, reference = fill_scene()

    def call(**changes):
        arguments = dict(ball_present=present, filled=filled, codes=codes, graded=graded, fill_xy_m=fill_xy,
                         reference_xy_m=reference, pitch=PITCH)
        arguments.update(changes)
        return lambda: broadcast_sim.measure_fill(**arguments)

    call()()
    # The fill is on gap frames only: a filled frame on which the ball is present is a mistake of the caller's (the ball
    # given may be the one after the fill), and it is not counted by any share.
    seen = present.copy()
    seen[filled] = True
    expect_error(ValueError, call(ball_present=seen), "filled", "ball_present", "before the fill")
    with_ball = filled.copy()
    with_ball[135] = True                                          # graded, and the ball is there
    expect_error(ValueError, call(filled=with_ball), "filled", "ball_present", "1 graded frame")
    unplaced = fill_xy.copy()
    unplaced[12] = np.nan
    expect_error(ValueError, call(fill_xy_m=unplaced), "fill_xy_m", "filled", "1 frame")
    # Outside the graded blocks nothing is checked or counted.
    outside = filled.copy()
    outside[236] = True
    call(filled=outside)()
    expect_error(ValueError, call(ball_present=present.astype(np.int8)), "ball_present", "bool")
    expect_error(ValueError, call(filled=filled[:100]), "filled", "240")
    expect_error(ValueError, call(codes=codes[:100]), "codes", "240")
    expect_error(ValueError, call(fill_xy_m=fill_xy[:, 0]), "fill_xy_m", "(n, 2)")
    expect_error(ValueError, call(reference_xy_m=reference[:100]), "reference_xy_m", "240")
    expect_error(ValueError, call(graded=[(1, 0, 300)]), "graded", "outside")
    expect_error(ValueError, call(pitch=(0, 50)), "pitch")


def plain_measure(present, filled, codes, graded, fill_xy, reference, pitch):
    """measure_fill by loops over frames, gaps and fragments: {name: (num, den)} and the medians, as lists."""
    length, width = pitch
    n = len(present)
    inside = [any(first <= t < stop for _, first, stop in graded) for t in range(n)]
    gap = [t for t in range(n) if inside[t] and not present[t]]
    fill = [t for t in range(n) if inside[t] and filled[t]]
    alive = [t for t in gap if codes[t] in (H, A)]
    dead = [t for t in gap if codes[t] == D]
    out = {"F1": (len([t for t in alive if filled[t]]), len(alive)),
           "F2": (len([t for t in dead if filled[t]]), len(dead)),
           "W1": (len(fill), len([t for t in range(n) if inside[t]])), "W2": (len(alive), len(gap)),
           "W3": (len([t for t in fill if fill_xy[t][0] < -1 or fill_xy[t][0] > length + 1 or fill_xy[t][1] < -1
                       or fill_xy[t][1] > width + 1]), len(fill))}
    edge = {"first": [0, 0], "last": [0, 0], "inside": [0, 0]}
    long_gaps = 0
    for _, first, stop in graded:
        t = first
        while t < stop:
            if present[t]:
                t += 1
                continue
            end = t
            while end < stop and not present[end]:
                end += 1
            if end - t >= 50:
                long_gaps += 1
                for name, frames in (("first", range(t, t + 25)), ("last", range(end - 25, end)),
                                     ("inside", range(t + 25, end - 25))):
                    edge[name][0] += sum(1 for u in frames if filled[u])
                    edge[name][1] += len(frames)
            t = end
    out["F3"] = (long_gaps, {name: tuple(pair) for name, pair in edge.items()})
    lengths = {True: [], False: []}
    for _, first, stop in graded:
        t = first
        while t < stop:
            if not filled[t]:
                t += 1
                continue
            end = t
            while end < stop and filled[end]:
                end += 1
            lengths[2 * sum(1 for u in range(t, end) if codes[u] in (H, A)) >= end - t].append(end - t)
            t = end
    out["F4"] = (lengths[True], lengths[False])
    out["F5"] = [math.hypot(fill_xy[t][0] - reference[t][0], fill_xy[t][1] - reference[t][1]) for t in fill
                 if not (math.isnan(reference[t][0]) or math.isnan(reference[t][1]))]
    return out


def random_scene(seed):
    """(present, filled, codes, graded, fill_xy, reference): gaps, fills inside them, codes in runs, positions around
    and beyond the lines, and a reference with holes."""
    r = np.random.default_rng([29, seed])
    n = 500
    graded = [(1, int(r.integers(0, 30)), int(r.integers(150, 220))), (2, int(r.integers(240, 260)),
                                                                      int(r.integers(400, n)))]
    present = np.ones(n, dtype=bool)
    t = 0
    while t < n:
        t += int(r.integers(1, 40))
        stop = t + int(r.integers(1, 110))
        present[t:stop] = False
        t = stop
    filled = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        t += int(r.integers(1, 30))
        stop = t + int(r.integers(1, 25))
        filled[t:stop] = True
        t = stop
    filled &= ~present
    codes = np.empty(n, dtype=np.int8)
    t = 0
    while t < n:
        stop = t + int(r.integers(3, 40))
        codes[t:stop] = r.choice([H, A, D, X], p=[0.3, 0.2, 0.4, 0.1])
        t = stop
    fill_xy = np.column_stack([r.uniform(-4.0, 104.0, n), r.uniform(-4.0, 54.0, n)])
    fill_xy[~filled] = np.nan
    reference = fill_xy + r.normal(0.0, 3.0, size=(n, 2))
    reference[r.random(n) < 0.15] = np.nan
    reference[r.random(n) < 0.05, 1] = np.nan
    return present, filled, codes, graded, fill_xy, reference


def test_measure_fill_matches_a_plain_loop_on_random_worlds():
    for seed in range(40):
        present, filled, codes, graded, fill_xy, reference = random_scene(seed)
        out = broadcast_sim.measure_fill(present, filled, codes, graded, fill_xy, reference, PITCH)
        want = plain_measure(present, filled, codes, graded, fill_xy, reference, PITCH)
        for name in ("F1", "F2", "W1", "W2", "W3"):
            num, den = want[name]
            assert out[name] == {"share": num / den if den else None, "num": num, "den": den}, (seed, name)
        gaps, edges = want["F3"]
        assert out["F3"]["gaps"] == gaps, seed
        for name, (num, den) in edges.items():
            assert out["F3"][name] == {"share": num / den if den else None, "num": num, "den": den}, (seed, name)
        alive, dead = want["F4"]
        assert out["F4"] == {"alive": {"median": statistics.median(alive) if alive else None, "n": len(alive)},
                             "dead": {"median": statistics.median(dead) if dead else None, "n": len(dead)}}, seed
        errors = want["F5"]
        assert out["F5"]["n"] == len(errors), seed
        if errors:    # math.hypot and numpy's may differ in the last bit; the hand-built scene above is exact
            assert math.isclose(out["F5"]["median_m"], statistics.median(errors), rel_tol=1e-12), seed
        else:
            assert out["F5"]["median_m"] is None, seed


# -- Pools ------------------------------------------------------------------------------------------------------------

def test_pools_from_lists_builds_the_arrays():
    masks = [[True, False, True], [False], [True, True, False, False]]
    runs = [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]]
    pools = broadcast_sim.Pools.from_lists(masks, [True, False, True], runs)
    assert pools.gap_lengths.tolist() == [3, 1, 4] and pools.gap_lengths.dtype == np.int64
    assert pools.gap_alive.tolist() == [True, False, True] and pools.gap_alive.dtype == np.bool_
    assert pools.gap_starts.tolist() == [0, 3, 4, 8] and pools.gap_starts.dtype == np.int64
    assert pools.gap_fill.tolist() == [True, False, True, False, True, True, False, False]
    assert pools.gap_fill.dtype == np.bool_
    assert pools.error_xy.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] and pools.error_xy.dtype == np.float64
    assert pools.error_starts.tolist() == [0, 2, 3] and pools.error_starts.dtype == np.int64
    for name in ("gap_lengths", "gap_alive", "gap_fill", "gap_starts", "error_xy", "error_starts"):
        assert not getattr(pools, name).flags.writeable, "{} is read-only".format(name)
    # The same arrays through the constructor, and an empty pool of each kind.
    direct = broadcast_sim.Pools(pools.gap_lengths, pools.gap_alive, pools.gap_fill, pools.gap_starts, pools.error_xy,
                                 pools.error_starts)
    assert direct.gap_fill.tolist() == pools.gap_fill.tolist()
    empty = broadcast_sim.Pools.from_lists([], [], [])
    assert empty.gap_lengths.shape == (0,) and empty.gap_starts.tolist() == [0] and empty.gap_fill.shape == (0,)
    assert empty.error_xy.shape == (0, 2) and empty.error_starts.tolist() == [0]
    assert empty.gap_alive.dtype == np.bool_ and empty.gap_fill.dtype == np.bool_
    # The arrays are the pools': a change to a list afterwards changes nothing, and a field cannot be set.
    masks[0][0] = False
    assert pools.gap_fill[0], "the pools hold copies"
    try:
        pools.gap_lengths = np.array([1])
    except AttributeError:
        pass
    else:
        raise AssertionError("a frozen dataclass: no field can be set")


def test_pools_repr_shows_no_array():
    """The pools are frame-level data of our match: the repr says nothing of them, and neither does a str."""
    masks = [[True, False, True, True, False, True, False]]
    runs = [[(123.456, -789.012)] * 3]
    pools = broadcast_sim.Pools.from_lists(masks, [True], runs)
    for text in (repr(pools), str(pools), "{}".format(pools)):
        assert text.startswith("Pools(") and "array" not in text and "[" not in text, text
        assert "123" not in text and "789" not in text and "True" not in text, text


def test_pools_validate_their_fields():
    base = dict(gap_lengths=np.array([3, 2]), gap_alive=np.array([True, False]),
                gap_fill=np.array([True, False, True, False, False]), gap_starts=np.array([0, 3, 5]),
                error_xy=np.array([[1.0, 2.0], [3.0, 4.0]]), error_starts=np.array([0, 2]))

    def build(**changes):
        return lambda: broadcast_sim.Pools(**{**base, **changes})

    build()()
    bad = [
        ("gap_lengths", np.array([3, 0]), "gap_lengths", "at least 1"),
        ("gap_lengths", np.array([3.0, 2.0]), "gap_lengths", "integer"),
        ("gap_lengths", np.array([[3, 2]]), "gap_lengths", "1-D"),
        ("gap_alive", np.array([True]), "gap_alive", "2 gaps"),
        ("gap_alive", np.array([1, 0]), "gap_alive", "bool"),
        ("gap_fill", np.array([True, False, True, False]), "Pools.gap_starts", "frames"),
        ("gap_fill", np.array([1, 0, 1, 0, 0]), "gap_fill", "bool"),
        ("gap_starts", np.array([0, 3]), "gap_starts", "3 offsets"),
        ("gap_starts", np.array([1, 3, 5]), "gap_starts", "start at 0"),
        ("gap_starts", np.array([0, 2, 5]), "gap_starts", "running totals"),
        ("error_xy", np.array([1.0, 2.0, 3.0, 4.0]), "error_xy", "2-D"),
        ("error_xy", np.array([[1.0, 2.0, 3.0]] * 2), "error_xy", "(m, 2)"),
        ("error_xy", np.array([[1.0, 2.0], [np.nan, 4.0]]), "error_xy", "finite"),
        ("error_xy", np.array([[1.0, 2.0], [3.0, np.inf]]), "error_xy", "finite"),
        ("error_xy", np.array([[True, False], [True, False]]), "error_xy", "number"),
        ("error_xy", np.array([[1.0, 2.0]]), "Pools.error_starts", "rows"),
        ("error_starts", np.array([], dtype=np.int64), "error_starts", "at least one"),
        ("error_starts", np.array([1, 2]), "error_starts", "start at 0"),
        ("error_starts", np.array([0, 0, 2]), "error_starts", "at least one frame"),
        ("error_starts", np.array([0, 3]), "Pools.error_xy", "rows"),
        ("error_starts", np.array([0.0, 2.0]), "error_starts", "integer"),
    ]
    for field, value, *fragments in bad:
        message = expect_error(ValueError, build(**{field: value}), "Pools." + field, *fragments)
        assert np.array2string(value) not in message and repr(value.tolist()) not in message, (
            "a message holds shapes and counts, never the array it was given: " + message)
    expect_error(ValueError, lambda: broadcast_sim.Pools.from_lists([[True]], [True, False], []), "alive", "2", "1")
    expect_error(ValueError, lambda: broadcast_sim.Pools.from_lists([[]], [True], []), "gap_lengths", "at least 1")
    expect_error(ValueError, lambda: broadcast_sim.Pools.from_lists([[1, 0]], [True], []), "gap_fill", "bool")
    expect_error(ValueError, lambda: broadcast_sim.Pools.from_lists([[True]], [True], [[(1.0, 2.0, 3.0)]]),
                 "error_runs", "(dx, dy)")
    expect_error(ValueError, lambda: broadcast_sim.Pools.from_lists([[True]], [True], [[]]), "error_starts",
                 "at least one frame")


def test_the_stand_in_constants_are_the_designs():
    """The constants of design 3.3 to 3.5, as declared before any run (T26, T37: none is tuned towards a gate)."""
    assert (broadcast_sim.POOLS_VERSION, broadcast_sim.DONORS, broadcast_sim.ALIVE_SHARE, broadcast_sim.NEAR_GAP_FRAMES,
            broadcast_sim.EDGE_FRAMES, broadcast_sim.LONG_GAP_FRAMES, broadcast_sim.BEYOND_LINE_M,
            broadcast_sim.RNG_SALT) == (1, 5, 0.5, 50, 25, 50, 1.0, 28)
    assert (broadcast_sim.H, broadcast_sim.A, broadcast_sim.D, broadcast_sim.X) == (pm.H, pm.A, pm.D, pm.X) == (
        0, 1, 2, 3), "possession_metrics' own codes"
