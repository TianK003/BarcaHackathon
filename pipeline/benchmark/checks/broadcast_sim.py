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

The wiring (Task 2b) is checked on the synthetic DFL matches of checks/common.py and on bar_ath_like, a stand-in for
BAR-ATH made of the synthetic bundle of pipeline/selftest.py with a given broadcast ball, made-up records of the
broadcast's files and made-up Smart Tagging; the decode of r024_both.toml is replaced by a stand-in that says which
sha256 its CSV has:

  * served_inputs_for: the plain inputs for a config without the fill; for one that fills, the twin's codes as the
    driver, the match's generator, the ball served exactly as simulate made it, the drivers kept per twin, no truth;
  * the pools: the pinned labels (refused otherwise, nothing left), a class with no donor and no error run refused, the
    cache key and what changes it, the pools and BAR-ATH's measure worked out by plain code, the cache read back and
    rebuilt when broken, the near-gap error runs on a hand-built scene, and `pools`' printout;
  * fidelity: the copies laid end to end and measured once, the config as run on DFL, the gates and checks with their
    exit codes, the report;
  * bar-ath: the tag check on the fill-only teacher frames, the alignment band and the live share first, the sanity,
    the exit codes, and the whole of it on the stand-in.

Nothing here reads data, and nothing prints an array.
"""

import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import io
import json
import math
import os
import statistics
import tempfile
import types

import numpy as np

import games
import possession_metrics as pm
from pipeline.benchmark import broadcast_sim, conventions, degrade, label_diff, tag_stats
from pipeline.benchmark import calibration as calibration_module
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (FILL_CONFIG, FILL_TWIN, leaks, made_up_pools, match_only_source,
                                              random_match, read, strict_json, synthetic_calibration,
                                              synthetic_matches)
from pipeline.config import ConfigError, load_config
from pipeline.evidence import controlled_touch, smart_tagging
from pipeline.inputs import BROADCAST_FOLDER, Inputs, atd_cache
from pipeline.inputs.broadcast import AlignmentError
from pipeline.selftest import (SYNTHETIC_FRAMES, SYNTHETIC_GRADED, expect_error, make_synthetic_bundle, quiet,
                               write_text)

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
PITCH = (100.0, 50.0)           # (length, width) in metres: dyadic, so that a position over the pitch is exact
SCALE = np.array(PITCH)
SEEDS = range(300)              # the seeds that the donor draw is read over
# What the BAR-ATH stand-in (bar_ath_like) writes for the broadcast's files that pools_key reads: the broadcast ATD
# cache's manifest and the homography file's text. Made up: they hold no array and no data.
BROADCAST_ATD_MANIFEST = {"sources": {"meta": {"bytes": 11, "sha256": "1" * 64},
                                      "raw": {"bytes": 22, "sha256": "2" * 64}}, "cache_version": 1}
HOMOGRAPHY_TEXT = "made up: the stand-in of the broadcast homography file\n"
# The alignment a run of the real match would report, at A20's offsets and above the live bar (made up).
ALIGNED = {"offsets": {1: 4014, 2: 26919}, "pairs": {1: 3000, 2: 2900}, "fine_scores": {1: 0.81, 2: 0.79},
           "coarse_scores": {1: 0.7, 2: 0.66}, "runner_up": {1: 0.1, 2: 0.12}, "valid_frames": 1000,
           "live_frames": 900, "live_share": 0.9, "usable_frames": 500}
# Smart Tagging of the stand-in: home votes on 140-190, away votes on 200-260 and 410-450 (a DEFENDING code names the
# team that defends), nothing elsewhere.
TAGS = ({"code": "BUILD UP", "tags": {"Team": "Barcelona"}, "start_frame": 140, "end_frame": 190},
        {"code": "PROGRESSION", "tags": {"Team": "Atletic Club"}, "start_frame": 200, "end_frame": 260},
        {"code": "DEFENDING IN DEFENSIVE THIRD", "tags": {"Team": "Barcelona"}, "start_frame": 410,
         "end_frame": 450},
        {"code": "SET PIECES", "tags": {"Team": "Barcelona"}, "start_frame": 0, "end_frame": 600})


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


# -- the wiring (Task 2b): the served inputs ---------------------------------------------------------------------------

@contextlib.contextmanager
def replaced(owner, name, value):
    """owner.name is `value` while the block runs, and what it was after it, raised or not."""
    real = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, real)


def wiring_configs(root):
    """(fill, twin): FILL_CONFIG and FILL_TWIN of checks/common.py, loaded from files written under `root`."""
    return (load_config(write_text(root, "fill.toml", FILL_CONFIG)),
            load_config(write_text(root, "twin.toml", FILL_TWIN)))


def fragment_lengths(ball):
    """The lengths of the maximal runs of frames where `ball` (n, 2) has a position."""
    padded = np.concatenate([[0], np.isfinite(ball).all(axis=1).astype(np.int8), [0]])
    return np.flatnonzero(np.diff(padded) == -1) - np.flatnonzero(np.diff(padded) == 1)


def test_served_inputs_for_none_is_plain():
    """A config without the fill is served the plain MatchInputs: the very one given, or one made as the runner makes
    it, with no broadcast ball and no driver decoded."""
    match = random_match()
    atd = match_inputs.clean_atd(match)
    calls = []
    with tempfile.TemporaryDirectory() as root:
        _, twin = wiring_configs(root)
        given = match_inputs.MatchInputs(match, atd, profile="gaps", seed=1, data_dir=root)
        with replaced(benchmark_run, "predict", lambda config, inputs: calls.append(config)):
            served = broadcast_sim.served_inputs_for(twin, match, atd, "gaps", 1, root, made_up_pools(), plain=given)
            made = broadcast_sim.served_inputs_for(twin, match, atd, "gaps", 1, root, None)
    assert served is given, "the very inputs given"
    assert calls == [], "a config without the fill has no driver to decode"
    assert isinstance(made, match_inputs.MatchInputs) and made.atd is atd
    assert (made.game, made.profile, made.seed, made.data_dir) == ("RND001", "gaps", 1, root)
    expect_error(match_inputs.AssetUnavailable, lambda: made.broadcast_ball, "broadcast")
    assert leaks(made, match) == [], leaks(made, match)


def test_served_inputs_for_uses_the_twin_driver():
    """A config that fills is served the simulated ball that its twin's labels drive (design 3.2): run.predict of the
    config with ball_fill "none" on the plain inputs, then simulate with the match's own generator,
    default_rng([seed, degrade.index_of(match), RNG_SALT]), and the ball served exactly as simulate made it: no run
    filter on DFL (ruling 11), so a fragment shorter than 13 frames stays. The driver's codes are kept in `drivers` per
    twin, match, profile and seed (ruling 9), so a second config with the same twin decodes nothing; nothing served
    holds the truth; and what cannot be served is refused before anything is decoded."""
    match = random_match()
    calibration = synthetic_calibration()
    atd = benchmark_run.make_atd(match, "gaps", 2, calibration)
    pools = made_up_pools()
    codes = np.where((np.arange(match.n_frames) // 37) % 2 == 0, H, D).astype(np.int8)
    seen = []

    def stub(config, inputs):
        seen.append((config, inputs))
        return codes.copy(), {}

    with tempfile.TemporaryDirectory() as root:
        fill, _ = wiring_configs(root)
        copied = load_config(write_text(root, "fill_copy.toml", FILL_CONFIG))
    plain = match_inputs.MatchInputs(match, atd, profile="gaps", seed=2)
    drivers = {}
    with replaced(benchmark_run, "predict", stub):
        served = [broadcast_sim.served_inputs_for(fill, match, atd, "gaps", 2, None, pools, plain=plain,
                                                  drivers=drivers),
                  broadcast_sim.served_inputs_for(fill, match, atd, "gaps", 2, None, pools, plain=plain,
                                                  drivers=drivers),
                  broadcast_sim.served_inputs_for(copied, match, atd, "gaps", 2, None, pools, plain=plain,
                                                  drivers=drivers)]
        assert len(seen) == 1, "one decode of the twin serves all three: {}".format(len(seen))
        served.append(broadcast_sim.served_inputs_for(fill, match, atd, "gaps", 2, None, pools))
        assert len(seen) == 2, "without `drivers` nothing is kept"
        expect_error(ValueError, lambda: broadcast_sim.served_inputs_for(fill, match, atd, "gaps", 2, None, None,
                                                                         plain=plain), "pools", "Pools")
        elsewhere = match_inputs.MatchInputs(match, benchmark_run.make_atd(match, "gaps", 1, calibration),
                                             profile="gaps", seed=1)
        expect_error(ValueError, lambda: broadcast_sim.served_inputs_for(fill, match, atd, "gaps", 2, None, pools,
                                                                         plain=elsewhere), "plain", "atd")
        assert len(seen) == 2, "refused before anything is decoded"
    driver, given = seen[0]
    assert driver.ball_fill == "none" and given is plain
    assert dataclasses.replace(driver, ball_fill="broadcast") == fill, "the config itself with the fill off, no more"
    assert isinstance(seen[1][1], match_inputs.MatchInputs) and seen[1][1].atd is atd and seen[1][0] == driver
    assert broadcast_sim.driver_key(driver) == broadcast_sim.driver_key(dataclasses.replace(copied, ball_fill="none"))
    assert broadcast_sim.driver_key(driver) != broadcast_sim.driver_key(fill), "the key holds ball_fill"

    expected = broadcast_sim.simulate(atd.ball_present, np.asarray(match.ball_xy) * np.array(match.pitch), codes,
                                      match.graded, match.pitch, pools,
                                      np.random.default_rng([2, degrade.index_of(match.match_id),
                                                             broadcast_sim.RNG_SALT]))
    lengths = fragment_lengths(expected)
    assert len(lengths) and (lengths < 13).any(), "the scene must have a fragment shorter than 13 frames to keep"
    for inputs in served:
        assert isinstance(inputs, match_inputs.MatchInputs) and inputs.atd is atd
        assert (inputs.game, inputs.profile, inputs.seed) == ("RND001", "gaps", 2)
        assert np.array_equal(inputs.broadcast_ball, expected, equal_nan=True), "served as simulate made it"
        assert leaks(inputs, match) == [], leaks(inputs, match)
    view = served[0].with_ball_fill("broadcast")
    assert np.array_equal(view.ball_filled, np.isfinite(expected).all(axis=1) & ~atd.ball_present)


# -- the wiring: the pools of BAR-ATH (design 3.3) ---------------------------------------------------------------------

def bar_ath_like(root, tags=(), summary=None):
    """BAR-ATH's Inputs in miniature: the synthetic bundle of pipeline/selftest.py under `root` (600 frames, two
    touching blocks, the ball missing on a third of the frames), with a given broadcast ball (broadcast_like), the
    made-up broadcast files that pools_key reads (broadcast_files), an alignment summary (`summary`: a dict, or an
    exception that reading it raises; default ALIGNED) and Smart Tagging (`tags`). What the pools, fidelity's BAR-ATH
    side and bar-ath read of the real match, and none of it. Its tactical ATD cache is built (so its manifest is
    there), and its attribute given_ball may be changed by a check."""
    make_synthetic_bundle(root)
    broadcast_files(root)

    def summarised(self):
        if isinstance(self.summary, Exception):
            raise self.summary
        return copy.deepcopy(self.summary)

    class BarAthLike(Inputs):
        broadcast_ball = property(lambda self: self.given_ball)
        broadcast_summary = property(summarised)
        smart_tagging = property(lambda self: [dict(tag) for tag in self.tags])

    inputs = BarAthLike(data_dir=root, n_frames=SYNTHETIC_FRAMES, graded=list(SYNTHETIC_GRADED))
    inputs.given_ball = broadcast_like(inputs)
    inputs.summary = copy.deepcopy(ALIGNED) if summary is None else summary
    inputs.tags = [dict(tag) for tag in tags]
    return inputs


def broadcast_files(root, manifest=None, homography=HOMOGRAPHY_TEXT):
    """Write, under `root`, what pools_key reads of the broadcast: the broadcast ATD cache's manifest (`manifest`,
    default BROADCAST_ATD_MANIFEST) and the broadcast homography file (`homography`, its text). Returns the
    homography file's path."""
    folder = os.path.join(root, "cache", BROADCAST_FOLDER)
    os.makedirs(folder, exist_ok=True)
    write_text(folder, atd_cache.MANIFEST, json.dumps(BROADCAST_ATD_MANIFEST if manifest is None else manifest))
    path = games.asset_paths(games.GAME, root, source="broadcast")["homography"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return write_text(os.path.dirname(path), os.path.basename(path), homography)


def broadcast_like(inputs):
    """A broadcast ball for bar_ath_like, normalised, NaN where it has none: ours plus (0.01, -0.02) where ours is seen,
    except on every 7th frame; and where ours is missing, a made-up place on two frames of three."""
    ours = np.asarray(inputs.atd.ball_xy)
    t = np.arange(len(ours))
    ball = ours + np.array([0.01, -0.02])
    ball[t % 7 == 0] = np.nan
    missing = np.isnan(ours).any(axis=1)
    made = np.column_stack([0.2 + 0.001 * t, 0.7 - 0.0005 * t])
    ball[missing & (t % 3 != 0)] = made[missing & (t % 3 != 0)]
    ball[missing & (t % 3 == 0)] = np.nan
    return ball


def wiring_codes(n):
    """Labels for the stand-in, as the decode of r024_both.toml would give them: H, then A, then D, in periods of 50."""
    t = np.arange(n) % 50
    return np.where(t < 25, H, np.where(t < 35, A, D)).astype(np.int8)


def labels_decoded(sha=None, codes=None, calls=None):
    """A stand-in for broadcast_sim.decode_labels: it writes a small file where the decode writes its CSV, records the
    path in `calls`, and returns `codes` (default wiring_codes) with `sha` (default the pinned R024_SHA256)."""
    def decode(inputs, path, log=print):
        if calls is not None:
            calls.append(path)
        write_text(os.path.dirname(path), os.path.basename(path), "frame,label\n")
        return (wiring_codes(inputs.n_frames) if codes is None else codes), (
            broadcast_sim.R024_SHA256 if sha is None else sha)
    return decode


def files_under(folder):
    """Every file under `folder`, as relative paths; [] for a folder that is not there."""
    return sorted(os.path.relpath(os.path.join(here, name), folder)
                  for here, _, names in os.walk(folder) for name in names)


def plain_near_gap_runs(present, ours_m, theirs_m, graded):
    """Ruling 2's error runs, one frame at a time: [[(dx, dy), ...], ...] in time order. A frame is in a run when it is
    graded, both balls are there, and it is 1 to NEAR_GAP_FRAMES frames before a gap's first frame or after its last,
    the gap in the frame's own block."""
    near = broadcast_sim.NEAR_GAP_FRAMES
    runs = []
    for _, start, stop in sorted(graded, key=lambda block: block[1]):
        gaps, t = [], start
        while t < stop:
            if present[t]:
                t += 1
                continue
            first = t
            while t < stop and not present[t]:
                t += 1
            gaps.append((first, t - 1))
        current = []
        for t in range(start, stop):
            both = present[t] and np.isfinite(ours_m[t]).all() and np.isfinite(theirs_m[t]).all()
            if both and any(1 <= first - t <= near or 1 <= t - last <= near for first, last in gaps):
                current.append((float(theirs_m[t, 0] - ours_m[t, 0]), float(theirs_m[t, 1] - ours_m[t, 1])))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
    return runs


def test_near_gap_errors_follow_the_frame_conditions():
    """Ruling 2 on a hand-built scene: a frame is in the error pool when it is graded, both balls are there, and it is
    1 to 50 frames before a gap's first frame or after its last, in that gap's own block; the runs are maximal and cut
    at the blocks, and each offset is the broadcast's position minus ours, in metres."""
    assert broadcast_sim.NEAR_GAP_FRAMES == 50
    n = 330
    t = np.arange(n, dtype=np.float64)
    for graded, expected in (
            # Block 2 starts 10 frames after block 1: frames 150-159 are not graded, 46 to 55 frames after gap A.
            ([(1, 0, 150), (2, 160, 310)], [(50, 60), (62, 100), (105, 120), (121, 150), (190, 240), (250, 305)]),
            # The blocks touch: frames 150-154 are within 50 of gap A's last frame, in another block, so not near.
            ([(1, 0, 150), (2, 150, 310)], [(50, 60), (62, 100), (105, 120), (121, 150), (190, 240), (250, 305)])):
        present = np.ones(n, dtype=bool)
        present[100:105] = False          # gap A, block 1: near 50-99 and 105-149 (the block ends at 150)
        present[240:250] = False          # gap B, block 2: near 190-239 and 250-299; 189 is 51 frames before it
        present[305:315] = False          # gap C, cut at block 2's end (305-310): near 255-304
        ours = np.column_stack([t, 0.5 * t])
        ours[~present] = np.nan
        theirs = np.column_stack([t + 1.0 + t / 128.0, 0.5 * t - 0.25])
        theirs[60:62] = np.nan            # the run before gap A is cut in two
        theirs[120] = np.nan
        error_xy, starts = broadcast_sim.near_gap_errors(present, ours, theirs, graded)
        frames = np.concatenate([np.arange(first, stop) for first, stop in expected])
        assert starts.tolist() == np.cumsum([0] + [stop - first for first, stop in expected]).tolist(), starts
        assert np.array_equal(error_xy, (theirs - ours)[frames]), "the broadcast minus ours, frame by frame"
        runs = plain_near_gap_runs(present, ours, theirs, graded)
        assert [len(run) for run in runs] == np.diff(starts).tolist()
        assert np.array_equal(error_xy, np.array([pair for run in runs for pair in run]))
        assert error_xy.dtype == np.float64 and starts.dtype == np.int64
    for bad, name in ((present[:-1], "ball_present"), (ours[:-1], "ball_xy_m"), (theirs[:, :1], "broadcast_xy_m")):
        arguments = {"ball_present": present, "ball_xy_m": ours, "broadcast_xy_m": theirs, name: bad}
        expect_error(ValueError, lambda arguments=arguments: broadcast_sim.near_gap_errors(graded=graded, **arguments),
                     name)


def test_pools_refuse_labels_that_are_not_e6496a9b():
    """The pools are built from r024_both.toml's labels, decoded in-process, and only from the CSV that main.toml writes
    today (sha256 e6496a9b...): other labels stop the step, naming both hashes, and nothing is left in the cache (the
    decode's CSV included). load_pools and load_bar_ath_measure never build: with no cache they say to run `pools`."""
    assert broadcast_sim.R024_SHA256 == "e6496a9b9805217ef085e52ed69cbb27a42b95368d65c1d188b4a2f339b61170"
    assert os.path.basename(broadcast_sim.R024_CONFIG) == "r024_both.toml" and os.path.isfile(broadcast_sim.R024_CONFIG)
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        with replaced(broadcast_sim, "decode_labels", labels_decoded(sha="0" * 64)):
            for build in (lambda: broadcast_sim.build_pools(inputs, log=quiet),
                          lambda: broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet)):
                expect_error(broadcast_sim.BroadcastSimError, build, "r024_both", "e6496a9b", "0" * 64)
        assert files_under(os.path.join(root, *broadcast_sim.CACHE_FOLDER)) == [], "nothing is written"
        for load in (broadcast_sim.load_pools, broadcast_sim.load_bar_ath_measure):
            expect_error(broadcast_sim.BroadcastSimError, lambda: load(root), "broadcast_sim.py pools")


def test_pools_refuse_a_class_without_donors_and_an_empty_error_pool():
    """Ruling 10: with no alive gap, no D gap, or no near-gap error run, the pools stop before anything is written."""
    assert issubclass(broadcast_sim.BroadcastSimError, ValueError)
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        n = inputs.n_frames
        for code, fragment in ((D, "alive"), (H, "D")):
            with replaced(broadcast_sim, "decode_labels", labels_decoded(codes=np.full(n, code, dtype=np.int8))):
                expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.prepare_pools(
                    root, inputs=inputs, log=quiet), fragment, "donor")
        blind = np.array(inputs.given_ball)
        blind[np.asarray(inputs.atd.ball_present)] = np.nan       # the broadcast only where ours is missing
        inputs.given_ball = blind
        with replaced(broadcast_sim, "decode_labels", labels_decoded()):
            expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.build_pools(inputs, log=quiet),
                         "error run")
        assert files_under(os.path.join(root, *broadcast_sim.CACHE_FOLDER)) == [], "nothing is written"


def code_path(name):
    """The file of a CODE_FILES entry, as code_sha256 opens it."""
    return os.path.join(broadcast_sim.ROOT, *name.split("/"))


@contextlib.contextmanager
def code_edited(name):
    """read_source gives the file of CODE_FILES entry `name` with one more line, and every other file as it is."""
    real = broadcast_sim.read_source
    target = os.path.normcase(os.path.abspath(code_path(name)))

    def edited(path):
        content = real(path)
        return content + b"# one more line\n" if os.path.normcase(os.path.abspath(path)) == target else content

    with replaced(broadcast_sim, "read_source", edited):
        yield


def test_pools_key_fingerprints_the_code_and_the_files():
    """The cache key holds the sha256 of the code that builds the pools (CODE_FILES, each read with CRLF as LF), the
    tactical and broadcast ATD caches' records as their manifests on disk hold them, the broadcast homography file's
    size and sha256, r024_both.toml's text, its CSV's sha256, POOLS_VERSION and the match's frames, pitch and graded
    blocks from games.py: a change to any of them is another folder, and a CRLF checkout of the same text is not a
    change. A cache manifest that cannot be read says to run `pools` first."""
    assert broadcast_sim.CODE_FILES == ("pipeline/benchmark/broadcast_sim.py", "pipeline/inputs/broadcast.py",
                                        "pipeline/inputs/__init__.py")
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        key = broadcast_sim.pools_key(root)
        text = read(broadcast_sim.R024_CONFIG).replace("\r\n", "\n")
        assert key["r024_config"] == text and key["r024_csv_sha256"] == broadcast_sim.R024_SHA256
        assert key["pools_version"] == broadcast_sim.POOLS_VERSION == 1
        manifest = inputs.atd_manifest
        assert key["tactical_atd"] == {"sources": manifest["sources"], "cache_version": manifest["cache_version"]}
        assert key["broadcast_atd"] == BROADCAST_ATD_MANIFEST
        homography = HOMOGRAPHY_TEXT.encode("utf-8")
        assert key["broadcast_homography"] == {"bytes": len(homography),
                                               "sha256": hashlib.sha256(homography).hexdigest()}
        for name in broadcast_sim.CODE_FILES:
            with open(code_path(name), "rb") as handle:
                lf = handle.read().replace(b"\r\n", b"\n")
            assert key["code_sha256"][name] == hashlib.sha256(lf).hexdigest(), name
        assert sorted(key["code_sha256"]) == sorted(broadcast_sim.CODE_FILES)
        assert key["n_frames"] == games.source_frames("tactical") and key["pitch"] == list(games.GAMES[games.GAME][
            "pitch"]) and key["graded"] == [[half, first, last + 1] for half, first, last in games.half_video_blocks(
                games.GAME)]
        assert json.loads(json.dumps(key)) == key, "JSON-normal"
        folder = broadcast_sim.cache_folder(root, key)
        assert os.path.dirname(folder) == os.path.join(root, "cache", "benchmark", "broadcast_sim")
        assert len(os.path.basename(folder)) == 16 and broadcast_sim.cache_folder(root, dict(reversed(key.items()))) \
            == folder, "the folder of a key is the same whatever the order of its entries"

        def folder_now():
            return broadcast_sim.cache_folder(root, broadcast_sim.pools_key(root))

        os.makedirs(os.path.join(root, "edited"))
        with replaced(broadcast_sim, "R024_CONFIG", write_text(os.path.join(root, "edited"), "r024_both.toml",
                                                               text + "# one more comment\n")):
            assert folder_now() != folder, "another text of r024_both.toml"
        os.makedirs(os.path.join(root, "crlf"))
        crlf = os.path.join(root, "crlf", "r024_both.toml")
        with open(crlf, "wb") as handle:
            handle.write(text.replace("\n", "\r\n").encode("utf-8"))
        with replaced(broadcast_sim, "R024_CONFIG", crlf):
            assert folder_now() == folder, "the same text with CRLF line endings"
        with replaced(broadcast_sim, "POOLS_VERSION", 2):
            assert folder_now() != folder
        with replaced(broadcast_sim, "R024_SHA256", "f" * 64):
            assert folder_now() != folder

        # The code: one more line in any of the three files is another key; read_source reads CRLF as LF.
        for name in broadcast_sim.CODE_FILES:
            with code_edited(name):
                assert folder_now() != folder, name
        assert folder_now() == folder
        lf_path = write_text(os.path.join(root, "crlf"), "lf.py", "a = 1\nb = 2\n")
        crlf_path = os.path.join(root, "crlf", "crlf.py")
        with open(crlf_path, "wb") as handle:
            handle.write(b"a = 1\r\nb = 2\r\n")
        assert broadcast_sim.read_source(crlf_path) == broadcast_sim.read_source(lf_path) == b"a = 1\nb = 2\n"

        # The files, as they are on disk.
        broadcast_files(root, homography=HOMOGRAPHY_TEXT + "one more row\n")
        assert folder_now() != folder, "another broadcast homography"
        broadcast_files(root)
        assert folder_now() == folder
        other = copy.deepcopy(BROADCAST_ATD_MANIFEST)
        other["sources"]["raw"]["sha256"] = "5" * 64
        broadcast_files(root, manifest=other)
        assert folder_now() != folder, "another broadcast ATD"
        broadcast_files(root, manifest=dict(BROADCAST_ATD_MANIFEST, cache_version=2))
        assert folder_now() != folder, "the broadcast ATD parsed another way"
        broadcast_files(root)
        tactical = os.path.join(root, "cache", atd_cache.FOLDER, atd_cache.MANIFEST)
        with open(tactical, "rb") as handle:
            kept = handle.read()
        changed = json.loads(kept.decode("utf-8"))
        changed["sources"]["atd_raw"]["sha256"] = "6" * 64
        write_text(os.path.dirname(tactical), atd_cache.MANIFEST, json.dumps(changed))
        assert folder_now() != folder, "another tactical ATD"
        with open(tactical, "wb") as handle:
            handle.write(kept)
        assert folder_now() == folder

        for name in (tactical, os.path.join(root, "cache", BROADCAST_FOLDER, atd_cache.MANIFEST)):
            os.rename(name, name + ".away")
            expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.pools_key(root), "manifest",
                         "broadcast_sim.py pools")
            os.rename(name + ".away", name)
        assert folder_now() == folder


def test_pools_key_loads_no_tracking_array():
    """pools_key reads files (two manifests, the homography file's bytes, the config, the code) and loads no array of
    BAR-ATH's tracking; so do load_pools and load_bar_ath_measure, and prepare_pools when the cache of the key is
    whole: with every way to an ATD array refused, and an `inputs` that holds a game and a data dir only, they work."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        with replaced(broadcast_sim, "decode_labels", labels_decoded()):
            pools, measure = broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet)
        key = broadcast_sim.pools_key(root)

        def refused(*args, **kwargs):
            raise AssertionError("an array of BAR-ATH's tracking was asked for")

        bare = types.SimpleNamespace(game=games.GAME, data_dir=root)
        with replaced(atd_cache, "load_or_build", refused), replaced(atd_cache, "_load", refused), \
                replaced(atd_cache, "parse", refused), replaced(broadcast_sim, "bar_ath_inputs", refused), \
                replaced(Inputs, "atd", property(refused)), replaced(Inputs, "broadcast_atd", property(refused)), \
                replaced(broadcast_sim, "decode_labels", refused):
            assert broadcast_sim.pools_key(root) == key
            assert broadcast_sim.digest(broadcast_sim.load_pools(root)) == broadcast_sim.digest(pools)
            assert broadcast_sim.load_bar_ath_measure(root) == measure
            again, again_measure = broadcast_sim.prepare_pools(root, inputs=bare, log=quiet)
            assert broadcast_sim.digest(again) == broadcast_sim.digest(pools) and again_measure == measure


def test_pools_are_built_cached_and_rebuilt_when_broken():
    """prepare_pools builds design 3.3's pools from BAR-ATH's inputs (each gap's real fill mask and its class by the
    decoded labels; the near-gap error runs), and BAR-ATH's side of fidelity beside them (ruling 1: the ball before the
    fill, the fill's own mask, the decoded labels, the broadcast ball in metres, no reference, and F5 the near-gap
    pool's median); it keeps both under the key's folder and deletes the decode's CSV as soon as it is read. load_pools
    and load_bar_ath_measure read them back without decoding; a broken cache, or one of other code, is absent: they
    refuse it, saying to run `pools`, and prepare_pools builds it again."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        calls = []
        cache = os.path.join(root, "cache")
        expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.load_pools(root), "no whole cache",
                     "broadcast_sim.py pools")
        with replaced(broadcast_sim, "decode_labels", labels_decoded(calls=calls)):
            pools, measure = broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet)
            assert len(calls) == 1
            folder = broadcast_sim.cache_folder(root, broadcast_sim.pools_key(root))
            assert calls == [os.path.join(root, *broadcast_sim.CACHE_FOLDER, "{}.{}.csv".format(
                broadcast_sim.LABELS_STEM, os.getpid()))], "decoded under the pools' cache"
            csvs = [name for name in files_under(cache) if name.endswith(".csv")]
            assert not os.path.exists(calls[0]) and not csvs, \
                "the decode's CSV is deleted once read: no CSV of BAR-ATH's labels stays in the cache"
            names = set(os.listdir(folder))
            kept = {broadcast_sim.MANIFEST, broadcast_sim.MEASURE_FILE}
            assert kept <= names and names - kept and all(name.endswith(".npy") for name in names - kept), names

            n = inputs.n_frames
            codes = wiring_codes(n)
            present = np.asarray(inputs.atd.ball_present)
            gaps = broadcast_sim.gaps_of(present, inputs.graded)
            filled = np.asarray(inputs.with_ball_fill("broadcast").ball_filled)
            assert pools.gap_lengths.tolist() == [stop - start for start, stop in gaps]
            assert pools.gap_alive.tolist() == broadcast_sim.gap_alive(gaps, codes).tolist()
            assert pools.gap_alive.any() and not pools.gap_alive.all(), "both classes"
            assert np.array_equal(pools.gap_fill, np.concatenate([filled[start:stop] for start, stop in gaps]))
            assert pools.gap_fill.any() and not pools.gap_fill.all()
            pitch = np.array(inputs.pitch)
            ours_m, theirs_m = np.asarray(inputs.atd.ball_xy) * pitch, np.asarray(inputs.given_ball) * pitch
            runs = plain_near_gap_runs(present, ours_m, theirs_m, inputs.graded)
            assert len(runs) > 1 and pools.error_starts.tolist() == np.cumsum([0] + [len(run) for run in runs]).tolist()
            assert np.array_equal(pools.error_xy, np.array([pair for run in runs for pair in run]))
            expected = broadcast_sim.measure_fill(present, filled, codes, inputs.graded, theirs_m,
                                                  np.full((n, 2), np.nan), inputs.pitch)
            distances = np.hypot(pools.error_xy[:, 0], pools.error_xy[:, 1])
            expected["F5"] = {"median_m": float(np.median(distances)), "n": int(len(distances))}
            assert measure == expected, "BAR-ATH's side, as ruling 1 says"
            assert measure["W1"]["num"] == int(filled.sum()) > 0

            assert broadcast_sim.digest(broadcast_sim.load_pools(root)) == broadcast_sim.digest(pools), "read back"
            assert broadcast_sim.load_bar_ath_measure(root) == measure
            again, again_measure = broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet)
            assert len(calls) == 1 and broadcast_sim.digest(again) == broadcast_sim.digest(pools)
            assert again_measure == measure
            with code_edited(broadcast_sim.CODE_FILES[1]):
                expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.load_pools(root),
                             "broadcast_sim.py pools")
            array = sorted(name for name in names if name.endswith(".npy"))[0]

            def cut_array():
                with open(os.path.join(folder, array), "r+b") as handle:
                    handle.truncate(20)

            def write(name, text):
                return lambda: write_text(folder, name, text)

            for damage in (cut_array, lambda: os.remove(os.path.join(folder, array)),
                           write(broadcast_sim.MEASURE_FILE, "not JSON"), write(broadcast_sim.MEASURE_FILE, "[1, 2]"),
                           write(broadcast_sim.MANIFEST, "{"), write(broadcast_sim.MANIFEST, '{"key": 3}')):
                damage()
                for load in (broadcast_sim.load_pools, broadcast_sim.load_bar_ath_measure):
                    expect_error(broadcast_sim.BroadcastSimError, lambda: load(root), "broadcast_sim.py pools")
                rebuilt, rebuilt_measure = broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet)
                assert broadcast_sim.digest(rebuilt) == broadcast_sim.digest(pools) and rebuilt_measure == measure
                assert broadcast_sim.digest(broadcast_sim.load_pools(root)) == broadcast_sim.digest(pools)
            assert len(calls) == 7, "each broken cache was built again, once: {}".format(len(calls))
            fresh = broadcast_sim.prepare_pools(root, inputs=inputs, log=quiet, rebuild=True)[0]
            assert len(calls) == 8 and broadcast_sim.digest(fresh) == broadcast_sim.digest(pools)
            assert not [name for name in files_under(cache) if name.endswith(".csv")]

    fields = {name: getattr(pools, name) for name in ("gap_lengths", "gap_alive", "gap_fill", "gap_starts", "error_xy",
                                                      "error_starts")}
    same = broadcast_sim.Pools(**{name: np.array(value) for name, value in fields.items()})
    assert broadcast_sim.digest(same) == broadcast_sim.digest(pools) and len(broadcast_sim.digest(pools)) == 64
    moved = broadcast_sim.Pools(**dict(fields, error_xy=fields["error_xy"] + 0.5))
    assert broadcast_sim.digest(moved) != broadcast_sim.digest(pools)
    flipped = np.array(fields["gap_alive"])
    flipped[0] = not flipped[0]
    assert broadcast_sim.digest(broadcast_sim.Pools(**dict(fields, gap_alive=flipped))) != broadcast_sim.digest(pools)


def main_of(argv):
    """(exit code, what was printed, stdout and stderr together) of broadcast_sim.main(argv)."""
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown), contextlib.redirect_stderr(shown):
        code = broadcast_sim.main(argv)
    return code, shown.getvalue()


def test_pools_command_prints_aggregates_only():
    """`pools` builds (or reads) the pools and BAR-ATH's measure and prints aggregates only (ruling 10): the gaps per
    class, the donor frames filled per class, the error runs and frames, the measure and the digest; a refusal is
    exit 2, with nothing written."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        with replaced(broadcast_sim, "bar_ath_inputs", lambda data_dir: inputs), \
                replaced(broadcast_sim, "decode_labels", labels_decoded()):
            code, text = main_of(["pools", "--data-dir", root])
            assert code == 0, text
            pools = broadcast_sim.load_pools(root)
        lines = text.rstrip("\n").split("\n")
        assert broadcast_sim.digest(pools) in text
        alive = int(pools.gap_alive.sum())
        assert any("alive" in line and str(alive) in line for line in lines), text
        assert any("error" in line and str(len(pools.error_starts) - 1) in line for line in lines), text
        for name in ("F1", "F2", "F3", "F4", "F5", "W1", "W2", "W3"):
            assert any(line.lstrip().startswith(name) for line in lines), (name, text)
        assert len(lines) <= 40 and max(len(line) for line in lines) <= 160 and "[" not in text and "array" not in text

    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root)
        with replaced(broadcast_sim, "bar_ath_inputs", lambda data_dir: inputs), \
                replaced(broadcast_sim, "decode_labels", labels_decoded(sha="0" * 64)):
            code, text = main_of(["pools", "--data-dir", root])
        assert code == 2 and "FAILED" in text and "e6496a9b" in text, (code, text)
        assert files_under(os.path.join(root, *broadcast_sim.CACHE_FOLDER)) == []


# -- the wiring: fidelity (design 3.5) --------------------------------------------------------------------------------

def measure_of(F1=(30, 100), F2=(4, 100), first=(30, 100), last=(22, 100), inside=(7, 100), alive=10.0, dead=6.0,
               F5=1.0, W1=(4, 100), W2=(20, 100), W3=(15, 100)):
    """A made-up measure_fill result: (num, den) for each share, a number or None for each median."""
    def share(pair):
        num, den = pair
        return {"share": num / den if den else None, "num": num, "den": den}
    return {"F1": share(F1), "F2": share(F2),
            "F3": {"gaps": 9, "first": share(first), "last": share(last), "inside": share(inside)},
            "F4": {"alive": {"median": alive, "n": 5}, "dead": {"median": dead, "n": 4}},
            "F5": {"median_m": F5, "n": 50}, "W1": share(W1), "W2": share(W2), "W3": share(W3)}


def test_fidelity_exit_codes():
    """Ruling 6: a gate (F1 to F3) that fails gives 1, even if a check fails too; otherwise a check (F4, F5) that fails
    gives 3; a statistic that is None on either side fails its gate or check; the warnings decide nothing; an error
    gives 2. The tolerances are the design's, and the report goes to <data>/runs/r028/fidelity-<config stem>-
    <profile>[.s<n>].json with its floats to 9 decimals and the pools' digest (ruling 8, and fix round 1: the config's
    name in it). With no cache of the pools, fidelity is an error (2) that says to run `pools` first."""
    assert broadcast_sim.GATE_TOLERANCES == {"F1": 0.05, "F2": 0.03, "F3": 0.05}
    assert broadcast_sim.CHECK_TOLERANCES == {"F4": 10.0, "F5": 0.2}
    ours = measure_of()

    def exit_of(**changes):
        return broadcast_sim.judge_fidelity(ours, measure_of(**changes))["exit"]

    assert exit_of() == 0
    assert exit_of(F1=(34, 100)) == 0 and exit_of(F1=(36, 100)) == 1 and exit_of(F1=(24, 100)) == 1
    assert exit_of(F2=(6, 100)) == 0 and exit_of(F2=(8, 100)) == 1
    for part in ("first", "last", "inside"):
        base = measure_of()["F3"][part]["num"]
        assert exit_of(**{part: (base + 4, 100)}) == 0 and exit_of(**{part: (base + 6, 100)}) == 1, part
    assert exit_of(F1=(0, 0)) == 1, "None on the copies' side fails its gate"
    assert exit_of(alive=20.0) == 0 and exit_of(alive=20.5) == 3 and exit_of(dead=-4.5) == 3, "within 10 frames"
    assert exit_of(dead=None) == 3 and exit_of(F5=1.25) == 3 and exit_of(F5=1.125) == 0
    assert exit_of(F1=(36, 100), alive=30.0) == 1, "a gate and a check fail: the gate's 1"
    assert exit_of(W1=(90, 100), W2=(0, 100), W3=(100, 100)) == 0, "the warnings decide nothing"
    assert broadcast_sim.judge_fidelity(measure_of(F5=None), ours)["exit"] == 3, "None on our side fails too"
    judged = broadcast_sim.judge_fidelity(ours, measure_of(F1=(36, 100)))
    assert set(judged["gates"]) == {"F1", "F2", "F3 first", "F3 last", "F3 inside"}
    assert set(judged["checks"]) == {"F4 alive", "F4 dead", "F5"} and set(judged["warnings"]) == {"W1", "W2", "W3"}
    row = judged["gates"]["F1"]
    assert (row["ours"], row["sim"], row["tolerance"], row["passed"]) == (0.3, 0.36, 0.05, False)
    assert abs(row["difference"] - 0.06) < 1e-12 and judged["gates"]["F2"]["passed"]
    assert judged["warnings"]["W1"] == {"ours": 0.04, "sim": 0.04}

    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "fill.toml", FILL_CONFIG)
        for changes, code in (({}, 0), ({"F1": (36, 100)}, 1), ({"alive": 30.0}, 3), ({"F1": (1, 3)}, 0)):
            def measured(config, changes=changes, **options):
                return broadcast_sim.fidelity_result(
                    config=config, dropped=(), profile=options["profile"], seeds=list(range(options["seeds"])),
                    match_ids=["RND001"], calibration=None, digest="ab" * 32, copies=options["seeds"], ours=ours,
                    sim=measure_of(**changes))
            with replaced(broadcast_sim, "fidelity", measured):
                got, text = main_of(["fidelity", "--profile", "gaps", "--seeds", "3", path, "--data-dir", root])
            assert got == code, (changes, got, text)
            report = strict_json(read(os.path.join(root, "runs", "r028", "fidelity-fill-gaps.json")))
            assert report["exit"] == code and report["broadcast_sim"] == "ab" * 32 and report["seeds"] == [0, 1, 2]
            assert report["config"]["text"] == FILL_CONFIG and report["sim"] == strict_json(json.dumps(
                benchmark_run.plain(measure_of(**changes))))
            assert "ab" * 32 in text and "F1" in text
        assert report["sim"]["F1"]["share"] == 0.333333333, "floats to 9 decimals"
        with replaced(broadcast_sim, "fidelity", measured):
            assert main_of(["fidelity", "--profile", "gaps", "--seeds", "1", path, "--data-dir", root])[0] == 0
        assert os.path.isfile(os.path.join(root, "runs", "r028", "fidelity-fill-gaps.s1.json")), "a one-seed look apart"
        other = write_text(root, "other.toml", FILL_CONFIG)
        with replaced(broadcast_sim, "fidelity", measured):
            assert main_of(["fidelity", "--profile", "gaps", "--seeds", "3", other, "--data-dir", root])[0] == 0
        assert os.path.isfile(os.path.join(root, "runs", "r028", "fidelity-other-gaps.json")), "another config apart"
        assert broadcast_sim.fidelity_path(root, "fill", "tracking", 3) == os.path.join(
            root, "runs", "r028", "fidelity-fill-tracking.json")

        def refusing(config, **options):
            raise broadcast_sim.BroadcastSimError("no pools here")

        def crashing(config, **options):
            raise RuntimeError("a bug in the declared code")

        for stand_in, words in ((refusing, ("FAILED", "no pools here")), (crashing, ("Traceback", "a bug"))):
            with replaced(broadcast_sim, "fidelity", stand_in):
                got, text = main_of(["fidelity", path, "--data-dir", root])
            assert got == 2 and all(word in text for word in words), (got, text)
        got, text = main_of(["fidelity", os.path.join(root, "absent.toml"), "--data-dir", root])
        assert got == 2 and "FAILED" in text, (got, text)
        # No cache of the pools under this data dir: the real fidelity refuses, before any match or calibration.
        got, text = main_of(["fidelity", "--profile", "gaps", path, "--data-dir", root])
        assert got == 2 and "FAILED" in text and "broadcast_sim.py pools" in text, (got, text)


def test_fidelity_pools_the_copies_end_to_end():
    """fidelity() serves every (match, seed) its simulated ball as the runner would (rulings 3 and 4: the config as run
    on DFL, without its match-only sources, which it prints, and its twin as the driver), lays the copies end to end
    with their blocks offset and measures them once with measure_fill, and judges them against BAR-ATH's measure as
    judge_fidelity does. A config that does not fill, and matches of different pitches, are refused."""
    matches = synthetic_matches()
    calibration = synthetic_calibration()
    pools = made_up_pools()
    ours = measure_of()
    drivers = []
    real = benchmark_run.predict

    def spy(config, inputs):
        drivers.append(config)
        return real(config, inputs)

    with tempfile.TemporaryDirectory() as root, match_only_source():
        config = load_config(write_text(root, "probed.toml", FILL_CONFIG + (
            '\n[[evidence]]\nkind = "probe_match_only"\nname = "probe"\n')))
        lines = []
        with replaced(benchmark_run, "predict", spy):
            result = broadcast_sim.fidelity(config, profile="gaps", seeds=2, matches=matches, calibration=calibration,
                                            pools=pools, ours=ours, data_dir=root, log=lines.append)
        again = broadcast_sim.fidelity(config, profile="gaps", seeds=2, matches=matches, calibration=calibration,
                                       pools=pools, ours=ours, data_dir=root, log=quiet)
        assert again == result, "the same inputs, the same result"
        dropped, removed = benchmark_run.drop_match_only(config)
        _, twin_only = wiring_configs(root)
    assert removed == ("probe",) and any("dropped" in line and "probe" in line for line in lines), lines
    assert result["dropped_match_only"] == ["probe"]
    assert len(drivers) == 4, "one decode per (match, seed): the twin's, and nothing else"
    twin = dataclasses.replace(dropped, ball_fill="none")
    assert all(driver == twin for driver in drivers), "the twin of the config as run on DFL"

    present, filled, codes, fill_xy, reference, graded, offset = [], [], [], [], [], [], 0
    for match in matches.values():
        pitch = np.array(match.pitch)
        for seed in (0, 1):
            atd = benchmark_run.make_atd(match, "gaps", seed, calibration)
            plain = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
            driven = benchmark_run.predict(twin, plain)[0]
            ball = broadcast_sim.simulate(atd.ball_present, np.asarray(match.ball_xy) * pitch, driven, match.graded,
                                          match.pitch, pools, np.random.default_rng(
                                              [seed, degrade.index_of(match.match_id), broadcast_sim.RNG_SALT]))
            served = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed, broadcast_ball=ball)
            present.append(atd.ball_present)
            filled.append(served.with_ball_fill("broadcast").ball_filled)
            codes.append(driven)
            fill_xy.append(ball * pitch)
            reference.append(np.asarray(match.ball_xy) * pitch)
            graded += [(half, start + offset, stop + offset) for half, start, stop in match.graded]
            offset += match.n_frames
    sim = broadcast_sim.measure_fill(np.concatenate(present), np.concatenate(filled), np.concatenate(codes), graded,
                                     np.concatenate(fill_xy), np.concatenate(reference), (105.0, 68.0))
    assert sim["W1"]["num"] > 0 and sim["F5"]["n"] > 0, "the copies are filled somewhere, with a reference"
    assert result["sim"] == sim and result["ours"] == ours
    judged = broadcast_sim.judge_fidelity(ours, sim)
    for key in ("gates", "checks", "warnings", "exit"):
        assert result[key] == judged[key], key
    assert result["copies"] == 4 and result["matches"] == ["RND001", "RND002"] and result["seeds"] == [0, 1]
    assert result["broadcast_sim"] == broadcast_sim.digest(pools)
    assert result["profile"] == degrade.profile_record("gaps")
    assert result["calibration"] == {"sha256": calibration_module.digest(calibration),
                                     "version": calibration_module.CALIBRATION_VERSION}
    assert result["config"]["text"] == config.text.replace("\r\n", "\n")
    json.dumps(benchmark_run.plain(result), allow_nan=False)

    options = dict(profile="gaps", seeds=1, calibration=calibration, pools=pools, ours=ours, log=quiet)
    expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.fidelity(twin_only, matches=matches,
                                                                                 **options), "twin", "fill")
    mixed = {"RND001": random_match("RND001"), "RND003": random_match("RND003", seed=2, pitch=(100.0, 50.0))}
    expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.fidelity(dropped, matches=mixed, **options),
                 "pitch")
    expect_error(benchmark_run.BenchmarkError, lambda: broadcast_sim.fidelity(dropped, matches=matches, **dict(
        options, profile="nope")), "nope")

    # Without pools or BAR-ATH's measure given, both are read from the cache (and nothing of BAR-ATH is loaded); with
    # no cache, fidelity refuses, saying to run `pools` first.
    asked = []

    def refused(*args, **kwargs):
        raise AssertionError("fidelity loads nothing of BAR-ATH")

    with replaced(broadcast_sim, "bar_ath_inputs", refused), replaced(broadcast_sim, "prepare_pools", refused), \
            replaced(broadcast_sim, "load_pools", lambda data_dir=None: asked.append(("pools", data_dir)) or pools), \
            replaced(broadcast_sim, "load_bar_ath_measure", lambda data_dir=None: (
                asked.append(("measure", data_dir)) or ours)):
        loaded = broadcast_sim.fidelity(dropped, profile="gaps", seeds=1, matches=matches, calibration=calibration,
                                        data_dir="D", log=quiet)
    assert asked == [("pools", "D"), ("measure", "D")], asked
    assert loaded["ours"] == ours and loaded["broadcast_sim"] == broadcast_sim.digest(pools)
    with tempfile.TemporaryDirectory() as empty, replaced(broadcast_sim, "prepare_pools", refused):
        expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.fidelity(
            dropped, profile="gaps", seeds=1, matches=matches, calibration=calibration, data_dir=empty, log=quiet),
            "broadcast_sim.py pools")


# -- the wiring: BAR-ATH, label-free (design 7.4) ---------------------------------------------------------------------

def test_tag_check_counts_fill_only_teacher_frames():
    """The tag check (ruling 7): the teacher is controlled_touch with conventions.TOUCH, run on the plain inputs and on
    the filled ones; the frames are the graded ones where its `home` is a number on the filled inputs and NaN on the
    plain ones; the measure is tag_stats.team_agreement with Smart Tagging's team votes on them; and the bar is the
    plain inputs' agreement over all graded frames minus 0.05, deciding only from 300 such frames that a tag votes
    on."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root, tags=TAGS)
        filled = inputs.with_ball_fill("broadcast")
        n = inputs.n_frames
        plain_home = np.full(n, np.nan)
        plain_home[100:150] = 0.99            # home, voted home on 140-150
        plain_home[400:420] = 0.01            # away, voted away on 410-420
        plain_home[500:510] = 0.99            # a touch the fill loses: not a fill-only frame
        filled_home = np.array(plain_home)
        filled_home[500:510] = np.nan
        filled_home[150:200] = 0.99           # fill-only, home: voted home on 150-190, no vote on 190-200
        filled_home[200:230] = 0.01           # fill-only, away: voted away
        filled_home[420:440] = 0.99           # fill-only, home: voted away
        calls = []

        def teacher(run, **params):
            calls.append((run.inputs, params))
            return types.SimpleNamespace(alive=np.full(n, np.nan),
                                         home=filled_home if run.inputs is filled else plain_home)

        with replaced(controlled_touch, "compute", teacher):
            result = broadcast_sim.tag_check(inputs, filled)
        assert [params for _, params in calls] == [conventions.TOUCH, conventions.TOUCH], calls
        assert {id(given) for given, _ in calls} == {id(inputs), id(filled)}
        graded = np.asarray(inputs.graded_mask)
        fill_only = graded & np.isfinite(filled_home) & np.isnan(plain_home)
        missing = ~np.asarray(inputs.atd.ball_present)
        team, tags = smart_tagging.votes(inputs), smart_tagging.tags(inputs)
        assert result["fill_only"] == tag_stats.team_agreement(team, filled_home, tags, graded=fill_only,
                                                               ball_missing=missing)
        assert result["plain"] == tag_stats.team_agreement(team, plain_home, tags, graded=graded, ball_missing=missing)
    # worked out by hand: 40 + 30 agree and 20 disagree on the 90 fill-only frames with a vote; 20 of 20 on the plain
    assert result["fill_only_frames"] == int(fill_only.sum()) == 100
    assert result["fill_only"]["n"] == 90 and result["fill_only"]["share"] == 70 / 90
    assert result["plain"]["n"] == 20 and result["plain"]["share"] == 1.0
    assert result["bar"] == 1.0 - 0.05 and result["decisive"] is False and result["passed"] is True, (
        "below the bar, but on fewer than 300 frames: reported, deciding nothing")
    assert (broadcast_sim.MIN_TAG_FRAMES, broadcast_sim.TAG_MARGIN) == (300, 0.05)


def test_bar_ath_exit_codes():
    """Ruling 6: the alignment band (d_1 and d_2 within 2 frames of 4,014 and 26,919) and the live share (at least 0.85)
    come first, 3 if either fails (an alignment refused too); then 1 if the tag check fails where it decides (300 frames
    or more), if a graded frame is X, or if D is outside [30%, 62%] of the graded frames; else 0; an error gives 2. The
    report goes to <data>/runs/r028/bar-ath-<config stem>.json."""
    assert broadcast_sim.EXPECTED_OFFSETS == {1: 4014, 2: 26919} and broadcast_sim.OFFSET_BAND == 2
    assert broadcast_sim.MIN_LIVE_SHARE == 0.85 and broadcast_sim.D_SHARE_BAND == (0.30, 0.62)

    def aligned(refused=None, **changes):
        summary = dict(copy.deepcopy(ALIGNED), **changes)
        return broadcast_sim.judge_alignment(summary, refused)

    assert aligned()["passed"] and aligned(offsets={1: 4016, 2: 26917})["passed"]
    assert not aligned(offsets={1: 4017, 2: 26919})["passed"] and not aligned(offsets={1: 4014, 2: 26922})["passed"]
    assert aligned(live_share=0.85)["passed"] and not aligned(live_share=0.849)["passed"]
    assert not aligned(offsets={1: 4014})["passed"], "a half without its offset"
    refused = broadcast_sim.judge_alignment(None, "half 1: no offset agrees")
    assert not refused["passed"] and refused["refused"] == "half 1: no offset agrees"

    def tags(n, share, plain=0.75):
        return broadcast_sim.judge_tags({"n": 2000, "share": plain}, {"n": n, "share": share}, n + 10)

    assert tags(300, 0.71)["passed"] and not tags(300, 0.69)["passed"] and tags(299, 0.1)["passed"]
    assert tags(300, 0.69)["decisive"] and not tags(299, 0.1)["decisive"]
    assert not broadcast_sim.judge_tags({"n": 0, "share": None}, {"n": 400, "share": 0.9}, 400)["passed"]

    def sanity(d_frames, x_frames=0):
        codes = np.full(120, X, dtype=np.int8)
        codes[10:110] = H
        codes[10:10 + d_frames] = D
        codes[110 - x_frames:110] = X
        return broadcast_sim.judge_sanity(codes, [(1, 10, 110)])

    assert sanity(30)["passed"] and sanity(62)["passed"] and not sanity(29)["passed"] and not sanity(63)["passed"]
    assert sanity(30)["d_share"] == 0.3 and sanity(40, x_frames=1)["x_graded"] == 1 and sanity(40)["x_graded"] == 0
    assert not sanity(40, x_frames=1)["passed"], "an X on a graded frame"

    good, bad_tag, bad_sanity = tags(300, 0.71), tags(300, 0.69), sanity(63)
    assert broadcast_sim.bar_ath_exit(aligned(), good, sanity(40)) == 0
    assert broadcast_sim.bar_ath_exit(aligned(), bad_tag, sanity(40)) == 1
    assert broadcast_sim.bar_ath_exit(aligned(), good, bad_sanity) == 1
    assert broadcast_sim.bar_ath_exit(aligned(live_share=0.8), None, None) == 3
    assert broadcast_sim.bar_ath_exit(refused, bad_tag, bad_sanity) == 3, "the STOP first"

    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "fill.toml", FILL_CONFIG)
        for alignment, tag, checked, code in ((aligned(), good, sanity(40), 0), (aligned(), bad_tag, sanity(40), 1),
                                              (aligned(offsets={1: 4000, 2: 26919}), None, None, 3)):
            def measured(config, alignment=alignment, tag=tag, checked=checked, **options):
                return broadcast_sim.bar_ath_result(config=config, alignment=alignment, digest="cd" * 32,
                                                    fill={"filled_frames": 7, "graded_frames": 100}, tag=tag,
                                                    sanity=checked, reported=None)
            with replaced(broadcast_sim, "bar_ath_inputs", lambda data_dir: None), \
                    replaced(broadcast_sim, "bar_ath", measured):
                got, text = main_of(["bar-ath", path, "--data-dir", root])
            assert got == code, (code, got, text)
            report = strict_json(read(os.path.join(root, "runs", "r028", "bar-ath-fill.json")))
            assert report["exit"] == code and report["broadcast_sim"] == "cd" * 32
            assert "4,014" in text or "4,000" in text, "what was measured is printed"

        def crashing(config, **options):
            raise RuntimeError("a bug in the declared code")

        with replaced(broadcast_sim, "bar_ath_inputs", lambda data_dir: None), \
                replaced(broadcast_sim, "bar_ath", crashing):
            got, text = main_of(["bar-ath", path, "--data-dir", root])
        assert got == 2 and "Traceback" in text, (got, text)


def test_bar_ath_end_to_end_on_a_synthetic_match():
    """bar_ath() on BAR-ATH in miniature: the alignment and the live share read from the inputs' summary; then the tag
    check, the candidate's labels (the config through the benchmark's predict, the fill applied there) with their X and
    D sanity, main.toml's labels beside them for the possessions per half and the frames relabelled, and the filled
    frames by the candidate's state. At a STOP nothing after the alignment runs; a config that does not fill is
    refused."""
    with tempfile.TemporaryDirectory() as root:
        inputs = bar_ath_like(root, tags=TAGS)
        fill, twin = wiring_configs(root)
        pools = made_up_pools()
        lines = []
        with replaced(broadcast_sim, "MAIN_CONFIG", twin.path):
            result = broadcast_sim.bar_ath(fill, inputs=inputs, pools=pools, log=lines.append)
        assert result["stage"] == "complete" and result["alignment"]["passed"], result["alignment"]
        assert result["broadcast_sim"] == broadcast_sim.digest(pools)
        codes = benchmark_run.predict(fill, inputs)[0]
        main_codes = benchmark_run.predict(twin, inputs)[0]
        graded = np.asarray(inputs.graded_mask)
        assert result["sanity"] == broadcast_sim.judge_sanity(codes, inputs.graded)
        assert result["sanity"]["d_share"] == float((codes[graded] == D).sum()) / graded.sum()
        view = inputs.with_ball_fill("broadcast")
        assert result["fill"]["filled_frames"] == int(view.ball_filled.sum()) > 0
        assert result["fill"]["graded_frames"] == int(graded.sum())
        assert result["tag"] == broadcast_sim.tag_check(inputs, view)
        diff = label_diff.diff(main_codes, codes, inputs)
        reported = result["reported"]
        assert reported["relabelled_share"] == diff["changed_share"]
        for half, (before, after) in diff["possessions"].items():
            row = reported["possessions"][half]
            assert (row["main"], row["candidate"]) == (before, after)
            assert row["warn"] == (abs(after - before) > 0.25 * before)
        pitch = np.array(inputs.pitch)
        state = broadcast_sim.measure_fill(np.asarray(inputs.atd.ball_present), np.asarray(view.ball_filled), codes,
                                           inputs.graded, np.asarray(inputs.given_ball) * pitch,
                                           np.full((inputs.n_frames, 2), np.nan), inputs.pitch)
        assert reported["filled_by_state"] == {"alive": state["F1"]["num"], "dead": state["F2"]["num"],
                                               "other": int(view.ball_filled.sum()) - state["F1"]["num"]
                                               - state["F2"]["num"]}
        assert reported["W3"] == state["W3"] and reported["W1"] == state["W1"]
        assert result["exit"] == broadcast_sim.bar_ath_exit(result["alignment"], result["tag"], result["sanity"])
        json.dumps(benchmark_run.plain(result), allow_nan=False)
        assert any("added" in line for line in lines), "the fill's added time is printed"

        for summary in (dict(copy.deepcopy(ALIGNED), offsets={1: 4017, 2: 26919}),
                        dict(copy.deepcopy(ALIGNED), live_share=0.84), AlignmentError("half 1: no offset agrees")):
            stopped = bar_ath_like(root, tags=TAGS, summary=summary)
            touched = []
            with replaced(controlled_touch, "compute", lambda *arguments, **keywords: touched.append(1)):
                halted = broadcast_sim.bar_ath(fill, inputs=stopped, pools=pools, log=quiet)
            assert halted["exit"] == 3 and halted["stage"] == "alignment", halted
            assert halted["tag"] is None and halted["sanity"] is None and touched == [], "nothing after the STOP"
        assert halted["alignment"]["refused"] == "half 1: no offset agrees"
        expect_error(broadcast_sim.BroadcastSimError, lambda: broadcast_sim.bar_ath(twin, inputs=inputs, pools=pools,
                                                                                    log=quiet), "twin", "fill")
