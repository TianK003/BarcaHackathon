"""Checks of the `tracking` profile of the degrader (pipeline/benchmark/degrade.py): the players as tracklets in 46
slots, the wrong-team episodes, the NaN-speed runs and the smoothing, and how the profile nests in `gaps`.

They run on a steady match (steady_match, in checks/common.py with the other fixtures it shares with the fidelity
checks), whose players are on the pitch for whole stretches, as in a real match, on pairs that come close in windows
(pairs_match) and on a made-up Calibration whose tracklet lengths and slot counts give figures worked out by hand (d =
3/55, a mean tracked span of 53 frames). The checks that need real data print aggregates only:
test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches (--dfl: the 7 downloaded DFL matches, a made-up
Calibration) and the two that need our own match as well (--data, NOT RUN where the DFL files are not there),
test_data_tracking_profile_on_a_real_match_with_our_real_calibration and
test_data_tracking_constants_follow_from_the_measurements. Every other check here reads nothing from the data
directory. The transplant, the draws common to every profile, what the `gaps` profile touches and what no profile may
see are checks/degrade.py's; the cap, the shortest-piece rule and the scale of the tracked spans are checks/cap.py's.
"""

import dataclasses
import json
import math
import os
import types

import numpy as np

from pipeline.benchmark import calibration, degrade, dfl
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import (MADE_UP_D, MADE_UP_MEAN, MADE_UP_MU, MADE_UP_SPEED_NAN,
                                              MADE_UP_WRONG_TEAM, NAN_RUNS, RECORDED_MEDIAN_S,
                                              RECORDED_OURS_MEDIAN_S, NotRun, ball_bytes, episode_parameters,
                                              is_carried, leaks, made_up_calibration, pairs_match, real_match,
                                              reference_episodes, smoothed_plain, steady_match,
                                              tracked_scale_from_the_record, without_new_parts)
from pipeline.selftest import expect_error

# Params
# The record of what was drawn (info["players"]): the parameters and counts, per half and in all. No frame.
HALF_KEYS = ("tracklets", "slots_used", "on_pitch_frames", "lost_frames", "onsets", "episodes", "skipped_far",
             "skipped_overlap", "episode_frames", "speed_nan_runs", "speed_nan_frames", "capped_frames",
             "cap_slot_frames", "cap_pieces_cut", "short_pieces", "short_slot_frames", "short_episode_pieces")
PLAYERS_KEYS = ("d", "mean_tracked_frames", "mean_lost_frames", "tracklets", "on_pitch_frames", "lost_frames",
                "lost_share", "scale", "wrong_team_target", "mean_episode_frames", "onset_rate", "smooth_window",
                "speed_nan_target", "onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames",
                "speed_nan_runs", "speed_nan_frames", "speed_nan_share", "capped_frames", "cap_slot_frames",
                "cap_pieces_cut", "short_pieces", "short_slot_frames", "short_episode_pieces", "tracked_scale",
                "min_piece_frames", "halves")


# -- the tracking profile ---------------------------------------------------------------------------------------------

def present_runs(column):
    """[(first, stop)] of the maximal runs of True in a 1-D bool list, written out one frame at a time."""
    runs, first = [], None
    for index, flag in enumerate(column):
        if flag and first is None:
            first = index
        elif not flag and first is not None:
            runs.append((first, index))
            first = None
    if first is not None:
        runs.append((first, len(column)))
    return runs


def slot_idle_gaps_s(present, spans, fps):
    """Seconds of every idle gap of a slot: a maximal run of absent frames with a present frame of the same slot on
    both sides, inside one span of (half, start, stop) rows (nothing crosses the end of a span).

    What the slot persistence statistics of A09 are made of, taken from a presence array alone so that the same
    function measures ours (the ATD cache) and the degraded match: a slot that is freed on one frame and taken again on
    the very next has no gap, and the absence before a slot's first frame and after its last in a span is no gap
    either (it is not idle between two uses). A float64 array, in order of span, slot and time."""
    present = np.asarray(present)
    gaps = []
    for _, start, stop in spans:
        for slot in range(present.shape[1]):
            seen = np.flatnonzero(present[start:stop, slot])
            steps = np.diff(seen) - 1
            gaps.append(steps[steps > 0])
    return (np.concatenate(gaps) if gaps else np.zeros(0)) / fps


def same_slot_returns(tracklets):
    """(returns, same): how many tracklets start after a lost span of their player's, and how many of them land back in
    the slot the player had before.

    `tracklets` is {half: [Tracklet]} as degrade.degraded_players gives it. A player's tracklets of a half are taken in
    time order; one is a *return* if it starts after the previous one has stopped (start > previous stop), so that a
    handover with no frame between them, which is an identity switch and not a loss, is not one. It is *same-slot* if
    its slot is the previous one's. Only the degraded side has identities: ours has none (a slot is not a player)."""
    returns = same = 0
    for records in tracklets.values():
        by_player = {}
        for t in records:
            by_player.setdefault(t.player, []).append(t)
        for player_records in by_player.values():
            player_records.sort(key=lambda t: t.start)
            for before, after in zip(player_records, player_records[1:]):
                if after.start > before.stop:
                    returns += 1
                    same += after.slot == before.slot
    return returns, same


def gap_figures(gaps):
    """(count, median in s, share of at most 3 s, share of over 60 s) of idle gaps in seconds; NaN shares of none."""
    if not gaps.size:
        return 0, math.nan, math.nan, math.nan
    return int(gaps.size), float(np.median(gaps)), float((gaps <= 3).mean()), float((gaps > 60).mean())


def test_slot_persistence_helpers_count_what_they_say():
    """Idle gaps and same-slot returns, on hand-built arrays and records, worked out one frame at a time.

    Two slots over 12 frames: slot 0 is present on 0-1, 4-6 and 8, slot 1 on 2, 6-9 and 11. Over the whole span the gaps
    are slot 0's 2 frames (2-3) and 1 frame (7), and slot 1's 3 frames (3-5) and 1 frame (10): the absence before a
    slot's first frame (slot 1 on 0-1) and after its last (slot 0 on 9-11) is not a gap. A span that starts on row 2
    cuts off what lies before it, and two spans do not join: split at row 6 the gaps are slot 0's 2 and 1 frames and
    slot 1's 1 frame (10), and the slot 1 absence of 3-5 ends the first span, so it is no gap."""
    columns = ([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    present = np.array(columns, dtype=bool).T

    def plain(spans):
        """The same gaps with one loop over the frames of each slot, no numpy."""
        found = []
        for _, start, stop in spans:
            for slot in range(len(columns)):
                last = None
                for row in range(start, stop):
                    if columns[slot][row]:
                        if last is not None and row - last > 1:
                            found.append(row - last - 1)
                        last = row
        return found

    for spans, expected in (([(1, 0, 12)], [2, 1, 3, 1]), ([(1, 2, 12)], [1, 3, 1]),
                            ([(1, 0, 6), (2, 6, 12)], [2, 1, 1])):
        got = slot_idle_gaps_s(present, spans, 25)
        assert sorted(got * 25) == sorted(plain(spans)) == sorted(expected), (spans, got * 25, plain(spans))
        assert got.dtype == np.float64
    assert slot_idle_gaps_s(present, [(1, 4, 7)], 25).size == 0, "present all through the span: no gap"
    assert slot_idle_gaps_s(np.zeros((5, 3), dtype=bool), [(1, 0, 5)], 25).size == 0, "never present: no gap"
    assert np.allclose(slot_idle_gaps_s(present, [(1, 0, 12)], 50), [0.04, 0.02, 0.06, 0.02]), "seconds, at the rate"
    figures = gap_figures(np.array([1.0, 2.0, 3.0, 4.0, 70.0]))
    assert figures == (5, 3.0, 3 / 5, 1 / 5), figures
    assert gap_figures(np.zeros(0))[0] == 0 and math.isnan(gap_figures(np.zeros(0))[1])

    T = degrade.Tracklet
    records = {1: [T(0, 0, 10, 3), T(0, 15, 20, 3), T(0, 25, 30, 5), T(1, 0, 10, 1), T(1, 10, 20, 2),
                   T(2, 5, 8, 7), T(2, 9, 12, 8), T(2, 20, 21, 7)],
               2: [T(0, 100, 110, 4), T(0, 112, 120, 4), T(1, 100, 105, 6)]}
    # Player 0 of half 1: 3 -> 3 (return, same), 3 -> 5 (return, other); player 1 hands over on the frame (not a
    # return); player 2: 7 -> 8 (return, other) and 8 -> 7 (return, other). Half 2, player 0: 4 -> 4 (return, same).
    assert same_slot_returns(records) == (5, 2)
    assert same_slot_returns({1: [T(0, 0, 5, 1)]}) == (0, 0) and same_slot_returns({}) == (0, 0)


def slot_use(present, team, spans):
    """{0: (pool, ever, tenth, hundredth), 1: (...)}: for each team, how many slots it has, how many are present on at
    least one frame of the spans, on at least a tenth of the frames of the spans and on at least a hundredth of them.

    What the two sides of the slot comparison are made of (R004, the README): the same function counts the degraded
    match and ours, over the rows of the spans (half, start, stop), all halves together. The shares are compared in
    integers (10 x the frames present against the frames, so that exactly a tenth counts), and a slot present only
    outside the spans does not count. Aggregates only."""
    present, team = np.asarray(present), np.asarray(team)
    frames = sum(stop - start for _, start, stop in spans)
    seen = np.zeros(present.shape[1], dtype=np.int64)
    for _, start, stop in spans:
        seen += present[start:stop].sum(axis=0, dtype=np.int64)
    out = {}
    for number in (0, 1):
        mine = seen[team == number]
        out[number] = (int(mine.size), int((mine >= 1).sum()), int((10 * mine >= frames).sum()),
                       int((100 * mine >= frames).sum()))
    return out


def test_slot_use_counts_what_it_says():
    """Slots ever present, on at least a tenth and on at least a hundredth of the frames of the spans, per team.

    240 rows, spans of 200 frames in all (rows 10-110 and 130-229, each 100). Home slots 0 to 3 are present on 1, 2, 20
    and 19 of those frames: ever 4, a hundredth (2 frames of 200, exactly) 3, a tenth (20 frames, exactly) 1; a slot
    with 1 frame is under a hundredth (0.5%), 19 frames are under a tenth (9.5%). Away slot 4 is present only outside the
    spans (rows 0-9, 230-239): it is not used; away slot 5 is present on every frame of the spans: 1 of 2 in all three
    counts. The frames outside the spans, and a frame at the very edge, count for nothing in the shares."""
    present = np.zeros((240, 6), dtype=bool)
    spans = [(1, 10, 110), (2, 130, 230)]
    present[10:11, 0] = True                      # 1 frame
    present[50:52, 1] = True                      # 2 frames
    present[100:110, 2] = True                    # 10 + 10 frames, over the two spans
    present[130:140, 2] = True
    present[20:39, 3] = True                      # 19 frames
    present[0:10, 4] = True
    present[230:240, 4] = True
    present[10:110, 5] = True
    present[130:230, 5] = True
    team = np.array([0, 0, 0, 0, 1, 1])
    assert slot_use(present, team, spans) == {0: (4, 4, 1, 3), 1: (2, 1, 1, 1)}, slot_use(present, team, spans)
    plain = {}
    frames = [row for _, start, stop in spans for row in range(start, stop)]
    for number in (0, 1):
        counts = [sum(1 for row in frames if present[row, slot]) for slot in range(6) if team[slot] == number]
        plain[number] = (len(counts), sum(c >= 1 for c in counts), sum(c / len(frames) >= 0.1 for c in counts),
                         sum(c / len(frames) >= 0.01 for c in counts))
    assert plain == slot_use(present, team, spans), (plain, slot_use(present, team, spans))
    assert len(frames) == 200


def test_tracking_parameters_are_worked_out_from_the_calibration():
    """d from the frames with more than 18 present, the lengths in frames at the match's rate, mu from both.

    The made-up Calibration has, over both halves (PRESENT_COUNTS), 290 frames with 22 present, 174 with 20, 116 with 19
    and 60 with 18 or fewer (30 with 18, 18 with 10 and 12 with 7): the last are left out (CAP is a strict bound, 18
    itself is in the tail), so d = 1 - 20.8 / 22 = 3 / 55. Its tracklets are 0.4, 1, 2, 3.2 and 4 s: 10, 25, 50, 80 and
    100 frames at 25 Hz, mean 53, and 20, 50, 100, 160 and 200 at 50 Hz. A lost span has the mean mu = 53 * d / (1 - d)
    = 159 / 52 frames."""
    calibration = made_up_calibration()
    assert degrade.CAP == 18
    assert abs(degrade.lost_share(calibration.present_counts) - 3 / 55) < 1e-12
    parameters = degrade.calibrated_parameters(calibration, 25, tracked_scale=1.0)
    assert parameters.lengths.tolist() == [10, 25, 50, 80, 100] and parameters.lengths.dtype == np.int64
    assert parameters.mean_tracked == MADE_UP_MEAN and abs(parameters.d - MADE_UP_D) < 1e-12
    assert abs(parameters.mu - MADE_UP_MU) < 1e-12 and abs(MADE_UP_MU - 159 / 52) < 1e-12
    assert not parameters.lengths.flags.writeable
    other_rate = degrade.calibrated_parameters(calibration, 50, tracked_scale=1.0)
    assert other_rate.lengths.tolist() == [20, 50, 100, 160, 200] and other_rate.mean_tracked == 106.0
    assert abs(other_rate.mu - 106 * (3 / 55) / (52 / 55)) < 1e-12

    # A tracked span is at least a frame, and the seconds are rounded as round() does (halves to even).
    seconds = (0.0, 0.01, 0.02, 0.07, 0.1, 0.14, 4.0, 40.0)   # x 25: 0, 0.25, 0.5, 1.75, 2.5, 3.5, 100, 1000 frames
    short = without_new_parts(calibration, tracklet_lengths_s=np.array(seconds))
    assert degrade.calibrated_parameters(short, 25, tracked_scale=1.0).lengths.tolist() == [
        max(1, round(s * 25)) for s in seconds] == [1, 1, 1, 2, 2, 4, 100, 1000]
    # Only the frames above CAP: a 19 is in and an 18 is not, and the halves are pooled.
    counts = {1: np.array([18, 18, 19], dtype=np.int16), 2: np.array([22, 3], dtype=np.int16)}
    assert abs(degrade.lost_share(counts) - (1 - 20.5 / 22)) < 1e-12
    # Nobody lost: d = 0, no lost span (mu = 0).
    nobody = degrade.calibrated_parameters(without_new_parts(
        calibration, present_counts={1: np.full(10, 22, dtype=np.int16)}, tracklet_lengths_s=np.array([1.0])), 25,
        tracked_scale=1.0)
    assert (nobody.d, nobody.mu, nobody.mean_tracked) == (0.0, 0.0, 25.0)

    expect_error(ValueError, lambda: degrade.lost_share({1: np.array([18, 4, 0], dtype=np.int16)}),
                 "more than 18", "present_counts")
    expect_error(ValueError, lambda: degrade.lost_share({}), "present_counts")
    # A lost span is at least a frame, so a mean below 1 cannot be drawn: it is refused, and says why.
    nearly = {1: np.array([22] * 999 + [21], dtype=np.int16)}
    expect_error(ValueError, lambda: degrade.tracking_parameters([1], degrade.lost_share(nearly)), "lost span",
                 "at least 1 frame")
    expect_error(ValueError, lambda: degrade.tracking_parameters([], 0.1), "lengths")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3, 0], 0.1), "lengths", "at least 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3], 1.0), "d", "below 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([3], -0.1), "d", "at least 0")
    expect_error(ValueError, lambda: degrade.calibrated_parameters(types.SimpleNamespace(), 25), "present_counts",
                 "tracklet_lengths_s", "wrong_team_episodes_s", "wrong_team_share", "speed_nan_runs", "speed_nan_share",
                 "calibration.load()")
    # A Calibration of an older layout, with the tracklets and not the new parts, is named for what it lacks.
    older = types.SimpleNamespace(present_counts=calibration.present_counts,
                                  tracklet_lengths_s=calibration.tracklet_lengths_s)
    expect_error(ValueError, lambda: degrade.calibrated_parameters(older, 25), "wrong_team_episodes_s",
                 "speed_nan_runs", "calibration.load()")


def test_walk_alternates_tracked_and_lost_spans_and_is_cut_by_the_run():
    """A run starts with a tracked span, a lost span of at least a frame follows each, and the run's end cuts."""
    parameters = degrade.calibrated_parameters(made_up_calibration(), 25)
    lengths = set(parameters.lengths.tolist())
    seen, ends = set(), set()
    for seed in range(60):
        rng = degrade.rng_for(seed, 0, "tracklets", 1)
        for length in (1, 7, 60, 500, 3000):
            spans = degrade.walk_run(rng, length, parameters)
            assert spans and spans[0][0] == 0 and spans[-1][1] <= length, (seed, length, spans)
            assert all(isinstance(value, int) for span in spans for value in span), "plain ints, for JSON and slices"
            assert all(stop - start >= 1 for start, stop in spans)
            assert all(spans[k][1] + 1 <= spans[k + 1][0] for k in range(len(spans) - 1)), "a lost span between each"
            for start, stop in spans[:-1]:
                assert stop - start in lengths, "a span that is followed by another was drawn whole"
                seen.add(stop - start)
            start, stop = spans[-1]
            assert stop - start in lengths or stop == length, "the last is whole, or cut by the end of the run"
            ends.add(stop == length)
            assert stop - start <= max(lengths)
    assert seen == lengths, "every length is drawn"
    assert ends == {True, False}, "a run ends inside a tracked span, and inside a lost one"
    # A run shorter than its first tracked span is that span, cut: one tracklet, the whole run.
    long_first = degrade.tracking_parameters([100], 0.1)
    assert degrade.walk_run(degrade.rng_for(0, 0, "tracklets", 1), 40, long_first) == [(0, 40)]
    # Nobody is lost (mu = 0): the run is one tracklet, whatever the lengths, and nothing is drawn.
    never = degrade.tracking_parameters([5, 9], 0.0)
    rng = degrade.rng_for(0, 0, "tracklets", 1)
    before = rng.bit_generator.state
    assert degrade.walk_run(rng, 1000, never) == [(0, 1000)] and rng.bit_generator.state == before
    # The lost spans are at least 1 even when mu is exactly 1.
    sticky = degrade.tracking_parameters([1], 0.5)    # mu = 1 * 0.5 / 0.5 = 1: a frame tracked, a frame lost
    assert sticky.mu == 1.0
    spans = degrade.walk_run(degrade.rng_for(0, 0, "tracklets", 1), 100, sticky)
    assert spans == [(2 * k, 2 * k + 1) for k in range(50)], spans


def test_lost_share_on_long_synthetic():
    """On long runs the share lost is d, the tracked spans have the lengths' mean and the lost ones have mean mu.

    The mean tracked span is 100 frames (50 to 150, every length once) and d is 0.05 and 0.3, whose mu are 100 * d /
    (1 - d) = 5.26 and 42.86 frames: a mu that forgot the (1 - d) would give a lost share of 0.0476 and 0.2308, which
    the second d tells from d by 0.07. Ten runs of 300,000 frames each: the share's own spread is about 0.001, and the
    brief's tolerance of 0.01 is ten times it. The whole tracked spans have a mean of 100 to within 5 standard errors
    (below)."""
    lengths = np.arange(50, 151)
    for d in (0.05, 0.3):
        parameters = degrade.tracking_parameters(lengths, d)
        assert parameters.mean_tracked == 100.0 and abs(parameters.mu - 100 * d / (1 - d)) < 1e-9
        total = tracked = 0
        whole, lost = [], []
        for run in range(10):
            spans = degrade.walk_run(degrade.rng_for(1, 0, "tracklets", run), 300_000, parameters)
            total += 300_000
            tracked += sum(stop - start for start, stop in spans)
            whole += [stop - start for start, stop in spans[:-1]]
            lost += [spans[k + 1][0] - spans[k][1] for k in range(len(spans) - 1)]
        share = 1 - tracked / total
        assert abs(share - d) < 0.01, (d, share)
        # The mean of the n whole tracked spans, each drawn uniformly from 50..150 (SD 29.15), has a standard error of
        # 29.15 / sqrt(n): about 0.17 frames for the 28,500 spans of d = 0.05 and 0.20 for the 21,000 of d = 0.3. The
        # bound is 5 of them, so that a mean that is off by a fraction of a frame is caught.
        error = np.std(lengths) / math.sqrt(len(whole))
        assert abs(np.mean(whole) - 100) < 5 * error and error < 0.25, (d, np.mean(whole), error)
        assert abs(np.mean(lost) / parameters.mu - 1) < 0.05, (d, np.mean(lost), parameters.mu)
        assert min(lost) >= 1 and max(lost) > 3 * parameters.mu, "a geometric tail, not a constant"

    # The same through the profile, on a steady match of 2 x 20,000 rows: what it reports and what the arrays show.
    match = steady_match(rows=(20000, 20000))
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    atd, info = degrade.degrade(view, "tracking", calibration, 0)
    clean_present = ~np.isnan(match.players_xy[:, :, 0])
    on_pitch = int(clean_present.sum())
    shown = int(atd.players_present.sum())
    assert abs(1 - shown / on_pitch - MADE_UP_D) < 0.01, (1 - shown / on_pitch, MADE_UP_D)
    assert abs(info["players"]["lost_share"] - (1 - shown / on_pitch)) < 1e-9
    assert info["players"]["on_pitch_frames"] == on_pitch and abs(info["players"]["d"] - MADE_UP_D) < 1e-9


def test_tracklets_never_overlap_in_a_slot():
    """A slot carries one tracklet at a time: no two overlap, and every slot-frame of the arrays has one writer.

    Many short tracklets (3 to 20 frames, 30% of the time lost) so that slots are freed and taken over again all the
    time. Checked on the records and on the arrays: the finite slot-frames are as many as the tracklets' frames, which
    they would not be if two tracklets had written the same one."""
    view = degrade.tracking_view(steady_match(rows=(400, 300)))
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    reused = jumped = 0
    for seed in range(6):
        xy, speed, tracklets, _ = degrade.degraded_players(view, parameters, seed)
        frames = 0
        for half, records in tracklets.items():
            assert len(records) > 30, (seed, half, len(records))
            by_slot, by_player = {}, {}
            for t in records:
                by_slot.setdefault(t.slot, []).append((t.start, t.stop))
                by_player.setdefault(t.player, set()).add(t.slot)
                frames += t.stop - t.start
            for slot, spans in by_slot.items():
                spans.sort()
                assert all(first[1] <= second[0] for first, second in zip(spans, spans[1:])), (seed, half, slot)
                reused += len(spans) > 1
            jumped += any(len(slots) > 1 for slots in by_player.values())
        assert int(np.isfinite(xy[:, :, 0]).sum()) == frames, "two tracklets wrote one slot-frame"
        assert (np.isfinite(xy[:, :, 0]) == np.isfinite(xy[:, :, 1])).all(), "x and y are present together"
    assert reused > 20 and jumped > 5, "slots are taken over and players change slot, as in ours"


def test_each_tracklet_takes_the_lowest_free_slot_of_its_pool():
    """In the order of (start, player), a tracklet takes the lowest slot of its team's pool that is free at its start.

    Worked out from the records alone: the slots below the one taken were all busy at the start (an earlier tracklet,
    still tracked on that frame or on the one before it, had each of them), and the one taken was not. A slot is free for
    a tracklet starting at t only when its previous tracklet stopped before t: one that ends at t (half-open: its last
    frame is t - 1) still holds the slot, and one that ends at t - 1 (so that row t - 1 is empty) has freed it. Since
    tracking version 3 there are never two tracklets of a slot on adjacent rows: that was the bug of versions 1 and 2."""
    view = degrade.tracking_view(steady_match(rows=(400, 300)))
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    checked = taken_over = after_one_empty_row = 0
    for seed in range(4):
        _, _, tracklets, _ = degrade.degraded_players(view, parameters, seed)
        for records in tracklets.values():
            assert [(t.start, t.player) for t in records] == sorted((t.start, t.player) for t in records)
            for index, t in enumerate(records):
                pool = range(0, 24) if view.player_team[t.player] == 0 else range(24, 46)
                busy = {u.slot for u in records[:index] if u.stop >= t.start}
                assert t.slot in pool and t.slot not in busy, (seed, t)
                assert all(slot in busy for slot in pool if slot < t.slot), (seed, t, sorted(busy))
                checked += 1
                taken_over += any(u.stop == t.start and u.slot == t.slot for u in records[:index])
                after_one_empty_row += any(u.stop + 1 == t.start and u.slot == t.slot for u in records[:index])
    assert checked > 500 and taken_over == 0, "a slot freed on a frame is not taken on that frame"
    assert after_one_empty_row > 0, "the row after it is: the rule is `before`, and both sides of it are exercised"


def test_slot_team_matches_pool():
    """slot_team is 24 home slots then 22 away, and every slot of a pool carries only players of its team."""
    match = steady_match()
    view = degrade.tracking_view(match)
    calibration = without_new_parts(made_up_calibration())     # nobody is carried by the other team's pool here
    # The smoothing moves a position off the player's own, so who is in a slot is read from positions that are not
    # smoothed (the same walk and slots: the draws do not depend on the smoothing).
    plain = dataclasses.replace(degrade.calibrated_parameters(calibration, match.fps), smooth=0)
    assert degrade.SMOOTH_WINDOW > 0 and plain.smooth == 0
    for seed in (0, 1, 2):
        atd, info = degrade.degrade(view, "tracking", calibration, seed)
        unsmoothed = degrade.degraded_players(view, plain, seed)[0]
        assert np.array_equal(np.isnan(unsmoothed), np.isnan(atd.players_xy)), "the same slots, smoothed or not"
        assert atd.slot_team.dtype == np.int8 and atd.slot_team.tolist() == [0] * 24 + [1] * 22
        assert (degrade.HOME_SLOTS, degrade.AWAY_SLOTS, degrade.N_SLOTS) == (24, 22, 46)
        assert atd.players_xy.shape == (match.n_frames, 46, 2) and atd.players_speed.shape == (match.n_frames, 46)
        assert atd.players_xy.dtype == atd.players_speed.dtype == np.float64 and atd.n_slots == 46
        for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
            assert not getattr(atd, name).flags.writeable, name
        # From the arrays alone: at every row, the positions in the home slots are positions of home players on the
        # pitch then, and the same for the away ones (the made-up positions are all different).
        for row in range(match.n_frames):
            for team, slots in ((0, slice(0, 24)), (1, slice(24, 46))):
                shown = {tuple(position) for position in unsmoothed[row, slots] if not np.isnan(position).any()}
                own = {tuple(position) for position in match.players_xy[row][match.player_team == team]
                       if not np.isnan(position).any()}
                assert shown <= own, (seed, row, team)
    # No other slot count or team layout: the pools are the tracker's, whatever the match's squads.
    big = steady_match(teams=(0,) * 16 + (1,) * 15)
    atd, _ = degrade.degrade(degrade.tracking_view(big), "tracking", calibration, 0)
    assert atd.slot_team.tolist() == [0] * 24 + [1] * 22 and atd.players_xy.shape[1] == 46


def test_tracking_raises_when_a_pool_runs_out():
    """A pool with no free slot is an error that names the match, the half, the frame and the pool: never a guess."""
    parameters = degrade.tracking_parameters([3, 5], 0.2)
    exact = degrade.tracking_view(steady_match(teams=(0,) * 24 + (1,) * 22, substitutions=False))
    for seed in range(3):
        degrade.degraded_players(exact, parameters, seed)    # 24 and 22 on the pitch at once fit exactly
    too_many_home = degrade.tracking_view(steady_match(teams=(0,) * 25 + (1,) * 3, substitutions=False))
    message = expect_error(ValueError, lambda: degrade.degraded_players(too_many_home, parameters, 0),
                           "RND001", "half 1", "frame 0", "home", "24 slots", "players_xy")
    assert "player" in message
    too_many_away = degrade.tracking_view(steady_match("RND002", teams=(0,) * 3 + (1,) * 23, substitutions=False))
    expect_error(ValueError, lambda: degrade.degraded_players(too_many_away, parameters, 0), "RND002", "half 1",
                 "frame 0", "away", "22 slots")
    # Through the profile as well, and in the second half: 25 home players who are only on in the second half.
    late = steady_match(teams=(0,) * 25 + (1,) * 3, substitutions=False)
    xy = np.array(late.players_xy)
    xy[late.graded[0][1]:late.graded[0][2], 6:] = np.nan
    xy.setflags(write=False)
    expect_error(ValueError, lambda: degrade.degrade(degrade.tracking_view(dataclasses.replace(late, players_xy=xy)),
                                                     "tracking", made_up_calibration(), 0),
                 "RND001", "half 2", "frame {}".format(late.graded[1][1]), "home")


def test_tracking_degenerate_equals_gaps():
    """One endless tracklet per stretch of a player (d = 0, huge lengths): each player's positions are those of gaps.

    On the steady match: 7 stretches in the first half and 6 in the second, so 13 tracklets, each a whole stretch, and
    the slots are the lowest free ones, worked out by hand: the home players 0, 1 and 2 take slots 0, 1 and 2 (2 comes
    on at row 60 and finds slot 2 free), the away players 3, 4 and 5 take 24, 25 and 26, and player 5, who is off from
    100 to 130, gets 26 back, which nobody has taken. Then, whatever the mapping, every row has the same positions as
    gaps has, only moved to other slots: the same count per team and the same values."""
    match = steady_match()
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    endless = degrade.tracking_parameters([10 ** 9], 0.0)
    assert endless.mu == 0.0
    xy, speed, tracklets, _ = degrade.degraded_players(view, endless, 3)
    expected = {
        1: [(0, 0, 240, 0), (1, 0, 240, 1), (3, 0, 240, 24), (4, 0, 150, 25), (5, 0, 100, 26), (2, 60, 240, 2),
            (5, 130, 240, 26)],
        2: [(0, 265, 465, 0), (1, 265, 465, 1), (2, 265, 465, 2), (3, 265, 465, 24), (4, 265, 465, 25),
            (5, 265, 465, 26)]}
    assert {half: [(t.player, t.start, t.stop, t.slot) for t in records]
            for half, records in tracklets.items()} == expected
    gaps, _ = degrade.degrade(view, "gaps", calibration, 3)
    assert gaps.players_xy.shape[1] == 6
    for records in tracklets.values():
        for t in records:
            assert np.array_equal(xy[t.start:t.stop, t.slot], gaps.players_xy[t.start:t.stop, t.player]), t
            assert np.isfinite(gaps.players_xy[t.start:t.stop, t.player]).all(), "a tracklet is a whole stretch"
    # Each stretch is a tracklet, found one frame at a time here and not by the code under test.
    for half, start, stop in match.graded:
        stretches = sorted((player, start + first, start + last) for player in range(6)
                           for first, last in present_runs(list(gaps.players_present[start:stop, player])))
        assert sorted((t.player, t.start, t.stop) for t in tracklets[half]) == stretches
    # Whatever the slots: every row has gaps' positions, nobody lost and nobody added.
    for _, start, stop in match.graded:
        shown = np.sort(xy[start:stop, :, 0], axis=1)
        assert np.isnan(shown[:, 6:]).all()
        assert np.array_equal(shown[:, :6], np.sort(gaps.players_xy[start:stop, :, 0], axis=1), equal_nan=True)
    assert np.isnan(xy[240:265]).all(), "the rows between the halves have no players"
    # The whole profile at d = 0 and through the Calibration: nobody lost in any frame.
    never = types.SimpleNamespace(**{**vars(calibration), "present_counts": {
        half: np.full(len(mask), 22, dtype=np.int16) for half, mask in calibration.ball_masks.items()}})
    atd, info = degrade.degrade(view, "tracking", never, 3)
    assert info["players"]["d"] == 0.0 and info["players"]["lost_frames"] == 0
    assert np.array_equal(np.isfinite(atd.players_xy[:, :, 0]).sum(axis=1), gaps.players_present.sum(axis=1))
    assert ball_bytes(atd) == ball_bytes(gaps), "the ball is gaps' whatever the players do"


def test_tracking_draws_come_from_the_tracklets_stream_in_a_fixed_order():
    """The spans are what rng_for(seed, match_index, "tracklets", half) gives when it is read as documented.

    Players in column order, and for each one his stretches in time order: a tracked length (an index into the lengths),
    then, if the stretch goes on, a lost span (a geometric draw of mean mu), and so on. Written out here with plain
    loops and a generator made from the documented key, so that a change in the order of the draws shows. The other
    halves, seeds and components draw apart."""
    match = steady_match()
    view = degrade.tracking_view(match)
    parameters = degrade.calibrated_parameters(made_up_calibration(), 25, min_piece=0)     # the walk's own spans
    lengths = parameters.lengths.tolist()
    assert degrade.COMPONENTS["tracklets"] == 1
    for seed in (0, 4):
        for half, start, stop in match.graded:
            rng = np.random.default_rng([seed, view.match_index, 1, half])
            expected = []
            for player in range(6):
                for first, last in present_runs(list(~np.isnan(match.players_xy[start:stop, player, 0]))):
                    row = first
                    while True:
                        tracked = lengths[int(rng.integers(len(lengths)))]
                        expected.append((player, start + row, start + min(row + tracked, last)))
                        row += tracked
                        if row >= last:
                            break
                        row += int(rng.geometric(1 / parameters.mu))
                        if row >= last:
                            break
            got = [(t.player, t.start, t.stop) for t in degrade.choose_tracklets(view, half, parameters, seed)]
            assert sorted(got) == sorted(expected), (seed, half)
    spans = {seed: [(t.player, t.start, t.stop) for t in degrade.choose_tracklets(view, 1, parameters, seed)]
             for seed in (0, 1)}
    assert spans[0] != spans[1], "another seed draws other spans"
    expect_error(ValueError, lambda: degrade.choose_tracklets(view, 3, parameters, 0), "half 3", "graded")


def test_tracking_speeds_are_finite_differences_within_a_tracklet():
    """A slot's speed is the step from the frame before, in m/s on the pitch, within a tracklet; NaN on its first.

    Worked out with math.hypot, one frame at a time, from the match's own positions through each tracklet's records.
    The first frame of a tracklet has no speed, and the frame before it in the same slot, if the slot had a player then,
    is another player's: there is none, since a slot is taken only from the row after it was freed (version 3; in
    versions 1 and 2 the step from one player to another had no speed, and the position jumped). The DFL speed channel
    is not used."""
    match = steady_match(rows=(400, 300))
    view = degrade.tracking_view(match)
    parameters = degrade.tracking_parameters([3, 5, 8, 20], 0.3)
    length, width = match.pitch
    handed_over = checked = 0
    for seed in range(3):
        xy, speed, tracklets, _ = degrade.degraded_players(view, parameters, seed)
        expected = np.full(speed.shape, np.nan)
        firsts = []
        for records in tracklets.values():
            for t in records:
                firsts.append((t.start, t.slot))
                for row in range(t.start + 1, t.stop):
                    dx = (match.players_xy[row, t.player, 0] - match.players_xy[row - 1, t.player, 0]) * length
                    dy = (match.players_xy[row, t.player, 1] - match.players_xy[row - 1, t.player, 1]) * width
                    expected[row, t.slot] = math.hypot(dx, dy) * match.fps
                    checked += 1
        assert np.allclose(speed, expected, rtol=1e-12, atol=0, equal_nan=True), seed
        assert all(np.isnan(speed[row, slot]) for row, slot in firsts), "NaN on the first frame of every tracklet"
        handed_over += sum(np.isfinite(xy[row - 1, slot, 0]) for row, slot in firsts if row > 0)
        assert (np.isfinite(xy[1:, :, 0]) & np.isfinite(xy[:-1, :, 0])).sum() == sum(
            t.stop - t.start - 1 for records in tracklets.values() for t in records), "adjacent pairs are within tracklets"
        assert np.isnan(speed[~np.isfinite(xy[:, :, 0])]).all(), "no speed where nobody is"
    assert checked > 1000 and handed_over == 0, (checked, handed_over)
    # DFL's own speed channel is not used: with another one, the same speeds.
    silent = np.zeros_like(match.players_speed)
    silent.setflags(write=False)
    other = degrade.tracking_view(dataclasses.replace(match, players_speed=silent))
    assert degrade.degraded_players(other, parameters, 2)[1].tobytes() == speed.tobytes()
    # A one-frame tracklet has a first frame and nothing else.
    one = degrade.tracking_parameters([1], 0.5)
    _, speed, _, _ = degrade.degraded_players(view, one, 0)
    assert np.isnan(speed).all()


def owners_of(tracklets, n_frames, n_slots=46):
    """(n_frames, n_slots) int: the number of the tracklet that holds each slot on each row (every tracklet of every half
    has a number of its own), -1 where nobody does."""
    owner = np.full((n_frames, n_slots), -1, dtype=np.int64)
    number = 0
    for half in sorted(tracklets):
        for t in tracklets[half]:
            owner[t.start:t.stop, t.slot] = number
            number += 1
    return owner


def handovers_of(owner):
    """[(row, slot)] of every pair of adjacent rows (row - 1, row) on which a slot is present on both and held by two
    different tracklets: a slot that changes player between two frames, which ours never does (A07)."""
    both = (owner[1:] >= 0) & (owner[:-1] >= 0) & (owner[1:] != owner[:-1])
    rows, slots = np.nonzero(both)
    return sorted(zip((rows + 1).tolist(), slots.tolist()))


def test_a_slot_never_changes_player_between_adjacent_present_rows():
    """On two adjacent rows a slot that is present on both is held by one tracklet: it never changes player.

    Ours cannot: its jitter is 3.28 mm over 3.2 million triples (A07), whose squares add to 34 m2, and one jump of 12 m
    between two frames adds 288 m2 (two triples of 12 m), 8 times all of it. A slot is free for a tracklet that starts
    at t only from the row after the one on which the last tracklet in it stopped, so there is at least one empty row
    between two tracklets in a slot. Version 2 let the next tracklet take the slot on the very row it was freed (that
    row's position was another player's): 372 times in 7 matches and 3 seeds (17.7 a match and seed, 9 to 30), a jump of
    13.8 m at the median, and 96% of the squares of its 8.49 cm jitter per slot with the 5-frame average (R004c).

    Worked out from the records, on rows that are frames of the match: every pair of adjacent rows of every slot, over the
    steady match (6 players, the walk, the episodes, the cap and the rule, all on) for twelve seeds, and the same for the
    positions: on two adjacent rows where a slot is present the position is the same player's, so it differs by no more
    than a player moves in a frame (the made-up players here jump about the pitch, so the check is on who it is, not on
    how far). The tracklets that share a slot are counted, so that a rule that stopped sharing slots at all (and passed
    vacuously) would be seen: at least one slot is taken again after exactly one empty row, the closest the rule allows."""
    cal = made_up_calibration()
    shared = taken_again = 0
    for match in (steady_match(), steady_match("RND002", 1, rows=(260, 190))):
        view = degrade.tracking_view(match)
        parameters = degrade.calibrated_parameters(cal, view.fps)
        for seed in range(12):
            limits = degrade.cap_limits(view, cal, seed)
            xy, _, tracklets, _ = degrade.degraded_players(view, parameters, seed, limits)
            owner = owners_of(tracklets, match.n_frames)
            assert handovers_of(owner) == [], (seed, handovers_of(owner)[:5])
            for records in tracklets.values():
                last = {}
                for t in sorted(records, key=lambda t: (t.slot, t.start)):
                    if t.slot in last:
                        shared += 1
                        taken_again += t.start == last[t.slot] + 1
                    last[t.slot] = t.stop
            present = np.isfinite(xy[:, :, 0])
            adjacent = present[1:] & present[:-1]
            for row, slot in zip(*np.nonzero(adjacent)):
                a, b = int(owner[row, slot]), int(owner[row + 1, slot])
                assert a == b and a >= 0, (seed, row, slot)
    assert shared > 50 and taken_again > 0, (shared, taken_again)


def test_tracklet_records_hold_no_frame_in_their_repr():
    """Under the cap a tracklet's cut points are where our slot count crossed 18, in a window of ours whose offset every
    report states: frame-level data of BAR-ATH, which must not reach a log by an accident of printing a record.

    A Tracklet shows its player and slot and not its rows, and a HalfPlayers shows neither its tracklets nor its counts'
    detail: with distinctive rows (123,457 and 124,001) neither number is in the repr of one, of a list of them, of the
    HalfPlayers of a real run, or of the tuple that degraded_players returns for its records. The fields are still
    there and compare, so the checks that read them are unchanged."""
    t = degrade.Tracklet(player=3, start=123457, stop=124001, slot=7)
    assert "123457" not in repr(t) and "124001" not in repr(t) and "player=3" in repr(t) and "slot=7" in repr(t)
    assert (t.start, t.stop) == (123457, 124001) and t == degrade.Tracklet(player=3, start=123457, stop=124001, slot=7)
    assert t != degrade.Tracklet(player=3, start=123457, stop=124002, slot=7)
    assert "12345" not in repr([t, t]) and "12400" not in repr({1: [t]})
    match = steady_match()
    view = degrade.tracking_view(match)
    cal = made_up_calibration()
    parameters = degrade.calibrated_parameters(cal, view.fps)
    half = degrade.choose_half(view, 1, parameters, 0, degrade.cap_limits(view, cal, 0)[1])
    assert half.tracklets and "Tracklet" not in repr(half) and "start" not in repr(half), repr(half)[:200]
    records = degrade.degraded_players(view, parameters, 0)[2]
    rows = {str(row) for t in records[1] for row in (t.start, t.stop) if row > 99}
    assert rows and not any(number in repr(records) for number in rows), "players and slots have two digits, rows three"


def test_a_handover_on_the_row_a_slot_is_freed_is_what_the_check_would_catch():
    """The check above does see a slot that changes player: on hand-built records it lists exactly the handovers.

    Slot 3 holds the tracklet of rows 0-4 and then another of rows 5-9 (a handover on row 5), slot 4 one of rows 0-4 and
    one of rows 6-9 (an empty row between: no handover), and slot 5 one of rows 2-6 alone."""
    records = {1: [degrade.Tracklet(player=0, start=0, stop=5, slot=3), degrade.Tracklet(player=1, start=5, stop=10, slot=3),
                   degrade.Tracklet(player=2, start=0, stop=5, slot=4), degrade.Tracklet(player=3, start=6, stop=10, slot=4),
                   degrade.Tracklet(player=4, start=2, stop=7, slot=5)]}
    assert handovers_of(owners_of(records, 12)) == [(5, 3)]
    assert handovers_of(owners_of({1: records[1][2:]}, 12)) == []


def test_dfl_a_slot_never_changes_player_on_a_real_match():
    """The same check on a real DFL match, J03WMX, with the made-up tracking figures (the rule is not about ours):
    over four seeds, with the cap on, no slot changes player between two adjacent rows on which it is present, and the
    tracklets of a slot are separated by at least one empty row. Aggregates only."""
    from pipeline.benchmark import sources
    match_id = sources.MATCH_IDS[0]
    for kind in dfl.SOURCE_KINDS:
        if not os.path.isfile(sources.local_path(match_id, kind)):
            raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                match_id, kind, dfl.FETCH_ADVICE))
    cal = made_up_calibration()
    match = real_match(match_id)
    view = degrade.tracking_view(match)
    parameters = degrade.calibrated_parameters(cal, view.fps)
    pairs = 0
    for seed in range(4):
        _, _, tracklets, _ = degrade.degraded_players(view, parameters, seed, degrade.cap_limits(view, cal, seed))
        owner = owners_of(tracklets, match.n_frames)
        assert handovers_of(owner) == [], (seed, handovers_of(owner)[:5])
        pairs += int(((owner[1:] >= 0) & (owner[:-1] >= 0)).sum())
    print("      {}: no slot changed player on any of the {:,} adjacent present pairs of 4 seeds".format(match_id, pairs))
    assert pairs > 10 ** 6


def jitter_inside_tracklets(xy_m, tracklets):
    """(triples, RMS in metres) of the second difference p[t+1] - 2 p[t] + p[t-1] taken inside one tracklet: the three
    rows of a triple all belong to it. `tracklets` is {half: [Tracklet]} as degraded_players gives it, `xy_m` the slots'
    positions in metres, (n_frames, n_slots, 2)."""
    squares, triples = 0.0, 0
    for records in tracklets.values():
        for t in records:
            run = xy_m[t.start:t.stop, t.slot]
            if run.shape[0] >= 3:
                second = run[2:] - 2.0 * run[1:-1] + run[:-2]
                squares += float((second * second).sum())
                triples += second.shape[0]
    return triples, math.sqrt(squares / triples) if triples else math.nan


def test_jitter_inside_tracklets_counts_what_it_says():
    """One slot on a line holds a tracklet of rows 0-3 at 0, 1, 2, 3 and another of rows 4-6 at 10, 11, 12 (a handover
    on row 4, as version 2 made them): inside the tracklets the second differences are 0, 0 (rows 1 and 2 of the first)
    and 0 (row 5 of the second), 3 triples with an RMS of 0; per slot they are those three and the two that straddle
    the handover, 6 and -6, so 5 triples and sqrt((36 + 36) / 5) by second_difference_shape. A tracklet of two rows has
    no triple."""
    xy = np.zeros((8, 1, 2))
    xy[:7, 0, 0] = [0, 1, 2, 3, 10, 11, 12]
    records = {1: [degrade.Tracklet(player=0, start=0, stop=4, slot=0), degrade.Tracklet(player=1, start=4, stop=7, slot=0),
                   degrade.Tracklet(player=2, start=7, stop=8, slot=0)]}
    assert jitter_inside_tracklets(xy, records) == (3, 0.0)
    present = np.array([True] * 7 + [False])[:, None]
    count, rms, _, _, _ = second_difference_shape(xy, present, [(1, 0, 8)])
    assert count == 5 and abs(rms - math.sqrt(72 / 5)) < 1e-12
    assert jitter_inside_tracklets(xy, {1: [degrade.Tracklet(player=0, start=0, stop=2, slot=0)]})[0] == 0


def test_tracking_is_deterministic_and_independent_of_the_other_matches():
    """The same seed gives the same bytes and the same record; another seed, match_index or match gives another.

    A match gets the same degradation whichever matches are run with it and in whatever order, and the record of what
    was drawn is JSON-ready: ints and floats only, no frame of anybody."""
    matches = {"RND001": steady_match("RND001", 0), "RND002": steady_match("RND002", 1, rows=(260, 190)),
               "J03WMX": steady_match("J03WMX", 2)}
    calibration = made_up_calibration()

    def everything(atd):
        names = ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team")
        return b"".join(getattr(atd, name).tobytes() for name in names)

    def keyed(order):
        return {(seed, match_id): everything(degrade.degrade(degrade.tracking_view(matches[match_id]), "tracking",
                                                           calibration, seed)[0])
                for seed in (0, 1) for match_id in order}

    forward, backward = keyed(["RND001", "RND002", "J03WMX"]), keyed(["J03WMX", "RND002", "RND001"])
    assert forward == backward and len(set(forward.values())) == len(forward)
    assert keyed(["RND002"])[(1, "RND002")] == forward[(1, "RND002")], "alone, a match gets what it got in the run"

    view = degrade.tracking_view(matches["RND001"])
    first, info = degrade.degrade(view, "tracking", calibration, 0)
    again, info_again = degrade.degrade(degrade.tracking_view(matches["RND001"]), "tracking", made_up_calibration(), 0)
    assert everything(first) == everything(again) and info == info_again
    elsewhere, _ = degrade.degrade(degrade.tracking_view(matches["RND001"], match_index=3), "tracking", calibration, 0)
    assert everything(elsewhere) != everything(first) and elsewhere.players_xy.tobytes() != first.players_xy.tobytes()
    assert json.loads(json.dumps(info, sort_keys=True)) == json.loads(json.dumps(info_again, sort_keys=True))
    assert sorted(info["players"]) == sorted(PLAYERS_KEYS), sorted(info["players"])
    assert sorted(info["players"]["halves"]) == [1, 2]
    assert all(sorted(record) == sorted(HALF_KEYS) for record in info["players"]["halves"].values())
    for name in ("onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames", "speed_nan_runs",
                 "speed_nan_frames", "capped_frames", "cap_slot_frames", "cap_pieces_cut", "short_pieces",
                 "short_slot_frames", "short_episode_pieces"):
        assert info["players"][name] == sum(record[name] for record in info["players"]["halves"].values()), name
    assert info["players"]["tracked_scale"] == degrade.TRACKED_SCALE and info["players"]["min_piece_frames"] == 8
    assert info["players"]["onsets"] == info["players"]["episodes"] + info["players"]["skipped_far"] + \
        info["players"]["skipped_overlap"]
    assert info["players"]["tracklets"] == sum(record["tracklets"] for record in info["players"]["halves"].values())
    assert info["players"]["lost_frames"] == info["players"]["on_pitch_frames"] - int(
        np.isfinite(first.players_xy[:, :, 0]).sum())
    scaled = np.maximum(1, np.rint(np.array([10, 25, 50, 80, 100]) * degrade.TRACKED_SCALE)).mean()
    assert abs(info["players"]["mean_tracked_frames"] - scaled) < 1e-9, "the made-up lengths at the frozen scale"
    assert abs(info["players"]["mean_lost_frames"] - scaled * 3 / 52) < 1e-9, "mu = mean * d / (1 - d), d = 3 / 55"
    assert all(isinstance(value, (int, float)) for value in info["players"].values() if not isinstance(value, dict))
    # Many seeds, many different spans: the draws are not stuck.
    drawn = {degrade.degrade(view, "tracking", calibration, seed)[0].players_xy.tobytes() for seed in range(12)}
    assert len(drawn) == 12


def test_tracking_serves_inputs_without_the_truth():
    """A tracking Atd served as MatchInputs holds nothing of the match or its flags, to any depth (the deep walk)."""
    match, calibration = steady_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    atd, _ = degrade.degrade(view, "tracking", calibration, 0)
    inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"
    assert leaks(view, match) == []
    assert inputs.profile == "tracking" and inputs.atd is atd and inputs.atd.n_slots == 46
    assert inputs.atd_manifest["profile"] == "tracking"
    assert "array" not in repr(atd), "no frame in a repr"


# -- the wrong-team episodes, the NaN-speed runs and the smoothing -----------------------------------------------------


def test_wrong_team_starts_at_close_pairs():
    """An episode starts only where an opposing pair is within 2 m, inclusive, and the player is one of the pair.

    Six players stand far apart except in windows (pairs_match): in half 1, rows 100-400, home 1 and away 4 are exactly
    2.0 m apart (inside: 2 m is within) and in rows 600-900 they are 2.000122 m apart (beyond); in half 2, rows
    200-500, two pairs are exactly 1.0 m apart (a tie), (home 0, away 3) and (home 1, away 4), and the tie goes to the
    lowest columns. The onsets are read back from the documented generator, and each is classed by where it falls: an
    onset in a window where a pair is within 2 m is an episode or an overlap, every other one is skipped as far, and
    both are counted."""
    windows = [(1, 100, 400, [(1, 4, 1 / 64)]), (1, 600, 900, [(1, 4, 1 / 64 + 2 ** -20)]),
               (2, 200, 500, [(0, 3, 1 / 128), (1, 4, 1 / 128)])]
    match = pairs_match(windows)
    view = degrade.tracking_view(match)
    parameters = episode_parameters()
    rate = 1.0 * 0.45 / 15
    near = {1: (100, 400), 2: (200, 500)}
    seen, totals = {1: set(), 2: set()}, {"episodes": 0, "skipped_far": 0, "skipped_overlap": 0}
    for seed in range(8):
        _, _, records, counts = degrade.degraded_players(view, parameters, seed)
        for half, start, stop in match.graded:
            onsets = start + np.flatnonzero(np.random.default_rng([seed, view.match_index, 2, half]).random(
                stop - start) < rate)
            lo, hi = start + near[half][0], start + near[half][1]
            close = sum(1 for t in onsets if lo <= t < hi)
            record = counts[half]
            assert record["onsets"] == len(onsets) and record["skipped_far"] == len(onsets) - close, (seed, half)
            assert record["episodes"] + record["skipped_overlap"] == close, (seed, half, record)
            carried = [t for t in records[half] if is_carried(t, view)]
            assert len(carried) == record["episodes"] and sum(t.stop - t.start for t in carried) == \
                record["episode_frames"]
            assert all(lo <= t.start < hi for t in carried), "an episode starts where a pair is within 2 m"
            seen[half] |= {t.player for t in carried}
            for name in totals:
                totals[name] += record[name]
    assert seen[1] == {1, 4} and seen[2] == {0, 3}, "one of the pair, and the lowest columns of a tie: " + str(seen)
    assert all(count > 0 for count in totals.values()), totals

    # No pair within 2 m anywhere: every onset is skipped, and nothing is carried.
    far = degrade.tracking_view(pairs_match([]))
    _, _, records, counts = degrade.degraded_players(far, parameters, 0)
    assert counts[1]["onsets"] > 0 and all(c["episodes"] == c["skipped_overlap"] == 0 for c in counts.values())
    assert all(c["skipped_far"] == c["onsets"] for c in counts.values())
    assert not any(is_carried(t, far) for records_ in records.values() for t in records_)
    # A team with nobody tracked at the onset has no pair: counted as far too.
    alone = steady_match(teams=(0, 0, 0, 0, 0, 0), substitutions=False)
    _, _, _, counts = degrade.degraded_players(degrade.tracking_view(alone), parameters, 0)
    assert counts[1]["onsets"] > 0 and counts[1]["skipped_far"] == counts[1]["onsets"]


def test_wrong_team_moves_one_player_to_other_pool():
    """An episode piece is its own tracklet in a slot of the other team's pool, and its player leaves his own pool.

    Everybody is tracked all the time (nobody lost) and there is always a pair within 2 m (pairs_match), so every row
    has 3 home and 3 away players, and the slots must show 3 - (home players carried) + (away players carried) at home.
    Worked out from the records one row at a time, and read back from the arrays: the carried player's position is
    in the other pool's slot and in none of his own pool's, on every row of the episode, and his pieces tile the half
    (before, the episode, after) with a new tracklet after it."""
    windows = [(1, 0, 1000, [(1, 4, 1 / 128)]), (2, 0, 500, [(0, 3, 1 / 128)]), (2, 500, 1000, [(2, 5, 1 / 128)])]
    match = pairs_match(windows)
    view = degrade.tracking_view(match)
    parameters = episode_parameters()
    checked = 0
    for seed in range(4):
        xy, _, records, counts = degrade.degraded_players(view, parameters, seed)
        for half, start, stop in match.graded:
            carried = [t for t in records[half] if is_carried(t, view)]
            assert carried and len(carried) == counts[half]["episodes"], (seed, half)
            for row in range(start, stop):
                moved = [0, 0]                       # players carried at this row, by their own team
                for t in carried:
                    moved[int(view.player_team[t.player])] += t.start <= row < t.stop
                home = int(np.isfinite(xy[row, :degrade.HOME_SLOTS, 0]).sum())
                away = int(np.isfinite(xy[row, degrade.HOME_SLOTS:, 0]).sum())
                assert (home, away) == (3 - moved[0] + moved[1], 3 - moved[1] + moved[0]), (seed, row, moved)
            for t in carried:
                own, other = (slice(0, 24), slice(24, 46)) if view.player_team[t.player] == 0 else (slice(24, 46),
                                                                                                    slice(0, 24))
                position = match.players_xy[t.start:t.stop, t.player]
                assert np.array_equal(xy[t.start:t.stop, t.slot], position), "the piece is his own track"
                assert (t.slot >= degrade.HOME_SLOTS) == (view.player_team[t.player] == 0), "a slot of the other pool"
                for row in (t.start, t.stop - 1):
                    assert not (xy[row, own] == match.players_xy[row, t.player]).all(axis=-1).any(), "he left his pool"
                    assert (xy[row, other] == match.players_xy[row, t.player]).all(axis=-1).sum() == 1
                checked += 1
            for player in range(6):
                pieces = sorted((t.start, t.stop, t.slot) for t in records[half] if t.player == player)
                assert pieces[0][0] == start and pieces[-1][1] == stop
                assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:])), "before, the episode, after: no gap"
    assert checked > 50, checked


def test_wrong_team_draws_come_from_the_wrong_team_stream_in_a_fixed_order():
    """The pieces and the counts are what rng_for(seed, match_index, "wrong_team", half) gives, read as documented.

    The onset rows are one uniform per row of the half's graded block (a row is an onset when its uniform is below the
    rate s * target / mean episode frames), then one pick per onset (0 is the home player of the closest pair, 1 the
    away player) and one duration per onset, each as one call, so that what an onset draws never depends on what the
    others do. Written out here with plain loops and a generator made from the documented key, including the skipped
    onsets, on tracklets that the walk has cut into several spans (d = 0.2): the pieces, their pools and the counts
    must be equal."""
    windows = [(1, 50, 900, [(1, 4, 1 / 64)]), (2, 0, 400, [(0, 3, 1 / 128)]), (2, 400, 1000, [(2, 5, 3 / 256)])]
    match = pairs_match(windows)
    view = degrade.tracking_view(match)
    parameters = episode_parameters(lengths=(30, 60, 200), d=0.2, episodes=(5, 10, 40, 300), share=0.9)
    totals = {"episodes": 0, "skipped_far": 0, "skipped_overlap": 0}
    for seed in range(6):
        for half, _, _ in match.graded:
            spans = degrade.walk_spans(view, half, parameters, seed)
            expected, expected_counts = reference_episodes(view, half, spans, parameters, seed)
            got = degrade.choose_half(view, half, parameters, seed)
            pieces = sorted((t.start, t.player, t.stop, int(t.slot >= degrade.HOME_SLOTS)) for t in got.tracklets)
            assert pieces == expected, (seed, half)
            assert {name: got.counts[name] for name in expected_counts} == expected_counts, (
                seed, half, got.counts, expected_counts)
            assert all(got.counts[name] == 0 for name in ("capped_frames", "cap_slot_frames", "cap_pieces_cut")), \
                "no limit given: no cap"
            for name in totals:
                totals[name] += expected_counts[name]
    assert all(count > 0 for count in totals.values()), "every kind of onset happens: " + str(totals)
    # The same key and nothing else decides: another seed or half draws other onsets.
    first, second = ([(t.player, t.start, t.stop) for t in degrade.choose_half(view, 1, parameters, seed).tracklets]
                     for seed in (0, 1))
    assert first != second


def test_wrong_team_onsets_are_a_bernoulli_process_at_the_documented_rate():
    """Over many rows the onsets are binomial with r = s * target / mean episode frames per row of the graded block.

    A steady match of 2 x 20,000 rows and the made-up Calibration's episodes (10, 30, 50 and 100 frames, mean 47.5) and
    share (0.1): r = s * 0.1 / 47.5. Counted over 6 seeds with the scale at 1 and at 3, within 5 standard errors of
    n * r (the binomial's own, sqrt(n r (1 - r))); the share of the onsets that are far is high on random positions,
    which are nowhere near a close pair, and every onset is an episode, far or an overlap."""
    match = steady_match(rows=(20000, 20000))
    view = degrade.tracking_view(match)
    calibration = made_up_calibration()
    for scale in (1.0, 3.0):
        parameters = dataclasses.replace(degrade.calibrated_parameters(calibration, 25), scale=scale)
        rate = scale * MADE_UP_WRONG_TEAM / 47.5
        assert abs(parameters.onset_rate - rate) < 1e-15
        onsets = rows = 0
        for seed in range(6):
            _, _, _, counts = degrade.degraded_players(view, parameters, seed)
            for record in counts.values():
                assert record["onsets"] == record["episodes"] + record["skipped_far"] + record["skipped_overlap"]
                onsets += record["onsets"]
            rows += sum(stop - start for _, start, stop in match.graded)
        expected, error = rows * rate, math.sqrt(rows * rate * (1 - rate))
        assert abs(onsets - expected) < 5 * error, (scale, onsets, expected, error)
    none = dataclasses.replace(parameters, scale=0.0)
    assert none.onset_rate == 0.0
    _, _, _, counts = degrade.degraded_players(view, none, 0)
    assert all(record["onsets"] == 0 for record in counts.values())


def reference_players(view, parameters, seed):
    """The walk's players, written out: (xy, speed, [(half, player, start, stop, slot)]), with no episodes, NaN runs
    or smoothing.

    The spans are the walk's (degrade.walk_run, drawn from the "tracklets" stream as documented), the slots the lowest
    free one of the player's own pool for a tracklet taken in order of (start, player) (free: its last tracklet stopped
    before this one starts, so that a row separates them; version 3), a speed the step from the frame before in m/s, NaN
    on a tracklet's first row."""
    pitch = np.asarray(view.pitch)
    xy = np.full((view.n_frames, 46, 2), np.nan)
    speed = np.full((view.n_frames, 46), np.nan)
    records = []
    for half, start, stop in view.graded:
        rng = degrade.rng_for(seed, view.match_index, "tracklets", half)
        spans = []
        for player in range(view.players_xy.shape[1]):
            column = [not math.isnan(view.players_xy[row, player, 0]) for row in range(start, stop)]
            for first, last in present_runs(column):
                for a, b in degrade.walk_run(rng, last - first, parameters):
                    spans.append((start + first + a, player, start + first + b))
        free = [-1] * 46
        for first, player, last in sorted(spans):
            pool = range(24) if view.player_team[player] == 0 else range(24, 46)
            slot = next(candidate for candidate in pool if free[candidate] < first)
            free[slot] = last
            records.append((half, player, first, last, slot))
            xy[first:last, slot] = view.players_xy[first:last, player]
            if last - first > 1:
                step = np.diff(view.players_xy[first:last, player] * pitch, axis=0)
                speed[first + 1:last, slot] = np.hypot(step[:, 0], step[:, 1]) * view.fps
    return xy, speed, records


def test_tracking_without_the_new_parts_is_the_walk_in_the_lowest_free_slots():
    """With no episodes (s = 0), no NaN runs (a target of 0) and no smoothing, the players are exactly the walk's,
    each tracklet in the lowest free slot of its pool (version 3's rule).

    Compared with the reference above, byte for byte: the positions, the speeds and the records of every tracklet,
    on two steady matches and three seeds, under two sets of parameters. The new parts draw only from their own
    generators, so with them on the walk (the spans) is the same too, and so is the ball (test_profiles_nest)."""
    cases = ((steady_match(rows=(400, 300)), degrade.tracking_parameters([3, 5, 8, 20], 0.3)),
             (steady_match(), degrade.calibrated_parameters(made_up_calibration(), 25, tracked_scale=1.0,
                                                           min_piece=0)))
    for match, parameters in cases:
        view = degrade.tracking_view(match)
        plain = dataclasses.replace(parameters, scale=0.0, speed_nan_share=0.0, smooth=0)
        for seed in range(3):
            xy, speed, records, counts = degrade.degraded_players(view, plain, seed)
            ref_xy, ref_speed, ref_records = reference_players(view, parameters, seed)
            assert xy.tobytes() == ref_xy.tobytes() and speed.tobytes() == ref_speed.tobytes(), seed
            got = [(half, t.player, t.start, t.stop, t.slot) for half, tracklets in records.items() for t in tracklets]
            assert got == ref_records, seed
            assert all(record["onsets"] == record["episodes"] == record["speed_nan_runs"] == 0
                       for record in counts.values())
            for half, _, _ in view.graded:
                assert degrade.walk_spans(view, half, plain, seed) == degrade.walk_spans(view, half, parameters, seed)


def test_profiles_nest():
    """For the same seed the tracking profile's ball is the gaps profile's, to the byte, with the new parts on.

    The ball is drawn from the "transplant" stream and the players from others, so what tracking adds is the players
    alone, and the difference between two results is one component: the same missing-ball mask, the same positions, the
    same NaN speed (T31), the same record of the transplants, and the record of the draws is gaps' with the players'
    figures beside it. The players are what differs: 46 slots where gaps has one per DFL player."""
    match, calibration = steady_match(), made_up_calibration()
    view = degrade.tracking_view(match)
    for seed in (0, 1, 2):
        tracking, tracking_info = degrade.degrade(view, "tracking", calibration, seed)
        gaps, gaps_info = degrade.degrade(view, "gaps", calibration, seed)
        assert tracking.ball_present.tolist() == gaps.ball_present.tolist(), "the same mask"
        assert ball_bytes(tracking) == ball_bytes(gaps) and np.isnan(tracking.ball_speed).all()
        assert tracking_info["transplants"] == gaps_info["transplants"]
        assert tracking_info["mask_share"] == gaps_info["mask_share"]
        assert sorted(tracking_info) == sorted(list(gaps_info) + ["players"]), sorted(tracking_info)
        assert (tracking_info["profile"], tracking_info["version"]) == ("tracking", 3)
        assert tracking.players_xy.shape[1] == 46 and gaps.players_xy.shape[1] == 6, "the players are what differs"
        players = tracking_info["players"]
        assert players["scale"] == degrade.WRONG_TEAM_SCALE > 0 and players["speed_nan_target"] == MADE_UP_SPEED_NAN
        assert players["speed_nan_runs"] > 0, "the new parts are on, so this is not vacuous"
        assert degrade.PROFILES["tracking"] == 3 and degrade.PROFILES["gaps"] == 1, "version 3 is frozen"


def test_speed_nan_share():
    """The share of the present slot-frames with a NaN speed lands on the target, to within 0.005 and in each half.

    A steady match of 2 x 20,000 rows, 6 players, a third of them lost now and then, runs of 3, 8, 21 and 40 frames and
    a target of 0.08: the share is counted by calibration.player_stats (the statistic the fidelity gate uses) on the
    graded blocks, and its numerator counts the tracklets' first rows too, which are NaN to begin with. A run goes on
    until the share reaches the target, so the last one overshoots by at most its own length: 40 frames of about 100,000
    present slot-frames a half."""
    match = steady_match(rows=(20000, 20000))
    view = degrade.tracking_view(match)
    parameters = degrade.tracking_parameters([200, 500, 900], 0.1, nan_runs=list(NAN_RUNS), speed_nan_share=0.08)
    length, width = match.pitch
    for seed in range(3):
        atd_xy, speed, records, counts = degrade.degraded_players(view, parameters, seed)
        present = np.isfinite(atd_xy[:, :, 0])
        stats = calibration.player_stats(present, degrade.slot_team(), atd_xy * np.array([length, width]), speed,
                                         match.graded, match.fps)
        assert abs(stats["speed_nan_share"] - 0.08) < 0.005, (seed, stats["speed_nan_share"])
        for half, start, stop in match.graded:
            here = present[start:stop]
            share = np.isnan(speed[start:stop])[here].mean()
            assert 0.08 <= share < 0.08 + max(NAN_RUNS) / here.sum(), (seed, half, share)
            assert counts[half]["speed_nan_runs"] > 100 and counts[half]["speed_nan_frames"] == int(
                np.isnan(speed[start:stop])[here].sum())
        assert np.isnan(speed[~present]).all(), "no speed where nobody is"
        # Nothing but the speed: the positions are the walk's, whether or not there are runs.
        plain, _, _, _ = degrade.degraded_players(view, dataclasses.replace(parameters, speed_nan_share=0.0), seed)
        assert plain.tobytes() == atd_xy.tobytes()
    # A target of 0 places no run: the NaN speeds are the tracklets' first rows and nothing else.
    xy, speed, records, counts = degrade.degraded_players(view, dataclasses.replace(parameters, speed_nan_share=0.0), 0)
    assert all(record["speed_nan_runs"] == 0 for record in counts.values())
    firsts = sum(len(tracklets) for tracklets in records.values())
    assert int(np.isnan(speed[np.isfinite(xy[:, :, 0])]).sum()) == firsts


def reference_speed_nan(view, half, xy, speed, tracklets, parameters, seed):
    """`speed` of a half with the NaN-speed runs placed as degrade.py documents them, one frame at a time.

    The present slot-frames of the half's graded block are listed in time order and, within a row, by slot; each run
    takes one draw for the start (an index into that list) and one for its length (an index into the run lengths),
    both from the half's "speed_nan" stream, marks the next L frames of that slot NaN, stops at the end of its
    tracklet, and the loop ends when the NaN speeds of the present slot-frames reach the target."""
    start, stop = next((first, last) for number, first, last in view.graded if number == half)
    rng = degrade.rng_for(seed, view.match_index, "speed_nan", half)
    present = [(row, slot) for row in range(start, stop) for slot in range(xy.shape[1])
               if not math.isnan(xy[row, slot, 0])]
    marked = {cell for cell in present if math.isnan(speed[cell])}
    out = np.array(speed)
    while len(marked) < parameters.speed_nan_share * len(present):
        row, slot = present[int(rng.integers(len(present)))]
        length = int(parameters.nan_runs[int(rng.integers(len(parameters.nan_runs)))])
        piece = next(t for t in tracklets if t.slot == slot and t.start <= row < t.stop)
        for later in range(row, min(row + length, piece.stop)):
            out[later, slot] = np.nan
            marked.add((later, slot))
    return out


def test_speed_nan_runs_follow_the_documented_draws():
    """The NaN speeds are what the "speed_nan" stream gives, read as documented, and a run never leaves its tracklet.

    The runs' starts are uniform over the present slot-frames of the graded block and their lengths come from the
    Calibration's runs (the loop of reference_speed_nan: plain Python, a generator made from the documented key);
    the speeds with runs must be equal to the speeds without them with the reference's runs added, frame for frame. A
    tracklet's NaN speeds are its first row and the runs that started inside it: the frame after a tracklet's last
    is another tracklet's own, so a run stopped there."""
    match = steady_match(rows=(400, 300))
    view = degrade.tracking_view(match)
    base = degrade.tracking_parameters([3, 5, 8, 20], 0.3, nan_runs=list(NAN_RUNS), speed_nan_share=0.25)
    plain = dataclasses.replace(base, speed_nan_share=0.0)
    runs = 0
    for seed in range(3):
        xy, speed, records, counts = degrade.degraded_players(view, base, seed)
        bare_xy, bare_speed, bare_records, _ = degrade.degraded_players(view, plain, seed)
        assert xy.tobytes() == bare_xy.tobytes() and records == bare_records
        for half, _, _ in match.graded:
            expected = reference_speed_nan(view, half, bare_xy, bare_speed, bare_records[half], base, seed)
            start, stop = next((first, last) for number, first, last in match.graded if number == half)
            assert np.array_equal(speed[start:stop], expected[start:stop], equal_nan=True), (seed, half)
            runs += counts[half]["speed_nan_runs"]
    assert runs > 20, runs


def test_smoothing_is_a_centred_moving_average_inside_each_piece():
    """A 5-frame centred moving average whose window shrinks at the ends to the rows inside; nothing crosses a piece.

    Worked out with loops on a few rows, then on every tracklet of a steady match under d = 0.3: the positions of a
    piece are the plain average of that piece's own rows (a window never reaches into another piece of the same
    player, or into the lost span between them), the lengths are unchanged, the speeds are the steps of the smoothed
    positions, and on white noise the second difference falls by the factor that the filter has on white noise,
    sqrt(4 / 25 / 6) = 0.163."""
    run = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 2.0], [9.0, 6.0], [16.0, 5.0], [25.0, 9.0], [36.0, 7.0]])
    out = degrade.smoothed(run, 5)
    assert out.shape == run.shape
    assert np.allclose(out, smoothed_plain(run.tolist(), 5), rtol=1e-12, atol=1e-12)
    assert np.allclose(out[0], run[:3].mean(axis=0)) and np.allclose(out[1], run[:4].mean(axis=0))
    assert np.allclose(out[3], run[1:6].mean(axis=0)) and np.allclose(out[6], run[4:].mean(axis=0))
    assert np.array_equal(degrade.smoothed(run[:1], 5), run[:1]), "one row has no neighbour"
    assert np.allclose(degrade.smoothed(run[:2], 5), run[:2].mean(axis=0)), "two rows: each is their mean"
    assert np.array_equal(degrade.smoothed(run, 1), run) and degrade.smoothed(run[:0], 5).shape == (0, 2)

    match = steady_match(rows=(400, 300))
    view = degrade.tracking_view(match)
    parameters = degrade.tracking_parameters([3, 5, 8, 20, 90], 0.3, smooth=5)
    length, width = match.pitch
    pieces = 0
    for seed in range(3):
        xy, speed, records, _ = degrade.degraded_players(view, parameters, seed)
        for half_records in records.values():
            for t in half_records:
                source = match.players_xy[t.start:t.stop, t.player].tolist()
                expected = smoothed_plain(source, 5)
                assert xy[t.start:t.stop, t.slot].shape == expected.shape == (t.stop - t.start, 2)
                assert np.allclose(xy[t.start:t.stop, t.slot], expected, rtol=1e-12, atol=1e-12), (seed, t)
                for row in range(t.start + 1, t.stop):
                    dx = (xy[row, t.slot, 0] - xy[row - 1, t.slot, 0]) * length
                    dy = (xy[row, t.slot, 1] - xy[row - 1, t.slot, 1]) * width
                    assert abs(speed[row, t.slot] - math.hypot(dx, dy) * match.fps) < 1e-9
                assert math.isnan(speed[t.start, t.slot])
                pieces += 1
    assert pieces > 100, pieces
    # White noise: the second difference of a 5-frame moving average is 0.163 of the raw one.
    noise = np.random.default_rng(3).normal(size=(60000, 2))
    second = lambda positions: float(np.sqrt(np.mean(np.sum((positions[2:] - 2 * positions[1:-1] + positions[:-2]) ** 2,
                                                            axis=1))))
    ratio = second(degrade.smoothed(noise, 5)[2:-2]) / second(noise[2:-2])
    assert abs(ratio - math.sqrt(4 / 25 / 6)) < 0.005, ratio
    expect_error(ValueError, lambda: degrade.smoothed(run, 4), "window", "odd")
    expect_error(ValueError, lambda: degrade.smoothed(run, 0), "window", "odd")


def test_tracking_parameters_hold_the_new_parts():
    """Episodes in frames from seconds at the rate, their mean, the NaN runs, the targets, the scale and the window.

    The made-up Calibration has episodes of 0.4, 1.2, 2 and 4 s: 10, 30, 50 and 100 frames at 25 Hz (mean 47.5) and 20,
    60, 100 and 200 at 50 Hz, each at least 1 frame; its NaN runs (frames) are not converted, its shares are 0.1 and
    0.08, and the scale and the window are the module's constants. The onset rate is s * target / the mean episode."""
    calibration = made_up_calibration()
    parameters = degrade.calibrated_parameters(calibration, 25)
    assert parameters.episodes.tolist() == [10, 30, 50, 100] and parameters.episodes.dtype == np.int64
    assert parameters.mean_episode == 47.5 and parameters.nan_runs.tolist() == list(NAN_RUNS)
    assert (parameters.wrong_team_share, parameters.speed_nan_share) == (MADE_UP_WRONG_TEAM, MADE_UP_SPEED_NAN)
    assert parameters.scale == degrade.WRONG_TEAM_SCALE and parameters.smooth == degrade.SMOOTH_WINDOW
    assert abs(parameters.onset_rate - degrade.WRONG_TEAM_SCALE * 0.1 / 47.5) < 1e-15
    assert not parameters.episodes.flags.writeable and not parameters.nan_runs.flags.writeable
    other = degrade.calibrated_parameters(calibration, 50)
    assert other.episodes.tolist() == [20, 60, 100, 200] and other.nan_runs.tolist() == list(NAN_RUNS)
    short = types.SimpleNamespace(**{**vars(calibration), "wrong_team_episodes_s": np.array([0.0, 0.01, 0.07])})
    assert degrade.calibrated_parameters(short, 25).episodes.tolist() == [1, 1, 2]
    assert degrade.calibrated_parameters(short, 25).mean_episode == 4 / 3, "the mean of the lengths drawn, in frames"
    assert degrade.WRONG_TEAM_SCALE > 0 and degrade.SMOOTH_WINDOW in (0, 3, 5, 7, 9) and degrade.NEAR_PAIR_M == 2.0
    # Nothing to draw: no share, no episodes drawn, no runs; and the neutral parameters have none of the new parts.
    nothing = degrade.calibrated_parameters(without_new_parts(calibration), 25)
    assert nothing.onset_rate == 0.0 and nothing.speed_nan_share == 0.0
    bare = degrade.tracking_parameters([30, 50], 0.1)
    assert (bare.onset_rate, bare.scale, bare.speed_nan_share, bare.smooth) == (0.0, 0.0, 0.0, 0)
    # Refusals say what to change.
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, wrong_team_share=1.0, scale=1.0,
                                                                  episodes=[5]), "wrong_team_share", "below 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, scale=-1.0), "scale", "at least 0")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, wrong_team_share=0.5, scale=1.0),
                 "episodes", "wrong_team_share")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, episodes=[5, 0]), "episodes", "at least 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, episodes=[2], wrong_team_share=0.9,
                                                                  scale=5.0), "onset rate", "below 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, speed_nan_share=0.1), "nan_runs",
                 "speed_nan_share")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, nan_runs=[3, 0], speed_nan_share=0.1),
                 "nan_runs", "at least 1")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, nan_runs=[3], speed_nan_share=0.6),
                 "speed_nan_share", "below")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, smooth=4), "smooth", "odd")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, smooth=-5), "smooth")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, scale=True), "scale")
    expect_error(ValueError, lambda: degrade.calibrated_parameters(types.SimpleNamespace(
        **{**vars(calibration), "wrong_team_episodes_s": np.zeros(0)}), 25), "wrong_team_episodes_s")


def test_dfl_tracking_never_exhausts_a_pool_on_the_real_matches():
    """No pool of slots runs out on any of the 7 downloaded matches, under several seeds and losses: they fit.

    The slots are assigned from the players' stretches on the pitch (where DFL gives a position), with parameters that
    make a slot change hands as often as it can (tracked spans of 1 to 250 frames, 30% of the time lost) and with the
    ones that lose nobody, and with wrong-team episodes at a rate far above ours (a team's pool then carries players of
    the other team as well as its own). The most players of one team on the pitch at once on any frame of the graded
    halves, and the most slots a pool uses, are printed per match (aggregates of public data); a pool has 24 and 22."""
    from pipeline.benchmark import sources
    stressing = degrade.tracking_parameters([1, 2, 5, 25, 250], 0.3)
    nobody = degrade.tracking_parameters([10 ** 9], 0.0)
    carrying = degrade.tracking_parameters([1, 2, 5, 25, 250], 0.3, episodes=[1, 5, 50, 500], wrong_team_share=0.5,
                                           scale=1.0)
    print("      {:<8} {:>10} {:>16} {:>14}".format("match", "tracklets", "most on, h / a", "slots, h / a"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        view = degrade.tracking_view(match)
        present = ~np.isnan(match.players_xy[:, :, 0])
        most = [int(present[:, match.player_team == team].sum(axis=1).max()) for team in (0, 1)]
        assert most[0] <= degrade.HOME_SLOTS and most[1] <= degrade.AWAY_SLOTS, (match_id, most)
        tracklets, slots = 0, [0, 0]
        for seed, parameters in ((0, stressing), (1, stressing), (2, stressing), (0, nobody), (0, carrying),
                                 (1, carrying)):
            for half, _, _ in view.graded:
                records = degrade.choose_tracklets(view, half, parameters, seed)     # raises if a pool runs out
                if parameters is stressing and seed == 0:
                    tracklets += len(records)
                for team, first in ((0, 0), (1, degrade.HOME_SLOTS)):
                    used = [t.slot - first + 1 for t in records if (t.slot >= degrade.HOME_SLOTS) == (team == 1)]
                    slots[team] = max([slots[team]] + used)
        print("      {:<8} {:>10,} {:>16} {:>14}".format(match_id, tracklets, "{} / {}".format(*most),
                                                         "{} / {}".format(*slots)))
        del match, view, present


def test_data_tracking_profile_on_a_real_match_with_our_real_calibration():
    """Our own Calibration on a downloaded DFL match: what the tracking profile makes, for information.

    Needs both our match (the Calibration is measured from it, or read from its cache) and a downloaded DFL match, so it
    runs with --data and says NOT RUN where the DFL files are not there. It prints d, the mean tracked span and the mean
    lost span of ours, and the statistics that calibration.player_stats gives on the degraded match beside the
    Calibration's: the histogram of slots present, the tracklets' median, the wrong-team share and the NaN-speed share.
    Nothing of that table is asserted: the gates of the profile are fidelity.py's (R004b), and the table is there to say
    by how much one match and one seed are off. What is asserted is the layout, that the ball is gaps', that the draws
    are the same twice, that nothing of the truth is reachable and that no pool runs out; that the cap reads the very
    window of our counts that the ball's mask was read from (the Transplant that the profile records), and that on every
    row the slots present are min(count, tracked) where our count is at most CAP and the tracked ones otherwise (the
    tracked ones being the players of the same draws without the cap, and with the shortest-piece rule off, which takes
    more players away, and the profile's own players are these and no more); and that the lost share of the walk, the
    episodes and the rule (no cap), pooled over the 7 matches and 5 seeds, is d to within 0.003 (2.71% against d = 2.50%
    with the scale of the tracked spans and the rule, 2.59% before them: a DFL half is 68,000 to 78,000 frames, where a
    cut at the end of a run is not negligible, and a longer span makes it weigh more). A mu without the (1 - d) would
    give about 2.53%, which no tolerance here can separate from 2.59%: the synthetic test_lost_share_on_long_synthetic,
    whose d = 0.3 makes it 23.1% against 30%, is what catches that. It also prints how many slots each team uses, in the
    degraded match and in ours (slot_use: the slots of the pool, those ever present, and those present on at least a
    tenth and at least a hundredth of the frames: the figures that R004 and the README quote), the cap's share of the
    frames and drops, how many tracklets are shorter than 8 frames (ours has none, A09) and the slot persistence of
    both: the idle gaps of a slot and the returns to the same slot, with the cap and, for comparison, without it.
    Aggregates only.
    """
    from pipeline.benchmark import sources
    match_id = sources.MATCH_IDS[0]
    for kind in dfl.SOURCE_KINDS:
        if not os.path.isfile(sources.local_path(match_id, kind)):
            raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                match_id, kind, dfl.FETCH_ADVICE))
    cal = calibration.load()
    match = real_match(match_id)
    view = degrade.tracking_view(match)
    parameters = degrade.calibrated_parameters(cal, view.fps)
    assert parameters.d == degrade.lost_share(cal.present_counts) and 0 < parameters.d < 0.2
    assert parameters.mu >= 1 and parameters.lengths.size == cal.tracklet_lengths_s.size
    atd, info = degrade.degrade(view, "tracking", cal, 0)
    again, info_again = degrade.degrade(view, "tracking", cal, 0)
    assert info == info_again and atd.players_xy.tobytes() == again.players_xy.tobytes()
    assert atd.players_speed.tobytes() == again.players_speed.tobytes()
    gaps, _ = degrade.degrade(view, "gaps", cal, 0)
    assert ball_bytes(atd) == ball_bytes(gaps)
    assert atd.players_xy.shape == (match.n_frames, 46, 2) and atd.slot_team.tolist() == [0] * 24 + [1] * 22
    inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=0)
    assert leaks(inputs, match) == [], "nothing of the match or its flags is reachable from the served inputs"

    # The cap: our counts are read from the very window that the ball's mask was, and every row has min(count, tracked).
    limits = degrade.cap_limits(view, cal, 0)
    alone = dataclasses.replace(parameters, min_piece=0)         # the cap by itself: the shortest-piece rule takes more
    tracked = np.isfinite(degrade.degraded_players(view, alone, 0)[0][:, :, 0]).sum(axis=1)
    shown = np.isfinite(degrade.degraded_players(view, alone, 0, limits)[0][:, :, 0]).sum(axis=1)
    assert degrade.degraded_players(view, parameters, 0, limits)[0].tobytes() == atd.players_xy.tobytes()
    assert (atd.players_present.sum(axis=1) <= shown).all(), "the shortest-piece rule only takes players away"
    low = frames = 0
    for half, _, stop in view.graded:
        record, kickoff = info["transplants"][half], view.kickoffs[half]
        window = cal.present_counts[record["source_half"]][record["offset"]:record["offset"] + stop - kickoff]
        assert window.size == stop - kickoff and np.array_equal(limits[half], window), "the ball's Transplant"
        want = np.where(limits[half] <= degrade.CAP, np.minimum(tracked[kickoff:stop], limits[half]),
                        tracked[kickoff:stop])
        assert np.array_equal(shown[kickoff:stop], want), half
        low, frames = low + int((limits[half] <= degrade.CAP).sum()), frames + limits[half].size
    assert info["players"]["capped_frames"] == low > 0 and info["players"]["cap_slot_frames"] > 0

    spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
    mine = calibration.player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy),
                                    atd.players_speed, spans, inputs.fps)

    def share(hist, low, high):
        return hist[low:high + 1].sum() / hist.sum()

    ours = cal.present_hist
    rows = (("slots present, mean", mine["present_mean"], cal.scalars["present_mean"], "{:.2f}"),
            ("23 or more present", share(mine["present_hist"], 23, 46), share(ours, 23, 46), "{:.1%}"),
            ("all 22 present", share(mine["present_hist"], 22, 22), share(ours, 22, 22), "{:.1%}"),
            ("19 to 21 present", share(mine["present_hist"], 19, 21), share(ours, 19, 21), "{:.1%}"),
            ("15 to 18 present", share(mine["present_hist"], 15, 18), share(ours, 15, 18), "{:.1%}"),
            ("14 or fewer present", share(mine["present_hist"], 0, 14), share(ours, 0, 14), "{:.1%}"),
            ("tracklets", mine["tracklets"], cal.scalars["tracklets"], "{:,}"),
            ("tracklet median, s", float(np.median(mine["tracklet_lengths_s"])), float(np.median(
                cal.tracklet_lengths_s)), "{:.1f}"),
            ("tracklets under 8 frames", float((np.rint(mine["tracklet_lengths_s"] * inputs.fps) < 8).mean()),
             float((np.rint(cal.tracklet_lengths_s * inputs.fps) < 8).mean()), "{:.2%}"),
            ("speeds NaN of the present", mine["speed_nan_share"], cal.speed_nan_share, "{:.2%}"),
            ("wrong-team frames", mine["wrong_team_share"], cal.wrong_team_share, "{:.1%}"))
    counts = np.concatenate([cal.present_counts[half] for half in sorted(cal.present_counts)])
    print("      {}: d {:.4f} (ours has more than {} slots present on {:.1%} of its frames), mean tracked span {:,.0f} "
          "frames ({:.1f} s), mean lost span {:.1f} frames ({:.2f} s)".format(
              match_id, parameters.d, degrade.CAP, (counts > degrade.CAP).mean(), parameters.mean_tracked,
              parameters.mean_tracked / view.fps, parameters.mu, parameters.mu / view.fps))
    print("      lost {:.2%} of the {:,} on-pitch frames, {:,} tracklets".format(
        info["players"]["lost_share"], info["players"]["on_pitch_frames"], info["players"]["tracklets"]))
    print("      the cap: {:,} of the {:,} frames from the kick-offs have our count at or below {} ({:.1%}), {:,} "
          "slot-frames dropped in {:,} pieces cut".format(
              low, frames, degrade.CAP, low / frames, info["players"]["cap_slot_frames"],
              info["players"]["cap_pieces_cut"]))
    print("      {:<26} {:>10} {:>10}".format("information, not gated", "degraded", "ours"))
    for name, degraded, measured, template in rows:
        print("      {:<26} {:>10} {:>10}".format(name, template.format(degraded), template.format(measured)))

    # Slot persistence (A09): how long a slot sits idle between two tracklets, and whether a player who comes back after
    # a lost span lands in the slot he had. The same function measures both sides, on each half from its kick-off to
    # the end of its graded block; ours has no identities, so it has no returns.
    from pipeline.inputs import Inputs
    ours_inputs = Inputs()
    ours_spans = [(half, ours_inputs.kickoffs[half], stop) for half, _, stop in ours_inputs.graded]
    use = {"degraded": slot_use(atd.players_present, atd.slot_team, spans),
           "ours": slot_use(ours_inputs.atd.players_present, ours_inputs.atd.slot_team, ours_spans)}
    print("      {:<28} {:>10} {:>10}".format("slots per team, home / away", "degraded", "ours"))
    for name, index in (("in the pool", 0), ("ever present", 1), ("present on >= 10% of frames", 2),
                        ("present on >= 1% of frames", 3)):
        print("      {:<28} {:>10} {:>10}".format(name, *("{} / {}".format(use[side][0][index], use[side][1][index])
                                                         for side in ("degraded", "ours"))))
    gaps_degraded = gap_figures(slot_idle_gaps_s(atd.players_present, spans, inputs.fps))
    gaps_ours = gap_figures(slot_idle_gaps_s(ours_inputs.atd.players_present, ours_spans, ours_inputs.fps))
    returns, same = same_slot_returns({half: degrade.choose_half(view, half, parameters, 0, limits[half]).tracklets
                                       for half, _, _ in view.graded})
    plain_returns, plain_same = same_slot_returns({half: degrade.choose_tracklets(view, half, parameters, 0)
                                                   for half, _, _ in view.graded})
    print("      {:<26} {:>10} {:>10}".format("slot idle gaps", "degraded", "ours"))
    for name, index, template in (("count", 0, "{:,}"), ("median, s", 1, "{:.2f}"), ("at most 3 s", 2, "{:.1%}"),
                                  ("over 60 s", 3, "{:.1%}")):
        print("      {:<26} {:>10} {:>10}".format(name, template.format(gaps_degraded[index]),
                                                  template.format(gaps_ours[index])))
    print("      returns after a lost span: {:,}, {:,} of them to the same slot ({:.1%}); without the cap {:,}, {:,} "
          "of them ({:.1%}); ours has no identities".format(
              returns, same, same / returns if returns else math.nan, plain_returns, plain_same,
              plain_same / plain_returns if plain_returns else math.nan))
    assert gaps_degraded[0] > 0 and gaps_ours[0] > 0 and returns > 0

    # The lost share over the 7 matches and 5 seeds: the walk and the slots only, no arrays.
    on_pitch = lost = 0
    by_match = []
    for other_id in sources.MATCH_IDS:
        other = degrade.tracking_view(match if other_id == match_id else real_match(other_id))
        present = ~np.isnan(other.players_xy[:, :, 0])
        on = gone = 0
        for seed in range(5):
            for half, start, stop in other.graded:
                records = degrade.choose_tracklets(other, half, parameters, seed)
                on += int(present[start:stop].sum())
                gone += int(present[start:stop].sum()) - sum(t.stop - t.start for t in records)
        by_match.append("{} {:.2%}".format(other_id, gone / on))
        on_pitch, lost = on_pitch + on, lost + gone
    print("      lost share, seeds 0-4: {}; pooled {:.2%} against d {:.2%}".format(
        ", ".join(by_match), lost / on_pitch, parameters.d))
    assert abs(lost / on_pitch - parameters.d) < 0.003, (lost / on_pitch, parameters.d)


def second_difference_norms(xy_m, present, spans):
    """The norms, in metres, of the second difference p[t+1] - 2 p[t] + p[t-1] of the slots present at all three
    frames, inside each span, in span order: the array that shape_of_norms reduces."""
    norms = []
    for _, start, stop in spans:
        inside = present[start:stop]
        triple = inside[2:] & inside[1:-1] & inside[:-2]
        second = xy_m[start + 2:stop] - 2.0 * xy_m[start + 1:stop - 1] + xy_m[start:stop - 2]
        norms.append(np.sqrt((second * second).sum(axis=2))[triple])
    return np.concatenate(norms)


def shape_of_norms(norms):
    """(triples, RMS, median, 99th percentile, share of the squares in the largest 0.05% of triples) of an array
    of second-difference norms, which may be several matches' laid end to end."""
    squares = np.sort(norms * norms)[::-1]
    top = max(1, int(0.0005 * squares.size))
    return (int(norms.size), float(np.sqrt(squares.mean())), float(np.median(norms)), float(np.quantile(norms, 0.99)),
            float(squares[:top].sum() / squares.sum()))


def second_difference_shape(xy_m, present, spans):
    """(triples, RMS, median, 99th percentile, share of the squares in the largest 0.05% of triples) of the second
    difference p[t+1] - 2 p[t] + p[t-1] of the slots present at all three frames, inside each span, in metres.

    The RMS is calibration.player_stats' jitter, and the rest says what it is made of: DFL's is a few rare jumps (a
    large share of its squares in a tiny share of the triples) on a 1 cm grid, and not noise. What a centred average
    does to them inside a piece is measured on the profile's own draws, with and without it (T35, R004c)."""
    return shape_of_norms(second_difference_norms(xy_m, present, spans))


def test_second_difference_shape_counts_what_it_says():
    """One slot on a line at 0, 1, 0, 0, 3, 0, 0: the second differences are 2, 1, 3, 6 and 3 (worked out by hand).

    RMS = sqrt((4 + 1 + 9 + 36 + 9) / 5), the median is 3, the 99th percentile 3 + 0.96 * (6 - 3) by the linear rule,
    and the largest 0.05% of 5 triples is one triple, whose square is 36 of the 59. A triple with a missing frame is
    left out, and so is one that crosses the end of a span."""
    line = np.zeros((7, 1, 2))
    line[:, 0, 0] = [0, 1, 0, 0, 3, 0, 0]
    present = np.ones((7, 1), dtype=bool)
    count, rms, median, p99, share = second_difference_shape(line, present, [(1, 0, 7)])
    assert count == 5 and abs(rms - math.sqrt(59 / 5)) < 1e-12 and median == 3.0
    assert abs(p99 - 5.88) < 1e-12 and abs(share - 36 / 59) < 1e-12
    present[3, 0] = False                       # the triples 2-4, 3-5 and 4-6... all those with row 3 are gone
    assert second_difference_shape(line, present, [(1, 0, 7)])[0] == 2
    assert second_difference_shape(line, np.ones((7, 1), dtype=bool), [(1, 0, 4), (2, 4, 7)])[0] == 2 + 1


def test_second_difference_shape_pools_matches_by_their_norms():
    """The shape of the second differences of two matches pooled is the shape of their norms laid end to end.

    The data check pools the degraded players of 7 matches into one row of its table of what the jitter is made of
    without holding their positions: each match gives its norms (second_difference_norms), and shape_of_norms
    reduces the pool. One call of second_difference_shape on the two matches laid end to end, each in its own
    span, must give the same row to the last digit, and a single match must give the one that
    second_difference_shape gives."""
    rng = np.random.default_rng(5)
    first, second = rng.normal(size=(40, 3, 2)), rng.normal(size=(55, 3, 2)) * 3.0
    present_first, present_second = rng.random((40, 3)) > 0.1, rng.random((55, 3)) > 0.2
    spans_first, spans_second = [(1, 2, 40)], [(1, 0, 30), (2, 30, 55)]
    pooled = shape_of_norms(np.concatenate([second_difference_norms(first, present_first, spans_first),
                                            second_difference_norms(second, present_second, spans_second)]))
    laid = second_difference_shape(np.concatenate([first, second]), np.concatenate([present_first, present_second]),
                                   [(1, 2, 40), (1, 40, 70), (2, 70, 95)])
    assert pooled == laid, (pooled, laid)
    assert pooled[0] == second_difference_shape(first, present_first, spans_first)[0] + second_difference_shape(
        second, present_second, spans_second)[0]
    assert shape_of_norms(second_difference_norms(first, present_first, spans_first)) == second_difference_shape(
        first, present_first, spans_first)


# The one measurement that WRONG_TEAM_SCALE was worked out from (docs/results.md R004): the observable wrong-team frames
# of the 7 DFL matches at s = 1.0, and the share in ours (Calibration.wrong_team_share).
RECORDED_OBSERVED = (39756, 1002644)
RECORDED_TARGET = 0.105391764


def test_data_tracking_constants_follow_from_the_measurements():
    """The frozen constants (the scale of the episodes, the smoothing window and the scale of the tracked spans),
    re-derived from the 7 DFL matches and our Calibration (aggregates).

    Needs our match (the Calibration) and the seven downloaded matches, so it runs with --data and says NOT RUN where
    they are not there. Three rules were set before anything was measured, and this applies them to the code as it is:
    (1) smoothing: the players' second-difference RMS of the 7 clean matches (calibration.player_stats' own statistic,
    from each kick-off to the end of the graded block, pooled by sums of squares) against
    Calibration.player_jitter_rms_m: SMOOTH_WINDOW is not 0 if and only if DFL's is more than twice ours, and the
    smoothing lowers the pooled RMS of the profile's own draws (smooth=0 against the profile's, the same pieces and the
    same triples; not against the clean players, whose RMS the cuts change as much as the smoothing does, and, in
    versions 1 and 2, a slot's handovers changed more than either: T35); (2) the
    scale: the observable wrong-team share (player_stats on the degraded match, seed 0, the 7 matches pooled by sums) at
    s = 1.0 was not within 0.03 of ours, so WRONG_TEAM_SCALE is s = 1.0 rescaled once by target / observed (to 3
    decimals): the constant is compared with that one recorded measurement (RECORDED_OBSERVED, RECORDED_TARGET), and not
    with a scale worked out again from this run's, which it would follow wherever the code went. At the frozen scale the
    share must be within 0.03 of ours, the gate's own band, measured now. It prints both, per match and pooled, the
    degraded players' jitter after the smoothing beside DFL's and ours, the same measured inside tracklets (equal to it:
    asserted, since no slot changes player between two adjacent rows), what the jitter is made of (its median, 99th
    percentile and the share of its squares in rare jumps: ours, each clean match, and the degraded players pooled,
    with the average and without it) and the wrong-team episodes that player_stats sees in the
    degraded matches beside ours (information). (3) the scale of the tracked spans: the walk draws its spans from our
    final tracklets, which already hold the cuts that the cap and the episodes add, so TRACKED_SCALE is lambda = ours'
    tracklet median (the Calibration's) over the pooled median of this profile at lambda = 1 with the shortest-piece
    rule in (seed 0, the 7 matches laid end to end, as fidelity pools them), to 3 decimals, once: the constant is
    compared with the recorded measurement (RECORDED_MEDIAN_S), the measurement is made again here and must reproduce it
    (the code is the code that it was made with), and the median, the share of the frames lost and the pieces dropped at
    the frozen lambda are printed (information). The first scale is measured on the players of version 1's
    parameters (no cap, no rule, lambda 1; version 3's slot rule) and the frozen one, with the tracklet median, on
    the profile's. The frozen scale of the tracked spans is
    asserted first, against the recorded constants alone (the synthetic
    test_tracked_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements asserts it too), so that a moved
    constant is reported as that.
    """
    from pipeline.benchmark import sources
    for match_id in sources.MATCH_IDS:
        for kind in dfl.SOURCE_KINDS:
            if not os.path.isfile(sources.local_path(match_id, kind)):
                raise NotRun("{} is not downloaded ({}); {} to fetch the DFL matches".format(
                    match_id, kind, dfl.FETCH_ADVICE))
    # The frozen scale of the tracked spans first: it is compared with recorded constants only, so that a constant that
    # has moved is reported as that, and not as whatever else a different scale makes of the measurements below.
    assert degrade.TRACKED_SCALE == tracked_scale_from_the_record() == 2.072, (
        "TRACKED_SCALE is frozen at the value worked out once from the recorded measurements", degrade.TRACKED_SCALE)
    cal = calibration.load()
    target = cal.wrong_team_share
    jitter, seen_episodes = {"clean": [0.0, 0], "unsmoothed": [0.0, 0], "degraded": [0.0, 0]}, []
    inside = {"unsmoothed": [0.0, 0], "degraded": [0.0, 0]}     # the same two, taken inside tracklets (squares, triples)
    pooled_norms = {"degraded": [], "unsmoothed": []}          # the same two draws' second-difference norms, per match
    wrong = {"first": [0, 0], "frozen": [0, 0]}
    medians = {"one": [], "frozen": []}
    spells = {"frozen": [0, 0, 0]}                      # frames on the pitch, frames lost, pieces dropped by the rule
    from pipeline.inputs import Inputs
    ours = Inputs()
    ours_spans = [(half, ours.kickoffs[half], stop) for half, _, stop in ours.graded]
    shapes = {"ours": second_difference_shape(ours.metres(ours.atd.players_xy), ours.atd.players_present, ours_spans)}
    print("      {:<8} {:>14} {:>16} {:>16}".format("match", "clean jitter m", "wrong, s = 1.0", "wrong, frozen s"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        view = degrade.tracking_view(match)
        spans = [(half, view.kickoffs[half], stop) for half, _, stop in view.graded]
        pitch = np.asarray(view.pitch)
        clean = match_inputs.clean_atd(match)
        stats = {"clean": calibration.player_stats(clean.players_present, clean.slot_team, clean.players_xy * pitch,
                                                   clean.players_speed, spans, view.fps)}
        base = degrade.calibrated_parameters(cal, view.fps)
        assert base.scale == degrade.WRONG_TEAM_SCALE and base.smooth == degrade.SMOOTH_WINDOW
        assert base.tracked_scale == degrade.TRACKED_SCALE and base.min_piece == degrade.MIN_PIECE_FRAMES
        version_1 = degrade.calibrated_parameters(cal, view.fps, tracked_scale=1.0, min_piece=0)
        unit = degrade.calibrated_parameters(cal, view.fps, tracked_scale=1.0)
        limits = degrade.cap_limits(view, cal, 0)
        for name, parameters, caps in (("first", dataclasses.replace(version_1, scale=1.0, smooth=0), None),
                                       ("one", unit, limits), ("frozen", base, limits)):
            xy, speed, records, counts = degrade.degraded_players(view, parameters, 0, caps)
            stats[name] = calibration.player_stats(np.isfinite(xy[:, :, 0]), degrade.slot_team(), xy * pitch, speed,
                                                   spans, view.fps)
            if name in medians:
                medians[name].append(stats[name]["tracklet_lengths_s"])
            if name == "one":
                continue
            wrong[name][0] += stats[name]["wrong_team_frames"]
            wrong[name][1] += stats[name]["frames"]
            if name == "frozen":
                jitter["degraded"][0] += stats[name]["jitter_rms_m"] ** 2 * stats[name]["jitter_triples"]
                jitter["degraded"][1] += stats[name]["jitter_triples"]
                # The same draws with no smoothing: the same pieces, so the same slots present and the same triples.
                rough_xy, rough_speed, rough_records, _ = degrade.degraded_players(
                    view, dataclasses.replace(base, smooth=0), 0, limits)
                assert np.array_equal(np.isfinite(rough_xy[:, :, 0]), np.isfinite(xy[:, :, 0])), match_id
                rough = calibration.player_stats(np.isfinite(rough_xy[:, :, 0]), degrade.slot_team(), rough_xy * pitch,
                                                 rough_speed, spans, view.fps)
                assert rough["jitter_triples"] == stats[name]["jitter_triples"], match_id
                jitter["unsmoothed"][0] += rough["jitter_rms_m"] ** 2 * rough["jitter_triples"]
                jitter["unsmoothed"][1] += rough["jitter_triples"]
                # The jitter taken inside tracklets only. Since version 3 a slot never changes player between two
                # adjacent present rows, so every triple of a slot is inside one tracklet and the two figures are the
                # same: asserted, and printed side by side. In versions 1 and 2 the per-slot figure held the triples
                # that straddle a slot handover, and was 6.47 and 8.49 cm with the average against 1.80 and 1.75 cm
                # inside the tracklets, on the same draws (3 seeds: R004c).
                for name_, array, own in (("degraded", xy, records), ("unsmoothed", rough_xy, rough_records)):
                    count, rms_inside = jitter_inside_tracklets(array * pitch, own)
                    per_slot = stats[name] if name_ == "degraded" else rough
                    assert count == per_slot["jitter_triples"], (match_id, name_, count, per_slot["jitter_triples"])
                    assert abs(rms_inside - per_slot["jitter_rms_m"]) < 1e-9, (match_id, name_, rms_inside)
                    inside[name_][0] += rms_inside ** 2 * count
                    inside[name_][1] += count
                    pooled_norms[name_].append(second_difference_norms(array * pitch, np.isfinite(array[:, :, 0]),
                                                                       spans))
                clean_present = np.isfinite(np.asarray(view.players_xy)[:, :, 0])
                for half, start, stop in view.graded:
                    spells["frozen"][0] += int(clean_present[start:stop].sum())
                    spells["frozen"][1] += int(clean_present[start:stop].sum()) - int(
                        np.isfinite(xy[start:stop, :, 0]).sum())
                    spells["frozen"][2] += counts[half]["short_pieces"]
        jitter["clean"][0] += stats["clean"]["jitter_rms_m"] ** 2 * stats["clean"]["jitter_triples"]
        jitter["clean"][1] += stats["clean"]["jitter_triples"]
        print("      {:<8} {:>14.4f} {:>16.4f} {:>16.4f}".format(
            match_id, stats["clean"]["jitter_rms_m"], stats["first"]["wrong_team_share"],
            stats["frozen"]["wrong_team_share"]))
        shapes[match_id] = second_difference_shape(clean.players_xy * pitch, clean.players_present, spans)
        episodes = stats["frozen"]["wrong_team_episodes_s"]
        seen_episodes.append((match_id, int(episodes.size), float(np.median(episodes)),
                              float(np.quantile(episodes, 0.9))))
        del match, view, clean
    rms = {name: math.sqrt(squares / triples) for name, (squares, triples) in jitter.items()}
    rms_inside = {name: math.sqrt(squares / triples) for name, (squares, triples) in inside.items()}
    # The degraded players' rows of the table of what the jitter is made of, pooled over the 7 matches (seed 0, the
    # frozen profile), through the same function as DFL's and ours: the draws with the average and without it.
    for name in ("degraded", "unsmoothed"):
        shapes[name] = shape_of_norms(np.concatenate(pooled_norms[name]))
        assert shapes[name][0] == jitter[name][1] and abs(shapes[name][1] - rms[name]) < 1e-9, (
            "the table's row is the figure above it", name, shapes[name][:2], rms[name])
    first, frozen = (wrong[name][0] / wrong[name][1] for name in ("first", "frozen"))
    print("      jitter, RMS of the second difference: DFL clean {:.4f} m ({:.1f} x ours), degraded with the {}-frame "
          "average {:.4f} m ({:.1f} x ours) and, the same draws without it, {:.4f} m; ours {:.4f} m".format(
              rms["clean"], rms["clean"] / cal.player_jitter_rms_m, degrade.SMOOTH_WINDOW, rms["degraded"],
              rms["degraded"] / cal.player_jitter_rms_m, rms["unsmoothed"], cal.player_jitter_rms_m))
    print("      jitter inside tracklets (the same draws, the triples of one tracklet): with the {}-frame average {:.4f} m, "
          "without it {:.4f} m; per slot, above, {:.4f} m and {:.4f} m: the same, as no slot changes player between "
          "two adjacent rows".format(degrade.SMOOTH_WINDOW, rms_inside["degraded"], rms_inside["unsmoothed"],
                                     rms["degraded"], rms["unsmoothed"]))
    print("      wrong-team share, seed 0, pooled by sums: s = 1.0 gives {:.5f} ({:,} of {:,} frames), s = {} gives "
          "{:.5f} ({:,}); ours {:.5f}, gate +-0.03".format(first, *wrong["first"], degrade.WRONG_TEAM_SCALE, frozen,
                                                          wrong["frozen"][0], target))
    print("      what the jitter is made of (second difference of a position, m): RMS, median, 99th percentile, "
          "share of the squares in the largest 0.05% of triples; ours, each match's clean players, and the degraded "
          "ones pooled over the matches (seed 0), with the {}-frame average and without it".format(
              degrade.SMOOTH_WINDOW))
    for name, (count, rms_m, median, p99, share) in shapes.items():
        print("      {:<10} {:>9,} triples   {:.4f} {:.4f} {:.4f}   {:.2f}".format(
            name, count, rms_m, median, p99, share))
    print("      wrong-team episodes as player_stats sees them at the frozen scale (seed 0, count, median, 90th "
          "percentile): {}; ours {} of median {:.2f} s and 90th percentile {:.2f} s".format(
              ", ".join("{} {} ({:.1f} s, {:.1f} s)".format(*row) for row in seen_episodes),
              cal.wrong_team_episodes_s.size, float(np.median(cal.wrong_team_episodes_s)),
              float(np.quantile(cal.wrong_team_episodes_s, 0.9))))
    assert (degrade.SMOOTH_WINDOW > 0) == (rms["clean"] > 2 * cal.player_jitter_rms_m), (degrade.SMOOTH_WINDOW, rms)
    assert abs(shapes["ours"][1] - cal.player_jitter_rms_m) < 1e-6, "the same statistic, here and in the Calibration"
    # The smoothing is tested on the same draws with and without it (smooth=0: the same pieces, the same slots present,
    # the same triples), and not against DFL's clean players: the degraded ones are cut by the walk, the episodes, the
    # cap and the shortest-piece rule first, and which of DFL's jumps the cuts leave decides the sign of a comparison
    # with the clean players, whatever the smoothing does (and, in versions 1 and 2, a slot's handovers did too: T35).
    assert rms["degraded"] < rms["unsmoothed"], ("smoothing lowers the jitter of the same draws", rms)
    # The frozen scale is the value that was worked out once, from the measurement recorded in R004 (39,756 of 1,002,644
    # frames, 0.03965, at s = 1.0, against ours 0.105391764), and not one worked out again from what this run measures:
    # a scale that followed every change of the code could never be seen to move. What this run measures is printed
    # above; the frozen scale stays what it was, and must still land inside the gate's band.
    worked_out = round(RECORDED_TARGET / (RECORDED_OBSERVED[0] / RECORDED_OBSERVED[1]), 3)
    assert degrade.WRONG_TEAM_SCALE == worked_out == 2.658, (degrade.WRONG_TEAM_SCALE, worked_out)
    assert abs(target - RECORDED_TARGET) < 1e-9, "the Calibration is the one the scale was worked out against"
    assert abs(frozen - target) <= 0.03, (frozen, target)

    # The scale of the tracked spans (rule 3): the same one-shot, from the recorded median, and the measurement again.
    ours_median = float(np.median(cal.tracklet_lengths_s))
    unit_median = float(np.median(np.concatenate(medians["one"])))
    frozen_median = float(np.median(np.concatenate(medians["frozen"])))
    print("      tracklet median, seed 0, 7 matches end to end: {:.2f} s at lambda 1 (recorded {:.2f} s), {:.2f} s "
          "at lambda {} (ours {:.2f} s); lost share {:.4f} with the cap, {:,} pieces dropped by the shortest-piece "
          "rule".format(
              unit_median, RECORDED_MEDIAN_S, frozen_median, degrade.TRACKED_SCALE, ours_median,
              spells["frozen"][1] / spells["frozen"][0], spells["frozen"][2]))
    assert abs(ours_median - RECORDED_OURS_MEDIAN_S) < 1e-9, "the Calibration is the one lambda was worked out against"
    assert abs(unit_median - RECORDED_MEDIAN_S) < 1e-9, "the recorded measurement reproduces with this code"
    assert degrade.TRACKED_SCALE == round(ours_median / RECORDED_MEDIAN_S, 3) == 2.072, degrade.TRACKED_SCALE
    assert degrade.MIN_PIECE_FRAMES == int(round(float(cal.tracklet_lengths_s.min()) * ours.fps)) == 8, (
        "the shortest tracklet of ours (A09)")
