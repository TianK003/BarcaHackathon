"""Checks of the tracking profile's cap of the low-count tail, its shortest-piece rule and the scale of its tracked
spans (degrade.cap_limits, cap_pieces, drop_short_pieces and TRACKED_SCALE, "lambda"): what Task 14 (R004b) added so that
the profile's players show our match's stretches with few players present, and our tracklets' lengths.

They moved out of checks/tracking.py, which had reached 2,200 lines, with nothing changed. What they share with it (the
steady match, the pairs that make wrong-team episodes, the record of the measurement that lambda was worked out from)
lives in checks/common.py. They run on cap_match, a match whose 22 players stand still on a line, so that a distance is a
short dyadic number and a cut can be worked out by hand, and on reference_cap, which writes the rule out one row at a
time with plain loops. Every check here is synthetic: none reads the data directory.
"""

import dataclasses
import math
import types

import numpy as np

from pipeline.benchmark import degrade
from pipeline.benchmark.checks.common import (RECORDED_MEDIAN_S, RECORDED_MEDIAN_V2_S, RECORDED_OURS_MEDIAN_S,
                                              episode_parameters, is_carried, made_up_calibration, pairs_match,
                                              reference_episodes, smoothed_plain, steady_match,
                                              tracked_scale_from_the_record, without_new_parts)
from pipeline.selftest import expect_error

# Params
FULL_SQUAD = (0,) * 11 + (1,) * 11      # 22 players on the pitch: the squads of the cap's checks (CAP is 18)
NO_CAP = {"capped_frames": 0, "cap_slot_frames": 0, "cap_pieces_cut": 0}


# -- the cap of the low-count tail ------------------------------------------------------------------------------------

def cap_match(ball, rows=(40, 30), teams=FULL_SQUAD, off=None):
    """A match whose players stand still all through both halves, on a line, and whose ball is `ball`.

    Player j is at x = (j + 1) / 32 and y = 0.5 (normalised; the pitch is 105 x 68 m): thirty-seconds are exact in
    binary, so a distance is a short dyadic number and two players on either side of the ball are exactly as far from
    it. `ball(half, row)` is the ball's (x, y) on a row of a half, counted from the half's start, or None where DFL has
    none. `off` is {player: first row of half 1 at which he is on the pitch}, nobody missing by default. Nobody is lost
    and nobody is off, so a piece of the cap's checks is a player's whole half."""
    base = steady_match(rows=rows, teams=teams, substitutions=False)
    xy = np.full(base.players_xy.shape, np.nan)
    ball_xy = np.full(base.ball_xy.shape, np.nan)
    for half, start, stop in base.graded:
        for player in range(len(teams)):
            xy[start:stop, player] = ((player + 1) / 32, 0.5)
        for row in range(stop - start):
            spot = ball(half, row)
            if spot is not None:
                ball_xy[start + row] = spot
    for player, first in (off or {}).items():
        xy[base.graded[0][1]:base.graded[0][1] + first, player] = np.nan
    xy.setflags(write=False)
    ball_xy.setflags(write=False)
    return dataclasses.replace(base, players_xy=xy, ball_xy=ball_xy)


def whole_halves(view, half):
    """[(first, player, last, pool)] of a half of a view whose players are all on the pitch for all of it."""
    start, stop = next((first, last) for number, first, last in view.graded if number == half)
    return [(start, player, stop, int(view.player_team[player])) for player in range(view.players_xy.shape[1])
            if not math.isnan(view.players_xy[start, player, 0])]


def limit_for(view, half, runs, base=22):
    """The cap's counts of a half: `base` on every row from its kick-off, and k on the rows (first, stop, k) of `runs`,
    counted from the kick-off."""
    kickoff, stop = view.kickoffs[half], next(last for number, _, last in view.graded if number == half)
    limit = np.full(stop - kickoff, base, dtype=np.int16)
    for first, last, k in runs:
        limit[first:last] = k
    return limit


def rows_kept(pieces):
    """{player: [rows]} that a list of pieces (first, player, last, pool) has him on, one row at a time."""
    kept = {}
    for first, player, last, _ in pieces:
        kept.setdefault(player, []).extend(range(first, last))
    return {player: sorted(rows) for player, rows in kept.items()}


def reference_cap(view, half, pieces, limit):
    """(pieces, counts) of the cap as degrade.py documents it, written out one row at a time with plain loops.

    The ball of a row is the match's own where it has one, else its last place in the half, else the centre spot. A
    run is a maximal stretch of rows whose count is at most CAP. At the first row of a run the pieces kept are the k
    nearest the ball (ties to the lower player column); afterwards the kept pieces that are still present stay, and if
    they are more than k the farthest go, and if fewer the nearest present pieces that are not kept come in; outside
    a run everybody is kept, and the kept set starts again at the next run. A piece is cut at the rows it is not kept
    on."""
    start, stop = next((first, last) for number, first, last in view.graded if number == half)
    kickoff = view.kickoffs[half]
    length, width = view.pitch
    last_seen, ball = (0.5, 0.5), {}
    for row in range(start, stop):
        spot = view.ball_xy[row]
        if not (math.isnan(spot[0]) or math.isnan(spot[1])):
            last_seen = (float(spot[0]), float(spot[1]))
        ball[row] = last_seen
    dropped = {index: [] for index in range(len(pieces))}
    kept = None
    for row in range(kickoff, stop):
        k = int(limit[row - kickoff])
        if k > degrade.CAP:
            kept = None
            continue
        present = [index for index, (first, _, last, _) in enumerate(pieces) if first <= row < last]
        kept = [] if kept is None else [index for index in kept if index in present]

        def far(index):
            spot = view.players_xy[row, pieces[index][1]]
            return (math.hypot((spot[0] - ball[row][0]) * length, (spot[1] - ball[row][1]) * width), pieces[index][1])

        if len(kept) > k:
            kept = sorted(kept, key=far)[:k]
        elif len(kept) < k:
            kept += sorted((index for index in present if index not in kept), key=far)[:k - len(kept)]
        for index in present:
            if index not in kept:
                dropped[index].append(row)
    out, cut, frames = [], 0, 0
    for index, (first, player, last, pool) in enumerate(pieces):
        gone = set(dropped[index])
        cut += bool(gone)
        frames += len(gone)
        row = first
        while row < last:
            if row in gone:
                row += 1
                continue
            end = row
            while end < last and end not in gone:
                end += 1
            out.append((row, player, end, pool))
            row = end
    low = sum(1 for k in limit if k <= degrade.CAP)
    return sorted(out), {"capped_frames": low, "cap_slot_frames": frames, "cap_pieces_cut": cut}


def test_cap_keeps_nearest():
    """On a capped row only the k tracked players nearest the ball stay: the rest are dropped on that row, and counted.

    22 players on a line (cap_match), the ball at x = 10/32, on the player in column 9. Distances in 1/32 of the pitch's
    length: column 9 is at 0, columns 8 and 10 at 1, and so on, with columns j and 18 - j equally far, so a tie goes to
    the lower column. In half 1, rows 10-19 have a count of 18 (CAP itself: capped), rows 25-29 a count of 1 and rows
    32-34 a count of 2; the nearest 18 are columns 0 to 17 (columns 0 and 18 are tied at 9, and column 0 stays), so
    columns 18 to 21 are dropped on rows 10-19; the nearest 1 is column 9; the nearest 2 are columns 9 and 8 (8 and 10
    are tied). Every other row of every piece is kept, and the counts are those of the drops: 18 rows, 40 + 105 + 60
    slot-frames, and every player but column 9 cut."""
    match = cap_match(lambda half, row: (10 / 32, 0.5))
    view = degrade.tracking_view(match)
    pieces = whole_halves(view, 1)
    start = match.graded[0][1]
    limit = limit_for(view, 1, [(10, 20, 18), (25, 30, 1), (32, 35, 2)])
    got, counts = degrade.cap_pieces(view, 1, list(pieces), limit)
    expected = {}
    for player in range(22):
        expected[player] = []
        for at in range(40):
            if 10 <= at < 20:
                stays = player <= 17
            elif 25 <= at < 30:
                stays = player == 9
            elif 32 <= at < 35:
                stays = player in (8, 9)
            else:
                stays = True
            if stays:
                expected[player].append(start + at)
    kept = rows_kept(got)
    wrong = {player: (kept[player], expected[player]) for player in kept if kept[player] != expected[player]}
    assert kept == expected, wrong
    assert counts == {"capped_frames": 10 + 5 + 3, "cap_slot_frames": 10 * 4 + 5 * 21 + 3 * 20,
                      "cap_pieces_cut": 21}, counts
    # The pieces are the players' stretches between the drops, each in its player's own pool, in order.
    assert all(pool == int(view.player_team[player]) for _, player, _, pool in got) and got == sorted(got)
    assert [piece for piece in got if piece[1] == 9] == [(start, 9, start + 40, 0)], "never dropped: one whole piece"
    assert [piece for piece in got if piece[1] == 21] == [
        (start, 21, start + 10, 1), (start + 20, 21, start + 25, 1), (start + 30, 21, start + 32, 1),
        (start + 35, 21, start + 40, 1)]
    # The same through the reference, and the input is not changed.
    assert (got, counts) == reference_cap(view, 1, pieces, limit)
    assert pieces == whole_halves(view, 1)
    # Half 2 has no capped row: its pieces come back as they went in.
    again, none = degrade.cap_pieces(view, 2, whole_halves(view, 2), limit_for(view, 2, []))
    assert again == whole_halves(view, 2) and none == NO_CAP


def test_cap_inactive_above_threshold():
    """A count above CAP caps nothing: 18 is capped, 19 is not, and no limit at all is no cap.

    The same 22 players and the same rows with 19 (or 22, or 46, the slots of a whole tracker) give back the pieces that
    went in, to the object; a count of 18 on one row drops the 4 players farthest from the ball on it, 0 drops everybody
    and a count that is not made of whole numbers or is not a count of each row of the half is refused."""
    match = cap_match(lambda half, row: (10 / 32, 0.5))
    view = degrade.tracking_view(match)
    pieces = whole_halves(view, 1)
    for k in (19, 22, 46):
        got, counts = degrade.cap_pieces(view, 1, pieces, limit_for(view, 1, [(5, 25, k)]))
        assert got == pieces and counts == NO_CAP, k
    got, counts = degrade.cap_pieces(view, 1, pieces, None)
    assert got == pieces and counts == NO_CAP
    got, counts = degrade.cap_pieces(view, 1, pieces, limit_for(view, 1, [(5, 6, degrade.CAP)]))
    assert counts == {"capped_frames": 1, "cap_slot_frames": 4, "cap_pieces_cut": 4} and len(got) == 22 + 4, counts
    got, counts = degrade.cap_pieces(view, 1, pieces, limit_for(view, 1, [(5, 6, 0)]))
    start = match.graded[0][1]
    assert counts == {"capped_frames": 1, "cap_slot_frames": 22, "cap_pieces_cut": 22}, counts
    assert all(start + 5 not in range(first, last) for first, _, last, _ in got), "nobody is kept on a count of 0"
    expect_error(ValueError, lambda: degrade.cap_pieces(view, 1, pieces, np.full(10, 22, dtype=np.int16)), "half 1",
                 "40", "10")
    expect_error(ValueError, lambda: degrade.cap_pieces(view, 1, pieces, np.full(40, 22.0)), "whole numbers")


def test_cap_is_deterministic_and_draws_nothing():
    """The cap draws no random number: the same pieces and counts give the same pieces, whatever generator exists.

    rng_for is replaced by a function that fails if it is called, and the cap runs on a capped half; it runs twice, with
    equal results, and equal to the reference."""
    match = cap_match(lambda half, row: (row / 64, 0.5))
    view = degrade.tracking_view(match)
    pieces = whole_halves(view, 1)
    limit = limit_for(view, 1, [(3, 12, 5), (20, 29, 18)])
    real = degrade.rng_for

    def never(*arguments):
        raise AssertionError("the cap drew from rng_for{}".format(arguments))

    degrade.rng_for = never
    try:
        first = degrade.cap_pieces(view, 1, pieces, limit)
        second = degrade.cap_pieces(view, 1, pieces, limit)
    finally:
        degrade.rng_for = real
    assert first == second and first[1]["cap_slot_frames"] > 0
    assert first == reference_cap(view, 1, pieces, limit)


def test_cap_membership_changes_only_when_the_count_requires_it():
    """Sticky membership: two distances that swap change nothing; a count that rises or falls, a piece that ends and a
    piece that starts are what does, and a run that ends lets everybody back.

    22 players on a line (cap_match) and the ball in three places: x = 1/32 on rows 0-9, 6/32 on rows 10-24 and 17/32
    from row 25. The counts, rows 0-14 of half 1 at 3, 15-17 at 5, 18-19 at 2, rows 20-24 at 22 (no cap) and 25-29 at 2.
    Worked out by hand, in 1/32 of the length: the first row keeps the 3 nearest the ball at 1/32: columns 0, 1, 2; the
    ball then moves to column 5 (rows 10-14), where the nearest 3 are 4, 5 and 6, but the three kept stay, as the count
    has not changed; the count rises to 5 on row 15 and the two nearest players that are not kept come in, column 5 (0
    from the ball) and column 4 (1, tied with 6: the lower column); on row 18 the count falls to 2 and the farthest of
    the five go (column 0 at 5, 1 at 4, 2 at 3), leaving 5 and 4; then rows 20-24 are not capped and everybody is back;
    the next run (rows 25-29, the ball at column 16) starts again, from nothing: the 2 nearest are 16 and 15 (15 and 17
    tied).

    Then a piece that ENDS inside a run is replaced by the nearest present player who is not kept, and a piece that
    STARTS inside it is not kept unless there is room. 23 players, the ball at column 0's place, the count 3 on rows
    0-14: column 0's piece stops at row 8, and the nearest players present who are not kept are column 3 (3/32 from the
    ball; 4 and 5 are farther): the three are 0, 1, 2 until row 7 and 1, 2, 3 from row 8. Column 22, who comes on at row
    10 on the ball's own place, nearer than all three, is not kept: the count is met. From row 15 (no cap) everybody is
    back. Last, room that nobody is in is not kept for a piece that has not started (the third scene, below)."""
    def ball(half, row):
        return ((1 if row < 10 else 6 if row < 25 else 17) / 32, 0.5)

    match = cap_match(ball)
    view = degrade.tracking_view(match)
    pieces = whole_halves(view, 1)
    limit = limit_for(view, 1, [(0, 15, 3), (15, 18, 5), (18, 20, 2), (25, 30, 2)])
    got, counts = degrade.cap_pieces(view, 1, pieces, limit)
    start = match.graded[0][1]
    plan = [(0, 15, {0, 1, 2}), (15, 18, {0, 1, 2, 4, 5}), (18, 20, {4, 5}), (25, 30, {15, 16})]
    expected = {player: [] for player in range(22)}
    for at in range(40):
        stays = next((members for first, last, members in plan if first <= at < last), set(range(22)))
        for player in sorted(stays):
            expected[player].append(start + at)
    assert rows_kept(got) == expected
    assert counts == {"capped_frames": 25, "cap_slot_frames": sum((last - first) * (22 - len(members))
                                                                  for first, last, members in plan),
                      "cap_pieces_cut": 22}, counts
    assert (got, counts) == reference_cap(view, 1, pieces, limit)

    squad = FULL_SQUAD + (0,)
    match = cap_match(lambda half, row: (1 / 32, 0.5), teams=squad)
    xy = np.array(match.players_xy)
    xy[:, 22] = (1 / 32, 0.5)
    xy.setflags(write=False)
    view = degrade.tracking_view(dataclasses.replace(match, players_xy=xy))
    start, stop = match.graded[0][1:]
    pieces = [(start, player, start + 8 if player == 0 else stop, int(squad[player])) for player in range(22)]
    pieces += [(start + 10, 22, stop, 0)]
    limit = limit_for(view, 1, [(0, 15, 3)])
    got, counts = degrade.cap_pieces(view, 1, sorted(pieces), limit)
    kept = {player: set(rows) for player, rows in rows_kept(got).items()}
    for at in range(15):
        members = {player for player in range(23) if start + at in kept.get(player, ())}
        assert members == ({0, 1, 2} if at < 8 else {1, 2, 3}), (at, members)
    assert all(start + at in kept[22] for at in range(15, stop - start)), "back when the run ends"
    assert all(start + at in kept[0] for at in range(8)) and start + 8 not in kept[0], "his piece is over"
    assert (got, counts) == reference_cap(view, 1, sorted(pieces), limit)

    # Room that nobody is in is not kept for a piece that has not started: three pieces (columns 0, 1, 2) whole, a fourth
    # (column 3, on the ball's own place, so the nearest of all) that starts on row 5, the count 5 on rows 0-4 (more than
    # the three pieces there are: two places spare) and 3 from row 5 to row 9. On row 5 the three kept are as many as
    # the count, so column 3 is dropped on rows 5-9, whichever is nearest; it is back on row 10, when the cap ends.
    match = cap_match(lambda half, row: (4 / 32, 0.5))
    view = degrade.tracking_view(match)
    start, stop = match.graded[0][1:]
    pieces = [(start, player, stop, 0) for player in range(3)] + [(start + 5, 3, stop, 0)]
    limit = limit_for(view, 1, [(0, 5, 5), (5, 10, 3)])
    got, counts = degrade.cap_pieces(view, 1, pieces, limit)
    kept = {player: set(rows) for player, rows in rows_kept(got).items()}
    assert all(all(start + at in kept[player] for at in range(40)) for player in range(3))
    assert sorted(kept[3]) == [start + at for at in range(10, 40)], "dropped on rows 5-9 whichever is nearest"
    assert counts == {"capped_frames": 10, "cap_slot_frames": 5, "cap_pieces_cut": 1}, counts
    assert (got, counts) == reference_cap(view, 1, pieces, limit)


def test_cap_ball_fallbacks():
    """The ball of the cap is DFL's own ball: its place on the row, else its last place in the half, else the centre.

    22 players on a line (cap_match), each capped row on its own (a run of one row, so that nothing is kept from the row
    before) with a count of 1, so that the player kept is the one nearest the ball, and a ball on a player's own place
    keeps him alone. Half 1: DFL has no ball on rows 0-4, so on row 2 nothing has been seen in the half and the ball is
    taken at the centre spot, x = 16/32, where column 15 stands; on rows 5-9 it is at (5 + 2 (row - 5))/32, so on row 7
    at 9/32, where column 8 stands; it is missing again from row 10, so on rows 12 and 30 it is where it was last seen,
    row 9's 13/32 (column 12). Half 2 has no ball on rows 0-19: its row 4 is at the centre spot (column 15) and not at
    half 1's last place, and from row 20 the ball is at 3/32, which is column 2's, on rows 21 and 30."""
    def ball(half, row):
        if half == 1 and 5 <= row < 10:
            return ((5 + 2 * (row - 5)) / 32, 0.5)
        if half == 2 and row >= 20:
            return (3 / 32, 0.5)
        return None

    match = cap_match(ball, rows=(40, 40))
    view = degrade.tracking_view(match)
    cases = ((1, [(2, 3, 1), (7, 8, 1), (12, 13, 1), (30, 31, 1)], ((2, 15), (7, 8), (12, 12), (30, 12))),
             (2, [(4, 5, 1), (21, 22, 1), (30, 31, 1)], ((4, 15), (21, 2), (30, 2))))
    for half, singles, expected in cases:
        pieces = whole_halves(view, half)
        limit = limit_for(view, half, singles)
        got, counts = degrade.cap_pieces(view, half, pieces, limit)
        start = next(first for number, first, _ in view.graded if number == half)
        kept = rows_kept(got)
        for at, player in expected:
            members = [p for p, rows in kept.items() if start + at in rows]
            assert members == [player], (half, at, members, player)
        assert counts["capped_frames"] == len(singles) and counts["cap_slot_frames"] == 21 * len(singles)
        assert (got, counts) == reference_cap(view, half, pieces, limit)


def test_cap_matches_the_documented_rule_on_random_pieces():
    """cap_pieces equals the reference, row by row, on pieces made by the walk and the episodes and random counts.

    Two squads, each with 3 players coming and going (substitutions), real tracklets (d = 0.2) cut by wrong-team
    episodes, a ball that is missing on a third of the rows, and counts that come in runs of random length and level,
    over several seeds and both halves. Twenty-two players, with counts that are mostly below the players present (the
    cap drops many); and six, with counts at or above the players present as often as below, so that a count is often
    more than the pieces there are to keep (the spare room is not used by pieces that are not there yet, which is
    what a piece that starts inside a run relies on). The pieces and the counts of the drops must be the reference's
    (reference_cap, above), and the input is left as it was."""
    squads = ((FULL_SQUAD, [22, 19, 18, 14, 9, 5, 2, 0], 100, 1000), ((0, 0, 0, 1, 1, 1), [19, 8, 7, 6, 5, 4, 3, 2, 0],
                                                                      20, 100))
    for teams, levels, cuts, drops in squads:
        match = steady_match(rows=(300, 260), teams=teams, substitutions=True)
        ball = np.random.default_rng(5).uniform(0.0, 1.0, size=match.ball_xy.shape)
        ball[np.random.default_rng(6).uniform(size=ball.shape[0]) < 0.3] = np.nan
        ball.setflags(write=False)
        view = degrade.tracking_view(dataclasses.replace(match, ball_xy=ball))
        parameters = degrade.tracking_parameters([20, 40, 90, 200], 0.2, episodes=[10, 30, 50], wrong_team_share=0.3,
                                                 scale=1.0)
        cut = frames = 0
        for seed in range(6):
            for half, start, stop in view.graded:
                rng = np.random.default_rng([seed, half, 11])
                limit = np.full(stop - view.kickoffs[half], 22, dtype=np.int16)
                row = 0
                while row < limit.size:
                    length = int(rng.integers(1, 40))
                    limit[row:row + length] = int(rng.choice(levels))
                    row += length
                pieces, _ = degrade.episode_pieces(view, half, degrade.walk_spans(view, half, parameters, seed),
                                                   parameters, seed)
                before = list(pieces)
                got = degrade.cap_pieces(view, half, pieces, limit)
                assert pieces == before, "the input is left as it was"
                assert got == reference_cap(view, half, pieces, limit), (len(teams), seed, half)
                cut, frames = cut + got[1]["cap_pieces_cut"], frames + got[1]["cap_slot_frames"]
        assert cut > cuts and frames > drops, (len(teams), cut, frames)


def test_cap_leaves_min_of_the_count_and_the_tracked_players_on_every_row():
    """After the cap a row has as many players as the count allows and no more than were tracked: min(count, tracked).

    On a squad of 22 (the made-up Calibration, whose counts have 18, 10 and 7 in runs among 22, 20 and 19), with the
    shortest-piece rule off so that the cap is alone: the number of slots present on a row is the number tracked without
    the cap (the same draws: the cap draws nothing) when the count is above CAP, and the smaller of the count and that
    number when it is not; the record's drops are the sum of the difference, and the rows outside the graded blocks have
    nobody. The profile itself, with the rule on, is the same players as degraded_players with its parameters and
    limits (what degrade passes), the cap's record is the same, and no row has more players than the cap leaves."""
    match = steady_match(rows=(240, 200), teams=FULL_SQUAD, substitutions=True)
    view = degrade.tracking_view(match)
    cal = made_up_calibration()
    profile = degrade.calibrated_parameters(cal, view.fps)
    parameters = dataclasses.replace(profile, min_piece=0)
    low = rows = 0
    outside = np.ones(match.n_frames, dtype=bool)
    for _, start, stop in view.graded:
        outside[start:stop] = False
    for seed in range(5):
        limits = degrade.cap_limits(view, cal, seed)
        before = np.isfinite(degrade.degraded_players(view, parameters, seed)[0][:, :, 0]).sum(axis=1)
        xy, _, _, counts = degrade.degraded_players(view, parameters, seed, limits)
        after = np.isfinite(xy[:, :, 0]).sum(axis=1)
        dropped = 0
        for half, start, stop in view.graded:
            kickoff = view.kickoffs[half]
            count = limits[half]
            want = np.where(count <= degrade.CAP, np.minimum(before[kickoff:stop], count), before[kickoff:stop])
            assert (after[start:kickoff] == before[start:kickoff]).all() and (after[kickoff:stop] == want).all(), (
                seed, half)
            dropped += int((before[start:stop] - after[start:stop]).sum())
            assert counts[half]["capped_frames"] == int((count <= degrade.CAP).sum())
            low, rows = low + counts[half]["capped_frames"], rows + stop - kickoff
        assert sum(record["cap_slot_frames"] for record in counts.values()) == dropped > 0, (seed, dropped)
        assert (after[outside] == 0).all()
        atd, info = degrade.degrade(view, "tracking", cal, seed)
        same = degrade.degraded_players(view, profile, seed, limits)[0]
        assert atd.players_xy.tobytes() == same.tobytes(), "the profile is these players, rule included"
        assert info["players"]["cap_slot_frames"] == dropped, "the rule is after the cap: the record is the cap's"
        assert info["players"]["capped_frames"] == sum(
            record["capped_frames"] for record in info["players"]["halves"].values())
        assert (atd.players_present.sum(axis=1) <= after).all(), "the rule only takes players away"
    assert 0.03 < low / rows < 0.3, low / rows


def test_cap_drop_breaks_slot_identity():
    """A player who comes back after a drop takes the lowest free slot, which may not be his own.

    23 players (the 22 of cap_match and a 12th home player, column 22, who comes on at row 12 of half 1), all tracked
    throughout, and the ball at x = 29/32, so the players farthest from it are the lowest columns. The count is 16 on
    rows 10-14 and 17 on rows 15-19 (both capped, so the run goes on while the count rises). Worked out by hand: on rows
    10 and 11, 22 are present and the 6 farthest, columns 0 to 5 (home, slots 0 to 5), are dropped; column 22 comes on
    at row 12 and is not kept (the 16 are all kept); on row 15 the count is 17, one more, and the nearest present player
    who is not kept comes in: column 22, standing at 23/32, nearer the ball than columns 0 to 5. His piece starts at row
    15 and takes the lowest free home slot: slot 0, which column 0 left on row 10. On row 20 the run ends and columns 0
    to 5 are back: column 0 finds slot 0 taken and takes the lowest free, slot 1; column 1 takes 2, column 2 takes 3,
    column 3 takes 4, column 4 takes 5 and column 5 takes 11 (6 to 10 are the other home players'). Everybody else keeps
    the slot he had (home 6 to 10 from the start: column j is slot j; away 11 to 21 are 24 to 34). The drops: 63
    slot-frames (6 on each of rows 10 and 11, 7 on each of rows 12 to 14, 6 on each of rows 15 to 19) in 10 capped rows,
    cutting 7 pieces."""
    squad = FULL_SQUAD + (0,)
    match = cap_match(lambda half, row: (29 / 32, 0.5), teams=squad, off={22: 12})
    view = degrade.tracking_view(match)
    start, stop = match.graded[0][1:]
    parameters = degrade.tracking_parameters([10 ** 9], 0.0)
    limit = limit_for(view, 1, [(10, 15, 16), (15, 20, 17)])
    chosen = degrade.choose_half(view, 1, parameters, 0, limit)
    by_player = {}
    for t in chosen.tracklets:
        by_player.setdefault(t.player, []).append((t.start - start, t.stop - start, t.slot))
    length = stop - start
    for player, slot in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 11)):
        assert by_player[player] == [(0, 10, player), (20, length, slot)], (player, by_player[player])
    assert by_player[22] == [(15, length, 0)], by_player[22]
    for player in range(6, 22):
        slot = player if player < 11 else 24 + player - 11
        assert by_player[player] == [(0, length, slot)], (player, by_player[player])
    assert chosen.counts["capped_frames"] == 10 and chosen.counts["cap_slot_frames"] == 2 * 6 + 3 * 7 + 5 * 6 == 63
    assert chosen.counts["cap_pieces_cut"] == 7
    # Without the cap the same players keep their slots and column 22 takes the lowest free one on row 12: slot 11.
    plain = degrade.choose_half(view, 1, parameters, 0)
    by_plain = {t.player: (t.start - start, t.slot) for t in plain.tracklets}
    assert by_plain[0] == (0, 0) and by_plain[22] == (12, 11) and len(plain.tracklets) == 23
    assert {name: plain.counts[name] for name in NO_CAP} == NO_CAP


def test_cap_limits_are_the_transplants_window_of_our_counts():
    """The cap reads present_counts through the same Transplant as the ball's mask: same source half, same offset.

    Counts that say where they come from (count j of a half is j), so the window is read off the values: for each seed
    and half, the limits are the source half's counts from the offset that the profile recorded for its ball (the
    "transplants" of gaps, as test_profiles_nest has them equal to tracking's), worked out one frame at a time, and
    where the source runs out they go on from the start of its last TAIL_FRAMES, as the mask does. One half is longer
    than the counts (so that it goes round), and the window is over the half's span from its kick-off."""
    match = steady_match(rows=(240, 400), teams=FULL_SQUAD, substitutions=False)
    view = degrade.tracking_view(match)
    base = made_up_calibration()
    counts = {half: np.arange(len(mask), dtype=np.int16) for half, mask in base.ball_masks.items()}
    cal = types.SimpleNamespace(**{**vars(base), "present_counts": counts})
    seen = set()
    for seed in range(8):
        limits = degrade.cap_limits(view, cal, seed)
        _, info = degrade.degrade(view, "gaps", cal, seed)
        assert sorted(limits) == [1, 2]
        for half, start, stop in view.graded:
            record = info["transplants"][half]
            source = counts[record["source_half"]]
            tail = min(degrade.TAIL_FRAMES, len(source))
            expected = []
            for j in range(stop - view.kickoffs[half]):
                index = record["offset"] + j
                if index >= len(source):
                    index = len(source) - tail + (index - len(source)) % tail
                expected.append(int(source[index]))
            assert limits[half].tolist() == expected, (seed, half)
            assert limits[half].dtype == np.int16 and not limits[half].flags.writeable
            assert record["frames"] == len(expected) == stop - view.kickoffs[half]
            seen.add((half, record["source_half"], record["offset"] + len(expected) > len(source)))
    assert {(2, 1, True), (2, 2, True)} <= seen, "the longer half goes round, from either source: " + str(sorted(seen))
    assert {(1, 1, False), (1, 2, False)} <= seen, str(sorted(seen))


def test_cap_reads_dfls_ball_and_not_the_degraded_one():
    """The ball that the cap stands for our camera is the clean one: a mask over the ball changes none of the players.

    The same counts and everything else, with the ball of our masks missing nowhere and missing everywhere: the degraded
    ball differs (and the Transplant is the same: the masks are as long as each other), the players are the same bytes.
    A cap that read the degraded ball would drop other players where it is missing."""
    match = steady_match(rows=(240, 200), teams=FULL_SQUAD, substitutions=True)
    view = degrade.tracking_view(match)
    base = made_up_calibration()
    nowhere = types.SimpleNamespace(**{**vars(base), "ball_masks": {h: np.zeros_like(m) for h, m in
                                                                    base.ball_masks.items()}})
    everywhere = types.SimpleNamespace(**{**vars(base), "ball_masks": {h: np.ones_like(m) for h, m in
                                                                       base.ball_masks.items()}})
    for seed in range(3):
        seen = degrade.degrade(view, "tracking", nowhere, seed)[0]
        hidden = degrade.degrade(view, "tracking", everywhere, seed)[0]
        assert seen.ball_present.sum() > 0 and hidden.ball_present.sum() == 0
        assert seen.players_xy.tobytes() == hidden.players_xy.tobytes()
        assert seen.players_speed.tobytes() == hidden.players_speed.tobytes()


def test_cap_limits_refuse_counts_that_are_not_the_masks_frames():
    """A Calibration's present_counts are the frames of its ball_masks: counts of another length are refused, by half.

    The cap and the ball read one window through one Transplant, so counts that are not the masks' frames would be read
    from frames that are not the ball's. The made-up Calibration's own counts are as long as its masks; a shorter or a
    longer array, and a half that is missing, are a ValueError that names the half and both lengths."""
    match = steady_match(rows=(240, 200), teams=FULL_SQUAD, substitutions=False)
    view = degrade.tracking_view(match)
    cal = made_up_calibration()
    assert {half: len(counts) for half, counts in degrade.cap_limits(view, cal, 0).items()} == {1: 240, 2: 200}
    short = types.SimpleNamespace(**{**vars(cal), "present_counts": {1: cal.present_counts[1][:300],
                                                                      2: cal.present_counts[2]}})
    expect_error(ValueError, lambda: degrade.cap_limits(view, short, 0), "half 1", "300", "330", "present_counts",
                 "ball_masks")
    long_ = types.SimpleNamespace(**{**vars(cal), "present_counts": {
        1: cal.present_counts[1], 2: np.concatenate([cal.present_counts[2], cal.present_counts[2]])}})
    expect_error(ValueError, lambda: degrade.cap_limits(view, long_, 0), "half 2", "620", "310")
    lacking = types.SimpleNamespace(**{**vars(cal), "present_counts": {1: cal.present_counts[1]}})
    expect_error(ValueError, lambda: degrade.cap_limits(view, lacking, 0), "half 2", "present_counts")
    expect_error(ValueError, lambda: degrade.degrade(view, "tracking", short, 0), "present_counts", "ball_masks")


def test_tracking_version_2_with_no_low_count_frame_is_version_1s_players():
    """Nesting: with no frame at or below CAP, the cap changes nothing, and the players are version 1's byte for byte.

    Counts all 22 (as long as the masks): the whole profile's players equal degraded_players without the cap, bytes of
    the positions and speeds, and so do its tracklets and the counts of the walk, the episodes and the NaN runs, on a
    full squad and on a steady match, under three seeds; and so do counts that are all just above CAP (19). The record
    says that nothing was capped. With the scale of the tracked spans at 1 and the shortest-piece rule off, set through
    the parameters (what version 1 was), the same holds for the players that those parameters make: a cap that finds no
    frame to act on is the walk, the episodes, the slots, the speeds and the smoothing of version 1 (with version 3's
    slot rule: a slot is free from the row after its tracklet stopped), and nothing else. The profile's version is 3 and
    the gaps, clean and context ones' are 1."""
    assert degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 3, "context": 1}
    for match in (steady_match(rows=(240, 200), teams=FULL_SQUAD, substitutions=True), steady_match()):
        view = degrade.tracking_view(match)
        for level in (22, 19):
            base = made_up_calibration()
            cal = types.SimpleNamespace(**{**vars(base), "present_counts": {
                half: np.full(len(mask), level, dtype=np.int16) for half, mask in base.ball_masks.items()}})
            parameters = degrade.calibrated_parameters(cal, view.fps)
            version_1 = degrade.calibrated_parameters(cal, view.fps, tracked_scale=1.0, min_piece=0)
            assert (version_1.tracked_scale, version_1.min_piece) == (1.0, 0)
            for seed in range(3):
                atd, info = degrade.degrade(view, "tracking", cal, seed)
                xy, speed, records, counts = degrade.degraded_players(view, parameters, seed)
                assert atd.players_xy.tobytes() == xy.tobytes() and atd.players_speed.tobytes() == speed.tobytes()
                assert info["version"] == 3 and info["players"]["capped_frames"] == 0
                assert info["players"]["cap_slot_frames"] == info["players"]["cap_pieces_cut"] == 0
                limits = degrade.cap_limits(view, cal, seed)
                for own in (parameters, version_1):
                    plain = degrade.degraded_players(view, own, seed)
                    again = degrade.degraded_players(view, own, seed, limits)
                    assert again[0].tobytes() == plain[0].tobytes() and again[1].tobytes() == plain[1].tobytes()
                    assert again[2] == plain[2] and again[3] == plain[3]


# -- the shortest piece, and the scale of the tracked spans -----------------------------------------------------------

def test_min_piece_rule_drops_every_piece_shorter_than_it():
    """A piece shorter than MIN_PIECE_FRAMES is dropped whatever made it, and counted; the others come back as they
    were.

    Hand-built pieces of a steady match with 6 players (home 0-2, away 3-5, so a piece in the pool of the other team is
    an episode piece). Player 0: 7 frames (dropped), 8 (kept: exactly the shortest) and 25 (kept). Player 3 (away): a
    one-frame piece (dropped). Player 1 (home) in the away pool, so an episode piece: 9 frames (kept) and 5 (dropped).
    Player 4 (away): 8 (kept) and 2 (dropped). So 4 pieces are dropped, holding 7 + 1 + 5 + 2 = 15 slot-frames, one of
    them an episode piece. A minimum of 0 is no rule: the pieces come back, with nothing counted. The rule is on the
    pieces it is handed, in the order it is handed them."""
    view = degrade.tracking_view(steady_match())
    pieces = sorted([(0, 0, 7, 0), (7, 0, 15, 0), (15, 0, 40, 0), (3, 3, 4, 1), (10, 1, 19, 1), (20, 1, 25, 1),
                     (30, 4, 38, 1), (38, 4, 40, 1)])
    assert degrade.MIN_PIECE_FRAMES == 8
    got, counts = degrade.drop_short_pieces(view, pieces, degrade.MIN_PIECE_FRAMES)
    assert got == [piece for piece in pieces if piece[2] - piece[0] >= 8] and len(got) == 4
    assert counts == {"short_pieces": 4, "short_slot_frames": 15, "short_episode_pieces": 1}, counts
    assert pieces == sorted(pieces) and len(pieces) == 8, "the input is left as it was"
    again, none = degrade.drop_short_pieces(view, pieces, 0)
    assert again == pieces and none == {"short_pieces": 0, "short_slot_frames": 0, "short_episode_pieces": 0}
    assert degrade.drop_short_pieces(view, pieces, 1)[0] == pieces, "a piece of one frame is not shorter than 1"
    assert degrade.drop_short_pieces(view, pieces, 26)[1]["short_pieces"] == 8, "the longest piece is 25 frames"
    expect_error(ValueError, lambda: degrade.drop_short_pieces(view, pieces, -1), "min_piece", "at least 0")
    expect_error(ValueError, lambda: degrade.drop_short_pieces(view, pieces, 2.5), "min_piece")
    expect_error(ValueError, lambda: degrade.tracking_parameters([30], 0.1, min_piece=-3), "min_piece")
    assert degrade.tracking_parameters([30], 0.1).min_piece == 0, "none by default: version 1's"


def test_min_piece_rule_runs_after_the_cap_and_before_the_slots():
    """The pieces that the walk, an episode or the cap leave short are not tracklets: the player is absent there.

    23 players (the 22 of cap_match and column 22, home, who comes on at row 35 of 40: a piece of 5 frames that the walk
    makes), all tracked, the ball at x = 10/32, so the four farthest are columns 18 to 21, and the count 18 on rows
    10-14 and 20-24 and above CAP between them: those four are dropped on rows 10-14, back on rows 15-19 (5 frames: a
    piece the cap makes, the flicker of a count that is 18, 19, 18) and dropped again on rows 20-24. With the rule on,
    in choose_half, columns 18 to 21 have the pieces before and after only (rows 0-9 and 25-39), column 22 has none, and
    the record counts 5 pieces of 5 slot-frames each, none of them an episode piece; with the rule off, the middle
    pieces and column 22's are tracklets. The rule is after the cap (it sees the cap's pieces: that is why the five are
    counted) and before the slots (a piece it drops takes no slot: column 22's would have been slot 11)."""
    squad = FULL_SQUAD + (0,)
    match = cap_match(lambda half, row: (10 / 32, 0.5), teams=squad, off={22: 35})
    view = degrade.tracking_view(match)
    start, stop = match.graded[0][1:]
    limit = limit_for(view, 1, [(10, 15, 18), (20, 25, 18)])
    on = degrade.tracking_parameters([10 ** 9], 0.0, min_piece=8)
    off = degrade.tracking_parameters([10 ** 9], 0.0)
    kept = degrade.choose_half(view, 1, on, 0, limit)
    plain = degrade.choose_half(view, 1, off, 0, limit)
    spans = {}
    for t in kept.tracklets:
        spans.setdefault(t.player, []).append((t.start - start, t.stop - start))
    for player in range(18, 22):
        assert spans[player] == [(0, 10), (25, 40)], (player, spans[player])
    assert 22 not in spans and all(spans[player] == [(0, 40)] for player in range(18))
    assert {name: kept.counts[name] for name in ("short_pieces", "short_slot_frames", "short_episode_pieces")} == {
        "short_pieces": 5, "short_slot_frames": 25, "short_episode_pieces": 0}, kept.counts
    assert kept.counts["capped_frames"] == 10 and kept.counts["cap_slot_frames"] == 40
    middle = {t.player: (t.start - start, t.stop - start, t.slot) for t in plain.tracklets if t.start - start == 15}
    assert sorted(middle) == [18, 19, 20, 21] and all(stop_ == 20 for _, stop_, _ in middle.values())
    assert {t.player: t.slot for t in plain.tracklets if t.start - start == 35} == {22: 11}, "slot 11 without the rule"
    assert len(plain.tracklets) == len(kept.tracklets) + 5
    assert plain.counts["short_pieces"] == 0 and not any(t.slot == 11 for t in kept.tracklets), "it took no slot"


def test_min_piece_rule_counts_the_episode_pieces():
    """An episode piece that is shorter than the minimum is dropped and counted as one, in the half's record.

    Six players who stand far apart except through half 1, where a home and an away player are 1 m apart (pairs_match),
    nobody lost, and wrong-team episodes of 3 frames only (the shortest that the rule can see): every episode is a piece
    of at most 3 frames in the other pool, and so a short one. The pieces that are short without the rule (the episodes,
    and now and then the piece that a second onset leaves before or after one) are worked out from the tracklets of the
    same draws without the rule, one at a time: with the rule on, those are the pieces dropped, their frames are the
    slot-frames counted, those of them in the other team's pool are the episode pieces, and every other piece is as it
    was. Nothing else about the draws moves, so the episodes are the same."""
    match = pairs_match([(1, 0, 1000, [(1, 4, 1 / 128)])])
    view = degrade.tracking_view(match)
    on = episode_parameters(share=0.06, episodes=(3,), min_piece=8)
    off = episode_parameters(share=0.06, episodes=(3,))
    dropped = episodes = 0
    for seed in range(4):
        _, _, records, counts = degrade.degraded_players(view, on, seed)
        for half, start, stop in match.graded:
            plain = degrade.choose_half(view, half, off, seed)
            short = [t for t in plain.tracklets if t.stop - t.start < degrade.MIN_PIECE_FRAMES]
            assert counts[half]["episodes"] == plain.counts["episodes"]
            assert counts[half]["short_pieces"] == len(short)
            assert counts[half]["short_slot_frames"] == sum(t.stop - t.start for t in short)
            assert counts[half]["short_episode_pieces"] == sum(1 for t in short if is_carried(t, view))
            assert {(t.player, t.start, t.stop) for t in records[half]} == {
                (t.player, t.start, t.stop) for t in plain.tracklets if t.stop - t.start >= 8}
            dropped, episodes = dropped + len(short), episodes + counts[half]["short_episode_pieces"]
    assert dropped > 20 and episodes > 15, (dropped, episodes)


def test_tracked_scale_multiplies_the_drawn_lengths_and_the_lost_share_stays_d():
    """The walk's tracked lengths are multiplied by lambda (rounded, at least 1) and the lost spans follow, so d holds.

    The made-up Calibration's tracked spans are 10, 25, 50, 80 and 100 frames at 25 Hz; at lambda 1.5 they are 15, 38
    (37.5: half to even), 75, 120 and 150, at 2 they are 20, 50, 100, 160 and 200, and at 0.5 they are 5, 12 (12.5), 25,
    40 and 50; a length is at least one frame (spans of 0.01 s and 0.5 s at lambda 0.04 are 1 frame each). d is the
    Calibration's and does not move; the mean lost span is mu = mean tracked * d / (1 - d) of the scaled lengths.
    Through the walk on a long steady match (with the shortest-piece rule off: it would drop the spans of 5 frames) the
    share of the frames that are lost is d to within 0.01 for each lambda (the same tolerance as
    test_lost_share_on_long_synthetic). The profile's own parameters carry the frozen constant, and a lambda that is not
    a positive finite number is refused, naming it."""
    cal = made_up_calibration()
    one = degrade.calibrated_parameters(cal, 25, tracked_scale=1.0)
    assert one.lengths.tolist() == [10, 25, 50, 80, 100] and one.tracked_scale == 1.0
    match = steady_match(rows=(20000, 20000))
    view = degrade.tracking_view(match)
    on_pitch = int(np.isfinite(match.players_xy[:, :, 0]).sum())
    for scale, expected in ((1.5, [15, 38, 75, 120, 150]), (2.0, [20, 50, 100, 160, 200]), (0.5, [5, 12, 25, 40, 50])):
        parameters = degrade.calibrated_parameters(cal, 25, tracked_scale=scale)
        assert parameters.lengths.tolist() == expected and parameters.lengths.dtype == np.int64, scale
        assert parameters.tracked_scale == scale and parameters.d == one.d, "d does not move"
        mean = sum(expected) / len(expected)
        assert parameters.mean_tracked == mean and abs(parameters.mu - mean * one.d / (1 - one.d)) < 1e-9
        assert abs(parameters.mu / one.mu - mean / one.mean_tracked) < 1e-9, "the lost spans follow the tracked ones"
        bare = dataclasses.replace(parameters, scale=0.0, speed_nan_share=0.0, min_piece=0)     # d alone: no rule
        xy, _, _, _ = degrade.degraded_players(view, bare, 0)
        lost = 1 - int(np.isfinite(xy[:, :, 0]).sum()) / on_pitch
        assert abs(lost - one.d) < 0.01, (scale, lost, one.d)
    nobody = without_new_parts(cal, tracklet_lengths_s=np.array([0.01, 0.5, 40.0]), present_counts={
        half: np.full(len(mask), 22, dtype=np.int16) for half, mask in cal.ball_masks.items()})
    assert degrade.calibrated_parameters(nobody, 25, tracked_scale=0.04).lengths.tolist() == [1, 1, 40]
    profile = degrade.calibrated_parameters(cal, 25)
    assert profile.tracked_scale == degrade.TRACKED_SCALE and profile.min_piece == degrade.MIN_PIECE_FRAMES
    expected = np.maximum(1, np.rint(np.array([10, 25, 50, 80, 100]) * degrade.TRACKED_SCALE)).astype(int)
    assert profile.lengths.tolist() == expected.tolist()
    for bad in (0, -1.0, float("nan"), float("inf"), True, "2"):
        expect_error(ValueError, lambda: degrade.calibrated_parameters(cal, 25, tracked_scale=bad), "tracked_scale")
    expect_error(ValueError, lambda: degrade.calibrated_parameters(cal, 25, min_piece=-1), "min_piece")


# -- the order of the parts under the cap -----------------------------------------------------------------------------

def moving_cap_match(ball, rows=(60, 40)):
    """cap_match's 22 players on their line, each wobbling in y by an irregular amount of at most 0.27 m, so that the
    smoothing and the speeds have something to say; the ball is `ball` (cap_match's). The wobble of a player on a row is
    0.004 * (((7 * row + 3 * player) % 11) - 5) / 5 of the pitch's width."""
    base = cap_match(ball, rows=rows)
    xy = np.array(base.players_xy)
    for _, start, stop in base.graded:
        for player in range(xy.shape[1]):
            for row in range(stop - start):
                xy[start + row, player, 1] = 0.5 + 0.004 * (((7 * row + 3 * player) % 11) - 5) / 5
    xy.setflags(write=False)
    return dataclasses.replace(base, players_xy=xy)


def test_cap_cuts_the_smoothing_and_the_speeds_at_the_drops():
    """A piece that the cap cuts is smoothed, and has its speed taken, on its own rows: nothing crosses a drop.

    22 players on a line, each wobbling in y (moving_cap_match), the ball on column 9, and a walk with nobody lost (d =
    0: a player is one span for the whole half, so every cut is the cap's). Our counts are 16 on rows 10-19, 17 on rows
    30-35, 14 on rows 45-49 and 10 on rows 50-54 of half 1, and 15 on rows 5-11 of half 2: the farthest players are
    dropped on those rows and come back as new pieces. Every tracklet of the profile is a piece of the cap as the
    reference (reference_cap, one row at a time) has it; on its rows the positions of its slot are the 5-frame centred
    moving average of the piece's own rows, the window shrinking to them at the drop (smoothed_plain), and its speed is
    NaN on its first row, which for a piece that follows a drop is the row after it, and the step from the row before,
    in m/s, on the others. The scene is not blind to a crossing: for every piece that follows a drop, the average of the
    whole uncapped half differs from the piece's on the first rows after the drop, and a speed taken across the drop
    would be finite where this one is NaN. (A smoothing and a speed taken over the uncapped piece and cut to the rows
    the cap leaves fail it; so does anything that moves them to before the cap.)"""
    match = moving_cap_match(lambda half, row: (10 / 32, 0.5))
    view = degrade.tracking_view(match)
    parameters = degrade.tracking_parameters([10 ** 9], 0.0, smooth=5)
    runs = {1: [(10, 20, 16), (30, 36, 17), (45, 50, 14), (50, 55, 10)], 2: [(5, 12, 15)]}
    limits = {half: limit_for(view, half, runs[half]) for half in (1, 2)}
    xy, speed, records, counts = degrade.degraded_players(view, parameters, 0, limits)
    length, width = view.pitch
    cuts = 0
    for half, start, stop in view.graded:
        expected = reference_cap(view, half, whole_halves(view, half), limits[half])[0]
        assert sorted((t.start, t.player, t.stop) for t in records[half]) == [(f, p, l) for f, p, l, _ in expected]
        assert counts[half]["capped_frames"] == sum(last - first for first, last, _ in runs[half])
        for t in records[half]:
            rows = view.players_xy[t.start:t.stop, t.player].tolist()
            piece = smoothed_plain(rows, 5)
            assert np.allclose(xy[t.start:t.stop, t.slot], piece, rtol=1e-12, atol=1e-12), (half, t)
            assert math.isnan(speed[t.start, t.slot]), "the first row of a piece has no step: {}".format((half, t))
            for row in range(t.start + 1, t.stop):
                dx = (xy[row, t.slot, 0] - xy[row - 1, t.slot, 0]) * length
                dy = (xy[row, t.slot, 1] - xy[row - 1, t.slot, 1]) * width
                assert abs(speed[row, t.slot] - math.hypot(dx, dy) * view.fps) < 1e-9, (half, t, row)
            if t.start > start:
                cuts += 1
                whole = smoothed_plain(view.players_xy[start:stop, t.player].tolist(), 5)
                assert not np.allclose(whole[t.start - start:t.start - start + 2], piece[:2], atol=1e-6), (
                    "a crossing of the drop would show in this piece's first rows", half, t)
    assert cuts >= 20, cuts


def test_cap_is_applied_to_the_pieces_after_the_episodes_and_leaves_their_counts_alone():
    """The episodes are drawn and counted on the players as the walk has them, and the cap drops pieces from what they
    leave: an onset on a capped row is still an onset, and its episode still an episode.

    Six players (pairs_match): home 1 and away 4 are 1 m apart through rows 100-899 of half 1, so every onset in them
    finds that pair, and the ball is at the far corner, by away 5. Our count is 3 on those rows, so the three nearest
    players stay (away 5, away 3 and home 2) and the pair is dropped, both of them, from row 100 to row 899. With the
    cap after the episodes, the onsets, episodes, skips and episode frames are those of the same half with no cap, and
    those of the reference (reference_episodes, one onset at a time, over the pairs that the walk's spans hold); and
    the tracklets are the reference cap's (reference_cap) over the reference's pieces, each in the pool that carries it:
    an episode that starts on a capped row is counted and never shows, and what shows of one that runs past row 899 is
    the piece from row 900. A cap taken first would not leave the pair there to be found, and count fewer episodes. The
    scene has onsets to see this: over four seeds, more than 20 episodes and 40 onsets that fall on capped rows."""
    base = pairs_match([(1, 100, 900, [(1, 4, 1 / 128)])])
    ball = np.empty(base.ball_xy.shape)
    ball[:] = (0.875, 0.75)
    ball.setflags(write=False)
    match = dataclasses.replace(base, ball_xy=ball)
    view = degrade.tracking_view(match)
    parameters = episode_parameters()
    start, stop = match.graded[0][1:]
    limit = limit_for(view, 1, [(100, 900, 3)])
    keys = ("onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames")
    episodes = capped_onsets = 0
    for seed in range(4):
        free_pieces, free_counts = reference_episodes(view, 1, degrade.walk_spans(view, 1, parameters, seed),
                                                      parameters, seed)
        plain = degrade.choose_half(view, 1, parameters, seed)
        got = degrade.choose_half(view, 1, parameters, seed, limit)
        assert {key: plain.counts[key] for key in keys} == free_counts, (seed, plain.counts)
        assert {key: got.counts[key] for key in keys} == free_counts, (seed, got.counts, free_counts)
        shown, cap_counts = reference_cap(view, 1, free_pieces, limit)
        assert sorted((t.start, t.player, t.stop, int(t.slot >= degrade.HOME_SLOTS)) for t in got.tracklets) == shown
        assert {key: got.counts[key] for key in cap_counts} == cap_counts, (seed, got.counts, cap_counts)
        assert {t.start for t in got.tracklets if is_carried(t, view)} <= {900}, "an episode shows only past the cap"
        assert any(is_carried(t, view) for t in plain.tracklets) or not free_counts["episodes"]
        uniform = np.random.default_rng([seed, view.match_index, 2, 1]).random(stop - start)
        rate = parameters.onset_rate
        capped_onsets += sum(1 for row in range(100, 900) if uniform[row] < rate)
        episodes += free_counts["episodes"]
    assert episodes > 20 and capped_onsets > 40, (episodes, capped_onsets)


# -- the frozen scale of the tracked spans ----------------------------------------------------------------------------

def test_tracked_scale_is_the_frozen_one_and_follows_from_the_recorded_measurements():
    """TRACKED_SCALE is 2.072, and 2.072 is what the two recorded measurements give: 41.28 / 19.92 = 2.0723, to 3
    decimals (with version 2's slot rule it was 41.28 / 20.22 = 2.042, R004b, and the slot rule of version 3 moved the
    median, R004c: a bug finding that reopened the constant once). Only recorded constants are compared, none measured,
    so the default run pins the frozen constant: it moves with a change of the code that nobody meant to be one, or with
    a retune towards a gate, a WARN or a score (R004b says never), and a moved constant fails here and says which one.
    The profile reads it and the shortest piece when it is called (calibrated_parameters), and
    degrade.MIN_PIECE_FRAMES is A09's 8 frames."""
    assert abs(RECORDED_OURS_MEDIAN_S / RECORDED_MEDIAN_S - 2.0723) < 5e-5
    assert round(RECORDED_OURS_MEDIAN_S / RECORDED_MEDIAN_V2_S, 3) == 2.042, "what version 2's constant was"
    assert tracked_scale_from_the_record() == 2.072
    assert degrade.TRACKED_SCALE == tracked_scale_from_the_record() == 2.072, (
        "TRACKED_SCALE is frozen: it was worked out once for version 3, from 41.28 s and 19.92 s, and is never "
        "touched again", degrade.TRACKED_SCALE)
    assert degrade.MIN_PIECE_FRAMES == 8
    profile = degrade.calibrated_parameters(made_up_calibration(), 25)
    assert profile.tracked_scale == 2.072 and profile.min_piece == 8


def test_calibrated_parameters_read_the_scale_and_the_shortest_piece_when_called():
    """lambda and the shortest piece are read from degrade.TRACKED_SCALE and MIN_PIECE_FRAMES at the call, as the scale
    of the episodes and the smoothing window are from WRONG_TEAM_SCALE and SMOOTH_WINDOW: a default argument is bound
    when the module loads, and a constant patched afterwards would silently keep its old value.

    With the four constants patched, calibrated_parameters and the tracking profile (degrade, whose record says what
    its parameters were) take the patched ones; an argument still wins; and with the constants put back the profile's
    are the frozen ones again."""
    cal = made_up_calibration()
    view = degrade.tracking_view(steady_match())
    names = ("TRACKED_SCALE", "MIN_PIECE_FRAMES", "WRONG_TEAM_SCALE", "SMOOTH_WINDOW")
    saved = {name: getattr(degrade, name) for name in names}
    try:
        for name, value in zip(names, (1.5, 3, 1.0, 3)):
            setattr(degrade, name, value)
        patched = degrade.calibrated_parameters(cal, 25)
        assert (patched.tracked_scale, patched.min_piece, patched.scale, patched.smooth) == (1.5, 3, 1.0, 3)
        assert patched.lengths.tolist() == [15, 38, 75, 120, 150], "the patched lambda is applied to the lengths"
        explicit = degrade.calibrated_parameters(cal, 25, tracked_scale=1.0, min_piece=0)
        assert (explicit.tracked_scale, explicit.min_piece) == (1.0, 0), "an argument wins over the constant"
        _, info = degrade.degrade(view, "tracking", cal, 0)
        assert (info["players"]["tracked_scale"], info["players"]["min_piece_frames"]) == (1.5, 3)
        assert (info["players"]["scale"], info["players"]["smooth_window"]) == (1.0, 3)
    finally:
        for name, value in saved.items():
            setattr(degrade, name, value)
    again = degrade.calibrated_parameters(cal, 25)
    assert (again.tracked_scale, again.min_piece) == (saved["TRACKED_SCALE"], saved["MIN_PIECE_FRAMES"])
    assert (again.scale, again.smooth) == (saved["WRONG_TEAM_SCALE"], saved["SMOOTH_WINDOW"])
