"""Checks of touch_backdate_stats (pipeline/benchmark/touch_backdate_stats.py): step 0 of P28, the gate.

touch_backdate_stats decides whether P28 is built into a config at all: (i), the share of the truth's ball-visible
team flips that have a touch of the new team and no episode, and (ii), the share of the frames the rule would back-date
where the truth names the touching team, which is also the confidence the rule is given. A mistake in either would
pass or fail the gate wrongly. So what is held against it here is a plain loop, one frame at a time: the flips, the
windows, the episode starts and the stretches worked out again from the rule as the spec states it, on the source's
touches and episodes, the caps by cutting each stretch, the rates per match and seed, the gate and the floor; and (i)'s
windows on hand-built touches and episodes with every count written out. Also:
that the source is never handed the truth (the inputs of every call go through the deep walk of common.leaks), that the
file holds counts and shares and no per-frame array, what the command line prints last, and what is refused.

The matches are common.random_match's (random flags, so every figure is meaningless) with the tracking replaced by a
relay of the ball between standing players (relay_match), which is what the rule looks at: on tracking drawn at random
frame by frame almost nothing is a touch.
"""

import contextlib
import dataclasses
import io
import os
import tempfile

import numpy as np

from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import touch_backdate_stats, touch_stats, truth
from pipeline.benchmark.checks.common import leaks, loading, random_match, read, strict_json
from pipeline.contracts import RunContext
from pipeline.evidence import controlled_touch, kinematics
from pipeline.selftest import expect_error, quiet

# Params
RULE = {"radius": 2.0, "max_rel_speed": 4.0, "min_frames": 2}   # the control rule that the checks measure
DUEL = 1.0
TAUS = (3.0, 6.0)
PROFILE, SEEDS = "generic", 2            # the spec's profile, which needs no Calibration, and two seeds to pool
TEAMS = (0, 0, 0, 0, 1, 1, 1, 1)
WINDOW = 12
CAPS = (12, 25, 50)
H, A = 0, 1                              # the truth's codes for the two teams (possession_metrics)
# The counts of a row of (i), in the order the loop and _flip_counts hold them.
FLIP_KEYS = ("flips", "visible", "touch", "episode", "touch_no_episode", "inside_episode", "touch_no_episode_outside")


def relay_match(match_id="RND001", seed=0, rows=(3000, 2500)):
    """random_match with its tracking replaced by a ball relayed between standing players; the flags stay random.

    In each half the ball goes from station to station on straight legs of 6 to 29 frames and lies 0, 8 or 12 frames
    at each: a stay of 0 is a deflection (a kink in the ball's path, no control), a longer one a control. The station's
    player, any of the eight (four a team), stands 0.3 to 1.5 m from it from a few frames before the ball arrives to a
    few after it leaves; one station in seven has nobody. The ball is missing for 1 to 3 frames with chance 0.005 a
    frame. Positions are normalised by the pitch, as a DflMatch's are, and NaN in both coordinates where missing.
    """
    base = random_match(match_id, seed=seed, rows=rows, teams=TEAMS)
    rng = np.random.default_rng([28, seed])
    pitch = np.array(base.pitch)
    ball = np.full((base.n_frames, 2), np.nan)
    players = np.full((base.n_frames, len(TEAMS), 2), np.nan)
    for _, start, stop in base.graded:
        here, t = rng.uniform([20.0, 15.0], [85.0, 53.0]), start
        while t < stop:
            stay, flight = int(rng.choice([0, 0, 8, 12])), int(rng.integers(6, 30))
            there = np.clip(here + rng.uniform(-12.0, 12.0, 2), 2.0, pitch - 2.0)
            leg = here + (there - here) * np.arange(1, flight + 1)[:, None] / flight
            ball[t:min(t + stay + 1, stop)] = here
            ball[min(t + stay + 1, stop):min(t + stay + 1 + flight, stop)] = leg[:max(0, stop - (t + stay + 1))]
            low, high = max(start, t - int(rng.integers(2, 12))), min(stop, t + stay + 1 + int(rng.integers(2, 12)))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            if rng.random() > 1 / 7:
                players[low:high, int(rng.integers(len(TEAMS)))] = here + rng.uniform(0.3, 1.5) * np.array(
                    [np.cos(angle), np.sin(angle)])
            here, t = there, t + stay + flight
        for first in start + np.flatnonzero(rng.random(stop - start) < 0.005):
            ball[first:min(first + int(rng.integers(1, 4)), stop)] = np.nan
    ball, players = ball / pitch, players / pitch
    ball.setflags(write=False)
    players.setflags(write=False)
    return dataclasses.replace(base, ball_xy=ball, players_xy=players)


def relay_matches():
    """Two relay matches of different lengths; the ids are not DFL's, so nothing is cached."""
    return {"RND001": relay_match("RND001"), "RND002": relay_match("RND002", seed=1, rows=(3500, 2000))}


def measured(setting=None, **more):
    options = dict(taus=TAUS, duel_radius=DUEL, profile=PROFILE, seeds=SEEDS, matches=relay_matches(), log=quiet)
    options.update(more)
    return touch_backdate_stats.measure(RULE if setting is None else setting, **options)


def plain_loop(match, seed, tau):
    """One match and seed, one frame at a time: (i)'s seven counts, the stretches [(first, stop, team is home, [code,
    ...])] by the rule as the spec states it, and the frames with the ball visible. From the source's evidence without
    a hold, its touches and the truth."""
    atd = benchmark_run.make_atd(match, PROFILE, seed)
    inputs = match_inputs.MatchInputs(match, atd, profile=PROFILE, seed=seed)
    home = controlled_touch.compute(RunContext(inputs=inputs, seed=0), confidence=0.99, duel_radius=DUEL, **RULE).home
    touch_home, touch_away = controlled_touch.touches(inputs, radius=RULE["radius"], duel_radius=DUEL, tau=tau)
    visible = kinematics.ball_track(inputs).visible
    codes, _ = truth.truth_codes(match)
    blocks = kinematics.blocks(inputs)

    def starts_here(frame, start, team_is_home):
        """An episode of the team starts at `frame`: kept, the team's, and the frame before is not (or is the block's
        edge)."""
        if np.isnan(home[frame]) or (home[frame] > 0.5) != team_is_home:
            return False
        return frame == start or np.isnan(home[frame - 1]) or (home[frame - 1] > 0.5) != team_is_home

    flips = seen = with_touch = with_episode = both = inside_episode = outside = 0
    for start, stop in blocks:
        for c in range(start + 1, stop):
            if codes[c] not in (H, A) or codes[c - 1] not in (H, A) or codes[c] == codes[c - 1]:
                continue
            flips += 1
            if not visible[c]:
                continue
            seen += 1
            new_is_home = codes[c] == H
            window = range(max(c - WINDOW, start), min(c + WINDOW, stop - 1) + 1)
            touch = any((touch_home if new_is_home else touch_away)[f] for f in window)
            episode = any(starts_here(f, start, new_is_home) for f in window)
            inside = bool(not np.isnan(home[c]) and (home[c] > 0.5) == new_is_home)   # the new team's episode runs
            with_touch += touch
            with_episode += episode
            both += touch and not episode
            inside_episode += inside
            outside += touch and not episode and not inside

    stretches = []
    for start, stop in blocks:
        episodes = []                                         # [first frame, last frame, team is home]
        for frame in range(start, stop):
            if np.isnan(home[frame]):
                continue
            side = bool(home[frame] > 0.5)
            if episodes and episodes[-1][1] == frame - 1 and episodes[-1][2] == side:
                episodes[-1][1] = frame
            else:
                episodes.append([frame, frame, side])
        for (_, old_last, old), (first, _, new) in zip(episodes, episodes[1:]):
            if old == new:
                continue
            own, other = (touch_home, touch_away) if new else (touch_away, touch_home)
            for frame in range(old_last + 1, first):
                if own[frame] and all(visible[frame:first]) and not any(other[frame + 1:first]):
                    stretches.append((frame, first, new, [int(code) for code in codes[frame:first]]))
                    break
    return (flips, seen, with_touch, with_episode, both, inside_episode, outside), stretches, int(visible.sum())


def capped_counts(stretches, cap):
    """(frames, agreeing frames, [length, ...]) of `stretches` with each cut to its last `cap` frames (None: whole)."""
    frames = agree = 0
    lengths = []
    for _, _, team_is_home, codes in stretches:
        kept = codes if cap is None else codes[-cap:]
        frames += len(kept)
        agree += sum(code == (H if team_is_home else A) for code in kept)
        lengths.append(len(kept))
    return frames, agree, lengths


def stretch_row(frames, agree, lengths):
    p25, median, p75 = (float(value) for value in np.percentile(lengths, [25, 50, 75]))
    return {"frames": frames, "agree": agree, "share": agree / frames, "stretches": len(lengths), "length_p25": p25,
            "length_median": median, "length_p75": p75}


def test_touch_backdate_stats_matches_a_loop():
    """(i), (ii) under every cap, the rates and the gate equal a frame-by-frame loop, per match and pooled over two
    seeds, for both taus; and the loop meets every case it should."""
    matches = relay_matches()
    result = measured()
    assert result["profile"] == PROFILE and result["seeds"] == [0, 1] and result["matches"] == sorted(matches)
    assert result["truth"] == "raw" and result["window"] == WINDOW and result["caps"] == list(CAPS)
    assert result["setting"] == dict(RULE, duel_radius=DUEL) and list(result["taus"]) == ["3", "6"]
    for tau in TAUS:
        entry = result["taus"][touch_backdate_stats.tau_label(tau)]
        assert entry["tau"] == tau
        pooled_flips = np.zeros(7, dtype=int)
        pooled = {cap: [0, 0, []] for cap in (None,) + CAPS}
        rates = []
        for match_id, match in matches.items():
            match_flips = np.zeros(7, dtype=int)
            of_match = {cap: [0, 0, []] for cap in (None,) + CAPS}
            for seed in range(SEEDS):
                flips, stretches, visible = plain_loop(match, seed, tau)
                match_flips += flips
                for cap in (None,) + CAPS:
                    frames, agree, lengths = capped_counts(stretches, cap)
                    for total in (of_match[cap], pooled[cap]):
                        total[0] += frames
                        total[1] += agree
                        total[2] += lengths
                rate = len(stretches) / (visible / (60.0 * match.fps))
                assert entry["stretches_per_minute"]["per_run"][match_id][str(seed)] == rate, (match_id, seed)
                rates.append(rate)
            row = entry["flips"]["per_match"][match_id]
            assert [row[key] for key in FLIP_KEYS] == list(match_flips), (tau, match_id, row, match_flips)
            assert row["share"] == match_flips[4] / match_flips[1]
            assert row["share_outside"] == match_flips[6] / match_flips[1]
            for cap in (None,) + CAPS:
                got = entry["backdated"][touch_backdate_stats.cap_label(cap)]["per_match"][match_id]
                assert got == stretch_row(*of_match[cap]), (tau, match_id, cap, got)
            pooled_flips += match_flips

        row = entry["flips"]["pooled"]
        assert [row[key] for key in FLIP_KEYS] == list(pooled_flips), (tau, row, pooled_flips)
        assert row["share"] == pooled_flips[4] / pooled_flips[1]
        assert row["share_outside"] == pooled_flips[6] / pooled_flips[1] and len(row) == len(FLIP_KEYS) + 2
        assert 0 < pooled_flips[6] < pooled_flips[4] and pooled_flips[5] > 0, (
            "some flips with a touch and no episode start must lie inside a running episode, and not all", pooled_flips)
        # A loop that meets nothing, or one case only, proves little: every count is there and none is all.
        assert 0 < pooled_flips[4] < pooled_flips[2] < pooled_flips[1] < pooled_flips[0], pooled_flips
        assert 0 < pooled_flips[3] < pooled_flips[1] and pooled_flips[2] - pooled_flips[4] > 0, pooled_flips
        for cap in (None,) + CAPS:
            got = entry["backdated"][touch_backdate_stats.cap_label(cap)]["pooled"]
            assert got == stretch_row(*pooled[cap]), (tau, cap, got)
            assert 0 < got["agree"] < got["frames"] and got["stretches"] == len(pooled[None][2]) >= 20, (cap, got)
        whole = entry["backdated"]["uncapped"]["pooled"]
        assert (entry["backdated"]["cap_12"]["pooled"]["frames"] < entry["backdated"]["cap_25"]["pooled"]["frames"]
                < whole["frames"]), "the caps of 12 and 25 frames must each cut something"
        assert max(pooled[None][2]) > 12 and min(pooled[None][2]) < 12

        assert set(entry["stretches_per_minute"]["per_run"]) == set(matches)
        assert (entry["stretches_per_minute"]["min"], entry["stretches_per_minute"]["max"]) == (min(rates), max(rates))
        assert len(rates) == len(matches) * SEEDS and min(rates) > 0 and len(set(rates)) > 1, rates

        gate = entry["gate"]
        assert gate["flips_share"] == row["share"] and gate["agreement"] == whole["share"]
        assert (gate["flips_needed"], gate["agreement_needed"]) == (0.10, 0.70)
        assert gate["flips_pass"] == (row["share"] >= 0.10) and gate["agreement_pass"] == (whole["share"] >= 0.70)
        assert gate["passed"] == (gate["flips_pass"] and gate["agreement_pass"])
        assert entry["backdate_confidence"] == touch_backdate_stats.floored(whole["share"])
    assert result["taus"]["3"] != result["taus"]["6"], "the two taus must not measure the same thing"


def test_touch_backdate_stats_floor_and_labels():
    """floored is confidence_from_agreement's flooring without its clip; the labels of the taus and the caps."""
    floored = touch_backdate_stats.floored
    assert floored(0.8765) == 0.87 and floored(0.57) == 0.57 and floored(0.7) == 0.7 and floored(0.699999) == 0.69
    assert floored(0.43) == 0.43 and floored(0.0) == 0.0 and floored(1.0) == 1.0, "no clip to [0.51, 0.99]"
    for share in (0.51, 0.5999, 0.7, 0.8349, 0.99):
        assert floored(share) == touch_stats.confidence_from_agreement(share), share
    assert isinstance(floored(0.8765), float)
    for bad in (-0.1, 1.1, float("nan"), True, "0.9", None):
        expect_error(ValueError, lambda bad=bad: floored(bad), "touch_backdate_stats", "share")
    assert [touch_backdate_stats.tau_label(tau) for tau in (3.0, 6.0, 4.5)] == ["3", "6", "4.5"]
    assert [touch_backdate_stats.cap_label(cap) for cap in (None, 12, 50)] == ["uncapped", "cap_12", "cap_50"]
    assert touch_backdate_stats.DEFAULT_TAUS == (3.0, 6.0) and touch_backdate_stats.DEFAULT_PROFILE == "generic"
    assert touch_backdate_stats.CAPS == CAPS and touch_backdate_stats.WINDOW == WINDOW
    assert (touch_backdate_stats.NEEDED_FLIPS, touch_backdate_stats.NEEDED_AGREEMENT) == (0.10, 0.70)


def test_touch_backdate_stats_never_passes_truth():
    """The inputs of every call of the source, compute, touches and backdated_stretches alike, hold neither the match
    nor its truth."""
    matches = relay_matches()
    seen = []
    real_compute, real_touches = controlled_touch.compute, controlled_touch.touches
    real_stretches = controlled_touch.backdated_stretches

    def spy_stretches(inputs, **parameters):
        seen.append(("stretches", inputs))
        return real_stretches(inputs, **parameters)

    def spy_compute(run, **parameters):
        seen.append(("compute", run.inputs))
        return real_compute(run, **parameters)

    def spy_touches(inputs, **parameters):
        seen.append(("touches", inputs))
        return real_touches(inputs, **parameters)

    controlled_touch.compute, controlled_touch.touches = spy_compute, spy_touches
    controlled_touch.backdated_stretches = spy_stretches
    try:
        measured()
    finally:
        controlled_touch.compute, controlled_touch.touches = real_compute, real_touches
        controlled_touch.backdated_stretches = real_stretches
    runs = len(matches) * SEEDS
    assert [name for name, _ in seen].count("touches") == runs * len(TAUS), "one call of touches per run and tau"
    assert [name for name, _ in seen].count("stretches") == runs * len(TAUS), "and one of backdated_stretches"
    assert [name for name, _ in seen].count("compute") == runs * (1 + len(TAUS) * (1 + len(CAPS))), (
        "the episodes once, then the rule under no cap and under each cap, per tau")
    per_match = len(seen) // len(matches)
    for index, match in enumerate(matches.values()):
        served = {id(inputs): inputs for _, inputs in seen[index * per_match:(index + 1) * per_match]}
        assert len(served) == SEEDS, "one MatchInputs per match and seed serves every call"
        for inputs in served.values():
            assert leaks(inputs, match) == [], leaks(inputs, match)


def longest_list(document):
    """The length of the longest list anywhere in a JSON document, and the most keys any of its objects has."""
    if isinstance(document, dict):
        inner = [longest_list(value) for value in document.values()]
        return (max([0] + [lists for lists, _ in inner]), max([len(document)] + [keys for _, keys in inner]))
    if isinstance(document, list):
        inner = [longest_list(value) for value in document]
        return (max([len(document)] + [lists for lists, _ in inner]), max([0] + [keys for _, keys in inner]))
    return 0, 0


def test_touch_backdate_stats_command_line():
    """main() prints the tables, then for each tau the gate's line and the confidence, and exits 0 whether the gate
    passes or not; --out is the result as JSON, counts and shares and no per-frame array; an error is FAILED and 1."""
    def serve(match_id):
        return relay_match(match_id, seed=len(match_id) + int(match_id[-1] == "X"))

    with tempfile.TemporaryDirectory() as root, loading(serve):
        out_path = os.path.join(root, "step0", "p28.json")
        arguments = ["--radius", "2.0", "--max-rel-speed", "4.0", "--min-frames", "2", "--seeds", "2", "--matches",
                     "J03WOH", "J03WMX", "--data-dir", root, "--out", out_path]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = touch_backdate_stats.main(arguments)
        out = shown.getvalue()
        assert code == 0, out
        result = touch_backdate_stats.measure(RULE, seeds=2, match_ids=["J03WOH", "J03WMX"], data_dir=root, log=quiet)
        assert result["profile"] == "generic" and result["setting"]["duel_radius"] == 1.0, "the defaults"
        assert list(result["taus"]) == ["3", "6"]
        document = strict_json(read(out_path))
        assert document == benchmark_run.plain(result), "the file is the result measure() returns"
        lists, keys = longest_list(document)
        assert lists <= 3 and keys <= 9, "counts and shares only: no list or object as long as a half ({}, {})".format(
            lists, keys)

        lines = out.rstrip("\n").split("\n")
        assert lines[-6].startswith("gate tau=3: (i) ") and lines[-3].startswith("gate tau=6: (i) "), lines[-6:]
        for line, name in ((lines[-6], "3"), (lines[-3], "6")):
            gate = result["taus"][name]["gate"]
            assert line == "gate tau={}: (i) {:.6f} >= 0.10 {}; (ii) {:.6f} >= 0.70 {} -> {}".format(
                name, gate["flips_share"], "pass" if gate["flips_pass"] else "FAIL", gate["agreement"],
                "pass" if gate["agreement_pass"] else "FAIL", "PASS" if gate["passed"] else "FAIL"), line
        for line, name in ((lines[-5], "3"), (lines[-2], "6")):
            assert line == "not gating tau={}: (i) excluding flips inside a new-team episode {:.6f}".format(
                name, result["taus"][name]["flips"]["pooled"]["share_outside"]), line
        for line, name in ((lines[-4], "3"), (lines[-1], "6")):
            assert line.startswith("suggested backdate_confidence tau={}: {:.2f} ".format(
                name, result["taus"][name]["backdate_confidence"])), line
        assert "(i) excluding flips inside a new-team episode (not gating)" in out, "the table of it, per match"
        assert not result["taus"]["3"]["gate"]["passed"], "random flags agree with nothing: the gate fails, and exit 0"
        assert "written   " in out and "J03WMX" in out and "cap 50 pooled" in out, out

        # A file that cannot be written costs the file, not the measurement: the tables and the gate are printed, exit 1
        blocker = os.path.join(root, "a_file")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("in the way of a folder")
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = touch_backdate_stats.main(arguments[:-1] + [os.path.join(blocker, "p28.json")])
        unwritten = shown.getvalue().rstrip("\n").split("\n")
        assert code == 1 and unwritten[-6:] == lines[-6:] and unwritten[-7].startswith("FAILED    "), unwritten[-8:]
        assert "written   " not in shown.getvalue() and "cap 50 pooled" in shown.getvalue()

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = touch_backdate_stats.main(arguments[:1] + ["0"] + arguments[2:])        # a radius of 0
        assert code == 1 and "FAILED" in shown.getvalue() and "radius" in shown.getvalue(), shown.getvalue()
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = touch_backdate_stats.main(arguments + ["--profile", "no_such"])
        assert code == 1 and "FAILED" in shown.getvalue() and "no_such" in shown.getvalue(), shown.getvalue()


def test_touch_backdate_stats_gate_lines():
    """The gate's line as the spec declares it, on made-up results: pass and FAIL of each part, nothing measured, the
    share that does not gate on its own line, and 6 decimals, so that 0.69999 is not printed as 0.7000 beside a FAIL."""
    def entry(flips, agreement, outside):
        flips_pass = flips is not None and flips >= 0.10
        agreement_pass = agreement is not None and agreement >= 0.70
        return {"flips": {"pooled": {"share_outside": outside}},
                "gate": {"flips_share": flips, "flips_needed": 0.10, "flips_pass": flips_pass, "agreement": agreement,
                         "agreement_needed": 0.70, "agreement_pass": agreement_pass,
                         "passed": flips_pass and agreement_pass},
                "backdate_confidence": None if agreement is None else touch_backdate_stats.floored(agreement)}

    lines = touch_backdate_stats.gate_lines({"taus": {
        "3": entry(0.1234, 0.7412, 0.0876), "6": entry(0.0999, 0.7, 0.05), "9": entry(0.5, 0.69999, 0.5),
        "12": entry(None, None, None)}})
    assert len(lines) == 12, lines
    assert lines[0] == "gate tau=3: (i) 0.123400 >= 0.10 pass; (ii) 0.741200 >= 0.70 pass -> PASS", lines[0]
    assert lines[1] == "not gating tau=3: (i) excluding flips inside a new-team episode 0.087600", lines[1]
    assert lines[2].startswith("suggested backdate_confidence tau=3: 0.74 "), lines[2]
    assert lines[3] == "gate tau=6: (i) 0.099900 >= 0.10 FAIL; (ii) 0.700000 >= 0.70 pass -> FAIL", lines[3]
    assert lines[4] == "not gating tau=6: (i) excluding flips inside a new-team episode 0.050000", lines[4]
    assert lines[5].startswith("suggested backdate_confidence tau=6: 0.70 "), lines[5]
    assert lines[6] == "gate tau=9: (i) 0.500000 >= 0.10 pass; (ii) 0.699990 >= 0.70 FAIL -> FAIL", lines[6]
    assert lines[8].startswith("suggested backdate_confidence tau=9: 0.69 "), lines[8]
    assert lines[9] == "gate tau=12: (i) n/a >= 0.10 FAIL; (ii) n/a >= 0.70 FAIL -> FAIL", lines[9]
    assert lines[10] == "not gating tau=12: (i) excluding flips inside a new-team episode n/a", lines[10]
    assert lines[11].startswith("suggested backdate_confidence tau=12: n/a "), lines[11]


def test_touch_backdate_stats_flip_windows_are_pinned():
    """(i) on hand-built touches and episodes, every count written out: a touch at exactly 12 frames from the flip
    counts and at 13 does not, an episode start at exactly 12 frames takes the flip out and at 13 does not, the window
    stops at the half's edges, only the ball at the flip itself matters, and a flip inside a running episode of the new
    team is in (i) and not in the count beside it.

    Two halves, frames 0..59 and 60..119. A count is [flips, ball visible, touch, episode, touch and no episode, inside
    an episode, touch and no episode and not inside one]."""
    n = 120
    graded = [(1, 0, 60), (2, 60, 120)]

    def mask(frames):
        out = np.zeros(n, dtype=bool)
        out[list(frames)] = True
        return out

    def counted(flips, touch_home=(), touch_away=(), home_episodes=(), away_episodes=(), missing=()):
        """The seven counts for `flips`, [(frame, the new team is home)], with touches dated on the frames given, kept
        episodes on the half-open spans given (as the source's evidence has them) and no ball on `missing`."""
        evidence = np.full(n, np.nan)
        for spans, value in ((home_episodes, 0.99), (away_episodes, 1.0 - 0.99)):
            for first, stop in spans:
                evidence[first:stop] = value
        return touch_backdate_stats._flip_counts(
            [(frame, 0, n - 1, new_is_home) for frame, new_is_home in flips], graded, ~mask(missing),
            (mask(touch_home), mask(touch_away)), touch_backdate_stats._episode_starts(evidence, graded),
            touch_backdate_stats._episode_frames(evidence)).tolist()

    home_at_30 = [(30, True)]
    assert counted(home_at_30) == [1, 1, 0, 0, 0, 0, 0]
    # A touch of the new team 12 frames from the flip counts, 13 frames from it does not; the other team's never does.
    assert counted(home_at_30, touch_home=[42]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted(home_at_30, touch_home=[43]) == [1, 1, 0, 0, 0, 0, 0]
    assert counted(home_at_30, touch_home=[18]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted(home_at_30, touch_home=[17]) == [1, 1, 0, 0, 0, 0, 0]
    assert counted(home_at_30, touch_away=[30]) == [1, 1, 0, 0, 0, 0, 0]
    assert counted([(30, False)], touch_away=[42]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted([(30, False)], touch_home=[42]) == [1, 1, 0, 0, 0, 0, 0]
    # An episode of the new team that starts 12 frames from the flip takes it out, 13 frames from it does not.
    assert counted(home_at_30, touch_home=[30], home_episodes=[(42, 50)]) == [1, 1, 1, 1, 0, 0, 0]
    assert counted(home_at_30, touch_home=[30], home_episodes=[(43, 50)]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted(home_at_30, touch_home=[30], home_episodes=[(18, 20)]) == [1, 1, 1, 1, 0, 0, 0]
    assert counted(home_at_30, touch_home=[30], home_episodes=[(17, 20)]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted(home_at_30, touch_home=[30], away_episodes=[(28, 35)]) == [1, 1, 1, 0, 1, 0, 1]
    # An episode that follows the other team's without a gap starts on its own first frame.
    assert counted(home_at_30, touch_home=[30], away_episodes=[(20, 40)], home_episodes=[(40, 45)]) == [
        1, 1, 1, 1, 0, 0, 0]
    # The new team's episode already running at the flip, begun more than 12 frames before: (i) has the flip, the
    # count that does not gate leaves it out. The other team's episode running there changes nothing.
    assert counted(home_at_30, touch_home=[30], home_episodes=[(10, 40)]) == [1, 1, 1, 0, 1, 1, 0]
    assert counted(home_at_30, touch_home=[30], away_episodes=[(10, 40)]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted(home_at_30, home_episodes=[(10, 40)]) == [1, 1, 0, 0, 0, 1, 0]
    assert counted(home_at_30, home_episodes=[(30, 33)]) == [1, 1, 0, 1, 0, 1, 0]
    # The ball at the flip: without it the flip is counted and nothing else of it; the frames beside it do not matter.
    assert counted(home_at_30, touch_home=[30], missing=[30]) == [1, 0, 0, 0, 0, 0, 0]
    assert counted(home_at_30, touch_home=[30], missing=[29, 31]) == [1, 1, 1, 0, 1, 0, 1]
    # The window stops at the half's edges: 59 is the first half's last frame and 60 the second's first.
    assert counted([(65, True)], touch_home=[59]) == [1, 1, 0, 0, 0, 0, 0]
    assert counted([(65, True)], touch_home=[60]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted([(55, True)], touch_home=[60]) == [1, 1, 0, 0, 0, 0, 0]
    assert counted([(55, True)], touch_home=[59]) == [1, 1, 1, 0, 1, 0, 1]
    assert counted([(65, True)], touch_home=[65], home_episodes=[(58, 60)]) == [1, 1, 1, 0, 1, 0, 1]
    # An episode that runs over the half's edge starts again on the half's first frame.
    assert counted([(65, True)], touch_home=[65], home_episodes=[(40, 62)]) == [1, 1, 1, 1, 0, 0, 0]
    # Several flips are summed: one with a touch, one without the ball, one inside an episode that began 5 frames ago.
    assert counted([(30, True), (90, False), (100, True)], touch_home=[30], touch_away=[103],
                   home_episodes=[(95, 110)], missing=[90]) == [3, 2, 1, 1, 1, 1, 1]


def test_touch_backdate_stats_refuses_what_it_cannot_measure():
    """A setting without a key or with one too many, no tau, a tau that is no speed, a repeated tau, two taus of one
    label, an unknown profile and no seed are each refused, before any match is loaded."""
    def fails(match_id):
        raise AssertionError("a match was loaded before the arguments were checked")

    with loading(fails):
        for setting, message in (({"radius": 1.0}, "min_frames"), (dict(RULE, extra=1), "extra"), ([RULE], "keys")):
            expect_error(ValueError, lambda setting=setting: touch_backdate_stats.measure(setting, log=quiet),
                         "touch_backdate_stats", message)
        for taus, message in (((), "no tau"), ((0.0,), "tau"), ((float("inf"),), "tau"), ((True,), "tau"),
                              (("3",), "tau"), ((3.0, 3), "twice")):
            expect_error(ValueError, lambda taus=taus: touch_backdate_stats.measure(RULE, taus=taus, log=quiet),
                         "touch_backdate_stats", message)
        # Two taus that the result would file under one name: 3.0000001 is "3" at 6 significant digits.
        assert touch_backdate_stats.tau_label(3.0000001) == touch_backdate_stats.tau_label(3.0) == "3"
        expect_error(ValueError, lambda: touch_backdate_stats.measure(RULE, taus=(6.0, 3.0, 3.0000001), log=quiet),
                     "touch_backdate_stats", "3.0 and 3.0000001", "named 3")
        expect_error(benchmark_run.BenchmarkError,
                     lambda: touch_backdate_stats.measure(RULE, profile="no_such", log=quiet), "no_such")
        expect_error(benchmark_run.BenchmarkError, lambda: touch_backdate_stats.measure(RULE, seeds=0, log=quiet),
                     "seeds")
        expect_error(benchmark_run.BenchmarkError,
                     lambda: touch_backdate_stats.measure(RULE, match_ids=["NOPE"], log=quiet), "NOPE")
