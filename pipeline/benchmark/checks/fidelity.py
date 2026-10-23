"""Checks of the fidelity report (pipeline/benchmark/fidelity.py): the gates, the warnings and what is pooled how.

They run on two synthetic matches and the Calibration that calibration.measure makes of the synthetic bundle
(common.synthetic_calibration), so that every number in a report can be worked out here by other code: the pooled
shares from the degrader and calibration.ball_stats called match by match, the nearest player's agreement by the
frame-by-frame loop of the calibration's own checks, the yardsticks' outputs by the runner's own report. The real
fidelity of `gaps` is measured by the command on the downloaded matches and is R003's, not a check.
"""

import ast
import contextlib
import dataclasses
import io
import json
import os
import tempfile
import types

import numpy as np

from pipeline import run as pipeline_run
from pipeline.benchmark import calibration, degrade, dfl, fidelity, run, sources
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import truth as truths
from pipeline.benchmark.checks.common import (brute_force_agreement, context_calibration, loading, random_match,
                                              read, steady_match, strict_json, synthetic_calibration,
                                              synthetic_matches, tracking_calibration, yardstick_path)
from pipeline.contracts import D
from pipeline.selftest import expect_error, quiet

# Params
SEEDS = 2
WIDTHS = (0, 25, 50, 125)
SUBSET = ".m-RND001-RND002"
IDS = ("J03WMX", "J03WN1")
# The plan's measurements, which gates must not hold: the share missing, the median gap, the agreement, the dead-ball
# visibility, and the tracking profile's 14-or-fewer share, wrong-team share, 22-present share, mean present, tracklet
# median and NaN-speed share. They were measured on other spans than the kick-off ones, so a centre comes from the
# Calibration.
PLAN_FIGURES = {0.407, 0.4069, 10.7, 0.778, 0.66, 0.066, 0.108, 0.495, 20.5, 35.5, 0.0796, 0.080, 0.286}
# What the players' degradation draws that a report pools (written out here, not read from fidelity.PLAYERS_DRAWS, so
# that a count the report forgets to pool is seen): the walk and the episodes, the NaN runs, the cap and the
# shortest-piece rule.
DRAW_NAMES = ("tracklets", "onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames", "speed_nan_runs",
              "speed_nan_frames", "capped_frames", "cap_slot_frames", "cap_pieces_cut", "short_pieces",
              "short_slot_frames", "short_episode_pieces")
# What the context profile's ball draws that a report pools (written out here, not read from fidelity.BALL_DRAWS).
DRAW_BALL = ("rows", "ball_missing_rows", "nan_feature_rows", "capped_rows", "onsets", "missing_frames",
             "at_risk_rows", "post_gap_rows")
TRACKING_GATES = ["missing_share", "median_gap_s", "present_share_22", "present_mean", "present_share_le14",
                  "wrong_team_share", "tracklet_median_s", "speed_nan_share"]


def inside(measured, centre, band, kind):
    """The rule of a band, written out here: absolute is +- band, relative is +- band of the centre."""
    return abs(measured - centre) <= (band if kind == "absolute" else band * abs(centre))


def worked_out(matches, cal, seeds=SEEDS):
    """The pooled figures of a gaps run by other routes than the report's: shares, median gap, agreement."""
    missing = frames = judged = agreed = dead = visible = 0
    gaps = []
    for match in matches.values():
        codes_true = truths.truth_codes(match)[0]
        for seed in range(seeds):
            atd, _ = degrade.degrade(degrade.tracking_view(match), "gaps", cal, seed)
            inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed)
            spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
            stats = calibration.ball_stats(atd.ball_present, spans, inputs.metres(atd.ball_xy), inputs.pitch,
                                           inputs.fps)
            missing, frames = missing + stats["missing_frames"], frames + stats["frames"]
            gaps.append(stats["gap_durations_s"])
            share, count = brute_force_agreement(inputs, codes_true)
            agreed, judged = agreed + round(share * count), judged + count
            mask = codes_true == D
            dead, visible = dead + int(mask.sum()), visible + sum(1 for frame in np.flatnonzero(mask)
                                                                    if atd.ball_present[frame])
    return {"missing_share": missing / frames, "median_gap_s": float(np.median(np.concatenate(gaps))),
            "nearest_agreement": agreed / judged, "dead_ball_visible_share": visible / dead}


def test_fidelity_report_synthetic():
    """Every gate and check is there, each worked out by other code and judged by its own band; a FAIL is a FAIL."""
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = fidelity.report("gaps", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=lines.append)
        assert sorted(result) == sorted(["profile", "seeds", "matches", "calibration", "git", "gates", "checks",
                                         "statistics", "draws", "players_draws", "ball_draws", "full_strength",
                                         "passed", "warnings", "notes"])
        # The git state that the runner's reports hold (pipeline.run.git_state): which code made the figures.
        assert result["git"] == pipeline_run.git_state() and set(result["git"]) == {"commit", "dirty"}, result["git"]
        assert result["players_draws"] == {}, "gaps degrades no player"
        assert result["ball_draws"] == {}, "gaps places its gaps by a transplant, which draws no count of its own"
        assert result["profile"] == {"name": "gaps", "version": 1} and result["seeds"] == [0, 1]
        assert result["matches"] == ["RND001", "RND002"]
        assert result["calibration"] == {"sha256": calibration.digest(cal), "version": calibration.CALIBRATION_VERSION}
        assert len(lines) == 2 * SEEDS and all(line.startswith("fidelity RND00") for line in lines), lines

        ours = worked_out(matches, cal)
        # The gates: the Calibration's own centre, the plan's tolerance, the pooled number worked out here.
        assert [gate["name"] for gate in result["gates"]] == ["missing_share", "median_gap_s"]
        centres = {"missing_share": cal.scalars["missing_share"], "median_gap_s": float(np.median(cal.gap_durations_s))}
        assert [(gate["tolerance"]) for gate in result["gates"]] == [0.02, 1.5]
        for gate in result["gates"]:
            name = gate["name"]
            assert set(gate) == {"name", "text", "measured", "centre", "tolerance", "kind", "status"}
            assert gate["kind"] == "absolute"
            assert abs(gate["centre"] - centres[name]) < 1e-9, (name, gate, centres[name])
            assert abs(gate["measured"] - ours[name]) < 1e-9, (name, gate["measured"], ours[name])
            assert gate["status"] == ("pass" if abs(gate["measured"] - gate["centre"]) <= gate["tolerance"] else "FAIL")
        assert result["passed"] == all(gate["status"] == "pass" for gate in result["gates"])

        # The checks: the four kinds, the yardsticks in the order of the Calibration, each judged by its band.
        names = ["nearest_agreement", "dead_ball_visible_share"] + [
            "{}:nearest_w{}".format(kind, width) for width in WIDTHS for kind in ("possessions_per_min", "d_share")]
        assert [check["name"] for check in result["checks"]] == names
        by_name = {check["name"]: check for check in result["checks"]}
        window = cal.public_window
        assert by_name["nearest_agreement"]["centre"] == window["nearest_agreement"]
        assert by_name["dead_ball_visible_share"]["centre"] == window["dead_ball_visible_share"]
        assert abs(by_name["nearest_agreement"]["measured"] - ours["nearest_agreement"]) < 1e-9
        assert abs(by_name["dead_ball_visible_share"]["measured"] - ours["dead_ball_visible_share"]) < 1e-9
        bands = {"nearest_agreement": ("absolute", 0.05), "dead_ball_visible_share": ("absolute", 0.10)}
        bands.update({"possessions_per_min:nearest_w{}".format(width): ("relative", 0.25) for width in WIDTHS})
        bands.update({"d_share:nearest_w{}".format(width): ("absolute", 0.05) for width in WIDTHS})
        for check in result["checks"]:
            kind, band = bands[check["name"]]
            assert (check["kind"], check["band"]) == (kind, band), check
            assert set(check) == {"name", "measured", "centre", "band", "kind", "status"}
            ok = inside(check["measured"], check["centre"], band, kind)
            assert check["status"] == ("ok" if ok else "WARN"), check
        for width in WIDTHS:
            row = cal.baseline_outputs[width]
            assert by_name["possessions_per_min:nearest_w{}".format(width)]["centre"] == row["poss_per_min"]
            assert by_name["d_share:nearest_w{}".format(width)]["centre"] == row["d_share"]
        assert result["warnings"] == sum(1 for check in result["checks"] if check["status"] == "WARN")

        # The yardsticks' outputs against the runner's own reports of the same runs: possessions per graded minute and
        # the D share, from counts made by the scoring path instead of pipeline.output.summarise.
        scored = run.evaluate_configs([yardstick_path(width) for width in WIDTHS], profile="gaps", seeds=SEEDS,
                                      matches=matches, calibration=cal, data_dir=root, log=quiet)
        for width, entry in zip(WIDTHS, scored["configs"]):
            report = entry["report"]
            assert abs(by_name["possessions_per_min:nearest_w{}".format(width)]["measured"]
                       - report["possessions_per_graded_minute"]["predicted"]) < 1e-6, width
            assert abs(by_name["d_share:nearest_w{}".format(width)]["measured"]
                       - report["labels"]["predicted"]["D"]) < 1e-6, width

        # The statistics, as information: the benchmark's against ours, the same names on both sides.
        stats = result["statistics"]
        assert abs(stats["missing_share"]["benchmark"] - ours["missing_share"]) < 1e-9
        assert abs(stats["missing_share"]["bar_ath"] - cal.scalars["missing_share"]) < 1e-9
        assert abs(stats["gap_s_median"]["benchmark"] - ours["median_gap_s"]) < 1e-9
        assert stats["gap_s_median"]["bar_ath"] is not None and stats["present_mean"]["bar_ath"] == \
            cal.scalars["present_mean"]
        assert all(set(row) == {"benchmark", "bar_ath"} for row in stats.values())
        spread = [stats["missing_share_min_over_match_seeds"]["benchmark"],
                  stats["missing_share_max_over_match_seeds"]["benchmark"]]
        assert 0.0 <= spread[0] <= stats["missing_share"]["benchmark"] <= spread[1] <= 1.0

        # The draws: one per match, seed and half, the degrader's own, and nothing else of the match.
        assert len(result["draws"]) == 2 * SEEDS * 2
        for draw in result["draws"]:
            assert set(draw) == {"match", "seed", "half", "source_half", "offset", "frames"}
            info = degrade.degrade(degrade.tracking_view(matches[draw["match"]]), "gaps", cal, draw["seed"])[1]
            assert info["transplants"][draw["half"]] == {key: draw[key] for key in ("source_half", "offset", "frames")}

        # Aggregates only, JSON-ready, and the same twice.
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        assert all(len(value) <= 64 for value in leaves(result) if isinstance(value, list)), "no frame-sized list"
        again = fidelity.report("gaps", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert again == result

        # A FAIL: masks that put the ball where ours never was (or never where ours was) are not our gaps.
        centre = cal.scalars["missing_share"]
        shape = {half: mask.shape for half, mask in cal.ball_masks.items()}
        wrong = dataclasses.replace(cal, ball_masks={half: np.full(shape[half], centre < 0.5) for half in shape})
        failed = fidelity.report("gaps", seeds=1, matches=matches, calibration=wrong, data_dir=root, log=quiet)
        assert failed["gates"][0]["status"] == "FAIL" and failed["passed"] is False, failed["gates"]


def leaves(value):
    """Every value anywhere inside a JSON-ready structure, containers included."""
    found = [value]
    if isinstance(value, dict):
        for item in value.values():
            found += leaves(item)
    elif isinstance(value, list):
        for item in value:
            found += leaves(item)
    return found


# -- the context profile's gates --------------------------------------------------------------------------------------

def near_line_matches():
    """steady_matches() whose ball is 1.0 m from the touch line on alternating stretches of 20 rows (the second, the
    fourth, ...) and in the middle of the pitch in the others, so that about half of the gaps that start do so near a
    line, which the checks of the near-line gate need: the random ball of the other fixtures is never within 2 m."""
    out = {}
    for match_id, match in steady_matches().items():
        ball = np.array(match.ball_xy)
        for _, start, stop in match.graded:
            near = (np.arange(stop - start) // 20) % 2 == 1
            ball[start:stop, 0] = 0.5
            ball[start:stop, 1] = np.where(near, 1.0 / 68.0, 0.5)
        ball.setflags(write=False)
        out[match_id] = dataclasses.replace(match, ball_xy=ball)
    return out


def worked_out_near_line(matches, cal, seeds=SEEDS):
    """(near-line onsets, interior gaps) of the context profile on `matches`, counted one frame at a time here.

    A gap is a maximal run of rows without a ball inside a half from its kick-off, interior if a row with a ball comes
    before it and one after; its onset is the row before, and it is near a line if the ball there is within 2 m of one
    (strictly: min(x, L - x, y, W - y) < 2 on the match's pitch)."""
    near = interior = 0
    for match in matches.values():
        length, width = match.pitch
        for seed in range(seeds):
            atd, _ = degrade.degrade(degrade.tracking_view(match), "context", cal, seed)
            for half, _, stop in match.graded:
                first = match.kickoffs[half]
                row = first
                while row < stop:
                    if np.isnan(atd.ball_xy[row, 0]):
                        gap = row
                        while row < stop and np.isnan(atd.ball_xy[row, 0]):
                            row += 1
                        if gap > first and row < stop:
                            x, y = atd.ball_xy[gap - 1, 0] * length, atd.ball_xy[gap - 1, 1] * width
                            interior += 1
                            near += min(x, length - x, y, width - y) < 2.0
                    else:
                        row += 1
    return near, interior


def test_fidelity_context_gates_are_trackings_and_the_near_line_share():
    """The context profile's gates are the tracking profile's eight and the share of the gaps that start near a line.

    The centre is the Calibration's own near_line_share (measured on the kick-off spans: never the plan's 0.286, which
    the module must not hold), the tolerance +-0.05 and absolute, and the figure is pooled by sums, near-line onsets
    over interior gaps, and not a mean of the parts' shares: 1 of 4 and 9 of 12 are 10 of 16 = 0.625, where the mean of
    0.25 and 0.75 would be 0.5. A share of gaps that do not exist is NaN and fails, as an empty median does. The
    near-line gate is required, like every other: none is pending."""
    gates = {gate["name"]: gate for gate in fidelity.GATES["context"]}
    assert list(gates) == TRACKING_GATES + ["near_line_share"]
    assert fidelity.GATES["context"][:-1] == fidelity.GATES["tracking"]
    assert (gates["near_line_share"]["tolerance"], gates["near_line_share"]["kind"]) == (0.05, "absolute")
    assert "near_line_share" not in fidelity.PENDING_GATES and fidelity.PENDING_GATES == ()
    cal = synthetic_calibration()
    assert fidelity.centre_of("near_line_share", cal) == cal.scalars["near_line_share"]
    moved = dataclasses.replace(cal, scalars={**cal.scalars, "near_line_share": 0.58})
    assert fidelity.centre_of("near_line_share", moved) == 0.58, "another Calibration, another centre"
    pool = fidelity.Pool()
    for interior, near in ((4, 1), (12, 9)):
        pool.add_ball({"frames": 100, "missing_frames": 30, "gaps": interior, "interior_gaps": interior,
                       "near_line_onsets": near, "gap_durations_s": np.array([1.0]), "missing_share": 0.3})
    assert fidelity.measured("near_line_share", pool) == 10 / 16 == 0.625
    spec = gates["near_line_share"]
    assert fidelity._gate(spec, pool, moved)["status"] == "pass", "0.625 is 0.045 from 0.58"
    far = dataclasses.replace(cal, scalars={**cal.scalars, "near_line_share": 0.55})
    assert fidelity._gate(spec, pool, far)["status"] == "FAIL", "0.625 is 0.075 from 0.55"
    empty = fidelity.Pool()
    empty.add_ball({"frames": 100, "missing_frames": 0, "gaps": 0, "interior_gaps": 0, "near_line_onsets": 0,
                    "gap_durations_s": np.zeros(0), "missing_share": 0.0})
    assert np.isnan(fidelity.measured("near_line_share", empty))
    assert fidelity._gate(spec, empty, moved)["status"] == "FAIL", "no gap, no share, no pass"
    with open(fidelity.__file__, encoding="utf-8") as handle:
        held = {node.value for node in ast.walk(ast.parse(handle.read()))
                if isinstance(node, ast.Constant) and isinstance(node.value, float)}
    assert not held & PLAN_FIGURES, sorted(held & PLAN_FIGURES)


def test_fidelity_context_report_synthetic():
    """The context profile's nine gates, the near-line one counted frame by frame here, the players' figures tracking's,
    and the ball's draws pooled.

    Two steady matches of 22 players whose ball is near a line on alternating stretches, the made-up tracking figures
    and a made-up hazard of 0.03 a frame (common.context_calibration). The near-line gate is the pooled near-line onsets
    over interior gaps of worked_out_near_line, with the Calibration's centre and the tolerance of 0.05. Every other
    gate's centre is the Calibration's. The players are tracking's whatever the ball does, so the players' pooled draws
    and the six gates about them equal those of a tracking report on the same matches, seeds and Calibration, which a
    context report must not move; the two ball gates are not tracking's. "ball_draws" are the sums of the degrader's own
    records of the ball over the matches and seeds, with the shares that R005 quotes (the rows after a gap over the rows
    at risk, the rows with a NaN feature over the rows), and the scale. The report is JSON-ready, small and the same
    twice, and the printout names the draws and the near-line gate."""
    matches, cal = near_line_matches(), context_calibration(p=0.03)
    with tempfile.TemporaryDirectory() as root:
        result = fidelity.report("context", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        tracking = fidelity.report("tracking", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        again = fidelity.report("context", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
    assert again == result and json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["profile"] == {"name": "context", "version": 1, "tracking_version": 3}, "and the players' version"
    assert tracking["profile"] == {"name": "tracking", "version": 3} and result["matches"] == ["RND001", "RND002"]
    assert [gate["name"] for gate in result["gates"]] == TRACKING_GATES + ["near_line_share"]
    assert not any(gate["status"] == "pending" for gate in result["gates"])
    by_name = {gate["name"]: gate for gate in result["gates"]}
    near, interior = worked_out_near_line(matches, cal)
    assert interior > 10 and 0 < near < interior, (near, interior)
    gate = by_name["near_line_share"]
    assert set(gate) == {"name", "text", "measured", "centre", "tolerance", "kind", "status"}
    assert abs(gate["measured"] - near / interior) < 1e-9 and gate["centre"] == round(cal.scalars["near_line_share"], 9)
    assert (gate["tolerance"], gate["kind"]) == (0.05, "absolute")
    assert gate["status"] == ("pass" if abs(gate["measured"] - gate["centre"]) <= 0.05 else "FAIL")
    assert abs(result["statistics"]["near_line_onset_share"]["benchmark"] - near / interior) < 1e-9
    assert result["passed"] == (not any(gate["status"] == "FAIL" for gate in result["gates"]))
    # The players are tracking's: their gates and their pooled draws.
    tracking_gates = {gate["name"]: gate for gate in tracking["gates"]}
    for name in TRACKING_GATES[2:]:
        assert by_name[name] == tracking_gates[name], name
    assert result["players_draws"] == tracking["players_draws"] and result["draws"] == tracking["draws"]
    assert by_name["missing_share"]["measured"] != tracking_gates["missing_share"]["measured"], "another ball"
    # The ball's draws: the degrader's own records, added up, and the shares worked out from the sums.
    sums = {name: 0 for name in DRAW_BALL}
    for match in matches.values():
        for seed in range(SEEDS):
            ball = degrade.degrade(degrade.tracking_view(match), "context", cal, seed)[1]["ball"]
            for name in DRAW_BALL:
                sums[name] += ball[name]
    draws = result["ball_draws"]
    assert {name: draws[name] for name in DRAW_BALL} == sums and sums["onsets"] > 0 and sums["post_gap_rows"] > 0
    assert set(draws) == set(DRAW_BALL) | {"scale", "post_gap_share", "nan_feature_share"}
    assert draws["scale"] == degrade.HAZARD_SCALE
    assert abs(draws["post_gap_share"] - sums["post_gap_rows"] / sums["at_risk_rows"]) < 1e-9
    assert abs(draws["nan_feature_share"] - sums["nan_feature_rows"] / sums["rows"]) < 1e-9
    assert tracking["ball_draws"] == {} and set(fidelity.BALL_DRAWS) == set(DRAW_BALL)
    assert all(len(value) <= 64 for value in leaves(result) if isinstance(value, list)), "no frame-sized list"
    text = "\n".join(fidelity.render(result))
    for fragment in ("near_line_share", "ball draws", "post_gap_rows", "capped_rows", "gates all pass"
                     if result["passed"] else "DO NOT ALL PASS"):
        assert fragment in text, (fragment, text)


def test_a_pool_keeps_the_ball_scales_sorted_and_refuses_two():
    """The scales that the ball's draws were made with are a sorted tuple, not a set that is read by iterating it.

    One scale is the report's, whatever order the runs came in and however often it comes; two are a FidelityError
    that lists them in order (the sums would be of two profiles); a pool with no ball draws holds none and reports
    none."""
    def draws(scale):
        return {"ball": {**{name: 1 for name in fidelity.BALL_DRAWS}, "scale": scale}}

    pool = fidelity.Pool()
    for scale in (0.7, 0.5, 0.7, 0.5):
        pool.add_ball_draws(draws(scale))
    assert pool.ball_scales == (0.5, 0.7), pool.ball_scales
    expect_error(fidelity.FidelityError, pool.ball_draws_report, "2 scales", "0.5, 0.7")
    same = fidelity.Pool()
    for _ in range(3):
        same.add_ball_draws(draws(0.699))
    assert same.ball_scales == (0.699,) and same.ball_draws_report()["scale"] == 0.699
    empty = fidelity.Pool()
    empty.add_ball_draws({})
    assert empty.ball_scales == () and empty.ball_draws_report() == {}


def test_fidelity_pools_by_sums_and_lays_the_gaps_end_to_end():
    """Over matches and seeds, counts add and the statistic comes from the sums; the median is of every gap there was.

    Two (match, seed) parts of different sizes: 30 of 100 frames missing with gaps of 1 s and 3 s, and 30 of 300 missing
    with gaps of 10, 20 and 30 s. The share is 60 / 400 = 0.15, not the mean of 0.3 and 0.1, and the median gap is that
    of the five gaps together, 10 s: neither of the parts' own medians (2 s and 20 s) nor their mean.
    """
    pool = fidelity.Pool()
    for frames, missing, gaps in ((100, 30, [1.0, 3.0]), (300, 30, [10.0, 20.0, 30.0])):
        pool.add_ball({"frames": frames, "missing_frames": missing, "gaps": len(gaps), "interior_gaps": len(gaps),
                       "near_line_onsets": 1, "gap_durations_s": np.array(gaps), "missing_share": missing / frames})
    assert fidelity.measured("missing_share", pool) == 60 / 400 == 0.15
    assert fidelity.measured("median_gap_s", pool) == 10.0
    pool.add_agreement(100, 0.8)
    pool.add_agreement(300, 0.6)
    pool.add_agreement(0, None)
    assert (pool.agreed, pool.judged) == (80 + 180, 400)
    expect_error(KeyError, lambda: fidelity.measured("nothing", pool), "nothing")


def test_fidelity_pools_players_of_matches_with_different_squads():
    """A squad is 39 or 40 people in the DFL matches, so the histograms of slots present are not the same length.

    The first real run stopped on it (41 and 40 do not add): the histograms are added by the count they count, the
    longer one setting the length, and the shares are those of the sums.
    """
    pool = fidelity.Pool()

    def stats(frames, hist, **more):
        hist = np.array(hist, dtype=np.int64)
        row = {"frames": frames, "present_hist": hist, "present_slot_frames": int((hist * np.arange(hist.size)).sum()),
               "tracklet_lengths_s": np.array([5.0]), "wrong_team_frames": 0, "speed_nan_frames": 0,
               "jitter_triples": 0, "jitter_rms_m": float("nan")}
        row.update(more)
        return row

    pool.add_players(stats(10, [0, 0, 4, 6]))            # 4 frames with 2 present, 6 with 3: a 3-slot squad
    pool.add_players(stats(10, [1, 0, 0, 0, 9]))         # 1 frame with none, 9 with 4: a 4-slot squad
    pool.add_players(stats(5, [0, 5], jitter_triples=4, jitter_rms_m=2.0))
    assert pool.players["present_hist"].tolist() == [1, 5, 4, 6, 9]
    assert pool.players["frames"] == 25 and pool.players["present_slot_frames"] == (8 + 18) + 36 + 5
    assert pool.players["jitter_triples"] == 4 and abs(pool.players["jitter_squares"] - 16.0) < 1e-12
    cal = synthetic_calibration()
    # And through the statistics: the share with 22 present is 0 here (the longest histogram has 5 bins), not an error.
    pool.add_ball({"frames": 10, "missing_frames": 1, "gaps": 1, "interior_gaps": 1, "near_line_onsets": 0,
                   "gap_durations_s": np.array([1.0]), "missing_share": 0.1})
    rows = fidelity.statistics(pool, cal)
    assert rows["present_share_22"]["benchmark"] == 0.0 and rows["present_share_le14"]["benchmark"] == 1.0


def test_fidelity_gates_are_the_calibrations_values_with_the_plans_tolerances():
    """A gate's centre is read from the Calibration (never a figure of the plan), its tolerance is the plan's."""
    assert fidelity.GATES["gaps"][0]["name"] == "missing_share" and fidelity.GATES["gaps"][0]["tolerance"] == 0.02
    assert fidelity.GATES["gaps"][1]["name"] == "median_gap_s" and fidelity.GATES["gaps"][1]["tolerance"] == 1.5
    assert len(fidelity.GATES["gaps"]) == 2 and "clean" not in fidelity.GATES
    # The tracking profile's gates are gaps' two and six more, with the plan's tolerances; the tracklet median's is
    # relative (50% of the centre) and the rest are absolute. None is pending: the low-count tail is built (version 2).
    tracking = {gate["name"]: gate for gate in fidelity.GATES["tracking"]}
    assert list(tracking) == TRACKING_GATES and fidelity.GATES["tracking"][:2] == fidelity.GATES["gaps"]
    assert {name: (gate["tolerance"], gate["kind"]) for name, gate in tracking.items()} == {
        "missing_share": (0.02, "absolute"), "median_gap_s": (1.5, "absolute"), "present_share_22": (0.06, "absolute"),
        "present_mean": (0.5, "absolute"), "present_share_le14": (0.02, "absolute"),
        "wrong_team_share": (0.03, "absolute"), "tracklet_median_s": (0.5, "relative"),
        "speed_nan_share": (0.01, "absolute")}
    assert fidelity.PENDING_GATES == (), "every tracking gate counts since the tail is built"
    assert set(fidelity.COUNT_GATES) <= set(tracking) and fidelity.COUNT_GATES == (
        "present_share_22", "present_mean", "present_share_le14")
    cal = synthetic_calibration()
    for name, expected in (("present_share_22", cal.scalars["present_share_22"]),
                           ("present_mean", cal.scalars["present_mean"]),
                           ("present_share_le14", cal.scalars["present_share_le14"]),
                           ("wrong_team_share", cal.wrong_team_share),
                           ("tracklet_median_s", float(np.median(cal.tracklet_lengths_s))),
                           ("speed_nan_share", cal.speed_nan_share)):
        assert fidelity.centre_of(name, cal) == expected, name
    moved = dataclasses.replace(cal, scalars={**cal.scalars, "present_mean": 17.25, "present_share_22": 0.4,
                                              "present_share_le14": 0.3}, wrong_team_share=0.2, speed_nan_share=0.3,
                                tracklet_lengths_s=np.array([7.0, 9.0, 100.0]))
    assert [fidelity.centre_of(name, moved) for name in TRACKING_GATES[2:]] == [0.4, 17.25, 0.3, 0.2, 9.0, 0.3], \
        "another Calibration, other centres"
    assert fidelity.centre_of("missing_share", cal) == cal.scalars["missing_share"]
    assert fidelity.centre_of("median_gap_s", cal) == float(np.median(cal.gap_durations_s))
    # Another Calibration, another centre: nothing in the module holds the plan's 0.407 or 10.7 s.
    other = dataclasses.replace(cal, scalars={**cal.scalars, "missing_share": 0.123},
                                gap_durations_s=np.array([3.0, 5.0, 100.0]))
    assert fidelity.centre_of("missing_share", other) == 0.123 and fidelity.centre_of("median_gap_s", other) == 5.0
    with open(fidelity.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    held = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float)}
    assert not held & PLAN_FIGURES, "the module holds a figure of the plan: {}".format(sorted(held & PLAN_FIGURES))
    expect_error(KeyError, lambda: fidelity.centre_of("nothing", cal), "nothing")


TEAMS_22 = (0,) * 11 + (1,) * 11        # a full squad on the pitch: the frames of a match at full strength


def player_off(match, player, first, stop):
    """`match` with the player in column `player` off the pitch (no position) on rows first .. stop - 1 of its first
    half, counted from the half's start: a sending-off, or an injury, for part of a half."""
    start = match.graded[0][1]
    xy = np.array(match.players_xy)
    xy[start + first:start + stop, player] = np.nan
    xy.setflags(write=False)
    return dataclasses.replace(match, players_xy=xy)


def steady_matches():
    """Two matches of 22 players who stand on the pitch for whole stretches (checks/common.py's steady_match): the first
    has all of them throughout, and in the second an away player (column 15) is off for rows 100-179 of the first half,
    so that 80 of its 450 frames have 21 players and are below full strength."""
    return {"RND001": steady_match("RND001", 0, teams=TEAMS_22, substitutions=False),
            "RND002": player_off(steady_match("RND002", 1, rows=(260, 190), teams=TEAMS_22, substitutions=False),
                                 15, 100, 180)}


def worked_out_tracking(matches, cal, seeds=SEEDS):
    """The pooled figures of the tracking gates by other routes than the report's: each match and seed degraded and
    measured by calibration.player_stats on its own, the histograms and counts added, the tracklets laid end to end.

    The figures that the report measures on the full-strength frames only (the share with 22 present, the mean present,
    the share with 14 or fewer) and the wrong-team share there are counted again here one frame at a time, in plain
    loops, over the frames where the clean match has 22 players; they are under "full", with the number of those frames
    and, for each match with fewer than all of them, their share, under "share_at_full". The flat figures are over all
    frames."""
    frames = slot_frames = wrong = nan = 0
    hist = np.zeros(degrade.N_SLOTS + 2, dtype=np.int64)
    tracklets, drawn, below = [], {}, {}
    full = {"hist": np.zeros(degrade.N_SLOTS + 2, dtype=np.int64), "frames": 0, "slot_frames": 0, "wrong": 0}
    full_frames = total_frames = 0
    for match_id, match in matches.items():
        clean = [int(np.isfinite(match.players_xy[row, :, 0]).sum()) for row in range(match.n_frames)]
        spans = [(half, match.kickoffs[half], stop) for half, _, stop in match.graded]
        kept = sum(1 for _, start, stop in spans for row in range(start, stop) if clean[row] == 22)
        seen = sum(stop - start for _, start, stop in spans)
        full_frames, total_frames = full_frames + kept, total_frames + seen
        if kept < seen:
            below[match_id] = kept / seen
        for seed in range(seeds):
            atd, info = degrade.degrade(degrade.tracking_view(match), "tracking", cal, seed)
            inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=seed)
            stats = calibration.player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy),
                                             atd.players_speed, spans, inputs.fps)
            frames, slot_frames = frames + stats["frames"], slot_frames + stats["present_slot_frames"]
            wrong, nan = wrong + stats["wrong_team_frames"], nan + stats["speed_nan_frames"]
            hist[:stats["present_hist"].size] += stats["present_hist"]
            tracklets.append(stats["tracklet_lengths_s"])
            for name in DRAW_NAMES:
                drawn[name] = drawn.get(name, 0) + info["players"][name]
            for _, start, stop in spans:
                for row in range(start, stop):
                    if clean[row] != 22:
                        continue
                    present = [slot for slot in range(atd.n_slots) if atd.players_present[row, slot]]
                    home = sum(1 for slot in present if atd.slot_team[slot] == 0)
                    away = len(present) - home
                    full["hist"][len(present)] += 1
                    full["frames"] += 1
                    full["slot_frames"] += len(present)
                    full["wrong"] += (home > 11 and away < 11) or (away > 11 and home < 11)
    return {"present_share_22": hist[22] / frames, "present_mean": slot_frames / frames,
            "present_share_le14": hist[:15].sum() / frames, "wrong_team_share": wrong / frames,
            "tracklet_median_s": float(np.median(np.concatenate(tracklets))), "speed_nan_share": nan / slot_frames,
            "players_draws": drawn,
            "full": {"present_share_22": full["hist"][22] / full["frames"],
                     "present_mean": full["slot_frames"] / full["frames"],
                     "present_share_le14": full["hist"][:15].sum() / full["frames"],
                     "wrong_team_share": full["wrong"] / full["frames"], "frames": full["frames"]},
            "full_strength": {"frames": total_frames, "full_frames": full_frames, "share": full_frames / total_frames},
            "share_at_full": below}


def test_fidelity_tracking_report_synthetic():
    """The tracking profile's eight gates, each worked out by other code, every one of them counted.

    Two steady matches of 22 players (one with a player off for part of a half), and the made-up tracklets, episodes,
    counts and NaN runs: every pooled number is the sums of calibration.player_stats called on each degraded (match,
    seed) here and not by the report, the median is that of the tracklets laid end to end, the centres are the
    Calibration's, a relative tolerance is a share of its centre, and no gate is pending (PENDING_GATES is empty since
    version 2 has the low-count tail). The three gates about how many slots are present are measured on the
    full-strength frames only (the next check). The pooled counts of what was drawn, the cap's included, are the
    degrader's own."""
    matches, cal = steady_matches(), tracking_calibration()
    with tempfile.TemporaryDirectory() as root:
        result = fidelity.report("tracking", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert result["profile"] == {"name": "tracking", "version": 3} and result["matches"] == ["RND001", "RND002"]
        assert [gate["name"] for gate in result["gates"]] == TRACKING_GATES
        ours = worked_out_tracking(matches, cal)
        gaps = worked_out(matches, cal)
        centres = {"missing_share": cal.scalars["missing_share"], "median_gap_s": float(np.median(cal.gap_durations_s)),
                   "present_share_22": cal.scalars["present_share_22"], "present_mean": cal.scalars["present_mean"],
                   "present_share_le14": cal.scalars["present_share_le14"], "wrong_team_share": cal.wrong_team_share,
                   "tracklet_median_s": float(np.median(cal.tracklet_lengths_s)),
                   "speed_nan_share": cal.speed_nan_share}
        for gate in result["gates"]:
            name = gate["name"]
            assert set(gate) == {"name", "text", "measured", "centre", "tolerance", "kind", "status"}
            assert abs(gate["centre"] - centres[name]) < 1e-9, (name, gate)
            expected = (ours["full"][name] if name in fidelity.COUNT_GATES
                        else ours[name] if name in ours else gaps[name])
            if name in ("missing_share", "median_gap_s"):
                assert gate["status"] in ("pass", "FAIL")
            else:
                assert abs(gate["measured"] - expected) < 1e-9, (name, gate["measured"], expected)
            limit = gate["tolerance"] * (abs(gate["centre"]) if gate["kind"] == "relative" else 1)
            inside = abs(gate["measured"] - gate["centre"]) <= limit
            want = "pending" if name in fidelity.PENDING_GATES else ("pass" if inside else "FAIL")
            assert gate["status"] == want, (name, gate)
        assert not any(gate["status"] == "pending" for gate in result["gates"]), "none is pending"
        assert result["passed"] == (not any(gate["status"] == "FAIL" for gate in result["gates"]))
        assert not any("pending" in note.lower() for note in result["notes"]), result["notes"]
        # The ball is the gaps profile's, whatever the players do: the same two gate values.
        by_name = {gate["name"]: gate for gate in result["gates"]}
        assert abs(by_name["missing_share"]["measured"] - gaps["missing_share"]) < 1e-9
        assert abs(by_name["median_gap_s"]["measured"] - gaps["median_gap_s"]) < 1e-9
        # What was drawn, pooled: the degrader's own counts, none of them zero here.
        assert result["players_draws"] == ours["players_draws"], (result["players_draws"], ours["players_draws"])
        assert set(result["players_draws"]) == set(DRAW_NAMES) == set(fidelity.PLAYERS_DRAWS)
        assert result["players_draws"]["speed_nan_runs"] > 0 and result["players_draws"]["tracklets"] > 0
        assert all(result["players_draws"][name] > 0 for name in (
            "capped_frames", "cap_slot_frames", "cap_pieces_cut", "short_pieces", "short_slot_frames"))
        stats = result["statistics"]
        for name in ("present_share_22", "present_mean", "present_share_le14", "wrong_team_share", "speed_nan_share"):
            assert abs(stats[name]["benchmark"] - by_name[name]["measured"]) < 1e-9, name
            assert abs(stats[name]["bar_ath"] - by_name[name]["centre"]) < 1e-9, name
        assert abs(stats["tracklet_s_median"]["benchmark"] - by_name["tracklet_median_s"]["measured"]) < 1e-9
        # Aggregates only, JSON-ready and the same twice; a different calibration is a different digest, not a crash.
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        assert all(len(value) <= 64 for value in leaves(result) if isinstance(value, list)), "no frame-sized list"
        assert fidelity.report("tracking", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root,
                               log=quiet) == result
        text = "\n".join(fidelity.render(result))
        for fragment in ("tracklet_median_s", "50%", "present_mean", "players' draws", "episodes", "cap_slot_frames",
                         "full strength", "gates all pass" if result["passed"] else "DO NOT ALL PASS"):
            assert fragment in text, (fragment, text)
        assert "pending" not in text, text


def test_fidelity_count_gates_leave_out_frames_below_full_strength():
    """The gates on how many slots are present are measured on the frames where the clean match has 22 players, and
    nothing else is.

    In the second match an away player is off for 80 frames of the first half (a sending-off, or an injury): its 80
    frames have 21 players on the pitch and can never show 22 slots. The share with 22 present, the mean present and the
    share with 14 or fewer must be those of the other 370 frames of that match and all 450 of the first (counted here
    one frame at a time), and not those of all frames, which are lower (the check asserts that they differ). The
    wrong-team share, the tracklet median, the NaN-speed share and the ball's gates stay on all frames, and the
    statistics hold both: the count figures on all frames beside the full-strength ones, and the wrong-team share on the
    full-strength frames as information. The report says how many frames are at full strength, and which matches
    are not all of them."""
    matches, cal = steady_matches(), tracking_calibration()
    with tempfile.TemporaryDirectory() as root:
        result = fidelity.report("tracking", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
    ours = worked_out_tracking(matches, cal)
    by_name = {gate["name"]: gate for gate in result["gates"]}
    assert ours["share_at_full"] == {"RND002": 370 / 450} and set(ours["share_at_full"]) == {"RND002"}
    for name in fidelity.COUNT_GATES:
        assert abs(by_name[name]["measured"] - ours["full"][name]) < 1e-9, (name, by_name[name], ours["full"][name])
    assert abs(ours["full"]["present_mean"] - ours["present_mean"]) > 0.02, "all frames give another mean present"
    assert abs(ours["full"]["present_share_22"] - ours["present_share_22"]) > 0.005
    for name in ("wrong_team_share", "tracklet_median_s", "speed_nan_share"):
        assert abs(by_name[name]["measured"] - ours[name]) < 1e-9, (name, "stays on all frames")
    stats = result["statistics"]
    for name in fidelity.COUNT_GATES:
        assert abs(stats[name + "_all_frames"]["benchmark"] - ours[name]) < 1e-9, name
        assert abs(stats[name]["benchmark"] - ours["full"][name]) < 1e-9, name
        assert stats[name + "_all_frames"]["bar_ath"] == stats[name]["bar_ath"], "ours is on all its frames, in both"
    assert abs(stats["wrong_team_share_full_strength"]["benchmark"] - ours["full"]["wrong_team_share"]) < 1e-9
    assert abs(stats["wrong_team_share"]["benchmark"] - ours["wrong_team_share"]) < 1e-9
    assert stats["wrong_team_share_full_strength"]["bar_ath"] == stats["wrong_team_share"]["bar_ath"]
    seen = result["full_strength"]
    assert seen["frames"] == ours["full_strength"]["frames"] and seen["full_frames"] == \
        ours["full_strength"]["full_frames"], seen
    assert abs(seen["share"] - ours["full_strength"]["share"]) < 1e-9 and set(seen["share_at_full"]) == {"RND002"}
    assert abs(seen["share_at_full"]["RND002"] - 370 / 450) < 1e-9
    assert any("full strength" in note and "22" in note and "all its frames" in note for note in result["notes"]), \
        result["notes"]
    # The printout says which share it prints: the match's frames AT full strength, not those below it.
    printed = "\n".join(fidelity.render(result))
    line = [row for row in printed.splitlines() if "full strength frames only" in row]
    assert len(line) == 1 and "RND002 82.2%" in line[0] and "share of their frames at full strength" in line[0], printed
    assert "below full strength" not in printed, "370 of 450 frames at full strength is not 82.2% below it"

    # Matches that never have 22 players (these have 6) have no full-strength frame: no count gate can be measured, and
    # the report says so instead of dividing by zero; a gaps report on them has no such gate and says nothing of it.
    short = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        expect_error(fidelity.FidelityError, lambda: fidelity.report(
            "tracking", seeds=1, matches=short, calibration=cal, data_dir=root, log=quiet), "22", "full strength",
            "present_mean")
        gaps = fidelity.report("gaps", seeds=1, matches=short, calibration=cal, data_dir=root, log=quiet)
    assert gaps["statistics"]["present_mean"]["benchmark"] is None and gaps["statistics"]["present_mean"]["bar_ath"]
    assert gaps["statistics"]["present_mean_all_frames"]["benchmark"] is not None
    assert gaps["full_strength"]["full_frames"] == 0 and gaps["full_strength"]["share"] == 0.0


def test_fidelity_full_strength_blocks_are_the_runs_of_22_inside_each_span():
    """full_strength_blocks: the runs of rows with exactly 22 players, inside each span, with distinct block numbers.

    Worked out by hand on 12 rows: counts 22 on rows 0-3, 21 on 4-5, 22 on 6-8, 20 on 9, 22 on 10-11; the first half is
    rows 0-7 (from its kick-off at row 1) and the second rows 8-11. So the runs are rows 1-3 and 6-7 of the first half
    and rows 8 and 10-11 of the second: a run does not cross the end of a span, and a block number is used once (the
    statistics call refuses two blocks of one half)."""
    counts = np.array([22, 22, 22, 22, 21, 21, 22, 22, 22, 20, 22, 22])
    blocks = fidelity.full_strength_blocks(counts, [(1, 1, 8), (2, 8, 12)])
    assert [(start, stop) for _, start, stop in blocks] == [(1, 4), (6, 8), (8, 9), (10, 12)], blocks
    assert len({number for number, _, _ in blocks}) == 4
    assert fidelity.full_strength_blocks(np.full(6, 21), [(1, 0, 6)]) == []
    assert [(s, e) for _, s, e in fidelity.full_strength_blocks(np.full(6, 22), [(1, 0, 6)])] == [(0, 6)]
    assert [(s, e) for _, s, e in fidelity.full_strength_blocks(np.array([23, 22, 22, 23]), [(1, 0, 4)])] == [(1, 3)], \
        "exactly 22: more than 22 is not full strength either"


def test_fidelity_gate_kind_is_validated():
    """A gate spec's kind is absolute or relative, and anything else is a FidelityError naming the gate and the kinds.

    A kind that was not recognised used to count as relative, which would widen or narrow a band without a word."""
    cal = synthetic_calibration()
    pool = fidelity.Pool()
    pool.players["tracklets"].append(np.array([20.0]))
    spec = {"name": "tracklet_median_s", "tolerance": 0.5, "kind": "proportional", "text": "t"}
    expect_error(fidelity.FidelityError, lambda: fidelity._gate(spec, pool, cal), "tracklet_median_s", "proportional",
                 "absolute", "relative")
    missing = {"name": "tracklet_median_s", "tolerance": 0.5, "text": "t"}
    expect_error(fidelity.FidelityError, lambda: fidelity._gate(missing, pool, cal), "tracklet_median_s", "kind")
    assert fidelity.KINDS == ("absolute", "relative")
    assert all(spec["kind"] in fidelity.KINDS for gates in fidelity.GATES.values() for spec in gates)


def test_fidelity_check_band_kind_is_validated():
    """A warning check's band is (kind, width) with kind absolute or relative; anything else is a FidelityError.

    Like a gate's (above), an unknown kind used to count as relative, which widens or narrows a band without a word.
    The error names the check and the kinds; the four bands the report uses are valid; and a valid band still judges
    as it did: around 0.8 an absolute band of 0.05 holds 0.76 and not 0.74, and a relative one of 5% (0.04) holds 0.77
    and not 0.75."""
    expect_error(fidelity.FidelityError, lambda: fidelity._check("nearest_agreement", 0.7, 0.8, ("proportional", 0.1)),
                 "nearest_agreement", "proportional", "absolute", "relative")
    expect_error(fidelity.FidelityError, lambda: fidelity._check("d_share:nearest_w50", 0.7, 0.8, (None, 0.1)),
                 "d_share:nearest_w50", "None")
    bands = (fidelity.NEAREST_BAND, fidelity.DEAD_VISIBLE_BAND, fidelity.POSSESSIONS_BAND, fidelity.D_SHARE_BAND)
    assert all(kind in fidelity.KINDS for kind, _ in bands)
    assert fidelity._check("c", 0.74, 0.8, ("absolute", 0.05))["status"] == "WARN"
    assert fidelity._check("c", 0.76, 0.8, ("absolute", 0.05))["status"] == "ok"
    assert fidelity._check("c", 0.77, 0.8, ("relative", 0.05))["status"] == "ok"
    assert fidelity._check("c", 0.75, 0.8, ("relative", 0.05))["status"] == "WARN"


def test_fidelity_relative_gate_is_a_share_of_its_centre():
    """A relative tolerance is that share of the centre, inclusive, and the pooled median decides it.

    A Calibration whose tracklets have a median of 20 s: a pooled median of 10 s is exactly 50% below it, inside; one
    of 9.99 s is outside. The tracklets are laid end to end over the parts and not averaged: parts of 1, 10 and 100 s
    and of 2 and 10 s have the median of all five lengths, 10 s (neither part's own median, 10 s and 6 s, nor their
    mean). The same distance from a centre of 40 s is the edge again, so the width grows with the centre."""
    cal = dataclasses.replace(synthetic_calibration(), tracklet_lengths_s=np.array([5.0, 20.0, 80.0]))
    spec = next(gate for gate in fidelity.GATES["tracking"] if gate["name"] == "tracklet_median_s")
    pool = fidelity.Pool()
    pool.players["tracklets"] += [np.array([1.0, 10.0, 100.0]), np.array([2.0, 10.0])]
    assert fidelity.measured("tracklet_median_s", pool) == 10.0 and fidelity.centre_of("tracklet_median_s", cal) == 20.0
    gate = fidelity._gate(spec, pool, cal)
    assert (gate["status"], gate["kind"], gate["tolerance"]) == ("pass", "relative", 0.5), gate
    below = fidelity.Pool()
    below.players["tracklets"] += [np.array([1.0, 9.99, 100.0]), np.array([2.0, 9.99])]
    assert fidelity.measured("tracklet_median_s", below) == 9.99
    assert fidelity._gate(spec, below, cal)["status"] == "FAIL"
    wide = dataclasses.replace(cal, tracklet_lengths_s=np.array([40.0]))
    edge = fidelity.Pool()
    edge.players["tracklets"].append(np.array([20.0]))
    assert fidelity._gate(spec, edge, wide)["status"] == "pass", "20 s is 50% below 40 s: inside"
    edge.players["tracklets"][0] = np.array([19.0])
    assert fidelity._gate(spec, edge, wide)["status"] == "FAIL", "19 s is not"
    edge.players["tracklets"][0] = np.array([20.0])
    absolute = {**spec, "kind": "absolute", "tolerance": 10.0}
    assert fidelity._gate(absolute, pool, cal)["status"] == "pass", "10 s from 20 s is inside +-10 s"
    assert fidelity._gate(absolute, edge, wide)["status"] == "FAIL", "20 s from 40 s is not: the same width everywhere"


def test_fidelity_pending_gates_do_not_change_the_exit_code_and_a_failing_gate_does():
    """A pending gate is printed with its numbers and never counted; a required gate that misses fails the command.

    The machinery stays although no gate of the profile is pending since version 2 has its tail: a gate listed in
    PENDING_GATES (here the three about how many slots are present, which the check lists itself) is printed as
    pending, exit 0 whatever it would measure, and one that is not listed counts. The gates' tolerances are opened and
    closed on purpose: wide open, every gate passes and the three pending ones are printed as pending, exit 0; with the
    required NaN-speed gate closed to a tolerance of 0 the command exits 1 and says so; with only the pending gates
    closed (they would FAIL if they were counted) the exit code is 0 again. With the real, empty list the same closed
    gates are counted: exit 1 and nothing is called pending. An empty PENDING_GATES counts all eight."""
    cal = tracking_calibration()
    matches = {match_id: steady_match(match_id, seed=sources.MATCH_IDS.index(match_id), teams=TEAMS_22,
                                      substitutions=False) for match_id in IDS}
    real_load, real_load_match = calibration.load, dfl.load_match
    real_gates, real_pending = fidelity.GATES, fidelity.PENDING_GATES
    listed = ("present_share_22", "present_mean", "present_share_le14")
    assert real_pending == (), "the profile's own list is empty"

    def serve(match_id):
        return matches[match_id]

    def with_tolerance(**tolerances):
        return {**real_gates, "tracking": tuple(
            {**spec, "tolerance": tolerances.get(spec["name"], 1e9), "kind": "absolute"}
            for spec in real_gates["tracking"])}

    try:
        with tempfile.TemporaryDirectory() as root:
            def command(gates, pending=listed):
                calibration.load = lambda data_dir=None, **keywords: cal
                fidelity.GATES, fidelity.PENDING_GATES = gates, pending
                shown = io.StringIO()
                with contextlib.redirect_stdout(shown), loading(serve):
                    code = fidelity.main(["--profile", "tracking", "--seeds", "1", "--matches", *IDS,
                                          "--data-dir", root, "--no-write"])
                return code, shown.getvalue()

            code, out = command(with_tolerance())
            assert code == 0 and "all pass (3 pending, not counted)" in out and "pending" in out, out
            assert out.count("pending") >= 4, "each pending gate says it is, and the note says why"
            code, out = command(with_tolerance(speed_nan_share=0.0))
            assert code == 1 and "DO NOT ALL PASS" in out and "FAIL" in out, out
            code, out = command(with_tolerance(present_share_22=0.0, present_mean=0.0, present_share_le14=0.0))
            assert code == 0 and "all pass (3 pending, not counted)" in out, out
            code, out = command(with_tolerance(present_share_22=0.0))
            assert code == 0 and "all pass (3 pending, not counted)" in out, (
                "the share with 22 present is listed as pending: " + out)
            code, out = command(with_tolerance(present_share_22=0.0, present_mean=0.0, present_share_le14=0.0),
                                pending=real_pending)
            assert code == 1 and "DO NOT ALL PASS" in out and "pending" not in out, (
                "with the real list, which is empty, the same gates are counted: " + out)
            code, out = command(with_tolerance(), pending=real_pending)
            assert code == 0 and "all pass" in out and "pending" not in out, out
    finally:
        calibration.load, dfl.load_match = real_load, real_load_match
        fidelity.GATES, fidelity.PENDING_GATES = real_gates, real_pending


def test_fidelity_report_refuses_what_it_cannot_measure():
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        def measure(**keywords):
            keywords.setdefault("matches", matches)
            keywords.setdefault("calibration", cal)
            return fidelity.report(keywords.pop("profile", "gaps"), data_dir=root, log=quiet, **keywords)

        expect_error(fidelity.FidelityError, lambda: measure(profile="clean"), "clean", "no gates", "degrades nothing",
                     "context", "gaps", "tracking")
        expect_error(fidelity.FidelityError, lambda: measure(profile="nonsense"), "nonsense", "not one of", "clean",
                     "context", "gaps", "tracking")
        expect_error(fidelity.FidelityError, lambda: measure(seeds=0), "seeds", "at least 1")
        expect_error(fidelity.FidelityError, lambda: measure(seeds=True), "seeds")
        expect_error(fidelity.FidelityError, lambda: measure(calibration=object()), "calibration", "Calibration",
                     "calibration.load()")
        expect_error(fidelity.FidelityError, lambda: measure(calibration=types.SimpleNamespace(ball_masks={})),
                     "SimpleNamespace", "Calibration")
        expect_error(run.BenchmarkError, lambda: measure(match_ids=["RND009"]), "RND009", "RND001")
        expect_error(run.BenchmarkError, lambda: measure(match_ids=["RND001", "RND001"]), "RND001", "twice")
        wrong = {"RND001": synthetic_matches()["RND002"]}
        expect_error(run.BenchmarkError, lambda: measure(matches=wrong), "RND001", "RND002", "wrong id")


def test_fidelity_report_is_written_where_its_name_says_and_the_same_every_time():
    """fidelity_<profile>[.m-<ids>][.s<n>].json by the runner's rule; sorted, LF, strict JSON, atomic, the same."""
    assert os.path.basename(fidelity.report_path("d", "gaps")) == "fidelity_gaps.json"
    assert os.path.basename(fidelity.report_path("d", "gaps", None, 3)) == "fidelity_gaps.json"
    assert os.path.basename(fidelity.report_path("d", "gaps", None, 1)) == "fidelity_gaps.s1.json"
    assert os.path.basename(fidelity.report_path("d", "gaps", ["J03WN1", "J03WMX"], 5)) == \
        "fidelity_gaps.m-J03WMX-J03WN1.s5.json"
    assert os.path.basename(fidelity.report_path("d", "gaps", list(reversed(sources.MATCH_IDS)), 3)) == \
        "fidelity_gaps.json", "all seven in any order, the default seeds"
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        result = fidelity.report("gaps", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        path = fidelity.write(result, root)
        assert path == os.path.join(root, "runs", "benchmark", "fidelity_gaps" + SUBSET + ".s2.json"), path
        raw = read(path, "rb")
        assert raw.endswith(b"\n") and b"\r" not in raw
        assert strict_json(raw.decode("utf-8")) == result
        assert json.dumps(result, sort_keys=True, indent=2) + "\n" == raw.decode("utf-8"), "sorted, one format"
        again = fidelity.report("gaps", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert fidelity.write(again, root) == path and read(path, "rb") == raw, "a rewrite changes nothing"
        other = fidelity.write(fidelity.report("gaps", seeds=1, matches=matches, calibration=cal, data_dir=root,
                                               log=quiet), root)
        assert os.path.basename(other) == "fidelity_gaps" + SUBSET + ".s1.json" and read(path, "rb") == raw
        assert sorted(os.listdir(os.path.dirname(path))) == sorted(
            ["fidelity_gaps" + SUBSET + ".s1.json", "fidelity_gaps" + SUBSET + ".s2.json"]), "no .tmp left"
        text = "\n".join(fidelity.render(result))
        git = result["git"]
        for fragment in ("missing_share", "median_gap_s", "nearest_agreement", "dead_ball_visible_share",
                         "possessions_per_min:nearest_w50", "d_share:nearest_w125", "pass" if result["passed"]
                         else "FAIL", "calibration " + result["calibration"]["sha256"][:12], "half-draws",
                         "commit " + (git["commit"] or "?") + (" +dirty" if git["dirty"] else "")):
            assert fragment in text, (fragment, text)


def test_fidelity_command_exit_code_follows_the_gates():
    """0 when every gate passes, 1 when one fails, 1 and a sentence for what it cannot do; WARNs never fail the run."""
    cal = synthetic_calibration()
    # The command reads matches by their registry ids: two random ones under the ids of two DFL matches.
    matches = {match_id: random_match(match_id, seed=sources.MATCH_IDS.index(match_id)) for match_id in IDS}

    def serve(match_id):
        return matches[match_id]

    # A Calibration whose gate centres are what the benchmark measures: the gates pass exactly.
    with tempfile.TemporaryDirectory() as root:
        measured = {gate["name"]: gate["measured"] for gate in fidelity.report(
            "gaps", seeds=1, matches=matches, calibration=cal, data_dir=root, log=quiet)["gates"]}
    fitted = dataclasses.replace(cal, scalars={**cal.scalars, "missing_share": measured["missing_share"]},
                                 gap_durations_s=np.array([measured["median_gap_s"]]))
    centre = cal.scalars["missing_share"]
    misfit = dataclasses.replace(cal, ball_masks={half: np.full(mask.shape, centre < 0.5)
                                                  for half, mask in cal.ball_masks.items()})
    real_load, real_load_match = calibration.load, dfl.load_match
    try:
        with tempfile.TemporaryDirectory() as root:
            def command(cal_used, *arguments):
                calibration.load = lambda data_dir=None, **keywords: cal_used
                shown = io.StringIO()
                with contextlib.redirect_stdout(shown), loading(serve):
                    code = fidelity.main(["--profile", "gaps", "--seeds", "1", "--matches", *IDS, "--data-dir", root,
                                          *arguments])
                return code, shown.getvalue()

            code, out = command(fitted)
            assert code == 0, out
            assert "all pass" in out and "report    " in out, out
            written = os.path.join(root, "runs", "benchmark", "fidelity_gaps.m-J03WMX-J03WN1.s1.json")
            assert os.path.exists(written)
            code, out = command(misfit)
            assert code == 1 and "FAIL" in out and "DO NOT ALL PASS" in out, out
            code, out = command(fitted, "--no-write")
            assert code == 0 and "not written" in out
            code, out = command(fitted, "--profile", "clean")
            assert code == 1 and out.startswith("FAILED") and "clean" in out and "Traceback" not in out, out
            code, out = command(fitted, "--matches", "J03XXX")
            assert code == 1 and out.startswith("FAILED") and "J03XXX" in out, out
    finally:
        calibration.load, dfl.load_match = real_load, real_load_match
