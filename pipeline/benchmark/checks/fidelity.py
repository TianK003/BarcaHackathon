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

from pipeline.benchmark import calibration, degrade, dfl, fidelity, run, sources
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import truth as truths
from pipeline.benchmark.checks.calibration import brute_force_agreement
from pipeline.benchmark.checks.common import synthetic_calibration
from pipeline.benchmark.checks.degrade import made_up_calibration, steady_match
from pipeline.benchmark.checks.inputs import random_match
from pipeline.benchmark.checks.run import loading, read, strict_json, synthetic_matches, yardstick_path
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
PLAN_FIGURES = {0.407, 0.4069, 10.7, 0.778, 0.66, 0.066, 0.108, 0.495, 20.5, 35.5, 0.0796, 0.080}
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
        assert sorted(result) == sorted(["profile", "seeds", "matches", "calibration", "gates", "checks", "statistics",
                                         "draws", "players_draws", "passed", "warnings", "notes"])
        assert result["players_draws"] == {}, "gaps degrades no player"
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
    # relative (50% of the centre) and the rest are absolute. Two of them are pending until the low-count tail is built.
    tracking = {gate["name"]: gate for gate in fidelity.GATES["tracking"]}
    assert list(tracking) == TRACKING_GATES and fidelity.GATES["tracking"][:2] == fidelity.GATES["gaps"]
    assert {name: (gate["tolerance"], gate["kind"]) for name, gate in tracking.items()} == {
        "missing_share": (0.02, "absolute"), "median_gap_s": (1.5, "absolute"), "present_share_22": (0.06, "absolute"),
        "present_mean": (0.5, "absolute"), "present_share_le14": (0.02, "absolute"),
        "wrong_team_share": (0.03, "absolute"), "tracklet_median_s": (0.5, "relative"),
        "speed_nan_share": (0.01, "absolute")}
    assert fidelity.PENDING_GATES == ("present_mean", "present_share_le14")
    assert set(fidelity.PENDING_GATES) <= set(tracking)
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


def tracking_calibration():
    """A Calibration with what the tracking profile reads: synthetic_calibration with the made-up tracklets, episodes
    and NaN-speed runs of checks/degrade.py (its other fields are the synthetic bundle's)."""
    made = made_up_calibration()
    return dataclasses.replace(
        synthetic_calibration(), present_counts=made.present_counts, tracklet_lengths_s=made.tracklet_lengths_s,
        wrong_team_episodes_s=made.wrong_team_episodes_s, wrong_team_share=made.wrong_team_share,
        speed_nan_runs=made.speed_nan_runs, speed_nan_share=made.speed_nan_share)


def steady_matches():
    """Two matches whose players stand on the pitch for whole stretches (checks/degrade.py's steady_match)."""
    return {"RND001": steady_match("RND001", 0), "RND002": steady_match("RND002", 1, rows=(260, 190))}


def worked_out_tracking(matches, cal, seeds=SEEDS):
    """The pooled figures of the tracking gates by other routes than the report's: each match and seed degraded and
    measured by calibration.player_stats on its own, the histograms and counts added, the tracklets laid end to end."""
    frames = slot_frames = wrong = nan = 0
    hist = np.zeros(degrade.N_SLOTS + 2, dtype=np.int64)
    tracklets, drawn = [], {}
    for match in matches.values():
        for seed in range(seeds):
            atd, info = degrade.degrade(degrade.tracking_view(match), "tracking", cal, seed)
            inputs = match_inputs.MatchInputs(match, atd, profile="tracking", seed=seed)
            spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
            stats = calibration.player_stats(atd.players_present, atd.slot_team, inputs.metres(atd.players_xy),
                                             atd.players_speed, spans, inputs.fps)
            frames, slot_frames = frames + stats["frames"], slot_frames + stats["present_slot_frames"]
            wrong, nan = wrong + stats["wrong_team_frames"], nan + stats["speed_nan_frames"]
            hist[:stats["present_hist"].size] += stats["present_hist"]
            tracklets.append(stats["tracklet_lengths_s"])
            for name in fidelity.PLAYERS_DRAWS:
                drawn[name] = drawn.get(name, 0) + info["players"][name]
    return {"present_share_22": hist[22] / frames, "present_mean": slot_frames / frames,
            "present_share_le14": hist[:15].sum() / frames, "wrong_team_share": wrong / frames,
            "tracklet_median_s": float(np.median(np.concatenate(tracklets))), "speed_nan_share": nan / slot_frames,
            "players_draws": drawn}


def test_fidelity_tracking_report_synthetic():
    """The tracking profile's eight gates, each worked out by other code, with the two pending ones not counted.

    Two steady matches and the made-up tracklets, episodes and NaN runs: every pooled number is the sums of
    calibration.player_stats called on each degraded (match, seed) here and not by the report, the median is that of
    the tracklets laid end to end, the centres are the Calibration's, a relative tolerance is a share of its centre,
    and a pending gate is pending whatever it measures. The pooled counts of what was drawn are the degrader's own."""
    matches, cal = steady_matches(), tracking_calibration()
    with tempfile.TemporaryDirectory() as root:
        result = fidelity.report("tracking", seeds=SEEDS, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert result["profile"] == {"name": "tracking", "version": 1} and result["matches"] == ["RND001", "RND002"]
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
            expected = ours[name] if name in ours else gaps[name]    # the ball's two are those of the gaps profile
            if name in ("missing_share", "median_gap_s"):
                assert gate["status"] in ("pass", "FAIL")
            else:
                assert abs(gate["measured"] - expected) < 1e-9, (name, gate["measured"], expected)
            limit = gate["tolerance"] * (abs(gate["centre"]) if gate["kind"] == "relative" else 1)
            inside = abs(gate["measured"] - gate["centre"]) <= limit
            want = "pending" if name in fidelity.PENDING_GATES else ("pass" if inside else "FAIL")
            assert gate["status"] == want, (name, gate)
        assert {gate["name"] for gate in result["gates"] if gate["status"] == "pending"} == set(fidelity.PENDING_GATES)
        assert result["passed"] == (not any(gate["status"] == "FAIL" for gate in result["gates"]))
        assert any("pending" in note.lower() and "present_mean" in note and "present_share_le14" in note
                   and "tail" in note for note in result["notes"]), result["notes"]
        # The ball is the gaps profile's, whatever the players do: the same two gate values.
        by_name = {gate["name"]: gate for gate in result["gates"]}
        assert abs(by_name["missing_share"]["measured"] - gaps["missing_share"]) < 1e-9
        assert abs(by_name["median_gap_s"]["measured"] - gaps["median_gap_s"]) < 1e-9
        # What was drawn, pooled: the degrader's own counts, none of them zero here.
        assert result["players_draws"] == ours["players_draws"], (result["players_draws"], ours["players_draws"])
        assert set(result["players_draws"]) == set(fidelity.PLAYERS_DRAWS)
        assert result["players_draws"]["speed_nan_runs"] > 0 and result["players_draws"]["tracklets"] > 0
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
        for fragment in ("tracklet_median_s", "50%", "pending", "present_mean", "players' draws", "episodes",
                         "(2 pending, not counted)"):
            assert fragment in text, (fragment, text)


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

    The gates' tolerances are opened and closed on purpose (the profile's own are the plan's): wide open, every
    required gate passes and the two pending ones are printed as pending, exit 0, whatever they would measure; with the
    required NaN-speed gate closed to a tolerance of 0 the command exits 1 and says so; with only a pending gate closed
    (it would FAIL if it were counted) the exit code is 0 again. An empty PENDING_GATES counts all eight."""
    cal = tracking_calibration()
    matches = {match_id: steady_match(match_id, seed=sources.MATCH_IDS.index(match_id)) for match_id in IDS}
    real_load, real_load_match = calibration.load, dfl.load_match
    real_gates, real_pending = fidelity.GATES, fidelity.PENDING_GATES

    def serve(match_id):
        return matches[match_id]

    def with_tolerance(**tolerances):
        return {**real_gates, "tracking": tuple(
            {**spec, "tolerance": tolerances.get(spec["name"], 1e9), "kind": "absolute"}
            for spec in real_gates["tracking"])}

    try:
        with tempfile.TemporaryDirectory() as root:
            def command(gates, pending=real_pending):
                calibration.load = lambda data_dir=None, **keywords: cal
                fidelity.GATES, fidelity.PENDING_GATES = gates, pending
                shown = io.StringIO()
                with contextlib.redirect_stdout(shown), loading(serve):
                    code = fidelity.main(["--profile", "tracking", "--seeds", "1", "--matches", *IDS,
                                          "--data-dir", root, "--no-write"])
                return code, shown.getvalue()

            code, out = command(with_tolerance())
            assert code == 0 and "all pass (2 pending, not counted)" in out and "pending" in out, out
            assert out.count("pending") >= 3, "each pending gate says it is, and the note says why"
            code, out = command(with_tolerance(speed_nan_share=0.0))
            assert code == 1 and "DO NOT ALL PASS" in out and "FAIL" in out, out
            code, out = command(with_tolerance(present_mean=0.0, present_share_le14=0.0))
            assert code == 0 and "all pass (2 pending, not counted)" in out, out
            code, out = command(with_tolerance(present_mean=0.0, present_share_le14=0.0), pending=())
            assert code == 1 and "DO NOT ALL PASS" in out and "pending" not in out, (
                "with the list empty the same gates are counted: " + out)
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

        expect_error(fidelity.FidelityError, lambda: measure(profile="clean"), "clean", "no gates", "gaps")
        expect_error(fidelity.FidelityError, lambda: measure(profile="clean"), "clean", "degrades nothing", "gaps",
                     "tracking")
        expect_error(fidelity.FidelityError, lambda: measure(profile="context"), "context", "not one of", "clean",
                     "gaps", "tracking")
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
        for fragment in ("missing_share", "median_gap_s", "nearest_agreement", "dead_ball_visible_share",
                         "possessions_per_min:nearest_w50", "d_share:nearest_w125", "pass" if result["passed"]
                         else "FAIL", "calibration " + result["calibration"]["sha256"][:12], "half-draws"):
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
