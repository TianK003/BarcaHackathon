"""Checks of diagnose (pipeline/benchmark/diagnose.py): A19's IoU ladder decomposed, and the calibration of a config's
sources and of its decoder's posterior.

The ladder is held to the scorer. One hand-built half has every class of matched pair (a clean pair a few frames off, a
merge, a split), a true possession that no prediction is matched to and two predictions that nothing is matched to,
and every count, every class's loss at each IoU and the clean-pair ladder loss are worked out here by hand. On random
code arrays the greedy pass written out in diagnose.py gives pm.matched_gt_flags' own flags at every IoU, and the pooled
spellF1 of the ladder is score.pooled's to the last bits. The calibration recovers a known miscalibration (calibrated,
overconfident by a factor of two in the logit, a constant), and its log-loss, Brier score and ECE are worked out on a
handful of numbers. End to end, diagnose() runs on the synthetic matches of checks/common.py: its report has the layout
A19 declares, the ladder's spellF1 is the runner's, the frames each measure counts are those an independent tally finds,
the posterior's rows lie in [0, 1], and no list in the report is longer than an aggregate's. The matches are random
flags and tracking: their scores mean nothing. No check reads data.
"""

import contextlib
import io
import json
import math
import os
import tempfile

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import diagnose, score
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import truth as truths
from pipeline.benchmark.checks.common import (loading, random_match, read, strict_json, synthetic_calibration,
                                              synthetic_matches, yardstick_path)
from pipeline.config import load_config
from pipeline.selftest import expect_error, quiet, write_text

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
IOUS = score.IOUS
N = 600
HALF = (100, 500)       # the hand-built half, [start, stop); X outside it
# The hand-built half (frames, label), truth and prediction. Truth: g0 H, g1 A, g2 A, g3 H, g4 H with D between them.
# Prediction: p0 H (g0's, 4 frames late, 3 frames long), p1 A over g1 and g2 (an X inside it is filled with A), p2 and
# p3 H inside g3, p4 A where the truth is D, p5 H (g4's, 8 frames early, 6 frames short), and an X at the end.
TRUTH = [(100, 120, H), (120, 150, D), (150, 200, A), (200, 210, D), (210, 250, A), (250, 300, D), (300, 400, H),
         (400, 430, D), (430, 490, H), (490, 500, D)]
PREDICTED = [(100, 104, D), (104, 123, H), (123, 150, D), (150, 200, A), (200, 210, X), (210, 255, A), (255, 300, D),
             (300, 350, H), (350, 360, D), (360, 400, H), (400, 405, D), (405, 415, A), (415, 422, D), (422, 484, H),
             (484, 495, D), (495, 500, X)]
# A config that runs every kind of source A19 names on a synthetic match, the hmm decoder and a refiner after it.
MAIN_LIKE = """
description = "nearest player, ball presence, P07's shape (all frames), P05's dead ball, the hmm, shift_edges"
seed = 0
scope = "halves"

[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "ball_presence"

[[evidence]]
kind = "shape_possession"
weight = 0.2
model = "pipeline/models/shape_possession_hgb.json"
scope = "all"

[[evidence]]
kind = "dead_ball_model"
model = "pipeline/models/dead_ball_hgb.json"

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 8
min_possession = 5
min_dead = 5

[[refine]]
kind = "shift_edges"
team = 1
into_dead = -2
"""
# Team evidence only where the ball is seen (the nearest player), so that the posterior's frames split both ways.
NEAREST_ONLY = """
scope = "halves"

[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "ball_presence"

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 8
min_possession = 3
min_dead = 3
"""
REPORT_KEYS = {"config", "profile", "seeds", "matches", "dropped_match_only", "ladder", "calibration",
               "tracking_calibration", "git", "data_dir"}
LADDER_FIGURES = ("spell_f1", "clean_loss", "split_loss", "merge_loss", "both_loss")


def codes_of(pieces):
    codes = np.full(N, X, dtype=np.int8)
    for start, stop, code in pieces:
        codes[start:stop] = code
    return codes


def close(a, b, tolerance=1e-12):
    return abs(a - b) <= tolerance


def random_truth(rng, length):
    """`length` frames of possessions of H or A (10 to 60 frames), each followed by a short D (1 to 12) half the
    time."""
    codes = np.full(length, D, dtype=np.int8)
    at = 0
    while at < length:
        span = int(rng.integers(10, 61))
        codes[at:at + span] = int(rng.choice([H, A]))
        at += span + (int(rng.integers(1, 13)) if rng.random() < 0.5 else 0)
    return codes


def perturbed(codes, rng, x_share):
    """`codes` with a stretch of 1 to 25 frames overwritten once per 20 frames, by the label of the frame before it
    (a third of the time: a run grows over its neighbour), or by a random H, A, D, or X on `x_share` of them: edges
    move, possessions split and merge, and a few appear from nothing."""
    codes = codes.copy()
    for _ in range(len(codes) // 20):
        start = int(rng.integers(1, len(codes)))
        if rng.random() < 1 / 3:
            label = codes[start - 1]
        else:
            label = X if rng.random() < x_share else int(rng.choice([H, A, D]))
        codes[start:start + int(rng.integers(1, 26))] = label
    return codes


# -- the ladder -------------------------------------------------------------------------------------------------------

def test_ladder_hand_built_half():
    """Every count of the hand-built half, worked out by hand.

    Spells: truth g0 H [100, 120), g1 A [150, 200), g2 A [210, 250), g3 H [300, 400), g4 H [430, 490); prediction, the
    X filled, p0 H [104, 123), p1 A [150, 255), p2 H [300, 350), p3 H [360, 400), p4 A [405, 415), p5 H [422, 484).
    Same-team overlaps and their IoUs: g4-p5 54/68, g0-p0 16/23, g3-p2 50/100, g1-p1 50/105, g3-p3 40/100, g2-p1
    40/105. The greedy pass at 0.3 takes g4-p5, g0-p0, g3-p2 and g1-p1, and skips g3-p3 (g3 is taken) and g2-p1 (p1 is).
    p1 overlaps two true spells and g1 only p1: a merge. g3 overlaps two predictions and p2 only g3: a split. g0-p0 and
    g4-p5 are clean. Matched at 0.3, 0.4, 0.5, 0.6, 0.7: 4, 4, 3, 2, 1 (the merge is lost from 0.5 at 0.476, the split
    from 0.6 at 0.5, g0-p0 at 0.7 at 0.696). With 5 true and 6 predicted spells a pair is 2/11 of F1 at its IoU, so the
    F1s are 8/11, 8/11, 6/11, 4/11, 2/11, spellF1 28/55, and the losses are clean 1 x 2/11 / 5 = 2/55, merge 3 x 2/55
    = 6/55, split 2 x 2/55 = 4/55, both 0.
    """
    truth, predicted = codes_of(TRUTH), codes_of(PREDICTED)
    half = diagnose.ladder_half(truth, predicted, *HALF)
    assert (half["n_gt"], half["n_pred"]) == (5, 6), half
    assert list(half["tp"]) == [4, 4, 3, 2, 1], half["tp"]
    assert {name: list(counts) for name, counts in half["lost"].items()} == {
        "clean": [0, 0, 0, 0, 1], "merge": [0, 0, 1, 1, 1], "split": [0, 0, 0, 1, 1], "both": [0, 0, 0, 0, 0]}, half
    assert half["pairs"] == {"clean": 2, "merge": 1, "split": 1, "both": 0}, half["pairs"]
    assert (half["pred_multi"], half["gt_multi"]) == (1, 1), half
    assert half["unmatched"] == {"gt": {"overlapping": 1, "alone": 0}, "pred": {"overlapping": 1, "alone": 1}}, half

    summary = diagnose.ladder_summary([half])
    assert (summary["n_gt"], summary["n_pred"], summary["halves"]) == (5, 6, 1), summary
    assert close(summary["spell_f1"], 28 / 55), summary["spell_f1"]
    for figure, expected in (("clean_loss", 2 / 55), ("merge_loss", 6 / 55), ("split_loss", 4 / 55),
                             ("both_loss", 0.0)):
        assert close(summary[figure], expected), (figure, summary[figure], expected)
    for index, iou in enumerate(IOUS):
        row = summary["by_iou"][iou]
        assert row["tp"] == [4, 4, 3, 2, 1][index] and close(row["f1"], 2 * row["tp"] / 11), (iou, row)
        for name, lost in (("clean", [0, 0, 0, 0, 1]), ("merge", [0, 0, 1, 1, 1]), ("split", [0, 0, 0, 1, 1]),
                           ("both", [0] * 5)):
            assert row["lost"][name] == lost[index], (iou, name, row)
            assert close(row["loss"][name], lost[index] * 2 / 55), (iou, name, row["loss"][name])
    assert close(summary["merge_share"], 1 / 6) and close(summary["split_share"], 1 / 5), summary
    assert summary["pairs"] == {"clean": 2, "merge": 1, "split": 1, "both": 0}, summary["pairs"]
    assert summary["unmatched"] == {"gt": {"overlapping": 1, "alone": 0},
                                    "pred": {"overlapping": 1, "alone": 1}}, summary["unmatched"]

    clean = summary["clean_pairs"]
    assert clean["n"] == 2 and close(clean["iou_median"], (54 / 68 + 16 / 23) / 2), clean
    assert [clean["iou_share_at_least"][iou] for iou in IOUS] == [1.0, 1.0, 1.0, 1.0, 0.5], clean
    # Start errors -8 (p5) and +4 (p0); end errors -6 and +3. Linear quartiles of two values: a quarter and three
    # quarters of the way from the lower to the higher.
    assert clean["start_error"] == {"median": -2.0, "q25": -5.0, "q75": 1.0, "within_5": 0.5}, clean["start_error"]
    assert clean["end_error"] == {"median": -1.5, "q25": -3.75, "q75": 0.75, "within_5": 0.5}, clean["end_error"]

    # The pairs pooled over two copies of the half are the same shares and losses, with every count doubled.
    twice = diagnose.ladder_summary([half, half])
    assert (twice["n_gt"], twice["n_pred"], twice["by_iou"][0.7]["lost"]["merge"]) == (10, 12, 2), twice
    assert close(twice["clean_loss"], 2 / 55) and close(twice["spell_f1"], 28 / 55), twice
    expect_error(ValueError, lambda: diagnose.ladder_summary([]), "ladder_summary", "no half")
    broken = truth.copy()
    broken[300] = X
    expect_error(ValueError, lambda: diagnose.ladder_half(broken, predicted, *HALF), "X", "truth")
    expect_error(ValueError, lambda: diagnose.ladder_half(truth, predicted[:-1], *HALF), "shape")


def test_ladder_pairs_are_the_scorers_matching():
    """On 30 seeded random matches of two halves, the pass's flags are pm.matched_gt_flags' at every IoU, and the
    spellF1 is score.pooled's.

    The truth is possessions of 10 to 60 frames, half of them followed by a short D (random_truth); the prediction is
    the truth with a stretch overwritten every 20 frames (perturbed), some of them X (filled as the scorer fills them),
    so that edges move, possessions split and merge, and IoUs tie. The pairs matched at an IoU are the pairs matched at
    0.3 whose IoU is at least that one: the matching at 0.3 cut there. The control is that there are hundreds of pairs,
    some of them tied, and every class among them.
    """
    graded = ((1, 20, 300), (2, 330, 580))
    halves, scored, pairs_seen, ties = [], [], 0, 0
    for seed in range(30):
        rng = np.random.default_rng([23, seed])
        truth = np.full(N, X, dtype=np.int8)
        predicted = np.full(N, X, dtype=np.int8)
        for _, start, stop in graded:
            truth[start:stop] = random_truth(rng, stop - start)       # no X: the truth has none in a half
            predicted[start:stop] = perturbed(truth[start:stop], rng, 0.1)
        for _, start, stop in graded:
            runs = [(start, stop)]
            gt_spells = pm.possession_spells(truth, runs)
            pred_spells = pm.possession_spells(pm.fill_abstentions(predicted, runs), runs)
            overlaps = pm.spell_overlaps(gt_spells, pred_spells)
            ties += len(overlaps) - len({iou for iou, _, _ in overlaps})
            pairs = diagnose.greedy_pairs(overlaps)
            pairs_seen += len(pairs)
            for iou in IOUS:
                flags = np.zeros(len(gt_spells), dtype=bool)
                flags[[gt for gt, _, value in pairs if value >= iou]] = True
                assert np.array_equal(flags, pm.matched_gt_flags(overlaps, len(gt_spells), iou)), (seed, start, iou)
            halves.append(diagnose.ladder_half(truth, predicted, start, stop))
        scored += score.score_halves(truth, predicted, graded, "RND{:03d}".format(seed), 0)
    assert pairs_seen > 200 and ties > 0,"too few pairs, or no tied IoU, to prove much: {} {}".format(pairs_seen, ties)
    summary = diagnose.ladder_summary(halves)
    assert all(summary["pairs"][name] > 0 for name in ("clean", "merge", "split", "both")), summary["pairs"]
    pooled = score.pooled(scored)
    assert close(summary["spell_f1"], pooled["spell_f1_mean"]), (summary["spell_f1"], pooled["spell_f1_mean"])
    assert (summary["n_gt"], summary["n_pred"]) == (pooled["n_gt"], pooled["n_pred"]), summary
    for iou in IOUS:
        assert close(summary["by_iou"][iou]["f1"], pooled["spell_f1"][iou]), iou
    # Every pair lost at an IoU is in one class: the class losses add up to the whole ladder's.
    whole = float(np.mean([2 * summary["by_iou"][0.3]["tp"] / (summary["n_gt"] + summary["n_pred"])] * len(IOUS)))
    parts = sum(summary[name + "_loss"] for name in ("clean", "merge", "split", "both"))
    assert close(summary["spell_f1"] + parts, whole, 1e-9), (summary["spell_f1"], parts, whole)


# -- the calibration --------------------------------------------------------------------------------------------------

def test_calibration_recovers_known_miscalibration():
    """Platt's alpha and beta recover what was put in; log-loss, Brier and ECE are worked out on five numbers.

    y is drawn from p (n = 200,000, np.random.default_rng(0)): p itself fits with alpha 1 and beta 0, and p' =
    sigmoid(2 logit(p)), twice as sure as the truth, with alpha 1/2 (overconfident). A constant 0.5 against half ones
    is calibrated exactly: ECE 0 to rounding, log-loss log 2, Brier 1/4, and only a bias can be fitted (alpha None).
    """
    rng = np.random.default_rng(0)
    n = 200_000
    p = rng.uniform(0.05, 0.95, size=n)
    y = rng.random(n) < p
    fair = diagnose.calibration(p, y)
    assert fair["converged"] and abs(fair["alpha"] - 1.0) < 0.05 and abs(fair["beta"]) < 0.05, fair
    assert fair["n"] == n and len(fair["reliability"]) == 10 and sum(row["n"] for row in fair["reliability"]) == n
    sure = diagnose.calibration(1.0 / (1.0 + np.exp(-2.0 * np.log(p / (1.0 - p)))), y)
    assert sure["converged"] and abs(sure["alpha"] - 0.5) < 0.05 and abs(sure["beta"]) < 0.05, sure
    assert sure["log_loss"] > fair["log_loss"] and sure["ece"] > fair["ece"], (sure, fair)

    flat = diagnose.calibration(np.full(1000, 0.5), np.arange(1000) % 2 == 0)
    assert flat["ece"] < 1e-12 and close(flat["log_loss"], math.log(2.0)) and close(flat["brier"], 0.25), flat
    assert flat["alpha"] is None and close(flat["beta"], 0.0) and flat["converged"], flat
    assert [row["n"] for row in flat["reliability"]] == [0] * 5 + [1000] + [0] * 4, flat["reliability"]

    # Five numbers: bins 1 (0.1, 0.1 against 0, 1), 9 (0.9, 0.9 against 1, 1) and 5 (0.55 against 0). ECE = 2/5 x 0.4 +
    # 2/5 x 0.1 + 1/5 x 0.55 = 0.31; Brier = (0.01 + 0.81 + 0.01 + 0.01 + 0.3025) / 5.
    small = diagnose.calibration([0.1, 0.1, 0.9, 0.9, 0.55], [0, 1, 1, 1, 0])
    assert close(small["ece"], 0.31) and close(small["brier"], 1.1425 / 5), small
    assert close(small["log_loss"], -(3 * math.log(0.9) + math.log(0.1) + math.log(0.45)) / 5), small
    rows = small["reliability"]
    assert (rows[1]["n"], rows[5]["n"], rows[9]["n"]) == (2, 1, 2) and rows[0]["mean_p"] is None, rows
    assert close(rows[1]["low"], 0.1) and close(rows[1]["high"], 0.2) and close(rows[9]["mean_y"], 1.0), rows
    edges = diagnose.calibration([0.0, 1.0], [0, 1])
    assert edges["reliability"][0]["n"] == 1 and edges["reliability"][9]["n"] == 1, "1.0 is in the last bin"
    assert close(edges["log_loss"], -math.log(1 - diagnose.CLIP)), "p is clipped away from 0 and 1 for the log"
    # Perfectly separated: the likelihood has no maximum, and the fit says so instead of a number.
    apart = diagnose.calibration([0.2, 0.3, 0.7, 0.8], [0, 0, 1, 1])
    assert not apart["converged"] and apart["alpha"] is None and apart["beta"] is None, apart
    empty = diagnose.calibration([], [])
    assert empty["n"] == 0 and empty["log_loss"] is None and len(empty["reliability"]) == 10, empty
    expect_error(ValueError, lambda: diagnose.calibration([0.5, 1.5], [0, 1]), "calibration", "[0, 1]")
    expect_error(ValueError, lambda: diagnose.calibration([0.5], [0, 1]), "calibration", "length")
    expect_error(ValueError, lambda: diagnose.calibration([0.5, 0.5], [0, 2]), "calibration", "0 or 1")


# -- diagnose() end to end --------------------------------------------------------------------------------------------

def walk_lists(value, where="report"):
    """[(where, length, is a reliability table)] of every list in `value`, however deep."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += walk_lists(item, "{}.{}".format(where, key))
    elif isinstance(value, list):
        found.append((where, len(value), where.endswith(".reliability")))
        for number, item in enumerate(value):
            found += walk_lists(item, "{}[{}]".format(where, number))
    return found


def calibrations(report):
    """Every calibration() result in the report's calibration block, by where it is."""
    found = {}
    for name, entry in report["calibration"].items():
        for channel, value in entry.items():
            if isinstance(value, dict) and "reliability" in value:
                found["{}.{}".format(name, channel)] = value
            elif isinstance(value, dict):
                for part, inner in value.items():
                    found["{}.{}.{}".format(name, channel, part)] = inner
    return found


def tallies(config_text, matches, root):
    """What diagnose's measures count, tallied here frame by frame from run.predict's sources and the raw truth."""
    config = load_config(write_text(root, "tally.toml", config_text))
    counts = {"graded": 0, "alive": 0, "speaks": 0, "home": {}, "visible": {}, "alive_of": {}}
    for match in matches.values():
        raw, _ = truths.truth_codes(match)
        atd = benchmark_run.make_atd(match, "gaps", 0, synthetic_calibration())
        inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=0)
        _, sources = benchmark_run.predict(config, inputs)
        for _, start, stop in match.graded:
            for frame in range(start, stop):
                counts["graded"] += 1
                live = raw[frame] in (H, A)
                counts["alive"] += live
                counts["speaks"] += any(not np.isnan(source.home[frame]) for source in sources.values())
                for name, source in sources.items():
                    if live and not np.isnan(source.home[frame]):
                        counts["home"][name] = counts["home"].get(name, 0) + 1
                        counts["visible"][name] = counts["visible"].get(name, 0) + bool(atd.ball_present[frame])
                    if not np.isnan(source.alive[frame]):
                        counts["alive_of"][name] = counts["alive_of"].get(name, 0) + 1
    return counts


def test_diagnose_end_to_end_on_synthetic_matches():
    """diagnose() on the synthetic matches (gaps, one seed): the declared layout, the runner's spellF1, exact counts.

    The ladder's pooled spellF1 under each truth is the one run.evaluate_truths reports for the same config, matches,
    profile and seed, so the ladder sees the codes the runner scores, after the refiner. Each source is measured on its
    channels where it speaks (nearest_player and shape_possession on home, ball_presence and dead_ball_model on alive),
    on as many frames as a tally from run.predict's own sources finds, and shape_possession is split by whether the
    ball is seen. The posterior's alive measure covers every graded frame, split by whether a team source speaks, and
    its home measure the truth's alive frames; every reliability row of it lies in [0, 1] and in its bin. No list in
    the report is longer than 64, and only the reliability tables hold 10 rows.
    """
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "main_like.toml", MAIN_LIKE)
        lines = []
        report = diagnose.diagnose(path, profile="gaps", seeds=1, matches=matches,
                                   calibration=synthetic_calibration(), data_dir=root, log=lines.append)
        assert set(report) == REPORT_KEYS, sorted(report)
        assert report["config"]["stem"] == "main_like" and len(report["config"]["sha256"]) == 64, report["config"]
        assert report["profile"]["name"] == "gaps" and report["seeds"] == [0], report
        assert report["matches"] == sorted(matches) and report["dropped_match_only"] == [], report
        assert set(report["ladder"]) == {"raw", "coarse", "mean"}, report["ladder"]
        runner = benchmark_run.evaluate_truths([path], profile="gaps", seeds=1, matches=matches,
                                               calibration=synthetic_calibration(), data_dir=root, log=quiet)
        for name in ("raw", "coarse"):
            pooled = runner[name]["configs"][0]["report"]["pooled"]
            assert close(report["ladder"][name]["spell_f1"], pooled["spell_f1_mean"], 1e-9), (name, pooled)
            assert report["ladder"][name]["n_gt"] == pooled["n_gt"] > 10, (name, report["ladder"][name]["n_gt"])
        for figure in LADDER_FIGURES:
            mean = (report["ladder"]["raw"][figure] + report["ladder"]["coarse"][figure]) / 2
            assert close(report["ladder"]["mean"][figure], mean, 2e-9), (figure, report["ladder"]["mean"])
        assert set(report["ladder"]["mean"]) == set(LADDER_FIGURES), report["ladder"]["mean"]

        counts = tallies(MAIN_LIKE, matches, root)
        block = report["calibration"]
        assert set(block) == {"nearest_player", "ball_presence", "shape_possession", "dead_ball_model", "posterior"}
        assert set(block["nearest_player"]) == {"kind", "weight", "home"}, block["nearest_player"]
        assert set(block["ball_presence"]) == {"kind", "weight", "alive"}, block["ball_presence"]
        assert set(block["dead_ball_model"]) == {"kind", "weight", "alive"}, block["dead_ball_model"]
        assert set(block["shape_possession"]) == {"kind", "weight", "home", "home_ball_visible", "home_ball_missing"}
        assert block["shape_possession"]["weight"] == 0.2 and block["shape_possession"]["kind"] == "shape_possession"
        for name in ("nearest_player", "shape_possession"):
            assert block[name]["home"]["n"] == counts["home"][name] > 0, (name, block[name]["home"]["n"])
        shape = block["shape_possession"]
        assert shape["home_ball_visible"]["n"] == counts["visible"]["shape_possession"] > 0, shape
        assert shape["home_ball_visible"]["n"] + shape["home_ball_missing"]["n"] == shape["home"]["n"], shape
        assert shape["home_ball_missing"]["n"] > 0, "the gaps profile hides the ball somewhere"
        for name in ("ball_presence", "dead_ball_model"):
            assert block[name]["alive"]["n"] == counts["alive_of"][name] == counts["graded"], (name, counts)
        posterior = block["posterior"]
        assert set(posterior) == {"alive", "home"} and set(posterior["alive"]) == {"all", "team_speaks", "team_silent"}
        assert posterior["alive"]["all"]["n"] == counts["graded"], (posterior["alive"]["all"]["n"], counts)
        assert posterior["alive"]["team_speaks"]["n"] == counts["speaks"], posterior["alive"]
        assert posterior["alive"]["team_silent"]["n"] == counts["graded"] - counts["speaks"], posterior["alive"]
        assert posterior["home"]["n"] + posterior["home"]["frames_without_alive_mass"] == counts["alive"], posterior
        for where, measure in calibrations(report).items():
            rows = measure["reliability"]
            assert len(rows) == 10 and sum(row["n"] for row in rows) == measure["n"], where
            for row in rows:
                if row["n"]:
                    assert row["low"] - 1e-12 <= row["mean_p"] <= row["high"] + 1e-12, (where, row)
                    assert 0.0 <= row["mean_y"] <= 1.0, (where, row)
            if measure["n"]:
                assert 0.0 <= measure["mean_p"] <= 1.0 and 0.0 <= measure["ece"] <= 1.0, (where, measure)

        for where, length, reliability in walk_lists(report):
            assert length <= diagnose.MAX_LIST, (where, length)
            assert not reliability or length == 10, (where, length)
        json.dumps(report, allow_nan=False)
        text = "\n".join(lines)
        assert "ladder" in text and "posterior" in text and "clean" in text, text

        # Team evidence only where the ball is seen: the posterior's alive frames split both ways, as the tally says.
        quiet_report = diagnose.diagnose(write_text(root, "nearest_only.toml", NEAREST_ONLY), profile="gaps", seeds=1,
                                         matches=matches, calibration=synthetic_calibration(), log=quiet)
        counts = tallies(NEAREST_ONLY, matches, root)
        alive = quiet_report["calibration"]["posterior"]["alive"]
        assert alive["team_speaks"]["n"] == counts["speaks"] > 0, alive["team_speaks"]["n"]
        assert alive["team_silent"]["n"] == counts["graded"] - counts["speaks"] > 0, alive["team_silent"]["n"]


def test_diagnose_command_line_writes_the_report():
    """main() writes <data dir>/runs/diagnose/<stem>[.<profile>][.m-<ids>][.s<n>].json, the report diagnose() returns.

    Served by random matches through dfl.load_match, on the clean profile (no Calibration to measure). A run that is
    not the declared one (another profile, some matches) carries them in its name, as the runner's reports do, so that
    it never replaces the declared report. With --no-write nothing is written. A config that cannot be diagnosed is
    "FAILED    <sentence>" and exit code 1.
    """
    def serve(match_id):
        return random_match(match_id, seed=len(match_id))

    with tempfile.TemporaryDirectory() as root, loading(serve):
        path = write_text(root, "nearest_only.toml", NEAREST_ONLY)
        arguments = [path, "--profile", "clean", "--matches", "J03WOH", "J03WMX", "--data-dir", root]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = diagnose.main(arguments)
        out = shown.getvalue()
        assert code == 0, out
        written = os.path.join(root, "runs", "diagnose", "nearest_only.clean.m-J03WMX-J03WOH.json")
        assert os.path.isfile(written) and "report    " in out and "ladder" in out, (os.listdir(root), out)
        report = strict_json(read(written))
        again = diagnose.diagnose(path, profile="clean", match_ids=["J03WMX", "J03WOH"], data_dir=root, log=quiet)
        assert report == again, "the file is the report diagnose() returns"
        assert diagnose.report_path(root, "main", "tracking", None, 3) == os.path.join(root, "runs", "diagnose",
                                                                                      "main.json")

        os.remove(written)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = diagnose.main(arguments + ["--no-write"])
        assert code == 0 and "not written" in shown.getvalue() and not os.path.exists(written), shown.getvalue()

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = diagnose.main([yardstick_path(50), "--profile", "clean", "--data-dir", root])
        assert code == 1 and shown.getvalue().startswith("FAILED") and "hmm" in shown.getvalue(), shown.getvalue()


def test_diagnose_refuses_a_non_hmm_decoder():
    """A majority_vote decoder has no posterior, the restart states have none either, and a source may not be called
    "posterior": each is a BenchmarkError naming the config and what to change, before any match is loaded."""
    def serve(match_id):
        raise AssertionError("a match was loaded for a config that is refused: " + match_id)

    with tempfile.TemporaryDirectory() as root, loading(serve):
        expect_error(benchmark_run.BenchmarkError,
                     lambda: diagnose.diagnose(yardstick_path(50), profile="clean", data_dir=root, log=quiet),
                     "nearest_w50", "majority_vote", "hmm")
        restart = NEAREST_ONLY.replace("min_dead = 3", 'min_dead = 3\nrestart_source = "nearest_player"')
        expect_error(benchmark_run.BenchmarkError,
                     lambda: diagnose.diagnose(write_text(root, "restart.toml", restart), profile="clean",
                                               data_dir=root, log=quiet),
                     "restart", "restart_source")
        named = NEAREST_ONLY.replace('kind = "ball_presence"', 'kind = "ball_presence"\nname = "posterior"')
        expect_error(benchmark_run.BenchmarkError,
                     lambda: diagnose.diagnose(write_text(root, "named.toml", named), profile="clean", data_dir=root,
                                               log=quiet),
                     "named", "posterior")
        file_scope = NEAREST_ONLY.replace('scope = "halves"', 'scope = "file"')
        expect_error(benchmark_run.BenchmarkError,
                     lambda: diagnose.diagnose(write_text(root, "whole.toml", file_scope), profile="clean",
                                               data_dir=root, log=quiet),
                     "whole", "scope")
