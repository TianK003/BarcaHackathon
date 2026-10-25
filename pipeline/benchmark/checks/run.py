"""Checks of the benchmark runner (pipeline/benchmark/run.py): configs scored on matches, reported, and repeatable.

The runner is what the team chooses between methods with, so what is held against it here is honesty and repeatability
more than arithmetic (score.py's checks own the arithmetic): that what it prints and writes is what the scoring
functions say, that a config's numbers do not depend on which other configs ran beside it, that the comparison's
direction is "other minus the first", that the same run gives the same bytes in another process with another hash seed,
that it refuses what it cannot score before it loads a match and names the source that needs a video, that a subset of
the matches never replaces a full report, and that the oracle, a prediction that is the truth itself, scores 1.0 on
every real match (--dfl). The oracle checks the chain from the loaded match to the score (the runner, the inputs and
their separator rows, both halves, the decoder path and the scoring) and not the link from the DFL file to the truth:
its answers and the runner's truth are the same truth.truth_codes of the same loaded match, so a mistake in the parse
or in truth_codes is on both sides and cancels. checks/dfl.py and checks/truth.py cover that link
(test_dfl_oracle_scores_perfectly says which). The matches of the synthetic checks are random_match's
(checks/common.py): two halves of random flags and random tracking, whose score means nothing and whose every number
can be recomputed here with the scoring functions, which is what the checks do.
"""

import contextlib
import gc
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import weakref

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import calibration, degrade, dfl, score, sources, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (MIN_AGREEMENT, context_calibration, loading, random_match, read,
                                              real_match, strict_json, synthetic_calibration, synthetic_matches,
                                              tracking_calibration, whole_positions, write_synthetic_dfl,
                                              yardstick_path)
from pipeline.config import ConfigError, load_config
from pipeline.selftest import P06_TRIAL_CONFIG, expect_error, quiet, write_text

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORT_KEYS = {"calibration", "config", "environment", "git", "halves", "labels", "matches", "per_match", "pooled",
               "possessions_per_graded_minute", "profile", "seeds", "t_interval", "truth", "versus"}
HALF_KEYS = {"match_id", "half", "seed", "n_gt", "n_pred", "tp", "confusion"}
ROUNDING = 1e-9                         # reports round their floats to 9 decimals
HASH_SEEDS = ("1", "2", "3")            # PYTHONHASHSEED of the three subprocesses of the determinism checks
SUBSET = ".m-RND001-RND002"             # what the file name of a report says of the synthetic matches: not all seven
ORACLE = """
scope = "halves"

[[evidence]]
kind = "oracle"
lag = {lag}
half = {half}
swap = {swap}

[decoder]
kind = "majority_vote"
half_width = 0
fallback = "D"
"""
HMM = """
description = "the hmm decoder on the nearest player's team"
seed = 0
scope = "halves"

[[evidence]]
kind = "nearest_player"

[decoder]
kind = "hmm"
switch_team = 6
switch_dead = 12
min_possession = 5
min_dead = 5
"""
SHIFT = """
[[refine]]
kind = "shift_edges"
team = 4
into_dead = -3
out_of_dead = 6
"""


# -- helpers ----------------------------------------------------------------------------------------------------------

def evaluate_clean(*arguments, **keywords):
    """benchmark_run.evaluate_configs on the clean profile unless one is named: what most of these checks are about.

    The runner's default profile is tracking (it needs our Calibration), and the checks that are about the runner, not
    about a degradation, run on the clean one, which needs none.
    """
    keywords.setdefault("profile", "clean")
    return benchmark_run.evaluate_configs(*arguments, **keywords)


def evaluate_gaps(*arguments, **keywords):
    """benchmark_run.evaluate_configs on the gaps profile unless one is named: the profile that the synthetic
    Calibration of the bundle can serve (it has our masks and none of the tracking profile's figures)."""
    keywords.setdefault("profile", "gaps")
    return benchmark_run.evaluate_configs(*arguments, **keywords)


@contextlib.contextmanager
def registered(registry, **components):
    """The components in `registry` (a stage's REGISTRY) while the block runs, and the registry as it was after it."""
    before = dict(registry)
    registry.update(components)
    try:
        yield
    finally:
        registry.clear()
        registry.update(before)


def independent_halves(match, config_path, data_dir, raw=None):
    """[HalfScore] of one config on one match, by the scoring functions called here and not through the runner."""
    config = load_config(config_path)
    codes_true = truth.truth_codes(match)[0] if raw is None else raw
    inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match), data_dir=data_dir)
    codes, _ = benchmark_run.predict(config, inputs)
    return score.score_halves(codes_true, codes, match.graded, match.match_id, 0)


def keys_of(value, found=None):
    """Every dict key anywhere in `value` (a JSON-ready report), as a set."""
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            keys_of(item, found)
    elif isinstance(value, list):
        for item in value:
            keys_of(item, found)
    return found


# -- the report -------------------------------------------------------------------------------------------------------

def test_runner_end_to_end_synthetic():
    """Three configs on two matches: the report's shape, and every number in it equal to the scoring functions' own.

    The first config is the reference. Its report has no comparison; the others' hold b - a against it, and the sign
    is checked from the reports' own per-match numbers. A report is a function of its config and the matches, so
    running a config alone gives its report from the three-config run except for the comparison. The files are LF,
    sorted, strict JSON, with no timing in them, and a second write replaces the first byte for byte.
    """
    matches = synthetic_matches()
    paths = [yardstick_path(width) for width in (50, 0, 25)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = evaluate_clean(paths, matches=matches, data_dir=root, log=lines.append)
        configs = result["configs"]
        assert [entry["stem"] for entry in configs] == ["nearest_w50", "nearest_w0", "nearest_w25"]
        assert result["reference"] == "nearest_w50" and result["data_dir"] == root
        reports = {entry["stem"]: entry["report"] for entry in configs}
        for stem, report in reports.items():
            assert set(report) == REPORT_KEYS, (stem, sorted(set(report) ^ REPORT_KEYS))
            assert report["profile"] == {"name": "clean", "version": 1} and report["seeds"] == [0], stem
            assert report["truth"] == {"name": "raw", "min_s": None} and report["calibration"] is None, stem
            assert set(report["git"]) == {"commit", "dirty"}
            assert set(report["environment"]) == {"python", "numpy", "lxml", "platform"}
            assert report["environment"]["numpy"] == np.__version__
            assert report["environment"]["python"] == platform.python_version()
            assert [record["id"] for record in report["matches"]] == ["RND001", "RND002"], stem
            assert [record["frames"] for record in report["matches"]] == [465, 585]
            assert [record["graded_frames"] for record in report["matches"]] == [440, 560]
            assert all(record["sources"] is None and "RND00" in record["sources_reason"]
                       for record in report["matches"]), "random matches have no DFL cache to name"
            assert sorted(report["per_match"]) == ["RND001", "RND002"]
            assert len(report["halves"]) == 4 and all(set(half) == HALF_KEYS for half in report["halves"])
            assert [(half["match_id"], half["half"]) for half in report["halves"]] == [
                ("RND001", 1), ("RND001", 2), ("RND002", 1), ("RND002", 2)], "matches in the order they were run"
            keys = keys_of(report)
            assert not keys & {"seconds", "timings", "elapsed", "time"}, "timings are printed, never written"
        for width, stem in ((50, "nearest_w50"), (0, "nearest_w0"), (25, "nearest_w25")):
            assert reports[stem]["config"]["text"] == read(yardstick_path(width))
            assert reports[stem]["config"]["path"].endswith("nearest_w{}.toml".format(width))
            assert "\r" not in reports[stem]["config"]["text"]

        # Every number is the scoring functions' own, from halves counted here and not by the runner.
        for width, stem in ((50, "nearest_w50"), (0, "nearest_w0"), (25, "nearest_w25")):
            report = reports[stem]
            halves = []
            for match in matches.values():
                halves += independent_halves(match, yardstick_path(width), root)
            assert [(half["match_id"], half["half"], half["seed"], half["n_gt"], half["n_pred"], tuple(half["tp"]),
                     tuple(tuple(row) for row in half["confusion"])) for half in report["halves"]] == [
                (h.match_id, h.half, h.seed, h.n_gt, h.n_pred, h.tp, h.confusion) for h in halves], stem
            for name, expected in score.pooled(halves).items():
                got = report["pooled"][name]
                if name == "spell_f1":
                    assert {float(iou): value for iou, value in got.items()} == {
                        iou: round(value, 9) for iou, value in expected.items()}, stem
                else:
                    assert abs(got - expected) < ROUNDING, (stem, name, got, expected)
            per = score.per_match(halves)
            for match_id, expected in per.items():
                for name in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt", "n_pred"):
                    assert abs(report["per_match"][match_id][name] - expected[name]) < ROUNDING, (stem, match_id, name)
            for metric in ("spell_f1_mean", "macro_f1"):
                mean, low, high = score.t_interval([per[match_id][metric] for match_id in sorted(per)])
                got = report["t_interval"][metric]
                assert got["n"] == 2 and got["level"] == 0.9
                assert all(abs(got[name] - value) < ROUNDING for name, value in
                           (("mean", mean), ("low", low), ("high", high))), (stem, metric, got)
            minutes = (440 + 560) / 25 / 60
            rate = report["possessions_per_graded_minute"]
            assert abs(rate["graded_minutes"] - minutes) < ROUNDING
            assert abs(rate["predicted"] - report["pooled"]["n_pred"] / minutes) < 1e-6
            assert abs(rate["true"] - report["pooled"]["n_gt"] / minutes) < 1e-6
            assert set(report["labels"]["true"]) == set("HAD") and set(report["labels"]["predicted"]) == set("HADX")
            for shares in report["labels"].values():
                assert abs(sum(shares.values()) - 1.0) < 10 * ROUNDING, shares

        # The first config is the reference: no comparison for it, and b - a against it for the others.
        assert reports["nearest_w50"]["versus"] is None
        for stem in ("nearest_w0", "nearest_w25"):
            versus = reports[stem]["versus"]
            assert versus["reference"]["stem"] == "nearest_w50" and len(versus["reference"]["config_sha256"]) == 64
            for metric in ("spell_f1_mean", "macro_f1"):
                other, first = reports[stem]["per_match"], reports["nearest_w50"]["per_match"]
                deltas = {match_id: other[match_id][metric] - first[match_id][metric] for match_id in other}
                got = versus[metric]
                assert all(abs(got["match_deltas"][match_id] - delta) < 2 * ROUNDING for match_id, delta in
                           deltas.items()), (stem, metric, got["match_deltas"], deltas)
                assert abs(got["mean_delta"] - np.mean(list(deltas.values()))) < 2 * ROUNDING
                assert abs(got["pooled_delta"] - (reports[stem]["pooled"][metric]
                                                  - reports["nearest_w50"]["pooled"][metric])) < 2 * ROUNDING
                assert got["n"] == 2 and 0.0 < got["p_sign_flip"] <= 1.0 and got["wins"] + got["losses"] <= 2
                assert got["by_competition"].keys() == {"other"}, "random matches are not DFL's: grouped as other"
                assert got["interval_excludes_zero"] == benchmark_run.interval_excludes_zero(got["low"], got["high"])

        # Another config beside it changes nothing of a report but its comparison.
        alone = evaluate_clean([paths[2]], matches=matches, data_dir=root, log=quiet)
        solo = alone["configs"][0]["report"]
        assert solo["versus"] is None
        assert {key: value for key, value in solo.items() if key != "versus"} == {
            key: value for key, value in reports["nearest_w25"].items() if key != "versus"}

        # The printout: the reference is named, the sign convention is said, and so is the interval's level.
        table = "\n".join(lines)
        for fragment in ("nearest_w50", "nearest_w0", "nearest_w25", "against nearest_w50", "other - nearest_w50",
                         "positive means", "90% t-interval", "sign-flip p", "spellF1", "macroF1", "alive/dead F1",
                         "pooled spellF1", "pooled macroF1", "pooled alive/dead F1"):
            assert fragment in table, (fragment, table)
        assert "beyond {:g} either way".format(score.FLIP_TOLERANCE) in table, "the tolerance is the code's, not a copy"
        assert benchmark_run.LEVEL == score.INTERVAL_LEVEL == 0.9 == reports["nearest_w50"]["t_interval"][
            "spell_f1_mean"]["level"]

        # The files: where they go, and what they are.
        written = benchmark_run.write_reports(result)
        folder = os.path.join(root, "runs", "benchmark", "clean")
        names = ["nearest_w50" + SUBSET + ".json", "nearest_w0" + SUBSET + ".json", "nearest_w25" + SUBSET + ".json"]
        assert written == [os.path.join(folder, name) for name in names], written
        assert sorted(os.listdir(folder)) == sorted(names), "no .tmp left"
        first_bytes = {}
        for path, entry in zip(written, configs):
            raw = read(path, "rb")
            first_bytes[path] = raw
            assert raw.endswith(b"\n") and b"\r" not in raw
            text = raw.decode("utf-8")
            parsed = strict_json(text)
            assert json.dumps(parsed, sort_keys=True, indent=2) + "\n" == text, "sorted, one format"
            assert parsed["pooled"]["n_gt"] == entry["report"]["pooled"]["n_gt"]
        assert benchmark_run.write_reports(result) == written
        assert all(read(path, "rb") == raw for path, raw in first_bytes.items()), "a rewrite changes nothing"
        other_folder = os.path.join(root, "elsewhere")
        assert benchmark_run.write_reports(result, data_dir=other_folder) == [
            os.path.join(other_folder, "runs", "benchmark", "clean", name) for name in names]


def test_runner_report_is_written_atomically():
    """The temporary file is renamed over the old report; if that fails, the old one is intact and no .tmp is left."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        result = evaluate_clean([yardstick_path(50)], matches=matches, data_dir=root, log=quiet)
        path = benchmark_run.write_reports(result)[0]
        assert os.path.basename(path) == "nearest_w50" + SUBSET + ".json", path
        good = read(path, "rb")
        folder = os.path.dirname(path)
        real_replace = os.replace

        def failing(source, destination):
            raise OSError("the disk is full")
        os.replace = failing
        try:
            result["configs"][0]["report"]["seeds"] = [99]          # a different report, which must not get in
            expect_error(OSError, lambda: benchmark_run.write_reports(result), "the disk is full")
        finally:
            os.replace = real_replace
        assert read(path, "rb") == good, "a failed write left a damaged report"
        assert os.listdir(folder) == [os.path.basename(path)], os.listdir(folder)


def test_runner_scores_against_the_coarse_truth_when_asked():
    """--truth coarse scores against truth.coarse_codes, says so, and files its report apart from the raw one."""
    matches = synthetic_matches()
    path = yardstick_path(25)
    with tempfile.TemporaryDirectory() as root:
        raw = evaluate_clean([path], matches=matches, data_dir=root, log=quiet)
        coarse = evaluate_clean([path], matches=matches, truth="coarse", data_dir=root, log=quiet)
        report = coarse["configs"][0]["report"]
        assert report["truth"] == {"name": "coarse", "min_s": truth.COARSE_MIN_S} == {"name": "coarse", "min_s": 5}
        assert raw["configs"][0]["report"]["truth"] == {"name": "raw", "min_s": None}
        halves = []
        for match in matches.values():
            codes_true = truth.coarse_codes(truth.truth_codes(match)[0], match.graded, match.fps)
            halves += independent_halves(match, path, root, raw=codes_true)
        assert [(half["n_gt"], half["n_pred"], tuple(half["tp"])) for half in report["halves"]] == [
            (h.n_gt, h.n_pred, h.tp) for h in halves]
        assert report["pooled"]["n_gt"] < raw["configs"][0]["report"]["pooled"]["n_gt"], (
            "the coarse truth absorbs short possessions, so it has fewer; this match must reach that branch")
        assert report["pooled"]["n_pred"] == raw["configs"][0]["report"]["pooled"]["n_pred"], "same predictions"
        names = [os.path.basename(p) for p in benchmark_run.write_reports(raw) + benchmark_run.write_reports(coarse)]
        assert names == ["nearest_w25" + SUBSET + ".json", "nearest_w25.coarse" + SUBSET + ".json"], names
        assert sorted(os.listdir(os.path.join(root, "runs", "benchmark", "clean"))) == sorted(names), (
            "the two reports must sit side by side, neither replacing the other")


def test_runner_clean_runs_one_seed_and_says_so():
    """The clean profile has nothing random to repeat: --seeds is ignored, with a note, and only seed 0 is run."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = evaluate_clean([yardstick_path(0)], seeds=3, matches=matches, data_dir=root,
                                log=lines.append)
        report = result["configs"][0]["report"]
        assert report["seeds"] == [0] and {half["seed"] for half in report["halves"]} == {0}
        assert len(report["halves"]) == 4
        assert any("seed" in line and "clean" in line and "ignored" in line for line in lines), lines
        quiet_lines = []
        default = evaluate_clean([yardstick_path(0)], matches=matches, data_dir=root,
                                 log=quiet_lines.append)
        assert not any("ignored" in line or "seed 0 only" in line for line in quiet_lines), (
            "no number of seeds was asked for, so there is nothing to say about ignoring one: " + str(quiet_lines))
        assert default["configs"][0]["report"] == report, "the default is the same run"
        one = evaluate_clean([yardstick_path(0)], seeds=1, matches=matches, data_dir=root,
                             log=quiet_lines.append)
        assert one["configs"][0]["report"] == report
        assert not any("ignored" in line for line in quiet_lines), "seeds=1 is what clean runs: nothing is ignored"


def test_runner_one_match_has_no_interval_and_says_so():
    """A t interval of one match is not a number: the report holds null, and the table says why, not NaN."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = evaluate_clean([yardstick_path(50), yardstick_path(25)], match_ids=["RND002"],
                                matches=matches, data_dir=root, log=lines.append)
        for entry in result["configs"]:
            interval = entry["report"]["t_interval"]["spell_f1_mean"]
            assert interval["n"] == 1 and interval["low"] is None and interval["high"] is None
            assert interval["mean"] == entry["report"]["pooled"]["spell_f1_mean"], "one match: the mean is its score"
            assert [record["id"] for record in entry["report"]["matches"]] == ["RND002"]
        versus = result["configs"][1]["report"]["versus"]["spell_f1_mean"]
        assert versus["low"] is None and versus["high"] is None and versus["n"] == 1
        assert versus["interval_excludes_zero"] is False
        table = "\n".join(lines)
        assert not re.search(r"\bnan\b", table.lower()) and "no interval" in table, table
        strict_json(json.dumps(result["configs"][1]["report"], sort_keys=True, allow_nan=False))


def test_runner_interval_excludes_zero_only_beyond_the_tie_tolerance():
    """compare()'s interval does not follow its tie rule, so the table must: it clears 0 by more than FLIP_TOLERANCE."""
    tolerance = score.FLIP_TOLERANCE
    clears = benchmark_run.interval_excludes_zero
    assert clears(0.01, 0.03) and clears(-0.03, -0.01)
    assert not clears(-0.01, 0.03) and not clears(0.0, 0.03) and not clears(-0.03, 0.0)
    # Float noise: an interval of 1e-16 around a delta of 1e-16 is above 0 on paper and is a tie for compare().
    assert not clears(1e-16, 3e-16) and not clears(-3e-16, -1e-16)
    assert not clears(tolerance, 1.0) and not clears(-1.0, -tolerance), "exactly at the tolerance is not beyond it"
    assert clears(2 * tolerance, 1.0) and clears(-1.0, -2 * tolerance)
    for none in ((None, None), (None, 1.0), (1.0, None), (float("nan"), float("nan")), (float("nan"), 1.0)):
        assert clears(*none) is False, none


# -- what it refuses --------------------------------------------------------------------------------------------------

def test_runner_rejects_unknown_profile():
    """The message names the profile asked for and lists the ones that exist, before any match is loaded."""
    matches = synthetic_matches()
    path = yardstick_path(50)
    with tempfile.TemporaryDirectory() as root:
        message = expect_error(benchmark_run.BenchmarkError, lambda: evaluate_clean(
            [path], profile="gappy", matches=matches, data_dir=root, log=quiet), "gappy", "not one of")
        for name in benchmark_run.PROFILES:
            assert name in message, (name, message)
        assert "clean" in message
        # The same through make_atd, which is what a later task routes to the degrader.
        match = matches["RND001"]
        expect_error(benchmark_run.BenchmarkError, lambda: benchmark_run.make_atd(match, "gappy", 0, None), "gappy",
                     "clean")
        atd = benchmark_run.make_atd(match, "clean", 0, None)
        assert all(getattr(atd, name) is getattr(match, field) for name, field in (
            ("players_xy", "players_xy"), ("ball_xy", "ball_xy"), ("slot_team", "player_team"))), "clean is DFL's own"
        # Before any match is loaded: with no matches to inject and an empty data folder, loading would raise OSError.
        expect_error(benchmark_run.BenchmarkError, lambda: evaluate_clean(
            [path], profile="gappy", data_dir=root, log=quiet), "gappy")


def test_runner_rejects_what_it_cannot_score():
    """Every refusal comes before a match is loaded, names the offender, and says what to do."""
    matches = synthetic_matches()
    path = yardstick_path(50)
    refused = benchmark_run.BenchmarkError
    with tempfile.TemporaryDirectory() as root:
        def evaluate(*configs, **keywords):
            keywords.setdefault("log", quiet)
            keywords.setdefault("data_dir", root)           # an empty folder: loading a real match would be an OSError
            return evaluate_clean(list(configs), **keywords)

        expect_error(refused, lambda: evaluate(), "no config", "CONFIG")
        expect_error(refused, lambda: evaluate(path, path), "nearest_w50", "twice")
        copy = write_text(root, "nearest_w50.toml", read(path))
        expect_error(refused, lambda: evaluate(path, copy), "nearest_w50", "same name", "report")
        expect_error(refused, lambda: evaluate(path, truth="smooth"), "smooth", "raw", "coarse")
        expect_error(refused, lambda: evaluate(path, seeds=0), "seeds", "at least 1")
        expect_error(refused, lambda: evaluate(path, seeds=True), "seeds")
        expect_error(refused, lambda: evaluate(path, match_ids=["J03WMX", "J03WMX"]), "J03WMX", "twice")
        expect_error(refused, lambda: evaluate(path, match_ids=["XXXXXX"]), "XXXXXX", "J03WMX")
        expect_error(refused, lambda: evaluate(path, match_ids=["RND009"], matches=matches), "RND009", "RND001")
        wrong = {"RND001": random_match("RND002")}                  # a match filed under the id of another
        expect_error(refused, lambda: evaluate(path, matches=wrong), "RND001", "RND002", "wrong id")
        expect_error(refused, lambda: evaluate(path, match_ids=[]), "no match")
        expect_error(refused, lambda: evaluate(path, calibration=object(), matches=matches), "clean", "calibration")
        # scope = "file" decodes across half time: refused up front, naming the config, with a loader that would fail.
        repro = os.path.join(ROOT, "pipeline", "configs", "smooth50_repro.toml")
        expect_error(refused, lambda: evaluate(path, repro), "smooth50_repro", 'scope = "file"', "halves")
        # A config the pipeline itself rejects is its own error, from the pipeline's loader, naming the file.
        broken = write_text(root, "broken.toml", 'scope = "halves"\n[[evidence]]\nkind = "no_such_source"\n'
                                                 '[decoder]\nkind = "majority_vote"\nhalf_width = 1\n')
        expect_error(ConfigError, lambda: evaluate(path, broken), "broken.toml", "no_such_source")
        assert not os.path.exists(os.path.join(root, "runs")), "a refused run writes nothing"


def test_runner_names_unavailable_source():
    """A source that needs the video fails the run, and the message names the config, the source and the asset.

    A benchmark match has no homography (the DFL data has no video); a component that asks for it raises
    AssetUnavailable. The runner finds which evidence source it was by running each on its own, and prints that source's
    name and kind with the asset. When no evidence source does it alone (a refiner did), it says that instead.
    """
    from pipeline import evidence as evidence_stage
    from pipeline import refine as refine_stage
    from pipeline.contracts import Evidence, no_opinion

    def needs_homography(run):
        run.inputs.homography
        return Evidence(alive=no_opinion(run.inputs.n_frames), home=no_opinion(run.inputs.n_frames))

    def refine_with_homography(ctx, decoded):
        ctx.run.inputs.homography
        return decoded

    text = ('scope = "halves"\n\n[[evidence]]\nkind = "nearest_player"\n\n[[evidence]]\nkind = "needs_homography"\n'
            'name = "pitch_lines"\n\n[decoder]\nkind = "majority_vote"\nhalf_width = 2\n')
    refiner = ('scope = "halves"\n\n[[evidence]]\nkind = "nearest_player"\n\n[decoder]\nkind = "majority_vote"\n'
               'half_width = 2\n\n[[refine]]\nkind = "straighten"\n')
    matches = synthetic_matches()
    before_evidence, before_refine = dict(evidence_stage.REGISTRY), dict(refine_stage.REGISTRY)
    with registered(evidence_stage.REGISTRY, needs_homography=needs_homography), \
            registered(refine_stage.REGISTRY, straighten=refine_with_homography):
        with tempfile.TemporaryDirectory() as root:
            video = write_text(root, "uses_video.toml", text)
            message = expect_error(benchmark_run.BenchmarkError, lambda: evaluate_clean(
                [yardstick_path(50), video], matches=matches, data_dir=root, log=quiet),
                "uses_video", "pitch_lines", "needs_homography", "homography", "public window")
            assert "nearest_player" not in message, "the source that worked is not the one to blame: " + message
            assert not os.path.exists(os.path.join(root, "runs")), "nothing is written for a run that failed"

            late = write_text(root, "late_video.toml", refiner)
            message = expect_error(benchmark_run.BenchmarkError, lambda: evaluate_clean(
                [late], matches=matches, data_dir=root, log=quiet), "late_video", "homography", "public window")
            assert "no evidence source" in message and "majority_vote" in message, message
    assert evidence_stage.REGISTRY == before_evidence and refine_stage.REGISTRY == before_refine, (
        "the registries must be as they were, or a later check would see the temporary components")
    # The pipeline's own message, through the inputs, is the one that is quoted.
    expect_error(match_inputs.AssetUnavailable, lambda: match_inputs.MatchInputs(
        matches["RND001"], match_inputs.clean_atd(matches["RND001"])).homography, "homography", "P01")


def test_runner_a_subset_never_replaces_a_full_report():
    """A run on some of the matches is filed under its own name, so the full report of the config survives it.

    A report of all seven matches is `<stem>.json` (or `<stem>.coarse.json`). Any other set of matches adds
    `.m-<ids sorted, joined by "-"` before `.json`, whatever order they were asked for in; all seven in another order
    are still all seven. Run on one match, `--matches J03WMX` used to replace R002's seven-match report.
    """
    def serve(match_id):
        return random_match(match_id, seed=sources.MATCH_IDS.index(match_id))

    path = yardstick_path(25)
    with tempfile.TemporaryDirectory() as root, loading(serve):
        def written(truth_name="raw", **keywords):
            result = evaluate_clean([path], truth=truth_name, data_dir=root, log=quiet, **keywords)
            [report_file] = benchmark_run.write_reports(result)
            return os.path.basename(report_file)

        assert written() == "nearest_w25.json"
        full = os.path.join(root, "runs", "benchmark", "clean", "nearest_w25.json")
        before = read(full, "rb")
        assert written(match_ids=["J03WMX"]) == "nearest_w25.m-J03WMX.json"
        assert written(match_ids=["J03WN1", "J03WMX"]) == "nearest_w25.m-J03WMX-J03WN1.json"
        assert written(match_ids=["J03WMX", "J03WN1"]) == "nearest_w25.m-J03WMX-J03WN1.json", "order is not in the name"
        assert written("coarse", match_ids=["J03WMX"]) == "nearest_w25.coarse.m-J03WMX.json"
        six = [match_id for match_id in sources.MATCH_IDS if match_id != "J03WR9"]
        assert written(match_ids=six) == "nearest_w25.m-" + "-".join(sorted(six)) + ".json", "six is not seven"
        assert read(full, "rb") == before, "a subset replaced the full report"
        assert written(match_ids=list(reversed(sources.MATCH_IDS))) == "nearest_w25.json", "all seven, in any order"
        assert written("coarse") == "nearest_w25.coarse.json"
        assert len(os.listdir(os.path.dirname(full))) == 6, sorted(os.listdir(os.path.dirname(full)))
    # The name by itself: no ids given is no claim about the matches, and the registry's own ids are all seven.
    assert os.path.basename(benchmark_run.report_path("d", "clean", "s", "raw")) == "s.json"
    assert os.path.basename(
        benchmark_run.report_path("d", "clean", "s", "coarse", ["J03WMX"])) == "s.coarse.m-J03WMX.json"
    assert os.path.basename(benchmark_run.report_path("d", "gaps", "s", "raw", sources.MATCH_IDS)) == "s.json"


def test_runner_report_names_keep_runs_of_different_kinds_apart():
    """A file name says what makes a run another run, so that no run replaces a report of another kind.

    `<stem>[.<truth>][.m-<ids>][.s<n>].json`: the truth when it is not the raw one, the matches when they are not all
    seven, and the number of degradation seeds when it is not the default (a one-seed run, a quick look, never replaces
    the three-seed report that a result is quoted from). The clean profile has nothing random: it runs seed 0 whatever
    it was asked, so its name never carries a seed count. Through the runner too: the file that was written.
    """
    default = benchmark_run.DEFAULT_SEEDS
    assert default == 3

    def name(profile, truth="raw", ids=None, seeds=None):
        return os.path.basename(benchmark_run.report_path("d", profile, "s", truth, ids, seeds))

    assert name("gaps") == "s.json" and name("gaps", seeds=default) == "s.json", "the default number is not named"
    assert name("gaps", seeds=1) == "s.s1.json" and name("gaps", seeds=5) == "s.s5.json"
    assert name("gaps", "coarse", ["J03WMX"], 5) == "s.coarse.m-J03WMX.s5.json", "truth, matches, seeds, in that order"
    assert name("gaps", "coarse", seeds=1) == "s.coarse.s1.json"
    assert name("clean", seeds=1) == name("clean", seeds=5) == name("clean") == "s.json", "clean has one seed"
    names = {name("gaps", truth, ids, seeds) for truth in ("raw", "coarse") for ids in (None, ["J03WMX"])
             for seeds in (None, 1, 2)}
    assert len(names) == 2 * 2 * 3, ("every kind of run has a name of its own", sorted(names))

    def serve(match_id):
        return random_match(match_id, seed=sources.MATCH_IDS.index(match_id))

    with tempfile.TemporaryDirectory() as root, loading(serve):
        result = evaluate_clean([yardstick_path(25)], profile="clean", seeds=5, data_dir=root,
                                log=quiet)
        assert [os.path.basename(path) for path in benchmark_run.write_reports(result)] == ["nearest_w25.json"]


def test_runner_refuses_a_config_whose_name_has_a_dot():
    """A config's file name is its report's name, whose parts are told apart by dots: a dot in it would be one.

    `x.coarse.toml` would write `x.coarse.json`, which is `x`'s own report under the coarse truth, and `x.m-J03WMX.toml`
    one of `x`'s reports on a single match: another run's file. Refused before a match is loaded, naming the config
    and saying to rename it.
    """
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        for stem in ("x.coarse", "x.m-J03WMX", "x.s1", "a.b"):
            path = write_text(root, stem + ".toml", read(yardstick_path(50)))
            expect_error(benchmark_run.BenchmarkError, lambda: evaluate_clean(
                [path], profile="clean", matches=matches, data_dir=root, log=quiet), stem, "dot", "rename")
        assert not os.path.exists(os.path.join(root, "runs")), "a refused run writes nothing"
        fine = write_text(root, "x_coarse-1.toml", read(yardstick_path(50)))
        evaluate_clean([fine], profile="clean", matches=matches, data_dir=root, log=quiet)


def test_runner_readme_names_checks_and_numbers_that_exist():
    """What the benchmark README quotes of the code is what the code has: the checks it names, the tolerance, the level.

    A stale name is how a document claims a check that no longer exists, or never did. Every test_* name in the README
    must be a check the harness collects; the tolerance and the interval level are the ones in score.py; the file naming
    for a subset and the MIN_AGREEMENT that it points to are there.
    """
    from pipeline.benchmark import selftest
    readme = read(os.path.join(ROOT, "pipeline", "benchmark", "README.md"))
    known = {name for name, _ in selftest.collect()}
    named = set(re.findall(r"\btest_[a-z0-9_]+\b", readme))
    assert "test_dfl_oracle_scores_perfectly" in named and named <= known, sorted(named - known)
    assert "{:g}".format(score.FLIP_TOLERANCE) in readme and "score.FLIP_TOLERANCE" in readme
    assert "{:.0%}".format(score.INTERVAL_LEVEL) in readme
    assert ".m-" in readme and "MIN_AGREEMENT" in readme and MIN_AGREEMENT == 0.6 and "60%" in readme


# -- the degraded profile ---------------------------------------------------------------------------------------------

def independent_gaps_halves(match, config_path, cal, seed, data_dir):
    """[HalfScore] of one config on one match under `gaps`: degrade, predict and score_halves called here."""
    config = load_config(config_path)
    codes_true = truth.truth_codes(match)[0]
    atd, _ = degrade.degrade(degrade.tracking_view(match), "gaps", cal, seed)
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=seed, data_dir=data_dir)
    codes, _ = benchmark_run.predict(config, inputs)
    return score.score_halves(codes_true, codes, match.graded, match.match_id, seed)


def half_tuples(halves):
    return [(half["match_id"], half["half"], half["seed"], half["n_gt"], half["n_pred"], tuple(half["tp"]),
             tuple(tuple(row) for row in half["confusion"])) for half in halves]


def test_runner_gaps_runs_on_degraded_tracking():
    """The gaps profile serves each match's degraded tracking, per seed, and the report says whose.

    Every number is checked against the degrader, predict and score_halves called here: a seed degrades the match with
    its own draws, the report holds the Calibration's digest and version, the file name carries `.s<n>` for a number of
    seeds that is not the default, and the printout names the calibration that made the run.
    """
    assert benchmark_run.PROFILES == degrade.PROFILES == {"clean": 1, "gaps": 1, "tracking": 3, "context": 1}
    matches, cal = synthetic_matches(), synthetic_calibration()
    paths = [yardstick_path(50), yardstick_path(0)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = evaluate_gaps(paths, seeds=2, matches=matches, calibration=cal, data_dir=root, log=lines.append)
        assert result["profile"] == {"name": "gaps", "version": 1} and result["seeds"] == [0, 1]
        for entry, width in zip(result["configs"], (50, 0)):
            report = entry["report"]
            assert report["profile"] == {"name": "gaps", "version": 1} and report["seeds"] == [0, 1]
            assert report["calibration"] == {"sha256": calibration.digest(cal),
                                             "version": calibration.CALIBRATION_VERSION}
            expected = []
            for match in matches.values():
                for seed in (0, 1):
                    expected += independent_gaps_halves(match, yardstick_path(width), cal, seed, root)
            assert len(report["halves"]) == 8 and half_tuples(report["halves"]) == [
                (h.match_id, h.half, h.seed, h.n_gt, h.n_pred, h.tp, h.confusion) for h in expected], width
            for name, value in score.pooled(expected).items():
                if name != "spell_f1":
                    assert abs(report["pooled"][name] - value) < ROUNDING, (width, name)
            minutes = (440 + 560) / 25 / 60 * 2
            assert abs(report["possessions_per_graded_minute"]["graded_minutes"] - minutes) < ROUNDING, "two seeds"
        table = "\n".join(lines)
        assert "profile   gaps (version 1)" in table and "seeds 0,1" in table, table
        assert any("calibration" in line and calibration.digest(cal)[:12] in line for line in lines), table

        # The same run is the same report; a different seed count is a different run, with its own name.
        again = evaluate_gaps(paths, seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert [entry["report"] for entry in again["configs"]] == [entry["report"] for entry in result["configs"]]
        three = evaluate_gaps(paths[:1], matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert three["seeds"] == [0, 1, 2] == three["configs"][0]["report"]["seeds"], "none asked for: DEFAULT_SEEDS"
        one = evaluate_gaps(paths[:1], seeds=1, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert one["seeds"] == [0]
        two = evaluate_gaps(paths[:1], seeds=2, matches=matches, calibration=cal, data_dir=root, log=quiet)
        names = [os.path.basename(benchmark_run.write_reports(each)[0]) for each in (two, three, one)]
        assert names == ["nearest_w50" + SUBSET + ".s2.json", "nearest_w50" + SUBSET + ".json",
                         "nearest_w50" + SUBSET + ".s1.json"], names
        folder = os.path.join(root, "runs", "benchmark", "gaps")
        assert sorted(os.listdir(folder)) == sorted(names), "three runs of three kinds, three files"
        report = strict_json(read(os.path.join(folder, names[0])))
        assert report["calibration"]["sha256"] == calibration.digest(cal) and report["seeds"] == [0, 1]

        # The profile does something: the same config on the clean tracking scores differently.
        clean = evaluate_clean(paths[:1], matches=matches, data_dir=root, log=quiet)
        assert clean["configs"][0]["report"]["pooled"] != result["configs"][0]["report"]["pooled"]


def test_hmm_runs_on_the_benchmark():
    """The hmm decoder is a decoder like any other to the runner: its config scores beside a yardstick's, under gaps.

    Two configs, nearest_w50 first (the reference) and an hmm one, on the synthetic matches with the synthetic
    Calibration: two scored configs come back, each with a half for every graded half and seed, and the hmm one is
    compared against the yardstick. What the scores are means nothing on random matches; that a decoder with a
    parameter table of its own reaches the benchmark's scoring is what is held here.
    """
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        paths = [yardstick_path(50), write_text(root, "hmm_trial.toml", HMM)]
        result = evaluate_gaps(paths, seeds=1, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert [entry["stem"] for entry in result["configs"]] == ["nearest_w50", "hmm_trial"], result["configs"]
        for entry in result["configs"]:
            assert len(entry["scores"]) == 4 and len(entry["report"]["halves"]) == 4, entry["stem"]
            assert 0.0 <= entry["report"]["pooled"]["spell_f1_mean"] <= 1.0, entry["stem"]
        hmm_report = result["configs"][1]["report"]
        assert result["configs"][0]["report"]["versus"] is None
        assert hmm_report["versus"]["reference"]["stem"] == "nearest_w50"
        assert hmm_report["config"]["text"] == HMM.replace("\r", ""), "the report holds the config as written"


def test_shift_edges_runs_on_the_benchmark():
    """A refiner is a component like any other to the runner: a config that carries shift_edges is scored like one.

    Two configs, the plain one first (the reference) and the same one with shift_edges after its decoder, on the
    synthetic matches with the synthetic Calibration under gaps: two scored configs come back, each with a half for
    every graded half and seed, the shifted one is compared against the plain one, its report holds the config as
    written, and its frames are not the plain one's in at least one half (the refiner ran inside the runner's
    prediction and its labels reached the scorer). The config is P06's shape, which decodes all three labels on these
    matches: HMM above decodes D throughout there, and a shift of that would move nothing. What the scores are means
    nothing on random matches.
    """
    matches, cal = synthetic_matches(), synthetic_calibration()
    with tempfile.TemporaryDirectory() as root:
        paths = [write_text(root, "p06_plain.toml", P06_TRIAL_CONFIG),
                 write_text(root, "p06_shifted.toml", P06_TRIAL_CONFIG + SHIFT)]
        result = evaluate_gaps(paths, seeds=1, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert [entry["stem"] for entry in result["configs"]] == ["p06_plain", "p06_shifted"], result["configs"]
        for entry in result["configs"]:
            assert len(entry["scores"]) == 4 and len(entry["report"]["halves"]) == 4, entry["stem"]
            assert 0.0 <= entry["report"]["pooled"]["spell_f1_mean"] <= 1.0, entry["stem"]
        plain, shifted = (entry["report"] for entry in result["configs"])
        assert plain["versus"] is None and shifted["versus"]["reference"]["stem"] == "p06_plain"
        assert shifted["config"]["text"] == (P06_TRIAL_CONFIG + SHIFT).replace("\r", ""), \
            "the report holds the config as written"
        assert any(a["confusion"] != b["confusion"] for a, b in zip(plain["halves"], shifted["halves"])), (
            "the shifted config scored the same frames in every half: the refiner did not reach the scorer")


def test_runner_default_profile_is_tracking_and_runs_it():
    """With no --profile the runner degrades with the tracking profile: its version 3, its Calibration, its report.

    The default is the profile that looks most like our match, the one whose gates all pass (R004c; context is not it,
    R005); a run that does not name one is the same run as one that names tracking, to the byte of its reports, and says
    tracking version 3 in the report and in the printout.
    The synthetic Calibration here has the made-up tracking figures (common.tracking_calibration), whose counts and
    masks are as long as each other."""
    assert benchmark_run.DEFAULT_PROFILE == "tracking" and benchmark_run.PROFILES["tracking"] == 3
    matches, cal = synthetic_matches(), tracking_calibration()
    paths = [yardstick_path(50), yardstick_path(0)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        default = benchmark_run.evaluate_configs(paths, seeds=2, matches=matches, calibration=cal, data_dir=root,
                                                 log=lines.append)
        named = benchmark_run.evaluate_configs(paths, profile="tracking", seeds=2, matches=matches, calibration=cal,
                                               data_dir=root, log=quiet)
        assert default["profile"] == {"name": "tracking", "version": 3}
        assert [entry["report"] for entry in default["configs"]] == [entry["report"] for entry in named["configs"]]
        assert all(entry["report"]["profile"] == {"name": "tracking", "version": 3} for entry in default["configs"])
        assert "profile   tracking (version 3)" in "\n".join(lines), lines


# -- what the profile says of itself
# ------------------------------------------------------------------------------------

def run_on(profile, cal, paths, root, lines, seeds=1):
    """evaluate_configs on `profile` with the synthetic matches and `cal`, its printout collected in `lines`."""
    return benchmark_run.evaluate_configs(paths, profile=profile, seeds=seeds, matches=synthetic_matches(),
                                          calibration=cal, data_dir=root, log=lines.append)


def test_runner_labels_context_as_sensitivity_only_and_says_where_each_profiles_gaps_come_from():
    """A profile that fails its gates is not offered as the yardstick: the help, the printout and the report say so.

    `context` is in SENSITIVITY_ONLY with the reason (R005: two of its gates fail), and the --profile help, a `note`
    line of the printout and a "sensitivity_only" key of the report carry it; `tracking`, the default, carries none of
    them, and its report has no such key. The line that names the Calibration says where the ball's gaps came from,
    which differs by profile: the transplant of ours for gaps and tracking (and tracking's tracklets, slots and counts
    besides), the hazard fitted on our match for context, whose gaps are drawn and not transplanted; the old line, that
    the gaps are 'as measured by calibration.py', was false for context. The report of a context run says which version
    of tracking its players are built from."""
    assert benchmark_run.SENSITIVITY_ONLY.keys() == {"context"} and benchmark_run.DEFAULT_PROFILE == "tracking"
    reason = benchmark_run.SENSITIVITY_ONLY["context"]
    assert "R005" in reason and "two of its gates" in reason and "sensitivity check" in reason, reason
    assert benchmark_run.PROVENANCE.keys() == set(degrade.PROFILES) - {degrade.CLEAN}, (
        "a line for every degraded profile")
    path = [yardstick_path(50)]
    calibrations = {"gaps": synthetic_calibration(), "tracking": tracking_calibration(),
                    "context": context_calibration()}
    with tempfile.TemporaryDirectory() as root:
        lines, reports, calibration_lines = {}, {}, {}
        for profile, cal in calibrations.items():
            lines[profile] = []
            result = run_on(profile, cal, path, root, lines[profile])
            reports[profile] = result["configs"][0]["report"]
            calibration_lines[profile] = [line for line in lines[profile] if line.startswith("calibration")]
            assert (len(calibration_lines[profile]) == 1
                    and calibration.digest(cal)[:12] in calibration_lines[profile][0])
        # The note: for context, with the reason; for the others, nothing.
        notes = [line for line in lines["context"] if line.startswith("note")]
        assert notes == ["note      profile context {}".format(reason)], notes
        for profile in ("gaps", "tracking"):
            assert not [line for line in lines[profile] if "sensitivity" in line or line.startswith("note")], profile
            assert "sensitivity_only" not in reports[profile], profile
        assert reports["context"]["sensitivity_only"] == reason
        assert reports["context"]["profile"] == {"name": "context", "version": 1, "tracking_version": 3}
        assert reports["tracking"]["profile"] == {"name": "tracking", "version": 3}
        # Where the gaps come from, by profile.
        gaps, tracking, context = (calibration_lines[profile][0] for profile in ("gaps", "tracking", "context"))
        assert len({gaps, tracking, context}) == 3, "one line for each profile"
        assert "transplanted" in gaps and "transplanted" in tracking and "tracklets" in tracking, (gaps, tracking)
        assert "tracklets" not in gaps and "hazard" not in gaps and "hazard" not in tracking, (gaps, tracking)
        assert "hazard" in context and "A08" in context and "transplanted" not in context, context
        assert "as measured by calibration.py" not in context, "false for context: its gaps are drawn"
        # The help carries the note for context and for no other profile.
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            try:
                benchmark_run.main(["--help"])
            except SystemExit as stop:
                assert stop.code == 0, stop.code
        helped = " ".join(shown.getvalue().split())
        assert "context " + reason in helped and helped.count("sensitivity") == 1, helped
        assert benchmark_run.DEFAULT_PROFILE + " " + reason not in helped and "(default tracking)" in helped, helped


def test_runner_report_is_in_match_id_order_whatever_order_the_matches_are_given():
    """[RND001, RND002] and [RND002, RND001] write the same bytes under the same name; each match runs when it is asked.

    The order of --matches is the order the matches are run in (one in memory at a time), and not something a report
    may depend on: its records, its halves and so everything made from them are in match-id order, a degraded profile's
    seeds in order within a match. The clean and the gaps profile (two seeds, so that the order within a match is
    seen), two configs so that the comparison is in it too."""
    matches = synthetic_matches()
    paths = [yardstick_path(50), yardstick_path(0)]
    with tempfile.TemporaryDirectory() as root:
        for profile, cal, seeds in (("clean", None, None), ("gaps", synthetic_calibration(), 2)):
            written = {}
            for order in (["RND001", "RND002"], ["RND002", "RND001"]):
                out = os.path.join(root, profile + "_" + "_".join(order))
                result = benchmark_run.evaluate_configs(paths, profile=profile, seeds=seeds, match_ids=order,
                                                        matches=matches, calibration=cal, data_dir=root, log=quiet)
                written[tuple(order)] = {os.path.relpath(path, out): read(path, "rb")
                                         for path in benchmark_run.write_reports(result, data_dir=out)}
                assert [record["id"] for record in result["matches"]] == ["RND001", "RND002"], (profile, order)
            first, second = written.values()
            assert first.keys() == second.keys() and len(first) == 2, (profile, sorted(first), sorted(second))
            assert first == second, "{}: the bytes of a report depend on the order of the matches".format(profile)
            report = strict_json(next(iter(first.values())).decode("utf-8"))
            expected = [("RND00" + str(number), seed, half) for number in (1, 2) for seed in report["seeds"]
                        for half in (1, 2)]
            assert [(half["match_id"], half["seed"], half["half"]) for half in report["halves"]] == expected
            assert [record["id"] for record in report["matches"]] == ["RND001", "RND002"]


def test_runner_refuses_a_calibration_it_cannot_use():
    """A degraded profile needs a Calibration (loaded from the data directory if none is given), clean takes none."""
    matches, cal = synthetic_matches(), synthetic_calibration()
    path = yardstick_path(50)
    refused = benchmark_run.BenchmarkError
    with tempfile.TemporaryDirectory() as root:
        expect_error(refused, lambda: benchmark_run.evaluate_configs(
            [path], profile="gaps", calibration=object(), matches=matches, data_dir=root, log=quiet),
            "calibration", "Calibration", "calibration.load()")
        expect_error(refused, lambda: evaluate_clean(
            [path], calibration=cal, matches=matches, data_dir=root, log=quiet), "clean", "calibration", "ignored")
        # None given: it is measured from, or read out of, the data directory, which here has nothing in it.
        try:
            benchmark_run.evaluate_configs([path], profile="gaps", matches=matches, data_dir=root, log=quiet)
        except OSError as error:
            assert "tactical" in str(error), str(error)
        else:
            raise AssertionError("an empty data directory has no calibration, and the run must say so")
        assert not os.path.exists(os.path.join(root, "runs")), "a refused run writes nothing"


def test_runner_names_the_order_of_the_published_baselines_and_how_far_a_run_agrees():
    """Kendall's tau of the order a run gives the four yardsticks against the published one, in the result and table."""
    matches, cal = synthetic_matches(), synthetic_calibration()
    paths = [yardstick_path(width) for width in (50, 125, 25, 0)]
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = evaluate_gaps(paths, seeds=1, matches=matches, calibration=cal, data_dir=root, log=lines.append)
        rank = result["rank"]
        assert rank["published"] == ["nearest_w50", "nearest_w125", "nearest_w25", "nearest_w0"]
        pooled = {entry["stem"]: score.pooled(entry["scores"])["spell_f1_mean"] for entry in result["configs"]}
        assert rank["observed"] == sorted(pooled, key=lambda stem: -pooled[stem]), "best first, by pooled spellF1"
        again = score.kendall_tau(rank["published"], pooled)
        assert rank["kendall_tau"] == again["tau"] and rank["pairs"] == 6
        assert any("Kendall" in line and "{:+.4f}".format(again["tau"]) in line for line in lines), lines
        # Fewer than two of the published four: nothing to say about their order.
        solo = evaluate_gaps(paths[:1], seeds=1, matches=matches, calibration=cal, data_dir=root, log=quiet)
        assert solo["rank"] is None


# -- one match at a time, and in what order ---------------------------------------------------------------------------

def test_runner_holds_one_match_at_a_time_and_in_the_order_given():
    """The next match is not loaded until the last one is gone, and matches run in the order they were asked for.

    A benchmark match is about 150 MB of arrays, and seven at once would be a gigabyte that nothing needs. The loader
    here hands out random matches and keeps weak references to each one and to its largest arrays; at every load after
    the first, all earlier ones must already be collected (after a gc.collect(), so that a reference cycle is not
    what is being asked about).
    """
    made = []

    def serve(match_id):
        gc.collect()
        alive = [name for name, refs in made if any(ref() is not None for ref in refs)]
        assert not alive, "{} is still in memory while {} is loaded".format(alive, match_id)
        match = random_match(match_id, seed=len(made))
        made.append((match_id, [weakref.ref(match), weakref.ref(match.players_xy), weakref.ref(match.ball_xy),
                                weakref.ref(match.ball_possession)]))
        return match

    ids = ["J03WOH", "J03WMX", "J03WR9"]
    lines = []
    with tempfile.TemporaryDirectory() as root, loading(serve):
        result = evaluate_clean([yardstick_path(50), yardstick_path(0)], match_ids=ids, data_dir=root,
                                log=lines.append)
    by_match = [line for line in lines if "spellF1 by match" in line]
    deltas = [line for line in lines if line.strip().startswith("per match") and "spellF1 mean" not in line]
    assert len(by_match) == 2 and len(deltas) == 1, lines
    for line in by_match + deltas:
        shown = re.findall(r"J03W[A-Z0-9]{2}", line)
        assert shown == sorted(ids), "scores by match and deltas by match are both in id order: " + line
    assert [name for name, _ in made] == ids, "each match is loaded once, in the order given"
    for entry in result["configs"]:
        # The report is in match-id order, whatever order the matches were run in (see the check of that name).
        assert [record["id"] for record in entry["report"]["matches"]] == sorted(ids)
        assert [half["match_id"] for half in entry["report"]["halves"]] == [i for i in sorted(ids) for _ in (1, 2)]
    gc.collect()
    assert all(ref() is None for _, refs in made for ref in refs), "a match is still held after the run"

    # Without ids: sources.MATCH_IDS' order.
    made.clear()
    with tempfile.TemporaryDirectory() as root, loading(serve):
        evaluate_clean([yardstick_path(25)], data_dir=root, log=quiet)
    assert [name for name, _ in made] == list(sources.MATCH_IDS), [name for name, _ in made]


# -- the command line -------------------------------------------------------------------------------------------------

def test_runner_main_runs_prints_and_writes():
    """main() on two configs: the exit code, the printout, the files, --no-write, --truth, and the failures."""
    def serve(match_id):
        return random_match(match_id, seed=sources.MATCH_IDS.index(match_id))

    first, second = yardstick_path(50), yardstick_path(25)
    with tempfile.TemporaryDirectory() as root, loading(serve):
        def run_main(*arguments):
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                code = benchmark_run.main(["--profile", "clean"] + list(arguments))   # a later --profile wins
            return code, shown.getvalue()

        code, out = run_main(first, second, "--data-dir", root, "--matches", "J03WMX", "J03WN1", "J03WOH")
        assert code == 0, out
        folder = os.path.join(root, "runs", "benchmark", "clean")
        subset = ".m-J03WMX-J03WN1-J03WOH"
        assert sorted(os.listdir(folder)) == ["nearest_w25" + subset + ".json", "nearest_w50" + subset + ".json"], out
        assert "nearest_w50" + subset + ".json" in out and "against nearest_w50" in out and "3 matches" in out, out
        assert "ignored" not in out, "no --seeds was given: nothing to say about seeds"
        report = strict_json(read(os.path.join(folder, "nearest_w25" + subset + ".json")))
        assert [record["id"] for record in report["matches"]] == ["J03WMX", "J03WN1", "J03WOH"]
        assert report["versus"]["reference"]["stem"] == "nearest_w50" and report["seeds"] == [0]

        code, out = run_main(first, "--data-dir", root, "--matches", "J03WMX", "--truth", "coarse", "--seeds", "5")
        assert code == 0 and "ignored" in out, out
        assert sorted(os.listdir(folder)) == sorted([
            "nearest_w25" + subset + ".json", "nearest_w50" + subset + ".json", "nearest_w50.coarse.m-J03WMX.json"])

        other = os.path.join(root, "nowhere")
        code, out = run_main(first, "--data-dir", other, "--matches", "J03WMX", "--no-write")
        assert code == 0 and not os.path.exists(other), "--no-write writes nothing, not even the folder"

        for arguments, fragment in (((first, "--profile", "gappy"), "clean"),
                                    ((first, "--matches", "J03XXX"), "J03XXX"),
                                    ((os.path.join(root, "absent.toml"),), "absent.toml")):
            code, out = run_main(*arguments, "--data-dir", other)
            assert code == 1 and out.startswith("FAILED") and fragment in out, (arguments, out)
            assert "Traceback" not in out and not os.path.exists(other)


# -- the same bytes in another process --------------------------------------------------------------------------------

def cross_process_matches(root, paths):
    """The matches of the determinism check: one served through the DFL cache of `root`, and two random ones."""
    synthetic = whole_positions(dfl.load_match("SYN001", data_dir=root, paths=paths))
    return {"SYN001": synthetic, **synthetic_matches()}


# The Calibration that each degraded profile's determinism check runs with. A process cannot be handed one, so each
# builds its own with the function of this table, which is a check of that function's determinism as well, since the
# report holds the Calibration's digest: the bundle's (gaps, which reads only the masks), with the made-up tracklets,
# counts, episodes and NaN-speed runs (tracking), and with a hazard of its own making besides (context).
CROSS_PROCESS_CALIBRATIONS = {"gaps": synthetic_calibration, "tracking": tracking_calibration,
                              "context": context_calibration}


def cross_process_run(root, out, paths, profile="clean", calibration=None):
    """What each process of the determinism checks does: evaluate on `profile` with `calibration`, and write to its own
    folder. The clean profile takes none; a degraded one is refused without one."""
    configs = [yardstick_path(width) for width in (50, 0, 125)]
    if profile == "clean":
        result = evaluate_clean(configs, matches=cross_process_matches(root, paths), data_dir=root, log=quiet)
    else:
        if calibration is None:
            raise ValueError("profile {!r} needs a Calibration; pass CROSS_PROCESS_CALIBRATIONS[{!r}]()".format(
                profile, profile))
        result = benchmark_run.evaluate_configs(configs, profile=profile, seeds=2, calibration=calibration,
                                                matches=cross_process_matches(root, paths), data_dir=root, log=quiet)
    return benchmark_run.write_reports(result, data_dir=out)


def test_runner_json_identical_across_processes():
    """The same run in three processes with different PYTHONHASHSEED (1, 2 and 3), and in this one, writes the same
    bytes.

    A set iterated anywhere between the scores and the report, or a dict built in hash order, would show here as
    another order of keys or of matches. The synthetic DFL match is served through the cache, which is built here
    first, so that the processes only read it and never build the same files at once.
    """
    report = identical_across_processes("clean", ".m-RND001-RND002-SYN001.json")
    assert [record["id"] for record in report["matches"]] == ["RND001", "RND002", "SYN001"], "in match-id order"
    assert report["matches"][2]["sources"] is not None, "the synthetic DFL match names its files"
    assert report["calibration"] is None and report["seeds"] == [0]


def test_runner_gaps_json_identical_across_processes():
    """The default profile too: the degradation, its draws and the Calibration's digest are the same in every process.

    Two seeds of the gaps profile on the same three matches, with the Calibration of the synthetic bundle measured by
    each process for itself. The reports hold the digest of it, so a measurement that depended on the process (a set, a
    hash seed, the order of a directory) would show there as well as in the scores.
    """
    report = identical_across_processes("gaps", ".m-RND001-RND002-SYN001.s2.json")
    assert [record["id"] for record in report["matches"]] == ["RND001", "RND002", "SYN001"]
    assert report["seeds"] == [0, 1] and len(report["halves"]) == 3 * 2 * 2
    assert report["calibration"]["sha256"] == calibration.digest(synthetic_calibration())


def test_runner_tracking_json_identical_across_processes():
    """The tracking profile too: the same bytes from three processes with PYTHONHASHSEED 1, 2 and 3, and from this one.

    The players' degradation (the walk, the episodes, the cap, the shortest-piece rule, the slots, the NaN runs and the
    smoothing) draws from generators keyed by (seed, match, component, half) and walks dicts and sets of players and
    pieces: an order that came from a hash would show here as another slot, another score or another digest. The
    Calibration is the made-up one of tracking_calibration, which each process builds for itself (its digest is in the
    reports, so a measurement that depended on the process would show too), on the same three matches and two seeds."""
    report = identical_across_processes("tracking", ".m-RND001-RND002-SYN001.s2.json")
    assert [record["id"] for record in report["matches"]] == ["RND001", "RND002", "SYN001"]
    assert report["seeds"] == [0, 1] and len(report["halves"]) == 3 * 2 * 2
    assert report["profile"] == {"name": "tracking", "version": 3}
    assert report["calibration"]["sha256"] == calibration.digest(tracking_calibration())


def test_runner_context_json_identical_across_processes():
    """The context profile too: three processes with PYTHONHASHSEED 1, 2 and 3 and this one write the same bytes.

    On top of the tracking profile's players, the ball's gaps come from the hazard: the features, the chance, the class
    and the third of every frame, the durations and the draws of the sampler, which no hash may order. The Calibration
    is context_calibration's, a made-up hazard on the made-up tracking figures, built by each process for itself."""
    report = identical_across_processes("context", ".m-RND001-RND002-SYN001.s2.json")
    assert [record["id"] for record in report["matches"]] == ["RND001", "RND002", "SYN001"]
    assert report["seeds"] == [0, 1] and len(report["halves"]) == 3 * 2 * 2
    assert report["profile"] == {"name": "context", "version": 1, "tracking_version": 3}
    assert report["sensitivity_only"] == benchmark_run.SENSITIVITY_ONLY["context"]
    assert report["calibration"]["sha256"] == calibration.digest(context_calibration())


def identical_across_processes(profile, suffix):
    """Run cross_process_run on `profile` in the processes of HASH_SEEDS, each with its own PYTHONHASHSEED, and here:
    the one report.

    Asserts that every file is the same bytes in the three places, and returns the parsed report of the last file by
    name.
    """
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "source"))
        paths = write_synthetic_dfl(os.path.join(root, "source"))
        assert dfl.load_match("SYN001", data_dir=root, paths=paths).report["cache"] == "built"       # warms the cache
        folders = {}
        for hash_seed in HASH_SEEDS:
            out = os.path.join(root, "out_" + hash_seed)
            code = ("import sys, json; from pipeline.benchmark.checks.run import CROSS_PROCESS_CALIBRATIONS as made, "
                    "cross_process_run as go; "
                    "go(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], "
                    "made[sys.argv[4]]() if sys.argv[4] in made else None)")
            done = subprocess.run([sys.executable, "-c", code, root, out, json.dumps(paths), profile], cwd=ROOT,
                                  env=dict(os.environ, PYTHONHASHSEED=hash_seed), capture_output=True, text=True,
                                  timeout=600)
            assert done.returncode == 0, "the run with PYTHONHASHSEED={} failed:\n{}".format(
                hash_seed, done.stderr[-2000:])
            folders[hash_seed] = os.path.join(out, "runs", "benchmark", profile)
        here = os.path.join(root, "out_here")
        cross_process_run(root, here, paths, profile,
                          CROSS_PROCESS_CALIBRATIONS[profile]() if profile in CROSS_PROCESS_CALIBRATIONS else None)
        folders["here"] = os.path.join(here, "runs", "benchmark", profile)

        names = sorted(os.listdir(folders["here"]))
        assert names == ["nearest_w0" + suffix, "nearest_w125" + suffix, "nearest_w50" + suffix], names
        for label, folder in folders.items():
            assert sorted(os.listdir(folder)) == names, (label, os.listdir(folder))
        for name in names:
            contents = {label: read(os.path.join(folder, name), "rb") for label, folder in folders.items()}
            assert len(set(contents.values())) == 1, "{} differs between {}".format(name, sorted(contents))
        report = strict_json(contents["here"].decode("utf-8"))
        assert report["pooled"]["n_gt"] > 0 and report["pooled"]["n_pred"] > 0, "a run that scored nothing is no proof"
        return report


# -- the oracle -------------------------------------------------------------------------------------------------------

def test_dfl_oracle_scores_perfectly():
    """The truth itself as the prediction scores 1.0 on every real match: the chain from the loaded match to the score.

    An evidence source `oracle` (registered here and removed afterwards) says home on the frames the truth calls H,
    away on the ones it calls A, and nothing on the rest; majority_vote with half_width = 0 and fallback D turns that
    into the truth again. The truth reaches it by closure, from truth.truth_codes, never through the inputs (a
    MatchInputs holds none). Run through evaluate_configs on the clean profile, spellF1, macroF1 and alive/dead F1 must
    each be exactly 1.0 on each of the seven matches and pooled.

    What it covers is everything between the loaded DflMatch and the score: the runner, MatchInputs, the separator rows
    between the halves and each half's own first and last row, the decoder path (predict, the blocks, majority_vote),
    score_halves and pooled. A row too many or too few at half time, a first row dropped, a half decoded one row late
    or a team swapped on the way there is not 1.0.

    What it cannot cover is the link from the DFL file to the truth. The oracle's answers and the runner's truth are the
    same truth.truth_codes of the same loaded match, and the oracle never reads the tracking, so a mistake in the parse
    or in truth_codes is on both sides and cancels (with H and A swapped inside truth_codes it still scores 1.0 on all
    seven matches). That link is checked elsewhere: the parse against kloppy and by frame number
    (test_dfl_parser_agrees_with_kloppy and test_dfl_second_half_spot_check, A05), and the teams by MIN_AGREEMENT in
    test_dfl_truth_conventions (the nearest player must name the truth's team on more than 60% of the alive
    ball-visible frames of every match and of each half: it does, at 71% to 82%, and with the teams swapped it would
    sit at 18% to 29%).

    Why all three scores are asserted, and exactly. Each is blind to something that another sees. macroF1 counts frames,
    so it sees a single wrong frame wherever it is; spellF1 matches whole possessions by IoU (down to 0.7), so a
    boundary a frame out is forgiven unless the possession is very short; alive/dead F1 sees only dead against alive,
    so it cannot see a team at all. A review of this check left the first row of the second half undecoded (X), and
    only macroF1 fell for that mutant: the scorer back-fills an X at a half's start, so spellF1 never sees it, and a
    check of spellF1 alone would have passed it. That is another mutant than the controls below, which are mutants of
    the oracle itself, on J03WMX: in the one-row slips (every label of a half shifted by one row) all three scores fall
    (by 0.0005 to 0.0102), and a swapped team leaves alive/dead F1 at exactly 1.0 while the other two fall far, so no
    one of them can stand for the rest.
    """
    from pipeline import evidence as evidence_stage
    from pipeline.contracts import Evidence, no_opinion
    answers = {match_id: truth.truth_codes(real_match(match_id))[0] for match_id in sources.MATCH_IDS}

    def oracle(run, lag=0, half=0, swap=False):
        """The truth of run.inputs.game; in half `half` (0: both) `lag` rows late, and/or with the teams swapped."""
        codes = answers[run.inputs.game].copy()
        for number, start, stop in run.inputs.graded:
            if half in (0, number):
                piece = codes[start:stop]
                if lag:
                    piece = np.concatenate([np.full(lag, pm.X, dtype=np.int8), piece[:-lag]])
                if swap:
                    piece = np.where(piece == pm.H, pm.A, np.where(piece == pm.A, pm.H, piece)).astype(np.int8)
                codes[start:stop] = piece
        home = no_opinion(run.inputs.n_frames)
        home[codes == pm.H] = 1.0
        home[codes == pm.A] = 0.0
        return Evidence(alive=no_opinion(run.inputs.n_frames), home=home)

    with registered(evidence_stage.REGISTRY, oracle=oracle), tempfile.TemporaryDirectory() as root:
        def config(name, lag=0, half=0, swap=False):
            return write_text(root, name + ".toml", ORACLE.format(lag=lag, half=half, swap=str(swap).lower()))

        def on_j03wmx(path, truth_name="raw"):
            result = evaluate_clean([path], match_ids=["J03WMX"], truth=truth_name, log=quiet)
            return result["configs"][0]["report"]["per_match"]["J03WMX"]

        exact = config("oracle_exact")
        result = evaluate_clean([exact], log=quiet)
        report = result["configs"][0]["report"]
        assert [record["id"] for record in report["matches"]] == list(sources.MATCH_IDS)
        assert len(report["halves"]) == 2 * len(sources.MATCH_IDS), "both halves of every match"
        print("      {:<8} {:>6} {:>9} {:>9} {:>9}".format("match", "poss", "spellF1", "macroF1", "alive/dead"))
        for match_id in sources.MATCH_IDS:
            row = report["per_match"][match_id]
            print("      {:<8} {:>6} {:>9.4f} {:>9.4f} {:>9.4f}".format(
                match_id, row["n_gt"], row["spell_f1_mean"], row["macro_f1"], row["alive_dead_f1"]))
            assert row["n_gt"] == row["n_pred"] > 100, "a match with few possessions would prove little: " + match_id
            assert row["spell_f1_mean"] == 1.0 and row["macro_f1"] == 1.0 and row["alive_dead_f1"] == 1.0, (
                "{}: the oracle must score exactly 1.0 and scored {}".format(match_id, row))
        pooled = report["pooled"]
        assert pooled["spell_f1_mean"] == pooled["macro_f1"] == pooled["alive_dead_f1"] == 1.0, pooled
        assert report["labels"]["predicted"]["X"] == 0.0 and report["labels"]["predicted"]["D"] > 0.3, (
            "the oracle must predict the truth's stoppages: " + str(report["labels"]))
        assert report["labels"]["predicted"] == report["labels"]["true"] | {"X": 0.0}

        # Controls: a slip of one row, or a swapped team, is not 1.0, and which score sees it is what it is.
        print("      controls on J03WMX         spellF1   macroF1 alive/dead")
        for name, kind in (("both halves one row late", dict(lag=1)), ("second half one row late", dict(lag=1, half=2)),
                           ("first half one row late", dict(lag=1, half=1))):
            row = on_j03wmx(config("oracle_late", **kind))
            print("      {:<26} {:>8.4f} {:>9.4f} {:>10.4f}".format(
                name, row["spell_f1_mean"], row["macro_f1"], row["alive_dead_f1"]))
            assert row["spell_f1_mean"] < 1.0 and row["macro_f1"] < 1.0 and row["alive_dead_f1"] < 1.0, (
                "{}: an oracle one frame late must not score 1.0 on any of the three: the check could not see an "
                "off-by-one ({})".format(name, row))
        row = on_j03wmx(config("oracle_swapped", half=2, swap=True))
        print("      {:<26} {:>8.4f} {:>9.4f} {:>10.4f}".format(
            "teams swapped, second half", row["spell_f1_mean"], row["macro_f1"], row["alive_dead_f1"]))
        assert row["alive_dead_f1"] == 1.0, "alive/dead F1 cannot see a team"
        assert row["spell_f1_mean"] < 0.9 and row["macro_f1"] < 0.9, "a swapped half must be seen by the other two"

        # Against the coarse truth the raw oracle is no longer perfect (its short possessions are absorbed there).
        row = on_j03wmx(exact, "coarse")
        print("      raw oracle, coarse truth   {:>8.4f} {:>9.4f} {:>10.4f}   possessions {} against {}".format(
            row["spell_f1_mean"], row["macro_f1"], row["alive_dead_f1"], row["n_pred"], row["n_gt"]))
        assert row["macro_f1"] < 1.0 and row["n_gt"] < row["n_pred"], "the coarse truth was not applied"
    assert "oracle" not in evidence_stage.REGISTRY, "the temporary oracle must be gone"
