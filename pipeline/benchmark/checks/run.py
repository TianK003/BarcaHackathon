"""Checks of the benchmark runner (pipeline/benchmark/run.py): configs scored on matches, reported, and repeatable.

The runner is what the team chooses between methods with, so what is held against it here is honesty and repeatability
more than arithmetic (score.py's checks own the arithmetic): that what it prints and writes is what the scoring
functions say, that a config's numbers do not depend on which other configs ran beside it, that the comparison's
direction is "other minus the first", that the same run gives the same bytes in another process with another hash seed,
that it refuses what it cannot score before it loads a match and names the source that needs a video, and that the
oracle, a prediction that is the truth itself, scores 1.0 on every real match (the check of the whole chain, from the
DFL file to the score, both halves included; --dfl). The matches of the synthetic checks are random_match's (checks/
inputs.py): two halves of random flags and random tracking, whose score means nothing and whose every number can be
recomputed here with the scoring functions, which is what the checks do.
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
from pipeline.benchmark import dfl, score, sources, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import real_match, write_synthetic_dfl
from pipeline.benchmark.checks.inputs import random_match, whole_positions
from pipeline.benchmark.checks.score import CONFIG_FOLDER
from pipeline.config import ConfigError, load_config
from pipeline.selftest import expect_error, quiet, write_text

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORT_KEYS = {"calibration", "config", "environment", "git", "halves", "labels", "matches", "per_match", "pooled",
               "possessions_per_graded_minute", "profile", "seeds", "t_interval", "truth", "versus"}
HALF_KEYS = {"match_id", "half", "seed", "n_gt", "n_pred", "tp", "confusion"}
ROUNDING = 1e-9                         # reports round their floats to 9 decimals
HASH_SEEDS = ("1", "2")                 # PYTHONHASHSEED of the two subprocesses of the determinism check
ORACLE = """
scope = "halves"

[[evidence]]
kind = "oracle"
lag = {lag}

[decoder]
kind = "majority_vote"
half_width = 0
fallback = "D"
"""


# -- helpers ----------------------------------------------------------------------------------------------------------

def yardstick_path(width):
    return os.path.join(CONFIG_FOLDER, "nearest_w{}.toml".format(width))


def synthetic_matches():
    """Two matches of random flags and tracking, of different lengths: the ids are not DFL's, so there is no cache."""
    return {"RND001": random_match("RND001"), "RND002": random_match("RND002", seed=1, rows=(300, 260))}


@contextlib.contextmanager
def loading(serve):
    """dfl.load_match answered by serve(match_id), for the runner's own path to a match; restored afterwards."""
    real = dfl.load_match

    def served(match_id, data_dir=None, rebuild=False, paths=None):
        return serve(match_id)
    dfl.load_match = served
    try:
        yield
    finally:
        dfl.load_match = real


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


def read(path, mode="r"):
    """The contents of a file, closed again: text as UTF-8 by default, bytes with mode "rb"."""
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as handle:
        return handle.read()


def strict_json(text):
    """The JSON of `text`, refusing NaN and infinity, which JSON has no way to write."""
    def refuse(name):
        raise AssertionError("the report holds {}, which is not JSON".format(name))
    return json.loads(text, parse_constant=refuse)


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
        result = benchmark_run.evaluate_configs(paths, matches=matches, data_dir=root, log=lines.append)
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
        alone = benchmark_run.evaluate_configs([paths[2]], matches=matches, data_dir=root, log=quiet)
        solo = alone["configs"][0]["report"]
        assert solo["versus"] is None
        assert {key: value for key, value in solo.items() if key != "versus"} == {
            key: value for key, value in reports["nearest_w25"].items() if key != "versus"}

        # The printout: the reference is named, the sign convention is said, and so is the interval's level.
        table = "\n".join(lines)
        for fragment in ("nearest_w50", "nearest_w0", "nearest_w25", "against nearest_w50", "other - nearest_w50",
                         "positive means", "90% t-interval", "sign-flip p", "spellF1", "macroF1", "alive/dead F1"):
            assert fragment in table, (fragment, table)

        # The files: where they go, and what they are.
        written = benchmark_run.write_reports(result)
        folder = os.path.join(root, "runs", "benchmark", "clean")
        assert written == [os.path.join(folder, "nearest_w50.json"), os.path.join(folder, "nearest_w0.json"),
                           os.path.join(folder, "nearest_w25.json")], written
        assert sorted(os.listdir(folder)) == ["nearest_w0.json", "nearest_w25.json", "nearest_w50.json"], "no .tmp left"
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
            os.path.join(other_folder, "runs", "benchmark", "clean", name) for name in
            ("nearest_w50.json", "nearest_w0.json", "nearest_w25.json")]


def test_runner_report_is_written_atomically():
    """The temporary file is renamed over the old report; if that fails, the old one is intact and no .tmp is left."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        result = benchmark_run.evaluate_configs([yardstick_path(50)], matches=matches, data_dir=root, log=quiet)
        path = benchmark_run.write_reports(result)[0]
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
        assert os.listdir(folder) == ["nearest_w50.json"], os.listdir(folder)


def test_runner_scores_against_the_coarse_truth_when_asked():
    """--truth coarse scores against truth.coarse_codes, says so, and files its report apart from the raw one."""
    matches = synthetic_matches()
    path = yardstick_path(25)
    with tempfile.TemporaryDirectory() as root:
        raw = benchmark_run.evaluate_configs([path], matches=matches, data_dir=root, log=quiet)
        coarse = benchmark_run.evaluate_configs([path], matches=matches, truth="coarse", data_dir=root, log=quiet)
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
        assert names == ["nearest_w25.json", "nearest_w25.coarse.json"], names
        assert sorted(os.listdir(os.path.join(root, "runs", "benchmark", "clean"))) == sorted(names), (
            "the two reports must sit side by side, neither replacing the other")


def test_runner_clean_runs_one_seed_and_says_so():
    """The clean profile has nothing random to repeat: --seeds is ignored, with a note, and only seed 0 is run."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = benchmark_run.evaluate_configs([yardstick_path(0)], seeds=3, matches=matches, data_dir=root,
                                                log=lines.append)
        report = result["configs"][0]["report"]
        assert report["seeds"] == [0] and {half["seed"] for half in report["halves"]} == {0}
        assert len(report["halves"]) == 4
        assert any("seed" in line and "clean" in line and "ignored" in line for line in lines), lines
        one = benchmark_run.evaluate_configs([yardstick_path(0)], seeds=1, matches=matches, data_dir=root,
                                             log=lines.append)
        assert one["configs"][0]["report"] == report


def test_runner_one_match_has_no_interval_and_says_so():
    """A t interval of one match is not a number: the report holds null, and the table says why, not NaN."""
    matches = synthetic_matches()
    with tempfile.TemporaryDirectory() as root:
        lines = []
        result = benchmark_run.evaluate_configs([yardstick_path(50), yardstick_path(25)], match_ids=["RND002"],
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
        message = expect_error(benchmark_run.BenchmarkError, lambda: benchmark_run.evaluate_configs(
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
        expect_error(benchmark_run.BenchmarkError, lambda: benchmark_run.evaluate_configs(
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
            return benchmark_run.evaluate_configs(list(configs), **keywords)

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
            message = expect_error(benchmark_run.BenchmarkError, lambda: benchmark_run.evaluate_configs(
                [yardstick_path(50), video], matches=matches, data_dir=root, log=quiet),
                "uses_video", "pitch_lines", "needs_homography", "homography", "public window")
            assert "nearest_player" not in message, "the source that worked is not the one to blame: " + message
            assert not os.path.exists(os.path.join(root, "runs")), "nothing is written for a run that failed"

            late = write_text(root, "late_video.toml", refiner)
            message = expect_error(benchmark_run.BenchmarkError, lambda: benchmark_run.evaluate_configs(
                [late], matches=matches, data_dir=root, log=quiet), "late_video", "homography", "public window")
            assert "no evidence source" in message and "majority_vote" in message, message
    assert evidence_stage.REGISTRY == before_evidence and refine_stage.REGISTRY == before_refine, (
        "the registries must be as they were, or a later check would see the temporary components")
    # The pipeline's own message, through the inputs, is the one that is quoted.
    expect_error(match_inputs.AssetUnavailable, lambda: match_inputs.MatchInputs(
        matches["RND001"], match_inputs.clean_atd(matches["RND001"])).homography, "homography", "P01")


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
    with tempfile.TemporaryDirectory() as root, loading(serve):
        result = benchmark_run.evaluate_configs([yardstick_path(50), yardstick_path(0)], match_ids=ids, data_dir=root,
                                                log=quiet)
    assert [name for name, _ in made] == ids, "each match is loaded once, in the order given"
    for entry in result["configs"]:
        assert [record["id"] for record in entry["report"]["matches"]] == ids
        assert [half["match_id"] for half in entry["report"]["halves"]] == [i for i in ids for _ in (1, 2)]
    gc.collect()
    assert all(ref() is None for _, refs in made for ref in refs), "a match is still held after the run"

    # Without ids: sources.MATCH_IDS' order.
    made.clear()
    with tempfile.TemporaryDirectory() as root, loading(serve):
        benchmark_run.evaluate_configs([yardstick_path(25)], data_dir=root, log=quiet)
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
                code = benchmark_run.main(list(arguments))
            return code, shown.getvalue()

        code, out = run_main(first, second, "--data-dir", root, "--matches", "J03WMX", "J03WN1", "J03WOH")
        assert code == 0, out
        folder = os.path.join(root, "runs", "benchmark", "clean")
        assert sorted(os.listdir(folder)) == ["nearest_w25.json", "nearest_w50.json"], out
        assert "nearest_w50.json" in out and "against nearest_w50" in out and "3 matches" in out, out
        report = strict_json(read(os.path.join(folder, "nearest_w25.json")))
        assert [record["id"] for record in report["matches"]] == ["J03WMX", "J03WN1", "J03WOH"]
        assert report["versus"]["reference"]["stem"] == "nearest_w50" and report["seeds"] == [0]

        code, out = run_main(first, "--data-dir", root, "--matches", "J03WMX", "--truth", "coarse", "--seeds", "5")
        assert code == 0 and "ignored" in out, out
        assert sorted(os.listdir(folder)) == ["nearest_w25.json", "nearest_w50.coarse.json", "nearest_w50.json"]

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


def cross_process_run(root, out, paths):
    """What each process of test_runner_json_identical_across_processes does: evaluate, and write to its own folder."""
    configs = [yardstick_path(width) for width in (50, 0, 125)]
    result = benchmark_run.evaluate_configs(configs, matches=cross_process_matches(root, paths), data_dir=root,
                                            log=quiet)
    return benchmark_run.write_reports(result, data_dir=out)


def test_runner_json_identical_across_processes():
    """The same run in two processes with different PYTHONHASHSEED, and in this one, writes the same bytes.

    A set iterated anywhere between the scores and the report, or a dict built in hash order, would show here as
    another order of keys or of matches. The synthetic DFL match is served through the cache, which is built here
    first, so that the two processes only read it and never build the same files at once.
    """
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "source"))
        paths = write_synthetic_dfl(os.path.join(root, "source"))
        assert dfl.load_match("SYN001", data_dir=root, paths=paths).report["cache"] == "built"       # warms the cache
        folders = {}
        for hash_seed in HASH_SEEDS:
            out = os.path.join(root, "out_" + hash_seed)
            code = ("import sys, json; from pipeline.benchmark.checks.run import cross_process_run as go; "
                    "go(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))")
            done = subprocess.run([sys.executable, "-c", code, root, out, json.dumps(paths)], cwd=ROOT,
                                  env=dict(os.environ, PYTHONHASHSEED=hash_seed), capture_output=True, text=True,
                                  timeout=600)
            assert done.returncode == 0, "the run with PYTHONHASHSEED={} failed:\n{}".format(
                hash_seed, done.stderr[-2000:])
            folders[hash_seed] = os.path.join(out, "runs", "benchmark", "clean")
        here = os.path.join(root, "out_here")
        cross_process_run(root, here, paths)
        folders["here"] = os.path.join(here, "runs", "benchmark", "clean")

        names = sorted(os.listdir(folders["here"]))
        assert names == ["nearest_w0.json", "nearest_w125.json", "nearest_w50.json"], names
        for label, folder in folders.items():
            assert sorted(os.listdir(folder)) == names, (label, os.listdir(folder))
        for name in names:
            contents = {label: read(os.path.join(folder, name), "rb") for label, folder in folders.items()}
            assert len(set(contents.values())) == 1, "{} differs between {}".format(name, sorted(contents))
        report = strict_json(contents["here"].decode("utf-8"))
        assert [record["id"] for record in report["matches"]] == ["SYN001", "RND001", "RND002"]
        assert report["matches"][0]["sources"] is not None, "the synthetic DFL match names its files"
        assert report["pooled"]["n_gt"] > 0 and report["pooled"]["n_pred"] > 0, "a run that scored nothing is no proof"


# -- the oracle -------------------------------------------------------------------------------------------------------

def test_dfl_oracle_scores_perfectly():
    """A prediction that is the truth itself scores 1.0 on every real match: the whole chain, both halves included.

    An evidence source `oracle` (registered here and removed afterwards) says home on the frames the truth calls H,
    away on the ones it calls A, and nothing on the rest; majority_vote with half_width = 0 and fallback D turns that
    into the truth again. The truth reaches it by closure, from truth.truth_codes, never through the inputs (a
    MatchInputs holds none). Run through evaluate_configs on the clean profile, so through load_match, MatchInputs,
    predict, score_halves and pooled, spellF1, macroF1 and alive/dead F1 must each be exactly 1.0 on each of the seven
    matches and pooled: a row too many or too few at the half time, a team swapped, a first row dropped or a frame
    taken for its neighbour is not 1.0. Two controls show that this is so: the same oracle one row late scores
    macroF1 below 1.0, and the raw oracle against the coarse truth scores below 1.0 too (the coarse truth is applied).
    """
    from pipeline import evidence as evidence_stage
    from pipeline.contracts import Evidence, no_opinion
    answers = {match_id: truth.truth_codes(real_match(match_id))[0] for match_id in sources.MATCH_IDS}

    def oracle(run, lag=0):
        codes = answers[run.inputs.game]
        if lag:
            codes = np.concatenate([np.full(lag, pm.X, dtype=np.int8), codes[:-lag]])
        home = no_opinion(run.inputs.n_frames)
        home[codes == pm.H] = 1.0
        home[codes == pm.A] = 0.0
        return Evidence(alive=no_opinion(run.inputs.n_frames), home=home)

    with registered(evidence_stage.REGISTRY, oracle=oracle), tempfile.TemporaryDirectory() as root:
        exact = write_text(root, "oracle_exact.toml", ORACLE.format(lag=0))
        late = write_text(root, "oracle_late.toml", ORACLE.format(lag=1))
        result = benchmark_run.evaluate_configs([exact], log=quiet)
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

        # Control 1: one row late is an off-by-one, and must not score 1.0.
        slipped = benchmark_run.evaluate_configs([late], match_ids=["J03WMX"], log=quiet)["configs"][0]["report"]
        row = slipped["per_match"]["J03WMX"]
        print("      one row late, J03WMX:   spellF1 {:.4f}   macroF1 {:.6f}".format(
            row["spell_f1_mean"], row["macro_f1"]))
        assert row["macro_f1"] < 1.0, "an oracle one frame late must not score 1.0: the check cannot see an off-by-one"

        # Control 2: against the coarse truth, the raw oracle is no longer perfect (its short possessions are absorbed).
        coarse = benchmark_run.evaluate_configs([exact], match_ids=["J03WMX"], truth="coarse", log=quiet)
        row = coarse["configs"][0]["report"]["per_match"]["J03WMX"]
        print("      raw oracle, coarse truth, J03WMX:   spellF1 {:.4f}   macroF1 {:.4f}   "
              "possessions {} against {}".format(row["spell_f1_mean"], row["macro_f1"], row["n_pred"], row["n_gt"]))
        assert row["macro_f1"] < 1.0 and row["n_gt"] < row["n_pred"], "the coarse truth was not applied"
    assert "oracle" not in evidence_stage.REGISTRY, "the temporary oracle must be gone"
