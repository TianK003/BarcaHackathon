"""Run pipeline configs on the benchmark's matches, score them, and say how far the scores can be trusted.

    python pipeline/benchmark/run.py CONFIG [CONFIG ...]     # e.g. pipeline/benchmark/configs/nearest_w50.toml first

    codes, sources = predict(config, inputs)         # one config on one match: the codes, and the evidence behind them
    result = evaluate_configs([path, path, ...])     # every config on every match, scored, compared, with its reports
    write_reports(result)                            # <data dir>/runs/benchmark/<profile>/<stem>.json, one per config

Prediction
----------
predict() runs a config on one MatchInputs (inputs.py) through the pipeline's own evidence, decode and refine stages,
without the CSV. The benchmark is only worth anything if it scores what the submission pipeline does, so prediction goes
through pipeline.run.compute_evidence, make_blocks and decode_blocks, the same functions pipeline.run.run calls, and a
change to them changes the benchmark with it. What predict leaves out of run() is the output stage: it writes no file.
Benchmark code never writes a CSV of labels (a CSV of labels of BAR-ATH is match data, CLAUDE.md), and a benchmark match
has no CSV to write; the scoring reads the codes directly (score.py) and pipeline.output.summarise describes them.

Seeded as run() seeds: `random` and `np.random` with config.seed before anything else, because library code reaches
for the global generators, and the RunContext carries the same seed for components that make their own generator.
Two calls with one config on one inputs give the same codes, and the same as pipeline.run.run on them
(test_predict_equals_pipeline_run).

Evaluation
----------
evaluate_configs() is what the team chooses between methods with, so it is built to be honest before it is convenient.

* One score, the official one. Each config runs on each match (and each degradation seed), and only the counts of every
  graded half are kept (score.HalfScore); the pooled spellF1, macroF1 and alive/dead F1 are computed once from the
  summed counts, and equal pm.evaluate of all the halves laid end to end (score.py). The mean of the matches' own
  scores is another number, and is reported next to it, with its interval.
* Seven matches are seven matches. What can be said of a gap between two configs is how it looks across them: the delta
  of each match, the mean of the deltas with a Student t interval (90%, score.t_interval), and an exact sign-flip test
  (score.sign_flip_p). Neither is a bootstrap: resampling 7 matches, or the frames inside them, would pretend to a
  sample the benchmark does not have. The interval is called "excluding 0" only where it clears 0 by more than
  score.FLIP_TOLERANCE, the margin under which compare() treats two scores as equal (interval_excludes_zero).
* The first config is the reference. Every other config is compared with score.compare(first, other), so a delta is
  other - first and a positive one means the other is better. Put the config you want to beat first.
* One match in memory at a time (about 150 MB each). A match is loaded, its truth computed once, its tracking served
  once per seed, every config run on that, and only the HalfScores kept before the next match is loaded.
* The truth is the DFL flags' (truth.py): "raw", or "coarse", the raw truth with its short sandwiched possessions
  absorbed (truth.coarse_codes), because DFL's grain is about twice ours (A06). The two are meant to be run side by
  side; a ranking that differs between them is a warning, not something to fix.
* Nothing in a report depends on the machine's speed, the order of a set or the clock: no timings are written (they are
  printed), floats are rounded to 9 decimals (BLAS summation order differs between machines), keys are sorted and the
  file is LF with an atomic rename. The same inputs, seed and versions give the same bytes, in any process
  (test_runner_json_identical_across_processes).

What the clean profile is. The tracking exactly as DFL gives it: the ball is present on every frame, so nearest_player
votes everywhere and majority_vote, which takes D only where nobody voted, labels almost nothing D, where the truth has
41% to 52% of its frames dead. That is expected and is why the degraded profiles exist; it is not a bug of the configs.

This file is called run.py and pipeline/benchmark/inputs.py is called inputs.py, like pipeline/run.py and
pipeline/inputs/. The pipeline's modules are imported by their absolute names (`from pipeline import run as
pipeline_run`): never `import run`, which with this folder on sys.path would be this file.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here (run.py, inputs.py, score.py, ...) can
    # shadow a top-level or a pipeline module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import dfl, score, sources  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
CLEAN = "clean"
PROFILES = {CLEAN: 1}              # name -> version of how its tracking is made (in the report); Task 11 swaps it
DEFAULT_PROFILE = CLEAN
DEFAULT_SEEDS = 3                  # degradation seeds per match; the clean profile has nothing random and runs seed 0
TRUTHS = ("raw", "coarse")
REPORT_FOLDER = ("runs", "benchmark")        # <data dir>/runs/benchmark/<profile>/<stem>[.<truth>].json
DECIMALS = 9                       # floats in a report are rounded to this: BLAS summation order varies by machine
LEVEL = 0.9                        # score.T_95 holds the two-sided 90% t quantiles
METRICS = (("spell_f1_mean", "spellF1"), ("macro_f1", "macroF1"))   # what a comparison reports, ranking metric first


class BenchmarkError(ValueError):
    """A run that cannot be done as asked. The message names the config, match or profile and says what to change."""


# -- one config on one match ------------------------------------------------------------------------------------------

def check_scope(config):
    """BenchmarkError unless `config` decodes each graded half on its own (scope = "halves")."""
    if config.scope == "file":
        raise BenchmarkError(
            "{}: scope = \"file\" decodes across half time and over ungraded frames, and the benchmark scores each "
            "graded half on its own; use scope = \"halves\" (only pipeline/configs/smooth50_repro.toml needs "
            "\"file\", to rebuild a published baseline byte for byte)".format(config.stem))


def predict(config, inputs):
    """(codes, sources): the codes the config gives on `inputs`, and the evidence they were decoded from.

    `codes` is the (n_frames,) int8 array of pm's codes that pipeline.run.run would write to its CSV: the decoder and
    the refiners, block by block, and X on every frame outside the graded halves. `sources` is {name: Evidence} over
    the whole match, read-only, the same thing RunResult.evidence holds: a failing source can be named from it, and
    pipeline.evidence.describe takes it. BenchmarkError (a ValueError) for scope = "file", which decodes across half
    time and over frames that are not graded (docs/traps.md T11); the benchmark scores each half on its own, so only
    "halves" means anything here, and the one config that needs "file" is the reproduction of the published baselines.
    """
    check_scope(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    run_context = RunContext(inputs=inputs, seed=config.seed)
    sources_ = pipeline_run.compute_evidence(config, run_context)
    blocks = pipeline_run.make_blocks(config.scope, inputs)
    codes = pipeline_run.decode_blocks(config, run_context, sources_, blocks)
    return codes, sources_


def make_atd(match, profile, seed, calibration=None):
    """The tracking to serve for `match` under `profile` and `seed`, as an Atd of the BAR-ATH cache's own layout.

    "clean" is DFL's tracking as it is (inputs.clean_atd): nothing is degraded, so the seed and the calibration are not
    used. The degraded profiles come with Task 11, which routes them to the degrader here. BenchmarkError, listing the
    profiles there are, for any other name.
    """
    _check_profile(profile)
    return match_inputs.clean_atd(match)


def _check_profile(profile):
    if profile not in PROFILES:
        raise BenchmarkError("profile {!r} is not one of: {}".format(profile, ", ".join(sorted(PROFILES))))


# -- every config on every match --------------------------------------------------------------------------------------

def evaluate_configs(config_paths, profile=DEFAULT_PROFILE, seeds=DEFAULT_SEEDS, match_ids=None, matches=None,
                     calibration=None, truth="raw", data_dir=None, log=print):
    """Run each config on each match and score it; the result holds a JSON-ready report per config.

    `config_paths` are TOML files for pipeline.config.load_config; the first is the reference of the comparisons.
    `profile` names how the tracking is made (PROFILES) and `seeds` is how many degradation seeds to run, 0 up to
    seeds - 1 (the clean profile has none to repeat: its one seed is 0, and a note says so). `match_ids` are the
    matches, in the order to run them (default: all seven, sources.MATCH_IDS' order); they are read through
    dfl.load_match from `data_dir` (default games.DATA_DIR). `matches` is {id: DflMatch} to use instead of loading (the
    checks pass synthetic ones), `calibration` the Calibration the profile needs (none of the profiles of this task
    does, and one is refused) and `truth` "raw" or "coarse" (truth.coarse_codes). `log` receives each progress line
    and then the printed table.

    Everything that can be refused is, before the first match is loaded: no config, one that does not load (ConfigError,
    from the pipeline's own loader), one with scope = "file", two with the same name, an unknown profile or truth or
    match, a repeated match, fewer than one seed. BenchmarkError says which. A source that needs an asset that a
    benchmark match does not have (the video's homography, the Smart Tagging) stops the run with its config, its name
    and kind and the asset named, instead of scoring zeros.

    Returns {"profile", "seeds", "truth", "matches", "git", "data_dir", "reference", "configs"}; "configs" is a list in
    the order given of {"stem", "report", "scores", "seconds"}: the report is what write_reports writes (a function of
    the config, the profile and the matches, with no timing), `scores` the HalfScores it was made from and `seconds`
    the time spent on the config (printed, never written).
    """
    configs = _load_configs(config_paths)
    _check_profile(profile)
    if truth not in TRUTHS:
        raise BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(TRUTHS)))
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    if calibration is not None:
        raise BenchmarkError("profile {!r} takes no calibration, and one was given: it would be ignored, and a report "
                             "must not suggest that it was used".format(profile))
    ids = _match_ids(match_ids, matches)
    data_dir_used = data_dir or games.DATA_DIR
    seed_list = [0] if profile == CLEAN else list(range(seeds))
    if profile == CLEAN and seeds != 1:
        log("note      profile {} has nothing random to repeat: seeds ignored, seed 0 only".format(CLEAN))

    scores = {config.stem: [] for config in configs}
    seconds = {config.stem: 0.0 for config in configs}
    records = []
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise BenchmarkError("the match given as {} says it is {}; its scores would be filed under the wrong id"
                                 .format(match_id, match.match_id))
        codes_true, carried = truths.truth_codes(match)
        if truth == "coarse":
            codes_true = truths.coarse_codes(codes_true, match.graded, match.fps)
        record = {"id": match.match_id, "frames": int(match.n_frames),
                  "graded_frames": int(sum(stop - start for _, start, stop in match.graded)), "fps": int(match.fps),
                  "carried_frames": int(carried["carried_possession_frames"] + carried["carried_status_frames"])}
        for seed in seed_list:
            atd = make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            if seed == seed_list[0]:
                record["sources"] = inputs.atd_manifest["sources"]
                record["sources_reason"] = inputs.atd_manifest["sources_reason"]
            for config in configs:
                started = time.perf_counter()
                codes = _codes(config, inputs)
                scores[config.stem] += score.score_halves(codes_true, codes, match.graded, match.match_id, seed)
                seconds[config.stem] += time.perf_counter() - started
            del atd, inputs, codes
        records.append(record)
        log("match     {}   {:,} rows, {:,} graded   truth {}".format(
            record["id"], record["frames"], record["graded_frames"], truth))
        del match, codes_true                    # the next match is loaded only when this one is gone

    git = pipeline_run.git_state()
    environment = _environment()
    profile_info = {"name": profile, "version": PROFILES[profile]}
    truth_info = {"name": truth, "min_s": truths.COARSE_MIN_S if truth == "coarse" else None}
    reference = configs[0]
    entries = []
    for config in configs:
        versus = None if config is reference else _versus(reference, config, scores)
        report = _report(config, scores[config.stem], records, seed_list, profile_info, truth_info, git, environment,
                         versus)
        entries.append({"stem": config.stem, "report": report, "scores": scores[config.stem],
                        "seconds": seconds[config.stem]})
    result = {"profile": profile_info, "seeds": seed_list, "truth": truth_info, "matches": records, "git": git,
              "data_dir": data_dir_used, "reference": reference.stem, "configs": entries}
    for line in render(result):
        log(line)
    return result


def _load_configs(config_paths):
    """[Config] for the paths, in order: loaded by the pipeline's own loader, each decoded per half, names unique."""
    config_paths = list(config_paths)
    if not config_paths:
        raise BenchmarkError("no config given; name at least one CONFIG.toml, and put the one to beat first (the "
                             "others are compared with it)")
    configs, paths = [], {}
    for path in config_paths:
        config = load_config(path)
        check_scope(config)
        if config.stem in paths:
            where = paths[config.stem]
            if os.path.normcase(where) == os.path.normcase(config.path):
                raise BenchmarkError("{} is given twice; a config is run once".format(config.stem))
            raise BenchmarkError("{} and {} have the same name, {}, so they would write the same report; rename one"
                                 .format(_shown(where), _shown(config.path), config.stem))
        paths[config.stem] = config.path
        configs.append(config)
    return configs


def _match_ids(match_ids, matches):
    """The ids to run, in order: those asked for (or all), each of them known and none repeated."""
    known = list(sources.MATCH_IDS) if matches is None else list(matches)
    ids = known if match_ids is None else list(match_ids)
    if not ids:
        raise BenchmarkError("no match to run on; give at least one of {}".format(", ".join(known)))
    repeated = sorted({match_id for match_id in ids if ids.count(match_id) > 1})
    if repeated:
        raise BenchmarkError("match {} is given twice; it would be scored twice and counted twice in every pooled "
                             "score".format(repeated[0]))
    unknown = [match_id for match_id in ids if match_id not in known]
    if unknown:
        raise BenchmarkError("unknown match id {}; {} {}".format(
            unknown[0], "the DFL benchmark has" if matches is None else "the matches given are", ", ".join(known)))
    return ids


def _codes(config, inputs):
    """predict's codes; an asset that a benchmark match does not have is a BenchmarkError that names its source."""
    try:
        return predict(config, inputs)[0]
    except match_inputs.AssetUnavailable as error:
        raise BenchmarkError(_unavailable(config, inputs, error)) from error


def _unavailable(config, inputs, error):
    """The sentence for an AssetUnavailable: which evidence source it was, found by running each source on its own."""
    run_context = RunContext(inputs=inputs, seed=config.seed)
    for component in config.evidence:
        try:
            pipeline_run.compute_evidence(dataclasses.replace(config, evidence=(component,)), run_context)
        except match_inputs.AssetUnavailable as own:
            return ("{}: the evidence source {!r} (kind {!r}) asked for an asset that a benchmark match does not have, "
                    "so this config cannot be benchmarked as it is. {}".format(config.stem, component.name,
                                                                                component.kind, own))
    refiners = "".join(", refiner {!r}".format(component.kind) for component in config.refine)
    return ("{}: no evidence source asked for it on its own, so it was the decoder {!r}{} that asked for an asset a "
            "benchmark match does not have. {}".format(config.stem, config.decoder.kind, refiners, error))


# -- the report -------------------------------------------------------------------------------------------------------

def _report(config, halves, records, seed_list, profile, truth, git, environment, versus):
    """The JSON-ready report of one config over all its HalfScores: counts, scores, intervals, and nothing timed."""
    per = score.per_match(halves)
    confusion = np.sum([np.array(half.confusion, dtype=np.int64) for half in halves], axis=0)
    total = float(confusion.sum())
    minutes = sum(record["graded_frames"] / record["fps"] / 60.0 for record in records) * len(seed_list)
    pooled = score.pooled(halves)
    intervals = {}
    for metric, _ in METRICS:
        mean, low, high = score.t_interval([per[match_id][metric] for match_id in sorted(per)])
        intervals[metric] = {"n": len(per), "level": LEVEL, "mean": mean, "low": low, "high": high}
    report = {
        "config": {"path": _shown(config.path), "text": config.text.replace("\r\n", "\n")},
        "profile": profile,
        "seeds": list(seed_list),
        "matches": records,
        "truth": truth,
        "git": git,
        "environment": environment,
        "pooled": pooled,
        "per_match": per,
        "t_interval": intervals,
        "possessions_per_graded_minute": {"graded_minutes": minutes, "predicted": pooled["n_pred"] / minutes,
                                          "true": pooled["n_gt"] / minutes},
        "labels": {"true": {pm.LABELS[row]: confusion[row].sum() / total for row in range(3)},
                   "predicted": {pm.LABELS[column]: confusion[:, column].sum() / total for column in range(4)}},
        "halves": [dataclasses.asdict(half) for half in halves],
        "versus": versus,
        "calibration": None,                 # the clean profile degrades nothing, so it measures nothing of ours
    }
    return _plain(report)


def _versus(first, config, scores):
    """`config` against `first`, the reference: score.compare (b - a) for each of METRICS, with the interval's verdict."""
    digest = hashlib.sha256(first.text.encode("utf-8")).hexdigest()
    versus = {"reference": {"stem": first.stem, "config_sha256": digest}}
    for metric, _ in METRICS:
        fields = score.compare(scores[first.stem], scores[config.stem], metric=metric)
        versus[metric] = dict(fields, interval_excludes_zero=interval_excludes_zero(fields["low"], fields["high"]))
    return versus


def interval_excludes_zero(low, high):
    """True only if the interval [low, high] clears 0 by more than score.FLIP_TOLERANCE; False for none, or for NaN.

    compare()'s interval does not follow its own tie rule: deltas of the order of 1e-16 (two F1s that are equal on
    paper) can give an interval that excludes 0 while wins and losses are 0 and p is 1.0. So an interval counts as
    excluding 0 here only if it is further than the tolerance from it, as a delta must be to count as a win or a loss.
    """
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return False
    return bool(low > score.FLIP_TOLERANCE or high < -score.FLIP_TOLERANCE)


def _plain(value):
    """`value` as JSON can hold it, whatever it is made of: rounded, with text keys, and without NaN or numpy types.

    Floats are rounded to DECIMALS (and -0.0 is 0.0), a NaN or an infinity (an interval of one match) becomes None,
    dictionary keys become text (the IoUs 0.3 ... 0.7 are "0.3" ... "0.7"), tuples become lists, numpy scalars Python
    ones. TypeError for anything else, so that a new field cannot reach a report unchecked.
    """
    if isinstance(value, dict):
        return {_key(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return round(number, DECIMALS) + 0.0 if math.isfinite(number) else None
    raise TypeError("{!r} ({}) cannot go into a report".format(value, type(value).__name__))


def _key(key):
    if isinstance(key, str):
        return key
    if isinstance(key, (float, np.floating)):
        return repr(round(float(key), DECIMALS))
    return str(key)


def _environment():
    from lxml import etree
    return {"python": platform.python_version(), "numpy": np.__version__, "lxml": etree.__version__,
            "platform": platform.platform()}


def _shown(path):
    """A path as people read it: relative to the repository when inside it, absolute otherwise, with forward slashes."""
    path = os.path.abspath(path)
    if os.path.splitdrive(path)[0].lower() == os.path.splitdrive(ROOT)[0].lower():
        relative = os.path.relpath(path, ROOT)
        if not relative.startswith(".."):
            return relative.replace(os.sep, "/")
    return path.replace(os.sep, "/")


def report_path(data_dir, profile, stem, truth):
    """<data dir>/runs/benchmark/<profile>/<stem>.json, or <stem>.<truth>.json for a truth that is not the raw one."""
    name = stem + ("" if truth == "raw" else "." + truth) + ".json"
    return os.path.join(data_dir, *REPORT_FOLDER, profile, name)


def write_reports(result, data_dir=None):
    """Write each config's report under `data_dir` (default: the one the run read from); the paths, in order.

    sort_keys, indent 2, ASCII, LF and a final newline; written to a temporary file in the same folder and renamed over
    the old report (os.replace), so that a run that is interrupted, or a disk that fills, never leaves half a report and
    never damages the one that was there. A failed write removes its temporary file and raises.
    """
    data_dir = data_dir or result["data_dir"]
    paths = []
    for entry in result["configs"]:
        path = report_path(data_dir, result["profile"]["name"], entry["stem"], result["truth"]["name"])
        text = json.dumps(entry["report"], sort_keys=True, indent=2, allow_nan=False) + "\n"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(temporary)
            raise
        paths.append(path)
    return paths


# -- the printout -----------------------------------------------------------------------------------------------------

def render(result):
    """The lines that evaluate_configs logs: one block per config, a table of them, and each against the reference."""
    reports = [(entry["stem"], entry["report"], entry["seconds"]) for entry in result["configs"]]
    git = result["git"]
    ids = [record["id"] for record in result["matches"]]
    lines = ["",
             "profile   {} (version {})   truth {}   seeds {}   {} matches   commit {}{}".format(
                 result["profile"]["name"], result["profile"]["version"], result["truth"]["name"],
                 ",".join(str(seed) for seed in result["seeds"]), len(ids), git["commit"] or "?",
                 " +dirty" if git["dirty"] else "")]
    for stem, report, seconds in reports:
        pooled, interval = report["pooled"], report["t_interval"]["spell_f1_mean"]
        rate, labels = report["possessions_per_graded_minute"], report["labels"]
        lines += ["",
                  "config    {}".format(stem),
                  "          pooled          spellF1 {}   macroF1 {}   alive/dead F1 {}".format(
                      _number(pooled["spell_f1_mean"]), _number(pooled["macro_f1"]), _number(pooled["alive_dead_f1"])),
                  "          per match       spellF1 mean {}   {}".format(
                      _number(interval["mean"]), _interval(interval)),
                  "          possessions     predicted {:.2f} per graded minute, true {:.2f}".format(
                      rate["predicted"], rate["true"]),
                  "          dead ball (D)   predicted {:.1%} of graded frames, true {:.1%}".format(
                      labels["predicted"]["D"], labels["true"]["D"]),
                  "          spellF1 by match   " + "  ".join(
                      "{} {}".format(match_id, _number(report["per_match"][match_id]["spell_f1_mean"]))
                      for match_id in ids),
                  "          seconds         {:.1f}".format(seconds)]
    lines += ["", "{:<22}{:>10}{:>10}{:>15}   {}".format("config", "spellF1", "macroF1", "alive/dead F1",
                                                         "mean per-match spellF1 [90% t-interval]")]
    for stem, report, _ in reports:
        pooled, interval = report["pooled"], report["t_interval"]["spell_f1_mean"]
        lines.append("{:<22}{:>10}{:>10}{:>15}   {} {}".format(
            stem, _number(pooled["spell_f1_mean"]), _number(pooled["macro_f1"]), _number(pooled["alive_dead_f1"]),
            _number(interval["mean"]), _bounds(interval)))

    reference = result["reference"]
    if len(reports) > 1:
        lines += ["", "against {} (the first config; delta = other - {}, so positive means the other is better)".format(
            reference, reference)]
        for stem, report, _ in reports[1:]:
            lines.append("{}".format(stem))
            for metric, label in METRICS:
                fields = report["versus"][metric]
                lines += _comparison(label, fields, label == METRICS[0][1])
        lines += ["", "wins and losses are matches whose delta is beyond 1e-12 either way, and an interval excludes 0 "
                      "only if it clears 0 by more than that.",
                  "with {} matches the smallest sign-flip p there can be is {:.4f}; neither it nor the interval "
                  "makes a gap real.".format(len(ids), 2.0 / 2 ** len(ids))]
    return lines


def _comparison(label, fields, detailed):
    interval = {"mean": fields["mean_delta"], "low": fields["low"], "high": fields["high"]}
    lines = ["  {:<8}  pooled delta {}   mean per-match delta {}   {}".format(
        label, _number(fields["pooled_delta"], signed=True), _number(fields["mean_delta"], signed=True),
        _interval(interval, fields["interval_excludes_zero"], signed=True)),
        "            sign-flip p {:.4f}   wins {}   losses {}   of {} matches".format(
            fields["p_sign_flip"], fields["wins"], fields["losses"], fields["n"])]
    if detailed:
        lines.append("            by competition   " + "   ".join(
            "{} {}".format(name, _number(delta, signed=True)) for name, delta in fields["by_competition"].items()))
        lines.append("            per match        " + "  ".join(
            "{} {}".format(match_id, _number(delta, signed=True))
            for match_id, delta in fields["match_deltas"].items()))
    return lines


def _number(value, signed=False):
    if value is None:
        return "n/a"
    return ("{:+.4f}" if signed else "{:.4f}").format(value)


def _bounds(interval, signed=False):
    """"[low, high]", or "no interval" where there are fewer than two matches to make one of."""
    if interval["low"] is None or interval["high"] is None:
        return "no interval (fewer than 2 matches)"
    return "[{}, {}]".format(_number(interval["low"], signed), _number(interval["high"], signed))


def _interval(interval, excludes=None, signed=False):
    """"90% t-interval [low, high]" and, for a delta, whether it excludes 0 (by more than the tie tolerance)."""
    bounds = _bounds(interval, signed)
    if interval["low"] is None or interval["high"] is None:
        return bounds
    text = "90% t-interval " + bounds
    if excludes is None:
        return text
    return text + (" excludes 0" if excludes else " includes 0")


# -- the command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run pipeline configs on the benchmark's matches; the first config is the reference that the "
                    "others are compared with.")
    parser.add_argument("configs", nargs="+", metavar="CONFIG",
                        help="TOML configs, e.g. pipeline/benchmark/configs/nearest_w50.toml; the first is the "
                             "reference that the others are compared with")
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help="how the tracking is made: {} (default {})".format(", ".join(sorted(PROFILES)),
                                                                                DEFAULT_PROFILE))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="degradation seeds per match (default {}; the clean profile has just one)".format(
                            DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--truth", choices=TRUTHS, default="raw",
                        help="the truth to score against: raw, or coarse (short sandwiched possessions absorbed)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--no-write", action="store_true", help="print only; write no report")
    args = parser.parse_args(argv)

    try:
        result = evaluate_configs(args.configs, profile=args.profile, seeds=args.seeds, match_ids=args.matches,
                                  truth=args.truth, data_dir=args.data_dir)
        paths = [] if args.no_write else write_reports(result)
    except (ConfigError, BenchmarkError, OSError) as error:
        # The expected failures get a sentence: a config to fix, a profile or match that does not exist, a source that
        # needs the video, data not downloaded yet. Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1
    print("")
    for path in paths:
        print("report    {}".format(_shown(path)))
    if args.no_write:
        print("report    not written (--no-write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
