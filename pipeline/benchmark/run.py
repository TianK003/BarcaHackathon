"""Run pipeline configs on the benchmark's matches, score them, and say how far the scores can be trusted.

    python pipeline/benchmark/run.py CONFIG [CONFIG ...]     # e.g. pipeline/benchmark/configs/nearest_w50.toml first

    codes, sources = predict(config, inputs)         # one config on one match: the codes, and the evidence behind them
    result = evaluate_configs([path, path, ...])     # every config on every match, scored, compared, with its reports
    results = evaluate_truths([path, ...])           # {"raw": result, "coarse": result}, one decode for both
    write_reports(result)                            # <data dir>/runs/benchmark/<profile>/<stem>[...].json, each

Prediction
----------
predict() runs a config on one MatchInputs (inputs.py) through the pipeline's own evidence, decode and refine stages,
without the CSV. The benchmark is only worth anything if it scores what the submission pipeline does, so prediction goes
through pipeline.run.prepare_inputs, compute_evidence, make_blocks and decode_blocks, the same functions
pipeline.run.run calls, and a change to them changes the benchmark with it. What predict leaves out of run() is the
output stage: it writes no file. Benchmark code never writes a CSV of labels (a CSV of labels of BAR-ATH is match data,
CLAUDE.md), and a benchmark match has no CSV to write; the scoring reads the codes directly (score.py) and
pipeline.output.summarise describes them.

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
  side; a ranking that differs between them is a warning, not something to fix. evaluate_truths (`--truth both`)
  scores each config's codes against both from one decode and gives exactly what two runs would: the decoder, not the
  scoring, is what a grid's time goes on.
* A config that fills ([inputs] ball_fill = "broadcast", R028) has no broadcast on a DFL match, so it is served a
  simulated one (broadcast_sim.served_inputs_for), driven by its own twin without the fill. Every report of a run that
  loaded the pools of that simulation holds their digest (`broadcast_sim`), and null when no config of the run fills.
* Nothing in a report depends on the machine's speed, the order of a set or the clock: no timings are written (they are
  printed), floats are rounded to 9 decimals (BLAS summation order differs between machines), keys are sorted and the
  file is LF with an atomic rename. The same inputs, seed and versions give the same bytes, in any process
  (test_runner_json_identical_across_processes).
* A run never replaces a report of another kind. The file is <stem>.json for all seven matches under the raw truth and
  the default number of seeds; <stem>.coarse.json under the coarse truth; for any other set of matches (one match, two,
  a different seven) `.m-<ids sorted, joined by "-">` is added, so `--matches J03WMX` writes <stem>.m-J03WMX.json next
  to the full report; and a number of seeds other than DEFAULT_SEEDS adds `.s<n>` (a degraded profile only: the clean
  profile has one seed, always), so a quick one-seed look never replaces the three-seed report. The order is
  <stem>[.<truth>][.m-<ids>][.s<n>].json (report_path). A config's own name may hold no dot, which would be read as one
  of these parts (BenchmarkError). What is not in the name is replaced: a report holds the digest of the Calibration its
  tracking was degraded with, and a later run with a newer Calibration writes over the older report of the same name.

The profiles (degrade.py). The tracking is served to every config as an Atd made by degrade.degrade from a TrackingView
of the match, which has no flag of the truth. There are four (PROFILES, each with the version of its recipe, which a
report holds). "clean" is DFL's tracking as it is and needs no Calibration. "gaps" has the ball missing where ours was
(the Calibration's masks, transplanted), so nearest_player votes on fewer frames, majority_vote has D to give, and the
scores are no longer those of a ball that is always there. "tracking", the default (version 3), adds the players broken
up as ours are (tracked spans and lost spans, 46 anonymous slots, a wrong-team slot now and then, NaN speeds in runs and
our frames with few players tracked): it is the profile whose gates all pass (R004c). "context" has tracking's players
and a ball whose gaps a hazard fitted on our match draws; it fails two of its gates (R005), so it is a sensitivity check
and not the yardstick, and the help, the printout and the report say so (SENSITIVITY_ONLY). Every profile but clean
needs the Calibration of BAR-ATH (calibration.load, measured once and cached), which evaluate_configs loads unless it is
given one, and whose digest goes into every report.

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
from pipeline import evidence as evidence_stage  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import degrade, dfl, score, sources  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
CLEAN = degrade.CLEAN
PROFILES = degrade.PROFILES        # name -> version of how its tracking is made (in the report): degrade.py's recipes
# The default is the profile whose eight gates all pass (R004c: tracking v3); context fails two of its nine (R005).
DEFAULT_PROFILE = "tracking"       # clean is the floor it is measured against
DEFAULT_SEEDS = 3                  # degradation seeds per match; the clean profile has nothing random and runs seed 0
# The order of the four yardsticks in the full-match scores that the organisers published (R000: smooth50 0.2433 >
# smooth125 0.2242 > smooth25 0.1996 > raw 0.0486), best first: what a profile's order is read against (rank_agreement).
PUBLISHED_ORDER = ("nearest_w50", "nearest_w125", "nearest_w25", "nearest_w0")
TRUTHS = ("raw", "coarse")
BOTH = "both"                      # --truth both: every truth of TRUTHS from one decode (evaluate_truths)
REPORT_FOLDER = ("runs", "benchmark")        # <data dir>/runs/benchmark/<profile>/<stem>[.<truth>][.m-..][.s<n>].json
DECIMALS = 9                       # floats in a report are rounded to this: BLAS summation order varies by machine
LEVEL = score.INTERVAL_LEVEL       # the coverage of the t intervals (the printout names it, checks tie T_95 to it)
METRICS = (("spell_f1_mean", "spellF1"), ("macro_f1", "macroF1"))   # what a comparison reports, ranking metric first
# A profile that is kept for a sensitivity check and not offered as the yardstick: name -> why. The --profile help, the
# printout and the report of a run on it say so (R005: context fails two of its gates).
SENSITIVITY_ONLY = {"context": "fails two of its gates (R005), the near-line share badly and the missing share "
                               "narrowly: a sensitivity check, not the yardstick"}
# What _unavailable adds when the source that asked for the missing asset is one that declares MATCH_ONLY.
DROP_HINT = (" Run with --drop-match-only to benchmark the config without its match-only sources (they are "
             "listed in the report).")
# Where a degraded profile's tracking comes from in our Calibration, as the printout names it beside the digest.
PROVENANCE = {
    "gaps": "the ball's gaps are our match's, transplanted (calibration.py)",
    "tracking": "the ball's gaps are our match's, transplanted, and the players' tracklets, slots and counts are ours "
                "(calibration.py)",
    "context": "the ball's gaps are drawn by the hazard fitted on our match (A08), the players are tracking's, from "
               "our tracklets, slots and counts",
}


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

    The config's [inputs] ball_fill is applied as run() applies it, by pipeline.run.prepare_inputs, before any source
    runs. A config that fills reads the MatchInputs' broadcast_ball, and on a match served without one it raises
    AssetUnavailable rather than run on the unfilled ball.
    """
    check_scope(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    inputs = pipeline_run.prepare_inputs(config, inputs)
    run_context = RunContext(inputs=inputs, seed=config.seed)
    sources_ = pipeline_run.compute_evidence(config, run_context)
    blocks = pipeline_run.make_blocks(config.scope, inputs)
    codes = pipeline_run.decode_blocks(config, run_context, sources_, blocks)
    return codes, sources_


def make_atd(match, profile, seed, calibration=None):
    """The tracking to serve for `match` under `profile` and `seed`, as an Atd of the BAR-ATH cache's own layout.

    The match is handed to the degrader as a TrackingView, which has the tracking and no flag of the truth
    (degrade.py), and degrade.degrade makes the Atd: "clean" is DFL's tracking as it is (nothing is degraded, so the
    seed and the calibration are not used); "gaps" has our ball's missing masks transplanted onto it; "tracking" adds
    the players broken up as ours are (tracklets, slots, a wrong-team slot now and then, NaN speeds, a cap on the frames
    where ours tracks few); "context" has tracking's players and a ball whose gaps the hazard fitted on our match draws.
    Every profile but clean needs `calibration`. BenchmarkError, listing the profiles there are, for any other name.
    """
    _check_profile(profile)
    return degrade.degrade(degrade.tracking_view(match), profile, calibration, seed)[0]


def _check_profile(profile):
    if profile not in PROFILES:
        raise BenchmarkError("profile {!r} is not one of: {}".format(profile, ", ".join(sorted(PROFILES))))


# -- every config on every match --------------------------------------------------------------------------------------

def evaluate_configs(config_paths, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None,
                     calibration=None, truth="raw", data_dir=None, log=print, drop_match_only=False, pools=None):
    """Run each config on each match and score it; the result holds a JSON-ready report per config.

    `config_paths` are TOML files for pipeline.config.load_config; the first is the reference of the comparisons.
    `profile` names how the tracking is made (PROFILES; the default is tracking) and `seeds` is how many degradation
    seeds to run, 0 up to seeds - 1 (None, the default, is DEFAULT_SEEDS). The clean profile has none to repeat: its one
    seed is 0, and a note says so when a number of seeds was asked for (and is not 1) and not otherwise. `match_ids` are
    the matches, in the order to run them (default: all seven, sources.MATCH_IDS' order; a report lists them in id order
    whatever order they ran in); they are read through
    dfl.load_match from `data_dir` (default games.DATA_DIR). `matches` is {id: DflMatch} to use instead of loading (the
    checks pass synthetic ones). `calibration` is the Calibration of BAR-ATH that a degraded profile takes its gaps
    from; none given, calibration.load(data_dir) reads it from its cache or measures it, before the first match; the
    clean profile degrades nothing and refuses one, so that a report cannot suggest that it was used. `truth` is "raw"
    or "coarse" (truth.coarse_codes). `log` receives each progress line and then the printed table. `drop_match_only`
    (default False) removes, from every config, the evidence sources that declare MATCH_ONLY (evidence.match_only: they
    need an asset a benchmark match does not have), runs what is left, and records the removed names in each report's
    "dropped_match_only" and in the printed block; see drop_match_only().

    A config that fills ([inputs] ball_fill = "broadcast", R028) is served a simulated broadcast ball on each match and
    seed (broadcast_sim.served_inputs_for: its twin without the fill is decoded on the plain inputs and drives the
    transplant), and `pools` is the broadcast_sim.Pools it is made from; none given, broadcast_sim.load_pools(data_dir)
    reads them from their cache before the calibration and the first match, and never builds them: an absent or
    broken cache is a BenchmarkError that says to run `broadcast_sim.py pools` first. broadcast_sim is imported only
    for a run in which a config fills. The twin's codes are kept per (match, seed) and shared by every config of the
    run with the same content, so a config whose content is a twin's is not decoded twice. Every report of a run that
    loaded the pools holds their digest ("broadcast_sim"), the reports of the configs that do not fill too, so that
    choose.py compares the reference with the candidate; a run in which no config fills holds null, pools given or not.

    Everything that can be refused is, before the first match is loaded: no config, one that does not load (ConfigError,
    from the pipeline's own loader), one with scope = "file", one whose name has a dot, two with the same name, an
    unknown profile or truth or match, a repeated match, fewer than one seed, a calibration that is not one or that the
    profile does not take, pools that are not a Pools for a run that fills. BenchmarkError says which. A source that
    needs an asset that a benchmark match does not have (the video's homography, the Smart Tagging) stops the run with
    its config, its name and kind and the asset named, instead of scoring zeros.

    Returns {"profile", "seeds", "truth", "matches", "git", "data_dir", "reference", "configs", "rank",
    "broadcast_sim"}; "configs" is a list in the order given of {"stem", "report", "scores", "seconds"}: the report is
    what write_reports writes (a function of the config, the profile, the matches and the calibration, with no timing),
    `scores` the HalfScores it was made from and `seconds` the time spent on the config (printed, never written). "rank"
    is rank_agreement of the configs, None unless at least two of them are among PUBLISHED_ORDER.
    """
    if truth not in TRUTHS:
        raise BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(TRUTHS)))
    return evaluate_truths(config_paths, profile=profile, seeds=seeds, match_ids=match_ids, matches=matches,
                           calibration=calibration, truth_names=(truth,), data_dir=data_dir, log=log,
                           drop_match_only=drop_match_only, pools=pools)[truth]


def evaluate_truths(config_paths, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None,
                    calibration=None, truth_names=TRUTHS, data_dir=None, log=print, drop_match_only=False,
                    pools=None):
    """{truth: result}: evaluate_configs under each of `truth_names`, every config decoded once per match and seed.

    Each config's codes on a match and seed are scored against each truth in turn, so a result is exactly what
    evaluate_configs gives under that truth alone (the same report, byte for byte once written, and the same printed
    table), and the decoding, nearly all of a config's time, is done once. A match's progress line names the truths
    joined by "+", and the tables are logged after the last match, in the order of `truth_names`. `truth_names` is one
    or more of TRUTHS, none repeated, and BenchmarkError otherwise, before anything is loaded. The other arguments are
    evaluate_configs'.
    """
    truth_names = tuple(truth_names) if isinstance(truth_names, (list, tuple)) else (truth_names,)
    if not truth_names or any(name not in TRUTHS for name in truth_names) or len(set(truth_names)) < len(truth_names):
        raise BenchmarkError("truths {!r}: name one or more of {}, each once".format(truth_names, ", ".join(TRUTHS)))
    configs, dropped = _load_configs(config_paths, drop_match_only)
    _check_profile(profile)
    asked_for_seeds = seeds is not None
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    if profile == CLEAN and calibration is not None:
        raise BenchmarkError("profile {!r} takes no calibration, and one was given: it would be ignored, and a report "
                             "must not suggest that it was used".format(profile))
    if profile != CLEAN and calibration is not None and not isinstance(calibration, calibration_module.Calibration):
        raise BenchmarkError("calibration must be a Calibration of our own match, as calibration.load() returns, and "
                             "it is a {}".format(type(calibration).__name__))
    ids = resolve_match_ids(match_ids, matches)
    filling = any(config.ball_fill != "none" for config in configs)
    simulator = None
    if filling:
        from pipeline.benchmark import broadcast_sim as simulator   # here, not above: broadcast_sim imports this module
        if pools is not None and not isinstance(pools, simulator.Pools):
            raise BenchmarkError("pools must be the Pools of the simulated broadcast (broadcast_sim.load_pools()), and "
                                 "it is a {}".format(type(pools).__name__))
    if filling and pools is None:
        try:
            pools = simulator.load_pools(data_dir=data_dir)      # read only: `broadcast_sim.py pools` builds them
        except simulator.BroadcastSimError as error:
            raise BenchmarkError(str(error)) from error
    if profile != CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    broadcast_sim = simulator.digest(pools) if filling else None
    data_dir_used = data_dir or games.DATA_DIR
    seed_list = [0] if profile == CLEAN else list(range(seeds))
    if profile == CLEAN and asked_for_seeds and seeds != 1:
        log("note      profile {} has nothing random to repeat: seeds ignored, seed 0 only".format(CLEAN))

    scores = {name: {config.stem: [] for config in configs} for name in truth_names}
    seconds = {config.stem: 0.0 for config in configs}
    records = []
    drivers = {}                # the codes of a twin, and of every config of the run, per (match, profile, seed)
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise BenchmarkError("the match given as {} says it is {}; its scores would be filed under the wrong id"
                                 .format(match_id, match.match_id))
        codes_raw, carried = truths.truth_codes(match)
        codes_true = {name: truths.coarse_codes(codes_raw, match.graded, match.fps) if name == "coarse" else codes_raw
                      for name in truth_names}
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
                if config.ball_fill == "none":
                    codes = _codes(config, inputs)
                    if filling:     # the driver of a twin with this content, if the run has one
                        drivers.setdefault(simulator.driver_slot(config, match.match_id, profile, seed), codes)
                else:
                    codes = _codes(config, _served(simulator, config, match, atd, profile, seed, data_dir, pools,
                                                   inputs, drivers))
                for name in truth_names:
                    scores[name][config.stem] += score.score_halves(codes_true[name], codes, match.graded,
                                                                    match.match_id, seed)
                seconds[config.stem] += time.perf_counter() - started
            del atd, inputs, codes
        drivers.clear()                          # one match in memory at a time, its codes too
        records.append(record)
        log("match     {}   {:,} rows, {:,} graded   truth {}".format(
            record["id"], record["frames"], record["graded_frames"], "+".join(truth_names)))
        del match, codes_raw, codes_true         # the next match is loaded only when this one is gone

    # The matches ran in the order asked for, and a report is a function of the matches and not of that order: its
    # records and halves are in match-id order (a stable sort, so a match's seeds and halves stay as they were run).
    records.sort(key=lambda record: record["id"])
    for by_stem in scores.values():
        for stem in by_stem:
            by_stem[stem].sort(key=lambda half: half.match_id)

    git = pipeline_run.git_state()
    environment = _environment()
    profile_info = degrade.profile_record(profile)       # name, version and, for context, tracking's version
    calibration_info = None if profile == CLEAN else {"sha256": calibration_module.digest(calibration),
                                                      "version": calibration_module.CALIBRATION_VERSION}
    results = {}
    for name in truth_names:
        truth_info = {"name": name, "min_s": truths.COARSE_MIN_S if name == "coarse" else None}
        results[name] = _result(configs, scores[name], seconds, records, seed_list, profile_info, truth_info, git,
                                environment, calibration_info, data_dir_used, dropped, broadcast_sim)
    for name in truth_names:
        for line in render(results[name]):
            log(line)
    return results


def _result(configs, scores, seconds, records, seed_list, profile_info, truth_info, git, environment,
            calibration_info, data_dir_used, dropped, broadcast_sim=None):
    """One truth's result, as evaluate_configs returns it: a report per config, each against the first."""
    reference = configs[0]
    entries = []
    for config in configs:
        versus = None if config is reference else _versus(reference, config, scores)
        report = _report(config, scores[config.stem], records, seed_list, profile_info, truth_info, git, environment,
                         versus, calibration_info, dropped[config.stem], broadcast_sim)
        entries.append({"stem": config.stem, "report": report, "scores": scores[config.stem],
                        "seconds": seconds[config.stem]})
    return {"profile": profile_info, "seeds": seed_list, "truth": truth_info, "matches": records, "git": git,
            "data_dir": data_dir_used, "reference": reference.stem, "configs": entries,
            "calibration": calibration_info, "broadcast_sim": broadcast_sim,
            "rank": rank_agreement({config.stem: score.pooled(scores[config.stem])["spell_f1_mean"]
                                    for config in configs})}


def drop_match_only(config):
    """(config without its match-only evidence, the names of what was removed, in config order).

    A source declares MATCH_ONLY in its own module (evidence.match_only) when it needs an asset that only our match
    has, the Smart Tagging or the whistle: a DFL match has neither, so a config that uses one cannot be benchmarked as
    it is, and the runner says so instead of scoring zeros. This is the explicit way round it: the same config
    without those sources, never silently. `text` and `path` are kept as written, so a report quotes the file that was
    given (and a latency file hashes it); only the evidence that runs changes, and the names removed go in the report,
    so that nobody reads its numbers as the full config's. A config with nothing flagged is returned as it was.
    """
    flagged = [component for component in config.evidence if evidence_stage.match_only(component.kind)]
    if not flagged:
        return config, ()
    kept = tuple(component for component in config.evidence if component not in flagged)
    return dataclasses.replace(config, evidence=kept), tuple(component.name for component in flagged)


def _load_configs(config_paths, drop_match_only_sources=False):
    """([Config], {stem: names dropped}): loaded by the pipeline's own loader, each decoded per half, names unique.

    With `drop_match_only_sources` each config is passed through drop_match_only; the dict is then the names removed
    from each, () for a config that lost none and for every config when the flag is off.
    """
    config_paths = list(config_paths)
    if not config_paths:
        raise BenchmarkError("no config given; name at least one CONFIG.toml, and put the one to beat first (the "
                             "others are compared with it)")
    configs, paths, dropped = [], {}, {}
    for path in config_paths:
        config = load_config(path)
        check_scope(config)
        if "." in config.stem:
            raise BenchmarkError(
                "{}: the file name of a config is the name of its report, whose parts "
                "(<name>.coarse.m-<ids>.s<n>.json) are told apart by dots, so a dot in the name would be read as one "
                "of them and the report would be another run's; rename {} without a dot".format(
                    config.stem, shown(config.path)))
        if config.stem in paths:
            where = paths[config.stem]
            if os.path.normcase(where) == os.path.normcase(config.path):
                raise BenchmarkError("{} is given twice; a config is run once".format(config.stem))
            raise BenchmarkError("{} and {} have the same name, {}, so they would write the same report; rename one"
                                 .format(shown(where), shown(config.path), config.stem))
        paths[config.stem] = config.path
        if drop_match_only_sources:
            config, dropped[config.stem] = drop_match_only(config)
        else:
            dropped[config.stem] = ()
        configs.append(config)
    return configs, dropped


def resolve_match_ids(match_ids, matches):
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


def _served(simulator, config, match, atd, profile, seed, data_dir, pools, plain, drivers):
    """broadcast_sim.served_inputs_for, its refusals as this module's BenchmarkError.

    Run by path, this file is __main__, and broadcast_sim imports a second copy of it as pipeline.benchmark.run: the
    twin's missing asset, raised in there, is that copy's BenchmarkError, another class than this one's. It is raised
    again as this one's, with its sentence, so that main catches it without importing broadcast_sim.
    """
    try:
        return simulator.served_inputs_for(config, match, atd, profile, seed, data_dir, pools, plain=plain,
                                           drivers=drivers)
    except (simulator.run.BenchmarkError, simulator.BroadcastSimError) as error:
        if isinstance(error, BenchmarkError):
            raise
        raise BenchmarkError(str(error)) from error


def _codes(config, inputs):
    """predict's codes; an asset that a benchmark match does not have is a BenchmarkError that names its source."""
    try:
        return predict(config, inputs)[0]
    except match_inputs.AssetUnavailable as error:
        raise BenchmarkError(_unavailable(config, inputs, error)) from error


def _unavailable(config, inputs, error):
    """The sentence for an AssetUnavailable: which evidence source it was, found by running each source on its own.

    A config that fills on inputs that serve no broadcast ball is said to be that first: the fill asked for the ball
    before any source ran, and the cause is how the match was served, not a source or the decoder (ruling 12).
    """
    if config.ball_fill != "none" and not _serves_broadcast_ball(inputs):
        return ("{}: [inputs] ball_fill = {!r} fills our ball from the broadcast's, and the simulated broadcast ball "
                "is missing from these inputs: a benchmark match has one only when it is served through "
                "broadcast_sim.served_inputs_for, as evaluate_truths serves a config that fills. {}".format(
                    config.stem, config.ball_fill, error))
    run_context = RunContext(inputs=inputs, seed=config.seed)
    for component in config.evidence:
        try:
            pipeline_run.compute_evidence(dataclasses.replace(config, evidence=(component,)), run_context)
        except match_inputs.AssetUnavailable as own:
            # The hint only for a source that declares MATCH_ONLY: with the flag on it would already be gone, and
            # for any other source the flag would not help.
            hint = DROP_HINT if evidence_stage.match_only(component.kind) else ""
            return ("{}: the evidence source {!r} (kind {!r}) asked for an asset that a benchmark match does not have, "
                    "so this config cannot be benchmarked as it is. {}{}".format(config.stem, component.name,
                                                                                  component.kind, own, hint))
    refiners = "".join(", refiner {!r}".format(component.kind) for component in config.refine)
    return ("{}: no evidence source asked for it on its own, so it was the decoder {!r}{} that asked for an asset a "
            "benchmark match does not have. {}".format(config.stem, config.decoder.kind, refiners, error))


def _serves_broadcast_ball(inputs):
    """Whether `inputs` serve a broadcast ball (a MatchInputs given one, BAR-ATH's) or raise AssetUnavailable for
    it."""
    try:
        inputs.broadcast_ball
    except match_inputs.AssetUnavailable:
        return False
    return True


# -- the report -------------------------------------------------------------------------------------------------------

def rank_agreement(pooled_spell_f1):
    """{"published", "observed", "kendall_tau", "concordant", "discordant", "tied", "pairs"} of a run, or None.

    `pooled_spell_f1` is {config stem: pooled spellF1}. Of the configs that are among PUBLISHED_ORDER, "observed" lists
    them best first (a tie, the order they were given in), and score.kendall_tau compares that ranking with the
    published one. None when fewer than two of them are there, because one config has no order. It is a check and never
    a goal (the plan's yardstick check): the published gaps are at or inside the noise.
    """
    present = [stem for stem in PUBLISHED_ORDER if stem in pooled_spell_f1]
    if len(present) < 2:
        return None
    agreement = score.kendall_tau(PUBLISHED_ORDER, pooled_spell_f1)
    observed = sorted(pooled_spell_f1, key=lambda stem: -pooled_spell_f1[stem])
    return {"published": list(PUBLISHED_ORDER), "observed": [stem for stem in observed if stem in present],
            "kendall_tau": agreement["tau"], "concordant": agreement["concordant"],
            "discordant": agreement["discordant"], "tied": agreement["tied"], "pairs": agreement["pairs"]}


def _report(config, halves, records, seed_list, profile, truth, git, environment, versus, calibration,
            dropped=(), broadcast_sim=None):
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
        "config": {"path": shown(config.path), "text": config.text.replace("\r\n", "\n")},
        "dropped_match_only": list(dropped),    # what --drop-match-only removed; [] if nothing, or the flag is off
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
        "calibration": calibration,          # {sha256, version} of the Calibration degraded with; None for clean
        "broadcast_sim": broadcast_sim,      # the digest of the pools the run loaded (a config of it fills), else None
    }
    if profile["name"] in SENSITIVITY_ONLY:
        report["sensitivity_only"] = SENSITIVITY_ONLY[profile["name"]]      # only for a profile that is one
    return plain(report)


def _versus(first, config, scores):
    """`config` against `first`, the reference: score.compare (b - a) for each of METRICS, with the interval's say."""
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


def plain(value):
    """`value` as JSON can hold it, whatever it is made of: rounded, with text keys, and without NaN or numpy types.

    Floats are rounded to DECIMALS (and -0.0 is 0.0), a NaN or an infinity (an interval of one match) becomes None,
    dictionary keys become text (the IoUs 0.3 ... 0.7 are "0.3" ... "0.7"), tuples become lists, numpy scalars Python
    ones. TypeError for anything else, so that a new field cannot reach a report unchecked.
    """
    if isinstance(value, dict):
        return {_key(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
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


def shown(path):
    """A path as people read it: relative to the repository when inside it, absolute otherwise, with forward slashes."""
    path = os.path.abspath(path)
    if os.path.splitdrive(path)[0].lower() == os.path.splitdrive(ROOT)[0].lower():
        relative = os.path.relpath(path, ROOT)
        if not relative.startswith(".."):
            return relative.replace(os.sep, "/")
    return path.replace(os.sep, "/")


def report_path(data_dir, profile, stem, truth, match_ids=None, seeds=None):
    """<data dir>/runs/benchmark/<profile>/<stem>[.<truth>][.m-<ids>][.s<n>].json: where a report goes.

    `.<truth>` is left out for the raw truth. `.m-<the ids, sorted, joined by "-">` is added when `match_ids` are not
    exactly the registry's seven (a subset, whatever its order), so that a run on one match or two never replaces the
    report of all seven that the same config wrote before: a full report is the one with none of these suffixes of its
    own. `.s<n>` is added when `seeds`, the number of degradation seeds the run used, is given and is not DEFAULT_SEEDS,
    for the same reason: a one-seed look at a degraded profile is not the three-seed report. The clean profile has no
    seeds to count (it runs seed 0 whatever it was asked), so its name never has one.
    """
    parts = [stem]
    if truth != "raw":
        parts.append(truth)
    parts += name_suffixes(profile, match_ids, seeds)
    return os.path.join(data_dir, *REPORT_FOLDER, profile, ".".join(parts) + ".json")


def name_suffixes(profile, match_ids=None, seeds=None):
    """["m-<ids>", "s<n>"], those of them that a file name must carry for a run of this profile, matches and seeds.

    The matches when they are not exactly the registry's seven, whatever their order; the number of seeds when it is
    given, not DEFAULT_SEEDS, and the profile is a degraded one (the clean profile has one seed, always). Shared with
    fidelity.py, whose reports follow the same rule for the same reason.
    """
    parts = []
    if match_ids is not None and set(match_ids) != set(sources.MATCH_IDS):
        parts.append("m-" + "-".join(sorted(match_ids)))
    if profile != CLEAN and seeds is not None and seeds != DEFAULT_SEEDS:
        parts.append("s{}".format(seeds))
    return parts


def write_reports(result, data_dir=None):
    """Write each config's report under `data_dir` (default: the one the run read from); the paths, in order.

    The file name carries the truth when it is not the raw one, the matches when they are not all seven and the number
    of seeds when it is not the default (report_path). The text is written by write_json: atomically, so that a run
    that is interrupted never leaves half a report and never damages the one that was there.
    """
    data_dir = data_dir or result["data_dir"]
    ids = [record["id"] for record in result["matches"]]
    paths = []
    for entry in result["configs"]:
        path = report_path(data_dir, result["profile"]["name"], entry["stem"], result["truth"]["name"], ids,
                           len(result["seeds"]))
        write_json(path, entry["report"])
        paths.append(path)
    return paths


def write_json(path, document):
    """`document` (JSON-ready: see plain) at `path`: sorted keys, indent 2, ASCII, LF and a final newline, atomically.

    The text goes to a temporary file in the same folder, renamed over the old file (os.replace), so that a run that is
    interrupted, or a disk that fills, never leaves half a file and never damages the one that was there. A failed
    write removes its temporary file and raises. fidelity.py writes its report the same way.
    """
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
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


# -- the printout -----------------------------------------------------------------------------------------------------

def render(result):
    """The lines that evaluate_configs logs: one block per config, a table of them, and each against the reference."""
    reports = [(entry["stem"], entry["report"], entry["seconds"]) for entry in result["configs"]]
    git = result["git"]
    ids = [record["id"] for record in result["matches"]]
    profile = result["profile"]
    versions = "version {}".format(profile["version"]) + (
        ", on tracking version {}".format(profile["tracking_version"]) if "tracking_version" in profile else "")
    lines = ["",
             "profile   {} ({})   truth {}   seeds {}   {} matches   commit {}{}".format(
                 profile["name"], versions, result["truth"]["name"], ",".join(str(seed) for seed in result["seeds"]),
                 len(ids), git["commit"] or "?", " +dirty" if git["dirty"] else "")]
    if profile["name"] in SENSITIVITY_ONLY:
        lines.append("note      profile {} {}".format(profile["name"], SENSITIVITY_ONLY[profile["name"]]))
    if result["calibration"] is not None:
        lines.append("calibration   {}... (version {}): {}".format(
            result["calibration"]["sha256"][:12], result["calibration"]["version"], PROVENANCE[profile["name"]]))
    if result.get("broadcast_sim") is not None:
        lines.append("broadcast_sim {}...: the pools of the simulated broadcast ball (BAR-ATH's gap fills and near-gap "
                     "errors, broadcast_sim.py), served to the configs that fill".format(result["broadcast_sim"][:12]))
    for stem, report, seconds in reports:
        pooled, interval = report["pooled"], report["t_interval"]["spell_f1_mean"]
        rate, labels = report["possessions_per_graded_minute"], report["labels"]
        lines += ["",
                  "config    {}".format(stem)]
        if report["dropped_match_only"]:
            lines.append("          dropped         match-only sources: {} (--drop-match-only; the numbers below are "
                         "those of the config without them)".format(", ".join(report["dropped_match_only"])))
        lines += [
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
                      for match_id in sorted(ids)),
                  "          seconds         {:.1f}".format(seconds)]
    lines += ["", "{:<22}{:>16}{:>16}{:>22}   {}".format(
        "config", "pooled spellF1", "pooled macroF1", "pooled alive/dead F1",
        "mean per-match spellF1 [{} t-interval]".format(_level()))]
    for stem, report, _ in reports:
        pooled, interval = report["pooled"], report["t_interval"]["spell_f1_mean"]
        lines.append("{:<22}{:>16}{:>16}{:>22}   {} {}".format(
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
        lines += ["", "wins and losses are matches whose delta is beyond {:g} either way (score.FLIP_TOLERANCE), "
                      "and an interval excludes 0 only if it clears 0 by more than that.".format(score.FLIP_TOLERANCE),
                  "with {} matches the smallest sign-flip p there can be is {:.4f}; neither it nor the interval "
                  "makes a gap real.".format(len(ids), 2.0 / 2 ** len(ids))]
    rank = result["rank"]
    if rank is not None:
        lines += ["", "order by pooled spellF1   {}".format(" > ".join(rank["observed"])),
                  "published order           {}".format(" > ".join(rank["published"])),
                  "Kendall tau {} of the {} pairs ({} concordant, {} discordant, {} tied); a check of the profile, not "
                  "a goal: the published gaps are at or inside the noise".format(
                      _number(rank["kendall_tau"], signed=True), rank["pairs"], rank["concordant"], rank["discordant"],
                      rank["tied"])]
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


def _level():
    """The coverage of the intervals as the printout names it: "90%"."""
    return "{:.0%}".format(LEVEL)


def _bounds(interval, signed=False):
    """"[low, high]", or "no interval" where there are fewer than two matches to make one of."""
    if interval["low"] is None or interval["high"] is None:
        return "no interval (fewer than 2 matches)"
    return "[{}, {}]".format(_number(interval["low"], signed), _number(interval["high"], signed))


def _interval(interval, excludes=None, signed=False):
    """"<level> t-interval [low, high]" and, for a delta, whether it excludes 0 (by more than the tie tolerance)."""
    bounds = _bounds(interval, signed)
    if interval["low"] is None or interval["high"] is None:
        return bounds
    text = "{} t-interval ".format(_level()) + bounds
    if excludes is None:
        return text
    return text + (" excludes 0" if excludes else " includes 0")


# -- the command line -------------------------------------------------------------------------------------------------

def profile_help():
    """The --profile help: the profiles, the default, and which of them are a sensitivity check and not the
    yardstick."""
    return "how the tracking is made: {} (default {}){}".format(
        ", ".join(sorted(PROFILES)), DEFAULT_PROFILE,
        "".join("; {} {}".format(name, reason) for name, reason in sorted(SENSITIVITY_ONLY.items())))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run pipeline configs on the benchmark's matches; the first config is the reference that the "
                    "others are compared with.")
    parser.add_argument("configs", nargs="+", metavar="CONFIG",
                        help="TOML configs, e.g. pipeline/benchmark/configs/nearest_w50.toml; the first is the "
                             "reference that the others are compared with")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {}; the clean profile has just one, and says so "
                             "if you ask for more)".format(DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--truth", choices=TRUTHS + (BOTH,), default="raw",
                        help="the truth to score against: raw, or coarse (short sandwiched possessions absorbed), or "
                             "both from one decode (the reports and tables of the two runs)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--no-write", action="store_true", help="print only; write no report")
    parser.add_argument("--drop-match-only", action="store_true",
                        help="drop the evidence sources that need an asset a benchmark match does not have "
                             "(MATCH_ONLY), and record them in the report")
    args = parser.parse_args(argv)

    try:
        results = evaluate_truths(args.configs, profile=args.profile, seeds=args.seeds, match_ids=args.matches,
                                  truth_names=TRUTHS if args.truth == BOTH else (args.truth,), data_dir=args.data_dir,
                                  drop_match_only=args.drop_match_only)
        paths = [] if args.no_write else [path for result in results.values() for path in write_reports(result)]
    except (ConfigError, BenchmarkError, OSError) as error:
        # The expected failures get a sentence: a config to fix, a profile or match that does not exist, a source that
        # needs the video, data not downloaded yet, no cache of the simulated broadcast's pools (evaluate_truths
        # raises it as a BenchmarkError). Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1
    print("")
    for path in paths:
        print("report    {}".format(shown(path)))
    if args.no_write:
        print("report    not written (--no-write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
