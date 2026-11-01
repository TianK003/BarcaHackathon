"""Checks of choose (pipeline/benchmark/choose.py): a grid's declared rule applied to the runner's written reports.

choose decides which config a grid adopts, so what is held against it here is the rule worked out by hand from the
runner's own results (the band, the coarse tie-break, the order given for a tie that is left), its deltas equal to the
runner's comparison, the leave-one-match-out picks and deltas recomputed in a plain loop, and its refusals: a report
that is missing, one whose config has changed since it was written, and reports of different runs. The matches are the
synthetic ones of checks/common.py, under the gaps profile and the synthetic Calibration; their scores mean nothing.

R024's objective (objective="mean") is held to the same: its choice, its maximin tie-break and its lomo recomputed
by hand, its comparison the mean of the two truths'. The synthetic nearest_w0 is the best under both truths, which
would let a wrong rule pick it too, so those checks first make PERFECT perfect under the coarse truth alone
(perfect_coarse): then the raw rule, the mean and the maximin each pick another config, and its raw and mean p differ.
objective="raw", the default, is held to choose.py as it was before the mean was added, run from git, to the bit.
"""

import json
import os
import shutil
import subprocess
import tempfile
import types

from pipeline.benchmark import choose, score
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import NotRun, synthetic_calibration, synthetic_matches, yardstick_path
from pipeline.selftest import expect_error, quiet

# Params
WIDTHS = (50, 25, 0, 125)          # the reference first, as a grid's process lists main.toml first
SEEDS = 2
BEFORE_MEAN = "7de5fd4c8e623b7eadd2832f74dad04dc97bb31e"   # the last commit of choose.py before --objective (P12's)
MEAN_STEMS = ["nearest_w25", "nearest_w0", "nearest_w125"]  # the candidates of the objective="mean" checks, in order
PERFECT = "nearest_w25"            # perfect under the coarse truth: its raw match deltas have two signs, its mean's one


def written(root, paths, seeds=SEEDS):
    """evaluate_truths on the gaps profile, both truths, with the reports written under `root`; the results."""
    results = benchmark_run.evaluate_truths(paths, profile="gaps", matches=synthetic_matches(), seeds=seeds,
                                            calibration=synthetic_calibration(), data_dir=root, log=quiet)
    for result in results.values():
        benchmark_run.write_reports(result, data_dir=root)
    return results


def chosen(root, stems, seeds=SEEDS, band=choose.TIE_BAND, metric="spell", **options):
    """choose.choose on the reports under `root`, against nearest_w50; `options` (objective=) are passed as they are,
    so that the checks written before the mean call choose exactly as they did."""
    return choose.choose(stems, "nearest_w50", profile="gaps", band=band, data_dir=root,
                         match_ids=sorted(synthetic_matches()), seeds=seeds, metric=metric, **options)


def perfect_coarse(root, stem):
    """Rewrite `stem`'s coarse report under `root` so that its spellF1 is 1.0 (every possession matched), its raw one
    and its confusion (macroF1) untouched: the config is then the best under the coarse truth and by the mean, while its
    min(raw, coarse) stays its low raw score."""
    path = benchmark_run.report_path(root, "gaps", stem, "coarse", sorted(synthetic_matches()), SEEDS)
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    for half in report["halves"]:
        half["n_pred"] = half["n_gt"]
        half["tp"] = [half["n_gt"]] * len(half["tp"])
    benchmark_run.write_json(path, report)


def pooled_of(root, stems, key="spell_f1_mean"):
    """{truth: {stem: score.pooled(the halves of its report)[key]}}: the reports' scores as they are on disk."""
    return {truth: {stem: score.pooled(halves_of(root, stem, truth))[key] for stem in stems}
            for truth in benchmark_run.TRUTHS}


def test_choose_applies_the_rule_and_matches_the_runner():
    """Scores and deltas equal the runner's; the choice and both lomo rules equal a plain recomputation."""
    paths = [yardstick_path(width) for width in WIDTHS]
    stems = ["nearest_w{}".format(width) for width in WIDTHS[1:]]
    with tempfile.TemporaryDirectory() as root:
        results = written(root, paths)
        pooled = {truth: {entry["stem"]: score.pooled(entry["scores"])["spell_f1_mean"]
                          for entry in results[truth]["configs"]} for truth in benchmark_run.TRUTHS}
        for band in (0.0, 1.0):                 # nothing tied but the best; everything tied, so coarse decides
            choice = chosen(root, stems, band=band)
            for stem in stems + ["nearest_w50"]:
                for truth in benchmark_run.TRUTHS:
                    assert choice["rows"][stem][truth]["spell"] == pooled[truth][stem], (stem, truth)
            raw, coarse = pooled["raw"], pooled["coarse"]
            best = max(stems, key=lambda stem: (raw[stem], -stems.index(stem)))
            tied = [stem for stem in stems if raw[stem] >= raw[best] - band]
            expected = max(tied, key=lambda stem: (coarse[stem], -stems.index(stem)))
            assert (choice["best"], choice["tied"], choice["chosen"]) == (best, tied, expected), (band, choice["tied"])
            entries = {entry["stem"]: entry for entry in results["raw"]["configs"]}
            runner = entries[expected]["report"]["versus"]["spell_f1_mean"]
            assert abs(choice["versus"]["raw"]["pooled_delta"] - runner["pooled_delta"]) < 1e-9
            assert choice["versus"]["raw"]["wins"] == runner["wins"]
        assert chosen(root, stems, band=1.0)["chosen"] == max(stems, key=lambda stem: pooled["coarse"][stem]), (
            "with every config tied, the coarse score must decide")

        choice = chosen(root, stems)
        rows = choice["rows"]
        matches = sorted(rows["nearest_w50"]["raw"]["per_match"])
        for name in ("objective", "rule"):
            for held in matches:
                kept = [m for m in matches if m != held]
                raw = {s: sum(rows[s]["raw"]["per_match"][m] for m in kept) / len(kept) for s in stems}
                coarse = {s: sum(rows[s]["coarse"]["per_match"][m] for m in kept) / len(kept) for s in stems}
                top = max(stems, key=lambda s: (raw[s], -stems.index(s)))
                if name == "rule":
                    tied = [s for s in stems if raw[s] >= raw[top] - choose.TIE_BAND]
                    top = max(tied, key=lambda s: (coarse[s], -stems.index(s)))
                assert choice["lomo"][name]["picks"][held] == top, (name, held)
                for truth in benchmark_run.TRUTHS:
                    delta = rows[top][truth]["per_match"][held] - rows["nearest_w50"][truth]["per_match"][held]
                    assert choice["lomo"][name][truth]["deltas"][held] == delta, (name, held, truth)


def test_choose_selects_on_macro_when_asked():
    """metric="macro" selects, compares and averages on macroF1; the default stays spellF1; a bad metric is refused."""
    stems = ["nearest_w25", "nearest_w0", "nearest_w125"]
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(width) for width in WIDTHS])
        before = chosen(root, stems, band=0.0)
        loser = min(stems, key=lambda stem: before["rows"][stem]["raw"]["spell"])
        # Rewrite the confusion of the worst spellF1 config's reports to a perfect one: spellF1 reads the tp counts and
        # is untouched, macroF1 reads the confusion and now ranks that config first, under both truths.
        for truth in benchmark_run.TRUTHS:
            path = benchmark_run.report_path(root, "gaps", loser, truth, sorted(synthetic_matches()), SEEDS)
            with open(path, encoding="utf-8") as stream:
                report = json.load(stream)
            for half in report["halves"]:
                half["confusion"] = [[100 if i == j else 0 for j in range(len(half["confusion"][0]))]
                                     for i in range(len(half["confusion"]))]
            benchmark_run.write_json(path, report)
        spell = chosen(root, stems, band=0.0)
        macro = chosen(root, stems, band=0.0, metric="macro")
        assert spell["chosen"] != loser and spell["chosen"] == before["chosen"], (
            "the default must still pick on spellF1")
        assert spell["metric"] == "spell" and macro["metric"] == "macro"
        assert macro["chosen"] == loser == max(stems, key=lambda s: macro["rows"][s]["raw"]["macro"]), macro["chosen"]
        assert macro["best"] == loser and macro["tied"] == [loser]
        scores = {truth: {stem: halves_of(root, stem, truth) for stem in ["nearest_w50"] + stems}
                  for truth in benchmark_run.TRUTHS}
        for truth in benchmark_run.TRUTHS:
            expected = score.compare(scores[truth]["nearest_w50"], scores[truth][loser], metric="macro_f1")
            for key in ("pooled_delta", "mean_delta", "p_sign_flip", "wins"):
                assert macro["versus"][truth][key] == expected[key], (truth, key)
            assert macro["rows"][loser][truth]["delta"] == expected["pooled_delta"], truth
        assert macro["versus"]["raw"]["pooled_delta"] != spell["versus"]["raw"]["pooled_delta"]
        per_match = score.per_match(scores["raw"][loser])
        assert macro["rows"][loser]["raw"]["per_match"] == {m: e["macro_f1"] for m, e in per_match.items()}
        assert macro["lomo"]["objective"]["picks"] == {m: loser for m in per_match}, "lomo must use macroF1 too"
        assert any("metric macroF1" in line for line in choose.render(macro)), "the header must name the metric"
        expect_error(choose.ChooseError, lambda: chosen(root, stems, metric="accuracy"), "metric")


def test_choose_cli_makes_a_macro_grid_name_its_band():
    """--metric macro without --band is refused before any report is read (0.005 is spellF1's band); with it, accepted."""
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = choose.main(["x", "--reference", "x", "--metric", "macro"])
    assert code == 1 and "band" in out.getvalue(), (code, out.getvalue())
    for argv in (["x", "--reference", "x", "--metric", "macro", "--band", "0.002"], ["x", "--reference", "x"]):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            choose.main(argv)                   # no reports exist here: it fails on those, not on the band
        assert "must name its band" not in out.getvalue(), argv


def halves_of(root, stem, truth):
    path = benchmark_run.report_path(root, "gaps", stem, truth, sorted(synthetic_matches()), SEEDS)
    with open(path, encoding="utf-8") as stream:
        return choose.halves(json.load(stream))


def test_choose_refuses_missing_stale_and_mixed_reports():
    """A missing report, a config changed since its report, and reports of two different runs each say so."""
    with tempfile.TemporaryDirectory() as root:
        config = os.path.join(root, "copied.toml")
        shutil.copyfile(yardstick_path(25), config)
        written(root, [yardstick_path(50), config])
        assert chosen(root, ["copied"])["chosen"] == "copied"
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w0"]), "nearest_w0", "no report")
        with open(config, "a", encoding="utf-8") as stream:
            stream.write("\n# edited after the run\n")
        expect_error(choose.ChooseError, lambda: chosen(root, ["copied"]), "copied", "changed")
        written(root, [yardstick_path(50), yardstick_path(25)])
        path = benchmark_run.report_path(root, "gaps", "nearest_w25", "coarse", sorted(synthetic_matches()), SEEDS)
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        report["calibration"]["sha256"] = "0" * 64                  # as if degraded with another Calibration
        benchmark_run.write_json(path, report)
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w25"]), "nearest_w25", "calibration")
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w25", "nearest_w25"]), "each once")


def test_choose_refuses_mixed_broadcast_sim():
    """Reports whose runs loaded different pools of the simulated broadcast (R028) are of different runs: SAME holds
    broadcast_sim, and a report with another digest is refused by name. A report written before the field existed
    reads as null, as a run with no config that fills writes it."""
    assert choose.SAME == ("matches", "seeds", "profile", "calibration", "broadcast_sim"), choose.SAME
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(50), yardstick_path(25)])
        assert chosen(root, ["nearest_w25"])["chosen"] == "nearest_w25"
        path = benchmark_run.report_path(root, "gaps", "nearest_w25", "raw", sorted(synthetic_matches()), SEEDS)
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        assert report["broadcast_sim"] is None
        del report["broadcast_sim"]                                 # as a report written before R028
        benchmark_run.write_json(path, report)
        assert chosen(root, ["nearest_w25"])["chosen"] == "nearest_w25", "no field reads as null"
        report["broadcast_sim"] = "ab" * 32                         # as if the run had loaded pools of its own
        benchmark_run.write_json(path, report)
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w25"]), "nearest_w25", "raw", "broadcast_sim")


# -- R024's objective: the mean of the two truths ---------------------------------------------------------------------

def test_choose_mean_objective_applies_r024s_rule():
    """objective="mean" (R024): the best mean of the two truths' pooled scores, every config within the band of it
    tied, the larger min(raw, coarse) among the tied, the order given for a tie that is left, recomputed from
    score.pooled for both metrics; its rows and its comparison with the reference are the means of the two truths'."""
    stems = MEAN_STEMS
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(width) for width in WIDTHS])
        perfect_coarse(root, PERFECT)
        for metric, key in (("spell", "spell_f1_mean"), ("macro", "macro_f1")):
            pooled = pooled_of(root, stems + ["nearest_w50"], key)
            raw, coarse = pooled["raw"], pooled["coarse"]
            mean = {stem: (raw[stem] + coarse[stem]) / 2 for stem in raw}
            for band in (0.0, 1.0):             # nothing tied but the best; everything tied, so the maximin decides
                choice = chosen(root, stems, band=band, metric=metric, objective="mean")
                assert choice["objective"] == "mean" and choice["metric"] == metric, (choice["objective"], metric)
                best = max(stems, key=lambda stem: (mean[stem], -stems.index(stem)))
                tied = [stem for stem in stems if mean[stem] >= mean[best] - band]
                expected = max(tied, key=lambda stem: (min(raw[stem], coarse[stem]), -stems.index(stem)))
                assert (choice["best"], choice["tied"], choice["chosen"]) == (best, tied, expected), (
                    metric, band, choice["best"], choice["tied"], choice["chosen"])

                rows = choice["rows"]
                for stem in stems + ["nearest_w50"]:
                    assert rows[stem]["mean"][metric] == mean[stem], (metric, stem)
                    for name in ("spell", "macro"):
                        assert rows[stem]["mean"][name] == (rows[stem]["raw"][name] + rows[stem]["coarse"][name]) / 2
                    assert rows[stem]["mean"]["per_match"] == {
                        match_id: (value + rows[stem]["coarse"]["per_match"][match_id]) / 2
                        for match_id, value in rows[stem]["raw"]["per_match"].items()}, (metric, stem)

                versus = choice["versus"]
                fields = versus["mean"]
                assert set(fields) == set(versus["raw"]), sorted(fields)       # every key that score.compare returns
                assert abs(fields["pooled_delta"] - (versus["raw"]["pooled_delta"] + versus["coarse"]["pooled_delta"])
                           / 2) < 1e-12, (metric, band)
                assert fields["match_deltas"] == {
                    match_id: (delta + versus["coarse"]["match_deltas"][match_id]) / 2
                    for match_id, delta in versus["raw"]["match_deltas"].items()}, (metric, band)
                deltas = list(fields["match_deltas"].values())
                assert (fields["mean_delta"], fields["low"], fields["high"]) == score.t_interval(deltas)
                assert fields["p_sign_flip"] == score.sign_flip_p(deltas), (metric, band)
                assert fields["wins"] == sum(1 for delta in deltas if delta > score.FLIP_TOLERANCE)
                assert fields["losses"] == sum(1 for delta in deltas if delta < -score.FLIP_TOLERANCE)
                assert fields["n"] == len(deltas)
                for stem in stems:              # each row's mean Δ, p and wins, from the runner's comparisons
                    raw_fields, coarse_fields = (
                        score.compare(halves_of(root, "nearest_w50", truth), halves_of(root, stem, truth), metric=key)
                        for truth in benchmark_run.TRUTHS)
                    own = [(delta + coarse_fields["match_deltas"][match_id]) / 2
                           for match_id, delta in raw_fields["match_deltas"].items()]
                    entry = rows[stem]["mean"]
                    assert entry["delta"] == (raw_fields["pooled_delta"] + coarse_fields["pooled_delta"]) / 2, stem
                    assert entry["p"] == score.sign_flip_p(own), (metric, stem, entry["p"])
                    assert entry["wins"] == sum(1 for delta in own if delta > score.FLIP_TOLERANCE), (metric, stem)
                assert rows[expected]["mean"]["delta"] == fields["pooled_delta"], (metric, band)

        # The rule must matter on these reports, or a rule that ignored the mean or the maximin would pass the above.
        raw_rule = {band: chosen(root, stems, band=band) for band in (0.0, 1.0)}
        mean_rule = {band: chosen(root, stems, band=band, objective="mean") for band in (0.0, 1.0)}
        assert mean_rule[0.0]["best"] != raw_rule[0.0]["best"], "the mean must rank otherwise than the raw score"
        assert mean_rule[1.0]["chosen"] not in (raw_rule[1.0]["chosen"], mean_rule[1.0]["best"]), (
            "with every config tied, the maximin must pick neither the coarse tie-break's config nor the best mean")


def test_choose_mean_lomo_matches_a_plain_loop():
    """objective="mean"'s two lomo rules, recomputed in a plain loop: each config's mean over the kept matches of its
    per-match mean of the two truths; the objective alone takes the best of those, the whole rule the band, then the
    larger min of its kept raw and coarse means; the order decides an exact tie. The held-out deltas are reported for
    raw, coarse and their mean, with their interval, p, wins and losses."""
    stems = MEAN_STEMS
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(width) for width in WIDTHS])
        perfect_coarse(root, PERFECT)
        for band in (choose.TIE_BAND, 1.0):
            choice = chosen(root, stems, band=band, objective="mean")
            rows = choice["rows"]
            matches = sorted(rows["nearest_w50"]["raw"]["per_match"])
            for name in ("objective", "rule"):
                deltas = {truth: [] for truth in ("raw", "coarse", "mean")}
                for held in matches:
                    kept = [m for m in matches if m != held]
                    goal = {s: sum((rows[s]["raw"]["per_match"][m] + rows[s]["coarse"]["per_match"][m]) / 2
                                   for m in kept) / len(kept) for s in stems}
                    floor = {s: min(sum(rows[s][truth]["per_match"][m] for m in kept) / len(kept)
                                    for truth in benchmark_run.TRUTHS) for s in stems}
                    top = max(stems, key=lambda s: (goal[s], -stems.index(s)))
                    if name == "rule":
                        tied = [s for s in stems if goal[s] >= goal[top] - band]
                        top = max(tied, key=lambda s: (floor[s], -stems.index(s)))
                    assert choice["lomo"][name]["picks"][held] == top, (band, name, held)
                    for truth in benchmark_run.TRUTHS:
                        deltas[truth].append(rows[top][truth]["per_match"][held]
                                             - rows["nearest_w50"][truth]["per_match"][held])
                    deltas["mean"].append((deltas["raw"][-1] + deltas["coarse"][-1]) / 2)
                assert sorted(choice["lomo"][name]) == sorted(["picks", "raw", "coarse", "mean"]), (band, name)
                for truth, values in deltas.items():
                    cell = choice["lomo"][name][truth]
                    assert cell["deltas"] == dict(zip(matches, values)), (band, name, truth)
                    assert (cell["mean"], cell["low"], cell["high"]) == score.t_interval(values), (band, name, truth)
                    assert cell["p"] == score.sign_flip_p(values), (band, name, truth)
                    assert cell["wins"] == sum(1 for value in values if value > score.FLIP_TOLERANCE)
                    assert cell["losses"] == sum(1 for value in values if value < -score.FLIP_TOLERANCE)
            if band == 1.0:
                assert choice["lomo"]["rule"]["picks"] != choice["lomo"]["objective"]["picks"], (
                    "with every config tied, the lomo's maximin must pick otherwise than the objective alone")


def test_choose_mean_maximin_breaks_a_tie():
    """Two configs tied on the mean within the band: the larger min(raw, coarse) wins although it is named second, and
    a config outside the band is never chosen; a tie on both the mean and the min goes to the config named first."""
    stems = ["first", "second", "third"]
    raw = {"first": 0.50, "second": 0.54, "third": 0.56}
    coarse = {"first": 0.60, "second": 0.555, "third": 0.52}     # means 0.55, 0.5475, 0.54; mins 0.50, 0.54, 0.52
    assert choose._select_mean(stems, raw, coarse, 0.005) == ("first", ["first", "second"], "second")
    assert choose._select_mean(stems, raw, coarse, 0.0) == ("first", ["first"], "first")
    assert choose._select_mean(stems, raw, coarse, 0.02) == ("first", stems, "second")
    assert choose._select(stems, raw, coarse, 0.005)[2] == "third", "the raw rule takes the best raw score"
    # The same scores under the two truths, swapped: the means are equal to the bit, and so are the mins.
    raw, coarse = {"a": 0.5, "b": 0.6}, {"a": 0.6, "b": 0.5}
    assert choose._select_mean(["a", "b"], raw, coarse, 0.0) == ("a", ["a", "b"], "a")
    assert choose._select_mean(["b", "a"], raw, coarse, 0.0) == ("b", ["b", "a"], "b")


def test_choose_mean_lomo_floor_is_the_min_of_the_kept_means():
    """The mean lomo's whole rule breaks a tie on min(kept raw mean, kept coarse mean), not on the kept mean of each
    match's min. X swings between the truths match by match ((0.9, 0.5) and (0.5, 0.9), four of each), so on every fold
    its kept raw and coarse means are both above Y's steady 0.66 (0.671 at worst) while its per-match mins are all 0.5;
    the band ties the two (means 0.70 and 0.66). Y is named first, so neither the wrong floor nor the order picks X."""
    matches = ["M{}".format(number) for number in range(1, 9)]

    def row(pairs):
        per_match = {truth: dict(zip(matches, values)) for truth, values in zip(("raw", "coarse"), zip(*pairs))}
        per_match["mean"] = {m: (per_match["raw"][m] + per_match["coarse"][m]) / 2 for m in matches}
        return {truth: {"per_match": values} for truth, values in per_match.items()}

    rows = {"ref": row([(0.5, 0.5)] * 8), "X": row([(0.9, 0.5), (0.5, 0.9)] * 4), "Y": row([(0.66, 0.66)] * 8)}
    x = rows["X"]
    for held in matches:                        # the property holds on every fold, not only on the whole
        kept = [m for m in matches if m != held]
        floor = min(sum(x[truth]["per_match"][m] for m in kept) / len(kept) for truth in ("raw", "coarse"))
        mins = sum(min(x["raw"]["per_match"][m], x["coarse"]["per_match"][m]) for m in kept) / len(kept)
        assert floor > 0.66 > mins, (held, floor, mins)
    estimate = choose.lomo(["Y", "X"], "ref", rows, 0.05, "mean")
    assert estimate["objective"]["picks"] == {m: "X" for m in matches}, estimate["objective"]["picks"]
    assert estimate["rule"]["picks"] == {m: "X" for m in matches}, (
        "the floor must be the min of the kept means", estimate["rule"]["picks"])


def choose_before_mean():
    """choose.py as it was at BEFORE_MEAN, read from git and run as a module of its own.

    A migration guard: it shows that adding the mean left objective="raw" as 7de5fd4c had it. The old code runs on
    today's run.py and score.py, so retire this check, or re-pin BEFORE_MEAN, when their API changes under it. Without
    git or that commit (a copy of the repository without its history) the check is NOT RUN, never passed."""
    try:
        source = subprocess.run(("git", "show", BEFORE_MEAN + ":pipeline/benchmark/choose.py"), cwd=choose.ROOT,
                                capture_output=True, encoding="utf-8", timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise NotRun("git cannot show choose.py at {} ({}), so the rule before the mean cannot be run".format(
            BEFORE_MEAN[:12], error))
    module = types.ModuleType("choose_before_mean")
    module.__file__ = choose.__file__           # its ROOT, from which it finds the reports' configs, is today's
    exec(compile(source, "choose.py@" + BEFORE_MEAN[:12], "exec"), module.__dict__)
    return module


def test_choose_raw_objective_is_unchanged():
    """objective="raw", the default and given, is choose.py before the mean was added, to the bit: every field of the
    result (rows, best, tied, chosen, versus, lomo, ...) for both metrics and two bands, and render's lines but for the
    header's new `objective raw`."""
    before = choose_before_mean()
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(width) for width in WIDTHS])
        perfect_coarse(root, PERFECT)            # so that the coarse tie-break decides something
        for metric in ("spell", "macro"):
            for band in (choose.TIE_BAND, 1.0):
                old = before.choose(MEAN_STEMS, "nearest_w50", profile="gaps", band=band, data_dir=root,
                                    match_ids=sorted(synthetic_matches()), seeds=SEEDS, metric=metric)
                for new in (chosen(root, MEAN_STEMS, band=band, metric=metric),
                            chosen(root, MEAN_STEMS, band=band, metric=metric, objective="raw")):
                    assert new["objective"] == "raw", new["objective"]
                    rest = {key: value for key, value in new.items() if key != "objective"}
                    assert json.dumps(benchmark_run.plain(rest)) == json.dumps(benchmark_run.plain(old)), (metric, band)
                    lines, old_lines = choose.render(new), before.render(old)
                    assert lines[1:] == old_lines[1:], (metric, band)
                    assert lines[0].replace("   objective raw", "", 1) == old_lines[0] != lines[0], lines[0]
                    assert choose.render(old) == lines, "a selection written before R024 (no objective) must render"


def test_choose_cli_objective_mean():
    """--objective mean reaches choose: the header says so, the comparison has its mean line and the JSON written is
    choose's own selection; without the flag the header says raw; an objective that is not one is refused, by name."""
    import contextlib
    import io
    assert choose.OBJECTIVES == ("raw", "mean"), choose.OBJECTIVES
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(width) for width in WIDTHS])
        ids = sorted(synthetic_matches())
        for stem in ["nearest_w50"] + MEAN_STEMS:       # main() reads the full run's names: copy without the suffixes
            for truth in benchmark_run.TRUTHS:
                shutil.copyfile(benchmark_run.report_path(root, "gaps", stem, truth, ids, SEEDS),
                                benchmark_run.report_path(root, "gaps", stem, truth))
        argv = MEAN_STEMS + ["--reference", "nearest_w50", "--profile", "gaps", "--data-dir", root]
        target = os.path.join(root, "choice.json")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = choose.main(argv + ["--objective", "mean", "--out", target])
        lines = out.getvalue().splitlines()
        assert code == 0 and "objective mean" in lines[0], (code, lines[:1])
        assert any(line.startswith("versus    nearest_w50 mean: ") for line in lines), "the mean comparison is missing"
        with open(target, encoding="utf-8") as stream:
            document = json.load(stream)
        expected = chosen(root, MEAN_STEMS, objective="mean")
        assert (document["objective"], document["chosen"]) == ("mean", expected["chosen"]), document["chosen"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = choose.main(argv)
        assert code == 0 and "objective raw" in out.getvalue().splitlines()[0], code
        expect_error(choose.ChooseError, lambda: chosen(root, MEAN_STEMS, objective="median"), "objective", "median")
