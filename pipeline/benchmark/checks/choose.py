"""Checks of choose (pipeline/benchmark/choose.py): a grid's declared rule applied to the runner's written reports.

choose decides which config a grid adopts, so what is held against it here is the rule worked out by hand from the
runner's own results (the band, the coarse tie-break, the order given for a tie that is left), its deltas equal to the
runner's comparison, the leave-one-match-out picks and deltas recomputed in a plain loop, and its refusals: a report
that is missing, one whose config has changed since it was written, and reports of different runs. The matches are the
synthetic ones of checks/common.py, under the gaps profile and the synthetic Calibration; their scores mean nothing.
"""

import json
import os
import shutil
import tempfile

from pipeline.benchmark import choose, score
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (synthetic_calibration, synthetic_camera_fit, synthetic_matches,
                                              yardstick_path)
from pipeline.selftest import expect_error, quiet

# Params
WIDTHS = (50, 25, 0, 125)          # the reference first, as a grid's process lists main.toml first
SEEDS = 2


def written(root, paths, seeds=SEEDS):
    """evaluate_truths on the gaps profile, both truths, with the reports written under `root`; the results."""
    results = benchmark_run.evaluate_truths(paths, profile="gaps", matches=synthetic_matches(), seeds=seeds,
                                            calibration=synthetic_calibration(), camera_fit=synthetic_camera_fit(),
                                            data_dir=root, log=quiet)
    for result in results.values():
        benchmark_run.write_reports(result, data_dir=root)
    return results


def chosen(root, stems, seeds=SEEDS, band=choose.TIE_BAND, metric="spell"):
    return choose.choose(stems, "nearest_w50", profile="gaps", band=band, data_dir=root,
                         match_ids=sorted(synthetic_matches()), seeds=seeds, metric=metric)


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


def test_choose_refuses_reports_of_two_cameras():
    """Reports whose simulated camera differs (P09: another fit, or none at all) are of different runs, and refused;
    reports from before P09, which have no `camera` key, compare as None and are still read together."""
    assert "camera" in choose.SAME
    with tempfile.TemporaryDirectory() as root:
        written(root, [yardstick_path(50), yardstick_path(25)])
        assert chosen(root, ["nearest_w25"])["chosen"] == "nearest_w25", "one run, one camera: chosen"
        path = benchmark_run.report_path(root, "gaps", "nearest_w25", "coarse", sorted(synthetic_matches()), SEEDS)
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        report["camera"]["sha256"] = "0" * 64                       # as if simulated with another camera fit
        benchmark_run.write_json(path, report)
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w25"]), "nearest_w25", "camera",
                     "different runs")
        del report["camera"]                                         # a report from before P09 beside one after it
        benchmark_run.write_json(path, report)
        expect_error(choose.ChooseError, lambda: chosen(root, ["nearest_w25"]), "nearest_w25", "camera")
        for stem in ("nearest_w50", "nearest_w25"):                  # every report from before P09: None == None
            for truth in benchmark_run.TRUTHS:
                path = benchmark_run.report_path(root, "gaps", stem, truth, sorted(synthetic_matches()), SEEDS)
                with open(path, encoding="utf-8") as stream:
                    report = json.load(stream)
                report.pop("camera", None)
                benchmark_run.write_json(path, report)
        assert chosen(root, ["nearest_w25"])["chosen"] == "nearest_w25", "reports without the key are one run"
