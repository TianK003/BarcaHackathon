"""Checks of the bar of P29 to P31 as a script (pipeline/benchmark/p29_bar.py) and of P29's four configs.

Everything is synthetic or a config file: the bar's thresholds on the unrounded numbers, the tie-break of the pick, the
three commands on made-up reports (as checks/p27.py builds them), and the four configs, which are parsed with tomllib
and compared with p27_final.toml, not loaded, because hold_curve and hold_scale belong to the touch component of
another lane. Filled by Task 3 of docs/superpowers/plans/2026-11-12-p29-p31.md.
"""

import contextlib
import io
import json
import os
import tempfile
import tomllib

from pipeline.benchmark import p29_bar, score, sources
from pipeline.benchmark import run as benchmark_run
from pipeline.selftest import write_text

PROFILE = "generic"
MADE_UP_CONFIG = "# a made-up config: the reports of checks/p29.py are of no run\n"
UNIT = 1_000_000                    # the possessions of each made-up match: a score is a count of millionths
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
SCALES = {"p29_s025": 0.25, "p29_s050": 0.5, "p29_s075": 0.75, "p29_s100": 1.0}


def reported(root, scores):
    """Write the raw and the coarse report of each stem of `scores`, {stem: (raw, coarse)} in millionths of spellF1
    that every match has, under the data folder `root`. Each match is one half of UNIT true and UNIT predicted
    possessions, that many matched at every IoU, with a perfect confusion so that macroF1 says nothing."""
    config = write_text(root, "made_up.toml", MADE_UP_CONFIG)
    for stem, pair in scores.items():
        for truth, matched in zip(benchmark_run.TRUTHS, pair):
            halves = [{"match_id": match_id, "half": 1, "seed": 0, "n_gt": UNIT, "n_pred": UNIT,
                       "tp": [matched] * len(score.IOUS), "confusion": [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0]]}
                      for match_id in sources.MATCH_IDS]
            report = {"config": {"path": config.replace(os.sep, "/"), "text": MADE_UP_CONFIG},
                      "profile": {"name": PROFILE, "version": benchmark_run.PROFILES[PROFILE]}, "seeds": [0, 1, 2],
                      "matches": [{"id": match_id} for match_id in sources.MATCH_IDS], "calibration": None,
                      "halves": halves, "pooled": {"alive_dead_f1": 0.97},
                      "possessions_per_graded_minute": {"predicted": 3.0}, "labels": {"predicted": {"D": 0.45}}}
            path = benchmark_run.report_path(root, PROFILE, stem, truth)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle)


def main_of(root, *argv):
    """(exit code, printed lines) of p29_bar.main(argv) on the reports under `root`."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = p29_bar.main(list(argv) + ["--data-dir", root])
    return code, out.getvalue().splitlines()


def test_p29_bar_thresholds():
    versus = {"mean": {"pooled_delta": 0.0050, "p_sign_flip": 0.05},
              "raw": {"pooled_delta": -0.0050}, "coarse": {"pooled_delta": 0.0150}}
    assert p29_bar.passes_bar(versus) and p29_bar.passes_guard(versus)
    for truth, key, value in (("mean", "pooled_delta", 0.0049), ("mean", "p_sign_flip", 0.051),
                              ("raw", "pooled_delta", -0.0051), ("coarse", "pooled_delta", -0.0051)):
        worse = {name: dict(cell) for name, cell in versus.items()}
        worse[truth][key] = value
        assert not p29_bar.passes_bar(worse)
    assert p29_bar.passes_guard({"raw": {"pooled_delta": -0.005}, "coarse": {"pooled_delta": 0.0}})
    assert not p29_bar.passes_guard({"raw": {"pooled_delta": 0.02}, "coarse": {"pooled_delta": -0.0051}})


def test_p29_pick_takes_the_first_in_order_within_the_band():
    order = ["a", "b", "c"]
    assert p29_bar.pick_from({"a": 0.5700, "b": 0.5740, "c": 0.5712}, order) == "a"
    assert p29_bar.pick_from({"a": 0.5600, "b": 0.5740, "c": 0.5712}, order) == "b"
    assert p29_bar.pick_from({"a": 0.5600, "b": 0.5700, "c": 0.5740}, ["c", "b", "a"]) == "c"


def test_p29_configs_are_p27_final_but_for_the_hold():
    with open(os.path.join(CONFIGS, "p27", "p27_final.toml"), "rb") as handle:
        final = tomllib.load(handle)
    final.pop("description")
    touch = next(entry for entry in final["evidence"] if entry["kind"] == "controlled_touch")
    assert touch.pop("hold") == 0.6
    folder = os.path.join(CONFIGS, "p29")
    names = sorted(name for name in os.listdir(folder) if name.endswith(".toml"))
    assert names == sorted(stem + ".toml" for stem in SCALES), names
    for stem, scale in SCALES.items():
        with open(os.path.join(folder, stem + ".toml"), "rb") as handle:
            config = tomllib.load(handle)
        assert config.pop("description").startswith("P29: "), stem
        own = next(entry for entry in config["evidence"] if entry["kind"] == "controlled_touch")
        assert (own.pop("hold_curve"), own.pop("hold_scale")) == ("pipeline/models/hold_curve.json", scale), stem
        assert "hold" not in own and config == final, stem


def test_p29_bar_commands_on_made_up_reports():
    """pick, bar and guard through main(): the reference at 0.5000 under both truths; p29_a 0.5100 and p29_b 0.5140
    (a is within the band of the best, and first in order), p29_flat +0.002 (under the bar), p29_edge +0.007 on average
    with the coarse truth 0.006 below the reference (clears the gain, fails the guard)."""
    with tempfile.TemporaryDirectory() as root:
        reported(root, {"ref": (500000, 500000), "p29_a": (510000, 510000), "p29_b": (514000, 514000),
                        "p29_c": (500000, 500000), "p29_flat": (502000, 502000), "p29_edge": (520000, 494000)})
        code, lines = main_of(root, "pick", "--reference", "ref", "p29_a", "p29_b", "p29_c")
        assert code == 0 and lines[0].split() == ["picked", "p29_a"], lines
        code, lines = main_of(root, "pick", "--reference", "ref", "p29_b", "p29_a", "p29_c")
        assert code == 0 and lines[0].split() == ["picked", "p29_b"], lines
        assert main_of(root, "bar", "--reference", "ref", "p29_a")[0] == 0
        assert main_of(root, "bar", "--reference", "ref", "p29_flat")[0] == 2
        assert main_of(root, "bar", "--reference", "ref", "p29_edge")[0] == 2
        assert main_of(root, "guard", "--reference", "ref", "p29_a")[0] == 0
        assert main_of(root, "guard", "--reference", "ref", "p29_flat")[0] == 0
        assert main_of(root, "guard", "--reference", "ref", "p29_edge")[0] == 2
        code, lines = main_of(root, "bar", "--reference", "ref", "p29_missing")
        assert code == 1 and lines[0].startswith("FAILED"), lines
