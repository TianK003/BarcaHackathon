"""Checks of P40's benchmark side: its config, the trainer's version 5 path, and the feature sanity step.

The features themselves are checked in pipeline/selftest.py (test_shape_features_v5_*), beside version 1 and 2's. Here:
the config `p40_v5` is `p32_w030` but for the shape model's path and the description (config files only: no model file
is read, so the check holds before the version 5 model is trained); the trainer knows version 5's file and refuses
retired versions; and `p40_sanity` prints aggregates only, passes on a healthy synthetic match, and exits 1 on a new
column that is all NaN or constant and on a version 2 column that is not equal (synthetic only).
"""

import copy
import io
import os
import tomllib

import numpy as np

from pipeline.benchmark import p40_sanity
from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark.checks.common import synthetic_calibration, tracked_match
from pipeline.evidence import shape_features

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGS = os.path.join(ROOT, "pipeline", "benchmark", "configs")
MODEL_V5 = "pipeline/models/shape_possession_v5_hgb.json"


def read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def shape_table(config):
    tables = [table for table in config["evidence"] if table["kind"] == "shape_possession"]
    assert len(tables) == 1, "{} shape_possession tables".format(len(tables))
    return tables[0]


def test_p40_config_is_p32_w030_but_for_the_shape_model():
    reference = read(os.path.join(CONFIGS, "p32", "p32_w030.toml"))
    config = read(os.path.join(CONFIGS, "p40", "p40_v5.toml"))
    assert sorted(os.listdir(os.path.join(CONFIGS, "p40"))) == ["p40_v5.toml"], "one candidate, no pick, no split"
    table = shape_table(config)
    assert table["model"] == MODEL_V5 and table["weight"] == 0.3 and table["scope"] == "all", table
    assert shape_table(reference)["model"] == "pipeline/models/shape_possession_v2_hgb.json"
    assert isinstance(config["description"], str) and config["description"] != reference["description"]
    left, right = copy.deepcopy(config), copy.deepcopy(reference)
    for each in (left, right):
        each.pop("description")
        shape_table(each).pop("model")
    assert left == right, "p40_v5 differs from p32_w030 in more than the shape model's path and the description"


def test_p40_trainer_knows_version_5():
    assert trainer.DEFAULT_MODEL_PATHS[5] == MODEL_V5
    assert trainer.DEFAULT_MODEL_PATHS[1] == "pipeline/models/shape_possession_hgb.json"
    assert trainer.DEFAULT_MODEL_PATHS[2] == "pipeline/models/shape_possession_v2_hgb.json"
    assert sorted(trainer.DEFAULT_MODEL_PATHS) == list(shape_features.VERSIONS) == [1, 2, 5]
    parser = trainer.build_parser()
    assert parser.parse_args(["--features", "5"]).features == 5
    for retired in ("3", "4"):
        try:
            with _quiet():
                parser.parse_args(["--features", retired])
        except SystemExit:
            continue
        raise AssertionError("--features {} must be refused (retired)".format(retired))


class _quiet:
    """Swallows what argparse prints to stderr while a refusal is expected."""
    def __enter__(self):
        import contextlib
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def __exit__(self, *exc):
        self._stack.close()
        return False


def synthetic_served():
    """(matches, calibration): the sanity step's inputs on a synthetic match served by the gaps profile."""
    return {"RND001": tracked_match("RND001"), "RND002": tracked_match("RND002", 1)}, synthetic_calibration()


def test_p40_sanity_passes_on_a_healthy_match_and_prints_aggregates_only():
    matches, calibration = synthetic_served()
    lines = []
    ok = p40_sanity.sanity("gaps", 1, matches=matches, calibration=calibration, log=lines.append)
    assert ok, "\n".join(lines)
    text = "\n".join(lines)
    for name in p40_sanity.NEW_COLUMNS:
        assert name in text, name
    assert len(p40_sanity.NEW_COLUMNS) == 9 and text.count("equal in version 5: yes") == 2
    assert len(lines) == 2 * (1 + 1 + 1 + 9), "two matches, each: heading, equality, table header, nine columns"
    assert "FAILED" not in text and "CONSTANT" not in text and "ALL NaN" not in text


def test_p40_sanity_fails_on_a_bad_column_or_an_unequal_v2_column():
    matches, calibration = synthetic_served()
    inputs = trainer.served_inputs(matches["RND001"], "gaps", 0, calibration, None)
    column = shape_features.columns(5)
    real = shape_features.features
    clean = real(inputs, 5)[1]

    def serving(change):
        def features(given, version=1):
            names, X = real(given, version)
            if version == 5:
                X = X.copy()
                change(X)
            return names, X
        return features
    cases = (
        ("ALL NaN", lambda X: X.__setitem__((slice(None), column["a_press_p90"]), np.nan)),
        ("CONSTANT", lambda X: X.__setitem__((slice(None), column["press_gap"]),
                                             np.where(np.isnan(X[:, column["press_gap"]]), np.nan, 1.0))),
        ("CONSTANT", lambda X: X.__setitem__((slice(None), column["h_press_share"]),
                                             np.where(np.isnan(X[:, column["h_press_share"]]), np.nan, 0.0))),
        ("equal in version 5: NO", lambda X: X.__setitem__(
            (slice(None), column["h_adv_mean"]), X[:, column["h_adv_mean"]] + 1.0)),
    )
    try:
        for expected, change in cases:
            shape_features.features = serving(change)
            lines, ok = p40_sanity.check(inputs)
            assert not ok and expected in "\n".join(lines), (expected, lines)
    finally:
        shape_features.features = real
    lines, ok = p40_sanity.check(inputs)
    assert ok, lines
    assert np.isfinite(clean[:, column["press_gap"]]).any(), "the synthetic match must give the columns something"
