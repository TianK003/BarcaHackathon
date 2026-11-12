"""Checks of P33 (shape features version 3): the trainer's version 3 and P33's two configs
(pipeline/benchmark/configs/p33/), p30_w030 but for the shape model and its weight.

The trainer check is a small real fit on three synthetic matches; the config check reads TOML only, so neither needs
the version 3 model file, which is trained after review.
"""

import copy
import glob
import os
import tempfile
import tomllib

from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark.checks.common import random_match, synthetic_calibration
from pipeline.evidence import shape_features, shape_possession, trees

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGS = os.path.join(ROOT, "pipeline", "benchmark", "configs")
MODEL_V3 = "pipeline/models/shape_possession_v3_hgb.json"
WEIGHTS = {"p33_w030": 0.3, "p33_w045": 0.45}
FOLD_IDS = ("RND001", "RND002", "RND003")
ROWS = (2500, 2000)                                # graded frames of a half: past the 20 s window, so no column is all NaN


def read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def shape_table(config):
    tables = [table for table in config["evidence"] if table["kind"] == "shape_possession"]
    assert len(tables) == 1, "{} shape_possession tables".format(len(tables))
    return tables[0]


def test_p33_trainer_version_3():
    """`version=3` trains on the 60 columns and the file says so; `shape_possession.load` accepts it once written."""
    matches = {match_id: random_match(match_id, seed=seed, rows=ROWS) for seed, match_id in enumerate(FOLD_IDS)}
    names = shape_features.feature_names(3)
    assert len(names) == 60 and trainer.DEFAULT_MODEL_PATHS[3] == MODEL_V3
    artifact = trainer.train("gaps", 1, 5, matches=matches, calibration=synthetic_calibration(), version=3)
    assert artifact["feature_version"] == 3 and artifact["features"] == list(names)
    assert sorted(artifact["folds"]) == list(FOLD_IDS) and "full" in artifact
    for name, model in list(artifact["folds"].items()) + [("full", artifact["full"])]:
        trees.check_trees(model, 60, name, "shape_possession")
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "v3.json")
        trainer.write(path, artifact)
        shape_possession.load(path)                  # the source accepts it: names, version and tree width agree


def test_p33_configs_are_p30_w030_but_for_the_shape_model():
    reference = read(os.path.join(CONFIGS, "p30", "p30_w030.toml"))
    paths = sorted(glob.glob(os.path.join(CONFIGS, "p33", "*.toml")))
    assert [os.path.splitext(os.path.basename(path))[0] for path in paths] == sorted(WEIGHTS), paths
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        config = read(path)
        table = shape_table(config)
        assert table["model"] == MODEL_V3 and table["weight"] == WEIGHTS[name], name
        assert isinstance(config["description"], str) and config["description"] != reference["description"], name
        left, right = copy.deepcopy(config), copy.deepcopy(reference)
        for each in (left, right):
            each.pop("description")
            for key in ("model", "weight"):
                shape_table(each).pop(key)
        assert left == right, "{} differs from p30_w030 in more than the shape model and its weight".format(name)
