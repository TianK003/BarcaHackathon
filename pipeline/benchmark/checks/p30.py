"""Checks of P30's three configs (pipeline/benchmark/configs/p30/): p27_final but for the shape model and its weight.

Config files only: no model file is read, so the checks hold before the version 2 model is trained.
"""

import copy
import glob
import os
import tomllib

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGS = os.path.join(ROOT, "pipeline", "benchmark", "configs")
MODEL_V2 = "pipeline/models/shape_possession_v2_hgb.json"
WEIGHTS = {"p30_w030": 0.3, "p30_w020": 0.2, "p30_w045": 0.45}


def read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def shape_table(config):
    tables = [table for table in config["evidence"] if table["kind"] == "shape_possession"]
    assert len(tables) == 1, "{} shape_possession tables".format(len(tables))
    return tables[0]


def test_p30_configs_are_p27_final_but_for_the_shape_model():
    reference = read(os.path.join(CONFIGS, "p27", "p27_final.toml"))
    paths = sorted(glob.glob(os.path.join(CONFIGS, "p30", "*.toml")))
    assert [os.path.splitext(os.path.basename(path))[0] for path in paths] == sorted(WEIGHTS), paths
    assert shape_table(reference)["model"] == "pipeline/models/shape_possession_hgb.json"
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        config = read(path)
        table = shape_table(config)
        assert table["model"] == MODEL_V2 and table["weight"] == WEIGHTS[name], name
        assert isinstance(config["description"], str) and config["description"] != reference["description"], name
        left, right = copy.deepcopy(config), copy.deepcopy(reference)
        for each in (left, right):
            each.pop("description")
            for key in ("model", "weight"):
                shape_table(each).pop(key)
        assert left == right, "{} differs from p27_final in more than the shape model and its weight".format(name)
