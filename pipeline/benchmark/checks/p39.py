"""Checks of P39's config (pipeline/benchmark/configs/p39/): p32_w030 with the shape source silent where the ball is seen.

Config files only. Written by the controller (docs/superpowers/plans/2026-11-13-p35-p39.md, P39).
"""

import os
import tomllib

from pipeline.benchmark.checks.common import read
from pipeline.config import load_config

# Params
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
SEEN_OFF = os.path.join(CONFIGS, "p39", "p39_seen_off.toml")
REFERENCE = os.path.join(CONFIGS, "p32", "p32_w030.toml")


def test_p39_config_is_p32_w030_but_for_the_shape_scope():
    """p39_seen_off.toml is p32_w030.toml but for its description and the shape source's scope, "gaps" against "all",
    the only config in its folder, and it loads as the runner loads a config."""
    seen_off, reference = tomllib.loads(read(SEEN_OFF)), tomllib.loads(read(REFERENCE))
    assert seen_off.pop("description") != reference.pop("description")
    own = next(entry for entry in seen_off["evidence"] if entry["kind"] == "shape_possession")
    theirs = next(entry for entry in reference["evidence"] if entry["kind"] == "shape_possession")
    assert (own.pop("scope"), theirs.pop("scope")) == ("gaps", "all"), (own, theirs)
    assert seen_off == reference, (seen_off, reference)
    assert sorted(os.listdir(os.path.dirname(SEEN_OFF))) == ["p39_seen_off.toml"]
    config = load_config(SEEN_OFF)
    assert config.stem == "p39_seen_off" and len(config.evidence) == 4, config
