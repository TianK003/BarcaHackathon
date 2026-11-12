"""Checks of P31's config (pipeline/benchmark/configs/p31/): p27_final but for the decoder's team_neutral_dead flag.

Config files only. Filled by Task 6 of docs/superpowers/plans/2026-11-12-p29-p31.md.
"""

import os
import tomllib

from pipeline.benchmark.checks.common import read
from pipeline.config import load_config

# Params
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
NEUTRAL = os.path.join(CONFIGS, "p31", "p31_neutral.toml")
FINAL = os.path.join(CONFIGS, "p27", "p27_final.toml")


def test_p31_config_is_p27_final_but_for_the_flag():
    """p31_neutral.toml is p27_final.toml but for its description and team_neutral_dead = true in the decoder's table,
    and it loads as the runner loads a config: every parameter checked against its component's signature."""
    neutral, final = tomllib.loads(read(NEUTRAL)), tomllib.loads(read(FINAL))
    assert neutral.pop("description") != final.pop("description")
    assert neutral["decoder"].pop("team_neutral_dead") is True, neutral["decoder"]
    assert "team_neutral_dead" not in final["decoder"], final["decoder"]
    assert neutral == final, (neutral, final)
    config = load_config(NEUTRAL)
    assert config.stem == "p31_neutral" and config.decoder.kind == "hmm", config
    assert config.decoder.params["team_neutral_dead"] is True and config.decoder.params["switch_team"] == 6, config
    assert "team_neutral_dead" not in load_config(FINAL).decoder.params
