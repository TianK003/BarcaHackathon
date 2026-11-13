"""Checks of P37: shape features version 4 in the trainer, the report's gap-class cells (handover, same team, unknown)
and P37's two configs (pipeline/benchmark/configs/p37/), p32_w030 with its shape source split in two.

The feature and scope checks are in pipeline/selftest.py; these are the trainer's. The version 4 model file is trained
after review, so no check reads it: the trainer check is a small real fit on three synthetic matches, and the config
check reads TOML only.
"""

import copy
import json
import os
import tempfile
import tomllib

import numpy as np

from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark.checks.common import random_match, synthetic_calibration
from pipeline.config import load_config
from pipeline.evidence import shape_features, shape_possession, trees
from pipeline.selftest import beside_ball, play_inputs

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIGS = os.path.join(ROOT, "pipeline", "benchmark", "configs")
REFERENCE = os.path.join(CONFIGS, "p32", "p32_w030.toml")
MODEL_V2 = "pipeline/models/shape_possession_v2_hgb.json"
MODEL_V4 = "pipeline/models/shape_possession_v4_hgb.json"
FOLD_IDS = ("RND001", "RND002", "RND003")
ROWS = (1500, 1200)                                # graded frames of a half: past the 10 s window of version 2


def read(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def shape_tables(config):
    return [table for table in config["evidence"] if table["kind"] == "shape_possession"]


def gap_scene():
    """A static ball with a home slot beside it on frames 20 to 39 and 200 to 209, an away slot on 100 to 119, and a
    home one again on 150 to 159 (so: home, away, home, home, with the ball hidden between every two), in two blocks of
    which the second holds no episode of its own. Returns (inputs, the frames the ball is missing on)."""
    n = 400
    players = np.full((n, 2, 2), np.nan)
    for frames in (range(20, 40), range(150, 160), range(200, 210)):
        players[list(frames), 0] = beside_ball(n, frames)[list(frames)]
    players[100:120, 1] = beside_ball(n, range(100, 120), offset=(-0.5, 0.0))[100:120]
    ball = np.tile([30.0, 30.0], (n, 1))
    hidden = np.zeros(n, dtype=bool)
    for start, stop in ((5, 15), (45, 95), (125, 145), (165, 195), (215, 250), (270, 300), (320, 390)):
        hidden[start:stop] = True
    ball[hidden] = np.nan
    inputs = play_inputs(players, ball, teams=np.array([0, 1]), graded=[(1, 0, 260), (2, 260, 400)])
    return inputs, hidden


def test_p37_gap_classes_on_a_synthetic_scene():
    """The last controller before a gap against the next after it, inside a block: HANDOVER, SAME_TEAM or UNKNOWN."""
    inputs, hidden = gap_scene()
    kinds = trainer.gap_classes(inputs)
    assert kinds.dtype == np.int8 and kinds.shape == (400,)
    assert (kinds[~hidden] == 0).all(), "only frames with the ball missing are classified"
    for span, want in (((5, 15), trainer.UNKNOWN),       # before the block's first episode
                       ((45, 95), trainer.HANDOVER),     # home 20-39, then away 100-119
                       ((125, 145), trainer.HANDOVER),   # away, then home 150-159
                       ((165, 195), trainer.SAME_TEAM),  # home 150-159, then home 200-209
                       ((215, 250), trainer.UNKNOWN),    # nothing after it in the block (the second block is another)
                       ((270, 300), trainer.UNKNOWN),    # the second block: nothing of the first one reaches it
                       ((320, 390), trainer.UNKNOWN)):
        assert (kinds[span[0]:span[1]] == want).all(), (span, kinds[span[0]:span[1]])
    # the three codes are the report's cells; a handover is only ever between kept frames of two teams
    assert {trainer.HANDOVER, trainer.SAME_TEAM, trainer.UNKNOWN} == {1, 2, 3}
    home, away = shape_features.control_masks(inputs)
    assert np.flatnonzero(home).tolist() == list(range(20, 40)) + list(range(150, 160)) + list(range(200, 210))
    assert np.flatnonzero(away).tolist() == list(range(100, 120))


def loop_classes(inputs):
    """gap_classes, one frame at a time: look backwards and forwards inside the block for a kept frame."""
    home, away = shape_features.control_masks(inputs)
    present = inputs.atd.ball_present
    out = np.zeros(len(present), dtype=np.int8)
    for _, start, stop in inputs.graded:
        for frame in range(start, stop):
            if present[frame]:
                continue
            before = next((t for t in range(frame, start - 1, -1) if home[t] or away[t]), None)
            after = next((t for t in range(frame, stop) if home[t] or away[t]), None)
            if before is None or after is None:
                out[frame] = trainer.UNKNOWN
            else:
                out[frame] = trainer.SAME_TEAM if bool(home[before]) == bool(home[after]) else trainer.HANDOVER
    return out


def test_p37_gap_classes_equal_a_loop_on_random_scenes():
    from pipeline.selftest import team_scene
    for seed in range(6):
        players, ball, teams, graded = team_scene(seed)
        inputs = play_inputs(players, ball, teams=teams, graded=graded)
        assert np.array_equal(trainer.gap_classes(inputs), loop_classes(inputs)), seed


def matches3():
    return {match_id: random_match(match_id, seed=seed, rows=ROWS) for seed, match_id in enumerate(FOLD_IDS)}


def test_p37_trainer_version_4_and_the_report_cells():
    """`version=4` trains on the 46 columns and the file says so; the report's three gap cells partition the alive gap
    frames of every version, and equal plain loops."""
    matches = matches3()
    calibration = synthetic_calibration()
    assert trainer.DEFAULT_MODEL_PATHS[4] == MODEL_V4 and 3 not in trainer.DEFAULT_MODEL_PATHS
    artifact = trainer.train("gaps", 1, 5, matches=matches, calibration=calibration, version=4)
    assert artifact["feature_version"] == 4 and artifact["features"] == list(shape_features.feature_names(4))
    for name, model in list(artifact["folds"].items()) + [("full", artifact["full"])]:
        trees.check_trees(model, 46, name, "shape_possession")
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "v4.json")
        trainer.write(path, artifact)
        shape_possession.load(path)                  # the source accepts it: names, version and tree width agree
    reports = {4: trainer.report(artifact, "gaps", 1, matches=matches, calibration=calibration)}
    reports[1] = trainer.report(trainer.train("gaps", 1, 5, matches=matches, calibration=calibration), "gaps", 1,
                                matches=matches, calibration=calibration)
    for version, report in reports.items():
        json.dumps(report, allow_nan=False)
        for cell in list(report["per_match"].values()) + [report["pooled"]]:
            assert set(cell) == {"alive", "missing", "touch"}, "today's cells keep their keys"
        gaps = report["gap_classes"]
        assert sorted(gaps) == ["per_match", "pooled"] and sorted(gaps["per_match"]) == list(FOLD_IDS)
        for name, cell in list(gaps["per_match"].items()) + [("pooled", gaps["pooled"])]:
            whole = report["pooled"] if name == "pooled" else report["per_match"][name]
            assert sorted(cell) == ["handover", "same_team", "unknown"]
            for part in cell.values():
                assert set(part) == {"frames", "log_loss", "accuracy"}
            assert sum(part["frames"] for part in cell.values()) == whole["missing"]["frames"] > 0, (version, name)
            total = sum(part["frames"] * part["log_loss"] for part in cell.values() if part["frames"])
            assert abs(total - whole["missing"]["frames"] * whole["missing"]["log_loss"]) <= 1e-9, (version, name)
        today = [line for line in trainer.render_report(report) if not line.startswith("gaps")]
        without = trainer.render_report({key: value for key, value in report.items() if key != "gap_classes"})
        assert today == without, "the existing lines are unchanged; the gap lines are added"
        lines = trainer.render_report(report)
        assert any(line.startswith("gaps") and "handover" in line for line in lines)
        assert all(len(line) <= 120 for line in lines) and all(match_id in "\n".join(lines) for match_id in FOLD_IDS)
    # the classes are the touch rule's alone: the same frames per class for version 1 and version 4
    for name in ("per_match", "pooled"):
        left, right = reports[1]["gap_classes"][name], reports[4]["gap_classes"][name]
        if name == "pooled":
            left, right = {"pooled": left}, {"pooled": right}
        for key in left:
            for kind in ("handover", "same_team", "unknown"):
                assert left[key][kind]["frames"] == right[key][kind]["frames"], (key, kind)
    # against a loop, on the first seed's frames (the source is silent on a few: the cells hold at most the loop's)
    frames = {"handover": 0, "same_team": 0, "unknown": 0}
    from pipeline.benchmark import truth
    for match_id, match in sorted(matches.items()):
        codes, _ = truth.truth_codes(match)
        inputs = trainer.served_inputs(match, "gaps", 0, calibration, None)
        kinds = loop_classes(inputs)
        alive = (codes == trainer.HOME) | (codes == trainer.AWAY)
        for key, code in (("handover", trainer.HANDOVER), ("same_team", trainer.SAME_TEAM),
                          ("unknown", trainer.UNKNOWN)):
            frames[key] += int((alive & (kinds == code)).sum())
    pooled = reports[4]["gap_classes"]["pooled"]
    for key in frames:
        assert pooled[key]["frames"] <= frames[key], key


def test_p37_configs_are_p32_w030_with_its_shape_source_split():
    reference = read(REFERENCE)
    split, gap = read(os.path.join(CONFIGS, "p37", "p37_split.toml")), read(os.path.join(CONFIGS, "p37", "p37_gap030.toml"))
    assert sorted(f for f in os.listdir(os.path.join(CONFIGS, "p37"))) == ["p37_gap030.toml", "p37_split.toml"]
    for config in (split, gap):
        assert isinstance(config["description"], str) and config["description"] != reference["description"]
    assert split["description"] != gap["description"]
    # the split: the one shape entry becomes two, in its place; everything else is the reference's
    (single,) = shape_tables(reference)
    assert single["scope"] == "all" and single["weight"] == 0.3 and single["model"] == MODEL_V2
    where = reference["evidence"].index(single)
    visible = dict(single, scope="visible")
    gaps = dict(single, name="shape_gaps", scope="gaps")
    want = copy.deepcopy(reference)
    want["evidence"][where:where + 1] = [visible, gaps]
    want.pop("description")
    got = copy.deepcopy(split)
    got.pop("description")
    assert got == want, "p37_split differs from p32_w030 in more than the split of the shape source"
    # the candidate: the same, with the gaps source reading the version 4 file
    want = copy.deepcopy(split)
    shape_tables(want)[1]["model"] = MODEL_V4
    want["description"] = gap["description"]
    assert gap == want, "p37_gap030 differs from p37_split in more than the gaps source's model"
    assert [table["scope"] for table in shape_tables(gap)] == ["visible", "gaps"]
    assert [table["model"] for table in shape_tables(gap)] == [MODEL_V2, MODEL_V4]
    assert all(table["weight"] == 0.3 for table in shape_tables(gap))
    # p37_split loads as the runner loads a config (every parameter checked against its component), with two sources
    config = load_config(os.path.join(CONFIGS, "p37", "p37_split.toml"))
    assert config.stem == "p37_split"
    sources = [part for part in config.evidence if part.kind == "shape_possession"]
    assert [part.name for part in sources] == ["shape_possession", "shape_gaps"]
    assert [part.params["scope"] for part in sources] == ["visible", "gaps"]
    assert len({part.name for part in config.evidence}) == len(config.evidence)
