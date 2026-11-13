"""Check of the report's coarse-truth cells of train_shape.py (kept from P34, which was dropped: R036).

Synthetic matches only; no model file is read. P34's weighted-label training path was removed with the lane.
"""

import json
import math
import os
import tempfile

from pipeline.benchmark import train_shape as trainer
from pipeline.benchmark import truth
from pipeline.benchmark.checks.common import A, H, random_match
from pipeline.contracts import RunContext
from pipeline.evidence import shape_possession

# Params
IDS = ("RND001", "RND002", "RND003")


def matches3():
    return {match_id: random_match(match_id, seed=seed, rows=(300 + 20 * seed, 260)) for seed, match_id
            in enumerate(IDS)}


def test_p34_report_has_the_coarse_cells():
    matches = matches3()
    artifact = trainer.train("clean", 1, 9, matches=matches)
    report = trainer.report(artifact, "clean", 1, matches=matches)
    json.dumps(report, allow_nan=False)
    # today's cells keep their keys
    for cell in list(report["per_match"].values()) + [report["pooled"]]:
        assert set(cell) == {"alive", "missing", "touch"}
        assert set(cell["alive"]) == {"frames", "silent", "log_loss", "accuracy"}
        assert set(cell["missing"]) == {"frames", "log_loss", "accuracy"}
    coarse = report["coarse"]
    assert sorted(coarse) == ["per_match", "pooled"] and sorted(coarse["per_match"]) == list(IDS)
    for name, cell in list(coarse["per_match"].items()) + [("pooled", coarse["pooled"])]:
        raw_cell = report["pooled"] if name == "pooled" else report["per_match"][name]
        assert set(cell) == {"alive", "missing"} and set(cell["alive"]) == {"frames", "log_loss", "accuracy"}
        assert set(cell["missing"]) == {"frames", "log_loss", "accuracy"}
        for part in ("alive", "missing"):
            assert cell[part]["frames"] == raw_cell[part]["frames"] > 0, (name, part)    # the same frames
    # the numbers, against plain loops over the frames, for the raw and the coarse truth
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "shape.json")
        trainer.write(path, artifact)
        for match_id, match in sorted(matches.items()):
            codes, _ = truth.truth_codes(match)
            inputs = trainer.served_inputs(match, "clean", 0, None, None)
            coarse_codes = truth.coarse_codes(codes, inputs.graded, inputs.fps)
            home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=path, scope="all").home
            sums = {}
            for label, truth_codes in (("raw", codes), ("coarse", coarse_codes)):
                for part in ("alive", "missing"):
                    sums[label, part] = [0, 0.0, 0]
                for frame in range(len(codes)):
                    if truth_codes[frame] not in (H, A) or math.isnan(home[frame]):
                        continue
                    p, truly_home = float(home[frame]), bool(truth_codes[frame] == H)
                    for part, take in (("alive", True), ("missing", not inputs.atd.ball_present[frame])):
                        if take:
                            cell = sums[label, part]
                            cell[0] += 1
                            cell[1] += -math.log(p if truly_home else 1.0 - p)
                            cell[2] += (p >= 0.5) == truly_home
            for (label, part), (frames, loss, right) in sums.items():
                cell = (report["per_match"][match_id] if label == "raw" else coarse["per_match"][match_id])[part]
                assert cell["frames"] == frames and frames > 0, (match_id, label, part)
                assert abs(cell["log_loss"] - loss / frames) <= 1e-9, (match_id, label, part)
                assert cell["accuracy"] == right / frames, (match_id, label, part)
    # the rendering keeps today's lines byte for byte and adds the coarse ones after them
    today = trainer.render_report({key: value for key, value in report.items() if key != "coarse"})
    lines = trainer.render_report(report)
    assert lines[:len(today)] == today and len(lines) > len(today)
    assert all(len(line) <= 120 for line in lines)
    assert all(line.startswith("coarse") for line in lines[len(today):])
    assert all(match_id in "\n".join(lines[len(today):]) for match_id in IDS)
