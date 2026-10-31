"""P19's label-free transfer check on BAR-ATH: its agreement with Smart Tagging's team votes, against the rule with its hold.

    python pipeline/benchmark/team_transfer.py [--model pipeline/models/team_model_hgb.json]
                                               [--config pipeline/configs/main.toml] [--data-dir D]   # HEAVY: loads BAR-ATH

Why. The benchmark scores P19 on DFL's matches, degraded to look like ours; nothing there says that the model reads our
match the way it reads those (the ball-dynamics columns are learnt on DFL's ball, research.md). Our match has no truth
and no human-produced data, but it has an independent pseudo-label: Smart Tagging's automatic phase intervals, which
name a team on some frames (`smart_tagging.votes`, 1 home, 0 away). They are in neither P19 nor the rule, so agreeing
with them is evidence that does not come from the model's own inputs. It is not accuracy: the tags are phases and not
possessions (T17); it is a comparison, the same frames and the same votes for P19 and for the rule it would sit beside.

The declared test (docs/results.md R021 part 1). On the frames that are graded, have the served ball visible and carry a
vote of 0 or 1, P19's `full` model says H at p >= 0.5 and the rule (`controlled_touch` with the parameters of the
config's own entry, hold 0.60) says H at its value >= 0.5. A frame on which a source is silent (NaN) is counted wrong
for it. P19 passes iff 100 (p19_right - rule_right) >= -frames: its agreement is at least the rule's minus 0.01, an
integer test with no rounding. The output also holds the NaN share of every feature on the graded, ball-visible frames,
an aggregate that catches a column that is broken on our match and fine on DFL's.

Nothing per frame is printed or written: counts, shares and the verdict. The label-free checks that go with this (the
same sha256 twice, the `D` share, no `X`, the possessions per half) are the pipeline's own and run on the adopted config,
not here. The command exits 0 either way: the verdict is in its output and in `<data>/runs/r021/transfer.json`.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import run  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, kinematics, smart_tagging, team_features, team_model  # noqa: E402

# Params
DEFAULT_MODEL = os.path.join(ROOT, "pipeline", "models", "team_model_hgb.json")
DEFAULT_CONFIG = os.path.join(ROOT, "pipeline", "configs", "main.toml")
HOME_LEVEL = 0.5                # a home probability at or above this names home
MARGIN = 0.01                   # P19's agreement may be this far below the rule's
FOLDER = ("runs", "r021")       # <data dir>/runs/r021/transfer.json


def agreement(home, vote, frames):
    """{"frames", "right", "silent", "agreement"}: how often `home` names the team the votes name, on `frames`.

    `home` is a probability per frame (NaN = silent), `vote` the team per frame (1 home, 0 away) and `frames` the bool
    mask of the frames judged. A silent frame is wrong. `agreement` is None when there is no frame.
    """
    home, vote, frames = np.asarray(home, dtype=np.float64), np.asarray(vote), np.asarray(frames, dtype=bool)
    if not (home.shape == vote.shape == frames.shape) or home.ndim != 1:
        raise ValueError("team_transfer: home, vote and frames must be 1-D arrays of one length, got {}, {} and {}"
                         .format(home.shape, vote.shape, frames.shape))
    silent = np.isnan(home) & frames
    with np.errstate(invalid="ignore"):
        right = frames & ~np.isnan(home) & ((home >= HOME_LEVEL) == (vote == 1))
    total, correct = int(frames.sum()), int(right.sum())
    return {"frames": total, "right": correct, "silent": int(silent.sum()),
            "agreement": correct / total if total else None}


def passes(p19, rule, margin=MARGIN):
    """True iff 100 (p19 right - rule right) >= -frames, on the same frames: P19's agreement is at least the rule's minus
    `margin` (0.01), worked in integers. ValueError for two cells of different frames, or a margin that is not 1/n."""
    if p19["frames"] != rule["frames"]:
        raise ValueError("team_transfer: P19 and the rule must be judged on the same frames, got {} and {}".format(
            p19["frames"], rule["frames"]))
    if isinstance(margin, bool) or not margin > 0:
        raise ValueError("team_transfer: margin must be 1 / a whole number (0.01 is 1 / 100), got {!r}".format(margin))
    denominator = round(1.0 / margin)
    if abs(denominator * margin - 1.0) > 1e-9:
        raise ValueError("team_transfer: margin must be 1 / a whole number (0.01 is 1 / 100), got {!r}".format(margin))
    return denominator * (p19["right"] - rule["right"]) >= -p19["frames"]


def measure(inputs, model=DEFAULT_MODEL, config_path=DEFAULT_CONFIG):
    """The transfer check on `inputs`: {frames, p19, rule, passes, margin, nan_shares}, aggregates only.

    `model` is the artifact (the fold of `inputs.game` if it has one, else `full`, as the source chooses), `config_path`
    the config whose `controlled_touch` entry is the rule. BenchmarkError for a config without exactly one.
    """
    config = load_config(config_path)
    touch = [part for part in config.evidence if part.kind == "controlled_touch"]
    if len(touch) != 1:
        raise run.BenchmarkError("team_transfer: {} must have exactly one controlled_touch entry (the rule P19 is "
                                 "compared with), it has {}".format(config.stem, len(touch)))
    run_context = RunContext(inputs=inputs, seed=0)
    home = team_model.compute(run_context, model=model).home
    rule = controlled_touch.compute(run_context, **touch[0].params).home
    votes = smart_tagging.votes(inputs)
    visible = kinematics.ball_track(inputs).visible
    frames = visible & ((votes == 0) | (votes == 1))
    p19_cell, rule_cell = agreement(home, votes, frames), agreement(rule, votes, frames)
    _, X = team_features.features(inputs)
    shares = np.isnan(X[visible]).mean(axis=0) if visible.any() else np.full(X.shape[1], 1.0)
    return {"frames": p19_cell["frames"], "visible_frames": int(visible.sum()), "p19": p19_cell, "rule": rule_cell,
            "gain_frames": p19_cell["right"] - rule_cell["right"], "margin": MARGIN,
            "passes": passes(p19_cell, rule_cell),
            "nan_shares": {name: float(share) for name, share in zip(team_features.FEATURE_NAMES, shares)}}


def render(result):
    """The printed lines: counts, agreements and the verdict, the features that are mostly NaN. No frame."""
    def number(value):
        return "n/a" if value is None else "{:.4f}".format(value)

    lines = ["transfer  {:,} graded frames with the ball visible and a team vote (of {:,} visible)".format(
        result["frames"], result["visible_frames"])]
    for name in ("p19", "rule"):
        cell = result[name]
        lines.append("transfer  {:<5} right {:>8,}  silent {:>8,}  agreement {}".format(
            name, cell["right"], cell["silent"], number(cell["agreement"])))
    lines.append("transfer  P19 minus the rule: {:+,} frames; it needs 100 x that >= -{:,}".format(
        result["gain_frames"], result["frames"]))
    lines.append("transfer  {}".format("PASSES: P19's agreement is at least the rule's minus 0.01" if result["passes"]
                                       else "FAILS: P19 agrees with the votes less than the rule does, by more than "
                                            "0.01"))
    heavy = sorted(((share, name) for name, share in result["nan_shares"].items() if share >= 0.5), reverse=True)
    lines.append("transfer  features at least half NaN on those frames: {}".format(
        ", ".join("{} {:.0%}".format(name, share) for share, name in heavy) or "none"))
    return lines


def load_inputs(data_dir):
    from pipeline.inputs import Inputs
    return Inputs(data_dir=data_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P19's agreement with Smart Tagging's team votes on BAR-ATH, against the rule with its hold "
                    "(HEAVY: loads the real match).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="the artifact (default pipeline/models/team_model_hgb.json)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="the config whose controlled_touch is the rule (default pipeline/configs/main.toml)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(args.data_dir)
        result = measure(inputs, args.model, args.config)
        for line in render(result):
            print(line)
        where = os.path.join(args.data_dir or inputs.data_dir, *FOLDER, "transfer.json")
        run.write_json(where, result)
        print("wrote     {}".format(where))
    except (run.BenchmarkError, ValueError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
