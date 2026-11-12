"""hold_stats: how often the last controlling team still has the ball, by time since its touch and ball hidden since.

    python pipeline/benchmark/hold_stats.py [--profile P] [--seeds N] [--data-dir D] [--write | --verify] [--out FILE]

    result = measure(profile="generic", seeds=3)       # {"pooled": {"no_gap": [[frames, agree] x 10], "gap": ...}, ...}

controlled_touch's hold (pipeline/evidence/controlled_touch.py) gives the stretch between control episodes a constant
weak vote for the last controlling team. P29 asks whether that vote should be a curve instead: the last team keeps the
ball less often the longer ago its last touch was, and less often again if the ball was hidden in between. This module
measures it: on the degraded tracking of every benchmark match and seed, over the hold frames (ball visible, outside
every kept episode, after the block's first) whose truth is H or A, how many frames there are and on how many the
truth's team is the last controller's, per bin of the time since that touch (controlled_touch.HOLD_BINS_S) and per
"gap" (the ball was hidden on some frame since) or "no_gap".

The gate (P29's step 0): the curve is worth building only if the agreement moves across the twenty cells. `deviation` is
the frame-weighted mean absolute difference between each cell's agreement and the pooled one; the gate passes when it
is at least GATE_DEVIATION. A flat hold (every cell near the pooled rate) is what the constant already says.

`--write` measures and writes the curve file controlled_touch's `hold_curve` reads (`curves`; by default
pipeline/models/hold_curve.json): per cell, (agree + 1) / (frames + 2), a Laplace-smoothed share that is 0.5 where
nothing was seen, once from all the matches (`full`) and once per match from the others (`folds`, used on that match so
that no number is scored in sample). `--verify` measures again and compares with the file, exit 0 or 1. The gate is
printed either way and does not change their exit code. An infinite `max_rel_speed` is written as "inf" in `touch`.

No truth reaches the source. hold_state is called on a MatchInputs (inputs.py), which holds the degraded tracking and no
flag; the truth is read here, from the match, only to count, and the result holds counts and no frame.
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.train_dead_ball import dumps, write  # noqa: E402  (the JSON layout train_shape.write uses)
from pipeline.config import ConfigError  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
# main.toml's control rule (R030); a check holds it against the config. Read at call time, never bound as a default.
TOUCH = {"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "duel_radius": 1.0}
GATE_DEVIATION = 0.05           # the frame-weighted mean |cell agreement - pooled agreement| the curve must reach
DEFAULT_PROFILE = run.DEFAULT_PROFILE
DEFAULT_CURVE_PATH = os.path.join(ROOT, "pipeline", "models", "hold_curve.json")
ROWS = ("no_gap", "gap")        # the ball was never hidden since the last touch, or was hidden on some frame


def measure(*, touch=None, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None, data_dir=None,
            log=print):
    """Frames and agreements of the hold frames by gap and bin, per match and pooled over matches and seeds.

    `touch` is the control rule, {radius, max_rel_speed, min_frames, duel_radius} (None: TOUCH, read now). The others
    are those of run.evaluate_configs: `profile`, `seeds` (0 up to seeds - 1; the clean profile has just seed 0),
    `match_ids`, `matches` ({id: DflMatch} to use instead of loading) and `data_dir`; `log` receives a line per match.

    Returns {"profile", "seeds", "matches", "bins_s", "touch", "pooled", "per_match"}; pooled and each per_match[id]
    are {"no_gap": [[frames, agree] x 10], "gap": [[frames, agree] x 10]} over the hold frames whose truth is H or A,
    `agree` counting those where the truth's team is the last controller's. ValueError (hold_stats) for a `touch` that
    misses, adds or misspells a key or holds a value controlled_touch refuses; the runner's BenchmarkError for a
    profile, seeds or match it refuses. Both come before any match is loaded.
    """
    touch = _checked_touch(TOUCH if touch is None else touch)
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    calibration = calibration_module.load(data_dir=data_dir) if run.needs_calibration(profile) else None
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    n_bins = len(controlled_touch.HOLD_BINS_S)

    per_match = {}
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)                  # read here, to count; never handed to the source
        alive_home = codes == pm.H
        alive = alive_home | (codes == pm.A)
        counts = np.zeros((len(ROWS), n_bins, 2), dtype=np.int64)
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            state = controlled_touch.hold_state(RunContext(inputs=inputs, seed=0), **touch)
            held = (state.elapsed >= 0) & alive
            bins = controlled_touch.hold_bin(state.elapsed, match.fps)
            agree = state.last_is_home == alive_home
            for row, in_row in enumerate((~state.gap, state.gap)):
                cell = held & in_row
                counts[row, :, 0] += np.bincount(bins[cell], minlength=n_bins)
                counts[row, :, 1] += np.bincount(bins[cell & agree], minlength=n_bins)
            del atd, inputs, state
        per_match[match_id] = counts
        log("match     {}   {} seeds".format(match_id, len(seed_list)))
        del match, codes

    pooled = np.sum(list(per_match.values()), axis=0)
    return {"profile": profile, "seeds": seed_list, "matches": sorted(ids),
            "bins_s": list(controlled_touch.HOLD_BINS_S), "touch": touch, "pooled": _cells(pooled),
            "per_match": {match_id: _cells(per_match[match_id]) for match_id in sorted(per_match)}}


def _checked_touch(touch):
    """`touch` as a dict copy, or ValueError unless it has exactly the keys of TOUCH and values controlled_touch takes."""
    if not isinstance(touch, dict) or set(touch) != set(TOUCH):
        given = sorted(touch) if isinstance(touch, dict) else touch
        raise ValueError("hold_stats: touch must be a dict with exactly the keys {}, got {!r}".format(
            ", ".join(sorted(TOUCH)), given))
    try:
        controlled_touch._checked(**touch)
    except ValueError as error:
        raise ValueError("hold_stats: touch is refused by the source: {}".format(error)) from error
    return dict(touch)


def _cells(counts):
    return {name: counts[row].tolist() for row, name in enumerate(ROWS)}


def gate(result):
    """{"frames", "agreement", "deviation", "passes"} over the twenty pooled cells of `result`.

    agreement is all agree over all frames; deviation is the sum over the cells of frames_c * |agree_c / frames_c -
    agreement|, over all frames; passes is deviation >= GATE_DEVIATION (to 1e-12). With no frame at all, agreement and
    deviation are None and passes is False.
    """
    cells = [cell for name in ROWS for cell in result["pooled"][name]]
    frames = sum(cell[0] for cell in cells)
    if not frames:
        return {"frames": 0, "agreement": None, "deviation": None, "passes": False}
    agreement = sum(cell[1] for cell in cells) / frames
    deviation = sum(cell[0] * abs(cell[1] / cell[0] - agreement) for cell in cells if cell[0]) / frames
    return {"frames": frames, "agreement": agreement, "deviation": deviation,
            "passes": bool(deviation >= GATE_DEVIATION - 1e-12)}


# -- the curve file ---------------------------------------------------------------------------------------------------

def curves(result):
    """The hold curve artifact of a measure() `result`: what controlled_touch's `hold_curve` reads.

    {"format": 1, "kind": "hold_curve", "bins_s", "touch", "profile", "seeds", "matches", "counts", "folds", "full"}:
    `counts` is the pooled [frames, agree] cells; `full` and each `folds[match]` are {"no_gap": [10 numbers], "gap":
    [10 numbers]}, each (agree + 1) / (frames + 2) of the pooled counts, for a fold of the pooled counts minus that
    match's. A max_rel_speed of infinity is written "inf" (JSON has none)."""
    pooled = result["pooled"]

    def curve(counts):
        return {name: [(cell[1] + 1) / (cell[0] + 2) for cell in counts[name]] for name in ROWS}

    folds = {}
    for match_id, own in result["per_match"].items():
        rest = {name: [[cell[0] - mine[0], cell[1] - mine[1]] for cell, mine in zip(pooled[name], own[name])]
                for name in ROWS}
        folds[match_id] = curve(rest)
    touch = {key: ("inf" if isinstance(value, float) and math.isinf(value) else value)
             for key, value in result["touch"].items()}
    return {"format": 1, "kind": "hold_curve", "bins_s": list(result["bins_s"]), "touch": touch,
            "profile": result["profile"], "seeds": list(result["seeds"]), "matches": list(result["matches"]),
            "counts": pooled, "folds": folds, "full": curve(pooled)}


def verify(path, fresh):
    """Compare the file at `path` with the measured artifact `fresh`; print what differs; the exit code (0 or 1)."""
    try:
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
    except (OSError, ValueError) as error:
        print("FAILED    cannot read the hold curve {} ({}); measure it with `python {}`".format(
            path, error, "pipeline/benchmark/hold_stats.py --write"))
        return 1
    problems = []
    _differences(committed, json.loads(dumps(fresh)), os.path.basename(path), problems)
    for problem in problems[:20]:
        print("FAILED    {}".format(problem))
    if problems:
        print("FAILED    {} differs from its measurement in {} place(s)".format(path, len(problems)))
        return 1
    print("verified  {} is what the code measures".format(path))
    return 0


def _differences(committed, fresh, where, problems):
    """Append to `problems` a line for every place where two parsed JSON values differ (exactly: floats are read back
    to the bit)."""
    if isinstance(committed, dict) and isinstance(fresh, dict):
        for key in sorted(set(committed) | set(fresh)):
            if key not in committed or key not in fresh:
                problems.append("{}: key {!r} is only in the {}".format(
                    where, key, "measurement" if key in fresh else "file"))
            else:
                _differences(committed[key], fresh[key], "{}.{}".format(where, key), problems)
    elif isinstance(committed, list) and isinstance(fresh, list) and len(committed) == len(fresh):
        for index, (one, other) in enumerate(zip(committed, fresh)):
            _differences(one, other, "{}[{}]".format(where, index), problems)
    elif committed != fresh or type(committed) is not type(fresh):
        problems.append("{}: file {!r}, measurement {!r}".format(where, committed, fresh))


# -- the table --------------------------------------------------------------------------------------------------------

def render(result):
    """The table's lines: per bin, the frames, their share and the agreement of each row, then the gate."""
    pooled = result["pooled"]
    total = sum(cell[0] for name in ROWS for cell in pooled[name])
    edges = result["bins_s"]
    lines = ["{:<14}{}".format("bin (s)", "".join("{:>10}{:>8}{:>8}".format(
        "frames", "share", "agree") for _ in ROWS))]
    lines.append("{:<14}{}".format("", "".join("{:^26}".format(name) for name in ROWS)))
    for index, low in enumerate(edges):
        name = "{:g}-{:g}".format(low, edges[index + 1]) if index + 1 < len(edges) else "{:g}+".format(low)
        cells = "".join("{:>10}{:>8}{:>8}".format(
            pooled[row][index][0], _pct(pooled[row][index][0], total),
            _pct(pooled[row][index][1], pooled[row][index][0])) for row in ROWS)
        lines.append("{:<14}{}".format(name, cells))
    verdict = gate(result)
    if verdict["frames"]:
        lines.append("gate: {} hold frames, agreement {:.1%}, deviation {:.4f} (needs {}): {}".format(
            verdict["frames"], verdict["agreement"], verdict["deviation"], GATE_DEVIATION,
            "PASSES" if verdict["passes"] else "fails"))
    else:
        lines.append("gate: no hold frame at all: fails")
    return lines


def _pct(part, whole):
    return "n/a" if not whole else "{:.1f}%".format(100.0 * part / whole)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure how often the last controlling team still has the ball on hold frames, by time since its "
                    "last touch and whether the ball was hidden since, and P29's gate on it.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--write", action="store_true",
                        help="also write the hold curve file (curves) to --out; the exit code is 0 unless it fails")
    parser.add_argument("--verify", action="store_true",
                        help="compare the file at --out with the measurement; exit 0 if equal, 1 if not")
    parser.add_argument("--out", help="the curve file (default {})".format(DEFAULT_CURVE_PATH))
    args = parser.parse_args(argv)
    if args.write and args.verify:
        parser.error("--write and --verify are two commands: write the file, or check the one there")
    if args.out and not (args.write or args.verify):
        parser.error("--out names the curve file: give --write or --verify with it")
    path = args.out or DEFAULT_CURVE_PATH
    try:
        result = measure(profile=args.profile, seeds=args.seeds, data_dir=args.data_dir)
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("")
    for line in render(result):
        print(line)
    if args.write:
        try:
            write(path, curves(result))
        except (OSError, ValueError) as error:
            print("FAILED    {}".format(error))
            return 1
        print("wrote     {} ({:,} bytes)".format(path, os.path.getsize(path)))
        return 0
    if args.verify:
        return verify(path, curves(result))
    return 0 if gate(result)["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
