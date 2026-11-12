"""hold_stats: how often the last controlling team still has the ball, by time since its touch and ball hidden since.

    python pipeline/benchmark/hold_stats.py [--profile P] [--seeds N] [--data-dir D]

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

No truth reaches the source. hold_state is called on a MatchInputs (inputs.py), which holds the degraded tracking and no
flag; the truth is read here, from the match, only to count, and the result holds counts and no frame.
"""

import argparse
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
from pipeline.config import ConfigError  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
# main.toml's control rule (R030); a check holds it against the config. Read at call time, never bound as a default.
TOUCH = {"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "duel_radius": 1.0}
GATE_DEVIATION = 0.05           # the frame-weighted mean |cell agreement - pooled agreement| the curve must reach
DEFAULT_PROFILE = run.DEFAULT_PROFILE
ROWS = ("no_gap", "gap")        # the ball was never hidden since the last touch, or was hidden on some frame


def measure(*, touch=None, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None, data_dir=None,
            log=print):
    """Frames and agreements of the hold frames by gap and bin, per match and pooled over matches and seeds.

    `touch` is the control rule, {radius, max_rel_speed, min_frames, duel_radius} (None: TOUCH, read now). The others
    are those of run.evaluate_configs: `profile`, `seeds` (0 up to seeds - 1; the clean profile has just seed 0),
    `match_ids`, `matches` ({id: DflMatch} to use instead of loading) and `data_dir`; `log` receives a line per match.

    Returns {"profile", "seeds", "matches", "bins_s", "touch", "pooled", "per_match"}; pooled and each per_match[id]
    are {"no_gap": [[frames, agree] x 10], "gap": [[frames, agree] x 10]} over the hold frames whose truth is H or A,
    `agree` counting those where the truth's team is the last controller's. ValueError from controlled_touch for a rule
    it refuses; the runner's BenchmarkError for a profile, seeds or match it refuses (before a match is loaded).
    """
    touch = dict(TOUCH if touch is None else touch)
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
    args = parser.parse_args(argv)
    try:
        result = measure(profile=args.profile, seeds=args.seeds, data_dir=args.data_dir)
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("")
    for line in render(result):
        print(line)
    return 0 if gate(result)["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
