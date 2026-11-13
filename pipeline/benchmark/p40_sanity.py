"""p40_sanity: P40's feature sanity step, run before the shape model of version 5 is trained.

    python pipeline/benchmark/p40_sanity.py --matches J03WMX --seeds 1      # one DFL match, degradation seed 0

It serves each match as the trainer does (degraded by the profile, no truth), computes `shape_features` versions 2 and
5 on it, and prints aggregates only: for each of the nine new columns (`press_mean`, `press_p90`, `press_share` per team
and the three cross columns) the share of NaN and the standard deviation over the graded frames, and whether version
2's 42 columns are equal in version 5 (bit for bit, NaN included). It exits 1 on a new column that is all NaN or
constant, or on a version 2 column that is not equal: those are bugs, not results. It trains nothing and scores
nothing; nothing of a frame is printed.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import train_shape  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
from pipeline.benchmark.train_dead_ball import served_inputs  # noqa: E402
from pipeline.evidence import shape_features  # noqa: E402

# Params
NEW_COLUMNS = (tuple("h_" + name for name in shape_features.PRESS_COLUMNS)
               + tuple("a_" + name for name in shape_features.PRESS_COLUMNS) + shape_features.PRESS_CROSS)


def check(inputs, other=None):
    """([line, ...], ok) for one served Inputs: the nine new columns' NaN share and standard deviation over the graded
    frames, and whether version 2's columns are equal in version 5. `other` is the same match served again (a second
    Inputs, so that nothing is shared in the memo): version 2 is computed on it, version 5 on `inputs`, and the
    equality is a comparison of two separate computations. Without it version 2 is taken from `inputs` itself."""
    names2, X2 = shape_features.features(inputs if other is None else other, 2)
    names5, X5 = shape_features.features(inputs, 5)
    column5 = shape_features.columns(5)
    frames = np.concatenate([np.arange(start, stop) for _, start, stop in inputs.graded])
    ok, lines = True, []
    equal = all(np.array_equal(X2[:, index], X5[:, column5[name]], equal_nan=True) for index, name in enumerate(names2))
    ok &= equal
    lines.append("version 2's {} columns equal in version 5: {}".format(len(names2), "yes" if equal else "NO"))
    lines.append("{:<20} {:>10} {:>12}  {}".format("column", "nan share", "std", ""))
    for name in NEW_COLUMNS:
        values = X5[frames, column5[name]]
        finite = values[np.isfinite(values)]
        std = float(finite.std()) if len(finite) else float("nan")
        flag = ""
        if len(finite) == 0:
            flag = "ALL NaN"
        elif not std > 0.0:
            flag = "CONSTANT"
        ok &= flag == ""
        lines.append("{:<20} {:>10.4f} {:>12.6f}  {}".format(name, 1.0 - len(finite) / len(values), std, flag))
    return lines, ok


def sanity(profile, seeds, match_ids=None, matches=None, data_dir=None, log=print, calibration=None):
    """True if every (match, seed) passes; prints each one's lines through `log`. `matches` is {id: DflMatch} to use
    instead of loading from `data_dir` and `calibration` the Calibration of a degraded profile (the checks pass synthetic
    ones)."""
    train_shape._known_profile(profile)
    ids = run.resolve_match_ids(match_ids, matches)
    calibration = train_shape._calibration_for(profile, calibration, data_dir)
    everything = True
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        for seed in train_shape._seed_list(profile, seeds):
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            lines, ok = check(inputs, served_inputs(match, profile, seed, calibration, data_dir))
            log("{} seed {}  ({})".format(match_id, seed, "ok" if ok else "FAILED"))
            for line in lines:
                log("  " + line)
            everything &= ok
            del inputs
    return bool(everything)


def main(argv=None):
    parser = argparse.ArgumentParser(description="P40's feature sanity step: the nine new columns of shape features "
                                                 "version 5 on a degraded DFL match, aggregates only.")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int, default=1, help="degradation seeds per match (default 1)")
    parser.add_argument("--matches", nargs="+", default=None, help="match ids (default: all seven)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    try:
        ok = sanity(args.profile, args.seeds, args.matches, data_dir=args.data_dir)
    except (BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("sanity    {}".format("passed" if ok else "FAILED: a new column is all NaN or constant, or a version 2 "
                                                    "column is not equal"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
