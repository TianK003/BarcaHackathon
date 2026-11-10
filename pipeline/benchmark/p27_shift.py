"""p27_shift: the shift of P27's Stage E, read from the pipeline's own lag on the benchmark (R012's rule).

    python pipeline/benchmark/p27_shift.py STEM        # e.g. p27_preshift

Reads <data dir>/runs/conventions/latency_<STEM>_dfl_generic_raw.json, which
`conventions.py latency pipeline/benchmark/configs/p27/<STEM>.toml dfl --profile generic --seeds 3 --truth raw` wrote,
and prints, per type of edge, the changes, the matched ones, the median lag and s_b = conventions.pipeline_lag: minus
the rounded median, and 0 below 30 matched changes. The last line starts with `NEXT` and holds the --shift argument of
p27_grid.py final (as `--shift=a,b,c`, which argparse takes whatever the sign of a). It reads DFL's lags and nothing of our match (conventions.py's `shift` command also reads the
public window, which is why it is not used here).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline.benchmark import conventions  # noqa: E402

PROFILE, TRUTH = conventions.SHIFT_DFL


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: p27_shift.py STEM")
        return 1
    stem = argv[0]
    path = conventions.latency_path(games.DATA_DIR, stem, "dfl", PROFILE, TRUTH)
    if not os.path.exists(path):
        print("FAILED    no latency file at {}; run conventions.py latency first".format(path))
        return 1
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("profile") != PROFILE or document.get("truth") != TRUTH:
        print("FAILED    {} is of profile {!r} and truth {!r}, not {} and {}".format(
            path, document.get("profile"), document.get("truth"), PROFILE, TRUTH))
        return 1
    shifts = []
    for kind in conventions.TYPES:
        cell = document["cells"][kind]
        lags = cell["lags"]
        median = float(np.median(lags)) if lags else float("nan")
        shift = conventions.pipeline_lag(lags)
        shifts.append(shift)
        print("type      {:12s} changes {:6d}  matched {:6d}  median lag {:+.1f} frames  s_b {:+d}{}".format(
            kind, cell["n"], len(lags), median, shift,
            "" if len(lags) >= conventions.MIN_DFL else "  (below {} matched: 0)".format(conventions.MIN_DFL)))
    # "--shift=" and not "--shift ": argparse would take a value that starts with a minus sign for an option.
    print("NEXT --shift={}".format(",".join(str(shift) for shift in shifts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
