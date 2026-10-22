"""Self-checks for the P01 benchmark: the DFL registry, download, match information and positions now; the rest later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB),
                                                    #   the 7 match information files parsed, J03WMX's positions parsed
                                                    #   and cross-checked against kloppy and by frame number, and the
                                                    #   T30 counts on all 7 positions files (about 12 minutes)
    python pipeline/benchmark/selftest.py --data    # + checks that read the real match's data (aggregates only)

This file is the harness and the one command: it collects the checks, filters them by flag, runs them and reports.
The checks themselves live in pipeline/benchmark/checks/, one module per topic (sources.py, dfl.py, ...), listed in
checks.MODULES; to add one, write a function named test_* in the module of its topic, or start a new module there and
list it. See pipeline/benchmark/checks/__init__.py.

The runner is pipeline/selftest.py's: every function named test_* is a check, it prints PASS or FAIL for each, and
the exit code is 1 if anything failed. Checks named test_dfl_* need the downloaded DFL files and run only with --dfl;
test_data_* need the real match and run only with --data. Nothing from the real match is ever printed -- counts,
shares and scores, never a row, a coordinate or a label sequence (CLAUDE.md, hard rules). The DFL files are public
(CC BY 4.0), so their aggregates may be printed freely.

The download checks run the real streaming, hashing and rename code against a fake opener that serves small bodies
from memory, so no network is touched and nothing outside a temporary folder is written. The parser checks read a
synthetic match written by write_synthetic_dfl (checks/common.py) in the real file format, so they need no download
either.
"""

import argparse
import inspect
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so
    # that `import pipeline` resolves from any working directory (as in pipeline/selftest.py).
    sys.path[0] = ROOT

# Importing it also puts evaluation/ on sys.path.
from pipeline.benchmark import checks  # noqa: E402
from pipeline.benchmark.checks.common import utf8_stdout  # noqa: E402


def collect(modules=None):
    """[(name, function)] of every check in `modules` (default checks.MODULES), sorted by name.

    A check is a function named test_* that the module defines itself: one it merely imports does not count. The
    names are unique across the modules, because the harness filters and orders by name alone.
    """
    found = {}
    for module in checks.MODULES if modules is None else modules:
        for name, value in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(value) and value.__module__ == module.__name__:
                if name in found:
                    raise ValueError("the check {} is defined twice ({} and {}); a check's name must be unique "
                                     "across pipeline/benchmark/checks/".format(
                                         name, found[name].__module__, module.__name__))
                found[name] = value
    return sorted(found.items(), key=lambda item: item[0])


def main(argv=None):
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB), parse the 7 "
                             "match information files, cross-check J03WMX's positions against kloppy and by frame "
                             "number, and count T30's facts on all 7 positions files (about 12 minutes)")
    parser.add_argument("--data", action="store_true", help="also run the checks that read the real match's data")
    args = parser.parse_args(argv)

    selected = collect()
    if not args.dfl:
        selected = [(name, check) for name, check in selected if not name.startswith("test_dfl_")]
    if not args.data:
        selected = [(name, check) for name, check in selected if not name.startswith("test_data_")]

    failures = 0
    for name, check in selected:
        started = time.perf_counter()
        try:
            check()
        except Exception as error:  # report every failure; one broken check must not hide the rest
            failures += 1
            print("FAIL  {}  {}: {}".format(name, type(error).__name__, error))
        else:
            print("PASS  {}  ({:.1f} s)".format(name, time.perf_counter() - started))

    skipped = [flag for flag, on in (("--dfl", args.dfl), ("--data", args.data)) if not on]
    print("\n{} of {} checks passed{}".format(
        len(selected) - failures, len(selected),
        "  (add {} for the rest)".format(" / ".join(skipped)) if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
