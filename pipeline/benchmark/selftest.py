"""Self-checks for the P01 benchmark: the DFL registry, download, parsers and cache now; the rest later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB),
                                                    #   the 7 matches loaded through the cache, J03WMX's cache
                                                    #   compared with a fresh parse, J03WMX cross-checked against
                                                    #   kloppy and by frame number, the T30 counts on all 7
                                                    #   matches, the truth's conventions on all 7 (A06), the 7
                                                    #   matches served through MatchInputs and J03WMX scored through
                                                    #   predict and pooled() against pm.evaluate (measured: 532 s
                                                    #   while the 7 caches are built under data/cache/dfl/, then 99 to
                                                    #   221 s, 111 s with the last two included: the machine's load
                                                    #   decides)
    python pipeline/benchmark/selftest.py --data    # + checks that read the real match's data (aggregates only): the
                                                    #   calibration reproduces A01's, A02's and R001's figures, and the
                                                    #   hazard of our ball gaps is fitted again and printed (A08)
                                                    #   (measured: about 19 s with the ATD cache and the calibration
                                                    #   cache present, 22 s while the calibration is built)

This file is the harness and the one command: it collects the checks, filters them by flag, runs them and reports.
The checks themselves live in pipeline/benchmark/checks/, one module per topic (sources.py, dfl.py, ...), listed in
checks.MODULES; to add one, write a function named test_* in the module of its topic, or start a new module there and
list it. See pipeline/benchmark/checks/__init__.py.

The run loop is its own but in the style of pipeline/selftest.py's: every function named test_* is a check, it prints
PASS or FAIL for each, and the exit code is 1 if anything failed. A check that could not run here (an optional
dependency such as kloppy is not installed) raises NotRun and is printed as NOT RUN with its reason: it is counted
apart from the passes and does not fail the run, but it is never reported as a pass. Checks named test_dfl_* need the
downloaded DFL files and run only with --dfl; test_data_* need the real match and run only with --data. Nothing from
the real match is ever printed -- counts, shares and scores, never a row, a coordinate or a label sequence (CLAUDE.md,
hard rules). The DFL files are public (CC BY 4.0), so their aggregates may be printed freely.

The download checks run the real streaming, hashing and rename code against a fake opener that serves small bodies
from memory, so no network is touched and nothing outside a temporary folder is written. The parser checks read a
synthetic match written by write_synthetic_dfl (checks/common.py) in the real file format, so they need no download
either.
"""

import argparse
import inspect
import os
import pkgutil
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
from pipeline.benchmark.checks.common import NotRun, utf8_stdout  # noqa: E402


def module_list_problems(path=None, modules=None, package=None):
    """What is wrong with the list of check modules, as sentences; an empty list if nothing is.

    `modules` (default checks.MODULES) must be exactly the modules of `package` (default the checks package) that
    are in the folders `path` (default the package's own), compared by their whole names. A module that is not listed
    would silently run none of its checks, and a listed one that is not a module of the package (the production
    module of the same name, say) would hide that. common.py is the exception: it holds what the others share, and no
    check, so it may be left out.
    """
    package = checks.__name__ if package is None else package
    listed = [module.__name__ for module in (checks.MODULES if modules is None else modules)]
    present = sorted(info.name for info in pkgutil.iter_modules(checks.__path__ if path is None else path))
    folder = package.replace(".", "/") + "/"
    problems = ["{} has {}.py, which MODULES does not list, so its checks would never run; add it to MODULES in "
                "{}__init__.py".format(folder, name, folder)
                for name in present if name != "common" and package + "." + name not in listed]
    for name in listed:
        parent, _, short = name.rpartition(".")
        if parent != package:
            # Listed from another package (the production module of the same name, say): the checks' own is the one.
            problems.append("MODULES lists {}, which is not a module of {}; list the module of the same name in {} "
                            "instead".format(name, package, folder) if short in present else
                            "MODULES lists {}, which is not a module of {}, and {} has no {}.py to list in its "
                            "place; remove it from MODULES, or write the module".format(name, package, folder, short))
        elif short not in present:
            problems.append("MODULES lists {}, but {} has no {}.py; remove it from MODULES, or write the module"
                            .format(name, folder, short))
    return problems


def collect(modules=None):
    """[(name, function)] of every check in `modules` (default checks.MODULES), sorted by name.

    A check is a function named test_* that the module defines itself: one it merely imports does not count. The
    names are unique across the modules, because the harness filters and orders by name alone. With the default
    modules, a list that is not exactly the modules of the checks package is a ValueError that says what is wrong
    (module_list_problems).
    """
    problems = module_list_problems() if modules is None else []
    if problems:
        raise ValueError("; ".join(problems))
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


def run_checks(selected, log=print):
    """Run each (name, check) in order and log one line for it; return {"passed", "not_run", "failed"} counts.

    PASS, NOT RUN (the check raised NotRun, and says why) or FAIL (anything else it raised). One broken check must
    not hide the rest, so every exception is reported and the loop goes on.
    """
    counts = {"passed": 0, "not_run": 0, "failed": 0}
    for name, check in selected:
        started = time.perf_counter()
        try:
            check()
        except NotRun as stop:
            counts["not_run"] += 1
            log("NOT RUN  {}: {}".format(name, stop.reason))
        except Exception as error:
            counts["failed"] += 1
            log("FAIL  {}  {}: {}".format(name, type(error).__name__, error))
        else:
            counts["passed"] += 1
            log("PASS  {}  ({:.1f} s)".format(name, time.perf_counter() - started))
    return counts


def summary(counts, skipped_flags):
    """The last line of a run, e.g. "38 passed, 1 not run  (add --dfl for the rest)": the three counts kept apart."""
    parts = ["{} passed".format(counts["passed"])]
    parts += ["{} {}".format(counts[key], label) for key, label in (("not_run", "not run"), ("failed", "failed"))
              if counts[key]]
    return ", ".join(parts) + ("  (add {} for the rest)".format(" / ".join(skipped_flags)) if skipped_flags else "")


def main(argv=None, modules=None):
    """Run the checks of `modules` (default: checks.MODULES) that the flags select; the exit code is 1 on a FAIL."""
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB), parse the 7 "
                             "match information files, load the 7 matches through the cache, compare J03WMX's "
                             "cache with a fresh parse, cross-check J03WMX against kloppy and by frame number, "
                             "count T30's facts on all 7, and measure the truth's conventions on all 7 (A06) "
                             "(measured: 532 s while the caches are built, then 99 to 221 s)")
    parser.add_argument("--data", action="store_true", help="also run the checks that read the real match's data")
    args = parser.parse_args(argv)

    try:
        selected = collect(modules)
    except ValueError as error:
        print("FAIL  collecting the checks: {}".format(error))
        return 1
    if not args.dfl:
        selected = [(name, check) for name, check in selected if not name.startswith("test_dfl_")]
    if not args.data:
        selected = [(name, check) for name, check in selected if not name.startswith("test_data_")]

    counts = run_checks(selected)
    skipped = [flag for flag, on in (("--dfl", args.dfl), ("--data", args.data)) if not on]
    print("\n" + summary(counts, skipped))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
