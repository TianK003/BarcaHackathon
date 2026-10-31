"""Self-checks for pathcrf_lab (P20 step 0, R022).

    python pathcrf_lab/selftest.py           # the synthetic checks of checks_main.py, in .venv: seconds, no data
    python pathcrf_lab/selftest.py --torch   # the checks of checks_torch.py, in .venv-pathcrf; NOT RUN while torch
                                             #   or the fetched repo is missing

The loop is the one of pipeline/benchmark/selftest.py: every function named test_* is a check, it prints PASS, FAIL or
NOT RUN (the check raised NotRun and says why; counted apart, never a pass), and the exit code is 1 on a FAIL. Nothing
from a match is ever printed.
"""

import argparse
import importlib
import inspect
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NotRun(Exception):
    """A check that could not run on this machine, as opposed to one that ran and failed."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def collect(module):
    """[(name, function)] of the test_* functions `module` defines itself, sorted by name."""
    return sorted((name, value) for name, value in vars(module).items()
                  if name.startswith("test_") and inspect.isfunction(value) and value.__module__ == module.__name__)


def run_checks(selected, log=print):
    """Run each (name, check) and log one line for it; return {"passed", "not_run", "failed"} counts."""
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
            log("FAIL  {}: {}: {}".format(name, type(error).__name__, error))
        else:
            counts["passed"] += 1
            log("PASS  {}  ({:.1f} s)".format(name, time.perf_counter() - started))
    return counts


def main(argv=None):
    """Run the checks the flags select; the exit code is 1 on a FAIL."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--torch", action="store_true", help="run checks_torch (needs .venv-pathcrf and the repo)")
    args = parser.parse_args(argv)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    module = importlib.import_module("pathcrf_lab.checks_torch" if args.torch else "pathcrf_lab.checks_main")
    counts = run_checks(collect(module))
    parts = ["{} passed".format(counts["passed"])]
    parts += ["{} {}".format(counts[key], label) for key, label in (("not_run", "not run"), ("failed", "failed"))
              if counts[key]]
    print("\n" + ", ".join(parts))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    # Run by path this file is __main__: import the package's own copy, so that NotRun is one class for the checks.
    sys.path.insert(0, ROOT)
    from pathcrf_lab.selftest import main as _main
    sys.exit(_main())
