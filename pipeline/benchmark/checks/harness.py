"""Checks of the harness itself (pipeline/benchmark/selftest.py): what it collects, how it reports, and the console."""

import contextlib
import io
import re
import sys
import tempfile
import types

from pipeline.benchmark.checks.common import utf8_stdout
from pipeline.selftest import expect_error, write_text


def fake_module(name, **functions):
    """A module called `name` whose attributes are `functions`, each of which says it was defined there."""
    module = types.ModuleType(name)
    for attribute, function in functions.items():
        function.__module__ = name
        setattr(module, attribute, function)
    return module


def run_main(modules, *flags):
    """(exit code, output lines) of the harness's main() over `modules`, with the timings left out of the lines."""
    from pipeline.benchmark import selftest
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        code = selftest.main(list(flags), modules=modules)
    return code, [re.sub(r"  \([0-9.]+ s\)$", "", line) for line in shown.getvalue().rstrip("\n").split("\n")]


def test_collect_takes_the_checks_a_module_defines_and_sorts_them_by_name():
    from pipeline.benchmark import selftest

    def test_b():
        pass

    def test_a():
        pass

    def helper():
        pass

    def test_c():
        pass

    first = fake_module("fake_one", test_b=test_b, test_a=test_a, helper=helper)
    second = fake_module("fake_two", test_c=test_c)
    first.test_imported = test_c  # defined in fake_two, only imported by fake_one: not fake_one's check
    first.test_not_a_function = 3
    names = [name for name, _ in selftest.collect([second, first])]
    assert names == ["test_a", "test_b", "test_c"], "sorted by name over all modules, once each: {}".format(names)

    def test_dup():
        pass

    twin = fake_module("fake_three", test_a=test_dup)
    expect_error(ValueError, lambda: selftest.collect([first, twin]), "test_a", "fake_one", "fake_three", "unique")


def test_a_check_module_that_is_not_listed_is_refused():
    """A module in checks/ that checks.MODULES does not list would run nothing, silently: the harness refuses that."""
    from pipeline.benchmark import checks, selftest
    assert selftest.unlisted_modules() == [], "every module of pipeline/benchmark/checks/ must be in MODULES"

    with tempfile.TemporaryDirectory() as folder:
        for name in ("__init__", "common", "alpha", "beta"):  # common holds no checks and is the one exception
            write_text(folder, name + ".py", "")
        listed = [fake_module("pkg.checks.alpha")]
        assert selftest.unlisted_modules([folder], listed) == ["beta"]
        assert selftest.unlisted_modules([folder], listed + [fake_module("pkg.checks.beta")]) == []

    real_modules = checks.MODULES
    checks.MODULES = real_modules[:-1]
    try:
        expect_error(ValueError, selftest.collect, real_modules[-1].__name__.rsplit(".", 1)[-1], "MODULES",
                     "checks/__init__.py")
    finally:
        checks.MODULES = real_modules

    # main() says so and fails, instead of a traceback or a quiet run of fewer checks.
    real_collect = selftest.collect

    def refusing(modules=None):
        raise ValueError("checks/beta.py is not in MODULES")

    selftest.collect = refusing
    try:
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = selftest.main([])
    finally:
        selftest.collect = real_collect
    assert code == 1 and "checks/beta.py is not in MODULES" in shown.getvalue(), (code, shown.getvalue())


def test_a_check_that_could_not_run_is_not_a_pass():
    from pipeline.benchmark.checks.common import NotRun

    def test_fine():
        pass

    def test_cannot():
        raise NotRun("kloppy is not installed")

    def test_broken():
        raise AssertionError("it is wrong")

    def test_dfl_needs_files():
        raise AssertionError("it must not run without --dfl")

    module = fake_module("fake_checks", test_fine=test_fine, test_cannot=test_cannot,
                         test_dfl_needs_files=test_dfl_needs_files)

    # A NOT RUN is named with its reason, counted apart from the passes, and does not fail the run.
    code, lines = run_main([module])
    assert lines == ["NOT RUN  test_cannot: kloppy is not installed", "PASS  test_fine", "",
                     "1 passed, 1 not run  (add --dfl / --data for the rest)"], lines
    assert code == 0, code

    module.test_broken = test_broken
    test_broken.__module__ = "fake_checks"
    code, lines = run_main([module], "--dfl", "--data")
    assert lines == ["FAIL  test_broken  AssertionError: it is wrong",
                     "NOT RUN  test_cannot: kloppy is not installed",
                     "FAIL  test_dfl_needs_files  AssertionError: it must not run without --dfl",
                     "PASS  test_fine", "", "1 passed, 1 not run, 2 failed"], lines
    assert code == 1, code

    code, lines = run_main([fake_module("fake_clean", test_fine=test_fine)], "--dfl", "--data")
    assert lines == ["PASS  test_fine", "", "1 passed"] and code == 0, (code, lines)


def test_stdout_is_utf8():
    """A team name such as "1. FC Köln" must reach a pipe in a legacy code page as UTF-8, not as "K?ln"."""
    saved = sys.stdout
    try:
        raw = io.BytesIO()
        sys.stdout = io.TextIOWrapper(raw, encoding="ascii", errors="replace", newline="\n")
        utf8_stdout()
        print("1. FC Köln")
        sys.stdout.flush()
        assert raw.getvalue() == "1. FC Köln\n".encode("utf-8"), raw.getvalue()
        sys.stdout = io.StringIO()  # no reconfigure(): the call must leave it alone
        utf8_stdout()
    finally:
        sys.stdout = saved
