"""Checks of the harness itself (pipeline/benchmark/selftest.py): what it collects, how it reports, and the console."""

import ast
import contextlib
import io
import os
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


def absolute_module(node, package):
    """The dotted name that an `ast.ImportFrom` of a module of `package` takes from, with a relative import made
    absolute: one dot is `package` itself, two are its parent, and so on. None for a relative import that climbs
    out of the top-level package (it names nothing here)."""
    if not node.level:
        return node.module
    parts = package.split(".")
    if node.level > len(parts):
        return None
    base = ".".join(parts[:len(parts) - node.level + 1])
    return base + ("." + node.module if node.module else "")


def imports_of_other_check_modules(folder, package):
    """[(file, line, module)] of every import in the modules of `folder` that reaches a module of `package` other than
    common, sorted by file and line. The package's own __init__ is not scanned: it imports all the modules to list them.

    `package` is the dotted name of the folder (pipeline.benchmark.checks). An import is found wherever it stands, at the
    top or inside a function, whether it names the module (from package.run import x, import package.run) or takes it
    from the package (from package import run), and whether it is absolute or relative (from .run import x, from .
    import run). A module of another package whose name only starts like it (pipeline.benchmark.checks_elsewhere)
    is not one.
    """
    allowed = package + ".common"
    found = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), name)
        for node in ast.walk(tree):
            module = absolute_module(node, package) if isinstance(node, ast.ImportFrom) else None
            if module == package:
                found += [(name, node.lineno, package + "." + alias.name) for alias in node.names
                          if package + "." + alias.name != allowed]
            elif module and module.startswith(package + "."):
                found += [(name, node.lineno, module)] if module != allowed else []
            elif isinstance(node, ast.Import):
                found += [(name, node.lineno, alias.name) for alias in node.names
                          if alias.name.startswith(package + ".") and alias.name != allowed]
    return sorted(found)


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
    """A check module that checks.MODULES does not list would run nothing, silently; one that is not a check module
    (a production module of the same name, say) would hide that. The harness refuses both."""
    import importlib
    from pipeline.benchmark import checks, selftest
    assert selftest.module_list_problems() == [], "every module of pipeline/benchmark/checks/ must be in MODULES"

    with tempfile.TemporaryDirectory() as folder:
        for name in ("__init__", "common", "alpha", "beta"):  # common holds no checks and is the one exception
            write_text(folder, name + ".py", "")

        def problems(*names):
            return selftest.module_list_problems([folder], [fake_module(name) for name in names], "pkg.checks")

        assert len(problems("pkg.checks.alpha")) == 1 and "beta.py" in problems("pkg.checks.alpha")[0]
        assert problems("pkg.checks.alpha", "pkg.checks.beta") == []
        # Names are compared whole: a production module called alpha does not stand in for the checks' alpha.
        found = problems("pipeline.benchmark.alpha", "pkg.checks.beta")
        assert len(found) == 2 and "alpha.py" in found[0] and "pipeline.benchmark.alpha" in found[1], found
        stranger = problems("pkg.checks.alpha", "pkg.other.beta", "pkg.checks.beta")
        assert len(stranger) == 1 and "pkg.other.beta" in stranger[0], stranger
        # An entry with no file of the same name in checks/ has nothing to "list instead": it says so.
        orphan = problems("pkg.checks.alpha", "pkg.checks.beta", "pkg.other.gamma", "pkg.checks.delta")
        assert len(orphan) == 2 and "pkg.other.gamma" in orphan[0] and "no gamma.py" in orphan[0] \
            and "instead" not in orphan[0], orphan
        assert "pkg.checks.delta" in orphan[1] and "no delta.py" in orphan[1], orphan

    real_modules = checks.MODULES
    checks.MODULES = real_modules[:-1]
    try:
        expect_error(ValueError, selftest.collect, real_modules[-1].__name__.rsplit(".", 1)[-1], "MODULES",
                     "checks/__init__.py")
    finally:
        checks.MODULES = real_modules
    # The mistake that was made: the production dfl.py listed in place of checks/dfl.py. 40 checks went quiet.
    production = importlib.import_module("pipeline.benchmark.dfl")
    checks.MODULES = tuple(production if module.__name__.endswith(".dfl") else module for module in real_modules)
    try:
        expect_error(ValueError, selftest.collect, "dfl.py", "pipeline.benchmark.dfl", "MODULES")
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


def test_a_check_module_imports_no_other_check_module():
    """A check module imports from checks/common.py and from production code, and from no other check module.

    Two check modules that import each other make their order of import matter, a helper that two of them share belongs
    to neither, and a module that is renamed breaks a stranger's import; the review of Task 14 found eight such imports
    (two in common.py, three in fidelity.py, three in run.py), and they were moved to common.py. The scan reads every import in the module, at the top and inside a function, in both
    forms (`from pipeline.benchmark.checks.run import x`, `from pipeline.benchmark.checks import run`, and `import
    pipeline.benchmark.checks.run`) and the relative ones (`from .run import x`, `from . import run`; the review of
    Task 17 found that the first scan let them pass). It must flag each of those on made-up modules, let the
    imports of common and of production code pass (and the __init__, which lists the modules by design), and find
    none in the real folder."""
    package = "pipeline.benchmark.checks"
    with tempfile.TemporaryDirectory() as folder:
        write_text(folder, "__init__.py", "from pipeline.benchmark.checks import alpha, beta\n")
        write_text(folder, "common.py", "")
        write_text(folder, "clean.py", "import os\nfrom pipeline.benchmark import dfl\nfrom pipeline.benchmark import checks\n"
                                       "from pipeline.benchmark.checks import common\n"
                                       "from pipeline.benchmark.checks.common import helper\nimport pipeline.benchmark.checks.common\n"
                                       "from pipeline.benchmark.checks_elsewhere import run\n")
        write_text(folder, "alpha.py", "from pipeline.benchmark.checks.beta import helper\n")
        write_text(folder, "beta.py", "def test_one():\n    from pipeline.benchmark.checks.alpha import helper\n")
        write_text(folder, "gamma.py", "from pipeline.benchmark.checks import common, alpha\n")
        write_text(folder, "delta.py", "import pipeline.benchmark.checks.alpha\n")
        # Relative imports are inside the package: one dot is the checks folder, two are pipeline.benchmark.
        write_text(folder, "epsilon.py", "from .alpha import helper\n")
        write_text(folder, "zeta.py", "def test_one():\n    from . import alpha, common\n")
        write_text(folder, "eta.py", "from . import common\nfrom .common import helper\nfrom .. import dfl\n"
                                      "from ..dfl import parse\nfrom .. import checks\nfrom ... import checks_elsewhere\n")
        found = imports_of_other_check_modules(folder, package)
        assert found == [("alpha.py", 1, package + ".beta"), ("beta.py", 2, package + ".alpha"),
                         ("delta.py", 1, package + ".alpha"), ("epsilon.py", 1, package + ".alpha"),
                         ("gamma.py", 1, package + ".alpha"), ("zeta.py", 2, package + ".alpha")], found
    here = os.path.dirname(os.path.abspath(__file__))
    assert imports_of_other_check_modules(here, package) == [], (
        "a check module imports another one; move what they share to pipeline/benchmark/checks/common.py")
