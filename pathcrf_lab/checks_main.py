"""Checks of pathcrf_lab that run in the main .venv on synthetic input: isolation, sources, adapter, scorer."""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _pipeline_files():
    """Every .py under pipeline/, evaluation/ and starter/, plus games.py."""
    found = [os.path.join(ROOT, "games.py")]
    for folder in ("pipeline", "evaluation", "starter"):
        for base, _, names in os.walk(os.path.join(ROOT, folder)):
            found += [os.path.join(base, name) for name in names if name.endswith(".py")]
    return found


def test_nothing_imports_pathcrf_lab():
    importing = re.compile(r"^\s*(import|from)\s+pathcrf_lab")
    files = _pipeline_files()
    assert len(files) > 10, "found only {} files to scan; the walk is broken".format(len(files))
    for path in files:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        name = os.path.relpath(path, ROOT)
        assert "pathcrf_lab" not in text, "{} mentions pathcrf_lab; the measurement must stay out of the " \
                                          "pipeline".format(name)
        assert not any(importing.match(line) for line in text.splitlines()), "{} imports pathcrf_lab".format(name)


def test_pathcrf_lab_imports_no_torch_at_module_level():
    code = ("import sys; sys.path.insert(0, {!r}); "
            "import pathcrf_lab.sources, pathcrf_lab.adapter, pathcrf_lab.runner, pathcrf_lab.score; "
            "assert 'torch' not in sys.modules, 'torch was imported at module level'").format(ROOT)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, "importing the modules failed or pulled in torch: {}".format(done.stderr.strip()[-300:])


def test_gitignore_covers_the_venv_and_data():
    with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    assert ".venv-pathcrf/" in lines, ".gitignore must list .venv-pathcrf/"
    assert "data/*" in lines, ".gitignore must still list data/*"
