"""Checks that need torch and the fetched PathCRF repo; run with `.venv-pathcrf` and `selftest.py --torch`.

A stub until the runner exists. Every check raises NotRun when torch does not import or sources.verify() is not all
ok, so that a machine without them reports NOT RUN and never a pass.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathcrf_lab.selftest import NotRun  # noqa: E402


def _need_torch_and_repo():
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise NotRun("torch does not import here ({})".format(error))
    from pathcrf_lab import sources
    bad = {path: status for path, status in sources.verify().items() if status != "ok"}
    if bad:
        raise NotRun("{} of the pinned PathCRF files are not ok on disk; fetch them first".format(len(bad)))


def test_torch_and_repo_present():
    _need_torch_and_repo()
