"""Checks of the console set-up that the benchmark harness relies on."""

import io
import sys

from pipeline.benchmark.checks.common import utf8_stdout


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
