"""Self-checks for the P01 benchmark: the DFL registry and download now; parser, truth, scoring, degradation later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB)
    python pipeline/benchmark/selftest.py --data    # + checks that read the real match's data (aggregates only)

The runner is pipeline/selftest.py's: every function named test_* is a check, it prints PASS or FAIL for each, and
the exit code is 1 if anything failed. Checks named test_dfl_* need the downloaded DFL files and run only with --dfl;
test_data_* need the real match and run only with --data. Nothing from the real match is ever printed -- counts,
shares and scores, never a row, a coordinate or a label sequence (CLAUDE.md, hard rules). The DFL files are public
(CC BY 4.0), so their aggregates may be printed freely.

The download checks run the real streaming, hashing and rename code against a fake opener that serves small bodies
from memory, so no network is touched and nothing outside a temporary folder is written.
"""

import argparse
import contextlib
import hashlib
import io
import os
import sys
import tempfile
import time
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so
    # that `import pipeline` resolves from any working directory (as in pipeline/selftest.py).
    sys.path[0] = ROOT

from pipeline.selftest import expect_error, quiet  # noqa: E402  (importing it also puts evaluation/ on sys.path)

# Params
# 2.56 MB: three 1 MiB reads, the last one short, so the streaming loop really loops.
GOOD_BODY = bytes(range(256)) * 10_000


# -- helpers ----------------------------------------------------------------------------------------------------------

class FakeOpener:
    """Stands in for urllib.request.urlopen: serves a body per URL and remembers every request it got.

    A value in `bodies` that is an exception is raised instead of served, which is how a dead server looks.
    """

    def __init__(self, bodies):
        self.bodies = bodies
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.get_header("User-agent"), timeout))
        body = self.bodies[request.full_url]
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body)

    @property
    def urls(self):
        return [url for url, _, _ in self.requests]


def refuse_every_request(request, timeout=None):
    raise AssertionError("a request was made to {}".format(request.full_url))


def probe_file(body=GOOD_BODY):
    """A RemoteFile for `body`, outside the real registry, so that no check touches the 2.6 GB of real data."""
    from pipeline.benchmark.sources import RemoteFile
    return RemoteFile("J03WMX", "information", "DFL_probe.xml", 12345, len(body),
                      hashlib.md5(body).hexdigest())


def present(remote, root, body):
    """Put `body` where the registry says `remote` lives under the data dir `root`, and return that path."""
    path = remote.path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(body)
    return path


def leftovers(root):
    """Every file under root, relative and sorted: what a download left behind."""
    found = []
    for folder, _, names in os.walk(root):
        found += [os.path.relpath(os.path.join(folder, name), root) for name in names]
    return sorted(found)


# -- registry ---------------------------------------------------------------------------------------------------------

def test_registry_shape():
    from pipeline.benchmark import sources
    assert sources.MATCH_IDS == ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"), \
        sources.MATCH_IDS
    assert sources.KINDS == ("information", "positions", "events"), sources.KINDS
    assert list(sources.FILES) == list(sources.MATCH_IDS), "FILES must follow MATCH_IDS: the order is match_index"
    assert dict(sources.COMPETITION) == {match_id: "DFL-COM-000001" if match_id in ("J03WMX", "J03WN1")
                                         else "DFL-COM-000002" for match_id in sources.MATCH_IDS}

    prefixes = {"information": "DFL_02_01_matchinformation_", "events": "DFL_03_02_events_raw_",
                "positions": "DFL_04_03_positions_raw_observed_"}
    file_ids = []
    for match_id in sources.MATCH_IDS:
        assert tuple(sources.FILES[match_id]) == sources.KINDS, (match_id, tuple(sources.FILES[match_id]))
        for kind, remote in sources.FILES[match_id].items():
            assert (remote.match_id, remote.kind) == (match_id, kind), remote
            expected_name = "{}{}_DFL-MAT-{}.xml".format(prefixes[kind], sources.COMPETITION[match_id], match_id)
            assert remote.name == expected_name, (remote.name, expected_name)
            assert len(remote.md5) == 32 and set(remote.md5) <= set("0123456789abcdef"), remote
            assert remote.size > 0
            file_ids.append(remote.file_id)
    assert len(file_ids) == 21 and len(set(file_ids)) == 21, "21 distinct figshare files expected"

    total = sum(remote.size for files in sources.FILES.values() for remote in files.values())
    assert total == 2_627_505_889, total
    assert sorted(sources.EXPECTED_BALL_FRAMES) == sorted(sources.MATCH_IDS)
    assert sum(sources.EXPECTED_BALL_FRAMES.values()) == 1_002_644, sum(sources.EXPECTED_BALL_FRAMES.values())

    # One file written out in full, so a typo in a pinned value cannot hide behind the loop above.
    positions = sources.FILES["J03WMX"]["positions"]
    assert positions.name == "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml"
    assert (positions.file_id, positions.size, positions.md5) == (51643514, 418524833,
                                                                  "e4273a2adf5dbd9d4ec42def2f01905c")
    assert positions.url == "https://ndownloader.figshare.com/files/51643514", positions.url
    assert positions.mirror_url == ("https://huggingface.co/datasets/pysport/idsse-data/resolve/"
                                    "a715a38dfbaf5f58e431727c2b78d174101a703c/" + positions.name), positions.mirror_url


def test_paths_follow_the_data_dir():
    from pipeline.benchmark import sources
    import games
    assert sources.external_dir() == os.path.join(games.DATA_DIR, "external", "dfl")
    assert sources.external_dir("somewhere") == os.path.join("somewhere", "external", "dfl")
    remote = sources.FILES["J03WOH"]["events"]
    assert sources.local_path("J03WOH", "events", "somewhere") == \
        os.path.join("somewhere", "external", "dfl", "J03WOH", remote.name)
    assert remote.path("somewhere") == sources.local_path("J03WOH", "events", "somewhere")
    assert sources.local_path("J03WOH", "events") == remote.path()


def test_select_keeps_registry_order_and_names_unknown_ids():
    from pipeline.benchmark import sources
    chosen = sources.select(["J03WQQ", "J03WMX"])
    assert [remote.match_id for remote in chosen] == ["J03WMX"] * 3 + ["J03WQQ"] * 3, "registry order, 3 kinds each"
    assert len(sources.select()) == 21
    expect_error(ValueError, lambda: sources.select(["J03WXX"]), "J03WXX", "J03WMX")


# -- verify -----------------------------------------------------------------------------------------------------------

def test_verify_reports_each_status():
    from pipeline.benchmark import sources
    remote = probe_file()
    with tempfile.TemporaryDirectory() as root:
        path = remote.path(root)
        assert sources.verify([remote], root) == {path: "missing"}
        present(remote, root, GOOD_BODY[:-1])
        assert sources.verify([remote], root) == {path: "size"}
        present(remote, root, GOOD_BODY[::-1])
        assert sources.verify([remote], root) == {path: "md5"}
        present(remote, root, GOOD_BODY)
        assert sources.verify([remote], root) == {path: "ok"}


# -- download ---------------------------------------------------------------------------------------------------------

def test_download_writes_verified_file():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY})
    with tempfile.TemporaryDirectory() as root:
        paths = sources.download([remote], root, opener=opener, log=quiet)
        assert paths == [remote.path(root)], paths
        with open(paths[0], "rb") as handle:
            assert handle.read() == GOOD_BODY, "the file on disk is not what the server sent"
        assert leftovers(root) == [os.path.relpath(paths[0], root)], leftovers(root)  # no .part left behind
        assert sources.verify([remote], root) == {paths[0]: "ok"}
    assert opener.requests == [(remote.url, "mtah-benchmark/1 (python-urllib)", 60)], opener.requests


def test_download_rejects_bad_md5():
    from pipeline.benchmark import sources
    remote = probe_file()
    wrong_md5 = GOOD_BODY[::-1]
    truncated = GOOD_BODY[:-1000]
    for served in (wrong_md5, truncated):
        # The mirror serves the same bad body: both sources fail, so the error has to say so, and keep nothing.
        opener = FakeOpener({remote.url: served, remote.mirror_url: served})
        with tempfile.TemporaryDirectory() as root:
            message = expect_error(sources.DownloadError,
                                   lambda: sources.download([remote], root, opener=opener, log=quiet),
                                   remote.name, remote.md5, hashlib.md5(served).hexdigest(),
                                   str(remote.size), str(len(served)))
            assert opener.urls == [remote.url, remote.mirror_url], "the mirror is tried once: {}".format(opener.urls)
            assert leftovers(root) == [], "a failed download left {}".format(leftovers(root))
            assert "figshare" in message and "mirror" in message, message


def test_download_skips_valid_file():
    from pipeline.benchmark import sources
    remote = probe_file()
    with tempfile.TemporaryDirectory() as root:
        path = present(remote, root, GOOD_BODY)
        stamp = os.stat(path).st_mtime_ns
        assert sources.download([remote], root, opener=refuse_every_request, log=quiet) == [path]
        assert os.stat(path).st_mtime_ns == stamp, "a file that was already ok was rewritten"


def test_download_replaces_a_damaged_file():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY})
    with tempfile.TemporaryDirectory() as root:
        present(remote, root, GOOD_BODY[::-1])  # right size, wrong content: a previous run that was cut short
        paths = sources.download([remote], root, opener=opener, log=quiet)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
    assert opener.urls == [remote.url]


def test_download_falls_back_to_mirror():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY[::-1], remote.mirror_url: GOOD_BODY})
    with tempfile.TemporaryDirectory() as root:
        paths = sources.download([remote], root, opener=opener, log=quiet)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
        assert leftovers(root) == [os.path.relpath(paths[0], root)], leftovers(root)
    assert opener.urls == [remote.url, remote.mirror_url], opener.urls


def test_download_falls_back_when_the_server_is_down():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: urllib.error.URLError("connection refused"), remote.mirror_url: GOOD_BODY})
    with tempfile.TemporaryDirectory() as root:
        paths = sources.download([remote], root, opener=opener, log=quiet)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
    assert opener.urls == [remote.url, remote.mirror_url], opener.urls


def test_download_reports_a_file_that_is_being_written():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY})
    real_replace = os.replace

    def locked(source, destination):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    with tempfile.TemporaryDirectory() as root:
        os.replace = locked
        try:
            expect_error(sources.DownloadError, lambda: sources.download([remote], root, opener=opener, log=quiet),
                         remote.name, "another download")
        finally:
            os.replace = real_replace
        assert leftovers(root) == [], leftovers(root)
    assert opener.urls == [remote.url], "a local problem must not be retried from the mirror: {}".format(opener.urls)


# -- command line -----------------------------------------------------------------------------------------------------

def test_cli_verify_lists_the_files_and_fails_when_one_is_not_ok():
    from pipeline.benchmark import sources
    with tempfile.TemporaryDirectory() as root:
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = sources.main(["--verify", "--matches", "J03WMX"], data_dir=root)
        text = shown.getvalue()
        assert code == 1, code
        assert text.count("missing") == 3 and "J03WMX" in text and "J03WN1" not in text, text
        assert leftovers(root) == [], "--verify must download nothing and write nothing"


def test_cli_rejects_an_unknown_match():
    from pipeline.benchmark import sources
    with contextlib.redirect_stderr(io.StringIO()) as shown:
        try:
            sources.main(["--matches", "J03WXX"], data_dir=tempfile.gettempdir())
        except SystemExit as stop:
            assert stop.code == 2, stop.code
        else:
            raise AssertionError("an unknown match id was accepted")
    assert "J03WXX" in shown.getvalue()


# -- the downloaded DFL files (--dfl) ---------------------------------------------------------------------------------

def test_dfl_files_verified():
    """All 21 files are on disk with the pinned size and md5. Reads about 2.6 GB."""
    from pipeline.benchmark import sources
    statuses = sources.verify()
    bad = {os.path.basename(path): status for path, status in statuses.items() if status != "ok"}
    assert len(statuses) == 21, len(statuses)
    assert not bad, "{} of 21 DFL files are not ok (e.g. {}); run python pipeline/benchmark/sources.py".format(
        len(bad), sorted(bad.items())[0])
    print("      {} files ok, {:,} bytes under {}".format(
        len(statuses), sum(remote.size for remote in sources.select()), sources.external_dir()))


# -- runner -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB)")
    parser.add_argument("--data", action="store_true", help="also run the checks that read the real match's data")
    args = parser.parse_args(argv)

    names = sorted(name for name in globals() if name.startswith("test_"))
    if not args.dfl:
        names = [name for name in names if not name.startswith("test_dfl_")]
    if not args.data:
        names = [name for name in names if not name.startswith("test_data_")]

    failures = 0
    for name in names:
        started = time.perf_counter()
        try:
            globals()[name]()
        except Exception as error:  # report every failure; one broken check must not hide the rest
            failures += 1
            print("FAIL  {}  {}: {}".format(name, type(error).__name__, error))
        else:
            print("PASS  {}  ({:.1f} s)".format(name, time.perf_counter() - started))

    skipped = [flag for flag, on in (("--dfl", args.dfl), ("--data", args.data)) if not on]
    print("\n{} of {} checks passed{}".format(
        len(names) - failures, len(names), "  (add {} for the rest)".format(" / ".join(skipped)) if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
