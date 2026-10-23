"""Checks of the DFL registry and the download (pipeline/benchmark/sources.py).

The download checks run the real streaming, hashing and rename code against a fake opener that serves small bodies
from memory, so no network is touched and nothing outside a temporary folder is written.
"""

import contextlib
import hashlib
import io
import os
import tempfile
import urllib.error

from pipeline.selftest import expect_error, quiet

# Params
# 2.56 MB: three 1 MiB reads, the last one short, so the streaming loop really loops.
GOOD_BODY = bytes(range(256)) * 10_000


# -- helpers ----------------------------------------------------------------------------------------------------------

class RecordingBody(io.BytesIO):
    """A response body that notes, at every read, how much was asked for and what the disk looked like."""

    def __init__(self, body, final_path):
        super().__init__(body)
        self.final_path = final_path
        self.read_sizes = []
        self.final_existed = []
        self.part_existed = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        self.final_existed.append(os.path.exists(self.final_path))
        self.part_existed.append(os.path.exists(self.final_path + ".part"))
        return super().read(size)


class FakeOpener:
    """Stands in for urllib.request.urlopen: serves a body per URL and remembers every request it got.

    A value in `bodies` that is an exception is raised instead of served, which is how a dead server looks. Given
    `final_path`, each served body (kept in `responses`) records what the disk looked like while it was read.
    """

    def __init__(self, bodies, final_path=None):
        self.bodies = bodies
        self.final_path = final_path
        self.requests = []
        self.responses = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.get_header("User-agent"), timeout))
        body = self.bodies[request.full_url]
        if isinstance(body, Exception):
            raise body
        response = io.BytesIO(body) if self.final_path is None else RecordingBody(body, self.final_path)
        self.responses.append(response)
        return response

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


@contextlib.contextmanager
def replace_raises(error):
    """os.replace fails with `error` inside the block, as it does when another process holds the target."""
    real_replace = os.replace

    def failing(source, destination):
        raise error

    os.replace = failing
    try:
        yield
    finally:
        os.replace = real_replace


def collecting_log():
    """(log, lines): a log function that keeps what it was given."""
    lines = []
    return lines.append, lines


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
    from_generator = sources.select(match_id for match_id in ("J03WQQ", "J03WMX"))
    assert from_generator == chosen, "a generator of ids must select the same files as a list ({} files)".format(
        len(from_generator))
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
    with tempfile.TemporaryDirectory() as root:
        opener = FakeOpener({remote.url: GOOD_BODY}, final_path=remote.path(root))
        present(remote, root, GOOD_BODY[::-1])  # right size, wrong content: a previous run that was cut short
        paths = sources.download([remote], root, opener=opener, log=quiet)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
        assert not any(opener.responses[0].final_existed), "the damaged file must be gone before the new bytes flow"
    assert opener.urls == [remote.url]


def test_download_leaves_no_damaged_file_behind():
    from pipeline.benchmark import sources
    remote = probe_file()
    down = urllib.error.URLError("connection refused")
    opener = FakeOpener({remote.url: down, remote.mirror_url: down})
    for damage in (GOOD_BODY[:-1], GOOD_BODY[::-1]):  # wrong size, then wrong content
        with tempfile.TemporaryDirectory() as root:
            present(remote, root, damage)
            expect_error(sources.DownloadError, lambda: sources.download([remote], root, opener=opener, log=quiet),
                         remote.name, "nothing was kept")
            assert leftovers(root) == [], "a final file is complete and right or absent: {}".format(leftovers(root))
            assert sources.verify([remote], root) == {remote.path(root): "missing"}


def test_download_streams_in_mebibyte_chunks_into_a_part_file():
    from pipeline.benchmark import sources
    remote = probe_file()
    with tempfile.TemporaryDirectory() as root:
        opener = FakeOpener({remote.url: GOOD_BODY}, final_path=remote.path(root))
        sources.download([remote], root, opener=opener, log=quiet)
    body = opener.responses[0]
    assert set(body.read_sizes) == {1 << 20}, "every read must ask for 1 MiB: {}".format(body.read_sizes)
    assert len(body.read_sizes) == 4, "3 chunks of data and the empty read that ends the body: {}".format(
        body.read_sizes)
    assert not any(body.final_existed), "the final file must not exist while the bytes are still arriving"
    assert all(body.part_existed), "the bytes must flow into <name>.part"


def test_download_falls_back_to_mirror():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY[::-1], remote.mirror_url: GOOD_BODY})
    log, lines = collecting_log()
    with tempfile.TemporaryDirectory() as root:
        paths = sources.download([remote], root, opener=opener, log=log)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
        assert leftovers(root) == [os.path.relpath(paths[0], root)], leftovers(root)
    assert opener.urls == [remote.url, remote.mirror_url], opener.urls
    # One line to say the download started, one for the fallback with its reason, one to say it finished: no
    # progress lines per chunk.
    assert [line.split()[0] for line in lines] == ["downloading", "retrying", "downloaded"], lines
    assert "J03WMX/information" in lines[1] and "figshare failed" in lines[1] and "md5" in lines[1], lines[1]


def test_download_falls_back_when_the_server_is_down():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: urllib.error.URLError("connection refused"), remote.mirror_url: GOOD_BODY})
    log, lines = collecting_log()
    with tempfile.TemporaryDirectory() as root:
        paths = sources.download([remote], root, opener=opener, log=log)
        assert sources.verify([remote], root) == {paths[0]: "ok"}
    assert opener.urls == [remote.url, remote.mirror_url], opener.urls
    assert "connection refused" in lines[1], "the log must give the reason figshare failed: {}".format(lines)


def test_download_reports_a_file_that_is_being_written():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY})
    locked = PermissionError(13, "The process cannot access the file because it is being used by another process")
    with tempfile.TemporaryDirectory() as root:
        with replace_raises(locked):
            expect_error(sources.DownloadError, lambda: sources.download([remote], root, opener=opener, log=quiet),
                         remote.name, "another download")
        assert leftovers(root) == [], leftovers(root)
    assert opener.urls == [remote.url], "a local problem must not be retried from the mirror: {}".format(opener.urls)


def test_download_reports_any_failure_to_move_the_file_into_place():
    from pipeline.benchmark import sources
    remote = probe_file()
    opener = FakeOpener({remote.url: GOOD_BODY})
    full = OSError(28, "No space left on device")
    with tempfile.TemporaryDirectory() as root:
        with replace_raises(full):
            message = expect_error(sources.DownloadError,
                                   lambda: sources.download([remote], root, opener=opener, log=quiet),
                                   remote.name, "No space left", "disk space")
        assert "another download" not in message, "only a PermissionError points at another download: " + message
        assert leftovers(root) == [], leftovers(root)
    assert opener.urls == [remote.url], opener.urls


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


def test_cli_takes_the_data_dir_as_a_flag():
    """--data-dir names the bundle as run.py's and fidelity.py's flag does: --verify looks there and nowhere else.

    A file of the wrong size put where the registry keeps J03WMX's match information under the folder `root` is found
    there ("size") and the other two are missing; the flag is in the parser's help beside MTAH_DATA_DIR, which is what
    the default follows (games.DATA_DIR); and a data dir in the argument list wins over main's own `data_dir`."""
    from pipeline.benchmark import dfl, sources
    remote = sources.FILES["J03WMX"]["information"]
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
        present(remote, root, GOOD_BODY)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = sources.main(["--verify", "--matches", "J03WMX", "--data-dir", root])
        text = shown.getvalue()
        assert code == 1 and text.count("missing") == 2 and text.count("size") == 1, text
        assert sources.external_dir(root) in text, "the summary says where it looked: " + text
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = sources.main(["--verify", "--matches", "J03WMX", "--data-dir", other], data_dir=root)
        assert code == 1 and shown.getvalue().count("missing") == 3, "the flag wins over main's own data_dir"
        assert leftovers(other) == [], "--verify must download nothing and write nothing"
        helped = io.StringIO()
        with contextlib.redirect_stdout(helped):
            try:
                sources.main(["--help"])
            except SystemExit as stop:
                assert stop.code == 0, stop.code
        assert "--data-dir" in helped.getvalue() and "MTAH_DATA_DIR" in helped.getvalue(), helped.getvalue()
    assert "MTAH_DATA_DIR" in dfl.FETCH_ADVICE and "--data-dir" in dfl.FETCH_ADVICE, dfl.FETCH_ADVICE


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
