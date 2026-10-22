"""Self-checks for the P01 benchmark: the DFL registry, download, match information and positions now; the rest later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB),
                                                    #   the 7 match information files parsed, and one positions file
    python pipeline/benchmark/selftest.py --data    # + checks that read the real match's data (aggregates only)

The runner is pipeline/selftest.py's: every function named test_* is a check, it prints PASS or FAIL for each, and
the exit code is 1 if anything failed. Checks named test_dfl_* need the downloaded DFL files and run only with --dfl;
test_data_* need the real match and run only with --data. Nothing from the real match is ever printed -- counts,
shares and scores, never a row, a coordinate or a label sequence (CLAUDE.md, hard rules). The DFL files are public
(CC BY 4.0), so their aggregates may be printed freely.

The download checks run the real streaming, hashing and rename code against a fake opener that serves small bodies
from memory, so no network is touched and nothing outside a temporary folder is written. The parser checks read a
synthetic match written by write_synthetic_dfl in the real file format, so they need no download either.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
from xml.sax.saxutils import quoteattr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so
    # that `import pipeline` resolves from any working directory (as in pipeline/selftest.py).
    sys.path[0] = ROOT

# Importing it also puts evaluation/ on sys.path.
from pipeline.selftest import expect_error, quiet, write_text  # noqa: E402

# Params
# 2.56 MB: three 1 MiB reads, the last one short, so the streaming loop really loops.
GOOD_BODY = bytes(range(256)) * 10_000
# The home and guest team of each DFL match, as Bassek, Rein, Weber & Memmert (2025), Sci Data 12:195, Table 1 lists
# them. Typed from the paper and never taken from the files' General@HomeTeamName / GuestTeamName: the parser reads
# the files, so a table "fixed" from them would make this check compare the files with themselves.
DFL_TEAMS = {"J03WMX": ("1. FC Köln", "FC Bayern München"), "J03WN1": ("VfL Bochum 1848", "Bayer 04 Leverkusen"),
             "J03WOH": ("Fortuna Düsseldorf", "SSV Jahn Regensburg"),
             "J03WOY": ("Fortuna Düsseldorf", "F.C. Hansa Rostock"),
             "J03WPY": ("Fortuna Düsseldorf", "1. FC Nürnberg"), "J03WQQ": ("Fortuna Düsseldorf", "FC St. Pauli"),
             "J03WR9": ("Fortuna Düsseldorf", "1. FC Kaiserslautern")}
# Squads of the synthetic match: (PersonId, ShirtNumber, FirstName, LastName, Shortname), None leaves the attribute
# out. Each is listed in the file in an order that is not PersonId order, so a parser that keeps file order fails.
SYNTHETIC_HOME = ("DFL-CLU-SYNHOM", "Synthetic Home FC",
                  (("DFL-OBJ-SYNH03", 4, "Hal", "Third", "H. Third"),
                   ("DFL-OBJ-SYNH01", 1, "Hanna", "Keeper", "H. Keeper"),
                   ("DFL-OBJ-SYNH02", 10, None, None, None)))  # no name at all: the PersonId stands in
SYNTHETIC_GUEST = ("DFL-CLU-SYNAWY", "Synthetic Away FC",
                   (("DFL-OBJ-SYNA02", 23, None, None, "B. Short"),  # only a short name: that is used
                    ("DFL-OBJ-SYNA03", 9, "Zoë Maria", "Müller", "Z. Müller"),  # non-ASCII, several given names
                    ("DFL-OBJ-SYNA01", 1, "Ann", "Goalie", "A. Goalie")))
# Positions of the synthetic match. The first half is N 10000-10009 and the second N 100000-100007, so the timeline
# has 10 + 25 + 8 = 43 rows and the second half starts on row 35. A coordinate is given as (jx, jy), a place on a grid
# with 8 intervals across and down the 104.5 x 67.5 m pitch: its normalised value is exactly jx / 8 and jy / 8 (x runs
# with X, y against Y), so a check can compare with == and no rounding hides an off-by-one or a missing flip. A None
# leaves that attribute out of the file. The tuples are in file order.
PERSON_A, PERSON_B, PERSON_C = "DFL-OBJ-SYNH01", "DFL-OBJ-SYNH02", "DFL-OBJ-SYNA01"
PERSON_REFEREE, PERSON_UNLISTED, PERSON_BALL = "DFL-OBJ-SYNR01", "DFL-OBJ-SYNZ01", "DFL-OBJ-SYNBAL"
SYNTHETIC_TIME = "2024-02-28T08:15:28.955+00:00"  # every frame says the same: the parser must not read T
# {(PersonId, GameSection): ((N, jx, jy, S in km/h as text), ...)}, in the order the file lists the FrameSets.
SYNTHETIC_PLAYER_FRAMES = {
    # A plays both halves, and her second half comes first in the file. Her first half has a frame before the half
    # (N 9999) and one after it (N 10010, which also lacks S): both are dropped, and neither may land on row -1 or on
    # a separator row.
    (PERSON_A, "secondHalf"): ((100000, 1, 1, "3.6"), (100001, 2, 3, "4.5"), (100002, 3, 5, None),
                               (100003, 4, 7, "6.3"), (100004, 5, 8, "7.2"), (100005, 6, 6, "8.1"),
                               (100006, 7, 4, "9.0"), (100007, 8, 2, "9.9")),
    (PERSON_A, "firstHalf"): ((9999, 0, 0, "1.0"),
                              (10000, 0, 2, "10.8"), (10001, 1, 3, "11.7"), (10002, 2, 2, "12.6"),
                              (10003, 3, 1, "13.5"), (10004, 4, 0, "14.4"), (10005, 5, 6, "15.3"),
                              (10006, None, 7, "16.2"), (10007, 7, 8, "17.1"), (10008, 8, 5, "18.0"),
                              (10009, 8, 1, "18.9"), (10010, 3, 3, None)),
    # B plays the first half only; his frames N 10004 and N 10003 are written in that order.
    (PERSON_B, "firstHalf"): ((10000, 8, 6, "5.4"), (10001, 7, 5, "6.3"), (10002, 6, 4, "7.2"),
                              (10004, 5, None, "9.0"), (10003, 5, 3, "8.1"), (10005, 4, 2, "9.9"),
                              (10006, 3, 1, "10.8"), (10007, 2, 0, None), (10008, 1, 7, "12.6"),
                              (10009, 0, 8, "13.5")),
    # C, of the away team, comes on at N 100003 (row 38).
    (PERSON_C, "secondHalf"): ((100003, 4, 4, "21.6"), (100004, 5, 5, "22.5"), (100005, None, 6, "23.4"),
                               (100006, 7, 7, "24.3"), (100007, 8, 8, "25.2")),
}
# {GameSection: ((N, jx, jy, Z in metres, S, BallPossession, BallStatus), ...)}; the text is what the file says.
# N 100004 has no ball frame, N 10007 is a dead ball that was last the away team's, N 100007 one that was the home
# team's, and each code is missing on a frame of its own.
SYNTHETIC_BALL_FRAMES = {
    "firstHalf": ((10000, 4, 4, "0.00", "7.2", "1", "1"), (10001, 5, 4, "0.25", "8.1", "1", "1"),
                  (10002, 6, 3, "0.50", "9.0", "1", "1"), (10003, 6, 2, "0.75", None, "1", "1"),
                  (10004, 5, 1, "1.00", "10.8", "1", "1"), (10005, 4, 1, "1.25", "11.7", "2", "1"),
                  (10006, 3, 2, "1.50", "12.6", "2", "1"), (10007, 3, 3, "1.75", "13.5", "2", "0"),
                  (10008, 2, 4, None, "14.4", "2", "1"), (10009, 2, 5, "2.25", "15.3", "2", None)),
    "secondHalf": ((100000, 1, 5, "0.00", "5.4", "2", "1"), (100001, 1, 6, None, "6.3", "2", "1"),
                   (100002, 2, 7, "0.50", "7.2", "2", "1"), (100003, 3, 7, "0.75", "8.1", "1", "1"),
                   (100005, 5, 6, "1.25", "9.9", "1", None), (100006, 6, 5, "1.50", "10.8", None, "1"),
                   (100007, 7, 4, "1.75", "11.7", "1", "0")),
}
# What parse_positions must report on the synthetic match: 17 ball frames, 33 stored player frames (A 10 + 8, B 10,
# C 5) and 2 dropped; X is missing on 2 stored frames, Y on 1, S on 3 (the dropped frame's missing S is not counted).
SYNTHETIC_REPORT = {
    "ball_frames": 17, "rows_without_ball": 1, "player_frames": 33, "player_frames_outside_halves": 2,
    "skipped_framesets": {"DFL-CLU-SYNAWY": 1, "referee": 1},
    "missing": {"X": 2, "Y": 1, "S": 3, "Z": 2, "BallPossession": 1, "BallStatus": 2}}


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


# -- the synthetic DFL match ------------------------------------------------------------------------------------------

def xml_attributes(values):
    """name="value" for every pair whose value is not None, escaped, in the order given."""
    return " ".join("{}={}".format(name, quoteattr(str(value))) for name, value in values.items() if value is not None)


def synthetic_team(role, team_id, name, squad):
    """One <Team> as DFL writes it: the real attributes and children, with a trainer who is not a player."""
    players = "\n".join(
        "          <Player {} />".format(xml_attributes({
            "PersonId": person_id, "ShirtNumber": shirt, "FirstName": first, "LastName": last, "Shortname": short,
            "Starting": "true", "TeamLeader": "false"}))
        for person_id, shirt, first, last, short in squad)
    return """      <Team {}>
        <Players>
{}
        </Players>
        <TrainerStaff>
          <Trainer PersonId="DFL-OBJ-SYNT{}" Role="headcoach" FirstName="Tia" LastName="Coach" Shortname="T. Coach" />
        </TrainerStaff>
        <OfficialStaff>
        </OfficialStaff>
      </Team>""".format(xml_attributes({
        "TeamId": team_id, "TeamName": name, "Role": role, "PlayerShirtType": "Heimtrikot",
        "PlayerShirtMainColor": "#F4F5F0", "LineUp": "4-2-3-1"}), players, role[0].upper() + "00")


def grid_x(j):
    """X in metres, as text, that normalises to exactly j / 8 on the synthetic 104.5 m pitch (None stays None)."""
    return None if j is None else repr((j - 4) * 13.0625)


def grid_y(j):
    """Y in metres, as text, that normalises to exactly j / 8 on the synthetic 67.5 m pitch: Y runs against y."""
    return None if j is None else repr(33.75 - 8.4375 * j)


def synthetic_frameset(section, team_id, person_id, frames, match_id):
    """One <FrameSet> as DFL writes it, one <Frame> per line; `frames` are the rendered <Frame> lines."""
    return "<FrameSet {}>\n{}\n</FrameSet>".format(xml_attributes({
        "GameSection": section, "MatchId": "DFL-MAT-" + match_id, "TeamId": team_id, "PersonId": person_id}),
        "\n".join(frames))


def player_frame(n, jx, jy, speed):
    return "<Frame {} />".format(xml_attributes({
        "N": n, "T": SYNTHETIC_TIME, "X": grid_x(jx), "Y": grid_y(jy), "D": "0.00", "S": speed, "A": "0.00",
        "M": "1"}))


def ball_frame(n, jx, jy, z, speed, possession, status):
    return "<Frame {} />".format(xml_attributes({
        "N": n, "T": SYNTHETIC_TIME, "X": grid_x(jx), "Y": grid_y(jy), "Z": z, "D": "0.00", "S": speed, "A": "0.00",
        "M": "1", "BallPossession": possession, "BallStatus": status}))


def synthetic_positions_text(match_id):
    """The positions file of the synthetic match, with the layout that the parser must not depend on.

    The referee and a player who is not in the squad list come first (the real file starts with the referees), then
    the players, one of them with his second half before his first, one with two frames out of order, and the ball
    last, as in the real file. Every FrameSet carries the match id, as the real ones do.
    """
    home, guest = SYNTHETIC_HOME[0], SYNTHETIC_GUEST[0]
    club_of = {PERSON_A: home, PERSON_B: home, PERSON_C: guest}
    sets = [
        synthetic_frameset("firstHalf", "referee", PERSON_REFEREE,
                           [player_frame(10000 + k, 4, 4, "2.0") for k in range(3)], match_id),
        synthetic_frameset("firstHalf", guest, PERSON_UNLISTED,
                           [player_frame(10000 + k, 1, 1, "2.0") for k in range(3)], match_id)]
    sets += [synthetic_frameset(section, club_of[person], person, [player_frame(*frame) for frame in frames], match_id)
             for (person, section), frames in SYNTHETIC_PLAYER_FRAMES.items()]
    sets += [synthetic_frameset(section, "BALL", PERSON_BALL, [ball_frame(*frame) for frame in frames], match_id)
             for section, frames in SYNTHETIC_BALL_FRAMES.items()]
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest RequestId="00000000-0000-4000-8000-000000000001" MessageTime="{time}" TransmissionComplete="true" \
TransmissionSuspended="false">
<Positions EventTime="{time}">
<MetaData MatchId="DFL-MAT-{match_id}" Type="pitch-size">
<PitchSize X="104.50" Y="67.50" />
</MetaData>
{sets}
</Positions>
</PutDataRequest>
""".format(time=SYNTHETIC_TIME, match_id=match_id, sets="\n".join(sets))


def write_synthetic_dfl(folder, match_id="SYN001"):
    """Write a small DFL match in the real file format into `folder`; return {kind: path}, kinds as in sources.FILES.

    The match information and the positions (not the events, which no parser reads yet). The element and attribute
    names are the real ones (copied from J03WMX's files), the values are made up and chosen so that every parser rule
    shows. Information: the guest team is listed first, so Role decides and not the order; each squad is out of
    PersonId order; names exercise the fallbacks and non-ASCII text; a trainer and a referee carry PersonIds that
    are not players; the pitch is not the usual 105 x 68; the two half durations differ. Positions: see
    synthetic_positions_text and the SYNTHETIC_* tables.
    """
    home_id, home_name, home_squad = SYNTHETIC_HOME
    guest_id, guest_name, guest_squad = SYNTHETIC_GUEST
    text = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest RequestId="00000000-0000-4000-8000-000000000000" MessageTime="2024-02-28T08:15:28.955+00:00"
    TransmissionComplete="true" TransmissionSuspended="false">
  <MatchInformation>
    <General {general} />
    <Environment Country="Deutschland" StadiumName="Synthetic Arena" NeutralVenue="false" Roof="open"
        PitchX="104.50" PitchY="67.50" />
    <Teams>
{guest}
{home}
    </Teams>
    <Referees>
      <Referee PersonId="DFL-OBJ-SYNR01" Role="referee" FirstName="Rob" LastName="Whistle" Shortname="R. Whistle" />
    </Referees>
    <OtherGameInformation TotalTimeFirstHalf="2700040" TotalTimeSecondHalf="2820120"
        PlayingTimeFirstHalf="1888960" PlayingTimeSecondHalf="1525720" />
  </MatchInformation>
</PutDataRequest>
""".format(
        general=xml_attributes({
            "TypeOfSport": "Fußball", "CompetitionName": "Synthetic League", "CompetitionId": "DFL-COM-SYNTH",
            "MatchId": "DFL-MAT-" + match_id, "MatchTitle": "{}:{}".format(home_name, guest_name),
            "HomeTeamName": home_name, "HomeTeamId": home_id, "GuestTeamName": guest_name, "GuestTeamId": guest_id}),
        guest=synthetic_team("guest", guest_id, guest_name, guest_squad),
        home=synthetic_team("home", home_id, home_name, home_squad))
    name = "DFL_02_01_matchinformation_DFL-COM-SYNTH_DFL-MAT-{}.xml".format(match_id)
    positions = "DFL_04_03_positions_raw_observed_DFL-COM-SYNTH_DFL-MAT-{}.xml".format(match_id)
    return {"information": write_text(folder, name, text),
            "positions": write_text(folder, positions, synthetic_positions_text(match_id))}


# -- the match information parser -------------------------------------------------------------------------------------

def test_match_information_synthetic():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info = dfl.parse_match_information(write_synthetic_dfl(folder)["information"])
    assert info.match_id == "SYN001", "the short id, without the DFL-MAT- prefix: " + info.match_id
    assert (info.home_id, info.home_name) == ("DFL-CLU-SYNHOM", "Synthetic Home FC"), info
    # The guest team is listed first in the file, and is the away team: Role decides, not the order.
    assert (info.away_id, info.away_name) == ("DFL-CLU-SYNAWY", "Synthetic Away FC"), info
    assert info.pitch == (104.5, 67.5) and all(type(side) is float for side in info.pitch), info.pitch
    assert info.total_time_raw == {1: 2700040.0, 2: 2820120.0}, info.total_time_raw
    assert all(type(key) is int and type(value) is float for key, value in info.total_time_raw.items())
    # Home first, each team in PersonId order; the trainer and the referee are not players; a name is FirstName and
    # LastName, else Shortname, else the PersonId.
    assert info.players == (
        dfl.Player("DFL-OBJ-SYNH01", 0, 1, "Hanna Keeper"),
        dfl.Player("DFL-OBJ-SYNH02", 0, 10, "DFL-OBJ-SYNH02"),
        dfl.Player("DFL-OBJ-SYNH03", 0, 4, "Hal Third"),
        dfl.Player("DFL-OBJ-SYNA01", 1, 1, "Ann Goalie"),
        dfl.Player("DFL-OBJ-SYNA02", 1, 23, "B. Short"),
        dfl.Player("DFL-OBJ-SYNA03", 1, 9, "Zoë Maria Müller"),
    ), info.players
    assert type(info.players) is tuple, type(info.players)
    assert all(type(player.team) is int and type(player.shirt) is int for player in info.players), info.players
    assert info == dfl.MatchInfo(
        match_id="SYN001", home_id="DFL-CLU-SYNHOM", away_id="DFL-CLU-SYNAWY", home_name="Synthetic Home FC",
        away_name="Synthetic Away FC", pitch=(104.5, 67.5), players=info.players,
        total_time_raw={1: 2700040.0, 2: 2820120.0}), "MatchInfo must compare by value"
    try:
        info.pitch = (1.0, 1.0)
    except AttributeError:
        pass
    else:
        raise AssertionError("MatchInfo must be frozen")


def test_match_information_follows_the_match_id():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = write_synthetic_dfl(folder, match_id="SYN002")["information"]
        assert path.endswith("_DFL-MAT-SYN002.xml"), path
        assert dfl.parse_match_information(path).match_id == "SYN002"


def damaged(path, damage):
    """Rewrite the file at `path` as damage(its text), which must change it; return the path."""
    with open(path, encoding="utf-8", newline="") as handle:
        intact = handle.read()
    text = damage(intact)
    assert text != intact, "the damage left the synthetic file as it was"
    return write_text(os.path.dirname(path), os.path.basename(path), text)


def test_match_information_rejects_bad_role():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = damaged(write_synthetic_dfl(folder)["information"],
                       lambda text: text.replace('Role="guest"', 'Role="visitor"'))
        expect_error(ValueError, lambda: dfl.parse_match_information(path), os.path.basename(path), "Role", "visitor",
                     '"home"', '"guest"')


def test_match_information_rejects_damaged_files():
    from pipeline.benchmark import dfl

    def without_teams(text):
        return re.sub(r"<Teams>.*</Teams>", "<Teams></Teams>", text, flags=re.DOTALL)

    cases = [  # label, how to damage the file, what the message must name besides the file
        ("two home teams", lambda text: text.replace('Role="guest"', 'Role="home"'), ['Role="home"', "two <Team>"]),
        ("no teams", without_teams, ['Role="home"', 'Role="guest"']),
        # The guest team is the first one in the file, so count=1 empties its squad.
        ("a team with no players",
         lambda text: re.sub(r"<Players>.*?</Players>", "<Players></Players>", text, count=1, flags=re.DOTALL),
         ["Synthetic Away FC", 'Role="guest"', "lists no <Player>", "--verify"]),
        ("two <Environment> elements", lambda text: re.sub(r"<Environment[^>]*/>", lambda match: match.group(0) * 2,
                                                           text),
         ["<Environment>", "2 <Environment> elements"]),
        ("no pitch length", lambda text: text.replace(' PitchX="104.50"', ""), ["PitchX", "Environment", "--verify"]),
        ("pitch that is no number", lambda text: text.replace('PitchY="67.50"', 'PitchY="wide"'),
         ["PitchY", "wide", "--verify"]),
        ("negative pitch", lambda text: text.replace('PitchX="104.50"', 'PitchX="-105"'), ["PitchX", "-105"]),
        ("shirt that is no number", lambda text: text.replace('ShirtNumber="9"', 'ShirtNumber="nine"'),
         ["ShirtNumber", "nine", "DFL-OBJ-SYNA03", "--verify"]),
        ("a shirt that is missing", lambda text: text.replace(' ShirtNumber="9"', ""),
         ["no ShirtNumber attribute", "DFL-OBJ-SYNA03", "--verify"]),
        ("a player in both teams", lambda text: text.replace("DFL-OBJ-SYNH01", "DFL-OBJ-SYNA01"),
         ["DFL-OBJ-SYNA01", "twice"]),
        ("no second half duration", lambda text: text.replace(' TotalTimeSecondHalf="2820120"', ""),
         ["TotalTimeSecondHalf", "--verify"]),
        ("match id without its prefix", lambda text: text.replace("DFL-MAT-SYN001", "SYN001"),
         ["MatchId", "DFL-MAT-"]),
        ("General naming the other home team",
         lambda text: text.replace('HomeTeamId="DFL-CLU-SYNHOM"', 'HomeTeamId="DFL-CLU-SYNAWY"'),
         ["HomeTeamId", "Role"]),
        ("a positions file instead", lambda text: "<PutDataRequest><Positions /></PutDataRequest>",
         ["MatchInformation", "Positions", "match information file"]),
        ("a file cut short", lambda text: text[:len(text) // 2], ["well-formed", "--verify"]),
    ]
    with tempfile.TemporaryDirectory() as folder:
        written = write_synthetic_dfl(folder)["information"]
        with open(written, encoding="utf-8", newline="") as handle:
            intact = handle.read()
        for label, damage, fragments in cases:
            write_text(folder, os.path.basename(written), intact)
            path = damaged(written, damage)
            try:
                expect_error(ValueError, lambda: dfl.parse_match_information(path), os.path.basename(path),
                             *fragments)
            except AssertionError as error:
                raise AssertionError("{}: {}".format(label, error)) from error


def test_match_information_names_a_missing_file():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "DFL_02_01_matchinformation_DFL-COM-SYNTH_DFL-MAT-SYN001.xml")
        expect_error(FileNotFoundError, lambda: dfl.parse_match_information(path), os.path.basename(path),
                     "pipeline/benchmark/sources.py")


# -- the positions parser ---------------------------------------------------------------------------------------------

MATCH_ARRAYS = ("frame_number", "players_xy", "players_speed", "player_team", "ball_xy", "ball_z", "ball_speed",
                "ball_status", "ball_possession")


def positions_of(folder, damage=None):
    """(info, path) of a fresh synthetic match in `folder`; `damage`, if given, rewrites the positions file first."""
    from pipeline.benchmark import dfl
    paths = write_synthetic_dfl(folder)
    info = dfl.parse_match_information(paths["information"])
    return info, (paths["positions"] if damage is None else damaged(paths["positions"], damage))


def leaves(value):
    """Every number at the bottom of nested dicts."""
    return [leaf for item in value.values() for leaf in leaves(item)] if isinstance(value, dict) else [value]


def synthetic_row(n):
    """The timeline row of frame number N in the synthetic match, or None outside both halves."""
    if 10000 <= n <= 10009:
        return n - 10000
    if 100000 <= n <= 100007:
        return 35 + n - 100000  # 10 rows, 25 of separator, then the second half
    return None


def frameset_block(text, section, person):
    """The text of the one <FrameSet> of this person and half."""
    pattern = r'<FrameSet GameSection="{}" [^>]*PersonId="{}">.*?</FrameSet>'.format(section, person)
    found = re.findall(pattern, text, flags=re.DOTALL)
    assert len(found) == 1, "{} FrameSets for {} in {}".format(len(found), person, section)
    return found[0]


def edit_frameset(section, person, change):
    """A damage for `damaged`: the FrameSet of this person and half becomes change(its text)."""
    def damage(text):
        block = frameset_block(text, section, person)
        return text.replace(block, change(block))
    return damage


def edit_frame(n, change):
    """A change for edit_frameset: the line of the frame with this N becomes change(line)."""
    def edit(block):
        lines = block.split("\n")
        hits = [i for i, line in enumerate(lines) if '<Frame N="{}" '.format(n) in line]
        assert len(hits) == 1, "{} frames with N={}".format(len(hits), n)
        lines[hits[0]] = change(lines[hits[0]])
        return "\n".join(lines)
    return edit


def edit_ball_frame(n, old, new):
    """A damage: in the ball's frame N, the text `old` becomes `new`."""
    return edit_frameset("firstHalf" if n < 100000 else "secondHalf", PERSON_BALL,
                         edit_frame(n, lambda line: line.replace(old, new)))


def both(first, second):
    """A damage that applies two damages, one after the other."""
    return lambda text: second(first(text))


def expect_positions_errors(cases, error=ValueError):
    """Each case (label, damage, fragments): parse the synthetic match damaged that way; it must raise `error`."""
    from pipeline.benchmark import dfl
    for label, damage, fragments in cases:
        with tempfile.TemporaryDirectory() as folder:
            info, path = positions_of(folder, damage)
            try:
                expect_error(error, lambda: dfl.parse_positions(path, info), os.path.basename(path), *fragments)
            except AssertionError as problem:
                raise AssertionError("{}: {}".format(label, problem)) from problem


def test_positions_synthetic_exact():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
    assert dfl.GAP_ROWS == 25 and dfl.SECTIONS == {"firstHalf": 1, "secondHalf": 2}, (dfl.GAP_ROWS, dfl.SECTIONS)
    assert (match.match_id, match.pitch, match.fps) == ("SYN001", (104.5, 67.5), 25), match
    assert (match.n_frames, match.n_players) == (43, 6), (match.n_frames, match.n_players)
    assert match.graded == ((1, 0, 10), (2, 35, 43)), match.graded
    assert match.kickoffs == {1: 0, 2: 35}, match.kickoffs
    assert all(type(value) is int for block in match.graded for value in block), match.graded
    assert match.frame_number[10] == -1 and match.frame_number[35] == 100000, match.frame_number[[10, 35]]
    assert match.ball_status[39] == -1, "N 100004 has no ball frame: its codes are -1"
    assert list(match.player_team) == [0, 0, 0, 1, 1, 1], "home first, then away, as the squads list them"
    assert [player.person_id for player in info.players][::3] == [PERSON_A, PERSON_C], info.players

    shapes = {"frame_number": ((43,), np.int64), "players_xy": ((43, 6, 2), np.float64),
              "players_speed": ((43, 6), np.float64), "player_team": ((6,), np.int8), "ball_xy": ((43, 2), np.float64),
              "ball_z": ((43,), np.float64), "ball_speed": ((43,), np.float64), "ball_status": ((43,), np.int8),
              "ball_possession": ((43,), np.int8)}
    assert sorted(shapes) == sorted(MATCH_ARRAYS)
    for name, (shape, dtype) in shapes.items():
        array = getattr(match, name)
        assert (array.shape, array.dtype) == (shape, np.dtype(dtype)), (name, array.shape, array.dtype)
        assert not array.flags.writeable, "{} must be read-only: a cached match returns it that way".format(name)

    # The rows of C: NaN before he comes on at N 100003 (rows 35-37), then set.
    columns = {PERSON_A: 0, PERSON_B: 1, PERSON_C: 3}
    assert np.isnan(match.players_xy[35:38, 3]).all() and np.isnan(match.players_speed[35:38, 3]).all()
    assert tuple(match.players_xy[38, 3]) == (0.5, 0.5), match.players_xy[38, 3]
    assert not np.isnan(match.players_speed[38, 3])
    # Normalised exactly, and y flipped: A's frame N 10002 is X = -26.125, Y = 16.875 and gives (0.25, 0.25); her
    # frame N 10009 is X = 52.25, Y = 25.3125 and gives (1.0, 0.125), where an unflipped y would be 0.875.
    assert (grid_x(2), grid_y(2)) == ("-26.125", "16.875"), (grid_x(2), grid_y(2))
    assert tuple(match.players_xy[2, 0]) == (0.25, 0.25) and tuple(match.players_xy[9, 0]) == (1.0, 0.125), \
        (match.players_xy[2, 0], match.players_xy[9, 0])
    # Speed is S / 3.6 (km/h to m/s), NaN where S is missing; the same for the ball.
    assert match.ball_speed[0] == 7.2 / 3.6 and np.isnan(match.ball_speed[3]), match.ball_speed[:4]
    assert match.players_speed[0, 0] == 10.8 / 3.6 and np.isnan(match.players_speed[7, 1]), match.players_speed[:8, :2]
    assert match.ball_z[4] == 1.0 and np.isnan(match.ball_z[8]), "Z is metres as written, NaN where missing"
    assert (match.ball_status[7], match.ball_possession[7]) == (0, 1), "a dead frame, last the away team's"
    assert (match.ball_status[42], match.ball_possession[42]) == (0, 0), "a dead frame, last the home team's"
    assert (match.ball_status[9], match.ball_possession[9]) == (-1, 1), "a missing code is -1, the other stays"
    assert (match.ball_status[41], match.ball_possession[41]) == (1, -1), (match.ball_status[41],
                                                                        match.ball_possession[41])

    # Everything, built independently from the tables that wrote the file.
    want_xy, want_speed = np.full((43, 6, 2), np.nan), np.full((43, 6), np.nan)
    for (person, _), frames in SYNTHETIC_PLAYER_FRAMES.items():
        for n, jx, jy, speed in frames:
            row = synthetic_row(n)
            if row is None:
                continue  # before or after its half: dropped
            want_xy[row, columns[person]] = (np.nan if jx is None else jx / 8, np.nan if jy is None else jy / 8)
            want_speed[row, columns[person]] = np.nan if speed is None else float(speed) / 3.6
    want_ball_xy, want_ball_z, want_ball_speed = np.full((43, 2), np.nan), np.full(43, np.nan), np.full(43, np.nan)
    want_possession, want_status = np.full(43, -1, np.int8), np.full(43, -1, np.int8)
    for frames in SYNTHETIC_BALL_FRAMES.values():
        for n, jx, jy, z, speed, possession, status in frames:
            row = synthetic_row(n)
            want_ball_xy[row] = (jx / 8, jy / 8)
            want_ball_z[row] = np.nan if z is None else float(z)
            want_ball_speed[row] = np.nan if speed is None else float(speed) / 3.6
            want_possession[row] = -1 if possession is None else {"1": 0, "2": 1}[possession]
            want_status[row] = -1 if status is None else int(status)
    want_frames = np.concatenate([np.arange(10000, 10010), np.full(25, -1), np.arange(100000, 100008)])
    for name, want in (("frame_number", want_frames), ("players_xy", want_xy), ("players_speed", want_speed),
                       ("ball_xy", want_ball_xy), ("ball_z", want_ball_z), ("ball_speed", want_ball_speed),
                       ("ball_status", want_status), ("ball_possession", want_possession)):
        try:
            np.testing.assert_array_equal(getattr(match, name), want)
        except AssertionError as error:
            raise AssertionError("{} differs from the tables that wrote the file: {}".format(name, error)) from error

    assert match.report == SYNTHETIC_REPORT, match.report
    assert json.loads(json.dumps(match.report, sort_keys=True)) == match.report, "the report must be JSON-ready"
    assert all(type(leaf) is int for leaf in leaves(match.report)), "counts must be Python ints, not numpy's"
    try:
        match.fps = 30
    except AttributeError:
        pass
    else:
        raise AssertionError("DflMatch must be frozen")
    assert dfl.DflMatch.__eq__ is object.__eq__, "eq=False: a field-wise == over arrays is meaningless"


def test_positions_ignore_the_order_of_the_file():
    """The ball first and the FrameSets reversed: the same match, because rows come from N and never from position."""
    from pipeline.benchmark import dfl

    def reversed_framesets(text):
        blocks = re.findall(r"<FrameSet .*?</FrameSet>", text, flags=re.DOTALL)
        assert len(blocks) == 8 and "\n".join(blocks) in text, len(blocks)
        return text.replace("\n".join(blocks), "\n".join(reversed(blocks)))

    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        as_written = dfl.parse_positions(path, info)
        reordered = dfl.parse_positions(damaged(path, reversed_framesets), info)
    for name in MATCH_ARRAYS:
        np.testing.assert_array_equal(getattr(reordered, name), getattr(as_written, name), err_msg=name)
    assert reordered.report == as_written.report and reordered.graded == as_written.graded
    assert reordered != as_written, "two parses are two objects"


def test_positions_rejects_unknown_section():
    # The section is checked on every FrameSet, those the parser skips (the referee) included.
    cases = [(label, edit_frameset("firstHalf", person, lambda block: block.replace(
        'GameSection="firstHalf"', 'GameSection="firstHalfExtra"')), ["firstHalfExtra", "GameSection", "secondHalf"])
        for label, person in (("a referee's FrameSet", PERSON_REFEREE), ("a player's", PERSON_B),
                              ("the ball's", PERSON_BALL))]
    expect_positions_errors(cases)


def test_positions_rejects_bad_label():
    expect_positions_errors([
        ("BallStatus 3", edit_ball_frame(10002, 'BallStatus="1"', 'BallStatus="3"'),
         ["1 ball frame(s)", "N=10002", "firstHalf", "BallStatus='3'", "BallStatus that is not 0 or 1"]),
        ("BallPossession 0", edit_ball_frame(10001, 'BallPossession="1"', 'BallPossession="0"'),
         ["N=10001", "BallPossession='0'"]),
        ("a possession that is no integer", edit_ball_frame(10001, 'BallPossession="1"', 'BallPossession="1.5"'),
         ["BallPossession='1.5'"]),
        ("a possession that is text", edit_ball_frame(100001, 'BallPossession="2"', 'BallPossession="home"'),
         ["N=100001", "secondHalf", "BallPossession='home'"]),
        ("an empty status", edit_ball_frame(10003, 'BallStatus="1"', 'BallStatus=""'), ["N=10003", "BallStatus=''"]),
        ("two bad frames, in two halves", both(edit_ball_frame(100007, 'BallStatus="0"', 'BallStatus="2"'),
                                               edit_ball_frame(10004, 'BallPossession="1"', 'BallPossession="9"')),
         ["2 ball frame(s)", "N=10004", "firstHalf", "BallPossession='9'"])])


def test_positions_rejects_team_mismatch():
    def team(section, person, club, other):
        return edit_frameset(section, person, lambda block: block.replace(
            'TeamId="{}"'.format(club), 'TeamId="{}"'.format(other)))

    expect_positions_errors([
        ("a home player under the away club", team("secondHalf", PERSON_A, "DFL-CLU-SYNHOM", "DFL-CLU-SYNAWY"),
         [PERSON_A, "home", "DFL-CLU-SYNHOM", 'TeamId="DFL-CLU-SYNAWY"', "swap home and away"]),
        ("an away player under the home club", team("secondHalf", PERSON_C, "DFL-CLU-SYNAWY", "DFL-CLU-SYNHOM"),
         [PERSON_C, "away", "DFL-CLU-SYNAWY", 'TeamId="DFL-CLU-SYNHOM"']),
        ("a player under a club that is neither", team("firstHalf", PERSON_B, "DFL-CLU-SYNHOM", "DFL-CLU-NOBODY"),
         [PERSON_B, "DFL-CLU-SYNHOM", 'TeamId="DFL-CLU-NOBODY"'])])


def test_positions_rejects_other_match():
    other = "DFL-MAT-OTHER9"
    fragments = [other, "DFL-MAT-SYN001", "different matches"]
    expect_positions_errors([
        ("every FrameSet", lambda text: text.replace('MatchId="DFL-MAT-SYN001"', 'MatchId="{}"'.format(other)),
         fragments),
        ("one skipped FrameSet", edit_frameset("firstHalf", PERSON_REFEREE, lambda block: block.replace(
            'MatchId="DFL-MAT-SYN001"', 'MatchId="{}"'.format(other))), fragments)])


def test_positions_rejects_duplicates():
    expect_positions_errors([
        ("a person's FrameSet twice", edit_frameset("firstHalf", PERSON_B, lambda block: block + "\n" + block),
         [PERSON_B, 'GameSection="firstHalf"', "appears twice"]),
        ("a frame number twice in a FrameSet", edit_frameset("firstHalf", PERSON_B, edit_frame(
            10003, lambda line: line.replace('N="10003"', 'N="10002"'))),
         [PERSON_B, 'GameSection="firstHalf"', "N=10002", "twice"]),
        ("a frame number twice in the ball's", edit_ball_frame(100003, 'N="100003"', 'N="100002"'),
         ['TeamId="BALL"', "N=100002", "twice"]),
        ("a second ball FrameSet", edit_frameset("firstHalf", PERSON_BALL, lambda block: block + "\n" + block),
         ["second ball FrameSet", "firstHalf", 'TeamId="BALL"'])])


def test_positions_rejects_a_match_without_a_ball_in_each_half():
    def without_frames(block):
        return "\n".join(line for line in block.split("\n") if not line.startswith("<Frame "))

    expect_positions_errors([
        ("no ball in the second half", edit_frameset("secondHalf", PERSON_BALL, lambda block: ""),
         ["no ball FrameSet", "secondHalf", 'TeamId="BALL"', "--verify"]),
        ("no ball in the first half", edit_frameset("firstHalf", PERSON_BALL, lambda block: ""),
         ["no ball FrameSet", "firstHalf"]),
        ("a ball FrameSet without frames", edit_frameset("secondHalf", PERSON_BALL, without_frames),
         ["ball FrameSet", "secondHalf", "no frames"]),
        ("a file without any FrameSet", lambda text: "<PutDataRequest><Positions /></PutDataRequest>",
         ["no ball FrameSet", "firstHalf", "positions file"])])


def test_positions_rejects_damaged_files():
    def frame(person, section, n, old, new):
        return edit_frameset(section, person, edit_frame(n, lambda line: line.replace(old, new)))

    def attribute_removed(person, section, name, value):
        return edit_frameset(section, person, lambda block: block.replace(' {}="{}"'.format(name, value), "", 1))

    expect_positions_errors([
        ("a file cut short", lambda text: text[:len(text) // 2], ["well-formed", "--verify"]),
        ("a frame without N", frame(PERSON_A, "secondHalf", 100001, ' N="100001"', ""),
         [PERSON_A, "secondHalf", "no N attribute", "--verify"]),
        ("an N that is no number", frame(PERSON_A, "secondHalf", 100001, 'N="100001"', 'N="ten"'),
         [PERSON_A, "N='ten'", "whole number"]),
        ("an X that is no number", frame(PERSON_A, "secondHalf", 100000, 'X="{}"'.format(grid_x(1)), 'X="abc"'),
         [PERSON_A, "X='abc'", "finite number"]),
        ("an infinite Y", frame(PERSON_B, "firstHalf", 10000, 'Y="{}"'.format(grid_y(6)), 'Y="inf"'),
         [PERSON_B, "Y='inf'", "finite number"]),
        ("an S that is no number", frame(PERSON_C, "secondHalf", 100003, 'S="21.6"', 'S="fast"'),
         [PERSON_C, "S='fast'"]),
        ("a Z that is no number", edit_ball_frame(10001, 'Z="0.25"', 'Z="high"'), ['TeamId="BALL"', "Z='high'"]),
        ("a FrameSet without PersonId", attribute_removed(PERSON_B, "firstHalf", "PersonId", PERSON_B),
         ["no PersonId attribute", "--verify"]),
        ("a FrameSet without TeamId", attribute_removed(PERSON_B, "firstHalf", "TeamId", "DFL-CLU-SYNHOM"),
         ["no TeamId attribute", PERSON_B]),
        ("a FrameSet without GameSection", attribute_removed(PERSON_B, "firstHalf", "GameSection", "firstHalf"),
         ["no GameSection attribute", PERSON_B])])


def test_positions_names_a_missing_file():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        os.remove(path)
        expect_error(FileNotFoundError, lambda: dfl.parse_positions(path, info), os.path.basename(path),
                     "pipeline/benchmark/sources.py")


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


def test_dfl_match_information_all():
    """All 7 match information files parse, and say what the paper says: teams, a real pitch, full squads."""
    from pipeline.benchmark import dfl, sources
    print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>21}".format(
        "match", "home", "guest", "pitch (m)", "players", "total (raw)"))
    for match_id in sources.MATCH_IDS:
        info = dfl.parse_match_information(sources.local_path(match_id, "information"))
        assert info.match_id == match_id, (info.match_id, match_id)
        assert (info.home_name, info.away_name) == DFL_TEAMS[match_id], (match_id, info.home_name, info.away_name)
        assert 100 <= info.pitch[0] <= 110 and 64 <= info.pitch[1] <= 75, (match_id, info.pitch)
        counts = [sum(player.team == team for player in info.players) for team in (0, 1)]
        assert min(counts) >= 14, "{}: players per team {}; a squad has at least 14".format(match_id, counts)
        assert [player.team for player in info.players] == sorted(player.team for player in info.players), \
            "{}: players must list home first".format(match_id)
        assert sorted(info.total_time_raw) == [1, 2] and min(info.total_time_raw.values()) > 0, info.total_time_raw
        print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>21}".format(
            match_id, info.home_name, info.away_name, "{:g} x {:g}".format(*info.pitch),
            "{} + {}".format(*counts), "{:,.0f} / {:,.0f}".format(*(info.total_time_raw[half] for half in (1, 2)))))


def test_dfl_positions_one_match():
    """J03WMX's positions parse into a timeline that agrees with the information file and with itself (418 MB)."""
    from pipeline.benchmark import dfl, sources
    match_id = "J03WMX"
    info = dfl.parse_match_information(sources.local_path(match_id, "information"))
    started = time.perf_counter()
    match = dfl.parse_positions(sources.local_path(match_id, "positions"), info)
    seconds = time.perf_counter() - started
    report = match.report
    print("      parsed in {:.1f} s: {:,} rows x {} players, {:,} ball frames, {:,} player frames "
          "({} dropped outside the halves)".format(seconds, match.n_frames, match.n_players, report["ball_frames"],
                                                    report["player_frames"], report["player_frames_outside_halves"]))
    print("      skipped FrameSets {}; missing attributes {}".format(report["skipped_framesets"], report["missing"]))
    assert set(report["skipped_framesets"]) == {"referee"}, \
        "{}: FrameSets of {} were skipped; a player who is not in the match information would be missing".format(
            match_id, sorted(set(report["skipped_framesets"]) - {"referee"}))

    expected = sources.EXPECTED_BALL_FRAMES[match_id]
    assert report["ball_frames"] == expected, (report["ball_frames"], expected)
    (_, start1, stop1), (_, start2, stop2) = match.graded
    starts = (match.frame_number[start1], match.frame_number[start2])
    assert starts == (10000, 100000), "the halves start at N {} and {}, not 10000 and 100000".format(*starts)
    assert (start1, stop1 + dfl.GAP_ROWS, stop2) == (0, start2, match.n_frames), (match.graded, match.n_frames)
    print("      rows without a ball frame: {}".format(report["rows_without_ball"]))
    assert report["rows_without_ball"] == 0, report["rows_without_ball"]

    in_play = match.frame_number >= 0
    present = ~np.isnan(match.players_xy[:, :, 0])
    per_team = [present[:, match.player_team == team].sum(axis=1) for team in (0, 1)]
    over = float(((per_team[0] > 11) | (per_team[1] > 11))[in_play].mean())
    print("      rows with more than 11 players of a team: {:.5f} (home max {}, away max {})".format(
        over, per_team[0].max(), per_team[1].max()))
    assert over < 0.001, over

    length, width = match.pitch
    x_metres = (match.players_xy[:, :, 0][present] - 0.5) * length
    inside = float((np.abs(x_metres) <= length / 2 + 5).mean())
    print("      player frames no more than 5 m beyond a goal line: {:.5f}".format(inside))
    assert inside >= 0.999, inside

    # S is in km/h: against the distance a player really covers between frames, S / 3.6 is the same speed.
    metres = match.players_xy * np.array([length, width])
    step = np.hypot(*np.moveaxis(np.diff(metres, axis=0), -1, 0)) * match.fps  # frame r - 1 to r, in m/s
    stated = match.players_speed[1:]
    fast = np.isfinite(step) & np.isfinite(stated) & (step > 2.0)
    ratio = float(np.median(stated[fast] / step[fast]))
    print("      median (S / 3.6) / finite-difference speed, {:,} frames above 2 m/s: {:.3f}".format(
        int(fast.sum()), ratio))
    assert 0.9 <= ratio <= 1.1, ratio

    # TotalTime* is in milliseconds or in seconds: whichever reading makes the rows add up is the one that is right.
    def agrees(divisor):
        return all(abs((stop - start) / match.fps - info.total_time_raw[half] / divisor)
                   <= 0.02 * info.total_time_raw[half] / divisor for half, start, stop in match.graded)

    matched = [name for name, divisor in (("seconds", 1.0), ("ms/1000", 1000.0)) if agrees(divisor)]
    print("      rows / 25 per half: {}; TotalTime read as {}".format(
        " and ".join("{:,.1f} s".format((stop - start) / match.fps) for _, start, stop in match.graded),
        " and ".join(matched) or "neither"))
    assert matched, "rows / 25 is within 2% of TotalTime{First,Second}Half neither as seconds nor as ms / 1000"


# -- runner -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB), parse the 7 "
                             "match information files and one positions file")
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
