"""Self-checks for the P01 benchmark: the DFL registry, download, match information and positions now; the rest later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB),
                                                    #   the 7 match information files parsed, J03WMX's positions parsed
                                                    #   and cross-checked against kloppy and by frame number, and the
                                                    #   T30 counts on all 7 positions files (about 12 minutes)
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
import dataclasses
import functools
import hashlib
import io
import json
import math
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
# The cross-checks on J03WMX (kloppy, and the spot check by frame number).
TOLERANCE_M = 1e-6                    # metres: two readers of one coordinate may not differ by more than this
KLOPPY_LIMIT = 5000                   # kloppy's limit= is per object: the first 5,000 frames, all in the first half
HALF_START_N = {1: 10000, 2: 100000}  # J03WMX's first frame number in each half (test_dfl_positions_one_match)
GAP_ROWS = 25                         # separator rows between the halves, as the plan fixes it (not read from dfl)
SPOT_FRAMES = 200                     # frame numbers drawn from each half for the spot check
# What docs/traps.md T30 and results.md A05 quote, measured through our parser on each match (2026-10-22): the ball
# frames, the rows with no ball frame (kloppy skips them), the ball frames with no BallPossession (it turns them into
# the away team), the ball frames with no BallStatus, the X, Y, S and Z that are missing on any frame, and the dead
# ball frames (its default only_alive=True drops them). A BallPossession that is not 1 or 2 is not a count: our parser
# refuses such a file, so every match that parses has none.
T30_FACTS = {
    match_id: {"ball_frames": frames, "rows_without_ball": 0, "no_possession": 0, "no_status": 0, "no_value": 0,
               "dead": dead}
    for match_id, frames, dead in (
        ("J03WMX", 145967, 60598), ("J03WN1", 141561, 73184), ("J03WOH", 137214, 65269), ("J03WOY", 142536, 66250),
        ("J03WPY", 146211, 64751), ("J03WQQ", 142345, 57993), ("J03WR9", 146810, 61494))}


# -- helpers ----------------------------------------------------------------------------------------------------------

def utf8_stdout():
    """Make stdout write UTF-8, whatever the console or the pipe thinks its code page is (team names have umlauts)."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)  # a replaced stdout may not have it
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


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
         ["<Environment>", "2 <Environment> elements", "--verify"]),
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

    # A duplicated element in an otherwise valid file is a damaged download, not another kind of file.
    with tempfile.TemporaryDirectory() as folder:
        path = damaged(write_synthetic_dfl(folder)["information"],
                       lambda text: re.sub(r"<Environment[^>]*/>", lambda match: match.group(0) * 2, text))
        message = expect_error(ValueError, lambda: dfl.parse_match_information(path), "--verify")
        assert "not a DFL match information file" not in message, message


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


def insert_after(marker, addition):
    """A damage: `addition` goes right after the first `marker`."""
    return lambda text: text.replace(marker, marker + addition, 1)


def open_the_first_frameset(text):
    """A damage: the first FrameSet (the referee's) is never closed, so each one after it starts inside it.

    The closing tag that is missing is put back at the end, which keeps the file well-formed, so that the error is the
    parser's own and not the XML parser's.
    """
    opened = text.replace("</FrameSet>\n<FrameSet", "\n<FrameSet", 1)
    assert opened != text, "the synthetic file has no two FrameSets in a row"
    return opened.replace("</Positions>", "</FrameSet>\n</Positions>")


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
         [PERSON_B, "DFL-CLU-SYNHOM", 'TeamId="DFL-CLU-NOBODY"']),
        # The club is checked before the skip and the ball rules, so neither a referee's TeamId nor the ball's lets a
        # rostered player through (as a referee he would be skipped silently, as the ball he would be read as it).
        ("a home player under the referees' TeamId", team("firstHalf", PERSON_B, "DFL-CLU-SYNHOM", "referee"),
         [PERSON_B, "home", "DFL-CLU-SYNHOM", 'TeamId="referee"', "swap home and away"]),
        ("an away player under the ball's TeamId", team("secondHalf", PERSON_C, "DFL-CLU-SYNAWY", "BALL"),
         [PERSON_C, "away", "DFL-CLU-SYNAWY", 'TeamId="BALL"', "swap home and away"])])


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
         ["no GameSection attribute", PERSON_B]),
        ("a <Frame> before any FrameSet", insert_after("</MetaData>\n", '<Frame N="1" />\n'),
         ["<Frame> outside any <FrameSet>", "positions file"]),
        ("a <Frame> between two FrameSets", insert_after("</FrameSet>\n", '<Frame N="1" />\n'),
         ["<Frame> outside any <FrameSet>", "positions file"]),
        ("a FrameSet inside another", open_the_first_frameset,
         ["starts inside", PERSON_REFEREE, PERSON_UNLISTED, "positions file"])])


def test_positions_names_a_missing_file():
    from pipeline.benchmark import dfl
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        os.remove(path)
        expect_error(FileNotFoundError, lambda: dfl.parse_positions(path, info), os.path.basename(path),
                     "pipeline/benchmark/sources.py")


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


# -- the downloaded DFL files (--dfl) ---------------------------------------------------------------------------------

PARSE_SECONDS = {}  # match id -> how long its positions took to parse (real_match), for the check that prints it


@functools.lru_cache(maxsize=1)
def real_match(match_id):
    """The DflMatch of a downloaded match, parsed once per run for the checks that follow each other.

    One match is about 150 MB of arrays and a parse takes about a minute, so only the last match is kept. The checks
    on J03WMX run back to back (their names sort that way) and share one parse. Task 5's disk cache
    (dfl.load_match) will replace this.
    """
    from pipeline.benchmark import dfl, sources
    info = dfl.parse_match_information(sources.local_path(match_id, "information"))
    started = time.perf_counter()
    match = dfl.parse_positions(sources.local_path(match_id, "positions"), info)
    PARSE_SECONDS[match_id] = time.perf_counter() - started
    return match


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
    match = real_match(match_id)
    report = match.report
    print("      parsed in {:.1f} s: {:,} rows x {} players, {:,} ball frames, {:,} player frames "
          "({} dropped outside the halves)".format(PARSE_SECONDS[match_id], match.n_frames, match.n_players,
                                                    report["ball_frames"], report["player_frames"],
                                                    report["player_frames_outside_halves"]))
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


# -- cross-checks of the parser on J03WMX: kloppy, and a second pass keyed by frame number (--dfl) ---------------------

def differing(ours, theirs):
    """How many entries differ by TOLERANCE_M or more; a NaN matches only a NaN."""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    nan_ours, nan_theirs = np.isnan(ours), np.isnan(theirs)
    both = ~nan_ours & ~nan_theirs
    return int((nan_ours != nan_theirs).sum() + (np.abs(ours - theirs)[both] >= TOLERANCE_M).sum())


def kloppy_load(match_id, only_alive):
    """kloppy's dataset of the first KLOPPY_LIMIT frames of a downloaded match, as the T30 advice would load it."""
    from kloppy import sportec
    from pipeline.benchmark import sources
    return sportec.load_tracking(
        meta_data=sources.local_path(match_id, "information"), raw_data=sources.local_path(match_id, "positions"),
        coordinates="sportec", only_alive=only_alive, limit=KLOPPY_LIMIT)


def kloppy_rows(dataset):
    """(frame numbers, timeline rows) of a dataset that covers the first half only, as the check must find it."""
    numbers = np.array([record.frame_id for record in dataset.records], dtype=np.int64)
    periods = {record.period.id for record in dataset.records}
    assert periods == {1}, "kloppy's limit is per object, so it should stop inside the first half; it read {}".format(
        sorted(periods))
    return numbers, numbers - HALF_START_N[1]


def kloppy_differences(dataset, match, info):
    """({what: how many values were compared}, {what: how many disagree}) for kloppy's frames against the timeline.

    The row of kloppy's frame N is N - HALF_START_N[1]; the timeline's own frame numbers must agree with that. The ball
    is compared in metres (ours converted back from 0-1), ALIVE against status 1, and "the home team has the ball"
    against possession 0. Three players are compared by PersonId: the two lowest ids among the home players and the
    lowest among the away players of kloppy's first frame. Their positions must agree wherever both readers have one,
    and presence must agree too: our NaN is the player missing from kloppy's frame. Kloppy's own team for each player
    of its first frame must be ours, and the number of players in every frame must be the same.
    """
    from kloppy.domain import BallState, Ground
    teams = {team.ground: team for team in dataset.metadata.teams}
    assert (teams[Ground.HOME].team_id, teams[Ground.AWAY].team_id) == (info.home_id, info.away_id), \
        "kloppy and the parser disagree about which team is home: {} / {}".format(
            [team.team_id for team in dataset.metadata.teams], (info.home_id, info.away_id))
    numbers, rows = kloppy_rows(dataset)
    assert (match.frame_number[rows] == numbers).all(), "the timeline's frame numbers are not N - {}".format(
        HALF_START_N[1])
    records = dataset.records
    length, width = match.pitch

    ball = np.array([(r.ball_coordinates.x, r.ball_coordinates.y, r.ball_coordinates.z) for r in records])
    alive = np.array([r.ball_state == BallState.ALIVE for r in records])
    home_has_ball = np.array([r.ball_owning_team.team_id == teams[Ground.HOME].team_id for r in records])
    differences = {
        "ball x": differing(match.ball_xy[rows, 0] * length - length / 2, ball[:, 0]),
        "ball y": differing(width / 2 - match.ball_xy[rows, 1] * width, ball[:, 1]),
        "ball z": differing(match.ball_z[rows], ball[:, 2]),
        "alive": int((alive != (match.ball_status[rows] == 1)).sum()),
        "home has the ball": int((home_has_ball != (match.ball_possession[rows] == 0)).sum())}
    compared = {"ball x": len(records), "ball y": len(records), "ball z": len(records), "alive": len(records),
                "home has the ball": len(records), "alive frames": int(alive.sum()),
                "home frames": int(home_has_ball.sum())}

    team_of = {player.person_id: player.team for player in info.players}
    kloppy_team = {player.player_id: 0 if player.team.ground == Ground.HOME else 1
                   for player in records[0].players_data}
    differences["player team"] = sum(team_of.get(person) != team for person, team in kloppy_team.items())
    compared["player team"] = len(kloppy_team)
    chosen = ([person for person, team in sorted(kloppy_team.items()) if team == 0][:2]
              + [person for person, team in sorted(kloppy_team.items()) if team == 1][:1])
    assert len(chosen) == 3 and {kloppy_team[person] for person in chosen} == {0, 1}, chosen
    column = {player.person_id: index for index, player in enumerate(info.players)}
    frames = [{player.player_id: data.coordinates for player, data in record.players_data.items()}
              for record in records]
    differences.update({"player presence": 0, "player x/y": 0})
    compared.update({"player presence": 0, "player x/y": 0})
    for person in chosen:
        theirs_present = np.array([person in frame for frame in frames])
        theirs = np.array([(frame[person].x, frame[person].y) if person in frame else (np.nan, np.nan)
                           for frame in frames])
        xy = match.players_xy[rows, column[person]]
        ours = np.column_stack([xy[:, 0] * length - length / 2, width / 2 - xy[:, 1] * width])
        ours_present = ~np.isnan(xy[:, 0])
        both = theirs_present & ours_present
        differences["player presence"] += int((theirs_present != ours_present).sum())
        differences["player x/y"] += differing(ours[both], theirs[both])
        compared["player presence"] += len(records)
        compared["player x/y"] += 2 * int(both.sum())
    ours_count = (~np.isnan(match.players_xy[rows, :, 0])).sum(axis=1)
    differences["players per frame"] = int((ours_count != np.array([len(frame) for frame in frames])).sum())
    compared["players per frame"] = len(records)
    return compared, differences


def kloppy_default_drops(dataset, match):
    """(returned that are dead in ours, alive rows of ours that kloppy left out, dead rows it dropped, N range, rows).

    The range is that of the frame numbers kloppy returned: with only_alive=True its limit counts alive frames, so it
    reaches further than the limit says. Inside the range, every alive row of ours must be among the frames returned
    and none of the frames returned may be dead.
    """
    numbers, rows = kloppy_rows(dataset)
    span = np.arange(numbers.min(), numbers.max() + 1) - HALF_START_N[1]
    returned = np.zeros(span.size, dtype=bool)
    returned[rows - span[0]] = True
    status = match.ball_status[span]
    return (int((returned & (status != 1)).sum()), int(((status == 1) & ~returned).sum()),
            int(((status == 0) & ~returned).sum()), (int(numbers.min()), int(numbers.max())), int(span.size))


def test_dfl_parser_agrees_with_kloppy():
    """On J03WMX's first 5,000 frames kloppy reads the same ball, alive flag, possession and players as we do.

    Then kloppy's default, only_alive=True, is run too, which makes T30's first claim a measurement: it returns no
    dead frame and leaves out no alive one inside the range it covers, so the dead frames are exactly what it drops.
    """
    try:
        from kloppy import sportec  # noqa: F401
    except ImportError:
        print("      kloppy is not installed (it is in requirements-starter.txt): nothing to compare with, passing")
        return
    from pipeline.benchmark import dfl, sources
    match_id = "J03WMX"
    info = dfl.parse_match_information(sources.local_path(match_id, "information"))
    match = real_match(match_id)

    full = kloppy_load(match_id, only_alive=False)
    numbers, _ = kloppy_rows(full)
    assert (numbers == np.arange(HALF_START_N[1], HALF_START_N[1] + KLOPPY_LIMIT)).all(), \
        "limit={} should return the first {} frames of the first half, in order".format(KLOPPY_LIMIT, KLOPPY_LIMIT)
    compared, differences = kloppy_differences(full, match, info)
    print("      kloppy only_alive=False, limit={}: {:,} frames, N {}-{} (the first half only); {:,} alive, {:,} home "
          "possession".format(KLOPPY_LIMIT, len(numbers), numbers[0], numbers[-1], compared["alive frames"],
                              compared["home frames"]))
    for what in ("ball x", "ball y", "ball z", "alive", "home has the ball", "player team", "player presence",
                 "player x/y", "players per frame"):
        print("        {:<18} {:>9,} compared, {} disagreements".format(what, compared[what], differences[what]))
    assert not any(differences.values()), "kloppy and the parser disagree: {}".format(
        {what: count for what, count in differences.items() if count})

    default = kloppy_load(match_id, only_alive=True)
    returned_dead, left_out, dropped, (first, last), span = kloppy_default_drops(default, match)
    print("      kloppy default (only_alive=True), same limit: {:,} frames over N {}-{} ({:,} rows): {} returned that "
          "are dead in ours, {} alive rows left out, {:,} dead rows dropped".format(
              len(default.records), first, last, span, returned_dead, left_out, dropped))
    assert (returned_dead, left_out) == (0, 0), (returned_dead, left_out)
    assert dropped > 0 and len(default.records) + dropped == span, "the default should drop exactly the dead rows"


def read_ball_frames(path, wanted):
    """{(half, N): the attributes of the ball's <Frame>} for the frame numbers in wanted, {half: N values}.

    A pass of its own, on purpose: it calls nothing of dfl, so that a mistake the parser makes cannot be made here too.
    It looks only at the FrameSets of the ball (TeamId BALL) and keeps the frames it was asked for, as written.
    """
    from lxml import etree
    halves = {"firstHalf": 1, "secondHalf": 2}
    wanted = {half: set(numbers) for half, numbers in wanted.items()}
    found, half, is_ball = {}, None, False
    for event, element in etree.iterparse(path, events=("start", "end"), tag=("FrameSet", "Frame"), huge_tree=True,
                                          resolve_entities=False, no_network=True):
        if element.tag == "FrameSet":
            if event == "start":
                half, is_ball = halves[element.get("GameSection")], element.get("TeamId") == "BALL"
        elif event == "end":
            if is_ball and int(element.get("N")) in wanted[half]:
                found[(half, int(element.get("N")))] = dict(element.attrib)
            element.clear()
            while element.getprevious() is not None:  # free what is done: the file is 400 MB
                del element.getparent()[0]
    return found


def spot_differences(match, path, sample):
    """The (N, attribute) pairs where the timeline's ball differs from the ball frame the file has at N.

    `sample` is {half: N values}. The row of N is placed here and not by the parser: N - 10000 in the first half, and
    n1 + 25 + (N - 100000) in the second, n1 being the rows of the first half. Every ball field the timeline stores is
    compared: X, Y (in metres), Z, S (km/h as m/s), BallStatus and BallPossession (1 and 2 as 0 and 1), and N itself.
    An N with no ball frame in the file must have the ball NaN and its codes -1.
    """
    found = read_ball_frames(path, sample)
    length, width = match.pitch
    n1 = match.graded[0][2]
    differences = []
    for half, numbers in sorted(sample.items()):
        for n in numbers:
            row = n - HALF_START_N[1] if half == 1 else n1 + GAP_ROWS + (n - HALF_START_N[2])
            frame = found.get((half, n))  # None: the file has no ball frame at N

            def number(name):
                return math.nan if frame is None or name not in frame else float(frame[name])

            def code(name, codes):
                return -1 if frame is None or name not in frame else codes[frame[name]]

            expected = {"N": n, "X": number("X"), "Y": number("Y"), "Z": number("Z"), "S": number("S") / 3.6,
                        "BallStatus": code("BallStatus", {"0": 0, "1": 1}),
                        "BallPossession": code("BallPossession", {"1": 0, "2": 1})}
            actual = {"N": int(match.frame_number[row]), "X": match.ball_xy[row, 0] * length - length / 2,
                      "Y": width / 2 - match.ball_xy[row, 1] * width, "Z": match.ball_z[row],
                      "S": match.ball_speed[row], "BallStatus": int(match.ball_status[row]),
                      "BallPossession": int(match.ball_possession[row])}
            differences += [(n, name) for name in expected if differing(actual[name], expected[name])]
    return differences


def test_spot_check_reads_the_synthetic_match_and_can_fail():
    """The spot check finds no difference on the synthetic match, and one in each field that is damaged.

    The synthetic match has a frame with no ball and a missing value in every attribute, so those rules are exercised.
    The damage is what a bug in the parser would be: a field wrong, a half shifted by one row, a frame number off.
    """
    from pipeline.benchmark import dfl
    sample = {1: list(range(10000, 10010)), 2: list(range(100000, 100008))}

    def shifted_second_half(array):
        moved = array.copy()
        moved[35:] = np.roll(array[35:], 1, axis=0)
        return moved

    damages = {  # field -> how to damage it
        "ball_xy": lambda array: array + 0.01, "ball_z": lambda array: array + 0.01,
        "ball_speed": lambda array: array + 0.01, "ball_status": lambda array: np.where(array == 1, 0, 1).astype("i1"),
        "ball_possession": lambda array: np.where(array == 1, 0, 1).astype("i1"),
        "frame_number": lambda array: array + 1}
    with tempfile.TemporaryDirectory() as folder:
        info, path = positions_of(folder)
        match = dfl.parse_positions(path, info)
        assert spot_differences(match, path, sample) == [], spot_differences(match, path, sample)
        for field, damage in damages.items():
            wrong = dataclasses.replace(match, **{field: damage(getattr(match, field))})
            assert spot_differences(wrong, path, sample), "damaging {} went unnoticed".format(field)
        late = dataclasses.replace(match, **{field: shifted_second_half(getattr(match, field)) for field in (
            "frame_number", "ball_xy", "ball_z", "ball_speed", "ball_status", "ball_possession")})
        assert spot_differences(late, path, sample), "a second half one row late went unnoticed"


def test_dfl_second_half_spot_check():
    """400 frames of J03WMX's ball, drawn from both halves, read by a second pass over the file, match the timeline.

    200 frame numbers come from the second half, which sits after the gap between the halves and so shows a wrong gap,
    and 200 from the first, which kloppy covers only in its first 5,000 frames. Each is looked up in the file by N.
    """
    from pipeline.benchmark import sources
    match_id = "J03WMX"
    match = real_match(match_id)
    sample = {}
    for half, seed in ((2, 0), (1, 1)):
        _, start, stop = match.graded[half - 1]
        sample[half] = sorted(int(n) for n in np.random.default_rng([seed]).choice(
            match.frame_number[start:stop], size=SPOT_FRAMES, replace=False))
    differences = spot_differences(match, sources.local_path(match_id, "positions"), sample)
    print("      {} frames per half (N {}-{} and {}-{}), 7 fields each (N, X, Y, Z, S, BallStatus, BallPossession): "
          "{} disagreements".format(SPOT_FRAMES, sample[1][0], sample[1][-1], sample[2][0], sample[2][-1],
                                    len(differences)))
    assert not differences, "the file and the timeline disagree at (N, attribute): {}".format(differences[:10])


def test_dfl_t30_facts():
    """The numbers behind docs/traps.md T30, on all 7 matches: how often kloppy's DFL loader would go wrong.

    kloppy turns a missing BallPossession into the away team, skips a frame that has no ball entry, and by default
    drops every dead-ball frame. The table counts each through our parser, which refuses a BallPossession that is not
    1 or 2 (so every match in it has 0 of those, the "other" column), and the counts are asserted exactly, as the docs
    quote them. About 6 minutes: one parse per match.
    """
    from pipeline.benchmark import sources
    print("      {:<7} {:>11} {:>13} {:>13} {:>5} {:>9} {:>11} {:>16}".format(
        "match", "ball frames", "rows w/o ball", "no possession", "other", "no status", "no x/y/s/z", "dead (share)"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        report = match.report
        dead = int((match.ball_status == 0).sum())
        measured = {"ball_frames": report["ball_frames"], "rows_without_ball": report["rows_without_ball"],
                    "no_possession": report["missing"]["BallPossession"],
                    "no_status": report["missing"]["BallStatus"],
                    "no_value": sum(report["missing"][name] for name in ("X", "Y", "S", "Z")), "dead": dead}
        print("      {:<7} {:>11,} {:>13,} {:>13,} {:>5} {:>9,} {:>11,} {:>8,} ({:.1%})".format(
            match_id, measured["ball_frames"], measured["rows_without_ball"], measured["no_possession"], 0,
            measured["no_status"], measured["no_value"], dead, dead / report["ball_frames"]))
        assert measured == T30_FACTS[match_id], "{}: measured {} but the docs say {}".format(
            match_id, measured, T30_FACTS[match_id])


# -- runner -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB), parse the 7 "
                             "match information files, cross-check J03WMX's positions against kloppy and by frame "
                             "number, and count T30's facts on all 7 positions files (about 12 minutes)")
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
