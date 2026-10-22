"""Self-checks for the P01 benchmark: the DFL registry, download and match information now; the rest later.

    python pipeline/benchmark/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/benchmark/selftest.py --dfl     # + the 21 downloaded DFL files: size and md5 (reads 2.6 GB),
                                                    #   and the 7 match information files parsed
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
import os
import re
import sys
import tempfile
import time
import urllib.error
from xml.sax.saxutils import quoteattr

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
# The home and guest team of each DFL match, spelled as General@HomeTeamName and General@GuestTeamName spell them in
# the files (the parser reads the names from the <Team> elements chosen by Role, so the two are independent).
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


def write_synthetic_dfl(folder, match_id="SYN001"):
    """Write a small DFL match in the real file format into `folder`; return {kind: path}, kinds as in sources.FILES.

    Only the match information file so far; the positions parser's task adds the positions file here. The element
    and attribute names are the real ones (copied from J03WMX's file), the values are made up and chosen so that
    every parser rule shows: the guest team is listed first, so Role decides and not the order; each squad is out of
    PersonId order; names exercise the fallbacks and non-ASCII text; a trainer and a referee carry PersonIds that
    are not players; the pitch is not the usual 105 x 68; the two half durations differ.
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
    return {"information": write_text(folder, name, text)}


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
        ("no pitch width", lambda text: text.replace(' PitchX="104.50"', ""), ["PitchX", "Environment"]),
        ("pitch that is no number", lambda text: text.replace('PitchY="67.50"', 'PitchY="wide"'), ["PitchY", "wide"]),
        ("negative pitch", lambda text: text.replace('PitchX="104.50"', 'PitchX="-105"'), ["PitchX", "-105"]),
        ("shirt that is no number", lambda text: text.replace('ShirtNumber="9"', 'ShirtNumber="nine"'),
         ["ShirtNumber", "nine", "DFL-OBJ-SYNA03"]),
        ("a player in both teams", lambda text: text.replace("DFL-OBJ-SYNH01", "DFL-OBJ-SYNA01"),
         ["DFL-OBJ-SYNA01", "twice"]),
        ("no second half duration", lambda text: text.replace(' TotalTimeSecondHalf="2820120"', ""),
         ["TotalTimeSecondHalf"]),
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
    print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>14}".format(
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
        print("      {:<7} {:<20} {:<22} {:>11} {:>9} {:>14}".format(
            match_id, info.home_name, info.away_name, "{:g} x {:g}".format(*info.pitch),
            "{} + {}".format(*counts), "{:,.0f} / {:,.0f}".format(*(info.total_time_raw[half] for half in (1, 2)))))


# -- runner -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", action="store_true",
                        help="also check the downloaded DFL files: size and md5 of all 21 (reads 2.6 GB), and parse "
                             "the 7 match information files")
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
