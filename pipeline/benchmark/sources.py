"""The seven open DFL/IDSSE Bundesliga matches behind the P01 benchmark: what they are, and a download that proves it.

    python pipeline/benchmark/sources.py                       # fetch what is missing or damaged, then it is all ok
    python pipeline/benchmark/sources.py --verify              # check size and md5 of what is on disk, download nothing
    python pipeline/benchmark/sources.py --matches J03WMX ...  # only these matches
    python pipeline/benchmark/sources.py --data-dir DIR        # into DIR/external/dfl (default: data/, or $MTAH_DATA_DIR)

Why these files
---------------
We may not hand-label our own match, not even to validate (organisers' clarification of 22 Oct 2026, CLAUDE.md), so
the choices that need a ground truth are validated on other matches. The DFL/IDSSE open data are seven Bundesliga
and 2. Bundesliga matches with 25 Hz tracking and event data under CC BY 4.0. They are not competition data, so the
hard rules about match data do not apply to them (their own licence does; CLAUDE.md, data/external/). Each match is
three XML files: the match information (teams, players, pitch), the raw events, and the observed positions (350-420 MB
each; 2.6 GB for all 21 files).

Why every file is pinned by size and md5
----------------------------------------
A truncated 400 MB download that still parses would give benchmark numbers that look fine and mean nothing, and a
silently changed upstream file would move every score between two runs. So the registry below pins the figshare file
id, the byte size and the md5 of each file, nothing is kept unless it matches, and the benchmark's own checks
(selftest.py --dfl) re-verify the files on disk. If a pinned value is ever wrong, the download says which file and
shows both hashes; fix the registry in the same commit that explains why.

How the download is safe
------------------------
Bytes stream to <file>.part in 1 MiB chunks, hashed as they arrive, and only a file that matches its pinned size and
md5 is renamed into place, so a final file is either complete and right or absent. A failed check -- or a dead
server -- is retried once from a Hugging Face mirror of the same data set, pinned to one commit so that it cannot
drift. A .part left by an interrupted run is simply overwritten by the next one.
"""

import argparse
import hashlib
import http.client
import os
import sys
import urllib.request
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so
    # that `import games` resolves from any working directory and no file in here can shadow a top-level module.
    sys.path[0] = ROOT

import games  # noqa: E402

# Params
# Their order is the stable match_index that later modules use to seed and to split.
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
# The first two matches are Bundesliga, the other five 2. Bundesliga; the competition id is part of every file name.
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
               "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
               "J03WR9": "DFL-COM-000002"}
KINDS = ("information", "positions", "events")
NAME_PREFIX = {"information": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
               "positions": "DFL_04_03_positions_raw_observed"}

FIGSHARE_URL = "https://ndownloader.figshare.com/files/{}"
MIRROR_URL = ("https://huggingface.co/datasets/pysport/idsse-data/resolve/"
              "a715a38dfbaf5f58e431727c2b78d174101a703c/{}")
USER_AGENT = "mtah-benchmark/1 (python-urllib)"
TIMEOUT = 60           # seconds, for the connection and for every read
CHUNK = 1 << 20        # bytes per read: a 400 MB file is never in memory

# Per match and kind: (figshare file id, bytes, md5).
_PINNED = {
    "J03WMX": {"information": (51643475, 11930, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
               "events": (51643493, 776578, "2d88e180a1f63c7e4157feb980fc7255"),
               "positions": (51643514, 418524833, "e4273a2adf5dbd9d4ec42def2f01905c")},
    "J03WN1": {"information": (51643472, 11722, "37b0f85588c0acfb995fe633c7ac2e0d"),
               "events": (51643496, 612114, "83e8ed7ea8e8ebbbdc61f61ac3534008"),
               "positions": (51643517, 391722803, "bacb2f609c1d1b0bfc906f6b9d4109f2")},
    "J03WOH": {"information": (51643478, 11745, "b9710781d400dd812f62b0a6e3d7bce8"),
               "events": (51643499, 635747, "ab66e51512f39d06f49f27dd3b4c2f40"),
               "positions": (51643520, 348945715, "3efff25a86a61768e7d253b4bba137a3")},
    "J03WOY": {"information": (51643481, 11952, "7b021e577fca917a63111d55cc29ae5b"),
               "events": (51643502, 683018, "64b1be01d6ff6848168efebd56f344b9"),
               "positions": (51643523, 362567859, "4e1063e0a0ae499920cf68ac1aa3ab4e")},
    "J03WPY": {"information": (51643487, 11922, "65a4568e4f374af94d3c1015cbc7c4f5"),
               "events": (51643505, 698898, "f68482441d38723d69f244f896482803"),
               "positions": (51643526, 371688499, "e56ff1336a334f8da51110fbe7cab764")},
    "J03WQQ": {"information": (51643484, 11932, "5979209af316a914ce57347d0633df6c"),
               "events": (51643508, 732186, "b7e9d82af4034539d802319be6c57776"),
               "positions": (51643529, 356219423, "9105bad987191d3c3c362c1fce9a4fcd")},
    "J03WR9": {"information": (51643490, 11823, "5e2d580d3d8ed97e32959de90d04992b"),
               "events": (51643511, 681050, "c13bc5efa5eca4a022cc678210b87d5f"),
               "positions": (51643532, 372934140, "e060bb7d7a5a3bfbd15bbf9bc631024d")},
}

# The number of ball frames in each positions file, both halves together. The parser does not read it: dfl.load_match
# compares only each file's size with the registry's (a hand-copied or other-version file is refused there) and
# sources.py --verify the md5; these counts are checked by checks/dfl.py (test_dfl_all_matches_load for all seven,
# test_dfl_positions_one_match for J03WMX) against what the parse reads, so that a parse that comes out differently
# fails there instead of shifting every label.
EXPECTED_BALL_FRAMES = {"J03WMX": 145967, "J03WN1": 141561, "J03WOH": 137214, "J03WOY": 142536,
                        "J03WPY": 146211, "J03WQQ": 142345, "J03WR9": 146810}


class DownloadError(OSError):
    """A file could not be fetched, or what arrived is not the file that was pinned. The message says what to do."""


def external_dir(data_dir=None):
    """Where the DFL files live: <data dir>/external/dfl. data/ is gitignored, and so is everything under it."""
    return os.path.join(data_dir or games.DATA_DIR, "external", "dfl")


@dataclass(frozen=True)
class RemoteFile:
    """One pinned file: where figshare has it, and the size and md5 it must have once it is on disk."""
    match_id: str
    kind: str
    name: str
    file_id: int
    size: int
    md5: str

    @property
    def url(self):
        return FIGSHARE_URL.format(self.file_id)

    @property
    def mirror_url(self):
        return MIRROR_URL.format(self.name)

    def path(self, data_dir=None):
        """<external dir>/<match id>/<name>"""
        return os.path.join(external_dir(data_dir), self.match_id, self.name)


FILES = {
    match_id: {kind: RemoteFile(match_id, kind,
                                "{}_{}_DFL-MAT-{}.xml".format(NAME_PREFIX[kind], COMPETITION[match_id], match_id),
                                *_PINNED[match_id][kind])
               for kind in KINDS}
    for match_id in MATCH_IDS}


def local_path(match_id, kind, data_dir=None):
    """Where the file of this match and kind lives (or will live) on disk."""
    return FILES[match_id][kind].path(data_dir)


def select(match_ids=None):
    """The RemoteFiles of the given matches, in registry order, KINDS order within a match. None means all 21."""
    if match_ids is not None:
        match_ids = tuple(match_ids)  # read once: a generator would be empty after the validation below
        unknown = [match_id for match_id in match_ids if match_id not in FILES]
        if unknown:
            raise ValueError("unknown DFL match id {}; the registry has {}".format(unknown[0], ", ".join(MATCH_IDS)))
    return [remote for match_id in MATCH_IDS if match_ids is None or match_id in match_ids
            for remote in FILES[match_id].values()]


def verify(files=None, data_dir=None):
    """{path: status} for each file, in order. Status is ok, missing, size (wrong byte count) or md5 (wrong content).

    The cheap check goes first: a file of the wrong size is reported without reading it.
    """
    return {remote.path(data_dir): _status(remote.path(data_dir), remote)
            for remote in (select() if files is None else files)}


def download(files=None, data_dir=None, opener=None, log=print):
    """Make every file ok on disk and return their paths, in order; DownloadError at the first one that cannot be.

    A file that already passes its size and md5 is left alone and costs no request. Anything else is downloaded;
    a damaged file is removed first, so a download that fails leaves nothing at the final path. `log` gets one line
    for a file that is ok, and two for one that is fetched (when it starts and when it is in place), and one more
    when figshare fails and the mirror is tried. `opener(request, timeout=...)` is
    urllib.request.urlopen by default; it is a parameter so that the checks can serve bodies from memory through
    this same code.
    """
    opener = opener or urllib.request.urlopen
    paths = []
    for remote in (select() if files is None else files):
        path = remote.path(data_dir)
        status = _status(path, remote)
        label = _label(remote)
        if status == "ok":
            log("ok          {:<22} {:>12,} bytes".format(label, remote.size))
        else:
            log("downloading {:<22} {:>12,} bytes ({})".format(label, remote.size, status))
            _fetch(remote, path, opener, log)
            log("downloaded  {}".format(label))
        paths.append(path)
    return paths


def _status(path, remote):
    if not os.path.isfile(path):
        return "missing"
    if os.path.getsize(path) != remote.size:
        return "size"
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return "ok" if digest.hexdigest() == remote.md5 else "md5"


def _label(remote):
    return "{}/{}".format(remote.match_id, remote.kind)


def _fetch(remote, path, opener, log):
    """Stream the file to <path>.part from figshare, or failing that from the mirror, then rename it into place.

    Whatever stood at the final path is removed first (download() calls this only for a missing or damaged file), so
    that a download which fails leaves the file absent, never damaged: the promise of the module docstring.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as error:
            raise DownloadError(
                "{}: the damaged file {} cannot be removed ({}); another process may be using it. Close it, or "
                "delete the file by hand, then re-run.".format(remote.name, path, error)) from error
    part = path + ".part"
    sources = (("figshare", remote.url), ("mirror", remote.mirror_url))
    failures = []
    for position, (source, url) in enumerate(sources):
        try:
            _stream(remote, url, part, opener)
            break
        except DownloadError as error:
            failures.append("  {} ({}): {}".format(source, url, error))
            if position + 1 < len(sources):
                log("retrying    {:<22} {} failed ({}); trying the {}".format(
                    _label(remote), source, error, sources[position + 1][0]))
    else:
        raise DownloadError(
            "{}: neither source gave a file with the pinned size and md5, and nothing was kept. Re-run to try "
            "again; if it keeps failing, check the network, then the pinned values in pipeline/benchmark/"
            "sources.py.\n{}".format(remote.name, "\n".join(failures)))
    try:
        os.replace(part, path)
    except OSError as error:  # a PermissionError is the usual one on Windows; a full or read-only disk is another
        _discard(part)
        advice = ("another download may still be writing the file. Wait for it to finish, then re-run."
                  if isinstance(error, PermissionError) else
                  "check the free disk space and the permissions of that folder, then re-run.")
        raise DownloadError("{}: the verified download could not be moved to {} ({}); {}".format(
            remote.name, path, error, advice)) from error


def _stream(remote, url, part, opener):
    """One attempt: URL -> part, hashed on the way. Leaves part only if the bytes are exactly the pinned file."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.md5(usedforsecurity=False)
    received = 0
    try:
        with opener(request, timeout=TIMEOUT) as response, open(part, "wb") as handle:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                handle.write(chunk)
                digest.update(chunk)
                received += len(chunk)
    except (OSError, http.client.HTTPException) as error:  # URLError, HTTPError, timeouts, resets, truncated bodies
        _discard(part)
        raise DownloadError("{}: {}".format(type(error).__name__, error)) from error
    if received != remote.size or digest.hexdigest() != remote.md5:
        _discard(part)
        raise DownloadError("got {} bytes with md5 {}, expected {} bytes with md5 {}".format(
            received, digest.hexdigest(), remote.size, remote.md5))


def _discard(path):
    """Remove a partial file; one that is already gone, or held by another process, is not worth failing for."""
    try:
        os.remove(path)
    except OSError:
        pass


def main(argv=None, data_dir=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="only check the files on disk (size and md5); download nothing")
    parser.add_argument("--matches", nargs="+", choices=MATCH_IDS, metavar="ID",
                        help="only these matches (default: all {}); choices: {}".format(len(MATCH_IDS),
                                                                                       ", ".join(MATCH_IDS)))
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR); the files go under "
                                           "DIR/external/dfl, as run.py and fidelity.py read them")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or data_dir
    files = select(args.matches)

    if args.verify:
        statuses = verify(files, data_dir)
        for remote in files:
            print("{:<8} {}/{}".format(statuses[remote.path(data_dir)], remote.match_id, remote.kind))
        ok = sum(status == "ok" for status in statuses.values())
    else:
        try:
            download(files, data_dir)
        except DownloadError as error:
            print("FAILED  {}".format(error))
            return 1
        ok = len(files)  # download() returns only once every file passes its size and md5
    print("\n{} of {} files ok under {}".format(ok, len(files), external_dir(data_dir)))
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
