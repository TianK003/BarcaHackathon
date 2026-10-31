"""The PathCRF files, pinned by commit, git blob SHA-1 and sha256, fetched file by file and never committed.

The published model's code and one checkpoint are about 19.8 MB, but the repository holds ten checkpoints and a clone
would be 190 MB, so each file is fetched from raw.githubusercontent.com at the pinned commit. A pin by commit is
exact, and the git blob SHA-1 of each file (`sha1(b"blob <size>\\0" + bytes)`) is fixed by that commit, so the bytes
are checked against the tree before anyone has measured anything. The sha256 is recorded at the first verified fetch
and enforced from then on: a hash that git did not fix is only a record of what was verified once, and says so.

Bytes stream to <file>.part, hashed as they arrive, and the file is kept only if its size, its blob SHA-1 and (once
recorded) its sha256 match; otherwise the part is removed and DownloadError says what arrived. This is the promise of
pipeline/benchmark/sources.py, with stdlib only: this module runs in both venvs, and in neither does it need more.

    python pathcrf_lab/sources.py              # fetch what is not ok, verify, write manifest.json
    python pathcrf_lab/sources.py --verify     # only check the files on disk
    python pathcrf_lab/sources.py --record     # print `path sha256` of every verified file, to paste into FILES

Files go under <data dir>/external/pathcrf/repo/, which is gitignored with everything under data/. PathCRF's licence
file is one of them: it travels with the code it governs.
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    # Run by path, Python put pathcrf_lab/ itself first on sys.path: put the repository root there instead.
    sys.path[0] = ROOT

import games  # noqa: E402


@dataclass(frozen=True)
class PinnedFile:
    """One pinned file: its path in the repository, its size, its git blob SHA-1 and its sha256 (None until recorded)."""
    path: str
    size: int
    blob_sha1: str
    sha256: Optional[str] = None


# Params
OWNER_REPO = "hyunsungkim-ds/pathcrf"
COMMIT = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
CHUNK = 1 << 20
TIMEOUT = 60
USER_AGENT = "pathcrf-lab-p20/1.0"
# The 27 files of research.md section 5: the checkpoint of trial 140 (Dynamic Masked CRF) and its arguments, the
# held-out match's parquet, the licence, and every module of the commit that the code imports (some lazily).
FILES = (
    PinnedFile("saved/140/model/state_dict_best_acc.pt", 19325114, "f8e48952d01d6d70db986948962822478884a336",
                "55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9"),
    PinnedFile("saved/140/args.json", 1171, "5ec32047d303c6cbfdbc17d59f81ee08c635b939",
                "b83d9c47fd5817c816a163b709b93d51b873bfc0e4f59a6d0b2254cf486c2489"),
    PinnedFile("data/sportec_synced/J03WR9.parquet", 40194, "3dd7ba08f8b41926ffcff67cf73e610b03defaed",
                "3561a33ee5648f5ace3794f0997bcb3a1f37d73dff330f8fd7153e1e3b3b318c"),
    PinnedFile("LICENSE", 16726, "d0a1fa1482eea82e19510e7920cbe3a03e41f691",
                "3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04"),
    PinnedFile("inference.py", 45360, "91d1d7a031d28536f25683ea24e031a366f5f4f1",
                "a4815e8b4a77edb163ac99ad5785390597d3bbd63b5b906bb51375d3e403816d"),
    PinnedFile("dataset.py", 24214, "84c32f2a5c18847765ba27298b2f2891a747ab7b",
                "a6e9f45f9358882c5a73958563ff7aa650e2f66a0cb24c6f2908658c5a71724a"),
    PinnedFile("models/__init__.py", 0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    PinnedFile("models/utils.py", 10695, "07501789d2874e7b34337ee55d2ad35bdf35a77d",
                "511e69f52dc833941c922ebab84f981995a77a78d61d3cac26e555bc5f46f7b2"),
    PinnedFile("models/set_lstm.py", 18778, "fa56ffece4f53f0956b6e40f9b629b34a20bd181",
                "ad1690e14f7693d26e19f11fe549367dba45fb4f470bf70972259665b378e963"),
    PinnedFile("models/dynamic_sparse_crf.py", 15262, "63535cb76922baca7e7fe50f043c13d50c6b8b85",
                "8eb08e643886c3d986c110d46ebe5e5d4307a8553fa9ee1bc3bc93c1a9f85869"),
    PinnedFile("models/dynamic_dense_crf.py", 8022, "c26473c6da467fa23e32f08a34e19e15b7720198",
                "acd217ef153c60fc9a6c84f0b1ca4b72bef3a2ed132127e571b076539ba93ab1"),
    PinnedFile("models/static_sparse_crf.py", 13023, "924aa419ffefc96f395970f2e42602a37f659305",
                "bdad038a32a98335c5b8f8c89f697944e00b6e3e43a9fb49dd9b746614904639"),
    PinnedFile("models/static_dense_crf.py", 5783, "79ecb4f544548dbfbafbe0578083cc46cc7f4c2c",
                "0ba5b56d0a7caed8135a8d73303a7361332907cece3d872bc4ea589ef2e62385"),
    PinnedFile("models/gat_lstm.py", 12889, "80a003da9dfb34a2a583d705749260939dfeb786",
                "fd9f7d61a004b146fb86c2ed54cdfc56789c89a2e3bde768887044d1cb52b02f"),
    PinnedFile("models/transportmer.py", 9540, "f112cf60af888b3381147b310a7b5f2ff64c40e2",
                "8b42f862941118aaf8f8327961223d8a105d61b15bd99fc07944921854ef5fc8"),
    PinnedFile("set_transformer/attention.py", 3134, "af210973104f781291064095403169227cc48bca",
                "0f96a7c887bcf9040f91731479c6367e7abed60297f2d8224725009c6ebe25c3"),
    PinnedFile("set_transformer/blocks.py", 4333, "5ef8a90d14a44a7bdaaac3565d17addd72d4f472",
                "9f94aa29dec3d3f7431f82d75d9399799583964e8633325e7bc9cef7865b0611"),
    PinnedFile("set_transformer/model.py", 2172, "b9eb0269f2e85678721830edd873f09698fbb01f",
                "b072091111475b8928f5cedb482d712367613b30116551c3ef23505c49cc6060"),
    PinnedFile("datatools/config.py", 1372, "546d38588fd258c5cebaa90d3709dc157b3e022b",
                "4179ecaf6459a831f83676e433d4319cbaad0cfcf1e726eb0423526b9ad60db9"),
    PinnedFile("datatools/utils.py", 19465, "ec4df34ff5d871a6446978b14209087d87070af3",
                "5b346aa4ea277b73874772e3ba09bdddf4b74414a18f5f6ac4cec1f9e33a0ec0"),
    PinnedFile("datatools/postprocess.py", 20768, "712aeaaff375f6eed537d7ae80e3aaba04ed27f3",
                "8503a89fb93f1843ee37d56730ff86122e7d7bfd06ad5176d4a32324e6750f9a"),
    PinnedFile("datatools/matplotsoccer.py", 13731, "72588230133257b7d2954ca35ec566b384532f94",
                "5f3d3a045154e5480de24bbf46b0bf8daf63b2d4640e5441d7a3b31315a25cda"),
    PinnedFile("datatools/evaluate_events.py", 13798, "5b551d6435c185dc8ede224203af199ecb81e509",
                "adb1195419ae81ddc44f6e18bc344befb4aea68b37e410b82b1538b6fd6406df"),
    PinnedFile("datatools/preprocess.py", 17210, "322f193bd6676346f6384c93d20f376f32bd277f",
                "a11c305cb5dc4433cb42b2d2ee09fd4a701583b60141e8ce1f40825f67beb821"),
    PinnedFile("datatools/sportec_data.py", 22650, "5bb690991032036fdba2675bc9663159f5f382f6",
                "34f4bbf719dbcc2d6500e2016c1cc7d69b0a7bd3a4326f8a6f85e5f239104d38"),
    PinnedFile("datatools/match_data.py", 7435, "e84eef3960521d7018005549cfce106e91599b6b",
                "708f38dfa779d736eefd75f78182022d9d5c77e5d0cfd354c6764138af46a211"),
    PinnedFile("applications/team_poss.py", 5507, "76608daa78506dfb5f1cdcca0ba02a45c61ed25e",
                "630ad7931ae96593b902d195b0e7c589d5b27ac9f4bae38ffeb19edad7a560e0"),
)


class DownloadError(OSError):
    """A file could not be fetched, or what arrived is not the file that was pinned. The message says what to do."""


def git_blob_sha1(data):
    """The git blob SHA-1 of `data`: sha1(b"blob <len>\\0" + data), what `git hash-object` prints."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def url(pinned):
    return RAW_URL.format(repo=OWNER_REPO, commit=COMMIT, path=pinned.path)


def pathcrf_dir(data_dir=None):
    """<data dir>/external/pathcrf. data/ is gitignored, and so is everything under it."""
    return os.path.join(data_dir or games.DATA_DIR, "external", "pathcrf")


def repo_dir(data_dir=None):
    """Where the pinned copy of the repository lives: <data dir>/external/pathcrf/repo."""
    return os.path.join(pathcrf_dir(data_dir), "repo")


def _local(pinned, data_dir):
    return os.path.join(repo_dir(data_dir), *pinned.path.split("/"))


def _hashes(path):
    """(size, blob SHA-1, sha256) of the file at `path`, streamed."""
    size = os.path.getsize(path)
    blob = hashlib.sha1(b"blob %d\0" % size)
    plain = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            blob.update(chunk)
            plain.update(chunk)
    return size, blob.hexdigest(), plain.hexdigest()


def _status(pinned, data_dir):
    path = _local(pinned, data_dir)
    if not os.path.isfile(path):
        return "missing"
    size, blob, plain = _hashes(path)
    if size != pinned.size:
        return "size"
    if blob != pinned.blob_sha1:
        return "blob"
    if pinned.sha256 is not None and plain != pinned.sha256:
        return "sha256"
    return "ok"


def verify(data_dir=None, files=None):
    """{path: "ok" | "missing" | "size" | "blob" | "sha256"} for every pinned file (default FILES), read from disk."""
    return {pinned.path: _status(pinned, data_dir) for pinned in (FILES if files is None else files)}


def fetch(data_dir=None, opener=None, log=print, files=None):
    """Make every file ok on disk, then write manifest.json; DownloadError at the first that cannot be.

    `opener(request, timeout=...)` defaults to urllib.request.urlopen; the checks give it one that serves bodies from
    memory. A file already ok is not fetched again. The manifest holds what is on disk now, `{commit, files: {path:
    {size, blob_sha1, sha256}}}`, with the sha256 computed here for the files whose pin has none yet.
    """
    opener = opener or urllib.request.urlopen
    files = FILES if files is None else files
    for pinned in files:
        path = _local(pinned, data_dir)
        if _status(pinned, data_dir) == "ok":
            log("ok       {}".format(pinned.path))
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _stream(pinned, path, opener)
        log("fetched  {}  ({} bytes)".format(pinned.path, pinned.size))
    entries = {}
    for pinned in files:
        size, blob, plain = _hashes(_local(pinned, data_dir))
        entries[pinned.path] = {"size": size, "blob_sha1": blob, "sha256": plain}
    with open(os.path.join(pathcrf_dir(data_dir), "manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"commit": COMMIT, "files": entries}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _stream(pinned, path, opener):
    """URL -> <path>.part, hashed on the way, then renamed into place only if it is exactly the pinned file."""
    part = path + ".part"
    request = urllib.request.Request(url(pinned), headers={"User-Agent": USER_AGENT})
    blob = hashlib.sha1(b"blob %d\0" % pinned.size)
    plain = hashlib.sha256()
    received = 0
    try:
        with opener(request, timeout=TIMEOUT) as response, open(part, "wb") as handle:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                handle.write(chunk)
                blob.update(chunk)
                plain.update(chunk)
                received += len(chunk)
    except (OSError, http.client.HTTPException) as error:  # URLError, HTTPError, timeouts, resets, truncated bodies
        _discard(part)
        raise DownloadError("{}: {}: {}".format(pinned.path, type(error).__name__, error)) from error
    wrong = None
    if received != pinned.size:
        wrong = "got {} bytes, expected {}".format(received, pinned.size)
    elif blob.hexdigest() != pinned.blob_sha1:
        wrong = "git blob SHA-1 {}, expected {}".format(blob.hexdigest(), pinned.blob_sha1)
    elif pinned.sha256 is not None and plain.hexdigest() != pinned.sha256:
        wrong = "sha256 {}, expected {}".format(plain.hexdigest(), pinned.sha256)
    if wrong:
        _discard(part)
        raise DownloadError("{}: {}; it is not the file pinned at commit {}".format(pinned.path, wrong, COMMIT))
    os.replace(part, path)


def _discard(path):
    """Remove a partial file; one that is already gone, or held by another process, is not worth failing for."""
    try:
        os.remove(path)
    except OSError:
        pass


def main(argv=None, data_dir=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="only check the files on disk; fetch nothing")
    mode.add_argument("--record", action="store_true",
                      help="print `path sha256` of every verified file, to paste into FILES; fetch nothing")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR); the files go under "
                                           "DIR/external/pathcrf/repo")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or data_dir

    if args.verify or args.record:
        statuses = verify(data_dir)
        if args.record:
            for pinned in FILES:
                if statuses[pinned.path] in ("ok", "sha256"):
                    print("{} {}".format(pinned.path, _hashes(_local(pinned, data_dir))[2]))
        else:
            for pinned in FILES:
                print("{:<8} {}".format(statuses[pinned.path], pinned.path))
        ok = sum(status == "ok" for status in statuses.values())
    else:
        try:
            fetch(data_dir)
        except DownloadError as error:
            print("FAILED  {}".format(error))
            return 1
        ok = len(FILES)  # fetch() returns only once every file passes its pins
    print("\n{} of {} files ok under {}".format(ok, len(FILES), repo_dir(data_dir)))
    return 0 if ok == len(FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
