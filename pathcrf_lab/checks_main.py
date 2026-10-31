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


# -- sources ---------------------------------------------------------------------------------------

def _fake_opener(bodies):
    """An opener that serves bodies[url] from memory, as urlopen would: a context manager with read(n)."""
    import io

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    def opener(request, timeout=None):
        return Response(bodies[request.full_url])

    return opener


def test_git_blob_sha1_of_hello():
    from pathcrf_lab import sources
    assert sources.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a", \
        "git_blob_sha1 differs from `git hash-object` on b'hello\n'"


def test_registry_is_the_commit():
    from pathcrf_lab import sources
    assert re.fullmatch(r"[0-9a-f]{40}", sources.COMMIT) and sources.COMMIT == \
        "1b7c6f85a6b78c92e4543fde72ca2b86a933f997", "COMMIT is not the pinned one"
    paths = [pinned.path for pinned in sources.FILES]
    assert len(paths) == 27 and len(set(paths)) == 27, "expected 27 unique pinned paths, got {}".format(len(paths))
    checkpoint = {pinned.path: pinned for pinned in sources.FILES}["saved/140/model/state_dict_best_acc.pt"]
    assert checkpoint.size == 19325114 and checkpoint.blob_sha1.startswith("f8e48952"), "the checkpoint pin is off"
    for pinned in sources.FILES:
        assert re.fullmatch(r"[0-9a-f]{40}", pinned.blob_sha1), "{} has no 40-hex blob SHA-1".format(pinned.path)
        link = sources.url(pinned)
        assert sources.COMMIT in link and "/main/" not in link and "/HEAD/" not in link, \
            "{} is not pinned by commit: {}".format(pinned.path, link)


def test_fetch_keeps_only_verified_bytes():
    import tempfile
    from pathcrf_lab import sources
    good = b"print('pinned')\n"
    pinned = sources.PinnedFile("pkg/good.py", len(good), sources.git_blob_sha1(good))
    bad = sources.PinnedFile("pkg/bad.py", len(good), sources.git_blob_sha1(good))
    bodies = {sources.url(pinned): good, sources.url(bad): b"print('other!')\n"}  # same size, other content
    assert len(bodies[sources.url(bad)]) == len(good), "the wrong body must be of the right size"
    with tempfile.TemporaryDirectory() as folder:
        sources.fetch(folder, opener=_fake_opener(bodies), log=lambda *_: None, files=(pinned,))
        placed = os.path.join(sources.repo_dir(folder), "pkg", "good.py")
        with open(placed, "rb") as handle:
            assert handle.read() == good, "the verified body was not put in place"
        import json
        with open(os.path.join(sources.pathcrf_dir(folder), "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        import hashlib
        assert manifest["commit"] == sources.COMMIT and \
            manifest["files"]["pkg/good.py"]["sha256"] == hashlib.sha256(good).hexdigest(), \
            "the manifest lacks the commit or the file's sha256"
        try:
            sources.fetch(folder, opener=_fake_opener(bodies), log=lambda *_: None, files=(bad,))
        except sources.DownloadError:
            pass
        else:
            raise AssertionError("a body of the right size and the wrong content was accepted")
        wrong = os.path.join(sources.repo_dir(folder), "pkg", "bad.py")
        assert not os.path.exists(wrong) and not os.path.exists(wrong + ".part"), \
            "a rejected download left a file or its .part behind"


def test_recorded_sha256_is_enforced():
    import tempfile
    from pathcrf_lab import sources
    body = b"x = 1\n"
    pinned = sources.PinnedFile("pkg/a.py", len(body), sources.git_blob_sha1(body))
    with tempfile.TemporaryDirectory() as folder:
        sources.fetch(folder, opener=_fake_opener({sources.url(pinned): body}), log=lambda *_: None, files=(pinned,))
        assert sources.verify(folder, files=(pinned,)) == {"pkg/a.py": "ok"}, "an unrecorded sha256 must not fail"
        wrong = replace_sha256(pinned, "0" * 64)
        assert sources.verify(folder, files=(wrong,)) == {"pkg/a.py": "sha256"}, \
            "a recorded sha256 that differs must be reported as 'sha256'"
        os.remove(os.path.join(sources.repo_dir(folder), "pkg", "a.py"))
        assert sources.verify(folder, files=(pinned,)) == {"pkg/a.py": "missing"}, "a missing file must be 'missing'"


def replace_sha256(pinned, sha256):
    from dataclasses import replace
    return replace(pinned, sha256=sha256)
