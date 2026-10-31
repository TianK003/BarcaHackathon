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


# -- runner, torch-free helpers --------------------------------------------------------------------

def test_team_code_by_name_not_block():
    from pathcrf_lab import runner
    names = ["home_3", "home_pad_9", "away_0", "away_pad_10", "out_left", "out_right", "out_bottom", "out_top"]
    got = [runner.team_code(name) for name in names]
    assert got == [1, 1, 0, 0, 2, 2, 2, 2], "team codes by name are {}".format(got)
    try:
        runner.team_code("x")
    except ValueError as error:
        assert "'x'" in str(error), "the error must name the value: {}".format(error)
    else:
        raise AssertionError("a node name that is none of home_, away_, out_ must raise")


def test_codes_from_senders_silent():
    import numpy as np
    from pathcrf_lab import runner
    codes = runner.codes_from_senders([float("nan"), None, "home_1", "away_2", "out_top"])
    assert codes.dtype == np.int8, "dtype is {}, expected int8".format(codes.dtype)
    assert codes.tolist() == [-1, -1, 1, 0, 2], "codes are {}".format(codes.tolist())


def test_synthetic_window_has_their_layout():
    import numpy as np
    from pathcrf_lab import runner
    window = runner.synthetic_window()
    assert window.shape == (250, 22 * 6), "the window is {}, expected 250 frames of 22 x 6 columns".format(window.shape)
    assert not window.isna().any().any(), "the synthetic window must have no NaN: their inference skips such windows"
    for team in ("home", "away"):
        for k in range(11):
            for suffix in ("x", "y", "vx", "vy", "speed", "accel"):
                assert "{}_{}_{}".format(team, k, suffix) in window.columns, "missing {}_{}_{}".format(team, k, suffix)
    assert window[["home_0_x", "away_0_x"]].mean().tolist() == sorted(window[["home_0_x", "away_0_x"]].mean().tolist()), \
        "home_0 (left keeper) must sit at the lower x"
    assert np.array_equal(window.to_numpy(), runner.synthetic_window().to_numpy()), "the window must be seeded"


# -- adapter ---------------------------------------------------------------------------------------

def _grid(n_frames, n_slots):
    import numpy as np
    return np.full((n_frames, n_slots, 2), np.nan)


def _line(xy_m, slot, first, last, x, y=10.0, dx=0.0):
    """Slot `slot` present on frames first..last (inclusive) at x (+ dx per frame), y."""
    for k, frame in enumerate(range(first, last + 1)):
        xy_m[frame, slot] = (x + dx * k, y)


def _present(xy):
    import numpy as np
    return ~np.isnan(xy[:, :, 0])


def test_alive_runs_inside_blocks_only():
    import numpy as np
    from pathcrf_lab import adapter
    codes = np.full(200, 3, dtype=np.int8)
    codes[10:30] = 0       # H, 20 frames
    codes[30:40] = 1       # A directly after: one run, a change of team does not cut it
    codes[40:60] = 2       # D
    codes[60:75] = 1       # 15 frames: short
    codes[100:116] = 0     # 16 frames: kept
    codes[120:135] = 0     # 15 frames, then the block's edge at 130 cuts it
    codes[150:170] = 0     # outside every block: never in a run
    runs, short = adapter.alive_runs(codes, [(1, 0, 130), (2, 140, 150)])
    assert runs == [(10, 40), (100, 116)], "runs are {}".format(runs)
    assert short == [(60, 75), (120, 130)], "short runs are {}".format(short)


def test_split_run_near_equal():
    from pathcrf_lab import adapter

    def lengths(a, b):
        return [stop - start for start, stop in adapter.split_run(a, b)]

    assert lengths(0, 250) == [250] and lengths(0, 251) == [126, 125] and lengths(7, 1007) == [250] * 4 \
        and lengths(0, 16) == [16], "split_run lengths are off"
    spans = adapter.split_run(100, 351)
    assert spans[0][0] == 100 and spans[-1][1] == 351 and spans[0][1] == spans[1][0], "windows must tile the run"


def test_join_limits_are_inclusive():
    import numpy as np
    from pathcrf_lab import adapter

    def joined(gap, dist, other_team=False):
        xy = _grid(200, 2)
        _line(xy, 0, 0, 9, 0.0)
        _line(xy, 1, 9 + gap, 9 + gap + 5, dist)
        team = np.array([0, 1 if other_team else 0], dtype=np.int8)
        return adapter.join(adapter.pieces(_present(xy), 0, 200), xy, team)[1]

    assert joined(50, 5.0) == 1, "a gap of 50 frames and 5.0 m must join"
    assert joined(51, 5.0) == 0, "a gap of 51 frames must not join"
    assert joined(10, 5.0001) == 0, "5.0001 m must not join"
    assert joined(10, 3.0, other_team=True) == 0, "pieces of different metadata teams must never join"


def test_join_is_greedy_by_smallest_gap():
    import numpy as np
    from pathcrf_lab import adapter
    team = np.zeros(4, dtype=np.int8)
    xy = _grid(100, 4)
    _line(xy, 0, 0, 9, 0.0)         # a ends at 9
    _line(xy, 1, 12, 15, 1.0)       # b begins at 12 (gap 3)
    _line(xy, 2, 19, 30, 2.0)       # c begins at 19 (gap 10 from a, 4 from b)
    chains, n = adapter.join(adapter.pieces(_present(xy), 0, 100), xy, team)
    assert n == 2 and [tuple(p.slot for p in c) for c in chains] == [(0, 1, 2)], \
        "a-b (gap 3) is taken first, then b-c (gap 4): one chain a, b, c"
    two = _grid(100, 3)
    _line(two, 0, 0, 9, 0.0)
    _line(two, 1, 12, 20, 1.0)      # gap 3
    _line(two, 2, 19, 30, 2.0)      # gap 10: left alone, a's end is used once
    chains, n = adapter.join(adapter.pieces(_present(two), 0, 100), two, np.zeros(3, dtype=np.int8))
    assert n == 1 and sorted(tuple(p.slot for p in c) for c in chains) == [(0, 1), (2,)], \
        "one ending piece, two beginnings: the gap of 3 joins and the other stays a node"


def test_fill_linear_and_held_exact():
    import numpy as np
    from pathcrf_lab import adapter
    xy = _grid(30, 2)
    _line(xy, 0, 5, 9, 10.0)
    _line(xy, 1, 14, 19, 14.0)
    team = np.zeros(2, dtype=np.int8)
    chains, _ = adapter.join(adapter.pieces(_present(xy), 0, 30), xy, team)
    result = adapter.nodes(chains, xy, 0, 30, team)
    track = result.xy[:, 0].astype(np.float64)
    assert np.allclose(track[:5], (10.0, 10.0)) and np.allclose(track[20:], (14.0, 10.0)), "edges must be held"
    gap = track[10:14, 0]
    assert np.allclose(gap, 10.0 + 4.0 * np.arange(1, 5) / 5.0), "the gap must be linear, got {}".format(gap)
    assert not np.isnan(track).any(), "a kept node has a NaN"
    assert result.flags["held_frames"] == 5 + 4 + 10, "filled frames {}".format(result.flags["held_frames"])


def test_over_eleven_drops_the_fewest_frames():
    import numpy as np
    from pathcrf_lab import adapter
    xy = _grid(100, 12)
    for slot in range(12):  # slots 5 and 9 are seen for 41 frames only: the tie for the fewest
        _line(xy, slot, 0, 40 if slot in (5, 9) else 99, 3.0 * slot)
    team = np.zeros(12, dtype=np.int8)
    chains, _ = adapter.join(adapter.pieces(_present(xy), 0, 100), xy, team)
    result = adapter.nodes(chains, xy, 0, 100, team)
    assert result.flags["dropped"] == 1, "exactly one chain must be dropped, got {}".format(result.flags["dropped"])
    first = result.xy[0, :11, 0].astype(np.float64).tolist()
    assert any(abs(v - 15.0) < 1e-4 for v in first) and not any(abs(v - 27.0) < 1e-4 for v in first), \
        "on a tie of the fewest frames the higher slot (9) is dropped and slot 5 kept"


def _atd(n, slots, teams):
    from types import SimpleNamespace
    import numpy as np
    return SimpleNamespace(players_xy=np.full((n, slots, 2), np.nan), slot_team=np.array(teams, dtype=np.int8),
                           ball_present=np.zeros(n, dtype=bool))


def test_wrong_team_slot_stays_in_its_metadata_team():
    from pathcrf_lab import adapter
    atd = _atd(60, 23, [0] * 12 + [1] * 11)   # slot 11 is a team-0 slot, here among team 1's players
    for slot in range(23):
        atd.players_xy[:, slot] = (0.1 + 0.03 * slot, 0.5)
    atd.players_xy[:, 11] = (0.9, 0.5)
    result = adapter.window_nodes(atd, 0, 60)
    names = result.names.tolist()
    assert sum(n.startswith("home_") for n in names) == 11 and sum(n.startswith("away_") for n in names) == 11, \
        "the 12 team-0 slots keep 11 and the 11 team-1 slots all stay: {}".format(names)
    assert result.flags["over11"] == 1 and result.flags["dropped"] == 1, "12 home pieces: over11 and one dropped"


def test_padding_when_fewer_than_eleven():
    import numpy as np
    from pathcrf_lab import adapter
    atd = _atd(40, 20, [0] * 11 + [1] * 9)
    for slot in range(20):
        atd.players_xy[:, slot] = (0.1 + 0.04 * slot, 0.5)
    result = adapter.window_nodes(atd, 0, 40)
    assert result.flags["padded_away"] == 2 and result.flags["padded_home"] == 0, "padding {}".format(result.flags)
    assert (result.team[20:] == -1).all() and (result.names[20:] == "").all() and np.isnan(result.xy[:, 20:]).all(), \
        "rows 20 and 21 must be unused"


def test_metres_are_pathcrfs_pitch():
    import numpy as np
    from pathcrf_lab import adapter
    atd = _atd(20, 1, [0])
    atd.players_xy[:, 0] = (0.5, 1.0)
    got = adapter.window_nodes(atd, 0, 20).xy[0, 0]
    assert np.allclose(got, (52.5, 68.0)), "(0.5, 1.0) is {} m, expected (52.5, 68.0)".format(got)


def test_adapter_reads_no_label():
    import inspect
    from pathcrf_lab import adapter
    source = inspect.getsource(adapter).lower()
    for word in ("truth", "ball_status", "ball_possession"):
        assert word not in source, "adapter.py mentions {}: it must read no label".format(word)
    for name, function in inspect.getmembers(adapter, inspect.isfunction):
        if not name.startswith("_") and function.__module__ == adapter.__name__:
            assert not any("truth" in p for p in inspect.signature(function).parameters), \
                "{} takes a parameter named for the truth".format(name)


def test_adapter_is_deterministic():
    import numpy as np
    from pathcrf_lab import adapter
    rng = np.random.default_rng([3, 4])
    atd = _atd(120, 24, [0] * 12 + [1] * 12)
    atd.players_xy[:] = rng.uniform(0.2, 0.8, size=atd.players_xy.shape)
    atd.players_xy[rng.random(atd.players_xy.shape[:2]) < 0.2] = np.nan
    first = adapter.window_nodes(atd, 0, 120)
    again = adapter.window_nodes(atd, 0, 120)
    assert first.xy.tobytes() == again.xy.tobytes() and first.names.tolist() == again.names.tolist() \
        and first.flags == again.flags, "the same inputs gave different nodes"


# -- adapter, prepare ------------------------------------------------------------------------------

def _inputs(n=600):
    from types import SimpleNamespace
    atd = _atd(n, 22, [0] * 11 + [1] * 11)
    for slot in range(22):
        atd.players_xy[:, slot] = (0.1 + 0.04 * slot, 0.5)
    atd.ball_present[::2] = True
    return SimpleNamespace(atd=atd, graded=[(1, 100, 400), (2, 450, 590)], n_frames=n)


def test_prepare_arrays_shapes_and_window_of_row():
    import numpy as np
    from pathcrf_lab import adapter
    codes = np.full(600, 3, dtype=np.int8)
    codes[120:420] = 0          # inside block 1 (100-400): 120..399, 280 frames, so 2 windows
    codes[130:140] = 2          # D: splits it into (120, 130) short and (140, 400) of 260 frames, so 2 windows
    codes[460:470] = 1          # 10 frames in block 2: short
    windows, frames, counts = adapter.prepare_arrays(_inputs(), codes, np.full(600, 0.5))
    n = len(windows["win_start"])
    assert n == 2, "expected 2 windows, got {}".format(n)
    assert windows["node_xy"].shape == (int((windows["win_stop"] - windows["win_start"]).sum()), 22, 2), "node_xy shape"
    assert windows["node_team"].shape == (n, 22) and windows["node_names"].shape == (n, 22) \
        and windows["flags"].shape == (n, 6), "per-window array shapes"
    row = frames["window_of_row"]
    covered = np.zeros(600, dtype=bool)
    for w, (a, b) in enumerate(zip(windows["win_start"], windows["win_stop"])):
        assert (row[a:b] == w).all(), "rows of window {} do not map to it".format(w)
        covered[a:b] = True
    assert ((row == -1) == ~covered).all(), "window_of_row is -1 exactly outside the windows"
    assert counts["short_runs"] == 2 and counts["short_frames"] == 20, "short runs: {}".format(counts)


def test_check_pinned():
    import tempfile
    from pathcrf_lab import adapter
    path = os.path.join(ROOT, adapter.MAIN_TOML)
    adapter.check_pinned(path, adapter.MAIN_TOML_SHA256)
    with open(path, "rb") as handle:
        lf = handle.read().replace(b"\r\n", b"\n")
    with tempfile.TemporaryDirectory() as folder:
        crlf = os.path.join(folder, "crlf.toml")
        with open(crlf, "wb") as handle:
            handle.write(lf.replace(b"\n", b"\r\n"))
        adapter.check_pinned(crlf, adapter.MAIN_TOML_SHA256)
        bad = os.path.join(folder, "bad.toml")
        with open(bad, "wb") as handle:
            handle.write(lf + b" ")
        try:
            adapter.check_pinned(bad, adapter.MAIN_TOML_SHA256)
        except ValueError as error:
            assert adapter.MAIN_TOML_SHA256 in str(error) and "bad.toml" in str(error), "the error must name both"
        else:
            raise AssertionError("a changed file was accepted")


def test_prepare_counts_are_aggregates():
    import numpy as np
    from pathcrf_lab import adapter
    codes = np.full(600, 3, dtype=np.int8)
    codes[120:300] = 0
    _, _, counts = adapter.prepare_arrays(_inputs(), codes, np.full(600, 0.5))
    assert all(isinstance(v, (int, float, str)) for v in counts.values()), "counts hold only ints, floats, strs"


# -- runner, degraded windows ----------------------------------------------------------------------

def _window_arrays(frames=60):
    """(xy, team, names, rows) of a synthetic window: 11 home nodes on the right, 10 away on the left, one unused."""
    import numpy as np
    rng = np.random.default_rng([8, 1])
    xy = np.full((frames, 22, 2), np.nan, dtype=np.float32)
    team = np.full(22, -1, dtype=np.int8)
    names = np.array([""] * 22, dtype="<U8")
    for row in range(21):
        side = 0 if row < 11 else 1
        base = rng.uniform((60, 5), (100, 63)) if side == 0 else rng.uniform((5, 5), (45, 63))
        xy[:, row] = base + np.cumsum(rng.normal(0, 0.05, size=(frames, 2)), axis=0)
        team[row] = side
        names[row] = "{}_{}".format("home" if side == 0 else "away", row if side == 0 else row - 11)
    team[21] = -1
    return xy, team, names, np.arange(1000, 1000 + frames)


def test_window_frame_layout():
    from pathcrf_lab import runner
    xy, team, names, rows = _window_arrays()
    frame = runner.window_frame(xy, team, names, rows)
    assert frame["frame_id"].tolist() == rows.tolist(), "frame_id must be the rows"
    for column, value in (("period_id", 1), ("phase_id", 1), ("episode_id", 1)):
        assert (frame[column] == value).all(), "{} must be {}".format(column, value)
    assert (frame["ball_state"] == "alive").all() and frame["ball_owning_team_id"].isna().all(), "ball state"
    assert frame[["ball_x", "ball_y"]].isna().all().all(), "the ball is unknown: NaN"
    assert abs(frame["timestamp"].iloc[25] - frame["timestamp"].iloc[0] - 1.0) < 1e-9, "timestamp is rows / 25"
    used = [c for c in frame.columns if c[:5] in ("home_", "away_")]
    assert sorted(used) == sorted("{}_{}".format(n, a) for n in names[:21] for a in "xy"), "one x and y per used node"
    assert not frame[used].isna().any().any(), "no NaN in a used node column"
    assert not any(c.startswith("away_10") for c in frame.columns), "no column for the unused node"
