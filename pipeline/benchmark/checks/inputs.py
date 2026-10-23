"""Checks of the benchmark's inputs (pipeline/benchmark/inputs.py): a DFL match served to the pipeline as if ours.

The synthetic checks run on a DflMatch built here out of random numbers, in the real layout (one timeline, the halves
a gap apart, NaN positions and codes of -1 on the gap), so that every answer can be worked out without a download. The
question each one asks is whether a component written for BAR-ATH, which sees only `inputs`, gets from a benchmark match
what it gets from ours: the same fields, the same units, the same read-only arrays, and nothing it could not get from
ours, which is the truth. The one that reads a downloaded match is test_dfl_* (--dfl); the DFL files are public
(CC BY 4.0), so printing its aggregates is fine.
"""

import ast
import dataclasses
import functools
import json
import os
import tempfile
import types

import numpy as np

import games
from pipeline.benchmark import dfl, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark.checks.common import real_match, write_synthetic_dfl
from pipeline.inputs import atd_cache
from pipeline.selftest import expect_error, make_synthetic_bundle, synthetic_inputs, write_text

# Params
HOME, AWAY = 0, 1
ASSET_MESSAGE = ("{0}: the benchmark matches have no {0} -- the DFL open data comes without video or tagging (P01). "
                 "Check this source on the public window instead.")
# Rows where nearest_player and truth.nearest_team name different teams, per match: ties between the best home and the
# best away player, which their two ways of computing the distance break differently. In nearest_player's arithmetic the
# two squared distances are exactly equal and argmin takes the lower slot, which is the home player's; in
# nearest_team's they are 5e-15 m2 apart, with the away player nearer. TIE_DISTANCE (m2) is how close the two best
# distances must be for a row to count as a tie.
EXPECTED_TIES = {"J03WMX": (58523,)}
TIE_DISTANCE = 1e-9
# The folders of pipeline/ whose components see the inputs, and the package none of them may reach: it holds the
# loader of the truth (dfl.load_match), and MatchInputs names the match (inputs.game).
PIPELINE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPONENT_FOLDERS = ("evidence", "decode", "refine")
BENCHMARK_PACKAGE = "pipeline.benchmark"
# What an Atd is made of, as the BAR-ATH cache builds it: (dtype, dimensions) of each array.
ATD_LAYOUT = {"players_xy": (np.float64, 3), "players_speed": (np.float64, 2), "ball_xy": (np.float64, 2),
              "ball_speed": (np.float64, 1), "slot_team": (np.int8, 1)}


# -- helpers ----------------------------------------------------------------------------------------------------------

def random_match(match_id="RND001", seed=0, rows=(240, 200), gap=25, teams=(0, 0, 0, 1, 1, 1), pitch=(105.0, 68.0),
                 fps=25, writeable=False):
    """A DflMatch of random positions and flags: the halves are rows[0] and rows[1] long, `gap` rows apart.

    Positions wander a little outside the pitch and about a quarter of the players, and a fifth of the balls, are
    missing, as they are in the real data; nothing is a tie. The gap rows are what DFL's separator is: no ball, no
    players, codes -1. The arrays are read-only, as dfl.load_match gives them, unless `writeable`.
    """
    rng = np.random.default_rng([9, seed])
    n, players = sum(rows) + gap, len(teams)
    graded = ((1, 0, rows[0]), (2, rows[0] + gap, n))
    inside = np.zeros(n, dtype=bool)
    for _, start, stop in graded:
        inside[start:stop] = True

    xy = rng.uniform(-0.02, 1.02, size=(n, players, 2))
    xy[rng.uniform(size=(n, players)) < 0.25] = np.nan
    ball = rng.uniform(-0.02, 1.02, size=(n, 2))
    ball[rng.uniform(size=n) < 0.2] = np.nan
    arrays = {
        "frame_number": np.where(inside, np.arange(n), -1).astype(np.int64),
        "players_xy": xy, "players_speed": rng.uniform(0, 9, size=(n, players)),
        "player_team": np.array(teams, dtype=np.int8), "ball_xy": ball, "ball_z": rng.uniform(0, 3, size=n),
        "ball_speed": rng.uniform(0, 30, size=n),
        "ball_status": rng.integers(0, 2, size=n).astype(np.int8),
        "ball_possession": rng.integers(0, 2, size=n).astype(np.int8)}
    for name in ("players_xy", "players_speed", "ball_xy", "ball_z", "ball_speed"):
        arrays[name][~inside] = np.nan
    for name in ("ball_status", "ball_possession"):
        arrays[name][~inside] = -1
    for array in arrays.values():
        array.setflags(write=writeable)
    return dfl.DflMatch(match_id=match_id, pitch=pitch, fps=fps, graded=graded,
                        kickoffs={half: start for half, start, _ in graded}, report={}, **arrays)


def whole_positions(match):
    """`match` with a position missing in both coordinates wherever it was missing in one.

    The synthetic DFL match keeps a lone X or a lone Y on a few frames, to test the parser; an Atd may not (clean_atd
    refuses it), so a check that serves that match through MatchInputs serves this one.
    """
    xy, ball = np.array(match.players_xy), np.array(match.ball_xy)
    xy[np.isnan(xy).any(axis=-1)] = np.nan
    ball[np.isnan(ball).any(axis=-1)] = np.nan
    xy.setflags(write=False)
    ball.setflags(write=False)
    return dataclasses.replace(match, players_xy=xy, ball_xy=ball)


def truth_forms(match):
    """Every array a component could rebuild the answer from, or is the answer: the flags and the codes made of them.

    Each of them whole, and each of its graded halves (a copy of a half's flags is as good a leak as all of them).
    """
    status, possession = np.asarray(match.ball_status), np.asarray(match.ball_possession)
    codes, _ = truth.truth_codes(match)
    whole = {"ball_status": status, "ball_possession": possession, "alive": status == 1, "dead": status == 0,
             "home": possession == HOME, "away": possession == AWAY, "codes": codes}
    forms = dict(whole)
    for half, start, stop in match.graded:
        forms.update({"{}[half {}]".format(name, half): array[start:stop] for name, array in whole.items()})
    return forms


def held_by(value):
    """[(how it is reached, what is reached)] for what `value` holds, one level down; [] for what holds nothing.

    A container's items, a numpy array's base (a view is the array it looks into) and, for an array of objects, its
    elements; an object's attributes (its __dict__ and its __slots__, which is a dataclass's fields), and what a
    function, a bound method and a functools.partial close over. Classes, modules and plain values hold nothing here.
    """
    if isinstance(value, dict):
        return [("[{!r}]".format(key), item) for key, item in value.items()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [("[{}]".format(number), item) for number, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        held = [(".base", value.base)] if isinstance(value.base, np.ndarray) else []
        return held + ([("[{}]".format(number), item) for number, item in enumerate(value.flat)]
                       if value.dtype == object else [])
    if isinstance(value, functools.partial):
        return ([(".func", value.func)] + [(".args[{}]".format(number), item) for number, item in enumerate(value.args)]
                + [(".keywords[{!r}]".format(key), item) for key, item in (value.keywords or {}).items()])
    if isinstance(value, types.MethodType):
        return [(".__self__", value.__self__), (".__func__", value.__func__)]
    if isinstance(value, types.FunctionType):
        held = [(".defaults[{}]".format(number), item) for number, item in enumerate(value.__defaults__ or ())]
        held += [(".kwdefaults[{!r}]".format(key), item) for key, item in (value.__kwdefaults__ or {}).items()]
        for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
            try:
                held.append((".closure[{!r}]".format(name), cell.cell_contents))
            except ValueError:                                       # a cell that was never filled
                pass
        return held
    if isinstance(value, (str, bytes, int, float, complex, range, type, types.ModuleType, np.generic, np.dtype)):
        return []
    held = [(".{}".format(name), item) for name, item in getattr(value, "__dict__", {}).items()]
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        for name in ((slots,) if isinstance(slots, str) else slots):
            if hasattr(value, name):
                held.append((".{}".format(name), getattr(value, name)))
    return held


def leaks(inputs, match):
    """Where `inputs` holds the match or an array equal to its flags or its truth, however deep: [what, ...], sorted.

    Walks everything reachable from `inputs` (held_by), once each: a visited set keeps a cycle from running for ever.
    A DflMatch is a finding by itself, an array of another dtype than object is compared with every form of the truth
    (truth_forms), and a view is found through its base: `inputs.window.base == ball_possession`. What it cannot see is
    a copy of a stretch that is not a whole array or a whole half, and anything made at run time that it does not
    hold; test_components_import_nothing_from_the_benchmark closes the way to the match itself.
    """
    forms = truth_forms(match)
    found, seen, kept = [], set(), []
    pending = [("inputs", inputs)]
    while pending:
        path, value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        kept.append(value)                    # alive until the walk ends, so that no id is reused by a new object
        if isinstance(value, dfl.DflMatch):
            found.append(path)
            continue
        if isinstance(value, np.ndarray) and value.dtype != object:
            found += ["{} == {}".format(path, form) for form, array in forms.items()
                      if value.shape == array.shape and np.array_equal(value, array)]
        pending += [(path + how, item) for how, item in held_by(value)]
    return sorted(found)


# -- the contract -----------------------------------------------------------------------------------------------------

def test_match_inputs_contract():
    """MatchInputs is an Inputs for a benchmark match: the same fields the pipeline's components read, and no truth."""
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    from pipeline.inputs import Inputs
    match = random_match()
    atd = match_inputs.clean_atd(match)
    with tempfile.TemporaryDirectory() as folder:
        inputs = match_inputs.MatchInputs(match, atd, data_dir=folder)
        assert isinstance(inputs, Inputs)
        assert inputs.game == "RND001" and inputs.data_dir == folder and inputs.rebuild is False
        assert inputs.cache_dir == os.path.join(folder, "cache", "benchmark", "inputs", "RND001", "clean", "0"), \
            inputs.cache_dir
        assert inputs.n_frames == 465 and inputs.fps == 25 and inputs.pitch == (105.0, 68.0)
        assert inputs.atd_status == "benchmark"
        assert isinstance(inputs.graded, list) and inputs.graded == [(1, 0, 240), (2, 265, 465)], inputs.graded
        assert inputs.kickoffs == {1: 0, 2: 265} == match.kickoffs
        assert inputs.atd is atd and inputs.atd_manifest["profile"] == "clean" and inputs.atd_manifest["seed"] == 0
        assert os.listdir(folder) == [], "nothing is written: the inputs are served from memory"

    default = match_inputs.MatchInputs(match, atd)
    assert default.data_dir == games.DATA_DIR and (default.profile, default.seed) == ("clean", 0)
    assert default.cache_dir == os.path.join(games.DATA_DIR, "cache", "benchmark", "inputs", "RND001", "clean", "0")
    other = match_inputs.MatchInputs(match, atd, profile="degraded", seed=4)
    assert other.cache_dir.endswith(os.path.join("RND001", "degraded", "4")), other.cache_dir

    # The shape of the match, from the base class: 1 or 2 inside a half, 0 on the gap between them.
    halves = inputs.half_of_frame
    assert halves.dtype == np.int8 and halves.shape == (465,) and not halves.flags.writeable
    assert halves[0] == 1 and halves[239] == 1 and (halves[240:265] == 0).all() and halves[265] == 2
    assert halves[464] == 2
    assert inputs.graded_mask.sum() == 440 and not inputs.graded_mask[240:265].any()
    assert inputs.graded_mask[:240].all() and inputs.graded_mask[265:].all()
    # Metres are the match's own pitch, 105 x 68 here, so a distance in them is a DFL distance.
    assert np.array_equal(inputs.metres(np.array([[0.5, 0.5], [1.0, 0.25]])), np.array([[52.5, 34.0], [105.0, 17.0]]))
    wide = random_match(pitch=(100.0, 50.0))
    square = match_inputs.MatchInputs(wide, match_inputs.clean_atd(wide))
    assert np.array_equal(square.metres(np.array([[1.0, 1.0]])), np.array([[100.0, 50.0]]))

    # The pipeline's own nearest_player, on these inputs, names the team that truth.nearest_team does: home = 1.0 where
    # the nearest player is team 0, 0.0 where it is team 1, NaN (no opinion) where there is no ball or nobody.
    evidence = REGISTRY["nearest_player"](RunContext(inputs=inputs, seed=0))
    nearest = truth.nearest_team(match)
    expected = np.where(nearest == HOME, 1.0, np.where(nearest == AWAY, 0.0, np.nan))
    assert {int(team) for team in np.unique(nearest)} == {-1, HOME, AWAY}, "the match must reach every branch"
    differ = ~((evidence.home == expected) | (np.isnan(evidence.home) & np.isnan(expected)))
    assert not differ.any(), "{} frames differ, first at row {}".format(
        int(differ.sum()), int(np.flatnonzero(differ)[0]))
    assert np.isnan(evidence.alive).all(), "nearest_player has no opinion on alive vs dead"

    # What a BAR-ATH asset is and a benchmark match has none of.
    for asset in ("homography", "smart_tagging"):
        message = expect_error(match_inputs.AssetUnavailable, lambda: getattr(inputs, asset), asset, "P01")
        assert message == ASSET_MESSAGE.format(asset), message
    assert issubclass(match_inputs.AssetUnavailable, LookupError)

    # Read-only, like the BAR-ATH Atd and for the same reason: one component must not change what the next one reads.
    for name in ATD_LAYOUT:
        assert not getattr(inputs.atd, name).flags.writeable, name
    assert not inputs.atd.players_present.flags.writeable and not inputs.atd.ball_present.flags.writeable

    # And no truth: nothing it holds is the match, the flags, or a mask or code made of them (test_match_inputs_hold_no_
    # truth tests the detector itself).
    assert leaks(inputs, match) == [], leaks(inputs, match)


def test_match_inputs_hold_no_truth():
    """Evidence components see ctx.run.inputs, so the match's flags must not be reachable through it.

    Not the DflMatch, not the flags, not an alive/dead or home/away mask made of them, not the truth's codes: neither
    as an attribute of the inputs nor as a field of its Atd, even after the lazy properties have been read.
    """
    match = random_match()
    inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
    inputs.half_of_frame, inputs.graded_mask, inputs.atd.players_present, inputs.atd.ball_present
    assert leaks(inputs, match) == [], leaks(inputs, match)

    # The detector detects: a held match, a held flag array, a held mask and a held truth are each found.
    for name, value in (("match", match), ("status", np.array(match.ball_status)),
                        ("mask", match.ball_status == 1), ("codes", truth.truth_codes(match)[0]),
                        ("possession", np.array(match.ball_possession).astype(np.int16))):
        probe = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
        setattr(probe, name, value)
        assert leaks(probe, match), "{}: a held {} was not found".format(name, name)

    # Everything it does keep is a scalar of the match or the tracking.
    kept = set(vars(match_inputs.MatchInputs(match, match_inputs.clean_atd(match))))
    held_back = {name for name in kept if "ball" in name or "possession" in name or name in ("match", "truth")}
    assert not held_back, sorted(kept)


class Holder:
    """An object that keeps what it is given, so that a probe can keep only its bound method."""

    def __init__(self, held):
        self.held = held

    def get(self):
        return self.held


def planted_leaks(match):
    """{what is hidden and where: plant(probe)}: ways to leave the truth in a MatchInputs below its first level.

    The review of Task 7 planted six that vars(inputs) and vars(inputs.atd) did not see (a match in the manifest, the
    flags in a list, the codes in a dict, the match in a tuple, in a namespace, a slice of the possession); the others
    are of the same kind: the match or a flag array held in a container, in an object, behind a view, in an array of
    objects, or behind a function that closes over it.
    """
    codes, _ = truth.truth_codes(match)
    flags = np.array(match.ball_status)

    def in_atd(probe):                                    # an Atd is frozen: a field it does not declare is set by hand
        object.__setattr__(probe.atd, "debug", {"flags": [flags]})

    return {
        "a match in the manifest": lambda probe: probe.atd_manifest.update(debug=match),
        "the flags in a list": lambda probe: setattr(probe, "history", [flags]),
        "the codes in a dict": lambda probe: setattr(probe, "cache", {"codes": codes}),
        "the match in a tuple": lambda probe: setattr(probe, "pair", ("x", match)),
        "the match in a namespace": lambda probe: setattr(probe, "options", types.SimpleNamespace(match=match)),
        "a slice of the possession": lambda probe: setattr(probe, "window", match.ball_possession[10:50]),
        "a slice of a mask": lambda probe: setattr(probe, "window", (match.ball_status == 1)[10:50]),
        "a half of the flags, copied": lambda probe: setattr(probe, "half", np.array(match.ball_status[265:465])),
        "the flags five levels down": lambda probe: setattr(
            probe, "deep", {"a": [(types.SimpleNamespace(b={"c": flags}),)]}),
        "the flags in the Atd": in_atd,
        "the match in an array of objects": lambda probe: setattr(probe, "table", np.array([match], dtype=object)),
        "the match in a closure": lambda probe: setattr(probe, "lookup", lambda: match),
        "the match in a bound method": lambda probe: setattr(probe, "lookup", Holder(match).get),
        "the flags in a partial": lambda probe: setattr(probe, "lookup", functools.partial(dict, flags=flags)),
    }


def test_leak_detector_looks_at_every_depth():
    """leaks() finds the truth wherever the inputs keep it, and nothing where they do not.

    The first version read only vars(inputs) and vars(inputs.atd): a DflMatch in the manifest, the flags in a list or a
    dict, the match in a tuple or a namespace, a slice of the possession, all got through it. Each of them is planted
    here in a fresh MatchInputs and must be found, and the same inputs, clean, must give nothing, including when they
    hold cycles (a walk without a visited set would not end) and views of arrays that are not the truth (following
    ndarray.base must not turn the tracking into a leak).
    """
    match = random_match()

    def fresh():
        return match_inputs.MatchInputs(match, match_inputs.clean_atd(match))

    assert len(planted_leaks(match)) == 14
    for what, plant in planted_leaks(match).items():
        probe = fresh()
        plant(probe)
        assert leaks(probe, match), "{}: not found".format(what)

    clean = fresh()
    assert leaks(clean, match) == []
    loop = []
    loop.append(loop)
    clean.loop, clean.me, clean.mutual = loop, clean, {}
    clean.mutual["itself"] = clean.mutual
    assert leaks(clean, match) == [], "a cycle of containers or of the inputs itself is not a leak"
    clean.window, clean.speeds = match.players_xy[10:50], [match.ball_speed[:100], (match.ball_z,)]
    clean.names = {"home": "H", "away": "A", "sizes": (1, 2.5, None, True)}
    assert leaks(clean, match) == [], "views and containers of the tracking are not the truth"

    # The finding names where it is, so that a failure says what to remove.
    probe = fresh()
    planted_leaks(match)["the codes in a dict"](probe)
    assert leaks(probe, match) == ["inputs.cache['codes'] == codes"], leaks(probe, match)
    probe = fresh()
    planted_leaks(match)["a match in the manifest"](probe)
    assert leaks(probe, match) == ["inputs._atd_manifest['debug']"], leaks(probe, match)
    probe = fresh()
    planted_leaks(match)["a slice of the possession"](probe)
    assert leaks(probe, match) == ["inputs.window.base == ball_possession"], leaks(probe, match)


# -- no way round the guard: the components do not import the benchmark ----------------------------------------------

def benchmark_imports(source, package):
    """[(line, name)] of the ways the module text `source`, of the package `package`, reaches pipeline.benchmark.

    Import statements, absolute or relative (resolved against `package`, so `from ..benchmark import dfl` in
    pipeline.evidence is found, and so is `from .. import benchmark`), a string that is the package's name (what
    importlib.import_module and __import__ take), and a dotted attribute chain that starts with it (`pipeline.benchmark
    .dfl.load_match` after a plain `import pipeline`). A name built at run time is not seen: the walk of leaks() is
    the guard for that.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:max(len(parts) - node.level + 1, 0)] + ([node.module] if node.module else []))
            names = [base] + [base + "." + alias.name for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        elif isinstance(node, ast.Attribute):
            names = [ast.unparse(node)]
        else:
            continue
        found.update((node.lineno, name) for name in names
                     if name == BENCHMARK_PACKAGE or name.startswith(BENCHMARK_PACKAGE + "."))
    return sorted(found)


def test_import_scanner_finds_every_way_in():
    """benchmark_imports sees each way a component could reach the benchmark, and not the imports that are fine."""
    ways = {
        "import pipeline.benchmark.dfl": 1,
        "import pipeline.benchmark.dfl as d": 1,
        "from pipeline.benchmark import dfl": 1,
        "from pipeline.benchmark.dfl import load_match": 1,
        "from pipeline import benchmark": 1,
        "from pipeline import contracts, benchmark as b": 1,
        "from ..benchmark import dfl": 1,
        "from .. import benchmark": 1,
        "from ..benchmark.dfl import load_match as load": 1,
        "import importlib\nimportlib.import_module('pipeline.benchmark.dfl')": 2,
        "__import__('pipeline.benchmark')": 1,
        "import pipeline\npipeline.benchmark.dfl.load_match('J03WMX')": 2,
        "def f():\n    from pipeline.benchmark import dfl\n    return dfl": 2,
    }
    for source, line in ways.items():
        found = benchmark_imports(source, "pipeline.evidence")
        assert found and {number for number, _ in found} == {line}, (source, found)
    fine = ["import numpy as np", "from pipeline.contracts import Evidence, no_opinion",
            "from starter.baseline import HOME", "import pipeline.inputs", "from . import nearest_player",
            "from .majority_vote import compute", "from .. import contracts", "from ..inputs import atd_cache",
            "x = 'pipeline.benchmarks'",
            "\"\"\"The benchmark (pipeline.benchmark) is not imported here.\"\"\"",
            "benchmark = 3\nrun.benchmark.score", "from pipeline import benchmarking"]
    for source in fine:
        found = benchmark_imports(source, "pipeline.evidence")
        assert found == [], (source, found)
    # A relative import climbs from the module's own package: in pipeline.benchmark itself `from . import dfl` is the
    # benchmark, and in the package above it `from .benchmark import dfl` is.
    assert benchmark_imports("from . import dfl", "pipeline.benchmark") != []
    assert benchmark_imports("from .benchmark import dfl", "pipeline") != []
    assert benchmark_imports("from ... import benchmark", "pipeline.evidence") == [], "past the top: not the benchmark"


def test_components_import_nothing_from_the_benchmark():
    """No module under pipeline/evidence, decode or refine imports pipeline.benchmark.

    The guard on the inputs is that MatchInputs holds no truth, but it names the match (`inputs.game`), and a component
    that imported dfl.load_match could read the flags itself, whatever the inputs hold. This closes that route: the
    components are the code a config runs, and none of them needs the benchmark. Every module of the three folders is
    scanned (the folders must hold some, or this proves nothing).
    """
    scanned, found = 0, []
    for folder in COMPONENT_FOLDERS:
        top = os.path.join(PIPELINE_FOLDER, folder)
        for here, dirs, files in os.walk(top):
            dirs.sort()
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(here, name)
                relative = os.path.relpath(path, PIPELINE_FOLDER).replace(os.sep, "/")
                package = ".".join(["pipeline"] + relative.split("/")[:-1])
                with open(path, encoding="utf-8") as handle:
                    found += ["{}:{} {}".format(relative, line, what)
                              for line, what in benchmark_imports(handle.read(), package)]
                scanned += 1
    assert scanned >= 4, "only {} modules found under pipeline/{}".format(scanned, ", ".join(COMPONENT_FOLDERS))
    assert not found, "components that reach pipeline.benchmark (they could load the truth): {}".format(found)


# -- clean_atd --------------------------------------------------------------------------------------------------------

def test_clean_atd_has_the_layout_of_the_bar_ath_cache():
    """The same fields, dtypes and dimensions as the Atd that atd_cache builds, with slot_team the players' teams."""
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        real = synthetic_inputs(root).atd
    assert {name: (getattr(real, name).dtype.type, getattr(real, name).ndim) for name in atd_cache.ARRAYS} == ATD_LAYOUT

    match = random_match()
    atd = match_inputs.clean_atd(match)
    assert isinstance(atd, atd_cache.Atd) and atd_cache.ARRAYS == tuple(ATD_LAYOUT)
    for name, (dtype, ndim) in ATD_LAYOUT.items():
        array = getattr(atd, name)
        assert array.dtype == dtype and array.ndim == ndim and not array.flags.writeable, name
    assert (atd.n_frames, atd.n_slots) == (match.n_frames, match.n_players) == (465, 6)
    assert atd.players_xy.shape == (465, 6, 2) and atd.players_speed.shape == (465, 6)
    assert atd.slot_team.tolist() == match.player_team.tolist() == [0, 0, 0, 1, 1, 1]
    # What the match holds is not copied: the arrays are the match's own, already read-only.
    for name in ("players_xy", "players_speed", "ball_xy", "ball_speed"):
        assert getattr(atd, name) is getattr(match, name), name
    assert atd.slot_team is match.player_team
    assert np.array_equal(atd.ball_present, ~np.isnan(match.ball_xy[:, 0]))


def test_clean_atd_never_changes_the_match_and_casts_what_does_not_fit():
    loose = random_match(writeable=True)
    atd = match_inputs.clean_atd(loose)
    for name in atd_cache.ARRAYS:
        assert not getattr(atd, name).flags.writeable, "{} must be read-only".format(name)
    for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "player_team"):
        assert getattr(loose, name).flags.writeable, "{}: the match's own array was frozen".format(name)
    assert np.shares_memory(atd.players_xy, loose.players_xy), "a view, not a copy"

    # A match whose arrays are of other dtypes (a hand-built one) is cast to the Atd's.
    base = random_match()
    odd = dataclasses.replace(base, players_xy=np.array(base.players_xy, dtype=np.float32),
                              player_team=np.array(base.player_team, dtype=np.int64))
    cast = match_inputs.clean_atd(odd)
    assert cast.players_xy.dtype == np.float64 and cast.slot_team.dtype == np.int8
    assert cast.slot_team.tolist() == [0, 0, 0, 1, 1, 1]
    assert np.array_equal(cast.players_xy, odd.players_xy.astype(np.float64), equal_nan=True)
    assert all(not getattr(cast, name).flags.writeable for name in atd_cache.ARRAYS)


def test_clean_atd_refuses_a_position_with_one_coordinate():
    """x and y are missing together, the Atd's invariant: nearest_player masks on x alone, so a half-position would lie.

    atd_cache checks it on BAR-ATH's file (0 exceptions); a DFL match is checked here, for the hand-built ones (the 7
    downloaded matches have none). Players and ball, each with a missing x and a present y and the other way round.
    """
    base = random_match()
    assert match_inputs.clean_atd(base) is not None               # x and y missing together: as random_match has them
    rows = np.flatnonzero(~np.isnan(base.players_xy[:, 0, 0]) & ~np.isnan(base.ball_xy[:, 0]))[:3]
    assert len(rows) == 3

    def with_hole(array, index):
        damaged = np.array(array)
        damaged[index] = np.nan
        damaged.setflags(write=False)
        return damaged

    for field, holes in (("players_xy", ((rows[0], 0, 0), (rows[1], 0, 1))), ("ball_xy", ((rows[0], 0), (rows[1], 1)))):
        for hole in holes:
            odd = dataclasses.replace(base, **{field: with_hole(getattr(base, field), hole)})
            what = "slot-frames" if field == "players_xy" else "ball frames"
            message = expect_error(ValueError, lambda: match_inputs.clean_atd(odd), "RND001", "x and y", "1 " + what)
            assert "0 ball frames" in message or "0 slot-frames" in message, message
    # Two of one kind and one of the other are counted as such.
    players = with_hole(with_hole(base.players_xy, (rows[0], 0, 0)), (rows[1], 0, 1))
    both = dataclasses.replace(base, players_xy=players, ball_xy=with_hole(base.ball_xy, (rows[2], 0)))
    expect_error(ValueError, lambda: match_inputs.clean_atd(both), "RND001", "2 slot-frames", "1 ball frames")


def test_match_inputs_refuses_an_atd_of_another_length_and_names_what_was_run():
    match = random_match()
    short = random_match(rows=(100, 100))
    expect_error(ValueError, lambda: match_inputs.MatchInputs(match, match_inputs.clean_atd(short)), "RND001", "225",
                 "465")

    # The manifest is what a report says about the run: the match's source files, as the DFL cache recorded them, and
    # the profile and the seed of the tracking served.
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "source")
        os.makedirs(source)
        real = whole_positions(dfl.load_match("SYN001", data_dir=root, paths=write_synthetic_dfl(source)))
        with open(os.path.join(root, "cache", "dfl", "SYN001", "manifest.json"), encoding="utf-8") as handle:
            cache = json.load(handle)
        inputs = match_inputs.MatchInputs(real, match_inputs.clean_atd(real), profile="degraded", seed=3, data_dir=root)
        manifest = inputs.atd_manifest
        assert manifest["sources"] == cache["sources"] and sorted(manifest["sources"]) == ["information", "positions"]
        assert all(len(source["sha256"]) == 64 for source in manifest["sources"].values())
        assert (manifest["match_id"], manifest["profile"], manifest["seed"]) == ("SYN001", "degraded", 3), manifest
        assert manifest["cache_version"] == cache["cache_version"]
        json.dumps(manifest, sort_keys=True)                         # it goes into a report: JSON-ready
        # A match that was not read through the cache has no sources to name, and says so.
        bare = match_inputs.MatchInputs(match, match_inputs.clean_atd(match), data_dir=root)
        assert bare.atd_manifest["sources"] is None


def test_manifest_names_sources_only_of_the_match_it_serves():
    """The DFL cache's files are copied into the manifest only if that cache is of this match, and the reason if not.

    The cache is found by the match's id, so a hand-built match that carries a real id (or a cache copied under another
    one) would otherwise be reported as made from files it was not made from. The manifest's match_id, fps, pitch and
    graded must be the match's own; then `sources` is copied, and `sources_reason` is None. Otherwise `sources` (and
    the cache's version) is None and `sources_reason` says why: which of the four differ, or that there is no manifest,
    that it cannot be read, or that it records no sources.
    """
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "source")
        os.makedirs(source)
        real = whole_positions(dfl.load_match("SYN001", data_dir=root, paths=write_synthetic_dfl(source)))
        cache_root = os.path.join(root, "cache", "dfl")
        with open(os.path.join(cache_root, "SYN001", "manifest.json"), encoding="utf-8") as handle:
            cache = json.load(handle)

        def filed_under(match_id, text):
            folder = os.path.join(cache_root, match_id)
            os.makedirs(folder, exist_ok=True)
            write_text(folder, "manifest.json", text)

        def manifest_of(match):
            manifest = match_inputs.MatchInputs(match, match_inputs.clean_atd(match), data_dir=root).atd_manifest
            json.dumps(manifest, sort_keys=True)
            return manifest

        def untied(match, *, names, absent=()):
            manifest = manifest_of(match)
            assert manifest["sources"] is None and manifest["cache_version"] is None, manifest
            reason = manifest["sources_reason"]
            assert isinstance(reason, str) and match.match_id in reason, reason
            for name in names:
                assert name in reason, (name, reason)
            for name in absent:
                assert name not in reason, (name, reason)
            return reason

        # The match the cache describes: its sources are named, and there is nothing to explain.
        tied = manifest_of(real)
        assert tied["sources"] == cache["sources"] and tied["cache_version"] == cache["cache_version"]
        assert tied["sources_reason"] is None and (tied["match_id"], tied["profile"], tied["seed"]) == (
            "SYN001", "clean", 0)

        # A hand-built match under the real id: another pitch and other halves, the same rate and id.
        built = random_match(match_id="SYN001")
        assert (built.fps, built.match_id) == (real.fps, real.match_id) and built.pitch != real.pitch
        reason = untied(built, names=("pitch", "graded"), absent=("fps", "match_id"))
        assert "105.0" in reason and "104.5" in reason, reason
        # One difference at a time, each named and no other.
        untied(dataclasses.replace(real, fps=30), names=("fps",), absent=("pitch", "graded", "match_id"))
        untied(dataclasses.replace(real, pitch=(105.0, 68.0)), names=("pitch",), absent=("fps", "graded", "match_id"))
        moved = tuple((half, start + 1, stop) for half, start, stop in real.graded)
        untied(dataclasses.replace(real, graded=moved), names=("graded",), absent=("fps", "pitch", "match_id"))
        # A cache filed under another id (a folder that was copied): only its match_id differs.
        filed_under("RND001", json.dumps(cache))
        untied(dataclasses.replace(real, match_id="RND001"), names=("match_id",), absent=("fps", "pitch", "graded"))

        # A manifest that cannot be used says so: not JSON, JSON but not an object, no sources in it, no file at all.
        for match_id, text, said in (("BAD001", "{not json", "cannot be read"), ("BAD002", "[1, 2]", "cannot be read"),
                                     ("BAD003", json.dumps({key: value for key, value in cache.items()
                                                            if key != "sources"} | {"match_id": "BAD003"}),
                                      "no sources")):
            filed_under(match_id, text)
            untied(dataclasses.replace(real, match_id=match_id), names=(said,))
        untied(dataclasses.replace(real, match_id="NONE001"), names=("no DFL cache",))


# -- a real match -----------------------------------------------------------------------------------------------------

def test_dfl_match_inputs_serve_the_real_matches():
    """The 7 downloaded matches through MatchInputs: the fields, the read-only arrays, no truth, the nearest player.

    The pipeline's nearest_player on a DFL match must name the team that truth.nearest_team does on every frame, which
    is the home mapping (home = 1.0 where team 0 is nearest) proven on real data. The two compute the distance with
    different floating-point operations, so a frame where the best home player and the best away player are as far
    from the ball as each other can be broken either way. One such frame exists in the 7 matches (EXPECTED_TIES:
    J03WMX row 58523, where two players, one of each team, are at the same x and 0.34 m either side of the ball across
    the pitch). In nearest_player's arithmetic their squared distances are exactly equal and argmin takes the lower
    slot, the home player's; in nearest_team's they are 5e-15 m2 apart and the away player is nearer. The check says
    so and requires that every row where the two differ is such a tie, and that there are no others. The comparison
    is not loosened for any other frame. It also requires that nearest_player names a team on every graded row of
    every match (the rows it cannot answer are the ungraded gap between the halves), and prints the share of all rows.
    A hand-built match under each real id must not be reported as made from that match's files.
    """
    from pipeline.benchmark import sources
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    print("      {:<8} {:>9} {:>11} {:>11}".format("match", "rows", "answered", "tie rows"))
    for match_id in sources.MATCH_IDS:
        match = real_match(match_id)
        inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
        assert inputs.game == match_id and inputs.n_frames == match.n_frames and inputs.fps == 25
        assert inputs.graded == [tuple(block) for block in match.graded] and inputs.kickoffs == match.kickoffs
        assert inputs.pitch == match.pitch
        assert inputs.graded_mask.sum() == sum(stop - start for _, start, stop in match.graded)
        assert all(not getattr(inputs.atd, name).flags.writeable for name in atd_cache.ARRAYS)
        assert all(getattr(inputs.atd, name) is getattr(match, field) for name, field in (
            ("players_xy", "players_xy"), ("ball_xy", "ball_xy"), ("slot_team", "player_team"))), "copied, not served"
        assert leaks(inputs, match) == [], leaks(inputs, match)
        assert all(len(source["sha256"]) == 64 for source in inputs.atd_manifest["sources"].values())
        assert inputs.atd_manifest["sources_reason"] is None
        # A hand-built match under this real id is not made from this match's files: its halves are other, and the
        # manifest names no sources and says which of the cache's fields it differs in.
        built = random_match(match_id=match_id)
        manifest = match_inputs.MatchInputs(built, match_inputs.clean_atd(built)).atd_manifest
        assert manifest["sources"] is None and "graded" in manifest["sources_reason"], manifest

        evidence = REGISTRY["nearest_player"](RunContext(inputs=inputs, seed=0))
        nearest = truth.nearest_team(match)
        expected = np.where(nearest == HOME, 1.0, np.where(nearest == AWAY, 0.0, np.nan))
        differ = ~((evidence.home == expected) | (np.isnan(evidence.home) & np.isnan(expected)))
        rows = tuple(np.flatnonzero(differ).tolist())
        scale = np.array(match.pitch)
        for row in rows:
            distance = (((match.players_xy[row] - match.ball_xy[row]) * scale) ** 2).sum(axis=1)
            distance = np.where(np.isnan(distance), np.inf, distance)
            best = [float(distance[match.player_team == team].min()) for team in (HOME, AWAY)]
            assert abs(best[0] - best[1]) < TIE_DISTANCE, (
                "{}: row {} differs and is not a tie: the best home and away distances are {} and {} m2".format(
                    match_id, row, *best))
        answered = ~np.isnan(evidence.home)
        assert answered[inputs.graded_mask].all(), "{}: nearest_player has no opinion on {} graded rows".format(
            match_id, int((~answered & inputs.graded_mask).sum()))
        print("      {:<8} {:>9,} {:>11.1%} {:>11}".format(match_id, match.n_frames, answered.mean(), len(rows)))
        assert rows == EXPECTED_TIES.get(match_id, ()), "{}: the tied rows are {}, expected {}".format(
            match_id, rows, EXPECTED_TIES.get(match_id, ()))
