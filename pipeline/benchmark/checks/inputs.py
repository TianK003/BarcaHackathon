"""Checks of the benchmark's inputs (pipeline/benchmark/inputs.py): a DFL match served to the pipeline as if ours.

The synthetic checks run on a DflMatch built here out of random numbers, in the real layout (one timeline, the halves
a gap apart, NaN positions and codes of -1 on the gap), so that every answer can be worked out without a download. The
question each one asks is whether a component written for BAR-ATH, which sees only `inputs`, gets from a benchmark match
what it gets from ours: the same fields, the same units, the same read-only arrays, and nothing it could not get from
ours, which is the truth. The one that reads a downloaded match is test_dfl_* (--dfl); the DFL files are public
(CC BY 4.0), so printing its aggregates is fine.
"""

import ast
import collections
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
from pipeline.benchmark.checks.common import (HOME, AWAY, leaks, random_match, real_match, whole_positions,
                                              write_synthetic_dfl)
from pipeline.inputs import atd_cache
from pipeline.selftest import expect_error, make_synthetic_bundle, synthetic_inputs, write_text

# Params
ASSET_MESSAGE = ("{0}: the benchmark matches have no {0} -- the DFL open data comes without video or tagging (P01). "
                 "Check this source on the public window instead.")
# Rows where nearest_player and truth.nearest_team name different teams, per match: ties between the best home and the
# best away player, which their two ways of computing the distance break differently. In nearest_player's arithmetic the
# two squared distances are exactly equal and argmin takes the lower slot, which is the home player's; in
# nearest_team's they are 5e-15 m2 apart, with the away player nearer. TIE_DISTANCE (m2) is how close the two best
# distances must be for a row to count as a tie.
EXPECTED_TIES = {"J03WMX": (58523,)}
TIE_DISTANCE = 1e-9
EXPECTED_ROWS = 1002819                # the rows of the seven timelines together: A06 says 1 tie of them
# The folders of pipeline/ whose components see the inputs, and the package none of them may reach: it holds the
# loader of the truth (dfl.load_match), and MatchInputs names the match (inputs.game).
PIPELINE_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPONENT_FOLDERS = ("evidence", "decode", "refine")
BENCHMARK_PACKAGE = "pipeline.benchmark"
# What an Atd is made of, as the BAR-ATH cache builds it: (dtype, dimensions) of each array.
ATD_LAYOUT = {"players_xy": (np.float64, 3), "players_speed": (np.float64, 2), "ball_xy": (np.float64, 2),
              "ball_speed": (np.float64, 1), "slot_team": (np.int8, 1)}


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


class Keeper:
    """Holds what it is given under another name than Holder does, so that a route through it has another label."""

    def __init__(self, kept):
        self.kept = kept


class Slotted:
    """An object with __slots__ and no __dict__: what it keeps is not in vars()."""
    __slots__ = ("held",)

    def __init__(self, held):
        self.held = held


@dataclasses.dataclass(slots=True)
class SlottedBox:
    """A dataclass with slots=True: its fields are slots, and only then."""
    held: object


def planted_leaks(match):
    """{what is hidden and where: plant(probe)}: ways to leave the truth in a MatchInputs below its first level.

    The review of Task 7 planted six that vars(inputs) and vars(inputs.atd) did not see (a match in the manifest, the
    flags in a list, the codes in a dict, the match in a tuple, in a namespace, a slice of the possession); the others
    are of the same kind: the match or a flag array held in a container (a set, a deque, a mapping proxy, a list of
    numbers), in an object (with __slots__ or not), behind a view, in an array of objects, in an attribute of a
    function or a partial, or behind a function that closes over it.
    """
    codes, _ = truth.truth_codes(match)
    flags = np.array(match.ball_status)

    def in_atd(probe):                                    # an Atd is frozen: a field it does not declare is set by hand
        object.__setattr__(probe.atd, "debug", {"flags": [flags]})

    def on_function(probe):
        def lookup():
            return None
        lookup.match = match
        probe.lookup = lookup

    def on_partial(probe):
        lookup = functools.partial(dict)
        lookup.match = match
        probe.lookup = lookup

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
        "the match in an attribute of a function": on_function,
        "the match in an attribute of a partial": on_partial,
        "the match in an object with __slots__": lambda probe: setattr(probe, "boxed", Slotted(match)),
        "the match in a dataclass with slots": lambda probe: setattr(probe, "boxed", SlottedBox(match)),
        "the match in a set": lambda probe: setattr(probe, "members", frozenset({Holder(match)})),
        "the flags in a deque": lambda probe: setattr(probe, "queue", collections.deque([flags])),
        "the match behind a mapping proxy": lambda probe: setattr(
            probe, "view", types.MappingProxyType({"match": match})),
        "the flags as a list of numbers": lambda probe: setattr(probe, "numbers", match.ball_status.tolist()),
        "the codes as a tuple of numbers": lambda probe: setattr(probe, "numbers", tuple(codes.tolist())),
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

    assert len(planted_leaks(match)) == 23
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
    clean.counts, clean.names_of_frames = list(range(match.n_frames)), ["H"] * match.n_frames
    clean.members, clean.queue = frozenset({1, "x", (2, 3)}), collections.deque([match.players_xy[:5]])
    clean.slotted, clean.lookup = Slotted(match.ball_z), functools.partial(dict, speed=match.ball_speed)
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
    # A member of a set is named without a position (its order is the hash's, which is the id's for an object): the
    # same name in every run.
    probe = fresh()
    planted_leaks(match)["the match in a set"](probe)
    assert leaks(probe, match) == ["inputs.members{member}.held"], leaks(probe, match)
    for what, name in (("the match in an object with __slots__", "inputs.boxed.held"),
                       ("the match in a dataclass with slots", "inputs.boxed.held"),
                       ("the match in an attribute of a function", "inputs.lookup.match"),
                       ("the match in an attribute of a partial", "inputs.lookup.match"),
                       ("the flags as a list of numbers", "inputs.numbers == ball_status")):
        probe = fresh()
        planted_leaks(match)[what](probe)
        assert leaks(probe, match) == [name], (what, leaks(probe, match))


def test_leak_labels_do_not_depend_on_the_order_of_a_set():
    """An object that two members of a set both hold is named by the same route in every run.

    The members of a set are all called "{member}", and their order is the hash's, which for an object is its address.
    A walk that took the routes in the order it found them named the match after whichever member the set happened to
    list last: `inputs.members{member}.kept` in one run and `inputs.members{member}.held` in the next. The reviewer's
    case, frozenset({A(match), B(match)}), is built here with the two classes allocated in each order, 20 times each
    (a set's order follows the objects' addresses, which these allocations move around), and the label must be the
    smaller route, `.held`, every time.
    """
    match = random_match()
    labels = set()
    for holder_first in (True, False):
        for _ in range(20):
            probe = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
            members = (Holder(match), Keeper(match)) if holder_first else (Keeper(match), Holder(match))
            probe.members = frozenset(members)
            labels.add(tuple(leaks(probe, match)))
    assert labels == {("inputs.members{member}.held",)}, sorted(labels)

    # Two routes of different depth to one match: the smaller string wins, wherever each is in the container.
    probe = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
    probe.a_deep = {"x": [Holder(match)]}
    probe.b_shallow = match
    assert leaks(probe, match) == ["inputs.a_deep['x'][0].held"], leaks(probe, match)


def test_leak_detector_limits_are_as_documented():
    """What leaks() says it does not see, it does not see; and what it says it sees (a float copy), it does.

    The docstring is the guard's list of limits, so it must be true: a class-level attribute is not reached (held_by
    looks into an instance's __dict__ and slots, and classes hold nothing), a list of floats is not compared with the
    flags, and a float ndarray copy is.
    """
    match = random_match()

    class Probe(match_inputs.MatchInputs):
        pass
    Probe._last = match                                   # on the class, not on any instance
    probe = Probe(match, match_inputs.clean_atd(match))
    assert leaks(probe, match) == [], "a class-level attribute is a documented limit"
    probe.floats = match.ball_possession.astype(float).tolist()
    assert leaks(probe, match) == [], "a list of floats is a documented limit"
    probe.copy = match.ball_possession.astype(float)
    assert leaks(probe, match) == ["inputs.copy == ball_possession"], leaks(probe, match)


# -- no way round the guard: the components do not import the benchmark ----------------------------------------------

def benchmark_imports(source, package):
    """[(line, name)] of the ways the module text `source`, of the package `package`, reaches pipeline.benchmark.

    Import statements, absolute or relative (resolved against `package`, so `from ..benchmark import dfl` in
    pipeline.evidence is found, and so is `from .. import benchmark`), a string that is the package's name (what
    importlib.import_module and __import__ take), and a dotted attribute chain that starts with it (`pipeline.benchmark
    .dfl.load_match` after a plain `import pipeline`).

    It is for an accidental import, not a determined one. It does not see a name that is never whole in the source:
    `__import__("pipeline", fromlist=["benchmark"])`, `import_module(".benchmark", "pipeline")`,
    `getattr(pipeline, "benchmark")`, "pipeline." + "benchmark", a name read from a file or the environment; nor a
    module that loads another by path (`runpy`, `importlib.util`). The walk of leaks() is the second line against what
    such a component would then get from the inputs, and no more.
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


def scan_components(root=PIPELINE_FOLDER, folders=COMPONENT_FOLDERS):
    """({folder: how many modules were scanned}, ["folder/module.py:line name", ...]) for the component folders of root.

    Every .py file under each folder, however deep, is read as a module of the package that its path says
    (pipeline.evidence, pipeline.refine.deeper, ...). ValueError for a folder that is not there or has no
    __init__.py: a misspelt or renamed folder would scan nothing, and nothing found is what a clean scan gives too.
    """
    scanned, found = {}, []
    for folder in folders:
        top = os.path.join(root, folder)
        if not os.path.isfile(os.path.join(top, "__init__.py")):
            raise ValueError("{}: no folder of components with an __init__.py there; fix COMPONENT_FOLDERS, or the "
                             "folder".format(top))
        scanned[folder] = 0
        for here, dirs, files in os.walk(top):
            dirs.sort()
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(here, name)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                package = ".".join(["pipeline"] + relative.split("/")[:-1])
                with open(path, encoding="utf-8") as handle:
                    found += ["{}:{} {}".format(relative, line, what)
                              for line, what in benchmark_imports(handle.read(), package)]
                scanned[folder] += 1
    return scanned, found


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
    that imported dfl.load_match could read the flags itself, whatever the inputs hold. This closes the plain route to
    it: the components are the code a config runs, and none of them needs the benchmark. It is a guard against an
    accident and not a proof (benchmark_imports lists what it cannot see), and its reach is those three folders only: a
    component that imports another module of pipeline/, which itself reaches the benchmark, is not seen. Every module
    of the three folders is scanned, and each folder must be there with its __init__.py and at least that module
    scanned: a folder that is renamed, or spelt wrong in COMPONENT_FOLDERS, is an error and not a clean scan of
    nothing.
    """
    scanned, found = scan_components()
    assert list(scanned) == list(COMPONENT_FOLDERS) and all(count >= 1 for count in scanned.values()), scanned
    assert scanned["evidence"] >= 2 and scanned["decode"] >= 2, scanned       # the package's and the components' own
    assert not found, "components that reach pipeline.benchmark (they could load the truth): {}".format(found)


def test_component_scan_cannot_lose_a_folder_or_a_nested_module():
    """scan_components fails on a folder that is not there, and finds a violation however deep it is."""
    expect_error(ValueError, lambda: scan_components(folders=("evidence", "decode", "refin")), "refin", "__init__.py")
    expect_error(ValueError, lambda: scan_components(folders=("evidence", "nowhere")), "nowhere")
    with tempfile.TemporaryDirectory() as root:
        for folder in COMPONENT_FOLDERS:
            os.makedirs(os.path.join(root, folder, "deeper", "still"))
        for relative, text in (("evidence/__init__.py", ""), ("decode/__init__.py", ""), ("refine/__init__.py", ""),
                               ("evidence/clean.py", "from pipeline.contracts import Evidence\n"),
                               ("decode/viaparent.py", "import pipeline.benchmark.dfl\n"),
                               ("refine/deeper/still/buried.py", "from ....benchmark import dfl\n"),  # up to pipeline
                               ("refine/deeper/notes.txt", "from pipeline.benchmark import dfl\n")):
            write_text(os.path.join(root, os.path.dirname(relative)), os.path.basename(relative), text)
        scanned, found = scan_components(root)
        assert scanned == {"evidence": 2, "decode": 2, "refine": 2}, scanned    # the .txt is not a module
        assert found == ["decode/viaparent.py:1 pipeline.benchmark.dfl",
                         "refine/deeper/still/buried.py:1 pipeline.benchmark",
                         "refine/deeper/still/buried.py:1 pipeline.benchmark.dfl"], found


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


def test_atd_repr_shows_no_array():
    """repr(atd) holds no value of the tracking: a clean one's, a degraded one's and one of the BAR-ATH layout's.

    A dataclass shows its fields, and numpy shows the head and tail of an array: an Atd printed by a log line, an
    assertion message or a debugger would put our match's tracking (and, once degraded, where the ball was removed) in
    a log. Every array field has repr=False, and the first value of each array does not appear. The BAR-ATH layout is
    a synthetic bundle's Atd here, as the check reads nothing of the real match; pipeline/selftest.py checks the Atd
    and the Homography of pipeline/inputs/ the same way, for whoever changes that package.
    """
    from pipeline.benchmark import degrade
    from pipeline.benchmark.checks.common import made_up_calibration, tracked_match
    assert all(field.repr is False for field in dataclasses.fields(atd_cache.Atd)), [
        field.name for field in dataclasses.fields(atd_cache.Atd) if field.repr]
    match = tracked_match()
    degraded, _ = degrade.degrade(degrade.tracking_view(match), "gaps", made_up_calibration(), 0)
    atds = {"clean": match_inputs.clean_atd(match), "degraded": degraded}
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        atds["synthetic BAR-ATH layout"] = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)]).atd
        for label, atd in atds.items():
            shown = repr(atd)
            assert shown.startswith("Atd(") and "array" not in shown and len(shown) < 100, (label, shown)
            for name in ("players_xy", "ball_xy", "players_speed"):
                first = getattr(atd, name).flat[0]
                if np.isfinite(first):
                    assert "{:.4f}".format(first) not in shown, (label, name)


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
    graded and report (the parse's counts, without the "cache" entry that says how this load went) must be the match's
    own; then `sources` is copied, and `sources_reason` is None. Otherwise `sources` (and the cache's version) is None
    and `sources_reason` says why: which of the five differ, or that there is no manifest, that it cannot be read, or
    that it records no sources.
    """
    with tempfile.TemporaryDirectory() as root:
        source = os.path.join(root, "source")
        os.makedirs(source)
        paths = write_synthetic_dfl(source)
        real = whole_positions(dfl.load_match("SYN001", data_dir=root, paths=paths))
        reused = whole_positions(dfl.load_match("SYN001", data_dir=root, paths=paths))
        assert (real.report["cache"], reused.report["cache"]) == ("built", "reused")
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
        assert manifest_of(reused) == tied, "'reused' in the match's report is how its load went, not part of the key"

        # A hand-built match under the real id: another pitch and other halves, the same rate and id.
        built = random_match(match_id="SYN001")
        assert (built.fps, built.match_id) == (real.fps, real.match_id) and built.pitch != real.pitch
        reason = untied(built, names=("pitch", "graded", "report"), absent=("fps", "match_id"))
        assert "105.0" in reason and "104.5" in reason, reason
        # One difference at a time, each named and no other.
        untied(dataclasses.replace(real, fps=30), names=("fps",), absent=("pitch", "graded", "match_id"))
        untied(dataclasses.replace(real, pitch=(105.0, 68.0)), names=("pitch",), absent=("fps", "graded", "match_id"))
        moved = tuple((half, start + 1, stop) for half, start, stop in real.graded)
        untied(dataclasses.replace(real, graded=moved), names=("graded",), absent=("fps", "pitch", "match_id"))
        # The report (the parse's own counts) is part of the key as well: on the real caches the pitch and the rate are
        # the same for all seven, and a hand-built match differs from them in little else.
        untied(dataclasses.replace(real, report={}), names=("report",), absent=("fps", "pitch", "graded", "match_id"))
        counted = dict(real.report, ball_frames=real.report["ball_frames"] + 1)
        untied(dataclasses.replace(real, report=counted), names=("report",), absent=("fps", "pitch", "graded"))
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
    every match; the rows it cannot answer are exactly the GAP_ROWS (25) between the halves of each,
    and the seven timelines have EXPECTED_ROWS rows together (A06 quotes both). The share of all rows is printed.
    A hand-built match under each real id must not be reported as made from that match's files.
    """
    from pipeline.benchmark import sources
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    print("      {:<8} {:>9} {:>11} {:>11}".format("match", "rows", "answered", "tie rows"))
    total_rows = 0
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
        assert manifest["sources"] is None and all(
            name in manifest["sources_reason"] for name in ("graded", "report")), manifest

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
        assert int((~answered).sum()) == dfl.GAP_ROWS, "{}: {} rows unanswered, the gap is {}".format(
            match_id, int((~answered).sum()), dfl.GAP_ROWS)
        total_rows += match.n_frames
        print("      {:<8} {:>9,} {:>11.1%} {:>11}".format(match_id, match.n_frames, answered.mean(), len(rows)))
        assert rows == EXPECTED_TIES.get(match_id, ()), "{}: the tied rows are {}, expected {}".format(
            match_id, rows, EXPECTED_TIES.get(match_id, ()))
    print("      {:<8} {:>9,}   (the {} rows of each that nearest_player cannot answer are its gap between the "
          "halves)".format("all", total_rows, dfl.GAP_ROWS))
    assert total_rows == EXPECTED_ROWS, "the seven timelines have {:,} rows, A06 says {:,}".format(
        total_rows, EXPECTED_ROWS)
