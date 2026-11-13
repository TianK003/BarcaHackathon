"""Self-checks for the pipeline: contracts, configs, and equivalence with the organisers' baseline.

    python pipeline/selftest.py           # synthetic checks only -- seconds, no data needed
    python pipeline/selftest.py --data    # + the real match: all four committed baselines, byte for byte

Every check prints PASS or FAIL. Nothing from the match data is ever printed -- counts, frame numbers and hashes,
never a label or a coordinate (CLAUDE.md, hard rules). The exit code is 1 if anything failed.

Two kinds of check live here, and the second is the point.

* Contract checks: each stage rejects malformed input with a sentence instead of producing a wrong file.
* Equivalence checks: the pipeline's first components are the baseline's two steps, rebuilt so that other sources
  can plug in beside them. The oracle is the organisers' own code in starter/baseline.py -- run on synthetic
  tracking here, and with --data on the real match, where the output must equal the committed baseline CSVs byte
  for byte. If that ever fails, something upstream of the decoder moved: alignment, parsing, team assignment or
  the writer.

No test framework, on purpose: the repository has none (CLAUDE.md), and a plain script runs in the bare team venv.
"""

import argparse
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if __name__ == "__main__":
    # Run by path, Python put pipeline/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no folder in here can shadow a top-level module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import pipeline  # noqa: E402,F401  (puts evaluation/ on sys.path)


# -- helpers ----------------------------------------------------------------------------------------------------------

def expect_error(error_type, function, *fragments):
    """Call function, insist it raises error_type, and that the message names every fragment.

    Checking the message and not just the type is deliberate: these errors are read by a teammate at 23:00 the night
    before the deadline, and "ValueError" alone tells them nothing.
    """
    try:
        function()
    except error_type as error:
        message = str(error)
        missing = [fragment for fragment in fragments if fragment not in message]
        assert not missing, "message lacks {}: {}".format(missing, message)
        return message
    raise AssertionError("expected {}, nothing was raised".format(error_type.__name__))


def write_text(folder, name, text):
    """Write a small text file with LF endings and return its path."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def fake_registries():
    """Stand-in components with known signatures, so the config checks do not depend on the real stages."""
    def fake_source(run):
        return None

    def fake_vote(ctx, *, half_width, fallback="D"):
        return None

    def fake_trim(ctx, decoded, *, frames=1):
        return decoded

    return {"evidence": {"fake_source": fake_source},
            "decode": {"fake_vote": fake_vote},
            "refine": {"fake_trim": fake_trim}}


MINIMAL_CONFIG = """
description = "test"

[[evidence]]
kind = "fake_source"

[decoder]
kind = "fake_vote"
half_width = 3
"""


def load_test_config(text, name="trial.toml"):
    """load_config on a throwaway file, against the fake registries."""
    from pipeline.config import load_config
    with tempfile.TemporaryDirectory() as folder:
        return load_config(write_text(folder, name, text), registries=fake_registries())


# -- contracts --------------------------------------------------------------------------------------------------------

def test_check_evidence_rejects_wrong_length():
    from pipeline.contracts import Evidence, check_evidence, no_opinion
    evidence = Evidence(alive=no_opinion(11), home=no_opinion(11), name="shape_probe")
    expect_error(ValueError, lambda: check_evidence(evidence, 10), "shape_probe", "11", "10")


def test_check_evidence_rejects_out_of_range():
    from pipeline.contracts import Evidence, check_evidence, no_opinion
    home = no_opinion(10)
    home[3] = 1.5
    evidence = Evidence(alive=no_opinion(10), home=home, name="range_probe")
    expect_error(ValueError, lambda: check_evidence(evidence, 10), "range_probe", "home", "[0, 1]")


def test_check_evidence_rejects_wrong_dtype():
    from pipeline.contracts import Evidence, check_evidence, no_opinion
    evidence = Evidence(alive=no_opinion(10), home=np.zeros(10, dtype=np.float32), name="dtype_probe")
    expect_error(ValueError, lambda: check_evidence(evidence, 10), "dtype_probe", "float64")


def test_check_evidence_accepts_all_nan_channel():
    from pipeline.contracts import Evidence, check_evidence, no_opinion
    home = np.zeros(10)
    home[::2] = 1.0
    check_evidence(Evidence(alive=no_opinion(10), home=home, name="ok"), 10)


def test_check_decoded_rejects_bad_codes():
    from pipeline.contracts import Block, Decoded, check_decoded
    labels = np.zeros(5, dtype=np.int8)
    labels[2] = 7
    expect_error(ValueError, lambda: check_decoded(Decoded(labels=labels), Block(1, 10, 15), "probe"),
                 "probe", "7")


# -- config -----------------------------------------------------------------------------------------------------------

def test_config_defaults():
    config = load_test_config(MINIMAL_CONFIG, name="my_trial.toml")
    assert config.stem == "my_trial", config.stem
    assert config.scope == "halves", config.scope
    assert config.seed == 0, config.seed
    assert len(config.evidence) == 1 and config.evidence[0].name == "fake_source"
    assert config.evidence[0].weight == 1.0
    assert config.decoder.params == {"half_width": 3}, config.decoder.params
    assert config.refine == ()


def test_config_unknown_top_level_key():
    from pipeline.config import ConfigError
    expect_error(ConfigError, lambda: load_test_config(MINIMAL_CONFIG + "\nsede = 1\n"), "sede")


def test_config_unknown_kind():
    from pipeline.config import ConfigError
    text = MINIMAL_CONFIG.replace('kind = "fake_source"', 'kind = "no_such_source"')
    expect_error(ConfigError, lambda: load_test_config(text), "no_such_source", "fake_source")


def test_config_bad_parameter():
    from pipeline.config import ConfigError
    text = MINIMAL_CONFIG.replace("half_width = 3", "half_widht = 3")
    expect_error(ConfigError, lambda: load_test_config(text), "fake_vote", "half_widht", "half_width")


def test_config_duplicate_evidence_name():
    from pipeline.config import ConfigError
    text = MINIMAL_CONFIG + '\n[[evidence]]\nkind = "fake_source"\n'
    expect_error(ConfigError, lambda: load_test_config(text), "fake_source", "twice")


def test_config_bad_scope():
    from pipeline.config import ConfigError
    expect_error(ConfigError, lambda: load_test_config('scope = "match"\n' + MINIMAL_CONFIG), "match", "halves")


# The sources of REGISTRY that need an asset only our match has. Add the next such source here.
MATCH_ONLY_KINDS = {"smart_tagging"}


def test_match_only_reads_the_module_flag():
    """match_only(kind) is the MATCH_ONLY of the module that defines the source, False where it declares none.

    A source that needs an asset only our match has (the Smart Tagging, the whistle) says so once, in its own module,
    and the benchmark runner reads it from there: the flag is not repeated in a config or in a list that could drift.
    """
    import types
    from pipeline import evidence
    for kind in evidence.REGISTRY:
        assert evidence.match_only(kind) is (kind in MATCH_ONLY_KINDS), (
            "only " + ", ".join(sorted(MATCH_ONLY_KINDS)) + " of today's REGISTRY is match-only: " + kind)

    module = types.ModuleType("pipeline.evidence._probe_match_only")
    module.MATCH_ONLY = True

    def compute(run):
        return None

    compute.__module__ = module.__name__
    sys.modules[module.__name__] = module
    evidence.REGISTRY["probe_match_only"] = compute
    try:
        assert evidence.match_only("probe_match_only") is True, "a module with MATCH_ONLY = True is match-only"
    finally:
        del evidence.REGISTRY["probe_match_only"]
        del sys.modules[module.__name__]
    expect_error(KeyError, lambda: evidence.match_only("no_such_source"), "no_such_source")


# -- synthetic tracking -----------------------------------------------------------------------------------------------
#
# A fake tactical bundle in the real file format, so the cache and the evidence can be checked against the baseline's
# own code without touching -- or printing -- any match data. Every awkward case the real file has, on purpose:
# absent slots, missing balls, a ball with nobody tracked, exact ties between a home and an away slot, NaN speeds on
# present slots, and numbers written with different precisions (the parser must be bit-exact on all of them).

# Params
SYNTHETIC_SLOTS = 46        # as many track slots as the real tactical ATD
SYNTHETIC_FRAMES = 600
SYNTHETIC_GRADED = [(1, 0, 300), (2, 300, SYNTHETIC_FRAMES)]


def make_synthetic_bundle(root, n_frames=SYNTHETIC_FRAMES, n_slots=SYNTHETIC_SLOTS, seed=0):
    """Write root/tactical/BAR-ATH_tactical_atd.{xml,txt}. Slots alternate home, away, home, ..."""
    rng = np.random.default_rng(seed)
    folder = os.path.join(root, "tactical")
    os.makedirs(folder, exist_ok=True)

    players = "".join(
        '<Player id="P{0}" teamId="{1}"><Name>Track {2}</Name><ShirtNumber>{0}</ShirtNumber></Player>'.format(
            slot, "T_HOME" if slot % 2 == 0 else "T_AWAY", slot + 1) for slot in range(n_slots))
    write_text(folder, "BAR-ATH_tactical_atd.xml",
               '<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata><Teams>'
               '<Team id="T_HOME"><Name>Home</Name></Team><Team id="T_AWAY"><Name>Away</Name></Team>'
               '</Teams><Players>{}</Players></Metadata></main>\n'.format(players))

    # Shortest round-trip repr (17 significant digits at most) is the hardest case for a parser; float() first,
    # because numpy 2 spells repr(np.float64) as "np.float64(...)".
    styles = ("{:.6f}".format, "{:.3f}".format, lambda value: repr(float(value)))
    lines = []
    for frame in range(n_frames):
        number = styles[frame % len(styles)]  # one style per frame, so duplicated positions stay identical strings
        present = rng.random(n_slots) < 0.45
        xs = rng.random(n_slots) * 1.04 - 0.02  # a little outside the pitch, as real tracking strays
        ys = rng.random(n_slots) * 1.04 - 0.02
        ball = (rng.random(), rng.random())
        ball_missing = rng.random() < 0.3
        if frame % 97 == 5:
            present[:] = False  # a ball and nobody tracked
        if frame % 13 == 0:
            # An exact tie across teams: a home and an away slot on the same spot, the ball on it as well. The
            # baseline's strict "<" keeps the lower slot index; so must np.argmin.
            home_slot = 2 * int(rng.integers(n_slots // 2))
            away_slot = 2 * int(rng.integers(n_slots // 2)) + 1
            present[home_slot] = present[away_slot] = True
            xs[away_slot], ys[away_slot] = xs[home_slot], ys[home_slot]
            ball, ball_missing = (xs[home_slot], ys[home_slot]), False
        groups = []
        for slot in range(n_slots):
            if not present[slot]:
                groups.append("NaN,NaN,NaN")
                continue
            speed = "NaN" if rng.random() < 0.1 else number(rng.random() * 9)
            groups.append("{},{},{}".format(number(xs[slot]), number(ys[slot]), speed))
        ball_text = "NaN,NaN,NaN" if ball_missing else "{},{},{}".format(
            number(ball[0]), number(ball[1]), "NaN" if rng.random() < 0.2 else number(rng.random() * 30))
        lines.append("{}:{}:{}".format(frame + 1, ";".join(groups), ball_text))
    write_text(folder, "BAR-ATH_tactical_atd.txt", "\n".join(lines) + "\n")
    return folder


def python_parse(raw_path, n_slots):
    """The raw ATD read with Python's float(), the way starter/baseline.py reads it -- the parser's reference."""
    rows = []
    with open(raw_path) as handle:
        for line in handle:
            _, players, ball = line.rstrip("\n").split(":")
            rows.append([float(token) for group in players.split(";") for token in group.split(",")]
                        + [float(token) for token in ball.split(",")])
    table = np.array(rows)
    players = table[:, :3 * n_slots].reshape(len(table), n_slots, 3)
    return {"players_xy": players[:, :, :2], "players_speed": players[:, :, 2],
            "ball_xy": table[:, -3:-1], "ball_speed": table[:, -1]}


def synthetic_inputs(root, **overrides):
    """An Inputs over a synthetic bundle in root, with a two-half layout scaled to the synthetic length."""
    from pipeline.inputs import Inputs
    options = dict(data_dir=root, n_frames=SYNTHETIC_FRAMES, graded=SYNTHETIC_GRADED)
    options.update(overrides)
    return Inputs(**options)


def edit_line(path, line_number, edit):
    """Rewrite one line of a text file through edit(line) -> line. For breaking a synthetic file on purpose."""
    with open(path) as handle:
        lines = handle.read().split("\n")
    lines[line_number] = edit(lines[line_number])
    write_text(os.path.dirname(path), os.path.basename(path), "\n".join(lines))


# -- inputs -----------------------------------------------------------------------------------------------------------

def test_inputs_defaults_match_the_match():
    from pipeline.inputs import Inputs
    inputs = Inputs()  # lazy: touches no data
    assert inputs.n_frames == 199319, inputs.n_frames
    assert inputs.graded == [(1, 11185, 90652), (2, 110808, 190198)], inputs.graded
    assert inputs.kickoffs == {1: 11185, 2: 113742}, inputs.kickoffs
    halves = inputs.half_of_frame
    assert [int(halves[f]) for f in (11184, 11185, 90651, 90652, 110807, 110808, 190197, 190198)] == \
        [0, 1, 1, 0, 0, 2, 2, 0]
    assert int(inputs.graded_mask.sum()) == (90652 - 11185) + (190198 - 110808)


def test_cache_matches_python_parse():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        atd = synthetic_inputs(root).atd
        reference = python_parse(os.path.join(root, "tactical", "BAR-ATH_tactical_atd.txt"), SYNTHETIC_SLOTS)
        for name, expected in reference.items():
            assert np.array_equal(getattr(atd, name), expected, equal_nan=True), "{} differs".format(name)
        assert atd.slot_team.tolist() == [slot % 2 for slot in range(SYNTHETIC_SLOTS)]


def test_cache_reused_when_fresh():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        first = synthetic_inputs(root)
        first.atd
        manifest = os.path.join(first.cache_dir, "atd_tactical", "manifest.json")
        stamp = os.stat(manifest).st_mtime_ns
        second = synthetic_inputs(root)
        second.atd
        assert (first.atd_status, second.atd_status) == ("built", "reused"), (first.atd_status, second.atd_status)
        assert os.stat(manifest).st_mtime_ns == stamp


def test_cache_rebuilt_when_source_changes():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root, seed=0)
        before = np.array(synthetic_inputs(root).atd.players_xy)
        make_synthetic_bundle(root, seed=1)  # same metadata, different positions
        inputs = synthetic_inputs(root)
        after = inputs.atd.players_xy
        assert inputs.atd_status == "built", inputs.atd_status
        assert not np.array_equal(before, after, equal_nan=True), "rebuilt cache still holds the old positions"


def test_cache_rejects_partial_nan():
    with tempfile.TemporaryDirectory() as root:
        folder = make_synthetic_bundle(root)

        def blank_one_y(line):
            frame, players, ball = line.split(":")
            groups = players.split(";")
            slot = next(index for index, group in enumerate(groups) if not group.startswith("NaN"))
            x, _, speed = groups[slot].split(",")
            groups[slot] = ",".join((x, "NaN", speed))
            return ":".join((frame, ";".join(groups), ball))

        edit_line(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), 1, blank_one_y)
        expect_error(ValueError, lambda: synthetic_inputs(root).atd, "not missing together")


def test_cache_rejects_short_line():
    with tempfile.TemporaryDirectory() as root:
        folder = make_synthetic_bundle(root)
        # Drop the last slot of one line. pandas would pad the row with NaN and read it as "ball missing".
        edit_line(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), 2,
                  lambda line: line[:line.rindex(";")] + line[line.rindex(":"):])
        expect_error(ValueError, lambda: synthetic_inputs(root).atd, "slots")


def test_cache_arrays_read_only():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        for inputs in (synthetic_inputs(root), synthetic_inputs(root)):  # freshly built, then loaded from disk
            atd = inputs.atd
            for name in ("players_xy", "players_speed", "ball_xy", "ball_speed", "slot_team"):
                assert not getattr(atd, name).flags.writeable, "{} is writable ({})".format(name, inputs.atd_status)


def test_cache_atd_repr_shows_no_array():
    """repr(atd) holds no value of the tracking: an Atd that a log line, an assertion message or a debugger prints.

    A dataclass shows its fields and numpy shows the head and tail of an array, which would put frames of the match
    (CLAUDE.md: never to a log) in the output. Every array field has repr=False; a distinctive value planted in each
    must not appear, in an Atd built here and in one parsed from the synthetic bundle."""
    import dataclasses
    from pipeline.inputs import atd_cache
    assert all(field.repr is False for field in dataclasses.fields(atd_cache.Atd)), [
        field.name for field in dataclasses.fields(atd_cache.Atd) if field.repr]
    marker = 0.314159265
    planted = atd_cache.Atd(players_xy=np.full((4, 3, 2), marker), players_speed=np.full((4, 3), marker + 1),
                            ball_xy=np.full((4, 2), marker + 2), ball_speed=np.full(4, marker + 3),
                            slot_team=np.array([0, 1, 1], dtype=np.int8))
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        parsed = synthetic_inputs(root).atd
        for label, atd in (("planted", planted), ("parsed", parsed)):
            shown = repr(atd)
            assert shown.startswith("Atd(") and "array" not in shown and len(shown) < 100, (label, shown)
            for name in ("players_xy", "players_speed", "ball_xy", "ball_speed"):
                first = getattr(atd, name).flat[0]
                if np.isfinite(first):
                    assert "{:.3f}".format(first) not in shown, (label, name)
    assert "0.314" not in repr(planted)


def test_inputs_homography_repr_shows_no_array():
    """repr(homography) holds no value either: its matrices are our match's homography, one per frame.

    The same leak as the Atd's (test_cache_atd_repr_shows_no_array): every array field has repr=False."""
    import dataclasses
    from pipeline.inputs import Homography
    assert all(field.repr is False for field in dataclasses.fields(Homography)), [
        field.name for field in dataclasses.fields(Homography) if field.repr]
    homography = Homography(matrices=np.full((4, 3, 3), 0.271828182), error=np.full(4, 1.41421356),
                            valid=np.ones(4, dtype=bool))
    shown = repr(homography)
    assert shown.startswith("Homography(") and "array" not in shown and len(shown) < 100, shown
    assert "0.271" not in shown and "1.414" not in shown and "True" not in shown, shown


def test_cache_rebuilt_when_arrays_missing():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        first = synthetic_inputs(root)
        first.atd
        os.remove(os.path.join(first.cache_dir, "atd_tactical", "players_xy.npy"))  # manifest left in place
        second = synthetic_inputs(root)
        assert second.atd.players_xy.shape == (SYNTHETIC_FRAMES, SYNTHETIC_SLOTS, 2)
        assert second.atd_status == "built", second.atd_status


def test_cache_rebuilt_when_manifest_truncated():
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        first = synthetic_inputs(root)
        first.atd
        path = os.path.join(first.cache_dir, "atd_tactical", "manifest.json")
        with open(path, "rb") as handle:
            head = handle.read()[:40]
        with open(path, "wb") as handle:
            handle.write(head)  # a manifest cut short: not JSON, so no cache
        second = synthetic_inputs(root)
        assert second.atd.players_xy.shape == (SYNTHETIC_FRAMES, SYNTHETIC_SLOTS, 2)
        assert second.atd_status == "built", second.atd_status
        with open(path, encoding="utf-8") as handle:
            assert handle.read().endswith("}\n"), "the rebuilt manifest is written whole"


# -- evidence ---------------------------------------------------------------------------------------------------------

def test_nearest_player_matches_baseline():
    from pipeline.contracts import check_evidence
    from starter import baseline
    with tempfile.TemporaryDirectory() as root:
        evidence, _, codes = nearest_player_runs(root)  # asserts that the match reaches every branch
    check_evidence(evidence, SYNTHETIC_FRAMES)
    expected = np.where(codes == baseline.HOME, 1.0, np.where(codes == baseline.AWAY, 0.0, np.nan))
    differ = ~((evidence.home == expected) | (np.isnan(evidence.home) & np.isnan(expected)))
    assert not differ.any(), "{} frames differ, first at frame {}".format(
        int(differ.sum()), int(np.flatnonzero(differ)[0]))
    assert np.isnan(evidence.alive).all(), "nearest_player has no opinion on alive vs dead"


def nearest_player_runs(root, **parameters):
    """(default output, output with `parameters`, the baseline's codes) on the synthetic match, which has home,
    away and no-opinion frames."""
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    from starter import baseline
    folder = make_synthetic_bundle(root)
    inputs = synthetic_inputs(root)
    run = RunContext(inputs=inputs, seed=0)
    hard = REGISTRY["nearest_player"](run)
    codes = baseline.nearest_player_labels(
        os.path.join(folder, "BAR-ATH_tactical_atd.txt"),
        baseline.team_of_each_track(os.path.join(folder, "BAR-ATH_tactical_atd.xml")),
        SYNTHETIC_FRAMES, inputs.pitch)
    assert {int(code) for code in np.unique(codes)} == {baseline.NOBODY, baseline.HOME, baseline.AWAY}, \
        "the synthetic match no longer reaches every branch"
    return hard, REGISTRY["nearest_player"](run, **parameters), codes


def test_nearest_player_confidence_default_is_unchanged():
    # The four committed baselines are rebuilt through this source (--data); the default must stay the hard 0 / 1.
    from starter import baseline
    with tempfile.TemporaryDirectory() as root:
        hard, explicit, codes = nearest_player_runs(root, confidence=1.0)
    expected = np.where(codes == baseline.HOME, 1.0, np.where(codes == baseline.AWAY, 0.0, np.nan))
    for evidence in (hard, explicit):
        assert np.array_equal(evidence.home, expected, equal_nan=True), "home differs from the baseline's"
        assert np.isnan(evidence.alive).all(), "nearest_player has no opinion on alive vs dead"


def test_nearest_player_confidence_sets_both_values():
    with tempfile.TemporaryDirectory() as root:
        hard, soft, _ = nearest_player_runs(root, confidence=0.78)
    known = ~np.isnan(soft.home)
    assert (hard.home == 1.0).any() and (hard.home == 0.0).any(), "the synthetic match lacks home or away frames"
    assert np.array_equal(np.isnan(soft.home), np.isnan(hard.home)), "NaN moved to other frames"
    assert np.isin(soft.home[known], [0.78, 1 - 0.78]).all(), "a value other than 0.78 or 1 - 0.78"
    assert (soft.home[hard.home == 1.0] == 0.78).all(), "a home frame is not exactly 0.78"
    assert (soft.home[hard.home == 0.0] == 1 - 0.78).all(), "an away frame is not exactly 1 - 0.78"
    assert np.isnan(soft.alive).all()


def test_nearest_player_rejects_bad_confidence():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    run = RunContext(inputs=None, seed=0)  # rejected before the match is touched
    for bad in (0.4, 1.5, float("nan"), True, "0.8"):
        expect_error(ValueError, lambda: REGISTRY["nearest_player"](run, confidence=bad), "nearest_player",
                     "confidence")


def ball_run(ball_present):
    """A RunContext over a stand-in match that has only what ball_presence reads: n_frames and atd.ball_present."""
    from types import SimpleNamespace
    from pipeline.contracts import RunContext
    atd = SimpleNamespace(ball_present=ball_present)
    return RunContext(inputs=SimpleNamespace(n_frames=len(ball_present), atd=atd), seed=0)


def test_ball_presence_values_and_margin():
    from pipeline.contracts import check_evidence
    from pipeline.evidence import REGISTRY
    ball = np.ones(40, dtype=bool)
    ball[:3] = False      # missing at the very start,
    ball[20:30] = False   # for ten frames in the middle,
    ball[38:] = False     # and at the very end
    source = REGISTRY["ball_presence"]
    seen = np.flatnonzero(ball)
    gap = np.flatnonzero(~ball)
    distance = np.abs(gap[:, None] - seen[None, :]).min(axis=1)  # to the nearest frame with a ball, by brute force

    evidence = source(ball_run(ball))
    assert (evidence.alive[ball] == 0.585).all() and (evidence.alive[~ball] == 0.2).all()
    assert evidence.alive.dtype == np.float64 and evidence.home.dtype == np.float64
    assert np.isnan(evidence.home).all()
    check_evidence(evidence, 40)

    evidence = source(ball_run(ball), margin=3)
    assert (evidence.alive[ball] == 0.585).all()
    assert (evidence.alive[gap[distance > 3]] == 0.2).all()
    assert np.isnan(evidence.alive[gap[distance <= 3]]).all()
    assert np.flatnonzero(evidence.alive == 0.2).tolist() == [23, 24, 25, 26], "the middle of the ten-frame gap"
    check_evidence(evidence, 40)

    nothing = source(ball_run(np.zeros(40, dtype=bool)), margin=3)  # no ball at all: nothing is near a ball frame
    assert (nothing.alive == 0.2).all() and np.isnan(nothing.home).all()
    check_evidence(nothing, 40)


def test_ball_presence_seen_half_is_no_push():
    from pipeline.evidence import REGISTRY
    ball = np.arange(30) % 7 != 0
    evidence = REGISTRY["ball_presence"](ball_run(ball), seen=0.5)
    # log(2 * 0.5) = 0 in the decoder's emission rule: a visible ball then says nothing about alive vs dead.
    assert (evidence.alive[ball] == 0.5).all() and (evidence.alive[~ball] == 0.2).all()


def test_ball_presence_reads_read_only_inputs():
    from pipeline.evidence import REGISTRY
    ball = np.arange(30) % 7 != 0
    ball.flags.writeable = False
    evidence = REGISTRY["ball_presence"](ball_run(ball), margin=2)
    assert (evidence.alive[ball] == 0.585).all()
    assert ball.flags.writeable is False and ball.sum() == 25, "the input was touched"


def test_ball_presence_rejects_bad_parameters():
    from pipeline.evidence import REGISTRY
    run = ball_run(np.ones(10, dtype=bool))
    source = REGISTRY["ball_presence"]
    for bad in (-1, 1.5, True):
        expect_error(ValueError, lambda: source(run, margin=bad), "ball_presence", "margin")
    for bad in (-0.1, 1.1, float("nan"), True, "0.2"):
        expect_error(ValueError, lambda: source(run, missing=bad), "ball_presence", "missing")
    for bad in (1.2, -0.5, float("nan"), True):
        expect_error(ValueError, lambda: source(run, seen=bad), "ball_presence", "seen")
    source(run, margin=np.int64(3), missing=np.float64(0.25), seen=1)  # numpy numbers from a sweep are fine


# -- shared kinematics ------------------------------------------------------------------------------------------------

def kinematics_inputs(atd, graded, fps=25, pitch=(105.3, 68.0)):
    """A stand-in Inputs with what kinematics reads: atd, graded, fps, pitch and metres."""
    scale = np.array(pitch)

    class StandIn:   # a class, not a SimpleNamespace: the memo needs an object that can be weakly referenced
        pass

    inputs = StandIn()
    inputs.atd, inputs.graded, inputs.fps, inputs.pitch = atd, graded, fps, pitch
    inputs.metres = lambda xy: xy * scale
    return inputs


def kinematics_atd(players_xy, ball_xy, speed_fill=None, seed=0):
    """An Atd over given positions; the speed channels are random, or all `speed_fill`."""
    from pipeline.inputs.atd_cache import Atd
    rng = np.random.default_rng(seed)
    n, slots = players_xy.shape[:2]
    players_speed = rng.random((n, slots)) * 9 if speed_fill is None else np.full((n, slots), speed_fill)
    ball_speed = rng.random(n) * 30 if speed_fill is None else np.full(n, speed_fill)
    return Atd(players_xy=players_xy, players_speed=players_speed, ball_xy=ball_xy, ball_speed=ball_speed,
               slot_team=(np.arange(slots) % 2).astype(np.int8))


def test_kinematics_derivative_is_exact_on_lines():
    from pipeline.evidence import kinematics
    fps = 25
    blocks = [(0, 30), (30, 60)]
    # (slot, start, stop): runs of 1, 2, 3, 7, 8 and 20 frames, some touching the half boundary from each side
    pieces = [(0, 0, 20), (0, 22, 23), (0, 25, 27), (1, 3, 6), (1, 10, 17), (1, 20, 28), (2, 20, 30), (2, 30, 45),
              (2, 50, 51), (2, 52, 60)]
    rng = np.random.default_rng(3)
    xy = rng.random((60, 3, 2)) * 100                    # junk on invalid frames
    valid = np.zeros((60, 3), dtype=bool)
    expected = np.full((60, 3, 2), np.nan)
    for slot, start, stop in pieces:
        slope = rng.normal(size=2) * 5                   # metres per second
        offset = rng.random(2) * 50
        t = np.arange(start, stop)[:, None] / fps
        xy[start:stop, slot] = offset + slope * t
        valid[start:stop, slot] = True
        if stop - start >= 2:
            expected[start:stop, slot] = slope
    vel = kinematics.derivative(xy, valid, blocks, fps)
    assert vel.shape == xy.shape
    both = ~np.isnan(expected)
    assert np.abs(vel[both] - expected[both]).max() < 1e-9
    assert np.isnan(vel[~both]).all(), "NaN on invalid frames and on a run of one frame"
    flat = kinematics.derivative(xy[:, 0], valid[:, 0], blocks, fps)    # the (n, 2) form agrees with the (n, K, 2) one
    assert np.array_equal(flat, vel[:, 0], equal_nan=True)


def test_kinematics_never_crosses_a_run_or_block():
    from pipeline.evidence import kinematics
    rng = np.random.default_rng(4)
    blocks = [(5, 40), (50, 90)]
    xy = rng.random((100, 4, 2)) * 100
    valid = rng.random((100, 4)) < 0.8
    valid[:, 3] = True                                    # one slot valid throughout, to meet the block edges
    base = kinematics.derivative(xy, valid, blocks, 25)
    junk = xy.copy()
    junk[~valid] = rng.random(((~valid).sum(), 2)) * 1000   # positions beyond a run's ends change nothing in it
    assert np.array_equal(kinematics.derivative(junk, valid, blocks, 25), base, equal_nan=True)
    moved = xy.copy()
    moved[40:] = rng.random((60, 4, 2)) * 1000              # the second block and the gap change nothing in the first
    assert np.array_equal(kinematics.derivative(moved, valid, blocks, 25)[:40], base[:40], equal_nan=True)
    moved = xy.copy()
    moved[:50] = rng.random((50, 4, 2)) * 1000
    assert np.array_equal(kinematics.derivative(moved, valid, blocks, 25)[50:], base[50:], equal_nan=True)
    assert np.isnan(base[:5]).all() and np.isnan(base[40:50]).all() and np.isnan(base[90:]).all()
    expect_error(ValueError, lambda: kinematics.derivative(xy, valid[:, :3], blocks, 25), "kinematics", "derivative")


def test_kinematics_chunks_change_nothing():
    from pipeline.evidence import kinematics
    rng = np.random.default_rng(7)
    xy = rng.random((200, 5, 2)) * 100
    valid = rng.random((200, 5)) < 0.9
    blocks = [(3, 90), (100, 199)]
    whole = kinematics.derivative(xy, valid, blocks, 25)
    saved = kinematics.CHUNK
    try:
        for chunk in (1, 4, 7, 64):
            kinematics.CHUNK = chunk
            assert np.array_equal(kinematics.derivative(xy, valid, blocks, 25), whole, equal_nan=True), chunk
    finally:
        kinematics.CHUNK = saved


def test_kinematics_uses_metres():
    from pipeline.evidence import kinematics
    n, fps = 30, 25
    t = np.arange(n)
    ball = np.stack([0.2 + 0.01 * t, np.full(n, 0.5)], axis=1)                          # 0.01 in x per frame
    players = np.stack([np.stack([np.full(n, 0.5), 0.1 + 0.01 * t], axis=1)], axis=1)   # and in y, for one slot
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 0, n)], fps=fps, pitch=(105.3, 68.0))
    ball_t, player_t = kinematics.ball_track(inputs), kinematics.player_track(inputs)
    assert np.allclose(ball_t.vel[:, 0], 0.01 * 105.3 * fps, atol=1e-9) and np.allclose(ball_t.vel[:, 1], 0, atol=1e-9)
    assert np.allclose(player_t.vel[:, 0, 1], 0.01 * 68.0 * fps, atol=1e-9)
    assert np.allclose(ball_t.speed, 0.01 * 105.3 * fps) and np.allclose(player_t.speed[:, 0], 0.01 * 68.0 * fps)
    assert np.allclose(ball_t.xy_m[:, 0], (0.2 + 0.01 * t) * 105.3) and ball_t.visible.all() and player_t.present.all()


def test_kinematics_window_mean_matches_a_loop():
    from pipeline.evidence import kinematics
    rng = np.random.default_rng(5)
    n = 120
    blocks = [(5, 30), (40, 41), (50, 110)]
    values = rng.normal(size=n)
    values[rng.random(n) < 0.2] = np.nan
    valid = rng.random(n) < 0.8      # some valid frames carry NaN: a track's speed on a one-frame run is such a value
    for before in (0, 3, 25):
        for after in (0, 3, 25):
            reference = np.full(n, np.nan)
            for start, stop in blocks:
                for t in range(start, stop):
                    low, high = max(start, t - before), min(stop, t + after + 1)
                    window = [v for v, ok in zip(values[low:high], valid[low:high]) if ok and not np.isnan(v)]
                    if window:
                        reference[t] = sum(window) / len(window)
            got = kinematics.window_mean(values, valid, blocks, before, after)
            assert np.array_equal(np.isnan(got), np.isnan(reference)), (before, after)
            assert np.nanmax(np.abs(got - reference)) < 1e-12, (before, after)
    assert (valid & np.isnan(values)).any() and not np.isnan(got[60:100]).all(), "no NaN rode in on a valid frame"
    expect_error(ValueError, lambda: kinematics.window_mean(values, valid, blocks, -1, 0), "kinematics", "before")


def test_kinematics_runs():
    from pipeline.evidence import kinematics
    starts, stops = kinematics.runs(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool))
    assert starts.tolist() == [0, 4, 6] and stops.tolist() == [2, 5, 9]
    starts, stops = kinematics.runs(np.zeros(4, dtype=bool))
    assert len(starts) == 0 and len(stops) == 0


def test_kinematics_outside_m():
    from pipeline.evidence import kinematics
    pitch = (100.0, 60.0)
    points = np.array([[50, 30], [10, 20], [0, 30], [100, 10], [30, 0], [-3, 30], [103, 20], [50, -2], [50, 64],
                       [103, 64], [-3, -4], [np.nan, np.nan]])
    expected = [-30, -10, 0, 0, 0, 3, 3, 2, 4, 5, 5, np.nan]
    got = kinematics.outside_m(points, pitch)
    assert got.shape == (12,) and np.allclose(got, expected, equal_nan=True), got
    assert kinematics.outside_m(points.reshape(3, 4, 2), pitch).shape == (3, 4)


def test_kinematics_ball_distances():
    from pipeline.evidence import kinematics
    players = np.array([[[3.0, 4.0], [np.nan, np.nan]], [[1.0, 1.0], [0.0, 2.0]]])
    ball = np.array([[0.0, 0.0], [np.nan, np.nan]])
    got = kinematics.ball_distances(players, ball)
    assert got.shape == (2, 2) and got[0, 0] == 5.0 and np.isnan(got[0, 1]) and np.isnan(got[1]).all()


def kinematics_random_inputs(n=600, slots=6, seed=0, **atd_options):
    rng = np.random.default_rng(seed)
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.3] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.3] = np.nan
    return kinematics_inputs(kinematics_atd(players, ball, **atd_options), SYNTHETIC_GRADED)


def test_kinematics_reads_read_only_inputs():
    from pipeline.evidence import kinematics
    inputs = kinematics_random_inputs()
    for array in (inputs.atd.players_xy, inputs.atd.ball_xy):
        array.flags.writeable = False
    before = (inputs.atd.players_xy.copy(), inputs.atd.ball_xy.copy())
    ball, players = kinematics.ball_track(inputs), kinematics.player_track(inputs)
    assert np.array_equal(inputs.atd.players_xy, before[0], equal_nan=True)
    assert np.array_equal(inputs.atd.ball_xy, before[1], equal_nan=True)
    for track in (ball, players):
        for name in ("xy_m", "vel", "speed"):
            assert getattr(track, name).flags.writeable is False, name
    assert ball.visible.flags.writeable is False and players.present.flags.writeable is False
    assert "array(" not in repr(ball) and "array(" not in repr(players), "an array is in the repr"
    assert np.isnan(players.xy_m[~players.present]).all() and np.isnan(ball.xy_m[~ball.visible]).all()
    assert (players.present == inputs.atd.players_present).all()   # SYNTHETIC_GRADED covers every frame


def test_kinematics_never_reads_speed_channels():
    from pipeline.evidence import kinematics
    with_values = kinematics_random_inputs(seed=1)
    without = kinematics_random_inputs(seed=1, speed_fill=np.nan)
    assert not np.isnan(with_values.atd.players_speed).any() and np.isnan(without.atd.players_speed).all()
    for name in ("xy_m", "vel", "speed"):
        assert np.array_equal(getattr(kinematics.player_track(with_values), name),
                              getattr(kinematics.player_track(without), name), equal_nan=True), name
        assert np.array_equal(getattr(kinematics.ball_track(with_values), name),
                              getattr(kinematics.ball_track(without), name), equal_nan=True), name


def test_kinematics_graded_blocks_only():
    from pipeline.evidence import kinematics
    inputs = kinematics_random_inputs()
    inputs.graded = [(1, 100, 300), (2, 350, 500)]
    ball, players = kinematics.ball_track(inputs), kinematics.player_track(inputs)
    outside = np.ones(600, dtype=bool)
    outside[100:300] = outside[350:500] = False
    assert not ball.visible[outside].any() and not players.present[outside].any()
    assert np.isnan(ball.vel[outside]).all() and np.isnan(players.vel[outside]).all()
    assert kinematics.blocks(inputs) == [(100, 300), (350, 500)]


def test_kinematics_speed():
    from pipeline.evidence import kinematics
    rng = np.random.default_rng(6)
    n, slots = 200_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 11_185, 90_652), (2, 110_808, 190_198)])
    inputs.atd.players_present                      # the Atd's own cached masks are not this module's cost
    inputs.atd.ball_present
    started = time.perf_counter()
    kinematics.ball_track(inputs)
    kinematics.player_track(inputs)
    elapsed = time.perf_counter() - started
    print("      kinematics speed: ball_track + player_track on {} x {} in {:.2f} s".format(n, slots, elapsed))
    assert elapsed < 2.0, "{:.2f} s for 200,000 frames x 46 slots (the budget is 2 s)".format(elapsed)


def test_kinematics_memo_is_per_object():
    import gc
    from pipeline.evidence import kinematics
    first = kinematics_random_inputs(seed=1)
    ball, players = kinematics.ball_track(first), kinematics.player_track(first)
    assert kinematics.ball_track(first) is ball and kinematics.player_track(first) is players
    other = kinematics_random_inputs(seed=2)            # other tracking, a new object: recomputed, not served stale
    assert kinematics.ball_track(other) is not ball and kinematics.player_track(other) is not players
    assert not np.array_equal(kinematics.ball_track(other).xy_m, ball.xy_m, equal_nan=True)
    held, held_players = len(kinematics._BALL_MEMO), len(kinematics._PLAYER_MEMO)
    del first, other, ball, players
    gc.collect()
    assert len(kinematics._BALL_MEMO) == held - 2, "the memo kept an object"
    assert len(kinematics._PLAYER_MEMO) == held_players - 2, "the memo kept an object"


# -- controlled_touch -------------------------------------------------------------------------------------------------
#
# Hand-built scenes in metres: slot 0 is home, slot 1 away, slot 2 home, ... (kinematics_atd). A slot is present only
# on the frames it is given, so "an episode of 10 frames" is a slot that is 0.5 m from a static ball for 10 frames.

TOUCH = dict(radius=2.0, max_rel_speed=3.0, min_frames=5, confidence=0.9)


def touch_scene(n, ball=None, slots=None, graded=None):
    """A RunContext over n frames of hand-built tracking in metres. `ball` is (n, 2) (NaN = missing; default a static
    ball at (30, 30)), `slots` a dict slot -> (n, 2) (NaN = absent), `graded` a list of (start, stop) blocks (default
    one block over every frame)."""
    from pipeline.contracts import RunContext
    scale = np.array([105.3, 68.0])
    ball = np.tile([30.0, 30.0], (n, 1)) if ball is None else np.asarray(ball, dtype=np.float64)
    players = np.full((n, max(max(slots) + 1 if slots else 2, 2), 2), np.nan)
    for slot, track in (slots or {}).items():
        players[:, slot] = track
    graded = [(1, 0, n)] if graded is None else [(index + 1, start, stop) for index, (start, stop) in enumerate(graded)]
    inputs = kinematics_inputs(kinematics_atd(players / scale, ball / scale), graded)
    return RunContext(inputs=inputs, seed=0)


def beside_ball(n, frames, offset=(0.5, 0.0), ball=(30.0, 30.0)):
    """A slot's track: `offset` metres from a static ball on `frames` (a range), absent elsewhere."""
    track = np.full((n, 2), np.nan)
    track[list(frames)] = np.array(ball) + np.array(offset)
    return track


def touch(run, **parameters):
    from pipeline.evidence import REGISTRY
    return REGISTRY["controlled_touch"](run, **dict(TOUCH, **parameters))


def test_touch_dribble_is_control():
    n = 40
    t = np.arange(n)
    ball = np.stack([20.0 + 0.2 * t, np.full(n, 30.0)], axis=1)       # 5 m/s along x
    mover = np.full((n, 2), np.nan)
    mover[10:20] = ball[10:20] + [0.5, 0.0]                           # a home slot 0.5 m ahead of it, same velocity
    evidence = touch(touch_scene(n, ball=ball, slots={0: mover}))
    assert np.flatnonzero(evidence.home == 0.9).tolist() == list(range(10, 20)), np.flatnonzero(evidence.home == 0.9)
    assert np.isnan(np.delete(evidence.home, range(10, 20))).all() and np.isnan(evidence.alive).all()


def test_touch_flight_is_no_evidence():
    n = 40
    t = np.arange(n)
    ball = np.stack([15.0 + 0.6 * t, np.full(n, 30.0)], axis=1)       # 15 m/s along x
    stationary = np.tile([24.0, 30.8], (n, 1))                        # an away slot 0.8 m from the ball's line
    near = np.hypot(*(ball - stationary).T) <= TOUCH["radius"]
    assert near.sum() >= 5, "the ball must pass within the radius for the check to mean something"
    assert np.isnan(touch(touch_scene(n, ball=ball, slots={1: stationary})).home).all()
    # Without the speed test the same frames are control: it is the relative speed that rules them out.
    loose = touch(touch_scene(n, ball=ball, slots={1: stationary}), max_rel_speed=np.inf, min_frames=1)
    assert (loose.home[near] == 1.0 - 0.9).all()


def test_touch_duel_is_no_evidence():
    n = 20
    home, away = beside_ball(n, range(n), (0.6, 0.0)), beside_ball(n, range(n), (-0.9, 0.0))
    assert np.isnan(touch(touch_scene(n, slots={0: home, 1: away}), duel_radius=1.0).home).all()
    # The same pair with the away slot beyond the duel radius: home controls it.
    away = beside_ball(n, range(n), (-1.1, 0.0))
    assert (touch(touch_scene(n, slots={0: home, 1: away}), duel_radius=1.0).home == 0.9).all()
    assert np.isnan(touch(touch_scene(n, slots={0: home, 1: away}), duel_radius=1.2).home).all()


def test_touch_short_episode_dropped():
    n = 40
    for length, kept in ((TOUCH["min_frames"] - 1, False), (TOUCH["min_frames"], True)):
        evidence = touch(touch_scene(n, slots={1: beside_ball(n, range(10, 10 + length))}))
        assert np.isnan(evidence.home).all() != kept, (length, kept)
        if kept:
            assert np.flatnonzero(evidence.home == 1.0 - 0.9).tolist() == list(range(10, 15))
    # A change of team ends an episode: 4 + 4 frames beside the same ball, one team and then the other, keep nothing.
    both = touch_scene(n, slots={0: beside_ball(n, range(10, 14)), 1: beside_ball(n, range(14, 18))})
    assert np.isnan(touch(both).home).all()


def test_touch_is_nan_without_the_ball():
    n = 40
    ball = np.tile([30.0, 30.0], (n, 1))
    ball[15:25] = np.nan
    slots = {0: beside_ball(n, range(n))}
    evidence = touch(touch_scene(n, ball=ball, slots=slots), min_frames=2)
    assert np.isnan(evidence.home[15:25]).all()
    assert (evidence.home[:15] == 0.9).all() and (evidence.home[25:] == 0.9).all()
    held = touch(touch_scene(n, ball=ball, slots=slots), min_frames=2, hold=0.6)
    assert np.isnan(held.home[15:25]).all(), "a hold never fills a missing ball"


def test_touch_outside_blocks_is_nan():
    n = 60
    slots = {0: beside_ball(n, range(n))}
    evidence = touch(touch_scene(n, slots=slots, graded=[(10, 30), (40, 55)]), hold=0.6)
    assert (evidence.home[10:30] == 0.9).all() and (evidence.home[40:55] == 0.9).all()
    outside = np.ones(n, dtype=bool)
    outside[10:30] = outside[40:55] = False
    assert np.isnan(evidence.home[outside]).all() and np.isnan(evidence.alive).all()
    # An episode is cut at the block's edge: two 20-frame pieces of one touch do not add up to 25.
    assert np.isnan(touch(touch_scene(n, slots=slots, graded=[(0, 20), (30, 50)]), min_frames=25).home).all()


def hold_scene(graded=None):
    """Home controls 5..14, away 35..44, on a static ball that is missing on 25..29, in a 60-frame match."""
    n = 60
    ball = np.tile([30.0, 30.0], (n, 1))
    ball[25:30] = np.nan
    slots = {0: beside_ball(n, range(5, 15)), 1: beside_ball(n, range(35, 45))}
    return touch_scene(n, ball=ball, slots=slots, graded=graded)


def test_touch_hold_carries_the_last_team():
    evidence = touch(hold_scene(), hold=0.6)
    expected = np.full(60, np.nan)
    for span, value in ((slice(5, 15), 0.9), (slice(15, 25), 0.6), (slice(30, 35), 0.6), (slice(35, 45), 1.0 - 0.9),
                        (slice(45, None), 1.0 - 0.6)):
        expected[span] = value
    assert np.array_equal(evidence.home, expected, equal_nan=True), np.flatnonzero(evidence.home != expected)


def test_touch_hold_stays_in_its_block():
    evidence = touch(hold_scene(graded=[(0, 30), (30, 60)]), hold=0.6)
    expected = np.full(60, np.nan)
    expected[5:15], expected[15:25] = 0.9, 0.6               # nothing before the block's first episode
    expected[35:45], expected[45:] = 1.0 - 0.9, 1.0 - 0.6   # the first block's home does not reach the second
    assert np.array_equal(evidence.home, expected, equal_nan=True), np.flatnonzero(evidence.home != expected)


def test_touch_hold_off_is_nan():
    evidence = touch(hold_scene(), hold=0.5)
    expected = np.full(60, np.nan)
    expected[5:15], expected[35:45] = 0.9, 1.0 - 0.9               # 0.5 is not written as a value: NaN is not 0.5
    assert np.array_equal(evidence.home, expected, equal_nan=True)
    assert np.array_equal(evidence.home, touch(hold_scene()).home, equal_nan=True), "hold defaults to off"


def test_touch_turnover_lands_on_new_control():
    n = 80
    slots = {0: beside_ball(n, range(10, 30)), 1: beside_ball(n, range(50, 70))}   # home, flight 30..49, away
    scene = touch_scene(n, slots=slots)
    options = dict(switch_team=1.0, switch_dead=100.0, min_possession=1, min_dead=1)
    held = hmm_labels(hmm_context(touch(scene, hold=0.55).home), **options)
    assert letters(held)[10:] == "H" * 40 + "A" * 30, letters(held)       # the first away control frame
    free = hmm_labels(hmm_context(touch(scene).home), **options)
    assert letters(free)[10:] == "H" * 20 + "A" * 50, letters(free)       # the tie rule: first frame after home's last


def touch_loop(run, radius, max_rel_speed, min_frames, confidence, duel_radius, hold):
    """The rule, frame by frame in plain Python, as the Design states it."""
    from pipeline.evidence import kinematics
    ball, players = kinematics.ball_track(run.inputs), kinematics.player_track(run.inputs)
    team = run.inputs.atd.slot_team
    home = np.full(len(ball.visible), np.nan)
    for start, stop in kinematics.blocks(run.inputs):
        frame_team = {}
        for t in range(start, stop):
            if not ball.visible[t]:
                continue
            distance = {s: float(np.hypot(*(players.xy_m[t, s] - ball.xy_m[t])))
                        for s in range(len(team)) if players.present[t, s]}
            if not distance:
                continue
            nearest = min(distance, key=lambda s: (distance[s], s))
            rivals = [d for s, d in distance.items() if team[s] != team[nearest]]
            relative = float(np.hypot(*(ball.vel[t] - players.vel[t, nearest])))
            if distance[nearest] <= radius and (not rivals or min(rivals) > duel_radius) and relative <= max_rel_speed:
                frame_team[t] = int(team[nearest])
        kept, t = {}, start
        while t < stop:
            if t not in frame_team:
                t += 1
                continue
            end = t
            while end + 1 < stop and frame_team.get(end + 1) == frame_team[t]:
                end += 1
            if end - t + 1 >= min_frames:
                kept.update({u: frame_team[t] for u in range(t, end + 1)})
            t = end + 1
        last = None
        for t in range(start, stop):
            if t in kept:
                last = kept[t]
                home[t] = confidence if last == 0 else 1.0 - confidence
            elif hold > 0.5 and last is not None and ball.visible[t]:
                home[t] = hold if last == 0 else 1.0 - hold
    return home


def test_touch_matches_a_loop():
    rng = np.random.default_rng(7)
    seen = {"home": 0, "away": 0, "held": 0}

    def walk(shape):   # a random walk from a point of a 4 m square
        return 15.0 + rng.random(shape[1:]) * 4.0 + np.cumsum(rng.normal(0.0, 0.05, shape), axis=0)

    for scene in range(200):
        n, slots = 80, 6
        ball = walk((n, 2))
        players = np.stack([walk((n, 2)) for _ in range(slots)], axis=1)
        for track, shape in ((players, (n, slots)), (ball, (n,))):
            track[np.cumsum(rng.random(shape) < 0.04, axis=0) % 2 == 1] = np.nan   # on, off, on ... in runs
        parameters = dict(radius=float(rng.uniform(1.0, 3.0)), duel_radius=float(rng.uniform(0.5, 2.5)),
                          max_rel_speed=float(rng.choice([1.0, 2.0, 4.0, np.inf])),
                          min_frames=int(rng.integers(1, 7)), confidence=float(rng.uniform(0.6, 1.0)))
        parameters["hold"] = 0.5 if scene % 2 else float(rng.uniform(0.5, parameters["confidence"]))
        run = touch_scene(n, ball=ball, slots={s: players[:, s] for s in range(slots)}, graded=[(5, 40), (45, 78)])
        evidence = touch(run, **parameters)
        expected = touch_loop(run, **parameters)
        assert np.array_equal(evidence.home, expected, equal_nan=True), (scene, parameters)
        seen["home"] += int((evidence.home > 0.5).sum())
        seen["away"] += int((evidence.home < 0.5).sum())
        if parameters["hold"] > 0.5:
            seen["held"] += int(np.isin(evidence.home, (parameters["hold"], 1.0 - parameters["hold"])).sum())
    assert min(seen.values()) > 100, "the random scenes no longer reach every branch: {}".format(seen)


def test_touch_rejects_bad_parameters():
    run = touch_scene(20)
    for name, bads in (("radius", (0, -1.0, True, np.inf, np.nan, "2")),
                       ("duel_radius", (0, -1.0, True, np.inf, np.nan)),
                       ("max_rel_speed", (0, -1.0, True, np.nan)), ("min_frames", (0, 1.5, True, -2)),
                       ("confidence", (0.5, 0.4, 1.1, True, np.nan)), ("hold", (0.4, 0.95, True, np.nan))):
        for bad in bads:
            expect_error(ValueError, lambda: touch(run, **{name: bad}), "controlled_touch", name)
    touch(run, radius=np.float64(2.0), min_frames=np.int64(3), max_rel_speed=np.inf, confidence=1, hold=0.9)


def test_touch_reads_read_only_inputs():
    n = 40
    run = touch_scene(n, slots={0: beside_ball(n, range(10, 25))})
    atd = run.inputs.atd
    for array in (atd.players_xy, atd.ball_xy, atd.slot_team):
        array.flags.writeable = False
    before = (atd.players_xy.copy(), atd.ball_xy.copy())
    evidence = touch(run, hold=0.6)
    assert np.array_equal(atd.players_xy, before[0], equal_nan=True)
    assert np.array_equal(atd.ball_xy, before[1], equal_nan=True)
    assert (evidence.home[10:25] == 0.9).all() and evidence.home.dtype == np.float64


CONTROLLED_TRIAL_CONFIG = """
[[evidence]]
kind = "controlled_touch"
radius = 200.0
duel_radius = 0.001
max_rel_speed = inf
min_frames = 1
confidence = 0.9
hold = 0.6

[decoder]
kind = "hmm"
switch_team = 1
switch_dead = 20
min_possession = 1
min_dead = 1
"""


def test_run_end_to_end_synthetic_touch():
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        config = load_config(write_text(root, "trial.toml", CONTROLLED_TRIAL_CONFIG))
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the hmm decoder never abstains inside a block"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        home = result.evidence["controlled_touch"].home
        assert (home > 0.5).sum() > 20 and (home < 0.5).sum() > 20, "the trial must give evidence for both teams"
        assert np.isnan(home[outside]).all()


# -- restart_team -----------------------------------------------------------------------------------------------------
#
# Hand-built scenes of 200 frames in metres, as touch_scene's: slot 0 is home, slot 1 away, slot 2 home, ... unless a
# scene gives `teams`, and the ball is missing unless a scene gives one. A synthetic P(alive) of 0.05 on [50, 150) and
# 0.95 elsewhere makes one stoppage whose restart is frame 150, so its setup window is [100, 150). Type codes are
# restart_team.TYPES's: 0 kick-off, 1 corner, 2 goal kick, 3 throw-in, 4 free kick; teams 0 home, 1 away, -1 none.

RESTART = dict(tail=50, use_ball=False, setup=50, min_stoppage=25)
RESTART_FRAMES = 200
CORNER_SCENE = {0: (0.5, 0.5), 1: (0.5, 6.5)}       # a home taker at a corner point, the nearest away slot 6 m off


def restart_scene(slots=None, ball=None, graded=None, teams=None, n=RESTART_FRAMES):
    """A RunContext over n frames of hand-built tracking in metres, built like touch_scene's. `slots` is a dict slot ->
    a point (2,) or a track (n, 2), or an (n, S, 2) array (NaN = absent); `ball` a point or (n, 2) (default missing
    throughout); `teams` the (S,) slot teams (default 0, 1, 0, 1, ...); `graded` a list of (start, stop) blocks (default
    one over every frame). The inputs also carry the `game` and `kickoffs` that dead_ball_model reads."""
    from pipeline.contracts import RunContext
    from pipeline.inputs.atd_cache import Atd
    scale = np.array(PLAY_PITCH)
    if isinstance(slots, dict):
        players = np.full((n, max(max(slots) + 1, 2), 2), np.nan)
        for slot, track in slots.items():
            players[:, slot] = track
    else:
        players = np.array(slots, dtype=np.float64)
    ball = np.full((n, 2), np.nan) if ball is None else np.broadcast_to(np.asarray(ball, dtype=np.float64), (n, 2))
    teams = np.arange(players.shape[1]) % 2 if teams is None else np.asarray(teams)
    atd = Atd(players_xy=players / scale, players_speed=np.full(players.shape[:2], np.nan), ball_xy=ball / scale,
              ball_speed=np.full(n, np.nan), slot_team=teams.astype(np.int8))
    graded = [(1, 0, n)] if graded is None else [(index + 1, start, stop) for index, (start, stop) in enumerate(graded)]
    inputs = kinematics_inputs(atd, graded, pitch=PLAY_PITCH)
    inputs.game, inputs.kickoffs = "BAR-ATH", {}
    return RunContext(inputs=inputs, seed=0)


def restart_alive(*dips, n=RESTART_FRAMES):
    """P(alive): 0.95, and 0.05 on every [start, stop) of `dips` (default the scenes' stoppage, [50, 150))."""
    alive = np.full(n, 0.95)
    for start, stop in dips or ((50, 150),):
        alive[start:stop] = 0.05
    return alive


def restart_calls(run, alive=None, **settings):
    from pipeline.evidence import restart_team
    return restart_team.calls_from_alive(run.inputs, restart_alive() if alive is None else alive,
                                         **dict(RESTART, **settings))


def restart_pairs(run, alive=None, **settings):
    """[(type, team), ...]: the calls on a scene, one per stoppage with an arrangement."""
    calls = restart_calls(run, alive, **settings)
    return [(int(kind), int(team)) for kind, team in zip(calls["type"], calls["team"])]


def point_calls(points, teams=None, ball=None, use_ball=False, home_left=True, ball_speed=0.0):
    """frame_calls on one frame: `points` a list of (x, y) metres (NaN = absent), `teams` theirs (default 0, 1, 0, ...),
    `ball` a point or None (missing), still unless `ball_speed` says otherwise. Returns the frame's (type, team)."""
    from pipeline.evidence import restart_team
    xy = np.asarray(points, dtype=np.float64)[None]
    teams = np.arange(xy.shape[1]) % 2 if teams is None else np.asarray(teams)
    ball_xy = np.full((1, 2), np.nan) if ball is None else np.asarray(ball, dtype=np.float64)[None]
    types, names = restart_team.frame_calls(xy, ~np.isnan(xy[..., 0]), teams, ball_xy, ~np.isnan(ball_xy[:, 0]),
                                            np.full(1, ball_speed), PLAY_PITCH, home_left, use_ball)
    assert types.dtype == np.int8 and names.dtype == np.int8 and types.shape == names.shape == (1,)
    return int(types[0]), int(names[0])


def restart_model_scene(slots, **options):
    """restart_scene with a still ball far from everyone, missing on [50, 150): write_model's stub model then makes
    one stoppage there, restarting at 150."""
    ball = np.tile([90.0, 10.0], (RESTART_FRAMES, 1))
    ball[50:150] = np.nan
    return restart_scene(slots, ball=ball, **options)


def restart_compute(run, path, **parameters):
    from pipeline.contracts import check_evidence
    from pipeline.evidence import REGISTRY
    evidence = REGISTRY["restart_team"](run, **dict(dict(model=path, tail=30), **parameters))
    check_evidence(evidence, len(run.inputs.atd.ball_xy))
    assert np.isnan(evidence.alive).all(), "restart_team has no opinion on whether the ball is in play"
    return evidence


def test_restart_team_corner_names_the_taker():
    from pipeline.evidence import restart_team
    assert restart_team.TYPES == ("kick_off", "corner", "goal_kick", "throw_in", "free_kick")
    assert restart_pairs(restart_scene(CORNER_SCENE)) == [(1, 0)]
    assert restart_pairs(restart_scene(CORNER_SCENE, teams=[1, 0])) == [(1, 1)]
    length, width = PLAY_PITCH
    for cx, cy in ((0.0, 0.0), (0.0, width), (length, 0.0), (length, width)):   # every corner point
        sx, sy = (1.0 if cx == 0 else -1.0), (1.0 if cy == 0 else -1.0)
        taker, rival = (cx + 0.5 * sx, cy + 0.5 * sy), (cx + 0.5 * sx, cy + 6.5 * sy)
        assert point_calls([taker, rival]) == (1, 0), (cx, cy)
        assert point_calls([taker, rival], teams=[1, 0]) == (1, 1), (cx, cy)
    # an opponent within 4.5 m is no corner (the taker still has the touchline to himself: a throw-in), and a player
    # 3.5 m from the corner point is not at it
    assert point_calls([(0.5, 0.5), (0.5, 4.5)]) == (3, 0)
    assert point_calls([(3.0, 1.8), (20.0, 30.0)]) == (-1, -1)


def test_restart_team_goal_kick_names_the_defender():
    from pipeline.evidence import kinematics, play_features
    home = [(3.0, 34.0), (25.0, 20.0), (25.0, 48.0)]            # the keeper in his goal area, two defenders
    away = [(70.0, 20.0), (70.0, 48.0), (12.0, 34.0)]           # the last one inside home's penalty area
    run = restart_scene(dict(enumerate(home + away)), teams=[0, 0, 0, 1, 1, 1])
    assert play_features.own_half_is_left(kinematics.player_track(run.inputs), run.inputs.atd.slot_team, 0,
                                          RESTART_FRAMES, PLAY_PITCH[0]), "home defends x < L/2 in this scene"
    assert restart_pairs(run) == [(2, 0)]
    crowded = restart_scene(dict(enumerate(home + away + [(14.0, 25.0)])), teams=[0, 0, 0, 1, 1, 1, 1])
    assert restart_pairs(crowded) == [], "two opponents inside the penalty area"
    # the other end is away's: its keeper there is away's goal kick, and an away slot in home's goal area is nobody's
    assert point_calls([(30.0, 34.0), (PLAY_PITCH[0] - 3.0, 34.0)]) == (2, 1)
    assert point_calls([(60.0, 30.0), (3.0, 34.0)]) == (-1, -1)
    # the goal area is 5.5 m deep and 18.32 m wide: its line is in, a step past it is out
    assert point_calls([(5.5, 43.0), (80.0, 34.0)]) == (2, 0)
    assert point_calls([(5.6, 34.0), (80.0, 34.0)]) == (-1, -1)


def test_restart_team_throw_in_names_the_thrower():
    assert restart_pairs(restart_scene({0: (40.0, 65.4), 1: (40.0, 68.4)})) == [(3, 1)]   # away 0.4 m beyond the line
    assert point_calls([(40.0, 1.4), (40.0, 4.4)]) == (3, 0)                               # the near touchline, inside
    assert point_calls([(40.0, 1.6), (40.0, 4.6)]) == (-1, -1)                             # 1.6 m in: not at the line
    assert point_calls([(40.0, 67.5), (40.0, 68.4)]) == (-1, -1)                           # 0.9 m apart: neither throws


def test_restart_team_kick_off_names_the_team_in_the_circle():
    players, teams = kickoff_formation(RESTART_FRAMES)
    assert restart_pairs(restart_scene(players, teams=teams)) == [(0, 0)]
    # the same formation with the labels exchanged: away is now the team at the centre spot, home defends the right
    assert restart_pairs(restart_scene(players, teams=1 - teams)) == [(0, 1)]
    # a home slot 0.85 m past the halfway line is still in its half (1 m of tolerance); 3.35 m past it is not
    inside = players.copy()
    inside[:, 5, 0] = 53.5
    assert restart_pairs(restart_scene(inside, teams=teams)) == [(0, 0)]
    beyond = players.copy()
    beyond[:, 5, 0] = 56.0
    assert restart_pairs(restart_scene(beyond, teams=teams)) == []
    # both teams inside the centre circle: no kick-off
    both = players.copy()
    both[:, 11] = (58.0, 30.0)
    assert restart_pairs(restart_scene(both, teams=teams)) == []


def test_restart_team_free_kick_needs_the_ball():
    slots = {0: (20.0, 34.0), 1: (75.5, 30.0), 2: (85.0, 30.0), 3: (60.0, 50.0)}
    still = restart_scene(slots, ball=(75.0, 30.0))
    assert restart_pairs(still) == [], "use_ball off: no free kick, and nothing else fires"
    assert restart_pairs(still, use_ball=True) == [(4, 1)]
    assert restart_pairs(restart_scene(slots), use_ball=True) == [], "no ball, no free kick"
    # a rolling ball (2 m/s) with its player beside it, an opponent within 4.5 m, a taker 2.5 m off: no free kick
    t = np.arange(RESTART_FRAMES)[:, None]
    rolling = np.array([75.0, 30.0]) + t * np.array([0.08, 0.0])
    moving = restart_scene({**slots, 1: rolling + [0.5, 0.0], 2: (40.0, 60.0)}, ball=rolling)   # nobody else near
    assert restart_pairs(moving, use_ball=True) == []
    assert restart_pairs(restart_scene({**slots, 2: (78.0, 30.0)}, ball=(75.0, 30.0)), use_ball=True) == []
    assert restart_pairs(restart_scene({**slots, 1: (77.5, 30.0)}, ball=(75.0, 30.0)), use_ball=True) == []


def test_restart_team_ball_only_places_the_rules():
    centre = (PLAY_PITCH[0] / 2, PLAY_PITCH[1] / 2)
    # a corner arrangement (home taker, away 6 m off) with the ball still at the centre spot, a home slot beside it and
    # an away slot 3.1 m from it: without the ball, the corner
    corner = [(0.5, 0.5), (0.5, 6.5), (53.0, 34.0), (55.0, 36.0)]
    assert point_calls(corner) == (1, 0)
    assert point_calls(corner, ball=centre) == (1, 0), "use_ball off: the ball is not read"
    # with it, only the kick-off rule runs there (and fails: both teams in the circle), the corner rule is silent
    assert point_calls(corner, ball=centre, use_ball=True) == (-1, -1)
    assert point_calls(corner, ball=(0.2, 0.3), use_ball=True) == (1, 0), "the ball at the corner: the corner rule"
    # a corner taker 2.2 m from the corner point, beside a ball 3.8 m from it on the touchline zone's edge: the ball
    # places a throw-in there, not a corner
    beside = [(2.0, 1.0), (2.0, 7.0)]
    assert point_calls(beside) == (1, 0)
    assert point_calls(beside, ball=(3.5, 1.5), use_ball=True) == (3, 0)
    # a kick-off arrangement with a home slot at a corner: the ball at the centre spot is the kick-off, at the corner
    # the corner, anywhere else (nobody beside it) nothing
    kickoff = [(53.0, 34.0), (60.0, 20.0), (0.5, 0.5)]
    assert point_calls(kickoff) == (0, 0)
    assert point_calls(kickoff, ball=centre, use_ball=True) == (0, 0)
    assert point_calls(kickoff, ball=(0.2, 0.3), use_ball=True) == (1, 0)
    assert point_calls(kickoff, ball=(75.0, 30.0), use_ball=True) == (-1, -1)
    # the taker is the player nearest the ball, within 2 m: 2.52 m from a corner ball is a corner without the ball only
    assert point_calls([(2.5, 0.3), (80.0, 30.0)]) == (1, 0)
    assert point_calls([(2.5, 0.3), (80.0, 30.0)], ball=(0.0, 0.0), use_ball=True) == (-1, -1)
    # goal kick: the ball in the goal area grown by 1 m (0.8 m past its line) is a goal kick, 1.3 m past it is not,
    # and the still ball with the keeper 1.8 m off is then a free kick
    keeper = [(5.0, 34.0), (70.0, 30.0)]
    assert point_calls(keeper, ball=(6.3, 34.0), use_ball=True) == (2, 0)
    assert point_calls(keeper, ball=(6.8, 34.0), use_ball=True) == (4, 0)
    # throw-in: the ball beyond the line is one, the ball 2 m inside it is not (and an opponent 3 m off stops a free
    # kick)
    thrower = [(40.0, 0.5), (40.0, 5.0)]
    assert point_calls(thrower) == (3, 0)
    assert point_calls(thrower, ball=(40.5, -0.3), use_ball=True) == (3, 0)
    assert point_calls(thrower, ball=(40.0, 2.0), use_ball=True) == (-1, -1)


def test_restart_team_stoppages_and_restart_frame():
    from pipeline.evidence import restart_team
    stoppages = restart_team.stoppages
    assert stoppages(restart_alive(), 0, 200, 25) == [(50, 150)]
    assert all(type(value) is int for value in stoppages(restart_alive(), 0, 200, 25)[0])
    assert stoppages(restart_alive((50, 70)), 0, 200, 25) == [], "a 20-frame dip"
    assert stoppages(restart_alive((50, 75)), 0, 200, 25) == [(50, 75)], "exactly min_stoppage frames"
    assert stoppages(restart_alive((150, 200)), 0, 200, 25) == [], "a dip to the block's end has no restart"
    assert stoppages(restart_alive((50, 150)), 0, 120, 25) == [], "nor one past the end of [start, stop)"
    assert stoppages(restart_alive((0, 40)), 0, 200, 25) == [(0, 40)], "a run from the block's first frame"
    assert stoppages(restart_alive((0, 40)), 20, 200, 25) == [], "only 20 of its frames are in [20, 200)"
    assert stoppages(restart_alive((10, 40), (60, 100)), 0, 200, 25) == [(10, 40), (60, 100)]
    exactly = restart_alive()
    exactly[100] = 0.5                                     # 0.5 is not dead: it splits the run in two
    assert stoppages(exactly, 0, 200, 25) == [(50, 100), (101, 150)]
    unknown = restart_alive()
    unknown[150:160] = np.nan                              # NaN ends a run, and is no restart
    assert stoppages(unknown, 0, 200, 25) == []
    unknown[100] = np.nan
    assert stoppages(unknown, 0, 200, 25) == [], "a run ended by NaN has no restart either"


def test_restart_team_votes_only_after_the_restart():
    from pipeline.evidence import restart_team
    calls = restart_calls(restart_scene(CORNER_SCENE), tail=30)
    assert set(calls) == {"first", "restart", "end", "type", "team"}
    assert calls["first"].tolist() == [50] and calls["restart"].tolist() == [150] and calls["end"].tolist() == [180]
    home = restart_team.votes(calls, RESTART_FRAMES, {1: 0.9})
    expected = np.full(RESTART_FRAMES, np.nan)
    expected[150:180] = 0.9
    assert home.dtype == np.float64 and np.array_equal(home, expected, equal_nan=True), np.flatnonzero(home == 0.9)
    away = restart_calls(restart_scene(CORNER_SCENE, teams=[1, 0]), tail=30)
    expected[150:180] = 1.0 - 0.9
    assert np.array_equal(restart_team.votes(away, RESTART_FRAMES, {1: 0.9}), expected, equal_nan=True)
    # a tail that runs past the block's end stops there
    late = restart_calls(restart_scene(CORNER_SCENE), alive=restart_alive((100, 180)), tail=50)
    assert late["restart"].tolist() == [180] and late["end"].tolist() == [200]


def test_restart_team_tail_stops_at_the_next_stoppage():
    from pipeline.evidence import restart_team
    scene = restart_scene(CORNER_SCENE)
    calls = restart_calls(scene, alive=restart_alive((50, 150), (165, 195)), tail=30)
    assert calls["restart"].tolist() == [150, 195] and calls["end"].tolist() == [165, 200]
    expected = np.full(RESTART_FRAMES, np.nan)
    expected[150:165] = expected[195:200] = 0.9
    assert np.array_equal(restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}), expected, equal_nan=True)
    # a next stoppage that runs to the block's end has no restart of its own, and still ends the vote
    calls = restart_calls(scene, alive=restart_alive((50, 150), (165, 200)), tail=30)
    assert calls["restart"].tolist() == [150] and calls["end"].tolist() == [165]
    # a dip shorter than min_stoppage is no stoppage, and does not
    calls = restart_calls(scene, alive=restart_alive((50, 150), (160, 170)), tail=30)
    assert calls["end"].tolist() == [180]


def test_restart_team_type_without_confidence_is_silent():
    from pipeline.evidence import restart_team
    calls = restart_calls(restart_scene(CORNER_SCENE))
    assert calls["type"].tolist() == [1]
    for confidences in ({}, {0: 0.9, 2: 0.9, 3: 0.9, 4: 0.9}):
        assert np.isnan(restart_team.votes(calls, RESTART_FRAMES, confidences)).all(), confidences
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        run = restart_model_scene(CORNER_SCENE)
        assert np.isnan(restart_compute(run, path, kick_off=0.9, goal_kick=0.9, throw_in=0.9).home).all()
        home = restart_compute(run, path, corner=0.8).home
        assert (home[150:180] == 0.8).all() and np.isnan(np.delete(home, range(150, 180))).all()


def test_restart_team_arrangement_needs_a_quarter():
    from pipeline.evidence import restart_team
    assert restart_team.MIN_WINDOW_SHARE == 0.25

    def window(*pairs, length=50):   # (type, team, frames) in turn, the rest of the window no call
        types, teams = np.full(length, -1, np.int8), np.full(length, -1, np.int8)
        at = 0
        for kind, team, frames in pairs:
            types[at:at + frames], teams[at:at + frames] = kind, team
            at += frames
        return types, teams

    arrangement = restart_team.arrangement
    assert arrangement(*window((3, 1, 12)), 0.25) is None
    assert arrangement(*window((3, 1, 13)), 0.25) == (3, 1)
    assert all(type(value) is int for value in arrangement(*window((3, 1, 13)), 0.25))
    assert arrangement(*window((3, 1, 13), (1, 0, 13)), 0.25) == (1, 0), "a tie goes to the lower type code"
    assert arrangement(*window((2, 1, 13), (2, 0, 13)), 0.25) == (2, 0), "then to home"
    assert arrangement(*window((4, 0, 14), (1, 1, 13)), 0.25) == (4, 0), "the most frames, whatever the priority"
    assert arrangement(*window(), 0.25) is None
    assert arrangement(*window(length=0), 0.25) is None
    assert arrangement(*window((1, 0, 2), length=8), 0.25) == (1, 0), "a quarter of a short window is of its length"
    assert arrangement(*window((1, 0, 1), length=8), 0.25) is None


def test_restart_team_block_edges():
    """Review Focus 1: a stoppage from a block's first frame, or to its last, votes nothing across the edge."""
    from pipeline.evidence import restart_team
    graded = [(0, 100), (100, 200)]
    scene = restart_scene(CORNER_SCENE, graded=graded)
    # from the second block's first frame (a pre-kick-off run): the call, and its vote inside the block
    calls = restart_calls(scene, alive=restart_alive((100, 130)), tail=50)
    assert calls["restart"].tolist() == [130] and calls["end"].tolist() == [180]
    # to the first block's last frame: no restart there, so no call, and nothing in the second block
    assert len(restart_calls(scene, alive=restart_alive((60, 100)))["restart"]) == 0
    # across the edge: the first block's piece has no restart, the second block's is a stoppage from its first frame
    assert restart_calls(scene, alive=restart_alive((60, 130)))["restart"].tolist() == [130]
    # a tail that would cross the edge stops at it
    calls = restart_calls(scene, alive=restart_alive((40, 80)), tail=50)
    assert calls["restart"].tolist() == [80] and calls["end"].tolist() == [100]
    home = restart_team.votes(calls, RESTART_FRAMES, {1: 0.9})
    assert (home[80:100] == 0.9).all() and np.isnan(home[100:]).all() and np.isnan(home[:80]).all()
    # frames between two blocks: no players, alive NaN, and the tail stops at the first block's end
    alive = restart_alive((40, 80))
    alive[90:110] = np.nan
    calls = restart_calls(restart_scene(CORNER_SCENE, graded=[(0, 90), (110, 200)]), alive, tail=50)
    assert calls["restart"].tolist() == [80] and calls["end"].tolist() == [90]
    # the window never crosses the block's start: 8 corner frames make a quarter of the 30 frames [100, 130), not of
    # 50, also when the frames before the block's start are dead too (the stoppage is cut at the block's edge)
    taker = np.full((RESTART_FRAMES, 2), np.nan)
    taker[100:108] = (0.5, 0.5)
    clipped = restart_scene({0: taker, 1: (0.5, 6.5)}, graded=graded)
    assert restart_pairs(clipped, alive=restart_alive((100, 130))) == [(1, 0)]
    assert restart_pairs(clipped, alive=restart_alive((60, 130))) == [(1, 0)]
    lead = np.full((RESTART_FRAMES, 2), np.nan)
    lead[80:100] = (0.5, 0.5)       # an away corner on the first block's last 20 (dead) frames: it would win [80, 130)
    edge = restart_scene({0: taker, 1: (0.5, 6.5), 2: (40.0, 60.0), 3: lead}, graded=graded)
    assert restart_pairs(edge, alive=restart_alive((60, 130))) == [(1, 0)]


def test_restart_team_reads_only_the_stoppage():
    """The arrangement is read on the stoppage's last `setup` frames, [max(first, restart - setup), restart): a stoppage
    shorter than `setup` is read whole and no further, so the play before it never counts."""
    # a 30-frame stoppage [120, 150): 8 corner frames make a quarter of its 30 frames (not of 50)
    taker = np.full((RESTART_FRAMES, 2), np.nan)
    taker[120:128] = (0.5, 0.5)
    short = restart_scene({0: taker, 1: (0.5, 6.5), 2: (40.0, 60.0)})
    assert restart_pairs(short, alive=restart_alive((120, 150))) == [(1, 0)]
    # an away thrower at the touchline on the 20 live frames before it: a throw-in on 20 of [100, 150)'s 50 frames
    # would win, but those frames are play, and the corner of the stoppage itself stands
    thrower = np.full((RESTART_FRAMES, 2), np.nan)
    thrower[100:120] = (40.0, 67.8)
    live = restart_scene({0: taker, 1: (0.5, 6.5), 2: (40.0, 60.0), 3: thrower})
    assert restart_pairs(live, alive=restart_alive((120, 150))) == [(1, 0)]
    # the same thrower inside a long stoppage [50, 150) is read (window [100, 150)): the throw-in, 20 frames of 50
    assert restart_pairs(live, alive=restart_alive((50, 150))) == [(3, 1)]
    # a stoppage of exactly `setup` frames and one longer read the same last 50 frames
    assert restart_pairs(short, alive=restart_alive((100, 150))) == restart_pairs(short) == []


def test_restart_team_absent_team_gives_no_call():
    """Review Focus 2: a team with no present slot, or no slot at all, gives no call; nor does the kick-off fire."""
    assert restart_pairs(restart_scene({0: (0.5, 0.5), 2: (30.0, 30.0)})) == [], "a corner with no opponent present"
    assert point_calls([(0.5, 0.5), (np.nan, np.nan)]) == (-1, -1)
    players, teams = kickoff_formation(RESTART_FRAMES)
    players[:, 11:] = np.nan                                  # the away team gone
    assert restart_pairs(restart_scene(players, teams=teams)) == []
    assert restart_pairs(restart_scene(nobody(RESTART_FRAMES, 4))) == []
    assert restart_pairs(restart_scene(nobody(RESTART_FRAMES, 4), ball=(75.0, 30.0)), use_ball=True) == []
    # a free kick needs an opponent on the frame too
    assert restart_pairs(restart_scene({0: (75.5, 30.0), 2: (20.0, 34.0)}, ball=(75.0, 30.0)), use_ball=True) == []
    # the away slot present on 12 of the window's 50 frames: too few; on 13, the corner
    rival = np.full((RESTART_FRAMES, 2), np.nan)
    rival[100:112] = (0.5, 6.5)
    assert restart_pairs(restart_scene({0: (0.5, 0.5), 1: rival})) == []
    rival[100:113] = (0.5, 6.5)
    assert restart_pairs(restart_scene({0: (0.5, 0.5), 1: rival})) == [(1, 0)]


def test_restart_team_two_takers_is_deterministic():
    """Review Focus 3: two players qualify for one rule on a frame: the nearer the line (or spot), then the lower
    slot."""
    from pipeline.evidence import restart_team
    length, width = PLAY_PITCH
    assert point_calls([(30.0, 0.8), (60.0, 0.3)]) == (3, 1)
    assert point_calls([(30.0, 0.3), (60.0, 0.8)]) == (3, 0)
    assert point_calls([(30.0, 0.5), (60.0, -0.5)]) == (3, 0), "as far beyond the line as inside it: the lower slot"
    assert point_calls([(30.0, 0.5), (60.0, width - 0.5)]) == (3, 0), "one at each touchline: the lower slot"
    assert point_calls([(60.0, 0.5), (30.0, 0.5)], teams=[1, 0]) == (3, 1), "the lower slot, whichever its team"
    assert point_calls([(1.0, 1.0), (length - 0.5, width - 0.5)]) == (1, 1), "the nearer its corner point"
    assert point_calls([(0.5, 0.5), (length - 0.5, width - 0.5)]) == (1, 0)
    assert point_calls([(3.0, 34.0), (length - 2.0, 34.0)]) == (2, 1), "both keepers: the nearer his goal line"
    # with the ball, the taker is the one nearest the ball, not the one nearest the line
    pair = [(40.0, 0.2), (42.0, 0.6)]
    assert point_calls(pair) == (3, 0)
    assert point_calls(pair, ball=(41.8, 0.0), use_ball=True) == (3, 1)
    # and the same frames give the same calls, every frame and every time
    run = restart_scene({0: (30.0, 0.5), 1: (60.0, 0.5)})
    assert restart_pairs(run) == restart_pairs(run) == [(3, 0)]
    xy = np.tile(np.array([[30.0, 0.5], [60.0, 0.5]])[None], (50, 1, 1))
    arguments = (xy, np.ones((50, 2), bool), np.array([0, 1]), np.full((50, 2), np.nan), np.zeros(50, bool),
                 np.full(50, np.nan), PLAY_PITCH, True, False)
    first, second = restart_team.frame_calls(*arguments), restart_team.frame_calls(*arguments)
    assert (first[0] == 3).all() and (first[1] == 0).all()
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_restart_team_more_than_eleven_a_side():
    """Review Focus 4: the rules read slot_team as given and never assume eleven a side."""
    players, teams = kickoff_formation(RESTART_FRAMES)
    extra = np.tile(np.array([[20.0, 50.0], [30.0, 55.0], [40.0, 60.0], [45.0, 45.0]])[None], (RESTART_FRAMES, 1, 1))
    players, teams = np.concatenate([players, extra], axis=1), np.concatenate([teams, [0, 0, 0, 0]])
    assert restart_pairs(restart_scene(players, teams=teams)) == [(0, 0)], "15 home slots and 11 away: a kick-off"
    wrong = teams.copy()
    wrong[5] = 1                                    # an away tracklet on a player in home's half breaks it, as given
    assert restart_pairs(restart_scene(players, teams=wrong)) == []
    # thirteen home slots crowd a corner, the one away slot far off: teammates are no opponents
    crowd = [(0.5, 0.5)] + [(1.0 + 0.5 * k, 2.0) for k in range(12)] + [(30.0, 30.0)]
    assert point_calls(crowd, teams=[0] * 13 + [1]) == (1, 0)
    # two away slots inside home's penalty area are two intruders, whoever they are
    keeper = [(3.0, 34.0)] + [(25.0, 20.0)] * 12 + [(12.0, 34.0), (12.0, 34.0)]
    assert point_calls(keeper, teams=[0] * 13 + [1, 1]) == (-1, -1)
    assert point_calls(keeper[:-1], teams=[0] * 13 + [1]) == (2, 0)


def test_restart_team_follows_each_blocks_direction():
    """Review Focus 5: the teams swap ends at half time, and each block's defended end comes from that block alone."""
    from pipeline.evidence import play_features
    assert play_features._own_half_is_left is play_features.own_half_is_left, "the old private name is an alias"
    length = PLAY_PITCH[0]
    first = np.array([(3.0, 34.0), (25.0, 20.0), (25.0, 48.0), (70.0, 20.0), (70.0, 48.0), (80.0, 34.0)])
    mirrored = first.copy()
    mirrored[:, 0] = length - mirrored[:, 0]
    players = np.empty((RESTART_FRAMES, 6, 2))
    players[:100], players[100:] = first, mirrored
    teams = [0, 0, 0, 1, 1, 1]
    alive = restart_alive((30, 80), (130, 180))
    run = restart_scene(players, teams=teams, graded=[(0, 100), (100, 200)])
    assert restart_pairs(run, alive=alive) == [(2, 0), (2, 0)], "home's keeper at the left end, then at the right"
    # second half, home on the right: the keeper in the left goal area is away's, whatever the first half said
    swapped = np.array([(80.0, 20.0), (80.0, 48.0), (70.0, 34.0), (3.0, 34.0), (25.0, 20.0), (25.0, 48.0)])
    players[100:] = swapped
    run = restart_scene(players, teams=teams, graded=[(0, 100), (100, 200)])
    assert restart_pairs(run, alive=alive) == [(2, 0), (2, 1)]


def restart_loop(xy, present, teams, ball, visible, speed, pitch, home_left, use_ball):
    """frame_calls' rules frame by frame and slot by slot in plain Python, as the module's docstring states them."""
    length, width = pitch
    centre, corners = (length / 2, width / 2), ((0.0, 0.0), (0.0, width), (length, 0.0), (length, width))

    def apart(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    calls = []
    for t in range(len(xy)):
        here = [s for s in range(len(teams)) if present[t, s] and np.isfinite(xy[t, s]).all()]
        at = {s: (float(xy[t, s, 0]), float(xy[t, s, 1])) for s in here}
        team = {s: int(teams[s]) for s in here}
        if {team[s] for s in here} != {0, 1}:
            calls.append((-1, -1))
            continue

        def rival(s):
            return min([apart(at[s], at[o]) for o in here if team[o] != team[s]])

        def kick_off():
            own = all(at[s][0] <= length / 2 + 1.0 if (team[s] == 0) == home_left else at[s][0] >= length / 2 - 1.0
                      for s in here)
            circle = {s: apart(at[s], centre) for s in here if apart(at[s], centre) <= 9.15}
            return circle if own and len({team[s] for s in circle}) == 1 else {}

        def corner():
            near = {s: min(apart(at[s], c) for c in corners) for s in here}
            return {s: d for s, d in near.items() if d <= 3.0 and rival(s) > 4.5}

        def goal_kick(left):
            def depth(point):
                return point[0] if left else length - point[0]
            defender = 0 if left == home_left else 1
            intruders = sum(1 for s in here if team[s] != defender and 0 <= depth(at[s]) <= 16.5
                            and abs(at[s][1] - width / 2) <= 20.16)
            return {s: depth(at[s]) for s in here if team[s] == defender and 0 <= depth(at[s]) <= 5.5
                    and abs(at[s][1] - width / 2) <= 9.16 and intruders <= 1}

        def throw_in():
            return {s: min(abs(at[s][1]), abs(width - at[s][1])) for s in here
                    if (at[s][1] <= 1.5 or at[s][1] >= width - 1.5) and rival(s) > 1.0}

        call = (-1, -1)
        if use_ball and visible[t] and np.isfinite(ball[t]).all():
            b = (float(ball[t, 0]), float(ball[t, 1]))
            taker = min(here, key=lambda s: (apart(at[s], b), s))
            if apart(at[taker], b) <= 2.0:
                def in_area(left):
                    depth = b[0] if left else length - b[0]
                    return np.hypot(max(-depth, depth - 5.5, 0.0), max(abs(b[1] - width / 2) - 9.16, 0.0)) <= 1.0
                checks = (apart(b, centre) <= 2.0 and taker in kick_off(),
                          min(apart(b, c) for c in corners) <= 3.0 and taker in corner(),
                          any(in_area(left) and taker in goal_kick(left) for left in (True, False)),
                          (b[1] <= 1.5 or b[1] >= width - 1.5) and taker in throw_in(),
                          speed[t] < 1.0 and all(apart(at[s], b) > 4.5 for s in here if team[s] != team[taker]))
                for code, fires in enumerate(checks):
                    if fires:
                        call = (code, team[taker])
                        break
        else:
            for code, rule in enumerate((kick_off, corner, lambda: {**goal_kick(True), **goal_kick(False)}, throw_in)):
                meets = rule()
                if meets:
                    call = (code, team[min(meets, key=lambda s: (meets[s], s))])
                    break
        calls.append(call)
    return calls


def test_restart_team_matches_a_loop():
    """frame_calls, vectorised over slots and frames, against restart_loop on random scenes that reach every rule."""
    from pipeline.evidence import play_features, restart_team
    pinned = dict(CENTRE_CIRCLE_M=9.15, OWN_HALF_TOLERANCE_M=1.0, CORNER_ZONE_M=3.0, CORNER_CLEARANCE_M=4.5,
                  GOAL_AREA_DEPTH_M=5.5, GOAL_AREA_WIDTH_M=18.32, PENALTY_AREA_DEPTH_M=16.5,
                  PENALTY_AREA_WIDTH_M=40.32, GOAL_KICK_MAX_INTRUDERS=1, TOUCHLINE_ZONE_M=1.5,
                  THROW_IN_CLEARANCE_M=1.0, TAKER_BALL_M=2.0, FREE_KICK_CLEARANCE_M=4.5, KICK_OFF_BALL_M=2.0,
                  GOAL_AREA_BALL_TOLERANCE_M=1.0, MIN_WINDOW_SHARE=0.25, ALIVE_LEVEL=0.5)
    assert {name: getattr(restart_team, name) for name in pinned} == pinned, "a constant of the design moved"
    assert play_features.STILL_SPEED == 1.0
    rng = np.random.default_rng(23)
    length, width = PLAY_PITCH
    # the places the rules look at, as (x, y, spread in metres)
    spots = [(length / 2, width / 2, 6.0), (1.0, 1.0, 2.0), (length - 1.0, width - 1.0, 2.0), (3.0, width / 2, 4.0),
             (length - 3.0, width / 2, 4.0), (30.0, 0.5, 1.5), (70.0, width - 0.5, 1.5), (12.0, width / 2, 8.0)]
    seen = {(code, mode): 0 for code in range(5) for mode in (False, True) if code < 4 or mode}
    for scene in range(300):
        n, slots = 12, int(rng.integers(4, 15))
        home_left, use_ball = bool(rng.random() < 0.5), bool(scene % 3)
        teams = rng.integers(0, 2, slots)
        # each slot near a spot or anywhere; in one scene of four, everyone in their own half and some at the centre
        # spot (kick-offs). The ball beside a random slot, at a spot, or missing.
        xy = np.empty((n, slots, 2))
        for s in range(slots):
            if scene % 4 == 0:
                left = (teams[s] == 0) == home_left
                xy[:, s] = ((0.0 if left else length / 2) + rng.random() * length / 2, rng.random() * width)
                if rng.random() < 0.25:
                    xy[:, s] = (length / 2, width / 2) + rng.normal(0.0, 4.0, 2)
            elif rng.random() < 0.5:
                sx, sy, spread = spots[rng.integers(len(spots))]
                xy[:, s] = (sx, sy) + rng.normal(0.0, spread, 2)
            else:
                xy[:, s] = rng.random(2) * (length, width)
        xy += rng.normal(0.0, 0.3, xy.shape)
        present = rng.random((n, slots)) < 0.85
        xy[~present & (rng.random((n, slots)) < 0.5)] = np.nan      # absent slots, with or without a position
        xy[present & (rng.random((n, slots)) < 0.02)] = np.nan      # and a present slot without one
        ball = xy[np.arange(n), rng.integers(0, slots, n)] + rng.normal(0.0, 1.0, (n, 2))
        spot = rng.random(n) < 0.3
        ball[spot] = np.array([spots[k][:2] for k in rng.integers(0, len(spots), spot.sum())]).reshape(-1, 2)
        if scene % 4 == 0:
            kick = rng.random(n) < 0.6
            ball[kick] = (length / 2, width / 2) + rng.normal(0.0, 1.0, (kick.sum(), 2))
            xy[:, 0] = ball[0] + rng.normal(0.0, 0.5, 2)                # and a slot beside it
        visible = rng.random(n) < 0.8
        ball[~visible & (rng.random(n) < 0.5)] = np.nan
        ball[visible & (rng.random(n) < 0.03)] = np.nan
        speed = np.where(rng.random(n) < 0.7, rng.random(n) * 0.9, 1.0 + rng.random(n) * 5.0)
        speed[rng.random(n) < 0.05] = np.nan
        types, named = restart_team.frame_calls(xy, present, teams, ball, visible, speed, PLAY_PITCH, home_left,
                                                use_ball)
        expected = restart_loop(xy, present, teams, ball, visible, speed, PLAY_PITCH, home_left, use_ball)
        got = list(zip(types.tolist(), named.tolist()))
        assert got == expected, (scene, [(t, a, b) for t, (a, b) in enumerate(zip(got, expected)) if a != b][:5])
        ball_mode = use_ball & visible & np.isfinite(ball).all(axis=1)
        for (code, _), mode in zip(got, ball_mode.tolist()):
            if code >= 0:
                seen[(code, mode)] += 1
    assert min(seen.values()) >= 10, "the random scenes no longer reach every rule in both modes: {}".format(seen)


def test_restart_team_frame_calls_across_a_chunk():
    """frame_calls on more frames than one CHUNK: the same calls as restart_loop, and as the frames called in pieces."""
    from pipeline.evidence import restart_team
    n, slots = 300, 14
    assert n > restart_team.CHUNK, "the check must span two chunks"
    rng = np.random.default_rng(29)
    length, width = PLAY_PITCH
    anchors = np.array([(1.0, 1.0), (3.0, width / 2), (40.0, 0.5), (length / 2, width / 2), (70.0, width - 0.5),
                        (length - 1.0, 1.0), (60.0, 30.0)])
    start = anchors[rng.integers(0, len(anchors), slots)] + rng.normal(0.0, 2.0, (slots, 2))
    xy = start[None] + np.cumsum(rng.normal(0.0, 0.15, (n, slots, 2)), axis=0)     # slots wandering near the spots
    teams = np.arange(slots) % 2
    present = rng.random((n, slots)) < 0.9
    ball = xy[np.arange(n), rng.integers(0, slots, n)] + rng.normal(0.0, 0.8, (n, 2))
    visible = rng.random(n) < 0.7
    speed = np.where(rng.random(n) < 0.6, 0.5, 3.0)
    for use_ball in (False, True):
        rest = (PLAY_PITCH, True, use_ball)
        types, named = restart_team.frame_calls(xy, present, teams, ball, visible, speed, *rest)
        expected = restart_loop(xy, present, teams, ball, visible, speed, *rest)
        assert list(zip(types.tolist(), named.tolist())) == expected, use_ball
        pieces = [restart_team.frame_calls(xy[a:b], present[a:b], teams, ball[a:b], visible[a:b], speed[a:b], *rest)
                  for a, b in ((0, 137), (137, n))]
        assert np.array_equal(types, np.concatenate([piece[0] for piece in pieces])), use_ball
        assert np.array_equal(named, np.concatenate([piece[1] for piece in pieces])), use_ball
        chunk = restart_team.CHUNK
        assert (types[:chunk] >= 0).sum() >= 20 and (types[chunk:] >= 0).sum() >= 10, "too few calls to compare"
        assert len(set(types[types >= 0].tolist())) >= 2, "the scene calls a single type"


def test_restart_team_rejects_bad_parameters():
    from pipeline.evidence import REGISTRY, restart_team
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        run = restart_model_scene(CORNER_SCENE)
        source = REGISTRY["restart_team"]
        good = dict(model=path, tail=30, corner=0.8)
        for name, bads in (("tail", (0, -1, 1.5, True, "30", None, np.inf)), ("setup", (0, -5, 2.5, True)),
                           ("min_stoppage", (0, 1.5, False)), ("use_ball", ("yes", 1, None, 0.0)),
                           ("min_prob", (0.0, 0.5, True))):
            for bad in bads:
                expect_error(ValueError, lambda: source(run, **dict(good, **{name: bad})), "restart_team", name)
        for name in restart_team.TYPES:
            for bad in (0.5, 0.4, 1.2, True, float("nan"), "0.9"):
                expect_error(ValueError, lambda: source(run, **dict(good, use_ball=True, **{name: bad})),
                             "restart_team", name)
        expect_error(ValueError, lambda: source(run, model=path, tail=30, free_kick=0.8), "restart_team", "free_kick",
                     "use_ball")
        expect_error(ValueError, lambda: source(run, model=path, tail=30), "restart_team", "confidence",
                     *restart_team.TYPES)
        # the helpers check their own arguments
        calls = restart_calls(restart_scene(CORNER_SCENE))
        expect_error(ValueError, lambda: restart_team.calls_from_alive(run.inputs, np.zeros(10), **RESTART),
                     "restart_team", "alive")
        for bad in ({7: 0.9}, {1: 0.5}, {1: True}):
            expect_error(ValueError, lambda: restart_team.votes(calls, RESTART_FRAMES, bad), "restart_team",
                         "confidence")
        expect_error(ValueError, lambda: restart_team.stoppages(restart_alive(), 0, 200, 0), "restart_team",
                     "min_stoppage")
        # numpy numbers from a sweep are fine, and so is a confidence of exactly 1
        source(run, model=path, tail=np.int64(30), setup=np.int64(40), min_stoppage=np.int64(25), use_ball=np.True_,
               kick_off=1, corner=np.float64(0.8), free_kick=0.7)


def test_restart_team_reads_read_only_inputs():
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        run = restart_model_scene(CORNER_SCENE)
        atd = run.inputs.atd
        for array in (atd.players_xy, atd.ball_xy, atd.slot_team):
            array.flags.writeable = False
        before = (atd.players_xy.copy(), atd.ball_xy.copy())
        evidence = restart_compute(run, path, corner=0.9)
        assert np.array_equal(atd.players_xy, before[0], equal_nan=True)
        assert np.array_equal(atd.ball_xy, before[1], equal_nan=True)
        expected = np.full(RESTART_FRAMES, np.nan)
        expected[150:180] = 0.9
        assert evidence.home.dtype == np.float64 and np.array_equal(evidence.home, expected, equal_nan=True)


def test_restart_team_is_deterministic():
    from pipeline.contracts import RunContext
    from pipeline.evidence import restart_team
    rng = np.random.default_rng(17)
    players = 5.0 + rng.random((RESTART_FRAMES, 22, 2)) * (np.array(PLAY_PITCH) - 10.0)
    players += np.cumsum(rng.normal(0.0, 0.05, players.shape), axis=0)       # a random walk from random points
    players[rng.random(players.shape[:2]) < 0.1] = np.nan
    players[:, 0] = (0.5, 0.5)                                                 # and a corner taker
    ball = np.tile([90.0, 10.0], (RESTART_FRAMES, 1))
    ball[30:60] = ball[110:140] = np.nan
    parameters = dict(tail=40, use_ball=True, kick_off=0.95, corner=0.9, goal_kick=0.8, throw_in=0.7, free_kick=0.6)
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        runs = [restart_scene(players.copy(), ball=ball.copy()) for _ in range(2)]   # fresh Inputs: no shared memo
        first, second = (restart_compute(run, path, **parameters) for run in runs)
        assert np.array_equal(first.home, second.home, equal_nan=True)
        assert not np.isnan(first.home).all(), "the random scene no longer gives a vote"
        settings = {key: parameters[key] for key in ("tail", "use_ball")}
        calls = [restart_team.calls(RunContext(inputs=run.inputs, seed=0), model=path, **settings) for run in runs]
        for key in ("restart", "end", "type", "team"):
            assert np.array_equal(calls[0][key], calls[1][key]), key


RESTART_TRIAL_CONFIG = """
[[evidence]]
kind = "dead_ball_model"
model = {model}

[[evidence]]
kind = "restart_team"
model = {model}
tail = 50
kick_off = 0.95
corner = 0.9
goal_kick = 0.8
throw_in = 0.7

[decoder]
kind = "hmm"
switch_team = 1
switch_dead = 20
min_possession = 1
min_dead = 1
"""


def test_restart_team_in_registry_and_config():
    from pipeline import output
    from pipeline.config import ConfigError, load_config
    from pipeline.contracts import RunContext, X
    from pipeline.evidence import REGISTRY, restart_team
    from pipeline.run import run
    assert REGISTRY["restart_team"] is restart_team.compute
    assert list(REGISTRY) == sorted(REGISTRY), "one line per source, in alphabetical order"
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        model = write_model(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        inputs.kickoffs = {1: 100, 2: 420}
        text = RESTART_TRIAL_CONFIG.format(model=json_string(model))
        config = load_config(write_text(root, "trial.toml", text))
        assert [source.kind for source in config.evidence] == ["dead_ball_model", "restart_team"]
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all() and (result.codes[~outside] != X).all()
        source = result.evidence["restart_team"]
        assert np.isnan(source.alive).all() and np.isnan(source.home[outside]).all()
        calls = restart_team.calls(RunContext(inputs=inputs, seed=0), model=model, tail=50)
        assert len(calls["restart"]) >= 1, "the synthetic match no longer gives a call"
        expected = restart_team.votes(calls, SYNTHETIC_FRAMES, {0: 0.95, 1: 0.9, 2: 0.8, 3: 0.7})
        assert np.array_equal(source.home, expected, equal_nan=True)
        # a misspelt type is refused when the config loads, before any data does
        typo = write_text(root, "typo.toml", text.replace("throw_in = 0.7", "throwin = 0.7"))
        expect_error(ConfigError, lambda: load_config(typo), "throwin", "restart_team")


def stoppage_share(confidence, frames):
    """sigmoid(logit(c) / n), written out from P17's Phase 2b ruling: the per-frame `home` of one observation at c
    spread over a stoppage of n frames."""
    logit = np.log(confidence) - np.log1p(-confidence)
    return 1.0 / (1.0 + np.exp(-logit / frames))


def test_restart_team_stoppage_end_span():
    """span = "stoppage_end" (P17, Phase 2b): one observation per called stoppage at the type's confidence c, spread
    evenly over the stoppage that ends at the restart, [first, r): sigmoid(logit(c) / n) on each of its n frames (1 -
    that for an away taker), so that their log-odds add up to logit(c). tail is refused there, and required with
    span = "after" (the default)."""
    from pipeline.contracts import check_evidence
    from pipeline.evidence import REGISTRY, restart_team
    assert restart_team.SPANS == ("after", "stoppage_end")
    logit = float(np.log(0.9 / 0.1))
    scene = restart_scene(CORNER_SCENE)
    calls = restart_calls(scene, alive=restart_alive((50, 150), (165, 195)), tail=30)
    assert calls["first"].tolist() == [50, 165] and calls["restart"].tolist() == [150, 195]
    assert calls["first"].dtype == np.int64 and not calls["first"].flags.writeable
    got = restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}, span="stoppage_end")
    expected = np.full(RESTART_FRAMES, np.nan)
    expected[50:150], expected[165:195] = stoppage_share(0.9, 100), stoppage_share(0.9, 30)
    assert got.dtype == np.float64 and np.array_equal(np.isnan(got), np.isnan(expected)), np.flatnonzero(~np.isnan(got))
    assert np.allclose(got, expected, rtol=0.0, atol=1e-15, equal_nan=True)
    for first, restart in ((50, 150), (165, 195)):
        values = got[first:restart]
        assert (values == values[0]).all() and 0.5 < values[0] < 0.9, "evenly, and each frame weaker than c"
        total = float(np.log(values / (1.0 - values)).sum())
        assert abs(total - logit) < 1e-12, "the stoppage's log-odds add up to {!r}, not logit(c)".format(total)
    assert np.array_equal(restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}),
                          restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}, span="after"), equal_nan=True)
    # an away taker: the complement, so the log-odds add up to -logit(c)
    away = restart_team.votes(restart_calls(restart_scene(CORNER_SCENE, teams=[1, 0])), RESTART_FRAMES, {1: 0.9},
                              span="stoppage_end")
    assert np.isnan(away[:50]).all() and np.isnan(away[150:]).all()
    assert np.allclose(away[50:150], 1.0 - stoppage_share(0.9, 100), rtol=0.0, atol=1e-15)
    assert abs(float(np.log(away[50:150] / (1.0 - away[50:150])).sum()) + logit) < 1e-12
    # spread itself: one frame is the confidence, and a certain call stays certain on every frame
    assert abs(restart_team.spread(0.9, 1) - 0.9) < 1e-15 and restart_team.spread(1.0, 100) == 1.0
    # a certain call would be n observations of ~13.8 nats, not one: refused here, by type, and still fine for "after"
    try:
        restart_team.votes(calls, RESTART_FRAMES, {1: 1.0}, span="stoppage_end")
    except ValueError as error:
        assert restart_team.TYPES[1] in str(error) and "1.0" in str(error), str(error)
    else:
        raise AssertionError("a confidence of 1.0 must be refused for span stoppage_end")
    assert (restart_team.votes(calls, RESTART_FRAMES, {1: 1.0}, span="after") == 1.0).any()
    assert np.isnan(restart_team.votes(calls, RESTART_FRAMES, {3: 0.9}, span="stoppage_end")).all(), \
        "a type without a confidence is silent with either span"
    for bad in ("before", "", None, 1, True, "Stoppage_End"):
        expect_error(ValueError, lambda: restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}, span=bad),
                     "restart_team", "span")
    # hand-made calls: without their stoppages' first frames, or with an empty stoppage, there is nothing to spread over
    bare = {key: calls[key] for key in ("restart", "end", "type", "team")}
    expect_error(ValueError, lambda: restart_team.votes(bare, RESTART_FRAMES, {1: 0.9}, span="stoppage_end"),
                 "restart_team", "stoppage_end", "first")
    assert np.array_equal(restart_team.votes(bare, RESTART_FRAMES, {1: 0.9}),
                          restart_team.votes(calls, RESTART_FRAMES, {1: 0.9}), equal_nan=True)
    empty = dict(bare, first=np.array([50, 195]))
    expect_error(ValueError, lambda: restart_team.votes(empty, RESTART_FRAMES, {1: 0.9}, span="stoppage_end"),
                 "restart_team", "stoppage_end", "[195, 195)")
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        run = restart_model_scene(CORNER_SCENE)
        source = REGISTRY["restart_team"]
        evidence = source(run, model=path, span="stoppage_end", corner=0.8)
        check_evidence(evidence, RESTART_FRAMES)
        assert np.isnan(evidence.home[:50]).all() and np.isnan(evidence.home[150:]).all()
        assert np.allclose(evidence.home[50:150], stoppage_share(0.8, 100), rtol=0.0, atol=1e-15)
        assert np.isnan(evidence.alive).all()
        after = source(run, model=path, tail=30, corner=0.8)
        assert (after.home[150:180] == 0.8).all() and np.isnan(np.delete(after.home, range(150, 180))).all()
        assert np.array_equal(source(run, model=path, span="after", tail=30, corner=0.8).home, after.home,
                              equal_nan=True)
        # a tail with "stoppage_end" is refused, whatever its value; none with "after" is refused too
        for tail in (30, 1, 0, -1, 1.5):
            expect_error(ValueError, lambda: source(run, model=path, span="stoppage_end", tail=tail, corner=0.8),
                         "restart_team", "tail", "stoppage_end")
        for given in ({}, {"span": "after"}):
            expect_error(ValueError, lambda: source(run, model=path, corner=0.8, **given), "restart_team", "tail",
                         "after")
        for bad in ("before", "", None, 1, True):
            expect_error(ValueError, lambda: source(run, model=path, span=bad, corner=0.8), "restart_team", "span")
        # the other checks still apply with "stoppage_end"
        expect_error(ValueError, lambda: source(run, model=path, span="stoppage_end"), "restart_team", "confidence")
        expect_error(ValueError, lambda: source(run, model=path, span="stoppage_end", free_kick=0.8), "restart_team",
                     "free_kick", "use_ball")


def test_restart_team_stoppage_vote_survives_a_shorter_dead_run():
    """The source's stoppage is [50, 150) and its corner call is home's; the decoder's own `alive` evidence makes D
    shorter, [60, 140). Before the stoppage away has the ball, after it 46 frames of weak hold for away (0.45) come
    before a home episode, so the decoder alone carries away across the stoppage (by 1.23 nats). Spread, the vote gives
    the D run 80 of its 100 frames' log-odds (1.76 of logit(0.9) = 2.20 nats), and the first team after it is home's.
    Written on the single frame r - 1 = 149, an H/A frame for this decoder, the same vote would be lost."""
    from pipeline.contracts import no_opinion
    from pipeline.evidence import REGISTRY
    home = home_runs((1.0 - VOTE, 60), (np.nan, 80), (0.45, 46), (VOTE, 14))
    alive = np.concatenate([np.full(60, 0.9), np.full(80, 0.05), np.full(60, 0.9)])
    costs = {"switch_team": 8.0, "switch_dead": 16.0, "min_possession": 25, "min_dead": 25}
    carried, restarted = "A" * 60 + "D" * 80 + "A" * 46 + "H" * 14, "A" * 60 + "D" * 80 + "H" * 60
    with tempfile.TemporaryDirectory() as root:
        source = REGISTRY["restart_team"](restart_model_scene(CORNER_SCENE), model=write_model(root),
                                          span="stoppage_end", corner=0.9)
    assert np.flatnonzero(~np.isnan(source.home)).tolist() == list(range(50, 150)), "the source's stoppage moved"
    assert (source.home[50:150] > 0.5).all(), "the corner is home's"
    assert letters(hmm_labels(hmm_context([home, no_opinion(200)], [alive, None]), restart_source="s1",
                              **costs)) == carried
    for switch_restart in (np.inf, 8.0):
        labels = hmm_labels(hmm_context([home, source.home], [alive, None]), restart_source="s1",
                            switch_restart=switch_restart, **costs)
        assert letters(labels) == restarted, (switch_restart, letters(labels))
    last_frame = no_opinion(200)
    last_frame[149] = 0.9
    assert letters(hmm_labels(hmm_context([home, last_frame], [alive, None]), restart_source="s1",
                              **costs)) == carried


RESTART_STATES_TRIAL_CONFIG = """
[[evidence]]
kind = "dead_ball_model"
model = {model}

[[evidence]]
kind = "restart_team"
model = {model}
span = "stoppage_end"
kick_off = 0.95
corner = 0.9
goal_kick = 0.8
throw_in = 0.7

[decoder]
kind = "hmm"
switch_team = 1
switch_dead = 20
min_possession = 1
min_dead = 1
restart_source = "restart_team"
switch_restart = inf
"""


def test_hmm_restart_states_run_end_to_end():
    """A config with restart_team's "stoppage_end" span and the decoder's restart states loads (no tail, TOML's inf)
    and runs through the pipeline: votes only over the stoppages that end at the restarts, a valid file, X outside
    the blocks."""
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import RunContext, X
    from pipeline.evidence import restart_team
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        model = write_model(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        inputs.kickoffs = {1: 100, 2: 420}
        text = RESTART_STATES_TRIAL_CONFIG.format(model=json_string(model))
        config = load_config(write_text(root, "trial.toml", text))
        params = config.decoder.params
        assert params["restart_source"] == "restart_team" and params["switch_restart"] == np.inf, params
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all() and (result.codes[~outside] != X).all()
        calls = restart_team.calls(RunContext(inputs=inputs, seed=0), model=model, tail=1)
        assert len(calls["restart"]) >= 1, "the synthetic match no longer gives a call"
        expected = restart_team.votes(calls, SYNTHETIC_FRAMES, {0: 0.95, 1: 0.9, 2: 0.8, 3: 0.7}, span="stoppage_end")
        home = result.evidence["restart_team"].home
        assert np.array_equal(home, expected, equal_nan=True)
        spans = [np.arange(first, restart) for first, restart in zip(calls["first"], calls["restart"])]
        assert np.array_equal(np.flatnonzero(~np.isnan(home)), np.concatenate(spans))


# -- decode -----------------------------------------------------------------------------------------------------------

def vote_context(home, start=0, stop=None, weights=None):
    """A BlockContext over [start, stop) of one or more `home` arrays, the way the runner builds one."""
    from pipeline.contracts import Block, BlockContext, Evidence, RunContext, no_opinion
    homes = home if isinstance(home, list) else [home]
    weights = weights or [1.0] * len(homes)
    stop = len(homes[0]) if stop is None else stop
    evidence = {}
    for index, (values, weight) in enumerate(zip(homes, weights)):
        source = Evidence(alive=no_opinion(len(values)), home=values, name="s{}".format(index), weight=weight)
        evidence[source.name] = source.window(start, stop)
    return BlockContext(run=RunContext(inputs=None, seed=0), block=Block(None, start, stop), evidence=evidence)


def hmm_context(home, alive=None, weights=None, start=0, stop=None):
    """A BlockContext over [start, stop) of one or more sources, the way the runner builds one.

    `home` is one array or a list of them, one source each; `alive` is None (every source has no opinion on it), one
    array for a single source, or a list of arrays and Nones beside `home`'s.
    """
    from pipeline.contracts import Block, BlockContext, Evidence, RunContext, no_opinion
    homes = home if isinstance(home, list) else [home]
    alives = alive if isinstance(alive, list) else [alive] * len(homes)
    assert len(alives) == len(homes), "one alive channel (or None) for each source"
    weights = weights or [1.0] * len(homes)
    stop = len(homes[0]) if stop is None else stop
    evidence = {}
    for index, (values, awake, weight) in enumerate(zip(homes, alives, weights)):
        source = Evidence(alive=no_opinion(len(values)) if awake is None else awake, home=values,
                          name="s{}".format(index), weight=weight)
        evidence[source.name] = source.window(start, stop)
    return BlockContext(run=RunContext(inputs=None, seed=0), block=Block(None, start, stop), evidence=evidence)


def baseline_like_codes(n_frames, seed=0):
    """Team codes shaped like the raw baseline's: long runs of one team or of nothing, with single-frame flips."""
    from starter.baseline import AWAY, HOME, NOBODY
    rng = np.random.default_rng(seed)
    pieces, total = [], 0
    while total < n_frames:
        length = int(rng.integers(1, 150))
        pieces.append(np.full(length, rng.choice([NOBODY, HOME, AWAY]), dtype=np.int8))
        total += length
    codes = np.concatenate(pieces)[:n_frames]
    flips = rng.random(n_frames) < 0.1
    codes[flips] = rng.choice(np.array([NOBODY, HOME, AWAY], dtype=np.int8), size=int(flips.sum()))
    # And one long ball gap with no opinion at all, as the real tracking has (median 10.7 s, docs/results.md A01),
    # so that even a wide window meets frames where nobody voted.
    codes[n_frames // 3:n_frames // 3 + 600] = NOBODY
    return codes


def test_majority_vote_matches_baseline_smooth():
    from pipeline.contracts import LETTERS
    from pipeline.decode import REGISTRY
    from starter import baseline
    codes = baseline_like_codes(6000)
    home = np.where(codes == baseline.HOME, 1.0, np.where(codes == baseline.AWAY, 0.0, np.nan))
    ctx = vote_context(home)
    for width in (0, 1, 7, 50, 125):
        expected = baseline.smooth(codes, width)
        got = LETTERS[REGISTRY["majority_vote"](ctx, half_width=width).labels]
        assert set(expected) == {"H", "A", "D"}, "the synthetic codes no longer reach every branch"
        assert np.array_equal(got, expected), "half_width {}: {} frames differ".format(
            width, int((got != expected).sum()))


def test_majority_vote_sees_only_its_block():
    from pipeline.contracts import D, no_opinion
    from pipeline.decode import REGISTRY
    home = no_opinion(200)
    home[:100] = 1.0  # votes everywhere in the first block, right up to its edge; none in the second
    decoded = REGISTRY["majority_vote"](vote_context(home, start=100, stop=200), half_width=50)
    assert len(decoded.labels) == 100 and (decoded.labels == D).all(), "votes leaked across the block edge"


def test_majority_vote_pools_sources_by_weight():
    from pipeline.contracts import A, H
    from pipeline.decode import REGISTRY
    says_home, says_away = np.ones(10), np.zeros(10)
    heavier_home = REGISTRY["majority_vote"](vote_context([says_home, says_away], weights=[3.0, 1.0]), half_width=0)
    heavier_away = REGISTRY["majority_vote"](vote_context([says_home, says_away], weights=[1.0, 3.0]), half_width=0)
    assert (heavier_home.labels == H).all() and (heavier_away.labels == A).all()


def test_majority_vote_rejects_bad_parameters():
    from pipeline.decode import REGISTRY
    ctx = vote_context(np.ones(10))
    expect_error(ValueError, lambda: REGISTRY["majority_vote"](ctx, half_width=-1), "half_width")
    expect_error(ValueError, lambda: REGISTRY["majority_vote"](ctx, half_width=2.5), "half_width")
    expect_error(ValueError, lambda: REGISTRY["majority_vote"](ctx, half_width=True), "half_width")
    REGISTRY["majority_vote"](ctx, half_width=np.int64(5))  # a numpy integer from a notebook sweep is fine
    expect_error(ValueError, lambda: REGISTRY["majority_vote"](ctx, half_width=5, fallback="Q"), "fallback", "Q")


# The hmm decoder (P06). Its model is pinned here from the definition and not from the code: a reference scorer and a
# reference validity rule written out below, and the best path found by trying every path.

VOTE = 0.78  # a source's confidence in the hmm checks: it moves a frame by log(0.78 / 0.22) = 1.266 nats


def home_runs(*runs):
    """A `home` array from consecutive (value, frames) runs; a NaN value is a stretch without an opinion."""
    return np.concatenate([np.full(frames, value) for value, frames in runs])


def hmm_labels(ctx, **parameters):
    """The hmm decoder's labels on ctx, with a team switch of 4 nats and a dead switch of 100 unless told otherwise."""
    from pipeline.decode import REGISTRY
    parameters = dict({"switch_team": 4.0, "switch_dead": 100.0}, **parameters)
    return REGISTRY["hmm"](ctx, **parameters).labels


def letters(labels):
    """The labels as a string of H, A, D, X: what a failed assertion should show."""
    from pipeline.contracts import LETTERS
    return "".join(LETTERS[labels])


def hmm_path_score(emissions, path, switch_team, switch_dead):
    """The model's score of one path: its emissions, less a penalty for each change (D is one side of the change)."""
    from pipeline.contracts import D
    score = sum(emissions[frame, state] for frame, state in enumerate(path))
    for before, after in zip(path, path[1:]):
        if before != after:
            score -= switch_dead if D in (before, after) else switch_team
    return score


def hmm_path_is_valid(path, min_possession, min_dead):
    """Every run of H or A lasts min_possession frames and every run of D min_dead, the first and last run excepted."""
    from pipeline.contracts import D
    runs = []
    for state in path:
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    return all(length >= (min_dead if state == D else min_possession) for state, length in runs[1:-1])


def test_hmm_matches_brute_force():
    import itertools
    from pipeline.decode import hmm
    constrained = 0
    for case in range(300):
        rng = np.random.default_rng([6, case])
        n_frames = int(rng.integers(1, 8))
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 3))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        min_possession, min_dead = (int(value) for value in rng.integers(1, 5, size=2))
        labels = hmm.viterbi(emissions, switch_team, switch_dead, min_possession, min_dead)
        assert labels.dtype == np.int8 and labels.shape == (n_frames,), (case, labels.dtype, labels.shape)
        path = labels.tolist()
        assert hmm_path_is_valid(path, min_possession, min_dead), (case, path)
        scores = {candidate: hmm_path_score(emissions, candidate, switch_team, switch_dead)
                  for candidate in itertools.product(range(3), repeat=n_frames)}
        best = max(score for candidate, score in scores.items()
                   if hmm_path_is_valid(candidate, min_possession, min_dead))
        got = hmm_path_score(emissions, path, switch_team, switch_dead)
        assert abs(got - best) < 1e-9, "case {}: viterbi scores {!r}, the best valid path {!r}".format(case, got, best)
        constrained += max(scores.values()) > best + 1e-9
    # A check in which no minimum ever bound would prove nothing about the minimums.
    assert constrained >= 30, "only {} of 300 cases had a minimum that mattered".format(constrained)


def test_hmm_minimum_one_is_unconstrained():
    import itertools
    from pipeline.decode import hmm
    for case in range(200):
        rng = np.random.default_rng([6, 1, case])
        n_frames = int(rng.integers(1, 8))
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 3))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        path = hmm.viterbi(emissions, switch_team, switch_dead, 1, 1).tolist()
        best = max(hmm_path_score(emissions, candidate, switch_team, switch_dead)
                   for candidate in itertools.product(range(3), repeat=n_frames))
        got = hmm_path_score(emissions, path, switch_team, switch_dead)
        assert abs(got - best) < 1e-9, "case {}: viterbi scores {!r}, the best path {!r}".format(case, got, best)


def test_hmm_missing_channel_adds_nothing():
    from pipeline.contracts import D, Evidence, no_opinion
    from pipeline.decode import hmm
    rng = np.random.default_rng([6, 100])
    home = np.where(rng.random(200) < 0.3, np.nan, rng.random(200))
    alive = np.where(rng.random(200) < 0.5, np.nan, rng.random(200))
    source = Evidence(alive=alive, home=home, name="s", weight=1.0)
    blank = Evidence(alive=no_opinion(200), home=no_opinion(200), name="blank", weight=3.0)
    assert np.array_equal(hmm.log_likelihoods([source], 200, 1e-6), hmm.log_likelihoods([source, blank], 200, 1e-6)), \
        "a source with no opinion changed the emissions"
    home_only = Evidence(alive=no_opinion(200), home=home, name="s", weight=1.0)
    emissions = hmm.log_likelihoods([home_only], 200, 1e-6)
    assert emissions.shape == (200, 3) and (emissions[:, D] == 0.0).all(), "home says nothing about dead"
    assert np.isfinite(emissions).all()
    # Nothing to go on: D, the fallback of majority_vote, whatever the minimums and even with free switches.
    for parameters in ({}, {"min_possession": 5, "min_dead": 5}, {"switch_team": 0.0, "switch_dead": 0.0}):
        labels = hmm_labels(hmm_context(no_opinion(100)), **parameters)
        assert (labels == D).all(), (parameters, letters(labels))


def test_hmm_clips_hard_probabilities():
    """nearest_player reports a hard 0 or 1; its log would be -inf, so the emissions clip at eps and stay finite."""
    from pipeline.contracts import A, D, H, Evidence
    from pipeline.decode import hmm
    eps = 1e-3
    source = Evidence(alive=np.array([1.0, 0.0, np.nan]), home=np.array([1.0, 0.0, 1.0]), name="s", weight=1.0)
    emissions = hmm.log_likelihoods([source], 3, eps)
    assert np.isfinite(emissions).all()
    near, far = np.log(2.0 * (1.0 - eps)), np.log(2.0 * eps)
    expected = np.array([[near + near, far + near, far],       # home and alive both 1
                         [far + far, near + far, near],        # home and alive both 0
                         [near, far, 0.0]])                    # home 1, alive unknown
    assert np.allclose(emissions, expected, atol=1e-12, rtol=0.0), (emissions, expected)
    assert emissions[0, H] > emissions[0, A] > emissions[0, D] and emissions[1, D] > emissions[1, A] > emissions[1, H]


def test_hmm_weight_is_an_exponent():
    from pipeline.contracts import Evidence
    from pipeline.decode import hmm
    rng = np.random.default_rng([6, 101])
    home = np.where(rng.random(300) < 0.3, np.nan, rng.random(300))
    alive = np.where(rng.random(300) < 0.4, np.nan, rng.random(300))
    other = rng.random(300)
    once = hmm.log_likelihoods([Evidence(alive=alive, home=home, name="s", weight=1.0)], 300, 1e-6)
    twice = hmm.log_likelihoods([Evidence(alive=alive, home=home, name="s", weight=2.0)], 300, 1e-6)
    assert np.array_equal(twice, 2.0 * once), "weight 2 is not exactly twice the emissions of weight 1"
    parameters = {"switch_team": 3.0, "switch_dead": 6.0, "min_possession": 4}
    heavy = hmm_labels(hmm_context(home, alive, weights=[2.0]), **parameters)
    given_twice = hmm_labels(hmm_context([home, home], [alive, alive]), **parameters)
    assert np.array_equal(heavy, given_twice), "weight 2 and the source given twice disagree on {} frames".format(
        int((heavy != given_twice).sum()))
    removed = hmm_labels(hmm_context(home, alive), **parameters)
    muted = hmm_labels(hmm_context([home, other], [alive, other], weights=[1.0, 0.0]), **parameters)
    assert np.array_equal(muted, removed), "a source of weight 0 still moved the labels"
    for bad in (np.nan, np.inf, -1.0):
        expect_error(ValueError, lambda: hmm_labels(hmm_context(home, weights=[bad])), "weight", "hmm")


def test_hmm_holds_through_a_gap():
    from pipeline.contracts import H
    home = home_runs((VOTE, 100), (np.nan, 300), (VOTE, 100))
    for switch_team in (1.0, 10.0):
        labels = hmm_labels(hmm_context(home), switch_team=switch_team, switch_dead=100.0)
        assert (labels == H).all(), "switch_team {}: {}".format(switch_team, letters(labels))


def test_hmm_tie_lands_on_the_first_frame():
    from pipeline.contracts import D
    # The 30 frames between the votes cost the same whichever team holds them, so the change could be anywhere in them.
    # The tie goes to the later run: the change of team lands on the first frame without evidence.
    home = home_runs((VOTE, 20), (np.nan, 30), (1.0 - VOTE, 20))
    for min_possession in (1, 10):
        labels = hmm_labels(hmm_context(home), switch_team=1.0, switch_dead=100.0, min_possession=min_possession)
        assert letters(labels) == "H" * 20 + "A" * 50, (min_possession, letters(labels))
    # Free switches and no evidence at all: every path scores 0, and the preference for D decides.
    labels = hmm_labels(hmm_context(home_runs((np.nan, 40))), switch_team=0.0, switch_dead=0.0)
    assert (labels == D).all(), letters(labels)


def test_hmm_tie_in_a_gap_with_alive_evidence():
    # The same tie as above, but with ball_presence's shape of evidence: every frame of the gap has `alive` and no
    # `home`, which adds the same number to H and to A. The running sums of H and A start from different values, so
    # without care fl(sum_H + x) and fl(sum_A + x) drift apart by ulps and the change of team lands wherever the
    # rounding noise peaks, not on the gap's first frame (viterbi subtracts H's column to make the tie exact).
    # The votes are long enough to beat the gap's evidence for D (1.39 nats a frame): no path pays a switch to
    # stay all D, so a shorter vote decodes to D throughout and tests nothing. With stretches of 60 frames, before the
    # fix the change landed on the gap's last frame (30-frame gap) and in the middle of it (50 frames).
    for gap in (30, 50):
        home = home_runs((VOTE, 60), (np.nan, gap), (1.0 - VOTE, 60))
        alive = np.concatenate([np.full(60, 0.585), np.full(gap, 0.2), np.full(60, 0.585)])
        for min_possession in (1, 10):
            labels = hmm_labels(hmm_context(home, alive), switch_team=1.0, switch_dead=100.0,
                                min_possession=min_possession)
            assert letters(labels) == "H" * 60 + "A" * (gap + 60), (gap, min_possession, letters(labels))


def test_hmm_ties_follow_the_preference_order():
    """Equal scores, made exact with whole-number emissions: continue the run, then D, then H, then A."""
    from pipeline.contracts import H
    from pipeline.decode import hmm

    def decode(rows, switch_team=0.5, switch_dead=0.5):
        return letters(hmm.viterbi(np.array(rows, dtype=float), switch_team, switch_dead, 1, 1))

    # Entering H from D or from A, both at the same cost and the same score: D.
    assert decode([[-5, 0, 0]] * 3 + [[1, -5, -5]] * 3) == "DDDHHH"
    # Entering D from H or from A: H. Entering A from D or from H: D.
    assert decode([[0, 0, -5]] * 3 + [[-5, -5, 1]] * 3) == "HHHDDD"
    assert decode([[0, -5, 0]] * 3 + [[-5, 1, -5]] * 3) == "DDDAAA"
    # Ending in D or H, H or A, equally well: D, then H. (H and A tie on a frame voting 0.5 with the ball alive.)
    assert decode([[1, 0, 1]] * 3) == "DDD" and decode([[1, 1, 0]] * 3) == "HHH"
    # Free switches, and the frames between two votes for H cost the same in any state: the run goes on, rather than
    # being handed to D (which is what the same rows say when they end without a vote: the final state is D then).
    rows = [[1, 0, 0]] + [[0, 0, 0]] * 3
    assert decode(rows + [[5, -5, -5]], switch_team=0.0, switch_dead=0.0) == "HHHHH"
    assert decode(rows, switch_team=0.0, switch_dead=0.0) == "HDDD"
    home = np.full(30, 0.5)
    assert (hmm_labels(hmm_context(home, np.full(30, 0.9)), switch_team=2.0, switch_dead=2.0) == H).all(), \
        "a dead heat between the teams goes to the home side"


def test_hmm_blip_against_the_penalty():
    # 5 x 1.266 = 6.3 nats of evidence for a blip does not pay two team switches of 10; 20 x 1.266 = 25.3 does.
    for blip, expected in ((5, "H" * 105), (20, "H" * 50 + "A" * 20 + "H" * 50)):
        home = home_runs((VOTE, 50), (1.0 - VOTE, blip), (VOTE, 50))
        labels = hmm_labels(hmm_context(home), switch_team=10.0, switch_dead=100.0)
        assert letters(labels) == expected, (blip, letters(labels))


def test_hmm_minimum_possession_removes_short_runs():
    from pipeline.contracts import A, H
    home = home_runs((VOTE, 50), (1.0 - VOTE, 20), (VOTE, 50))
    # The blip is worth keeping at a team switch of 1 (25.3 nats against 2), and so is a longer A run, until it has to
    # swallow too much of the H frames around it: 40 frames hold the blip and 20 frames voting H, so 0 nats against 2.
    labels = hmm_labels(hmm_context(home), switch_team=1.0, switch_dead=100.0, min_possession=40)
    assert (labels == H).all(), letters(labels)
    # At 25 frames the run still pays (the blip and 5 frames of H: 19 nats against 2), so it grows to the minimum.
    labels = hmm_labels(hmm_context(home), switch_team=1.0, switch_dead=100.0, min_possession=25)
    away = np.flatnonzero(labels == A)
    assert len(away) == 25 and away[-1] - away[0] == 24 and away[0] <= 50 and away[-1] >= 69, letters(labels)

    interior = 0
    for case in range(50):
        rng = np.random.default_rng([6, 1000 + case])
        home = np.where(rng.random(2000) < 0.3, np.nan, rng.random(2000))
        alive = np.where(rng.random(2000) < 0.5, np.nan, rng.random(2000))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.2, 3.0, size=2))
        min_possession, min_dead = (int(value) for value in rng.integers(1, 61, size=2))
        labels = hmm_labels(hmm_context(home, alive), switch_team=switch_team, switch_dead=switch_dead,
                            min_possession=min_possession, min_dead=min_dead)
        assert hmm_path_is_valid(labels.tolist(), min_possession, min_dead), "case {}".format(case)
        interior += max(0, int((np.diff(labels) != 0).sum()) + 1 - 2)  # runs, less the first and last
    assert interior >= 200, "only {} interior runs in 50 decodes: the minimums were hardly tested".format(interior)


def test_hmm_sees_only_its_block():
    rng = np.random.default_rng([6, 102])
    home, alive = rng.random(300), rng.random(300)
    parameters = {"switch_team": 2.0, "switch_dead": 3.0, "min_possession": 5, "min_dead": 5}
    first = hmm_labels(hmm_context(home, alive, start=100, stop=200), **parameters)
    elsewhere_home, elsewhere_alive = home.copy(), alive.copy()
    elsewhere_home[:100], elsewhere_home[200:] = 1.0 - home[:100], np.nan
    elsewhere_alive[:100], elsewhere_alive[200:] = np.nan, 1.0 - alive[200:]
    second = hmm_labels(hmm_context(elsewhere_home, elsewhere_alive, start=100, stop=200), **parameters)
    assert len(first) == 100 and np.array_equal(first, second), "evidence outside the block moved a label"


def test_hmm_accepts_read_only_evidence():
    rng = np.random.default_rng([6, 103])
    home, alive = rng.random(200), np.where(rng.random(200) < 0.5, np.nan, rng.random(200))
    before = home.copy(), alive.copy()
    home.flags.writeable = alive.flags.writeable = False
    labels = hmm_labels(hmm_context(home, alive), switch_team=2.0, switch_dead=3.0, min_possession=5)
    assert len(labels) == 200
    assert np.array_equal(home, before[0]) and np.array_equal(alive, before[1], equal_nan=True), "evidence was written"


def test_hmm_rejects_bad_parameters():
    from pipeline.decode import REGISTRY
    ctx = hmm_context(np.full(10, VOTE))

    def decode(**parameters):
        return REGISTRY["hmm"](ctx, **dict({"switch_team": 1.0, "switch_dead": 1.0}, **parameters))

    for name in ("switch_team", "switch_dead"):
        for bad in (-1.0, np.nan, np.inf, True):
            expect_error(ValueError, lambda: decode(**{name: bad}), "hmm", name)
    for name in ("min_possession", "min_dead"):
        for bad in (0, 1.5, True):
            expect_error(ValueError, lambda: decode(**{name: bad}), "hmm", name)
    for bad in (0.0, 0.5, np.nan):
        expect_error(ValueError, lambda: decode(eps=bad), "hmm", "eps")
    for bad in (1, 0, "yes", None):
        expect_error(ValueError, lambda: decode(posterior=bad), "hmm", "posterior")
    # viterbi takes emissions directly: it names what is wrong with them, the shape or the values.
    from pipeline.decode import hmm
    expect_error(ValueError, lambda: hmm.viterbi(np.zeros((5, 2)), 1.0, 1.0, 1, 1), "hmm", "shape (5, 2)")
    for bad in (np.nan, np.inf, -np.inf):
        emissions = np.zeros((5, 3))
        emissions[2, 1] = bad
        expect_error(ValueError, lambda: hmm.viterbi(emissions, 1.0, 1.0, 1, 1), "hmm", "finite", "1 of 15")
    # Whole numbers from a notebook sweep, a penalty of 0 and an integer penalty from a config are all fine.
    decode(switch_team=10, switch_dead=np.float64(5.0), min_possession=np.int64(5), min_dead=3, eps=1e-3)
    decode(switch_team=0, switch_dead=0)
    assert decode(posterior=np.True_).posterior is not None  # a flag from a notebook comparison is fine too


def test_hmm_speed():
    """80,000 frames, two sources, minimums of 50: Viterbi must stay cheap enough for a sweep over seven matches.

    The posterior, which no sweep asks for, gets a looser bound: it is O(minimum) a frame, in numpy calls.
    """
    from pipeline.decode import REGISTRY
    rng = np.random.default_rng([6, 104])
    n_frames = 80000
    team = np.repeat(rng.integers(0, 2, n_frames // 100), 100)  # who has the ball, in stretches of 4 s
    homes, alives = [], []
    for noise in (0.25, 0.4):
        votes_home = rng.random(n_frames) < np.where(team == 0, 1.0 - noise, noise)
        home = np.where(votes_home, VOTE, 1.0 - VOTE)
        home[np.repeat(rng.random(n_frames // 200) < 0.4, 200)] = np.nan  # ball gaps of 8 s
        alive = rng.choice([0.2, 0.585], size=n_frames)
        alive[np.repeat(rng.random(n_frames // 200) < 0.5, 200)] = np.nan
        homes.append(home)
        alives.append(alive)
    ctx = hmm_context(homes, alives)
    started = time.perf_counter()
    labels = hmm_labels(ctx, switch_team=4.0, switch_dead=8.0, min_possession=50, min_dead=50)
    seconds = time.perf_counter() - started
    print("      hmm: {:,} frames, two sources, both minimums 50: {:.2f} s".format(n_frames, seconds))
    assert len(labels) == n_frames and seconds < 4.0, "{:.2f} s for {:,} frames".format(seconds, n_frames)
    started = time.perf_counter()
    decoded = REGISTRY["hmm"](ctx, switch_team=4.0, switch_dead=8.0, min_possession=50, min_dead=50, posterior=True)
    seconds = time.perf_counter() - started
    print("      hmm: the same with posterior=True: {:.2f} s".format(seconds))
    assert decoded.posterior.shape == (n_frames, 3) and seconds < 10.0, "{:.2f} s for the posterior".format(seconds)


# The hmm decoder's posterior: P(H), P(A), P(D) per frame, the marginal over the valid paths with weight exp(score).

def hmm_brute_force_marginals(emissions, switch_team, switch_dead, min_possession, min_dead):
    """The definition of the posterior, by trying every path: (n_frames, 3) marginals and the number of valid paths."""
    import itertools
    paths = [path for path in itertools.product(range(3), repeat=len(emissions))
             if hmm_path_is_valid(path, min_possession, min_dead)]
    scores = np.array([hmm_path_score(emissions, path, switch_team, switch_dead) for path in paths])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    marginals = np.zeros((len(emissions), 3))
    for path, weight in zip(paths, weights):
        for frame, state in enumerate(path):
            marginals[frame, state] += weight
    return marginals, len(paths)


def test_hmm_posterior_matches_brute_force():
    from pipeline.decode import hmm
    constrained = shorter_than_a_minimum = 0
    for case in range(200):
        rng = np.random.default_rng([6, 2000 + case])
        n_frames = int(rng.integers(1, 7))
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 3))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        min_possession, min_dead = (int(value) for value in rng.integers(1, 5, size=2))
        expected, valid_paths = hmm_brute_force_marginals(emissions, switch_team, switch_dead, min_possession, min_dead)
        got = hmm.marginals(emissions, switch_team, switch_dead, min_possession, min_dead)
        assert got.dtype == np.float64 and got.shape == (n_frames, 3), (case, got.dtype, got.shape)
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-12), (
            "case {} ({} frames, minimums {} and {}): {!r} against the brute force {!r}".format(
                case, n_frames, min_possession, min_dead, got, expected))
        constrained += valid_paths < 3 ** n_frames
        shorter_than_a_minimum += n_frames < max(min_possession, min_dead)
    # A check where no minimum ever bound, or no block was cut short of one, would prove nothing about the exemptions.
    assert constrained >= 30 and shorter_than_a_minimum >= 20, "{} constrained cases, {} short blocks of 200".format(
        constrained, shorter_than_a_minimum)


def test_hmm_posterior_rows_sum_to_one():
    from pipeline.decode import REGISTRY, hmm
    for case in range(12):
        rng = np.random.default_rng([6, 3000 + case])
        n_frames = int(rng.integers(200, 1500))
        home = np.where(rng.random(n_frames) < 0.3, np.nan, rng.choice([0.0, 0.22, 0.78, 1.0], size=n_frames))
        alive = np.where(rng.random(n_frames) < 0.5, np.nan, rng.random(n_frames))
        weight = float(rng.choice([1.0, 3.0, 25.0]))  # at 25 the emissions span hundreds of nats, not tens
        parameters = {"switch_team": float(rng.uniform(0.5, 30.0)), "switch_dead": float(rng.uniform(0.5, 30.0)),
                      "min_possession": int(rng.integers(1, 60)), "min_dead": int(rng.integers(1, 60))}
        posterior = REGISTRY["hmm"](hmm_context(home, alive, weights=[weight]), posterior=True, **parameters).posterior
        assert posterior.dtype == np.float64 and posterior.shape == (n_frames, 3), (case, posterior.shape)
        assert np.isfinite(posterior).all() and (posterior >= 0.0).all() and (posterior <= 1.0).all(), case
        totals = posterior.sum(axis=1)
        assert np.allclose(totals, 1.0, rtol=0.0, atol=1e-12), "case {}: rows sum to {:.15f} to {:.15f}".format(
            case, totals.min(), totals.max())
    # An empty block has an empty posterior, and a block without evidence is a prior over paths, not a NaN.
    assert hmm.marginals(np.zeros((0, 3)), 1.0, 1.0, 1, 1).shape == (0, 3)
    blank = hmm.marginals(np.zeros((40, 3)), 2.0, 5.0, 10, 10)
    assert np.isfinite(blank).all() and np.allclose(blank.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_hmm_posterior_is_symmetric_in_time():
    """The model has no arrow of time (penalties are symmetric, the first and last run are both exempt), so a block
    read backwards has the reversed posterior. Brute force stops at 6 frames; this holds the long blocks, where the
    forward and the backward pass run for thousands of frames, to the same answer from both ends."""
    from pipeline.decode import hmm
    for case in range(8):
        rng = np.random.default_rng([6, 5000 + case])
        n_frames = int(rng.integers(300, 2000))
        emissions = rng.normal(0.0, float(rng.choice([0.3, 2.0, 30.0])), size=(n_frames, 3))
        parameters = (float(rng.uniform(0.0, 20.0)), float(rng.uniform(0.0, 20.0)),
                      int(rng.integers(1, 150)), int(rng.integers(1, 150)))
        forwards = hmm.marginals(emissions, *parameters)
        backwards = hmm.marginals(emissions[::-1], *parameters)[::-1]
        assert np.allclose(forwards, backwards, rtol=0.0, atol=1e-10), "case {}: {:.1e} apart".format(
            case, np.abs(forwards - backwards).max())


def test_hmm_posterior_leaves_labels_alone():
    from pipeline.decode import REGISTRY
    for case in range(20):
        rng = np.random.default_rng([6, 4000 + case])
        n_frames = int(rng.integers(1, 800))
        home = np.where(rng.random(n_frames) < 0.3, np.nan, rng.random(n_frames))
        alive = np.where(rng.random(n_frames) < 0.5, np.nan, rng.random(n_frames))
        parameters = {"switch_team": float(rng.uniform(0.0, 6.0)), "switch_dead": float(rng.uniform(0.0, 6.0)),
                      "min_possession": int(rng.integers(1, 40)), "min_dead": int(rng.integers(1, 40))}
        ctx = hmm_context(home, alive)
        without = REGISTRY["hmm"](ctx, **parameters).labels
        with_it = REGISTRY["hmm"](ctx, posterior=True, **parameters).labels
        assert with_it.dtype == np.int8 and np.array_equal(without, with_it), "case {}: {} labels moved".format(
            case, int((without != with_it).sum()))


def test_hmm_posterior_off_by_default():
    from pipeline.decode import REGISTRY
    ctx = hmm_context(np.full(30, VOTE))
    assert REGISTRY["hmm"](ctx, switch_team=1.0, switch_dead=1.0).posterior is None
    assert REGISTRY["hmm"](ctx, switch_team=1.0, switch_dead=1.0, posterior=False).posterior is None
    posterior = REGISTRY["hmm"](ctx, switch_team=1.0, switch_dead=1.0, posterior=True).posterior
    assert posterior is not None and posterior.shape == (30, 3)


# The hmm decoder's restart states (P17, Phase 2b). With restart_source, D splits into D_H ("dead, home restarts") and
# D_A, columns 2 and 3 of the four-state path (H, A, D_H, D_A); the labels map both to D. The checks pin the
# three-state path to its old code, the four-state path to a brute force over every path, and the two paths to each
# other where there is no restart evidence.

def hmm_viterbi_before_restart_states(emissions, switch_team, switch_dead, min_possession, min_dead):
    """hmm.viterbi as it was at 58fc542, before the restart states, verbatim but for its argument checks: the pin that
    the three-state path has not moved by one label."""
    import math
    from pipeline.contracts import A, D, H
    PREFERENCE, STATES = (D, H, A), (H, A, D)
    emissions = np.asarray(emissions, dtype=np.float64)
    switch_team, switch_dead, min_possession, min_dead = (float(switch_team), float(switch_dead), int(min_possession),
                                                          int(min_dead))
    emissions = emissions - emissions[:, H:H + 1]
    n_frames = len(emissions)
    labels = np.empty(n_frames, dtype=np.int8)
    if n_frames == 0:
        return labels

    minimum = (min_possession, min_possession, min_dead)  # by state code
    penalty = ((0.0, switch_team, switch_dead),
               (switch_team, 0.0, switch_dead),
               (switch_dead, switch_dead, 0.0))
    predecessors = tuple(tuple(other for other in PREFERENCE if other != state) for state in STATES)
    cumulative = [[0.0] + np.cumsum(emissions[:, state]).tolist() for state in STATES]

    best = [0.0, 0.0, 0.0]
    start = [0, 0, 0]
    entry = [[0.0] * n_frames for _ in STATES]
    origin = [[0] * n_frames for _ in STATES]
    started = [[0] * n_frames for _ in STATES]
    for frame in range(1, n_frames):
        mature = [cumulative[state][frame] + best[state] for state in STATES]  # L[state][frame - 1]
        for state in STATES:
            value, source = -math.inf, D
            for other in predecessors[state]:
                candidate = mature[other] - penalty[other][state]
                if candidate > value:
                    value, source = candidate, other
            entry[state][frame] = value - cumulative[state][frame]
            origin[state][frame] = source
        for state in STATES:
            first = frame - minimum[state] + 1  # the latest start that has reached the minimum by this frame
            if first >= 1 and entry[state][first] > best[state]:
                best[state], start[state] = entry[state][first], first
            started[state][frame] = start[state]

    final_score, final_state, final_start = -math.inf, D, 0
    for state in PREFERENCE:
        value, begun = best[state], start[state]
        for first in range(max(1, n_frames - minimum[state] + 1), n_frames):
            if entry[state][first] > value:
                value, begun = entry[state][first], first
        if cumulative[state][n_frames] + value > final_score:
            final_score, final_state, final_start = cumulative[state][n_frames] + value, state, begun

    stop, state, begun = n_frames, final_state, final_start
    while True:
        labels[begun:stop] = state
        if begun == 0:
            return labels
        stop, state = begun, origin[state][begun]
        begun = started[state][stop - 1]


def hmm_random_scene(rng, n_frames):
    """(homes, alives, weights) of one to three random sources over n_frames: hard and soft votes, exact ties (0.5),
    ball_presence's two values, NaN, and one stretch where no source says who has the ball."""
    homes, alives = [], []
    for _ in range(int(rng.integers(1, 4))):
        home = np.where(rng.random(n_frames) < 0.5, rng.choice([0.0, 0.22, 0.5, 0.78, 1.0], n_frames),
                        rng.random(n_frames))
        home[rng.random(n_frames) < 0.3] = np.nan
        alive = np.where(rng.random(n_frames) < 0.5, rng.choice([0.2, 0.585], n_frames), rng.random(n_frames))
        alive[rng.random(n_frames) < 0.4] = np.nan
        homes.append(home)
        alives.append(alive)
    gap, length = int(rng.integers(0, n_frames)), int(rng.integers(0, 300))
    for home in homes:
        home[gap:gap + length] = np.nan
    return homes, alives, [float(weight) for weight in rng.choice([0.5, 1.0, 2.0], size=len(homes))]


def hmm_random_costs(rng, longest=60):
    """switch_team, switch_dead, min_possession and min_dead for a random decode, free switches among them."""
    return {"switch_team": float(rng.choice([0.0, 1.0, rng.uniform(0.0, 20.0)])),
            "switch_dead": float(rng.choice([0.0, 2.0, rng.uniform(0.0, 20.0)])),
            "min_possession": int(rng.integers(1, longest)), "min_dead": int(rng.integers(1, longest))}


def test_hmm_restart_states_off_is_unchanged():
    from pipeline.decode import REGISTRY, hmm
    for case in range(40):
        rng = np.random.default_rng([6, 6000 + case])
        n_frames = int(rng.integers(1, 1500))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        costs = hmm_random_costs(rng)
        ctx = hmm_context(homes, alives, weights=weights)
        expected = hmm_viterbi_before_restart_states(hmm.log_likelihoods(ctx.evidence.values(), n_frames, hmm.EPS),
                                                     **costs)
        for extra in ({}, {"restart_source": None}, {"restart_source": None, "switch_restart": np.inf}):
            labels = REGISTRY["hmm"](ctx, **costs, **extra).labels
            assert labels.dtype == np.int8 and np.array_equal(labels, expected), "case {} {}: {} labels moved".format(
                case, extra, int((labels != expected).sum()))
    # Whole-number emissions, where exact ties are everywhere and only the tie rule decides.
    for case in range(300):
        rng = np.random.default_rng([6, 6100 + case])
        n_frames = int(rng.integers(1, 60))
        emissions = rng.integers(-2, 3, size=(n_frames, 3)).astype(np.float64)
        costs = (float(rng.choice([0.0, 0.5, 1.0, 2.0])), float(rng.choice([0.0, 0.5, 1.0, 2.0])),
                 int(rng.integers(1, 6)), int(rng.integers(1, 6)))
        labels, expected = hmm.viterbi(emissions, *costs), hmm_viterbi_before_restart_states(emissions, *costs)
        assert labels.dtype == np.int8 and np.array_equal(labels, expected), (case, letters(labels), letters(expected))


def test_hmm_restart_states_without_evidence_match_three_states():
    """A restart source with no `home` anywhere leaves D_H and D_A the same state twice over: every three-state path
    has a four-state twin of the same score (each D run takes the state of the team after it), and the tie order
    D_H, D_A, H, A picks the twin of the three-state choice, whatever switch_restart is."""
    from pipeline.contracts import A, D, H, no_opinion
    from pipeline.decode import REGISTRY, hmm
    from pipeline.decode.hmm import D_A, D_H
    crossings = 0
    for case in range(40):
        rng = np.random.default_rng([6, 6200 + case])
        n_frames = int(rng.integers(1, 1500))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        costs = hmm_random_costs(rng)
        # the named source says nothing, or speaks on `alive` only, which is read as every other source's is
        silent_alive = None if case % 2 == 0 else np.where(rng.random(n_frames) < 0.5, np.nan, rng.random(n_frames))
        ctx = hmm_context(homes + [no_opinion(n_frames)], alives + [silent_alive], weights=weights + [1.0])
        name = "s{}".format(len(homes))
        three = REGISTRY["hmm"](ctx, **costs).labels
        for switch_restart in (np.inf, float(rng.uniform(0.01, 10.0)), 1e-9):
            four = REGISTRY["hmm"](ctx, restart_source=name, switch_restart=switch_restart, **costs).labels
            assert four.dtype == np.int8 and np.array_equal(four, three), "case {}, switch_restart {}: {} differ" \
                .format(case, switch_restart, int((four != three).sum()))
        changes = np.flatnonzero(np.diff(three) != 0) + 1
        crossings += int(sum(three[change - 1] == D and three[change] != D for change in changes))
    assert crossings >= 50, "only {} restarts in 40 decodes: the twins were hardly tested".format(crossings)
    # Whole-number emissions, D's column given to both dead states: exact ties everywhere, and only the tie order
    # D_H, D_A, H, A makes the four states choose the three states' path.
    for case in range(300):
        rng = np.random.default_rng([6, 6250 + case])
        n_frames = int(rng.integers(1, 40))
        emissions = rng.integers(-2, 3, size=(n_frames, 3)).astype(np.float64)
        costs = (float(rng.choice([0.0, 0.5, 1.0, 2.0])), float(rng.choice([0.0, 0.5, 1.0, 2.0])),
                 int(rng.integers(1, 5)), int(rng.integers(1, 5)))
        switch_restart = float(rng.choice([np.inf, 0.5, 1.0, 2.0]))
        path = hmm.restart_viterbi(emissions[:, [0, 1, 2, 2]], *costs, switch_restart)
        three = hmm.viterbi(emissions, *costs)
        assert np.array_equal(hmm.RESTART_LABELS[path], three), (case, path.tolist(), letters(three))
    # The three-state tie cases (test_hmm_ties_follow_the_preference_order), with D_H or D_A where D was.
    for rows, expected in (([[-5, 0, 0]] * 3 + [[1, -5, -5]] * 3, [D_H] * 3 + [H] * 3),   # into H from D_H, not A
                           ([[0, 0, -5]] * 3 + [[-5, -5, 1]] * 3, [H] * 3 + [D_H] * 3),   # into D_x from H; D_H ends
                           ([[0, -5, 0]] * 3 + [[-5, 1, -5]] * 3, [D_A] * 3 + [A] * 3),   # into A from D_A, not H
                           ([[1, 0, 1]] * 3, [D_H] * 3), ([[0, 0, 0]] * 4, [D_H] * 4)):
        path = hmm.restart_viterbi(np.array(rows, dtype=float)[:, [0, 1, 2, 2]], 0.5, 0.5, 1, 1, np.inf).tolist()
        assert path == expected, (rows, path, expected)


def test_hmm_restart_emissions_follow_the_rule():
    """The four-state emissions: H and A are every source's but the named one's `home`, D_H and D_A both D's, plus the
    named source's `home` as w log 2h and w log 2(1 - h); its `alive` is read as any source's."""
    from dataclasses import replace
    from pipeline.contracts import A, D, H
    from pipeline.decode import hmm
    from pipeline.decode.hmm import D_A, D_H
    for case in range(10):
        rng = np.random.default_rng([6, 6500 + case])
        n_frames = int(rng.integers(1, 500))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        named = len(homes) // 2                              # first, middle or last of the config's sources
        ctx = hmm_context(homes, alives, weights=weights)
        name = "s{}".format(named)
        got = hmm.restart_log_likelihoods(ctx.evidence, name, n_frames, hmm.EPS)
        assert got.dtype == np.float64 and got.shape == (n_frames, 4)
        source = ctx.evidence[name]
        silenced = {key: replace(value, home=np.full(n_frames, np.nan)) if key == name else value
                    for key, value in ctx.evidence.items()}
        three = hmm.log_likelihoods(silenced.values(), n_frames, hmm.EPS)
        assert np.array_equal(got[:, H], three[:, H]) and np.array_equal(got[:, A], three[:, A]), case
        h = np.clip(source.home, hmm.EPS, 1.0 - hmm.EPS)
        known = ~np.isnan(source.home)
        for column, vote in ((D_H, np.log(2.0 * h)), (D_A, np.log(2.0 * (1.0 - h)))):
            expected = three[:, D] + np.where(known, source.weight * vote, 0.0)
            assert np.allclose(got[:, column], expected, rtol=0.0, atol=1e-12), (case, column)
            assert np.array_equal(got[~known, column], three[~known, D]), (case, column)
        assert known.any() and (~known).any(), "the scene no longer has frames with and without a vote"
    # The weight is an exponent on the vote as on any source, and the vote is read where it is, alive or dead.
    vote = np.array([np.nan, 0.92, 0.08])
    ctx = hmm_context([vote], [np.array([0.9, 0.05, np.nan])], weights=[2.0])
    got = hmm.restart_log_likelihoods(ctx.evidence, "s0", 3, hmm.EPS)
    expected = np.array([[np.log(1.8) * 2, np.log(1.8) * 2, np.log(0.2) * 2, np.log(0.2) * 2],
                         [np.log(0.1) * 2, np.log(0.1) * 2, (np.log(1.9) + np.log(1.84)) * 2,
                          (np.log(1.9) + np.log(0.16)) * 2],
                         [0.0, 0.0, np.log(0.16) * 2, np.log(1.84) * 2]])
    assert np.allclose(got, expected, rtol=0.0, atol=1e-12), (got, expected)


def test_hmm_restart_vote_picks_the_first_team():
    """A stoppage, then 46 frames of weak hold evidence for home (0.55, the team before it), then an away control
    episode. Without restart states the decoder carries home across the stoppage and changes team at the episode: one
    change of team (8 nats) is cheaper than 46 frames against the hold (9.23). A vote for away at 0.92 on the
    stoppage's last frame (+0.61 nats for D_A, -1.83 for D_H) tips it by 1.21: the first run after the stoppage is
    away's, and the stoppage stays where it was."""
    from pipeline.contracts import no_opinion
    home = home_runs((VOTE, 100), (np.nan, 50), (0.55, 46), (1.0 - VOTE, 100))
    alive = np.concatenate([np.full(100, 0.9), np.full(50, 0.05), np.full(146, 0.9)])
    costs = {"switch_team": 8.0, "switch_dead": 16.0, "min_possession": 25, "min_dead": 25}
    carried, restarted = "H" * 100 + "D" * 50 + "H" * 46 + "A" * 100, "H" * 100 + "D" * 50 + "A" * 146
    vote = no_opinion(len(home))
    ctx = hmm_context([home, vote], [alive, None])
    assert letters(hmm_labels(ctx, **costs)) == carried
    assert letters(hmm_labels(ctx, restart_source="s1", **costs)) == carried, "no vote: as the three states"
    vote[149] = 1.0 - 0.92
    ctx = hmm_context([home, vote], [alive, None])
    for switch_restart in (np.inf, 8.0):
        labels = hmm_labels(ctx, restart_source="s1", switch_restart=switch_restart, **costs)
        assert letters(labels) == restarted, (switch_restart, letters(labels))
    # read as an ordinary `home` source, the vote falls on a dead frame and moves nothing
    assert letters(hmm_labels(ctx, **costs)) == carried
    # a vote for home keeps home's run; the weight is an exponent on the vote as on any source
    vote[149] = 0.92
    assert letters(hmm_labels(hmm_context([home, vote], [alive, None]), restart_source="s1", **costs)) == carried
    vote[149] = 1.0 - 0.92
    muted = hmm_context([home, vote], [alive, None], weights=[1.0, 0.0])
    assert letters(hmm_labels(muted, restart_source="s1", **costs)) == carried


def restart_brute_force(emissions, switch_team, switch_dead, min_possession, min_dead, switch_restart, paths):
    """Every four-state path in `paths` ((P, n) of columns H, A, D_H, D_A) scored by the design's definition, written
    out here and not taken from the module: (scores, valid). A forbidden change scores -inf."""
    inf = np.inf
    cross = switch_dead + switch_restart
    change = np.array([[0.0, switch_team, switch_dead, switch_dead],       # from H
                       [switch_team, 0.0, switch_dead, switch_dead],       # from A
                       [switch_dead, cross, 0.0, inf],                     # from D_H: to H, A, D_H, D_A
                       [cross, switch_dead, inf, 0.0]])                    # from D_A
    n_frames = paths.shape[1]
    scores = emissions[np.arange(n_frames), paths].sum(axis=1)
    if n_frames > 1:
        scores = scores - change[paths[:, :-1], paths[:, 1:]].sum(axis=1)
    return scores, restart_paths_valid(paths, min_possession, min_dead)


def restart_paths_valid(paths, min_possession, min_dead):
    """(P,) bool: every run of H or A lasts min_possession frames and every run of D_H or D_A min_dead, the first and
    last run excepted."""
    minimum = np.array([min_possession, min_possession, min_dead, min_dead])
    valid = np.ones(len(paths), dtype=bool)
    begun = np.zeros(len(paths), dtype=np.int64)
    for frame in range(1, paths.shape[1]):
        changed = paths[:, frame] != paths[:, frame - 1]
        short = frame - begun < minimum[paths[:, frame - 1]]
        valid &= ~(changed & (begun > 0) & short)   # the run [begun, frame) ended here: not the first, nor the last
        begun = np.where(changed, frame, begun)
    return valid


def test_hmm_restart_vote_never_crosses_its_stoppage():
    """restart_viterbi against every path of tiny blocks (n <= 8, 4^n paths): its path is valid, never changes D_H to
    D_A or back, never restarts with the other team when switch_restart is inf, and scores the best of all."""
    import itertools
    from pipeline.decode import hmm
    from pipeline.decode.hmm import D_A, D_H
    from pipeline.contracts import A, H
    assert (H, A, D_H, D_A) == (0, 1, 2, 3) and hmm.RESTART_PREFERENCE == (D_H, D_A, H, A)
    every = {}
    bound = forbidden = 0
    for case in range(300):
        rng = np.random.default_rng([6, 6300 + case])
        n_frames = int(rng.integers(1, 9))
        if n_frames not in every:
            every[n_frames] = np.array(list(itertools.product(range(4), repeat=n_frames)), dtype=np.int64)
        paths = every[n_frames]
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 4))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        switch_restart = np.inf if rng.random() < 0.5 else float(rng.uniform(0.1, 4.0))
        min_possession, min_dead = (int(value) for value in rng.integers(1, 5, size=2))
        costs = (switch_team, switch_dead, min_possession, min_dead, switch_restart)
        path = hmm.restart_viterbi(emissions, *costs)
        assert path.dtype == np.int8 and path.shape == (n_frames,), (case, path.dtype, path.shape)
        pairs = set(zip(path.tolist(), path.tolist()[1:]))
        assert not pairs & {(D_H, D_A), (D_A, D_H)}, (case, path.tolist())
        if switch_restart == np.inf:
            assert not pairs & {(D_H, A), (D_A, H)}, (case, path.tolist())
        assert restart_paths_valid(path[None].astype(np.int64), min_possession, min_dead)[0], (case, path.tolist())
        scores, valid = restart_brute_force(emissions, *costs, paths)
        best = scores[valid].max()
        got = restart_brute_force(emissions, *costs, path[None].astype(np.int64))[0][0]
        assert np.isfinite(got) and abs(got - best) < 1e-9, "case {}: {!r} against the best valid path {!r}".format(
            case, got, best)
        # the same paths where D_H <-> D_A and the cross restarts cost switch_dead like any other change
        bound += restart_unconstrained_scores(emissions, paths, switch_team, switch_dead)[valid].max() > best + 1e-9
        forbidden += switch_restart == np.inf
    # A check in which no forbidden or priced change was ever worth taking would prove nothing about the constraint.
    assert bound >= 30 and forbidden >= 100, "{} cases bound, {} with switch_restart inf".format(bound, forbidden)


def restart_unconstrained_scores(emissions, paths, switch_team, switch_dead):
    """Scores of `paths` with every change allowed, D_H <-> D_A included, each into or out of a dead state at
    switch_dead: the model without the restart constraint, against which the constraint has to bind."""
    change = np.array([[0.0, switch_team, switch_dead, switch_dead],
                       [switch_team, 0.0, switch_dead, switch_dead],
                       [switch_dead, switch_dead, 0.0, switch_dead],
                       [switch_dead, switch_dead, switch_dead, 0.0]])
    n_frames = paths.shape[1]
    scores = emissions[np.arange(n_frames), paths].sum(axis=1)
    if n_frames > 1:
        scores = scores - change[paths[:, :-1], paths[:, 1:]].sum(axis=1)
    return scores


def test_hmm_restart_states_respect_minimums():
    from pipeline.contracts import D, no_opinion
    from pipeline.decode import REGISTRY, hmm
    # The first-team scene with a minimum stoppage of 60 frames: the 50 dead frames grow to 60 (ten more frames of D
    # cost about 21 nats, giving the stoppage up 147), and the run after it is away's.
    home = home_runs((VOTE, 100), (np.nan, 50), (0.55, 46), (1.0 - VOTE, 100))
    alive = np.concatenate([np.full(100, 0.9), np.full(50, 0.05), np.full(146, 0.9)])
    vote = no_opinion(len(home))
    vote[149] = 1.0 - 0.92
    labels = hmm_labels(hmm_context([home, vote], [alive, None]), restart_source="s1", switch_team=8.0,
                        switch_dead=16.0, min_possession=25, min_dead=60)
    dead = np.flatnonzero(labels == D)
    assert len(dead) == 60 and dead[-1] - dead[0] == 59 and dead[0] <= 100 and dead[-1] >= 149, letters(labels)
    assert letters(labels[dead[-1] + 1:]) == "A" * (len(labels) - dead[-1] - 1), letters(labels)
    # Random blocks with restart votes at random frames: the four-state path is valid with each state's minimum, and
    # decode's labels are that path with D_H and D_A written D.
    interior = 0
    for case in range(30):
        rng = np.random.default_rng([6, 6400 + case])
        n_frames = int(rng.integers(200, 2000))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        votes = no_opinion(n_frames)
        at = rng.random(n_frames) < 0.02
        votes[at] = rng.choice([0.08, 0.2, 0.8, 0.92], int(at.sum()))
        ctx = hmm_context(homes + [votes], alives + [None], weights=weights + [float(rng.choice([1.0, 3.0]))])
        name = "s{}".format(len(homes))
        costs = hmm_random_costs(rng)
        switch_restart = np.inf if case % 2 == 0 else float(rng.uniform(0.5, 10.0))
        emissions = hmm.restart_log_likelihoods(ctx.evidence, name, n_frames, hmm.EPS)
        assert emissions.shape == (n_frames, 4) and np.isfinite(emissions).all()
        path = hmm.restart_viterbi(emissions, costs["switch_team"], costs["switch_dead"], costs["min_possession"],
                                   costs["min_dead"], switch_restart)
        assert restart_paths_valid(path[None].astype(np.int64), costs["min_possession"], costs["min_dead"])[0], case
        labels = REGISTRY["hmm"](ctx, restart_source=name, switch_restart=switch_restart, **costs).labels
        assert np.array_equal(labels, hmm.RESTART_LABELS[path]), case
        assert hmm_path_is_valid(labels.tolist(), costs["min_possession"], costs["min_dead"]), case
        interior += max(0, int((np.diff(path) != 0).sum()) + 1 - 2)
    assert interior >= 100, "only {} interior runs in 30 decodes: the minimums were hardly tested".format(interior)


def test_hmm_restart_states_refuse_posterior_and_unknown_source():
    from pipeline.contracts import no_opinion
    from pipeline.decode import REGISTRY, hmm
    ctx = hmm_context([np.full(10, VOTE), no_opinion(10)])

    def decode(**parameters):
        return REGISTRY["hmm"](ctx, **dict({"switch_team": 1.0, "switch_dead": 1.0}, **parameters))

    expect_error(ValueError, lambda: decode(restart_source="s1", posterior=True), "hmm", "posterior", "restart_source")
    expect_error(ValueError, lambda: decode(restart_source="restart_team"), "hmm", "restart_source", "'restart_team'",
                 "s0", "s1")
    for bad in ("", 3, True, ["s1"], b"s1"):
        expect_error(ValueError, lambda: decode(restart_source=bad), "hmm", "restart_source")
    for bad in (0, 0.0, -1.0, np.nan, -np.inf, True, np.True_, "8", None):
        expect_error(ValueError, lambda: decode(restart_source="s1", switch_restart=bad), "hmm", "switch_restart")
    # switch_restart is read only with restart_source: a finite one without it would be ignored, so it is refused
    expect_error(ValueError, lambda: decode(switch_restart=8.0), "hmm", "switch_restart", "restart_source")
    # the three-state checks still apply
    expect_error(ValueError, lambda: decode(restart_source="s1", min_dead=0), "hmm", "min_dead")
    # whole numbers, numpy numbers and inf are fine
    for good in (8, np.float64(8.0), np.int64(3), 1e-6, np.inf, float("inf")):
        assert len(decode(restart_source="s1", switch_restart=good).labels) == 10
    assert decode(switch_restart=np.inf).posterior is None
    # restart_viterbi and restart_log_likelihoods check their own arguments
    expect_error(ValueError, lambda: hmm.restart_viterbi(np.zeros((5, 3)), 1.0, 1.0, 1, 1, np.inf), "hmm",
                 "shape (5, 3)")
    expect_error(ValueError, lambda: hmm.restart_viterbi(np.zeros((5, 4)), 1.0, 1.0, 1, 1, 0.0), "hmm",
                 "switch_restart")
    expect_error(ValueError, lambda: hmm.restart_log_likelihoods(ctx.evidence, "s9", 10, hmm.EPS), "hmm",
                 "restart_source", "'s9'")


# P31: team_neutral_dead. Under the flag each `home` term also gives D the larger of the two team terms, so that team
# evidence ranks H against A and leaves the better of them level with D. Off, the decoder is what it was.

def test_hmm_team_neutral_dead_emissions():
    from pipeline.contracts import Evidence, no_opinion
    from pipeline.decode import hmm
    source = Evidence(alive=no_opinion(3), home=np.array([0.93, 0.07, np.nan]), name="s", weight=2.0)
    off = hmm.log_likelihoods([source], 3, hmm.EPS)
    on = hmm.log_likelihoods([source], 3, hmm.EPS, neutral_dead=True)
    assert np.array_equal(off[:, hmm.H], on[:, hmm.H]) and np.array_equal(off[:, hmm.A], on[:, hmm.A]), (off, on)
    assert (off[:, hmm.D] == 0.0).all(), off[:, hmm.D]
    expected = np.array([2 * np.log(1.86), 2 * np.log(1.86), 0.0])
    assert np.allclose(on[:, hmm.D], expected, atol=1e-12, rtol=0.0), (on[:, hmm.D], expected)
    assert np.array_equal(on[:, hmm.D], np.maximum(on[:, hmm.H], on[:, hmm.A])), on


def test_hmm_team_neutral_dead_leaves_alive_terms():
    from pipeline.contracts import Evidence, no_opinion
    from pipeline.decode import hmm
    rng = np.random.default_rng([6, 5100])
    alive = np.where(rng.random(100) < 0.3, np.nan, rng.random(100))
    source = Evidence(alive=alive, home=no_opinion(100), name="s", weight=1.5)
    assert np.array_equal(hmm.log_likelihoods([source], 100, hmm.EPS),
                          hmm.log_likelihoods([source], 100, hmm.EPS, neutral_dead=True)), "an alive term moved"


def test_hmm_team_neutral_dead_decodes_dead():
    from pipeline.contracts import D, H
    from pipeline.decode import REGISTRY, hmm
    ctx = hmm_context([np.full(100, 0.93), np.full(100, np.nan)], [None, np.full(100, 0.45)])
    parameters = {"switch_team": 6, "switch_dead": 32, "min_possession": 35, "min_dead": 25}
    off = REGISTRY["hmm"](ctx, **parameters).labels
    on = REGISTRY["hmm"](ctx, team_neutral_dead=True, **parameters)
    assert (off == H).all(), letters(off)
    assert (on.labels == D).all(), letters(on.labels)
    with_posterior = REGISTRY["hmm"](ctx, team_neutral_dead=True, posterior=True, **parameters).posterior
    assert (np.argmax(with_posterior, axis=1) == hmm.D).all(), with_posterior.min(axis=0)


def test_hmm_team_neutral_dead_refusals():
    from pipeline.contracts import no_opinion
    from pipeline.decode import REGISTRY
    ctx = hmm_context([np.full(10, VOTE), no_opinion(10)])

    def decode(**parameters):
        return REGISTRY["hmm"](ctx, **dict({"switch_team": 1.0, "switch_dead": 1.0}, **parameters))

    for bad in (1, 0, "yes", None):
        expect_error(ValueError, lambda: decode(team_neutral_dead=bad), "hmm", "team_neutral_dead")
    expect_error(ValueError, lambda: decode(team_neutral_dead=True, restart_source="s1"), "hmm", "team_neutral_dead",
                 "restart_source")
    assert len(decode(team_neutral_dead=np.True_).labels) == 10


# P35: a per-frame team cost from a named hazard source (hazard_source, hazard_scale). The named source's `home` is a
# hazard z; a change of team into frame t costs clip(switch_team - hazard_scale logit z_t, HAZARD_FLOOR, 2 switch_team),
# and switch_team where z_t is NaN. Without a hazard, viterbi runs the loop it always ran (team_cost=None).

def hmm_priced_scores(emissions, paths, switch_dead, team_cost):
    """(P,) scores of the three-state `paths` ((P, n) of codes H, A, D) by the definition, written out here and not
    taken from the module: the emissions along each, less team_cost[t] for a change between H and A from frame t - 1
    to t and switch_dead for a change into or out of D. team_cost[0] is never read: no change enters the first frame."""
    from pipeline.contracts import D
    n_frames = paths.shape[1]
    scores = emissions[np.arange(n_frames), paths].sum(axis=1)
    if n_frames > 1:
        before, after = paths[:, :-1], paths[:, 1:]
        changed = before != after
        dead = changed & ((before == D) | (after == D))
        scores = scores - np.where(changed & ~dead, team_cost[1:], 0.0).sum(axis=1) - dead.sum(axis=1) * switch_dead
    return scores


def hmm_hazard_context(homes, alives, weights, hazard, hazard_alive=None, at=None):
    """(ctx, name): hmm_context of the team sources with a hazard source inserted at position `at` (default last), whose
    `home` is `hazard` and `alive` is hazard_alive (None: no opinion), and the hazard source's name."""
    at = len(homes) if at is None else at
    ctx = hmm_context(homes[:at] + [hazard] + homes[at:], alives[:at] + [hazard_alive] + alives[at:],
                      weights=weights[:at] + [1.0] + weights[at:])
    return ctx, "s{}".format(at)


def test_hmm_hazard_matches_brute_force():
    """viterbi with a per-frame team cost against every path of tiny blocks (n <= 8, 3^n paths, min_possession 1 to 3,
    min_dead 1 to 2, random continuous emissions and costs): its path is valid, scores the best of all valid paths, and
    is the best one (continuous random numbers leave no tie). team_cost[0] is never read."""
    import itertools
    from pipeline.decode import hmm
    every = {}
    priced = bound = 0
    for case in range(400):
        rng = np.random.default_rng([35, case])
        n_frames = int(rng.integers(1, 9))
        if n_frames not in every:
            every[n_frames] = np.array(list(itertools.product(range(3), repeat=n_frames)), dtype=np.int64)
        paths = every[n_frames]
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 3))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        team_cost = rng.uniform(0.0, 6.0, size=n_frames)
        min_possession, min_dead = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        costs = (switch_team, switch_dead, min_possession, min_dead)
        labels = hmm.viterbi(emissions, *costs, team_cost=team_cost)
        assert labels.dtype == np.int8 and labels.shape == (n_frames,), (case, labels.dtype, labels.shape)
        assert restart_paths_valid(labels[None].astype(np.int64), min_possession, min_dead)[0], (case, letters(labels))
        valid = restart_paths_valid(paths, min_possession, min_dead)   # the three codes are its first three columns
        scores = np.where(valid, hmm_priced_scores(emissions, paths, switch_dead, team_cost), -np.inf)
        best = int(np.argmax(scores))
        got = hmm_priced_scores(emissions, labels[None].astype(np.int64), switch_dead, team_cost)[0]
        assert abs(got - scores[best]) < 1e-9, "case {}: viterbi scores {!r}, the best valid path {!r}".format(
            case, got, scores[best])
        assert np.array_equal(labels, paths[best]), (case, letters(labels), letters(paths[best]))
        for first in (0.0, 1e3):                                       # what frame 0's cost is changes nothing
            moved = team_cost.copy()
            moved[0] = first
            assert np.array_equal(hmm.viterbi(emissions, *costs, team_cost=moved), labels), (case, first)
        # The per-frame costs have to matter (the best path under a flat cost of their mean is another one), and so do
        # the minimums (the best path with every run allowed is another one), or the check proves nothing about them.
        flat = np.where(valid, hmm_priced_scores(emissions, paths, switch_dead,
                                                 np.full(n_frames, team_cost[1:].mean() if n_frames > 1 else 0.0)),
                        -np.inf)
        priced += int(np.argmax(flat)) != best
        bound += hmm_priced_scores(emissions, paths, switch_dead, team_cost).max() > scores[best] + 1e-9
    assert priced >= 30 and bound >= 30, "{} cases where the per-frame cost decided, {} where a minimum bound".format(
        priced, bound)


def test_hmm_hazard_flat_cost_is_today():
    """team_cost filled with switch_team gives the labels of team_cost=None, to the label: on random scenes, on
    whole-number emissions where exact ties are everywhere, and on the gap scenes whose change of team only the tie rule
    places (on the gap's first frame)."""
    from pipeline.decode import hmm
    for case in range(40):
        rng = np.random.default_rng([35, 1000 + case])
        n_frames = int(rng.integers(1, 1500))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        costs = hmm_random_costs(rng)
        emissions = hmm.log_likelihoods(hmm_context(homes, alives, weights=weights).evidence.values(), n_frames,
                                        hmm.EPS)
        today = hmm.viterbi(emissions, **costs)
        flat = hmm.viterbi(emissions, **costs, team_cost=np.full(n_frames, costs["switch_team"]))
        assert flat.dtype == np.int8 and np.array_equal(flat, today), "case {}: {} labels moved".format(
            case, int((flat != today).sum()))
        assert np.array_equal(today, hmm_viterbi_before_restart_states(emissions, **costs)), case
    for case in range(300):
        rng = np.random.default_rng([35, 1100 + case])
        n_frames = int(rng.integers(1, 60))
        emissions = rng.integers(-2, 3, size=(n_frames, 3)).astype(np.float64)
        costs = (float(rng.choice([0.0, 0.5, 1.0, 2.0])), float(rng.choice([0.0, 0.5, 1.0, 2.0])),
                 int(rng.integers(1, 6)), int(rng.integers(1, 6)))
        today = hmm.viterbi(emissions, *costs)
        flat = hmm.viterbi(emissions, *costs, team_cost=np.full(n_frames, costs[0]))
        assert np.array_equal(flat, today), (case, letters(flat), letters(today))
    for gap in (30, 50):
        home = home_runs((VOTE, 60), (np.nan, gap), (1.0 - VOTE, 60))
        alive = np.concatenate([np.full(60, 0.585), np.full(gap, 0.2), np.full(60, 0.585)])
        emissions = hmm.log_likelihoods(hmm_context(home, alive).evidence.values(), len(home), hmm.EPS)
        for min_possession in (1, 10):
            costs = (1.0, 100.0, min_possession, 1)
            flat = hmm.viterbi(emissions, *costs, team_cost=np.full(len(home), 1.0))
            assert letters(flat) == "H" * 60 + "A" * (gap + 60), (gap, min_possession, letters(flat))
            assert np.array_equal(flat, hmm.viterbi(emissions, *costs)), (gap, min_possession)


def test_hmm_hazard_costs_follow_the_rule():
    """hazard_costs: switch_team - scale logit(z) clipped to [HAZARD_FLOOR, 2 switch_team], z clipped to [eps, 1 - eps]
    first, and switch_team exactly where z is NaN, at z = 0.5 and at scale 0."""
    from pipeline.decode import hmm
    assert hmm.HAZARD_FLOOR == 1.0
    z = np.array([np.nan, 0.5, 0.9, 0.1, 0.999, 1e-4, 1.0, 0.0])
    got = hmm.hazard_costs(z, 6.0, 1.0, 1e-6)
    logit = np.log(0.9 / 0.1)
    expected = np.array([6.0, 6.0, 6.0 - logit, 6.0 + logit, 1.0, 12.0, 1.0, 12.0])
    assert got.dtype == np.float64 and got.shape == (8,) and np.allclose(got, expected, rtol=0.0, atol=1e-12), got
    assert got[0] == 6.0 and got[1] == 6.0, "NaN and z = 0.5 must cost exactly switch_team"
    half = hmm.hazard_costs(z, 6.0, 0.5, 1e-6)
    assert np.allclose(half[2:4], [6.0 - logit / 2, 6.0 + logit / 2], rtol=0.0, atol=1e-12), half
    assert (hmm.hazard_costs(z, 6.0, 0.0, 1e-6) == 6.0).all(), "scale 0 is switch_team on every frame"
    assert (hmm.hazard_costs(z, 1.0, 0.0, 1e-6) == 1.0).all(), "and at switch_team = HAZARD_FLOOR too"
    # eps is the decoder's clip: at eps 1e-2 the hard 1 is logit(0.99) = 4.6 nats, inside the clip at switch_team 8
    assert abs(hmm.hazard_costs(np.array([1.0]), 8.0, 1.0, 1e-2)[0] - (8.0 - np.log(99.0))) < 1e-12
    before = z.copy()
    z.flags.writeable = False
    hmm.hazard_costs(z, 6.0, 1.0, 1e-6)
    assert np.array_equal(z, before, equal_nan=True), "the hazard was written"


def test_hmm_hazard_silent_or_scale_zero_is_today():
    """A hazard source NaN everywhere (at any scale), and any hazard at hazard_scale 0, give the labels of the same
    block without the source: the named source adds nothing to the emissions, neither its `home` nor its `alive`, and
    a cost of switch_team on every frame is today's decoder. Each case has two scenes: hmm_random_scene's noise at
    random costs, and team runs that noisy votes make out, with ball gaps, where the decoder changes team from H to A
    directly (the changes the hazard prices) and the same hazards at scale 1 move labels."""
    from pipeline.contracts import D, no_opinion
    from pipeline.decode import REGISTRY
    moved = direct = 0
    for case in range(30):
        rng = np.random.default_rng([35, 2000 + case])
        n_frames = int(rng.integers(1, 1500))
        homes, alives, weights = hmm_random_scene(rng, n_frames)
        costs = dict(hmm_random_costs(rng), switch_team=float(rng.choice([1.0, 6.0, rng.uniform(1.0, 20.0)])))
        team = np.repeat(rng.integers(0, 2, n_frames // 40 + 1), 40)[:n_frames]
        seen = ~np.repeat(rng.random(n_frames // 60 + 1) < 0.3, 60)[:n_frames]
        votes = np.where(rng.random(n_frames) < np.where(team == 0, 0.75, 0.25), VOTE, 1.0 - VOTE)
        votes[~seen] = np.nan
        runs = {"switch_team": float(rng.choice([1.0, 6.0])), "switch_dead": 50.0,
                "min_possession": int(rng.integers(1, 20))}
        hazard_alive = np.where(rng.random(n_frames) < 0.5, np.nan, rng.random(n_frames))
        at = int(rng.integers(0, len(homes) + 1))
        hazard = np.where(rng.random(n_frames) < 0.4, np.nan, rng.beta(0.5, 0.5, n_frames))
        scenes = (([homes, alives, weights], costs, at),
                  ([[votes], [np.where(seen, 0.585, 0.2)], [1.0]], runs, int(rng.integers(0, 2))))
        for scene, (sources, parameters, position) in enumerate(scenes):
            without = REGISTRY["hmm"](hmm_context(sources[0], sources[1], weights=sources[2]), **parameters).labels
            ctx, name = hmm_hazard_context(*sources, no_opinion(n_frames), hazard_alive, position)
            for scale in ({}, {"hazard_scale": 2.0}, {"hazard_scale": 0.0}):
                labels = REGISTRY["hmm"](ctx, hazard_source=name, **scale, **parameters).labels
                assert labels.dtype == np.int8 and np.array_equal(labels, without), "case {} scene {} {}: {} labels " \
                    "moved".format(case, scene, scale, int((labels != without).sum()))
            ctx, name = hmm_hazard_context(*sources, hazard, hazard_alive, position)
            for zero in (0.0, 0, np.float64(0.0)):
                labels = REGISTRY["hmm"](ctx, hazard_source=name, hazard_scale=zero, **parameters).labels
                assert np.array_equal(labels, without), "case {} scene {}: {} labels moved at scale 0".format(
                    case, scene, int((labels != without).sum()))
            if scene == 1:
                direct += int(((without[1:] != without[:-1]) & (without[1:] != D) & (without[:-1] != D)).sum())
                moved += not np.array_equal(REGISTRY["hmm"](ctx, hazard_source=name, **parameters).labels, without)
    # Equal labels where the decoder never changed team directly would say nothing about the per-frame cost.
    assert direct >= 100 and moved >= 10, "{} direct changes of team in 30 decodes, the hazard moved labels in {} of " \
        "30 at scale 1".format(direct, moved)


def test_hmm_hazard_peak_places_the_change():
    """A stretch with no team evidence between two teams' runs: the change of team lands on the stretch's first frame
    (the tie rule); a hazard peak at frame k puts it on k, a dip at the first frame moves it off, and a hazard that
    names no frame moves nothing. Where the team evidence decides, a peak does not make a change."""
    from pipeline.contracts import no_opinion
    from pipeline.decode import REGISTRY
    home = home_runs((VOTE, 60), (np.nan, 30), (1.0 - VOTE, 60))
    alive = np.concatenate([np.full(60, 0.585), np.full(30, 0.2), np.full(60, 0.585)])   # ball_presence's shape
    costs = {"switch_team": 6.0, "switch_dead": 100.0, "min_possession": 10}

    def labels(hazard, **extra):
        ctx, name = hmm_hazard_context([home], [alive], [1.0], hazard)
        return letters(REGISTRY["hmm"](ctx, hazard_source=name, **extra, **costs).labels)

    assert letters(REGISTRY["hmm"](hmm_context(home, alive), **costs).labels) == "H" * 60 + "A" * 90
    assert labels(no_opinion(150)) == "H" * 60 + "A" * 90
    for k in (60, 61, 75, 89):
        hazard = no_opinion(150)
        hazard[k] = 0.9
        for scale in (0.5, 1.0, 2.0):
            assert labels(hazard, hazard_scale=scale) == "H" * k + "A" * (150 - k), (k, scale)
        assert labels(hazard, hazard_scale=0.0) == "H" * 60 + "A" * 90, k
    dip = no_opinion(150)
    dip[60] = 0.1
    assert labels(dip) == "H" * 61 + "A" * 89, "a dip at the first frame makes the next one the cheapest"
    # A peak inside H's votes: the 10 frames a change there would give A cost far more than the peak takes off.
    peak = np.full(150, 0.5)
    peak[30] = 1.0
    assert labels(peak) == "H" * 60 + "A" * 90


def test_hmm_hazard_speed():
    """80,000 frames, two team sources and a hazard where the ball is seen, minimums of 50: the decode with the hazard
    takes at most 1.5 times today's on the same frames without it (the fastest of three runs each, interleaved: the
    minimum is the least disturbed by a loaded machine)."""
    from pipeline.decode import REGISTRY
    rng = np.random.default_rng([35, 3000])
    n_frames = 80000
    team = np.repeat(rng.integers(0, 2, n_frames // 100), 100)
    seen = ~np.repeat(rng.random(n_frames // 200) < 0.4, 200)
    homes, alives = [], []
    for noise in (0.25, 0.4):
        home = np.where(rng.random(n_frames) < np.where(team == 0, 1.0 - noise, noise), VOTE, 1.0 - VOTE)
        home[~seen] = np.nan
        homes.append(home)
        alives.append(np.where(seen, 0.585, 0.2))
    hazard = np.where(seen, rng.beta(0.5, 4.0, n_frames), np.nan)
    without = hmm_context(homes, alives)
    ctx, name = hmm_hazard_context(homes, alives, [1.0, 1.0], hazard)
    costs = {"switch_team": 6.0, "switch_dead": 8.0, "min_possession": 50, "min_dead": 50}
    today, priced = [], []
    for _ in range(3):
        started = time.perf_counter()
        expected = REGISTRY["hmm"](without, **costs).labels
        today.append(time.perf_counter() - started)
        started = time.perf_counter()
        labels = REGISTRY["hmm"](ctx, hazard_source=name, **costs).labels
        priced.append(time.perf_counter() - started)
    ratio = min(priced) / min(today)
    print("      hmm: {:,} frames, with a hazard {:.2f} s, without {:.2f} s (x{:.2f})".format(
        n_frames, min(priced), min(today), ratio))
    assert len(labels) == len(expected) == n_frames and not np.array_equal(labels, expected), "the hazard did nothing"
    assert ratio <= 1.5, "the hazard's decode takes {:.2f} times today's".format(ratio)


def test_hmm_hazard_refusals():
    """What decode refuses with a hazard, each a ValueError naming the decoder, the parameter, what is allowed and the
    value given; and what viterbi refuses of a team_cost."""
    from pipeline.decode import REGISTRY, hmm
    ctx = hmm_context([np.full(10, VOTE), np.full(10, 0.3)])

    def decode(context=ctx, **parameters):
        return REGISTRY["hmm"](context, **dict({"switch_team": 6.0, "switch_dead": 8.0}, **parameters))

    # hazard_scale without a source would be read by nothing
    expect_error(ValueError, lambda: decode(hazard_scale=1.0), "hmm", "hazard_scale", "hazard_source")
    for bad in (-0.5, -np.inf, np.nan, np.inf, True, np.True_, "1"):
        expect_error(ValueError, lambda: decode(hazard_source="s1", hazard_scale=bad), "hmm", "hazard_scale",
                     repr(bad))
    for bad in ("", 3, True, ["s1"], b"s1"):
        expect_error(ValueError, lambda: decode(hazard_source=bad), "hmm", "hazard_source", repr(bad))
    expect_error(ValueError, lambda: decode(hazard_source="s9"), "hmm", "hazard_source", "'s9'", "'s0'", "'s1'")
    expect_error(ValueError, lambda: decode(hazard_source="s1", restart_source="s0"), "hmm", "hazard_source",
                 "restart_source")
    expect_error(ValueError, lambda: decode(hazard_source="s1", team_neutral_dead=True), "hmm", "hazard_source",
                 "team_neutral_dead")
    # below HAZARD_FLOOR a frame at the base rate would cost more than switch_team, and scale 0 would not be today's
    for low in (0.0, 0.5, 0.999):
        expect_error(ValueError, lambda: decode(hazard_source="s1", switch_team=low), "hmm", "hazard_source",
                     "switch_team", "HAZARD_FLOOR", repr(low))
    # the hazard's strength is hazard_scale: a weight on its source would be read by nothing
    weighted = hmm_context([np.full(10, VOTE), np.full(10, 0.3)], weights=[1.0, 0.5])
    expect_error(ValueError, lambda: decode(weighted, hazard_source="s1"), "hmm", "'s1'", "weight", "hazard_scale",
                 "0.5")
    # the three-state checks still apply
    expect_error(ValueError, lambda: decode(hazard_source="s1", min_dead=0), "hmm", "min_dead")
    for good in ({}, {"hazard_scale": None}, {"hazard_scale": 0}, {"hazard_scale": 2}, {"hazard_scale": 0.5},
                 {"hazard_scale": np.float64(1.5)}, {"hazard_scale": np.int64(1)}, {"switch_team": 1.0}):
        assert len(decode(hazard_source="s1", **good).labels) == 10, good
    assert decode(hazard_source="s1", posterior=False).posterior is None
    assert decode(hazard_source="s1", posterior=True).posterior.shape == (10, 3), "the posterior takes the hazard"
    # viterbi's and marginals' team_cost: one finite number of nats >= 0 per frame
    emissions = np.zeros((5, 3))
    for function in (hmm.viterbi, hmm.marginals):
        for bad in (np.zeros(4), np.zeros(6), np.zeros((5, 1)), np.array([1.0, np.nan, 1.0, 1.0, 1.0]),
                    np.array([1.0, 1.0, np.inf, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, -0.1, 1.0])):
            expect_error(ValueError, lambda: function(emissions, 1.0, 1.0, 1, 1, team_cost=bad), "hmm", "team_cost")
        assert len(function(emissions, 1.0, 1.0, 1, 1, team_cost=[0, 1, 2, 3, 4])) == 5     # a list of whole numbers
        assert len(function(np.zeros((0, 3)), 1.0, 1.0, 1, 1, team_cost=np.zeros(0))) == 0


def test_hmm_hazard_config_cross_check():
    """load_config: an hmm decoder's hazard_source must name an evidence source of kind "turnover_hazard", and every
    such source must be named by it (unnamed, it would vote as a team source). Each a ConfigError naming both; the
    real hmm.decode's signature with stand-in sources (the real source is another task's)."""
    from pipeline.config import ConfigError, load_config
    from pipeline.decode import hmm

    def fake_source(run):
        return None

    def fake_hazard(run, *, model="m.json"):
        return None

    registries = {"evidence": {"fake_source": fake_source, "turnover_hazard": fake_hazard},
                  "decode": {"hmm": hmm.decode, "fake_vote": fake_registries()["decode"]["fake_vote"]},
                  "refine": {}}
    team = '[[evidence]]\nkind = "fake_source"\nname = "team"\n\n'
    hazard = '[[evidence]]\nkind = "turnover_hazard"\nname = "{}"\n\n'
    decoder = '[decoder]\nkind = "hmm"\nswitch_team = 6\nswitch_dead = 32\n'

    def load(text):
        with tempfile.TemporaryDirectory() as folder:
            return load_config(write_text(folder, "p35_trial.toml", text), registries=registries)

    config = load(team + hazard.format("hazard") + decoder + 'hazard_source = "hazard"\nhazard_scale = 0.5\n')
    assert [component.kind for component in config.evidence] == ["fake_source", "turnover_hazard"], config.evidence
    assert config.decoder.params == {"switch_team": 6, "switch_dead": 32, "hazard_source": "hazard",
                                     "hazard_scale": 0.5}, config.decoder.params
    assert load(hazard.format("h2") + team + decoder + 'hazard_source = "h2"\n').decoder.params["hazard_source"] == "h2"
    assert "hazard_source" not in load(team + decoder).decoder.params, "no hazard: the config is as it was"
    # hazard_source names a source of another kind, or none at all
    expect_error(ConfigError, lambda: load(team + hazard.format("hazard") + decoder + 'hazard_source = "team"\n'),
                 "p35_trial.toml", "hazard_source", "'team'", "turnover_hazard", "'hazard'")
    expect_error(ConfigError, lambda: load(team + decoder + 'hazard_source = "hazard"\n'), "p35_trial.toml",
                 "hazard_source", "'hazard'", "turnover_hazard")
    # a turnover_hazard source that the decoder does not name: with no hazard_source, another one, or a second source
    expect_error(ConfigError, lambda: load(team + hazard.format("hazard") + decoder), "p35_trial.toml", "'hazard'",
                 "turnover_hazard", "hazard_source")
    expect_error(ConfigError, lambda: load(team + hazard.format("hazard") + hazard.format("other") + decoder +
                                           'hazard_source = "hazard"\n'), "p35_trial.toml", "'other'",
                 "turnover_hazard", "hazard_source", "'hazard'")
    vote = '[decoder]\nkind = "fake_vote"\nhalf_width = 3\n'
    expect_error(ConfigError, lambda: load(team + hazard.format("hazard") + vote), "p35_trial.toml", "'hazard'",
                 "turnover_hazard", "hazard_source")


def hmm_priced_brute_force_marginals(emissions, switch_dead, team_cost, min_possession, min_dead, paths):
    """The posterior with a per-frame team cost by its definition, over the valid ones of `paths` ((P, n) of codes H,
    A, D, every path of the block): each valid path weighs exp(its hmm_priced_scores score). (n_frames, 3)."""
    valid = paths[restart_paths_valid(paths, min_possession, min_dead)]
    scores = hmm_priced_scores(emissions, valid, switch_dead, team_cost)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return np.stack([weights @ (valid == state) for state in range(3)], axis=1)


def test_hmm_hazard_posterior_matches_brute_force():
    """marginals with a per-frame team cost against every path of tiny blocks (n <= 7, min_possession 1 to 3, min_dead
    1 to 2, blocks shorter than a minimum among them, random continuous emissions and costs): the forward pass's birth
    into frame t and the backward pass's leave towards frame t + 1 both price a change of team at the cost of the
    frame its new run starts on."""
    import itertools
    from pipeline.decode import hmm
    every = {}
    priced = short = 0
    for case in range(200):
        rng = np.random.default_rng([35, 4000 + case])
        n_frames = int(rng.integers(1, 8))
        if n_frames not in every:
            every[n_frames] = np.array(list(itertools.product(range(3), repeat=n_frames)), dtype=np.int64)
        emissions = rng.uniform(-3.0, 3.0, size=(n_frames, 3))
        switch_team, switch_dead = (float(value) for value in rng.uniform(0.1, 4.0, size=2))
        team_cost = rng.uniform(0.0, 6.0, size=n_frames)
        min_possession, min_dead = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        costs = (switch_team, switch_dead, min_possession, min_dead)
        got = hmm.marginals(emissions, *costs, team_cost=team_cost)
        expected = hmm_priced_brute_force_marginals(emissions, switch_dead, team_cost, min_possession, min_dead,
                                                    every[n_frames])
        assert got.dtype == np.float64 and got.shape == (n_frames, 3), (case, got.dtype, got.shape)
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-12), "case {} ({} frames, minimums {} and {}): {!r} " \
            "against the brute force {!r}".format(case, n_frames, min_possession, min_dead, got, expected)
        flat = hmm.marginals(emissions, *costs, team_cost=np.full(n_frames, team_cost[1:].mean() if n_frames > 1
                                                                  else 0.0))
        priced += np.abs(flat - got).max() > 1e-6
        short += n_frames < max(min_possession, min_dead)
    # The per-frame costs have to move the posterior away from a flat cost of their mean, and some blocks have to be
    # shorter than a minimum, or the check proves nothing about them.
    assert priced >= 100 and short >= 10, "{} cases priced, {} blocks shorter than a minimum, of 200".format(
        priced, short)


def test_hmm_hazard_posterior_flat_cost_is_today():
    """marginals with team_cost filled with switch_team is marginals without it, to the last bit, on random blocks;
    and a long block read backwards, its costs moved with it (the change into reversed frame u is the change into the
    original frame n - u), has the reversed posterior: the forward and the backward pass read the same frame's cost
    over thousands of frames."""
    from pipeline.decode import hmm
    moved = 0
    for case in range(8):
        rng = np.random.default_rng([35, 5000 + case])
        n_frames = int(rng.integers(300, 2000))
        emissions = rng.normal(0.0, float(rng.choice([0.3, 2.0, 30.0])), size=(n_frames, 3))
        costs = (float(rng.uniform(0.0, 20.0)), float(rng.uniform(0.0, 20.0)), int(rng.integers(1, 150)),
                 int(rng.integers(1, 150)))
        today = hmm.marginals(emissions, *costs)
        flat = hmm.marginals(emissions, *costs, team_cost=np.full(n_frames, costs[0]))
        assert np.array_equal(flat, today), "case {}: {:.1e} apart".format(case, np.abs(flat - today).max())
        team_cost = rng.uniform(1.0, 2.0 * costs[0] + 1.0, size=n_frames)
        forwards = hmm.marginals(emissions, *costs, team_cost=team_cost)
        reversed_cost = np.concatenate([team_cost[:1], team_cost[1:][::-1]])
        backwards = hmm.marginals(emissions[::-1], *costs, team_cost=reversed_cost)[::-1]
        assert np.allclose(forwards, backwards, rtol=0.0, atol=1e-10), "case {}: {:.1e} apart".format(
            case, np.abs(forwards - backwards).max())
        moved += np.abs(forwards - today).max() > 1e-6     # emissions of 30 nats a frame may leave nothing to move
    assert moved >= 4, "the costs moved the posterior in only {} of 8 long blocks".format(moved)


def test_hmm_hazard_decode_posterior():
    """decode's posterior with a hazard is marginals' at hazard_costs on the other sources' emissions, its labels are
    the same with the posterior on and off, and a silent hazard, or scale 0, gives the posterior of the block without
    the source, to the last bit."""
    from pipeline.contracts import no_opinion
    from pipeline.decode import REGISTRY, hmm
    for case in range(12):
        rng = np.random.default_rng([35, 6000 + case])
        n_frames = int(rng.integers(200, 1200))
        team = np.repeat(rng.integers(0, 2, n_frames // 40 + 1), 40)[:n_frames]
        votes = np.where(rng.random(n_frames) < np.where(team == 0, 0.75, 0.25), VOTE, 1.0 - VOTE)
        alive = np.where(rng.random(n_frames) < 0.5, np.nan, rng.random(n_frames))
        hazard = np.where(rng.random(n_frames) < 0.4, np.nan, rng.beta(0.5, 0.5, n_frames))
        costs = {"switch_team": float(rng.choice([1.0, 6.0])), "switch_dead": float(rng.uniform(2.0, 30.0)),
                 "min_possession": int(rng.integers(1, 30)), "min_dead": int(rng.integers(1, 30))}
        scale = float(rng.choice([0.5, 1.0, 2.0]))
        ctx, name = hmm_hazard_context([votes], [alive], [1.0], hazard, None, int(rng.integers(0, 2)))
        decoded = REGISTRY["hmm"](ctx, hazard_source=name, hazard_scale=scale, posterior=True, **costs)
        labels = REGISTRY["hmm"](ctx, hazard_source=name, hazard_scale=scale, **costs).labels
        assert np.array_equal(decoded.labels, labels), "case {}: the posterior moved a label".format(case)
        emissions = hmm.log_likelihoods(hmm_context(votes, alive).evidence.values(), n_frames, hmm.EPS)
        expected = hmm.marginals(emissions, *costs.values(),
                                 team_cost=hmm.hazard_costs(hazard, costs["switch_team"], scale, hmm.EPS))
        assert np.array_equal(decoded.posterior, expected), case
        without = REGISTRY["hmm"](hmm_context(votes, alive), posterior=True, **costs).posterior
        silent, silent_name = hmm_hazard_context([votes], [alive], [1.0], no_opinion(n_frames))
        assert np.array_equal(REGISTRY["hmm"](silent, hazard_source=silent_name, posterior=True, **costs).posterior,
                              without), case
        assert np.array_equal(REGISTRY["hmm"](ctx, hazard_source=name, hazard_scale=0.0, posterior=True,
                                              **costs).posterior, without), case
        assert np.abs(decoded.posterior - without).max() > 1e-6, "case {}: the hazard did not move it".format(case)


# -- dead ball (P05) --------------------------------------------------------------------------------------------------
#
# Hand-built scenes, in metres on the default 105.3 x 68.0 pitch (centre spot (52.65, 34), penalty spots (11, 34) and
# (94.3, 34), corners at the four corners). The Atd's own speed channels are NaN in every scene: nothing here may read
# them.

PLAY_NAMES = (
    "ball_missing", "ball_outside_m", "ball_speed", "ball_still_s", "ball_spot_m", "ball_in_goal_area", "since_seen_s",
    "until_seen_s", "last_outside_m", "last_speed", "next_outside_m", "next_spot_m", "n_present", "speed_mean",
    "speed_p90", "slow_share", "speed_mean_past4", "speed_mean_future4", "slow_share_c1", "slow_share_c4",
    "slow_share_c10", "speed_mean_c10", "own_half_share", "centre_player_m", "box_crowd", "corner_player_m",
    "outside_player_m", "goal_area_players", "team_gap_m", "spread_m")
PLAY_PITCH = (105.3, 68.0)
PLAY_COLUMN = {name: index for index, name in enumerate(PLAY_NAMES)}


def play_inputs(players_m, ball_m, teams=None, graded=None, kickoffs=None, pitch=PLAY_PITCH):
    """A stand-in Inputs over positions given in metres: players_m (n, S, 2) and ball_m (n, 2), NaN where absent."""
    from pipeline.inputs.atd_cache import Atd
    players_m, ball_m = np.asarray(players_m, dtype=np.float64), np.asarray(ball_m, dtype=np.float64)
    n, slots = players_m.shape[:2]
    scale = np.array(pitch)
    teams = (np.arange(slots) % 2) if teams is None else np.asarray(teams)
    atd = Atd(players_xy=players_m / scale, players_speed=np.full((n, slots), np.nan), ball_xy=ball_m / scale,
              ball_speed=np.full(n, np.nan), slot_team=teams.astype(np.int8))
    inputs = kinematics_inputs(atd, graded or [(1, 0, n)], pitch=pitch)
    inputs.kickoffs = dict(kickoffs or {})
    return inputs


def play_columns(inputs, *names):
    from pipeline.evidence import play_features
    got_names, X = play_features.features(inputs)
    assert got_names == PLAY_NAMES
    return [X[:, PLAY_COLUMN[name]] for name in names]


def nobody(n, slots=1):
    return np.full((n, slots, 2), np.nan)


def nothing(n):
    return np.full((n, 2), np.nan)


def walkers(n, speed, starts, direction=(1.0, 0.0), fps=25):
    """(n, S, 2) metres: slot s starts at starts[s] and moves at `speed` m/s along the unit vector `direction`."""
    starts = np.asarray(starts, dtype=np.float64)
    step = speed / fps * np.asarray(direction, dtype=np.float64)
    return starts[None, :, :] + np.arange(n)[:, None, None] * step


def kickoff_formation(n, home_left=True, others_speed=0.0):
    """22 slots (0-10 home, 11-21 away): home in x 12-50 and away in x 56-94 (swapped ends if not home_left), slot 0
    on the centre line 0.45 m from the centre spot and still, the rest moving along y at `others_speed`.
    Returns (players_m, teams)."""
    xs_home, xs_away = np.linspace(12, 50, 10), np.linspace(56, 94, 11)
    ys = np.linspace(12, 20, 11)
    home = np.stack([np.concatenate(([52.2], xs_home)), np.concatenate(([34.0], ys[:10]))], axis=1)
    away = np.stack([xs_away, ys], axis=1)
    if not home_left:
        home[:, 0], away[:, 0] = PLAY_PITCH[0] - home[:, 0], PLAY_PITCH[0] - away[:, 0]
    starts = np.concatenate([home, away])
    players = walkers(n, others_speed, starts, direction=(0.0, 1.0))
    players[:, 0] = starts[0]
    return players, np.array([0] * 11 + [1] * 11)


def test_features_shape_and_names():
    from pipeline.evidence import play_features
    assert play_features.FEATURE_NAMES == PLAY_NAMES and len(PLAY_NAMES) == 30
    assert play_features.FEATURE_VERSION == 1
    rng = np.random.default_rng(1)
    atd = kinematics_atd(rng.random((300, 4, 2)), rng.random((300, 2)))
    inputs = kinematics_inputs(atd, [(1, 50, 120), (2, 200, 250)])
    names, X = play_features.features(inputs)
    assert names == PLAY_NAMES and X.shape == (300, 30) and X.dtype == np.float64
    outside = np.ones(300, dtype=bool)
    outside[50:120] = outside[200:250] = False
    assert np.isnan(X[outside]).all(), "rows outside the graded blocks must be all NaN"
    assert not np.isnan(X[~outside][:, PLAY_COLUMN["n_present"]]).any()
    assert not np.isnan(X[~outside][:, PLAY_COLUMN["ball_missing"]]).any()


def test_features_hand_scenes():
    # a ball 2 m beyond the touchline
    n = 100
    outside, missing = play_columns(play_inputs(nobody(n), np.tile([50.0, -2.0], (n, 1))), "ball_outside_m",
                                    "ball_missing")
    assert np.allclose(outside, 2.0) and not missing.any()

    # a ball still on the corner point: the still run grows, is capped at 30 s, and the spot is under the ball
    n = 900
    still, spot, speed, goal = play_columns(play_inputs(nobody(n), np.zeros((n, 2))), "ball_still_s", "ball_spot_m",
                                            "ball_speed", "ball_in_goal_area")
    assert np.allclose(spot, 0.0) and np.allclose(speed, 0.0)
    assert np.isclose(still[0], 1 / 25) and np.isclose(still[74], 3.0) and np.isclose(still[749], 30.0)
    assert np.allclose(still[750:], 30.0), "ball_still_s must be capped at 30 s"
    assert np.allclose(goal, 0.0), "the corner point is not in a goal area"
    moving = play_columns(play_inputs(nobody(60), np.stack([np.linspace(30, 45, 60), np.full(60, 20.0)], axis=1)),
                          "ball_still_s")[0]
    assert np.allclose(moving, 0.0), "a moving ball has no still run"

    # a ball gap: visible 5-9 (outside, 10 m/s), missing 10-39, visible 40-59 (still, inside), missing 60-69
    n = 70
    ball = np.full((n, 2), np.nan)
    ball[5:10] = np.stack([50.0 + 0.4 * np.arange(5), np.full(5, -1.0)], axis=1)
    ball[40:60] = [10.0, 20.0]
    cols = play_columns(play_inputs(nobody(n), ball), "ball_missing", "since_seen_s", "until_seen_s",
                        "last_outside_m", "last_speed", "next_outside_m", "next_spot_m", "ball_in_goal_area")
    missing, since, until, last_out, last_speed, next_out, next_spot, goal = cols
    assert np.isnan(since[:5]).all() and np.isclose(since[5], 0.0) and np.isclose(since[25], 16 / 25)
    assert np.isclose(until[2], 3 / 25) and np.isclose(until[25], 15 / 25) and np.isclose(until[45], 0.0)
    assert np.isnan(until[60:]).all() and np.isclose(since[65], 6 / 25)
    assert np.isnan(last_out[:5]).all() and np.allclose(last_out[5:40], 1.0) and np.allclose(last_out[40:], -10.0)
    assert np.allclose(last_speed[5:40], 10.0) and np.allclose(last_speed[40:], 0.0)
    assert np.allclose(next_out[:10], 1.0) and np.allclose(next_out[10:60], -10.0) and np.isnan(next_out[60:]).all()
    assert np.allclose(next_spot[10:60], np.hypot(1.0, 14.0)), "the nearest spot of (10, 20) is the penalty spot"
    assert missing[:5].all() and not missing[5:10].any() and missing[10:40].all() and missing[60:].all()
    assert np.isnan(goal[:5]).all() and goal[5] == 0.0
    # since_seen and until_seen are capped at 60 s
    n = 2000
    ball = np.full((n, 2), np.nan)
    ball[0] = ball[n - 1] = [30.0, 30.0]
    since, until = play_columns(play_inputs(nobody(n), ball), "since_seen_s", "until_seen_s")
    assert np.isclose(since[1990], 60.0) and np.isclose(until[10], 60.0) and np.isclose(since[500], 20.0)
    # the goal area is 5.5 m deep and 18.32 m wide, centred on each goal
    ball = np.array([[3.0, 34.0], [3.0, 43.0], [3.0, 44.0], [6.0, 34.0], [100.0, 30.0], [99.0, 20.0]])
    goal = play_columns(play_inputs(nobody(6), ball), "ball_in_goal_area")[0]
    assert goal.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    # everyone walking: slow_share 1; everyone running: 0
    n = 200
    starts = np.stack([np.linspace(10, 90, 8), np.linspace(10, 60, 8)], axis=1)
    walking = play_inputs(walkers(n, 1.0, starts), nothing(n))
    slow, mean, p90, present, c1 = play_columns(walking, "slow_share", "speed_mean", "speed_p90", "n_present",
                                                "slow_share_c1")
    assert np.allclose(slow, 1.0) and np.allclose(mean, 1.0) and np.allclose(p90, 1.0) and np.allclose(present, 8)
    assert np.allclose(c1, 1.0)
    running = play_columns(play_inputs(walkers(n, 5.0, starts, direction=(0.0, 1.0)), nothing(n)), "slow_share",
                           "speed_mean")
    assert np.allclose(running[0], 0.0) and np.allclose(running[1], 5.0)

    # a kick-off formation
    players, teams = kickoff_formation(100)
    own, centre, corner, gap = play_columns(play_inputs(players, nothing(100), teams=teams),
                                            "own_half_share", "centre_player_m", "corner_player_m", "team_gap_m")
    assert np.allclose(own, 1.0) and np.allclose(centre, 0.45), (own[0], centre[0])
    assert np.all(gap > 20.0) and np.all(corner > 5.0)

    # a crowd in a box: 7 slots in the left penalty area (2 of them in the goal area), 3 in the right one (1 in its
    # goal area), one beyond the touchline and 2 in midfield
    xs = [3, 4, 8, 10, 12, 14, 16, 90, 95, 101, 30, 60, 40]
    ys = [34, 30, 20, 40, 50, 25, 34, 20, 40, 34, 34, 34, -1.5]
    box = play_inputs(np.tile(np.stack([xs, ys], axis=1)[None].astype(float), (30, 1, 1)), nothing(30))
    crowd, goal_players, outside_p = play_columns(box, "box_crowd", "goal_area_players", "outside_player_m")
    assert np.allclose(crowd, 7) and np.allclose(goal_players, 3), (crowd[0], goal_players[0])
    assert np.allclose(outside_p, 1.5)
    nobody_there = play_columns(play_inputs(nobody(10, 3), nothing(10)), "box_crowd", "n_present", "centre_player_m",
                                "speed_mean", "own_half_share", "team_gap_m")
    assert np.allclose(nobody_there[0], 0.0) and np.allclose(nobody_there[1], 0.0)
    assert all(np.isnan(column).all() for column in nobody_there[2:])


def test_features_see_only_their_block():
    from pipeline.evidence import play_features
    rng = np.random.default_rng(4)
    n, graded = 500, [(1, 0, 200), (2, 250, 450)]
    players = rng.random((n, 6, 2)) * PLAY_PITCH
    ball = rng.random((n, 2)) * PLAY_PITCH
    players[rng.random((n, 6)) < 0.3] = np.nan
    ball[rng.random(n) < 0.3] = np.nan
    base = play_features.features(play_inputs(players, ball, graded=graded))[1]
    players2, ball2 = players.copy(), ball.copy()
    players2[250:450] = rng.random((200, 6, 2)) * PLAY_PITCH
    ball2[250:450] = rng.random((200, 2)) * PLAY_PITCH
    players2[200:250] = 7.0       # frames between the blocks change nothing either
    ball2[200:250] = 7.0
    changed = play_features.features(play_inputs(players2, ball2, graded=graded))[1]
    assert np.array_equal(base[:200], changed[:200], equal_nan=True), "block 2 leaked into block 1"
    assert np.isnan(changed[200:250]).all() and np.isnan(changed[450:]).all()
    assert not np.array_equal(base[250:450], changed[250:450], equal_nan=True)
    moved = players.copy()
    moved[:200] = rng.random((200, 6, 2)) * PLAY_PITCH     # and block 1 into block 2
    again = play_features.features(play_inputs(moved, ball, graded=graded))[1]
    assert np.array_equal(base[250:450], again[250:450], equal_nan=True), "block 1 leaked into block 2"


def test_own_half_follows_each_block():
    first, teams = kickoff_formation(100, home_left=True)
    second, _ = kickoff_formation(100, home_left=False)
    inputs = play_inputs(np.concatenate([first, second]), nothing(200), teams=teams,
                         graded=[(1, 0, 100), (2, 100, 200)])
    own, centre = play_columns(inputs, "own_half_share", "centre_player_m")
    assert np.allclose(own, 1.0), "teams that swap ends keep their own half in each block"
    assert np.allclose(centre, 0.45)
    # the same two stretches in ONE block: one pair of ends for the whole block, so one stretch is in the wrong half
    one = play_inputs(np.concatenate([first, second]), nothing(200), teams=teams, graded=[(1, 0, 200)])
    assert play_columns(one, "own_half_share")[0].min() < 0.5


def play_reference(inputs):
    """Every feature with plain loops, from the tracks' own positions and speeds (kinematics is checked elsewhere)."""
    import math
    from pipeline.evidence import kinematics
    ball, players = kinematics.ball_track(inputs), kinematics.player_track(inputs)
    length, width = inputs.pitch
    fps = inputs.fps
    team_of = [int(team) for team in inputs.atd.slot_team]
    n, slots = players.present.shape
    X = np.full((n, 30), np.nan)
    spots = [(length / 2, width / 2), (11.0, width / 2), (length - 11.0, width / 2), (0.0, 0.0), (0.0, width),
             (length, 0.0), (length, width)]

    def beyond(x, y):
        bx, by = max(-x, x - length), max(-y, y - width)
        if bx > 0 or by > 0:
            return math.hypot(max(bx, 0.0), max(by, 0.0))
        return max(bx, by)

    def in_goal(x, y):
        return abs(y - width / 2) <= 9.16 and (0 <= x <= 5.5 or length - 5.5 <= x <= length)

    def in_box(x, y, left):
        return abs(y - width / 2) <= 20.16 and (0 <= x <= 16.5 if left else length - 16.5 <= x <= length)

    def window(values, t, start, stop, before, after):
        chosen = [values[u] for u in range(max(start, t - before), min(stop, t + after + 1))
                  if not math.isnan(values[u])]
        return sum(chosen) / len(chosen) if chosen else math.nan

    for _, start, stop in inputs.graded:
        sums, counts = [0.0, 0.0], [0, 0]
        for t in range(start, stop):
            for s in range(slots):
                if players.present[t, s]:
                    sums[team_of[s]] += players.xy_m[t, s, 0]
                    counts[team_of[s]] += 1
        means = [sums[k] / counts[k] if counts[k] else math.nan for k in (0, 1)]
        if counts[0] and counts[1]:
            home_left = means[0] < means[1]
        elif counts[0]:
            home_left = means[0] < length / 2
        elif counts[1]:
            home_left = means[1] > length / 2
        else:
            home_left = True
        speed_mean = [math.nan] * n
        slow = [math.nan] * n
        for t in range(start, stop):
            row = X[t]
            seen = bool(ball.visible[t])
            row[0] = 0.0 if seen else 1.0
            if seen:
                x, y = ball.xy_m[t]
                row[1] = beyond(x, y)
                row[2] = ball.speed[t]
                count = 0
                for u in range(t, start - 1, -1):
                    if ball.visible[u] and ball.speed[u] < 1.0:
                        count += 1
                    else:
                        break
                row[3] = min(count / fps, 30.0)
                row[4] = min(math.hypot(x - sx, y - sy) for sx, sy in spots)
                row[5] = 1.0 if in_goal(x, y) else 0.0
            back = [u for u in range(t, start - 1, -1) if ball.visible[u]]
            ahead = [u for u in range(t, stop) if ball.visible[u]]
            if back:
                u = back[0]
                row[6] = min((t - u) / fps, 60.0)
                row[8] = beyond(*ball.xy_m[u])
                row[9] = ball.speed[u]
            if ahead:
                u = ahead[0]
                row[7] = min((u - t) / fps, 60.0)
                row[10] = beyond(*ball.xy_m[u])
                row[11] = min(math.hypot(ball.xy_m[u, 0] - sx, ball.xy_m[u, 1] - sy) for sx, sy in spots)
            here = [s for s in range(slots) if players.present[t, s]]
            row[12] = len(here)
            speeds = [players.speed[t, s] for s in here if not math.isnan(players.speed[t, s])]
            if speeds:
                speed_mean[t] = sum(speeds) / len(speeds)
                slow[t] = sum(1 for value in speeds if value < 2.0) / len(speeds)
                row[13], row[14], row[15] = speed_mean[t], np.percentile(speeds, 90), slow[t]
            if here:
                pos = [players.xy_m[t, s] for s in here]
                own = 0
                for s in here:
                    px = players.xy_m[t, s, 0]
                    defends_left = (team_of[s] == 0) == home_left
                    own += (px <= length / 2 + 1.0) if defends_left else (px >= length / 2 - 1.0)
                row[22] = own / len(here)
                row[23] = min(math.hypot(px - length / 2, py - width / 2) for px, py in pos)
                row[24] = max(sum(1 for px, py in pos if in_box(px, py, left)) for left in (True, False))
                row[25] = min(math.hypot(px - cx, py - cy) for px, py in pos
                              for cx, cy in spots[3:])
                row[26] = max(beyond(px, py) for px, py in pos)
                row[27] = sum(1 for px, py in pos if in_goal(px, py))
            else:
                row[24] = row[27] = 0.0
            centroids, spreads = [], []
            for team in (0, 1):
                mine = [players.xy_m[t, s] for s in here if team_of[s] == team]
                if mine:
                    cx = sum(p[0] for p in mine) / len(mine)
                    cy = sum(p[1] for p in mine) / len(mine)
                    centroids.append((cx, cy))
                    spreads.append(math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in mine) / len(mine)))
            if len(centroids) == 2:
                row[28] = math.hypot(centroids[0][0] - centroids[1][0], centroids[0][1] - centroids[1][1])
                row[29] = sum(spreads) / 2
        for t in range(start, stop):
            row = X[t]
            row[16] = window(speed_mean, t, start, stop, 4 * fps, 0)
            row[17] = window(speed_mean, t, start, stop, 0, 4 * fps)
            row[18] = window(slow, t, start, stop, fps, fps)
            row[19] = window(slow, t, start, stop, 4 * fps, 4 * fps)
            row[20] = window(slow, t, start, stop, 10 * fps, 10 * fps)
            row[21] = window(speed_mean, t, start, stop, 10 * fps, 10 * fps)
    return X


def play_random_inputs(seed):
    """A small random scene: two blocks, random-walk players of slow and fast slots, a ball that comes and goes."""
    rng = np.random.default_rng(seed)
    n, slots = 160, 6
    graded = [(1, 10, 80), (2, 90, 150)]

    def toggled(shape, rate):
        flips = rng.random(shape) < rate
        return (np.cumsum(flips, axis=0) + rng.integers(2, size=shape[1:])) % 2 == 0

    walk = rng.choice([0.0002, 0.003], size=slots)
    players = (rng.random((1, slots, 2)) * 1.1 - 0.05 + np.cumsum(rng.normal(size=(n, slots, 2)) * walk[None, :, None],
                                                                   axis=0)) * PLAY_PITCH
    players[~toggled((n, slots), 0.06)] = np.nan
    start = [(0.0, 0.0), (0.5, 0.5), (0.1, 0.5), None, None][seed % 5]
    start = rng.random(2) * 1.1 - 0.05 if start is None else np.array(start)
    ball = (start + np.cumsum(rng.normal(size=(n, 2)) * rng.choice([0.00005, 0.0003]), axis=0)) * PLAY_PITCH
    ball[~toggled((n,), 0.07)] = np.nan
    teams = np.arange(slots) % 2 if seed % 3 else rng.integers(2, size=slots)
    return play_inputs(players, ball, teams=teams, graded=graded)


def test_features_match_a_loop():
    from pipeline.evidence import play_features
    for seed in range(50):
        inputs = play_random_inputs(seed)
        got = play_features.features(inputs)[1]
        want = play_reference(inputs)
        for index, name in enumerate(PLAY_NAMES):
            assert np.allclose(got[:, index], want[:, index], rtol=1e-9, atol=1e-9, equal_nan=True), \
                "seed {}: column {} differs, first at frame {}".format(
                    seed, name, int(np.flatnonzero(~np.isclose(got[:, index], want[:, index], rtol=1e-9, atol=1e-9,
                                                              equal_nan=True))[0]))


def test_features_chunks_change_nothing():
    from pipeline.evidence import play_features
    inputs = play_random_inputs(11)
    whole = play_features.features(inputs)[1]
    saved = play_features.CHUNK
    try:
        for chunk in (1, 7, 64):
            play_features.CHUNK = chunk
            again = play_features._build(inputs)
            assert np.array_equal(again, whole, equal_nan=True), chunk
    finally:
        play_features.CHUNK = saved


def test_features_read_only_inputs():
    from pipeline.evidence import play_features
    inputs = play_random_inputs(7)
    arrays = [inputs.atd.players_xy, inputs.atd.ball_xy, inputs.atd.slot_team]
    saved = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    names, X = play_features.features(inputs)
    for array, copy in zip(arrays, saved):
        assert np.array_equal(array, copy, equal_nan=True)
    assert not X.flags.writeable, "features must hand out read-only arrays"
    again_names, again = play_features.features(inputs)
    assert again is X and again_names is names, "features is remembered per Inputs object"
    other_names, other = play_features.features(play_random_inputs(8))
    assert other is not X and not np.array_equal(other, X, equal_nan=True)


RULE_VALUES = dict(alive_out=0.11, alive_still=0.22, alive_slow=0.33, alive_kickoff=0.44, alive_none=0.99)


def rule_alive(inputs, **values):
    from pipeline.contracts import RunContext, check_evidence
    from pipeline.evidence import REGISTRY
    evidence = REGISTRY["dead_ball_rules"](RunContext(inputs=inputs, seed=0), **dict(RULE_VALUES, **values))
    check_evidence(evidence, len(inputs.atd.ball_xy))
    assert np.isnan(evidence.home).all(), "dead_ball_rules has no opinion on who has the ball"
    return evidence.alive


def test_dead_ball_rules_triggers():
    n = 100
    walking = walkers(n, 1.0, np.stack([np.linspace(10, 90, 8), np.linspace(10, 60, 8)], axis=1))
    # out: a ball 2 m beyond the touchline, moving
    along = np.stack([20 + 0.4 * np.arange(n), np.full(n, -2.0)], axis=1)
    assert np.allclose(rule_alive(play_inputs(nobody(n), along)), 0.11)
    # a ball 0.4 m beyond is not out (the threshold is 0.5 m); nor does a moving ball trigger anything else
    near = np.stack([20 + 0.4 * np.arange(n), np.full(n, -0.4)], axis=1)
    assert np.allclose(rule_alive(play_inputs(nobody(n), near)), 0.99)
    # still: 2 s (50 frames) at a restart spot, or on a line, or in a goal area; not in midfield
    for where in ([52.65, 34.0], [30.0, 0.5], [3.0, 34.0], [94.3, 35.5]):
        alive = rule_alive(play_inputs(nobody(n), np.tile(where, (n, 1))))
        assert np.allclose(alive[:49], 0.99) and np.allclose(alive[49:], 0.22), where
    assert np.allclose(rule_alive(play_inputs(nobody(n), np.tile([30.0, 20.0], (n, 1)))), 0.99)
    # a ball that is missing is not still
    assert np.allclose(rule_alive(play_inputs(nobody(n), nothing(n))), 0.99)
    # slow: everyone walking
    assert np.allclose(rule_alive(play_inputs(walking, nothing(n))), 0.33)
    # kickoff: a kick-off formation whose slot at the centre stands and whose others run
    players, teams = kickoff_formation(n, others_speed=5.0)
    assert np.allclose(rule_alive(play_inputs(players, nothing(n), teams=teams)), 0.44)
    # the minimum when several fire
    assert np.allclose(rule_alive(play_inputs(players, along, teams=teams)), 0.11)
    assert np.allclose(rule_alive(play_inputs(players, along, teams=teams), alive_out=0.6), 0.44)
    still_ball = np.tile([52.65, 34.0], (n, 1))
    alive = rule_alive(play_inputs(walking, still_ball))
    assert np.allclose(alive[:49], 0.33) and np.allclose(alive[49:], 0.22)
    alive = rule_alive(play_inputs(walking, still_ball), alive_still=0.5)
    assert np.allclose(alive, 0.33)
    # no opinion outside the graded blocks
    inputs = play_inputs(np.concatenate([walking, walking]), nothing(2 * n), graded=[(1, 20, 80), (2, 120, 180)])
    alive = rule_alive(inputs)
    assert np.isnan(alive[:20]).all() and np.isnan(alive[80:120]).all() and np.isnan(alive[180:]).all()
    assert np.allclose(alive[20:80], 0.33) and np.allclose(alive[120:180], 0.33)


def test_dead_ball_pre_kickoff_is_dead():
    from pipeline.evidence import play_features
    n = 200
    walking = walkers(n, 1.0, np.stack([np.linspace(10, 90, 8), np.linspace(10, 60, 8)], axis=1))
    inputs = play_inputs(walking, nothing(n), graded=[(1, 0, 100), (2, 100, 200)], kickoffs={1: 40, 2: 150})
    alive = rule_alive(inputs)
    assert play_features.PRE_KICKOFF_ALIVE == 0.001
    assert np.allclose(alive[:40], 0.001) and np.allclose(alive[100:150], 0.001)
    assert np.allclose(alive[40:100], 0.33) and np.allclose(alive[150:], 0.33), "from the kick-off on, the rules decide"
    mask = play_features.pre_kickoff_mask(inputs)
    assert mask.shape == (n,) and mask.dtype == bool
    assert mask[:40].all() and mask[100:150].all() and not mask[40:100].any() and not mask[150:].any()


def test_dead_ball_pre_kickoff_needs_the_kickoff_in_its_block():
    from pipeline.evidence import play_features
    # the default synthetic bundle grades frames 0-600 while Inputs carries BAR-ATH's kick-offs (11185, 113742): the
    # rule must stay silent, or every synthetic frame would count as before the kick-off
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root)
        assert inputs.kickoffs == {1: 11185, 2: 113742}
        assert not play_features.pre_kickoff_mask(inputs).any()
        alive = rule_alive(inputs)
        assert not np.any(alive == 0.001) and not np.isnan(alive[:600]).any()
    n = 100
    nobody_scene = (nobody(n), nothing(n))
    # a kick-off at the block's first frame leaves nothing before it; one at its stop (or past it) is not inside it
    for kickoffs in ({1: 0}, {1: 100}, {1: 500}, {1: -5}, {}):
        inputs = play_inputs(*nobody_scene, graded=[(1, 0, 100)], kickoffs=kickoffs)
        assert not play_features.pre_kickoff_mask(inputs).any(), kickoffs
        assert np.allclose(rule_alive(inputs), 0.99), kickoffs
    inputs = play_inputs(*nobody_scene, graded=[(1, 10, 100)], kickoffs={1: 11})
    assert play_features.pre_kickoff_mask(inputs).sum() == 1 and play_features.pre_kickoff_mask(inputs)[10]
    # the second half's kick-off lies in the first block only by accident of numbering: each half uses its own
    inputs = play_inputs(np.full((200, 1, 2), np.nan), nothing(200), graded=[(1, 0, 100), (2, 100, 200)],
                         kickoffs={1: 50, 2: 50})
    mask = play_features.pre_kickoff_mask(inputs)
    assert mask[:50].all() and not mask[50:].any()


def test_dead_ball_rules_rejects_bad_parameters():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    run = RunContext(inputs=None, seed=0)       # a bad parameter is refused before the inputs are touched
    rules = REGISTRY["dead_ball_rules"]
    for name in RULE_VALUES:
        incomplete = {key: value for key, value in RULE_VALUES.items() if key != name}
        expect_error(TypeError, lambda: rules(run, **incomplete), name)
        for bad in (-0.1, 1.5, float("nan"), True, False, "0.5", None):
            expect_error(ValueError, lambda: rules(run, **dict(RULE_VALUES, **{name: bad})), "dead_ball_rules", name)
    expect_error(TypeError, lambda: rules(run, RULE_VALUES), "positional")
    for edge in (0.0, 1.0, np.float64(0.25)):
        rules(RunContext(inputs=play_inputs(nobody(5), nothing(5)), seed=0), **dict(RULE_VALUES, alive_none=edge))


def test_features_speed():
    from pipeline.evidence import kinematics, play_features
    rng = np.random.default_rng(9)
    n, slots = 180_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 75_000), (2, 100_000, 170_000)])
    inputs.kickoffs = {}
    inputs.atd.players_present                      # the Atd's own cached masks are not this module's cost
    inputs.atd.ball_present
    kinematics.ball_track(inputs)                   # the tracks have their own check (test_kinematics_speed)
    kinematics.player_track(inputs)
    started = time.perf_counter()
    play_features.features(inputs)
    elapsed = time.perf_counter() - started
    print("      features speed: 140,000 graded frames x {} slots (tracks built beforehand) in {:.2f} s".format(
        slots, elapsed))
    assert elapsed < 3.0, "{:.2f} s for 140,000 frames x 46 slots (the budget is 3 s)".format(elapsed)


# -- team shape and movement (P07) ------------------------------------------------------------------------------------
#
# Hand-built scenes in metres on the default 105.3 x 68.0 pitch, and random scenes of two blocks whose teams swap ends.
# No scene needs a ball, and test_shape_features_ignore_the_ball puts one in and sees nothing change. The names are
# spelled out here, apart from the module, so that a reordered column cannot pass by agreeing with itself.

SHAPE_NAMES = (
    "h_n", "h_adv_mean", "h_adv_p10", "h_adv_p90", "h_width", "h_vel_adv", "h_speed_mean", "h_att_third",
    "h_own_third", "h_d_adv_1s", "h_d_adv_2s", "h_vel_adv_1s",
    "a_n", "a_adv_mean", "a_adv_p10", "a_adv_p90", "a_width", "a_vel_adv", "a_speed_mean", "a_att_third",
    "a_own_third", "a_d_adv_1s", "a_d_adv_2s", "a_vel_adv_1s",
    "adv_gap", "n_diff")
SHAPE_COLUMN = {name: index for index, name in enumerate(SHAPE_NAMES)}


def shape_scene(seed, n=240, slots=16):
    """(players_m, teams, graded): two blocks, the teams swapping ends between them, every slot drifting at its own
    velocity and dropping out for a few frames at a time. In seeds 3, 7, ... one team is gone from the second block."""
    rng = np.random.default_rng(seed)
    length, width = PLAY_PITCH
    graded = [(1, 12, 130), (2, 150, 230)]
    teams = rng.permutation(np.array([0] * (slots // 2) + [1] * (slots - slots // 2)))
    home_x, away_x = rng.random(slots) * 60.0 + 5.0, rng.random(slots) * 60.0 + 40.0
    first = np.stack([np.where(teams == 0, home_x, away_x), rng.random(slots) * 64.0 + 2.0], axis=1)
    second = first * [-1.0, 1.0] + [length, 0.0]                         # everybody on the other side of the pitch
    base = np.where((np.arange(n) < 140)[:, None, None], first[None], second[None])
    drift = rng.normal(size=(slots, 2)) * 1.5                            # m/s
    players = (base + np.arange(n)[:, None, None] * drift[None] / 25
               + np.cumsum(rng.normal(size=(n, slots, 2)) * 0.04, axis=0))
    on = np.ones(slots, dtype=bool)
    for frame in range(n):                                               # a slot is lost for about 10 frames in 60
        on = np.where(on, rng.random(slots) >= 0.02, rng.random(slots) < 0.1)
        players[frame, ~on] = np.nan
    if seed % 4 == 3:
        players[150:, teams == 0] = np.nan
    return players, teams, graded


def shape_inputs(players, teams, graded, ball=None):
    return play_inputs(players, nothing(len(players)) if ball is None else ball, teams=teams, graded=graded)


def shape_matrix(players, teams, graded, ball=None):
    from pipeline.evidence import shape_features
    names, X = shape_features.features(shape_inputs(players, teams, graded, ball))
    assert names == SHAPE_NAMES
    return X


def shape_values(X, *names):
    return [X[:, SHAPE_COLUMN[name]] for name in names]


def test_shape_features_shape_and_names():
    from pipeline.evidence import shape_features
    assert shape_features.FEATURE_VERSION == 1
    assert shape_features.FEATURE_NAMES == SHAPE_NAMES and len(SHAPE_NAMES) == 26
    assert shape_features.COLUMN == SHAPE_COLUMN
    players, teams, graded = shape_scene(0)
    X = shape_matrix(players, teams, graded)
    assert X.shape == (len(players), 26) and X.dtype == np.float64
    assert np.isfinite(X[graded[0][1]:graded[0][2]]).mean() > 0.5, "the scene must give the columns something to say"


def test_shape_features_ignore_the_ball():
    rng = np.random.default_rng(5)
    for seed in range(3):
        players, teams, graded = shape_scene(seed)
        n = len(players)
        ball = rng.random((n, 2)) * PLAY_PITCH
        ball[rng.random(n) < 0.3] = np.nan
        hidden = shape_matrix(players, teams, graded)
        seen = shape_matrix(players, teams, graded, ball)
        assert hidden is not seen
        assert np.array_equal(hidden, seen, equal_nan=True), "seed {}: a column reads the ball".format(seed)
        assert np.isfinite(hidden).any()


def test_shape_features_permutation_invariant():
    rng = np.random.default_rng(6)
    for seed in range(6):
        players, teams, graded = shape_scene(seed)
        order = rng.permutation(len(teams))
        base = shape_matrix(players, teams, graded)
        permuted = shape_matrix(players[:, order], teams[order], graded)
        assert np.allclose(base, permuted, rtol=1e-12, atol=1e-9, equal_nan=True), "seed {}".format(seed)


def test_shape_features_swap():
    from pipeline.evidence import shape_features
    for seed in range(8):
        players, teams, graded = shape_scene(seed)
        X = shape_matrix(players, teams, graded)
        relabelled = shape_matrix(players, 1 - teams, graded)
        swapped = shape_features.swap(X)
        assert swapped is not X and not np.shares_memory(swapped, X) and swapped.flags.writeable
        assert swapped.shape == X.shape and swapped.dtype == np.float64
        assert np.array_equal(swapped, relabelled, equal_nan=True), "seed {}: swap is not exact".format(seed)
        assert np.array_equal(shape_features.swap(swapped), X, equal_nan=True), "swap is not an involution"
        assert np.allclose(swapped, relabelled, rtol=1e-12, atol=1e-9, equal_nan=True)
    # a hand-made row: the two team blocks trade places, the two cross columns change sign, NaN stays NaN
    row = np.arange(26, dtype=np.float64)
    row[5], row[24] = np.nan, 0.0
    flipped = shape_features.swap(row)
    assert flipped.tolist()[:12] == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    assert np.isnan(flipped[17]) and flipped[12 + 6] == 6.0 and flipped[24] == 0.0 and flipped[25] == -25.0
    assert not np.signbit(flipped[24]), "a zero stays +0.0, so that swap(X) is the relabelled features to the bit"
    assert row[0] == 0.0 and np.isnan(row[5]), "swap must leave its argument alone"
    expect_error(ValueError, lambda: shape_features.swap(np.zeros((4, 30))), "shape_features.swap", "26")
    expect_error(ValueError, lambda: shape_features.swap(np.zeros(())), "shape_features.swap", "26")


def test_shape_features_mirror_pitch():
    length = PLAY_PITCH[0]
    for seed in range(8):
        players, teams, graded = shape_scene(seed)
        mirrored = players.copy()
        mirrored[..., 0] = length - mirrored[..., 0]
        assert np.allclose(shape_matrix(players, teams, graded), shape_matrix(mirrored, teams, graded),
                           rtol=1e-12, atol=1e-9, equal_nan=True), "seed {}".format(seed)


def test_shape_features_known_values():
    length = PLAY_PITCH[0]
    n = 160
    home = [(10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
    away = [(60.0, 10.0), (70.0, 50.0)]
    players = np.tile(np.array(home + away)[None], (n, 1, 1))
    teams = np.array([0, 0, 0, 1, 1])

    def check(X, tag):
        values = dict(zip(SHAPE_NAMES, X[80]))
        want = {"h_n": 3, "h_adv_mean": 20.0, "h_adv_p10": 12.0, "h_adv_p90": 28.0, "h_width": 38.0 - 22.0,
                "h_vel_adv": 0.0, "h_speed_mean": 0.0, "h_att_third": 0.0, "h_own_third": 1.0,
                "h_d_adv_1s": 0.0, "h_d_adv_2s": 0.0, "h_vel_adv_1s": 0.0,
                "a_n": 2, "a_adv_mean": length - 65.0, "a_adv_p10": length - 70.0 + 0.1 * 10.0,
                "a_adv_p90": length - 70.0 + 0.9 * 10.0, "a_width": 46.0 - 14.0, "a_vel_adv": 0.0,
                "a_speed_mean": 0.0, "a_att_third": 0.0, "a_own_third": 0.0, "a_d_adv_1s": 0.0, "a_d_adv_2s": 0.0,
                "a_vel_adv_1s": 0.0, "adv_gap": 20.0 - (length - 65.0), "n_diff": 1}
        assert set(want) == set(SHAPE_NAMES)
        for name, expected in want.items():
            assert np.isclose(values[name], expected, rtol=1e-12, atol=1e-9), (tag, name, values[name], expected)
        assert np.isclose(values["h_adv_p10"], np.percentile([10.0, 20.0, 30.0], 10))
        assert np.isclose(values["h_adv_p90"], np.percentile([10.0, 20.0, 30.0], 90))

    check(shape_matrix(players, teams, [(1, 0, n)]), "home on the left")
    # the same scene seen from the other end of the pitch (everyone's x mirrored): nothing changes
    mirrored = players.copy()
    mirrored[..., 0] = length - mirrored[..., 0]
    check(shape_matrix(mirrored, teams, [(1, 0, n)]), "home on the right")
    # the third of the pitch is in adv: a forward line and a back line, two slots each, and a lone midfielder
    rows = np.array([(5.0, 34.0), (30.0, 10.0), (50.0, 34.0), (75.0, 20.0), (98.0, 50.0)])
    thirds = shape_matrix(np.tile(rows[None], (n, 1, 1)), np.array([0, 0, 0, 0, 0]), [(1, 0, n)])
    assert thirds[80, SHAPE_COLUMN["h_own_third"]] == 2 / 5 and thirds[80, SHAPE_COLUMN["h_att_third"]] == 2 / 5
    assert thirds[80, SHAPE_COLUMN["a_n"]] == 0 and thirds[80, SHAPE_COLUMN["n_diff"]] == 5
    # one team on the move: both walk along +x at 2 m/s, so the home team (attacking +x) gains 2 m/s and the away team
    # (attacking -x) loses 2 m/s; adv_mean moves 4 m in a second each way round
    n = 200
    starts = np.array(home + away)
    walking = walkers(n, 2.0, starts, direction=(1.0, 0.0))
    X = shape_matrix(walking, teams, [(1, 0, n)])
    (h_vel, a_vel, h_speed, a_speed, h_vel_1s, a_vel_1s, h_d1, h_d2, a_d1, a_d2) = shape_values(
        X, "h_vel_adv", "a_vel_adv", "h_speed_mean", "a_speed_mean", "h_vel_adv_1s", "a_vel_adv_1s", "h_d_adv_1s",
        "h_d_adv_2s", "a_d_adv_1s", "a_d_adv_2s")
    assert np.allclose(h_vel, 2.0) and np.allclose(a_vel, -2.0) and np.allclose(h_speed, 2.0)
    assert np.allclose(a_speed, 2.0) and np.allclose(h_vel_1s, 2.0) and np.allclose(a_vel_1s, -2.0)
    assert np.allclose(h_d1[25:175], 4.0) and np.allclose(a_d1[25:175], -4.0)
    assert np.allclose(h_d2[50:150], 8.0) and np.allclose(a_d2[50:150], -8.0)
    # ... and a team that walks across the pitch the other way round: the same numbers, mirrored
    mirrored = walking.copy()
    mirrored[..., 0] = length - mirrored[..., 0]
    again = shape_matrix(mirrored, teams, [(1, 0, n)])
    assert np.allclose(again, X, rtol=1e-12, atol=1e-9, equal_nan=True)


def test_shape_features_nan_rules():
    rng = np.random.default_rng(8)
    # outside the graded blocks every column is NaN, `n` and `n_diff` included
    players, teams, graded = shape_scene(1)
    X = shape_matrix(players, teams, graded)
    outside = np.ones(len(players), dtype=bool)
    for _, start, stop in graded:
        outside[start:stop] = False
    assert np.isnan(X[outside]).all(), "rows outside the graded blocks must be all NaN"
    assert not np.isnan(X[~outside][:, [SHAPE_COLUMN["h_n"], SHAPE_COLUMN["a_n"], SHAPE_COLUMN["n_diff"]]]).any()

    # a team with nobody present has n = 0 and NaN elsewhere, windows included, even where its slots are back within
    # a second; the other team's columns and n_diff are unaffected
    n = 160
    players = rng.random((n, 6, 2)) * PLAY_PITCH
    players[:, 3:, 0] = players[:, 3:, 0] * 0.4 + 60.0
    players[:, :3, 0] = players[:, :3, 0] * 0.4
    players[60:100, 3:] = np.nan
    X = shape_matrix(players, np.array([0, 0, 0, 1, 1, 1]), [(1, 0, n)])
    away = [name for name in SHAPE_NAMES if name.startswith("a_") and name != "a_n"]
    gone = np.zeros(n, dtype=bool)
    gone[60:100] = True
    assert (X[gone, SHAPE_COLUMN["a_n"]] == 0).all() and (X[~gone, SHAPE_COLUMN["a_n"]] == 3).all()
    for name in away + ["adv_gap"]:
        assert np.isnan(X[gone, SHAPE_COLUMN[name]]).all(), name
    for name in ("a_adv_mean", "a_d_adv_1s", "a_d_adv_2s", "a_vel_adv_1s", "adv_gap"):
        assert np.isfinite(X[~gone, SHAPE_COLUMN[name]]).any(), name
    assert (X[gone, SHAPE_COLUMN["n_diff"]] == 3).all() and not np.isnan(X[gone, SHAPE_COLUMN["h_adv_mean"]]).any()
    assert np.isfinite(X[gone, SHAPE_COLUMN["h_d_adv_1s"]]).any()
    # frame 80 is the case: its slots are gone, and the frames a second either side of it are not
    assert np.isfinite(X[[55, 105], SHAPE_COLUMN["a_adv_mean"]]).all() and np.isnan(X[80, SHAPE_COLUMN["a_d_adv_1s"]])

    # no slot of a team at all, and nobody at all
    X = shape_matrix(players[:, :3], np.array([0, 0, 0]), [(1, 0, n)])
    assert (X[:, SHAPE_COLUMN["a_n"]] == 0).all() and np.isnan(X[:, 12 + 1:24]).all() and np.isnan(X[:, 24]).all()
    assert (X[:, SHAPE_COLUMN["n_diff"]] == 3).all()
    X = shape_matrix(nobody(20, 4), np.array([0, 1, 0, 1]), [(1, 0, 20)])
    assert (X[:, SHAPE_COLUMN["h_n"]] == 0).all() and (X[:, SHAPE_COLUMN["a_n"]] == 0).all()
    assert (X[:, SHAPE_COLUMN["n_diff"]] == 0).all()
    assert np.isnan(X[:, [SHAPE_COLUMN[name] for name in SHAPE_NAMES if name not in ("h_n", "a_n", "n_diff")]]).all()

    # d_adv_1s needs 25 frames each side inside the block and d_adv_2s 50; a full row of slots is there throughout
    n, (start, stop) = 200, (20, 180)
    players = rng.random((n, 8, 2)) * PLAY_PITCH
    X = shape_matrix(players, np.arange(8) % 2, [(1, start, stop)])
    for name, k in (("d_adv_1s", 25), ("d_adv_2s", 50)):
        for prefix in ("h_", "a_"):
            column = X[:, SHAPE_COLUMN[prefix + name]]
            assert np.isnan(column[:start + k]).all() and np.isnan(column[stop - k:]).all(), prefix + name
            assert np.isfinite(column[start + k:stop - k]).all(), prefix + name
    # two blocks that touch: the look-ahead stops at the half's end, so the first block's last second is NaN
    X = shape_matrix(rng.random((200, 8, 2)) * PLAY_PITCH, np.arange(8) % 2, [(1, 0, 100), (2, 100, 200)])
    column = X[:, SHAPE_COLUMN["h_d_adv_1s"]]
    assert np.isnan(column[75:125]).all() and np.isfinite(column[25:75]).all() and np.isfinite(column[125:175]).all()
    assert np.isnan(column[:25]).all() and np.isnan(column[175:]).all()


def test_shape_features_see_only_their_block():
    rng = np.random.default_rng(12)
    players, teams, graded = shape_scene(2)
    (_, first_start, first_stop), (_, second_start, second_stop) = graded
    base = shape_matrix(players, teams, graded)
    moved = players.copy()
    moved[second_start:second_stop] = rng.random((second_stop - second_start, len(teams), 2)) * PLAY_PITCH
    moved[first_stop:second_start] = 7.0                                  # the frames between the blocks change nothing
    again = shape_matrix(moved, teams, graded)
    assert np.array_equal(base[:first_stop], again[:first_stop], equal_nan=True), "block 2 leaked into block 1"
    assert np.isnan(again[first_stop:second_start]).all() and np.isnan(again[second_stop:]).all()
    assert not np.array_equal(base[second_start:second_stop], again[second_start:second_stop], equal_nan=True)
    moved = players.copy()
    moved[:first_stop] = rng.random((first_stop, len(teams), 2)) * PLAY_PITCH
    third = shape_matrix(moved, teams, graded)
    assert np.array_equal(base[second_start:], third[second_start:], equal_nan=True), "block 1 leaked into block 2"


def test_shape_features_follow_each_blocks_direction():
    # the same kick-off formation twice, the teams swapping ends: the home team is on the left in the first block and
    # on the right in the second, and each team's adv is measured from the goal line it defends in its own block
    first, teams = kickoff_formation(100, home_left=True)
    second, _ = kickoff_formation(100, home_left=False)
    X = shape_matrix(np.concatenate([first, second]), teams, [(1, 0, 100), (2, 100, 200)])
    h_mean, a_mean, h_own, h_att = shape_values(X, "h_adv_mean", "a_adv_mean", "h_own_third", "h_att_third")
    home_adv = (52.2 + np.linspace(12, 50, 10).sum()) / 11               # x of the home slots in the first block
    assert np.allclose(h_mean, home_adv) and np.allclose(a_mean, PLAY_PITCH[0] - 75.0)
    assert np.allclose(h_own, 6 / 11) and np.allclose(h_att, 0.0)        # 6 of 11 slots under L/3, none over 2L/3
    assert np.allclose(X[:, SHAPE_COLUMN["adv_gap"]], home_adv - (PLAY_PITCH[0] - 75.0))
    # one block for both stretches: a single pair of ends, so one of the two stretches has the teams the wrong way
    one = shape_matrix(np.concatenate([first, second]), teams, [(1, 0, 200)])
    assert not np.allclose(one[:100, SHAPE_COLUMN["h_adv_mean"]], one[100:, SHAPE_COLUMN["h_adv_mean"]])


def shape_reference(inputs):
    """Every shape feature with plain loops over the present slots, from the tracks' own positions and velocities
    (kinematics is checked elsewhere) and each block's direction from `play_features.own_half_is_left`."""
    from pipeline.evidence import kinematics, play_features
    players = kinematics.player_track(inputs)
    length, fps = inputs.pitch[0], inputs.fps
    teams = np.asarray(inputs.atd.slot_team)
    n, slots = players.present.shape
    X = np.full((n, 26), np.nan)
    one_second = int(round(fps))
    for _, start, stop in inputs.graded:
        home_left = play_features.own_half_is_left(players, teams, start, stop, length)
        for base, team in ((0, 0), (12, 1)):
            right = home_left if team == 0 else not home_left
            sign = 1.0 if right else -1.0
            for t in range(start, stop):
                mine = [s for s in range(slots) if teams[s] == team and players.present[t, s]]
                X[t, base] = len(mine)
                if not mine:
                    continue
                x, y = players.xy_m[t, mine, 0], players.xy_m[t, mine, 1]
                adv = x if right else length - x
                X[t, base + 1] = np.mean(adv)
                X[t, base + 2], X[t, base + 3] = np.percentile(adv, [10, 90])
                low, high = np.percentile(y, [10, 90])
                X[t, base + 4] = high - low
                known = [s for s in mine if np.isfinite(players.speed[t, s])]
                if known:
                    X[t, base + 5] = np.mean([sign * players.vel[t, s, 0] for s in known])
                    X[t, base + 6] = np.mean([players.speed[t, s] for s in known])
                X[t, base + 7] = np.mean(adv > 2 * length / 3)
                X[t, base + 8] = np.mean(adv < length / 3)
            count, mean, velocity = X[start:stop, base], X[start:stop, base + 1], X[start:stop, base + 5]
            for t in range(stop - start):
                if count[t] == 0:
                    continue
                for k, column in ((one_second, 9), (2 * one_second, 10)):
                    if t - k >= 0 and t + k < stop - start:
                        X[start + t, base + column] = mean[t + k] - mean[t - k]
                near = velocity[max(t - one_second, 0):min(t + one_second, stop - start - 1) + 1]
                near = near[np.isfinite(near)]
                if len(near):
                    X[start + t, base + 11] = near.mean()
    X[:, 24] = X[:, 1] - X[:, 13]
    X[:, 25] = X[:, 0] - X[:, 12]
    return X


def test_shape_features_match_a_loop():
    from pipeline.evidence import shape_features
    for seed in range(24):
        players, teams, graded = shape_scene(seed)
        inputs = shape_inputs(players, teams, graded)
        got, want = shape_features.features(inputs)[1], shape_reference(inputs)
        for index, name in enumerate(SHAPE_NAMES):
            close = np.isclose(got[:, index], want[:, index], rtol=1e-9, atol=1e-9, equal_nan=True)
            assert close.all(), "seed {}: column {} differs, first at frame {}".format(
                seed, name, int(np.flatnonzero(~close)[0]))


def test_shape_features_chunks_change_nothing():
    from pipeline.evidence import shape_features
    players, teams, graded = shape_scene(11)
    inputs = shape_inputs(players, teams, graded)
    whole = shape_features.features(inputs)[1]
    saved = shape_features.CHUNK
    try:
        for chunk in (1, 7, 64):
            shape_features.CHUNK = chunk
            assert np.array_equal(shape_features._build(inputs), whole, equal_nan=True), chunk
    finally:
        shape_features.CHUNK = saved


def test_shape_features_memo_and_read_only():
    from pipeline.evidence import shape_features
    players, teams, graded = shape_scene(7)
    inputs = shape_inputs(players, teams, graded)
    arrays = [inputs.atd.players_xy, inputs.atd.ball_xy, inputs.atd.slot_team]
    saved = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    names, X = shape_features.features(inputs)
    for array, copy in zip(arrays, saved):
        assert np.array_equal(array, copy, equal_nan=True), "features must not write to the inputs"
    assert not X.flags.writeable, "features must hand out read-only arrays"
    again_names, again = shape_features.features(inputs)
    assert again is X and again_names is names, "features is remembered per Inputs object"
    other_names, other = shape_features.features(shape_inputs(*shape_scene(8)))
    assert other is not X and not np.array_equal(other, X, equal_nan=True)


def test_shape_features_speed():
    from pipeline.evidence import shape_features
    rng = np.random.default_rng(10)
    n, slots = 180_000, 30
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 85_000), (2, 100_000, 180_000)])
    started = time.perf_counter()                   # from a fresh Inputs: the tracks and the Atd's masks are in it
    shape_features.features(inputs)
    elapsed = time.perf_counter() - started
    print("      shape features speed: 160,000 graded frames x {} slots (tracks included) in {:.2f} s".format(
        slots, elapsed))
    assert elapsed < 6.0, "{:.2f} s for 160,000 frames x 30 slots (the budget is 6 s)".format(elapsed)


V2_TEAM = ("n", "adv_mean", "adv_p10", "adv_p90", "width", "vel_adv", "speed_mean", "att_third", "own_third",
           "d_adv_1s", "d_adv_2s", "vel_adv_1s", "length", "spread", "adv_past_2s", "adv_past_5s", "adv_past_10s",
           "adv_fut_2s", "adv_fut_5s", "adv_fut_10s")


def shape_matrix_v2(players, teams, graded):
    from pipeline.evidence import shape_features
    names, X = shape_features.features(shape_inputs(players, teams, graded), 2)
    assert len(names) == 42
    return {name: X[:, index] for index, name in enumerate(names)}, X


def test_shape_features_v2_names_and_shape():
    from pipeline.evidence import shape_features
    names = shape_features.feature_names(2)
    assert names == tuple("h_" + t for t in V2_TEAM) + tuple("a_" + t for t in V2_TEAM) + ("adv_gap", "n_diff")
    assert len(names) == 42 and len(set(names)) == 42
    assert shape_features.VERSIONS == (1, 2) and shape_features.ONE_SIDED_S == (2.0, 5.0, 10.0)
    assert shape_features.feature_names(1) == shape_features.FEATURE_NAMES == SHAPE_NAMES
    assert shape_features.columns(2) == {name: index for index, name in enumerate(names)}
    assert shape_features.columns(1) == shape_features.COLUMN
    players, teams, graded = shape_scene(0)
    inputs = shape_inputs(players, teams, graded)
    assert shape_features.features(inputs)[1] is shape_features.features(inputs, 1)[1]
    names2, X2 = shape_features.features(inputs, 2)
    assert names2 == names and X2.shape == (len(players), 42) and X2.dtype == np.float64
    assert not X2.flags.writeable and shape_features.features(inputs, 2)[1] is X2
    assert np.isfinite(X2[graded[0][1]:graded[0][2], shape_features.columns(2)["h_spread"]]).any()
    for retired in (3, 4):                           # P33's and P37's versions are retired and never reused
        expect_error(ValueError, lambda: shape_features.features(inputs, retired), "shape_features", "version",
                     str(retired))
        expect_error(ValueError, lambda: shape_features.feature_names(retired), "shape_features", "version",
                     str(retired))


def test_shape_features_v2_keeps_v1():
    from pipeline.evidence import shape_features
    for seed in range(4):
        players, teams, graded = shape_scene(seed)
        inputs = shape_inputs(players, teams, graded)
        names1, X1 = shape_features.features(inputs, 1)
        names2, X2 = shape_features.features(inputs, 2)
        column = shape_features.columns(2)
        for index, name in enumerate(names1):
            assert np.array_equal(X1[:, index], X2[:, column[name]], equal_nan=True), (seed, name)


def test_shape_features_v2_known_values():
    n = 600
    # a team of two slots 10 m apart: both are 5 m from their centroid
    players = np.tile(np.array([(10.0, 10.0), (20.0, 10.0), (60.0, 30.0), (70.0, 40.0)])[None], (n, 1, 1))
    values, _ = shape_matrix_v2(players, np.array([0, 0, 1, 1]), [(1, 0, n)])
    assert np.allclose(values["h_spread"][:n], 5.0, atol=1e-9)
    assert np.allclose(values["a_spread"][:n], np.hypot(10.0, 10.0) / 2, atol=1e-9)
    assert np.allclose(values["h_length"], values["h_adv_p90"] - values["h_adv_p10"], atol=1e-12)
    # every slot advances 0.1 m a frame (2.5 m/s): the mean advance moves 0.1 k m over k frames
    starts = np.array([(10.0, 10.0), (20.0, 20.0), (60.0, 30.0)])
    walking = starts[None] + np.arange(n)[:, None, None] * np.array([0.1, 0.0])[None, None]
    values, _ = shape_matrix_v2(walking, np.array([0, 0, 1]), [(1, 0, n)])
    for name, want, k in (("adv_past_2s", 5.0, 50), ("adv_fut_5s", 12.5, 125), ("adv_past_10s", 25.0, 250)):
        column = values["h_" + name]
        assert np.isfinite(column).any()
        assert np.allclose(column[np.isfinite(column)], want, atol=1e-9, rtol=0), name
        assert np.isfinite(column).sum() == n - k, name
    # the away team attacks the other way, and walks +x: it loses ground
    assert np.allclose(values["a_adv_past_2s"][50:], -5.0, atol=1e-9)
    assert np.allclose(values["h_spread"], np.hypot(10.0, 10.0) / 2, atol=1e-9)


def test_shape_features_v2_nan_at_block_edges():
    rng = np.random.default_rng(31)
    n = 1400
    players = rng.random((n, 8, 2)) * PLAY_PITCH
    graded = [(1, 100, 700), (2, 750, 1350)]
    values, X = shape_matrix_v2(players, np.arange(8) % 2, graded)
    for prefix in ("h_", "a_"):
        for start, stop in ((100, 700), (750, 1350)):
            for name, k, side in (("adv_past_2s", 50, "past"), ("adv_past_10s", 250, "past"),
                                  ("adv_fut_2s", 50, "fut"), ("adv_fut_10s", 250, "fut")):
                column = values[prefix + name]
                if side == "past":
                    assert np.isnan(column[start:start + k]).all() and np.isfinite(column[start + k]), name
                    assert np.isfinite(column[start + k:stop]).all(), name
                else:
                    assert np.isnan(column[stop - k:stop]).all() and np.isfinite(column[stop - k - 1]), name
                    assert np.isfinite(column[start:stop - k]).all(), name
    outside = np.ones(n, dtype=bool)
    outside[100:700] = outside[750:1350] = False
    assert np.isnan(X[outside]).all()
    # a team that is gone for a stretch has NaN in every column that reads that stretch
    gone = players.copy()
    gone[300:320, 1::2] = np.nan
    values, _ = shape_matrix_v2(gone, np.arange(8) % 2, graded)
    assert np.isnan(values["a_spread"][300:320]).all() and np.isnan(values["a_length"][300:320]).all()
    assert np.isnan(values["a_adv_past_2s"][300:320]).all() and np.isnan(values["a_adv_past_2s"][350:370]).all()
    assert np.isfinite(values["a_adv_past_2s"][320:350]).all() and np.isnan(values["a_adv_fut_2s"][250:270]).all()
    assert np.isnan(values["a_adv_fut_2s"][300:320]).all() and np.isfinite(values["a_adv_fut_2s"][270:300]).all()
    assert np.isfinite(values["a_adv_past_2s"][371]) and np.isfinite(values["h_adv_past_2s"][300:370]).all()


def test_shape_features_v2_swap():
    from pipeline.evidence import shape_features
    for seed in range(8):
        players, teams, graded = shape_scene(seed)
        X2 = shape_features.features(shape_inputs(players, teams, graded), 2)[1]
        relabelled = shape_features.features(shape_inputs(players, 1 - teams, graded), 2)[1]
        swapped = shape_features.swap(X2)
        assert swapped is not X2 and not np.shares_memory(swapped, X2) and swapped.shape == X2.shape
        assert np.array_equal(swapped, relabelled, equal_nan=True), "seed {}: swap is not exact".format(seed)
        assert np.array_equal(shape_features.swap(swapped), X2, equal_nan=True)
    row = np.arange(42, dtype=np.float64)
    flipped = shape_features.swap(row)
    assert flipped[:20].tolist() == list(range(20, 40)) and flipped[20:40].tolist() == list(range(20))
    assert flipped[40] == -40.0 and flipped[41] == -41.0
    expect_error(ValueError, lambda: shape_features.swap(np.zeros((4, 30))), "shape_features.swap", "30")


# -- the shape possession source (P07) --------------------------------------------------------------------------------
#
# Hand-made artifacts whose trees are stumps on the shape features, so that every number is worked out here and not by
# the trainer. A stump on adv_gap with the baseline 0.7 and the leaves -1.0 (gap <= 0) and 2.0 (gap > 0) is not
# antisymmetric by itself (its sigmoid is not); the source's (r(x) - r(swap(x))) / 2 makes it so, and cancels the
# baseline: +-1.5 where the gap has a sign, 0 where it is 0 or NaN.

SHAPE_TRAINER = "pipeline/benchmark/train_shape.py"
STUMP_LOGIT = 1.5


def shape_model(baseline, left, right, feature="adv_gap", threshold=0.0, missing_left=0, prior=0.5, version=1):
    from pipeline.evidence import shape_features
    return {"baseline": baseline, "prior_home": prior,
            "trees": [stump(shape_features.columns(version)[feature], threshold, left, right, missing_left)]}


def write_shape_model(folder, folds=None, full=None, name="shape_hgb.json", version=1, **changes):
    import json
    from pipeline.evidence import shape_features
    artifact = {"format": 1, "kind": "hgb", "feature_version": version,
                "features": list(shape_features.feature_names(version)), "profile": "tracking", "seeds": 3, "stride": 5,
                "provenance": {},
                "folds": {"FOLD1": shape_model(-0.2, 1.0, -1.0, version=version)} if folds is None else folds,
                "full": shape_model(0.7, -1.0, 2.0, version=version) if full is None else full}
    artifact.update(changes)
    return write_text(folder, name, json.dumps(artifact, sort_keys=True) + "\n")


def shape_home(inputs, path, game="BAR-ATH", **values):
    from pipeline.contracts import RunContext, check_evidence
    from pipeline.evidence import REGISTRY
    inputs.game = game
    evidence = REGISTRY["shape_possession"](RunContext(inputs=inputs, seed=0), model=path, **values)
    check_evidence(evidence, len(inputs.atd.ball_xy))
    assert np.isnan(evidence.alive).all(), "shape_possession has no opinion on whether the ball is alive"
    assert not (evidence.home.flags.writeable or evidence.alive.flags.writeable), "the arrays are read-only"
    return evidence.home


def stump_home(X, magnitude=STUMP_LOGIT, min_prob=0.02):
    """The home probability of a stump on adv_gap, worked out from the gap alone: sign(gap) * magnitude as a logit,
    0 where the gap is 0 or NaN, then clipped."""
    gap = X[:, SHAPE_COLUMN["adv_gap"]]
    logit_home = np.where(np.isnan(gap), 0.0, np.sign(gap)) * magnitude
    return np.clip(1.0 / (1.0 + np.exp(-logit_home)), min_prob, 1.0 - min_prob)


def someone_present(players):
    return (~np.isnan(players[:, :, 0])).any(axis=1)


def block_mask(n, graded):
    mask = np.zeros(n, dtype=bool)
    for _, start, stop in graded:
        mask[start:stop] = True
    return mask


def test_shape_possession_values():
    # logit(home) = (r(x) - r(swap(x))) / 2: the baseline cancels, so no prior is ever corrected for
    with tempfile.TemporaryDirectory() as root:
        for baseline in (0.7, -3.0, 0.0):
            path = write_shape_model(root, full=shape_model(baseline, -1.0, 2.0))
            for seed in (0, 1, 3, 5):
                players, teams, graded = shape_scene(seed)
                home = shape_home(shape_inputs(players, teams, graded), path, scope="all")
                X = shape_matrix(players, teams, graded)
                known = ~np.isnan(home)
                assert known.sum() > 100 and (np.abs(home[known] - 0.5) > 0.2).any()
                assert np.allclose(home[known], stump_home(X)[known], rtol=0, atol=1e-12), (baseline, seed)
        # a model that gives every row the same leaf has no opinion to give: exactly 0.5, whatever its baseline
        flat = write_shape_model(root, full=shape_model(2.5, 0.8, 0.8), name="flat.json")
        home = shape_home(shape_inputs(*shape_scene(0)), flat, scope="all")
        assert (home[~np.isnan(home)] == 0.5).all() and (~np.isnan(home)).any()


def test_shape_possession_is_antisymmetric():
    from pipeline.evidence import shape_features, trees
    with tempfile.TemporaryDirectory() as root:
        # none of the three trees is symmetric: one reads the home block only, one the away block only, one the gap
        lopsided = {"baseline": 0.9, "prior_home": 0.5, "trees": [
            stump(SHAPE_COLUMN["h_adv_mean"], 50.0, -0.8, 1.7, missing_left=1),
            stump(SHAPE_COLUMN["a_vel_adv"], 0.0, 0.6, -0.3),
            stump(SHAPE_COLUMN["adv_gap"], 0.0, -1.0, 2.0)]}
        path = write_shape_model(root, full=lopsided)
        for seed in range(8):
            players, teams, graded = shape_scene(seed)
            home = shape_home(shape_inputs(players, teams, graded), path, scope="all")
            relabelled = shape_home(shape_inputs(players, 1 - teams, graded), path, scope="all")
            known = ~np.isnan(home)
            assert known.sum() > 100 and np.array_equal(known, ~np.isnan(relabelled)), seed
            assert np.allclose(home[known], 1.0 - relabelled[known], rtol=0, atol=1e-12), \
                "seed {}: relabelling the teams is not 1 - home".format(seed)
            # the check has teeth: the trees alone, run on the two labellings, are not antisymmetric
            X = shape_matrix(players, teams, graded)[known]
            alone = 1.0 / (1.0 + np.exp(-trees.score(lopsided, X)))
            alone_relabelled = 1.0 / (1.0 + np.exp(-trees.score(lopsided, shape_features.swap(X))))
            assert np.abs(alone - (1.0 - alone_relabelled)).max() > 0.05, "the model is symmetric on its own"


def test_shape_possession_scope():
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root)
        rng = np.random.default_rng(11)
        for seed in (0, 2, 3, 6):
            players, teams, graded = shape_scene(seed)
            n = len(players)
            players[60:70] = np.nan                         # both teams gone for ten frames of the first block
            ball = rng.random((n, 2)) * PLAY_PITCH
            ball[rng.random(n) < 0.4] = np.nan
            ball[:6] = ball[130:140] = np.nan               # ungraded frames, with players and no ball
            inputs = shape_inputs(players, teams, graded, ball)
            graded_rows = block_mask(n, graded)
            someone = someone_present(players)
            seen = ~np.isnan(ball[:, 0])
            assert (graded_rows & someone & seen).any() and (graded_rows & someone & ~seen).any()
            assert (graded_rows & ~someone).any() and (~graded_rows & someone & ~seen).any()
            home_all = shape_home(inputs, path, scope="all")
            assert np.array_equal(~np.isnan(home_all), graded_rows & someone), seed
            home_gaps = shape_home(inputs, path, scope="gaps")
            assert np.array_equal(~np.isnan(home_gaps), graded_rows & someone & ~seen), seed
            assert np.array_equal(shape_home(inputs, path), home_gaps, equal_nan=True), "gaps is the default"
            both = ~np.isnan(home_gaps)
            assert np.array_equal(home_all[both], home_gaps[both]), "the scope only chooses where to speak"
        # a team that is absent says nothing, but the other team's presence still gives an opinion (seed 3)
        players, teams, graded = shape_scene(3)
        X = shape_matrix(players, teams, graded)
        lonely = (X[:, SHAPE_COLUMN["h_n"]] == 0) & (X[:, SHAPE_COLUMN["a_n"]] > 0)
        assert lonely.any()
        home = shape_home(shape_inputs(players, teams, graded), path, scope="all")
        assert not np.isnan(home[lonely]).any()


def test_shape_possession_fold_choice():
    with tempfile.TemporaryDirectory() as root:
        folds = {"FOLD1": shape_model(0.0, 1.0, -1.0), "FOLD2": shape_model(0.0, 3.0, -3.0)}
        path = write_shape_model(root, folds=folds, full=shape_model(0.0, -1.0, 1.0))
        players, teams, graded = shape_scene(0)
        X = shape_matrix(players, teams, graded)
        inputs = shape_inputs(players, teams, graded)
        values = dict(scope="all", min_prob=0.0001)
        full = shape_home(inputs, path, **values)
        known = ~np.isnan(full)
        assert np.allclose(full[known], stump_home(X, 1.0, 0.0001)[known], rtol=0, atol=1e-12)
        side = np.sign(X[:, SHAPE_COLUMN["adv_gap"]])[known]                     # +1 home ahead, -1 away, 0/NaN mute
        assert (side == 1).any() and (side == -1).any(), "the scene must give the models something to disagree on"
        for game, magnitude in (("FOLD1", -1.0), ("FOLD2", -3.0)):
            fold = shape_home(inputs, path, game=game, **values)
            assert np.array_equal(known, ~np.isnan(fold))
            expected = 1.0 / (1.0 + np.exp(-np.where(np.isnan(side), 0.0, side) * magnitude))
            assert np.allclose(fold[known], expected, rtol=0, atol=1e-12), "{} is the fold of its own game".format(game)
        for game in ("BAR-ATH", "NOT-IN-FOLDS", "full"):
            other = shape_home(inputs, path, game=game, **values)
            assert np.array_equal(other, full, equal_nan=True), "{}: a game with no fold gets `full`".format(game)
        # the file is read again when it changes. The new model is as long as the old one, and a rewrite within one tick
        # of the file system's clock keeps the modification time too (230 of 300 in a probe on Windows), which is all
        # the cache compares with the size: so the test gives the file a later time itself, as a retrain would have
        before = os.stat(path).st_mtime_ns
        write_shape_model(root, folds=folds, full=shape_model(0.0, 5.0, -5.0))
        os.utime(path, ns=(before + 10 ** 9, before + 10 ** 9))
        changed = shape_home(inputs, path, **values)
        expected = 1.0 / (1.0 + np.exp(5.0 * np.where(np.isnan(side), 0.0, side)))
        assert np.allclose(changed[known], expected, rtol=0, atol=1e-12)


def test_shape_possession_rejects_bad_artifacts():
    import json
    from pipeline.evidence import shape_features
    with tempfile.TemporaryDirectory() as root:
        inputs = shape_inputs(*shape_scene(0))
        renamed = list(SHAPE_NAMES)
        renamed[3] = "h_adv_p95"
        good = shape_model(0.0, -1.0, 1.0)

        def refused(path, *fragments):
            return expect_error(ValueError, lambda: shape_home(inputs, path), "shape_possession", SHAPE_TRAINER,
                                *fragments)

        for label, changes in (("format", {"format": 2}), ("format", {"format": None}),
                               ("features", {"feature_version": 2}),    # version 2 with version 1's 26 names
                               ("features", {"features": list(shape_features.FEATURE_NAMES_V2)}),   # and the reverse
                               ("feature_version", {"feature_version": max(shape_features.VERSIONS) + 1}),
                               ("feature_version", {"feature_version": 3}),    # retired (P33)
                               ("feature_version", {"feature_version": 4}),    # retired (P37)
                               ("feature_version", {"feature_version": 0}),
                               ("feature_version", {"feature_version": None}),
                               ("feature_version", {"feature_version": True}),
                               ("feature_version", {"feature_version": 2.0}),
                               ("feature_version", {"feature_version": "1"}),
                               ("features", {"features": list(SHAPE_NAMES)[::-1]}), ("features", {"features": renamed}),
                               ("features", {"features": list(SHAPE_NAMES)[:-1]}),
                               ("features", {"features": ["a"] * 30}),
                               ("kind", {"kind": "logistic"}), ("kind", {"kind": None}),
                               ("folds", {"folds": [good]}), ("folds", {"folds": "FOLD1"})):
            refused(write_shape_model(root, name="other.json", **changes), label)
        refused(write_text(root, "list.json", "[1, 2]"), "model object")
        # a version 2 file whose trees reach for column 42 (it has 0 to 41)
        refused(write_shape_model(root, name="narrow2.json", version=2,
                                  full=dict(good, trees=[stump(42, 0.0, -1.0, 1.0)])), "tree 0", "full")
        refused(write_shape_model(root, name="wide1.json", full=dict(good, trees=[stump(30, 0.0, -1.0, 1.0)])),
                "tree 0", "full")
        for bad in (0.6, 0.0, 1.0, -0.5, True, "0.5", None):
            refused(write_shape_model(root, full=dict(good, prior_home=bad), name="prior.json"), "prior_home", "full")
            refused(write_shape_model(root, folds={"FOLD1": dict(good, prior_home=bad)}, name="prior.json"),
                    "prior_home", "FOLD1")
        no_prior = dict(good)
        no_prior.pop("prior_home")
        refused(write_shape_model(root, full=no_prior, name="noprior.json"), "prior_home")
        ragged = shape_model(0.0, -1.0, 1.0)
        ragged["trees"][0]["left"] = [1, 0]                      # arrays of unequal lengths
        refused(write_shape_model(root, full=ragged, name="ragged.json"), "tree 0", "full")
        wide = shape_model(0.0, -1.0, 1.0)
        wide["trees"][0]["feature"][0] = len(SHAPE_NAMES)        # a column the features do not have
        refused(write_shape_model(root, folds={"FOLD1": wide}, name="wide.json"), "tree 0", "FOLD1")
        refused(write_shape_model(root, full=dict(good, baseline=None), name="nobase.json"), "baseline")
        no_full = json.loads(open(write_shape_model(root), encoding="utf-8").read())
        no_full.pop("full")
        refused(write_text(root, "nofull.json", json.dumps(no_full)), "full")
        refused(os.path.join(root, "absent.json"), "absent.json")
        write_text(root, "text.json", "not json at all")
        refused(os.path.join(root, "text.json"), "text.json")
        # dead_ball_model's file is a model of other features: it must not load here
        refused(write_model(root, name="dead.json"), "features")


def test_shape_possession_reads_a_v2_file():
    from pipeline.evidence import shape_features
    column = shape_features.columns(2)
    with tempfile.TemporaryDirectory() as root:
        # a stump on the home team's advance over the next 2 s, leaves -1.0 (<= 0 or NaN, missing_left) and 2.0
        stump_model = shape_model(0.4, -1.0, 2.0, feature="h_adv_fut_2s", missing_left=1, version=2)
        path = write_shape_model(root, version=2, full=stump_model, folds={"FOLD1": stump_model})
        sides, same = 0, 0
        for seed in range(4):
            players, teams, graded = shape_scene(seed)
            inputs = shape_inputs(players, teams, graded)
            home = shape_home(inputs, path, scope="all")
            X = shape_features.features(inputs, 2)[1]

            def leaf(values):
                left = np.isnan(values) | (values <= 0.0)         # a NaN goes where missing_left sends it: left
                return np.where(left, -1.0, 2.0)
            own, other = X[:, column["h_adv_fut_2s"]], X[:, column["a_adv_fut_2s"]]
            want = np.clip(1.0 / (1.0 + np.exp(-(leaf(own) - leaf(other)) / 2.0)), 0.02, 0.98)
            known = ~np.isnan(home)
            assert known.sum() > 100, seed
            assert np.allclose(home[known], want[known], rtol=0, atol=1e-12), seed
            apart = known & (leaf(own) != leaf(other))
            together = known & (leaf(own) == leaf(other))
            assert np.allclose(np.abs(np.log(home[apart] / (1.0 - home[apart]))), 1.5, rtol=0, atol=1e-12)
            assert np.array_equal(home[together], np.full(int(together.sum()), 0.5))
            sides += int(apart.sum())
            same += int(together.sum())
        assert sides > 50 and same > 50, "the scene must hold both cases ({}, {})".format(sides, same)


def test_shape_possession_parameters():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root)
        players, teams, graded = shape_scene(3)         # the home team is gone from the second block
        inputs = shape_inputs(players, teams, graded)
        for bad in ("GAPS", "", "none", None, True, 1, ["gaps"]):
            expect_error(ValueError, lambda: shape_home(inputs, path, scope=bad), "shape_possession", "scope", "gaps",
                         "all")
        for bad in (0.0, 0.5, 0.6, -0.1, True, False, "0.1", None, float("nan"), float("inf")):
            expect_error(ValueError, lambda: shape_home(inputs, path, min_prob=bad), "shape_possession", "min_prob")
        for bad in (None, 5, "", True):
            expect_error(ValueError, lambda: shape_home(inputs, bad), "shape_possession", "model")
        expect_error(TypeError, lambda: REGISTRY["shape_possession"](RunContext(inputs=inputs, seed=0)), "model")
        shape_home(inputs, path, scope="all", min_prob=np.float64(0.25))        # numpy numbers from a sweep are fine
        shape_home(inputs, path, scope="gaps", min_prob=0.001)
        # the defaults are the design's: scope "gaps", min_prob 0.02
        assert np.array_equal(shape_home(inputs, path), shape_home(inputs, path, scope="gaps", min_prob=0.02),
                              equal_nan=True)
        # the clip: a score far out is held at min_prob and 1 - min_prob, and a mute row gets 0.5
        extreme = write_shape_model(root, full=shape_model(0.0, -40.0, 40.0), name="extreme.json")
        gap = shape_matrix(players, teams, graded)[:, SHAPE_COLUMN["adv_gap"]]
        assert (gap > 0).any() and (gap < 0).any() and (np.isnan(gap) & block_mask(len(gap), graded)).any()
        for min_prob in (0.02, 0.1, 0.4):
            home = shape_home(inputs, extreme, scope="all", min_prob=min_prob)
            known = ~np.isnan(home)
            assert known.any() and home[known].min() >= min_prob and home[known].max() <= 1.0 - min_prob
            assert np.allclose(home[known & (gap > 0)], 1.0 - min_prob, rtol=0, atol=1e-12)
            assert np.allclose(home[known & (gap < 0)], min_prob, rtol=0, atol=1e-12)
            assert np.array_equal(home[known & ~(np.abs(gap) > 0)], np.full((known & ~(np.abs(gap) > 0)).sum(), 0.5))
        home = shape_home(inputs, path, scope="all")
        assert home.dtype == np.float64 and home.shape == (len(players),)


def test_shape_possession_reads_read_only_inputs():
    from pipeline.evidence import shape_features
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root)
        players, teams, graded = shape_scene(5)
        inputs = shape_inputs(players, teams, graded, np.full((len(players), 2), np.nan))
        arrays = [inputs.atd.players_xy, inputs.atd.ball_xy, inputs.atd.slot_team]
        saved = [array.copy() for array in arrays]
        for array in arrays:
            array.flags.writeable = False
        before = shape_features.features(inputs)[1].copy()
        first = shape_home(inputs, path)
        for array, copy in zip(arrays, saved):
            assert np.array_equal(array, copy, equal_nan=True), "the source must not write to the inputs"
        assert np.array_equal(shape_features.features(inputs)[1], before, equal_nan=True), "the features were changed"
        second = shape_home(inputs, path)
        assert np.array_equal(first, second, equal_nan=True), "a second call on the same inputs gives other numbers"


def test_shape_possession_remembers_its_scores_per_inputs():
    from pipeline.evidence import trees
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root)
        inputs = shape_inputs(*shape_scene(0))
        calls = []
        real = trees.score
        trees.score = lambda model, X: calls.append(len(X)) or real(model, X)
        try:
            first = shape_home(inputs, path, scope="all")
            walked = len(calls)
            assert walked >= 1
            second = shape_home(inputs, path, scope="all")
            assert len(calls) == walked and np.array_equal(first, second, equal_nan=True), "walked the trees again"
            gaps = shape_home(inputs, path, scope="gaps", min_prob=0.3)             # another scope and clip: no walk
            assert len(calls) == walked
            assert np.nanmin(gaps) >= 0.3 and np.nanmax(gaps) <= 0.7, "a different min_prob must re-clip the score"
            shape_home(inputs, path, game="FOLD1", scope="all")                      # another model of the file
            assert len(calls) == 2 * walked
            shape_home(inputs, path, game="FOLD1", scope="all")
            assert len(calls) == 2 * walked
            shape_home(shape_inputs(*shape_scene(0)), path)
            assert len(calls) == 3 * walked, "the memo is per Inputs object"
        finally:
            trees.score = real


def test_shape_possession_refuses_infinite_features():
    from pipeline.evidence import shape_features
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root)
        inputs = shape_inputs(*shape_scene(0))
        names, X = shape_features.features(inputs)
        broken = X.copy()
        broken[40, SHAPE_COLUMN["h_adv_mean"]] = np.inf
        real = shape_features.features
        shape_features.features = lambda given, version=1: (names, broken)
        try:
            expect_error(ValueError, lambda: shape_home(inputs, path), "shape_possession", "infinite")
        finally:
            shape_features.features = real


def test_shape_possession_speed():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY, shape_features
    rng = np.random.default_rng(12)
    n, slots = 180_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 75_000), (2, 100_000, 170_000)])
    inputs.game = "BAR-ATH"
    inputs.atd.players_present                      # the Atd's own cached masks are not this module's cost
    inputs.atd.ball_present
    model = random_hgb_model(rng, prior=0.5)        # of the size the trainer writes, over the 26 shape columns
    model.pop("prior_alive")
    model["prior_home"] = 0.5
    for tree in model["trees"]:
        tree["feature"] = [feature % len(SHAPE_NAMES) if feature >= 0 else feature for feature in tree["feature"]]
    with tempfile.TemporaryDirectory() as root:
        path = write_shape_model(root, full=model)
        started = time.perf_counter()
        shape_features.features(inputs)             # warm: the features have their own check
        features_s = time.perf_counter() - started
        started = time.perf_counter()
        evidence = REGISTRY["shape_possession"](RunContext(inputs=inputs, seed=0), model=path)
        first_s = time.perf_counter() - started
        started = time.perf_counter()
        REGISTRY["shape_possession"](RunContext(inputs=inputs, seed=0), model=path, scope="all", min_prob=0.01)
        again_s = time.perf_counter() - started
        print("      shape_possession speed: 140,000 graded frames, 300 trees of 15 leaves, two walks: first inference "
              "{:.2f} s (features {:.2f} s before it, {:.2f} s in all), a second config on the same inputs {:.2f} s"
              .format(first_s, features_s, features_s + first_s, again_s))
        assert np.isnan(evidence.home[:5_000]).all() and not np.isnan(evidence.home[5_000:75_000]).all()
        assert first_s < 6.0, "{:.2f} s for 300 trees, twice, on 140,000 frames with warm features (the budget is " \
                              "6 s)".format(first_s)


# -- team features (P19, version 2 for P32) ---------------------------------------------------------------------------
#
# Hand-built scenes in metres on the default pitch, the names spelled out here so that a reordered column cannot pass
# by agreeing with itself. The ATD's speed channels are NaN (or random garbage) in every scene: nothing may read them.
# Version 1 (P19's 56 columns) is in git at commit 15e7e14; version 2 drops over11, since_gap_s and until_gap_s, adds
# ball_adv and ball_vel_adv to each team, and takes shape_features' version 2 (42 columns) as its shape block.

TEAM_BLOCK = ("ball_d1", "ball_d2", "closing", "rel_speed", "reach_s", "in_episode", "since_episode_s",
              "until_episode_s", "ball_adv", "ball_vel_adv")
TEAM_SHARED = ("ball_speed_02", "ball_speed_05", "ball_speed_10", "ball_acc_02", "ball_acc_05", "ball_acc_10",
               "ball_turn_02", "ball_turn_05", "ball_turn_10", "ball_line_m")
TEAM_SHAPE = tuple("h_" + name for name in V2_TEAM) + tuple("a_" + name for name in V2_TEAM) + ("adv_gap", "n_diff")
TEAM_NAMES = (tuple("h_" + name for name in TEAM_BLOCK) + tuple("a_" + name for name in TEAM_BLOCK) + TEAM_SHARED
              + TEAM_SHAPE)
TEAM_COLUMN = {name: index for index, name in enumerate(TEAM_NAMES)}


def team_matrix(players, ball, teams, graded):
    from pipeline.evidence import team_features
    names, X = team_features.features(play_inputs(players, ball, teams=teams, graded=graded))
    assert names == TEAM_NAMES
    return X


def team_values(X, *names):
    return [X[:, TEAM_COLUMN[name]] for name in names]


def team_scene(seed, n=240, slots=16):
    """shape_scene's two blocks and a ball that rides on slot 0 (0.4 m off it) and is lost for stretches of frames."""
    players, teams, graded = shape_scene(seed, n=n, slots=slots)
    rng = np.random.default_rng([seed, 19])
    ball = players[:, 0] + np.array([0.4, 0.3])
    lost = np.zeros(n, dtype=bool)
    for start in rng.integers(0, n, size=6):
        lost[start:start + rng.integers(5, 30)] = True
    ball[lost] = np.nan
    return players, ball, teams, graded


def test_team_features_shape_and_names():
    from pipeline.evidence import shape_features, team_features
    assert team_features.FEATURE_VERSION == 2 and team_features.SHAPE_VERSION == 2
    assert team_features.TEAM_COLUMNS == TEAM_BLOCK and team_features.SHARED_COLUMNS == TEAM_SHARED
    assert team_features.FEATURE_NAMES == TEAM_NAMES and len(TEAM_NAMES) == 72 and len(set(TEAM_NAMES)) == 72
    assert len(TEAM_BLOCK) == 10 and len(TEAM_SHARED) == 10 and len(TEAM_SHAPE) == 42
    assert TEAM_NAMES[-42:] == TEAM_SHAPE == shape_features.feature_names(2) and team_features.COLUMN == TEAM_COLUMN
    players, ball, teams, graded = team_scene(0)
    X = team_matrix(players, ball, teams, graded)
    assert X.shape == (len(players), 72) and X.dtype == np.float64
    assert np.isfinite(X[graded[0][1]:graded[0][2]]).mean() > 0.5, "the scene must give the columns something to say"
    inputs = play_inputs(players, ball, teams=teams, graded=graded)
    assert np.array_equal(team_features.features(inputs)[1][:, 30:], shape_features.features(inputs, 2)[1],
                          equal_nan=True), "the shape block is shape_features' version 2, unchanged"


def test_team_features_version_1_columns_are_gone():
    from pipeline.evidence import team_features
    for name in ("h_over11", "a_over11", "since_gap_s", "until_gap_s"):
        assert name not in team_features.FEATURE_NAMES and name not in team_features.COLUMN, name
    assert "over11" not in team_features.TEAM_COLUMNS
    assert not {"since_gap_s", "until_gap_s"} & set(team_features.SHARED_COLUMNS)
    assert not hasattr(team_features, "MAX_SLOTS") and not hasattr(team_features, "GAP_CAP_S"), "version 1's constants"


def test_team_features_constants_are_the_designs():
    import inspect
    from pipeline.config import load_config
    from pipeline.evidence import controlled_touch, team_features
    assert [int(round(seconds * 25)) for seconds in team_features.WINDOWS_S] == [5, 12, 25]
    assert int(round(team_features.CLOSING_S * 25)) == 5
    assert team_features.REACTION_S == 0.7 and team_features.REACH_SPEED == 5.0
    assert team_features.EPISODE_CAP_S == 30.0
    assert team_features.TOUCH_RULE == {"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1, "duel_radius": 1.0}
    config = load_config(os.path.join(HERE, "configs", "main.toml"))
    rule = [part.params for part in config.evidence if part.kind == "controlled_touch"]
    assert len(rule) == 1
    # main.toml leaves duel_radius out, so controlled_touch's own default stands for it there
    default = inspect.signature(controlled_touch.compute).parameters["duel_radius"].default
    main_rule = dict({"duel_radius": default}, **rule[0])
    assert {key: main_rule[key] for key in team_features.TOUCH_RULE} == team_features.TOUCH_RULE, \
        "TOUCH_RULE is main.toml's touch rule (R013, kept by R030)"


def test_team_features_known_values():
    n = 100
    players = np.tile(np.array([(50.0, 34.0), (53.0, 34.0), (60.0, 34.0)])[None], (n, 1, 1))
    teams = np.array([0, 1, 1])
    X = team_matrix(players, np.tile([50.5, 34.0], (n, 1)), teams, [(1, 0, n)])
    got = dict(zip(TEAM_NAMES, X[50]))
    want = {"h_ball_d1": 0.5, "a_ball_d1": 2.5, "a_ball_d2": 9.5, "h_closing": 0.0, "a_closing": 0.0,
            "h_rel_speed": 0.0, "a_rel_speed": 0.0, "h_reach_s": 0.7 + 0.5 / 5.0, "a_reach_s": 0.7 + 2.5 / 5.0,
            "h_ball_adv": 50.5, "a_ball_adv": PLAY_PITCH[0] - 50.5, "h_ball_vel_adv": 0.0, "a_ball_vel_adv": 0.0,
            "ball_line_m": -34.0}
    for name in TEAM_SHARED[:9]:
        want[name] = 0.0
    for name, expected in want.items():
        assert abs(got[name] - expected) < 1e-12, (name, got[name], expected)
    assert np.isnan(got["h_ball_d2"]), "one present slot has no second-nearest"
    # a ball at 10 m/s along x: speed 10, no acceleration, no turn
    ball = np.stack([20.0 + 10.0 / 25 * np.arange(n), np.full(n, 34.0)], axis=1)
    X = team_matrix(players, ball, teams, [(1, 0, n)])
    for tag in ("02", "05", "10"):
        speed, acc, turn = team_values(X, "ball_speed_" + tag, "ball_acc_" + tag, "ball_turn_" + tag)
        assert abs(speed[50] - 10.0) < 1e-9 and abs(acc[50]) < 1e-9 and turn[50] == 0.0, tag
    # a right-angle turn at t = 50: +x before it, +y after it
    t = np.arange(n)
    ball = np.stack([np.where(t <= 50, 30.0 + 0.4 * (t - 50), 30.0), np.where(t <= 50, 34.0, 34.0 + 0.4 * (t - 50))],
                    axis=1)
    turn = team_values(team_matrix(players, ball, teams, [(1, 0, n)]), "ball_turn_02")[0]
    assert abs(turn[50] - np.pi / 2) < 1e-12, turn[50]
    assert turn[40] == 0.0 and turn[60] == 0.0, "a straight stretch turns by nothing"
    # approaching: the nearest slot closes on a still ball at 5 m/s, so `closing` and `rel_speed` are 5
    walking = walkers(n, 5.0, [(30.0, 34.0), (80.0, 34.0)], direction=(1.0, 0.0))
    X = team_matrix(walking, np.tile([60.0, 34.0], (n, 1)), np.array([0, 1]), [(1, 0, n)])
    closing, rel = team_values(X, "h_closing", "h_rel_speed")
    assert abs(closing[50] - 5.0) < 1e-9 and abs(rel[50] - 5.0) < 1e-9
    assert np.isnan(closing[3]) and np.isnan(closing[n - 3]), "closing needs 5 frames on each side inside the block"


def test_team_features_ball_direction():
    # Two blocks, the teams swapping ends between them (home defends x < L/2 in the first, x > L/2 in the second), and
    # a ball that moves towards the goal home attacks at 2 m/s in each: +x in the first block, -x in the second
    length = PLAY_PITCH[0]
    n = 300
    graded = [(1, 0, 140), (2, 160, 300)]
    first = np.array([(20.0, 30.0), (30.0, 40.0), (70.0, 30.0), (80.0, 40.0)])
    second = first * [-1.0, 1.0] + [length, 0.0]
    players = np.where((np.arange(n) < 150)[:, None, None], first[None], second[None])
    teams = np.array([0, 0, 1, 1])
    t = np.arange(n)
    x = np.where(t < 150, 40.0 + 2.0 / 25 * t, length - 40.0 - 2.0 / 25 * (t - 160))
    ball = np.stack([x, np.full(n, 34.0)], axis=1)
    X = team_matrix(players, ball, teams, graded)
    h_adv, a_adv, h_vel, a_vel = team_values(X, "h_ball_adv", "a_ball_adv", "h_ball_vel_adv", "a_ball_vel_adv")
    for start, stop, home_attacks_right in ((0, 140, True), (160, 300, False)):
        rows = slice(start, stop)
        assert np.allclose(h_vel[rows], 2.0, rtol=0, atol=1e-9), "towards home's attacking goal: +2 m/s for home"
        assert np.allclose(a_vel[rows], -2.0, rtol=0, atol=1e-9), "and -2 m/s for away, which attacks the other one"
        # the team's own frame: metres from its own goal line, x where it attacks towards x = L, else L - x
        home_frame = x[rows] if home_attacks_right else length - x[rows]
        assert np.allclose(h_adv[rows], home_frame, rtol=0, atol=1e-9), (start, stop)
        assert np.allclose(a_adv[rows], length - home_frame, rtol=0, atol=1e-9), (start, stop)
    assert np.isnan(X[140:160][:, [TEAM_COLUMN[name] for name in ("h_ball_adv", "a_ball_vel_adv")]]).all()

    # The very frame shape_features gives a team's adv_mean: a ball carried exactly by a team's only slot has that
    # team's adv_mean as its ball_adv and its vel_adv as its ball_vel_adv, in either block (shape_scene's teams swap
    # ends between its blocks, and in seed 3 one team is gone from the second block)
    from pipeline.evidence import shape_features
    column = shape_features.columns(2)
    for seed in range(4):
        players, teams, graded = shape_scene(seed, slots=2)
        for team, prefix in ((0, "h_"), (1, "a_")):
            carried = players[:, int(np.flatnonzero(teams == team)[0])]
            inputs = play_inputs(players, carried, teams=teams, graded=graded)
            X = team_matrix(players, carried, teams, graded)
            shape = shape_features.features(inputs, 2)[1]
            adv, vel = team_values(X, prefix + "ball_adv", prefix + "ball_vel_adv")
            assert np.array_equal(np.isnan(adv), np.isnan(shape[:, column[prefix + "adv_mean"]])), (seed, prefix)
            assert np.isfinite(adv).sum() > 50, (seed, prefix)          # seed 3: home is gone from block 2
            assert np.allclose(adv, shape[:, column[prefix + "adv_mean"]], rtol=0, atol=1e-9, equal_nan=True)
            assert np.allclose(vel, shape[:, column[prefix + "vel_adv"]], rtol=0, atol=1e-9, equal_nan=True)


def episode_scene(n, home_frames, away_frames):
    players = np.full((n, 2, 2), np.nan)
    players[:, 0] = beside_ball(n, home_frames)
    players[:, 1] = beside_ball(n, away_frames, offset=(-0.5, 0.0))
    return players, np.tile([30.0, 30.0], (n, 1)), np.array([0, 1])


def test_team_features_episodes():
    from pipeline.contracts import RunContext
    from pipeline.evidence import controlled_touch
    players, ball, teams = episode_scene(200, range(50, 60), range(120, 130))
    inputs = play_inputs(players, ball, teams=teams, graded=[(1, 0, 200)])
    touch = controlled_touch.compute(RunContext(inputs=inputs, seed=0), radius=1.5, max_rel_speed=6.0, min_frames=1,
                                     confidence=0.99, hold=0.5).home
    assert np.flatnonzero(np.isfinite(touch)).tolist() == list(range(50, 60)) + list(range(120, 130)), "the scene"
    assert (touch[50:60] > 0.5).all() and (touch[120:130] < 0.5).all()
    X = team_matrix(players, ball, teams, [(1, 0, 200)])
    h_in, a_in, h_since, h_until, a_since, a_until = team_values(
        X, "h_in_episode", "a_in_episode", "h_since_episode_s", "h_until_episode_s", "a_since_episode_s",
        "a_until_episode_s")
    assert h_in.tolist() == [1.0 if 50 <= t < 60 else 0.0 for t in range(200)]
    assert a_in.tolist() == [1.0 if 120 <= t < 130 else 0.0 for t in range(200)]
    assert h_since[70] == (70 - 59) / 25 and h_since[40] == 30.0 and h_until[40] == 0.4 and h_until[55] == 0.0
    assert h_since[55] == 0.0 and h_until[70] == 30.0, "no later episode: the cap"
    assert a_until[70] == (120 - 70) / 25 and a_since[150] == (150 - 129) / 25
    # an episode 40 s away is capped at 30 s
    players, ball, teams = episode_scene(1300, range(1100, 1110), range(0))
    X = team_matrix(players, ball, teams, [(1, 0, 1300)])
    since, until = team_values(X, "h_since_episode_s", "h_until_episode_s")
    assert since[1200] == (1200 - 1109) / 25 and until[10] == 30.0 and since[10] == 30.0, "44 s away"
    # an episode never reaches across a block's edge
    players, ball, teams = episode_scene(200, range(50, 60), range(0))
    X = team_matrix(players, ball, teams, [(1, 0, 100), (2, 120, 200)])
    since, until = team_values(X, "h_since_episode_s", "h_until_episode_s")
    assert since[130] == 30.0 and until[120] == 30.0, "block 2 does not see block 1's episode"
    assert until[0] == 2.0 and since[99] == (99 - 59) / 25


def test_team_features_where_the_ball_is_hidden():
    n = 200
    players = np.tile(np.array([(10.0, 10.0), (60.0, 30.0)])[None], (n, 1, 1))
    ball = np.stack([30.0 + 2.0 / 25 * np.arange(n), np.full(n, 30.0)], axis=1)     # moving, so ball_vel_adv is not 0
    ball[80:100] = np.nan
    X = team_matrix(players, ball, np.array([0, 1]), [(1, 0, n)])
    ball_columns = [TEAM_COLUMN[name] for name in TEAM_SHARED]
    assert np.isnan(X[80:100][:, ball_columns]).all(), "no ball, no ball column"
    needs_the_ball = [TEAM_COLUMN[prefix + name] for prefix in ("h_", "a_")
                      for name in ("ball_d1", "reach_s", "rel_speed", "ball_adv", "ball_vel_adv")]
    assert np.isnan(X[80:100][:, needs_the_ball]).all()
    assert np.isfinite(X[:80][:, needs_the_ball]).all() and np.isfinite(X[100:][:, needs_the_ball]).all()
    h_vel, a_vel = team_values(X, "h_ball_vel_adv", "a_ball_vel_adv")
    assert np.allclose(h_vel[100:], 2.0, rtol=0, atol=1e-9) and np.allclose(a_vel[:80], -2.0, rtol=0, atol=1e-9), \
        "the velocity of a run is its own: the gap does not reach into it"
    in_episode = [TEAM_COLUMN[prefix + name] for prefix in ("h_", "a_")
                  for name in ("in_episode", "since_episode_s", "until_episode_s")]
    assert np.isfinite(X[80:100][:, in_episode]).all(), "the episode columns do not need the ball"
    assert np.isfinite(X[80:100][:, 30:][:, [TEAM_SHAPE.index("h_adv_mean"), TEAM_SHAPE.index("a_n")]]).all()


def test_team_features_swap():
    from pipeline.evidence import shape_features, team_features
    for seed in range(6):
        players, ball, teams, graded = team_scene(seed)
        X = team_matrix(players, ball, teams, graded)
        relabelled = team_matrix(players, ball, 1 - teams, graded)
        swapped = team_features.swap(X)
        assert swapped is not X and not np.shares_memory(swapped, X) and swapped.flags.writeable
        assert np.array_equal(swapped, relabelled, equal_nan=True), "seed {}: swap is not exact".format(seed)
        assert np.array_equal(team_features.swap(swapped), X, equal_nan=True), "swap is not an involution"
        assert np.array_equal(swapped[:, 20:30], X[:, 20:30], equal_nan=True), "the shared columns are untouched"
        assert np.array_equal(swapped[:, 30:], shape_features.swap(X[:, 30:]), equal_nan=True)
        assert np.array_equal(swapped[:, :10], X[:, 10:20], equal_nan=True)
        assert np.array_equal(swapped[:, 10:20], X[:, :10], equal_nan=True)
        assert np.nansum(team_values(X, "h_in_episode", "a_in_episode")) > 0, "seed {}: no episodes".format(seed)
        assert np.isfinite(team_values(X, "h_ball_vel_adv")[0]).sum() > 50, "seed {}: no direction".format(seed)
    row = np.arange(72, dtype=np.float64)
    flipped = team_features.swap(row)
    assert flipped[:10].tolist() == list(range(10, 20)) and flipped[10:20].tolist() == list(range(10))
    assert flipped[20:30].tolist() == list(range(20, 30)), "the shared ten stay"
    assert np.array_equal(flipped[30:], shape_features.swap(row[30:])), "the 42 go through shape_features.swap"
    for width in (26, 42, 56, 71, 73):                      # shape's two widths, version 1's 56, one either side of 72
        expect_error(ValueError, lambda: team_features.swap(np.zeros((4, width))), "team_features.swap", "72",
                     str(width))
    expect_error(ValueError, lambda: team_features.swap(np.zeros(())), "team_features.swap", "72")


def test_team_features_ignore_speed_channels():
    from pipeline.evidence import team_features
    players, ball, teams, graded = team_scene(2)
    pitch = np.array(PLAY_PITCH)
    both = []
    for seed in (1, 2):
        atd = kinematics_atd(players / pitch, ball / pitch, seed=seed)
        atd.slot_team[:] = teams
        both.append(team_features.features(kinematics_inputs(atd, graded))[1])
    assert np.array_equal(both[0], both[1], equal_nan=True), "a column reads the ATD's own speed channels (T31)"
    assert np.isfinite(both[0]).any()


def test_team_features_see_only_their_block():
    rng = np.random.default_rng(13)
    players, ball, teams, graded = team_scene(2)
    (_, first_start, first_stop), (_, second_start, second_stop) = graded
    base = team_matrix(players, ball, teams, graded)
    moved, moved_ball = players.copy(), ball.copy()
    moved[second_start:second_stop] = rng.random((second_stop - second_start, len(teams), 2)) * PLAY_PITCH
    moved_ball[second_start:second_stop] = moved[second_start:second_stop, 0]
    moved[first_stop:second_start], moved_ball[first_stop:second_start] = 7.0, 7.0
    again = team_matrix(moved, moved_ball, teams, graded)
    assert np.array_equal(base[:first_stop], again[:first_stop], equal_nan=True), "block 2 leaked into block 1"
    assert np.isnan(again[:first_start]).all() and np.isnan(again[first_stop:second_start]).all()
    assert np.isnan(again[second_stop:]).all(), "rows outside the blocks are all NaN"
    assert not np.array_equal(base[second_start:second_stop], again[second_start:second_stop], equal_nan=True)
    moved, moved_ball = players.copy(), ball.copy()
    moved[:first_stop] = rng.random((first_stop, len(teams), 2)) * PLAY_PITCH
    moved_ball[:first_stop] = moved[:first_stop, 0]
    third = team_matrix(moved, moved_ball, teams, graded)
    assert np.array_equal(base[second_start:], third[second_start:], equal_nan=True), "block 1 leaked into block 2"


def test_team_features_memo_and_read_only():
    from pipeline.evidence import team_features
    players, ball, teams, graded = team_scene(7)
    inputs = play_inputs(players, ball, teams=teams, graded=graded)
    names, X = team_features.features(inputs)
    assert not X.flags.writeable, "features must hand out read-only arrays"
    again_names, again = team_features.features(inputs)
    assert again is X and again_names is names, "features is remembered per Inputs object"
    players, ball, teams, graded = team_scene(8)
    other = team_features.features(play_inputs(players, ball, teams=teams, graded=graded))[1]
    assert other is not X and not np.array_equal(other, X, equal_nan=True)


def test_team_features_speed():
    from pipeline.evidence import team_features
    rng = np.random.default_rng(14)
    n, slots = 140_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 70_000), (2, 75_000, 135_000)])
    started = time.perf_counter()                   # from a fresh Inputs: the tracks and the Atd's masks are in it
    team_features.features(inputs)
    elapsed = time.perf_counter() - started
    print("      team features speed: 125,000 graded frames x {} slots (tracks, shape and touch included) in {:.2f} "
          "s".format(slots, elapsed))
    assert elapsed < 10.0, "{:.2f} s for 125,000 frames x 46 slots (the budget is 10 s)".format(elapsed)


# -- the team model source (P19, team features version 2 for P32) -----------------------------------------------------
#
# Hand-made artifacts whose trees are stumps on `h_ball_d1`, so that every number is worked out here and not by the
# trainer. A stump on h_ball_d1 is not antisymmetric by itself; the source's (r(x) - r(swap(x))) / 2 makes it so: the
# logit is (leaf(h_ball_d1) - leaf(a_ball_d1)) / 2, the baseline cancels, and a NaN distance goes to the missing side.

TEAM_TRAINER = "pipeline/benchmark/train_team.py"
TEAM_THRESHOLD = 5.0
TEAM_VERSION = 2                     # team_features' version, written out: a file of version 1 must not load


def team_model_stub(baseline, left, right, feature="h_ball_d1", threshold=TEAM_THRESHOLD, missing_left=0, prior=0.5):
    return {"baseline": baseline, "prior_home": prior,
            "trees": [stump(TEAM_COLUMN[feature], threshold, left, right, missing_left)]}


def write_team_model(folder, folds=None, full=None, name="team_hgb.json", **changes):
    import json
    artifact = {"format": 1, "kind": "hgb", "feature_version": TEAM_VERSION,
                "features": list(TEAM_NAMES), "profile": "generic", "seeds": 3, "stride": 5, "provenance": {},
                "folds": {"FOLD1": team_model_stub(-0.2, 1.0, -1.0)} if folds is None else folds,
                "full": team_model_stub(0.7, -1.0, 2.0) if full is None else full}
    artifact.update(changes)
    return write_text(folder, name, json.dumps(artifact, sort_keys=True) + "\n")


def team_home(inputs, path, game="BAR-ATH", **values):
    from pipeline.contracts import RunContext, check_evidence
    from pipeline.evidence import REGISTRY
    inputs.game = game
    evidence = REGISTRY["team_model"](RunContext(inputs=inputs, seed=0), model=path, **values)
    check_evidence(evidence, len(inputs.atd.ball_xy))
    assert np.isnan(evidence.alive).all(), "team_model has no opinion on whether the ball is alive"
    assert not (evidence.home.flags.writeable or evidence.alive.flags.writeable), "the arrays are read-only"
    return evidence.home


def team_inputs(players, ball, teams, graded):
    return play_inputs(players, ball, teams=teams, graded=graded)


def stump_team_home(X, left, right, threshold=TEAM_THRESHOLD, missing_left=0, min_prob=0.02):
    """The home probability of a stump on h_ball_d1, worked out from the two nearest distances alone."""
    def leaf(distance):
        goes_left = np.where(np.isnan(distance), bool(missing_left), distance <= threshold)
        return np.where(goes_left, left, right)

    logit_home = (leaf(X[:, TEAM_COLUMN["h_ball_d1"]]) - leaf(X[:, TEAM_COLUMN["a_ball_d1"]])) / 2.0
    return np.clip(1.0 / (1.0 + np.exp(-logit_home)), min_prob, 1.0 - min_prob)


def test_team_model_values():
    # logit(home) = (r(x) - r(swap(x))) / 2: the baseline cancels, so no prior is ever corrected for
    with tempfile.TemporaryDirectory() as root:
        for baseline in (0.7, -3.0, 0.0):
            path = write_team_model(root, full=team_model_stub(baseline, -1.0, 2.0))
            for seed in (0, 1, 3, 5):
                players, ball, teams, graded = team_scene(seed)
                inputs = team_inputs(players, ball, teams, graded)
                home = team_home(inputs, path)
                X = team_matrix(players, ball, teams, graded)
                known = ~np.isnan(home)
                assert known.sum() > 20 and (np.abs(home[known] - 0.5) > 0.2).any()
                assert np.allclose(home[known], stump_team_home(X, -1.0, 2.0)[known], rtol=0, atol=1e-12), \
                    (baseline, seed)
        # a model that gives every row the same leaf has no opinion to give: exactly 0.5, whatever its baseline
        flat = write_team_model(root, full=team_model_stub(2.5, 0.8, 0.8), name="flat.json")
        home = team_home(team_inputs(*team_scene(0)), flat)
        assert (home[~np.isnan(home)] == 0.5).all() and (~np.isnan(home)).any()


def test_team_model_is_antisymmetric():
    from pipeline.evidence import team_features, trees
    with tempfile.TemporaryDirectory() as root:
        # none of the trees is symmetric: one reads the home block only, one the away block only, two others a team
        # column and a shape column that swap or change sign (a shared column would cancel by construction)
        lopsided = {"baseline": 0.9, "prior_home": 0.5, "trees": [
            stump(TEAM_COLUMN["h_ball_d1"], 3.0, -0.8, 1.7, missing_left=1),
            stump(TEAM_COLUMN["a_reach_s"], 1.0, 0.6, -0.3),
            stump(TEAM_COLUMN["h_until_episode_s"], 0.5, 0.7, -0.4),
            stump(TEAM_COLUMN["adv_gap"], 0.0, -0.5, 1.5),
            stump(TEAM_COLUMN["h_ball_vel_adv"], 0.0, -0.6, 0.9),            # version 2's direction of the ball
            stump(TEAM_COLUMN["a_adv_fut_2s"], 0.0, 0.4, -0.2)]}              # and P30's one-sided advance
        path = write_team_model(root, full=lopsided)
        for seed in range(8):
            players, ball, teams, graded = team_scene(seed)
            home = team_home(team_inputs(players, ball, teams, graded), path)
            relabelled = team_home(team_inputs(players, ball, 1 - teams, graded), path)
            known = ~np.isnan(home)
            assert known.sum() > 20 and np.array_equal(known, ~np.isnan(relabelled)), seed
            assert np.allclose(home[known], 1.0 - relabelled[known], rtol=0, atol=1e-12), \
                "seed {}: relabelling the teams is not 1 - home".format(seed)
            # the check has teeth: the trees alone, run on the two labellings, are not antisymmetric
            X = team_matrix(players, ball, teams, graded)[known]
            alone = 1.0 / (1.0 + np.exp(-trees.score(lopsided, X)))
            alone_relabelled = 1.0 / (1.0 + np.exp(-trees.score(lopsided, team_features.swap(X))))
            assert np.abs(alone - (1.0 - alone_relabelled)).max() > 0.05, "the model is symmetric on its own"
        # on random rows the logit is exactly odd under swap
        rng = np.random.default_rng(21)
        rows = rng.normal(size=(400, len(TEAM_NAMES))) * 5.0
        rows[rng.random(rows.shape) < 0.2] = np.nan
        both = (trees.score(lopsided, rows) - trees.score(lopsided, team_features.swap(rows))) / 2.0
        flipped = (trees.score(lopsided, team_features.swap(rows)) - trees.score(lopsided, rows)) / 2.0
        assert np.allclose(both, -flipped, rtol=0, atol=1e-12)


def test_team_model_silent_where_the_ball_is_missing():
    with tempfile.TemporaryDirectory() as root:
        path = write_team_model(root)
        for seed in (0, 3):
            players, ball, teams, graded = team_scene(seed)
            home = team_home(team_inputs(players, ball, teams, graded), path)
            inside = block_mask(len(home), graded)
            seen = inside & ~np.isnan(ball[:, 0])
            assert seen.sum() > 50 and (inside & np.isnan(ball[:, 0])).sum() > 5, "the scene needs both kinds"
            assert np.isnan(home[~seen]).all(), "no opinion where the ball is missing or outside the blocks"
            assert not np.isnan(home[seen]).any(), "an opinion on every graded frame with the ball"


def test_team_model_fold_choice():
    with tempfile.TemporaryDirectory() as root:
        folds = {"FOLD1": team_model_stub(0.0, 1.0, -1.0), "FOLD2": team_model_stub(0.0, 3.0, -3.0)}
        path = write_team_model(root, folds=folds, full=team_model_stub(0.0, -1.0, 1.0))
        players, ball, teams, graded = team_scene(0)
        X = team_matrix(players, ball, teams, graded)
        inputs = team_inputs(players, ball, teams, graded)
        values = dict(min_prob=0.0001)
        full = team_home(inputs, path, **values)
        known = ~np.isnan(full)
        assert np.allclose(full[known], stump_team_home(X, -1.0, 1.0, min_prob=0.0001)[known], rtol=0, atol=1e-12)
        for game, left, right in (("FOLD1", 1.0, -1.0), ("FOLD2", 3.0, -3.0)):
            fold = team_home(inputs, path, game=game, **values)
            assert np.array_equal(known, ~np.isnan(fold))
            assert np.allclose(fold[known], stump_team_home(X, left, right, min_prob=0.0001)[known], rtol=0,
                               atol=1e-12), "{} is the fold of its own game".format(game)
            assert not np.allclose(fold[known], full[known]), "the folds must differ from `full`"
        for game in ("BAR-ATH", "NOT-IN-FOLDS", "full"):
            other = team_home(inputs, path, game=game, **values)
            assert np.array_equal(other, full, equal_nan=True), "{}: a game with no fold gets `full`".format(game)
        # the file is read again when it changes
        write_team_model(root, folds=folds, full=team_model_stub(0.0, 5.0, -5.0))
        changed = team_home(inputs, path, **values)
        assert np.allclose(changed[known], stump_team_home(X, 5.0, -5.0, min_prob=0.0001)[known], rtol=0, atol=1e-12)


def test_team_model_rejects_bad_artifacts():
    with tempfile.TemporaryDirectory() as root:
        inputs = team_inputs(*team_scene(0))
        renamed = list(TEAM_NAMES)
        renamed[3] = "h_rel_speed_x"
        good = team_model_stub(0.0, -1.0, 1.0)
        # version 1's 56 names (P19's, at 15e7e14): the same blocks with over11, the two gap columns and P07's 26
        version_1 = (tuple("{}{}".format(prefix, name) for prefix in ("h_", "a_") for name in (
            "ball_d1", "ball_d2", "closing", "rel_speed", "reach_s", "over11", "in_episode", "since_episode_s",
            "until_episode_s")) + TEAM_SHARED + ("since_gap_s", "until_gap_s") + SHAPE_NAMES)
        assert len(version_1) == 56

        def refused(path, *fragments):
            return expect_error(ValueError, lambda: team_home(inputs, path), "team_model", TEAM_TRAINER, *fragments)

        for label, changes in (("format", {"format": 2}), ("format", {"format": None}),
                               ("feature_version", {"feature_version": 1}),
                               ("feature_version", {"feature_version": 3}),
                               ("feature_version", {"feature_version": 0}),
                               ("feature_version", {"feature_version": None}),
                               ("feature_version", {"feature_version": True}),
                               ("feature_version", {"feature_version": 2.0}),
                               ("feature_version", {"feature_version": "2"}),
                               ("feature_version", {"feature_version": 1, "features": list(version_1)}),
                               ("features", {"features": list(TEAM_NAMES)[::-1]}), ("features", {"features": renamed}),
                               ("features", {"features": list(TEAM_NAMES)[:-1]}),
                               ("features", {"features": list(version_1)}),
                               ("features", {"features": list(SHAPE_NAMES)}),
                               ("features", {"features": list(TEAM_SHAPE)}),
                               ("kind", {"kind": "logistic"}), ("kind", {"kind": None}),
                               ("folds", {"folds": [good]}), ("folds", {"folds": "FOLD1"})):
            refused(write_team_model(root, name="other.json", **changes), label)
        refused(write_text(root, "list.json", "[1, 2]"), "model object")
        for bad in (0.6, 0.0, 1.0, -0.5, True, "0.5", None):
            refused(write_team_model(root, full=dict(good, prior_home=bad), name="prior.json"), "prior_home", "full")
            refused(write_team_model(root, folds={"FOLD1": dict(good, prior_home=bad)}, name="prior.json"),
                    "prior_home", "FOLD1")
        ragged = team_model_stub(0.0, -1.0, 1.0)
        ragged["trees"][0]["left"] = [1, 0]                      # arrays of unequal lengths
        refused(write_team_model(root, full=ragged, name="ragged.json"), "tree 0", "full")
        wide = team_model_stub(0.0, -1.0, 1.0)
        wide["trees"][0]["feature"][0] = len(TEAM_NAMES)         # a column the features do not have
        refused(write_team_model(root, folds={"FOLD1": wide}, name="wide.json"), "tree 0", "FOLD1")
        refused(write_team_model(root, full=dict(good, baseline=None), name="nobase.json"), "baseline")
        refused(os.path.join(root, "absent.json"), "absent.json")
        # P30's file is a model of other features, of the same version number: it must not load here
        refused(write_shape_model(root, name="shape.json", version=2), "features")
        refused(write_shape_model(root, name="shape1.json"), "feature_version")


def test_team_model_parameters():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    with tempfile.TemporaryDirectory() as root:
        path = write_team_model(root)
        inputs = team_inputs(*team_scene(3))
        for bad in (0.0, 0.5, 0.6, -0.1, True, False, "0.02", None, float("nan"), float("inf")):
            expect_error(ValueError, lambda: team_home(inputs, path, min_prob=bad), "team_model", "min_prob")
        for bad in (None, 5, "", True):
            expect_error(ValueError, lambda: team_home(inputs, bad), "team_model", "model")
        expect_error(TypeError, lambda: REGISTRY["team_model"](RunContext(inputs=inputs, seed=0)), "model")
        expect_error(TypeError, lambda: team_home(inputs, path, scope="all"), "scope")
        team_home(inputs, path, min_prob=np.float64(0.25))                      # numpy numbers from a sweep are fine
        # the default is the design's 0.02, and the clip holds a score far out at min_prob and 1 - min_prob
        assert np.array_equal(team_home(inputs, path), team_home(inputs, path, min_prob=0.02), equal_nan=True)
        extreme = write_team_model(root, full=team_model_stub(0.0, -40.0, 40.0), name="extreme.json")
        X = team_matrix(*team_scene(3))
        for min_prob in (0.02, 0.1, 0.4):
            home = team_home(inputs, extreme, min_prob=min_prob)
            known = ~np.isnan(home)
            assert known.any() and home[known].min() >= min_prob and home[known].max() <= 1.0 - min_prob
            assert np.allclose(home[known], stump_team_home(X, -40.0, 40.0, min_prob=min_prob)[known], rtol=0,
                               atol=1e-12)
        assert home.dtype == np.float64 and home.shape == (len(X),)


def test_team_model_remembers_its_scores_per_inputs():
    from pipeline.evidence import trees
    with tempfile.TemporaryDirectory() as root:
        path = write_team_model(root)
        inputs = team_inputs(*team_scene(0))
        calls = []
        real = trees.score
        trees.score = lambda model, X: calls.append(len(X)) or real(model, X)
        try:
            first = team_home(inputs, path)
            walked = len(calls)
            assert walked >= 1
            second = team_home(inputs, path)
            assert len(calls) == walked and np.array_equal(first, second, equal_nan=True), "walked the trees again"
            clipped = team_home(inputs, path, min_prob=0.3)                         # another clip: no walk
            assert len(calls) == walked
            assert np.nanmin(clipped) >= 0.3 and np.nanmax(clipped) <= 0.7, "a different min_prob must re-clip"
            team_home(inputs, path, game="FOLD1")                                    # another model of the file
            assert len(calls) == 2 * walked
            team_home(inputs, path, game="FOLD1")
            assert len(calls) == 2 * walked
            team_home(team_inputs(*team_scene(0)), path)
            assert len(calls) == 3 * walked, "the memo is per Inputs object"
        finally:
            trees.score = real


def test_team_model_refuses_infinite_features():
    from pipeline.evidence import team_features
    with tempfile.TemporaryDirectory() as root:
        path = write_team_model(root)
        inputs = team_inputs(*team_scene(0))
        names, X = team_features.features(inputs)
        broken = X.copy()
        seen = np.flatnonzero(~np.isnan(X[:, TEAM_COLUMN["ball_line_m"]]))
        broken[seen[3], TEAM_COLUMN["ball_line_m"]] = np.inf
        real = team_features.features
        team_features.features = lambda given: (names, broken)
        try:
            expect_error(ValueError, lambda: team_home(inputs, path), "team_model", "infinite")
        finally:
            team_features.features = real


def test_team_model_speed():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY, team_features
    rng = np.random.default_rng(22)
    n, slots = 140_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 70_000), (2, 75_000, 135_000)])
    inputs.game = "BAR-ATH"
    inputs.atd.players_present                      # the Atd's own cached masks are not this module's cost
    inputs.atd.ball_present
    model = random_hgb_model(rng, prior=0.5)        # of the size the trainer writes, over the 72 team columns
    model.pop("prior_alive")
    model["prior_home"] = 0.5
    for tree in model["trees"]:
        tree["feature"] = [feature % len(TEAM_NAMES) if feature >= 0 else feature for feature in tree["feature"]]
    with tempfile.TemporaryDirectory() as root:
        path = write_team_model(root, full=model)
        started = time.perf_counter()
        team_features.features(inputs)              # warm: the features have their own check
        features_s = time.perf_counter() - started
        started = time.perf_counter()
        evidence = REGISTRY["team_model"](RunContext(inputs=inputs, seed=0), model=path)
        first_s = time.perf_counter() - started
        started = time.perf_counter()
        REGISTRY["team_model"](RunContext(inputs=inputs, seed=0), model=path, min_prob=0.01)
        again_s = time.perf_counter() - started
        print("      team_model speed: 125,000 graded frames, 300 trees of 15 leaves, two walks over the ball-visible "
              "frames: first inference {:.2f} s (features {:.2f} s before it), a second config on the same inputs "
              "{:.2f} s".format(first_s, features_s, again_s))
        assert np.isnan(evidence.home[:5_000]).all() and not np.isnan(evidence.home[5_000:70_000]).all()
        assert first_s < 18.0, "{:.2f} s for 300 trees, twice, on the ball-visible frames (the budget is 18 s)".format(
            first_s)


SHAPE_TRIAL_CONFIG = """
[[evidence]]
kind = "nearest_player"

[[evidence]]
kind = "shape_possession"
model = {model}
scope = "all"
min_prob = 0.1

[decoder]
kind = "hmm"
switch_team = 5
switch_dead = 20
min_possession = 1
min_dead = 1
"""


def test_run_end_to_end_synthetic_shape_possession():
    """Config -> registry -> evidence -> hmm -> CSV: the source is found by its kind and its parameters are checked."""
    from pipeline import output
    from pipeline.config import ConfigError, load_config
    from pipeline.contracts import X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        text = SHAPE_TRIAL_CONFIG.format(model=json_string(write_shape_model(root)))
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        out_path = os.path.join(root, "shape.csv")
        result = run(load_config(write_text(root, "shape.toml", text)), inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all() and (result.codes[~outside] != X).all()
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        source = result.evidence["shape_possession"]
        assert np.isnan(source.alive).all() and np.isnan(source.home[outside]).all()
        assert (~np.isnan(source.home[~outside])).mean() > 0.5, "the source speaks on most graded frames"
        assert np.nanmin(source.home) >= 0.1 and np.nanmax(source.home) <= 0.9
        for old, new in (("scope = \"all\"", "scope = \"everywhere\""), ("min_prob = 0.1", "min_prob = 0.7")):
            bad = write_text(root, "bad.toml", text.replace(old, new))
            expect_error(ValueError, lambda: run(load_config(bad), inputs, os.path.join(root, "bad.csv"), log=quiet),
                         "shape_possession")
        unknown = write_text(root, "unknown.toml", text.replace("min_prob = 0.1", "min_prob = 0.1\nradius = 2"))
        expect_error(ConfigError, lambda: load_config(unknown), "shape_possession", "radius")


# -- the dead-ball model (P05) ----------------------------------------------------------------------------------------
#
# A hand-made artifact of a couple of trees, so that every number is worked out here and not by the trainer. Its one
# tree splits on ball_missing (ball seen: one leaf, ball missing: another) on a baseline; the fold's tree has other
# numbers, so that a check can tell which of the two models was used.

MODEL_BASELINE, MODEL_SEEN, MODEL_MISSING, MODEL_PRIOR = 0.3, 1.0, -2.0, 0.6
FOLD_BASELINE, FOLD_SEEN, FOLD_MISSING, FOLD_PRIOR = -0.4, 0.5, -3.0, 0.25


def stump(feature, threshold, left, right, missing_left=0):
    """One tree of three nodes, in the artifact's per-tree arrays: a split on `feature` and two leaves."""
    return {"feature": [feature, -1, -1], "threshold": [threshold, 0.0, 0.0], "missing_left": [missing_left, 0, 0],
            "left": [1, 0, 0], "right": [2, 0, 0], "leaf": [0, 1, 1], "value": [0.0, left, right]}


def hgb_model(baseline, seen, missing, prior):
    return {"prior_alive": prior, "baseline": baseline,
            "trees": [stump(PLAY_COLUMN["ball_missing"], 0.5, seen, missing)]}


def write_model(folder, folds=None, full=None, name="dead_ball_hgb.json", **changes):
    import json
    from pipeline.evidence import play_features
    artifact = {"format": 1, "kind": "hgb", "feature_version": play_features.FEATURE_VERSION,
                "features": list(PLAY_NAMES), "profile": "tracking", "seeds": 3, "stride": 5, "provenance": {},
                "folds": {"FOLD1": hgb_model(FOLD_BASELINE, FOLD_SEEN, FOLD_MISSING, FOLD_PRIOR)} if folds is None
                else folds,
                "full": hgb_model(MODEL_BASELINE, MODEL_SEEN, MODEL_MISSING, MODEL_PRIOR) if full is None else full}
    artifact.update(changes)
    return write_text(folder, name, json.dumps(artifact, sort_keys=True) + "\n")


def model_alive(inputs, path, game="BAR-ATH", **values):
    from pipeline.contracts import RunContext, check_evidence
    from pipeline.evidence import REGISTRY
    inputs.game = game
    evidence = REGISTRY["dead_ball_model"](RunContext(inputs=inputs, seed=0), model=path, **values)
    check_evidence(evidence, len(inputs.atd.ball_xy))
    assert np.isnan(evidence.home).all(), "dead_ball_model has no opinion on who has the ball"
    return evidence.alive


def half_seen_inputs(n=100, **kwargs):
    """Walkers and a ball that is seen on frames 0-49 and missing on 50-99 (one block)."""
    ball = np.full((n, 2), np.nan)
    ball[:n // 2] = [30.0, 20.0]
    walking = walkers(n, 1.0, np.stack([np.linspace(10, 90, 8), np.linspace(10, 60, 8)], axis=1))
    return play_inputs(walking, ball, **kwargs)


def logit(p):
    return np.log(p / (1.0 - p))


def test_dead_ball_model_prior_correction():
    # logit(alive) = logit(p) - logit(prior_alive), with p = sigmoid(baseline + leaf) for the full model
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        inputs = half_seen_inputs()
        alive = model_alive(inputs, path)
        for rows, leaf in ((slice(0, 50), MODEL_SEEN), (slice(50, 100), MODEL_MISSING)):
            expected = 1.0 / (1.0 + np.exp(-(MODEL_BASELINE + leaf - logit(MODEL_PRIOR))))
            assert np.allclose(alive[rows], expected, rtol=0, atol=1e-12), (alive[rows][0], expected)
        # the clip: a score far out is held at min_prob and 1 - min_prob, and min_prob is a parameter
        extreme = write_model(root, full=hgb_model(0.0, 40.0, -40.0, 0.5), name="extreme.json")
        alive = model_alive(inputs, extreme)
        assert np.allclose(alive[:50], 0.999) and np.allclose(alive[50:], 0.001)
        alive = model_alive(inputs, extreme, min_prob=0.01)
        assert np.allclose(alive[:50], 0.99) and np.allclose(alive[50:], 0.01)
        # a prior of 0.5 corrects nothing
        plain = write_model(root, full=hgb_model(0.0, 1.0, -1.0, 0.5), name="plain.json")
        alive = model_alive(inputs, plain)
        assert np.allclose(alive[:50], 1 / (1 + np.exp(-1.0))) and np.allclose(alive[50:], 1 / (1 + np.exp(1.0)))
        for bad in (0.0, 0.5, -0.1, True, "0.1", None, float("nan")):
            expect_error(ValueError, lambda: model_alive(inputs, path, min_prob=bad), "dead_ball_model", "min_prob")
        for bad in (0.0, 1.0, 1.5, True, None):
            broken = write_model(root, full=dict(hgb_model(0.0, 1.0, -1.0, 0.5), prior_alive=bad), name="prior.json")
            expect_error(ValueError, lambda: model_alive(inputs, broken), "dead_ball_model", "prior_alive")


def test_dead_ball_model_uses_the_fold_of_the_game():
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        inputs = half_seen_inputs()
        fold = model_alive(inputs, path, game="FOLD1")
        for rows, leaf in ((slice(0, 50), FOLD_SEEN), (slice(50, 100), FOLD_MISSING)):
            expected = 1.0 / (1.0 + np.exp(-(FOLD_BASELINE + leaf - logit(FOLD_PRIOR))))
            assert np.allclose(fold[rows], expected, rtol=0, atol=1e-12)
        full = model_alive(inputs, path, game="BAR-ATH")
        assert not np.allclose(full, fold)
        other = model_alive(inputs, path, game="NOT-IN-FOLDS")
        assert np.array_equal(other, full), "a game with no fold is scored by `full`"
        # the file is read again when it changes
        write_model(root, full=hgb_model(5.0, 0.0, 0.0, 0.5))
        changed = model_alive(inputs, path, game="BAR-ATH")
        assert np.allclose(changed, 1.0 / (1.0 + np.exp(-5.0))) and not np.allclose(changed, full)


def test_dead_ball_model_refuses_other_features():
    import json
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY
    with tempfile.TemporaryDirectory() as root:
        inputs = half_seen_inputs()
        renamed = list(PLAY_NAMES)
        renamed[3] = "ball_still_ms"
        for label, changes in (("format", {"format": 2}), ("feature_version", {"feature_version": 2}),
                               ("features", {"features": list(PLAY_NAMES)[::-1]}), ("features", {"features": renamed}),
                               ("features", {"features": list(PLAY_NAMES)[:-1]})):
            path = write_model(root, name="other.json", **changes)
            expect_error(ValueError, lambda: model_alive(inputs, path), "dead_ball_model", label,
                         "pipeline/benchmark/train_dead_ball.py")
        expect_error(ValueError, lambda: model_alive(inputs, write_model(root, name="kind.json", kind="forest")),
                     "dead_ball_model", "kind", "forest")
        expect_error(ValueError, lambda: model_alive(inputs, os.path.join(root, "absent.json")),
                     "dead_ball_model", "absent.json", "pipeline/benchmark/train_dead_ball.py")
        write_text(root, "text.json", "not json at all")
        expect_error(ValueError, lambda: model_alive(inputs, os.path.join(root, "text.json")), "dead_ball_model",
                     "text.json")
        ragged = hgb_model(0.0, 1.0, -1.0, 0.5)
        ragged["trees"][0]["left"] = [1, 0]                    # arrays of unequal lengths
        expect_error(ValueError, lambda: model_alive(inputs, write_model(root, full=ragged, name="ragged.json")),
                     "dead_ball_model", "tree")
        no_full = json.loads(open(write_model(root), encoding="utf-8").read())
        no_full.pop("full")
        write_text(root, "nofull.json", json.dumps(no_full))
        expect_error(ValueError, lambda: model_alive(inputs, os.path.join(root, "nofull.json")), "dead_ball_model",
                     "full")
        for bad in (None, 5, "", True):
            expect_error(ValueError, lambda: model_alive(inputs, bad), "dead_ball_model", "model")
        expect_error(TypeError, lambda: REGISTRY["dead_ball_model"](RunContext(inputs=inputs, seed=0)), "model")


def test_dead_ball_model_pre_kickoff_is_dead():
    from pipeline.evidence import play_features
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        n = 200
        walking = walkers(n, 1.0, np.stack([np.linspace(10, 90, 8), np.linspace(10, 60, 8)], axis=1))
        ball = np.tile([30.0, 20.0], (n, 1))
        inputs = play_inputs(walking, ball, graded=[(1, 0, 100), (2, 100, 200)], kickoffs={1: 40, 2: 150})
        alive = model_alive(inputs, path)
        assert np.allclose(alive[:40], play_features.PRE_KICKOFF_ALIVE)
        assert np.allclose(alive[100:150], play_features.PRE_KICKOFF_ALIVE)
        expected = 1.0 / (1.0 + np.exp(-(MODEL_BASELINE + MODEL_SEEN - logit(MODEL_PRIOR))))
        assert np.allclose(alive[40:100], expected) and np.allclose(alive[150:], expected)
        # outside the blocks: no opinion. And a kick-off outside its block is silent, as in the rules
        inputs = play_inputs(walking, ball, graded=[(1, 20, 80), (2, 120, 180)], kickoffs={1: 500, 2: 0})
        alive = model_alive(inputs, path)
        assert np.isnan(alive[:20]).all() and np.isnan(alive[80:120]).all() and np.isnan(alive[180:]).all()
        assert np.allclose(alive[20:80], expected) and np.allclose(alive[120:180], expected)


DEAD_BALL_TRIAL_CONFIG = """
[[evidence]]
kind = "controlled_touch"
radius = 200.0
duel_radius = 0.001
max_rel_speed = inf
min_frames = 1
confidence = 0.9
hold = 0.6

[[evidence]]
{source}

[decoder]
kind = "hmm"
switch_team = 1
switch_dead = 20
min_possession = 1
min_dead = 1
"""


def test_run_end_to_end_synthetic_dead_ball():
    """Each P05 source, through config -> registry -> evidence -> hmm: the frames before a kick-off come out D."""
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import CODES, X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        model = write_model(root)
        lines = ["kind = \"dead_ball_rules\""] + ["{} = {}".format(k, v) for k, v in RULE_VALUES.items()]
        sources = {"dead_ball_rules": "\n".join(lines),
                   "dead_ball_model": "kind = \"dead_ball_model\"\nmodel = {}".format(json_string(model))}
        for name, source in sources.items():
            inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
            inputs.kickoffs = {1: 100, 2: 420}
            config = load_config(write_text(root, name + ".toml", DEAD_BALL_TRIAL_CONFIG.format(source=source)))
            out_path = os.path.join(root, name + ".csv")
            result = run(config, inputs, out_path, log=quiet)
            before = np.zeros(SYNTHETIC_FRAMES, dtype=bool)
            before[50:100] = before[320:420] = True
            outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
            outside[50:280] = outside[320:560] = False
            assert (result.codes[before] == CODES["D"]).all(), "{}: frames before a kick-off are not D".format(name)
            assert (result.codes[outside] == X).all(), name
            assert (result.codes[~outside] != X).all(), "{}: no X inside a block".format(name)
            output.validate(out_path, SYNTHETIC_FRAMES, result.codes)


def json_string(text):
    import json
    return json.dumps(text)


def test_dead_ball_model_imports_numpy_only():
    import ast
    import subprocess
    banned = {"sklearn", "scipy", "joblib", "threadpoolctl", "pandas", "narwhals", "cloudpickle"}
    folder = os.path.join(ROOT, "pipeline", "evidence")
    found = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".py"):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                for node in ast.walk(ast.parse(handle.read())):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level:
                        modules = [node.module or ""]
                    else:
                        modules = []
                    found += ["{}: {}".format(name, module) for module in modules if module.split(".")[0] in banned]
    assert not found, "pipeline/evidence must be numpy only: {}".format(found)
    code = ("import sys; import pipeline.evidence.dead_ball_model, pipeline.evidence; "
            "bad = sorted(name for name in ('sklearn', 'scipy', 'joblib') if name in sys.modules); "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, "importing the model source pulled in: " + done.stdout + done.stderr


def test_dead_ball_model_read_only_inputs():
    from pipeline.evidence import play_features
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        inputs = play_random_inputs(5)
        arrays = [inputs.atd.players_xy, inputs.atd.ball_xy, inputs.atd.slot_team]
        saved = [array.copy() for array in arrays]
        for array in arrays:
            array.flags.writeable = False
        before = play_features.features(inputs)[1].copy()
        first = model_alive(inputs, path)
        for array, copy in zip(arrays, saved):
            assert np.array_equal(array, copy, equal_nan=True)
        assert np.array_equal(play_features.features(inputs)[1], before, equal_nan=True), "the features were changed"
        second = model_alive(inputs, path)
        assert np.array_equal(first, second, equal_nan=True), "a second call on the same inputs gives other numbers"
        assert not first.flags.writeable, "the arrays handed out are read-only"


def random_hgb_model(rng, trees=300, leaves=15, prior=0.7):
    """A model of the size the trainer writes: `trees` random trees of `leaves` leaves over random features."""
    forest = []
    for _ in range(trees):
        feature, threshold, left, right = [-1], [0.0], [0], [0]
        open_leaves = [0]
        for _ in range(leaves - 1):                          # split a random open leaf in two
            node = open_leaves.pop(int(rng.integers(len(open_leaves))))
            child = len(feature)
            feature[node], threshold[node] = int(rng.integers(30)), float(rng.normal())
            feature += [-1, -1]
            threshold += [0.0, 0.0]
            left += [0, 0]
            right += [0, 0]
            left[node], right[node] = child, child + 1
            open_leaves += [child, child + 1]
        missing = rng.integers(2, size=len(feature)).tolist()
        forest.append({"feature": feature, "threshold": threshold, "missing_left": missing,
                       "left": left, "right": right, "leaf": [int(f < 0) for f in feature],
                       "value": [float(rng.normal()) * 0.1 if f < 0 else 0.0 for f in feature]})
    return {"prior_alive": prior, "baseline": 0.2, "trees": forest}


def test_dead_ball_model_speed():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY, play_features
    rng = np.random.default_rng(11)
    n, slots = 180_000, 46
    players = rng.random((n, slots, 2))
    players[rng.random((n, slots)) < 0.4] = np.nan
    ball = rng.random((n, 2))
    ball[rng.random(n) < 0.4] = np.nan
    inputs = kinematics_inputs(kinematics_atd(players, ball), [(1, 5_000, 75_000), (2, 100_000, 170_000)])
    inputs.kickoffs, inputs.game = {}, "BAR-ATH"
    inputs.atd.players_present                      # the Atd's own cached masks are not this module's cost
    inputs.atd.ball_present
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root, full=random_hgb_model(rng))
        started = time.perf_counter()
        play_features.features(inputs)              # warm: the features are P05's other half and have their own check
        features_s = time.perf_counter() - started
        started = time.perf_counter()
        evidence = REGISTRY["dead_ball_model"](RunContext(inputs=inputs, seed=0), model=path)
        first_s = time.perf_counter() - started
        started = time.perf_counter()
        REGISTRY["dead_ball_model"](RunContext(inputs=inputs, seed=0), model=path, min_prob=0.01)
        again_s = time.perf_counter() - started
        print("      dead_ball_model speed: 140,000 graded frames, 300 trees of 15 leaves: first inference {:.2f} s "
              "(features {:.2f} s before it, {:.2f} s in all), a second config on the same inputs {:.2f} s".format(
                  first_s, features_s, features_s + first_s, again_s))
        assert not np.isnan(evidence.alive[5_000:75_000]).any() and np.isnan(evidence.alive[:5_000]).all()
        assert first_s < 3.0, "{:.2f} s for 300 trees on 140,000 frames with warm features (the budget is 3 s)".format(
            first_s)


def test_dead_ball_model_remembers_its_scores_per_inputs():
    from pipeline.contracts import RunContext
    from pipeline.evidence import REGISTRY, dead_ball_model
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        inputs = half_seen_inputs()
        inputs.game = "BAR-ATH"
        calls = []
        real = dead_ball_model.raw_score
        dead_ball_model.raw_score = lambda model, X: calls.append(len(X)) or real(model, X)
        try:
            run = RunContext(inputs=inputs, seed=0)
            first = REGISTRY["dead_ball_model"](run, model=path).alive
            second = REGISTRY["dead_ball_model"](run, model=path).alive
            assert len(calls) == 1, "a second call on the same Inputs walked the trees again"
            assert np.array_equal(first, second) and not first.flags.writeable and not second.flags.writeable
            clipped = REGISTRY["dead_ball_model"](run, model=path, min_prob=0.4).alive
            assert len(calls) == 1 and clipped.min() >= 0.4 and clipped.max() <= 0.6
            assert np.allclose(clipped, np.clip(first, 0.4, 0.6)), "a different min_prob must re-clip the cached score"
            inputs.game = "FOLD1"                                         # another model of the file: walked afresh
            REGISTRY["dead_ball_model"](run, model=path)
            assert len(calls) == 2
            other = half_seen_inputs()
            other.game = "BAR-ATH"
            REGISTRY["dead_ball_model"](RunContext(inputs=other, seed=0), model=path)
            assert len(calls) == 3, "the memo is per Inputs object"
        finally:
            dead_ball_model.raw_score = real


def test_dead_ball_model_refuses_infinite_features():
    from pipeline.evidence import play_features
    with tempfile.TemporaryDirectory() as root:
        path = write_model(root)
        inputs = half_seen_inputs()
        inputs.atd.ball_xy[10] = [np.inf, 20.0 / 68.0]
        assert np.isinf(play_features.features(inputs)[1]).any(), "the scene no longer makes an infinite feature"
        expect_error(ValueError, lambda: model_alive(inputs, path), "dead_ball_model", "infinite")


# -- the shared tree walker (P07) -------------------------------------------------------------------------------------

def test_trees_score_matches_a_reference_walk():
    # The walk is written for speed (whole-array byte masks, bottom up), so it is held against the plain definition on
    # the committed model: follow `left` where x <= threshold, or where x is NaN and `missing_left`; sum the leaves.
    import json
    from pipeline.evidence import play_features, trees
    with open(os.path.join(ROOT, "pipeline", "models", "dead_ball_hgb.json"), encoding="utf-8") as handle:
        model = json.load(handle)["full"]
    n_features = len(play_features.FEATURE_NAMES)
    splits = {}                                           # feature -> the finite thresholds the trees split it on
    for tree in model["trees"]:
        for feature, threshold, leaf in zip(tree["feature"], tree["threshold"], tree["leaf"]):
            if not leaf and threshold < 1e300:
                splits.setdefault(feature, []).append(threshold)
    low = np.array([min(splits.get(feature, [0.0])) for feature in range(n_features)])
    high = np.array([max(splits.get(feature, [1.0])) for feature in range(n_features)])
    rng = np.random.default_rng([7, 1])
    # each column runs a little past its first and last split, so rows fall on both sides of most of them
    X = (low - 0.1 * (high - low)) + 1.2 * (high - low) * rng.random((300, n_features))
    for number, tree in enumerate(model["trees"][:50]):          # a value that sits exactly on a threshold goes left
        X[number, tree["feature"][0]] = tree["threshold"][0]
    X[rng.random(X.shape) < 0.2] = np.nan                        # a fifth of the values are missing

    def reference(row):                                   # plain Python floats: the same arithmetic, much faster
        total = float(model["baseline"])
        for tree in model["trees"]:
            node = 0
            while not tree["leaf"][node]:
                x = row[tree["feature"][node]]
                goes_left = x <= tree["threshold"][node] or (x != x and tree["missing_left"][node])    # x != x: NaN
                node = tree["left"][node] if goes_left else tree["right"][node]
            total += tree["value"][node]
        return total

    expected = np.array([reference(row) for row in X.tolist()])
    assert len(np.unique(expected)) > 100, "the rows no longer reach different leaves"
    assert np.isnan(X).any() and (~np.isnan(X)).any()
    got = trees.score(model, X)
    assert got.shape == (300,) and got.dtype == np.float64
    assert np.array_equal(got, expected), "trees.score differs from the plain walk by up to {:.3g}".format(
        float(np.abs(got - expected).max()))
    saved = trees.CHUNK
    try:
        for chunk in (64, 150, 299, 300):                          # the chunks only bound the temporaries
            trees.CHUNK = chunk
            assert np.array_equal(trees.score(model, X), expected), "CHUNK = {} changes the scores".format(chunk)
    finally:
        trees.CHUNK = saved
    assert trees.score(model, X[:0]).shape == (0,), "no rows give no scores"


def test_trees_check_names_its_caller():
    # check_trees moved out of dead_ball_model; its messages start with the caller's name, so dead_ball_model's stay
    # word for word and another learned source can use it under its own.
    from pipeline.evidence import trees
    good = hgb_model(MODEL_BASELINE, MODEL_SEEN, MODEL_MISSING, MODEL_PRIOR)
    assert trees.check_trees(good, len(PLAY_NAMES), "full of m.json", "shape_possession") is None

    def broken(**changes):
        tree = dict(good["trees"][0], **changes)
        return dict(good, trees=[tree])

    too_many = {key: list(value) + [0] * 255 for key, value in good["trees"][0].items()}
    for label, model, fragments in (
            ("baseline", dict(good, baseline="0.3"), ("model full of m.json", "baseline")),
            ("trees", dict(good, trees=None), ("model full of m.json", "trees")),
            ("ragged", broken(left=[1, 0]), ("tree 0 of full of m.json", "seven node arrays")),
            ("missing key", dict(good, trees=[{"feature": [0]}]), ("tree 0 of full of m.json", "seven node arrays")),
            ("nodes", dict(good, trees=[too_many]), ("tree 0 of full of m.json", "258 nodes", "at most 255")),
            ("feature", broken(feature=[len(PLAY_NAMES), -1, -1]), ("tree 0 of full of m.json", "node 0")),
            ("child", broken(left=[0, 0, 0]), ("tree 0 of full of m.json", "node 0", "no later child"))):
        message = expect_error(ValueError, lambda: trees.check_trees(model, len(PLAY_NAMES), "full of m.json",
                                                                     "shape_possession"), *fragments)
        assert message.startswith("shape_possession: "), "{}: {}".format(label, message)
    # a feature count that does not reach the column of a split is refused
    expect_error(ValueError, lambda: trees.check_trees(good, PLAY_COLUMN["ball_missing"], "x", "shape_possession"),
                 "shape_possession", "node 0")


def test_trees_helpers_name_their_caller():
    # what the learned sources share besides the walk: each message starts with the caller's name, so a source's own
    # tests (dead_ball_model's, shape_possession's) check its wording and this checks that there is one
    import json
    import weakref
    from pipeline.evidence import trees
    who = "some_source"
    for bad in (None, 5, "", True):
        message = expect_error(ValueError, lambda: trees.model_path(bad, who, "pipeline/models/x.json"), "model",
                               "pipeline/models/x.json")
        assert message.startswith("some_source: "), message
    assert trees.model_path("pipeline/models/x.json", who, "y") == os.path.abspath(
        os.path.join(ROOT, "pipeline", "models", "x.json"))
    assert trees.model_path(os.path.abspath(__file__), who, "y") == os.path.abspath(__file__)
    for bad in (0.0, 0.5, 1.0, -0.1, True, "0.1", None, float("nan")):
        assert expect_error(ValueError, lambda: trees.check_min_prob(bad, who), "min_prob").startswith("some_source: ")
    value = trees.check_min_prob(np.float64(0.25), who)
    assert value == 0.25 and type(value) is float
    assert expect_error(ValueError, lambda: trees.check_finite(np.array([[0.0, np.inf]]), who, "G1", "some_features"),
                        "G1", "infinite", "some_features").startswith("some_source: ")
    assert trees.check_finite(np.array([[0.0, np.nan]]), who, "G1", "some_features") is None
    with np.errstate(over="raise", invalid="raise", divide="raise"):         # no overflow, however far out
        far = trees.sigmoid(np.array([-1e4, -40.0, 0.0, 40.0, 1e4]))
    assert far[0] == 0.0 and far[2] == 0.5 and far[4] == 1.0 and np.all(np.diff(far) >= 0)
    assert np.allclose(trees.sigmoid(np.array([-3.0, 2.0])), 1.0 / (1.0 + np.exp([3.0, -2.0])), rtol=0, atol=1e-15)

    good = hgb_model(MODEL_BASELINE, MODEL_SEEN, MODEL_MISSING, MODEL_PRIOR)
    artifact = {"format": 1, "kind": "hgb", "feature_version": 3, "features": ["a", "b"], "folds": {"G1": good},
                "full": good}
    checked = []

    def check(artifact, path, **changes):
        options = dict(source="some_features", version=3, names=("a", "b"), kinds=("hgb", "other"),
                       check_model=lambda model, where: checked.append(where))
        options.update(changes)
        trees.check_artifact(artifact, path, who, "retrain it, please", **options)

    check(artifact, "m.json")
    assert checked == ["G1 of m.json", "full of m.json"], checked
    for label, changed, options in (
            ("holds no model object", [1], {}), ("format", dict(artifact, format=2), {}),
            ("some_features is at 3", dict(artifact, feature_version=4), {}),
            ("features", dict(artifact, features=["b", "a"]), {}), ("some_features", artifact, {"names": ("a", "c")}),
            ("kind", dict(artifact, kind="forest"), {}), ("folds", dict(artifact, folds=[]), {}),
            ("full", {key: value for key, value in artifact.items() if key != "full"}, {})):
        message = expect_error(ValueError, lambda: check(changed, "m.json", **options), label, "m.json",
                               "retrain it, please")
        assert message.startswith("some_source: "), message

    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "m.json", json.dumps(artifact))
        cache, parsed = {}, []

        def load():
            return trees.load_artifact(path, who, lambda artifact, where: parsed.append(where), cache,
                                       unreadable="train it", not_json="retrain it")

        first = load()
        assert load() is first and parsed == [path], "a file that did not change is not read or checked again"
        write_text(root, "m.json", json.dumps(dict(artifact, seeds=3)))
        assert load() is not first and len(parsed) == 2
        message = expect_error(ValueError, lambda: trees.load_artifact(
            os.path.join(root, "absent.json"), who, None, cache, unreadable="train it", not_json="retrain it"),
            "absent.json", "train it")
        assert message.startswith("some_source: "), message
        write_text(root, "text.json", "not json")
        message = expect_error(ValueError, lambda: trees.load_artifact(
            os.path.join(root, "text.json"), who, None, cache, unreadable="train it", not_json="retrain it"),
            "text.json", "retrain it")
        assert message.startswith("some_source: "), message
        assert os.path.join(root, "text.json") not in cache, "a file that is refused is not remembered"

        class Holder:                                    # weakly referenceable, like an Inputs
            pass

        memo, built, holder = weakref.WeakKeyDictionary(), [], Holder()

        def build():
            built.append(1)
            return np.arange(3.0)

        array = trees.remember(memo, holder, path, "G1", build)
        assert not array.flags.writeable and trees.remember(memo, holder, path, "G1", build) is array
        assert len(built) == 1
        assert trees.remember(memo, holder, path, "full", build) is not array and len(built) == 2
        assert len(memo[holder]) == 1, "one model at a time per Inputs"
        write_text(root, "m.json", json.dumps(dict(artifact, seeds=44)))
        trees.remember(memo, holder, path, "full", build)
        assert len(built) == 3, "a file that changed is a new stamp"
        trees.remember(memo, Holder(), path, "full", build)
        assert len(built) == 4, "the memo is per object"
        del holder
        assert len(memo) == 0, "the memo dies with the Inputs"


# -- refine -----------------------------------------------------------------------------------------------------------

def test_config_refiner_unknown_kind_lists_the_registry():
    from pipeline.config import ConfigError, load_config
    text = ('[[evidence]]\nkind = "nearest_player"\n\n[decoder]\nkind = "majority_vote"\nhalf_width = 50\n\n'
            '[[refine]]\nkind = "min_duration"\n')
    with tempfile.TemporaryDirectory() as folder:
        path = write_text(folder, "trial.toml", text)
        expect_error(ConfigError, lambda: load_config(path), "min_duration", "available", "shift_edges")


# shift_edges (P02): move each kind of edge by a fixed number of frames. The design's algorithm is written out below as
# scalar loops, and the helpers work in letters so that a failed assertion shows the labels.

def shift_codes(text):
    """int8 label codes from a string of H, A, D and X."""
    from pipeline.contracts import CODES
    return np.array([CODES[letter] for letter in text], dtype=np.int8)


def shift_labels(labels, start=0, kickoffs=None, posterior=None, **parameters):
    """The Decoded that the shift_edges refiner returns for a block at [start, start + len(labels)).

    `kickoffs` is the dict of video frames that Inputs.kickoffs holds (none by default); the inputs here are a stand-in
    that has that and nothing else, which is all the refiner may ask of them.
    """
    from types import SimpleNamespace
    from pipeline.contracts import Block, BlockContext, Decoded, RunContext
    from pipeline.refine import REGISTRY
    inputs = SimpleNamespace(kickoffs={} if kickoffs is None else kickoffs)
    ctx = BlockContext(run=RunContext(inputs=inputs, seed=0), block=Block(None, start, start + len(labels)),
                       evidence={})
    return REGISTRY["shift_edges"](ctx, Decoded(labels=labels, posterior=posterior), **parameters)


def shift_reference(labels, offsets, fixed):
    """(boundaries, targets) by the design's rule as plain loops: target, clip, then a pass each way.

    A fixed boundary keeps its position as its target, and nothing more: the passes may still push it, which is why
    shift_edges.apply is held to this reference only where it leaves every fixed boundary where it was.
    """
    n = len(labels)
    cuts = [i for i in range(1, n) if labels[i] != labels[i - 1]]
    targets = [cut if keep else cut + int(offset) for cut, offset, keep in zip(cuts, offsets, fixed)]
    targets = [min(max(target, 1), n - 1) for target in targets]
    for i in range(1, len(targets)):
        targets[i] = max(targets[i], targets[i - 1] + 1)
    for i in reversed(range(len(targets))):
        targets[i] = min(targets[i], (targets[i + 1] if i + 1 < len(targets) else n) - 1)
    return cuts, targets


def shift_paint(labels, cuts, targets):
    """The labels with the run that starts at each of `cuts` starting at the matching target instead."""
    n = len(labels)
    painted = np.empty(n, dtype=np.int8)
    labels_of_runs = [labels[0]] + [labels[cut] for cut in cuts]
    edges = [0] + list(targets) + [n]
    for label, first, stop in zip(labels_of_runs, edges, edges[1:]):
        painted[first:stop] = label
    return painted


def run_labels(labels):
    """The labels of the runs of `labels`, in order, as a list."""
    import itertools
    return [int(key) for key, _ in itertools.groupby(labels.tolist())]


def test_shift_edges_zero_is_identity():
    labels = shift_codes("HHHAAADDDHHHXXAAD")
    posterior = np.full((len(labels), 3), 1.0 / 3.0)
    for parameters in ({}, {"team": 0, "into_dead": 0, "out_of_dead": 0}, {"team": np.int64(0)}):
        result = shift_labels(labels, posterior=posterior, **parameters)
        assert result.labels.tobytes() == labels.tobytes() and result.labels.dtype == np.int8, parameters
        assert result.posterior is posterior, "all zero must hand the posterior on as it was: {}".format(parameters)
    # Anything else moves labels, so the posterior would no longer describe them: it is dropped.
    moved = shift_labels(labels, posterior=posterior, team=1)
    assert moved.posterior is None and moved.labels.tobytes() != labels.tobytes()
    # A block with no edge at all is unchanged by any shift, and still drops the posterior (the rule is on the setting).
    flat = shift_codes("D" * 20)
    result = shift_labels(flat, posterior=np.zeros((20, 3)), team=5, into_dead=-5, out_of_dead=5)
    assert result.labels.tobytes() == flat.tobytes() and result.posterior is None


def test_shift_edges_moves_each_type():
    labels = shift_codes("H" * 40 + "A" * 40 + "D" * 40 + "H" * 40)
    for parameters, expected in (
            ({"team": 3}, "H" * 43 + "A" * 37 + "D" * 40 + "H" * 40),
            ({"team": -3}, "H" * 37 + "A" * 43 + "D" * 40 + "H" * 40),
            ({"into_dead": -5}, "H" * 40 + "A" * 35 + "D" * 45 + "H" * 40),
            ({"into_dead": 5}, "H" * 40 + "A" * 45 + "D" * 35 + "H" * 40),
            ({"out_of_dead": 2}, "H" * 40 + "A" * 40 + "D" * 42 + "H" * 38),
            ({"out_of_dead": -2}, "H" * 40 + "A" * 40 + "D" * 38 + "H" * 42),
            ({"team": 3, "into_dead": -5, "out_of_dead": 2}, "H" * 43 + "A" * 32 + "D" * 47 + "H" * 38)):
        got = letters(shift_labels(labels, **parameters).labels)
        assert got == expected, "{}: got {}".format(parameters, got)
    # The type is the pair of labels: H -> A and A -> H are both a team change, D -> A is a way out of the dead ball.
    labels = shift_codes("A" * 30 + "H" * 30 + "D" * 30 + "A" * 30)
    got = letters(shift_labels(labels, team=4, into_dead=-3, out_of_dead=2).labels)
    assert got == "A" * 34 + "H" * 23 + "D" * 35 + "A" * 28, got


def test_shift_edges_boundaries_and_types():
    from pipeline.contracts import LETTERS
    from pipeline.refine import shift_edges
    expected = {"HA": "team", "AH": "team", "HD": "into_dead", "AD": "into_dead", "DH": "out_of_dead",
                "DA": "out_of_dead"}
    for case in range(100):
        rng = np.random.default_rng([8, 1000 + case])
        labels = rng.integers(0, 4, size=int(rng.integers(0, 40))).astype(np.int8)
        positions, types = shift_edges.boundaries(labels)
        cuts = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
        assert positions.dtype == np.int64 and positions.tolist() == cuts, (case, positions, cuts)
        assert isinstance(types, tuple) and len(types) == len(cuts), (case, types)
        for cut, kind in zip(cuts, types):
            pair = LETTERS[labels[cut - 1]] + LETTERS[labels[cut]]
            assert kind == expected.get(pair, shift_edges.ABSTAIN), (case, pair, kind)
    assert set(expected.values()) == set(shift_edges.TYPES)
    # A change into or out of X is not one of the three kinds: it has no offset, and the refiner leaves it where it is.
    positions, types = shift_edges.boundaries(shift_codes("HHXXAADD"))
    assert positions.tolist() == [2, 4, 6] and types == (shift_edges.ABSTAIN, shift_edges.ABSTAIN, "into_dead"), types


def shift_random_labels(rng, n, long_runs):
    """n labels from H, A and D: independent per frame (dense boundaries), or runs of 1-9 frames (sparse ones)."""
    if not long_runs:
        return rng.integers(0, 3, size=n).astype(np.int8)
    labels = []
    while len(labels) < n:
        label = int(rng.integers(0, 3))
        while labels and label == labels[-1]:
            label = int(rng.integers(0, 3))
        labels += [label] * int(rng.integers(1, 10))
    return np.array(labels[:n], dtype=np.int8)


def test_shift_edges_brute_force():
    from pipeline.refine import shift_edges
    exact = displaced = anchored = 0
    for case in range(500):
        rng = np.random.default_rng([8, case])
        n = int(rng.integers(2, 31))
        labels = shift_random_labels(rng, n, long_runs=bool(case // 2 % 2))
        before = labels.copy()
        positions, _ = shift_edges.boundaries(labels)
        m = len(positions)
        offsets = rng.integers(-10, 11, size=m)
        fixed = rng.random(m) < (0.3 if case % 2 else 0.0)           # odd cases hold some boundaries, even ones none
        got = shift_edges.apply(labels, offsets, fixed)
        assert got.dtype == np.int8 and got.shape == (n,) and np.array_equal(labels, before), case
        new_positions, _ = shift_edges.boundaries(got)
        # The runs keep their labels and their order, and every run keeps at least one frame.
        assert run_labels(got) == run_labels(labels), (case, letters(labels), letters(got))
        assert len(new_positions) == m, (case, letters(labels), letters(got))
        assert (new_positions >= 1).all() and (new_positions <= n - 1).all() and (np.diff(new_positions) >= 1).all()
        # A fixed boundary never moves.
        assert np.array_equal(new_positions[fixed], positions[fixed]), (case, letters(labels), fixed)
        # With none held, each boundary is where a closed form puts it. Clipped to the block, target j is c_j and
        # pushes boundary i > j to c_j + (i - j) at the earliest; the block's end holds boundary i to n - m + i at the
        # latest. A boundary that nothing pushes and nothing clips moves by exactly its offset.
        targets = positions + offsets
        clipped = np.clip(targets, 1, n - 1)
        if not fixed.any():
            for i in range(m):
                reached = max((int(clipped[j]) + i - j for j in range(i)), default=1)
                expected = min(max(int(clipped[i]), reached), n - m + i)
                assert new_positions[i] == expected, (case, i, letters(labels), offsets)
                if targets[i] == clipped[i] and reached <= targets[i] <= n - m + i:
                    assert new_positions[i] == targets[i], (case, i, letters(labels), offsets)
                    exact += 1
                else:
                    displaced += new_positions[i] != targets[i]
        # The design's algorithm, as loops: the same labels wherever it leaves every fixed boundary alone.
        cuts, reference = shift_reference(labels, offsets, fixed)
        if np.array_equal(np.array(reference, dtype=np.int64)[fixed], positions[fixed]):
            assert np.array_equal(got, shift_paint(labels, cuts, reference)), (case, letters(labels), offsets, fixed)
        else:
            anchored += 1                               # there the design's passes would have pushed a fixed boundary
    # A check in which nothing was ever clipped, pushed or held would prove nothing about the passes.
    print("      boundaries unclipped {}, clipped or pushed {}; cases where a fixed boundary stopped one: {}".format(
        exact, displaced, anchored))
    assert exact >= 250 and displaced >= 500 and anchored >= 50, (exact, displaced, anchored)


def test_shift_edges_keeps_kickoffs():
    labels = shift_codes("D" * 30 + "H" * 30 + "A" * 20)
    # The boundary D -> H is at frame 1030 of the match: the block starts at 1000, so it cannot be moved.
    kept = shift_labels(labels, start=1000, kickoffs={1: 1030, 2: 5000}, team=3, out_of_dead=5)
    assert letters(kept.labels) == "D" * 30 + "H" * 33 + "A" * 17, letters(kept.labels)
    # Without a kick-off there, the same setting moves it. A kick-off is a frame of the match and not of the block:
    # 30 is where the boundary stands in the block, and it is not a kick-off there.
    for kickoffs in ({}, {2: 5000}, {1: 30}):
        moved = shift_labels(labels, start=1000, kickoffs=kickoffs, team=3, out_of_dead=5)
        assert letters(moved.labels) == "D" * 35 + "H" * 28 + "A" * 17, (kickoffs, letters(moved.labels))
    # Any kind of boundary at a kick-off frame stays, and the others move.
    held = shift_labels(labels, start=1000, kickoffs={1: 1060}, team=3, out_of_dead=5)
    assert letters(held.labels) == "D" * 35 + "H" * 25 + "A" * 20, letters(held.labels)
    # A kick-off at the first frame of the block is no boundary (the first frame is not a change): nothing is held.
    got = letters(shift_labels(labels, start=0, kickoffs={1: 0}, out_of_dead=5).labels)
    assert got == "D" * 35 + "H" * 25 + "A" * 20, got


def test_shift_edges_leaves_edges_of_x_alone():
    # H -> X and X -> A are fixed; a boundary beside them is stopped by them and does not push them.
    labels = shift_codes("H" * 10 + "A" * 2 + "X" * 10)
    got = letters(shift_labels(labels, team=5, into_dead=-3, out_of_dead=3).labels)
    assert got == "H" * 11 + "A" + "X" * 10, got
    labels = shift_codes("X" * 10 + "H" * 2 + "A" * 10)
    got = letters(shift_labels(labels, team=-5).labels)
    assert got == "X" * 10 + "H" + "A" * 11, got
    labels = shift_codes("HHHH" + "XXXX" + "DDDD" + "AAAA")
    got = letters(shift_labels(labels, team=2, into_dead=-1, out_of_dead=2).labels)
    assert got == "HHHH" + "XXXX" + "D" * 6 + "AA", got                # H -> X and X -> D stay; D -> A is a way out
    # In between, D -> A has the room that X leaves it and no more.
    labels = shift_codes("H" * 5 + "X" * 5 + "D" * 5 + "A" * 5 + "X" * 5)
    got = letters(shift_labels(labels, out_of_dead=3).labels)
    assert got == "H" * 5 + "X" * 5 + "D" * 8 + "A" * 2 + "X" * 5, got
    got = letters(shift_labels(labels, out_of_dead=9).labels)
    assert got == "H" * 5 + "X" * 5 + "D" * 9 + "A" * 1 + "X" * 5, got


def test_shift_edges_rejects_bad_parameters():
    labels = shift_codes("HHHAAADDD")
    for name in ("team", "into_dead", "out_of_dead"):
        for bad in (True, np.True_, 1.5, 3.0, 251, -251, np.nan, np.inf, "3", None):
            expect_error(ValueError, lambda: shift_labels(labels, **{name: bad}), "shift_edges", name)
        # The largest shift is accepted either way, as is a whole number from numpy or from a notebook sweep.
        for good in (250, -250, np.int64(7), np.int8(-7)):
            shift_labels(labels, **{name: good})


def test_shift_edges_apply_rejects_bad_arguments():
    from pipeline.refine import shift_edges
    labels = shift_codes("HHAADD")                                    # two boundaries
    offsets, fixed = np.array([1, -1]), np.array([False, False])
    shift_edges.apply(labels, offsets, fixed)
    for bad in (offsets[:1], np.array([1.0, -1.0]), np.array([True, False]), np.zeros((2, 1), dtype=np.int64)):
        expect_error(ValueError, lambda: shift_edges.apply(labels, bad, fixed), "shift_edges", "offsets")
    for bad in (fixed[:1], np.array([0, 1]), np.zeros((2, 1), dtype=bool)):
        expect_error(ValueError, lambda: shift_edges.apply(labels, offsets, bad), "shift_edges", "fixed")
    expect_error(ValueError, lambda: shift_edges.boundaries(np.zeros((2, 2), dtype=np.int8)), "shift_edges", "1-D")
    expect_error(ValueError, lambda: shift_edges.boundaries(np.array([0, 4], dtype=np.int8)), "shift_edges", "0-3")
    expect_error(ValueError, lambda: shift_edges.boundaries(np.array([0.0, 1.0])), "shift_edges", "integer")
    # Nothing to move: no boundary, no offsets, and a copy comes back.
    flat = shift_codes("AAAA")
    got = shift_edges.apply(flat, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool))
    assert got.tobytes() == flat.tobytes() and got is not flat and got.dtype == np.int8


def test_shift_edges_accepts_read_only_labels():
    labels = shift_codes("H" * 20 + "A" * 20 + "D" * 20)
    labels.flags.writeable = False
    before = labels.tobytes()
    posterior = np.full((60, 3), 1.0 / 3.0)
    posterior.flags.writeable = False
    for parameters in ({"team": 4, "into_dead": -4}, {}):
        result = shift_labels(labels, posterior=posterior, **parameters)
        assert labels.tobytes() == before and len(result.labels) == 60, parameters
    assert letters(shift_labels(labels, team=4, into_dead=-4).labels) == "H" * 24 + "A" * 12 + "D" * 24


# -- output -----------------------------------------------------------------------------------------------------------

def test_write_submission_bytes():
    import hashlib
    from pipeline import output
    from pipeline.contracts import A, D, H, X
    codes = np.array([X, X, H, H, A, D, X], dtype=np.int8)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "trial.csv")
        digest = output.write_submission(path, codes)
        with open(path, "rb") as handle:
            data = handle.read()
        # LF only, header exactly as the organisers' files, a newline after the last row (docs/traps.md T21).
        assert data == b"frame,label\n0,X\n1,X\n2,H\n3,H\n4,A\n5,D\n6,X\n", "unexpected bytes"
        assert digest == hashlib.sha256(data).hexdigest()
        assert not os.path.exists(path + ".tmp"), "the temporary file was left behind"
        output.validate(path, len(codes), codes)


def test_validate_rejects_short_file():
    from pipeline import output
    from pipeline.contracts import H
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "trial.csv")
        output.write_submission(path, np.full(7, H, dtype=np.int8))
        expect_error(output.SubmissionInvalid, lambda: output.validate(path, 8), "row count 7 != 8")


def test_out_path_rules():
    from pipeline import ROOT, output
    data_dir = os.path.join(ROOT, "data")
    outside = os.path.join(tempfile.gettempdir(), "mtah_trial.csv")
    for path in (os.path.join(ROOT, "evaluation", "results", "trial.csv"),
                 os.path.join(data_dir, "runs", "trial.csv"), outside):
        output.check_out_path(path, data_dir)
    expect_error(ValueError, lambda: output.check_out_path(
        os.path.join(ROOT, "evaluation", "results", "baseline_mine.csv"), data_dir), "baseline_", "T24")
    # git matches .gitignore case-insensitively on Windows and macOS, so the exception catches this one too.
    expect_error(ValueError, lambda: output.check_out_path(
        os.path.join(ROOT, "evaluation", "results", "Baseline_Mine.csv"), data_dir), "baseline_")
    expect_error(ValueError, lambda: output.check_out_path(os.path.join(ROOT, "out", "main.csv"), data_dir),
                 "gitignore")
    expect_error(ValueError, lambda: output.check_out_path(
        os.path.join(ROOT, "evaluation", "results", "nested", "trial.csv"), data_dir), "gitignore")
    expect_error(ValueError, lambda: output.check_out_path(outside[:-4] + ".txt", data_dir), ".csv")
    # .gitignore's *.csv pattern is case-sensitive where git is (Linux), so ".CSV" is not covered and is refused.
    expect_error(ValueError, lambda: output.check_out_path(outside[:-4] + ".CSV", data_dir), ".csv")


def test_summarise_counts_possessions_like_the_scorer():
    from pipeline import output
    from pipeline.contracts import A, D, H, X
    codes = np.array([X, H, H, D, D, H, H, X, X, H, X, H, A, X], dtype=np.int8)
    # Block 1 is "H H D D H H": a stoppage splits it into two possessions (docs/traps.md T05). Block 2 is "H X H A":
    # the scorer fills the X with the label before it, so that is one H possession and one A possession.
    summary = output.summarise(codes, [(1, 1, 7), (2, 9, 13)], fps=25)
    assert summary["possessions"] == 4, summary["possessions"]
    assert summary["graded_frames"] == 10, summary["graded_frames"]
    assert summary["labels_graded"] == {"H": 6, "A": 1, "D": 2, "X": 1}, summary["labels_graded"]


# -- the whole pipeline -----------------------------------------------------------------------------------------------

TRIAL_CONFIG = """
scope = "{scope}"

[[evidence]]
kind = "nearest_player"

[decoder]
kind = "majority_vote"
half_width = {width}
"""


def quiet(*_):
    """A log function that prints nothing, for runs inside checks."""


def test_run_end_to_end_synthetic():
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        config = load_config(write_text(root, "trial.toml", TRIAL_CONFIG.format(scope="halves", width=5)))
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the stand-in decoder never abstains inside a block"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        source = result.evidence["nearest_player"]
        assert not (source.alive.flags.writeable or source.home.flags.writeable), \
            "evidence must be read-only once checked: decoders get views of it"


HMM_TRIAL_CONFIG = """
[[evidence]]
kind = "nearest_player"

[decoder]
kind = "hmm"
switch_team = 5
switch_dead = 20
min_possession = 5
min_dead = 5
"""


def test_run_end_to_end_synthetic_hmm():
    from pipeline import output
    from pipeline.config import ConfigError, load_config
    from pipeline.contracts import X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        config = load_config(write_text(root, "trial.toml", HMM_TRIAL_CONFIG))
        assert config.decoder.kind == "hmm"
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the hmm decoder never abstains inside a block"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        # Both penalties are required: a config that forgets one is refused before any data loads.
        incomplete = HMM_TRIAL_CONFIG.replace("switch_dead = 20\n", "")
        expect_error(ConfigError, lambda: load_config(write_text(root, "incomplete.toml", incomplete)), "switch_dead")


# The shape of every Stage-1 config of P06 (pipeline/benchmark/configs/p06/): both sources and the hmm decoder, here at
# the grid's first point (p06s1_t001_d004). On the synthetic match's 230-frame blocks a heavier penalty decodes to D
# throughout, which would let the check below pass without a team ever being decoded.
P06_TRIAL_CONFIG = """
description = "trial"
seed = 0
scope = "halves"

[[evidence]]
kind = "nearest_player"
confidence = 0.78

[[evidence]]
kind = "ball_presence"
margin = 0
missing = 0.2
seen = 0.585

[decoder]
kind = "hmm"
switch_team = 1
switch_dead = 4
min_possession = 1
min_dead = 1
"""


def test_run_end_to_end_synthetic_p06():
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import A, D, H, X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        config = load_config(write_text(root, "trial.toml", P06_TRIAL_CONFIG))
        assert [source.kind for source in config.evidence] == ["nearest_player", "ball_presence"]
        assert config.decoder.kind == "hmm"
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the hmm decoder never abstains inside a block"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        assert {int(code) for code in np.unique(result.codes[~outside])} == {H, A, D}, "a label never decoded"
        assert set(result.evidence) == {"nearest_player", "ball_presence"}
        assert np.isnan(result.evidence["ball_presence"].home).all(), "ball_presence says nothing about the team"
        assert np.isnan(result.evidence["nearest_player"].alive).all(), "nearest_player says nothing about alive"


# The shape of every config of P07's grid (pipeline/benchmark/configs/p07/): main.toml's evidence and decoder plus
# `shape_possession` with the committed model, here at weight 0.1 and scope "gaps" (p07_w100_gaps). The synthetic
# match is not a benchmark fold, so the source reads the file's `full` model (a game with a fold is scored by its own).
P07_TRIAL_CONFIG = """
description = "trial"
seed = 0
scope = "halves"

[[evidence]]
kind = "controlled_touch"
radius = 2.0
max_rel_speed = 6.0
min_frames = 3
confidence = 0.91
hold = 0.55

[[evidence]]
kind = "dead_ball_model"
weight = 1.0
model = "pipeline/models/dead_ball_hgb.json"

[[evidence]]
kind = "shape_possession"
weight = 0.1
model = "pipeline/models/shape_possession_hgb.json"
scope = "gaps"
min_prob = 0.02

[decoder]
kind = "hmm"
switch_team = 4
switch_dead = 32
min_possession = 25
min_dead = 25
"""


def test_run_end_to_end_synthetic_p07():
    """The committed model in main.toml's pipeline plus the shape source writes a valid file, no X in a block."""
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.contracts import X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        config = load_config(write_text(root, "trial.toml", P07_TRIAL_CONFIG))
        assert [source.kind for source in config.evidence] == ["controlled_touch", "dead_ball_model",
                                                               "shape_possession"]
        assert config.evidence[2].weight == 0.1 and config.decoder.kind == "hmm"
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the hmm decoder never abstains inside a block"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        assert set(result.evidence) == {"controlled_touch", "dead_ball_model", "shape_possession"}
        shape = result.evidence["shape_possession"]
        assert np.isnan(shape.alive).all(), "the shape of the teams says nothing about alive"
        speaks = ~np.isnan(shape.home)
        assert speaks.any(), "the committed model said nothing on the synthetic match"
        assert not speaks[outside].any() and not speaks[inputs.atd.ball_present].any(), \
            "scope \"gaps\" speaks only inside a block, where the ball is missing"
        assert np.nanmin(shape.home) >= 0.02 and np.nanmax(shape.home) <= 0.98, "min_prob clips home"


SHIFT_REFINER = """
[[refine]]
kind = "shift_edges"
team = 3
into_dead = -2
out_of_dead = 4
"""


def test_run_end_to_end_synthetic_shift():
    """A config with a [[refine]] shift_edges writes a valid file, whose runs are the unrefined file's, moved."""
    from pipeline import output
    from pipeline.config import ConfigError, load_config
    from pipeline.contracts import A, D, H, X
    from pipeline.run import run
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        plain = run(load_config(write_text(root, "plain.toml", P06_TRIAL_CONFIG)), inputs,
                    os.path.join(root, "plain.csv"), log=quiet)
        config = load_config(write_text(root, "trial.toml", P06_TRIAL_CONFIG + SHIFT_REFINER))
        assert [component.kind for component in config.refine] == ["shift_edges"]
        assert config.refine[0].params == {"team": 3, "into_dead": -2, "out_of_dead": 4}
        out_path = os.path.join(root, "trial.csv")
        result = run(config, inputs, out_path, log=quiet)
        outside = np.ones(SYNTHETIC_FRAMES, dtype=bool)
        outside[50:280] = outside[320:560] = False
        assert (result.codes[outside] == X).all(), "frames outside the graded blocks must be X"
        assert (result.codes[~outside] != X).all(), "the refiner never writes an X"
        output.validate(out_path, SYNTHETIC_FRAMES, result.codes)
        assert {int(code) for code in np.unique(result.codes[~outside])} == {H, A, D}, "a label never decoded"
        # The refiner moved edges: not one run is gone or new, in either half, but the labels are not the same.
        assert not np.array_equal(result.codes, plain.codes), "the shift changed nothing"
        for start, stop in ((50, 280), (320, 560)):
            assert run_labels(result.codes[start:stop]) == run_labels(plain.codes[start:stop]), (start, stop)
        # A misspelt parameter is refused when the config loads, before any data does.
        typo = write_text(root, "typo.toml", P06_TRIAL_CONFIG + SHIFT_REFINER + "teams = 1\n")
        expect_error(ConfigError, lambda: load_config(typo), "teams", "shift_edges")


def test_run_file_scope_equals_baseline_script():
    """The synthetic twin of the --data regression: same bytes as starter/baseline.py writes, line endings aside."""
    from pipeline.config import load_config
    from pipeline.run import run
    from starter import baseline
    with tempfile.TemporaryDirectory() as root:
        folder = make_synthetic_bundle(root)
        inputs = synthetic_inputs(root)
        config = load_config(write_text(root, "trial.toml", TRIAL_CONFIG.format(scope="file", width=7)))
        run(config, inputs, os.path.join(root, "trial.csv"), log=quiet)
        codes = baseline.nearest_player_labels(
            os.path.join(folder, "BAR-ATH_tactical_atd.txt"),
            baseline.team_of_each_track(os.path.join(folder, "BAR-ATH_tactical_atd.xml")),
            SYNTHETIC_FRAMES, inputs.pitch)
        baseline.write_submission(os.path.join(root, "reference.csv"), baseline.smooth(codes, 7))
        with open(os.path.join(root, "reference.csv"), "rb") as handle:
            reference = handle.read().replace(b"\r\n", b"\n")  # baseline.py writes in text mode: CRLF on Windows
        with open(os.path.join(root, "trial.csv"), "rb") as handle:
            assert handle.read() == reference, "the pipeline's file differs from starter/baseline.py's"


def test_run_by_path_from_another_directory():
    import subprocess
    with tempfile.TemporaryDirectory() as elsewhere:
        script = os.path.join(ROOT, "pipeline", "run.py")
        shown = subprocess.run([sys.executable, script, "--help"], cwd=elsewhere, capture_output=True, text=True,
                               timeout=120)
        assert shown.returncode == 0 and "usage" in shown.stdout, shown.stderr[-500:]
        typo = write_text(elsewhere, "typo.toml", TRIAL_CONFIG.format(scope="halves", width=5).replace(
            "half_width", "half_widht"))
        refused = subprocess.run([sys.executable, script, typo], cwd=elsewhere, capture_output=True, text=True,
                                 timeout=120)
        assert refused.returncode == 1 and "half_widht" in refused.stdout, refused.stdout[-500:]


def test_git_hygiene():
    """Nothing git would commit is match data, a cache, or a CSV of labels (docs/traps.md T25)."""
    import subprocess
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                                capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        print("      git is not available here; nothing checked")
        return
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in status.splitlines() if line.strip()]
    risky = [path for path in paths if (path.startswith("data/") and path != "data/README.md")
             or path.lower().endswith((".csv", ".tmp", ".npy", ".npz", ".mp4"))]
    assert not risky, "git would pick up {} file(s) that may hold match data, e.g. {}".format(len(risky), risky[0])


# -- the real match (--data) ------------------------------------------------------------------------------------------

def test_data_regression_four_baselines():
    """scope = "file" + nearest_player + majority_vote rebuilds all four committed baselines, byte for byte."""
    import dataclasses
    from pipeline.config import load_config
    from pipeline.inputs import Inputs
    from pipeline.run import run
    config = load_config(os.path.join(HERE, "configs", "smooth50_repro.toml"))
    inputs = Inputs()
    with tempfile.TemporaryDirectory() as folder:
        for name, width in (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50),
                            ("baseline_smooth125", 125)):
            decoder = dataclasses.replace(config.decoder, params=dict(config.decoder.params, half_width=width))
            out_path = os.path.join(folder, name.replace("baseline_", "repro_") + ".csv")
            result = run(dataclasses.replace(config, decoder=decoder), inputs, out_path, log=quiet)
            with open(os.path.join(ROOT, "evaluation", "results", name + ".csv"), "rb") as handle:
                committed = handle.read().replace(b"\r\n", b"\n")  # core.autocrlf may have checked it out as CRLF
            with open(out_path, "rb") as handle:
                ours = handle.read()
            if ours != committed:
                ours_rows, committed_rows = ours.split(b"\n"), committed.split(b"\n")
                differ = [index for index, (a, b) in enumerate(zip(ours_rows, committed_rows)) if a != b]
                raise AssertionError("{}: {} rows differ (row counts {} vs {}), first at file row {}".format(
                    name, len(differ), len(ours_rows), len(committed_rows), differ[0] if differ else None))
            print("      {:<20} identical  sha256 {}...  (ATD cache {})".format(
                name, result.sha256[:16], inputs.atd_status))


def test_data_main_config_deterministic():
    """The team config twice in a row: same bytes, a valid file, and its numbers -- for information only."""
    from pipeline import output
    from pipeline.config import load_config
    from pipeline.inputs import Inputs
    from pipeline.run import run
    config = load_config(os.path.join(HERE, "configs", "main.toml"))
    inputs = Inputs()
    with tempfile.TemporaryDirectory() as folder:
        first = run(config, inputs, os.path.join(folder, "first.csv"), log=quiet)
        # a fresh Inputs: the evidence memos live on the object, and a shared one would hide the evidence stage
        second = run(config, Inputs(), os.path.join(folder, "second.csv"), log=quiet)
        assert first.sha256 == second.sha256, "two runs of the same config wrote different files (docs/traps.md T22)"
        output.validate(os.path.join(folder, "first.csv"), inputs.n_frames, first.codes)
        scores, _ = output.score(first.codes)
        print("      main.toml: {} possessions in the graded halves; public window macroF1 {:.4f} spellF1 {:.4f} "
              "(correctness only, T01)".format(first.summary["possessions"], scores["macro_f1"],
                                               scores["spell_f1_mean"]))


# -- smart_tagging ----------------------------------------------------------------------------------------------------

def tagged(items, graded=((1, 10, 60), (2, 80, 130)), n_frames=140):
    """A stand-in Inputs holding a hand-built tag list: items are (code, start, stop, team-or-None[, group key])."""
    from types import SimpleNamespace
    tags = []
    for item in items:
        code, start, stop, team = item[:4]
        key = item[4] if len(item) > 4 else "Team"
        tags.append({"code": code, "start_frame": start, "end_frame": stop, "start_s": start / 25.0,
                     "end_s": stop / 25.0, "tags": {} if team is None else {key: team}})
    return SimpleNamespace(smart_tagging=tags, graded=[tuple(block) for block in graded], n_frames=n_frames)


def tag_run(inputs):
    from types import SimpleNamespace
    return SimpleNamespace(inputs=inputs)


def test_smart_tagging_vote_table():
    from pipeline.evidence import smart_tagging
    for code in smart_tagging.TEAM_CODES:
        team = smart_tagging.votes(tagged([(code, 20, 30, "Barcelona")]))
        assert (team[20:30] == 1).all() and (team[:20] == -1).all(), code
    team = smart_tagging.votes(tagged([("DEFENDING IN OWN THIRD", 20, 30, "Barcelona"),
                                       ("DEFENDING IN MIDDLE THIRD", 40, 50, "Atletic Club")]))
    assert (team[20:30] == 0).all() and (team[40:50] == 1).all(), "a defending tag votes for the other team"
    team = smart_tagging.votes(tagged([("DEFENSIVE TRANSITION", 20, 30, "Atletic Club")]))
    assert (team[20:30] == 1).all(), "a defensive transition of the away team is the home team's ball"
    team = smart_tagging.votes(tagged([("SET PIECES", 20, 30, "Barcelona"), ("GOALS", 40, 45, "Barcelona")]))
    assert (team == -1).all(), "set pieces and goals are not voted on at all (R017 dropped the dead votes)"
    team = smart_tagging.votes(tagged([("PRESSING", 20, 30, "Barcelona"), ("BUILD UP", 40, 50, "N/A"),
                                       ("BUILD UP", 50, 55, None)]))
    assert (team == -1).all(), "an unknown code, N/A or a missing team gives nothing"
    team = smart_tagging.votes(tagged([(" build up ", 20, 30, " BARCELONA ", " team ")]))
    assert (team[20:30] == 1).all(), "case and spaces do not matter"


def test_smart_tagging_overlaps():
    from pipeline.evidence import smart_tagging
    inputs = tagged([("BUILD UP", 20, 40, "Barcelona"), ("PROGRESSION", 30, 50, "Barcelona")])
    team = smart_tagging.votes(inputs)
    assert (team[20:50] == 1).all(), "agreeing overlapping tags agree"
    evidence = smart_tagging.compute(tag_run(inputs), c_team=0.8)
    assert np.allclose(evidence.home[20:50], 0.8), "two agreeing tags are one vote in frame mode, not more"
    inputs = tagged([("BUILD UP", 20, 40, "Barcelona"), ("BUILD UP", 30, 50, "Atletic Club")])
    team = smart_tagging.votes(inputs)
    assert (team[20:30] == 1).all() and (team[30:40] == -2).all() and (team[40:50] == 0).all(), team[18:52]
    evidence = smart_tagging.compute(tag_run(inputs), c_team=0.8)
    assert np.isnan(evidence.home[30:40]).all() and not np.isnan(evidence.home[20:30]).any(), "NaN on the overlap only"
    assert np.allclose(evidence.home[40:50], 0.2), "an away vote is 1 - c"
    tag = smart_tagging.compute(tag_run(inputs), c_team=0.8, spread="tag")
    assert np.isnan(tag.home[30:40]).all(), "disagreeing tags are NaN in tag mode too"


def test_smart_tagging_cut_at_graded_blocks():
    from pipeline.evidence import smart_tagging
    inputs = tagged([("BUILD UP", 5, 15, "Barcelona"), ("BUILD UP", 62, 78, "Barcelona"),
                     ("BUILD UP", 55, 65, "Barcelona"), ("BUILD UP", 20, 30, "Barcelona")])
    team = smart_tagging.votes(inputs)
    assert (team[10:15] == 1).all() and (team[:10] == -1).all(), "a tag crossing a block's start is cut there"
    assert (team[60:80] == -1).all(), "a tag inside the halftime gap leaves nothing"
    assert (team[55:60] == 1).all() and (team[60:62] == -1).all(), "a tag crossing a block's end is cut there"
    team = smart_tagging.votes(tagged([("BUILD UP", 20, 30, "Barcelona")]))
    assert int((team == 1).sum()) == 10 and team[29] == 1 and team[30] == -1, "end_frame is exclusive"
    inputs = tagged([("BUILD UP", 5, 15, "Barcelona"), ("BUILD UP", 55, 65, "Atletic Club")])
    spans = smart_tagging.tag_spans(inputs)
    assert spans == [("team", 1, 10, 15), ("team", 0, 55, 60)], spans
    evidence = smart_tagging.compute(tag_run(inputs), c_team=0.8, spread="tag")
    expected = 1.0 / (1.0 + math.exp(-math.log(4.0) / 5))
    assert np.allclose(evidence.home[10:15], expected, atol=1e-12), "n_tag counts the frames inside the blocks only"
    outside = np.ones(140, dtype=bool)
    outside[10:60] = outside[80:130] = False
    for mode in ("frame", "tag"):
        evidence = smart_tagging.compute(tag_run(inputs), c_team=0.8, spread=mode)
        assert np.isnan(evidence.home[outside]).all() and np.isnan(evidence.alive).all(), mode
        assert evidence.home.shape == (140,) and evidence.alive.dtype == np.float64, mode


def test_smart_tagging_tag_spread_sums_to_one_observation():
    from pipeline.evidence import smart_tagging
    for c in (0.6, 0.8, 0.95):
        inputs = tagged([("BUILD UP", 20, 45, "Barcelona"), ("SET PIECES", 80, 100, None)])
        evidence = smart_tagging.compute(tag_run(inputs), c_team=c, spread="tag")
        home = evidence.home[20:45]
        assert abs(float(np.sum(np.log(home / (1 - home)))) - math.log(c / (1 - c))) < 1e-12, c
        assert np.isnan(evidence.alive).all() and np.isnan(evidence.home[80:100]).all()
    inputs = tagged([("BUILD UP", 20, 40, "Barcelona"), ("PROGRESSION", 30, 50, "Barcelona")])
    evidence = smart_tagging.compute(tag_run(inputs), c_team=0.8, spread="tag")
    assert evidence.home[35] > evidence.home[25], "in tag mode two agreeing tags add their observations"


def test_smart_tagging_rejects_bad_parameters():
    from pipeline.evidence import smart_tagging
    run = tag_run(tagged([("BUILD UP", 20, 30, "Barcelona")]))
    expect_error(TypeError, lambda: smart_tagging.compute(run), "c_team")
    expect_error(TypeError, lambda: smart_tagging.compute(run, c_team=0.8, c_dead=0.9), "c_dead")
    for bad in (0.5, 1.2, True, "0.8", 0.0, float("nan")):
        expect_error(ValueError, lambda bad=bad: smart_tagging.compute(run, c_team=bad), "c_team", "smart_tagging")
    expect_error(ValueError, lambda: smart_tagging.compute(run, c_team=0.8, spread="x"), "spread", "frame")
    assert smart_tagging.compute(run, c_team=1.0).home[25] == 1.0, "c = 1 is allowed"


def test_smart_tagging_is_match_only():
    from pipeline import evidence
    assert evidence.match_only("smart_tagging") is True, "the tags exist for our match only"


# -- runner -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", action="store_true",
                        help="also run the checks on the real match data (about a minute the first time, while the "
                             "ATD cache builds)")
    args = parser.parse_args(argv)

    names = sorted(name for name in globals() if name.startswith("test_"))
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

    print("\n{} of {} checks passed{}".format(
        len(names) - failures, len(names), "" if args.data else "  (add --data for the real-match regression)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
