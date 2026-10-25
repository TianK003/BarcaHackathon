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
    valid = ~np.isnan(values) & (rng.random(n) < 0.8)
    for before in (0, 3, 25):
        for after in (0, 3, 25):
            reference = np.full(n, np.nan)
            for start, stop in blocks:
                for t in range(start, stop):
                    low, high = max(start, t - before), min(stop, t + after + 1)
                    window = [v for v, ok in zip(values[low:high], valid[low:high]) if ok]
                    if window:
                        reference[t] = sum(window) / len(window)
            got = kinematics.window_mean(values, valid, blocks, before, after)
            assert np.array_equal(np.isnan(got), np.isnan(reference)), (before, after)
            assert np.nanmax(np.abs(got - reference)) < 1e-12, (before, after)
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
    held = len(kinematics._BALL_MEMO)
    del first, other, ball, players
    gc.collect()
    assert len(kinematics._BALL_MEMO) == held - 2, "the memo kept an object"
    assert len(kinematics._PLAYER_MEMO) == held - 2, "the memo kept an object"


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
    from pipeline.evidence import play_features
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
    started = time.perf_counter()
    play_features.features(inputs)
    elapsed = time.perf_counter() - started
    print("      features speed: 140,000 graded frames x {} slots (tracks included) in {:.2f} s".format(slots, elapsed))
    assert elapsed < 3.0, "{:.2f} s for 140,000 frames x 46 slots (the budget is 3 s)".format(elapsed)


# -- refine -----------------------------------------------------------------------------------------------------------

def test_config_refiner_registry_starts_empty():
    from pipeline.config import ConfigError, load_config
    text = ('[[evidence]]\nkind = "nearest_player"\n\n[decoder]\nkind = "majority_vote"\nhalf_width = 50\n\n'
            '[[refine]]\nkind = "min_duration"\n')
    with tempfile.TemporaryDirectory() as folder:
        path = write_text(folder, "trial.toml", text)
        expect_error(ConfigError, lambda: load_config(path), "min_duration", "REGISTRY is empty")


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
        second = run(config, inputs, os.path.join(folder, "second.csv"), log=quiet)
        assert first.sha256 == second.sha256, "two runs of the same config wrote different files (docs/traps.md T22)"
        output.validate(os.path.join(folder, "first.csv"), inputs.n_frames, first.codes)
        scores, _ = output.score(first.codes)
        print("      main.toml: {} possessions in the graded halves; public window macroF1 {:.4f} spellF1 {:.4f} "
              "(correctness only, T01)".format(first.summary["possessions"], scores["macro_f1"],
                                               scores["spell_f1_mean"]))


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
