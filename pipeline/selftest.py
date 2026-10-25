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
