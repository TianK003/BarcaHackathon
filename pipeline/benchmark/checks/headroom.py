"""Checks of the headroom runner (pipeline/benchmark/headroom.py): a config run beside variants that carry the truth.

The runner measures what a perfect dead-ball detector or a perfect possessor would be worth to a config, by running it
on the benchmark next to variants whose evidence is built from the benchmark's own truth. What is held against it here
is that the oracle cannot leak: the variants are what they say (the six stems in their order, each with the evidence
the README promises, written as files in the one folder), the oracles say what the truth says on exactly the frames
they should and are never in a production registry, the truth they read is the loaded match's own (the coarse one under
--truth coarse) and reaches them by closure and not through the inputs, and the registry is as it was after a run, even
a run that raised. The one that anchors the rest is test_headroom_full_oracle_is_perfect: with both oracles and a
decoder free to follow them, the score is exactly 1.0 on all three metrics, the same assertion that the oracle of
test_dfl_oracle_scores_perfectly makes of the runner, so the variants are known to carry the truth to the score.

The matches are the synthetic ones of checks/common.py (random flags and tracking, whose scores mean nothing), under the
gaps profile and the synthetic Calibration, or the clean profile where a check is about loading.
"""

import contextlib
import dataclasses
import gc
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import tomllib
import weakref

import numpy as np

from pipeline import evidence as evidence_stage
from pipeline.benchmark import headroom, truth
from pipeline.benchmark import inputs as match_inputs
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark.checks.common import (PROBE_CONFIG, loading, match_only_source, random_match, read,
                                              synthetic_calibration, synthetic_matches, yardstick_path)
from pipeline.config import Component, load_config
from pipeline.contracts import A, D, H, X, RunContext, check_evidence
from pipeline.selftest import expect_error, quiet, write_text

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STEMS = ("alive", "home_visible", "home_all", "alive_home_visible", "alive_home_all")
# The evidence kinds of each variant of HMM, in order: the home oracle takes the place of nearest_player, the alive
# oracle is added at the end.
KINDS = {"alive": ["nearest_player", "oracle_alive"],
         "home_visible": ["oracle_home_visible"],
         "home_all": ["oracle_home_all"],
         "alive_home_visible": ["oracle_home_visible", "oracle_alive"],
         "alive_home_all": ["oracle_home_all", "oracle_alive"]}
# A config that follows its evidence wherever it leads: a switch costs next to nothing and a run may last one frame.
# With both oracles every frame carries about 13.8 nats for the true state (eps 1e-6), so the decode is the truth.
HMM = """
description = "the hmm decoder on the nearest player's team, nearly free to switch"
seed = 0
scope = "halves"

[[evidence]]
kind = "nearest_player"

[decoder]
kind = "hmm"
switch_team = 0.01
switch_dead = 0.01
min_possession = 1
min_dead = 1
"""
# A config with a name, a weight, a float, an int and a scientific float to write and read back.
FULL = """
description = "a \\"quoted\\" description with a backslash \\\\ and an umlaut: ü"
seed = 3
scope = "halves"

[[evidence]]
kind = "ball_presence"
margin = 3
missing = 0.2
seen = 0.585

[[evidence]]
kind = "nearest_player"
name = "possessor"
weight = 2.5
confidence = 0.78

[decoder]
kind = "hmm"
switch_team = 8
switch_dead = 16.5
min_possession = 25
min_dead = 25
eps = 1e-6
"""
NO_HOME = """
scope = "halves"

[[evidence]]
kind = "ball_presence"

[decoder]
kind = "hmm"
switch_team = 8
switch_dead = 16
"""
# A config as main.toml is: no nearest_player, a source that is not one, so that only the restart variants can be made.
NO_NEAREST = """
description = "ball presence and nothing that names a possessor"
seed = 0
scope = "halves"

[[evidence]]
kind = "ball_presence"

[decoder]
kind = "hmm"
switch_team = 0.01
switch_dead = 0.01
min_possession = 1
min_dead = 1
"""
RESTART_STEMS = ("restart", "restart_run")
PLUS_STEMS = ("plus_home_visible", "plus_home_gaps", "plus_home_all", "alive_plus_home_visible",
              "alive_plus_home_gaps", "alive_plus_home_all")
# A config as main.toml is in the one way the tests need: several sources and no nearest_player at all.
TWO_SOURCES = """
description = "two ball_presence sources, no nearest_player"
seed = 0
scope = "halves"

[[evidence]]
kind = "ball_presence"

[[evidence]]
kind = "ball_presence"
name = "second"
weight = 0.5

[decoder]
kind = "hmm"
switch_team = 0.01
switch_dead = 0.01
min_possession = 1
min_dead = 1
"""
FAILS_MID_RUN = HMM.replace("switch_team = 0.01", "switch_team = -1")


# -- helpers ----------------------------------------------------------------------------------------------------------

def on_gaps(path, root, **keywords):
    """headroom.run_headroom on the synthetic matches under the gaps profile, one seed, nothing printed."""
    keywords.setdefault("profile", "gaps")
    keywords.setdefault("seeds", 1)
    keywords.setdefault("matches", synthetic_matches())
    keywords.setdefault("calibration", synthetic_calibration())
    keywords.setdefault("data_dir", root)
    keywords.setdefault("log", quiet)
    return headroom.run_headroom(path, **keywords)


def variants_folder(root):
    return os.path.join(root, "runs", "benchmark", "headroom", "configs")


@contextlib.contextmanager
def oracles_registered():
    """The oracles in the evidence registry, so that a written variant can be loaded; gone afterwards."""
    with headroom.registered(evidence_stage.REGISTRY, **headroom.oracles({})):
        yield


def pooled_of(result):
    return {entry["stem"]: entry["report"]["pooled"] for entry in result["configs"]}


def load_config_of(text):
    """A Config of `text`, written to a temporary file and loaded by the pipeline's own loader."""
    with tempfile.TemporaryDirectory() as root:
        return load_config(write_text(root, "writer.toml", text))


def oracle_inputs(match_id="RND001"):
    match = random_match(match_id)
    return match, match_inputs.MatchInputs(match, match_inputs.clean_atd(match))


# -- the variants -----------------------------------------------------------------------------------------------------

def test_headroom_variants_and_order():
    """CONFIG first, from its own file, then the five variants in the order given, each a file in the fixed folder.

    The stems are <stem>__alive, __home_visible, __home_all, __alive_home_visible and __alive_home_all; CONFIG is the
    reference of the comparisons, so a delta is a variant minus CONFIG. A variant is written as TOML into
    <data dir>/runs/benchmark/headroom/configs/ and nowhere else (the reports of a run that does not write are none),
    and loads through the pipeline's own loader with the evidence the README promises: the home oracle in the place of
    nearest_player, the alive oracle added after the config's own sources, and everything else (seed, scope, decoder,
    refiners) exactly the config's. The printout names every variant.
    """
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "hmm_free.toml", HMM)
        lines = []
        result = on_gaps(path, root, log=lines.append)
        stems = [entry["stem"] for entry in result["configs"]]
        assert stems == ["hmm_free"] + ["hmm_free__" + name for name in STEMS], stems
        assert result["reference"] == "hmm_free" and result["configs"][0]["report"]["versus"] is None
        assert all(entry["report"]["versus"]["reference"]["stem"] == "hmm_free" for entry in result["configs"][1:])
        assert result["configs"][0]["report"]["config"]["text"] == read(path), "CONFIG runs from its own file, as it is"

        folder = variants_folder(root)
        assert sorted(os.listdir(folder)) == sorted(stem + ".toml" for stem in stems[1:]), os.listdir(folder)
        assert sorted(os.listdir(root)) == ["hmm_free.toml", "runs"] and os.listdir(os.path.join(root, "runs")) == [
            "benchmark"] and os.listdir(os.path.join(root, "runs", "benchmark")) == ["headroom"], (
            "a run that does not write writes the variants and nothing else: " + str(sorted(os.listdir(root))))
        for entry in result["configs"][1:]:
            assert entry["report"]["config"]["path"].endswith(
                "/runs/benchmark/headroom/configs/" + entry["stem"] + ".toml"), entry["report"]["config"]["path"]

        base = load_config(path)
        with oracles_registered():
            for name in STEMS:
                variant = load_config(os.path.join(folder, "hmm_free__" + name + ".toml"))
                assert [component.kind for component in variant.evidence] == KINDS[name], (name, variant.evidence)
                assert (variant.seed, variant.scope, variant.decoder, variant.refine) == (
                    base.seed, base.scope, base.decoder, base.refine), name
                assert variant.description.startswith(base.description) and "hmm_free" in variant.description

        table = "\n".join(lines)
        for name in STEMS:
            assert any(line.startswith("variant") and "hmm_free__" + name in line for line in lines), (name, table)
        assert "against hmm_free" in table, table


def test_headroom_full_oracle_is_perfect():
    """With both oracles and a decoder that follows them, pooled spellF1, macroF1 and alive/dead F1 are exactly 1.0.

    The alive oracle is 1 on the truth's H and A frames and 0 on its D frames, the home oracle 1 on H and 0 on A, each
    about 13.8 nats for the true state at eps 1e-6, and the hmm of HMM pays 0.01 to switch: the decode is the truth.
    That is the runner's anchor, as test_dfl_oracle_scores_perfectly is the benchmark's: the variants carry the truth
    through the inputs, the decoder, the blocks and the scoring. The controls are the config itself and the variants
    that have only a part of the truth, which must fall short: an oracle that scored 1.0 whatever it was given would
    prove nothing.
    """
    with tempfile.TemporaryDirectory() as root:
        result = on_gaps(write_text(root, "hmm_free.toml", HMM), root)
        pooled = pooled_of(result)
        perfect = pooled["hmm_free__alive_home_all"]
        assert perfect["spell_f1_mean"] == perfect["macro_f1"] == perfect["alive_dead_f1"] == 1.0, perfect
        assert perfect["n_gt"] == perfect["n_pred"] > 50, "few possessions would prove little: " + str(perfect)
        for stem in ("hmm_free", "hmm_free__alive", "hmm_free__home_visible", "hmm_free__home_all",
                     "hmm_free__alive_home_visible"):
            assert pooled[stem]["macro_f1"] < 1.0, (stem, pooled[stem])
        # The alive oracle alone cannot name a team, and the home oracle alone cannot call a dead ball.
        assert pooled["hmm_free__alive"]["alive_dead_f1"] == 1.0, pooled["hmm_free__alive"]
        assert pooled["hmm_free__home_all"]["alive_dead_f1"] < 1.0, pooled["hmm_free__home_all"]


def test_headroom_coarse_truth_reaches_the_oracles():
    """Under --truth coarse the oracles carry the coarse truth, so the full oracle is perfect against it too.

    The coarse truth absorbs the raw truth's short sandwiched possessions, so a raw oracle scored against it would fall
    short: that it scores exactly 1.0 says that the closure was given coarse_codes of the same loaded match, which the
    runner scores against. The control is that the two truths differ on these matches, and that the config's reports
    say which truth they were scored against.
    """
    matches = synthetic_matches()
    differ = 0
    for match in matches.values():
        raw = truth.truth_codes(match)[0]
        differ += int((truth.coarse_codes(raw, match.graded, match.fps) != raw).sum())
    assert differ > 0, "the coarse truth must differ from the raw one on these matches, or this check proves nothing"
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "hmm_free.toml", HMM)
        coarse = on_gaps(path, root, truth="coarse")
        assert coarse["truth"]["name"] == "coarse" and all(
            entry["report"]["truth"]["name"] == "coarse" for entry in coarse["configs"])
        perfect = pooled_of(coarse)["hmm_free__alive_home_all"]
        assert perfect["spell_f1_mean"] == perfect["macro_f1"] == perfect["alive_dead_f1"] == 1.0, perfect
        # The raw run's oracle is the raw truth: perfect against it, and it differs from the coarse run's evidence only
        # in what it was handed, so the two runs score the same configs on different truths.
        raw = on_gaps(path, root)
        assert pooled_of(raw)["hmm_free__alive_home_all"]["macro_f1"] == 1.0
        assert pooled_of(raw)["hmm_free"] != pooled_of(coarse)["hmm_free"], "the truths are not the same truth"


def test_headroom_refuses_what_it_cannot_make_variants_of():
    """A config without a nearest_player, one decoded over the file, one whose name has a dot: refused, nothing written.

    The home variants replace the config's nearest_player, so a config without one (or with two, which of them is not
    the runner's to say) has no home variant to make; it is refused before any file is written, with a sentence naming
    the config and the source. The runner's own refusals (scope = "file", a dot in the name) come first, here and with
    the same words, so that no variant of a config that could not run is left in the folder. A parameter that is not a
    string, a number or a bool cannot be written, and says which.
    """
    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "runs")
        expect_error(benchmark_run.BenchmarkError, lambda: on_gaps(write_text(root, "no_home.toml", NO_HOME), root),
                     "no_home", "nearest_player", "home")
        twice = NO_HOME.replace('kind = "ball_presence"', 'kind = "nearest_player"\n\n[[evidence]]\n'
                                'kind = "nearest_player"\nname = "second"')
        message = expect_error(benchmark_run.BenchmarkError,
                               lambda: on_gaps(write_text(root, "twice.toml", twice), root),
                               "twice", "nearest_player", "second", "keep one")
        assert "2" in message, message
        expect_error(benchmark_run.BenchmarkError,
                     lambda: on_gaps(write_text(root, "whole_file.toml", HMM.replace('"halves"', '"file"')), root),
                     "whole_file", "scope")
        expect_error(benchmark_run.BenchmarkError,
                     lambda: on_gaps(write_text(root, "has.dot.toml", HMM), root), "has.dot", "dot", "rename")
        assert not os.path.exists(folder), "a refused config leaves no variant behind: " + str(os.listdir(root))

        config = load_config(write_text(root, "hmm_free.toml", HMM))
        listed = dataclasses.replace(config, decoder=Component(
            kind="hmm", name="hmm", weight=1.0, params={"switch_team": 8, "values": [1, 2]}))
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.toml_text(listed, listed.evidence, "d", ["n"]),
                     "decoder", "values", "list", "str, int, float and bool")
        expect_error(benchmark_run.BenchmarkError, lambda: on_gaps(
            write_text(root, "nope.toml", HMM), root, profile="nope"), "nope", "profile")


def test_headroom_refuses_a_config_that_fills():
    """Headroom writes its variants without [inputs], so a config whose ball is filled ([inputs] ball_fill =
    "broadcast", R028) would lose its fill in every variant, silently: it is refused instead, with a ValueError that
    names the config and says that headroom does not serve the fill, before any variant is written (ruling 12)."""
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "filling.toml", HMM + '\n[inputs]\nball_fill = "broadcast"\n')
        config = load_config(path)
        assert config.ball_fill == "broadcast"
        expect_error(ValueError, lambda: headroom.toml_text(config, config.evidence, "d", ["n"]),
                     "filling", "headroom does not serve the fill", "ball_fill")
        expect_error(benchmark_run.BenchmarkError, lambda: on_gaps(path, root), "filling",
                     "headroom does not serve the fill")
        assert not os.path.exists(os.path.join(root, "runs")), "refused before any variant is written"
        plain = load_config(write_text(root, "plain.toml", HMM + '\n[inputs]\nball_fill = "none"\n'))
        assert "[inputs]" not in headroom.toml_text(plain, plain.evidence, "d", ["n"]), "no fill, nothing to keep"


# -- the restart oracle (P17) -----------------------------------------------------------------------------------------

def test_headroom_restart_oracle_marks_each_restart():
    codes = np.array([H] * 10 + [D] * 10 + [A] * 30 + [D] * 5 + [H] * 5, dtype=np.int8)   # one block [0, 60)
    home = headroom.restart_home(codes, [(1, 0, 60)], 4)
    expect = np.full(60, np.nan)
    expect[20:24] = 0.0
    expect[55:59] = 1.0
    assert np.array_equal(home, expect, equal_nan=True)
    whole = headroom.restart_home(codes, [(1, 0, 60)], None)
    expect = np.full(60, np.nan)
    expect[20:50] = 0.0
    expect[55:60] = 1.0
    assert np.array_equal(whole, expect, equal_nan=True)
    assert not home.flags.writeable and home.dtype == np.float64 and headroom.RESTART_TAIL == 50


def test_headroom_restart_oracle_stops_at_the_next_change_and_the_block():
    codes = np.array([D] * 5 + [A] * 3 + [H] * 12 + [D] * 4 + [A] * 6, dtype=np.int8)     # blocks [0, 20), [20, 30)
    home = headroom.restart_home(codes, [(1, 0, 20), (2, 20, 30)], 50)
    expect = np.full(30, np.nan)
    expect[5:8] = 0.0
    expect[24:30] = 0.0
    assert np.array_equal(home, expect, equal_nan=True)     # the leading D run (a kick-off) counts
    expect_error(ValueError, lambda: headroom.restart_home(codes, [(1, 0, 30)], 0), "restart", "tail")


def test_headroom_restart_oracle_needs_a_team_after_the_stoppage():
    codes = np.array([H] * 5 + [D] * 5 + [X] * 2 + [H] * 3 + [D] * 5, dtype=np.int8)
    assert np.isnan(headroom.restart_home(codes, [(1, 0, 20)], 50)).all()   # D->X and D to the block's end: nothing


def test_headroom_restart_variants_need_no_home_source():
    # a config with ball_presence and no nearest_player, as main.toml is (it has controlled_touch)
    with tempfile.TemporaryDirectory() as root:
        config = load_config(write_text(root, "no_nearest.toml", NO_NEAREST))
        texts = headroom.variant_texts(config, variants=("restart", "restart_run"))
        assert [stem for stem, _ in texts] == [config.stem + "__restart", config.stem + "__restart_run"]
        with oracles_registered():
            for (stem, text), kind in zip(texts, ("oracle_restart", "oracle_restart_run")):
                variant = load_config(write_text(root, stem + ".toml", text))
                assert list(variant.evidence) == list(config.evidence) + [
                    Component(kind=kind, name=kind, weight=1.0, params={})], variant.evidence
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.check_config(config), "nearest_player")
        headroom.check_config(config, variants=("restart",))        # a restart variant replaces nothing
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.check_config(config, variants=("restart", "alive_home_all")),
                     "nearest_player")
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.check_config(config, variants=("restart", "restart")),
                     "restart", "twice")
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.variant_texts(config, variants=("nope",)), "nope")
        # a config that has a nearest_player keeps it: the oracle is appended to its evidence
        base = load_config(write_text(root, "hmm_free.toml", HMM))
        (_, text), = headroom.variant_texts(base, variants=("restart",))
        with oracles_registered():
            kinds = [c.kind for c in load_config(write_text(root, "hmm_free__restart.toml", text)).evidence]
        assert kinds == ["nearest_player", "oracle_restart"], kinds


def test_headroom_restart_variants_run_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "no_nearest.toml", NO_NEAREST)
        result = on_gaps(path, root, variants=RESTART_STEMS)
        stems = [entry["stem"] for entry in result["configs"]]
        assert stems == ["no_nearest"] + ["no_nearest__" + name for name in RESTART_STEMS], stems
        assert sorted(os.listdir(variants_folder(root))) == sorted(stem + ".toml" for stem in stems[1:])
        result = on_gaps(write_text(root, "hmm_free.toml", HMM), root, variants=("restart",))
        assert [entry["stem"] for entry in result["configs"]] == ["hmm_free", "hmm_free__restart"]
    # the command line takes the same names, and refuses others
    with tempfile.TemporaryDirectory() as root:
        shown = io.StringIO()
        try:
            with contextlib.redirect_stderr(shown):
                headroom.main([write_text(root, "x.toml", HMM), "--variants", "nope"])
        except SystemExit as stop:
            assert stop.code == 2, stop.code
        else:
            raise AssertionError("--variants nope was accepted")
        assert headroom.VARIANT_NAMES == STEMS + RESTART_STEMS + PLUS_STEMS, headroom.VARIANT_NAMES


# -- the file the variants are written as -----------------------------------------------------------------------------

def test_headroom_variants_round_trip_what_config_holds():
    """A variant read back is the config with the oracles in: names, weights, parameters, description, everything else.

    The writer handles the flat tables that a Config holds. This one has a named source with a weight, an int, floats of
    both kinds (0.585, 1e-6, 16.5) and a description with a quote, a backslash and an umlaut. The oracle that replaces a
    source takes its weight and the name of its own kind and none of its parameters (confidence means nothing to it);
    the one that is added has weight 1. What the config says apart from its evidence is read back equal.
    """
    with tempfile.TemporaryDirectory() as root:
        base = load_config(write_text(root, "full.toml", FULL))
        headroom.write_variants(base, root)
        folder = variants_folder(root)
        with oracles_registered():
            for name in STEMS:
                variant = load_config(os.path.join(folder, "full__" + name + ".toml"))
                assert (variant.seed, variant.scope, variant.decoder, variant.refine) == (
                    base.seed, base.scope, base.decoder, base.refine), name
                assert variant.description.startswith(base.description), (name, variant.description)
                expected = [base.evidence[0]]
                if "home" in name:
                    kind = "oracle_home_" + ("visible" if "visible" in name else "all")
                    expected.append(Component(kind=kind, name=kind, weight=2.5, params={}))
                else:
                    expected.append(base.evidence[1])
                if name.startswith("alive"):
                    expected.append(Component(kind="oracle_alive", name="oracle_alive", weight=1.0, params={}))
                assert list(variant.evidence) == expected, (name, variant.evidence)
        # The same text, byte for byte, on a second write: the paths and the report bytes stay stable.
        before = {name: read(os.path.join(folder, name), "rb") for name in os.listdir(folder)}
        headroom.write_variants(base, root)
        assert {name: read(os.path.join(folder, name), "rb") for name in os.listdir(folder)} == before
        assert all(b"\r" not in text and text.endswith(b"\n") for text in before.values()), "LF, one final newline"
        assert sorted(os.listdir(folder)) == sorted("full__" + name + ".toml" for name in STEMS), os.listdir(folder)


def test_headroom_toml_writer_reads_back_as_written():
    """toml_text writes str, int, float and bool as TOML that tomllib reads back as the same values.

    Strings with a quote, a backslash, a newline, a tab, a control character and a non-ASCII letter, ints of both
    signs, floats including 1e-06 and infinity, both bools, and a key that is not a bare key.
    """
    config = load_config_of(HMM)
    tricky = {"text": 'a "quoted" back\\slash\nnew\tline \x01 \x7f ü 中', "zero": 0, "big": -(2 ** 40),
              "small": 1e-06, "half": 0.5, "negative": -2.5, "infinite": math.inf, "yes": True, "no": False,
              "odd key": 1, "dashed-key_9": 2}
    changed = dataclasses.replace(config, refine=(Component(kind="fake", name="fake", weight=1.0, params=tricky),))
    text = headroom.toml_text(changed, changed.evidence, 'say "hi" \\', ["one line", "another"])
    assert text.startswith("# one line\n# another\n") and text.endswith("\n") and "\r" not in text, text[:80]
    table = tomllib.loads(text)
    assert table["description"] == 'say "hi" \\' and table["seed"] == 0 and table["scope"] == "halves"
    assert table["refine"] == [dict(tricky, kind="fake")], table["refine"]
    assert [entry["kind"] for entry in table["evidence"]] == ["nearest_player"] and "name" not in table["evidence"][0]
    assert "weight" not in table["evidence"][0], "a default name and weight are not written"
    assert table["decoder"] == {"kind": "hmm", "switch_team": 0.01, "switch_dead": 0.01, "min_possession": 1,
                                "min_dead": 1}
    assert isinstance(table["decoder"]["min_possession"], int) and not isinstance(table["decoder"]["min_possession"],
                                                                                   bool)
    # A bool is not written as an int, nor an int as a float.
    assert "yes = true" in text and "no = false" in text and "zero = 0\n" in text and "half = 0.5" in text


# -- the oracles ------------------------------------------------------------------------------------------------------

def test_headroom_oracles_say_what_the_truth_says():
    """Each oracle is the truth on exactly the frames it should speak on, and NaN on the rest.

    oracle_alive: alive 1 on the truth's H and A, 0 on its D, NaN on X (outside the graded halves); home NaN everywhere.
    oracle_home_visible: home 1 on H and 0 on A where the tracking has the ball, NaN elsewhere; alive NaN everywhere.
    oracle_home_all: home 1 on H and 0 on A on every frame the truth calls H or A, ball or not; alive NaN. Each is a
    valid Evidence, and for a match whose truth was not loaded an oracle refuses and says so (it cannot guess).
    """
    match, inputs = oracle_inputs()
    codes = truth.truth_codes(match)[0]
    run_context = RunContext(inputs=inputs, seed=0)
    made = headroom.oracles({"RND001": codes})
    assert sorted(made) == ["oracle_alive", "oracle_home_all", "oracle_home_gaps", "oracle_home_visible",
                            "oracle_restart", "oracle_restart_run"], sorted(made)
    live, home = (codes == H) | (codes == A), codes == H
    seen = inputs.atd.ball_present
    assert 0 < (live & seen).sum() < live.sum(), "the check needs live frames with the ball seen and not seen"
    assert (codes == D).any() and (codes == X).any()

    def check(kind, alive, home_expected):
        evidence = made[kind](run_context)
        check_evidence(dataclasses.replace(evidence, name=kind), inputs.n_frames)
        for channel, expected in (("alive", alive), ("home", home_expected)):
            got = getattr(evidence, channel)
            assert np.array_equal(got, expected, equal_nan=True), (kind, channel, int(np.isnan(got).sum()))

    nan = np.full(inputs.n_frames, np.nan)
    check("oracle_alive", np.where(live, 1.0, np.where(codes == D, 0.0, np.nan)), nan)
    check("oracle_home_visible", nan, np.where(live & seen, np.where(home, 1.0, 0.0), np.nan))
    check("oracle_home_all", nan, np.where(live, np.where(home, 1.0, 0.0), np.nan))

    other = match_inputs.MatchInputs(random_match("RND009"), match_inputs.clean_atd(random_match("RND009")))
    for kind in made:
        expect_error(ValueError, lambda: made[kind](RunContext(inputs=other, seed=0)), kind, "RND009", "truth")


def test_headroom_restores_the_registry():
    """The evidence registry is equal before and after a run, and after a run that raised in the middle of it.

    Inside the context manager the three oracles are in it (the variants could not load otherwise); outside there is
    none. The run that raises is one whose decoder refuses its parameter once the oracles are registered and the
    variants have been written: the registry is restored on the way out, as the checks' own `registered` does.
    """
    before = dict(evidence_stage.REGISTRY)
    with tempfile.TemporaryDirectory() as root:
        on_gaps(write_text(root, "hmm_free.toml", HMM), root)
        assert evidence_stage.REGISTRY == before, "registry changed by a run"
        seen_inside = []
        real = benchmark_run.evaluate_configs

        def spy(*arguments, **keywords):
            seen_inside.append(sorted(key for key in evidence_stage.REGISTRY if key.startswith("oracle_")))
            return real(*arguments, **keywords)

        benchmark_run.evaluate_configs = spy
        try:
            on_gaps(write_text(root, "again.toml", HMM), root)
        finally:
            benchmark_run.evaluate_configs = real
        assert seen_inside == [["oracle_alive", "oracle_home_all", "oracle_home_gaps", "oracle_home_visible",
                                "oracle_restart", "oracle_restart_run"]], seen_inside

        expect_error(ValueError, lambda: on_gaps(write_text(root, "fails.toml", FAILS_MID_RUN), root),
                     "hmm", "switch_team")
        assert evidence_stage.REGISTRY == before, "registry not restored after a run that raised"
        expect_error(benchmark_run.BenchmarkError, lambda: on_gaps(write_text(root, "hmm_free.toml", HMM), root,
                                                                   profile="nope"), "nope")
        assert evidence_stage.REGISTRY == before, "registry not restored after a refused run"
    with headroom.registered(evidence_stage.REGISTRY, extra=lambda run: None):
        assert "extra" in evidence_stage.REGISTRY
        try:
            with headroom.registered(evidence_stage.REGISTRY, more=lambda run: None):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
        assert "more" not in evidence_stage.REGISTRY and "extra" in evidence_stage.REGISTRY
    assert evidence_stage.REGISTRY == before


def test_headroom_never_in_production():
    """After importing headroom, pipeline.evidence.REGISTRY has no oracle_* key: they live in its context manager.

    Checked in a clean interpreter, where nothing else of this harness has registered or restored anything: import the
    module, list the three stages' registries. A production config could then never name an oracle, and the config
    loader would refuse it ("has no component"). The same holds in this process after the checks above ran.
    """
    code = ("import pipeline.benchmark.headroom, pipeline.evidence, pipeline.decode, pipeline.refine; "
            "print(sorted(pipeline.evidence.REGISTRY), sorted(pipeline.decode.REGISTRY), "
            "sorted(pipeline.refine.REGISTRY))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "oracle" not in done.stdout, "an oracle is in a production registry after importing headroom: " + done.stdout
    assert not [key for key in evidence_stage.REGISTRY if key.startswith("oracle")], sorted(evidence_stage.REGISTRY)
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "oracle_alive.toml", HMM.replace("nearest_player", "oracle_alive"))
        expect_error(ValueError, lambda: load_config(path), "oracle_alive", "no component")


# -- loading and writing ----------------------------------------------------------------------------------------------

def test_headroom_loads_each_match_once_and_keeps_none():
    """Through the runner's own loading, each match is loaded once, its truth is the loaded match's, and none is kept.

    Without `matches` the matches are read one at a time, as the runner reads them, by dfl.load_match: here served by
    random matches that keep weak references, so that a match still in memory when the next is loaded, or after the run,
    is seen. The truth the oracles read is made from the match that the runner loaded (the full oracle scores exactly
    1.0 on it), keyed by its id; the order of the matches is the one asked for.
    """
    made = []

    def serve(match_id):
        gc.collect()
        alive = [name for name, refs in made if any(ref() is not None for ref in refs)]
        assert not alive, "{} is still in memory while {} is loaded".format(alive, match_id)
        match = random_match(match_id, seed=len(made))
        made.append((match_id, [weakref.ref(match), weakref.ref(match.players_xy), weakref.ref(match.ball_xy),
                                weakref.ref(match.ball_possession)]))
        return match

    ids = ["J03WOH", "J03WMX", "J03WR9"]
    with tempfile.TemporaryDirectory() as root, loading(serve):
        result = headroom.run_headroom(write_text(root, "hmm_free.toml", HMM), profile="clean", match_ids=ids,
                                       data_dir=root, log=quiet)
    assert [name for name, _ in made] == ids, "each match loaded once, in the order asked for: " + str(made)
    assert [record["id"] for record in result["matches"]] == sorted(ids)
    perfect = pooled_of(result)["hmm_free__alive_home_all"]
    assert perfect["spell_f1_mean"] == perfect["macro_f1"] == perfect["alive_dead_f1"] == 1.0, perfect
    gc.collect()
    assert all(ref() is None for _, refs in made for ref in refs), "a match is still held after the run"


def test_headroom_writes_the_variants_reports_only_when_asked_and_never_the_references():
    """write=True writes the five variants' reports (<stem>__alive.json, ...) and not CONFIG's; without it, none.

    The reports are the runner's own (write_reports) with the variants' stems, in <data dir>/runs/benchmark/<profile>/;
    a second run writes the same bytes, the variants' files included, since their folder and text are fixed. CONFIG's
    report is the runner's to write: here it is one that run.py wrote when CONFIG was the second of two configs, so that
    it has a `versus` block, and a headroom run with write=True leaves it byte for byte as it was (a report written from
    the headroom run would have replaced it with one that has no comparison).
    """
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "hmm_free.toml", HMM)
        folder = os.path.join(root, "runs", "benchmark", "gaps")
        on_gaps(path, root)
        assert not os.path.exists(folder), "no report unless asked for"
        suffix = ".m-RND001-RND002.s1.json"          # the synthetic matches are not the seven: the runner's own rule
        runner = benchmark_run.evaluate_configs(
            [yardstick_path(50), path], profile="gaps", seeds=1, matches=synthetic_matches(),
            calibration=synthetic_calibration(), data_dir=root, log=quiet)
        benchmark_run.write_reports(runner)
        reference = os.path.join(folder, "hmm_free" + suffix)
        reference_bytes = read(reference, "rb")
        assert json.loads(reference_bytes)["versus"]["reference"]["stem"] == "nearest_w50", "it has a versus block"
        lines = []
        on_gaps(path, root, write=True, log=lines.append)
        names = sorted(["hmm_free__" + name + suffix for name in STEMS] + ["hmm_free" + suffix, "nearest_w50" + suffix])
        assert sorted(os.listdir(folder)) == names, sorted(os.listdir(folder))
        assert read(reference, "rb") == reference_bytes, "CONFIG's report, with its versus block, was replaced"
        reported = [line for line in lines if line.startswith("report")]
        assert len(reported) == 5 and not any("hmm_free" + suffix in line for line in reported), reported
        names = sorted(name for name in names if "__" in name)
        first = {name: read(os.path.join(folder, name), "rb") for name in names}
        variants = {name: read(os.path.join(variants_folder(root), name), "rb")
                    for name in os.listdir(variants_folder(root))}
        on_gaps(path, root, write=True)
        assert {name: read(os.path.join(folder, name), "rb") for name in names} == first, "reports differ on a rerun"
        assert {name: read(os.path.join(variants_folder(root), name), "rb")
                for name in os.listdir(variants_folder(root))} == variants, "variants differ on a rerun"


def test_headroom_a_second_write_leaves_the_variants_alone():
    """A second write_variants of the same config does not touch the files, even one that another process has open.

    The raw and the coarse run are meant to go side by side, and write the same variants (the texts do not depend on
    the truth) into one folder. On Windows os.replace onto a file that is open elsewhere is a PermissionError, so a
    file that already holds the bytes is left alone: the modification times, set back to 2001 so that a coarse clock
    cannot hide a rewrite, are as they were, no temporary file is left, and the call succeeds with every variant open
    for reading. A config that changes (its seed here) is a text that changes, and is written. write_text says which
    it did.
    """
    with tempfile.TemporaryDirectory() as root:
        base = load_config(write_text(root, "full.toml", FULL))
        written = headroom.write_variants(base, root)
        folder = variants_folder(root)
        paths = [path for _, path in written]
        assert len(paths) == 5
        old = 1_000_000_000 * 10 ** 9           # nanoseconds: 2001-09-09
        for path in paths:
            os.utime(path, ns=(old, old))
        with contextlib.ExitStack() as holding:
            for path in paths:
                holding.enter_context(open(path, "rb"))
            again = headroom.write_variants(base, root)
        assert again == written
        assert [os.stat(path).st_mtime_ns for path in paths] == [old] * 5, "a file that held the bytes was rewritten"
        assert sorted(os.listdir(folder)) == sorted(os.path.basename(path) for path in paths), os.listdir(folder)
        assert headroom.write_text(paths[0], read(paths[0])) is False, "same bytes: not written"

        changed = headroom.write_variants(dataclasses.replace(base, seed=base.seed + 1), root)
        assert [path for _, path in changed] == paths
        assert all(os.stat(path).st_mtime_ns != old and tomllib.loads(read(path))["seed"] == base.seed + 1
                   for path in paths), "a changed config must be written"
        assert headroom.write_text(paths[0], "seed = 9\n") is True and read(paths[0]) == "seed = 9\n"


def test_headroom_a_write_that_loses_a_race_for_the_same_bytes_is_not_an_error():
    """A rename refused because the other run got there first, with the same bytes, is fine; with other bytes it is not.

    Two runs that find no file both write one, and the second rename can be refused (Windows: the first run's loader has
    the file open). os.replace is made to do exactly that here, after putting the other run's bytes in place. Where they
    are the bytes that were to be written, write_text returns False and leaves no temporary file; where they are not,
    the PermissionError is raised, since the file is not what this run meant and the runner would score another config.
    """
    real = os.replace

    def beaten_by(content):
        def replace(source, destination):
            with open(destination, "wb") as handle:     # the other run wrote its file...
                handle.write(content)
            raise PermissionError(13, "the file is in use")  # ...and has it open: this run's rename is refused
        return replace

    with tempfile.TemporaryDirectory() as root:
        folder = os.path.join(root, "configs")
        path = os.path.join(folder, "x.toml")
        text = "seed = 1\n"
        os.replace = beaten_by(text.encode("utf-8"))
        try:
            assert headroom.write_text(path, text) is False
        finally:
            os.replace = real
        assert read(path) == text and os.listdir(folder) == ["x.toml"], os.listdir(folder)

        os.remove(path)
        os.replace = beaten_by(b"seed = 2\n")
        try:
            expect_error(PermissionError, lambda: headroom.write_text(path, text), "in use")
        finally:
            os.replace = real
        assert os.listdir(folder) == ["x.toml"], "no temporary file is left: " + str(os.listdir(folder))


def test_headroom_visible_oracle_follows_the_degraded_ball():
    """Under the gaps profile oracle_home_visible speaks on the live frames where the degraded ball is, not the clean.

    The runner serves a degraded Atd, whose ball is missing where ours was (the Calibration's masks), and the oracle
    reads run.inputs.atd.ball_present: its opinions are exactly the truth's H and A frames on which that ball is
    present, with the truth's value (1 on H, 0 on A). The control is that this set is not the one of the clean ball on
    these matches, so that an oracle that read the clean tracking would fail it. oracle_alive does not look at the
    ball: it speaks on every graded frame.
    """
    match = random_match("RND001")
    atd = benchmark_run.make_atd(match, "gaps", 0, synthetic_calibration())
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=0)
    codes = truth.truth_codes(match)[0]
    live = (codes == H) | (codes == A)
    degraded, clean = np.asarray(inputs.atd.ball_present), np.asarray(match_inputs.clean_atd(match).ball_present)
    assert ((live & degraded) != (live & clean)).any(), "the gaps profile must change the ball, or this proves nothing"

    made = headroom.oracles({"RND001": codes})
    run_context = RunContext(inputs=inputs, seed=0)
    home = made["oracle_home_visible"](run_context).home
    assert np.array_equal(~np.isnan(home), live & degraded), "it must speak exactly where the degraded ball is"
    assert not (~np.isnan(home) & ~degraded).any() and not (~np.isnan(home) & clean & ~degraded).any()
    assert np.array_equal(home[live & degraded], np.where(codes[live & degraded] == H, 1.0, 0.0))
    alive = made["oracle_alive"](run_context).alive
    assert np.array_equal(~np.isnan(alive), codes != X), "oracle_alive does not depend on the ball"


def test_headroom_command_line_prints_the_table_and_the_report_lines():
    """main() runs CONFIG and its variants, prints the runner's table with CONFIG as the reference, says what it wrote.

    Served by random matches through dfl.load_match, on the clean profile (no Calibration to measure). With --write the
    five variants' reports are named; without it the output says that none was written. A config that cannot be run is
    "FAILED    <sentence>" and exit code 1, not a traceback.
    """
    def serve(match_id):
        return random_match(match_id, seed=len(match_id))

    with tempfile.TemporaryDirectory() as root, loading(serve):
        path = write_text(root, "hmm_free.toml", HMM)
        arguments = [path, "--profile", "clean", "--matches", "J03WOH", "J03WMX", "--data-dir", root]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = headroom.main(arguments + ["--write"])
        out = shown.getvalue()
        assert code == 0, out
        assert "against hmm_free (the first config" in out, out
        for name in STEMS:
            assert "hmm_free__" + name in out, (name, out)
        assert sum(line.startswith("report") for line in out.split("\n")) == 5, out

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = headroom.main(arguments)
        assert code == 0 and "not written" in shown.getvalue(), shown.getvalue()

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = headroom.main([write_text(root, "no_home.toml", NO_HOME), "--profile", "clean", "--data-dir", root])
        assert code == 1 and shown.getvalue().startswith("FAILED") and "nearest_player" in shown.getvalue(), (
            code, shown.getvalue())
        assert headroom.DEFAULT_PROFILE == benchmark_run.DEFAULT_PROFILE


def test_headroom_command_line_drops_match_only_sources_when_asked():
    """--drop-match-only reaches the runner: a config with a match-only source (main.toml's Smart Tagging) runs with it.

    The probe source is match-only and asks for the video's homography, which a benchmark match does not have. Without
    the flag the command fails with a sentence that names the flag (exit 1); with it the run succeeds (exit 0), the
    variants are made from the config as written, and the output says what was dropped.
    """
    def serve(match_id):
        return random_match(match_id, seed=len(match_id))

    with match_only_source(needs_homography=True), tempfile.TemporaryDirectory() as root, loading(serve):
        path = write_text(root, "uses_tags.toml", PROBE_CONFIG.format(name="tags"))
        arguments = [path, "--profile", "clean", "--matches", "J03WOH", "--data-dir", root, "--variants", "alive"]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = headroom.main(arguments)
        assert code == 1 and "--drop-match-only" in shown.getvalue(), (code, shown.getvalue())

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            code = headroom.main(arguments + ["--drop-match-only"])
        out = shown.getvalue()
        assert code == 0, out
        assert "match-only" in out and "tags" in out and "uses_tags__alive" in out, out


# -- the plus_ variants and the gaps oracle (P18) ---------------------------------------------------------------------

def test_headroom_gaps_oracle_speaks_only_where_the_ball_is_missing():
    """oracle_home_gaps: 1 on the truth's H and 0 on its A, on the frames where the degraded ball is missing; NaN else.

    The complement of oracle_home_visible within the live frames: together they are oracle_home_all. alive is NaN. The
    controls are that both the missing and the seen live frames exist, so a swapped condition cannot pass.
    """
    match = random_match("RND001")
    atd = benchmark_run.make_atd(match, "gaps", 0, synthetic_calibration())
    inputs = match_inputs.MatchInputs(match, atd, profile="gaps", seed=0)
    codes = truth.truth_codes(match)[0]
    live = (codes == H) | (codes == A)
    seen = np.asarray(inputs.atd.ball_present)
    assert (live & seen).any() and (live & ~seen).any(), "the check needs live frames with the ball seen and missing"
    made = headroom.oracles({"RND001": codes})
    run_context = RunContext(inputs=inputs, seed=0)
    evidence = made["oracle_home_gaps"](run_context)
    check_evidence(dataclasses.replace(evidence, name="oracle_home_gaps"), inputs.n_frames)
    assert np.array_equal(~np.isnan(evidence.home), live & ~seen), "it must speak exactly where the ball is missing"
    assert np.array_equal(evidence.home[live & ~seen], np.where(codes[live & ~seen] == H, 1.0, 0.0))
    assert np.isnan(evidence.alive).all()
    visible, everywhere = made["oracle_home_visible"](run_context).home, made["oracle_home_all"](run_context).home
    assert np.array_equal(np.where(np.isnan(visible), evidence.home, visible), everywhere, equal_nan=True)


def test_headroom_plus_variants_add_the_oracles_after_the_configs_own_sources():
    """A config with no nearest_player (main.toml's shape) gets the plus_ variants; the classic ones still refuse it.

    The stems are <stem>__<suffix> in PLUS_VARIANTS' order; the evidence is the config's own kinds, oracle_alive if
    the variant is an alive one, then the added oracle, the added ones at weight 1.0, and seed, scope, decoder and
    refiners are the config's.
    """
    suffixes = [suffix for suffix, *_ in headroom.PLUS_VARIANTS]
    assert tuple(suffixes) == PLUS_STEMS, suffixes
    with tempfile.TemporaryDirectory() as root:
        config = load_config(write_text(root, "two.toml", TWO_SOURCES))
        texts = headroom.variant_texts(config, variants=suffixes)
        assert [stem for stem, _ in texts] == [config.stem + "__" + suffix for suffix in suffixes]
        own = [component.kind for component in config.evidence]
        with oracles_registered():
            for (stem, text), (suffix, alive, kind, _) in zip(texts, headroom.PLUS_VARIANTS):
                variant = load_config(write_text(root, stem + ".toml", text))
                assert [c.kind for c in variant.evidence] == own + (["oracle_alive"] if alive else []) + [kind], suffix
                assert list(variant.evidence[:len(own)]) == list(config.evidence), suffix
                assert all(c.weight == 1.0 for c in variant.evidence[len(own):]), suffix
                assert (variant.seed, variant.scope, variant.decoder, variant.refine) == (
                    config.seed, config.scope, config.decoder, config.refine), suffix
        headroom.check_config(config, suffixes)
        expect_error(benchmark_run.BenchmarkError, lambda: headroom.check_config(config), "nearest_player")
        assert headroom.variant_specs(None) == headroom.variant_specs(STEMS), "variants=None is still the classic five"


def test_headroom_plus_variants_add_and_classic_ones_still_replace():
    """With a nearest_player, home_all puts the oracle in its place; plus_home_all keeps it and appends the oracle."""
    with tempfile.TemporaryDirectory() as root:
        base = load_config(write_text(root, "hmm_free.toml", HMM))
        texts = dict(headroom.variant_texts(base, variants=("home_all", "plus_home_all", "alive_plus_home_gaps")))
        with oracles_registered():
            kinds = {stem: [c.kind for c in load_config(write_text(root, stem + ".toml", text)).evidence]
                     for stem, text in texts.items()}
        assert kinds == {"hmm_free__home_all": ["oracle_home_all"],
                         "hmm_free__plus_home_all": ["nearest_player", "oracle_home_all"],
                         "hmm_free__alive_plus_home_gaps": ["nearest_player", "oracle_alive", "oracle_home_gaps"]}, kinds


def test_headroom_alive_plus_home_all_is_perfect():
    """alive_plus_home_all scores 1.0 on all three metrics, though the config keeps its own source; plus_home_gaps not.

    The kept source is soft (confidence 0.9, about 2.2 nats): a hard 0 / 1 nearest_player is as loud as the oracle, 13.8
    nats at eps 1e-6, and where it is wrong the two tie.
    """
    soft = HMM.replace('kind = "nearest_player"', 'kind = "nearest_player"' + chr(10) + 'confidence = 0.9')
    with tempfile.TemporaryDirectory() as root:
        result = on_gaps(write_text(root, "hmm_free.toml", soft), root,
                         variants=["alive_plus_home_all", "plus_home_gaps"])
        pooled = pooled_of(result)
        perfect = pooled["hmm_free__alive_plus_home_all"]
        assert perfect["spell_f1_mean"] == perfect["macro_f1"] == perfect["alive_dead_f1"] == 1.0, perfect
        assert pooled["hmm_free__plus_home_gaps"]["macro_f1"] < 1.0, pooled["hmm_free__plus_home_gaps"]
