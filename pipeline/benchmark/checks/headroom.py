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
from pipeline.benchmark.checks.common import (loading, random_match, read, synthetic_calibration, synthetic_matches)
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
    assert sorted(made) == ["oracle_alive", "oracle_home_all", "oracle_home_visible"], sorted(made)
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


def test_headroom_oracles_are_registered_only_while_a_run_lasts():
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
        assert seen_inside == [["oracle_alive", "oracle_home_all", "oracle_home_visible"]], seen_inside

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


def test_headroom_writes_reports_only_when_asked_and_the_same_bytes_twice():
    """write=True writes one report per config next to the runner's (<stem>.json, <stem>__alive.json, ...); else none.

    The reports are the runner's own (write_reports) with the variants' stems, in <data dir>/runs/benchmark/<profile>/;
    a second run writes the same bytes, the variants' files included, since their folder and text are fixed.
    """
    with tempfile.TemporaryDirectory() as root:
        path = write_text(root, "hmm_free.toml", HMM)
        on_gaps(path, root)
        assert not os.path.exists(os.path.join(root, "runs", "benchmark", "gaps")), "no report unless asked for"
        lines = []
        on_gaps(path, root, write=True, log=lines.append)
        folder = os.path.join(root, "runs", "benchmark", "gaps")
        names = sorted(["hmm_free.json"] + ["hmm_free__" + name + ".json" for name in STEMS])
        suffix = ".m-RND001-RND002"          # the synthetic matches are not the seven: the runner's own rule
        names = sorted(name[:-len(".json")] + suffix + ".s1.json" for name in names)
        assert sorted(os.listdir(folder)) == names, sorted(os.listdir(folder))
        assert sum(line.startswith("report") for line in lines) == 6, lines
        first = {name: read(os.path.join(folder, name), "rb") for name in names}
        variants = {name: read(os.path.join(variants_folder(root), name), "rb")
                    for name in os.listdir(variants_folder(root))}
        on_gaps(path, root, write=True)
        assert {name: read(os.path.join(folder, name), "rb") for name in names} == first, "reports differ on a rerun"
        assert {name: read(os.path.join(variants_folder(root), name), "rb")
                for name in os.listdir(variants_folder(root))} == variants, "variants differ on a rerun"


def test_headroom_command_line_prints_the_table_and_the_report_lines():
    """main() runs CONFIG and its variants, prints the runner's table with CONFIG as the reference, says what it wrote.

    Served by random matches through dfl.load_match, on the clean profile (no Calibration to measure). With --write the
    six reports are named; without it the output says that none was written. A config that cannot be run is
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
        assert sum(line.startswith("report") for line in out.split("\n")) == 6, out

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
