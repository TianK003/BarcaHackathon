"""Checks of R021's grid (pipeline/benchmark/team_grid.py): the configs, their order and the bar's verdict.

* Stage 1's eight files are main.toml's text plus one `team_model` block, the description's amendment and, for `gaps`,
  the one scope line: the same `load_config` result but for those, and a line diff that holds nothing else
  (test_team_grid_stage1_is_main_verbatim_plus_the_block);
* Stage 2's nine files change two decoder lines and nothing else, and the one with Stage 1's own values is its text to
  the byte (test_team_grid_stage2_changes_two_decoder_lines);
* the names and the tie order are the declared ones, and the union of 8 and 9 paths is 16 stems with Stage 1's kept
  (test_team_grid_names_and_tie_order);
* an anchor found other than once is refused, and the same input writes the same bytes twice, from LF or CRLF text
  (test_team_grid_refuses_missing_or_repeated_anchors, test_team_grid_writes_the_same_bytes_twice);
* the verdict is the declared arithmetic with no rounding: -0.0003 under a threshold is a miss, p 0.0500 passes and 0.0501
  fails, each veto fires alone, a LOMO with 5 wins fails, and Stage 2b is pre-approved only as declared
  (test_team_grid_verdict_is_mechanical).

Synthetic only: the configs are made from the repository's main.toml, read as a file, and no result of any run is used.
"""

import contextlib
import functools
import io
import json
import os
import tempfile

from pipeline.benchmark import choose
from pipeline.benchmark import run as benchmark_run
from pipeline.benchmark import team_grid as grid
from pipeline.benchmark.checks.common import synthetic_calibration, synthetic_matches, yardstick_path
from pipeline.config import load_config
from pipeline.selftest import expect_error, quiet, write_text

WEIGHT_STEMS = ("w0100", "w0250", "w0500", "w1000")


def main_lines():
    with open(grid.DEFAULT_MAIN, newline="", encoding="utf-8") as handle:
        return handle.read().replace("\r\n", "\n").split("\n")[:-1]


def read_lines(path):
    with open(path, "rb") as handle:
        data = handle.read()
    assert b"\r" not in data, "a generated config is LF"
    return data.decode("utf-8").split("\n")[:-1]


def test_team_grid_stage1_is_main_verbatim_plus_the_block():
    main = load_config(grid.DEFAULT_MAIN)
    lines = main_lines()
    with tempfile.TemporaryDirectory() as root:
        paths = grid.stage1(grid.DEFAULT_MAIN, root)
        assert len(paths) == 8
        for path in paths:
            stem = os.path.basename(path)[:-5]
            weight = float(int(stem.split("_")[1][1:]) / 1000)
            scope = stem.split("_")[2]
            config = load_config(path)
            assert config.stem == stem and config.seed == main.seed and config.scope == main.scope
            assert config.decoder == main.decoder and config.refine == main.refine, "decoder and shift are main's"
            kinds = [part.kind for part in config.evidence]
            assert kinds == [part.kind for part in main.evidence] + ["team_model"], "the block is added last"
            for ours, theirs in zip(config.evidence[:-1], main.evidence):
                if ours.kind == "shape_possession":
                    assert ours.params == dict(theirs.params, scope=scope) and ours.weight == theirs.weight
                else:
                    assert ours == theirs, ours.kind
            team = config.evidence[-1]
            assert team.weight == weight and team.params == {"model": grid.MODEL, "min_prob": 0.02}, stem
            assert config.description == main.description + " + team_model (P19, R021) at weight {}; " \
                "shape_possession scope {}".format(repr(weight), scope)
            # the text: main's lines, in order, with only the three intended differences
            text = read_lines(path)
            assert len(text) == len(lines) + 6, "six new lines: the block and its blank line"
            at = text.index('kind = "team_model"') - 1
            assert text[at] == "[[evidence]]" and text[at + 5] == "" and text[at + 1:at + 5][2].startswith("model = ")
            rest = text[:at] + text[at + 6:]
            changed = [index for index, (a, b) in enumerate(zip(rest, lines)) if a != b]
            wanted = [index for index, line in enumerate(lines) if line.startswith("description = ")]
            if scope == "gaps":
                wanted += [index for index, line in enumerate(lines) if line.startswith('scope = "all"')]
            assert changed == sorted(wanted), (stem, changed, wanted)
            if scope == "gaps":
                (index,) = [i for i in changed if lines[i].startswith('scope = "all"')]
                assert rest[index] == lines[index].replace('scope = "all"', 'scope = "gaps"')
            assert text[-1] == lines[-1] and at < text.index("[decoder]")


def test_team_grid_stage2_changes_two_decoder_lines():
    with tempfile.TemporaryDirectory() as root:
        (chosen,) = [p for p in grid.stage1(grid.DEFAULT_MAIN, root) if p.endswith("r021a_w0500_gaps.toml")]
        base = read_lines(chosen)
        paths = grid.stage2(chosen, root)
        assert len(paths) == 9
        for path in paths:
            stem = os.path.basename(path)[:-5]
            switch, minimum = int(stem.split("_")[1][1:]), int(stem.split("_")[2][2:])
            text = read_lines(path)
            changed = [i for i, (a, b) in enumerate(zip(text, base)) if a != b]
            assert len(text) == len(base) and len(changed) <= 2
            config, original = load_config(path), load_config(chosen)
            assert config.evidence == original.evidence and config.refine == original.refine
            assert config.decoder.params["switch_team"] == switch and config.decoder.params["min_possession"] == minimum
            assert dict(config.decoder.params, switch_team=0, min_possession=0) == dict(
                original.decoder.params, switch_team=0, min_possession=0), "switch_dead and min_dead are main's"
            for i in changed:
                assert text[i].startswith(("switch_team", "min_possession")), text[i]
        with open(chosen, "rb") as a, open(os.path.join(root, "r021b_t04_mp035.toml"), "rb") as b:
            assert a.read() == b.read(), "Stage 2's copy of Stage 1's choice is its text to the byte"
        other = grid.stage2(chosen, root, prefix="r021c", switch_team=(8, 16, 32), min_possession=(75, 100, 150))
        assert grid.stems(other)[0] == "r021c_t08_mp075" and grid.stems(other)[-1] == "r021c_t32_mp150"
        assert load_config(other[-1]).decoder.params["min_possession"] == 150


def test_team_grid_names_and_tie_order():
    with tempfile.TemporaryDirectory() as root:
        one = grid.stage1(grid.DEFAULT_MAIN, root)
        assert grid.stems(one) == ["r021a_{}_{}".format(w, s) for w in WEIGHT_STEMS for s in ("all", "gaps")]
        chosen = one[5]                                                      # r021a_w0500_gaps
        two = grid.stage2(chosen, root)
        assert grid.stems(two) == ["r021b_t{:02d}_mp{:03d}".format(t, m) for t in (4, 8, 16) for m in (35, 50, 75)]
        union = grid.union_order(one + two)
        assert len(union) == 16 and len(set(union)) == 16
        assert "r021a_w0500_gaps" in union and "r021b_t04_mp035" not in union, "Stage 1's stem is the one kept"
        expected = ["r021a_w0100_all", "r021a_w0100_gaps", "r021a_w0250_all", "r021a_w0250_gaps", "r021a_w0500_all",
                    "r021a_w0500_gaps", "r021b_t04_mp050", "r021b_t04_mp075", "r021b_t08_mp035", "r021b_t08_mp050",
                    "r021b_t08_mp075", "r021b_t16_mp035", "r021b_t16_mp050", "r021b_t16_mp075",
                    "r021a_w1000_all", "r021a_w1000_gaps"]
        assert union == expected, union
        # Stage 2's path first would keep Stage 2's stem instead: the order of the arguments is the rule
        assert "r021b_t04_mp035" in grid.union_order(two + one)
        # Stage 2b's configs, if they ever run, are de-duplicated against the stages before them
        again = grid.stage2(chosen, root, prefix="r021c", switch_team=(8, 16, 32), min_possession=(75, 100, 150))
        full = grid.union_order(one + two + again)
        assert len(full) == 16 + 9 - 2 and "r021c_t08_mp075" not in full and "r021c_t16_mp075" not in full
        assert "r021c_t32_mp150" in full
        # a config that is not a team_model config has no place in the union
        expect_error(ValueError, lambda: grid.union_order([grid.DEFAULT_MAIN]), "team_model", "shape_possession")


def test_team_grid_refuses_missing_or_repeated_anchors():
    lines = main_lines()
    with tempfile.TemporaryDirectory() as root:
        def variant(name, edit):
            return write_text(root, name, "\n".join(edit(list(lines))) + "\n")

        def duplicate(prefix):
            def edit(text):
                index = next(i for i, line in enumerate(text) if line.startswith(prefix))
                text.insert(index, text[index])
                return text
            return edit

        def drop(prefix):
            return lambda text: [line for line in text if not line.startswith(prefix)]

        for name, edit, fragment in (
                ("two_scope.toml", duplicate('scope = "all"'), "scope"),
                ("no_scope.toml", drop('scope = "all"'), "scope"),
                ("two_desc.toml", duplicate("description = "), "description"),
                ("no_desc.toml", drop("description = "), "description"),
                ("no_decoder.toml", lambda t: [l for l in t if l.strip() != "[decoder]"], "[decoder]"),
                ("two_decoder.toml", lambda t: t + ["", "[decoder]"], "[decoder]"),
                ("no_shape.toml", lambda t: [l.replace("shape_possession", "dead_ball_model") for l in t],
                 "shape_possession")):
            path = variant(name, edit)
            expect_error(ValueError, lambda: grid.stage1(path, os.path.join(root, "out")), "team_grid", fragment,
                         "exactly 1")
        assert not os.path.exists(os.path.join(root, "out")), "nothing is written before the anchors are checked"
        (good,) = [p for p in grid.stage1(grid.DEFAULT_MAIN, root) if p.endswith("w0500_all.toml")]
        text = read_lines(good)
        for name, edit, fragment in (
                ("no_switch.toml", lambda t: [l for l in t if not l.startswith("switch_team")], "switch_team"),
                ("two_possession.toml", lambda t: t + ["min_possession = 3"], "min_possession")):
            path = write_text(root, name, "\n".join(edit(list(text))) + "\n")
            expect_error(ValueError, lambda: grid.stage2(path, os.path.join(root, "out2")), "team_grid", fragment)


def test_team_grid_writes_the_same_bytes_twice():
    with tempfile.TemporaryDirectory() as root:
        first = grid.stage1(grid.DEFAULT_MAIN, os.path.join(root, "a"))
        second = grid.stage1(grid.DEFAULT_MAIN, os.path.join(root, "b"))
        for a, b in zip(first, second):
            with open(a, "rb") as x, open(b, "rb") as y:
                assert x.read() == y.read(), os.path.basename(a)
        # main.toml with CRLF line ends gives the same files: the output does not depend on the checkout's endings
        crlf = os.path.join(root, "crlf.toml")
        with open(crlf, "wb") as handle:
            handle.write("\r\n".join(main_lines()).encode("utf-8") + b"\r\n")
        third = grid.stage1(crlf, os.path.join(root, "c"))
        for a, b in zip(first, third):
            with open(a, "rb") as x, open(b, "rb") as y:
                assert x.read() == y.read(), os.path.basename(a)
        before = [os.path.getmtime(p) for p in first]
        grid.stage1(grid.DEFAULT_MAIN, os.path.join(root, "a"))
        assert before == [os.path.getmtime(p) for p in first], "a file that already holds the bytes is not rewritten"


STEMS = ["a", "b", "c", "d"]                 # a stand-in union of stems, the stage's choice "c" among them
YARDSTICKS = (50, 25, 0, 125)                # the reference first, as a grid's process lists main.toml first
SEEDS = 2


@functools.lru_cache(maxsize=1)
def real_documents():
    """The three documents as choose.py writes them (its own `choose` and the writer's `plain`), on synthetic reports.

    {"profile": the runner's profile record, "stage", "union", "context", "stems", "chosen", "reference", "root"}: the
    yardstick configs under the gaps profile with the synthetic Calibration, so that the shape of what team_grid reads
    is the shape choose.py really writes and cannot drift from it in a fixture. Their numbers mean nothing.
    """
    paths = [yardstick_path(width) for width in YARDSTICKS]
    stems = ["nearest_w{}".format(width) for width in YARDSTICKS[1:]]
    root = tempfile.mkdtemp()
    results = benchmark_run.evaluate_truths(paths, profile="gaps", matches=synthetic_matches(), seeds=SEEDS,
                                            calibration=synthetic_calibration(), data_dir=root, log=quiet)
    for result in results.values():
        benchmark_run.write_reports(result, data_dir=root)
    options = dict(profile="gaps", data_dir=root, match_ids=sorted(synthetic_matches()), seeds=SEEDS)
    stage = choose.choose(stems[:2], "nearest_w50", **options)
    union = choose.choose(stems, "nearest_w50", **options)
    context = choose.choose([stage["chosen"]], "nearest_w50", **options)
    written = {}
    for name, made in (("stage", stage), ("union", union), ("context", context)):
        path = os.path.join(root, name + ".json")
        benchmark_run.write_json(path, benchmark_run.plain(made))          # as choose.py --out writes it
        with open(path, encoding="utf-8") as handle:
            written[name] = json.load(handle)
    return dict(written, profile=written["stage"]["profile"], stems=stems, chosen=stage["chosen"],
                reference="nearest_w50", root=root)


def profile_record(name):
    """The runner's profile record of `real_documents`, under another name: what a document of that profile holds."""
    return dict(real_documents()["profile"], name=name)


def choice(delta, p, coarse=0.0, alive=0.0, lomo_mean=0.01, wins=7, context_raw=0.0, context_coarse=0.0):
    """The three choose.py --out documents, with what the verdict reads and what it checks them by."""
    meta = {"reference": "main", "metric": "spell", "profile": profile_record("tracking")}
    stage = dict(meta, chosen="c", order=["a", "b", "c"],
                 versus={"raw": {"pooled_delta": delta, "p_sign_flip": p}, "coarse": {"pooled_delta": coarse}},
                 rows={"c": {"alive_dead": round(0.9731 + alive, 9)}, "main": {"alive_dead": 0.9731}})
    union = dict(meta, chosen="d", order=list(STEMS), lomo={"rule": {"raw": {"mean": lomo_mean, "wins": wins}}})
    context = dict(meta, profile=profile_record("context"), chosen="c", order=["c"],
                   versus={"raw": {"pooled_delta": context_raw}, "coarse": {"pooled_delta": context_coarse}})
    return stage, union, context


def V(*arguments, **keywords):
    return grid.verdict(*choice(*arguments, **keywords), STEMS)


def test_team_grid_verdict_reads_the_files_choose_writes():
    """The verdict on documents that choose.py really wrote: its shapes, not the fixtures', pass the checks."""
    real = real_documents()
    assert isinstance(real["stage"]["profile"], dict) and real["stage"]["profile"]["name"] == "gaps", real["stage"]
    # these files are a gaps run against nearest_w50: refused under the declared tracking/context/main
    expect_error(ValueError, lambda: grid.verdict(real["stage"], real["union"], real["context"], real["stems"]),
                 "refused", "reference")
    saved = (grid.PROFILE, grid.CONTEXT, grid.REFERENCE, grid.union_order)
    grid.PROFILE, grid.CONTEXT, grid.REFERENCE = "gaps", "gaps", real["reference"]
    grid.union_order = lambda paths: list(real["stems"])
    try:
        result = grid.verdict(real["stage"], real["union"], real["context"], real["stems"])
        assert result["chosen"] == real["chosen"]
        assert set(result) == {"chosen", "raw", "vetoes", "lomo", "passes", "retry_2b"}
        assert result["raw"]["delta"] == real["stage"]["versus"]["raw"]["pooled_delta"]
        expect_error(ValueError, lambda: grid.verdict(real["stage"], real["union"], real["context"],
                                                      real["stems"][::-1]), "refused", "declared")
        grid.PROFILE = "tracking"
        expect_error(ValueError, lambda: grid.verdict(real["stage"], real["union"], real["context"], real["stems"]),
                     "stage", "'gaps'", "'tracking'")
        grid.PROFILE = "gaps"
        # the command on the real files: the verdict is printed and the exit code is 0
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = grid.main(["verdict", "--stage", os.path.join(real["root"], "stage.json"),
                              "--union", os.path.join(real["root"], "union.json"),
                              "--context", os.path.join(real["root"], "context.json"),
                              "--union-configs", "unused.toml"])
        assert code == 0 and "verdict   chosen {}".format(real["chosen"]) in buffer.getvalue(), buffer.getvalue()
    finally:
        grid.PROFILE, grid.CONTEXT, grid.REFERENCE, grid.union_order = saved


def test_team_grid_verdict_refuses_the_wrong_files():
    good = choice(0.01, 0.01)
    assert grid.verdict(*good, STEMS)["chosen"] == "c"

    def refused(edit, *fragments, stems=STEMS):
        stage, union, context = (dict(document) for document in choice(0.01, 0.01))
        edit(stage, union, context)
        expect_error(ValueError, lambda: grid.verdict(stage, union, context, stems), "team_grid", "refused", *fragments)

    for position, name in enumerate(("stage", "union", "context")):
        def wrong_reference(*documents, position=position):
            documents[position]["reference"] = "r013_retune"
        refused(wrong_reference, name, "reference")

        def wrong_profile(*documents, position=position):
            documents[position]["profile"] = profile_record("gaps")
        refused(wrong_profile, name, "profile")

        def bare_name(*documents, position=position):               # a shape that choose.py does not write
            documents[position]["profile"] = documents[position]["profile"]["name"]
        refused(bare_name, name, "profile")

        def wrong_metric(*documents, position=position):
            documents[position]["metric"] = "macro"
        refused(wrong_metric, name, "metric")

        def missing_key(*documents, position=position):
            del documents[position]["order"]
        refused(missing_key, name, "choose.py")
    refused(lambda s, u, c: u.update(profile=profile_record("context")), "union", "tracking")   # veto file as union
    refused(lambda s, u, c: c.update(profile=profile_record("tracking")), "context", "context")  # bar file as context
    refused(lambda s, u, c: c.update(chosen="d", order=["d"]), "context", "c")           # another config's context run
    refused(lambda s, u, c: c.update(order=["c", "d"]), "context")
    refused(lambda s, u, c: s.update(chosen="z"), "z")                                  # a choice outside the union
    refused(lambda s, u, c: u.update(order=["a", "b", "c"]), "union", "declared")        # a union missing a config
    refused(lambda s, u, c: u.update(order=["a", "c", "b", "d"]), "union", "order")      # the declared order, changed
    refused(lambda s, u, c: None, "union", stems=["a", "b", "c"])                        # declared stems differ
    refused(lambda s, u, c: u.update(order=["a", "b", "c", "d", "d"]), "union")
    # a CLI refusal is an exit code of 1 and a sentence, not a verdict
    with tempfile.TemporaryDirectory() as root:
        paths = []
        for name, document in zip(("stage", "union", "context"), choice(0.01, 0.01)):
            document = dict(document, reference="other") if name == "union" else document
            paths.append(write_text(root, name + ".json", json.dumps(document)))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = grid.main(["verdict", "--stage", paths[0], "--union", paths[1], "--context", paths[2],
                              "--union-configs"] + grid.stage1(grid.DEFAULT_MAIN, os.path.join(root, "grid")))
        assert code == 1 and "FAILED" in buffer.getvalue() and "refused" in buffer.getvalue(), buffer.getvalue()
        assert "PASSES" not in buffer.getvalue() and "raw delta" not in buffer.getvalue(), "no verdict is printed"


def test_team_grid_stage1_validates_scopes_before_writing():
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "out")
        expect_error(ValueError, lambda: grid.stage1(grid.DEFAULT_MAIN, out, scopes=("all", "bogus")), "team_grid",
                     "scope", "bogus")
        assert not os.path.exists(out), "no file is written for a grid that is refused"
        # a config whose shape_possession has no scope has no place in the union: the scope is never guessed
        (written,) = [p for p in grid.stage1(grid.DEFAULT_MAIN, root) if p.endswith("w0500_all.toml")]
        text = "\n".join(read_lines(written)) + "\n"
        bare = write_text(root, "no_scope_key.toml", text.replace('scope = "all"', "# no scope", 1))
        expect_error(ValueError, lambda: grid.union_order([bare]), "team_grid", "scope")


def test_team_grid_verdict_is_mechanical():
    result = V(0.0050, 0.0500)
    assert result["passes"] is True and result["retry_2b"] is False and result["chosen"] == "c"
    assert result["raw"]["passes"] and result["lomo"]["passes"] and not any(v["fires"] for v in result["vetoes"].values())
    assert V(0.0047, 0.01)["raw"]["passes"] is False, "-0.0003 under the threshold is a miss"
    assert V(0.0050, 0.0501)["raw"]["passes"] is False, "p 0.0500 passes and 0.0501 fails"
    assert V(0.02, 0.0500)["raw"]["passes"] is True
    # each veto fires alone, at -0.0051 and not at -0.005
    for name, key in (("coarse", "coarse"), ("context_raw", "context_raw"), ("context_coarse", "context_coarse"),
                      ("alive_dead", "alive")):
        hit = V(0.01, 0.01, **{key: -0.0051})
        assert [n for n, v in hit["vetoes"].items() if v["fires"]] == [name], name
        assert hit["passes"] is False
        assert V(0.01, 0.01, **{key: -0.005})["vetoes"][name]["fires"] is False, name   # the edge is not a miss
    # the LOMO: 5 wins fails, 6 passes, a zero or negative mean fails
    assert V(0.01, 0.01, wins=5)["lomo"]["passes"] is False
    assert V(0.01, 0.01, wins=5)["passes"] is False
    assert V(0.01, 0.01, wins=6)["lomo"]["passes"] is True
    assert V(0.01, 0.01, lomo_mean=0.0)["lomo"]["passes"] is False
    assert V(0.01, 0.01, lomo_mean=-0.001)["lomo"]["passes"] is False
    # Stage 2b is pre-approved only if raw and the LOMO pass and only a coarse or context veto fires
    assert V(0.01, 0.01, coarse=-0.01)["retry_2b"] is True
    assert V(0.01, 0.01, context_raw=-0.01)["retry_2b"] is True
    assert V(0.01, 0.01, context_coarse=-0.01, coarse=-0.01)["retry_2b"] is True
    assert V(0.01, 0.01, coarse=-0.01, alive=-0.01)["retry_2b"] is False
    assert V(0.01, 0.01, alive=-0.01)["retry_2b"] is False
    assert V(0.0047, 0.01, coarse=-0.01)["retry_2b"] is False, "the raw bar must pass"
    assert V(0.01, 0.0501, coarse=-0.01)["retry_2b"] is False
    assert V(0.01, 0.01, coarse=-0.01, wins=5)["retry_2b"] is False, "the LOMO must pass"
    assert V(0.01, 0.01)["retry_2b"] is False, "nothing to retry when it passes"
    lines = grid.render_verdict(V(0.0047, 0.01, coarse=-0.01))
    assert any("NOT met" in line for line in lines) and any("FIRES" in line for line in lines)
