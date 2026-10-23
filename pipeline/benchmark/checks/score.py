"""Checks of the benchmark's scoring and prediction (pipeline/benchmark/score.py, run.py) and of the yardstick configs.

The one that matters is test_pooled_equals_official_evaluate: whatever later tasks rank, they rank on pooled(), and
pooled() must be the organisers' own pm.evaluate of the matches laid end to end, to the last digit. It is built the
honest way: two matches, two seeds, two halves each, laid end to end with one X row (in truth and prediction alike)
before every block, so that pm.valid_runs of the concatenation is exactly the blocks and the official scorer sees what
score_halves saw. The others are small cases worked out by hand from the rules (never read off what the code printed),
the contract of predict (the same codes as pipeline.run.run, seeded the same way) and of the four configs that stand
for the organisers' baselines. The check that reads a downloaded match is test_dfl_* (--dfl); the DFL files are public
(CC BY 4.0), so printing its aggregates is fine.
"""

import dataclasses
import itertools
import math
import os
import random
import tempfile
from fractions import Fraction

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import score, truth
from pipeline.selftest import (TRIAL_CONFIG, expect_error, make_synthetic_bundle, quiet, synthetic_inputs,
                               write_text)

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
TOLERANCE = 1e-12                      # pooled() against pm.evaluate: the same integer counts, the same arithmetic
CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
YARDSTICKS = {0: "baseline_raw", 25: "baseline_smooth25", 50: "baseline_smooth50", 125: "baseline_smooth125"}
ZERO_CONFUSION = ((0, 0, 0, 0),) * 3


# -- helpers ----------------------------------------------------------------------------------------------------------

def runs(*pairs):
    """int8 codes from (code, length) pairs: runs((H, 3), (D, 2)) is H H H D D."""
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in pairs])


def yardstick(width):
    """The config of one yardstick, loaded the way a run loads it (so its signatures are checked too)."""
    from pipeline.config import load_config
    return load_config(os.path.join(CONFIG_FOLDER, "nearest_w{}.toml".format(width)))


def random_truth(seed, half_rows, gap):
    """(truth codes, graded) of one match: two halves of random runs of H, A and D, `gap` X rows between them.

    Runs are short (5-30 frames) and long (30-200), D one run in five, so that spells of every size are there and
    the greedy matching has ties to break.
    """
    rng = np.random.default_rng([8, seed])
    pieces, graded, row = [], [], 0
    for half, rows in enumerate(half_rows, start=1):
        length = 0
        while length < rows:
            size = int(rng.integers(5, 30)) if rng.uniform() < 0.6 else int(rng.integers(30, 200))
            size = min(size, rows - length)
            pieces.append(runs((int(rng.choice([H, A, H, A, D])), size)))
            length += size
        graded.append((half, row, row + rows))
        row += rows
        if half == 1:
            pieces.append(runs((X, gap)))
            row += gap
    return np.concatenate(pieces), tuple(graded)


def swapped_teams(codes):
    """Every H as A and every A as H, D and X kept: spells that overlap the truth's, none of the right team."""
    return np.where(codes == H, A, np.where(codes == A, H, codes)).astype(np.int8)


def imperfect(codes, graded, rng, lag, flicker):
    """A prediction of `codes`: each half moved `lag` frames later, with single-frame flips and one swapped stretch."""
    pred = codes.copy()
    for _, start, stop in graded:
        half = codes[start:stop]
        moved = np.concatenate([np.full(lag, half[0], dtype=np.int8), half])[:len(half)]
        flips = rng.uniform(size=len(moved)) < flicker
        moved[flips] = rng.choice(np.array([H, A, D], dtype=np.int8), size=int(flips.sum()))
        first = int(rng.integers(0, len(moved) - 60))
        moved[first:first + 50] = swapped_teams(moved[first:first + 50])
        pred[start:stop] = moved
    return pred


def laid_end_to_end(parts):
    """(truth, prediction, blocks) of [(truth, prediction, graded), ...] with one X row, in both, before every part.

    `blocks` are the halves' (start, stop) in the concatenation, which pm.valid_runs of its truth must give back.
    """
    truths, preds, blocks, offset = [], [], [], 0
    for codes, pred, graded in parts:
        truths += [np.array([X], dtype=np.int8), codes]
        preds += [np.array([X], dtype=np.int8), pred]
        offset += 1
        blocks += [(start + offset, stop + offset) for _, start, stop in graded]
        offset += len(codes)
    return np.concatenate(truths), np.concatenate(preds), blocks


# -- the equality with the official scorer ----------------------------------------------------------------------------

def test_pooled_equals_official_evaluate():
    """pooled() of the halves of two matches and two seeds is pm.evaluate of the blocks laid end to end.

    Four (match, seed) blocks of two halves each. Awkward cases on purpose: a prediction that is X at the head of a
    half and in its middle, a half where nothing matches (every team swapped: spells, but tp = 0), a half that is X
    throughout, a prediction that is not X on the truth's gap rows (ungraded: it must count for nothing), and a
    pooled F1 that differs from the mean of the halves' (so a wrong way of pooling shows).
    """
    rng = np.random.default_rng([8, 99])
    matches = {"SYN_A": random_truth(1, (1400, 1100), 25), "SYN_B": random_truth(2, (1200, 1300), 40)}
    cases = [("SYN_A", 0, "abstains at the head and in the middle"), ("SYN_A", 1, "swapped, then silent"),
             ("SYN_B", 0, "plain"), ("SYN_B", 1, "predicts on the gap")]

    parts, scores = {}, []
    for match_id, seed, how in cases:
        codes, graded = matches[match_id]
        pred = imperfect(codes, graded, rng, lag=int(rng.integers(0, 12)), flicker=0.02)
        (_, start1, stop1), (_, start2, stop2) = graded
        if how.startswith("abstains"):
            pred[start1:start1 + 15] = X                  # a half that starts silent
            pred[start1 + 300:start1 + 340] = X           # one that goes quiet in the middle of a possession
            pred[start2 + 200:start2 + 260] = X
        elif how.startswith("swapped"):
            pred[start1:stop1] = swapped_teams(codes[start1:stop1])
            pred[start2:stop2] = X
        elif how.startswith("predicts"):
            pred[stop1:start2] = rng.choice(np.array([H, A, D], dtype=np.int8), size=start2 - stop1)
        parts[(match_id, seed)] = (codes, pred, graded)
        scores += score.score_halves(codes, pred, graded, match_id, seed)

    truth_all, pred_all, blocks = laid_end_to_end(list(parts.values()))
    assert pm.valid_runs(truth_all) == blocks and len(blocks) == len(scores) == 8, (
        "the X rows must leave exactly the 8 halves as the scorer's valid runs")

    official = pm.evaluate(truth_all, pred_all)
    got = score.pooled(scores)
    assert got["halves"] == 8
    assert got["n_gt"] == official["n_gt_spells"] and got["n_pred"] == official["n_pred_spells"], (
        got["n_gt"], got["n_pred"], official["n_gt_spells"], official["n_pred_spells"])
    assert list(got["spell_f1"]) == list(pm.SPELL_IOUS) == list(score.IOUS)
    for iou in pm.SPELL_IOUS:
        assert abs(got["spell_f1"][iou] - official["spell_f1"][iou]["f1"]) < TOLERANCE, iou
    for key in ("spell_f1_mean", "macro_f1", "alive_dead_f1"):
        assert abs(got[key] - official[key]) < TOLERANCE, (key, got[key], official[key])

    # The cases are there, and the comparison is not one of 1.0 with 1.0 or of 0.0 with 0.0.
    assert 0.1 < got["spell_f1_mean"] < 0.95 and got["n_gt"] > 60 and got["n_pred"] > 60, got
    assert any(sum(s.tp) == 0 and s.n_pred > 0 for s in scores), "a half with spells and not one match"
    assert any(s.n_pred == 0 for s in scores), "a half that is X throughout"
    assert any(s.confusion[0][X] + s.confusion[1][X] + s.confusion[2][X] > 0 for s in scores), "abstentions are charged"
    mean_of_halves = float(np.mean([score.pooled([s])["spell_f1_mean"] for s in scores]))
    assert abs(mean_of_halves - official["spell_f1_mean"]) > 1e-3, (
        "the mean of the halves' F1 must differ from the pooled one, or this check cannot tell the two apart")

    # Per match: its halves and seeds, again the official scorer's, on that match's parts alone.
    per_match = score.per_match(scores)
    assert list(per_match) == ["SYN_A", "SYN_B"], list(per_match)
    for match_id, result in per_match.items():
        mine = [parts[(m, seed)] for m, seed, _ in cases if m == match_id]
        truth_one, pred_one, _ = laid_end_to_end(mine)
        expected = pm.evaluate(truth_one, pred_one)
        assert result["halves"] == 4 and result["n_gt"] == expected["n_gt_spells"], (match_id, result["n_gt"])
        assert result["n_pred"] == expected["n_pred_spells"], (match_id, result["n_pred"])
        for key in ("spell_f1_mean", "macro_f1", "alive_dead_f1"):
            assert abs(result[key] - expected[key]) < TOLERANCE, (match_id, key)


# -- what one half is scored on ---------------------------------------------------------------------------------------

def test_score_halves_backfills_inside_half_only():
    """An abstention is filled from its own half: never from the half before, and never across the gap between them.

    Counted by hand. Truth: half 1 is A x 20 then H x 20 (rows 0-39), five gap rows, half 2 is A x 40 (rows 45-84).
    The prediction is perfect in half 1, and in half 2 abstains for its first 10 rows: the head is back-filled with the
    first label the half knows (A), so half 2 is one possession of A, matched at every IoU. Forward-filling across
    the gap would make those 10 rows H (the last label of half 1, or the gap's) and give half 2 a second spell. The
    confusion is of the unfilled codes: those 10 abstentions are 10 misses on the A row.
    """
    truth_codes = np.concatenate([runs((A, 20), (H, 20)), runs((X, 5)), runs((A, 40))])
    graded = ((1, 0, 40), (2, 45, 85))
    for gap in (X, H, A, D):                               # whatever is predicted on the ungraded gap changes nothing
        pred = np.concatenate([runs((A, 20), (H, 20)), runs((gap, 5)), runs((X, 10), (A, 30))])
        first, second = score.score_halves(truth_codes, pred, graded, "M", 7)
        assert first == score.HalfScore("M", 1, 7, 2, 2, (2, 2, 2, 2, 2), ((20, 0, 0, 0), (0, 20, 0, 0), (0, 0, 0, 0)))
        assert second == score.HalfScore("M", 2, 7, 1, 1, (1, 1, 1, 1, 1),
                                         ((0, 0, 0, 0), (0, 30, 0, 10), (0, 0, 0, 0))), (gap, second)
    # (Confusion rows are the truth H, A, D, and columns the prediction H, A, D, X; half 1's row 0 is its H x 20.)

    # The head of the first half is back-filled too (from the first known label of that half).
    head = score.score_halves(runs((A, 20), (H, 20)), runs((X, 5), (A, 15), (H, 20)), ((1, 0, 40),), "M", 0)[0]
    assert head == score.HalfScore("M", 1, 0, 2, 2, (2, 2, 2, 2, 2), ((20, 0, 0, 0), (0, 15, 0, 5), (0, 0, 0, 0))), head

    # In the middle an abstention carries the label before it forward, so a possession that went quiet is one
    # possession: H x 10, X x 10, H x 10 is H x 30.
    quiet_possession = score.score_halves(runs((H, 30)), runs((H, 10), (X, 10), (H, 10)), ((1, 0, 30),), "M", 0)[0]
    assert quiet_possession == score.HalfScore("M", 1, 0, 1, 1, (1, 1, 1, 1, 1),
                                               ((20, 0, 0, 10), (0, 0, 0, 0), (0, 0, 0, 0))), quiet_possession

    # ... and across a hand-over it drags the old label over the new one (CLAUDE.md: at transitions, guess). Truth H x
    # 15 then A x 15; the prediction is H x 10, X x 10, A x 10, which fills to H x 20 then A x 10. H: 15 / 20 = 0.75,
    # matched at every IoU. A: 10 / 15 = 0.667, matched at 0.3 to 0.6 and not at 0.7.
    handover = score.score_halves(runs((H, 15), (A, 15)), runs((H, 10), (X, 10), (A, 10)), ((1, 0, 30),), "M", 0)[0]
    assert handover == score.HalfScore("M", 1, 0, 2, 2, (2, 2, 2, 2, 1),
                                       ((10, 0, 0, 5), (0, 10, 0, 5), (0, 0, 0, 0))), handover

    # A half that is X throughout has no possession and every graded frame a miss.
    silent = score.score_halves(runs((A, 10), (D, 5)), runs((X, 15)), ((1, 0, 15),), "M", 0)[0]
    assert silent == score.HalfScore("M", 1, 0, 1, 0, (0, 0, 0, 0, 0),
                                     ((0, 0, 0, 0), (0, 0, 0, 10), (0, 0, 0, 5))), silent


def test_score_halves_refuses_what_it_cannot_score_like_the_official_scorer():
    """The pooled score is pm.evaluate's only if the blocks are the truth's own valid runs: anything else is refused."""
    truth_codes = np.concatenate([runs((A, 20), (H, 20)), runs((X, 5)), runs((A, 40))])
    pred = np.concatenate([runs((A, 20), (H, 20)), runs((X, 5)), runs((A, 40))])
    graded = ((1, 0, 40), (2, 45, 85))
    assert len(score.score_halves(truth_codes, pred, graded, "M", 0)) == 2
    expect_error(ValueError, lambda: score.score_halves(truth_codes, pred[:-1], graded, "M", 0), "M", "85", "84")
    expect_error(ValueError, lambda: score.score_halves(truth_codes, pred, ((1, 0, 40),), "M", 0), "M", "valid runs",
                 "(45, 85)")
    expect_error(ValueError, lambda: score.score_halves(truth_codes, pred, ((1, 0, 45), (2, 45, 85)), "M", 0), "M",
                 "valid runs")
    inside = truth_codes.copy()
    inside[10] = X
    expect_error(ValueError, lambda: score.score_halves(inside, pred, graded, "M", 0), "M", "valid runs")
    odd = pred.copy()
    odd[3] = 4                                              # a code that pm.confusion would count into another cell
    expect_error(ValueError, lambda: score.score_halves(truth_codes, odd, graded, "M", 0), "M", "4")


# -- pooling ----------------------------------------------------------------------------------------------------------

def test_pooled_sums_counts_and_follows_the_scorers_conventions():
    assert score.IOUS == pm.SPELL_IOUS
    one = score.HalfScore("M1", 1, 0, 1, 2, (1, 1, 1, 0, 0), ZERO_CONFUSION)
    two = score.HalfScore("M1", 2, 1, 3, 2, (2, 1, 0, 0, 0), ZERO_CONFUSION)
    other = score.HalfScore("M2", 1, 0, 3, 3, (3, 3, 3, 3, 3), ((3, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0)))
    assert dataclasses.is_dataclass(one)
    try:
        one.n_gt = 5
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("a HalfScore is frozen")

    # The two halves of M1: 4 true and 4 predicted possessions, matched 3, 2, 1, 0, 0 times over the five IoUs, so
    # precision = recall = 3/4, 2/4, 1/4, 0, 0 and F1 is the same: mean 0.3. (The halves' own F1s are 0.4 and 0.24,
    # which average to 0.32: a mean of the halves is not the pooled score.)
    got = score.pooled([one, two])
    assert (got["n_gt"], got["n_pred"], got["halves"]) == (4, 4, 2), got
    expected = [0.75, 0.5, 0.25, 0.0, 0.0]
    assert all(math.isclose(got["spell_f1"][iou], value, abs_tol=1e-15) for iou, value in zip(score.IOUS, expected))
    assert math.isclose(got["spell_f1_mean"], 0.3, abs_tol=1e-15), got["spell_f1_mean"]
    assert math.isclose(score.pooled([one])["spell_f1_mean"], 0.4, abs_tol=1e-15)
    assert math.isclose(score.pooled([two])["spell_f1_mean"], 0.24, abs_tol=1e-15)
    assert got["macro_f1"] == 0.0 and got["alive_dead_f1"] == 0.0, "an all-zero confusion scores 0.0, as the scorer's"

    # Nothing true and nothing predicted is 0.0, not NaN and not an error (the scorer's convention): tp = n = 0.
    empty = score.pooled([score.HalfScore("M", 1, 0, 0, 0, (0, 0, 0, 0, 0), ZERO_CONFUSION)])
    assert empty["spell_f1_mean"] == 0.0 and all(value == 0.0 for value in empty["spell_f1"].values()), empty
    # Nothing predicted against something true is 0.0 as well; nothing true against something predicted too.
    assert score.pooled([score.HalfScore("M", 1, 0, 3, 0, (0,) * 5, ZERO_CONFUSION)])["spell_f1_mean"] == 0.0
    assert score.pooled([score.HalfScore("M", 1, 0, 0, 3, (0,) * 5, ZERO_CONFUSION)])["spell_f1_mean"] == 0.0
    expect_error(ValueError, lambda: score.pooled([]), "pooled", "no HalfScore")

    # The frame metrics are the scorer's, on the summed confusion.
    cm = np.array(one.confusion) + np.array(two.confusion) + np.array(other.confusion)
    got = score.pooled([one, two, other])
    assert got["macro_f1"] == pm.macro_f1(cm) and got["alive_dead_f1"] == pm.alive_dead_f1(cm)
    assert (got["n_gt"], got["n_pred"], got["halves"]) == (7, 7, 3)

    # per match: the counts of its halves and its seeds, matches in name order.
    per_match = score.per_match([other, two, one])
    assert list(per_match) == ["M1", "M2"], list(per_match)
    assert per_match["M1"] == score.pooled([one, two]) and per_match["M2"] == score.pooled([other])
    assert per_match["M1"]["halves"] == 2 and per_match["M1"]["n_gt"] == 4 and per_match["M2"]["n_gt"] == 3


# -- predict ----------------------------------------------------------------------------------------------------------

def test_predict_equals_pipeline_run():
    """predict is pipeline.run.run without the CSV: the same codes, and the evidence run() would hand back."""
    from pipeline import run as pipeline_run
    from pipeline.benchmark import run as benchmark_run
    from pipeline.config import load_config
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_bundle(root)
        inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
        trial = load_config(write_text(root, "trial.toml", TRIAL_CONFIG.format(scope="halves", width=5)))
        for config in (trial, yardstick(0), yardstick(25), yardstick(50), yardstick(125)):
            result = pipeline_run.run(config, inputs, os.path.join(root, "out.csv"), log=quiet)
            codes, sources = benchmark_run.predict(config, inputs)
            assert codes.dtype == np.int8 and codes.shape == (600,), (codes.dtype, codes.shape)
            assert np.array_equal(codes, result.codes), "{}: predict and run() differ on {} frames".format(
                config.stem, int((codes != result.codes).sum()))
            assert (codes[:50] == X).all() and (codes[280:320] == X).all() and (codes[560:] == X).all()
            assert (codes[50:280] != X).all() and (codes[320:560] != X).all()
            assert list(sources) == list(result.evidence) == ["nearest_player"]
            for name, source in sources.items():
                assert source.name == name and source.weight == 1.0
                for channel in ("alive", "home"):
                    assert np.array_equal(getattr(source, channel), getattr(result.evidence[name], channel),
                                          equal_nan=True), (config.stem, name, channel)
                    assert not getattr(source, channel).flags.writeable, "evidence is read-only once checked"


def test_predict_seeds_the_global_generators_like_run():
    """Library code that reaches for random / np.random gets config.seed, as under run(), whatever the state before."""
    from pipeline import evidence as evidence_stage
    from pipeline import run as pipeline_run
    from pipeline.benchmark import run as benchmark_run
    from pipeline.config import load_config
    from pipeline.contracts import Evidence, no_opinion

    def noisy_probe(run):
        """A source whose opinion comes from the two legacy global generators, as a careless component's would."""
        threshold = random.random()
        home = (np.random.random(run.inputs.n_frames) < threshold).astype(np.float64)
        return Evidence(alive=no_opinion(run.inputs.n_frames), home=home)

    text = 'seed = {}\n[[evidence]]\nkind = "noisy_probe"\n[decoder]\nkind = "majority_vote"\nhalf_width = 3\n'
    evidence_stage.REGISTRY["noisy_probe"] = noisy_probe
    try:
        with tempfile.TemporaryDirectory() as root:
            make_synthetic_bundle(root)
            inputs = synthetic_inputs(root, graded=[(1, 50, 280), (2, 320, 560)])
            config = load_config(write_text(root, "noisy.toml", text.format(3)))
            expected = pipeline_run.run(config, inputs, os.path.join(root, "out.csv"), log=quiet).codes
            random.seed(1234)
            np.random.seed(1234)                                     # whatever state the process was in
            codes, _ = benchmark_run.predict(config, inputs)
            assert np.array_equal(codes, expected), "predict did not seed the generators as run() does"
            assert np.array_equal(benchmark_run.predict(config, inputs)[0], codes), "twice, the same"
            other = load_config(write_text(root, "noisy4.toml", text.format(4)))
            assert not np.array_equal(benchmark_run.predict(other, inputs)[0], codes), (
                "another seed must give another draw, or this check proves nothing")
    finally:
        del evidence_stage.REGISTRY["noisy_probe"]


def test_predict_rejects_file_scope():
    """scope = "file" decodes across half time and over ungraded frames; the benchmark scores each half on its own."""
    from pipeline.benchmark import run as benchmark_run
    from pipeline.config import load_config
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(CONFIG_FOLDER)), "configs",
                                      "smooth50_repro.toml"))
    assert config.scope == "file"
    # Refused before anything is read: the inputs are not even looked at.
    expect_error(ValueError, lambda: benchmark_run.predict(config, None), "smooth50_repro", 'scope = "file"', "halves")


# -- the yardstick configs --------------------------------------------------------------------------------------------

def test_yardstick_configs_load_and_pass_the_signature_checks():
    """nearest_w{0,25,50,125}.toml: the organisers' four baselines as per-half configs, differing only in half_width."""
    texts = {}
    for width, baseline in YARDSTICKS.items():
        config = yardstick(width)                         # load_config checks every parameter against its signature
        assert config.stem == "nearest_w{}".format(width) and config.seed == 0 and config.scope == "halves", config
        assert [(c.kind, c.name, c.weight, c.params) for c in config.evidence] == [
            ("nearest_player", "nearest_player", 1.0, {})], config.evidence
        assert config.decoder.kind == "majority_vote", config.decoder
        assert config.decoder.params == {"half_width": width, "fallback": "D"}, config.decoder.params
        assert config.refine == (), config.refine
        assert baseline in config.description and "per half" in config.description and "file" in config.description, (
            "the description must name {} and the scope difference: {}".format(baseline, config.description))
        texts[width] = config.text
    # Four files that differ in their width and in what they say about the baseline they mirror, and in nothing else.
    def shared(text):
        skipped = ("#", "description", "half_width")
        return [line for line in text.splitlines() if line.strip() and not line.startswith(skipped)]
    assert all(shared(texts[width]) == shared(texts[0]) for width in YARDSTICKS), (
        "the yardsticks differ beyond half_width and their descriptions")
    assert len({texts[width] for width in YARDSTICKS}) == 4

    # Their header says how they differ, and it is true: in line 1 (the baseline each mirrors), the description and
    # half_width. So the other comment lines are the same in all four, the comment on the half_width line is too, and
    # the # of that line is in the column of the # on the fallback line under it.
    sentence = "differ in line 1, in the description and in half_width, and in nothing else."
    def comment_lines(text):
        return [line for line in text.splitlines()[1:] if line.startswith("#")]
    def trailing_comment(text, key):
        line = next(line for line in text.splitlines() if line.startswith(key))
        return line.partition("#")[2], line.index("#")
    for width in YARDSTICKS:
        assert comment_lines(texts[width]) == comment_lines(texts[0]), width
        assert " ".join(line.lstrip("# ") for line in comment_lines(texts[width])).count(sentence) == 1, width
        assert "in this paragraph" not in texts[width], "the header claims the files differ in a paragraph"
        note, column = trailing_comment(texts[width], "half_width")
        fallback_column = trailing_comment(texts[width], "fallback")[1]
        assert note == trailing_comment(texts[0], "half_width")[0] and column == fallback_column, (
            "{}: the # on the half_width line is in column {}, the one on the fallback line in {}".format(
                width, column, fallback_column))


# -- match-level statistics -------------------------------------------------------------------------------------------

def student_quantile(df, probability=0.95):
    """The probability-quantile of Student's t with df degrees of freedom, from its density, with nothing from the code.

    CDF(x) = 1/2 + the integral of the density from 0 to x (Simpson's rule, 800 steps); the root is found by bisection.
    """
    def density(t):
        return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
                        - (df + 1) / 2 * math.log1p(t * t / df))

    def cdf(x, steps=800):
        width = x / steps
        total = density(0.0) + density(x) + sum((4 if step % 2 else 2) * density(step * width)
                                                for step in range(1, steps))
        return 0.5 + total * width / 3

    low, high = 0.0, 20.0
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if cdf(middle) < probability else (low, middle)
    return (low + high) / 2


def brute_force_flip_p(deltas):
    """The sign-flip p by itertools, in exact arithmetic: each delta is the decimal that it prints as, as a Fraction.

    Every pattern of signs, and the share whose sum is at least as far from 0 as the data's: no tolerance, because
    nothing is rounded (0.1 + 0.2 is 0.3 here, and 0.4 + 0.2 - 0.2 is 0.4).
    """
    exact = [Fraction(repr(float(delta))) for delta in deltas]
    observed = abs(sum(exact))
    count = sum(1 for signs in itertools.product((1, -1), repeat=len(exact))
                if abs(sum(sign * delta for sign, delta in zip(signs, exact))) >= observed)
    return count / 2 ** len(exact)


def spell_half(match_id, n_gt, n_pred, tp, seed=0, half=1, confusion=ZERO_CONFUSION):
    """A HalfScore with the same true positives at every IoU: its spellF1 is that of tp, n_pred and n_gt."""
    return score.HalfScore(match_id, half, seed, n_gt, n_pred, (tp,) * 5, confusion)


def spell_f1(tp, n_pred, n_gt):
    """F1 from its definition: the harmonic mean of precision and recall, 0 where either is empty."""
    precision, recall = tp / n_pred, tp / n_gt
    return 2 * precision * recall / (precision + recall)


def test_t_table_is_the_two_sided_90_percent_quantile():
    """T_95 is the table of the brief, and each entry is the 95th percentile of t with that many degrees of freedom."""
    assert score.T_95 == {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
                          10: 1.812}
    assert all(type(value) is float for value in score.T_95.values())
    for df, value in score.T_95.items():
        exact = student_quantile(df)
        assert abs(value - exact) <= 5.1e-4, "df {}: the table says {}, the density gives {:.5f}".format(
            df, value, exact)


def test_t_interval_known():
    """[1..7]: mean 4, sd sqrt(28 / 6) = 2.1602 (ddof 1), six degrees of freedom: 1.943 * 2.1602 / sqrt(7) each side."""
    mean, low, high = score.t_interval([1, 2, 3, 4, 5, 6, 7])
    half = 1.943 * math.sqrt(28 / 6) / math.sqrt(7)
    assert abs(half - 1.5864) < 1e-4, half                              # worked by hand: 1.943 * 2.1602 / 2.6458
    assert mean == 4.0 and math.isclose(low, 4 - half, rel_tol=1e-12) and math.isclose(high, 4 + half, rel_tol=1e-12)
    assert all(type(value) is float for value in (mean, low, high))
    # Two values: sd = sqrt(2), one degree of freedom (6.314): 2 +/- 6.314 * sqrt(2) / sqrt(2).
    mean, low, high = score.t_interval((1, 3))
    assert mean == 2.0 and math.isclose(low, 2 - 6.314) and math.isclose(high, 2 + 6.314)
    # Equal values have no spread; an array is read like a list; the order does not matter.
    assert score.t_interval(np.array([0.25, 0.25, 0.25])) == (0.25, 0.25, 0.25)
    assert score.t_interval([7, 1, 4, 2, 6, 3, 5]) == score.t_interval([1, 2, 3, 4, 5, 6, 7])
    # Eleven values is ten degrees of freedom, the last row of the table. 0..10 have sample variance 11 * 12 / 12 = 11,
    # so the half-width is 1.812 * sqrt(11) / sqrt(11) = 1.812.
    mean, low, high = score.t_interval(list(range(11)))
    assert mean == 5.0 and math.isclose(high - mean, 1.812, rel_tol=1e-9)
    assert math.isclose(mean - low, 1.812, rel_tol=1e-9)


def test_t_interval_edge_cases():
    """Fewer than two values have a mean and no interval; more than eleven are beyond the table, and say so."""
    one = score.t_interval([0.4])
    assert one[0] == 0.4 and math.isnan(one[1]) and math.isnan(one[2])
    none = score.t_interval([])
    assert all(math.isnan(value) for value in none) and all(type(value) is float for value in none)
    assert all(type(value) is float for value in one)
    message = expect_error(ValueError, lambda: score.t_interval(list(range(12))), "t_interval", "12",
                           "10 degrees of freedom")
    assert "T_95" in message, message
    expect_error(ValueError, lambda: score.t_interval(np.zeros((3, 2))), "t_interval", "one-dimensional")
    # NaN is carried, not hidden.
    assert all(math.isnan(value) for value in score.t_interval([1.0, float("nan"), 3.0]))


def test_sign_flip_all_positive():
    """Seven deltas of one sign: only all-plus and all-minus are as far from 0 as the data, 2 of 128 patterns, 1/64."""
    assert score.sign_flip_p([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]) == 1 / 64
    assert score.sign_flip_p([-5, -1, -2, -9, -3, -4, -8]) == 1 / 64
    assert type(score.sign_flip_p([1, 2, 3])) is float
    # Sixteen is the largest case: 65,536 patterns, and 2 of them for deltas of one sign.
    assert score.sign_flip_p(np.linspace(0.1, 1.6, 16)) == 2 / 65536


def test_sign_flip_small_case():
    """Three deltas, the 8 patterns counted by hand."""
    # 2, 1, -1: the sums over +/-2 +/-1 +/-1 are 4, 2, 2, 0, 0, -2, -2, -4; observed 2 (a tie with three others), so the
    # six with |sum| >= 2 count: 6 / 8.
    assert score.sign_flip_p([2, 1, -1]) == 0.75
    # 3, 1, 1: 5, 3, 3, 1, -1, -3, -3, -5 -- observed 5, only the two extremes: 2 / 8.
    assert score.sign_flip_p([3, 1, 1]) == 0.25
    # 3, -1, 1: observed sum 3, patterns with |sum| >= 3 are +-5 (two) and +-3 (four): 6 / 8.
    assert score.sign_flip_p([3, -1, 1]) == 0.75
    # Exact ties must not be split by rounding. -0.4, -0.2, 0.2: the sums over the 8 patterns are -0.4, -0.8, 0, -0.4,
    # 0.4, 0, 0.8, 0.4, so the six with |sum| >= 0.4 count, 6 / 8, though as floats those sums of 0.4 are not all the
    # same number (without FLIP_TOLERANCE the answer is 4 / 8). The same for -0.3, -0.1, 0.1: 6 / 8.
    assert score.sign_flip_p([-0.4, -0.2, 0.2]) == 0.75
    assert score.sign_flip_p([-0.3, -0.1, 0.1]) == 0.75
    # -0.1, 0.2, 0.3: sums of +-0.6 (2), +-0.4 (2), +-0.2 (2) and 0 (2); the two of 0.4 and the two of 0.6 count: 4 / 8.
    assert score.sign_flip_p([-0.1, 0.2, 0.3]) == 0.5
    assert score.sign_flip_p([0.1, 0.2, -0.3]) == 1.0                   # observed mean 0 (to rounding): every pattern
    assert score.sign_flip_p([0.3, 0.2, 0.1]) == 0.25                   # observed 0.6: the two extremes
    # Nothing to tell apart: every pattern is as far from 0 as the data, which is at 0.
    assert score.sign_flip_p([0.0, 0.0, 0.0, 0.0]) == 1.0
    assert score.sign_flip_p([0.5]) == 1.0                              # one value: its flip is as far as it is
    # Against an independent count in exact arithmetic, on random deltas with exact ties (multiples of 0.05, 0.1 or
    # 0.01) and without.
    rng = np.random.default_rng([8, 5])
    for n in (2, 3, 5, 8, 10):
        for deltas in (np.round(rng.normal(size=n), 2), rng.integers(-3, 4, size=n) * 0.05,
                       rng.integers(-5, 6, size=n) / 10, rng.normal(size=n)):
            assert score.sign_flip_p(deltas) == brute_force_flip_p(deltas.tolist()), (n, deltas)
    # What it refuses: none, too many (17 would be 131,072 patterns), and values that are not numbers.
    expect_error(ValueError, lambda: score.sign_flip_p([]), "sign_flip_p", "no deltas")
    expect_error(ValueError, lambda: score.sign_flip_p(np.zeros(17)), "sign_flip_p", "17", "16")
    expect_error(ValueError, lambda: score.sign_flip_p([1.0, float("nan")]), "sign_flip_p", "NaN")
    expect_error(ValueError, lambda: score.sign_flip_p(np.zeros((2, 2))), "sign_flip_p", "one-dimensional")


def hand_built_configs():
    """Two configs on four matches, one half each, worked by hand: (scores_a, scores_b, the deltas b - a).

    spellF1 of M1: 0.5 -> 0.75 (tp 2 -> 3 of 4 and 4), M2: 0.5 -> 0.5 (20 of 40 either way: the big match that does not
    move), M3: 2/3 -> 0.5 (it predicted 4 possessions for 2 true ones and now predicts 2, matching 1), M4: 0.4 -> 0.6
    (2 -> 3 of 5 and 5).
    """
    a = [spell_half("M1", 4, 4, 2), spell_half("M2", 40, 40, 20), spell_half("M3", 2, 4, 2), spell_half("M4", 5, 5, 2)]
    b = [spell_half("M1", 4, 4, 3), spell_half("M2", 40, 40, 20), spell_half("M3", 2, 2, 1), spell_half("M4", 5, 5, 3)]
    return a, b, {"M1": 0.25, "M2": 0.0, "M3": 0.5 - 2 / 3, "M4": 0.2}


def test_compare_arithmetic_and_direction():
    """Delta is b - a, from the pooled counts of each match; wins and losses count its sign; the rest follows."""
    a, b, expected = hand_built_configs()
    assert math.isclose(spell_f1(2, 4, 4), 0.5) and math.isclose(spell_f1(1, 2, 2), 0.5)
    assert math.isclose(spell_f1(2, 4, 2), 2 / 3)
    result = score.compare(a, b, {"M1": "C1", "M2": "C1", "M3": "C2"})
    assert list(result) == ["pooled_delta", "match_deltas", "mean_delta", "low", "high", "p_sign_flip", "wins",
                            "losses", "n", "by_competition"], list(result)
    assert list(result["match_deltas"]) == ["M1", "M2", "M3", "M4"]
    for match_id, delta in expected.items():
        assert math.isclose(result["match_deltas"][match_id], delta, abs_tol=1e-12), (match_id, result["match_deltas"])
    assert (result["wins"], result["losses"], result["n"]) == (2, 1, 4)        # M2 is a tie: neither
    # The pooled delta is the delta of the two pooled scores, not the mean of the matches' deltas (0.0294 against
    # 0.0708): a matched 26 of 53 predicted and 51 true possessions, b 27 of 51 and 51 (F1 = 2 tp / (predicted + true)).
    assert math.isclose(result["pooled_delta"], spell_f1(27, 51, 51) - spell_f1(26, 53, 51), abs_tol=1e-12)
    assert abs(result["pooled_delta"] - 0.0294) < 5e-5 and abs(result["mean_delta"] - 0.0708) < 5e-5
    mean, low, high = score.t_interval(list(expected.values()))
    assert math.isclose(result["mean_delta"], mean, abs_tol=1e-12)
    assert math.isclose(result["mean_delta"], (0.25 + 0.0 - 1 / 6 + 0.2) / 4, abs_tol=1e-12)
    assert math.isclose(result["low"], low, abs_tol=1e-12) and math.isclose(result["high"], high, abs_tol=1e-12)
    half = 2.353 * np.std(list(expected.values()), ddof=1) / 2                 # three degrees of freedom, sqrt(4) = 2
    assert math.isclose(result["high"] - result["mean_delta"], half, rel_tol=1e-9)
    assert result["low"] < 0 < result["high"]
    assert result["p_sign_flip"] == brute_force_flip_p(list(expected.values()))
    # By competition: the mean delta of its matches, in the order of the names; a match the mapping lacks is "other".
    assert list(result["by_competition"]) == ["C1", "C2", "other"], result["by_competition"]
    assert math.isclose(result["by_competition"]["C1"], 0.125)
    assert math.isclose(result["by_competition"]["C2"], 0.5 - 2 / 3)
    assert math.isclose(result["by_competition"]["other"], 0.2)

    # The other way round is the other sign, with the same evidence against 0.
    reverse = score.compare(b, a, {"M1": "C1", "M2": "C1", "M3": "C2"})
    assert (reverse["wins"], reverse["losses"]) == (1, 2)
    assert reverse["match_deltas"] == {m: -d for m, d in result["match_deltas"].items()}
    assert reverse["mean_delta"] == -result["mean_delta"] and reverse["p_sign_flip"] == result["p_sign_flip"]
    assert math.isclose(reverse["low"], -result["high"]) and math.isclose(reverse["high"], -result["low"])
    assert all(type(result[key]) is float for key in ("pooled_delta", "mean_delta", "low", "high", "p_sign_flip"))
    assert all(type(value) is float for value in result["match_deltas"].values())
    assert all(type(result[key]) is int for key in ("wins", "losses", "n"))


def test_compare_identical():
    """A config against itself: no difference anywhere, p of 1.0, and not one win or loss."""
    a, _, _ = hand_built_configs()
    result = score.compare(a, list(a))
    assert result["pooled_delta"] == 0.0 and result["mean_delta"] == 0.0 and result["match_deltas"] == dict.fromkeys(
        ("M1", "M2", "M3", "M4"), 0.0)
    assert result["low"] == 0.0 and result["high"] == 0.0
    assert result["p_sign_flip"] == 1.0 and (result["wins"], result["losses"], result["n"]) == (0, 0, 4)
    assert result["by_competition"] == {"other": 0.0}, "synthetic matches are in no competition"


def test_compare_groups_the_real_matches_by_competition():
    """With no mapping given, the benchmark's own: two Bundesliga matches and five 2. Bundesliga, by their ids."""
    from pipeline.benchmark import sources
    assert list(sources.MATCH_IDS) == sorted(sources.MATCH_IDS)
    a = [spell_half(match_id, 10, 10, 4) for match_id in sources.MATCH_IDS]
    gains = dict(zip(sources.MATCH_IDS, (1, 2, 3, 4, 5, 6, 7)))
    b = [spell_half(match_id, 10, 10, 4 + gains[match_id] % 3) for match_id in sources.MATCH_IDS]
    result = score.compare(a, b)
    assert result["n"] == 7 and list(result["by_competition"]) == ["DFL-COM-000001", "DFL-COM-000002"]
    for competition in result["by_competition"]:
        members = [m for m in sources.MATCH_IDS if sources.COMPETITION[m] == competition]
        assert math.isclose(result["by_competition"][competition],
                            sum(result["match_deltas"][m] for m in members) / len(members)), competition
    assert len([m for m in sources.MATCH_IDS if sources.COMPETITION[m] == "DFL-COM-000002"]) == 5
    # Seven matches is six degrees of freedom (T_95[6]): the interval and the p are those of the deltas, unweighted.
    mean, low, high = score.t_interval(list(result["match_deltas"].values()))
    assert (result["mean_delta"], result["low"], result["high"]) == (mean, low, high)
    assert result["p_sign_flip"] == brute_force_flip_p(list(result["match_deltas"].values()))


def test_compare_serves_macro_f1_through_the_metric_keyword():
    """metric="macro_f1" is the same comparison of the frame metric, from the matches' summed confusions."""
    # Truth rows H, A, D; columns H, A, D, X. M1 gets the home/away calls right in b and not in a, M2 is the other way,
    # and M3 is better in b than in a.
    good = ((8, 0, 0, 0), (0, 8, 0, 0), (0, 0, 4, 0))
    poor = ((4, 4, 0, 0), (4, 4, 0, 0), (0, 0, 4, 0))
    middling = ((6, 2, 0, 0), (2, 6, 0, 0), (0, 0, 4, 0))
    a = [spell_half("M1", 3, 3, 1, confusion=poor), spell_half("M2", 3, 3, 1, confusion=good),
         spell_half("M3", 3, 3, 1, confusion=middling)]
    b = [spell_half("M1", 3, 3, 1, confusion=good), spell_half("M2", 3, 3, 1, confusion=poor),
         spell_half("M3", 3, 3, 1, confusion=good)]
    macro = {name: pm.macro_f1(np.array(matrix)) for name, matrix in (("good", good), ("poor", poor),
                                                                      ("middling", middling))}
    expected = {"M1": macro["good"] - macro["poor"], "M2": macro["poor"] - macro["good"],
                "M3": macro["good"] - macro["middling"]}
    result = score.compare(a, b, metric="macro_f1")
    assert result["match_deltas"] == expected and expected["M1"] > 0.2 and expected["M3"] > 0.05
    assert (result["wins"], result["losses"], result["n"]) == (2, 1, 3)
    pooled_a = np.array(poor) + np.array(good) + np.array(middling)
    pooled_b = np.array(good) + np.array(poor) + np.array(good)
    assert result["pooled_delta"] == pm.macro_f1(pooled_b) - pm.macro_f1(pooled_a) > 0.01
    # The spell counts are identical, so the default metric sees nothing, and a mistyped metric is an error.
    assert score.compare(a, b)["match_deltas"] == dict.fromkeys(("M1", "M2", "M3"), 0.0)
    assert score.compare(a, b)["pooled_delta"] == 0.0
    assert score.compare(a, b, metric="spell_f1_mean") == score.compare(a, b)
    expect_error(ValueError, lambda: score.compare(a, b, metric="f1"), "compare", "f1", "macro_f1", "spell_f1_mean")


def test_compare_rejects_mismatched_matches():
    """Two configs are compared match by match, so they must have been scored on the same matches, halves and seeds."""
    a, b, _ = hand_built_configs()
    expect_error(ValueError, lambda: score.compare(a[:3], b), "compare", "M4", "different matches")
    expect_error(ValueError, lambda: score.compare(a, b[1:]), "compare", "M1", "different matches")
    other = [spell_half("M9" if half.match_id == "M3" else half.match_id, half.n_gt, half.n_pred, half.tp[0])
             for half in b]
    message = expect_error(ValueError, lambda: score.compare(a, other), "compare", "M3", "M9")
    assert "different matches" in message
    # The same matches scored on other seeds (or with a half missing) are not a pairing either: the pooled counts would
    # be of different things. Named by (match, half, seed).
    reseeded = [spell_half(half.match_id, half.n_gt, half.n_pred, half.tp[0], seed=1) for half in b]
    expect_error(ValueError, lambda: score.compare(a, reseeded), "compare", "M1", "half 1", "seed 0", "seed 1")
    expect_error(ValueError, lambda: score.compare(a, b + [spell_half("M1", 4, 4, 3, half=2)]), "compare", "M1",
                 "half 2")
    expect_error(ValueError, lambda: score.compare([], []), "compare", "no HalfScore")
    expect_error(ValueError, lambda: score.compare(a, []), "compare", "no HalfScore")
    # Twelve matches is more than the t table holds (ten degrees of freedom): refused by the interval, with its limit.
    many_a = [spell_half("N{:02d}".format(i), 4, 4, 2) for i in range(12)]
    many_b = [spell_half("N{:02d}".format(i), 4, 4, 3) for i in range(12)]
    expect_error(ValueError, lambda: score.compare(many_a, many_b), "t_interval", "10 degrees of freedom")


# -- a real match -----------------------------------------------------------------------------------------------------

def test_dfl_pooled_equals_official_evaluate_on_a_real_match():
    """On J03WMX the yardsticks' pooled score is pm.evaluate's of the same codes, to the last digit.

    The truth of a real match is its own concatenation (the gap between the halves is X), so pm.evaluate reads it as
    it is. Four yardsticks, raw to +/-5 s, through the whole machinery: MatchInputs, predict, score_halves, pooled.
    Prints how far each pooled number is from the official one (the largest of its scores), and no score: what the
    yardsticks score is for the tasks that run them on every match, with the degrader, to report.
    """
    from pipeline.benchmark import inputs as match_inputs
    from pipeline.benchmark import run as benchmark_run
    from pipeline.benchmark.checks.common import real_match
    match = real_match("J03WMX")
    truth_codes, _ = truth.truth_codes(match)
    assert pm.valid_runs(truth_codes) == [(start, stop) for _, start, stop in match.graded]
    inputs = match_inputs.MatchInputs(match, match_inputs.clean_atd(match))
    for width in YARDSTICKS:
        codes, _ = benchmark_run.predict(yardstick(width), inputs)
        got = score.pooled(score.score_halves(truth_codes, codes, match.graded, match.match_id, 0))
        official = pm.evaluate(truth_codes, codes)
        assert got["n_gt"] == official["n_gt_spells"] > 100 and got["n_pred"] == official["n_pred_spells"] > 100, width
        assert 0.0 < got["spell_f1_mean"] < 1.0, "a score of 0 or 1 would prove little"
        gaps = [abs(got[key] - official[key]) for key in ("spell_f1_mean", "macro_f1", "alive_dead_f1")]
        gaps += [abs(got["spell_f1"][iou] - official["spell_f1"][iou]["f1"]) for iou in pm.SPELL_IOUS]
        assert max(gaps) < TOLERANCE, (width, gaps)
        print("      nearest_w{:<4} pooled() against pm.evaluate on {}: largest difference {:.1e}".format(
            width, match.match_id, max(gaps)))
