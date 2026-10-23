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
import math
import os
import random
import tempfile

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
