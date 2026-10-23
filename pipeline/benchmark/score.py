"""Scoring a benchmark prediction exactly as the organisers' scorer would, half by half, and pooling by counts.

    scores = score_halves(truth, codes, match.graded, match.match_id, seed)    # two HalfScore, one per graded half
    pooled(scores)       # one score for any set of halves, matches and seeds: the official scorer's on all of them
    per_match(scores)    # the same, one per match, over its halves and its seeds
    compare(scores_a, scores_b)    # config b against config a, match by match: deltas, an interval, a sign-flip p

Why it is built like this. A benchmark ranks pipeline choices over several matches and degradation seeds, and what it
ranks on has to be the number the organisers' scorer would give if all of them were one file. pm.evaluate on the blocks
laid end to end (one X row between them, so that pm.valid_runs is exactly the blocks) is that number, and pooled() is
equal to it, to the last digit; test_pooled_equals_official_evaluate builds that concatenation and asserts it, and the
--dfl check asserts it on a real match. The way to get there is to keep counts and never averages: a half's HalfScore is
its spell counts (true, predicted, and matched at each IoU) and its confusion matrix, which add up over halves, matches
and seeds, and the F1s are computed once, from the sums, as pm.spell_scores and pm.macro_f1 do. The mean of the halves'
own F1s is another number, and not the scorer's.

Per half, score_halves mirrors pm.evaluate step by step, on the block [start, stop) of the arrays:
  1. pm.fill_abstentions on [(start, stop)]: an X is filled forward with the last label of the half, back-filled at its
     head, and nothing is carried across the gap between the halves (the temporal metrics never cross it);
  2. pm.possession_spells of the truth and of the filled prediction, which are the maximal runs of H or of A;
  3. pm.spell_overlaps of the two, and pm.matched_gt_flags at each IoU of pm.SPELL_IOUS, greedy one-to-one, whose sum is
     the number of true possessions matched (tp);
  4. pm.scored_confusion(pm.confusion(...)) of the UNFILLED codes: the 3x4 matrix with true H, A, D in the rows and
     predicted H, A, D, X in the columns, in which an abstention on a graded frame is a miss (the scorer's whole X
     policy).
The matching is done per half and not over the pooled spells, which is the same thing: spells of different blocks never
overlap, and the greedy pass consumes a spell only against spells of its own half.

The scorer's conventions for an empty case are kept: with nothing true, nothing predicted, or nothing matched, a
precision, a recall and an F1 are 0.0 (never NaN, never an error), so a half with no possession at all in truth and
prediction alike (tp = n = 0) scores 0.0, as pm.spell_scores gives it. pooled([]) is an error, not 0.0: it would be a
mistake (an empty filter) that a score of 0.0 would hide.

Match-level statistics. The benchmark has 7 matches, and five of them are Fortuna Dusseldorf's home games, so they are
neither many nor independent draws of "a match". What can honestly be said of a difference between two configs is how
it looks over those 7: the delta of each match (the unit of comparison, as the matches are what varies), its mean with
a Student t interval of at most 10 degrees of freedom (T_95), and an exact sign-flip test, which assumes nothing about
the shape of the deltas and enumerates all 2^n sign patterns (n <= 16) instead of sampling them. Neither is a
bootstrap: resampling 7 matches, or the frames inside them, would pretend to a sample the benchmark does not have. The
intervals say how far to trust a gap, and are not licence to pick between two configs on a gap they do not clear; with
n = 7 the smallest p that the sign-flip test can give is 2/128 = 1/64. compare() sets two configs' HalfScores against
each other, pairs them by match, and reports them per match, pooled, by competition and with these two statistics.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

import possession_metrics as pm
from pipeline.benchmark import sources

# Params
IOUS = pm.SPELL_IOUS
# The two-sided 90% quantiles of Student's t (the 0.95 quantile) by degrees of freedom: a 90% interval for a mean of
# df + 1 values. Ten degrees of freedom is the last row; the benchmark has 7 matches (6 degrees of freedom).
T_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
MAX_FLIP = 16                  # sign_flip_p enumerates 2^n patterns: 65,536 for n = 16, 8 MB of signs
FLIP_TOLERANCE = 1e-12         # patterns whose mean is this close to the observed one count as equal (see sign_flip_p)
METRICS = ("spell_f1_mean", "macro_f1", "alive_dead_f1")     # the scalar scores of pooled() that compare() can take
OTHER_COMPETITION = "other"    # where compare() files a match that the competition mapping does not name


@dataclass(frozen=True)
class HalfScore:
    """What one graded half of one match, for one degradation seed, adds to a pooled score: counts, no averages.

    `n_gt` and `n_pred` are the possessions of the truth and of the filled prediction; `tp[i]` is how many true
    possessions were matched at IOUS[i]; `confusion` is the 3x4 scored confusion of the unfilled codes (rows the truth's
    H, A, D, columns the prediction's H, A, D, X), as a tuple of tuples of ints so that a HalfScore is hashable and
    prints as what it is.
    """
    match_id: str
    half: int
    seed: int
    n_gt: int
    n_pred: int
    tp: tuple
    confusion: tuple


def score_halves(truth, codes, graded, match_id, seed):
    """[HalfScore, ...] one per block of `graded`, in order: the counts pm.evaluate would use for those halves.

    `truth` and `codes` are int8 arrays of pm's codes with one entry per row of the match; `graded` is ((half, start,
    stop), ...), half-open, as DflMatch.graded and Inputs.graded give it. ValueError, naming the match, for arrays of
    different length, for a code outside H, A, D, X (pm.confusion would count it into another cell), and for blocks that
    are not exactly the truth's own valid runs (pm.valid_runs): that is what makes the pooled score the official
    scorer's, which finds the halves in the truth and not in a list.
    """
    truth, codes = np.asarray(truth), np.asarray(codes)
    if truth.ndim != 1 or truth.shape != codes.shape:
        raise ValueError("{}: the truth has shape {} and the codes {}; both must be one entry per row of the match"
                         .format(match_id, truth.shape, codes.shape))
    for name, array in (("truth", truth), ("codes", codes)):
        odd = sorted(set(np.unique(array).tolist()) - set(range(pm.N_LABELS)))
        if odd:
            raise ValueError("{}: the {} hold the codes {}, and the scorer's are 0-{} (H, A, D, X)".format(
                match_id, name, odd, pm.N_LABELS - 1))
    blocks = [(int(start), int(stop)) for _, start, stop in graded]
    if pm.valid_runs(truth) != blocks:
        raise ValueError("{}: the graded blocks {} are not the truth's valid runs {} (the stretches that are not X); "
                         "the scorer finds the halves in the truth, so the two must say the same".format(
                             match_id, blocks, pm.valid_runs(truth)))

    scores = []
    for (half, _, _), (start, stop) in zip(graded, blocks):
        runs = [(start, stop)]
        filled = pm.fill_abstentions(codes, runs)
        gt_spells = pm.possession_spells(truth, runs)
        pred_spells = pm.possession_spells(filled, runs)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        tp = tuple(int(pm.matched_gt_flags(overlaps, len(gt_spells), iou).sum()) for iou in IOUS)
        confusion = pm.scored_confusion(pm.confusion(truth[start:stop], codes[start:stop]))
        scores.append(HalfScore(match_id=match_id, half=half, seed=seed, n_gt=len(gt_spells), n_pred=len(pred_spells),
                                tp=tp, confusion=tuple(tuple(int(count) for count in row) for row in confusion)))
    return scores


def pooled(scores):
    """The score of all the halves in `scores` as one: what pm.evaluate gives for them laid end to end.

    {"spell_f1_mean", "spell_f1": {iou: f1}, "macro_f1", "alive_dead_f1", "n_gt", "n_pred", "halves"}. Counts are
    summed over the halves (and so over their matches and seeds) first, and the F1s computed from the sums: the
    spell F1 at each IoU from tp, n_pred and n_gt, and its mean over the IoUs (the ranking metric, spell_f1_mean); the
    macro F1 and the alive/dead F1 from the summed confusion, with the scorer's own functions. `halves` is how many
    HalfScores were pooled. ValueError for none.
    """
    scores = list(scores)
    if not scores:
        raise ValueError("pooled: no HalfScore to pool; an empty list would score 0.0 and hide the mistake that "
                         "emptied it (a filter on a match id that is not there, say)")
    n_gt = sum(score.n_gt for score in scores)
    n_pred = sum(score.n_pred for score in scores)
    matched = np.sum([score.tp for score in scores], axis=0)
    confusion = np.sum([np.array(score.confusion, dtype=np.int64) for score in scores], axis=0)

    f1s = []
    for true_positive in matched.tolist():
        # As pm.spell_scores: false positives are the predictions not matched, false negatives the truths not matched,
        # and an empty denominator is 0.0.
        precision = true_positive / float(n_pred) if n_pred else 0.0
        recall = true_positive / float(n_gt) if n_gt else 0.0
        f1s.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "spell_f1_mean": float(np.mean(f1s)),
        "spell_f1": dict(zip(IOUS, f1s)),
        "macro_f1": pm.macro_f1(confusion),
        "alive_dead_f1": pm.alive_dead_f1(confusion),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "halves": len(scores),
    }


def per_match(scores):
    """{match_id: pooled(that match's HalfScores)}, in match-id order: counts summed over its halves and its seeds."""
    scores = list(scores)
    return {match_id: pooled([score for score in scores if score.match_id == match_id])
            for match_id in sorted({score.match_id for score in scores})}


# -- match-level statistics -------------------------------------------------------------------------------------------

def t_interval(values):
    """(mean, low, high): the mean of `values` and its two-sided 90% Student t interval, as Python floats.

    mean +/- T_95[n - 1] * sd / sqrt(n), with the sample standard deviation (ddof = 1). With fewer than two values
    there is no spread to measure: the bounds are NaN and the mean is as computed (NaN for no values at all). Eleven
    values (10 degrees of freedom) are the most the table holds; more is a ValueError that says so, because a
    quantile quietly taken from a nearer row would give an interval that is too narrow. NaN in `values` gives NaN.
    """
    values = _vector(values, "t_interval")
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan")
    if n - 1 not in T_95:
        raise ValueError("t_interval: {} values are {} degrees of freedom, and T_95 holds the t quantiles up to {} "
                         "degrees of freedom ({} values); add the rows (two-sided 90%) before comparing more "
                         "matches".format(n, n - 1, max(T_95), max(T_95) + 1))
    half = T_95[n - 1] * float(values.std(ddof=1)) / float(np.sqrt(n))
    return mean, mean - half, mean + half


def sign_flip_p(deltas):
    """The exact two-sided sign-flip (randomisation) p of the mean of `deltas`: the share of sign patterns as extreme.

    Under "the two configs are exchangeable" each match's delta is as likely to be negative as positive, so every one
    of the 2^n patterns of signs is as likely as the observed one. p is the share of the patterns whose mean is at
    least as far from 0 as the observed mean, |mean| >= |observed mean| - FLIP_TOLERANCE. The tolerance is there
    because floating-point sums of the same numbers in another order of signs differ in the last bits (-0.4 - 0.2 + 0.2
    is -0.4000000000000001 and 0.4 - 0.2 + 0.2 is 0.4: the same distance from 0 on paper, two numbers in the
    machine), which would split exact ties between "as extreme" and "less": a tie counts as extreme. The observed
    pattern and its mirror image are always counted, so p >= 2 / 2^n, and deltas that are all 0 (identical configs)
    give 1.0. A Python float.

    ValueError for no deltas, for more than MAX_FLIP (2^17 patterns would be too many to hold and to mean much: the
    benchmark has 7 matches), for NaN or infinity, and for anything that is not a one-dimensional list of numbers.
    """
    deltas = _vector(deltas, "sign_flip_p")
    n = len(deltas)
    if not n:
        raise ValueError("sign_flip_p: no deltas; there is nothing to flip")
    if n > MAX_FLIP:
        raise ValueError("sign_flip_p: {} deltas are {:,} sign patterns, and the exact test is for at most {} "
                         "deltas ({:,} patterns)".format(n, 2 ** n, MAX_FLIP, 2 ** MAX_FLIP))
    if not np.isfinite(deltas).all():
        raise ValueError("sign_flip_p: the deltas hold NaN or infinity ({} of {}); a pattern's mean would not be a "
                         "number".format(int((~np.isfinite(deltas)).sum()), n))
    # Row k of `bits` is the k-th pattern: bit j of k says whether delta j is flipped.
    bits = (np.arange(2 ** n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)) & 1
    means = (deltas * (1.0 - 2.0 * bits)).sum(axis=1) / n
    as_extreme = np.abs(means) >= abs(float(deltas.sum()) / n) - FLIP_TOLERANCE
    return float(np.count_nonzero(as_extreme)) / 2 ** n


def compare(scores_a, scores_b, competition=None, *, metric="spell_f1_mean"):
    """Config b against config a, match by match: {the delta of each match, its mean and interval, a p, wins, ...}.

    `scores_a` and `scores_b` are the HalfScores of the two configs (score_halves, over every match and seed), and
    every delta is b - a: a is the reference and a positive delta means b is better. The benchmark runner calls
    compare(first, other), the first config being the reference. `metric` is the scalar of pooled() that is compared:
    "spell_f1_mean", the ranking metric (the default), "macro_f1" or "alive_dead_f1".

    Returns a dict:
      pooled_delta   the delta of the two configs' pooled scores, over all halves, matches and seeds
      match_deltas   {match_id: delta of that match's pooled score (per_match), its halves and seeds summed}, by id
      mean_delta     the mean of the match deltas, each match counting once: the statistic the next three are about
      low, high      its two-sided 90% t interval (t_interval)
      p_sign_flip    the exact sign-flip p of the match deltas (sign_flip_p)
      wins, losses   how many matches have a delta above 0, and below 0; a delta of exactly 0 is neither
      n              the number of matches
      by_competition {competition: the mean delta of its matches}, by name; `competition` is a {match_id: name}
                     mapping, default sources.COMPETITION, and a match it does not name goes under "other"

    ValueError, naming what differs, if either config has no HalfScore, if the two were not scored on the same
    matches, or on the same halves and seeds of them (the pooled counts would then be of different things), if
    `metric` is not one of METRICS, and, from t_interval and sign_flip_p, if there are more than 11 matches.
    """
    scores_a, scores_b = list(scores_a), list(scores_b)
    if metric not in METRICS:
        raise ValueError("compare: metric {!r} is not one of {}".format(metric, ", ".join(METRICS)))
    for name, scores in (("scores_a", scores_a), ("scores_b", scores_b)):
        if not scores:
            raise ValueError("compare: {} holds no HalfScore; an empty list would hide the mistake that emptied it "
                             "(a filter on a match id that is not there, say)".format(name))
    matches_a, matches_b = ({score.match_id for score in scores} for scores in (scores_a, scores_b))
    if matches_a != matches_b:
        raise ValueError("compare: the two configs were scored on different matches (only in scores_a: {}; only in "
                         "scores_b: {}); a paired comparison needs the same matches".format(
                             ", ".join(sorted(matches_a - matches_b)) or "none",
                             ", ".join(sorted(matches_b - matches_a)) or "none"))
    cells_a, cells_b = (Counter((score.match_id, score.half, score.seed) for score in scores)
                        for scores in (scores_a, scores_b))
    if cells_a != cells_b:
        def cells(counter):
            return ", ".join("{} half {} seed {}".format(*cell) for cell in sorted(counter)) or "none"
        raise ValueError("compare: the two configs were scored on different halves or seeds of the same matches "
                         "(only in scores_a: {}; only in scores_b: {}); their pooled counts would be of different "
                         "things".format(cells(cells_a - cells_b), cells(cells_b - cells_a)))

    per_a, per_b = per_match(scores_a), per_match(scores_b)
    match_deltas = {match_id: float(per_b[match_id][metric] - per_a[match_id][metric]) for match_id in sorted(per_a)}
    deltas = list(match_deltas.values())
    mean, low, high = t_interval(deltas)
    names = sources.COMPETITION if competition is None else competition
    groups = {}
    for match_id, delta in match_deltas.items():
        groups.setdefault(names.get(match_id, OTHER_COMPETITION), []).append(delta)
    return {
        "pooled_delta": float(pooled(scores_b)[metric] - pooled(scores_a)[metric]),
        "match_deltas": match_deltas,
        "mean_delta": mean,
        "low": low,
        "high": high,
        "p_sign_flip": sign_flip_p(deltas),
        "wins": sum(1 for delta in deltas if delta > 0),
        "losses": sum(1 for delta in deltas if delta < 0),
        "n": len(deltas),
        "by_competition": {name: float(np.mean(groups[name])) for name in sorted(groups)},
    }


def _vector(values, caller):
    """`values` as a one-dimensional float64 array, or a ValueError naming `caller`."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("{}: expected a one-dimensional list of numbers, got an array of shape {}".format(
            caller, array.shape))
    return array
