"""Scoring a benchmark prediction exactly as the organisers' scorer would, half by half, and pooling by counts.

    scores = score_halves(truth, codes, match.graded, match.match_id, seed)    # two HalfScore, one per graded half
    pooled(scores)       # one score for any set of halves, matches and seeds: the official scorer's on all of them
    per_match(scores)    # the same, one per match, over its halves and its seeds

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
"""

from dataclasses import dataclass

import numpy as np

import possession_metrics as pm

# Params
IOUS = pm.SPELL_IOUS


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
