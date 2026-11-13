"""A19: where a config's spellF1 goes between IoU 0.3 and 0.7, and whether its sources and its posterior mean what they
say. Aggregates of the seven DFL matches only.

    python pipeline/benchmark/diagnose.py CONFIG [--profile P] [--seeds N] [--matches ID ...] [--data-dir D]
                                          [--drop-match-only] [--no-write]

    report = diagnose(config_path)              # the ladder under both truths and the calibration: what is written
    half = ladder_half(truth, codes, start, stop)   # one graded half: its counts and its clean pairs
    ladder_summary([half, ...])                 # pooled: the ladder, its losses by class, the clean pairs' edges
                                                # (and, with a ball mask, C1's edges_by_kind: boundary kind x ball state)
    calibration(p, y)                           # log-loss, Brier, ECE, a reliability table and a Platt fit

docs/results.md A19 (part 1, declared before any run) is the authority for every definition here; this docstring says
how the code reads it.

B, the ladder
-------------
`main.toml`'s pooled spellF1 on `tracking` falls from 0.670 at IoU 0.3 to 0.401 at 0.7 (raw truth). Part of that is
edges placed a few frames off on possessions that are otherwise right, which R025's posterior edges could move. Part is
predicted possessions that span two true ones of their team (merges) or true ones cut into several (splits), which
moving an edge cannot fix. The ladder says how much is which, per truth (raw and coarse), pooled over every graded half
of every match and seed.

* Spells and matching are the scorer's. Per graded half, the spells are pm.possession_spells of the truth and of the
  prediction filled by pm.fill_abstentions, as score.score_halves has them; the overlaps are pm.spell_overlaps, which
  lists only same-team overlapping pairs; and the pairs are the greedy pass of pm.matched_gt_flags written out so that
  it also says which pairs it matched (greedy_pairs). That pass runs down one list sorted by IoU and stops at the first
  IoU below its threshold, so the matching at any IoU is the matching at 0.3 cut there: a pair matched at 0.3 and not at
  θ is the ladder's loss at θ. ladder_half asserts, on every half it is given, that the cut gives pm.matched_gt_flags'
  own flags at every IoU, and diagnose() that every half's counts are score.score_halves', so the numbers here are the
  runner's to the last bit, on the real matches as on the checks'.
* Classes. A pair (g, p) matched at 0.3 is put in one class by how many spells of its team each overlaps: clean if p
  overlaps no other true spell and g no other predicted one; merge if p overlaps 2 or more true spells and g only p;
  split if g overlaps 2 or more predicted spells and p only g; both if both.
* Losses. A pair lost at θ costs 2 / (n_gt + n_pred) of F1 at θ, and a fifth of that of spellF1. The clean-pair ladder
  loss is the pooled spellF1 if every clean pair matched at 0.3 were also matched at every θ, minus the actual pooled
  spellF1: the mean over θ of F1(tp_θ + clean pairs lost at θ), minus the mean of F1(tp_θ), with the scorer's F1 from
  the pooled counts (score.pooled). split_loss, merge_loss and both_loss are the same figure for those classes. It is
  not a bound on what R025 can gain: moving an edge can also raise a split pair's IoU or lift a pair below 0.3 into a
  match, and the figure assumes perfect edges. It is the loss that edges alone explain, and R025's gate (GATE, on the
  mean of the two truths: R023's objective).
* Beside them: the shares of predicted spells that overlap 2 or more true spells of their team (merge_share) and of true
  spells that overlap 2 or more predicted ones (split_share); the spells unmatched at 0.3, by whether they overlap any
  spell of their team; and the clean pairs' IoU (median, and the share at or above each θ) and signed edge errors in
  frames, predicted start minus true start and predicted stop minus true stop (median, quartiles by numpy's linear rule,
  and the share within ±EDGE_TOLERANCE frames).

The codes the ladder reads are the config's own, refiners included (run.predict): what the runner scores.

D, calibration
--------------
Against the raw truth only, the frame-level one (the coarse truth is a construct of grain), on the graded frames.

* Each source of the config, pooled over the matches and seeds, on the frames where it speaks. Its `home` against the
  truth's team on the truth's alive frames (y = the truth is H), and, for a source of a kind in SHAPE_KINDS (P07's
  shape_possession), split by whether the degraded ball is present (inputs.atd.ball_present); its `alive` against the
  truth's alive (y = the truth is not D). A channel the source never speaks on is left out. Every learned model is
  out of fold: dead_ball_model and shape_possession score a benchmark match with the fold that never saw it.
* The decoder's posterior: hmm.marginals at the config's own [decoder] parameters (switch_team, switch_dead,
  min_possession, min_dead, eps, and team_neutral_dead, which turns on the emissions' D term of every `home` vote),
  on hmm.log_likelihoods of every source cut to the graded block, so before any
  refiner (shift_edges). P(H) + P(A) against alive, on all graded frames and split by whether a team source speaks on
  the frame (any source with a finite `home`): `hmm` adds w log 2h and w log 2(1 - h) to H and A and, by default, nothing to D, so
  wherever a team source speaks the better alive state sits w log 2 max(h, 1 - h) above D, and beta on team_speaks
  measures that 50/50 assumption (with team_neutral_dead on, P31, D gets the larger of the two terms too and that
  gap is gone; the figures then say what the decoder does with the flag). And P(H) / (P(H) + P(A)) against the team, on the truth's alive frames; a frame
  whose P(H) + P(A) is 0 (underflow) has no such ratio and is counted apart (frames_without_alive_mass).
* Added after the review of part 1, reported only and gating nothing: the same alive measure by which team sources
  speak on the frame (alive_by_team_sources), keyed by the sorted names of the sources with a finite `home` joined by
  "+", or "none" (speaker_groups). In main.toml shape_possession speaks on nearly every graded frame, so the declared
  split is nearly all team_speaks; this says which sources the 50/50 assumption meets where. At most MAX_SPEAKER_KEYS
  keys: beyond them the smallest combinations are folded under "other".
* Metrics (calibration): the log-loss and the Brier score; the ECE over BINS equal-width bins and the reliability
  table (bin k holds floor(p * BINS) = k, p = 1 in the last); and a Platt fit by maximum likelihood, logit P(truth) =
  alpha logit(p) + beta, p clipped to [CLIP, 1 - CLIP] (the log-loss's clip too): alpha below 1 is overconfident, above
  1 underconfident, beta is a bias. The decoder reads a source at weight w as w logit(p), so its alpha is the weight
  that would make the source calibrated on its own, reported beside w (frames are not independent: a reading, not a
  target). The fit is Newton's method from (1, 0) with step halving, deterministic. Where logit(p) takes one value only
  a bias can be fitted (alpha None); where the two classes do not overlap in logit(p), or one is missing, the
  likelihood has no maximum and alpha and beta are None, with converged false.

What it writes
--------------
<data dir>/runs/diagnose/<stem>.json for the declared run (the tracking profile, all seven matches, the default number
of seeds); another profile, set of matches or number of seeds adds .<profile>, .m-<ids> and .s<n>, as the runner's
reports do (report_path), so that a quick look never replaces the declared report. The report is aggregates only, and
no list in it is longer than MAX_LIST (write_report refuses one that is); its layout is in diagnose(). Nothing of
BAR-ATH is read beyond what every degraded benchmark run reads, its Calibration, whose digest alone is written.

Memory: one match at a time, as the runner; the (p, y) of each measure are kept until the end, about 10 bytes a frame a
measure (some 3 million graded frames for 7 matches x 3 seeds), and the posterior's forward table of a half is
n_frames x (2 min_possession + min_dead) x 8 bytes (about 55 MB for a DFL half at main.toml's minimums).
"""

import argparse
import hashlib
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, as
    # run.py does, so that `import pipeline` resolves and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import degrade, dfl, run, score  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError  # noqa: E402
from pipeline.contracts import A, D, H, X  # noqa: E402
from pipeline.decode import hmm  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
DEFAULT_PROFILE = run.DEFAULT_PROFILE
IOUS = score.IOUS                     # the scorer's IoUs, 0.3 ... 0.7
FLOOR = IOUS[0]                       # the matching at 0.3, of which the matching at every other IoU is a cut
CLASSES = ("clean", "merge", "split", "both")   # a matched pair's class (A19), in the order they are reported
START_KINDS = ("team", "out_of_d", "edge")   # C1: what the truth's label just before a true start is (edge: none)
END_KINDS = ("team", "into_d", "edge")       # ... and what the label just after a true end is
BALL_STATES = ("seen", "gap")                # ... crossed with the ball at the true boundary frame
CELL_STATES = BALL_STATES + ("all",)         # a cell's states: the two, and both pooled (quartiles do not pool)
EDGE_TOLERANCE = 5                    # frames: the share of the clean pairs' edge errors within +- this
# EDGE_TOLERANCE is named in the report's key "within_5": change both together.
GATE = 0.015                          # A19: R025 is built only if the mean clean-pair ladder loss is at least this
BINS = 10                             # equal-width bins of the ECE and the reliability table
CLIP = 1e-6                           # p is clipped to [CLIP, 1 - CLIP] for the log-loss and Platt's logit
NEWTON_STEPS = 100                    # the most Newton iterations of the Platt fit
NEWTON_TOLERANCE = 1e-10              # a Newton step this small in both parameters ends the fit, converged
SMALLEST_STEP = 1e-12                 # step halving gives up below this fraction of a Newton step
SHAPE_KINDS = ("shape_possession",)   # home sources also measured apart where the ball is seen and where it is not
POSTERIOR = "posterior"               # the decoder's key in the calibration block, beside the sources' names
NO_TEAM_SOURCE = "none"               # alive_by_team_sources: the frames on which no source has a `home`
OTHER_SPEAKERS = "other"              # ... and the combinations folded together beyond MAX_SPEAKER_KEYS
MAX_SPEAKER_KEYS = 16                 # the most keys alive_by_team_sources has (2^k for k team sources, at most)
RESERVED = (POSTERIOR, NO_TEAM_SOURCE, OTHER_SPEAKERS)   # names a source may not have: the report uses them
REPORT_FOLDER = ("runs", "diagnose")  # <data dir>/runs/diagnose/<stem>[.<profile>][.m-<ids>][.s<n>].json
MAX_LIST = 64                         # the longest list a report may hold: aggregates only
POSTERIOR_PARAMETERS = ("switch_team", "switch_dead", "min_possession", "min_dead", "eps", "team_neutral_dead")


# -- B, the ladder ----------------------------------------------------------------------------------------------------

def greedy_pairs(overlaps, threshold=FLOOR):
    """[(gt index, pred index, iou)]: the pairs pm.matched_gt_flags matches at `threshold`, in the order it takes them.

    The same pass over the same list (pm.spell_overlaps, sorted by IoU descending, then gt and pred index): stop at the
    first IoU below the threshold, skip a pair either of whose spells is taken, take the rest.
    """
    pairs, gt_taken, pred_taken = [], set(), set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if gt_index in gt_taken or pred_index in pred_taken:
            continue
        gt_taken.add(gt_index)
        pred_taken.add(pred_index)
        pairs.append((gt_index, pred_index, iou))
    return pairs


def pair_class(pred_overlaps, gt_overlaps):
    """The class of a matched pair from how many same-team spells its prediction and its true spell each overlap."""
    if pred_overlaps >= 2 and gt_overlaps >= 2:
        return "both"
    if pred_overlaps >= 2:
        return "merge"
    if gt_overlaps >= 2:
        return "split"
    return "clean"


def boundary_kinds(truth, start, stop, gt_start, gt_stop):
    """(start kind, end kind) of a true spell [gt_start, gt_stop) in the half [start, stop) of `truth`.

    Start: "edge" at the half's first frame, else "out_of_d" if the frame before is D, else "team" (the other team: a
    spell is a maximal run, so the frame before it is never its own team's). End: "edge" where the spell reaches the
    half's end, else "into_d" if the frame after it is D, else "team".
    """
    if gt_start == start:
        before = "edge"
    else:
        before = "out_of_d" if truth[gt_start - 1] == D else "team"
    if gt_stop == stop:
        after = "edge"
    else:
        after = "into_d" if truth[gt_stop] == D else "team"
    return before, after


def ladder_half(truth, codes, start, stop, ball_present=None):
    """One graded half's ladder: summable counts, and the clean pairs' IoUs and edge errors as lists.

    `truth` and `codes` are the match's int8 codes (the truth without X inside [start, stop), the prediction as
    run.predict gives it); the half is [start, stop). Returns {"n_gt", "n_pred", "tp" (per IoU of IOUS: the scorer's
    pm.matched_gt_flags count), "lost" ({class: per IoU, the pairs matched at 0.3 and not at it}), "pairs" ({class:
    pairs matched at 0.3}), "pred_multi" (predicted spells overlapping 2 or more true ones of their team), "gt_multi"
    (true spells overlapping 2 or more predicted ones), "unmatched" ({"gt", "pred"}: {"overlapping", "alone"}, the
    spells unmatched at 0.3 by whether they overlap a spell of their team), "clean_iou", "clean_start_error",
    "clean_end_error"}, and, only with `ball_present` (C1: a bool per frame of the match, the ball seen), "clean_edges"
    ({"start", "end"}: per clean pair, in the order of clean_iou, (kind, ball state, error) with the kind from
    boundary_kinds, the ball state "seen" or "gap" on the true spell's first frame for a start and its last frame for
    an end, and the error predicted minus true). ValueError for arrays of different shapes, a mask of another shape,
    a half outside them, or an X in the truth's half.
    """
    truth, codes = np.asarray(truth), np.asarray(codes)
    if truth.ndim != 1 or truth.shape != codes.shape:
        raise ValueError("ladder_half: the truth has shape {} and the codes {}; both must be the match's codes, one "
                         "per frame".format(truth.shape, codes.shape))
    if not 0 <= start < stop <= len(truth):
        raise ValueError("ladder_half: the half [{}, {}) is not inside the {} frames of the codes".format(
            start, stop, len(truth)))
    if (truth[start:stop] == X).any():
        raise ValueError("ladder_half: the truth has X inside the half [{}, {}); a graded half of the truth is H, A "
                         "and D only (pass the graded blocks, as score.score_halves takes them)".format(start, stop))
    if ball_present is not None:
        ball_present = np.asarray(ball_present)
        if ball_present.shape != truth.shape:
            raise ValueError("ladder_half: ball_present has shape {} and the codes {}; it must hold one bool per frame "
                             "of the match".format(ball_present.shape, truth.shape))
        ball_present = ball_present.astype(bool)
    runs = [(start, stop)]
    gt_spells = pm.possession_spells(truth, runs)
    pred_spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    n_gt, n_pred = len(gt_spells), len(pred_spells)
    gt_count = np.bincount(np.array([gt for _, gt, _ in overlaps], dtype=np.int64), minlength=n_gt)
    pred_count = np.bincount(np.array([pred for _, _, pred in overlaps], dtype=np.int64), minlength=n_pred)

    pairs = greedy_pairs(overlaps)
    classes = [pair_class(pred_count[pred], gt_count[gt]) for gt, pred, _ in pairs]
    tp, lost = [], {name: [] for name in CLASSES}
    for iou in IOUS:
        flags = pm.matched_gt_flags(overlaps, n_gt, iou)
        cut = sorted(gt for gt, _, value in pairs if value >= iou)
        if cut != np.flatnonzero(flags).tolist():
            raise AssertionError("ladder_half: the matching at {} is not the matching at {} cut there, as the greedy "
                                 "pass of pm.matched_gt_flags makes it; the scorer has changed, and A19's ladder no "
                                 "longer reads it".format(iou, FLOOR))
        tp.append(int(flags.sum()))
        for name in CLASSES:
            lost[name].append(sum(1 for (_, _, value), kind in zip(pairs, classes) if kind == name and value < iou))

    gt_matched = {gt for gt, _, _ in pairs}
    pred_matched = {pred for _, pred, _ in pairs}
    unmatched = {"gt": {"overlapping": 0, "alone": 0}, "pred": {"overlapping": 0, "alone": 0}}
    for side, count, matched in (("gt", gt_count, gt_matched), ("pred", pred_count, pred_matched)):
        for index in range(len(count)):
            if index not in matched:
                unmatched[side]["overlapping" if count[index] else "alone"] += 1
    clean = [(gt, pred, value) for (gt, pred, value), kind in zip(pairs, classes) if kind == "clean"]
    half = {"n_gt": n_gt, "n_pred": n_pred, "tp": tp, "lost": lost,
            "pairs": {name: classes.count(name) for name in CLASSES},
            "pred_multi": int((pred_count >= 2).sum()), "gt_multi": int((gt_count >= 2).sum()),
            "unmatched": unmatched,
            "clean_iou": [value for _, _, value in clean],
            "clean_start_error": [pred_spells[pred][0] - gt_spells[gt][0] for gt, pred, _ in clean],
            "clean_end_error": [pred_spells[pred][1] - gt_spells[gt][1] for gt, pred, _ in clean]}
    if ball_present is not None:
        edges = {"start": [], "end": []}
        for gt, pred, _ in clean:
            gt_start, gt_stop = gt_spells[gt][:2]
            pred_start, pred_stop = pred_spells[pred][:2]
            before, after = boundary_kinds(truth, start, stop, gt_start, gt_stop)
            edges["start"].append((before, "seen" if ball_present[gt_start] else "gap", pred_start - gt_start))
            edges["end"].append((after, "seen" if ball_present[gt_stop - 1] else "gap", pred_stop - gt_stop))
        half["clean_edges"] = edges
    return half


def spell_f1(tp, n_gt, n_pred):
    """The F1 at one IoU from pooled counts, as score.pooled (and pm.spell_scores) compute it: 0.0 where empty."""
    precision = tp / float(n_pred) if n_pred else 0.0
    recall = tp / float(n_gt) if n_gt else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def ladder_summary(halves):
    """The ladder of `halves` (ladder_half's results) pooled: counts summed first, every figure from the sums.

    {"halves", "n_gt", "n_pred", "spell_f1" (score.pooled's for the same halves), "by_iou" ({IoU: {"tp", "f1", "lost":
    {class: pairs}, "loss": {class: spellF1}}}), "clean_loss", "merge_loss", "split_loss", "both_loss", "pairs" ({class:
    pairs matched at 0.3}), "merge_share", "split_share" (None with nothing to share), "unmatched", "clean_pairs" ({"n",
    "iou_median", "iou_share_at_least": {IoU: share}, "start_error", "end_error": {"median", "q25", "q75", "within_5"}},
    its figures None where there is no clean pair)}, and, if the halves were made with a ball mask, "edges_by_kind"
    ({"start": {kind of START_KINDS: {state of CELL_STATES: {"n", "median", "q25", "q75", "within_5"}}}, "end": the
    same by END_KINDS; state "all" pools "seen" and "gap"}: C1, the clean pairs' edge errors per cell, Nones for an
    empty one; the cells' counts add up to clean_pairs' n). The per-IoU losses of a class add up to its loss, to rounding.
    ValueError for no half, or for halves of which only some were made with a ball mask.
    """
    halves = list(halves)
    if not halves:
        raise ValueError("ladder_summary: no half to pool; an empty list would hide the mistake that emptied it")
    with_edges = ["clean_edges" in half for half in halves]
    if any(with_edges) and not all(with_edges):
        raise ValueError("ladder_summary: {} of {} halves were made with a ball_present mask and the others without; "
                         "the cells by ball state would count some of the clean pairs only".format(
                             sum(with_edges), len(halves)))
    n_gt = sum(half["n_gt"] for half in halves)
    n_pred = sum(half["n_pred"] for half in halves)
    tp = [sum(half["tp"][index] for half in halves) for index in range(len(IOUS))]
    lost = {name: [sum(half["lost"][name][index] for half in halves) for index in range(len(IOUS))]
            for name in CLASSES}
    f1s = [spell_f1(count, n_gt, n_pred) for count in tp]
    actual = float(np.mean(f1s))
    summary = {"halves": len(halves), "n_gt": n_gt, "n_pred": n_pred, "spell_f1": actual, "by_iou": {}}
    for index, iou in enumerate(IOUS):
        summary["by_iou"][iou] = {
            "tp": tp[index], "f1": f1s[index],
            "lost": {name: lost[name][index] for name in CLASSES},
            "loss": {name: (spell_f1(tp[index] + lost[name][index], n_gt, n_pred) - f1s[index]) / len(IOUS)
                     for name in CLASSES}}
    for name in CLASSES:
        kept = [spell_f1(tp[index] + lost[name][index], n_gt, n_pred) for index in range(len(IOUS))]
        summary[name + "_loss"] = float(np.mean(kept)) - actual
    summary["pairs"] = {name: sum(half["pairs"][name] for half in halves) for name in CLASSES}
    summary["merge_share"] = sum(half["pred_multi"] for half in halves) / n_pred if n_pred else None
    summary["split_share"] = sum(half["gt_multi"] for half in halves) / n_gt if n_gt else None
    summary["unmatched"] = {side: {where: sum(half["unmatched"][side][where] for half in halves)
                                   for where in ("overlapping", "alone")} for side in ("gt", "pred")}
    ious = np.array([value for half in halves for value in half["clean_iou"]], dtype=np.float64)
    summary["clean_pairs"] = {
        "n": len(ious),
        "iou_median": float(np.median(ious)) if len(ious) else None,
        "iou_share_at_least": {iou: float((ious >= iou).mean()) if len(ious) else None for iou in IOUS},
        "start_error": _errors([value for half in halves for value in half["clean_start_error"]]),
        "end_error": _errors([value for half in halves for value in half["clean_end_error"]])}
    if all(with_edges):
        summary["edges_by_kind"] = {
            side: {kind: {state: _errors([error for half in halves for k, s, error in half["clean_edges"][side]
                                          if k == kind and state in (s, "all")], count=True)
                           for state in CELL_STATES}
                   for kind in kinds} for side, kinds in (("start", START_KINDS), ("end", END_KINDS))}
    return summary


def _errors(values, count=False):
    """{"median", "q25", "q75", "within_5"} of signed frame errors (numpy's linear percentiles); Nones for none.
    With `count`, {"n", ...}: C1's cells."""
    values = np.asarray(values, dtype=np.float64)
    head = {"n": len(values)} if count else {}
    if not len(values):
        return dict(head, median=None, q25=None, q75=None, within_5=None)
    q25, median, q75 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    return dict(head, median=median, q25=q25, q75=q75, within_5=float((np.abs(values) <= EDGE_TOLERANCE).mean()))


# -- D, calibration ---------------------------------------------------------------------------------------------------

def calibration(p, y, bins=BINS):
    """How far the probabilities `p` mean what they say of the outcomes `y` (1 or True where the event happened).

    {"n", "mean_p", "mean_y", "log_loss", "brier", "ece", "reliability", "alpha", "beta", "converged"}: the log-loss
    with p clipped to [CLIP, 1 - CLIP]; the Brier score; the ECE over `bins` equal-width bins, the sum over the bins of
    n_bin / n x |mean p - mean y| (bin k holds floor(p x bins) = k, and p = 1 the last); `reliability`, one row per bin,
    {"low", "high", "n", "mean_p", "mean_y"} (the means None for an empty bin); and Platt's alpha and beta (module
    docstring), None where the likelihood has no maximum (and alpha None where logit(p) takes one value). With n = 0
    every figure is None and the bins are empty. ValueError for p outside [0, 1] or not finite, y other than 0 and 1,
    arrays of different lengths, or bins below 1.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("calibration: bins must be a whole number of at least 1, got {!r}".format(bins))
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
        raise ValueError("calibration: p and y must be one-dimensional and of the same length, got shapes {} and {}"
                         .format(p.shape, y.shape))
    if len(p) and (not np.isfinite(p).all() or p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("calibration: p must hold probabilities in [0, 1], and {} of {} are not".format(
            int((~np.isfinite(p) | (p < 0.0) | (p > 1.0)).sum()), len(p)))
    if len(y) and not np.isin(y, (0, 1)).all():
        raise ValueError("calibration: y must be 0 or 1 (or a boolean), and {} of {} values are neither".format(
            int((~np.isin(y, (0, 1))).sum()), len(y)))
    y = y.astype(bool)
    n = len(p)
    index = np.minimum((p * bins).astype(np.int64), bins - 1)
    counts = np.bincount(index, minlength=bins)
    sum_p = np.bincount(index, weights=p, minlength=bins)
    sum_y = np.bincount(index, weights=y.astype(np.float64), minlength=bins)
    reliability = [{"low": k / bins, "high": (k + 1) / bins, "n": int(counts[k]),
                    "mean_p": float(sum_p[k] / counts[k]) if counts[k] else None,
                    "mean_y": float(sum_y[k] / counts[k]) if counts[k] else None} for k in range(bins)]
    if not n:
        return {"n": 0, "mean_p": None, "mean_y": None, "log_loss": None, "brier": None, "ece": None,
                "reliability": reliability, "alpha": None, "beta": None, "converged": False}
    clipped = np.clip(p, CLIP, 1.0 - CLIP)
    alpha, beta, converged = platt(np.log(clipped) - np.log1p(-clipped), y)
    return {"n": n, "mean_p": float(p.mean()), "mean_y": float(y.mean()),
            "log_loss": -float(np.mean(np.where(y, np.log(clipped), np.log1p(-clipped)))),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": float(np.abs(sum_p - sum_y).sum() / n),
            "reliability": reliability, "alpha": alpha, "beta": beta, "converged": converged}


# diagnose() takes an argument named `calibration`, the runner's (the Calibration of BAR-ATH a profile is made with),
# which hides the function above inside it: it calls the function by this name.
_calibrate = calibration


def platt(x, y):
    """(alpha, beta, converged): the maximum-likelihood fit of P(y) = sigmoid(alpha x + beta), x = logit(p).

    Newton's method from (1, 0), the identity, with the step halved until the likelihood does not fall; it ends when a
    Newton step is below NEWTON_TOLERANCE in both parameters. The maximum exists only where both classes occur and
    overlap in x (in one dimension that is all it takes); otherwise (None, None, False). Where x takes one value, only
    beta is fitted, in closed form: (None, logit(mean y) - x, True).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return None, None, False
    if x.min() == x.max():
        share = float(y.mean())
        return None, float(np.log(share) - np.log1p(-share) - x[0]), True
    if x[~y].max() <= x[y].min() or x[y].max() <= x[~y].min():
        return None, None, False                     # separated: the likelihood rises for ever along alpha
    target = y.astype(np.float64)

    def likelihood(a, b):
        z = a * x + b
        return float(np.sum(target * z - np.logaddexp(0.0, z)))

    alpha, beta = 1.0, 0.0
    current = likelihood(alpha, beta)
    for _ in range(NEWTON_STEPS):
        z = alpha * x + beta
        s = np.exp(-np.logaddexp(0.0, -z))                                  # sigmoid(z), without overflow
        residual, weight = target - s, s * (1.0 - s)
        gradient = np.array([np.sum(residual * x), np.sum(residual)])
        information = np.array([[np.sum(weight * x * x), np.sum(weight * x)], [np.sum(weight * x), np.sum(weight)]])
        step = np.linalg.lstsq(information, gradient, rcond=None)[0]
        if np.abs(step).max() < NEWTON_TOLERANCE:
            return alpha, beta, True
        fraction = 1.0
        while True:
            trial = likelihood(alpha + fraction * step[0], beta + fraction * step[1])
            if trial >= current:
                break
            fraction /= 2.0
            if fraction < SMALLEST_STEP:
                return None, None, False
        alpha, beta, current = alpha + fraction * step[0], beta + fraction * step[1], trial
    return None, None, False


def speaker_groups(codes, names, limit=MAX_SPEAKER_KEYS):
    """{key: [codes]}: the frames' speaker codes grouped under the names of the team sources that speak on them.

    Bit i of a code is set where the i-th of `names` has a finite `home` on the frame. A key is those names, sorted and
    joined by "+", or NO_TEAM_SOURCE for a code of 0. There are at most 2^k keys for k sources; beyond `limit` of them,
    the limit - 1 that hold the most frames are kept (a tie by key) and the rest are folded under OTHER_SPEAKERS, so the
    report stays an aggregate however many sources a config has.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("speaker_groups: limit must be a whole number of at least 2, got {!r}".format(limit))
    present, counts = np.unique(np.asarray(codes, dtype=np.int64), return_counts=True)
    entries = []
    for code, count in zip(present.tolist(), counts.tolist()):
        spoke = sorted(name for bit, name in enumerate(names) if code >> bit & 1)
        entries.append((-count, "+".join(spoke) or NO_TEAM_SOURCE, code))
    entries.sort()
    if len(entries) <= limit:
        return {key: [code] for _, key, code in entries}
    groups = {key: [code] for _, key, code in entries[:limit - 1]}
    groups[OTHER_SPEAKERS] = sorted(code for _, _, code in entries[limit - 1:])
    return groups


class _Pool:
    """The (p, y) of one measure, gathered half by half under a key, with a flag or a code per frame to split them by.

    `arrays` joins them in key order, so that the figures do not depend on the order the matches ran in."""

    def __init__(self):
        self._parts = []

    def add(self, key, p, y, flag=None):
        p = np.asarray(p, dtype=np.float64)
        flag = np.zeros(len(p), dtype=bool) if flag is None else np.asarray(flag)
        self._parts.append((key, p, np.asarray(y, dtype=bool), flag))

    def arrays(self):
        parts = sorted(self._parts, key=lambda part: part[0])
        if not parts:
            return np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
        return tuple(np.concatenate([part[column] for part in parts]) for column in (1, 2, 3))


# -- the run ----------------------------------------------------------------------------------------------------------

def posterior_parameters(config):
    """{switch_team, switch_dead, min_possession, min_dead, eps, team_neutral_dead}: the config's [decoder] parameters
    for hmm.marginals, hmm.decode's defaults where it leaves one out. BenchmarkError unless the decoder is hmm without restart_source."""
    if config.decoder.kind != "hmm":
        raise run.BenchmarkError(
            "{}: the decoder is {!r}, which has no posterior; A19 reads the posterior of the hmm decoder "
            "(hmm.marginals), so diagnose a config whose [decoder] kind is \"hmm\"".format(
                config.stem, config.decoder.kind))
    values = {name: parameter.default for name, parameter in inspect.signature(hmm.decode).parameters.items()
              if parameter.default is not inspect.Parameter.empty}
    values.update(config.decoder.params)
    if values["restart_source"] is not None:
        raise run.BenchmarkError(
            "{}: [decoder] sets restart_source = {!r}; the restart states have no posterior (hmm.marginals is the "
            "three states' only), so diagnose the config without restart_source".format(
                config.stem, values["restart_source"]))
    return {name: values[name] for name in POSTERIOR_PARAMETERS}


def _predict(config, inputs):
    """run.predict; an asset that a benchmark match does not have is a BenchmarkError that names its source."""
    try:
        return run.predict(config, inputs)
    except match_inputs.AssetUnavailable as error:
        raise run.BenchmarkError(run._unavailable(config, inputs, error)) from error


def diagnose(config_path, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None, calibration=None,
             data_dir=None, drop_match_only=False, log=print):
    """A19 of one config: the ladder under both truths and the calibration of its sources and posterior; the report.

    The arguments are run.evaluate_configs': `profile` (default generic since P27), `seeds` (0 up to seeds - 1; default
    run.DEFAULT_SEEDS; the clean profile has seed 0 only), `match_ids` (default all seven), `matches` ({id: DflMatch}
    to use instead of loading: the checks pass synthetic ones), `calibration` (the Calibration of BAR-ATH a degraded
    profile is made with; none given, calibration.load(data_dir) reads it), `data_dir`, and `drop_match_only`, which
    removes the sources that declare MATCH_ONLY (main.toml's smart_tagging) as the runner's flag does and records them.
    `log` receives a line per match and then the tables (render).

    Refused with BenchmarkError (or ConfigError from the pipeline's loader) before any match is loaded: what the runner
    refuses (scope = "file", an unknown profile or match, seeds below 1, a calibration the profile does not take), a
    decoder that is not hmm or that sets restart_source (posterior_parameters), and a source named as one of the
    report's own keys (RESERVED: "posterior", "none", "other").

    Returns the report, JSON-ready (run.plain): {"config": {"path", "stem", "sha256"}, "profile", "seeds", "matches",
    "dropped_match_only", "ladder": {"raw", "coarse" (ladder_summary), "mean": {"spell_f1", "clean_loss", "split_loss",
    "merge_loss", "both_loss"}}, "calibration": {source name: {"kind", "weight", "home", "home_ball_visible",
    "home_ball_missing", "alive"} (the channels it speaks on), "posterior": {"alive": {"all", "team_speaks",
    "team_silent"}, "alive_by_team_sources": {key: ...} (speaker_groups), "home"}}, "tracking_calibration" ({"sha256",
    "version"} of the Calibration, None for clean), "git", "data_dir"}. Per-match figures are not kept.
    """
    configs, dropped = run._load_configs([config_path], drop_match_only)
    config = configs[0]
    decoder = posterior_parameters(config)
    names = [component.name for component in config.evidence]
    for name in RESERVED:
        if name in names:
            raise run.BenchmarkError("{}: an evidence source is named {!r}, which the report's calibration block uses "
                                     "for its own keys ({}); give it another name".format(config.stem, name,
                                                                                         ", ".join(RESERVED)))
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    if not run.needs_calibration(profile) and calibration is not None:
        raise run.BenchmarkError("profile {!r} takes no calibration, and one was given: it would be ignored, and a "
                                 "report must not suggest that it was used".format(profile))
    if run.needs_calibration(profile) and calibration is not None and not isinstance(calibration,
                                                                             calibration_module.Calibration):
        raise run.BenchmarkError("calibration must be a Calibration of our own match, as calibration.load() returns, "
                                 "and it is a {}".format(type(calibration).__name__))
    ids = run.resolve_match_ids(match_ids, matches)
    if run.needs_calibration(profile) and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    shape = {component.name for component in config.evidence if component.kind in SHAPE_KINDS}

    halves = {name: [] for name in run.TRUTHS}
    scored = {name: [] for name in run.TRUTHS}
    pools = {}                                       # (source name or POSTERIOR, channel) -> _Pool
    without_mass = 0                                 # the truth's alive frames whose P(H) + P(A) is 0

    def pool(name, channel):
        return pools.setdefault((name, channel), _Pool())

    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise run.BenchmarkError("the match given as {} says it is {}; its figures would be filed under the wrong "
                                     "id".format(match_id, match.match_id))
        raw, _ = truths.truth_codes(match)
        truth_of = {"raw": raw, "coarse": truths.coarse_codes(raw, match.graded, match.fps)}
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            codes, sources = _predict(config, inputs)
            for name in run.TRUTHS:
                found = score.score_halves(truth_of[name], codes, match.graded, match.match_id, seed)
                for (half, start, stop), counted in zip(match.graded, found):
                    rung = ladder_half(truth_of[name], codes, start, stop, inputs.atd.ball_present)
                    if (rung["n_gt"], rung["n_pred"], tuple(rung["tp"])) != (counted.n_gt, counted.n_pred,
                                                                              counted.tp):
                        raise AssertionError("diagnose: the ladder of {} half {} seed {} does not count what "
                                             "score.score_halves counts".format(match.match_id, half, seed))
                    halves[name].append(rung)
                scored[name] += found
            ball_seen = inputs.atd.ball_present
            for half, start, stop in match.graded:
                key = (match.match_id, seed, half)
                truth = raw[start:stop]
                live = truth != D
                speakers = np.zeros(stop - start, dtype=np.int64)       # bit i: the i-th source has a home here
                for bit, (name, source) in enumerate(sources.items()):
                    home, alive = source.home[start:stop], source.alive[start:stop]
                    known = ~np.isnan(home)
                    speakers |= known.astype(np.int64) << bit
                    if (known & live).any():
                        frames = known & live
                        pool(name, "home").add(key, home[frames], truth[frames] == H,
                                               ball_seen[start:stop][frames] if name in shape else None)
                    known = ~np.isnan(alive)
                    if known.any():
                        pool(name, "alive").add(key, alive[known], live[known])
                window = [source.window(start, stop) for source in sources.values()]
                emissions = hmm.log_likelihoods(window, stop - start, decoder["eps"], decoder["team_neutral_dead"])
                posterior = hmm.marginals(emissions, decoder["switch_team"], decoder["switch_dead"],
                                          decoder["min_possession"], decoder["min_dead"])
                in_play = posterior[:, H] + posterior[:, A]
                pool(POSTERIOR, "alive").add(key, np.clip(in_play, 0.0, 1.0), live, speakers)
                defined = live & (in_play > 0.0)
                pool(POSTERIOR, "home").add(key, np.clip(posterior[defined, H] / in_play[defined], 0.0, 1.0),
                                            truth[defined] == H)
                without_mass += int((live & ~defined).sum())
                del emissions, posterior, in_play
            del atd, inputs, codes, sources, ball_seen
        log("match     {}   {} halves x {} seed{}".format(match.match_id, len(match.graded), len(seed_list),
                                                           "" if len(seed_list) == 1 else "s"))
        del match, raw, truth_of                     # the next match is loaded only when this one is gone

    ladder = {}
    for name in run.TRUTHS:
        ladder[name] = ladder_summary(halves[name])
        official = score.pooled(scored[name])["spell_f1_mean"]
        if abs(ladder[name]["spell_f1"] - official) > 1e-12:
            raise AssertionError("diagnose: the ladder's pooled spellF1 under the {} truth is {!r} and score.pooled "
                                 "gives {!r}".format(name, ladder[name]["spell_f1"], official))
    ladder["mean"] = {figure: (ladder["raw"][figure] + ladder["coarse"][figure]) / 2.0
                      for figure in ("spell_f1", "clean_loss", "split_loss", "merge_loss", "both_loss")}

    measured = {}
    for component in config.evidence:
        entry = {"kind": component.kind, "weight": component.weight}
        if (component.name, "home") in pools:
            p, y, visible = pools.pop((component.name, "home")).arrays()
            entry["home"] = _calibrate(p, y)
            if component.name in shape:
                entry["home_ball_visible"] = _calibrate(p[visible], y[visible])
                entry["home_ball_missing"] = _calibrate(p[~visible], y[~visible])
        if (component.name, "alive") in pools:
            entry["alive"] = _calibrate(*pools.pop((component.name, "alive")).arrays()[:2])
        measured[component.name] = entry
    p, y, speakers = pools.pop((POSTERIOR, "alive")).arrays()
    speaks = speakers != 0
    alive = {"all": _calibrate(p, y), "team_speaks": _calibrate(p[speaks], y[speaks]),
             "team_silent": _calibrate(p[~speaks], y[~speaks])}
    by_speakers = {}
    for group, codes_in in speaker_groups(speakers, names).items():
        chosen = np.isin(speakers, codes_in)
        by_speakers[group] = _calibrate(p[chosen], y[chosen])
    p, y, _ = pools.pop((POSTERIOR, "home")).arrays()
    measured[POSTERIOR] = {"alive": alive, "alive_by_team_sources": by_speakers,
                           "home": dict(_calibrate(p, y), frames_without_alive_mass=without_mass)}
    del p, y, speakers, speaks

    report = run.plain({
        "config": {"path": run.shown(config.path), "stem": config.stem,
                   "sha256": hashlib.sha256(config.text.encode("utf-8")).hexdigest()},
        "profile": degrade.profile_record(profile),
        "seeds": seed_list,
        "matches": sorted(ids),
        "dropped_match_only": list(dropped[config.stem]),
        "ladder": ladder,
        "calibration": measured,
        "tracking_calibration": None if not run.needs_calibration(profile) else {
            "sha256": calibration_module.digest(calibration), "version": calibration_module.CALIBRATION_VERSION},
        "git": pipeline_run.git_state(),
        "data_dir": run.shown(data_dir or games.DATA_DIR),
    })
    for line in render(report):
        log(line)
    return report


# -- the report and the printout --------------------------------------------------------------------------------------

def report_path(data_dir, stem, profile, match_ids=None, seeds=None):
    """<data dir>/runs/diagnose/<stem>[.<profile>][.m-<ids>][.s<n>].json: where a report goes.

    The declared run (the default profile, all seven matches, the default number of seeds) is <stem>.json. The default
    profile is generic since P27, so a bare <stem>.json written before 10 Nov 2026 is A19's `tracking` report and one
    written since is generic's. A profile other than the default adds its name, and the matches and seeds add what run.name_suffixes adds to a runner's
    report for the same reason: a run on fewer matches or seeds never replaces the declared report.
    """
    parts = [stem] + ([profile] if profile != DEFAULT_PROFILE else []) + run.name_suffixes(profile, match_ids, seeds)
    return os.path.join(data_dir, *REPORT_FOLDER, ".".join(parts) + ".json")


def check_aggregate(value, where="report"):
    """ValueError naming the place if `value` holds, however deep, a list longer than MAX_LIST: aggregates only."""
    if isinstance(value, dict):
        for key, item in value.items():
            check_aggregate(item, "{}.{}".format(where, key))
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST:
            raise ValueError("{} holds a list of {} entries, and a report of diagnose holds aggregates only, no list "
                             "longer than {}".format(where, len(value), MAX_LIST))
        for number, item in enumerate(value):
            check_aggregate(item, "{}[{}]".format(where, number))


def write_report(report, data_dir=None):
    """Write `report` (diagnose's) at report_path under `data_dir` (default games.DATA_DIR), atomically, as
    run.write_json writes the runner's; the path. ValueError, before anything is written, for a list longer than
    MAX_LIST."""
    check_aggregate(report)
    path = report_path(data_dir or games.DATA_DIR, report["config"]["stem"], report["profile"]["name"],
                       report["matches"], len(report["seeds"]))
    run.write_json(path, report)
    return path


def render(report):
    """The lines diagnose() logs: the ladder under each truth and their mean, R025's gate, and the calibration table."""
    profile, git = report["profile"], report["git"]
    lines = ["",
             "diagnose  {}   profile {} (version {})   seeds {}   {} matches   commit {}{}".format(
                 report["config"]["stem"], profile["name"], profile["version"],
                 ",".join(str(seed) for seed in report["seeds"]), len(report["matches"]), git["commit"] or "?",
                 " +dirty" if git["dirty"] else "")]
    if report["dropped_match_only"]:
        lines.append("dropped   match-only sources: {} (--drop-match-only; the figures are those of the config without "
                     "them)".format(", ".join(report["dropped_match_only"])))
    for name in run.TRUTHS:
        lines += _ladder_lines(name, report["ladder"][name])
    mean = report["ladder"]["mean"]
    lines += ["",
              "ladder    mean of the two truths   spellF1 {}   losses: clean {}   merge {}   split {}   both {}".format(
                  *(_number(mean[figure]) for figure in ("spell_f1", "clean_loss", "merge_loss", "split_loss",
                                                         "both_loss"))),
              "gate      R025 (A19): the mean clean-pair ladder loss is {}, {} {}".format(
                  _number(mean["clean_loss"]), "at least" if mean["clean_loss"] >= GATE else "below", GATE)]
    lines += ["", "{:<66}{:>13}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "calibration (raw truth, graded frames)", "n", "log-loss", "Brier", "ECE", "alpha", "beta", "weight")]
    for name, entry in report["calibration"].items():
        if name == POSTERIOR:
            continue
        for channel in ("home", "home_ball_visible", "home_ball_missing", "alive"):
            if channel in entry:
                lines.append(_calibration_line("{} {}".format(name, channel), entry[channel], entry["weight"]))
    posterior = report["calibration"][POSTERIOR]
    for part in ("all", "team_speaks", "team_silent"):
        lines.append(_calibration_line("posterior alive, {}".format(part), posterior["alive"][part]))
    for group, measure in sorted(posterior["alive_by_team_sources"].items()):
        lines.append(_calibration_line("posterior alive, team sources {}".format(group), measure))
    lines.append(_calibration_line("posterior home", posterior["home"]))
    if posterior["home"]["frames_without_alive_mass"]:
        lines.append("  {:,} alive frames have P(H) + P(A) = 0, and no home ratio".format(
            posterior["home"]["frames_without_alive_mass"]))
    return lines


def _ladder_lines(name, ladder):
    lines = ["",
             "ladder    truth {}: {:,} true and {:,} predicted possessions in {} halves, pooled spellF1 {}".format(
                 name, ladder["n_gt"], ladder["n_pred"], ladder["halves"], _number(ladder["spell_f1"])),
             "          {:<6}{:>8}{:>9}   pairs lost since 0.3: {:>6}{:>7}{:>7}{:>6}".format(
                 "IoU", "matched", "F1", *CLASSES)]
    for iou, row in ladder["by_iou"].items():
        lines.append("          {:<6}{:>8,}{:>9}   {:>28}{:>7}{:>7}{:>6}".format(
            iou, row["tp"], _number(row["f1"]), *(row["lost"][kind] for kind in CLASSES)))
    unmatched, clean = ladder["unmatched"], ladder["clean_pairs"]
    lines += ["          losses in spellF1: clean {}   merge {}   split {}   both {}   (pairs matched at 0.3: {})"
              .format(*(_number(ladder[kind + "_loss"]) for kind in CLASSES),
                  ", ".join("{} {:,}".format(kind, ladder["pairs"][kind]) for kind in CLASSES)),
              "          merges: {} of predicted possessions overlap 2+ true ones of their team; splits: {} of true "
              "ones overlap 2+ predicted".format(_share(ladder["merge_share"]), _share(ladder["split_share"])),
              "          unmatched at 0.3: true {:,} overlapping, {:,} alone; predicted {:,} overlapping, {:,} "
              "alone".format(unmatched["gt"]["overlapping"], unmatched["gt"]["alone"], unmatched["pred"]["overlapping"],
                             unmatched["pred"]["alone"]),
              "          clean pairs {:,}: IoU median {}, at least {}".format(
                  clean["n"], _number(clean["iou_median"]),
                  "  ".join("{} {}".format(iou, _share(share)) for iou, share in clean["iou_share_at_least"].items()))]
    for side in ("start", "end"):
        errors = clean[side + "_error"]
        lines.append("          {:<5} error (predicted - true, frames): median {}, quartiles [{}, {}], within +-{} {}"
                     .format(side, _frames(errors["median"]), _frames(errors["q25"]), _frames(errors["q75"]),
                             EDGE_TOLERANCE, _share(errors["within_5"])))
    if "edges_by_kind" in ladder:
        lines += render_edges(name, ladder["edges_by_kind"])
    return lines


def render_edges(name, cells):
    """C1's table under the truth `name`: one row per boundary kind and ball state, the clean pairs' edge errors."""
    lines = ["          clean pairs' edge errors by boundary kind and ball state (truth {}, predicted - true, frames; "
             "ball at the true spell's first frame for a start, its last for an end)".format(name),
             "          {:<10}{:<10}{:<6}{:>8}{:>9}{:>9}{:>9}{:>10}".format(
                 "boundary", "kind", "ball", "n", "median", "q25", "q75", "within+-{}".format(EDGE_TOLERANCE))]
    for side, kinds in (("start", START_KINDS), ("end", END_KINDS)):
        for kind in kinds:
            for state in CELL_STATES:
                one = cells[side][kind][state]
                lines.append("          {:<10}{:<10}{:<6}{:>8,}{:>9}{:>9}{:>9}{:>10}".format(
                    side, kind, state, one["n"], _frames(one["median"]), _frames(one["q25"]), _frames(one["q75"]),
                    _share(one["within_5"])))
    return lines


def _calibration_line(label, measure, weight=None):
    return "  {:<64}{:>13,}{:>10}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        label, measure["n"], _number(measure["log_loss"]), _number(measure["brier"]), _number(measure["ece"]),
        _number(measure["alpha"]), _number(measure["beta"], signed=True), "" if weight is None else _number(weight))


def _number(value, signed=False):
    if value is None:
        return "n/a"
    return ("{:+.4f}" if signed else "{:.4f}").format(value)


def _share(value):
    return "n/a" if value is None else "{:.1%}".format(value)


def _frames(value):
    return "n/a" if value is None else "{:+.1f}".format(value)


# -- the command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A19: CONFIG's IoU ladder decomposed (clean, merge, split pairs) under both truths, and the "
                    "calibration of its sources and of its hmm decoder's posterior against the raw truth, on the "
                    "benchmark's matches. Aggregates only.")
    parser.add_argument("config", metavar="CONFIG", help="a TOML config whose [decoder] is hmm, e.g. "
                                                         "pipeline/configs/main.toml (with --drop-match-only)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--drop-match-only", action="store_true",
                        help="drop the evidence sources that need an asset a benchmark match does not have "
                             "(MATCH_ONLY, main.toml's smart_tagging), and record them in the report")
    parser.add_argument("--no-write", action="store_true", help="print only; write no report")
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.config, profile=args.profile, seeds=args.seeds, match_ids=args.matches,
                          data_dir=args.data_dir, drop_match_only=args.drop_match_only)
        path = None if args.no_write else write_report(report, args.data_dir)
    except (ConfigError, run.BenchmarkError, OSError) as error:
        # The expected failures get a sentence, as in run.py: a config to fix, a profile or match that does not exist,
        # data not downloaded yet. Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1
    print("")
    print("report    {}".format(run.shown(path)) if path else "report    not written (--no-write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
