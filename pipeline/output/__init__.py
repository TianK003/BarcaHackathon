"""Stage 5 -- output: write the submission, check it exactly as the organisers will, and say what it claims.

The deliverable is a file that our repository regenerates byte for byte (SUBMISSION.md, RULES.md), so this stage is
strict about bytes:

* header `frame,label`, one row per frame 0..n-1, LF line endings, written in binary mode -- text mode on Windows
  writes CRLF, which changes the file and its hash between machines (docs/traps.md T21);
* written to a temporary file and renamed into place, so an interrupted run never leaves half a submission;
* read back and checked with the official validator's own functions, plus a round trip of the labels;
* never named baseline_*.csv, and never left inside the repository where .gitignore does not cover it.

summarise() reports what the file claims -- label counts and the number of possessions, counted the way the scorer
counts them, inside the graded halves with X bridged (T06). score() runs the official metrics against a ground
truth; on the public window that is a correctness check, never a way to choose between versions (T01).
"""

import hashlib
import json
import os

import numpy as np

import possession_metrics as pm
from pipeline import ROOT
from pipeline.contracts import LETTERS

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
HEADER = "frame,label\n"
# .gitignore ignores evaluation/results/*.csv except baseline_*.csv, which the organisers commit on purpose. A file of
# ours with that prefix would overwrite one of theirs and be picked up by the exception (docs/traps.md T24).
RESERVED_PREFIX = "baseline_"


class SubmissionInvalid(ValueError):
    """The file fails the official checks: the scorer would skip it and score nothing."""


class UnsafeOutPath(ValueError):
    """Writing the submission there could put match-derived labels, or a fake baseline, into a commit."""


class ScoreError(ValueError):
    """The ground truth asked for cannot be scored against: not the same length as the labels."""


def default_out_path(stem):
    """evaluation/results/<stem>.csv -- where the scorer looks, and where .gitignore keeps it out of commits."""
    return os.path.join(RESULTS_DIR, stem + ".csv")


def check_out_path(path, data_dir):
    """UnsafeOutPath unless a submission may be written to `path` without risking the repository.

    Inside the repository, .gitignore covers exactly two places for a file like this: directly in
    evaluation/results/ (the pattern is evaluation/results/*.csv, so not in a sub-folder), and anywhere under data/.
    Outside the repository anything goes -- the final <team_name>.csv can live wherever the upload starts from.
    """
    path = os.path.abspath(path)
    name = os.path.basename(path)
    if not name.lower().endswith(".csv"):
        raise UnsafeOutPath("{}: a submission is a .csv file".format(path))
    # Lower-cased, because git on Windows and macOS matches .gitignore patterns case-insensitively
    # (core.ignorecase), so the exception would catch Baseline_x.csv as well.
    if name.lower().startswith(RESERVED_PREFIX):
        raise UnsafeOutPath("{}: names starting with {!r} belong to the organisers' committed baselines, and the "
                         ".gitignore exception for them would commit this file too (docs/traps.md T24)".format(
                             path, RESERVED_PREFIX))
    if _inside(path, ROOT) and not (_same(os.path.dirname(path), RESULTS_DIR) or _inside(path, data_dir)):
        raise UnsafeOutPath("{}: inside the repository, .gitignore only covers evaluation/results/*.csv and data/, so a "
                         "file of labels here could be committed (docs/traps.md T25). Write it to one of those, or "
                         "outside the repository.".format(path))


def write_submission(path, codes):
    """Write codes (the scorer's 0..3) as a submission CSV and return the sha256 of its bytes."""
    data = (HEADER + "".join("{},{}\n".format(frame, letter) for frame, letter in enumerate(LETTERS[codes]))
            ).encode("ascii")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # "<name>.csv.tmp" is gitignored (*.tmp), so even a crash between these two lines leaves nothing committable.
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def validate(path, n_frames, codes=None):
    """Raise SubmissionInvalid unless the file passes the official checks -- the same calls
    evaluation/validate_submission.py and the scorer make -- and, given codes, reads back as exactly those labels."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # unparseable CSV, missing columns: the scorer's own catch-all
        raise SubmissionInvalid("{}: could not read: {}".format(path, error))
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise SubmissionInvalid("{}: {}".format(os.path.basename(path), reason))
    if codes is not None and not np.array_equal(pm.label_codes(table), codes):
        raise SubmissionInvalid("{}: reading the file back does not give the labels that were written".format(
            os.path.basename(path)))


def summarise(codes, graded, fps):
    """What the labels claim: counts over the file and over graded frames, and the possessions inside the graded
    halves with X bridged -- the scorer's way of counting (docs/traps.md T06), not validate_submission.py's."""
    runs = [(start, stop) for _, start, stop in graded]
    graded_mask = np.zeros(len(codes), dtype=bool)
    for start, stop in runs:
        graded_mask[start:stop] = True
    spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    lengths = [stop - start for start, stop, _ in spells]
    return {
        "rows": int(len(codes)),
        "labels_all": _counts(codes),
        "labels_graded": _counts(codes[graded_mask]),
        "graded_frames": int(graded_mask.sum()),
        "possessions": len(spells),
        "mean_possession_s": round(float(np.mean(lengths)) / fps, 2) if lengths else 0.0,
    }


def score(codes, gt="public"):
    """(scores, gt_codes): the official metrics of codes against a ground truth.

    gt is resolved by the scorer's own rules: "public" for the 5 minutes in this repository, or a path to any
    frame,label CSV of this match, such as another run's output. There are no hand labels of this match to plug in
    (docs/traps.md T27); P01's benchmark is other matches. Scores here are for checking; the public window ranks the
    baselines the wrong way round (docs/traps.md T01).
    """
    import run_evaluation  # the official resolver for "public" / a path; imported late, only when scoring
    gt_path = run_evaluation.find_gt(gt)
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    if len(gt_codes) != len(codes):
        raise ScoreError("{}: {} rows, but the labels have {} -- a ground truth covers every frame of the tactical "
                         "video, X where it has no label".format(gt_path, len(gt_codes), len(codes)))
    result = pm.evaluate(gt_codes, codes)
    scores = {name: result[name] for name in ("macro_f1", "spell_f1_mean", "alive_dead_f1", "n_pred_spells",
                                              "n_gt_spells", "mean_spell_duration_s")}
    scores["ground_truth"] = (os.path.relpath(gt_path, ROOT).replace(os.sep, "/") if _inside(gt_path, ROOT)
                              else gt_path)
    return scores, gt_codes


def write_manifest(path, manifest):
    """The run's record as JSON -- sorted keys and LF, so two manifests of the same run compare equal."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        # default=_plain: numpy scalars are not JSON, and a future summary field will sooner or later be one.
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_plain)
        handle.write("\n")


def _plain(value):
    """np.int64(3) -> 3, np.float64(0.5) -> 0.5; anything else stays an error rather than becoming a string."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("{} is not JSON serialisable".format(type(value).__name__))


def _counts(codes):
    return {letter: int(np.sum(codes == code)) for code, letter in enumerate(pm.LABELS)}


def _inside(path, folder):
    """Is path inside folder (or folder itself)? Case-insensitive where the file system is (Windows)."""
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives on Windows: certainly not inside
        return False


def _same(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))
