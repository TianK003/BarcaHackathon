"""P35's trainer: fit the turnover hazard on the benchmark's matches, write it as JSON, and run its gate.

    python pipeline/benchmark/train_turnover.py [--profile generic] [--seeds 3] [--stride 5] [--out PATH] [--verify]
                                                [--data-dir D]
    python pipeline/benchmark/train_turnover.py --dry-run [--matches ID ...] [--seeds 1] [--data-dir D]
    python pipeline/benchmark/train_turnover.py --report [--out PATH] [--profile generic] [--seeds 3] [--data-dir D]

    artifact = train(profile="generic", seeds=3, stride=5)          # the dict that is written, JSON-ready
    print(dry_run("generic", 1, ["J03WMX"]))                         # rows, positives and NaN shares, nothing fitted
    cells = report(artifact, "generic", 3, 5)["per_match"]           # out of fold: AUC, log-loss, base-rate log-loss

This is `train_team.py` (P32) with another label, and everything that is not the label is imported from there and from
P05's and P07's trainers, not rewritten: `HGB_SETTINGS` (never tuned), `export_hgb`, `compare`, `dumps` and `write`,
`served_inputs`, the classifier itself, the dry run's peak-memory helper and projection constants, and the helpers that
refuse a profile, a calibration given to `generic` or a match filed under another id. Read train_team.py's docstring for
the argument; what differs is below.

The question. Not "which team has the ball" but "does a direct change of team lie within 12 frames of this one": a
turnover hazard, to be read by the hmm decoder as a per-frame switch cost (P35, pipeline/evidence/turnover_hazard.py).

The rows are P32's: every `stride`-th frame of a graded block from its start where the raw truth is H or A and the
served ball is visible. The label is 1 where a direct change of team (the truth's code at c - 1 and at c both alive and
different, no D between) lies at c with |t - c| <= WINDOW_FRAMES in the same graded block, 0 otherwise. A change across
a dead ball is not a direct one; a change at the first frame of a block has no frame before it in the block. The label is
read from `truth.truth_codes(match)` beside the features and never through the inputs.

Every row is shown twice, as it is and as `team_features.swap(row)` with the **same** label (a change is a change
whichever team is which), so the classes keep their own balance and the model's `base_rate`, the share of the rows
that are positive, is stored in each model for the source's z = p / (p + base_rate). The source's score is symmetrised
(r(x) + r(swap x)) / 2, so it is swap-invariant to rounding.

The artifact is pipeline/models/turnover_hgb.json: train_team's layout with `target` "team_change_pm12", `window_frames`
12 and `base_rate` in each of the folds (fitted on the other six matches) and `full`; `feature_version` 2.

--dry-run gathers rows and fits nothing, as train_team's: rows per match, the positive share, the bytes of X, the NaN
share of every feature, the names of any all-NaN or constant column, the peak memory and the 8 GB projection. The one
declared run (J03WMX, `generic`, one seed, stride 5) must land in ROW_WINDOW (P32's rows, amendment 4 of the P35 design)
and its positive share in POSITIVE_WINDOW, and the projected peak must be at most 8 GB, or the exit code is 1.

--report is the gate (P35, step 0): the artifact's folds scored each on its own held-out match, served as the pipeline
serves it, on its rows. Pooled and per match: the AUC (a plain rank formula, ties averaged), the log-loss and the
log-loss of the fold's own training base rate said at every row. GO iff the pooled AUC is at least AUC_MIN and the
pooled log-loss is at most LOSS_RATIO_MAX times the pooled base-rate log-loss (a screen, not evidence: amendment 4).
Beside it, gating nothing: for amendment 10's transfer gate, per match the mean p (clipped as the source clips it) over
every graded frame with the ball seen, alive or dead; and the localisation: for each true direct change with the ball
seen, the offset of p's maximum within +-LOCALISE_FRAMES (inside its block): median, quartiles, the share within +-5.
The output is sums and ratios: no per-frame array.
"""

import argparse
import hashlib
import json
import numbers
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import train_team  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
from pipeline.benchmark.train_dead_ball import (  # noqa: E402
    HGB_SETTINGS, LOG_FLOOR, Gathered, compare, dumps, export_hgb, served_inputs, write)
from pipeline.benchmark.train_shape import hgb_estimator  # noqa: E402
from pipeline.evidence import kinematics, team_features, trees, turnover_hazard  # noqa: E402

import games  # noqa: E402  (importing pipeline above put the repository root on sys.path)
import possession_metrics as pm  # noqa: E402  (importing pipeline above put evaluation/ on sys.path)

# Params
FORMAT = trees.FORMAT
KIND = "hgb"
TARGET = turnover_hazard.TARGET
WINDOW_FRAMES = turnover_hazard.WINDOW_FRAMES
HOME, AWAY = pm.H, pm.A
DEFAULT_STRIDE = train_team.DEFAULT_STRIDE
DEFAULT_MODEL_PATH = os.path.join(ROOT, "pipeline", "models", "turnover_hgb.json")
SOURCE_FILES = (("train_turnover.py", os.path.join(HERE, "train_turnover.py")),
                ("turnover_hazard.py", os.path.join(ROOT, "pipeline", "evidence", "turnover_hazard.py"))
                ) + tuple(train_team.SOURCE_FILES[1:])
TRAINING_ROWS = train_team.TRAINING_ROWS      # P32's rows are P35's (amendment 4)
COPIES = train_team.COPIES
PEAK_LIMIT_GB = train_team.PEAK_LIMIT_GB
ROW_WINDOW = (8_290, 12_434)                  # P32's window (amendment 4): 10,208 expected for J03WMX seed 0, +-20%
ROW_WINDOW_RUN = ("generic", 1, DEFAULT_STRIDE, ["J03WMX"])
POSITIVE_WINDOW = (0.06, 0.14)                # the positive share of the declared run's rows (336 changes x 25 / 79,015)
AUC_MIN = 0.80                                # the gate: pooled held-out AUC at least this
LOSS_RATIO_MAX = 0.90                         # ... and pooled held-out log-loss at most this times the base rate's
GATE_TOLERANCE = 1e-12                        # the pool is a ratio of float sums
LOCALISE_FRAMES = 25                          # the localisation's search window around a true change
LOCALISE_NEAR = 5                             # its share "within +-5"
REPORT_FOLDER = ("runs", "p35")


# -- the label and the rows ---------------------------------------------------------------------------------------------

def direct_changes(codes, graded):
    """[(frame, start, stop)]: every frame c of a graded block (start, stop) where the truth changes team directly.

    That is codes[c - 1] and codes[c] both H or A and different, with c - 1 in the same block: a change through D is
    two changes of state and none of team, and the first frame of a block has no frame before it. In frame order.
    """
    codes = np.asarray(codes)
    found = []
    for _, start, stop in graded:
        block = codes[start:stop]
        alive = (block == HOME) | (block == AWAY)
        changed = alive[1:] & alive[:-1] & (block[1:] != block[:-1])
        found.extend((start + 1 + int(i), start, stop) for i in np.flatnonzero(changed))
    return found


def change_labels(codes, graded, window=WINDOW_FRAMES):
    """(n_frames,) bool: True where a direct change of team lies within `window` frames, in the same graded block.

    Frames outside the blocks are False. A change at c marks c - window ... c + window clipped to its block, so a change
    near a block's edge marks only the frames inside it. ValueError for a window that is not a whole number >= 0.
    """
    if isinstance(window, bool) or not isinstance(window, numbers.Integral) or window < 0:
        raise ValueError("train_turnover: window must be a whole number of at least 0, got {!r}".format(window))
    labels = np.zeros(len(codes), dtype=bool)
    for frame, start, stop in direct_changes(codes, graded):
        labels[max(start, frame - window):min(stop, frame + window + 1)] = True
    return labels


def row_frames(inputs, codes, stride):
    """The frames of the rows: every `stride`-th frame of each graded block from its start where the truth is H or A and
    the served ball is visible (train_team.rows' frames). ValueError for a stride that is not a whole number >= 1 and
    for `codes` that do not hold one code per frame."""
    stride = _whole("stride", stride, 1)
    codes = np.asarray(codes)
    if codes.shape != (inputs.n_frames,):
        raise ValueError("train_turnover: codes must hold one code per frame of the match ({}), got shape {}"
                         .format(inputs.n_frames, codes.shape))
    frames = np.concatenate([np.arange(start, stop, stride) for _, start, stop in inputs.graded])
    visible = kinematics.ball_track(inputs).visible
    return frames[((codes[frames] == HOME) | (codes[frames] == AWAY)) & visible[frames]]


def rows(inputs, codes, stride):
    """(X, y): the features of the rows `row_frames` names, and 1 where a direct change lies within WINDOW_FRAMES.

    `codes` is the truth's int8 pm codes per frame of the match, kept apart from the inputs, which have none.
    """
    frames = row_frames(inputs, codes, stride)
    labels = change_labels(codes, inputs.graded)
    _, X = team_features.features(inputs)
    return X[frames], labels[frames].astype(np.int64)


def gather(profile, seeds, stride, matches=None, calibration=None, data_dir=None, match_ids=None, log=None):
    """Gathered: the rows of every match under every seed of `profile`, served as the pipeline serves them.

    As train_team.gather, with this module's rows. BenchmarkError for an unknown profile, fewer than two matches, a
    match filed under another id than its own, and a calibration given to a profile that takes none.
    """
    return _gather(profile, seeds, stride, 2, matches, calibration, data_dir, match_ids, log)


def _gather(profile, seeds, stride, minimum, matches, calibration, data_dir, match_ids, log):
    seeds = _whole("seeds", seeds, 1)
    _whole("stride", stride, 1)
    train_team._known_profile(profile)
    ids = run.resolve_match_ids(match_ids, matches)
    if len(ids) < minimum:
        raise BenchmarkError("train_turnover: a fold is fitted on the other matches, so it needs at least two "
                             "matches, got {}".format(len(ids)))
    calibration = train_team._calibration_for(profile, calibration, data_dir)
    seed_list = train_team._seed_list(profile, seeds)
    log = log or (lambda text: None)

    data, sources = {}, {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        train_team._own_id(match, match_id)
        codes, _ = truths.truth_codes(match)
        parts = []
        for seed in seed_list:
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            parts.append(rows(inputs, codes, stride))
            if seed == seed_list[0]:
                recorded = inputs.atd_manifest["sources"]
                sources[match_id] = None if recorded is None else {name: recorded[name]["sha256"]
                                                                   for name in sorted(recorded)}
            del inputs
        data[match_id] = (np.concatenate([X for X, _ in parts]), np.concatenate([y for _, y in parts]))
        log("rows      {}   {:,} rows, {:.1%} near a change".format(
            match_id, len(data[match_id][1]), float(data[match_id][1].mean()) if len(data[match_id][1])
            else float("nan")))
        del match, codes
    digest = calibration_module.digest(calibration) if run.needs_calibration(profile) else None
    return Gathered(data, digest, sources, len(seed_list))


# -- the model --------------------------------------------------------------------------------------------------------

def augment(X, y):
    """(X', y'): the rows, then their mirror images (`team_features.swap`) with the SAME label.

    ValueError for X that is not (n, 72) or a y that is not one 0 or 1 per row.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    names = len(team_features.FEATURE_NAMES)
    if X.ndim != 2 or X.shape[1] != names or y.shape != (len(X),) or not np.isin(y, (0, 1)).all():
        raise ValueError("train_turnover: X must be (n, {}) and y one 0 or 1 per row, got X {} and y {}"
                         .format(names, X.shape, y.shape))
    return np.concatenate([X, team_features.swap(X)]), np.concatenate([y, y]).astype(np.int64)


def fit(X, y):
    """One model, as a JSON-ready dict {baseline, trees, base_rate}: fitted on the rows X, y and their mirrors.

    `base_rate` is the share of the rows that are positive (the mirrors keep it). ValueError for the shapes `augment`
    refuses and for rows of one class only: a hazard needs frames with a change near them and frames without.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    augmented = augment(X, y)                       # checks the shapes
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("train_turnover: the rows hold one class only ({} near a change among {}); a hazard needs "
                         "both".format(int(y.sum()), len(y)))
    model = export_hgb(hgb_estimator(*augmented))
    model["base_rate"] = float(y.mean())
    return model


def build(gathered, profile, stride):
    """The artifact: one model per held-out match (fitted on the others), `full`, and the provenance."""
    ids = sorted(gathered.data)

    def fitted(match_ids):
        X = np.concatenate([gathered.data[match_id][0] for match_id in match_ids])
        y = np.concatenate([gathered.data[match_id][1] for match_id in match_ids])
        return fit(X, y)

    return {"format": FORMAT, "kind": KIND, "target": TARGET, "window_frames": WINDOW_FRAMES,
            "feature_version": team_features.FEATURE_VERSION, "features": list(team_features.FEATURE_NAMES),
            "profile": profile, "seeds": gathered.seeds, "stride": stride, "provenance": provenance(gathered),
            "folds": {held_out: fitted([other for other in ids if other != held_out]) for held_out in ids},
            "full": fitted(ids)}


def train(profile, seeds, stride, matches=None, calibration=None, data_dir=None, log=None):
    """The artifact: gather the rows, then fit every fold and `full` (see the module docstring)."""
    gathered = gather(profile, seeds, stride, matches=matches, calibration=calibration, data_dir=data_dir, log=log)
    return build(gathered, profile, stride)


def provenance(gathered):
    """What the model was made from: scikit-learn's version, the Calibration (None on generic and clean), the DFL files
    and our sources."""
    import sklearn
    return {"scikit_learn": sklearn.__version__, "calibration_sha256": gathered.calibration_sha256,
            "dfl_sources": {match_id: gathered.dfl_sources[match_id] for match_id in sorted(gathered.dfl_sources)},
            "source_sha256": source_hashes()}


def source_hashes():
    """{file name: sha256} of the files the model depends on, read with LF endings on every platform."""
    hashes = {}
    for name, path in SOURCE_FILES:
        with open(path, "rb") as handle:
            hashes[name] = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    return hashes


def verify(path, fresh):
    """Compare the committed file at `path` with the retrained `fresh`; print what differs; the exit code (0 or 1)."""
    try:
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
    except (OSError, ValueError) as error:
        print("FAILED    cannot read the committed model {} ({}); train it with `python {}`".format(
            path, error, turnover_hazard.TRAINER))
        return 1
    problems, notes = compare(committed, fresh)
    for note in notes:
        print("note      {}".format(note))
    for problem in problems:
        print("FAILED    {}".format(problem))
    if problems:
        print("FAILED    {} differs from its retraining in {} place(s){}".format(
            path, len(problems), " or more" if len(problems) >= 20 else ""))
        return 1
    print("verified  {} is what the code trains{}".format(path, " (source hashes aside)" if notes else ""))
    return 0


# -- the dry run ------------------------------------------------------------------------------------------------------

def dry_run(profile, seeds, match_ids, stride=DEFAULT_STRIDE, matches=None, calibration=None, data_dir=None, log=None,
            row_window="declared", positive_window="declared", peak_limit_gb=PEAK_LIMIT_GB):
    """Aggregates of the rows of `match_ids` and nothing fitted: the numbers the pause before training reads.

    {profile, seeds, stride, per_match: {id: {rows, positive_share, x_bytes}}, rows, positive_share, nan_share,
    all_nan, constant, constant_values, peak_memory_bytes, projected_peak_gb, projection_basis, checks: {rows,
    positive, projection}, ok}. As train_team.dry_run, with a positive share beside the rows. `row_window` and
    `positive_window` are (low, high), inclusive, which the run must land in; "declared" means ROW_WINDOW and
    POSITIVE_WINDOW for the one run they are about (ROW_WINDOW_RUN) and none for any other, None means none. `ok` is
    false for an all-NaN or constant column, a count or share outside its window, or a projection over the limit.
    No row and no per-frame value is returned.
    """
    gathered = _gather(profile, seeds, stride, 1, matches, calibration, data_dir, match_ids, log)
    ids = sorted(gathered.data)
    X = np.concatenate([gathered.data[match_id][0] for match_id in ids])
    positives = int(sum(int(gathered.data[match_id][1].sum()) for match_id in ids))
    names = list(team_features.FEATURE_NAMES)
    finite = np.isfinite(X)
    nan_share = {name: float(1.0 - finite[:, i].mean()) if len(X) else 1.0 for i, name in enumerate(names)}
    all_nan = [name for i, name in enumerate(names) if not finite[:, i].any()]
    constant = [name for i, name in enumerate(names) if finite[:, i].any()
                and X[finite[:, i], i].min() == X[finite[:, i], i].max()]
    peak = train_team.peak_memory_bytes()
    training_gb = TRAINING_ROWS * 2 * len(names) * 8 * COPIES / 1e9
    row_bytes = sum(int(gathered.data[match_id][0].nbytes) for match_id in ids)
    projected = (row_bytes if peak is None else peak) / 1e9 + training_gb
    declared = (profile, seeds, stride, list(match_ids or [])) == ROW_WINDOW_RUN
    if row_window == "declared":
        row_window = ROW_WINDOW if declared else None
    if positive_window == "declared":
        positive_window = POSITIVE_WINDOW if declared else None
    share = positives / len(X) if len(X) else None
    rows_ok = None if row_window is None else bool(row_window[0] <= len(X) <= row_window[1])
    positive_ok = None if positive_window is None else bool(
        share is not None and positive_window[0] <= share <= positive_window[1])
    projection_ok = bool(projected <= peak_limit_gb)
    constant_values = {name: float(X[finite[:, names.index(name)], names.index(name)][0]) for name in constant}
    ok = not all_nan and not constant and rows_ok is not False and positive_ok is not False and projection_ok
    return {"profile": profile, "seeds": gathered.seeds, "stride": stride,
            "per_match": {match_id: {"rows": int(len(gathered.data[match_id][1])),
                                     "positive_share": float(gathered.data[match_id][1].mean())
                                     if len(gathered.data[match_id][1]) else None,
                                     "x_bytes": int(gathered.data[match_id][0].nbytes)} for match_id in ids},
            "rows": int(len(X)), "positive_share": share, "nan_share": nan_share, "all_nan": all_nan,
            "constant": constant, "constant_values": constant_values, "peak_memory_bytes": peak,
            "projected_peak_gb": projected, "projection_basis": "peak" if peak is not None else "row bytes",
            "checks": {"rows": {"window": None if row_window is None else list(row_window), "ok": rows_ok},
                       "positive": {"window": None if positive_window is None else list(positive_window),
                                    "ok": positive_ok},
                       "projection": {"limit_gb": peak_limit_gb, "ok": projection_ok}},
            "ok": bool(ok)}


def render_dry_run(report):
    lines = ["dry run   profile {}, {} seed(s), stride {}: rows only, nothing fitted".format(
        report["profile"], report["seeds"], report["stride"])]
    for match_id, cell in sorted(report["per_match"].items()):
        lines.append("dry run   {:<8} {:>9,} rows, {} near a change, X {:,} bytes".format(
            match_id, cell["rows"], "-" if cell["positive_share"] is None else "{:.1%}".format(cell["positive_share"]),
            cell["x_bytes"]))
    for name, share in report["nan_share"].items():
        lines.append("dry run   NaN {:>6.1%}  {}".format(share, name))
    lines.append("dry run   all-NaN columns: {}".format(", ".join(report["all_nan"]) or "none"))
    lines.append("dry run   constant columns: {}".format(
        ", ".join("{} (always {:g})".format(name, report["constant_values"][name]) for name in report["constant"])
        or "none"))
    peak, projected = report["peak_memory_bytes"], report["projected_peak_gb"]
    lines.append("dry run   peak memory {}; projected peak of the full run {:.2f} GB (from {})".format(
        "unknown" if peak is None else "{:.2f} GB".format(peak / 1e9), projected, report["projection_basis"]))
    rows, positive = report["checks"]["rows"], report["checks"]["positive"]
    projection = report["checks"]["projection"]
    if rows["ok"] is None:
        verdict = "not checked (not the declared run)"
    else:
        verdict = "{} the window [{:,}, {:,}]".format("PASS, inside" if rows["ok"] else "FAIL, outside",
                                                      *rows["window"])
    lines.append("dry run   rows {:,}: {}".format(report["rows"], verdict))
    share = report["positive_share"]
    if positive["ok"] is None:
        verdict = "not checked (not the declared run)"
    else:
        verdict = "{} the window [{:.0%}, {:.0%}]".format("PASS, inside" if positive["ok"] else "FAIL, outside",
                                                          *positive["window"])
    lines.append("dry run   positive share {}: {}".format("-" if share is None else "{:.2%}".format(share), verdict))
    lines.append("dry run   projected peak {}: {:.2f} GB against {:g} GB".format(
        "PASS" if projection["ok"] else "FAIL", projected, projection["limit_gb"]))
    lines.append("dry run   {}".format("OK" if report["ok"] else "FAILED: the pause's conditions are not met"))
    return lines


# -- the report -------------------------------------------------------------------------------------------------------

def auc(p, y):
    """The area under the ROC curve of scores `p` for labels `y` (0 or 1), by the rank formula: the sum of the
    positives' ranks, ties given their average rank, less n1 (n1 + 1) / 2, over n1 n0. None where a class is absent."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    positive = y == 1
    n1 = int(positive.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]          # 1-based, ties averaged
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def log_loss_sum(p, y):
    """The summed log-loss of probabilities `p` (clipped to LOG_FLOOR of 0 and 1) for labels `y`."""
    p = np.clip(np.asarray(p, dtype=np.float64), LOG_FLOOR, 1.0 - LOG_FLOOR)
    y = np.asarray(y)
    return float(-np.where(y == 1, np.log(p), np.log(1.0 - p)).sum())


def go(pooled_auc, loss, base_loss):
    """P35's gate: {auc, auc_min, auc_ok, log_loss, base_log_loss, loss_ratio, loss_ratio_max, loss_ok, go}.

    GO iff the AUC is at least AUC_MIN and the log-loss is at most LOSS_RATIO_MAX times the base rate's, each to
    GATE_TOLERANCE; a missing number (None) fails its half.
    """
    auc_ok = pooled_auc is not None and pooled_auc >= AUC_MIN - GATE_TOLERANCE
    loss_ok = loss is not None and base_loss is not None and loss <= LOSS_RATIO_MAX * base_loss + GATE_TOLERANCE
    ratio = None if loss is None or not base_loss else loss / base_loss
    return {"auc": pooled_auc, "auc_min": AUC_MIN, "auc_ok": bool(auc_ok), "log_loss": loss,
            "base_log_loss": base_loss, "loss_ratio": ratio, "loss_ratio_max": LOSS_RATIO_MAX,
            "loss_ok": bool(loss_ok), "go": bool(auc_ok and loss_ok)}


def localisation(offsets):
    """{changes, median, q1, q3, within5}: the offsets (frames, p's maximum less the change's frame) summarised; the
    share within +-LOCALISE_NEAR. All None but `changes` for none."""
    if len(offsets) == 0:
        return {"changes": 0, "median": None, "q1": None, "q3": None, "within5": None}
    offsets = np.asarray(offsets, dtype=np.float64)
    q1, median, q3 = (float(v) for v in np.percentile(offsets, [25, 50, 75]))
    return {"changes": int(len(offsets)), "median": median, "q1": q1, "q3": q3,
            "within5": float((np.abs(offsets) <= LOCALISE_NEAR).mean())}


def report(artifact, profile=run.DEFAULT_PROFILE, seeds=run.DEFAULT_SEEDS, stride=DEFAULT_STRIDE, matches=None,
           calibration=None, data_dir=None, min_prob=0.001):
    """The artifact's folds against their held-out matches: the gate, the transfer means and the localisation.

    {profile, seeds, stride, model_sha256, per_match: {id: {rows, positives, base_rate, auc, log_loss, base_log_loss,
    mean_p_seen, seen_frames}}, pooled: {rows, positives, auc, log_loss, base_log_loss}, go: go(...), transfer: {low,
    high}, localisation}. Each match is scored by the fold of its own id on its rows (every `stride`-th graded frame with
    the ball seen, truth H or A), the seeds pooled; the base-rate log-loss is of that fold's own training base rate. The
    pooled numbers pool every row of every match. `mean_p_seen` is the mean of p, clipped to [min_prob, 1 - min_prob] as
    the source clips it, over every graded frame with the ball seen whatever the truth (alive and dead), and `transfer`
    is [half the lowest, twice the highest] of them (amendment 10). BenchmarkError for an artifact that lacks the fold of
    a match (`full` saw it). Numbers only, no per-frame array.
    """
    seeds = _whole("seeds", seeds, 1)
    train_team._known_profile(profile)
    ids = run.resolve_match_ids(None, matches)
    folds = artifact.get("folds") if isinstance(artifact, dict) else None
    absent = [match_id for match_id in sorted(ids) if not isinstance(folds, dict) or match_id not in folds]
    if absent:
        raise BenchmarkError("train_turnover: the artifact has no fold for {}; the source would score that match with "
                             "`full`, which saw it, and the numbers would be in sample; retrain with `python {}`"
                             .format(", ".join(absent), turnover_hazard.TRAINER))
    calibration = train_team._calibration_for(profile, calibration, data_dir)
    seed_list = train_team._seed_list(profile, seeds)

    cells, scores, labels, offsets = {}, [], [], []
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        train_team._own_id(match, match_id)
        codes, _ = truths.truth_codes(match)
        truth_labels = change_labels(codes, match.graded)
        changes = direct_changes(codes, match.graded)
        model = folds[match_id]
        p_rows, y_rows, seen, p_seen = [], [], 0, 0.0
        for seed in seed_list:
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            logit = turnover_hazard.change_logit(inputs, model)
            frames = row_frames(inputs, codes, stride)
            p_rows.append(trees.sigmoid(logit[frames]))
            y_rows.append(truth_labels[frames].astype(np.int64))
            known = ~np.isnan(logit)
            seen += int(known.sum())
            p_seen += float(np.clip(trees.sigmoid(logit[known]), min_prob, 1.0 - min_prob).sum())
            for frame, start, stop in changes:
                if not known[frame]:
                    continue
                low, high = max(start, frame - LOCALISE_FRAMES), min(stop, frame + LOCALISE_FRAMES + 1)
                window = logit[low:high]
                offsets.append(low + int(np.nanargmax(window)) - frame)
            del inputs, logit, known
        p, y = np.concatenate(p_rows), np.concatenate(y_rows)
        base = float(model["base_rate"])
        cells[match_id] = {"rows": int(len(y)), "positives": int(y.sum()), "base_rate": base, "auc": auc(p, y),
                           "log_loss": log_loss_sum(p, y) / len(y) if len(y) else None,
                           "base_log_loss": log_loss_sum(np.full(len(y), base), y) / len(y) if len(y) else None,
                           "mean_p_seen": p_seen / seen if seen else None, "seen_frames": seen,
                           "_sums": (log_loss_sum(p, y), log_loss_sum(np.full(len(y), base), y))}
        scores.append(p)
        labels.append(y)
        del match, codes
    p, y = np.concatenate(scores), np.concatenate(labels)
    loss = sum(cell["_sums"][0] for cell in cells.values()) / len(y) if len(y) else None
    base_loss = sum(cell["_sums"][1] for cell in cells.values()) / len(y) if len(y) else None
    for cell in cells.values():
        del cell["_sums"]
    pooled = {"rows": int(len(y)), "positives": int(y.sum()), "auc": auc(p, y), "log_loss": loss,
              "base_log_loss": base_loss}
    means = [cell["mean_p_seen"] for cell in cells.values() if cell["mean_p_seen"] is not None]
    return {"profile": profile, "seeds": len(seed_list), "stride": stride,
            "model_sha256": hashlib.sha256(dumps(artifact).encode("utf-8")).hexdigest(), "per_match": cells,
            "pooled": pooled, "go": go(pooled["auc"], loss, base_loss),
            "transfer": {"low": 0.5 * min(means) if means else None, "high": 2.0 * max(means) if means else None},
            "localisation": localisation(offsets)}


def render_report(result):
    """The report's lines: a table of the matches and the pool, the transfer means, the localisation, the verdict."""
    def number(value, digits=4):
        return "-" if value is None else "{:.{}f}".format(value, digits)

    lines = ["report    profile {}, {} seed(s): out of fold, seeds pooled per match".format(
                 result["profile"], result["seeds"]),
             "report    {:<8} {:>8} {:>8} | {:>7} {:>9} {:>9} | {:>10} {:>12}".format(
                 "match", "rows", "positive", "AUC", "log-loss", "base ll", "mean p", "seen frames")]
    for name, cell in list(sorted(result["per_match"].items())) + [("pooled", result["pooled"])]:
        lines.append("report    {:<8} {:>8,} {:>8,} | {:>7} {:>9} {:>9} | {:>10} {:>12}".format(
            name, cell["rows"], cell["positives"], number(cell["auc"]), number(cell["log_loss"]),
            number(cell["base_log_loss"]), number(cell.get("mean_p_seen"), 5),
            "-" if "seen_frames" not in cell else "{:,}".format(cell["seen_frames"])))
    transfer = result["transfer"]
    lines.append("report    transfer (gates nothing here; Tian runs it at adoption): the full model's mean p over "
                 "BAR-ATH's graded frames with the ball seen must lie in [{}, {}] = [half the lowest, twice the "
                 "highest] of the means above".format(number(transfer["low"], 5), number(transfer["high"], 5)))
    place = result["localisation"]
    lines.append("report    localisation (gates nothing): {:,} true direct changes with the ball seen; offset of p's "
                 "maximum within +-{} frames: median {}, quartiles {} to {}, within +-{}: {}".format(
                     place["changes"], LOCALISE_FRAMES, number(place["median"], 1), number(place["q1"], 1),
                     number(place["q3"], 1), LOCALISE_NEAR,
                     "-" if place["within5"] is None else "{:.1%}".format(place["within5"])))
    verdict = result["go"]
    lines.append("report    the gate: pooled AUC {} against at least {:.2f} ({}); pooled log-loss {} against at most "
                 "{:.2f} x the base rate's {} = {} ({})".format(
                     number(verdict["auc"]), verdict["auc_min"], "held" if verdict["auc_ok"] else "not held",
                     number(verdict["log_loss"]), verdict["loss_ratio_max"], number(verdict["base_log_loss"]),
                     "-" if verdict["base_log_loss"] is None else number(verdict["loss_ratio_max"]
                                                                         * verdict["base_log_loss"]),
                     "held" if verdict["loss_ok"] else "not held"))
    lines.append("report    {}".format("GO: the benchmark of P35's configs follows once the decoder is merged"
                                       if verdict["go"] else "NO GO: P35 stops here, the negative result is recorded"))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train P35's turnover hazard (a direct change of team within 12 frames, on team features version "
                    "2) on the benchmark matches (folds and full), write it as JSON, verify it against a retraining, "
                    "count its rows (--dry-run) or run its gate (--report).")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="every this many graded frames is a row (default {})".format(DEFAULT_STRIDE))
    parser.add_argument("--out", help="where to write the artifact, or to read it from with --report (default "
                                      "pipeline/models/turnover_hgb.json)")
    parser.add_argument("--verify", action="store_true",
                        help="retrain and compare with the committed file instead of writing it: exit 1 if "
                             "anything but the source hashes differs")
    parser.add_argument("--dry-run", action="store_true",
                        help="gather the rows and fit nothing: rows per match, the positive share, NaN shares, dead "
                             "columns, peak memory; exit 1 on an all-NaN or a constant column, rows or a positive "
                             "share outside the declared windows or a projected peak over {:g} GB".format(PEAK_LIMIT_GB))
    parser.add_argument("--report", action="store_true",
                        help="train nothing: read the artifact and score each fold on its held-out match (AUC, "
                             "log-loss, base-rate log-loss, GO / NO GO); writes <data>/{}/report.json".format(
                                 "/".join(REPORT_FOLDER)))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to count (--dry-run only)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    given = {text.split("=")[0] for text in (sys.argv[1:] if argv is None else argv)}
    if args.seeds < 1 or args.stride < 1:
        parser.error("--seeds and --stride must be at least 1")
    if args.dry_run and (args.verify or args.report):
        parser.error("--dry-run takes no --verify or --report: it gathers rows and fits nothing")
    if args.matches and not args.dry_run:
        parser.error("--matches is for --dry-run only: training and the report use all seven matches")
    if args.report and (args.verify or "--stride" in given):
        parser.error("--report takes no --verify or --stride: it reads the artifact and tests it, and trains nothing")
    path = args.out or DEFAULT_MODEL_PATH
    try:
        if args.dry_run:
            result = dry_run(args.profile, args.seeds, args.matches, args.stride, data_dir=args.data_dir, log=print)
            for line in render_dry_run(result):
                print(line)
            return 0 if result["ok"] else 1
        if args.report:
            try:
                with open(path, encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, ValueError) as error:
                raise BenchmarkError("cannot read the model {} ({}); train it with `python {}`".format(
                    path, error, turnover_hazard.TRAINER)) from error
            result = report(artifact, args.profile, args.seeds, data_dir=args.data_dir)
            for line in render_report(result):
                print(line)
            where = os.path.join(args.data_dir or games.DATA_DIR, *REPORT_FOLDER, "report.json")
            os.makedirs(os.path.dirname(where), exist_ok=True)
            with open(where, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(dumps(result))
            print("wrote     {}".format(where))
            return 0
        artifact = train(args.profile, args.seeds, args.stride, data_dir=args.data_dir, log=print)
        if args.verify:
            return verify(path, artifact)
        write(path, artifact)
        print("wrote     {} ({:,} bytes)".format(path, os.path.getsize(path)))
        return 0
    except (BenchmarkError, OSError) as error:
        # The expected failures get a sentence, as in run.py: a profile or match that does not exist, data not
        # downloaded yet. Anything else is a bug and keeps its traceback.
        print("FAILED    {}".format(error))
        return 1


# -- small things -----------------------------------------------------------------------------------------------------

def _whole(name, value, minimum):
    """`value` as an int, or ValueError unless it is a whole number of at least `minimum` (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError("train_turnover: {} must be a whole number of at least {}, got {!r}"
                         .format(name, minimum, value))
    return int(value)


if __name__ == "__main__":
    sys.exit(main())
