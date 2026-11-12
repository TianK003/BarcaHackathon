"""P19's trainer, restored for P32: fit the team model on the benchmark's matches, write it as JSON, and run its
feasibility test.

    python pipeline/benchmark/train_team.py [--profile generic] [--seeds 3] [--stride 5] [--out PATH] [--verify]
                                            [--data-dir D]
    python pipeline/benchmark/train_team.py --dry-run [--matches ID ...] [--seeds 1] [--data-dir D]
    python pipeline/benchmark/train_team.py --feasibility [--config pipeline/benchmark/configs/p30/p30_w030.toml]
                                            [--out PATH]

    artifact = train(profile="generic", seeds=3, stride=5)                  # the dict that is written, JSON-ready
    print(dry_run("generic", 1, ["J03WMX"]))                                 # rows and NaN shares, nothing fitted
    cells = feasibility(artifact, DEFAULT_CONFIG)["per_match"]              # out of fold, against p30_w030's labels

This is P07's trainer (train_shape.py) with `team_features`' 72 columns (version 2, P32), and everything that is not
about the columns is P05's and P07's, imported and not rewritten: `HGB_SETTINGS` (never tuned), `export_hgb`,
`compare`, `dumps` and `write`, `served_inputs`, and the classifier itself (`train_shape.hgb_estimator`). Read
train_shape.py's docstring for the argument; what differs is below. P19 wrote it on 31 Oct (R021, on the `tracking`
profile, 56 columns) and it was removed when R021's feasibility gate failed; it is in git at commit 15e7e14. P32
restores it with three changes: the default profile is the benchmark's, `generic` (P27), which reads nothing of our
match and takes no Calibration (none is loaded, and one that is given is refused, as train_shape does); the columns are
team_features' version 2; and the feasibility's go rule is on log-loss (below).

The rows. The model answers which team has the ball *where the ball is seen*: a row is every `stride`-th frame of a
graded block from its start where the truth is H or A and the served ball is visible (`kinematics.ball_track(...)
.visible`), and 1 where it is H. The gaps are P07's. The label is read from `truth.truth_codes(match)` beside the
features and never through the inputs (test_team_trainer_never_passes_truth). Every row is shown twice, as it is and as
`team_features.swap(row)` with the label flipped, so `prior_home` is exactly 0.5 and the source's output is made
antisymmetric by (r(x) - r(swap(x))) / 2 (pipeline/evidence/team_model.py).

The artifact is train_shape's layout with `team_features`' names and `feature_version` 2, at
pipeline/models/team_model_hgb.json; a fold per match (fitted on the other six) plus `full`. `provenance` hashes the
seven sources the model depends on and carries no git commit; its Calibration digest is None on `generic`; `--verify`
retrains and exits 1 on anything but a differing source hash, which is a note.

--dry-run gathers rows for the named matches (one match, one seed is enough) and fits nothing. It prints aggregates:
rows per match, the home share, the bytes of X, the NaN share of every feature, the names of any all-NaN or constant
column (a feature that cannot carry information, which is a bug in the features and not something to train through: the
exit code is 1), the peak memory of the process, and the projected peak of the real run, this process's peak plus the
augmented rows of the whole training set (194,814 rows x 2 x 72 columns x 8 bytes, times 4 for the copies that the
binning and the fit make). The one declared run (J03WMX, `generic`, one seed, stride 5) must land in ROW_WINDOW, the
declared estimate 10,362 within 20% (P32, pro rata from P30's report), or the exit code is 1; so must a projected peak
over 8 GB. The controller reads it at the pause before training.

--feasibility is the gate before any benchmark run (docs/superpowers/specs/2026-11-13-p32-p34-design.md, P32, step 0).
On each match and seed it serves the degraded inputs, decodes them with the config given (default p30_w030, the night's
candidate; match-only sources dropped, as the benchmark runs it), and takes the model's `home` out of fold, on R021's
frames: graded, truth H or A, served ball visible and decoded H or A by the config (its dead-ball errors are not
credited to the model). Beside it is the rule with its hold (controlled_touch with the parameters of the config's own
entry, its confidence and hold included), a silent frame counted as wrong and scored at 0.5 in the log-loss.

The go (P32's, replacing R021's): the model's pooled log-loss on those frames, all seven matches and their seeds, is at
most LOSS_MAX = 0.2700 (to 1e-12), and the model's log-loss is below the rule-with-hold's on every match. R021's
per-frame statistic, the model's accuracy minus the config's decoded label's (and R021's integer test of a 0.020 gain in
6 of 7 matches and the pool), is still computed and printed, and gates nothing: it compares one source with the decoded
output of all of them, which is not the question P32 asks (the bar on spellF1 is). R021's frame classes are reported
too: in an episode, in the hold, silent, and whether either team shows more than 11 present slots (T32; `generic` has
no wrong-team slot). The output is sums and ratios: no per-frame array.
"""

import argparse
import hashlib
import json
import math
import numbers
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path. Swap in the repository root instead, so that
    # `import pipeline` resolves from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
from pipeline.benchmark.train_dead_ball import (  # noqa: E402
    HGB_SETTINGS, LOG_FLOOR, Gathered, compare, dumps, export_hgb, served_inputs, write)
from pipeline.benchmark.train_shape import hgb_estimator  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, kinematics, team_features, team_model, trees  # noqa: E402

import games  # noqa: E402  (importing pipeline above put the repository root on sys.path)
import possession_metrics as pm  # noqa: E402  (importing pipeline above put evaluation/ on sys.path)

# Params
FORMAT = trees.FORMAT
KIND = "hgb"
HOME, AWAY = pm.H, pm.A                       # the codes of the truth that are alive; a dead ball has no team to name
DEFAULT_STRIDE = 5
PRIOR_HOME = team_model.PRIOR_HOME            # the share of homes among the rows and their mirrors
DEFAULT_MODEL_PATH = os.path.join(ROOT, "pipeline", "models", "team_model_hgb.json")
# The night's candidate (P30, R033), against which P32 is measured (the design's "What every lane shares")
DEFAULT_CONFIG = os.path.join(ROOT, "pipeline", "benchmark", "configs", "p30", "p30_w030.toml")
SOURCE_FILES = (("train_team.py", os.path.join(HERE, "train_team.py")),
                ("team_features.py", os.path.join(ROOT, "pipeline", "evidence", "team_features.py")),
                ("shape_features.py", os.path.join(ROOT, "pipeline", "evidence", "shape_features.py")),
                ("play_features.py", os.path.join(ROOT, "pipeline", "evidence", "play_features.py")),
                ("kinematics.py", os.path.join(ROOT, "pipeline", "evidence", "kinematics.py")),
                ("controlled_touch.py", os.path.join(ROOT, "pipeline", "evidence", "controlled_touch.py")),
                ("trees.py", os.path.join(ROOT, "pipeline", "evidence", "trees.py")))
HOME_LEVEL = 0.5                              # a home probability at or above this names home
LN2 = math.log(2.0)                           # the log-loss of saying 0.5
TRAINING_ROWS = 194_814                       # the full run's rows, P30's report: (1,659,315 - 685,247) / 5
COPIES = 4                                    # the binned copy, the augmented copy and the fit's own, with room
LOSS_MAX = 0.2700                             # the go: the pooled held-out log-loss is at most this (P32)
LOSS_TOLERANCE = 1e-12                        # ... to this: the pool is a ratio of float sums
R021_MATCHES = 6                              # R021's per-frame test, which gates nothing now: matches of 7 that gain
R021_NUMERATOR = 50                           # 50 * (p19_right - main_right) >= frames: a gain of 0.020 in accuracy
MAX_SLOTS = 11                                # a frame class (T32): a team shows more present slots than players
FEASIBILITY_FOLDER = ("runs", "p32")
ROW_WINDOW = (8_290, 12_434)                  # the dry run's rows, J03WMX seed 0: 10,362 estimated, within 20%
ROW_WINDOW_RUN = ("generic", 1, DEFAULT_STRIDE, ["J03WMX"])     # the one dry run that the window is about
PEAK_LIMIT_GB = 8.0                           # the pause's projected peak (advisor-01.md section 3.1, kept by P32)


# -- the rows ---------------------------------------------------------------------------------------------------------

def rows(inputs, codes, stride):
    """(X, y): the features of every `stride`-th graded frame from each block's start where the truth is H or A and the
    served ball is visible, and 1 where it is H.

    `codes` is the truth's int8 pm codes per frame of the match, kept apart from the inputs, which have none. A dead
    ball (D), a frame outside the blocks (X) and a frame without a ball are no rows.
    """
    stride = _whole("stride", stride, 1)
    codes = np.asarray(codes)
    if codes.shape != (inputs.n_frames,):
        raise ValueError("train_team: codes must hold one code per frame of the match ({}), got shape {}"
                         .format(inputs.n_frames, codes.shape))
    frames = np.concatenate([np.arange(start, stop, stride) for _, start, stop in inputs.graded])
    visible = kinematics.ball_track(inputs).visible
    frames = frames[((codes[frames] == HOME) | (codes[frames] == AWAY)) & visible[frames]]
    _, X = team_features.features(inputs)
    return X[frames], (codes[frames] == HOME).astype(np.int64)


def gather(profile, seeds, stride, matches=None, calibration=None, data_dir=None, match_ids=None, log=None):
    """Gathered: the rows of every match under every seed of `profile`, served as the pipeline serves them.

    As train_shape.gather: `matches` is {id: DflMatch} to use instead of loading the seven from `data_dir`;
    `calibration` is the Calibration of a profile that reads one (gaps, tracking, context), loaded when none is given;
    `generic` and `clean` take none. BenchmarkError for an unknown profile, fewer than two matches (a fold needs another
    match to be fitted on), a match filed under another id than its own, and a calibration given to a profile that
    takes none.
    """
    return _gather(profile, seeds, stride, 2, matches, calibration, data_dir, match_ids, log)


def _gather(profile, seeds, stride, minimum, matches, calibration, data_dir, match_ids, log):
    seeds = _whole("seeds", seeds, 1)
    _whole("stride", stride, 1)
    _known_profile(profile)
    ids = run.resolve_match_ids(match_ids, matches)
    if len(ids) < minimum:
        raise BenchmarkError("train_team: a fold is fitted on the other matches, so it needs at least two matches, "
                             "got {}".format(len(ids)))
    calibration = _calibration_for(profile, calibration, data_dir)
    seed_list = _seed_list(profile, seeds)
    log = log or (lambda text: None)

    data, sources = {}, {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        _own_id(match, match_id)
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
        log("rows      {}   {:,} rows, {:.1%} home".format(match_id, len(data[match_id][1]),
                                                         float(data[match_id][1].mean()) if len(data[match_id][1])
                                                         else float("nan")))
        del match, codes
    digest = calibration_module.digest(calibration) if run.needs_calibration(profile) else None
    return Gathered(data, digest, sources, len(seed_list))


# -- the model --------------------------------------------------------------------------------------------------------

def augment(X, y):
    """(X', y'): the rows, then their mirror images (`team_features.swap`) with the label flipped.

    The classes balance exactly, whatever the rows' own balance. ValueError for X that is not (n, 72) or a y that is
    not one 0 or 1 per row.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    names = len(team_features.FEATURE_NAMES)
    if X.ndim != 2 or X.shape[1] != names or y.shape != (len(X),) or not np.isin(y, (0, 1)).all():
        raise ValueError("train_team: X must be (n, {}) and y one 0 or 1 per row, got X {} and y {}"
                         .format(names, X.shape, y.shape))
    return np.concatenate([X, team_features.swap(X)]), np.concatenate([y, 1 - y]).astype(np.int64)


def fit(X, y):
    """One model, as a JSON-ready dict {baseline, trees, prior_home}: fitted on the rows X, y and their mirrors.

    ValueError for the shapes `augment` refuses and for rows of one class only.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    augmented = augment(X, y)                       # checks the shapes
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("train_team: the rows hold one class only ({} home among {}); a model needs both home and "
                         "away frames".format(int(y.sum()), len(y)))
    model = export_hgb(hgb_estimator(*augmented))
    model["prior_home"] = PRIOR_HOME
    return model


def build(gathered, profile, stride):
    """The artifact: one model per held-out match (fitted on the others), `full`, and the provenance."""
    ids = sorted(gathered.data)

    def fitted(match_ids):
        X = np.concatenate([gathered.data[match_id][0] for match_id in match_ids])
        y = np.concatenate([gathered.data[match_id][1] for match_id in match_ids])
        return fit(X, y)

    return {"format": FORMAT, "kind": KIND, "feature_version": team_features.FEATURE_VERSION,
            "features": list(team_features.FEATURE_NAMES), "profile": profile, "seeds": gathered.seeds,
            "stride": stride, "provenance": provenance(gathered),
            "folds": {held_out: fitted([other for other in ids if other != held_out]) for held_out in ids},
            "full": fitted(ids)}


def train(profile, seeds, stride, matches=None, calibration=None, data_dir=None, log=None):
    """The artifact: gather the rows, then fit every fold and `full` (see the module docstring)."""
    gathered = gather(profile, seeds, stride, matches=matches, calibration=calibration, data_dir=data_dir, log=log)
    return build(gathered, profile, stride)


def provenance(gathered):
    """What the model was made from: scikit-learn's version, the Calibration (None on generic and clean), the DFL files
    and our seven sources."""
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
            path, error, team_model.TRAINER))
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

def peak_memory_bytes():
    """The peak memory of this process in bytes, or None where the platform does not say.

    Windows: `PeakWorkingSetSize` of GetProcessMemoryInfo, through ctypes. Elsewhere `getrusage`'s ru_maxrss, which is
    bytes on macOS and kilobytes on Linux. Standard library only.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            # Library objects of our own, with the handle typed as a handle: ctypes passes an untyped int as a 32-bit
            # C int, which truncates the 64-bit pseudo-handle of the current process, and the call then fails with
            # "invalid handle" (R021's dry run printed "peak memory unknown" for that reason). Typing the shared
            # ctypes.windll functions instead would change them for every other caller in the process.
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetCurrentProcess.argtypes = []
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, AttributeError, OSError):
        return None


def dry_run(profile, seeds, match_ids, stride=DEFAULT_STRIDE, matches=None, calibration=None, data_dir=None, log=None,
            row_window="declared", peak_limit_gb=PEAK_LIMIT_GB):
    """Aggregates of the rows of `match_ids` and nothing fitted: the numbers the pause before training reads.

    {profile, seeds, stride, per_match: {id: {rows, home_share, x_bytes}}, rows, nan_share: {feature: share},
    all_nan: [names], constant: [names], constant_values: {name: value}, peak_memory_bytes, projected_peak_gb,
    projection_basis, checks: {rows: {window, ok}, projection: {limit_gb, ok}}, ok}. A column is constant when its
    finite values are all one number. One match is enough: no fold is fitted. No row and no per-frame value is
    returned.

    The two gates of the pause are held here and not only printed. `row_window` is (low, high) rows, inclusive, which
    the run must land in; "declared" means ROW_WINDOW for the one run it is about (ROW_WINDOW_RUN: J03WMX, `generic`,
    1 seed, stride 5) and no window for any other, None means no window. The projected peak is the training set's bytes
    (the augmented rows of the whole run, 194,814 x 2 x 72 x 8, times the copies the binning and the fit make) plus a
    baseline: this process's peak where the platform says it, else the bytes of the rows gathered; it must be at most
    `peak_limit_gb`. `ok` is false for an all-NaN or a constant column, a row count outside the window or a projection
    over the limit.
    """
    gathered = _gather(profile, seeds, stride, 1, matches, calibration, data_dir, match_ids, log)
    ids = sorted(gathered.data)
    X = np.concatenate([gathered.data[match_id][0] for match_id in ids])
    names = list(team_features.FEATURE_NAMES)
    finite = np.isfinite(X)
    nan_share = {name: float(1.0 - finite[:, i].mean()) if len(X) else 1.0 for i, name in enumerate(names)}
    all_nan = [name for i, name in enumerate(names) if not finite[:, i].any()]
    constant = [name for i, name in enumerate(names) if finite[:, i].any()
                and X[finite[:, i], i].min() == X[finite[:, i], i].max()]
    peak = peak_memory_bytes()
    training_gb = TRAINING_ROWS * 2 * len(names) * 8 * COPIES / 1e9
    row_bytes = sum(int(gathered.data[match_id][0].nbytes) for match_id in ids)
    projected = (row_bytes if peak is None else peak) / 1e9 + training_gb
    if row_window == "declared":
        row_window = ROW_WINDOW if (profile, seeds, stride, list(match_ids or [])) == (
            ROW_WINDOW_RUN[0], ROW_WINDOW_RUN[1], ROW_WINDOW_RUN[2], ROW_WINDOW_RUN[3]) else None
    rows_ok = None if row_window is None else bool(row_window[0] <= len(X) <= row_window[1])
    projection_ok = bool(projected <= peak_limit_gb)
    constant_values = {name: float(X[finite[:, names.index(name)], names.index(name)][0]) for name in constant}
    ok = not all_nan and not constant and rows_ok is not False and projection_ok
    return {"profile": profile, "seeds": gathered.seeds, "stride": stride,
            "per_match": {match_id: {"rows": int(len(gathered.data[match_id][1])),
                                     "home_share": float(gathered.data[match_id][1].mean())
                                     if len(gathered.data[match_id][1]) else None,
                                     "x_bytes": int(gathered.data[match_id][0].nbytes)} for match_id in ids},
            "rows": int(len(X)), "nan_share": nan_share, "all_nan": all_nan, "constant": constant,
            "constant_values": constant_values, "peak_memory_bytes": peak, "projected_peak_gb": projected,
            "projection_basis": "peak" if peak is not None else "row bytes",
            "checks": {"rows": {"window": None if row_window is None else list(row_window), "ok": rows_ok},
                       "projection": {"limit_gb": peak_limit_gb, "ok": projection_ok}},
            "ok": bool(ok)}


def render_dry_run(report):
    lines = ["dry run   profile {}, {} seed(s), stride {}: rows only, nothing fitted".format(
        report["profile"], report["seeds"], report["stride"])]
    for match_id, cell in sorted(report["per_match"].items()):
        lines.append("dry run   {:<8} {:>9,} rows, {} home, X {:,} bytes".format(
            match_id, cell["rows"], "-" if cell["home_share"] is None else "{:.1%}".format(cell["home_share"]),
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
    rows, projection = report["checks"]["rows"], report["checks"]["projection"]
    if rows["ok"] is None:
        verdict = "not checked (not the declared run)"
    else:
        verdict = "{} the window [{:,}, {:,}]".format("PASS, inside" if rows["ok"] else "FAIL, outside",
                                                      *rows["window"])
    lines.append("dry run   rows {:,}: {}".format(report["rows"], verdict))
    lines.append("dry run   projected peak {}: {:.2f} GB against {:g} GB".format(
        "PASS" if projection["ok"] else "FAIL", projected, projection["limit_gb"]))
    lines.append("dry run   {}".format("OK" if report["ok"] else "FAILED: the pause's conditions are not met"))
    return lines


# -- the feasibility --------------------------------------------------------------------------------------------------

def feasibility(artifact, config_path, profile=run.DEFAULT_PROFILE, seeds=run.DEFAULT_SEEDS, matches=None,
                calibration=None, data_dir=None):
    """The model against the config's decoded labels and the rule with its hold, out of fold, on R021's frames.

    {profile, seeds, config, config_sha256, model_sha256, per_match: {id: cell}, pooled: cell, go} where a cell is
    {frames, p19: {right, accuracy, log_loss}, main: {right, accuracy}, rule: {right, accuracy, silent, silent_share,
    log_loss}, coarse: the same on the coarse truth, classes: {episode, hold, silent, over11, not_over11: {frames, p19,
    main}}}, the seeds of a match pooled by sums, and `go` is go(per_match). (`p19` is the model and `main` the config's
    decoded label, R021's names.) The frames are graded, truth H or A, served ball visible, and decoded H or A by the
    config (match-only sources dropped); the model says H at p >= 0.5; the rule is controlled_touch with the parameters
    of the config's own entry, a silent frame wrong and scored at 0.5. `coarse` is the same on the coarse truth's own
    frames. The classes split the raw frames by the rule's value (an episode, the hold, silent) and by whether either
    team shows more than 11 present slots (over11_frames). BenchmarkError for an artifact that lacks the fold of a
    match (the source would score it with `full`, which saw it), for a config without exactly one controlled_touch
    entry, and for a calibration given to a profile that takes none. Numbers only, no per-frame array.
    """
    seeds = _whole("seeds", seeds, 1)
    _known_profile(profile)
    ids = run.resolve_match_ids(None, matches)
    folds = artifact.get("folds") if isinstance(artifact, dict) else None
    absent = [match_id for match_id in sorted(ids) if not isinstance(folds, dict) or match_id not in folds]
    if absent:
        raise BenchmarkError("train_team: the artifact has no fold for {}; the source would score that match with "
                             "`full`, which saw it, and the numbers would be in sample; retrain with `python {}`"
                             .format(", ".join(absent), team_model.TRAINER))
    config, _ = run.drop_match_only(load_config(config_path))
    touch = [component for component in config.evidence if component.kind == "controlled_touch"]
    if len(touch) != 1:
        raise BenchmarkError("train_team: {} must have exactly one controlled_touch entry (the rule the model is "
                             "compared with), it has {}".format(config.stem, len(touch)))
    calibration = _calibration_for(profile, calibration, data_dir)
    seed_list = _seed_list(profile, seeds)

    tallies = {}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, os.path.basename(DEFAULT_MODEL_PATH))
        write(path, artifact)
        for match_id in sorted(ids):
            match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
            _own_id(match, match_id)
            codes, _ = truths.truth_codes(match)
            coarse = truths.coarse_codes(codes, match.graded, match.fps)
            tally = _empty_tally()
            for seed in seed_list:
                inputs = served_inputs(match, profile, seed, calibration, data_dir)
                _add(tally, _seed_tally(config, touch[0].params, inputs, codes, coarse, path))
                del inputs
            tallies[match_id] = tally
            del match, codes, coarse
    pooled = _empty_tally()
    for tally in tallies.values():
        _add(pooled, tally)
    cells = {match_id: _cell(tallies[match_id]) for match_id in sorted(tallies)}
    with open(config.path, "rb") as handle:
        config_sha256 = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    return {"profile": profile, "seeds": len(seed_list), "config": config.stem, "config_sha256": config_sha256,
            "model_sha256": hashlib.sha256(dumps(artifact).encode("utf-8")).hexdigest(), "per_match": cells,
            "pooled": _cell(pooled), "go": go(cells)}


def go(cells):
    """P32's gate, from the cells {match: {frames, p19: {right, log_loss}, main: {right}, rule: {log_loss}}}.

    {pooled_log_loss, loss_max, pooled, lower: {match: bool}, matches_lower, go, per_frame: {wins, pooled, gates}}.
    `pooled_log_loss` is the model's log-loss over the frames of every match, sum(frames x log_loss) / sum(frames)
    (None with no frame); `pooled` is whether it is at most LOSS_MAX, to LOSS_TOLERANCE. `lower[match]` is whether the
    model's log-loss on the match is below the rule-with-hold's, strictly; a match with no frame is not lower. Go iff
    the pool holds and every match is lower. `per_frame` is R021's integer test (`_gain`: 0.020 of accuracy over the
    decoded label in R021_MATCHES matches and the pool), which `gates` nothing: it is reported, never read here.
    """
    frames = sum(cell["frames"] for cell in cells.values())
    loss = sum(cell["frames"] * cell["p19"]["log_loss"] for cell in cells.values() if cell["frames"] > 0)
    pooled_log_loss = loss / frames if frames else None
    pooled = pooled_log_loss is not None and pooled_log_loss <= LOSS_MAX + LOSS_TOLERANCE
    lower = {match_id: bool(cell["frames"] > 0 and cell["p19"]["log_loss"] < cell["rule"]["log_loss"])
             for match_id, cell in sorted(cells.items())}
    return {"pooled_log_loss": pooled_log_loss, "loss_max": LOSS_MAX, "pooled": bool(pooled), "lower": lower,
            "matches_lower": sum(lower.values()), "go": bool(pooled and lower and all(lower.values())),
            "per_frame": dict(per_frame(cells), gates=False)}


def per_frame(cells):
    """{wins, pooled}: R021's integer test on the cells {match: {frames, p19: {right}, main: {right}}}, which gates
    nothing here.

    A match wins iff 50 * (p19_right - main_right) >= frames: the model beats the config's decoded label by 0.020 of
    the frames or more, no rounding anywhere; `pooled` is the same test of the sums over every match. R021 needed 6 wins
    and the pool (R021_MATCHES).
    """
    wins = sum(1 for cell in cells.values() if _gain(cell))
    total = {"frames": sum(cell["frames"] for cell in cells.values()),
             "p19": {"right": sum(cell["p19"]["right"] for cell in cells.values())},
             "main": {"right": sum(cell["main"]["right"] for cell in cells.values())}}
    return {"wins": wins, "pooled": _gain(total)}


def _gain(cell):
    """R021's integer test of one cell; a cell with no frame is no win."""
    return cell["frames"] > 0 and R021_NUMERATOR * (cell["p19"]["right"] - cell["main"]["right"]) >= cell["frames"]


def over11_frames(inputs):
    """(n,) bool: the frames on which either team shows more than MAX_SLOTS present slots (T32), a frame class of the
    feasibility's report. Version 2 of the features has no such column (the `generic` profile has no wrong-team
    slot), so it is counted here, from the slots' presence and the metadata's teams."""
    present = kinematics.player_track(inputs).present
    teams = np.asarray(inputs.atd.slot_team)
    return (present[:, teams == 0].sum(axis=1) > MAX_SLOTS) | (present[:, teams == 1].sum(axis=1) > MAX_SLOTS)


# A tally is sums of integers and floats: frames, rights and log-loss sums, so that seeds and matches pool exactly.
CLASSES = ("episode", "hold", "silent", "over11", "not_over11")


def _empty_tally():
    def scored():
        return {"frames": 0, "p19": [0, 0.0], "main": 0, "rule": [0, 0.0, 0]}   # rights and log-loss; rule silent last
    return {"raw": scored(), "coarse": scored(), "classes": {name: {"frames": 0, "p19": 0, "main": 0}
                                                             for name in CLASSES}}


def _add(total, part):
    for name in ("raw", "coarse"):
        a, b = total[name], part[name]
        a["frames"] += b["frames"]
        a["main"] += b["main"]
        a["p19"] = [a["p19"][0] + b["p19"][0], a["p19"][1] + b["p19"][1]]
        a["rule"] = [x + y for x, y in zip(a["rule"], b["rule"])]
    for name in CLASSES:
        for key in ("frames", "p19", "main"):
            total["classes"][name][key] += part["classes"][name][key]


def _loss(p, is_home):
    p = np.clip(p, LOG_FLOOR, 1.0 - LOG_FLOOR)
    return float(-np.where(is_home, np.log(p), np.log(1.0 - p)).sum())


def _score(frames, is_home, home, decoded_home, rule):
    """The sums of one truth on the frames `frames` (a bool mask): the model, the decoded label and the rule."""
    p, r = home[frames], rule[frames]
    truly_home = is_home[frames]
    silent = np.isnan(r)
    rule_p = np.where(silent, 0.5, r)
    return {"frames": int(frames.sum()),
            "p19": [int(((p >= HOME_LEVEL) == truly_home).sum()), _loss(p, truly_home)],
            "main": int((decoded_home[frames] == truly_home).sum()),
            "rule": [int((~silent & ((r >= HOME_LEVEL) == truly_home)).sum()), _loss(rule_p, truly_home),
                     int(silent.sum())]}


def _seed_tally(config, touch_params, inputs, codes, coarse, model_path):
    """The sums of one match under one seed, scored by the model at `model_path`, which is the fold of the match."""
    run_context = RunContext(inputs=inputs, seed=0)
    decoded, _ = run.predict(config, inputs)
    home = team_model.compute(run_context, model=model_path).home
    rule = controlled_touch.compute(run_context, **touch_params).home
    visible = kinematics.ball_track(inputs).visible
    served = visible & ~np.isnan(home) & ((decoded == HOME) | (decoded == AWAY))
    decoded_home = decoded == HOME
    part = {}
    for name, truth_codes in (("raw", codes), ("coarse", coarse)):
        frames = served & ((truth_codes == HOME) | (truth_codes == AWAY))
        part[name] = _score(frames, truth_codes == HOME, home, decoded_home, rule)
        if name == "raw":
            raw_frames, raw_home = frames, truth_codes == HOME
    over = over11_frames(inputs)
    confidence = touch_params["confidence"]
    with np.errstate(invalid="ignore"):
        episode = ~np.isnan(rule) & (np.abs(rule - 0.5) >= confidence - 0.5 - 1e-9)
    groups = {"episode": episode, "hold": ~np.isnan(rule) & ~episode, "silent": np.isnan(rule), "over11": over,
              "not_over11": ~over}
    part["classes"] = {}
    for name, mask in groups.items():
        frames = raw_frames & mask
        scored = _score(frames, raw_home, home, decoded_home, rule)
        part["classes"][name] = {"frames": scored["frames"], "p19": scored["p19"][0], "main": scored["main"]}
    return part


def _cell(tally):
    def ratio(right, frames):
        return right / frames if frames else None

    def scored(part):
        frames = part["frames"]
        p19_right, p19_loss = part["p19"]
        rule_right, rule_loss, silent = part["rule"]
        return {"frames": frames,
                "p19": {"right": p19_right, "accuracy": ratio(p19_right, frames),
                        "log_loss": ratio(p19_loss, frames)},
                "main": {"right": part["main"], "accuracy": ratio(part["main"], frames)},
                "rule": {"right": rule_right, "accuracy": ratio(rule_right, frames), "silent": silent,
                         "silent_share": ratio(silent, frames), "log_loss": ratio(rule_loss, frames)}}

    cell = scored(tally["raw"])
    cell["coarse"] = scored(tally["coarse"])
    cell["classes"] = {name: {"frames": part["frames"],
                              "p19": {"right": part["p19"], "accuracy": ratio(part["p19"], part["frames"])},
                              "main": {"right": part["main"], "accuracy": ratio(part["main"], part["frames"])}}
                       for name, part in tally["classes"].items()}
    return cell


def render_feasibility(report):
    """The report's lines: a table of the matches and the pool, the classes, R021's statistic (labelled as gating
    nothing), then the gate and the verdict."""
    def number(value):
        return "-" if value is None else "{:.4f}".format(value)

    def difference(cell):
        accuracy, main = cell["p19"]["accuracy"], cell["main"]["accuracy"]
        return "-" if accuracy is None or main is None else "{:+.4f}".format(accuracy - main)

    verdict = report["go"]
    lines = ["feasible  profile {}, {} seed(s), against {}: out of fold, seeds pooled per match".format(
                 report["profile"], report["seeds"], report["config"]),
             "feasible  {:<8} {:>10} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>16} {:>8} {:>8} | {:>9}".format(
                 "match", "frames", "p19 acc", "main acc", "rule acc", "p19-main", "p19 ll", "rule ll", "rule silent",
                 "coarse19", "coarse m", "ll lower")]
    for name, cell in list(sorted(report["per_match"].items())) + [("pooled", report["pooled"])]:
        share = cell["rule"]["silent_share"]
        silent = "{:,} ({})".format(cell["rule"]["silent"], "-" if share is None else "{:.1%}".format(share))
        lower = verdict["lower"].get(name)
        lines.append(
            "feasible  {:<8} {:>10,} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>16} {:>8} {:>8} | {:>9}".format(
                name, cell["frames"], number(cell["p19"]["accuracy"]), number(cell["main"]["accuracy"]),
                number(cell["rule"]["accuracy"]), difference(cell), number(cell["p19"]["log_loss"]),
                number(cell["rule"]["log_loss"]), silent, number(cell["coarse"]["p19"]["accuracy"]),
                number(cell["coarse"]["main"]["accuracy"]), "-" if lower is None else "yes" if lower else "no"))
    for name, part in report["pooled"]["classes"].items():
        lines.append("feasible  class {:<10} {:>10,} frames, p19 {} main {}".format(
            name, part["frames"], number(part["p19"]["accuracy"]), number(part["main"]["accuracy"])))
    statistic = verdict["per_frame"]
    lines.append("feasible  R021's per-frame statistic (p19-main above), which gates nothing: a gain of 0.020 in {} of "
                 "{} matches (R021 needed {}), pooled gain {}".format(
                     statistic["wins"], len(report["per_match"]), R021_MATCHES,
                     "held" if statistic["pooled"] else "not held"))
    lines.append("feasible  the gate: pooled p19 log-loss {} against at most {:.4f} ({}); lower than the rule's on {} "
                 "of {} matches (every one needed)".format(
                     number(verdict["pooled_log_loss"]), verdict["loss_max"], "held" if verdict["pooled"] else
                     "not held", verdict["matches_lower"], len(verdict["lower"])))
    lines.append("feasible  {}".format("GO: the benchmark of P32's configs follows" if verdict["go"]
                                       else "NO GO: P32 stops here, the negative result is recorded"))
    return lines


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train P32's team model (P19's, team features version 2) on the benchmark matches (folds and "
                    "full), write it as JSON, verify it against a retraining, count its rows (--dry-run) or run its "
                    "feasibility test (--feasibility).")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="every this many graded frames is a row (default {})".format(DEFAULT_STRIDE))
    parser.add_argument("--out", help="where to write the artifact, or to read it from with --feasibility (default "
                                      "pipeline/models/team_model_hgb.json)")
    parser.add_argument("--verify", action="store_true",
                        help="retrain and compare with the committed file instead of writing it: exit 1 if "
                             "anything but the source hashes differs")
    parser.add_argument("--dry-run", action="store_true",
                        help="gather the rows and fit nothing: rows per match, NaN shares, dead columns, peak memory; "
                             "exit 1 on an all-NaN or a constant column, rows outside the declared window or a "
                             "projected peak over {:g} GB".format(PEAK_LIMIT_GB))
    parser.add_argument("--feasibility", action="store_true",
                        help="train nothing: read the artifact and score it out of fold against the config's decoded "
                             "labels and the rule with its hold; writes <data>/{}/feasibility.json".format(
                                 "/".join(FEASIBILITY_FOLDER)))
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="the config the model is compared with (--feasibility; default "
                             "pipeline/benchmark/configs/p30/p30_w030.toml)")
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to count (--dry-run only)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    given = {text.split("=")[0] for text in (sys.argv[1:] if argv is None else argv)}
    if args.seeds < 1 or args.stride < 1:
        parser.error("--seeds and --stride must be at least 1")
    if args.dry_run and (args.verify or args.feasibility):
        parser.error("--dry-run takes no --verify or --feasibility: it gathers rows and fits nothing")
    if args.matches and not args.dry_run:
        parser.error("--matches is for --dry-run only: training and the feasibility use all seven matches")
    if args.feasibility and (args.verify or "--stride" in given):
        parser.error("--feasibility takes no --verify or --stride: it reads the artifact and tests it, and trains "
                     "nothing")
    if "--config" in given and not args.feasibility:
        parser.error("--config is for --feasibility only")
    path = args.out or DEFAULT_MODEL_PATH
    try:
        if args.dry_run:
            report = dry_run(args.profile, args.seeds, args.matches, args.stride, data_dir=args.data_dir, log=print)
            for line in render_dry_run(report):
                print(line)
            return 0 if report["ok"] else 1
        if args.feasibility:
            try:
                with open(path, encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, ValueError) as error:
                raise BenchmarkError("cannot read the model {} ({}); train it with `python {}`".format(
                    path, error, team_model.TRAINER)) from error
            report = feasibility(artifact, args.config, args.profile, args.seeds, data_dir=args.data_dir)
            for line in render_feasibility(report):
                print(line)
            where = os.path.join(args.data_dir or games.DATA_DIR, *FEASIBILITY_FOLDER, "feasibility.json")
            os.makedirs(os.path.dirname(where), exist_ok=True)
            with open(where, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(dumps(report))
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
        raise ValueError("train_team: {} must be a whole number of at least {}, got {!r}".format(name, minimum, value))
    return int(value)


def _known_profile(profile):
    if profile not in run.PROFILES:
        raise BenchmarkError("train_team: profile {!r} is not one of: {}".format(profile,
                                                                                 ", ".join(sorted(run.PROFILES))))


def _calibration_for(profile, calibration, data_dir):
    """The Calibration a profile that reads one takes its masks from (measured or read from the cache if none is
    given); `generic` and `clean` read none, so a calibration given to them is refused rather than silently unused."""
    if not run.needs_calibration(profile) and calibration is not None:
        raise BenchmarkError("train_team: profile {!r} takes no calibration, and one was given".format(profile))
    if run.needs_calibration(profile) and calibration is None:
        return calibration_module.load(data_dir=data_dir)
    return calibration


def _seed_list(profile, seeds):
    return [0] if profile == run.CLEAN else list(range(seeds))


def _own_id(match, match_id):
    if match.match_id != match_id:
        raise BenchmarkError("train_team: the match given as {} says it is {}; its fold would be filed under one id "
                             "and looked up under the other, and scored by the model that saw it"
                             .format(match_id, match.match_id))


if __name__ == "__main__":
    sys.exit(main())
