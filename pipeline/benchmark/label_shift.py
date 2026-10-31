"""P22(a)'s tools: does the EM share beat guessing 0.5 on the DFL matches, and what does it do on BAR-ATH.

    python pipeline/benchmark/label_shift.py feasibility [--profile tracking] [--seeds 3] [--model PATH]
                                                         [--data-dir D] [--out PATH]
    python pipeline/benchmark/label_shift.py bar-ath --strength S [--model PATH] [--data-dir D] [--out PATH]

    result = feasibility()                  # {profile, seeds, model, per_match: {id: cell}, gate}
    result = bar_ath(0.5)                   # {prior_home, iterations, converged, frames, band, inside_band, touch}

Why it exists. `shape_possession` can add s * logit(pi) to its logit, pi the home share that Saerens' EM reads off the
model's own outputs (pipeline/evidence/shape_possession.py). That is worth building into a config only if pi is a
better guess of the share than 0.5, which is what a hybrid HMM's flat prior already says. On the seven DFL matches the
truth is known, so the question is asked there, before any score is: each match is served as the pipeline serves it,
scored by the fold that never saw it (`folds[inputs.game]`, as train_shape.report does), and EM is run on the logit of
every graded frame with an opinion, one pi per match and seed. `gate` is the advisor's rule: the mean |pi - pi_true|
over the matches must be at most GATE_FACTOR times the mean |0.5 - pi_true|. The truth is read beside the inputs
(`truths.truth_codes`), never through them: pi comes from the MatchInputs alone.

Two figures are reported and never gated (the advisor's D1, for the retry of R020b): `slope`, the calibration slope of
the logit against the truth on the alive frames (1 is calibrated; EM assumes it, and an underconfident model, a slope
above 1, makes EM crawl), and `pi_hat_alive`, EM run on the truth's alive frames only, which is the share EM would see
if a dead ball were never asked about. The first is a slope-only logistic fit: on mirrored rows the intercept is 0, so
there is none to fit (research.md section 2).

On BAR-ATH there is no truth, so `bar_ath` reports what is label-free: pi of the `full` model, whether it lies in
BAND (the advisor's; outside it a person looks at `label_diff` aggregates before anything is adopted), and
`touch_agreement` (train_shape.py) with and without the shift. The shift must not cost more than TOUCH_SLACK of it.

Nothing here prints or writes a frame: a per-match cell holds counts, shares and one number per seed. Both commands
are heavy (they load the DFL matches or our tracking); the JSON is written as run.py writes its reports.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import train_shape as trainer  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.benchmark.run import BenchmarkError  # noqa: E402
from pipeline.benchmark.train_dead_ball import served_inputs  # noqa: E402
from pipeline.evidence import shape_possession, trees  # noqa: E402

# Params
GATE_FACTOR = 0.75                   # the advisor's: the mean error must be at most this times that of guessing 0.5
BAND = (0.39, 0.68)                  # the advisor's band for BAR-ATH's pi; outside it, sign-off with label_diff
TOUCH_SLACK = 0.005                  # shifted touch_agreement must be at least the unshifted one minus this
SLOPE_BAND = (0.8, 1.25)             # outside it the slope names miscalibration, the cause a retry may name
DEFAULT_MODEL_PATH = trainer.DEFAULT_MODEL_PATH
OUT_DIR = ("runs", "label_shift")    # <data dir>/runs/label_shift/feasibility.json and bar_ath.json
SLOPE_STEPS, SLOPE_TOLERANCE = 50, 1e-10


# -- the pieces, pure -------------------------------------------------------------------------------------------------

def _inside(codes, graded):
    mask = np.zeros(len(codes), dtype=bool)
    for _, start, stop in graded:
        mask[start:stop] = True
    return mask


def home_share(codes, graded):
    """The share of H among the H and A codes inside the graded blocks (pi_true); None with no alive frame there."""
    codes = np.asarray(codes)
    inside = _inside(codes, graded)
    home = int(((codes == trainer.HOME) & inside).sum())
    away = int(((codes == trainer.AWAY) & inside).sum())
    return home / (home + away) if home + away else None


def gate(rows):
    """The advisor's rule on `rows` = [{"pi_true": float, "pi_hat": [float per seed]}], one per match.

    error is the mean over matches of the mean over seeds of |pi_hat - pi_true|; baseline the mean over matches of
    |0.5 - pi_true|; ratio = error / baseline (None if the baseline is 0); passes if error <= factor * baseline.
    """
    error = float(np.mean([np.mean([abs(p - row["pi_true"]) for p in row["pi_hat"]]) for row in rows]))
    baseline = float(np.mean([abs(0.5 - row["pi_true"]) for row in rows]))
    return {"error": error, "baseline": baseline, "ratio": error / baseline if baseline else None,
            "factor": GATE_FACTOR, "passes": bool(error <= GATE_FACTOR * baseline)}


def calibration_slope(z, y):
    """The slope a of the logistic fit P(y = 1) = sigmoid(a * z), no intercept; 1 where z is calibrated.

    Newton from a = 1, stopping at |step| < 1e-10 or after 50 steps; nothing random. None if z has no spread (all 0).
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.any(z):
        return None
    a = 1.0
    for _ in range(SLOPE_STEPS):
        p = trees.sigmoid(a * z)
        step = float(np.sum((y - p) * z) / np.sum(p * (1.0 - p) * z * z))
        a += step
        if abs(step) < SLOPE_TOLERANCE:
            break
    return float(a)


def touch_verdict(unshifted, shifted):
    """True if the shifted touch agreement is at least the unshifted one minus TOUCH_SLACK; False if either is None."""
    return bool(unshifted is not None and shifted is not None and shifted >= unshifted - TOUCH_SLACK)


# -- the feasibility on DFL -------------------------------------------------------------------------------------------

def feasibility(model=DEFAULT_MODEL_PATH, *, profile=run.DEFAULT_PROFILE, seeds=run.DEFAULT_SEEDS, matches=None,
                calibration=None, data_dir=None, log=None):
    """Does EM's pi beat 0.5 on the DFL matches? {profile, seeds, model, per_match: {id: cell}, gate}.

    A cell is {pi_true, alive_frames, seeds: [{pi_hat, iterations, converged, frames} per seed], abs_error,
    abs_baseline, slope, pi_hat_alive}; slope and pi_hat_alive are the advisor's D1, reported and never gated (slope
    pools the seeds' alive opinion frames; pi_hat_alive is the mean over seeds, None where EM raises on them).
    BenchmarkError for an unknown profile, a model that cannot be read or lacks the fold of a match (the source would
    score it with `full`, which saw it), a match with no alive frame, and a share too extreme to be shifted by.
    Numbers and strings only, no per-frame array. `matches` is {id: DflMatch} to use instead of loading them.
    """
    seeds = trainer._whole("seeds", seeds, 1)
    trainer._known_profile(profile)
    ids = run.resolve_match_ids(None, matches)
    path = str(model)
    try:
        folds = shape_possession.load(path)["folds"]
    except ValueError as error:
        raise BenchmarkError("label_shift: {}".format(error)) from error
    absent = [match_id for match_id in sorted(ids) if match_id not in folds]
    if absent:
        raise BenchmarkError("label_shift: the model has no fold for {}; the source would score that match with "
                             "`full`, which saw it, and the share would be read in sample; retrain with `python {}`"
                             .format(", ".join(absent), "pipeline/benchmark/train_shape.py"))
    calibration = trainer._calibration_for(profile, calibration, data_dir)
    seed_list = trainer._seed_list(profile, seeds)

    per_match = {}
    for match_id in sorted(ids):
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        trainer._own_id(match, match_id)
        codes, _ = truths.truth_codes(match)
        codes = np.asarray(codes)
        pi_true = home_share(codes, match.graded)
        if pi_true is None:
            raise BenchmarkError("label_shift: {} has no alive frame in its graded blocks, so there is no true share "
                                 "to compare with".format(match_id))
        alive = ((codes == trainer.HOME) | (codes == trainer.AWAY)) & _inside(codes, match.graded)
        is_home = codes == trainer.HOME
        cells, alive_shares, pooled_z, pooled_y = [], [], [], []
        for seed in seed_list:
            inputs = served_inputs(match, profile, seed, calibration, data_dir)
            z = shape_possession.logit_home(inputs, path)
            try:
                found = shape_possession.em_prior(z)
            except ValueError as error:
                raise BenchmarkError("label_shift: {}, seed {}: {}".format(match_id, seed, error)) from error
            cells.append({"pi_hat": found["prior_home"], "iterations": found["iterations"],
                          "converged": found["converged"], "frames": found["frames"]})
            masked = np.where(alive, z, np.nan)
            try:
                alive_shares.append(shape_possession.em_prior(masked)["prior_home"])
            except ValueError:
                alive_shares.append(None)
            take = alive & np.isfinite(z)
            pooled_z.append(z[take])
            pooled_y.append(is_home[take])
            del inputs, z, masked
        hats = [cell["pi_hat"] for cell in cells]
        per_match[match_id] = {
            "pi_true": pi_true, "alive_frames": int(alive.sum()), "seeds": cells,
            "abs_error": float(np.mean([abs(p - pi_true) for p in hats])), "abs_baseline": abs(0.5 - pi_true),
            "slope": calibration_slope(np.concatenate(pooled_z), np.concatenate(pooled_y)),
            "pi_hat_alive": None if None in alive_shares else float(np.mean(alive_shares))}
        if log:
            log("{}: pi_true {:.4f}, pi_hat {}".format(match_id, pi_true, ", ".join("{:.4f}".format(p) for p in hats)))
        del match, codes
    verdict = gate([{"pi_true": cell["pi_true"], "pi_hat": [s["pi_hat"] for s in cell["seeds"]]}
                    for cell in per_match.values()])
    return {"profile": profile, "seeds": len(seed_list), "model": run.shown(path), "per_match": per_match,
            "gate": verdict}


def render_feasibility(result):
    """The printed lines: one a match (the true share, EM's share per seed, the errors, the slope), then the gate."""
    def number(value):
        return "-" if value is None else "{:.3f}".format(value)
    lines = ["label_shift feasibility: profile {}, {} seeds, model {}".format(
        result["profile"], result["seeds"], result["model"])]
    for match_id, cell in sorted(result["per_match"].items()):
        converged = all(seed["converged"] for seed in cell["seeds"])
        lines.append("  {}  pi_true {}  pi_hat {}  |err| {} vs {}  slope {}  pi_alive {}{}".format(
            match_id, number(cell["pi_true"]), " ".join(number(s["pi_hat"]) for s in cell["seeds"]),
            number(cell["abs_error"]), number(cell["abs_baseline"]), number(cell["slope"]),
            number(cell["pi_hat_alive"]), "" if converged else "  NOT CONVERGED"))
    verdict = result["gate"]
    lines.append("  mean |err| {} against {} for guessing 0.5: ratio {}, the gate is {}: {}".format(
        number(verdict["error"]), number(verdict["baseline"]), number(verdict["ratio"]), number(verdict["factor"]),
        "PASS" if verdict["passes"] else "FAIL"))
    return lines


# -- the checks on BAR-ATH --------------------------------------------------------------------------------------------

def bar_ath(strength, *, model=DEFAULT_MODEL_PATH, data_dir=None, inputs=None):
    """EM's share on our match, its band, and touch_agreement with and without the shift.

    {prior_home, iterations, converged, frames, band, inside_band, touch: {unshifted: {frames, agreement}, shifted:
    {...}, difference, passes}}: `full` is the model (BAR-ATH has no fold), `shifted` is prior = "em" at `strength`.
    `inputs` stands in for Inputs(data_dir) in the checks. Label-free: aggregates and nothing else.
    """
    if inputs is None:
        from pipeline.inputs import Inputs
        inputs = Inputs(data_dir=data_dir)
    path = str(model)
    try:
        found = shape_possession.em_prior(shape_possession.logit_home(inputs, path))
    except ValueError as error:
        raise BenchmarkError("label_shift: {}".format(error)) from error
    unshifted = trainer.touch_agreement(inputs, model=path)
    shifted = trainer.touch_agreement(inputs, model=path, prior="em", prior_strength=strength)
    low, high = unshifted["agreement"], shifted["agreement"]
    return {"prior_home": found["prior_home"], "iterations": found["iterations"], "converged": found["converged"],
            "frames": found["frames"], "band": list(BAND),
            "inside_band": bool(BAND[0] <= found["prior_home"] <= BAND[1]),
            "touch": {"unshifted": unshifted, "shifted": shifted,
                      "difference": None if low is None or high is None else high - low,
                      "passes": touch_verdict(low, high)}}


def render_bar_ath(result):
    touch = result["touch"]

    def number(value):
        return "-" if value is None else "{:.4f}".format(value)
    return ["label_shift bar-ath: pi_hat {} after {} updates ({}), {} frames; band {}-{}: {}".format(
                number(result["prior_home"]), result["iterations"],
                "converged" if result["converged"] else "NOT CONVERGED", result["frames"], result["band"][0],
                result["band"][1], "inside" if result["inside_band"] else "OUTSIDE"),
            "  touch agreement {} on {} frames, shifted {} on {}: difference {}, slack {}: {}".format(
                number(touch["unshifted"]["agreement"]), touch["unshifted"]["frames"],
                number(touch["shifted"]["agreement"]), touch["shifted"]["frames"], number(touch["difference"]),
                TOUCH_SLACK, "PASS" if touch["passes"] else "FAIL")]


# -- the command line -------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="P22(a): EM's share against guessing 0.5 on the DFL matches, and the checks on BAR-ATH "
                    "(HEAVY: both load tracking).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("feasibility", "bar-ath"):
        command = commands.add_parser(name)
        command.add_argument("--model", default=DEFAULT_MODEL_PATH, help="the shape model JSON (default: committed)")
        command.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        command.add_argument("--out", help="the JSON to write (default: <data dir>/runs/label_shift/<command>.json)")
        if name == "feasibility":
            command.add_argument("--profile", default=run.DEFAULT_PROFILE, help="the degradation profile")
            command.add_argument("--seeds", type=int, default=run.DEFAULT_SEEDS, help="degradation seeds per match")
        else:
            command.add_argument("--strength", type=float, required=True, help="prior_strength, in (0, 1]")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "feasibility" and args.seeds < 1:
        build_parser().error("--seeds must be at least 1")
    try:
        if args.command == "feasibility":
            result = feasibility(args.model, profile=args.profile, seeds=args.seeds, data_dir=args.data_dir, log=print)
            lines, name = render_feasibility(result), "feasibility.json"
        else:
            result = bar_ath(args.strength, model=args.model, data_dir=args.data_dir)
            lines, name = render_bar_ath(result), "bar_ath.json"
        for line in lines:
            print(line)
        out = args.out or os.path.join(args.data_dir or games.DATA_DIR, *OUT_DIR, name)
        run.write_json(out, run.plain(result))
        print("wrote     {}".format(run.shown(out)))
        return 0
    except (BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
