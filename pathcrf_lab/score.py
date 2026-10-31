"""The scorer, in .venv: the only reader of the truth, and the mechanical go/no-go of R022 part 1.

Everything before this module is label-free: the adapter and the runner never see a truth code. Here the truth meets
the two predictions, on the same frames, and the declaration of R022 part 1 is applied as written:

- the reference gate first: P07, scored as R011 scored it (silent frames left out) on the literal frames, must give
  96,442 frames and accuracy 0.8495, or the whole comparison is not on the frames R011 used and the lane stops;
- the verdict: PathCRF minus P07 on the primary frames (the literal frames where main.toml decodes H or A), pooled
  over the seeds by frames, at least +0.04 pooled and +0.02 on each seed. Both models are scored on the same frames,
  so the delta is a fraction of integers and is compared as one: floats would call 0.94 - 0.90 a miss of 0.04;
- PathCRF is right iff its sender's team is the truth's; out of play and silence are wrong (D4). P07 is right iff it
  speaks and its side of 0.5 is the truth's; its silence is wrong on the primary frames;
- reported, not gated: the literal set, the coarse truth, the clean run on the same frames, the window classes, and
  the shares of out of play, silence and held node-frames; the retry condition.

    python pathcrf_lab/score.py --match J03WR9 --profile tracking --seeds 0 1 2 [--data-dir D]

Only aggregates are printed and written (<data>/runs/pathcrf/r022.json): counts and accuracies, never a row.
"""

import argparse
import json
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    # Run by path, Python put pathcrf_lab/ itself first on sys.path: put the repository root there instead.
    sys.path[0] = ROOT
if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.append(os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402

# Params
GO_POOLED = Fraction(4, 100)
GO_PER_SEED = Fraction(2, 100)
HOME_LEVEL = 0.5
REFERENCE = {"frames": 96442, "accuracy": 0.8495}
OUT = 2
SILENT = -1
CLASSES = ("join", "over11", "neither", "outside")
SUMS = ("frames", "p07_right", "pathcrf_right", "out", "silent_pathcrf", "silent_p07")


def tally(truth, wanted, p07_home, pathcrf):
    """Counts on the frames `wanted` (a bool array): both models on the same frames, silence and out of play wrong."""
    truth, wanted = np.asarray(truth), np.asarray(wanted, dtype=bool)
    home = truth[wanted] == pm.H
    p07 = np.asarray(p07_home)[wanted]
    code = np.asarray(pathcrf)[wanted]
    speaks = ~np.isnan(p07)
    p07_right = speaks & ((np.where(speaks, p07, 0.0) >= HOME_LEVEL) == home)
    team = (code == 0) | (code == 1)
    pathcrf_right = team & ((code == 1) == home)
    return {"frames": int(wanted.sum()), "p07_right": int(p07_right.sum()), "pathcrf_right": int(pathcrf_right.sum()),
            "out": int((code == OUT).sum()), "silent_pathcrf": int((code == SILENT).sum()),
            "silent_p07": int((~speaks).sum())}


def reference_tally(truth, literal, p07_home):
    """R011's scoring of P07: the literal frames where it speaks (silent frames left out); {frames, right}."""
    p07 = np.asarray(p07_home)
    take = np.asarray(literal, dtype=bool) & ~np.isnan(p07)
    home = np.asarray(truth)[take] == pm.H
    return {"frames": int(take.sum()), "right": int(((p07[take] >= HOME_LEVEL) == home).sum())}


def pooled(parts):
    """The sums of tallies (or of reference tallies) over seeds: by frames, never a mean of the seeds' accuracies."""
    keys = parts[0].keys() if parts else SUMS
    return {key: sum(part[key] for part in parts) for key in keys}


def reference_gate(parts):
    """(ok, frames, accuracy): the pooled reference tallies against R011's 96,442 frames and 0.8495 (four decimals)."""
    total = pooled(parts)
    accuracy = total["right"] / total["frames"] if total["frames"] else float("nan")
    ok = total["frames"] == REFERENCE["frames"] and round(accuracy, 4) == REFERENCE["accuracy"]
    return ok, total["frames"], accuracy


def delta(part):
    """PathCRF minus P07 as an exact Fraction of the frames, or None without frames."""
    if not part["frames"]:
        return None
    return Fraction(part["pathcrf_right"] - part["p07_right"], part["frames"])


def verdict(per_seed, pooled_part):
    """{"go", "pooled_delta", "per_seed_delta", "reason"}: go iff pooled >= +0.04 and every seed >= +0.02, exactly.

    Deltas are compared as Fractions of integer counts; they become floats only in the returned dict, after the decision.
    """
    deltas = [delta(part) for part in per_seed]
    total = delta(pooled_part)
    if total is None or any(d is None for d in deltas):
        return {"go": False, "pooled_delta": None, "per_seed_delta": [None] * len(per_seed),
                "reason": "no frames to compare"}
    misses = []
    if total < GO_POOLED:
        misses.append("pooled {:+.4f} < +0.04".format(float(total)))
    misses += ["seed {} {:+.4f} < +0.02".format(i, float(d)) for i, d in enumerate(deltas) if d < GO_PER_SEED]
    return {"go": not misses, "pooled_delta": float(total), "per_seed_delta": [float(d) for d in deltas],
            "reason": "; ".join(misses) if misses else "pooled >= +0.04 and every seed >= +0.02"}


def retry_allowed(by_class, *, go=False):
    """R022 part 1's retry condition: a no-go whose pooled delta on the windows with neither a join nor a team over 11
    is at least +0.04. `by_class` is {class: pooled tally}."""
    part = by_class.get("neither")
    d = delta(part) if part else None
    return (not go) and d is not None and d >= GO_POOLED


def clean_codes(match, clean_npz):
    """(codes per row, alignment): their clean run mapped from DFL's frame numbers to rows.

    alignment = {mapped, unmapped, alive_agreement}; the clean codes are used only if nothing is unmapped and their
    `alive` agrees with the match's alive frames on every mapped graded row, which `score` checks.
    """
    from pipeline.benchmark import truth as truths
    return clean_codes_with(match, clean_npz, truths.truth_codes(match)[0])


def clean_codes_with(match, clean_npz, truth):
    """clean_codes with the truth's codes given: the pure part, which the checks run on a synthetic match."""
    row_of = {int(n): r for r, n in enumerate(np.asarray(match.frame_number)) if n >= 0}
    rows = np.array([row_of.get(int(n), -1) for n in clean_npz["frame_number"]], dtype=np.int64)
    mapped = rows >= 0
    codes = np.full(match.n_frames, SILENT, dtype=np.int8)
    codes[rows[mapped]] = clean_npz["team"][mapped]
    graded = mapped & (truth[np.where(mapped, rows, 0)] != pm.X)
    alive_truth = (truth[rows[graded]] == pm.H) | (truth[rows[graded]] == pm.A)
    agreement = float((alive_truth == np.asarray(clean_npz["alive"])[graded]).mean()) if graded.any() else 0.0
    return codes, {"mapped": int(mapped.sum()), "unmapped": int((~mapped).sum()), "alive_agreement": agreement}


def _seed_tables(truth, coarse, graded_mask, frames, flags, pathcrf, clean):
    """{table name: tally} for one seed: primary, literal, coarse, clean and the window classes on the primary frames."""
    main = frames["main_codes"]
    literal = graded_mask & ((truth == pm.H) | (truth == pm.A)) & ~frames["ball_present"]
    primary = literal & ((main == pm.H) | (main == pm.A))
    home = frames["p07_home"]
    window = frames["window_of_row"]
    in_window = window >= 0
    joins = np.zeros(len(window), dtype=bool)
    over = np.zeros(len(window), dtype=bool)
    joins[in_window] = flags[window[in_window], 0] > 0
    over[in_window] = flags[window[in_window], 1] > 0
    classes = {"join": in_window & joins, "over11": in_window & over, "neither": in_window & ~joins & ~over,
               "outside": ~in_window}
    tables = {"primary": tally(truth, primary, home, pathcrf), "literal": tally(truth, literal, home, pathcrf),
              "coarse": tally(coarse, primary, home, pathcrf)}
    if clean is not None:
        tables["clean"] = tally(truth, primary, home, clean)
    for name, mask in classes.items():
        tables["class_" + name] = tally(truth, primary & mask, home, pathcrf)
    return tables, reference_tally(truth, literal, home)


def score(match_id, profile, seeds, *, data_dir=None):
    """The report: reference gate, tables per seed and pooled, the verdict and the retry condition."""
    import games
    from pipeline.benchmark import dfl
    from pipeline.benchmark import truth as truths
    data_dir = data_dir or games.DATA_DIR
    match = dfl.load_match(match_id, data_dir=data_dir)
    truth, _ = truths.truth_codes(match)
    coarse = truths.coarse_codes(truth, match.graded, match.fps)
    graded_mask = np.zeros(match.n_frames, dtype=bool)
    for _, start, stop in match.graded:
        graded_mask[start:stop] = True

    clean, alignment = None, None
    path = os.path.join(data_dir, "cache", "pathcrf", match_id, "clean", "clean.npz")
    if os.path.exists(path):
        with np.load(path) as npz:
            codes, alignment = clean_codes(match, npz)
        if alignment["unmapped"] == 0 and alignment["alive_agreement"] == 1.0:
            clean = codes

    per_seed, references, held, node_frames = [], [], 0, 0
    for seed in seeds:
        folder = os.path.join(data_dir, "cache", "pathcrf", match_id, profile, "seed{}".format(seed))
        with np.load(os.path.join(folder, "frames.npz")) as f:
            frames = {k: f[k] for k in f.files}
        with np.load(os.path.join(folder, "windows.npz")) as w:
            flags = w["flags"]
            held += int(flags[:, 5].sum())
            node_frames += int(((w["node_team"] >= 0).sum(axis=1) * (w["win_stop"] - w["win_start"])).sum())
        with np.load(os.path.join(folder, "pathcrf.npz")) as p:
            pathcrf = p["team"]
        tables, reference = _seed_tables(truth, coarse, graded_mask, frames, flags, pathcrf, clean)
        per_seed.append(tables)
        references.append(reference)

    ok, ref_frames, ref_accuracy = reference_gate(references)
    names = per_seed[0].keys()
    total = {name: pooled([seed[name] for seed in per_seed]) for name in names}
    report = {"match": match_id, "profile": profile, "seeds": list(seeds),
              "reference": {"ok": ok, "frames": ref_frames, "accuracy": ref_accuracy, "expected": REFERENCE},
              "clean_alignment": alignment, "clean_used": clean is not None, "pooled": total, "per_seed": per_seed,
              "held_share": held / node_frames if node_frames else 0.0}
    if not ok:
        report["verdict"] = None
        report["stop"] = "STOP: the reference does not reproduce"
        return report
    report["verdict"] = verdict([seed["primary"] for seed in per_seed], total["primary"])
    classes = {name[len("class_"):]: total[name] for name in names if name.startswith("class_")}
    report["retry_allowed"] = retry_allowed(classes, go=report["verdict"]["go"])
    return report


def render(report):
    """The report's lines: counts and accuracies to four decimals, deltas signed; never a row."""
    def acc(part, key):
        return "{:.4f}".format(part[key] / part["frames"]) if part["frames"] else "-"

    lines = ["R022 {} {} seeds {}".format(report["match"], report["profile"], report["seeds"]),
             "reference P07 (literal, silent left out): {} frames, {:.4f} (expected {} and {})".format(
                 report["reference"]["frames"], report["reference"]["accuracy"],
                 REFERENCE["frames"], REFERENCE["accuracy"])]
    if report.get("stop"):
        return lines + [report["stop"]]
    lines.append("table             frames   P07  PathCRF   delta   out  silent")
    for name, part in report["pooled"].items():
        d = delta(part)
        lines.append("{:<14} {:>9} {:>7} {:>7} {:>8} {:>5} {:>6}".format(
            name, part["frames"], acc(part, "p07_right"), acc(part, "pathcrf_right"),
            "-" if d is None else "{:+.4f}".format(float(d)),
            "{:.3f}".format(part["out"] / part["frames"]) if part["frames"] else "-",
            "{:.3f}".format(part["silent_pathcrf"] / part["frames"]) if part["frames"] else "-"))
    v = report["verdict"]
    lines.append("held node-frame share {:.4f}".format(report["held_share"]))
    lines.append("clean used: {} {}".format(report["clean_used"], report["clean_alignment"]))
    lines.append("VERDICT: {} ({})".format("GO" if v["go"] else "NO-GO", v["reason"]))
    lines.append("pooled delta {} per seed {}".format(
        None if v["pooled_delta"] is None else "{:+.4f}".format(v["pooled_delta"]),
        ["{:+.4f}".format(d) for d in v["per_seed_delta"]]))
    lines.append("retry allowed: {}".format(report["retry_allowed"]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", default="J03WR9")
    parser.add_argument("--profile", default="tracking")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    import games
    data_dir = args.data_dir or games.DATA_DIR
    report = score(args.match, args.profile, args.seeds, data_dir=data_dir)
    for line in render(report):
        print(line)
    out = os.path.join(data_dir, "runs", "pathcrf")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "r022.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=lambda o: None)
        handle.write("\n")
    return 1 if report.get("stop") else 0


if __name__ == "__main__":
    sys.exit(main())
