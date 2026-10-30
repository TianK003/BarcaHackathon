"""label_diff: how a config changes BAR-ATH's labels beside a reference config's, in aggregates only.

    python pipeline/benchmark/label_diff.py REFERENCE.toml CONFIG.toml [CONFIG.toml ...] [--data-dir D]

    out = diff(reference_codes, codes, inputs)       # {"d_share", "x_share", "possessions", "changed_share", ...}
    for line in render("config.toml", out): print(line)

Why it exists. A source that cannot be benchmarked (Smart Tagging: it exists for our match only, P08)
is judged by what it does to the labels of the match it can be run on, and the label-free checks of P12 and P09 ask the
same question of a decoder change. With no truth to score against, the honest figures are how much of the output moves
and where: the share of graded frames whose label differs, how many of those are frames where the ball is not seen (a
change there is the new source speaking where the tracking cannot), the dead-ball and abstention shares, the possessions
per half (the scorer's definition: a maximal run of H or of A inside a graded block, read after the scorer's own fill
of abstentions, so that H X H is one possession) and the dead
balls' onsets, which a timing source moves without adding or removing any. Possessions and onsets are counted on the
codes with X forward-filled inside each graded block (pm.fill_abstentions, as the temporal metrics see them); the shares
and the changed frames are counted on the raw codes, where an X is a label.

A D onset is the first frame of a D run inside a block. A reference onset moved if the nearest new onset within
MATCH_WINDOW frames (the nearer, the earlier on a tie) is at a different frame; with none that near it is `lost`, and
it is neither moved nor kept. `median_move` is the median of the moves in frames, signed new minus reference, and None
if none moved.

Both code arrays are the whole match's int8 codes (what benchmark.run.predict returns, the CSV the pipeline would write)
and only the graded frames are counted. Nothing here prints a frame number or a per-frame value (CLAUDE.md, hard rules);
the CLI runs each config on BAR-ATH through the benchmark's own predict, writes no CSV, and is heavy (it loads the
match's tracking and runs every source).
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

from pipeline.config import ConfigError, load_config  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
H, A, D, X = pm.H, pm.A, pm.D, pm.X
MATCH_WINDOW = 50       # frames: a reference onset is paired with the nearest new onset within this far


def _runs(codes, start, stop, wanted):
    """{code: [(first, last_exclusive)]} of the maximal runs inside [start, stop) of each code in `wanted`."""
    out = {code: [] for code in wanted}
    block = codes[start:stop]
    if not len(block):
        return out
    edges = np.flatnonzero(np.diff(block)) + 1
    for first, last in zip(np.r_[0, edges], np.r_[edges, len(block)]):
        if int(block[first]) in out:
            out[int(block[first])].append((start + int(first), start + int(last)))
    return out


def _possessions(codes, blocks):
    """{half: count} of the runs of H plus the runs of A inside the half's graded block."""
    counts = {}
    for half, start, stop in blocks:
        runs = _runs(codes, start, stop, (H, A))
        counts[half] = len(runs[H]) + len(runs[A])
    return counts


def _d_onsets(codes, blocks):
    """The sorted first frames of the D runs inside the blocks."""
    return np.array(sorted(first for _, start, stop in blocks for first, _ in _runs(codes, start, stop, (D,))[D]),
                    dtype=np.int64)


def _moves(ref, new):
    """(moves, lost): the signed moves (new - ref) of the reference onsets that found a new onset within the window."""
    moves, lost = [], 0
    for onset in ref:
        gaps = new - onset
        near = np.flatnonzero(np.abs(gaps) <= MATCH_WINDOW)
        if not len(near):
            lost += 1
            continue
        # the nearest; on a tie the earlier (the smaller signed gap)
        best = min(near, key=lambda i: (abs(gaps[i]), gaps[i]))
        if gaps[best] != 0:
            moves.append(int(gaps[best]))
    return moves, lost


def diff(reference_codes, codes, inputs):
    """The aggregates of how `codes` differ from `reference_codes` on the graded frames of `inputs` (module docstring).

    Both are (n_frames,) int8 arrays of the match's codes. ValueError if either is not that long. Counts and shares
    only: no frame number is in the result.
    """
    reference_codes, codes = np.asarray(reference_codes), np.asarray(codes)
    for name, array in (("reference_codes", reference_codes), ("codes", codes)):
        if array.shape != (inputs.n_frames,):
            raise ValueError("{} must have one code per frame ({} of them), got shape {}: pass the whole match's "
                             "codes, as predict returns them".format(name, inputs.n_frames, array.shape))
    graded = np.asarray(inputs.graded_mask, dtype=bool)
    n = int(graded.sum())
    if not n:
        raise ValueError("there are no graded frames in these inputs, so there is nothing to compare")
    ref, new = reference_codes[graded], codes[graded]
    changed = ref != new
    n_changed = int(changed.sum())
    ball_missing = np.isnan(np.asarray(inputs.atd.ball_xy)[graded]).any(axis=1)

    spans = [(start, stop) for _, start, stop in inputs.graded]
    filled_ref, filled_new = pm.fill_abstentions(reference_codes, spans), pm.fill_abstentions(codes, spans)
    ref_onsets, new_onsets = _d_onsets(filled_ref, inputs.graded), _d_onsets(filled_new, inputs.graded)
    moves, lost = _moves(ref_onsets, new_onsets)
    before, after = _possessions(filled_ref, inputs.graded), _possessions(filled_new, inputs.graded)
    return {
        "d_share": [float((ref == D).sum()) / n, float((new == D).sum()) / n],
        "x_share": [float((ref == X).sum()) / n, float((new == X).sum()) / n],
        "possessions": {half: [before[half], after[half]] for half in before},
        "changed_share": n_changed / n,
        "changed_in_gaps_share": float((changed & ball_missing).sum()) / n_changed if n_changed else 0.0,
        "d_onsets": {"ref": int(len(ref_onsets)), "new": int(len(new_onsets)), "moved": len(moves), "lost": lost,
                     "median_move": float(np.median(moves)) if moves else None},
    }


def render(name, out):
    """The printed lines for one config's aggregates: figures, never a frame."""
    onsets = out["d_onsets"]
    move = "none moved" if onsets["median_move"] is None else "median move {:+.1f} frames".format(onsets["median_move"])
    return [
        "config    {}".format(name),
        "          changed {:.2%} of graded frames, {:.1%} of them where the ball is not seen".format(
            out["changed_share"], out["changed_in_gaps_share"]),
        "          D {:.2%} -> {:.2%}   X {:.2%} -> {:.2%}".format(*out["d_share"], *out["x_share"]),
        "          possessions {}".format("   ".join(
            "half {}: {} -> {}".format(half, *pair) for half, pair in sorted(out["possessions"].items()))),
        "          D onsets {} -> {}, {} moved, {} lost, {}".format(
            onsets["ref"], onsets["new"], onsets["moved"], onsets["lost"], move),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="How configs change BAR-ATH's labels beside a reference config's, in aggregates (HEAVY: runs "
                    "every config on the real match, writes no CSV).")
    parser.add_argument("reference", metavar="REFERENCE",
                        help="the reference TOML config, e.g. pipeline/configs/main.toml")
    parser.add_argument("configs", nargs="+", metavar="CONFIG", help="the TOML configs to compare with it")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)

    from pipeline.benchmark import run
    from pipeline.inputs import Inputs
    try:
        inputs = Inputs(data_dir=args.data_dir)
        reference, _ = run.predict(load_config(args.reference), inputs)
        for path in args.configs:
            codes, _ = run.predict(load_config(path), inputs)
            print("\n".join(render(os.path.basename(path), diff(reference, codes, inputs))))
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
