"""The adapter, in .venv: the degraded match to windows of complete nodes that PathCRF's own code can read.

PathCRF was built for tracking where every player has one identity for the whole episode. Ours has 46 anonymous slots
that appear, vanish and swap, and the model refuses a window with a NaN in an active node. The adapter is the
simplest completion that meets the advisor's rules (advisor-01 section 6, D1 to D4), and the cost of every
completion is charged to PathCRF, never corrected:

- D1, windows: the maximal runs that main.toml decodes as H or A inside a graded block (a change between H and A does
  not cut a run, or main.toml's team boundaries would leak into the competitor's input), cut into windows of at most
  250 frames (10 s). A run under 16 frames gets none: their kinematics need 16 frames.
- Pieces: inside a window, a slot's maximal present runs, each in its metadata team (a wrong-team slot stays where
  the metadata XML puts it).
- Joins: a piece that ends is joined to one that begins in the same metadata team, at most 2 s later and 5.0 m away,
  greedily by smallest gap, so that one player's track is one node again; chains are allowed.
- At most 11 nodes per team: the chains with the fewest observed frames are dropped.
- D2, complete tracks: a join's gap is filled linearly and a node is held at its first or last position to the
  window's edges. The share of node-frames filled or held is reported; a node with no observed frame in a window is
  a pad node, as in their code (their `_pad_nodes` names it `<team>_pad_k`).

Nothing here reads a label of the match: `prepare` hands the match to the benchmark's `served_inputs`, which gives
the degrader a view without it, and every function below takes arrays of the served tracking and main.toml's codes.
Pure numpy up to `prepare`; the pipeline's modules are imported inside it, so that importing this file costs nothing
and loads no torch.

    python pathcrf_lab/adapter.py prepare --match J03WR9 --profile tracking --seeds 0 1 2 [--data-dir D]

The files go to <data>/cache/pathcrf/<match>/<profile>/seed<s>/ (windows.npz, frames.npz, meta.json), all under
data/ and so never committed. Only counts are printed.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import namedtuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    # Run by path, Python put pathcrf_lab/ itself first on sys.path: put the repository root there instead.
    sys.path[0] = ROOT
if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.append(os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402

# Params
MAX_WINDOW_FRAMES = 250
MIN_RUN_FRAMES = 16
MAX_JOIN_FRAMES = 50
MAX_JOIN_M = 5.0
PER_TEAM = 11
PITCH = (105.0, 68.0)
MAIN_TOML = "pipeline/configs/main.toml"
MAIN_TOML_SHA256 = "124a38ad019995e276cdd6d1094f764e226e167408c0adaa7c32c466bac25873"
SHAPE_MODEL = "pipeline/models/shape_possession_hgb.json"
SHAPE_SHA256 = "641e0bad7438a5a987f4aa8ac54ea48302b22d2f7375ea655e67a8ded1802974"
FLAG_NAMES = ("joins", "over11", "dropped", "padded_home", "padded_away", "held_frames")

Piece = namedtuple("Piece", "slot first last")  # last is inclusive
Nodes = namedtuple("Nodes", "xy team names flags")


def alive_runs(codes, graded):
    """(runs, short): the maximal runs of H or A inside each graded block, half-open, and those under MIN_RUN_FRAMES.

    `graded` is inputs.graded, [(half, start, stop)]. A run ends at a D or X frame and at a block's edge; a change
    between H and A does not end it. `short` holds the runs that are too short for a window, as (start, stop).
    """
    codes = np.asarray(codes)
    live = (codes == pm.H) | (codes == pm.A)
    runs, short = [], []
    for _, start, stop in graded:
        inside = np.concatenate([[False], live[start:stop], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(inside))
        for a, b in zip(edges[::2], edges[1::2]):
            (runs if b - a >= MIN_RUN_FRAMES else short).append((start + int(a), start + int(b)))
    return runs, short


def split_run(start, stop):
    """[(start, stop)]: ceil(L / MAX_WINDOW_FRAMES) windows of near-equal length, the longer ones first."""
    length = stop - start
    count = -(-length // MAX_WINDOW_FRAMES)
    base, extra = divmod(length, count)
    out, at = [], start
    for i in range(count):
        size = base + (1 if i < extra else 0)
        out.append((at, at + size))
        at += size
    return out


def pieces(present, start, stop):
    """[Piece(slot, first, last)]: the maximal runs of True per slot of present[start:stop], in absolute indices."""
    found = []
    block = np.asarray(present)[start:stop]
    for slot in range(block.shape[1]):
        padded = np.concatenate([[0], block[:, slot].astype(np.int8), [0]])
        edges = np.flatnonzero(np.diff(padded))
        for a, b in zip(edges[::2], edges[1::2]):
            found.append(Piece(slot, start + int(a), start + int(b) - 1))
    return found


def join(piece_list, xy_m, slot_team):
    """(chains, n_joins): pieces linked end to start, each end and each start used once.

    A pair (a, b) is a candidate when both are in the same metadata team, b starts 1 to MAX_JOIN_FRAMES frames after a
    ends and the two positions are at most MAX_JOIN_M metres apart (inclusive). Candidates are taken in the order
    (gap, distance, a.slot, b.slot, a.last). A chain is a list of pieces in time order.
    """
    candidates = []
    for i, a in enumerate(piece_list):
        for j, b in enumerate(piece_list):
            gap = b.first - a.last
            if slot_team[a.slot] != slot_team[b.slot] or not 1 <= gap <= MAX_JOIN_FRAMES:
                continue
            dist = float(math.hypot(*(xy_m[b.first, b.slot] - xy_m[a.last, a.slot])))
            if dist <= MAX_JOIN_M:
                candidates.append((gap, dist, a.slot, b.slot, a.last, i, j))
    candidates.sort()
    following, has_before = {}, set()
    for *_, i, j in candidates:
        if i in following or j in has_before:
            continue
        following[i] = j
        has_before.add(j)
    chains = []
    for i in range(len(piece_list)):
        if i in has_before:
            continue
        chain, at = [piece_list[i]], i
        while at in following:
            at = following[at]
            chain.append(piece_list[at])
        chains.append(chain)
    return chains, len(following)


def _track(chain, xy_m, start, stop):
    """(stop - start, 2) track of one chain: observed where seen, linear across join gaps, held outside; filled count."""
    track = np.empty((stop - start, 2), dtype=np.float64)
    observed = 0
    first = chain[0]
    track[: first.first - start] = xy_m[first.first, first.slot]
    for k, piece in enumerate(chain):
        track[piece.first - start: piece.last - start + 1] = xy_m[piece.first: piece.last + 1, piece.slot]
        observed += piece.last - piece.first + 1
        if k + 1 < len(chain):
            after = chain[k + 1]
            n = after.first - piece.last
            fraction = (np.arange(1, n) / n)[:, None]
            a, b = xy_m[piece.last, piece.slot], xy_m[after.first, after.slot]
            track[piece.last - start + 1: after.first - start] = a + fraction * (b - a)
    last = chain[-1]
    track[last.last - start + 1:] = xy_m[last.last, last.slot]
    return track, (stop - start) - observed


def nodes(chains, xy_m, start, stop, slot_team):
    """Nodes: at most PER_TEAM tracks per team in rows 0 to 10 (home) and 11 to 21 (away), NaN on unused rows.

    The chains with the fewest observed frames are dropped, and on a tie the one with the higher first slot.
    `flags` counts joins (chains minus pieces are not needed: the caller's), dropped chains, padded rows per team and
    the frames filled by interpolation or by holding (D2).
    """
    xy = np.full((stop - start, 2 * PER_TEAM, 2), np.nan, dtype=np.float32)
    team = np.full(2 * PER_TEAM, -1, dtype=np.int8)
    names = [""] * (2 * PER_TEAM)
    dropped = held = 0
    padded = []
    for side in (0, 1):
        mine = [c for c in chains if slot_team[c[0].slot] == side]
        weight = lambda c: (-sum(p.last - p.first + 1 for p in c), c[0].slot)  # noqa: E731
        kept = sorted(sorted(mine, key=weight)[:PER_TEAM], key=lambda c: (c[0].first, c[0].slot))
        dropped += len(mine) - len(kept)
        for k, chain in enumerate(kept):
            row = side * PER_TEAM + k
            track, filled = _track(chain, xy_m, start, stop)
            xy[:, row] = track
            team[row] = side
            names[row] = "{}_{}".format("home" if side == 0 else "away", k)
            held += filled
        padded.append(PER_TEAM - len(kept))
    flags = {"joins": 0, "over11": 0, "dropped": dropped, "padded_home": padded[0], "padded_away": padded[1],
             "held_frames": held}
    return Nodes(xy, team, np.array(names), flags)


def window_nodes(atd, start, stop):
    """Nodes of the window [start, stop) of `atd`: metres, pieces, joins, at most 11 a team, gaps filled."""
    present_all = ~np.isnan(atd.players_xy[start:stop, :, 0])
    xy_m = atd.players_xy[start:stop].astype(np.float64) * np.array(PITCH)
    found = pieces(present_all, 0, stop - start)
    chains, n_joins = join(found, xy_m, atd.slot_team)
    result = nodes(chains, xy_m, 0, stop - start, atd.slot_team)
    over11 = any((present_all[:, atd.slot_team == side].sum(axis=1) > PER_TEAM).any() for side in (0, 1))
    result.flags["joins"] = n_joins
    result.flags["over11"] = int(over11)
    return result


def prepare_arrays(inputs, main_codes, p07_home):
    """(windows, frames, counts): the files' arrays and their aggregates, from the served inputs and two label-free
    per-frame outputs (main.toml's codes and P07's home probability)."""
    runs, short = alive_runs(main_codes, inputs.graded)
    spans = [window for run in runs for window in split_run(*run)]
    built = [window_nodes(inputs.atd, a, b) for a, b in spans]
    n = len(spans)
    flags = np.array([[b.flags[name] for name in FLAG_NAMES] for b in built], dtype=np.int32).reshape(n, len(FLAG_NAMES))
    windows = {
        "win_start": np.array([a for a, _ in spans], dtype=np.int64),
        "win_stop": np.array([b for _, b in spans], dtype=np.int64),
        "node_xy": (np.concatenate([b.xy for b in built]) if built
                    else np.zeros((0, 2 * PER_TEAM, 2), dtype=np.float32)),
        "node_team": np.array([b.team for b in built], dtype=np.int8).reshape(n, 2 * PER_TEAM),
        "node_names": np.array([b.names for b in built]).reshape(n, 2 * PER_TEAM),
        "flags": flags,
    }
    window_of_row = np.full(inputs.n_frames, -1, dtype=np.int32)
    for w, (a, b) in enumerate(spans):
        window_of_row[a:b] = w
    frames = {"main_codes": np.asarray(main_codes, dtype=np.int8), "p07_home": np.asarray(p07_home, dtype=np.float64),
              "ball_present": np.asarray(inputs.atd.ball_present, dtype=bool), "window_of_row": window_of_row}
    graded_rows = sum(stop - start for _, start, stop in inputs.graded)
    kept_nodes = int((windows["node_team"] >= 0).sum(axis=1).dot(windows["win_stop"] - windows["win_start"]))
    held = int(flags[:, FLAG_NAMES.index("held_frames")].sum())
    counts = {
        "windows": n, "rows_covered": int((window_of_row >= 0).sum()), "short_runs": len(short),
        "short_frames": int(sum(b - a for a, b in short)),
        "joins": int(flags[:, 0].sum()), "over11_windows": int(flags[:, 1].sum()),
        "dropped_nodes": int(flags[:, 2].sum()), "padded_home": int(flags[:, 3].sum()),
        "padded_away": int(flags[:, 4].sum()), "held_frames": held, "node_frames": kept_nodes,
        "held_share": held / kept_nodes if kept_nodes else 0.0,
        "main_d_share": float((np.asarray(main_codes)[np.concatenate(
            [np.arange(a, b) for _, a, b in inputs.graded])] == pm.D).sum()) / graded_rows,
    }
    return windows, frames, counts


def check_pinned(path, sha256):
    """ValueError naming the file and both hashes unless its bytes, CRLF taken as LF, hash to `sha256`.

    The worktree has core.autocrlf on, and the declaration pins the committed LF bytes.
    """
    with open(path, "rb") as handle:
        got = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    if got != sha256:
        raise ValueError("{} hashes to {} (LF bytes), but the declaration pins {}; the lane must run the pinned "
                         "file".format(path, got, sha256))


def prepare(match_id, profile, seed, *, data_dir=None):
    """Serve the degraded match, decode it with main.toml, take P07's fold, write the window files; the counts.

    The match goes only to served_inputs, which hands the degrader a view without its labels. Needs .venv, the DFL
    cache and the Calibration (heavy: the controller runs it).
    """
    import games
    from pipeline.benchmark import calibration, degrade, dfl, run
    from pipeline.benchmark.train_dead_ball import served_inputs
    from pipeline.config import load_config
    from pipeline.contracts import RunContext
    from pipeline.evidence import shape_possession

    data_dir = data_dir or games.DATA_DIR
    main_toml, shape_model = (os.path.join(ROOT, p) for p in (MAIN_TOML, SHAPE_MODEL))
    check_pinned(main_toml, MAIN_TOML_SHA256)
    check_pinned(shape_model, SHAPE_SHA256)
    match = dfl.load_match(match_id, data_dir=data_dir)
    inputs = served_inputs(match, profile, seed, calibration.load(data_dir=data_dir), data_dir)
    config, dropped = run.drop_match_only(load_config(main_toml))
    assert tuple(dropped) == ("smart_tagging",), "main.toml dropped {} for a DFL match, expected smart_tagging".format(
        dropped)
    main_codes = run.predict(config, inputs)[0]
    p07_home = shape_possession.compute(RunContext(inputs=inputs, seed=0), model=shape_model, scope="all").home
    windows, frames, counts = prepare_arrays(inputs, main_codes, p07_home)

    folder = os.path.join(data_dir, "cache", "pathcrf", match_id, profile, "seed{}".format(seed))
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "windows.npz"), **windows)
    np.savez_compressed(os.path.join(folder, "frames.npz"), **frames)
    meta = {"match": match_id, "seed": seed, "profile": profile, "profile_version": degrade.PROFILES[profile],
            "main_toml_sha256": MAIN_TOML_SHA256, "shape_model_sha256": SHAPE_SHA256,
            "params": {"MAX_WINDOW_FRAMES": MAX_WINDOW_FRAMES, "MIN_RUN_FRAMES": MIN_RUN_FRAMES,
                       "MAX_JOIN_FRAMES": MAX_JOIN_FRAMES, "MAX_JOIN_M": MAX_JOIN_M, "PER_TEAM": PER_TEAM,
                       "PITCH": list(PITCH)}}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("prepare")
    one.add_argument("--match", default="J03WR9")
    one.add_argument("--profile", default="tracking")
    one.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    one.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    for seed in args.seeds:
        counts = prepare(args.match, args.profile, seed, data_dir=args.data_dir)
        print("seed {}: ".format(seed) + ", ".join(
            "{}={}".format(k, round(v, 4) if isinstance(v, float) else v) for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
