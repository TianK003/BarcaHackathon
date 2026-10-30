"""whistle_stats: is the whistle detector's output near our own decoded stoppages, label-free (P10, R018).

    python pipeline/benchmark/whistle_stats.py feasibility|bar [--data-dir D]      # HEAVY: decodes the match's audio
    near = near_stoppage(w, dead, blocks)          # per detection: a decoded D frame in [w - 25, w + 50], in w's block
    out = feasibility(w, dead, blocks)             # per half counts, share_near, the control, the ratio, the lag quartiles
    out = bar(w, dead, blocks)                     # share_near with a 90% bootstrap over detections, the control, the ratio

The pseudo-labels. There is no ground truth for a whistle, and there must not be one (no hand labels on this match, no
event data). The comparison is with main.toml's own decoded D runs on BAR-ATH, as main.toml stands at R018's declaration
commit (its sha256 is printed). P05 sees a foul only when the players slow down, about a second after the whistle, so a
whistle counts as near a stoppage when a decoded D frame lies in [w - 25, w + 50] frames (-1 s to +2 s) inside w's graded
block. It is never measured on the exact frames after the whistle, which would count the most useful whistles (the ones
P05 is late for) against it.

The control. Each half's detections are shifted circularly inside that half by 5, 10 ... 60 s and the share is measured
again; the control is the mean over the offsets. It keeps the number of detections and their spacing, and loses only
their alignment with the stoppages: crowd whistling, which is not tied to play, would score about the control.

The lags. For each detection, the frames from w to the nearest edge of a decoded D run inside the block: the run's first
frame, or the first frame after it (an edge at a block's edge is not one: nothing is known beyond it). Signed, edge minus
w: negative when the stoppage began before the whistle. Only their quartiles leave this module.

The gate (R018 part 1) and the bar are the declared rules, in `gate` and `bar_verdict`; the numbers are the declaration's.
Nothing here prints or writes a frame: counts, shares, quartiles, hashes and pass/fail (CLAUDE.md, hard rules).
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.evidence import whistle  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
BEFORE, AFTER = 25, 50                      # frames: [w - 1 s, w + 2 s]
OFFSETS_S = tuple(range(5, 61, 5))          # the control's circular shifts
FPS = 25
GATE = {"per_half": (10, 120), "share_near": 0.50, "ratio": 2.0}                 # R018 part 1, as declared
BAR = {"share_near": 0.70, "low": 0.60, "ratio": 1.5}
SEED = (18, 0)
NEAR_DB = 0.5                               # a feature this close to its threshold is one a different ffmpeg could flip
CONFIG = "pipeline/configs/main.toml"
CRLF, LF = b"\r\n", b"\n"                   # the config's hash is printed raw and with CRLF read as LF (the blob's)


def _bounds(w, blocks):
    """(starts, stops) of the block of each frame of `w`; ValueError if one is in no block."""
    w = np.asarray(w, dtype=np.int64)
    starts = np.array([start for _, start, _ in blocks], dtype=np.int64)
    stops = np.array([stop for _, _, stop in blocks], dtype=np.int64)
    index = np.searchsorted(starts, w, side="right") - 1
    if (index < 0).any() or (w >= stops[np.clip(index, 0, None)]).any():
        raise ValueError("every detection must lie in a graded block")
    return index, starts[index], stops[index]


def _near_each(w, dead, blocks, shift=0):
    """Bool per detection: a D frame in [w - BEFORE, w + AFTER] clipped to the block, `w` first moved by `shift`."""
    csum = np.r_[0, np.cumsum(np.asarray(dead, dtype=bool))]
    w = np.asarray(w, dtype=np.int64)
    _, starts, stops = _bounds(w, blocks)
    if shift:
        w = starts + (w - starts + shift) % (stops - starts)
    low, high = np.maximum(w - BEFORE, starts), np.minimum(w + AFTER + 1, stops)
    return csum[high] - csum[low] > 0


def near_stoppage(w, dead, blocks):
    """Per detection w: True where a decoded D frame (`dead`, bool per frame) lies in [w - 25, w + 50] inside w's block."""
    return _near_each(w, dead, blocks)


def shift_within_halves(w, blocks, offset_frames):
    """w moved forward by `offset_frames`, circularly inside its own block: never out of its half."""
    w = np.asarray(w, dtype=np.int64)
    _, starts, stops = _bounds(w, blocks)
    return starts + (w - starts + int(offset_frames)) % (stops - starts)


def _control(w, dead, blocks, offsets_s):
    return float(np.mean([_near_each(w, dead, blocks, shift=int(s * FPS)).mean() for s in offsets_s]))


def _ratio(share, control):
    return None if control <= 0 else share / control


def _edge_lags(w, dead, blocks):
    """Signed frames from each detection to the nearest edge of a D run inside its block; NaN where the block has none."""
    dead = np.asarray(dead, dtype=bool)
    w = np.asarray(w, dtype=np.int64)
    index, _, _ = _bounds(w, blocks)
    lags = np.full(len(w), np.nan)
    for b, (_, start, stop) in enumerate(blocks):
        inside = dead[start:stop].astype(np.int8)
        change = np.flatnonzero(np.diff(np.r_[0, inside, 0])) + start     # a run's first frame, the first after it
        edges = change[(change > start) & (change < stop)]               # none at the block's own edges
        mine = np.flatnonzero(index == b)
        if len(edges) and len(mine):
            gaps = edges[None, :] - w[mine][:, None]
            lags[mine] = gaps[np.arange(len(mine)), np.abs(gaps).argmin(axis=1)]
    return lags


def feasibility(w, dead, blocks, *, offsets_s=OFFSETS_S):
    """The feasibility figures (module docstring). "lags" holds the signed per-detection lags: `public` drops them."""
    w = np.asarray(w, dtype=np.int64)
    if not len(w):
        # no detection is a failed gate, on file: the counts are zero and the statistics are undefined
        return {"per_half": {int(half): 0 for half, _, _ in blocks}, "n": 0, "share_near": None,
                "control_share_near": None, "ratio": None, "lag_to_edge": {"q1": None, "median": None, "q3": None},
                "lags": []}
    near = _near_each(w, dead, blocks)
    index, _, _ = _bounds(w, blocks)
    share = float(near.mean())
    control = _control(w, dead, blocks, offsets_s)
    lags = _edge_lags(w, dead, blocks)
    known = lags[~np.isnan(lags)]
    q1, median, q3 = (float(q) for q in np.percentile(known, [25, 50, 75])) if len(known) else (None,) * 3
    return {"per_half": {int(blocks[b][0]): int((index == b).sum()) for b in range(len(blocks))},
            "n": int(len(w)), "share_near": share, "control_share_near": control, "ratio": _ratio(share, control),
            "lag_to_edge": {"q1": q1, "median": median, "q3": q3},
            "lags": [int(x) for x in known]}


def public(out):
    """`out` without its per-detection lags: what may be written to a report."""
    return {key: value for key, value in out.items() if key != "lags"}


def bar(w, dead, blocks, *, draws=2000, seed=SEED, offsets_s=OFFSETS_S):
    """share_near with a 90% bootstrap over detections (low, high), the control's share and the ratio."""
    w = np.asarray(w, dtype=np.int64)
    if not len(w):
        return {"n": 0, "share_near": None, "low": None, "high": None, "control_share_near": None, "ratio": None}
    near = _near_each(w, dead, blocks)
    rng = np.random.default_rng(list(seed))
    means = near[rng.integers(0, len(near), size=(int(draws), len(near)))].mean(axis=1)
    share = float(near.mean())
    control = _control(w, dead, blocks, offsets_s)
    return {"n": int(len(w)), "share_near": share, "low": float(np.percentile(means, 5)),
            "high": float(np.percentile(means, 95)), "control_share_near": control, "ratio": _ratio(share, control)}


def gate(out):
    """R018's feasibility gate on `feasibility`'s figures: {"pass", "checks"} (each declared condition, by name)."""
    low, high = GATE["per_half"]
    checks = {"per_half": all(low <= n <= high for n in out["per_half"].values()),
              "share_near": out["share_near"] is not None and out["share_near"] >= GATE["share_near"],
              "ratio": out["ratio"] is not None and out["ratio"] >= GATE["ratio"]}
    return {"pass": all(checks.values()), "checks": checks}


def bar_verdict(out):
    """R018's bar on `bar`'s figures: {"pass", "checks"}."""
    checks = {"share_near": out["share_near"] is not None and out["share_near"] >= BAR["share_near"],
              "low": out["low"] is not None and out["low"] >= BAR["low"],
              "ratio": out["ratio"] is not None and out["ratio"] >= BAR["ratio"]}
    return {"pass": all(checks.values()), "checks": checks}


def _q(value):
    return "n/a" if value is None else "{:+.1f}".format(value)


def _share(value):
    return "n/a" if value is None else "{:.3f}".format(value)


def _ratio_text(out):
    return "n/a" if out["ratio"] is None else "{:.2f}".format(out["ratio"])


def _verdict_text(name, verdict):
    return "{:<13}{}   {}".format(name, "PASS" if verdict["pass"] else "FAIL", "   ".join(
        "{} {}".format(key, "ok" if ok else "no") for key, ok in verdict["checks"].items()))


def render_feasibility(out, verdict):
    """The printed lines of the feasibility figures: counts, shares, quartiles and pass/fail, never a frame."""
    lag = out["lag_to_edge"]
    return [
        "detections   {} ({})".format(out["n"], "   ".join(
            "half {}: {}".format(half, n) for half, n in sorted(out["per_half"].items()))),
        "share near   {}   control {}   ratio {}".format(_share(out["share_near"]), _share(out["control_share_near"]),
                                                         _ratio_text(out)),
        "lag to edge  quartiles {} / {} / {} frames (signed, edge minus whistle)".format(
            _q(lag["q1"]), _q(lag["median"]), _q(lag["q3"])),
        _verdict_text("gate", verdict),
    ]


def render_bar(out, verdict):
    """The printed lines of the bar's figures: shares and pass/fail."""
    return [
        "detections   {}".format(out["n"]),
        "share near   {}   90% bootstrap {} to {}   control {}   ratio {}".format(
            _share(out["share_near"]), _share(out["low"]), _share(out["high"]), _share(out["control_share_near"]),
            _ratio_text(out)),
        _verdict_text("bar", verdict),
    ]


def near_threshold(features, first, stop, tonal_db):
    """How many detections have a feature within NEAR_DB of its threshold on a frame of the blast or the one after it.

    These are the detections a different ffmpeg build, resampling by another rounding, could flip (design 5.D).
    """
    features = np.asarray(features)
    count = 0
    for a, b in zip(first, stop):
        seen = features[a:min(b + 1, len(features))]
        if (np.abs(seen[:, 0] - tonal_db) <= NEAR_DB).any() or \
                (np.abs(seen[:, 1] - whistle.BAND_DB) <= NEAR_DB).any():
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P10's whistle statistics against main.toml's decoded stoppages on BAR-ATH, in aggregates "
                    "(HEAVY: decodes the match's audio the first time, runs main.toml, writes no CSV).")
    parser.add_argument("mode", choices=("feasibility", "bar"))
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)

    from pipeline.benchmark import run
    from pipeline.inputs import Inputs
    try:
        path = os.path.join(ROOT, CONFIG)
        with open(path, "rb") as handle:
            raw_bytes = handle.read()
        # both, so that a CRLF working copy (Windows) and the committed LF blob can each be matched to the declaration
        sha = {"raw": hashlib.sha256(raw_bytes).hexdigest(),
               "LF": hashlib.sha256(raw_bytes.replace(CRLF, LF)).hexdigest()}
        print("config       {} sha256 raw {} LF {}".format(CONFIG, sha["raw"], sha["LF"]))
        inputs = Inputs(data_dir=args.data_dir)
        codes, _ = run.predict(load_config(path), inputs)
        dead = np.asarray(codes) == pm.D
        features = inputs.audio_features
        feature_sha = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
        print("features     sha256 {}".format(feature_sha))
        tonal_db = whistle.tonal_threshold(features, inputs.graded_mask)
        print("tonal threshold  {:.2f} dB (the {}th percentile of tonal_db over the graded frames)".format(
            tonal_db, whistle.TONAL_QUANTILE))
        first, stop = whistle.runs(features, inputs.graded_mask)
        flippable = near_threshold(features, first, stop, tonal_db)
        if args.mode == "feasibility":
            out = feasibility(first, dead, inputs.graded)
            verdict = gate(out)
            lines = render_feasibility(out, verdict)
        else:
            out = bar(first, dead, inputs.graded)
            verdict = bar_verdict(out)
            lines = render_bar(out, verdict)
        lines.append("near a threshold  {} of {} detections have a feature within {} dB of one".format(
            flippable, len(first), NEAR_DB))
        print("\n".join(lines))
        data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
        folder = os.path.join(data_dir, "runs", "whistle")
        os.makedirs(folder, exist_ok=True)
        report = {"mode": args.mode, "config": CONFIG, "config_sha256": sha, "feature_sha256": feature_sha, "near_threshold": flippable, "tonal_threshold_db": tonal_db,
                  "verdict": verdict, "figures": public(out)}
        with open(os.path.join(folder, args.mode + ".json"), "w", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (ConfigError, run.BenchmarkError, OSError, ValueError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
