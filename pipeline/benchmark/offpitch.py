"""The past-the-line ruler, and our match's target for it (P18, R015).

    stats = line_stats(ball_present, blocks, ball_xy_m, pitch)       # the share of visible ball frames past a line
    spans = kickoff_spans(inputs)                                    # BAR-ATH's spans, as calibration.measure has them
    ours = target(data_dir=None, rebuild=False, inputs=None)         # line_stats of BAR-ATH, cached as JSON

    python pipeline/benchmark/offpitch.py target [--rebuild] [--data-dir D]     # prints the dict

Why it exists. Our tracker sometimes reports the ball outside the pitch: a frame on which the ball has a position that
is past one of the four lines, and a dead-ball detector can read that as "out of play" for nothing but the position.
DFL's tracking reports the ball only while it is in play, so a benchmark match shows the pipeline no such frames. The
`offpitch` profile (degrade.py, a later task) therefore hides DFL's ball once it is more than m metres past a line, and
m has to be fitted so that the degraded DFL matches show as many of those frames as ours does. "As many" needs a ruler,
and the ruler has to be one function: line_stats measures BAR-ATH here (target) and the fit of m applies the same
function, unchanged, to the degraded matches, the way calibration.ball_stats is the shared ruler of the gaps. It takes
plain arrays and knows nothing of either match.

Definitions (metres, from a pitch corner; the line distance is calibration's, min(x, L - x, y, W - y), negative beyond a
line)
    visible         the flag says the ball is present and BOTH coordinates are finite (calibration._usable). A flag set
                    where the position is absent is a ValueError: the two come from different trackings.
    beyond          a visible frame whose line distance is below 0, strictly: a ball exactly on a line is on the pitch.
    beyond share    beyond frames / visible frames, over the blocks (NaN when no frame is visible), the number the fit
                    of m matches; beyond_m_quantiles are the quantiles of how far past the line the beyond frames are
                    (-line distance), for the report and for a look at the tail, not for the fit.

The target. Ours is measured on the kick-off spans (each half from its kick-off to the end of its graded block), as
calibration.measure's spans are, because the degraded matches start at their kick-off. kickoff_spans repeats measure's
construction, since measure keeps its spans in a local variable and calibration.py is not edited (its hash keys the
Calibration's cache). The result is cached as JSON under <data dir>/cache/benchmark/offpitch/target-<16 hex>.json, the
key being the sha256 of the ATD cache's source record, this file's own sha256 (LF-normalised), numpy's version, the pitch
and the fps, so that an edit of this file, a new ATD or a new numpy is a new file and the old ones are left alone.

The fit comes in a later task: this module measures and does not fit. The assumption behind matching the share is that
our out-of-play time is like DFL's, that is, that when the ball leaves the pitch the same share of the match's visible
frames is spent beyond a line in the two matches once DFL's ball is hidden from m metres out.

It holds no frame. The target has scalars and quantiles only (visible_frames, beyond_frames, beyond_line_share,
beyond_m_quantiles, a version and a key), and nothing about BAR-ATH is printed but what that file holds.
"""

import argparse
import hashlib
import json
import math
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

from pipeline.benchmark import calibration  # noqa: E402

# Params
VERSION = 1                                                    # of the target's definitions: bump on a change of them
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90))   # of the distance beyond
CACHE_FOLDER = ("cache", "benchmark", "offpitch")              # <data dir>/cache/benchmark/offpitch/target-<16 hex>.json
KEY_DIGITS = 16                                                # hex digits of the key's sha256 that name the file
DECIMALS = 9                                                   # a stored float is rounded to 1e-9, as calibration's are


def line_stats(ball_present, blocks, ball_xy_m, pitch):
    """The ball's frames past a line inside the blocks, as a dict.

    ball_present  (n,) bool: the tracker sees the ball.
    blocks        [(half, start, stop)], half-open, disjoint, one per half: the stretches to measure.
    ball_xy_m     (n, 2) metres from a pitch corner, NaN where the ball is missing.
    pitch         (length, width) in metres.

    Returns visible_frames, beyond_frames (visible with a line distance below 0, strictly), beyond_line_share
    (beyond / visible, NaN when none is visible) and beyond_m_quantiles, {name: float} of -line distance over the
    beyond frames (QUANTILES; {} when there is none). ValueError, naming the argument, for a flag that is not a (n,)
    bool array, a position that is not (n, 2), a flag set where the position is absent, a bad pitch and bad blocks.
    """
    flag = calibration._bool_array("ball_present", ball_present, 1)
    n = len(flag)
    xy = np.asarray(ball_xy_m, dtype=np.float64)
    if xy.shape != (n, 2):
        raise ValueError("offpitch.line_stats: ball_present has {} frames but ball_xy_m has shape {}; both are one "
                         "value per frame of the match, the position in metres".format(n, xy.shape))
    length, width = calibration._pitch(pitch)
    blocks = calibration._blocks(blocks, n)
    finite = np.isfinite(xy).all(axis=1)
    stray = int((~flag & finite).sum())
    if stray:
        raise ValueError("offpitch.line_stats: ball_present and ball_xy_m disagree on {} frames: the ball is marked "
                         "absent where a position is given; build both from the same tracking".format(stray))
    visible = flag & finite
    in_blocks = np.zeros(n, dtype=bool)
    for _, start, stop in blocks:
        in_blocks[start:stop] = True
    visible &= in_blocks
    x, y = xy[visible, 0], xy[visible, 1]
    line = np.minimum.reduce([x, length - x, y, width - y])
    past = -line[line < 0.0]
    visible_frames = int(visible.sum())
    return {
        "visible_frames": visible_frames,
        "beyond_frames": int(past.size),
        "beyond_line_share": past.size / visible_frames if visible_frames else math.nan,
        "beyond_m_quantiles": {name: float(np.quantile(past, level)) for name, level in QUANTILES} if past.size else {},
    }


def kickoff_spans(inputs):
    """[(half, kickoff, stop)]: each graded half from its kick-off to the end of its block (calibration.measure's spans).

    inputs needs .graded and .kickoffs. ValueError for a half without a kick-off, or one that is not inside its graded
    block.
    """
    spans = []
    for half, start, stop in inputs.graded:
        if half not in inputs.kickoffs:
            raise ValueError("offpitch.kickoff_spans: inputs.kickoffs has no kick-off for half {}".format(half))
        if not start <= inputs.kickoffs[half] < stop:
            raise ValueError("offpitch.kickoff_spans: half {}: the kick-off, frame {}, is not inside its graded block "
                             "{}-{}".format(half, inputs.kickoffs[half], start, stop))
        spans.append((half, inputs.kickoffs[half], stop))
    return spans


def target(data_dir=None, rebuild=False, inputs=None):
    """line_stats of BAR-ATH on its kick-off spans, with "version" and "key"; from the cache when it is fresh.

    data_dir defaults to games.DATA_DIR (MTAH_DATA_DIR) through Inputs; the cache is target_path(data_dir, key).
    rebuild=True measures again. `inputs` stands in for the real match (the self-checks pass a synthetic bundle). The
    dict holds scalars and quantiles only; a NaN share is stored as null and read back as NaN.
    """
    if inputs is None:
        from pipeline.inputs import Inputs  # here, not above: it pulls in pandas and the starter loaders
        inputs = Inputs(data_dir=data_dir)
    data_dir = data_dir or inputs.data_dir
    key = _key(inputs)
    path = target_path(data_dir, key)
    if not rebuild:
        saved = _read(path, key)
        if saved is not None:
            return saved
    stats = line_stats(inputs.atd.ball_present, kickoff_spans(inputs), inputs.metres(inputs.atd.ball_xy), inputs.pitch)
    result = {"version": VERSION, "key": key,
              "visible_frames": stats["visible_frames"], "beyond_frames": stats["beyond_frames"],
              "beyond_line_share": _round(stats["beyond_line_share"]),
              "beyond_m_quantiles": {name: _round(value) for name, value in stats["beyond_m_quantiles"].items()}}
    _write(path, result)
    return _read(path, key) or result


def target_path(data_dir, key):
    """<data dir>/cache/benchmark/offpitch/target-<key>.json."""
    return os.path.join(data_dir, *CACHE_FOLDER, "target-{}.json".format(key))


def _key(inputs):
    """The first 16 hex digits of the sha256 of everything the numbers depend on."""
    parts = {
        "atd_sources": inputs.atd_manifest["sources"],
        "offpitch_sha256": OFFPITCH_SHA256,
        "numpy": np.__version__,
        "pitch": [float(side) for side in inputs.pitch],
        "fps": inputs.fps,
        "version": VERSION,
    }
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:KEY_DIGITS]


# The hash of this file, taken as the module is imported (calibration.CALIBRATION_SHA256's rule).
OFFPITCH_SHA256 = calibration._text_sha256(__file__)


def _round(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), DECIMALS)


def _write(path, result):
    """The JSON, written whole to a temp file in the folder and moved over the old one, so no reader sees half."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            json.dump(result, out, indent=1, sort_keys=True, allow_nan=False)
            out.write("\n")
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def _read(path, key):
    """The cached target as a dict, or None if there is none, it cannot be read, or it is not this key's and version."""
    try:
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or saved.get("key") != key or saved.get("version") != VERSION:
        return None
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description="P18's ruler: the share of visible ball frames past a line.")
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("target", help="print our match's target (aggregates only), measuring it if need be")
    cmd.add_argument("--rebuild", action="store_true", help="measure again instead of reading the cache")
    cmd.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    args = parser.parse_args(argv)
    result = target(data_dir=args.data_dir, rebuild=args.rebuild)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
