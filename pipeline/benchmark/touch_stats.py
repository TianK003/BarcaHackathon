"""touch_stats: how much of the game P04's control rule speaks about, how often it is right, the confidence it earns.

    python pipeline/benchmark/touch_stats.py [--grid stage_a|r013] [--profile P] [--seeds N] [--matches ID ...]
                                             [--data-dir D] [--out PATH]

    result = measure(DEFAULT_GRID, profile="tracking", seeds=3)      # {"settings": {label: {...}}, ...}

controlled_touch (pipeline/evidence/controlled_touch.py) says who has the ball only inside control episodes, and it
says it with a `confidence` that the source cannot know for itself: how often the team of a kept episode is the one the
truth names depends on the tracking and on the flags of the benchmark. So the confidence is measured, not tuned
(docs/proposals.md P04; Stage A of the tuning protocol): for each setting of the rule, on the degraded tracking of every
benchmark match and seed, among the alive graded frames where the served tracking has the ball,

* coverage is the share of those frames that lie inside a kept episode;
* agreement is the share of the frames inside a kept episode whose team is the truth's;
* confidence = floor(100 x pooled agreement) / 100, clipped to [0.51, 0.99] (confidence_from_agreement). It is the rate
  rounded down to a hundredth, so that the evidence never claims more than was measured, and it is a measurement on all
  seven matches, which the entry that reports it says.

Flip lags are a diagnostic of the flags' convention and not part of the rule. For each change of team in the truth with
no dead frame between (the frame before is alive, of the other team) at frame c, the lag is the number of frames from c
to the start of the new team's kept episode that begins nearest to c (the earlier on a tie). Only starts in
[max(c - 5 s, p), min(c + 5 s, n)] count, in the same graded half, where p and n are the previous and the next change of
the truth code of any kind (into or out of D included), or the half's first and last frame: the window never reaches
past the neighbouring changes, so a flip is never matched to an episode of another possession. No start in it, and the
change is not found. A negative lag means the episode began before the flag turned. It is reported as a median, a 25th
and a 75th percentile and the share of changes for which an episode was found, pooled over matches and seeds. The README
says what a flag that turns late or early would show in it.

No truth reaches the source. The source is called on a MatchInputs (inputs.py), which holds the degraded tracking and no
flag, with the confidence 0.99 and no hold, so that every kept frame is a non-NaN home value. The truth is read here,
from the match, only to count: it is never given to the source nor kept in the result, which holds counts and shares and
no frame. Matches are loaded one at a time and released, as the runner does, and the kinematics the source computes is
memoised per MatchInputs, so one MatchInputs per match and seed serves every setting.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numbers  # noqa: E402

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, kinematics  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
# Stage A's twelve settings: radius (m) outermost, then max_rel_speed (m/s), then min_frames, as the protocol does.
DEFAULT_GRID = tuple({"radius": radius, "max_rel_speed": speed, "min_frames": frames}
                     for radius in (1.0, 1.5, 2.0) for speed in (3.0, 6.0) for frames in (3, 8))
# R013's Stage A (docs/results.md): R008's edges extended, in the same order.
R013_GRID = tuple({"radius": radius, "max_rel_speed": speed, "min_frames": frames}
                  for radius in (1.5, 2.0, 2.5, 3.0) for speed in (6.0, 9.0, 12.0) for frames in (1, 2, 3, 5))
GRIDS = {"stage_a": DEFAULT_GRID, "r013": R013_GRID}    # what --grid names; stage_a is the default
KEYS = ("radius", "max_rel_speed", "min_frames")
MEASURED_CONFIDENCE = 0.99          # what the source is called with: any value in (0.5, 1], only the sign is read
LAG_SECONDS = 5                     # a flip is looked for this far either side of the change
CONFIDENCE_FLOOR, CONFIDENCE_CEILING = 0.51, 0.99
DEFAULT_PROFILE = run.DEFAULT_PROFILE


def confidence_from_agreement(agreement):
    """floor(100 x agreement) / 100, clipped to [0.51, 0.99]: the rate rounded down, never above what was measured.

    100 x agreement is rounded to 9 places before the floor, because 100 x 0.57 is 56.99999999999999 in floating point
    and would give 0.56. ValueError for anything but a real number in [0, 1] (bool and NaN included).
    """
    if isinstance(agreement, bool) or not isinstance(agreement, numbers.Real) or not 0.0 <= agreement <= 1.0:
        raise ValueError("touch_stats: agreement must be a number in [0, 1], got {!r}".format(agreement))
    floored = math.floor(round(100.0 * agreement, 9)) / 100.0
    return float(min(CONFIDENCE_CEILING, max(CONFIDENCE_FLOOR, floored)))


def label(setting):
    """A stable name for a setting: r1.0_v3.0_k3 (radius, max_rel_speed, min_frames)."""
    return "r{}_v{}_k{}".format(float(setting["radius"]), float(setting["max_rel_speed"]), int(setting["min_frames"]))


def measure(settings, *, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None, calibration=None,
            data_dir=None, log=print):
    """Coverage, agreement, flip lags and confidence of each of `settings`, per match and pooled over matches and seeds.

    `settings` is a list of {radius, max_rel_speed, min_frames} (DEFAULT_GRID is Stage A's). The other arguments are
    those of run.evaluate_configs: `profile` (how the tracking is degraded), `seeds` (0 up to seeds - 1; the clean
    profile has just seed 0), `match_ids`, `matches` ({id: DflMatch} to use instead of loading), `calibration` (read
    from the cache if none is given and the profile needs one) and `data_dir`; `log` receives a line per match.

    Returns {"profile", "seeds", "matches", "settings"}; "settings" maps label(setting) to {"setting", "pooled",
    "per_match", "flip_lag", "confidence"}. pooled and each per_match row hold frames (alive, ball visible), kept,
    coverage and agreement (None where nothing was kept or seen); flip_lag holds changes, found, share_found, median,
    p25 and p75 (None if none was found); confidence is confidence_from_agreement(pooled agreement), None if nothing
    was kept. ValueError (touch_stats) for settings that are none, miss or add a key, or repeat; the runner's
    BenchmarkError for a profile, seeds or match it refuses. All of it is refused before a match is loaded.
    """
    settings = _checked(settings)
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))

    labels = [label(setting) for setting in settings]
    counts = {name: {} for name in labels}                    # label -> match id -> [frames, kept, agree]
    lags = {name: [] for name in labels}
    changes = {name: 0 for name in labels}
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)                  # read here, to count; never handed to the source
        alive_home = codes == pm.H
        alive = alive_home | (codes == pm.A)
        flips = _flips(codes, match.graded)
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            context = RunContext(inputs=inputs, seed=0)
            visible = kinematics.ball_track(inputs).visible
            population = alive & visible
            for setting, name in zip(settings, labels):
                home = controlled_touch.compute(context, confidence=MEASURED_CONFIDENCE, **setting).home
                kept = ~np.isnan(home)
                row = counts[name].setdefault(match_id, [0, 0, 0])
                row[0] += int(population.sum())
                row[1] += int((population & kept).sum())
                row[2] += int((population & kept & ((home > 0.5) == alive_home)).sum())
                found, n_flips = _flip_lags(home, flips, match.fps, [start for _, start, _ in match.graded])
                lags[name] += found
                changes[name] += n_flips
            del atd, inputs, context
        log("match     {}   {} settings x {} seeds".format(match_id, len(settings), len(seed_list)))
        del match, codes

    out = {}
    for setting, name in zip(settings, labels):
        total = np.sum(list(counts[name].values()), axis=0)
        pooled = _row(*total)
        out[name] = {"setting": dict(setting), "pooled": pooled,
                     "per_match": {match_id: _row(*counts[name][match_id]) for match_id in sorted(counts[name])},
                     "flip_lag": _summary(lags[name], changes[name]),
                     "confidence": (None if pooled["agreement"] is None
                                    else confidence_from_agreement(pooled["agreement"]))}
    return {"profile": profile, "seeds": seed_list, "matches": sorted(ids), "settings": out}


def _checked(settings):
    """`settings` as a list of dicts with exactly KEYS and none repeated; ValueError (touch_stats) otherwise."""
    settings = list(settings)
    if not settings:
        raise ValueError("touch_stats: no setting to measure; give at least one of {}".format(", ".join(KEYS)))
    seen = set()
    for setting in settings:
        if not isinstance(setting, dict) or set(setting) != set(KEYS):
            raise ValueError("touch_stats: a setting needs exactly the keys {} and got {!r}; extra: {}, missing: {}"
                             .format(", ".join(KEYS), setting, sorted(set(setting) - set(KEYS)) if isinstance(
                                 setting, dict) else "?", sorted(set(KEYS) - set(setting)) if isinstance(
                                     setting, dict) else "?"))
        name = label(setting)
        if name in seen:
            raise ValueError("touch_stats: setting {} is given twice; it would be measured twice".format(name))
        seen.add(name)
    return [dict(setting) for setting in settings]


def _flips(codes, graded):
    """[(frame, lo, hi, new team is home)]: each change of team with no dead frame between, and its neighbours.

    lo is the previous change of the truth code of any kind (into or out of D included) in the same graded block, else
    the block's first frame; hi is the next such change, else the block's last frame. A change is a frame whose code
    differs from the one before.
    """
    flips = []
    for _, start, stop in graded:
        changes = [frame for frame in range(start + 1, stop) if codes[frame] != codes[frame - 1]]
        for number, frame in enumerate(changes):
            now, before = codes[frame], codes[frame - 1]
            if now in (pm.H, pm.A) and before in (pm.H, pm.A):
                lo = changes[number - 1] if number else start
                hi = changes[number + 1] if number + 1 < len(changes) else stop - 1
                flips.append((frame, lo, hi, bool(now == pm.H)))
    return flips


def _flip_lags(home, flips, fps, block_starts):
    """([lag, ...], number of flips): frames from each flip to the nearest episode start of the new team.

    An episode starts on a kept frame whose predecessor is not kept, or is of the other team, and always on the first
    frame of a block. For a flip at frame c the start taken is the new team's nearest to c (the earlier on a tie), among
    those in [max(c - 5 s, lo), min(c + 5 s, hi)], so the window never reaches past the truth's neighbouring changes;
    no start in it, no lag (not found).
    """
    kept = ~np.isnan(home)
    team_home = home > 0.5
    start_mask = kept.copy()
    start_mask[1:] &= ~(kept[:-1] & (team_home[:-1] == team_home[1:]))
    for first in block_starts:
        start_mask[first] = kept[first]
    starts = {True: np.flatnonzero(start_mask & team_home), False: np.flatnonzero(start_mask & ~team_home)}
    reach = LAG_SECONDS * fps
    found = []
    for frame, lo, hi, new_is_home in flips:
        candidates = starts[new_is_home]
        first = np.searchsorted(candidates, max(frame - reach, lo), side="left")
        last = np.searchsorted(candidates, min(frame + reach, hi), side="right")
        if first < last:
            inside = candidates[first:last]
            found.append(int(inside[np.argmin(np.abs(inside - frame))]) - frame)   # argmin: the earlier on a tie
    return found, len(flips)


def _row(frames, kept, agree):
    frames, kept, agree = int(frames), int(kept), int(agree)
    return {"frames": frames, "kept": kept, "coverage": kept / frames if frames else None,
            "agreement": agree / kept if kept else None}


def _summary(lags, changes):
    if not lags:
        return {"changes": changes, "found": 0, "share_found": 0.0 if changes else None, "median": None, "p25": None,
                "p75": None}
    p25, median, p75 = (float(value) for value in np.percentile(lags, [25, 50, 75]))
    return {"changes": changes, "found": len(lags), "share_found": len(lags) / changes, "median": median,
            "p25": p25, "p75": p75}


# -- the table --------------------------------------------------------------------------------------------------------

def render(result):
    """The table's lines: one per setting, aggregates only, then what the columns are."""
    lines = ["{:<16}{:>9}{:>11}{:>17}{:>9}{:>9}{:>9}{:>8}{:>12}".format(
        "setting", "coverage", "agreement", "per-match agr.", "lag med", "lag p25", "lag p75", "found", "confidence")]
    for name, entry in result["settings"].items():
        agreements = [row["agreement"] for row in entry["per_match"].values() if row["agreement"] is not None]
        flip = entry["flip_lag"]
        lines.append("{:<16}{:>9}{:>11}{:>17}{:>9}{:>9}{:>9}{:>8}{:>12}".format(
            name, _pct(entry["pooled"]["coverage"]), _pct(entry["pooled"]["agreement"]),
            "{}-{}".format(_pct(min(agreements)), _pct(max(agreements))) if agreements else "n/a",
            _frames(flip["median"]), _frames(flip["p25"]), _frames(flip["p75"]), _pct(flip["share_found"]),
            "n/a" if entry["confidence"] is None else "{:.2f}".format(entry["confidence"])))
    lines.append("coverage and agreement among alive graded frames with the ball visible; lags in frames, pooled over "
                 "matches {} and seeds {}".format(",".join(result["matches"]), result["seeds"]))
    return lines


def _pct(value):
    return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def _frames(value):
    return "n/a" if value is None else "{:.0f}".format(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure P04's control rule on the benchmark: coverage, agreement with the truth, flip lags, and "
                    "the confidence each setting of a grid earns (floor of the pooled agreement to a hundredth).")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="stage_a",
                        help="the settings: stage_a, R008's 12 (default), or r013, R013's 48")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--out", help="also write the result as JSON here (counts and shares, no frame)")
    args = parser.parse_args(argv)
    try:
        result = measure(GRIDS[args.grid], profile=args.profile, seeds=args.seeds, match_ids=args.matches,
                         data_dir=args.data_dir)
        if args.out:
            run.write_json(args.out, run.plain(result))
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("")
    for line in render(result):
        print(line)
    if args.out:
        print("written   {}".format(run.shown(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
