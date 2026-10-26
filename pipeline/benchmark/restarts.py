"""restarts: how the stoppages end, on the benchmark, and what the predicted stoppages of BAR-ATH look like to the ball.

    python pipeline/benchmark/restarts.py anatomy CONFIG [--profile P] [--seeds N] [--truth raw|coarse]
                                                         [--matches ID ...] [--data-dir D] [--out PATH]
    python pipeline/benchmark/restarts.py bar-ath CONFIG [--data-dir D] [--out PATH]

    result = anatomy("pipeline/configs/main.toml", profile="tracking", seeds=3, truth="raw")

A possession that follows a dead ball starts at the restart, and which team restarts is not always the team that had the
ball before it: after a ball out of play, a foul or a goal the other team often takes it. A decoder that carries the
team across a stoppage (the hold of the HMM, the majority vote's smoothing) is then wrong at the start of the next
possession, and spellF1 pays for it as a possession of the wrong team (docs/proposals.md P17). Before a channel that
names the restarting team is built, these measures say how big the thing it would fix is, and how much of it the
pipeline already gets.

Anatomy (the benchmark, truth read only to count). A restart is a maximal run of D inside a graded block that is followed,
inside the block, by H or A (true_restarts); a run that ends the block or meets an X is no restart. For each:

* the stoppage's length (median_stoppage_s, restart - start over fps): how long the pipeline has to bridge;
* flip_share: among the restarts whose frame before the run is H or A inside the block (before != -1), the share where
  the restarting team (after) is not that team. It is the part of the problem that carrying the team across the
  stoppage gets wrong by construction, and it is what a restart-team channel could earn;
* the outcome in the OUTCOME_WINDOW frames after the restart (outcomes), 0 right (more frames of the restarting team than
  of the other in the pipeline's labels), 1 wrong (some H/A, and not right; a tie is wrong, as the first possession
  that matters is the one with the majority), 2 still D (no H/A at all: the stoppage runs on). The window stops at the
  block's end. Right, wrong and still-D are reported as shares of all restarts and again split by flip and no flip, which
  shows whether the pipeline's mistakes sit on the flips (carried team) or on the no-flips (noise).

The pipeline's labels are made as the benchmark makes them: the tracking is degraded under the profile and seed
(run.make_atd), served as a MatchInputs with no flag of the truth, and decoded by run.predict; the result pools over
matches and seeds by sums (restarts counted once per seed, the shares those of the sums), as touch_stats does. The truth
is read here, from the match, only to find the restarts: it is never handed to the pipeline, nor kept in the result,
which holds counts and shares and no frame. Matches are loaded one at a time and released.

BAR-ATH (label-free, aggregates only). There is no truth for our match, only the pipeline's own predicted stoppages.
stoppage_ball_shares counts those followed by H or A and, for each, whether the ball is missing on more than half of the
BALL_WINDOW frames before the stoppage (no_ball_before) and after it (no_ball_after). A channel that reads the ball's
last or first position around a restart can only work where the ball is seen there; on the benchmark's `tracking`
profile the ball gaps are a transplant of ours, and these shares are the match against which that profile can be held.
Only counts and shares leave bar_ath: no frame, no label sequence. A stoppage that opens its block has no frames before it
and is left out of no_ball_before's denominator, and one that is followed by nothing in its block is no stoppage here.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path: swap pipeline/benchmark/ for the repository root, as run.py does, so that `import pipeline` resolves
    # from any working directory and no file in here can shadow a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import dfl, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.evidence import kinematics  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
OUTCOME_WINDOW = 50         # frames after a restart in which the pipeline's team is read (2 s at 25 fps)
BALL_WINDOW = 50            # frames before and after a predicted stoppage in which the ball's visibility is read
RIGHT, WRONG, STILL_D = 0, 1, 2     # the outcomes of a restart
OUTCOME_NAMES = ("right", "wrong", "still_d")
DEFAULT_PROFILE = run.DEFAULT_PROFILE


def _stoppages(codes, graded):
    """[(half, block start, block stop, run start, restart)]: each run of D followed by H or A inside its block."""
    found = []
    for half, start, stop in graded:
        block = np.asarray(codes[start:stop])
        dead = block == pm.D
        edges = np.diff(np.concatenate(([False], dead, [False])).astype(np.int8))
        for first, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end < len(block) and block[end] in (pm.H, pm.A):
                found.append((int(half), int(start), int(stop), int(start + first), int(start + end)))
    return found


def true_restarts(codes, graded):
    """{"start", "restart", "before", "after"}: int64 arrays, one entry per D run of a block that H or A follows.

    `start` is the run's first frame and `restart` the first frame after it, whose code is `after`. `before` is the code
    of the frame before the run when that frame is in the block and is H or A, else -1 (the run opens the block, or
    follows an X). `graded` is ((half, start, stop), ...), half-open rows.
    """
    rows = _stoppages(codes, graded)
    start = np.array([row[3] for row in rows], dtype=np.int64)
    restart = np.array([row[4] for row in rows], dtype=np.int64)
    before = np.full(len(rows), -1, dtype=np.int64)
    for number, (_, block_start, _, first, _) in enumerate(rows):
        if first > block_start and codes[first - 1] in (pm.H, pm.A):
            before[number] = codes[first - 1]
    after = np.array([codes[frame] for frame in restart], dtype=np.int64)
    return {"start": start, "restart": restart, "before": before, "after": after}


def outcomes(predicted, restarts, graded, window=OUTCOME_WINDOW):
    """int8 per restart: RIGHT, WRONG or STILL_D in `predicted[e : min(e + window, stop)]`, e the restart, stop its block's.

    Right: more frames of the restarting team (`after`) than of the other team. Wrong: at least one H/A frame, and not
    right (a tie included). Still D: no H or A frame (D and X only).
    """
    ends = [(int(start), int(stop)) for _, start, stop in graded]
    out = np.zeros(len(restarts["restart"]), dtype=np.int8)
    for number, (frame, team) in enumerate(zip(restarts["restart"], restarts["after"])):
        stop = next(end for first, end in ends if first <= frame < end)
        seen = np.asarray(predicted[frame:min(frame + window, stop)])
        mine, other = int((seen == team).sum()), int(((seen == pm.H) | (seen == pm.A)).sum()) - int((seen == team).sum())
        out[number] = RIGHT if mine > other else (WRONG if mine + other else STILL_D)
    return out


def anatomy(config_path, *, profile=DEFAULT_PROFILE, seeds=None, truth="raw", match_ids=None, matches=None,
            calibration=None, data_dir=None, log=print):
    """How the truth's stoppages end and how often the config names the restarting team, per match and pooled.

    `config_path` is a pipeline config (scope = "halves"); the other arguments are those of touch_stats.measure: `profile`,
    `seeds` (0 up to seeds - 1; the clean profile has just seed 0), `truth` ("raw" or "coarse"), `match_ids`, `matches`
    ({id: DflMatch} to use instead of loading), `calibration` and `data_dir`; `log` receives a line per match.

    Returns {"profile", "seeds", "truth", "matches", "pooled", "per_match"}; pooled and each per-match row hold
    restarts (counted once per seed), median_stoppage_s, flip_share (None where no restart has a team before it), the
    shares right, wrong and still_d of all restarts, and "flip" and "no_flip": {restarts, right, wrong, still_d} of the
    restarts whose team before the stoppage differs from (flip) or is the restarting team (no_flip); a share is None
    where its denominator is 0. Counts and shares only, no frame. The runner's BenchmarkError for a profile, seeds, truth
    or match it refuses, before a match is loaded.
    """
    run._check_profile(profile)
    if truth not in run.TRUTHS:
        raise run.BenchmarkError("truth {!r} is not one of: {}".format(truth, ", ".join(run.TRUTHS)))
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    config = load_config(config_path)
    run.check_scope(config)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))

    pieces = {}                                              # match id -> (durations, before, after, outcomes)
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)                 # read here, to count; never handed to the pipeline
        if truth == "coarse":
            codes = truths.coarse_codes(codes, match.graded, match.fps)
        found = true_restarts(codes, match.graded)
        durations = (found["restart"] - found["start"]) / match.fps
        results = []
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            predicted, _ = run.predict(config, inputs)
            results.append(outcomes(predicted, found, match.graded))
            del atd, inputs, predicted
        pieces[match_id] = (np.tile(durations, len(seed_list)), np.tile(found["before"], len(seed_list)),
                            np.tile(found["after"], len(seed_list)), np.concatenate(results))
        log("match     {}   {} restarts x {} seeds".format(match_id, len(found["start"]), len(seed_list)))
        del match, codes, found

    pooled = _row(*(np.concatenate([pieces[match_id][part] for match_id in sorted(pieces)]) for part in range(4)))
    return {"profile": profile, "seeds": seed_list, "truth": truth, "matches": sorted(ids), "pooled": pooled,
            "per_match": {match_id: _row(*pieces[match_id]) for match_id in sorted(pieces)}}


def _row(durations, before, after, outcome):
    known = before != -1
    flip = known & (after != before)
    no_flip = known & (after == before)
    row = {"restarts": int(len(outcome)),
           "median_stoppage_s": float(np.median(durations)) if len(durations) else None,
           "flip_share": float(flip.sum() / known.sum()) if known.any() else None}
    row.update(_shares(outcome))
    row["flip"] = dict(restarts=int(flip.sum()), **_shares(outcome[flip]))
    row["no_flip"] = dict(restarts=int(no_flip.sum()), **_shares(outcome[no_flip]))
    return row


def _shares(outcome):
    counts = np.bincount(outcome, minlength=3)
    return {name: (float(count / len(outcome)) if len(outcome) else None) for name, count in zip(OUTCOME_NAMES, counts)}


def stoppage_ball_shares(codes, ball_visible, graded, window=BALL_WINDOW):
    """Counts and shares over the predicted D runs that H or A follows: how often the ball is missing around them.

    Returns {"stoppages", "per_half" ({half: count}), "no_ball_before", "no_ball_after"}: the last two are the shares of
    the stoppages whose `window` frames before the run (inside the block), or after the restart (up to the block's end),
    have the ball missing on more than half of them. A stoppage with no frame on that side is left out of that share's
    denominator; a share is None where nothing is left. Nothing but counts and shares comes out.
    """
    missing = ~np.asarray(ball_visible, dtype=bool)
    per_half, before, after = {}, [], []
    for half, block_start, block_stop, first, restart in _stoppages(codes, graded):
        per_half[half] = per_half.get(half, 0) + 1
        for side, frames in ((before, missing[max(first - window, block_start):first]),
                             (after, missing[restart:min(restart + window, block_stop)])):
            if len(frames):
                side.append(bool(2 * frames.sum() > len(frames)))
    return {"stoppages": int(sum(per_half.values())), "per_half": per_half,
            "no_ball_before": float(np.mean(before)) if before else None,
            "no_ball_after": float(np.mean(after)) if after else None}


def bar_ath(config_path, data_dir=None):
    """stoppage_ball_shares of the config's labels on BAR-ATH and of its ball's visibility. Aggregates only."""
    from pipeline.inputs import Inputs
    config = load_config(config_path)
    inputs = Inputs(data_dir=data_dir)
    codes, _ = run.predict(config, inputs)
    return stoppage_ball_shares(codes, kinematics.ball_track(inputs).visible, inputs.graded)


# -- the table --------------------------------------------------------------------------------------------------------

def render(result):
    """The table's lines of an anatomy: one per match and the pool, aggregates only, then what the columns are."""
    head = "{:<12}{:>9}{:>9}{:>8}{:>8}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}".format(
        "match", "restarts", "med s", "flip", "right", "wrong", "still D", "right|flip", "wrong|flip", "right|same",
        "wrong|same")
    lines = [head]
    for name, row in list(result["per_match"].items()) + [("pooled", result["pooled"])]:
        lines.append("{:<12}{:>9}{:>9}{:>8}{:>8}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}".format(
            name, row["restarts"], "n/a" if row["median_stoppage_s"] is None else "{:.1f}".format(
                row["median_stoppage_s"]), _pct(row["flip_share"]), _pct(row["right"]), _pct(row["wrong"]),
            _pct(row["still_d"]), _pct(row["flip"]["right"]), _pct(row["flip"]["wrong"]),
            _pct(row["no_flip"]["right"]), _pct(row["no_flip"]["wrong"])))
    lines.append("restarts: a D run followed by H/A in a graded half, truth {}, pooled over matches {} and seeds {}; "
                 "flip: the restarting team is not the team before the stoppage; right/wrong/still D: the pipeline's "
                 "labels in the {} frames after the restart (flip|same: split by flip or not)".format(
                     result["truth"], ",".join(result["matches"]), result["seeds"], OUTCOME_WINDOW))
    return lines


def render_bar_ath(shares):
    return ["stoppages     {}".format(shares["stoppages"]),
            "per half      {}".format(", ".join("{}: {}".format(half, count)
                                                for half, count in sorted(shares["per_half"].items())) or "none"),
            "no ball before the stoppage  {}   (more than half of the last {} frames)".format(
                _pct(shares["no_ball_before"]), BALL_WINDOW),
            "no ball after the restart    {}   (more than half of the first {} frames)".format(
                _pct(shares["no_ball_after"]), BALL_WINDOW)]


def _pct(value):
    return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The anatomy of the stoppages: how they end on the benchmark's truth and what the pipeline makes of "
                    "the restart, and the ball around the predicted stoppages of BAR-ATH (aggregates only).")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("anatomy", help="the benchmark: restarts of the truth against the pipeline's labels")
    one.add_argument("config", metavar="CONFIG", help="a pipeline config (scope = \"halves\")")
    one.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
    one.add_argument("--seeds", type=int, help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    one.add_argument("--truth", choices=run.TRUTHS, default="raw", help="the flags' truth: raw (default) or coarse")
    one.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    two = commands.add_parser("bar-ath", help="our match: the ball around the predicted stoppages, no truth")
    two.add_argument("config", metavar="CONFIG", help="a pipeline config")
    for command in (one, two):
        command.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        command.add_argument("--out", help="also write the result as JSON here (counts and shares; folders are made)")
    args = parser.parse_args(argv)
    try:
        if args.command == "anatomy":
            result = anatomy(args.config, profile=args.profile, seeds=args.seeds, truth=args.truth,
                             match_ids=args.matches, data_dir=args.data_dir)
            lines = render(result)
        else:
            result = bar_ath(args.config, data_dir=args.data_dir)
            lines = render_bar_ath(result)
        if args.out:
            run.write_json(args.out, run.plain(result))
    except (ConfigError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("")
    for line in lines:
        print(line)
    if args.out:
        print("written   {}".format(run.shown(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
