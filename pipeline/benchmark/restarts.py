"""restarts: how the stoppages end, on the benchmark, and what the predicted stoppages of BAR-ATH look like to the ball.

    python pipeline/benchmark/restarts.py anatomy CONFIG [--profile P] [--seeds N] [--truth raw|coarse]
                                                         [--matches ID ...] [--data-dir D] [--out PATH]
    python pipeline/benchmark/restarts.py bar-ath CONFIG [--data-dir D] [--out PATH]
    python pipeline/benchmark/restarts.py stats [--model M] [--profile P] [--seeds N] [--matches ID ...]
                                                [--data-dir D] [--out PATH]
    python pipeline/benchmark/restarts.py configs STATS_JSON --out-dir DIR [--main pipeline/configs/main.toml]
    python pipeline/benchmark/restarts.py events [--model M] [--tail T] [--use-ball] [--profile P] [--seeds N]
                                                 [--matches ID ...] [--data-dir D] [--out PATH]

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

Stats (P17's confidence per type, measured on the benchmark). pipeline/evidence/restart_team.py names, after each
predicted stoppage, the type of restart and the team that takes it, and votes `home` for that team on [restart, end). How
much a vote of a type is worth cannot be known by the source: it is measured here, as touch_stats measures
controlled_touch's, on the degraded tracking of every benchmark match and seed, for each setting of the grid (GRID: the
vote's `tail` and `use_ball`). Per type, pooled over matches and seeds by sums:

* frames: the voted frames, those of [restart, end) of every call of the type, whose truth code is H or A (a D frame in
  the span is not counted); agree: those whose truth is the team the call names; agreement = agree / frames;
* stoppages_per_seed: the calls of the type, summed over matches and seeds, over the number of seeds;
* coverage: the share of the truth's restarts (true_restarts, pooled) that have a call of this type with its restart
  within COVERAGE_WINDOW frames of the true one (coverage_all: of any type);
* confidence: None (the type does not vote) if agreement < MIN_AGREEMENT or stoppages_per_seed < MIN_STOPPAGES_PER_SEED,
  declared in the design before any measurement; otherwise touch_stats.confidence_from_agreement(agreement), the rate
  rounded down to a hundredth. Nothing is tuned: the grid is read, and the rule is applied.

The truth reaches only the counting here, as in anatomy, never the source, which is called on a MatchInputs; the result
holds counts and shares of DFL matches, nothing of BAR-ATH. Task 10 reads settings[label]["types"][type]["confidence"].

Events (a diagnostic; it chooses nothing). For one setting, each DFL restart event (dfl_events.load_restarts: KickOff,
ThrowIn, CornerKick, GoalKick, FreeKick, Penalty) is matched to the nearest call whose restart is within EVENT_WINDOW
frames (the earlier on a tie): recall per kind is the share of its events with a call of any type, and the confusion
table counts the matched pairs by the call's type and the event's kind. Pooled over matches and seeds.
"""

import argparse
import json
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
from pipeline.benchmark import dfl_events, touch_stats  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import kinematics, restart_team  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
OUTCOME_WINDOW = 50         # frames after a restart in which the pipeline's team is read (2 s at 25 fps)
BALL_WINDOW = 50            # frames before and after a predicted stoppage in which the ball's visibility is read
RIGHT, WRONG, STILL_D = 0, 1, 2     # the outcomes of a restart
OUTCOME_NAMES = ("right", "wrong", "still_d")
DEFAULT_PROFILE = run.DEFAULT_PROFILE
# stats: the settings measured, and the rule that drops a type (declared in the design, before any measurement)
GRID = tuple({"tail": tail, "use_ball": ball} for ball in (False, True) for tail in (25, 50, 100, 150))
MIN_AGREEMENT = 0.60                # a type whose votes agree with the truth less often than this does not vote
MIN_STOPPAGES_PER_SEED = 30         # nor one with fewer calls than this per seed (mean over the seeds)
COVERAGE_WINDOW = 50                # a call covers a true restart if its restart is this close (2 s)
EVENT_WINDOW = 50                   # events: an event is matched to a call whose restart is this close
DEFAULT_MODEL = "pipeline/models/dead_ball_hgb.json"   # as main.toml writes it


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


# -- stats: the confidence of each type's vote ------------------------------------------------------------------------

def label(setting):
    """A stable name for a setting: t050_b0 (tail, use_ball as 0 or 1)."""
    return "t{:03d}_b{}".format(int(setting["tail"]), int(bool(setting["use_ball"])))


def confidence_of(agreement, stoppages_per_seed):
    """The type's confidence under the drop rule: None under MIN_AGREEMENT or MIN_STOPPAGES_PER_SEED, else the floor."""
    if agreement is None or agreement < MIN_AGREEMENT or stoppages_per_seed < MIN_STOPPAGES_PER_SEED:
        return None
    return touch_stats.confidence_from_agreement(agreement)


def _checked_settings(settings):
    settings = [dict(setting) for setting in settings]
    if not settings:
        raise ValueError("restarts: no setting to measure; give at least one of tail, use_ball")
    names = set()
    for setting in settings:
        if set(setting) != {"tail", "use_ball"}:
            raise ValueError("restarts: a setting needs exactly the keys tail and use_ball, got {!r}".format(setting))
        if label(setting) in names:
            raise ValueError("restarts: setting {} is given twice".format(label(setting)))
        names.add(label(setting))
    return settings


def _seed_list(profile, seeds):
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    return [0] if profile == run.CLEAN else list(range(seeds))


def _span_counts(codes, calls, n_types):
    """(frames, agree) per type: the voted frames of each call whose truth is H or A, and those it names right."""
    frames, agree = np.zeros(n_types, dtype=np.int64), np.zeros(n_types, dtype=np.int64)
    for restart, end, kind, team in zip(calls["restart"].tolist(), calls["end"].tolist(), calls["type"].tolist(),
                                        calls["team"].tolist()):
        span = np.asarray(codes[restart:end])
        frames[kind] += int(((span == pm.H) | (span == pm.A)).sum())
        agree[kind] += int((span == (pm.H if team == 0 else pm.A)).sum())
    return frames, agree


def _covered(restart, calls, n_types, window=COVERAGE_WINDOW):
    """(per type, any): the true restarts with a call of that type, or of any, within `window` frames."""
    per_type, anyone = np.zeros(n_types, dtype=np.int64), 0
    called = np.asarray(calls["restart"])
    for frame in np.asarray(restart).tolist():
        near = np.abs(called - frame) <= window
        anyone += int(near.any())
        per_type += np.isin(np.arange(n_types), np.asarray(calls["type"])[near]).astype(np.int64)
    return per_type, anyone


def measure_restarts(settings, *, model, profile=DEFAULT_PROFILE, seeds=None, match_ids=None, matches=None,
                     calibration=None, data_dir=None, log=print):
    """Agreement, coverage and confidence per type of restart_team's calls, for each of `settings`, on the benchmark.

    `settings` is a list of {tail, use_ball} (GRID is the design's); `model` the dead-ball model the source reads. The
    other arguments are touch_stats.measure's. Returns {"profile", "seeds", "matches", "settings"}; "settings" maps
    label(setting) to {"setting", "coverage_all", "types": {type: {frames, agree, agreement, stoppages_per_seed,
    coverage, confidence}}}, pooled over matches and seeds by sums (agreement and coverage are None where nothing
    was voted or no restart exists). Counts and shares of the benchmark only. The source is called once per match, seed
    and setting, on a MatchInputs with no flag of the truth (the dead-ball model's P(alive) is memoised per MatchInputs,
    so the trees are walked once per match and seed).
    """
    settings = _checked_settings(settings)
    seed_list = _seed_list(profile, seeds)
    ids = run.resolve_match_ids(match_ids, matches)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    names, n_types = [label(setting) for setting in settings], len(restart_team.TYPES)
    frames = {name: np.zeros(n_types, dtype=np.int64) for name in names}
    agree = {name: np.zeros(n_types, dtype=np.int64) for name in names}
    stops = {name: np.zeros(n_types, dtype=np.int64) for name in names}
    covered = {name: np.zeros(n_types, dtype=np.int64) for name in names}
    covered_any = dict.fromkeys(names, 0)
    owed = 0
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)                 # read here, to count; never handed to the source
        found = true_restarts(codes, match.graded)
        owed += len(found["restart"]) * len(seed_list)
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            context = RunContext(inputs=inputs, seed=0)
            for setting, name in zip(settings, names):
                made = restart_team.calls(context, model=model, tail=setting["tail"], use_ball=setting["use_ball"])
                here_frames, here_agree = _span_counts(codes, made, n_types)
                frames[name] += here_frames
                agree[name] += here_agree
                stops[name] += np.bincount(np.asarray(made["type"], dtype=np.int64), minlength=n_types)
                here_covered, here_any = _covered(found["restart"], made, n_types)
                covered[name] += here_covered
                covered_any[name] += here_any
            del atd, inputs, context
        log("match     {}   {} settings x {} seeds".format(match_id, len(settings), len(seed_list)))
        del match, codes, found

    out = {}
    for setting, name in zip(settings, names):
        types = {}
        for code, kind in enumerate(restart_team.TYPES):
            n, a = int(frames[name][code]), int(agree[name][code])
            agreement = a / n if n else None
            per_seed = int(stops[name][code]) / len(seed_list)
            types[kind] = {"frames": n, "agree": a, "agreement": agreement, "stoppages_per_seed": per_seed,
                           "coverage": int(covered[name][code]) / owed if owed else None,
                           "confidence": confidence_of(agreement, per_seed)}
        out[name] = {"setting": dict(setting), "coverage_all": covered_any[name] / owed if owed else None,
                     "types": types}
    return {"profile": profile, "seeds": seed_list, "matches": sorted(ids), "settings": out}


# -- events: the calls against DFL's own restart events (a diagnostic) ----------------------------------------------------

def event_table(calls, events, window=EVENT_WINDOW):
    """{"events", "matched", "confusion"} of one match and seed: each event matched to the nearest call within `window`.

    `events` is dfl_events.load_restarts' dict (row, kind). events and matched map each kind of dfl_events.KINDS to a
    count (the events, and those with a call within `window` of their row, the earlier on a tie); confusion maps each
    call type to {kind: pairs}. The calls are in time order, as restart_team.calls gives them.
    """
    kinds, types = dfl_events.KINDS, restart_team.TYPES
    table = {"events": dict.fromkeys(kinds, 0), "matched": dict.fromkeys(kinds, 0),
             "confusion": {kind: dict.fromkeys(kinds, 0) for kind in types}}
    called = np.asarray(calls["restart"])
    for row, kind in zip(np.asarray(events["row"]).tolist(), np.asarray(events["kind"]).tolist()):
        table["events"][kind] += 1
        if not len(called):
            continue
        nearest = int(np.argmin(np.abs(called - row)))          # argmin: the earlier call on a tie
        if abs(int(called[nearest]) - row) <= window:
            table["matched"][kind] += 1
            table["confusion"][types[int(calls["type"][nearest])]][kind] += 1
    return table


def events_diagnostic(*, model, tail=50, use_ball=False, profile=DEFAULT_PROFILE, seeds=None, match_ids=None,
                      matches=None, events=None, calibration=None, data_dir=None, log=print):
    """Per DFL kind the recall of the calls (events within EVENT_WINDOW of a call) and the call-type by kind confusion.

    `events` ({id: dfl_events.load_restarts' dict}) is used instead of loading when given. Returns {"profile", "seeds",
    "matches", "setting", "kinds": {kind: {events, matched, recall}}, "confusion"}, pooled over matches and seeds by
    sums. It chooses nothing: no constant is read from it.
    """
    setting = _checked_settings([{"tail": tail, "use_ball": use_ball}])[0]
    seed_list = _seed_list(profile, seeds)
    ids = run.resolve_match_ids(match_ids, matches)
    if profile != run.CLEAN and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    kinds, types = dfl_events.KINDS, restart_team.TYPES
    total = {"events": dict.fromkeys(kinds, 0), "matched": dict.fromkeys(kinds, 0),
             "confusion": {kind: dict.fromkeys(kinds, 0) for kind in types}}
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        here = events[match_id] if events is not None else dfl_events.load_restarts(match_id, data_dir=data_dir)
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            made = restart_team.calls(RunContext(inputs=inputs, seed=0), model=model, tail=setting["tail"],
                                      use_ball=setting["use_ball"])
            one = event_table(made, here)
            for part in ("events", "matched"):
                for kind in kinds:
                    total[part][kind] += one[part][kind]
            for kind in types:
                for event_kind in kinds:
                    total["confusion"][kind][event_kind] += one["confusion"][kind][event_kind]
            del atd, inputs
        log("match     {}   {} events x {} seeds".format(match_id, len(here["row"]), len(seed_list)))
        del match, here
    return {"profile": profile, "seeds": seed_list, "matches": sorted(ids), "setting": setting,
            "kinds": {kind: {"events": total["events"][kind], "matched": total["matched"][kind],
                             "recall": (total["matched"][kind] / total["events"][kind]) if total["events"][kind]
                             else None} for kind in kinds},
            "confusion": total["confusion"]}


# -- the table --------------------------------------------------------------------------------------------------------

# -- configs: R014's candidates, main.toml plus the measured restart vote ---------------------------------------------

def _config_block(setting, types, model, eol, span="after"):
    """The [[evidence]] block of one setting: the source, its model and setting, and one key per voting type."""
    use_ball = "true" if setting["use_ball"] else "false"
    if span == "after":
        lines = ["# R014 (P17): who restarts after a stoppage, tail {}, use_ball {}, the measured confidences (restarts.py "
                 "stats)".format(setting["tail"], use_ball),
                 "[[evidence]]", 'kind = "restart_team"', 'model = "{}"'.format(model),
                 "tail = {}".format(setting["tail"]), "use_ball = {}".format(use_ball)]
    else:
        lines = ["# R014b (P17): who restarts, read on the stoppage's last frame (span {}), use_ball {}, the confidences "
                 "measured at tail {} (restarts.py stats)".format(span, use_ball, setting["tail"]),
                 "[[evidence]]", 'kind = "restart_team"', 'model = "{}"'.format(model), 'span = "{}"'.format(span),
                 "use_ball = {}".format(use_ball)]
    lines += ["{} = {!r}".format(name, float(types[name]["confidence"])) for name in restart_team.TYPES
              if types.get(name, {}).get("confidence") is not None]
    return eol.join(lines) + eol + eol


def _toml_float(value):
    return "inf" if value == float("inf") else repr(float(value))


def _switch_name(value):
    return "in" if value == float("inf") else "{:02d}".format(int(value))


def write_configs(stats, main_path, folder, model=DEFAULT_MODEL, log=print, span="after", switch_restart=None,
                  label_from=None):
    """One config per measured setting: main.toml's text, a restart_team block before [decoder], the description amended.

    `stats` is the JSON of `stats --out` (settings[label]: "setting" and "types", each type's "confidence" a float, or
    None when it does not vote). The text is copied verbatim; the block goes immediately before [decoder] (and before the
    comment lines that head it), and the description gains the source and the setting. A setting with no confidence at
    all would add a source that never votes: it is skipped and reported. Returns the paths written, in label order.

    span "after" (default) is R014: one file `r014_<label>.toml` per setting, the block carrying its `tail`. span
    "stoppage_end" is R014b: the block carries `span = "stoppage_end"` and no tail, and the confidences of each use_ball
    come from the label `t<label_from:03d>_b<use_ball>` (label_from, required, an int tail such as 25). With
    `switch_restart` (a list of floats, inf allowed; stoppage_end only), the [decoder] table also gains
    `restart_source = "restart_team"` and `switch_restart = X`, right after its last key line, and one file
    `r014b_s<SS>_b<0|1>.toml` is written per (switch_restart, use_ball), SS being 04, 08, 16 or `in` for inf.
    Without switch_restart, stoppage_end writes `r014b_b<0|1>.toml`.
    """
    if span not in ("after", "stoppage_end"):
        raise ValueError("restarts: span must be 'after' or 'stoppage_end', not {!r}".format(span))
    if span == "after" and (switch_restart or label_from is not None):
        raise ValueError("restarts: switch_restart and label_from belong to span 'stoppage_end'")
    if span == "stoppage_end" and label_from is None:
        raise ValueError("restarts: span 'stoppage_end' needs label_from, the tail whose confidences it reuses")
    with open(main_path, newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    decoders = [i for i, line in enumerate(lines) if line.strip() == "[decoder]"]
    descriptions = [i for i, line in enumerate(lines) if line.startswith("description = ")]
    if len(decoders) != 1:
        raise ValueError("restarts: {} has {} [decoder] lines, want exactly 1".format(main_path, len(decoders)))
    if len(descriptions) != 1:
        raise ValueError("restarts: {} has {} 'description = ' lines, want exactly 1".format(
            main_path, len(descriptions)))
    eol = "\r\n" if lines[decoders[0]].endswith("\r\n") else "\n"   # the block is written as the file is
    at = decoders[0]
    while at > 0 and lines[at - 1].startswith("#"):
        at -= 1
    last_key = None                                      # the decoder's last key line, for the restart keys
    for i in range(decoders[0] + 1, len(lines)):
        text = lines[i].strip()
        if not text or text.startswith("["):
            break
        if not text.startswith("#"):
            last_key = i
    if switch_restart and last_key is None:
        raise ValueError("restarts: {} has no key line in [decoder]".format(main_path))
    cut = descriptions[0]
    body = lines[cut].rstrip("\r\n")
    if not body.endswith('"'):
        raise ValueError("restarts: the description line of {} does not end with a quote".format(main_path))
    os.makedirs(folder, exist_ok=True)
    def make(entry, file_name, note, switch=None):
        amended = list(lines)
        amended[cut] = body[:-1] + ' + restart_team ({}, measured confidences)"'.format(note) + lines[cut][len(body):]
        if switch is not None:
            if not lines[last_key].endswith(("\n", "\r")):
                raise ValueError("restarts: the last [decoder] line of {} has no line ending".format(main_path))
            amended[last_key] += eol.join(('restart_source = "restart_team"',
                                           "switch_restart = {}".format(_toml_float(switch)))) + eol
        amended.insert(at, _config_block(entry["setting"], entry["types"], model, eol, span))
        path = os.path.join(folder, file_name)
        with open(path, "w", newline="") as handle:
            handle.write("".join(amended))
        return path

    def silent(entry):
        return all(entry["types"].get(kind, {}).get("confidence") is None for kind in restart_team.TYPES)

    written = []
    if span == "after":
        for name in sorted(stats["settings"]):
            entry = stats["settings"][name]
            if silent(entry):
                log("skipped   {}: no type has a confidence, so the source would never vote".format(name))
                continue
            note = "tail {}, use_ball {}".format(entry["setting"]["tail"],
                                                 "true" if entry["setting"]["use_ball"] else "false")
            written.append(make(entry, "r014_{}.toml".format(name), note))
        return written
    for switch in (switch_restart or [None]):
        for use_ball in (0, 1):
            name = "t{:03d}_b{}".format(label_from, use_ball)
            entry = stats["settings"][name]
            if silent(entry):
                log("skipped   {}: no type has a confidence, so the source would never vote".format(name))
                continue
            note = "span stoppage_end, use_ball {}".format("true" if use_ball else "false")
            if switch is None:
                file_name = "r014b_b{}.toml".format(use_ball)
            else:
                note += ", switch_restart {}".format(_toml_float(switch))
                file_name = "r014b_s{}_b{}.toml".format(_switch_name(switch), use_ball)
            written.append(make(entry, file_name, note, switch))
    return written


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


def render_stats(result):
    """One block per setting: a line per type with its calls per seed, coverage, agreement and confidence."""
    lines = []
    for name, entry in result["settings"].items():
        lines.append("{}   (coverage of all true restarts {})".format(name, _pct(entry["coverage_all"])))
        lines.append("  {:<10}{:>12}{:>10}{:>11}{:>12}".format("type", "calls/seed", "coverage", "agreement",
                                                               "confidence"))
        for kind, row in entry["types"].items():
            lines.append("  {:<10}{:>12.1f}{:>10}{:>11}{:>12}".format(
                kind, row["stoppages_per_seed"], _pct(row["coverage"]), _pct(row["agreement"]),
                "dropped" if row["confidence"] is None else "{:.2f}".format(row["confidence"])))
    lines.append("agreement: voted frames whose truth is the called team (H/A only); coverage: true restarts with a call "
                 "of the type within {} frames; dropped: agreement < {} or fewer than {} calls per seed; matches {}, "
                 "seeds {}".format(COVERAGE_WINDOW, MIN_AGREEMENT, MIN_STOPPAGES_PER_SEED, ",".join(result["matches"]),
                                   result["seeds"]))
    return lines


def render_events(result):
    kinds = list(result["kinds"])
    lines = ["{:<12}{:>9}{:>9}{:>9}".format("kind", "events", "matched", "recall")]
    for kind, row in result["kinds"].items():
        lines.append("{:<12}{:>9}{:>9}{:>9}".format(kind, row["events"], row["matched"], _pct(row["recall"])))
    lines.append("")
    lines.append("{:<12}".format("call/event") + "".join("{:>11}".format(kind) for kind in kinds))
    for call, row in result["confusion"].items():
        lines.append("{:<12}".format(call) + "".join("{:>11}".format(row[kind]) for kind in kinds))
    lines.append("setting {}; an event is matched to the nearest call whose restart is within {} frames; matches {}, "
                 "seeds {}".format(label(result["setting"]), EVENT_WINDOW, ",".join(result["matches"]),
                                   result["seeds"]))
    return lines


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
    three = commands.add_parser("stats", help="the benchmark: the confidence of each restart type's vote, per setting")
    four = commands.add_parser("events", help="the benchmark: the calls against DFL's restart events (diagnostic)")
    five = commands.add_parser("configs", help="R014's configs: main.toml plus the measured restart vote, per setting")
    five.add_argument("stats", metavar="STATS_JSON", help="the JSON of `stats --out`")
    five.add_argument("--out-dir", required=True, help="where the r014_t*_b*.toml files go (folders are made)")
    five.add_argument("--span", choices=("after", "stoppage_end"), default="after",
                      help="after (R014, default) or stoppage_end (R014b: no tail, the vote on the last stoppage frame)")
    five.add_argument("--from-tail", type=int, metavar="TAIL",
                      help="stoppage_end: the tail whose settings t<TAIL>_b0 and _b1 supply the confidences")
    five.add_argument("--switch-restart", nargs="+", type=float, metavar="X",
                      help="stoppage_end: add restart_source and switch_restart X (inf allowed) to [decoder]")
    five.add_argument("--main", default=os.path.join(ROOT, "pipeline", "configs", "main.toml"),
                      help="the config to copy (default pipeline/configs/main.toml)")
    for command in (three, four):
        command.add_argument("--model", default=DEFAULT_MODEL, help="the dead-ball model (default {})".format(
            DEFAULT_MODEL))
        command.add_argument("--profile", default=DEFAULT_PROFILE, help=run.profile_help())
        command.add_argument("--seeds", type=int, help="degradation seeds per match (default {})".format(
            run.DEFAULT_SEEDS))
        command.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run (default: all 7)")
    four.add_argument("--tail", type=int, default=50, help="the setting's tail (default 50)")
    four.add_argument("--use-ball", action="store_true", help="read the ball in the arrangement (default: no)")
    for command in (one, two, three, four):
        command.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
        command.add_argument("--out", help="also write the result as JSON here (counts and shares; folders are made)")
    args = parser.parse_args(argv)
    if args.command == "configs":
        try:
            with open(args.stats) as handle:
                stats = json.load(handle)
            written = write_configs(stats, args.main, args.out_dir, span=args.span,
                                    switch_restart=args.switch_restart, label_from=args.from_tail)
        except (ValueError, KeyError, OSError) as error:
            print("FAILED    {}".format(error))
            return 1
        for path in written:
            print("written   {}".format(run.shown(path)))
        return 0
    try:
        if args.command == "anatomy":
            result = anatomy(args.config, profile=args.profile, seeds=args.seeds, truth=args.truth,
                             match_ids=args.matches, data_dir=args.data_dir)
            lines = render(result)
        elif args.command == "stats":
            result = measure_restarts(GRID, model=args.model, profile=args.profile, seeds=args.seeds,
                                      match_ids=args.matches, data_dir=args.data_dir)
            lines = render_stats(result)
        elif args.command == "events":
            result = events_diagnostic(model=args.model, tail=args.tail, use_ball=args.use_ball,
                                       profile=args.profile, seeds=args.seeds, match_ids=args.matches,
                                       data_dir=args.data_dir)
            lines = render_events(result)
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
