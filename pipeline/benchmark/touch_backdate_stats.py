"""touch_backdate_stats: step 0 of P28, the gate of back-dating a possession to the team's first touch.

    python pipeline/benchmark/touch_backdate_stats.py --radius R --max-rel-speed V --min-frames K [--duel-radius 1.0]
        [--taus 3 6] [--profile generic] [--seeds 3] [--matches ID ...] [--data-dir D] [--out PATH]

    result = measure({"radius": 1.5, "max_rel_speed": 6.0, "min_frames": 1}, taus=(3.0, 6.0))

P28 (docs/superpowers/specs/2026-11-10-p28-touch-backdating-design.md) believes that a team which wins the ball with
a touch and controls it a moment later is heard by controlled_touch one pass flight late. Before anything is built into
a config, this measures, for the control rule given on the command line and each tau, on the degraded tracking of every
benchmark match and seed, against the raw truth:

* (i)  among the truth's team flips (touch_stats' definition: a change of team with no dead frame between, at frame c)
  where the served ball is visible at c, the share that have a touch of the new team dated within WINDOW frames of c
  and no kept episode of the new team starting within WINDOW frames of c, both in c's graded half. Needed: 10%.
  Beside it, and not gating: the same count without the flips at which a kept episode of the new team is already
  running (it started more than WINDOW frames before c, so (i) calls it "no episode", and the rule has nothing to
  back-date there), as a share of the same ball-visible flips.
* (ii) over every frame that the rule would back-date, the share where the truth names the touching team; a frame the
  truth calls D counts against. Needed: 70%. The same with the stretch capped at each of CAPS frames, for the one retry
  the spec allows.
* per match and seed, the back-dated stretches per graded minute with the ball visible
  (controlled_touch.backdated_stretches, which any match's Inputs can be given): the band of the one label-free check
  that the spec makes on our match comes from their lowest and highest.

The gate passes for a tau when (i) and (ii) both do. It is a measurement: the exit code is 0 whether it passes or not.
The suggested backdate_confidence is the floor of (ii) to a hundredth (touch_stats.confidence_from_agreement's
flooring, without its clip: the config clips it to [hold, confidence]).

One code path. The touches are controlled_touch.touches' and the back-dated frames are read off the evidence of
controlled_touch.compute itself, called with backdate_tau and, for the capped rows, backdate_cap: confidence 0.99, no
hold and backdate_confidence MARK, so that a frame is back-dated exactly where `home` is MARK or 1 - MARK, and a
stretch is a maximal run of such frames (two stretches are never adjacent: a kept episode lies between them). The
episode starts are those of the same call without backdate_tau, taken as touch_stats takes them.

No truth reaches the source. It is called on a MatchInputs (inputs.py), which holds the degraded tracking and no flag.
The truth is read here, from the match, only to count: it is never given to the source nor kept in the result, which
holds counts and shares and no frame. Matches are loaded one at a time and released, as the runner does.
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
from pipeline.benchmark import degrade, dfl, run, touch_stats  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.contracts import RunContext  # noqa: E402
from pipeline.evidence import controlled_touch, kinematics  # noqa: E402

import possession_metrics as pm  # noqa: E402  isort:skip

# Params
KEYS = touch_stats.KEYS                 # radius, max_rel_speed, min_frames: the control rule that is measured
DEFAULT_TAUS = (3.0, 6.0)               # m/s: the spec's only grid dimension
DEFAULT_DUEL_RADIUS = 1.0               # controlled_touch's own default
DEFAULT_PROFILE = degrade.GENERIC       # the spec measures step 0 on the profile that reads nothing of our match
WINDOW = 12                             # frames either side of a flip where a touch or an episode start is looked for
CAPS = (12, 25, 50)                     # frames: the caps that (ii) is reported under as well
NEEDED_FLIPS, NEEDED_AGREEMENT = 0.10, 0.70      # the gate: (i) and (ii), declared in the spec before any run
MEASURED_CONFIDENCE = touch_stats.MEASURED_CONFIDENCE   # what the source is called with; only where it speaks is read
MARK = 0.75                             # backdate_confidence in the calls: a back-dated frame is MARK or 1 - MARK
UNCAPPED = "uncapped"                   # the name of the rule's own row beside cap_12, cap_25, cap_50


def floored(share):
    """floor(100 x share) / 100: confidence_from_agreement's flooring of touch_stats, without its clip.

    100 x share is rounded to 9 places before the floor, as there (100 x 0.57 is 56.99999999999999 in floating point).
    ValueError for anything but a real number in [0, 1].
    """
    if isinstance(share, bool) or not isinstance(share, numbers.Real) or not 0.0 <= share <= 1.0:
        raise ValueError("touch_backdate_stats: a share must be a number in [0, 1], got {!r}".format(share))
    return math.floor(round(100.0 * share, 9)) / 100.0


def tau_label(tau):
    """The name of a tau in the result and the printout: 3, 6, 4.5."""
    return "{:g}".format(tau)


def cap_label(cap):
    """The name of a row of (ii): uncapped, cap_12, cap_25, cap_50."""
    return UNCAPPED if cap is None else "cap_{}".format(cap)


def measure(setting, *, taus=DEFAULT_TAUS, duel_radius=DEFAULT_DUEL_RADIUS, profile=DEFAULT_PROFILE, seeds=None,
            match_ids=None, matches=None, calibration=None, data_dir=None, log=print):
    """Step 0's counts for the control rule `setting` and each of `taus`, per match and pooled over matches and seeds.

    `setting` is {radius, max_rel_speed, min_frames} and `duel_radius` the rule's fourth parameter; `taus` are in
    metres per second. The other arguments are those of touch_stats.measure: `profile` (how the tracking is degraded),
    `seeds` (0 up to seeds - 1; the clean profile has just seed 0), `match_ids`, `matches` ({id: DflMatch} to use
    instead of loading), `calibration` (read from the cache if none is given and the profile needs one; generic and
    clean need none) and `data_dir`; `log` receives a line per match.

    Returns {"profile", "seeds", "matches", "truth", "setting", "window", "caps", "taus"}; "taus" maps tau_label(tau)
    to {"tau", "flips", "backdated", "stretches_per_minute", "gate", "backdate_confidence"}:

    * flips: {"pooled", "per_match"} of rows {flips, visible, touch, episode, touch_no_episode, share, inside_episode,
      touch_no_episode_outside, share_outside}. `flips` counts every team flip of the truth and the other counts those
      where the served ball is visible at the flip: all of them, those with a touch of the new team within WINDOW
      frames, those with an episode start of the new team within WINDOW frames, and those with the touch and without
      the episode; share = touch_no_episode / visible, (i). Not gating: inside_episode, the ball-visible flips at which
      a kept episode of the new team is running; touch_no_episode_outside, (i)'s count without them; and share_outside
      = touch_no_episode_outside / visible, the same denominator as (i)'s.
    * backdated: {cap_label(cap): {"pooled", "per_match"}} for no cap and each of CAPS, of rows {frames, agree, share,
      stretches, length_p25, length_median, length_p75}: the back-dated frames, those where the truth names the
      touching team, their share (ii), the stretches and the quartiles of their lengths in frames.
    * stretches_per_minute: {"per_run": {match id: {seed: value}}, "min", "max"}: uncapped stretches per minute of
      graded frames with the ball visible, for each match and seed.
    * gate: {flips_share, flips_needed, flips_pass, agreement, agreement_needed, agreement_pass, passed}.
    * backdate_confidence: floored(the uncapped pooled (ii)), None if nothing was back-dated.

    A share or a quartile of nothing is None, and the gate then fails. ValueError (touch_backdate_stats) for a setting
    that misses or adds a key and for taus that are none, not positive finite numbers, repeated, or two of one label
    (tau_label, 6 significant digits: they would be one entry of the result); the runner's
    BenchmarkError for a profile, seeds or match it refuses. All of that is refused before a match is loaded; a value
    that controlled_touch refuses (a radius of 0) is its ValueError, on the first match.
    """
    setting = _checked_setting(setting)
    taus = _checked_taus(taus)
    run._check_profile(profile)
    seeds = run.DEFAULT_SEEDS if seeds is None else seeds
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise run.BenchmarkError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    if run.needs_calibration(profile) and calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    seed_list = [0] if profile == run.CLEAN else list(range(seeds))
    rule = dict(setting, duel_radius=duel_radius, confidence=MEASURED_CONFIDENCE)

    flip_counts = {tau: {} for tau in taus}                    # tau -> match id -> the seven counts of (i)
    stretch_counts = {tau: {cap: {} for cap in (None,) + CAPS} for tau in taus}   # cap -> match id -> [frames, agree]
    lengths = {tau: {cap: {} for cap in (None,) + CAPS} for tau in taus}          # -> cap -> match id -> [length, ...]
    rates = {tau: {} for tau in taus}                          # tau -> match id -> seed -> stretches per minute
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        codes, _ = truths.truth_codes(match)                   # the raw truth, read here to count; never handed on
        flips = touch_stats._flips(codes, match.graded)
        for seed in seed_list:
            atd = run.make_atd(match, profile, seed, calibration)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            context = RunContext(inputs=inputs, seed=0)
            visible = kinematics.ball_track(inputs).visible
            episodes = controlled_touch.compute(context, **rule).home
            starts, running = _episode_starts(episodes, match.graded), _episode_frames(episodes)
            for tau in taus:
                touched = controlled_touch.touches(inputs, radius=setting["radius"], duel_radius=duel_radius, tau=tau)
                row = flip_counts[tau].setdefault(match_id, np.zeros(7, dtype=np.int64))
                row += _flip_counts(flips, match.graded, visible, touched, starts, running)
                stretches, with_ball = controlled_touch.backdated_stretches(inputs, backdate_tau=tau,
                                                                            duel_radius=duel_radius, **setting)
                rates[tau].setdefault(match_id, {})[seed] = (stretches / (with_ball / (60.0 * match.fps))
                                                             if with_ball else None)
                for cap in (None,) + CAPS:
                    home = controlled_touch.compute(context, backdate_tau=tau, backdate_confidence=MARK,
                                                    backdate_cap=cap, **rule).home
                    frames, agree, found = _stretches(home, codes)
                    row = stretch_counts[tau][cap].setdefault(match_id, np.zeros(2, dtype=np.int64))
                    row += (frames, agree)
                    lengths[tau][cap].setdefault(match_id, []).extend(found)
                    if len(found) != stretches:
                        # A stretch is read off the evidence as a maximal run of marked frames. Two stretches cannot
                        # touch (a kept episode lies between them) and a cap leaves a frame of each: if this ever
                        # fails, the evidence is not what this module takes it for, and no figure of it stands.
                        raise RuntimeError("touch_backdate_stats: {} seed {} tau {} cap {}: {} runs of back-dated "
                                           "frames in the evidence and {} stretches by controlled_touch."
                                           "backdated_stretches".format(match_id, seed, tau_label(tau), cap,
                                                                        len(found), stretches))
            del atd, inputs, context, visible, episodes, starts, running
        log("match     {}   {} taus x {} seeds".format(match_id, len(taus), len(seed_list)))
        del match, codes, flips

    out = {}
    for tau in taus:
        flips_pooled = _flip_row(np.sum(list(flip_counts[tau].values()), axis=0))
        backdated = {}
        for cap in (None,) + CAPS:
            counts, found = stretch_counts[tau][cap], lengths[tau][cap]
            backdated[cap_label(cap)] = {
                "pooled": _stretch_row(np.sum(list(counts.values()), axis=0),
                                       [length for match_id in sorted(found) for length in found[match_id]]),
                "per_match": {match_id: _stretch_row(counts[match_id], found[match_id]) for match_id in sorted(counts)}}
        agreement = backdated[UNCAPPED]["pooled"]["share"]
        values = [value for per_seed in rates[tau].values() for value in per_seed.values() if value is not None]
        flips_pass = flips_pooled["share"] is not None and flips_pooled["share"] >= NEEDED_FLIPS
        agreement_pass = agreement is not None and agreement >= NEEDED_AGREEMENT
        out[tau_label(tau)] = {
            "tau": tau,
            "flips": {"pooled": flips_pooled,
                      "per_match": {match_id: _flip_row(flip_counts[tau][match_id])
                                    for match_id in sorted(flip_counts[tau])}},
            "backdated": backdated,
            "stretches_per_minute": {
                "per_run": {match_id: {str(seed): value for seed, value in sorted(rates[tau][match_id].items())}
                            for match_id in sorted(rates[tau])},
                "min": min(values) if values else None, "max": max(values) if values else None},
            "gate": {"flips_share": flips_pooled["share"], "flips_needed": NEEDED_FLIPS, "flips_pass": flips_pass,
                     "agreement": agreement, "agreement_needed": NEEDED_AGREEMENT, "agreement_pass": agreement_pass,
                     "passed": flips_pass and agreement_pass},
            "backdate_confidence": None if agreement is None else floored(agreement)}
    return {"profile": profile, "seeds": seed_list, "matches": sorted(ids), "truth": "raw",
            "setting": dict(setting, duel_radius=duel_radius), "window": WINDOW, "caps": list(CAPS), "taus": out}


def _checked_setting(setting):
    """`setting` as a dict with exactly KEYS; ValueError (touch_backdate_stats) otherwise."""
    if not isinstance(setting, dict) or set(setting) != set(KEYS):
        raise ValueError("touch_backdate_stats: the setting needs exactly the keys {} and got {!r}; extra: {}, "
                         "missing: {}".format(
                             ", ".join(KEYS), setting,
                             sorted(set(setting) - set(KEYS)) if isinstance(setting, dict) else "?",
                             sorted(set(KEYS) - set(setting)) if isinstance(setting, dict) else "?"))
    return dict(setting)


def _checked_taus(taus):
    """`taus` as a list of floats, each a positive finite number and none repeated; ValueError otherwise."""
    taus = list(taus)
    if not taus:
        raise ValueError("touch_backdate_stats: no tau to measure; give at least one, in metres per second")
    for tau in taus:
        if isinstance(tau, bool) or not isinstance(tau, numbers.Real) or not 0 < tau < math.inf:
            raise ValueError("touch_backdate_stats: a tau must be a positive finite number of metres per second, "
                             "got {!r}".format(tau))
    taus = [float(tau) for tau in taus]
    repeated = sorted({tau for tau in taus if taus.count(tau) > 1})
    if repeated:
        raise ValueError("touch_backdate_stats: tau {} is given twice; it would be measured twice".format(
            tau_label(repeated[0])))
    labels = [tau_label(tau) for tau in taus]
    for label in labels:
        same = [tau for tau, name in zip(taus, labels) if name == label]
        if len(same) > 1:
            raise ValueError("touch_backdate_stats: the taus {} would both be named {} in the result (a tau is named "
                             "by 6 significant digits), and one would replace the other; give taus that differ "
                             "there".format(" and ".join(repr(tau) for tau in same), label))
    return taus


def _episode_starts(home, graded):
    """(home's, away's): bool masks of the first frame of every kept episode, as touch_stats._flip_lags takes them.

    An episode starts on a kept frame whose predecessor is not kept, or is of the other team, and always on the first
    frame of a block. `home` is the source's evidence without a hold, so a frame is kept where it is not NaN.
    """
    kept = ~np.isnan(home)
    team_home = home > 0.5
    start = kept.copy()
    start[1:] &= ~(kept[:-1] & (team_home[:-1] == team_home[1:]))
    for _, first, _ in graded:
        start[first] = kept[first]
    return start & team_home, start & ~team_home


def _episode_frames(home):
    """(home's, away's): bool masks of the frames inside a kept episode of each team, from the evidence without a
    hold."""
    kept = ~np.isnan(home)
    team_home = home > 0.5
    return kept & team_home, kept & ~team_home


def _flip_counts(flips, graded, visible, touched, starts, running):
    """[flips, visible, touch, episode, touch and no episode, inside an episode, touch and no episode and not inside
    one] of one match and seed: the counts of (i), and of (i) without the flips inside an episode of the new team.

    `touched`, `starts` and `running` are (home's, away's) masks: the dated touches, the first frames of the kept
    episodes and all their frames. A flip at c with the ball visible has a touch (an episode) when a touch (an episode
    start) of the new team lies in [c - WINDOW, c + WINDOW], cut at the edges of c's graded half, and is inside an
    episode when c is a frame of a kept episode of the new team.
    """
    counts = np.zeros(7, dtype=np.int64)
    # Running sums, so that "any in a window" is one subtraction: [0] home's, [1] away's.
    touch_sums = [np.concatenate(([0], np.cumsum(mask))) for mask in touched]
    start_sums = [np.concatenate(([0], np.cumsum(mask))) for mask in starts]
    for frame, _, _, new_is_home in flips:
        counts[0] += 1
        if not visible[frame]:
            continue
        counts[1] += 1
        first, stop = next((start, stop) for _, start, stop in graded if start <= frame < stop)
        low, high = max(frame - WINDOW, first), min(frame + WINDOW, stop - 1) + 1
        team = 0 if new_is_home else 1
        touch = touch_sums[team][high] > touch_sums[team][low]
        episode = start_sums[team][high] > start_sums[team][low]
        inside = running[team][frame]
        counts[2] += touch
        counts[3] += episode
        counts[4] += touch and not episode
        counts[5] += inside
        counts[6] += touch and not episode and not inside
    return counts


def _stretches(home, codes):
    """(frames, agreeing frames, [length, ...]) of the back-dated stretches in one call's evidence: (ii)'s counts.

    A frame is back-dated where `home` is MARK (for home) or 1 - MARK (for away), and agrees where the truth's code is
    that team's: D, and the other team, count against.
    """
    for_home, for_away = home == MARK, home == 1.0 - MARK
    agree = (for_home & (codes == pm.H)) | (for_away & (codes == pm.A))
    starts, stops = kinematics.runs(for_home | for_away)
    return int(for_home.sum() + for_away.sum()), int(agree.sum()), (stops - starts).tolist()


def _flip_row(counts):
    flips, visible, touch, episode, both, inside, outside = (int(count) for count in counts)
    return {"flips": flips, "visible": visible, "touch": touch, "episode": episode, "touch_no_episode": both,
            "share": both / visible if visible else None, "inside_episode": inside,
            "touch_no_episode_outside": outside, "share_outside": outside / visible if visible else None}


def _stretch_row(counts, found):
    frames, agree = (int(count) for count in counts)
    p25, median, p75 = (float(value) for value in np.percentile(found, [25, 50, 75])) if found else (None, None, None)
    return {"frames": frames, "agree": agree, "share": agree / frames if frames else None, "stretches": len(found),
            "length_p25": p25, "length_median": median, "length_p75": p75}


# -- the tables -------------------------------------------------------------------------------------------------------

def render(result):
    """The printout's lines but the gate's: per tau the tables of (i), (ii) and the stretches per minute. Aggregates."""
    setting = result["setting"]
    lines = ["step 0 of P28: radius {} m, max_rel_speed {} m/s, min_frames {}, duel_radius {} m; profile {}, seeds {}, "
             "{} truth, matches {}".format(setting["radius"], setting["max_rel_speed"], setting["min_frames"],
                                           setting["duel_radius"], result["profile"], result["seeds"], result["truth"],
                                           ",".join(result["matches"]))]
    for name, entry in result["taus"].items():
        lines += ["", "tau {} m/s".format(name),
                  "  (i) team flips of the truth; a touch or an episode start of the new team within {} frames".format(
                      result["window"]),
                  "  {:<12}{:>8}{:>14}{:>8}{:>10}{:>20}{:>9}".format(
                      "", "flips", "ball visible", "touch", "episode", "touch, no episode", "share")]
        for label, row in [("pooled", entry["flips"]["pooled"])] + sorted(entry["flips"]["per_match"].items()):
            lines.append("  {:<12}{:>8}{:>14}{:>8}{:>10}{:>20}{:>9}".format(
                label, row["flips"], row["visible"], row["touch"], row["episode"], row["touch_no_episode"],
                _pct(row["share"])))
        lines += ["  (i) excluding flips inside a new-team episode (not gating): a kept episode of the new team is "
                  "running at the flip",
                  "  {:<12}{:>22}{:>34}{:>24}".format(
                      "", "inside an episode", "touch, no episode, not inside", "share of ball visible")]
        for label, row in [("pooled", entry["flips"]["pooled"])] + sorted(entry["flips"]["per_match"].items()):
            lines.append("  {:<12}{:>22}{:>34}{:>24}".format(
                label, row["inside_episode"], row["touch_no_episode_outside"], _pct(row["share_outside"])))
        lines += ["  (ii) back-dated frames where the truth names the touching team; stretch lengths in frames",
                  "  {:<22}{:>9}{:>9}{:>9}{:>11}{:>8}{:>8}{:>8}".format(
                      "", "frames", "agree", "share", "stretches", "p25", "median", "p75")]
        for cap in (None,) + tuple(result["caps"]):
            rows = [("pooled", entry["backdated"][cap_label(cap)]["pooled"])]
            if cap is None:
                rows += sorted(entry["backdated"][cap_label(cap)]["per_match"].items())
            for label, row in rows:
                lines.append("  {:<22}{:>9}{:>9}{:>9}{:>11}{:>8}{:>8}{:>8}".format(
                    "{} {}".format("no cap" if cap is None else "cap {}".format(cap), label), row["frames"],
                    row["agree"], _pct(row["share"]), row["stretches"], _number(row["length_p25"]),
                    _number(row["length_median"]), _number(row["length_p75"])))
        rate = entry["stretches_per_minute"]
        lines.append("  back-dated stretches per graded minute with the ball visible, per match and seed: "
                     "min {}, max {}".format(_rate(rate["min"]), _rate(rate["max"])))
        for match_id, per_seed in rate["per_run"].items():
            lines.append("  {:<12}{}".format(match_id, "".join("{:>10}".format(_rate(value))
                                                                for value in per_seed.values())))
    return lines


def gate_lines(result):
    """The last lines of the printout: for each tau the gate, as the spec declares it, the share that does not gate
    ((i) without the flips inside an episode of the new team) and the confidence that (ii) gives. The shares have 6
    decimals, so that one just under its bar does not print as the bar beside a FAIL."""
    lines = []
    for name, entry in result["taus"].items():
        gate = entry["gate"]
        lines.append("gate tau={}: (i) {} >= {:.2f} {}; (ii) {} >= {:.2f} {} -> {}".format(
            name, _share(gate["flips_share"]), gate["flips_needed"], "pass" if gate["flips_pass"] else "FAIL",
            _share(gate["agreement"]), gate["agreement_needed"], "pass" if gate["agreement_pass"] else "FAIL",
            "PASS" if gate["passed"] else "FAIL"))
        lines.append("not gating tau={}: (i) excluding flips inside a new-team episode {}".format(
            name, _share(entry["flips"]["pooled"]["share_outside"])))
        lines.append("suggested backdate_confidence tau={}: {} (the floor of (ii) to a hundredth; the config clips it "
                     "to [hold, confidence])".format(
                         name, "n/a" if entry["backdate_confidence"] is None
                         else "{:.2f}".format(entry["backdate_confidence"])))
    return lines


def _pct(value):
    return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def _share(value):
    return "n/a" if value is None else "{:.6f}".format(value)


def _number(value):
    return "n/a" if value is None else "{:.0f}".format(value)


def _rate(value):
    return "n/a" if value is None else "{:.3f}".format(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Step 0 of P28, a measurement: for a control rule and each tau, how many of the truth's team "
                    "flips have a touch of the new team and no episode (i), how often the frames the rule would "
                    "back-date are the touching team's (ii), and the gate the spec declares on the two. The exit code "
                    "is 0 whether the gate passes or not.")
    parser.add_argument("--radius", type=float, required=True, help="the control rule's radius, in metres")
    parser.add_argument("--max-rel-speed", type=float, required=True,
                        help="the control rule's max_rel_speed, in metres per second")
    parser.add_argument("--min-frames", type=int, required=True, help="the control rule's min_frames")
    parser.add_argument("--duel-radius", type=float, default=DEFAULT_DUEL_RADIUS,
                        help="the control rule's duel_radius, in metres (default {})".format(DEFAULT_DUEL_RADIUS))
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS), metavar="TAU",
                        help="the least change of the ball's velocity at a touch, in metres per second (default "
                             "{})".format(" ".join(tau_label(tau) for tau in DEFAULT_TAUS)))
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help="how the tracking is made: {} (default {}, which the spec measures on)".format(
                            ", ".join(sorted(run.PROFILES)), DEFAULT_PROFILE))
    parser.add_argument("--seeds", type=int,
                        help="degradation seeds per match (default {})".format(run.DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to run, in this order (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--out", help="also write the result as JSON here (counts and shares, no frame)")
    args = parser.parse_args(argv)
    setting = {"radius": args.radius, "max_rel_speed": args.max_rel_speed, "min_frames": args.min_frames}
    try:
        result = measure(setting, taus=args.taus, duel_radius=args.duel_radius, profile=args.profile, seeds=args.seeds,
                         match_ids=args.matches, data_dir=args.data_dir)
    except (run.BenchmarkError, ValueError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    print("")
    for line in render(result):
        print(line)
    print("")
    code = 0
    if args.out:
        # After the tables, so that a file that cannot be written does not cost the minutes the measurement took.
        try:
            run.write_json(args.out, run.plain(result))
            print("written   {}".format(run.shown(args.out)))
        except OSError as error:
            print("FAILED    {} was not written: {}".format(run.shown(args.out), error))
            code = 1
    for line in gate_lines(result):
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
