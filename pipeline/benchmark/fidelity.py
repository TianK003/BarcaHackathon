"""How far a degraded benchmark looks like our match, measured as our match was: gates and warnings.

    python pipeline/benchmark/fidelity.py --profile gaps --seeds 3     # -> <data dir>/runs/benchmark/fidelity_gaps.json
    python pipeline/benchmark/fidelity.py --profile tracking --seeds 3     # -> .../fidelity_tracking.json
    report = fidelity.report("gaps", seeds=3)                          # the same, as a dict (aggregates only)

What it is for. P01 validates pipeline choices on matches that are not ours, after degrading them until their tracking
looks like ours. "Looks like ours" has to be measured, and measured the same way on both sides: calibration.py owns
two statistics functions, ball_stats and player_stats, that measure BAR-ATH, and this module applies those same two
functions, unchanged, to each degraded benchmark match, and pools the results over the matches and the degradation
seeds. A difference between the two numbers is then a difference between the matches, never between two ways of
counting. Both sides are measured on the same kind of span: each half from its kick-off to the end of its graded block
(a benchmark half starts at its kick-off; calibration.measure does the same for ours).

Two kinds of criteria, as the plan fixes them. Neither is ever tuned towards.

Gates (GATES, per profile). The profile does not count as built until they hold: they say that the code reproduces what
it was calibrated on. Each is the Calibration's own value (measured on the kick-off spans, never a number copied from
the plan, which was measured on other spans) plus or minus a tolerance, which is absolute (+- tolerance) or relative
(+- tolerance of the centre: the tracklet median's). For gaps: the missing share of the ball +-0.02 and the median gap
+-1.5 s. For tracking, those two and: the share of frames with exactly 22 slots present +-0.06, the mean number present
+-0.5, the share with 14 or fewer present +-0.02, the wrong-team share +-0.03, the tracklet median +-50% and the
NaN-speed share +-0.01. A gate is printed as measured, centre, tolerance, and pass or FAIL; a FAIL makes the command
exit 1, and the profile is not frozen until it passes.

Pending gates (PENDING_GATES). A gate that cannot be met yet, because the part of the profile it depends on is not
built, is printed as measured, centre and tolerance with the status `pending`, is not counted by `passed` nor by the
exit code, and the report says why (NOTES). Today these are the mean number present and the share with 14 or fewer
present: they depend on the low-count tail (our frames with 18 or fewer slots present, 10.6%), which the tracking
profile does not have yet. The task that builds the tail empties the list, and then they are gates like the rest.

Checks (the warnings). Pre-registered bands on things the degradation does not fit but a benchmark that resembles our
match should show anyway: the nearest player's agreement with the truth on alive, ball-visible frames (compared with
the public window's 77.8%, by the same pipeline.evidence.describe), the share of dead frames on which the ball is
visible, and, for each yardstick config, the possessions it outputs per graded minute and its D share (against what the
four yardsticks output on ours). A miss is a WARN that results.md discusses. Some misses are no fault of the degradation
(the truth's conventions differ from ours, A06), and some are what the profile does not model: its gaps are drawn
without looking at the play, so what ties ours to it (a dead ball being likelier to be missing, say) is not in them.
The bands are:

    nearest agreement      +-0.05      dead-ball ball visible share   +-0.10
    possessions per minute +-25%       D share                        +-0.05

The yardsticks' outputs of ours (Calibration.baseline_outputs) are over our graded halves, as the published baselines
are, and not over the kick-off spans of the rest of the Calibration: an accepted offset of at most 1% of the frames
(the 117 s of the second half before its kick-off, all D, and the first half's lead-in).

What is pooled how. Every sum is over all matches and seeds, as the runner pools (score.pooled): counts add and the
statistic is computed from the sums (missing share = missing frames over frames), and the arrays of gaps and tracklets
are laid end to end (the median gap is the median of every gap there was). The report also holds the benchmark's other
statistics next to ours, as information (STATISTICS), and each draw of the degrader (which half of ours, from which
frame): two integers a half, nothing of anybody's frames.

Nothing of our match that is frame-level is printed or written: a report holds counts, shares, quantiles and those
draws. Reports are deterministic (sorted keys, LF, floats rounded to 9 decimals, no timing, atomic write) and carry
the Calibration's digest, so a report says which measurement of ours it was made against. The file name follows the
runner's rule (run.name_suffixes): `fidelity_<profile>.json` for all seven matches and the default seeds, `.m-<ids>`
and `.s<n>` otherwise, so that a quick look never replaces the report that a result is quoted from.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if __name__ == "__main__":
    # Run by path, Python put pipeline/benchmark/ itself first on sys.path: swap in the repository root instead, as
    # run.py does, so that `import pipeline` resolves from any working directory and nothing in here shadows a module.
    sys.path[0] = ROOT

import numpy as np  # noqa: E402

import games  # noqa: E402
from pipeline import evidence, output  # noqa: E402  (importing pipeline also puts evaluation/ on sys.path)
from pipeline.benchmark import calibration as calibration_module  # noqa: E402
from pipeline.benchmark import degrade, dfl, run  # noqa: E402
from pipeline.benchmark import inputs as match_inputs  # noqa: E402
from pipeline.benchmark import truth as truths  # noqa: E402
from pipeline.config import ConfigError, load_config  # noqa: E402
from pipeline.contracts import D  # noqa: E402

# Params
DEFAULT_SEEDS = 3
PASS, FAIL, PENDING, OK, WARN = "pass", "FAIL", "pending", "ok", "WARN"
DECIMALS = run.DECIMALS
# The gates of each profile, in the order they are printed: name, tolerance (absolute, or relative to the centre when
# "kind" says so), and what the number is. The centre of each is the Calibration's (centre_of). A later profile's gates
# include the earlier one's, as the plan says.
BALL_GATES = (
    {"name": "missing_share", "tolerance": 0.02, "kind": "absolute", "text": "share of the frames without a ball"},
    {"name": "median_gap_s", "tolerance": 1.5, "kind": "absolute",
     "text": "median length of a gap in the ball, seconds"},
)
GATES = {
    "gaps": BALL_GATES,
    "tracking": BALL_GATES + (
        {"name": "present_share_22", "tolerance": 0.06, "kind": "absolute",
         "text": "share of the frames with exactly 22 slots present"},
        {"name": "present_mean", "tolerance": 0.5, "kind": "absolute", "text": "mean number of slots present"},
        {"name": "present_share_le14", "tolerance": 0.02, "kind": "absolute",
         "text": "share of the frames with 14 or fewer slots present"},
        {"name": "wrong_team_share", "tolerance": 0.03, "kind": "absolute",
         "text": "share of the frames with a team in surplus (a wrong-team slot)"},
        {"name": "tracklet_median_s", "tolerance": 0.5, "kind": "relative",
         "text": "median length of a tracklet, seconds"},
        {"name": "speed_nan_share", "tolerance": 0.01, "kind": "absolute",
         "text": "share of the present slot-frames with a NaN speed"},
    ),
}
# The gates that cannot be met until the low-count tail is built: both are about how many slots are present, which
# the tail's frames (18 or fewer present) bring down. Printed with the status "pending" and never counted; the tail's
# task empties this tuple.
PENDING_GATES = ("present_mean", "present_share_le14")
# The warning checks: the band, and whether it is absolute (+- band) or relative (+- band of the centre).
NEAREST_BAND = ("absolute", 0.05)
DEAD_VISIBLE_BAND = ("absolute", 0.10)
POSSESSIONS_BAND = ("relative", 0.25)
D_SHARE_BAND = ("absolute", 0.05)
FILE_FOLDER = run.REPORT_FOLDER                   # <data dir>/runs/benchmark/fidelity_<profile>[.m-<ids>][.s<n>].json
# The counts of what the players' degradation drew that a report adds up over matches, seeds and halves.
PLAYERS_DRAWS = ("tracklets", "onsets", "episodes", "skipped_far", "skipped_overlap", "episode_frames",
                 "speed_nan_runs", "speed_nan_frames")
NOTES = ("The yardsticks' outputs of ours (possessions per minute, D share) are over our graded halves, as the "
         "published baselines are, and not over the kick-off spans of the rest of the Calibration: an accepted offset "
         "of at most 1% of the frames.",
         "The centre of a gate is the Calibration's own value, measured on the kick-off spans; the plan's figures were "
         "measured on other spans, and are not used.")
PENDING_NOTE = ("Pending gates ({}) are printed and not counted: they depend on the low-count tail (our frames with 18 "
                "or fewer slots present), which the tracking profile does not have yet, and the task that builds it "
                "empties PENDING_GATES.")


class FidelityError(ValueError):
    """A fidelity report that cannot be made as asked; the message says what to change."""


# -- the report -------------------------------------------------------------------------------------------------------

def report(profile, seeds=DEFAULT_SEEDS, match_ids=None, matches=None, calibration=None, data_dir=None, log=print):
    """The fidelity of `profile` on the benchmark's matches, against our match's Calibration: a JSON-ready dict.

    Each match (all seven, or `match_ids`; `matches` is {id: DflMatch} to use instead of loading, as in
    run.evaluate_configs) is degraded under seeds 0 .. seeds - 1, and everything is measured on the degraded tracking:
    ball_stats and player_stats of calibration.py, the nearest player's agreement with the truth, the dead-ball
    visibility and the four yardsticks' outputs. `calibration` is the Calibration of BAR-ATH, calibration.load() if
    none is given. One match is in memory at a time.

    Returns {"profile", "seeds", "matches", "calibration", "gates", "checks", "statistics", "draws", "players_draws",
    "passed", "warnings", "notes"}. "gates" is a list of {name, text, measured, centre, tolerance, kind, status}
    (status pass, FAIL or pending: PENDING_GATES), "checks" a list of {name, measured, centre, band, kind, status} (ok
    or WARN), "passed" is whether no gate failed (a pending one is not counted) and "warnings" how many checks warn.
    "players_draws" is {} for a profile that draws nothing for its players and, for tracking, the counts of what was
    drawn, pooled over matches, seeds and halves (onsets, episodes, skipped_far, skipped_overlap, episode_frames,
    speed_nan_runs, tracklets). FidelityError for a profile that has no gates (clean, which degrades nothing) or that
    is not one, for seeds below 1 and for a calibration that is not a Calibration, and run.BenchmarkError for matches
    that are not as asked.
    """
    if profile not in degrade.PROFILES:
        raise FidelityError("profile {!r} is not one of: {}".format(profile, ", ".join(sorted(degrade.PROFILES))))
    if profile not in GATES:
        raise FidelityError("profile {!r} degrades nothing, so it has no gates to hold it to; fidelity is measured "
                            "for: {}".format(profile, ", ".join(sorted(GATES))))
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise FidelityError("seeds must be a whole number of at least 1, got {!r}".format(seeds))
    ids = run.resolve_match_ids(match_ids, matches)
    if calibration is None:
        calibration = calibration_module.load(data_dir=data_dir)
    if not isinstance(calibration, calibration_module.Calibration):
        raise FidelityError("calibration must be a Calibration of our own match, as calibration.load() returns, and it "
                            "is a {}".format(type(calibration).__name__))
    configs = [load_config(os.path.join(calibration_module.CONFIG_FOLDER, "nearest_w{}.toml".format(width)))
               for width in calibration_module.YARDSTICK_WIDTHS]

    pool = Pool()
    for match_id in ids:
        match = matches[match_id] if matches is not None else dfl.load_match(match_id, data_dir=data_dir)
        if match.match_id != match_id:
            raise run.BenchmarkError("the match given as {} says it is {}; its figures would be filed under the wrong "
                                     "id".format(match_id, match.match_id))
        codes_true, _ = truths.truth_codes(match)
        view = degrade.tracking_view(match)
        measured_players = None                       # (the players' arrays, their statistics) of the last seed
        for seed in range(seeds):
            atd, info = degrade.degrade(view, profile, calibration, seed)
            inputs = match_inputs.MatchInputs(match, atd, profile=profile, seed=seed, data_dir=data_dir)
            spans = [(half, inputs.kickoffs[half], stop) for half, _, stop in inputs.graded]
            pool.add_ball(calibration_module.ball_stats(atd.ball_present, spans, inputs.metres(atd.ball_xy),
                                                        inputs.pitch, inputs.fps))
            if measured_players is None or measured_players[0] is not atd.players_xy:
                # The same arrays give the same statistics: a profile that leaves the players alone (gaps) measures
                # them once for a match, however many seeds there are.
                measured_players = (atd.players_xy, calibration_module.player_stats(
                    atd.players_present, atd.slot_team, inputs.metres(atd.players_xy), atd.players_speed, spans,
                    inputs.fps))
            pool.add_players(measured_players[1])
            pool.add_draws(match_id, seed, info)
            pool.add_players_draws(info)
            dead = codes_true == D
            pool.add_dead(int(dead.sum()), int(atd.ball_present[dead].sum()))
            for position, config in enumerate(configs):
                codes, sources_ = run.predict(config, inputs)
                summary = output.summarise(codes, inputs.graded, inputs.fps)
                pool.add_yardstick(calibration_module.YARDSTICK_WIDTHS[position], summary, inputs.fps)
                if position == 0:
                    # The evidence is the same in all four configs: the first one's is read, as measure() does.
                    row = next(row for row in evidence.describe(sources_, inputs.graded_mask, codes_true)
                               if row["name"] == calibration_module.NEAREST_SOURCE)
                    pool.add_agreement(row["home_judged_frames"], row["home_agreement"])
            log("fidelity {}  seed {}   missing {:.1%} of the ball".format(
                match_id, seed, pool.last_missing_share))
            del atd, inputs
        del match, codes_true, view, measured_players

    gates = [_gate(spec, pool, calibration) for spec in GATES[profile]]
    pending = [gate["name"] for gate in gates if gate["status"] == PENDING]
    result = {
        "profile": {"name": profile, "version": degrade.PROFILES[profile]},
        "seeds": list(range(seeds)),
        "matches": list(ids),
        "calibration": {"sha256": calibration_module.digest(calibration),
                        "version": calibration_module.CALIBRATION_VERSION},
        "gates": gates,
        "checks": _checks(pool, calibration),
        "statistics": statistics(pool, calibration),
        "draws": pool.draws,
        "players_draws": dict(pool.players_draws),
        "notes": list(NOTES) + ([PENDING_NOTE.format(", ".join(pending))] if pending else []),
    }
    result["passed"] = not any(gate["status"] == FAIL for gate in gates)
    result["warnings"] = sum(1 for check in result["checks"] if check["status"] == WARN)
    return run.plain(result)


class Pool:
    """What is summed over every (match, seed): counts add, arrays are laid end to end, and nothing is averaged."""

    def __init__(self):
        self.ball = {"frames": 0, "missing_frames": 0, "gaps": 0, "interior_gaps": 0, "near_line_onsets": 0,
                     "gap_durations_s": []}
        self.players = {"frames": 0, "present_slot_frames": 0, "present_hist": np.zeros(0, dtype=np.int64),
                        "tracklets": [], "wrong_team_frames": 0, "speed_nan_frames": 0, "jitter_squares": 0.0,
                        "jitter_triples": 0}
        self.players_draws = {}
        self.agreed = self.judged = 0
        self.dead_frames = self.dead_visible = 0
        self.yardsticks = {}
        self.draws = []
        self.missing_shares = []
        self.last_missing_share = float("nan")

    def add_ball(self, stats):
        for name in ("frames", "missing_frames", "gaps", "interior_gaps", "near_line_onsets"):
            self.ball[name] += stats[name]
        self.ball["gap_durations_s"].append(stats["gap_durations_s"])
        self.last_missing_share = stats["missing_share"]
        self.missing_shares.append(stats["missing_share"])

    def add_players(self, stats):
        players = self.players
        for name in ("frames", "present_slot_frames", "wrong_team_frames", "speed_nan_frames"):
            players[name] += stats[name]
        # A squad has 39 or 40 people in the DFL matches, so the histograms are of different lengths: add by count.
        total = np.zeros(max(players["present_hist"].size, stats["present_hist"].size), dtype=np.int64)
        total[:players["present_hist"].size] += players["present_hist"]
        total[:stats["present_hist"].size] += stats["present_hist"]
        players["present_hist"] = total
        players["tracklets"].append(stats["tracklet_lengths_s"])
        if stats["jitter_triples"]:
            players["jitter_squares"] += stats["jitter_rms_m"] ** 2 * stats["jitter_triples"]
            players["jitter_triples"] += stats["jitter_triples"]

    def add_agreement(self, judged, agreement):
        if judged:
            self.judged += int(judged)
            self.agreed += int(round(agreement * judged))

    def add_dead(self, dead, visible):
        self.dead_frames += dead
        self.dead_visible += visible

    def add_yardstick(self, width, summary, fps):
        row = self.yardsticks.setdefault(width, {"possessions": 0, "graded_frames": 0, "d_frames": 0, "seconds": 0.0})
        row["possessions"] += summary["possessions"]
        row["graded_frames"] += summary["graded_frames"]
        row["d_frames"] += summary["labels_graded"]["D"]
        row["seconds"] += summary["graded_frames"] / fps

    def add_draws(self, match_id, seed, info):
        for half, record in sorted(info.get("transplants", {}).items()):
            self.draws.append({"match": match_id, "seed": seed, "half": half, **record})

    def add_players_draws(self, info):
        """Add up the counts of what the players' degradation drew (info["players"]); none for a profile without one."""
        for name in PLAYERS_DRAWS:
            if name in info.get("players", {}):
                self.players_draws[name] = self.players_draws.get(name, 0) + info["players"][name]

    # What the statistics are, from the sums.
    def gap_durations(self):
        return np.concatenate(self.ball["gap_durations_s"]) if self.ball["gap_durations_s"] else np.zeros(0)

    def tracklet_lengths(self):
        return np.concatenate(self.players["tracklets"]) if self.players["tracklets"] else np.zeros(0)


# -- gates and checks -------------------------------------------------------------------------------------------------

def centre_of(name, calibration):
    """The Calibration's value of the quantity a gate is about: measured on our kick-off spans, as the benchmark is."""
    scalars = calibration.scalars
    if name == "missing_share":
        return scalars["missing_share"]
    if name == "median_gap_s":
        return float(np.median(calibration.gap_durations_s))
    if name in ("present_share_22", "present_mean", "present_share_le14"):
        return scalars[name]
    if name == "wrong_team_share":
        return calibration.wrong_team_share
    if name == "tracklet_median_s":
        return float(np.median(calibration.tracklet_lengths_s))
    if name == "speed_nan_share":
        return calibration.speed_nan_share
    raise KeyError(name)


def measured(name, pool):
    """The benchmark's value of the quantity a gate is about, from the pool's sums (Pool): never a mean of means."""
    players = pool.players
    frames, slot_frames, hist = players["frames"], players["present_slot_frames"], players["present_hist"]
    if name == "missing_share":
        return pool.ball["missing_frames"] / pool.ball["frames"]
    if name == "median_gap_s":
        gaps = pool.gap_durations()
        return float(np.median(gaps)) if gaps.size else float("nan")
    if name == "present_share_22":
        return int(hist[calibration_module.FULL_COUNT:calibration_module.FULL_COUNT + 1].sum()) / frames
    if name == "present_mean":
        return slot_frames / frames
    if name == "present_share_le14":
        return int(hist[:calibration_module.LOW_COUNT + 1].sum()) / frames
    if name == "wrong_team_share":
        return players["wrong_team_frames"] / frames
    if name == "tracklet_median_s":
        lengths = pool.tracklet_lengths()
        return float(np.median(lengths)) if lengths.size else float("nan")
    if name == "speed_nan_share":
        return players["speed_nan_frames"] / slot_frames if slot_frames else float("nan")
    raise KeyError(name)


def _gate(spec, pool, calibration):
    """One gate as the report holds it: the pooled value, the Calibration's centre, the tolerance, its kind, the status.

    The tolerance is +- spec["tolerance"], or that share of the centre's size when the spec's kind is "relative". A gate
    in PENDING_GATES is "pending" whatever it measures: it is printed and never counted.
    """
    got, centre = _round(measured(spec["name"], pool)), _round(centre_of(spec["name"], calibration))
    limit = spec["tolerance"] if spec["kind"] == "absolute" else spec["tolerance"] * abs(centre)
    if spec["name"] in PENDING_GATES:
        status = PENDING
    else:
        status = PASS if abs(got - centre) <= limit else FAIL
    return {"name": spec["name"], "text": spec["text"], "measured": got, "centre": centre,
            "tolerance": spec["tolerance"], "kind": spec["kind"], "status": status}


def _check(name, got, centre, band):
    kind, width = band
    got, centre = _round(got), _round(centre)
    limit = width if kind == "absolute" else width * abs(centre)
    inside = abs(got - centre) <= limit
    return {"name": name, "measured": got, "centre": centre, "band": width, "kind": kind,
            "status": OK if inside else WARN}


def _checks(pool, calibration):
    """The warning checks, in a fixed order: nearest agreement, dead-ball visibility, then each yardstick's two."""
    window = calibration.public_window
    checks = [_check("nearest_agreement", pool.agreed / pool.judged if pool.judged else float("nan"),
                     window["nearest_agreement"], NEAREST_BAND),
              _check("dead_ball_visible_share",
                     pool.dead_visible / pool.dead_frames if pool.dead_frames else float("nan"),
                     window["dead_ball_visible_share"], DEAD_VISIBLE_BAND)]
    for width in calibration_module.YARDSTICK_WIDTHS:
        row, ours = pool.yardsticks[width], calibration.baseline_outputs[width]
        minutes = row["seconds"] / 60.0
        checks.append(_check("possessions_per_min:nearest_w{}".format(width), row["possessions"] / minutes,
                             ours["poss_per_min"], POSSESSIONS_BAND))
        checks.append(_check("d_share:nearest_w{}".format(width), row["d_frames"] / row["graded_frames"],
                             ours["d_share"], D_SHARE_BAND))
    return checks


def statistics(pool, calibration):
    """The benchmark's other statistics next to ours, as {name: {"benchmark", "bar_ath"}}: information, no band.

    The median gap and the missing share are the gates; the rest say what the profile does not (yet) model: where the
    gaps are, who is tracked, how long, with the wrong team, with what speeds.
    """
    scalars, ball, players = calibration.scalars, pool.ball, pool.players
    gaps, tracklets = pool.gap_durations(), pool.tracklet_lengths()
    ours_gaps, ours_tracklets = calibration.gap_durations_s, calibration.tracklet_lengths_s

    def quantile(values, level):
        return float(np.quantile(values, level)) if values.size else float("nan")

    rows = {
        "missing_share": (ball["missing_frames"] / ball["frames"], scalars["missing_share"]),
        "gaps_per_10k_frames": (ball["gaps"] / ball["frames"] * 1e4, scalars["gaps"] / scalars["frames"] * 1e4),
        "gap_s_p10": (quantile(gaps, 0.10), quantile(ours_gaps, 0.10)),
        "gap_s_median": (quantile(gaps, 0.50), quantile(ours_gaps, 0.50)),
        "gap_s_p90": (quantile(gaps, 0.90), quantile(ours_gaps, 0.90)),
        "near_line_onset_share": (ball["near_line_onsets"] / ball["interior_gaps"] if ball["interior_gaps"]
                                  else float("nan"), scalars["near_line_share"]),
        "present_mean": (measured("present_mean", pool), centre_of("present_mean", calibration)),
        "present_share_22": (measured("present_share_22", pool), centre_of("present_share_22", calibration)),
        "present_share_le14": (measured("present_share_le14", pool), centre_of("present_share_le14", calibration)),
        "tracklet_s_median": (quantile(tracklets, 0.50), quantile(ours_tracklets, 0.50)),
        "wrong_team_share": (measured("wrong_team_share", pool), centre_of("wrong_team_share", calibration)),
        "speed_nan_share": (measured("speed_nan_share", pool), centre_of("speed_nan_share", calibration)),
        "jitter_rms_m": (float(np.sqrt(players["jitter_squares"] / players["jitter_triples"]))
                         if players["jitter_triples"] else float("nan"), calibration.player_jitter_rms_m),
    }
    spread = (min(pool.missing_shares), max(pool.missing_shares))
    rows["missing_share_min_over_match_seeds"] = (spread[0], None)
    rows["missing_share_max_over_match_seeds"] = (spread[1], None)
    return {name: {"benchmark": _round(benchmark), "bar_ath": None if ours is None else _round(ours)}
            for name, (benchmark, ours) in rows.items()}


def _round(value):
    return round(float(value), DECIMALS)


# -- the printout and the file ----------------------------------------------------------------------------------------

def render(result):
    """The lines that the command prints: the gates, the checks, the statistics, and the verdict."""
    lines = ["",
             "fidelity of profile {} (version {})   {} matches   seeds {}   calibration {}...".format(
                 result["profile"]["name"], result["profile"]["version"], len(result["matches"]),
                 ",".join(str(seed) for seed in result["seeds"]), result["calibration"]["sha256"][:12]),
             "", "gates (centre = our Calibration's value, measured on the kick-off spans; the plan's tolerance)",
             "  {:<18}{:>12}{:>12}{:>12}   {}".format("gate", "measured", "centre", "+-", "status")]
    for gate in result["gates"]:
        tolerance = "{:g}%".format(100 * gate["tolerance"]) if gate["kind"] == "relative" else _cell(gate["tolerance"])
        lines.append("  {:<18}{:>12}{:>12}{:>12}   {}   ({})".format(
            gate["name"], _cell(gate["measured"]), _cell(gate["centre"]), tolerance, gate["status"], gate["text"]))
    lines += ["", "checks (pre-registered warning bands; the degradation is never adjusted towards them)",
              "  {:<32}{:>10}{:>10}{:>10}   {}".format("check", "measured", "ours", "band", "status")]
    for check in result["checks"]:
        band = "+-{:g}{}".format(100 * check["band"] if check["kind"] == "relative" else check["band"],
                                 "%" if check["kind"] == "relative" else "")
        lines.append("  {:<32}{:>10}{:>10}{:>10}   {}".format(
            check["name"], _cell(check["measured"]), _cell(check["centre"]), band, check["status"]))
    lines += ["", "statistics, benchmark against our match (information)",
              "  {:<38}{:>12}{:>12}".format("statistic", "benchmark", "ours")]
    for name, row in result["statistics"].items():
        lines.append("  {:<38}{:>12}{:>12}".format(
            name, _cell(row["benchmark"]), _cell(row["bar_ath"])))
    halves = [draw["source_half"] for draw in result["draws"]]
    pending = [gate["name"] for gate in result["gates"] if gate["status"] == PENDING]
    if result["players_draws"]:
        lines += ["", "players' draws, pooled over matches, seeds and halves: " + ", ".join(
            "{} {:,}".format(name, count) for name, count in result["players_draws"].items())]
    lines += ["", "draws: {} half-draws, {} from our first half and {} from our second".format(
        len(halves), halves.count(1), halves.count(2))]
    if pending:
        lines.append(PENDING_NOTE.format(", ".join(pending)))
    lines.append("{} gates {}{}; {} of {} checks warn".format(
        len(result["gates"]), "all pass" if result["passed"] else "DO NOT ALL PASS",
        " ({} pending, not counted)".format(len(pending)) if pending else "", result["warnings"],
        len(result["checks"])))
    return lines


def _cell(value):
    """A number as the tables print it; "-" for what there is none of (a share of nothing is None in a report)."""
    return "-" if value is None else "{:.4f}".format(value)


def report_path(data_dir, profile, match_ids=None, seeds=None):
    """<data dir>/runs/benchmark/fidelity_<profile>[.m-<ids>][.s<n>].json, by the runner's rule (run.name_suffixes)."""
    name = ".".join(["fidelity_" + profile] + run.name_suffixes(profile, match_ids, seeds)) + ".json"
    return os.path.join(data_dir, *FILE_FOLDER, name)


def write(result, data_dir=None):
    """Write the report under `data_dir` (default games.DATA_DIR) with run.write_json; the path."""
    path = report_path(data_dir or games.DATA_DIR, result["profile"]["name"], result["matches"], len(result["seeds"]))
    run.write_json(path, result)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure how far a degraded profile looks like our match: gates (exit 1 if one fails), warnings.")
    parser.add_argument("--profile", default=run.DEFAULT_PROFILE,
                        help="the profile to measure: {} (default {})".format(", ".join(sorted(GATES)),
                                                                              run.DEFAULT_PROFILE))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="degradation seeds per match (default {})".format(DEFAULT_SEEDS))
    parser.add_argument("--matches", nargs="+", metavar="ID", help="the matches to measure (default: all 7)")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--no-write", action="store_true", help="print only; write no report")
    args = parser.parse_args(argv)
    try:
        result = report(args.profile, seeds=args.seeds, match_ids=args.matches, data_dir=args.data_dir)
        path = None if args.no_write else write(result, args.data_dir)
    except (ConfigError, FidelityError, run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    for line in render(result):
        print(line)
    print("")
    print("report    {}".format("not written (--no-write)" if path is None else run.shown(path)))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
