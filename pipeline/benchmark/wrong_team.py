"""P21 step 0: do wrong-team slots cluster at the ball? One run on our match, aggregates only.

    python pipeline/benchmark/wrong_team.py [--data-dir D] [--out PATH]

    out = measure(inputs)        # {"ratio": {...}, "surplus": {...}, "onset": {...}}

Why it exists. T32: a slot keeps its team in the metadata while it follows a player of the other side, so a team shows
more than 11 slots (on at least 10.8% of the graded frames, A07). `controlled_touch` takes the team of the nearest slot
at confidence 0.93, so a swap next to the ball injects a confident wrong team, and the benchmark's own wrong-team
episodes are placed at the closest home-away pair anywhere on the pitch, not at the ball (P21). If our swaps do not
cluster at the ball, the benchmark's bound (A14: all of `tracking`'s player degradation costs `main.toml` 0.0129) applies
and P21 is dropped without the ceiling run.

The measure (the advisor's, 2026-10-31). The candidate frames are `controlled_touch`'s rule without its duel test: a
graded frame, the ball seen, the nearest present slot within RADIUS metres, moving with the ball (|v_ball - v_nearest| at
most MAX_REL_SPEED), velocities from successive positions and never the speed channels (T31). `over_eleven` says a team
shows more than TEAM_SIZE present slots. The ratio is the share of candidate frames that are over 11, over the share of
all graded frames that are. **A ratio below RULE (1.5) drops P21; at or above it the result is reported and the proposal's
ceiling run is not run, since it would not measure swaps at the ball.** Two things pull the ratio towards 1 and so a low
one does not prove that swaps never start at the ball: a surplus is a state that persists after its swap (median 2.7 s)
while the ball moves on, and a set piece crowds players away from the ball.

Beside it, reported and deciding nothing: `surplus`, T32's stricter definition (one team over 11 and the other under 11,
which a real match cannot show, as `calibration.player_stats` counts it), and `onset` (the advisor's D3): the share of
the over-11 runs' first frames that lie within ONSET_WINDOW_S of a candidate frame, over the share of graded frames that
do. It tells the event from the state. A run that begins on a block's first frame is carried in, not begun, and is not
counted. Nothing here prints or writes a frame: counts, shares and ratios. The tool stays after this lane, because P19's
transfer check reports its ratio.
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

import games  # noqa: E402
from pipeline.evidence import kinematics  # noqa: E402

# Params
RADIUS = 1.5                    # metres: main.toml's controlled_touch radius (a check holds them equal)
MAX_REL_SPEED = 6.0             # metres per second: main.toml's controlled_touch max_rel_speed
TEAM_SIZE = 11                  # a team with more present slots than this has a slot on a player of the other side
RULE = 1.5                      # the advisor's: a ratio below it drops P21
ONSET_WINDOW_S = 1.0            # D3: an onset is near a candidate frame within this many seconds
HOME = 0                        # slot_team's code for the home team
OUT_DIR = ("runs", "wrong_team")


def _counts(present, slot_team):
    """(home, away) present slots per frame."""
    present = np.asarray(present, dtype=bool)
    is_home = np.asarray(slot_team) == HOME
    return present[:, is_home].sum(axis=1), present[:, ~is_home].sum(axis=1)


def over_eleven(present, slot_team):
    """(n,) bool: either team has more than TEAM_SIZE present slots."""
    home, away = _counts(present, slot_team)
    return (home > TEAM_SIZE) | (away > TEAM_SIZE)


def surplus(present, slot_team):
    """(n,) bool: T32's stricter rule, one team over TEAM_SIZE present slots and the other under it."""
    home, away = _counts(present, slot_team)
    return ((home > TEAM_SIZE) & (away < TEAM_SIZE)) | ((away > TEAM_SIZE) & (home < TEAM_SIZE))


def graded_mask(inputs):
    mask = np.zeros(inputs.atd.n_frames, dtype=bool)
    for start, stop in kinematics.blocks(inputs):
        mask[start:stop] = True
    return mask


def candidates(inputs, *, radius=RADIUS, max_rel_speed=MAX_REL_SPEED, duel_radius=None):
    """(n,) bool: graded, ball seen, the nearest present slot within `radius` and moving with the ball.

    The rule of `controlled_touch` (velocities from positions, the lowest slot index on a tie) without its duel test:
    `duel_radius=None` skips it, a number applies it as `controlled_touch` does (no slot of the nearest slot's opposing
    team within it).
    """
    ball = kinematics.ball_track(inputs)
    players = kinematics.player_track(inputs)
    frames = np.arange(inputs.atd.n_frames)
    distance = kinematics.ball_distances(players.xy_m, ball.xy_m)
    distance[np.isnan(distance)] = np.inf                  # absent slot, absent ball, outside the blocks: never near
    nearest = np.argmin(distance, axis=1)
    nearest_distance = distance[frames, nearest]
    delta = ball.vel - players.vel[frames, nearest]
    with np.errstate(invalid="ignore"):                    # NaN velocities compare False: not a candidate
        near = ball.visible & (nearest_distance <= radius) & (np.hypot(delta[:, 0], delta[:, 1]) <= max_rel_speed)
        if duel_radius is None:
            return near
        nearest_is_home = inputs.atd.slot_team[nearest] == HOME
        rival = np.where((inputs.atd.slot_team == HOME)[None, :] != nearest_is_home[:, None], distance, np.inf)
        return near & (rival.min(axis=1) > duel_radius)


def ratio(over, candidate, graded):
    """The advisor's ratio on three (n,) masks; only graded frames count.

    {graded_frames, candidate_frames, over_graded_frames, over_candidate_frames, share_candidates, share_graded, ratio,
    verdict}: share_candidates is the share of candidate frames that are over, share_graded the share of graded frames
    that are, ratio their quotient (from the counts, so that an exact 1.5 is exact; None with no candidate or no over
    frame), verdict "P21 dropped" below RULE, "report only" from it, "undefined" with no ratio.
    """
    graded = np.asarray(graded, dtype=bool)
    over = np.asarray(over, dtype=bool) & graded
    candidate = np.asarray(candidate, dtype=bool) & graded
    frames, picked = int(graded.sum()), int(candidate.sum())
    over_all, over_picked = int(over.sum()), int((over & candidate).sum())
    value = over_picked * frames / (picked * over_all) if picked and over_all else None
    return {"graded_frames": frames, "candidate_frames": picked, "over_graded_frames": over_all,
            "over_candidate_frames": over_picked,
            "share_candidates": over_picked / picked if picked else None,
            "share_graded": over_all / frames if frames else None, "ratio": value,
            "verdict": "undefined" if value is None else "P21 dropped" if value < RULE else "report only"}


def onset_ratio(over, candidate, blocks, window):
    """D3: how near the starts of over-11 runs are to candidate frames, against how near graded frames are.

    {onsets, onsets_near, share_onsets_near, share_graded_near, ratio}: an onset is the first frame of a run of `over`
    inside one of `blocks` ((start, stop) pairs), except a run that begins on the block's first frame; near means within
    `window` frames of a candidate frame of the same block. The ratio is None with no onset or no near frame. Report only.
    """
    over, candidate = np.asarray(over, dtype=bool), np.asarray(candidate, dtype=bool)
    onsets = near_onsets = graded = near_graded = 0
    for start, stop in blocks:
        if stop <= start:
            continue
        is_candidate = candidate[start:stop]
        total = np.concatenate([[0], np.cumsum(is_candidate)])
        frames = np.arange(stop - start)
        near = total[np.minimum(frames + window + 1, stop - start)] - total[np.maximum(frames - window, 0)] > 0
        graded += stop - start
        near_graded += int(near.sum())
        first, _ = kinematics.runs(over[start:stop])
        first = first[first > 0]
        onsets += len(first)
        near_onsets += int(near[first].sum())
    share_onsets = near_onsets / onsets if onsets else None
    share_graded = near_graded / graded if graded else None
    value = share_onsets / share_graded if share_onsets is not None and share_graded else None
    return {"onsets": onsets, "onsets_near": near_onsets, "share_onsets_near": share_onsets,
            "share_graded_near": share_graded, "ratio": value}


def measure(inputs):
    """{ratio, surplus, onset} on `inputs`: the advisor's ratio on `over_eleven`, the same on `surplus` and D3's onsets.

    `surplus` and `onset` are reported and decide nothing; only `ratio`'s verdict is the rule. Counts, shares and ratios.
    """
    players = kinematics.player_track(inputs)
    graded = graded_mask(inputs)
    candidate = candidates(inputs)
    over = over_eleven(players.present, inputs.atd.slot_team)
    window = int(round(ONSET_WINDOW_S * inputs.fps))
    stricter = ratio(surplus(players.present, inputs.atd.slot_team), candidate, graded)
    stricter["verdict"] = "report only"                     # only `ratio` decides (A16 part 1)
    return {"ratio": ratio(over, candidate, graded), "surplus": stricter,
            "onset": onset_ratio(over, candidate, kinematics.blocks(inputs), window)}


def render(result):
    """The printed lines: counts, shares and ratios, no frame."""
    def number(value):
        return "-" if value is None else "{:.4f}".format(value)
    lines = []
    for name in ("ratio", "surplus"):
        cell = result[name]
        lines.append("{:8} {:,} candidate frames of {:,} graded; over 11 on {} of the candidates against {} of all "
                     "graded: ratio {} ({})".format(
                         name, cell["candidate_frames"], cell["graded_frames"], number(cell["share_candidates"]),
                         number(cell["share_graded"]), number(cell["ratio"]), cell["verdict"]))
    onset = result["onset"]
    lines.append("onset    {:,} runs begin, {} of them within {} s of a candidate against {} of graded frames: "
                 "ratio {} (report only)".format(onset["onsets"], number(onset["share_onsets_near"]), ONSET_WINDOW_S,
                                                 number(onset["share_graded_near"]), number(onset["ratio"])))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P21 step 0: the share of candidate frames with a team over 11 slots, against all graded frames "
                    "(HEAVY: loads our tracking; aggregates only).")
    parser.add_argument("--data-dir", help="the data bundle (default: data/, or $MTAH_DATA_DIR)")
    parser.add_argument("--out", help="the JSON to write (default: <data dir>/runs/wrong_team/ratio.json)")
    args = parser.parse_args(argv)

    from pipeline.benchmark import run
    from pipeline.inputs import Inputs
    try:
        result = measure(Inputs(data_dir=args.data_dir))
        for line in render(result):
            print(line)
        out = args.out or os.path.join(args.data_dir or games.DATA_DIR, *OUT_DIR, "ratio.json")
        run.write_json(out, run.plain(result))
        print("wrote     {}".format(run.shown(out)))
    except (run.BenchmarkError, OSError) as error:
        print("FAILED    {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
