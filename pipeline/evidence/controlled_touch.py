"""P04 -- who controls the ball: home evidence from control episodes only, NaN everywhere else.

    [[evidence]]
    kind = "controlled_touch"
    radius = 2.0           # metres: the nearest player must be this close
    max_rel_speed = 4.0    # m/s: and move with the ball (|v_ball - v_player|); inf switches the test off
    min_frames = 5         # an episode shorter than this is dropped
    confidence = 0.9       # home on a home episode (1 - confidence on an away one), in (0.5, 1.0]
    duel_radius = 1.0      # optional: the other team must be farther than this, or the ball is contested
    hold = 0.5             # optional, in [0.5, confidence]: the weak vote of the last team; 0.5 = off
    hold_curve = "pipeline/models/hold_curve.json"   # optional: the hold follows this measured curve (needs hold 0.5)
    hold_scale = 1.0       # optional, in (0, 1], only with hold_curve: 1 = the curve, less = shrunk towards 0.5

`nearest_player` names a team on every frame the ball is visible and is wrong on 22% of the in-play ones: a pass in
flight, a duel or a deflection is not a possession (docs/traps.md T14). The rules of football in the data's terms:
possession changes only when the other team *controls* the ball, and a loose or contested ball stays with the last team.
This source therefore speaks only inside control episodes and says nothing between them, and the decoder, which holds
its state through a stretch without evidence, supplies the stickiness.

A frame is a candidate when the ball is visible, the nearest present slot p (either team) is within `radius`, the other
team's nearest slot is farther than `duel_radius` (or absent), and the ball and p move together: |v_ball - v_p| is at
most `max_rel_speed`, both velocities from positions (kinematics.py; the ATD's own speed channels are not read, T31).
The zones are Vidal-Codina et al. (2022)'s: a ball within a couple of metres of one player and no other, moving as he
does, is his; two teams inside the duel radius make a contested ball, which is no evidence. A NaN relative speed (a
slot or the ball seen on one frame only) is not a candidate. An episode is a maximal run of consecutive candidate frames
with one team, kept when it lasts at least `min_frames`; `home` is `confidence` on a home episode and `1 - confidence`
on an away one.

Back-dating (T03): the evidence starts on the episode's *first* frame, not when `min_frames` of it have been seen. The
rule looks at the whole episode, so confirmation costs no latency, and a change of team decoded from it lands where
control began. Nothing is added on the end either.

`hold` > 0.5 gives the stretch between episodes a weak vote for the last controlling team: on a frame with the ball
visible, outside every kept episode and after the block's first one, `home` is `hold` (home last) or `1 - hold` (away
last). With a small `hold` the decoder keeps the old team through a pass in flight, so a turnover lands on the new
team's first control frame; without it the change lands on the first frame after the old team's last one, the HMM's tie
rule (pipeline/decode/hmm.py). A frame without a ball stays NaN. `hold` = 0.5 is off and is written as NaN, not 0.5:
NaN is not 0.5 (pipeline/evidence/README.md).

`hold_curve` replaces the constant by a measured one (P29, pipeline/benchmark/hold_stats.py): the last controller's team
keeps the ball less often the longer ago its last touch was, and less often again if the ball was hidden since. The file
holds, for the two cases (`no_gap`, `gap`: the ball was hidden on some frame since the last kept one) and the ten bins of
`HOLD_BINS_S` (seconds since that frame), the share of hold frames on which the last controller's team had the ball,
measured on the benchmark's other matches. On a hold frame the vote for that team is q = 0.5 + hold_scale * (p - 0.5)
with p the curve's cell, clipped to [1 - confidence, confidence], so it never out-votes a control episode; `home` is q
if that team is home, else 1 - q. The curve is `folds[inputs.game]` (built without that match) if the file has one, else
`full`. `hold_curve` and a `hold` other than 0.5 would say the same thing twice and are refused; so is a file that was
measured for another control rule or another bin layout. Without `hold_curve` nothing changes.

Everything is per graded block (kinematics.blocks): an episode is cut at the block's edge, a hold never crosses it, and
frames outside the blocks are NaN. `alive` is NaN everywhere: whether the ball is in play is P05's question.
`confidence` is measured, not tuned (docs/proposals.md P04).
"""

import collections
import math
import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import kinematics, trees
from starter.baseline import HOME


HOLD_BINS_S = (0.0, 0.4, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0)   # seconds since the last kept frame; last is open

HoldState = collections.namedtuple("HoldState", "last_is_home elapsed gap")


NAME = "controlled_touch"
CURVE_FORMAT = 1
CURVE_ROWS = ("no_gap", "gap")                         # the order of the table's rows, as hold_stats writes them
CURVE_COMMAND = "python pipeline/benchmark/hold_stats.py --write"
_CURVES = {}                                           # absolute path -> (stamp, artifact), for trees.load_artifact


def compute(run, *, radius, max_rel_speed, min_frames, confidence, duel_radius=1.0, hold=0.5, hold_curve=None,
            hold_scale=1.0):
    """Evidence(home = confidence / 1 - confidence in kept control episodes, hold / 1 - hold between them (or the
    clipped curve of `hold_curve`, scaled by `hold_scale`), else NaN; alive = NaN).

    ValueError for a parameter it refuses, for `hold_curve` together with a `hold` above 0.5, for a `hold_scale` other
    than 1.0 without `hold_curve`, and for a curve file that is unreadable, not a hold curve, or measured for another
    control rule."""
    radius, max_rel_speed, min_frames, duel_radius = _checked(radius, max_rel_speed, min_frames, duel_radius)
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real) or not 0.5 < confidence <= 1.0:
        raise ValueError("controlled_touch: confidence must be a number in (0.5, 1.0] (0.5 would say nothing), "
                         "got {!r}".format(confidence))
    confidence = float(confidence)
    if isinstance(hold, bool) or not isinstance(hold, numbers.Real) or not 0.5 <= hold <= confidence:
        raise ValueError("controlled_touch: hold must be a number in [0.5, confidence = {}] (0.5 = no hold; more than "
                         "confidence would out-vote a control episode), got {!r}".format(confidence, hold))
    hold = float(hold)
    if isinstance(hold_scale, bool) or not isinstance(hold_scale, numbers.Real) or not 0.0 < hold_scale <= 1.0:
        raise ValueError("controlled_touch: hold_scale must be a number in (0, 1] (1 = the curve as measured, less "
                         "shrinks it towards 0.5), got {!r}".format(hold_scale))
    hold_scale = float(hold_scale)
    if hold_curve is None and hold_scale != 1.0:
        raise ValueError("controlled_touch: hold_scale {} scales the hold curve, and no hold_curve is given".format(
            hold_scale))
    if hold_curve is not None and hold != 0.5:
        raise ValueError("controlled_touch: hold_curve gives the hold its own votes, so hold must stay 0.5 (off) with "
                         "it, got hold = {!r}".format(hold))

    inputs = run.inputs
    n_frames = inputs.atd.n_frames
    table = None
    if hold_curve is not None:
        if not isinstance(hold_curve, str) or not hold_curve:
            raise ValueError("controlled_touch: hold_curve must be the path of a hold curve file relative to the "
                             "repository root, such as \"pipeline/models/hold_curve.json\", got {!r}".format(
                                 hold_curve))
        path = trees.model_path(hold_curve, NAME, "pipeline/models/hold_curve.json")
        artifact = _curve_artifact(path, dict(radius=radius, max_rel_speed=max_rel_speed, min_frames=min_frames,
                                              duel_radius=duel_radius))
        folds = artifact["folds"]
        game = getattr(inputs, "game", None)
        curve = folds[game] if isinstance(game, str) and game in folds else artifact["full"]
        table = np.array([curve[name] for name in CURVE_ROWS], dtype=np.float64)
    candidate, nearest_is_home, visible = _candidates(inputs, radius, max_rel_speed, duel_radius)
    home = no_opinion(n_frames)
    for start, stop in kinematics.blocks(inputs):
        if stop > start:
            if table is None:
                home[start:stop] = _block_home(candidate[start:stop], nearest_is_home[start:stop],
                                               visible[start:stop], min_frames, confidence, hold)
            else:
                home[start:stop] = _block_curve_home(candidate[start:stop], nearest_is_home[start:stop],
                                                     visible[start:stop], min_frames, confidence, table, hold_scale,
                                                     inputs.fps)
    return Evidence(alive=no_opinion(n_frames), home=home)


def hold_state(run, *, radius, max_rel_speed, min_frames, duel_radius=1.0):
    """HoldState of every frame: what the hold's vote is made of (P29), on the frames `compute` gives a hold.

    The hold frames are those with the ball visible, outside every kept episode and after the block's first one, the
    same `compute(..., hold > 0.5)` votes on. There `elapsed` is the number of frames since the last kept frame (>= 1)
    and `last_is_home` that frame's team; `gap` says the ball was hidden on some frame in between. Elsewhere `elapsed`
    is -1 and `gap` is False; `last_is_home` is the last kept frame's team on every frame after a block's first kept
    frame (False before it) and means something only where `elapsed` >= 0. The same checks on the parameters as
    `compute`; ValueError otherwise.
    """
    radius, max_rel_speed, min_frames, duel_radius = _checked(radius, max_rel_speed, min_frames, duel_radius)
    inputs = run.inputs
    n_frames = inputs.atd.n_frames
    candidate, nearest_is_home, visible = _candidates(inputs, radius, max_rel_speed, duel_radius)
    last_is_home = np.zeros(n_frames, dtype=bool)
    elapsed = np.full(n_frames, -1, dtype=np.int64)
    gap = np.zeros(n_frames, dtype=bool)
    for start, stop in kinematics.blocks(inputs):
        if stop > start:
            kept = _block_kept(candidate[start:stop], nearest_is_home[start:stop], min_frames)
            last, elapsed[start:stop], gap[start:stop] = _block_hold_state(kept, visible[start:stop])
            last_is_home[start:stop] = nearest_is_home[start:stop][np.maximum(last, 0)] & (last >= 0)
    return HoldState(last_is_home=last_is_home, elapsed=elapsed, gap=gap)


def hold_bin(elapsed, fps):
    """The bin (0 to len(HOLD_BINS_S) - 1) of each elapsed frame count, -1 where `elapsed` is negative.

    The edges are rint(HOLD_BINS_S * fps) frames; a bin holds its lower edge, the last is open.
    """
    edges = np.rint(np.asarray(HOLD_BINS_S) * float(fps)).astype(np.int64)
    elapsed = np.asarray(elapsed)
    return np.where(elapsed < 0, -1, np.searchsorted(edges, elapsed, side="right") - 1).astype(np.int64)


def _checked(radius, max_rel_speed, min_frames, duel_radius):
    """The four parameters of the control rule, checked and as (float, float, int, float); ValueError otherwise."""
    radius = _positive("radius", radius, "metres from the ball", allow_inf=False)
    duel_radius = _positive("duel_radius", duel_radius, "metres from the ball", allow_inf=False)
    max_rel_speed = _positive("max_rel_speed", max_rel_speed, "metres per second; inf turns the test off",
                              allow_inf=True)
    if isinstance(min_frames, bool) or not isinstance(min_frames, numbers.Integral) or min_frames < 1:
        raise ValueError("controlled_touch: min_frames must be a whole number of frames >= 1, got {!r}".format(
            min_frames))
    return radius, max_rel_speed, int(min_frames), duel_radius


def _candidates(inputs, radius, max_rel_speed, duel_radius):
    """(candidate, nearest slot is home, ball visible), each of full length: what compute and hold_state start from."""
    atd = inputs.atd
    ball = kinematics.ball_track(inputs)
    players = kinematics.player_track(inputs)
    frames = np.arange(atd.n_frames)

    distance = kinematics.ball_distances(players.xy_m, ball.xy_m)
    distance[np.isnan(distance)] = np.inf                  # absent slot, absent ball, outside the blocks: never near
    nearest = np.argmin(distance, axis=1)                  # the lowest slot index on a tie, as nearest_player
    nearest_distance = distance[frames, nearest]
    nearest_is_home = atd.slot_team[nearest] == HOME
    rival = np.where((atd.slot_team == HOME)[None, :] != nearest_is_home[:, None], distance, np.inf).min(axis=1)
    delta = ball.vel - players.vel[frames, nearest]
    with np.errstate(invalid="ignore"):                    # NaN velocities compare False: not a candidate
        candidate = (ball.visible & (nearest_distance <= radius) & (rival > duel_radius)
                     & (np.hypot(delta[:, 0], delta[:, 1]) <= max_rel_speed))
    return candidate, nearest_is_home, ball.visible


def _block_kept(candidate, is_home, min_frames):
    """One block's kept mask: the frames of the episodes (runs of one team's candidates) of at least min_frames."""
    kept = np.zeros(len(candidate), dtype=bool)
    for team_is_home in (True, False):
        starts, stops = kinematics.runs(candidate & (is_home == team_is_home))
        for first, last in zip(starts, stops):
            if last - first >= min_frames:
                kept[first:last] = True
    return kept


def _block_home(candidate, is_home, visible, min_frames, confidence, hold):
    """One block's home evidence: kept episodes, then the hold between them."""
    m = len(candidate)
    kept = _block_kept(candidate, is_home, min_frames)
    home = np.full(m, np.nan)
    home[kept] = np.where(is_home[kept], confidence, 1.0 - confidence)
    if hold > 0.5:
        # The index of the last kept frame at or before each frame, -1 before the first: its team is the last team.
        last = np.maximum.accumulate(np.where(kept, np.arange(m), -1))
        between = visible & ~kept & (last >= 0)
        home[between] = np.where(is_home[last[between]], hold, 1.0 - hold)
    return home


def _block_curve_home(candidate, is_home, visible, min_frames, confidence, table, scale, fps):
    """One block's home evidence with the hold following `table` ([no_gap, gap] x bins of p): kept episodes at
    confidence, hold frames at q = 0.5 + scale * (p - 0.5) clipped to [1 - confidence, confidence] for the last team."""
    kept = _block_kept(candidate, is_home, min_frames)
    home = np.full(len(candidate), np.nan)
    home[kept] = np.where(is_home[kept], confidence, 1.0 - confidence)
    last, elapsed, gap = _block_hold_state(kept, visible)
    between = elapsed >= 0
    p = table[gap[between].astype(np.int64), hold_bin(elapsed[between], fps)]
    q = np.clip(0.5 + scale * (p - 0.5), 1.0 - confidence, confidence)
    home[between] = np.where(is_home[last[between]], q, 1.0 - q)
    return home


def _curve_artifact(path, touch):
    """The hold curve file at `path`, read once and checked as a curve measured for the control rule `touch` (a dict of
    radius, max_rel_speed, min_frames, duel_radius); ValueError naming the file and the command otherwise."""
    artifact = trees.load_artifact(
        path, NAME, _check_curve, _CURVES, unreadable="measure it with `{}`".format(CURVE_COMMAND),
        not_json="measure it again with `{}`".format(CURVE_COMMAND))
    named = artifact["touch"]
    if any(float(named[key]) != float(touch[key]) for key in touch):               # "inf" reads as float("inf")
        raise ValueError("controlled_touch: the hold curve {} was measured for the control rule {} and this source "
                         "runs {}; measure it again with `{}`".format(path, named, touch, CURVE_COMMAND))
    return artifact


def _check_curve(artifact, path):
    """ValueError unless `artifact` is a format-1 hold curve with ten numbers in [0, 1] per row, `full` and each fold."""
    def refuse(reason):
        raise ValueError("controlled_touch: {} is not a usable hold curve ({}); measure it again with `{}`".format(
            path, reason, CURVE_COMMAND))

    if not isinstance(artifact, dict):
        refuse("the top level is not an object")
    if artifact.get("format") != CURVE_FORMAT or artifact.get("kind") != "hold_curve":
        refuse("format {!r} and kind {!r}, expected format {} and kind \"hold_curve\"".format(
            artifact.get("format"), artifact.get("kind"), CURVE_FORMAT))
    if artifact.get("bins_s") != list(HOLD_BINS_S):
        refuse("its bins_s are {!r} and the code's are {}".format(artifact.get("bins_s"), list(HOLD_BINS_S)))
    touch = artifact.get("touch")
    if not isinstance(touch, dict) or set(touch) != {"radius", "max_rel_speed", "min_frames", "duel_radius"}:
        refuse("its touch must name radius, max_rel_speed, min_frames and duel_radius, got {!r}".format(touch))
    for key, value in touch.items():
        if isinstance(value, bool) or not (isinstance(value, numbers.Real) or value == "inf"):
            refuse("touch {} is {!r}, not a number".format(key, value))
    folds = artifact.get("folds")
    if not isinstance(folds, dict):
        refuse("its folds are not an object")
    for where, curve in [("full", artifact.get("full"))] + [("folds[{}]".format(game), folds[game]) for game in folds]:
        if not isinstance(curve, dict) or set(curve) != set(CURVE_ROWS):
            refuse("{} must have exactly the rows {}".format(where, ", ".join(CURVE_ROWS)))
        for name in CURVE_ROWS:
            cells = curve[name]
            if not isinstance(cells, list) or len(cells) != len(HOLD_BINS_S) or any(
                    isinstance(cell, bool) or not isinstance(cell, numbers.Real) or not 0.0 <= cell <= 1.0
                    or not math.isfinite(cell) for cell in cells):
                refuse("{} {} must be {} numbers in [0, 1]".format(where, name, len(HOLD_BINS_S)))


def _block_hold_state(kept, visible):
    """(last, elapsed, gap) of one block: the index of the last kept frame at or before each frame (-1 before the
    first), the frames since it on a hold frame (visible, not kept, after the first kept frame; else -1), and whether
    the ball was hidden on any frame since it on a hold frame (else False)."""
    index = np.arange(len(kept))
    last = np.maximum.accumulate(np.where(kept, index, -1))
    between = visible & ~kept & (last >= 0)
    elapsed = np.where(between, index - last, -1)
    hidden = np.cumsum(~visible)
    gap = between & (hidden - hidden[np.maximum(last, 0)] > 0)
    return last, elapsed, gap


def _positive(name, value, unit, allow_inf):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not value > 0 or (
            not allow_inf and not np.isfinite(value)):
        raise ValueError("controlled_touch: {} must be a positive {}number ({}), got {!r}".format(
            name, "" if allow_inf else "finite ", unit, value))
    return float(value)
