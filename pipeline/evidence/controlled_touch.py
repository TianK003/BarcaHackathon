"""P04 -- who controls the ball: home evidence from control episodes only, NaN everywhere else.

    [[evidence]]
    kind = "controlled_touch"
    radius = 2.0           # metres: the nearest player must be this close
    max_rel_speed = 4.0    # m/s: and move with the ball (|v_ball - v_player|); inf switches the test off
    min_frames = 5         # an episode shorter than this is dropped
    confidence = 0.9       # home on a home episode (1 - confidence on an away one), in (0.5, 1.0]
    duel_radius = 1.0      # optional: the other team must be farther than this, or the ball is contested
    hold = 0.5             # optional, in [0.5, confidence]: the weak vote of the last team; 0.5 = off
    backdate_tau = 3.0     # optional, m/s: P28, the new team's vote from its first touch; absent = off
    backdate_confidence = 0.8   # with backdate_tau: the vote on the back-dated frames, in [hold, confidence], > 0.5
    backdate_cap = 25      # optional, frames: at most this many back-dated frames before an episode; absent = no cap

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

Back-dating to the first touch (P28, docs/superpowers/specs/2026-11-10-p28-touch-backdating-design.md; off unless
`backdate_tau` is given, and the output is then what it was before P28, byte for byte). A first-time pass, a header or
a deflection is no episode, so a team that wins the ball with a touch and controls it a moment later is heard only
from the control on. A *touch* (touches) is dated at frame t when the ball is visible on every frame of
[t - 7, t + 7] in t's block, the spatial test above holds at t (nearest slot within `radius`, the other team's nearest
beyond `duel_radius`; no speed test) and the ball's velocity changes: |v(t + 4) - v(t - 4)| >= `backdate_tau`, the
norm of the difference of kinematics' two velocity vectors. The frames whose change is that large form maximal runs in
a block, and a run gives at most one touch: at the frame of its largest change, the earliest on a tie, if the window
and the spatial test hold there, by the team of the nearest slot there. The rule: a kept episode of team T whose
previous kept episode in the block is the other team O's gets T's vote from the earliest touch of T between the two
from which the ball is visible on every frame up to T's episode and after which no touch of O is dated. The frames from
that touch to the frame before T's episode say T with `backdate_confidence`, over the hold; with `backdate_cap` only the
last `backdate_cap` of them do. A touch never makes or lengthens an episode and never counts towards `min_frames`.
`backdate_confidence` is measured (pipeline/benchmark/touch_backdate_stats.py), not tuned.

Everything is per graded block (kinematics.blocks): an episode is cut at the block's edge, a hold never crosses it, a
touch's window never leaves it, and frames outside the blocks are NaN. `alive` is NaN everywhere: whether the ball is
in play is P05's question. `confidence` is measured, not tuned (docs/proposals.md P04).
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import kinematics
from starter.baseline import HOME

# Params
# P28's touch (the spec's section "The touch": docs/superpowers/specs/2026-11-10-p28-touch-backdating-design.md), in
# frames, declared there before any run and not tuned. A touch at t needs the ball on every frame of
# [t - TOUCH_WINDOW, t + TOUCH_WINDOW], and its velocity is compared TOUCH_LAG frames either side of t. With
# kinematics' 7-frame derivative v(t - 4) reads the frames t - 7 ... t - 1 and v(t + 4) reads t + 1 ... t + 7: the two
# share no frame, neither reads t, and both lie in the window.
TOUCH_WINDOW = 7
TOUCH_LAG = 4


def compute(run, *, radius, max_rel_speed, min_frames, confidence, duel_radius=1.0, hold=0.5, backdate_tau=None,
            backdate_confidence=None, backdate_cap=None):
    """Evidence(home = confidence / 1 - confidence in kept control episodes, hold / 1 - hold between them, else NaN;
    alive = NaN). With backdate_tau, the frames from a team's first touch to its episode say that team with
    backdate_confidence (the module docstring); without it nothing of P28 runs."""
    radius = _positive("radius", radius, "metres from the ball", allow_inf=False)
    duel_radius = _positive("duel_radius", duel_radius, "metres from the ball", allow_inf=False)
    max_rel_speed = _positive("max_rel_speed", max_rel_speed, "metres per second; inf turns the test off",
                              allow_inf=True)
    if isinstance(min_frames, bool) or not isinstance(min_frames, numbers.Integral) or min_frames < 1:
        raise ValueError("controlled_touch: min_frames must be a whole number of frames >= 1, got {!r}".format(
            min_frames))
    min_frames = int(min_frames)
    if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real) or not 0.5 < confidence <= 1.0:
        raise ValueError("controlled_touch: confidence must be a number in (0.5, 1.0] (0.5 would say nothing), "
                         "got {!r}".format(confidence))
    confidence = float(confidence)
    if isinstance(hold, bool) or not isinstance(hold, numbers.Real) or not 0.5 <= hold <= confidence:
        raise ValueError("controlled_touch: hold must be a number in [0.5, confidence = {}] (0.5 = no hold; more than "
                         "confidence would out-vote a control episode), got {!r}".format(confidence, hold))
    hold = float(hold)
    backdate_tau, backdate_confidence, backdate_cap = _backdate(backdate_tau, backdate_confidence, backdate_cap, hold,
                                                                confidence)

    inputs = run.inputs
    n_frames = inputs.atd.n_frames
    ball, players, nearest, nearest_is_home, in_zone = _zone(inputs, radius, duel_radius)
    delta = ball.vel - players.vel[np.arange(n_frames), nearest]
    with np.errstate(invalid="ignore"):                    # NaN velocities compare False: not a candidate
        candidate = in_zone & (np.hypot(delta[:, 0], delta[:, 1]) <= max_rel_speed)
    touched = None if backdate_tau is None else _touches(inputs, ball, in_zone, nearest_is_home, backdate_tau)

    home = no_opinion(n_frames)
    for start, stop in kinematics.blocks(inputs):
        if stop > start:
            block = slice(start, stop)
            episodes = _episodes(candidate[block], nearest_is_home[block], min_frames)
            home[block] = _block_home(episodes, nearest_is_home[block], ball.visible[block], confidence, hold)
            if touched is not None:
                for first, last, team_is_home in _block_stretches(episodes, ball.visible[block], touched[0][block],
                                                                  touched[1][block]):
                    if backdate_cap is not None:
                        first = max(first, last - backdate_cap)    # the last backdate_cap frames before the episode
                    home[start + first:start + last] = (backdate_confidence if team_is_home
                                                        else 1.0 - backdate_confidence)
    return Evidence(alive=no_opinion(n_frames), home=home)


def touches(inputs, *, radius, duel_radius, tau):
    """(home, away): bool (n,) each, True on the frame where a touch of that team is dated (P28; module docstring).

    The spatial test is the episode rule's own, on the same arrays (_zone), with no speed test; `tau` is the least
    change of the ball's velocity, in metres per second. No frame is True in both, and none outside the graded blocks.
    """
    radius = _positive("radius", radius, "metres from the ball", allow_inf=False)
    duel_radius = _positive("duel_radius", duel_radius, "metres from the ball", allow_inf=False)
    tau = _positive("tau", tau, "metres per second", allow_inf=False)
    ball, _, _, nearest_is_home, in_zone = _zone(inputs, radius, duel_radius)
    return _touches(inputs, ball, in_zone, nearest_is_home, tau)


def _zone(inputs, radius, duel_radius):
    """(ball, players, nearest, nearest_is_home, in_zone): the spatial test, for the episodes and the touches alike.

    `nearest` is the nearest present slot of each frame (either team) and `in_zone` is True where the ball is visible,
    that slot is within `radius` and the other team's nearest slot is farther than `duel_radius`, or absent.
    """
    atd = inputs.atd
    ball = kinematics.ball_track(inputs)
    players = kinematics.player_track(inputs)
    distance = kinematics.ball_distances(players.xy_m, ball.xy_m)
    distance[np.isnan(distance)] = np.inf                  # absent slot, absent ball, outside the blocks: never near
    nearest = np.argmin(distance, axis=1)                  # the lowest slot index on a tie, as nearest_player
    nearest_distance = distance[np.arange(atd.n_frames), nearest]
    nearest_is_home = atd.slot_team[nearest] == HOME
    rival = np.where((atd.slot_team == HOME)[None, :] != nearest_is_home[:, None], distance, np.inf).min(axis=1)
    return ball, players, nearest, nearest_is_home, ball.visible & (nearest_distance <= radius) & (rival > duel_radius)


def _touches(inputs, ball, in_zone, nearest_is_home, tau):
    """(home, away) bool (n,): the frames where a touch is dated, by the team of the nearest slot there."""
    home = np.zeros(len(in_zone), dtype=bool)
    away = np.zeros(len(in_zone), dtype=bool)
    for start, stop in kinematics.blocks(inputs):
        m = stop - start
        if m <= 2 * TOUCH_WINDOW:
            continue                                       # no frame of the block has its whole window in it
        vel, visible = ball.vel[start:stop], ball.visible[start:stop]
        step = vel[2 * TOUCH_LAG:] - vel[:-2 * TOUCH_LAG]  # v(t + 4) - v(t - 4), both frames in the block
        change = np.full(m, np.nan)
        change[TOUCH_LAG:m - TOUCH_LAG] = np.hypot(step[:, 0], step[:, 1])
        with np.errstate(invalid="ignore"):                # a NaN velocity compares False: not a touch
            kinked = change >= tau
        seen = np.concatenate(([0], np.cumsum(visible)))   # the ball on every frame of [t - 7, t + 7], in the block
        whole = np.zeros(m, dtype=bool)
        whole[TOUCH_WINDOW:m - TOUCH_WINDOW] = (seen[2 * TOUCH_WINDOW + 1:] - seen[:m - 2 * TOUCH_WINDOW]
                                                == 2 * TOUCH_WINDOW + 1)
        for first, last in zip(*kinematics.runs(kinked)):
            dated = first + int(np.argmax(change[first:last]))         # argmax: the earliest on a tie
            if whole[dated] and in_zone[start + dated]:
                (home if nearest_is_home[start + dated] else away)[start + dated] = True
    return home, away


def _episodes(candidate, is_home, min_frames):
    """(firsts, stops, homes): one block's kept episodes in time order, half-open, and whether each is home's."""
    firsts, stops, homes = [], [], []
    for team_is_home in (True, False):
        starts, ends = kinematics.runs(candidate & (is_home == team_is_home))
        long = ends - starts >= min_frames
        firsts.append(starts[long])
        stops.append(ends[long])
        homes.append(np.full(int(long.sum()), team_is_home))
    firsts, stops, homes = np.concatenate(firsts), np.concatenate(stops), np.concatenate(homes)
    order = np.argsort(firsts, kind="stable")              # a frame has one nearest slot, so no two episodes overlap
    return firsts[order], stops[order], homes[order]


def _block_home(episodes, is_home, visible, confidence, hold):
    """One block's home evidence: kept episodes, then the hold between them."""
    m = len(visible)
    kept = np.zeros(m, dtype=bool)
    for first, last in zip(episodes[0], episodes[1]):
        kept[first:last] = True
    home = np.full(m, np.nan)
    home[kept] = np.where(is_home[kept], confidence, 1.0 - confidence)
    if hold > 0.5:
        # The index of the last kept frame at or before each frame, -1 before the first: its team is the last team.
        last = np.maximum.accumulate(np.where(kept, np.arange(m), -1))
        between = visible & ~kept & (last >= 0)
        home[between] = np.where(is_home[last[between]], hold, 1.0 - hold)
    return home


def _block_stretches(episodes, visible, touch_home, touch_away):
    """[(first, stop, team is home)]: the stretches of one block that P28's rule back-dates, half-open, uncapped.

    For each kept episode of a team T whose previous kept episode is the other team O's: the frames from T's earliest
    touch between the two, from which the ball is visible on every frame up to T's episode and after which no touch of O
    is dated, to the frame before T's episode. No such touch, no stretch.
    """
    firsts, stops, homes = episodes
    # The last frame at or before each frame where the ball is missing, -1 if there is none.
    missing = np.maximum.accumulate(np.where(visible, -1, np.arange(len(visible))))
    stretches = []
    for k in range(1, len(firsts)):
        low, high = int(stops[k - 1]), int(firsts[k])      # the frames between O's episode and T's
        if homes[k] == homes[k - 1] or high <= low:
            continue
        own, other = (touch_home, touch_away) if homes[k] else (touch_away, touch_home)
        low = max(low, int(missing[high - 1]) + 1)         # after the last frame without a ball
        rivals = np.flatnonzero(other[low:high])
        if rivals.size:
            low += int(rivals[-1]) + 1                     # and after O's last touch
        mine = np.flatnonzero(own[low:high])
        if mine.size:
            stretches.append((low + int(mine[0]), high, bool(homes[k])))
    return stretches


def _backdate(tau, confidence, cap, hold, episode_confidence):
    """(backdate_tau, backdate_confidence, backdate_cap) checked: floats and an int, or three None when P28 is off."""
    if tau is None:
        given = [name for name, value in (("backdate_confidence", confidence), ("backdate_cap", cap))
                 if value is not None]
        if given:
            raise ValueError("controlled_touch: {} is given without backdate_tau, which turns the back-dating on; "
                             "give backdate_tau (metres per second) or drop {}".format(
                                 " and ".join(given), "them" if len(given) > 1 else "it"))
        return None, None, None
    tau = _positive("backdate_tau", tau, "metres per second: the least change of the ball's velocity at a touch",
                    allow_inf=False)
    if (isinstance(confidence, bool) or not isinstance(confidence, numbers.Real)
            or not hold <= confidence <= episode_confidence or not confidence > 0.5):
        raise ValueError("controlled_touch: backdate_confidence is required with backdate_tau and must be a number in "
                         "[hold = {}, confidence = {}] and above 0.5 (0.5 would say nothing, less than hold less than "
                         "no touch does, more than confidence would out-vote a control episode), got {!r}".format(
                             hold, episode_confidence, confidence))
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, numbers.Integral) or cap < 1):
        raise ValueError("controlled_touch: backdate_cap must be a whole number of frames >= 1 (leave it out for no "
                         "cap), got {!r}".format(cap))
    return tau, float(confidence), None if cap is None else int(cap)


def _positive(name, value, unit, allow_inf):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not value > 0 or (
            not allow_inf and not np.isfinite(value)):
        raise ValueError("controlled_touch: {} must be a positive {}number ({}), got {!r}".format(
            name, "" if allow_inf else "finite ", unit, value))
    return float(value)
