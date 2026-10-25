"""P04 -- who controls the ball: home evidence from control episodes only, NaN everywhere else.

    [[evidence]]
    kind = "controlled_touch"
    radius = 2.0           # metres: the nearest player must be this close
    max_rel_speed = 4.0    # m/s: and move with the ball (|v_ball - v_player|); inf switches the test off
    min_frames = 5         # an episode shorter than this is dropped
    confidence = 0.9       # home on a home episode (1 - confidence on an away one), in (0.5, 1.0]
    duel_radius = 1.0      # optional: the other team must be farther than this, or the ball is contested
    hold = 0.5             # optional, in [0.5, confidence]: the weak vote of the last team; 0.5 = off

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

Everything is per graded block (kinematics.blocks): an episode is cut at the block's edge, a hold never crosses it, and
frames outside the blocks are NaN. `alive` is NaN everywhere: whether the ball is in play is P05's question. `confidence`
is measured, not tuned (docs/proposals.md P04).
"""

import numbers

import numpy as np

from pipeline.contracts import Evidence, no_opinion
from pipeline.evidence import kinematics
from starter.baseline import HOME


def compute(run, *, radius, max_rel_speed, min_frames, confidence, duel_radius=1.0, hold=0.5):
    """Evidence(home = confidence / 1 - confidence in kept control episodes, hold / 1 - hold between them, else NaN;
    alive = NaN)."""
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

    inputs = run.inputs
    atd = inputs.atd
    ball = kinematics.ball_track(inputs)
    players = kinematics.player_track(inputs)
    n_frames = atd.n_frames
    frames = np.arange(n_frames)

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

    home = no_opinion(n_frames)
    for start, stop in kinematics.blocks(inputs):
        if stop > start:
            home[start:stop] = _block_home(candidate[start:stop], nearest_is_home[start:stop],
                                           ball.visible[start:stop], min_frames, confidence, hold)
    return Evidence(alive=no_opinion(n_frames), home=home)


def _block_home(candidate, is_home, visible, min_frames, confidence, hold):
    """One block's home evidence: kept episodes, then the hold between them."""
    m = len(candidate)
    kept = np.zeros(m, dtype=bool)
    for team_is_home in (True, False):
        starts, stops = kinematics.runs(candidate & (is_home == team_is_home))
        for first, last in zip(starts, stops):
            if last - first >= min_frames:
                kept[first:last] = True
    home = np.full(m, np.nan)
    home[kept] = np.where(is_home[kept], confidence, 1.0 - confidence)
    if hold > 0.5:
        # The index of the last kept frame at or before each frame, -1 before the first: its team is the last team.
        last = np.maximum.accumulate(np.where(kept, np.arange(m), -1))
        between = visible & ~kept & (last >= 0)
        home[between] = np.where(is_home[last[between]], hold, 1.0 - hold)
    return home


def _positive(name, value, unit, allow_inf):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not value > 0 or (
            not allow_inf and not np.isfinite(value)):
        raise ValueError("controlled_touch: {} must be a positive {}number ({}), got {!r}".format(
            name, "" if allow_inf else "finite ", unit, value))
    return float(value)
