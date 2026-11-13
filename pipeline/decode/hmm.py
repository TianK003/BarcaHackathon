"""P06's decoder: a three-state hidden Markov model over H, A and D, decoded by Viterbi, once per block; with
restart_source (P17), D splits into two restart states; with hazard_source (P35), a change of team costs a price per
frame.

    [decoder]
    kind = "hmm"
    switch_team = 8        # nats a change of team (H <-> A) costs
    switch_dead = 16       # nats a change into or out of D costs
    min_possession = 25    # frames: the shortest run of H or of A (default 1: no minimum)
    min_dead = 25          # frames: the shortest run of D (default 1)
    eps = 1e-6             # probabilities are clipped to [eps, 1 - eps] before the log (default 1e-6)
    posterior = false      # true: also return P(H), P(A), P(D) per frame (default false: Decoded.posterior is None)
    restart_source = "restart_team"   # optional: the evidence source whose `home` says who restarts (default: none)
    switch_restart = inf   # nats a restart by the other team costs on top of switch_dead (default inf: forbidden)
    hazard_source = "hazard"   # optional: the evidence source whose `home` is a turnover hazard (default: none)
    hazard_scale = 1.0     # with hazard_source: how far the hazard's log-odds move switch_team (default 1.0)

The model
---------
Per frame t and state s (H, A, D = 0, 1, 2) the evidence gives an emission score e[t, s] in nats, a log-likelihood
ratio against a prior of 0.5 -- the reading pipeline/decode/README.md has always promised. For each source, with w its
weight (an exponent on the source's likelihood, so weight 2 counts the source twice and weight 0 not at all):

    where `home` is known:   h = clip(home, eps, 1 - eps)    e[H] += w log 2h        e[A] += w log 2(1 - h)
    where `alive` is known:  a = clip(alive, eps, 1 - eps)   e[H] += w log 2a        e[A] += w log 2a
                                                             e[D] += w log 2(1 - a)

A channel that is NaN adds 0 to every state: it cannot push the decoder anywhere, which is what "no opinion" means.
(The product P(H) = alive * home would put a missing `alive` at a coin flip and hand D a head start on every frame
where only `home` is known. `home` is a likelihood ratio between H and A only, so D gets nothing from it.)

Team-neutral dead (P31)
-----------------------
With team_neutral_dead = true the `home` line gains a term, and nothing else changes:

    where `home` is known:   e[D] += w max(log 2h, log 2(1 - h))

A confident team vote then lifts D as much as it lifts the better team, so it ranks H against A and no longer says
"the ball is in play": the rule above leaves D at 0 where only `home` speaks, which makes every confident team term a
vote against a stoppage. Off (the default), the emissions, the labels and the posterior are what they were, bit for
bit. It is the three states' only: with restart_source it is refused.

A path is one label per frame of the block. Its score is the sum of e[t, s_t] along it, less switch_team for every
H <-> A change and switch_dead for every change into or out of D. There is no penalty at the block's edges. The
decoder returns a *valid* path of maximal score: every run (maximal constant stretch) of H or A lasts at least
min_possession frames and every run of D at least min_dead -- except the block's first and last run, which the edge
of the block may have cut short. They may be the same run, and a block shorter than a minimum is still decodable: it
is its two edge runs, however short.

Restart states (P17)
--------------------
Without restart_source, everything above is the model, by the same code path and to the same label. With
restart_source naming one of the block's evidence sources, D splits in two, D_H ("dead, home restarts") and D_A
("dead, away restarts"), and the decoder runs four states, H, A, D_H and D_A:

    emissions   H and A as above, from every source except the named one's `home`
                D_H and D_A both take D's emission: the `alive` terms of every source, the named one's included
                the named source's `home` = h adds  w log 2h  to D_H  and  w log 2(1 - h)  to D_A, nothing to H or A
    changes     H or A  -> D_H or D_A       switch_dead
                D_H -> H,  D_A -> A         switch_dead
                D_H -> A,  D_A -> H         switch_dead + switch_restart (inf, the default: forbidden)
                D_H <-> D_A                 forbidden
    minimums    min_possession for H and A, min_dead for D_H and D_A
    labels      D_H and D_A are both written D

restart_source expects a source that speaks on dead frames only, restart_team's span "stoppage_end". A source that
votes on live frames (span "after") would raise the dead states there by up to w log 2c a frame (a term on D_H or
D_A, whichever it names, and none on H and A) and pull D into the restart.

Why. The Laws of the Game decide who restarts after a stoppage, and the first team after it is that team. R014 voted
for it in `home` on a fixed span after the source's own restart estimate and gained +0.004 spellF1: a fixed span keeps
pushing after the restarting team's first possession has ended, and it starts where the source thinks play resumed,
not where the decoder ends D. As a state, the restart team is read inside the stoppage (restart_team's span
"stoppage_end": one observation at the type's measured confidence c, spread over the stoppage's n frames as
sigmoid(logit(c) / n) each, so that the log-odds between D_H and D_A add up to logit(c)), carried by the decoder to
whichever frame it ends D on, and decides only the first team after it: later evidence ends that possession as it ends
any other. Spread, the vote survives a D run that covers only part of the stoppage, with that part of its log-odds; on
the stoppage's last frame alone it would be lost whenever the decoder ends D a frame early. And it hardly touches where
D ends: it says nothing of H against A, and raises the better dead state by about w logit(c) / 2n a frame (0.011 nats
at c = 0.9 over 4 s), where the span after the restart raised the alive states by up to w log 2c on every frame of it.

D_H and D_A cannot follow one another, so a stoppage has one restart team, and a vote on any of its frames counts for
all of it. switch_restart prices a restart by the team the vote did not name: inf is the model in which the first team
after a stoppage is always its restart team (true of the ground truth by its definition), a finite one lets strong
contrary evidence overrule the vote.

Without restart evidence (the named source NaN in `home` everywhere, or of weight 0), the four states give the
three-state labels, for any switch_restart: D_H and D_A then have the same emissions and the same changes in and out,
so every three-state path has a four-state twin of the same score (each D run takes the state of the team after it,
D_H at the block's end), no four-state path scores more than the three-state path it maps to, and the tie order below
picks the twin of the three-state choice. pipeline/selftest.py checks that on random blocks, and the four-state
decoder against every path of tiny blocks.

Per-frame team cost (P35)
-------------------------
With hazard_source naming one of the block's evidence sources, a change of team no longer costs switch_team on every
frame. The named source's `home` is a hazard z_t: how likely a direct change of team is near frame t, against its base
rate (turnover_hazard, P35's model, writes z = p / (p + b), so that logit z = log(p / b)). It speaks where the ball is
seen and is NaN elsewhere. A change between H and A from frame t - 1 to frame t costs

    team_cost[t] = clip(switch_team - hazard_scale * logit(clip(z_t, eps, 1 - eps)), HAZARD_FLOOR, 2 * switch_team)

and exactly switch_team where z_t is NaN. Nothing else changes: a change into or out of D costs switch_dead, the
minimums, the validity and the tie order are the three states' above, and the emissions are every other source's (the
named source adds nothing to them, neither its `home` nor its `alive`). HAZARD_FLOOR, 1 nat, keeps a change from ever
becoming free, and the cap of 2 * switch_team keeps a frame the hazard rules out from becoming a wall that no team
evidence could climb. Both were declared with P35 and are not knobs.

Which frame's cost. A change from frame t - 1 to frame t costs team_cost[t]: the cost belongs to the first frame of the
run it starts, and is paid once, however long that run lasts. In the recursion over run starts (below) a run of A
entered at frame a from H has the entry score L[H][a - 1] - team_cost[a], which Q[a] carries from then on, so the
running maximum over the starts is the same recursion. team_cost[0] is never read: no change enters a block's first
frame.

Why. In an input-dependent HMM the cost of a change at t is minus the log of its probability at t. switch_team is that
cost at the base rate, and z_t moves it by its log-odds against the base rate: a change is cheaper where the evidence of
a hand-over is strong and dearer where it is weak, and hazard_scale above 1 counts the hazard more than once (the
emissions are counted several times over, A25, so the right scale in the decoder's units may be above 1).

Off (no hazard_source, the default), viterbi runs the loop it always ran: which of the two loops runs is chosen once,
before either. A hazard NaN everywhere, a hazard of 0.5, and hazard_scale 0 all cost exactly switch_team on every
frame, and a team_cost of switch_team on every frame gives the labels of the loop without it, to the label: the same
candidates in the same order and the same floats. pipeline/selftest.py checks that on random blocks and on exactly
tied ones, and the priced decoder against every path of tiny blocks.

Refused: hazard_scale without hazard_source (nothing would read it); a hazard_source that names none of the block's
sources; a scale that is negative, infinite, NaN or not a number; a hazard with restart_source or with
team_neutral_dead (it prices the three states' changes of team under today's emissions); a hazard with switch_team
below HAZARD_FLOOR (a frame at the base rate would then cost more than switch_team, and hazard_scale 0 would not be the
decoder without a hazard); and a weight other than 1 on the named source (the hazard's strength is hazard_scale, and a
weight would be read by nothing). posterior = true takes the same per-frame costs (The posterior, below).
pipeline/config.py refuses, before any data is read, a hazard_source that is not a "turnover_hazard" source and a
"turnover_hazard" source that the decoder does not name (read as an ordinary source, its `home` would vote for the
home team wherever a change is likely).

Why each parameter exists
-------------------------
* switch_team is how many nats of evidence a change of team has to beat. Two changes (out and back) cost twice that,
  so a blip of n contrary frames is believed only when n times its per-frame evidence exceeds 2 * switch_team: that is
  what removes the flicker the nearest-player labels are made of (docs/traps.md T04).
* switch_dead is the same for stoppages. It is separate because the two kinds of change are not equally likely, and
  because `alive` evidence is lopsided where `home` evidence is not: a seen ball is weak evidence of play (0.34 nats a
  frame from ball_presence) and a missing one is not weak evidence of a stoppage (1.39 nats, more than nearest_player's
  1.27 between H and A at confidence 0.78).
* min_possession and min_dead are the hard floor under it: a run shorter than the minimum is not a possession, whatever
  the evidence says, and the decoder pays for the frames it has to give a short run to reach the minimum. A minimum
  makes a run grow to its length where the evidence is only half there (a 20-frame blip with min_possession 25 becomes
  a 25-frame run if that still pays), and removes it only where it does not.
* eps keeps the log of a hard 0 or 1 finite. nearest_player reports exactly those, and a frame at eps would count
  log(2e-6) = -13 nats against a state, more than most switch penalties: it is a numerical guard, not a knob.
* restart_source and switch_restart are the restart states' (above). switch_restart must be a number > 0 or inf, and
  is refused when given finite without restart_source, where nothing would read it.
* hazard_source and hazard_scale are the per-frame team cost's (above). hazard_scale must be a finite number >= 0, is
  1.0 when left out with a source, and is refused without one.

Ties
----
Exactly equal scores are common (a stretch without evidence costs the same whoever holds it) and are decided the same
way every time, in the order the recursion meets its choices: a path prefers to extend its current run (and, between two
ways of doing so, the longer one); failing that to come from D, then from H, then from A; at the end of the block the
final state is D, then H, then A, the longer final run first. With restart states the order is D_H, then D_A, then H,
then A, in both places. Three consequences hold the tie rule in place:

* a block with no evidence at all decodes to all D, like majority_vote's fallback;
* a change of team across a stretch without evidence lands on the stretch's first frame: the run that ends the
  stretch has the earlier start, because it extends the longer;
* without restart evidence, D_H's choices are D's: it comes first, so the four states pick the three states' path.

With a per-frame team cost the order is the same. Across a stretch without team evidence the change of team then lands
on the stretch's cheapest frame, the earliest of them on a tie: where the hazard is silent there (NaN, so switch_team
on every frame), that is the stretch's first frame, as without it.

How it is solved
----------------
The textbook form is a Viterbi over an expanded chain, each state split into "run length so far" substates 1..m, the
last with a self-loop: first and last run exempt means starting in the last substate and ending in any. Here it is
written as the same recursion over run starts, which costs O(1) per frame and state instead of O(m). Let
L[s][t] be the best score of a valid prefix ending at t in state s whose current run has reached the minimum (or is the
first run). A run of s that began at frame a adds e[a..t] to the score at which it was entered, E[s][a], the best
L[r][a-1] - penalty over the other states r. With cum the running sum of e, the score of that run at frame t is
E[a] - cum[a-1] + cum[t]: so the best way to have a mature run at t is cum[t] plus a running maximum, over the starts
a <= t - m + 1, of Q[a] = E[a] - cum[a-1]. Both edge exemptions are written in: L[s][0] is "mature" (the first run
needs no length), and the final answer also looks at runs that began fewer than m frames before the end (the last run
needs none). The maximum is kept with a strict comparison, so the earlier start survives a tie.

The recursion is written once (`_best_path`) over any number of states, a matrix of change costs in which inf forbids a
change (a forbidden change is no way in at all), a minimum per state and a tie order. `viterbi` gives it H, A and D,
`restart_viterbi` H, A, D_H and D_A. test_hmm_speed's 80,000 frames with both minimums at 50 take 0.33 s with three
states and 0.40 to 0.45 s with four (best of three, the machine otherwise idle, 26 Oct 2026).

A per-frame team cost (team_cost, from hazard_costs) is a second copy of the loop, chosen once before it, in which an
H <-> A candidate is priced at team_cost[frame] instead of its penalty; without one the first copy runs, as it always
did. test_hmm_hazard_speed's 80,000 frames with a hazard take 1.07 to 1.08 times the decode without it (0.17 s against
0.16 s, medians of three, 13 Nov 2026).

The posterior
-------------
With posterior = true, Decoded.posterior is the (len(block), 3) marginal probability of H, A and D per frame under the
same model: a valid path has weight exp(its score), and a frame's probability of a state is the total weight of the
valid paths that hold it there over the total weight of all valid paths. Each row sums to 1. The labels do not change:
they are the best single path, not the per-frame argmax of the posterior, and the two part company where a stretch is
close. The posterior is the model's belief given its evidence, so it is as calibrated as the evidence is and no more.

It comes from `marginals`, a forward-backward pass over the textbook chain, in log space, written as a vectorised step
per frame. The recursion over run starts above does not carry over: it works with running sums of the emissions as long
as the block, and the cancellation in them (of the order of 1e-10 nats over 80,000 frames) is harmless for a maximum
but not for a probability of 1e-12. The chain costs O(min_possession + min_dead) per frame and n_frames *
(2 * min_possession + min_dead) * 8 bytes, so it is off by default: the benchmark never asks for it. 80,000 frames with
both minimums at 50 take about 2 s.

The posterior is the three states' only: posterior = true with restart_source is refused. With hazard_source it takes
the decoder's per-frame team cost (marginals' team_cost): the forward pass's birth of a run into frame t is priced at
team_cost[t], and the backward pass's leave from frame t towards t + 1 at team_cost[t + 1], the same frame's cost seen
from the other side. Without one, both passes run as they always did (the choice is made once, before each), and a
team_cost of switch_team on every frame gives the same posterior to the last bit. pipeline/selftest.py checks it
against every path of tiny blocks and against a long block read backwards with its costs moved with it.

What it does not do yet
-----------------------
* No duration priors beyond the minimums: a run is as likely to last 2 minutes as 5 seconds, once it is long enough.
  A semi-Markov decoder would take a log-prior over run lengths where the minimum is a hard step.
* The restart states know who restarts, nothing about how long that team keeps the ball, and nothing about a stoppage
  the `alive` evidence does not decode: no D run, no restart team. They have no posterior (`marginals` is three-state).
* D is only as good as its `alive` evidence. With nearest_player alone nothing speaks for D, so the decoder chooses it
  rarely and only in gaps: H -> D -> A beats H -> A when switch_team > 2 * switch_dead and the gap is at least min_dead
  frames, and with switch_dead = 0 a trailing stretch without evidence ends in D by the tie preference. A dead-ball
  source (P05) is what gives it something real to decode.
* The two penalties and the minimums are set by hand in the config; the benchmark (P01) is where they are chosen.
"""

import math
import numbers

import numpy as np

from pipeline.contracts import A, D, Decoded, H

# Params
EPS = 1e-6                    # default clip of a probability away from 0 and 1
PRIOR_CHANCE = 2.0            # 1 / 0.5: log(x / prior) = log(2x) for a prior of one half
PREFERENCE = (D, H, A)        # on a tie between two predecessors, the first one listed wins
STATES = (H, A, D)            # also the columns of the emissions
D_H, D_A = 2, 3               # the restart states, "dead, home restarts" and "dead, away restarts": columns, not codes
RESTART_STATES = (H, A, D_H, D_A)                      # the columns of the four-state emissions and path
RESTART_PREFERENCE = (D_H, D_A, H, A)                  # the tie order with restart states
RESTART_LABELS = np.array([H, A, D, D], dtype=np.int8)  # four-state column -> label code: both dead states are D
RESTART_LABELS.flags.writeable = False
CHUNK = 4096                  # frames per step of the posterior's final normalisation: bounds its temporary arrays
HAZARD_FLOOR = 1.0            # nats: the least a change of team costs under a hazard (P35, declared; not a knob)
DEFAULT_HAZARD_SCALE = 1.0    # hazard_scale when hazard_source is given without one


def decode(ctx, *, switch_team, switch_dead, min_possession=1, min_dead=1, eps=EPS, posterior=False,
           restart_source=None, switch_restart=math.inf, team_neutral_dead=False, hazard_source=None,
           hazard_scale=None):
    """Decoded(labels) for ctx.block: the best valid path through the block's emissions.

    With posterior=True, Decoded.posterior is also the (len(block), 3) marginal probability of H, A and D per frame
    under the same model (`marginals`); the labels are the same either way. Otherwise posterior is None.

    With restart_source, the name of one of ctx.evidence's sources, the four states of the module docstring: that
    source's `home` says who restarts, switch_restart prices a restart by the other team, and D_H and D_A are written
    D. There is no posterior then.

    With team_neutral_dead (a bool, default False), every `home` term also gives D the larger of the two team terms:
    team evidence ranks H against A and leaves the better of them level with D (the module docstring). It is the three
    states' only, so it is refused with restart_source.

    With hazard_source, the name of one of ctx.evidence's sources, that source's `home` is a hazard z and a change of
    team into frame t costs hazard_costs(z, switch_team, hazard_scale, eps)[t] instead of switch_team (the module
    docstring's per-frame team cost); the source adds nothing to the emissions, and the posterior, if asked for, takes
    the same costs. hazard_scale is a finite number >= 0, None meaning DEFAULT_HAZARD_SCALE. Refused (ValueError):
    hazard_scale without hazard_source; a hazard_source that is not a non-empty string or names none of the block's
    sources, or whose source has a weight other than 1; a hazard with restart_source or with team_neutral_dead; and a
    hazard with switch_team below HAZARD_FLOOR.
    """
    _check_costs(switch_team, switch_dead, min_possession, min_dead)
    _check_eps(eps)
    if not isinstance(posterior, (bool, np.bool_)):
        raise ValueError("hmm: posterior must be True or False, got {!r}".format(posterior))
    if not isinstance(team_neutral_dead, (bool, np.bool_)):
        raise ValueError("hmm: team_neutral_dead must be True or False, got {!r}".format(team_neutral_dead))
    _check_restart(restart_source, switch_restart)
    if team_neutral_dead and restart_source is not None:
        raise ValueError("hmm: team_neutral_dead = true is the three states' only, and restart_source {!r} asks for "
                         "the four restart states; leave one of them out".format(restart_source))
    hazard_scale = _check_hazard(hazard_source, hazard_scale, switch_team, restart_source, team_neutral_dead)
    if hazard_source is not None:
        hazard = _hazard_source(ctx.evidence, hazard_source, len(ctx.block))
        emissions = log_likelihoods([source for name, source in ctx.evidence.items() if name != hazard_source],
                                    len(ctx.block), eps)
        team_cost = hazard_costs(hazard.home, float(switch_team), hazard_scale, eps)
        labels = viterbi(emissions, switch_team, switch_dead, min_possession, min_dead, team_cost)
        if not posterior:
            return Decoded(labels=labels)
        return Decoded(labels=labels, posterior=marginals(emissions, switch_team, switch_dead, min_possession,
                                                          min_dead, team_cost))
    if restart_source is None:
        emissions = log_likelihoods(ctx.evidence.values(), len(ctx.block), eps, bool(team_neutral_dead))
        labels = viterbi(emissions, switch_team, switch_dead, min_possession, min_dead)
        if not posterior:
            return Decoded(labels=labels)
        return Decoded(labels=labels, posterior=marginals(emissions, switch_team, switch_dead, min_possession,
                                                          min_dead))
    if posterior:
        raise ValueError("hmm: posterior = true is the three states' only, and restart_source {!r} asks for the four "
                         "restart states; leave one of them out".format(restart_source))
    emissions = restart_log_likelihoods(ctx.evidence, restart_source, len(ctx.block), eps)
    path = restart_viterbi(emissions, switch_team, switch_dead, min_possession, min_dead, switch_restart)
    return Decoded(labels=RESTART_LABELS[path])


def log_likelihoods(sources, n_frames, eps, neutral_dead=False):
    """(n_frames, 3) float64 emissions in nats, columns H, A, D: the sum over `sources` of the rule in the docstring.

    `sources` is an iterable of Evidence, each of length n_frames. A source's weight must be finite and >= 0 (an
    exponent on its likelihood), whether or not it has an opinion anywhere. With neutral_dead, where a source's `home`
    is known D also gets the larger of the H and A terms, as the docstring's team-neutral rule says. Reads the
    evidence, never writes it: the runner hands over read-only arrays.
    """
    _check_eps(eps)
    emissions = np.zeros((n_frames, 3))
    for source in sources:
        _add_source(emissions, source, n_frames, eps, neutral_dead=neutral_dead)
    return emissions


def restart_log_likelihoods(evidence, restart_source, n_frames, eps):
    """(n_frames, 4) float64 emissions in nats, columns H, A, D_H, D_A (RESTART_STATES): the restart states' rule.

    `evidence` maps source names to Evidence (ctx.evidence), and `restart_source` names the one whose `home` says who
    restarts: it goes to D_H and D_A, and nothing else's does. Everything else is log_likelihoods', with D's column in
    both dead states. Reads the evidence, never writes it.
    """
    _check_eps(eps)
    if not isinstance(restart_source, str) or restart_source not in evidence:
        raise ValueError("hmm: restart_source {!r} names no evidence source of this block; its sources are {}".format(
            restart_source, ", ".join(repr(name) for name in evidence) or "none"))
    three = np.zeros((n_frames, 3))
    for name, source in evidence.items():
        _add_source(three, source, n_frames, eps, home=name != restart_source)
    emissions = np.empty((n_frames, len(RESTART_STATES)))
    emissions[:, H] = three[:, H]
    emissions[:, A] = three[:, A]
    emissions[:, D_H] = three[:, D]
    emissions[:, D_A] = three[:, D]
    named = evidence[restart_source]
    known = ~np.isnan(named.home)
    if known.any():
        home = np.clip(named.home[known], eps, 1.0 - eps)
        emissions[known, D_H] += named.weight * np.log(PRIOR_CHANCE * home)
        emissions[known, D_A] += named.weight * np.log(PRIOR_CHANCE * (1.0 - home))
    return emissions


def hazard_costs(hazard, switch_team, hazard_scale, eps):
    """(n_frames,) float64: the nats a change of team into each frame costs under the hazard z = `hazard` (a source's
    `home`, NaN where it has no opinion), by the rule in the module docstring:

        clip(switch_team - hazard_scale * logit(clip(z, eps, 1 - eps)), HAZARD_FLOOR, 2 * switch_team)

    and exactly switch_team where z is NaN. The arguments are decode's, already checked (switch_team >= HAZARD_FLOOR, so
    z = 0.5 and hazard_scale 0 give exactly switch_team). Reads the hazard, never writes it.
    """
    hazard = np.asarray(hazard, dtype=np.float64)
    z = np.clip(hazard, eps, 1.0 - eps)                  # a NaN stays NaN, and is replaced below
    costs = np.clip(switch_team - hazard_scale * np.log(z / (1.0 - z)), HAZARD_FLOOR, 2.0 * switch_team)
    costs[np.isnan(hazard)] = switch_team
    return costs


def _add_source(emissions, source, n_frames, eps, home=True, neutral_dead=False):
    """Add one source's terms to the (n_frames, 3) `emissions` in place, by the rule in the docstring: its `home` to H
    and A (unless home is False; with neutral_dead, also the larger of the two to D), its `alive` to H, A and D.
    ValueError on a bad weight or length."""
    weight = source.weight
    if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError("hmm: evidence {!r} has weight {!r}; a weight is an exponent on its likelihood and must "
                         "be a finite number >= 0".format(source.name, weight))
    if len(source.home) != n_frames or len(source.alive) != n_frames:
        raise ValueError("hmm: evidence {!r} has {} and {} frames (home, alive), the block has {}".format(
            source.name, len(source.home), len(source.alive), n_frames))
    if home:
        known = ~np.isnan(source.home)
        if known.any():
            values = np.clip(source.home[known], eps, 1.0 - eps)
            home_term = weight * np.log(PRIOR_CHANCE * values)
            away_term = weight * np.log(PRIOR_CHANCE * (1.0 - values))
            emissions[known, H] += home_term
            emissions[known, A] += away_term
            if neutral_dead:
                emissions[known, D] += np.maximum(home_term, away_term)
    known = ~np.isnan(source.alive)
    if known.any():
        alive = np.clip(source.alive[known], eps, 1.0 - eps)
        in_play = weight * np.log(PRIOR_CHANCE * alive)
        emissions[known, H] += in_play
        emissions[known, A] += in_play
        emissions[known, D] += weight * np.log(PRIOR_CHANCE * (1.0 - alive))


def viterbi(emissions, switch_team, switch_dead, min_possession, min_dead, team_cost=None):
    """int8 labels of a valid path of maximal score through `emissions` ((n_frames, 3), columns H, A, D).

    Valid, scored and tied as the module docstring says. n_frames = 0 gives no labels. With team_cost, one finite
    number of nats >= 0 per frame, a change between H and A from frame t - 1 to frame t costs team_cost[t] instead of
    switch_team (team_cost[0] is never read: no change enters the first frame); ValueError naming what is wrong with its
    shape or values. None, the default, runs the recursion as it always ran.
    """
    switch_team, switch_dead, min_possession, min_dead = _check_costs(
        switch_team, switch_dead, min_possession, min_dead)
    emissions = _check_emissions(emissions, len(STATES))
    if team_cost is not None:
        team_cost = _check_team_cost(team_cost, len(emissions))
    # Every path takes one state per frame, so subtracting a frame's H score from all three states moves every path's
    # score by the same constant: in exact arithmetic that changes no decision. In floats it changes which of several
    # exactly tied paths wins, and that is its purpose: it makes the tie inside a ball gap exact, so that the module
    # docstring's tie rule decides it and rounding does not. There `alive` adds the same number to H and to A, so their
    # difference is exactly 0 where only `alive` is known, instead of two running sums that drift apart by ulps (and
    # would put the change of team where the rounding noise peaks, not on the gap's first frame). A new array: the
    # input is never written.
    emissions = emissions - emissions[:, H:H + 1]
    penalty = ((0.0, switch_team, switch_dead),          # from H, to H, A, D
               (switch_team, 0.0, switch_dead),          # from A
               (switch_dead, switch_dead, 0.0))          # from D
    # The columns are the label codes, so the path is the labels.
    return _best_path(emissions, penalty, (min_possession, min_possession, min_dead), PREFERENCE, team_cost)


def restart_viterbi(emissions, switch_team, switch_dead, min_possession, min_dead, switch_restart):
    """int8 four-state path, columns H, A, D_H, D_A (RESTART_STATES), of a valid path of maximal score through
    `emissions` ((n_frames, 4)): the restart states' changes, minimums and ties (module docstring). RESTART_LABELS[path]
    are its labels. n_frames = 0 gives an empty path.
    """
    switch_team, switch_dead, min_possession, min_dead = _check_costs(
        switch_team, switch_dead, min_possession, min_dead)
    switch_restart = _check_switch_restart(switch_restart)
    emissions = _check_emissions(emissions, len(RESTART_STATES))
    emissions = emissions - emissions[:, H:H + 1]        # viterbi's exact ties, for the same reason
    cross = switch_dead + switch_restart                 # a restart by the team the vote did not name
    penalty = ((0.0, switch_team, switch_dead, switch_dead),     # from H, to H, A, D_H, D_A
               (switch_team, 0.0, switch_dead, switch_dead),     # from A
               (switch_dead, cross, 0.0, math.inf),              # from D_H: D_A never follows it
               (cross, switch_dead, math.inf, 0.0))              # from D_A: nor D_H this
    minimum = (min_possession, min_possession, min_dead, min_dead)
    return _best_path(emissions, penalty, minimum, RESTART_PREFERENCE)


def _best_path(emissions, penalty, minimum, preference, team_cost=None):
    """int8 column indices of a valid path of maximal score through `emissions` ((n_frames, k), finite): the recursion
    over run starts of the module docstring, over any k states. penalty[r][s] is what a change from r to s costs (inf:
    forbidden; the diagonal is not read), minimum[s] the shortest run of s, and `preference` the k states in the order a
    tie is decided. With team_cost (an (n_frames,) float64 array, checked), a change between columns H and A whose new
    run starts at frame t costs team_cost[t] instead of its penalty. n_frames = 0 gives an empty path.
    """
    n_frames, n_states = emissions.shape
    path = np.empty(n_frames, dtype=np.int8)
    if n_frames == 0:
        return path
    states = range(n_states)
    # The states that can be left for s, with what the change costs, in the order a tie between them is decided; a
    # forbidden change is no way in at all.
    predecessors = tuple(tuple((other, penalty[other][state]) for other in preference
                               if other != state and penalty[other][state] != math.inf) for state in states)
    # cumulative[s][k]: the emissions of s over the first k frames. The run that began at frame a scores
    # entry + cumulative[s][t + 1] - cumulative[s][a] at frame t.
    cumulative = [[0.0] + np.cumsum(emissions[:, state]).tolist() for state in states]

    # best[s]: max of Q over the starts that make a mature run at the frame being decoded, and start[s] the start it
    # came from; 0.0 and frame 0 are the first run, mature from its first frame. entry[s][a] is Q, origin[s][a] the
    # state the run was entered from, and started[s][t] the start that best[s] had at frame t.
    best = [0.0] * n_states
    start = [0] * n_states
    entry = [[0.0] * n_frames for _ in states]
    origin = [[0] * n_frames for _ in states]
    started = [[0] * n_frames for _ in states]
    # Which loop runs is chosen here, once: without a per-frame team cost it is the loop as it always was.
    if team_cost is None:
        for frame in range(1, n_frames):
            mature = [cumulative[state][frame] + best[state] for state in states]  # L[state][frame - 1]
            for state in states:
                value, source = -math.inf, preference[0]
                for other, cost in predecessors[state]:
                    candidate = mature[other] - cost
                    if candidate > value:
                        value, source = candidate, other
                entry[state][frame] = value - cumulative[state][frame]
                origin[state][frame] = source
            for state in states:
                first = frame - minimum[state] + 1  # the latest start that has reached the minimum by this frame
                if first >= 1 and entry[state][first] > best[state]:
                    best[state], start[state] = entry[state][first], first
                started[state][frame] = start[state]
    else:
        # The same loop, but a change between H and A into this frame costs team_cost[frame]: the run it starts is
        # entered at that price, which Q (entry) carries for as long as the run lasts. The same candidates in the same
        # order and the same strict comparison, so the tie order is the loop's above, and a team_cost equal to the
        # penalty on every frame gives the same floats and the same path.
        priced = tuple(tuple((other, cost, (other, state) in ((H, A), (A, H))) for other, cost in predecessors[state])
                       for state in states)
        frame_cost = team_cost.tolist()
        for frame in range(1, n_frames):
            mature = [cumulative[state][frame] + best[state] for state in states]  # L[state][frame - 1]
            change = frame_cost[frame]
            for state in states:
                value, source = -math.inf, preference[0]
                for other, cost, team in priced[state]:
                    candidate = mature[other] - (change if team else cost)
                    if candidate > value:
                        value, source = candidate, other
                entry[state][frame] = value - cumulative[state][frame]
                origin[state][frame] = source
            for state in states:
                first = frame - minimum[state] + 1
                if first >= 1 and entry[state][first] > best[state]:
                    best[state], start[state] = entry[state][first], first
                started[state][frame] = start[state]

    # The block's last run needs no minimum, so a run that began fewer than m frames ago may end the path as well.
    final_score, final_state, final_start = -math.inf, preference[0], 0
    for state in preference:
        value, begun = best[state], start[state]
        for first in range(max(1, n_frames - minimum[state] + 1), n_frames):
            if entry[state][first] > value:
                value, begun = entry[state][first], first
        if cumulative[state][n_frames] + value > final_score:
            final_score, final_state, final_start = cumulative[state][n_frames] + value, state, begun

    stop, state, begun = n_frames, final_state, final_start
    while True:
        path[begun:stop] = state
        if begun == 0:
            return path
        stop, state = begun, origin[state][begun]
        begun = started[state][stop - 1]


def marginals(emissions, switch_team, switch_dead, min_possession, min_dead, team_cost=None):
    """(n_frames, 3) float64: the probability of H, A and D at each frame under the model in the module docstring.

    The weight of a valid path is exp(its score), and the marginal of (frame, state) is the total weight of the valid
    paths that hold the state there, over the total weight of all valid paths. Each row sums to 1 (to rounding).
    Same arguments as `viterbi`, team_cost included (a change between H and A into frame t costs team_cost[t]; None
    runs the passes as they always ran; ValueError for a team_cost of the wrong shape or values); n_frames = 0 gives an
    empty array. The memory is n_frames * (2 * min_possession + min_dead) * 8 bytes, the forward table below.

    A forward-backward pass over the expanded chain: state s is split into substates "run length so far" 1..m_s, the
    last with a self-loop, so a run is born in substate 1 of s from the *last* substate of another state, at the cost of
    that change, and cannot be left before it has reached its m_s. A path starts in the last substate (the first run
    needs no length) and ends in any substate (neither does the last), exactly as the textbook form of `viterbi`. Each
    frame is one vectorised step over all substates in log space with `logaddexp`; the marginal of a state is the sum,
    over its substates, of exp(forward + backward - log of the total weight).

    Both passes subtract their own maximum from every frame's vector. That moves a frame's forward and backward values
    by a scalar each, which the per-frame normalisation at the end divides out (forward * backward, summed over the
    substates, is the same total weight at every frame), so no running sum of emissions, hundreds of thousands of
    nats long, ever enters a subtraction. The recursion over run starts in `viterbi` has such sums, which is why it
    cannot be reused here: its cancellation is harmless for a maximum and not for a probability of 1e-12.
    """
    switch_team, switch_dead, min_possession, min_dead = _check_costs(
        switch_team, switch_dead, min_possession, min_dead)
    emissions = _check_emissions(emissions)
    n_frames = len(emissions)
    if team_cost is not None:
        team_cost = _check_team_cost(team_cost, n_frames)
    posterior = np.empty((n_frames, 3))
    if n_frames == 0:
        return posterior

    size = (min_possession, min_possession, min_dead)           # substates of each state, by state code
    first = np.array([0, size[0], size[0] + size[1]])           # index of each state's first substate (run length 1)
    last = first + np.array(size) - 1                           # ... and of its last (the minimum, with the self-loop)
    owner = np.repeat(np.arange(3), size)                       # the state of each substate
    # link[r, s]: the log-weight of a change from r to s; -inf on the diagonal, which is not a change.
    link = -np.array([[0.0, switch_team, switch_dead],
                      [switch_team, 0.0, switch_dead],
                      [switch_dead, switch_dead, 0.0]])
    np.fill_diagonal(link, -np.inf)

    # forward[t, x]: log of the total weight of the valid prefixes through frame t that end in substate x, emission at
    # t included, less a scalar per frame. A prefix that starts in a substate other than a last one is not valid.
    forward = np.empty((n_frames, len(owner)))
    forward[0] = -np.inf
    forward[0, last] = emissions[0]
    forward[0] -= forward[0].max()
    # Which loops run is chosen here, once: without a per-frame team cost they are the loops as they always were. With
    # one, each pass is the same step with the change between H and A priced at the cost of the frame its new run
    # starts on: the birth into frame t at team_cost[t], the leave from frame t towards t + 1 at team_cost[t + 1].
    priced = None if team_cost is None else link.copy()
    if team_cost is None:
        for frame in range(1, n_frames):
            previous, row = forward[frame - 1], forward[frame]
            mature = previous[last]                                          # a run that may be left, per state
            born = np.logaddexp.reduce(mature[:, None] + link, axis=0)       # into each state's substate 1
            row[1:] = previous[:-1]                                          # a run one frame older
            row[first] = -np.inf                                             # (the shift put another state's here)
            row[last] = np.logaddexp(row[last], mature)                      # the self-loop of the last substate
            row[first] = np.logaddexp(row[first], born)                      # ... substate 1 when the minimum is 1
            row += emissions[frame][owner]
            row -= row.max()
    else:
        for frame in range(1, n_frames):
            priced[H, A] = priced[A, H] = -team_cost[frame]
            previous, row = forward[frame - 1], forward[frame]
            mature = previous[last]
            born = np.logaddexp.reduce(mature[:, None] + priced, axis=0)
            row[1:] = previous[:-1]
            row[first] = -np.inf
            row[last] = np.logaddexp(row[last], mature)
            row[first] = np.logaddexp(row[first], born)
            row += emissions[frame][owner]
            row -= row.max()

    # backward[x]: log of the total weight of every continuation after frame t from substate x, emission at t not
    # included, less a scalar; at the last frame every substate may end the path, so every continuation is empty.
    # Folded into `forward` on the way down, which then holds forward * backward: no second table.
    backward = np.zeros(len(owner))
    if team_cost is None:
        for frame in range(n_frames - 2, -1, -1):
            ahead = emissions[frame + 1][owner] + backward                   # the next frame, its emission paid
            # np.empty_like is safe: the shift below fills every entry but the last, and `last` covers the last index.
            following = np.empty_like(ahead)
            following[:-1] = ahead[1:]                                       # a run one frame older: the only way on
            leave = np.logaddexp.reduce(ahead[first][None, :] + link, axis=1)  # a mature run ends: into another state
            following[last] = np.logaddexp(ahead[last], leave)               # ... or goes on in the last substate
            following -= following.max()
            backward = following
            forward[frame] += backward
    else:
        for frame in range(n_frames - 2, -1, -1):
            priced[H, A] = priced[A, H] = -team_cost[frame + 1]
            ahead = emissions[frame + 1][owner] + backward
            following = np.empty_like(ahead)
            following[:-1] = ahead[1:]
            leave = np.logaddexp.reduce(ahead[first][None, :] + priced, axis=1)
            following[last] = np.logaddexp(ahead[last], leave)
            following -= following.max()
            backward = following
            forward[frame] += backward

    for lo in range(0, n_frames, CHUNK):
        rows = forward[lo:lo + CHUNK]
        rows -= rows.max(axis=1, keepdims=True)
        np.exp(rows, out=rows)
        states = np.add.reduceat(rows, first, axis=1)
        posterior[lo:lo + CHUNK] = states / states.sum(axis=1, keepdims=True)
    return posterior


def _check_emissions(emissions, columns=3):
    """`emissions` as an (n_frames, columns) float64 array; ValueError naming what is wrong with its shape or values."""
    emissions = np.asarray(emissions, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] != columns:
        raise ValueError("hmm: emissions must be an (n_frames, {}) array, got shape {}".format(columns,
                                                                                            emissions.shape))
    if not np.isfinite(emissions).all():
        raise ValueError("hmm: emissions must be finite, but {} of {} values are NaN or infinite".format(
            int((~np.isfinite(emissions)).sum()), emissions.size))
    return emissions


def _check_costs(switch_team, switch_dead, min_possession, min_dead):
    """The four parameters as float, float, int, int; ValueError naming the first one that is wrong."""
    for name, value in (("switch_team", switch_team), ("switch_dead", switch_dead)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
            raise ValueError("hmm: {} must be a finite number of nats >= 0, got {!r}".format(name, value))
    # numbers.Integral admits numpy integers too, for sweeps over np.arange(...) in a notebook; bool is excluded.
    for name, value in (("min_possession", min_possession), ("min_dead", min_dead)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError("hmm: {} must be a whole number of frames >= 1, got {!r}".format(name, value))
    return float(switch_team), float(switch_dead), int(min_possession), int(min_dead)


def _check_restart(restart_source, switch_restart):
    """ValueError naming restart_source or switch_restart, whichever is wrong; a finite switch_restart needs a source.
    Whether the source is one of the block's is restart_log_likelihoods' check."""
    if restart_source is not None and (not isinstance(restart_source, str) or not restart_source):
        raise ValueError("hmm: restart_source must be the name of an evidence source (a non-empty string), or left "
                         "out, got {!r}".format(restart_source))
    if _check_switch_restart(switch_restart) != math.inf and restart_source is None:
        raise ValueError("hmm: switch_restart = {!r} is read only with restart_source, and none is given; name the "
                         "evidence source that says who restarts, or leave switch_restart out".format(switch_restart))


def _check_switch_restart(switch_restart):
    """switch_restart as a float > 0 (inf allowed); ValueError otherwise, NaN and bool included."""
    if (isinstance(switch_restart, (bool, np.bool_)) or not isinstance(switch_restart, numbers.Real)
            or not switch_restart > 0):
        raise ValueError("hmm: switch_restart must be a number of nats > 0, or inf (a restart by the team the vote did "
                         "not name is then forbidden), got {!r}".format(switch_restart))
    return float(switch_restart)


def _check_hazard(hazard_source, hazard_scale, switch_team, restart_source, team_neutral_dead):
    """hazard_scale as a float (DEFAULT_HAZARD_SCALE where None is given with a source; None without a source).
    ValueError naming hazard_source or hazard_scale, whichever is wrong, or the setting a hazard cannot go with: a scale
    without a source, a source that is not a non-empty string, a scale that is not a finite number >= 0,
    restart_source, team_neutral_dead, and switch_team below HAZARD_FLOOR. Whether the source is one of the block's is
    _hazard_source's check."""
    if hazard_source is None:
        if hazard_scale is not None:
            raise ValueError("hmm: hazard_scale = {!r} is read only with hazard_source, and none is given; name the "
                             "evidence source whose `home` is the hazard, or leave hazard_scale out".format(
                                 hazard_scale))
        return None
    if not isinstance(hazard_source, str) or not hazard_source:
        raise ValueError("hmm: hazard_source must be the name of an evidence source (a non-empty string), or left "
                         "out, got {!r}".format(hazard_source))
    if hazard_scale is None:
        hazard_scale = DEFAULT_HAZARD_SCALE
    elif (isinstance(hazard_scale, (bool, np.bool_)) or not isinstance(hazard_scale, numbers.Real)
            or not math.isfinite(hazard_scale) or hazard_scale < 0):
        raise ValueError("hmm: hazard_scale must be a finite number >= 0 (it multiplies the hazard's log-odds; 0 "
                         "leaves switch_team on every frame), got {!r}".format(hazard_scale))
    if restart_source is not None:
        raise ValueError("hmm: hazard_source {!r} prices the three states' changes of team, and restart_source {!r} "
                         "asks for the four restart states; leave one of them out".format(hazard_source,
                                                                                         restart_source))
    if team_neutral_dead:
        raise ValueError("hmm: hazard_source {!r} is not written for team_neutral_dead = true (P35 prices the changes "
                         "of today's emissions); leave one of them out".format(hazard_source))
    if float(switch_team) < HAZARD_FLOOR:
        raise ValueError("hmm: hazard_source {!r} needs switch_team >= HAZARD_FLOOR ({} nat), got {!r}: below it a "
                         "frame at the hazard's base rate would cost more than switch_team, and hazard_scale 0 would "
                         "not be the decoder without a hazard".format(hazard_source, HAZARD_FLOOR, switch_team))
    return float(hazard_scale)


def _hazard_source(evidence, hazard_source, n_frames):
    """The source of `evidence` (ctx.evidence) named hazard_source, whose `home` is the hazard. ValueError if the block
    has no such source, if its weight is not 1 (hazard_scale is the hazard's strength: a weight would be read by
    nothing), or if its length is not the block's."""
    if hazard_source not in evidence:
        raise ValueError("hmm: hazard_source {!r} names no evidence source of this block; its sources are {}".format(
            hazard_source, ", ".join(repr(name) for name in evidence) or "none"))
    source = evidence[hazard_source]
    if source.weight != 1.0:
        raise ValueError("hmm: evidence {!r} is the hazard_source, whose weight is not read: its strength is "
                         "hazard_scale; leave its weight at 1, got {!r}".format(hazard_source, source.weight))
    if len(source.home) != n_frames:
        raise ValueError("hmm: evidence {!r} has {} frames of hazard, the block has {}".format(
            hazard_source, len(source.home), n_frames))
    return source


def _check_team_cost(team_cost, n_frames):
    """team_cost as an (n_frames,) float64 array; ValueError naming what is wrong with its shape or values: one finite
    number of nats >= 0 per frame."""
    team_cost = np.asarray(team_cost, dtype=np.float64)
    if team_cost.shape != (n_frames,):
        raise ValueError("hmm: team_cost must be one number of nats per frame, shape ({},), got shape {}".format(
            n_frames, team_cost.shape))
    finite = np.isfinite(team_cost)
    if not finite.all() or (team_cost < 0).any():
        raise ValueError("hmm: team_cost must be finite numbers of nats >= 0, but {} of {} values are NaN, infinite "
                         "or negative".format(int((~finite).sum() + (team_cost[finite] < 0).sum()), team_cost.size))
    return team_cost


def _check_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not 0.0 < eps < 0.5:
        raise ValueError("hmm: eps must be a number in (0, 0.5), got {!r}".format(eps))
