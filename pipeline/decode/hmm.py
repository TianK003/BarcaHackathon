"""P06's decoder: a three-state hidden Markov model over H, A and D, decoded by Viterbi, once per block.

    [decoder]
    kind = "hmm"
    switch_team = 8        # nats a change of team (H <-> A) costs
    switch_dead = 16       # nats a change into or out of D costs
    min_possession = 25    # frames: the shortest run of H or of A (default 1: no minimum)
    min_dead = 25          # frames: the shortest run of D (default 1)
    eps = 1e-6             # probabilities are clipped to [eps, 1 - eps] before the log (default 1e-6)
    posterior = false      # true: also return P(H), P(A), P(D) per frame (default false: Decoded.posterior is None)

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

A path is one label per frame of the block. Its score is the sum of e[t, s_t] along it, less switch_team for every
H <-> A change and switch_dead for every change into or out of D. There is no penalty at the block's edges. The
decoder returns a *valid* path of maximal score: every run (maximal constant stretch) of H or A lasts at least
min_possession frames and every run of D at least min_dead -- except the block's first and last run, which the edge
of the block may have cut short. They may be the same run, and a block shorter than a minimum is still decodable: it
is its two edge runs, however short.

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

Ties
----
Exactly equal scores are common (a stretch without evidence costs the same whoever holds it) and are decided the same
way every time, in the order the recursion meets its choices: a path prefers to extend its current run (and, between two
ways of doing so, the longer one); failing that to come from D, then from H, then from A; at the end of the block the
final state is D, then H, then A, the longer final run first. Two consequences hold the tie rule in place:

* a block with no evidence at all decodes to all D, like majority_vote's fallback;
* a change of team across a stretch without evidence lands on the stretch's first frame: the run that ends the
  stretch has the earlier start, because it extends the longer.

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

What it does not do yet
-----------------------
* No duration priors beyond the minimums: a run is as likely to last 2 minutes as 5 seconds, once it is long enough.
  A semi-Markov decoder would take a log-prior over run lengths where the minimum is a hard step.
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
CHUNK = 4096                  # frames per step of the posterior's final normalisation: bounds its temporary arrays


def decode(ctx, *, switch_team, switch_dead, min_possession=1, min_dead=1, eps=EPS, posterior=False):
    """Decoded(labels) for ctx.block: the best valid path through the block's emissions.

    With posterior=True, Decoded.posterior is also the (len(block), 3) marginal probability of H, A and D per frame
    under the same model (`marginals`); the labels are the same either way. Otherwise posterior is None.
    """
    _check_costs(switch_team, switch_dead, min_possession, min_dead)
    _check_eps(eps)
    if not isinstance(posterior, (bool, np.bool_)):
        raise ValueError("hmm: posterior must be True or False, got {!r}".format(posterior))
    emissions = log_likelihoods(ctx.evidence.values(), len(ctx.block), eps)
    labels = viterbi(emissions, switch_team, switch_dead, min_possession, min_dead)
    if not posterior:
        return Decoded(labels=labels)
    return Decoded(labels=labels, posterior=marginals(emissions, switch_team, switch_dead, min_possession, min_dead))


def log_likelihoods(sources, n_frames, eps):
    """(n_frames, 3) float64 emissions in nats, columns H, A, D: the sum over `sources` of the rule in the docstring.

    `sources` is an iterable of Evidence, each of length n_frames. A source's weight must be finite and >= 0 (an
    exponent on its likelihood), whether or not it has an opinion anywhere. Reads the evidence, never writes it: the
    runner hands over read-only arrays.
    """
    _check_eps(eps)
    emissions = np.zeros((n_frames, 3))
    for source in sources:
        weight = source.weight
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError("hmm: evidence {!r} has weight {!r}; a weight is an exponent on its likelihood and must "
                             "be a finite number >= 0".format(source.name, weight))
        if len(source.home) != n_frames or len(source.alive) != n_frames:
            raise ValueError("hmm: evidence {!r} has {} and {} frames (home, alive), the block has {}".format(
                source.name, len(source.home), len(source.alive), n_frames))
        known = ~np.isnan(source.home)
        if known.any():
            home = np.clip(source.home[known], eps, 1.0 - eps)
            emissions[known, H] += weight * np.log(PRIOR_CHANCE * home)
            emissions[known, A] += weight * np.log(PRIOR_CHANCE * (1.0 - home))
        known = ~np.isnan(source.alive)
        if known.any():
            alive = np.clip(source.alive[known], eps, 1.0 - eps)
            in_play = weight * np.log(PRIOR_CHANCE * alive)
            emissions[known, H] += in_play
            emissions[known, A] += in_play
            emissions[known, D] += weight * np.log(PRIOR_CHANCE * (1.0 - alive))
    return emissions


def viterbi(emissions, switch_team, switch_dead, min_possession, min_dead):
    """int8 labels of a valid path of maximal score through `emissions` ((n_frames, 3), columns H, A, D).

    Valid, scored and tied as the module docstring says. n_frames = 0 gives no labels.
    """
    switch_team, switch_dead, min_possession, min_dead = _check_costs(
        switch_team, switch_dead, min_possession, min_dead)
    emissions = _check_emissions(emissions)
    # Every path takes one state per frame, so subtracting a frame's H score from all three states moves every path's
    # score by the same constant: in exact arithmetic that changes no decision. In floats it changes which of several
    # exactly tied paths wins, and that is its purpose: it makes the tie inside a ball gap exact, so that the module
    # docstring's tie rule decides it and rounding does not. There `alive` adds the same number to H and to A, so their
    # difference is exactly 0 where only `alive` is known, instead of two running sums that drift apart by ulps (and
    # would put the change of team where the rounding noise peaks, not on the gap's first frame). A new array: the
    # input is never written.
    emissions = emissions - emissions[:, H:H + 1]
    n_frames = len(emissions)
    labels = np.empty(n_frames, dtype=np.int8)
    if n_frames == 0:
        return labels

    minimum = (min_possession, min_possession, min_dead)  # by state code
    penalty = ((0.0, switch_team, switch_dead),
               (switch_team, 0.0, switch_dead),
               (switch_dead, switch_dead, 0.0))
    # The states that can be left for s, in the order a tie between them is decided.
    predecessors = tuple(tuple(other for other in PREFERENCE if other != state) for state in STATES)
    # cumulative[s][k]: the emissions of s over the first k frames. The run that began at frame a scores
    # entry + cumulative[s][t + 1] - cumulative[s][a] at frame t.
    cumulative = [[0.0] + np.cumsum(emissions[:, state]).tolist() for state in STATES]

    # best[s]: max of Q over the starts that make a mature run at the frame being decoded, and start[s] the start it
    # came from; 0.0 and frame 0 are the first run, mature from its first frame. entry[s][a] is Q, origin[s][a] the
    # state the run was entered from, and started[s][t] the start that best[s] had at frame t.
    best = [0.0, 0.0, 0.0]
    start = [0, 0, 0]
    entry = [[0.0] * n_frames for _ in STATES]
    origin = [[0] * n_frames for _ in STATES]
    started = [[0] * n_frames for _ in STATES]
    for frame in range(1, n_frames):
        mature = [cumulative[state][frame] + best[state] for state in STATES]  # L[state][frame - 1]
        for state in STATES:
            value, source = -math.inf, D
            for other in predecessors[state]:
                candidate = mature[other] - penalty[other][state]
                if candidate > value:
                    value, source = candidate, other
            entry[state][frame] = value - cumulative[state][frame]
            origin[state][frame] = source
        for state in STATES:
            first = frame - minimum[state] + 1  # the latest start that has reached the minimum by this frame
            if first >= 1 and entry[state][first] > best[state]:
                best[state], start[state] = entry[state][first], first
            started[state][frame] = start[state]

    # The block's last run needs no minimum, so a run that began fewer than m frames ago may end the path as well.
    final_score, final_state, final_start = -math.inf, D, 0
    for state in PREFERENCE:
        value, begun = best[state], start[state]
        for first in range(max(1, n_frames - minimum[state] + 1), n_frames):
            if entry[state][first] > value:
                value, begun = entry[state][first], first
        if cumulative[state][n_frames] + value > final_score:
            final_score, final_state, final_start = cumulative[state][n_frames] + value, state, begun

    stop, state, begun = n_frames, final_state, final_start
    while True:
        labels[begun:stop] = state
        if begun == 0:
            return labels
        stop, state = begun, origin[state][begun]
        begun = started[state][stop - 1]


def marginals(emissions, switch_team, switch_dead, min_possession, min_dead):
    """(n_frames, 3) float64: the probability of H, A and D at each frame under the model in the module docstring.

    The weight of a valid path is exp(its score), and the marginal of (frame, state) is the total weight of the valid
    paths that hold the state there, over the total weight of all valid paths. Each row sums to 1 (to rounding).
    Same arguments as `viterbi`; n_frames = 0 gives an empty array. The memory is n_frames * (2 * min_possession +
    min_dead) * 8 bytes, the forward table below.

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
    for frame in range(1, n_frames):
        previous, row = forward[frame - 1], forward[frame]
        mature = previous[last]                                          # a run that may be left, per state
        born = np.logaddexp.reduce(mature[:, None] + link, axis=0)       # into each state's substate 1
        row[1:] = previous[:-1]                                          # a run one frame older
        row[first] = -np.inf                                             # (the shift put another state's here)
        row[last] = np.logaddexp(row[last], mature)                      # the self-loop of the last substate
        row[first] = np.logaddexp(row[first], born)                      # ... which is substate 1 when the minimum is 1
        row += emissions[frame][owner]
        row -= row.max()

    # backward[x]: log of the total weight of every continuation after frame t from substate x, emission at t not
    # included, less a scalar; at the last frame every substate may end the path, so every continuation is empty.
    # Folded into `forward` on the way down, which then holds forward * backward: no second table.
    backward = np.zeros(len(owner))
    for frame in range(n_frames - 2, -1, -1):
        ahead = emissions[frame + 1][owner] + backward                   # the next frame, its emission paid
        following = np.empty_like(ahead)
        following[:-1] = ahead[1:]                                       # a run one frame older: the only way on
        leave = np.logaddexp.reduce(ahead[first][None, :] + link, axis=1)  # a mature run ends: into another state
        following[last] = np.logaddexp(ahead[last], leave)               # ... or goes on in the last substate
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


def _check_emissions(emissions):
    """`emissions` as an (n_frames, 3) float64 array; ValueError naming what is wrong with its shape or values."""
    emissions = np.asarray(emissions, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] != 3:
        raise ValueError("hmm: emissions must be an (n_frames, 3) array, got shape {}".format(emissions.shape))
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


def _check_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not 0.0 < eps < 0.5:
        raise ValueError("hmm: eps must be a number in (0, 0.5), got {!r}".format(eps))
