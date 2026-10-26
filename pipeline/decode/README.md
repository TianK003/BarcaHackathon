# Stage 3 — decode

**The point of this stage:** turn per-frame opinions into a *sequence* of labels that looks like football —
possessions that are intervals, stoppages between them, and as few spurious changes as the evidence allows. This is
where the ranking metric is won or lost: `spellF1` matches whole possessions, so one flicker splits a possession and
costs three ([T04](../../docs/traps.md)). The raw nearest-player labels report 131 possessions in the public window
against 17 real ones; the decoder's job is to get from the first number to the second without losing the real
changes.

## In → out

| in | out |
|---|---|
| `ctx.block` — the frames to label: one graded half | `Decoded(labels, posterior)` for exactly those frames |
| `ctx.evidence` — name → `Evidence`, **cut down to the block**, in config order | `labels`: int8 codes `H, A, D, X` = 0, 1, 2, 3 (the scorer's own) |
| `ctx.run.inputs` — the whole match, if positions are needed (slice with `ctx.block.take(...)`) | `posterior`: `(len, 3)` P(H), P(A), P(D), or `None` |
| keyword parameters from `[decoder]` | |

```python
def decode(ctx, *, half_width, fallback="D"):    # ctx: BlockContext
    ...
    return Decoded(labels=labels)                # int8, len(ctx.block) frames
```

## Blocks: why a decoder never sees across halftime

The scorer evaluates each graded half on its own, so a possession can never run through halftime, and a decoder that
smooths across the break invents one ([T11](../../docs/traps.md)). The runner enforces this for everyone. It calls
the decoder once per block:

| block | frames | note |
|---|---|---|
| 1st half | 11,185 – 90,651 | graded from the kick-off |
| 2nd half | 110,808 – 190,197 | graded from 2,934 frames **before** the kick-off at 113,742; all of those are `D` ([T02](../../docs/traps.md)) |

The runner also hands the decoder evidence that has already been cut to the block, and writes `X` on every frame
outside the blocks, where nothing is graded. The one exception is `scope = "file"`: a single block over the whole
file, which only `configs/smooth50_repro.toml` uses, because the organisers' baseline smooths straight through
halftime and reproducing it byte for byte means doing the same. The runner prints a warning whenever it is used.

## `majority_vote` (a stand-in)

Step 2 of `starter/baseline.py`. Per frame, the `home` values within ±`half_width` frames (clipped to the block) are
summed into a home score and an away score; the higher one wins, ties go to home, and a window where no source had
an opinion gets `fallback`. With one hard 0/1 source this is exactly `baseline.smooth()`, which is how the pipeline
rebuilds all four published baselines byte for byte.

| parameter | default | meaning |
|---|---|---|
| `half_width` | — (required) | frames each side; 50 = ±2 s. 0 = the raw per-frame labels |
| `fallback` | `"D"` | the label where nobody voted in the window |

What it cannot do, and why `hmm` replaces it:

- It **ignores `alive`**, so `D` only comes from the fallback — wherever the tracking lost the ball. That is wrong on
  66% of dead-ball frames, where the ball is visible ([T12](../../docs/traps.md)).
- It has **no notion of a possession as an interval**: what survives the vote is kept, so blips still split
  possessions.
- It pools several sources with a weighted mean of `home`, which only approximates a likelihood weight.

## `hmm`: the global decoder (P06)

A three-state hidden Markov model over `H`, `A` and `D`, decoded by Viterbi once per block, so a possession is an
interval whose edges have to be paid for; with `restart_source`, `D` splits into two restart states
([below](#restart-states-p17)). Its module docstring ([`hmm.py`](hmm.py)) is the specification; the checks in
`pipeline/selftest.py` pin it against a brute force over every path.

```toml
[decoder]
kind = "hmm"
switch_team = 8        # nats a change of team costs
switch_dead = 16       # nats a change into or out of D costs
min_possession = 25    # frames; default 1
min_dead = 25          # frames; default 1
```

**Emissions.** Each channel adds a log-likelihood ratio against a prior of one half, times the source's `weight` (an
exponent on its likelihood: weight 2 counts the source twice, weight 0 not at all). Probabilities are clipped to
`[eps, 1 - eps]` first, since `nearest_player` reports hard 0s and 1s unless it is given a `confidence`.

| channel, where it is known | H | A | D |
|---|---|---|---|
| `home = h` | `log 2h` | `log 2(1 - h)` | 0 |
| `alive = a` | `log 2a` | `log 2a` | `log 2(1 - a)` |

A channel that is `NaN` adds 0 to every state. That is what "no opinion" means, and it is why `P(H) = alive * home` was
not used: it would put a missing `alive` at a coin flip and hand `D` a head start wherever only `home` is known.

**The path.** Its score is the sum of the emissions along it, less `switch_team` for every `H` to `A` change and
`switch_dead` for every change into or out of `D`; there is no penalty at a block's edges. A path is valid when every
run of `H` or `A` lasts at least `min_possession` frames and every run of `D` at least `min_dead`, except the block's
first and last run, which its edges may have cut short. The decoder returns a valid path of maximal score.

| parameter | default | meaning |
|---|---|---|
| `switch_team` | — (required) | nats of evidence a change of team has to beat; a blip of n contrary frames is believed when n times its per-frame evidence exceeds twice this |
| `switch_dead` | — (required) | the same for a stoppage; separate, because `alive` evidence is lopsided where `home` evidence is not: a seen ball is weak evidence of play (0.34 nats a frame) and a missing one is not weak evidence of a stoppage (1.39 nats) |
| `min_possession` | 1 | the shortest run of `H` or of `A`; a hard floor under the penalty, which makes a run grow to its length where the evidence is half there |
| `min_dead` | 1 | the shortest run of `D` |
| `eps` | 1e-6 | the clip; a numerical guard, not a knob |
| `posterior` | `false` | `true` also returns `Decoded.posterior`, the probability of `H`, `A` and `D` per frame ([below](#the-posterior)); the labels are the same either way |
| `restart_source` | none | the name of the evidence source whose `home` says who restarts after a stoppage: turns on the restart states ([below](#restart-states-p17)) |
| `switch_restart` | `inf` | with `restart_source`: nats a restart by the team the vote did not name costs on top of `switch_dead`; `inf` forbids it |

All four numbers are validated (no bool, no NaN or infinity, minimums are whole numbers of frames >= 1) and named in the
error, and so are the restart states' two: `switch_restart` must be > 0 or `inf`, and is refused when finite without a
`restart_source`, which must name one of the block's sources. Tune the numbers on P01's benchmark, never on the public
window, which since 23 Oct 2026 is a rule and not only advice ([T01](../../docs/traps.md)). The benchmark's choice is
`switch_team` 8, `switch_dead` 32, `min_possession` 75 and `min_dead` 1 ([R006 and R007](../../docs/results.md), kept in
[`configs/p06_hmm.toml`](../configs/p06_hmm.toml)); that config does not clear P06's bar and is not the submission
config.

**Ties.** Equal scores are decided the same way every time: a path extends its current run first, then comes from `D`,
then from `H`, then from `A`; the block ends in `D`, then `H`, then `A`. So a block without evidence is all `D`, and a
change of team across a stretch without evidence lands on the stretch's first frame. With restart states the order is
`D_H`, `D_A`, `H`, `A`.

**Cost.** About 2 µs a frame (80,000 frames with both minimums at 50 take 0.15 s): a plain-Python recursion over run
starts with a running maximum, where the textbook chain with one substate per run length would cost O(minimum) a frame.
The recursion is written once over any number of states (a matrix of change costs in which `inf` forbids a change, a
minimum per state, a tie order); the four restart states take about 1.4 times the three's time.

### The posterior

```toml
[decoder]
kind = "hmm"
switch_team = 8
switch_dead = 16
posterior = true       # default false: Decoded.posterior is None
```

With `posterior = true`, `Decoded.posterior` is a `(len(block), 3)` float64 array: the probability of `H`, `A` and `D`
at each frame under the same model, so a refiner can tell where an `X` is free (inside a possession, where `H` or `A` is
close to 1) and where it would be expensive (at a hand-over). A valid path has weight `exp(score)`, and a frame's
probability of a state is the total weight of the valid paths that hold it there, over the total weight of all valid
paths. Each row sums to 1.

- **The labels do not change.** They are the best single path (Viterbi), not the per-frame argmax of the posterior, and
  the two part company where a stretch is close. A check pins that the labels are identical with the posterior on and
  off.
- **It is the model's belief, no more.** It is as calibrated as the evidence it is given; `nearest_player`'s
  `confidence` and `ball_presence`'s two values are what make it more or less sure.
- **How it is computed.** A forward-backward pass over the textbook chain (each state split into "run length so far"
  substates, the last with a self-loop, the first and last run exempt from the minimum), in log space, one vectorised
  step per frame. The recursion over run starts that Viterbi uses sums the emissions over the whole block, and the
  cancellation in those sums is harmless for a maximum but not for a probability of 1e-12, so it is not reused. The
  pass is checked against an enumeration of every valid path (`pipeline/selftest.py`, blocks of up to 6 frames, with
  blocks shorter than a minimum among them) and against reading a long block backwards.
- **Cost.** It is O(`min_possession` + `min_dead`) a frame and holds `n_frames * (2 * min_possession + min_dead) * 8`
  bytes (96 MB for 80,000 frames with both minimums at 50), so it is off by default and no benchmark grid asks for it.
  80,000 frames with both minimums at 50 take about 2 s.
- **Three states only.** `posterior = true` with `restart_source` is refused.

### Restart states (P17)

```toml
[[evidence]]
kind = "restart_team"
model = "pipeline/models/dead_ball_hgb.json"
span = "stoppage_end"  # one vote per stoppage, on its last frame (no tail)
corner = 0.9           # a confidence per type that votes, measured by restarts.py stats

[decoder]
kind = "hmm"
switch_team = 8
switch_dead = 16
restart_source = "restart_team"   # the source's name: its `home` says who restarts
switch_restart = inf               # default inf: the first team after a stoppage is its restart team
```

The Laws of the Game decide who restarts after a stoppage, and that team has the first possession after it.
`restart_team` reads who it is from the players' arrangement during the stoppage. As a plain `home` vote on a fixed span
after the restart it gained +0.004 spellF1 ([R014](../../docs/results.md)): a fixed span keeps pushing after the first
possession has ended, and it starts at the source's own restart estimate, not at the decoder's `D` edge. With
`restart_source`, the decoder carries the restart team itself: `D` splits into `D_H` ("dead, home restarts") and `D_A`
("dead, away restarts"), so the team is read inside the stoppage, carried to whichever frame the decoder ends `D` on,
and decides only the first team after it. Later evidence ends that possession as it ends any other.

| | `H` | `A` | `D_H` | `D_A` |
|---|---|---|---|---|
| any source's `alive = a` | `log 2a` | `log 2a` | `log 2(1 - a)` | `log 2(1 - a)` |
| any other source's `home = h` | `log 2h` | `log 2(1 - h)` | 0 | 0 |
| the named source's `home = h` | 0 | 0 | `log 2h` | `log 2(1 - h)` |

| change | cost |
|---|---|
| `H` or `A` to `D_H` or `D_A` | `switch_dead` |
| `D_H` to `H`, `D_A` to `A` | `switch_dead` |
| `D_H` to `A`, `D_A` to `H` | `switch_dead + switch_restart` (`inf`: forbidden) |
| `D_H` to `D_A`, and back | forbidden: a stoppage has one restart team |

- **Minimums and labels.** `min_possession` covers `H` and `A`, `min_dead` covers `D_H` and `D_A`, and both dead states
  are written `D`.
- **One frame of evidence.** `restart_team`'s span `"stoppage_end"` writes its vote on the stoppage's last frame only,
  at the type's confidence. That is one observation of who restarts, so the confidence stays a likelihood: written on
  every frame of a 20 s stoppage it would count 500 times. A vote on any frame of a `D` run counts for all of it.
- **The `D` edges hardly move.** The named source says nothing about `H` against `A`. On its one frame it raises the
  better dead state by at most `w log 2c` (0.61 nats at 0.92), where a vote after the restart raised the alive states
  by as much on every frame of its span.
- **Without restart evidence it is the three-state decoder.** If the named source's `home` is `NaN` everywhere, or its
  weight is 0, `D_H` and `D_A` are the same state twice over: every three-state path has a four-state twin of the same
  score, and the tie order `D_H`, `D_A`, `H`, `A` picks the twin of the three-state choice. The checks hold that on
  random blocks for any `switch_restart`, and the four-state decoder against every path of blocks of up to 8 frames.
- **Without `restart_source`** the decoder runs the three-state code path, to the label: a copy of the routine as it was
  before the restart states pins it.
- **Refused:** `posterior = true` with it, a `restart_source` that names none of the block's sources (the error lists
  them), and a finite `switch_restart` without one, which nothing would read.

**What it does not do yet.**

- **No duration priors beyond the minimums:** once a run is long enough it is as likely to last two minutes as five
  seconds. A semi-Markov version would take a prior over run lengths where the minimum is a hard step.
- **The restart states know who restarts, nothing more:** not how long that team keeps the ball, and nothing about a
  stoppage the `alive` evidence does not decode (no `D` run, no restart team). They have no posterior.
- **`D` is only as good as its `alive` evidence.** `nearest_player` says nothing about it, so with that alone the
  decoder chooses `D` rarely, and only in gaps: `H`, `D`, `A` beats `H`, `A` across a gap when `switch_team` is more
  than twice `switch_dead` and the gap is at least `min_dead` frames, and with `switch_dead = 0` a trailing stretch
  without evidence ends in `D` by the tie preference. `ball_presence` (whether the tracker sees the ball) was the one
  `alive` source until P05, and its `D` scored 0.5676 on alive/dead F1 on the benchmark against `nearest_w50`'s 0.6094
  ([R007](../../docs/results.md)). A dead-ball source (P05, `dead_ball_model`) is what gave it something real to
  decode, and it is the `alive` source of `main.toml` (alive/dead F1 0.9726 in R010, on a benchmark that
  overstates it, T38, T39).

## Adding a decoder

1. Write `pipeline/decode/<name>.py` with `decode(ctx, **params) -> Decoded`.
2. Add one line to `REGISTRY` in [`__init__.py`](__init__.py).
3. Point a config's `[decoder]` at it: `kind = "<name>"` plus its parameters.
4. Run `python pipeline/selftest.py` — contracts still hold — and `python pipeline/run.py <config> --score`.

The runner checks what comes back: one int8 code per frame of the block, codes in `H, A, D, X`, and a posterior of
the right shape if there is one.
