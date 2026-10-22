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

## What is here now: `majority_vote` (a stand-in)

Step 2 of `starter/baseline.py`. Per frame, the `home` values within ±`half_width` frames (clipped to the block) are
summed into a home score and an away score; the higher one wins, ties go to home, and a window where no source had
an opinion gets `fallback`. With one hard 0/1 source this is exactly `baseline.smooth()`, which is how the pipeline
rebuilds all four published baselines byte for byte.

| parameter | default | meaning |
|---|---|---|
| `half_width` | — (required) | frames each side; 50 = ±2 s. 0 = the raw per-frame labels |
| `fallback` | `"D"` | the label where nobody voted in the window |

What it cannot do, and why it will be replaced:

- It **ignores `alive`**, so `D` only comes from the fallback — wherever the tracking lost the ball. That is wrong on
  66% of dead-ball frames, where the ball is visible ([T12](../../docs/traps.md)).
- It has **no notion of a possession as an interval**: what survives the vote is kept, so blips still split
  possessions.
- It pools several sources with a weighted mean of `home`, which only approximates a likelihood weight.

## The decoder this stage is waiting for: P06

A hidden (semi-)Markov model over `H`, `A`, `D`, decoded with Viterbi per block. The contract is shaped for it:

- **Emissions from the two channels, as likelihood ratios.** Each channel says how much more likely each state
  became after seeing the frame, relative to a prior: `home = h` adds `log(h / π)` to H, `log((1 − h) / (1 − π))`
  to A and 0 to D (π ≈ 0.5); `alive = a` adds `log(a / π_a)` to H *and* to A, and `log((1 − a) / (1 − π_a))` to D.
  A missing channel adds 0 to every state, so it cannot push the decoder anywhere — which is exactly what NaN
  promises. (Writing `log P(H) = log a + log h` instead, and 0 for a missing `alive`, would hand D a head start on
  every frame where only `home` is known.) Multiply each source's terms by its `weight`.
- **Clip probabilities to [ε, 1 − ε] first.** `nearest_player` reports hard 0s and 1s, whose logs are −∞.
- **A switch penalty and duration priors** decide how much evidence a change has to beat. Tune them on P01's
  benchmark of other matches, never on the public window ([T01](../../docs/traps.md)).
- **Return the posterior.** It is what tells a refiner where an `X` is free (inside a possession) and where it is
  expensive (at a hand-over) — [EVALUATION.md](../../EVALUATION.md), P12.

## Adding a decoder

1. Write `pipeline/decode/<name>.py` with `decode(ctx, **params) -> Decoded`.
2. Add one line to `REGISTRY` in [`__init__.py`](__init__.py).
3. Point a config's `[decoder]` at it: `kind = "<name>"` plus its parameters.
4. Run `python pipeline/selftest.py` — contracts still hold — and `python pipeline/run.py <config> --score`.

The runner checks what comes back: one int8 code per frame of the block, codes in `H, A, D, X`, and a posterior of
the right shape if there is one.
