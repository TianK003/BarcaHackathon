# Stage 4 — refine

**The point of this stage:** last-mile corrections to a decoded half, aimed at how the scorer counts. A refiner
fixes what is cheaper to fix on the label sequence than in the decoder: a blip that would split a possession, a
boundary that sits half a second late, an `X` that would be dragged across a hand-over. Each one is a small, separate,
switchable step, so its effect can be measured on its own — and switched off when it does not pay.

## In → out

| in | out |
|---|---|
| `ctx` — the same `BlockContext` the decoder got: one block, the evidence cut to it, the whole match in `ctx.run.inputs` | a new `Decoded` for exactly the same frames |
| `decoded` — the block's current labels (and posterior, if the decoder made one) | |
| keyword parameters from the `[[refine]]` entry | |

```python
def refine(ctx, decoded, *, min_frames=10):     # ctx: BlockContext, decoded: Decoded
    labels = decoded.labels.copy()               # never modify what you were given
    ...
    return Decoded(labels=labels, posterior=decoded.posterior)
```

Refiners run in config order, after the decoder, one block at a time; each gets the previous one's output. The runner
checks every result the way it checks the decoder's — same length, int8 codes in `H, A, D, X` — so a refiner cannot
leak across halftime either.

```toml
[[refine]]
kind = "min_duration"   # example: this one does not exist yet
min_frames = 10
```

## What is here now

One refiner, `shift_edges` ([`shift_edges.py`](shift_edges.py), P02). It moves each kind of edge of a decoded block by
a fixed number of frames, in case our edges turn out to be systematically early or late against the truth, and not by
the same amount for every kind.

```toml
[[refine]]
kind = "shift_edges"
team = 0           # frames: a change of team (H <-> A)
into_dead = 0      # frames: a change from H or A into D
out_of_dead = 0    # frames: a change from D into H or A
```

A boundary is a change of label inside the block, typed by the pair of labels. A positive value moves it later (the new
label starts later), a negative one earlier. Values are whole numbers of frames with |value| <= 250 (bool is refused).
The code ships with no value of its own: every default is 0, and all three at 0 returns the block exactly as it came,
labels and posterior. The values are for P02's measurement to set ([docs/proposals.md](../../docs/proposals.md)), and
`main.toml` sets them by the declared rule of [R012](../../docs/results.md): team 1, into_dead -2, out_of_dead 0. They
are the pipeline's own lag on the DFL matches, negated (`conventions.py latency` and `shift`, in the
[benchmark README](../benchmark/README.md#conventions-when-our-truth-flips-and-the-edge-shift)); the gap to our truth's
own convention that the public window could add came out 0. The shift is 1 to 2 frames, at 439 boundaries, and its gain
on the benchmark is inside the noise: the entry is there because the rule said so, not because it is shown to help.

**The rule.**

- A boundary touching an `X` never moves, and neither does one at a kick-off frame (`ctx.run.inputs.kickoffs`,
  compared in frames of the match: `ctx.block.start` plus its position).
- Each boundary gets the target `b + offset`; a fixed one keeps `b`.
  1. A target may not cross a fixed boundary: the *j*-th boundary after a fixed one at *p* is not earlier than
     *p + j*, and the *j*-th before one at *q* not later than *q - j*.
  2. Each target is clipped to `[1, len - 1]`.
  3. A pass from left to right sets `t[i] = max(t[i], t[i-1] + 1)`, then one from right to left sets
     `t[i] = min(t[i], t[i+1] - 1)`, with `t = len` beyond the last.
- So the runs keep their labels and their order, and each keeps at least one frame. Nothing is added, removed or merged,
  which is why a shift cannot repair flicker. Step 1 is the one thing added to the clip and the two passes: without it
  a boundary moving past a fixed one would push it. Where no fixed boundary would be pushed the result is the plain clip
  and two passes, which `selftest.py` checks against the algorithm written as loops.
- Unless all three values are 0, the posterior is dropped (`None`): it no longer describes the labels.

`apply(labels, offsets, fixed)` is the same rule as a pure function, with an offset and a flag for each boundary of
`labels`, and `boundaries(labels)` lists them as `(positions, types)`. They are public because P02's measuring tool
moves boundaries by its own offsets with them. A refiner placed after `shift_edges` sees the shifted labels.

## Other candidates (P12, then P11)

| refiner | what it does | check it against |
|---|---|---|
| drop short blips | remove possession runs shorter than *m* frames, and `D` blips inside a possession | the public GT has real possessions of 0.76 s and 1.32 s — *m* must be validated, not guessed ([T04](../../docs/traps.md)) |
| explicit `D` at restarts | `D`, never `X`, before kick-offs and once the ball is out — an `X` there is forward-filled into a long fake possession | [T02](../../docs/traps.md); kick-off frames are in `ctx.run.inputs.kickoffs`, but detect the stoppage from data (P05) rather than writing labels from constants |
| free `X` | abstain inside a possession where the decoder is unsure — free on `spellF1`, a small cost on `macroF1` | needs the decoder's `posterior`; never abstain across a hand-over ([EVALUATION.md](../../EVALUATION.md)) |
| snap to touches (P11) | move a boundary to the touch or pass event nearest to it | the shortest public possessions tolerate 0.3–1.5 s of *total* boundary error (A01) |

## Adding a refiner

1. Write `pipeline/refine/<name>.py` with `refine(ctx, decoded, **params) -> Decoded`.
2. Add one line to `REGISTRY` in [`__init__.py`](__init__.py).
3. Add a `[[refine]]` entry to an experiment config, run it with `--score`, and log what it changed in
   [docs/results.md](../../docs/results.md) — including when the answer is "nothing worth keeping".

Remember the rule that makes the submission legal: the labels must come from an algorithm. A refiner applies a rule
to the whole match; it never edits particular frames by hand (RULES.md).
