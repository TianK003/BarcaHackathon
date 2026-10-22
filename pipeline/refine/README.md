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

Nothing, on purpose. `REGISTRY` is empty because every rule that belongs here trades one error for another, and the
right setting has to come from validation (P01), not from a guess. With no `[[refine]]` entries the decoder's labels
go straight to the output.

## First candidates (P12, then P11)

| refiner | what it does | check it against |
|---|---|---|
| drop short blips | remove possession runs shorter than *m* frames, and `D` blips inside a possession | the public GT has real possessions of 0.76 s and 1.32 s — *m* must be validated, not guessed ([T04](../../docs/traps.md)) |
| shift transitions | move every change by the measured mean latency | `lat_f` on the scoreboard and `evaluation/reports/<name>_transitions.csv` ([T03](../../docs/traps.md)) |
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
