# Stage 2 — evidence

**The point of this stage:** separate *what the data says about each frame* from *what the label sequence should
be*. An evidence source looks at the match and gives an opinion per frame; it never decides a label. Deciding —
weighing sources, enforcing that possessions are intervals, paying for every change — is the decoder's job (stage 3).
Keeping the two apart is what lets five people work in parallel: a new source plugs in without touching the decoder,
and a better decoder improves every source at once.

## In → out

| in | out |
|---|---|
| `run.inputs` — the whole match, aligned (stage 1); `run.seed` | `Evidence(alive, home)`: two float64 arrays of length `n_frames` |
| keyword parameters from the config entry | |

```python
def compute(run, *, some_parameter=1.0):     # run: RunContext; parameters are keyword-only
    ...
    return Evidence(alive=..., home=...)     # name and weight are filled in by the runner
```

| channel | meaning | NaN means |
|---|---|---|
| `alive` | P(ball in play) on this frame | no opinion — leave it to the other sources |
| `home` | P(home team has the ball, *given* it is in play) | no opinion |

Why two channels: the ground truth is built in two steps — `D` where the ball is dead, otherwise the team that has
it — and most sources answer only one of those questions. A decoder recombines them:
`P(H) = alive·home`, `P(A) = alive·(1 − home)`, `P(D) = 1 − alive`.

The runner checks every source before any decoder sees it: both channels present, float64, exactly `n_frames` long,
every value in [0, 1] or NaN. A length of ~164k means the broadcast was indexed ([T09](../../docs/traps.md)).

## What is here now

| kind | alive | home | parameters | notes |
|---|---|---|---|---|
| `nearest_player` | NaN everywhere | 1.0 / 0.0 by the team of the nearest tracked slot; NaN without a ball or a player | none | step 1 of `starter/baseline.py`, bit for bit. Opinion on 59.3% of graded frames; agrees with the public GT on 77.8% of 5,640 frames (= A01) |

The nearest player is not the possessor — wrong on 22% of ball-visible, in-play frames ([T14](../../docs/traps.md)) —
and its hard 0/1 claims a certainty it does not have. P04 is its replacement.

## Writing a source

1. Create `pipeline/evidence/<your_source>.py`:

   ```python
   """One paragraph: what the source measures, why it should work, what it cannot see."""

   import numpy as np

   from pipeline.contracts import Evidence, no_opinion


   def compute(run, *, radius_m=2.0):
       inputs = run.inputs
       atd = inputs.atd                                  # read-only arrays, see pipeline/inputs/README.md
       home = no_opinion(inputs.n_frames)                # start from "no opinion" everywhere
       ...                                               # fill in the frames you have something to say about
       return Evidence(alive=no_opinion(inputs.n_frames), home=home)
   ```

2. Register it — one line in `REGISTRY` in [`__init__.py`](__init__.py): `"your_source": your_source.compute,`
3. Use it — copy `pipeline/configs/main.toml` to `pipeline/configs/<experiment>.toml` and add
   `[[evidence]]` with `kind = "your_source"` (plus `name`, `weight` and your parameters if needed).
4. Check it — `python pipeline/run.py pipeline/configs/<experiment>.toml --score` prints its coverage of the graded
   frames and how often it agrees with the ground truth: `home` on frames the GT calls `H`/`A`, `alive` on every
   graded frame. Then log the run in [docs/results.md](../../docs/results.md).

## Rules of thumb

- **Probabilities, not labels.** Calibrate them: a source that is right 78% of the time should say about 0.78, not
  1.0. A probabilistic decoder clips to [ε, 1 − ε] (see the decode README), so a hard 0 or 1 is survivable, but it
  shouts down every other source.
- **NaN is not 0.5.** NaN means "I can't see this frame"; 0.5 means "I looked and it's a coin flip". Both are
  legitimate, but they are different claims.
- **`weight`** (config) is how much the decoder trusts a source. For a probabilistic decoder (P06) it is meant as an
  exponent on the source's likelihood; the stand-in majority vote only approximates that with a weighted mean.
- **Look anywhere in time.** A source sees the whole match, and we process offline, so looking ahead is fine — P09's
  camera lags the ball by 35 frames, and P04 back-dates a change to where control began
  ([T03](../../docs/traps.md)). Only decoders and refiners are confined to one half.
- **Metres before distances** (`inputs.metres`), **direction per half** (`inputs.half_of_frame`; teams swap ends —
  [T16](../../docs/traps.md)), and **slots are tracklets, not players** ([T15](../../docs/traps.md)).
- **Randomness from the seed:** `np.random.default_rng([run.seed, 1234])`, with a constant of your own, so adding a
  source never shifts another's random stream ([T22](../../docs/traps.md)).
- **No ground truth as input**, and nothing about this fixture from outside the bundle
  ([T26](../../docs/traps.md)). A model trained on our hand labels must predict those windows out of fold
  ([T27](../../docs/traps.md)).
- **Never print rows.** Aggregates only, in logs, notebooks committed to git, and docs.

## Planned extensions (build them when the first source needs them)

- **An evidence cache** for slow sources (a trained P07 model, video features): keyed by
  `(kind, canonical JSON of the parameters, ATD sha256, the source file's hash)`, stored under
  `data/cache/evidence/`. Until then every run recomputes every source, which takes about a second today.
- **A `next_home` channel** — P(home restarts | the ball is dead) — for P05's restart types and P08's set-piece tags.
  It will be an optional field with a default, so existing sources keep working unchanged.

## Proposals that land here

| proposal | channel | note |
|---|---|---|
| P04 sticky ownership from controlled touches | `home` | supersedes `nearest_player` |
| P05 dead-ball detector | `alive` (+ `next_home` later) | the ball is visible on 66% of dead frames ([T12](../../docs/traps.md)) |
| P07 team shape and movement | `home` | for the 41% of frames without a ball |
| P08 Smart Tagging | both | phases are not possessions ([T17](../../docs/traps.md)) |
| P09 camera aim | — | a feature for P07, not a source on its own |
| P10 referee whistle | `alive` | audio confirmed, starts at t = 0 (A04) |
| P13 / P14 / P15 | via better inputs or pre-trained models | |
