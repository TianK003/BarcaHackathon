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
| `nearest_player` | NaN everywhere | `confidence` / `1 − confidence` by the team of the nearest tracked slot; NaN without a ball or a player | `confidence` ∈ [0.5, 1.0], default 1.0 | step 1 of `starter/baseline.py`, bit for bit at the default (hard 1.0 / 0.0). Opinion on 59.3% of graded frames; agrees with the public GT on 77.8% of 5,640 frames (= A01) |
| `ball_presence` | `seen` with a ball; `missing` on a missing frame more than `margin` frames from one; NaN nearer | NaN everywhere | `margin` ≥ 0 frames, default 0; `missing` ∈ [0, 1], default 0.2; `seen` ∈ [0, 1], default 0.585 | the baseline's "no ball means stoppage" as evidence, for P06's decoder to weigh: exactly that only at `baseline_raw`'s half-width 0, and broader than `main`'s vote at `margin` 0 (below). P05 replaces it |
| `controlled_touch` | NaN everywhere | `confidence` / `1 − confidence` on a kept control episode; `hold` / `1 − hold` between episodes if `hold` > 0.5; NaN elsewhere and without a ball | `radius`, `max_rel_speed` (may be `inf`), `min_frames`, `confidence` ∈ (0.5, 1], `duel_radius` default 1.0, `hold` ∈ [0.5, confidence], default 0.5 = off | P04: a player within `radius` of the ball, no opponent within `duel_radius`, moving with it, for at least `min_frames`. Silent in flight and in a duel, so the decoder keeps the last team there |
| `dead_ball_rules` | the smallest of `alive_out`, `alive_still`, `alive_slow`, `alive_kickoff` among the rules that fire, else `alive_none`; 0.001 before a half's kick-off; NaN outside the graded blocks | NaN everywhere | the five values, each ∈ [0, 1], required, no default | P05's rules on `play_features`: ball beyond a line, still at a restart spot, a collective slowdown, a kick-off formation. Thresholds fixed (laws of the game, literature), values measured later |

The nearest player is not the possessor — wrong on 22% of ball-visible, in-play frames ([T14](../../docs/traps.md)).
P04 is its replacement.

**`confidence`** is the number a probabilistic decoder is given in place of the hard 0/1. The default 1.0 keeps the
baseline's claim of certainty, which is what the four baselines are rebuilt through. P06's configs use 0.78, the
public window's share of in-play, ball-visible frames on which the nearest player's team is the right one (A01; DFL's
matches give 71–82%). It is a measurement fixed once, never tuned; it moves no frame's side, so a majority vote reads it
the same as the hard values bar float rounding in an exact tie.

**`ball_presence`** gives the decoder its only evidence for `D`. Its two values come from A01 by rule: the ball is
missing on 33.6% of dead-ball frames and 6.5% of in-play ones in the public window, a ratio of 5.2 that is rounded down
to 4 because that window is the best-covered block of the match (`missing = 0.2`); and it is seen on 66.4% against
93.5%, a ratio of 1.41 for alive (`seen = 0.585`). `seen = 0.5` adds nothing to the decoder's emissions and is the
control "without it". A missing frame moves `D` over `H` and `A` by log(0.8 / 0.2) = 1.39 nats, four times what `seen`
gives alive (0.34) and more than `nearest_player` at 0.78 (1.27), so across a long gap it outweighs any `switch_dead` in
P06's grid: a gap longer than about 2 * `switch_dead` / 1.39 frames decodes to `D`, its edges included. At `margin` 0
that is broader than the baseline's rule, not weaker: `main`'s vote (half-width 50) gives `D` only more than 50 frames
inside a gap, so a gap of 47 to 100 frames (at `switch_dead` 32), and the first and last 50 frames of a longer one, are
`D` here and keep their team there, and only a shorter gap (and every visible stretch) can be overruled. R007's `margin`
50 control measures the difference: the predicted `D` share falls from 43.1% to 38.3% and raw spellF1 rises by +0.0071.
And a missing ball is not a dead ball ([T12](../../docs/traps.md)): 41% of graded frames have no ball, in long gaps, and
in-play gaps of 10 s or more are normal. `margin` leaves the edges of a gap to other evidence: only missing frames more
than `margin` frames from the nearest frame with a ball speak.

**`controlled_touch`** reports who controls the ball, and only then. A frame is a candidate when the ball is visible,
the nearest slot is within `radius`, the other team's nearest is beyond `duel_radius` and the two move together
(|v_ball − v_player| ≤ `max_rel_speed`, both from positions, never the ATD speed channels). The zones follow
Vidal-Codina et al. (2022): within a couple of metres of one player and no other, at his speed, the ball is his; two
teams inside the duel radius make a contested ball, which is no evidence. A maximal run of candidate frames of one team
lasting at least `min_frames` is an episode, and the evidence starts on its *first* frame, so a change of team decoded
from it lands where control began: that is the back-dating of [T03](../../docs/traps.md), with no lag added. Between
episodes the source is NaN, and `hold` > 0.5 adds a weak vote for the last controlling team on frames with the ball
visible (never before a block's first episode, never across blocks): it makes the decoder keep the old team through a
pass in flight, so a turnover lands on the new team's first control frame; with `hold` off it lands on the first frame
after the old team's last one, the HMM's tie rule. `confidence` is measured, not tuned.

## Shared kinematics

[`kinematics.py`](kinematics.py) (numpy only) is what the sources that look at motion share, so that they agree to the
bit: `ball_track(inputs)` and `player_track(inputs)` give positions in metres and velocities and speeds in metres per
second, NaN outside the graded blocks and where the ball or slot is absent; `derivative`, `window_mean`, `runs`,
`ball_distances` and `outside_m` are the pieces. A velocity is the 7-frame Savitzky-Golay derivative inside a *run* (a
maximal stretch of valid frames in one graded block), falling back to central and one-sided differences at a run's ends
and NaN on a run of one frame, so nothing reads across a gap, a slot handover or half time. It never reads
`atd.players_speed` or `atd.ball_speed` ([T28, T31](../../docs/traps.md)). Both tracks are remembered per `Inputs`
object (a `weakref.WeakKeyDictionary`, in memory only); both together take about 1.3 s on 200,000 frames of 46 slots.

## P05: play features and the dead-ball rules

[`play_features.py`](play_features.py) (numpy only) turns the tracking into 30 per-frame numbers about the play:
the ball's state (beyond a line, still, at a restart spot, in a goal area, how long unseen and what it was doing at
each end of a gap), the players' speeds over windows, the set-piece shapes (kick-off, corner, goal area, touchline)
and the teams' shape. `features(inputs)` gives `(FEATURE_NAMES, X)` with `X` of shape (n, 30), NaN outside the graded
blocks, remembered per `Inputs` object and read-only; the definitions are fixed (`FEATURE_VERSION = 1`) and a team's
half is decided per block. The features also feed a classifier (a later task). About 1.5 s on 140,000 graded frames of
46 slots with the kinematics tracks already built.

[`dead_ball_rules.py`](dead_ball_rules.py) reads four rules off those features: `out` (ball more than 0.5 m beyond a
line), `still` (2 s motionless at a restart spot, on a line or in a goal area), `slow` (80% of the slots under 2 m/s
over +/- 4 s) and `kickoff` (both teams in their own half, a slot within 2 m of the centre spot). The thresholds come
from the laws of the game and the literature and are never tuned; the five probabilities are measurements, taken from
a later task and passed as required parameters. `pre_kickoff_mask` and `PRE_KICKOFF_ALIVE` (0.001) make a graded frame
before its half's kick-off dead, only where the kick-off lies inside the block (T02).

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
  ([T26](../../docs/traps.md)). No hand labels on this match; train on other matches (P01, P13), and a model
  trained on the public GT must predict the public window out of fold ([T27](../../docs/traps.md)).
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
| P05 dead-ball detector | `alive` (+ `next_home` later) | supersedes `ball_presence`; the ball is visible on 66% of dead frames ([T12](../../docs/traps.md)) |
| P07 team shape and movement | `home` | for the 41% of frames without a ball |
| P08 Smart Tagging | both | phases are not possessions ([T17](../../docs/traps.md)) |
| P09 camera aim | — | a feature for P07, not a source on its own |
| P10 referee whistle | `alive` | audio confirmed, starts at t = 0 (A04) |
| P13 / P14 / P15 | via better inputs or pre-trained models | |
