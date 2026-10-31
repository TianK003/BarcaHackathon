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
| `ball_presence` | `seen` with a ball; `missing` on a missing frame more than `margin` frames from one; NaN nearer | NaN everywhere | `margin` ≥ 0 frames, default 0; `missing` ∈ [0, 1], default 0.2; `seen` ∈ [0, 1], default 0.585 | the baseline's "no ball means stoppage" as evidence, for P06's decoder to weigh: exactly that only at `baseline_raw`'s half-width 0, and broader than the baseline's vote at `margin` 0 (below). P05 replaced it |
| `controlled_touch` | NaN everywhere | `confidence` / `1 − confidence` on a kept control episode; `hold` / `1 − hold` between episodes if `hold` > 0.5; NaN elsewhere and without a ball | `radius`, `max_rel_speed` (may be `inf`), `min_frames`, `confidence` ∈ (0.5, 1], `duel_radius` default 1.0, `hold` ∈ [0.5, confidence], default 0.5 = off | P04: a player within `radius` of the ball, no opponent within `duel_radius`, moving with it, for at least `min_frames`. Silent in flight and in a duel unless `hold` > 0.5, which gives a weak vote for the last controlling team there |
| `dead_ball_rules` | the smallest of `alive_out`, `alive_still`, `alive_slow`, `alive_kickoff` among the rules that fire, else `alive_none`; 0.001 before a half's kick-off; NaN outside the graded blocks | NaN everywhere | the five values, each ∈ [0, 1], required, no default | P05's rules on `play_features`: ball beyond a line, still at a restart spot, a collective slowdown, a kick-off formation. Thresholds fixed (laws of the game, literature), values measured (R009) |
| `dead_ball_model` | the model's P(alive) from the 30 `play_features`, prior-corrected (logit p − logit prior_alive) and clipped to [`min_prob`, 1 − `min_prob`]; 0.001 before a half's kick-off; NaN outside the graded blocks | NaN everywhere | `model` (a path from the repo root, required), `min_prob` ∈ (0, 0.5), default 0.001 | P05's learned opinion, trained by `pipeline/benchmark/train_dead_ball.py` on the seven DFL matches degraded like ours. The model of a match is the one that never saw it (`folds[inputs.game]`); BAR-ATH and any other game use `full`. numpy only; the first call on 140,000 graded frames with 300 trees takes about 2 s with the features already computed, and its raw scores are remembered per `Inputs`, so configs that share a model walk the trees once |
| `shape_possession` | NaN everywhere | the model's P(home has the ball) from the 26 `shape_features`, as logit = (r(x) − r(swap(x))) / 2, which is exactly antisymmetric in the teams and carries no prior, clipped to [`min_prob`, 1 − `min_prob`]; NaN outside the graded blocks, where neither team has a present slot and, with `scope = "gaps"`, wherever the ball is present | `model` (a path from the repo root, required), `scope` "gaps" (default) or "all", `min_prob` ∈ (0, 0.5), default 0.02 | P07's opinion for the 41% of frames without a ball, from the players alone; `main.toml` uses it at weight 0.2 with scope "all" (R011). Trained by `pipeline/benchmark/train_shape.py` on the seven DFL matches with the ball hidden; the model of a match is the one that never saw it (`folds[inputs.game]`), BAR-ATH and any other game use `full`. numpy only; the first call on 140,000 graded frames with 300 trees (two walks) takes 2 to 3.5 s with the features already computed, and its logits are remembered per `Inputs` |
| `restart_team` | NaN everywhere | per called stoppage whose type has a confidence c: c (home restarts) or 1 − c, with `span` "after" on [restart, min(restart + `tail`, the next stoppage, the block's end)), with `span` "stoppage_end" over the stoppage that ends at the restart, [first, restart) of n frames, sigmoid(logit(c)/n) a frame (1 − that for away): one observation in all; NaN everywhere else | `model` (the dead-ball model's file, required), `span` "after" (default) or "stoppage_end", `tail` frames ≥ 1 (required with "after", refused with "stoppage_end"), `use_ball` default false, `setup` default 50 frames, `min_stoppage` default 25 frames, `min_prob` default 0.001, and a confidence ∈ (0.5, 1] for each type that votes, at least one: `kick_off`, `corner`, `goal_kick`, `throw_in`, `free_kick` (only with `use_ball`) | P17: who restarts after a stoppage, from the players' arrangement in the stoppage's last `setup` frames before the dead-ball model's restart and the Laws of the Game. With "after" it votes only from the restart on, so where `D` ends stays the dead-ball model's call; "stoppage_end" is for the decoder's restart states (`hmm`'s `restart_source`). The confidences are measured on the benchmark (`restarts.py stats`), never tuned |
| `smart_tagging` | NaN everywhere (team votes only; the dead votes were measured in R017 and dropped) | `c_team`: `c_team` / 1 − `c_team` where the tags name one team (the team of a possession-phase tag, the other team for a defending one), NaN where they disagree or say nothing | `c_team` ∈ (0.5, 1.0], required; `spread` "frame" (default) or "tag" (one observation per tag, its log-odds spread over the tag's frames) | P08 (R017): in `main.toml` (c_team 0.76, `frame`, weight 0.2, Tian's pick). Smart Tagging's automatic phases as team votes, cut at the graded blocks. `MATCH_ONLY`: only our match has them, so the benchmark drops it with `--drop-match-only`. The confidences are declared by the measurement against our pseudo-labels, never tuned |

The nearest player is not the possessor — wrong on 22% of ball-visible, in-play frames ([T14](../../docs/traps.md)).
P04 is its replacement.

**`confidence`** is the number a probabilistic decoder is given in place of the hard 0/1. The default 1.0 keeps the
baseline's claim of certainty, which is what the four baselines are rebuilt through. P06's configs use 0.78, the
public window's share of in-play, ball-visible frames on which the nearest player's team is the right one (A01; DFL's
matches give 71–82%). It is a measurement fixed once, never tuned; it moves no frame's side, so a majority vote reads it
the same as the hard values bar float rounding in an exact tie.

**Constants from the public window (since 23 Oct 2026, `f8f2629`).** `confidence` 0.78 above and `ball_presence`'s
`missing` 0.2 and `seen` 0.585 below were derived from the public window's statistics (A01). The organisers' rule says
that window is for checking the pipeline and is "not training data and not a tuning set", so those values are not to be
used in a submission config unless re-derived from other games (DFL's matches give 71 to 82% for the nearest player,
A06). Neither is in `main.toml`; they stay in P06's research configs (R006 to R010, `p06_hmm.toml`).

**`ball_presence`** gives the decoder its only evidence for `D`. Its two values come from A01 by rule: the ball is
missing on 33.6% of dead-ball frames and 6.5% of in-play ones in the public window, a ratio of 5.2 that is rounded down
to 4 because that window is the best-covered block of the match (`missing = 0.2`); and it is seen on 66.4% against
93.5%, a ratio of 1.41 for alive (`seen = 0.585`). `seen = 0.5` adds nothing to the decoder's emissions and is the
control "without it". A missing frame moves `D` over `H` and `A` by log(0.8 / 0.2) = 1.39 nats, four times what `seen`
gives alive (0.34) and more than `nearest_player` at 0.78 (1.27), so across a long gap it outweighs any `switch_dead` in
P06's grid: a gap longer than about 2 * `switch_dead` / 1.39 frames decodes to `D`, its edges included. At `margin` 0
that is broader than the baseline's rule, not weaker: the baseline's vote, `nearest_w50` (half-width 50), gives `D` only
more than 50 frames inside a gap, so a gap of 47 to 100 frames (at `switch_dead` 32), and the first and last 50 frames
of a longer one, are `D` here and keep their team there, and only a shorter gap (and every visible stretch) can be
overruled. R007's `margin` 50 control measures the difference: the predicted `D` share falls from 43.1% to 38.3% and raw
spellF1 rises by +0.0071.
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
half is decided per block. The features also feed a classifier (`dead_ball_model`, below). About 1.5 s on 140,000 graded
frames of 46 slots with the kinematics tracks already built.

[`dead_ball_rules.py`](dead_ball_rules.py) reads four rules off those features: `out` (ball more than 0.5 m beyond a
line), `still` (2 s motionless at a restart spot, on a line or in a goal area), `slow` (80% of the slots under 2 m/s
over +/- 4 s) and `kickoff` (both teams in their own half, a slot within 2 m of the centre spot). The thresholds come
from the laws of the game and the literature and are never tuned; the five probabilities are measurements, taken from
`train_dead_ball.py --rules` (R009) and passed as required parameters. `pre_kickoff_mask` and `PRE_KICKOFF_ALIVE`
(0.001) make a graded frame before its half's kick-off dead, only where the kick-off lies inside the block (T02).

[`dead_ball_model.py`](dead_ball_model.py) is the learned counterpart of the rules. `python
pipeline/benchmark/train_dead_ball.py --model hgb` fits a gradient-boosted classifier (fixed settings, never tuned)
or, with `--model logistic`, the reference, on the benchmark's degraded tracking and writes
`pipeline/models/dead_ball_<kind>.json`: a model per held-out match and one on all seven, as plain numbers, no pickle.
The source reads the file with numpy alone (scikit-learn is imported only under `pipeline/benchmark/` and pinned in
`requirements-benchmark.txt`), refuses a file whose `format`, `feature_version` or `features` are not `play_features`'
and names the trainer command, picks `folds[inputs.game]` or else `full`, and gives the decoder a likelihood:
`logit(alive) = logit(p) - logit(prior_alive)`, since the decoder sets the class prior itself and DFL's balance must
not leak into it. `--verify` retrains and compares with the committed file (only the three source hashes of the
provenance may differ), and `--rules` measures the five values `dead_ball_rules` needs (see
`pipeline/benchmark/README.md`, "Dead-ball model").

The walk of the trees, and the reading, checking and remembering around it, are [`trees.py`](trees.py)'s: numpy only,
shared by `dead_ball_model` and `shape_possession`, with each source's messages starting with its own name.

## P07: team shape and the shape source

[`shape_features.py`](shape_features.py) (numpy only) turns the players alone into 26 per-frame numbers about the two
teams: how many slots are present, how far up the pitch the team stands (the mean, and the 10th and 90th percentiles),
its width, its velocity along its attacking direction, the shares of it in the attacking and the defending third, and
how far it moved across the one and the two seconds either side of the frame; twelve for home (`h_*`), twelve for
away (`a_*`), then the gap in advance and in numbers. No column reads the ball, which is the point: a model can be
trained on matches whose ball has been hidden and be asked about the frames of ours where it is missing. Each team is
seen in its own frame (its attacking direction, decided per block, T16), so nothing in the features says which team is
Barcelona, and `swap(X)` exchanges the two teams' columns exactly.

[`shape_possession.py`](shape_possession.py) runs a classifier on them, with `dead_ball_model`'s file layout (`folds`,
`full`, plain JSON, no pickle, refused on a `format`, `feature_version` or `features` that are not shape_features' and
naming the trainer, `pipeline/benchmark/train_shape.py`). What it says is a probability that home has the ball, from
the trees' raw score r(x) on the frame's features and on the same frame with the teams relabelled:

    logit(home) = (r(x) − r(swap(x))) / 2

*Why symmetrised.* The two teams are interchangeable in what the features mean, so the model has no business knowing
which one is home, and a fit on the training rows and their mirror images is only nearly symmetric. This makes it
exactly so: relabelling the teams turns `home` into `1 − home` to rounding, and whatever the trees learned that favours
"home" as such (the baseline, a lean towards the first team) cancels. For the same reason there is no prior correction:
the rows and their mirrors have exactly as many homes as aways (`prior_home` is 0.5 in the file, or the file is
refused), so the probability is already a likelihood and the decoder sets the class prior itself. `alive` is NaN
everywhere: the shape of two teams is no evidence on whether the ball is in play.

*Where it speaks.* Only on graded frames where at least one team has a present slot. With `scope = "gaps"`, the
default, only where the ball is missing, because that is where nothing else says who has it, and on the premise that
ball-based evidence is better the rest of the time; `scope = "all"` also speaks where the ball is seen, and R011 found
that premise wrong on `tracking`: `all` is better than `gaps` at every weight, and `main.toml` uses it. A team that is
present while the other is gone still gives an opinion (the absent team's columns are NaN, which the trees were fitted
to handle). `home` is clipped to [`min_prob`, 1 − `min_prob`], 0.02 by default, since the decoder reads a probability
as log(p / (1 − p)) nats a frame.
The trainer and the committed model are separate steps of P07: a config that names a model file that is missing, or
one that is not shape_features', fails on load with the trainer's command.

## P17: who restarts after a stoppage

[`restart_team.py`](restart_team.py) (numpy only) names the team that restarts play after each predicted stoppage and
votes for it on the first frames of play. The decoder otherwise carries the team it had across the stoppage until the
first control episode names one, and seven stoppages in ten hand the ball to the other team (A12). A perfect 2 s vote
is worth +0.0198 pooled raw spellF1 to `main.toml` (A12's ceiling); this is the label-free estimate of that vote.

*Where the stoppages are.* `dead_ball_model`'s P(alive), with its pre-kick-off rule, from the `model` and `min_prob`
given. With the config's own dead-ball source's (`main.toml`'s file and clip) the two agree on where the ball is dead,
and the model's scores are remembered per `Inputs`, so the trees are walked once. A stoppage is a run of
P(alive) < 0.5 of at least `min_stoppage` frames; its restart is the frame after it. A run that reaches the block's
end, or is ended by a NaN, has no restart.

*Who restarts.* On the stoppage's last `setup` frames, [max(first, restart − `setup`), restart), each frame calls at
most one (type, team), by the first of these rules that fires; the stoppage takes the pair called on the most frames,
if that is at least a quarter of the window. The window never reaches back into play: a stoppage shorter than `setup`
is read whole and no further, since the live frames before it hold arrangements of their own (a winger at the
touchline makes a throw-in). The opponents' distances are half the law's, because players encroach; the zones allow
for tracking error. All are declared, none tuned. This is Vidal-Codina et al. (2022)'s approach: set pieces read
deterministically from the players' configuration during the dead ball.

| type | fires when | team | law |
|---|---|---|---|
| kick-off | every present slot within 1 m of its own half, both teams present, exactly one team with a slot in the centre circle (9.15 m) | the team in the circle | 8 |
| corner | a slot within 3 m of a corner point, no opponent within 4.5 m of him | his | 17 |
| goal kick | at an end, a slot of the team defending it inside its goal area (5.5 × 18.32 m), at most 1 opponent inside that penalty area (16.5 × 40.32 m) | the defending team | 16 |
| throw-in | a slot within 1.5 m of a touchline or beyond it, no opponent within 1 m of him | his | 15 |
| free kick (`use_ball` only) | the ball visible and still (< 1 m/s), its nearest slot within 2 m, no opponent of his within 4.5 m of the ball | his | 13 |

Each team's defended end is decided per block (`play_features.own_half_is_left`, now public; the old private name is
an alias). Several slots meeting one rule: the nearest its line or spot wins (touchline, corner point, own goal line,
centre spot), then the lowest slot. Every rule needs both teams on the frame (an opponent clearance with nobody there is
vacuous), and slots are read with the team the metadata gives them, however many a team shows (T15, T32).

*The ball* (`use_ball`) says where, never what. On a frame of the window where it is visible, each rule runs only at
its place nearest the ball (kick-off: ball within 2 m of the centre spot; corner: within 3 m of the nearest corner
point; goal kick: in that end's goal area grown by 1 m; throw-in: within 1.5 m of a touchline or beyond), with the slot
nearest the ball, within 2 m, as its player. The free kick is the one type with no players-only rule. Frames without
the ball use the players-only rules. DFL tracks the ball out of play and we do not (T39), so `use_ball` is a grid axis
of R014, not a default.

*The vote* (`span = "after"`, the default). `home` = c (home restarts) or 1 − c on [restart, min(restart + `tail`, the
next stoppage's first frame, the block's end)), for a type with a confidence; NaN elsewhere, and `alive` NaN everywhere.
It starts at the restart, never on the stoppage: a `home` vote raises the alive states against `D` by up to w·log(2c)
nats a frame, so a vote of that strength on the stoppage would move the dead-ball edges (the stoppage span below spreads
one observation so thin that it does not). It does not stop at the first control episode: at `controlled_touch`'s 0.93
the episode out-votes it anyway, and stopping there would couple the two sources.

*The vote over the stoppage* (`span = "stoppage_end"`: the stoppage that ends at the restart; P17's Phase 2b). The vote
after the restart gained +0.004 spellF1 (R014, short of its bar): a span of fixed length keeps pushing after the
restarting team's first possession has ended, and it starts at this source's restart, not at the decoder's `D` edge.
The decoder's restart states (`hmm`'s `restart_source`, [decode README](../decode/README.md#restart-states-p17)) carry
the restart team themselves: they read this source's `home` on the dead states only, `D_H` ("dead, home restarts")
against `D_A`, and hand it to whichever frame the decoder ends `D` on. For them the source writes one observation of who
restarts per called stoppage, at the type's confidence c, spread evenly over the stoppage's own frames [first,
restart): each of its n frames gets sigmoid(logit(c) / n), 1 − that for an away taker, and the decoder, which adds
logit(h) a frame between `D_H` and `D_A`, sums them back to logit(c).

- *One observation, not n:* c on every frame of a 20 s stoppage would count 500 times, and c would stop being a
  likelihood.
- *Spread, not on one frame:* the stoppage's last frame, restart − 1, is its most marginal (P(alive) just under 0.5). A
  decoder that ends `D` one frame earlier puts it in `H` or `A`, where the vote adds nothing, and the restart team is
  lost. Spread, a `D` run that covers k of the stoppage's n frames keeps k / n of the observation.
- *Thin:* about logit(c) / 2n nats a frame for the better dead state against `H` and `A` (0.011 at c = 0.9 over 4 s),
  so where `D` ends stays the dead-ball evidence's call.

`tail` is not read then, and a config that gives one is refused (one with `span = "after"` and no `tail` is refused
too). Read by a decoder without restart states, the frames would be ordinary `home` votes on dead frames, so
`"stoppage_end"` belongs with `restart_source`.

`frame_calls`, `stoppages`, `arrangement`, `calls_from_alive`, `calls` and `votes` are public: the checks drive them
directly (`test_restart_team_*`, including a plain-loop reference of the rules), and `restarts.py stats` reads `calls`
to measure each type's agreement on the benchmark. About 2 ms a stoppage, with the tracks and the model's scores
already built.

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
5. If the source needs an asset only our match has (the Smart Tagging, the whistle), set `MATCH_ONLY = True` at the top
   level of its module: `evidence.match_only(kind)` reads it, and the benchmark's `--drop-match-only` then runs a config
   without the source, naming it in the report, since a DFL match cannot serve it.

## Rules of thumb

- **Probabilities, not labels.** Calibrate them (on other games, never on the public window): a source that is right
  78% of the time should say about 0.78, not 1.0. A probabilistic decoder clips to [ε, 1 − ε] (see the decode README),
  so a hard 0 or 1 is survivable, but it shouts down every other source.
- **NaN is not 0.5.** NaN means "I can't see this frame"; 0.5 means "I looked and it's a coin flip". Both are
  legitimate, but they are different claims.
- **`weight`** (config) is how much the decoder trusts a source. For a probabilistic decoder (P06) it is meant as an
  exponent on the source's likelihood; the stand-in majority vote only approximates that with a weighted mean.
- **Look anywhere in time.** A source sees the whole match, and we process offline, so looking ahead is fine — the
  camera lags the ball by about 35 frames (T18), and P04 back-dates a change to where control began
  ([T03](../../docs/traps.md)). Only decoders and refiners are confined to one half.
- **Metres before distances** (`inputs.metres`), **direction per half** (`inputs.half_of_frame`; teams swap ends —
  [T16](../../docs/traps.md)), and **slots are tracklets, not players** ([T15](../../docs/traps.md)).
- **Randomness from the seed:** `np.random.default_rng([run.seed, 1234])`, with a constant of your own, so adding a
  source never shifts another's random stream ([T22](../../docs/traps.md)).
- **No ground truth as input**, and nothing about this fixture from outside the bundle
  ([T26](../../docs/traps.md)). No hand labels on this match, and no hand-corrected tracking; train on other matches
  (P01, P13). The public window is for checking that the pipeline is correct: nothing is trained, fitted or chosen on
  it, out of fold or not (RULES.md, `f8f2629`, [T27](../../docs/traps.md)).
- **Never print rows.** Aggregates only, in logs, notebooks committed to git, and docs.

## Planned extensions (build them when the first source needs them)

- **An evidence cache** for slow sources (video features, say): keyed by
  `(kind, canonical JSON of the parameters, ATD sha256, the source file's hash)`, stored under
  `data/cache/evidence/`. Until then every run recomputes every source: 14.9 s for `main.toml`'s three, in the run of
  25 Oct that wrote R011's file, with the cache warm.
- **A `next_home` channel** — P(home restarts | the ball is dead) — for P05's restart types and P08's set-piece tags.
  It will be an optional field with a default, so existing sources keep working unchanged. P17 does without it:
  `restart_team` votes in `home`, from the restart on or (`span = "stoppage_end"`) over the stoppage, where `hmm`'s
  `restart_source` reads that one source's `home` as who restarts, so the contract is unchanged.

## Proposals that land here

| proposal | channel | note |
|---|---|---|
| P04 sticky ownership from controlled touches | `home` | supersedes `nearest_player` |
| P05 dead-ball detector | `alive` (+ `next_home` later) | supersedes `ball_presence`; the ball is visible on 66% of dead frames ([T12](../../docs/traps.md)) |
| P07 team shape and movement | `home` | for the 41% of frames without a ball: `shape_possession` (done, R011) |
| P08 Smart Tagging | `home` (done, R017) | phases are not possessions ([T17](../../docs/traps.md)); the dead votes were dropped |
| P09 camera aim | — | dropped, R019 (the simulated camera failed two fidelity gates; code removed) |
| P10 referee whistle | `alive` | dropped, R018/R018b (whistles near stoppages 1.5× chance, below the declared 2×; code removed) |
| P13 / P14 / P15 | via better inputs or pre-trained models | |
| P17 restart-team channel | `home`, from each restart on, or over each stoppage for the decoder's restart states | `restart_team` (built; its confidences and adoption are R014's, the restart states' R014b's; P17 dropped, neither met its bar, so it stays in the code, off by default) |
