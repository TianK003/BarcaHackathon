# P27: a degradation that measures nothing of our match

Written on 9 Nov 2026, late evening, before any run on the new profile. Tian decided the redo that evening
([rules.md](../../rules.md), "The organisers' reply of 5 Nov 2026"); Tian is away overnight, so the decisions Tian
would make here went to an advisor agent and are listed in the morning summary. Everything in "The protocol" is
declared before its run and is not changed after a result is seen.

## Why

The organisers' reply of 5 Nov: "anything that is not explicitly allowed is not permitted". The benchmark's `gaps`,
`tracking` and `context` profiles read a `Calibration` measured on our match (its missing-ball masks, tracklet lengths,
wrong-team share, NaN-speed share, low-count frames, a hazard model). Both models were trained on DFL matches degraded
that way, and every constant of `main.toml` was chosen on them. Nothing allows that explicitly, and "If you need to
train (or retrain), you MUST use only external data".

## What may shape the new profile

Three sources, and nothing else:

1. **A fact the organisers publish in the challenge documents.** DATA.md describes the tracking in numbers. Those
   numbers are part of the task statement that every team reads. DATA.md also tells teams to use them: "Any approach
   assuming a ball position exists on every frame will meet this on day two, so design for it on day one."
2. **A rule declared here, before any run,** with its reason, where the documents give no number.
3. **The file format** that the pipeline has to parse (46 slots in two team pools, NaN for a missing observation).

What may not shape it: anything we measured on our match (A01 to A03, A07 to A09, the `Calibration`), and any number
picked because it lands near one of those. The second part cannot be proved from the inside, since we have seen those
measurements. So each declared rule below is a round range with a stated reason, and none is tuned.

**No Calibration is loaded for this profile, by code:** `degrade.needs_calibration("generic")` is False, the runner
and the trainers refuse a Calibration for it, and a selftest proves that `calibration.load` is not called.

## The profile: `generic`, version 1

| part | setting | source |
|---|---|---|
| ball missing, share of the frames | 0.41 | DATA.md: "The ball is missing 43% of all frames, and 41% of the graded ones" |
| length of a gap | log-uniform between 1 s and 60 s | declared. DATA.md names the causes ("occlusions, crowded boxes and balls in the air") and no length. A ball in the air is gone for about a second; a crowded box or a stoppage for up to a minute. Log-uniform makes every scale between them equally likely |
| where the gaps fall | a Poisson process, blind to the play | declared: the simplest process with that share. DATA.md says the ball "goes missing disproportionately on the frames that matter", which this does not model (a limitation, below) |
| ball speed | NaN on every frame | declared: the profile promises no ball speed, so no component can lean on one (the pipeline derives it from positions) |
| slots | 46: 24 home, 22 away | the file format: DATA.md gives 46, and the split is the layout of the metadata the pipeline parses |
| players lost | each on-pitch player is lost on 1 − 20.5 / 22 = 6.8% of the frames | DATA.md: "Expect around 22 tracked objects during play — the mean over graded frames is 20.5" |
| length of a tracked span | log-uniform between 2 s and 300 s | declared. DATA.md: slots "appear, vanish and swap identity as the tracker loses and re-acquires people", with no length. No component reads a slot's identity, so this sets only how often a player drops out |
| length of a lost span | geometric, mean = mean tracked span × d / (1 − d) (4.4 s) | follows from the two rows above |
| smoothing | a centred 7-frame moving average over each tracked piece | declared, from the literature: the 7-frame window of the velocity filter that is the usual first step on 25 Hz tracking (L. Shaw, Friends of Tracking, `Metrica_Velocities`) |
| wrong-team slots, NaN-speed runs, frames with few players, a shortest tracklet | none | DATA.md documents none of them |

**What the declared gap law gives:** a median gap of 7.7 s, a mean of 14.4 s and a 90th percentile of 40 s, by
arithmetic. A check made after the law was written, on external data only: in SkillCorner's open data (MIT licence,
automatic broadcast tracking, three A-League matches read) the ball is undetected on 39% to 46% of the frames, in gaps
with a median of 5.4 to 5.8 s, a mean of 11.5 to 14.0 s and a 90th percentile of 27 to 36 s. So the declared law looks
like a real automatic tracker's gaps on other games. No number was taken from that check.

**No fidelity gate against our match.** `fidelity.py` measures a profile against the `Calibration`; this profile is
not built to pass it and is never run through it.

**Limitations, stated now.**
- The gaps ignore the play. On our match they do not (DATA.md), so the benchmark still cannot value features about
  where the ball is lost, and it still overstates the dead-ball model (T38, T39).
- No wrong-team slots and no long NaN-speed runs. Our tracking has both (T31, T32). The old benchmark put the whole
  player degradation at 0.013 spellF1 (A14), so the cost is expected to be small, and it is not measured on our match.
- The gap law is one declared range. A second range would give other constants.

## The protocol

All runs: `--profile generic`, seeds 0 to 2, the seven DFL matches, both truths. The objective is R024's, which Tian
kept on 9 Nov: the unweighted mean of the pooled raw and coarse spellF1s.

**The choice rule, every stage** (R024's, through `choose.py --objective mean`): the best mean; within a tie band of
0.005 of it, the larger min(raw, coarse); then the stage's own tie order. Nothing is added to a grid after its
results.

**Stage 0. Retrain both models** on `generic` with the settings they had (seeds 0 to 2, every fifth graded frame, the
trees' settings unchanged), then `--verify` each. Each benchmark match is read by the model that never saw it, as
before.
- *The shape model's stop rule (R011's):* pooled held-out log-loss on ball-missing frames below 0.6831. If it fails,
  the shape source leaves the config and Stage C's weight dimension is dropped.

**Stage 1. The touch rule's confidence,** measured and not tuned: `touch_stats.py --grid r013 --profile generic
--seeds 3`, the floor of each setting's pooled agreement to a hundredth (48 settings).

**Stage A. The touch rule** (R013's Stage A grid, 192 configs): radius {1.5, 2.0, 2.5, 3.0} m × relative speed {6, 9,
12} m/s × `min_frames` {1, 2, 3, 5} × hold {0.52, 0.55, 0.60, 0.70}, each with its Stage 1 confidence. The rest is the
incumbent: the dead-ball model at weight 1.0, the shape model at 0.2 on every graded frame, the hmm 4 / 32 / 35 / 25,
no shift. Tie order: larger `min_frames`, smaller radius, smaller speed, smaller hold.

**Stage C. The decoder and the shape weight** (R013's Stage C grid, 144 configs) at Stage A's choice: `switch_team`
{1, 2, 3, 4, 6, 8} × `min_possession` {1, 10, 15, 25, 35, 50} × shape weight {0.1, 0.2, 0.3, 0.5}. Tie order: larger
`switch_team`, larger `min_possession`, smaller weight. A choice on the edge of the grid is reported as such and is
not extended tonight.

**Stage D. The dimensions that R009, R010 and R011 chose and R013 left alone,** at Stage C's choice, one group at a
time (12 configs): `switch_dead` {16, 32, 64} × `min_dead` {1, 25}; the dead-ball weight {0.1, 0.25, 0.5, 1.0}; the
shape scope {gaps, all}; and the config without the shape source. *Rule:* the incumbent value stays unless another
beats it by more than the tie band (0.005 on the mean). The shape source stays if the config with it is not worse
than the config without it. If more than one group changes, the combined config is run as well: it is kept only if
its mean is not below the best single change's; otherwise only the best single change is kept. The rules of Stages A,
C and D are applied by `pipeline/benchmark/p27_choose.py`, and the configs are written by `p27_grid.py`, so the whole
re-choice can be re-run from the repository.

**Stage E. The shift** (R012's rule): `conventions.py latency` of the final pre-shift config on `generic`, raw truth;
the shift of a type of edge is minus the rounded median lag, and 0 below 30 matched changes. *Guard:* the shifted
config is kept only if its mean is above the unshifted one's and neither truth falls by more than 0.005.

**The bar for the result** (R010's): the final config beats `nearest_w50` under the raw truth and `nearest_w125`
under the coarse truth, each with a sign-flip p of at most 0.05 over the seven matches.

**Checks on our match, pass or fail only** (allowed: "Running automatic checks using only the provided data"; they
choose nothing, T01):
- the file validates, has no `X` inside a graded block, and two runs give the same sha256;
- predicted `D` share inside [30%, 62%] (R010's band, set on DFL);
- the shape model's transfer gate (R011's rule): on our match, `touch_agreement` of the model fitted on all seven
  matches is at least the lowest held-out value among the seven DFL matches in the new `--report`. If it fails, the
  shape source leaves the config, and nothing else changes;
- the public window is scored once as a correctness check, and its number chooses nothing.

A failed check stops the upload and goes to Tian. It does not send us back to change the profile.

**What is not re-chosen:** which components exist (the touch rule, the two tree models, the hmm, the shift), the
trees' settings ("fixed beforehand, not tuned", R009, R011), `min_prob` 0.02 and the hmm's structure. Each was
adopted on the old benchmark by a bar. Stage D's ablation covers the shape source, and the final bar covers the
whole.

**If the night runs short:** the stages are run in this order and none is skipped. A config lands in `main.toml`
only after Stage C. If Stage C is not done by 06:30, `main.toml` stays as it is and the morning summary says how far
the runs got.

## What happens to the old profiles and reports

`gaps`, `tracking` and `context` stay in the code for the record of R002 to R029, marked as reading our match
(`degrade.NEEDS_CALIBRATION`). Nothing is trained or chosen on them again. The benchmark's default profile is
`generic`.
