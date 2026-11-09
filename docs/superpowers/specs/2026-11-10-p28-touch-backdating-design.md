# P28: back-dating a possession to the team's first touch

Written on 9 Nov 2026, 23:50, before any run. Tian is away overnight; the lane, its gate and its bar are the rulings
of an advisor agent standing in for Tian (the run's ledger, D1 to D3), made after a research agent ranked what has not
been tried. It runs after P27 has landed, on the `generic` profile and P27's constants. Nothing below is changed
after a result is seen.

## Belief

`controlled_touch` speaks only where the ball moves with the nearest player (at most 6 m/s apart). A first-time
pass, a header or a deflection that the same team then controls is not an episode. So when a team wins the ball with
a touch and controls it a moment later, the decoder's change of team lands on the control, one pass flight late. The
truth's flag flips at the touch: the public truth counts a won header as a possession (issue #17), and DFL's flag
flips with the new team 0.64 m from the ball (A15).

**What speaks against it** (the advisor's check of R008 and R013): when a new-team episode is found near a flip, it
starts 0 to 5 frames *before* the flag, so the found half is not late. The unfound half is about as large as the
share of flips with no ball at all (A15). The belief holds only if a real share of the *ball-visible* flips have a
touch and no episode. Step 0 measures exactly that.

**What would falsify it:** step 0's gate failing, or the bar failing.

## The touch

A touch at frame t, by the team of the nearest present slot:
- the ball is visible on every frame of [t − 7, t + 7];
- the existing spatial test holds at t, with the config's own `radius` and `duel_radius`: the nearest slot is within
  `radius` of the ball and the other team's nearest slot is farther than `duel_radius`;
- the ball's velocity changes: |v(t + 4) − v(t − 4)| ≥ τ, with `kinematics`' velocities. Frames that pass form runs,
  and the touch is dated at the frame of its run's largest change (the earliest on a tie).

τ is the only grid dimension: {3, 6} m/s. The zone test is Vidal-Codina et al. (2022)'s; a change of the ball's
velocity inside a player's zone is their sign of a touch. `kinematics.py` is not changed.

## Step 0: the gate (a diagnostic, no pipeline change)

On `generic`, raw truth, the seven matches × three seeds pooled, with P27's touch rule, for each τ:

- **(i)** among the truth's team flips where the ball is visible at the flip, the share that have a touch of the new
  team within ±12 frames *and* no kept episode of the new team starting within ±12 frames. *Needed: at least 10%.*
- **(ii)** over every frame the rule below would back-date, the share where the truth names the touching team.
  *Needed: at least 70%.* A frame where the truth is `D` counts against.

Both are needed, for at least one τ. The gate replaces the research agent's draft ("p75 of the lag at least +8 frames
or a match rate below 70%"), which could not fail: the match rate is 43% already. Step 0 also reports (ii) with the
stretch capped at 12, 25 and 50 frames, for the one retry.

If the gate fails for both τ, P28 stops here: nothing is built into the pipeline and the entry is logged as a
negative result.

## The rule (built only to be tested after the gate; off by default)

`controlled_touch` gains `backdate_tau` (m/s; absent is off, and the output is then byte-identical to today's),
`backdate_confidence` and `backdate_cap` (frames; absent is no cap).

A kept episode of team T that follows a kept episode of the other team O (in the same graded block) gets T's vote
from the earliest touch of T after O's episode ended, if the ball is visible on every frame from that touch to T's
episode and no touch of O lies between. The frames from the touch to the frame before T's episode say T with
`backdate_confidence`.

- A touch never counts towards `min_frames` and never makes an episode. "The hold follows any touch" is not built:
  CHALLENGE.md lists a touch without control as no change of possession.
- `backdate_confidence` is measured, not tuned: the floor to a hundredth of step 0's (ii), clipped to
  [`hold`, `confidence`].
- `alive` stays NaN.
- Not touched: `kinematics.py`, the episode rule, P27's constants, the hmm, both models.

## The test and the bar

Two configs, τ = 3 and τ = 6 (those that passed the gate), on top of P27's final pre-shift config, against that
config; `choose.py --objective mean` picks one. The shift is then re-derived by P27's Stage E rule.

**The bar (R014's form, on the mean of the two truths):** pooled Δ ≥ +0.005 with a sign-flip p ≤ 0.05 over the seven
matches; neither truth's spellF1 nor the alive/dead F1 more than 0.005 below the reference; the leave-one-match-out
mean above 0. The advisor puts a pass at about one in ten. A guard of the form "not shown to hurt" is not enough
here: a kink rule chosen on DFL's exact ball carries a transfer risk to our tracker's ball.

**One check on our match, pass or fail, made once** (allowed as an automatic check; it chooses nothing): for the one
chosen config, the number of back-dated stretches per graded minute with the ball visible lies between half the
lowest and twice the highest of the 21 benchmark runs. The band is written into the results entry before the match
is run. A fail drops the candidate with no retry, since a retry shaped by that number would be tuning.

**One retry,** only if the first version's mean Δ is above 0 and the bar is missed: `backdate_cap` set to the longest
of 12, 25 or 50 frames whose stretches still agree 70% in step 0's table. Otherwise P28 is dropped and its code
removed.

**Time:** no benchmark run starts after 06:30 on 10 Nov.
